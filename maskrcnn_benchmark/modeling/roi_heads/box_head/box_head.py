# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import ot
import numpy as np
import torch
import scipy.optimize
from maskrcnn_benchmark.modeling.matcher import Matcher

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from torch import nn
from torch.nn import functional as F
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg


class ROIBoxHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.mean_folder_path = "/home/hl/Student/DYJ/one/NEW_NEW_mean_cov/mean"
        self.covariance_folder_path = "/home/hl/Student/DYJ/one/NEW_NEW_mean_cov/cov"

    def forward(self, features, proposals, targets=None, semantic=None, supTarget=None, oneStage=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        if oneStage:
            xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized = self.feature_extractor(features, proposals, semantic, oneStage)
            class_logits, box_regression = self.predictor(xc, xr)

            if not self.training:
                result = self.post_processor((class_logits, box_regression), proposals)
                return xc, result, {}

            labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)   # 【1024】
            regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)  # [1024, 4]
            loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], labels, regression_targets)
            return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg))
        else:
            xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized = self.feature_extractor(features, proposals, semantic, oneStage)

            if self.training:
                labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)   # 【1024】
                regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)  # [1024, 4]

                fg_index = labels != 0
                pos_embedding = xc[fg_index]
                pos_label = labels[fg_index]

                mean_custom_order = ["Scissors_vector", "Wrench_vector", "Gun_vector", "Bullet_vector",
                                     "HandCuffs_vector", "Knife_vector", "Lighter_vector"]
                covariance_custom_order = ["Scissors_matrix", "Wrench_matrix", "Gun_matrix", "Bullet_matrix",
                                           "HandCuffs_matrix", "Knife_matrix", "Lighter_matrix"]
                means = []    # [[2048],[2048]]
                covariances = []  # # [[2048,2048],[2048,2048]]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                for file_name in mean_custom_order:
                    file_path = os.path.join(self.mean_folder_path, file_name + ".pt")
                    if os.path.exists(file_path):
                        data = torch.load(file_path).to(device)
                        means.append(data)
                    else:
                        print(f"File {file_path} does not exist.")

                for file_name in covariance_custom_order:
                    file_path = os.path.join(self.covariance_folder_path, file_name + ".pt")
                    if os.path.exists(file_path):
                        data1 = torch.load(file_path).to(device)
                        covariances.append(data1)
                    else:
                        print(f"File {file_path} does not exist.")

                if len(pos_embedding) > 0:
                    uniq_c = torch.unique(pos_label)  # [1, 6]de biao qian
                    biaoqian = []
                    sample_mean = []
                    sample_covariances = []
                    NOW_means_list = []
                    NOW_cov_matrix_list = []
                    for c in uniq_c:
                        c = int(c)
                        select_index = torch.nonzero(pos_label == c, as_tuple=False).squeeze(1)
                        embedding_temp = pos_embedding[select_index]  # []
                        calibrated_mean, calibrated_cov = distribution_calibration(embedding_temp, means, covariances, k=1)  # [1024]
                        biaoqian.append(c)
                        sample_mean.append(calibrated_mean)
                        sample_covariances.append(calibrated_cov)
                        NOW_means = embedding_temp.mean(dim=0)
                        X_centered = embedding_temp - NOW_means
                        X_centered_cpu = X_centered.detach().cpu().numpy()
                        NOW_cov_matrix = np.dot(X_centered_cpu.T, X_centered_cpu) / (embedding_temp.shape[0] - 1)
                        NOW_cov_matrix = torch.tensor(NOW_cov_matrix)
                        NOW_means_list.append(NOW_means)
                        NOW_cov_matrix_list.append(NOW_cov_matrix)

                    samples = []
                    if len(sample_mean) == 0:
                        class_logits, box_regression = self.predictor(xc, xr)
                        new_labels = labels
                        new_bboxes = regression_targets
                        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], new_labels, new_bboxes)
                        return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg))
                    else:
                        sample_num = 150
                        for i in range(len(sample_mean)):
                            mean = sample_mean[i]
                            covariance_matrix = sample_covariances[i]
                            mean_now = NOW_means_list[i]
                            covariance_matrix_now = NOW_cov_matrix_list[i]
                            covariance_matrix_now = make_positive_definite(covariance_matrix_now).cuda()
                            multivariate_normal_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)
                            multivariate_normal_dist_now = torch.distributions.MultivariateNormal(mean_now, covariance_matrix_now)
                            X = multivariate_normal_dist_now.sample((sample_num,))
                            Y = multivariate_normal_dist.sample((sample_num,))
                            cost_matrix = compute_cost_matrix(X, Y)
                            a_w = np.ones(sample_num) / sample_num
                            b_w = np.ones(sample_num) / sample_num
                            cost_matrix_np = cost_matrix.cpu().numpy()
                            ot_plan = ot.emd(a_w, b_w, cost_matrix_np)
                            X_transformed = torch.tensor(np.dot(ot_plan, Y.cpu().numpy()), dtype=torch.float32).cuda()
                            samples.append(X_transformed)

                        samples = torch.cat(samples, dim=0)
                        enhanced_xc = torch.cat((xc, samples), dim=0)
                        LEN_label = len(biaoqian)
                        labels_bu = []
                        bboxes_bu = []
                        xr_bu = []
                        for i in range(LEN_label):
                            la = biaoqian[i]
                            cc1 = torch.LongTensor([la]).repeat(sample_num).to(device)
                            labels_bu.append(cc1)
                            cc2 = regression_targets[labels == la]
                            cc2 = cc2[0].unsqueeze(0).repeat(sample_num, 1)
                            bboxes_bu.append(cc2)
                            cc3 = xr[labels == la]
                            cc3 = cc3[0].unsqueeze(0).repeat(sample_num, 1)
                            xr_bu.append(cc3)

                        labels_bu = torch.cat(labels_bu, dim=0)
                        bboxes_bu = torch.cat(bboxes_bu, dim=0)
                        xr_bu = torch.cat(xr_bu, dim=0)
                        new_labels = torch.cat((labels, labels_bu), dim=0)
                        new_bboxes = torch.cat((regression_targets, bboxes_bu), dim=0)
                        new_xr = torch.cat((xr, xr_bu), dim=0)
                        class_logits, box_regression = self.predictor(enhanced_xc, new_xr)
                        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], new_labels, new_bboxes)
                        return (enhanced_xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg))
            else:
                class_logits, box_regression = self.predictor(xc, xr)
                result = self.post_processor((class_logits, box_regression), proposals)
                return xc, result, {}


def build_roi_box_head(cfg, in_channels):

    return ROIBoxHead(cfg, in_channels)


def distribution_calibration(query, base_means, base_cov, k=1, alpha=0.21):

    dist = []
    reduced_tensor = torch.mean(query.float(), dim=0)

    for i in range(len(base_means)):
        dist.append(F.cosine_similarity(reduced_tensor, base_means[i], dim=0))
    index = torch.topk(torch.tensor(dist), 1, largest=True).indices
    selected_means = torch.stack([base_means[i] for i in index], dim=0)
    selected_cov = torch.stack([base_cov[i] for i in index], dim=0)

    selected_means = selected_means.mean(dim=0)
    selected_cov = selected_cov.mean(dim=0)


    return selected_means, selected_cov


def compute_cost_matrix(X, Y):
    n = X.size(0)
    cost_matrix = torch.cdist(X, Y, p=2)**2
    return cost_matrix

def make_positive_definite(matrix, epsilon=1e-4):
    identity = torch.eye(matrix.size(0)).to(matrix.device)
    return matrix + epsilon * identity
