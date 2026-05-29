# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

import numpy as np

class CVAE(nn.Module):

    def __init__(self, in_channels, latent_dim, hidden_dim):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(in_channels*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim*2, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, in_channels), nn.BatchNorm1d(in_channels), nn.Sigmoid())

    def encode(self, input, s):
        result = torch.cat((input, s), dim=1)
        result = self.encoder(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z, s):
        z_s = torch.cat((z, s), dim=1)
        z_s = self.decoder_input(z_s)
        z_out = self.decoder(z_s)
        return z_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, s, **kwargs):
        mu, log_var = self.encode(input, s)
        z = self.reparameterize(mu, log_var)
        z_out = self.decode(z, s)
        return z_out, mu, log_var

    def loss_function(self, input, rec, mu, log_var, kld_weight=0.5):
        recons_loss = F.mse_loss(rec, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return loss

    def sample(self, n, s):  # 定义生成过程
        z = torch.randn(n, self.latent_dim).cuda()  # 从标准正态分布中采样得到n个采样变量Z，长度为latent_dim
        images = self.decode(z, s)  # 经过解码过程，得到生成样本Y
        return images  #  返回生成样本Y

class ROIBoxHead(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.cvae = CVAE(1024, 1024, 1024)

        self.head_semantic = nn.Sequential(nn.Linear(1152, 1024), nn.BatchNorm1d(1024),  nn.LeakyReLU(),
                                           nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU())

    # features, proposals, targets, sup_features, supTarget, oneStage

    def forward(self, features, proposals, targets=None, semantic=None, supTarget=None, oneStage=None):

        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized = self.feature_extractor(features, proposals)

        sem_vector = self.head_semantic(semantic)                   # [N, 512]
        random_index = np.random.choice(range(sem_vector.size(0)))
        s = sem_vector[random_index].repeat(xc.size(0), 1)          # [512*B, 512]

        if self.training:
            xc_recon, mu, log_var = self.cvae(xc, s)
            loss_vae = self.cvae.loss_function(xc, xc_recon, mu, log_var)
            with torch.no_grad():
                xc_sample = self.cvae.sample(xc.size(0), s)  # 生成512*B个样本
            xc_final = xc_sample + xc
            class_logits, box_regression = self.predictor(xc_final, xr)

            if xc_cpe_normalized is None:
                loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], xc_cpe_normalized, xc_sup_cpe_normalized, supTarget)
                return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_vae=0.1 * loss_vae))
            else:
                loss_classifier, loss_box_reg, loss_dacl = self.loss_evaluator([class_logits], [box_regression], xc_cpe_normalized, xc_sup_cpe_normalized, supTarget)
                return (xc, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_dacl=loss_dacl, loss_vae=0.1 * loss_vae))
        else:
            sample_xc = self.cvae.sample(xc.size(0), s)  # 生成512*B个样本
            xc_final = sample_xc + xc
            class_logits, box_regression = self.predictor(xc_final, xr)
            result = self.post_processor((class_logits, box_regression), proposals)
            return xc_final, result, {}

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class and make it a parameter in the config.
    """
    return ROIBoxHead(cfg, in_channels)