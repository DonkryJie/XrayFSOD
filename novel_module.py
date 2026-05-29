import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
#  import fvcore.nn.weight_init as weight_init
from maskrcnn_benchmark.config import cfg
import gc
import torch.nn.functional as F
import pickle

class NovelModule(nn.Module):

    def __init__(self, ):
        super().__init__()

        self.num_classes = 11 + 1  # 1 for background
        self.bg_clsid = 0
        self.prototypes = {k: None for k in range(self.num_classes)}
        self.bg_bottom_k = 256
        self.prototypes_fuse_alpha = 0.3
        self.iou_thresh = 0.7

        self.feature_extractor = nn.Sequential(
            nn.Linear(256*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

        self.prototypes_feature = { k : None for k in range(self.num_classes)}

        '''
        if cfg.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT != None:
            with open(cfg.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT, 'rb') as f:
                self.prototypes_feature = pickle.load(f)
        '''

        self.prototypes_feature_fuse_alpha = 0.3

    def forward(self, box_features, proposals, gt_bb):
        gt_classes = []
        for i in range(len(proposals)):
            zzzz = proposals[i].get_field("labels")  # [512]
            gt_classes.append(zzzz)
        gt_classes = torch.cat(gt_classes, dim=0)


        concat_boxes = torch.cat([b.bbox for b in proposals], dim=0)
        roi_bb = torch.cat([concat_boxes], dim=1).detach()  # [512, 4]

        intersection_xmin = torch.max(roi_bb[:, 0].unsqueeze(1), gt_bb[:, 0].unsqueeze(0))
        intersection_ymin = torch.max(roi_bb[:, 1].unsqueeze(1), gt_bb[:, 1].unsqueeze(0))
        intersection_xmax = torch.min(roi_bb[:, 2].unsqueeze(1), gt_bb[:, 2].unsqueeze(0))
        intersection_ymax = torch.min(roi_bb[:, 3].unsqueeze(1), gt_bb[:, 3].unsqueeze(0))

        # Calculate intersection area
        intersection_width = torch.clamp(intersection_xmax - intersection_xmin, min=0)
        intersection_height = torch.clamp(intersection_ymax - intersection_ymin, min=0)
        intersection_area = intersection_width * intersection_height

        # Calculate union area
        area_box_a = (roi_bb[:, 2] - roi_bb[:, 0]) * (roi_bb[:, 3] - roi_bb[:, 1])
        area_box_b = (gt_bb[:, 2] - gt_bb[:, 0]) * (gt_bb[:, 3] - gt_bb[:, 1])
        union_area = area_box_a.unsqueeze(1) + area_box_b.unsqueeze(0) - intersection_area

        # Calculate IoU
        IOU = intersection_area / union_area


        ious, _ = torch.max(IOU, dim=1, keepdim=True)  # [512, 1]
        ious = ious.squeeze(-1)                        # [512, 1]

        bg_mask = gt_classes == self.bg_clsid   #  BG=TURE

        bg_features = box_features[bg_mask]  # [512, 256, 7, 7]
        bg_ious = ious[bg_mask]

        # sorted by ious, choose the k lowest ious for bg
        sorted_bg_ious, sorted_bg_ids = torch.sort(bg_ious)
        retain_num = min(self.bg_bottom_k, bg_ious.shape[0]) #  256
        sorted_bg_ids_retained = sorted_bg_ids[:retain_num]
        bg_features = bg_features[sorted_bg_ids_retained] #  ([256, 256, 8, 8])

        # merge new proposals into prototype for non-bg classes
        filter_mask = ious > self.iou_thresh  # R x K  # [512, 1]

        filter_inds = filter_mask.nonzero()

        num_filtered = filter_inds.shape[0]
        gt_classes = gt_classes[filter_mask]

        ious = ious[filter_mask]
        box_features = box_features[filter_mask]

        gt_classes = gt_classes.chunk(num_filtered, 0)
        ious = ious.chunk(num_filtered, 0)
        box_features = [torch.squeeze(x, dim=0) for x in box_features.chunk(num_filtered, 0)]


        proposals_per_class = {k: {'iou': [], 'feature': []} for k in range(self.num_classes)}
        for gt, iou, feature in zip(gt_classes, ious, box_features):
            ids = gt.item()
            proposals_per_class[ids]['iou'].append(iou)
            proposals_per_class[ids]['feature'].append(feature)

        # aggregate each prototype of this batch according to the iou weight
        prototypes_per_batch = {k: None for k in range(self.num_classes)}

        for ids, proposals in proposals_per_class.items():

            if len(proposals['iou']) == 0 and len(proposals['feature']) == 0:
                continue
            ious = torch.cat(proposals['iou']).reshape(-1, 1, 1, 1)
            ious = ious.cuda()

            features = torch.stack(proposals['feature'], dim=0)

            prototypes_per_batch[ids] = torch.div(torch.sum(features * ious, dim=0), torch.sum(ious))
            del ious, features, proposals
            gc.collect()

        # build bg's prototype
        prototypes_per_batch[self.bg_clsid] = torch.mean(bg_features, dim=0)


        # update global prototype and prototype feature
        for ids, prototype in prototypes_per_batch.items():

            if prototype is None:
                continue

            if self.prototypes[ids] is None:
                self.prototypes[ids] = prototype.clone()
            else:
                current_prototype = self.prototypes[ids].detach()

                self.prototypes[ids] = self.prototypes_fuse_alpha * prototype + (1 - self.prototypes_fuse_alpha) * current_prototype

            ddd = self.prototypes[ids].unsqueeze(dim=0)
            ddd2 = torch.flatten(ddd, start_dim=1)  # [1, 16384]

            new_prototypes_feature = self.feature_extractor(ddd2)  # [1, 1024]
            self.prototypes_feature[ids] = new_prototypes_feature.detach()
            del new_prototypes_feature, prototype

        return

    def get_class_prototype_feature(self):
        features = [feature for feature in self.prototypes_feature.values()]
        if None in features:
            return None
        all_features = torch.cat(features, dim=0)
        return all_features

    def get_all_prototype_feature(self, labels):
        if labels == None:   #  change !!!!!
            return None
        return torch.cat([self.prototypes_feature[i] for i in labels.tolist()], dim=0)

    def get_prototype_ids_cosinesim(self, box_features):
        all_features = self.get_class_prototype_feature()
        if all_features == None:   #  change !!!!!
            return None
        all_features = F.normalize(all_features, dim=1)
        box_features = F.normalize(box_features, dim=1)
        similarities = torch.exp(torch.matmul(box_features, all_features.T))
        sim_max, sim_max_ids = torch.max(similarities, dim=1, keepdim=True)
        return sim_max_ids.squeeze( )

def prototypes_loss(all_features, factor):
    if all_features == None:
        return None
    all_features = F.normalize(all_features, dim=1)
    similarities = torch.matmul(all_features, all_features.T)
    abs_sim = torch.abs(similarities)

    mask = torch.ones_like(abs_sim)
    mask.fill_diagonal_(0)
    bool_mask = mask == 1
    abs_sim_masked = abs_sim[bool_mask]
    loss = torch.mean(abs_sim_masked)

    return factor * loss
