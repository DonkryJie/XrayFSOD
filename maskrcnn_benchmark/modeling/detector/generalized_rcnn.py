# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Implements the Generalized R-CNN framework.
"""
import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import torch.nn.functional as F


class Get_gradientmask_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradientmask_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0

def edge_loss(pred, edge):

    smooth = 1
    p = 2
    valid_mask = torch.ones_like(edge)
    pred = pred.contiguous().view(pred.shape[0], -1)
    edge = edge.contiguous().view(edge.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(pred, edge) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((pred.pow(p) + edge.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def structure_loss(pred, mask):

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def uncertainty_loss(uncertainty, sample, mask):
    kl_loss = torch.nn.KLDivLoss(size_average=False, reduce=False)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    bce = criterion(sample, mask)

    uncertainty = uncertainty.squeeze(1)
    uncertainty = uncertainty.cuda(non_blocking=True)
    uncertainty = F.log_softmax(uncertainty, dim=1)
    uncertainty = uncertainty.unsqueeze(1).float()

    kl = kl_loss(uncertainty, mask).mean()

    loss_u = 0.3 * kl + bce
    return loss_u

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.grad_sobel_gt = Get_gradientmask_nopadding()

    def forward(self, images, masks_gt, targets=None, oneStage=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)

        masks_gt = to_image_list(masks_gt)
        masks_gt = masks_gt.tensors

        b, c, h, w = masks_gt.shape

        edges_gt = self.grad_sobel_gt(masks_gt)

        features, mask_2, egde1 = self.backbone(images.tensors)


        sup_feats = None


        if self.training:
            edge_losses = structure_loss(F.interpolate(torch.sigmoid(egde1), size=(h, w), mode='bilinear', align_corners=False), edges_gt)

            aid_loss = 0.2 * edge_losses
            aid_losses = {"aid_losses": aid_loss}

        proposals, proposal_losses = self.rpn(images, features, targets)

        supTarget = None
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, sup_feats, supTarget, oneStage)
        else:
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(aid_losses)
            return losses

        return result