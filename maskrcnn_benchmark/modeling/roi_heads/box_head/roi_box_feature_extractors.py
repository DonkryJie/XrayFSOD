# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc

# from VAE_feat import VAE_feat

# from VAE_vector import VAE_vector
from novel_module import NovelModule


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_GODDD")
class FPN2MLPFeatureExtractor_GODDD(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_GODDD, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        # representation_size = 2 * representation_size
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.novel_module = NovelModule()

        self.prototype_fuse_layer = [nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1) for i in range(20 + 1)]
        for layer in self.prototype_fuse_layer:
            layer.weight = nn.Parameter(torch.ones(1, 1, 1))
            layer.bias = nn.Parameter(torch.zeros(1))
            if torch.cuda.is_available():
                layer = layer.cuda()

    def forward(self, x, proposals=None, sup_features=None, oneStage=None, gt_bb=None):

        # gt_bb = [N, 4]
        roi_x = self.pooler(x, proposals)

        if self.training:
            self.novel_module(roi_x, proposals, gt_bb)

        roi_x = roi_x.view(roi_x.size(0), -1)

        xr = F.relu(self.fc6r(roi_x))
        xr = F.relu(self.fc7r(xr))

        xc = F.relu(self.fc6c(roi_x))
        xc = F.relu(self.fc7c(xc))  # [512, 1024]

        xc_cpe_normalized = None
        xc_sup_cpe_normalized = None

        all_class_prototype_features = self.novel_module.get_class_prototype_feature()

        return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized, all_class_prototype_features


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio)

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION)

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        representation_size = 2*representation_size
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:
            roi_x = self.pooler(x, proposals)
            roi_x = roi_x.view(roi_x.size(0), -1)

            xr = F.relu(self.fc6r(roi_x))
            xr = F.relu(self.fc7r(xr))

            xc = F.relu(self.fc6c(roi_x))
            xc = F.relu(self.fc7c(xc))  # xc torch.Size([1024, 2048])

            xc_cpe_normalized = None
            xc_sup_cpe_normalized = None

            return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_VAE_feat")
class FPN2MLPFeatureExtractor_VAE_feat(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_VAE_feat, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler

        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.vae = VAE_feat(256, 256, 256)



    def forward(self, x, proposals=None, sup_features=None, oneStage=None):

        roi = self.pooler(x, proposals)

        if self.training:
            roi_recon, mu, log_var = self.vae(roi)
            loss_vae = self.vae.loss_function(roi, roi_recon, mu, log_var)
            roi_aug = self.vae.sample(roi.size(0))  # 获得1024个生成样本
        else:
            roi_aug = self.vae.sample(roi.size(0))  # 获得1024个生成样本

        roi_final = roi + roi_aug

        roi_final = roi_final.view(roi_final.size(0), -1)

        xr = F.relu(self.fc6r(roi_final))
        xr = F.relu(self.fc7r(xr))

        xc = F.relu(self.fc6c(roi_final))
        xc = F.relu(self.fc7c(xc))

        xc_cpe_normalized = None
        xc_sup_cpe_normalized = None

        if self.training:
            return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized, loss_vae
        else:
            return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_VAE_vector")
class FPN2MLPFeatureExtractor_VAE_vector(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_VAE_vector, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.vae = VAE_vector(256, 256, 256)

        self.reduceDim = nn.Linear(256*8*8+256, 256*8*8)

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):

        roi = self.pooler(x, proposals)

        roi_vce = self.avgpooler(roi)                 # [512*b, 256, 2, 2]
        roi_vce = roi_vce.view(roi_vce.size(0), -1)   # [512*b, 1024]

        roi = roi.view(roi.size(0), -1)               # [512*b, 256*8*8]

        if self.training:
            if oneStage:
                roi_recon, mu, log_var = self.vae(roi_vce)
                loss_vae = self.vae.loss_function(roi_vce, roi_recon, mu, log_var)
                roi_aug = self.vae.sample(roi_vce.size(0))  # 获得1024个生成样本
            else:
                roi_aug = self.vae.sample(roi_vce.size(0))  # 获得1024个生成样本
        else:
            roi_aug = self.vae.sample(roi_vce.size(0))  # 获得1024个生成样本

        roi_final = torch.cat((roi, roi_aug), dim=1)
        roi_final = self.reduceDim(roi_final)

        xr = F.relu(self.fc6r(roi_final))
        xr = F.relu(self.fc7r(xr))

        xc = F.relu(self.fc6c(roi_final))
        xc = F.relu(self.fc7c(xc))

        xc_cpe_normalized = None
        xc_sup_cpe_normalized = None

        if self.training:
            if oneStage:
                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized, loss_vae
            else:
                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_VAE")
class FPN2MLPFeatureExtractor_VAE(nn.Module):
    """
    Heads for FPN for classification.
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_VAE, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size



    def forward(self, x, proposals=None, sup_features=None, oneStage=None):

        roi = self.pooler(x, proposals)
        roi = roi.view(roi.size(0), -1)

        xr = F.relu(self.fc6r(roi))
        xr = F.relu(self.fc7r(xr))

        xc = F.relu(self.fc6c(roi))
        xc = F.relu(self.fc7c(xc))

        xc_cpe_normalized = None
        xc_sup_cpe_normalized = None

        return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_DaCL")
class FPN2MLPFeatureExtractor_DaCL(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_DaCL, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.fc_sup1 = make_fc(input_size, representation_size, use_gn)
        self.fc_sup2 = make_fc(representation_size, representation_size, use_gn)

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_sup = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            # print('sup_features: ', sup_features.shape)  # [15, 256, 8, 8]

            # sup_features = self.avgpooler(sup_features)  # [15, 256, 8, 8]

            sup_features = F.interpolate(sup_features, size=(8, 8), mode='bilinear', align_corners=True)  # 下采样为和支持特征同样的尺度

            sup_features = sup_features.view(sup_features.size(0), -1)  # [15, 256*8*8]

            if oneStage:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)

                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_cpe_normalized = None
                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
            else:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)
                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xc_cpe = self.head(xc)                                  # [512, 128]
                xc_cpe_normalized = F.normalize(xc_cpe, dim=1)          # [512, 128]

                xc_sup = F.relu(self.fc_sup1(sup_features))
                xc_sup = F.relu(self.fc_sup2(xc_sup))                   # [15, 1024]

                xc_sup_cpe = self.head_sup(xc_sup)                      # [15, 128]
                xc_sup_cpe_normalized = F.normalize(xc_sup_cpe, dim=1)  # [15, 128]

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor_DaCL_vae")
class FPN2MLPFeatureExtractor_DaCL_vae(nn.Module):
    """
    Heads for FPN for classification
    """
    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor_DaCL_vae, self).__init__()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.avgpooler = nn.AdaptiveAvgPool2d((resolution, resolution))
        self.fc6c = make_fc(input_size, representation_size, use_gn)
        self.fc7c = make_fc(representation_size, representation_size, use_gn)
        self.fc6r = make_fc(input_size, representation_size, use_gn)
        self.fc7r = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

        self.fc_sup1 = make_fc(representation_size, representation_size, use_gn)

        self.head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

        self.head_sup = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

    def forward(self, x, proposals=None, sup_features=None, oneStage=None):
        if proposals is not None:

            if oneStage:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)

                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                xc_cpe_normalized = None
                xc_sup_cpe_normalized = None

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
            else:
                x = self.pooler(x, proposals)
                x = x.view(x.size(0), -1)
                xc = F.relu(self.fc6c(x))
                xc = F.relu(self.fc7c(xc))

                xc_cpe = self.head(xc)                                  # [512, 128]
                xc_cpe_normalized = F.normalize(xc_cpe, dim=1)          # [512, 128]

                xc_sup = F.relu(self.fc_sup1(sup_features))             # [15, 1024]

                xc_sup_cpe = self.head_sup(xc_sup)                      # [15, 128]
                xc_sup_cpe_normalized = F.normalize(xc_sup_cpe, dim=1)  # [15, 128]

                xr = F.relu(self.fc6r(x))
                xr = F.relu(self.fc7r(xr))

                return xc, xr, xc_cpe_normalized, xc_sup_cpe_normalized
        else:
            features = []
            for feature in x:
                feature = self.avgpooler(feature)
                feature = feature.view(feature.size(0), -1)
                feature = F.relu(self.fc6c(feature))
                feature = self.fc7c(feature)
                features.append(feature)
            return features

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification.
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(output_size=(resolution, resolution), scales=scales, sampling_ratio=sampling_ratio)
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x

def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)
