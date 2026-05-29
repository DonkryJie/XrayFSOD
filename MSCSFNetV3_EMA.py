import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn.layers import GConv

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicConv2d_Gabor(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_Gabor, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.gconv = GConv(out_planes, out_planes//4, kernel_size=3, padding=1, stride=1, M=4, nScale=1, bias=False, expand=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x2 = x
        xg = self.gconv(x)
        x3 = x2 + xg
        x4 = self.bn(x3)
        x4 = self.relu(x4)
        x4 = self.conv2(x4)
        x4 = self.bn2(x4)
        x4 = self.relu2(x4)
        return x4

class CSFF(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(CSFF, self).__init__()
        self.cat2 = BasicConv2d(hidden_channels * 2, out_channels, kernel_size=1, padding=0)
        self.param_free_norm = nn.BatchNorm2d(hidden_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True))

        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, y, edge):

        y = F.interpolate(y, size=x.size()[2:], mode='nearest')
        xy = self.cat2(torch.cat((x, y), dim=1)) + x + y
        normalized = self.param_free_norm(xy)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv  = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out

class DenseInteractionDecoder(nn.Module):

    def __init__(self, channel):
        super(DenseInteractionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(2*channel, 2*channel, 3, padding=1)


        self.conv_concat3 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, channel, 1, padding=0)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x4, x3, x2):

        x4_1 = x4
        x3_1 = self.conv_upsample1(self.upsample(x4_1)) * x3
        x2_1 = self.conv_upsample2(self.upsample(x3_1)) * self.conv_upsample3(x2) * x2

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x4_1))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x2_2 = torch.cat((x2_1, self.conv_upsample6(self.upsample(x3_2))), 1)

        x2_2 = self.conv_concat4(x2_2)
        x = self.conv4(x2_2)
        x = self.conv5(x)

        return x

class CropLayer(nn.Module):
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

        self.convx1 = nn.Sequential(
            torch.nn.Conv2d(channels, 256, kernel_size=1, padding=0),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        output = self.convx1(out)

        return output

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.Translayer1_1 = BasicConv2d(256,  64, 1)
        self.Translayer2_1 = BasicConv2d(512,  64, 1)
        self.Translayer3_1 = BasicConv2d(1024, 64, 1)
        self.Translayer4_1 = BasicConv2d(2048, 64, 1)

        self.EMA1 = BasicConv2d_Gabor(256,  256, 1)
        self.EMA2 = BasicConv2d_Gabor(512,  256, 1)
        self.EMA3 = BasicConv2d_Gabor(1024, 256, 1)
        self.EMA4 = BasicConv2d_Gabor(2048, 256, 1)

        self.linear_x = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.CSFF1 = CSFF(256, 256)
        self.CSFF2 = CSFF(256, 256)
        self.CSFF3 = CSFF(256, 256)
        self.CSFF4 = CSFF(256, 256)

        self.DID = DenseInteractionDecoder(64)

    def forward(self, x):

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        egde1 = self.DID(x4_t, x3_t, x2_t)  # wz1: [2, 1, 80, 80]

        E_1 = self.EMA1(x1)  # [2, 256, 80, 80]
        E_2 = self.EMA2(x2)  # [2, 256, 40, 40]
        E_3 = self.EMA3(x3)  # [2, 256, 20, 20]
        E_4 = self.EMA4(x4)  # [2, 256, 10, 10]

        R_4 = self.CSFF4(E_4, E_4, egde1)  # R_4: [2, 256, 10, 10]
        R_3 = self.CSFF3(E_3, R_4, egde1)  # R_3: [2, 256, 20, 20]
        R_2 = self.CSFF2(E_2, R_3, egde1)  # R_2: [2, 256, 40, 40]
        R_1 = self.CSFF1(E_1, R_2, egde1)  # R_1: [2, 256, 80, 80]

        mask_4 = self.linear_x(R_4)  # [2, 1, 10, 10]

        return R_1, R_2, R_3, R_4, mask_4, egde1


if __name__ == '__main__':

    input = []
    input1 = torch.randn(2, 256,  80, 80)
    input2 = torch.randn(2, 512,  40, 40)
    input3 = torch.randn(2, 1024, 20, 20)
    input4 = torch.randn(2, 2048, 10, 10)

    input.append(input1)
    input.append(input2)
    input.append(input3)
    input.append(input4)

    net = Network()

    output = net(input)

    print('output', output[1].shape)



