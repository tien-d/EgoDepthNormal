import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models
import collections
import math
import logging

from networks.fpn.blocks import ResNetPyramids, ChannelReduction, ExtendedUpsample
# from networks.depth_completion import *
from networks.network_utils import count_parameters
# from networks.warping_2dof_alignment import Warping2DOFAlignment
# from depth_completion import ResNetPyramids


class ExtendedFPNSurfaceNormalConvexUpsampling(nn.Module):
    def __init__(self, arguments):
        self.args = arguments
        assert (self.args.resnet_arch in [18, 34, 50, 101, 152]), \
            'Only ResNet-18/34/50/101/152 are defined, but got {} layers here!'.format(self.args.resnet_arch)
        super(ExtendedFPNSurfaceNormalConvexUpsampling, self).__init__()

        self.resnet_rgb = ResNetPyramids(in_channels=3, pretrained=True, resnet_arch=self.args.resnet_arch)
        self.num_encoders = 1

        if self.args.resnet_arch <= 34:
            max_channels = 512
        else:
            max_channels = 2048

        self.input_channels = [max_channels // 8, max_channels // 4, max_channels // 2, max_channels]

        self.feature1_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[0] * self.num_encoders, out_ch=(self.input_channels[0] // 2) * self.num_encoders)
        )

        self.feature2_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[1] * self.num_encoders, out_ch=self.input_channels[0] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        self.feature3_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[2] * self.num_encoders, out_ch=self.input_channels[1] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        self.feature4_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, out_spatial=(15, 20), ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        self.normal_upsampling = nn.Sequential(
            nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 4 * 3 * 3, 1)
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders,
                      (self.input_channels[0] // 4) * self.num_encoders, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((self.input_channels[0] // 4) * self.num_encoders, 3, 3, 1, 1),
        )

        if self.__class__.__name__ == 'ExtendedFPNSurfaceNormalConvexUpsampling':
            logging.info("Backbone: ResNet-{}. Number of parameters in model: {}".format(self.args.resnet_arch,
                                                                                         count_parameters(self)))

    # Upsample normal [H/4, W/4, 3] -> [H, W, 3] using convex combination
    @staticmethod
    def upsample_normal(normal, mask):
        N, _, H, W = normal.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_normal = F.unfold(normal, [3, 3], padding=1)
        up_normal = up_normal.view(N, 3, 9, 1, 1, H, W)

        up_normal = torch.sum(mask * up_normal, dim=2)
        up_normal = up_normal.permute(0, 1, 4, 2, 5, 3)
        return up_normal.reshape(N, 3, 4 * H, 4 * W)


    def forward(self, image):
        i1, i2, i3, i4 = self.resnet_rgb(image)
        z1 = self.feature1_upsampling(i1)
        z2 = self.feature2_upsampling(i2)
        z3 = self.feature3_upsampling(i3)
        z4 = self.feature4_upsampling(i4)
        h_0 = z1 + z2 + z3 + z4

        # Initial normal estimate
        n_0 = self.feature_concat(h_0)
        m_0 = self.normal_upsampling(h_0)
        y = self.upsample_normal(normal=n_0, mask=m_0)
        return y


class ExtendedFPNDepth(nn.Module):
    def __init__(self, arguments):
        self.args = arguments

        if self.args.use_spatial_rectifier:
            self.rectifier = Warping2DOFAlignment(device='cuda')
        else:
            self.rectifier = None

        assert (self.args.resnet_arch in [18, 34, 50, 101, 152]), \
            'Only ResNet-18/34/50/101/152 are defined, but got {} layers here!'.format(self.args.resnet_arch)
        super(ExtendedFPNDepth, self).__init__()

        self.resnet_rgb = ResNetPyramids(in_channels=3, pretrained=True, resnet_arch=self.args.resnet_arch)
        self.num_encoders = 1

        if self.args.resnet_arch <= 34:
            max_channels = 512
        else:
            max_channels = 2048

        self.input_channels = [max_channels // 8, max_channels // 4, max_channels // 2, max_channels]

        self.feature1_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[0] * self.num_encoders, out_ch=(self.input_channels[0] // 2) * self.num_encoders),
        )

        self.feature2_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[1] * self.num_encoders, out_ch=self.input_channels[0] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        self.feature3_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[2] * self.num_encoders, out_ch=self.input_channels[1] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        # self.feature4_upsampling = nn.Sequential(
        #     ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
        #     ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, out_spatial=(15, 20), ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders // 2, scale_upsample=2, ch_downsample=1), # this is for 480x640 input image
        # )

        # 384x384 and 480x640
        self.feature4_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        )

        # # 240x320
        # self.feature4_upsampling = nn.Sequential(
        #     ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
        #     ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, out_spatial=(15, 20), ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        # )

        self.feature_concat = nn.Sequential(
            nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders,
                      (self.input_channels[0] // 4) * self.num_encoders, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((self.input_channels[0] // 4) * self.num_encoders, 1, 1, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        # # gradual upsampling
        # self.feature_concat = nn.Sequential(
        #     nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders,
        #               (self.input_channels[0] // 4) * self.num_encoders, 3, 1, 1),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d((self.input_channels[0] // 4) * self.num_encoders,
        #               (self.input_channels[0] // 8) * self.num_encoders, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d((self.input_channels[0] // 8) * self.num_encoders,
        #               (self.input_channels[0] // 16) * self.num_encoders, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((self.input_channels[0] // 16) * self.num_encoders,
        #               1, kernel_size=1, padding=0),
        #     # nn.ReLU(inplace=True),
        # )

        # self.feature1_upsampling = nn.Sequential(
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1) # add for testing
        # )
        #
        # self.feature2_upsampling = nn.Sequential(
        #     ChannelReduction(in_ch=self.input_channels[1] * self.num_encoders, out_ch=self.input_channels[0] * self.num_encoders),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1) # add for testing
        # )
        #
        # self.feature3_upsampling = nn.Sequential(
        #     ChannelReduction(in_ch=self.input_channels[2] * self.num_encoders, out_ch=self.input_channels[1] * self.num_encoders),
        #     ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1) # add for testing
        # )
        #
        # self.feature4_upsampling = nn.Sequential(
        #     ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
        #     ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=1),  # add for testing
        # )
        #
        # self.feature_concat = nn.Sequential(
        #     nn.Conv2d((self.input_channels[0]) * self.num_encoders,
        #               (self.input_channels[0] // 2) * self.num_encoders, 3, 1, 1),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders,
        #               (self.input_channels[0] // 8) * self.num_encoders, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((self.input_channels[0] // 8) * self.num_encoders,
        #               1, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        # )

        if self.__class__.__name__ == 'ExtendedFPNDepth':
            logging.info("Backbone: ResNet-{}. Number of parameters in model: {}".format(self.args.resnet_arch,
                                                                                         count_parameters(self)))

    def forward(self, image, q=None):
        if self.rectifier is None:
            i1, i2, i3, i4 = self.resnet_rgb(image)
            z1 = self.feature1_upsampling(i1)
            z2 = self.feature2_upsampling(i2)
            z3 = self.feature3_upsampling(i3)
            z4 = self.feature4_upsampling(i4)
            y = self.feature_concat(z1 + z2 + z3 + z4)
            return y
        else:
            R_inv, img_sampler, inv_img_sampler = self.rectifier.image_sampler_forward_inverse(q)

            # Step 1: Warp input to be canonical
            w_x = torch.nn.functional.grid_sample(image, img_sampler, padding_mode='zeros', mode='bilinear')

            # Step 2: Canonical view
            i1, i2, i3, i4 = self.resnet_rgb(w_x)
            z1 = self.feature1_upsampling(i1)
            z2 = self.feature2_upsampling(i2)
            z3 = self.feature3_upsampling(i3)
            z4 = self.feature4_upsampling(i4)
            w_y = self.feature_concat(z1 + z2 + z3 + z4)

            # Step 3: Inverse warp the output to be pixel wise with input
            y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
            y = y.view(image.shape[0], image.shape[1], image.shape[2] * image.shape[3])
            z = (R_inv.bmm(y)).view(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
            return z