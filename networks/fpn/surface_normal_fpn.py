import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models
import collections
import math
import logging
from networks.network_utils import count_parameters
# from depth_completion import ResNetPyramids
from networks.fpn.blocks import ResNetPyramids, ChannelReduction, ExtendedUpsample


class ExtendedFPNSurfaceNormal(nn.Module):
    def __init__(self, arguments):
        self.args = arguments

        assert (self.args.resnet_arch in [18, 34, 50, 101, 152]), \
            'Only ResNet-18/34/50/101/152 are defined, but got {} layers here!'.format(self.args.resnet_arch)
        super(ExtendedFPNSurfaceNormal, self).__init__()

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

        # 240x320
        # self.feature4_upsampling = nn.Sequential(
        #     ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
        #     ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, out_spatial=(15, 20), ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        #     ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        # )

        # 384x384 and 480x640
        self.feature4_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders,
                      (self.input_channels[0] // 4) * self.num_encoders, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((self.input_channels[0] // 4) * self.num_encoders, 3, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        if self.__class__.__name__ == 'ExtendedFPNSurfaceNormal':
            logging.info("Backbone: ResNet-{}. Number of parameters in model: {}".format(self.args.resnet_arch,
                                                                                         count_parameters(self)))

    def forward(self, image, q=None):
        i1, i2, i3, i4 = self.resnet_rgb(image)
        z1 = self.feature1_upsampling(i1)
        z2 = self.feature2_upsampling(i2)
        z3 = self.feature3_upsampling(i3)
        z4 = self.feature4_upsampling(i4)
        y = self.feature_concat(z1 + z2 + z3 + z4)
        return y