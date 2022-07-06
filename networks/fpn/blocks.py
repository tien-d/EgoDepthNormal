import logging
from typing import List
import torch
import torch.nn as nn
import torchvision
import collections
import math
from networks.network_utils import weights_init


class ResNetPyramids(nn.Module):
    def __init__(self, in_channels=3, pretrained=False, resnet_arch=18):
        super(ResNetPyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(resnet_arch)](pretrained=pretrained)

        self.channel = in_channels

        self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        # self.layer1 = nn.Sequential(
        #     pretrained_model.conv1, pretrained_model.bn1, pretrained_model.relu, pretrained_model.maxpool, pretrained_model.layer1
        # )
        # self.layer2 = pretrained_model.layer2
        # self.layer3 = pretrained_model.layer3
        # self.layer4 = pretrained_model.layer4

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        if pretrained is False:
            weights_init(self.modules(), init_type='kaiming')

    def forward(self, x):
        # print('x.shape: ', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class ChannelReduction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ChannelReduction, self).__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.channel_reduction(x)
        return x


class ExtendedUpsample(nn.Module):
    def __init__(self, in_ch, scale_upsample=2, ch_downsample=2, out_spatial=None):
        super(ExtendedUpsample, self).__init__()
        if out_spatial is not None:
            self.extended_upsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // ch_downsample, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // ch_downsample, in_ch // ch_downsample, 3, 1, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Upsample(size=out_spatial, mode='bilinear', align_corners=False),
            )
        else:
            self.extended_upsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // ch_downsample, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // ch_downsample, in_ch // ch_downsample, 3, 1, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_upsample, mode='bilinear', align_corners=False),
            )

    def forward(self, x):
        x = self.extended_upsample(x)
        return x