# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F

class UNetLayer(nn.Module):
    """U-Net模型的层"""
    def __init__(self, num_channels_in, num_channels_out, ndims=2):

        super(UNetLayer, self).__init__()

        conv_op = nn.Conv2d if ndims == 2 else nn.Conv3d
        batch_nrom_op = nn.BatchNorm2d if ndims == 2 else nn.BatchNorm3d

        conv1 = conv_op(num_channels_in,  num_channels_out, kernel_size=3, padding=1)
        conv2 = conv_op(num_channels_out, num_channels_out, kernel_size=3, padding=1)

        bn1 = batch_nrom_op(num_channels_out)
        bn2 = batch_nrom_op(num_channels_out)
        self.unet_layer = nn.Sequential(conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU())

    def forward(self, x):
        return self.unet_layer(x)

class UNetDecoder(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.up_conv = nn.Conv2d(config.first_layer_channels * 2 ** depth, config.first_layer_channels * 2 ** (depth - 1), kernel_size=1, stride=1)
        self.unet_layer = UNetLayer(config.first_layer_channels * 2 ** depth, config.first_layer_channels * 2 ** (depth - 1))

    def forward(self, x, skip_x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        up_x = self.up_conv(x)
        cat_x = torch.cat((skip_x, up_x), dim=1)
        out_x = self.unet_layer(cat_x)
        return out_x


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        """U-Net的框架层"""
        # U-Net: 第一个卷积层
        self.first_layer = UNetLayer(config.num_input_channels, config.first_layer_channels)

        # 第一个池化层 + 卷积
        self.unet_encoder1 = nn.Sequential(nn.MaxPool2d(2),
                                           UNetLayer(config.first_layer_channels * 2 ** 0, config.first_layer_channels * 2 ** 1))

        # 第二个池化层 + 卷积
        self.unet_encoder2 = nn.Sequential(nn.MaxPool2d(2),
                                           UNetLayer(config.first_layer_channels * 2 ** 1, config.first_layer_channels * 2 ** 2))

        # 第三个池化层 + 卷积
        self.unet_encoder3 = nn.Sequential(nn.MaxPool2d(2),
                                           UNetLayer(config.first_layer_channels * 2 ** 2, config.first_layer_channels * 2 ** 3))

        # 第四个池化层 + 卷积
        self.unet_encoder4 = nn.Sequential(nn.MaxPool2d(2),
                                           UNetLayer(config.first_layer_channels * 2 ** 3, config.first_layer_channels * 2 ** 4))

        # 第四个上采样 + 卷积
        self.unet_decoder4 = UNetDecoder(config, depth=4)

        # 第三个上采样 + 卷积
        self.unet_decoder3 = UNetDecoder(config, depth=3)

        # 第二个上采样 + 卷积
        self.unet_decoder2 = UNetDecoder(config, depth=2)

        # 第一个上采样 + 卷积
        self.unet_decoder1 = UNetDecoder(config, depth=1)

        # U-Net: 最后一个卷积'''
        self.final_layer = nn.Conv2d(in_channels=config.first_layer_channels, out_channels=config.num_classes, kernel_size=1)

    def forward(self, x):
        """输入图像, 关键点template的坐标, 邻接矩阵, 度矩阵"""

        """U-Net模型的层"""
        # 第一层
        x = self.first_layer(x)

        # 第一个下采样
        x_encoder1 = self.unet_encoder1(x)
        # 第二个下采样
        x_encoder2 = self.unet_encoder2(x_encoder1)
        # 第三个下采样
        x_encoder3 = self.unet_encoder3(x_encoder2)
        # 第四个下采样
        x_encoder4 = self.unet_encoder4(x_encoder3)

        # 第四个上采样
        x_decoder4 = self.unet_decoder4(x_encoder4, x_encoder3)
        # 第三个上采样
        x_decoder3 = self.unet_decoder3(x_decoder4, x_encoder2)
        # 第二个上采样
        x_decoder2 = self.unet_decoder2(x_decoder3, x_encoder1)
        # 第一个上采样
        x_decoder1 = self.unet_decoder1(x_decoder2, x)

        # 最后层
        out = self.final_layer(x_decoder1)
        """返回：热图，以及不同尺度的顶点位置预测"""
        return out

