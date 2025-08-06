# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F
from itertools import product, combinations, chain
from torchviz import make_dot
from .base_module import UNetLayer, LearntNeighbourhoodSampling, Features2Features, Feature2VertexLayer, adjacency_matrix

# 论文标题：
# Image2Landmarks: Multi-scale Feature based Graph Convolutional Network for Medical Landmark Detection


def prepective_transform(vertices, transform_matrix):
    """进行仿射变换
    :param vertices: 默认的顶点坐标
    :param transform_matrix: 变换矩阵的参数
    :return:
    """
    x = transform_matrix[:, 0].view(-1, 1) * vertices[:, :, 0] + \
        transform_matrix[:, 1].view(-1, 1) * vertices[:, :, 1] + \
        transform_matrix[:, 2].view(-1, 1)
    y = transform_matrix[:, 3].view(-1, 1) * vertices[:, :, 0] + \
        transform_matrix[:, 4].view(-1, 1) * vertices[:, :, 1] + \
        transform_matrix[:, 5].view(-1, 1)
    transform_vertices = torch.stack([x, y], dim=-1)
    return transform_vertices



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


class Image2Shape(nn.Module):
    def __init__(self, config):
        super(Image2Shape, self).__init__()
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

        """GCN图卷积的框架层"""
        # 第五个特征采样 -> unet_encoder4
        self.graph_sample5 = LearntNeighbourhoodSampling(features_count=config.first_layer_channels * 2 ** 4)
        self.up_f2f_layer5 = Features2Features(in_features=config.first_layer_channels * 2 ** 4 + 2, out_features=128)
        # self.up_f2v_layer5 = Feature2VertexLayer(in_features=32, hidden_layer_count=1)
        self.up_f2v_layer5 = nn.Linear(in_features=128, out_features=6)
        # 第四个特征采样
        self.graph_sample4 = LearntNeighbourhoodSampling(features_count=config.first_layer_channels * 2 ** 3)
        self.up_f2f_layer4 = Features2Features(in_features=config.first_layer_channels * 2 ** 3 + 2, out_features=128)
        self.up_f2v_layer4 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第三个特征采样
        self.graph_sample3 = LearntNeighbourhoodSampling(features_count=config.first_layer_channels * 2 ** 2)
        self.up_f2f_layer3 = Features2Features(in_features=config.first_layer_channels * 2 ** 2 + 2, out_features=128)
        self.up_f2v_layer3 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第二个特征采样
        self.graph_sample2 = LearntNeighbourhoodSampling(features_count=config.first_layer_channels * 2 ** 1)
        self.up_f2f_layer2 = Features2Features(in_features=config.first_layer_channels * 2 ** 1 + 2, out_features=128)
        self.up_f2v_layer2 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第一个特征采样
        self.graph_sample1 = LearntNeighbourhoodSampling(features_count=config.first_layer_channels * 2 ** 0)
        self.up_f2f_layer1 = Features2Features(in_features=config.first_layer_channels * 2 ** 0 + 2, out_features=128)
        self.up_f2v_layer1 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)

    def forward(self, x, vertices, A, D):
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

        """图卷积操作的层"""
        # 第五个图卷积解码层
        features = self.graph_sample5(x_encoder4, vertices)
        latent_features = torch.cat([features, vertices], dim=2)
        latent_features = self.up_f2f_layer5(latent_features, A, D)
        # delta_vertices = self.up_f2v_layer5(latent_features, A, D)
        # vertices5 = vertices + delta_vertices
        latent_features = torch.mean(latent_features, dim=1)
        transform_matrix = self.up_f2v_layer5(latent_features)
        vertices5 = prepective_transform(vertices, transform_matrix)
        # 第四个图卷积解码层
        features = self.graph_sample4(x_decoder4, vertices5)
        latent_features = torch.cat([features, vertices5], dim=2)
        latent_features = self.up_f2f_layer4(latent_features, A, D)
        delta_vertices = self.up_f2v_layer4(latent_features, A, D)
        vertices4 = vertices5 + delta_vertices
        # 第三个图卷积解码层
        features = self.graph_sample3(x_decoder3, vertices4)
        latent_features = torch.cat([features, vertices4], dim=2)
        latent_features = self.up_f2f_layer3(latent_features, A, D)
        delta_vertices = self.up_f2v_layer3(latent_features, A, D)
        vertices3 = vertices4 + delta_vertices
        # 第二个图卷积解码层
        features = self.graph_sample2(x_decoder2, vertices3)
        latent_features = torch.cat([features, vertices3], dim=2)
        latent_features = self.up_f2f_layer2(latent_features, A, D)
        delta_vertices = self.up_f2v_layer2(latent_features, A, D)
        vertices2 = vertices3 + delta_vertices
        # 第一个图卷积解码层
        features = self.graph_sample1(x_decoder1, vertices2)
        latent_features = torch.cat([features, vertices2], dim=2)
        latent_features = self.up_f2f_layer1(latent_features, A, D)
        delta_vertices = self.up_f2v_layer1(latent_features, A, D)
        vertices1 = vertices2 + delta_vertices

        """返回：热图，以及不同尺度的顶点位置预测"""
        return out, vertices5, vertices4, vertices3, vertices2, vertices1


if __name__ == '__main__':
    class Config():
        def __init__(self):
            super(Config, self).__init__()


    def load_config(exp_id):
        cfg = Config()

        """实验配置"""
        cfg.experiment_idx = exp_id
        cfg.trial_id = None

        cfg.save_dir_prefix = 'Experiment_'  # prefix for experiment folder
        cfg.name = 'voxel2mesh'

        """数据集配置"""
        cfg.patch_shape = (64, 64, 64)

        cfg.ndims = 2
        cfg.augmentation_shift_range = 10

        """模型配置"""
        cfg.first_layer_channels = 32
        cfg.num_input_channels = 1
        cfg.steps = 4

        # Only supports batch size 1 at the moment.
        cfg.batch_size = 1

        cfg.num_classes = 19
        cfg.batch_norm = True
        cfg.graph_conv_layer_count = 4

        """优化器"""
        cfg.learning_rate = 1e-4

        """训练配置"""
        cfg.numb_of_itrs = 300000
        cfg.eval_every = 1000  # saves results to disk

        return cfg


    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    config = load_config(100)
    model = Image2Shape(config).to(device)
    import numpy as np
    x = torch.from_numpy(np.zeros((1, 1, 800, 640))).float()
    vertices = torch.tensor([[[0, 0],
                              [2, 1],
                              [4, 5]]]).float() / (7 - 1) * 2 - 1
    A, D = adjacency_matrix(vertices)
    y = model(x, vertices, A, D)
    # print(model)
    print(y)
    # summary(model, (1, 800, 640), device='cpu')

    make_dot(y, params=dict(list(model.named_parameters()))).render("U-Net_torchviz", format="png")
