# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F
from itertools import product, combinations, chain
from .base_module import PatternSampling, PointSampling, Features2Features, Feature2VertexLayer, adjacency_matrix
from .backbone import FPN101_NO, FPN50, FPN18, ResNext50_FPN

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


class Image2Shape(nn.Module):
    def __init__(self, config):
        super(Image2Shape, self).__init__()
        """骨干网络"""
        self.backbone = FPN101_NO(pretrained=True, output_stride=16)

        n_feature = 1024
        """GCN图卷积的框架层"""
        # 第一个特征采样
        self.graph_sample1 = PatternSampling(features_count=n_feature)
        self.up_f2f_layer1 = Features2Features(in_features=n_feature + 2, out_features=128, hidden_layer_count=1)
        # self.up_f2v_layer1 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        self.up_f2v_layer1 = nn.Linear(in_features=128, out_features=9)
        # 第二个特征采样
        self.graph_sample2 = PatternSampling(features_count=n_feature)
        self.up_f2f_layer2 = Features2Features(in_features=n_feature + 2, out_features=128, hidden_layer_count=1)
        self.up_f2v_layer2 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第三个特征采样
        self.graph_sample3 = PatternSampling(features_count=n_feature)
        self.up_f2f_layer3 = Features2Features(in_features=n_feature + 2, out_features=128, hidden_layer_count=1)
        self.up_f2v_layer3 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第四个特征采样
        self.graph_sample4 = PatternSampling(features_count=n_feature)
        self.up_f2f_layer4 = Features2Features(in_features=n_feature + 2, out_features=128, hidden_layer_count=1)
        self.up_f2v_layer4 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第五个特征采样 -> unet_encoder4
        self.graph_sample5 = PatternSampling(features_count=n_feature)
        self.up_f2f_layer5 = Features2Features(in_features=n_feature + 2, out_features=128, hidden_layer_count=1)
        self.up_f2v_layer5 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)
        # 第六个特征采样 -> unet_encoder4
        self.graph_sample6 = PatternSampling(features_count=n_feature)
        self.up_f2f_layer6 = Features2Features(in_features=n_feature + 2, out_features=128, hidden_layer_count=1)
        self.up_f2v_layer6 = Feature2VertexLayer(in_features=128, hidden_layer_count=1)

    def upsample_cat(self, p1, p2, p3, p4):
        p2 = nn.functional.interpolate(p2, size=p1.size()[2:], mode='bilinear')
        p3 = nn.functional.interpolate(p3, size=p1.size()[2:], mode='bilinear')
        p4 = nn.functional.interpolate(p4, size=p1.size()[2:], mode='bilinear')
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, x, vertices0, A, D):
        """输入图像, 关键点template的坐标, 邻接矩阵, 度矩阵"""

        """骨干网络提取的特征"""
        # 四个特征图都是256 channels
        p2, p3, p4, p5 = self.backbone(x)
        # 特征融合为 1024 通道
        feat_map = self.upsample_cat(p2, p3, p4, p5)

        """第一个图卷积操作：全局变换层"""
        features = self.graph_sample1(feat_map, vertices0)
        latent_features = torch.cat([features, vertices0], dim=2)
        latent_features = self.up_f2f_layer1(latent_features, A, D)  # B, N, 128
        # delta_vertices = self.up_f2v_layer1(latent_features, A, D)
        # vertices1 = vertices0 + delta_vertices
        latent_features = torch.mean(latent_features, dim=1)
        transform_matrix = self.up_f2v_layer1(latent_features)
        vertices1 = prepective_transform(vertices0, transform_matrix)

        """第二个图卷积操作"""
        features = self.graph_sample2(feat_map, vertices1)
        latent_features = torch.cat([features, vertices1], dim=2)
        latent_features = self.up_f2f_layer2(latent_features, A, D)
        delta_vertices = self.up_f2v_layer2(latent_features, A, D)
        vertices2 = vertices1 + delta_vertices

        """第三个图卷积操作"""
        features = self.graph_sample3(feat_map, vertices2)
        latent_features = torch.cat([features, vertices2], dim=2)
        latent_features = self.up_f2f_layer3(latent_features, A, D)
        delta_vertices = self.up_f2v_layer3(latent_features, A, D)
        vertices3 = vertices2 + delta_vertices

        """第四个图卷积操作"""
        features = self.graph_sample4(feat_map, vertices3)
        latent_features = torch.cat([features, vertices3], dim=2)
        latent_features = self.up_f2f_layer4(latent_features, A, D)
        delta_vertices = self.up_f2v_layer4(latent_features, A, D)
        vertices4 = vertices3 + delta_vertices

        """第五个图卷积操作"""
        features = self.graph_sample5(feat_map, vertices4)
        latent_features = torch.cat([features, vertices4], dim=2)
        latent_features = self.up_f2f_layer5(latent_features, A, D)
        delta_vertices = self.up_f2v_layer5(latent_features, A, D)
        vertices5 = vertices4 + delta_vertices

        """第六个图卷积操作"""
        features = self.graph_sample5(feat_map, vertices5)
        latent_features = torch.cat([features, vertices5], dim=2)
        latent_features = self.up_f2f_layer6(latent_features, A, D)
        delta_vertices = self.up_f2v_layer6(latent_features, A, D)
        vertices6 = vertices5 + delta_vertices

        """返回：不同尺度的顶点位置预测"""
        return vertices1, vertices2, vertices3, vertices4, vertices5, vertices6


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
