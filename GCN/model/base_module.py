# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

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


class LearntNeighbourhoodSampling(nn.Module):
    """可学习的近邻采样层"""
    def __init__(self, features_count, image_h=800, image_w=640):
        super(LearntNeighbourhoodSampling, self).__init__()

        # self.shape = torch.tensor([W, H]).cuda().float()

        # 固定模式的采样
        # self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (config.steps + 1 - step) / (W), 2 ** (config.steps + 1 - step) / (H)]]])[None]
        # self.shift = self.shift.cuda()

        # 通过采样点特征来预测采样的偏移点
        self.shift_delta = nn.Conv1d(features_count, 9*2, kernel_size=1, padding=0)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        # 差值特征的学习
        self.feature_diff_1 = nn.Linear(features_count + 2, features_count)
        self.feature_diff_2 = nn.Linear(features_count, features_count)

        # 聚合周围的特征
        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 3 * 3), padding=0)
        torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)

        # 中心点特征的学习
        self.feature_center_1 = nn.Linear(features_count + 2, features_count)
        self.feature_center_2 = nn.Linear(features_count, features_count)

    def forward(self, image_features, vertices):
        """
        :param image_features: image features, (B, C, H, W)
        :param vertices: graph vertices coordinates, (B, N, 2), specially, (B, N, (c, r)) = (B, N, (x, y))
        :return:
        """
        # 参数解释：B: Batch size;  N: points number, center shape: (B, N, 1, 2)
        B, N, _ = vertices.shape
        center = vertices[:, :, None, :]

        # 通过grid_sample采样：sample features shape: (B, C, N, 1)
        features = F.grid_sample(image_features, center, mode='bilinear', padding_mode='border', align_corners=True)

        # 采样点特征来进行采样点预测：shift_delta features shape: (B, C=9*2, N) -> permute (B, N, C=9*2) -> view (B, N, 9, 2)
        features = features[:, :, :, 0]
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 9, 2)

        # 将第一个采样点的偏置设置为0，保证可以采样到点：setting first shift to zero so it samples at the exact point
        shift_delta[:, :, 0, :] = shift_delta[:, :, 0, :] * 0

        # 位置叠加起来：[B, N, 1, 2] + [B, N, 9, 2] => [B, N, 9, 2]
        neighbourhood = vertices[:, :, None] + shift_delta

        # 通过grid_sample进行特征采样：features shape: (B, C, N, 9)
        features = F.grid_sample(image_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)

        # 将特征和坐标拼接起来组合为新的特征：features + neighbourhood: (B, (C+2), N, 9)
        features = torch.cat([features, neighbourhood.permute(0, 3, 1, 2)], dim=1)

        # 偏置点特征和中心点特征的差：0 is the index of the center cordinate in shifts
        features_diff_from_center = features - features[:, :, :, 0][:, :, :, None]

        # nn.Linear function => permute: (B, 9, N, (C+2)) => feature diff: (B, 9, N, C)
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)

        # permute函数将特征从：(B, 9, N, C) => (B, C, N, 9)
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])

        # sum neighbourhood: (B, C, N, 1) => transpose: (B, N, C)
        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        # 中心点特征学习 wf, transpose: (B, N, (C+2))
        center_features = features[:, :, :, 0].transpose(2, 1)

        center_features = self.feature_center_1(center_features)
        center_features = self.feature_center_2(center_features)

        # add function, (B, N, C)
        features = center_features + features_diff_from_center

        return features


class LearntNeighbourhoodSamplingV2(nn.Module):
    """没有学习，直接均值平均的近邻采样层"""
    def __init__(self, features_count, image_h=800, image_w=640):
        super(LearntNeighbourhoodSamplingV2, self).__init__()

        # self.shape = torch.tensor([W, H]).cuda().float()

        # 固定模式的采样
        # self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (config.steps + 1 - step) / (W), 2 ** (config.steps + 1 - step) / (H)]]])[None]
        # self.shift = self.shift.cuda()

        # 通过采样点特征来预测采样的偏移点
        self.shift_delta = nn.Conv1d(features_count, 9*2, kernel_size=1, padding=0)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

    def forward(self, image_features, vertices):
        """
        :param image_features: image features, (B, C, H, W)
        :param vertices: graph vertices coordinates, (B, N, 2), specially, (B, N, (c, r)) = (B, N, (x, y))
        :return:
        """
        # 参数解释：B: Batch size;  N: points number, center shape: (B, N, 1, 2)
        B, N, _ = vertices.shape
        center = vertices[:, :, None, :]

        # 通过grid_sample采样：sample features shape: (B, C, N, 1)
        features = F.grid_sample(image_features, center, mode='bilinear', padding_mode='border', align_corners=True)

        # 采样点特征来进行采样点预测：shift_delta features shape: (B, C=9*2, N) -> permute (B, N, C=9*2) -> view (B, N, 9, 2)
        features = features[:, :, :, 0]
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 9, 2)

        # 将第一个采样点的偏置设置为0，保证可以采样到点：setting first shift to zero so it samples at the exact point
        shift_delta[:, :, 0, :] = shift_delta[:, :, 0, :] * 0

        # 位置叠加起来：[B, N, 1, 2] + [B, N, 9, 2] => [B, N, 9, 2]
        neighbourhood = vertices[:, :, None] + shift_delta

        # 通过grid_sample进行特征采样：features shape: (B, C, N, 9)
        features = F.grid_sample(image_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)

        # 进行特征聚合
        features = torch.mean(features, dim=-1)
        features = features.transpose(2, 1)

        return features


class PatternSampling(nn.Module):
    """没有学习，直接均值平均的近邻采样层"""
    def __init__(self, features_count, image_h=800, image_w=640):
        super(PatternSampling, self).__init__()

        # self.shape = torch.tensor([W, H]).cuda().float()

        # 固定模式的采样
        self.shift = torch.tensor([[-1, 1],
                                   [0, 1],
                                   [1, 1],
                                   [-1, 0],
                                   [0, 0],
                                   [1, 0],
                                   [-1, -1],
                                   [0, -1],
                                   [1, -1]])[None].float()[None]

        # 通过采样点特征来预测采样的偏移点
        self.shift_delta = nn.Conv1d(features_count, 9*2, kernel_size=1, padding=0)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

    def forward(self, image_features, vertices):
        """
        :param image_features: image features, (B, C, H, W)
        :param vertices: graph vertices coordinates, (B, N, 2), specially, (B, N, (c, r)) = (B, N, (x, y))
        :return:
        """
        # 转换类型
        shift_delta = self.shift.to(image_features.device)

        # 参数解释：B: Batch size;  N: points number, center shape: (B, N, 1, 2)
        B, N, _ = vertices.shape
        center = vertices[:, :, None, :]

        # 通过grid_sample采样：sample features shape: (B, C, N, 1)
        features = F.grid_sample(image_features, center, mode='bilinear', padding_mode='border', align_corners=True)

        # 采样点特征来进行采样点预测：shift_delta features shape: (B, C=9*2, N) -> permute (B, N, C=9*2) -> view (B, N, 9, 2)
        # features = features[:, :, :, 0]
        # shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 9, 2)

        # 将第一个采样点的偏置设置为0，保证可以采样到点：setting first shift to zero so it samples at the exact point
        shift_delta[:, :, 0, :] = shift_delta[:, :, 0, :] * 0
        shift_delta = shift_delta.repeat(B, 1, 1, 1)
        shift_delta = shift_delta.expand(B, N, 9, 2)

        # 位置叠加起来：[B, N, 1, 2] + [B, N, 9, 2] => [B, N, 9, 2]
        neighbourhood = vertices[:, :, None] + shift_delta

        # 通过grid_sample进行特征采样：features shape: (B, C, N, 9)
        features = F.grid_sample(image_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)

        # 进行特征聚合
        features = torch.mean(features, dim=-1)
        features = features.transpose(2, 1)

        return features


class LearntNeighbourhoodSamplingV3(nn.Module):
    """点的采样层"""
    def __init__(self, features_count, image_h=800, image_w=640):
        super(LearntNeighbourhoodSamplingV3, self).__init__()

        # self.shape = torch.tensor([W, H]).cuda().float()

        # 固定模式的采样
        # self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (config.steps + 1 - step) / (W), 2 ** (config.steps + 1 - step) / (H)]]])[None]
        # self.shift = self.shift.cuda()

        # # 通过采样点特征来预测采样的偏移点
        # self.shift_delta = nn.Conv1d(features_count, 9*2, kernel_size=1, padding=0)
        # self.shift_delta.weight.data.fill_(0.0)
        # self.shift_delta.bias.data.fill_(0.0)

    def forward(self, image_features, vertices):
        """
        :param image_features: image features, (B, C, H, W)
        :param vertices: graph vertices coordinates, (B, N, 2), specially, (B, N, (c, r)) = (B, N, (x, y))
        :return:
        """
        # 参数解释：B: Batch size;  N: points number, center shape: (B, N, 1, 2)
        B, N, _ = vertices.shape
        center = vertices[:, :, None, :]

        # 通过grid_sample采样：sample features shape: (B, C, N, 1)
        features = F.grid_sample(image_features, center, mode='bilinear', padding_mode='border', align_corners=True)

        # 采样点特征来进行采样点预测：shift_delta features shape: (B, C=9*2, N) -> permute (B, N, C=9*2) -> view (B, N, 9, 2)
        features = features[:, :, :, 0]
        features = features.transpose(2, 1)

        return features


class PointSampling(nn.Module):
    """点的采样层"""
    def __init__(self, features_count, image_h=800, image_w=640):
        super(PointSampling, self).__init__()

        # self.shape = torch.tensor([W, H]).cuda().float()

        # 固定模式的采样
        # self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (config.steps + 1 - step) / (W), 2 ** (config.steps + 1 - step) / (H)]]])[None]
        # self.shift = self.shift.cuda()

        # # 通过采样点特征来预测采样的偏移点
        # self.shift_delta = nn.Conv1d(features_count, 9*2, kernel_size=1, padding=0)
        # self.shift_delta.weight.data.fill_(0.0)
        # self.shift_delta.bias.data.fill_(0.0)

    def forward(self, image_features, vertices):
        """
        :param image_features: image features, (B, C, H, W)
        :param vertices: graph vertices coordinates, (B, N, 2), specially, (B, N, (c, r)) = (B, N, (x, y))
        :return:
        """
        # 参数解释：B: Batch size;  N: points number, center shape: (B, N, 1, 2)
        B, N, _ = vertices.shape
        center = vertices[:, :, None, :]

        # 通过grid_sample采样：sample features shape: (B, C, N, 1)
        features = F.grid_sample(image_features, center, mode='bilinear', padding_mode='border', align_corners=True)

        # 采样点特征来进行采样点预测：shift_delta features shape: (B, C=9*2, N) -> permute (B, N, C=9*2) -> view (B, N, 9, 2)
        features = features[:, :, :, 0]
        features = features.transpose(2, 1)

        return features


def adjacency_matrix(vertices, faces=None):
    """
    :param vertices: graph vertices coordinates, (B, N, 2), specially, (B, N, (c, r)) = (B, N, (x, y))
    :param faces: None
    :return:
    """
    # batch-size, N个关键点，D维度的坐标
    B, N, D = vertices.shape

    # 默认初始化为全连接
    A = torch.ones(1, N, N)
    D = torch.diag(1 / torch.squeeze(torch.sum(A, dim=1)))[None]

    A = A.repeat(B, 1, 1)
    D = D.repeat(B, 1, 1)

    return A, D


class GraphConvLayer(nn.Module):
    """图卷积的定义的卷积学习层"""
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, batch_norm=False):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.neighbours_fc = nn.Linear(in_features, out_features)

        # self.fc.weight.data.zero_()
        # self.fc.bias.data.zero_()
        # self.neighbours_fc.weight.data.zero_()
        # self.neighbours_fc.bias.data.zero_()

    def forward(self, input, A, Dinv):
        """
        :param input: (B, N, C)
        :param A:
        :param Dinv:
        :return:
        """
        # 行归一化，对连接矩阵A进行归一化
        coeff = torch.bmm(Dinv, A)

        y = self.fc(input)
        y_neightbours = torch.bmm(coeff, input)
        y_neightbours = self.neighbours_fc(y_neightbours)

        y = y + y_neightbours

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features is not None)


class Features2Features(nn.Module):
    """特征到特征的学习，用于特征的升维或者降维"""
    def __init__(self, in_features, out_features, hidden_layer_count=1, graph_conv=GraphConvLayer):
        super(Features2Features, self).__init__()

        self.gconv_first = graph_conv(in_features, out_features)
        gconv_hidden = []
        for i in range(hidden_layer_count):
            gconv_hidden += [graph_conv(out_features, out_features)]
        self.gconv_hidden = nn.Sequential(*gconv_hidden)
        self.gconv_last = graph_conv(out_features, out_features)

    def forward(self, features, adjacency_matrix, degree_matrix, ):
        features = F.relu(self.gconv_first(features, adjacency_matrix, degree_matrix))
        for gconv_hidden in self.gconv_hidden:
            features = F.relu(gconv_hidden(features, adjacency_matrix, degree_matrix))
        return self.gconv_last(features, adjacency_matrix, degree_matrix,)


class Feature2VertexLayer(nn.Module):
    """将在特征解码为节点坐标"""
    def __init__(self, in_features, hidden_layer_count, batch_norm=False):
        super(Feature2VertexLayer, self).__init__()
        self.gconv = []
        for i in range(hidden_layer_count, 1, -1):
            self.gconv += [GraphConvLayer(i * in_features // hidden_layer_count, (i-1) * in_features // hidden_layer_count, batch_norm)]
        self.gconv_layer = nn.Sequential(*self.gconv)
        self.gconv_last = GraphConvLayer(in_features // hidden_layer_count, 2, batch_norm)

    def forward(self, features, adjacency_matrix, degree_matrix):
        for gconv_hidden in self.gconv:
            features = F.relu(gconv_hidden(features, adjacency_matrix, degree_matrix))
        return self.gconv_last(features, adjacency_matrix, degree_matrix)


class Feature2VertexLayerV2(nn.Module):
    """将在特征解码为节点坐标"""
    def __init__(self, in_features, hidden_layer_count, batch_norm=False):
        super(Feature2VertexLayerV2, self).__init__()
        self.gconv = []
        for i in range(hidden_layer_count, 1, -1):
            self.gconv += [GraphConvLayer(i * in_features // hidden_layer_count, (i-1) * in_features // hidden_layer_count, batch_norm)]
        self.gconv_layer = nn.Sequential(*self.gconv)
        self.gconv_last = GraphConvLayer(in_features // hidden_layer_count, 2, batch_norm)

    def forward(self, features, adjacency_matrix, degree_matrix):
        for gconv_hidden in self.gconv:
            features = F.relu(gconv_hidden(features, adjacency_matrix, degree_matrix))
        return self.gconv_last(features, adjacency_matrix, degree_matrix)


if __name__ == '__main__':
    x = torch.randint(0, 10, (1, 6, 7, 7)).float()
    print(x)
    vertices = torch.tensor([[[0, 0],
                              [6, 0],
                              [4, 5]]]).float() / (7 - 1) * 2 - 1
    print(vertices)
    print(vertices.shape)

    sample_layer = LearntNeighbourhoodSampling(6)

    sample_layer.forward(x, vertices)

    A, D = adjacency_matrix(vertices)
    print(A)
    print(D)

