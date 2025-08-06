# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
本脚本用于检查 HRNetDataset 的输出 heatmap。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from train_HRNet import HRNetDataset, annotation_dir_path, train_image_dir_path

# 实例化数据集
# 只取一条数据做可视化
if __name__ == '__main__':
    dataset = HRNetDataset(load_image_dir=train_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
    print(f"数据集样本数: {len(dataset)}")
    idx = 0  # 可修改为任意索引
    sample = dataset[idx]
    if len(sample) == 3:
        filename, image, heatmap = sample
    else:
        filename, image, heatmap, *_ = sample
    print(f"样本文件名: {filename}")
    print(f"image shape: {image.shape}")
    print(f"heatmap shape: {heatmap.shape}")
    print(np.average(image))
    # 可视化所有通道的 heatmap
    num_landmarks = heatmap.shape[0]
    fig, axes = plt.subplots(1, num_landmarks, figsize=(4*num_landmarks, 4))
    if num_landmarks == 1:
        axes = [axes]
    for i in range(num_landmarks):
        axes[i].imshow(heatmap[i], cmap='jet')
        axes[i].set_title(f'Landmark {i+1}')
        axes[i].axis('off')
    plt.suptitle(f'Heatmaps for {filename}')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 10))
    plt.imshow(image[0])
    plt.title(f'Image for {filename}')
    plt.axis('off')
    plt.colorbar(axes[0].images[0], ax=axes, orientation='horizontal', fraction=0.02, pad=0.04)
    plt.show()
