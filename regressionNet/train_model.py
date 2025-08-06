# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from sympy import flatten
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from all_dataset_class import BaseLandmarkDataset

from model import load_model
n_landmarks=8
# 训练的环境的配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('允许设备: ', device)
class RegressionNetDataset(BaseLandmarkDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            target_size=(512, 512),
            num_landmarks=n_landmarks,
            use_template=True,
            heatmap_sigma=3,
            heatmap_radius=50,
            heatmap_downsample=1
        )
# 数据集的配置
annotation_dir_path = '../data/knee/pixel_labels'
train_image_dir_path = '../data/knee/train_images'
valid_image_dir_path = '../data/knee/val_images'
model_save_folder_path = '../results/knee/RegressionNet/model_save'
script_dir = os.path.dirname(os.path.abspath(__file__))
annotation_dir_path= os.path.join(script_dir, annotation_dir_path)
train_image_dir_path = os.path.join(script_dir, train_image_dir_path)
valid_image_dir_path = os.path.join(script_dir, valid_image_dir_path)
model_save_folder_path = os.path.join(script_dir, model_save_folder_path)
train_dataset = RegressionNetDataset(load_image_dir=train_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=True)
valid_dataset = RegressionNetDataset(load_image_dir=valid_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
print('数据集加载成功!!')

model = load_model(model='resnet50', n_landmarks=n_landmarks)
model = model.to(device)
criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
if __name__=='__main__':
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    start_epoch = 0
    train_max_epoch = 1000
    num_epoch_no_improvement = 0
    epoch_patience = 30
    if not os.path.exists(model_save_folder_path):
        os.makedirs(model_save_folder_path)
        
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)

    if os.path.exists(os.path.join(model_save_folder_path, 'model.pt')):
        print('加载训练好的权重了~')
        state_dict = torch.load(os.path.join(model_save_folder_path, 'model.pt'), map_location=device)
        start_epoch = state_dict['epoch']
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    for epoch in range(start_epoch, train_max_epoch):
        scheduler.step(epoch)
        model.train()

        for iteration, (filename, image, heatmap, landmark, normalized_template) in enumerate(train_loader):
            image = image.float().to(device)
            landmark = landmark.float().to(device)
            # 模型预测
            predict_landmark = model(image)
            # 损失函数计算
            landmark = landmark.view(landmark.size(0), -1)
            loss = criterion(predict_landmark, landmark)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 5))
            if iteration % 5 == 0:
                print('Iteration: ', iteration, 'Loss: ', round(loss.item(), 5))
        print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, train_max_epoch, np.mean(train_losses)))

        with torch.no_grad():
            model.eval()
            MRE_list = []
            print("validating....")
            for iteration, (filename, image, _, landmark, _) in enumerate(valid_loader):
                image = image.float().to(device)
                landmark = landmark.float().to(device)
                # 模型预测
                predict_landmark = model(image)
                # 损失函数计算
                flattened_landmark = landmark.view(landmark.size(0), -1)
                loss = criterion(predict_landmark, flattened_landmark)
                valid_losses.append(round(loss.item(), 5))
                predict_landmark = predict_landmark.view(predict_landmark.size(0), n_landmarks, 2)
                predict_landmark = predict_landmark.cpu().numpy()
                landmark = landmark.cpu().numpy()
                # 合并batch和landmark维度，计算所有点的欧氏距离
                predict_landmark_flat = predict_landmark.reshape(-1, 2)
                landmark_flat = landmark.reshape(-1, 2)
                mre = np.linalg.norm(predict_landmark_flat - landmark_flat, axis=1)
                MRE_list.extend(mre)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        MRE = np.mean(MRE_list)
        SDR = np.sum(np.asarray(MRE_list) < 2) / np.size(np.array(MRE_list)) * 100
        print('Epoch {}, validation MRE is {:.4f}, SDR is {:.4f}'.format(epoch + 1, MRE, SDR))
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1, valid_loss, train_loss))
        train_losses = []
        valid_losses = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(model_save_folder_path, "model.pt"))
            print("Saving model ", os.path.join(model_save_folder_path, "model.pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == epoch_patience:
            print("Early Stopping")
            break

