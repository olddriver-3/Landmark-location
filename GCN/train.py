# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import load_config
from model.base_module import adjacency_matrix
from model.image2shape_ResNet101_NOFPN import Image2Shape
from all_dataset_class import BaseLandmarkDataset
from torch.utils.tensorboard import SummaryWriter
# from dataset import GCNDataset
config = load_config(exp_id=100)


# 训练的环境的配置
torch.cuda.set_device(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('允许设备: ', device)

# GCN数据集
class GCNDataset(BaseLandmarkDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            target_size=(512, 384),
            num_landmarks=8,
            use_template=True,
            heatmap_sigma=3,
            heatmap_radius=50,
            heatmap_downsample=1
        )

annotation_dir_path = '../data/knee/pixel_labels'
train_image_dir_path = '../data/knee/train_images'
valid_image_dir_path = '../data/knee/val_images'
model_save_folder_path = '../results/knee/GCN/model_save'
script_dir = os.path.dirname(os.path.abspath(__file__))
annotation_dir_path= os.path.join(script_dir, annotation_dir_path)
print('标注数据集路径: ', annotation_dir_path)
train_image_dir_path = os.path.join(script_dir, train_image_dir_path)
print('训练数据集路径: ', train_image_dir_path)
valid_image_dir_path = os.path.join(script_dir, valid_image_dir_path)
print('验证数据集路径: ', valid_image_dir_path)
model_save_folder_path = os.path.join(script_dir, model_save_folder_path)
print('模型保存路径: ', model_save_folder_path)
train_dataset = GCNDataset(load_image_dir=train_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=True)
valid_dataset = GCNDataset(load_image_dir=valid_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
print('数据集加载成功!!')



model = Image2Shape(config).to(device)
criterion_global = lambda x, y: torch.mean(F.relu(torch.abs(x - y) - 0.1))
criterion = lambda x, y: torch.mean(torch.abs(x - y))

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

if __name__ == '__main__':
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 5
    start_epoch = 0
    train_max_epoch = 1000
    num_epoch_no_improvement = 0
    epoch_patience = 10
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)
    writer = SummaryWriter(log_dir=os.path.join(model_save_folder_path, 'runs'))
    if not os.path.exists(model_save_folder_path):
        os.makedirs(model_save_folder_path)

    if os.path.exists(os.path.join(model_save_folder_path, 'model.pt')):
        print('加载训练好的权重了~')
        state_dict = torch.load(os.path.join(model_save_folder_path, 'model.pt'), map_location=device,weights_only=False)
        best_loss = state_dict['best_loss']
        start_epoch = state_dict['epoch']
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    for epoch in range(start_epoch, train_max_epoch):
        scheduler.step(epoch)
        model.train()
        for iteration, (name, image, heatmap, landmark, template) in enumerate(train_loader):
            image = image.float().to(device)
            heatmap = heatmap.float().to(device)
            landmark = landmark.float().to(device)
            template = template.float().to(device)
            # 图卷积的连接关系
            A, D = adjacency_matrix(template)
            A = A.float().to(device)
            D = D.float().to(device)
            # 模型预测
            predict_vertices1, \
            predict_vertices2,\
            predict_vertices3,\
            predict_vertices4, \
            predict_vertices5, \
            predict_vertices6 = model(image, template, A, D)
            # 损失函数计算（可以进行深监督的损失函数，也可以不用）
            loss = criterion_global(predict_vertices1, landmark) * 10 + \
                criterion(predict_vertices2, landmark) + \
                criterion(predict_vertices3, landmark) + \
                criterion(predict_vertices4, landmark) + \
                criterion(predict_vertices5, landmark) + \
                criterion(predict_vertices6, landmark)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 5))
            if iteration % 5 == 0:
                print('Iteration: ', iteration, 'Loss: ', round(loss.item(), 5))
        print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, train_max_epoch, np.mean(train_losses)))

        if epoch % 5 != 0:
            continue

        with torch.no_grad():
            model.eval()
            MRE_list = []
            print("validating....")
            for iteration, (name, image, heatmap, landmark, template) in enumerate(valid_loader):
                image = image.float().to(device)
                heatmap = heatmap.float().to(device)
                landmark = landmark.float().to(device)
                template = template.float().to(device)
                # 图卷积的连接关系
                A, D = adjacency_matrix(template)
                A = A.float().to(device)
                D = D.float().to(device)
                # 模型预测
                predict_vertices1, \
                predict_vertices2, \
                predict_vertices3, \
                predict_vertices4, \
                predict_vertices5, \
                predict_vertices6 = model(image, template, A, D)
                # 损失函数计算
                loss = criterion_global(predict_vertices1, landmark) + \
                    criterion(predict_vertices2, landmark) + \
                    criterion(predict_vertices3, landmark) + \
                    criterion(predict_vertices4, landmark) + \
                    criterion(predict_vertices5, landmark) + \
                    criterion(predict_vertices6, landmark)
                valid_losses.append(round(loss.item(), 5))
                # 取出最后一个图卷积的数据操作
                delta_distance = torch.abs(predict_vertices6.cpu() - landmark.cpu()) / 2
                delta_distance = delta_distance.numpy()
                delta_distance = delta_distance * np.asarray([64, 80]) * 3
                distance = np.sqrt(np.sum(np.square(delta_distance), axis=-1))
                MRE_list.append(distance)
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        MRE = np.mean(MRE_list)
        SDR = np.sum(np.asarray(MRE_list) < 2) / np.size(np.array(MRE_list)) * 100
        writer.add_scalar('Train/Epoch_Loss', train_loss, epoch+1)
        writer.add_scalar('Valid/Epoch_Loss', valid_loss, epoch+1)
        writer.add_scalar('Valid/MRE', MRE, epoch+1)
        writer.add_scalar('Valid/SDR', SDR, epoch+1)
        print('Epoch {}, validation MRE is {:.4f}, SDR is {:.4f}'.format(epoch + 1, MRE, SDR))
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1, valid_loss, train_loss))
        train_losses = []
        valid_losses = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'best_loss': best_loss,
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

