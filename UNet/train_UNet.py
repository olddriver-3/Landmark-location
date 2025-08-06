


import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from all_dataset_class import BaseLandmarkDataset
from UNet import UNet
import yaml

# 训练环境配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('允许设备: ', device)

# 标记点数量（可根据数据集调整）
num_landmarks = 6

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
config['num_classes'] = num_landmarks
with open(config_path, 'w', encoding='utf-8') as file:
    yaml.dump(config, file, sort_keys=False, default_flow_style=False)
print("YAML文件已更新")

# UNet数据集定义
class UNetDataset(BaseLandmarkDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            target_size=(512, 512),
            num_landmarks=num_landmarks,
            use_template=False,
            heatmap_sigma=3,
            heatmap_radius=50,
            heatmap_downsample=1
        )

annotation_dir_path = '../data/chest_6/pixel_labels'
train_image_dir_path = '../data/chest_6/train_images'
valid_image_dir_path = '../data/chest_6/val_images'
model_save_folder_path = '../results/chest_6/UNet/model_save'
annotation_dir_path = os.path.join(script_dir, annotation_dir_path)
train_image_dir_path = os.path.join(script_dir, train_image_dir_path)
valid_image_dir_path = os.path.join(script_dir, valid_image_dir_path)
model_save_folder_path = os.path.join(script_dir, model_save_folder_path)
print('标注数据集路径: ', annotation_dir_path)
print('训练数据集路径: ', train_image_dir_path)
print('验证数据集路径: ', valid_image_dir_path)
print('模型保存路径: ', model_save_folder_path)

train_dataset = UNetDataset(load_image_dir=train_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=True)
valid_dataset = UNetDataset(load_image_dir=valid_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
print('数据集加载成功!!')

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 8), shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)

    # 构造Config对象以适配UNet
    class ConfigObj:
        def __init__(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)

    model = UNet(ConfigObj(config)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 2e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 5
    start_epoch = 0
    train_max_epoch = config.get('num_epochs', 1000)
    num_epoch_no_improvement = 0
    epoch_patience = 10
    if not os.path.exists(model_save_folder_path):
        os.makedirs(model_save_folder_path)

    model_pt_path = os.path.join(model_save_folder_path, 'model.pt')
    if os.path.exists(model_pt_path):
        print('加载训练好的权重了~')
        state_dict = torch.load(model_pt_path, map_location=device)
        start_epoch = state_dict.get('epoch', 0)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    for epoch in range(start_epoch, train_max_epoch):
        scheduler.step(epoch)
        model.train()
        for iteration, (image_file, image, heatmap) in enumerate(train_loader):
            image = torch.tensor(image, dtype=torch.float32).to(device)
            heatmap = torch.tensor(heatmap, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            predict_heatmap = model(image)
            loss = criterion(predict_heatmap, heatmap)
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
            for iteration, (image_file, image, heatmap) in enumerate(valid_loader):
                image = torch.tensor(image, dtype=torch.float32).to(device)
                heatmap = torch.tensor(heatmap, dtype=torch.float32).to(device)
                predict_heatmap = model(image)
                loss = criterion(predict_heatmap, heatmap)
                valid_losses.append(round(loss.item(), 5))
                # 计算MRE（Mean Radial Error）
                pred_coords = []
                true_coords = []
                for i in range(predict_heatmap.shape[1]):
                    pred_hm = predict_heatmap[0, i].cpu().numpy()
                    true_hm = heatmap[0, i].cpu().numpy()
                    pred_idx = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
                    true_idx = np.unravel_index(np.argmax(true_hm), true_hm.shape)
                    pred_coords.append(pred_idx)
                    true_coords.append(true_idx)
                pred_coords = np.array(pred_coords)
                true_coords = np.array(true_coords)
                if pred_coords.shape == true_coords.shape:
                    distances = np.linalg.norm(pred_coords - true_coords, axis=1)
                    MRE_list.extend(distances)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        if len(MRE_list) > 0:
            MRE = np.mean(MRE_list)
            SDR = np.sum(np.asarray(MRE_list) < 2) / np.size(np.array(MRE_list)) * 100
        else:
            MRE = 0
            SDR = 0
        print('Epoch {}, validation MRE is {:.6f}, SDR is {:.2f}%'.format(epoch + 1, MRE, SDR))
        print("Epoch {}, validation loss is {:.6f}, training loss is {:.6f}".format(epoch+1, valid_loss, train_loss))
        train_losses = []
        valid_losses = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.6f} to {:.6f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_pt_path)
            print("Saving model ", model_pt_path)
        else:
            print("Validation loss does not decrease from {:.6f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == epoch_patience:
            print("Early Stopping")
            break
