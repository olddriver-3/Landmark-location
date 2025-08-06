import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.transform as sktf
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from all_dataset_class import BaseLandmarkDataset
from model import load_model

class LandmarkDetector:
    def __init__(self, num_landmarks=6, target_size=(512, 512), heatmap_downsample=4):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_landmarks = num_landmarks
        self.target_size = target_size
        self.heatmap_downsample = heatmap_downsample
        self.heatmap_size = (target_size[0]//heatmap_downsample, target_size[1]//heatmap_downsample)
        
        # 更新配置文件
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, 'config.yaml')
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        config['DATASET']['NUM_JOINTS_HALF_BODY'] = num_landmarks
        config['MODEL']['NUM_JOINTS'] = num_landmarks
        with open(self.config_path, 'w') as file:
            yaml.dump(config, file, sort_keys=False, default_flow_style=False)
        
        # 初始化模型
        self.model = load_model().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.scheduler = None

    def create_dataset(self, image_dir, annotation_dir=None, do_augmentation=False):
        """创建数据集对象"""
        return BaseLandmarkDataset(
            load_image_dir=image_dir,
            load_annotation_dir=annotation_dir if annotation_dir else image_dir,
            target_size=self.target_size,
            num_landmarks=self.num_landmarks,
            use_template=False,
            heatmap_sigma=3,
            heatmap_radius=50,
            heatmap_downsample=self.heatmap_downsample,
            do_augmentation=do_augmentation
        )

    def train(self, train_image_dir, annotation_dir, valid_image_dir, model_save_folder,
              batch_size=8, max_epochs=1000, patience=10, lr=2e-4):
        """训练模型"""
        # 准备数据
        train_dataset = self.create_dataset(train_image_dir, annotation_dir, True)
        valid_dataset = self.create_dataset(valid_image_dir, annotation_dir, False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)
        
        # 训练循环
        best_loss = float('inf')
        num_epoch_no_improvement = 0
        os.makedirs(model_save_folder, exist_ok=True)
        
        for epoch in range(max_epochs):
            self.scheduler.step(epoch)
            self.model.train()
            train_losses = []
            
            for image_file, image, heatmap in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                image = image.float().to(self.device)
                heatmap = heatmap.float().to(self.device)
                pred_heatmap = self.model(image)
                loss = self.criterion(pred_heatmap, heatmap)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # 验证
            valid_loss, mre, sdr = self._validate(valid_loader)
            train_loss = np.mean(train_losses)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Valid Loss={valid_loss:.6f}, MRE={mre:.4f}, SDR={sdr:.2f}%")
            
            # 保存最佳模型
            if valid_loss < best_loss:
                best_loss = valid_loss
                num_epoch_no_improvement = 0
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, os.path.join(model_save_folder, "model.pt"))
                print(f"Saved best model at epoch {epoch+1}")
            else:
                num_epoch_no_improvement += 1
                if num_epoch_no_improvement >= patience:
                    print("Early stopping")
                    break

    def _validate(self, loader):
        """验证辅助函数"""
        self.model.eval()
        losses = []
        mre_list = []
        
        with torch.no_grad():
            for image_file, image, heatmap in loader:
                image = image.float().to(self.device)
                heatmap = heatmap.float().to(self.device)
                pred_heatmap = self.model(image)
                loss = self.criterion(pred_heatmap, heatmap)
                losses.append(loss.item())
                
                # 计算MRE
                pred_coords = []
                true_coords = []
                for i in range(pred_heatmap.shape[1]):
                    pred_hm = pred_heatmap[0, i].cpu().numpy()
                    true_hm = heatmap[0, i].cpu().numpy()
                    pred_idx = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
                    true_idx = np.unravel_index(np.argmax(true_hm), true_hm.shape)
                    pred_coords.append(pred_idx)
                    true_coords.append(true_idx)
                
                pred_coords = np.array(pred_coords)
                true_coords = np.array(true_coords)
                if pred_coords.shape == true_coords.shape:
                    distances = np.linalg.norm(pred_coords - true_coords, axis=1)
                    mre_list.extend(distances)
        
        valid_loss = np.mean(losses)
        mre = np.mean(mre_list) if mre_list else 0
        sdr = (np.sum(np.array(mre_list) < 2) / len(mre_list) * 100) if mre_list else 0
        
        return valid_loss, mre, sdr

    def _process_image(self, image_path, annotation_path=None):
        """处理单个图像的核心逻辑"""
        # 读取原始图像
        orig_img = skio.imread(image_path)
        orig_h, orig_w = orig_img.shape[:2]
        
        # 预处理图像
        image = sktf.resize(orig_img, self.target_size)
        image = np.transpose(image, (2, 0, 1))
        input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(self.device)
        
        # 预测
        with torch.no_grad():
            pred_heatmap = self.model(input_tensor).cpu().numpy()[0]
        
        # 提取预测点
        pred_coords = []
        for i in range(self.num_landmarks):
            hm = pred_heatmap[i]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            pred_coords.append([x, y])
        pred_coords = np.array(pred_coords)
        
        # 映射回原始尺寸
        scale_x = orig_w / self.heatmap_size[1]
        scale_y = orig_h / self.heatmap_size[0]
        pred_coords_orig = np.stack([pred_coords[:, 0] * scale_x, pred_coords[:, 1] * scale_y], axis=1)
        
        # 提取金标准点（如果有）
        gt_coords_orig = None
        if annotation_path and os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                gt_coords = []
                for line in f.readlines()[:self.num_landmarks]:
                    coords = [int(x) for x in line.strip().split(',')]
                    gt_coords.append(coords)
                gt_coords = np.array(gt_coords)
            
            # 映射到热图坐标
            scale_x_hm = self.heatmap_size[1] / orig_w
            scale_y_hm = self.heatmap_size[0] / orig_h
            gt_coords_hm = np.stack([gt_coords[:, 0] * scale_x_hm, gt_coords[:, 1] * scale_y_hm], axis=1)
            
            # 再映射回原始坐标（用于可视化）
            gt_coords_orig = np.stack([gt_coords_hm[:, 0] * scale_x, gt_coords_hm[:, 1] * scale_y], axis=1)
        
        return orig_img, pred_coords_orig, gt_coords_orig

    def valid_all(self, image_dir, annotation_dir, result_dir):
        """带金标准的批量验证"""
        dataset = self.create_dataset(image_dir, annotation_dir)
        os.makedirs(result_dir, exist_ok=True)
        mre_list = []
        
        for filename, _, _ in tqdm(dataset, desc="Validating"):
            img_path = os.path.join(image_dir, filename)
            ann_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + '.txt')
            
            orig_img, pred_coords, gt_coords = self._process_image(img_path, ann_path)
            
            # 计算MRE（在热图空间）
            scale_x = self.target_size[1] / self.heatmap_size[1]
            scale_y = self.target_size[0] / self.heatmap_size[0]
            pred_coords_input = pred_coords.copy()
            pred_coords_input[:, 0] /= scale_x
            pred_coords_input[:, 1] /= scale_y
            
            gt_coords_input = gt_coords.copy()
            gt_coords_input[:, 0] /= scale_x
            gt_coords_input[:, 1] /= scale_y
            
            distances = np.linalg.norm(pred_coords_input - gt_coords_input, axis=1)
            mre_list.extend(distances)
            
            # 可视化并保存
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.imshow(orig_img)
            ax.scatter(gt_coords[:, 0], gt_coords[:, 1], c='g', s=30, label='GT', marker='o')
            ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='r', s=30, label='Pred', marker='x')
            ax.legend()
            ax.set_title(filename)
            ax.axis('off')
            save_path = os.path.join(result_dir, filename.replace('.png', '_result.png'))
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        
        # 计算指标
        mre = np.mean(mre_list)
        sdr = np.sum(np.array(mre_list) < 2) / len(mre_list) * 100
        print(f"Validation MRE: {mre:.4f}, SDR: {sdr:.2f}%")
        return mre, sdr

    def test_all(self, image_dir, result_dir):
        """无金标准的批量测试"""
        os.makedirs(result_dir, exist_ok=True)
        filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.bmp',".JPG"))]
        
        for filename in tqdm(filenames, desc="Testing"):
            img_path = os.path.join(image_dir, filename)
            orig_img, pred_coords, _ = self._process_image(img_path)
            
            # 可视化并保存
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.imshow(orig_img)
            ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='r', s=30, label='Pred', marker='x')
            ax.legend()
            ax.set_title(filename)
            ax.axis('off')
            save_path = os.path.join(result_dir, filename.replace('.png', '_result.png'))
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    
    def single_valid(self, image_name, image_dir, annotation_dir):
        """带金标准的单图像验证（显示不保存）"""
        img_path = os.path.join(image_dir, image_name)
        ann_path = os.path.join(annotation_dir, os.path.splitext(image_name)[0] + '.txt')
        
        orig_img, pred_coords, gt_coords = self._process_image(img_path, ann_path)
        
        # 可视化
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(orig_img)
        ax.scatter(gt_coords[:, 0], gt_coords[:, 1], c='g', s=30, label='GT', marker='o')
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='r', s=30, label='Pred', marker='x')
        ax.legend()
        ax.set_title(image_name)
        ax.axis('off')
        plt.show()
    
    def single_test(self, image_name, image_dir):
        """无金标准的单图像测试（显示不保存）"""
        img_path = os.path.join(image_dir, image_name)
        orig_img, pred_coords, _ = self._process_image(img_path)
        
        # 可视化
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(orig_img)
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], c='r', s=30, label='Pred', marker='x')
        ax.legend()
        ax.set_title(image_name)
        ax.axis('off')
        plt.show()
    
    def load_model(self, model_path):
        """加载预训练模型"""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.eval()
        print(f"Loaded model from {model_path}")

# 使用示例
if __name__ == "__main__":
    detector = LandmarkDetector(num_landmarks=8)
    
    # 训练
    # detector.train(
    #     train_image_dir='../data/chest_6/train_images',
    #     annotation_dir='../data/chest_6/pixel_labels',
    #     valid_image_dir='../data/chest_6/val_images',
    #     model_save_folder='../results/model_save'
    # )
    
    # 加载预训练模型
    detector.load_model('../final_results/knee/HRNet/model_save/model.pt')
    
    # # 验证
    # detector.valid_all(
    #     image_dir='./data/chest_6/val_images',
    #     annotation_dir='./data/chest_6/pixel_labels',
    #     result_dir='./results/chest_6/HRNet/validation_results'
    # )
    
    #测试
    detector.test_all(
        image_dir='../data/clinic_knee/test_images',
        result_dir='../results/clinic_knee/HRNet/test_results_visualized'
    )
    
    # 单图像验证
    # detector.single_valid(
    #     image_name='example.png',
    #     image_dir='../data/chest_6/val_images',
    #     annotation_dir='../data/chest_6/pixel_labels'
    # )
    
    # 单图像测试
    # detector.single_test(
    #     image_name='example.png',
    #     image_dir='../data/chest_6/test_images'
    # )
