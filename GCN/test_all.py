from matplotlib.dates import WE
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from config import load_config
from model.base_module import adjacency_matrix
from model.image2shape_ResNet101_NOFPN import Image2Shape
import skimage.io as sk_io
from all_dataset_class import BaseLandmarkDataset
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

def denormalize_landmarks(landmarks, target_shape):
    # 将归一化坐标还原到指定尺寸（如网络输入尺寸）
    w, h = target_shape[1], target_shape[0]
    x = (landmarks[:, 0] + 1) / 2 * (w - 1)
    y = (landmarks[:, 1] + 1) / 2 * (h - 1)
    return np.stack([x, y], axis=1)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = load_config(exp_id=100)
    annotation_dir_path = '../data/clinic_knee/pixel_labels'
    valid_image_dir_path = '../data/clinic_knee/test_images'
    model_save_folder_path = '../final_results/knee/GCN/model_save'
    result_dir = '../results/clinic_knee/GCN/test_results_visualized'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir_path= os.path.join(script_dir, annotation_dir_path)
    valid_image_dir_path = os.path.join(script_dir, valid_image_dir_path)
    model_save_folder_path = os.path.join(script_dir, model_save_folder_path)
    result_dir = os.path.join(script_dir, result_dir)
    print('模型保存路径: ', model_save_folder_path)
    test_dataset = GCNDataset(load_image_dir=valid_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
    model = Image2Shape(config).to(device)
    mode = 'test'  # 'test' or 'valid'
    model_path = os.path.join(model_save_folder_path, 'model.pt')
    if not os.path.exists(model_path):
        print('模型权重不存在，请先训练模型。')
        return
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    MRE_list = []
    print("Testing and visualizing results...")
    maxdistance=0
    maxdistance_idx=0
    # 遍历测试集
    for idx in range(len(test_dataset)):
        name, image, heatmap, landmark, template = test_dataset[idx]
        # image: (3, 512, 384) 网络输入尺寸
        # 读取原始图像（未resize）
        # 支持 bmp 和 png
        img_path = os.path.join(valid_image_dir_path, name)
        if not os.path.exists(img_path):
            print(f'原图 {img_path} 不存在，跳过')
            continue
        orig_img = sk_io.imread(img_path)
        orig_h, orig_w = orig_img.shape[:2]

        # 网络输入尺寸
        net_h, net_w = 512, 384

        # 预测
        image_tensor = torch.tensor(image).unsqueeze(0).float().to(device)
        template_tensor = torch.tensor(template).unsqueeze(0).float().to(device)
        A, D = adjacency_matrix(template_tensor)
        A = A.float().to(device)
        D = D.float().to(device)
        with torch.no_grad():
            preds = model(image_tensor, template_tensor, A, D)
            pred_landmarks = preds[-1].squeeze(0).cpu().numpy()

        # 先还原到网络输入尺寸
        pred_points_net = denormalize_landmarks(pred_landmarks, (net_h, net_w))

        # 再映射到原图尺寸
        scale_x = orig_w / net_w
        scale_y = orig_h / net_h
        pred_points_orig = np.zeros_like(pred_points_net)
        pred_points_orig[:, 0] = pred_points_net[:, 0] * scale_x
        pred_points_orig[:, 1] = pred_points_net[:, 1] * scale_y

        # 画图
        fig, ax = plt.subplots(figsize=(8, 10))

        # 计算欧氏距离
        if mode=='valid':
            gt_points_net = denormalize_landmarks(landmark, (net_h, net_w))
            gt_points_orig = np.zeros_like(gt_points_net)
            gt_points_orig[:, 0] = gt_points_net[:, 0] * scale_x
            gt_points_orig[:, 1] = gt_points_net[:, 1] * scale_y
            ax.scatter(gt_points_orig[:, 0], gt_points_orig[:, 1], c='g', s=30, label='GT', marker='o')
            # 添加连接线
            for i in range(len(gt_points_orig)):
                ax.plot([gt_points_orig[i, 0], pred_points_orig[i, 0]], 
                    [gt_points_orig[i, 1], pred_points_orig[i, 1]], 
                    c='b', linestyle='--', alpha=0.5)
            if pred_points_net.shape == gt_points_net.shape:
                distances = np.linalg.norm(pred_points_net - gt_points_net, axis=1)
                MRE_list.extend(distances)
                if  sum(distances)>maxdistance:
                    maxdistance = sum(distances)
                    maxdistance_idx = idx
        elif mode=='test':
            #保存预测坐标

            np.savetxt(os.path.join(result_dir, name.replace('.png', '_pred.txt').replace('.bmp', '_pred.txt')), pred_points_orig)

        save_path = os.path.join(result_dir, name.replace('.png', '_result.png').replace('.bmp', '_result.png'))

        ax.imshow(orig_img)
        ax.scatter(pred_points_orig[:, 0], pred_points_orig[:, 1], c='r', s=30, label='Pred', marker='x')
        # 添加连接线
        ax.legend()
        ax.set_title(name)
        ax.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f'Saved: {save_path}')
            # 仅保存预测结果

        print(f"Processed {idx+1}/{len(test_dataset)}: {name}")
    if mode=='valid':
        name, image, heatmap, landmark, template = test_dataset[maxdistance_idx]
        print(f"最大距离的图像: {name}, 累计距离: {maxdistance}")
        # 计算平均MRE和成功检测率
        MRE_list = np.array(MRE_list)
        if len(MRE_list) > 0:
            MRE = np.mean(MRE_list)
            SDR = np.sum(np.asarray(MRE_list) < 2) / np.size(np.array(MRE_list)) * 100
        else:
            MRE = 0
            SDR = 0
        np.savetxt(os.path.join(result_dir, 'GCN_MRE_list.txt'), MRE_list)
        print(f"Test finished. MRE: {MRE:.4f}, SDR(<2px): {SDR:.2f}%")

if __name__ == '__main__':
    main()