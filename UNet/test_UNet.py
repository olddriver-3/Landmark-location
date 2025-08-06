import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from all_dataset_class import BaseLandmarkDataset
from UNet import UNet
import yaml

# 配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_landmarks = 8
input_size = (512, 512)
heatmap_size = (512, 512)  # UNet输出未降采样

# 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
annotation_dir = os.path.join(script_dir, '../data/clinic_knee/pixel_labels')
val_image_dir = os.path.join(script_dir, '../data/clinic_knee/test_images')
model_save_path = os.path.join(script_dir, '../final_results/knee/UNet/model_save/model.pt')
result_dir = os.path.join(script_dir, '../results/clinic_knee/UNet/test_results_visualized')
os.makedirs(result_dir, exist_ok=True)
# 数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
config['num_classes'] = num_landmarks
with open(config_path, 'w', encoding='utf-8') as file:
    yaml.dump(config, file, sort_keys=False, default_flow_style=False)
print("YAML文件已更新")
class UNetTestDataset(BaseLandmarkDataset):
    def __init__(self, load_image_dir, load_annotation_dir, **kwargs):
        super().__init__(
            load_image_dir=load_image_dir,
            load_annotation_dir=load_annotation_dir,
            target_size=input_size,
            num_landmarks=num_landmarks,
            use_template=False,
            heatmap_sigma=3,
            heatmap_radius=50,
            heatmap_downsample=1,  # UNet输出未降采样
            do_augmentation=False,
        )

def load_model():
    # 读取config.yaml
    import yaml
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    class ConfigObj:
        def __init__(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)
    return UNet(ConfigObj(config))

if __name__=='__main__':
    # 加载模型
    model = load_model().to(device)
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()

    # 构建DataLoader
    test_dataset = UNetTestDataset(load_image_dir=val_image_dir, load_annotation_dir=annotation_dir, do_augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    MRE_list = []
    print("Testing and visualizing results...")
    with torch.no_grad():
        for idx, (filename, image, heatmap) in enumerate(test_dataset):
            # image: (3, 512, 512), heatmap: (num_landmarks, 512, 512)
            img_path = os.path.join(val_image_dir, filename)
            orig_img = skio.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]

            # 送入模型
            input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
            pred_heatmap = model(input_tensor)  # (1, num_landmarks, 512, 512)
            pred_heatmap = pred_heatmap.cpu().numpy()[0]  # (num_landmarks, 512, 512)

            # 取每个通道最大值点作为预测点
            pred_coords = []
            for i in range(num_landmarks):
                hm = pred_heatmap[i]
                y, x = np.unravel_index(np.argmax(hm), hm.shape)
                pred_coords.append([x, y])
            pred_coords = np.array(pred_coords)

            # 金标准点（热图最大值）
            # gt_heatmap = heatmap
            # # gt_coords = []
            # for i in range(num_landmarks):
            #     hm = gt_heatmap[i]
            #     y, x = np.unravel_index(np.argmax(hm), hm.shape)
            #     gt_coords.append([x, y])
            # gt_coords = np.array(gt_coords)

            # 坐标映射回原图
            scale_x = orig_w / heatmap_size[1]
            scale_y = orig_h / heatmap_size[0]
            pred_coords_orig = np.stack([pred_coords[:,0]*scale_x, pred_coords[:,1]*scale_y], axis=1)
            # gt_coords_orig = np.stack([gt_coords[:,0]*scale_x, gt_coords[:,1]*scale_y], axis=1)

            # # 计算欧氏距离（输入分辨率下）
            # scale_x = input_size[1] / heatmap_size[1]
            # scale_y = input_size[0] / heatmap_size[0]
            # pred_coords_input = np.stack([pred_coords[:,0]*scale_x, pred_coords[:,1]*scale_y], axis=1)
            # gt_coords_input = np.stack([gt_coords[:,0]*scale_x, gt_coords[:,1]*scale_y], axis=1)
            # if pred_coords_input.shape == gt_coords_input.shape:
            #     distances = np.linalg.norm(pred_coords_input - gt_coords_input, axis=1)
            #     MRE_list.extend(distances)

            # 可视化
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.imshow(orig_img)
            # ax.scatter(gt_coords_orig[:, 0], gt_coords_orig[:, 1], c='g', s=30, label='GT', marker='o')
            ax.scatter(pred_coords_orig[:, 0], pred_coords_orig[:, 1], c='r', s=30, label='Pred', marker='x')
            ax.legend()
            ax.set_title(filename)
            ax.axis('off')
            save_path = os.path.join(result_dir, filename.replace('.png', '_result.png').replace('.bmp', '_result.png'))
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f'Saved: {save_path}')
            print(f"Processed {idx+1}/{len(test_dataset)}: {filename}")

    # 统计指标
    np.savetxt(os.path.join(result_dir, 'UNet_MRE_list.txt'), MRE_list)
    if len(MRE_list) > 0:
        MRE = np.mean(MRE_list)
        SDR = np.sum(np.asarray(MRE_list) < 2) / np.size(np.array(MRE_list)) * 100
    else:
        MRE = 0
        SDR = 0
    print(f"Test finished. MRE: {MRE:.4f}, SDR(<2px): {SDR:.2f}%")