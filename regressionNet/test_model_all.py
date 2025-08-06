import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from all_dataset_class import BaseLandmarkDataset
from model import load_model
import skimage.io as sk_io
n_landmarks=8
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

def denormalize_landmarks(landmarks, target_shape):
    w, h = target_shape[1], target_shape[0]
    x = (landmarks[:, 0] + 1) / 2 * (w - 1)
    y = (landmarks[:, 1] + 1) / 2 * (h - 1)
    return np.stack([x, y], axis=1)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir_path = os.path.join(script_dir, '../data/clinic_knee/pixel_labels')
    valid_image_dir_path = os.path.join(script_dir, '../data/clinic_knee/test_images')
    model_save_folder_path = os.path.join(script_dir, '../final_results/knee/RegressionNet/model_save')
    result_dir = os.path.join(script_dir, '../results/clinic_knee/RegressionNet/test_results_visualized')
    print('模型保存路径: ', model_save_folder_path)
    test_dataset = RegressionNetDataset(load_image_dir=valid_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
    model = load_model(model='resnet50', n_landmarks=n_landmarks).to(device)

    model_path = os.path.join(model_save_folder_path, 'model.pt')
    if not os.path.exists(model_path):
        print('模型权重不存在，请先训练模型。')
        return
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    MRE_list = []
    print("Testing and visualizing results...")
    maxdistance=0
    maxdistance_idx=0
    for idx in range(len(test_dataset)):
        name, image, heatmap, landmark, template = test_dataset[idx]
        img_path = os.path.join(valid_image_dir_path, name)
        if not os.path.exists(img_path):
            print(f'原图 {img_path} 不存在，跳过')
            continue
        orig_img = sk_io.imread(img_path)
        orig_h, orig_w = orig_img.shape[:2]
        net_h, net_w = 512, 512
        image_tensor = torch.tensor(image).unsqueeze(0).float().to(device)
        with torch.no_grad():
            preds = model(image_tensor)
            pred_landmarks = preds.squeeze(0).cpu().numpy().reshape(-1, 2)
        pred_points_net = denormalize_landmarks(pred_landmarks, (net_h, net_w))
        # gt_points_net = denormalize_landmarks(landmark, (net_h, net_w))
        scale_x = orig_w / net_w
        scale_y = orig_h / net_h
        pred_points_orig = np.zeros_like(pred_points_net)
        # gt_points_orig = np.zeros_like(gt_points_net)
        pred_points_orig[:, 0] = pred_points_net[:, 0] * scale_x
        pred_points_orig[:, 1] = pred_points_net[:, 1] * scale_y
        # gt_points_orig[:, 0] = gt_points_net[:, 0] * scale_x
        # gt_points_orig[:, 1] = gt_points_net[:, 1] * scale_y
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(orig_img)
        # ax.scatter(gt_points_orig[:, 0], gt_points_orig[:, 1], c='g', s=30, label='GT', marker='o')
        ax.scatter(pred_points_orig[:, 0], pred_points_orig[:, 1], c='r', s=30, label='Pred', marker='x')
        # for i in range(len(gt_points_orig)):
        #     ax.plot([gt_points_orig[i, 0], pred_points_orig[i, 0]], 
        #             [gt_points_orig[i, 1], pred_points_orig[i, 1]], 
        #             c='b', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title(name)
        ax.axis('off')
        save_path = os.path.join(result_dir, name.replace('.png', '_result.png').replace('.bmp', '_result.png'))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f'Saved: {save_path}')
        # if pred_points_net.shape == gt_points_net.shape:
        #     distances = np.linalg.norm(pred_points_net - gt_points_net, axis=1)
        #     MRE_list.extend(distances)
        #     if  sum(distances)>maxdistance:
        #         maxdistance = sum(distances)
        #         maxdistance_idx = idx
        print(f"Processed {idx+1}/{len(test_dataset)}: {name}")
    name, image, heatmap, landmark, template = test_dataset[maxdistance_idx]
    print(f"最大距离的图像: {name}, 累计距离: {maxdistance}")
    MRE_list = np.array(MRE_list)
    if len(MRE_list) > 0:
        MRE = np.mean(MRE_list)
        SDR = np.sum(np.asarray(MRE_list) < 2) / np.size(np.array(MRE_list)) * 100
    else:
        MRE = 0
        SDR = 0
    np.savetxt(os.path.join(result_dir, 'RegressionNet_MRE_list.txt'), MRE_list)
    print(f"Test finished. MRE: {MRE:.4f}, SDR(<2px): {SDR:.2f}%")

if __name__ == '__main__':
    main()
