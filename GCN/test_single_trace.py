import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from config import load_config
from model.base_module import adjacency_matrix
from model.image2shape_ResNet101_NOFPN import Image2Shape
import skimage.io as sk_io
from all_dataset_class import BaseLandmarkDataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

class GCNDataset(BaseLandmarkDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            target_size=(512, 384),
            num_landmarks=6,
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

def single_trace(filename):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = load_config(exp_id=100)
    annotation_dir_path = '../data/knee/pixel_labels'
    valid_image_dir_path = '../data/knee/val_images'
    model_save_folder_path = '../final_results/knee/GCN/model_save'
    result_dir = '../results/knee/GCN/test_results_visualized'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir_path= os.path.join(script_dir, annotation_dir_path)
    valid_image_dir_path = os.path.join(script_dir, valid_image_dir_path)
    model_save_folder_path = os.path.join(script_dir, model_save_folder_path)
    result_dir = os.path.join(script_dir, result_dir)
    print('模型保存路径: ', model_save_folder_path)
    test_dataset = GCNDataset(load_image_dir=valid_image_dir_path, load_annotation_dir=annotation_dir_path, do_augmentation=False)
    model = Image2Shape(config).to(device)

    model_path = os.path.join(model_save_folder_path, 'model.pt')
    if not os.path.exists(model_path):
        print('模型权重不存在，请先训练模型。')
        return
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    targetidx= 0
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    MRE_list = []
    for idx, (name, image, heatmap, landmark, template) in enumerate(test_dataset):
        if name == filename:
            targetidx= idx 
            break
    name, image, heatmap, landmark, template= test_dataset[targetidx]
    img_path = os.path.join(valid_image_dir_path, name)
    if not os.path.exists(img_path):
        print(f'原图 {img_path} 不存在，跳过')
        return
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
        pred_landmarks_0 = preds[0].squeeze(0).cpu().numpy()
        pred_landmarks_1 = preds[1].squeeze(0).cpu().numpy()
        pred_landmarks_2 = preds[2].squeeze(0).cpu().numpy()
        pred_landmarks_3 = preds[3].squeeze(0).cpu().numpy()
        pred_landmarks_4 = preds[4].squeeze(0).cpu().numpy()
        pred_landmarks_5 = preds[5].squeeze(0).cpu().numpy()


    # 先还原到网络输入尺寸
    pred_points_net_0 = denormalize_landmarks(pred_landmarks_0, (net_h, net_w))
    pred_points_net_1 = denormalize_landmarks(pred_landmarks_1, (net_h, net_w))
    pred_points_net_2 = denormalize_landmarks(pred_landmarks_2, (net_h, net_w))
    pred_points_net_3 = denormalize_landmarks(pred_landmarks_3, (net_h, net_w))
    pred_points_net_4 = denormalize_landmarks(pred_landmarks_4, (net_h, net_w))
    pred_points_net_5 = denormalize_landmarks(pred_landmarks_5, (net_h, net_w))
    gt_points_net = denormalize_landmarks(landmark, (net_h, net_w))
    # 再映射到原图尺寸
    #帮我将下面内容改为展示6*pred_landmarks的形式

    scale_x = orig_w / net_w
    scale_y = orig_h / net_h
    pred_points_orig_0 = np.zeros_like(pred_points_net_0)
    pred_points_orig_1 = np.zeros_like(pred_points_net_1)
    pred_points_orig_2 = np.zeros_like(pred_points_net_2)
    pred_points_orig_3 = np.zeros_like(pred_points_net_3)
    pred_points_orig_4 = np.zeros_like(pred_points_net_4)
    pred_points_orig_5 = np.zeros_like(pred_points_net_5)
    gt_points_orig = np.zeros_like(gt_points_net)
    pred_points_orig_0[:, 0] = pred_points_net_0[:, 0] * scale_x
    pred_points_orig_0[:, 1] = pred_points_net_0[:, 1] * scale_y
    pred_points_orig_1[:, 0] = pred_points_net_1[:, 0] * scale_x
    pred_points_orig_1[:, 1] = pred_points_net_1[:, 1] * scale_y
    pred_points_orig_2[:, 0] = pred_points_net_2[:, 0] * scale_x
    pred_points_orig_2[:, 1] = pred_points_net_2[:, 1] * scale_y
    pred_points_orig_3[:, 0] = pred_points_net_3[:, 0] * scale_x
    pred_points_orig_3[:, 1] = pred_points_net_3[:, 1] * scale_y
    pred_points_orig_4[:, 0] = pred_points_net_4[:, 0] * scale_x
    pred_points_orig_4[:, 1] = pred_points_net_4[:, 1] * scale_y
    pred_points_orig_5[:, 0] = pred_points_net_5[:, 0] * scale_x
    pred_points_orig_5[:, 1] = pred_points_net_5[:, 1] * scale_y
    gt_points_orig[:, 0] = gt_points_net[:, 0] * scale_x
    gt_points_orig[:, 1] = gt_points_net[:, 1] * scale_y

    # 画图
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(orig_img)
    ax.scatter(gt_points_orig[:, 0], gt_points_orig[:, 1], c='g', s=30, label='GT', marker='o')
    ax.scatter(pred_points_orig_0[:, 0], pred_points_orig_0[:, 1], c='r', s=30, marker='x')
    ax.scatter(pred_points_orig_1[:, 0], pred_points_orig_1[:, 1], c='r', s=30, marker='x')
    ax.scatter(pred_points_orig_2[:, 0], pred_points_orig_2[:, 1], c='r', s=30, marker='x')
    ax.scatter(pred_points_orig_3[:, 0], pred_points_orig_3[:, 1], c='r', s=30, marker='x')
    ax.scatter(pred_points_orig_4[:, 0], pred_points_orig_4[:, 1], c='r', s=30, marker='x')
    ax.scatter(pred_points_orig_5[:, 0], pred_points_orig_5[:, 1], c='r', s=30, marker='x')
    #在每个点上方添加文字标签
    # for i in range(len(gt_points_orig)):
    #     # ax.text(gt_points_orig[i, 0], gt_points_orig[i, 1] - 5, f'GT=', color='g', fontsize=10)
    #     ax.text(pred_points_orig_0[i, 0], pred_points_orig_0[i, 1] - 5, f'init', color='r', fontsize=10)
    #     ax.text(pred_points_orig_1[i, 0], pred_points_orig_1[i, 1] - 5, f'1', color='r', fontsize=10)
    #     ax.text(pred_points_orig_2[i, 0], pred_points_orig_2[i, 1] - 5, f'2', color='r', fontsize=10)
    #     ax.text(pred_points_orig_3[i, 0], pred_points_orig_3[i, 1] - 5, f'3', color='r', fontsize=10)
    #     ax.text(pred_points_orig_4[i, 0], pred_points_orig_4[i, 1] - 5, f'4', color='r', fontsize=10)
    #     ax.text(pred_points_orig_5[i, 0], pred_points_orig_5[i, 1] - 5, f'5', color='r', fontsize=10)

    ax.legend()
    ax.set_title(name)
    ax.axis('off')
    # 对 gt_points_orig[0, :] 坐标附近添加放大插图，要有矩形框和连接线
    # 设置放大区域的大小
    zoom_size = 60  # 放大区域边长
    x0, y0 = gt_points_orig[0, 0], gt_points_orig[0, 1]
    x1, y1 = int(max(x0 - zoom_size // 2, 0)), int(max(y0 - zoom_size // 2, 0))
    x2, y2 = int(min(x0 + zoom_size // 2, orig_w)), int(min(y0 + zoom_size // 2, orig_h))

    # 在主图上画矩形框
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='orange', facecolor='none')
    ax.add_patch(rect)

    # 添加放大插图
    target_landmark_idx=0
    axins = inset_axes(ax, width="30%", height="30%", loc='upper right', borderpad=2)
    axins.imshow(orig_img)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1)  # 注意y轴反向
    axins.scatter(gt_points_orig[target_landmark_idx, 0], gt_points_orig[target_landmark_idx, 1], c='g', s=40, marker='o')
    axins.scatter(pred_points_orig_0[target_landmark_idx, 0], pred_points_orig_0[target_landmark_idx, 1], c='r', s=40, marker='x')
    axins.scatter(pred_points_orig_1[target_landmark_idx, 0], pred_points_orig_1[target_landmark_idx, 1], c='r', s=40, marker='x')
    axins.scatter(pred_points_orig_2[target_landmark_idx, 0], pred_points_orig_2[target_landmark_idx, 1], c='r', s=40, marker='x')
    axins.scatter(pred_points_orig_3[target_landmark_idx, 0], pred_points_orig_3[target_landmark_idx, 1], c='r', s=40, marker='x')
    axins.scatter(pred_points_orig_4[target_landmark_idx, 0], pred_points_orig_4[target_landmark_idx, 1], c='r', s=40, marker='x')
    axins.scatter(pred_points_orig_5[target_landmark_idx, 0], pred_points_orig_5[target_landmark_idx, 1], c='r', s=40, marker='x')
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title('Zoomed Landmark 0', fontsize=10)
    axins.plot([pred_points_orig_1[target_landmark_idx, 0], pred_points_orig_0[target_landmark_idx, 0]], 
                [pred_points_orig_1[target_landmark_idx, 1], pred_points_orig_0[target_landmark_idx, 1]], 
                c='b', linestyle='--', alpha=0.5)
    axins.plot([pred_points_orig_2[target_landmark_idx, 0], pred_points_orig_1[target_landmark_idx, 0]], 
                [pred_points_orig_2[target_landmark_idx, 1], pred_points_orig_1[target_landmark_idx, 1]], 
                c='b', linestyle='--', alpha=0.5)
    axins.plot([pred_points_orig_3[target_landmark_idx, 0], pred_points_orig_2[target_landmark_idx, 0]], 
                [pred_points_orig_3[target_landmark_idx, 1], pred_points_orig_2[target_landmark_idx, 1]], 
                c='b', linestyle='--', alpha=0.5)
    axins.plot([pred_points_orig_4[target_landmark_idx, 0], pred_points_orig_3[target_landmark_idx, 0]], 
                [pred_points_orig_4[target_landmark_idx, 1], pred_points_orig_3[target_landmark_idx, 1]], 
                c='b', linestyle='--', alpha=0.5)
    axins.plot([pred_points_orig_5[target_landmark_idx, 0], pred_points_orig_4[target_landmark_idx, 0]], 
                [pred_points_orig_5[target_landmark_idx, 1], pred_points_orig_4[target_landmark_idx, 1]], 
                c='b', linestyle='--', alpha=0.5)
    axins.plot([gt_points_orig[target_landmark_idx, 0], pred_points_orig_5[target_landmark_idx, 0]], 
                [gt_points_orig[target_landmark_idx, 1], pred_points_orig_5[target_landmark_idx, 1]], 
                c='b', linestyle='--', alpha=0.5)
    axins.text(pred_points_orig_0[target_landmark_idx, 0], pred_points_orig_0[target_landmark_idx, 1] - 5, f'init', color='r', fontsize=10)
    # 连接主图和插图
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="orange", lw=1)
    save_path ="trace_result.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved: {save_path}')

if __name__ == '__main__':
    single_trace('OP90_2.png')