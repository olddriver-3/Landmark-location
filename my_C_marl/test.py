from modelv5 import LandmarkTrainer
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import torch
import os
from all_dataset_class import BaseLandmarkDataset
import yaml
def visualize_rewards(trainer, sample_idx=None):
    """
    可视化每个agent每个状态下所有可能动作的奖赏值
    :param trainer: LandmarkTrainer实例
    :param sample_idx: 指定样本ID，如果为None则随机选择
    """
    # 重置环境
    state = trainer.env.reset(sample_idx)
    print(f"Visualizing rewards for sample: {sample_idx if sample_idx else 'random'}")
    
    # 获取环境信息
    image = trainer.env.image.permute(1, 2, 0).cpu().numpy() * 255
    image = image.astype(np.uint8)
    if image.shape[2] == 1:  # 单通道转三通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 创建可视化图像
    vis_image = image.copy()
    
    # 绘制目标点（绿色）
    for i, landmark in enumerate(trainer.env.landmarks):
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(vis_image, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis_image, f'T{i}', (x+8, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 绘制当前点（黄色）
    for i, pos in enumerate(trainer.env.positions):
        x, y = int(pos[0]), int(pos[1])
        posl=trainer.env.landmarks[i]
        xl,yl=int(posl[0]), int(posl[1])
        cv2.circle(vis_image, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(vis_image, f'A{i}', (x+8, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(vis_image,(x,y),(xl,yl),(255,0,0),2)

    # 计算每个agent所有可能动作的奖赏
    all_rewards = np.zeros((trainer.config['num_landmarks'], 4))
    for agent_idx in range(trainer.config['num_landmarks']):
        if trainer.env.terminated[agent_idx]:
            continue  # 跳过已终止的agent

        original_pos = trainer.env.positions[agent_idx].clone()
        for action in range(4):
            # 模拟动作
            dx, dy = 0, 0
            if action == 0: dy = -1  # 上
            elif action == 1: dy = 1   # 下
            elif action == 2: dx = -1  # 左
            elif action == 3: dx = 1   # 右
            
            new_pos = torch.tensor([original_pos[0] + dx, original_pos[1] + dy])
            
            # 检查边界条件
            x, y = new_pos
            if x < 0 or x >= trainer.env.image_size or y < 0 or y >= trainer.env.image_size:
                all_rewards[agent_idx][action] = -0.1  # 越界惩罚
            else:
                # 计算奖励
                all_rewards[agent_idx][action] = trainer.env._calculate_reward(
                    agent_idx, original_pos, new_pos
                )
    
    # 可视化每个agent的奖赏值
    for agent_idx in range(trainer.config['num_landmarks']):
        if trainer.env.terminated[agent_idx]:
            continue  # 跳过已终止的agent
        
        pos = trainer.env.positions[agent_idx]
        x, y = int(pos[0]), int(pos[1])
        
        # 动作方向偏移量
        offsets = [(0, -30), (0, 30), (-30, 0), (30, 0)]  # 上、下、左、右

        for action in range(4):
            reward = all_rewards[agent_idx][action]
            dx, dy = offsets[action]
            text_x, text_y = x + dx, y + dy
            
            # 设置文本颜色（红色表示惩罚，绿色表示奖励）
            color = (0, 0, 255) if reward < 0 else (0, 255, 0)
            
            # 绘制动作方向指示器
            cv2.arrowedLine(vis_image, (x, y), (text_x, text_y), color, 2, tipLength=0.3)
            
            # 绘制奖赏值
            cv2.putText(vis_image, f"{reward:.2f}", (text_x-15, text_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # # 添加动作标签
            # action_labels = ["Up", "Down", "Left", "Right"]
            # cv2.putText(vis_image, action_labels[action], (text_x-20, text_y-10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # # 添加边界说明
    # boundary_text = "Boundary Conditions:"
    # cv2.putText(vis_image, boundary_text, (20, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # conditions = [
    #     "1. Moving outside image: -0.1 penalty",
    #     "2. Reaching target (<3px): +1.0 reward",
    #     "3. Normal move: improvement - 0.01"
    # ]
    
    # for i, condition in enumerate(conditions):
    #     cv2.putText(vis_image, condition, (40, 65 + i*30), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    
    # 显示图像
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Reward Visualization (Sample: {sample_idx if sample_idx else 'random'})")
    plt.axis('off')
    plt.show()
    
    # 打印数值结果
    print("\nReward Values for Each Agent:")
    for agent_idx in range(trainer.config['num_landmarks']):
        print(f"Agent {agent_idx}:")
        for action, reward in enumerate(all_rewards[agent_idx]):
            print(f"  {['Up', 'Down', 'Left', 'Right'][action]}: {reward:.4f}")




CONFIG = """
gamma: 1
pre_sample_size: 200
episode_steps: 200
batch_size: 32
memory_capacity: 10000
memory_length: 4
eps_start: 1.0
eps_min: 0.1
eps_decay: 0.001
learning_rate: 0.0001
target_update_freq: 50
num_landmarks: 6
image_size: 512
patch_size: 60
max_episodes: 5000
termination_repeat: 8
"""

# 解析配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = yaml.safe_load(CONFIG)

# 设置数据路径
script_dir = os.path.dirname(os.path.abspath(__file__))
annotation_dir_path = os.path.join(script_dir, '../data/chest_6/pixel_labels')
image_dir_path = os.path.join(script_dir, '../data/chest_6/train_images')
save_model_path = os.path.join(script_dir, '../results/chest_6/C_marl/model_savev5')
os.makedirs(save_model_path, exist_ok=True)
log_model_path = os.path.join(save_model_path, 'log')
os.makedirs(log_model_path, exist_ok=True)

# 创建数据集实例
base_dataset = BaseLandmarkDataset(
    load_image_dir=image_dir_path,
    load_annotation_dir=annotation_dir_path,
    target_size=(config['image_size'], config['image_size']),
    num_landmarks=config['num_landmarks'],
    use_template=True
)
if __name__ == "__main__":
    # 创建训练器实例
    # 创建训练器实例后调用
    trainer = LandmarkTrainer(config, base_dataset, save_model_path)
    visualize_rewards(trainer, sample_idx=0)  # 可视化特定样本
    visualize_rewards(trainer)  # 随机样本
