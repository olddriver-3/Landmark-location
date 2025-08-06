import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import os
from collections import deque, defaultdict
import torch.nn.functional as F
from zmq import device
from all_dataset_class import BaseLandmarkDataset
import cv2
# 配置文件常量
CONFIG = """
gamma: 1
pre_sample_size: 5000
episode_steps: 500
batch_size: 32
memory_capacity: 10000
memory_length: 4
eps_start: 1.0
eps_min: 0.1
eps_decay: 0.001
learning_rate: 0.0001
target_update_freq: 50
num_landmarks: 6
image_size: 128
patch_size: 45
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
save_model_path = os.path.join(script_dir, '../results/chest_6/C_marl/model_save')
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

# DQN网络结构 - 每个关键点独立处理
class DQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.C = config['memory_length']
        self.num_landmarks = config['num_landmarks']
        
        # 共享的卷积层
        self.conv0 = nn.Conv2d(self.C, 32, kernel_size=5, padding=1)
        self.maxpool0 = nn.MaxPool2d(2)
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=5, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU()
        
        # 为每个关键点创建独立的全连接层
        self.fc1 = nn.ModuleList([
            nn.Linear(256, 128) for _ in range(self.num_landmarks)
        ])
        self.prelu4 = nn.ModuleList([
            nn.PReLU() for _ in range(self.num_landmarks)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(128, 64) for _ in range(self.num_landmarks)
        ])
        self.prelu5 = nn.ModuleList([
            nn.PReLU() for _ in range(self.num_landmarks)
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(64, 4) for _ in range(self.num_landmarks)
        ])
    
    def forward(self, x):
        """输入形状: (batch, num_landmarks, C, 45, 45)"""
        batch_size = x.size(0)
        outputs = []
        
        # 对每个关键点独立处理
        for i in range(self.num_landmarks):
            # 提取当前关键点的状态序列 (batch, C, 45, 45)
            x_i = x[:, i, :, :, :]
            
            # 共享的卷积层
            x_i = self.conv0(x_i)
            x_i = self.prelu0(x_i)
            x_i = self.maxpool0(x_i)
            
            x_i = self.conv1(x_i)
            x_i = self.prelu1(x_i)
            x_i = self.maxpool1(x_i)
            
            x_i = self.conv2(x_i)
            x_i = self.prelu2(x_i)
            x_i = self.maxpool2(x_i)
            
            x_i = self.conv3(x_i)
            x_i = self.prelu3(x_i)
            
            # 展平特征
            x_i = x_i.view(-1, 256)
            
            # 关键点特定的全连接层
            x_i = self.fc1[i](x_i)
            x_i = self.prelu4[i](x_i)
            x_i = self.fc2[i](x_i)
            x_i = self.prelu5[i](x_i)
            x_i = self.fc3[i](x_i)
            
            outputs.append(x_i)
        
        # 堆叠所有关键点的输出 (batch, num_landmarks, 4)
        return torch.stack(outputs, dim=1)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, config):
        self.capacity = config['memory_capacity']
        self.buffer = deque(maxlen=self.capacity)
        self.patch_size = config['patch_size']
        self.num_landmarks = config['num_landmarks']
    
    def add_step(self, transition):
        """添加完整的状态序列转移"""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """采样batch_size个完整的状态序列转移"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack([s.cpu() for s in states])  # (batch, num_landmarks, C, 1, 45, 45)
        actions = torch.stack([a.cpu() for a in actions])  # (batch, num_landmarks)
        rewards = torch.stack([r.cpu() for r in rewards])  # (batch, num_landmarks)
        next_states = torch.stack([ns.cpu() for ns in next_states])  # (batch, num_landmarks, C, 1, 45, 45)
        dones = torch.stack([d.cpu() for d in dones])  # (batch, num_landmarks)
        
        # 调整状态维度 (batch, num_landmarks, C, 45, 45)
        states = states.squeeze(-3)  # 移除单通道维度
        next_states = next_states.squeeze(-3)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# 环境模拟器
class LandmarkEnv:
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.image = None
        self.landmarks = None
        self.positions = None
        self.position_history = defaultdict(lambda: deque(maxlen=28))  # 每个点28步位置历史
        self.state_queues = [deque(maxlen=config['memory_length']) for _ in range(config['num_landmarks'])]  # 每个点的状态队列
        self.steps = 0
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']
        self.half_patch = self.patch_size // 2
        self.num_landmarks = config['num_landmarks']
        self.terminated_repeats = config['termination_repeat']
        self.terminated = [False] * self.num_landmarks
    
    def _is_out_of_bound(self, pos):
        """检查位置是否越界"""
        x, y = pos.int().tolist()
        return (x < self.half_patch or x >= self.image_size - self.half_patch or 
                y < self.half_patch or y >= self.image_size - self.half_patch)
    
    def reset(self, idx=None):
        """重置环境，使用随机起始位置"""
        if idx is None:
            idx = random.randint(0, len(self.dataset)-1)
        
        # 从数据集中加载样本
        _, image_tensor, _, normalized_landmarks, _ = self.dataset[idx]
        
        # 将图像张量转换为浮点类型
        self.image = torch.tensor(image_tensor).float()
        self.normalized_landmarks = normalized_landmarks
        
        # 将归一化坐标转换为像素坐标
        self.landmarks = self._denormalize_points(normalized_landmarks)
        
        # 生成随机起始位置（确保能获取完整图像块）
        self.positions = torch.zeros(self.num_landmarks, 2)
        for i in range(self.num_landmarks):
            while True:
                x = random.randint(self.half_patch, self.image_size - self.half_patch - 1)
                y = random.randint(self.half_patch, self.image_size - self.half_patch - 1)
                if not self._is_out_of_bound(torch.tensor([x, y])):
                    self.positions[i] = torch.tensor([x, y])
                    break
        
        # 重置历史和终止状态
        self.position_history = defaultdict(lambda: deque(maxlen=28))
        self.terminated = [False] * self.num_landmarks
        self.steps = 0
        
        # 重置状态队列并填充初始状态
        self._reset_state_queues()
        return self._get_state_sequence()
    
    def _reset_state_queues(self):
        """重置状态队列并填充初始状态"""
        for i in range(self.num_landmarks):
            self.state_queues[i].clear()
            current_state = self._get_current_state()
            # 用当前状态填充整个队列（初始时无历史）
            for _ in range(self.config['memory_length']):
                self.state_queues[i].append(current_state[i].clone())
    
    def _denormalize_points(self, normalized_points):
        """将归一化坐标(-1,1)转换为像素坐标"""
        denorm_points = []
        for point in normalized_points:
            # 归一化坐标转换为像素坐标
            x = (point[0] + 1) * (self.image_size - 1) / 2
            y = (point[1] + 1) * (self.image_size - 1) / 2
            denorm_points.append([x, y])
        return torch.tensor(denorm_points)
    
    def _get_patch(self, position):
        """获取关键点周围的图像块"""
        x, y = position.int().tolist()
        return self.image[:1, y-self.half_patch:y+self.half_patch+1, 
                         x-self.half_patch:x+self.half_patch+1]
    
    def _get_current_state(self):
        """获取当前时刻的状态（不含历史）"""
        states = []
        for i, pos in enumerate(self.positions):
            if self.terminated[i] or self._is_out_of_bound(pos):
                states.append(torch.zeros(1, self.patch_size, self.patch_size))
            else:
                patch = self._get_patch(pos)
                states.append(patch if patch is not None else 
                             torch.zeros(1, self.patch_size, self.patch_size))
        return torch.stack(states)
    
    def _get_state_sequence(self):
        """获取完整状态序列（时序顺序：最早->最新）"""
        sequences = []
        for i in range(self.num_landmarks):
            # 转换为张量 (C, 1, 45, 45)
            seq = torch.stack(list(self.state_queues[i]), dim=0)
            sequences.append(seq)
        # 堆叠所有关键点 (num_landmarks, C, 1, 45, 45)
        return torch.stack(sequences, dim=0)
    
    def _calculate_reward(self, landmark_idx, old_pos, new_pos):
        """计算单个关键点的奖励（改进版）"""
        target = self.landmarks[landmark_idx]
        
        # 计算距离改进（归一化到[0,1]范围）
        old_dist = torch.norm(old_pos - target)
        new_dist = torch.norm(new_pos - target)
        max_possible_dist = self.image_size * 1.414  # 图像对角线距离
        
        # 基础奖励 = 距离改进比例
        improvement = (old_dist - new_dist) / max_possible_dist
        
        # 越界惩罚
        if self._is_out_of_bound(new_pos):
            return -0.1  # 减小惩罚值
        
        # 成功到达奖励
        if new_dist < 3.0:  # 3像素内视为成功
            return 1.0
        
        # 正常移动奖励 = 改进比例 - 小惩罚（鼓励高效移动）
        return improvement.item() - 0.01
    
    def _check_termination(self, landmark_idx):
        """检查是否满足终止条件（位置重复4次及以上）"""
        pos_history = self.position_history[landmark_idx]
        if len(pos_history) < 4:
            return False
        
        # 统计位置出现次数
        pos_counts = {}
        for pos in pos_history:
            pos_tuple = tuple(pos.tolist())
            pos_counts[pos_tuple] = pos_counts.get(pos_tuple, 0) + 1

        # 检查是否有位置重复8次及以上
        return any(count >= self.terminated_repeats for count in pos_counts.values())
    
    def step(self, actions):
        """执行动作，返回新状态序列、奖励、终止标志"""
        old_positions = self.positions.clone()
        rewards = torch.zeros(self.num_landmarks)
        dones = torch.zeros(self.num_landmarks, dtype=torch.bool)
        
        # 执行动作 (0:上, 1:下, 2:左, 3:右)
        for i in range(self.num_landmarks):
            if self.terminated[i]:
                # 已终止的点不移动，奖励为0
                rewards[i] = 0.0
                dones[i] = True
                continue
                
            dx, dy = 0, 0
            if actions[i] == 0: dy = -1  # 上
            elif actions[i] == 1: dy = 1   # 下
            elif actions[i] == 2: dx = -1  # 左
            elif actions[i] == 3: dx = 1   # 右
            
            new_pos = torch.tensor([old_positions[i,0] + dx, old_positions[i,1] + dy])
            
            # 更新位置
            self.positions[i] = new_pos
            self.position_history[i].append(new_pos.clone())
            
            # 计算奖励
            rewards[i] = self._calculate_reward(i, old_positions[i], new_pos)
            
            # 检查终止条件
            if rewards[i] == -0.1 or self._check_termination(i):
                self.terminated[i] = True
                dones[i] = True
        
        # 更新状态队列
        current_state = self._get_current_state()
        for i in range(self.num_landmarks):
            if not self.terminated[i]:
                self.state_queues[i].append(current_state[i].clone())
        
        self.steps += 1
        new_state_seq = self._get_state_sequence()
        return new_state_seq, rewards, dones

# DQN智能体
class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(config).to(self.device)
        self.target_network = DQN(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.MSELoss()
        self.eps = config['eps_start']
        self.C = config['memory_length']
    
    def select_action(self, state, explore=True):
        """使用ε-贪婪策略选择动作"""
        # state形状: (num_landmarks, C, 1, 45, 45)
        # 增加批次维度
        state_seq = state.unsqueeze(0).to(self.device)  # (1, num_landmarks, C, 1, 45, 45)
        
        if explore and random.random() < self.eps:
            # 随机动作（排除已终止的点）
            actions = []
            for i in range(self.config['num_landmarks']):
                # 检查是否终止（使用最新状态）
                if torch.isclose(state[i, -1], torch.zeros_like(state[i, -1])).all():
                    actions.append(-1)
                else:
                    actions.append(random.randint(0, 3))
            return torch.tensor(actions)
        
        with torch.no_grad():
            # 调整维度 (1, num_landmarks, C, 45, 45)
            state_reshaped = state.permute(2,0,1, 3, 4).to(device)  # (1, num_landmarks, C, 45, 45)
            
            # 通过网络获取Q值
            q_values = self.q_network(state_reshaped)  # (1, num_landmarks, 4)
            q_values = q_values.squeeze(0)  # (num_landmarks, 4)
            
            # 为已终止的点设置Q值为极小值
            terminated_mask = torch.stack([
                torch.isclose(state[i, -1], torch.zeros_like(state[i, -1])).all()
                for i in range(self.config['num_landmarks'])
            ])
            q_values[terminated_mask] = -1e9
            
            actions = torch.argmax(q_values, dim=1)  # (num_landmarks,)
            
            # 将已终止点的动作设为-1
            actions[terminated_mask] = -1
            
            return actions
    
    def update_network(self, batch, gamma):
        """更新Q网络"""
        states, actions, rewards, next_states, dones = batch
        
        # 移动到设备
        states = states.to(self.device)  # (batch, num_landmarks, C, 45, 45)
        actions = actions.to(self.device)  # (batch, num_landmarks)
        rewards = rewards.to(self.device)  # (batch, num_landmarks)
        next_states = next_states.to(self.device)  # (batch, num_landmarks, C, 45, 45)
        dones = dones.to(self.device)  # (batch, num_landmarks)
        
        # 计算当前Q值
        current_q = self.q_network(states)  # (batch, num_landmarks, 4)
        
        # 选择动作对应的Q值
        actions = actions.long()
        batch_size, num_landmarks = actions.size()
        current_q = current_q[torch.arange(batch_size).unsqueeze(1), 
                             torch.arange(num_landmarks).unsqueeze(0), 
                             actions]  # (batch, num_landmarks)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states)  # (batch, num_landmarks, 4)
            next_q = next_q.max(-1)[0]  # 在动作维度取最大值 (batch, num_landmarks)
            
            # 计算目标Q值
            target_q = rewards + gamma * next_q * (~dones).float()
        
        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        """衰减ε值"""
        self.eps = max(self.config['eps_min'], self.eps - self.config['eps_decay'])
    
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())

# 训练主类
class LandmarkTrainer:
    def __init__(self, config, dataset, save_path):
        self.config = config
        self.dataset = dataset
        self.save_path = save_path
        self.env = LandmarkEnv(config, self.dataset)
        self.agent = DQNAgent(config)
        self.buffer = ReplayBuffer(config)
        self.update_counter = 0
    
    def pre_sample(self):
        """预填充经验回放缓冲区"""
        print("Pre-sampling experience...")
        while len(self.buffer) < self.config['pre_sample_size']:
            state = self.env.reset()
            
            for step in range(self.config['episode_steps']):
                action = self.agent.select_action(state, explore=True)
                next_state, reward, done = self.env.step(action)
                
                # 存储完整的状态序列转移
                self.buffer.add_step((state, action, reward, next_state, done))
                
                state = next_state
                
                if all(done):
                    break
        
        print(f"Pre-sampling complete. Buffer size: {len(self.buffer)}")
    
    def train(self):
        """训练主循环"""
        self.pre_sample()
        print("Starting training...")
        
        for episode in range(1, self.config['max_episodes'] + 1):
            state = self.env.reset()
            total_reward = 0
            episode_losses = []
            
            for step in range(self.config['episode_steps']):
                # 选择并执行动作
                action = self.agent.select_action(state, explore=True)
                next_state, reward, done = self.env.step(action)
                total_reward += reward.sum().item()
                
                # 存储完整的状态序列转移
                self.buffer.add_step((state, action, reward, next_state, done))
                
                # 定期训练（每4步更新一次）
                self.update_counter += 1
                if self.update_counter % 4 == 0 and len(self.buffer) >= self.config['batch_size']:
                    batch = self.buffer.sample(self.config['batch_size'])
                    loss = self.agent.update_network(batch, self.config['gamma'])
                    episode_losses.append(loss)
                
                # 更新状态
                state = next_state
                
                # 检查是否终止
                if all(done):
                    break
            
            # 更新目标网络
            if episode % self.config['target_update_freq'] == 0:
                self.agent.update_target_network()
            
            # 衰减ε值
            self.agent.update_epsilon()
            
            # 记录日志
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            print(f"Episode {episode}/{self.config['max_episodes']}, "
                  f"Total Reward: {total_reward:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Epsilon: {self.agent.eps:.4f}")
            
            # 定期保存模型
            if episode % 100 == 0:
                model_path = os.path.join(self.save_path, f"dqn_model_ep{episode}.pth")
                self.agent.save_model(model_path)
                print(f"Model saved to {model_path}")
            
            # 定期测试
            if episode % 10 == 0:
                self.single_test()
        
        print("Training completed!")
        final_model_path = os.path.join(self.save_path, "dqn_model_final.pth")
        self.agent.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    def single_test(self, sample_idx=None, vis=False):
        """单样本预测测试，支持可视化"""
        state = self.env.reset(sample_idx)
        total_reward = 0
        step = 0
        video_writer = None
        
        print("\nStarting test...")
        print(f"Initial positions: {self.env.positions.tolist()}")
        print(f"Target landmarks: {self.env.landmarks.tolist()}")
        
        # 可视化准备
        if vis:
            # 获取完整图像
            full_image = self.env.image.permute(1, 2, 0).cpu().numpy() * 255
            full_image = full_image.astype(np.uint8)
            if full_image.shape[2] == 1:  # 如果是单通道，转换为三通道
                full_image = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
            
            # 创建视频写入器
            video_path = os.path.join(self.save_path, "test_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_path, fourcc, 5.0, 
                (full_image.shape[1], full_image.shape[0])
            )

        while True:
            action = self.agent.select_action(state, explore=False)
            next_state, reward, done = self.env.step(action)
            total_reward += reward.sum().item()
            
            # 可视化处理
            if vis:
                # 获取当前图像
                frame = self.env.image.permute(1, 2, 0).cpu().numpy() * 255
                frame = frame.astype(np.uint8)
                if frame.shape[2] == 1:  # 如果是单通道，转换为三通道
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame = frame.copy()
                
                # 绘制目标点（绿色）
                for i, target in enumerate(self.env.landmarks):
                    x, y = int(target[0]), int(target[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"T{i}", (x+8, y+8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 绘制当前点（黄色）和观察区域（红色框）
                for i, pos in enumerate(self.env.positions):
                    x, y = int(pos[0]), int(pos[1])
                    
                    # 绘制当前点
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                    cv2.putText(frame, f"A{i}", (x+8, y+8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # 绘制观察区域（红色框）
                    half = self.env.half_patch
                    cv2.rectangle(frame, (x-half, y-half), (x+half, y+half), 
                                 (0, 0, 255), 1)
                    
                    # 绘制到目标的连线（蓝色虚线）
                    target = self.env.landmarks[i]
                    tx, ty = int(target[0]), int(target[1])
                    cv2.line(frame, (x, y), (tx, ty), (255, 0, 0), 1, cv2.LINE_AA)
                    
                    # 计算并显示距离
                    distance = torch.norm(pos - target).item()
                    mid_x, mid_y = (x + tx) // 2, (y + ty) // 2
                    cv2.putText(frame, f"{distance:.1f}px", (mid_x, mid_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 添加步数和奖励信息
                cv2.putText(frame,f"ID: {sample_idx}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Step: {step}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Total Reward: {total_reward:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 写入视频帧
                video_writer.write(frame)
            
            # 检查终止条件
            if all(done) or step > self.config['episode_steps']:
                break
            
            state = next_state
            step += 1
        
        # 计算最终误差
        final_positions = self.env.positions
        target_positions = self.env.landmarks
        distances = torch.norm(final_positions - target_positions, dim=1)
        mean_distance = distances.mean().item()
        
        print(f"Test completed! Total Reward: {total_reward:.2f}, Mean Distance: {mean_distance:.4f}")
        
        # 完成视频写入
        if vis and video_writer:
            video_writer.release()
            print(f"Test video saved to {video_path}")
        
        return None

# 主程序
if __name__ == "__main__":
    # 创建训练器实例
    trainer = LandmarkTrainer(config, base_dataset, save_model_path)
    
    # 开始训练
    trainer.train()
    
    # # 测试训练好的模型
    # final_model_path = os.path.join(save_model_path, "dqn_model_final.pth")
    # trainer.agent.load_model(final_model_path)
    # test_history = trainer.single_test(vis=True)
