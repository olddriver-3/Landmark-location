from re import A
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
import math
from tqdm import tqdm
# 配置文件常量
CONFIG = """
gamma: 1
pre_sample_size: 400
episode_steps: 200
batch_size: 32
memory_capacity: 50000
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
        states, actions, rewards, next_states, all_next_states,all_rewards,dones,reachbound= zip(*batch)
        
        # 转换为张量
        states = torch.stack([s.cpu() for s in states])
        actions = torch.stack([a.cpu() for a in actions])
        rewards = torch.stack([r.cpu() for r in rewards])
        next_states = torch.stack([ns.cpu() for ns in next_states])
        all_next_states = torch.stack([ans.cpu() for ans in all_next_states])
        all_rewards = torch.stack([ar.cpu() for ar in all_rewards])
        dones = torch.stack([d.cpu() for d in dones])
        reachbound = torch.stack([rb.cpu() for rb in reachbound])
        # 调整状态维度
        states = states.squeeze(-3)
        next_states = next_states.squeeze(-3)

        return states, actions, rewards, next_states, all_next_states, all_rewards, dones, reachbound

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
        self.terminated =torch.zeros(self.num_landmarks, dtype=torch.bool)  # 终止状态
    # def _is_out_of_bound(self, pos):
    #     """检查位置是否越界"""
    #     x, y = pos.int().tolist()
    #     return (x < self.half_patch or x >= self.image_size - self.half_patch or 
    #             y < self.half_patch or y >= self.image_size - self.half_patch)
    
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
        
        # 生成随机起始位置
        self.positions = torch.zeros(self.num_landmarks, 2)
        for i in range(self.num_landmarks):
            while True:
                x = random.randint(self.half_patch, self.image_size - self.half_patch - 1)
                y = random.randint(self.half_patch, self.image_size - self.half_patch - 1)
                # if not self._is_out_of_bound(torch.tensor([x, y])):
                self.positions[i] = torch.tensor([x, y])
                break
        
        # 重置历史和终止状态
        self.terminated =torch.zeros(self.num_landmarks, dtype=torch.bool) # 终止状态
        self.position_history = defaultdict(lambda: deque(maxlen=28))
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
        if x < 0 or x >= self.image_size or y < 0 or y >= self.image_size:
            raise ValueError("Position out of bounds")
        if (x < self.half_patch or x >= self.image_size - self.half_patch or
            y < self.half_patch or y >= self.image_size - self.half_patch):
            #在半个patch的0填充
            padding_image = F.pad(self.image[0], (self.half_patch, self.half_patch, 
                                                   self.half_patch, self.half_patch), mode='constant', value=0)
            return padding_image[torch.newaxis,y:y+2*self.half_patch+1, 
                         x:x+2*self.half_patch+1]
        return self.image[:1, y-self.half_patch:y+self.half_patch+1, 
                         x-self.half_patch:x+self.half_patch+1]
    
    def _get_current_state(self):
        """获取当前时刻的状态（不含历史）"""
        states = []
        for i, pos in enumerate(self.positions):
            # if self.terminated[i]: #or self._is_out_of_bound(pos):
            #     states.append(torch.zeros(1, self.patch_size, self.patch_size))#训练时忽略
            # else:
            patch = self._get_patch(pos)
            states.append(patch)
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
        max_possible_dist = self.patch_size * 1.414  # 图像对角线距离
        
        # 基础奖励 = 距离改进比例
        improvement = (old_dist - new_dist) / max_possible_dist
        
        # # 越界惩罚
        # if self._is_out_of_bound(new_pos):
        #     return -0.1  # 减小惩罚值
        
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
    
    def _check_reach_bound(self, pos):
        x,y=pos.int()
        reachbound = torch.zeros(4, dtype=bool)  # [y下边界, y上边界, x下边界, x上边界]
        if x==0:
            reachbound[2]=True
        if x==self.image_size-1:
            reachbound[3]=True
        if y==0:
            reachbound[0]=True
        if y==self.image_size-1:
            reachbound[1]=True
        return reachbound
    def step(self, actions, mode="test"):
        """执行动作，返回新状态、奖励、终止标志，以及所有动作的可能下一状态"""
        rewards = torch.zeros(self.num_landmarks)
        overstep_punish=-0.1#越界惩罚
        # 保存当前位置，因为我们会尝试所有动作
        original_positions = self.positions.clone()
        next_positions = torch.zeros_like(self.positions)
        # 是否接触边界
        #[y下边界, y上边界,x下边界,x上边界]
        reachbound = torch.zeros((self.num_landmarks, 4), dtype=bool)
        # 执行实际动作
        for i in range(self.num_landmarks):
            if self.terminated[i]:
                #actions[i]被忽略
                rewards[i] =  self._calculate_reward(i, original_positions[i-1], original_positions[i])
                reachbound[i] = self._check_reach_bound(self.positions[i])
                continue
                
            dx, dy = 0, 0
            if actions[i] == 0: dy = -1
            elif actions[i] == 1: dy = 1
            elif actions[i] == 2: dx = -1
            elif actions[i] == 3: dx = 1
            
            new_pos = torch.tensor([original_positions[i,0] + dx, original_positions[i,1] + dy])
            reachbound[i] = self._check_reach_bound(new_pos)
            rewards[i] = self._calculate_reward(i, original_positions[i], new_pos)
            self.positions[i] = new_pos
            self.position_history[i].append(new_pos.clone())
            next_positions[i] = new_pos
            if self._check_termination(i):
                self.terminated[i] = True
        
        # 更新状态队列（实际动作）
        try:
            current_state = self._get_current_state()
        except ValueError:
            print("Error: Current state could not be retrieved.")
        for i in range(self.num_landmarks):
            if not self.terminated[i]:
                self.state_queues[i].append(current_state[i].clone())
        next_state_seq = self._get_state_sequence()
        if mode=="train":
            # 计算所有动作的可能下一状态
            all_next_states = torch.zeros(
                self.num_landmarks, 
                self.config['memory_length'], 
                4,  # 动作数量
                self.patch_size, 
                self.patch_size
            )
            # 计算所有动作的可能奖赏
            all_rewards = torch.zeros(
                self.num_landmarks,
                4  # 动作数量
            )
            for i in range(self.num_landmarks):
                if self.terminated[i]:
                    #下一步的所有动作下一状态和奖赏全0且被忽略

                    continue
                    
                # 保存当前状态队列
                saved_queue = list(self.state_queues[i])
                
                for action in range(4):
                    # 恢复原始位置
                    self.positions[i] = original_positions[i]
                    
                    # 模拟执行动作
                    dx, dy = 0, 0
                    if action == 0: dy = -1
                    elif action == 1: dy = 1
                    elif action == 2: dx = -1
                    elif action == 3: dx = 1

                    new_pos = torch.tensor([original_positions[i,0] + dx, original_positions[i,1] + dy])
                    #如果越界
                    x,y=new_pos
                    if x < 0 or x >= self.image_size or y < 0 or y >= self.image_size:
                        all_rewards[i][action] = overstep_punish
                        sim_seq = torch.zeros(self.config['memory_length'],1, self.patch_size, self.patch_size)  # 训练时忽略
                    else:
                        all_rewards[i][action] = self._calculate_reward(i, original_positions[i], new_pos)
                        # 获取模拟后的状态
                        self.positions[i] = new_pos
                        sim_state = self._get_current_state()[i]
                        # 更新状态队列（模拟）
                        sim_queue = deque(saved_queue, maxlen=self.config['memory_length'])
                        sim_queue.append(sim_state.clone())
                        sim_seq = torch.stack(list(sim_queue), dim=0)  # (C, 1, 45, 45)

                    # 存储到结果张量
                    all_next_states[i, :, action, :, :] = sim_seq.squeeze(1)
                
                # 恢复实际位置
                self.positions[i] = next_positions[i]  # 使用实际动作后的位置

            return next_state_seq, rewards, self.terminated, all_next_states, all_rewards, reachbound
        else:
            return next_state_seq, rewards, self.terminated, reachbound
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
    
    def select_action(self, state, reachbound=None,explore=True,randomchoose=False):
        """使用ε-贪婪策略选择动作"""
        # state形状: (num_landmarks, C, 1, 45, 45)
        # 增加批次维度
        state_seq = state.unsqueeze(0).to(self.device)  # (1, num_landmarks, C, 1, 45, 45)
        if reachbound is not None:
            if reachbound.any():
                # 有可用的行为
                pass
        if (explore and random.random() < self.eps) or randomchoose:
            # 随机动作（排除已终止的点）
            actions = []
            if reachbound is not None:
                for availability in reachbound:
                    # 找到 False 的索引（可用的行为）
                    available_indices = torch.where(~availability)[0]
                    selected_action = available_indices[torch.randint(0, len(available_indices), (1,))]
                    actions.append(selected_action)
            else:
                for i in range(self.config['num_landmarks']):
                    actions.append(random.randint(0, 3))
            return torch.tensor(actions)

        
        with torch.no_grad():
            # 调整维度 (1, num_landmarks, C, 45, 45)
            state_reshaped = state.permute(2, 0, 1, 3, 4).to(device)  # (1, num_landmarks, C, 45, 45)
            #
            # 通过网络获取Q值
            q_values = self.q_network(state_reshaped)  # (1, num_landmarks, 4)
            q_values = q_values.squeeze(0)  # (num_landmarks, 4)
            if reachbound is not None:
                q_values[reachbound]=-1e9

            actions = torch.argmax(q_values, dim=1)  # (num_landmarks,)
             
            return actions
    
    def update_network(self, batch, gamma):
        """更新Q网络"""
        states, actions, rewards, next_states, all_next_states, all_rewards,dones,reachbound= batch
        
        # 移动到设备
        states = states.to(self.device)  # (batch, num_landmarks, C, 45, 45)
        actions = actions.to(self.device)  # (batch, num_landmarks)
        rewards = rewards.to(self.device)  # (batch, num_landmarks)
        all_next_states = all_next_states.to(self.device)  # (batch, num_landmarks, C, 4, 45, 45)
        all_rewards=all_rewards.to(self.device)  # (batch, num_landmarks, 4)
        dones = dones.to(self.device)  # (batch, num_landmarks)
        reachbound =reachbound.to(self.device)  # (batch, num_landmarks, 4)
        dones = dones.unsqueeze(2).repeat(1, 1, 4) # (batch, num_landmarks, 4)
        # 计算当前Q值（所有动作）
        current_q_all = self.q_network(states)  # (batch, num_landmarks, 4)
        
        # 计算目标Q值
        with torch.no_grad():
            
            batch_size, num_landmarks, C, num_actions, H, W = all_next_states.shape
            # 预测所有动作的下一状态的q值
            target_q = []
            for i in range(num_actions):
                action_next_state = all_next_states[:, :, :, i, :, :].view(batch_size, num_landmarks, C, H, W)
                #当reachbound[b, a, i]==True时all_next_states[b, a, :, i, :, :]全0
                next_q_flat = self.target_network(action_next_state)
                next_q_flat[reachbound]=-1e9#不影响max的结果
                next_q_max = next_q_flat.max(dim=2).values
                target_q.append(next_q_max)
            target_q = torch.stack(target_q, dim=2)  # (batch, num_landmarks, 4)

        target_q = all_rewards + gamma * target_q * (~dones).float()
        # 计算损失（所有动作）
        loss = self.loss_fn(current_q_all, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
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
        #用tqdm来追踪当前buffer的填充进度
        pbar = tqdm(total=self.config['pre_sample_size'], desc="Pre-sampling")
        while len(self.buffer) < self.config['pre_sample_size']:
            state = self.env.reset()
            reachbound = None
            for step in range(self.config['episode_steps']):
                action = self.agent.select_action(state, reachbound, explore=True)
                next_state, reward, done, all_next_states, all_rewards, reachbound = self.env.step(action, mode='train')

                # 存储元组（包含所有动作的可能下一状态）
                self.buffer.add_step((state, action, reward, next_state, all_next_states, all_rewards, done, reachbound))

                state = next_state
                
                if all(done):
                    break

            pbar.update(len(self.buffer) - pbar.n)

        pbar.close()
        print(f"Pre-sampling complete. Buffer size: {len(self.buffer)}")
    
    def train(self):
        """训练主循环"""
        self.pre_sample()
        print("Starting training...")
        
        for episode in range(1, self.config['max_episodes'] + 1):
            state = self.env.reset()
            total_reward = 0
            episode_losses = []
            reachbound=None
            for step in range(self.config['episode_steps']):
                action = self.agent.select_action(state,reachbound,explore=True)
                next_state, reward, done, all_next_states, all_rewards,reachbound= self.env.step(action, mode="train")
                total_reward += reward.sum().item()
                
                # 存储元组
                self.buffer.add_step((state, action, reward, next_state, all_next_states, all_rewards, done, reachbound))

                # 定期训练
                self.update_counter += 1
                if self.update_counter % 4 == 0 and len(self.buffer) >= self.config['batch_size']:
                    batch = self.buffer.sample(self.config['batch_size'])
                    loss = self.agent.update_network(batch, self.config['gamma'])
                    episode_losses.append(loss)
                for i in range(self.config['num_landmarks']):
                    if not done[i]:
                        state[i] = next_state[i]

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
                  f"Avg Loss: {avg_loss:.16f}, "
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
        reachbound=None
        
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
            action = self.agent.select_action(state,reachbound, explore=False,randomchoose=False)
            next_state, reward, done, reachbound = self.env.step(action)
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
    

import matplotlib.pyplot as plt
def img_plot(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()
def img_plot_sequence(img_seq, titles=None):
    num_imgs = img_seq.shape[0]
    fig, axes = plt.subplots(1, num_imgs, figsize=(15, 5))
    for i in range(num_imgs):
        axes[i].imshow(img_seq[i])
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.show()


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
