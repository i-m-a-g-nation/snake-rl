"""DQN Agent 实现。支持 Double DQN。"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN 智能体，支持 Double DQN。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr: float = 5e-4,
        batch_size: int = 128,
        replay_size: int = 100000,
        target_update_interval: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 50000,
        use_double_dqn: bool = True,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.use_double_dqn = use_double_dqn

        # 网络
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器和损失
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # 经验回放
        self.replay_buffer = ReplayBuffer(replay_size)

        # 训练步数计数
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """epsilon-greedy 选择动作。"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def compute_epsilon(self) -> float:
        """计算当前 epsilon (线性衰减)。"""
        progress = min(self.train_step_count / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """存储经验到回放缓冲区。"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """执行一步训练，返回 loss 值。"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值
        q_values = self.policy_net(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # 目标 Q 值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 用 policy_net 选动作，target_net 评估
                next_actions = self.policy_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # 标准 DQN
                next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        """保存模型。"""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step_count": self.train_step_count,
        }, path)

    def load(self, path: str) -> None:
        """加载模型。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step_count = checkpoint["train_step_count"]
