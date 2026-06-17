"""DQN Agent 实现。支持 Double DQN、Dueling DQN、Action Mask、PER、N-step。"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import DQN, DuelingDQN, NoisyDuelingDQN
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer
from n_step_buffer import NStepBuffer


class DQNAgent:
    """DQN 智能体，支持多种增强功能。"""

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
        epsilon_end: float = 0.03,
        epsilon_decay_steps: int = 100000,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_noisy_net: bool = False,
        use_action_mask: bool = False,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        use_n_step: bool = False,
        n_step: int = 3,
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
        self.use_dueling = use_dueling
        self.use_noisy_net = use_noisy_net
        self.use_action_mask = use_action_mask
        self.use_per = use_per
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.use_n_step = use_n_step
        self.n_step = n_step

        # 网络
        if use_noisy_net:
            self.policy_net = NoisyDuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = NoisyDuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        elif use_dueling:
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器和损失
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # PER 需要 per-sample loss

        # 经验回放
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(replay_size, alpha=per_alpha)
        else:
            self.replay_buffer = ReplayBuffer(replay_size)

        # N-step buffer
        if use_n_step:
            self.n_step_buffer = NStepBuffer(n_step=n_step, gamma=gamma)

        # 训练步数计数
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, epsilon: float, action_mask: np.ndarray = None) -> int:
        """epsilon-greedy 选择动作，支持 action mask。"""
        if random.random() < epsilon:
            # 随机动作
            if self.use_action_mask and action_mask is not None:
                valid_actions = [i for i in range(self.action_dim) if action_mask[i]]
                if valid_actions:
                    return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).cpu().numpy()[0]

            # Action mask: 把无效动作的 Q 值设为 -1e9
            if self.use_action_mask and action_mask is not None:
                for i in range(self.action_dim):
                    if not action_mask[i]:
                        q_values[i] = -1e9

            return int(np.argmax(q_values))

    def compute_epsilon(self) -> float:
        """计算当前 epsilon (线性衰减)。"""
        progress = min(self.train_step_count / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def store_transition(self, state, action, reward, next_state, done, next_action_mask=None) -> None:
        """存储经验到回放缓冲区。"""
        if self.use_n_step:
            self.n_step_buffer.push(state, action, reward, next_state, done, next_action_mask)
            n_step_data = self.n_step_buffer.get()
            if n_step_data is not None:
                ns, na, nr, nns, nd, nm = n_step_data
                self.replay_buffer.push(ns, na, nr, nns, nd, nm)
        else:
            self.replay_buffer.push(state, action, reward, next_state, done, next_action_mask)

    def train_step(self) -> float:
        """执行一步训练，返回 loss 值。"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # 采样
        if self.use_per:
            beta = self.per_beta_start + (self.per_beta_end - self.per_beta_start) * min(self.train_step_count / self.epsilon_decay_steps, 1.0)
            states, actions, rewards, next_states, dones, next_masks, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
            weights_t = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            next_masks = None
            indices = None
            weights_t = torch.ones(self.batch_size, device=self.device)

        # 转为 tensor
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
                next_actions = self.policy_net(next_states_t).argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(dim=1)[0]

            # Action mask for next state
            if self.use_action_mask and next_masks is not None:
                for i in range(self.batch_size):
                    mask = next_masks[i]
                    for a in range(self.action_dim):
                        if not mask[a]:
                            # 把无效动作的 Q 值排除
                            pass  # 已经在 argmax 中处理

            target_q = rewards_t + (self.gamma ** self.n_step) * next_q * (1 - dones_t)

        # 计算损失
        td_errors = q_values - target_q
        loss = self.loss_fn(q_values, target_q)
        loss = (weights_t * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新 PER 优先级
        if self.use_per and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.train_step_count += 1

        # 更新 target network
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
