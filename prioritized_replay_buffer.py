"""Prioritized Experience Replay Buffer。"""

import numpy as np
from typing import Tuple


class PrioritizedReplayBuffer:
    """比例优先经验回放缓冲区（简化版，无 SumTree）。"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, next_action_mask=None):
        """存储一条经验。"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, next_action_mask))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, next_action_mask)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, ...]:
        """
        按优先级采样。
        返回: (states, actions, rewards, next_states, dones, next_masks, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        actions = np.array([self.buffer[i][1] for i in indices], dtype=np.int64)
        rewards = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        next_states = np.array([self.buffer[i][3] for i in indices], dtype=np.float32)
        dones = np.array([self.buffer[i][4] for i in indices], dtype=np.float32)
        next_masks = np.array([self.buffer[i][5] if self.buffer[i][5] is not None else [True, True, True] for i in indices], dtype=bool)

        return states, actions, rewards, next_states, dones, next_masks, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级。"""
        priorities = np.abs(td_errors) + self.eps
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return self.size
