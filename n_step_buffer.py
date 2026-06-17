"""N-step return buffer。"""

import numpy as np
from collections import deque
from typing import Tuple


class NStepBuffer:
    """N-step return 缓冲区。"""

    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, next_action_mask=None):
        """存储一步经验。"""
        self.buffer.append((state, action, reward, next_state, done, next_action_mask))

    def get(self) -> Tuple:
        """
        获取 n-step return。
        返回: (state, action, n_step_reward, next_state, done, next_action_mask)
        """
        if len(self.buffer) == 0:
            return None

        # 计算 n-step return
        n_step_reward = 0.0
        for i, (state, action, reward, next_state, done, next_action_mask) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * reward
            if done:
                # Episode 提前结束
                return state, action, n_step_reward, next_state, done, next_action_mask

        # 使用最早的 transition
        state, action, _, _, _, _ = self.buffer[0]
        _, _, _, next_state, done, next_action_mask = self.buffer[-1]

        return state, action, n_step_reward, next_state, done, next_action_mask

    def clear(self):
        """清空缓冲区。"""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
