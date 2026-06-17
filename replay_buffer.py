"""经验回放缓冲区。"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """固定大小的经验回放缓冲区。"""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask=None,
    ) -> None:
        """存储一条经验。"""
        self.buffer.append((state, action, reward, next_state, done, next_action_mask))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """随机采样一批经验。"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
