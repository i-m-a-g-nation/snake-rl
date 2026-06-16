"""DQN 神经网络模型。"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q-Network: 多层感知机 (MLP)。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回每个动作的 Q 值。"""
        return self.net(x)
