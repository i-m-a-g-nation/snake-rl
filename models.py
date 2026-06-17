"""DQN 神经网络模型。支持 DQN 和 Dueling DQN。"""

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


class DuelingDQN(nn.Module):
    """Dueling DQN: 分离状态价值和动作优势。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim

        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回每个动作的 Q 值。
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        """
        shared_out = self.shared(x)
        value = self.value_stream(shared_out)
        advantage = self.advantage_stream(shared_out)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
