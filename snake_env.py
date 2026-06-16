"""Snake RL 环境，兼容 Gymnasium 风格接口。"""

import numpy as np
from snake_game import SnakeGame


class SnakeEnv:
    """
    Gymnasium 风格的 Snake 环境。
    即使没有 gymnasium 库，也提供 reset/step/render/close 接口。
    """

    def __init__(self, grid_size: int = 20, max_steps: int = 1000, seed: int = None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed_val = seed
        self.game = SnakeGame(grid_size=grid_size, seed=seed)
        self.steps_since_food = 0
        self.prev_distance = 0.0

    def reset(self, seed: int = None):
        """
        重置环境。
        返回: (observation, info)
        """
        if seed is not None:
            self.seed_val = seed
        self.game = SnakeGame(grid_size=self.grid_size, seed=self.seed_val)
        self.steps_since_food = 0
        self.prev_distance = self.game.distance_to_food()

        obs = np.array(self.game.get_state(), dtype=np.float32)
        info = {"score": self.game.score, "steps": self.game.steps}
        return obs, info

    def step(self, action: int):
        """
        执行一步动作。
        返回: (observation, reward, terminated, truncated, info)
        """
        old_distance = self.prev_distance

        result = self.game.step(action)
        reward = result["reward"]
        terminated = result["done"]
        self.steps_since_food += 1

        # 距离奖励: 离食物更近/更远
        if not terminated and result["ate_food"]:
            self.steps_since_food = 0
        elif not terminated:
            new_distance = self.game.distance_to_food()
            if new_distance < old_distance:
                reward += 0.1
            elif new_distance > old_distance:
                reward -= 0.1
            self.prev_distance = new_distance

        # 最大步数截断
        truncated = False
        if self.steps_since_food >= self.max_steps:
            truncated = True
            terminated = True
            reward -= 5.0  # 无限绕圈惩罚

        obs = np.array(self.game.get_state(), dtype=np.float32)
        info = {
            "score": self.game.score,
            "steps": self.game.steps,
            "ate_food": result["ate_food"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """终端渲染。"""
        print(self.game.render_terminal())

    def close(self):
        """关闭环境（无资源需释放）。"""
        pass

    @property
    def observation_space_dim(self) -> int:
        return self.game.get_state_dim()

    @property
    def action_space_dim(self) -> int:
        return self.game.get_action_dim()
