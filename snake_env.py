"""Snake RL 环境，兼容 Gymnasium 风格接口。改进版：新奖励、重复检测、动态步数限制。"""

import numpy as np
from collections import deque
from snake_game import SnakeGame


class SnakeEnv:
    """
    Gymnasium 风格的 Snake 环境。
    即使没有 gymnasium 库，也提供 reset/step/render/close 接口。
    """

    def __init__(self, grid_size: int = 20, seed: int = None):
        self.grid_size = grid_size
        self.seed_val = seed
        self.game = SnakeGame(grid_size=grid_size, seed=seed)
        self.steps_since_food = 0
        self.prev_distance = 0.0
        # 重复状态检测
        self.state_history = deque(maxlen=100)
        # 统计
        self.no_food_timeout_count = 0
        self.repeat_penalty_count = 0

    @property
    def no_food_step_limit(self) -> int:
        """动态步数限制: max(100, len(snake) * 20)"""
        return max(100, len(self.game.snake) * 20)

    def reset(self, seed: int = None):
        """重置环境。返回: (observation, info)"""
        if seed is not None:
            self.seed_val = seed
        self.game = SnakeGame(grid_size=self.grid_size, seed=self.seed_val)
        self.steps_since_food = 0
        self.prev_distance = self.game.distance_to_food()
        self.state_history.clear()

        obs = np.array(self.game.get_state(), dtype=np.float32)
        info = {"score": self.game.score, "steps": self.game.steps}
        return obs, info

    def _get_state_key(self):
        """获取状态键，用于重复检测。"""
        head = self.game.snake[0]
        return (head, self.game.direction, self.game.food)

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

        no_food_timeout = False
        repeat_penalty = False

        if not terminated:
            if result["ate_food"]:
                self.steps_since_food = 0
            else:
                # 距离奖励（降低强度）
                new_distance = self.game.distance_to_food()
                if new_distance < old_distance:
                    reward += 0.02
                elif new_distance > old_distance:
                    reward -= 0.02
                self.prev_distance = new_distance

            # 重复状态检测（绕圈惩罚）
            state_key = self._get_state_key()
            self.state_history.append(state_key)
            repeat_count = sum(1 for s in self.state_history if s == state_key)
            if repeat_count >= 3:
                reward -= 0.2
                repeat_penalty = True
                self.repeat_penalty_count += 1

            # 死胡同检测: 可达空间极小
            free_space = self.game.get_free_space_after_action(0)
            if free_space < 0.05:
                reward -= 0.5

            # Tail 可达性惩罚
            if not self.game.get_tail_reachable_after_action(action):
                reward -= 0.2

            # Food 可达性惩罚
            if not self.game.get_food_reachable_after_action(action):
                reward -= 0.1

            # 动态步数限制
            if self.steps_since_food >= self.no_food_step_limit:
                terminated = True
                reward -= 10.0
                no_food_timeout = True
                self.no_food_timeout_count += 1

        truncated = no_food_timeout

        # 确定死亡原因
        death_reason = None
        if terminated:
            if no_food_timeout:
                death_reason = "no_food_timeout"
            elif result["death_reason"]:
                death_reason = result["death_reason"]
            else:
                death_reason = "unknown"

        obs = np.array(self.game.get_state(), dtype=np.float32)
        info = {
            "score": self.game.score,
            "steps": self.game.steps,
            "ate_food": result["ate_food"],
            "no_food_timeout": no_food_timeout,
            "repeat_penalty": repeat_penalty,
            "death_reason": death_reason,
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
