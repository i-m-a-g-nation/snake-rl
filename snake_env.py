"""Snake RL 环境，兼容 Gymnasium API。"""

import numpy as np
from collections import deque
from snake_game import SnakeGame

# 尝试导入 gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

# 根据是否有 gymnasium 决定基类
if HAS_GYM:
    _BaseEnv = gym.Env
else:
    _BaseEnv = object


class SnakeEnv(_BaseEnv):
    """
    Gymnasium 风格的 Snake 环境。
    如果 gymnasium 已安装，则继承 gym.Env；否则自实现同名接口。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size: int = 20, seed: int = None, state_mode: str = "basic17", render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.seed_val = seed
        self.state_mode = state_mode
        self.render_mode = render_mode
        self.game = SnakeGame(grid_size=grid_size, seed=seed)
        self.steps_since_food = 0
        self.prev_distance = 0.0
        self.state_history = deque(maxlen=100)
        self.no_food_timeout_count = 0
        self.repeat_penalty_count = 0

        # 状态维度
        self._state_dim = self.game.get_state_dim(state_mode)
        self._is_grid = (state_mode == "grid")

        # 定义 action_space 和 observation_space
        if HAS_GYM:
            self.action_space = spaces.Discrete(3)
            if self._is_grid:
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0,
                    shape=self._state_dim,
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self._state_dim,),
                    dtype=np.float32,
                )

    @property
    def no_food_step_limit(self) -> int:
        """动态步数限制: max(100, len(snake) * 20)"""
        return max(100, len(self.game.snake) * 20)

    def reset(self, seed=None, options=None):
        """重置环境。返回: (observation, info)"""
        if seed is not None:
            self.seed_val = seed
        self.game = SnakeGame(grid_size=self.grid_size, seed=self.seed_val)
        self.steps_since_food = 0
        self.prev_distance = self.game.distance_to_food()
        self.state_history.clear()

        obs = np.array(self.game.get_state(self.state_mode), dtype=np.float32)
        action_mask = np.array(self.game.get_action_mask(), dtype=bool)
        info = {"score": self.game.score, "steps": self.game.steps, "action_mask": action_mask}
        return obs, info

    def _get_state_key(self):
        """获取状态键，用于重复检测。"""
        head = self.game.snake[0]
        return (head, self.game.direction, self.game.food)

    def step(self, action):
        """
        执行一步动作。
        返回: (observation, reward, terminated, truncated, info)
        """
        # 确保 action 是 Python int
        action = int(action)

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
                new_distance = self.game.distance_to_food()
                if new_distance < old_distance:
                    reward += 0.02
                elif new_distance > old_distance:
                    reward -= 0.02
                self.prev_distance = new_distance

            state_key = self._get_state_key()
            self.state_history.append(state_key)
            repeat_count = sum(1 for s in self.state_history if s == state_key)
            if repeat_count >= 3:
                reward -= 0.2
                repeat_penalty = True
                self.repeat_penalty_count += 1

            free_space = self.game.get_free_space_after_action(0)
            if free_space < 0.05:
                reward -= 0.5

            if self.state_mode == "reachable23":
                if not self.game.get_tail_reachable_after_action(action):
                    reward -= 0.2
                if not self.game.get_food_reachable_after_action(action):
                    reward -= 0.1

            if self.steps_since_food >= self.no_food_step_limit:
                terminated = True
                reward -= 10.0
                no_food_timeout = True
                self.no_food_timeout_count += 1

        truncated = no_food_timeout

        death_reason = None
        if terminated:
            if no_food_timeout:
                death_reason = "no_food_timeout"
            elif result["death_reason"]:
                death_reason = result["death_reason"]
            else:
                death_reason = "unknown"

        obs = np.array(self.game.get_state(self.state_mode), dtype=np.float32)
        action_mask = np.array(self.game.get_action_mask(), dtype=bool)
        info = {
            "score": self.game.score,
            "steps": self.game.steps,
            "ate_food": result["ate_food"],
            "no_food_timeout": no_food_timeout,
            "repeat_penalty": repeat_penalty,
            "death_reason": death_reason,
            "action_mask": action_mask,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """终端渲染。"""
        print(self.game.render_terminal())

    def close(self):
        """关闭环境（无资源需释放）。"""
        pass

    @property
    def observation_space_dim(self) -> int:
        return self._state_dim

    @property
    def action_space_dim(self) -> int:
        return 3
