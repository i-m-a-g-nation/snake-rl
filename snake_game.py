"""Snake 游戏纯逻辑，不绑定 RL。"""

import random
from collections import deque
from typing import Optional

# 动作定义: 0=直行, 1=左转, 2=右转
ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2

# 方向: 上右下左 (顺时针)
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

DIRECTIONS = {
    DIR_UP: (-1, 0),
    DIR_RIGHT: (0, 1),
    DIR_DOWN: (1, 0),
    DIR_LEFT: (0, -1),
}


class SnakeGame:
    """Snake 游戏逻辑类。"""

    def __init__(self, grid_size: int = 20, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        """重置游戏状态。"""
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.direction = DIR_RIGHT
        self.score = 0
        self.steps = 0
        self.done = False
        self._place_food()

    def _place_food(self) -> None:
        """随机放置食物，不与蛇身重叠。"""
        empty = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self.snake:
                    empty.append((r, c))
        if not empty:
            self.food = None
            return
        self.food = self.rng.choice(empty)

    def _turn(self, current_dir: int, action: int) -> int:
        """根据动作计算新方向。action: 0=直行, 1=左转, 2=右转。"""
        if action == ACTION_STRAIGHT:
            return current_dir
        elif action == ACTION_LEFT:
            return (current_dir - 1) % 4
        else:  # ACTION_RIGHT
            return (current_dir + 1) % 4

    def step(self, action: int) -> dict:
        """
        执行一步动作。
        返回: {"reward": float, "done": bool, "score": int, "ate_food": bool}
        """
        if self.done:
            return {"reward": 0.0, "done": True, "score": self.score, "ate_food": False}

        # 计算新方向 (不允许直接反向)
        new_dir = self._turn(self.direction, action)
        dr, dc = DIRECTIONS[new_dir]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # 碰撞检测: 撞墙
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            self.done = True
            return {"reward": -10.0, "done": True, "score": self.score, "ate_food": False}

        # 碰撞检测: 撞自己 (除了尾巴，因为尾巴会移走)
        tail = self.snake[-1]
        if new_head in self.snake and new_head != tail:
            self.done = True
            return {"reward": -10.0, "done": True, "score": self.score, "ate_food": False}

        # 移动蛇
        self.direction = new_dir
        self.snake.appendleft(new_head)
        self.steps += 1

        # 检查是否吃到食物
        ate_food = False
        if new_head == self.food:
            self.score += 1
            ate_food = True
            self._place_food()
            reward = 10.0
        else:
            self.snake.pop()
            reward = -0.01  # 普通移动惩罚

        return {"reward": reward, "done": False, "score": self.score, "ate_food": ate_food}

    def get_state(self) -> list:
        """
        获取低维状态表示 (11维):
        [danger_straight, danger_left, danger_right,
         dir_up, dir_down, dir_left, dir_right,
         food_left, food_right, food_up, food_down]
        """
        head_r, head_c = self.snake[0]

        # 当前方向对应的三个方向: 直行, 左转, 右转
        straight_dir = self.direction
        left_dir = (self.direction - 1) % 4
        right_dir = (self.direction + 1) % 4

        def is_danger(d):
            dr, dc = DIRECTIONS[d]
            nr, nc = head_r + dr, head_c + dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
                return 1
            if (nr, nc) in self.snake:
                return 1
            return 0

        danger_straight = is_danger(straight_dir)
        danger_left = is_danger(left_dir)
        danger_right = is_danger(right_dir)

        # 方向 one-hot
        dir_up = 1 if self.direction == DIR_UP else 0
        dir_down = 1 if self.direction == DIR_DOWN else 0
        dir_left = 1 if self.direction == DIR_LEFT else 0
        dir_right = 1 if self.direction == DIR_RIGHT else 0

        # 食物相对位置
        if self.food is None:
            food_left = food_right = food_up = food_down = 0
        else:
            fr, fc = self.food
            food_left = 1 if fc < head_c else 0
            food_right = 1 if fc > head_c else 0
            food_up = 1 if fr < head_r else 0
            food_down = 1 if fr > head_r else 0

        return [
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down,
        ]

    def get_state_dim(self) -> int:
        """返回状态维度。"""
        return 11

    def get_action_dim(self) -> int:
        """返回动作维度。"""
        return 3

    def distance_to_food(self) -> float:
        """计算蛇头到食物的曼哈顿距离。"""
        if self.food is None:
            return 0.0
        hr, hc = self.snake[0]
        fr, fc = self.food
        return abs(hr - fr) + abs(hc - fc)

    def render_terminal(self) -> str:
        """终端字符画渲染，返回字符串。"""
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # 画蛇身
        for i, (r, c) in enumerate(self.snake):
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r][c] = "O" if i == 0 else "o"

        # 画食物
        if self.food:
            fr, fc = self.food
            grid[fr][fc] = "*"

        lines = ["+" + "---" * self.grid_size + "+"]
        for row in grid:
            lines.append("|" + " ".join(f"{ch} " for ch in row) + "|")
        lines.append("+" + "---" * self.grid_size + "+")
        lines.append(f"Score: {self.score}  Steps: {self.steps}")
        return "\n".join(lines)


def absolute_direction_to_relative_action(current_direction: int, target_direction: int) -> int:
    """
    将绝对方向转换为相对动作。
    current_direction: 当前蛇的方向 (DIR_UP=0, DIR_RIGHT=1, DIR_DOWN=2, DIR_LEFT=3)
    target_direction: 目标方向
    返回: ACTION_STRAIGHT=0, ACTION_LEFT=1, ACTION_RIGHT=2
    如果是反方向，返回 ACTION_STRAIGHT (不允许反向移动)
    """
    diff = (target_direction - current_direction) % 4
    if diff == 0:
        return ACTION_STRAIGHT  # 同方向，直行
    elif diff == 1:
        return ACTION_RIGHT  # 右转 90°
    elif diff == 3:
        return ACTION_LEFT  # 左转 90°
    else:  # diff == 2，反方向
        return ACTION_STRAIGHT  # 不允许反向，保持直行
