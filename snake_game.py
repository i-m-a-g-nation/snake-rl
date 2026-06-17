"""Snake 游戏纯逻辑，不绑定 RL。支持 flood fill 可达空间计算。"""

import random
from collections import deque
from typing import Optional, Tuple

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
        返回: {"reward", "done", "score", "ate_food", "death_reason"}
        death_reason: "wall_collision" | "self_collision" | None
        """
        if self.done:
            return {"reward": 0.0, "done": True, "score": self.score, "ate_food": False, "death_reason": None}

        new_dir = self._turn(self.direction, action)
        dr, dc = DIRECTIONS[new_dir]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # 碰撞检测: 撞墙
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            self.done = True
            return {"reward": -10.0, "done": True, "score": self.score, "ate_food": False, "death_reason": "wall_collision"}

        # 碰撞检测: 撞自己 (除了尾巴，因为尾巴会移走)
        tail = self.snake[-1]
        if new_head in self.snake and new_head != tail:
            self.done = True
            return {"reward": -10.0, "done": True, "score": self.score, "ate_food": False, "death_reason": "self_collision"}

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
            reward = -0.01

        return {"reward": reward, "done": False, "score": self.score, "ate_food": ate_food, "death_reason": None}

    def _simulate_action(self, action: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        模拟一步动作，不修改游戏状态。
        返回: (new_head, new_direction, would_die)
        """
        new_dir = self._turn(self.direction, action)
        dr, dc = DIRECTIONS[new_dir]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return new_head, new_dir, True

        # 撞自己检测: 需要考虑尾巴是否会在本步移走
        # 如果不吃食物，尾巴会移走，所以尾巴位置不算危险
        tail = self.snake[-1]
        if new_head in self.snake and new_head != tail:
            return new_head, new_dir, True

        return new_head, new_dir, False

    def _flood_fill(self, start: Tuple[int, int], obstacles: set) -> int:
        """
        从 start 位置出发，计算可达空间大小。
        obstacles: 不可通行的位置集合
        """
        if start[0] < 0 or start[0] >= self.grid_size or start[1] < 0 or start[1] >= self.grid_size:
            return 0
        if start in obstacles:
            return 0

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) not in visited and (nr, nc) not in obstacles:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return len(visited)

    def _build_obstacles_after_action(self, action: int):
        """模拟动作后构建障碍物集合，返回 (new_head, would_die, obstacles)。"""
        new_head, new_dir, would_die = self._simulate_action(action)
        if would_die:
            return new_head, True, set()

        obstacles = set(self.snake)
        obstacles.add(new_head)
        # 如果不吃食物，尾巴会移走
        if self.food is not None and new_head != self.food:
            obstacles.discard(self.snake[-1])

        return new_head, False, obstacles

    def get_free_space_after_action(self, action: int) -> float:
        """计算执行某动作后从新蛇头可达的空格比例。"""
        new_head, would_die, obstacles = self._build_obstacles_after_action(action)
        if would_die:
            return 0.0

        total_cells = self.grid_size * self.grid_size
        reachable = self._flood_fill(new_head, obstacles)
        return reachable / total_cells

    def _is_reachable(self, start, target, obstacles) -> bool:
        """BFS 判断从 start 是否能到达 target。"""
        if start[0] < 0 or start[0] >= self.grid_size or start[1] < 0 or start[1] >= self.grid_size:
            return False
        if start in obstacles:
            return False
        if start == target:
            return True

        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) == target:
                        return True
                    if (nr, nc) not in visited and (nr, nc) not in obstacles:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def get_tail_reachable_after_action(self, action: int) -> bool:
        """模拟动作后，从新蛇头是否能到达蛇尾。"""
        new_head, would_die, obstacles = self._build_obstacles_after_action(action)
        if would_die:
            return False

        # 蛇尾位置（移动后尾巴可能移走，所以用当前尾巴的前一个位置作为目标）
        # 实际上我们检查是否能到达蛇尾附近（蛇尾的邻居也算）
        tail = self.snake[-1]
        # 如果不吃食物，尾巴会移走，目标是当前尾巴位置
        if self.food is not None and new_head != self.food:
            # 尾巴会移走，检查能否到达尾巴的前一个位置（即移动后的尾巴）
            if len(self.snake) >= 2:
                target = self.snake[-2]  # 移动后的新尾巴
            else:
                target = tail
        else:
            target = tail

        # 从新蛇头检查能否到达 target
        return self._is_reachable(new_head, target, obstacles)

    def get_food_reachable_after_action(self, action: int) -> bool:
        """模拟动作后，从新蛇头是否能到达食物。"""
        if self.food is None:
            return True

        new_head, would_die, obstacles = self._build_obstacles_after_action(action)
        if would_die:
            return False

        return self._is_reachable(new_head, self.food, obstacles)

    def get_state(self, state_mode: str = "basic17") -> list:
        """
        获取状态表示。
        state_mode: "basic17" 或 "reachable23"
        """
        head_r, head_c = self.snake[0]

        straight_dir = self.direction
        left_dir = (self.direction - 1) % 4
        right_dir = (self.direction + 1) % 4

        def is_danger(d):
            """危险判断: 与 step() 的碰撞逻辑一致（尾巴会移走不算危险）。"""
            dr, dc = DIRECTIONS[d]
            nr, nc = head_r + dr, head_c + dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
                return 1
            tail = self.snake[-1]
            if (nr, nc) in self.snake and (nr, nc) != tail:
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
            dist = 0.0
        else:
            fr, fc = self.food
            food_left = 1 if fc < head_c else 0
            food_right = 1 if fc > head_c else 0
            food_up = 1 if fr < head_r else 0
            food_down = 1 if fr > head_r else 0
            dist = abs(head_r - fr) + abs(head_c - fc)

        # 归一化特征
        max_dist = self.grid_size * 2
        distance_to_food_normalized = dist / max_dist

        max_len = self.grid_size * self.grid_size
        snake_length_normalized = len(self.snake) / max_len

        max_no_food = max(100, len(self.snake) * 20)
        steps_since_food_normalized = min(self.steps / max_no_food, 1.0)

        # Flood fill 可达空间
        free_space_straight = self.get_free_space_after_action(ACTION_STRAIGHT)
        free_space_left = self.get_free_space_after_action(ACTION_LEFT)
        free_space_right = self.get_free_space_after_action(ACTION_RIGHT)

        # basic17: 17 维基础状态
        state = [
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down,
            distance_to_food_normalized,
            snake_length_normalized,
            steps_since_food_normalized,
            free_space_straight, free_space_left, free_space_right,
        ]

        if state_mode == "reachable23":
            # Tail 可达性
            tail_reachable_straight = 1.0 if self.get_tail_reachable_after_action(ACTION_STRAIGHT) else 0.0
            tail_reachable_left = 1.0 if self.get_tail_reachable_after_action(ACTION_LEFT) else 0.0
            tail_reachable_right = 1.0 if self.get_tail_reachable_after_action(ACTION_RIGHT) else 0.0

            # Food 可达性
            food_reachable_straight = 1.0 if self.get_food_reachable_after_action(ACTION_STRAIGHT) else 0.0
            food_reachable_left = 1.0 if self.get_food_reachable_after_action(ACTION_LEFT) else 0.0
            food_reachable_right = 1.0 if self.get_food_reachable_after_action(ACTION_RIGHT) else 0.0

            state.extend([
                tail_reachable_straight, tail_reachable_left, tail_reachable_right,
                food_reachable_straight, food_reachable_left, food_reachable_right,
            ])

        return state

    def get_state_dim(self, state_mode: str = "basic17") -> int:
        """返回状态维度。"""
        if state_mode == "reachable23":
            return 23
        return 17

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
        for i, (r, c) in enumerate(self.snake):
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r][c] = "O" if i == 0 else "o"
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
    """将绝对方向转换为相对动作。"""
    diff = (target_direction - current_direction) % 4
    if diff == 0:
        return ACTION_STRAIGHT
    elif diff == 1:
        return ACTION_RIGHT
    elif diff == 3:
        return ACTION_LEFT
    else:
        return ACTION_STRAIGHT
