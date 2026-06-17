"""启发式规划器教师，用于生成示范数据。"""

import random
from collections import deque
from typing import Optional, Tuple, List

from snake_game import (
    SnakeGame, DIRECTIONS, DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT,
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
)


def bfs_path(start: Tuple[int, int], target: Tuple[int, int], obstacles: set, grid_size: int) -> Optional[List[Tuple[int, int]]]:
    """BFS 找最短路径。返回路径列表（不含 start），或 None。"""
    if start == target:
        return []
    if start in obstacles or target in obstacles:
        return None

    visited = set()
    visited.add(start)
    queue = deque([(start, [])])

    while queue:
        (r, c), path = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) == target:
                    return path + [(nr, nc)]
                if (nr, nc) not in visited and (nr, nc) not in obstacles:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def reachable_area(start: Tuple[int, int], obstacles: set, grid_size: int) -> int:
    """计算从 start 出发的可达空间大小。"""
    if start in obstacles:
        return 0
    r, c = start
    if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
        return 0

    visited = set()
    visited.add(start)
    queue = deque([start])

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in visited and (nr, nc) not in obstacles:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return len(visited)


def is_reachable(start: Tuple[int, int], target: Tuple[int, int], obstacles: set, grid_size: int) -> bool:
    """BFS 判断是否能从 start 到达 target。"""
    if start == target:
        return True
    if start in obstacles or target in obstacles:
        return False

    visited = set()
    visited.add(start)
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) == target:
                    return True
                if (nr, nc) not in visited and (nr, nc) not in obstacles:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def simulate_action(game: SnakeGame, action: int):
    """模拟动作，返回 (new_head, would_die, would_eat)。"""
    new_dir = game._turn(game.direction, action)
    dr, dc = DIRECTIONS[new_dir]
    head_r, head_c = game.snake[0]
    new_head = (head_r + dr, head_c + dc)

    r, c = new_head
    if r < 0 or r >= game.grid_size or c < 0 or c >= game.grid_size:
        return new_head, True, False

    tail = game.snake[-1]
    if new_head in game.snake and new_head != tail:
        return new_head, True, False

    would_eat = (new_head == game.food)
    return new_head, False, would_eat


def build_obstacles_after_action(game: SnakeGame, action: int, would_eat: bool):
    """模拟动作后构建障碍物。"""
    new_dir = game._turn(game.direction, action)
    dr, dc = DIRECTIONS[new_dir]
    head_r, head_c = game.snake[0]
    new_head = (head_r + dr, head_c + dc)

    obstacles = set(game.snake)
    obstacles.add(new_head)
    if not would_eat:
        obstacles.discard(game.snake[-1])
    return obstacles


def direction_to_action(game: SnakeGame, target_dir: int) -> int:
    """将目标方向转换为相对动作。"""
    diff = (target_dir - game.direction) % 4
    if diff == 0:
        return ACTION_STRAIGHT
    elif diff == 1:
        return ACTION_RIGHT
    elif diff == 3:
        return ACTION_LEFT
    else:
        return ACTION_STRAIGHT


def next_position_in_path(game: SnakeGame, path: List[Tuple[int, int]]):
    """根据路径计算下一步方向和动作。"""
    if not path:
        return None

    head = game.snake[0]
    next_pos = path[0]
    dr = next_pos[0] - head[0]
    dc = next_pos[1] - head[1]

    # 计算目标方向
    if dr == -1 and dc == 0:
        target_dir = DIR_UP
    elif dr == 1 and dc == 0:
        target_dir = DIR_DOWN
    elif dr == 0 and dc == -1:
        target_dir = DIR_LEFT
    elif dr == 0 and dc == 1:
        target_dir = DIR_RIGHT
    else:
        return None

    return direction_to_action(game, target_dir)


def choose_planner_action(game: SnakeGame) -> int:
    """
    启发式规划器选择动作。
    策略：
    1. 如果有安全路径到食物，沿路径走
    2. 否则，选择能到达尾巴或最大可达空间的动作
    """
    if game.done:
        return ACTION_STRAIGHT

    grid_size = game.grid_size
    head = game.snake[0]
    tail = game.snake[-1]
    food = game.food
    snake_len = len(game.snake)

    # 构建当前障碍物
    obstacles = set(game.snake)

    # 尝试找 BFS 路径到食物
    if food is not None:
        path_to_food = bfs_path(head, food, obstacles, grid_size)

        if path_to_food:
            # 检查吃完食物后是否安全
            # 模拟走到食物
            sim_obstacles = set(game.snake)
            sim_obstacles.add(path_to_food[-1])
            # 吃完后尾巴不移走
            tail_ok = is_reachable(path_to_food[-1], tail, sim_obstacles, grid_size)

            if tail_ok:
                # 安全路径，沿路径走
                action = next_position_in_path(game, path_to_food)
                if action is not None:
                    return action

    # 没有安全路径到食物，选择最优动作
    best_action = ACTION_STRAIGHT
    best_score = -float("inf")

    for action in range(3):
        new_head, would_die, would_eat = simulate_action(game, action)

        if would_die:
            score = -1000
        else:
            obstacles = build_obstacles_after_action(game, action, would_eat)
            area = reachable_area(new_head, obstacles, grid_size)

            # 检查尾部可达性
            if would_eat:
                target_tail = tail
            else:
                if len(game.snake) >= 2:
                    target_tail = game.snake[-2]
                else:
                    target_tail = tail

            tail_ok = is_reachable(new_head, target_tail, obstacles, grid_size)

            # 评分
            score = 0

            # 能安全吃到食物
            if would_eat:
                # 检查吃完后空间
                after_obstacles = set(game.snake)
                after_obstacles.add(new_head)
                after_area = reachable_area(new_head, after_obstacles, grid_size)
                if after_area >= snake_len + 3:
                    score += 400
                else:
                    score += 50  # 能吃但危险

            # 尾部可达
            if tail_ok:
                score += 200

            # 可达空间
            score += area / (grid_size * grid_size) * 100

            # 食物距离（曼哈顿）
            if food is not None:
                dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
                score -= dist * 2  # 越近越好

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def run_episode(game: SnakeGame, max_steps: int = 1000):
    """运行一局，返回 (states, actions, rewards, next_states, dones, score)。"""
    game.reset()
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    steps_since_food = 0

    while not game.done and game.steps < max_steps:
        state = game.get_state("basic17")
        action = choose_planner_action(game)
        result = game.step(action)

        next_state = game.get_state("basic17")
        done = result["done"]

        states.append(state)
        actions.append(action)
        rewards.append(result["reward"])
        next_states.append(next_state)
        dones.append(1.0 if done else 0.0)

        steps_since_food += 1
        if result["ate_food"]:
            steps_since_food = 0

        # 防止无限循环
        if steps_since_food > max(100, len(game.snake) * 20):
            break

    return states, actions, rewards, next_states, dones, game.score, game.steps
