"""更强的启发式规划器教师，用于 Expert Iteration-lite。"""

import random
from collections import deque
from typing import Optional, Tuple, List, Set

from snake_game import (
    SnakeGame, DIRECTIONS, DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT,
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
)


def bfs_path(start: Tuple[int, int], target: Tuple[int, int], blocked: Set[Tuple[int, int]], grid_size: int) -> Optional[List[Tuple[int, int]]]:
    """BFS 找最短路径。返回路径列表（含 target，不含 start），或 None。"""
    if start == target:
        return []
    if start in blocked or target in blocked:
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
                if (nr, nc) not in visited and (nr, nc) not in blocked:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def bfs_reachable_count(start: Tuple[int, int], blocked: Set[Tuple[int, int]], grid_size: int) -> int:
    """计算从 start 出发的可达空间大小。"""
    r, c = start
    if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
        return 0
    if start in blocked:
        return 0

    visited = set()
    visited.add(start)
    queue = deque([start])

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in visited and (nr, nc) not in blocked:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return len(visited)


def is_reachable(start: Tuple[int, int], target: Tuple[int, int], blocked: Set[Tuple[int, int]], grid_size: int) -> bool:
    """BFS 判断是否能从 start 到达 target。"""
    if start == target:
        return True
    r, c = start
    if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
        return False
    if start in blocked or target in blocked:
        return False

    visited = set()
    visited.add(start)
    queue = deque([start])

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) == target:
                    return True
                if (nr, nc) not in visited and (nr, nc) not in blocked:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def simulate_action(game: SnakeGame, action: int) -> Tuple[Tuple[int, int], bool, bool]:
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


def build_blocked_after_action(game: SnakeGame, action: int, would_eat: bool) -> Set[Tuple[int, int]]:
    """模拟动作后构建障碍物集合。"""
    new_dir = game._turn(game.direction, action)
    dr, dc = DIRECTIONS[new_dir]
    head_r, head_c = game.snake[0]
    new_head = (head_r + dr, head_c + dc)

    blocked = set(game.snake)
    blocked.add(new_head)
    if not would_eat:
        blocked.discard(game.snake[-1])
    return blocked


def is_food_path_safe(game: SnakeGame, path_to_food: List[Tuple[int, int]]) -> bool:
    """检查吃完食物后是否安全。"""
    if not path_to_food:
        return False

    grid_size = game.grid_size
    snake_len = len(game.snake)

    # 对于短蛇，只要不会立即困住就行
    if snake_len <= 5:
        return True

    # 模拟走到食物后的状态
    simulated_snake = list(game.snake)
    for pos in path_to_food:
        simulated_snake.insert(0, pos)
        if pos != game.food:
            simulated_snake.pop()

    blocked = set(simulated_snake)
    head_after = path_to_food[-1]

    # 检查空间是否足够
    space = bfs_reachable_count(head_after, blocked, grid_size)

    # 空间要求放宽：只要大于蛇长即可
    return space >= snake_len


def find_safe_food_path(game: SnakeGame) -> Optional[List[Tuple[int, int]]]:
    """找安全的食物路径。"""
    if game.food is None:
        return None

    head = game.snake[0]
    food = game.food
    blocked = set(game.snake)

    path = bfs_path(head, food, blocked, game.grid_size)
    if path and is_food_path_safe(game, path):
        return path
    return None


def follow_tail_strategy(game: SnakeGame) -> Optional[int]:
    """跟随尾巴策略。"""
    head = game.snake[0]
    tail = game.snake[-1]
    blocked = set(game.snake)

    # 尾巴会移走，所以目标是尾巴的前一个位置
    if len(game.snake) >= 2:
        target = game.snake[-2]
    else:
        target = tail

    path = bfs_path(head, target, blocked, game.grid_size)
    if path and len(path) > 0:
        # 计算第一步方向
        next_pos = path[0]
        dr = next_pos[0] - head[0]
        dc = next_pos[1] - head[1]

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

        diff = (target_dir - game.direction) % 4
        if diff == 0:
            return ACTION_STRAIGHT
        elif diff == 1:
            return ACTION_RIGHT
        elif diff == 3:
            return ACTION_LEFT

    return None


def score_action(game: SnakeGame, action: int) -> float:
    """评分一个动作。"""
    new_head, would_die, would_eat = simulate_action(game, action)

    if would_die:
        return -10000

    blocked = build_blocked_after_action(game, action, would_eat)
    area = bfs_reachable_count(new_head, blocked, game.grid_size)
    snake_len = len(game.snake)

    score = 0.0

    # 可达空间
    score += area * 2

    # 尾部可达性
    tail = game.snake[-1]
    if would_eat:
        target_tail = tail
    else:
        if len(game.snake) >= 2:
            target_tail = game.snake[-2]
        else:
            target_tail = tail

    tail_ok = is_reachable(new_head, target_tail, blocked, game.grid_size)
    if tail_ok:
        score += 200

    # 食物距离
    if game.food is not None:
        dist = abs(new_head[0] - game.food[0]) + abs(new_head[1] - game.food[1])
        score -= dist * 3

    # 安全吃到食物
    if would_eat:
        after_blocked = set(game.snake)
        after_blocked.add(new_head)
        after_area = bfs_reachable_count(new_head, after_blocked, game.grid_size)
        # 对短蛇放宽要求
        if snake_len <= 5 or after_area >= snake_len:
            score += 1000
        else:
            score += 100  # 仍然尝试吃

    # 检查是否会形成小陷阱
    if area < snake_len:
        score -= 300

    return score


def choose_expert_action(game: SnakeGame) -> int:
    """
    专家规划器选择动作。
    策略：
    1. 如果有安全路径到食物，沿路径走
    2. 否则跟随尾巴
    3. 否则选择最高分动作
    """
    if game.done:
        return ACTION_STRAIGHT

    head = game.snake[0]

    # 策略 1: 安全食物路径
    safe_path = find_safe_food_path(game)
    if safe_path and len(safe_path) > 0:
        next_pos = safe_path[0]
        dr = next_pos[0] - head[0]
        dc = next_pos[1] - head[1]

        if dr == -1 and dc == 0:
            target_dir = DIR_UP
        elif dr == 1 and dc == 0:
            target_dir = DIR_DOWN
        elif dr == 0 and dc == -1:
            target_dir = DIR_LEFT
        elif dr == 0 and dc == 1:
            target_dir = DIR_RIGHT
        else:
            target_dir = game.direction

        diff = (target_dir - game.direction) % 4
        if diff == 0:
            return ACTION_STRAIGHT
        elif diff == 1:
            return ACTION_RIGHT
        elif diff == 3:
            return ACTION_LEFT

    # 策略 2: 跟随尾巴
    tail_action = follow_tail_strategy(game)
    if tail_action is not None:
        # 检查跟随尾巴是否安全
        new_head, would_die, _ = simulate_action(game, tail_action)
        if not would_die:
            return tail_action

    # 策略 3: 选择最高分动作
    best_action = ACTION_STRAIGHT
    best_score = -float("inf")

    for action in range(3):
        s = score_action(game, action)
        if s > best_score:
            best_score = s
            best_action = action

    return best_action


def run_episode_strong(game: SnakeGame, max_steps: int = 2000):
    """运行一局，返回数据。"""
    game.reset()
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    steps_since_food = 0

    while not game.done and game.steps < max_steps:
        state = game.get_state("basic17")
        action = choose_expert_action(game)
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

        # 动态超时
        if steps_since_food > max(200, len(game.snake) * 30):
            break

    return states, actions, rewards, next_states, dones, game.score, game.steps
