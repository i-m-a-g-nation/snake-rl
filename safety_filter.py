"""推理阶段安全规划器。只在 inference 时使用，不影响训练。"""

import numpy as np
from collections import deque
from snake_game import SnakeGame, DIRECTIONS, ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT


def _flood_fill_count(game: SnakeGame, start: tuple, obstacles: set) -> int:
    """从 start 出发计算可达空间大小。"""
    r, c = start
    if r < 0 or r >= game.grid_size or c < 0 or c >= game.grid_size:
        return 0
    if start in obstacles:
        return 0

    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < game.grid_size and 0 <= nc < game.grid_size:
                if (nr, nc) not in visited and (nr, nc) not in obstacles:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return len(visited)


def _is_reachable(game: SnakeGame, start: tuple, target: tuple, obstacles: set) -> bool:
    """BFS 判断从 start 是否能到达 target。"""
    if start == target:
        return True
    r, c = start
    if r < 0 or r >= game.grid_size or c < 0 or c >= game.grid_size:
        return False
    if start in obstacles:
        return False

    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < game.grid_size and 0 <= nc < game.grid_size:
                if (nr, nc) == target:
                    return True
                if (nr, nc) not in visited and (nr, nc) not in obstacles:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def _simulate_action(game: SnakeGame, action: int):
    """模拟一步动作，返回 (new_head, new_dir, would_die, would_eat)。"""
    new_dir = game._turn(game.direction, action)
    dr, dc = DIRECTIONS[new_dir]
    head_r, head_c = game.snake[0]
    new_head = (head_r + dr, head_c + dc)

    r, c = new_head
    if r < 0 or r >= game.grid_size or c < 0 or c >= game.grid_size:
        return new_head, new_dir, True, False

    tail = game.snake[-1]
    if new_head in game.snake and new_head != tail:
        return new_head, new_dir, True, False

    would_eat = (new_head == game.food)
    return new_head, new_dir, False, would_eat


def _build_obstacles_after_action(game: SnakeGame, action: int, would_eat: bool):
    """模拟动作后构建障碍物集合。"""
    new_dir = game._turn(game.direction, action)
    dr, dc = DIRECTIONS[new_dir]
    head_r, head_c = game.snake[0]
    new_head = (head_r + dr, head_c + dc)

    obstacles = set(game.snake)
    obstacles.add(new_head)
    if not would_eat:
        obstacles.discard(game.snake[-1])
    return obstacles


def choose_safe_action(game: SnakeGame, q_values: np.ndarray, config: dict = None):
    """
    推理阶段安全规划器。
    输入:
        game: 当前 SnakeGame 状态
        q_values: DQN 输出的 Q 值 (shape: [action_dim])
        config: 配置参数
    输出:
        (selected_action, debug_info)
    """
    if config is None:
        config = {}

    safety_margin = config.get("safety_margin", 3)
    min_reachable_ratio = config.get("min_reachable_ratio", 0.1)

    action_dim = len(q_values)
    snake_len = len(game.snake)
    total_cells = game.grid_size * game.grid_size

    # 按 Q 值从高到低排序动作
    sorted_actions = np.argsort(q_values)[::-1]

    best_action = int(sorted_actions[0])  # 默认回退到最大 Q 动作
    best_score = -float("inf")

    debug_info = {
        "override": False,
        "override_reason": "",
        "action_scores": {},
    }

    for action in sorted_actions:
        action = int(action)
        new_head, new_dir, would_die, would_eat = _simulate_action(game, action)

        # 立即死亡
        if would_die:
            debug_info["action_scores"][action] = {
                "q": float(q_values[action]),
                "immediate_death": True,
                "reachable_area": 0,
                "tail_reachable": False,
                "safe": False,
            }
            continue

        # 构建障碍物
        obstacles = _build_obstacles_after_action(game, action, would_eat)

        # 计算可达空间
        reachable_area = _flood_fill_count(game, new_head, obstacles)
        reachable_ratio = reachable_area / total_cells

        # 检查是否能到达尾巴
        tail = game.snake[-1]
        if would_eat:
            # 吃食物后尾巴不移走
            target_tail = tail
        else:
            # 不吃食物，尾巴移走，检查能否到达新尾巴
            if len(game.snake) >= 2:
                target_tail = game.snake[-2]
            else:
                target_tail = tail

        tail_reachable = _is_reachable(game, new_head, target_tail, obstacles)

        # 安全性评分
        safe = True
        reason = ""

        # 检查可达空间是否足够
        if reachable_area < snake_len + safety_margin:
            safe = False
            reason = "low_space"

        # 检查尾部可达性
        if not tail_reachable:
            safe = False
            reason = "tail_unreachable"

        # 综合评分: Q 值 + 安全奖励
        score = float(q_values[action])
        if safe:
            score += 100  # 安全动作优先
        if tail_reachable:
            score += 50
        score += reachable_ratio * 10

        debug_info["action_scores"][action] = {
            "q": float(q_values[action]),
            "immediate_death": False,
            "reachable_area": reachable_area,
            "reachable_ratio": round(reachable_ratio, 3),
            "tail_reachable": tail_reachable,
            "safe": safe,
            "reason": reason,
            "score": round(score, 2),
        }

        if score > best_score:
            best_score = score
            best_action = action

    # 判断是否覆盖了原始 DQN 选择
    dqn_action = int(sorted_actions[0])
    if best_action != dqn_action:
        debug_info["override"] = True
        best_info = debug_info["action_scores"].get(best_action, {})
        debug_info["override_reason"] = best_info.get("reason", "safer_choice")

    return best_action, debug_info
