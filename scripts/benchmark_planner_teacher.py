"""评估 planner teacher 的性能。"""

import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_game import SnakeGame
from planner_teacher import run_episode as run_weak_episode
from stronger_planner_teacher import run_episode_strong as run_strong_episode
from utils import ensure_dir


def benchmark(episodes: int = 100, teacher: str = "strong", state_mode: str = "basic17", save_path: str = None, grid_size: int = 20, seed: int = 42):
    """评估 planner teacher。"""
    print(f"评估 Planner Teacher")
    print(f"  Teacher: {teacher}")
    print(f"  Episodes: {episodes}")
    print(f"  State mode: {state_mode}")
    print("-" * 60)

    scores = []
    steps_list = []
    death_reasons = {"wall_collision": 0, "self_collision": 0, "no_food_timeout": 0}
    records = []

    run_fn = run_strong_episode if teacher == "strong" else run_weak_episode

    for ep in range(1, episodes + 1):
        game = SnakeGame(grid_size=grid_size, seed=seed + ep)
        states, actions, rewards, next_states, dones, score, steps = run_fn(game)

        scores.append(score)
        steps_list.append(steps)

        # 判断死因
        if game.done:
            if steps >= max(200 if teacher == "strong" else 100, len(game.snake) * 30 if teacher == "strong" else len(game.snake) * 20):
                death_reasons["no_food_timeout"] += 1
            elif score > 0:
                death_reasons["self_collision"] += 1
            else:
                death_reasons["wall_collision"] += 1

        records.append({
            "episode": ep,
            "score": score,
            "steps": steps,
            "death_reason": "timeout" if steps >= max(200, len(game.snake) * 30) else ("self" if score > 0 else "wall"),
        })

        if ep % 20 == 0:
            avg = np.mean(scores[-20:])
            print(f"  Episode {ep:5d} | Score: {score:3d} | Avg20: {avg:.1f}")

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = max(scores)
    min_score = min(scores)
    avg_steps = np.mean(steps_list)

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"Planner Teacher 评估结果 ({episodes} 局)")
    print(f"{'=' * 60}")
    print(f"  Teacher:     {teacher}")
    print(f"  平均得分:    {avg_score:.2f} +/- {std_score:.2f}")
    print(f"  最高得分:    {max_score}")
    print(f"  最低得分:    {min_score}")
    print(f"  平均步数:    {avg_steps:.1f}")
    print()
    print(f"  死因分布:    {death_reasons}")

    # 判断
    OLD_BEST = 35.40
    print(f"\n{'=' * 60}")
    if avg_score >= OLD_BEST:
        print(f"Teacher avg ({avg_score:.2f}) >= 当前最强 ({OLD_BEST})")
        print("可以继续生成 expert demonstrations。")
    else:
        print(f"Teacher avg ({avg_score:.2f}) < 当前最强 ({OLD_BEST})")
        print("不建议用此 teacher 做 imitation pretraining。")
        print("建议继续使用当前推荐模型: checkpoints/best_model_dueling_basic17.pt")

    # 保存
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\n详细记录已保存: {save_path}")

    return avg_score, std_score, max_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 planner teacher")
    parser.add_argument("--episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--teacher", type=str, default="strong", choices=["weak", "strong"], help="Teacher 类型")
    parser.add_argument("--state-mode", type=str, default="basic17", help="状态模式")
    parser.add_argument("--save-path", type=str, default=None, help="保存路径")
    parser.add_argument("--grid-size", type=int, default=20, help="网格大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    benchmark(args.episodes, args.teacher, args.state_mode, args.save_path, args.grid_size, args.seed)
