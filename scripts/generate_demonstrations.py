"""生成示范数据脚本。"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snake_game import SnakeGame
from planner_teacher import run_episode
from utils import ensure_dir


def generate_demonstrations(episodes: int = 1000, state_mode: str = "basic17", output: str = "data/demonstrations/planner_basic17_1000.npz", grid_size: int = 20, seed: int = 42):
    """生成示范数据。"""
    ensure_dir(os.path.dirname(output))

    print(f"生成示范数据...")
    print(f"  Episodes: {episodes}")
    print(f"  State mode: {state_mode}")
    print(f"  Grid size: {grid_size}")
    print(f"  Output: {output}")
    print("-" * 60)

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    scores = []
    steps_list = []
    death_reasons = {"wall_collision": 0, "self_collision": 0, "no_food_timeout": 0}

    for ep in range(1, episodes + 1):
        game = SnakeGame(grid_size=grid_size, seed=seed + ep)
        states, actions, rewards, next_states, dones, score, steps = run_episode(game)

        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_next_states.extend(next_states)
        all_dones.extend(dones)

        scores.append(score)
        steps_list.append(steps)

        # 判断死因
        if game.done:
            if steps >= max(100, len(game.snake) * 20):
                death_reasons["no_food_timeout"] += 1
            elif score > 0:
                death_reasons["self_collision"] += 1
            else:
                death_reasons["wall_collision"] += 1

        if ep % 100 == 0:
            avg = np.mean(scores[-100:])
            print(f"  Episode {ep:5d} | Score: {score:3d} | Avg100: {avg:.1f}")

    avg_score = np.mean(scores)
    max_score = max(scores)
    avg_steps = np.mean(steps_list)

    # 输出统计
    print(f"\n{'=' * 60}")
    print(f"示范数据统计")
    print(f"{'=' * 60}")
    print(f"  平均得分: {avg_score:.2f}")
    print(f"  最高得分: {max_score}")
    print(f"  平均步数: {avg_steps:.1f}")
    print(f"  总样本数: {len(all_states)}")
    print(f"  死因分布: {death_reasons}")

    # 警告
    if avg_score < 20:
        print(f"\n警告: planner avg_score={avg_score:.2f} < 20，示范质量较低！")
        print("建议增加 episodes 或改进 planner。")

    # 保存
    np.savez(
        output,
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_states=np.array(all_next_states, dtype=np.float32),
        dones=np.array(all_dones, dtype=np.float32),
        episode_scores=np.array(scores, dtype=np.int64),
        planner_metadata={
            "episodes": episodes,
            "state_mode": state_mode,
            "grid_size": grid_size,
            "seed": seed,
            "avg_score": avg_score,
            "max_score": max_score,
        },
    )

    print(f"\n示范数据已保存: {output}")
    return avg_score, max_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成示范数据")
    parser.add_argument("--episodes", type=int, default=1000, help="生成局数")
    parser.add_argument("--state-mode", type=str, default="basic17", help="状态模式")
    parser.add_argument("--output", type=str, default="data/demonstrations/planner_basic17_1000.npz", help="输出路径")
    parser.add_argument("--grid-size", type=int, default=20, help="网格大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    generate_demonstrations(args.episodes, args.state_mode, args.output, args.grid_size, args.seed)
