"""评估模型死亡原因分布。"""

import argparse
import os
import sys

import numpy as np

from snake_env import SnakeEnv
from agent import DQNAgent
from utils import ensure_dir, save_train_log


def evaluate_deaths(model_path: str, episodes: int = 100, grid_size: int = 20, seed: int = 3000):
    """评估模型死亡原因分布。"""
    env = SnakeEnv(grid_size=grid_size, seed=seed)
    agent = DQNAgent(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
    )
    agent.load(model_path)
    agent.policy_net.eval()

    print(f"模型: {model_path}")
    print(f"设备: {agent.device}")
    print(f"评估局数: {episodes}")
    print("-" * 60)

    records = []
    scores = []
    steps_list = []
    death_counts = {
        "wall_collision": 0,
        "self_collision": 0,
        "no_food_timeout": 0,
        "unknown": 0,
    }

    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=seed + ep)
        done = False

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = info["score"]
        steps = info["steps"]
        death_reason = info.get("death_reason", "unknown")

        scores.append(score)
        steps_list.append(steps)

        if death_reason in death_counts:
            death_counts[death_reason] += 1
        else:
            death_counts["unknown"] += 1

        records.append({
            "episode": ep,
            "score": score,
            "steps": steps,
            "death_reason": death_reason,
        })

    env.close()

    # 统计
    avg_score = np.mean(scores)
    max_score = max(scores)
    avg_steps = np.mean(steps_list)
    total = episodes

    wall_rate = death_counts["wall_collision"] / total * 100
    self_rate = death_counts["self_collision"] / total * 100
    timeout_rate = death_counts["no_food_timeout"] / total * 100
    unknown_rate = death_counts["unknown"] / total * 100

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"评估结果 ({episodes} 局)")
    print(f"{'=' * 60}")
    print(f"  平均得分:     {avg_score:.2f}")
    print(f"  最高得分:     {max_score}")
    print(f"  平均步数:     {avg_steps:.1f}")
    print()
    print(f"  撞墙次数:     {death_counts['wall_collision']}  ({wall_rate:.1f}%)")
    print(f"  撞自己次数:   {death_counts['self_collision']}  ({self_rate:.1f}%)")
    print(f"  超时次数:     {death_counts['no_food_timeout']}  ({timeout_rate:.1f}%)")
    print(f"  未知次数:     {death_counts['unknown']}  ({unknown_rate:.1f}%)")

    # 诊断
    print(f"\n{'=' * 60}")
    print("诊断")
    print(f"{'=' * 60}")

    rates = {
        "self_collision": self_rate,
        "wall_collision": wall_rate,
        "no_food_timeout": timeout_rate,
    }
    max_reason = max(rates, key=rates.get)

    if max_reason == "self_collision":
        print("  主要问题: 撞自己 (self_collision)")
        print("  说明: 蛇变长后路径规划不足")
        print("  建议:")
        print("    - 增加 tail_reachable / food_reachable 特征")
        print("    - 增强 flood fill 权重")
        print("    - 延长训练让 agent 学会绕行")
    elif max_reason == "no_food_timeout":
        print("  主要问题: 超时 (no_food_timeout)")
        print("  说明: 仍有绕圈行为，找不到食物")
        print("  建议:")
        print("    - 增强重复状态惩罚")
        print("    - 增加路径奖励 (朝食物方向移动)")
        print("    - 降低 epsilon_end 让策略更确定")
    else:
        print("  主要问题: 撞墙 (wall_collision)")
        print("  说明: 基础避障能力不稳定")
        print("  建议:")
        print("    - 增加更多训练 episodes")
        print("    - 检查 danger 特征是否正确")
        print("    - 增强 wall_collision 惩罚")

    # 保存 CSV
    ensure_dir("logs")
    csv_path = "logs/death_eval.csv"
    save_train_log(csv_path, records)
    print(f"\n详细记录已保存: {csv_path}")

    return {
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_steps": avg_steps,
        "death_counts": death_counts,
        "rates": rates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估模型死亡原因分布")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="模型路径")
    parser.add_argument("--episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--grid-size", type=int, default=20, help="网格大小")
    parser.add_argument("--seed", type=int, default=3000, help="随机种子")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        print("请先运行训练: python train.py --episodes 1000")
        sys.exit(1)

    evaluate_deaths(args.model, args.episodes, args.grid_size, args.seed)
