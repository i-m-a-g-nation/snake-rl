"""对比不同训练方法的结果。"""

import os
import csv
import sys

import numpy as np

from utils import ensure_dir, save_train_log


def load_sb3_eval(csv_path: str):
    """加载 SB3 评估 CSV，返回统计信息。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    steps = []
    death_counts = {"wall_collision": 0, "self_collision": 0, "no_food_timeout": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row["score"]))
            steps.append(int(row["steps"]))
            reason = row.get("death_reason", "unknown")
            if reason in death_counts:
                death_counts[reason] += 1

    total = len(scores)
    if total == 0:
        return None

    return {
        "avg_score": round(np.mean(scores), 2),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_steps": round(np.mean(steps), 1),
        "wall_collision_rate": round(death_counts["wall_collision"] / total * 100, 1),
        "self_collision_rate": round(death_counts["self_collision"] / total * 100, 1),
        "timeout_rate": round(death_counts["no_food_timeout"] / total * 100, 1),
    }


def load_hand_dqn_eval(csv_path: str):
    """加载手写 DQN 评估 CSV，返回统计信息。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    steps = []
    death_counts = {"wall_collision": 0, "self_collision": 0, "no_food_timeout": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row["score"]))
            steps.append(int(row.get("steps", 0)))
            reason = row.get("death_reason", "unknown")
            if reason in death_counts:
                death_counts[reason] += 1

    total = len(scores)
    if total == 0:
        return None

    return {
        "avg_score": round(np.mean(scores), 2),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_steps": round(np.mean(steps), 1),
        "wall_collision_rate": round(death_counts["wall_collision"] / total * 100, 1),
        "self_collision_rate": round(death_counts["self_collision"] / total * 100, 1),
        "timeout_rate": round(death_counts["no_food_timeout"] / total * 100, 1),
    }


def compare():
    """对比不同方法的结果。"""
    ensure_dir("outputs")

    results = []

    # 手写 DQN
    hand_dqn_path = "logs/train_log.csv"
    hand_dqn_stats = load_hand_dqn_eval(hand_dqn_path)
    if hand_dqn_stats:
        results.append({
            "method": "hand_dqn_basic17_best",
            "training_unit": "episodes",
            "training_amount": "3000",
            **hand_dqn_stats,
            "notes": "手写 Double DQN",
        })

    # SB3 200k
    sb3_200k_path = "logs/sb3_eval.csv"
    sb3_200k_stats = load_sb3_eval(sb3_200k_path)
    if sb3_200k_stats:
        results.append({
            "method": "sb3_dqn_basic17_200k",
            "training_unit": "timesteps",
            "training_amount": "200000",
            **sb3_200k_stats,
            "notes": "recommended_sb3_model",
        })

    # SB3 500k continue
    sb3_500k_path = "logs/sb3_runs/sb3_dqn_basic17_500k_continue/sb3_eval.csv"
    sb3_500k_stats = load_sb3_eval(sb3_500k_path)
    if sb3_500k_stats:
        results.append({
            "method": "sb3_dqn_basic17_500k_continue",
            "training_unit": "timesteps",
            "training_amount": "500000",
            **sb3_500k_stats,
            "notes": "degraded_after_continue_training",
        })

    # SB3 1000k (如果存在)
    sb3_1000k_path = "logs/sb3_runs/sb3_dqn_basic17_1000k/sb3_eval.csv"
    sb3_1000k_stats = load_sb3_eval(sb3_1000k_path)
    if sb3_1000k_stats:
        results.append({
            "method": "sb3_dqn_basic17_1000k",
            "training_unit": "timesteps",
            "training_amount": "1000000",
            **sb3_1000k_stats,
            "notes": "SB3 DQN 1000k timesteps",
        })

    if not results:
        print("没有找到评估结果。请先运行训练和评估。")
        return

    # 输出表格
    print(f"\n{'=' * 90}")
    print("训练结果对比")
    print(f"{'=' * 90}")
    print(f"{'方法':<35} {'训练量':<12} {'平均分':<8} {'最高分':<8} {'撞墙%':<8} {'撞自己%':<10} {'超时%':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['method']:<35} {r['training_amount']:<12} {r['avg_score']:<8} {r['max_score']:<8} "
              f"{r['wall_collision_rate']:<8} {r['self_collision_rate']:<10} {r['timeout_rate']:<8}")
    print(f"{'=' * 90}")

    # 保存 CSV
    csv_path = "outputs/compare_results.csv"
    save_train_log(csv_path, results)
    print(f"\n对比结果已保存: {csv_path}")


if __name__ == "__main__":
    compare()
