"""对比不同训练方法的结果。"""

import os
import csv
import glob

import numpy as np

from utils import ensure_dir, save_train_log


def load_eval_results(csv_path: str):
    """加载评估 CSV，返回统计信息。"""
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


def load_multiseed_results(csv_path: str):
    """加载多 seed 评估 CSV，返回统计信息。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["avg_score"]))

    if not scores:
        return None

    return {
        "mean_avg_score": round(np.mean(scores), 2),
        "std_avg_score": round(np.std(scores), 2),
        "mean_max_score": round(np.mean([float(row["max_score"]) for row in csv.DictReader(open(csv_path, "r", encoding="utf-8"))]), 1),
    }


def compare():
    """对比不同方法的结果。"""
    ensure_dir("outputs")

    results = []

    # 旧最佳模型
    old_path = "logs/torch_eval.csv"
    old_stats = load_eval_results(old_path)
    if old_stats:
        results.append({
            "method": "old_best_basic17",
            "model_path": "checkpoints/best_model_basic17.pt",
            **old_stats,
            "notes": "old_stable_model",
        })

    # Baseline current retrain
    baseline_path = "logs/ablation/baseline_current/eval.csv"
    baseline_stats = load_eval_results(baseline_path)
    if baseline_stats:
        results.append({
            "method": "baseline_current_retrain",
            "model_path": "checkpoints/ablation/baseline_current/best_model.pt",
            **baseline_stats,
            "notes": "double_dqn_retrain",
        })

    # Mask only
    mask_path = "logs/ablation/mask_only/eval.csv"
    mask_stats = load_eval_results(mask_path)
    if mask_stats:
        results.append({
            "method": "mask_only",
            "model_path": "checkpoints/ablation/mask_only/best_model.pt",
            **mask_stats,
            "notes": "action_mask_harmful",
        })

    # Mask dueling
    mask_dueling_path = "logs/ablation/mask_dueling/eval.csv"
    mask_dueling_stats = load_eval_results(mask_dueling_path)
    if mask_dueling_stats:
        results.append({
            "method": "mask_dueling",
            "model_path": "checkpoints/ablation/mask_dueling/best_model.pt",
            **mask_dueling_stats,
            "notes": "mask_dueling_mixed",
        })

    # Dueling only
    dueling_path = "logs/ablation/dueling_only/eval.csv"
    dueling_stats = load_eval_results(dueling_path)
    if dueling_stats:
        results.append({
            "method": "dueling_only",
            "model_path": "checkpoints/ablation/dueling_only/best_model.pt",
            **dueling_stats,
            "notes": "dueling_without_mask",
        })

    # SB3 200k
    sb3_path = "logs/sb3_eval.csv"
    sb3_stats = load_eval_results(sb3_path)
    if sb3_stats:
        results.append({
            "method": "sb3_dqn_200k",
            "model_path": "checkpoints/sb3_best/best_model.zip",
            **sb3_stats,
            "notes": "sb3_baseline",
        })

    # SB3 500k continue
    sb3_500k_path = "logs/sb3_runs/sb3_dqn_basic17_500k_continue/sb3_eval.csv"
    sb3_500k_stats = load_eval_results(sb3_500k_path)
    if sb3_500k_stats:
        results.append({
            "method": "sb3_dqn_500k_continue",
            "model_path": "checkpoints/sb3_runs/sb3_dqn_basic17_500k_continue/best_model/best_model.zip",
            **sb3_500k_stats,
            "notes": "degraded_continue_training",
        })

    if not results:
        print("没有找到评估结果。请先运行训练和评估。")
        return

    # 按 avg_score 排序
    results.sort(key=lambda x: x["avg_score"], reverse=True)

    # 输出表格
    print(f"\n{'=' * 100}")
    print("训练结果对比")
    print(f"{'=' * 100}")
    print(f"{'方法':<35} {'平均分':<10} {'最高分':<10} {'撞墙%':<10} {'撞自己%':<12} {'超时%':<10} {'备注'}")
    print("-" * 100)
    for r in results:
        print(f"{r['method']:<35} {r['avg_score']:<10} {r['max_score']:<10} "
              f"{r['wall_collision_rate']:<10} {r['self_collision_rate']:<12} "
              f"{r['timeout_rate']:<10} {r['notes']}")
    print(f"{'=' * 100}")

    # 保存 CSV
    csv_path = "outputs/compare_results.csv"
    save_train_log(csv_path, results)
    print(f"\n对比结果已保存: {csv_path}")


if __name__ == "__main__":
    compare()
