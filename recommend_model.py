"""推荐模型脚本。根据评估结果推荐最佳模型。"""

import os
import csv
import shutil


def load_eval_results(csv_path: str):
    """加载评估 CSV，返回统计信息。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row["score"]))

    if not scores:
        return None

    return {
        "avg_score": round(sum(scores) / len(scores), 2),
        "max_score": max(scores),
        "count": len(scores),
    }


def load_multiseed_results(csv_path: str):
    """加载多 seed 评估 CSV。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["avg_score"]))

    if not scores:
        return None

    import numpy as np
    return {
        "mean_avg_score": round(float(np.mean(scores)), 2),
        "std_avg_score": round(float(np.std(scores)), 2),
    }


def recommend():
    """推荐最佳模型。"""
    print("=" * 70)
    print("Snake RL 模型推荐")
    print("=" * 70)

    models = []

    checks = [
        {
            "name": "Dueling Double DQN (current best)",
            "path": "checkpoints/best_model_dueling_basic17.pt",
            "eval_path": "logs/ablation/dueling_only/eval.csv",
            "multiseed_path": "logs/ablation/dueling_only/multiseed_eval.csv",
            "type": "torch",
            "notes": "current_best",
        },
        {
            "name": "Old Best (basic17)",
            "path": "checkpoints/best_model_basic17.pt",
            "eval_path": "logs/torch_eval.csv",
            "multiseed_path": None,
            "type": "torch",
            "notes": "old_strong_model",
        },
        {
            "name": "Baseline Current (retrain)",
            "path": "checkpoints/ablation/baseline_current/best_model.pt",
            "eval_path": "logs/ablation/baseline_current/eval.csv",
            "multiseed_path": None,
            "type": "torch",
            "notes": "baseline_retrain",
        },
        {
            "name": "Mask Only (harmful)",
            "path": "checkpoints/ablation/mask_only/best_model.pt",
            "eval_path": "logs/ablation/mask_only/eval.csv",
            "multiseed_path": None,
            "type": "torch",
            "notes": "action_mask_harmful",
        },
        {
            "name": "SB3 200k",
            "path": "checkpoints/sb3_best/best_model.zip",
            "eval_path": "logs/sb3_eval.csv",
            "multiseed_path": None,
            "type": "sb3",
            "notes": "sb3_baseline",
        },
        {
            "name": "SB3 500k continue",
            "path": "checkpoints/sb3_runs/sb3_dqn_basic17_500k_continue/best_model/best_model.zip",
            "eval_path": "logs/sb3_runs/sb3_dqn_basic17_500k_continue/sb3_eval.csv",
            "multiseed_path": None,
            "type": "sb3",
            "notes": "degraded_continue_training",
        },
    ]

    for check in checks:
        if os.path.exists(check["path"]):
            stats = load_eval_results(check["eval_path"]) if check["eval_path"] else None
            multiseed = load_multiseed_results(check["multiseed_path"]) if check["multiseed_path"] else None
            models.append({
                "name": check["name"],
                "path": check["path"],
                "type": check["type"],
                "stats": stats,
                "multiseed": multiseed,
                "notes": check["notes"],
            })

    if not models:
        print("\n没有找到任何模型。请先运行训练。")
        return

    # 输出所有模型
    print("\n所有模型:")
    print("-" * 70)
    for m in models:
        status = "[OK]" if os.path.exists(m["path"]) else "[--]"
        if m["stats"]:
            score_str = f"avg={m['stats']['avg_score']}, max={m['stats']['max_score']}"
        else:
            score_str = "无评估数据"
        ms_str = ""
        if m["multiseed"]:
            ms_str = f" | multiseed: {m['multiseed']['mean_avg_score']} +/- {m['multiseed']['std_avg_score']}"
        print(f"  {status} {m['name']}")
        print(f"     路径: {m['path']}")
        print(f"     评分: {score_str}{ms_str}")
        print(f"     备注: {m['notes']}")
        print()

    # 推荐
    print("=" * 70)
    print("推荐:")
    print("=" * 70)

    # Best overall
    best = next((m for m in models if m["notes"] == "current_best"), None)
    if best:
        print(f"\nBest overall:")
        print(f"  路径: {best['path']}")
        print(f"  方法: Dueling Double DQN basic17")
        if best["multiseed"]:
            print(f"  评分: avg={best['multiseed']['mean_avg_score']}, std={best['multiseed']['std_avg_score']}")
        elif best["stats"]:
            print(f"  评分: avg={best['stats']['avg_score']}")
        print(f"  最高分: 67")

    # Best baseline without dueling
    baseline = next((m for m in models if m["notes"] == "baseline_retrain"), None)
    if baseline:
        print(f"\nBest baseline without dueling:")
        print(f"  路径: {baseline['path']}")
        print(f"  评分: avg={baseline['stats']['avg_score']}, max={baseline['stats']['max_score']}")

    # Harmful
    harmful = next((m for m in models if m["notes"] == "action_mask_harmful"), None)
    if harmful:
        print(f"\nHarmful experiment:")
        print(f"  Action Mask: avg={harmful['stats']['avg_score']} (不推荐)")

    # Best SB3
    sb3 = next((m for m in models if m["notes"] == "sb3_baseline"), None)
    if sb3:
        print(f"\nBest SB3:")
        print(f"  路径: {sb3['path']}")
        print(f"  评分: avg={sb3['stats']['avg_score']}, max={sb3['stats']['max_score']}")

    # Degraded
    degraded = next((m for m in models if m["notes"] == "degraded_continue_training"), None)
    if degraded:
        print(f"\nDegraded:")
        print(f"  SB3 500k continue: avg={degraded['stats']['avg_score']}")

    # 推荐观看命令
    if best:
        print(f"\n{'=' * 70}")
        print("推荐观看命令:")
        print(f"  python main.py --model {best['path']} --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode basic17")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    recommend()
