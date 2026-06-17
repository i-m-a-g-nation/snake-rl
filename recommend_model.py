"""推荐模型脚本。根据评估结果推荐最佳模型。"""

import os
import csv


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
            "name": "Old Best (basic17)",
            "path": "checkpoints/best_model_basic17.pt",
            "eval_path": "logs/torch_eval.csv",
            "multiseed_path": None,
            "type": "torch",
            "notes": "old_stable_model",
        },
        {
            "name": "Dueling Only (new candidate)",
            "path": "checkpoints/ablation/dueling_only/best_model.pt",
            "eval_path": "logs/ablation/dueling_only/eval.csv",
            "multiseed_path": "logs/ablation/dueling_only/multiseed_eval.csv",
            "type": "torch",
            "notes": "current_best_candidate",
        },
        {
            "name": "Baseline Current",
            "path": "checkpoints/ablation/baseline_current/best_model.pt",
            "eval_path": "logs/ablation/baseline_current/eval.csv",
            "multiseed_path": None,
            "type": "torch",
            "notes": "double_dqn_retrain",
        },
        {
            "name": "SB3 200k",
            "path": "checkpoints/sb3_best/best_model.zip",
            "eval_path": "logs/sb3_eval.csv",
            "multiseed_path": None,
            "type": "sb3",
            "notes": "sb3_baseline",
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

    # 按 avg_score 排序
    scored = [m for m in models if m["stats"]]
    scored.sort(key=lambda x: x["stats"]["avg_score"], reverse=True)

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

    # 找最佳模型
    if scored:
        best = scored[0]
        print(f"\n当前最强模型: {best['name']}")
        print(f"  路径: {best['path']}")
        print(f"  评分: avg={best['stats']['avg_score']}, max={best['stats']['max_score']}")
        if best["multiseed"]:
            print(f"  多 Seed: {best['multiseed']['mean_avg_score']} +/- {best['multiseed']['std_avg_score']}")

        # 判断是否超过旧模型
        old_model = next((m for m in models if m["notes"] == "old_stable_model"), None)
        if old_model and old_model["stats"]:
            if best["stats"]["avg_score"] > old_model["stats"]["avg_score"]:
                print(f"\n超过旧模型! ({old_model['stats']['avg_score']} -> {best['stats']['avg_score']})")

                # 复制为推荐模型
                import shutil
                recommend_path = "checkpoints/best_model_basic17_v2_candidate.pt"
                if not os.path.exists(recommend_path):
                    shutil.copy2(best["path"], recommend_path)
                    print(f"已复制到: {recommend_path}")

                model_type = "torch" if best["type"] == "torch" else "sb3"
                print(f"\n推荐观看命令:")
                print(f"  python main.py --model {recommend_path} --model-type {model_type} --episodes 5 --fps 10 --terminal-render --state-mode basic17")
            else:
                print(f"\n未超过旧模型 ({old_model['stats']['avg_score']})")
                print(f"继续推荐: {old_model['path']}")
                print(f"\n推荐观看命令:")
                print(f"  python main.py --model {old_model['path']} --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode basic17")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    recommend()
