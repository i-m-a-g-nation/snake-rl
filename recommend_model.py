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


def recommend():
    """推荐最佳模型。"""
    print("=" * 70)
    print("Snake RL 模型推荐")
    print("=" * 70)

    models = []

    # 检查各个模型
    checks = [
        {
            "name": "SB3 200k best",
            "path": "checkpoints/sb3_best/best_model.zip",
            "eval_path": "logs/sb3_eval.csv",
            "type": "sb3",
            "notes": "recommended_sb3_model",
        },
        {
            "name": "SB3 500k continue",
            "path": "checkpoints/sb3_runs/sb3_dqn_basic17_500k_continue/best_model/best_model.zip",
            "eval_path": "logs/sb3_runs/sb3_dqn_basic17_500k_continue/sb3_eval.csv",
            "type": "sb3",
            "notes": "degraded_after_continue_training",
        },
        {
            "name": "Hand-written DQN best",
            "path": "checkpoints/best_model.pt",
            "eval_path": None,
            "type": "torch",
            "notes": "hand_written_double_dqn",
        },
    ]

    for check in checks:
        if os.path.exists(check["path"]):
            stats = load_eval_results(check["eval_path"]) if check["eval_path"] else None
            models.append({
                "name": check["name"],
                "path": check["path"],
                "type": check["type"],
                "stats": stats,
                "notes": check["notes"],
            })

    if not models:
        print("\n没有找到任何模型。请先运行训练。")
        return

    # 按 avg_score 排序
    scored_models = [m for m in models if m["stats"]]
    scored_models.sort(key=lambda x: x["stats"]["avg_score"], reverse=True)

    # 输出结果
    print("\n找到的模型:")
    print("-" * 70)
    for m in models:
        status = "[OK]" if os.path.exists(m["path"]) else "[--]"
        score_str = f"avg={m['stats']['avg_score']}, max={m['stats']['max_score']}" if m["stats"] else "无评估数据"
        print(f"  {status} {m['name']}")
        print(f"     路径: {m['path']}")
        print(f"     评分: {score_str}")
        print(f"     备注: {m['notes']}")
        print()

    # 推荐
    print("=" * 70)
    print("推荐:")
    print("=" * 70)

    # 最佳 SB3
    sb3_models = [m for m in scored_models if m["type"] == "sb3"]
    if sb3_models:
        best_sb3 = sb3_models[0]
        print(f"\n最佳 SB3 模型: {best_sb3['name']}")
        print(f"  路径: {best_sb3['path']}")
        print(f"  评分: avg={best_sb3['stats']['avg_score']}, max={best_sb3['stats']['max_score']}")
        print(f"\n  观看命令:")
        print(f"  python main.py --model {best_sb3['path']} --model-type sb3 --episodes 5 --fps 10 --terminal-render --state-mode basic17")

    # 最佳手写 DQN
    torch_models = [m for m in scored_models if m["type"] == "torch"]
    if torch_models:
        best_torch = torch_models[0]
        print(f"\n最佳手写 DQN 模型: {best_torch['name']}")
        print(f"  路径: {best_torch['path']}")
        print(f"\n  观看命令:")
        print(f"  python main.py --model {best_torch['path']} --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode basic17")

    # 总体推荐
    if scored_models:
        best_overall = scored_models[0]
        print(f"\n{'=' * 70}")
        print(f"总体推荐: {best_overall['name']}")
        print(f"  路径: {best_overall['path']}")
        print(f"  评分: avg={best_overall['stats']['avg_score']}, max={best_overall['stats']['max_score']}")
        print(f"\n  推荐观看命令:")
        model_type = "sb3" if best_overall["type"] == "sb3" else "torch"
        print(f"  python main.py --model {best_overall['path']} --model-type {model_type} --episodes 5 --fps 10 --terminal-render --state-mode basic17")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    recommend()
