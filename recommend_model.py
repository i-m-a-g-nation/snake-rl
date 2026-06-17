"""推荐模型脚本。根据评估结果推荐最佳模型。"""

import os
import csv


def load_eval_results(csv_path: str):
    """加载评估 CSV，返回统计信息。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    death_counts = {"wall_collision": 0, "self_collision": 0, "no_food_timeout": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(int(row["score"]))
            reason = row.get("death_reason", "unknown")
            if reason in death_counts:
                death_counts[reason] += 1

    total = len(scores)
    if not scores:
        return None

    return {
        "avg_score": round(sum(scores) / len(scores), 2),
        "max_score": max(scores),
        "count": total,
        "wall_collision_rate": round(death_counts["wall_collision"] / total * 100, 1),
        "self_collision_rate": round(death_counts["self_collision"] / total * 100, 1),
        "timeout_rate": round(death_counts["no_food_timeout"] / total * 100, 1),
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
            "name": "Hand-written DQN basic17 best",
            "path": "checkpoints/best_model_basic17.pt",
            "eval_path": "logs/torch_eval.csv",
            "type": "torch",
            "notes": "best_overall",
        },
        {
            "name": "Hand-written DQN reachable23",
            "path": "checkpoints/best_model.pt",
            "eval_path": None,
            "type": "torch",
            "notes": "reachable23_experiment",
        },
        {
            "name": "SB3 200k best",
            "path": "checkpoints/sb3_best/best_model.zip",
            "eval_path": "logs/sb3_eval.csv",
            "type": "sb3",
            "notes": "sb3_baseline",
        },
        {
            "name": "SB3 500k continue",
            "path": "checkpoints/sb3_runs/sb3_dqn_basic17_500k_continue/best_model/best_model.zip",
            "eval_path": "logs/sb3_runs/sb3_dqn_basic17_500k_continue/sb3_eval.csv",
            "type": "sb3",
            "notes": "degraded_after_continue_training",
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

    # 分类
    torch_models = [m for m in models if m["type"] == "torch" and m["stats"]]
    sb3_models = [m for m in models if m["type"] == "sb3" and m["stats"]]
    all_scored = [m for m in models if m["stats"]]

    # 按 avg_score 排序
    all_scored.sort(key=lambda x: x["stats"]["avg_score"], reverse=True)
    torch_models.sort(key=lambda x: x["stats"]["avg_score"], reverse=True)
    sb3_models.sort(key=lambda x: x["stats"]["avg_score"], reverse=True)

    # 输出所有模型
    print("\n所有模型:")
    print("-" * 70)
    for m in models:
        status = "[OK]" if os.path.exists(m["path"]) else "[--]"
        if m["stats"]:
            score_str = f"avg={m['stats']['avg_score']}, max={m['stats']['max_score']}"
        else:
            score_str = "无评估数据"
        print(f"  {status} {m['name']}")
        print(f"     路径: {m['path']}")
        print(f"     评分: {score_str}")
        print(f"     备注: {m['notes']}")
        print()

    # 推荐
    print("=" * 70)
    print("推荐:")
    print("=" * 70)

    # 最佳手写 DQN
    if torch_models:
        best_torch = torch_models[0]
        print(f"\n最佳手写 DQN: {best_torch['name']}")
        print(f"  路径: {best_torch['path']}")
        print(f"  评分: avg={best_torch['stats']['avg_score']}, max={best_torch['stats']['max_score']}")

    # 最佳 SB3
    if sb3_models:
        # 排除 degraded
        good_sb3 = [m for m in sb3_models if "degraded" not in m["notes"]]
        if good_sb3:
            best_sb3 = good_sb3[0]
        else:
            best_sb3 = sb3_models[0]
        print(f"\n最佳 SB3 DQN: {best_sb3['name']}")
        print(f"  路径: {best_sb3['path']}")
        print(f"  评分: avg={best_sb3['stats']['avg_score']}, max={best_sb3['stats']['max_score']}")
        if "degraded" in best_sb3["notes"]:
            print(f"  注意: 此模型已标记为 degraded")

    # 总体推荐
    if all_scored:
        best_overall = all_scored[0]
        print(f"\n{'=' * 70}")
        print(f"总体推荐: {best_overall['name']}")
        print(f"  路径: {best_overall['path']}")
        print(f"  评分: avg={best_overall['stats']['avg_score']}, max={best_overall['stats']['max_score']}")

        # 判断哪个更强
        if torch_models and sb3_models:
            best_torch_score = torch_models[0]["stats"]["avg_score"]
            good_sb3 = [m for m in sb3_models if "degraded" not in m["notes"]]
            best_sb3_score = good_sb3[0]["stats"]["avg_score"] if good_sb3 else 0

            print(f"\n{'=' * 70}")
            if best_torch_score > best_sb3_score:
                print("当前最强 agent 是手写 Double DQN，不是 SB3 DQN。")
                print("SB3 DQN 保留为标准库 baseline。")
            else:
                print("当前最强 agent 是 SB3 DQN。")
                print("手写 DQN 用于学习算法原理。")

        # 推荐观看命令
        model_type = "sb3" if best_overall["type"] == "sb3" else "torch"
        print(f"\n推荐观看命令:")
        print(f"  python main.py --model {best_overall['path']} --model-type {model_type} --episodes 5 --fps 10 --terminal-render --state-mode basic17")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    recommend()
