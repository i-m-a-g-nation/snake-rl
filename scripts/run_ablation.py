"""Ablation 实验脚本。比较不同 DQN 变体的效果。"""

import os
import sys
import csv
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ensure_dir

# 旧模型基准
OLD_MODEL_AVG = 23.93
OLD_MODEL_MAX = 57
OLD_MODEL_PATH = "checkpoints/best_model_basic17.pt"

# 实验配置
EXPERIMENTS = {
    "baseline_current": {
        "args": ["--state-mode", "basic17", "--double-dqn"],
        "notes": "Double DQN baseline (same as old model)",
    },
    "mask_only": {
        "args": ["--state-mode", "basic17", "--double-dqn", "--action-mask"],
        "notes": "Double DQN + Action Mask",
    },
    "mask_dueling": {
        "args": ["--state-mode", "basic17", "--double-dqn", "--action-mask", "--dueling"],
        "notes": "Double DQN + Action Mask + Dueling",
    },
    "full_v2": {
        "args": ["--state-mode", "basic17", "--double-dqn", "--action-mask", "--dueling", "--per", "--n-step", "3"],
        "notes": "Full v2: Double DQN + Action Mask + Dueling + PER + N-step=3",
    },
}

EPISODES = 1000
EVAL_EPISODES = 100
PYTHON = sys.executable


def run_experiment(exp_name: str, exp_config: dict, episodes: int):
    """运行单个实验。"""
    print(f"\n{'=' * 70}")
    print(f"实验: {exp_name}")
    print(f"{'=' * 70}")

    save_path = f"checkpoints/ablation/{exp_name}/final_model.pt"
    best_path = f"checkpoints/ablation/{exp_name}/best_model.pt"
    log_dir = f"logs/ablation/{exp_name}"

    ensure_dir(os.path.dirname(save_path))
    ensure_dir(os.path.dirname(best_path))
    ensure_dir(log_dir)

    # 训练命令
    train_cmd = [
        PYTHON, "train.py",
        "--episodes", str(episodes),
        "--save-path", save_path,
        "--best-path", best_path,
        "--log-dir", log_dir,
        "--eval-interval", "100",
        "--eval-episodes", "20",
        "--warmup-steps", "2000",
        "--seed", "42",
    ] + exp_config["args"]

    print(f"训练命令: {' '.join(train_cmd)}")
    print("-" * 70)

    # 执行训练
    result = subprocess.run(train_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"警告: {exp_name} 训练可能未正常完成")

    return best_path, log_dir


def evaluate_experiment(exp_name: str, model_path: str, log_dir: str):
    """评估实验结果。"""
    print(f"\n评估: {exp_name}")

    eval_csv = os.path.join(log_dir, "eval.csv")

    eval_cmd = [
        PYTHON, "evaluate_torch.py",
        "--model", model_path,
        "--episodes", str(EVAL_EPISODES),
        "--state-mode", "basic17",
        "--save-path", eval_csv,
        "--seed", "5000",
    ]

    print(f"评估命令: {' '.join(eval_cmd)}")

    result = subprocess.run(eval_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"警告: {exp_name} 评估可能未正常完成")

    # 读取评估结果
    return load_eval_results(eval_csv)


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

    import numpy as np
    return {
        "avg_score": round(float(np.mean(scores)), 2),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_steps": round(float(np.mean(steps)), 1),
        "wall_collision_rate": round(death_counts["wall_collision"] / total * 100, 1),
        "self_collision_rate": round(death_counts["self_collision"] / total * 100, 1),
        "timeout_rate": round(death_counts["no_food_timeout"] / total * 100, 1),
    }


def main():
    """运行 ablation 实验。"""
    print("=" * 70)
    print("DQN v2 Ablation 实验")
    print("=" * 70)
    print(f"每个实验: {EPISODES} episodes")
    print(f"评估: {EVAL_EPISODES} episodes")
    print(f"旧模型基准: avg={OLD_MODEL_AVG}, max={OLD_MODEL_MAX}")
    print("=" * 70)

    ensure_dir("logs/ablation")
    results = []

    for exp_name, exp_config in EXPERIMENTS.items():
        # 运行训练
        best_path, log_dir = run_experiment(exp_name, exp_config, EPISODES)

        # 评估
        eval_results = evaluate_experiment(exp_name, best_path, log_dir)

        if eval_results:
            beats_old = eval_results["avg_score"] > OLD_MODEL_AVG
            results.append({
                "exp_name": exp_name,
                "episodes": EPISODES,
                **eval_results,
                "beats_old_model": beats_old,
                "notes": exp_config["notes"],
            })
        else:
            results.append({
                "exp_name": exp_name,
                "episodes": EPISODES,
                "avg_score": 0,
                "max_score": 0,
                "min_score": 0,
                "avg_steps": 0,
                "wall_collision_rate": 0,
                "self_collision_rate": 0,
                "timeout_rate": 0,
                "beats_old_model": False,
                "notes": f"{exp_config['notes']} (评估失败)",
            })

    # 保存结果
    csv_path = "logs/ablation/ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["exp_name", "episodes", "avg_score", "max_score", "min_score",
                      "avg_steps", "wall_collision_rate", "self_collision_rate",
                      "timeout_rate", "beats_old_model", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 输出结果
    print("\n" + "=" * 70)
    print("Ablation 结果")
    print("=" * 70)
    print(f"{'实验':<25} {'平均分':<10} {'最高分':<10} {'撞自己%':<10} {'超时%':<10} {'超过旧模型':<12}")
    print("-" * 70)

    for r in results:
        beats = "YES" if r["beats_old_model"] else "no"
        print(f"{r['exp_name']:<25} {r['avg_score']:<10} {r['max_score']:<10} {r['self_collision_rate']:<10} {r['timeout_rate']:<10} {beats:<12}")

    # 找出最佳实验
    best_exp = max(results, key=lambda x: x["avg_score"])
    print(f"\n{'=' * 70}")
    print("推荐")
    print(f"{'=' * 70}")
    print(f"最佳实验: {best_exp['exp_name']}")
    print(f"平均分: {best_exp['avg_score']}")
    print(f"最高分: {best_exp['max_score']}")

    if best_exp["avg_score"] > OLD_MODEL_AVG:
        print(f"\n超过旧模型! 推荐新模型: checkpoints/ablation/{best_exp['exp_name']}/best_model.pt")
        print("建议继续训练到 3000 episodes。")
    else:
        print(f"\n未超过旧模型 ({OLD_MODEL_AVG})。")
        print(f"继续推荐: {OLD_MODEL_PATH}")

    # 检查 PER/n-step 是否有问题
    if "full_v2" in [r["exp_name"] for r in results]:
        full_v2 = next(r for r in results if r["exp_name"] == "full_v2")
        mask_dueling = next((r for r in results if r["exp_name"] == "mask_dueling"), None)
        if mask_dueling and full_v2["avg_score"] < mask_dueling["avg_score"]:
            print("\n警告: full_v2 低于 mask_dueling!")
            print("PER 或 n-step 可能需要调优/调试；不要假设 full_v2 更好。")

    print(f"\n结果已保存: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
