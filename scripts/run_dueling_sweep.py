"""Dueling Double DQN 超参数搜索脚本。"""

import os
import sys
import csv
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ensure_dir

PYTHON = sys.executable
STATE_MODE = "basic17"
OLD_BEST_AVG = 35.40

# 超参数配置
SWEEP_CONFIGS = {
    "A": {"lr": 5e-4, "eps_end": 0.03, "eps_decay": 100000, "target_update": 500},
    "B": {"lr": 3e-4, "eps_end": 0.03, "eps_decay": 100000, "target_update": 500},
    "C": {"lr": 5e-4, "eps_end": 0.01, "eps_decay": 100000, "target_update": 500},
    "D": {"lr": 5e-4, "eps_end": 0.03, "eps_decay": 120000, "target_update": 500},
    "E": {"lr": 5e-4, "eps_end": 0.03, "eps_decay": 80000, "target_update": 500},
    "F": {"lr": 5e-4, "eps_end": 0.03, "eps_decay": 100000, "target_update": 1000},
    "G": {"lr": 3e-4, "eps_end": 0.01, "eps_decay": 120000, "target_update": 500},
    "H": {"lr": 3e-4, "eps_end": 0.03, "eps_decay": 120000, "target_update": 1000},
}


def load_eval_results(csv_path: str):
    """加载评估结果。"""
    if not os.path.exists(csv_path):
        return None

    # 检查是否是多 seed 评估结果
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # 如果有 avg_score 字段，是多 seed 结果
    if "avg_score" in rows[0]:
        scores = [float(row["avg_score"]) for row in rows]
        import numpy as np
        return {
            "avg_score": round(float(np.mean(scores)), 2),
            "std_score": round(float(np.std(scores)), 2),
            "max_score": round(float(max(scores)), 1),
            "avg_steps": 0,
            "wall_collision_rate": 0,
            "self_collision_rate": 0,
            "timeout_rate": 0,
        }

    # 否则是单次评估结果
    scores = [int(row.get("score", 0)) for row in rows]
    total = len(scores)
    if total == 0:
        return None

    import numpy as np
    return {
        "avg_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores)), 2),
        "max_score": max(scores),
        "avg_steps": 0,
        "wall_collision_rate": 0,
        "self_collision_rate": 0,
        "timeout_rate": 0,
    }


def run_single(run_name: str, config: dict, seed: int, episodes: int, quick: bool = False):
    """运行单个实验。"""
    run_dir = f"checkpoints/dueling_sweep/{run_name}"
    log_dir = f"logs/dueling_sweep/{run_name}"
    ensure_dir(run_dir)
    ensure_dir(log_dir)

    save_path = os.path.join(run_dir, "final_model.pt")
    best_path = os.path.join(run_dir, "best_model.pt")

    # 训练
    train_cmd = [
        PYTHON, "train.py",
        "--episodes", str(episodes),
        "--state-mode", STATE_MODE,
        "--double-dqn",
        "--dueling",
        "--lr", str(config["lr"]),
        "--epsilon-end", str(config["eps_end"]),
        "--epsilon-decay", str(config["eps_decay"]),
        "--target-update", str(config["target_update"]),
        "--save-path", save_path,
        "--best-path", best_path,
        "--log-dir", log_dir,
        "--eval-interval", "100",
        "--eval-episodes", "20",
        "--warmup-steps", "2000",
        "--seed", str(seed),
    ]

    print(f"\n  训练: {run_name} (seed={seed}, episodes={episodes})")
    result = subprocess.run(train_cmd, capture_output=False, timeout=1800 if not quick else 300)

    if result.returncode != 0:
        print(f"  警告: {run_name} 训练可能未正常完成")
        return None

    # 评估
    eval_csv = os.path.join(log_dir, "eval.csv")
    eval_cmd = [
        PYTHON, "evaluate_torch.py",
        "--model", best_path,
        "--episodes", "100",
        "--state-mode", STATE_MODE,
        "--num-seeds", "3" if quick else "5",
        "--save-path", eval_csv,
    ]

    print(f"  评估: {run_name}")
    subprocess.run(eval_cmd, capture_output=False, timeout=300 if quick else 600)

    return load_eval_results(eval_csv)


def main():
    parser = argparse.ArgumentParser(description="Dueling Double DQN 超参数搜索")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="随机种子列表")
    parser.add_argument("--episodes", type=int, default=3000, help="训练轮数")
    parser.add_argument("--quick", action="store_true", help="快速模式 (100 episodes)")
    parser.add_argument("--configs", type=str, default=None, help="指定配置 (如 A,B,C)")
    args = parser.parse_args()

    ensure_dir("checkpoints/dueling_sweep")
    ensure_dir("logs/dueling_sweep")

    # 选择配置
    if args.configs:
        config_names = args.configs.split(",")
    else:
        config_names = list(SWEEP_CONFIGS.keys())

    episodes = 100 if args.quick else args.episodes

    print("=" * 70)
    print("Dueling Double DQN 超参数搜索")
    print("=" * 70)
    print(f"配置: {config_names}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes: {episodes}")
    print(f"目标: avg_score > {OLD_BEST_AVG}")
    print("=" * 70)

    results = []

    for config_name in config_names:
        if config_name not in SWEEP_CONFIGS:
            print(f"未知配置: {config_name}")
            continue

        config = SWEEP_CONFIGS[config_name]

        for seed in args.seeds:
            run_name = f"config_{config_name}_seed_{seed}"

            print(f"\n{'=' * 70}")
            print(f"配置 {config_name}: lr={config['lr']}, eps_end={config['eps_end']}, eps_decay={config['eps_decay']}, target_update={config['target_update']}")
            print(f"Seed: {seed}")
            print(f"{'=' * 70}")

            eval_result = run_single(run_name, config, seed, episodes, args.quick)

            record = {
                "run_name": run_name,
                "config": config_name,
                "seed": seed,
                "episodes": episodes,
                "learning_rate": config["lr"],
                "epsilon_end": config["eps_end"],
                "epsilon_decay_steps": config["eps_decay"],
                "target_update_interval": config["target_update"],
            }

            if eval_result:
                record.update(eval_result)
                record["beats_current_best"] = eval_result["avg_score"] > OLD_BEST_AVG
                record["notes"] = ""
            else:
                record.update({
                    "avg_score": 0, "std_score": 0, "max_score": 0,
                    "avg_steps": 0, "wall_collision_rate": 0,
                    "self_collision_rate": 0, "timeout_rate": 0,
                    "beats_current_best": False,
                    "notes": "eval_failed",
                })

            results.append(record)

            # 保存中间结果
            csv_path = "logs/dueling_sweep/dueling_sweep_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = list(record.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

    # 汇总结果
    print(f"\n{'=' * 70}")
    print("Sweep 结果汇总")
    print(f"{'=' * 70}")
    print(f"{'Run':<35} {'Avg':<10} {'Std':<10} {'Max':<10} {'Beats':<10}")
    print("-" * 70)

    best_result = None
    for r in results:
        beats = "YES" if r.get("beats_current_best") else "no"
        print(f"{r['run_name']:<35} {r.get('avg_score', 0):<10} {r.get('std_score', 0):<10} {r.get('max_score', 0):<10} {beats:<10}")

        if r.get("beats_current_best"):
            if best_result is None or r.get("avg_score", 0) > best_result.get("avg_score", 0):
                best_result = r

    print(f"\n{'=' * 70}")
    if best_result:
        print(f"最佳配置: {best_result['run_name']}")
        print(f"  avg={best_result['avg_score']}, std={best_result['std_score']}, max={best_result['max_score']}")
        print(f"  超过当前最强 ({OLD_BEST_AVG})!")

        # 复制为推荐模型
        import shutil
        src = f"checkpoints/dueling_sweep/{best_result['run_name']}/best_model.pt"
        dst = "checkpoints/best_model_dueling_basic17_sweep.pt"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  已复制到: {dst}")
    else:
        print(f"没有配置超过当前最强 ({OLD_BEST_AVG})")
        print("继续推荐: checkpoints/best_model_dueling_basic17.pt")

    print(f"\n结果已保存: logs/dueling_sweep/dueling_sweep_results.csv")


if __name__ == "__main__":
    main()
