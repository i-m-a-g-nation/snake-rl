"""DQfD-lite 自动化脚本。"""

import os
import sys
import csv
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ensure_dir

PYTHON = sys.executable
STATE_MODE = "basic17"
DEMO_FILE = "data/demonstrations/planner_basic17_1000.npz"
PRETRAINED_MODEL = "checkpoints/dqfd_lite/pretrained_dueling_basic17.pt"
BEST_MODEL = "checkpoints/dqfd_lite/best_model.pt"
FINAL_MODEL = "checkpoints/dqfd_lite/final_model.pt"
LOG_DIR = "logs/dqfd_lite"
OLD_BEST_AVG = 35.40


def run_step(name: str, cmd: list):
    """运行一个步骤。"""
    print(f"\n{'=' * 60}")
    print(f"步骤: {name}")
    print(f"{'=' * 60}")
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def load_eval_results(csv_path: str):
    """加载评估结果。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row.get("avg_score", 0)))

    if not scores:
        return None

    import numpy as np
    return {
        "mean_avg_score": round(float(np.mean(scores)), 2),
        "std_avg_score": round(float(np.std(scores)), 2),
        "max_score": round(float(max(scores)), 1),
    }


def main():
    """运行 DQfD-lite 流程。"""
    ensure_dir("data/demonstrations")
    ensure_dir("checkpoints/dqfd_lite")
    ensure_dir(LOG_DIR)

    print("=" * 60)
    print("DQfD-lite 实验")
    print("=" * 60)

    # 1. 生成示范数据
    success = run_step("生成示范数据", [
        PYTHON, "scripts/generate_demonstrations.py",
        "--episodes", "1000",
        "--state-mode", STATE_MODE,
        "--output", DEMO_FILE,
    ])
    if not success:
        print("示范数据生成失败!")
        return

    # 2. 预训练
    success = run_step("监督预训练", [
        PYTHON, "pretrain_from_demonstrations.py",
        "--demo", DEMO_FILE,
        "--state-mode", STATE_MODE,
        "--dueling",
        "--epochs", "20",
        "--batch-size", "256",
        "--save-path", PRETRAINED_MODEL,
        "--log-path", os.path.join(LOG_DIR, "pretrain_log.csv"),
    ])
    if not success:
        print("预训练失败!")
        return

    # 3. RL 微调
    success = run_step("RL 微调", [
        PYTHON, "train.py",
        "--episodes", "1000",
        "--state-mode", STATE_MODE,
        "--double-dqn",
        "--dueling",
        "--load-model", PRETRAINED_MODEL,
        "--save-path", FINAL_MODEL,
        "--best-path", BEST_MODEL,
        "--log-dir", LOG_DIR,
        "--eval-interval", "100",
        "--eval-episodes", "20",
        "--warmup-steps", "2000",
        "--seed", "42",
    ])
    if not success:
        print("RL 微调失败!")
        return

    # 4. 多 seed 评估
    success = run_step("多 Seed 评估", [
        PYTHON, "evaluate_torch.py",
        "--model", BEST_MODEL,
        "--episodes", "100",
        "--state-mode", STATE_MODE,
        "--num-seeds", "5",
        "--save-path", os.path.join(LOG_DIR, "eval.csv"),
    ])

    # 5. 读取结果
    results = load_eval_results(os.path.join(LOG_DIR, "eval.csv"))

    print(f"\n{'=' * 60}")
    print("DQfD-lite 结果")
    print(f"{'=' * 60}")

    if results:
        print(f"  mean_avg_score: {results['mean_avg_score']} +/- {results['std_avg_score']}")
        print(f"  max_score: {results['max_score']}")

        if results["mean_avg_score"] > OLD_BEST_AVG:
            print(f"\n超过当前最强! ({OLD_BEST_AVG} -> {results['mean_avg_score']})")
            # 复制为推荐模型
            import shutil
            recommend_path = "checkpoints/best_model_dqfd_lite_basic17.pt"
            shutil.copy2(BEST_MODEL, recommend_path)
            print(f"已复制到: {recommend_path}")
        else:
            print(f"\n未超过当前最强 ({OLD_BEST_AVG})")
            print("继续推荐: checkpoints/best_model_dueling_basic17.pt")
    else:
        print("评估结果读取失败")

    # 保存结果
    results_csv = os.path.join(LOG_DIR, "dqfd_lite_results.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean_avg_score", "std_avg_score", "max_score", "beats_old_best"])
        if results:
            writer.writerow([
                "dqfd_lite_basic17",
                results["mean_avg_score"],
                results["std_avg_score"],
                results["max_score"],
                results["mean_avg_score"] > OLD_BEST_AVG,
            ])

    print(f"\n结果已保存: {results_csv}")


if __name__ == "__main__":
    main()
