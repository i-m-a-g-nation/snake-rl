"""Safety Filter 对比评估脚本。"""

import os
import sys
import csv
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ensure_dir

PYTHON = sys.executable
MODEL = "checkpoints/best_model_dueling_basic17.pt"
EPISODES = 100
NUM_SEEDS = 5
STATE_MODE = "basic17"


def run_eval(name: str, use_safety_filter: bool):
    """运行评估。"""
    print(f"\n{'=' * 60}")
    print(f"评估: {name}")
    print(f"{'=' * 60}")

    save_path = f"logs/safety_filter/{name}_eval.csv"

    cmd = [
        PYTHON, "evaluate_torch.py",
        "--model", MODEL,
        "--episodes", str(EPISODES),
        "--state-mode", STATE_MODE,
        "--num-seeds", str(NUM_SEEDS),
        "--save-path", save_path,
    ]

    if use_safety_filter:
        cmd.append("--safety-filter")

    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    # 读取结果
    return load_results(save_path, name)


def load_results(csv_path: str, name: str):
    """加载评估结果。"""
    if not os.path.exists(csv_path):
        return None

    scores = []
    override_rates = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row.get("avg_score", 0)))
            override_rates.append(float(row.get("override_rate", 0)))

    if not scores:
        return None

    import numpy as np
    return {
        "method": name,
        "avg_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores)), 2),
        "max_score": round(float(max(scores)), 1),
        "override_rate": round(float(np.mean(override_rates)), 1),
    }


def main():
    """运行对比评估。"""
    ensure_dir("logs/safety_filter")

    print("=" * 60)
    print("Safety Filter 对比评估")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"评估局数: {EPISODES}")
    print(f"多 Seed 数: {NUM_SEEDS}")
    print("=" * 60)

    results = []

    # A. 原始策略
    r1 = run_eval("dueling_basic17_raw", use_safety_filter=False)
    if r1:
        r1["notes"] = "raw_dqn_policy"
        results.append(r1)

    # B. Safety filter
    r2 = run_eval("dueling_basic17_safety", use_safety_filter=True)
    if r2:
        r2["notes"] = "safety_filter_inference"
        results.append(r2)

    # 保存结果
    csv_path = "logs/safety_filter/safety_filter_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "avg_score", "std_score", "max_score", "override_rate", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 输出对比
    print(f"\n{'=' * 60}")
    print("对比结果")
    print(f"{'=' * 60}")
    print(f"{'方法':<35} {'平均分':<12} {'最高分':<10} {'覆盖率':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<35} {r['avg_score']:<12} {r['max_score']:<10} {r['override_rate']:<10}")

    # 判断是否提升
    if len(results) == 2:
        raw = next(r for r in results if "raw" in r["method"])
        safety = next(r for r in results if "safety" in r["method"])

        print(f"\n{'=' * 60}")
        if safety["avg_score"] > raw["avg_score"]:
            print(f"Safety filter 提升了性能! ({raw['avg_score']} -> {safety['avg_score']})")
            print(f"推荐命令:")
            print(f"  python main.py --model {MODEL} --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode {STATE_MODE} --safety-filter")
        else:
            print(f"Safety filter 未提升性能 ({raw['avg_score']} vs {safety['avg_score']})")
            print(f"继续推荐原始策略:")
            print(f"  python main.py --model {MODEL} --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode {STATE_MODE}")

    print(f"\n结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
