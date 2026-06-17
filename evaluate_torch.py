"""评估手写 DQN 的 .pt 模型。支持 safety filter。"""

import argparse
import os
import sys

import numpy as np
import torch

from snake_env import SnakeEnv
from agent import DQNAgent
from models import DQN, DuelingDQN, NoisyDuelingDQN, CNNDuelingDQN
from utils import ensure_dir, save_train_log

# 尝试导入 safety_filter
try:
    from safety_filter import choose_safe_action
    HAS_SAFETY_FILTER = True
except ImportError:
    HAS_SAFETY_FILTER = False


def evaluate(model_path: str, episodes: int = 100, grid_size: int = 20, state_mode: str = "basic17", seed: int = 3000, save_path: str = None, use_safety_filter: bool = False):
    """评估手写 DQN 模型。"""
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练: python train.py --episodes 3000")
        sys.exit(1)

    # 检测模型类型
    checkpoint = torch.load(model_path, map_location="cpu")
    is_cnn = any("cnn" in k for k in checkpoint["policy_net"].keys())
    is_noisy = any("weight_sigma" in k for k in checkpoint["policy_net"].keys()) and not is_cnn
    is_dueling = any("shared" in k for k in checkpoint["policy_net"].keys()) and not is_cnn and not is_noisy

    # 创建环境和 agent
    env = SnakeEnv(grid_size=grid_size, seed=seed, state_mode=state_mode)
    agent = DQNAgent(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        use_dueling=is_dueling,
        use_noisy_net=is_noisy,
        use_cnn=is_cnn,
    )
    agent.load(model_path)
    agent.policy_net.eval()

    print(f"加载模型: {model_path}")
    print(f"设备: {agent.device}")
    print(f"状态模式: {state_mode}")
    print(f"评估局数: {episodes}")
    print(f"Safety Filter: {'ON' if use_safety_filter else 'OFF'}")
    print("-" * 60)

    records = []
    scores = []
    steps_list = []
    death_counts = {
        "wall_collision": 0,
        "self_collision": 0,
        "no_food_timeout": 0,
        "unknown": 0,
    }

    # Safety filter 统计
    total_overrides = 0
    total_steps = 0
    immediate_death_prevented = 0
    low_space_avoid = 0
    tail_unreachable_avoid = 0

    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=seed + ep)
        done = False
        ep_overrides = 0

        while not done:
            # 获取动作
            if use_safety_filter and HAS_SAFETY_FILTER:
                # 获取 Q 值
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_values = agent.policy_net(state_t).cpu().numpy()[0]
                action, debug_info = choose_safe_action(env.game, q_values)
                if debug_info.get("override", False):
                    ep_overrides += 1
                    total_overrides += 1
                    reason = debug_info.get("override_reason", "")
                    if reason == "low_space":
                        low_space_avoid += 1
                    elif reason == "tail_unreachable":
                        tail_unreachable_avoid += 1
                    else:
                        immediate_death_prevented += 1
            else:
                action = agent.select_action(state, epsilon=0.0)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_steps += 1

        score = info["score"]
        steps = info["steps"]
        death_reason = info.get("death_reason", "unknown")

        scores.append(score)
        steps_list.append(steps)

        if death_reason in death_counts:
            death_counts[death_reason] += 1
        else:
            death_counts["unknown"] += 1

        records.append({
            "episode": ep,
            "score": score,
            "steps": steps,
            "death_reason": death_reason,
            "safety_overrides": ep_overrides,
        })

    env.close()

    # 统计
    avg_score = np.mean(scores)
    max_score = max(scores)
    min_score = min(scores)
    avg_steps = np.mean(steps_list)
    total = episodes

    wall_rate = death_counts["wall_collision"] / total * 100
    self_rate = death_counts["self_collision"] / total * 100
    timeout_rate = death_counts["no_food_timeout"] / total * 100
    unknown_rate = death_counts["unknown"] / total * 100
    override_rate = total_overrides / total_steps * 100 if total_steps > 0 else 0

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"评估结果 ({episodes} 局)")
    print(f"{'=' * 60}")
    print(f"  平均得分:     {avg_score:.2f}")
    print(f"  最高得分:     {max_score}")
    print(f"  最低得分:     {min_score}")
    print(f"  平均步数:     {avg_steps:.1f}")
    print()
    print(f"  撞墙次数:     {death_counts['wall_collision']}  ({wall_rate:.1f}%)")
    print(f"  撞自己次数:   {death_counts['self_collision']}  ({self_rate:.1f}%)")
    print(f"  超时次数:     {death_counts['no_food_timeout']}  ({timeout_rate:.1f}%)")
    print(f"  未知次数:     {death_counts['unknown']}  ({unknown_rate:.1f}%)")

    if use_safety_filter:
        print()
        print(f"  Safety Filter 统计:")
        print(f"    总步数:       {total_steps}")
        print(f"    覆盖次数:     {total_overrides}")
        print(f"    覆盖率:       {override_rate:.1f}%")
        print(f"    阻止立即死亡: {immediate_death_prevented}")
        print(f"    避免低空间:   {low_space_avoid}")
        print(f"    避免尾部不可达: {tail_unreachable_avoid}")

    # 诊断
    print(f"\n{'=' * 60}")
    print("诊断")
    print(f"{'=' * 60}")

    rates = {
        "self_collision": self_rate,
        "wall_collision": wall_rate,
        "no_food_timeout": timeout_rate,
    }
    max_reason = max(rates, key=rates.get)

    if max_reason == "self_collision":
        print("  主要问题: 撞自己 (self_collision)")
        print("  说明: 蛇变长后路径规划不足")
    elif max_reason == "no_food_timeout":
        print("  主要问题: 超时 (no_food_timeout)")
        print("  说明: 仍有绕圈行为，找不到食物")
    else:
        print("  主要问题: 撞墙 (wall_collision)")
        print("  说明: 基础避障能力不稳定")

    # 保存 CSV
    csv_path = save_path or "logs/torch_eval.csv"
    ensure_dir(os.path.dirname(csv_path))

    # 添加安全过滤器信息到记录
    for r in records:
        r["safety_filter_enabled"] = use_safety_filter

    save_train_log(csv_path, records)
    print(f"\n详细记录已保存: {csv_path}")

    return {
        "avg_score": avg_score,
        "max_score": max_score,
        "min_score": min_score,
        "avg_steps": avg_steps,
        "death_counts": death_counts,
        "rates": rates,
        "total_overrides": total_overrides,
        "override_rate": override_rate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估手写 DQN 模型")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt", help="模型路径")
    parser.add_argument("--episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--grid-size", type=int, default=20, help="网格大小")
    parser.add_argument("--state-mode", type=str, default="basic17", choices=["basic17", "reachable23", "grid"], help="状态模式")
    parser.add_argument("--seed", type=int, default=3000, help="随机种子")
    parser.add_argument("--save-path", type=str, default=None, help="CSV 保存路径")
    parser.add_argument("--num-seeds", type=int, default=1, help="多 seed 评估数量")
    parser.add_argument("--safety-filter", action="store_true", help="启用推理阶段安全规划器")
    args = parser.parse_args()

    if args.safety_filter and not HAS_SAFETY_FILTER:
        print("警告: safety_filter 模块未找到，禁用安全规划器。")
        args.safety_filter = False

    if args.num_seeds > 1:
        # 多 seed 评估
        all_results = []
        for i in range(args.num_seeds):
            seed = args.seed + i * 1000
            print(f"\n--- Seed {i+1}/{args.num_seeds} (seed={seed}) ---")
            result = evaluate(args.model, args.episodes, args.grid_size, args.state_mode, seed, None, args.safety_filter)
            if result:
                all_results.append(result)

        if all_results:
            avg_scores = [r["avg_score"] for r in all_results]
            max_scores = [r["max_score"] for r in all_results]
            print(f"\n{'=' * 60}")
            print(f"多 Seed 评估结果 ({args.num_seeds} seeds)")
            print(f"{'=' * 60}")
            print(f"  mean_avg_score: {np.mean(avg_scores):.2f} +/- {np.std(avg_scores):.2f}")
            print(f"  mean_max_score: {np.mean(max_scores):.1f}")
            print(f"  best_avg_score: {max(avg_scores):.2f}")

            if args.safety_filter:
                total_overrides = sum(r["total_overrides"] for r in all_results)
                avg_override_rate = np.mean([r["override_rate"] for r in all_results])
                print(f"  mean_override_rate: {avg_override_rate:.1f}%")

            # 保存结果
            csv_path = args.save_path or "logs/multiseed_eval.csv"
            ensure_dir(os.path.dirname(csv_path))
            records = []
            for i, r in enumerate(all_results):
                records.append({
                    "seed_index": i,
                    "seed": args.seed + i * 1000,
                    "avg_score": r["avg_score"],
                    "max_score": r["max_score"],
                    "min_score": r["min_score"],
                    "avg_steps": r["avg_steps"],
                    "self_collision_rate": r["rates"]["self_collision"],
                    "wall_collision_rate": r["rates"]["wall_collision"],
                    "timeout_rate": r["rates"]["no_food_timeout"],
                    "override_rate": r["override_rate"],
                    "safety_filter": args.safety_filter,
                })
            save_train_log(csv_path, records)
            print(f"\n详细记录已保存: {csv_path}")
    else:
        evaluate(args.model, args.episodes, args.grid_size, args.state_mode, args.seed, args.save_path, args.safety_filter)
