"""DQN 训练入口。支持 Double DQN、评估、best model 保存。"""

import argparse
import os
import sys
import time

import numpy as np

from snake_env import SnakeEnv
from agent import DQNAgent
from utils import ensure_dir, append_train_log


def evaluate_agent(agent: DQNAgent, grid_size: int = 20, eval_episodes: int = 20, seed: int = 1000):
    """评估 Agent（epsilon=0，不训练）。返回 (avg_score, max_score, avg_steps)。"""
    env = SnakeEnv(grid_size=grid_size, seed=seed)
    scores = []
    steps_list = []

    for ep in range(eval_episodes):
        state, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["score"])
        steps_list.append(info["steps"])

    env.close()
    return np.mean(scores), max(scores), np.mean(steps_list)


def train(args):
    """训练 DQN Agent。"""
    ensure_dir("checkpoints")
    ensure_dir("logs")

    # 环境
    env = SnakeEnv(grid_size=20, seed=args.seed)
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim

    # Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        gamma=0.99,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        target_update_interval=args.target_update,
        epsilon_start=1.0,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay,
        use_double_dqn=args.double_dqn,
    )

    print(f"设备: {agent.device}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"Double DQN: {args.double_dqn}")
    print(f"训练 episodes: {args.episodes}")
    print(f"Warmup steps: {args.warmup_steps}")
    print("-" * 70)

    log_path = "logs/train_log.csv"
    best_eval_score = -float("inf")
    scores = []
    total_rewards = []
    total_steps = 0
    no_food_timeout_total = 0
    repeat_penalty_total = 0

    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + episode if args.seed else None)
        episode_reward = 0.0
        done = False
        ep_no_food_timeout = 0
        ep_repeat_penalty = 0

        while not done:
            epsilon = agent.compute_epsilon()
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            # Warmup: 攒够经验后再训练
            if total_steps >= args.warmup_steps:
                loss = agent.train_step()
            else:
                loss = 0.0

            state = next_state
            episode_reward += reward
            total_steps += 1

            if info.get("no_food_timeout"):
                ep_no_food_timeout += 1
            if info.get("repeat_penalty"):
                ep_repeat_penalty += 1

        score = info["score"]
        scores.append(score)
        total_rewards.append(episode_reward)
        no_food_timeout_total += ep_no_food_timeout
        repeat_penalty_total += ep_repeat_penalty

        avg50_score = np.mean(scores[-50:])

        # 日志记录
        log_record = {
            "episode": episode,
            "score": score,
            "total_reward": round(episode_reward, 2),
            "epsilon": round(epsilon, 4),
            "best_score": max(scores),
            "loss": round(loss, 4) if loss else 0,
            "avg50_score": round(avg50_score, 2),
            "eval_avg_score": "",
            "eval_max_score": "",
            "no_food_timeout_count": ep_no_food_timeout,
            "repeat_penalty_count": ep_repeat_penalty,
        }

        # 定期评估
        if episode % args.eval_interval == 0:
            eval_avg, eval_max, eval_steps = evaluate_agent(
                agent, grid_size=20, eval_episodes=args.eval_episodes, seed=2000
            )
            log_record["eval_avg_score"] = round(eval_avg, 2)
            log_record["eval_max_score"] = eval_max

            # 保存 best model
            if eval_avg > best_eval_score:
                best_eval_score = eval_avg
                agent.save("checkpoints/best_model.pt")

            print(
                f"Episode {episode:5d} | "
                f"Score: {score:3d} | "
                f"Avg50: {avg50_score:5.1f} | "
                f"Eps: {epsilon:.4f} | "
                f"EvalAvg: {eval_avg:5.1f} | "
                f"EvalMax: {eval_max:3d}"
            )
        elif episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:5d} | "
                f"Score: {score:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Eps: {epsilon:.4f} | "
                f"Avg50: {avg50_score:5.1f}"
            )

        append_train_log(log_path, log_record)

        # 定期保存 checkpoint
        if episode % 500 == 0:
            agent.save(f"checkpoints/dqn_snake_ep{episode}.pt")

    # 最终保存
    save_path = args.save_path or "checkpoints/final_model.pt"
    agent.save(save_path)

    print(f"\n{'=' * 70}")
    print(f"训练完成!")
    print(f"最终模型: {save_path}")
    print(f"最佳模型: checkpoints/best_model.pt")
    print(f"最佳 eval 平均分: {best_eval_score:.2f}")
    print(f"训练日志: {log_path}")
    print(f"绕圈超时总次数: {no_food_timeout_total}")
    print(f"重复惩罚总次数: {repeat_penalty_total}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake DQN 训练")
    parser.add_argument("--episodes", type=int, default=3000, help="训练轮数")
    parser.add_argument("--render", action="store_true", help="是否渲染 (会变慢)")
    parser.add_argument("--save-path", type=str, default=None, help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--double-dqn", action="store_true", default=True, help="使用 Double DQN")
    parser.add_argument("--no-double-dqn", dest="double_dqn", action="store_false", help="禁用 Double DQN")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--replay-size", type=int, default=100000, help="回放缓冲区大小")
    parser.add_argument("--target-update", type=int, default=500, help="目标网络更新间隔")
    parser.add_argument("--epsilon-end", type=float, default=0.02, help="最终探索率")
    parser.add_argument("--epsilon-decay", type=int, default=50000, help="探索率衰减步数")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="预热步数")
    parser.add_argument("--eval-interval", type=int, default=100, help="评估间隔 (episodes)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="评估局数")
    args = parser.parse_args()
    train(args)
