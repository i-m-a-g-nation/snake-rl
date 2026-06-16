"""DQN 训练入口。"""

import argparse
import os
import sys
import time

import numpy as np

from snake_env import SnakeEnv
from agent import DQNAgent
from utils import ensure_dir, append_train_log


def train(args):
    """训练 DQN Agent。"""
    # 创建目录
    ensure_dir("checkpoints")
    ensure_dir("logs")

    # 环境
    env = SnakeEnv(grid_size=20, max_steps=1000, seed=args.seed)
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim

    # Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_size=50000,
        target_update_interval=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
    )

    print(f"设备: {agent.device}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"训练 episodes: {args.episodes}")
    print("-" * 70)

    log_path = "logs/train_log.csv"
    best_score = 0
    scores = []
    total_rewards = []

    for episode in range(1, args.episodes + 1):
        state, info = env.reset(seed=args.seed + episode if args.seed else None)
        episode_reward = 0.0
        done = False

        while not done:
            epsilon = agent.compute_epsilon()
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            episode_reward += reward

        score = info["score"]
        scores.append(score)
        total_rewards.append(episode_reward)
        if score > best_score:
            best_score = score

        # 日志
        log_record = {
            "episode": episode,
            "score": score,
            "total_reward": round(episode_reward, 2),
            "epsilon": round(epsilon, 4),
            "best_score": best_score,
            "loss": round(loss, 4) if loss else 0,
        }
        append_train_log(log_path, log_record)

        # 打印
        if episode % 10 == 0 or episode == 1:
            avg_score = np.mean(scores[-50:])
            avg_reward = np.mean(total_rewards[-50:])
            print(
                f"Episode {episode:5d} | "
                f"Score: {score:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Eps: {epsilon:.4f} | "
                f"Best: {best_score:3d} | "
                f"Avg50: {avg_score:.1f}"
            )

        # 定期保存
        if episode % 100 == 0:
            agent.save(f"checkpoints/dqn_snake_ep{episode}.pt")

    # 最终保存
    save_path = args.save_path or "checkpoints/dqn_snake.pt"
    agent.save(save_path)
    print(f"\n训练完成! 模型已保存到: {save_path}")
    print(f"最佳得分: {best_score}")
    print(f"训练日志: {log_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake DQN 训练")
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--render", action="store_true", help="是否渲染 (终端模式)")
    parser.add_argument("--save-path", type=str, default=None, help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    train(args)
