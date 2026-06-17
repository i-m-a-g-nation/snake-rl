"""Stable-Baselines3 DQN 训练脚本。"""

import argparse
import os
import sys

# 检查 stable_baselines3 是否安装
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from snake_env import SnakeEnv
from utils import ensure_dir


def train(args):
    """训练 SB3 DQN Agent。"""
    if not HAS_SB3:
        print("错误: stable_baselines3 未安装。")
        print("请运行: pip install stable-baselines3")
        sys.exit(1)

    ensure_dir("checkpoints")
    ensure_dir("logs")

    # 创建环境
    env = SnakeEnv(grid_size=20, seed=args.seed, state_mode=args.state_mode)

    # 检查环境兼容性
    print("检查环境兼容性...")
    check_env(env)
    print("环境检查通过!")

    print(f"状态模式: {args.state_mode}")
    print(f"状态维度: {env.observation_space_dim}")
    print(f"训练步数: {args.timesteps}")
    print("-" * 70)

    # 创建 DQN 模型
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log="logs/",
    )

    # 训练
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
    )

    # 保存模型
    save_path = args.save_path or "checkpoints/sb3_dqn_snake.zip"
    model.save(save_path)
    print(f"\n训练完成! 模型已保存到: {save_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake SB3 DQN 训练")
    parser.add_argument("--timesteps", type=int, default=200000, help="训练步数")
    parser.add_argument("--state-mode", type=str, default="basic17", choices=["basic17", "reachable23"], help="状态模式")
    parser.add_argument("--save-path", type=str, default=None, help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--buffer-size", type=int, default=100000, help="回放缓冲区大小")
    parser.add_argument("--learning-starts", type=int, default=5000, help="开始训练步数")
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--train-freq", type=int, default=4, help="训练频率")
    parser.add_argument("--gradient-steps", type=int, default=1, help="梯度步数")
    parser.add_argument("--target-update", type=int, default=1000, help="目标网络更新间隔")
    parser.add_argument("--exploration-fraction", type=float, default=0.4, help="探索率衰减比例")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="最终探索率")
    args = parser.parse_args()
    train(args)
