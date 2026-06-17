"""Stable-Baselines3 DQN 训练脚本。支持 EvalCallback、CheckpointCallback、TensorBoard。"""

import argparse
import os
import sys

# 检查 stable_baselines3 是否安装
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

import torch
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
    ensure_dir(args.best_model_dir)
    ensure_dir(args.log_dir)
    ensure_dir("checkpoints/sb3_checkpoints")

    # 创建训练环境和评估环境
    train_env = Monitor(SnakeEnv(state_mode=args.state_mode, seed=args.seed))
    eval_env = Monitor(SnakeEnv(state_mode=args.state_mode, seed=args.seed + 1000))

    # 检查环境兼容性
    print("检查环境兼容性...")
    check_env(SnakeEnv(state_mode=args.state_mode), warn=True)
    print("环境检查通过!")

    # 设备信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"状态模式: {args.state_mode}")
    print(f"训练步数: {args.timesteps}")
    print(f"评估频率: 每 {args.eval_freq} steps")
    print(f"Checkpoint 频率: 每 {args.checkpoint_freq} steps")
    print(f"Best model 保存目录: {args.best_model_dir}")
    print("-" * 70)

    # 创建 DQN 模型
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        device=device,
    )

    # 创建 Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.best_model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path="checkpoints/sb3_checkpoints",
        name_prefix="dqn_snake",
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    # 训练
    print("开始训练...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name="dqn_snake_basic17",
    )

    # 保存 final model
    save_path = args.save_path or "checkpoints/sb3_dqn_snake.zip"
    model.save(save_path)

    print(f"\n{'=' * 70}")
    print(f"训练完成!")
    print(f"Final model: {save_path}")
    print(f"Best model: {args.best_model_dir}/best_model.zip")
    print(f"状态模式: {args.state_mode}")
    print(f"训练步数: {args.timesteps}")
    print(f"设备: {device}")
    print(f"\n评估命令:")
    print(f"  python evaluate_sb3.py --model {args.best_model_dir}/best_model.zip --episodes 100 --state-mode {args.state_mode}")
    print(f"\n观看命令:")
    print(f"  python main.py --model {args.best_model_dir}/best_model.zip --model-type sb3 --episodes 5 --fps 10 --terminal-render --state-mode {args.state_mode}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake SB3 DQN 训练")
    parser.add_argument("--timesteps", type=int, default=200000, help="训练步数")
    parser.add_argument("--state-mode", type=str, default="basic17", choices=["basic17", "reachable23"], help="状态模式")
    parser.add_argument("--save-path", type=str, default=None, help="Final 模型保存路径")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard 日志目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--buffer-size", type=int, default=100000, help="回放缓冲区大小")
    parser.add_argument("--learning-starts", type=int, default=10000, help="开始训练步数")
    parser.add_argument("--batch-size", type=int, default=128, help="批量大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--train-freq", type=int, default=4, help="训练频率")
    parser.add_argument("--gradient-steps", type=int, default=1, help="梯度步数")
    parser.add_argument("--target-update-interval", type=int, default=1000, help="目标网络更新间隔")
    parser.add_argument("--exploration-fraction", type=float, default=0.5, help="探索率衰减比例")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="最终探索率")
    parser.add_argument("--eval-freq", type=int, default=10000, help="评估频率 (steps)")
    parser.add_argument("--eval-episodes", type=int, default=20, help="评估局数")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint 频率 (steps)")
    parser.add_argument("--best-model-dir", type=str, default="checkpoints/sb3_best", help="Best model 保存目录")
    parser.add_argument("--log-dir", type=str, default="logs/sb3", help="评估日志目录")
    args = parser.parse_args()
    train(args)
