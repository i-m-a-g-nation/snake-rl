"""Stable-Baselines3 DQN 训练脚本。支持继续训练、EpisodeStatsCallback。"""

import argparse
import os
import sys
import csv

# 检查 stable_baselines3 是否安装
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

import numpy as np
import torch
from snake_env import SnakeEnv
from utils import ensure_dir


class EpisodeStatsCallback(BaseCallback):
    """统计 episode 信息的 Callback。"""

    def __init__(self, log_dir: str, log_interval: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.episode_count = 0
        self.episode_lengths = []
        self.episode_rewards = []
        self.recent_scores = []
        self.csv_path = os.path.join(log_dir, "episode_stats.csv")
        self._last_log_timesteps = 0

    def _init_callback(self):
        ensure_dir(self.log_dir)
        # 写 CSV header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "episodes", "avg_episode_length", "recent_avg_score", "recent_avg_reward"])

    def _on_step(self):
        # 检查是否有 episode 结束
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if "r" in info and "l" in info:
                    ep_reward = info["r"]
                    ep_length = info["l"]
                    ep_score = info.get("score", 0)

                    self.episode_count += 1
                    self.episode_lengths.append(ep_length)
                    self.episode_rewards.append(ep_reward)
                    self.recent_scores.append(ep_score)

                    # 只保留最近 100 个
                    if len(self.recent_scores) > 100:
                        self.recent_scores = self.recent_scores[-100:]
                    if len(self.episode_lengths) > 100:
                        self.episode_lengths = self.episode_lengths[-100:]
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards = self.episode_rewards[-100:]

        # 定期记录
        if self.num_timesteps - self._last_log_timesteps >= self.log_interval:
            self._last_log_timesteps = self.num_timesteps
            avg_len = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            avg_score = np.mean(self.recent_scores[-100:]) if self.recent_scores else 0
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0

            # 写 CSV
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    self.episode_count,
                    round(avg_len, 1),
                    round(avg_score, 2),
                    round(avg_reward, 2),
                ])

            if self.verbose > 0:
                print(f"  Timesteps: {self.num_timesteps} | Episodes: {self.episode_count} | "
                      f"AvgEpLen: {avg_len:.1f} | RecentAvgScore: {avg_score:.2f}")

        return True


def train(args):
    """训练 SB3 DQN Agent。"""
    if not HAS_SB3:
        print("错误: stable_baselines3 未安装。")
        print("请运行: pip install stable-baselines3")
        sys.exit(1)

    # 目录结构
    run_dir = f"checkpoints/sb3_runs/{args.run_name}"
    log_dir = f"logs/sb3_runs/{args.run_name}"
    best_model_dir = os.path.join(run_dir, "best_model")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    eval_log_dir = os.path.join(log_dir, "eval")

    ensure_dir(run_dir)
    ensure_dir(log_dir)
    ensure_dir(best_model_dir)
    ensure_dir(checkpoint_dir)
    ensure_dir(eval_log_dir)

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
    print(f"Run 名称: {args.run_name}")
    print(f"Best model 目录: {best_model_dir}")
    print("-" * 70)

    # 加载或创建模型
    if args.load_model and os.path.exists(args.load_model):
        print(f"从已有模型继续训练: {args.load_model}")
        model = DQN.load(args.load_model, env=train_env, device=device)
        reset_num_timesteps = not args.continue_training
    else:
        if args.load_model:
            print(f"警告: 模型文件不存在 {args.load_model}，将从零开始训练")
        print("创建新模型...")
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
            device=device,
        )
        reset_num_timesteps = True

    # 创建 Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="dqn_snake",
    )

    episode_stats_callback = EpisodeStatsCallback(
        log_dir=log_dir,
        log_interval=args.log_interval,
        verbose=1,
    )

    callback = CallbackList([eval_callback, checkpoint_callback, episode_stats_callback])

    # 训练
    print("开始训练...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=args.run_name,
    )

    # 保存 final model
    final_model_path = os.path.join(run_dir, "final_model.zip")
    model.save(final_model_path)

    print(f"\n{'=' * 70}")
    print(f"训练完成!")
    print(f"Run 名称: {args.run_name}")
    print(f"Final model: {final_model_path}")
    print(f"Best model: {best_model_dir}/best_model.zip")
    print(f"状态模式: {args.state_mode}")
    print(f"训练步数: {args.timesteps}")
    print(f"设备: {device}")
    print(f"\n评估命令:")
    print(f"  python evaluate_sb3.py --model {best_model_dir}/best_model.zip --episodes 100 --state-mode {args.state_mode}")
    print(f"\n观看命令:")
    print(f"  python main.py --model {best_model_dir}/best_model.zip --model-type sb3 --episodes 5 --fps 10 --terminal-render --state-mode {args.state_mode}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake SB3 DQN 训练")
    parser.add_argument("--timesteps", type=int, default=200000, help="训练步数")
    parser.add_argument("--state-mode", type=str, default="basic17", choices=["basic17", "reachable23"], help="状态模式")
    parser.add_argument("--run-name", type=str, default="sb3_dqn_basic17_200k", help="运行名称")
    parser.add_argument("--load-model", type=str, default=None, help="从已有模型继续训练")
    parser.add_argument("--continue-training", action="store_true", help="继续训练（不重置 timestep 计数）")
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
    parser.add_argument("--log-interval", type=int, default=5000, help="日志记录间隔 (steps)")
    args = parser.parse_args()
    train(args)
