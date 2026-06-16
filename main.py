"""加载训练好的模型，观看 Agent 玩 Snake。"""

import argparse
import os
import sys
import time

import numpy as np

from snake_env import SnakeEnv
from agent import DQNAgent

# 尝试导入 pygame
try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def watch_terminal(model_path: str, episodes: int = 5, grid_size: int = 20):
    """终端模式观看 Agent。"""
    env = SnakeEnv(grid_size=grid_size, max_steps=1000)
    agent = DQNAgent(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
    )
    agent.load(model_path)
    agent.policy_net.eval()

    print(f"加载模型: {model_path}")
    print(f"设备: {agent.device}")
    print(f"观看 {episodes} 局...")
    print("=" * 50)

    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=ep * 100)
        done = False
        total_reward = 0.0
        actions_taken = []

        while not done:
            with np.errstate(all="ignore"):
                action = agent.select_action(state, epsilon=0.0)  # 贪心，无探索
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            actions_taken.append(action)
            state = next_state
            total_reward += reward

        score = info["score"]
        action_names = {0: "直行", 1: "左转", 2: "右转"}
        action_summary = [action_names[a] for a in actions_taken[-10:]]

        print(
            f"Episode {ep:2d} | Score: {score:3d} | "
            f"Steps: {info['steps']:5d} | Reward: {total_reward:8.2f} | "
            f"最后动作: {' '.join(action_summary)}"
        )

    env.close()
    print("=" * 50)
    print("观看结束。")


def watch_pygame(model_path: str, episodes: int = 5, grid_size: int = 20, fps: int = 10):
    """Pygame 图形界面观看 Agent。"""
    CELL_SIZE = 30
    WIDTH = grid_size * CELL_SIZE
    HEIGHT = grid_size * CELL_SIZE + 40

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake - DQN Agent Watch")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    env = SnakeEnv(grid_size=grid_size, max_steps=1000)
    agent = DQNAgent(
        state_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
    )
    agent.load(model_path)
    agent.policy_net.eval()

    print(f"加载模型: {model_path}")

    for ep in range(1, episodes + 1):
        state, info = env.reset(seed=ep * 100)
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

            # 绘制
            screen.fill((0, 0, 0))
            game = env.game

            for r, c in game.snake:
                color = (0, 200, 0) if (r, c) == game.snake[0] else (0, 150, 0)
                pygame.draw.rect(
                    screen,
                    color,
                    (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
                )

            if game.food:
                fr, fc = game.food
                pygame.draw.rect(
                    screen,
                    (255, 0, 0),
                    (fc * CELL_SIZE, fr * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
                )

            score_text = font.render(
                f"Ep:{ep} Score:{game.score} Steps:{game.steps}", True, (255, 255, 255)
            )
            screen.blit(score_text, (10, HEIGHT - 35))

            pygame.display.flip()
            clock.tick(fps)

        print(f"Episode {ep} | Score: {info['score']}")

    pygame.quit()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="观看 DQN Agent 玩 Snake")
    parser.add_argument("--model", type=str, default="checkpoints/dqn_snake.pt", help="模型路径")
    parser.add_argument("--episodes", type=int, default=5, help="观看局数")
    parser.add_argument("--grid-size", type=int, default=20, help="网格大小")
    parser.add_argument("--fps", type=int, default=10, help="帧率 (仅 pygame)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        print("请先运行训练: python train.py --episodes 1000")
        sys.exit(1)

    if HAS_PYGAME:
        print("使用 pygame 图形界面观看。")
        watch_pygame(args.model, args.episodes, args.grid_size, args.fps)
    else:
        print("使用终端模式观看 (安装 pygame 可启用图形界面)。")
        watch_terminal(args.model, args.episodes, args.grid_size)
