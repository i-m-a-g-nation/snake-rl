"""人工游玩 Snake (终端版)。"""

import os
import sys
import time

from snake_game import SnakeGame, ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT

# 尝试导入 pygame
try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


def play_terminal():
    """终端版人工游玩。"""
    game = SnakeGame(grid_size=15, seed=None)
    print("=" * 50)
    print("Snake 终端版 - 人工游玩")
    print("=" * 50)
    print("控制方式:")
    print("  w / i = 左转 (相对于当前方向)")
    print("  o     = 直行")
    print("  p     = 右转 (相对于当前方向)")
    print("  q     = 退出")
    print("  r     = 重新开始")
    print("=" * 50)
    input("按 Enter 开始...")

    action_map = {"w": ACTION_LEFT, "i": ACTION_LEFT, "o": ACTION_STRAIGHT, "p": ACTION_RIGHT}

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(game.render_terminal())
        print("动作: w/i=左转, o=直行, p=右转, q=退出, r=重开")

        try:
            key = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == "q":
            break
        if key == "r":
            game.reset()
            continue

        if key in action_map:
            result = game.step(action_map[key])
            if result["done"]:
                os.system("cls" if os.name == "nt" else "clear")
                print(game.render_terminal())
                print(f"\n游戏结束! 最终得分: {result['score']}")
                again = input("再来一局? (y/n): ").strip().lower()
                if again == "y":
                    game.reset()
                    continue
                else:
                    break
        else:
            print("无效输入，请使用 w/i/o/p/q/r")

    print("感谢游玩!")


def play_pygame():
    """Pygame 版人工游玩。"""
    CELL_SIZE = 30
    GRID_SIZE = 20
    WIDTH = GRID_SIZE * CELL_SIZE
    HEIGHT = GRID_SIZE * CELL_SIZE + 40

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake - Human Play")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    game = SnakeGame(grid_size=GRID_SIZE)
    # 方向映射: 上下左右 -> 动作
    dir_to_action = {}  # 需要根据当前方向计算

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key in (pygame.K_UP, pygame.K_w):
                    # 计算需要转向到朝上的动作
                    target_dir = 0  # UP
                    action = _dir_to_action(game.direction, target_dir)
                    if action is not None:
                        result = game.step(action)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    target_dir = 1  # RIGHT
                    action = _dir_to_action(game.direction, target_dir)
                    if action is not None:
                        result = game.step(action)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    target_dir = 2  # DOWN
                    action = _dir_to_action(game.direction, target_dir)
                    if action is not None:
                        result = game.step(action)
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    target_dir = 3  # LEFT
                    action = _dir_to_action(game.direction, target_dir)
                    if action is not None:
                        result = game.step(action)

        # 绘制
        screen.fill((0, 0, 0))

        # 绘制网格
        for r, c in game.snake:
            color = (0, 200, 0) if (r, c) == game.snake[0] else (0, 150, 0)
            pygame.draw.rect(
                screen, color, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            )

        if game.food:
            fr, fc = game.food
            pygame.draw.rect(
                screen, (255, 0, 0), (fc * CELL_SIZE, fr * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            )

        # 分数
        score_text = font.render(f"Score: {game.score}  Steps: {game.steps}", True, (255, 255, 255))
        screen.blit(score_text, (10, HEIGHT - 35))

        pygame.display.flip()
        clock.tick(10)

        if game.done:
            print(f"Game Over! Score: {game.score}")
            game.reset()

    pygame.quit()


def _dir_to_action(current_dir: int, target_dir: int) -> int:
    """根据当前方向和目标方向计算动作。"""
    diff = (target_dir - current_dir) % 4
    if diff == 0:
        return 0  # 直行
    elif diff == 1:
        return 2  # 右转
    elif diff == 3:
        return 1  # 左转
    else:
        return None  # 反向，不允许


if __name__ == "__main__":
    if HAS_PYGAME:
        print("检测到 pygame，使用图形界面模式。")
        play_pygame()
    else:
        print("未检测到 pygame，使用终端模式。")
        play_terminal()
