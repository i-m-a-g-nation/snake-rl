"""人工游玩 Snake - 终端实时刷新版。"""

import sys
import time

from snake_game import (
    SnakeGame, ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
    DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT,
    absolute_direction_to_relative_action,
)
from terminal_input import (
    get_key_nonblocking, clear_screen, hide_cursor, show_cursor,
    move_cursor_home, parse_arrow_key,
)

# 尝试导入 pygame
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

DIRECTION_NAMES = {DIR_UP: "UP", DIR_RIGHT: "RIGHT", DIR_DOWN: "DOWN", DIR_LEFT: "LEFT"}


def play_terminal_realtime(fps: int = 8):
    """终端实时刷新版人工游玩。"""
    game = SnakeGame(grid_size=15, seed=None)
    paused = False
    last_action = ACTION_STRAIGHT

    hide_cursor()
    clear_screen()

    try:
        while True:
            # 处理输入
            key = get_key_nonblocking()
            if key == "Q":
                break
            elif key == "R":
                game.reset()
                paused = False
                last_action = ACTION_STRAIGHT
            elif key == "SPACE":
                paused = not paused
            elif key in ("UP", "DOWN", "LEFT", "RIGHT"):
                target_dir = parse_arrow_key(key)
                if target_dir is not None:
                    action = absolute_direction_to_relative_action(game.direction, target_dir)
                    last_action = action

            # 游戏步进
            if not paused and not game.done:
                result = game.step(last_action)
                last_action = ACTION_STRAIGHT  # 重置为直行

            # 渲染
            move_cursor_home()
            _render_frame(game, paused, fps)

            # 游戏结束处理
            if game.done:
                # 等待用户输入 r 重开或 q 退出
                while True:
                    k = get_key_nonblocking()
                    if k == "Q":
                        return
                    elif k == "R":
                        game.reset()
                        paused = False
                        last_action = ACTION_STRAIGHT
                        break
                    time.sleep(0.05)

            time.sleep(1.0 / fps)

    finally:
        show_cursor()


def _render_frame(game: SnakeGame, paused: bool, fps: int):
    """渲染一帧到终端。"""
    grid_size = game.grid_size
    lines = []

    # 标题
    lines.append(" Snake - Terminal Realtime")
    lines.append("")

    # 顶部边框
    lines.append("+" + "---" * grid_size + "+")

    # 网格
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
    for i, (r, c) in enumerate(game.snake):
        if 0 <= r < grid_size and 0 <= c < grid_size:
            grid[r][c] = "O" if i == 0 else "o"
    if game.food:
        fr, fc = game.food
        grid[fr][fc] = "*"
    for row in grid:
        lines.append("|" + " ".join(f"{ch} " for ch in row) + "|")

    # 底部边框
    lines.append("+" + "---" * grid_size + "+")

    # 状态信息
    lines.append(f"  Score: {game.score}   Steps: {game.steps}   FPS: {fps}")
    lines.append(f"  Direction: {DIRECTION_NAMES.get(game.direction, '?')}")
    if paused:
        lines.append("  [PAUSED]")
    if game.done:
        lines.append("  [GAME OVER] Press R to restart, Q to quit")
    else:
        lines.append("  Controls: Arrows=Move  SPACE=Pause  R=Restart  Q=Quit")

    # 填充空行避免残影
    output = "\n".join(lines)
    sys.stdout.write(output)
    sys.stdout.flush()


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
    current_action = ACTION_STRAIGHT

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
                    current_action = ACTION_STRAIGHT
                elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    key_map = {
                        pygame.K_UP: DIR_UP,
                        pygame.K_DOWN: DIR_DOWN,
                        pygame.K_LEFT: DIR_LEFT,
                        pygame.K_RIGHT: DIR_RIGHT,
                    }
                    target_dir = key_map[event.key]
                    current_action = absolute_direction_to_relative_action(
                        game.direction, target_dir
                    )

        # 执行动作
        if not game.done:
            game.step(current_action)
            current_action = ACTION_STRAIGHT

        # 绘制
        screen.fill((0, 0, 0))
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
        score_text = font.render(f"Score: {game.score}  Steps: {game.steps}", True, (255, 255, 255))
        screen.blit(score_text, (10, HEIGHT - 35))

        pygame.display.flip()
        clock.tick(10)

        if game.done:
            print(f"Game Over! Score: {game.score}")
            game.reset()

    pygame.quit()


if __name__ == "__main__":
    if HAS_PYGAME:
        print("检测到 pygame，使用图形界面模式。")
        play_pygame()
    else:
        play_terminal_realtime(fps=8)
