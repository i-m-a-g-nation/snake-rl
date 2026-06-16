"""终端输入模块，Windows 优先使用 msvcrt 实现非阻塞键盘输入。"""

import sys
import os

# Windows 实现
if sys.platform == "win32":
    import msvcrt

    def get_key_nonblocking():
        """非阻塞获取按键，返回统一字符串或 None。"""
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getwch()
        # 方向键前缀: 0x00 或 0xE0
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            arrow_map = {
                "H": "UP",
                "P": "DOWN",
                "K": "LEFT",
                "M": "RIGHT",
            }
            return arrow_map.get(ch2, None)
        # 普通按键
        if ch == "q" or ch == "Q":
            return "Q"
        if ch == "r" or ch == "R":
            return "R"
        if ch == " ":
            return "SPACE"
        return None

    def clear_screen():
        os.system("cls")

    def hide_cursor():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def show_cursor():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    def move_cursor_home():
        sys.stdout.write("\033[H")
        sys.stdout.flush()

else:
    # Unix fallback (非 Windows)
    import select
    import tty
    import termios

    _old_settings = None

    def get_key_nonblocking():
        """Unix 非阻塞按键。"""
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                arrow_map = {"A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT"}
                return arrow_map.get(ch3, None)
        if ch in ("q", "Q"):
            return "Q"
        if ch in ("r", "R"):
            return "R"
        if ch == " ":
            return "SPACE"
        return None

    def clear_screen():
        os.system("clear")

    def hide_cursor():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def show_cursor():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    def move_cursor_home():
        sys.stdout.write("\033[H")
        sys.stdout.flush()


def parse_arrow_key(key_str):
    """解析方向键字符串，返回方向常量或 None。"""
    if key_str == "UP":
        return 0  # DIR_UP
    elif key_str == "RIGHT":
        return 1  # DIR_RIGHT
    elif key_str == "DOWN":
        return 2  # DIR_DOWN
    elif key_str == "LEFT":
        return 3  # DIR_LEFT
    return None
