"""Interactive terminal player for the built-in Sokoban levels."""

from __future__ import annotations

import argparse
import sys
import termios
import tty
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal, TextIO

from sokoban_agent.env import Action, SokobanEnv

Command = Action | Literal["reset", "quit"]

_KEY_COMMANDS: dict[str, Command] = {
    "w": Action.UP,
    "W": Action.UP,
    "\x1b[A": Action.UP,
    "d": Action.RIGHT,
    "D": Action.RIGHT,
    "\x1b[C": Action.RIGHT,
    "s": Action.DOWN,
    "S": Action.DOWN,
    "\x1b[B": Action.DOWN,
    "a": Action.LEFT,
    "A": Action.LEFT,
    "\x1b[D": Action.LEFT,
    "r": "reset",
    "R": "reset",
    "q": "quit",
    "Q": "quit",
    "\x03": "quit",
    "": "quit",
}


def parse_key(key: str) -> Command | None:
    """Translate one terminal key sequence into a game command."""

    return _KEY_COMMANDS.get(key)


def read_key(stream: TextIO = sys.stdin) -> str:
    """Read one key, including a three-byte terminal arrow sequence."""

    key = stream.read(1)
    if key == "\x1b":
        key += stream.read(2)
    return key


@contextmanager
def raw_terminal(stream: TextIO = sys.stdin) -> Iterator[None]:
    """Temporarily make terminal key presses available without Enter."""

    file_descriptor = stream.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        yield
    finally:
        termios.tcsetattr(
            file_descriptor,
            termios.TCSADRAIN,
            previous_settings,
        )


def _screen(
    env: SokobanEnv,
    info: dict[str, object],
    *,
    finished: bool,
) -> str:
    board = env.render()
    if not isinstance(board, str):
        raise RuntimeError("terminal play requires ANSI rendering")

    status = ""
    if info["success"]:
        status = "\n🎉 성공! R로 다시 시작하거나 Q로 종료하세요."
    elif info["deadlock"]:
        status = "\n상자가 모서리에 갇혔습니다. R로 다시 시작하세요."
    elif finished:
        status = "\n이동 횟수를 모두 사용했습니다. R로 다시 시작하세요."
    elif info["invalid_move"]:
        status = "\n그 방향으로는 이동할 수 없습니다."

    return (
        "\033[2J\033[H"
        f"Sokoban · {info['level_id']}\n\n"
        f"{board}\n\n"
        f"이동: {info['steps']}  "
        f"목표 위 상자: {info['boxes_on_targets']}\n"
        "방향키/WASD 이동 · R 재시작 · Q 종료"
        f"{status}\n"
    )


def play(
    *,
    level_id: str = "tiny-walk",
    key_reader: Callable[[], str] = read_key,
    output: TextIO = sys.stdout,
) -> None:
    """Run an interactive game until the player quits."""

    env = SokobanEnv(render_mode="ansi")
    _, info = env.reset(options={"level_id": level_id})
    finished = False

    try:
        while True:
            output.write(_screen(env, info, finished=finished))
            output.flush()

            command = parse_key(key_reader())
            if command == "quit":
                break
            if command == "reset":
                _, info = env.reset(options={"level_id": level_id})
                finished = False
            elif isinstance(command, Action) and not finished:
                _, _, terminated, truncated, info = env.step(command)
                finished = terminated or truncated
    finally:
        env.close()
        output.write("\n게임을 종료했습니다.\n")
        output.flush()


def main() -> None:
    """Parse command-line options and start terminal play."""

    parser = argparse.ArgumentParser(description="터미널에서 Sokoban을 플레이합니다.")
    parser.add_argument(
        "--level",
        choices=("tiny-push", "tiny-walk"),
        default="tiny-walk",
        help="플레이할 기본 레벨 (기본값: tiny-walk)",
    )
    args = parser.parse_args()

    if not sys.stdin.isatty():
        parser.error("키보드 입력이 가능한 터미널에서 실행해 주세요.")

    with raw_terminal():
        play(level_id=args.level)


if __name__ == "__main__":
    main()
