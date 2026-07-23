from io import StringIO

import pytest

from sokoban_agent.env import Action
from sokoban_agent.play import parse_key, play


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("w", Action.UP),
        ("\x1b[A", Action.UP),
        ("d", Action.RIGHT),
        ("\x1b[C", Action.RIGHT),
        ("s", Action.DOWN),
        ("\x1b[B", Action.DOWN),
        ("a", Action.LEFT),
        ("\x1b[D", Action.LEFT),
        ("r", "reset"),
        ("q", "quit"),
        ("x", None),
    ],
)
def test_parse_key(key: str, expected: Action | str | None) -> None:
    assert parse_key(key) == expected


def test_play_can_complete_tiny_walk_and_quit() -> None:
    keys = iter(["w", "d", "s", "d", "w", "q"])
    output = StringIO()

    play(
        level_id="tiny-walk",
        key_reader=lambda: next(keys),
        output=output,
    )

    rendered = output.getvalue()
    assert "🎉 성공!" in rendered
    assert "이동: 5" in rendered
    assert "게임을 종료했습니다." in rendered
