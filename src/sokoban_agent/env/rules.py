"""Pure Sokoban rules shared by the Gymnasium environment and search agents."""

from __future__ import annotations

import numpy as np

from sokoban_agent.env.levels import Position, SokobanLevel
from sokoban_agent.env.model import Action, MoveResult, SokobanState, Tile

_DIRECTIONS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}


def initial_state(level: SokobanLevel) -> SokobanState:
    """Build the immutable starting state for a level."""

    return SokobanState(player=level.player, boxes=level.boxes)


def apply_action(
    level: SokobanLevel,
    state: SokobanState,
    action: Action,
) -> MoveResult:
    """Apply one primitive action without mutating the input state."""

    row_delta, column_delta = _DIRECTIONS[action]
    player = state.player
    destination = (player[0] + row_delta, player[1] + column_delta)
    invalid_move = destination in level.walls
    pushed = False
    box_left_target = False
    box_entered_target = False
    boxes = state.boxes

    if not invalid_move and destination in boxes:
        beyond = (
            destination[0] + row_delta,
            destination[1] + column_delta,
        )
        if beyond in level.walls or beyond in boxes:
            invalid_move = True
        else:
            pushed = True
            box_left_target = destination in level.targets
            box_entered_target = beyond in level.targets
            boxes = frozenset((*boxes - {destination}, beyond))

    next_state = state
    if not invalid_move:
        next_state = SokobanState(player=destination, boxes=boxes)

    return MoveResult(
        state=next_state,
        invalid_move=invalid_move,
        pushed=pushed,
        box_left_target=box_left_target,
        box_entered_target=box_entered_target,
    )


def is_success(level: SokobanLevel, state: SokobanState) -> bool:
    """Return whether every box is on a target."""

    return state.boxes == level.targets


def has_static_corner_deadlock(
    level: SokobanLevel,
    state: SokobanState,
) -> bool:
    """Detect a non-target box trapped in a static wall corner."""

    for row, column in state.boxes - level.targets:
        up = (row - 1, column) in level.walls
        right = (row, column + 1) in level.walls
        down = (row + 1, column) in level.walls
        left = (row, column - 1) in level.walls
        if (up and right) or (right and down) or (down and left) or (
            left and up
        ):
            return True
    return False


def observation_for(
    level: SokobanLevel,
    state: SokobanState,
) -> np.ndarray:
    """Encode a level and dynamic state as the public uint8 observation."""

    board = np.full(level.shape, int(Tile.FLOOR), dtype=np.uint8)
    for row, column in level.walls:
        board[row, column] = int(Tile.WALL)
    for row, column in level.targets:
        board[row, column] = int(Tile.TARGET)
    for row, column in state.boxes:
        board[row, column] = int(
            Tile.BOX_ON_TARGET
            if (row, column) in level.targets
            else Tile.BOX
        )
    board[state.player] = int(
        Tile.PLAYER_ON_TARGET
        if state.player in level.targets
        else Tile.PLAYER
    )
    return board


def decode_observation(
    observation: np.ndarray,
) -> tuple[SokobanLevel, SokobanState]:
    """Decode a valid environment observation for a state-space search."""

    if observation.ndim != 2:
        raise ValueError("Sokoban observation must have two dimensions")

    positions: dict[Tile, set[Position]] = {
        tile: set() for tile in Tile
    }
    for row, column in np.ndindex(observation.shape):
        try:
            tile = Tile(int(observation[row, column]))
        except ValueError as error:
            raise ValueError("Sokoban observation contains an unknown tile") from error
        positions[tile].add((row, column))

    players = positions[Tile.PLAYER] | positions[Tile.PLAYER_ON_TARGET]
    if len(players) != 1:
        raise ValueError("Sokoban observation must contain exactly one player")
    player = next(iter(players))
    boxes = positions[Tile.BOX] | positions[Tile.BOX_ON_TARGET]
    targets = (
        positions[Tile.TARGET]
        | positions[Tile.BOX_ON_TARGET]
        | positions[Tile.PLAYER_ON_TARGET]
    )
    if not boxes or len(boxes) != len(targets):
        raise ValueError(
            "Sokoban observation must contain matching boxes and targets"
        )

    level = SokobanLevel(
        level_id="observation",
        height=int(observation.shape[0]),
        width=int(observation.shape[1]),
        walls=frozenset(positions[Tile.WALL]),
        targets=frozenset(targets),
        boxes=frozenset(boxes),
        player=player,
    )
    return level, SokobanState(player=player, boxes=frozenset(boxes))
