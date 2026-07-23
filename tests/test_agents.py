import numpy as np
import pytest

from sokoban_agent.agents import (
    Agent,
    BFSAgent,
    NoSolutionError,
    RandomAgent,
    SearchLimitError,
    solve_bfs,
)
from sokoban_agent.env import (
    DEFAULT_LEVELS,
    Action,
    FixedLevelProvider,
    SokobanEnv,
    parse_level,
)


def _accept_agent(agent: Agent) -> Agent:
    return agent


def _initial_observation(level_id: str) -> tuple[np.ndarray, dict[str, object]]:
    env = SokobanEnv()
    observation, info = env.reset(options={"level_id": level_id})
    env.close()
    return observation, info


def test_random_agent_satisfies_protocol_and_returns_actions() -> None:
    observation, info = _initial_observation("tiny-walk")
    agent = _accept_agent(RandomAgent())
    agent.reset(observation, info, seed=11)

    actions = [agent.act(observation, info) for _ in range(100)]

    assert all(isinstance(action, Action) for action in actions)
    assert set(actions) <= set(Action)


def test_random_agent_replays_sequence_after_seeded_reset() -> None:
    observation, info = _initial_observation("tiny-walk")
    agent = RandomAgent()
    agent.reset(observation, info, seed=17)
    first = [agent.act(observation, info) for _ in range(32)]

    agent.reset(observation, info, seed=17)
    second = [agent.act(observation, info) for _ in range(32)]

    assert first == second


@pytest.mark.parametrize(
    ("level_id", "expected_steps"),
    [("tiny-push", 1), ("tiny-walk", 5)],
)
def test_bfs_plan_solves_built_in_level(
    level_id: str,
    expected_steps: int,
) -> None:
    env = SokobanEnv(level_provider=DEFAULT_LEVELS)
    observation, _ = env.reset(options={"level_id": level_id})

    plan = solve_bfs(observation)
    for action in plan:
        _, _, terminated, truncated, info = env.step(action)

    assert len(plan) == expected_steps
    assert terminated
    assert not truncated
    assert info["success"]


def test_bfs_plan_is_deterministic() -> None:
    observation, _ = _initial_observation("tiny-walk")

    assert solve_bfs(observation) == solve_bfs(observation)


def test_bfs_rejects_static_corner_deadlock() -> None:
    level = parse_level(
        "stuck",
        ["#####", "## .#", "#$ @#", "#   #", "#####"],
    )
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))
    observation, _ = env.reset()

    with pytest.raises(NoSolutionError, match="corner deadlock"):
        solve_bfs(observation)


def test_bfs_reports_search_limit() -> None:
    observation, _ = _initial_observation("tiny-walk")

    with pytest.raises(SearchLimitError, match="expansion limit"):
        solve_bfs(observation, max_expanded_states=1)


def test_bfs_agent_yields_plan_one_action_at_a_time() -> None:
    env = SokobanEnv()
    observation, info = env.reset(options={"level_id": "tiny-walk"})
    agent = _accept_agent(BFSAgent())
    agent.reset(observation, info)
    terminated = False

    while not terminated:
        action = agent.act(observation, info)
        observation, _, terminated, truncated, info = env.step(action)
        assert not truncated

    assert info["success"]
