import pytest

from sokoban_agent.env import (
    DEFAULT_LEVELS,
    Action,
    FixedLevelProvider,
    SokobanEnv,
    parse_level,
)
from sokoban_agent.planning import (
    AStarPlanner,
    BFSPlanner,
    NoSolutionError,
    Planner,
    PlanningContext,
    PlanningOutcome,
    RandomPlanner,
    SearchGuardPlanner,
    SearchLimitError,
    solve_astar,
    solve_bfs,
)


def _accept_planner(planner: Planner) -> Planner:
    return planner


def _initial_context(level_id: str, seed: int = 0) -> PlanningContext:
    env = SokobanEnv()
    observation, info = env.reset(options={"level_id": level_id})
    env.close()
    return PlanningContext(observation, info, (), (), seed)


def test_random_planner_satisfies_protocol_and_returns_actions() -> None:
    context = _initial_context("tiny-walk", seed=11)
    planner = _accept_planner(RandomPlanner())
    planner.reset(seed=11)

    actions = [planner.plan(context).actions[0] for _ in range(100)]

    assert all(isinstance(action, Action) for action in actions)
    assert set(actions) <= set(Action)


def test_random_planner_replays_sequence_after_seeded_reset() -> None:
    context = _initial_context("tiny-walk", seed=17)
    planner = RandomPlanner()
    planner.reset(seed=17)
    first = [planner.plan(context).actions[0] for _ in range(32)]

    planner.reset(seed=17)
    second = [planner.plan(context).actions[0] for _ in range(32)]

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
    context = _initial_context("tiny-walk")

    assert solve_bfs(context.observation) == solve_bfs(context.observation)


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
    context = _initial_context("tiny-walk")

    with pytest.raises(SearchLimitError, match="expansion limit"):
        solve_bfs(context.observation, max_expanded_states=1)


def test_bfs_planner_records_failed_search_work() -> None:
    outcome = BFSPlanner(max_expanded_states=1).plan(
        _initial_context("tiny-walk")
    )

    assert outcome.error_kind == "search"
    assert outcome.algorithm_calls == 1
    assert outcome.algorithm_elapsed_seconds >= 0


def test_bfs_planner_returns_a_complete_plan() -> None:
    context = _initial_context("tiny-walk")
    planner = _accept_planner(BFSPlanner())
    planner.reset(seed=3)

    outcome = planner.plan(context)

    assert outcome.error is None
    assert len(outcome.actions) == 5
    assert outcome.elapsed_seconds >= 0
    assert outcome.algorithm_calls == 1
    assert outcome.algorithm_expanded_states > 0


@pytest.mark.parametrize(
    ("level_id", "expected_steps"),
    [("tiny-push", 1), ("tiny-walk", 5)],
)
def test_astar_plan_solves_built_in_level(
    level_id: str,
    expected_steps: int,
) -> None:
    env = SokobanEnv(level_provider=DEFAULT_LEVELS)
    observation, _ = env.reset(options={"level_id": level_id})

    plan = solve_astar(observation)
    for action in plan:
        _, _, terminated, truncated, info = env.step(action)

    assert len(plan) == expected_steps
    assert terminated
    assert not truncated
    assert info["success"]


def test_astar_planner_reports_search_metrics() -> None:
    outcome = AStarPlanner().plan(_initial_context("tiny-walk"))

    assert len(outcome.actions) == 5
    assert outcome.algorithm_calls == 1
    assert outcome.algorithm_expanded_states > 0
    assert outcome.algorithm_elapsed_seconds >= 0


def test_astar_solves_a_two_box_level() -> None:
    level = parse_level(
        "two-boxes",
        ["#######", "# . . #", "# $ $ #", "#  @  #", "#######"],
    )
    env = SokobanEnv(level_provider=FixedLevelProvider([level]))
    observation, _ = env.reset()

    plan = solve_astar(observation)
    for action in plan:
        _, _, terminated, _, info = env.step(action)

    assert terminated
    assert info["success"]


class FixedPlanner:
    def __init__(self, action: Action) -> None:
        self.action = action

    @property
    def name(self) -> str:
        return "fixed"

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def plan(self, context: PlanningContext) -> PlanningOutcome:
        del context
        return PlanningOutcome(actions=(self.action,), llm_calls=1)


def test_search_guard_keeps_a_solvable_primary_proposal() -> None:
    planner = SearchGuardPlanner(FixedPlanner(Action.UP))

    outcome = planner.plan(_initial_context("tiny-push"))

    assert outcome.actions == (Action.UP,)
    assert outcome.algorithm_calls == 0
    assert outcome.algorithm_fallbacks == 0
    assert outcome.llm_calls == 1
    assert "보강이 필요 없습니다" in str(outcome.guard_summary)


def test_search_guard_appends_and_reuses_grounded_suffix() -> None:
    planner = SearchGuardPlanner(FixedPlanner(Action.UP))
    context = _initial_context("tiny-walk")

    first = planner.plan(context)
    second = planner.plan(context)

    assert first.actions == (
        Action.UP,
        Action.RIGHT,
        Action.DOWN,
        Action.RIGHT,
        Action.UP,
    )
    assert first.algorithm_calls == 1
    assert first.algorithm_expanded_states > 0
    assert second.actions == first.actions
    assert second.algorithm_calls == 0
    assert second.algorithm_expanded_states == 0
    assert "후속 행동" in str(first.guard_summary)


def test_search_guard_falls_back_to_bfs_for_blocked_proposal() -> None:
    planner = SearchGuardPlanner(FixedPlanner(Action.DOWN))

    outcome = planner.plan(_initial_context("tiny-push"))

    assert outcome.actions == (Action.UP,)
    assert outcome.algorithm_calls == 1
    assert outcome.algorithm_fallbacks == 1
    assert outcome.llm_calls == 1
    assert "막혀 있습니다" in str(outcome.guard_summary)
