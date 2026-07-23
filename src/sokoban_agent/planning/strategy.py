"""Structured, externally verifiable Sokoban strategy contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Direction = Literal["UP", "RIGHT", "DOWN", "LEFT"]
FailureKind = Literal[
    "support_unreachable",
    "destination_blocked",
    "protected_constraint_violated",
    "static_deadlock",
    "unexpected_state",
]
ViolationCode = Literal[
    "unknown_box",
    "unknown_target",
    "assignment_conflict",
    "unavailable_push",
    "expected_effect_mismatch",
    "protected_cell_conflict",
    "static_deadlock_push",
]


class StrategyModel(BaseModel):
    """Strict immutable base for checkpoint-safe planning artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Cell(StrategyModel):
    """One board coordinate."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)


class BoxFact(StrategyModel):
    """Stable logical box identity at the current observation."""

    box_id: str = Field(pattern=r"^B[1-9][0-9]*$")
    position: Cell


class TargetFact(StrategyModel):
    """Stable logical target identity."""

    target_id: str = Field(pattern=r"^T[1-9][0-9]*$")
    position: Cell


class PushOption(StrategyModel):
    """One currently reachable and geometrically legal push."""

    box_id: str = Field(pattern=r"^B[1-9][0-9]*$")
    direction: Direction
    support: Cell
    destination: Cell
    creates_static_deadlock: bool


class PullDistance(StrategyModel):
    """Wall-aware reverse-pull distance for one box-target pair."""

    box_id: str = Field(pattern=r"^B[1-9][0-9]*$")
    target_id: str = Field(pattern=r"^T[1-9][0-9]*$")
    distance: int | None = Field(default=None, ge=0)


class BoardAnalysis(StrategyModel):
    """Deterministic board facts exposed to a strategy proposer."""

    boxes: tuple[BoxFact, ...] = Field(min_length=1)
    targets: tuple[TargetFact, ...] = Field(min_length=1)
    dead_squares: tuple[Cell, ...]
    reachable_cells: tuple[Cell, ...]
    push_options: tuple[PushOption, ...]
    reverse_pull_distances: tuple[PullDistance, ...]

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> BoardAnalysis:
        box_ids = [box.box_id for box in self.boxes]
        target_ids = [target.target_id for target in self.targets]
        box_positions = [box.position for box in self.boxes]
        if len(set(box_ids)) != len(box_ids):
            raise ValueError("box ids must be unique")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target ids must be unique")
        if len(set(box_positions)) != len(box_positions):
            raise ValueError("box positions must be unique")
        return self


class BoxTargetAssignment(StrategyModel):
    """One hypothesized box-to-target responsibility."""

    box_id: str
    target_id: str
    reason: str = Field(min_length=1, max_length=160)


class ProtectedConstraint(StrategyModel):
    """Cells that must remain clear while the hypothesis is active."""

    kind: Literal["keep_clear"]
    cells: tuple[Cell, ...] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=160)


class PushSubgoal(StrategyModel):
    """A single intended push selected by the strategy."""

    kind: Literal["push"]
    box_id: str
    target_id: str
    direction: Direction
    destination: Cell


class ExpectedEffect(StrategyModel):
    """Observable box transition expected after grounding the subgoal."""

    box_id: str
    from_position: Cell
    to_position: Cell


class GroundedPushPlan(StrategyModel):
    """Shortest player path followed by exactly one intended push."""

    box_id: str = Field(pattern=r"^B[1-9][0-9]*$")
    support: Cell
    player_actions: tuple[Direction, ...]
    push_action: Direction
    push_count: Literal[1] = 1
    expanded_player_states: int = Field(ge=1)


class FailureCondition(StrategyModel):
    """An explicit condition that should trigger strategy revision."""

    kind: FailureKind
    description: str = Field(min_length=1, max_length=200)


class StrategyHypothesis(StrategyModel):
    """One falsifiable strategy and its currently active subgoal."""

    summary: str = Field(min_length=1, max_length=240)
    assignments: tuple[BoxTargetAssignment, ...] = Field(min_length=1)
    protected_constraints: tuple[ProtectedConstraint, ...]
    subgoal: PushSubgoal
    expected_effect: ExpectedEffect
    failure_conditions: tuple[FailureCondition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_assignments(self) -> StrategyHypothesis:
        box_ids = [assignment.box_id for assignment in self.assignments]
        target_ids = [assignment.target_id for assignment in self.assignments]
        if len(set(box_ids)) != len(box_ids):
            raise ValueError("a box can have only one target assignment")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("a target can have only one box assignment")
        return self


class StrategyViolation(StrategyModel):
    """One structured reason a hypothesis cannot be executed."""

    code: ViolationCode
    message: str


class PlanRevision(StrategyModel):
    """An observable strategy change justified by environment evidence."""

    step: int = Field(ge=0)
    disposition: Literal["maintain", "modify", "withdraw"]
    changed_fields: tuple[str, ...]
    evidence: str = Field(min_length=1, max_length=240)


def validate_strategy(
    analysis: BoardAnalysis,
    strategy: StrategyHypothesis,
) -> list[StrategyViolation]:
    """Return semantic violations against the current board facts."""

    violations: list[StrategyViolation] = []
    boxes = {box.box_id: box for box in analysis.boxes}
    targets = {target.target_id: target for target in analysis.targets}
    assignments = {
        assignment.box_id: assignment.target_id
        for assignment in strategy.assignments
    }

    for assignment in strategy.assignments:
        if assignment.box_id not in boxes:
            violations.append(
                _violation("unknown_box", f"{assignment.box_id}가 없습니다")
            )
        if assignment.target_id not in targets:
            violations.append(
                _violation(
                    "unknown_target",
                    f"{assignment.target_id}가 없습니다",
                )
            )

    subgoal = strategy.subgoal
    if subgoal.box_id not in boxes:
        violations.append(
            _violation("unknown_box", f"{subgoal.box_id}가 없습니다")
        )
    if subgoal.target_id not in targets:
        violations.append(
            _violation("unknown_target", f"{subgoal.target_id}가 없습니다")
        )
    if assignments.get(subgoal.box_id) != subgoal.target_id:
        violations.append(
            _violation(
                "assignment_conflict",
                "현재 하위 목표가 상자-목표 배정과 다릅니다",
            )
        )

    matching_pushes = [
        option
        for option in analysis.push_options
        if option.box_id == subgoal.box_id
        and option.direction == subgoal.direction
        and option.destination == subgoal.destination
    ]
    matching_push = bool(matching_pushes)
    if not matching_push:
        violations.append(
            _violation(
                "unavailable_push",
                "현재 보드에서 선택한 push를 실행할 수 없습니다",
            )
        )
    elif matching_pushes[0].creates_static_deadlock:
        violations.append(
            _violation(
                "static_deadlock_push",
                "선택한 push가 정적 dead square에 상자를 둡니다",
            )
        )

    effect = strategy.expected_effect
    box = boxes.get(subgoal.box_id)
    if (
        effect.box_id != subgoal.box_id
        or effect.to_position != subgoal.destination
        or (box is not None and effect.from_position != box.position)
    ):
        violations.append(
            _violation(
                "expected_effect_mismatch",
                "예상 효과가 현재 하위 목표와 일치하지 않습니다",
            )
        )

    protected_cells = {
        cell
        for constraint in strategy.protected_constraints
        for cell in constraint.cells
    }
    if subgoal.destination in protected_cells:
        violations.append(
            _violation(
                "protected_cell_conflict",
                "하위 목표가 보호할 칸을 상자로 점유합니다",
            )
        )
    return violations


def _violation(code: ViolationCode, message: str) -> StrategyViolation:
    return StrategyViolation(code=code, message=message)
