import assert from "node:assert/strict";
import test from "node:test";

import { applyUpdate, normalizeEvent } from "../src/events.js";

test("partial LangGraph updates accumulate into one view event", () => {
  const first = applyUpdate({}, {
    analyze: {
      observation: [[1, 1, 1], [1, 4, 1], [1, 2, 1]],
      max_steps: 15,
      info: { steps: 0 },
      strategy_hypothesis: {
        summary: "상자를 목표로 보낸다",
        assignments: [{
          box_id: "B1",
          target_id: "T1",
          reason: "통로를 먼저 연다",
        }],
      },
    },
  });
  const second = applyUpdate(first.state, {
    execute_until_push: {
      observation: [[1, 1, 1], [1, 0, 1], [1, 6, 1]],
      info: { steps: 3, success: true },
      action_history: ["DOWN", "DOWN", "UP"],
      execution_result: { actions_executed: ["UP"], push_count: 1 },
      push_count: 2,
      status: "success",
    },
  });
  const event = normalizeEvent({
    id: "evt-2",
    threadId: "thread-1",
    node: second.node,
    state: second.state,
  });

  assert.equal(event.board, "###\n# #\n#+#");
  assert.equal(event.action, "UP");
  assert.equal(event.step, 3);
  assert.equal(event.phase, "success");
  assert.equal(event.strategy.hypothesis, "상자를 목표로 보낸다");
  assert.equal(event.strategy.assignment, "B1 → T1 · 통로를 먼저 연다");
  assert.equal(event.pushCount, 2);
});

test("baseline board and summaries remain supported", () => {
  const event = normalizeEvent({
    id: "evt",
    threadId: "thread",
    node: "execute_action",
    state: {
      board: "#####\n# @ #\n#####",
      action_count: 1,
      status: "성공",
      success: true,
      planner_goal: "목표로 이동",
      guard_summary: "안전",
    },
  });
  assert.equal(event.board.includes("@"), true);
  assert.equal(event.strategy.hypothesis, "목표로 이동");
  assert.equal(event.strategy.risk, "안전");
});

test("terminal and error states map to distinct phases", () => {
  const makeEvent = (state) => normalizeEvent({
    id: "evt",
    threadId: "thread",
    node: "reflect",
    state: { observation: [[4]], ...state },
  });
  assert.equal(makeEvent({ deadlock: true }).phase, "deadlock");
  assert.equal(makeEvent({ truncated: true }).phase, "truncated");
  assert.equal(makeEvent({ status: "reflection_failed" }).phase, "error");
});
