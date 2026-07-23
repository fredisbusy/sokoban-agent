import json

from langgraph.checkpoint.memory import InMemorySaver

from sokoban_agent.env import SokobanEnv
from sokoban_agent.graph import AgenticRuntimeContext, build_agentic_graph


def test_agentic_graph_initializes_json_safe_checkpoint_state() -> None:
    env = SokobanEnv(max_steps=15)
    graph = build_agentic_graph(checkpointer=InMemorySaver())
    context = AgenticRuntimeContext(
        env=env,
        prompt_name="sokoban-strategy",
        prompt_commit="abc123",
        model_name="test-model",
    )
    config = {"configurable": {"thread_id": "agentic-contract"}}

    try:
        result = graph.invoke(
            {"level_id": "tiny-push", "seed": 7, "max_steps": 15},
            config,
            context=context,
        )
    finally:
        env.close()

    assert result["level_id"] == "tiny-push"
    assert result["observation"] == [
        [1, 1, 1, 1, 1],
        [1, 0, 2, 0, 1],
        [1, 0, 3, 0, 1],
        [1, 0, 4, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    assert result["prompt"] == {
        "name": "sokoban-strategy",
        "commit": "abc123",
    }
    assert result["model_name"] == "test-model"
    assert result["board_analysis"] is None
    assert result["strategy_hypothesis"] is None
    assert result["active_subgoal"] is None
    assert result["protected_constraints"] == []
    assert result["expected_effect"] is None
    assert result["failure_conditions"] == []
    assert result["plan_revisions"] == []
    assert result["feedback"] == []
    assert result["decision_events"] == [
        {
            "step": 0,
            "stage": "initialize",
            "summary": "tiny-push 레벨을 초기화했습니다",
        }
    ]
    json.dumps(result)


def test_agentic_graph_reducer_accumulates_decision_events_per_thread() -> None:
    env = SokobanEnv(max_steps=15)
    graph = build_agentic_graph(checkpointer=InMemorySaver())
    context = AgenticRuntimeContext(
        env=env,
        prompt_name="sokoban-strategy",
        prompt_commit="abc123",
        model_name="test-model",
    )
    config = {"configurable": {"thread_id": "agentic-events"}}
    graph_input = {"level_id": "tiny-push", "seed": 0, "max_steps": 15}

    try:
        graph.invoke(graph_input, config, context=context)
        result = graph.invoke(graph_input, config, context=context)
    finally:
        env.close()

    assert [event["stage"] for event in result["decision_events"]] == [
        "initialize",
        "initialize",
    ]


def test_agentic_graph_excludes_global_search_oracle_nodes() -> None:
    graph = build_agentic_graph()

    node_names = set(graph.get_graph().nodes)

    assert not any(
        "astar" in name or "oracle" in name or "search_guard" in name
        for name in node_names
    )
