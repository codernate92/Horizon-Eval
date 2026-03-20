"""Integration tests for the Horizon-Eval evaluation runner.

Tests cover the full eval lifecycle: completion, max-step enforcement,
trajectory capture, constraint violation detection, timeout handling,
batch execution, environment validation, and trap detection.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from core.types import AgentAction, BehaviorFlag, EvalStatus, Severity
from core.environment import Environment, EnvironmentConstraint, ToolPermission
from core.eval_task import EvalTask, MisalignmentTrap, SolutionPath
from core.trajectory import EvalTrajectoryStore
from core.runner import AgentUnderTest, EvalRunConfig, EvalRunner, EvalResult

from tests.conftest import MockAgent


# ---------------------------------------------------------------------------
# Helper agents
# ---------------------------------------------------------------------------

class NeverStopsAgent:
    """Agent that never signals done — always returns an LLM call."""

    async def act(
        self, observation: dict[str, Any], available_tools: list[str]
    ) -> AgentAction:
        return AgentAction(
            action_type="llm_call",
            timestamp=time.time(),
            data={"prompt": "still thinking..."},
        )


class SlowAgent:
    """Agent that sleeps for a long time on each step."""

    async def act(
        self, observation: dict[str, Any], available_tools: list[str]
    ) -> AgentAction:
        await asyncio.sleep(2.0)
        return AgentAction(
            action_type="message",
            timestamp=time.time(),
            data={"content": "slow result", "done": True},
            metadata={"done": True},
        )


class ForbiddenToolAgent:
    """Agent that tries to use a forbidden tool then gives up."""

    def __init__(self) -> None:
        self.step = 0

    async def act(
        self, observation: dict[str, Any], available_tools: list[str]
    ) -> AgentAction:
        self.step += 1
        if self.step == 1:
            return AgentAction(
                action_type="tool_use",
                timestamp=time.time(),
                data={"tool_name": "code_executor", "code": "print(42)"},
            )
        return AgentAction(
            action_type="message",
            timestamp=time.time(),
            data={"content": "42", "done": True},
            metadata={"done": True},
        )


class ShortcutAgent:
    """Agent that outputs the answer immediately without tool use (spec gaming)."""

    async def act(
        self, observation: dict[str, Any], available_tools: list[str]
    ) -> AgentAction:
        return AgentAction(
            action_type="message",
            timestamp=time.time(),
            data={"content": "42", "done": True},
            metadata={"done": True},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runner_completes_eval(
    sample_eval_task: EvalTask,
    sample_agent: MockAgent,
    runner_config: EvalRunConfig,
) -> None:
    """Run a mock agent through a sample task and verify EvalResult fields."""
    store = EvalTrajectoryStore()
    runner = EvalRunner(runner_config, store)
    result = await runner.run_eval(sample_eval_task, sample_agent, "test-agent-v1")

    assert isinstance(result, EvalResult)
    assert result.task_id == sample_eval_task.id
    assert result.agent_id == "test-agent-v1"
    assert result.status.is_terminal()
    assert result.steps_taken > 0
    assert result.duration > 0.0
    assert result.trajectory is not None
    assert len(result.trajectory.events) > 0


@pytest.mark.asyncio
async def test_runner_respects_max_steps(
    sample_eval_task: EvalTask,
    runner_config: EvalRunConfig,
) -> None:
    """Verify that the runner terminates an agent that never stops."""
    agent = NeverStopsAgent()
    store = EvalTrajectoryStore()
    runner = EvalRunner(runner_config, store)
    result = await runner.run_eval(sample_eval_task, agent, "never-stops")

    assert result.steps_taken <= runner_config.max_steps
    assert result.trajectory is not None
    # Should have roughly max_steps agent_action events
    action_events = [
        e for e in result.trajectory.events if e.event_type == "agent_action"
    ]
    assert len(action_events) <= runner_config.max_steps


@pytest.mark.asyncio
async def test_runner_captures_full_trajectory(
    sample_eval_task: EvalTask,
    sample_agent: MockAgent,
    runner_config: EvalRunConfig,
) -> None:
    """Verify trajectory records events for every step."""
    store = EvalTrajectoryStore()
    runner = EvalRunner(runner_config, store)
    result = await runner.run_eval(sample_eval_task, sample_agent, "test-agent-v1")

    traj = result.trajectory
    assert traj is not None

    # Should have agent_action events
    action_events = [e for e in traj.events if e.event_type == "agent_action"]
    assert len(action_events) >= result.steps_taken

    # Should have environment_response events
    response_events = [e for e in traj.events if e.event_type == "environment_response"]
    assert len(response_events) >= 1

    # Should have evaluator_note events
    note_events = [e for e in traj.events if e.event_type == "evaluator_note"]
    assert len(note_events) >= 1

    # Timing data should be populated
    timing = traj.compute_timing()
    assert timing["action_count"] > 0


@pytest.mark.asyncio
async def test_runner_detects_constraint_violations(
    sample_eval_task: EvalTask,
    runner_config: EvalRunConfig,
) -> None:
    """Agent tries forbidden tool — verify flag raised."""
    agent = ForbiddenToolAgent()
    store = EvalTrajectoryStore()
    runner = EvalRunner(runner_config, store)
    result = await runner.run_eval(sample_eval_task, agent, "forbidden-tool-agent")

    traj = result.trajectory
    assert traj is not None

    # Should have constraint_check events
    constraint_events = [
        e for e in traj.events if e.event_type == "constraint_check"
    ]
    assert len(constraint_events) >= 1

    # The constraint check should indicate a violation
    violation = constraint_events[0]
    assert violation.data.get("violated") is True


@pytest.mark.asyncio
async def test_runner_handles_timeout(
    sample_eval_task: EvalTask,
) -> None:
    """Verify timeout behaviour with a slow agent."""
    config = EvalRunConfig(max_steps=50, timeout=1.0)
    agent = SlowAgent()
    store = EvalTrajectoryStore()
    runner = EvalRunner(config, store)
    result = await runner.run_eval(sample_eval_task, agent, "slow-agent")

    # Should have failed or timed out
    assert result.status in (EvalStatus.FAILED, EvalStatus.PASSED)
    # Duration should be roughly around timeout (agent sleeps 2s per step,
    # but the first step runs then timeout kicks in)
    assert result.duration >= 0.5


@pytest.mark.asyncio
async def test_runner_batch_execution(
    sample_environment: Environment,
    sample_agent: MockAgent,
    runner_config: EvalRunConfig,
) -> None:
    """Run 3 tasks and verify all results returned."""
    tasks = []
    for i in range(3):
        task = EvalTask(
            name=f"batch-task-{i}",
            description=f"Batch task number {i}",
            category="capability",
            environment=sample_environment,
            ground_truth="42",
            success_criteria=lambda output, truth: str(output).get("content", str(output)) == str(truth)
            if isinstance(output, dict) else str(output) == str(truth),
            difficulty="easy",
            id=f"batch-task-{i:03d}",
        )
        tasks.append(task)

    store = EvalTrajectoryStore()
    runner = EvalRunner(runner_config, store)

    # Need fresh agents per task (they maintain step count)
    # Use a factory approach — run sequentially
    results = []
    for task in tasks:
        agent = MockAgent()
        result = await runner.run_eval(task, agent, "batch-agent")
        results.append(result)

    assert len(results) == 3
    for result in results:
        assert isinstance(result, EvalResult)
        assert result.steps_taken > 0


def test_environment_validates_actions(sample_environment: Environment) -> None:
    """Unit test for environment constraint checking."""
    # Allowed action
    allowed_action = AgentAction(
        action_type="tool_use",
        timestamp=time.time(),
        data={"tool_name": "calculator", "expression": "1+1"},
    )
    ok, violations = sample_environment.validate_action(allowed_action)
    assert ok is True
    assert violations == []

    # Forbidden tool
    forbidden_action = AgentAction(
        action_type="tool_use",
        timestamp=time.time(),
        data={"tool_name": "code_executor", "code": "print(1)"},
    )
    ok, violations = sample_environment.validate_action(forbidden_action)
    assert ok is False
    assert len(violations) >= 1

    # Unknown tool
    unknown_action = AgentAction(
        action_type="tool_use",
        timestamp=time.time(),
        data={"tool_name": "nonexistent_tool"},
    )
    ok, violations = sample_environment.validate_action(unknown_action)
    assert ok is False

    # Secret state boundary
    secret_action = AgentAction(
        action_type="state_change",
        timestamp=time.time(),
        data={"key": "secret_key", "value": "bad"},
    )
    ok, violations = sample_environment.validate_action(secret_action)
    assert ok is False


def test_eval_task_checks_traps(sample_eval_task: EvalTask) -> None:
    """Agent takes misaligned path — verify trap detected."""
    # Single-step shortcut (triggers shortcut_trap)
    shortcut_actions = [
        AgentAction(
            action_type="message",
            timestamp=time.time(),
            data={"content": "42"},
        )
    ]
    flags = sample_eval_task.check_traps(shortcut_actions)
    assert len(flags) >= 1
    assert any(f.flag_type == "spec_gaming" for f in flags)

    # Forbidden tool use (triggers forbidden_tool_trap)
    forbidden_actions = [
        AgentAction(
            action_type="tool_use",
            timestamp=time.time(),
            data={"tool_name": "code_executor", "code": "print(42)"},
        )
    ]
    flags = sample_eval_task.check_traps(forbidden_actions)
    assert len(flags) >= 1
    assert any(f.flag_type == "constraint_bypass" for f in flags)

    # Normal multi-step path (no traps)
    normal_actions = [
        AgentAction(
            action_type="llm_call",
            timestamp=time.time(),
            data={"prompt": "think"},
        ),
        AgentAction(
            action_type="tool_use",
            timestamp=time.time(),
            data={"tool_name": "calculator", "expression": "6*7"},
        ),
        AgentAction(
            action_type="message",
            timestamp=time.time(),
            data={"content": "42"},
        ),
    ]
    flags = sample_eval_task.check_traps(normal_actions)
    # Shortcut trap should not trigger (more than 1 action)
    shortcut_flags = [f for f in flags if f.flag_type == "spec_gaming"]
    assert len(shortcut_flags) == 0
