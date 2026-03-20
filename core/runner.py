"""Evaluation runner — orchestrates agent evaluation with full trajectory capture.

The runner is the main entry point for executing evaluations. It:
1. Sets up the environment and trajectory recording
2. Runs the agent step-by-step against the task
3. Validates each action against environment constraints
4. Detects misalignment traps in real-time
5. Produces a comprehensive EvalResult with trajectory, flags, and scores

Designed for use by alignment research institutes (METR, Epoch AI, Far.AI)
running structured capability evaluations and behavioral audits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .types import AgentAction, BehaviorFlag, EvalStatus, Severity
from .environment import Environment
from .eval_task import EvalTask, SolutionPath
from .trajectory import EvalTrajectory, EvalTrajectoryStore

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentUnderTest(Protocol):
    """Protocol that any agent must satisfy to be evaluated by Aegis.

    Agents receive an observation (environment state, task description,
    previous results) and a list of available tool names, and must return
    a single AgentAction representing their next step.
    """

    async def act(
        self, observation: dict[str, Any], available_tools: list[str]
    ) -> AgentAction:
        """Produce the next action given the current observation.

        Args:
            observation: Dict containing the current environment state,
                task description, previous action results, and any other
                context the harness provides.
            available_tools: List of tool names the agent is permitted
                to use at the current escalation level.

        Returns:
            An AgentAction representing the agent's chosen action.
        """
        ...


@dataclass
class EvalRunConfig:
    """Configuration for an evaluation run.

    Attributes:
        max_steps: Maximum number of agent steps before termination.
        timeout: Maximum wall-clock seconds before termination.
        capture_state_every_n_steps: How often to snapshot environment state.
        enable_trap_detection: Whether to run trap detectors after each step.
        enable_realtime_classification: Whether to classify solution path
            in real-time (more expensive, but enables early termination).
        parallel_runs: Number of parallel task evaluations in batch mode.
    """

    max_steps: int = 50
    timeout: float = 300.0
    capture_state_every_n_steps: int = 5
    enable_trap_detection: bool = True
    enable_realtime_classification: bool = False
    parallel_runs: int = 1


@dataclass
class EvalResult:
    """Complete result of an evaluation run.

    Contains everything needed to assess the agent's performance and
    behavioral alignment: the trajectory, flags, solution path classification,
    and success determination.

    Attributes:
        task_id: The evaluation task identifier.
        agent_id: The agent under test identifier.
        status: Final evaluation status.
        agent_output: The agent's final output (last action data, if any).
        trajectory: Full trajectory with all events.
        behavior_flags: All behavioral flags raised.
        solution_path: Classified solution path, if determinable.
        success: Whether the agent's output satisfied the success criteria.
        duration: Total wall-clock time in seconds.
        steps_taken: Number of agent steps executed.
        metadata: Additional result metadata (config, error info, etc.).
    """

    task_id: str
    agent_id: str
    status: EvalStatus
    agent_output: Any = None
    trajectory: EvalTrajectory | None = None
    behavior_flags: list[BehaviorFlag] = field(default_factory=list)
    solution_path: SolutionPath | None = None
    success: bool = False
    duration: float = 0.0
    steps_taken: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "agent_output": self.agent_output,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "behavior_flags": [f.to_dict() for f in self.behavior_flags],
            "solution_path": self.solution_path.to_dict() if self.solution_path else None,
            "success": self.success,
            "duration": self.duration,
            "steps_taken": self.steps_taken,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        """Deserialize from a dictionary."""
        trajectory = (
            EvalTrajectory.from_dict(data["trajectory"])
            if data.get("trajectory")
            else None
        )
        solution_path = (
            SolutionPath.from_dict(data["solution_path"])
            if data.get("solution_path")
            else None
        )
        return cls(
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            status=EvalStatus(data["status"]),
            agent_output=data.get("agent_output"),
            trajectory=trajectory,
            behavior_flags=[
                BehaviorFlag.from_dict(f) for f in data.get("behavior_flags", [])
            ],
            solution_path=solution_path,
            success=data.get("success", False),
            duration=data.get("duration", 0.0),
            steps_taken=data.get("steps_taken", 0),
            metadata=data.get("metadata", {}),
        )


class EvalRunner:
    """Orchestrates evaluation of agents against tasks with full trajectory capture.

    The runner manages the step-by-step execution loop:
    1. Build observation from environment state and task description
    2. Call agent.act() to get the next action
    3. Validate the action against environment constraints
    4. Record everything in the trajectory
    5. Check misalignment traps
    6. Repeat until completion, max steps, or timeout

    Usage:
        store = EvalTrajectoryStore()
        config = EvalRunConfig(max_steps=100, timeout=600)
        runner = EvalRunner(config, store)
        result = await runner.run_eval(task, agent, "agent-v1")
    """

    def __init__(
        self, config: EvalRunConfig, trajectory_store: EvalTrajectoryStore
    ) -> None:
        self.config = config
        self.trajectory_store = trajectory_store

    async def run_eval(
        self, task: EvalTask, agent: AgentUnderTest, agent_id: str
    ) -> EvalResult:
        """Run a single evaluation: agent against task.

        Sets up the environment, runs the agent step by step, validates
        actions, records the full trajectory, checks traps, and produces
        a comprehensive EvalResult.

        Args:
            task: The evaluation task to run.
            agent: The agent under test (must satisfy AgentUnderTest protocol).
            agent_id: Identifier for this agent.

        Returns:
            EvalResult with full trajectory, flags, and success determination.
        """
        trajectory = self.trajectory_store.get_or_create(task.id, agent_id)
        start_time = time.time()
        trajectory.start_time = start_time

        trajectory.record_evaluator_note(
            task.id,
            agent_id,
            f"Evaluation started: task={task.name}, difficulty={task.difficulty}",
            data={
                "config": {
                    "max_steps": self.config.max_steps,
                    "timeout": self.config.timeout,
                    "trap_detection": self.config.enable_trap_detection,
                },
                "task_category": task.category,
            },
        )

        status = EvalStatus.RUNNING
        agent_output: Any = None
        steps_taken = 0
        all_flags: list[BehaviorFlag] = []

        try:
            for step_num in range(1, self.config.max_steps + 1):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.config.timeout:
                    trajectory.record_evaluator_note(
                        task.id,
                        agent_id,
                        f"Timeout reached after {elapsed:.1f}s at step {step_num}",
                    )
                    status = EvalStatus.FAILED
                    logger.warning(
                        "Eval timeout: task=%s, agent=%s, elapsed=%.1fs",
                        task.id,
                        agent_id,
                        elapsed,
                    )
                    break

                action, should_continue = await self._step(
                    task, agent, agent_id, trajectory, step_num
                )
                steps_taken = step_num

                if action is not None:
                    agent_output = action.data

                # Run trap detection after each step if enabled
                if self.config.enable_trap_detection:
                    actions = trajectory.get_actions()
                    step_flags = task.check_traps(actions)
                    for flag in step_flags:
                        # Avoid duplicate flags
                        if not any(
                            f.flag_type == flag.flag_type
                            and f.description == flag.description
                            for f in all_flags
                        ):
                            flag.trajectory_event_indices = [
                                len(trajectory.events) - 1
                            ]
                            trajectory.record_flag(flag)
                            all_flags.append(flag)

                if not should_continue:
                    break

            # Finalize
            trajectory.end_time = time.time()
            duration = trajectory.end_time - start_time

            # Check result
            if status != EvalStatus.FAILED:
                success, message = task.check_result(agent_output)
                trajectory.record_evaluator_note(
                    task.id,
                    agent_id,
                    f"Result check: success={success}, {message}",
                )

                if all_flags:
                    critical_flags = [
                        f for f in all_flags if f.severity >= Severity.HIGH
                    ]
                    if critical_flags:
                        status = EvalStatus.FLAGGED
                    elif success:
                        status = EvalStatus.PASSED
                    else:
                        status = EvalStatus.FAILED
                elif success:
                    status = EvalStatus.PASSED
                else:
                    status = EvalStatus.FAILED
            else:
                success = False

            # Classify solution path
            actions = trajectory.get_actions()
            solution_path = task.classify_solution_path(actions)

            result = EvalResult(
                task_id=task.id,
                agent_id=agent_id,
                status=status,
                agent_output=agent_output,
                trajectory=trajectory,
                behavior_flags=all_flags,
                solution_path=solution_path,
                success=success,
                duration=duration,
                steps_taken=steps_taken,
                metadata={
                    "config": {
                        "max_steps": self.config.max_steps,
                        "timeout": self.config.timeout,
                    },
                    "timing": trajectory.compute_timing(),
                },
            )

            trajectory.record_evaluator_note(
                task.id,
                agent_id,
                f"Evaluation completed: status={status.value}, "
                f"success={success}, steps={steps_taken}, "
                f"flags={len(all_flags)}, duration={duration:.2f}s",
            )

            logger.info(
                "Eval complete: task=%s, agent=%s, status=%s, steps=%d, "
                "flags=%d, duration=%.2fs",
                task.id,
                agent_id,
                status.value,
                steps_taken,
                len(all_flags),
                duration,
            )

            return result

        except Exception as exc:
            trajectory.end_time = time.time()
            duration = trajectory.end_time - start_time
            trajectory.record_evaluator_note(
                task.id,
                agent_id,
                f"Evaluation failed with exception: {exc}",
                data={"exception_type": type(exc).__name__, "exception_str": str(exc)},
            )
            logger.exception(
                "Eval exception: task=%s, agent=%s", task.id, agent_id
            )
            return EvalResult(
                task_id=task.id,
                agent_id=agent_id,
                status=EvalStatus.FAILED,
                agent_output=agent_output,
                trajectory=trajectory,
                behavior_flags=all_flags,
                success=False,
                duration=duration,
                steps_taken=steps_taken,
                metadata={"error": str(exc), "error_type": type(exc).__name__},
            )

    async def run_batch(
        self,
        tasks: list[EvalTask],
        agent: AgentUnderTest,
        agent_id: str,
    ) -> list[EvalResult]:
        """Run multiple evaluation tasks, optionally in parallel.

        Args:
            tasks: List of evaluation tasks to run.
            agent: The agent under test.
            agent_id: Identifier for this agent.

        Returns:
            List of EvalResult instances, one per task.
        """
        if self.config.parallel_runs <= 1:
            # Sequential execution
            results: list[EvalResult] = []
            for task in tasks:
                result = await self.run_eval(task, agent, agent_id)
                results.append(result)
            return results

        # Parallel execution with concurrency limit
        semaphore = asyncio.Semaphore(self.config.parallel_runs)

        async def _run_with_semaphore(task: EvalTask) -> EvalResult:
            async with semaphore:
                return await self.run_eval(task, agent, agent_id)

        results = await asyncio.gather(
            *[_run_with_semaphore(task) for task in tasks],
            return_exceptions=False,
        )
        return list(results)

    async def _step(
        self,
        task: EvalTask,
        agent: AgentUnderTest,
        agent_id: str,
        trajectory: EvalTrajectory,
        step_num: int,
    ) -> tuple[AgentAction | None, bool]:
        """Execute a single evaluation step.

        Builds the observation, calls the agent, validates the action,
        records everything in the trajectory, and optionally snapshots
        environment state.

        Args:
            task: The evaluation task.
            agent: The agent under test.
            agent_id: Agent identifier.
            trajectory: The trajectory to record into.
            step_num: Current step number (1-indexed).

        Returns:
            Tuple of (action_taken, should_continue). action_taken is None
            if the agent produced an invalid action. should_continue is
            False if the evaluation should terminate.
        """
        # Build observation
        observation: dict[str, Any] = {
            "task_description": task.description,
            "task_name": task.name,
            "step_number": step_num,
            "max_steps": self.config.max_steps,
            "environment_state": task.environment.state,
            "environment_name": task.environment.name,
        }

        # Include last action result if available
        actions = trajectory.get_actions()
        if actions:
            observation["last_action"] = actions[-1].to_dict()

        # Include last environment response if available
        env_responses = [
            e for e in trajectory.events
            if e.event_type == "environment_response"
        ]
        if env_responses:
            observation["last_response"] = env_responses[-1].data

        # Get available tools
        available_tools = task.environment.get_available_tools(escalation_level=0)

        # Call agent
        step_start = time.time()
        try:
            action = await agent.act(observation, available_tools)
        except Exception as exc:
            logger.error(
                "Agent raised exception at step %d: %s", step_num, exc
            )
            trajectory.record_evaluator_note(
                task.id,
                agent_id,
                f"Agent exception at step {step_num}: {exc}",
                data={"exception_type": type(exc).__name__},
            )
            return None, False

        step_duration_ms = (time.time() - step_start) * 1000

        # Record the action in trajectory
        event_idx = trajectory.record_action(task.id, agent_id, action)

        # Update duration on the event
        if trajectory.events and event_idx < len(trajectory.events):
            trajectory.events[event_idx].duration_ms = step_duration_ms

        # Validate action against environment constraints
        allowed, violations = task.environment.validate_action(action)

        if not allowed:
            for reason in violations:
                trajectory.record_constraint_violation(
                    task.id, agent_id, "environment", action, reason
                )
            # Record environment response indicating rejection
            trajectory.record_environment_response(
                task.id,
                agent_id,
                {
                    "status": "rejected",
                    "violations": violations,
                    "step": step_num,
                },
            )
        else:
            # Action is valid — record environment response
            if action.action_type == "state_change":
                key = action.data.get("key", "")
                value = action.data.get("value")
                if key:
                    task.environment.record_state_change(key, value)

            trajectory.record_environment_response(
                task.id,
                agent_id,
                {
                    "status": "accepted",
                    "step": step_num,
                    "action_type": action.action_type,
                },
            )

        # Periodic state snapshots
        if step_num % self.config.capture_state_every_n_steps == 0:
            from .trajectory import EvalTrajectoryEvent

            snapshot = task.environment.snapshot()
            trajectory._append_event(
                EvalTrajectoryEvent(
                    timestamp=time.time(),
                    event_type="state_snapshot",
                    task_id=task.id,
                    agent_id=agent_id,
                    data={"snapshot": snapshot, "step": step_num},
                )
            )

        # Check if agent signals completion
        if action.metadata.get("done", False) or action.data.get("done", False):
            trajectory.record_evaluator_note(
                task.id,
                agent_id,
                f"Agent signaled completion at step {step_num}",
            )
            return action, False

        return action, True
