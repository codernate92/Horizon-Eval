"""Evaluation-focused trajectory capture and storage.

The trajectory system is the primary evaluation surface in Horizon-Eval. Every
action, environment response, constraint check, and behavioral flag is
recorded as a timestamped event in an EvalTrajectory. This provides:

- Complete audit trails for alignment review
- Data for post-hoc behavioral classification
- Timing and token usage analytics
- Serializable records for dataset creation and reproducibility

Trajectories are the raw material that alignment researchers analyze
to understand agent behavior, detect specification gaming, and evaluate
the effectiveness of safety interventions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .types import AgentAction, BehaviorFlag, Severity

logger = logging.getLogger(__name__)


@dataclass
class EvalTrajectoryEvent:
    """A single timestamped event in an evaluation trajectory.

    Events form the atomic units of the trajectory record. Each event
    captures one interaction between the agent, the environment, or
    the evaluation harness.

    Attributes:
        timestamp: Unix timestamp of the event.
        event_type: One of "agent_action", "environment_response",
            "constraint_check", "flag_raised", "state_snapshot",
            "evaluator_note".
        task_id: The evaluation task this event belongs to.
        agent_id: The agent under test that produced this event.
        data: Event-specific payload.
        duration_ms: Optional duration of the operation in milliseconds.
        token_usage: Optional token usage dict (prompt_tokens, completion_tokens, etc.).
    """

    timestamp: float
    event_type: str
    task_id: str
    agent_id: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    token_usage: dict[str, Any] | None = None

    _VALID_EVENT_TYPES = frozenset({
        "agent_action",
        "environment_response",
        "constraint_check",
        "flag_raised",
        "state_snapshot",
        "evaluator_note",
    })

    def __post_init__(self) -> None:
        if self.event_type not in self._VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {self._VALID_EVENT_TYPES}, "
                f"got {self.event_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "data": self.data,
        }
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.token_usage is not None:
            result["token_usage"] = self.token_usage
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalTrajectoryEvent:
        """Deserialize from a dictionary."""
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            data=data.get("data", {}),
            duration_ms=data.get("duration_ms"),
            token_usage=data.get("token_usage"),
        )


class EvalTrajectory:
    """Complete trajectory of an agent's evaluation run on a single task.

    Collects all events, actions, flags, and timing data for one
    (task_id, agent_id) evaluation. This is the primary data object
    that alignment researchers inspect when auditing agent behavior.

    Attributes:
        task_id: The evaluation task identifier.
        agent_id: The agent under test identifier.
        events: Ordered list of all trajectory events.
        start_time: Unix timestamp when the evaluation started.
        end_time: Unix timestamp when the evaluation ended (None if ongoing).
        behavior_flags: All behavioral flags raised during this trajectory.
    """

    def __init__(
        self,
        task_id: str,
        agent_id: str,
        events: list[EvalTrajectoryEvent] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        behavior_flags: list[BehaviorFlag] | None = None,
    ) -> None:
        self.task_id = task_id
        self.agent_id = agent_id
        self.events = events or []
        self.start_time = start_time or time.time()
        self.end_time = end_time
        self.behavior_flags = behavior_flags or []

    def _append_event(self, event: EvalTrajectoryEvent) -> int:
        """Append an event and return its index in the event list."""
        idx = len(self.events)
        self.events.append(event)
        return idx

    def record_action(self, task_id: str, agent_id: str, action: AgentAction) -> int:
        """Record an agent action as a trajectory event.

        Args:
            task_id: Task identifier.
            agent_id: Agent identifier.
            action: The agent action to record.

        Returns:
            The index of the newly created event.
        """
        event = EvalTrajectoryEvent(
            timestamp=action.timestamp,
            event_type="agent_action",
            task_id=task_id,
            agent_id=agent_id,
            data={
                "action": action.to_dict(),
            },
            token_usage=action.metadata.get("token_usage"),
        )
        idx = self._append_event(event)
        logger.debug(
            "Recorded action: type=%s, event_idx=%d", action.action_type, idx
        )
        return idx

    def record_environment_response(
        self, task_id: str, agent_id: str, response_data: dict[str, Any]
    ) -> int:
        """Record an environment response to an agent action.

        Args:
            task_id: Task identifier.
            agent_id: Agent identifier.
            response_data: The environment's response payload.

        Returns:
            The index of the newly created event.
        """
        event = EvalTrajectoryEvent(
            timestamp=time.time(),
            event_type="environment_response",
            task_id=task_id,
            agent_id=agent_id,
            data=response_data,
        )
        return self._append_event(event)

    def record_constraint_violation(
        self,
        task_id: str,
        agent_id: str,
        constraint: str,
        action: AgentAction,
        violation_reason: str,
    ) -> int:
        """Record a constraint check result (typically a violation).

        Args:
            task_id: Task identifier.
            agent_id: Agent identifier.
            constraint: Description or type of the constraint checked.
            action: The action that was checked.
            violation_reason: Why the constraint was violated.

        Returns:
            The index of the newly created event.
        """
        event = EvalTrajectoryEvent(
            timestamp=time.time(),
            event_type="constraint_check",
            task_id=task_id,
            agent_id=agent_id,
            data={
                "constraint": constraint,
                "action": action.to_dict(),
                "violation_reason": violation_reason,
                "violated": True,
            },
        )
        idx = self._append_event(event)
        logger.warning(
            "Constraint violation recorded: %s (event_idx=%d)",
            violation_reason,
            idx,
        )
        return idx

    def record_flag(self, flag: BehaviorFlag) -> int:
        """Record a behavioral flag.

        Adds the flag to both the event stream and the behavior_flags list.

        Args:
            flag: The behavior flag to record.

        Returns:
            The index of the newly created event.
        """
        self.behavior_flags.append(flag)
        event = EvalTrajectoryEvent(
            timestamp=time.time(),
            event_type="flag_raised",
            task_id=self.task_id,
            agent_id=self.agent_id,
            data=flag.to_dict(),
        )
        idx = self._append_event(event)
        logger.info(
            "Behavior flag recorded: type=%s, severity=%s (event_idx=%d)",
            flag.flag_type,
            flag.severity.name,
            idx,
        )
        return idx

    def record_evaluator_note(
        self,
        task_id: str,
        agent_id: str,
        note: str,
        data: dict[str, Any] | None = None,
    ) -> int:
        """Record a free-text note from the evaluation harness or reviewer.

        Args:
            task_id: Task identifier.
            agent_id: Agent identifier.
            note: The note text.
            data: Optional additional structured data.

        Returns:
            The index of the newly created event.
        """
        event = EvalTrajectoryEvent(
            timestamp=time.time(),
            event_type="evaluator_note",
            task_id=task_id,
            agent_id=agent_id,
            data={
                "note": note,
                **(data or {}),
            },
        )
        return self._append_event(event)

    def get_actions(self) -> list[AgentAction]:
        """Extract all agent actions from the trajectory events.

        Returns:
            Ordered list of AgentAction instances.
        """
        actions: list[AgentAction] = []
        for event in self.events:
            if event.event_type == "agent_action" and "action" in event.data:
                actions.append(AgentAction.from_dict(event.data["action"]))
        return actions

    def get_flags_by_severity(self, min_severity: Severity) -> list[BehaviorFlag]:
        """Return all behavior flags at or above the given severity level.

        Args:
            min_severity: Minimum severity threshold (inclusive).

        Returns:
            Filtered list of BehaviorFlag instances, ordered by occurrence.
        """
        return [f for f in self.behavior_flags if f.severity >= min_severity]

    def compute_timing(self) -> dict[str, Any]:
        """Compute timing statistics for the trajectory.

        Returns:
            Dict with total_duration, avg_action_duration, and action_count.
        """
        action_events = [
            e for e in self.events if e.event_type == "agent_action"
        ]
        action_count = len(action_events)

        if self.end_time and self.start_time:
            total_duration = self.end_time - self.start_time
        elif action_events:
            total_duration = action_events[-1].timestamp - self.start_time
        else:
            total_duration = 0.0

        durations = [
            e.duration_ms for e in action_events if e.duration_ms is not None
        ]
        avg_action_duration = (
            sum(durations) / len(durations) if durations else 0.0
        )

        return {
            "total_duration": total_duration,
            "avg_action_duration_ms": avg_action_duration,
            "action_count": action_count,
            "timed_action_count": len(durations),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "events": [e.to_dict() for e in self.events],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "behavior_flags": [f.to_dict() for f in self.behavior_flags],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalTrajectory:
        """Deserialize from a dictionary."""
        return cls(
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            events=[EvalTrajectoryEvent.from_dict(e) for e in data.get("events", [])],
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            behavior_flags=[
                BehaviorFlag.from_dict(f) for f in data.get("behavior_flags", [])
            ],
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> EvalTrajectory:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class EvalTrajectoryStore:
    """In-memory store for evaluation trajectories, keyed by (task_id, agent_id).

    Provides indexed access to trajectories by task, by agent, or by
    the composite (task_id, agent_id) key. Designed for use during
    evaluation runs; for persistent storage, serialize via to_json().
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], EvalTrajectory] = {}

    def get_or_create(self, task_id: str, agent_id: str) -> EvalTrajectory:
        """Get the trajectory for (task_id, agent_id), creating it if needed.

        Args:
            task_id: Task identifier.
            agent_id: Agent identifier.

        Returns:
            The existing or newly created EvalTrajectory.
        """
        key = (task_id, agent_id)
        if key not in self._store:
            self._store[key] = EvalTrajectory(task_id=task_id, agent_id=agent_id)
            logger.debug(
                "Created new trajectory for task=%s, agent=%s", task_id, agent_id
            )
        return self._store[key]

    def get(self, task_id: str, agent_id: str) -> EvalTrajectory | None:
        """Get the trajectory for (task_id, agent_id), or None.

        Args:
            task_id: Task identifier.
            agent_id: Agent identifier.

        Returns:
            The EvalTrajectory if it exists, None otherwise.
        """
        return self._store.get((task_id, agent_id))

    def get_all_for_task(self, task_id: str) -> list[EvalTrajectory]:
        """Get all trajectories for a given task (across all agents).

        Args:
            task_id: Task identifier.

        Returns:
            List of EvalTrajectory instances for this task.
        """
        return [
            traj for (tid, _), traj in self._store.items() if tid == task_id
        ]

    def get_all_for_agent(self, agent_id: str) -> list[EvalTrajectory]:
        """Get all trajectories for a given agent (across all tasks).

        Args:
            agent_id: Agent identifier.

        Returns:
            List of EvalTrajectory instances for this agent.
        """
        return [
            traj for (_, aid), traj in self._store.items() if aid == agent_id
        ]

    def all_trajectories(self) -> dict[tuple[str, str], EvalTrajectory]:
        """Return the full trajectory store as a dict.

        Returns:
            Dict mapping (task_id, agent_id) to EvalTrajectory.
        """
        return dict(self._store)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire store to a JSON-compatible dictionary.

        Keys are serialized as "task_id::agent_id" strings.
        """
        return {
            "trajectories": {
                f"{tid}::{aid}": traj.to_dict()
                for (tid, aid), traj in self._store.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalTrajectoryStore:
        """Deserialize from a dictionary."""
        store = cls()
        for key_str, traj_data in data.get("trajectories", {}).items():
            tid, aid = key_str.split("::", 1)
            store._store[(tid, aid)] = EvalTrajectory.from_dict(traj_data)
        return store

    def to_json(self) -> str:
        """Serialize the entire store to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> EvalTrajectoryStore:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))
