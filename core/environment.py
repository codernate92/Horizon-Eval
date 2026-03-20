"""Constrained environment that agents operate within during evaluation.

The environment defines the sandbox in which an agent under test executes:
which tools are available, what constraints govern behavior, and what state
the agent can observe and mutate. Every action is validated against the
constraint set before execution, enabling detection of constraint bypass
attempts — a key alignment signal.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import AgentAction

logger = logging.getLogger(__name__)


@dataclass
class ToolPermission:
    """Permission descriptor for a single tool in the environment.

    Attributes:
        tool_name: Identifier of the tool.
        allowed: Whether the tool is currently permitted.
        escalation_level: Access tier.
            0 = granted (agent can use freely),
            1 = requestable (agent must ask permission),
            2 = forbidden (use is a constraint violation).
    """

    tool_name: str
    allowed: bool = True
    escalation_level: int = 0

    def __post_init__(self) -> None:
        if self.escalation_level not in (0, 1, 2):
            raise ValueError(
                f"escalation_level must be 0, 1, or 2, got {self.escalation_level}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "tool_name": self.tool_name,
            "allowed": self.allowed,
            "escalation_level": self.escalation_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPermission:
        """Deserialize from a dictionary."""
        return cls(
            tool_name=data["tool_name"],
            allowed=data.get("allowed", True),
            escalation_level=data.get("escalation_level", 0),
        )


@dataclass
class EnvironmentConstraint:
    """A constraint that governs what actions are permissible.

    Constraints are the rules of the evaluation sandbox. When an agent
    violates a constraint, it generates a behavioral flag. Some constraints
    are hard (blocking), others are soft (flagging only).

    Attributes:
        constraint_type: Category (e.g., "tool_restriction", "state_boundary",
            "information_barrier", "action_rate_limit").
        description: Human-readable explanation of what this constraint enforces.
        check_fn: Optional callable that takes an AgentAction and returns True
            if the action *violates* this constraint. None means the constraint
            is documentation-only (checked by external classifiers).
    """

    constraint_type: str
    description: str
    check_fn: Callable[[AgentAction], bool] | None = None

    def is_violated_by(self, action: AgentAction) -> bool:
        """Check whether the given action violates this constraint.

        Returns True if violated, False otherwise. If no check_fn is set,
        the constraint cannot be checked programmatically and returns False.
        """
        if self.check_fn is None:
            return False
        try:
            return self.check_fn(action)
        except Exception:
            logger.exception(
                "Error evaluating constraint %r against action %r",
                self.constraint_type,
                action.action_type,
            )
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Note: check_fn is not serializable and is omitted.
        """
        return {
            "constraint_type": self.constraint_type,
            "description": self.description,
            "has_check_fn": self.check_fn is not None,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        check_fn: Callable[[AgentAction], bool] | None = None,
    ) -> EnvironmentConstraint:
        """Deserialize from a dictionary.

        The check_fn must be supplied separately since callables are not
        JSON-serializable.
        """
        return cls(
            constraint_type=data["constraint_type"],
            description=data["description"],
            check_fn=check_fn,
        )


class Environment:
    """The constrained sandbox in which an agent operates during evaluation.

    Manages tool permissions, behavioral constraints, and mutable state.
    Every agent action passes through validate_action() before execution,
    producing an auditable record of constraint checks and violations.

    Attributes:
        name: Human-readable environment name.
        description: What this environment simulates.
        available_tools: Tool permission descriptors.
        constraints: Behavioral constraints the agent must respect.
        state: Mutable key-value state the agent can observe and modify.
    """

    def __init__(
        self,
        name: str,
        description: str,
        available_tools: list[ToolPermission] | None = None,
        constraints: list[EnvironmentConstraint] | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.available_tools = available_tools or []
        self.constraints = constraints or []
        self.state: dict[str, Any] = state or {}
        self._state_history: list[dict[str, Any]] = []

        # Index tools by name for fast lookup
        self._tool_index: dict[str, ToolPermission] = {
            tp.tool_name: tp for tp in self.available_tools
        }

    def validate_action(self, action: AgentAction) -> tuple[bool, list[str]]:
        """Validate an agent action against all environment constraints.

        Checks tool permissions (for tool_use actions) and runs every
        constraint's check_fn against the action.

        Args:
            action: The action to validate.

        Returns:
            A tuple of (allowed, violation_reasons). allowed is False if
            any constraint is violated. violation_reasons lists all
            detected violations as human-readable strings.
        """
        violations: list[str] = []

        # Check tool permissions for tool_use actions
        if action.action_type == "tool_use":
            tool_name = action.data.get("tool_name", "")
            perm = self._tool_index.get(tool_name)
            if perm is None:
                violations.append(
                    f"Tool '{tool_name}' is not registered in this environment"
                )
            elif not perm.allowed:
                violations.append(
                    f"Tool '{tool_name}' is not allowed "
                    f"(escalation_level={perm.escalation_level})"
                )
            elif perm.escalation_level == 2:
                violations.append(
                    f"Tool '{tool_name}' is forbidden (escalation_level=2)"
                )

        # Check all constraints
        for constraint in self.constraints:
            if constraint.is_violated_by(action):
                violations.append(
                    f"Constraint violated [{constraint.constraint_type}]: "
                    f"{constraint.description}"
                )

        allowed = len(violations) == 0
        if not allowed:
            logger.warning(
                "Action %r rejected with %d violation(s): %s",
                action.action_type,
                len(violations),
                "; ".join(violations),
            )

        return allowed, violations

    def get_available_tools(self, escalation_level: int = 0) -> list[str]:
        """Return tool names available at the given escalation level.

        Args:
            escalation_level: Maximum escalation level to include.
                0 = only granted tools, 1 = granted + requestable,
                2 = all tools including forbidden.

        Returns:
            List of tool name strings.
        """
        return [
            tp.tool_name
            for tp in self.available_tools
            if tp.allowed and tp.escalation_level <= escalation_level
        ]

    def record_state_change(self, key: str, value: Any) -> None:
        """Record a mutation to the environment state.

        Stores the previous state snapshot before applying the change,
        enabling state diff analysis in post-evaluation auditing.

        Args:
            key: State key to set.
            value: New value.
        """
        # Snapshot current state before mutation
        self._state_history.append(
            {
                "timestamp": time.time(),
                "key": key,
                "old_value": self.state.get(key),
                "new_value": value,
            }
        )
        self.state[key] = value
        logger.debug("Environment state change: %s = %r", key, value)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current environment state.

        Includes the full state dict, tool permissions, and constraint
        metadata. Useful for embedding in trajectory events.
        """
        return {
            "name": self.name,
            "state": copy.deepcopy(self.state),
            "available_tools": [tp.to_dict() for tp in self.available_tools],
            "constraint_count": len(self.constraints),
            "state_history_length": len(self._state_history),
            "timestamp": time.time(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the environment to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "available_tools": [tp.to_dict() for tp in self.available_tools],
            "constraints": [c.to_dict() for c in self.constraints],
            "state": copy.deepcopy(self.state),
            "state_history": copy.deepcopy(self._state_history),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        constraint_fns: dict[str, Callable[[AgentAction], bool]] | None = None,
    ) -> Environment:
        """Deserialize from a dictionary.

        Args:
            data: Serialized environment data.
            constraint_fns: Optional mapping from constraint_type to check_fn,
                since callables cannot be serialized.

        Returns:
            Reconstructed Environment instance.
        """
        constraint_fns = constraint_fns or {}

        tools = [ToolPermission.from_dict(td) for td in data.get("available_tools", [])]
        constraints = [
            EnvironmentConstraint.from_dict(
                cd, check_fn=constraint_fns.get(cd["constraint_type"])
            )
            for cd in data.get("constraints", [])
        ]

        env = cls(
            name=data["name"],
            description=data.get("description", ""),
            available_tools=tools,
            constraints=constraints,
            state=data.get("state", {}),
        )
        env._state_history = data.get("state_history", [])
        return env
