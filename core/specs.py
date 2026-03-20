"""Portable task specifications for hermetic evaluation runs.

The existing ``EvalTask`` class is useful for in-memory execution, but it is
not a strong interchange format for reproducible benchmark authoring. This
module adds a serializable task-spec layer with deterministic fingerprints,
budget limits, hermetic-execution policy, canary tokens, and evidence
requirements. The spec can be compiled into an ``EvalTask`` at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .environment import Environment, EnvironmentConstraint, ToolPermission
from .eval_task import EvalTask, SolutionPath
from .types import AgentAction


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class ToolSpec:
    """Serializable tool descriptor for portable tasks."""

    tool_name: str
    allowed: bool = True
    escalation_level: int = 0

    def to_permission(self) -> ToolPermission:
        return ToolPermission(
            tool_name=self.tool_name,
            allowed=self.allowed,
            escalation_level=self.escalation_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "allowed": self.allowed,
            "escalation_level": self.escalation_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolSpec":
        return cls(
            tool_name=data["tool_name"],
            allowed=data.get("allowed", True),
            escalation_level=data.get("escalation_level", 0),
        )


ConstraintMode = Literal[
    "tool_forbidden",
    "state_key_forbidden",
    "substring_block",
]


@dataclass(frozen=True)
class ConstraintSpec:
    """Serializable constraint with a small built-in rule vocabulary."""

    constraint_type: str
    description: str
    mode: ConstraintMode
    target: str
    case_sensitive: bool = False

    def build_check_fn(self) -> callable:
        target = self.target if self.case_sensitive else self.target.lower()

        def _tool_forbidden(action: AgentAction) -> bool:
            candidate = action.data.get("tool_name", "")
            value = candidate if self.case_sensitive else str(candidate).lower()
            return action.action_type == "tool_use" and value == target

        def _state_key_forbidden(action: AgentAction) -> bool:
            candidate = action.data.get("key", "")
            value = candidate if self.case_sensitive else str(candidate).lower()
            return action.action_type == "state_change" and value == target

        def _substring_block(action: AgentAction) -> bool:
            blob = json.dumps(action.data, sort_keys=True)
            haystack = blob if self.case_sensitive else blob.lower()
            return target in haystack

        dispatch = {
            "tool_forbidden": _tool_forbidden,
            "state_key_forbidden": _state_key_forbidden,
            "substring_block": _substring_block,
        }
        return dispatch[self.mode]

    def to_constraint(self) -> EnvironmentConstraint:
        return EnvironmentConstraint(
            constraint_type=self.constraint_type,
            description=self.description,
            check_fn=self.build_check_fn(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "description": self.description,
            "mode": self.mode,
            "target": self.target,
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstraintSpec":
        return cls(
            constraint_type=data["constraint_type"],
            description=data["description"],
            mode=data["mode"],
            target=data["target"],
            case_sensitive=data.get("case_sensitive", False),
        )


SuccessMode = Literal["state_key_equals", "output_key_equals"]


@dataclass(frozen=True)
class SuccessSpec:
    """Serializable success criterion for a task."""

    mode: SuccessMode
    key: str
    expected: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "key": self.key,
            "expected": self.expected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuccessSpec":
        return cls(
            mode=data["mode"],
            key=data["key"],
            expected=data.get("expected"),
        )


@dataclass(frozen=True)
class EvidenceSpec:
    """Minimum evidence required to treat a completion as credible."""

    required_state_keys: tuple[str, ...] = ()
    required_action_types: tuple[str, ...] = ()
    min_steps: int = 1
    min_tool_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_state_keys": list(self.required_state_keys),
            "required_action_types": list(self.required_action_types),
            "min_steps": self.min_steps,
            "min_tool_calls": self.min_tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceSpec":
        return cls(
            required_state_keys=tuple(data.get("required_state_keys", [])),
            required_action_types=tuple(data.get("required_action_types", [])),
            min_steps=data.get("min_steps", 1),
            min_tool_calls=data.get("min_tool_calls", 0),
        )


@dataclass(frozen=True)
class TaskBudget:
    """Run-budget constraints for a task."""

    max_steps: int = 20
    timeout_seconds: float = 120.0
    max_tool_calls: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "max_tool_calls": self.max_tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskBudget":
        return cls(
            max_steps=data.get("max_steps", 20),
            timeout_seconds=data.get("timeout_seconds", 120.0),
            max_tool_calls=data.get("max_tool_calls"),
        )


@dataclass(frozen=True)
class HermeticPolicy:
    """Hermetic execution policy to include in run provenance."""

    network_access: bool = False
    writable_roots: tuple[str, ...] = ()
    env_allowlist: tuple[str, ...] = ()
    seed: int = 0
    container_image: str = "python:3.12-slim"

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_access": self.network_access,
            "writable_roots": list(self.writable_roots),
            "env_allowlist": list(self.env_allowlist),
            "seed": self.seed,
            "container_image": self.container_image,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HermeticPolicy":
        return cls(
            network_access=data.get("network_access", False),
            writable_roots=tuple(data.get("writable_roots", [])),
            env_allowlist=tuple(data.get("env_allowlist", [])),
            seed=data.get("seed", 0),
            container_image=data.get("container_image", "python:3.12-slim"),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Serializable benchmark task specification."""

    slug: str
    name: str
    prompt: str
    category: str
    difficulty: str
    environment_name: str
    environment_description: str
    initial_state: dict[str, Any]
    success: SuccessSpec
    tools: tuple[ToolSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    evidence: EvidenceSpec = field(default_factory=EvidenceSpec)
    budget: TaskBudget = field(default_factory=TaskBudget)
    hermetic: HermeticPolicy = field(default_factory=HermeticPolicy)
    canary_tokens: tuple[str, ...] = ()
    protected_state_keys: tuple[str, ...] = ()
    require_human_qa: bool = True
    intended_paths: tuple[SolutionPath, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.slug:
            raise ValueError("slug must be non-empty")
        if self.budget.max_steps <= 0:
            raise ValueError("budget.max_steps must be positive")
        if self.budget.timeout_seconds <= 0:
            raise ValueError("budget.timeout_seconds must be positive")
        if self.evidence.min_steps <= 0:
            raise ValueError("evidence.min_steps must be positive")
        if self.evidence.min_tool_calls < 0:
            raise ValueError("evidence.min_tool_calls cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "prompt": self.prompt,
            "category": self.category,
            "difficulty": self.difficulty,
            "environment_name": self.environment_name,
            "environment_description": self.environment_description,
            "initial_state": self.initial_state,
            "success": self.success.to_dict(),
            "tools": [tool.to_dict() for tool in self.tools],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "evidence": self.evidence.to_dict(),
            "budget": self.budget.to_dict(),
            "hermetic": self.hermetic.to_dict(),
            "canary_tokens": list(self.canary_tokens),
            "protected_state_keys": list(self.protected_state_keys),
            "require_human_qa": self.require_human_qa,
            "intended_paths": [path.to_dict() for path in self.intended_paths],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            slug=data["slug"],
            name=data["name"],
            prompt=data["prompt"],
            category=data["category"],
            difficulty=data["difficulty"],
            environment_name=data["environment_name"],
            environment_description=data.get("environment_description", ""),
            initial_state=data.get("initial_state", {}),
            success=SuccessSpec.from_dict(data["success"]),
            tools=tuple(ToolSpec.from_dict(tool) for tool in data.get("tools", [])),
            constraints=tuple(
                ConstraintSpec.from_dict(constraint)
                for constraint in data.get("constraints", [])
            ),
            evidence=EvidenceSpec.from_dict(data.get("evidence", {})),
            budget=TaskBudget.from_dict(data.get("budget", {})),
            hermetic=HermeticPolicy.from_dict(data.get("hermetic", {})),
            canary_tokens=tuple(data.get("canary_tokens", [])),
            protected_state_keys=tuple(data.get("protected_state_keys", [])),
            require_human_qa=data.get("require_human_qa", True),
            intended_paths=tuple(
                SolutionPath.from_dict(path) for path in data.get("intended_paths", [])
            ),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "TaskSpec":
        return cls.from_dict(json.loads(payload))

    def fingerprint(self) -> str:
        self.validate()
        payload = _stable_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def build_environment(self) -> Environment:
        return Environment(
            name=self.environment_name,
            description=self.environment_description,
            available_tools=[tool.to_permission() for tool in self.tools],
            constraints=[constraint.to_constraint() for constraint in self.constraints],
            state=dict(self.initial_state),
        )

    def build_eval_task(self) -> EvalTask:
        self.validate()
        environment = self.build_environment()

        def _success_criteria(agent_output: Any, _ground_truth: Any) -> bool:
            if self.success.mode == "state_key_equals":
                return environment.state.get(self.success.key) == self.success.expected
            if self.success.mode == "output_key_equals":
                if not isinstance(agent_output, dict):
                    return False
                return agent_output.get(self.success.key) == self.success.expected
            return False

        return EvalTask(
            name=self.name,
            description=self.prompt,
            category=self.category,
            environment=environment,
            ground_truth=self.success.expected,
            intended_paths=list(self.intended_paths),
            misalignment_traps=[],
            success_criteria=_success_criteria,
            metadata={
                **self.metadata,
                "task_spec_slug": self.slug,
                "task_spec_fingerprint": self.fingerprint(),
                "hermetic_policy": self.hermetic.to_dict(),
                "evidence_spec": self.evidence.to_dict(),
            },
            time_limit=self.budget.timeout_seconds,
            difficulty=self.difficulty,
            id=self.slug,
        )
