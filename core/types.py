"""Shared enums and lightweight types for the Aegis evaluation framework.

These types form the foundation of the evaluation data model. They are used
across the trajectory system, eval tasks, environment constraints, and
the runner to provide a consistent vocabulary for capability evaluation
and behavioral auditing of LLM agents.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from typing import Any


class EvalStatus(enum.Enum):
    """Status of an evaluation run through its lifecycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    FLAGGED = "flagged"

    def is_terminal(self) -> bool:
        """Return True if this status represents a finished evaluation."""
        return self in (
            EvalStatus.COMPLETED,
            EvalStatus.PASSED,
            EvalStatus.FAILED,
            EvalStatus.FLAGGED,
        )


class Severity(enum.Enum):
    """Severity level for behavioral flags and misalignment detections.

    Ordered from least to most severe. The numeric value enables
    comparison: Severity.HIGH.value > Severity.LOW.value.
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __ge__(self, other: Severity) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value >= other.value

    def __gt__(self, other: Severity) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value > other.value

    def __le__(self, other: Severity) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: Severity) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value < other.value


@dataclass
class BehaviorFlag:
    """A detected behavioral anomaly or alignment concern during evaluation.

    Behavior flags are the primary output of the alignment auditing system.
    They capture instances where an agent's behavior deviates from intended
    behavior, attempts specification gaming, or exhibits other concerning
    patterns.

    Attributes:
        flag_type: Category of the flag (e.g., "spec_gaming", "reward_hack",
            "constraint_bypass", "deceptive_alignment").
        severity: How concerning this behavior is.
        description: Human-readable explanation of what was detected.
        evidence: List of evidence dicts, each containing relevant data
            (action snapshots, state diffs, etc.).
        confidence: Model confidence in this detection, 0.0 to 1.0.
        trajectory_event_indices: Indices into the trajectory event list
            that triggered this flag, enabling precise audit trails.
    """

    flag_type: str
    severity: Severity
    description: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    trajectory_event_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "flag_type": self.flag_type,
            "severity": self.severity.name,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "trajectory_event_indices": self.trajectory_event_indices,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehaviorFlag:
        """Deserialize from a dictionary."""
        return cls(
            flag_type=data["flag_type"],
            severity=Severity[data["severity"]],
            description=data["description"],
            evidence=data.get("evidence", []),
            confidence=data.get("confidence", 0.5),
            trajectory_event_indices=data.get("trajectory_event_indices", []),
        )


@dataclass
class AgentAction:
    """A single action taken by an agent under evaluation.

    Captures everything needed to reconstruct and audit the agent's
    decision-making process.

    Attributes:
        action_type: One of "llm_call", "tool_use", "message", "state_change".
        timestamp: Unix timestamp of when the action occurred.
        data: The action payload — tool call args, message content, etc.
        metadata: Additional context (model name, token counts, latency, etc.).
    """

    action_type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    _VALID_ACTION_TYPES = frozenset({"llm_call", "tool_use", "message", "state_change"})

    def __post_init__(self) -> None:
        if self.action_type not in self._VALID_ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {self._VALID_ACTION_TYPES}, "
                f"got {self.action_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "action_type": self.action_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentAction:
        """Deserialize from a dictionary."""
        return cls(
            action_type=data["action_type"],
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CapabilityScore:
    """A measured capability score from an evaluation.

    Represents a single capability measurement with statistical context,
    suitable for inclusion in capability profiles and benchmark reports.

    Attributes:
        capability: Name of the capability being measured (e.g.,
            "code_generation", "deception_resistance", "tool_use").
        score: Normalized score between 0.0 and 1.0.
        confidence_interval: (lower, upper) bounds of the 95% CI.
        sample_size: Number of evaluation instances contributing to this score.
        notes: Free-text notes about the measurement context.
    """

    capability: str
    score: float
    confidence_interval: tuple[float, float] = (0.0, 1.0)
    sample_size: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"score must be between 0.0 and 1.0, got {self.score}"
            )
        lo, hi = self.confidence_interval
        if lo > hi:
            raise ValueError(
                f"confidence_interval lower bound ({lo}) must be <= upper bound ({hi})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "capability": self.capability,
            "score": self.score,
            "confidence_interval": list(self.confidence_interval),
            "sample_size": self.sample_size,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityScore:
        """Deserialize from a dictionary."""
        ci = data.get("confidence_interval", [0.0, 1.0])
        return cls(
            capability=data["capability"],
            score=data["score"],
            confidence_interval=(ci[0], ci[1]),
            sample_size=data.get("sample_size", 1),
            notes=data.get("notes", ""),
        )
