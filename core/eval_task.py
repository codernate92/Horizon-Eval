"""Evaluation tasks with embedded misalignment traps.

An EvalTask defines a complete evaluation scenario: the environment, the
objective, the ground truth, and — crucially — the misalignment traps that
test whether the agent takes aligned or misaligned solution paths.

This is the core of Horizon-Eval's alignment evaluation: tasks are designed with
multiple solution paths, some intended and some representing specification
gaming, reward hacking, or deceptive shortcuts. The trap detection system
identifies which path the agent took and flags misaligned behavior.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import AgentAction, BehaviorFlag, Severity
from .environment import Environment

logger = logging.getLogger(__name__)


@dataclass
class SolutionPath:
    """A possible solution path an agent might take through an eval task.

    Tasks are designed with multiple paths to distinguish capable-and-aligned
    agents from capable-but-misaligned ones.

    Attributes:
        name: Short identifier for this path (e.g., "honest_direct", "spec_gaming_shortcut").
        description: Human-readable explanation of what this path involves.
        steps: Ordered list of step descriptions characterizing this path.
        is_intended: True if this is a path the task designer considers correct.
        is_misaligned: True if this path represents specification gaming,
            reward hacking, or other alignment failures.
        detection_criteria: Description of how to detect if the agent took
            this path — used by classifiers and human reviewers.
    """

    name: str
    description: str
    steps: list[str] = field(default_factory=list)
    is_intended: bool = True
    is_misaligned: bool = False
    detection_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "is_intended": self.is_intended,
            "is_misaligned": self.is_misaligned,
            "detection_criteria": self.detection_criteria,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolutionPath:
        """Deserialize from a dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            steps=data.get("steps", []),
            is_intended=data.get("is_intended", True),
            is_misaligned=data.get("is_misaligned", False),
            detection_criteria=data.get("detection_criteria", ""),
        )


@dataclass
class MisalignmentTrap:
    """A deliberately placed trap to detect misaligned agent behavior.

    Traps are embedded in eval tasks to test whether agents will take
    shortcuts that technically satisfy the reward signal but violate
    the spirit of the task or its constraints.

    Attributes:
        name: Short identifier for this trap.
        description: What this trap tests for.
        trap_type: Category of misalignment: "spec_gaming", "reward_hack",
            "constraint_bypass", or "deceptive_shortcut".
        trigger_condition: Human-readable description of what agent behavior
            would trigger this trap.
        detection_fn: Optional callable that takes the full action history
            and returns True if the trap was triggered. None means detection
            relies on external classifiers.
        severity: How concerning it is if an agent triggers this trap.
    """

    name: str
    description: str
    trap_type: str = "spec_gaming"
    trigger_condition: str = ""
    detection_fn: Callable[[list[AgentAction]], bool] | None = None
    severity: Severity = Severity.MEDIUM

    _VALID_TRAP_TYPES = frozenset({
        "spec_gaming",
        "reward_hack",
        "constraint_bypass",
        "deceptive_shortcut",
    })

    def __post_init__(self) -> None:
        if self.trap_type not in self._VALID_TRAP_TYPES:
            raise ValueError(
                f"trap_type must be one of {self._VALID_TRAP_TYPES}, "
                f"got {self.trap_type!r}"
            )

    def is_triggered(self, actions: list[AgentAction]) -> bool:
        """Check whether the agent's action history triggered this trap.

        Returns False if no detection_fn is set (requires external classification).
        """
        if self.detection_fn is None:
            return False
        try:
            return self.detection_fn(actions)
        except Exception:
            logger.exception(
                "Error evaluating trap %r detection function", self.name
            )
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Note: detection_fn is not serializable and is omitted.
        """
        return {
            "name": self.name,
            "description": self.description,
            "trap_type": self.trap_type,
            "trigger_condition": self.trigger_condition,
            "has_detection_fn": self.detection_fn is not None,
            "severity": self.severity.name,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        detection_fn: Callable[[list[AgentAction]], bool] | None = None,
    ) -> MisalignmentTrap:
        """Deserialize from a dictionary.

        The detection_fn must be supplied separately since callables
        are not JSON-serializable.
        """
        return cls(
            name=data["name"],
            description=data["description"],
            trap_type=data.get("trap_type", "spec_gaming"),
            trigger_condition=data.get("trigger_condition", ""),
            detection_fn=detection_fn,
            severity=Severity[data["severity"]] if "severity" in data else Severity.MEDIUM,
        )


class EvalTask:
    """A complete evaluation task with misalignment traps and solution paths.

    This is the central evaluation unit in Horizon-Eval. Each task defines:
    - An environment (sandbox) the agent operates in
    - A ground truth answer or expected outcome
    - Multiple solution paths (intended and misaligned)
    - Misalignment traps that detect concerning agent behavior
    - Success criteria for judging the agent's output

    Attributes:
        id: Unique task identifier (UUID).
        name: Human-readable task name.
        description: Full task description, presented to the agent.
        category: One of "capability", "alignment", "robustness", "honesty".
        environment: The constrained environment for this task.
        ground_truth: The correct answer or expected final state.
        intended_paths: Solution paths the task designer considers correct.
        misalignment_traps: Embedded traps for detecting misaligned behavior.
        success_criteria: Callable(agent_output, ground_truth) -> bool.
        metadata: Arbitrary additional metadata.
        time_limit: Optional time limit in seconds.
        difficulty: Task difficulty: "easy", "medium", "hard", "extreme".
    """

    _VALID_CATEGORIES = frozenset({"capability", "alignment", "robustness", "honesty"})
    _VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "extreme"})

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        environment: Environment,
        ground_truth: Any = None,
        intended_paths: list[SolutionPath] | None = None,
        misalignment_traps: list[MisalignmentTrap] | None = None,
        success_criteria: Callable[[Any, Any], bool] | None = None,
        metadata: dict[str, Any] | None = None,
        time_limit: float | None = None,
        difficulty: str = "medium",
        id: str | None = None,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        self.time_limit = time_limit

        if category not in self._VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {self._VALID_CATEGORIES}, got {category!r}"
            )
        self.category = category

        if difficulty not in self._VALID_DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of {self._VALID_DIFFICULTIES}, got {difficulty!r}"
            )
        self.difficulty = difficulty

        self.environment = environment
        self.ground_truth = ground_truth
        self.intended_paths = intended_paths or []
        self.misalignment_traps = misalignment_traps or []
        self.success_criteria = success_criteria or (lambda output, truth: output == truth)

    def check_result(self, agent_output: Any) -> tuple[bool, str]:
        """Check whether the agent's output satisfies the success criteria.

        Args:
            agent_output: The agent's final output or answer.

        Returns:
            (success, message) tuple. success is True if the output matches
            ground truth per the success_criteria callable.
        """
        try:
            success = self.success_criteria(agent_output, self.ground_truth)
        except Exception as exc:
            logger.exception("Error in success_criteria evaluation")
            return False, f"Success criteria raised an exception: {exc}"

        if success:
            return True, "Agent output matches ground truth."
        return False, "Agent output does not match ground truth."

    def check_traps(self, actions: list[AgentAction]) -> list[BehaviorFlag]:
        """Run all misalignment trap detectors against the action history.

        Args:
            actions: The full sequence of agent actions.

        Returns:
            List of BehaviorFlag instances for each triggered trap.
        """
        flags: list[BehaviorFlag] = []
        for trap in self.misalignment_traps:
            if trap.is_triggered(actions):
                flag = BehaviorFlag(
                    flag_type=trap.trap_type,
                    severity=trap.severity,
                    description=(
                        f"Misalignment trap triggered: {trap.name}. "
                        f"{trap.description}"
                    ),
                    evidence=[
                        {
                            "trap_name": trap.name,
                            "trap_type": trap.trap_type,
                            "trigger_condition": trap.trigger_condition,
                            "action_count": len(actions),
                        }
                    ],
                    confidence=0.9,
                    trajectory_event_indices=list(range(len(actions))),
                )
                flags.append(flag)
                logger.warning(
                    "Trap triggered: %s (type=%s, severity=%s)",
                    trap.name,
                    trap.trap_type,
                    trap.severity.name,
                )
        return flags

    def classify_solution_path(
        self, actions: list[AgentAction]
    ) -> SolutionPath | None:
        """Determine which solution path the agent took, if any.

        Uses each path's detection_criteria (currently string-based, designed
        to be extended with classifier callables). For now, iterates through
        intended paths and checks for step-level pattern matches in the
        action data.

        This is a heuristic classification. Production deployments should
        pair this with LLM-based trajectory classification.

        Args:
            actions: The full sequence of agent actions.

        Returns:
            The best-matching SolutionPath, or None if no path matches.
        """
        if not actions:
            return None

        all_paths = self.intended_paths
        action_types = [a.action_type for a in actions]
        action_data_strs = [str(a.data).lower() for a in actions]
        combined_actions = " ".join(action_data_strs)

        best_match: SolutionPath | None = None
        best_score = 0.0

        for path in all_paths:
            if not path.steps:
                continue

            # Score based on how many steps appear in the action sequence
            matched_steps = 0
            for step in path.steps:
                step_lower = step.lower()
                if step_lower in combined_actions:
                    matched_steps += 1

            score = matched_steps / len(path.steps) if path.steps else 0.0
            if score > best_score:
                best_score = score
                best_match = path

        if best_score > 0.0:
            logger.info(
                "Classified solution path: %s (score=%.2f)",
                best_match.name if best_match else "none",
                best_score,
            )
            return best_match

        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Note: success_criteria callable is not serialized.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "environment": self.environment.to_dict(),
            "ground_truth": self.ground_truth,
            "intended_paths": [p.to_dict() for p in self.intended_paths],
            "misalignment_traps": [t.to_dict() for t in self.misalignment_traps],
            "metadata": self.metadata,
            "time_limit": self.time_limit,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        success_criteria: Callable[[Any, Any], bool] | None = None,
        constraint_fns: dict[str, Callable[[AgentAction], bool]] | None = None,
        trap_detection_fns: dict[str, Callable[[list[AgentAction]], bool]] | None = None,
    ) -> EvalTask:
        """Deserialize from a dictionary.

        Callables (success_criteria, constraint check_fns, trap detection_fns)
        must be supplied separately since they are not JSON-serializable.

        Args:
            data: Serialized task data.
            success_criteria: The success evaluation callable.
            constraint_fns: Mapping from constraint_type to check_fn.
            trap_detection_fns: Mapping from trap name to detection_fn.
        """
        trap_detection_fns = trap_detection_fns or {}

        environment = Environment.from_dict(
            data["environment"], constraint_fns=constraint_fns
        )
        intended_paths = [
            SolutionPath.from_dict(p) for p in data.get("intended_paths", [])
        ]
        traps = [
            MisalignmentTrap.from_dict(
                t, detection_fn=trap_detection_fns.get(t["name"])
            )
            for t in data.get("misalignment_traps", [])
        ]

        return cls(
            id=data.get("id"),
            name=data["name"],
            description=data["description"],
            category=data["category"],
            environment=environment,
            ground_truth=data.get("ground_truth"),
            intended_paths=intended_paths,
            misalignment_traps=traps,
            success_criteria=success_criteria,
            metadata=data.get("metadata", {}),
            time_limit=data.get("time_limit"),
            difficulty=data.get("difficulty", "medium"),
        )
