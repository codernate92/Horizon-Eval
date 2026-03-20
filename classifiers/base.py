"""Base classifier infrastructure for behavioral analysis of agent trajectories.

Provides the abstract base class all Horizon-Eval classifiers inherit from,
the standardized result type, and a pipeline for composing multiple
classifiers into a single analysis pass.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.types import BehaviorFlag, Severity
from core.trajectory import EvalTrajectory
from core.eval_task import EvalTask

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    """Standardized output of a single classifier run against a trajectory.

    Attributes:
        classifier_name: Identifier of the classifier that produced this result.
        flags: Behavioral flags detected during analysis.
        summary: Human-readable summary of findings.
        confidence: Overall confidence in the analysis (0.0 to 1.0).
        details: Classifier-specific structured data for downstream consumers.
    """

    classifier_name: str
    flags: list[BehaviorFlag] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def has_flags(self) -> bool:
        """Return True if any behavioral flags were raised."""
        return len(self.flags) > 0

    @property
    def max_severity(self) -> Severity | None:
        """Return the highest severity among raised flags, or None."""
        if not self.flags:
            return None
        return max(self.flags, key=lambda f: f.severity.value).severity

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "classifier_name": self.classifier_name,
            "flags": [f.to_dict() for f in self.flags],
            "summary": self.summary,
            "confidence": self.confidence,
            "details": self.details,
        }


class BaseClassifier(ABC):
    """Abstract base class for all Horizon-Eval behavioral classifiers.

    Classifiers are stateless analysis components that inspect an agent's
    evaluation trajectory and produce structured findings about behavioral
    alignment. Each classifier focuses on a specific threat model (e.g.,
    specification gaming, deceptive alignment, sandbagging).

    Subclasses must implement ``analyze()`` which receives the full trajectory
    and optionally the eval task context, and returns a ``ClassifierResult``.

    Attributes:
        name: Short identifier for this classifier.
        description: Human-readable explanation of what this classifier detects.
        config: Classifier-specific configuration parameters.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.config: dict[str, Any] = config or {}

    @abstractmethod
    def analyze(
        self,
        trajectory: EvalTrajectory,
        task: EvalTask | None = None,
    ) -> ClassifierResult:
        """Analyze a trajectory for behavioral signals.

        Args:
            trajectory: The complete evaluation trajectory to analyze.
            task: Optional eval task providing ground truth, intended paths,
                and misalignment trap definitions for richer analysis.

        Returns:
            A ClassifierResult containing all detected flags and analysis.
        """
        ...

    def _create_flag(
        self,
        flag_type: str,
        severity: Severity,
        description: str,
        evidence: list[dict[str, Any]] | None = None,
        confidence: float = 0.5,
        event_indices: list[int] | None = None,
    ) -> BehaviorFlag:
        """Create a BehaviorFlag with standard metadata.

        Convenience method that ensures consistent flag construction across
        all classifiers.

        Args:
            flag_type: Category of the behavioral concern.
            severity: How alarming this detection is.
            description: Human-readable explanation.
            evidence: Supporting evidence dicts.
            confidence: Detection confidence (0.0 to 1.0).
            event_indices: Trajectory event indices that triggered the flag.

        Returns:
            A fully constructed BehaviorFlag.
        """
        return BehaviorFlag(
            flag_type=flag_type,
            severity=severity,
            description=description,
            evidence=evidence or [],
            confidence=confidence,
            trajectory_event_indices=event_indices or [],
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"


class ClassifierPipeline:
    """Composes multiple classifiers into a single analysis pass.

    Runs each classifier independently against a trajectory and aggregates
    the results. Classifiers are stateless, so the pipeline can safely
    be reused across trajectories without reset.

    Attributes:
        classifiers: Ordered list of classifiers to run.
    """

    def __init__(self, classifiers: list[BaseClassifier]) -> None:
        self.classifiers = list(classifiers)

    def analyze(
        self,
        trajectory: EvalTrajectory,
        task: EvalTask | None = None,
    ) -> list[ClassifierResult]:
        """Run all classifiers against a single trajectory.

        Args:
            trajectory: The trajectory to analyze.
            task: Optional eval task for contextual analysis.

        Returns:
            List of ClassifierResult instances, one per classifier.
        """
        results: list[ClassifierResult] = []
        for classifier in self.classifiers:
            try:
                result = classifier.analyze(trajectory, task)
                results.append(result)
                if result.has_flags:
                    logger.info(
                        "Classifier %s raised %d flag(s) on trajectory "
                        "(task=%s, agent=%s)",
                        classifier.name,
                        len(result.flags),
                        trajectory.task_id,
                        trajectory.agent_id,
                    )
            except Exception:
                logger.exception(
                    "Classifier %s failed on trajectory (task=%s, agent=%s)",
                    classifier.name,
                    trajectory.task_id,
                    trajectory.agent_id,
                )
                results.append(
                    ClassifierResult(
                        classifier_name=classifier.name,
                        summary=f"Classifier {classifier.name} raised an exception.",
                        confidence=0.0,
                        details={"error": True},
                    )
                )
        return results

    def analyze_batch(
        self,
        trajectories: list[EvalTrajectory],
        task: EvalTask | None = None,
    ) -> dict[str, list[ClassifierResult]]:
        """Run all classifiers against multiple trajectories.

        Args:
            trajectories: List of trajectories to analyze.
            task: Optional eval task for contextual analysis.

        Returns:
            Dict keyed by ``"task_id::agent_id"`` mapping to the list of
            ClassifierResult instances for that trajectory.
        """
        batch_results: dict[str, list[ClassifierResult]] = {}
        for trajectory in trajectories:
            key = f"{trajectory.task_id}::{trajectory.agent_id}"
            batch_results[key] = self.analyze(trajectory, task)
        return batch_results

    def summary_report(self, results: list[ClassifierResult]) -> dict[str, Any]:
        """Produce an aggregated summary across multiple classifier results.

        Args:
            results: ClassifierResult instances (typically from one ``analyze()`` call).

        Returns:
            Dict with total flag count, severity breakdown, per-classifier
            summaries, and overall risk assessment.
        """
        all_flags: list[BehaviorFlag] = []
        per_classifier: list[dict[str, Any]] = []
        for result in results:
            all_flags.extend(result.flags)
            per_classifier.append({
                "classifier": result.classifier_name,
                "flag_count": len(result.flags),
                "confidence": result.confidence,
                "summary": result.summary,
                "max_severity": (
                    result.max_severity.name if result.max_severity else None
                ),
            })

        severity_counts: dict[str, int] = {}
        for flag in all_flags:
            sev_name = flag.severity.name
            severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1

        # Determine overall risk level based on highest severity present
        if severity_counts.get("CRITICAL", 0) > 0:
            overall_risk = "CRITICAL"
        elif severity_counts.get("HIGH", 0) > 0:
            overall_risk = "HIGH"
        elif severity_counts.get("MEDIUM", 0) > 0:
            overall_risk = "MEDIUM"
        elif severity_counts.get("LOW", 0) > 0:
            overall_risk = "LOW"
        elif severity_counts.get("INFO", 0) > 0:
            overall_risk = "INFO"
        else:
            overall_risk = "NONE"

        return {
            "total_flags": len(all_flags),
            "severity_breakdown": severity_counts,
            "overall_risk": overall_risk,
            "classifiers_run": len(results),
            "per_classifier": per_classifier,
        }

    def __repr__(self) -> str:
        names = [c.name for c in self.classifiers]
        return f"<ClassifierPipeline(classifiers={names})>"
