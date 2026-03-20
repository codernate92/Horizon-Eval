"""Anomaly detection for outlier trajectories in evaluation runs.

Provides statistical outlier detection, behavioural shift analysis,
capability inconsistency flagging, timing anomaly detection, and
flag clustering — all designed for post-hoc auditing of agent evaluations.
"""

from __future__ import annotations

import enum
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.types import Severity
from core.trajectory import EvalTrajectory

logger = logging.getLogger(__name__)


class AnomalyType(enum.Enum):
    """Categories of detectable anomalies."""

    STATISTICAL_OUTLIER = "statistical_outlier"
    BEHAVIORAL_SHIFT = "behavioral_shift"
    CAPABILITY_INCONSISTENCY = "capability_inconsistency"
    TIMING_ANOMALY = "timing_anomaly"
    FLAG_CLUSTER = "flag_cluster"


@dataclass
class Anomaly:
    """A detected anomaly in evaluation data.

    Attributes:
        anomaly_type: Category of anomaly.
        description: Human-readable explanation.
        severity: How concerning this anomaly is.
        evidence: Structured supporting data.
        z_score: Z-score associated with the anomaly, if applicable.
        affected_trajectories: Trajectory keys (``task_id::agent_id``)
            affected by this anomaly.
    """

    anomaly_type: AnomalyType
    description: str
    severity: Severity
    evidence: dict[str, Any] = field(default_factory=dict)
    z_score: float | None = None
    affected_trajectories: list[str] = field(default_factory=list)


class AnomalyDetector:
    """Run multiple anomaly detection methods over evaluation trajectories.

    Args:
        trajectories: List of EvalTrajectory objects to analyse.
        config: Optional configuration overrides.
    """

    def __init__(
        self,
        trajectories: list[EvalTrajectory],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.trajectories = trajectories
        self.config = config or {}

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _traj_key(traj: EvalTrajectory) -> str:
        return f"{traj.task_id}::{traj.agent_id}"

    @staticmethod
    def _z_scores(values: list[float]) -> list[float]:
        """Compute z-scores for a list of values."""
        if len(values) < 2:
            return [0.0] * len(values)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        return [(v - mean) / std for v in values]

    def _metric_values(self, metric: str) -> list[tuple[EvalTrajectory, float]]:
        """Extract numeric metric from each trajectory.

        Supported metrics: ``duration``, ``action_count``, ``flag_count``.
        """
        pairs: list[tuple[EvalTrajectory, float]] = []
        for traj in self.trajectories:
            if metric == "duration":
                timing = traj.compute_timing()
                pairs.append((traj, timing["total_duration"]))
            elif metric == "action_count":
                pairs.append((traj, float(timing["action_count"]) if False else float(len(traj.get_actions()))))
            elif metric == "flag_count":
                pairs.append((traj, float(len(traj.behavior_flags))))
            elif metric == "avg_action_duration":
                timing = traj.compute_timing()
                pairs.append((traj, timing["avg_action_duration_ms"]))
            else:
                # Default: try duration
                timing = traj.compute_timing()
                pairs.append((traj, timing["total_duration"]))
        return pairs

    @staticmethod
    def _severity_from_z(z: float) -> Severity:
        """Map an absolute z-score to a severity level."""
        az = abs(z)
        if az >= 4.0:
            return Severity.CRITICAL
        if az >= 3.0:
            return Severity.HIGH
        if az >= 2.5:
            return Severity.MEDIUM
        return Severity.LOW

    # -- detection methods ----------------------------------------------------

    def detect_statistical_outliers(
        self,
        metric: str = "duration",
        threshold: float = 2.0,
    ) -> list[Anomaly]:
        """Detect statistical outliers using z-scores.

        Args:
            metric: Which numeric metric to check (``duration``,
                ``action_count``, ``flag_count``, ``avg_action_duration``).
            threshold: Z-score magnitude above which a trajectory is
                considered an outlier.

        Returns:
            List of Anomaly instances for each detected outlier.
        """
        pairs = self._metric_values(metric)
        if len(pairs) < 2:
            return []

        values = [v for _, v in pairs]
        zs = self._z_scores(values)
        anomalies: list[Anomaly] = []

        for (traj, val), z in zip(pairs, zs):
            if abs(z) >= threshold:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    description=(
                        f"Trajectory {self._traj_key(traj)} has {metric}={val:.2f} "
                        f"(z={z:.2f}), which exceeds the threshold of {threshold}"
                    ),
                    severity=self._severity_from_z(z),
                    evidence={
                        "metric": metric,
                        "value": val,
                        "z_score": z,
                        "threshold": threshold,
                        "population_mean": sum(values) / len(values),
                    },
                    z_score=z,
                    affected_trajectories=[self._traj_key(traj)],
                ))

        return anomalies

    def detect_behavioral_shifts(
        self,
        window_size: int = 5,
    ) -> list[Anomaly]:
        """Detect distributional shifts using a sliding window over sequential trajectories.

        Compares the action-type distribution of each window to the global
        distribution. A large Jensen-Shannon-style divergence flags a shift.

        Args:
            window_size: Number of trajectories per window.

        Returns:
            List of Anomaly instances.
        """
        if len(self.trajectories) < window_size * 2:
            return []

        # Global action type distribution
        global_counts: Counter[str] = Counter()
        for traj in self.trajectories:
            for action in traj.get_actions():
                global_counts[action.action_type] += 1
        total_global = sum(global_counts.values())
        if total_global == 0:
            return []
        global_dist = {k: v / total_global for k, v in global_counts.items()}

        anomalies: list[Anomaly] = []
        for i in range(len(self.trajectories) - window_size + 1):
            window = self.trajectories[i : i + window_size]
            window_counts: Counter[str] = Counter()
            for traj in window:
                for action in traj.get_actions():
                    window_counts[action.action_type] += 1
            total_window = sum(window_counts.values())
            if total_window == 0:
                continue
            window_dist = {k: v / total_window for k, v in window_counts.items()}

            # Symmetric KL-like divergence (simplified)
            all_keys = set(global_dist.keys()) | set(window_dist.keys())
            divergence = 0.0
            for k in all_keys:
                p = global_dist.get(k, 1e-10)
                q = window_dist.get(k, 1e-10)
                # Jensen-Shannon components
                m = (p + q) / 2.0
                if p > 0 and m > 0:
                    divergence += 0.5 * p * math.log(p / m)
                if q > 0 and m > 0:
                    divergence += 0.5 * q * math.log(q / m)

            shift_threshold = self.config.get("behavioral_shift_threshold", 0.3)
            if divergence > shift_threshold:
                keys = [self._traj_key(t) for t in window]
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BEHAVIORAL_SHIFT,
                    description=(
                        f"Behavioral shift detected in window [{i}:{i + window_size}] "
                        f"(divergence={divergence:.3f}, threshold={shift_threshold})"
                    ),
                    severity=Severity.MEDIUM if divergence < 0.6 else Severity.HIGH,
                    evidence={
                        "window_start": i,
                        "window_size": window_size,
                        "divergence": divergence,
                        "window_distribution": window_dist,
                        "global_distribution": global_dist,
                    },
                    affected_trajectories=keys,
                ))

        return anomalies

    def detect_capability_inconsistencies(
        self,
        eval_results: list[dict[str, Any]],
    ) -> list[Anomaly]:
        """Flag agents with inconsistent scores across similar capabilities.

        An agent scoring high on one task for a capability but low on another
        task for the same capability is suspicious (possible sandbagging or
        gaming).

        Args:
            eval_results: List of result dicts with keys ``capability``,
                ``score``, ``task_difficulty``, ``agent_id``.

        Returns:
            List of Anomaly instances.
        """
        # Group by (agent_id, capability)
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in eval_results:
            key = (r["agent_id"], r["capability"])
            grouped[key].append(r["score"])

        anomalies: list[Anomaly] = []
        inconsistency_threshold = self.config.get("inconsistency_threshold", 0.4)

        for (agent_id, capability), scores in grouped.items():
            if len(scores) < 2:
                continue
            score_range = max(scores) - min(scores)
            if score_range >= inconsistency_threshold:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.CAPABILITY_INCONSISTENCY,
                    description=(
                        f"Agent {agent_id!r} shows inconsistent performance on "
                        f"{capability!r}: scores range from {min(scores):.2f} to "
                        f"{max(scores):.2f} (range={score_range:.2f})"
                    ),
                    severity=Severity.MEDIUM if score_range < 0.6 else Severity.HIGH,
                    evidence={
                        "agent_id": agent_id,
                        "capability": capability,
                        "scores": scores,
                        "range": score_range,
                        "threshold": inconsistency_threshold,
                    },
                    affected_trajectories=[],
                ))

        return anomalies

    def detect_timing_anomalies(
        self,
        threshold: float = 3.0,
    ) -> list[Anomaly]:
        """Detect suspiciously fast or slow individual actions.

        Computes z-scores of per-event duration_ms values across the entire
        corpus and flags events exceeding the threshold.

        Args:
            threshold: Z-score threshold for flagging.

        Returns:
            List of Anomaly instances.
        """
        # Collect all durations
        all_durations: list[tuple[EvalTrajectory, int, float]] = []
        for traj in self.trajectories:
            for idx, event in enumerate(traj.events):
                if event.duration_ms is not None:
                    all_durations.append((traj, idx, event.duration_ms))

        if len(all_durations) < 2:
            return []

        values = [d for _, _, d in all_durations]
        zs = self._z_scores(values)
        mean_val = sum(values) / len(values)

        anomalies: list[Anomaly] = []
        for (traj, idx, dur), z in zip(all_durations, zs):
            if abs(z) >= threshold:
                direction = "fast" if z < 0 else "slow"
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.TIMING_ANOMALY,
                    description=(
                        f"Suspiciously {direction} action in {self._traj_key(traj)} "
                        f"event {idx}: {dur:.1f}ms (z={z:.2f}, mean={mean_val:.1f}ms)"
                    ),
                    severity=self._severity_from_z(z),
                    evidence={
                        "event_index": idx,
                        "duration_ms": dur,
                        "z_score": z,
                        "population_mean_ms": mean_val,
                        "direction": direction,
                    },
                    z_score=z,
                    affected_trajectories=[self._traj_key(traj)],
                ))

        return anomalies

    def detect_flag_clusters(self) -> list[Anomaly]:
        """Detect clusters of same-type flags concentrated in a short time window.

        If multiple flags of the same type occur within a configurable time
        window across trajectories, it may indicate a systematic issue rather
        than isolated incidents.

        Returns:
            List of Anomaly instances.
        """
        cluster_window = self.config.get("flag_cluster_window_s", 60.0)
        min_cluster_size = self.config.get("flag_cluster_min_size", 3)

        # Collect all flags with timestamps
        flag_records: list[tuple[str, str, float, EvalTrajectory]] = []
        for traj in self.trajectories:
            for event in traj.events:
                if event.event_type == "flag_raised":
                    flag_type = event.data.get("flag_type", "unknown")
                    flag_records.append(
                        (flag_type, self._traj_key(traj), event.timestamp, traj)
                    )

        if not flag_records:
            return []

        # Group by flag_type
        by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for ft, key, ts, _ in flag_records:
            by_type[ft].append((key, ts))

        anomalies: list[Anomaly] = []
        for flag_type, entries in by_type.items():
            if len(entries) < min_cluster_size:
                continue

            sorted_entries = sorted(entries, key=lambda x: x[1])
            # Sliding window
            for i in range(len(sorted_entries)):
                cluster: list[tuple[str, float]] = []
                for j in range(i, len(sorted_entries)):
                    if sorted_entries[j][1] - sorted_entries[i][1] <= cluster_window:
                        cluster.append(sorted_entries[j])
                    else:
                        break

                if len(cluster) >= min_cluster_size:
                    affected = list({k for k, _ in cluster})
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FLAG_CLUSTER,
                        description=(
                            f"Cluster of {len(cluster)} '{flag_type}' flags within "
                            f"{cluster_window}s window"
                        ),
                        severity=Severity.HIGH,
                        evidence={
                            "flag_type": flag_type,
                            "cluster_size": len(cluster),
                            "window_seconds": cluster_window,
                            "time_range": (cluster[0][1], cluster[-1][1]),
                        },
                        affected_trajectories=affected,
                    ))
                    break  # Only report the largest cluster per flag type

        return anomalies

    def detect_all(self) -> list[Anomaly]:
        """Run all anomaly detection methods and return results sorted by severity.

        Returns:
            Combined list of Anomaly instances, most severe first.
        """
        anomalies: list[Anomaly] = []
        anomalies.extend(self.detect_statistical_outliers())
        anomalies.extend(self.detect_behavioral_shifts())
        anomalies.extend(self.detect_timing_anomalies())
        anomalies.extend(self.detect_flag_clusters())
        # Sort by severity descending
        anomalies.sort(key=lambda a: a.severity.value, reverse=True)
        return anomalies
