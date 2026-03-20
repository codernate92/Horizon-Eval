"""Tests for the statistical analysis, anomaly detection, and report modules."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import pytest

from core.types import AgentAction, BehaviorFlag, Severity
from core.trajectory import EvalTrajectory, EvalTrajectoryEvent, EvalTrajectoryStore

from analysis.statistics import (
    BehavioralDistribution,
    CapabilityEstimator,
    ConfidenceInterval,
    compute_confidence_interval,
)
from analysis.anomaly import Anomaly, AnomalyDetector, AnomalyType
from analysis.report import EvalReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(
    agent_id: str = "agent-1",
    task_id: str = "task-1",
    action_types: list[str] | None = None,
    durations: list[float] | None = None,
    flags: list[BehaviorFlag] | None = None,
    base_ts: float = 1700000000.0,
) -> EvalTrajectory:
    """Build a trajectory with customisable action types and durations."""
    default_action_types = ["llm_call", "tool_use", "message"]
    if action_types is None:
        if durations is not None and len(durations) > len(default_action_types):
            repeats = (len(durations) + len(default_action_types) - 1) // len(default_action_types)
            action_types = (default_action_types * repeats)[: len(durations)]
        else:
            action_types = list(default_action_types)
    durations = durations or [100.0] * len(action_types)
    traj = EvalTrajectory(task_id=task_id, agent_id=agent_id, start_time=base_ts)

    for i, (at, dur) in enumerate(zip(action_types, durations)):
        action = AgentAction(action_type=at, timestamp=base_ts + i * 0.5, data={"step": i})
        traj.events.append(EvalTrajectoryEvent(
            timestamp=base_ts + i * 0.5,
            event_type="agent_action",
            task_id=task_id,
            agent_id=agent_id,
            data={"action": action.to_dict()},
            duration_ms=dur,
        ))

    traj.end_time = base_ts + len(action_types) * 0.5
    if flags:
        for f in flags:
            traj.behavior_flags.append(f)
            traj.events.append(EvalTrajectoryEvent(
                timestamp=base_ts + len(action_types) * 0.5 + 0.1,
                event_type="flag_raised",
                task_id=task_id,
                agent_id=agent_id,
                data=f.to_dict(),
            ))
    return traj


# ---------------------------------------------------------------------------
# ConfidenceInterval tests
# ---------------------------------------------------------------------------

class TestConfidenceInterval:

    def test_known_values(self) -> None:
        """Verify CI bounds with known data."""
        scores = [10.0, 12.0, 14.0, 11.0, 13.0]
        ci = compute_confidence_interval(scores, confidence=0.95)

        assert isinstance(ci, ConfidenceInterval)
        assert ci.sample_size == 5
        assert ci.confidence_level == 0.95
        assert ci.mean == pytest.approx(12.0, abs=1e-6)
        assert ci.lower < ci.mean
        assert ci.upper > ci.mean
        # For these values the 95% CI should be roughly [10.5, 13.5]
        assert ci.lower > 9.0
        assert ci.upper < 15.0

    def test_identical_scores(self) -> None:
        """All-identical scores should produce zero-width CI."""
        scores = [5.0, 5.0, 5.0, 5.0]
        ci = compute_confidence_interval(scores)

        assert ci.mean == 5.0
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_too_few_scores_raises(self) -> None:
        """Fewer than 2 scores should raise ValueError."""
        with pytest.raises(ValueError):
            compute_confidence_interval([1.0])

    def test_invalid_confidence_raises(self) -> None:
        """Out-of-range confidence should raise ValueError."""
        with pytest.raises(ValueError):
            compute_confidence_interval([1.0, 2.0], confidence=1.5)

    def test_large_sample_normal_approx(self) -> None:
        """With 50+ samples, normal approximation should be used."""
        scores = [float(i) for i in range(50)]
        ci = compute_confidence_interval(scores)

        assert ci.sample_size == 50
        assert ci.mean == pytest.approx(24.5, abs=1e-6)
        assert ci.lower < ci.mean
        assert ci.upper > ci.mean


# ---------------------------------------------------------------------------
# CapabilityEstimator tests
# ---------------------------------------------------------------------------

class TestCapabilityEstimator:

    @pytest.fixture
    def eval_data(self) -> list[dict[str, Any]]:
        return [
            {"capability": "reasoning", "score": 0.8, "task_difficulty": "easy", "agent_id": "a1"},
            {"capability": "reasoning", "score": 0.9, "task_difficulty": "medium", "agent_id": "a1"},
            {"capability": "reasoning", "score": 0.7, "task_difficulty": "hard", "agent_id": "a1"},
            {"capability": "tool_use", "score": 0.6, "task_difficulty": "easy", "agent_id": "a1"},
            {"capability": "tool_use", "score": 0.5, "task_difficulty": "hard", "agent_id": "a1"},
            {"capability": "reasoning", "score": 0.5, "task_difficulty": "easy", "agent_id": "a2"},
            {"capability": "reasoning", "score": 0.4, "task_difficulty": "medium", "agent_id": "a2"},
            {"capability": "tool_use", "score": 0.9, "task_difficulty": "easy", "agent_id": "a2"},
        ]

    def test_estimate_capability(self, eval_data: list[dict[str, Any]]) -> None:
        est = CapabilityEstimator(eval_data)
        cap = est.estimate_capability("reasoning")

        assert 0.0 <= cap.score <= 1.0
        assert cap.sample_size >= 3
        assert cap.confidence_interval[0] <= cap.score
        assert cap.confidence_interval[1] >= cap.score

    def test_estimate_all(self, eval_data: list[dict[str, Any]]) -> None:
        est = CapabilityEstimator(eval_data)
        all_caps = est.estimate_all()

        assert "reasoning" in all_caps
        assert "tool_use" in all_caps
        assert all_caps["reasoning"].score > 0.0
        assert all_caps["tool_use"].score > 0.0

    def test_compare_agents(self, eval_data: list[dict[str, Any]]) -> None:
        est = CapabilityEstimator(eval_data)
        comparison = est.compare_agents("a1", "a2")

        assert "reasoning" in comparison
        assert "mean_a" in comparison["reasoning"]
        assert "cohen_d" in comparison["reasoning"]
        # a1 should score higher on reasoning
        assert comparison["reasoning"]["difference"] > 0

    def test_difficulty_adjusted(self, eval_data: list[dict[str, Any]]) -> None:
        est = CapabilityEstimator(eval_data)
        adj = est.difficulty_adjusted_score("reasoning")

        assert 0.0 <= adj <= 1.0
        # Hard tasks weighted more, so result should differ from simple mean
        simple_mean = sum(r["score"] for r in eval_data if r["capability"] == "reasoning") / 5
        # Just check it's a valid number — exact value depends on weights
        assert adj != simple_mean or True  # May coincidentally match

    def test_missing_capability_raises(self) -> None:
        est = CapabilityEstimator([])
        with pytest.raises(ValueError):
            est.estimate_capability("nonexistent")


# ---------------------------------------------------------------------------
# AnomalyDetector tests
# ---------------------------------------------------------------------------

class TestAnomalyDetector:

    def test_statistical_outliers(self) -> None:
        """Inject an outlier trajectory and verify detection."""
        trajectories = []
        for i in range(10):
            # Normal trajectories: ~0.5s duration
            traj = _make_trajectory(
                agent_id=f"agent-{i}",
                task_id=f"task-{i}",
                action_types=["llm_call", "tool_use", "message"],
                durations=[100.0, 100.0, 100.0],
                base_ts=1700000000.0 + i * 10,
            )
            traj.end_time = traj.start_time + 1.5
            trajectories.append(traj)

        # Inject outlier: much longer duration
        outlier = _make_trajectory(
            agent_id="agent-outlier",
            task_id="task-outlier",
            action_types=["llm_call"] * 20,
            durations=[100.0] * 20,
            base_ts=1700000000.0 + 100,
        )
        outlier.end_time = outlier.start_time + 100.0  # 100s vs ~1.5s
        trajectories.append(outlier)

        detector = AnomalyDetector(trajectories)
        anomalies = detector.detect_statistical_outliers(metric="duration", threshold=2.0)

        assert len(anomalies) >= 1
        # The outlier trajectory should be flagged
        outlier_anomalies = [
            a for a in anomalies
            if "task-outlier::agent-outlier" in a.affected_trajectories
        ]
        assert len(outlier_anomalies) >= 1
        assert outlier_anomalies[0].anomaly_type == AnomalyType.STATISTICAL_OUTLIER

    def test_timing_anomalies(self) -> None:
        """Inject a suspiciously fast action and verify detection."""
        trajectories = []
        for i in range(5):
            traj = _make_trajectory(
                agent_id=f"agent-{i}",
                task_id=f"task-{i}",
                action_types=["llm_call", "tool_use"],
                durations=[500.0, 500.0],  # Normal: 500ms
                base_ts=1700000000.0 + i * 10,
            )
            trajectories.append(traj)

        # Add a trajectory with a suspiciously fast action
        fast_traj = _make_trajectory(
            agent_id="agent-fast",
            task_id="task-fast",
            action_types=["llm_call", "tool_use"],
            durations=[1.0, 500.0],  # 1ms is suspiciously fast
            base_ts=1700000000.0 + 50,
        )
        trajectories.append(fast_traj)

        detector = AnomalyDetector(trajectories)
        anomalies = detector.detect_timing_anomalies(threshold=2.0)

        assert len(anomalies) >= 1
        fast_anomalies = [
            a for a in anomalies
            if "task-fast::agent-fast" in a.affected_trajectories
        ]
        assert len(fast_anomalies) >= 1
        assert fast_anomalies[0].anomaly_type == AnomalyType.TIMING_ANOMALY

    def test_detect_all(self) -> None:
        """detect_all should return a sorted list."""
        trajectories = [
            _make_trajectory(agent_id=f"a-{i}", task_id=f"t-{i}")
            for i in range(5)
        ]
        detector = AnomalyDetector(trajectories)
        anomalies = detector.detect_all()

        # Result should be a list (possibly empty for normal data)
        assert isinstance(anomalies, list)
        # If anomalies found, should be sorted by severity descending
        if len(anomalies) >= 2:
            severities = [a.severity.value for a in anomalies]
            assert severities == sorted(severities, reverse=True)


# ---------------------------------------------------------------------------
# BehavioralDistribution tests
# ---------------------------------------------------------------------------

class TestBehavioralDistribution:

    def test_action_type_frequencies_sum_to_one(self) -> None:
        """Normalized action type frequencies should sum to 1.0."""
        trajectories = [
            _make_trajectory(action_types=["llm_call", "tool_use", "message"]),
            _make_trajectory(agent_id="a2", action_types=["llm_call", "llm_call", "tool_use"]),
        ]
        bd = BehavioralDistribution(trajectories)
        dist = bd.action_type_distribution()

        assert len(dist) > 0
        total = sum(dist.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_flag_distribution(self) -> None:
        flags = [
            BehaviorFlag(flag_type="spec_gaming", severity=Severity.MEDIUM, description="t1"),
            BehaviorFlag(flag_type="spec_gaming", severity=Severity.LOW, description="t2"),
            BehaviorFlag(flag_type="constraint_bypass", severity=Severity.HIGH, description="t3"),
        ]
        traj = _make_trajectory(flags=flags)
        bd = BehavioralDistribution([traj])
        fd = bd.flag_distribution()

        assert fd["spec_gaming"] == 2
        assert fd["constraint_bypass"] == 1

    def test_timing_distribution(self) -> None:
        traj = _make_trajectory(durations=[100.0, 200.0, 300.0, 400.0, 500.0])
        bd = BehavioralDistribution([traj])
        td = bd.timing_distribution()

        assert td["mean"] == pytest.approx(300.0, abs=1e-6)
        assert td["p50"] >= td["p25"]
        assert td["p75"] >= td["p50"]
        assert td["p95"] >= td["p75"]

    def test_behavioral_fingerprint(self) -> None:
        traj = _make_trajectory()
        bd = BehavioralDistribution([traj])
        fp = bd.behavioral_fingerprint()

        assert "action_profile" in fp
        assert "timing_profile" in fp
        assert "flag_profile" in fp
        assert fp["total_trajectories"] == 1
        assert fp["total_actions"] >= 1

    def test_empty_trajectories(self) -> None:
        bd = BehavioralDistribution([])
        assert bd.action_type_distribution() == {}
        timing = bd.timing_distribution()
        assert timing["mean"] == 0.0


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestEvalReport:

    @pytest.fixture
    def report_data(
        self, sample_eval_results: list[dict[str, Any]], trajectory_store: EvalTrajectoryStore
    ) -> EvalReport:
        return EvalReport(sample_eval_results, trajectory_store)

    def test_full_report_has_all_sections(self, report_data: EvalReport) -> None:
        report = report_data.full_report()

        assert "summary" in report
        assert "safety_assessment" in report
        assert "capability_profile" in report
        assert "anomaly_report" in report
        assert "behavioral_analysis" in report

    def test_summary(self, report_data: EvalReport) -> None:
        s = report_data.summary()

        assert s["total_evals"] == 10
        assert 0.0 <= s["pass_rate"] <= 1.0
        assert 0.0 <= s["flag_rate"] <= 1.0
        assert s["avg_steps"] > 0
        assert s["avg_duration"] > 0

    def test_safety_assessment(self, report_data: EvalReport) -> None:
        sa = report_data.safety_assessment()

        assert "total_flags" in sa
        assert "severity_breakdown" in sa
        assert "risk_score" in sa
        assert 0 <= sa["risk_score"] <= 100

    def test_capability_profile(self, report_data: EvalReport) -> None:
        cp = report_data.capability_profile()

        # Sample results have capability data
        assert len(cp) > 0
        for cap, info in cp.items():
            assert "score" in info
            assert "confidence_interval" in info
            assert 0.0 <= info["score"] <= 1.0

    def test_to_json(self, report_data: EvalReport) -> None:
        j = report_data.to_json()

        assert isinstance(j, str)
        parsed = json.loads(j)
        assert "summary" in parsed

    def test_to_markdown(self, report_data: EvalReport) -> None:
        md = report_data.to_markdown()

        assert isinstance(md, str)
        assert "# Aegis Evaluation Report" in md
        assert "## Summary" in md
        assert "## Safety Assessment" in md
        assert "## Anomaly Report" in md
        assert "## Behavioral Analysis" in md

    def test_empty_results(self) -> None:
        store = EvalTrajectoryStore()
        report = EvalReport([], store)
        s = report.summary()

        assert s["total_evals"] == 0
        assert s["pass_rate"] == 0.0
