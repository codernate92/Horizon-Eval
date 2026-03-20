"""Full evaluation report generation.

Combines statistical analysis, anomaly detection, and behavioral profiling
into structured reports suitable for alignment research review.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.types import BehaviorFlag, EvalStatus, Severity
from core.trajectory import EvalTrajectory, EvalTrajectoryStore
from core.runner import EvalResult

from .statistics import (
    BehavioralDistribution,
    CapabilityEstimator,
    compute_confidence_interval,
)
from .anomaly import AnomalyDetector

logger = logging.getLogger(__name__)


class EvalReport:
    """Generate comprehensive evaluation reports.

    Aggregates EvalResult dicts, trajectory data, and task metadata into
    structured report sections: summary, safety assessment, capability
    profile, anomaly report, and behavioral analysis.

    Args:
        eval_results: List of EvalResult dicts. Each dict must contain
            at least ``task_id``, ``agent_id``, ``status``, ``success``,
            ``duration``, ``steps_taken``, and ``behavior_flags``.
        trajectory_store: Populated EvalTrajectoryStore.
        task_metadata: Optional dict of task-level metadata keyed by task_id.
    """

    def __init__(
        self,
        eval_results: list[dict[str, Any]],
        trajectory_store: EvalTrajectoryStore,
        task_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.eval_results = eval_results
        self.trajectory_store = trajectory_store
        self.task_metadata = task_metadata or {}

    # -- helpers --------------------------------------------------------------

    def _all_trajectories(self) -> list[EvalTrajectory]:
        """Collect all trajectories from the store."""
        return list(self.trajectory_store.all_trajectories().values())

    def _all_flags(self) -> list[BehaviorFlag]:
        """Collect all behavior flags from eval results."""
        flags: list[BehaviorFlag] = []
        for r in self.eval_results:
            raw_flags = r.get("behavior_flags", [])
            for f in raw_flags:
                if isinstance(f, BehaviorFlag):
                    flags.append(f)
                elif isinstance(f, dict):
                    flags.append(BehaviorFlag.from_dict(f))
        return flags

    # -- report sections ------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """High-level evaluation summary.

        Returns:
            Dict with pass_rate, flag_rate, avg_steps, avg_duration,
            total_evals, and capability_overview.
        """
        total = len(self.eval_results)
        if total == 0:
            return {
                "total_evals": 0,
                "pass_rate": 0.0,
                "flag_rate": 0.0,
                "avg_steps": 0.0,
                "avg_duration": 0.0,
                "capability_overview": {},
            }

        passed = sum(1 for r in self.eval_results if r.get("success", False))
        flagged = sum(1 for r in self.eval_results if r.get("status") in ("flagged", EvalStatus.FLAGGED))
        durations = [r.get("duration", 0.0) for r in self.eval_results]
        steps = [r.get("steps_taken", 0) for r in self.eval_results]

        # Quick capability overview
        cap_scores: dict[str, list[float]] = {}
        for r in self.eval_results:
            cap = r.get("capability")
            score = r.get("score")
            if cap and score is not None:
                cap_scores.setdefault(cap, []).append(score)
        cap_overview = {
            k: sum(v) / len(v) for k, v in cap_scores.items()
        }

        return {
            "total_evals": total,
            "pass_rate": passed / total,
            "flag_rate": flagged / total,
            "avg_steps": sum(steps) / total,
            "avg_duration": sum(durations) / total,
            "capability_overview": cap_overview,
        }

    def safety_assessment(self) -> dict[str, Any]:
        """Aggregated safety assessment based on behavioral flags.

        Returns:
            Dict with severity_breakdown, top_concerns, and risk_score (0-100).
        """
        flags = self._all_flags()

        severity_counts: dict[str, int] = {}
        flag_type_counts: dict[str, int] = {}
        for f in flags:
            sev = f.severity.name if isinstance(f.severity, Severity) else str(f.severity)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            flag_type_counts[f.flag_type] = flag_type_counts.get(f.flag_type, 0) + 1

        # Top concerns by count
        top_concerns = sorted(
            flag_type_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Risk score: weighted sum of severity counts, normalized to 0-100
        severity_weights = {
            "INFO": 1,
            "LOW": 5,
            "MEDIUM": 15,
            "HIGH": 30,
            "CRITICAL": 50,
        }
        raw_risk = sum(
            severity_weights.get(sev, 0) * count
            for sev, count in severity_counts.items()
        )
        total = max(len(self.eval_results), 1)
        # Normalize: 100 points would mean every eval has a CRITICAL flag
        risk_score = min(100.0, (raw_risk / total) * 2)

        return {
            "total_flags": len(flags),
            "severity_breakdown": severity_counts,
            "top_concerns": [{"flag_type": ft, "count": c} for ft, c in top_concerns],
            "risk_score": round(risk_score, 1),
        }

    def capability_profile(self) -> dict[str, Any]:
        """Per-capability scores with confidence intervals.

        Returns:
            Dict mapping capability names to score information.
            Returns empty dict if no capability data is available.
        """
        # Filter eval results that have capability/score fields
        cap_results = [
            r for r in self.eval_results
            if r.get("capability") and r.get("score") is not None
        ]
        if not cap_results:
            return {}

        estimator = CapabilityEstimator(cap_results)
        all_scores = estimator.estimate_all()
        profile: dict[str, Any] = {}
        for cap, cs in all_scores.items():
            profile[cap] = {
                "score": cs.score,
                "confidence_interval": list(cs.confidence_interval),
                "sample_size": cs.sample_size,
                "difficulty_adjusted": estimator.difficulty_adjusted_score(cap),
            }
        return profile

    def anomaly_report(self) -> dict[str, Any]:
        """Run anomaly detection on all trajectories.

        Returns:
            Dict with anomaly_count and list of anomaly details.
        """
        trajectories = self._all_trajectories()
        if not trajectories:
            return {"anomaly_count": 0, "anomalies": []}

        detector = AnomalyDetector(trajectories, self.task_metadata.get("anomaly_config"))
        anomalies = detector.detect_all()

        # Also check capability inconsistencies if data is available
        cap_results = [
            r for r in self.eval_results
            if r.get("capability") and r.get("score") is not None
        ]
        if cap_results:
            anomalies.extend(detector.detect_capability_inconsistencies(cap_results))

        return {
            "anomaly_count": len(anomalies),
            "anomalies": [
                {
                    "type": a.anomaly_type.value,
                    "description": a.description,
                    "severity": a.severity.name,
                    "z_score": a.z_score,
                    "affected_trajectories": a.affected_trajectories,
                    "evidence": a.evidence,
                }
                for a in anomalies
            ],
        }

    def behavioral_analysis(self) -> dict[str, Any]:
        """Behavioral distribution and fingerprint analysis.

        Returns:
            Dict with action_distribution, timing, and behavioral_fingerprint.
        """
        trajectories = self._all_trajectories()
        if not trajectories:
            return {
                "action_distribution": {},
                "timing": {},
                "behavioral_fingerprint": {},
            }

        bd = BehavioralDistribution(trajectories)
        return {
            "action_distribution": bd.action_type_distribution(),
            "timing": bd.timing_distribution(),
            "flag_distribution": bd.flag_distribution(),
            "behavioral_fingerprint": bd.behavioral_fingerprint(),
        }

    def full_report(self) -> dict[str, Any]:
        """Generate the complete report combining all sections.

        Returns:
            Dict with keys: summary, safety_assessment, capability_profile,
            anomaly_report, behavioral_analysis.
        """
        return {
            "summary": self.summary(),
            "safety_assessment": self.safety_assessment(),
            "capability_profile": self.capability_profile(),
            "anomaly_report": self.anomaly_report(),
            "behavioral_analysis": self.behavioral_analysis(),
        }

    def to_json(self) -> str:
        """Serialize the full report to a JSON string."""
        return json.dumps(self.full_report(), indent=2, default=str)

    def to_markdown(self) -> str:
        """Render the full report as human-readable Markdown.

        Returns:
            Markdown string with tables and sections.
        """
        report = self.full_report()
        lines: list[str] = []

        # Title
        lines.append("# Aegis Evaluation Report")
        lines.append("")

        # Summary
        s = report["summary"]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Evaluations | {s['total_evals']} |")
        lines.append(f"| Pass Rate | {s['pass_rate']:.1%} |")
        lines.append(f"| Flag Rate | {s['flag_rate']:.1%} |")
        lines.append(f"| Avg Steps | {s['avg_steps']:.1f} |")
        lines.append(f"| Avg Duration | {s['avg_duration']:.2f}s |")
        lines.append("")

        if s.get("capability_overview"):
            lines.append("### Capability Overview")
            lines.append("")
            lines.append("| Capability | Mean Score |")
            lines.append("|------------|-----------|")
            for cap, score in s["capability_overview"].items():
                lines.append(f"| {cap} | {score:.3f} |")
            lines.append("")

        # Safety Assessment
        sa = report["safety_assessment"]
        lines.append("## Safety Assessment")
        lines.append("")
        lines.append(f"**Risk Score:** {sa['risk_score']}/100")
        lines.append(f"**Total Flags:** {sa['total_flags']}")
        lines.append("")

        if sa["severity_breakdown"]:
            lines.append("### Severity Breakdown")
            lines.append("")
            lines.append("| Severity | Count |")
            lines.append("|----------|-------|")
            for sev, count in sa["severity_breakdown"].items():
                lines.append(f"| {sev} | {count} |")
            lines.append("")

        if sa["top_concerns"]:
            lines.append("### Top Concerns")
            lines.append("")
            for tc in sa["top_concerns"]:
                lines.append(f"- **{tc['flag_type']}**: {tc['count']} occurrence(s)")
            lines.append("")

        # Capability Profile
        cp = report["capability_profile"]
        if cp:
            lines.append("## Capability Profile")
            lines.append("")
            lines.append("| Capability | Score | 95% CI | Difficulty-Adjusted | Samples |")
            lines.append("|------------|-------|--------|---------------------|---------|")
            for cap, info in cp.items():
                ci = info["confidence_interval"]
                lines.append(
                    f"| {cap} | {info['score']:.3f} | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
                    f"{info['difficulty_adjusted']:.3f} | {info['sample_size']} |"
                )
            lines.append("")

        # Anomaly Report
        ar = report["anomaly_report"]
        lines.append("## Anomaly Report")
        lines.append("")
        lines.append(f"**Anomalies Detected:** {ar['anomaly_count']}")
        lines.append("")

        if ar["anomalies"]:
            for i, a in enumerate(ar["anomalies"], 1):
                lines.append(f"### Anomaly {i}: {a['type']}")
                lines.append(f"- **Severity:** {a['severity']}")
                lines.append(f"- **Description:** {a['description']}")
                if a["z_score"] is not None:
                    lines.append(f"- **Z-Score:** {a['z_score']:.2f}")
                lines.append("")

        # Behavioral Analysis
        ba = report["behavioral_analysis"]
        lines.append("## Behavioral Analysis")
        lines.append("")

        if ba.get("action_distribution"):
            lines.append("### Action Type Distribution")
            lines.append("")
            lines.append("| Action Type | Frequency |")
            lines.append("|-------------|-----------|")
            for at, freq in ba["action_distribution"].items():
                lines.append(f"| {at} | {freq:.3f} |")
            lines.append("")

        if ba.get("timing"):
            t = ba["timing"]
            lines.append("### Timing Profile")
            lines.append("")
            lines.append("| Statistic | Value (ms) |")
            lines.append("|-----------|-----------|")
            for k, v in t.items():
                lines.append(f"| {k} | {v:.1f} |")
            lines.append("")

        fp = ba.get("behavioral_fingerprint", {})
        if fp:
            lines.append("### Behavioral Fingerprint")
            lines.append("")
            lines.append(f"- **Total Trajectories:** {fp.get('total_trajectories', 0)}")
            lines.append(f"- **Total Actions:** {fp.get('total_actions', 0)}")
            lines.append(f"- **Total Flags:** {fp.get('total_flags', 0)}")
            lines.append(f"- **Flag Rate:** {fp.get('flag_rate', 0):.2f} per trajectory")
            lines.append("")

        lines.append("---")
        lines.append("*Generated by Aegis Evaluation Framework*")
        lines.append("")

        return "\n".join(lines)
