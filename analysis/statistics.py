"""Statistical analysis for evaluation results.

Provides confidence interval computation, capability estimation with
difficulty weighting, and behavioral distribution analysis for comparing
agent trajectories.

Uses scipy.stats when available for precise t-distribution calculations;
falls back to manual approximations for environments without scipy.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.types import CapabilityScore, Severity
from core.trajectory import EvalTrajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# scipy availability — graceful fallback
# ---------------------------------------------------------------------------
try:
    from scipy import stats as scipy_stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    logger.info("scipy not available; using manual statistical approximations")


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    """A computed confidence interval for a sample statistic.

    Attributes:
        mean: Sample mean.
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        sample_size: Number of observations.
    """

    mean: float
    lower: float
    upper: float
    confidence_level: float
    sample_size: int


def _t_critical_approx(confidence: float, df: int) -> float:
    """Approximate the t-distribution critical value without scipy.

    Uses the approximation from Abramowitz and Stegun for the inverse
    normal, then applies the Cornish-Fisher correction for the t-distribution.
    Accurate to ~0.01 for df >= 2.
    """
    alpha = 1.0 - confidence
    p = 1.0 - alpha / 2.0

    # Rational approximation of the inverse normal (Abramowitz & Stegun 26.2.23)
    t_val = p
    if p < 0.5:
        t_val = 1.0 - p
    t_inner = math.sqrt(-2.0 * math.log(1.0 - t_val))

    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    z = t_inner - (c0 + c1 * t_inner + c2 * t_inner**2) / (
        1.0 + d1 * t_inner + d2 * t_inner**2 + d3 * t_inner**3
    )
    if p < 0.5:
        z = -z

    # Cornish-Fisher expansion for t-distribution
    g1 = (z**3 + z) / (4.0 * df)
    g2 = (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    return z + g1 + g2


def compute_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute a confidence interval for the mean of *scores*.

    Uses the t-distribution for small samples (n < 30) and the normal
    approximation for large samples. Requires at least 2 data points.

    Args:
        scores: List of numeric observations.
        confidence: Desired confidence level (0 < confidence < 1).

    Returns:
        A ConfidenceInterval dataclass.

    Raises:
        ValueError: If fewer than 2 scores are provided or confidence
            is outside (0, 1).
    """
    if len(scores) < 2:
        raise ValueError("At least 2 scores are required to compute a CI")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    std_err = math.sqrt(variance / n)

    if std_err == 0.0:
        return ConfidenceInterval(
            mean=mean,
            lower=mean,
            upper=mean,
            confidence_level=confidence,
            sample_size=n,
        )

    df = n - 1
    use_t = n < 30

    if use_t:
        if _HAS_SCIPY:
            t_crit = scipy_stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df)
        else:
            t_crit = _t_critical_approx(confidence, df)
        margin = t_crit * std_err
    else:
        # Normal approximation for large n
        if _HAS_SCIPY:
            z_crit = scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
        else:
            z_crit = _t_critical_approx(confidence, 10_000)  # large df ≈ normal
        margin = z_crit * std_err

    return ConfidenceInterval(
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
        sample_size=n,
    )


# ---------------------------------------------------------------------------
# Capability estimation
# ---------------------------------------------------------------------------

_DIFFICULTY_WEIGHTS: dict[str, float] = {
    "easy": 0.5,
    "medium": 1.0,
    "hard": 1.5,
    "extreme": 2.0,
}


class CapabilityEstimator:
    """Estimate capability scores from structured evaluation result dicts.

    Each input dict must contain the keys ``capability``, ``score``,
    ``task_difficulty``, and ``agent_id``.

    Args:
        eval_results: List of result dicts.
    """

    def __init__(self, eval_results: list[dict[str, Any]]) -> None:
        self.eval_results = eval_results

    # -- helpers --------------------------------------------------------------

    def _scores_for(
        self,
        capability: str,
        agent_id: str | None = None,
    ) -> list[float]:
        """Return score values for a given capability and optional agent."""
        return [
            r["score"]
            for r in self.eval_results
            if r["capability"] == capability
            and (agent_id is None or r["agent_id"] == agent_id)
        ]

    def _difficulties_for(self, capability: str) -> list[str]:
        """Return difficulty labels for a given capability."""
        return [
            r["task_difficulty"]
            for r in self.eval_results
            if r["capability"] == capability
        ]

    def _all_capabilities(self) -> list[str]:
        """Return deduplicated list of capability names."""
        seen: set[str] = set()
        caps: list[str] = []
        for r in self.eval_results:
            cap = r["capability"]
            if cap not in seen:
                seen.add(cap)
                caps.append(cap)
        return caps

    # -- public API -----------------------------------------------------------

    def estimate_capability(self, capability: str) -> CapabilityScore:
        """Produce a point estimate with CI for a single capability.

        Args:
            capability: The capability name to estimate.

        Returns:
            A CapabilityScore with confidence interval.

        Raises:
            ValueError: If no results exist for the capability.
        """
        scores = self._scores_for(capability)
        if not scores:
            raise ValueError(f"No eval results for capability {capability!r}")

        if len(scores) == 1:
            return CapabilityScore(
                capability=capability,
                score=scores[0],
                confidence_interval=(0.0, 1.0),
                sample_size=1,
                notes="Single sample; CI is uninformative.",
            )

        ci = compute_confidence_interval(scores)
        # Clamp to [0, 1]
        lower = max(0.0, ci.lower)
        upper = min(1.0, ci.upper)
        return CapabilityScore(
            capability=capability,
            score=ci.mean,
            confidence_interval=(lower, upper),
            sample_size=ci.sample_size,
        )

    def estimate_all(self) -> dict[str, CapabilityScore]:
        """Estimate all capabilities present in the results.

        Returns:
            Mapping from capability name to CapabilityScore.
        """
        return {cap: self.estimate_capability(cap) for cap in self._all_capabilities()}

    def compare_agents(self, agent_a: str, agent_b: str) -> dict[str, Any]:
        """Per-capability comparison between two agents with effect sizes.

        Args:
            agent_a: First agent identifier.
            agent_b: Second agent identifier.

        Returns:
            Dict mapping capability names to comparison dicts containing
            mean_a, mean_b, difference, and cohen_d (effect size).
        """
        comparisons: dict[str, Any] = {}
        for cap in self._all_capabilities():
            scores_a = self._scores_for(cap, agent_a)
            scores_b = self._scores_for(cap, agent_b)
            if not scores_a or not scores_b:
                continue

            mean_a = sum(scores_a) / len(scores_a)
            mean_b = sum(scores_b) / len(scores_b)
            diff = mean_a - mean_b

            # Pooled standard deviation for Cohen's d
            n_a, n_b = len(scores_a), len(scores_b)
            var_a = sum((x - mean_a) ** 2 for x in scores_a) / max(n_a - 1, 1)
            var_b = sum((x - mean_b) ** 2 for x in scores_b) / max(n_b - 1, 1)
            pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(
                n_a + n_b - 2, 1
            )
            pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-9
            cohen_d = diff / pooled_std

            comparisons[cap] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "difference": diff,
                "cohen_d": cohen_d,
                "sample_size_a": n_a,
                "sample_size_b": n_b,
            }

        return comparisons

    def difficulty_adjusted_score(self, capability: str) -> float:
        """Compute a difficulty-weighted capability score.

        Harder tasks contribute more to the aggregate score. Weight
        mapping: easy=0.5, medium=1.0, hard=1.5, extreme=2.0.

        Args:
            capability: The capability to score.

        Returns:
            Weighted average score in [0, 1].

        Raises:
            ValueError: If no results exist for the capability.
        """
        relevant = [
            r for r in self.eval_results if r["capability"] == capability
        ]
        if not relevant:
            raise ValueError(f"No eval results for capability {capability!r}")

        total_weight = 0.0
        weighted_sum = 0.0
        for r in relevant:
            w = _DIFFICULTY_WEIGHTS.get(r["task_difficulty"], 1.0)
            weighted_sum += r["score"] * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Behavioral distribution analysis
# ---------------------------------------------------------------------------


class BehavioralDistribution:
    """Compute distributional statistics over evaluation trajectories.

    Accepts a list of EvalTrajectory objects and provides action-type
    frequencies, flag counts, timing percentiles, per-agent comparisons,
    and a composite behavioral fingerprint.

    Args:
        trajectories: List of EvalTrajectory instances to analyse.
    """

    def __init__(self, trajectories: list[EvalTrajectory]) -> None:
        self.trajectories = trajectories

    # -- helpers --------------------------------------------------------------

    def _all_action_types(self) -> list[str]:
        """Collect every action type across all trajectories."""
        types: list[str] = []
        for traj in self.trajectories:
            for action in traj.get_actions():
                types.append(action.action_type)
        return types

    def _all_durations(self) -> list[float]:
        """Collect per-event duration_ms values across all trajectories."""
        durations: list[float] = []
        for traj in self.trajectories:
            for event in traj.events:
                if event.duration_ms is not None:
                    durations.append(event.duration_ms)
        return durations

    @staticmethod
    def _percentile(sorted_data: list[float], p: float) -> float:
        """Compute the p-th percentile of sorted data (0 <= p <= 100)."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p / 100.0
        f = int(math.floor(k))
        c = min(f + 1, len(sorted_data) - 1)
        if f == c:
            return sorted_data[f]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    # -- public API -----------------------------------------------------------

    def action_type_distribution(self) -> dict[str, float]:
        """Normalized frequency of each action type across all trajectories.

        Returns:
            Dict mapping action_type to its relative frequency (sums to 1.0).
        """
        types = self._all_action_types()
        if not types:
            return {}
        counts = Counter(types)
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def flag_distribution(self) -> dict[str, int]:
        """Count of behavioral flags grouped by flag_type.

        Returns:
            Dict mapping flag_type to count.
        """
        counts: dict[str, int] = {}
        for traj in self.trajectories:
            for flag in traj.behavior_flags:
                counts[flag.flag_type] = counts.get(flag.flag_type, 0) + 1
        return counts

    def timing_distribution(self) -> dict[str, float]:
        """Descriptive statistics for action durations (in ms).

        Returns:
            Dict with keys: mean, std, p25, p50, p75, p95.
        """
        durations = self._all_durations()
        if not durations:
            return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}

        mean = sum(durations) / len(durations)
        variance = sum((d - mean) ** 2 for d in durations) / max(len(durations) - 1, 1)
        std = math.sqrt(variance)
        sorted_d = sorted(durations)

        return {
            "mean": mean,
            "std": std,
            "p25": self._percentile(sorted_d, 25),
            "p50": self._percentile(sorted_d, 50),
            "p75": self._percentile(sorted_d, 75),
            "p95": self._percentile(sorted_d, 95),
        }

    def per_agent_comparison(self, agent_ids: list[str]) -> dict[str, Any]:
        """Compute per-agent action and timing distributions.

        Args:
            agent_ids: Agent identifiers to include.

        Returns:
            Dict mapping agent_id to sub-dicts with action_distribution
            and timing keys.
        """
        result: dict[str, Any] = {}
        for aid in agent_ids:
            agent_trajs = [t for t in self.trajectories if t.agent_id == aid]
            if not agent_trajs:
                result[aid] = {"action_distribution": {}, "timing": {}}
                continue

            sub = BehavioralDistribution(agent_trajs)
            result[aid] = {
                "action_distribution": sub.action_type_distribution(),
                "timing": sub.timing_distribution(),
                "flag_distribution": sub.flag_distribution(),
                "trajectory_count": len(agent_trajs),
            }
        return result

    def behavioral_fingerprint(self) -> dict[str, Any]:
        """Compute a composite behavioural signature across all trajectories.

        The fingerprint is a structured dict that uniquely characterises
        the behavioural patterns observed, suitable for comparing agents
        or detecting distributional shifts over time.

        Returns:
            Dict with action_profile, timing_profile, and flag_profile.
        """
        action_dist = self.action_type_distribution()
        timing = self.timing_distribution()
        flags = self.flag_distribution()

        total_actions = sum(
            len(t.get_actions()) for t in self.trajectories
        )
        total_flags = sum(flags.values())

        return {
            "action_profile": action_dist,
            "timing_profile": timing,
            "flag_profile": flags,
            "total_trajectories": len(self.trajectories),
            "total_actions": total_actions,
            "total_flags": total_flags,
            "flag_rate": total_flags / max(len(self.trajectories), 1),
        }
