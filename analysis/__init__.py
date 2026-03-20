"""Aegis analysis — statistical analysis, anomaly detection, and reporting.

Re-exports key classes for convenient access:

    from analysis import CapabilityEstimator, AnomalyDetector, EvalReport
"""

from .statistics import (
    ConfidenceInterval,
    compute_confidence_interval,
    CapabilityEstimator,
    BehavioralDistribution,
)

from .anomaly import (
    AnomalyType,
    Anomaly,
    AnomalyDetector,
)

from .report import EvalReport

__all__ = [
    # statistics
    "ConfidenceInterval",
    "compute_confidence_interval",
    "CapabilityEstimator",
    "BehavioralDistribution",
    # anomaly
    "AnomalyType",
    "Anomaly",
    "AnomalyDetector",
    # report
    "EvalReport",
]
