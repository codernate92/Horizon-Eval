"""Aegis core — model evaluation and behavioral auditing framework for LLM agents.

Re-exports all key types for convenient access:

    from aegis.core import EvalTask, EvalRunner, EvalResult, BehaviorFlag
"""

from .types import (
    EvalStatus,
    Severity,
    BehaviorFlag,
    AgentAction,
    CapabilityScore,
)

from .environment import (
    ToolPermission,
    EnvironmentConstraint,
    Environment,
)

from .eval_task import (
    SolutionPath,
    MisalignmentTrap,
    EvalTask,
)

from .trajectory import (
    EvalTrajectoryEvent,
    EvalTrajectory,
    EvalTrajectoryStore,
)

from .runner import (
    AgentUnderTest,
    EvalRunConfig,
    EvalResult,
    EvalRunner,
)
from .specs import (
    ToolSpec,
    ConstraintSpec,
    SuccessSpec,
    EvidenceSpec,
    TaskBudget,
    HermeticPolicy,
    TaskSpec,
)
from .qa import (
    QACheck,
    QAAttestation,
    QAGateResult,
    default_checklist,
    evaluate_qa_gate,
    make_default_attestation,
)
from .monitors import (
    IntegrityMonitor,
    ConstraintBypassMonitor,
    CompletionWithoutEvidenceMonitor,
    CanaryExposureMonitor,
    ProtectedStateMonitor,
    DEFAULT_MONITORS,
)
from .bundle import BundlePaths, RunBundleWriter
from .harness import HarnessConfig, HarnessRun, IntegrityHarness

__all__ = [
    # types
    "EvalStatus",
    "Severity",
    "BehaviorFlag",
    "AgentAction",
    "CapabilityScore",
    # environment
    "ToolPermission",
    "EnvironmentConstraint",
    "Environment",
    # eval_task
    "SolutionPath",
    "MisalignmentTrap",
    "EvalTask",
    # trajectory
    "EvalTrajectoryEvent",
    "EvalTrajectory",
    "EvalTrajectoryStore",
    # runner
    "AgentUnderTest",
    "EvalRunConfig",
    "EvalResult",
    "EvalRunner",
    # portable specs
    "ToolSpec",
    "ConstraintSpec",
    "SuccessSpec",
    "EvidenceSpec",
    "TaskBudget",
    "HermeticPolicy",
    "TaskSpec",
    # qa
    "QACheck",
    "QAAttestation",
    "QAGateResult",
    "default_checklist",
    "evaluate_qa_gate",
    "make_default_attestation",
    # monitors
    "IntegrityMonitor",
    "ConstraintBypassMonitor",
    "CompletionWithoutEvidenceMonitor",
    "CanaryExposureMonitor",
    "ProtectedStateMonitor",
    "DEFAULT_MONITORS",
    # bundle / harness
    "BundlePaths",
    "RunBundleWriter",
    "HarnessConfig",
    "HarnessRun",
    "IntegrityHarness",
]
