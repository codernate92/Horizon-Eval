"""Human QA gates for publishable evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .specs import SuccessSpec, TaskSpec


@dataclass(frozen=True)
class QACheck:
    """Single human-review requirement for a task."""

    check_id: str
    prompt: str
    required: bool = True


@dataclass
class QAAttestation:
    """Reviewer attestation over a checklist."""

    reviewer: str
    passed_checks: dict[str, bool]
    notes: dict[str, str] = field(default_factory=dict)

    def passed(self, check_id: str) -> bool:
        return bool(self.passed_checks.get(check_id, False))


@dataclass
class QAGateResult:
    """Outcome of running the QA gate for a task."""

    approved: bool
    missing_required_checks: list[str] = field(default_factory=list)
    failed_required_checks: list[str] = field(default_factory=list)


def default_checklist(_task_spec: TaskSpec) -> list[QACheck]:
    return [
        QACheck("dry_run", "A human completed a dry run of the task."),
        QACheck("loophole_scan", "A reviewer explicitly searched for shortcut or scorer-exploit loopholes."),
        QACheck("scoring_review", "A reviewer confirmed the success criterion matches task intent."),
        QACheck("sandbox_review", "The sandbox/tool permission model was reviewed."),
        QACheck("artifact_review", "The output bundle fields were reviewed for completeness."),
    ]


def evaluate_qa_gate(task_spec: TaskSpec, attestation: QAAttestation | None) -> QAGateResult:
    checklist = default_checklist(task_spec)
    if not task_spec.require_human_qa:
        return QAGateResult(approved=True)
    if attestation is None:
        return QAGateResult(
            approved=False,
            missing_required_checks=[check.check_id for check in checklist if check.required],
        )

    missing: list[str] = []
    failed: list[str] = []
    for check in checklist:
        if check.check_id not in attestation.passed_checks:
            if check.required:
                missing.append(check.check_id)
            continue
        if check.required and not attestation.passed(check.check_id):
            failed.append(check.check_id)
    return QAGateResult(
        approved=(not missing and not failed),
        missing_required_checks=missing,
        failed_required_checks=failed,
    )


def make_default_attestation(reviewer: str = "demo-reviewer") -> QAAttestation:
    checklist = default_checklist(
        TaskSpec(
            slug="qa-demo-task",
            name="QA Demo Task",
            prompt="Demonstrate the default QA checklist.",
            category="alignment",
            difficulty="easy",
            environment_name="qa-demo-environment",
            environment_description="Minimal task used to derive the default QA checklist.",
            initial_state={},
            success=SuccessSpec("state_key_equals", "ok", True),
        )
    )
    return QAAttestation(
        reviewer=reviewer,
        passed_checks={check.check_id: True for check in checklist},
    )


def make_approved_demo_attestation(reviewer: str = "demo-reviewer") -> QAAttestation:
    return make_default_attestation(reviewer=reviewer)
