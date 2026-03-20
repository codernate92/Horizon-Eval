"""Higher-level harness that adds QA, monitors, reporting, and bundle export."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bundle import BundlePaths, RunBundleWriter
from .monitors import DEFAULT_MONITORS, IntegrityMonitor, dedupe_flags
from .qa import QAAttestation, QAGateResult, evaluate_qa_gate
from .reporting import render_html_report
from .runner import AgentUnderTest, EvalResult, EvalRunConfig, EvalRunner
from .specs import TaskSpec
from .trajectory import EvalTrajectoryStore
from .types import EvalStatus, Severity


@dataclass
class HarnessConfig:
    output_root: Path = Path("artifacts")
    write_bundle: bool = True
    monitors: tuple[IntegrityMonitor, ...] = field(default_factory=lambda: DEFAULT_MONITORS)


@dataclass
class HarnessRun:
    task_spec: TaskSpec
    result: EvalResult
    qa_gate: QAGateResult
    bundle_paths: BundlePaths | None = None


class IntegrityHarness:
    """Runs portable task specs through the evaluation stack."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()

    async def run(
        self,
        task_spec: TaskSpec,
        agent: AgentUnderTest,
        agent_id: str,
        qa_attestation: QAAttestation | None = None,
    ) -> HarnessRun:
        task_spec.validate()
        qa_gate = evaluate_qa_gate(task_spec, qa_attestation)
        if not qa_gate.approved:
            missing = ", ".join(qa_gate.missing_required_checks + qa_gate.failed_required_checks)
            raise ValueError(f"QA gate not approved for task {task_spec.slug}: {missing}")

        task = task_spec.build_eval_task()
        runner = EvalRunner(
            EvalRunConfig(
                max_steps=task_spec.budget.max_steps,
                timeout=task_spec.budget.timeout_seconds,
            ),
            EvalTrajectoryStore(),
        )
        result = await runner.run_eval(task, agent, agent_id)
        result.metadata["task_fingerprint"] = task_spec.fingerprint()
        result.metadata["final_environment_state"] = dict(task.environment.state)
        result.metadata["state_history"] = list(task.environment._state_history)  # pylint: disable=protected-access
        result.metadata["hermetic_policy"] = task_spec.hermetic.to_dict()
        result.metadata["evidence_spec"] = task_spec.evidence.to_dict()

        new_flags = []
        for monitor in self.config.monitors:
            new_flags.extend(monitor.evaluate(task_spec, result))
        new_flags = dedupe_flags(new_flags)

        if result.trajectory is not None:
            existing = {(flag.flag_type, flag.description) for flag in result.behavior_flags}
            for flag in new_flags:
                key = (flag.flag_type, flag.description)
                if key not in existing:
                    result.trajectory.record_flag(flag)
                    result.behavior_flags.append(flag)
                    existing.add(key)

        if any(flag.severity >= Severity.HIGH for flag in result.behavior_flags):
            if result.status != EvalStatus.FAILED:
                result.status = EvalStatus.FLAGGED

        bundle_paths = None
        if self.config.write_bundle:
            self.config.output_root.mkdir(parents=True, exist_ok=True)
            bundle_writer = RunBundleWriter(self.config.output_root)
            bundle_paths = bundle_writer.write_bundle(
                task_spec=task_spec,
                result=result,
                qa_attestation=qa_attestation,
                qa_gate=qa_gate,
                report_html=render_html_report(task_spec, result, qa_gate),
            )

        return HarnessRun(
            task_spec=task_spec,
            result=result,
            qa_gate=qa_gate,
            bundle_paths=bundle_paths,
        )

    def run_sync(
        self,
        task_spec: TaskSpec,
        agent: AgentUnderTest,
        agent_id: str,
        qa_attestation: QAAttestation | None = None,
    ) -> HarnessRun:
        return asyncio.run(self.run(task_spec, agent, agent_id, qa_attestation))
