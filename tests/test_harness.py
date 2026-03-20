from pathlib import Path

import pytest

from core.harness import HarnessConfig, IntegrityHarness
from core.qa import QAAttestation, make_approved_demo_attestation
from core.types import EvalStatus
from scenarios.demo import (
    AlignedDemoAgent,
    CanaryLeakAgent,
    ShortcutDemoAgent,
    build_demo_task_spec,
)


def test_aligned_agent_passes_and_writes_bundle(tmp_path: Path) -> None:
    harness = IntegrityHarness(HarnessConfig(output_root=tmp_path, write_bundle=True))
    run = harness.run_sync(
        build_demo_task_spec(),
        AlignedDemoAgent(),
        agent_id="aligned-demo",
        qa_attestation=make_approved_demo_attestation(),
    )

    assert run.result.status == EvalStatus.PASSED
    assert run.bundle_paths is not None
    assert run.bundle_paths.manifest_path.exists()
    assert run.bundle_paths.report_path.exists()


def test_shortcut_agent_is_flagged(tmp_path: Path) -> None:
    harness = IntegrityHarness(HarnessConfig(output_root=tmp_path, write_bundle=False))
    run = harness.run_sync(
        build_demo_task_spec(),
        ShortcutDemoAgent(),
        agent_id="shortcut-demo",
        qa_attestation=make_approved_demo_attestation(),
    )

    assert run.result.status == EvalStatus.FLAGGED
    assert any(flag.flag_type == "constraint_bypass" for flag in run.result.behavior_flags)
    assert any(flag.flag_type == "reward_hack" for flag in run.result.behavior_flags)


def test_canary_leak_is_critical_flag(tmp_path: Path) -> None:
    harness = IntegrityHarness(HarnessConfig(output_root=tmp_path, write_bundle=False))
    run = harness.run_sync(
        build_demo_task_spec(),
        CanaryLeakAgent(),
        agent_id="canary-demo",
        qa_attestation=make_approved_demo_attestation(),
    )

    assert any(flag.flag_type == "canary_exposure" for flag in run.result.behavior_flags)


def test_qa_gate_blocks_unapproved_task(tmp_path: Path) -> None:
    harness = IntegrityHarness(HarnessConfig(output_root=tmp_path, write_bundle=False))
    attestation = QAAttestation(reviewer="qa", passed_checks={"dry_run": True})

    with pytest.raises(ValueError):
        harness.run_sync(
            build_demo_task_spec(),
            AlignedDemoAgent(),
            agent_id="blocked-demo",
            qa_attestation=attestation,
        )
