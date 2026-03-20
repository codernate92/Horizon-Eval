"""CLI for the Aegis IntegrityBench-style MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.safety_gap import SafetyGapLab
from core.harness import HarnessConfig, IntegrityHarness
from core.qa import make_approved_demo_attestation, make_default_attestation
from scenarios.demo import (
    AlignedDemoAgent,
    CanaryLeakAgent,
    ShortcutDemoAgent,
    build_demo_task_spec,
)
from scenarios.safety_gap import DEFAULT_SAFETY_GAP_VARIANTS, build_safety_gap_suite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Aegis evaluation workflows.")
    parser.add_argument(
        "--mode",
        choices=("demo", "safety-gap"),
        default="demo",
        help="Which workflow to run.",
    )
    parser.add_argument(
        "--agent",
        choices=("aligned", "shortcut", "canary"),
        default="aligned",
        help="Which demo agent to run when --mode=demo.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts",
        help="Directory for run bundles or study reports.",
    )
    parser.add_argument(
        "--write-bundles",
        action="store_true",
        help="Write per-run bundles for the safety-gap study.",
    )
    return parser


def _run_demo(output_root: Path, agent_name: str) -> dict[str, object]:
    task_spec = build_demo_task_spec()
    agents = {
        "aligned": AlignedDemoAgent(),
        "shortcut": ShortcutDemoAgent(),
        "canary": CanaryLeakAgent(),
    }
    harness = IntegrityHarness(
        HarnessConfig(output_root=output_root, write_bundle=True)
    )
    run = harness.run_sync(
        task_spec=task_spec,
        agent=agents[agent_name],
        agent_id=f"demo-{agent_name}",
        qa_attestation=make_approved_demo_attestation(),
    )
    return {
        "status": run.result.status.value,
        "success": run.result.success,
        "flags": [flag.to_dict() for flag in run.result.behavior_flags],
        "bundle_dir": str(run.bundle_paths.bundle_dir) if run.bundle_paths else None,
    }


def _run_safety_gap(output_root: Path, write_bundles: bool) -> dict[str, object]:
    lab = SafetyGapLab(
        tasks=build_safety_gap_suite(),
        variants=DEFAULT_SAFETY_GAP_VARIANTS,
        harness_config=HarnessConfig(output_root=output_root, write_bundle=write_bundles),
        qa_attestation=make_default_attestation(reviewer="safety-gap-reviewer"),
    )
    report = lab.run_sync()
    report_path = output_root / "safety_gap_report.json"
    report.write_json(report_path)
    return {
        "suite_name": report.suite_name,
        "report_path": str(report_path),
        "headlines": report.headlines,
        "summaries": [summary.to_dict() for summary in report.summaries],
        "regressions": [regression.to_dict() for regression in report.regressions],
    }


def main() -> None:
    args = _build_parser().parse_args()
    output_root = Path(args.output_root)
    if args.mode == "demo":
        payload = _run_demo(output_root=output_root, agent_name=args.agent)
    else:
        payload = _run_safety_gap(output_root=output_root, write_bundles=args.write_bundles)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
