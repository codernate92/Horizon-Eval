"""CLI for the Aegis IntegrityBench-style MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.harness import HarnessConfig, IntegrityHarness
from core.qa import make_approved_demo_attestation
from scenarios.demo import (
    AlignedDemoAgent,
    CanaryLeakAgent,
    ShortcutDemoAgent,
    build_demo_task_spec,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Aegis demo evaluations.")
    parser.add_argument(
        "--agent",
        choices=("aligned", "shortcut", "canary"),
        default="aligned",
        help="Which demo agent to run.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts",
        help="Directory for run bundles.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    task_spec = build_demo_task_spec()
    agents = {
        "aligned": AlignedDemoAgent(),
        "shortcut": ShortcutDemoAgent(),
        "canary": CanaryLeakAgent(),
    }
    harness = IntegrityHarness(
        HarnessConfig(output_root=Path(args.output_root), write_bundle=True)
    )
    run = harness.run_sync(
        task_spec=task_spec,
        agent=agents[args.agent],
        agent_id=f"demo-{args.agent}",
        qa_attestation=make_approved_demo_attestation(),
    )
    payload = {
        "status": run.result.status.value,
        "success": run.result.success,
        "flags": [flag.to_dict() for flag in run.result.behavior_flags],
        "bundle_dir": str(run.bundle_paths.bundle_dir) if run.bundle_paths else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
