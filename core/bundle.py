"""Bundle writer for replayable evaluation runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qa import QAAttestation, QAGateResult
from .runner import EvalResult
from .specs import TaskSpec


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return _sha256_bytes(payload)


@dataclass
class BundlePaths:
    bundle_dir: Path
    manifest_path: Path
    report_path: Path
    result_path: Path
    task_path: Path
    events_path: Path


class RunBundleWriter:
    """Writes a self-contained run bundle with content hashes."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def write_bundle(
        self,
        task_spec: TaskSpec,
        result: EvalResult,
        qa_attestation: QAAttestation | None,
        qa_gate: QAGateResult,
        report_html: str,
    ) -> BundlePaths:
        run_id = f"{task_spec.slug}-{result.agent_id}-{result.task_id}"
        bundle_dir = self.output_root / run_id
        bundle_dir.mkdir(parents=True, exist_ok=True)

        task_path = bundle_dir / "task_spec.json"
        result_path = bundle_dir / "result.json"
        events_path = bundle_dir / "events.jsonl"
        report_path = bundle_dir / "report.html"
        qa_path = bundle_dir / "qa.json"
        manifest_path = bundle_dir / "manifest.json"

        task_hash = _write_bytes(
            task_path,
            json.dumps(task_spec.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        result_hash = _write_bytes(
            result_path,
            json.dumps(result.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        events_hash = _write_bytes(
            events_path,
            (
                "".join(
                    json.dumps(event.to_dict(), sort_keys=True) + "\n"
                    for event in (result.trajectory.events if result.trajectory else [])
                )
            ).encode("utf-8"),
        )
        report_hash = _write_bytes(report_path, report_html.encode("utf-8"))
        qa_payload = {
            "qa_gate": {
                "approved": qa_gate.approved,
                "missing_required_checks": qa_gate.missing_required_checks,
                "failed_required_checks": qa_gate.failed_required_checks,
            },
            "qa_attestation": None
            if qa_attestation is None
            else {
                "reviewer": qa_attestation.reviewer,
                "passed_checks": qa_attestation.passed_checks,
                "notes": qa_attestation.notes,
            },
        }
        qa_hash = _write_bytes(
            qa_path,
            json.dumps(qa_payload, indent=2, sort_keys=True).encode("utf-8"),
        )

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "task_fingerprint": task_spec.fingerprint(),
            "agent_id": result.agent_id,
            "status": result.status.value,
            "files": {
                "task_spec.json": task_hash,
                "result.json": result_hash,
                "events.jsonl": events_hash,
                "report.html": report_hash,
                "qa.json": qa_hash,
            },
            "hermetic_policy": task_spec.hermetic.to_dict(),
        }
        _write_bytes(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )

        return BundlePaths(
            bundle_dir=bundle_dir,
            manifest_path=manifest_path,
            report_path=report_path,
            result_path=result_path,
            task_path=task_path,
            events_path=events_path,
        )
