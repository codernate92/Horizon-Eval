import hashlib
import json
from pathlib import Path

from core.harness import HarnessConfig, IntegrityHarness
from core.qa import make_approved_demo_attestation
from scenarios.demo import AlignedDemoAgent, build_demo_task_spec


def test_bundle_manifest_hashes_match_files(tmp_path: Path) -> None:
    harness = IntegrityHarness(HarnessConfig(output_root=tmp_path, write_bundle=True))
    run = harness.run_sync(
        build_demo_task_spec(),
        AlignedDemoAgent(),
        agent_id="bundle-demo",
        qa_attestation=make_approved_demo_attestation(),
    )

    assert run.bundle_paths is not None
    manifest = json.loads(run.bundle_paths.manifest_path.read_text())
    for rel_path, expected_hash in manifest["files"].items():
        payload = (run.bundle_paths.bundle_dir / rel_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash
