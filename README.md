# Aegis

Evaluation-integrity MVP for long-horizon agent benchmarks.

## What It Is

Aegis is an IntegrityBench-style benchmark harness for portable agent-evaluation tasks with:

- serializable task specs
- hermetic execution policy metadata
- human QA gates
- trajectory capture
- integrity monitors for shortcutting, constraint bypass, canary leakage, and protected-state tampering
- replayable run bundles with hashes and HTML reports

This is the kind of infrastructure that is useful for serious eval work, not just demos.

## MVP Surface

- `core/specs.py`: portable task SDK
- `core/harness.py`: end-to-end execution with monitors, QA, and bundle export
- `core/bundle.py`: immutable run-bundle writer
- `core/reporting.py`: HTML report generation
- `core/monitors.py`: integrity checks
- `scenarios/demo.py`: demo task plus aligned and shortcut agents
- `cli.py`: runnable entrypoint

## Run The Demo

```bash
python3 cli.py --agent aligned
python3 cli.py --agent shortcut
python3 cli.py --agent canary
python3 -m pytest tests
```

Each run writes a bundle under `artifacts/` containing:

- `task_spec.json`
- `result.json`
- `events.jsonl`
- `qa.json`
- `report.html`
- `manifest.json`

## Why This Is Interesting

The key problem is not just “can an agent solve a task?” but “can it solve the task without gaming the scorer, bypassing constraints, or leaking hidden benchmark structure?” Aegis focuses on that layer.

## Notes

- The current MVP is stdlib-only by design to keep the surface small and auditable.
- The next serious extension would be a true hermetic Docker runner, hidden holdout task variants, and richer scorer-integrity checks.
