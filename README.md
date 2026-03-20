# Horizon-Eval

Evaluation-integrity MVP for long-horizon agent benchmarks.

## What It Is

Horizon-Eval is an IntegrityBench-style benchmark harness for portable agent-evaluation tasks with:

- serializable task specs
- hermetic execution policy metadata
- human QA gates
- trajectory capture
- integrity monitors for shortcutting, constraint bypass, canary leakage, and protected-state tampering
- replayable run bundles with hashes and HTML reports
- a safety-gap and safeguard-regression lab for comparing base, mitigated, and attacked variants

This is the kind of infrastructure that is useful for serious eval work, not just demos.

## MVP Surface

- `core/specs.py`: portable task SDK
- `core/harness.py`: end-to-end execution with monitors, QA, and bundle export
- `core/bundle.py`: immutable run-bundle writer
- `core/reporting.py`: HTML report generation
- `core/monitors.py`: integrity checks
- `scenarios/demo.py`: demo task plus aligned and shortcut agents
- `scenarios/safety_gap.py`: default safety-gap suite and scripted model variants
- `analysis/safety_gap.py`: variant comparison, safety-gap metrics, and regression reporting
- `cli.py`: runnable entrypoint

## Run The Demo

```bash
python3 cli.py --agent aligned
python3 cli.py --agent shortcut
python3 cli.py --agent canary
python3 cli.py --mode safety-gap --output-root artifacts/safety-gap
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

The key problem is not just “can an agent solve a task?” but “can it solve the task without gaming the scorer, bypassing constraints, or leaking hidden benchmark structure?” Horizon-Eval focuses on that layer.

The new safety-gap lab makes a second question explicit: “How much safer is the deployed, mitigated model than the underlying base model, and how much of that gain survives prompt-based safeguard erosion?” That is the practical release-gating question most benchmark stacks still do not answer cleanly.

## Notes

- The current MVP is stdlib-only by design to keep the surface small and auditable.
- The next serious extension would be a true hermetic Docker runner, hidden holdout task variants, and richer scorer-integrity checks.
