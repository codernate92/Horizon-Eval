"""HTML reporting for evaluation runs."""

from __future__ import annotations

import html

from .runner import EvalResult
from .specs import TaskSpec
from .qa import QAGateResult


def render_html_report(
    task_spec: TaskSpec,
    result: EvalResult,
    qa_gate: QAGateResult,
) -> str:
    flags = "".join(
        f"<li><strong>{html.escape(flag.flag_type)}</strong>: {html.escape(flag.description)}</li>"
        for flag in result.behavior_flags
    ) or "<li>No flags.</li>"

    events = "".join(
        f"<tr><td>{index}</td><td>{html.escape(event.event_type)}</td><td><pre>{html.escape(str(event.data))}</pre></td></tr>"
        for index, event in enumerate(result.trajectory.events if result.trajectory else [])
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(task_spec.name)} - Eval Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    .status {{ font-weight: 700; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
  </style>
</head>
<body>
  <h1>{html.escape(task_spec.name)}</h1>
  <p>{html.escape(task_spec.prompt)}</p>
  <h2>Summary</h2>
  <table>
    <tr><th>Status</th><td class="status">{html.escape(result.status.value)}</td></tr>
    <tr><th>Success</th><td>{result.success}</td></tr>
    <tr><th>Steps</th><td>{result.steps_taken}</td></tr>
    <tr><th>Duration (s)</th><td>{result.duration:.3f}</td></tr>
    <tr><th>Task Fingerprint</th><td>{html.escape(result.metadata.get("task_fingerprint", ""))}</td></tr>
    <tr><th>QA Approved</th><td>{qa_gate.approved}</td></tr>
  </table>
  <h2>Flags</h2>
  <ul>{flags}</ul>
  <h2>Trajectory</h2>
  <table>
    <tr><th>#</th><th>Event Type</th><th>Data</th></tr>
    {events}
  </table>
</body>
</html>
"""
