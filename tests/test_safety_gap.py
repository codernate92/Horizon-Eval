from pathlib import Path
import json

from analysis.safety_gap import SafetyGapLab
from core.harness import HarnessConfig
from core.qa import make_default_attestation
from scenarios.safety_gap import DEFAULT_SAFETY_GAP_VARIANTS, build_safety_gap_suite


def test_safety_gap_lab_orders_variants(tmp_path: Path) -> None:
    lab = SafetyGapLab(
        tasks=build_safety_gap_suite(),
        variants=DEFAULT_SAFETY_GAP_VARIANTS,
        harness_config=HarnessConfig(output_root=tmp_path, write_bundle=False),
        qa_attestation=make_default_attestation(reviewer="pytest"),
    )

    report = lab.run_sync()
    summaries = {summary.variant_id: summary for summary in report.summaries}

    assert summaries["mitigated"].mean_risk_points < summaries["base"].mean_risk_points
    assert summaries["mitigated"].mean_risk_points < summaries["attacked"].mean_risk_points
    assert summaries["mitigated"].clean_pass_rate > summaries["base"].clean_pass_rate
    assert summaries["mitigated"].clean_pass_rate > summaries["attacked"].clean_pass_rate
    assert summaries["mitigated"].capability_pass_rate == 1.0

    assert report.headlines["safety_gap"]["risk_reduction"] > 0
    assert report.headlines["attack_regression"]["risk_regression"] > 0
    assert any(regression.clean_pass_dropped for regression in report.regressions)


def test_safety_gap_report_serializes(tmp_path: Path) -> None:
    lab = SafetyGapLab(
        tasks=build_safety_gap_suite(),
        variants=DEFAULT_SAFETY_GAP_VARIANTS,
        harness_config=HarnessConfig(output_root=tmp_path, write_bundle=False),
        qa_attestation=make_default_attestation(reviewer="pytest"),
    )

    report = lab.run_sync()
    payload = json.loads(report.to_json())

    assert payload["suite_name"] == "safety-gap-suite"
    assert len(payload["summaries"]) == 3
    assert len(payload["records"]) == len(build_safety_gap_suite()) * 3
