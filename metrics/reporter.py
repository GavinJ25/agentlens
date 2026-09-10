"""
Console reporting and final report.json writer.

Renders a human-readable console table summarising pass/fail status per
group and metric, and writes the full structured report to
outputs/results/report.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COL_WIDTHS = {"group": 18, "metric": 24, "value": 10, "threshold": 10, "status": 8}


def _format_row(group: str, metric: str, value: float, threshold: float, passed: bool) -> str:
    """
    Format a single console table row with fixed-width columns.

    Args:
        group:     Group name (e.g. "g1_text").
        metric:    Metric name (e.g. "rouge_l").
        value:     Observed metric value.
        threshold: Configured cutoff value.
        passed:    Whether the metric passed the threshold.

    Returns:
        A formatted string row.
    """
    status = "PASS" if passed else "FAIL"
    return (
        f"{group:<{_COL_WIDTHS['group']}}"
        f"{metric:<{_COL_WIDTHS['metric']}}"
        f"{value:<{_COL_WIDTHS['value']}.4f}"
        f"{threshold:<{_COL_WIDTHS['threshold']}.4f}"
        f"{status:<{_COL_WIDTHS['status']}}"
    )


def print_console_report(suite_report: dict[str, Any]) -> None:
    """
    Print a human-readable summary table of all group threshold checks.

    Args:
        suite_report: Output of metrics.aggregator.aggregate_suite().
    """
    header = (
        f"{'Group':<{_COL_WIDTHS['group']}}"
        f"{'Metric':<{_COL_WIDTHS['metric']}}"
        f"{'Value':<{_COL_WIDTHS['value']}}"
        f"{'Threshold':<{_COL_WIDTHS['threshold']}}"
        f"{'Status':<{_COL_WIDTHS['status']}}"
    )
    separator = "-" * sum(_COL_WIDTHS.values())

    lines = [separator, header, separator]

    for group_name, group_data in suite_report.get("groups", {}).items():
        check = group_data.get("threshold_check")
        if not check or not check.get("checks"):
            lines.append(f"{group_name:<{_COL_WIDTHS['group']}}(no threshold checks)")
            continue
        for c in check["checks"]:
            lines.append(_format_row(
                group_name, c["metric"], c["value"], c["threshold"], c["passed"]
            ))

    lines.append(separator)
    overall = "PASS" if suite_report.get("overall_passed") else "FAIL"
    lines.append(f"Overall suite result: {overall}")
    lines.append(separator)

    report_text = "\n".join(lines)
    print(report_text)
    logger.info("Console report printed. Overall result: %s", overall)


def write_report(suite_report: dict[str, Any], cfg: dict[str, Any]) -> Path:
    """
    Write the full suite report to outputs/results/report.json.

    Args:
        suite_report: Output of metrics.aggregator.aggregate_suite().
        cfg:          Full config dict loaded from config.yaml.

    Returns:
        Path to the written report.json file.
    """
    output_dir = Path(cfg["output"]["dir"]) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"

    report_path.write_text(json.dumps(suite_report, indent=2, default=str), encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path


def write_group_scores(
    group_name: str,
    group_data: dict[str, Any],
    cfg: dict[str, Any],
) -> Path:
    """
    Write a single group's aggregated scores to outputs/results/{group_name}_scores.json.

    Args:
        group_name: Name of the group, e.g. "g1_text".
        group_data: Output of metrics.aggregator.aggregate_group() for this group.
        cfg:        Full config dict loaded from config.yaml.

    Returns:
        Path to the written {group_name}_scores.json file.
    """
    output_dir = Path(cfg["output"]["dir"]) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / f"{group_name}_scores.json"

    scores_path.write_text(json.dumps(group_data, indent=2, default=str), encoding="utf-8")
    logger.debug("Group scores written to %s", scores_path)
    return scores_path