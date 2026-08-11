#!/usr/bin/env python3
"""Generate the frozen Phase-A baseline/angle matrix analysis without a weighted score."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import geometry_experiment as geometry  # noqa: E402


BASELINES = (0.0, 5.0, 12.5)
ANGLES = (5, 10, 15, 20)
INVALID_CONFIG = "B00_A05"
PRIMARY = "B05_A05"
BACKUP = "B00_A20"
VALIDATION = ("B00_A10", "B00_A15")

NUMERIC_HEATMAPS = (
    ("sensitivity_combined_px_per_mm", "01_sensitivity_combined.png", "Sensitivity combined (px/mm)", "viridis"),
    ("sigma_z_pred_combined_mm", "02_sigma_z_pred_combined.png", "Predicted height repeatability (mm)", "viridis_r"),
    ("sensitivity_h10", "03_sensitivity_h10.png", "H10 sensitivity (px/mm)", "viridis"),
    ("sensitivity_h30", "04_sensitivity_h30.png", "H30 sensitivity (px/mm)", "viridis"),
    ("sigma_pixel_h10_p95_px", "05_sigma_pixel_h10_p95.png", "H10 sigma_pixel P95 (px)", "viridis_r"),
    ("sigma_pixel_h30_p95_px", "06_sigma_pixel_h30_p95.png", "H30 sigma_pixel P95 (px)", "viridis_r"),
    ("reference_cv_interior_rmse_px", "07_reference_cv_interior_rmse.png", "Reference interior CV RMSE (px)", "viridis_r"),
)

OUTPUT_FIELDS = (
    "config_id",
    "baseline_scale_reading",
    "laser_angle_deg",
    "matrix_x_index",
    "matrix_y_index",
    "status",
    "phase_a_valid",
    "fov_status",
    "primary_quality_status",
    "primary_warnings",
    "all_warnings",
    "needs_manual_review",
    "h1_status",
    "h1_valid_column_fraction",
    "h1_detectability",
    "reference_cv_interior_rmse_px",
    "reference_cv_interior_p95_px",
    "sensitivity_h1",
    "sensitivity_h10",
    "sensitivity_h30",
    "sensitivity_combined_px_per_mm",
    "sigma_pixel_h10_p95_px",
    "sigma_pixel_h30_p95_px",
    "sigma_z_pred_h10_p95_mm",
    "sigma_z_pred_h30_p95_mm",
    "sigma_z_pred_combined_mm",
    "roi_trim_change_h10",
    "roi_trim_change_h30",
    "candidate_role",
    "candidate_selection_reason",
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text_number(value: Any) -> str:
    number = _number(value)
    return "NaN" if number is None else format(number, ".15g")


def _load_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if len(rows) != 12:
        raise ValueError(f"geometry_master_summary 应有12行，实际 {len(rows)}")
    expected = {row["config_id"] for row in geometry.build_initial_rows()}
    actual = {row.get("config_id") for row in rows}
    if actual != expected:
        raise ValueError(f"geometry matrix config_id 不完整：{sorted(expected - actual)}")
    if next(row for row in rows if row["config_id"] == INVALID_CONFIG)["status"] != "invalid_fov":
        raise ValueError("B00_A05 必须保持 invalid_fov")
    return rows


def _warning_list(value: str) -> list[str]:
    if not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"warnings 不是列表：{value}")
    return [str(item) for item in parsed]


def _config_key(baseline: float, angle: int) -> str:
    prefix = "B00" if baseline == 0 else "B05" if baseline == 5 else "B12p5"
    return f"{prefix}_A{angle:02d}"


def _candidate_role(config_id: str) -> tuple[str, str]:
    if config_id == PRIMARY:
        return (
            "recommended_primary",
            "lowest sigma_z_pred_combined among valid configs; H1 available; primary metrics complete",
        )
    if config_id == BACKUP:
        return (
            "recommended_backup",
            "second-lowest sigma_z_pred_combined with substantially higher sensitivity than the primary",
        )
    if config_id == "B00_A10":
        return (
            "recommended_validation_candidate",
            "high sensitivity and third-lowest sigma_z_pred_combined; retains a diagnostic-only plateau warning",
        )
    if config_id == "B00_A15":
        return (
            "recommended_validation_candidate",
            "clean status with sensitivity and sigma_z close to B00_A10; tests warning-free alternative",
        )
    return "not_selected", "not retained by the hierarchical Phase-A screen"


def _enrich_rows(
    summary_rows: list[dict[str, Any]],
    data_root: Path,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for summary in summary_rows:
        config_id = summary["config_id"]
        baseline = float(summary["baseline_scale_reading"])
        angle = int(summary["laser_angle_deg"])
        status = summary["status"]
        all_warnings = _warning_list(summary.get("warnings", ""))
        h1_status = "invalid_fov"
        h1_valid_fraction: float | None = None
        primary_warnings: list[str] = []
        primary_complete = False
        if status != "invalid_fov":
            analysis_path = data_root / config_id / "analysis" / "multiheight_analysis.json"
            analysis = geometry.load_document(analysis_path)
            rois = analysis.get("rois")
            if not isinstance(rois, dict):
                raise ValueError(f"{config_id} multiheight_analysis 缺少 rois")
            h1 = rois["h001"]
            h1_status = str(h1["status"])
            h1_valid_fraction = _number(h1.get("valid_column_fraction"))
            primary_warnings = list(dict.fromkeys(
                str(warning)
                for roi_id in ("h010", "h030")
                for warning in rois[roi_id].get("warnings", [])
            ))
            primary_complete = all(
                bool(rois[roi_id].get("formal_statistics_allowed"))
                for roi_id in ("h010", "h030")
            )
        phase_a_valid = bool(status not in {"invalid_fov", "failed"} and primary_complete)
        if not phase_a_valid:
            primary_quality = "excluded"
        elif primary_warnings:
            primary_quality = "pass_with_diagnostic_warning"
        else:
            primary_quality = "clean"
        h1_detectability = (
            "invalid_fov" if status == "invalid_fov"
            else "available" if h1_status == "ok"
            else "warning/unavailable"
        )
        candidate_role, candidate_reason = _candidate_role(config_id)
        row: dict[str, Any] = {
            "config_id": config_id,
            "baseline_scale_reading": _text_number(baseline),
            "laser_angle_deg": angle,
            "matrix_x_index": BASELINES.index(baseline),
            "matrix_y_index": ANGLES.index(angle),
            "status": status,
            "phase_a_valid": str(phase_a_valid).lower(),
            "fov_status": "invalid_fov" if status == "invalid_fov" else "captured_in_fov",
            "primary_quality_status": primary_quality,
            "primary_warnings": json.dumps(primary_warnings, ensure_ascii=False),
            "all_warnings": json.dumps(all_warnings, ensure_ascii=False),
            "needs_manual_review": summary["needs_manual_review"],
            "h1_status": h1_status,
            "h1_valid_column_fraction": _text_number(h1_valid_fraction),
            "h1_detectability": h1_detectability,
            "candidate_role": candidate_role,
            "candidate_selection_reason": candidate_reason,
        }
        for field in (
            "reference_cv_interior_rmse_px",
            "reference_cv_interior_p95_px",
            "sensitivity_h1",
            "sensitivity_h10",
            "sensitivity_h30",
            "sensitivity_combined_px_per_mm",
            "sigma_pixel_h10_p95_px",
            "sigma_pixel_h30_p95_px",
            "sigma_z_pred_h10_p95_mm",
            "sigma_z_pred_h30_p95_mm",
            "sigma_z_pred_combined_mm",
            "roi_trim_change_h10",
            "roi_trim_change_h30",
        ):
            row[field] = _text_number(summary.get(field))
        enriched.append(row)
    enriched.sort(key=lambda row: (int(row["matrix_y_index"]), int(row["matrix_x_index"])))
    return enriched


def _matrix(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = np.full((len(ANGLES), len(BASELINES)), np.nan, dtype=np.float64)
    for row in rows:
        values[int(row["matrix_y_index"]), int(row["matrix_x_index"])] = (
            _number(row[field]) if _number(row[field]) is not None else np.nan
        )
    return values


def _save_numeric_heatmap(
    path: Path,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    cmap_name: str,
) -> None:
    values = _matrix(rows, field)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#bdbdbd")
    figure, axis = plt.subplots(figsize=(8.4, 7.2))
    image = axis.imshow(values, cmap=cmap, interpolation="none", aspect="auto")
    axis.set_xticks(range(len(BASELINES)), ["0", "5", "12.5"])
    axis.set_yticks(range(len(ANGLES)), [str(angle) for angle in ANGLES])
    axis.set_xlabel("baseline_scale_reading (mechanical scale, not optical baseline)")
    axis.set_ylabel("laser_angle_deg")
    axis.set_title(title)
    for y, angle in enumerate(ANGLES):
        for x, baseline in enumerate(BASELINES):
            config_id = _config_key(baseline, angle)
            value = values[y, x]
            if np.isfinite(value):
                red, green, blue, _alpha = cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "black" if luminance >= 0.55 else "white"
                label = f"{config_id}\n{value:.5g}"
            else:
                color = "#424242"
                label = f"{config_id}\ninvalid_fov" if config_id == INVALID_CONFIG else f"{config_id}\nNaN"
            axis.text(x, y, label, ha="center", va="center", color=color, fontsize=9)
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_h1_status_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    categories = {"invalid_fov": 0, "warning/unavailable": 1, "available": 2}
    colors = ["#bdbdbd", "#ffb74d", "#66bb6a"]
    values = np.zeros((len(ANGLES), len(BASELINES)), dtype=np.int8)
    labels: dict[tuple[int, int], tuple[str, str]] = {}
    for row in rows:
        y = int(row["matrix_y_index"])
        x = int(row["matrix_x_index"])
        category = str(row["h1_detectability"])
        values[y, x] = categories[category]
        labels[(y, x)] = (row["config_id"], category)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    figure, axis = plt.subplots(figsize=(8.4, 7.2))
    axis.imshow(values, cmap=cmap, norm=norm, interpolation="none", aspect="auto")
    axis.set_xticks(range(len(BASELINES)), ["0", "5", "12.5"])
    axis.set_yticks(range(len(ANGLES)), [str(angle) for angle in ANGLES])
    axis.set_xlabel("baseline_scale_reading (mechanical scale, not optical baseline)")
    axis.set_ylabel("laser_angle_deg")
    axis.set_title("H1 detectability / status (diagnostic only)")
    for (y, x), (config_id, category) in labels.items():
        display = "warning /\nunavailable" if category == "warning/unavailable" else category
        axis.text(x, y, f"{config_id}\n{display}", ha="center", va="center", fontsize=9)
    axis.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, categories)],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_matrix(rows: list[dict[str, Any]], field: str, digits: int = 5) -> list[str]:
    by_id = {row["config_id"]: row for row in rows}
    lines = [
        f"| laser_angle_deg \\ baseline_scale_reading | 0 | 5 | 12.5 |",
        "|---:|---:|---:|---:|",
    ]
    for angle in ANGLES:
        cells: list[str] = []
        for baseline in BASELINES:
            row = by_id[_config_key(baseline, angle)]
            value = _number(row[field])
            cells.append("invalid_fov" if row["status"] == "invalid_fov" else "NaN" if value is None else f"{value:.{digits}f}")
        lines.append(f"| {angle} | " + " | ".join(cells) + " |")
    return lines


def _h1_markdown_matrix(rows: list[dict[str, Any]]) -> list[str]:
    by_id = {row["config_id"]: row for row in rows}
    lines = [
        "| laser_angle_deg \\ baseline_scale_reading | 0 | 5 | 12.5 |",
        "|---:|---|---|---|",
    ]
    for angle in ANGLES:
        cells = [by_id[_config_key(baseline, angle)]["h1_detectability"] for baseline in BASELINES]
        lines.append(f"| {angle} | " + " | ".join(cells) + " |")
    return lines


def _values_for_angles(rows_by_id: dict[str, dict[str, Any]], baseline: float, field: str) -> str:
    values = []
    for angle in ANGLES:
        row = rows_by_id[_config_key(baseline, angle)]
        value = _number(row[field])
        values.append("invalid_fov" if row["status"] == "invalid_fov" else "NaN" if value is None else f"{value:.5f}")
    return " → ".join(values)


def _values_for_baselines(rows_by_id: dict[str, dict[str, Any]], angle: int, field: str) -> str:
    values = []
    for baseline in BASELINES:
        row = rows_by_id[_config_key(baseline, angle)]
        value = _number(row[field])
        values.append("invalid_fov" if row["status"] == "invalid_fov" else "NaN" if value is None else f"{value:.5f}")
    return " → ".join(values)


def _write_analysis_report(path: Path, rows: list[dict[str, Any]]) -> None:
    by_id = {row["config_id"]: row for row in rows}
    lines = [
        "# Phase-A baseline × laser-angle geometry analysis",
        "",
        "## Interpretation boundary",
        "",
        "This report uses the frozen extraction chain and the final `geometry_master_summary.csv`. No Steger, band, reference, ROI, trim, or metric definition is changed. `baseline_scale_reading` is the mechanical support scale, not the measured camera–laser optical baseline.",
        "",
        "`sigma_z_pred_combined_mm` is predicted height repeatability derived from image-space temporal repeatability and geometric sensitivity. It is not final 3D measurement accuracy. Final accuracy must be verified after formal calibration using independent gauge blocks or traceable standards.",
        "",
        "B00_A05 remains `invalid_fov`; every numeric matrix keeps that cell as NaN/gray and no interpolation is performed.",
        "",
        "## 3 × 4 matrices",
        "",
        "### sensitivity_combined_px_per_mm",
        "",
        *_markdown_matrix(rows, "sensitivity_combined_px_per_mm"),
        "",
        "### sigma_z_pred_combined_mm",
        "",
        *_markdown_matrix(rows, "sigma_z_pred_combined_mm"),
        "",
        "### sensitivity_h10",
        "",
        *_markdown_matrix(rows, "sensitivity_h10"),
        "",
        "### sensitivity_h30",
        "",
        *_markdown_matrix(rows, "sensitivity_h30"),
        "",
        "### sigma_pixel_h10_p95_px",
        "",
        *_markdown_matrix(rows, "sigma_pixel_h10_p95_px"),
        "",
        "### sigma_pixel_h30_p95_px",
        "",
        *_markdown_matrix(rows, "sigma_pixel_h30_p95_px"),
        "",
        "### reference_cv_interior_rmse_px",
        "",
        *_markdown_matrix(rows, "reference_cv_interior_rmse_px"),
        "",
        "### H1 detectability/status (diagnostic only)",
        "",
        *_h1_markdown_matrix(rows),
        "",
        "H1 never enters `sensitivity_combined_px_per_mm` or `sigma_z_pred_combined_mm`.",
        "",
        "## Fixed baseline: angle 5 → 10 → 15 → 20",
        "",
    ]
    baseline_narratives = {
        0.0: "Angle 5 is infeasible because the laser leaves the camera FOV. Across the captured 10–20° conditions, sensitivity decreases with angle, while predicted sigma_Z is nearly unchanged from 10° to 15° and improves at 20°. H10 sigma_pixel improves toward 20°; reference CV worsens at 20°. H1 is available for every captured condition.",
        5.0: "Sensitivity decreases monotonically with angle. Predicted sigma_Z is best at 5°, worsens through 15°, then partially recovers at 20°. H10/H30 sigma_pixel are non-monotonic. Reference CV worsens monotonically with angle. H1 remains available throughout.",
        12.5: "Sensitivity decreases monotonically with angle. Predicted sigma_Z is best at 5°, worst at 15°, and partially recovers at 20°. Pixel repeatability is non-monotonic; reference CV worsens monotonically with angle. H1 is available at 5° and 20°, warning at 10°, and unavailable for formal H1 statistics at 15° because of ROI-trim sensitivity.",
    }
    for baseline in BASELINES:
        lines.extend([
            f"### baseline_scale_reading = {baseline:g}",
            "",
            f"- sensitivity combined: {_values_for_angles(by_id, baseline, 'sensitivity_combined_px_per_mm')}",
            f"- sigma_Z predicted combined: {_values_for_angles(by_id, baseline, 'sigma_z_pred_combined_mm')}",
            f"- sigma_pixel H10 P95: {_values_for_angles(by_id, baseline, 'sigma_pixel_h10_p95_px')}",
            f"- sigma_pixel H30 P95: {_values_for_angles(by_id, baseline, 'sigma_pixel_h30_p95_px')}",
            f"- reference CV RMSE: {_values_for_angles(by_id, baseline, 'reference_cv_interior_rmse_px')}",
            "",
            baseline_narratives[baseline],
            "",
        ])
    lines.extend([
        "## Fixed angle: baseline scale 0 → 5 → 12.5",
        "",
    ])
    angle_narratives = {
        5: "The scale-0 condition is invalid_fov. From scale 5 to 12.5, sensitivity decreases and predicted sigma_Z worsens, while reference CV improves slightly. Both feasible H1 observations are available.",
        10: "Sensitivity decreases strongly as the scale increases; predicted sigma_Z worsens. Reference CV improves with scale. H10 sigma_pixel is non-monotonic, H30 sigma_pixel improves, and H1 becomes warning at scale 12.5.",
        15: "Sensitivity decreases and predicted sigma_Z worsens with scale. Reference CV improves. H10/H30 sigma_pixel are non-monotonic; H1 is unavailable at scale 12.5 but remains diagnostic-only.",
        20: "Sensitivity decreases and predicted sigma_Z worsens with scale. Reference CV improves. Pixel repeatability is non-monotonic, with scale 12.5 giving the lowest H30 sigma_pixel P95. H1 remains available for all three baselines.",
    }
    for angle in ANGLES:
        lines.extend([
            f"### laser_angle_deg = {angle}",
            "",
            f"- sensitivity combined: {_values_for_baselines(by_id, angle, 'sensitivity_combined_px_per_mm')}",
            f"- sigma_Z predicted combined: {_values_for_baselines(by_id, angle, 'sigma_z_pred_combined_mm')}",
            f"- sigma_pixel H10 P95: {_values_for_baselines(by_id, angle, 'sigma_pixel_h10_p95_px')}",
            f"- sigma_pixel H30 P95: {_values_for_baselines(by_id, angle, 'sigma_pixel_h30_p95_px')}",
            f"- reference CV RMSE: {_values_for_baselines(by_id, angle, 'reference_cv_interior_rmse_px')}",
            "",
            angle_narratives[angle],
            "",
        ])
    lines.extend([
        "## Cross-factor conclusions",
        "",
        "- At every fixed angle with valid data, increasing `baseline_scale_reading` lowers combined sensitivity and generally worsens predicted sigma_Z, while reference CV improves. This is a measured Phase-A trade-off, not a claim about actual optical baseline because the scale reading is not the measured baseline.",
        "- At fixed baseline 5 and 12.5, sensitivity decreases monotonically as laser angle increases. The scale-0 captured subset shows the same decline from 10° to 20°.",
        "- Pixel repeatability is not monotonic in either factor, so sigma_Z cannot be inferred from sensitivity alone.",
        "- FOV feasibility is categorical in the available data: 11 conditions were captured in view; B00_A05 is invalid_fov. No quantitative FOV-margin measurement exists in the summary, so the report does not invent one.",
        "- All 11 captured configurations have complete H10/H30 formal statistics and `needs_manual_review=false`. Existing primary warnings are diagnostic stable-plateau warnings and do not invalidate trim3-median formal statistics.",
        "",
    ])
    geometry._atomic_write_text(path, "\n".join(lines))


def _write_candidate_report(path: Path, rows: list[dict[str, Any]]) -> None:
    by_id = {row["config_id"]: row for row in rows}

    def metrics(config_id: str) -> str:
        row = by_id[config_id]
        return (
            f"sigma_Z={float(row['sigma_z_pred_combined_mm']):.5f} mm, "
            f"sensitivity={float(row['sensitivity_combined_px_per_mm']):.5f} px/mm, "
            f"reference CV RMSE={float(row['reference_cv_interior_rmse_px']):.5f} px, "
            f"H1={row['h1_detectability']}, quality={row['primary_quality_status']}"
        )

    lines = [
        "# Phase-A hierarchical candidate selection",
        "",
        "No weighted composite score is used.",
        "",
        "## Selection hierarchy",
        "",
        "1. Exclude `invalid_fov` and `failed`. B00_A05 is excluded as invalid_fov; no captured configuration failed.",
        "2. Check H10/H30 formal completeness and primary warnings. All 11 captured configurations are formally complete; diagnostic stable-plateau warnings are retained but are not hard failures.",
        "3. Prefer lower `sigma_z_pred_combined_mm`.",
        "4. When sigma_Z is close, prefer higher `sensitivity_combined_px_per_mm`.",
        "5. Use reference CV, H1 detectability, warning cleanliness, and known FOV feasibility as supporting evidence. Quantitative FOV margin is not available and is not fabricated.",
        "",
        "## Selected candidates",
        "",
        f"### recommended_primary: `{PRIMARY}`",
        "",
        f"- {metrics(PRIMARY)}.",
        "- Lowest predicted sigma_Z among all valid configurations. It remains the primary even though its sensitivity is below the scale-0 candidates, because sigma_Z has priority in the declared hierarchy.",
        "",
        f"### recommended_backup: `{BACKUP}`",
        "",
        f"- {metrics(BACKUP)}.",
        "- Second-lowest predicted sigma_Z and substantially higher sensitivity than the primary. It carries a diagnostic-only H30 stable-plateau warning and belongs to the scale-0 family where the 5° condition was invalid_fov, so next-stage FOV margin must be checked explicitly.",
        "",
        "### recommended_validation_candidates",
        "",
        f"- `{VALIDATION[0]}` — {metrics(VALIDATION[0])}. It provides the highest combined sensitivity and third-lowest predicted sigma_Z, with a diagnostic-only H10 plateau warning.",
        f"- `{VALIDATION[1]}` — {metrics(VALIDATION[1])}. Its sigma_Z is almost identical to B00_A10 while its summary status is clean; retaining both tests whether the small sensitivity advantage at 10° survives formal calibration and independent standards.",
        "",
        "```yaml",
        f"recommended_primary: {PRIMARY}",
        f"recommended_backup: {BACKUP}",
        "recommended_validation_candidates:",
        *(f"  - {config_id}" for config_id in VALIDATION),
        "```",
        "",
        "## Why other configurations are not retained in the first validation set",
        "",
        "- B05_A10/A15/A20 have higher predicted sigma_Z and lower sensitivity than B05_A05; they do not improve the primary two metrics within the same mechanical baseline scale.",
        "- B12p5 conditions provide better reference CV in several comparisons, but their combined sensitivity is lower and predicted sigma_Z is higher. Reference CV is a fifth-layer supporting metric and cannot override both primary metrics without downstream calibration evidence.",
        "- B00_A05 remains invalid_fov and is never interpolated or reconsidered as a candidate.",
        "",
        "## Required next-stage interpretation",
        "",
        "The retained four configurations are candidates, not a final structure decision. `sigma_z_pred_combined_mm` predicts height repeatability from image repeatability and geometric sensitivity; it is not final 3D measurement accuracy. Each retained structure must complete formal camera/laser calibration and then be tested with independent gauge blocks or standards before a final geometry is selected.",
        "",
    ]
    geometry._atomic_write_text(path, "\n".join(lines))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    experiment = project_root / "experiments" / "geometry_baseline_angle"
    results_dir = experiment / "results"
    summary_path = results_dir / "geometry_master_summary.csv"
    data_root = experiment / "data"
    heatmap_dir = results_dir / "phase_a_heatmaps"
    analysis_csv = results_dir / "phase_a_geometry_analysis.csv"
    analysis_md = results_dir / "phase_a_geometry_analysis.md"
    candidates_md = results_dir / "phase_a_candidate_selection.md"

    frozen_paths = [
        summary_path,
        experiment / "configs" / "analysis.yaml",
        experiment / "configs" / "roi_registry.yaml",
        project_root.parent / "calibration" / "config" / "realtime_steger.yaml",
    ]
    for config_id in geometry.CAPTURED_CONFIG_IDS:
        analysis_dir = data_root / config_id / "analysis"
        frozen_paths.extend([
            analysis_dir / "reference_analysis.json",
            analysis_dir / "reference_by_column.csv",
            analysis_dir / "multiheight_analysis.json",
        ])
    hashes_before = {
        str(path): geometry.sha256_file(path) for path in frozen_paths if path.is_file()
    }

    summary_rows = _load_summary(summary_path)
    rows = _enrich_rows(summary_rows, data_root)
    if sum(row["phase_a_valid"] == "true" for row in rows) != 11:
        raise ValueError("Phase-A valid config 数量不是11")
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for field, filename, title, cmap in NUMERIC_HEATMAPS:
        _save_numeric_heatmap(heatmap_dir / filename, rows, field, title, cmap)
    _save_h1_status_heatmap(heatmap_dir / "08_h1_detectability_status.png", rows)
    geometry._write_csv(analysis_csv, OUTPUT_FIELDS, rows)
    _write_analysis_report(analysis_md, rows)
    _write_candidate_report(candidates_md, rows)

    hashes_after = {
        str(path): geometry.sha256_file(path) for path in frozen_paths if path.is_file()
    }
    if hashes_after != hashes_before:
        raise RuntimeError("Phase-A result analysis modified frozen inputs")
    print(f"heatmaps = {heatmap_dir}")
    print(f"analysis csv = {analysis_csv}")
    print(f"analysis report = {analysis_md}")
    print(f"candidate report = {candidates_md}")
    print(f"recommended_primary = {PRIMARY}")
    print(f"recommended_backup = {BACKUP}")
    print(f"recommended_validation_candidates = {list(VALIDATION)}")
    print("weighted score generated = false")
    print("frozen inputs modified = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
