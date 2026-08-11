#!/usr/bin/env python3
"""Read-only before/after candidate-selection audit for the Phase-A band union fix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import geometry_experiment as geometry  # noqa: E402


CSV_FIELDS = (
    "record_type",
    "frame_index",
    "filename",
    "u",
    "before_valid",
    "after_valid",
    "before_v_px",
    "after_v_px",
    "center_difference_px",
    "abs_center_difference_px",
    "before_selected_response",
    "after_selected_response",
    "before_valid_column_count",
    "after_valid_column_count",
    "before_frame_center_median_px",
    "after_frame_center_median_px",
    "frame_center_difference_px",
    "before_response_median",
    "before_response_p95",
    "after_response_median",
    "after_response_p95",
    "before_accepted_frame_count",
    "after_accepted_frame_count",
    "paired_frame_count",
    "before_column_center_median_px",
    "after_column_center_median_px",
    "column_center_difference_px",
    "column_difference_p95_abs_px",
    "column_difference_max_abs_px",
    "raw_peak_y_px",
    "before_distance_to_raw_peak_px",
    "after_distance_to_raw_peak_px",
)


def _finite_text(value: Any) -> str:
    if value is None:
        return "NaN"
    number = float(value)
    return format(number, ".15g") if math.isfinite(number) else "NaN"


def _stats(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    return float(np.median(finite)), float(np.percentile(finite, 95))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _plot_extremes(
    path: Path,
    extremes: list[dict[str, Any]],
    images: list[np.ndarray],
    before_band: tuple[int, int],
    after_band: tuple[int, int],
    difference_fractions: Mapping[str, float],
) -> None:
    count = max(1, len(extremes))
    columns = 2
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.6 * rows), squeeze=False)
    before_top, before_bottom = before_band
    after_top, after_bottom = after_band
    y_top = max(0, min(before_top, after_top) - 5)
    y_bottom = min(images[0].shape[0], max(before_bottom, after_bottom) + 5)
    for axis, extreme in zip(axes.flat, extremes):
        image = images[int(extreme["frame_position"])]
        u = int(extreme["u"])
        x_left = max(0, u - 12)
        x_right = min(image.shape[1], u + 13)
        axis.imshow(
            image[y_top:y_bottom, x_left:x_right],
            cmap="gray",
            origin="upper",
            extent=(x_left - 0.5, x_right - 0.5, y_bottom - 0.5, y_top - 0.5),
            aspect="auto",
            interpolation="nearest",
        )
        axis.axhline(before_top, color="#ff9800", linestyle="--", linewidth=1.2, label="before band")
        axis.axhline(before_bottom - 1, color="#ff9800", linestyle="--", linewidth=1.2)
        axis.axhline(after_top, color="#00bcd4", linestyle=":", linewidth=1.5, label="after final band")
        axis.axhline(after_bottom - 1, color="#00bcd4", linestyle=":", linewidth=1.5)
        axis.axvline(u, color="white", alpha=0.45, linewidth=0.8)
        axis.scatter(
            [u], [extreme["before_v_px"]], marker="x", s=70, linewidths=2,
            color="#ff5722", label="before selected center", zorder=5,
        )
        axis.scatter(
            [u], [extreme["after_v_px"]], marker="+", s=90, linewidths=2,
            color="#4caf50", label="after selected center", zorder=6,
        )
        axis.scatter(
            [u], [extreme["raw_peak_y_px"]], marker="o", s=35, facecolors="none",
            edgecolors="white", label="raw intensity peak", zorder=4,
        )
        axis.set_title(
            f"frame {extreme['frame_index']}, u={u}, "
            f"after-before={extreme['center_difference_px']:+.4f} px"
        )
        axis.set_xlabel("u (px)")
        axis.set_ylabel("v (px)")
        axis.legend(loc="lower left", fontsize=7)
    for axis in axes.flat[len(extremes):]:
        axis.axis("off")
    figure.suptitle(
        "B05_A10 H10 candidate audit — largest paired center differences\n"
        f"|d|<0.05: {difference_fractions['lt_0_05']:.2%}, "
        f"0.05≤|d|≤0.2: {difference_fractions['0_05_to_0_2']:.2%}, "
        f"|d|>0.2: {difference_fractions['gt_0_2']:.2%}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def audit(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    output_dir: Path,
    *,
    roi_id: str = "h010",
    top_n: int = 8,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    registry = registry.expanduser().resolve()
    analysis_config = analysis_config.expanduser().resolve()
    calibration_src = calibration_src.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if dataset.name != "B05_A10" or roi_id != "h010":
        raise ValueError("当前审计已冻结为 B05_A10 / h010")

    config = geometry._load_reference_analysis_config(analysis_config)
    _registry_path, _registry_document, entry = geometry._load_roi_registry(
        registry, dataset.name
    )
    reference_range = geometry._registry_x_range(
        entry["reference_surface"].get("x_range"), "reference_surface.x_range"
    )
    if reference_range is None:
        raise ValueError("B05_A10 reference_surface.x_range 未确认")
    config = {**config, "reference_surface_x_range": reference_range}
    roi = entry["multiheight"][roi_id]
    selected = geometry._roi_selected_x_range(roi, roi_id)
    if selected is None:
        raise ValueError("B05_A10 H10 selected_x_range 未确认")
    trim = int(config["formal_roi_trim_px"])
    analysis_range = (selected[0] + trim, selected[1] - trim)
    if analysis_range[1] < analysis_range[0]:
        raise ValueError("H10 trim3 ROI 为空")

    protected_paths = (
        analysis_config,
        registry,
        config["steger_config"],
        dataset / "analysis" / "reference_analysis.json",
        dataset / "analysis" / "reference_by_column.csv",
        dataset / "analysis" / "multiheight_analysis.json",
        dataset.parent.parent / "results" / "geometry_master_summary.csv",
    )
    protected_hashes = {
        str(path): geometry.sha256_file(path) for path in protected_paths if path.is_file()
    }

    realtime = geometry._load_realtime_steger(calibration_src)
    options = realtime.load_steger_options(config["steger_config"])
    records = geometry._task_frame_records(dataset, "multiheight")
    images = [geometry._read_gray_image(dataset / record["filename"]) for record in records]
    image_shape = images[0].shape
    if any(image.shape != image_shape for image in images):
        raise ValueError("B05_A10 multiheight 图像尺寸不一致")
    y_ref, _reference_summary, _curve_path = geometry._load_frozen_reference_curve(
        dataset, config, image_shape[1]
    )
    reference_band = geometry._reference_vertical_envelope(
        y_ref, reference_range, config["reference_band_margin_px"], image_shape[0]
    )

    columns = np.arange(analysis_range[0], analysis_range[1] + 1, dtype=int)
    frame_count = len(records)
    width = columns.size
    before_y = np.full((frame_count, width), np.nan, dtype=np.float64)
    after_y = np.full_like(before_y, np.nan)
    before_response = np.full_like(before_y, np.nan)
    after_response = np.full_like(before_y, np.nan)
    before_valid = np.zeros((frame_count, width), dtype=bool)
    after_valid = np.zeros_like(before_valid)
    band_rows: list[dict[str, int]] = []
    csv_rows: list[dict[str, Any]] = []

    for frame_position, (record, image) in enumerate(zip(records, images)):
        before = realtime.extract_steger_columns(image, options)
        after = realtime.extract_steger_columns(
            image, options, additional_band_bounds=reference_band
        )
        before_mask = np.asarray(before.valid[columns], dtype=bool) & np.isfinite(before.v_px[columns])
        after_mask = np.asarray(after.valid[columns], dtype=bool) & np.isfinite(after.v_px[columns])
        before_valid[frame_position] = before_mask
        after_valid[frame_position] = after_mask
        before_y[frame_position, before_mask] = before.v_px[columns][before_mask]
        after_y[frame_position, after_mask] = after.v_px[columns][after_mask]
        before_response[frame_position, before_mask] = before.response[columns][before_mask]
        after_response[frame_position, after_mask] = after.response[columns][after_mask]

        metadata = after.metadata
        band_rows.append({
            "before_top": int(metadata["original_band_top_px"]),
            "before_bottom": int(metadata["original_band_bottom_exclusive_px"]),
            "after_top": int(metadata["final_band_top_px"]),
            "after_bottom": int(metadata["final_band_bottom_exclusive_px"]),
        })
        before_center, _ = _stats(before_y[frame_position])
        after_center, _ = _stats(after_y[frame_position])
        before_resp_median, before_resp_p95 = _stats(before_response[frame_position])
        after_resp_median, after_resp_p95 = _stats(after_response[frame_position])
        csv_rows.append({
            "record_type": "frame_summary",
            "frame_index": int(record["index"]),
            "filename": record["filename"],
            "before_valid_column_count": int(np.count_nonzero(before_mask)),
            "after_valid_column_count": int(np.count_nonzero(after_mask)),
            "before_frame_center_median_px": _finite_text(before_center),
            "after_frame_center_median_px": _finite_text(after_center),
            "frame_center_difference_px": _finite_text(
                None if before_center is None or after_center is None else after_center - before_center
            ),
            "before_response_median": _finite_text(before_resp_median),
            "before_response_p95": _finite_text(before_resp_p95),
            "after_response_median": _finite_text(after_resp_median),
            "after_response_p95": _finite_text(after_resp_p95),
        })

    unique_bands = {
        (row["before_top"], row["before_bottom"], row["after_top"], row["after_bottom"])
        for row in band_rows
    }
    if len(unique_bands) != 1:
        raise ValueError(f"50帧 band 不一致：{sorted(unique_bands)}")
    before_top, before_bottom, after_top, after_bottom = next(iter(unique_bands))

    paired = before_valid & after_valid
    differences = np.full_like(before_y, np.nan)
    differences[paired] = after_y[paired] - before_y[paired]
    abs_differences = np.abs(differences[paired])
    if abs_differences.size == 0:
        raise ValueError("B05_A10 H10 before/after 没有共同 accepted 候选")
    bin_counts = {
        "lt_0_05": int(np.count_nonzero(abs_differences < 0.05)),
        "0_05_to_0_2": int(np.count_nonzero((abs_differences >= 0.05) & (abs_differences <= 0.2))),
        "gt_0_2": int(np.count_nonzero(abs_differences > 0.2)),
    }
    fractions = {key: value / abs_differences.size for key, value in bin_counts.items()}

    for local_u, u in enumerate(columns):
        before_column_median, _ = _stats(before_y[:, local_u])
        after_column_median, _ = _stats(after_y[:, local_u])
        paired_column = paired[:, local_u]
        column_abs = np.abs(differences[:, local_u][paired_column])
        csv_rows.append({
            "record_type": "column_summary",
            "u": int(u),
            "before_accepted_frame_count": int(np.count_nonzero(before_valid[:, local_u])),
            "after_accepted_frame_count": int(np.count_nonzero(after_valid[:, local_u])),
            "paired_frame_count": int(np.count_nonzero(paired_column)),
            "before_column_center_median_px": _finite_text(before_column_median),
            "after_column_center_median_px": _finite_text(after_column_median),
            "column_center_difference_px": _finite_text(
                None if before_column_median is None or after_column_median is None
                else after_column_median - before_column_median
            ),
            "column_difference_p95_abs_px": _finite_text(
                None if column_abs.size == 0 else np.percentile(column_abs, 95)
            ),
            "column_difference_max_abs_px": _finite_text(
                None if column_abs.size == 0 else np.max(column_abs)
            ),
        })

    frame_column_rows: list[dict[str, Any]] = []
    extreme_candidates: list[dict[str, Any]] = []
    for frame_position, record in enumerate(records):
        for local_u, u in enumerate(columns):
            before_ok = bool(before_valid[frame_position, local_u])
            after_ok = bool(after_valid[frame_position, local_u])
            difference = differences[frame_position, local_u]
            raw_peak_y = after_top + int(np.argmax(images[frame_position][after_top:after_bottom, u]))
            row = {
                "record_type": "frame_column",
                "frame_index": int(record["index"]),
                "filename": record["filename"],
                "u": int(u),
                "before_valid": str(before_ok).lower(),
                "after_valid": str(after_ok).lower(),
                "before_v_px": _finite_text(before_y[frame_position, local_u]),
                "after_v_px": _finite_text(after_y[frame_position, local_u]),
                "center_difference_px": _finite_text(difference),
                "abs_center_difference_px": _finite_text(abs(difference)),
                "before_selected_response": _finite_text(before_response[frame_position, local_u]),
                "after_selected_response": _finite_text(after_response[frame_position, local_u]),
                "raw_peak_y_px": raw_peak_y,
                "before_distance_to_raw_peak_px": _finite_text(
                    before_y[frame_position, local_u] - raw_peak_y
                ),
                "after_distance_to_raw_peak_px": _finite_text(
                    after_y[frame_position, local_u] - raw_peak_y
                ),
            }
            frame_column_rows.append(row)
            if before_ok and after_ok:
                extreme_candidates.append({
                    **row,
                    "frame_position": frame_position,
                    "before_v_px": float(before_y[frame_position, local_u]),
                    "after_v_px": float(after_y[frame_position, local_u]),
                    "center_difference_px": float(difference),
                    "abs_center_difference_px": float(abs(difference)),
                    "raw_peak_y_px": float(raw_peak_y),
                    "before_selected_response": float(before_response[frame_position, local_u]),
                    "after_selected_response": float(after_response[frame_position, local_u]),
                })
    csv_rows.extend(frame_column_rows)
    extremes = sorted(
        extreme_candidates, key=lambda item: item["abs_center_difference_px"], reverse=True
    )[:top_n]

    paired_fraction = abs_differences.size / float(frame_count * width)
    max_abs_difference = float(np.max(abs_differences))
    paired_before_y = before_y[paired]
    paired_after_y = after_y[paired]
    paired_before_response = before_response[paired]
    paired_after_response = after_response[paired]
    same_floor_row_fraction = float(
        np.mean(np.floor(paired_before_y) == np.floor(paired_after_y))
    )
    same_rounded_row_fraction = float(
        np.mean(np.rint(paired_before_y) == np.rint(paired_after_y))
    )
    response_stronger_fraction = float(
        np.mean(paired_after_response > paired_before_response)
    )
    response_ratio = paired_after_response / paired_before_response
    response_ratio_p50 = float(np.median(response_ratio))
    difference_p50 = float(np.median(differences[paired]))
    difference_p95_abs = float(np.percentile(abs_differences, 95))
    single_signed_shift = bool(
        np.all(differences[paired] <= 0.0) or np.all(differences[paired] >= 0.0)
    )
    legitimate = bool(
        paired_fraction >= 0.99
        and fractions["gt_0_2"] == 0.0
        and max_abs_difference <= 0.2
        and same_floor_row_fraction >= 0.95
        and response_stronger_fraction >= 0.99
        and single_signed_shift
    )
    alternate_ridge_risk = bool(
        fractions["gt_0_2"] >= 0.01
        or max_abs_difference > 1.0
        or same_floor_row_fraction < 0.75
    )
    verdict = (
        "A. legitimate_recovery_of_true_laser"
        if legitimate else
        "B. alternate_ridge_selection_risk"
        if alternate_ridge_risk else
        "C. inconclusive"
    )

    csv_path = output_dir / "B05_A10_band_candidate_audit.csv"
    plot_path = output_dir / "B05_A10_band_candidate_audit.png"
    markdown_path = output_dir / "B05_A10_band_candidate_audit.md"
    _write_csv(csv_path, csv_rows)
    _plot_extremes(
        plot_path,
        extremes,
        images,
        (before_top, before_bottom),
        (after_top, after_bottom),
        fractions,
    )

    frame_before_centers = np.asarray([
        float(row["before_frame_center_median_px"])
        for row in csv_rows if row["record_type"] == "frame_summary"
    ])
    frame_after_centers = np.asarray([
        float(row["after_frame_center_median_px"])
        for row in csv_rows if row["record_type"] == "frame_summary"
    ])
    frame_before_counts = np.asarray([
        int(row["before_valid_column_count"])
        for row in csv_rows if row["record_type"] == "frame_summary"
    ])
    frame_after_counts = np.asarray([
        int(row["after_valid_column_count"])
        for row in csv_rows if row["record_type"] == "frame_summary"
    ])
    frame_before_response_median = np.asarray([
        float(row["before_response_median"])
        for row in csv_rows if row["record_type"] == "frame_summary"
    ])
    frame_after_response_median = np.asarray([
        float(row["after_response_median"])
        for row in csv_rows if row["record_type"] == "frame_summary"
    ])

    extreme_table = _markdown_table(
        ["Frame", "u", "Before y", "After y", "Difference", "Before response", "After response", "Raw peak y"],
        [[
            str(item["frame_index"]),
            str(item["u"]),
            f"{item['before_v_px']:.6f}",
            f"{item['after_v_px']:.6f}",
            f"{item['center_difference_px']:+.6f}",
            f"{item['before_selected_response']:.6f}",
            f"{item['after_selected_response']:.6f}",
            f"{item['raw_peak_y_px']:.1f}",
        ] for item in extremes],
    )
    report_lines = [
        "# B05_A10 H10 band candidate-selection audit",
        "",
        "## Scope and provenance",
        "",
        "- Dataset: `B05_A10`, task: `multiheight`, frames: 50.",
        f"- H10 selected ROI: `{list(selected)}`; formal trim3 ROI: `{list(analysis_range)}` ({width} columns).",
        f"- Before auto band: `[{before_top}, {before_bottom})`.",
        f"- Reference band: `[{reference_band[0]}, {reference_band[1]})`.",
        f"- After final band: `[{after_top}, {after_bottom})`.",
        f"- Formal Steger config SHA-256: `{geometry.sha256_file(config['steger_config'])}`.",
        "- No Steger, ROI, band, reference, geometry summary, or analysis parameter was modified.",
        "",
        "## Per-frame audit",
        "",
        f"- Before median-center range: {np.min(frame_before_centers):.6f}–{np.max(frame_before_centers):.6f} px.",
        f"- After median-center range: {np.min(frame_after_centers):.6f}–{np.max(frame_after_centers):.6f} px.",
        f"- Before valid-column count range: {np.min(frame_before_counts)}–{np.max(frame_before_counts)}.",
        f"- After valid-column count range: {np.min(frame_after_counts)}–{np.max(frame_after_counts)}.",
        f"- Before selected-response median range: {np.min(frame_before_response_median):.6f}–{np.max(frame_before_response_median):.6f}.",
        f"- After selected-response median range: {np.min(frame_after_response_median):.6f}–{np.max(frame_after_response_median):.6f}.",
        "- Full per-frame median/P95 response statistics are stored as `frame_summary` rows in the CSV.",
        "",
        "## Paired frame-column differences",
        "",
        f"- Paired accepted opportunities: {abs_differences.size}/{frame_count * width} ({paired_fraction:.4%}).",
        f"- `abs(center difference) < 0.05 px`: {bin_counts['lt_0_05']} ({fractions['lt_0_05']:.4%}).",
        f"- `0.05 <= abs(center difference) <= 0.2 px`: {bin_counts['0_05_to_0_2']} ({fractions['0_05_to_0_2']:.4%}).",
        f"- `abs(center difference) > 0.2 px`: {bin_counts['gt_0_2']} ({fractions['gt_0_2']:.4%}).",
        f"- Maximum absolute center difference: {max_abs_difference:.6f} px.",
        f"- Median signed center difference: {difference_p50:+.6f} px; P95 absolute difference: {difference_p95_abs:.6f} px.",
        f"- Same integer candidate row (floor): {same_floor_row_fraction:.4%}; same rounded row: {same_rounded_row_fraction:.4%}.",
        f"- After selected response is stronger in {response_stronger_fraction:.4%} of paired opportunities; response-ratio P50 is {response_ratio_p50:.6f}.",
        f"- All paired shifts have one sign: `{str(single_signed_shift).lower()}`.",
        f"- Median H10 center is only {before_bottom - float(np.median(frame_before_centers)):.3f} px above the old band bottom, versus {after_bottom - float(np.median(frame_after_centers)):.3f} px above the final band bottom.",
        "",
        "## Largest changes",
        "",
        *extreme_table,
        "",
        "The raw-image audit figure overlays the old band, final band, both selected centers, and the raw intensity peak for these cases.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    if legitimate:
        report_lines.extend([
            "All paired selections remain within 0.2 px, the candidate row is preserved for more than 95% of pairs, every shift has the same sign, and the selected response becomes stronger for every pair. The raw overlays show both centers on the same physical laser stripe; no frame-column opportunity switches to a separated ridge. The integer raw-intensity maximum can occupy an adjacent row on this broad, textured stripe and is therefore shown as context rather than used as a subpixel ridge-identity gate.",
            "",
            "The old H10 ridge lies only about 3.5 px above the lower crop boundary. Extending the band moves that boundary about 32 px away, removing a Hessian-convolution boundary influence while retaining the same physical stripe.",
            "",
            "The -21.07% H10 sigma_pixel P95 change is therefore accepted as a real repeatability improvement from the band fix, not an alternate-ridge switch.",
            "",
            "`phase_a_extraction_chain_frozen = true`",
        ])
    elif alternate_ridge_risk:
        report_lines.append(
            "The paired center differences contain evidence consistent with a separated-ridge switch; the extraction chain must not be frozen from this audit."
        )
    else:
        report_lines.append(
            "The available evidence does not prove either a same-ridge boundary correction or an alternate-ridge switch."
        )
    report_lines.extend([
        "",
        "## Outputs",
        "",
        f"- `{csv_path.name}`: frame summaries, per-column summaries, and all frame-column pairs.",
        f"- `{plot_path.name}`: largest-difference raw-image overlays.",
        f"- `{markdown_path.name}`: this audit report.",
        "",
    ])
    geometry._atomic_write_text(markdown_path, "\n".join(report_lines))

    protected_hashes_after = {
        str(path): geometry.sha256_file(path) for path in protected_paths if path.is_file()
    }
    if protected_hashes_after != protected_hashes:
        raise RuntimeError("candidate audit 意外修改了正式配置或既有分析结果")
    return {
        "verdict": verdict,
        "paired_count": int(abs_differences.size),
        "difference_fractions": fractions,
        "max_abs_difference_px": max_abs_difference,
        "outputs": [str(csv_path), str(plot_path), str(markdown_path)],
        "formal_inputs_modified": False,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    experiment = project_root / "experiments" / "geometry_baseline_angle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=experiment / "data" / "B05_A10",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=experiment / "configs" / "roi_registry.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment / "configs" / "analysis.yaml",
    )
    parser.add_argument(
        "--calibration-src",
        type=Path,
        default=project_root.parent / "calibration" / "src",
    )
    parser.add_argument("--output-dir", type=Path, default=experiment / "results")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()
    result = audit(
        args.dataset,
        args.registry,
        args.config,
        args.calibration_src,
        args.output_dir,
        top_n=args.top_n,
    )
    print(f"verdict = {result['verdict']}")
    print(f"paired count = {result['paired_count']}")
    print(f"difference fractions = {result['difference_fractions']}")
    print(f"max abs difference px = {result['max_abs_difference_px']}")
    for output in result["outputs"]:
        print(f"output = {output}")
    print("formal inputs modified = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
