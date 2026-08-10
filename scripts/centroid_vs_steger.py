#!/usr/bin/env python3
"""Run the diagnostic-only Phase-A centroid versus formal Steger comparison."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_TOOL_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CALIBRATION_TOOL_ROOT.parent
EXPERIMENT_DIR = CALIBRATION_TOOL_ROOT / "experiments" / "geometry_baseline_angle"
DATA_ROOT = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
ANALYSIS_CONFIG = EXPERIMENT_DIR / "configs" / "analysis.yaml"
ROI_REGISTRY = EXPERIMENT_DIR / "configs" / "roi_registry.yaml"
STEGER_CONFIG = WORKSPACE_ROOT / "calibration" / "config" / "realtime_steger.yaml"
CALIBRATION_SRC = WORKSPACE_ROOT / "calibration" / "src"
MEASUREMENT_TOOL_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"

for import_root in (SCRIPT_DIR, MEASUREMENT_TOOL_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import geometry_experiment as geometry  # noqa: E402
from laser.backends import centroid_backend  # noqa: E402


CONFIG_IDS = ("B12p5_A20", "B05_A15")
ROI_META = (
    ("h001", "H1", 1.0),
    ("h010", "H10", 10.0),
    ("h030", "H30", 30.0),
)
TRIMS = (0, 2, 3, 4)
FORMAL_TRIM = 3
VALID_FRACTION_MIN = 0.8
BACKGROUND_Y_RANGE = (1180, 1280)

# Reuse the project's implemented centroid. The experiment disables segment-level
# rejection/correction so the narrow H1 ROI tests the local centroid itself.
CENTROID_OPTIONS: dict[str, Any] = {
    "background_kernel": 51,
    "min_local_contrast_dn": 20.0,
    "centroid_window_radius": 5,
    "segment_min_columns": 1,
    "continuity_max_column_gap": 2,
    "continuity_max_vertical_jump": 14.0,
    "correction_window": 1,
    "correction_max_shift": 0.0,
    "scan_axis": "column",
}

CSV_FIELDS = (
    "config_id",
    "method",
    "roi_id",
    "roi_label",
    "height_mm",
    "trim_px",
    "x_start",
    "x_end",
    "frame_count",
    "detected_frame_fraction",
    "detected_event_fraction",
    "valid_column_fraction",
    "y_median_px",
    "sigma_pixel_p50_px",
    "sigma_pixel_p95_px",
    "delta_y_median_px",
    "sensitivity_px_per_mm",
    "roi_trim_relative_change_vs_trim0",
    "background_false_detection_rate",
    "center_shift_px",
    "sensitivity_relative_change",
    "sigma_pixel_p95_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_paths() -> list[Path]:
    paths = {STEGER_CONFIG.resolve()}
    for config_id in CONFIG_IDS:
        analysis_dir = DATA_ROOT / config_id / "analysis"
        paths.update(path.resolve() for path in analysis_dir.glob("reference*"))
        multiheight = analysis_dir / "multiheight_analysis.json"
        if multiheight.is_file():
            paths.add(multiheight.resolve())
    paths.update(path.resolve() for path in RESULTS_DIR.glob("geometry_master_summary*"))
    return sorted(path for path in paths if path.is_file())


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def _load_formal_summary(config_id: str) -> dict[str, Any]:
    path = DATA_ROOT / config_id / "analysis" / "multiheight_analysis.json"
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("centre_extractor") != "realtime_steger.extract_steger_columns":
        raise ValueError(f"{config_id} formal summary is not realtime Steger")
    return value


def _centroid_stack(dataset: Path, task_id: str) -> dict[str, Any]:
    records = geometry._task_frame_records(dataset, task_id)
    y_stack: np.ndarray | None = None
    valid_stack: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    for frame_index, record in enumerate(records):
        image = geometry._read_gray_image(dataset / record["filename"])
        if image_shape is None:
            image_shape = image.shape
            y_stack = np.full((len(records), image.shape[1]), np.nan, dtype=np.float64)
            valid_stack = np.zeros((len(records), image.shape[1]), dtype=bool)
        elif image.shape != image_shape:
            raise ValueError(f"inconsistent image shape in {dataset.name}/{task_id}")
        points = centroid_backend(image, CENTROID_OPTIONS)
        if points.size:
            columns = np.rint(points[:, 0]).astype(np.int64)
            inside = (columns >= 0) & (columns < image.shape[1])
            columns = columns[inside]
            values = points[inside, 1]
            assert y_stack is not None and valid_stack is not None
            y_stack[frame_index, columns] = values
            valid_stack[frame_index, columns] = np.isfinite(values)
    assert y_stack is not None and valid_stack is not None and image_shape is not None
    return {
        "records": records,
        "image_shape": image_shape,
        "y_stack": y_stack,
        "valid_stack": valid_stack,
    }


def _steger_stack(dataset: Path, realtime: Any, options: Mapping[str, Any]) -> dict[str, Any]:
    records = geometry._task_frame_records(dataset, "multiheight")
    y_stack: np.ndarray | None = None
    valid_stack: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    for frame_index, record in enumerate(records):
        image = geometry._read_gray_image(dataset / record["filename"])
        if image_shape is None:
            image_shape = image.shape
            y_stack = np.full((len(records), image.shape[1]), np.nan, dtype=np.float64)
            valid_stack = np.zeros((len(records), image.shape[1]), dtype=bool)
        elif image.shape != image_shape:
            raise ValueError(f"inconsistent image shape in {dataset.name}/multiheight")
        extracted = realtime.extract_steger_columns(image, options)
        valid = np.asarray(extracted.valid, dtype=bool) & np.isfinite(extracted.v_px)
        assert y_stack is not None and valid_stack is not None
        y_stack[frame_index, valid] = np.asarray(extracted.v_px)[valid]
        valid_stack[frame_index] = valid
    assert y_stack is not None and valid_stack is not None and image_shape is not None
    return {
        "records": records,
        "image_shape": image_shape,
        "y_stack": y_stack,
        "valid_stack": valid_stack,
    }


def _aggregate(stack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    y_stack = np.asarray(stack["y_stack"], dtype=np.float64)
    valid_stack = np.asarray(stack["valid_stack"], dtype=bool)
    width = y_stack.shape[1]
    valid_count = np.sum(valid_stack, axis=0)
    valid_fraction = valid_count.astype(np.float64) / y_stack.shape[0]
    y_median = np.full(width, np.nan, dtype=np.float64)
    sigma = np.full(width, np.nan, dtype=np.float64)
    for column in np.flatnonzero(valid_count):
        values = y_stack[valid_stack[:, column], column]
        y_median[column] = float(np.median(values))
        sigma[column] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "u": np.arange(width, dtype=np.float64),
        "y_median": y_median,
        "sigma": sigma,
        "valid_count": valid_count,
        "valid_fraction": valid_fraction,
    }


def _centroid_reference(dataset: Path, x_range: tuple[int, int], base_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stack = _centroid_stack(dataset, "reference")
    aggregates = _aggregate(stack)
    config = dict(base_config)
    config["reference_surface_x_range"] = x_range
    surface = geometry._build_surface_reference(aggregates, config)
    return surface, aggregates


def _background_false_detection_rate(dataset: Path, h1_range: tuple[int, int]) -> float:
    records = geometry._task_frame_records(dataset, "multiheight")
    y0, y1 = BACKGROUND_Y_RANGE
    x0, x1 = h1_range
    detected = 0
    total = len(records) * (x1 - x0 + 1)
    for record in records:
        image = geometry._read_gray_image(dataset / record["filename"])
        crop = image[y0 : y1 + 1, :]
        points = centroid_backend(crop, CENTROID_OPTIONS)
        if points.size:
            columns = np.rint(points[:, 0]).astype(np.int64)
            detected += int(np.count_nonzero((columns >= x0) & (columns <= x1)))
    return detected / total


def _formal_trim_value(summary: Mapping[str, Any], roi_id: str, trim: int, field: str) -> float | None:
    rows = summary["rois"][roi_id]["roi_trim_sensitivity"]
    item = next(row for row in rows if int(row["trim_px"]) == trim)
    value = item.get(field)
    return None if value is None else float(value)


def _finite_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def _metrics_row(
    *,
    config_id: str,
    method: str,
    roi_id: str,
    roi_label: str,
    height_mm: float,
    trim: int,
    bounds: tuple[int, int],
    stack: Mapping[str, Any],
    aggregates: Mapping[str, np.ndarray],
    reference_y: np.ndarray | None,
    formal_summary: Mapping[str, Any],
    background_fdr: float | None,
) -> dict[str, Any]:
    x0, x1 = bounds[0] + trim, bounds[1] - trim
    columns = np.arange(x0, x1 + 1, dtype=np.int64)
    y_stack = np.asarray(stack["y_stack"], dtype=np.float64)[:, columns]
    valid_stack = np.asarray(stack["valid_stack"], dtype=bool)[:, columns]
    valid_fraction = np.asarray(aggregates["valid_fraction"])[columns]
    reliable = valid_fraction >= VALID_FRACTION_MIN
    y_by_column = np.asarray(aggregates["y_median"])[columns]
    sigma = np.asarray(aggregates["sigma"])[columns]
    delta: float | None
    sensitivity: float | None
    if method == "steger":
        delta = _formal_trim_value(formal_summary, roi_id, trim, "delta_y_median_px")
        sensitivity = _formal_trim_value(
            formal_summary, roi_id, trim, "sensitivity_median_px_per_mm"
        )
    else:
        assert reference_y is not None
        delta_values = y_by_column - reference_y[columns]
        delta = _finite_median(delta_values[reliable])
        sensitivity = abs(delta) / height_mm if delta is not None else None
    return {
        "config_id": config_id,
        "method": method,
        "roi_id": roi_id,
        "roi_label": roi_label,
        "height_mm": height_mm,
        "trim_px": trim,
        "x_start": x0,
        "x_end": x1,
        "frame_count": int(y_stack.shape[0]),
        "detected_frame_fraction": float(np.mean(np.any(valid_stack, axis=1))),
        "detected_event_fraction": float(np.mean(valid_stack)),
        "valid_column_fraction": float(np.mean(reliable)),
        "y_median_px": _finite_median(y_by_column[reliable]),
        "sigma_pixel_p50_px": _percentile(sigma[reliable], 50.0),
        "sigma_pixel_p95_px": _percentile(sigma[reliable], 95.0),
        "delta_y_median_px": delta,
        "sensitivity_px_per_mm": sensitivity,
        "roi_trim_relative_change_vs_trim0": None,
        "background_false_detection_rate": background_fdr if method == "centroid" else None,
        "center_shift_px": None,
        "sensitivity_relative_change": None,
        "sigma_pixel_p95_ratio": None,
    }


def _ratio_change(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return abs(value - reference) / abs(reference)


def _populate_comparisons(rows: list[dict[str, Any]], aggregates: Mapping[tuple[str, str], Mapping[str, np.ndarray]]) -> None:
    by_key = {
        (row["config_id"], row["method"], row["roi_id"], row["trim_px"]): row
        for row in rows
    }
    for config_id in CONFIG_IDS:
        for roi_id, _label, _height in ROI_META:
            centroid_trim0 = by_key[(config_id, "centroid", roi_id, 0)]["sensitivity_px_per_mm"]
            steger_trim0 = by_key[(config_id, "steger", roi_id, 0)]["sensitivity_px_per_mm"]
            for method, trim0 in (("centroid", centroid_trim0), ("steger", steger_trim0)):
                for trim in TRIMS:
                    row = by_key[(config_id, method, roi_id, trim)]
                    row["roi_trim_relative_change_vs_trim0"] = _ratio_change(
                        row["sensitivity_px_per_mm"], trim0
                    )
            for trim in TRIMS:
                centroid = by_key[(config_id, "centroid", roi_id, trim)]
                steger = by_key[(config_id, "steger", roi_id, trim)]
                columns = np.arange(centroid["x_start"], centroid["x_end"] + 1)
                cagg = aggregates[(config_id, "centroid")]
                sagg = aggregates[(config_id, "steger")]
                common = (
                    (np.asarray(cagg["valid_fraction"])[columns] >= VALID_FRACTION_MIN)
                    & (np.asarray(sagg["valid_fraction"])[columns] >= VALID_FRACTION_MIN)
                )
                shifts = (
                    np.asarray(cagg["y_median"])[columns]
                    - np.asarray(sagg["y_median"])[columns]
                )
                centroid["center_shift_px"] = _finite_median(shifts[common])
                centroid["sensitivity_relative_change"] = _ratio_change(
                    centroid["sensitivity_px_per_mm"], steger["sensitivity_px_per_mm"]
                )
                c_sigma = centroid["sigma_pixel_p95_px"]
                s_sigma = steger["sigma_pixel_p95_px"]
                centroid["sigma_pixel_p95_ratio"] = (
                    c_sigma / s_sigma
                    if c_sigma is not None and s_sigma is not None and s_sigma > 0.0
                    else None
                )


def _formal_recheck(rows: list[dict[str, Any]], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for config_id in CONFIG_IDS:
        for roi_id, _label, _height in ROI_META:
            row = next(
                item
                for item in rows
                if item["config_id"] == config_id
                and item["method"] == "steger"
                and item["roi_id"] == roi_id
                and item["trim_px"] == FORMAL_TRIM
            )
            formal = summaries[config_id]["rois"][roi_id]
            expected_sigma = formal.get("sigma_pixel_p95_px")
            actual_sigma = row["sigma_pixel_p95_px"]
            difference = (
                abs(float(actual_sigma) - float(expected_sigma))
                if actual_sigma is not None and expected_sigma is not None
                else None
            )
            checks.append(
                {
                    "config_id": config_id,
                    "roi_id": roi_id,
                    "rerun_sigma_pixel_p95_px": actual_sigma,
                    "formal_sigma_pixel_p95_px": expected_sigma,
                    "absolute_difference_px": difference,
                    "consistent": difference is None or difference < 1.0e-9,
                }
            )
    return {"checks": checks, "all_consistent": all(item["consistent"] for item in checks)}


def _acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def row(config_id: str, method: str, roi_id: str) -> dict[str, Any]:
        return next(
            item
            for item in rows
            if item["config_id"] == config_id
            and item["method"] == method
            and item["roi_id"] == roi_id
            and item["trim_px"] == FORMAL_TRIM
        )

    h1_values = {
        config_id: row(config_id, "centroid", "h001")["valid_column_fraction"]
        for config_id in CONFIG_IDS
    }
    background_values = {
        config_id: row(config_id, "centroid", "h001")["background_false_detection_rate"]
        for config_id in CONFIG_IDS
    }
    fidelity_rows = [
        row("B12p5_A20", "centroid", roi_id) for roi_id in ("h010", "h030")
    ]
    comparable_rows = [
        row(config_id, "centroid", roi_id)
        for config_id, roi_ids in (("B12p5_A20", ("h001", "h010", "h030")), ("B05_A15", ("h010", "h030")))
        for roi_id in roi_ids
    ]
    criteria = {
        "h1_valid_column_fraction_gte_0_8": {
            "threshold": 0.8,
            "values": h1_values,
            "passed": all(value >= 0.8 for value in h1_values.values()),
        },
        "background_false_detection_rate_lt_0_02": {
            "threshold": 0.02,
            "values": background_values,
            "passed": all(value is not None and value < 0.02 for value in background_values.values()),
        },
        "b12_h10_h30_abs_median_center_shift_lt_0_05_px": {
            "threshold_px": 0.05,
            "values": {item["roi_label"]: item["center_shift_px"] for item in fidelity_rows},
            "passed": all(
                item["center_shift_px"] is not None and abs(item["center_shift_px"]) < 0.05
                for item in fidelity_rows
            ),
        },
        "b12_h10_h30_sensitivity_relative_change_lt_0_02": {
            "threshold": 0.02,
            "values": {
                item["roi_label"]: item["sensitivity_relative_change"] for item in fidelity_rows
            },
            "passed": all(
                item["sensitivity_relative_change"] is not None
                and item["sensitivity_relative_change"] < 0.02
                for item in fidelity_rows
            ),
        },
        "centroid_sigma_pixel_p95_ratio_lte_1_25": {
            "threshold": 1.25,
            "scope": "all formally comparable ROIs: B12 H1/H10/H30 and B05 H10/H30",
            "values": {
                f"{item['config_id']}:{item['roi_label']}": item["sigma_pixel_p95_ratio"]
                for item in comparable_rows
            },
            "passed": all(
                item["sigma_pixel_p95_ratio"] is not None
                and item["sigma_pixel_p95_ratio"] <= 1.25
                for item in comparable_rows
            ),
        },
    }
    all_passed = all(item["passed"] for item in criteria.values())
    h1_recovered = h1_values["B05_A15"] >= 0.8
    background_safe = criteria["background_false_detection_rate_lt_0_02"]["passed"]
    if all_passed:
        verdict = "B. switch_phaseA_to_centroid"
    elif h1_recovered and background_safe:
        verdict = "C. centroid_only_for_diagnostic"
    else:
        verdict = "A. keep_steger"
    return {
        "criteria": criteria,
        "all_passed": all_passed,
        "b05_h1_recovered": h1_recovered,
        "verdict": verdict,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    geometry._atomic_write_text(path, buffer.getvalue())


def _plot(path: Path, rows: list[dict[str, Any]], acceptance: Mapping[str, Any]) -> None:
    formal = [
        row for row in rows if row["method"] == "centroid" and row["trim_px"] == FORMAL_TRIM
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    configs = list(CONFIG_IDS)
    x = np.arange(len(configs))
    width = 0.24
    for index, (roi_id, label, _height) in enumerate(ROI_META):
        values = [
            next(row for row in formal if row["config_id"] == config and row["roi_id"] == roi_id)["valid_column_fraction"]
            for config in configs
        ]
        axes[0, 0].bar(x + (index - 1) * width, values, width, label=label)
    axes[0, 0].axhline(0.8, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_xticks(x, configs)
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_title("Centroid valid-column fraction (trim 3)")
    axes[0, 0].legend()

    fdr = [
        next(row for row in formal if row["config_id"] == config and row["roi_id"] == "h001")["background_false_detection_rate"]
        for config in configs
    ]
    axes[0, 1].bar(configs, fdr, color="#607d8b")
    axes[0, 1].axhline(0.02, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Background false-detection rate")

    b12 = [row for row in formal if row["config_id"] == "B12p5_A20" and row["roi_id"] in ("h010", "h030")]
    labels = [row["roi_label"] for row in b12]
    shifts = [abs(row["center_shift_px"]) for row in b12]
    sensitivity_changes = [100.0 * row["sensitivity_relative_change"] for row in b12]
    bx = np.arange(len(labels))
    left_axis = axes[1, 0]
    right_axis = left_axis.twinx()
    left_axis.bar(bx - 0.18, shifts, 0.36, label="|center shift| px", color="#2196f3")
    right_axis.bar(bx + 0.18, sensitivity_changes, 0.36, label="sensitivity change %", color="#ff9800")
    left_axis.axhline(0.05, color="#2196f3", linestyle="--", linewidth=1)
    right_axis.axhline(2.0, color="#ff9800", linestyle="--", linewidth=1)
    left_axis.set_xticks(bx, labels)
    left_axis.set_title("B12p5_A20 fidelity (trim 3)")
    left_axis.legend(loc="upper left")
    right_axis.legend(loc="upper right")

    comparable = [row for row in formal if row["sigma_pixel_p95_ratio"] is not None]
    comp_labels = [f"{row['config_id']}\n{row['roi_label']}" for row in comparable]
    ratios = [row["sigma_pixel_p95_ratio"] for row in comparable]
    axes[1, 1].bar(comp_labels, ratios, color="#8bc34a")
    axes[1, 1].axhline(1.25, color="red", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Centroid / Steger sigma P95")
    axes[1, 1].tick_params(axis="x", labelsize=8)

    passed = sum(item["passed"] for item in acceptance["criteria"].values())
    fig.suptitle(f"Centroid vs formal Steger — {acceptance['verdict']} ({passed}/5 criteria)")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    geometry._atomic_write_bytes(path, buffer.getvalue())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    protected = _protected_paths()
    before_hashes = _hashes(protected)

    analysis_config = geometry._load_reference_analysis_config(ANALYSIS_CONFIG)
    registry = _load_yaml(ROI_REGISTRY)["configs"]
    steger_document = _load_yaml(STEGER_CONFIG)
    steger_options = steger_document["steger"]
    realtime = geometry._load_realtime_steger(CALIBRATION_SRC)
    summaries = {config_id: _load_formal_summary(config_id) for config_id in CONFIG_IDS}

    rows: list[dict[str, Any]] = []
    aggregate_index: dict[tuple[str, str], Mapping[str, np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    for config_id in CONFIG_IDS:
        print(f"[{config_id}] extracting centroid reference (50 frames)", flush=True)
        dataset = DATA_ROOT / config_id
        entry = registry[config_id]
        reference_range = tuple(int(value) for value in entry["reference_surface"]["x_range"])
        surface, reference_aggregate = _centroid_reference(dataset, reference_range, analysis_config)
        print(f"[{config_id}] extracting centroid multiheight (50 frames)", flush=True)
        centroid_stack = _centroid_stack(dataset, "multiheight")
        centroid_aggregate = _aggregate(centroid_stack)
        print(f"[{config_id}] re-evaluating formal Steger multiheight (50 frames)", flush=True)
        steger_stack = _steger_stack(dataset, realtime, steger_options)
        steger_aggregate = _aggregate(steger_stack)
        aggregate_index[(config_id, "centroid")] = centroid_aggregate
        aggregate_index[(config_id, "steger")] = steger_aggregate
        h1_range = tuple(int(value) for value in entry["multiheight"]["h001"]["selected_x_range"])
        print(f"[{config_id}] measuring background false detections", flush=True)
        background_fdr = _background_false_detection_rate(dataset, h1_range)
        provenance[config_id] = {
            "reference_surface_x_range": reference_range,
            "centroid_reference_model": surface["model_info"],
            "centroid_reference_observed_column_count": int(np.count_nonzero(surface["fit_mask"])),
            "background_roi": [h1_range[0], h1_range[1], *BACKGROUND_Y_RANGE],
            "background_false_detection_rate": background_fdr,
        }
        for roi_id, label, height in ROI_META:
            bounds = tuple(int(value) for value in entry["multiheight"][roi_id]["selected_x_range"])
            for trim in TRIMS:
                rows.append(
                    _metrics_row(
                        config_id=config_id,
                        method="centroid",
                        roi_id=roi_id,
                        roi_label=label,
                        height_mm=height,
                        trim=trim,
                        bounds=bounds,
                        stack=centroid_stack,
                        aggregates=centroid_aggregate,
                        reference_y=np.asarray(surface["y_ref_smooth"]),
                        formal_summary=summaries[config_id],
                        background_fdr=background_fdr,
                    )
                )
                rows.append(
                    _metrics_row(
                        config_id=config_id,
                        method="steger",
                        roi_id=roi_id,
                        roi_label=label,
                        height_mm=height,
                        trim=trim,
                        bounds=bounds,
                        stack=steger_stack,
                        aggregates=steger_aggregate,
                        reference_y=None,
                        formal_summary=summaries[config_id],
                        background_fdr=None,
                    )
                )

    _populate_comparisons(rows, aggregate_index)
    acceptance = _acceptance(rows)
    formal_recheck = _formal_recheck(rows, summaries)
    csv_path = RESULTS_DIR / "centroid_vs_steger.csv"
    png_path = RESULTS_DIR / "centroid_vs_steger.png"
    json_path = RESULTS_DIR / "centroid_vs_steger_summary.json"
    _write_csv(csv_path, rows)
    _plot(png_path, rows, acceptance)

    after_hashes = _hashes(protected)
    if before_hashes != after_hashes:
        changed = sorted(path for path in before_hashes if before_hashes[path] != after_hashes.get(path))
        raise RuntimeError(f"protected Phase-A outputs changed: {changed}")

    summary = {
        "schema_version": 1,
        "diagnostic_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "centroid_vs_steger",
        "configs": list(CONFIG_IDS),
        "frames_per_task": 50,
        "formal_roi_trim_px": FORMAL_TRIM,
        "extractors": {
            "centroid": {
                "implementation": str((MEASUREMENT_TOOL_ROOT / "laser" / "backends.py").resolve()),
                "callable": "laser.backends.centroid_backend",
                "options": CENTROID_OPTIONS,
                "parameter_note": "segment_min_columns=1 and correction_window=1 isolate the implemented local background-subtracted weighted centroid for narrow H1 ROIs",
            },
            "steger": {
                "implementation": str((CALIBRATION_SRC / "realtime_steger.py").resolve()),
                "config": str(STEGER_CONFIG.resolve()),
                "config_sha256": _sha256(STEGER_CONFIG),
                "options": steger_options,
                "delta_and_sensitivity_source": "existing formal multiheight_analysis.json",
            },
        },
        "metric_definitions": {
            "detected_frame_fraction": "fraction of the 50 frames containing at least one detected column in the trimmed ROI",
            "detected_event_fraction": "detected frame-column events divided by 50 times trimmed ROI width",
            "valid_column_fraction": "fraction of trimmed ROI columns detected in at least 80% of frames",
            "y_median_px": "median of per-column 50-frame medians over valid columns",
            "sigma_pixel": "per-column sample standard deviation over detected frames, summarized over valid columns",
            "delta_y_median_px": "median same-column object center minus method-specific smooth reference center",
            "center_shift_px": "median same-column centroid center minus Steger center over columns valid for both",
            "background_false_detection_rate": "centroid detections in a laser-free background ROI divided by 50 times ROI width",
        },
        "provenance": provenance,
        "formal_steger_rerun_check": formal_recheck,
        "acceptance": acceptance,
        "verdict": acceptance["verdict"],
        "protected_phase_a_files": {
            "unchanged": True,
            "sha256_before_and_after": before_hashes,
        },
        "metrics": rows,
        "outputs": [str(csv_path.resolve()), str(json_path.resolve()), str(png_path.resolve())],
    }
    geometry._atomic_write_text(
        json_path, json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"verdict": acceptance["verdict"], "outputs": summary["outputs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
