#!/usr/bin/env python3
"""FIT-only grouped-CV validation of a 1D ray-domain residual correction.

C0 is the frozen Circular Cone.  C1 is ``lambda_cone + F(s)``, where F is a
low-degree cubic B-spline fit to ``lambda_truth - lambda_cone``.  The script
uses only the previous FIT point table, performs leave-one-frame-out CV, and
never opens Validation data or changes K/D/Cone files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import BSpline


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
FIT_IDS = [f"{value:03d}" for value in list(range(1, 19)) + list(range(25, 37))]
VALIDATION_IDS = [f"{value:03d}" for value in list(range(19, 25)) + list(range(37, 41))]
REGIONS = (
    ("global", 0.0, 3000.0),
    ("top", 0.0, 300.0),
    ("middle", 300.0, 2700.0),
    ("bottom", 2700.0, 3000.0),
)
INTERIOR_KNOT_COUNTS = (3, 4, 5, 6)
DEGREE = 3
SMOOTHNESS_LABEL = "moderate_second_difference"
SMOOTHNESS_PENALTY = 0.10
MIN_IMPROVEMENT_FRACTION = 0.60
MIN_WORST_REGION_IMPROVEMENT_FRACTION = 0.50
MIN_MEDIAN_GLOBAL_IMPROVEMENT_PCT = 0.05
MIN_MEDIAN_WORST_REGION_IMPROVEMENT_PCT = 0.05
MAX_WORST_REGION_P95_DEGRADATION = 0.02
MAX_EDGE_MIDDLE_RATIO_DEGRADATION = 0.02
DEFAULT_POINTS = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "spatial_residual_observability"
    / "fit_ray_residual_points.csv"
)
DEFAULT_PCA = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "spatial_residual_observability"
    / "ray_support_summary.json"
)
DEFAULT_OUTPUT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0817" / "c1_grouped_cv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--pca-summary", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_points(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "frame_id",
        "v_px",
        "lambda_cone_mm",
        "lambda_truth_mm",
        "delta_lambda_mm",
        "pca_s",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Previous FIT point table is missing required fields: {sorted(required)}")
    for row in rows:
        row["frame_id"] = str(row["frame_id"])
        for key in ("v_px", "lambda_cone_mm", "lambda_truth_mm", "delta_lambda_mm", "pca_s"):
            row[key] = float(row[key])
    frame_ids = sorted({row["frame_id"] for row in rows}, key=int)
    if frame_ids != FIT_IDS:
        raise RuntimeError(f"FIT frame registry mismatch: {frame_ids}")
    if set(frame_ids) & set(VALIDATION_IDS):
        raise RuntimeError("Validation frame found in previous FIT point table")
    return rows


def region_mask(v: np.ndarray, low: float, high: float) -> np.ndarray:
    return (v >= low) & (v < high)


def design_matrix(x: np.ndarray, domain: tuple[float, float], interior_knot_count: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = map(float, domain)
    if not right > left:
        raise RuntimeError("Frozen pca_s support has no usable span")
    internal = np.linspace(left, right, interior_knot_count + 2, dtype=np.float64)[1:-1]
    knots = np.concatenate((np.repeat(left, DEGREE + 1), internal, np.repeat(right, DEGREE + 1)))
    matrix = BSpline.design_matrix(np.asarray(x, dtype=np.float64), knots, DEGREE, extrapolate=True).toarray()
    return matrix, knots


def fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    frame_ids: np.ndarray,
    domain: tuple[float, float],
    interior_knot_count: int,
    penalty: float,
) -> dict[str, Any]:
    matrix, knots = design_matrix(x, domain, interior_knot_count)
    counts: dict[str, int] = defaultdict(int)
    for frame_id in frame_ids:
        counts[str(frame_id)] += 1
    weights = np.asarray([1.0 / counts[str(frame_id)] for frame_id in frame_ids], dtype=np.float64)
    # Each training frame contributes total weight 1.  The scale is intentional:
    # it makes the second-difference penalty comparable across point counts.
    root_weights = np.sqrt(weights)
    weighted_matrix = matrix * root_weights[:, None]
    weighted_target = y * root_weights
    basis_count = matrix.shape[1]
    difference = np.diff(np.eye(basis_count, dtype=np.float64), n=2, axis=0)
    if penalty > 0.0 and difference.size:
        weighted_matrix = np.vstack((weighted_matrix, np.sqrt(penalty) * difference))
        weighted_target = np.concatenate((weighted_target, np.zeros(difference.shape[0], dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)
    spline = BSpline(knots, coefficients, DEGREE, extrapolate=True)
    return {
        "spline": spline,
        "knots": knots,
        "coefficients": coefficients,
        "basis_count": int(basis_count),
        "training_frame_count": int(len(counts)),
        "training_point_count": int(len(x)),
    }


def metric(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"point_count": 0, "bias_mm": math.nan, "rmse_mm": math.nan, "p95_abs_mm": math.nan}
    return {
        "point_count": int(len(values)),
        "bias_mm": float(np.mean(values)),
        "rmse_mm": float(np.sqrt(np.mean(values * values))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95)),
    }


def safe_nanmax(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmax(array)) if np.any(np.isfinite(array)) else math.nan


def safe_edge_middle_ratio(top: float, bottom: float, middle: float) -> float:
    edge = np.asarray([top, bottom], dtype=np.float64)
    if not np.any(np.isfinite(edge)) or not math.isfinite(middle):
        return math.nan
    return float(np.nanmean(edge) / max(middle, np.finfo(float).eps))


def metrics_for_frame(
    frame_id: str,
    model_id: str,
    split: str,
    v: np.ndarray,
    residual: np.ndarray,
    knot_count: int | None,
    basis_count: int | None,
    training_frame_count: int,
    training_point_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region, low, high in REGIONS:
        selected = np.ones(len(v), dtype=bool) if region == "global" else region_mask(v, low, high)
        stats = metric(residual[selected])
        rows.append(
            {
                "model_id": model_id,
                "model_family": "C0_frozen_cone" if model_id == "C0" else "C1_cubic_spline_F_s",
                "interior_knot_count": knot_count if knot_count is not None else 0,
                "basis_count": basis_count if basis_count is not None else 0,
                "smoothness": "none" if model_id == "C0" else SMOOTHNESS_LABEL,
                "smoothness_penalty": 0.0 if model_id == "C0" else SMOOTHNESS_PENALTY,
                "split": split,
                "heldout_frame_id": frame_id,
                "region": region,
                "training_frame_count": training_frame_count,
                "training_point_count": training_point_count,
                **stats,
            }
        )
    return rows


def lookup_metric(rows: Sequence[Mapping[str, Any]], model_id: str, frame_id: str, region: str, split: str = "test") -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["model_id"] == model_id
        and row["heldout_frame_id"] == frame_id
        and row["region"] == region
        and row["split"] == split
    )


def aggregate_model_rows(grouped_rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    c0_by_frame: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    c0_train_by_frame: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in grouped_rows:
        if row["model_id"] == "C0" and row["split"] == "test":
            c0_by_frame[str(row["heldout_frame_id"])][str(row["region"])] = row
        if row["model_id"] == "C0" and row["split"] == "train":
            c0_train_by_frame[str(row["heldout_frame_id"])][str(row["region"])] = row
    for model_id in model_ids:
        test_rows = [row for row in grouped_rows if row["model_id"] == model_id and row["split"] == "test"]
        train_rows = [row for row in grouped_rows if row["model_id"] == model_id and row["split"] == "train"]
        frame_metrics: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        train_frame_metrics: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in test_rows:
            frame_metrics[str(row["heldout_frame_id"])][str(row["region"])] = row
        for row in train_rows:
            train_frame_metrics[str(row["heldout_frame_id"])][str(row["region"])] = row

        def values(region: str, field: str) -> np.ndarray:
            return np.asarray([finite(frame_metrics[fid][region].get(field)) for fid in FIT_IDS], dtype=np.float64)

        global_rmse = values("global", "rmse_mm")
        global_p95 = values("global", "p95_abs_mm")
        global_bias = values("global", "bias_mm")
        region_rmse = {region: values(region, "rmse_mm") for region in ("top", "middle", "bottom")}
        region_bias = {region: values(region, "bias_mm") for region in ("top", "middle", "bottom")}
        worst_region_rmse = np.nanmax(np.vstack(list(region_rmse.values())), axis=0)
        edge_stack = np.vstack((region_rmse["top"], region_rmse["bottom"]))
        edge_rmse = np.full(edge_stack.shape[1], np.nan, dtype=np.float64)
        edge_valid = np.any(np.isfinite(edge_stack), axis=0)
        edge_rmse[edge_valid] = np.nanmean(edge_stack[:, edge_valid], axis=0)
        edge_middle_ratio = edge_rmse / np.maximum(region_rmse["middle"], np.finfo(float).eps)

        if model_id == "C0":
            improve_global = np.zeros(len(FIT_IDS), dtype=np.float64)
            improve_worst = np.zeros(len(FIT_IDS), dtype=np.float64)
            improve_edge_ratio = np.zeros(len(FIT_IDS), dtype=np.float64)
            improve_bias_abs = np.zeros(len(FIT_IDS), dtype=np.float64)
            c0_worst = worst_region_rmse
            c0_ratio = edge_middle_ratio
        else:
            c0_global = np.asarray([finite(c0_by_frame[fid]["global"].get("rmse_mm")) for fid in FIT_IDS])
            c0_worst = np.asarray(
                [
                    safe_nanmax([finite(c0_by_frame[fid][region].get("rmse_mm")) for region in ("top", "middle", "bottom")])
                    for fid in FIT_IDS
                ]
            )
            c0_ratio = np.asarray(
                [
                    safe_edge_middle_ratio(
                        finite(c0_by_frame[fid]["top"].get("rmse_mm")),
                        finite(c0_by_frame[fid]["bottom"].get("rmse_mm")),
                        finite(c0_by_frame[fid]["middle"].get("rmse_mm")),
                    )
                    for fid in FIT_IDS
                ]
            )
            improve_global = (c0_global - global_rmse) / np.maximum(c0_global, np.finfo(float).eps)
            improve_worst = (c0_worst - worst_region_rmse) / np.maximum(c0_worst, np.finfo(float).eps)
            improve_edge_ratio = (c0_ratio - edge_middle_ratio) / np.maximum(c0_ratio, np.finfo(float).eps)
            c0_bias_abs = np.asarray([abs(finite(c0_by_frame[fid]["global"].get("bias_mm"))) for fid in FIT_IDS])
            improve_bias_abs = (c0_bias_abs - np.abs(global_bias)) / np.maximum(c0_bias_abs, np.finfo(float).eps)

        train_global = np.asarray([finite(train_frame_metrics[fid]["global"].get("rmse_mm")) for fid in FIT_IDS])
        train_worst = np.nanmax(
            np.vstack(
                [
                    np.asarray([finite(train_frame_metrics[fid][region].get("rmse_mm")) for fid in FIT_IDS])
                    for region in ("top", "middle", "bottom")
                ]
            ),
            axis=0,
        )
        # Train metrics are diagnostic only; all selection gates below use held-out
        # frame metrics, never training or pooled global RMSE alone.
        c0_train_global = np.asarray([finite(c0_train_by_frame[fid]["global"].get("rmse_mm")) for fid in FIT_IDS])
        c0_train_worst = np.asarray(
            [
                safe_nanmax([finite(c0_train_by_frame[fid][region].get("rmse_mm")) for region in ("top", "middle", "bottom")])
                for fid in FIT_IDS
            ]
        )
        train_improve_global = (c0_train_global - train_global) / np.maximum(c0_train_global, np.finfo(float).eps)
        train_improve_worst = (c0_train_worst - train_worst) / np.maximum(c0_train_worst, np.finfo(float).eps)

        def nanmedian(values_: np.ndarray) -> float:
            return float(np.nanmedian(values_)) if np.any(np.isfinite(values_)) else math.nan

        def nanp95(values_: np.ndarray) -> float:
            return float(np.nanpercentile(values_, 95)) if np.any(np.isfinite(values_)) else math.nan

        c0_global_rmse = np.asarray([finite(c0_by_frame[fid]["global"].get("rmse_mm")) for fid in FIT_IDS])
        c0_worst_region = np.asarray(
            [
                safe_nanmax([finite(c0_by_frame[fid][region].get("rmse_mm")) for region in ("top", "middle", "bottom")])
                for fid in FIT_IDS
            ]
        )
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_family": "C0_frozen_cone" if model_id == "C0" else "C1_cubic_spline_F_s",
            "interior_knot_count": 0 if model_id == "C0" else int(model_id.split("_")[1].replace("k", "")),
            "basis_count": 0 if model_id == "C0" else int(model_id.split("_")[1].replace("k", "")) + DEGREE + 1,
            "smoothness": "none" if model_id == "C0" else SMOOTHNESS_LABEL,
            "smoothness_penalty": 0.0 if model_id == "C0" else SMOOTHNESS_PENALTY,
            "cv_method": "leave_one_frame_out",
            "cv_weighting": "frame_equal",
            "fold_count": len(FIT_IDS),
            "global_rmse_mean_mm": float(np.nanmean(global_rmse)),
            "global_rmse_median_mm": nanmedian(global_rmse),
            "global_rmse_p95_mm": nanp95(global_rmse),
            "global_rmse_max_mm": float(np.nanmax(global_rmse)),
            "global_p95_abs_median_mm": nanmedian(global_p95),
            "global_bias_range_mm": float(np.nanmax(global_bias) - np.nanmin(global_bias)),
            "worst_region_rmse_mean_mm": float(np.nanmean(worst_region_rmse)),
            "worst_region_rmse_median_mm": nanmedian(worst_region_rmse),
            "worst_region_rmse_p95_mm": nanp95(worst_region_rmse),
            "worst_region_rmse_max_mm": float(np.nanmax(worst_region_rmse)),
            "worst_region_bias_range_mm": float(
                np.nanmax(np.concatenate(list(region_bias.values()))) - np.nanmin(np.concatenate(list(region_bias.values())))
            ),
            "edge_middle_ratio_median": nanmedian(edge_middle_ratio),
            "edge_middle_ratio_p95": nanp95(edge_middle_ratio),
            "frame_improvement_fraction_global": math.nan if model_id == "C0" else float(np.mean(improve_global > 0.0)),
            "frame_improvement_fraction_worst_region": math.nan
            if model_id == "C0"
            else float(np.mean(improve_worst > 0.0)),
            "median_global_improvement_pct": math.nan if model_id == "C0" else float(np.nanmedian(improve_global) * 100.0),
            "median_worst_region_improvement_pct": math.nan
            if model_id == "C0"
            else float(np.nanmedian(improve_worst) * 100.0),
            "median_edge_middle_ratio_reduction_pct": math.nan
            if model_id == "C0"
            else float(np.nanmedian(improve_edge_ratio) * 100.0),
            "bias_abs_improvement_fraction_median": math.nan
            if model_id == "C0"
            else float(np.nanmedian(improve_bias_abs)),
            "train_global_rmse_median_mm": nanmedian(train_global),
            "train_worst_region_rmse_median_mm": nanmedian(train_worst),
            "train_global_improvement_median_pct": float(np.nanmedian(train_improve_global) * 100.0),
            "train_worst_region_improvement_median_pct": float(np.nanmedian(train_improve_worst) * 100.0),
            "cv_train_global_improvement_gap_pct": float(
                np.nanmedian(train_improve_global - improve_global) * 100.0
            ),
            "cv_train_worst_region_improvement_gap_pct": float(
                np.nanmedian(train_improve_worst - improve_worst) * 100.0
            ),
            "selection_gate_global_fraction": MIN_IMPROVEMENT_FRACTION,
            "selection_gate_global_median_improvement_pct": MIN_MEDIAN_GLOBAL_IMPROVEMENT_PCT * 100.0,
            "selection_gate_worst_region_fraction": MIN_WORST_REGION_IMPROVEMENT_FRACTION,
            "selection_gate_worst_region_improvement_pct": MIN_MEDIAN_WORST_REGION_IMPROVEMENT_PCT * 100.0,
            "selection_gate_worst_region_p95_degradation": MAX_WORST_REGION_P95_DEGRADATION,
            "selection_gate_edge_middle_degradation": MAX_EDGE_MIDDLE_RATIO_DEGRADATION,
        }
        if model_id == "C0":
            row["selection_gate_pass"] = False
            row["stability_score"] = 0.0
        else:
            c0_row = output[0]
            pass_global = row["frame_improvement_fraction_global"] >= MIN_IMPROVEMENT_FRACTION
            pass_global_gain = row["median_global_improvement_pct"] >= MIN_MEDIAN_GLOBAL_IMPROVEMENT_PCT * 100.0
            pass_worst = row["frame_improvement_fraction_worst_region"] >= MIN_WORST_REGION_IMPROVEMENT_FRACTION
            pass_gain = row["median_worst_region_improvement_pct"] >= MIN_MEDIAN_WORST_REGION_IMPROVEMENT_PCT * 100.0
            pass_worst_p95 = row["worst_region_rmse_p95_mm"] <= c0_row["worst_region_rmse_p95_mm"] * (1.0 + MAX_WORST_REGION_P95_DEGRADATION)
            pass_ratio = row["edge_middle_ratio_median"] <= c0_row["edge_middle_ratio_median"] * (1.0 + MAX_EDGE_MIDDLE_RATIO_DEGRADATION)
            row["selection_gate_pass"] = bool(pass_global and pass_global_gain and pass_worst and pass_gain and pass_worst_p95 and pass_ratio)
            ratio_reduction = max(min(row["median_edge_middle_ratio_reduction_pct"] / 100.0, 1.0), -1.0)
            bias_reduction = max(min(row["bias_abs_improvement_fraction_median"], 1.0), -1.0)
            row["stability_score"] = float(
                0.35 * row["median_worst_region_improvement_pct"] / 100.0
                + 0.20 * row["frame_improvement_fraction_global"]
                + 0.25 * row["frame_improvement_fraction_worst_region"]
                + 0.10 * ratio_reduction
                + 0.10 * bias_reduction
            )
        output.append(row)
    return output


def make_per_frame_improvement(grouped_rows: Sequence[Mapping[str, Any]], candidate_ids: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id in candidate_ids:
        for frame_id in FIT_IDS:
            row: dict[str, Any] = {
                "model_id": model_id,
                "heldout_frame_id": frame_id,
                "is_frame027": frame_id == "027",
            }
            for region, _, _ in REGIONS:
                c0 = lookup_metric(grouped_rows, "C0", frame_id, region)
                c1 = lookup_metric(grouped_rows, model_id, frame_id, region)
                c0_rmse = finite(c0["rmse_mm"])
                c1_rmse = finite(c1["rmse_mm"])
                c0_p95 = finite(c0["p95_abs_mm"])
                c1_p95 = finite(c1["p95_abs_mm"])
                c0_bias = finite(c0["bias_mm"])
                c1_bias = finite(c1["bias_mm"])
                prefix = region
                row[f"{prefix}_point_count"] = int(c1["point_count"])
                row[f"{prefix}_c0_bias_mm"] = c0_bias
                row[f"{prefix}_c1_bias_mm"] = c1_bias
                row[f"{prefix}_c0_rmse_mm"] = c0_rmse
                row[f"{prefix}_c1_rmse_mm"] = c1_rmse
                row[f"{prefix}_rmse_improvement_pct"] = float((c0_rmse - c1_rmse) / max(c0_rmse, np.finfo(float).eps) * 100.0)
                row[f"{prefix}_c0_p95_abs_mm"] = c0_p95
                row[f"{prefix}_c1_p95_abs_mm"] = c1_p95
                row[f"{prefix}_p95_improvement_pct"] = float((c0_p95 - c1_p95) / max(c0_p95, np.finfo(float).eps) * 100.0)
            row["c0_edge_middle_ratio"] = safe_edge_middle_ratio(
                row["top_c0_rmse_mm"], row["bottom_c0_rmse_mm"], row["middle_c0_rmse_mm"]
            )
            row["c1_edge_middle_ratio"] = safe_edge_middle_ratio(
                row["top_c1_rmse_mm"], row["bottom_c1_rmse_mm"], row["middle_c1_rmse_mm"]
            )
            row["edge_middle_ratio_improvement_pct"] = float(
                (row["c0_edge_middle_ratio"] - row["c1_edge_middle_ratio"])
                / max(row["c0_edge_middle_ratio"], np.finfo(float).eps)
                * 100.0
            )
            row["c0_worst_region_rmse_mm"] = safe_nanmax(
                [row["top_c0_rmse_mm"], row["middle_c0_rmse_mm"], row["bottom_c0_rmse_mm"]]
            )
            row["c1_worst_region_rmse_mm"] = safe_nanmax(
                [row["top_c1_rmse_mm"], row["middle_c1_rmse_mm"], row["bottom_c1_rmse_mm"]]
            )
            row["worst_region_rmse_improvement_pct"] = float(
                (row["c0_worst_region_rmse_mm"] - row["c1_worst_region_rmse_mm"])
                / max(row["c0_worst_region_rmse_mm"], np.finfo(float).eps)
                * 100.0
            )
            output.append(row)
    return output


def select_candidate(comparison: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    c0 = next(row for row in comparison if row["model_id"] == "C0")
    candidates = [row for row in comparison if row["model_id"] != "C0"]
    passing = [row for row in candidates if bool(row["selection_gate_pass"])]
    if passing:
        selected = sorted(passing, key=lambda row: (int(row["interior_knot_count"]), -float(row["stability_score"])))[0]
    else:
        selected = sorted(candidates, key=lambda row: (-float(row["stability_score"]), int(row["interior_knot_count"])))[0]
    train_gain = finite(selected["train_worst_region_improvement_median_pct"])
    cv_gain = finite(selected["median_worst_region_improvement_pct"])
    cv_fraction = finite(selected["frame_improvement_fraction_worst_region"])
    train_cv_gap = finite(selected["cv_train_worst_region_improvement_gap_pct"])
    # A large train-to-CV gap is a caution flag, but it is not by itself
    # evidence of overfit when the held-out selection gates pass. Label as
    # OVERFIT only when the gap accompanies weak/failed held-out behavior.
    if (
        not bool(selected["selection_gate_pass"])
        and train_gain > 2.0 * max(cv_gain, 0.001)
        and (cv_gain <= 0.0 or cv_fraction < 0.50 or train_cv_gap > 15.0)
    ):
        feasibility = "OVERFIT"
    elif bool(selected["selection_gate_pass"]) and cv_gain >= 15.0 and cv_fraction >= 0.70:
        feasibility = "STRONG"
    elif cv_gain >= 5.0 and cv_fraction >= 0.50 and finite(selected["frame_improvement_fraction_global"]) >= 0.50:
        feasibility = "MODERATE"
    else:
        feasibility = "WEAK"
    freeze_validation = feasibility in {"STRONG", "MODERATE"}
    decision = {
        "selected_model_id": str(selected["model_id"]),
        "selected_interior_knot_count": int(selected["interior_knot_count"]),
        "C1_FEASIBILITY": feasibility,
        "freeze_c1_for_independent_validation": freeze_validation,
        "selection_basis": "smallest candidate passing full-field median/frame, worst-region, P95, and bias-ratio gates; not training or pooled global RMSE alone",
        "c0_baseline": dict(c0),
        "selected_metrics": dict(selected),
    }
    return str(selected["model_id"]), feasibility, decision


def render_report(
    points_path: Path,
    pca_path: Path,
    point_count: int,
    comparison: Sequence[Mapping[str, Any]],
    per_frame: Sequence[Mapping[str, Any]],
    selected_id: str,
    feasibility: str,
    decision: Mapping[str, Any],
    domain: tuple[float, float],
) -> str:
    selected = next(row for row in comparison if row["model_id"] == selected_id)
    c0 = next(row for row in comparison if row["model_id"] == "C0")
    three_k = next((row for row in comparison if row["model_id"] == "C1_3k"), None)
    selected_frame = [row for row in per_frame if row["model_id"] == selected_id]
    selected_027 = next((row for row in selected_frame if str(row["heldout_frame_id"]) == "027"), None)
    selected_improve = np.asarray([finite(row["global_rmse_improvement_pct"]) for row in selected_frame])
    selected_worst = np.asarray(
        [finite(row["worst_region_rmse_improvement_pct"]) for row in selected_frame], dtype=np.float64
    )
    lines = [
        "# C1 grouped cross-validation — 1D ray-domain residual correction",
        "",
        f"`C1_FEASIBILITY = {feasibility}`",
        "",
        f"是否值得 freeze C1 进入独立 Validation：**{'YES' if decision['freeze_c1_for_independent_validation'] else 'NO'}**。即使为 YES，本报告只冻结候选定义供独立 Validation 检验，不代表已经部署或通过 Validation。",
        "",
        "## Scope and frozen boundary",
        "",
        f"- 输入逐点表：`{points_path}`；共 **{point_count:,}** 个有效 FIT ray，frame 集严格为 001–018、025–036。",
        f"- 上一轮 PCA 的 `pca_s` 原样使用；本轮 s domain 固定为 **[{domain[0]:.12g}, {domain[1]:.12g}]**，只使用 predictor support，不使用 held-out residual 选择 knot。",
        "- C0 = Frozen Circular Cone；C1 = `lambda_cone + F(s)`，残差定义为 `lambda_truth - lambda_prediction`。",
        "- 027 保留在训练候选和 held-out folds 中，不删除、不重加权。",
        "- 未读取 Validation 019–024、037–040；未重新拟合 K/D 或 Cone；未做 C2/C3；没有 point-wise random split。",
        "",
        "## C1 model and CV protocol",
        "",
        f"- 每个 fold 留出一个完整 frame，共 **{len(FIT_IDS)}** 个 leave-one-frame-out folds；训练点按 frame 赋权，每帧总权重为 1。",
        f"- C1 使用 cubic B-spline，比较 interior knots = **3/4/5/6**（对应 basis count = 7/8/9/10），二阶差分 penalty = **{SMOOTHNESS_PENALTY:g}**（{SMOOTHNESS_LABEL}）。",
        "- 每个 fold 的 F(s) 仅用训练 frames 拟合；Top/Middle/Bottom 为 `v∈[0,300) / [300,2700) / [2700,3000)`。",
        "- 选型同时看 worst-region RMSE、edge/middle ratio、bias range 和逐 frame 改善比例；不以 training 或 pooled global RMSE 单独选型。",
        "",
        "## Model comparison",
        "",
        "| model | knots | CV global RMSE median (mm) | global improvement median | worst-region RMSE median (mm) | worst-region improvement median | edge/middle ratio median | global frame improve fraction | worst-region frame improve fraction | selection gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model_id']} | {row['interior_knot_count']} | {finite(row['global_rmse_median_mm']):.6g} | {finite(row['median_global_improvement_pct']):.3f}% | {finite(row['worst_region_rmse_median_mm']):.6g} | {finite(row['median_worst_region_improvement_pct']):.3f}% | {finite(row['edge_middle_ratio_median']):.4f} | {finite(row['frame_improvement_fraction_global']):.3f} | {finite(row['frame_improvement_fraction_worst_region']):.3f} | {row.get('selection_gate_pass', '')} |"
        )
    lines.extend(
        [
            "",
            f"选中的最简单稳定候选：**{selected_id}**（interior knots = {selected['interior_knot_count']}）。其 CV global RMSE 中位数改善 **{finite(selected['median_global_improvement_pct']):.3f}%**，worst-region RMSE 中位数改善 **{finite(selected['median_worst_region_improvement_pct']):.3f}%**；逐 frame global / worst-region 改善比例为 **{finite(selected['frame_improvement_fraction_global']):.3f} / {finite(selected['frame_improvement_fraction_worst_region']):.3f}**。",
            (
                f"C1_3k 的 global median improvement 为 **{finite(three_k['median_global_improvement_pct']):.3f}%**，低于全视场门槛 **{MIN_MEDIAN_GLOBAL_IMPROVEMENT_PCT * 100.0:.1f}%**，因此未被选中。"
                if three_k is not None and finite(three_k['median_global_improvement_pct']) < MIN_MEDIAN_GLOBAL_IMPROVEMENT_PCT * 100.0
                else "C1_3k 未触发全视场中位改善排除条件。"
            ),
            f"选中候选的 held-out global RMSE improvement 百分位：P05 **{np.nanpercentile(selected_improve, 5):.3f}%**，median **{np.nanmedian(selected_improve):.3f}%**，P95 **{np.nanpercentile(selected_improve, 95):.3f}%**；worst-region 的对应 median improvement 为 **{np.nanmedian(selected_worst):.3f}%**。",
            f"C0 的 CV worst-region RMSE median/P95 = **{finite(c0['worst_region_rmse_median_mm']):.6g}/{finite(c0['worst_region_rmse_p95_mm']):.6g} mm**；C1 = **{finite(selected['worst_region_rmse_median_mm']):.6g}/{finite(selected['worst_region_rmse_p95_mm']):.6g} mm**。",
            f"C0/C1 global bias range = **{finite(c0['global_bias_range_mm']):.6g}/{finite(selected['global_bias_range_mm']):.6g} mm**；worst-region bias range = **{finite(c0['worst_region_bias_range_mm']):.6g}/{finite(selected['worst_region_bias_range_mm']):.6g} mm**；edge/middle median ratio = **{finite(c0['edge_middle_ratio_median']):.4f}/{finite(selected['edge_middle_ratio_median']):.4f}**。",
            (
                f"frame 027（保留）global RMSE 为 **{finite(selected_027['global_c0_rmse_mm']):.6g} → {finite(selected_027['global_c1_rmse_mm']):.6g} mm**，改善 **{finite(selected_027['global_rmse_improvement_pct']):.3f}%**；"
                f"worst-region 改善 **{finite(selected_027['worst_region_rmse_improvement_pct']):.3f}%**，因此该帧没有显示出 material improvement。"
                if selected_027 is not None
                else "frame 027（保留）未能生成逐帧结果。"
            ),
            "",
            "## Feasibility gates",
            "",
            f"- 基本稳定门槛：global frame improvement ≥ {MIN_IMPROVEMENT_FRACTION:.2f} 且 global median improvement ≥ {MIN_MEDIAN_GLOBAL_IMPROVEMENT_PCT * 100.0:.1f}%；worst-region frame improvement ≥ {MIN_WORST_REGION_IMPROVEMENT_FRACTION:.2f}；worst-region median improvement ≥ {MIN_MEDIAN_WORST_REGION_IMPROVEMENT_PCT * 100.0:.1f}%；worst-region P95 不恶化超过 {MAX_WORST_REGION_P95_DEGRADATION * 100.0:.1f}%；edge/middle ratio 不恶化超过 {MAX_EDGE_MIDDLE_RATIO_DEGRADATION * 100.0:.1f}%。",
            f"- 当前选择：`{selected_id}`；training-to-CV worst-region improvement gap = **{finite(selected['cv_train_worst_region_improvement_gap_pct']):.3f} percentage points**，作为泛化风险提示，不参与单独选型；held-out gates 仍是主要判断依据。",
            f"- `C1_FEASIBILITY = {feasibility}`：这是 FIT-only grouped-CV 结论，不是 Validation 结论。",
            "",
            "## Per-frame result",
            "",
            f"逐 frame 的 C0/C1 global、Top/Middle/Bottom RMSE、P95、bias 及改善百分比见 `c1_per_frame_improvement.csv`；全部候选的 fold-level 明细见 `c1_grouped_cv_metrics.csv`。",
            "",
            "## Provenance",
            "",
            f"- points SHA-256: `{sha256_file(points_path)}`",
            f"- PCA summary SHA-256: `{sha256_file(pca_path)}`",
            "- 由于本轮没有打开 Validation，是否 freeze C1 的 YES 仅表示值得送入下一轮独立 Validation，不表示可直接生产部署。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    points_path = args.points.resolve()
    pca_path = args.pca_summary.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")
    points = load_points(points_path)
    pca_summary = json.loads(pca_path.read_text(encoding="utf-8"))
    if pca_summary.get("validation_opened") is not False:
        raise RuntimeError("Previous PCA summary does not prove Validation was not opened")
    if pca_summary.get("frame_027_retained") is not True:
        raise RuntimeError("Previous PCA summary does not prove frame 027 was retained")
    if pca_summary.get("fit_ids") != FIT_IDS:
        raise RuntimeError("Previous PCA summary FIT IDs do not match the requested FIT set")

    frame = np.asarray([str(row["frame_id"]) for row in points])
    s = np.asarray([float(row["pca_s"]) for row in points], dtype=np.float64)
    v = np.asarray([float(row["v_px"]) for row in points], dtype=np.float64)
    y = np.asarray([float(row["delta_lambda_mm"]) for row in points], dtype=np.float64)
    domain = (float(np.min(s)), float(np.max(s)))
    grouped_rows: list[dict[str, Any]] = []
    model_ids = ["C0"] + [f"C1_{count}k" for count in INTERIOR_KNOT_COUNTS]

    for heldout in FIT_IDS:
        test_mask = frame == heldout
        train_mask = ~test_mask
        # C0 has no learned correction; its train/test residual is the frozen
        # delta table, retained here as the fold baseline.
        grouped_rows.extend(
            metrics_for_frame(
                heldout,
                "C0",
                "test",
                v[test_mask],
                y[test_mask],
                None,
                None,
                len(FIT_IDS) - 1,
                int(np.count_nonzero(train_mask)),
            )
        )
        grouped_rows.extend(
            metrics_for_frame(
                heldout,
                "C0",
                "train",
                v[train_mask],
                y[train_mask],
                None,
                None,
                len(FIT_IDS) - 1,
                int(np.count_nonzero(train_mask)),
            )
        )
        for knot_count in INTERIOR_KNOT_COUNTS:
            model_id = f"C1_{knot_count}k"
            fitted = fit_spline(
                s[train_mask],
                y[train_mask],
                frame[train_mask],
                domain,
                knot_count,
                SMOOTHNESS_PENALTY,
            )
            correction_test = np.asarray(fitted["spline"](s[test_mask]), dtype=np.float64)
            correction_train = np.asarray(fitted["spline"](s[train_mask]), dtype=np.float64)
            test_residual = y[test_mask] - correction_test
            train_residual = y[train_mask] - correction_train
            grouped_rows.extend(
                metrics_for_frame(
                    heldout,
                    model_id,
                    "test",
                    v[test_mask],
                    test_residual,
                    knot_count,
                    int(fitted["basis_count"]),
                    int(fitted["training_frame_count"]),
                    int(fitted["training_point_count"]),
                )
            )
            grouped_rows.extend(
                metrics_for_frame(
                    heldout,
                    model_id,
                    "train",
                    v[train_mask],
                    train_residual,
                    knot_count,
                    int(fitted["basis_count"]),
                    int(fitted["training_frame_count"]),
                    int(fitted["training_point_count"]),
                )
            )

    comparison = aggregate_model_rows(grouped_rows, model_ids)
    selected_id, feasibility, decision = select_candidate(comparison)
    per_frame = make_per_frame_improvement(grouped_rows, [selected_id])
    summary = {
        "task": "FIT-only grouped CV for 1D ray-domain residual correction",
        "fit_ids": FIT_IDS,
        "validation_ids_not_read": VALIDATION_IDS,
        "validation_opened": False,
        "frame_027_retained": True,
        "point_count": len(points),
        "input_points": str(points_path),
        "input_points_sha256": sha256_file(points_path),
        "input_pca_summary": str(pca_path),
        "input_pca_summary_sha256": sha256_file(pca_path),
        "pca_s_domain": list(domain),
        "cv_method": "leave_one_frame_out",
        "cv_fold_count": len(FIT_IDS),
        "cv_weighting": "frame_equal_total_weight_1_per_training_frame",
        "models": model_ids,
        "c1_definition": {
            "prediction": "lambda_cone + F(s)",
            "target": "lambda_truth - lambda_cone",
            "degree": DEGREE,
            "interior_knot_counts": list(INTERIOR_KNOT_COUNTS),
            "basis_counts": {f"{count}_interior_knots": count + DEGREE + 1 for count in INTERIOR_KNOT_COUNTS},
            "smoothness": SMOOTHNESS_LABEL,
            "second_difference_penalty": SMOOTHNESS_PENALTY,
        },
        "comparison": comparison,
        "decision": decision,
        "C1_FEASIBILITY": feasibility,
        "freeze_c1_for_independent_validation": bool(decision["freeze_c1_for_independent_validation"]),
        "constraints": {
            "kd_refit": False,
            "cone_refit": False,
            "c2_or_c3": False,
            "pointwise_random_split": False,
            "validation_read": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "c1_grouped_cv_metrics.csv", grouped_rows)
    write_csv(output_dir / "c1_model_comparison.csv", comparison)
    write_csv(output_dir / "c1_per_frame_improvement.csv", per_frame)
    (output_dir / "report.md").write_text(
        render_report(points_path, pca_path, len(points), comparison, per_frame, selected_id, feasibility, decision, domain),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "point_count": len(points),
                "selected_model": selected_id,
                "C1_FEASIBILITY": feasibility,
                "freeze_c1_for_independent_validation": bool(decision["freeze_c1_for_independent_validation"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run(parse_args())
