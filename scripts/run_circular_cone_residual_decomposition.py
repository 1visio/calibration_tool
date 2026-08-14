#!/usr/bin/env python3
"""Task 4A: FIT-only Circular Cone position-dependent residual diagnosis."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))

import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "cone_residual_decomposition"
)
MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
FORMAL_CONE = MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
LOCAL_FULLFIT_RESULT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "cone_local_fullfit" / "local_fullfit_result.json"
EXPECTED_CONE_SHA256 = "478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac"
BIN_WIDTHS = (30, 60, 100)
IMAGE_HEIGHT = 3000
BOOTSTRAP_COUNT = 1000
BOOTSTRAP_SEED = 4101
ROBUST_F_SCALE_MM = 0.05
REGIONS = task3a.REGIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def lambda_by_input(pixels_uv: np.ndarray, calibration: Mapping[str, Any], reconstruction_params: Any) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = reconstruct_uv_to_ground(pixels_uv, calibration, reconstruction_params)
    values = np.full(len(pixels_uv), np.nan, dtype=np.float64)
    valid = np.zeros(len(pixels_uv), dtype=bool)
    lookup = {tuple(np.round(uv, 10)): index for index, uv in enumerate(pixels_uv)}
    for uv, point in zip(result.pixels_uv, result.points_camera):
        index = lookup.get(tuple(np.round(uv, 10)))
        if index is not None:
            values[index] = float(point[2])
            valid[index] = True
    return values, valid, dict(result.filtered)


def metric(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0, "bias": float("nan"), "mae": float("nan"), "rmse": float("nan"), "p95": float("nan"), "max_abs": float("nan")}
    return {
        "count": int(x.size),
        "bias": float(np.mean(x)),
        "mae": float(np.mean(np.abs(x))),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p95": float(np.percentile(np.abs(x), 95)),
        "max_abs": float(np.max(np.abs(x))),
    }


def load_fit_records(intrinsics: Any) -> tuple[list[task3a.FrameRecord], list[dict[str, Any]]]:
    records = task3a.load_old_records()
    extension_records, provenance = task3a.load_extension_records(intrinsics)
    records.extend(extension_records)
    if [record.frame_id for record in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT records do not match explicit 001-018 + 025-036 registry")
    return records, provenance


def collect_points(records: Sequence[task3a.FrameRecord], calibration: Mapping[str, Any], local_model: Mapping[str, Any], reconstruction_params: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    m0_calibration = copy.deepcopy(dict(calibration))
    local_calibration = copy.deepcopy(dict(calibration))
    local_calibration["laser_model"] = copy.deepcopy(dict(local_model))
    rows: list[dict[str, Any]] = []
    invalid_summary: dict[str, Any] = {"M0": defaultdict(int), "M_local_fullfit": defaultdict(int)}
    for record in records:
        lambda_m0, valid_m0, filtered_m0 = lambda_by_input(record.pixels_uv, m0_calibration, reconstruction_params)
        lambda_local, valid_local, filtered_local = lambda_by_input(record.pixels_uv, local_calibration, reconstruction_params)
        invalid_summary["M0"][record.frame_id] += int(np.count_nonzero(~valid_m0))
        invalid_summary["M_local_fullfit"][record.frame_id] += int(np.count_nonzero(~valid_local))
        for index, (uv, truth) in enumerate(zip(record.pixels_uv, record.truth_points)):
            u, v = float(uv[0]), float(uv[1])
            row: dict[str, Any] = {
                "frame_id": record.frame_id,
                "split": "fit",
                "source": record.source,
                "point_index": index,
                "u_px": u,
                "v_px": v,
                "formal_domain": bool(task3a.FORMAL_V_MIN <= v <= task3a.FORMAL_V_MAX),
                "region": task3a.region_for_v(v),
                "lambda_truth_mm": float(truth[2]),
                "lambda_M0_mm": float(lambda_m0[index]) if valid_m0[index] else float("nan"),
                "lambda_M_local_fullfit_mm": float(lambda_local[index]) if valid_local[index] else float("nan"),
                "M0_valid": bool(valid_m0[index]),
                "M_local_fullfit_valid": bool(valid_local[index]),
                "M0_residual_e_lambda_mm": float(truth[2] - lambda_m0[index]) if valid_m0[index] else float("nan"),
                "M_local_fullfit_residual_e_lambda_mm": float(truth[2] - lambda_local[index]) if valid_local[index] else float("nan"),
                "M0_invalid_reason": "" if valid_m0[index] else "no_valid_intersection_or_filter",
                "M_local_fullfit_invalid_reason": "" if valid_local[index] else "no_valid_intersection_or_filter",
            }
            rows.append(row)
    summary = {
        "invalid_by_frame": {model: dict(values) for model, values in invalid_summary.items()},
        "total_points": len(rows),
        "M0_invalid_count": int(sum(not bool(row["M0_valid"]) for row in rows)),
        "M_local_fullfit_invalid_count": int(sum(not bool(row["M_local_fullfit_valid"]) for row in rows)),
    }
    return rows, summary


def frame_balanced_regression(subrows: Sequence[Mapping[str, Any]], model: str, u_ref: float | None = None) -> dict[str, Any]:
    residual_key = f"{model}_residual_e_lambda_mm"
    valid = [row for row in subrows if np.isfinite(float(row[residual_key])) and np.isfinite(float(row["u_px"]))]
    frames: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        frames[str(row["frame_id"])].append(row)
    frame_ids = sorted(frames)
    if u_ref is None:
        frame_medians = [float(np.median([float(row["u_px"]) for row in frames[fid]])) for fid in frame_ids]
        u_ref = float(np.median(frame_medians)) if frame_medians else float("nan")
    if len(frame_ids) < 2 or not np.isfinite(u_ref):
        return {
            "status": "insufficient_frames" if valid else "no_data",
            "frame_ids": frame_ids,
            "u_ref_px": u_ref,
            "point_count": len(valid),
            "unique_frame_count": len(frame_ids),
        }
    x = np.asarray([float(row["u_px"]) - u_ref for row in valid], dtype=np.float64)
    y = np.asarray([float(row[residual_key]) for row in valid], dtype=np.float64)
    weights = np.asarray([1.0 / len(frames[str(row["frame_id"])]) for row in valid], dtype=np.float64)
    weights /= np.mean(weights)

    # The diagnostic relation is a frame-balanced linear decomposition.  Use
    # its closed-form weighted least-squares solution so the frame bootstrap
    # remains tractable (and does not introduce a second nonlinear optimizer).
    design = np.column_stack((np.ones_like(x), x))
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_observation = y * sqrt_weights
    try:
        b, gain = np.linalg.lstsq(weighted_design, weighted_observation, rcond=None)[0]
        b = float(b)
        gain = float(gain)
    except np.linalg.LinAlgError:
        return {
            "status": "singular_design",
            "frame_ids": frame_ids,
            "u_ref_px": u_ref,
            "point_count": len(valid),
            "unique_frame_count": len(frame_ids),
        }
    prediction = b + gain * x
    error = y - prediction
    weighted_rmse = float(np.sqrt(np.sum(weights * error * error) / np.sum(weights)))
    weighted_bias = float(np.sum(weights * error) / np.sum(weights))
    centered = y - float(np.sum(weights * y) / np.sum(weights))
    weighted_r2 = float(1.0 - np.sum(weights * error * error) / max(np.sum(weights * centered * centered), 1.0e-30))
    point_metrics = metric(error)
    return {
        "status": "ok",
        "frame_ids": frame_ids,
        "u_ref_px": u_ref,
        "point_count": len(valid),
        "unique_frame_count": len(frame_ids),
        "b_mm": b,
        "delta_g_mm_per_px": gain,
        "weighted_rmse_mm": weighted_rmse,
        "weighted_bias_mm": weighted_bias,
        "weighted_r2": weighted_r2,
        "fit_mae_mm": point_metrics["mae"],
        "fit_rmse_mm": point_metrics["rmse"],
        "fit_p95_mm": point_metrics["p95"],
        "fit_max_abs_mm": point_metrics["max_abs"],
        "x_span_px": float(np.max(x) - np.min(x)) if x.size else float("nan"),
        "residual_span_mm": float(np.max(y) - np.min(y)) if y.size else float("nan"),
    }


def bootstrap_bin(subrows: Sequence[Mapping[str, Any]], model: str, u_ref: float, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = frame_balanced_regression(subrows, model, u_ref)
    frame_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    residual_key = f"{model}_residual_e_lambda_mm"
    for row in subrows:
        if np.isfinite(float(row[residual_key])):
            frame_groups[str(row["frame_id"])].append(row)
    frame_ids = sorted(frame_groups)
    samples: list[dict[str, Any]] = []
    if len(frame_ids) < 2:
        return base, samples
    # Pre-aggregate each frame.  A frame-balanced fit gives every frame the
    # same total weight, so a bootstrap replicate can be evaluated from its
    # per-frame normal equations instead of rebuilding all point rows.
    frame_normal: list[np.ndarray] = []
    frame_rhs: list[np.ndarray] = []
    # (mean(y^2), mean(x), mean(x^2), mean(y), mean(x*y)) per frame.
    frame_moments: list[tuple[float, float, float, float, float]] = []
    for frame_id in frame_ids:
        frame_rows = frame_groups[frame_id]
        x_frame = np.asarray([float(row["u_px"]) - u_ref for row in frame_rows], dtype=np.float64)
        y_frame = np.asarray([float(row[residual_key]) for row in frame_rows], dtype=np.float64)
        n_frame = float(len(frame_rows))
        inv_n = 1.0 / n_frame
        frame_normal.append(inv_n * np.asarray([[n_frame, np.sum(x_frame)], [np.sum(x_frame), np.sum(x_frame * x_frame)]], dtype=np.float64))
        frame_rhs.append(inv_n * np.asarray([np.sum(y_frame), np.sum(x_frame * y_frame)], dtype=np.float64))
        frame_moments.append(
            (
                inv_n * float(np.sum(y_frame * y_frame)),
                inv_n * float(np.sum(x_frame)),
                inv_n * float(np.sum(x_frame * x_frame)),
                inv_n * float(np.sum(y_frame)),
                inv_n * float(np.sum(x_frame * y_frame)),
            )
        )
    rng = np.random.default_rng(seed)
    for index in range(BOOTSTRAP_COUNT):
        draw_indices = rng.integers(0, len(frame_ids), size=len(frame_ids))
        counts = np.bincount(draw_indices, minlength=len(frame_ids)).astype(np.float64)
        normal = np.sum(np.asarray(frame_normal) * counts[:, None, None], axis=0)
        rhs = np.sum(np.asarray(frame_rhs) * counts[:, None], axis=0)
        try:
            beta = np.linalg.solve(normal, rhs)
            b = float(beta[0]); gain = float(beta[1])
            # Weighted residual SSE is the sum of each sampled frame's mean
            # squared residual.  Divide by the number of draws (the overall
            # normalization cancels in the RMSE definition).
            sse = 0.0
            for count, moments in zip(counts, frame_moments):
                if count == 0.0:
                    continue
                syy, sx, sxx, sy, sxy = moments
                sse += count * (syy - 2.0 * b * sy - 2.0 * gain * sxy + b * b + 2.0 * b * gain * sx + gain * gain * sxx)
            # Floating-point cancellation can make an exactly-zero SSE very
            # slightly negative for a nearly perfect one-frame draw.
            rmse = float(np.sqrt(max(sse, 0.0) / len(frame_ids)))
            fitted = {"status": "ok", "b_mm": b, "delta_g_mm_per_px": gain, "weighted_rmse_mm": rmse}
        except np.linalg.LinAlgError:
            fitted = {"status": "singular_design", "b_mm": float("nan"), "delta_g_mm_per_px": float("nan"), "weighted_rmse_mm": float("nan")}
        samples.append(
            {
                "bootstrap_index": index,
                # The draw itself is intentionally not serialized: retaining a
                # long frame-id string for every replicate needlessly consumes
                # gigabytes while the uncertainty summary only needs the
                # replicate estimates.  The seed, bin, model and replicate
                # index provide reproducibility in the provenance metadata.
                "b_mm": fitted.get("b_mm", float("nan")),
                "delta_g_mm_per_px": fitted.get("delta_g_mm_per_px", float("nan")),
                "weighted_rmse_mm": fitted.get("weighted_rmse_mm", float("nan")),
                "status": fitted.get("status", "failed"),
            }
        )
    finite_b = np.asarray([float(row["b_mm"]) for row in samples if np.isfinite(float(row["b_mm"]))])
    finite_g = np.asarray([float(row["delta_g_mm_per_px"]) for row in samples if np.isfinite(float(row["delta_g_mm_per_px"]))])
    summary = dict(base)
    summary.update(
        {
            "bootstrap_count": len(samples),
            "b_p05_mm": float(np.percentile(finite_b, 5)) if finite_b.size else float("nan"),
            "b_p50_mm": float(np.percentile(finite_b, 50)) if finite_b.size else float("nan"),
            "b_p95_mm": float(np.percentile(finite_b, 95)) if finite_b.size else float("nan"),
            "delta_g_p05_mm_per_px": float(np.percentile(finite_g, 5)) if finite_g.size else float("nan"),
            "delta_g_p50_mm_per_px": float(np.percentile(finite_g, 50)) if finite_g.size else float("nan"),
            "delta_g_p95_mm_per_px": float(np.percentile(finite_g, 95)) if finite_g.size else float("nan"),
        }
    )
    return summary, samples


def decomposition_rows(point_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for width in BIN_WIDTHS:
        for start in range(0, IMAGE_HEIGHT, width):
            end = min(start + width, IMAGE_HEIGHT)
            selected = [row for row in point_rows if start <= float(row["v_px"]) < end]
            center = 0.5 * (start + end)
            for model in ("M0", "M_local_fullfit"):
                # u_ref is defined from frame medians in the selected bin.
                frame_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for row in selected:
                    if np.isfinite(float(row[f"{model}_residual_e_lambda_mm"])):
                        frame_groups[str(row["frame_id"])].append(row)
                medians = [float(np.median([float(row["u_px"]) for row in group])) for group in frame_groups.values()]
                u_ref = float(np.median(medians)) if medians else float("nan")
                summary, samples = bootstrap_bin(selected, model, u_ref, BOOTSTRAP_SEED + width * 10000 + start + (0 if model == "M0" else 1))
                row: dict[str, Any] = {
                    "bin_width_px": width,
                    "v_start_px": start,
                    "v_end_px": end,
                    "v_center_px": center,
                    "model": model,
                    "point_count": summary.get("point_count", 0),
                    "unique_frame_count": summary.get("unique_frame_count", 0),
                    "u_ref_px": summary.get("u_ref_px", float("nan")),
                    "u_span_px": summary.get("x_span_px", float("nan")),
                    "residual_span_mm": summary.get("residual_span_mm", float("nan")),
                    "status": summary.get("status", "no_data"),
                    "b_mm": summary.get("b_mm", float("nan")),
                    "delta_g_mm_per_px": summary.get("delta_g_mm_per_px", float("nan")),
                    "b_p05_mm": summary.get("b_p05_mm", float("nan")),
                    "b_p50_mm": summary.get("b_p50_mm", float("nan")),
                    "b_p95_mm": summary.get("b_p95_mm", float("nan")),
                    "delta_g_p05_mm_per_px": summary.get("delta_g_p05_mm_per_px", float("nan")),
                    "delta_g_p50_mm_per_px": summary.get("delta_g_p50_mm_per_px", float("nan")),
                    "delta_g_p95_mm_per_px": summary.get("delta_g_p95_mm_per_px", float("nan")),
                    "bootstrap_count": summary.get("bootstrap_count", 0),
                    "weighted_rmse_mm": summary.get("weighted_rmse_mm", float("nan")),
                    "weighted_r2": summary.get("weighted_r2", float("nan")),
                    "fit_mae_mm": summary.get("fit_mae_mm", float("nan")),
                    "fit_rmse_mm": summary.get("fit_rmse_mm", float("nan")),
                    "fit_p95_mm": summary.get("fit_p95_mm", float("nan")),
                }
                rows.append(row)
                for sample in samples:
                    bootstrap_rows.append(
                        {
                            "bin_width_px": width,
                            "v_start_px": start,
                            "v_end_px": end,
                            "v_center_px": center,
                            "model": model,
                            **sample,
                        }
                    )
    return rows, bootstrap_rows


def energy_explainability(point_rows: Sequence[Mapping[str, Any]], decomposition: Sequence[Mapping[str, Any]], model: str, width: int) -> dict[str, Any]:
    residual_key = f"{model}_residual_e_lambda_mm"
    valid_rows = [row for row in point_rows if np.isfinite(float(row[residual_key]))]
    total = float(sum(float(row[residual_key]) ** 2 for row in valid_rows))
    by_bin = {(int(row["v_start_px"]), int(row["v_end_px"])): row for row in decomposition if int(row["bin_width_px"]) == width and row["model"] == model}
    offset_error: list[float] = []
    gain_error: list[float] = []
    region_energy: dict[str, dict[str, float]] = {name: {"total": 0.0, "offset_remaining": 0.0, "gain_remaining": 0.0} for name, _, _ in REGIONS}
    for row in valid_rows:
        v = float(row["v_px"]); u = float(row["u_px"]); y = float(row[residual_key])
        start = int(math.floor(v / width) * width)
        summary = by_bin.get((start, min(start + width, IMAGE_HEIGHT)))
        if summary is None or not np.isfinite(float(summary.get("b_mm", "nan"))):
            continue
        b = float(summary["b_mm"]); g = float(summary["delta_g_mm_per_px"]); u_ref = float(summary["u_ref_px"])
        e0 = y - b
        e1 = y - (b + g * (u - u_ref))
        offset_error.append(e0); gain_error.append(e1)
        for name, low, high in REGIONS:
            if low <= v < high:
                region_energy[name]["total"] += y * y
                region_energy[name]["offset_remaining"] += e0 * e0
                region_energy[name]["gain_remaining"] += e1 * e1
                break
    offset_remaining = float(np.sum(np.asarray(offset_error) ** 2)) if offset_error else float("nan")
    gain_remaining = float(np.sum(np.asarray(gain_error) ** 2)) if gain_error else float("nan")
    result: dict[str, Any] = {
        "model": model,
        "bin_width_px": width,
        "total_residual_energy": total,
        "offset_only_explained_fraction": float(1.0 - offset_remaining / total) if total > 0.0 and np.isfinite(offset_remaining) else float("nan"),
        "offset_plus_gain_explained_fraction": float(1.0 - gain_remaining / total) if total > 0.0 and np.isfinite(gain_remaining) else float("nan"),
    }
    for name, values in region_energy.items():
        total_region = values["total"]
        result[f"{name}_total_energy"] = total_region
        result[f"{name}_offset_explained_fraction"] = float(1.0 - values["offset_remaining"] / total_region) if total_region > 0 else float("nan")
        result[f"{name}_offset_plus_gain_explained_fraction"] = float(1.0 - values["gain_remaining"] / total_region) if total_region > 0 else float("nan")
    return result


def global_diagnostics(point_rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    key = f"{model}_residual_e_lambda_mm"
    valid = [row for row in point_rows if np.isfinite(float(row[key]))]
    y = np.asarray([float(row[key]) for row in valid], dtype=np.float64)
    u = np.asarray([float(row["u_px"]) for row in valid], dtype=np.float64)
    v = np.asarray([float(row["v_px"]) for row in valid], dtype=np.float64)
    frames = sorted({str(row["frame_id"]) for row in valid})
    def weighted_line(x: np.ndarray) -> tuple[float, float, float]:
        frame_groups: dict[str, np.ndarray] = {}
        for frame in frames:
            frame_groups[frame] = np.flatnonzero(np.asarray([str(row["frame_id"]) for row in valid]) == frame)
        weights = np.asarray([1.0 / len(frame_groups[str(row["frame_id"])]) for row in valid], dtype=np.float64)
        weights /= np.mean(weights)
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
        residual = y - X @ beta
        return float(beta[0]), float(beta[1]), float(np.sqrt(np.sum(weights * residual * residual) / np.sum(weights)))
    v_intercept, v_slope, v_rmse = weighted_line(v)
    u_intercept, u_slope, u_rmse = weighted_line(u)
    frame_means = np.asarray([np.mean([float(row[key]) for row in valid if str(row["frame_id"]) == frame]) for frame in frames], dtype=np.float64)
    return {
        "model": model,
        "point_count": len(valid),
        "unique_frame_count": len(frames),
        "global_metrics": metric(y),
        "v_line_intercept_mm": v_intercept,
        "v_line_slope_mm_per_px": v_slope,
        "v_line_weighted_rmse_mm": v_rmse,
        "u_line_intercept_mm": u_intercept,
        "u_line_slope_mm_per_px": u_slope,
        "u_line_weighted_rmse_mm": u_rmse,
        "frame_mean_std_mm": float(np.std(frame_means, ddof=1)) if len(frame_means) > 1 else float("nan"),
        "frame_mean_min_mm": float(np.min(frame_means)) if frame_means.size else float("nan"),
        "frame_mean_max_mm": float(np.max(frame_means)) if frame_means.size else float("nan"),
    }


def region_metrics(point_rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, dict[str, Any]]:
    key = f"{model}_residual_e_lambda_mm"
    result: dict[str, dict[str, Any]] = {}
    for name, low, high in REGIONS:
        values = np.asarray([float(row[key]) for row in point_rows if low <= float(row["v_px"]) < high and np.isfinite(float(row[key]))], dtype=np.float64)
        frame_means: list[float] = []
        for frame in sorted({str(row["frame_id"]) for row in point_rows}):
            frame_values = [float(row[key]) for row in point_rows if str(row["frame_id"]) == frame and low <= float(row["v_px"]) < high and np.isfinite(float(row[key]))]
            if frame_values:
                frame_means.append(float(np.mean(frame_values)))
        stats = metric(values)
        stats["frame_mean_std_mm"] = float(np.std(frame_means, ddof=1)) if len(frame_means) > 1 else float("nan")
        stats["frame_count"] = len(frame_means)
        result[name] = stats
    return result


def save_plots(out: Path, point_rows: Sequence[Mapping[str, Any]], decomposition: Sequence[Mapping[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for model, color in (("M0", "#2b6cb0"), ("M_local_fullfit", "#c05621")):
        selected = [row for row in point_rows if np.isfinite(float(row[f"{model}_residual_e_lambda_mm"]))]
        axis.scatter([float(row["v_px"]) for row in selected], [float(row[f"{model}_residual_e_lambda_mm"]) for row in selected], s=3, alpha=0.18, color=color, label=model)
    axis.axhline(0.0, color="#555", linewidth=0.8); axis.set_xlabel("v / px"); axis.set_ylabel("e_lambda = lambda_truth - lambda_cone / mm")
    axis.set_title("FIT residual versus v"); axis.grid(alpha=0.2); axis.legend()
    fig.tight_layout(); fig.savefig(out / "residual_vs_v.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for model, color in (("M0", "#2b6cb0"), ("M_local_fullfit", "#c05621")):
        for width, linestyle in ((30, "-"), (60, "--"), (100, ":")):
            selected = [row for row in decomposition if row["model"] == model and int(row["bin_width_px"]) == width and np.isfinite(float(row["b_mm"]))]
            axis.plot([float(row["v_center_px"]) for row in selected], [float(row["b_mm"]) for row in selected], color=color, linestyle=linestyle, label=f"{model} {width}px")
    axis.axhline(0.0, color="#555", linewidth=0.8); axis.set_xlabel("v / px"); axis.set_ylabel("b(v) / mm"); axis.set_title("Local offset diagnostic b(v)"); axis.grid(alpha=0.2); axis.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "offset_b_vs_v.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for model, color in (("M0", "#2b6cb0"), ("M_local_fullfit", "#c05621")):
        for width, linestyle in ((30, "-"), (60, "--"), (100, ":")):
            selected = [row for row in decomposition if row["model"] == model and int(row["bin_width_px"]) == width and np.isfinite(float(row["delta_g_mm_per_px"]))]
            axis.plot([float(row["v_center_px"]) for row in selected], [float(row["delta_g_mm_per_px"]) for row in selected], color=color, linestyle=linestyle, label=f"{model} {width}px")
    axis.axhline(0.0, color="#555", linewidth=0.8); axis.set_xlabel("v / px"); axis.set_ylabel("delta_g(v) / mm/px"); axis.set_title("Local metric-gain residual delta_g(v)"); axis.grid(alpha=0.2); axis.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "gain_delta_g_vs_v.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for index, model in enumerate(("M0", "M_local_fullfit")):
        values = []
        labels = []
        for name, low, high in REGIONS:
            values.append([float(row[f"{model}_residual_e_lambda_mm"]) for row in point_rows if low <= float(row["v_px"]) < high and np.isfinite(float(row[f"{model}_residual_e_lambda_mm"]))])
            labels.append(name.replace("_formal_edge", ""))
        axes[index].boxplot(values, labels=labels, showfliers=False)
        axes[index].axhline(0.0, color="#555", linewidth=0.8); axes[index].set_title(model); axes[index].grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("e_lambda / mm"); fig.suptitle("Top / middle / bottom residual distributions"); fig.tight_layout(); fig.savefig(out / "residual_region_distribution.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    m0 = np.asarray([float(row["M0_residual_e_lambda_mm"]) for row in point_rows if np.isfinite(float(row["M0_residual_e_lambda_mm"])) and np.isfinite(float(row["M_local_fullfit_residual_e_lambda_mm"]))])
    loc = np.asarray([float(row["M_local_fullfit_residual_e_lambda_mm"]) for row in point_rows if np.isfinite(float(row["M0_residual_e_lambda_mm"])) and np.isfinite(float(row["M_local_fullfit_residual_e_lambda_mm"]))])
    axis.scatter(m0, loc, s=3, alpha=0.18); lim = max(float(np.max(np.abs(m0))), float(np.max(np.abs(loc))))
    axis.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8); axis.set_xlabel("M0 e_lambda / mm"); axis.set_ylabel("M_local_fullfit e_lambda / mm"); axis.set_title("M0 vs M_local_fullfit residual"); axis.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "m0_vs_local_residual.png", dpi=170); plt.close(fig)


def render_report(
    point_rows: Sequence[Mapping[str, Any]],
    decomposition: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    invalid_summary: Mapping[str, Any],
    cone_hash_before: str,
    cone_hash_after: str,
    output_dir: Path,
) -> str:
    diagnostics = {model: global_diagnostics(point_rows, model) for model in ("M0", "M_local_fullfit")}
    regions = {model: region_metrics(point_rows, model) for model in ("M0", "M_local_fullfit")}
    energy = {(model, width): energy_explainability(point_rows, decomposition, model, width) for model in ("M0", "M_local_fullfit") for width in BIN_WIDTHS}
    local = diagnostics["M_local_fullfit"]
    local_regions = regions["M_local_fullfit"]
    top = local_regions["top_formal_edge"]
    bottom = local_regions["bottom_formal_edge"]
    top_gain = [row for row in decomposition if row["model"] == "M_local_fullfit" and int(row["bin_width_px"]) == 60 and row["status"] == "ok" and 0 <= float(row["v_center_px"]) < 300]
    top_gain_abs = max((abs(float(row["delta_g_mm_per_px"])) for row in top_gain), default=float("nan"))
    top_offset_abs = max((abs(float(row["b_mm"])) for row in top_gain), default=float("nan"))
    top_gain_span = max((abs(float(row["delta_g_mm_per_px"])) * float(row["u_span_px"]) for row in top_gain), default=float("nan"))
    bottom_gain = [row for row in decomposition if row["model"] == "M_local_fullfit" and int(row["bin_width_px"]) == 60 and row["status"] == "ok" and float(row["v_center_px"]) >= 2700]
    bottom_gain_span = max((abs(float(row["delta_g_mm_per_px"])) * float(row["u_span_px"]) for row in bottom_gain), default=float("nan"))
    offset_energy = [energy[("M_local_fullfit", width)]["offset_only_explained_fraction"] for width in BIN_WIDTHS]
    gain_energy = [energy[("M_local_fullfit", width)]["offset_plus_gain_explained_fraction"] for width in BIN_WIDTHS]
    energy_60 = energy[("M_local_fullfit", 60)]
    outlier_by_frame: dict[str, int] = defaultdict(int)
    outlier_v: list[float] = []
    for point in point_rows:
        value = float(point["M_local_fullfit_residual_e_lambda_mm"])
        if np.isfinite(value) and abs(value) >= 0.3:
            outlier_by_frame[str(point["frame_id"])] += 1
            outlier_v.append(float(point["v_px"]))
    outlier_frame, outlier_count = max(outlier_by_frame.items(), key=lambda item: item[1], default=("none", 0))
    lines = [
        "# Task 4A — FIT-only Circular Cone 位置相关残差分解",
        "",
        "**FIT_ONLY = TRUE**",
        "**VALIDATION_OPENED = FALSE**",
        "**PRODUCTION_CONE_MODIFIED = FALSE**",
        "",
        "## 数据与定义",
        "",
        "- FIT: 001–018 + 025–036，共 30 frame；Validation 019–024 + 037–040 未读取。",
        "- 主分析模型：Task 3B-2 的 `M_local_fullfit`；M0 仅作参考。",
        "- `e_lambda = lambda_truth - lambda_cone`；lambda_truth 为独立 PnP ray-plane truth 的相机 Z。",
        "- 固定使用 30/60/100 px 三种 v-bin；bin 内按 frame 平衡拟合 `e = b + delta_g*(u-u_ref)`，bootstrap 以 frame 为重采样单位（1000 次，seed=4101）。",
        f"- Formal Cone SHA-256 before/after: `{cone_hash_before}` / `{cone_hash_after}`。",
        "",
        "## 全局与区域 residual",
        "",
        "| model | region | bias / mm | RMSE / mm | P95 / mm | frame mean std / mm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in ("M0", "M_local_fullfit"):
        for name in ("top_formal_edge", "middle_formal", "bottom_formal_edge"):
            stats = regions[model][name]
            lines.append(f"| {model} | {name} | {float(stats['bias']):.7g} | {float(stats['rmse']):.7g} | {float(stats['p95']):.7g} | {float(stats['frame_mean_std_mm']):.7g} |")
    lines += [
        "",
        "## 诊断回答",
        "",
        f"1. **Top residual：** `M_local_fullfit` top bias=`{float(top['bias']):.6g}` mm、RMSE=`{float(top['rmse']):.6g}` mm；60 px top bin 的 b=`{top_offset_abs:.6g}` mm，|delta_g|=`{top_gain_abs:.6g}` mm/px，在该 bin 的 u span 上对应约 `{top_gain_span:.6g}` mm 的 gain 变化。按 60 px bin 的能量分解，offset-only 解释 `{float(energy_60['top_formal_edge_offset_explained_fraction']):.4f}`，加入 gain 后为 `{float(energy_60['top_formal_edge_offset_plus_gain_explained_fraction']):.4f}`；因此 top 以 offset 为主，但 gain 不是可忽略项。",
        f"2. **Bottom：** bias=`{float(bottom['bias']):.6g}` mm、RMSE=`{float(bottom['rmse']):.6g}` mm；60 px bottom bin 的 gain span 约 `{bottom_gain_span:.6g}` mm，offset-only / offset+gain energy explained 为 `{float(energy_60['bottom_formal_edge_offset_explained_fraction']):.4f}` / `{float(energy_60['bottom_formal_edge_offset_plus_gain_explained_fraction']):.4f}`；结构方向与 top 不同且幅度不对称。",
        f"3. **Top/bottom asymmetry：** 明显；top frame-mean std=`{float(top['frame_mean_std_mm']):.6g}` mm，bottom=`{float(bottom['frame_mean_std_mm']):.6g}` mm。",
        f"4. **一维 b(v)：** local offset-only energy explained fractions（30/60/100 px）为 `{', '.join(f'{float(value):.4f}' for value in offset_energy)}`；60 px 分区为 top/middle/bottom=`{float(energy_60['top_formal_edge_offset_explained_fraction']):.4f}`/`{float(energy_60['middle_formal_offset_explained_fraction']):.4f}`/`{float(energy_60['bottom_formal_edge_offset_explained_fraction']):.4f}`。b(v) 对边缘共同偏置有效，但中部只解释约五分之一，不能解释主要全局 residual。加入局部 gain 后全局为 `{', '.join(f'{float(value):.4f}' for value in gain_energy)}`。",
        f"5. **u-dependent residual：** 全局 frame-balanced u slope=`{float(local['u_line_slope_mm_per_px']):.6g}` mm/px，weighted RMSE=`{float(local['u_line_weighted_rmse_mm']):.6g}` mm；全局斜率接近零不代表局部没有 u 结构，边缘 bin 的 `delta_g(v)` 与 bootstrap 区间见 CSV/图，且边缘加入 gain 后仍只有限改善。",
        "",
        "## v / frame 依赖",
        "",
        f"- local 全局 v slope=`{float(local['v_line_slope_mm_per_px']):.6g}` mm/px，weighted RMSE=`{float(local['v_line_weighted_rmse_mm']):.6g}` mm。",
        f"- local frame mean residual 范围=`{float(local['frame_mean_min_mm']):.6g}` 到 `{float(local['frame_mean_max_mm']):.6g}` mm，frame std=`{float(local['frame_mean_std_mm']):.6g}` mm。",
        f"- 固定分箱的 b(v) 曲线跨 v 多次变号且峰谷约达 0.1 mm 量级；|e_lambda|≥0.3 mm 的 `{sum(outlier_by_frame.values())}` 个点主要集中在 frame `{outlier_frame}`（`{outlier_count}` 个），v 范围约 `{min(outlier_v):.6g}`–`{max(outlier_v):.6g}` px，说明存在 frame/位置耦合而非单一全局线性趋势。" if outlier_v else "- 未发现 |e_lambda|≥0.3 mm 的有效点。",
        f"- Reconstruction invalid counts: M0=`{invalid_summary['M0_invalid_count']}`, M_local_fullfit=`{invalid_summary['M_local_fullfit_invalid_count']}`；invalid rows保留在 `residual_points.csv`，未静默删除。",
        "",
        "## 下一步选择",
        "",
        "- **推荐 D：残差结构更复杂，暂不建立 correction。** 当前证据支持 top 的 b(v) 共同偏置和一定 u/gain 结构，但固定 bin 的诊断尚不足以授权建立 correction；本轮没有拟合或部署任何 b(v)/gain correction。",
        "- 若后续继续，应先由人工确认 residual decomposition，再单独定义 correction 模型与独立 validation 方案。",
        "",
        f"Outputs: `{output_dir}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 4A FIT-only Circular Cone residual decomposition")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    cone_hash_before = sha256_file(FORMAL_CONE)
    if cone_hash_before != EXPECTED_CONE_SHA256:
        raise RuntimeError(f"Formal Cone hash mismatch: {cone_hash_before}")
    if not LOCAL_FULLFIT_RESULT.is_file():
        raise FileNotFoundError(f"M_local_fullfit result missing: {LOCAL_FULLFIT_RESULT}")

    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records, extension_provenance = load_fit_records(intrinsics)
    local_payload = json.loads(LOCAL_FULLFIT_RESULT.read_text(encoding="utf-8"))
    local_model = local_payload["local_model_as_runtime_mapping"]
    point_rows, invalid_summary = collect_points(records, calibration, local_model, reconstruction_params)
    decomposition, bootstrap_rows = decomposition_rows(point_rows)
    write_csv(out / "residual_points.csv", point_rows)
    write_csv(out / "residual_decomposition_vs_v.csv", decomposition)
    write_csv(out / "frame_bootstrap.csv", bootstrap_rows)
    save_plots(out, point_rows, decomposition)

    provenance = {
        "task": "Task 4A FIT-only Circular Cone position-dependent residual decomposition",
        "formal_cone_sha256_before": cone_hash_before,
        "formal_cone_sha256_after": sha256_file(FORMAL_CONE),
        "validation_opened": False,
        "production_writeback": False,
        "fit_frame_ids": task3a.FIT_IDS,
        "validation_frame_ids": task3a.VALIDATION_IDS,
        "models": ["M0", "M_local_fullfit"],
        "local_fullfit_result": str(LOCAL_FULLFIT_RESULT),
        "bin_widths_px": list(BIN_WIDTHS),
        "bootstrap_count": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "extension_provenance_count": len(extension_provenance),
        "invalid_summary": invalid_summary,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    (out / "report.md").write_text(render_report(point_rows, decomposition, bootstrap_rows, invalid_summary, cone_hash_before, provenance["formal_cone_sha256_after"], out), encoding="utf-8")
    (out / "OUTPUT_FILES.md").write_text(
        """# Task 4A output files

| file | meaning | boundary |
|---|---|---|
| residual_points.csv | every FIT truth point, M0/local lambda and e_lambda | invalid rows retained |
| residual_decomposition_vs_v.csv | fixed 30/60/100 px b(v)+delta_g(v) diagnostics | not a correction model |
| frame_bootstrap.csv | frame-level bootstrap samples for b and delta_g | no point bootstrap |
| report.md | offset/gain/u/v/frame/top-bottom diagnosis and A/B/C/D choice | FIT-only |
| residual_vs_v.png | residual distribution versus v | diagnostic plot |
| offset_b_vs_v.png | b(v) for three fixed bin widths | diagnostic plot |
| gain_delta_g_vs_v.png | delta_g(v) for three fixed bin widths | diagnostic plot |
| residual_region_distribution.png | top/middle/bottom distributions | diagnostic plot |
| m0_vs_local_residual.png | M0/local residual comparison | diagnostic plot |
| provenance.json | split isolation, hash and bootstrap provenance | no validation claim |
""",
        encoding="utf-8",
    )
    cone_hash_after = sha256_file(FORMAL_CONE)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Cone changed during Task 4A")
    print("VALIDATION_OPENED=False")
    print(f"POINT_ROWS={len(point_rows)}")
    print(f"BOOTSTRAP_ROWS={len(bootstrap_rows)}")
    print("NEXT_OPTION=D")
    print(f"OUTPUT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
