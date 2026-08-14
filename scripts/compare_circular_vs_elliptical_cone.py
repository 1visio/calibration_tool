#!/usr/bin/env python3
"""Task 5A: FIT-only Circular Cone versus strict Elliptical Cone comparison.

Both candidates minimize the same camera-Z ray intersection residual. The
Elliptical Cone is a strict axial quadratic cone and adds only two shape DOF
relative to the nested Circular Cone. No position-dependent correction exists
in this script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import circular_cone_local_parameterization as local  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
import run_circular_cone_local_fullfit as task3b2  # noqa: E402
import run_circular_cone_residual_decomposition as task4a  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "circular_vs_elliptical_cone"
)
TASK4C_RESULT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "leave027_cone_refit_control"
    / "leave027_cone_refit.json"
)
TARGET_FRAME = "027"
MODEL_NAMES = ("Circular", "Elliptical")
REGION_NAMES = ("global", "top_formal_edge", "middle_formal", "bottom_formal_edge")
MAX_FIT_POINTS = 3000
V_BIN_WIDTH = 60
DEPTH_MIN_MM = 100.0
DEPTH_MAX_MM = 1500.0
INVALID_RESIDUAL_MM = 1000.0
ANGLE_SCALE = math.radians(1.0)
CIRCULAR_SCALES = np.asarray([ANGLE_SCALE, ANGLE_SCALE, 10.0, 10.0, 10.0, 0.1])
ELLIPTICAL_SCALES = np.asarray([ANGLE_SCALE, ANGLE_SCALE, 10.0, 10.0, 10.0, 0.2, 0.2, 0.2])
PARAMETER_NAMES = {
    "Circular": ("theta_axis", "phi_axis", "c1", "c2", "s_ref", "log_q"),
    "Elliptical": ("theta_axis", "phi_axis", "c1", "c2", "s_ref", "S00", "S01", "S11"),
}


@dataclass
class FitResult:
    kind: str
    params: np.ndarray
    initial: np.ndarray
    scales: np.ndarray
    success: bool
    status: int
    message: str
    robust_cost: float
    raw_cost: float
    raw_mse: float
    selected_count: int
    invalid_count: int
    nfev: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5A Circular versus Elliptical Cone")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=task3b2.MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def axis_basis(theta: float, phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = local.angles_to_axis(theta, phi)
    e1, e2 = local.transverse_basis(axis)
    return axis, e1, e2


def build_reference_anchor(records: Sequence[task3a.FrameRecord]) -> np.ndarray:
    centers = np.asarray([np.mean(record.truth_points, axis=0) for record in records], dtype=np.float64)
    return np.mean(centers, axis=0)


def circular_model_to_params(model: Mapping[str, Any], p_ref: np.ndarray) -> np.ndarray:
    theta = local.legacy_model_to_theta(model)
    axis, e1, e2 = axis_basis(float(theta[0]), float(theta[1]))
    apex = np.asarray(theta[2:5], dtype=np.float64)
    s_ref = float((p_ref - apex) @ axis)
    c_ref = apex + s_ref * axis
    offset = c_ref - p_ref
    q = 1.0 / math.tan(float(theta[5]))
    return np.asarray(
        [theta[0], theta[1], float(offset @ e1), float(offset @ e2), s_ref, math.log(q)],
        dtype=np.float64,
    )


def circular_to_elliptical_params(circular: np.ndarray) -> np.ndarray:
    theta, phi, c1, c2, s_ref, log_q = np.asarray(circular, dtype=np.float64)
    log_h = 2.0 * log_q
    return np.asarray([theta, phi, c1, c2, s_ref, log_h, 0.0, log_h], dtype=np.float64)


def parameter_bounds(kind: str, q_min: float, q_max: float) -> tuple[np.ndarray, np.ndarray]:
    if kind == "Circular":
        return (
            np.asarray([0.0, -math.pi, -3000.0, -3000.0, -3000.0, math.log(q_min)]),
            np.asarray([math.pi, math.pi, 3000.0, 3000.0, 3000.0, math.log(q_max)]),
        )
    log_h_min = 2.0 * math.log(q_min)
    log_h_max = 2.0 * math.log(q_max)
    return (
        np.asarray([0.0, -math.pi, -3000.0, -3000.0, -3000.0, log_h_min - 1.0, -6.0, log_h_min - 1.0]),
        np.asarray([math.pi, math.pi, 3000.0, 3000.0, 3000.0, log_h_max + 1.0, 6.0, log_h_max + 1.0]),
    )


def geometry_from_params(
    kind: str,
    params: np.ndarray,
    p_ref: np.ndarray,
    q_min: float,
    q_max: float,
) -> dict[str, Any]:
    p = np.asarray(params, dtype=np.float64)
    theta, phi, c1, c2, s_ref = p[:5]
    axis, e1, e2 = axis_basis(float(theta), float(phi))
    c_ref = p_ref + c1 * e1 + c2 * e2
    apex = c_ref - s_ref * axis
    if kind == "Circular":
        q = math.exp(float(p[5]))
        h = np.eye(2, dtype=np.float64) * q * q
    else:
        s_matrix = np.asarray([[p[5], p[6]], [p[6], p[7]]], dtype=np.float64)
        h = expm(s_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    principal_q = np.sqrt(np.maximum(eigenvalues, 0.0))
    feasible = (
        np.isfinite(h).all()
        and np.isfinite(apex).all()
        and np.all(principal_q >= q_min)
        and np.all(principal_q <= q_max)
    )
    principal_alpha = np.degrees(np.arctan2(1.0, principal_q))
    orientation_deg = math.degrees(math.atan2(float(eigenvectors[1, 0]), float(eigenvectors[0, 0])))
    return {
        "axis": axis,
        "e1": e1,
        "e2": e2,
        "apex": apex,
        "H": h,
        "principal_q": principal_q,
        "principal_alpha_deg": principal_alpha,
        "principal_orientation_deg_in_transverse_basis": orientation_deg,
        "feasible": bool(feasible),
    }


def solve_quadratic(a: np.ndarray, b: np.ndarray, c: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    a, b, c = np.broadcast_arrays(np.asarray(a), np.asarray(b), np.asarray(c))
    roots = np.full((a.size, 2), np.nan, dtype=np.float64)
    af, bf, cf = a.ravel(), b.ravel(), c.ravel()
    linear = np.abs(af) < epsilon
    linear_valid = linear & (np.abs(bf) >= epsilon)
    roots[linear_valid, 0] = -cf[linear_valid] / bf[linear_valid]
    quadratic = ~linear
    discriminant = bf * bf - 4.0 * af * cf
    valid = quadratic & (discriminant >= 0.0)
    if np.any(valid):
        sqrt_disc = np.sqrt(np.maximum(discriminant[valid], 0.0))
        aq, bq, cq = af[valid], bf[valid], cf[valid]
        stable_q = -0.5 * (bq + np.copysign(sqrt_disc, bq))
        r1 = stable_q / aq
        r2 = np.where(np.abs(stable_q) >= epsilon, cq / stable_q, (-bq - sqrt_disc) / (2.0 * aq))
        roots[valid, 0] = r1
        roots[valid, 1] = r2
    return roots


def intersect_rays(
    rays: np.ndarray,
    geometry: Mapping[str, Any],
    z_hint: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not bool(geometry["feasible"]):
        return np.full(len(rays), np.nan), np.zeros(len(rays), dtype=bool)
    axis = np.asarray(geometry["axis"])
    e1 = np.asarray(geometry["e1"])
    e2 = np.asarray(geometry["e2"])
    apex = np.asarray(geometry["apex"])
    h = np.asarray(geometry["H"])
    ray_t = np.column_stack([rays @ e1, rays @ e2])
    apex_t = np.asarray([apex @ e1, apex @ e2])
    ray_a = rays @ axis
    apex_a = float(apex @ axis)
    aa = np.einsum("ni,ij,nj->n", ray_t, h, ray_t) - ray_a * ray_a
    bb = -2.0 * (ray_t @ h @ apex_t) + 2.0 * ray_a * apex_a
    cc_value = float(apex_t @ h @ apex_t - apex_a * apex_a)
    roots = solve_quadratic(aa, bb, np.full(len(rays), cc_value))
    points = roots[:, :, None] * rays[:, None, :]
    axial = np.einsum("nki,i->nk", points - apex[None, None, :], axis)
    valid = (
        np.isfinite(roots)
        & (roots >= DEPTH_MIN_MM)
        & (roots <= DEPTH_MAX_MM)
        & (axial >= 0.0)
    )
    scores = np.where(valid, np.abs(roots - z_hint), np.inf)
    choice = np.argmin(scores, axis=1)
    selected = roots[np.arange(len(rays)), choice]
    selected_valid = np.isfinite(scores[np.arange(len(rays)), choice])
    selected[~selected_valid] = np.nan
    return selected, selected_valid


def rays_for_pixels(pixels: np.ndarray, intrinsics: Any) -> np.ndarray:
    normalized = cv2.undistortPoints(
        np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    ).reshape(-1, 2)
    return np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])


def select_fit_arrays(
    records: Sequence[task3a.FrameRecord], intrinsics: Any, max_points: int = MAX_FIT_POINTS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_frame = max(20, max_points // max(len(records), 1))
    rays: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    for record in records:
        indices = task3a.triplets.uniform_subsample(np.arange(len(record.truth_points), dtype=int), per_frame)
        rays.append(rays_for_pixels(record.pixels_uv[indices], intrinsics))
        truth.append(record.truth_points[indices, 2])
        frame_ids.append(np.full(len(indices), record.frame_id, dtype="U3"))
    return np.concatenate(rays), np.concatenate(truth), np.concatenate(frame_ids)


def fit_candidate(
    kind: str,
    records: Sequence[task3a.FrameRecord],
    intrinsics: Any,
    p_ref: np.ndarray,
    initial: np.ndarray,
    q_min: float,
    q_max: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
) -> FitResult:
    rays, truth_z, frame_ids = select_fit_arrays(records, intrinsics)
    weights = task3a.frame_equal_weights(frame_ids)
    sqrt_weights = np.sqrt(weights)
    z_hint = float(np.median(truth_z))
    scales = CIRCULAR_SCALES if kind == "Circular" else ELLIPTICAL_SCALES
    bounds = parameter_bounds(kind, q_min, q_max)

    def objective(params: np.ndarray) -> np.ndarray:
        geometry = geometry_from_params(kind, params, p_ref, q_min, q_max)
        lam, valid = intersect_rays(rays, geometry, z_hint)
        residual = np.full(len(truth_z), INVALID_RESIDUAL_MM, dtype=np.float64)
        residual[valid] = lam[valid] - truth_z[valid]
        return residual * sqrt_weights

    result = least_squares(
        objective,
        np.minimum(np.maximum(np.asarray(initial, dtype=np.float64), bounds[0] + 1.0e-10), bounds[1] - 1.0e-10),
        bounds=bounds,
        x_scale=scales,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=0,
    )
    final_geometry = geometry_from_params(kind, result.x, p_ref, q_min, q_max)
    lam, valid = intersect_rays(rays, final_geometry, z_hint)
    raw = np.full(len(truth_z), INVALID_RESIDUAL_MM, dtype=np.float64)
    raw[valid] = lam[valid] - truth_z[valid]
    weighted = raw * sqrt_weights
    return FitResult(
        kind=kind,
        params=np.asarray(result.x, dtype=np.float64),
        initial=np.asarray(initial, dtype=np.float64),
        scales=scales,
        success=bool(result.success and np.all(valid)),
        status=int(result.status),
        message=str(result.message),
        robust_cost=float(result.cost),
        raw_cost=0.5 * float(np.sum(weighted * weighted)),
        raw_mse=float(np.mean(weighted * weighted)),
        selected_count=len(truth_z),
        invalid_count=int(np.count_nonzero(~valid)),
        nfev=int(result.nfev),
    )


def evaluate_records(
    records: Sequence[task3a.FrameRecord],
    kind: str,
    params: np.ndarray,
    p_ref: np.ndarray,
    intrinsics: Any,
    q_min: float,
    q_max: float,
    z_hint: float,
) -> dict[str, dict[str, np.ndarray]]:
    geometry = geometry_from_params(kind, params, p_ref, q_min, q_max)
    output: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        rays = rays_for_pixels(record.pixels_uv, intrinsics)
        lam, valid = intersect_rays(rays, geometry, z_hint)
        residual = np.full(len(lam), np.nan)
        residual[valid] = lam[valid] - record.truth_points[valid, 2]
        output[record.frame_id] = {"lambda": lam, "valid": valid, "residual": residual}
    return output


def region_bounds(region: str) -> tuple[float, float]:
    if region == "global":
        return -float("inf"), float("inf")
    for name, low, high in task3a.REGIONS:
        if name == region:
            return float(low), float(high)
    raise KeyError(region)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    index = min(int(np.searchsorted(cumulative, probability * cumulative[-1], side="left")), len(values) - 1)
    return float(values[index])


def metric(values: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    x, w = values[valid], weights[valid]
    if x.size == 0:
        return {"count": 0, "bias": float("nan"), "mae": float("nan"), "rmse": float("nan"), "p95": float("nan"), "max_abs": float("nan")}
    w = w / np.sum(w)
    return {
        "count": int(x.size),
        "bias": float(np.sum(w * x)),
        "mae": float(np.sum(w * np.abs(x))),
        "rmse": float(np.sqrt(np.sum(w * x * x))),
        "p95": weighted_quantile(np.abs(x), w, 0.95),
        "max_abs": float(np.max(np.abs(x))),
    }


def metrics_for_records(
    records: Sequence[task3a.FrameRecord],
    evaluation: Mapping[str, Mapping[str, np.ndarray]],
    region: str,
    weighting: str,
) -> dict[str, Any]:
    low, high = region_bounds(region)
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    frame_count = 0
    invalid = 0
    for record in records:
        result = evaluation[record.frame_id]
        mask_region = (record.pixels_uv[:, 1] >= low) & (record.pixels_uv[:, 1] < high)
        valid = mask_region & result["valid"] & np.isfinite(result["residual"])
        invalid += int(np.count_nonzero(mask_region & ~result["valid"]))
        selected = result["residual"][valid]
        if selected.size == 0:
            continue
        frame_count += 1
        values.append(selected)
        weights.append(np.ones(len(selected)) if weighting == "point_equal" else np.full(len(selected), 1.0 / len(selected)))
    stats = metric(np.concatenate(values) if values else np.empty(0), np.concatenate(weights) if weights else np.empty(0))
    stats["frame_count"] = frame_count
    stats["invalid_count"] = invalid
    return stats


def build_metric_rows(
    main_records: Sequence[task3a.FrameRecord],
    frame027: Sequence[task3a.FrameRecord],
    evaluations: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    scopes = {"main_fit_29": main_records, "frame027_sensitivity": frame027}
    for scope, records in scopes.items():
        for weighting in ("point_equal", "frame_equal"):
            for model in MODEL_NAMES:
                for region in REGION_NAMES:
                    stats = metrics_for_records(records, evaluations[model], region, weighting)
                    row = {
                        "row_type": "region_summary",
                        "scope": scope,
                        "weighting": weighting,
                        "model": model,
                        "region": region,
                        **stats,
                    }
                    rows.append(row)
                    lookup[(scope, weighting, model, region)] = row

    # Fixed 60px v bins, frame-equal, main analysis only.
    for model in MODEL_NAMES:
        for start in range(0, 3000, V_BIN_WIDTH):
            end = start + V_BIN_WIDTH
            values: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            frames = 0
            for record in main_records:
                result = evaluations[model][record.frame_id]
                mask = (record.pixels_uv[:, 1] >= start) & (record.pixels_uv[:, 1] < end) & result["valid"]
                selected = result["residual"][mask]
                if selected.size:
                    frames += 1
                    values.append(selected)
                    weights.append(np.full(len(selected), 1.0 / len(selected)))
            stats = metric(np.concatenate(values) if values else np.empty(0), np.concatenate(weights) if weights else np.empty(0))
            rows.append({
                "row_type": "v_bin_summary",
                "scope": "main_fit_29",
                "weighting": "frame_equal",
                "model": model,
                "region": task3a.region_for_v(start + 0.5 * V_BIN_WIDTH),
                "v_start_px": start,
                "v_end_px": end,
                "v_center_px": start + 0.5 * V_BIN_WIDTH,
                "frame_count": frames,
                **stats,
            })
    return rows, lookup


def grid_delta_metrics(
    kind: str,
    candidate: np.ndarray,
    full: np.ndarray,
    grid_rays: np.ndarray,
    grid_v: np.ndarray,
    p_ref: np.ndarray,
    q_min: float,
    q_max: float,
    z_hint: float,
) -> dict[str, Any]:
    candidate_lambda, candidate_valid = intersect_rays(grid_rays, geometry_from_params(kind, candidate, p_ref, q_min, q_max), z_hint)
    full_lambda, full_valid = intersect_rays(grid_rays, geometry_from_params(kind, full, p_ref, q_min, q_max), z_hint)
    common = candidate_valid & full_valid
    delta = candidate_lambda - full_lambda
    output: dict[str, Any] = {}
    for region in REGION_NAMES:
        low, high = region_bounds(region)
        mask = common & (grid_v >= low) & (grid_v < high)
        output[region] = metric(delta[mask], np.ones(np.count_nonzero(mask)))
    output["common_valid_count"] = int(np.count_nonzero(common))
    return output


def run_jackknife(
    main_records: Sequence[task3a.FrameRecord],
    full_fits: Mapping[str, FitResult],
    intrinsics: Any,
    p_ref: np.ndarray,
    q_min: float,
    q_max: float,
    loss: str,
    f_scale: float,
    max_nfev: int,
    z_hint: float,
) -> list[dict[str, Any]]:
    grid_uv, _, _ = task3a.build_grid(main_records)
    grid_rays = rays_for_pixels(grid_uv, intrinsics)
    grid_v = grid_uv[:, 1]
    rows: list[dict[str, Any]] = []
    for omitted_record in main_records:
        train = [record for record in main_records if record.frame_id != omitted_record.frame_id]
        for kind in MODEL_NAMES:
            full = full_fits[kind]
            fit = fit_candidate(kind, train, intrinsics, p_ref, full.params, q_min, q_max, loss, f_scale, max_nfev)
            heldout_eval = evaluate_records([omitted_record], kind, fit.params, p_ref, intrinsics, q_min, q_max, z_hint)
            heldout = metrics_for_records([omitted_record], heldout_eval, "global", "point_equal")
            grid = grid_delta_metrics(kind, fit.params, full.params, grid_rays, grid_v, p_ref, q_min, q_max, z_hint)
            normalized = (fit.params - full.params) / full.scales
            row: dict[str, Any] = {
                "row_type": "jackknife_holdout",
                "frame_id": omitted_record.frame_id,
                "model": kind,
                "train_frame_count": len(train),
                "fit_success": fit.success,
                "fit_status": fit.status,
                "fit_nfev": fit.nfev,
                "fit_robust_cost": fit.robust_cost,
                "fit_raw_mse": fit.raw_mse,
                "fit_invalid_count": fit.invalid_count,
                "parameter_normalized_delta_l2": float(np.linalg.norm(normalized)),
                "heldout_bias_mm": heldout["bias"],
                "heldout_rmse_mm": heldout["rmse"],
                "heldout_p95_mm": heldout["p95"],
                "heldout_invalid_count": heldout["invalid_count"],
            }
            for region in REGION_NAMES:
                row[f"grid_{region}_bias_delta_mm"] = grid[region]["bias"]
                row[f"grid_{region}_p95_abs_delta_mm"] = grid[region]["p95"]
                row[f"grid_{region}_max_abs_delta_mm"] = grid[region]["max_abs"]
            for index, name in enumerate(PARAMETER_NAMES[kind]):
                row[f"parameter_{name}"] = fit.params[index]
                row[f"delta_parameter_{name}"] = fit.params[index] - full.params[index]
                row[f"normalized_delta_parameter_{name}"] = normalized[index]
            rows.append(row)
            print(
                f"JACK {omitted_record.frame_id} {kind}: success={fit.success} "
                f"heldout_rmse={heldout['rmse']:.7g} grid_p95={grid['global']['p95']:.7g}",
                flush=True,
            )
    return rows


def append_full_frame_rows(
    rows: list[dict[str, Any]],
    main_records: Sequence[task3a.FrameRecord],
    evaluations: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    for record in main_records:
        for model in MODEL_NAMES:
            values = metrics_for_records([record], evaluations[model], "global", "point_equal")
            rows.append({
                "row_type": "full_model_frame_evaluation",
                "frame_id": record.frame_id,
                "model": model,
                "heldout_bias_mm": values["bias"],
                "heldout_rmse_mm": values["rmse"],
                "heldout_p95_mm": values["p95"],
                "heldout_invalid_count": values["invalid_count"],
            })


def circular_params_to_runtime_model(params: np.ndarray, p_ref: np.ndarray, z_range: Sequence[float]) -> dict[str, Any]:
    theta, phi, c1, c2, s_ref, log_q = np.asarray(params, dtype=np.float64)
    axis, e1, e2 = axis_basis(theta, phi)
    apex = p_ref + c1 * e1 + c2 * e2 - s_ref * axis
    alpha = math.atan2(1.0, math.exp(log_q))
    return {
        "model_type": "circular_cone",
        "axis_unit_camera": axis.tolist(),
        "apex_camera_mm": apex.tolist(),
        "half_apex_angle_deg": math.degrees(alpha),
        "z_valid_range_mm": [float(z_range[0]), float(z_range[1])],
    }


def formal_circular_crosscheck(
    records: Sequence[task3a.FrameRecord],
    params: np.ndarray,
    p_ref: np.ndarray,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    intrinsics: Any,
    q_min: float,
    q_max: float,
    z_hint: float,
) -> dict[str, Any]:
    z_all = np.concatenate([record.truth_points[:, 2] for record in records])
    model = circular_params_to_runtime_model(params, p_ref, [float(np.min(z_all)), float(np.max(z_all))])
    max_delta = 0.0
    common_count = 0
    generic_invalid = 0
    formal_invalid = 0
    for record in records:
        rays = rays_for_pixels(record.pixels_uv, intrinsics)
        generic, generic_valid = intersect_rays(rays, geometry_from_params("Circular", params, p_ref, q_min, q_max), z_hint)
        candidate = copy.deepcopy(dict(calibration)); candidate["laser_model"] = model
        formal, formal_valid, _ = task4a.lambda_by_input(record.pixels_uv, candidate, reconstruction_params)
        common = generic_valid & formal_valid
        if np.any(common):
            max_delta = max(max_delta, float(np.max(np.abs(generic[common] - formal[common]))))
        common_count += int(np.count_nonzero(common))
        generic_invalid += int(np.count_nonzero(~generic_valid))
        formal_invalid += int(np.count_nonzero(~formal_valid))
    return {
        "common_valid_count": common_count,
        "max_abs_lambda_delta_mm": max_delta,
        "generic_invalid_count": generic_invalid,
        "formal_invalid_count": formal_invalid,
        "pass_1e_6_mm": bool(max_delta < 1.0e-6),
    }


def save_residual_plot(
    output_dir: Path,
    main_records: Sequence[task3a.FrameRecord],
    evaluations: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
    colors = {"Circular": "#2563eb", "Elliptical": "#c2410c"}
    for axis, model in zip(axes, MODEL_NAMES):
        all_v: list[np.ndarray] = []
        all_e: list[np.ndarray] = []
        for record in main_records:
            result = evaluations[model][record.frame_id]
            valid = result["valid"]
            all_v.append(record.pixels_uv[valid, 1])
            all_e.append(result["residual"][valid])
        axis.scatter(np.concatenate(all_v), np.concatenate(all_e), s=2, alpha=0.10, color=colors[model])
        centers: list[float] = []
        medians: list[float] = []
        p95s: list[float] = []
        for start in range(0, 3000, V_BIN_WIDTH):
            per_frame: list[np.ndarray] = []
            for record in main_records:
                result = evaluations[model][record.frame_id]
                mask = (record.pixels_uv[:, 1] >= start) & (record.pixels_uv[:, 1] < start + V_BIN_WIDTH) & result["valid"]
                if np.any(mask):
                    per_frame.append(result["residual"][mask])
            if per_frame:
                x = np.concatenate(per_frame)
                w = np.concatenate([np.full(len(values), 1.0 / len(values)) for values in per_frame])
                centers.append(start + 0.5 * V_BIN_WIDTH)
                medians.append(weighted_quantile(x, w, 0.5))
                p95s.append(weighted_quantile(np.abs(x), w, 0.95))
        axis.plot(centers, medians, color="#111827", linewidth=1.6, label="frame-equal median")
        axis.plot(centers, p95s, color="#dc2626", linestyle="--", linewidth=1.2, label="frame-equal P95 |e|")
        axis.plot(centers, -np.asarray(p95s), color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.45)
        axis.axhline(0.0, color="#555", linewidth=0.8); axis.axvline(300.0, color="#777", linestyle=":"); axis.axvline(2700.0, color="#777", linestyle=":")
        axis.set_title(model); axis.set_ylabel("lambda_model - lambda_truth / mm"); axis.grid(alpha=0.2); axis.legend()
    axes[-1].set_xlabel("v / px")
    fig.suptitle("Circular versus Elliptical Cone ray-depth residual — 29 FIT frames")
    fig.tight_layout(); fig.savefig(output_dir / "residual_vs_v_comparison.png", dpi=170); plt.close(fig)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.7g}"


def classify(
    lookup: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    jackknife_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    def rmse(model: str, region: str) -> float:
        return float(lookup[("main_fit_29", "frame_equal", model, region)]["rmse"])

    reductions = {
        region: (rmse("Circular", region) - rmse("Elliptical", region)) / rmse("Circular", region)
        for region in REGION_NAMES
    }
    top_bias = {model: float(lookup[("main_fit_29", "frame_equal", model, "top_formal_edge")]["bias"]) for model in MODEL_NAMES}
    bottom_bias = {model: float(lookup[("main_fit_29", "frame_equal", model, "bottom_formal_edge")]["bias"]) for model in MODEL_NAMES}
    asymmetry = {model: abs(top_bias[model] - bottom_bias[model]) for model in MODEL_NAMES}
    asymmetry_reduction = (asymmetry["Circular"] - asymmetry["Elliptical"]) / asymmetry["Circular"]
    jack = [row for row in jackknife_rows if row["row_type"] == "jackknife_holdout"]
    failures = [row for row in jack if not bool(row["fit_success"])]
    by_frame = {frame: {} for frame in {str(row["frame_id"]) for row in jack}}
    for row in jack:
        by_frame[str(row["frame_id"])][str(row["model"])] = float(row["heldout_rmse_mm"])
    paired_reductions = np.asarray([
        (values["Circular"] - values["Elliptical"]) / values["Circular"]
        for values in by_frame.values() if set(values) == set(MODEL_NAMES) and values["Circular"] > 0.0
    ])
    ellipse_grid_p95 = np.asarray([
        float(row["grid_global_p95_abs_delta_mm"]) for row in jack if row["model"] == "Elliptical"
    ])
    parameter_stability: dict[str, dict[str, float]] = {}
    prediction_stability: dict[str, dict[str, float]] = {}
    for model in MODEL_NAMES:
        parameter_l2 = np.asarray([
            float(row["parameter_normalized_delta_l2"]) for row in jack if row["model"] == model
        ])
        grid_p95 = np.asarray([
            float(row["grid_global_p95_abs_delta_mm"]) for row in jack if row["model"] == model
        ])
        parameter_stability[model] = {
            "normalized_delta_l2_median": float(np.median(parameter_l2)),
            "normalized_delta_l2_max": float(np.max(parameter_l2)),
        }
        prediction_stability[model] = {
            "grid_p95_median_mm": float(np.median(grid_p95)),
            "grid_p95_max_mm": float(np.max(grid_p95)),
        }
    stable = not failures and ellipse_grid_p95.size == len(by_frame) and float(np.max(ellipse_grid_p95)) <= 0.10
    if not stable or paired_reductions.size != len(by_frame):
        result = "D"
    elif (
        reductions["top_formal_edge"] >= 0.20
        and reductions["bottom_formal_edge"] >= 0.10
        and reductions["middle_formal"] >= -0.05
        and asymmetry_reduction >= 0.25
        and float(np.median(paired_reductions)) >= 0.10
    ):
        result = "B"
    elif (
        max(reductions["global"], reductions["top_formal_edge"], reductions["bottom_formal_edge"]) >= 0.10
        and float(np.median(paired_reductions)) > 0.0
    ):
        result = "C"
    else:
        result = "A"
    return result, {
        "rmse_reduction_fraction_elliptical_vs_circular": reductions,
        "top_bottom_bias_asymmetry_mm": asymmetry,
        "asymmetry_reduction_fraction": asymmetry_reduction,
        "jackknife_paired_heldout_rmse_reduction_median": float(np.median(paired_reductions)) if paired_reductions.size else float("nan"),
        "jackknife_paired_heldout_improved_frame_count": int(np.count_nonzero(paired_reductions > 0.0)),
        "jackknife_frame_count": len(by_frame),
        "elliptical_jackknife_grid_p95_median_mm": float(np.median(ellipse_grid_p95)) if ellipse_grid_p95.size else float("nan"),
        "elliptical_jackknife_grid_p95_max_mm": float(np.max(ellipse_grid_p95)) if ellipse_grid_p95.size else float("nan"),
        "jackknife_parameter_stability": parameter_stability,
        "jackknife_prediction_stability": prediction_stability,
        "jackknife_failure_count": len(failures),
        "stable": stable,
        "fixed_rule": {
            "B": "top RMSE reduction>=20%, bottom>=10%, middle degradation<=5%, asymmetry reduction>=25%, jackknife median heldout reduction>=10%",
            "C": "at least one global/top/bottom RMSE reduction>=10% and jackknife median heldout reduction>0, but B is not met",
            "A": "stable comparison with no material improvement gate met",
            "D": "fit/jackknife failure, incomplete pairing, or elliptical max jackknife grid P95>0.10mm",
        },
    }


def render_report(
    lookup: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    fits: Mapping[str, FitResult],
    geometries: Mapping[str, Mapping[str, Any]],
    classification: Mapping[str, Any],
    result: str,
    crosscheck: Mapping[str, Any],
    hash_before: str,
    hash_after: str,
    output_dir: Path,
) -> str:
    label = {
        "A": "Circular sufficient",
        "B": "Elliptical materially better",
        "C": "Elliptical improves but仍不足",
        "D": "Evidence insufficient",
    }[result]
    ellipse_alpha = np.asarray(geometries["Elliptical"]["principal_alpha_deg"], dtype=np.float64)
    ellipse_hits_alpha_bound = bool(np.max(ellipse_alpha) >= 89.95 - 1.0e-4)
    lines = [
        "# Task 5A — Circular Cone vs Elliptical Cone 模型形式对照",
        "",
        "**FIT_ONLY = TRUE**  ",
        "**VALIDATION_OPENED = FALSE**  ",
        "**PRODUCTION_M0_MODIFIED = FALSE**  ",
        "**EMPIRICAL_CORRECTION_ADDED = FALSE**",
        "",
        f"`MODEL_FORM_RESULT = {result}. {label}`",
        "",
        "## 模型与公平性约束",
        "",
        "- 主拟合：001–018 + 025–036，临时排除027，共29帧；027仅在两个冻结候选上单独测试。",
        "- 两模型统一最小化 `r_lambda=lambda_model-lambda_truth`，使用相同frame-balanced sampling、frame-equal weighting、soft_l1、f_scale、least_squares和max_nfev。",
        "- 两模型共用正nappe、固定相机深度[100,1500] mm和固定truth-domain z hint选根；拟合时不使用候选模型各自的z_valid_range过滤，避免产生不同objective样本集。",
        "- Circular参数化：axis line + apex axial location + one circular slope，共6 DOF。",
        "- Elliptical使用严格二次锥 `x_perp^T H x_perp-axial^2=0`，H为2×2 SPD；共8 DOF。Circular是其嵌套子模型 `H=q^2 I`，只放宽圆对称的两个DOF。",
        "- 没有b(v)、spline、polynomial、LUT或任何u/v位置项。",
        f"- Circular诊断求交与正式reconstruct公共有效点最大lambda差=`{fmt(crosscheck['max_abs_lambda_delta_mm'])}` mm，1e-6 gate=`{crosscheck['pass_1e_6_mm']}`。",
        f"- 正式M0 SHA-256 before/after：`{hash_before}` / `{hash_after}`。",
        "",
        "## Full-FIT optimizer",
        "",
        "| model | DOF | success | robust cost | raw MSE | selected | nfev |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        fit = fits[model]
        lines.append(f"| {model} | {len(fit.params)} | {fit.success} | {fmt(fit.robust_cost)} | {fmt(fit.raw_mse)} | {fit.selected_count} | {fit.nfev} |")
    lines += [
        "",
        f"- Circular half angle=`{fmt(geometries['Circular']['principal_alpha_deg'][0])}` deg。",
        f"- Elliptical principal half angles=`{fmt(geometries['Elliptical']['principal_alpha_deg'][0])}` / `{fmt(geometries['Elliptical']['principal_alpha_deg'][1])}` deg，transverse orientation=`{fmt(geometries['Elliptical']['principal_orientation_deg_in_transverse_basis'])}` deg。",
        f"- Elliptical是否命中89.95°主半角硬边界：`{ellipse_hits_alpha_bound}`。命中边界意味着该方向接近退化，不能仅凭Full-FIT训练改善认定模型形式已充分。",
        "",
        "## 主29帧 frame-equal 指标",
        "",
        "| region | model | Bias mm | RMSE mm | P95 mm | RMSE change vs Circular |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for region in REGION_NAMES:
        circ = lookup[("main_fit_29", "frame_equal", "Circular", region)]
        for model in MODEL_NAMES:
            row = lookup[("main_fit_29", "frame_equal", model, region)]
            change = (float(row["rmse"]) - float(circ["rmse"])) / float(circ["rmse"])
            lines.append(f"| {region} | {model} | {fmt(row['bias'])} | {fmt(row['rmse'])} | {fmt(row['p95'])} | {change:+.3%} |")
    lines += [
        "",
        "## 027 sensitivity（不参与模型选择）",
        "",
        "| model | Bias mm | RMSE mm | P95 mm |",
        "|---|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        row = lookup[("frame027_sensitivity", "point_equal", model, "global")]
        lines.append(f"| {model} | {fmt(row['bias'])} | {fmt(row['rmse'])} | {fmt(row['p95'])} |")
    reductions = classification["rmse_reduction_fraction_elliptical_vs_circular"]
    lines += [
        "",
        "## Jackknife稳定性与最终判断",
        "",
        f"- Elliptical 29折 prediction-grid P95 median/max=`{fmt(classification['elliptical_jackknife_grid_p95_median_mm'])}` / `{fmt(classification['elliptical_jackknife_grid_p95_max_mm'])}` mm；失败数=`{classification['jackknife_failure_count']}`。",
        f"- Circular normalized parameter-delta L2 median/max=`{fmt(classification['jackknife_parameter_stability']['Circular']['normalized_delta_l2_median'])}` / `{fmt(classification['jackknife_parameter_stability']['Circular']['normalized_delta_l2_max'])}`；Elliptical=`{fmt(classification['jackknife_parameter_stability']['Elliptical']['normalized_delta_l2_median'])}` / `{fmt(classification['jackknife_parameter_stability']['Elliptical']['normalized_delta_l2_max'])}`。参数值及逐参数delta见 `frame_cv_comparison.csv`。",
        "- Elliptical参数delta较小需要结合主半角命中硬边界理解；边界会限制该方向漂移，因此模型稳定性主要以prediction-grid drift和held-out RMSE判断。",
        f"- 配对heldout RMSE改善中位数=`{float(classification['jackknife_paired_heldout_rmse_reduction_median']):.3%}`；改善frame数=`{classification['jackknife_paired_heldout_improved_frame_count']}/{classification['jackknife_frame_count']}`。",
        f"- Top RMSE改善=`{float(reductions['top_formal_edge']):.3%}`，Bottom=`{float(reductions['bottom_formal_edge']):.3%}`，Middle=`{float(reductions['middle_formal']):.3%}`，top/bottom bias asymmetry改善=`{float(classification['asymmetry_reduction_fraction']):.3%}`。",
        "",
        f"1. Top是否明显下降：`{float(reductions['top_formal_edge']) >= 0.20}`。",
        f"2. Bottom是否同步改善：`{float(reductions['bottom_formal_edge']) >= 0.10}`。",
        f"3. Middle是否基本不恶化：`{float(reductions['middle_formal']) >= -0.05}`。",
        f"4. top+/bottom− asymmetry是否明显减弱：`{float(classification['asymmetry_reduction_fraction']) >= 0.25}`。",
        f"5. Elliptical改善在frame jackknife中是否稳定：`{classification['stable'] and float(classification['jackknife_paired_heldout_rmse_reduction_median']) > 0.0}`。",
        "",
        f"**最终：`MODEL_FORM_RESULT = {result}. {label}`。**",
        "",
        "该结果只授权下一步研究判断，不授权部署Elliptical Cone，也不是Validation结论。",
        "",
        "### 固定判据",
        "",
        f"- B：{classification['fixed_rule']['B']}。",
        f"- C：{classification['fixed_rule']['C']}。",
        f"- A：{classification['fixed_rule']['A']}。",
        f"- D：{classification['fixed_rule']['D']}。",
        "",
        f"Outputs: `{output_dir}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    hash_before = task3b2.sha256_file(task3b2.FORMAL_CONE)
    if hash_before != task3b2.EXPECTED_CONE_SHA256:
        raise RuntimeError(f"Formal M0 hash mismatch: {hash_before}")
    task4c = json.loads(TASK4C_RESULT.read_text(encoding="utf-8"))
    if task4c.get("validation_opened") is not False or task4c.get("production_writeback") is not False:
        raise RuntimeError("Task 4C provenance violates Task 5A boundary")

    # FIT-only loading. No validation loader is called.
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records = task3a.load_old_records()
    extension_records, extension_provenance = task3a.load_extension_records(intrinsics)
    records += extension_records
    if [record.frame_id for record in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT registry mismatch")
    main_records = [record for record in records if record.frame_id != TARGET_FRAME]
    frame027 = [record for record in records if record.frame_id == TARGET_FRAME]
    if len(main_records) != 29 or len(frame027) != 1:
        raise RuntimeError("Expected 29 main frames and one 027 sensitivity frame")

    p_ref = build_reference_anchor(main_records)
    z_all = np.concatenate([record.truth_points[:, 2] for record in main_records])
    z_hint = float(np.median(z_all))
    cone_cfg = dict(task3a.triplets.safe_yaml_load(task3b2.FORMAL_FIT_CONFIG)["models"]["cone"])
    q_min = 1.0 / math.tan(math.radians(float(cone_cfg.get("alpha_max_deg", 89.95))))
    q_max = 1.0 / math.tan(math.radians(float(cone_cfg.get("alpha_min_deg", 60.0))))
    loss = str(cone_cfg.get("loss", "soft_l1"))
    f_scale = float(cone_cfg.get("f_scale_mm", 0.1))
    max_nfev = int(cone_cfg.get("max_nfev", 3000))

    leave_model = task4c["parameters"]["M_leave027"]["runtime_mapping"]
    circular_initial = circular_model_to_params(leave_model, p_ref)
    circular_fit = fit_candidate("Circular", main_records, intrinsics, p_ref, circular_initial, q_min, q_max, loss, f_scale, max_nfev)
    if not circular_fit.success:
        raise RuntimeError(f"Circular full fit failed: {circular_fit.message}; invalid={circular_fit.invalid_count}")

    ellipse_base = circular_to_elliptical_params(circular_fit.params)
    ellipse_seeds = [ellipse_base.copy()]
    seed_diag = ellipse_base.copy(); seed_diag[5] += 0.05; seed_diag[7] -= 0.05; ellipse_seeds.append(seed_diag)
    seed_offdiag = ellipse_base.copy(); seed_offdiag[6] = 0.05; ellipse_seeds.append(seed_offdiag)
    ellipse_candidates = [
        fit_candidate("Elliptical", main_records, intrinsics, p_ref, seed, q_min, q_max, loss, f_scale, max_nfev)
        for seed in ellipse_seeds
    ]
    successful_ellipse = [fit for fit in ellipse_candidates if fit.success]
    if not successful_ellipse:
        raise RuntimeError("All Elliptical full-fit deterministic seeds failed")
    elliptical_fit = min(successful_ellipse, key=lambda fit: fit.robust_cost)
    full_fits = {"Circular": circular_fit, "Elliptical": elliptical_fit}

    evaluations = {
        model: evaluate_records(records, model, fit.params, p_ref, intrinsics, q_min, q_max, z_hint)
        for model, fit in full_fits.items()
    }
    metric_rows, lookup = build_metric_rows(main_records, frame027, evaluations)
    jackknife_rows = run_jackknife(main_records, full_fits, intrinsics, p_ref, q_min, q_max, loss, f_scale, max_nfev, z_hint)
    append_full_frame_rows(jackknife_rows, main_records, evaluations)
    result, classification = classify(lookup, jackknife_rows)
    geometries = {
        model: geometry_from_params(model, fit.params, p_ref, q_min, q_max)
        for model, fit in full_fits.items()
    }
    crosscheck = formal_circular_crosscheck(
        records, circular_fit.params, p_ref, calibration, reconstruction_params, intrinsics, q_min, q_max, z_hint
    )
    if not crosscheck["pass_1e_6_mm"]:
        raise RuntimeError(f"Circular generic/formal intersection mismatch: {crosscheck}")

    task3b2.write_csv(output_dir / "circular_vs_elliptical_metrics.csv", metric_rows)
    task3b2.write_csv(output_dir / "frame_cv_comparison.csv", jackknife_rows)
    save_residual_plot(output_dir, main_records, evaluations)
    hash_after = task3b2.sha256_file(task3b2.FORMAL_CONE)
    if hash_after != hash_before:
        raise RuntimeError("Formal M0 changed during Task 5A")
    (output_dir / "report.md").write_text(
        render_report(lookup, full_fits, geometries, classification, result, crosscheck, hash_before, hash_after, output_dir),
        encoding="utf-8",
    )
    provenance = {
        "task": "Task 5A Circular Cone vs Elliptical Cone model-form comparison",
        "main_fit_ids": [record.frame_id for record in main_records],
        "sensitivity_case_ids": [TARGET_FRAME],
        "validation_ids_not_opened": task3a.VALIDATION_IDS,
        "validation_opened": False,
        "formal_m0_modified": False,
        "empirical_correction_added": False,
        "formal_cone_sha256_before": hash_before,
        "formal_cone_sha256_after": hash_after,
        "objective": "lambda_model_camera_Z - lambda_truth_camera_Z",
        "sampling": "uniform per frame, fit_max_points=3000",
        "weighting": "frame_equal",
        "loss": loss,
        "f_scale_mm": f_scale,
        "max_nfev": max_nfev,
        "reference_anchor_mm": p_ref.tolist(),
        "model_definition": {
            "Circular": "x_perp^T(q^2 I)x_perp-axial^2=0; 6 DOF",
            "Elliptical": "x_perp^T H x_perp-axial^2=0, H=exp(S) SPD; 8 DOF",
        },
        "full_fits": {
            model: {
                "params": fit.params.tolist(),
                "success": fit.success,
                "status": fit.status,
                "message": fit.message,
                "robust_cost": fit.robust_cost,
                "raw_mse": fit.raw_mse,
                "selected_count": fit.selected_count,
                "nfev": fit.nfev,
                "geometry": {
                    "axis": geometries[model]["axis"].tolist(),
                    "apex": geometries[model]["apex"].tolist(),
                    "H": geometries[model]["H"].tolist(),
                    "principal_q": geometries[model]["principal_q"].tolist(),
                    "principal_alpha_deg": geometries[model]["principal_alpha_deg"].tolist(),
                    "principal_orientation_deg_in_transverse_basis": geometries[model]["principal_orientation_deg_in_transverse_basis"],
                },
            }
            for model, fit in full_fits.items()
        },
        "elliptical_deterministic_seed_costs": [fit.robust_cost for fit in ellipse_candidates],
        "classification": classification,
        "model_form_result": result,
        "circular_formal_crosscheck": crosscheck,
        "extension_provenance_count": len(extension_provenance),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=task3b2.json_default), encoding="utf-8"
    )
    print(f"MODEL_FORM_RESULT={result}")
    print(f"CIRCULAR_COST={circular_fit.robust_cost:.9g}")
    print(f"ELLIPTICAL_COST={elliptical_fit.robust_cost:.9g}")
    print(f"JACKKNIFE_ROWS={len([row for row in jackknife_rows if row['row_type']=='jackknife_holdout'])}")
    print("VALIDATION_OPENED=False")
    print(f"OUTPUT={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
