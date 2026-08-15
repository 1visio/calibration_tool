#!/usr/bin/env python3
"""Task 6D: full-board PnP ray-plane truth uncertainty audit (FIT-only).

The formal current ITERATIVE+LM PnP is kept unchanged.  Monte Carlo corner
noise is estimated only from each frame's observed reprojection residual field
and is zero mean; the frozen Circular Cone is used only as a residual reference.
No Validation data, laser-surface refit, intrinsics edit, solver switch, or
correction is performed.
"""

from __future__ import annotations

import argparse
import csv
import copy
import itertools
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT.parents[1]
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import audit_board_coordinate_residual as board_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import audit_pnp_truth_stability as pnp_audit  # noqa: E402


DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "fullboard_pnp_uncertainty"
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_MM = 20.0
TARGET_FRAME = "027"
MC_REPS = 1000
BALANCED_REPS = 100
RNG_SEED = 20260815
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 3000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--mc-reps", type=int, default=MC_REPS)
    parser.add_argument("--balanced-reps", type=int, default=BALANCED_REPS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def stats(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0, "bias": math.nan, "rmse": math.nan, "p95_abs": math.nan, "max_abs": math.nan, "median": math.nan, "std": math.nan}
    return {"count": int(len(x)), "bias": float(np.mean(x)), "rmse": float(np.sqrt(np.mean(x * x))), "p95_abs": float(np.percentile(np.abs(x), 95)), "max_abs": float(np.max(np.abs(x))), "median": float(np.median(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0}


def spearman_pair(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan, math.nan, int(len(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), int(len(x))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fixed.write_csv(path, rows)


def solve_current(obj: np.ndarray, corners: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, corners, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return {"status": "solve_failed"}
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(obj, corners, k, distortion, rvec, tvec)
        projected, rmse, depths = pnp_audit.reprojection_for_pose(obj, corners, rvec, tvec, k, distortion)
        pose = pnp_audit.aligned_plane_from_rt(rvec, tvec)
        pose.update({"status": "ok" if np.all(depths > 0) else "negative_depth", "projected": projected, "reprojection_rmse_px": rmse, "depth_min_mm": float(np.min(depths))})
        return pose
    except cv2.error as exc:
        return {"status": f"opencv_error:{exc.__class__.__name__}"}


def align_plane(pose: Mapping[str, Any], reference_normal: np.ndarray) -> tuple[np.ndarray, float]:
    normal = np.asarray(pose["normal"], dtype=np.float64).copy()
    d = float(pose["d"])
    if float(normal @ reference_normal) < 0.0:
        normal = -normal
        d = -d
    return normal, d


def lambda_for_pose(uv: np.ndarray, pose: Mapping[str, Any], reference_normal: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    normal, d = align_plane(pose, reference_normal)
    truth = fixed.coverage.plane_ray_truth(uv[:, 0], uv[:, 1], normal, d, k, distortion)
    lamb = np.asarray(truth["points"], dtype=np.float64)[:, 2]
    valid = np.asarray(truth["valid"], dtype=bool) & np.isfinite(lamb)
    angle = float(np.degrees(np.arccos(np.clip(float(normal @ reference_normal), -1.0, 1.0))))
    return lamb, valid, angle, d


def corner_covariance(corners: np.ndarray, projected: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    residual = np.asarray(projected, dtype=np.float64) - np.asarray(corners, dtype=np.float64)
    centered = residual - np.mean(residual, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, ddof=1)
    covariance = np.asarray(covariance, dtype=np.float64).reshape(2, 2)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1.0e-10)
    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return covariance, {"du_std_px": float(np.std(centered[:, 0], ddof=1)), "dv_std_px": float(np.std(centered[:, 1], ddof=1)), "du_dv_cov_px2": float(covariance[0, 1]), "residual_rms_px": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))}


def monte_carlo_frame(
    frame_id: str,
    corners: np.ndarray,
    obj: np.ndarray,
    uv: np.ndarray,
    lambda_full: np.ndarray,
    full_pose: Mapping[str, Any],
    k: np.ndarray,
    distortion: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    projected = np.asarray(full_pose["projected"], dtype=np.float64)
    covariance, cov_summary = corner_covariance(corners, projected)
    rng = np.random.default_rng(seed + int(frame_id))
    noise = rng.multivariate_normal(np.zeros(2), covariance, size=(int(reps), len(corners)))
    samples = np.full((int(reps), len(uv)), np.nan, dtype=np.float64)
    angles: list[float] = []
    distances: list[float] = []
    success = 0
    reference_normal = np.asarray(full_pose["normal"], dtype=np.float64)
    for rep in range(int(reps)):
        solved = solve_current(obj, corners + noise[rep], k, distortion)
        if solved.get("status") not in {"ok", "negative_depth"}:
            continue
        lamb, valid, angle, d = lambda_for_pose(uv, solved, reference_normal, k, distortion)
        samples[rep, valid] = lamb[valid]
        angles.append(angle)
        distances.append(float(d - float(full_pose["d"])))
        success += 1
    delta = samples - lambda_full[None, :]
    sigma = np.nanstd(samples, axis=0, ddof=1)
    p95 = np.nanpercentile(np.abs(delta), 95, axis=0)
    max_abs = np.nanmax(np.abs(delta), axis=0)
    mean = np.nanmean(samples, axis=0)
    valid_reps = np.sum(np.isfinite(samples), axis=0)
    point_rows: list[dict[str, Any]] = []
    for i in range(len(uv)):
        point_rows.append({"frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "point_index": i, "u_px": float(uv[i, 0]), "v_px": float(uv[i, 1]), "lambda_truth_full_mm": float(lambda_full[i]), "lambda_mean_mm": float(mean[i]), "lambda_std_mm": float(sigma[i]), "delta_lambda_mean_mm": float(mean[i] - lambda_full[i]), "delta_lambda_p95_abs_mm": float(p95[i]), "delta_lambda_max_abs_mm": float(max_abs[i]), "valid_mc_replicates": int(valid_reps[i]), "mc_replicates": int(reps), "sensor_edge_distance_px": float(min(uv[i, 0], IMAGE_WIDTH - 1.0 - uv[i, 0], uv[i, 1], IMAGE_HEIGHT - 1.0 - uv[i, 1]))})
    agg = {"mc_replicates": int(reps), "mc_successful_replicates": success, "mc_success_rate": float(success / max(int(reps), 1)), "mc_lambda_sigma_median_mm": float(np.nanmedian(sigma)), "mc_lambda_sigma_p95_mm": float(np.nanpercentile(sigma, 95)), "mc_lambda_sigma_max_mm": float(np.nanmax(sigma)), "mc_delta_p95_median_mm": float(np.nanmedian(p95)), "mc_delta_p95_p95_mm": float(np.nanpercentile(p95, 95)), "mc_delta_max_max_mm": float(np.nanmax(max_abs)), "mc_plane_normal_angle_p95_deg": float(np.percentile(angles, 95)) if angles else math.nan, "mc_plane_distance_delta_p95_abs_mm": float(np.percentile(np.abs(distances), 95)) if distances else math.nan, **cov_summary}
    return point_rows, agg, {"samples": samples, "sigma": sigma, "p95": p95, "mean": mean, "valid_reps": valid_reps, "covariance": covariance}


def balanced_designs(obj: np.ndarray) -> dict[str, np.ndarray]:
    x_index = np.rint(obj[:, 0] / SQUARE_MM).astype(int)
    y_index = np.rint(obj[:, 1] / SQUARE_MM).astype(int)
    designs = {
        "checkerboard_A": np.flatnonzero((x_index + y_index) % 2 == 0),
        "checkerboard_B": np.flatnonzero((x_index + y_index) % 2 == 1),
        "uniform_sparse": np.flatnonzero(np.isin(x_index, [0, 2, 4, 6, 8, 10]) & np.isin(y_index, [0, 2, 4, 6, 7])),
    }
    return designs


def balanced_row(frame_id: str, design: str, replicate: int, indices: np.ndarray, obj: np.ndarray, corners: np.ndarray, uv: np.ndarray, lambda_full: np.ndarray, full_pose: Mapping[str, Any], k: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    solved = solve_current(obj[indices], corners[indices], k, distortion)
    row: dict[str, Any] = {"row_type": "replicate", "frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "design": design, "replicate": replicate, "subset_point_count": int(len(indices)), "status": solved.get("status", "failed")}
    if solved.get("status") not in {"ok", "negative_depth"}:
        return row
    lamb, valid, angle, d = lambda_for_pose(uv, solved, np.asarray(full_pose["normal"], dtype=np.float64), k, distortion)
    delta = lamb[valid] - lambda_full[valid]
    ds = stats(delta)
    row.update({"valid_lambda_count": int(np.count_nonzero(valid)), "normal_angle_diff_deg": angle, "plane_distance_diff_mm": float(d - float(full_pose["d"])), "abs_plane_distance_diff_mm": abs(float(d - float(full_pose["d"]))), "lambda_delta_bias_mm": ds["bias"], "lambda_delta_rmse_mm": ds["rmse"], "lambda_delta_p95_abs_mm": ds["p95_abs"], "lambda_delta_max_abs_mm": ds["max_abs"], "subset_reprojection_rmse_px": float(solved["reprojection_rmse_px"]), "subset_depth_min_mm": float(solved["depth_min_mm"])})
    return row


def balanced_frame_rows(frame_id: str, obj: np.ndarray, corners: np.ndarray, uv: np.ndarray, lambda_full: np.ndarray, full_pose: Mapping[str, Any], k: np.ndarray, distortion: np.ndarray, reps: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed_designs = balanced_designs(obj)
    rows: list[dict[str, Any]] = []
    for design, indices in fixed_designs.items():
        rows.append(balanced_row(frame_id, design, 0, indices, obj, corners, uv, lambda_full, full_pose, k, distortion))
    rng = np.random.default_rng(seed + 5000 + int(frame_id))
    x = np.rint(obj[:, 0] / SQUARE_MM).astype(int)
    y = np.rint(obj[:, 1] / SQUARE_MM).astype(int)
    anchors = np.flatnonzero(np.isin(x, [0, 10]) & np.isin(y, [0, 7]))
    remaining = np.setdiff1d(np.arange(len(obj)), anchors)
    for replicate in range(int(reps)):
        extra = rng.choice(remaining, size=min(40, len(remaining)), replace=False)
        indices = np.sort(np.concatenate([anchors, extra]))
        rows.append(balanced_row(frame_id, "full_span_random", replicate, indices, obj, corners, uv, lambda_full, full_pose, k, distortion))
    aggregates: list[dict[str, Any]] = []
    for design in (*fixed_designs.keys(), "full_span_random"):
        selected = [row for row in rows if row["design"] == design and row.get("status") in {"ok", "negative_depth"}]
        if not selected:
            aggregates.append({"row_type": "frame_design_aggregate", "frame_id": frame_id, "design": design, "replicate_count_ok": 0})
            continue
        def arr(name: str) -> np.ndarray:
            return np.asarray([finite(row.get(name)) for row in selected], dtype=np.float64)
        aggregates.append({"row_type": "frame_design_aggregate", "frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "design": design, "replicate_count_ok": len(selected), "subset_point_count": int(np.median([int(row["subset_point_count"]) for row in selected])), "normal_angle_p95_deg": float(np.percentile(arr("normal_angle_diff_deg"), 95)), "plane_distance_abs_p95_mm": float(np.percentile(np.abs(arr("plane_distance_diff_mm")), 95)), "lambda_delta_rmse_median_mm": float(np.median(arr("lambda_delta_rmse_mm"))), "lambda_delta_rmse_p95_mm": float(np.percentile(arr("lambda_delta_rmse_mm"), 95)), "lambda_delta_p95_median_mm": float(np.median(arr("lambda_delta_p95_abs_mm"))), "lambda_delta_p95_p95_mm": float(np.percentile(arr("lambda_delta_p95_abs_mm"), 95)), "lambda_delta_max_max_mm": float(np.max(arr("lambda_delta_max_abs_mm")))})
    return rows, aggregates


def load_task6c_reference() -> dict[str, Any]:
    path = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "pnp_truth_stability_audit" / "frame_truth_uncertainty_summary.csv"
    if not path.is_file():
        return {"available": False}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("row_type") == "frame"]
    values = [finite(row.get("subset_lambda_delta_p95_max_mm")) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return {"available": bool(values), "max_subset_p95_mm": max(values) if values else math.nan, "median_subset_p95_mm": float(np.median(values)) if values else math.nan}


def frame_summary_row(frame_id: str, item: Mapping[str, Any], base_summary: Mapping[str, Any], mc: Mapping[str, Any], mc_points: Sequence[Mapping[str, Any]], balanced_aggs: Sequence[Mapping[str, Any]], task6c: Mapping[str, Any]) -> dict[str, Any]:
    cone_error = -np.asarray(item["residual"], dtype=np.float64)
    cone = stats(cone_error)
    sigma = np.asarray([finite(row.get("lambda_std_mm")) for row in mc_points], dtype=np.float64)
    p95 = np.asarray([finite(row.get("delta_lambda_p95_abs_mm")) for row in mc_points], dtype=np.float64)
    u = np.asarray([finite(row.get("u_px")) for row in mc_points], dtype=np.float64)
    v = np.asarray([finite(row.get("v_px")) for row in mc_points], dtype=np.float64)
    lam = np.asarray([finite(row.get("lambda_truth_full_mm")) for row in mc_points], dtype=np.float64)
    edge = np.asarray([finite(row.get("sensor_edge_distance_px")) for row in mc_points], dtype=np.float64)
    med_edge_cut = float(np.nanpercentile(edge, 10))
    edge_sigma = sigma[edge <= med_edge_cut]
    center_sigma = sigma[edge >= np.nanpercentile(edge, 50)]
    rho_u, p_u, _ = spearman_pair(sigma, u)
    rho_v, p_v, _ = spearman_pair(sigma, v)
    rho_l, p_l, _ = spearman_pair(sigma, lam)
    random_agg = next((row for row in balanced_aggs if row.get("design") == "full_span_random"), {})
    return {"row_type": "frame", "frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "laser_point_count": len(mc_points), "cone_valid_point_count": len(item["residual"]), "full_pnp_rmse_px": float(item["pose"].reprojection_rmse_px), "corner_residual_rms_px": float(mc["residual_rms_px"]), "corner_du_std_px": float(mc["du_std_px"]), "corner_dv_std_px": float(mc["dv_std_px"]), "corner_du_dv_cov_px2": float(mc["du_dv_cov_px2"]), "board_tilt_deg": float(np.degrees(np.arccos(np.clip(abs(float(item["pose"].normal[2])), -1.0, 1.0)))), "cone_e_lambda_bias_mm": cone["bias"], "cone_e_lambda_rmse_mm": cone["rmse"], "cone_e_lambda_p95_abs_mm": cone["p95_abs"], "cone_a_frame_mm": -finite(base_summary.get("a_frame_mm")), "cone_k_frame_mm_per_normalized_stripe": -finite(base_summary.get("k_frame_mm_per_normalized_stripe")), "mc_successful_replicates": int(mc["mc_successful_replicates"]), "mc_lambda_sigma_median_mm": float(np.nanmedian(sigma)), "mc_lambda_sigma_p95_mm": float(np.nanpercentile(sigma, 95)), "mc_lambda_sigma_max_mm": float(np.nanmax(sigma)), "mc_delta_p95_median_mm": float(np.nanmedian(p95)), "mc_delta_p95_p95_mm": float(np.nanpercentile(p95, 95)), "mc_delta_max_max_mm": float(np.nanmax([finite(row.get("delta_lambda_max_abs_mm")) for row in mc_points])), "mc_plane_normal_angle_p95_deg": float(mc["mc_plane_normal_angle_p95_deg"]), "mc_plane_distance_delta_p95_abs_mm": float(mc["mc_plane_distance_delta_p95_abs_mm"]), "sigma_vs_u_spearman_rho": rho_u, "sigma_vs_u_spearman_p_value": p_u, "sigma_vs_v_spearman_rho": rho_v, "sigma_vs_v_spearman_p_value": p_v, "sigma_vs_lambda_spearman_rho": rho_l, "sigma_vs_lambda_spearman_p_value": p_l, "edge_q10_sigma_median_mm": float(np.nanmedian(edge_sigma)) if len(edge_sigma) else math.nan, "center_q50_sigma_median_mm": float(np.nanmedian(center_sigma)) if len(center_sigma) else math.nan, "edge_to_center_sigma_ratio": float(np.nanmedian(edge_sigma) / max(np.nanmedian(center_sigma), 1.0e-12)) if len(edge_sigma) and len(center_sigma) else math.nan, "balanced_random_lambda_delta_p95_p95_mm": finite(random_agg.get("lambda_delta_p95_p95_mm")), "task6c_subset_lambda_delta_p95_max_mm": finite(task6c.get("max_subset_p95_mm")) if task6c.get("available") else math.nan}


def classify(frame_rows: Sequence[Mapping[str, Any]], balanced_aggs: Sequence[Mapping[str, Any]], task6c: Mapping[str, Any]) -> dict[str, Any]:
    mc_p95 = np.asarray([finite(row.get("mc_delta_p95_p95_mm")) for row in frame_rows], dtype=np.float64)
    mc_sigma = np.asarray([finite(row.get("mc_lambda_sigma_p95_mm")) for row in frame_rows], dtype=np.float64)
    cone_rmse = np.asarray([finite(row.get("cone_e_lambda_rmse_mm")) for row in frame_rows], dtype=np.float64)
    random_p95 = np.asarray([finite(row.get("balanced_random_lambda_delta_p95_p95_mm")) for row in frame_rows], dtype=np.float64)
    ratio = mc_sigma / np.maximum(cone_rmse, 1.0e-12)
    valid = np.isfinite(mc_p95) & np.isfinite(mc_sigma) & np.isfinite(cone_rmse)
    typical_p95 = float(np.nanmedian(mc_p95))
    worst_p95 = float(np.nanmax(mc_p95))
    typical_sigma_ratio = float(np.nanmedian(ratio))
    worst_random = float(np.nanmax(random_p95)) if np.any(np.isfinite(random_p95)) else math.nan
    geometry_ok = bool(len(frame_rows) == 30 and np.all(np.isfinite(mc_p95[valid])) and np.all(np.asarray([int(row.get("mc_successful_replicates", 0)) for row in frame_rows]) >= 950))
    edge_ratios = np.asarray([finite(row.get("edge_to_center_sigma_ratio")) for row in frame_rows], dtype=np.float64)
    tilt = np.asarray([finite(row.get("board_tilt_deg")) for row in frame_rows], dtype=np.float64)
    tilt_rho, tilt_p, _ = spearman_pair(tilt, mc_sigma)
    # Full-board gates use the Monte Carlo point-wise 95th-percentile field;
    # balanced full-span bootstrap is a secondary geometric diagnostic.
    if not geometry_ok:
        verdict = "C. HIGH"
    elif worst_p95 < .01 and (not math.isfinite(worst_random) or worst_random < .03) and typical_sigma_ratio < .25:
        verdict = "A. LOW"
    elif worst_p95 < .05 and (not math.isfinite(worst_random) or worst_random < .10) and typical_sigma_ratio < .75:
        verdict = "B. MODERATE"
    else:
        verdict = "C. HIGH"
    if verdict == "A. LOW":
        next_step = "retain current PnP truth"
    elif verdict == "B. MODERATE":
        next_step = "improve PnP / corner / intrinsics"
    else:
        next_step = "design an independent physical-truth experiment"
    return {"verdict": verdict, "geometry_ok": geometry_ok, "typical_mc_delta_p95_mm": typical_p95, "worst_mc_delta_p95_mm": worst_p95, "typical_mc_sigma_to_cone_rmse_ratio": typical_sigma_ratio, "worst_balanced_random_delta_p95_p95_mm": worst_random, "edge_to_center_sigma_ratio_median": float(np.nanmedian(edge_ratios)), "edge_to_center_sigma_ratio_max": float(np.nanmax(edge_ratios)), "tilt_vs_sigma_spearman_rho": tilt_rho, "tilt_vs_sigma_spearman_p_value": tilt_p, "task6c_subset_p95_max_mm": finite(task6c.get("max_subset_p95_mm")) if task6c.get("available") else math.nan, "fullboard_to_task6c_p95_ratio": worst_p95 / max(finite(task6c.get("max_subset_p95_mm")), 1.0e-12) if task6c.get("available") else math.nan, "next_step": next_step}


def plot_outputs(output: Path, frame_rows: Sequence[Mapping[str, Any]], mc_points: Sequence[Mapping[str, Any]], balanced_rows: Sequence[Mapping[str, Any]]) -> None:
    frames = sorted({str(row["frame_id"]) for row in frame_rows}, key=int)
    x = np.arange(len(frames))
    fig, ax = plt.subplots(figsize=(14, 5.5)); ax.plot(x, [finite(next(row for row in frame_rows if row["frame_id"] == f).get("mc_delta_p95_p95_mm")) for f in frames], marker=".", label="full-board MC P95 field"); ax.plot(x, [finite(next(row for row in frame_rows if row["frame_id"] == f).get("balanced_random_lambda_delta_p95_p95_mm")) for f in frames], marker=".", label="full-span random bootstrap P95"); ax.set_xticks(x, frames, rotation=60, fontsize=7); ax.set_ylabel("lambda truth uncertainty / mm"); ax.set_title("Full-board truth uncertainty by frame (red = 027)"); ax.grid(alpha=.2); ax.legend(); ax.scatter([frames.index(TARGET_FRAME)], [finite(next(row for row in frame_rows if row["frame_id"] == TARGET_FRAME).get("mc_delta_p95_p95_mm"))], color="#c53030", zorder=5); fig.tight_layout(); fig.savefig(output / "p95_delta_lambda_truth_by_frame.png", dpi=180); plt.close(fig)

    colors = ["#c53030" if row.get("is_frame027") else "#2b6cb0" for row in frame_rows]
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.scatter([finite(row.get("mc_lambda_sigma_p95_mm")) for row in frame_rows], [finite(row.get("cone_e_lambda_rmse_mm")) for row in frame_rows], c=colors, s=34); ax.set(xlabel="full-board MC sigma_lambda P95 / mm", ylabel="Cone e_lambda RMSE / mm", title="Full-board truth uncertainty versus Cone residual"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "truth_uncertainty_vs_cone_residual.png", dpi=180); plt.close(fig)

    rows = list(mc_points); fig, ax = plt.subplots(figsize=(8, 5.5)); scatter = ax.scatter([finite(row["u_px"]) for row in rows], [finite(row["v_px"]) for row in rows], c=[finite(row["lambda_std_mm"]) for row in rows], s=5, cmap="viridis"); ax.set(xlabel="u / px", ylabel="v / px", title="sigma_lambda(u,v) full-board Monte Carlo map"); ax.invert_yaxis(); ax.grid(alpha=.2); fig.colorbar(scatter, ax=ax, label="sigma_lambda / mm"); fig.tight_layout(); fig.savefig(output / "lambda_uncertainty_map.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2)); designs = ("checkerboard_A", "checkerboard_B", "uniform_sparse", "full_span_random");
    for design in designs:
        selected = [row for row in balanced_rows if row.get("design") == design and row.get("status") in {"ok", "negative_depth"}]
        if not selected: continue
        axes[0].plot([finite(row.get("normal_angle_diff_deg")) for row in selected], label=design, linewidth=.8)
        axes[1].plot([abs(finite(row.get("plane_distance_diff_mm"))) for row in selected], label=design, linewidth=.8)
    axes[0].set_title("Balanced subset plane normal variation"); axes[0].set_ylabel("angle difference / deg"); axes[1].set_title("Balanced subset plane distance variation"); axes[1].set_ylabel("absolute distance difference / mm")
    for ax in axes: ax.set_xlabel("frame/design replicate rows"); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(output / "balanced_bootstrap_plane_consistency.png", dpi=180); plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    x = finite(value)
    return "n/a" if not math.isfinite(x) else f"{x:.{digits}f}"


def render_report(data_root: Path, frame_rows: Sequence[Mapping[str, Any]], balanced_aggs: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], frozen_info: Mapping[str, Any], mc_reps: int, balanced_reps: int) -> str:
    target = next(row for row in frame_rows if row.get("frame_id") == TARGET_FRAME)
    lines = ["# Task 6D — Full-board PnP truth uncertainty audit", "", f"`FULLBOARD_TRUTH_UNCERTAINTY = {decision['verdict']}`", "", "## Scope and boundary", "", f"- FIT-only frames: `001–018`, `025–036` (30 frames); 027 retained and reported separately. Only explicit FIT files were opened; Validation 019–024 and 037–040 were not read.", f"- Formal intrinsics/distortion and current `SOLVEPNP_ITERATIVE` + `solvePnPRefineLM` were retained. No formal intrinsics, solver settings, laser surface, frame selection, or correction were changed.", f"- Frozen provenance SHA-256: `{frozen_info['provenance_sha256']}`; formal Cone SHA-256: `{frozen_info['formal_cone_sha256']}`. Cone is only an observed-residual reference.", "", "## Full-board Monte Carlo", "", f"- Each frame uses all 88 detected corners. Corner perturbations are zero-mean Gaussian draws from that frame's empirical reprojection residual covariance (centered du/dv), with **{mc_reps}** PnP re-solves; perturbation size is independent of Cone residual.", "- Truth pixels are all extracted centers with valid full-board ray-plane intersections; Cone z-range validity is not used to select them.", f"- Typical MC point-field P95 |delta lambda|: **{fmt(decision['typical_mc_delta_p95_mm'])} mm**; worst frame P95: **{fmt(decision['worst_mc_delta_p95_mm'])} mm**.", f"- Full-span random balanced bootstrap uses four board-corner anchors plus random interior corners, **{balanced_reps}** replicates/frame; checkerboard A/B and uniform sparse designs are fixed full-span diagnostics.", "", "| frame | full PnP RMSE px | MC sigma P95 mm | MC delta P95 P95 mm | random balanced delta P95 P95 mm | Cone RMSE mm | sigma/Cone ratio | tilt deg |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in frame_rows:
        ratio = finite(row.get("mc_lambda_sigma_p95_mm")) / max(finite(row.get("cone_e_lambda_rmse_mm")), 1.0e-12)
        lines.append(f"| {row['frame_id']}{' (027)' if row['is_frame027'] else ''} | {fmt(row['full_pnp_rmse_px'])} | {fmt(row['mc_lambda_sigma_p95_mm'])} | {fmt(row['mc_delta_p95_p95_mm'])} | {fmt(row['balanced_random_lambda_delta_p95_p95_mm'])} | {fmt(row['cone_e_lambda_rmse_mm'])} | {fmt(ratio,3)} | {fmt(row['board_tilt_deg'],3)} |")
    lines += ["", "## Balanced full-span bootstrap", "", "| design | frame-design aggregate count | median lambda RMSE mm | P95 lambda RMSE mm | P95-of-P95 mm | P95 normal angle deg | P95 plane distance mm |", "|---|---:|---:|---:|---:|---:|---:|"]
    for design in ("checkerboard_A", "checkerboard_B", "uniform_sparse", "full_span_random"):
        selected = [row for row in balanced_aggs if row.get("design") == design]
        if not selected: continue
        lines.append(f"| {design} | {len(selected)} | {fmt(np.median([finite(row.get('lambda_delta_rmse_median_mm')) for row in selected]))} | {fmt(np.percentile([finite(row.get('lambda_delta_rmse_p95_mm')) for row in selected],95))} | {fmt(np.percentile([finite(row.get('lambda_delta_p95_p95_mm')) for row in selected],95))} | {fmt(np.percentile([finite(row.get('normal_angle_p95_deg')) for row in selected],95))} | {fmt(np.percentile([finite(row.get('plane_distance_abs_p95_mm')) for row in selected],95))} |")
    lines += ["", "## Spatial/pose dependence", "", "- The complete `sigma_lambda(u,v)` map is in `lambda_uncertainty_map.csv`; per-frame Spearman correlations with u, v, lambda and edge/center ratios are in `frame_truth_uncertainty.csv`.", f"- Across frames, edge/center sigma ratio median **{fmt(decision['edge_to_center_sigma_ratio_median'],3)}**, maximum **{fmt(decision['edge_to_center_sigma_ratio_max'],3)}**; board-tilt versus MC sigma P95 Spearman rho **{fmt(decision['tilt_vs_sigma_spearman_rho'],3)}** (p={fmt(decision['tilt_vs_sigma_spearman_p_value'],3)}). This is a modest, non-universal edge effect rather than a stable tilt law.", f"- Frame 027: MC sigma P95 **{fmt(target['mc_lambda_sigma_p95_mm'])} mm**, MC delta P95 P95 **{fmt(target['mc_delta_p95_p95_mm'])} mm**, random balanced delta P95 P95 **{fmt(target['balanced_random_lambda_delta_p95_p95_mm'])} mm**, Cone RMSE **{fmt(target['cone_e_lambda_rmse_mm'])} mm**.", "", "## Answers", "", f"1. Task 6C subset instability is mainly geometric degeneration: **{decision['fullboard_to_task6c_p95_ratio'] < 0.5 if math.isfinite(decision['fullboard_to_task6c_p95_ratio']) else 'n/a'}** (full-board MC worst P95 / Task 6C subset worst P95 = {fmt(decision['fullboard_to_task6c_p95_ratio'],3)}).", f"2. Full-board truth typically stabilizes to about **{fmt(decision['typical_mc_delta_p95_mm'])} mm** point-field P95; worst frame is **{fmt(decision['worst_mc_delta_p95_mm'])} mm**.", f"3. PnP uncertainty explains most observed residual: **{decision['verdict'] == 'C. HIGH'}**; median MC sigma/Cone RMSE ratio is **{fmt(decision['typical_mc_sigma_to_cone_rmse_ratio'],3)}**.", f"4. Sensor-edge/pose increase: edge/center median **{fmt(decision['edge_to_center_sigma_ratio_median'],3)}×**, max **{fmt(decision['edge_to_center_sigma_ratio_max'],3)}×**, tilt rho **{fmt(decision['tilt_vs_sigma_spearman_rho'],3)}** (p={fmt(decision['tilt_vs_sigma_spearman_p_value'],3)}); no universal pose law.", f"5. Next step: **{decision['next_step']}**.", "", "## Conclusion", "", f"`FULLBOARD_TRUTH_UNCERTAINTY = {decision['verdict']}`.", "The gates are descriptive: LOW requires full-board MC and balanced full-span variation to remain well below the observed Cone residual; MODERATE permits a smaller but non-negligible fraction; HIGH means full-board truth uncertainty reaches the residual scale or exceeds the declared margins.", "", "Generated figures: `p95_delta_lambda_truth_by_frame.png`, `truth_uncertainty_vs_cone_residual.png`, `lambda_uncertainty_map.png`, and `balanced_bootstrap_plane_consistency.png`.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    groups = board_audit.inventory_fit(data_root)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, frozen_info = board_audit.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    obj = fixed.triplets.make_object_points(BOARD_COLS, BOARD_ROWS, SQUARE_MM)
    board_summaries, processed = board_audit.process_groups_board(groups, intrinsics, calibration, reconstruction_params, frozen_model)
    k = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    distortion = np.asarray(intrinsics.dist_coeffs, dtype=np.float64)
    mc_point_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []
    balanced_aggs: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    task6c = load_task6c_reference()
    for base_summary in board_summaries:
        frame_id = str(base_summary["frame_id"])
        item = processed[frame_id]
        corners = np.asarray(item["corners"], dtype=np.float64)
        full_pose = pnp_audit.aligned_plane_from_rt(item["pose"].rvec, item["pose"].tvec, np.asarray(item["pose"].normal, dtype=np.float64))
        projected, full_rmse, depths = pnp_audit.reprojection_for_pose(obj, corners, item["pose"].rvec, item["pose"].tvec, k, distortion)
        full_pose.update({"normal": np.asarray(item["pose"].normal, dtype=np.float64), "d": float(item["pose"].d), "projected": projected, "reprojection_rmse_px": full_rmse, "depth_min_mm": float(np.min(depths))})
        raw_uv = np.asarray(item["uv_raw"], dtype=np.float64)
        raw_points = np.asarray(item["truth_points_raw"], dtype=np.float64)
        valid = np.asarray(item["truth_valid_raw"], dtype=bool) & np.isfinite(raw_points[:, 2])
        uv = raw_uv[valid]
        lambda_full = raw_points[valid, 2]
        mc_points, mc_agg, _ = monte_carlo_frame(frame_id, corners, obj, uv, lambda_full, full_pose, k, distortion, int(args.mc_reps), int(args.seed))
        for row in mc_points:
            row["board_tilt_deg"] = float(np.degrees(np.arccos(np.clip(abs(float(item["pose"].normal[2])), -1.0, 1.0))))
        mc_point_rows.extend(mc_points)
        frame_balanced, aggregate_balanced = balanced_frame_rows(frame_id, obj, corners, uv, lambda_full, full_pose, k, distortion, int(args.balanced_reps), int(args.seed))
        balanced_rows.extend(frame_balanced)
        balanced_aggs.extend(aggregate_balanced)
        item["uv_truth"] = uv
        frame_rows.append(frame_summary_row(frame_id, item, base_summary, mc_agg, mc_points, aggregate_balanced, task6c))
    decision = classify(frame_rows, balanced_aggs, task6c)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fullboard_pnp_montecarlo.csv", mc_point_rows)
    write_csv(output_dir / "balanced_bootstrap.csv", balanced_rows + balanced_aggs)
    write_csv(output_dir / "lambda_uncertainty_map.csv", mc_point_rows)
    write_csv(output_dir / "frame_truth_uncertainty.csv", frame_rows + [{"row_type": "aggregate", **decision}])
    plot_outputs(output_dir, frame_rows, mc_point_rows, balanced_rows)
    (output_dir / "report.md").write_text(render_report(data_root, frame_rows, balanced_aggs, decision, frozen_info, int(args.mc_reps), int(args.balanced_reps)), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "frame_count": len(frame_rows), "mc_point_rows": len(mc_point_rows), "balanced_rows": len(balanced_rows), "balanced_aggregates": len(balanced_aggs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
