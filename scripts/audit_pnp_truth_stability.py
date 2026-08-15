#!/usr/bin/env python3
"""Task 6C: multi-pose PnP/ray-plane truth stability audit (FIT-only).

The script reuses the formal intrinsics, existing chessboard/Steger pipeline,
and frozen Circular Cone only as an observed-residual reference.  It does not
fit a laser surface, change PnP/intrinsics/Steger settings, open Validation,
or create a correction.
"""

from __future__ import annotations

import argparse
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


DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "pnp_truth_stability_audit"
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_MM = 20.0
TARGET_FRAME = "027"
BOOTSTRAP_SEED = 20260815


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def stats(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "bias": math.nan, "rmse": math.nan, "p95_abs": math.nan, "max_abs": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(len(x)),
        "bias": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p95_abs": float(np.percentile(np.abs(x), 95)),
        "max_abs": float(np.max(np.abs(x))),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


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


def subset_specs(obj: np.ndarray) -> dict[str, np.ndarray]:
    x = obj[:, 0]
    y = obj[:, 1]
    # The center row/column is included in both neighboring halves so every
    # subset retains a broad, non-degenerate spatial baseline.
    masks = {
        "full": np.ones(len(obj), dtype=bool),
        "left_half": x <= 100.0,
        "right_half": x >= 100.0,
        "top_half": y <= 60.0,
        "bottom_half": y >= 60.0,
        "central_subset": (x >= 40.0) & (x <= 160.0) & (y >= 20.0) & (y <= 100.0),
    }
    output: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        indices = np.flatnonzero(mask)
        xy = obj[indices, :2]
        centered = xy - np.mean(xy, axis=0)
        if len(indices) < 6 or np.linalg.matrix_rank(centered) < 2 or np.ptp(xy[:, 0]) < SQUARE_MM or np.ptp(xy[:, 1]) < SQUARE_MM:
            raise RuntimeError(f"Degenerate PnP subset {name}: n={len(indices)}")
        output[name] = indices
    return output


def aligned_plane_from_rt(rvec: np.ndarray, tvec: np.ndarray, reference_normal: np.ndarray | None = None) -> dict[str, Any]:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    normal = rotation[:, 2].astype(np.float64)
    normal /= np.linalg.norm(normal)
    d = -float(normal @ t)
    if reference_normal is not None and float(normal @ reference_normal) < 0.0:
        normal = -normal
        d = -d
    return {"rvec": np.asarray(rvec, dtype=np.float64).reshape(3), "tvec": t, "rotation": rotation, "normal": normal, "d": d}


def reprojection_for_pose(obj: np.ndarray, corners: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, distortion)
    projected = projected.reshape(-1, 2)
    residual = projected - np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    depths = (np.asarray(obj, dtype=np.float64) @ rotation.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3))[:, 2]
    return projected, rmse, depths


def solve_iterative(obj: np.ndarray, corners: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, corners, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return {"status": "solve_failed", "method": "iterative_lm"}
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(obj, corners, k, distortion, rvec, tvec)
        projected, rmse, depths = reprojection_for_pose(obj, corners, rvec, tvec, k, distortion)
        pose = aligned_plane_from_rt(rvec, tvec)
        pose.update({"status": "ok" if np.all(depths > 0) else "negative_depth", "method": "iterative_lm", "projected": projected, "reprojection_rmse_px": rmse, "depth_min_mm": float(np.min(depths))})
        return pose
    except cv2.error as exc:
        return {"status": f"opencv_error:{exc.__class__.__name__}", "method": "iterative_lm"}


def solve_ippe(obj: np.ndarray, corners: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    if not hasattr(cv2, "SOLVEPNP_IPPE") or not hasattr(cv2, "solvePnPGeneric"):
        return {"status": "unsupported", "method": "ippe"}
    try:
        result = cv2.solvePnPGeneric(obj, corners, k, distortion, flags=cv2.SOLVEPNP_IPPE)
        rvecs, tvecs = result[1], result[2]
        candidates: list[dict[str, Any]] = []
        for rvec, tvec in zip(rvecs, tvecs):
            projected, rmse, depths = reprojection_for_pose(obj, corners, rvec, tvec, k, distortion)
            pose = aligned_plane_from_rt(rvec, tvec)
            pose.update({"status": "ok" if np.all(depths > 0) else "negative_depth", "method": "ippe", "projected": projected, "reprojection_rmse_px": rmse, "depth_min_mm": float(np.min(depths))})
            candidates.append(pose)
        if not candidates:
            return {"status": "no_candidate", "method": "ippe"}
        positive = [candidate for candidate in candidates if candidate["status"] == "ok"]
        return min(positive or candidates, key=lambda candidate: float(candidate["reprojection_rmse_px"]))
    except cv2.error as exc:
        return {"status": f"opencv_error:{exc.__class__.__name__}", "method": "ippe"}


def solve_homography(obj: np.ndarray, corners: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    """Calibrated plane homography decomposition, diagnostic only."""
    try:
        object_xy = np.asarray(obj[:, :2], dtype=np.float64)
        normalized = cv2.undistortPoints(np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2), k, distortion).reshape(-1, 2)
        homography, inlier = cv2.findHomography(object_xy, normalized, method=0)
        if homography is None:
            return {"status": "homography_failed", "method": "homography"}
        h1, h2, h3 = homography[:, 0], homography[:, 1], homography[:, 2]
        scale = 1.0 / max(float(np.linalg.norm(h1)), 1.0e-12)
        candidates: list[dict[str, Any]] = []
        for sign in (1.0, -1.0):
            r1, r2, t = sign * scale * h1, sign * scale * h2, sign * scale * h3
            r3 = np.cross(r1, r2)
            raw_rotation = np.column_stack([r1, r2, r3])
            u, _, vt = np.linalg.svd(raw_rotation)
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
            rvec, _ = cv2.Rodrigues(rotation)
            projected, rmse, depths = reprojection_for_pose(obj, corners, rvec, t, k, distortion)
            pose = aligned_plane_from_rt(rvec, t)
            pose.update({"status": "ok" if np.all(depths > 0) else "negative_depth", "method": "homography", "projected": projected, "reprojection_rmse_px": rmse, "depth_min_mm": float(np.min(depths)), "homography_inlier_fraction": float(np.mean(np.asarray(inlier).reshape(-1) > 0)) if inlier is not None else math.nan})
            candidates.append(pose)
        positive = [candidate for candidate in candidates if candidate["status"] == "ok"]
        return min(positive or candidates, key=lambda candidate: float(candidate["reprojection_rmse_px"]))
    except (cv2.error, np.linalg.LinAlgError, ValueError) as exc:
        return {"status": f"error:{exc.__class__.__name__}", "method": "homography"}


def pnp_subset_row(
    frame_id: str,
    subset: str,
    indices: np.ndarray,
    obj: np.ndarray,
    corners: np.ndarray,
    uv: np.ndarray,
    lambda_full: np.ndarray,
    full_pose: Mapping[str, Any],
    k: np.ndarray,
    distortion: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if subset == "full":
        pose = dict(full_pose)
        pose["method"] = "full_iterative_lm"
        pose["status"] = "ok"
        pose["projected"] = np.asarray(full_pose["projected"])
        pose["reprojection_rmse_px"] = float(full_pose["reprojection_rmse_px"])
        pose["depth_min_mm"] = float(full_pose["depth_min_mm"])
    else:
        solved = solve_iterative(obj[indices], corners[indices], k, distortion)
        if solved.get("status") not in {"ok", "negative_depth"}:
            return ({"frame_id": frame_id, "subset": subset, "subset_point_count": int(len(indices)), "status": solved.get("status", "failed"), "solver": "iterative_lm"}, None)
        pose = solved
        pose["method"] = "iterative_lm"
    normal = np.asarray(pose["normal"], dtype=np.float64)
    d = float(pose["d"])
    reference_normal = np.asarray(full_pose["normal"], dtype=np.float64)
    if float(normal @ reference_normal) < 0.0:
        normal = -normal
        d = -d
    truth = fixed.coverage.plane_ray_truth(uv[:, 0], uv[:, 1], normal, d, k, distortion)
    lambda_subset = np.asarray(truth["points"], dtype=np.float64)[:, 2]
    valid = np.asarray(truth["valid"], dtype=bool) & np.isfinite(lambda_subset) & np.isfinite(lambda_full)
    delta = lambda_subset[valid] - lambda_full[valid]
    angle = float(np.degrees(np.arccos(np.clip(float(normal @ reference_normal), -1.0, 1.0))))
    plane_delta = float(d - float(full_pose["d"]))
    delta_stats = stats(delta)
    row = {
        "frame_id": frame_id,
        "is_frame027": frame_id == TARGET_FRAME,
        "subset": subset,
        "solver": "full_iterative_lm" if subset == "full" else "iterative_lm",
        "subset_point_count": int(len(indices)),
        "status": pose.get("status", "ok"),
        "valid_lambda_count": int(np.count_nonzero(valid)),
        "normal_angle_diff_deg": angle,
        "plane_distance_diff_mm": plane_delta,
        "abs_plane_distance_diff_mm": abs(plane_delta),
        "subset_reprojection_rmse_px": float(pose["reprojection_rmse_px"]),
        "subset_depth_min_mm": float(pose["depth_min_mm"]),
        "lambda_delta_bias_mm": delta_stats["bias"],
        "lambda_delta_rmse_mm": delta_stats["rmse"],
        "lambda_delta_p95_abs_mm": delta_stats["p95_abs"],
        "lambda_delta_max_abs_mm": delta_stats["max_abs"],
    }
    pose["normal_aligned"] = normal
    pose["d_aligned"] = d
    pose["lambda"] = lambda_subset
    pose["valid"] = valid
    return row, pose


def solver_row(frame_id: str, method: str, solved: Mapping[str, Any], full_pose: Mapping[str, Any], uv: np.ndarray, lambda_full: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> dict[str, Any]:
    base = {"frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "method": method, "status": solved.get("status", "failed")}
    if solved.get("status") not in {"ok", "negative_depth"}:
        return base
    normal = np.asarray(solved["normal"], dtype=np.float64)
    d = float(solved["d"])
    reference_normal = np.asarray(full_pose["normal"], dtype=np.float64)
    if float(normal @ reference_normal) < 0.0:
        normal = -normal
        d = -d
    truth = fixed.coverage.plane_ray_truth(uv[:, 0], uv[:, 1], normal, d, k, distortion)
    lamb = np.asarray(truth["points"], dtype=np.float64)[:, 2]
    valid = np.asarray(truth["valid"], dtype=bool) & np.isfinite(lamb) & np.isfinite(lambda_full)
    delta_stats = stats(lamb[valid] - lambda_full[valid])
    base.update({"valid_lambda_count": int(np.count_nonzero(valid)), "solver_reprojection_rmse_px": float(solved["reprojection_rmse_px"]), "solver_depth_min_mm": float(solved["depth_min_mm"]), "normal_angle_diff_deg": float(np.degrees(np.arccos(np.clip(float(normal @ reference_normal), -1.0, 1.0)))), "plane_distance_diff_mm": float(d - float(full_pose["d"])), "abs_plane_distance_diff_mm": abs(float(d - float(full_pose["d"]))), "lambda_delta_bias_mm": delta_stats["bias"], "lambda_delta_rmse_mm": delta_stats["rmse"], "lambda_delta_p95_abs_mm": delta_stats["p95_abs"], "lambda_delta_max_abs_mm": delta_stats["max_abs"], "homography_inlier_fraction": solved.get("homography_inlier_fraction", math.nan)})
    return base


def corner_rows(frame_id: str, obj: np.ndarray, corners: np.ndarray, projected: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    center = np.array([(BOARD_COLS - 1) * SQUARE_MM / 2.0, (BOARD_ROWS - 1) * SQUARE_MM / 2.0])
    for i, (object_point, observed, prediction) in enumerate(zip(obj, corners, projected)):
        delta = np.asarray(prediction) - np.asarray(observed)
        board_xy = object_point[:2]
        centered = board_xy - center
        radial = float(np.linalg.norm(centered) / max(np.linalg.norm(center), 1.0e-12))
        rows.append({"frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "corner_index": i, "board_x_mm": float(board_xy[0]), "board_y_mm": float(board_xy[1]), "observed_u_px": float(observed[0]), "observed_v_px": float(observed[1]), "projected_u_px": float(prediction[0]), "projected_v_px": float(prediction[1]), "du_px": float(delta[0]), "dv_px": float(delta[1]), "residual_norm_px": float(np.linalg.norm(delta)), "board_x_normalized": float(centered[0] / center[0]), "board_y_normalized": float(centered[1] / center[1]), "board_radial_normalized": radial, "left_right": "left" if board_xy[0] < center[0] else ("right" if board_xy[0] > center[0] else "center"), "top_bottom": "top" if board_xy[1] < center[1] else ("bottom" if board_xy[1] > center[1] else "center")})
    return rows


def pattern_summary(corner_rows_all: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(corner_rows_all)
    du = np.asarray([float(r["du_px"]) for r in rows])
    dv = np.asarray([float(r["dv_px"]) for r in rows])
    radial = np.asarray([float(r["board_radial_normalized"]) for r in rows])
    mag = np.asarray([float(r["residual_norm_px"]) for r in rows])
    def mean_for(field: str, value: str, outcome: str) -> float:
        values = np.asarray([float(r[outcome]) for r in rows if r[field] == value], dtype=np.float64)
        return float(np.mean(values)) if len(values) else math.nan
    rho_radial, p_radial, _ = spearman_pair(radial, mag)
    return {"left_du_mean_px": mean_for("left_right", "left", "du_px"), "right_du_mean_px": mean_for("left_right", "right", "du_px"), "left_dv_mean_px": mean_for("left_right", "left", "dv_px"), "right_dv_mean_px": mean_for("left_right", "right", "dv_px"), "top_du_mean_px": mean_for("top_bottom", "top", "du_px"), "bottom_du_mean_px": mean_for("top_bottom", "bottom", "du_px"), "top_dv_mean_px": mean_for("top_bottom", "top", "dv_px"), "bottom_dv_mean_px": mean_for("top_bottom", "bottom", "dv_px"), "left_right_du_difference_px": mean_for("left_right", "right", "du_px") - mean_for("left_right", "left", "du_px"), "top_bottom_dv_difference_px": mean_for("top_bottom", "bottom", "dv_px") - mean_for("top_bottom", "top", "dv_px"), "radial_magnitude_spearman_rho": rho_radial, "radial_magnitude_spearman_p_value": p_radial, "corner_du_rmse_px": float(np.sqrt(np.mean(du * du))), "corner_dv_rmse_px": float(np.sqrt(np.mean(dv * dv))), "corner_residual_rmse_px": float(np.sqrt(np.mean(mag * mag)))}


def frame_summary_row(frame_id: str, item: Mapping[str, Any], subset_rows: Sequence[Mapping[str, Any]], solver_rows: Sequence[Mapping[str, Any]], pattern: Mapping[str, Any]) -> dict[str, Any]:
    valid_subsets = [r for r in subset_rows if r.get("status") in {"ok", "negative_depth"} and r.get("subset") != "full"]
    valid_solvers = [r for r in solver_rows if r.get("status") in {"ok", "negative_depth"} and r.get("method") != "full_iterative_lm"]
    cone_error = -np.asarray(item["residual"], dtype=np.float64)  # lambda_cone - lambda_truth, user convention
    cone = stats(cone_error)
    def max_field(rows: Sequence[Mapping[str, Any]], name: str) -> float:
        values = [safe_float(row.get(name)) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        return float(max(values)) if values else math.nan
    def median_field(rows: Sequence[Mapping[str, Any]], name: str) -> float:
        values = [safe_float(row.get(name)) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        return float(np.median(values)) if values else math.nan
    return {"row_type": "frame", "frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "laser_point_count": int(len(item.get("uv_truth", item["residual"]))), "cone_valid_point_count": int(len(item["residual"])), "full_pnp_rmse_px": float(item["pose"].reprojection_rmse_px), "cone_e_lambda_bias_mm": cone["bias"], "cone_e_lambda_rmse_mm": cone["rmse"], "cone_e_lambda_p95_abs_mm": cone["p95_abs"], "cone_a_frame_mm": -safe_float(item.get("a_frame_mm", math.nan)), "cone_k_frame_mm_per_normalized_stripe": -safe_float(item.get("k_frame_mm_per_normalized_stripe", math.nan)), "subset_count_ok": len(valid_subsets), "solver_count_ok": len(valid_solvers), "subset_normal_angle_max_deg": max_field(valid_subsets, "normal_angle_diff_deg"), "subset_plane_distance_max_abs_mm": max_field(valid_subsets, "abs_plane_distance_diff_mm"), "subset_lambda_delta_rmse_max_mm": max_field(valid_subsets, "lambda_delta_rmse_mm"), "subset_lambda_delta_p95_max_mm": max_field(valid_subsets, "lambda_delta_p95_abs_mm"), "subset_lambda_delta_max_abs_max_mm": max_field(valid_subsets, "lambda_delta_max_abs_mm"), "subset_lambda_delta_rmse_median_mm": median_field(valid_subsets, "lambda_delta_rmse_mm"), "solver_lambda_delta_rmse_max_mm": max_field(valid_solvers, "lambda_delta_rmse_mm"), "solver_lambda_delta_p95_max_mm": max_field(valid_solvers, "lambda_delta_p95_abs_mm"), "solver_normal_angle_max_deg": max_field(valid_solvers, "normal_angle_diff_deg"), "solver_plane_distance_max_abs_mm": max_field(valid_solvers, "abs_plane_distance_diff_mm"), **dict(pattern)}


def aggregate_correlations(frame_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("subset_lambda_delta_p95_max_mm", "subset_lambda_delta_rmse_max_mm", "solver_lambda_delta_p95_max_mm", "solver_lambda_delta_rmse_max_mm")
    outcomes = ("cone_e_lambda_rmse_mm", "cone_e_lambda_p95_abs_mm", "cone_a_frame_mm", "cone_k_frame_mm_per_normalized_stripe")
    output: dict[str, Any] = {}
    for field in fields:
        for outcome in outcomes:
            rho, pvalue, n = spearman_pair([safe_float(row.get(field)) for row in frame_rows], [safe_float(row.get(outcome)) for row in frame_rows])
            output[f"spearman_{field}_vs_{outcome}_rho"] = rho
            output[f"spearman_{field}_vs_{outcome}_p_value"] = pvalue
            output[f"spearman_{field}_vs_{outcome}_n"] = n
    return output


def plot_outputs(output: Path, frame_rows: Sequence[Mapping[str, Any]], subset_rows: Sequence[Mapping[str, Any]], corner_rows_all: Sequence[Mapping[str, Any]]) -> None:
    frames = sorted({str(row["frame_id"]) for row in frame_rows}, key=int)
    x = np.arange(len(frames))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for subset in ("left_half", "right_half", "top_half", "bottom_half", "central_subset"):
        values = [next((safe_float(row.get("lambda_delta_p95_abs_mm")) for row in subset_rows if row.get("frame_id") == frame and row.get("subset") == subset), math.nan) for frame in frames]
        ax.plot(x, values, marker=".", linewidth=1.0, label=subset)
    ax.axhline(0, color="black", linewidth=.6); ax.set_xticks(x, frames, rotation=60, fontsize=7); ax.set_ylabel("P95 |delta lambda_truth| / mm"); ax.set_title("PnP corner-subset truth instability by frame (red = 027)"); ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8); ax.scatter([frames.index(TARGET_FRAME)], [max(safe_float(r.get("lambda_delta_p95_abs_mm")) for r in subset_rows if r.get("frame_id") == TARGET_FRAME and r.get("subset") != "full")], color="#c53030", zorder=5); fig.tight_layout(); fig.savefig(output / "p95_delta_lambda_truth_by_frame.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5)); colors = ["#c53030" if row.get("is_frame027") else "#2b6cb0" for row in frame_rows]; ax.scatter([safe_float(row.get("subset_lambda_delta_rmse_max_mm")) for row in frame_rows], [safe_float(row.get("cone_e_lambda_rmse_mm")) for row in frame_rows], c=colors, s=34); ax.axhline(0, color="black", linewidth=.6); ax.set(xlabel="max subset truth RMSE / mm", ylabel="frozen Cone e_lambda RMSE / mm", title="Truth instability versus observed Cone residual"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "truth_instability_vs_cone_residual_rmse.png", dpi=180); plt.close(fig)

    # Mean residual vectors at each checkerboard corner across FIT frames.
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in corner_rows_all:
        grouped.setdefault(int(row["corner_index"]), []).append(row)
    obj_by_index = {index: (float(rows[0]["board_x_mm"]), float(rows[0]["board_y_mm"])) for index, rows in grouped.items()}
    mean_du = {index: float(np.mean([safe_float(r["du_px"]) for r in rows])) for index, rows in grouped.items()}
    mean_dv = {index: float(np.mean([safe_float(r["dv_px"]) for r in rows])) for index, rows in grouped.items()}
    fig, ax = plt.subplots(figsize=(8, 5.5)); xs = np.asarray([obj_by_index[i][0] for i in sorted(obj_by_index)]); ys = np.asarray([obj_by_index[i][1] for i in sorted(obj_by_index)]); us = np.asarray([mean_du[i] for i in sorted(obj_by_index)]); vs = np.asarray([mean_dv[i] for i in sorted(obj_by_index)]); ax.quiver(xs, ys, us, vs, np.hypot(us, vs), cmap="viridis", angles="xy", scale_units="xy", scale=0.03); ax.set(xlabel="board X / mm", ylabel="board Y / mm", title="Mean full-board PnP reprojection residual vector field"); ax.invert_yaxis(); ax.grid(alpha=.2); fig.colorbar(ax.collections[0], ax=ax, label="mean residual magnitude / px"); fig.tight_layout(); fig.savefig(output / "corner_reprojection_residual_vector_field.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for subset in ("left_half", "right_half", "top_half", "bottom_half", "central_subset"):
        values_n = [next((safe_float(row.get("normal_angle_diff_deg")) for row in subset_rows if row.get("frame_id") == frame and row.get("subset") == subset), math.nan) for frame in frames]
        values_d = [next((safe_float(row.get("abs_plane_distance_diff_mm")) for row in subset_rows if row.get("frame_id") == frame and row.get("subset") == subset), math.nan) for frame in frames]
        axes[0].plot(x, values_n, marker=".", linewidth=.9, label=subset); axes[1].plot(x, values_d, marker=".", linewidth=.9, label=subset)
    for ax, ylabel in zip(axes, ("normal difference / deg", "plane distance difference / mm")):
        ax.set_xticks(x, frames, rotation=60, fontsize=7); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[0].set_title("Subset plane normal consistency"); axes[1].set_title("Subset plane distance consistency"); axes[1].legend(ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(output / "plane_normal_distance_consistency_by_subset.png", dpi=180); plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    x = safe_float(value)
    return "n/a" if not math.isfinite(x) else f"{x:.{digits}f}"


def classify(frame_rows: Sequence[Mapping[str, Any]], subset_rows: Sequence[Mapping[str, Any]], solver_rows: Sequence[Mapping[str, Any]], pattern: Mapping[str, Any]) -> dict[str, Any]:
    ok_subsets = [row for row in subset_rows if row.get("status") in {"ok", "negative_depth"} and row.get("subset") != "full"]
    ok_solvers = [row for row in solver_rows if row.get("status") in {"ok", "negative_depth"} and row.get("method") != "full_iterative_lm"]
    max_subset_p95 = max((safe_float(row.get("lambda_delta_p95_abs_mm")) for row in ok_subsets), default=math.nan)
    max_subset_rmse = max((safe_float(row.get("lambda_delta_rmse_mm")) for row in ok_subsets), default=math.nan)
    max_solver_p95 = max((safe_float(row.get("lambda_delta_p95_abs_mm")) for row in ok_solvers), default=math.nan)
    max_solver_rmse = max((safe_float(row.get("lambda_delta_rmse_mm")) for row in ok_solvers), default=math.nan)
    cone_rmse = np.asarray([safe_float(row.get("cone_e_lambda_rmse_mm")) for row in frame_rows])
    cone_median = float(np.nanmedian(cone_rmse))
    cone_max = float(np.nanmax(cone_rmse))
    correlations = aggregate_correlations(frame_rows)
    rho_candidates = [abs(safe_float(value)) for key, value in correlations.items() if key.endswith("_rho") and math.isfinite(safe_float(value))]
    max_abs_rho = max(rho_candidates, default=math.nan)
    geometry_ok = bool(len(frame_rows) == 30 and len(ok_subsets) >= 30 * 5 and len(ok_solvers) >= 30 and all(safe_float(row.get("full_pnp_rmse_px")) <= .40 for row in frame_rows))
    # Declared descriptive gates: truth p95 <0.02 mm and solver p95 <0.02 mm
    # is a strong stability margin relative to the observed 0.05--0.30 mm.
    if not geometry_ok:
        verdict = "D. INSUFFICIENT"
    elif math.isfinite(max_subset_p95) and math.isfinite(max_solver_p95) and max_subset_p95 < .02 and max_solver_p95 < .02 and (not math.isfinite(max_abs_rho) or max_abs_rho < .35):
        verdict = "A. STRONG"
    elif math.isfinite(max_subset_p95) and math.isfinite(max_solver_p95) and max_subset_p95 < .05 and max_solver_p95 < .05:
        verdict = "B. MODERATE"
    else:
        verdict = "C. WEAK"
    if verdict == "A. STRONG":
        next_step = "continue trusting current multi-pose truth"
    elif verdict in {"B. MODERATE", "C. WEAK"}:
        next_step = "upgrade PnP truth construction"
    else:
        next_step = "design a new same-exposure independent validation experiment"
    return {"verdict": verdict, "geometry_ok": geometry_ok, "max_subset_lambda_delta_p95_mm": max_subset_p95, "max_subset_lambda_delta_rmse_mm": max_subset_rmse, "max_solver_lambda_delta_p95_mm": max_solver_p95, "max_solver_lambda_delta_rmse_mm": max_solver_rmse, "median_cone_e_lambda_rmse_mm": cone_median, "max_cone_e_lambda_rmse_mm": cone_max, "max_abs_frame_correlation": max_abs_rho, "corner_pattern_left_right_du_difference_px": pattern.get("left_right_du_difference_px", math.nan), "corner_pattern_top_bottom_dv_difference_px": pattern.get("top_bottom_dv_difference_px", math.nan), "next_step": next_step, **correlations}


def render_report(data_root: Path, frame_rows: Sequence[Mapping[str, Any]], subset_rows: Sequence[Mapping[str, Any]], solver_rows: Sequence[Mapping[str, Any]], corner_pattern: Mapping[str, Any], decision: Mapping[str, Any], frozen_info: Mapping[str, Any]) -> str:
    target = next(row for row in frame_rows if row.get("frame_id") == TARGET_FRAME)
    subset_names = ("left_half", "right_half", "top_half", "bottom_half", "central_subset")
    lines = ["# Task 6C — Multi-pose PnP truth stability audit", "", f"`PNP_TRUTH_STABILITY = {decision['verdict']}`", "", "## Scope and boundary", "", f"- FIT-only frames: `001–018`, `025–036` ({len(frame_rows)} frames, including retained sensitivity frame `027`). Only explicit FIT files were opened; Validation 019–024 and 037–040 were not read.", f"- Formal intrinsics/distortion and the existing 11×8, 20 mm PnP/Steger path were reused. No formal intrinsics were changed, no laser surface was fitted, no frame was deleted, and no correction was created.", f"- Frozen provenance SHA-256: `{frozen_info['provenance_sha256']}`; formal Cone SHA-256: `{frozen_info['formal_cone_sha256']}`. Cone is used only as the observed residual reference.", "", "## Full-board baseline and subset definitions", "", "- Full-board baseline is the current `solvePnP(ITERATIVE)` + `solvePnPRefineLM` result from the existing detector. Each subset uses the same detected corner coordinates and the same laser pixels as that frame's full baseline.", "- Truth-only lambda statistics use all extracted centers with valid full-board ray-plane intersections; the frozen Cone z-range mask is not used to select truth pixels. Cone-valid points are reported separately for the residual reference.", "- Subsets: `left_half`, `right_half`, `top_half`, `bottom_half` (center row/column retained in neighboring halves), and `central_subset` (x=40–160 mm, y=20–100 mm). All have rank-2 spatial support.", "- Truth delta convention: `delta_lambda_truth = lambda_subset − lambda_full`. Frozen Cone reference convention here: `e_lambda = lambda_cone − lambda_truth_full`.", "", "## Stability summary", "", f"- Maximum full-board PnP RMSE: **{fmt(max(safe_float(row['full_pnp_rmse_px']) for row in frame_rows))} px**.", f"- Maximum subset P95 |delta lambda_truth|: **{fmt(decision['max_subset_lambda_delta_p95_mm'])} mm**; maximum subset RMSE: **{fmt(decision['max_subset_lambda_delta_rmse_mm'])} mm**.", f"- Solver diagnostic maximum P95 |delta lambda_truth|: **{fmt(decision['max_solver_lambda_delta_p95_mm'])} mm**; maximum solver RMSE: **{fmt(decision['max_solver_lambda_delta_rmse_mm'])} mm**.", f"- Frozen Cone e_lambda RMSE across frames: median **{fmt(decision['median_cone_e_lambda_rmse_mm'])} mm**, maximum **{fmt(decision['max_cone_e_lambda_rmse_mm'])} mm**.", f"- Frame 027: subset truth P95 max **{fmt(target['subset_lambda_delta_p95_max_mm'])} mm**, solver P95 max **{fmt(target['solver_lambda_delta_p95_max_mm'])} mm**, Cone e_lambda RMSE **{fmt(target['cone_e_lambda_rmse_mm'])} mm**, a_frame **{fmt(target['cone_a_frame_mm'])} mm**, k_frame **{fmt(target['cone_k_frame_mm_per_normalized_stripe'])} mm/normalized stripe**.", "", "| subset | median lambda RMSE / mm | max lambda RMSE / mm | median P95 / mm | max P95 / mm | max normal angle / deg | max plane distance / mm |", "|---|---:|---:|---:|---:|---:|---:|"]
    for subset in subset_names:
        rows = [row for row in subset_rows if row.get("subset") == subset and row.get("status") in {"ok", "negative_depth"}]
        lines.append(f"| {subset} | {fmt(np.median([safe_float(r.get('lambda_delta_rmse_mm')) for r in rows]))} | {fmt(np.max([safe_float(r.get('lambda_delta_rmse_mm')) for r in rows]))} | {fmt(np.median([safe_float(r.get('lambda_delta_p95_abs_mm')) for r in rows]))} | {fmt(np.max([safe_float(r.get('lambda_delta_p95_abs_mm')) for r in rows]))} | {fmt(np.max([safe_float(r.get('normal_angle_diff_deg')) for r in rows]))} | {fmt(np.max([safe_float(r.get('abs_plane_distance_diff_mm')) for r in rows]))} |")
    lines += ["", "## Planar solver comparison", "", "| method | successful frames | median lambda RMSE / mm | max lambda RMSE / mm | max P95 / mm | max normal angle / deg |", "|---|---:|---:|---:|---:|---:|"]
    for method in ("ippe", "homography"):
        rows = [row for row in solver_rows if row.get("method") == method and row.get("status") in {"ok", "negative_depth"}]
        lines.append(f"| {method} | {len(rows)} | {fmt(np.median([safe_float(r.get('lambda_delta_rmse_mm')) for r in rows])) if rows else 'n/a'} | {fmt(np.max([safe_float(r.get('lambda_delta_rmse_mm')) for r in rows])) if rows else 'n/a'} | {fmt(np.max([safe_float(r.get('lambda_delta_p95_abs_mm')) for r in rows])) if rows else 'n/a'} | {fmt(np.max([safe_float(r.get('normal_angle_diff_deg')) for r in rows])) if rows else 'n/a'} |")
    lines += ["", "## Reprojection residual field", "", f"- Mean left/right du difference: **{fmt(corner_pattern.get('left_right_du_difference_px'),5)} px**; mean top/bottom dv difference: **{fmt(corner_pattern.get('top_bottom_dv_difference_px'),5)} px**; radial magnitude Spearman rho: **{fmt(corner_pattern.get('radial_magnitude_spearman_rho'),5)}** (p={fmt(corner_pattern.get('radial_magnitude_spearman_p_value'),4)}).", "- The complete per-corner du/dv field is in `corner_reprojection_field.csv`; vector-field and subset consistency figures are generated without fitting a correction.", "", "## Truth instability versus observed Cone residual", "", f"- Maximum absolute frame-level correlation between truth-instability metrics and Cone RMSE/bias/a/k: **{fmt(decision['max_abs_frame_correlation'],4)}**; detailed Spearman values are in `frame_truth_uncertainty_summary.csv`.", "- This comparison is quantitative only: it does not assign truth uncertainty to a correction term.", "", "## Answers", "", f"1. Full-board PnP is subset-sensitive at the reported scale: **{decision['verdict'] != 'A. STRONG'}**; see all subset rows and plane metrics.", f"2. IPPE/homography produce materially different lambda_truth: **{safe_float(decision['max_solver_lambda_delta_p95_mm']) >= .02}** under the 0.02 mm diagnostic margin.", f"3. PnP truth uncertainty is sufficient to explain the observed frame effect: **{decision['verdict'] == 'C. WEAK'}**; observed Cone RMSE range is approximately 0.05–0.3 mm while the detailed truth deltas are in the CSV.", f"4. Pose-related reprojection pattern detected: **{abs(safe_float(corner_pattern.get('left_right_du_difference_px'))) > .02 or abs(safe_float(corner_pattern.get('top_bottom_dv_difference_px'))) > .02 or abs(safe_float(corner_pattern.get('radial_magnitude_spearman_rho'))) >= .35}** under declared descriptive margins.", f"5. Next step: **{decision['next_step']}**.", "", "## Conclusion", "", f"`PNP_TRUTH_STABILITY = {decision['verdict']}`.", "Classification gates are declared descriptive gates: STRONG requires all subset/solver P95 deltas <0.02 mm and no meaningful frame correlation; MODERATE requires maxima <0.05 mm; WEAK means truth instability is at the observed residual scale or clearly correlated; otherwise INSUFFICIENT.", "", "Generated figures: `p95_delta_lambda_truth_by_frame.png`, `truth_instability_vs_cone_residual_rmse.png`, `corner_reprojection_residual_vector_field.png`, and `plane_normal_distance_consistency_by_subset.png`.", ""]
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
    subsets = subset_specs(obj)
    board_summaries, processed = board_audit.process_groups_board(groups, intrinsics, calibration, reconstruction_params, frozen_model)
    k = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    distortion = np.asarray(intrinsics.dist_coeffs, dtype=np.float64)
    subset_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    corner_rows_all: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for base_summary in board_summaries:
        frame_id = str(base_summary["frame_id"])
        item = processed[frame_id]
        pose = item["pose"]
        corners = np.asarray(item["corners"], dtype=np.float64)
        projected, full_rmse, depths = reprojection_for_pose(obj, corners, pose.rvec, pose.tvec, k, distortion)
        full_pose = aligned_plane_from_rt(pose.rvec, pose.tvec, np.asarray(pose.normal, dtype=np.float64))
        full_pose.update({"normal": np.asarray(pose.normal, dtype=np.float64), "d": float(pose.d), "projected": projected, "reprojection_rmse_px": full_rmse, "depth_min_mm": float(np.min(depths))})
        corner_rows_all.extend(corner_rows(frame_id, obj, corners, projected))
        raw_uv = np.asarray(item["uv_raw"], dtype=np.float64)
        raw_points = np.asarray(item["truth_points_raw"], dtype=np.float64)
        raw_valid = np.asarray(item["truth_valid_raw"], dtype=bool)
        truth_valid = raw_valid & np.isfinite(raw_points[:, 2])
        uv = raw_uv[truth_valid]
        lambda_full = raw_points[truth_valid, 2]
        # Keep the all-ray truth count separate from the frozen-Cone-valid
        # residual array retained by the shared 6A processor.
        item["uv_truth"] = uv
        local_subset_rows: list[dict[str, Any]] = []
        for subset, indices in subsets.items():
            row, _ = pnp_subset_row(frame_id, subset, indices, obj, corners, uv, lambda_full, full_pose, k, distortion)
            subset_rows.append(row)
            local_subset_rows.append(row)
        for method, solver in (("full_iterative_lm", full_pose), ("ippe", solve_ippe(obj, corners, k, distortion)), ("homography", solve_homography(obj, corners, k, distortion))):
            if method == "full_iterative_lm":
                solver = dict(solver)
                solver["status"] = "ok"
                solver["method"] = method
            solver_rows.append(solver_row(frame_id, method, solver, full_pose, uv, lambda_full, k, distortion))
        local_solver_rows = [row for row in solver_rows if row.get("frame_id") == frame_id]
        pattern_rows = [row for row in corner_rows_all if row.get("frame_id") == frame_id]
        frame_rows.append(frame_summary_row(frame_id, {**item, "pose": pose, "a_frame_mm": base_summary.get("a_frame_mm"), "k_frame_mm_per_normalized_stripe": base_summary.get("k_frame_mm_per_normalized_stripe")}, local_subset_rows, local_solver_rows, pattern_summary(pattern_rows)))
    corner_pattern = pattern_summary(corner_rows_all)
    decision = classify(frame_rows, subset_rows, solver_rows, corner_pattern)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pnp_subset_truth_stability.csv", subset_rows)
    write_csv(output_dir / "pnp_solver_comparison.csv", solver_rows)
    write_csv(output_dir / "corner_reprojection_field.csv", corner_rows_all)
    summary_rows = list(frame_rows) + [{"row_type": "aggregate", **decision, **corner_pattern}]
    write_csv(output_dir / "frame_truth_uncertainty_summary.csv", summary_rows)
    plot_outputs(output_dir, frame_rows, subset_rows, corner_rows_all)
    (output_dir / "report.md").write_text(render_report(data_root, frame_rows, subset_rows, solver_rows, corner_pattern, decision, frozen_info), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "frame_count": len(frame_rows), "subset_rows": len(subset_rows), "solver_rows": len(solver_rows), "corner_rows": len(corner_rows_all)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
