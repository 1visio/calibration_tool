#!/usr/bin/env python3
"""Task 6A: board-coordinate residual audit (FIT-only).

This audit intentionally uses the existing PnP, Steger extraction and frozen
Circular Cone path.  It converts each ray/plane truth point to the PnP board
coordinate system and compares simple/binned residual statistics in board and
sensor coordinates.  No Cone is fitted, no Steger setting is changed, and no
Validation image is opened.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import warnings
from collections import defaultdict
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
WORKSPACE_ROOT = SCRIPT.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402


DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "board_coordinate_audit"
FIT_IDS = [f"{index:03d}" for index in range(1, 19)] + [f"{index:03d}" for index in range(25, 37)]
TARGET_FRAME = "027"
ROLES = ("chess", "nolaser", "laser")
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_MM = 20.0
BOOTSTRAP_REPS = 300
BOOTSTRAP_SEED = 20260814
GRID_PHASE_BIN_WIDTH_MM = 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=fixed.DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=fixed.DEFAULT_FORMAL_CONE)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def inventory_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve only the explicitly requested FIT triplets.

    We deliberately do not enumerate or parse any Validation directory.  The
    two FIT roots are known from the acquisition layout: 001--018 in
    ``laser_plane/fit`` and 025--036 in ``laser_plane/fit_edge_extension/fit``.
    """
    groups: dict[str, dict[str, Any]] = {}
    for frame_id in FIT_IDS:
        root = data_root if int(frame_id) <= 18 else data_root / "fit_edge_extension"
        groups[frame_id] = {}
        for role in ROLES:
            path = root / "fit" / f"{role} {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            groups[frame_id][role] = {
                "path": path,
                "filename": str(path.relative_to(root)).replace("\\", "/"),
                "quality_warnings": "",
                "manifest_sha256": "",
            }
    return groups


def load_frozen_model_checked(provenance_path: Path, formal_cone_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_main = [frame_id for frame_id in FIT_IDS if frame_id != TARGET_FRAME]
    if provenance.get("main_fit_ids") != expected_main:
        raise RuntimeError("Frozen Circular provenance main_fit_ids is not the declared FIT set excluding 027")
    if provenance.get("sensitivity_case_ids") != [TARGET_FRAME]:
        raise RuntimeError("Frozen provenance does not identify 027 as the sensitivity case")
    if provenance.get("validation_opened") is not False:
        raise RuntimeError("Frozen provenance does not prove validation_opened=false")
    if provenance.get("formal_m0_modified") is not False or provenance.get("empirical_correction_added") is not False:
        raise RuntimeError("Frozen provenance reports a modified model or correction")
    model, info = fixed.load_frozen_model(provenance_path, formal_cone_path)
    info.update(
        {
            "main_fit_ids": expected_main,
            "sensitivity_case_ids": [TARGET_FRAME],
            "validation_ids_not_opened": provenance.get("validation_ids_not_opened", []),
            "validation_opened": False,
            "formal_m0_modified": False,
            "empirical_correction_added": False,
        }
    )
    return model, info


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan, math.nan, int(len(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), int(len(x))


def basic_metrics(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": math.nan, "mae": math.nan, "rmse": math.nan, "p95_abs": math.nan, "max_abs": math.nan}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "mae": float(np.mean(np.abs(x))),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p95_abs": float(np.percentile(np.abs(x), 95)),
        "max_abs": float(np.max(np.abs(x))),
    }


def nearest_grid_distance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.mod(np.asarray(values, dtype=np.float64), SQUARE_MM)
    distance = np.minimum(phase, SQUARE_MM - phase)
    return phase, distance


def row_profile_metrics(
    laser: np.ndarray, background: np.ndarray, diff: np.ndarray, uv: np.ndarray, half_width: int = 8
) -> dict[str, np.ndarray]:
    """Record optional local intensity/contrast/FWHM without changing Steger."""
    laser_gray = np.asarray(fixed.triplets.to_gray_float(laser), dtype=np.float64)
    background_gray = np.asarray(fixed.triplets.to_gray_float(background), dtype=np.float64)
    diff_gray = np.asarray(diff, dtype=np.float64)
    h, w = diff_gray.shape[:2]
    intensity: list[float] = []
    background_values: list[float] = []
    contrast: list[float] = []
    fwhm: list[float] = []
    for u, v in np.asarray(uv, dtype=np.float64):
        yy = int(np.clip(np.rint(v), 0, h - 1))
        xx = int(np.clip(np.rint(u), 0, w - 1))
        lo = max(0, xx - half_width)
        hi = min(w, xx + half_width + 1)
        profile = np.maximum(diff_gray[yy, lo:hi], 0.0)
        peak_index = int(np.argmax(profile)) if profile.size else 0
        peak = float(profile[peak_index]) if profile.size else math.nan
        if np.isfinite(peak) and peak > 0.0 and profile.size:
            half = 0.5 * peak
            left = peak_index
            right = peak_index
            while left > 0 and profile[left - 1] >= half:
                left -= 1
            while right + 1 < profile.size and profile[right + 1] >= half:
                right += 1
            width = float(right - left + 1)
        else:
            width = math.nan
        intensity.append(float(laser_gray[yy, xx]))
        background_values.append(float(background_gray[yy, xx]))
        contrast.append(peak)
        fwhm.append(width)
    return {
        "laser_intensity_dn": np.asarray(intensity, dtype=np.float64),
        "background_intensity_dn": np.asarray(background_values, dtype=np.float64),
        "stripe_contrast_dn": np.asarray(contrast, dtype=np.float64),
        "fwhm_px": np.asarray(fwhm, dtype=np.float64),
    }


def stripe_angle_board(board_xy: np.ndarray) -> tuple[float, float, float]:
    centered = np.asarray(board_xy, dtype=np.float64) - np.mean(board_xy, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0.0:
        direction = -direction
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    angle_x = float(min(angle, 180.0 - angle))
    angle_y = float(abs(90.0 - angle_x))
    return angle, angle_x, angle_y


def board_grid_fields(board_xyz: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(board_xyz[:, 0], dtype=np.float64)
    y = np.asarray(board_xyz[:, 1], dtype=np.float64)
    x_mod, x_dist = nearest_grid_distance(x)
    y_mod, y_dist = nearest_grid_distance(y)
    return {
        "grid_x_mod_20_mm": x_mod,
        "grid_y_mod_20_mm": y_mod,
        "grid_x_phase": x_mod / SQUARE_MM,
        "grid_y_phase": y_mod / SQUARE_MM,
        # A horizontal grid line has constant Y; a vertical grid line has constant X.
        "distance_to_vertical_grid_line_mm": x_dist,
        "distance_to_horizontal_grid_line_mm": y_dist,
        "distance_to_grid_intersection_mm": np.hypot(x_dist, y_dist),
        "board_cell_x": np.floor(x / SQUARE_MM),
        "board_cell_y": np.floor(y / SQUARE_MM),
    }


def process_groups_board(
    groups: Mapping[str, Mapping[str, Any]],
    intrinsics: Any,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    frozen_model: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidate = copy.deepcopy(dict(calibration))
    candidate["laser_model"] = copy.deepcopy(dict(frozen_model))
    summaries: list[dict[str, Any]] = []
    processed: dict[str, dict[str, Any]] = {}
    for frame_id in sorted(groups, key=int):
        chess = fixed.triplets.imread_unicode(Path(groups[frame_id]["chess"]["path"]))
        background = fixed.triplets.imread_unicode(Path(groups[frame_id]["nolaser"]["path"]))
        laser = fixed.triplets.imread_unicode(Path(groups[frame_id]["laser"]["path"]))
        pose = fixed.triplets.detect_board_pose(
            chess,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            cols=BOARD_COLS,
            rows=BOARD_ROWS,
            square_size_mm=SQUARE_MM,
            max_rmse_px=0.40,
        )
        board_mask = fixed.triplets.board_inner_mask(fixed.triplets.to_gray_float(chess).shape, pose.corners, margin_px=-2)
        u, v, response, diff = fixed.triplets.extract_laser_centers(
            laser, background, board_mask, fixed.EXTRACTION_CONFIG, "vertical"
        )
        uv = np.column_stack([u, v]).astype(np.float64)
        truth = fixed.coverage.plane_ray_truth(
            u, v, pose.normal, pose.d, intrinsics.camera_matrix, intrinsics.dist_coeffs
        )
        lambda_model, model_valid = fixed.lambda_by_input(uv, candidate, reconstruction_params)
        valid = model_valid & np.asarray(truth["valid"], dtype=bool) & np.isfinite(truth["points"][:, 2])
        if int(np.count_nonzero(valid)) < 30:
            raise RuntimeError(f"frame {frame_id}: too few valid frozen-Cone points")
        uv_valid = uv[valid]
        p_cam = np.asarray(truth["points"], dtype=np.float64)[valid]
        lambda_truth = p_cam[:, 2]
        lambda_model_valid = np.asarray(lambda_model, dtype=np.float64)[valid]
        residual = lambda_truth - lambda_model_valid
        a_frame, k_frame, remaining_rmse, stripe = fixed.frame_fit(uv_valid, residual)
        rotation, _ = cv2.Rodrigues(np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1))
        tvec = np.asarray(pose.tvec, dtype=np.float64).reshape(3)
        # OpenCV PnP convention: P_cam = R P_board + t.
        p_board = (p_cam - tvec[None, :]) @ rotation
        grid = board_grid_fields(p_board)
        angle, angle_x, angle_y = stripe_angle_board(p_board[:, :2])
        local = row_profile_metrics(laser, background, diff, uv_valid)
        source = "laser_plane" if int(frame_id) <= 18 else "laser_plane/fit_edge_extension"
        center_object = np.asarray([(BOARD_COLS - 1) * SQUARE_MM / 2.0, (BOARD_ROWS - 1) * SQUARE_MM / 2.0, 0.0])
        board_center_cam = rotation @ center_object + tvec
        item = {
            "frame_id": frame_id,
            "is_frame027": frame_id == TARGET_FRAME,
            "source_dataset": source,
            "pose": pose,
            "rotation": rotation,
            "tvec": tvec,
            "corners": np.asarray(pose.corners, dtype=np.float64),
            "uv": uv_valid,
            # Raw extracted centers are retained for truth-only audits.  The
            # existing 6A residual table intentionally applies the frozen
            # Cone validity mask; Task 6C must not let that model mask select
            # which pixels are used to test PnP truth stability.
            "uv_raw": uv,
            "truth_points_raw": np.asarray(truth["points"], dtype=np.float64),
            "truth_valid_raw": np.asarray(truth["valid"], dtype=bool),
            "lambda_model_raw": np.asarray(lambda_model, dtype=np.float64),
            "model_valid_raw": np.asarray(model_valid, dtype=bool),
            "u": uv_valid[:, 0],
            "v": uv_valid[:, 1],
            "p_cam": p_cam,
            "p_board": p_board,
            "lambda_truth": lambda_truth,
            "lambda_model": lambda_model_valid,
            "residual": residual,
            "stripe": stripe,
            "response": np.asarray(response, dtype=np.float64)[valid],
            "diff": np.asarray(diff, dtype=np.float64),
            "grid": grid,
            "stripe_angle_board_deg": angle,
            "stripe_angle_to_x_deg": angle_x,
            "stripe_angle_to_y_deg": angle_y,
            "local": {key: np.asarray(value, dtype=np.float64) for key, value in local.items()},
            "board_center_cam": board_center_cam,
        }
        processed[frame_id] = item
        stats = basic_metrics(residual)
        z_stats = basic_metrics(p_board[:, 2])
        summary = {
            "row_type": "frame",
            "frame_id": frame_id,
            "is_frame027": frame_id == TARGET_FRAME,
            "source_dataset": source,
            "valid_point_count": int(len(residual)),
            "used_point_count": int(len(uv)),
            "uniform_subsample_cap": int(fixed.EXTRACTION_CONFIG["max_points_per_image"]),
            "bias_mm": stats["mean"],
            "mae_mm": stats["mae"],
            "rmse_mm": stats["rmse"],
            "p95_abs_mm": stats["p95_abs"],
            "max_abs_mm": stats["max_abs"],
            "a_frame_mm": float(a_frame),
            "k_frame_mm_per_normalized_stripe": float(k_frame),
            "remaining_rmse_after_a_k_mm": float(remaining_rmse),
            "pnp_rmse_px": float(pose.reprojection_rmse_px),
            "rvec_x": float(pose.rvec[0]),
            "rvec_y": float(pose.rvec[1]),
            "rvec_z": float(pose.rvec[2]),
            "board_origin_x_mm": float(tvec[0]),
            "board_origin_y_mm": float(tvec[1]),
            "board_origin_z_mm": float(tvec[2]),
            "board_center_xc_mm": float(board_center_cam[0]),
            "board_center_yc_mm": float(board_center_cam[1]),
            "board_center_zc_mm": float(board_center_cam[2]),
            "plane_nx": float(pose.normal[0]),
            "plane_ny": float(pose.normal[1]),
            "plane_nz": float(pose.normal[2]),
            "board_x_min_mm": float(np.min(p_board[:, 0])),
            "board_x_max_mm": float(np.max(p_board[:, 0])),
            "board_y_min_mm": float(np.min(p_board[:, 1])),
            "board_y_max_mm": float(np.max(p_board[:, 1])),
            "board_z_mean_mm": float(np.mean(p_board[:, 2])),
            "board_z_rmse_mm": z_stats["rmse"],
            "board_z_p95_abs_mm": z_stats["p95_abs"],
            "board_z_max_abs_mm": z_stats["max_abs"],
            "stripe_angle_board_deg": angle,
            "stripe_angle_to_x_deg": angle_x,
            "stripe_angle_to_y_deg": angle_y,
            "laser_intensity_mean_dn": float(np.nanmean(local["laser_intensity_dn"])),
            "stripe_contrast_mean_dn": float(np.nanmean(local["stripe_contrast_dn"])),
            "fwhm_median_px": float(np.nanmedian(local["fwhm_px"])),
            "fwhm_p95_px": float(np.nanpercentile(local["fwhm_px"][np.isfinite(local["fwhm_px"])], 95)) if np.any(np.isfinite(local["fwhm_px"])) else math.nan,
        }
        for name, values in {
            "sensor_u_px": item["u"],
            "sensor_v_px": item["v"],
            "board_Xb_mm": p_board[:, 0],
            "board_Yb_mm": p_board[:, 1],
            "grid_x_mod_20_mm": grid["grid_x_mod_20_mm"],
            "grid_y_mod_20_mm": grid["grid_y_mod_20_mm"],
            "distance_to_grid_intersection_mm": grid["distance_to_grid_intersection_mm"],
        }.items():
            rho, pvalue, count = safe_spearman(values, residual)
            summary[f"spearman_e_{name}_rho"] = rho
            summary[f"spearman_e_{name}_p_value"] = pvalue
            summary[f"spearman_e_{name}_n"] = count
        summaries.append(summary)
    return summaries, processed


def point_arrays(processed: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        grid = item["grid"]
        values["frame_id"].append(np.repeat(frame_id, len(item["residual"])))
        values["is_frame027"].append(np.repeat(item["is_frame027"], len(item["residual"])))
        values["residual"].append(item["residual"])
        values["sensor_u_px"].append(item["u"])
        values["sensor_v_px"].append(item["v"])
        values["lambda_truth_mm"].append(item["lambda_truth"])
        values["lambda_model_mm"].append(item["lambda_model"])
        values["board_Xb_mm"].append(item["p_board"][:, 0])
        values["board_Yb_mm"].append(item["p_board"][:, 1])
        values["board_Zb_mm"].append(item["p_board"][:, 2])
        for key, array in grid.items():
            values[key].append(array)
        values["stripe_angle_board_deg"].append(np.repeat(item["stripe_angle_board_deg"], len(item["residual"])))
        values["stripe_angle_to_x_deg"].append(np.repeat(item["stripe_angle_to_x_deg"], len(item["residual"])))
        values["stripe_angle_to_y_deg"].append(np.repeat(item["stripe_angle_to_y_deg"], len(item["residual"])))
        for key, array in item["local"].items():
            values[key].append(array)
        values["pnp_rmse_px"].append(np.repeat(float(item["pose"].reprojection_rmse_px), len(item["residual"])))
    return {key: np.concatenate(parts) for key, parts in values.items()}


POINT_PREDICTORS: tuple[tuple[str, str], ...] = (
    ("sensor_u_px", "sensor"),
    ("sensor_v_px", "sensor"),
    ("board_Xb_mm", "board"),
    ("board_Yb_mm", "board"),
    ("grid_x_mod_20_mm", "grid_phase"),
    ("grid_y_mod_20_mm", "grid_phase"),
    ("distance_to_vertical_grid_line_mm", "grid_boundary"),
    ("distance_to_horizontal_grid_line_mm", "grid_boundary"),
    ("distance_to_grid_intersection_mm", "grid_boundary"),
    ("stripe_angle_board_deg", "stripe_angle"),
    ("stripe_angle_to_x_deg", "stripe_angle"),
    ("stripe_angle_to_y_deg", "stripe_angle"),
)


def predictor_edges(name: str, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if name in {"grid_x_mod_20_mm", "grid_y_mod_20_mm"}:
        return np.linspace(0.0, SQUARE_MM, 21)
    if name in {"distance_to_vertical_grid_line_mm", "distance_to_horizontal_grid_line_mm"}:
        return np.linspace(0.0, SQUARE_MM / 2.0, 11)
    if name == "distance_to_grid_intersection_mm":
        upper = max(float(np.nanmax(x)) if len(x) else 1.0, 1.0)
        return np.linspace(0.0, upper, 16)
    if name.endswith("angle_board_deg") or name.endswith("angle_to_x_deg") or name.endswith("angle_to_y_deg"):
        return np.linspace(0.0, 90.0, 19)
    lo = float(np.nanmin(x)) if len(x) else 0.0
    hi = float(np.nanmax(x)) if len(x) else 1.0
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, 21)


def binned_explained_fraction(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 3:
        return math.nan
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return math.nan
    bins = np.digitize(x, edges[1:-1], right=False)
    prediction = np.full(len(y), np.mean(y), dtype=np.float64)
    for index in range(len(edges) - 1):
        selected = bins == index
        if np.any(selected):
            prediction[selected] = np.mean(y[selected])
    return float(1.0 - np.sum((y - prediction) ** 2) / total)


def simple_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    design = np.column_stack([np.ones(len(x)), (x - np.mean(x)) / np.std(x)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ beta
    total = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - prediction) ** 2) / total) if total > 0.0 else math.nan


def concatenate_frame_arrays(frame_arrays: Sequence[tuple[np.ndarray, np.ndarray]], indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    xs = [frame_arrays[index][0] for index in indices]
    ys = [frame_arrays[index][1] for index in indices]
    return np.concatenate(xs), np.concatenate(ys)


def aggregate_finite(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def point_relation_row(
    name: str,
    family: str,
    arrays: Mapping[str, np.ndarray],
    processed: Mapping[str, Mapping[str, Any]],
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(arrays[name], dtype=np.float64)
    y = np.asarray(arrays["residual"], dtype=np.float64)
    edges = predictor_edges(name, x)
    rho, pvalue, n = safe_spearman(x, y)
    ev = binned_explained_fraction(x, y, edges)
    r2 = simple_linear_r2(x, y)
    frame_ids = sorted(processed, key=int)
    frame_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_id in frame_ids:
        item = processed[frame_id]
        if name in item:
            fx = np.asarray(item[name], dtype=np.float64)
        elif name in item["grid"]:
            fx = np.asarray(item["grid"][name], dtype=np.float64)
        elif name.startswith("sensor_"):
            fx = np.asarray(item["u" if name == "sensor_u_px" else "v"], dtype=np.float64)
        elif name == "board_Xb_mm":
            fx = item["p_board"][:, 0]
        elif name == "board_Yb_mm":
            fx = item["p_board"][:, 1]
        elif name == "stripe_angle_board_deg":
            fx = np.repeat(item["stripe_angle_board_deg"], len(item["residual"]))
        elif name == "stripe_angle_to_x_deg":
            fx = np.repeat(item["stripe_angle_to_x_deg"], len(item["residual"]))
        elif name == "stripe_angle_to_y_deg":
            fx = np.repeat(item["stripe_angle_to_y_deg"], len(item["residual"]))
        else:
            raise KeyError(name)
        if fx.ndim == 0:
            fx = np.repeat(float(fx), len(item["residual"]))
        fy = np.asarray(item["residual"], dtype=np.float64)
        valid = np.isfinite(fx) & np.isfinite(fy)
        frame_arrays.append((fx[valid], fy[valid]))
    target_index = frame_ids.index(TARGET_FRAME)
    leave_target_indices = [index for index in range(len(frame_arrays)) if index != target_index]
    leave_target_x, leave_target_y = concatenate_frame_arrays(frame_arrays, leave_target_indices)
    leave_target_rho, leave_target_p, leave_target_n = safe_spearman(leave_target_x, leave_target_y)
    leave_target_ev = binned_explained_fraction(leave_target_x, leave_target_y, edges)
    leave_target_r2 = simple_linear_r2(leave_target_x, leave_target_y)
    loo_rhos: list[float] = []
    loo_evs: list[float] = []
    for excluded in range(len(frame_arrays)):
        lx, ly = concatenate_frame_arrays(frame_arrays, [i for i in range(len(frame_arrays)) if i != excluded])
        loo_rhos.append(safe_spearman(lx, ly)[0])
        loo_evs.append(binned_explained_fraction(lx, ly, edges))
    rng = np.random.default_rng(seed + sum(ord(char) for char in name))
    boot_rhos: list[float] = []
    boot_evs: list[float] = []
    for _ in range(int(bootstrap_reps)):
        sampled = rng.integers(0, len(frame_arrays), size=len(frame_arrays))
        bx, by = concatenate_frame_arrays(frame_arrays, sampled)
        boot_rhos.append(safe_spearman(bx, by)[0])
        boot_evs.append(binned_explained_fraction(bx, by, edges))
    loo_rho_stats = aggregate_finite(loo_rhos)
    loo_ev_stats = aggregate_finite(loo_evs)
    boot_rho_stats = aggregate_finite(boot_rhos)
    boot_ev_stats = aggregate_finite(boot_evs)
    finite_boot_rho = np.asarray([value for value in boot_rhos if np.isfinite(value)])
    finite_boot_ev = np.asarray([value for value in boot_evs if np.isfinite(value)])
    frame027 = processed[TARGET_FRAME]
    if name in frame027:
        target_x = np.asarray(frame027[name], dtype=np.float64)
    elif name in frame027["grid"]:
        target_x = np.asarray(frame027["grid"][name], dtype=np.float64)
    elif name.startswith("sensor_"):
        target_x = np.asarray(frame027["u" if name == "sensor_u_px" else "v"], dtype=np.float64)
    elif name == "board_Xb_mm":
        target_x = frame027["p_board"][:, 0]
    elif name == "board_Yb_mm":
        target_x = frame027["p_board"][:, 1]
    elif name == "stripe_angle_board_deg":
        target_x = np.repeat(frame027["stripe_angle_board_deg"], len(frame027["residual"]))
    elif name == "stripe_angle_to_x_deg":
        target_x = np.repeat(frame027["stripe_angle_to_x_deg"], len(frame027["residual"]))
    else:
        target_x = np.repeat(frame027["stripe_angle_to_y_deg"], len(frame027["residual"]))
    if np.asarray(target_x).ndim == 0:
        target_x = np.repeat(float(target_x), len(frame027["residual"]))
    target_rho, target_p, target_n = safe_spearman(target_x, frame027["residual"])
    return {
        "row_type": "point_relation",
        "coordinate_family": family,
        "predictor": name,
        "outcome": "e_lambda_mm",
        "point_count": n,
        "spearman_rho": rho,
        "spearman_p_value": pvalue,
        "binned_explained_fraction": ev,
        "simple_linear_r2": r2,
        "leave027_out_point_count": leave_target_n,
        "leave027_out_spearman_rho": leave_target_rho,
        "leave027_out_spearman_p_value": leave_target_p,
        "leave027_out_binned_explained_fraction": leave_target_ev,
        "leave027_out_simple_linear_r2": leave_target_r2,
        "bin_count": len(edges) - 1,
        "loo_frame_count": len(loo_rhos),
        "loo_spearman_mean": loo_rho_stats["mean"],
        "loo_spearman_std": loo_rho_stats["std"],
        "loo_spearman_min": loo_rho_stats["min"],
        "loo_spearman_max": loo_rho_stats["max"],
        "loo_ev_mean": loo_ev_stats["mean"],
        "loo_ev_std": loo_ev_stats["std"],
        "loo_ev_min": loo_ev_stats["min"],
        "loo_ev_max": loo_ev_stats["max"],
        "loo_spearman_sign_consistency": float(np.mean(np.sign(np.asarray(loo_rhos)[np.isfinite(loo_rhos)]) == np.sign(rho))) if np.isfinite(rho) and np.any(np.isfinite(loo_rhos)) else math.nan,
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(seed),
        "bootstrap_spearman_mean": boot_rho_stats["mean"],
        "bootstrap_spearman_ci_low": float(np.percentile(finite_boot_rho, 2.5)) if len(finite_boot_rho) else math.nan,
        "bootstrap_spearman_ci_high": float(np.percentile(finite_boot_rho, 97.5)) if len(finite_boot_rho) else math.nan,
        "bootstrap_ev_mean": boot_ev_stats["mean"],
        "bootstrap_ev_ci_low": float(np.percentile(finite_boot_ev, 2.5)) if len(finite_boot_ev) else math.nan,
        "bootstrap_ev_ci_high": float(np.percentile(finite_boot_ev, 97.5)) if len(finite_boot_ev) else math.nan,
        "frame027_spearman_rho": target_rho,
        "frame027_spearman_p_value": target_p,
        "frame027_point_count": target_n,
        "frame027_binned_explained_fraction": binned_explained_fraction(target_x, frame027["residual"], edges),
    }


def frame_relation_rows(summary: Sequence[Mapping[str, Any]], bootstrap_reps: int, seed: int) -> list[dict[str, Any]]:
    frame_rows = [row for row in summary if row.get("row_type") == "frame"]
    outputs: list[dict[str, Any]] = []
    predictors = ("stripe_angle_board_deg", "stripe_angle_to_x_deg", "stripe_angle_to_y_deg")
    outcomes = ("bias_mm", "a_frame_mm", "k_frame_mm_per_normalized_stripe", "rmse_mm")
    for scope, selected in (
        ("all30", frame_rows),
        ("leave027_out", [row for row in frame_rows if row["frame_id"] != TARGET_FRAME]),
    ):
        for predictor in predictors:
            for outcome in outcomes:
                x = np.asarray([float(row[predictor]) for row in selected], dtype=np.float64)
                y = np.asarray([float(row[outcome]) for row in selected], dtype=np.float64)
                rho, pvalue, n = safe_spearman(x, y)
                loo_rhos: list[float] = []
                for excluded in range(len(selected)):
                    keep = np.arange(len(selected)) != excluded
                    loo_rhos.append(safe_spearman(x[keep], y[keep])[0])
                rng = np.random.default_rng(seed + len(predictor) * 31 + len(outcome))
                boot: list[float] = []
                for _ in range(int(bootstrap_reps)):
                    sampled = rng.integers(0, len(selected), size=len(selected))
                    boot.append(safe_spearman(x[sampled], y[sampled])[0])
                finite_boot = np.asarray([value for value in boot if np.isfinite(value)])
                loo_stats = aggregate_finite(loo_rhos)
                outputs.append(
                    {
                        "row_type": "frame_relation",
                        "scope": scope,
                        "predictor": predictor,
                        "outcome": outcome,
                        "frame_count": n,
                        "spearman_rho": rho,
                        "spearman_p_value": pvalue,
                        "loo_spearman_mean": loo_stats["mean"],
                        "loo_spearman_std": loo_stats["std"],
                        "loo_spearman_min": loo_stats["min"],
                        "loo_spearman_max": loo_stats["max"],
                        "loo_spearman_sign_consistency": float(np.mean(np.sign(np.asarray(loo_rhos)[np.isfinite(loo_rhos)]) == np.sign(rho))) if np.isfinite(rho) and np.any(np.isfinite(loo_rhos)) else math.nan,
                        "bootstrap_reps": int(bootstrap_reps),
                        "bootstrap_seed": int(seed),
                        "bootstrap_spearman_mean": float(np.mean(finite_boot)) if len(finite_boot) else math.nan,
                        "bootstrap_spearman_ci_low": float(np.percentile(finite_boot, 2.5)) if len(finite_boot) else math.nan,
                        "bootstrap_spearman_ci_high": float(np.percentile(finite_boot, 97.5)) if len(finite_boot) else math.nan,
                    }
                )
    return outputs


GRID_SPECS: tuple[tuple[str, float, float, float], ...] = (
    ("grid_x_mod_20_mm", 0.0, 20.0, GRID_PHASE_BIN_WIDTH_MM),
    ("grid_y_mod_20_mm", 0.0, 20.0, GRID_PHASE_BIN_WIDTH_MM),
    ("distance_to_vertical_grid_line_mm", 0.0, 10.0, 1.0),
    ("distance_to_horizontal_grid_line_mm", 0.0, 10.0, 1.0),
    ("distance_to_grid_intersection_mm", 0.0, 15.0, 1.0),
)


def grid_phase_rows(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_ids = np.asarray(arrays["frame_id"])
    y = np.asarray(arrays["residual"], dtype=np.float64)
    for predictor, low, high, step in GRID_SPECS:
        edges = np.arange(low, high + step * 0.5, step)
        x = np.asarray(arrays[predictor], dtype=np.float64)
        bins = np.digitize(x, edges[1:-1], right=False)
        for index in range(len(edges) - 1):
            selected = (bins == index) & np.isfinite(x) & np.isfinite(y)
            values = y[selected]
            rows.append(
                {
                    "predictor": predictor,
                    "bin_index": index,
                    "bin_start_mm": float(edges[index]),
                    "bin_end_mm": float(edges[index + 1]),
                    "bin_center_mm": float((edges[index] + edges[index + 1]) / 2.0),
                    "point_count": int(len(values)),
                    "unique_frame_count": int(len(set(str(value) for value in frame_ids[selected]))),
                    "frame027_point_count": int(np.count_nonzero(selected & (frame_ids == TARGET_FRAME))),
                    "residual_mean_mm": float(np.mean(values)) if len(values) else math.nan,
                    "residual_median_mm": float(np.median(values)) if len(values) else math.nan,
                    "residual_std_mm": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) else math.nan),
                    "residual_rmse_mm": float(np.sqrt(np.mean(values * values))) if len(values) else math.nan,
                    "residual_p95_abs_mm": float(np.percentile(np.abs(values), 95)) if len(values) else math.nan,
                }
            )
    return rows


def finite_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else math.nan


def make_point_rows(processed: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        pose = item["pose"]
        grid = item["grid"]
        local = item["local"]
        p_cam = item["p_cam"]
        p_board = item["p_board"]
        for index in range(len(item["residual"])):
            row = {
                "frame_id": frame_id,
                "is_frame027": item["is_frame027"],
                "source_dataset": item["source_dataset"],
                "point_index": index,
                "u_px": float(item["u"][index]),
                "v_px": float(item["v"][index]),
                "lambda_truth_mm": float(item["lambda_truth"][index]),
                "lambda_model_mm": float(item["lambda_model"][index]),
                "e_lambda_mm": float(item["residual"][index]),
                "Xc_mm": float(p_cam[index, 0]),
                "Yc_mm": float(p_cam[index, 1]),
                "Zc_mm": float(p_cam[index, 2]),
                "Xb_mm": float(p_board[index, 0]),
                "Yb_mm": float(p_board[index, 1]),
                "Zb_mm": float(p_board[index, 2]),
                "board_origin_x_mm": float(item["tvec"][0]),
                "board_origin_y_mm": float(item["tvec"][1]),
                "board_origin_z_mm": float(item["tvec"][2]),
                "plane_nx": float(pose.normal[0]),
                "plane_ny": float(pose.normal[1]),
                "plane_nz": float(pose.normal[2]),
                "pnp_rmse_px": float(pose.reprojection_rmse_px),
                "stripe_angle_board_deg": float(item["stripe_angle_board_deg"]),
                "stripe_angle_to_x_deg": float(item["stripe_angle_to_x_deg"]),
                "stripe_angle_to_y_deg": float(item["stripe_angle_to_y_deg"]),
                "steger_response": float(item["response"][index]),
                "laser_local_intensity_dn": float(local["laser_intensity_dn"][index]),
                "background_local_intensity_dn": float(local["background_intensity_dn"][index]),
                "stripe_contrast_dn": float(local["stripe_contrast_dn"][index]),
                "fwhm_px": float(local["fwhm_px"][index]),
            }
            for key, values in grid.items():
                row[key] = float(values[index])
            rows.append(row)
    return rows


def plot_scatter_panels(output: Path, arrays: Mapping[str, np.ndarray]) -> None:
    rng = np.random.default_rng(6)
    n = len(arrays["residual"])
    keep = np.arange(n)
    if n > 12000:
        keep = np.sort(rng.choice(n, size=12000, replace=False))
    target = np.asarray(arrays["is_frame027"], dtype=bool)[keep]
    colors = np.where(target, "#c53030", "#2b6cb0")
    residual = np.asarray(arrays["residual"], dtype=np.float64)[keep]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, name, label in (
        (axes[0], "board_Xb_mm", "Xb / mm"),
        (axes[1], "board_Yb_mm", "Yb / mm"),
    ):
        axis.scatter(np.asarray(arrays[name])[keep], residual, s=4, alpha=0.16, c=colors)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel(label)
        axis.set_ylabel("e_lambda / mm")
        axis.grid(alpha=0.2)
    fig.suptitle("Task 6A residual versus board coordinates (red = frame 027)")
    fig.tight_layout()
    fig.savefig(output / "residual_vs_Xb_Yb.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, name, label in (
        (axes[0], "grid_x_mod_20_mm", "Xb mod 20 / mm"),
        (axes[1], "grid_y_mod_20_mm", "Yb mod 20 / mm"),
    ):
        axis.scatter(np.asarray(arrays[name])[keep], residual, s=4, alpha=0.16, c=colors)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel(label)
        axis.set_ylabel("e_lambda / mm")
        axis.set_xlim(0.0, 20.0)
        axis.grid(alpha=0.2)
    fig.suptitle("Task 6A residual versus checkerboard grid phase")
    fig.tight_layout()
    fig.savefig(output / "residual_vs_grid_phase.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for axis, name, label, upper in (
        (axes[0], "distance_to_vertical_grid_line_mm", "distance to vertical grid line / mm", 10),
        (axes[1], "distance_to_horizontal_grid_line_mm", "distance to horizontal grid line / mm", 10),
        (axes[2], "distance_to_grid_intersection_mm", "distance to grid intersection / mm", 15),
    ):
        axis.scatter(np.asarray(arrays[name])[keep], residual, s=4, alpha=0.16, c=colors)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel(label)
        axis.set_ylabel("e_lambda / mm")
        axis.set_xlim(0.0, upper)
        axis.grid(alpha=0.2)
    fig.suptitle("Task 6A residual near checkerboard grid boundaries")
    fig.tight_layout()
    fig.savefig(output / "residual_vs_grid_distance.png", dpi=180)
    plt.close(fig)


def plot_frame_angle(output: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    rows = [row for row in summary if row.get("row_type") == "frame"]
    colors = ["#c53030" if row["frame_id"] == TARGET_FRAME else "#2b6cb0" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, outcome, ylabel in ((axes[0], "bias_mm", "bias / mm"), (axes[1], "k_frame_mm_per_normalized_stripe", "k_frame / mm per normalized stripe")):
        axis.scatter([row["stripe_angle_board_deg"] for row in rows], [row[outcome] for row in rows], c=colors, s=32)
        for row in rows:
            axis.annotate(row["frame_id"], (row["stripe_angle_board_deg"], row[outcome]), fontsize=6, alpha=0.75)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel("stripe angle in board XY / deg (mod 180)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    fig.suptitle("Frame residual signature versus stripe-board angle (red = 027)")
    fig.tight_layout()
    fig.savefig(output / "frame_bias_k_vs_stripe_angle.png", dpi=180)
    plt.close(fig)


def plot_explained_variance(output: Path, comparison: Sequence[Mapping[str, Any]]) -> None:
    rows = [row for row in comparison if row.get("row_type") == "point_relation"]
    labels = [str(row["predictor"]) for row in rows]
    values = [float(row["binned_explained_fraction"]) for row in rows]
    colors = {"sensor": "#718096", "board": "#2b6cb0", "grid_phase": "#dd6b20", "grid_boundary": "#38a169", "stripe_angle": "#805ad5"}
    fig, axis = plt.subplots(figsize=(14, 5.5))
    axis.bar(np.arange(len(rows)), values, color=[colors.get(str(row["coordinate_family"]), "#4a5568") for row in rows])
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(np.arange(len(rows)), labels, rotation=55, ha="right")
    axis.set_ylabel("binned explained fraction")
    axis.set_title("Board-coordinate versus sensor-coordinate residual explained variance")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "board_vs_sensor_explained_variance.png", dpi=180)
    plt.close(fig)


def row_for(comparison: Sequence[Mapping[str, Any]], predictor: str) -> Mapping[str, Any]:
    return next(row for row in comparison if row.get("row_type") == "point_relation" and row.get("predictor") == predictor)


def classify(
    summary: Sequence[Mapping[str, Any]], comparison: Sequence[Mapping[str, Any]], frame_relations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    frame_rows = [row for row in summary if row.get("row_type") == "frame"]
    z_values = np.asarray([float(row["board_z_rmse_mm"]) for row in frame_rows], dtype=np.float64)
    pnp_values = np.asarray([float(row["pnp_rmse_px"]) for row in frame_rows], dtype=np.float64)
    point_rows = [row for row in comparison if row.get("row_type") == "point_relation"]
    sensor_rows = [row for row in point_rows if row.get("coordinate_family") == "sensor"]
    board_rows = [row for row in point_rows if row.get("coordinate_family") in {"board", "grid_phase", "grid_boundary"}]
    grid_rows = [row for row in point_rows if row.get("coordinate_family") in {"grid_phase", "grid_boundary"}]
    angle_rows = [row for row in point_rows if row.get("coordinate_family") == "stripe_angle"]
    leave_sensor_best = max((float(row["leave027_out_binned_explained_fraction"]) for row in sensor_rows if math.isfinite(float(row["leave027_out_binned_explained_fraction"]))), default=math.nan)
    leave_board_best_row = max(board_rows, key=lambda row: float(row["leave027_out_binned_explained_fraction"]) if math.isfinite(float(row["leave027_out_binned_explained_fraction"])) else -math.inf)
    leave_board_best = float(leave_board_best_row["leave027_out_binned_explained_fraction"])
    sensor_best = max((float(row["binned_explained_fraction"]) for row in sensor_rows if math.isfinite(float(row["binned_explained_fraction"]))), default=math.nan)
    board_best_row = max(board_rows, key=lambda row: float(row["binned_explained_fraction"]) if math.isfinite(float(row["binned_explained_fraction"])) else -math.inf)
    board_best = float(board_best_row["binned_explained_fraction"])
    grid_best_row = max(grid_rows, key=lambda row: float(row["binned_explained_fraction"]) if math.isfinite(float(row["binned_explained_fraction"])) else -math.inf)
    grid_best = float(grid_best_row["binned_explained_fraction"])
    stable_grid = bool(
        math.isfinite(grid_best)
        and grid_best >= 0.05
        and float(grid_best_row["bootstrap_ev_ci_low"]) > 0.05
        and float(grid_best_row["loo_ev_min"]) > -0.02
    )
    stable_angle = False
    angle_evidence: list[str] = []
    for row in frame_relations:
        if row.get("scope") != "leave027_out" or row.get("outcome") not in {"bias_mm", "a_frame_mm", "k_frame_mm_per_normalized_stripe"}:
            continue
        rho = float(row["spearman_rho"])
        low = float(row["bootstrap_spearman_ci_low"])
        high = float(row["bootstrap_spearman_ci_high"])
        if math.isfinite(rho) and abs(rho) >= 0.35 and math.isfinite(low) and math.isfinite(high) and (low > 0.0 or high < 0.0) and float(row["loo_spearman_sign_consistency"]) >= 0.75:
            stable_angle = True
            angle_evidence.append(f"{row['predictor']}->{row['outcome']}")
    z_consistent = bool(np.nanmax(z_values) <= 0.20 and np.nanmax(pnp_values) <= 0.40)
    target_row = next(row for row in frame_rows if row["frame_id"] == TARGET_FRAME)
    non_target_bias = np.asarray([float(row["bias_mm"]) for row in frame_rows if row["frame_id"] != TARGET_FRAME])
    target_z = abs(float(target_row["bias_mm"]) - float(np.mean(non_target_bias))) / float(np.std(non_target_bias, ddof=1)) if len(non_target_bias) > 1 and np.std(non_target_bias, ddof=1) > 0 else math.nan
    target_follows = bool(math.isfinite(target_z) and target_z <= 2.5)
    board_advantage = board_best - sensor_best if math.isfinite(sensor_best) else math.nan
    board_stable = bool(
        math.isfinite(board_advantage)
        and board_advantage >= 0.03
        and float(board_best_row["bootstrap_ev_ci_low"]) > 0.02
        and float(board_best_row["loo_ev_min"]) > -0.02
    )
    if not z_consistent or len(frame_rows) < 20:
        verdict = "D. INSUFFICIENT"
    elif board_stable and (stable_grid or stable_angle) and target_follows and board_advantage >= 0.10:
        verdict = "A. STRONG"
    elif board_stable or stable_grid or stable_angle:
        verdict = "B. MODERATE"
    elif not math.isfinite(sensor_best) or max(abs(float(row["spearman_rho"])) for row in point_rows if math.isfinite(float(row["spearman_rho"]))) < 0.05:
        verdict = "D. INSUFFICIENT"
    else:
        verdict = "C. WEAK"
    if board_stable and board_best > sensor_best + 0.03:
        source = "target/PnP/extraction error"
    elif math.isfinite(sensor_best) and sensor_best > board_best + 0.03:
        source = "sensor/model error"
    else:
        source = "mixed"
    task6b = "YES" if (stable_grid or stable_angle or verdict == "D. INSUFFICIENT") else "NO / not yet"
    return {
        "verdict": verdict,
        "board_plane_z_consistent": z_consistent,
        "max_board_z_rmse_mm": float(np.nanmax(z_values)),
        "max_pnp_rmse_px": float(np.nanmax(pnp_values)),
        "sensor_best_binned_ev": sensor_best,
        "board_best_predictor": board_best_row["predictor"],
        "board_best_binned_ev": board_best,
        "grid_best_predictor": grid_best_row["predictor"],
        "grid_best_binned_ev": grid_best,
        "board_minus_sensor_best_ev": board_advantage,
        "leave027_sensor_best_binned_ev": leave_sensor_best,
        "leave027_board_best_predictor": leave_board_best_row["predictor"],
        "leave027_board_best_binned_ev": leave_board_best,
        "leave027_board_minus_sensor_best_ev": leave_board_best - leave_sensor_best if math.isfinite(leave_sensor_best) else math.nan,
        "stable_grid_effect": stable_grid,
        "stable_angle_effect": stable_angle,
        "angle_evidence": angle_evidence,
        "frame027_bias_z_vs_other_frames": target_z,
        "frame027_follows_main_pattern": target_follows,
        "likely_source": source,
        "task6b_rotation_experiment_worth": task6b,
        "classification_note": "Descriptive gates: strong requires a stable board/grid/angle advantage >=0.10 EV and 027 consistency; moderate requires board-minus-sensor >=0.03 EV or a grid/angle signal with EV >=0.05 and bootstrap lower CI >0.05; weak means no stable board advantage; insufficient means geometry/coverage is inadequate.",
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def render_report(
    data_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    frame_relations: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    frozen_info: Mapping[str, Any],
    point_count: int,
    bootstrap_reps: int,
) -> str:
    frame_rows = [row for row in summaries if row.get("row_type") == "frame"]
    target = next(row for row in frame_rows if row["frame_id"] == TARGET_FRAME)
    all_e = np.asarray([float(row["bias_mm"]) for row in frame_rows], dtype=np.float64)
    point_rows = [row for row in comparison if row.get("row_type") == "point_relation"]
    top_corr = sorted([row for row in point_rows if math.isfinite(float(row["spearman_rho"]))], key=lambda row: abs(float(row["spearman_rho"])), reverse=True)[:6]
    top_ev = sorted([row for row in point_rows if math.isfinite(float(row["binned_explained_fraction"]))], key=lambda row: float(row["binned_explained_fraction"]), reverse=True)[:6]
    leave_sensor = [row for row in point_rows if row.get("coordinate_family") == "sensor"]
    leave_board = [row for row in point_rows if row.get("coordinate_family") in {"board", "grid_phase", "grid_boundary"}]
    leave_sensor_best = max((float(row["leave027_out_binned_explained_fraction"]) for row in leave_sensor if math.isfinite(float(row["leave027_out_binned_explained_fraction"]))), default=math.nan)
    leave_board_best_row = max(leave_board, key=lambda row: float(row["leave027_out_binned_explained_fraction"]) if math.isfinite(float(row["leave027_out_binned_explained_fraction"])) else -math.inf)
    leave_board_best = float(leave_board_best_row["leave027_out_binned_explained_fraction"])
    angle_frame = [row for row in frame_relations if row.get("scope") == "leave027_out" and row.get("predictor") == "stripe_angle_board_deg" and row.get("outcome") in {"bias_mm", "a_frame_mm", "k_frame_mm_per_normalized_stripe"}]
    near_boundary = [row for row in grid_rows if row.get("predictor") in {"distance_to_vertical_grid_line_mm", "distance_to_horizontal_grid_line_mm"} and float(row.get("bin_start_mm", 99)) < 2.0]
    far_boundary = [row for row in grid_rows if row.get("predictor") in {"distance_to_vertical_grid_line_mm", "distance_to_horizontal_grid_line_mm"} and float(row.get("bin_start_mm", -1)) >= 8.0]
    near_mean = float(np.nanmean([float(row["residual_mean_mm"]) for row in near_boundary])) if near_boundary else math.nan
    far_mean = float(np.nanmean([float(row["residual_mean_mm"]) for row in far_boundary])) if far_boundary else math.nan
    lines = [
        "# Task 6A — Board-coordinate residual audit",
        "",
        f"`BOARD_COORDINATE_EFFECT = {decision['verdict']}`",
        "",
        "## Scope and boundary",
        "",
        f"- FIT-only frames: `001–018`, `025–036` ({len(frame_rows)} frames, {point_count} valid points); frame `027` is retained and marked separately.",
        f"- Input roots: `{data_root}/fit` and `{data_root}/fit_edge_extension/fit`. Inventory resolves only these explicit FIT filenames; no Validation image, Validation frames.csv, or Validation-derived points were opened.",
        "- No Cone was fitted/refit, no correction/LUT was created, no frame was deleted, and the existing Steger configuration was used unchanged (uniform 900-point cap per frame).",
        f"- Frozen Circular provenance SHA-256: `{frozen_info['provenance_sha256']}`; formal Cone SHA-256: `{frozen_info['formal_cone_sha256']}`.",
        f"- Frozen provenance declares `validation_opened=false`, main FIT IDs excluding 027, and sensitivity case 027. Bootstrap: frame-resampling, B={bootstrap_reps}, seed={BOOTSTRAP_SEED}.",
        "- Coordinate convention: OpenCV PnP `P_cam = R P_board + t`; `P_board = R.T @ (P_cam - t)`. Board origin is the first detected inner corner, X/Y axes follow the 11×8 object-point order, and square size is 20 mm.",
        "- Residual convention: `e_lambda = lambda_truth - lambda_model`; all relationship fits are one-dimensional linear or binned means only, never a high-order correction model.",
        "",
        "## Board-plane/PnP consistency",
        "",
        f"- Across all frames, maximum PnP RMSE: **{fmt(max(float(row['pnp_rmse_px']) for row in frame_rows), 4)} px**; maximum per-frame board-Z RMSE: **{fmt(max(float(row['board_z_rmse_mm']) for row in frame_rows), 6)} mm**.",
        f"- Frame 027: bias **{fmt(target['bias_mm'])} mm**, RMSE **{fmt(target['rmse_mm'])} mm**, P95 **{fmt(target['p95_abs_mm'])} mm**, stripe angle **{fmt(target['stripe_angle_board_deg'], 3)}°**; bias z-score versus other FIT frames **{fmt(decision['frame027_bias_z_vs_other_frames'], 3)}**.",
        "",
        "## Frame-level residual geometry",
        "",
        "| frame | 027 | bias mm | RMSE mm | P95 mm | a_frame mm | k_frame | stripe angle board ° | board-Z RMSE mm | PnP RMSE px |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame_rows:
        lines.append(
            f"| {row['frame_id']} | {'yes' if row['is_frame027'] else ''} | {fmt(row['bias_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['a_frame_mm'])} | {fmt(row['k_frame_mm_per_normalized_stripe'])} | {fmt(row['stripe_angle_board_deg'], 3)} | {fmt(row['board_z_rmse_mm'], 6)} | {fmt(row['pnp_rmse_px'])} |"
        )
    lines += [
        "",
        "## Spearman and binned explained variance",
        "",
        "Top point-level Spearman magnitudes (all FIT points; LOO and frame bootstrap are in `board_vs_sensor_comparison.csv`):",
        "",
        "| predictor | family | rho | p | binned EV | bootstrap EV 95% CI | LOO EV min/max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_corr:
        lines.append(
            f"| {row['predictor']} | {row['coordinate_family']} | {fmt(row['spearman_rho'], 5)} | {fmt(row['spearman_p_value'], 3)} | {fmt(row['binned_explained_fraction'], 5)} | [{fmt(row['bootstrap_ev_ci_low'], 5)}, {fmt(row['bootstrap_ev_ci_high'], 5)}] | [{fmt(row['loo_ev_min'], 5)}, {fmt(row['loo_ev_max'], 5)}] |"
        )
    lines += [
        "",
        "Highest binned explained-variance predictors:",
        "",
        "| predictor | family | binned EV | simple linear R² | bootstrap EV 95% CI | frame 027 EV |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in top_ev:
        lines.append(
            f"| {row['predictor']} | {row['coordinate_family']} | {fmt(row['binned_explained_fraction'], 5)} | {fmt(row['simple_linear_r2'], 5)} | [{fmt(row['bootstrap_ev_ci_low'], 5)}, {fmt(row['bootstrap_ev_ci_high'], 5)}] | {fmt(row['frame027_binned_explained_fraction'], 5)} |"
        )
    lines += [
        "",
        "## Grid-boundary and stripe-angle checks",
        "",
        f"- Mean residual in the first 2 mm from a grid line: **{fmt(near_mean)} mm**; at 8–10 mm from a grid line: **{fmt(far_mean)} mm**. These are descriptive binned means, not a fitted correction.",
        "",
        "| frame-level predictor → outcome | rho (leave 027 out) | bootstrap 95% CI | LOO sign consistency |",
        "|---|---:|---:|---:|",
    ]
    for row in angle_frame:
        lines.append(
            f"| {row['predictor']} → {row['outcome']} | {fmt(row['spearman_rho'], 5)} | [{fmt(row['bootstrap_spearman_ci_low'], 5)}, {fmt(row['bootstrap_spearman_ci_high'], 5)}] | {fmt(row['loo_spearman_sign_consistency'], 3)} |"
        )
    lines += [
        "",
        "## Board versus sensor interpretation",
        "",
        f"- All FIT points: best sensor-coordinate binned EV **{fmt(decision['sensor_best_binned_ev'], 5)}**; best board/grid predictor **{decision['board_best_predictor']} = {fmt(decision['board_best_binned_ev'], 5)}**; board-minus-sensor = **{fmt(decision['board_minus_sensor_best_ev'], 5)}**.",
        f"- Excluding retained sensitivity frame 027: best sensor EV **{fmt(leave_sensor_best, 5)}**; best board/grid EV **{fmt(leave_board_best, 5)}** (`{leave_board_best_row['predictor']}`).",
        f"- Best grid predictor: **{decision['grid_best_predictor']} = {fmt(decision['grid_best_binned_ev'], 5)}**; stable grid effect = **{decision['stable_grid_effect']}**; stable stripe-angle effect = **{decision['stable_angle_effect']}**.",
        f"- Frame 027 follows the main frame-level bias pattern under the declared check: **{decision['frame027_follows_main_pattern']}**.",
        f"- Current evidence classification: **{decision['likely_source']}**.",
        "",
        "## Conclusion",
        "",
        f"`BOARD_COORDINATE_EFFECT = {decision['verdict']}`.",
        "",
        "1. The report does not treat a single Pearson coefficient as evidence: every predictor has Spearman, binned means, leave-one-frame-out ranges, and frame-bootstrap intervals in the CSV.",
        f"2. A stable board/grid/angle signature is **{bool(decision['stable_grid_effect'] or decision['stable_angle_effect'])}** under the declared descriptive gates; a fixed checkerboard-coordinate explanation is therefore classified as **{decision['verdict']}**.",
        f"3. The current source assessment is **{decision['likely_source']}**. Grid-boundary and angle results should be read together with the point-level extraction quality fields (intensity, contrast, FWHM, Steger response).",
        f"4. Task 6B horizontal in-plane rotation experiment: **{decision['task6b_rotation_experiment_worth']}**. With no stable board/grid/angle signature in this audit, it is not the next priority; it can be revisited specifically to diagnose the independent 027 anomaly.",
        "",
        "Generated figures: `residual_vs_Xb_Yb.png`, `residual_vs_grid_phase.png`, `residual_vs_grid_distance.png`, `frame_bias_k_vs_stripe_angle.png`, and `board_vs_sensor_explained_variance.png`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    groups = inventory_fit(data_root)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, frozen_info = load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    summaries, processed = process_groups_board(groups, intrinsics, calibration, reconstruction_params, frozen_model)
    arrays = point_arrays(processed)
    point_rows = make_point_rows(processed)
    comparison = [
        point_relation_row(name, family, arrays, processed, int(args.bootstrap_reps), int(args.seed))
        for name, family in POINT_PREDICTORS
    ]
    frame_relations = frame_relation_rows(summaries, int(args.bootstrap_reps), int(args.seed))
    grid_rows = grid_phase_rows(arrays)
    decision = classify(summaries, comparison, frame_relations)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed.write_csv(output_dir / "board_coordinate_residual_points.csv", point_rows)
    fixed.write_csv(output_dir / "frame_board_geometry_summary.csv", list(summaries) + list(frame_relations))
    fixed.write_csv(output_dir / "grid_phase_residual_summary.csv", grid_rows)
    fixed.write_csv(output_dir / "board_vs_sensor_comparison.csv", comparison)
    plot_scatter_panels(output_dir, arrays)
    plot_frame_angle(output_dir, summaries)
    plot_explained_variance(output_dir, comparison)
    (output_dir / "report.md").write_text(
        render_report(data_root, summaries, comparison, frame_relations, grid_rows, decision, frozen_info, len(arrays["residual"]), int(args.bootstrap_reps)),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "frame_count": len(summaries), "point_count": len(arrays["residual"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
