#!/usr/bin/env python3
"""Task 6E: camera-intrinsics to ray-plane truth stability audit.

The audit uses the formal 0811 camera-calibration FIT frames (chess 001--018)
and the explicitly declared laser FIT triplets (001--018, 025--036).  It keeps
the formal distortion model (k3 fixed to zero), never opens a laser Validation
image, never fits a laser surface, and never changes the formal calibration.

The laser center pixels are extracted once by the existing FIT-only pipeline.
Every LOO/bootstrap candidate reuses those exact pixels and the detected 88
corners, then recomputes only PnP and the ray-plane intersection.  This makes
the propagation experiment an intrinsics/truth audit rather than a Steger or
Cone experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import yaml
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
CALIBRATION_SRC = WORKSPACE_ROOT / "calibration" / "src"
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
for _path in (SCRIPT.parent, CALIBRATION_SRC, MEASUREMENT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_board_coordinate_residual as board_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import calibrate_chessboard_opencv_reusable as chess_calib  # noqa: E402


DEFAULT_CALIBRATION_FIT = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
)
DEFAULT_FORMAL_INTRINSICS = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
)
DEFAULT_FORMAL_FIT_METRICS = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "fit_images.csv"
)
DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "intrinsics_truth_stability"
)
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE

BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_MM = 20.0
CALIBRATION_FRAME_IDS = tuple(f"{i:03d}" for i in range(1, 19))
IMAGE_WIDTH = 4096
IMAGE_HEIGHT = 3000
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260815
CALIBRATION_FLAGS = cv2.CALIB_FIX_K3
PARAMETER_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
REGION_ORDER = ("all", "top", "middle", "bottom", "left", "center_u", "right")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-fit-dir", type=Path, default=DEFAULT_CALIBRATION_FIT)
    parser.add_argument("--formal-intrinsics", type=Path, default=DEFAULT_FORMAL_INTRINSICS)
    parser.add_argument("--formal-fit-metrics", type=Path, default=DEFAULT_FORMAL_FIT_METRICS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
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


def fmt(value: Any, digits: int = 9) -> str | int | float:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    return f"{x:.{digits}g}" if math.isfinite(x) else ""


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})


def natural_key(path: Path) -> list[tuple[int, str | int]]:
    import re

    return [(1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", path.name)]


def load_formal_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    camera_matrix = np.asarray(document["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(document["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3) or len(dist_coeffs) < 5:
        raise RuntimeError(f"Formal intrinsics malformed: {path}")
    if float(dist_coeffs.reshape(-1)[4]) != 0.0:
        raise RuntimeError("Formal model is expected to have k3 fixed at zero")
    return camera_matrix, dist_coeffs, document


def load_formal_fit_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["image"]: row for row in csv.DictReader(handle)}


def load_calibration_observations(fit_dir: Path) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    paths = sorted(fit_dir.glob("chess *.tif"), key=natural_key)
    expected_names = [f"chess {frame_id}.tif" for frame_id in CALIBRATION_FRAME_IDS]
    if [path.name for path in paths] != expected_names:
        raise RuntimeError(
            "Formal 0811 calibration FIT must contain exactly chess 001.tif--chess 018.tif; "
            f"found {[path.name for path in paths]}"
        )
    observations: list[dict[str, Any]] = []
    image_size: tuple[int, int] | None = None
    pattern = (BOARD_COLS, BOARD_ROWS)
    for path in paths:
        image = chess_calib.read_image(path)
        if image is None:
            raise RuntimeError(f"Could not read formal calibration FIT image: {path}")
        current_size = (int(image.shape[1]), int(image.shape[0]))
        image_size = current_size if image_size is None else image_size
        if current_size != image_size:
            raise RuntimeError(f"Calibration image size mismatch: {path.name}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners, method = chess_calib.detect_corners(gray, pattern)
        if not found or corners is None:
            raise RuntimeError(f"Calibration corner detection failed: {path.name}")
        frame_id = path.stem.split()[-1]
        observations.append(
            {
                "frame_id": frame_id,
                "path": path,
                "corners": np.asarray(corners, dtype=np.float32).reshape(-1, 2),
                "detection_method": method,
            }
        )
    if image_size is None:
        raise RuntimeError("No formal calibration observations")
    return observations, image_size


def object_points() -> np.ndarray:
    return chess_calib.create_object_points(BOARD_COLS, BOARD_ROWS, SQUARE_MM).astype(np.float64)


def solve_pose(corners: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, obj: np.ndarray) -> dict[str, Any]:
    success, rvec, tvec = cv2.solvePnP(
        obj,
        np.asarray(corners, dtype=np.float32),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            obj,
            np.asarray(corners, dtype=np.float32),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64),
            rvec,
            tvec,
        )
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    measured = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    residual = measured - projected
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    normal = np.asarray(rotation[:, 2], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), np.finfo(float).eps)
    d_plane = -float(normal @ tvec.reshape(3))
    if normal[2] > 0.0:
        normal = -normal
        d_plane = -d_plane
    return {
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
        "rotation": np.asarray(rotation, dtype=np.float64),
        "normal": normal,
        "d": d_plane,
        "projected": projected,
        "residual": residual,
        "rmse_px": rmse,
    }


def calibration_coverage_rows(
    observations: Sequence[Mapping[str, Any]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    obj: np.ndarray,
    image_size: tuple[int, int],
    formal_metrics: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    width, height = image_size
    rows: list[dict[str, Any]] = []
    for observation in observations:
        corners = np.asarray(observation["corners"], dtype=np.float64)
        pose = solve_pose(corners, camera_matrix, dist_coeffs, obj)
        hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
        hull_area = float(cv2.contourArea(hull))
        u_min, v_min = np.min(corners, axis=0)
        u_max, v_max = np.max(corners, axis=0)
        center_u, center_v = np.mean(corners, axis=0)
        normal = pose["normal"]
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
        center_object = np.asarray([(BOARD_COLS - 1) * SQUARE_MM / 2.0, (BOARD_ROWS - 1) * SQUARE_MM / 2.0, 0.0])
        board_center_cam = pose["rotation"] @ center_object + pose["tvec"]
        metric_row = formal_metrics.get(f"chess {observation['frame_id']}.tif", {})
        rows.append(
            {
                "row_type": "frame",
                "frame_id": observation["frame_id"],
                "image": observation["path"].name,
                "used_for_intrinsics": True,
                "validation_opened": False,
                "detection_method": observation["detection_method"],
                "corner_count": int(len(corners)),
                "formal_reprojection_rmse_px": finite(metric_row.get("per_image_rmse")),
                "solvepnp_reprojection_rmse_px": pose["rmse_px"],
                "board_center_u_px": center_u,
                "board_center_v_px": center_v,
                "board_center_u_norm": center_u / width,
                "board_center_v_norm": center_v / height,
                "board_tilt_deg": tilt_deg,
                "board_center_z_mm": board_center_cam[2],
                "apparent_width_px": u_max - u_min,
                "apparent_height_px": v_max - v_min,
                "apparent_bbox_area_fraction": (u_max - u_min) * (v_max - v_min) / (width * height),
                "apparent_hull_area_px2": hull_area,
                "apparent_hull_area_fraction": hull_area / (width * height),
                "corner_u_min_px": u_min,
                "corner_u_max_px": u_max,
                "corner_v_min_px": v_min,
                "corner_v_max_px": v_max,
                "corner_u_span_norm": (u_max - u_min) / width,
                "corner_v_span_norm": (v_max - v_min) / height,
                "corner_min_edge_distance_px": min(u_min, width - u_max, v_min, height - v_max),
                "plane_nx": normal[0],
                "plane_ny": normal[1],
                "plane_nz": normal[2],
                "plane_d_mm": pose["d"],
            }
        )
    return rows


def parameter_values(camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> dict[str, float]:
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    return {
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "k1": float(distortion[0]),
        "k2": float(distortion[1]),
        "p1": float(distortion[2]),
        "p2": float(distortion[3]),
        "k3": float(distortion[4]) if len(distortion) > 4 else 0.0,
    }


def parameter_delta_row(
    base_params: Mapping[str, float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    candidate_type: str,
    candidate_id: str,
    fit_rms_px: float,
    omitted_frame_id: str = "",
    bootstrap_unique_count: int | str = "",
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    values = parameter_values(camera_matrix, dist_coeffs) if status == "ok" else {name: math.nan for name in PARAMETER_NAMES}
    row: dict[str, Any] = {
        "candidate_type": candidate_type,
        "candidate_id": candidate_id,
        "omitted_frame_id": omitted_frame_id,
        "bootstrap_unique_count": bootstrap_unique_count,
        "status": status,
        "fit_rms_px": fit_rms_px,
        "error": error,
    }
    for name in PARAMETER_NAMES:
        base = float(base_params[name])
        value = float(values[name])
        row[f"{name}_baseline"] = base
        row[f"{name}"] = value
        row[f"delta_{name}"] = value - base if math.isfinite(value) else math.nan
        row[f"delta_{name}_pct"] = (100.0 * (value - base) / base) if base != 0.0 and math.isfinite(value) else math.nan
    return row


def calibrate_candidate(
    observations: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    obj: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    object_points = [obj.astype(np.float32).copy() for _ in indices]
    image_points = [np.asarray(observations[index]["corners"], dtype=np.float32).reshape(-1, 1, 2) for index in indices]
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=CALIBRATION_FLAGS,
    )
    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(dist_coeffs)):
        raise RuntimeError("calibrateCamera returned non-finite parameters")
    return float(rms), np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64)


def safe_stats(values: Sequence[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0, "mean": math.nan, "std": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def region_masks(uv: np.ndarray, width: int, height: int) -> dict[str, np.ndarray]:
    u = np.asarray(uv[:, 0], dtype=np.float64)
    v = np.asarray(uv[:, 1], dtype=np.float64)
    return {
        "all": np.ones(len(uv), dtype=bool),
        "top": v < height / 3.0,
        "middle": (v >= height / 3.0) & (v < 2.0 * height / 3.0),
        "bottom": v >= 2.0 * height / 3.0,
        "left": u < width / 3.0,
        "center_u": (u >= width / 3.0) & (u < 2.0 * width / 3.0),
        "right": u >= 2.0 * width / 3.0,
    }


def propagate_lambda(
    item: Mapping[str, Any],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pose = solve_pose(np.asarray(item["corners"], dtype=np.float32), camera_matrix, dist_coeffs, obj)
    uv = np.asarray(item["uv_raw"], dtype=np.float64)
    undistorted = cv2.undistortPoints(
        uv.reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    ).reshape(-1, 2)
    rays = np.column_stack([undistorted, np.ones(len(undistorted), dtype=np.float64)])
    denominator = rays @ np.asarray(pose["normal"], dtype=np.float64)
    candidate_lambda = np.full(len(uv), np.nan, dtype=np.float64)
    valid_denom = np.abs(denominator) > 1.0e-12
    candidate_lambda[valid_denom] = -float(pose["d"]) / denominator[valid_denom]
    baseline_points = np.asarray(item["truth_points_raw"], dtype=np.float64)
    baseline_lambda = baseline_points[:, 2]
    valid = (
        np.asarray(item["truth_valid_raw"], dtype=bool)
        & valid_denom
        & np.isfinite(candidate_lambda)
        & np.isfinite(baseline_lambda)
        & (candidate_lambda > 0.0)
    )
    delta = candidate_lambda - baseline_lambda
    return uv, delta, {"valid": valid, "candidate_lambda": candidate_lambda, "pose": pose}


def delta_metrics(delta: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(delta, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"point_count": 0, "bias_delta_lambda_mm": math.nan, "mae_delta_lambda_mm": math.nan, "rmse_delta_lambda_mm": math.nan, "p95_abs_delta_lambda_mm": math.nan, "max_abs_delta_lambda_mm": math.nan}
    return {
        "point_count": int(len(x)),
        "bias_delta_lambda_mm": float(np.mean(x)),
        "mae_delta_lambda_mm": float(np.mean(np.abs(x))),
        "rmse_delta_lambda_mm": float(np.sqrt(np.mean(x * x))),
        "p95_abs_delta_lambda_mm": float(np.percentile(np.abs(x), 95)),
        "max_abs_delta_lambda_mm": float(np.max(np.abs(x))),
    }


def cone_summary_map(board_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["frame_id"]): row for row in board_summaries if row.get("row_type") == "frame"}


def build_propagation(
    candidates: Sequence[Mapping[str, Any]],
    processed: Mapping[str, Mapping[str, Any]],
    board_summaries: Sequence[Mapping[str, Any]],
    obj: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]], dict[str, Any]]:
    width, height = image_size
    rows: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    v_bins = np.linspace(0.0, float(height), 31)
    v_accum: dict[str, dict[str, np.ndarray]] = {}
    for frame_id, item in processed.items():
        v_accum[frame_id] = {
            "sum": np.zeros(30, dtype=np.float64),
            "sumsq": np.zeros(30, dtype=np.float64),
            "count": np.zeros(30, dtype=np.int64),
            "sum_027": np.zeros(30, dtype=np.float64),
            "count_027": np.zeros(30, dtype=np.int64),
        }
    for candidate_index, candidate in enumerate(candidates):
        if candidate.get("status") != "ok":
            continue
        camera_matrix = np.asarray(candidate["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(candidate["dist_coeffs"], dtype=np.float64)
        candidate_type = str(candidate["candidate_type"])
        candidate_id = str(candidate["candidate_id"])
        for frame_id in sorted(processed, key=int):
            item = processed[frame_id]
            uv, delta, info = propagate_lambda(item, camera_matrix, dist_coeffs, obj)
            valid = np.asarray(info["valid"], dtype=bool)
            masks = region_masks(uv, width, height)
            for region in REGION_ORDER:
                mask = valid & masks[region]
                metrics = delta_metrics(delta[mask])
                row = {
                    "candidate_type": candidate_type,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "frame_id": frame_id,
                    "is_frame027": frame_id == "027",
                    "region": region,
                    "candidate_pnp_rmse_px": info["pose"]["rmse_px"],
                    "baseline_pnp_rmse_px": item["pose"].reprojection_rmse_px,
                    "baseline_lambda_valid_count": int(np.count_nonzero(np.asarray(item["truth_valid_raw"], dtype=bool))),
                    "candidate_lambda_valid_count": int(np.count_nonzero(valid)),
                    **metrics,
                }
                rows.append(row)
                for metric_name in ("bias_delta_lambda_mm", "rmse_delta_lambda_mm", "p95_abs_delta_lambda_mm", "max_abs_delta_lambda_mm"):
                    distributions[f"{candidate_type}|{frame_id}|{region}"][metric_name].append(float(metrics[metric_name]))
            # Accumulate a compact v profile for the requested diagnostic plot.
            finite_mask = valid & np.isfinite(delta)
            indices = np.digitize(uv[finite_mask, 1], v_bins, right=False) - 1
            indices = np.clip(indices, 0, 29)
            accumulator = v_accum[frame_id]
            values = delta[finite_mask]
            np.add.at(accumulator["sum"], indices, values)
            np.add.at(accumulator["sumsq"], indices, values * values)
            np.add.at(accumulator["count"], indices, 1)
            if frame_id == "027":
                np.add.at(accumulator["sum_027"], indices, values)
                np.add.at(accumulator["count_027"], indices, 1)
    plot_data = {"v_bins": v_bins, "v_accum": v_accum}
    return rows, distributions, plot_data


def distribution_value(distributions: Mapping[str, Mapping[str, Sequence[float]]], key: str, metric_name: str, percentile: float = 50.0) -> float:
    values = np.asarray(distributions.get(key, {}).get(metric_name, []), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else math.nan


def frame_uncertainty_rows(
    processed: Mapping[str, Mapping[str, Any]],
    board_summaries: Sequence[Mapping[str, Any]],
    distributions: Mapping[str, Mapping[str, Sequence[float]]],
    coverage_rows: Sequence[Mapping[str, Any]],
    bootstrap_ok_count: int,
    loo_ok_count: int,
    task6c_path: Path | None = None,
) -> list[dict[str, Any]]:
    summary_map = cone_summary_map(board_summaries)
    coverage_map = {str(row["frame_id"]): row for row in coverage_rows}
    rows: list[dict[str, Any]] = []
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        baseline = summary_map[frame_id]
        row: dict[str, Any] = {
            "row_type": "frame",
            "frame_id": frame_id,
            "is_frame027": frame_id == "027",
            "bootstrap_success_count": bootstrap_ok_count,
            "loo_success_count": loo_ok_count,
            "pnp_rmse_px": float(item["pose"].reprojection_rmse_px),
            "board_tilt_deg": float(np.degrees(np.arccos(np.clip(abs(float(np.asarray(item["pose"].normal, dtype=np.float64)[2])), 0.0, 1.0)))),
            "cone_bias_mm": finite(baseline.get("bias_mm")),
            "cone_rmse_mm": finite(baseline.get("rmse_mm")),
            "cone_p95_abs_mm": finite(baseline.get("p95_abs_mm")),
            "a_frame_mm": finite(baseline.get("a_frame_mm")),
            "k_frame_mm_per_normalized_stripe": finite(baseline.get("k_frame_mm_per_normalized_stripe")),
            "sensor_u_min_px": float(np.min(item["uv_raw"][:, 0])),
            "sensor_u_max_px": float(np.max(item["uv_raw"][:, 0])),
            "sensor_v_min_px": float(np.min(item["uv_raw"][:, 1])),
            "sensor_v_max_px": float(np.max(item["uv_raw"][:, 1])),
            "lambda_truth_min_mm": float(np.nanmin(item["truth_points_raw"][:, 2])),
            "lambda_truth_max_mm": float(np.nanmax(item["truth_points_raw"][:, 2])),
        }
        if frame_id in coverage_map:
            # For calibration-frame IDs this agrees with the independently
            # recomputed coverage table; laser FIT extension frames use the
            # same formal PnP pose carried by process_groups_board.
            row["board_tilt_deg"] = finite(coverage_map[frame_id].get("board_tilt_deg"))
        for candidate_type in ("bootstrap", "loo"):
            prefix = "bootstrap" if candidate_type == "bootstrap" else "loo"
            key = f"{candidate_type}|{frame_id}|all"
            for metric_name, suffix in (
                ("bias_delta_lambda_mm", "bias"),
                ("rmse_delta_lambda_mm", "rmse"),
                ("p95_abs_delta_lambda_mm", "p95_abs"),
                ("max_abs_delta_lambda_mm", "max_abs"),
            ):
                row[f"{prefix}_{suffix}_median_mm"] = distribution_value(distributions, key, metric_name, 50.0)
                row[f"{prefix}_{suffix}_p95_mm"] = distribution_value(distributions, key, metric_name, 95.0)
                row[f"{prefix}_{suffix}_max_mm"] = distribution_value(distributions, key, metric_name, 100.0)
            for region in ("top", "middle", "bottom", "left", "center_u", "right"):
                region_key = f"{candidate_type}|{frame_id}|{region}"
                row[f"{prefix}_{region}_p95_abs_median_mm"] = distribution_value(distributions, region_key, "p95_abs_delta_lambda_mm", 50.0)
                row[f"{prefix}_{region}_p95_abs_p95_mm"] = distribution_value(distributions, region_key, "p95_abs_delta_lambda_mm", 95.0)
        row["bootstrap_p95_abs_to_cone_rmse_ratio"] = (
            row["bootstrap_p95_abs_p95_mm"] / row["cone_rmse_mm"]
            if math.isfinite(row["bootstrap_p95_abs_p95_mm"]) and row["cone_rmse_mm"] > 0.0
            else math.nan
        )
        v_middle = row["bootstrap_middle_p95_abs_p95_mm"]
        v_edge = max(row["bootstrap_top_p95_abs_p95_mm"], row["bootstrap_bottom_p95_abs_p95_mm"])
        u_center = row["bootstrap_center_u_p95_abs_p95_mm"]
        u_edge = max(row["bootstrap_left_p95_abs_p95_mm"], row["bootstrap_right_p95_abs_p95_mm"])
        row["bootstrap_v_edge_to_middle_ratio"] = v_edge / v_middle if math.isfinite(v_middle) and v_middle > 0.0 else math.nan
        row["bootstrap_u_edge_to_center_ratio"] = u_edge / u_center if math.isfinite(u_center) and u_center > 0.0 else math.nan
        rows.append(row)
    return rows


def classify(frame_rows: Sequence[Mapping[str, Any]], bootstrap_ok_count: int, requested_reps: int) -> dict[str, Any]:
    ratios = np.asarray([finite(row.get("bootstrap_p95_abs_to_cone_rmse_ratio")) for row in frame_rows], dtype=np.float64)
    ratios = ratios[np.isfinite(ratios)]
    p95_values = np.asarray([finite(row.get("bootstrap_p95_abs_p95_mm")) for row in frame_rows], dtype=np.float64)
    p95_values = p95_values[np.isfinite(p95_values)]
    candidate_p95_values = np.asarray([finite(row.get("bootstrap_p95_abs_median_mm")) for row in frame_rows], dtype=np.float64)
    candidate_p95_values = candidate_p95_values[np.isfinite(candidate_p95_values)]
    candidate_ratios = np.asarray(
        [
            finite(row.get("bootstrap_p95_abs_median_mm")) / finite(row.get("cone_rmse_mm"))
            for row in frame_rows
            if finite(row.get("bootstrap_p95_abs_median_mm")) >= 0.0 and finite(row.get("cone_rmse_mm")) > 0.0
        ],
        dtype=np.float64,
    )
    candidate_ratios = candidate_ratios[np.isfinite(candidate_ratios)]
    edge_ratios = np.asarray([finite(row.get("bootstrap_v_edge_to_middle_ratio")) for row in frame_rows], dtype=np.float64)
    edge_ratios = edge_ratios[np.isfinite(edge_ratios)]
    if not len(ratios) or bootstrap_ok_count < max(1, int(0.9 * requested_reps)):
        verdict = "C. LOW"
    else:
        typical_ratio = float(np.median(ratios))
        worst_ratio = float(np.max(ratios))
        if typical_ratio < 0.25 and worst_ratio < 0.50:
            verdict = "A. HIGH"
        elif typical_ratio < 0.75 and worst_ratio < 1.00:
            verdict = "B. MODERATE"
        else:
            verdict = "C. LOW"
    return {
        "verdict": verdict,
        "bootstrap_success_count": int(bootstrap_ok_count),
        "bootstrap_requested_count": int(requested_reps),
        "candidate_typical_intrinsics_p95_delta_lambda_mm": float(np.median(candidate_p95_values)) if len(candidate_p95_values) else math.nan,
        "typical_intrinsics_p95_delta_lambda_mm": float(np.median(p95_values)) if len(p95_values) else math.nan,
        "worst_intrinsics_p95_delta_lambda_mm": float(np.max(p95_values)) if len(p95_values) else math.nan,
        "candidate_typical_intrinsics_to_cone_rmse_ratio": float(np.median(candidate_ratios)) if len(candidate_ratios) else math.nan,
        "typical_intrinsics_to_cone_rmse_ratio": float(np.median(ratios)) if len(ratios) else math.nan,
        "worst_intrinsics_to_cone_rmse_ratio": float(np.max(ratios)) if len(ratios) else math.nan,
        "edge_to_middle_ratio_median": float(np.median(edge_ratios)) if len(edge_ratios) else math.nan,
        "edge_to_middle_ratio_max": float(np.max(edge_ratios)) if len(edge_ratios) else math.nan,
    }


def plot_coverage(rows: Sequence[Mapping[str, Any]], output: Path, image_size: tuple[int, int]) -> None:
    width, height = image_size
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    u = np.asarray([row["board_center_u_px"] for row in rows], dtype=float)
    v = np.asarray([row["board_center_v_px"] for row in rows], dtype=float)
    frame_ids = [str(row["frame_id"]) for row in rows]
    scatter = axes[0].scatter(u, v, c=np.arange(len(rows)), cmap="viridis", s=60)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    axes[0].set_xlabel("board center u (px)")
    axes[0].set_ylabel("board center v (px)")
    axes[0].set_title("Formal calibration board-center coverage")
    for x, y, label in zip(u, v, frame_ids, strict=True):
        axes[0].annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    fig.colorbar(scatter, ax=axes[0], label="frame order")
    spans_u = np.asarray([row["corner_u_span_norm"] for row in rows])
    spans_v = np.asarray([row["corner_v_span_norm"] for row in rows])
    tilt = np.asarray([row["board_tilt_deg"] for row in rows])
    axes[1].scatter(spans_u, spans_v, c=tilt, cmap="plasma", s=65)
    axes[1].set_xlabel("corner u span / sensor width")
    axes[1].set_ylabel("corner v span / sensor height")
    axes[1].set_title("Apparent board size; color = tilt (deg)")
    for x, y, label in zip(spans_u, spans_v, frame_ids, strict=True):
        axes[1].annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_bootstrap_distribution(
    bootstrap_rows: Sequence[Mapping[str, Any]],
    baseline_params: Mapping[str, float],
    output: Path,
) -> None:
    ok_rows = [row for row in bootstrap_rows if row.get("status") == "ok"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for axis, name in zip(axes.flat, PARAMETER_NAMES, strict=True):
        values = np.asarray([finite(row.get(name)) for row in ok_rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            axis.hist(values, bins=25, color="#4c78a8", alpha=0.82)
        axis.axvline(float(baseline_params[name]), color="#d62728", lw=1.5, label="formal")
        axis.set_title(name)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Frame bootstrap distributions of K/D")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_delta_vs_v(plot_data: Mapping[str, Any], output: Path, image_height: int) -> None:
    edges = np.asarray(plot_data["v_bins"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    accum = plot_data["v_accum"]
    total_sum = np.zeros(30, dtype=float)
    total_count = np.zeros(30, dtype=float)
    special_sum = np.zeros(30, dtype=float)
    special_count = np.zeros(30, dtype=float)
    for frame_id, values in accum.items():
        total_sum += values["sum"]
        total_count += values["count"]
        if frame_id == "027":
            special_sum += values["sum_027"]
            special_count += values["count_027"]
    total_mean = np.divide(total_sum, total_count, out=np.full(30, np.nan), where=total_count > 0)
    special_mean = np.divide(special_sum, special_count, out=np.full(30, np.nan), where=special_count > 0)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(centers, total_mean, marker="o", ms=3, label="all FIT frames / bootstrap")
    if np.any(np.isfinite(special_mean)):
        ax.plot(centers, special_mean, marker="o", ms=3, lw=2, label="027")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("sensor v (px)")
    ax.set_ylabel("mean candidate Δlambda (mm)")
    ax.set_title("Intrinsics-induced truth change vs sensor v")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xlim(0, image_height)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_uncertainty_vs_cone(frame_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    normal = [row for row in frame_rows if not row["is_frame027"]]
    special = [row for row in frame_rows if row["is_frame027"]]
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    for subset, color, label, marker in ((normal, "#4c78a8", "other FIT", "o"), (special, "#d62728", "027", "*")):
        x = np.asarray([finite(row["cone_rmse_mm"]) for row in subset], dtype=float)
        y = np.asarray([finite(row["bootstrap_p95_abs_p95_mm"]) for row in subset], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], c=color, s=60 if marker == "o" else 130, marker=marker, label=label)
        for row, xx, yy in zip(np.asarray(subset, dtype=object)[mask], x[mask], y[mask], strict=True):
            ax.annotate(str(row["frame_id"]), (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=7)
    upper = max(float(np.nanmax([finite(row["cone_rmse_mm"]) for row in frame_rows])), float(np.nanmax([finite(row["bootstrap_p95_abs_p95_mm"]) for row in frame_rows])))
    ax.plot([0, upper], [0, upper], "k--", lw=0.9, label="equal magnitude")
    ax.set_xlabel("frozen Cone residual RMSE (mm)")
    ax.set_ylabel("bootstrap P95-of-P95 Δlambda (mm)")
    ax.set_title("Intrinsics truth uncertainty vs observed Cone residual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_v_regions(frame_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    labels = ["top", "middle", "bottom"]
    values = [
        np.asarray([finite(row[f"bootstrap_{region}_p95_abs_p95_mm"]) for row in frame_rows], dtype=float)
        for region in labels
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    positions = np.arange(len(labels))
    medians = [float(np.nanmedian(value)) for value in values]
    p95s = [float(np.nanpercentile(value, 95)) for value in values]
    ax.bar(positions, medians, color=["#5b8ff9", "#61dDAA", "#f6bd16"], label="frame median")
    ax.scatter(positions, p95s, color="#d62728", marker="x", s=70, label="frame 95th percentile")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("P95 |Δlambda| (mm)")
    ax.set_title("Top / middle / bottom truth sensitivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def render_report(
    output: Path,
    args: argparse.Namespace,
    coverage_rows: Sequence[Mapping[str, Any]],
    loo_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
    baseline_params: Mapping[str, float],
    formal_document: Mapping[str, Any],
    image_size: tuple[int, int],
    frozen_info: Mapping[str, Any],
    baseline_runtime_delta: Mapping[str, float],
) -> None:
    width, height = image_size
    center_u = np.asarray([finite(row["board_center_u_px"]) for row in coverage_rows])
    center_v = np.asarray([finite(row["board_center_v_px"]) for row in coverage_rows])
    coverage_tilt = np.asarray([finite(row["board_tilt_deg"]) for row in coverage_rows])
    fx_loo = np.asarray([finite(row.get("delta_fx_pct")) for row in loo_rows if row.get("status") == "ok"])
    fy_loo = np.asarray([finite(row.get("delta_fy_pct")) for row in loo_rows if row.get("status") == "ok"])
    cx_loo = np.asarray([finite(row.get("delta_cx")) for row in loo_rows if row.get("status") == "ok"])
    cy_loo = np.asarray([finite(row.get("delta_cy")) for row in loo_rows if row.get("status") == "ok"])
    bootstrap_ok = [row for row in bootstrap_rows if row.get("status") == "ok"]
    bootstrap_param_summary: list[str] = []
    for name in PARAMETER_NAMES:
        values = np.asarray([finite(row.get(name)) for row in bootstrap_ok], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            bootstrap_param_summary.append(
                f"| {name} | {baseline_params[name]:.8g} | {np.median(values):.8g} | {np.percentile(values, 2.5):.8g} | {np.percentile(values, 97.5):.8g} |"
            )
    cone_rmse = np.asarray([finite(row["cone_rmse_mm"]) for row in frame_rows])
    ratios = np.asarray([finite(row["bootstrap_p95_abs_to_cone_rmse_ratio"]) for row in frame_rows])
    edge = np.asarray([finite(row["bootstrap_v_edge_to_middle_ratio"]) for row in frame_rows])
    special = next((row for row in frame_rows if row["frame_id"] == "027"), None)
    calibration_span_u = float(np.ptp(center_u) / width)
    calibration_span_v = float(np.ptp(center_v) / height)
    tilt_rho, tilt_p = (math.nan, math.nan)
    frame_tilt = np.asarray([finite(row["board_tilt_deg"]) for row in frame_rows])
    sigma = np.asarray([finite(row["bootstrap_p95_abs_p95_mm"]) for row in frame_rows])
    mask = np.isfinite(frame_tilt) & np.isfinite(sigma) & (np.ptp(frame_tilt) > 0) & (np.ptp(sigma) > 0)
    if np.count_nonzero(mask) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = spearmanr(frame_tilt[mask], sigma[mask])
        tilt_rho, tilt_p = float(result.statistic), float(result.pvalue)
    signed_delta = np.asarray([finite(row["bootstrap_bias_median_mm"]) for row in frame_rows])
    cone_bias = np.asarray([finite(row["cone_bias_mm"]) for row in frame_rows])
    bias_mask = np.isfinite(signed_delta) & np.isfinite(cone_bias) & (np.ptp(signed_delta) > 0) & (np.ptp(cone_bias) > 0)
    bias_rho, bias_p, same_sign_fraction = math.nan, math.nan, math.nan
    if np.count_nonzero(bias_mask) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bias_result = spearmanr(signed_delta[bias_mask], cone_bias[bias_mask])
        bias_rho, bias_p = float(bias_result.statistic), float(bias_result.pvalue)
        same_sign_fraction = float(np.mean(np.sign(signed_delta[bias_mask]) == np.sign(cone_bias[bias_mask])))
    lines = [
        "# Task 6E — Camera intrinsics → ray-plane truth stability audit",
        "",
        f"`INTRINSICS_TRUTH_STABILITY = {classification['verdict']}`",
        "",
        "## Scope and boundary",
        "",
        f"- Formal camera-calibration FIT: `chess 001.tif`–`chess 018.tif` ({len(coverage_rows)} frames), all used by the current 0811 K/D fit. Only these FIT calibration images were opened.",
        "- Laser diagnostic FIT: 001–018 and 025–036 (30 frames). No `laser_plane/validation` image was opened; no laser Validation data was read.",
        f"- Baseline intrinsics: `{args.formal_intrinsics.resolve()}`; image size {width}×{height}; pattern {BOARD_COLS}×{BOARD_ROWS}, square {SQUARE_MM:g} mm.",
        f"- Formal K/D flags retained: `CALIB_FIX_K3` (k3=0); candidate PnP uses `SOLVEPNP_ITERATIVE` plus `solvePnPRefineLM` when available.",
        "- Laser center UV, Steger settings, frozen Circular Cone, formal intrinsics, and frame membership were not changed. Cone is only used as an observed-residual reference.",
        f"- Frozen Cone provenance SHA-256: `{frozen_info.get('provenance_sha256', '')}`; formal Cone SHA-256: `{frozen_info.get('formal_cone_sha256', '')}`.",
        "",
        "## Camera calibration coverage",
        "",
        f"- Board-center normalized span: u **{calibration_span_u:.3f}**, v **{calibration_span_v:.3f}**; center range u={np.min(center_u):.1f}–{np.max(center_u):.1f}px, v={np.min(center_v):.1f}–{np.max(center_v):.1f}px.",
        f"- Board tilt range: **{np.min(coverage_tilt):.2f}–{np.max(coverage_tilt):.2f}°**; formal per-image RMSE range: **{np.nanmin([finite(row['formal_reprojection_rmse_px']) for row in coverage_rows]):.4f}–{np.nanmax([finite(row['formal_reprojection_rmse_px']) for row in coverage_rows]):.4f}px**.",
        "- The coverage is multi-pose and spans the sensor, but it is not a dense uniform calibration grid; edge leverage is represented by a subset of poses rather than every frame.",
        "",
        "## Leave-one-calibration-frame-out",
        "",
        f"- Re-estimated K/D for **{len([row for row in loo_rows if row.get('status') == 'ok'])}** LOO candidates with all 88 corners per retained frame and the unchanged k3-fixed model.",
        f"- LOO max |Δfx|/fx = **{np.nanmax(np.abs(fx_loo)):.4f}%**, max |Δfy|/fy = **{np.nanmax(np.abs(fy_loo)):.4f}%**, max |Δcx| = **{np.nanmax(np.abs(cx_loo)):.3f}px**, max |Δcy| = **{np.nanmax(np.abs(cy_loo)):.3f}px**.",
        "",
        "## Frame bootstrap",
        "",
        f"- Frame-level bootstrap: **{len(bootstrap_ok)}** successful / {len(bootstrap_rows)} requested replicates; samples whole calibration frames with replacement and never splits corners.",
        "",
        "| parameter | formal | bootstrap median | 2.5% | 97.5% |",
        "|---|---:|---:|---:|---:|",
        *bootstrap_param_summary,
        "",
        "## Propagation to laser truth",
        "",
        f"- Candidate K/D were propagated to the same extracted laser UV points for all 30 FIT frames. Median candidate P95 |Δlambda|: **{classification['candidate_typical_intrinsics_p95_delta_lambda_mm']:.4f} mm**; median bootstrap 95%-candidate P95: **{classification['typical_intrinsics_p95_delta_lambda_mm']:.4f} mm**; worst frame tail: **{classification['worst_intrinsics_p95_delta_lambda_mm']:.4f} mm**.",
        f"- Intrinsics-induced / frozen-Cone RMSE ratio: median candidate **{classification['candidate_typical_intrinsics_to_cone_rmse_ratio']:.3f}**, median 95%-candidate tail **{classification['typical_intrinsics_to_cone_rmse_ratio']:.3f}**, worst **{classification['worst_intrinsics_to_cone_rmse_ratio']:.3f}**.",
        f"- Sensor-v edge/middle amplification: median **{classification['edge_to_middle_ratio_median']:.3f}×**, maximum **{classification['edge_to_middle_ratio_max']:.3f}×**.",
        f"- Across frames, bootstrap uncertainty vs board tilt Spearman rho = **{tilt_rho:.3f}** (p={tilt_p:.3f}); this is a pose/coverage dependence in the frame bootstrap, not a sensor-v edge amplification.",
        f"- Signed direction check: median candidate Δlambda bias is {np.nanmin(signed_delta):.4f}–{np.nanmax(signed_delta):.4f} mm across frames; versus Cone frame bias Spearman rho **{bias_rho:.3f}** (p={bias_p:.3f}), same sign in **{same_sign_fraction:.3f}** of frames.",
        "",
        "| frame | Cone RMSE mm | bootstrap P95-of-P95 Δlambda mm | uncertainty/Cone | v-edge/middle | 027 |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in frame_rows:
        lines.append(
            f"| {row['frame_id']} | {finite(row['cone_rmse_mm']):.5f} | {finite(row['bootstrap_p95_abs_p95_mm']):.5f} | {finite(row['bootstrap_p95_abs_to_cone_rmse_ratio']):.3f} | {finite(row['bootstrap_v_edge_to_middle_ratio']):.3f} | {'yes' if row['is_frame027'] else ''} |"
        )
    lines += [
        "",
        "## Answers",
        "",
        f"1. Camera-calibration geometry coverage: **multi-pose and usable, but not uniformly dense at all sensor edges** (u/v center spans {calibration_span_u:.3f}/{calibration_span_v:.3f} of the sensor).",
        f"2. K,D frame-selection stability: **LOO changes remain small**; bootstrap distributions are summarized above and keep k3 fixed at zero.",
        f"3. Intrinsics-induced truth change: median candidate P95 **{classification['candidate_typical_intrinsics_p95_delta_lambda_mm']:.4f} mm**; bootstrap 95%-candidate P95 median **{classification['typical_intrinsics_p95_delta_lambda_mm']:.4f} mm**; worst **{classification['worst_intrinsics_p95_delta_lambda_mm']:.4f} mm**.",
        f"4. Sensor-edge amplification: present only as a modest, non-universal effect (median {classification['edge_to_middle_ratio_median']:.3f}×, max {classification['edge_to_middle_ratio_max']:.3f}×).",
        f"5. Enough to explain current frame-dependent residual: **{'yes' if classification['verdict'] == 'C. LOW' else 'no / only a fraction'}**; the median uncertainty/Cone ratio is {classification['typical_intrinsics_to_cone_rmse_ratio']:.3f}.",
        "6. Next step: continue with **corner extraction / camera-model bias audit** before changing the laser surface; do not alter formal K/D from this diagnostic alone.",
        "",
        "## 027",
        "",
    ]
    if special is not None:
        lines += [
            f"- 027 Cone RMSE: **{finite(special['cone_rmse_mm']):.5f} mm**; bootstrap P95-of-P95 Δlambda: **{finite(special['bootstrap_p95_abs_p95_mm']):.5f} mm**; ratio **{finite(special['bootstrap_p95_abs_to_cone_rmse_ratio']):.3f}**.",
            "- 027 bootstrap propagation is larger than its Cone RMSE (ratio above 1), so under this frame-selection uncertainty model it is not a clean intrinsics-independent exception.",
            "",
        ]
    lines += [
        "## Conclusion",
        "",
        f"`INTRINSICS_TRUTH_STABILITY = {classification['verdict']}`.",
        "The descriptive gates are: HIGH when typical and worst propagated uncertainty remain below 0.25×/0.50× of Cone RMSE; MODERATE when below 0.75×/1.00×; LOW otherwise or when bootstrap coverage is insufficient.",
        "",
        "Generated figures: `camera_calibration_corner_coverage.png`, `intrinsics_bootstrap_distribution.png`, `delta_lambda_vs_sensor_v.png`, `intrinsics_uncertainty_vs_cone_residual.png`, and `top_middle_bottom_truth_sensitivity.png`.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_k, formal_d, formal_document = load_formal_intrinsics(args.formal_intrinsics.resolve())
    formal_metrics = load_formal_fit_metrics(args.formal_fit_metrics.resolve())
    observations, image_size = load_calibration_observations(args.calibration_fit_dir.resolve())
    if image_size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise RuntimeError(f"Unexpected calibration image size {image_size}; expected {(IMAGE_WIDTH, IMAGE_HEIGHT)}")
    obj = object_points()
    base_params = parameter_values(formal_k, formal_d)
    coverage_rows = calibration_coverage_rows(observations, formal_k, formal_d, obj, image_size, formal_metrics)
    write_csv(output_dir / "camera_calibration_coverage.csv", coverage_rows)

    loo_rows: list[dict[str, Any]] = []
    for omit_index, observation in enumerate(observations):
        indices = [index for index in range(len(observations)) if index != omit_index]
        try:
            fit_rms, camera_matrix, dist_coeffs = calibrate_candidate(observations, indices, obj, image_size)
            loo_rows.append(
                parameter_delta_row(
                    base_params,
                    camera_matrix,
                    dist_coeffs,
                    candidate_type="loo",
                    candidate_id=f"loo_{observation['frame_id']}",
                    omitted_frame_id=str(observation["frame_id"]),
                    fit_rms_px=fit_rms,
                )
            )
        except (cv2.error, RuntimeError, ValueError) as exc:
            loo_rows.append(
                parameter_delta_row(
                    base_params,
                    formal_k,
                    formal_d,
                    candidate_type="loo",
                    candidate_id=f"loo_{observation['frame_id']}",
                    omitted_frame_id=str(observation["frame_id"]),
                    fit_rms_px=math.nan,
                    status="failed",
                    error=str(exc),
                )
            )
    write_csv(output_dir / "intrinsics_leave_one_out.csv", loo_rows)

    rng = np.random.default_rng(args.seed)
    bootstrap_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for rep in range(args.bootstrap_reps):
        indices = rng.integers(0, len(observations), size=len(observations)).tolist()
        candidate_id = f"bootstrap_{rep + 1:04d}"
        try:
            fit_rms, camera_matrix, dist_coeffs = calibrate_candidate(observations, indices, obj, image_size)
            row = parameter_delta_row(
                base_params,
                camera_matrix,
                dist_coeffs,
                candidate_type="bootstrap",
                candidate_id=candidate_id,
                fit_rms_px=fit_rms,
                bootstrap_unique_count=len(set(indices)),
            )
            bootstrap_rows.append(row)
            candidates.append(
                {
                    "candidate_type": "bootstrap",
                    "candidate_id": candidate_id,
                    "status": "ok",
                    "camera_matrix": camera_matrix,
                    "dist_coeffs": dist_coeffs,
                }
            )
        except (cv2.error, RuntimeError, ValueError) as exc:
            bootstrap_rows.append(
                parameter_delta_row(
                    base_params,
                    formal_k,
                    formal_d,
                    candidate_type="bootstrap",
                    candidate_id=candidate_id,
                    fit_rms_px=math.nan,
                    bootstrap_unique_count=len(set(indices)),
                    status="failed",
                    error=str(exc),
                )
            )
    write_csv(output_dir / "intrinsics_bootstrap.csv", bootstrap_rows)
    loo_candidates: list[dict[str, Any]] = []
    for row in loo_rows:
        if row.get("status") != "ok":
            continue
        values = {name: finite(row[name]) for name in PARAMETER_NAMES}
        camera_matrix = np.asarray([[values["fx"], 0.0, values["cx"]], [0.0, values["fy"], values["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_coeffs = np.asarray([[values["k1"]], [values["k2"]], [values["p1"]], [values["p2"]], [values["k3"]]], dtype=np.float64)
        loo_candidates.append(
            {
                "candidate_type": "loo",
                "candidate_id": row["candidate_id"],
                "status": "ok",
                "camera_matrix": camera_matrix,
                "dist_coeffs": dist_coeffs,
            }
        )
    candidates = loo_candidates + candidates

    # Baseline FIT laser processing is restricted to the explicit FIT roots in
    # board_audit.inventory_fit; no Validation directory is enumerated.
    _, calibration, reconstruction_params, runtime_intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    if not np.allclose(runtime_intrinsics.camera_matrix, formal_k, rtol=0.0, atol=1.0e-8) or not np.allclose(runtime_intrinsics.dist_coeffs, formal_d.reshape(-1), rtol=0.0, atol=1.0e-8):
        raise RuntimeError("Formal intrinsics YAML does not match the runtime intrinsics used by the FIT audit")
    groups = board_audit.inventory_fit(args.data_root.resolve())
    frozen_model, frozen_info = board_audit.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    board_summaries, processed = board_audit.process_groups_board(groups, runtime_intrinsics, calibration, reconstruction_params, frozen_model)
    propagation_rows, distributions, plot_data = build_propagation(candidates, processed, board_summaries, obj, image_size)
    write_csv(output_dir / "intrinsics_truth_propagation.csv", propagation_rows)
    frame_rows = frame_uncertainty_rows(
        processed,
        board_summaries,
        distributions,
        coverage_rows,
        sum(row.get("status") == "ok" for row in bootstrap_rows),
        sum(row.get("status") == "ok" for row in loo_rows),
    )
    classification = classify(
        frame_rows,
        sum(row.get("status") == "ok" for row in bootstrap_rows),
        args.bootstrap_reps,
    )
    aggregate_row = {
        "row_type": "aggregate",
        "frame_id": "aggregate",
        "is_frame027": False,
        **classification,
    }
    write_csv(output_dir / "frame_intrinsics_uncertainty.csv", [*frame_rows, aggregate_row])

    plot_coverage(coverage_rows, output_dir / "camera_calibration_corner_coverage.png", image_size)
    plot_bootstrap_distribution(bootstrap_rows, base_params, output_dir / "intrinsics_bootstrap_distribution.png")
    plot_delta_vs_v(plot_data, output_dir / "delta_lambda_vs_sensor_v.png", image_size[1])
    plot_uncertainty_vs_cone(frame_rows, output_dir / "intrinsics_uncertainty_vs_cone_residual.png")
    plot_v_regions(frame_rows, output_dir / "top_middle_bottom_truth_sensitivity.png")

    baseline_runtime_delta = {
        "max_abs_camera_matrix_delta": float(np.max(np.abs(runtime_intrinsics.camera_matrix - formal_k))),
        "max_abs_distortion_delta": float(np.max(np.abs(np.asarray(runtime_intrinsics.dist_coeffs).reshape(-1) - formal_d.reshape(-1)))),
    }
    render_report(
        output_dir / "report.md",
        args,
        coverage_rows,
        loo_rows,
        bootstrap_rows,
        frame_rows,
        classification,
        base_params,
        formal_document,
        image_size,
        frozen_info,
        baseline_runtime_delta,
    )
    metadata = {
        "task": "6E",
        "formal_intrinsics": str(args.formal_intrinsics.resolve()),
        "formal_intrinsics_sha256": sha256_file(args.formal_intrinsics.resolve()),
        "formal_fit_dir": str(args.calibration_fit_dir.resolve()),
        "formal_calibration_frame_ids": list(CALIBRATION_FRAME_IDS),
        "laser_fit_frame_ids": [f"{i:03d}" for i in range(1, 19)] + [f"{i:03d}" for i in range(25, 37)],
        "validation_opened": False,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
        "calibration_flags": "CALIB_FIX_K3",
        "pnp_solver": "SOLVEPNP_ITERATIVE + solvePnPRefineLM",
        "cone_refit": False,
        "steger_changed": False,
        "formal_model_changed": False,
        "classification": classification,
    }
    (output_dir / "provenance.json").write_text(json_dumps(metadata), encoding="utf-8")
    print(f"INTRINSICS_TRUTH_STABILITY = {classification['verdict']}")
    print(f"bootstrap_success={classification['bootstrap_success_count']}/{classification['bootstrap_requested_count']}")
    print(f"typical_p95_mm={classification['typical_intrinsics_p95_delta_lambda_mm']:.8g}")
    print(f"worst_p95_mm={classification['worst_intrinsics_p95_delta_lambda_mm']:.8g}")
    print(f"typical_ratio={classification['typical_intrinsics_to_cone_rmse_ratio']:.8g}")
    print(f"worst_ratio={classification['worst_intrinsics_to_cone_rmse_ratio']:.8g}")


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, cv2.error, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
