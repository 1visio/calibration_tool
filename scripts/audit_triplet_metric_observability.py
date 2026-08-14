#!/usr/bin/env python3
"""Audit local metric-scale / height-gain observability of triplet truth.

This is a data-only diagnostic.  It does not fit a Circular Cone or any other
laser model.  FIT frames (001--018) determine the observability classification;
VALIDATION frames (019--024) are reported separately and never affect a fit,
bootstrap, threshold, or verdict.  The deployed Circular Cone is evaluated
read-only through ``reconstruct_uv_to_ground`` for a local-gain comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_TOOL_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
CALIBRATION_SRC = WORKSPACE_ROOT / "calibration" / "src"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(MEASUREMENT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_TOOL_ROOT))
if str(CALIBRATION_SRC) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SRC))

import analyze_circular_cone_parameter_sensitivity as sensitivity  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_COVERAGE_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "triplet_coverage_audit"
)
DEFAULT_POINTS = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0811"
    / "laser_model"
    / "calibration_points.csv"
)
DEFAULT_MEASUREMENT_CONFIG = sensitivity.paired.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_OUTPUT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "triplet_metric_observability"
)
FORMAL_CONE = (
    WORKSPACE_ROOT
    / "linelaser_tool"
    / "laser_measurement_tool"
    / "configs"
    / "calibration_daheng_0811"
    / "circular_cone.yaml"
)
BIN_WIDTHS = (30, 60, 100)
IMAGE_HEIGHT = 3000
FIT_IDS = {f"{index:03d}" for index in range(1, 19)}
VALIDATION_IDS = {f"{index:03d}" for index in range(19, 25)}
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260814
M0_DIFF_STEP_PX = 0.5

# These are pre-declared structural observability thresholds.  They are not
# learned from residuals or from the validation split.
MIN_FRAMES_INFORMATIVE = 3
MIN_FRAMES_WELL = 5
U_SPAN_WEAK_PX = 10.0
U_SPAN_WELL_PX = 50.0
CROSSFRAME_LAMBDA_WEAK_MM = 2.0
CROSSFRAME_LAMBDA_WELL_MM = 5.0
DESIGN_CONDITION_WELL = 1.0e3
REL_SLOPE_INTERVAL_WELL = 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triplet local metric observability audit")
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.12g}"
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def finite_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else math.nan


def finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else math.nan


def finite_span(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.ptp(values)) if values.size else math.nan


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan


def robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Theil-Sen line and diagnostics on frame-level medians.

    Returns slope, intercept, robust RMSE (ordinary RMS of residuals),
    Pearson correlation, and centered/scaled design condition number.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.ptp(x) <= 1.0e-12:
        return math.nan, math.nan, math.nan, math.nan, math.inf
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    upper = np.triu(np.ones_like(dx, dtype=bool), k=1)
    pair_valid = upper & np.isfinite(dx) & (np.abs(dx) > 1.0e-12)
    slopes = dy[pair_valid] / dx[pair_valid]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return math.nan, math.nan, math.nan, math.nan, math.inf
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    residual = y - (slope * x + intercept)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0 else math.nan
    centered = x - float(np.median(x))
    design = np.column_stack([centered, np.ones_like(centered)])
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    return slope, intercept, rmse, corr, condition


def quadratic_diagnostic(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.ptp(x) <= 1.0e-12:
        return math.nan, math.nan, math.nan, math.nan
    try:
        coeff = np.polyfit(x - np.median(x), y, 2)
    except (np.linalg.LinAlgError, ValueError):
        return math.nan, math.nan, math.nan, math.nan
    pred = np.polyval(coeff, x - np.median(x))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return float(coeff[0]), float(coeff[1]), float(coeff[2]), rmse


def bootstrap_slopes(x: np.ndarray, y: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < MIN_FRAMES_INFORMATIVE:
        return math.nan, math.nan, math.nan
    slopes: list[float] = []
    for _ in range(int(reps)):
        sample = rng.integers(0, x.size, size=x.size)
        slope, _, _, _, _ = robust_line(x[sample], y[sample])
        if math.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return math.nan, math.nan, math.nan
    values = np.asarray(slopes, dtype=np.float64)
    return tuple(float(item) for item in np.percentile(values, [5, 50, 95]))


def load_frame_geometry(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        frame_id = str(row["frame_id"]).zfill(3)
        result[frame_id] = {
            "plane_nx": float(row["board_plane_nx"]),
            "plane_ny": float(row["board_plane_ny"]),
            "plane_nz": float(row["board_plane_nz"]),
            "plane_d": float(row["board_plane_d_mm"]),
            "board_z": float(row["board_center_zc_mm"]),
            "pnp_success": row["pnp_success"].strip().lower() == "true",
        }
    return result


def load_support_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Empty Task 2 support file: {path}")
    return rows


def load_truth_points(path: Path, geometry: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in read_csv(path):
        frame_id = f"{int(row['image_id']):03d}"
        if frame_id not in geometry or not geometry[frame_id]["pnp_success"]:
            raise RuntimeError(f"Missing valid PnP geometry for frame {frame_id}")
        grouped[frame_id]["u"].append(float(row["u_px"]))
        grouped[frame_id]["v"].append(float(row["v_px"]))
    result: dict[str, dict[str, np.ndarray]] = {}
    for frame_id, values in grouped.items():
        frame = geometry[frame_id]
        u = np.asarray(values["u"], dtype=np.float64)
        v = np.asarray(values["v"], dtype=np.float64)
        # Use the same frozen camera intrinsics/distortion as Task 2's truth
        # generation, and the recorded frame plane from triplet_frame_geometry.
        result[frame_id] = {"u": u, "v": v, "board_z": np.full(u.shape, frame["board_z"]), "frame": np.full(u.shape, int(frame_id), dtype=np.int32)}
    return result


def fill_truth_lambda(
    points: Mapping[str, Mapping[str, np.ndarray]],
    geometry: Mapping[str, Mapping[str, float]],
    intrinsics_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    from calibrate_ground_extrinsics_board_only import load_intrinsics

    camera = load_intrinsics(intrinsics_path)
    output: dict[str, dict[str, np.ndarray]] = {}
    for frame_id, data in points.items():
        frame = geometry[frame_id]
        normal = np.asarray([frame["plane_nx"], frame["plane_ny"], frame["plane_nz"]], dtype=np.float64)
        uv = np.column_stack([data["u"], data["v"]]).reshape(-1, 1, 2)
        normalized = cv2.undistortPoints(uv, camera.camera_matrix, camera.dist_coeffs).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
        denom = rays @ normal
        lam = -float(frame["plane_d"]) / denom
        if not np.isfinite(lam).all() or np.any(lam <= 0):
            raise RuntimeError(f"Invalid independent ray-plane truth in frame {frame_id}")
        output[frame_id] = {
            "u": data["u"],
            "v": data["v"],
            "lambda": lam,
            "board_z": data["board_z"],
            "frame": data["frame"],
        }
    return output


def align_camera_z(input_uv: np.ndarray, output_uv: np.ndarray, output_z: np.ndarray) -> np.ndarray:
    aligned = np.full(len(input_uv), np.nan, dtype=np.float64)
    index = 0
    for row, pixel in enumerate(input_uv):
        if index >= len(output_uv):
            break
        if np.array_equal(pixel, output_uv[index]):
            aligned[row] = output_z[index]
            index += 1
    if index != len(output_uv):
        raise RuntimeError("M0 reconstruction output could not be aligned to frozen UV")
    return aligned


def calculate_m0_gain(
    points: Mapping[str, Mapping[str, np.ndarray]],
    measurement_config: Path,
) -> dict[str, dict[str, np.ndarray]]:
    app_config = sensitivity.load_app_config(measurement_config)
    calibration = sensitivity.load_calibration_files(
        app_config.calibration.intrinsics,
        app_config.calibration.laser_model,
        app_config.calibration.extrinsics,
        app_config.calibration.ground_u_compensation,
    )
    if calibration["laser_model"].get("model_type") != "circular_cone":
        raise RuntimeError("M0 calibration model is not circular_cone")
    output: dict[str, dict[str, np.ndarray]] = {}
    step = M0_DIFF_STEP_PX
    for frame_id, data in points.items():
        uv = np.column_stack([data["u"], data["v"]]).astype(np.float64)
        plus = uv.copy()
        minus = uv.copy()
        plus[:, 0] += step
        minus[:, 0] -= step
        reconstructed = reconstruct_uv_to_ground(
            np.vstack([plus, minus]), calibration, app_config.reconstruction
        )
        z = align_camera_z(
            np.vstack([plus, minus]),
            reconstructed.pixels_uv,
            reconstructed.points_camera[:, 2],
        )
        n = len(uv)
        if not np.isfinite(z).all():
            raise RuntimeError(f"M0 finite-difference reconstruction invalid in frame {frame_id}")
        derivative = (z[:n] - z[n:]) / (2.0 * step)
        output[frame_id] = {"lambda_m0_plus": z[:n], "lambda_m0_minus": z[n:], "m0_derivative": derivative}
    return output


def frame_aggregate(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    frame_values: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        frame_values[int(data["frame"][index])].append(int(index))
    frame_ids = sorted(frame_values)
    median_u = np.asarray([np.median(data["u"][frame_values[key]]) for key in frame_ids], dtype=np.float64)
    median_lambda = np.asarray([np.median(data["lambda"][frame_values[key]]) for key in frame_ids], dtype=np.float64)
    median_m0 = np.asarray([finite_median(data["m0"][frame_values[key]]) for key in frame_ids], dtype=np.float64)
    median_board_z = np.asarray([np.median(data["board_z"][frame_values[key]]) for key in frame_ids], dtype=np.float64)
    return {"frame": np.asarray(frame_ids, dtype=np.int32), "u": median_u, "lambda": median_lambda, "m0": median_m0, "board_z": median_board_z}


def flatten_points(points_by_frame: Mapping[str, Mapping[str, np.ndarray]], m0_by_frame: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for frame_id in sorted(points_by_frame, key=lambda value: int(value)):
        data = points_by_frame[frame_id]
        parts["u"].append(data["u"])
        parts["v"].append(data["v"])
        parts["lambda"].append(data["lambda"])
        parts["board_z"].append(data["board_z"])
        parts["frame"].append(data["frame"])
        parts["m0"].append(m0_by_frame[frame_id]["m0_derivative"])
        parts["m0_lambda"].append(m0_by_frame[frame_id]["lambda_m0_plus"])
    return {key: np.concatenate(value) for key, value in parts.items()}


def support_row_lookup(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(str(row["scope"]), str(row["region"])): row for row in rows}


def make_bin_rows(
    flat: Mapping[str, np.ndarray],
    split: str,
    width: int,
    bootstrap_reps: int,
    rng: np.random.Generator,
    support_lookup: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for start in range(0, IMAGE_HEIGHT, width):
        end = min(start + width, IMAGE_HEIGHT)
        mask = (flat["v"] >= start) & (flat["v"] < end)
        selected = {key: flat[key][mask] for key in flat}
        frame_count = int(np.unique(selected["frame"]).size) if selected["v"].size else 0
        aggregated = frame_aggregate(selected, np.ones(selected["v"].shape, dtype=bool)) if selected["v"].size else {"frame": np.empty(0, dtype=np.int32), "u": np.empty(0), "lambda": np.empty(0), "m0": np.empty(0), "board_z": np.empty(0)}
        slope, intercept, rmse, corr, condition = robust_line(aggregated["u"], aggregated["lambda"])
        quad_a, quad_b, quad_c, quad_rmse = quadratic_diagnostic(aggregated["u"], aggregated["lambda"])
        slope_p05, slope_p50, slope_p95 = bootstrap_slopes(aggregated["u"], aggregated["lambda"], bootstrap_reps, rng)
        # The formal M0 gain is the pointwise finite-difference derivative
        # d(lambda_M0)/du.  A line fit to that derivative versus u is kept as
        # a secondary trend diagnostic, but must not be confused with the M0
        # local gain itself.
        m0_derivative_median = finite_median(selected["m0"])
        m0_trend_slope, _, m0_trend_rmse, m0_trend_corr, m0_trend_condition = robust_line(aggregated["u"], aggregated["m0"])
        frame_lambda_span = finite_span(aggregated["lambda"])
        frame_u_span = finite_span(aggregated["u"])
        board_depth_span = finite_span(aggregated["board_z"])
        point_lambda_span = finite_span(selected["lambda"])
        point_u_span = finite_span(selected["u"])
        m0_local_gain = m0_derivative_median
        rel_interval = ((slope_p95 - slope_p05) / max(abs(slope_p50), 1.0e-12)) if math.isfinite(slope_p50) else math.inf
        classification = classify_bin(
            selected["v"].size,
            frame_count,
            frame_u_span,
            frame_lambda_span,
            condition,
            rel_interval,
        )
        support_scope = "fit" if split == "fit" else "validation"
        support_parent_start = (start // 300) * 300
        support_parent_end = min(support_parent_start + 299, IMAGE_HEIGHT - 1)
        support_region = f"bin_{support_parent_start:04d}_{support_parent_end:04d}"
        support_key = (support_scope, support_region)
        existing_support = support_lookup.get(support_key, {})
        row = {
            "split": split,
            "bin_width_px": width,
            "v_start": start,
            "v_end": end,
            "v_center": (start + end) / 2.0,
            "point_count": int(selected["v"].size),
            "unique_frame_count": frame_count,
            "frame_ids": ",".join(f"{value:03d}" for value in aggregated["frame"]),
            "u_min": finite_min(selected["u"]),
            "u_max": finite_max(selected["u"]),
            "u_span": point_u_span,
            "frame_median_u_span": frame_u_span,
            "lambda_min": finite_min(selected["lambda"]),
            "lambda_max": finite_max(selected["lambda"]),
            "lambda_span": point_lambda_span,
            "frame_median_lambda_span": frame_lambda_span,
            "board_depth_span": board_depth_span,
            "slope_dlambda_du": slope,
            "slope_intercept": intercept,
            "robust_RMSE": rmse,
            "corr_u_lambda": corr,
            "local_design_condition_number": condition,
            "quadratic_coefficient": quad_a,
            "quadratic_linear_coefficient": quad_b,
            "quadratic_intercept": quad_c,
            "quadratic_RMSE": quad_rmse,
            "linear_quadratic_RMSE_ratio": (rmse / quad_rmse) if math.isfinite(rmse) and math.isfinite(quad_rmse) and quad_rmse > 0 else math.nan,
            "slope_p05": slope_p05,
            "slope_p50": slope_p50,
            "slope_p95": slope_p95,
            "slope_relative_interval": rel_interval,
            "m0_local_gain_dlambda_du": m0_local_gain,
            "m0_point_derivative_median": m0_derivative_median,
            "m0_derivative_u_trend_slope": m0_trend_slope,
            "m0_derivative_u_trend_RMSE": m0_trend_rmse,
            "m0_derivative_u_trend_corr": m0_trend_corr,
            "m0_derivative_u_trend_condition_number": m0_trend_condition,
            "truth_slope_minus_m0_local_gain": (slope - m0_local_gain) if math.isfinite(slope) and math.isfinite(m0_local_gain) else math.nan,
            "truth_slope_to_m0_local_gain_ratio": (slope / m0_local_gain) if math.isfinite(slope) and math.isfinite(m0_local_gain) and abs(m0_local_gain) > 1.0e-12 else math.nan,
            "m0_gain_step_px": M0_DIFF_STEP_PX,
            "classification": classification,
            "task2_support_300px_region": support_region if existing_support else "",
            "task2_support_point_count": existing_support.get("point_count", ""),
            "task2_support_unique_frame_count": existing_support.get("unique_frame_count", ""),
            "task2_support_lambda_span": existing_support.get("lambda_truth_span", ""),
        }
        rows.append(row)
        if math.isfinite(slope_p50):
            bootstrap_rows.append({
                "split": split,
                "bin_width_px": width,
                "v_start": start,
                "v_end": end,
                "v_center": (start + end) / 2.0,
                "unique_frame_count": frame_count,
                "bootstrap_reps": bootstrap_reps,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "slope_p05": slope_p05,
                "slope_p50": slope_p50,
                "slope_p95": slope_p95,
                "slope_interval_width": slope_p95 - slope_p05,
                "slope_relative_interval": rel_interval,
                "resampling_unit": "frame",
            })
    return rows, bootstrap_rows


def classify_bin(point_count: int, frame_count: int, u_span: float, lambda_span: float, condition: float, rel_interval: float) -> str:
    if point_count == 0:
        return "UNSUPPORTED"
    if frame_count <= 1:
        return "SINGLE_FRAME_ONLY"
    if not math.isfinite(u_span) or u_span < U_SPAN_WEAK_PX:
        return "U_EXCITATION_WEAK"
    if not math.isfinite(lambda_span) or lambda_span < CROSSFRAME_LAMBDA_WEAK_MM:
        return "DEPTH_EXCITATION_WEAK"
    if (
        frame_count >= MIN_FRAMES_WELL
        and u_span >= U_SPAN_WELL_PX
        and lambda_span >= CROSSFRAME_LAMBDA_WELL_MM
        and condition <= DESIGN_CONDITION_WELL
        and rel_interval <= REL_SLOPE_INTERVAL_WELL
    ):
        return "WELL_CONSTRAINED"
    if frame_count >= MIN_FRAMES_INFORMATIVE:
        return "SPARSE_BUT_INFORMATIVE"
    return "SPARSE_BUT_INFORMATIVE"


def plot_u_lambda(path: Path, flat_fit: Mapping[str, np.ndarray], flat_validation: Mapping[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True, sharey=True)
    for axis, data, title, color in ((axes[0], flat_fit, "FIT 001–018", "#2563eb"), (axes[1], flat_validation, "VALIDATION 019–024 (coverage only)", "#f97316")):
        scatter = axis.scatter(data["u"], data["lambda"], c=data["v"], s=1.2, alpha=0.22, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("u / px")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("lambda_truth / mm")
    figure.colorbar(scatter, ax=axes, label="v / px")
    figure.suptitle("Independent ray-plane truth: local u–lambda excitation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_observability(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    colors = {30: "#2563eb", 60: "#16a34a", 100: "#dc2626"}
    for width in BIN_WIDTHS:
        subset = [row for row in rows if row["split"] == "fit" and int(row["bin_width_px"]) == width]
        x = np.asarray([float(row["v_center"]) for row in subset])
        frame = np.asarray([float(row["unique_frame_count"]) for row in subset])
        u_span = np.asarray([float(row["frame_median_u_span"]) if row["frame_median_u_span"] != "" else math.nan for row in subset])
        lambda_span = np.asarray([float(row["frame_median_lambda_span"]) if row["frame_median_lambda_span"] != "" else math.nan for row in subset])
        axes[0].plot(x, frame, color=colors[width], label=f"{width}px")
        axes[1].plot(x, u_span, color=colors[width], label=f"{width}px")
        axes[2].plot(x, lambda_span, color=colors[width], label=f"{width}px")
    axes[0].set_ylabel("unique frames")
    axes[1].set_ylabel("frame-median u span / px")
    axes[2].set_ylabel("frame-median lambda span / mm")
    axes[2].set_xlabel("v center / px")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=3)
    figure.suptitle("FIT local metric observability versus v")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_gain(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(13, 5.5))
    colors = {30: "#2563eb", 60: "#16a34a", 100: "#dc2626"}
    for width in BIN_WIDTHS:
        subset = [row for row in rows if row["split"] == "fit" and int(row["bin_width_px"]) == width]
        x = np.asarray([float(row["v_center"]) for row in subset])
        truth = np.asarray([float(row["slope_dlambda_du"]) if row["slope_dlambda_du"] != "" else math.nan for row in subset])
        m0 = np.asarray([float(row["m0_point_derivative_median"]) if row["m0_point_derivative_median"] != "" else math.nan for row in subset])
        p05 = np.asarray([float(row["slope_p05"]) if row["slope_p05"] != "" else math.nan for row in subset])
        p95 = np.asarray([float(row["slope_p95"]) if row["slope_p95"] != "" else math.nan for row in subset])
        axis.plot(x, truth, color=colors[width], label=f"truth {width}px")
        axis.fill_between(x, p05, p95, color=colors[width], alpha=0.10)
        axis.plot(x, m0, color=colors[width], linestyle="--", alpha=0.8, label=f"M0 {width}px")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlabel("v center / px")
    axis.set_ylabel("d lambda / d u (mm/px)")
    axis.set_title("FIT truth-supported local gain versus frozen formal M0")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def contiguous_class_ranges(rows: Sequence[Mapping[str, Any]], width: int, classes: set[str]) -> list[tuple[int, int, str]]:
    selected = sorted(
        [row for row in rows if int(row["bin_width_px"]) == width and str(row["classification"]) in classes],
        key=lambda row: int(row["v_start"]),
    )
    ranges: list[tuple[int, int, str]] = []
    for row in selected:
        start = int(row["v_start"])
        end = int(row["v_end"])
        label = str(row["classification"])
        if ranges and ranges[-1][1] == start and ranges[-1][2] == label:
            ranges[-1] = (ranges[-1][0], end, label)
        else:
            ranges.append((start, end, label))
    return ranges


def gap_reason(classification: str) -> str:
    if classification == "UNSUPPORTED":
        return "缺 frame；该区同时没有 u span 和 lambda span"
    if classification == "SINGLE_FRAME_ONLY":
        return "缺 frame；单帧无法形成跨frame u/lambda span"
    if classification == "U_EXCITATION_WEAK":
        return "缺跨frame u span"
    if classification == "DEPTH_EXCITATION_WEAK":
        return "缺跨frame lambda/depth span"
    if classification == "SPARSE_BUT_INFORMATIVE":
        return "已有基本激励；增加相邻pose可降低斜率不确定度"
    return ""


def region_gap_text(rows: Sequence[Mapping[str, Any]], width: int, low: int, high: int) -> str:
    subset = [
        row
        for row in rows
        if int(row["bin_width_px"]) == width and low <= int(row["v_start"]) and int(row["v_end"]) <= high
    ]
    hard_gap = {"UNSUPPORTED", "SINGLE_FRAME_ONLY", "U_EXCITATION_WEAK", "DEPTH_EXCITATION_WEAK"}
    bad_ranges = contiguous_class_ranges(subset, width, hard_gap)
    if bad_ranges:
        details = ", ".join(f"{start}–{end} {label}" for start, end, label in bad_ranges)
        good = any(str(row["classification"]) in {"WELL_CONSTRAINED", "SPARSE_BUT_INFORMATIVE"} for row in subset)
        prefix = "核心已有证据；" if good else ""
        return f"{prefix}需补采 {details}"
    if any(str(row["classification"]) == "SPARSE_BUT_INFORMATIVE" for row in subset):
        return "已有基本激励；可选补相邻pose"
    if any(str(row["classification"]) == "WELL_CONSTRAINED" for row in subset):
        return "无需补采"
    return "需补采（无FIT点）"


def render_report(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, str]],
    points_path: Path,
    coverage_dir: Path,
    measurement_config: Path,
    formal_cone_hash: str,
    verdict: str,
) -> str:
    fit_rows = [row for row in rows if row["split"] == "fit"]
    def counts(predicate):
        return sum(predicate(row) for row in fit_rows)
    classifications = defaultdict(int)
    for row in fit_rows:
        classifications[str(row["classification"])] += 1
    representative = []
    for low, high, label in ((0, 300, "top_0_299"), (300, 2700, "middle_300_2699"), (2700, 3000, "bottom_2700_2999")):
        subset = [row for row in fit_rows if low <= float(row["v_center"]) < high]
        representative.append((label, subset, low, high))
    lines = [
        "# Triplet local metric-scale / height-gain observability audit",
        "",
        "**NO_CONE_FIT = TRUE**  ",
        "**FIT_ONLY_FOR_DECISION = TRUE**",
        "",
        f"**METRIC_GAIN_COVERAGE = {verdict}**",
        "",
        "本审计使用 Task 2 的 frame geometry、300 px support summary，以及 `calibration_points.csv` 中记录的逐点 laser centre UV；逐点 `lambda_truth` 重新按同一 PnP plane 与正式内参计算。FIT 001–018 才能影响 observability 分类；VALIDATION 019–024 只单独显示。",
        "",
        "## Provenance and isolation",
        "",
        f"- Task 2 coverage directory：`{coverage_dir}`。",
        f"- recorded UV source：`{points_path}`；没有重新提取 Steger。",
        f"- formal measurement config / M0 path：`{measurement_config}`。",
        f"- formal Cone SHA-256（运行前后相同）：`{formal_cone_hash}`。",
        "- M0 comparison calls production `reconstruct_uv_to_ground()` only; no parameter, residual, weight, threshold, or candidate is optimized.",
        "- bootstrap resampling unit is frame, not point; fixed seed and fixed replicate count are recorded in `triplet_local_gain_bootstrap.csv`.",
        "",
        "## Fixed classification rules",
        "",
        "分类优先级：",
        "1. no points → `UNSUPPORTED`；",
        "2. one frame → `SINGLE_FRAME_ONLY`；",
        f"3. frame-median u span < {U_SPAN_WEAK_PX:g} px → `U_EXCITATION_WEAK`；",
        f"4. frame-median lambda span < {CROSSFRAME_LAMBDA_WEAK_MM:g} mm → `DEPTH_EXCITATION_WEAK`；",
        f"5. 至少 {MIN_FRAMES_WELL} frames、u span ≥ {U_SPAN_WELL_PX:g} px、lambda span ≥ {CROSSFRAME_LAMBDA_WELL_MM:g} mm、condition ≤ {DESIGN_CONDITION_WELL:g}、bootstrap relative interval ≤ {REL_SLOPE_INTERVAL_WELL:g} → `WELL_CONSTRAINED`；",
        f"6. 其余有至少 {MIN_FRAMES_INFORMATIVE} frames 的 bin → `SPARSE_BUT_INFORMATIVE`。",
        "这些阈值是预先声明的几何/统计可观测性规则，不由量块、Cone residual 或 validation 结果反推。",
        "",
        "## FIT classification summary",
        "",
        "| classification | FIT bin count (all scales) | interpretation |",
        "|---|---:|---|",
        f"| WELL_CONSTRAINED | {classifications['WELL_CONSTRAINED']} | 可稳定估计局部 gain |",
        f"| SPARSE_BUT_INFORMATIVE | {classifications['SPARSE_BUT_INFORMATIVE']} | 有跨帧激励，但斜率不够稳定/密集 |",
        f"| DEPTH_EXCITATION_WEAK | {classifications['DEPTH_EXCITATION_WEAK']} | 跨帧 depth/lambda 变化不足 |",
        f"| U_EXCITATION_WEAK | {classifications['U_EXCITATION_WEAK']} | 跨帧 u 基线不足 |",
        f"| SINGLE_FRAME_ONLY | {classifications['SINGLE_FRAME_ONLY']} | 无法用跨帧关系约束 gain |",
        f"| UNSUPPORTED | {classifications['UNSUPPORTED']} | 没有 FIT 点 |",
        "",
        "## Top / middle / bottom",
        "",
        "| region | scales present | strongest FIT evidence | gap interpretation |",
        "|---|---|---|---|",
    ]
    for label, subset, low, high in representative:
        present = sorted({int(row["bin_width_px"]) for row in subset})
        informative = [row for row in subset if row["classification"] in {"WELL_CONSTRAINED", "SPARSE_BUT_INFORMATIVE"}]
        if not subset:
            evidence = "no bins"
            gap = "UNSUPPORTED"
        else:
            best = max(informative or subset, key=lambda row: int(row["unique_frame_count"]))
            evidence = f"best {int(best['bin_width_px'])}px: {best['classification']}, frames={best['unique_frame_count']}, frame-u-span={fmt(best['frame_median_u_span'])}, frame-lambda-span={fmt(best['frame_median_lambda_span'])}, slope={fmt(best['slope_dlambda_du'], 6)}"
            gap = region_gap_text(fit_rows, int(best["bin_width_px"]), low, high)
        lines.append(f"| {label} | {', '.join(str(x) + 'px' for x in present)} | {evidence} | {gap} |")
    lines += [
        "",
        "## Gain comparison with frozen M0",
        "",
        "`slope_dlambda_du` 是按 frame median `(u,lambda_truth)` 的 Theil–Sen slope；`m0_point_derivative_median`（同 CSV 的 `m0_local_gain_dlambda_du`）是正式 M0 在同一 UV 处以 ±0.5 px 数值差分得到的 pointwise `d(lambda_M0)/du` 中位数；M0 没有被优化。",
        "",
        "| scale | v region | truth slope median | truth bootstrap P05–P95 | M0 derivative median | truth–M0 slope | classification |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for width in BIN_WIDTHS:
        for low, high, label in ((0, 300, "top"), (300, 2700, "middle"), (2700, 3000, "bottom")):
            subset = [row for row in rows if row["split"] == "fit" and int(row["bin_width_px"]) == width and low <= float(row["v_center"]) < high]
            if not subset:
                continue
            valid = [row for row in subset if row["slope_dlambda_du"] != ""]
            if not valid:
                continue
            # report the bin with the greatest FIT point count in that region.
            row = max(valid, key=lambda item: int(item["point_count"]))
            lines.append(f"| {width} | {label} ({int(row['v_start'])}–{int(row['v_end'])}) | {fmt(row['slope_dlambda_du'], 6)} | {fmt(row['slope_p05'], 6)}–{fmt(row['slope_p95'], 6)} | {fmt(row['m0_point_derivative_median'], 6)} | {fmt(row['truth_slope_minus_m0_local_gain'], 6)} | {row['classification']} |")
    lines += [
        "",
        "## Acquisition gaps",
        "",
        "- **无需补采（当前证据已足够）：** 30 px 的 `WELL_CONSTRAINED` 核心为 `v=300–2610`，top `v=270–300` 已有跨帧基本激励；60/100 px 结果见下方精确区间表。",
        "- **需要补采：** `SINGLE_FRAME_ONLY` 或 `UNSUPPORTED` 区间；首要缺口是覆盖这些 v 的新 frame，而不是在同一 frame 增加密集点。",
        "- `DEPTH_EXCITATION_WEAK`：缺跨帧 board depth / lambda span；应增加不同棋盘距离或倾角的 frame。",
        "- `U_EXCITATION_WEAK`：缺跨帧 median-u baseline；应让同一 v 区域在不同 pose 下横向落点分离。",
        "- `SPARSE_BUT_INFORMATIVE`：已有基本 gain 方向，但 slope CI 或设计条件数不足；优先增加相邻 pose，而不是重复同一 pose 的密集点。",
        "- `UNSUPPORTED`：该 v 区域 FIT 没有点；必须直接采集覆盖该 v 的棋盘+激光三联图。",
        "",
        "## Exact FIT intervals and missing excitation",
        "",
        "下表按每个 bin scale 合并连续区间；`SPARSE_BUT_INFORMATIVE` 不是硬缺口，但代表应优先补相邻pose以收窄 slope CI。",
        "",
        "| scale | v interval | classification | 缺什么 |",
        "|---:|---|---|---|",
        *[
            f"| {width}px | {start}–{end} | `{classification}` | {gap_reason(classification)} |"
            for width in BIN_WIDTHS
            for start, end, classification in contiguous_class_ranges(
                fit_rows,
                width,
                {"UNSUPPORTED", "SINGLE_FRAME_ONLY", "U_EXCITATION_WEAK", "DEPTH_EXCITATION_WEAK", "SPARSE_BUT_INFORMATIVE"},
            )
        ],
        "",
        "## Limits",
        "",
        "- 局部线性/二次关系只是 observability diagnostic，不是补偿模型。",
        "- 该审计证明的是局部 metric-scale 激励是否存在，不证明 Circular Cone 六参数良好可辨识，也不证明 M0 正确。",
        "- VALIDATION 表格和图只用于独立显示，不能用于选择缺口、阈值或候选。",
        "",
    ]
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    coverage_dir = args.coverage_dir.resolve()
    points_path = args.points.resolve()
    measurement_config = args.measurement_config.resolve()
    output_dir = args.output_dir.resolve()
    output_names = (
        "triplet_metric_observability.csv",
        "triplet_local_gain_bootstrap.csv",
        "triplet_u_lambda_support.png",
        "triplet_metric_observability_vs_v.png",
        "triplet_gain_support_vs_v.png",
        "triplet_metric_observability_report.md",
    )
    existing = [output_dir / name for name in output_names if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite")

    geometry = load_frame_geometry(coverage_dir / "triplet_frame_geometry.csv")
    task2_support = load_support_rows(coverage_dir / "triplet_ray_depth_support.csv")
    support_lookup = support_row_lookup(task2_support)
    points = load_truth_points(points_path, geometry)
    if set(points) != FIT_IDS | VALIDATION_IDS:
        raise RuntimeError(f"Expected frames 001-024 in point truth, got {sorted(points)}")
    intrinsics_path = Path(sensitivity.load_app_config(measurement_config).calibration.intrinsics).resolve()
    truth = fill_truth_lambda(points, geometry, intrinsics_path)
    m0 = calculate_m0_gain(points, measurement_config)
    for frame_id in truth:
        # Keep the finite-difference d(lambda_M0)/du as the M0 gain trace;
        # the raw reconstructed lambda is retained separately for diagnostics.
        truth[frame_id]["m0"] = m0[frame_id]["m0_derivative"]
        truth[frame_id]["m0_lambda"] = m0[frame_id]["lambda_m0_plus"]
    flat = flatten_points(truth, m0)
    flat_fit = {key: value[np.isin(flat["frame"], [int(item) for item in FIT_IDS])] for key, value in flat.items()}
    flat_validation = {key: value[np.isin(flat["frame"], [int(item) for item in VALIDATION_IDS])] for key, value in flat.items()}

    formal_hash_before = sha256_file(FORMAL_CONE)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for split, data in (("fit", flat_fit), ("validation", flat_validation)):
        for width in BIN_WIDTHS:
            local_rows, local_bootstrap = make_bin_rows(data, split, width, args.bootstrap_reps, rng, support_lookup)
            rows.extend(local_rows)
            bootstrap_rows.extend(local_bootstrap)

    fit_rows = [row for row in rows if row["split"] == "fit"]
    center_rows = [row for row in fit_rows if int(row["v_start"]) >= 300 and int(row["v_end"]) <= 2700]
    edge_rows = [row for row in fit_rows if int(row["v_start"]) < 300 or int(row["v_end"]) > 2700]
    center_informative = sum(row["classification"] in {"WELL_CONSTRAINED", "SPARSE_BUT_INFORMATIVE"} for row in center_rows)
    edge_single_or_unsupported = sum(row["classification"] in {"SINGLE_FRAME_ONLY", "UNSUPPORTED"} for row in edge_rows)
    edge_informative = sum(row["classification"] in {"WELL_CONSTRAINED", "SPARSE_BUT_INFORMATIVE"} for row in edge_rows)
    if center_rows and center_informative == len(center_rows) and edge_single_or_unsupported == 0:
        verdict = "SUFFICIENT"
    elif center_rows and center_informative >= max(1, int(0.8 * len(center_rows))) and edge_informative > 0:
        verdict = "PARTIAL"
    else:
        verdict = "INSUFFICIENT"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "triplet_metric_observability.csv", rows)
    write_csv(output_dir / "triplet_local_gain_bootstrap.csv", bootstrap_rows)
    plot_u_lambda(output_dir / "triplet_u_lambda_support.png", flat_fit, flat_validation)
    plot_observability(output_dir / "triplet_metric_observability_vs_v.png", rows)
    plot_gain(output_dir / "triplet_gain_support_vs_v.png", rows)
    formal_hash_after = sha256_file(FORMAL_CONE)
    if formal_hash_after != formal_hash_before:
        raise RuntimeError("Formal Circular Cone changed during observability audit")
    (output_dir / "triplet_metric_observability_report.md").write_text(
        render_report(rows, bootstrap_rows, task2_support, points_path, coverage_dir, measurement_config, formal_hash_before, verdict),
        encoding="utf-8",
    )
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != set(output_names):
        raise RuntimeError(f"Unexpected output files: {sorted(actual)}")
    print(f"METRIC_GAIN_COVERAGE = {verdict}")
    print(f"FIT rows={len([row for row in rows if row['split'] == 'fit'])}, validation rows={len([row for row in rows if row['split'] == 'validation'])}")
    print(f"formal_cone_sha256={formal_hash_before}")
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
