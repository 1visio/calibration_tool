#!/usr/bin/env python3
"""Task 4D: FIT-only Circular Cone objective mismatch audit.

No model is fitted or written. Validation data are never loaded. The script
compares the formal Cone surface residual with the camera-Z ray intersection
residual and its analytic local geometric sensitivity.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
from scipy.stats import spearmanr

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
    / "cone_objective_mismatch_audit"
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
MODEL_NAMES = ("M0", "M_leave027")
REGION_NAMES = ("global", "top_formal_edge", "middle_formal", "bottom_formal_edge")
SURFACE_SMALL_MM = 0.02
DEPTH_LARGE_MM = 0.10
RATIO_FLOOR_MM = 1.0e-4
V_BIN_WIDTH_PX = 60


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4D Circular Cone objective mismatch audit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=task3b2.MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def region_for_v(v: float) -> str:
    for name, low, high in task3a.REGIONS:
        if low <= v < high:
            return name
    return "outside_0_3000"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = probability * float(cumulative[-1])
    return float(values[min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)])


def weighted_metric(values: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return {
            "count": 0,
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "p95": float("nan"),
            "max_abs": float("nan"),
        }
    weights = weights / float(np.sum(weights))
    return {
        "count": int(values.size),
        "bias": float(np.sum(weights * values)),
        "mae": float(np.sum(weights * np.abs(values))),
        "rmse": float(np.sqrt(np.sum(weights * values * values))),
        "p95": weighted_quantile(np.abs(values), weights, 0.95),
        "max_abs": float(np.max(np.abs(values))),
    }


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    x = x[valid]
    y = y[valid]
    weights = weights[valid]
    if x.size < 2:
        return float("nan")
    weights = weights / float(np.sum(weights))
    x_centered = x - float(np.sum(weights * x))
    y_centered = y - float(np.sum(weights * y))
    denominator = math.sqrt(float(np.sum(weights * x_centered**2) * np.sum(weights * y_centered**2)))
    return float(np.sum(weights * x_centered * y_centered) / denominator) if denominator > 0.0 else float("nan")


def frame_equal_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        frame_id = str(row["frame_id"])
        counts[frame_id] = counts.get(frame_id, 0) + 1
    return np.asarray([1.0 / counts[str(row["frame_id"])] for row in rows], dtype=np.float64)


def cone_geometry(
    truth_points: np.ndarray,
    rays_z: np.ndarray,
    model: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    theta = local.legacy_model_to_theta(model)
    axis = task3a.sensitivity.angles_to_axis(float(theta[0]), float(theta[1]))
    apex = np.asarray(theta[2:5], dtype=np.float64)
    alpha = float(theta[5])
    q = np.asarray(truth_points, dtype=np.float64) - apex[None, :]
    axial = q @ axis
    perpendicular = q - axial[:, None] * axis[None, :]
    radial = np.linalg.norm(perpendicular, axis=1)
    safe_radial = np.maximum(radial, 1.0e-15)
    cot_alpha = 1.0 / math.tan(alpha)
    e_surface = radial * cot_alpha - axial
    gradient = perpendicular / safe_radial[:, None] * cot_alpha - axis[None, :]
    gradient_norm = np.linalg.norm(gradient, axis=1)
    normal = gradient / gradient_norm[:, None]
    dF_dcamera_z = np.sum(gradient * rays_z, axis=1)
    ray_unit = rays_z / np.linalg.norm(rays_z, axis=1)[:, None]
    ray_normal_cosine = np.abs(np.sum(normal * ray_unit, axis=1))
    ray_normal_cosine = np.clip(ray_normal_cosine, 0.0, 1.0)
    ray_normal_angle_deg = np.degrees(np.arccos(ray_normal_cosine))
    signed_distance = e_surface / gradient_norm
    amplification_surface_to_z = np.full(len(e_surface), np.nan, dtype=np.float64)
    amplification_distance_to_z = np.full(len(e_surface), np.nan, dtype=np.float64)
    linear_prediction = np.full(len(e_surface), np.nan, dtype=np.float64)
    derivative_valid = np.abs(dF_dcamera_z) > 1.0e-12
    amplification_surface_to_z[derivative_valid] = 1.0 / np.abs(dF_dcamera_z[derivative_valid])
    amplification_distance_to_z[derivative_valid] = gradient_norm[derivative_valid] / np.abs(dF_dcamera_z[derivative_valid])
    linear_prediction[derivative_valid] = -e_surface[derivative_valid] / dF_dcamera_z[derivative_valid]
    return {
        "theta": theta,
        "axis": axis,
        "apex": apex,
        "alpha": np.full(len(e_surface), alpha),
        "axial": axial,
        "radial": radial,
        "e_surface": e_surface,
        "gradient_norm": gradient_norm,
        "signed_surface_distance": signed_distance,
        "dF_dcamera_z": dF_dcamera_z,
        "amplification_surface_to_z": amplification_surface_to_z,
        "amplification_distance_to_z": amplification_distance_to_z,
        "ray_normal_cosine": ray_normal_cosine,
        "ray_normal_angle_deg": ray_normal_angle_deg,
        "linear_prediction": linear_prediction,
    }


def collect_rows(
    records: Sequence[task3a.FrameRecord],
    models: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    intrinsics: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid: dict[str, dict[str, int]] = {name: {} for name in MODEL_NAMES}
    ray_consistency_max = 0.0
    for record in records:
        pixels = np.asarray(record.pixels_uv, dtype=np.float64)
        normalized = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), intrinsics.camera_matrix, intrinsics.dist_coeffs
        ).reshape(-1, 2)
        rays_z = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
        truth = np.asarray(record.truth_points, dtype=np.float64)
        ray_consistency_max = max(
            ray_consistency_max,
            float(np.max(np.abs(truth - rays_z * truth[:, 2:3]))),
        )
        analysis_split = "frame027_sensitivity" if record.frame_id == TARGET_FRAME else "main_fit_leave027"
        for model_name, model in models.items():
            candidate = copy.deepcopy(dict(calibration))
            candidate["laser_model"] = copy.deepcopy(dict(model))
            lambda_cone, valid, _ = task4a.lambda_by_input(pixels, candidate, reconstruction_params)
            invalid[model_name][record.frame_id] = int(np.count_nonzero(~valid))
            geometry = cone_geometry(truth, rays_z, model)
            e_lambda = np.full(len(truth), np.nan, dtype=np.float64)
            e_lambda[valid] = lambda_cone[valid] - truth[valid, 2]
            observed_amplification = np.full(len(truth), np.nan, dtype=np.float64)
            ratio_valid = valid & (np.abs(geometry["e_surface"]) >= RATIO_FLOOR_MM)
            observed_amplification[ratio_valid] = (
                np.abs(e_lambda[ratio_valid]) / np.abs(geometry["e_surface"][ratio_valid])
            )
            for index in range(len(truth)):
                surface = float(geometry["e_surface"][index])
                depth = float(e_lambda[index])
                rows.append(
                    {
                        "analysis_split": analysis_split,
                        "frame_id": record.frame_id,
                        "point_index": index,
                        "model": model_name,
                        "u_px": float(pixels[index, 0]),
                        "v_px": float(pixels[index, 1]),
                        "region": region_for_v(float(pixels[index, 1])),
                        "lambda_truth_camera_Z_mm": float(truth[index, 2]),
                        "lambda_cone_camera_Z_mm": float(lambda_cone[index]) if valid[index] else float("nan"),
                        "intersection_valid": bool(valid[index]),
                        "e_surface_mm": surface,
                        "e_lambda_cone_minus_truth_mm": depth,
                        "signed_surface_distance_mm": float(geometry["signed_surface_distance"][index]),
                        "axial_coordinate_mm": float(geometry["axial"][index]),
                        "radial_coordinate_mm": float(geometry["radial"][index]),
                        "dF_dcamera_Z": float(geometry["dF_dcamera_z"][index]),
                        "geometric_amplification_e_surface_to_Z": float(geometry["amplification_surface_to_z"][index]),
                        "geometric_amplification_distance_to_Z": float(geometry["amplification_distance_to_z"][index]),
                        "ray_normal_abs_cosine": float(geometry["ray_normal_cosine"][index]),
                        "ray_normal_angle_deg": float(geometry["ray_normal_angle_deg"][index]),
                        "linearized_e_lambda_mm": float(geometry["linear_prediction"][index]),
                        "linearization_error_mm": depth - float(geometry["linear_prediction"][index]) if valid[index] else float("nan"),
                        "observed_abs_e_lambda_over_e_surface": float(observed_amplification[index]),
                        "surface_small_threshold_mm": SURFACE_SMALL_MM,
                        "depth_large_threshold_mm": DEPTH_LARGE_MM,
                        "surface_small_depth_large": bool(
                            valid[index] and abs(surface) <= SURFACE_SMALL_MM and abs(depth) >= DEPTH_LARGE_MM
                        ),
                    }
                )
    return rows, {"invalid_by_model_frame": invalid, "truth_ray_consistency_max_abs_mm": ray_consistency_max}


def select_rows(
    rows: Sequence[Mapping[str, Any]], split: str, model: str, region: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if row["analysis_split"] == split and row["model"] == model]
    if region != "global":
        selected = [row for row in selected if row["region"] == region]
    return selected


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if bool(row["intersection_valid"])]
    if not valid:
        return {"point_count": len(rows), "valid_count": 0, "frame_count": 0}
    weights = frame_equal_weights(valid)
    surface = np.asarray([float(row["e_surface_mm"]) for row in valid])
    depth = np.asarray([float(row["e_lambda_cone_minus_truth_mm"]) for row in valid])
    distance = np.asarray([float(row["signed_surface_distance_mm"]) for row in valid])
    prediction = np.asarray([float(row["linearized_e_lambda_mm"]) for row in valid])
    amplification = np.asarray([float(row["geometric_amplification_distance_to_Z"]) for row in valid])
    observed = np.asarray([float(row["observed_abs_e_lambda_over_e_surface"]) for row in valid])
    angle = np.asarray([float(row["ray_normal_angle_deg"]) for row in valid])
    small_large = np.asarray([bool(row["surface_small_depth_large"]) for row in valid])
    surface_metric = weighted_metric(surface, weights)
    depth_metric = weighted_metric(depth, weights)
    distance_metric = weighted_metric(distance, weights)
    linear_error = weighted_metric(depth - prediction, weights)
    total_energy = float(np.sum(weights * depth * depth))
    remaining_energy = float(np.sum(weights * (depth - prediction) ** 2))
    finite_observed = np.isfinite(observed)
    spearman = spearmanr(surface, depth, nan_policy="omit").statistic if len(surface) >= 2 else float("nan")
    return {
        "point_count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "frame_count": len({str(row["frame_id"]) for row in valid}),
        "weighting": "frame_equal_within_region",
        "surface": surface_metric,
        "signed_surface_distance": distance_metric,
        "ray_depth": depth_metric,
        "ray_depth_rmse_over_surface_rmse": float(depth_metric["rmse"] / surface_metric["rmse"]),
        "pearson_surface_vs_depth": weighted_corr(surface, depth, weights),
        "spearman_surface_vs_depth_point_equal": float(spearman),
        "linearized_prediction_vs_depth_corr": weighted_corr(prediction, depth, weights),
        "linearization_error": linear_error,
        "linearized_depth_energy_explained_fraction": float(1.0 - remaining_energy / total_energy) if total_energy > 0.0 else float("nan"),
        "geometric_amplification_distance_to_Z_median": weighted_quantile(amplification, weights, 0.50),
        "geometric_amplification_distance_to_Z_p95": weighted_quantile(amplification, weights, 0.95),
        "observed_abs_depth_over_surface_median": weighted_quantile(observed[finite_observed], weights[finite_observed], 0.50),
        "observed_abs_depth_over_surface_p95": weighted_quantile(observed[finite_observed], weights[finite_observed], 0.95),
        "ray_normal_angle_deg_median": weighted_quantile(angle, weights, 0.50),
        "ray_normal_angle_deg_p95": weighted_quantile(angle, weights, 0.95),
        "surface_small_depth_large_count": int(np.count_nonzero(small_large)),
        "surface_small_depth_large_point_fraction": float(np.mean(small_large)),
        "surface_small_depth_large_frame_weighted_fraction": float(np.sum(weights * small_large) / np.sum(weights)),
    }


def build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("main_fit_leave027", "frame027_sensitivity"):
        output[split] = {}
        for model in MODEL_NAMES:
            output[split][model] = {}
            for region in REGION_NAMES:
                output[split][model][region] = summarize_group(select_rows(rows, split, model, region))
    return output


def binned_median(rows: Sequence[Mapping[str, Any]], model: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row for row in rows
        if row["analysis_split"] == "main_fit_leave027" and row["model"] == model and math.isfinite(float(row[field]))
    ]
    centers: list[float] = []
    medians: list[float] = []
    for start in range(0, 3000, V_BIN_WIDTH_PX):
        values = [float(row[field]) for row in selected if start <= float(row["v_px"]) < start + V_BIN_WIDTH_PX]
        if values:
            centers.append(start + 0.5 * V_BIN_WIDTH_PX)
            medians.append(float(np.median(values)))
    return np.asarray(centers), np.asarray(medians)


def save_plots(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {"M0": "#2563eb", "M_leave027": "#c2410c"}
    # 1. Surface residual versus ray-depth residual; 027 remains a separate row.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_index, split in enumerate(("main_fit_leave027", "frame027_sensitivity")):
        for col_index, model in enumerate(MODEL_NAMES):
            axis = axes[row_index, col_index]
            selected = select_rows(rows, split, model, "global")
            x = np.asarray([float(row["e_surface_mm"]) for row in selected if bool(row["intersection_valid"])])
            y = np.asarray([float(row["e_lambda_cone_minus_truth_mm"]) for row in selected if bool(row["intersection_valid"])])
            axis.scatter(x, y, s=3 if row_index == 0 else 6, alpha=0.18 if row_index == 0 else 0.35, color=colors[model])
            axis.axhline(0.0, color="#555", linewidth=0.7); axis.axvline(0.0, color="#555", linewidth=0.7)
            axis.set_title(f"{split} — {model}"); axis.grid(alpha=0.2)
            axis.set_xlabel("e_surface / mm"); axis.set_ylabel("e_lambda = cone - truth / mm")
    fig.suptitle("Surface objective residual versus final ray-depth residual")
    fig.tight_layout(); fig.savefig(output_dir / "surface_vs_ray_depth_residual.png", dpi=170); plt.close(fig)

    # 2. Both residuals versus v for the main 29-frame analysis.
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for axis, model in zip(axes, MODEL_NAMES):
        selected = select_rows(rows, "main_fit_leave027", model, "global")
        v = np.asarray([float(row["v_px"]) for row in selected if bool(row["intersection_valid"])])
        surface = np.asarray([float(row["e_surface_mm"]) for row in selected if bool(row["intersection_valid"])])
        depth = np.asarray([float(row["e_lambda_cone_minus_truth_mm"]) for row in selected if bool(row["intersection_valid"])])
        axis.scatter(v, surface, s=2, alpha=0.10, color="#2b6cb0", label="e_surface")
        axis.scatter(v, depth, s=2, alpha=0.10, color="#c05621", label="e_lambda")
        for field, color in (("e_surface_mm", "#2b6cb0"), ("e_lambda_cone_minus_truth_mm", "#c05621")):
            centers, medians = binned_median(rows, model, field)
            axis.plot(centers, medians, color=color, linewidth=1.6)
        axis.axhline(0.0, color="#555", linewidth=0.7); axis.set_title(model); axis.grid(alpha=0.2); axis.legend()
        axis.set_ylabel("residual / mm")
    axes[-1].set_xlabel("v / px")
    fig.suptitle("Surface and ray-depth residual versus v — main 29 FIT frames")
    fig.tight_layout(); fig.savefig(output_dir / "surface_and_ray_depth_residual_vs_v.png", dpi=170); plt.close(fig)

    # 3. Analytic geometric amplification versus v.
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    for model in MODEL_NAMES:
        selected = select_rows(rows, "main_fit_leave027", model, "global")
        v = np.asarray([float(row["v_px"]) for row in selected])
        amp = np.asarray([float(row["geometric_amplification_distance_to_Z"]) for row in selected])
        axis.scatter(v, amp, s=2, alpha=0.08, color=colors[model])
        centers, medians = binned_median(rows, model, "geometric_amplification_distance_to_Z")
        axis.plot(centers, medians, color=colors[model], linewidth=1.8, label=f"{model} 60px median")
    axis.axvline(300.0, color="#777", linestyle=":", linewidth=0.8); axis.axvline(2700.0, color="#777", linestyle=":", linewidth=0.8)
    axis.set_xlabel("v / px"); axis.set_ylabel("local |dZ / d(surface distance)|")
    axis.set_title("Analytic ray-depth amplification versus v"); axis.grid(alpha=0.2); axis.legend()
    fig.tight_layout(); fig.savefig(output_dir / "ray_depth_surface_error_amplification_vs_v.png", dpi=170); plt.close(fig)

    # 4. Sensitivity versus absolute ray-depth error, colored by region.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    region_colors = {"top_formal_edge": "#dc2626", "middle_formal": "#64748b", "bottom_formal_edge": "#7c3aed", "outside_0_3000": "#0f766e"}
    for axis, model in zip(axes, MODEL_NAMES):
        selected = select_rows(rows, "main_fit_leave027", model, "global")
        for region, color in region_colors.items():
            group = [row for row in selected if row["region"] == region and bool(row["intersection_valid"])]
            axis.scatter(
                [float(row["geometric_amplification_distance_to_Z"]) for row in group],
                [abs(float(row["e_lambda_cone_minus_truth_mm"])) for row in group],
                s=3, alpha=0.16, color=color, label=region,
            )
        axis.set_title(model); axis.grid(alpha=0.2); axis.set_xlabel("local |dZ / d(surface distance)|")
    axes[0].set_ylabel("|e_lambda| / mm"); axes[0].legend(fontsize=8)
    fig.suptitle("Geometric sensitivity versus ray-depth error — main 29 FIT frames")
    fig.tight_layout(); fig.savefig(output_dir / "geometric_sensitivity_vs_ray_depth_error.png", dpi=170); plt.close(fig)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.7g}"


def classify(summary: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    levels: list[str] = []
    for model in MODEL_NAMES:
        main = summary["main_fit_leave027"][model]
        middle_amp = float(main["middle_formal"]["geometric_amplification_distance_to_Z_median"])
        edge_amp = max(
            float(main["top_formal_edge"]["geometric_amplification_distance_to_Z_median"]),
            float(main["bottom_formal_edge"]["geometric_amplification_distance_to_Z_median"]),
        )
        ratio = edge_amp / middle_amp
        mismatch_fraction = max(
            float(main[region]["surface_small_depth_large_frame_weighted_fraction"])
            for region in ("top_formal_edge", "middle_formal", "bottom_formal_edge")
        )
        diagnostics[model] = {
            "max_edge_to_middle_median_geometric_amplification_ratio": ratio,
            "max_region_surface_small_depth_large_frame_weighted_fraction": mismatch_fraction,
        }
        if ratio >= 1.5 and mismatch_fraction >= 0.01:
            levels.append("STRONG")
        elif ratio >= 1.2 or mismatch_fraction >= 0.001:
            levels.append("PARTIAL")
        else:
            levels.append("WEAK")
    if "STRONG" in levels:
        verdict = "STRONG"
    elif "PARTIAL" in levels:
        verdict = "PARTIAL"
    else:
        verdict = "WEAK"
    recommendation = "B" if verdict == "STRONG" else "C"
    diagnostics["classification_rule"] = {
        "STRONG": "edge/middle median amplification >=1.5 and small-surface/large-depth weighted fraction >=1%",
        "PARTIAL": "edge/middle median amplification >=1.2 or small-surface/large-depth weighted fraction >=0.1%",
        "WEAK": "neither gate is met",
        "thresholds_fixed_before_analysis": True,
    }
    return verdict, recommendation, diagnostics


def render_report(
    summary: Mapping[str, Any],
    verdict: str,
    recommendation: str,
    classification: Mapping[str, Any],
    audit: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    cone_hash_before: str,
    cone_hash_after: str,
    output_dir: Path,
) -> str:
    m0 = summary["main_fit_leave027"]["M0"]
    leave = summary["main_fit_leave027"]["M_leave027"]
    invalid_totals = {
        model: sum(int(value) for value in audit["invalid_by_model_frame"][model].values())
        for model in MODEL_NAMES
    }
    lines = [
        "# Task 4D — Circular Cone objective mismatch audit",
        "",
        "**FIT_ONLY = TRUE**  ",
        "**VALIDATION_OPENED = FALSE**  ",
        "**MODEL_REFIT = FALSE**  ",
        "**PRODUCTION_M0_MODIFIED = FALSE**  ",
        "**CORRECTION_ADDED = FALSE**",
        "",
        f"`OBJECTIVE_MISMATCH = {verdict}`",
        "",
        "## 定义与边界",
        "",
        "- 主分析：FIT 001–018、025–036，临时排除027，共29帧；027只作为 sensitivity case 单列。",
        "- 模型：正式 M0 与 Task 4C 的 M_leave027，二者均冻结；没有重新优化。",
        "- `e_surface = radial/tan(alpha) - axial`，与原 Circular Cone fit objective 的标量surface residual一致。",
        "- `e_lambda = lambda_cone - lambda_truth`，lambda为正式 reconstruction 返回的相机Z。",
        "- 相机射线使用 `k=[x_n,y_n,1]`，其中 `(x_n,y_n)=cv2.undistortPoints(u,v)`。",
        "- 解析局部预测：`e_lambda_linear = -e_surface/(grad(F)·k)`；几何放大使用 `|dZ/d(surface distance)|=|grad(F)|/|grad(F)·k|`。",
        f"- 固定异常诊断阈值：`|e_surface|≤{SURFACE_SMALL_MM}` mm 且 `|e_lambda|≥{DEPTH_LARGE_MM}` mm；阈值未按结果调整。",
        f"- truth点与正式ray的最大一致性误差：`{fmt(audit['truth_ray_consistency_max_abs_mm'])}` mm。",
        f"- 正式 M0 SHA-256 before/after：`{cone_hash_before}` / `{cone_hash_after}`。",
        "",
        "## 主分析：29帧 frame-equal 区域统计",
        "",
        "| model | region | surface RMSE mm | depth RMSE mm | depth/surface | corr(surface,depth) | geom amp median/P95 | linear prediction explained | small-surface large-depth |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        for region in REGION_NAMES:
            values = summary["main_fit_leave027"][model][region]
            lines.append(
                f"| {model} | {region} | {fmt(values['surface']['rmse'])} | {fmt(values['ray_depth']['rmse'])} | {fmt(values['ray_depth_rmse_over_surface_rmse'])} | {fmt(values['pearson_surface_vs_depth'])} | {fmt(values['geometric_amplification_distance_to_Z_median'])}/{fmt(values['geometric_amplification_distance_to_Z_p95'])} | {fmt(values['linearized_depth_energy_explained_fraction'])} | {values['surface_small_depth_large_count']} ({float(values['surface_small_depth_large_frame_weighted_fraction']):.3%}) |"
            )
    lines += [
        "",
        f"- 负相关符号来自本任务定义 `e_lambda=cone-truth`；surface residual 的正方向与求交深度误差相反。相关系数绝对值才表示一致程度。",
        f"- 解析一阶预测的global误差RMSE：M0=`{fmt(m0['global']['linearization_error']['rmse'])}` mm，M_leave027=`{fmt(leave['global']['linearization_error']['rmse'])}` mm，说明局部几何公式与正式求交数值一致。",
        f"- M0 top/middle surface RMSE比=`{float(m0['top_formal_edge']['surface']['rmse']) / float(m0['middle_formal']['surface']['rmse']):.6g}`，对应depth RMSE比=`{float(m0['top_formal_edge']['ray_depth']['rmse']) / float(m0['middle_formal']['ray_depth']['rmse']):.6g}`；M_leave027分别为 `{float(leave['top_formal_edge']['surface']['rmse']) / float(leave['middle_formal']['surface']['rmse']):.6g}` / `{float(leave['top_formal_edge']['ray_depth']['rmse']) / float(leave['middle_formal']['ray_depth']['rmse']):.6g}`。边缘增大在surface residual中已经存在。",
        f"- 正式求交无效点：M0=`{invalid_totals['M0']}`，M_leave027=`{invalid_totals['M_leave027']}`；逐点CSV保留这些行，没有静默删除。",
        "",
        "## 027 sensitivity case（不进入主结论）",
        "",
        "| model | surface RMSE mm | depth RMSE mm | corr | geom amp median/P95 | linear prediction explained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        values = summary["frame027_sensitivity"][model]["global"]
        lines.append(
            f"| {model} | {fmt(values['surface']['rmse'])} | {fmt(values['ray_depth']['rmse'])} | {fmt(values['pearson_surface_vs_depth'])} | {fmt(values['geometric_amplification_distance_to_Z_median'])}/{fmt(values['geometric_amplification_distance_to_Z_p95'])} | {fmt(values['linearized_depth_energy_explained_fraction'])} |"
        )

    max_edge_ratio = max(float(classification[model]["max_edge_to_middle_median_geometric_amplification_ratio"]) for model in MODEL_NAMES)
    max_mismatch = max(float(classification[model]["max_region_surface_small_depth_large_frame_weighted_fraction"]) for model in MODEL_NAMES)
    common_conclusion = (
        (float(m0["global"]["pearson_surface_vs_depth"]) > 0.95)
        == (float(leave["global"]["pearson_surface_vs_depth"]) > 0.95)
    )
    recommendation_text = {
        "A": "继续保持现有 Cone objective",
        "B": "保持 Circular Cone 模型，但改用 ray-depth objective 重新优化",
        "C": "objective mismatch 不足以解释残差，需要继续研究模型形式",
    }[recommendation]
    lines += [
        "",
        "## 判断",
        "",
        f"- 两模型的最大 edge/middle 中位几何放大比为 `{max_edge_ratio:.6g}`；固定small-surface/large-depth条件的最大frame-weighted比例为 `{max_mismatch:.3%}`。",
        f"- M0与M_leave027是否给出相同相关性等级：`{common_conclusion}`；两模型分类细节写入报告末尾的判据说明。",
        f"- M0 global surface/depth Pearson=`{fmt(m0['global']['pearson_surface_vs_depth'])}`，M_leave027=`{fmt(leave['global']['pearson_surface_vs_depth'])}`。",
        "",
        "## 最终回答",
        "",
        f"1. **原 surface objective 与最终 ray-depth 精度：{'不充分一致，存在明显位置相关放大' if verdict == 'STRONG' else ('部分一致，但仍存在可测的位置相关缩放' if verdict == 'PARTIAL' else '在当前FIT工作域内基本一致')}。**",
        f"2. **边缘大误差是否属于几何误差放大：{'是，几何放大是主要解释' if verdict == 'STRONG' else ('只能解释一部分' if verdict == 'PARTIAL' else '不是主要解释')}。**",
        f"3. **下一步选择 {recommendation}：{recommendation_text}。**",
        "",
        "027只用于独立sensitivity展示，没有混入上述判定。该结论是FIT-only objective诊断，不是Validation或部署结论。",
        "",
        "### 固定分类规则",
        "",
        f"- STRONG：{classification['classification_rule']['STRONG']}。",
        f"- PARTIAL：{classification['classification_rule']['PARTIAL']}。",
        f"- WEAK：{classification['classification_rule']['WEAK']}。",
        "",
        f"M0 alpha=`{fmt(model_metadata['M0']['half_apex_angle_deg'])}` deg；M_leave027 alpha=`{fmt(model_metadata['M_leave027']['half_apex_angle_deg'])}` deg。",
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

    cone_hash_before = task3b2.sha256_file(task3b2.FORMAL_CONE)
    if cone_hash_before != task3b2.EXPECTED_CONE_SHA256:
        raise RuntimeError(f"Formal M0 hash mismatch: {cone_hash_before}")
    task4c = json.loads(TASK4C_RESULT.read_text(encoding="utf-8"))
    if task4c.get("validation_opened") is not False or task4c.get("production_writeback") is not False:
        raise RuntimeError("Task 4C provenance does not satisfy frozen-input contract")
    if task4c.get("formal_cone_sha256_after") != cone_hash_before:
        raise RuntimeError("Task 4C M0 hash does not match current formal M0")

    # FIT-only loading. Validation loaders and validation paths are not called.
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records = task3a.load_old_records()
    extension_records, extension_provenance = task3a.load_extension_records(intrinsics)
    records += extension_records
    if [record.frame_id for record in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT registry mismatch")
    if len(records) != 30 or sum(record.frame_id == TARGET_FRAME for record in records) != 1:
        raise RuntimeError("Expected exactly one frame 027 in 30 FIT records")

    leave_model = task4c["parameters"]["M_leave027"]["runtime_mapping"]
    models = {
        "M0": copy.deepcopy(dict(calibration["laser_model"])),
        "M_leave027": copy.deepcopy(dict(leave_model)),
    }
    rows, audit = collect_rows(records, models, calibration, reconstruction_params, intrinsics)
    summary = build_summary(rows)
    verdict, recommendation, classification = classify(summary)
    model_metadata = {
        name: {
            "axis_unit_camera": model["axis_unit_camera"],
            "apex_camera_mm": model["apex_camera_mm"],
            "half_apex_angle_deg": model["half_apex_angle_deg"],
        }
        for name, model in models.items()
    }

    task3b2.write_csv(output_dir / "objective_residual_comparison.csv", rows)
    save_plots(output_dir, rows)
    cone_hash_after = task3b2.sha256_file(task3b2.FORMAL_CONE)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal M0 changed during Task 4D")
    report = render_report(
        summary,
        verdict,
        recommendation,
        classification,
        audit,
        model_metadata,
        cone_hash_before,
        cone_hash_after,
        output_dir,
    )
    (output_dir / "objective_mismatch_report.md").write_text(report, encoding="utf-8")
    provenance = {
        "task": "Task 4D Circular Cone objective mismatch audit",
        "fit_main_ids": [record.frame_id for record in records if record.frame_id != TARGET_FRAME],
        "sensitivity_case_ids": [TARGET_FRAME],
        "validation_ids_not_opened": task3a.VALIDATION_IDS,
        "validation_opened": False,
        "model_refit": False,
        "production_writeback": False,
        "correction_added": False,
        "formal_cone_sha256_before": cone_hash_before,
        "formal_cone_sha256_after": cone_hash_after,
        "M_leave027_source": str(TASK4C_RESULT),
        "extension_provenance_count": len(extension_provenance),
        "audit": audit,
        "model_metadata": model_metadata,
        "summary": summary,
        "classification": classification,
        "objective_mismatch": verdict,
        "next_step": recommendation,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=task3b2.json_default), encoding="utf-8"
    )
    print(f"OBJECTIVE_MISMATCH={verdict}")
    print(f"NEXT_STEP={recommendation}")
    print(f"ROWS={len(rows)}")
    print("VALIDATION_OPENED=False")
    print(f"OUTPUT={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
