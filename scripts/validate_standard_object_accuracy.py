#!/usr/bin/env python3
"""Frozen C0/C1 full-field standard-object accuracy acceptance.

This audit deliberately uses the four existing standard-object measurement
directories only.  It does not read the laser-plane Validation set, refit K/D
or the Cone, and does not re-extract laser centers.  C0 and C1 consume the
same ``u,v`` rows from the existing laser-center/height/baseline CSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
MEASUREMENT_ROOT = CALIBRATION_TOOL_ROOT.parent / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
from measurement.height_measure import measure_height_line  # noqa: E402
from reconstruction.reconstructor import apply_ground_u_compensation  # noqa: E402


NOMINAL_HEIGHT_MM = 50.0
ACCURACY_TARGET_MM = 0.2
IMAGE_HEIGHT = 3000.0

POSITION_ORDER = (
    ("frame_065292_measure", "Top", "top"),
    ("frame_063995_measure", "Middle_Upper", "middle"),
    ("frame_062878_measure", "Middle_Lower", "middle"),
    ("frame_061303_measure", "Bottom", "bottom"),
)
REGION_ORDER = ("Top", "Middle", "Bottom")

DEFAULT_STANDARD_ROOT = Path(
    r"D:\Docs\linelaserscan\0704line-laser-3d-scanner\laser_measurement_tool\output_daheng_0811"
)
DEFAULT_C1_MODEL = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "c1_independent_validation"
    / "frozen_c1_model.json"
)
DEFAULT_PROVENANCE = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "circular_vs_elliptical_cone"
    / "provenance.json"
)
DEFAULT_FORMAL_CONE = (
    MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
)
DEFAULT_MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "standard_object_accuracy"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", type=Path, default=DEFAULT_STANDARD_ROOT)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nominal-height-mm", type=float, default=NOMINAL_HEIGHT_MM)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_uv(path: Path) -> np.ndarray:
    rows = csv_rows(path)
    if not rows or not {"u", "v"}.issubset(rows[0]):
        raise RuntimeError(f"{path} must contain u,v columns")
    values = np.asarray([[float(row["u"]), float(row["v"])] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path} contains non-finite u,v")
    return values


def uv_keys(values: np.ndarray) -> set[tuple[float, float]]:
    return {tuple(np.round(row, 10)) for row in np.asarray(values, dtype=np.float64)}


def metric(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "point_count": 0,
            "bias_mm": math.nan,
            "mae_mm": math.nan,
            "rmse_mm": math.nan,
            "p95_abs_mm": math.nan,
            "max_abs_mm": math.nan,
        }
    return {
        "point_count": int(len(values)),
        "bias_mm": float(np.mean(values)),
        "mae_mm": float(np.mean(np.abs(values))),
        "rmse_mm": float(np.sqrt(np.mean(values * values))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95)),
        "max_abs_mm": float(np.max(np.abs(values))),
    }


def pct_change(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or abs(before) <= np.finfo(float).eps:
        return math.nan
    return float((after - before) / abs(before) * 100.0)


def fmt(value: Any, digits: int = 6) -> str:
    number = finite(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def resolve_existing(path: Path, *alternatives: Path) -> Path:
    candidates = (path, *alternatives)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("None of the paths exists: " + ", ".join(str(item) for item in candidates))


def ground_points_from_lambda(
    uv: np.ndarray,
    lambdas: np.ndarray,
    valid: np.ndarray,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.asarray(uv, dtype=np.float64)
    values = np.asarray(lambdas, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
    camera = rays * values[:, None]
    finite_camera = np.isfinite(camera).all(axis=1)
    mask &= finite_camera
    R = np.asarray(calibration["R"], dtype=np.float64)
    t = np.asarray(calibration["t"], dtype=np.float64).reshape(3)
    ground = camera @ R.T + t
    compensation = calibration.get("ground_u_compensation")
    if compensation is not None:
        ground = apply_ground_u_compensation(ground, pixels, compensation)
    mask &= np.isfinite(ground).all(axis=1)
    return np.ascontiguousarray(ground), mask, np.ascontiguousarray(normalized)


def pca_s_values(normalized: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    pca = model["pca_s"]
    centered = np.asarray(normalized, dtype=np.float64) - np.asarray(
        [float(pca["center_xn"]), float(pca["center_yn"])], dtype=np.float64
    )
    axis_s = np.asarray([float(pca["axis_s_xn"]), float(pca["axis_s_yn"])], dtype=np.float64)
    return np.asarray(centered @ axis_s, dtype=np.float64)


def reconstruct_pair(
    uv: np.ndarray,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
    spline: Any,
) -> dict[str, Any]:
    lambda_c0, valid_c0 = fixed.lambda_by_input(uv, calibration, reconstruction_params)
    _, _, normalized = ground_points_from_lambda(uv, lambda_c0, valid_c0, calibration)
    s = pca_s_values(normalized, c1_model)
    correction = np.asarray(spline(s), dtype=np.float64)
    lambda_c1 = np.asarray(lambda_c0, dtype=np.float64) + correction
    min_depth = float(reconstruction_params.min_camera_depth_mm)
    max_depth = float(reconstruction_params.max_camera_depth_mm)
    valid_c1 = (
        np.asarray(valid_c0, dtype=bool)
        & np.isfinite(lambda_c1)
        & (lambda_c1 > 0.0)
        & (lambda_c1 >= min_depth)
        & (lambda_c1 <= max_depth)
    )
    ground_c0, valid_c0_final, _ = ground_points_from_lambda(
        uv, lambda_c0, valid_c0, calibration
    )
    ground_c1, valid_c1_final, _ = ground_points_from_lambda(
        uv, lambda_c1, valid_c1, calibration
    )
    return {
        "ground_c0": ground_c0,
        "ground_c1": ground_c1,
        "valid_c0": valid_c0_final,
        "valid_c1": valid_c1_final,
        "lambda_c0": np.asarray(lambda_c0, dtype=np.float64),
        "lambda_c1": lambda_c1,
        "s": s,
        "correction": correction,
    }


def relative_height_result(
    baseline_ground: np.ndarray,
    height_ground: np.ndarray,
    nominal_height_mm: float,
) -> dict[str, Any]:
    measurement = measure_height_line(baseline_ground, height_ground)
    height_inliers = height_ground[measurement.height_fit.inlier_mask]
    profile = measurement.ground_profile_fit
    if profile is None:
        local_ground = np.zeros(len(height_inliers), dtype=np.float64)
    else:
        local_ground = profile.predict_z(height_inliers[:, :2])
    relative_heights = height_inliers[:, 2] - local_ground
    errors = relative_heights - float(nominal_height_mm)
    result = metric(errors)
    result.update(
        {
            "measured_height_mean_mm": float(measurement.height_mean_mm),
            "measured_height_median_mm": float(measurement.height_median_mm),
            "measured_height_std_mm": float(measurement.height_std_mm),
            "height_difference_mae_mm": float(np.mean(np.abs(errors))),
            "height_difference_bias_mm": float(np.mean(errors)),
            "height_difference_rmse_mm": float(np.sqrt(np.mean(errors * errors))),
            "baseline_inlier_count": int(measurement.baseline_inlier_count),
            "height_inlier_count": int(measurement.height_inlier_count),
            "profile_slope_z_per_mm": float(profile.slope_z_per_mm) if profile is not None else math.nan,
            "profile_intercept_z_mm": float(profile.intercept_z_mm) if profile is not None else math.nan,
            "errors": errors,
        }
    )
    return result


def input_record(root: Path, directory: str, label: str, region: str) -> dict[str, Any]:
    path = root / directory
    if not path.is_dir():
        raise FileNotFoundError(path)
    center_path = path / "laser_center.csv"
    baseline_path = path / "baseline_points.csv"
    height_path = path / "height_points.csv"
    result_path = path / "result.json"
    for required in (center_path, baseline_path, height_path, result_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    center_uv = read_uv(center_path)
    baseline_uv = read_uv(baseline_path)
    height_uv = read_uv(height_path)
    center_keys = uv_keys(center_uv)
    missing_baseline = uv_keys(baseline_uv) - center_keys
    missing_height = uv_keys(height_uv) - center_keys
    if missing_baseline or missing_height:
        raise RuntimeError(
            f"{directory}: baseline/height contains UV not present in laser_center.csv "
            f"(baseline={len(missing_baseline)}, height={len(missing_height)})"
        )
    metadata = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "directory": directory,
        "label": label,
        "region": region,
        "path": path,
        "center_path": center_path,
        "baseline_path": baseline_path,
        "height_path": height_path,
        "result_path": result_path,
        "center_uv": center_uv,
        "baseline_uv": baseline_uv,
        "height_uv": height_uv,
        "center_sha256": sha256_file(center_path),
        "baseline_sha256": sha256_file(baseline_path),
        "height_sha256": sha256_file(height_path),
        "result_sha256": sha256_file(result_path),
        "metadata": metadata,
    }


def evaluate_position(
    item: Mapping[str, Any],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
    spline: Any,
    nominal_height_mm: float,
) -> dict[str, Any]:
    baseline_uv = np.asarray(item["baseline_uv"], dtype=np.float64)
    height_uv = np.asarray(item["height_uv"], dtype=np.float64)
    baseline_pair = reconstruct_pair(
        baseline_uv, calibration, reconstruction_params, c1_model, spline
    )
    height_pair = reconstruct_pair(
        height_uv, calibration, reconstruction_params, c1_model, spline
    )
    results: dict[str, Any] = {
        "directory": item["directory"],
        "position": item["label"],
        "region": item["region"],
        "v_median_px": float(np.median(height_uv[:, 1])),
        "v_min_px": float(np.min(height_uv[:, 1])),
        "v_max_px": float(np.max(height_uv[:, 1])),
        "center_point_count": int(len(item["center_uv"])),
        "raw_baseline_count": int(len(baseline_uv)),
        "raw_height_count": int(len(height_uv)),
        "center_sha256": item["center_sha256"],
        "baseline_sha256": item["baseline_sha256"],
        "height_sha256": item["height_sha256"],
        "result_sha256": item["result_sha256"],
        "models": {},
    }
    for model_id, pair in (("C0", (baseline_pair, height_pair, "c0")), ("C1_4k", (baseline_pair, height_pair, "c1"))):
        baseline_result, height_result, suffix = pair
        baseline_ground = baseline_result[f"ground_{suffix}"]
        height_ground = height_result[f"ground_{suffix}"]
        baseline_valid = baseline_result[f"valid_{suffix}"]
        height_valid = height_result[f"valid_{suffix}"]
        if int(np.count_nonzero(baseline_valid)) < 30 or int(np.count_nonzero(height_valid)) < 30:
            raise RuntimeError(
                f"{item['directory']} {model_id}: too few valid points "
                f"(baseline={baseline_valid.sum()}, height={height_valid.sum()})"
            )
        measured = relative_height_result(
            baseline_ground[baseline_valid],
            height_ground[height_valid],
            nominal_height_mm,
        )
        measured.pop("errors")
        measured.update(
            {
                "model_id": model_id,
                "valid_baseline_count": int(np.count_nonzero(baseline_valid)),
                "valid_height_count": int(np.count_nonzero(height_valid)),
                "s_min": float(np.nanmin(height_result["s"])) if len(height_result["s"]) else math.nan,
                "s_max": float(np.nanmax(height_result["s"])) if len(height_result["s"]) else math.nan,
                "s_extrapolated_count": int(
                    np.count_nonzero(
                        (height_result["s"] < float(c1_model["pca_s"]["domain_min"]))
                        | (height_result["s"] > float(c1_model["pca_s"]["domain_max"]))
                    )
                ),
                "mean_c1_correction_mm": float(np.nanmean(height_result["correction"]))
                if model_id == "C1_4k"
                else 0.0,
                "max_abs_c1_correction_mm": float(np.nanmax(np.abs(height_result["correction"])))
                if model_id == "C1_4k"
                else 0.0,
            }
        )
        results["models"][model_id] = measured
    return results


def serialise_position_rows(position_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in position_results:
        for model_id, metrics in item["models"].items():
            row = {
                "record_type": "position",
                "position": item["position"],
                "region": item["region"],
                "directory": item["directory"],
                "v_median_px": item["v_median_px"],
                "v_min_px": item["v_min_px"],
                "v_max_px": item["v_max_px"],
                "model": model_id,
                "nominal_height_mm": metrics.get("nominal_height_mm", math.nan),
                "point_count": metrics["point_count"],
                "bias_mm": metrics["bias_mm"],
                "mae_mm": metrics["mae_mm"],
                "rmse_mm": metrics["rmse_mm"],
                "p95_abs_mm": metrics["p95_abs_mm"],
                "max_abs_mm": metrics["max_abs_mm"],
                "height_difference_mae_mm": metrics["height_difference_mae_mm"],
                "height_difference_bias_mm": metrics["height_difference_bias_mm"],
                "height_difference_rmse_mm": metrics["height_difference_rmse_mm"],
                "measured_height_mean_mm": metrics["measured_height_mean_mm"],
                "measured_height_median_mm": metrics["measured_height_median_mm"],
                "measured_height_std_mm": metrics["measured_height_std_mm"],
                "baseline_inlier_count": metrics["baseline_inlier_count"],
                "height_inlier_count": metrics["height_inlier_count"],
                "valid_baseline_count": metrics["valid_baseline_count"],
                "valid_height_count": metrics["valid_height_count"],
                "profile_slope_z_per_mm": metrics["profile_slope_z_per_mm"],
                "profile_intercept_z_mm": metrics["profile_intercept_z_mm"],
                "s_min": metrics["s_min"],
                "s_max": metrics["s_max"],
                "s_extrapolated_count": metrics["s_extrapolated_count"],
                "mean_c1_correction_mm": metrics["mean_c1_correction_mm"],
                "max_abs_c1_correction_mm": metrics["max_abs_c1_correction_mm"],
                "within_0_2mm": bool(metrics["max_abs_mm"] <= ACCURACY_TARGET_MM),
                "center_sha256": item["center_sha256"],
                "baseline_sha256": item["baseline_sha256"],
                "height_sha256": item["height_sha256"],
            }
            rows.append(row)
    return rows


def aggregate_errors(position_results: Sequence[Mapping[str, Any]], model_id: str, region: str | None = None) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for item in position_results:
        if region is not None and item["region"] != region:
            continue
        metrics = item["models"][model_id]
        # Re-evaluate from the stored point-level metrics is impossible; the
        # evaluator keeps the vectors in a private key until this function is
        # called.  This branch is replaced by evaluate_all's error cache.
        values = metrics.get("_errors")
        if values is not None:
            arrays.append(np.asarray(values, dtype=np.float64))
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    value = ""
                elif isinstance(value, (bool, np.bool_)):
                    value = "true" if bool(value) else "false"
                output[key] = value
            writer.writerow(output)


def regional_rows(position_results: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    position_by_model: dict[str, list[Mapping[str, Any]]] = {"C0": [], "C1_4k": []}
    for item in position_results:
        for model_id, metrics in item["models"].items():
            position_by_model[model_id].append({"position": item["position"], "region": item["region"], "metrics": metrics})
            rows.append(
                {
                    "record_type": "position",
                    "model": model_id,
                    "name": item["position"],
                    "region": item["region"],
                    **{key: value for key, value in metrics.items() if not key.startswith("_")},
                    "within_0_2mm": bool(metrics["max_abs_mm"] <= ACCURACY_TARGET_MM),
                }
            )
    for model_id in ("C0", "C1_4k"):
        for region_name, report_name in (("top", "Top"), ("middle", "Middle"), ("bottom", "Bottom")):
            errors = aggregate_errors(position_results, model_id, region_name)
            m = metric(errors)
            rows.append(
                {
                    "record_type": "region",
                    "model": model_id,
                    "name": report_name,
                    "region": region_name,
                    **m,
                    "within_0_2mm": bool(math.isfinite(m["max_abs_mm"]) and m["max_abs_mm"] <= ACCURACY_TARGET_MM),
                }
            )
        position_metrics = position_by_model[model_id]
        rmse = np.asarray([finite(item["metrics"]["rmse_mm"]) for item in position_metrics])
        p95 = np.asarray([finite(item["metrics"]["p95_abs_mm"]) for item in position_metrics])
        bias = np.asarray([finite(item["metrics"]["bias_mm"]) for item in position_metrics])
        errors = aggregate_errors(position_results, model_id)
        overall = metric(errors)
        summary_row = {
            "record_type": "overall",
            "model": model_id,
            "name": "All_positions",
            "region": "global",
            **overall,
            "worst_position_rmse_mm": float(np.nanmax(rmse)),
            "worst_position_p95_abs_mm": float(np.nanmax(p95)),
            "position_bias_range_mm": float(np.nanmax(bias) - np.nanmin(bias)),
            "position_rmse_range_mm": float(np.nanmax(rmse) - np.nanmin(rmse)),
            "position_p95_range_mm": float(np.nanmax(p95) - np.nanmin(p95)),
            "all_positions_within_0_2mm": bool(
                all(finite(item["metrics"]["max_abs_mm"]) <= ACCURACY_TARGET_MM for item in position_metrics)
            ),
        }
        rows.append(summary_row)
        summary[model_id] = summary_row
    return rows, summary


def report_text(
    standard_root: Path,
    output_dir: Path,
    c1_model_path: Path,
    c1_model_sha256: str,
    c1_parameter_sha256: str,
    c0_info: Mapping[str, Any],
    position_rows: Sequence[Mapping[str, Any]],
    regional: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    nominal_height_mm: float,
    full_fov: str,
    production: str,
    consistency_notes: Sequence[str],
) -> str:
    c0 = summary["C0"]
    c1 = summary["C1_4k"]
    lines = [
        "# Frozen C0 / C1_4k standard-object full-FOV acceptance",
        "",
        f"`FULL_FOV_ACCURACY = {full_fov}`",
        f"`C1_PRODUCTION = {production}`",
        "",
        "## Scope and frozen boundary",
        "",
        f"- 仅读取四个标准件目录：`{standard_root}` 下的 `frame_061303_measure`、`frame_065292_measure`、`frame_063995_measure`、`frame_062878_measure`。",
        "- 未读取 laser-plane Validation（019–024、037–040），未重新拟合 K/D 或 Cone，未训练新 correction。",
        "- C0 = Frozen Circular Cone；C1 = Frozen `C1_4k`，即 `lambda_cone + F(s)`；PCA s、knots、penalty 和 region definition 均未修改。",
        "- C0/C1 对每个位置使用完全相同的既有 `laser_center.csv` 及其中已对应的 baseline/height `u,v` 子集；本轮没有重新提取 laser center。",
        f"- 工程真值按该组约 50 mm 量块的固定标称值 **{nominal_height_mm:.3f} mm** 计算。四个目录的 `result.json` 未含 nominal 字段，因此此值是本报告必须显式记录的外部标准件规格假设。",
        "",
        "## Frozen provenance",
        "",
        f"- C1 model: `{c1_model_path}`",
        f"- C1 model SHA-256: `{c1_model_sha256}`",
        f"- C1 parameter SHA-256: `{c1_parameter_sha256}`",
        f"- Frozen C0 provenance SHA-256: `{c0_info.get('provenance_sha256', '')}`",
        f"- Frozen formal Cone SHA-256: `{c0_info.get('formal_cone_sha256', '')}`",
        "",
        "## Position-level accuracy",
        "",
        "误差定义：`(height point Zg - fitted local baseline Zg) - nominal height`；Bias 为带符号均值，P95/Max 为绝对误差。",
        "",
        "| position | v median | model | n | Bias (mm) | MAE (mm) | RMSE (mm) | P95 (mm) | Max abs (mm) | within 0.2 mm |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in position_rows:
        lines.append(
            f"| {row['position']} | {fmt(row['v_median_px'], 1)} | {row['model']} | {row['point_count']} | "
            f"{fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | "
            f"{fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} | {row['within_0_2mm']} |"
        )
    lines.extend(
        [
            "",
            "## Full-field consistency comparison",
            "",
            "| model | global RMSE | global MAE | worst-position RMSE | worst-position P95 | position bias range | position RMSE range | all positions <= 0.2 mm |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
            f"| C0 | {fmt(c0['rmse_mm'])} | {fmt(c0['mae_mm'])} | {fmt(c0['worst_position_rmse_mm'])} | {fmt(c0['worst_position_p95_abs_mm'])} | {fmt(c0['position_bias_range_mm'])} | {fmt(c0['position_rmse_range_mm'])} | {c0['all_positions_within_0_2mm']} |",
            f"| C1_4k | {fmt(c1['rmse_mm'])} | {fmt(c1['mae_mm'])} | {fmt(c1['worst_position_rmse_mm'])} | {fmt(c1['worst_position_p95_abs_mm'])} | {fmt(c1['position_bias_range_mm'])} | {fmt(c1['position_rmse_range_mm'])} | {c1['all_positions_within_0_2mm']} |",
            "",
            f"- C1 global RMSE change vs C0: **{fmt(pct_change(c0['rmse_mm'], c1['rmse_mm']), 3)}%**; global MAE change: **{fmt(pct_change(c0['mae_mm'], c1['mae_mm']), 3)}%**。负值表示改善。",
            f"- C1 worst-position RMSE change: **{fmt(pct_change(c0['worst_position_rmse_mm'], c1['worst_position_rmse_mm']), 3)}%**；position bias range change: **{fmt(pct_change(c0['position_bias_range_mm'], c1['position_bias_range_mm']), 3)}%**；position RMSE range change: **{fmt(pct_change(c0['position_rmse_range_mm'], c1['position_rmse_range_mm']), 3)}%**。",
            "",
            "## Decision notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in consistency_notes)
    lines.extend(
        [
            "",
            "C1 的生产建议以全位置 Max abs error <= 0.2 mm、最坏位置 RMSE、位置间 Bias/RMSE range 和 Middle 区域不发生明显退化共同判断，而不是只看 pooled global RMSE。",
            "",
            "## Artifacts",
            "",
            f"- `standard_object_accuracy.csv`、`regional_consistency.csv`：`{output_dir}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    standard_root = resolve_existing(
        args.standard_root,
        Path(str(args.standard_root).replace("linelascan", "linelaserscan")),
    )
    c1_model_path = resolve_existing(args.c1_model)
    provenance_path = resolve_existing(args.frozen_provenance)
    formal_cone_path = resolve_existing(args.formal_cone)
    measurement_config = resolve_existing(args.measurement_config)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")
    nominal_height_mm = float(args.nominal_height_mm)
    if not math.isfinite(nominal_height_mm):
        raise ValueError("nominal height must be finite")

    c1_model, c1_model_sha256 = frozen.load_frozen_json(c1_model_path)
    spline = frozen.frozen_spline(c1_model)
    c0_model, c0_info = board.load_frozen_model_checked(provenance_path, formal_cone_path)
    frozen_c0_runtime = c1_model["frozen_cone"]["runtime_model"]
    for key in ("axis_unit_camera", "apex_camera_mm", "half_apex_angle_deg", "z_valid_range_mm"):
        if not np.allclose(np.asarray(c0_model[key], dtype=np.float64), np.asarray(frozen_c0_runtime[key], dtype=np.float64), atol=1.0e-10):
            raise RuntimeError(f"Frozen C0 runtime mismatch in {key}")

    items = [input_record(standard_root, directory, label, region) for directory, label, region in POSITION_ORDER]
    _, calibration, reconstruction_params, _ = fixed.load_runtime(measurement_config)
    calibration = dict(calibration)
    calibration["laser_model"] = c0_model
    if calibration.get("ground_u_compensation") is not None:
        raise RuntimeError("Frozen C1 model declares no ground_u_compensation, but runtime has one")

    position_results: list[dict[str, Any]] = []
    for item in items:
        result = evaluate_position(
            item,
            calibration,
            reconstruction_params,
            c1_model,
            spline,
            nominal_height_mm,
        )
        # Keep point-level errors only in memory for aggregation; never write
        # them as a second raw-data artifact.
        for model_id in ("C0", "C1_4k"):
            uv = np.asarray(item["height_uv"], dtype=np.float64)
            pair = reconstruct_pair(uv, calibration, reconstruction_params, c1_model, spline)
            suffix = "c0" if model_id == "C0" else "c1"
            base_uv = np.asarray(item["baseline_uv"], dtype=np.float64)
            base_pair = reconstruct_pair(base_uv, calibration, reconstruction_params, c1_model, spline)
            hv = pair[f"valid_{suffix}"]
            bv = base_pair[f"valid_{suffix}"]
            m = relative_height_result(
                base_pair[f"ground_{suffix}"][bv],
                pair[f"ground_{suffix}"][hv],
                nominal_height_mm,
            )
            result["models"][model_id]["_errors"] = m["errors"]
        position_results.append(result)

    position_rows = serialise_position_rows(position_results)
    regional, summary = regional_rows(position_results)
    c0_summary = summary["C0"]
    c1_summary = summary["C1_4k"]
    c0_positions = {row["position"]: row for row in position_rows if row["model"] == "C0"}
    c1_positions = {row["position"]: row for row in position_rows if row["model"] == "C1_4k"}

    c1_all_pass = bool(c1_summary["all_positions_within_0_2mm"])
    c1_global_not_worse = bool(c1_summary["rmse_mm"] <= c0_summary["rmse_mm"] and c1_summary["mae_mm"] <= c0_summary["mae_mm"])
    worst_improved = bool(c1_summary["worst_position_rmse_mm"] < c0_summary["worst_position_rmse_mm"])
    bias_range_improved = bool(c1_summary["position_bias_range_mm"] < c0_summary["position_bias_range_mm"])
    rmse_range_improved = bool(c1_summary["position_rmse_range_mm"] < c0_summary["position_rmse_range_mm"])
    consistency_improved = worst_improved and bias_range_improved and rmse_range_improved
    middle_rows_c0 = [c0_positions[name] for name in ("Middle_Upper", "Middle_Lower")]
    middle_rows_c1 = [c1_positions[name] for name in ("Middle_Upper", "Middle_Lower")]
    middle_not_sacrificed = all(
        c1["rmse_mm"] <= max(c0["rmse_mm"] * 1.02, c0["rmse_mm"] + 0.005)
        for c0, c1 in zip(middle_rows_c0, middle_rows_c1)
    )
    if c1_all_pass and c1_global_not_worse and consistency_improved and middle_not_sacrificed:
        full_fov = "PASS"
        production = "YES"
    elif c1_all_pass:
        full_fov = "PARTIAL"
        production = "CONDITIONAL"
    else:
        full_fov = "FAIL"
        production = "NO"

    notes = [
        f"C1 全部四个位置 Max abs error 均 {'满足' if c1_all_pass else '不满足'} {ACCURACY_TARGET_MM:.1f} mm 目标；C0 的结果为 {'全部满足' if c0_summary['all_positions_within_0_2mm'] else '至少一个位置超标'}。",
        f"C1 最坏位置 RMSE {'改善' if worst_improved else '未改善'}，位置 Bias range {'收窄' if bias_range_improved else '未收窄'}，位置 RMSE range {'收窄' if rmse_range_improved else '未收窄'}。",
        f"Middle_Upper/Middle_Lower 相对 C0 的 RMSE {'未出现明显退化' if middle_not_sacrificed else '出现明显退化'}（阈值为 2% 或 0.005 mm 的较宽者）。",
        f"Top 位置 C0→C1 RMSE：{fmt(c0_positions['Top']['rmse_mm'])} → {fmt(c1_positions['Top']['rmse_mm'])} mm；这是 {'退化但仍在 0.2 mm 目标内' if c1_positions['Top']['rmse_mm'] > c0_positions['Top']['rmse_mm'] else '改善'}，不能描述为每个位置都改善。",
        f"Bottom 位置 C0→C1 RMSE：{fmt(c0_positions['Bottom']['rmse_mm'])} → {fmt(c1_positions['Bottom']['rmse_mm'])} mm；其 Max abs error 为 {fmt(c1_positions['Bottom']['max_abs_mm'])} mm。",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "standard_object_accuracy.csv", position_rows)
    write_csv(output_dir / "regional_consistency.csv", regional)
    (output_dir / "report.md").write_text(
        report_text(
            standard_root,
            output_dir,
            c1_model_path,
            c1_model_sha256,
            str(c1_model["parameter_sha256"]),
            c0_info,
            position_rows,
            regional,
            summary,
            nominal_height_mm,
            full_fov,
            production,
            notes,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "FULL_FOV_ACCURACY": full_fov,
                "C1_PRODUCTION": production,
                "c1_model_sha256": c1_model_sha256,
                "c1_parameter_sha256": c1_model["parameter_sha256"],
                "nominal_height_mm": nominal_height_mm,
                "validation_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run(parse_args())
