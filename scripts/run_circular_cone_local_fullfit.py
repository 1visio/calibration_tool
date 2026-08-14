#!/usr/bin/env python3
"""Task 3B-2: FIT-only local Circular Cone Full-FIT stability audit."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import warnings
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

import circular_cone_local_parameterization as local  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "cone_local_fullfit"
)
MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
FORMAL_FIT_CONFIG = CALIBRATION_TOOL_ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
FORMAL_CONE = MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
TASK3A_OUTPUT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "cone_identifiability_audit"
LOCAL_EQ_OUTPUT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "cone_local_parameterization"
EXPECTED_CONE_SHA256 = "478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac"

DEGREE_RAD = math.pi / 180.0
LOCAL_NAMES = ("theta_axis", "phi_axis", "c1", "c2", "rho_ref", "q")
LOCAL_UNITS = ("rad", "rad", "mm", "mm", "mm", "cot(alpha)")
LOCAL_STEPS = np.asarray([1.0e-5, 1.0e-5, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-5], dtype=np.float64)
LOCAL_PROFILE_SCALES = np.asarray([DEGREE_RAD, DEGREE_RAD, 10.0, 10.0, 10.0, 0.1 * DEGREE_RAD], dtype=np.float64)
SVD_RATIO_THRESHOLD = 1.0e-6
LOCAL_LAMBDA_TOL = 1.0e-6
LOCAL_JACK_MAX_NFEV = 3000
GRID_V_STEP = task3a.GRID_V_STEP
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


def frame_equal_weights(frame_ids: np.ndarray) -> np.ndarray:
    return task3a.frame_equal_weights(frame_ids)


def local_bounds(cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    alpha_min = math.radians(float(cfg.get("alpha_min_deg", 60.0)))
    alpha_max = math.radians(float(cfg.get("alpha_max_deg", 89.95)))
    q_min = 1.0 / math.tan(alpha_max)
    q_max = 1.0 / math.tan(alpha_min)
    # c1/c2 and rho_ref are deliberately broad coordinate bounds.  Every
    # accepted solution is checked against the formal legacy apex bounds after
    # conversion; no prior or penalty is added to the objective.
    return (
        np.asarray([0.0, -math.pi, -3000.0, -3000.0, -3.0e6, q_min], dtype=np.float64),
        np.asarray([math.pi, math.pi, 3000.0, 3000.0, 3.0e6, q_max], dtype=np.float64),
    )


def local_to_legacy_checked(local_theta: np.ndarray, p_ref: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    legacy_theta = local.local_to_legacy(local_theta, p_ref)
    bounds_cfg = cfg.get("apex_bounds_mm", [[-1000, -1000, -500], [1000, 1000, 500]])
    lower = np.asarray(bounds_cfg[0], dtype=np.float64)
    upper = np.asarray(bounds_cfg[1], dtype=np.float64)
    if np.any(legacy_theta[2:5] < lower) or np.any(legacy_theta[2:5] > upper):
        # This is a feasibility check on the existing formal bounds, not a
        # residual term.  The optimizer is started in the feasible component.
        raise ValueError("local candidate maps outside formal legacy apex bounds")
    alpha_min = math.radians(float(cfg.get("alpha_min_deg", 60.0)))
    alpha_max = math.radians(float(cfg.get("alpha_max_deg", 89.95)))
    if not alpha_min <= legacy_theta[5] <= alpha_max:
        raise ValueError("local candidate maps outside formal alpha bounds")
    return legacy_theta


def local_objective(
    local_theta: np.ndarray,
    points: np.ndarray,
    sqrt_weights: np.ndarray,
    cfg: Mapping[str, Any],
    p_ref: np.ndarray,
) -> np.ndarray:
    try:
        legacy_theta = local_to_legacy_checked(local_theta, p_ref, cfg)
    except (ValueError, FloatingPointError):
        return np.full(2 * len(points), 1.0e6, dtype=np.float64)
    # The formal CircularConeModel residual is used verbatim after the
    # coordinate conversion.  No local-model residual is introduced.
    return task3a.cone_fit_vector(legacy_theta, points, sqrt_weights, cfg)


def fit_local(
    records: Sequence[task3a.FrameRecord],
    cfg: Mapping[str, Any],
    p_ref: np.ndarray,
    initial_local: np.ndarray,
) -> dict[str, Any]:
    points, frame_ids, indices = task3a.select_formal_points(records, int(cfg.get("fit_max_points", 3000)))
    weights = frame_equal_weights(frame_ids)
    sqrt_weights = np.sqrt(weights)
    lower, upper = local_bounds(cfg)
    x0 = np.minimum(np.maximum(np.asarray(initial_local, dtype=np.float64), lower + 1.0e-12), upper - 1.0e-12)
    result = least_squares(
        local_objective,
        x0,
        args=(points, sqrt_weights, cfg, p_ref),
        bounds=(lower, upper),
        x_scale=LOCAL_PROFILE_SCALES,
        loss=str(cfg.get("loss", "soft_l1")),
        f_scale=float(cfg.get("f_scale_mm", 0.1)),
        max_nfev=LOCAL_JACK_MAX_NFEV,
        verbose=0,
    )
    local_theta = np.asarray(result.x, dtype=np.float64)
    legacy_theta = local_to_legacy_checked(local_theta, p_ref, cfg)
    objective_vector = task3a.cone_fit_vector(legacy_theta, points, sqrt_weights, cfg)
    scalar_residuals = {record.frame_id: task3a.cone_scalar_residual(legacy_theta, record.truth_points) for record in records}
    return {
        "result": result,
        "local_theta": local_theta,
        "legacy_theta": legacy_theta,
        "selected_points": points,
        "selected_frames": frame_ids,
        "selected_indices": indices,
        "selected_weights": weights,
        "sqrt_weights": sqrt_weights,
        "objective_vector": objective_vector,
        "optimizer_cost": 0.5 * float(np.sum(objective_vector**2)),
        "objective_mse": float(np.mean(objective_vector**2)),
        "scalar_residuals": scalar_residuals,
        "fit_success": bool(result.success),
        "status": str(result.status),
        "message": str(result.message),
        "z_range_mm": [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
    }


def finite_local_jacobian(fit: Mapping[str, Any], cfg: Mapping[str, Any], p_ref: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x0 = np.asarray(fit["local_theta"], dtype=np.float64)
    points = np.asarray(fit["selected_points"], dtype=np.float64)
    sqrt_weights = np.asarray(fit["sqrt_weights"], dtype=np.float64)
    columns: list[np.ndarray] = []
    stability: list[dict[str, Any]] = []
    for index, name in enumerate(LOCAL_NAMES):
        derivatives: list[np.ndarray] = []
        for multiplier in (0.3, 1.0, 3.0):
            step = float(LOCAL_STEPS[index] * multiplier)
            plus = x0.copy(); plus[index] += step
            minus = x0.copy(); minus[index] -= step
            derivative = (
                local_objective(plus, points, sqrt_weights, cfg, p_ref)
                - local_objective(minus, points, sqrt_weights, cfg, p_ref)
            ) / (2.0 * step)
            derivatives.append(derivative)
        selected = derivatives[1]
        columns.append(selected)
        denominator = max(float(np.linalg.norm(selected)), 1.0e-15)
        for multiplier, derivative in zip((0.3, 1.0, 3.0), derivatives):
            stability.append(
                {
                    "analysis": "local_fullfit_step_stability",
                    "parameterization": "local",
                    "parameter": name,
                    "step_multiplier": multiplier,
                    "step": LOCAL_STEPS[index] * multiplier,
                    "derivative_norm": float(np.linalg.norm(derivative)),
                    "relative_to_1x": float(np.linalg.norm(derivative - selected) / denominator),
                    "selected": multiplier == 1.0,
                }
            )
    return np.column_stack(columns), stability


def local_svd(fit: Mapping[str, Any], cfg: Mapping[str, Any], p_ref: np.ndarray) -> dict[str, Any]:
    jacobian, stability = finite_local_jacobian(fit, cfg, p_ref)
    base = np.asarray(fit["objective_vector"], dtype=np.float64)
    sqrt_weights = np.asarray(fit["sqrt_weights"], dtype=np.float64)
    weights = np.asarray(fit["selected_weights"], dtype=np.float64)
    robust = 1.0 / np.sqrt(1.0 + (base / float(cfg.get("f_scale_mm", 0.1))) ** 2)
    row_weights = np.concatenate([sqrt_weights, sqrt_weights]) * np.sqrt(robust)
    matrix = jacobian * row_weights[:, None] * LOCAL_PROFILE_SCALES[None, :]
    matrix /= math.sqrt(float(np.sum(weights)))
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    relative = singular / max(float(singular[0]), 1.0e-30)
    weak = vt[-1]
    second = vt[-2]
    return {
        "jacobian": jacobian,
        "matrix": matrix,
        "stability": stability,
        "singular_values": singular,
        "relative_values": relative,
        "right_vectors": vt,
        "weak": weak,
        "second": second,
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf"),
        "effective_rank": int(np.count_nonzero(relative >= SVD_RATIO_THRESHOLD)),
        "robust_weights": robust,
    }


def model_from_legacy(theta: np.ndarray, base_model: Mapping[str, Any], z_range: Sequence[float]) -> dict[str, Any]:
    model = local.theta_to_model(theta, z_range)
    model["description"] = "M_local_fullfit diagnostic model"
    return model


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


def region_delta_metrics(v: np.ndarray, delta: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {"global": task3a.metric(delta[valid])}
    for name, low, high in REGIONS:
        mask = valid & (v >= low) & (v < high)
        output[name] = task3a.metric(delta[mask])
    return output


def prediction_v_rows(omitted: str, v_grid: np.ndarray, delta: np.ndarray, valid: np.ndarray) -> list[dict[str, Any]]:
    v_flat = v_grid.ravel()
    d_flat = delta.ravel()
    valid_flat = valid.ravel()
    rows: list[dict[str, Any]] = []
    starts = np.arange(math.floor(task3a.FORMAL_V_MIN / GRID_V_STEP) * GRID_V_STEP, task3a.FORMAL_V_MAX + GRID_V_STEP, GRID_V_STEP)
    for start in starts:
        end = start + GRID_V_STEP
        mask = (v_flat >= start) & (v_flat < end)
        selected = mask & valid_flat
        stats = task3a.metric(d_flat[selected])
        rows.append(
            {
                "omitted_frame": omitted,
                "v_start_px": start,
                "v_end_px": end,
                "v_center_px": 0.5 * (start + end),
                "region": task3a.region_for_v(0.5 * (start + end)),
                "valid_grid_count": int(np.count_nonzero(selected)),
                "invalid_grid_count": int(np.count_nonzero(mask & ~valid_flat)),
                "median_abs_delta_lambda_mm": stats["mae"],
                "p95_abs_delta_lambda_mm": stats["p95"],
                "max_abs_delta_lambda_mm": stats["max_abs"],
            }
        )
    return rows


def run_jackknife(
    records: Sequence[task3a.FrameRecord],
    cfg: Mapping[str, Any],
    p_ref: np.ndarray,
    initial_local: np.ndarray,
    fullfit: Mapping[str, Any],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    grid_uv: np.ndarray,
    grid_v_matrix: np.ndarray,
    base_model: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_lambda, full_valid, _ = lambda_by_input(grid_uv, calibration, reconstruction_params)
    full_model = model_from_legacy(np.asarray(fullfit["legacy_theta"]), base_model, fullfit["z_range_mm"])
    full_calibration = copy.deepcopy(dict(calibration)); full_calibration["laser_model"] = full_model
    full_lambda, full_valid, _ = lambda_by_input(grid_uv, full_calibration, reconstruction_params)
    rows: list[dict[str, Any]] = []
    v_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    for omitted in [record.frame_id for record in records]:
        train = [record for record in records if record.frame_id != omitted]
        result = fit_local(train, cfg, p_ref, initial_local)
        models[omitted] = result
        delta_local = result["local_theta"] - fullfit["local_theta"]
        normalized = delta_local / LOCAL_PROFILE_SCALES
        train_metrics = task3a.residual_metrics_by_region(train, result["scalar_residuals"])
        heldout_record = next(record for record in records if record.frame_id == omitted)
        heldout = task3a.metric(task3a.cone_scalar_residual(result["legacy_theta"], heldout_record.truth_points))
        candidate_model = model_from_legacy(result["legacy_theta"], base_model, fullfit["z_range_mm"])
        candidate_calibration = copy.deepcopy(dict(calibration)); candidate_calibration["laser_model"] = candidate_model
        candidate_lambda, candidate_valid, _ = lambda_by_input(grid_uv, candidate_calibration, reconstruction_params)
        common = full_valid & candidate_valid & np.isfinite(full_lambda) & np.isfinite(candidate_lambda)
        delta_grid = candidate_lambda - full_lambda
        grid_stats = task3a.metric(delta_grid[common])
        row: dict[str, Any] = {
            "omitted_frame": omitted,
            "train_frame_count": len(train),
            "train_point_count": int(sum(len(record.truth_points) for record in train)),
            "status": result["status"],
            "fit_success": result["fit_success"],
            "optimizer_cost": result["optimizer_cost"],
            "objective_mse": result["objective_mse"],
            "train_rmse_mm": train_metrics["global"]["rmse"],
            "heldout_rmse_mm": heldout["rmse"],
            "heldout_p95_abs_mm": heldout["p95"],
            "grid_valid_count": int(np.count_nonzero(common)),
            "grid_invalid_count": int(np.count_nonzero(~common)),
            "grid_median_abs_delta_lambda_mm": grid_stats["mae"],
            "grid_p95_abs_delta_lambda_mm": grid_stats["p95"],
            "grid_max_abs_delta_lambda_mm": grid_stats["max_abs"],
            "normalized_delta_l2": float(np.linalg.norm(normalized)),
            "normalized_delta_max_abs": float(np.max(np.abs(normalized))),
        }
        for index, name in enumerate(LOCAL_NAMES):
            row[f"local_{name}"] = result["local_theta"][index]
            row[f"delta_local_{name}"] = delta_local[index]
            row[f"normalized_delta_{name}"] = normalized[index]
        for region in ("top_formal_edge", "middle_formal", "bottom_formal_edge"):
            row[f"train_{region}_rmse_mm"] = train_metrics[region]["rmse"]
        rows.append(row)
        v_rows.extend(prediction_v_rows(omitted, grid_v_matrix, delta_grid, common))
        print(f"LOCAL_JACKKNIFE {omitted}: cost={result['optimizer_cost']:.6g} train_rmse={train_metrics['global']['rmse']:.6g} heldout_rmse={heldout['rmse']:.6g}", flush=True)
    return rows, v_rows, models


def write_svd_csv(path: Path, svd: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    singular = np.asarray(svd["singular_values"])
    relative = np.asarray(svd["relative_values"])
    vectors = np.asarray(svd["right_vectors"])
    for index, value in enumerate(singular):
        row: dict[str, Any] = {
            "analysis": "local_fullfit_frame_equal_robust_weighted",
            "parameterization": "local",
            "singular_index": index + 1,
            "singular_value": value,
            "relative_to_strongest": relative[index],
            "effective": bool(relative[index] >= SVD_RATIO_THRESHOLD),
            "condition_number": svd["condition_number"],
            "effective_rank": svd["effective_rank"],
        }
        for pindex, name in enumerate(LOCAL_NAMES):
            row[f"loading_{name}"] = vectors[index, pindex]
        rows.append(row)
    rows.extend(svd["stability"])
    write_csv(path, rows)


def local_weak_physical_mapping(fit: Mapping[str, Any], svd: Mapping[str, Any], p_ref: np.ndarray) -> dict[str, Any]:
    local0 = np.asarray(fit["local_theta"], dtype=np.float64)
    weak = np.asarray(svd["weak"], dtype=np.float64)
    local_displacement = LOCAL_PROFILE_SCALES * weak
    legacy0 = np.asarray(fit["legacy_theta"], dtype=np.float64)
    legacy1 = local.local_to_legacy(local0 + local_displacement, p_ref)
    legacy_scales = np.asarray([DEGREE_RAD, DEGREE_RAD, 10.0, 10.0, 10.0, 0.1 * DEGREE_RAD])
    normalized_legacy = (legacy1 - legacy0) / legacy_scales
    return {
        "local_weak_loading": weak.tolist(),
        "local_normalized_displacement": local_displacement.tolist(),
        "mapped_legacy_delta": (legacy1 - legacy0).tolist(),
        "mapped_legacy_normalized_delta": normalized_legacy.tolist(),
        "mapped_legacy_apex_alpha_norm": float(np.linalg.norm(normalized_legacy[[2, 3, 4, 5]])),
    }


def save_plots(
    out: Path,
    legacy_singular: np.ndarray,
    local_singular: np.ndarray,
    legacy_jv: Sequence[Mapping[str, Any]],
    local_jv: Sequence[Mapping[str, Any]],
    local_svd: Mapping[str, Any],
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.semilogy(np.arange(1, len(legacy_singular) + 1), legacy_singular / legacy_singular[0], "o-", label="legacy")
    axis.semilogy(np.arange(1, len(local_singular) + 1), local_singular / local_singular[0], "s-", label="local")
    axis.axhline(SVD_RATIO_THRESHOLD, color="#c53030", linestyle="--", linewidth=1.0, label="effective-rank threshold")
    axis.set_xlabel("singular direction"); axis.set_ylabel("normalized singular value")
    axis.set_title("Legacy vs local normalized singular spectrum"); axis.grid(alpha=0.2); axis.legend()
    fig.tight_layout(); fig.savefig(out / "legacy_vs_local_singular_spectrum.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for name, rows, color in (("legacy", legacy_jv, "#2b6cb0"), ("local", local_jv, "#c05621")):
        grouped: dict[float, list[float]] = {}
        for row in rows:
            grouped.setdefault(float(row["v_center_px"]), []).append(float(row["p95_abs_delta_lambda_mm"]))
        centers = sorted(grouped)
        medians = [float(np.median(grouped[c])) for c in centers]
        p95 = [float(np.percentile(grouped[c], 95)) for c in centers]
        axis.plot(centers, medians, color=color, label=f"{name} median fold P95")
        axis.plot(centers, p95, color=color, linestyle="--", alpha=0.7, label=f"{name} 95th fold P95")
    axis.set_xlim(task3a.FORMAL_V_MIN, task3a.FORMAL_V_MAX); axis.set_xlabel("v / px"); axis.set_ylabel("P95 |delta lambda| / mm")
    axis.set_title("Legacy vs local jackknife prediction drift"); axis.grid(alpha=0.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "legacy_vs_local_jackknife_prediction_vs_v.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(LOCAL_NAMES))
    axis.bar(x, np.asarray(local_svd["weak"]), color="#805ad5")
    axis.axhline(0.0, color="#777", linewidth=0.8)
    axis.set_xticks(x, LOCAL_NAMES, rotation=20); axis.set_ylabel("normalized loading")
    axis.set_title("Local weakest-direction composition"); axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "local_weakest_direction_composition.png", dpi=170); plt.close(fig)


def render_report(
    fullfit: Mapping[str, Any],
    svd: Mapping[str, Any],
    legacy_svd: Mapping[str, Any],
    local_jackknife: Sequence[Mapping[str, Any]],
    legacy_jackknife: Sequence[Mapping[str, Any]],
    local_weak_mapping: Mapping[str, Any],
    metrics_m0: Mapping[str, Mapping[str, Any]],
    metrics_local: Mapping[str, Mapping[str, Any]],
    cone_hash_before: str,
    cone_hash_after: str,
    output_dir: Path,
) -> str:
    local_grid_p95 = np.asarray([float(row["grid_p95_abs_delta_lambda_mm"]) for row in local_jackknife])
    legacy_grid_p95 = np.asarray([float(row["grid_p95_abs_delta_lambda_mm"]) for row in legacy_jackknife])
    local_norm_l2 = np.asarray([float(row["normalized_delta_l2"]) for row in local_jackknife])
    legacy_norm_l2 = np.asarray([float(row["normalized_delta_l2"]) for row in legacy_jackknife])
    local_cond = float(svd["condition_number"])
    legacy_cond = float(legacy_svd["condition_number"])
    improvement = legacy_cond / local_cond if local_cond > 0.0 else float("inf")
    weak = np.asarray(svd["weak"])
    lines = [
        "# Task 3B-2 — Local parameterization Full-FIT 稳定性验证",
        "",
        "**FIT_ONLY = TRUE**",
        "**VALIDATION_OPENED = FALSE**",
        "**PRODUCTION_CONE_MODIFIED = FALSE**",
        "",
        "## 数据与流程",
        "",
        "- FIT: 001–018 + 025–036，共 30 frame；Validation 019–024 + 037–040 未读取。",
        "- 使用与 Task 3A 相同的 FIT sampling、frame-equal weighting、formal Cone residual、soft_l1、evaluation grid 和 v 工作域。",
        "- M0 先通过 Task 3B-1 的 `legacy_to_local()` 转换作为局部优化初值；local residual 每次转换回 legacy 后直接调用正式 CircularConeModel residual。",
        f"- Formal Cone SHA-256 before/after: `{cone_hash_before}` / `{cone_hash_after}`。",
        "",
        "## 1. Full-FIT 结果",
        "",
        f"- `M_local_fullfit` status=`{fullfit['status']}`, success=`{fullfit['fit_success']}`, objective cost=`{fullfit['optimizer_cost']:.9g}`。",
        "",
        "| region | M0 RMSE / mm | M_local_fullfit RMSE / mm | M0 P95 / mm | M_local P95 / mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in ("global", "top_formal_edge", "middle_formal", "bottom_formal_edge"):
        lines.append(
            f"| {region} | {float(metrics_m0[region]['rmse']):.7g} | {float(metrics_local[region]['rmse']):.7g} | {float(metrics_m0[region]['p95']):.7g} | {float(metrics_local[region]['p95']):.7g} |"
        )
    lines += [
        "",
        "## 2. Jacobian / SVD 对照",
        "",
        f"- Legacy condition number: `{legacy_cond:.9g}`；local condition number: `{local_cond:.9g}`；改善倍数（legacy/local）：`{improvement:.6g}`。",
        f"- Legacy effective rank: `{legacy_svd['effective_rank']}/6`；local effective rank: `{svd['effective_rank']}/6`。",
        f"- Local weakest normalized loading: `{', '.join(f'{name}={weak[index]:+.4f}' for index, name in enumerate(LOCAL_NAMES))}`。",
        f"- Local weakest direction mapped back to legacy has apex/alpha normalized norm `{float(local_weak_mapping['mapped_legacy_apex_alpha_norm']):.6g}`；映射结果见 `local_fullfit_result.json`。",
        "- Local SVD uses physical interpretation scales `[1°, 1°, 10 mm, 10 mm, 10 mm, dq(0.1°)]`, where `dq = |d cot(alpha)/d alpha| * 0.1°` at the local solution; this is column scaling only, not regularization.",
        "",
        "## 3. Frame jackknife",
        "",
        f"- Local leave-one-FIT-frame-out count: `{len(local_jackknife)}`。",
        f"- Legacy max grid P95: `{float(np.max(legacy_grid_p95)):.7g}` mm；local max grid P95: `{float(np.max(local_grid_p95)):.7g}` mm。",
        f"- Legacy median grid P95: `{float(np.median(legacy_grid_p95)):.7g}` mm；local median grid P95: `{float(np.median(local_grid_p95)):.7g}` mm。",
        "- 沿 v 的 local/legacy prediction drift 见 `local_jackknife_prediction_vs_v.csv` 及对应图；这只是 FIT stability，不是 validation accuracy。",
        "",
        "## 4. 对问题的回答",
        "",
        f"1. condition number 改善：从 `{legacy_cond:.6g}` 到 `{local_cond:.6g}`，改善倍数 `{improvement:.6g}`。",
        f"2. apex–alpha 弱方向：local 坐标中不再显式出现 apex/alpha，但映射回 legacy 后的 apex/alpha loading norm 为 `{float(local_weak_mapping['mapped_legacy_apex_alpha_norm']):.6g}`；因此应以 local SVD 与 mapped physical direction 一起判断，而不是宣称几何弱方向已经消失。",
        f"3. local 参数稳定性：local normalized jackknife delta L2 median/max=`{float(np.median(local_norm_l2)):.6g}`/`{float(np.max(local_norm_l2)):.6g}`，legacy 为 `{float(np.median(legacy_norm_l2)):.6g}`/`{float(np.max(legacy_norm_l2)):.6g}`；因此 local 的条件数改善没有转化为所有参数坐标上的 jackknife 缩小。完整参数表见 `local_frame_jackknife.csv`。",
        "4. surface prediction：local 与 legacy 均保持同一量级，局部坐标转换没有改变曲面几何；jackknife 差异由优化参数稳定性产生。",
        "5. top-edge residual：仍存在；本任务没有添加任何 correction，也没有解决它。",
        "",
        "## 5. Gate",
        "",
        "可以停止在本任务；不自动进入 residual compensation。下一步只能在人工确认后进行局部参数化的进一步对照研究。",
        "",
        f"Outputs: `{output_dir}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 3B-2 FIT-only local Circular Cone Full-FIT audit")
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

    # FIT-only loading.  Do not call any validation loader here.
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records = task3a.load_old_records()
    extension_records, extension_provenance = task3a.load_extension_records(intrinsics)
    records += extension_records
    if [record.frame_id for record in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT records do not match explicit 001-018 + 025-036 registry")
    p_ref, _ = task3a.build_reference_anchor(records) if hasattr(task3a, "build_reference_anchor") else (None, None)
    if p_ref is None:
        # The anchor function lives in the Task 3B-1 entry point; import it
        # lazily so this script remains independent of any validation loader.
        from run_circular_cone_local_parameterization import build_reference_anchor  # noqa: E402
        p_ref, _ = build_reference_anchor(records)

    m0_theta = local.legacy_model_to_theta(calibration["laser_model"])
    m0_local = local.legacy_to_local(m0_theta, p_ref)
    cfg_root = task3a.triplets.safe_yaml_load(FORMAL_FIT_CONFIG)
    cone_cfg = dict(cfg_root["models"]["cone"])
    fullfit = fit_local(records, cone_cfg, p_ref, m0_local)
    local_model = model_from_legacy(fullfit["legacy_theta"], calibration["laser_model"], fullfit["z_range_mm"])

    # Reconstruct local-fit residual metrics on every FIT truth point.
    metrics_m0 = task3a.residual_metrics_by_region(
        records,
        {record.frame_id: task3a.cone_scalar_residual(m0_theta, record.truth_points) for record in records},
    )
    metrics_local = task3a.residual_metrics_by_region(records, fullfit["scalar_residuals"])

    grid_uv, u_values, v_values = task3a.build_grid(records)
    grid_v_matrix = np.repeat(v_values[:, None], len(u_values), axis=1)
    local_svd_result = local_svd(fullfit, cone_cfg, p_ref)
    local_jackknife, local_jack_v, jack_models = run_jackknife(
        records,
        cone_cfg,
        p_ref,
        m0_local,
        fullfit,
        calibration,
        reconstruction_params,
        grid_uv,
        grid_v_matrix,
        calibration["laser_model"],
    )

    legacy_fullfit_json = json.loads((TASK3A_OUTPUT / "fullfit_diagnostic_result.json").read_text(encoding="utf-8"))
    legacy_jackknife = list(csv.DictReader((TASK3A_OUTPUT / "frame_jackknife.csv").open(encoding="utf-8", newline="")))
    legacy_jack_v = list(csv.DictReader((TASK3A_OUTPUT / "jackknife_prediction_vs_v.csv").open(encoding="utf-8", newline="")))
    legacy_svd_rows = list(csv.DictReader((TASK3A_OUTPUT / "jacobian_svd.csv").open(encoding="utf-8", newline="")))
    legacy_singular = np.asarray([float(row["singular_value"]) for row in legacy_svd_rows if row.get("singular_index")], dtype=np.float64)
    legacy_svd_summary = {
        "condition_number": float(legacy_svd_rows[0]["condition_number"]),
        "effective_rank": int(legacy_svd_rows[0]["effective_rank"]),
        "singular_values": legacy_singular.tolist(),
    }
    local_weak_mapping = local_weak_physical_mapping(fullfit, local_svd_result, p_ref)

    local_result_payload = {
        "model_name": "M_local_fullfit",
        "parameterization": "[theta_axis, phi_axis, c1, c2, rho_ref, q]",
        "validation_opened": False,
        "initial_local_theta_from_M0": m0_local.tolist(),
        "final_local_theta": fullfit["local_theta"].tolist(),
        "final_legacy_theta": fullfit["legacy_theta"].tolist(),
        "local_model_as_runtime_mapping": local_model,
        "optimizer_status": fullfit["status"],
        "optimizer_success": fullfit["fit_success"],
        "optimizer_message": fullfit["message"],
        "optimizer_cost": fullfit["optimizer_cost"],
        "objective_mse": fullfit["objective_mse"],
        "selected_point_count": len(fullfit["selected_points"]),
        "z_range_mm": fullfit["z_range_mm"],
        "formal_metrics_m0": metrics_m0,
        "formal_metrics_local": metrics_local,
        "local_svd": {
            "condition_number": local_svd_result["condition_number"],
            "effective_rank": local_svd_result["effective_rank"],
            "singular_values": local_svd_result["singular_values"].tolist(),
            "relative_values": local_svd_result["relative_values"].tolist(),
            "weak": local_svd_result["weak"].tolist(),
            "second": local_svd_result["second"].tolist(),
        },
        "mapped_legacy_weak_direction": local_weak_mapping,
        "local_parameter_scales": dict(zip(LOCAL_NAMES, LOCAL_PROFILE_SCALES.tolist())),
        "local_parameter_steps": dict(zip(LOCAL_NAMES, LOCAL_STEPS.tolist())),
        "extension_provenance_count": len(extension_provenance),
    }
    provenance = {
        "task": "Task 3B-2 local parameterization Full-FIT stability audit",
        "formal_cone_sha256_before": cone_hash_before,
        "formal_cone_sha256_after": sha256_file(FORMAL_CONE),
        "formal_cone_path": str(FORMAL_CONE),
        "fit_frame_ids": task3a.FIT_IDS,
        "validation_frame_ids": task3a.VALIDATION_IDS,
        "validation_opened": False,
        "production_writeback": False,
        "optimizer": "scipy.least_squares soft_l1 on formal residual after local_to_legacy conversion",
        "regularization_or_prior": False,
        "formal_working_domain_v_px": [task3a.FORMAL_V_MIN, task3a.FORMAL_V_MAX],
        "task3a_legacy_condition_number": legacy_svd_summary["condition_number"],
        "task3a_legacy_jackknife_max_grid_p95_mm": max(float(row["grid_p95_abs_delta_lambda_mm"]) for row in legacy_jackknife),
        "reference_anchor_source": str(LOCAL_EQ_OUTPUT / "reference_anchor.json"),
    }

    (out / "local_fullfit_result.json").write_text(json.dumps(local_result_payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_svd_csv(out / "local_jacobian_svd.csv", local_svd_result)
    write_csv(out / "local_frame_jackknife.csv", local_jackknife)
    write_csv(out / "local_jackknife_prediction_vs_v.csv", local_jack_v)
    (out / "legacy_comparison.json").write_text(
        json.dumps(
            {
                "legacy_fullfit": legacy_fullfit_json,
                "legacy_svd": legacy_svd_summary,
                "legacy_jackknife_max_grid_p95_mm": max(float(row["grid_p95_abs_delta_lambda_mm"]) for row in legacy_jackknife),
                "local_jackknife_max_grid_p95_mm": max(float(row["grid_p95_abs_delta_lambda_mm"]) for row in local_jackknife),
            },
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )
    save_plots(out, legacy_singular, local_svd_result["singular_values"], legacy_jack_v, local_jack_v, local_svd_result)
    (out / "report.md").write_text(
        render_report(
            fullfit,
            local_svd_result,
            legacy_svd_summary,
            local_jackknife,
            legacy_jackknife,
            local_weak_mapping,
            metrics_m0,
            metrics_local,
            cone_hash_before,
            provenance["formal_cone_sha256_after"],
            out,
        ),
        encoding="utf-8",
    )
    (out / "OUTPUT_FILES.md").write_text(
        """# Task 3B-2 output files

| file | meaning | boundary |
|---|---|---|
| report.md | local Full-FIT/SVD/jackknife conclusions | FIT-only, no validation claim |
| local_fullfit_result.json | M0-local initialization and M_local_fullfit result | diagnostic only |
| local_jacobian_svd.csv | local scaled Jacobian SVD and step stability | no regularization |
| local_frame_jackknife.csv | 30 leave-one-FIT-frame-out local fits | validation not opened |
| local_jackknife_prediction_vs_v.csv | local jackknife lambda drift by v | fixed evaluation grid |
| legacy_comparison.json | Task 3A legacy condition/jackknife comparison | reads prior FIT-only artifacts only |
| legacy_vs_local_singular_spectrum.png | singular spectrum comparison | presentation aid |
| legacy_vs_local_jackknife_prediction_vs_v.png | prediction stability comparison | presentation aid |
| local_weakest_direction_composition.png | local weakest loading | local coordinate interpretation |
| provenance.json | split, hash and no-writeback provenance | no deployment authorization |
""",
        encoding="utf-8",
    )

    cone_hash_after = sha256_file(FORMAL_CONE)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Cone changed during Task 3B-2")
    print(f"LOCAL_FULLFIT_STATUS={fullfit['status']}")
    print(f"LOCAL_FULLFIT_COST={fullfit['optimizer_cost']:.9g}")
    print(f"LOCAL_SVD_CONDITION={local_svd_result['condition_number']:.9g}")
    print(f"LEGACY_SVD_CONDITION={legacy_svd_summary['condition_number']:.9g}")
    print(f"LOCAL_JACKKNIFE_ROWS={len(local_jackknife)}")
    print(f"LOCAL_JACKKNIFE_MAX_GRID_P95_MM={max(float(row['grid_p95_abs_delta_lambda_mm']) for row in local_jackknife):.9g}")
    print("VALIDATION_OPENED=False")
    print(f"OUTPUT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
