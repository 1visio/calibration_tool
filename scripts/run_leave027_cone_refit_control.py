#!/usr/bin/env python3
"""Task 4C: FIT-only leave-frame-027-out Circular Cone refit control.

The production M0 is hash-guarded and never written. Validation data are not
loaded. The only fit change relative to Task 3B-2 is removal of frame 027.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

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
    / "leave027_cone_refit_control"
)
TASK3B2_OUTPUT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "cone_local_fullfit"
)
TARGET_FRAME = "027"
MODEL_NAMES = ("M0", "M_local_fullfit", "M_leave027")
REGION_NAMES = ("global", "top_formal_edge", "middle_formal", "bottom_formal_edge")
LEGACY_NAMES = ("theta_axis", "phi_axis", "A_x", "A_y", "A_z", "alpha")
LEGACY_SCALES = np.asarray(
    [task3b2.DEGREE_RAD, task3b2.DEGREE_RAD, 10.0, 10.0, 10.0, 0.1 * task3b2.DEGREE_RAD],
    dtype=np.float64,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4C leave-027-out Cone refit control")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=task3b2.MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def json_default(value: Any) -> Any:
    return task3b2.json_default(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    task3b2.write_csv(path, rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    threshold = probability * float(cumulative[-1])
    return float(x[min(int(np.searchsorted(cumulative, threshold, side="left")), len(x) - 1)])


def metric_with_weights(values: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
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


def region_bounds(region: str) -> tuple[float, float]:
    if region == "global":
        return -float("inf"), float("inf")
    for name, low, high in task3a.REGIONS:
        if name == region:
            return float(low), float(high)
    raise KeyError(region)


def evaluate_models(
    records: Sequence[task3a.FrameRecord],
    models: Mapping[str, Mapping[str, Any]],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]]]:
    residuals: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}
    invalid: dict[str, dict[str, int]] = {name: {} for name in models}
    for name, model in models.items():
        candidate = copy.deepcopy(dict(calibration))
        candidate["laser_model"] = copy.deepcopy(dict(model))
        for record in records:
            values, valid, _ = task4a.lambda_by_input(record.pixels_uv, candidate, reconstruction_params)
            residual = np.full(len(record.truth_points), np.nan, dtype=np.float64)
            residual[valid] = record.truth_points[valid, 2] - values[valid]
            residuals[name][record.frame_id] = residual
            invalid[name][record.frame_id] = int(np.count_nonzero(~valid))
    return residuals, invalid


def metrics_for_scope(
    records: Sequence[task3a.FrameRecord],
    residuals: Mapping[str, np.ndarray],
    region: str,
    weighting: str,
) -> dict[str, float | int]:
    low, high = region_bounds(region)
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    contributing_frames = 0
    for record in records:
        residual = np.asarray(residuals[record.frame_id], dtype=np.float64)
        mask = np.isfinite(residual) & (record.pixels_uv[:, 1] >= low) & (record.pixels_uv[:, 1] < high)
        selected = residual[mask]
        if selected.size == 0:
            continue
        contributing_frames += 1
        values.append(selected)
        if weighting == "point_equal":
            weights.append(np.ones(selected.size, dtype=np.float64))
        elif weighting == "frame_equal":
            weights.append(np.full(selected.size, 1.0 / selected.size, dtype=np.float64))
        else:
            raise ValueError(weighting)
    if not values:
        result = metric_with_weights(np.empty(0), np.empty(0))
    else:
        result = metric_with_weights(np.concatenate(values), np.concatenate(weights))
    result["frame_count"] = contributing_frames
    return result


def comparison_rows(
    all_records: Sequence[task3a.FrameRecord],
    residuals: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    scopes = {
        "baseline_fit_30": list(all_records),
        "diagnostic_fit_29": [record for record in all_records if record.frame_id != TARGET_FRAME],
        "frame027_holdout": [record for record in all_records if record.frame_id == TARGET_FRAME],
    }
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for scope_name, records in scopes.items():
        for weighting in ("point_equal", "frame_equal"):
            for model_name in MODEL_NAMES:
                for region in REGION_NAMES:
                    stats = metrics_for_scope(records, residuals[model_name], region, weighting)
                    row: dict[str, Any] = {
                        "evaluation_scope": scope_name,
                        "weighting": weighting,
                        "residual_definition": "e_lambda=lambda_truth-lambda_model_camera_Z_mm",
                        "model": model_name,
                        "region": region,
                        **stats,
                    }
                    rows.append(row)
                    lookup[(scope_name, weighting, model_name, region)] = row
    for row in rows:
        baseline = lookup[(row["evaluation_scope"], row["weighting"], "M_local_fullfit", row["region"])]
        row["delta_bias_vs_M_local_fullfit_mm"] = float(row["bias"]) - float(baseline["bias"])
        row["delta_rmse_vs_M_local_fullfit_mm"] = float(row["rmse"]) - float(baseline["rmse"])
        row["delta_p95_vs_M_local_fullfit_mm"] = float(row["p95"]) - float(baseline["p95"])
    return rows, lookup


def grid_delta_rows(
    records: Sequence[task3a.FrameRecord],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    model_full: Mapping[str, Any],
    model_leave: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grid_uv, u_values, v_values = task3a.build_grid(records)
    full_cal = copy.deepcopy(dict(calibration)); full_cal["laser_model"] = copy.deepcopy(dict(model_full))
    leave_cal = copy.deepcopy(dict(calibration)); leave_cal["laser_model"] = copy.deepcopy(dict(model_leave))
    lambda_full, valid_full, _ = task4a.lambda_by_input(grid_uv, full_cal, reconstruction_params)
    lambda_leave, valid_leave, _ = task4a.lambda_by_input(grid_uv, leave_cal, reconstruction_params)
    common = valid_full & valid_leave & np.isfinite(lambda_full) & np.isfinite(lambda_leave)
    delta = lambda_leave - lambda_full
    rows: list[dict[str, Any]] = []
    for index, v_value in enumerate(v_values):
        start = index * len(u_values)
        stop = start + len(u_values)
        selected = common[start:stop]
        stats = task4a.metric(delta[start:stop][selected])
        rows.append(
            {
                "v_px": float(v_value),
                "region": task3a.region_for_v(float(v_value)),
                "u_min_px": float(u_values[0]),
                "u_max_px": float(u_values[-1]),
                "grid_u_count": len(u_values),
                "common_valid_count": int(np.count_nonzero(selected)),
                "invalid_count": int(len(u_values) - np.count_nonzero(selected)),
                "delta_definition": "lambda_M_leave027-lambda_M_local_fullfit_camera_Z_mm",
                "bias_delta_lambda_mm": stats["bias"],
                "mae_delta_lambda_mm": stats["mae"],
                "rmse_delta_lambda_mm": stats["rmse"],
                "p95_abs_delta_lambda_mm": stats["p95"],
                "max_abs_delta_lambda_mm": stats["max_abs"],
            }
        )
    summary: dict[str, Any] = {"global": task4a.metric(delta[common])}
    v_flat = grid_uv[:, 1]
    for name, low, high in task3a.REGIONS:
        mask = common & (v_flat >= low) & (v_flat < high)
        summary[name] = task4a.metric(delta[mask])
    summary["common_valid_count"] = int(np.count_nonzero(common))
    summary["invalid_count"] = int(np.count_nonzero(~common))
    return rows, summary


def save_delta_plot(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    selected = [row for row in rows if int(row["common_valid_count"]) > 0]
    v = np.asarray([float(row["v_px"]) for row in selected])
    bias = np.asarray([float(row["bias_delta_lambda_mm"]) for row in selected])
    p95 = np.asarray([float(row["p95_abs_delta_lambda_mm"]) for row in selected])
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(v, bias, color="#2b6cb0", label="mean signed delta")
    axis.plot(v, p95, color="#c05621", linestyle="--", label="P95 |delta|")
    axis.plot(v, -p95, color="#c05621", linestyle="--", alpha=0.45)
    axis.axhline(0.0, color="#555", linewidth=0.8)
    axis.axvline(300.0, color="#777", linewidth=0.8, linestyle=":")
    axis.axvline(2700.0, color="#777", linewidth=0.8, linestyle=":")
    axis.set_xlim(task3a.FORMAL_V_MIN, task3a.FORMAL_V_MAX)
    axis.set_xlabel("v / px")
    axis.set_ylabel("M_leave027 - M_local_fullfit lambda / mm")
    axis.set_title("Leave-027 refit surface change on the frozen evaluation grid")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "leave027_model_delta_vs_v.png", dpi=170)
    plt.close(fig)


def fmt(value: Any) -> str:
    number = float(value)
    return "nan" if not math.isfinite(number) else f"{number:.7g}"


def render_report(
    comparison: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    fit: Mapping[str, Any],
    grid_summary: Mapping[str, Any],
    parameter_payload: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
    cone_hash_before: str,
    cone_hash_after: str,
    output_dir: Path,
) -> str:
    def row(scope: str, model: str, region: str) -> Mapping[str, Any]:
        return comparison[(scope, "point_equal", model, region)]

    top_full = row("baseline_fit_30", "M_local_fullfit", "top_formal_edge")
    top_leave = row("baseline_fit_30", "M_leave027", "top_formal_edge")
    bottom_m0 = row("baseline_fit_30", "M0", "bottom_formal_edge")
    bottom_full = row("baseline_fit_30", "M_local_fullfit", "bottom_formal_edge")
    bottom_leave = row("baseline_fit_30", "M_leave027", "bottom_formal_edge")
    top_improves = float(top_leave["rmse"]) < float(top_full["rmse"]) and float(top_leave["p95"]) < float(top_full["p95"])
    bottom_vs_full_improves = float(bottom_leave["rmse"]) < float(bottom_full["rmse"]) and float(bottom_leave["p95"]) < float(bottom_full["p95"])
    bottom_gain_retained = float(bottom_leave["rmse"]) < float(bottom_m0["rmse"])
    top_sign_persists = float(top_leave["bias"]) > 0.0
    bottom_sign_persists = float(bottom_leave["bias"]) < 0.0
    grid_p95 = float(grid_summary["global"]["p95"])
    normalized_l2 = float(parameter_payload["delta_leave027_minus_local_fullfit"]["local_normalized_l2"])
    local_delta = parameter_payload["delta_leave027_minus_local_fullfit"]["local"]
    legacy_delta = parameter_payload["delta_leave027_minus_local_fullfit"]["legacy"]
    top_rmse_change = float(top_leave["rmse"]) - float(top_full["rmse"])
    top_rmse_change_fraction = top_rmse_change / float(top_full["rmse"])
    bottom_rmse_change = float(bottom_leave["rmse"]) - float(bottom_full["rmse"])
    bottom_rmse_change_fraction = bottom_rmse_change / float(bottom_full["rmse"])
    materially_pulls = grid_p95 >= 0.01 or normalized_l2 >= 3.0
    if materially_pulls and top_sign_persists and bottom_sign_persists:
        choice = "B"
    elif materially_pulls and top_improves and bottom_vs_full_improves:
        choice = "A"
    else:
        choice = "C"

    lines = [
        "# Task 4C — Leave-027-out Circular Cone refit control",
        "",
        "**FIT_ONLY = TRUE**  ",
        "**VALIDATION_OPENED = FALSE**  ",
        "**PRODUCTION_M0_MODIFIED = FALSE**  ",
        "**CORRECTION_OR_PRIOR_ADDED = FALSE**",
        "",
        "## 控制变量",
        "",
        "- Baseline FIT：001–018 + 025–036（30帧）；diagnostic FIT 只临时排除027（29帧）。",
        "- 完全复用 Task 3B-2 的 local parameterization、正式 Cone residual、固定采样、frame-equal weighting、soft_l1、bounds、x_scale、optimizer 和 evaluation grid。",
        "- M0 作为初值且全程冻结；M_local_fullfit 直接读取 Task 3B-2 产物，没有重算或覆盖。",
        f"- 正式 Cone SHA-256 before/after：`{cone_hash_before}` / `{cone_hash_after}`。",
        f"- 独立29帧 refit 与 Task 3B-2 既有 jackknife(027) 参数最大绝对差：`{fmt(reproducibility['max_abs_local_parameter_difference'])}`；cost差：`{fmt(reproducibility['optimizer_cost_difference'])}`。",
        "",
        "## 29帧优化结果",
        "",
        f"- status=`{fit['status']}`, success=`{fit['fit_success']}`, message=`{fit['message']}`。",
        f"- selected points=`{len(fit['selected_points'])}`, objective cost=`{fmt(fit['optimizer_cost'])}`, objective MSE=`{fmt(fit['objective_mse'])}`。",
        f"- M_leave027 相对 M_local_fullfit 的 local normalized delta L2=`{normalized_l2:.6g}`。",
        f"- 冻结 evaluation grid 上 `lambda_leave-lambda_full`：P95=`{fmt(grid_summary['global']['p95'])}` mm，max=`{fmt(grid_summary['global']['max_abs'])}` mm。",
        "",
        "### Cone参数变化（M_leave027 − M_local_fullfit）",
        "",
        "| parameterization | parameter | delta | unit |",
        "|---|---|---:|---|",
        f"| local | theta_axis | {fmt(local_delta['theta_axis'])} | rad |",
        f"| local | phi_axis | {fmt(local_delta['phi_axis'])} | rad |",
        f"| local | c1 | {fmt(local_delta['c1'])} | mm |",
        f"| local | c2 | {fmt(local_delta['c2'])} | mm |",
        f"| local | rho_ref | {fmt(local_delta['rho_ref'])} | mm |",
        f"| local | q | {fmt(local_delta['q'])} | cot(alpha) |",
        f"| legacy | A_x | {fmt(legacy_delta['A_x'])} | mm |",
        f"| legacy | A_y | {fmt(legacy_delta['A_y'])} | mm |",
        f"| legacy | A_z | {fmt(legacy_delta['A_z'])} | mm |",
        f"| legacy | alpha | {fmt(legacy_delta['alpha'])} | rad |",
        "",
        "这些参数变化大部分沿 Task 3B-2 已知的几何弱方向；物理影响应以 evaluation-grid 的 delta_lambda 为准，不能把远端 apex 数十毫米漂移直接解释成测量面移动数十毫米。",
        "",
        "## 全部30帧 truth 上的 e_lambda 对照（point-equal）",
        "",
        "`e_lambda = lambda_truth - lambda_model`。把027仍保留在评估集中，确保这里只改变拟合模型，不改变评估数据。",
        "",
        "| region | model | bias / mm | RMSE / mm | P95 / mm |",
        "|---|---|---:|---:|---:|",
    ]
    for region in REGION_NAMES:
        for model in MODEL_NAMES:
            values = row("baseline_fit_30", model, region)
            lines.append(
                f"| {region} | {model} | {fmt(values['bias'])} | {fmt(values['rmse'])} | {fmt(values['p95'])} |"
            )
    lines += [
        "",
        "## Edge判断",
        "",
        f"- Top 相对30帧 M_local_fullfit：RMSE `{fmt(top_full['rmse'])}` → `{fmt(top_leave['rmse'])}` mm（`{top_rmse_change_fraction:+.3%}`），P95 `{fmt(top_full['p95'])}` → `{fmt(top_leave['p95'])}` mm；**{'仅有轻微数值改善，不能视为top residual被解决' if top_improves else '未同时改善'}**。",
        f"- Bottom 相对30帧 M_local_fullfit：RMSE `{fmt(bottom_full['rmse'])}` → `{fmt(bottom_leave['rmse'])}` mm（`{bottom_rmse_change_fraction:+.3%}`），P95 `{fmt(bottom_full['p95'])}` → `{fmt(bottom_leave['p95'])}` mm；**{'改善' if bottom_vs_full_improves else '相对30帧模型变差'}**。",
        f"- Bottom 相对 M0 的 RMSE 为 `{fmt(bottom_m0['rmse'])}` → `{fmt(bottom_leave['rmse'])}` mm；Task 3B-2 的 bottom 改善**{'保留' if bottom_gain_retained else '未保留'}**。",
        f"- M_leave027 的 edge bias 仍为 top `{fmt(top_leave['bias'])}` mm、bottom `{fmt(bottom_leave['bias'])}` mm；top+/bottom− 结构**{'仍存在' if top_sign_persists and bottom_sign_persists else '未同时保留'}**。",
        "",
        "## 最终回答",
        "",
        f"1. **027 是否显著拉偏 Full-FIT Cone：{'是，有可测的参数与曲面影响' if materially_pulls else '现有FIT控制下证据不足'}。** 参数坐标变化需要结合 grid surface drift 解读，不能仅凭弱方向中的大参数漂移下结论。",
        f"2. **排除027后：top {'只有不到1%的轻微数值改善，系统正偏仍在' if top_improves else '没有在RMSE和P95上同时改善'}；bottom相对30帧模型变差，但相对M0的改善{'仍保留' if bottom_gain_retained else '没有保留'}。**",
        f"3. **选择 {choice}：{'027 有影响，但 top/bottom 系统残差仍存在' if choice == 'B' else ('027 是 combined-fit 的主要干扰源' if choice == 'A' else '证据不足')}。**",
        "",
        "这是一项 FIT-only diagnostic control，不是删除027的授权，也不是 validation accuracy 结论。",
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
        raise RuntimeError(f"Formal M0 hash mismatch before run: {cone_hash_before}")

    # FIT-only loading. No validation loader or validation path is referenced.
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records = task3a.load_old_records()
    extension_records, extension_provenance = task3a.load_extension_records(intrinsics)
    records += extension_records
    if [record.frame_id for record in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT registry does not equal explicit 001-018 + 025-036")
    diagnostic_records = [record for record in records if record.frame_id != TARGET_FRAME]
    if len(records) != 30 or len(diagnostic_records) != 29 or any(record.frame_id == TARGET_FRAME for record in diagnostic_records):
        raise RuntimeError("leave-027 split construction failed")

    try:
        p_ref, _ = task3a.build_reference_anchor(records)
    except AttributeError:
        from run_circular_cone_local_parameterization import build_reference_anchor  # noqa: E402

        p_ref, _ = build_reference_anchor(records)

    fullfit_payload = json.loads((TASK3B2_OUTPUT / "local_fullfit_result.json").read_text(encoding="utf-8"))
    m0_legacy = local.legacy_model_to_theta(calibration["laser_model"])
    m0_local = local.legacy_to_local(m0_legacy, p_ref)
    full_local = np.asarray(fullfit_payload["final_local_theta"], dtype=np.float64)
    full_legacy = np.asarray(fullfit_payload["final_legacy_theta"], dtype=np.float64)
    cone_cfg_root = task3a.triplets.safe_yaml_load(task3b2.FORMAL_FIT_CONFIG)
    cone_cfg = dict(cone_cfg_root["models"]["cone"])

    leave_fit = task3b2.fit_local(diagnostic_records, cone_cfg, p_ref, m0_local)
    if not leave_fit["fit_success"]:
        raise RuntimeError(f"leave-027 optimizer failed: {leave_fit['message']}")
    leave_local = np.asarray(leave_fit["local_theta"], dtype=np.float64)
    leave_legacy = np.asarray(leave_fit["legacy_theta"], dtype=np.float64)

    model_full = copy.deepcopy(dict(fullfit_payload["local_model_as_runtime_mapping"]))
    model_leave = task3b2.model_from_legacy(leave_legacy, calibration["laser_model"], leave_fit["z_range_mm"])
    model_leave["description"] = "M_leave027 diagnostic model; FIT-only, not for deployment"
    models = {
        "M0": copy.deepcopy(dict(calibration["laser_model"])),
        "M_local_fullfit": model_full,
        "M_leave027": model_leave,
    }
    residuals, invalid = evaluate_models(records, models, calibration, reconstruction_params)
    rows, comparison = comparison_rows(records, residuals)
    grid_rows, grid_summary = grid_delta_rows(
        records, calibration, reconstruction_params, model_full, model_leave
    )

    surface_metrics = {
        "M0": task3a.residual_metrics_by_region(
            records,
            {record.frame_id: task3a.cone_scalar_residual(m0_legacy, record.truth_points) for record in records},
        ),
        "M_local_fullfit": task3a.residual_metrics_by_region(
            records,
            {record.frame_id: task3a.cone_scalar_residual(full_legacy, record.truth_points) for record in records},
        ),
        "M_leave027": task3a.residual_metrics_by_region(
            records,
            {record.frame_id: task3a.cone_scalar_residual(leave_legacy, record.truth_points) for record in records},
        ),
    }

    jackknife_rows = list(
        csv.DictReader((TASK3B2_OUTPUT / "local_frame_jackknife.csv").open(encoding="utf-8", newline=""))
    )
    saved_027 = next(row for row in jackknife_rows if row["omitted_frame"] == TARGET_FRAME)
    saved_local = np.asarray([float(saved_027[f"local_{name}"]) for name in task3b2.LOCAL_NAMES])
    reproducibility = {
        "source": str(TASK3B2_OUTPUT / "local_frame_jackknife.csv"),
        "max_abs_local_parameter_difference": float(np.max(np.abs(leave_local - saved_local))),
        "optimizer_cost_difference": float(leave_fit["optimizer_cost"] - float(saved_027["optimizer_cost"])),
        "grid_p95_saved_task3b2_mm": float(saved_027["grid_p95_abs_delta_lambda_mm"]),
        "grid_p95_recomputed_mm": float(grid_summary["global"]["p95"]),
    }

    local_delta = leave_local - full_local
    legacy_delta = leave_legacy - full_legacy
    parameter_payload = {
        "parameterization": "[theta_axis, phi_axis, c1, c2, rho_ref, q]",
        "M0": {
            "local_theta": dict(zip(task3b2.LOCAL_NAMES, m0_local.tolist())),
            "legacy_theta": dict(zip(LEGACY_NAMES, m0_legacy.tolist())),
        },
        "M_local_fullfit": {
            "local_theta": dict(zip(task3b2.LOCAL_NAMES, full_local.tolist())),
            "legacy_theta": dict(zip(LEGACY_NAMES, full_legacy.tolist())),
            "runtime_mapping": model_full,
        },
        "M_leave027": {
            "local_theta": dict(zip(task3b2.LOCAL_NAMES, leave_local.tolist())),
            "legacy_theta": dict(zip(LEGACY_NAMES, leave_legacy.tolist())),
            "runtime_mapping": model_leave,
        },
        "delta_leave027_minus_local_fullfit": {
            "local": dict(zip(task3b2.LOCAL_NAMES, local_delta.tolist())),
            "local_normalized": dict(zip(task3b2.LOCAL_NAMES, (local_delta / task3b2.LOCAL_PROFILE_SCALES).tolist())),
            "local_normalized_l2": float(np.linalg.norm(local_delta / task3b2.LOCAL_PROFILE_SCALES)),
            "legacy": dict(zip(LEGACY_NAMES, legacy_delta.tolist())),
            "legacy_normalized": dict(zip(LEGACY_NAMES, (legacy_delta / LEGACY_SCALES).tolist())),
        },
    }
    payload = {
        "task": "Task 4C leave-027-out Circular Cone refit control",
        "fit_baseline_ids": task3a.FIT_IDS,
        "diagnostic_fit_ids": [record.frame_id for record in diagnostic_records],
        "temporarily_excluded_frame": TARGET_FRAME,
        "validation_ids_not_opened": task3a.VALIDATION_IDS,
        "validation_opened": False,
        "production_writeback": False,
        "correction_polynomial_prior_added": False,
        "formal_cone_sha256_before": cone_hash_before,
        "formal_cone_sha256_after": None,
        "fit_contract": {
            "implementation": "run_circular_cone_local_fullfit.fit_local",
            "model": "formal CircularConeModel residual via local_to_legacy",
            "loss": str(cone_cfg.get("loss", "soft_l1")),
            "f_scale_mm": float(cone_cfg.get("f_scale_mm", 0.1)),
            "weighting": "frame_equal",
            "fit_max_points": int(cone_cfg.get("fit_max_points", 3000)),
            "selected_point_count": len(leave_fit["selected_points"]),
            "max_nfev": task3b2.LOCAL_JACK_MAX_NFEV,
            "x_scale": dict(zip(task3b2.LOCAL_NAMES, task3b2.LOCAL_PROFILE_SCALES.tolist())),
            "bounds": [bound.tolist() for bound in task3b2.local_bounds(cone_cfg)],
            "initialization": "M0 converted with Task 3B-1 legacy_to_local",
            "reference_anchor_uses_baseline_30_fit": True,
        },
        "optimizer": {
            "status": leave_fit["status"],
            "success": leave_fit["fit_success"],
            "message": leave_fit["message"],
            "cost": leave_fit["optimizer_cost"],
            "objective_mse": leave_fit["objective_mse"],
            "z_range_mm": leave_fit["z_range_mm"],
        },
        "parameters": parameter_payload,
        "formal_cone_surface_metrics_on_baseline_30": surface_metrics,
        "evaluation_grid_delta_summary": grid_summary,
        "invalid_intersections": invalid,
        "reproducibility_against_task3b2_jackknife_027": reproducibility,
        "extension_provenance_count": len(extension_provenance),
    }

    write_csv(output_dir / "model_region_comparison.csv", rows)
    write_csv(output_dir / "leave027_model_delta_vs_v.csv", grid_rows)
    save_delta_plot(output_dir, grid_rows)

    cone_hash_after = task3b2.sha256_file(task3b2.FORMAL_CONE)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal M0 changed during Task 4C")
    payload["formal_cone_sha256_after"] = cone_hash_after
    (output_dir / "leave027_cone_refit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(
            comparison,
            leave_fit,
            grid_summary,
            parameter_payload,
            reproducibility,
            cone_hash_before,
            cone_hash_after,
            output_dir,
        ),
        encoding="utf-8",
    )

    print(f"LEAVE027_STATUS={leave_fit['status']}")
    print(f"LEAVE027_COST={leave_fit['optimizer_cost']:.9g}")
    print(f"GRID_P95_DELTA_MM={float(grid_summary['global']['p95']):.9g}")
    print(f"REPRO_MAX_LOCAL_DELTA={reproducibility['max_abs_local_parameter_difference']:.9g}")
    print("VALIDATION_OPENED=False")
    print(f"OUTPUT={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
