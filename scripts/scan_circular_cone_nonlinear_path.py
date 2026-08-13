#!/usr/bin/env python3
"""FIT-only exact nonlinear scan along the three frozen sensitivity directions.

This is a decision-gate experiment, not an optimizer.  Candidate Circular Cone
parameters are installed only in an in-memory calibration dictionary and are
evaluated through the public ``reconstruct_uv_to_ground`` implementation.
Validation frames are not reconstructed or scored.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyze_circular_cone_parameter_sensitivity as sensitivity  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_SENSITIVITY_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_parameter_sensitivity"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_nonlinear_path_scan"
)
OUTPUT_NAMES = (
    "cone_nonlinear_path_scan.csv",
    "cone_nonlinear_path_regions.csv",
    "cone_nonlinear_path_scan.png",
    "cone_nonlinear_path_scan_report.md",
    "OUTPUT_FILES.md",
)
T_VALUES = tuple(index / 128.0 for index in range(17)) + (
    3.0 / 16.0,
    1.0 / 4.0,
    1.0 / 2.0,
    3.0 / 4.0,
    1.0,
)
LOWER_BOUNDS = np.asarray(
    [0.0, -math.pi, -1000.0, -1000.0, -500.0, math.radians(60.0)],
    dtype=np.float64,
)
UPPER_BOUNDS = np.asarray(
    [math.pi, math.pi, 1000.0, 1000.0, 500.0, math.radians(89.95)],
    dtype=np.float64,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FIT-only exact nonlinear scan along Circular Cone sensitivity directions."
    )
    parser.add_argument("--data-root", type=Path, default=sensitivity.paired.DEFAULT_DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=sensitivity.paired.DEFAULT_PNP_AUDIT)
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=sensitivity.paired.DEFAULT_MEASUREMENT_CONFIG,
    )
    parser.add_argument("--sensitivity-dir", type=Path, default=DEFAULT_SENSITIVITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_reference_solutions(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows: dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["parameter"]] = row
    names = [spec.name for spec in sensitivity.PARAMETERS]
    if set(rows) != set(names):
        raise RuntimeError(f"Unexpected sensitivity parameter rows: {sorted(rows)}")
    theta0 = np.asarray([float(rows[name]["current_value"]) for name in names])
    deltas = {
        weighting: np.asarray(
            [float(rows[name][f"delta_{weighting}"]) for name in names],
            dtype=np.float64,
        )
        for weighting in sensitivity.WEIGHTINGS
    }
    return theta0, deltas


def add_metrics(row: dict[str, Any], prefix: str, metrics: sensitivity.MetricSet) -> None:
    row[f"{prefix}_bias_mm"] = metrics.bias_mm
    row[f"{prefix}_mae_mm"] = metrics.mae_mm
    row[f"{prefix}_rmse_mm"] = metrics.rmse_mm
    row[f"{prefix}_p95_abs_mm"] = metrics.p95_abs_mm
    row[f"{prefix}_residual_energy"] = metrics.residual_energy


def empty_metrics(row: dict[str, Any], prefix: str) -> None:
    for suffix in (
        "bias_mm",
        "mae_mm",
        "rmse_mm",
        "p95_abs_mm",
        "residual_energy",
    ):
        row[f"{prefix}_{suffix}"] = float("nan")


def scan_paths(
    data: sensitivity.PreparedData,
    jacobians: sensitivity.JacobianResult,
    solutions: Mapping[str, sensitivity.LinearSolution],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline, v_px, frame_ids = sensitivity.split_arrays(data, "fit")
    if not np.all(jacobians.fit_valid_mask):
        raise RuntimeError("Path scan requires a fully valid FIT Jacobian")
    if len(baseline) != len(jacobians.fit_jacobian):
        raise RuntimeError("FIT baseline and Jacobian lengths differ")

    global_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for weighting in sensitivity.WEIGHTINGS:
        solution = solutions[weighting]
        weights = sensitivity.weights_for(v_px, frame_ids, weighting)
        before = sensitivity.calculate_metrics(baseline, weights)
        linear_direction = jacobians.fit_jacobian @ solution.delta

        for t_value in T_VALUES:
            theta = data.theta0 + t_value * solution.delta
            bounds_ok = bool(np.all(theta >= LOWER_BOUNDS) and np.all(theta <= UPPER_BOUNDS))
            exact = sensitivity.evaluate_candidate(theta, data, "fit")
            invalid_count = int(np.count_nonzero(exact.invalid_mask))
            linear_residual = baseline + t_value * linear_direction
            linear = sensitivity.calculate_metrics(linear_residual, weights)
            row: dict[str, Any] = {
                "weighting": weighting,
                "t": t_value,
                "total_points": len(baseline),
                "invalid_count": invalid_count,
                "valid_rate": 1.0 - invalid_count / float(len(baseline)),
                "bounds_ok": bounds_ok,
                "axis_norm": float(
                    np.linalg.norm(sensitivity.angles_to_axis(theta[0], theta[1]))
                ),
                "max_normalized_parameter_displacement": float(
                    np.max(
                        np.abs(t_value * solution.delta / sensitivity.PARAMETER_SCALES)
                    )
                ),
                "theta_axis_rad": theta[0],
                "phi_axis_rad": theta[1],
                "A_x_mm": theta[2],
                "A_y_mm": theta[3],
                "A_z_mm": theta[4],
                "alpha_rad": theta[5],
                "alpha_deg": math.degrees(float(theta[5])),
            }
            add_metrics(row, "before", before)
            add_metrics(row, "linear", linear)
            row["linear_explained_fraction"] = (
                (before.residual_energy - linear.residual_energy)
                / before.residual_energy
            )

            if invalid_count == 0 and bounds_ok:
                exact_metrics = sensitivity.calculate_metrics(exact.residual_mm, weights)
                gap = sensitivity.calculate_metrics(
                    exact.residual_mm - linear_residual, weights
                )
                add_metrics(row, "exact", exact_metrics)
                add_metrics(row, "linearization_gap", gap)
                exact_explained = (
                    before.residual_energy - exact_metrics.residual_energy
                ) / before.residual_energy
                linear_gain = before.residual_energy - linear.residual_energy
                row["exact_explained_fraction"] = exact_explained
                row["exact_to_linear_gain_ratio"] = (
                    (before.residual_energy - exact_metrics.residual_energy) / linear_gain
                    if linear_gain > 0.0
                    else float("nan")
                )
                row["gap_to_baseline_rmse"] = gap.rmse_mm / before.rmse_mm
            else:
                empty_metrics(row, "exact")
                empty_metrics(row, "linearization_gap")
                row["exact_explained_fraction"] = float("nan")
                row["exact_to_linear_gain_ratio"] = float("nan")
                row["gap_to_baseline_rmse"] = float("nan")
            global_rows.append(row)

            for region, v_min, v_max in sensitivity.region_definitions()[1:]:
                assert v_min is not None and v_max is not None
                mask = (v_px >= v_min) & (v_px < v_max)
                if not np.any(mask):
                    continue
                region_invalid = int(np.count_nonzero(exact.invalid_mask[mask]))
                region_before = sensitivity.calculate_metrics(baseline[mask], weights[mask])
                region_linear = sensitivity.calculate_metrics(
                    linear_residual[mask], weights[mask]
                )
                region_row: dict[str, Any] = {
                    "weighting": weighting,
                    "t": t_value,
                    "region": region,
                    "v_min_px": v_min,
                    "v_max_px": v_max,
                    "sample_count": int(np.count_nonzero(mask)),
                    "invalid_count": region_invalid,
                }
                add_metrics(region_row, "before", region_before)
                add_metrics(region_row, "linear", region_linear)
                if region_invalid == 0 and bounds_ok:
                    region_exact = sensitivity.calculate_metrics(
                        exact.residual_mm[mask], weights[mask]
                    )
                    region_gap = sensitivity.calculate_metrics(
                        exact.residual_mm[mask] - linear_residual[mask], weights[mask]
                    )
                    add_metrics(region_row, "exact", region_exact)
                    add_metrics(region_row, "linearization_gap", region_gap)
                    region_row["exact_explained_fraction"] = (
                        region_before.residual_energy - region_exact.residual_energy
                    ) / region_before.residual_energy
                else:
                    empty_metrics(region_row, "exact")
                    empty_metrics(region_row, "linearization_gap")
                    region_row["exact_explained_fraction"] = float("nan")
                region_rows.append(region_row)

    zero_rows = [row for row in global_rows if float(row["t"]) == 0.0]
    if len(zero_rows) != len(sensitivity.WEIGHTINGS):
        raise RuntimeError("Missing t=0 path rows")
    for row in zero_rows:
        if int(row["invalid_count"]) != 0 or not math.isclose(
            float(row["exact_rmse_mm"]),
            float(row["before_rmse_mm"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("Exact t=0 reconstruction does not reproduce the baseline")
    return global_rows, region_rows


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: sensitivity.csv_value(row[field]) for field in fields})


def select_best_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for weighting in sensitivity.WEIGHTINGS:
        candidates = [
            row
            for row in rows
            if row["weighting"] == weighting
            and int(row["invalid_count"]) == 0
            and bool(row["bounds_ok"])
            and np.isfinite(float(row["exact_residual_energy"]))
        ]
        if not candidates:
            raise RuntimeError(f"No feasible path point for {weighting}")
        selected[weighting] = min(
            candidates, key=lambda row: float(row["exact_residual_energy"])
        )
    return selected


def path_verdict(
    global_rows: Sequence[Mapping[str, Any]],
    best_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, str, list[str]]:
    all_valid = all(int(row["invalid_count"]) == 0 for row in global_rows)
    best_explained = {
        weighting: float(best_rows[weighting]["exact_explained_fraction"])
        for weighting in sensitivity.WEIGHTINGS
    }
    t1_explained = {
        weighting: float(
            next(
                row
                for row in global_rows
                if row["weighting"] == weighting and float(row["t"]) == 1.0
            )["exact_explained_fraction"]
        )
        for weighting in sensitivity.WEIGHTINGS
    }
    local_descent = (
        "CONFIRMED"
        if all_valid and min(best_explained.values()) > 0.0
        else "PARTIAL"
        if max(best_explained.values()) > 0.0
        else "REJECTED"
    )
    full_step = (
        "CONFIRMED"
        if all_valid and min(t1_explained.values()) > 0.0
        else "REJECTED"
    )
    if full_step == "CONFIRMED":
        decision = "FULL_STEP_COMPATIBLE"
    elif local_descent == "CONFIRMED":
        decision = "DAMPED_RELINEARIZATION_REQUIRED"
    else:
        decision = "DIRECTION_REJECTED"
    reasons = [
        "all scanned candidates retained every FIT intersection" if all_valid else "some scanned candidates lost FIT intersections",
        "best exact explained fraction: "
        + ", ".join(f"{key}={value:.6f}" for key, value in best_explained.items()),
        "t=1 exact explained fraction: "
        + ", ".join(f"{key}={value:.6f}" for key, value in t1_explained.items()),
    ]
    return full_step, local_descent, decision, reasons


def save_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {
        "point_equal": "#2b6cb0",
        "frame_equal": "#2f855a",
        "v_region_equal": "#c05621",
    }
    figure, axes = plt.subplots(3, 1, figsize=(9.4, 10.2))
    for weighting in sensitivity.WEIGHTINGS:
        selected = sorted(
            [row for row in rows if row["weighting"] == weighting],
            key=lambda row: float(row["t"]),
        )
        t_values = np.asarray([float(row["t"]) for row in selected])
        exact_rmse = np.asarray([float(row["exact_rmse_mm"]) for row in selected])
        linear_rmse = np.asarray([float(row["linear_rmse_mm"]) for row in selected])
        gap = np.asarray([float(row["gap_to_baseline_rmse"]) for row in selected])
        axes[0].plot(
            t_values,
            exact_rmse,
            color=colors[weighting],
            marker="o",
            linewidth=2.0,
            label=f"{weighting} exact",
        )
        axes[0].plot(
            t_values,
            linear_rmse,
            color=colors[weighting],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"{weighting} linear",
        )
        zoom = t_values <= 0.125
        axes[1].plot(
            t_values[zoom],
            exact_rmse[zoom],
            color=colors[weighting],
            marker="o",
            linewidth=2.0,
            label=f"{weighting} exact",
        )
        axes[1].plot(
            t_values[zoom],
            linear_rmse[zoom],
            color=colors[weighting],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"{weighting} linear",
        )
        axes[2].plot(
            t_values,
            gap,
            color=colors[weighting],
            marker="o",
            linewidth=1.8,
            label=weighting,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("weighted FIT RMSE / mm (log)")
    axes[0].set_title("Full path: exact production reconstruction vs first-order prediction")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlim(0.0, 0.125)
    axes[1].set_ylabel("weighted FIT RMSE / mm")
    axes[1].set_title("Zoom: damped descent region")
    axes[1].grid(alpha=0.25)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("path fraction t in Theta0 + t DeltaTheta")
    axes[2].set_ylabel("RMS(exact-linear) / baseline RMSE (log)")
    axes[2].set_title("Nonlinearity gap")
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    data: sensitivity.PreparedData,
    solutions: Mapping[str, sensitivity.LinearSolution],
    global_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    best_rows: Mapping[str, Mapping[str, Any]],
    full_step: str,
    local_descent: str,
    decision: str,
    reasons: Sequence[str],
    cone_path: Path,
    cone_hash: str,
) -> str:
    by_key = {
        (str(row["weighting"]), float(row["t"]), str(row["region"])): row
        for row in region_rows
    }
    lines = [
        "# Circular Cone exact nonlinear path scan",
        "",
        f"**FULL_LINEAR_STEP = {full_step}**  ",
        f"**LOCAL_DESCENT_DIRECTION = {local_descent}**  ",
        f"**PATH_SCAN_DECISION = {decision}**",
        "",
        "本步骤是 FIT-only 决策门，不是非线性优化。仅扫描 `Theta(t)=Theta0+t*DeltaTheta`；"
        "所有 candidate 只存在内存，并通过正式 `reconstruct_uv_to_ground()` 计算。",
        "",
        "## 隔离与复现",
        "",
        f"- 只重建 FIT 001–010，共 {len(data.residual_mm)} 个冻结 laser points。",
        "- VALIDATION 011–013 图像未打开、未重建、未评分，也没有参与路径、阈值或判定。",
        f"- FIT frozen-pixel SHA-256：`{data.used_pixel_sha256}`。",
        "- 三组 `DeltaTheta` 在本次 FIT-only 重算后与 sensitivity CSV 数值一致。",
        "- 扫描点：`t=0…1/8` 以 `1/128` 细扫，另含 `3/16, 1/4, 1/2, 3/4, 1`。",
        "- 保持正式 runtime depth/range/root-selection 配置；没有调用拟合脚本私有求交实现。",
        "",
        "## Global exact result",
        "",
        "| weighting | best t | baseline RMSE | best exact RMSE | exact RMSE at t=1 | exact explained at best | exact explained at t=1 | gap/baseline at best | invalid over path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weighting in sensitivity.WEIGHTINGS:
        best = best_rows[weighting]
        t1 = next(
            row
            for row in global_rows
            if row["weighting"] == weighting and float(row["t"]) == 1.0
        )
        invalid = sum(
            int(row["invalid_count"])
            for row in global_rows
            if row["weighting"] == weighting
        )
        lines.append(
            f"| {weighting} | {float(best['t']):.5g} | "
            f"{float(best['before_rmse_mm']):.6f} | {float(best['exact_rmse_mm']):.6f} | "
            f"{float(t1['exact_rmse_mm']):.6f} | {float(best['exact_explained_fraction']):.6f} | "
            f"{float(t1['exact_explained_fraction']):.6f} | "
            f"{float(best['gap_to_baseline_rmse']):.6f} | {invalid} |"
        )
    lines += [
        "",
        "## Best scanned point: top / middle / bottom",
        "",
        "| weighting | region | exact RMSE | exact explained | invalid |",
        "|---|---|---:|---:|---:|",
    ]
    for weighting in sensitivity.WEIGHTINGS:
        best_t = float(best_rows[weighting]["t"])
        for region in ("top_0_299", "middle_300_2699", "bottom_2700_2999"):
            row = by_key[(weighting, best_t, region)]
            lines.append(
                f"| {weighting} | {region} | {float(row['exact_rmse_mm']):.6f} | "
                f"{float(row['exact_explained_fraction']):.6f} | {int(row['invalid_count'])} |"
            )
    lines += [
        "",
        "## Decision gate",
        "",
        f"**FULL_LINEAR_STEP = {full_step}**  ",
        f"**LOCAL_DESCENT_DIRECTION = {local_descent}**  ",
        f"**PATH_SCAN_DECISION = {decision}**",
        "",
    ]
    lines.extend(f"- {reason}。" for reason in reasons)
    lines += [
        "- 判据：三条方向均存在 exact energy 下降且全路径无 invalid，才确认 LOCAL_DESCENT；"
        "三条方向在 t=1 仍保持正 exact explained fraction，才确认 FULL_LINEAR_STEP。",
    ]
    if decision == "FULL_STEP_COMPATIBLE":
        lines += [
            "- 关键结果：完整线性步能由真实 production ray–cone reconstruction 兑现。",
            "- 本步骤仍不证明 full nonlinear optimum 或验证集性能；按研究顺序在此停下。",
        ]
    elif decision == "DAMPED_RELINEARIZATION_REQUIRED":
        lines += [
            "- 关键结果：三条方向在足够小的步长下都是下降方向，但完整线性步全部灾难性失效。"
            "这拒绝的是 one-shot `Theta0+DeltaTheta`，不是 Circular Cone，也不是小步非线性优化。",
            "- 下一阶段若继续，必须采用 scaled/damped trust region，每次接受小步后重新计算 Jacobian；"
            "不得把本次 best path point 或 t=1 参数当作正式候选写回。",
            "- 按研究顺序在此停下。",
        ]
    else:
        lines += [
            "- 关键结果：真实 Cone 路径未兑现线性改善，不应直接进入六维非线性重优化。",
            "- 按研究顺序在此停下。",
        ]
    lines += [
        "",
        "## 参数与不变项",
        "",
        "以下增量只定义扫描方向，没有写出候选 YAML：",
        "",
        "| parameter | point_equal delta | frame_equal delta | v_region_equal delta |",
        "|---|---:|---:|---:|",
    ]
    for index, spec in enumerate(sensitivity.PARAMETERS):
        lines.append(
            f"| {spec.name} ({spec.unit}) | {solutions['point_equal'].delta[index]:+.9g} | "
            f"{solutions['frame_equal'].delta[index]:+.9g} | "
            f"{solutions['v_region_equal'].delta[index]:+.9g} |"
        )
    lines += [
        "",
        f"- Formal Cone：`{cone_path}`",
        f"- Formal Cone SHA-256（运行前后相同）：`{cone_hash}`",
        f"- Frozen baseline 最大复核误差：{data.baseline_metric_max_error:.3e}",
        "- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、"
        "runtime reconstruction 与正式 Cone 均未修改。",
        "",
    ]
    return "\n".join(lines)


def render_output_files(decision: str) -> str:
    return "\n".join(
        [
            "# Circular Cone nonlinear path-scan outputs",
            "",
            f"**PATH_SCAN_DECISION = {decision}**",
            "",
            "| 文件 | 内容 | 主要用途 | 边界 |",
            "|---|---|---|---|",
            "| cone_nonlinear_path_scan.csv | 三方向各 t 的 exact/linear global 指标、invalid、bounds 和参数值 | 检查真实 loss 路径及线性化误差 | 不是 optimizer trace |",
            "| cone_nonlinear_path_regions.csv | top/middle/bottom、每300px bin、外推区的 exact path 指标 | 检查改善是否只来自中部 | 仅 FIT |",
            "| cone_nonlinear_path_scan.png | exact 与 linear RMSE 路径及 nonlinearity gap | 组会主图 | 不含 validation |",
            "| cone_nonlinear_path_scan_report.md | 隔离、关键数值、区域表现与决策门结论 | 本步骤主报告 | 不证明新参数可发布 |",
            "| OUTPUT_FILES.md | 文件索引 | 快速导航 | 不增加科学证据 |",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    pnp_audit = args.pnp_audit.resolve()
    measurement_config = args.measurement_config.resolve()
    sensitivity_dir = args.sensitivity_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    app_config = sensitivity.load_app_config(measurement_config)
    cone_path = Path(app_config.calibration.laser_model).resolve()
    cone_hash_before = sensitivity.sha256_file(cone_path)
    data, _ = sensitivity.prepare_data(
        data_root,
        pnp_audit,
        measurement_config,
        include_splits=("fit",),
    )
    expected_frames = {f"{index:03d}" for index in range(1, 11)}
    observed_frames = {frame.frame_id for frame in data.frames}
    if observed_frames != expected_frames or set(data.split) != {"fit"}:
        raise RuntimeError(
            f"FIT-only isolation failed: frames={sorted(observed_frames)}, splits={sorted(set(data.split))}"
        )

    jacobians = sensitivity.compute_fit_jacobian(data)
    solutions = sensitivity.solve_fit(data, jacobians)
    reference_theta0, reference_deltas = read_reference_solutions(
        sensitivity_dir / "cone_parameter_sensitivity.csv"
    )
    if not np.allclose(data.theta0, reference_theta0, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError("Theta0 differs from the frozen sensitivity result")
    for weighting in sensitivity.WEIGHTINGS:
        if not np.allclose(
            solutions[weighting].delta,
            reference_deltas[weighting],
            rtol=1.0e-9,
            atol=1.0e-9,
        ):
            raise RuntimeError(f"Recomputed FIT delta differs for {weighting}")

    global_rows, region_rows = scan_paths(data, jacobians, solutions)
    best_rows = select_best_rows(global_rows)
    full_step, local_descent, decision, reasons = path_verdict(global_rows, best_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_paths["cone_nonlinear_path_scan.csv"], global_rows)
    write_rows(output_paths["cone_nonlinear_path_regions.csv"], region_rows)
    save_plot(output_paths["cone_nonlinear_path_scan.png"], global_rows)
    output_paths["cone_nonlinear_path_scan_report.md"].write_text(
        render_report(
            data,
            solutions,
            global_rows,
            region_rows,
            best_rows,
            full_step,
            local_descent,
            decision,
            reasons,
            cone_path,
            cone_hash_before,
        ),
        encoding="utf-8",
    )
    output_paths["OUTPUT_FILES.md"].write_text(
        render_output_files(decision), encoding="utf-8"
    )

    cone_hash_after = sensitivity.sha256_file(cone_path)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Circular Cone changed during path scan")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output files: {sorted(actual_names)}")

    print(f"FULL_LINEAR_STEP={full_step}")
    print(f"LOCAL_DESCENT_DIRECTION={local_descent}")
    print(f"PATH_SCAN_DECISION={decision}")
    for weighting in sensitivity.WEIGHTINGS:
        row = best_rows[weighting]
        print(
            f"{weighting}: best_t={float(row['t']):.6g}, "
            f"exact_rmse={float(row['exact_rmse_mm']):.9g}, "
            f"explained={float(row['exact_explained_fraction']):.9g}, "
            f"invalid={int(row['invalid_count'])}"
        )
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
