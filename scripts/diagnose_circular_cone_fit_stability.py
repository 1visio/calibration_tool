#!/usr/bin/env python3
"""FIT-only frame jackknife and weak-direction profile for Circular Cone.

This script deliberately never loads validation frames.  It reuses the public
production reconstruction through the sensitivity/trust-region helpers and
writes only diagnostic artifacts in a new 0813 output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyze_circular_cone_parameter_sensitivity as sensitivity  # noqa: E402
import reoptimize_circular_cone_fit_trust_region as trust  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_FIT_OUTPUT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_nonlinear_fit_trust_region"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_fit_stability"
)
OUTPUT_NAMES = (
    "cone_frame_jackknife.csv",
    "cone_frame_jackknife_regions.csv",
    "cone_frame_jackknife.png",
    "cone_weak_direction_profile.csv",
    "cone_weak_direction_profile.png",
    "cone_fit_stability_report.md",
    "OUTPUT_FILES.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FIT-only Circular Cone frame jackknife and weak-direction profile."
    )
    parser.add_argument("--data-root", type=Path, default=sensitivity.paired.DEFAULT_DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=sensitivity.paired.DEFAULT_PNP_AUDIT)
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=sensitivity.paired.DEFAULT_MEASUREMENT_CONFIG,
    )
    parser.add_argument("--fit-output", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-accepted-steps", type=int, default=20)
    parser.add_argument(
        "--weightings",
        default=",".join(sensitivity.WEIGHTINGS),
        help="comma-separated subset of point_equal,frame_equal,v_region_equal",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_candidates(path: Path, theta0: np.ndarray) -> dict[str, np.ndarray]:
    rows: dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["parameter"]] = row
    candidates: dict[str, np.ndarray] = {}
    for weighting in sensitivity.WEIGHTINGS:
        candidates[weighting] = np.asarray(
            [float(rows[spec.name][f"final_{weighting}"]) for spec in sensitivity.PARAMETERS],
            dtype=np.float64,
        )
    return candidates


def subset_data(
    data: sensitivity.PreparedData,
    frame_ids: set[str],
) -> sensitivity.PreparedData:
    frames = [frame for frame in data.frames if frame.frame_id in frame_ids]
    if not frames:
        raise RuntimeError("Cannot create an empty FIT subset")
    pixels = np.concatenate([frame.pixels_uv for frame in frames], axis=0)
    splits = np.full(len(pixels), "fit", dtype="U10")
    ids = np.concatenate(
        [np.full(len(frame.pixels_uv), frame.frame_id, dtype="U3") for frame in frames]
    )
    planes = np.concatenate(
        [np.repeat(frame.plane[None, :], len(frame.pixels_uv), axis=0) for frame in frames],
        axis=0,
    )
    residual = np.concatenate([frame.residual_mm for frame in frames])
    return sensitivity.PreparedData(
        frames=frames,
        pixels_uv=np.ascontiguousarray(pixels),
        split=splits,
        frame_id=ids,
        planes=np.ascontiguousarray(planes),
        residual_mm=np.ascontiguousarray(residual),
        calibration=data.calibration,
        reconstruction_params=data.reconstruction_params,
        theta0=data.theta0.copy(),
        used_pixel_sha256=data.used_pixel_sha256,
        baseline_metric_max_error=data.baseline_metric_max_error,
    )


def metric_row(
    residual: np.ndarray,
    baseline: np.ndarray,
    v_px: np.ndarray,
    frame_ids: np.ndarray,
    weighting: str,
) -> dict[str, float]:
    weights = sensitivity.weights_for(v_px, frame_ids, weighting)
    before = sensitivity.calculate_metrics(baseline, weights)
    after = sensitivity.calculate_metrics(residual, weights)
    return {
        "before_bias_mm": before.bias_mm,
        "before_mae_mm": before.mae_mm,
        "before_rmse_mm": before.rmse_mm,
        "before_p95_abs_mm": before.p95_abs_mm,
        "before_residual_energy": before.residual_energy,
        "after_bias_mm": after.bias_mm,
        "after_mae_mm": after.mae_mm,
        "after_rmse_mm": after.rmse_mm,
        "after_p95_abs_mm": after.p95_abs_mm,
        "after_residual_energy": after.residual_energy,
        "explained_fraction": (
            before.residual_energy - after.residual_energy
        )
        / before.residual_energy,
    }


def frame_region_rows(
    result: trust.OptimizationResult,
    data: sensitivity.PreparedData,
    weighting: str,
    heldout_frame: str,
) -> list[dict[str, Any]]:
    heldout = subset_data(data, {heldout_frame})
    baseline = heldout.residual_mm
    candidate = sensitivity.evaluate_candidate(result.theta, heldout, "fit")
    if np.any(candidate.invalid_mask):
        raise RuntimeError(f"Held-out frame became invalid: {weighting}/{heldout_frame}")
    _, v_px, frame_ids = sensitivity.split_arrays(heldout, "fit")
    weights = sensitivity.weights_for(v_px, frame_ids, weighting)
    rows: list[dict[str, Any]] = []
    for region, v_min, v_max in sensitivity.region_definitions()[1:]:
        assert v_min is not None and v_max is not None
        mask = (v_px >= v_min) & (v_px < v_max)
        if not np.any(mask):
            continue
        before = sensitivity.calculate_metrics(baseline[mask], weights[mask])
        after = sensitivity.calculate_metrics(candidate.residual_mm[mask], weights[mask])
        rows.append(
            {
                "heldout_frame": heldout_frame,
                "weighting": weighting,
                "region": region,
                "v_min_px": v_min,
                "v_max_px": v_max,
                "sample_count": int(np.count_nonzero(mask)),
                "before_rmse_mm": before.rmse_mm,
                "after_rmse_mm": after.rmse_mm,
                "before_p95_abs_mm": before.p95_abs_mm,
                "after_p95_abs_mm": after.p95_abs_mm,
                "explained_fraction": (
                    before.residual_energy - after.residual_energy
                )
                / before.residual_energy,
            }
        )
    return rows


def run_jackknife(
    data: sensitivity.PreparedData,
    weightings: Sequence[str],
    max_accepted_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], trust.OptimizationResult]]:
    original_max_steps = trust.MAX_ACCEPTED_STEPS
    original_max_trials = trust.MAX_TRIALS
    trust.MAX_ACCEPTED_STEPS = max_accepted_steps
    trust.MAX_TRIALS = max(max_accepted_steps * 2, 40)
    frame_ids = sorted({frame.frame_id for frame in data.frames})
    all_ids = set(frame_ids)
    rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str], trust.OptimizationResult] = {}
    try:
        for heldout_frame in frame_ids:
            train_data = subset_data(data, all_ids - {heldout_frame})
            initial_eval = trust.evaluate_jacobian(
                train_data.theta0, train_data, base_residual=train_data.residual_mm
            )
            if any(np.any(mask) for mask in initial_eval.invalid_masks):
                raise RuntimeError(f"Initial jackknife Jacobian invalid for {heldout_frame}")
            for weighting in weightings:
                result = trust.optimize_weighting(
                    weighting, train_data, initial_eval.jacobian
                )
                results[(heldout_frame, weighting)] = result
                train_baseline, train_v, train_frames = sensitivity.split_arrays(
                    train_data, "fit"
                )
                train_metrics = metric_row(
                    result.residual_mm,
                    train_baseline,
                    train_v,
                    train_frames,
                    weighting,
                )
                heldout_data = subset_data(data, {heldout_frame})
                heldout_baseline = heldout_data.residual_mm
                heldout_candidate = sensitivity.evaluate_candidate(
                    result.theta, heldout_data, "fit"
                )
                if np.any(heldout_candidate.invalid_mask):
                    raise RuntimeError(
                        f"Held-out invalid reconstruction {heldout_frame}/{weighting}"
                    )
                _, heldout_v, heldout_frames = sensitivity.split_arrays(
                    heldout_data, "fit"
                )
                heldout_metrics = metric_row(
                    heldout_candidate.residual_mm,
                    heldout_baseline,
                    heldout_v,
                    heldout_frames,
                    weighting,
                )
                normalized_delta = (
                    result.theta - data.theta0
                ) / sensitivity.PARAMETER_SCALES
                row: dict[str, Any] = {
                    "heldout_frame": heldout_frame,
                    "weighting": weighting,
                    "train_frame_count": len(train_data.frames),
                    "train_point_count": len(train_data.residual_mm),
                    "status": result.status,
                    "accepted_steps": result.accepted_steps,
                    "trial_count": result.trial_count,
                    "final_radius": result.final_radius,
                    "train_invalid_count": 0,
                    "heldout_invalid_count": 0,
                    "train_theta_scaled_l2": float(np.linalg.norm(normalized_delta)),
                    "train_theta_scaled_max_abs": float(np.max(np.abs(normalized_delta))),
                }
                row.update({f"train_{key}": value for key, value in train_metrics.items()})
                row.update({f"heldout_{key}": value for key, value in heldout_metrics.items()})
                for index, spec in enumerate(sensitivity.PARAMETERS):
                    row[f"final_{spec.name}"] = result.theta[index]
                    row[f"delta_scale_{spec.name}"] = normalized_delta[index]
                rows.append(row)
                region_rows.extend(
                    frame_region_rows(result, data, weighting, heldout_frame)
                )
                print(
                    f"JACKKNIFE heldout={heldout_frame} weighting={weighting} "
                    f"status={result.status} train={train_metrics['after_rmse_mm']:.6f} "
                    f"heldout={heldout_metrics['after_rmse_mm']:.6f} "
                    f"explained={heldout_metrics['explained_fraction']:.6f}",
                    flush=True,
                )
    finally:
        trust.MAX_ACCEPTED_STEPS = original_max_steps
        trust.MAX_TRIALS = original_max_trials
    return rows, region_rows, results


def profile_final_candidates(
    data: sensitivity.PreparedData,
    fit_output: Path,
    weightings: Sequence[str],
) -> list[dict[str, Any]]:
    candidates = read_candidates(
        fit_output / "cone_nonlinear_fit_candidates.csv", data.theta0
    )
    rows: list[dict[str, Any]] = []
    for weighting in weightings:
        theta = candidates[weighting]
        base_eval = sensitivity.evaluate_candidate(theta, data, "fit")
        if np.any(base_eval.invalid_mask):
            raise RuntimeError(f"Final candidate invalid before profile: {weighting}")
        jacobian_eval = trust.evaluate_jacobian(
            theta, data, base_residual=base_eval.residual_mm
        )
        if any(np.any(mask) for mask in jacobian_eval.invalid_masks):
            raise RuntimeError(f"Final candidate Jacobian invalid: {weighting}")
        _, v_px, frame_ids = sensitivity.split_arrays(data, "fit")
        weights = sensitivity.weights_for(v_px, frame_ids, weighting)
        matrix, _ = trust.weighted_system(
            jacobian_eval.jacobian, base_eval.residual_mm, weights
        )
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        weak_vector = vt[-1]
        scale = sensitivity.PARAMETER_SCALES
        feasible_min = -np.inf
        feasible_max = np.inf
        for index in range(len(sensitivity.PARAMETERS)):
            direction = scale[index] * weak_vector[index]
            if direction > 0.0:
                feasible_max = min(
                    feasible_max,
                    (trust.UPPER_BOUNDS[index] - theta[index]) / direction,
                )
                feasible_min = max(
                    feasible_min,
                    (trust.LOWER_BOUNDS[index] - theta[index]) / direction,
                )
            elif direction < 0.0:
                feasible_max = min(
                    feasible_max,
                    (trust.LOWER_BOUNDS[index] - theta[index]) / direction,
                )
                feasible_min = max(
                    feasible_min,
                    (trust.UPPER_BOUNDS[index] - theta[index]) / direction,
                )
        profile_min = max(float(feasible_min), -8.0)
        profile_max = min(float(feasible_max), 8.0)
        if profile_min > 0.0 or profile_max < 0.0:
            raise RuntimeError(f"Weak profile lost t=0 feasibility: {weighting}")
        t_values = np.unique(
            np.concatenate(
                [
                    np.linspace(profile_min, profile_max, 41),
                    np.asarray([0.0]),
                ]
            )
        )
        for t_value in t_values:
            profile_theta = theta + float(t_value) * scale * weak_vector
            exact = sensitivity.evaluate_candidate(profile_theta, data, "fit")
            invalid = int(np.count_nonzero(exact.invalid_mask))
            if invalid:
                continue
            metrics = sensitivity.calculate_metrics(exact.residual_mm, weights)
            baseline_metrics = sensitivity.calculate_metrics(data.residual_mm, weights)
            rows.append(
                {
                    "weighting": weighting,
                    "profile_type": "weak_singular_direction",
                    "t": float(t_value),
                    "weak_singular_value": singular[-1],
                    "weak_to_strong_ratio": singular[-1] / singular[0],
                    "invalid_count": invalid,
                    "theta_axis_rad": profile_theta[0],
                    "phi_axis_rad": profile_theta[1],
                    "A_x_mm": profile_theta[2],
                    "A_y_mm": profile_theta[3],
                    "A_z_mm": profile_theta[4],
                    "alpha_rad": profile_theta[5],
                    "alpha_deg": math.degrees(float(profile_theta[5])),
                    "rmse_mm": metrics.rmse_mm,
                    "p95_abs_mm": metrics.p95_abs_mm,
                    "residual_energy": metrics.residual_energy,
                    "explained_fraction_vs_theta0": (
                        baseline_metrics.residual_energy - metrics.residual_energy
                    )
                    / baseline_metrics.residual_energy,
                }
            )
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: sensitivity.csv_value(row[field]) for field in fields})


def save_jackknife_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {"point_equal": "#2b6cb0", "frame_equal": "#2f855a", "v_region_equal": "#c05621"}
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 8.2), sharex=True)
    frame_order = sorted({row["heldout_frame"] for row in rows})
    x = np.arange(len(frame_order))
    for weighting in sensitivity.WEIGHTINGS:
        selected = [
            next(row for row in rows if row["heldout_frame"] == frame and row["weighting"] == weighting)
            for frame in frame_order
        ]
        axes[0].plot(
            x,
            [float(row["heldout_after_rmse_mm"]) for row in selected],
            marker="o",
            color=colors[weighting],
            linewidth=1.7,
            label=weighting,
        )
        axes[1].plot(
            x,
            [float(row["heldout_explained_fraction"]) for row in selected],
            marker="o",
            color=colors[weighting],
            linewidth=1.7,
            label=weighting,
        )
    axes[0].set_ylabel("held-out FIT RMSE / mm")
    axes[0].set_title("10-fold frame jackknife: held-out frame prediction")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_ylabel("held-out energy explained")
    axes[1].set_xlabel("held-out frame")
    axes[1].set_xticks(x, frame_order)
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_profile_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    colors = {"point_equal": "#2b6cb0", "frame_equal": "#2f855a", "v_region_equal": "#c05621"}
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.8), sharex=True)
    for weighting in sorted({row["weighting"] for row in rows}):
        selected = sorted(
            [row for row in rows if row["weighting"] == weighting],
            key=lambda row: float(row["t"]),
        )
        t_values = [float(row["t"]) for row in selected]
        axes[0].plot(
            t_values,
            [float(row["rmse_mm"]) for row in selected],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[weighting],
            label=weighting,
        )
        axes[1].plot(
            t_values,
            [float(row["explained_fraction_vs_theta0"]) for row in selected],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors[weighting],
            label=weighting,
        )
    axes[0].set_ylabel("FIT RMSE / mm")
    axes[0].set_title("Exact profile along final weakest singular direction")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_xlabel("profile coordinate t (interpretation-scaled weak direction)")
    axes[1].set_ylabel("explained vs Theta0")
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    data: sensitivity.PreparedData,
    jackknife_rows: Sequence[Mapping[str, Any]],
    jackknife_region_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    weightings: Sequence[str],
    max_accepted_steps: int,
    cone_path: Path,
    cone_hash: str,
) -> str:
    lines = [
        "# Circular Cone FIT stability: frame jackknife and weak-direction profile",
        "",
        "**FIT_ONLY = TRUE**",
        "",
        "本轮只使用 FIT 001–010；VALIDATION 011–013 未打开、未参与任何参数、步长、阈值或判定。"
        "正式 Cone 参数没有写回。",
        "",
        "## Frame jackknife",
        "",
        f"- 10 折按帧留一；每折用 9 帧重新运行 scaled/damped trust-region，最多 {max_accepted_steps} 个 accepted steps。",
        f"- weighting：{', '.join(weightings)}。",
        "- held-out frame 只用于评估该折 candidate，没有参与该折优化。",
        "",
        "| weighting | held-out RMSE median | held-out RMSE max | explained median | explained min | positive folds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weighting in weightings:
        rows = [row for row in jackknife_rows if row["weighting"] == weighting]
        explained = np.asarray([float(row["heldout_explained_fraction"]) for row in rows])
        rmse = np.asarray([float(row["heldout_after_rmse_mm"]) for row in rows])
        lines.append(
            f"| {weighting} | {np.median(rmse):.6f} | {np.max(rmse):.6f} | "
            f"{np.median(explained):.6f} | {np.min(explained):.6f} | "
            f"{int(np.count_nonzero(explained > 0.0))}/{len(rows)} |"
        )
    lines += [
        "",
        "| heldout frame | weighting | train RMSE | heldout RMSE | heldout explained | status | scaled delta L2 |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in jackknife_rows:
        lines.append(
            f"| {row['heldout_frame']} | {row['weighting']} | {float(row['train_after_rmse_mm']):.6f} | "
            f"{float(row['heldout_after_rmse_mm']):.6f} | {float(row['heldout_explained_fraction']):.6f} | "
            f"{row['status']} | {float(row['train_theta_scaled_l2']):.5g} |"
        )

    lines += [
        "",
        "## Held-out top / middle / bottom",
        "",
        "| weighting | region | held-out RMSE median | held-out explained median | positive folds |",
        "|---|---|---:|---:|---:|",
    ]
    for weighting in weightings:
        for region in ("top_0_299", "middle_300_2699", "bottom_2700_2999"):
            rows = [
                row
                for row in jackknife_region_rows
                if row["weighting"] == weighting and row["region"] == region
            ]
            explained = np.asarray([float(row["explained_fraction"]) for row in rows])
            rmse = np.asarray([float(row["after_rmse_mm"]) for row in rows])
            lines.append(
                f"| {weighting} | {region} | {np.median(rmse):.6f} | "
                f"{np.median(explained):.6f} | {int(np.count_nonzero(explained > 0.0))}/{len(rows)} |"
            )

    lines += [
        "",
        "## Weak-direction profile",
        "",
        "对 full-FIT nonlinear candidate 的最终 Jacobian 做 scaled SVD；沿最小右奇异向量"
        "进行 exact production reconstruction 扫描。由于 candidate 已在 `A_z=500 mm` 上界，"
        "profile 是受 bounds 限制的一侧/非对称 profile。它不是新的优化结果。",
        "",
        "| weighting | weak/strong singular ratio | profile t range | best RMSE | RMSE at t=0 | best explained vs Theta0 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for weighting in weightings:
        rows = [row for row in profile_rows if row["weighting"] == weighting]
        best = min(rows, key=lambda row: float(row["rmse_mm"]))
        t_values = [float(row["t"]) for row in rows]
        zero = min(rows, key=lambda row: abs(float(row["t"])))
        lines.append(
            f"| {weighting} | {float(best['weak_to_strong_ratio']):.3e} | "
            f"[{min(t_values):.4g},{max(t_values):.4g}] | {float(best['rmse_mm']):.6f} | "
            f"{float(zero['rmse_mm']):.6f} | {float(best['explained_fraction_vs_theta0']):.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- 如果 jackknife 的 held-out residual 仍稳定改善，说明 FIT 上得到的是跨姿态共同表面，而不是单帧过拟合。",
        "- 如果不同 weighting 的参数变化很大但 profile/重建表面变化很小，说明物理参数不可单独辨识，"
        "应把结论写成‘表面可拟合、参数有耦合’，不能发布某个 apex/alpha 数值。",
        "- 本报告不评价 VALIDATION；只有在 FIT 稳定性核验完成后，才允许做一次冻结的 011–013 最终评价。",
        "",
        "## Provenance / 不变项",
        "",
        f"- Formal Cone：`{cone_path}`",
        f"- Formal Cone SHA-256（运行前后相同）：`{cone_hash}`",
        f"- FIT frozen-pixel SHA-256：`{data.used_pixel_sha256}`",
        "- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 与正式 Cone 均未修改。",
        "",
    ]
    return "\n".join(lines)


def render_output_files() -> str:
    return "\n".join(
        [
            "# Circular Cone FIT stability outputs",
            "",
            "本目录只包含 FIT-only frame jackknife 与 weak-direction profile；没有 validation 结果，也没有候选 YAML。",
            "",
            "| 文件 | 作用 | 边界 |",
            "|---|---|---|",
            "| cone_frame_jackknife.csv | 10 折训练/留一帧 exact 指标与参数移动 | 不是 validation |",
            "| cone_frame_jackknife_regions.csv | 留一帧 top/middle/bottom 与每300px区域 | 仅 FIT |",
            "| cone_frame_jackknife.png | 留一帧 RMSE 与 explained 曲线 | 组会图，不含 validation |",
            "| cone_weak_direction_profile.csv | 最弱 SVD 方向的 exact profile | 不是新优化结果 |",
            "| cone_weak_direction_profile.png | weak-direction loss/profile 图 | bounds 限制的一维扫描 |",
            "| cone_fit_stability_report.md | 本阶段主报告 | 不授权写回参数 |",
            "| OUTPUT_FILES.md | 输出索引 | 不增加科学证据 |",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    pnp_audit = args.pnp_audit.resolve()
    measurement_config = args.measurement_config.resolve()
    fit_output = args.fit_output.resolve()
    output_dir = args.output_dir.resolve()
    weightings = tuple(name.strip() for name in args.weightings.split(",") if name.strip())
    if not weightings or not set(weightings) <= set(sensitivity.WEIGHTINGS):
        raise ValueError(f"Unsupported weighting selection: {weightings}")
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite: " + ", ".join(str(path) for path in existing)
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
    if set(data.split) != {"fit"} or {frame.frame_id for frame in data.frames} != {
        f"{index:03d}" for index in range(1, 11)
    }:
        raise RuntimeError("Stability analysis did not load exactly FIT 001-010")

    jackknife_rows, jackknife_region_rows, _ = run_jackknife(
        data, weightings, args.max_accepted_steps
    )
    profile_rows = profile_final_candidates(data, fit_output, weightings)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_paths["cone_frame_jackknife.csv"], jackknife_rows)
    write_rows(output_paths["cone_frame_jackknife_regions.csv"], jackknife_region_rows)
    save_jackknife_plot(output_paths["cone_frame_jackknife.png"], jackknife_rows)
    write_rows(output_paths["cone_weak_direction_profile.csv"], profile_rows)
    save_profile_plot(output_paths["cone_weak_direction_profile.png"], profile_rows)
    output_paths["cone_fit_stability_report.md"].write_text(
        render_report(
            data,
            jackknife_rows,
            jackknife_region_rows,
            profile_rows,
            weightings,
            args.max_accepted_steps,
            cone_path,
            cone_hash_before,
        ),
        encoding="utf-8",
    )
    output_paths["OUTPUT_FILES.md"].write_text(render_output_files(), encoding="utf-8")
    cone_hash_after = sensitivity.sha256_file(cone_path)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Circular Cone changed during FIT stability analysis")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output files: {sorted(actual_names)}")
    print(f"JACKKNIFE_ROWS={len(jackknife_rows)}")
    print(f"PROFILE_ROWS={len(profile_rows)}")
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
