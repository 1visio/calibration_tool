#!/usr/bin/env python3
"""Audit Frozen C0 residuals for points recovered by the full board mask.

The audit opens only old FIT 001--018, 025--036 and new FIT 049--054.  It
keeps the existing PnP/Steger/continuity path and frozen Circular Cone model;
it does not fit C0/C1 and never enumerates or opens Validation data.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_board_mask_exclusion_049_054 as mask_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


OLD_FIT_IDS = tuple(f"{index:03d}" for index in range(1, 19)) + tuple(f"{index:03d}" for index in range(25, 37))
NEW_FIT_IDS = tuple(f"{index:03d}" for index in range(49, 55))
REGIONS = (("Top", 0.0, 300.0), ("Middle", 300.0, 2700.0), ("Bottom", 2700.0, 3000.0))
ROLES = ("chess", "nolaser", "laser")
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE_MM = 20.0
CURRENT_MASK_MARGIN_PX = -2
LASER_ORIENTATION = "vertical"
UNCERTAINTY_MAX_MM = 0.033
TREND_BINS = 18

DEFAULT_OLD_DATA_ROOT = ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_NEW_DATA_ROOT = ROOT / "projects" / "daheng" / "data" / "laser_plane_0817"
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_OUTPUT_DIR = ROOT / "projects" / "daheng" / "outputs" / "0817" / "recovered_edge_c0_residual"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-data-root", type=Path, default=DEFAULT_OLD_DATA_ROOT)
    parser.add_argument("--new-data-root", type=Path, default=DEFAULT_NEW_DATA_ROOT)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=fixed.DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=fixed.DEFAULT_FORMAL_CONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def inventory_old_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for frame_id in OLD_FIT_IDS:
        root = data_root if int(frame_id) <= 18 else data_root / "fit_edge_extension"
        groups[frame_id] = {}
        for role in ROLES:
            path = root / "fit" / f"{role} {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            groups[frame_id][role] = {"path": path}
    return groups


def inventory_new_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    return mask_audit.inventory_new_fit(data_root)


def finite_range(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    return float(np.min(values)), float(np.max(values))


def region_for_v(v: float) -> str:
    for name, low, high in REGIONS:
        if low <= float(v) < high:
            return name
    return "Outside"


def clean(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def extract_points(
    *,
    laser: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    pose: Any,
    intrinsics: Any,
    frozen_calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
) -> dict[str, np.ndarray | int]:
    u, v, _, _ = fixed.triplets.extract_laser_centers(
        laser,
        background,
        mask,
        fixed.EXTRACTION_CONFIG,
        LASER_ORIENTATION,
    )
    uv = np.column_stack([u, v]).astype(np.float64)
    truth = fixed.coverage.plane_ray_truth(
        u,
        v,
        pose.normal,
        pose.d,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    lambda_cone, model_valid = fixed.lambda_by_input(
        uv,
        frozen_calibration,
        reconstruction_params,
    )
    effective = (
        np.asarray(model_valid, dtype=bool)
        & np.asarray(truth["valid"], dtype=bool)
        & np.isfinite(np.asarray(truth["points"], dtype=np.float64)[:, 2])
        & np.isfinite(np.asarray(lambda_cone, dtype=np.float64))
    )
    uv_effective = uv[effective]
    normalized = cv2.undistortPoints(
        uv_effective.reshape(-1, 1, 2),
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    ).reshape(-1, 2)
    s = base.pca_s_values(normalized, c1_model)
    lambda_truth = np.asarray(truth["points"], dtype=np.float64)[effective, 2]
    lambda_cone_effective = np.asarray(lambda_cone, dtype=np.float64)[effective]
    return {
        "steger_count": int(len(uv)),
        "effective_count": int(np.count_nonzero(effective)),
        "u": uv_effective[:, 0],
        "v": uv_effective[:, 1],
        "s": s,
        "lambda_truth": lambda_truth,
        "lambda_cone": lambda_cone_effective,
        "residual": lambda_truth - lambda_cone_effective,
    }


def append_point_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    frame_id: str,
    result: Mapping[str, Any],
    old_mask: np.ndarray,
) -> None:
    u = np.asarray(result["u"], dtype=np.float64)
    v = np.asarray(result["v"], dtype=np.float64)
    s = np.asarray(result["s"], dtype=np.float64)
    lambda_truth = np.asarray(result["lambda_truth"], dtype=np.float64)
    lambda_cone = np.asarray(result["lambda_cone"], dtype=np.float64)
    residual = np.asarray(result["residual"], dtype=np.float64)
    inside_old = mask_audit.points_in_mask(u, v, old_mask)
    for index in range(len(u)):
        rows.append(
            {
                "record_type": "point",
                "dataset": dataset,
                "frame_id": frame_id,
                "point_class": "old_mask_existing" if inside_old[index] else "new_mask_recovered",
                "outside_old_mask": not bool(inside_old[index]),
                "region": region_for_v(v[index]),
                "u_px": u[index],
                "v_px": v[index],
                "s": s[index],
                "lambda_truth_mm": lambda_truth[index],
                "lambda_cone_mm": lambda_cone[index],
                "residual_mm": residual[index],
            }
        )


def metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    residual = np.asarray([float(row["residual_mm"]) for row in rows], dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return {"n": 0, "bias": math.nan, "rmse": math.nan, "p95": math.nan, "max_abs": math.nan, "frames": 0}
    return {
        "n": int(len(residual)),
        "bias": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "p95": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
        "frames": int(len({str(row["frame_id"]) for row in rows})),
    }


def select_rows(rows: Sequence[Mapping[str, Any]], dataset: str | None = None, point_class: str | None = None, region: str | None = None) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        if dataset is not None and row["dataset"] != dataset:
            continue
        if point_class is not None and row["point_class"] != point_class:
            continue
        if region is not None and row["region"] != region:
            continue
        selected.append(row)
    return selected


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def trend_stats(rows: Sequence[Mapping[str, Any]], axis: str, bins: int = TREND_BINS) -> dict[str, Any]:
    x = np.asarray([float(row[axis]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["residual_mm"]) for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.ptp(x) == 0.0:
        return {"n": int(len(x)), "slope": math.nan, "rho": math.nan, "bin_range": math.nan, "bin_rmse": math.nan, "bin_jump_max": math.nan, "bin_count": 0}
    slope = float(np.polyfit(x, y, 1)[0])
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    medians: list[float] = []
    residual_from_bin: list[float] = []
    for index in range(bins):
        mask = (x >= edges[index]) & (x <= edges[index + 1] if index == bins - 1 else x < edges[index + 1])
        if not np.any(mask):
            continue
        median = float(np.median(y[mask]))
        medians.append(median)
        residual_from_bin.extend((y[mask] - median).tolist())
    jumps = np.abs(np.diff(np.asarray(medians, dtype=np.float64))) if len(medians) > 1 else np.empty(0)
    return {
        "n": int(len(x)),
        "slope": slope,
        "rho": rank_corr(x, y),
        "bin_range": float(np.ptp(medians)) if medians else math.nan,
        "bin_rmse": float(np.sqrt(np.mean(np.square(residual_from_bin)))) if residual_from_bin else math.nan,
        "bin_jump_max": float(np.max(jumps)) if len(jumps) else math.nan,
        "bin_count": int(len(medians)),
    }


def frame_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["frame_id"]), []).append(row)
    frame_metrics = [metric(group) for group in grouped.values() if group]
    biases = np.asarray([item["bias"] for item in frame_metrics if math.isfinite(float(item["bias"]))], dtype=np.float64)
    rmses = np.asarray([item["rmse"] for item in frame_metrics if math.isfinite(float(item["rmse"]))], dtype=np.float64)
    return {
        "frame_count": int(len(frame_metrics)),
        "bias_min": float(np.min(biases)) if len(biases) else math.nan,
        "bias_max": float(np.max(biases)) if len(biases) else math.nan,
        "bias_range": float(np.ptp(biases)) if len(biases) else math.nan,
        "bias_std": float(np.std(biases)) if len(biases) else math.nan,
        "rmse_min": float(np.min(rmses)) if len(rmses) else math.nan,
        "rmse_max": float(np.max(rmses)) if len(rmses) else math.nan,
        "rmse_range": float(np.ptp(rmses)) if len(rmses) else math.nan,
    }


def classify_status(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = metric(rows)
    s_trend = trend_stats(rows, "s")
    v_trend = trend_stats(rows, "v_px")
    frame = frame_consistency(rows)
    high_frequency = max(
        [value for value in (float(s_trend["bin_rmse"]), float(v_trend["bin_rmse"])) if math.isfinite(value)],
        default=math.inf,
    )
    max_jump = max(
        [value for value in (float(s_trend["bin_jump_max"]), float(v_trend["bin_jump_max"])) if math.isfinite(value)],
        default=math.inf,
    )
    bias_range = float(frame["bias_range"])
    p95 = float(summary["p95"])
    if p95 <= UNCERTAINTY_MAX_MM and high_frequency <= UNCERTAINTY_MAX_MM and bias_range <= UNCERTAINTY_MAX_MM:
        status = "STABLE"
    elif high_frequency <= 0.050 and max_jump <= 0.080 and bias_range <= 0.080:
        status = "CORRECTABLE_BY_C1"
    else:
        status = "NEED_C0_REFIT"
    return status, summary, s_trend, {"v": v_trend, "frame": frame, "high_frequency_rmse": high_frequency, "max_bin_jump": max_jump}


def write_points_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "record_type",
        "dataset",
        "frame_id",
        "point_class",
        "outside_old_mask",
        "region",
        "u_px",
        "v_px",
        "s",
        "lambda_truth_mm",
        "lambda_cone_mm",
        "residual_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field, "")) for field in fields})


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def report_table(rows: Sequence[Mapping[str, Any]], datasets: Sequence[str], classes: Sequence[str]) -> list[str]:
    lines = [
        "| dataset | point class | region | n | frames | bias (mm) | RMSE (mm) | P95 abs (mm) | max abs (mm) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        for point_class in classes:
            for region, _, _ in REGIONS:
                item = metric(select_rows(rows, dataset, point_class, region))
                lines.append(
                    f"| {dataset} | {point_class} | {region} | {item['n']} | {item['frames']} | {fmt(item['bias'])} | {fmt(item['rmse'])} | {fmt(item['p95'])} | {fmt(item['max_abs'])} |"
                )
    return lines


def report_text(
    *,
    output_dir: Path,
    old_root: Path,
    new_root: Path,
    c1_path: Path,
    c1_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    status: str,
    recovered_summary: Mapping[str, Any],
    recovered_s_trend: Mapping[str, Any],
    recovered_other: Mapping[str, Any],
) -> str:
    classes = ("old_mask_existing", "new_mask_recovered")
    lines = [
        "# Frozen Circular Cone C0 residual audit for recovered edge FIT points",
        "",
        f"`C0_EDGE_STATUS = {status}`",
        "",
        "## Scope and residual definition",
        "",
        f"- 只打开旧 FIT `001–018、025–036`（`{old_root}`）和新 FIT `049–054`（`{new_root / 'fit'}`）的 108 张三联图；没有枚举或读取 Validation。",
        "- 两组数据均使用新完整棋盘物理 mask：PnP 投影 `X=[-20,220] mm`、`Y=[-20,160] mm`，0 mm inset、无腐蚀。",
        "- 保持 PnP、Steger、`vertical` 每 row 单点、continuity 和 900 点上限不变；没有拟合或修改 C0/C1。",
        f"- Frozen C1 artifact：`{c1_path}`；SHA-256 = `{c1_sha256}`。C1 仅用于固定 PCA `s` 坐标。",
        "- `old_mask_existing`：new-mask effective point 落在旧 inner-corner hull + margin -2 mask 内；`new_mask_recovered`：该点落在旧 mask 外。",
        "- 残差定义为相机 ray 深度误差：`residual_mm = lambda_truth - lambda_cone`；`lambda_truth` 来自当前 frame 的 PnP plane-ray truth，`lambda_cone` 来自 Frozen Circular Cone C0 production reconstruction。",
        "",
        "## Frozen Top/Middle/Bottom regions",
        "",
        "| region | sensor v interval |",
        "|---|---:|",
        "| Top | [0, 300) px |",
        "| Middle | [300, 2700) px |",
        "| Bottom | [2700, 3000) px |",
        "",
        "## Bias / RMSE / P95 / max",
        "",
    ]
    lines.extend(report_table(rows, ("old_fit", "new_fit"), classes))
    lines.extend(
        [
            "",
            "## Recovered-edge cross-frame consistency and trends",
            "",
            "状态判定只使用 `new_mask_recovered` 且 region 为 Top/Bottom 的点；Middle recovery 保留在前面的区域表中作为诊断，不混入 edge decision。",
            "",
            "| dataset | recovered n | frames | frame bias range (mm) | frame RMSE range (mm) | s rho | s slope (mm/unit s) | s binned range | s within-bin RMSE | s max adjacent jump | v rho | v slope (mm/px) | v binned range | v within-bin RMSE | v max adjacent jump |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("old_fit", "new_fit", "combined"):
        recovered = [
            row
            for row in select_rows(rows, None if dataset == "combined" else dataset, "new_mask_recovered")
            if row["region"] in {"Top", "Bottom"}
        ]
        summary = metric(recovered)
        s_trend = trend_stats(recovered, "s")
        v_trend = trend_stats(recovered, "v_px")
        frame = frame_consistency(recovered)
        lines.append(
            f"| {dataset} | {summary['n']} | {frame['frame_count']} | {fmt(frame['bias_range'])} | {fmt(frame['rmse_range'])} | "
            f"{fmt(s_trend['rho'],3)} | {fmt(s_trend['slope'],4)} | {fmt(s_trend['bin_range'])} | {fmt(s_trend['bin_rmse'])} | {fmt(s_trend['bin_jump_max'])} | "
            f"{fmt(v_trend['rho'],3)} | {fmt(v_trend['slope'],6)} | {fmt(v_trend['bin_range'])} | {fmt(v_trend['bin_rmse'])} | {fmt(v_trend['bin_jump_max'])} |"
        )
    lines.extend(
        [
            "",
            "### Region-specific cross-frame consistency",
            "",
            "| dataset | region | recovered n | frames | frame bias range (mm) | frame RMSE range (mm) | s binned range | s within-bin RMSE | s max adjacent jump |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("old_fit", "new_fit", "combined"):
        for region in ("Top", "Bottom"):
            recovered_region = [
                row
                for row in select_rows(rows, None if dataset == "combined" else dataset, "new_mask_recovered", region)
            ]
            summary = metric(recovered_region)
            trend = trend_stats(recovered_region, "s")
            frame = frame_consistency(recovered_region)
            lines.append(
                f"| {dataset} | {region} | {summary['n']} | {frame['frame_count']} | {fmt(frame['bias_range'])} | {fmt(frame['rmse_range'])} | "
                f"{fmt(trend['bin_range'])} | {fmt(trend['bin_rmse'])} | {fmt(trend['bin_jump_max'])} |"
            )
    lines.extend(
        [
            "",
            "### Recovered-edge aggregate (Top/Bottom only)",
            "",
            f"- n={recovered_summary['n']}, bias={fmt(recovered_summary['bias'])} mm, RMSE={fmt(recovered_summary['rmse'])} mm, P95 abs={fmt(recovered_summary['p95'])} mm, max abs={fmt(recovered_summary['max_abs'])} mm。",
            f"- s trend: binned range={fmt(recovered_s_trend['bin_range'])} mm，within-bin RMSE={fmt(recovered_s_trend['bin_rmse'])} mm，最大相邻 bin 跳变={fmt(recovered_s_trend['bin_jump_max'])} mm。",
            f"- v trend: binned range={fmt(recovered_other['v']['bin_range'])} mm，within-bin RMSE={fmt(recovered_other['v']['bin_rmse'])} mm，最大相邻 bin 跳变={fmt(recovered_other['v']['bin_jump_max'])} mm。",
            f"- frame bias range={fmt(recovered_other['frame']['bias_range'])} mm，frame RMSE range={fmt(recovered_other['frame']['rmse_range'])} mm。",
            "",
            "## Decision rule",
            "",
            f"- PnP truth uncertainty reference: 0.025–0.033 mm；本审计使用上限 {UNCERTAINTY_MAX_MM:.3f} mm。",
            "- STABLE：Top/Bottom recovered 点的 P95、bin 内残差和跨 frame bias range 均不超过 0.033 mm。",
            "- CORRECTABLE_BY_C1：Top/Bottom recovered 残差超过 uncertainty，但 s/v 低频 trend 后的 bin 内 RMSE ≤0.050 mm、跨 frame bias range ≤0.080 mm、无明显大跳变（≤0.080 mm），说明主要是平滑空间项。",
            "- NEED_C0_REFIT：跨 frame 不一致、bin 内噪声或跳变超过上述范围。以上仅为诊断判据，没有训练 C1。",
            "",
            f"- 结论：`C0_EDGE_STATUS = {status}`。",
        ]
    )
    if status == "STABLE":
        lines.append("- 下一步建议：C0 对新增边缘点已足够稳定，不需要因 mask 扩展重拟合 C0；C1 也不是必需项。")
    elif status == "CORRECTABLE_BY_C1":
        lines.append("- 下一步建议：保持 Frozen C0，只进行低自由度 C1 residual correction；不建议重新拟合 C0。")
    else:
        lines.append("- 下一步建议：新增边缘点表现出非平滑或跨 frame 不一致，先重新评估/拟合 C0，再考虑 C1。")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `recovered_edge_c0_residual.csv`: `{output_dir / 'recovered_edge_c0_residual.csv'}`",
            f"- `residual_vs_s.png`: `{output_dir / 'residual_vs_s.png'}`",
            f"- `residual_vs_v.png`: `{output_dir / 'residual_vs_v.png'}`",
            f"- `report.md`: `{output_dir / 'report.md'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def make_plot(path: Path, rows: Sequence[Mapping[str, Any]], x_field: str, xlabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    colors = {"old_mask_existing": "#777777", "new_mask_recovered": "#d95f02"}
    for axis, dataset in zip(axes, ("old_fit", "new_fit")):
        for point_class in ("old_mask_existing", "new_mask_recovered"):
            selected = select_rows(rows, dataset, point_class)
            if not selected:
                continue
            x = np.asarray([float(row[x_field]) for row in selected], dtype=np.float64)
            y = np.asarray([float(row["residual_mm"]) for row in selected], dtype=np.float64)
            axis.scatter(x, y, s=2, alpha=0.16 if point_class == "old_mask_existing" else 0.32, color=colors[point_class], label=point_class)
        recovered = select_rows(rows, dataset, "new_mask_recovered")
        if recovered:
            x = np.asarray([float(row[x_field]) for row in recovered], dtype=np.float64)
            y = np.asarray([float(row["residual_mm"]) for row in recovered], dtype=np.float64)
            trend = trend_stats(recovered, x_field)
            # Draw a diagnostic linear trend only; it is not a fitted correction model.
            if len(x) >= 3 and math.isfinite(float(trend["slope"])):
                intercept = float(np.mean(y) - float(trend["slope"]) * np.mean(x))
                order = np.argsort(x)
                axis.plot(x[order], intercept + float(trend["slope"]) * x[order], color="#000000", linewidth=1.2, label="recovered linear diagnostic")
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.axhline(UNCERTAINTY_MAX_MM, color="#999999", linestyle="--", linewidth=0.8)
        axis.axhline(-UNCERTAINTY_MAX_MM, color="#999999", linestyle="--", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Frozen C0 residual: lambda_truth - lambda_cone / mm")
    fig.suptitle(f"Frozen C0 residual vs {xlabel}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    old_root = args.old_data_root.resolve()
    new_root = args.new_data_root.resolve()
    output_dir = args.output_dir.resolve()
    c1_path = args.c1_model.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    old_groups = inventory_old_fit(old_root)
    new_groups = inventory_new_fit(new_root)
    c1_model, c1_sha256 = frozen.load_frozen_json(c1_path)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, _ = board.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    frozen_calibration = dict(calibration)
    frozen_calibration["laser_model"] = dict(frozen_model)

    point_rows: list[dict[str, Any]] = []
    for dataset, frame_ids, groups in (("old_fit", OLD_FIT_IDS, old_groups), ("new_fit", NEW_FIT_IDS, new_groups)):
        for frame_id in frame_ids:
            chess = fixed.triplets.imread_unicode(Path(groups[frame_id]["chess"]["path"]))
            background = fixed.triplets.imread_unicode(Path(groups[frame_id]["nolaser"]["path"]))
            laser = fixed.triplets.imread_unicode(Path(groups[frame_id]["laser"]["path"]))
            pose = fixed.triplets.detect_board_pose(
                chess,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
                cols=BOARD_COLS,
                rows=BOARD_ROWS,
                square_size_mm=SQUARE_SIZE_MM,
                max_rmse_px=0.40,
            )
            shape = fixed.triplets.to_gray_float(chess).shape
            old_mask = fixed.triplets.board_inner_mask(shape, pose.corners, margin_px=CURRENT_MASK_MARGIN_PX)
            new_mask = mask_audit.polygon_mask(shape, mask_audit.projected_outer_boundary(pose, intrinsics))
            result = extract_points(
                laser=laser,
                background=background,
                mask=new_mask,
                pose=pose,
                intrinsics=intrinsics,
                frozen_calibration=frozen_calibration,
                reconstruction_params=reconstruction_params,
                c1_model=c1_model,
            )
            append_point_rows(point_rows, dataset=dataset, frame_id=frame_id, result=result, old_mask=old_mask)

    recovered_all = select_rows(point_rows, point_class="new_mask_recovered")
    recovered = [row for row in recovered_all if row["region"] in {"Top", "Bottom"}]
    status, recovered_summary, recovered_s_trend, recovered_other = classify_status(recovered)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_points_csv(output_dir / "recovered_edge_c0_residual.csv", point_rows)
    make_plot(output_dir / "residual_vs_s.png", point_rows, "s", "Frozen PCA s")
    make_plot(output_dir / "residual_vs_v.png", point_rows, "v_px", "sensor v / px")
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            old_root=old_root,
            new_root=new_root,
            c1_path=c1_path,
            c1_sha256=c1_sha256,
            rows=point_rows,
            status=status,
            recovered_summary=recovered_summary,
            recovered_s_trend=recovered_s_trend,
            recovered_other=recovered_other,
        ),
        encoding="utf-8",
    )
    print(
        {
            "output_dir": str(output_dir),
            "C0_EDGE_STATUS": status,
            "point_count": len(point_rows),
            "recovered_count": len(recovered_all),
            "recovered_edge_count": len(recovered),
            "recovered_summary": recovered_summary,
            "validation_read": False,
        }
    )


if __name__ == "__main__":
    run(parse_args())
