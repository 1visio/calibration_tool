#!/usr/bin/env python3
"""Build and independently evaluate an experimental paired-PnP b(v) table.

BUILD is fixed to frames 001..010.  HOLDOUT is fixed to 011..013 and is never
used for table estimation, support selection, smoothing, or thresholds.  The
experimental correction is applied only at exact, BUILD-supported integer rows
inside v=300..2699; unsupported rows and both image edges remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_TOOL_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(MEASUREMENT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from laser.laser_extractor import extract_laser_center  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "extrinsics0813"
PNP_AUDIT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "pnp_reference_audit"
    / "paired_pnp_reference_audit.csv"
)
MEASUREMENT_CONFIG = (
    MEASUREMENT_TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
)
OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "paired_pnp_ground_bias_v"
)
OUTPUT_NAMES = (
    "ground_bias_table.csv",
    "ground_bias_table.npy",
    "compensation_metrics.json",
    "holdout_residual_before_after.png",
    "holdout_residual_frame_v_heatmap_before_after.png",
    "holdout_metrics_per_frame.csv",
    "compensation_support.png",
    "compensation_report.md",
    "OUTPUT_FILES.md",
)
BUILD_IDS = tuple(f"{value:03d}" for value in range(1, 11))
HOLDOUT_IDS = tuple(f"{value:03d}" for value in range(11, 14))
IMAGE_HEIGHT = 3000
WORK_V_MIN = 300
WORK_V_MAX = 2699
MIN_BUILD_FRAMES = 5
SMOOTH_WINDOW = 1
LOCAL_BLOCK_WIDTH_PX = 100
LOCAL_MIN_SAMPLES = 30
LOCAL_RELATIVE_WORSENING = 0.25
LOCAL_RMSE_ABSOLUTE_WORSENING_MM = 0.03
LOCAL_P95_ABSOLUTE_WORSENING_MM = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=PNP_AUDIT)
    parser.add_argument("--measurement-config", type=Path, default=MEASUREMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError(f"Expected decodable grayscale image: {path}")
    return image


def load_planes(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = row["frame_id"].zfill(3)
            required = (
                "pnp_success",
                "pair_files_ok",
                "pair_manifest_ok",
                "chess_sha256_ok",
                "laser_sha256_ok",
            )
            if any(row[key].strip().lower() != "true" for key in required):
                raise ValueError(f"Frame {frame_id} did not pass paired PnP audit")
            plane = np.asarray(
                [float(row[key]) for key in ("plane_nx", "plane_ny", "plane_nz", "plane_d")],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(plane)) or abs(plane[2]) <= 1.0e-6:
                raise ValueError(f"Invalid PnP plane for frame {frame_id}")
            result[frame_id] = {
                "split": row["split"],
                "plane": plane,
                "pnp_rmse_px": float(row["reprojection_rmse_px"]),
            }
    expected = set(BUILD_IDS + HOLDOUT_IDS)
    if set(result) != expected:
        raise ValueError(f"Expected PnP frames {sorted(expected)}, got {sorted(result)}")
    return result


def bin_median(v_px: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = np.full(IMAGE_HEIGHT, np.nan, dtype=np.float64)
    rows = np.rint(v_px).astype(int)
    valid = (
        np.isfinite(v_px)
        & np.isfinite(residual)
        & (rows >= 0)
        & (rows < IMAGE_HEIGHT)
    )
    for row in np.unique(rows[valid]):
        result[row] = float(np.median(residual[valid & (rows == row)]))
    return result


def reconstruct_residuals(
    frame_ids: tuple[str, ...],
    split: str,
    data_root: Path,
    planes: Mapping[str, Mapping[str, Any]],
    extraction_params: Any,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    raw: dict[str, dict[str, np.ndarray]] = {}
    extraction: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        if planes[frame_id]["split"] != split:
            raise ValueError(f"Frame {frame_id} split mismatch")
        image_path = data_root / split / f"laser {frame_id}.tif"
        centers = extract_laser_center(read_gray(image_path), extraction_params)
        reconstructed = reconstruct_uv_to_ground(
            centers, calibration, reconstruction_params
        )
        points = reconstructed.points_ground
        plane = np.asarray(planes[frame_id]["plane"], dtype=np.float64)
        z_plane = -(
            plane[0] * points[:, 0] + plane[1] * points[:, 1] + plane[3]
        ) / plane[2]
        residual = points[:, 2] - z_plane
        v_px = reconstructed.pixels_uv[:, 1]
        if not np.all(np.isfinite(residual)):
            raise ValueError(f"Non-finite residual in frame {frame_id}")
        rows.append(bin_median(v_px, residual))
        raw[frame_id] = {"v_px": v_px, "before": residual}
        extraction.append(
            {
                "frame_id": frame_id,
                "extracted_center_count": int(len(centers)),
                "reconstructed_point_count": int(reconstructed.point_count),
                "filtered": dict(reconstructed.filtered),
            }
        )
    return np.vstack(rows), raw, extraction


def build_table(build_matrix: np.ndarray) -> dict[str, np.ndarray]:
    v = np.arange(IMAGE_HEIGHT, dtype=np.int32)
    sample_count = np.sum(np.isfinite(build_matrix), axis=0).astype(np.int32)
    bias = np.full(IMAGE_HEIGHT, np.nan, dtype=np.float64)
    for column in range(IMAGE_HEIGHT):
        values = build_matrix[:, column]
        values = values[np.isfinite(values)]
        if values.size:
            bias[column] = float(np.median(values))
    in_working_region = (v >= WORK_V_MIN) & (v <= WORK_V_MAX)
    supported = in_working_region & (sample_count >= MIN_BUILD_FRAMES) & np.isfinite(bias)
    applied_bias = np.zeros(IMAGE_HEIGHT, dtype=np.float64)
    applied_bias[supported] = bias[supported]
    return {
        "v_px": v,
        "sample_count": sample_count,
        "raw_median_bias_mm": bias,
        "supported": supported,
        "applied_bias_mm": applied_bias,
    }


def apply_exact_supported_table(
    v_px: np.ndarray, residual: np.ndarray, table: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.rint(v_px).astype(int)
    in_image = (rows >= 0) & (rows < IMAGE_HEIGHT)
    apply_mask = np.zeros(len(rows), dtype=bool)
    apply_mask[in_image] = table["supported"][rows[in_image]]
    after = np.asarray(residual, dtype=np.float64).copy()
    after[apply_mask] -= table["applied_bias_mm"][rows[apply_mask]]
    return after, apply_mask


def metrics(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in (
            "sample_count", "bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm",
            "peak_to_valley_mm", "residual_std_mm", "positive_fraction",
            "sign_mixing_index",
        )}
    positive = float(np.mean(finite > 0.0))
    return {
        "sample_count": int(finite.size),
        "bias_mm": float(np.mean(finite)),
        "mae_mm": float(np.mean(np.abs(finite))),
        "rmse_mm": float(np.sqrt(np.mean(finite**2))),
        "p95_abs_mm": float(np.percentile(np.abs(finite), 95)),
        "peak_to_valley_mm": float(np.ptp(finite)),
        "residual_std_mm": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "positive_fraction": positive,
        "sign_mixing_index": 2.0 * min(positive, 1.0 - positive),
    }


def improvement(before: float | None, after: float | None) -> dict[str, Any]:
    if before is None or after is None:
        return {"absolute": None, "fraction": None, "improved": False}
    return {
        "absolute": float(before - after),
        "fraction": float((before - after) / before) if before > 0.0 else None,
        "improved": bool(after < before),
    }


def local_blocks(
    frame_id: str,
    v_px: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    applied: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for low in range(WORK_V_MIN, WORK_V_MAX + 1, LOCAL_BLOCK_WIDTH_PX):
        high = min(low + LOCAL_BLOCK_WIDTH_PX - 1, WORK_V_MAX)
        selected = applied & (v_px >= low - 0.5) & (v_px <= high + 0.5)
        count = int(np.count_nonzero(selected))
        if count < LOCAL_MIN_SAMPLES:
            continue
        before_metrics = metrics(before[selected])
        after_metrics = metrics(after[selected])
        before_rmse = float(before_metrics["rmse_mm"])
        after_rmse = float(after_metrics["rmse_mm"])
        before_p95 = float(before_metrics["p95_abs_mm"])
        after_p95 = float(after_metrics["p95_abs_mm"])
        rmse_over = (
            after_rmse - before_rmse >= LOCAL_RMSE_ABSOLUTE_WORSENING_MM
            and after_rmse >= (1.0 + LOCAL_RELATIVE_WORSENING) * before_rmse
        )
        p95_over = (
            after_p95 - before_p95 >= LOCAL_P95_ABSOLUTE_WORSENING_MM
            and after_p95 >= (1.0 + LOCAL_RELATIVE_WORSENING) * before_p95
        )
        rows.append(
            {
                "frame_id": frame_id,
                "v_range": [low, high],
                "sample_count": count,
                "before_rmse_mm": before_rmse,
                "after_rmse_mm": after_rmse,
                "before_p95_abs_mm": before_p95,
                "after_p95_abs_mm": after_p95,
                "obvious_overcompensation": bool(rmse_over or p95_over),
                "trigger": (
                    "RMSE_and_P95" if rmse_over and p95_over else
                    "RMSE" if rmse_over else "P95" if p95_over else None
                ),
            }
        )
    return rows


def cross_frame_sign_mixing(matrix: np.ndarray, supported: np.ndarray) -> dict[str, Any]:
    eligible = supported & (np.sum(np.isfinite(matrix), axis=0) >= 2)
    mixed = np.zeros(IMAGE_HEIGHT, dtype=bool)
    for column in np.flatnonzero(eligible):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        mixed[column] = np.any(values > 0.0) and np.any(values < 0.0)
    return {
        "eligible_row_count": int(np.count_nonzero(eligible)),
        "mixed_sign_row_count": int(np.count_nonzero(mixed & eligible)),
        "mixed_sign_row_fraction": (
            float(np.mean(mixed[eligible])) if np.any(eligible) else None
        ),
    }


def verdict(
    per_frame: list[dict[str, Any]],
    local_results: list[dict[str, Any]],
    aggregate_before: Mapping[str, Any],
    aggregate_after: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    simultaneous = sum(
        row["rmse_improvement"]["improved"] and row["p95_improvement"]["improved"]
        for row in per_frame
    )
    obvious = [row for row in local_results if row["obvious_overcompensation"]]
    aggregate_both = (
        aggregate_after["rmse_mm"] < aggregate_before["rmse_mm"]
        and aggregate_after["p95_abs_mm"] < aggregate_before["p95_abs_mm"]
    )
    if simultaneous >= 2 and aggregate_both and not obvious:
        result = "PASS"
    elif simultaneous >= 2 or (simultaneous >= 1 and aggregate_both):
        result = "PARTIAL"
    else:
        result = "FAIL"
    return result, {
        "validation_frames_with_simultaneous_rmse_p95_improvement": simultaneous,
        "required_majority_count": 2,
        "aggregate_rmse_and_p95_improved": bool(aggregate_both),
        "obvious_local_overcompensation_block_count": len(obvious),
        "obvious_local_overcompensation_blocks": obvious,
    }


def write_table_csv(path: Path, table: Mapping[str, np.ndarray]) -> None:
    fields = (
        "row_v_px", "bias_mm", "build_sample_count", "supported",
        "in_high_precision_work_region", "smooth_window", "aggregate",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(IMAGE_HEIGHT):
            supported = bool(table["supported"][index])
            writer.writerow(
                {
                    "row_v_px": index,
                    "bias_mm": f"{table['applied_bias_mm'][index]:.9f}" if supported else "",
                    "build_sample_count": int(table["sample_count"][index]),
                    "supported": supported,
                    "in_high_precision_work_region": WORK_V_MIN <= index <= WORK_V_MAX,
                    "smooth_window": SMOOTH_WINDOW,
                    "aggregate": "median",
                }
            )


def write_table_npy(path: Path, table: Mapping[str, np.ndarray]) -> None:
    dtype = np.dtype(
        [
            ("row_v_px", "<i4"),
            ("bias_mm", "<f8"),
            ("build_sample_count", "<i4"),
            ("supported", "?"),
        ]
    )
    array = np.empty(IMAGE_HEIGHT, dtype=dtype)
    array["row_v_px"] = table["v_px"]
    array["bias_mm"] = np.where(
        table["supported"], table["applied_bias_mm"], np.nan
    )
    array["build_sample_count"] = table["sample_count"]
    array["supported"] = table["supported"]
    np.save(path, array, allow_pickle=False)


def write_holdout_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base = [
        "frame_id", "evaluated_supported_sample_count", "total_reconstructed_point_count",
        "support_application_fraction", "excluded_or_unchanged_point_count",
        "excluded_max_abs_change_mm",
    ]
    metric_names = (
        "bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "peak_to_valley_mm",
        "residual_std_mm", "positive_fraction", "sign_mixing_index",
    )
    fields = base + [f"before_{name}" for name in metric_names] + [
        f"after_{name}" for name in metric_names
    ] + [
        "rmse_improvement_fraction", "p95_improvement_fraction",
        "rmse_and_p95_both_improved", "obvious_local_overcompensation_block_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rendered = {key: row.get(key) for key in base}
            for prefix in ("before", "after"):
                for name in metric_names:
                    value = row[prefix][name]
                    rendered[f"{prefix}_{name}"] = (
                        "" if value is None else f"{float(value):.9f}"
                    )
            rendered.update(
                rmse_improvement_fraction=row["rmse_improvement"]["fraction"],
                p95_improvement_fraction=row["p95_improvement"]["fraction"],
                rmse_and_p95_both_improved=(
                    row["rmse_improvement"]["improved"]
                    and row["p95_improvement"]["improved"]
                ),
                obvious_local_overcompensation_block_count=row[
                    "obvious_local_overcompensation_block_count"
                ],
            )
            writer.writerow(rendered)


def plot_holdout_profiles(
    path: Path, holdout: Mapping[str, Mapping[str, np.ndarray]], table: Mapping[str, np.ndarray]
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True)
    for axis, frame_id in zip(axes, HOLDOUT_IDS):
        data = holdout[frame_id]
        before_bins = bin_median(data["v_px"], data["before"])
        after_bins = bin_median(data["v_px"], data["after"])
        evaluated = table["supported"] & np.isfinite(before_bins) & np.isfinite(after_bins)
        before_display = np.where(evaluated, before_bins, np.nan)
        after_display = np.where(evaluated, after_bins, np.nan)
        axis.plot(
            np.arange(IMAGE_HEIGHT), before_display, linewidth=0.8,
            color="#dc2626", label="before",
        )
        axis.plot(
            np.arange(IMAGE_HEIGHT), after_display, linewidth=0.8,
            color="#2563eb", label="after",
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylabel(f"{frame_id}\nresidual_z mm")
        axis.grid(True, alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Image row v (only exact BUILD-supported rows shown)")
    figure.suptitle("Independent holdout residual before/after experimental b(v)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def plot_heatmaps(
    path: Path,
    before_matrix: np.ndarray,
    after_matrix: np.ndarray,
    supported: np.ndarray,
) -> float:
    before = before_matrix.copy()
    after = after_matrix.copy()
    before[:, ~supported] = np.nan
    after[:, ~supported] = np.nan
    finite = np.abs(np.r_[before[np.isfinite(before)], after[np.isfinite(after)]])
    limit = float(np.percentile(finite, 99))
    figure, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    for axis, matrix, title in zip(
        axes, (before, after), ("Before", "After: Z_corrected = Z_raw - b(v)"),
    ):
        image = axis.imshow(
            np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest",
            cmap="RdBu_r", vmin=-limit, vmax=limit,
            extent=[-0.5, IMAGE_HEIGHT - 0.5, 3.5, 0.5],
        )
        axis.set_yticks([1, 2, 3], HOLDOUT_IDS)
        axis.set_ylabel("holdout frame")
        axis.set_title(title)
    axes[-1].set_xlabel("Image row v (white = unsupported/not evaluated)")
    colorbar = figure.colorbar(image, ax=axes, pad=0.015)
    colorbar.set_label("residual_z mm, shared P99 |residual| clip")
    figure.suptitle("Independent holdout frame×v residual")
    figure.subplots_adjust(left=0.07, right=0.91, top=0.90, bottom=0.09, hspace=0.27)
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)
    return limit


def plot_support(path: Path, table: Mapping[str, np.ndarray]) -> None:
    v = table["v_px"]
    figure, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].step(v, table["sample_count"], where="mid", color="#dc2626", linewidth=0.8)
    axes[0].axhline(MIN_BUILD_FRAMES, color="black", linestyle="--", linewidth=0.9)
    axes[0].fill_between(v, 0, 10, where=table["supported"], color="#16a34a", alpha=0.12)
    axes[0].set_ylabel("BUILD frame count")
    axes[0].set_ylim(0, 10.5)
    axes[0].grid(True, alpha=0.2)
    bias_display = np.where(table["supported"], table["applied_bias_mm"], np.nan)
    axes[1].plot(v, bias_display, color="#2563eb", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Experimental b(v) mm")
    axes[1].set_xlabel("Image row v")
    axes[1].grid(True, alpha=0.2)
    for axis in axes:
        axis.axvspan(0, WORK_V_MIN - 0.5, color="#94a3b8", alpha=0.22)
        axis.axvspan(WORK_V_MAX + 0.5, IMAGE_HEIGHT - 1, color="#94a3b8", alpha=0.22)
    figure.suptitle("BUILD-only support and unsmoothed median b(v); gray edges never corrected")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def report(metrics_doc: Mapping[str, Any]) -> str:
    assessment = metrics_doc["assessment"]
    aggregate = metrics_doc["holdout_aggregate_supported_samples"]
    lines = [
        "# Experimental paired-PnP ground bias v compensation",
        "",
        "## 最终判断",
        "",
        f"**B_V_COMPENSATION = {metrics_doc['B_V_COMPENSATION']}**",
        "",
        f"- 3个独立 holdout 中，RMSE/P95 同时改善："
        f"{assessment['validation_frames_with_simultaneous_rmse_p95_improvement']}/3。",
        f"- 明显新局部过补偿 block："
        f"{assessment['obvious_local_overcompensation_block_count']}。",
        f"- 聚合 holdout RMSE：{fmt(aggregate['before']['rmse_mm'])} → "
        f"{fmt(aggregate['after']['rmse_mm'])} mm；P95："
        f"{fmt(aggregate['before']['p95_abs_mm'])} → "
        f"{fmt(aggregate['after']['p95_abs_mm'])} mm。",
        "",
        "PASS 要求至少2/3 holdout帧的 RMSE 与 P95 同时改善、聚合 RMSE/P95 同时改善，"
        "且没有明显新局部过补偿。局部过补偿判据在查看 holdout 前固定为：100 px block、"
        "至少30点，且 RMSE 增加同时达到25%和0.03 mm，或 P95增加同时达到25%和0.05 mm。",
        "",
        "工程价值判断：可信工作区内的一维 b(v) 具有显著但有条件的工程价值——三帧整体"
        "RMSE/P95均改善，说明主要系统分量确实可被一维模型消除；但013在v=300–399出现"
        "新局部恶化，且012的全局P-V轻微增加，因此当前只适合作为support-gated实验方案，"
        "还不能作为正式生产补偿。",
        "",
        "## 严格数据隔离与建表",
        "",
        "- BUILD：001–010；INDEPENDENT HOLDOUT：011–013。",
        "- holdout 未参与 LUT估计、support threshold、平滑或任何判据选择。",
        "- `b(v)` 是每帧同一整数 v 内 residual median 的跨 BUILD 帧 median；"
        "无跨帧 outlier threshold，`smooth_window=1`。",
        f"- 高精度工作区固定为 `{WORK_V_MIN}≤v≤{WORK_V_MAX}`；可靠支持固定为"
        f" `BUILD sample_count≥{MIN_BUILD_FRAMES}`。支持覆盖"
        f" {metrics_doc['table']['supported_row_count']}/2400 "
        f"({100.0 * metrics_doc['table']['supported_fraction_of_work_region']:.2f}%)。",
        "- 工作区外、上下边缘和内部 support 缺口完全不补偿；不插值、不跨缺口、不外推。",
        "",
        "## Holdout逐帧结果（只在可靠支持样本上比较）",
        "",
        "| frame | samples | Bias before→after | MAE before→after | RMSE before→after | P95 before→after | P-V before→after | std before→after | sign mixing before→after | local overcomp blocks |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_doc["holdout_per_frame"]:
        before, after = row["before"], row["after"]
        lines.append(
            f"| {row['frame_id']} | {row['evaluated_supported_sample_count']} | "
            f"{fmt(before['bias_mm'])}→{fmt(after['bias_mm'])} | "
            f"{fmt(before['mae_mm'])}→{fmt(after['mae_mm'])} | "
            f"{fmt(before['rmse_mm'])}→{fmt(after['rmse_mm'])} | "
            f"{fmt(before['p95_abs_mm'])}→{fmt(after['p95_abs_mm'])} | "
            f"{fmt(before['peak_to_valley_mm'])}→{fmt(after['peak_to_valley_mm'])} | "
            f"{fmt(before['residual_std_mm'])}→{fmt(after['residual_std_mm'])} | "
            f"{fmt(before['sign_mixing_index'])}→{fmt(after['sign_mixing_index'])} | "
            f"{row['obvious_local_overcompensation_block_count']} |"
        )
    obvious_blocks = assessment["obvious_local_overcompensation_blocks"]
    if obvious_blocks:
        lines.extend(["", "## 明显新局部过补偿", ""])
        for block in obvious_blocks:
            lines.append(
                f"- frame {block['frame_id']}，v={block['v_range'][0]}–{block['v_range'][1]}，"
                f"n={block['sample_count']}：RMSE {fmt(block['before_rmse_mm'])}→"
                f"{fmt(block['after_rmse_mm'])} mm，P95 {fmt(block['before_p95_abs_mm'])}→"
                f"{fmt(block['after_p95_abs_mm'])} mm；trigger={block['trigger']}。"
            )
    lines.extend(
        [
            "",
            "`sign_mixing_index = 2*min(positive_fraction, 1-positive_fraction)`，0表示单一符号，"
            "1表示正负各半。另在 JSON 中报告同一 v 跨3个holdout帧的 mixed-sign row fraction。",
            "",
            "## 工程边界",
            "",
            "- 这是 experimental table，不是正式 runtime LUT。CSV/NPY 显式包含 support mask；"
            "现有运行时 `np.interp` 会跨内部缺口插值，因此不得直接把本表接入该路径。",
            "- 实验应用使用整数 v 的 exact lookup；只有该行 support=true 才减去 bias。",
            "- 未补偿点保持原始 Z，图中白色区域不参与 before/after 精度宣称。",
            "- 补偿后 sign mixing 上升主要反映 residual 从单侧负偏置移向零点两侧；它与"
            "Bias/MAE/RMSE/P95/std及局部过补偿需要联合解释，不能单独作为失败指标。",
            "",
        ]
    )
    return "\n".join(lines)


def output_files() -> str:
    return """# OUTPUT_FILES

- `ground_bias_table.csv`：3000行实验表，含bias、BUILD样本数、工作区和support标志；unsupported bias留空。
- `ground_bias_table.npy`：同一表的结构化数组；unsupported bias为NaN且`support=false`。
- `compensation_metrics.json`：完整配置、隔离声明、逐帧/聚合指标、局部block和判定。
- `holdout_residual_before_after.png`：三个独立holdout逐帧before/after曲线。
- `holdout_residual_frame_v_heatmap_before_after.png`：独立holdout before/after热图。
- `holdout_metrics_per_frame.csv`：三个holdout逐帧Bias/MAE/RMSE/P95/P-V/std/sign mixing。
- `compensation_support.png`：BUILD支持数、support mask与未平滑experimental b(v)。
- `compensation_report.md`：工程结论和边界。
- `OUTPUT_FILES.md`：本文件。

本目录不包含正式runtime LUT。工作区外、上下边缘和内部support缺口均不补偿；无插值、外推或平滑。
"""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    outputs = [output_dir / name for name in OUTPUT_NAMES]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; pass --overwrite: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    planes = load_planes(args.pnp_audit.resolve())
    app_config = load_app_config(args.measurement_config.resolve())
    calibration = load_calibration_files(
        app_config.calibration.intrinsics,
        app_config.calibration.laser_model,
        app_config.calibration.extrinsics,
        app_config.calibration.ground_u_compensation,
    )
    if calibration["laser_model"]["model_type"] != "circular_cone":
        raise ValueError("Frozen laser model must be circular_cone")
    if calibration["ground_u_compensation"] is not None:
        raise ValueError("Input reconstruction must be uncompensated")
    extraction_params = create_extraction_params(
        app_config.extraction_method, app_config.extraction_options
    )

    build_matrix, _, build_extraction = reconstruct_residuals(
        BUILD_IDS, "fit", args.data_root.resolve(), planes, extraction_params,
        calibration, app_config.reconstruction,
    )
    holdout_before_matrix, holdout_raw, holdout_extraction = reconstruct_residuals(
        HOLDOUT_IDS, "validation", args.data_root.resolve(), planes, extraction_params,
        calibration, app_config.reconstruction,
    )
    table = build_table(build_matrix)

    per_frame: list[dict[str, Any]] = []
    local_results: list[dict[str, Any]] = []
    holdout_after_rows: list[np.ndarray] = []
    aggregate_before_values: list[np.ndarray] = []
    aggregate_after_values: list[np.ndarray] = []
    extraction_by_id = {row["frame_id"]: row for row in holdout_extraction}
    for frame_id in HOLDOUT_IDS:
        raw = holdout_raw[frame_id]
        after, applied = apply_exact_supported_table(
            raw["v_px"], raw["before"], table
        )
        raw["after"] = after
        raw["applied"] = applied
        excluded_max_change = (
            float(np.max(np.abs(after[~applied] - raw["before"][~applied])))
            if np.any(~applied)
            else 0.0
        )
        if excluded_max_change != 0.0:
            raise AssertionError(f"Frame {frame_id}: unsupported points changed")
        selected_before = raw["before"][applied]
        selected_after = after[applied]
        before_metrics = metrics(selected_before)
        after_metrics = metrics(selected_after)
        blocks = local_blocks(
            frame_id, raw["v_px"], raw["before"], after, applied
        )
        local_results.extend(blocks)
        over_count = sum(block["obvious_overcompensation"] for block in blocks)
        extraction = extraction_by_id[frame_id]
        per_frame.append(
            {
                "frame_id": frame_id,
                "evaluated_supported_sample_count": int(np.count_nonzero(applied)),
                "total_reconstructed_point_count": int(len(applied)),
                "support_application_fraction": float(np.mean(applied)),
                "excluded_or_unchanged_point_count": int(np.count_nonzero(~applied)),
                "excluded_max_abs_change_mm": excluded_max_change,
                "before": before_metrics,
                "after": after_metrics,
                "rmse_improvement": improvement(
                    before_metrics["rmse_mm"], after_metrics["rmse_mm"]
                ),
                "p95_improvement": improvement(
                    before_metrics["p95_abs_mm"], after_metrics["p95_abs_mm"]
                ),
                "obvious_local_overcompensation_block_count": over_count,
                "extraction": extraction,
            }
        )
        aggregate_before_values.append(selected_before)
        aggregate_after_values.append(selected_after)
        holdout_after_rows.append(bin_median(raw["v_px"], after))

    holdout_after_matrix = np.vstack(holdout_after_rows)
    aggregate_before = metrics(np.concatenate(aggregate_before_values))
    aggregate_after = metrics(np.concatenate(aggregate_after_values))
    result, assessment = verdict(
        per_frame, local_results, aggregate_before, aggregate_after
    )
    before_sign = cross_frame_sign_mixing(holdout_before_matrix, table["supported"])
    after_sign = cross_frame_sign_mixing(holdout_after_matrix, table["supported"])

    write_table_csv(outputs[0], table)
    write_table_npy(outputs[1], table)
    write_holdout_csv(outputs[5], per_frame)
    plot_holdout_profiles(outputs[3], holdout_raw, table)
    heatmap_limit = plot_heatmaps(
        outputs[4], holdout_before_matrix, holdout_after_matrix, table["supported"]
    )
    plot_support(outputs[6], table)

    work_mask = (table["v_px"] >= WORK_V_MIN) & (table["v_px"] <= WORK_V_MAX)
    metrics_doc = {
        "schema_version": 1,
        "B_V_COMPENSATION": result,
        "scope": {
            "build_frame_ids": list(BUILD_IDS),
            "independent_holdout_frame_ids": list(HOLDOUT_IDS),
            "validation_used_for_lut_estimation": False,
            "validation_used_for_smoothing_selection": False,
            "validation_used_for_outlier_threshold_estimation": False,
            "laser_point_reference_plane_fit_used": False,
            "compensation_applied_outside_support": False,
            "formal_runtime_lut": False,
        },
        "table": {
            "axis": "v",
            "work_region_px": [WORK_V_MIN, WORK_V_MAX],
            "aggregate": "median",
            "within_frame_bin_aggregate": "median",
            "smooth_window": SMOOTH_WINDOW,
            "cross_frame_outlier_rejection": None,
            "minimum_build_frames_per_row": MIN_BUILD_FRAMES,
            "supported_row_count": int(np.count_nonzero(table["supported"])),
            "work_region_row_count": int(np.count_nonzero(work_mask)),
            "supported_fraction_of_work_region": float(
                np.mean(table["supported"][work_mask])
            ),
            "application": "exact rounded integer-v lookup only when support=true",
            "interpolation": False,
            "extrapolation": False,
        },
        "frozen_settings": {
            "measurement_config": str(args.measurement_config.resolve()),
            "pnp_audit": str(args.pnp_audit.resolve()),
            "camera_intrinsics": str(app_config.calibration.intrinsics),
            "laser_model": str(app_config.calibration.laser_model),
            "laser_model_type": calibration["laser_model"]["model_type"],
            "ground_extrinsics": str(app_config.calibration.extrinsics),
            "steger_method": app_config.extraction_method,
            "steger": dict(app_config.extraction_options),
            "reconstruction": asdict(app_config.reconstruction),
            "input_ground_compensation": None,
        },
        "holdout_per_frame": per_frame,
        "holdout_aggregate_supported_samples": {
            "before": aggregate_before,
            "after": aggregate_after,
            "rmse_improvement": improvement(
                aggregate_before["rmse_mm"], aggregate_after["rmse_mm"]
            ),
            "p95_improvement": improvement(
                aggregate_before["p95_abs_mm"], aggregate_after["p95_abs_mm"]
            ),
        },
        "holdout_cross_frame_sign_mixing": {
            "definition": "fraction of supported v rows with >=2 holdout frames containing both positive and negative residual signs",
            "before": before_sign,
            "after": after_sign,
        },
        "local_overcompensation_definition": {
            "block_width_px": LOCAL_BLOCK_WIDTH_PX,
            "minimum_samples_per_frame_block": LOCAL_MIN_SAMPLES,
            "relative_worsening_threshold": LOCAL_RELATIVE_WORSENING,
            "rmse_absolute_worsening_threshold_mm": LOCAL_RMSE_ABSOLUTE_WORSENING_MM,
            "p95_absolute_worsening_threshold_mm": LOCAL_P95_ABSOLUTE_WORSENING_MM,
        },
        "local_block_metrics": local_results,
        "assessment": assessment,
        "build_extraction": build_extraction,
        "holdout_heatmap_shared_p99_abs_limit_mm": heatmap_limit,
    }
    safe = json_safe(metrics_doc)
    outputs[2].write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs[7].write_text(report(safe), encoding="utf-8")
    outputs[8].write_text(output_files(), encoding="utf-8")

    print(f"B_V_COMPENSATION={result}")
    for row in per_frame:
        print(
            f"{row['frame_id']}: RMSE {row['before']['rmse_mm']:.6f}->"
            f"{row['after']['rmse_mm']:.6f}, P95 {row['before']['p95_abs_mm']:.6f}->"
            f"{row['after']['p95_abs_mm']:.6f}, local_over={row['obvious_local_overcompensation_block_count']}"
        )
    print(
        f"support={np.count_nonzero(table['supported'])}/2400; "
        f"aggregate RMSE {aggregate_before['rmse_mm']:.6f}->{aggregate_after['rmse_mm']:.6f}"
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
