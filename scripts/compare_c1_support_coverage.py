#!/usr/bin/env python3
"""Compare frozen C1 training support with standard-object v/s coverage.

The FIT file is the exact point table used to freeze C1_4k.  Standard-object
points are read from the existing laser-center, baseline, and height CSVs;
their frozen-PCA ``s`` values are recomputed from the same camera intrinsics
and distortion coefficients.  No C1, K/D, or Cone fitting is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


FIT_IDS = tuple(
    [f"{value:03d}" for value in range(1, 19)]
    + [f"{value:03d}" for value in range(25, 37)]
)
VALIDATION_IDS = {f"{value:03d}" for value in range(19, 25)} | {
    f"{value:03d}" for value in range(37, 41)
}

DATASETS = {
    "20mm": {
        "directories": (
            "frame_012686_measure",
            "frame_011317_measure",
            "frame_009614_measure",
            "frame_008310_measure",
            "frame_007020_measure",
            "frame_005772_measure",
            "frame_004021_measure",
            "frame_000974_measure",
        ),
    },
    "50mm": {
        "directories": (
            "frame_061303_measure",
            "frame_065292_measure",
            "frame_063995_measure",
            "frame_062878_measure",
        ),
    },
}

# Keep the existing 50 mm acceptance position semantics: the first and last
# targets are the declared Top and Bottom positions even though their v medians
# are not at the same pixel thresholds as the 20 mm sweep.
REGION_OVERRIDES = {
    "50mm": {
        "frame_065292_measure": "top",
        "frame_063995_measure": "middle",
        "frame_062878_measure": "middle",
        "frame_061303_measure": "bottom",
    }
}

POINT_SOURCES = (
    ("laser_center", "center_uv"),
    ("baseline", "baseline_uv"),
    ("height", "height_uv"),
)

DEFAULT_FIT_POINTS = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "spatial_residual_observability"
    / "fit_ray_residual_points.csv"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "c1_support_comparison"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-points", type=Path, default=DEFAULT_FIT_POINTS)
    parser.add_argument("--standard-root", type=Path, default=base.DEFAULT_STANDARD_ROOT)
    parser.add_argument("--c1-model", type=Path, default=base.DEFAULT_C1_MODEL)
    parser.add_argument("--measurement-config", type=Path, default=base.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_existing(path: Path, *alternatives: Path) -> Path:
    return base.resolve_existing(path, *alternatives)


def finite(value: Any) -> float:
    return base.finite(value)


def fmt(value: Any, digits: int = 6) -> str:
    number = finite(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def quantile_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return {f"{prefix}_p{q:02d}": math.nan for q in (1, 5, 25, 50, 75, 95, 99)}
    quantiles = np.percentile(finite_values, [1, 5, 25, 50, 75, 95, 99])
    return {
        f"{prefix}_p{q:02d}": float(value)
        for q, value in zip((1, 5, 25, 50, 75, 95, 99), quantiles)
    }


def outside_distance(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.maximum(np.maximum(lower - values, values - upper), 0.0)


def status_from_masks(v_out: np.ndarray, s_out: np.ndarray) -> tuple[str, str, str]:
    v_status = "extrapolation" if bool(np.any(v_out)) else "interpolation"
    s_status = "extrapolation" if bool(np.any(s_out)) else "interpolation"
    both_inside = ~(np.asarray(v_out, dtype=bool) | np.asarray(s_out, dtype=bool))
    if len(both_inside) == 0 or bool(np.all(both_inside)):
        coverage = "interpolation"
    elif bool(np.all(~both_inside)):
        coverage = "extrapolation"
    else:
        coverage = "mixed"
    return v_status, s_status, coverage


def summary_row(
    *,
    record_type: str,
    dataset: str,
    position: str,
    directory: str,
    region: str,
    frame_id: str,
    point_source: str,
    values_v: np.ndarray,
    values_s: np.ndarray,
    fit_v_min: float,
    fit_v_max: float,
    fit_s_min: float,
    fit_s_max: float,
    frozen_s_min: float,
    frozen_s_max: float,
    height_v_median: float | None = None,
) -> dict[str, Any]:
    values_v = np.asarray(values_v, dtype=np.float64)
    values_s = np.asarray(values_s, dtype=np.float64)
    valid = np.isfinite(values_v) & np.isfinite(values_s)
    values_v = values_v[valid]
    values_s = values_s[valid]
    v_out = (values_v < fit_v_min) | (values_v > fit_v_max)
    s_out = (values_s < frozen_s_min) | (values_s > frozen_s_max)
    v_distance = outside_distance(values_v, fit_v_min, fit_v_max)
    s_distance = outside_distance(values_s, frozen_s_min, frozen_s_max)
    v_status, s_status, coverage = status_from_masks(v_out, s_out)

    def outside_stat(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
        selected = values[mask]
        return float(np.percentile(selected, percentile)) if len(selected) else 0.0

    row: dict[str, Any] = {
        "record_type": record_type,
        "dataset": dataset,
        "position": position,
        "directory": directory,
        "region": region,
        "frame_id": frame_id,
        "point_source": point_source,
        "point_index": "",
        "u_px": "",
        "v_px": "",
        "s": "",
        "raw_point_count": int(len(values_v)),
        "valid_point_count": int(len(values_v)),
        "height_v_median_px": height_v_median if height_v_median is not None else "",
        "v_min_px": float(np.min(values_v)) if len(values_v) else math.nan,
        "v_max_px": float(np.max(values_v)) if len(values_v) else math.nan,
        "v_median_px": float(np.median(values_v)) if len(values_v) else math.nan,
        "s_min": float(np.min(values_s)) if len(values_s) else math.nan,
        "s_max": float(np.max(values_s)) if len(values_s) else math.nan,
        "s_median": float(np.median(values_s)) if len(values_s) else math.nan,
        "fit_v_min_px": fit_v_min,
        "fit_v_max_px": fit_v_max,
        "fit_s_min": fit_s_min,
        "fit_s_max": fit_s_max,
        "frozen_s_domain_min": frozen_s_min,
        "frozen_s_domain_max": frozen_s_max,
        "v_inside_count": int(np.count_nonzero(~v_out)),
        "v_extrapolated_count": int(np.count_nonzero(v_out)),
        "v_extrapolated_fraction": float(np.mean(v_out)) if len(v_out) else math.nan,
        "v_outside_distance_mean_px": outside_stat(v_distance, v_out, 50.0),
        "v_outside_distance_p95_px": outside_stat(v_distance, v_out, 95.0),
        "v_outside_distance_max_px": float(np.max(v_distance)) if len(v_distance) else math.nan,
        "s_inside_count": int(np.count_nonzero(~s_out)),
        "s_extrapolated_count": int(np.count_nonzero(s_out)),
        "s_extrapolated_fraction": float(np.mean(s_out)) if len(s_out) else math.nan,
        "s_outside_distance_mean": outside_stat(s_distance, s_out, 50.0),
        "s_outside_distance_p95": outside_stat(s_distance, s_out, 95.0),
        "s_outside_distance_max": float(np.max(s_distance)) if len(s_distance) else math.nan,
        "both_inside_count": int(np.count_nonzero(~v_out & ~s_out)),
        "v_only_extrapolated_count": int(np.count_nonzero(v_out & ~s_out)),
        "s_only_extrapolated_count": int(np.count_nonzero(~v_out & s_out)),
        "v_and_s_extrapolated_count": int(np.count_nonzero(v_out & s_out)),
        "v_status": v_status,
        "s_status": s_status,
        "coverage_status": coverage,
    }
    row.update(quantile_fields("v", values_v))
    row.update(quantile_fields("s", values_s))
    return row


def point_row(
    *,
    record_type: str,
    dataset: str,
    position: str,
    directory: str,
    region: str,
    frame_id: str,
    point_source: str,
    point_index: int,
    u: float,
    v: float,
    s: float,
    fit_v_min: float,
    fit_v_max: float,
    fit_s_min: float,
    fit_s_max: float,
    frozen_s_min: float,
    frozen_s_max: float,
    height_v_median: float | None = None,
) -> dict[str, Any]:
    v_out = bool(v < fit_v_min or v > fit_v_max)
    s_out = bool(s < frozen_s_min or s > frozen_s_max)
    v_status, s_status, coverage = status_from_masks(
        np.asarray([v_out]), np.asarray([s_out])
    )
    return {
        "record_type": record_type,
        "dataset": dataset,
        "position": position,
        "directory": directory,
        "region": region,
        "frame_id": frame_id,
        "point_source": point_source,
        "point_index": point_index,
        "u_px": u,
        "v_px": v,
        "s": s,
        "raw_point_count": "",
        "valid_point_count": "",
        "height_v_median_px": height_v_median if height_v_median is not None else "",
        "v_min_px": "",
        "v_max_px": "",
        "v_median_px": "",
        "s_min": "",
        "s_max": "",
        "s_median": "",
        "fit_v_min_px": fit_v_min,
        "fit_v_max_px": fit_v_max,
        "fit_s_min": fit_s_min,
        "fit_s_max": fit_s_max,
        "frozen_s_domain_min": frozen_s_min,
        "frozen_s_domain_max": frozen_s_max,
        "v_inside_count": "",
        "v_extrapolated_count": "",
        "v_extrapolated_fraction": "",
        "v_outside_distance_mean_px": "",
        "v_outside_distance_p95_px": "",
        "v_outside_distance_max_px": max(fit_v_min - v, v - fit_v_max, 0.0),
        "s_inside_count": "",
        "s_extrapolated_count": "",
        "s_extrapolated_fraction": "",
        "s_outside_distance_mean": "",
        "s_outside_distance_p95": "",
        "s_outside_distance_max": max(frozen_s_min - s, s - frozen_s_max, 0.0),
        "both_inside_count": "",
        "v_only_extrapolated_count": "",
        "s_only_extrapolated_count": "",
        "v_and_s_extrapolated_count": "",
        "v_status": v_status,
        "s_status": s_status,
        "coverage_status": coverage,
    }


def load_fit_points(path: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    required = {"frame_id", "u_px", "v_px", "pca_s"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"{path} must contain {sorted(required)}")
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        frame_id = str(row["frame_id"]).zfill(3)
        if frame_id in VALIDATION_IDS:
            raise RuntimeError(f"Validation frame {frame_id} appeared in FIT input {path}")
        if frame_id not in FIT_IDS:
            raise RuntimeError(f"Unexpected frame {frame_id} appeared in FIT input {path}")
        parsed.append(
            {
                "frame_id": frame_id,
                "source_dataset": row.get("source_dataset", ""),
                "point_index": int(row.get("point_index_valid", index)),
                "u": float(row["u_px"]),
                "v": float(row["v_px"]),
                "s": float(row["pca_s"]),
            }
        )
    if {row["frame_id"] for row in parsed} != set(FIT_IDS):
        raise RuntimeError("FIT input frame set does not equal 001–018, 025–036")
    return parsed, np.asarray([row["v"] for row in parsed]), np.asarray([row["s"] for row in parsed])


def load_standard_items(standard_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        probes: list[dict[str, Any]] = []
        for directory in spec["directories"]:
            item = base.input_record(standard_root, directory, directory, "middle")
            item["probe_v_median"] = float(np.median(item["height_uv"][:, 1]))
            probes.append(item)
        probes.sort(key=lambda item: item["probe_v_median"])
        for index, item in enumerate(probes, start=1):
            v_median = float(item["probe_v_median"])
            region = REGION_OVERRIDES.get(dataset, {}).get(
                item["directory"],
                "top" if v_median < 300.0 else "bottom" if v_median >= 2700.0 else "middle",
            )
            item.update(
                {
                    "dataset": dataset,
                    "position": f"P{index:02d}_v{v_median:.1f}",
                    "region": region,
                }
            )
            items.append(item)
    return items


def standard_s_values(uv: np.ndarray, calibration: Mapping[str, Any], c1_model: Mapping[str, Any]) -> np.ndarray:
    pixels = np.asarray(uv, dtype=np.float64)
    K = np.asarray(calibration["K"], dtype=np.float64)
    D = np.asarray(calibration["D"], dtype=np.float64)
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    return base.pca_s_values(normalized, c1_model)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    value = ""
                elif isinstance(value, (bool, np.bool_)):
                    value = "true" if bool(value) else "false"
                output[key] = value
            writer.writerow(output)


def make_plot(
    output_path: Path,
    fit_v: np.ndarray,
    fit_s: np.ndarray,
    position_summaries: Sequence[Mapping[str, Any]],
    standard_points: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    fit_v_min: float,
    fit_v_max: float,
    fit_s_min: float,
    fit_s_max: float,
) -> None:
    rng = np.random.default_rng(20260817)
    position_summaries = list(position_summaries)
    labels = [f"{row['dataset']} {row['position']}" for row in position_summaries]
    y = np.arange(len(position_summaries), dtype=float)
    colors = {"20mm": "#dd6b20", "50mm": "#805ad5"}

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 15),
        gridspec_kw={"height_ratios": (1.1, 1.1, 2.0)},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.axvspan(fit_v_min, fit_v_max, color="#c6f6d5", alpha=0.65, label="FIT v support")
    for index, row in enumerate(position_summaries):
        color = colors[row["dataset"]]
        ax.plot([row["v_min_px"], row["v_max_px"]], [index, index], color=color, lw=7, solid_capstyle="round")
        ax.scatter([row["v_median_px"]], [index], color="black", s=18, zorder=3)
        if int(row["v_extrapolated_count"]) > 0:
            ax.scatter([row["v_min_px"], row["v_max_px"]], [index, index], color="#c53030", s=22, zorder=4)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 3000)
    ax.set_ylabel("standard position")
    ax.set_title("v coverage: FIT support versus standard-object height points")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.axvspan(fit_s_min, fit_s_max, color="#c6f6d5", alpha=0.65, label="Frozen C1 s support")
    for index, row in enumerate(position_summaries):
        color = colors[row["dataset"]]
        ax.plot([row["s_min"], row["s_max"]], [index, index], color=color, lw=7, solid_capstyle="round")
        ax.scatter([row["s_median"]], [index], color="black", s=18, zorder=3)
        if int(row["s_extrapolated_count"]) > 0:
            ax.scatter([row["s_min"], row["s_max"]], [index, index], color="#c53030", s=22, zorder=4)
    s_margin = max(0.01, 0.08 * (fit_s_max - fit_s_min))
    ax.set_xlim(min(fit_s_min, min(row["s_min"] for row in position_summaries)) - s_margin, max(fit_s_max, max(row["s_max"] for row in position_summaries)) + s_margin)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylabel("standard position")
    ax.set_title("s coverage: frozen C1 domain versus standard-object height points")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper right")

    ax = axes[2]
    fit_indices = np.arange(len(fit_s))
    if len(fit_indices) > 9000:
        fit_indices = rng.choice(fit_indices, size=9000, replace=False)
    ax.scatter(fit_s[fit_indices], fit_v[fit_indices], s=3, alpha=0.12, color="#4a5568", label="FIT points")
    ax.axvspan(fit_s_min, fit_s_max, color="#c6f6d5", alpha=0.22)
    ax.axhspan(fit_v_min, fit_v_max, color="#c6f6d5", alpha=0.22)
    ax.axvline(fit_s_min, color="#276749", ls="--", lw=1)
    ax.axvline(fit_s_max, color="#276749", ls="--", lw=1)
    ax.axhline(fit_v_min, color="#276749", ls="--", lw=1)
    ax.axhline(fit_v_max, color="#276749", ls="--", lw=1)

    for dataset in DATASETS:
        center_s, center_v, _ = standard_points[(dataset, "laser_center")]
        indices = np.arange(len(center_s))
        if len(indices) > 1800:
            indices = rng.choice(indices, size=1800, replace=False)
        ax.scatter(center_s[indices], center_v[indices], s=4, alpha=0.05, color=colors[dataset])
        height_s, height_v, height_out = standard_points[(dataset, "height")]
        inside = ~height_out
        ax.scatter(height_s[inside], height_v[inside], s=17, alpha=0.9, color=colors[dataset], edgecolors="white", linewidths=0.25, label=f"{dataset} height inside")
        ax.scatter(height_s[~inside], height_v[~inside], s=24, alpha=0.95, color="#c53030", marker="x", label=f"{dataset} height extrapolation")

    ax.set_ylim(3000, 0)
    ax.set_xlim(min(fit_s_min, np.min(fit_s)) - s_margin, max(fit_s_max, np.max(fit_s)) + s_margin)
    ax.set_xlabel("frozen PCA s")
    ax.set_ylabel("v / px (image coordinates)")
    ax.set_title("joint v/s coverage (FIT cloud, standard laser-center cloud, and height points)")
    ax.grid(alpha=0.2)
    handles, labels_seen = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels_seen):
        unique.setdefault(label, handle)
    unique.setdefault("FIT support", Line2D([0], [0], color="#276749", ls="--"))
    ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=2, loc="best")

    fig.suptitle(
        f"C1_4k support comparison | FIT v [{fit_v_min:.1f}, {fit_v_max:.1f}] px | "
        f"Frozen s [{fit_s_min:.6f}, {fit_s_max:.6f}]",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def report_text(
    *,
    output_dir: Path,
    fit_path: Path,
    fit_sha256: str,
    c1_model_path: Path,
    c1_model_sha256: str,
    c1_parameter_sha256: str,
    fit_summary: Mapping[str, Any],
    fit_frame_count: int,
    frame_027_retained: bool,
    frozen_domain_match: bool,
    standard_summaries: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    height_rows = [row for row in standard_summaries if row["point_source"] == "height"]
    center_rows = [row for row in standard_summaries if row["point_source"] == "laser_center"]
    lines = [
        "# Frozen C1_4k v/s support comparison",
        "",
        f"`C1_SUPPORT_COVERAGE = {decision['C1_SUPPORT_COVERAGE']}`",
        f"`NEED_TOP_BOTTOM_FIT = {decision['need_top_bottom_fit']}`",
        "",
        "## Scope and frozen boundary",
        "",
        f"- FIT 输入：`{fit_path}`；只保留 `001–018、025–036`，共 {fit_summary['valid_point_count']} 个实际 C1 拟合点、{fit_frame_count} 帧；frame 027 retained = `{str(frame_027_retained).lower()}`。",
        f"- C1 模型：`{c1_model_path}`；model SHA-256 = `{c1_model_sha256}`；parameter SHA-256 = `{c1_parameter_sha256}`。",
        f"- FIT CSV SHA-256 = `{fit_sha256}`。",
        "- 标准件只读取既有 20 mm、50 mm 测试目录的 `laser_center.csv`、`baseline_points.csv`、`height_points.csv`；未读取 Validation 019–024、037–040。",
        "- 未重新拟合 C1、K/D 或 Cone，未修改 knots/penalty；s 使用 Frozen PCA 定义。",
        "- `v` 支持域定义为 FIT 实际拟合点的全局 min/max；`s` 支持域定义为 Frozen C1 的 `domain_min/domain_max`，并与 FIT 实际 `pca_s` min/max 做一致性核对。",
        "- 覆盖判定按轴独立：点在闭区间内为 interpolation，区间外为 extrapolation；超出距离是到最近域边界的距离。",
        "- 标准件位置的主要判断使用 height subset；CSV 同时保留 laser-center、baseline、height 三类点的逐点状态。",
        "",
        "## FIT training support",
        "",
        f"- FIT v support：`[{fmt(fit_summary['v_min_px'], 3)}, {fmt(fit_summary['v_max_px'], 3)}] px`。",
        f"- FIT s support：`[{fmt(fit_summary['s_min'], 9)}, {fmt(fit_summary['s_max'], 9)}]`。",
        f"- Frozen C1 s domain 与 FIT s min/max 一致：`{str(frozen_domain_match).lower()}`。",
        "",
        "| coordinate | p01 | p05 | p25 | median | p75 | p95 | p99 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| v / px | {fmt(fit_summary['v_p01'], 3)} | {fmt(fit_summary['v_p05'], 3)} | {fmt(fit_summary['v_p25'], 3)} | {fmt(fit_summary['v_p50'], 3)} | {fmt(fit_summary['v_p75'], 3)} | {fmt(fit_summary['v_p95'], 3)} | {fmt(fit_summary['v_p99'], 3)} | {fmt(fit_summary['v_min_px'], 3)} | {fmt(fit_summary['v_max_px'], 3)} |",
        f"| s | {fmt(fit_summary['s_p01'], 6)} | {fmt(fit_summary['s_p05'], 6)} | {fmt(fit_summary['s_p25'], 6)} | {fmt(fit_summary['s_p50'], 6)} | {fmt(fit_summary['s_p75'], 6)} | {fmt(fit_summary['s_p95'], 6)} | {fmt(fit_summary['s_p99'], 6)} | {fmt(fit_summary['s_min'], 9)} | {fmt(fit_summary['s_max'], 9)} |",
        "",
        "## Standard-object height-point coverage",
        "",
        "下表的 outside count/fraction 同时报告 v-domain 和 s-domain；`coverage` 是两者联合状态。",
        "",
        "| dataset | position | v range / px | v outside | max v distance / px | s range | s outside | max s distance | coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in height_rows:
        lines.append(
            f"| {row['dataset']} | {row['position']} | [{fmt(row['v_min_px'], 1)}, {fmt(row['v_max_px'], 1)}] | "
            f"{row['v_extrapolated_count']} ({100.0 * finite(row['v_extrapolated_fraction']):.1f}%) | {fmt(row['v_outside_distance_max_px'], 3)} | "
            f"[{fmt(row['s_min'], 6)}, {fmt(row['s_max'], 6)}] | {row['s_extrapolated_count']} ({100.0 * finite(row['s_extrapolated_fraction']):.1f}%) | "
            f"{fmt(row['s_outside_distance_max'], 6)} | {row['coverage_status']} |"
        )
    lines.extend(
        [
            "",
            "## Standard-object laser-center coverage",
            "",
            "laser-center 覆盖通常比 height subset 更宽；它反映原始测量行中所有可见点的域外比例，不替代上表的目标位置判断。",
            "",
            "| dataset | position | laser-center v range | v outside | laser-center s range | s outside | coverage |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in center_rows:
        lines.append(
            f"| {row['dataset']} | {row['position']} | [{fmt(row['v_min_px'], 1)}, {fmt(row['v_max_px'], 1)}] | "
            f"{row['v_extrapolated_count']} ({100.0 * finite(row['v_extrapolated_fraction']):.1f}%) | "
            f"[{fmt(row['s_min'], 6)}, {fmt(row['s_max'], 6)}] | {row['s_extrapolated_count']} ({100.0 * finite(row['s_extrapolated_fraction']):.1f}%) | {row['coverage_status']} |"
        )

    top_20 = next(row for row in height_rows if row["dataset"] == "20mm" and row["region"] == "top")
    bottom_20 = next(row for row in height_rows if row["dataset"] == "20mm" and row["region"] == "bottom")
    top_bottom_edge = [row for row in height_rows if row["region"] in {"top", "bottom"}]
    edge_s_gap = [row for row in top_bottom_edge if int(row["s_extrapolated_count"]) > 0]
    edge_v_gap = [row for row in top_bottom_edge if int(row["v_extrapolated_count"]) > 0]
    edge_s_gap_text = ", ".join(
        f"{row['dataset']} {row['position']}" for row in edge_s_gap
    ) or "none"
    edge_v_gap_text = ", ".join(
        f"{row['dataset']} {row['position']}" for row in edge_v_gap
    ) or "none"
    lines.extend(
        [
            "",
            "## Edge findings",
            "",
            f"- v≈125 对应 {top_20['dataset']} {top_20['position']}：FIT v 下界为 {fmt(fit_summary['v_min_px'], 3)} px，height v range 为 [{fmt(top_20['v_min_px'], 1)}, {fmt(top_20['v_max_px'], 1)}] px，{top_20['v_extrapolated_count']}/{top_20['valid_point_count']} 点 v 外推；s 也有 {top_20['s_extrapolated_count']}/{top_20['valid_point_count']} 点外推。",
            f"- v≈2905 对应 {bottom_20['dataset']} {bottom_20['position']}：FIT v 上界为 {fmt(fit_summary['v_max_px'], 3)} px，height v range 为 [{fmt(bottom_20['v_min_px'], 1)}, {fmt(bottom_20['v_max_px'], 1)}] px，{bottom_20['v_extrapolated_count']}/{bottom_20['valid_point_count']} 点 v 外推；s 有 {bottom_20['s_extrapolated_count']}/{bottom_20['valid_point_count']} 点外推。",
            f"- Top/Bottom height positions with s extrapolation: {edge_s_gap_text}。",
            f"- Top/Bottom height positions with v extrapolation: {edge_v_gap_text}。",
            "- C1 实际自变量是 s，因此补采优先级以 s-domain gap 为准；v-domain gap 是对应的传感器位置证据。",
            "",
            "## Decision",
            "",
            f"- `C1_SUPPORT_COVERAGE = {decision['C1_SUPPORT_COVERAGE']}`。",
            f"- `NEED_TOP_BOTTOM_FIT = {decision['need_top_bottom_fit']}`。",
            f"- 判定依据：{decision['decision_reason']}",
            "- 建议补采/扩展 FIT：优先覆盖 20 mm Top（v≈125，s 约 -0.18）和 20 mm Bottom（v≈2905，s 约 +0.20）；50 mm Bottom 仍有小段 s 超域，也应作为边界余量一并覆盖。",
            "- 当前 C1 参数本身没有被修改；在补采前，s-domain 外只能视为外推，不能当作已验证的 interpolation。",
            "",
            "## Artifacts",
            "",
            f"- `c1_support_comparison.csv`: `{output_dir / 'c1_support_comparison.csv'}`",
            f"- `c1_support_comparison.png`: `{output_dir / 'c1_support_comparison.png'}`",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    fit_path = resolve_existing(args.fit_points)
    standard_root = resolve_existing(
        args.standard_root,
        Path(str(args.standard_root).replace("linelascan", "linelaserscan")),
    )
    c1_model_path = resolve_existing(args.c1_model)
    measurement_config = resolve_existing(args.measurement_config)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    c1_model, c1_model_sha256 = frozen.load_frozen_json(c1_model_path)
    fit_rows, fit_v, fit_s = load_fit_points(fit_path)
    fit_v_min = float(np.min(fit_v))
    fit_v_max = float(np.max(fit_v))
    fit_s_min = float(np.min(fit_s))
    fit_s_max = float(np.max(fit_s))
    frozen_s_min = float(c1_model["pca_s"]["domain_min"])
    frozen_s_max = float(c1_model["pca_s"]["domain_max"])
    frozen_domain_match = bool(
        np.isclose(fit_s_min, frozen_s_min, atol=1.0e-12)
        and np.isclose(fit_s_max, frozen_s_max, atol=1.0e-12)
    )

    fit_summary = summary_row(
        record_type="fit_summary",
        dataset="FIT",
        position="All_fit_points",
        directory="",
        region="all",
        frame_id="",
        point_source="fit",
        values_v=fit_v,
        values_s=fit_s,
        fit_v_min=fit_v_min,
        fit_v_max=fit_v_max,
        fit_s_min=fit_s_min,
        fit_s_max=fit_s_max,
        frozen_s_min=frozen_s_min,
        frozen_s_max=frozen_s_max,
    )
    fit_summary["fit_source_sha256"] = sha256_file(fit_path)
    fit_summary["frozen_domain_match"] = frozen_domain_match

    all_output_rows: list[dict[str, Any]] = [fit_summary]
    by_frame: dict[str, list[dict[str, Any]]] = {frame_id: [] for frame_id in FIT_IDS}
    for row in fit_rows:
        by_frame[row["frame_id"]].append(row)
        all_output_rows.append(
            point_row(
                record_type="fit_point",
                dataset="FIT",
                position=f"Frame_{row['frame_id']}",
                directory="",
                region="all",
                frame_id=row["frame_id"],
                point_source="fit",
                point_index=row["point_index"],
                u=row["u"],
                v=row["v"],
                s=row["s"],
                fit_v_min=fit_v_min,
                fit_v_max=fit_v_max,
                fit_s_min=fit_s_min,
                fit_s_max=fit_s_max,
                frozen_s_min=frozen_s_min,
                frozen_s_max=frozen_s_max,
            )
        )
    for frame_id in FIT_IDS:
        frame_rows = by_frame[frame_id]
        all_output_rows.append(
            summary_row(
                record_type="fit_frame_summary",
                dataset="FIT",
                position=f"Frame_{frame_id}",
                directory="",
                region="all",
                frame_id=frame_id,
                point_source="fit",
                values_v=np.asarray([row["v"] for row in frame_rows]),
                values_s=np.asarray([row["s"] for row in frame_rows]),
                fit_v_min=fit_v_min,
                fit_v_max=fit_v_max,
                fit_s_min=fit_s_min,
                fit_s_max=fit_s_max,
                frozen_s_min=frozen_s_min,
                frozen_s_max=frozen_s_max,
            )
        )

    _, calibration, _, _ = fixed.load_runtime(measurement_config)
    standard_items = load_standard_items(standard_root)
    standard_summaries: list[dict[str, Any]] = []
    plot_points: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for item in standard_items:
        height_v_median = float(item["probe_v_median"])
        for point_source, uv_key in POINT_SOURCES:
            uv = np.asarray(item[uv_key], dtype=np.float64)
            values_v = uv[:, 1]
            values_s = standard_s_values(uv, calibration, c1_model)
            row = summary_row(
                record_type="standard_position_summary",
                dataset=item["dataset"],
                position=item["position"],
                directory=item["directory"],
                region=item["region"],
                frame_id="",
                point_source=point_source,
                values_v=values_v,
                values_s=values_s,
                fit_v_min=fit_v_min,
                fit_v_max=fit_v_max,
                fit_s_min=fit_s_min,
                fit_s_max=fit_s_max,
                frozen_s_min=frozen_s_min,
                frozen_s_max=frozen_s_max,
                height_v_median=height_v_median,
            )
            row["center_sha256"] = item["center_sha256"]
            row["baseline_sha256"] = item["baseline_sha256"]
            row["height_sha256"] = item["height_sha256"]
            row["result_sha256"] = item["result_sha256"]
            standard_summaries.append(row)
            all_output_rows.append(row)
            point_v_out = (values_v < fit_v_min) | (values_v > fit_v_max)
            point_s_out = (values_s < frozen_s_min) | (values_s > frozen_s_max)
            plot_points[(item["dataset"], point_source)] = (values_s, values_v, point_s_out)
            for point_index, ((u, v), s) in enumerate(zip(uv, values_s)):
                all_output_rows.append(
                    point_row(
                        record_type="standard_point",
                        dataset=item["dataset"],
                        position=item["position"],
                        directory=item["directory"],
                        region=item["region"],
                        frame_id="",
                        point_source=point_source,
                        point_index=point_index,
                        u=float(u),
                        v=float(v),
                        s=float(s),
                        fit_v_min=fit_v_min,
                        fit_v_max=fit_v_max,
                        fit_s_min=fit_s_min,
                        fit_s_max=fit_s_max,
                        frozen_s_min=frozen_s_min,
                        frozen_s_max=frozen_s_max,
                        height_v_median=height_v_median,
                    )
                )

    primary = [row for row in standard_summaries if row["point_source"] == "height"]
    edge_rows = [row for row in primary if row["region"] in {"top", "bottom"}]
    edge_s_gap = [row for row in edge_rows if int(row["s_extrapolated_count"]) > 0]
    any_s_gap = any(int(row["s_extrapolated_count"]) > 0 for row in primary)
    any_v_gap = any(int(row["v_extrapolated_count"]) > 0 for row in primary)
    strong_edge_s_gap = any(
        finite(row["s_extrapolated_fraction"]) >= 0.25 for row in edge_s_gap
    )
    if not any_s_gap and not any_v_gap:
        coverage = "SUFFICIENT"
        reason = "所有标准件 height points 同时落在 FIT v support 与 Frozen C1 s domain 内。"
    elif strong_edge_s_gap:
        coverage = "INSUFFICIENT"
        reason = "Top/Bottom height points 存在明显 s-domain 外推（至少一个边缘位置的 s 外推比例达到 25% 以上）。"
    else:
        coverage = "PARTIAL"
        reason = "存在标准件点超出 v 或 s 支持域，但未形成明显的 Top/Bottom s-domain 缺口。"
    decision = {
        "C1_SUPPORT_COVERAGE": coverage,
        "need_top_bottom_fit": "YES" if edge_s_gap else "NO",
        "decision_reason": reason,
        "edge_s_gap_positions": [f"{row['dataset']} {row['position']}" for row in edge_s_gap],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "c1_support_comparison.csv", all_output_rows)
    position_primary = [row for row in primary]
    make_plot(
        output_dir / "c1_support_comparison.png",
        fit_v,
        fit_s,
        position_primary,
        plot_points,
        fit_v_min,
        fit_v_max,
        frozen_s_min,
        frozen_s_max,
    )
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            fit_path=fit_path,
            fit_sha256=fit_summary["fit_source_sha256"],
            c1_model_path=c1_model_path,
            c1_model_sha256=c1_model_sha256,
            c1_parameter_sha256=str(c1_model["parameter_sha256"]),
            fit_summary=fit_summary,
            fit_frame_count=len(by_frame),
            frame_027_retained="027" in by_frame and bool(by_frame["027"]),
            frozen_domain_match=frozen_domain_match,
            standard_summaries=standard_summaries,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "C1_SUPPORT_COVERAGE": coverage,
                "need_top_bottom_fit": decision["need_top_bottom_fit"],
                "fit_point_count": len(fit_rows),
                "fit_v_domain": [fit_v_min, fit_v_max],
                "fit_s_domain": [fit_s_min, fit_s_max],
                "frozen_s_domain": [frozen_s_min, frozen_s_max],
                "frozen_domain_match": frozen_domain_match,
                "validation_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run(parse_args())
