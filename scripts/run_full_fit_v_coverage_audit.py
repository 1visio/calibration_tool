#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit v-domain coverage and pose redundancy for all FIT frames.

Only FIT triplets are opened.  The script deliberately stops after the
physical-board-mask extraction and coverage statistics; it does not fit any
laser surface model and it never resolves or opens a Validation root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_laser_models_from_triplets as triplets  # noqa: E402


FIT_GROUPS = (
    (
        "fit_001_018",
        ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit",
        tuple(range(1, 19)),
    ),
    (
        "fit_025_036",
        ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension" / "fit",
        tuple(range(25, 37)),
    ),
    (
        "fit_049_054",
        ROOT / "projects" / "daheng" / "data" / "laser_plane_0817" / "fit",
        tuple(range(49, 55)),
    ),
)
PATTERNS = {
    "chess": "chess {id:03d}.tif",
    "background": "nolaser {id:03d}.tif",
    "laser": "laser {id:03d}.tif",
}
V_MIN_PX = 0.0
V_MAX_PX = 3000.0
BIN_WIDTH_PX = 100.0
BIN_EDGES = np.arange(V_MIN_PX, V_MAX_PX + BIN_WIDTH_PX, BIN_WIDTH_PX)
BIN_COUNT = len(BIN_EDGES) - 1
DEFAULT_CONFIG = ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0817" / "full_fit_v_coverage_audit"


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def resolve_intrinsics(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "calibration_result.yaml"
        if not candidate.exists():
            matches = sorted(path.glob("*.yaml"))
            if len(matches) != 1:
                raise FileNotFoundError(f"无法唯一定位内参 YAML：{path}")
            candidate = matches[0]
        return candidate
    return path


def inventory_group(root: Path, ids: Sequence[int]) -> None:
    for image_id in ids:
        frame_id = f"{image_id:03d}"
        for name in (f"chess {frame_id}.tif", f"nolaser {frame_id}.tif", f"laser {frame_id}.tif"):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(path)


def extract_all_fit(cfg: Mapping[str, Any], k: np.ndarray, d: np.ndarray, image_size: Tuple[int, int] | None, output_dir: Path) -> pd.DataFrame:
    board_cfg = dict(cfg["board"])
    extraction_cfg = dict(cfg.get("extraction", {}))
    mode = str(extraction_cfg.get("board_mask_mode", triplets.FULL_BOARD_PHYSICAL)).strip().lower()
    inset_mm = float(extraction_cfg.get("board_mask_inset_mm", 0.0))
    if mode != triplets.FULL_BOARD_PHYSICAL or abs(inset_mm) > 1.0e-12:
        raise RuntimeError(
            "本审计要求 new full_board_physical mask 且 inset=0；"
            f"当前 mode={mode!r}, inset_mm={inset_mm}"
        )
    laser_cfg = triplets.parse_laser_config(cfg.get("laser"))
    orientation = triplets.normalize_laser_orientation(laser_cfg.orientation)
    patterns = dict(cfg.get("patterns", PATTERNS))
    preview_dir = output_dir / "extraction_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for group_name, root, ids in FIT_GROUPS:
        inventory_group(root, ids)
        print(f"Extracting {group_name}: {root}")
        dataset_cfg = {"root": str(root), "ids": list(ids)}
        frame_df = triplets.process_dataset(
            group_name,
            dataset_cfg,
            patterns,
            k,
            d,
            image_size,
            board_cfg,
            extraction_cfg,
            orientation,
            preview_dir,
        )
        expected = {str(image_id) for image_id in ids}
        actual = set(frame_df["image_id"].astype(str))
        if actual != expected:
            raise RuntimeError(f"{group_name} 提取不完整：expected={sorted(expected)}, actual={sorted(actual)}")
        frame_df["fit_group"] = group_name
        frame_df["split"] = "fit"
        frame_df["frame_id"] = frame_df["image_id"].map(lambda value: f"{int(value):03d}")
        frames.append(frame_df)

    result = pd.concat(frames, ignore_index=True)
    result["lambda_truth_mm"] = result["Zc_mm"].to_numpy(dtype=float) / np.maximum(
        result["ray_z"].to_numpy(dtype=float), 1.0e-12
    )
    v_values = result["v_px"].to_numpy(dtype=float)
    result["in_v_domain"] = (v_values >= V_MIN_PX) & (v_values < V_MAX_PX)
    result["v_bin_index"] = np.where(
        result["in_v_domain"], np.floor((v_values - V_MIN_PX) / BIN_WIDTH_PX).astype(int), -1
    )
    result["v_bin"] = result["v_bin_index"].map(
        lambda value: f"v_{int(BIN_EDGES[value]):04d}_{int(BIN_EDGES[value + 1]):04d}" if value >= 0 else "outside"
    )
    result.to_csv(output_dir / "full_fit_points.csv", index=False, encoding="utf-8-sig")
    return result


def frame_contribution_json(group: pd.DataFrame) -> str:
    counts = group["frame_id"].value_counts().sort_index()
    total = max(int(counts.sum()), 1)
    contribution = {
        str(frame_id): {"count": int(count), "ratio": float(count / total)}
        for frame_id, count in counts.items()
    }
    return json.dumps(contribution, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bin_rows(points: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, Set[str]], float]:
    in_domain = points[points["in_v_domain"]].copy()
    bin_frame_sets: Dict[int, Set[str]] = {}
    raw_counts: List[int] = []
    for index in range(BIN_COUNT):
        group = in_domain[in_domain["v_bin_index"] == index]
        bin_frame_sets[index] = set(group["frame_id"].astype(str))
        raw_counts.append(len(group))
    populated_counts = np.asarray([value for value in raw_counts if value > 0], dtype=float)
    median_populated_count = float(np.median(populated_counts)) if populated_counts.size else 0.0
    sparse_threshold = max(50.0, 0.25 * median_populated_count)

    rows: List[Dict[str, Any]] = []
    for index in range(BIN_COUNT):
        v_lo = float(BIN_EDGES[index])
        v_hi = float(BIN_EDGES[index + 1])
        group = in_domain[in_domain["v_bin_index"] == index]
        frame_set = bin_frame_sets[index]
        point_count = int(len(group))
        frame_count = int(len(frame_set))
        if point_count == 0:
            support_class = "unsupported"
        elif frame_count == 1:
            support_class = "single-frame"
        elif point_count < sparse_threshold or frame_count <= 2:
            support_class = "sparse"
        elif frame_count >= max(18, int(np.ceil(36 * 0.5))):
            support_class = "highly-redundant"
        else:
            support_class = "well-supported"

        def span(column: str) -> Tuple[float, float, float, float]:
            if group.empty:
                return (np.nan, np.nan, np.nan, np.nan)
            values = group[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return (np.nan, np.nan, np.nan, np.nan)
            return (float(np.min(values)), float(np.max(values)), float(np.ptp(values)), float(np.median(values)))

        lambda_min, lambda_max, lambda_span, lambda_median = span("lambda_truth_mm")
        z_min, z_max, z_span, z_median = span("Zc_mm")
        rows.append(
            {
                "v_bin": f"v_{int(v_lo):04d}_{int(v_hi):04d}",
                "v_bin_lo_px": v_lo,
                "v_bin_hi_px": v_hi,
                "point_count": point_count,
                "unique_frame_count": frame_count,
                "frame_ids": ";".join(sorted(frame_set)),
                "frame_contribution_json": frame_contribution_json(group),
                "lambda_truth_min_mm": lambda_min,
                "lambda_truth_max_mm": lambda_max,
                "lambda_truth_span_mm": lambda_span,
                "lambda_truth_median_mm": lambda_median,
                "Z_min_mm": z_min,
                "Z_max_mm": z_max,
                "Z_span_mm": z_span,
                "Z_median_mm": z_median,
                "support_class": support_class,
            }
        )
    return pd.DataFrame(rows), bin_frame_sets, sparse_threshold


def pose_rows(points: pd.DataFrame, bin_frame_sets: Mapping[int, Set[str]]) -> pd.DataFrame:
    in_domain = points[points["in_v_domain"]].copy()
    all_frame_ids = sorted(points["frame_id"].astype(str).unique())
    rows: List[Dict[str, Any]] = []
    for frame_id in all_frame_ids:
        frame_all = points[points["frame_id"].astype(str) == frame_id]
        frame_domain = in_domain[in_domain["frame_id"].astype(str) == frame_id]
        occupied = set(frame_domain["v_bin_index"].astype(int))
        exclusive = {index for index in occupied if bin_frame_sets.get(index, set()) == {frame_id}}
        shared = occupied - exclusive
        overlap_rate = float(len(shared) / len(occupied)) if occupied else np.nan
        other_counts = [len(bin_frame_sets[index]) - 1 for index in occupied]
        pair_rates: List[float] = []
        for other_id in all_frame_ids:
            if other_id == frame_id:
                continue
            other_bins = set(in_domain[in_domain["frame_id"].astype(str) == other_id]["v_bin_index"].astype(int))
            if occupied:
                pair_rates.append(float(len(occupied & other_bins) / len(occupied)))
        if not occupied:
            redundancy_class = "NO_COVERAGE"
        elif overlap_rate >= 0.80:
            redundancy_class = "HIGH"
        elif overlap_rate >= 0.50:
            redundancy_class = "MODERATE"
        else:
            redundancy_class = "LOW"

        def v_range(group: pd.DataFrame) -> Tuple[float, float, float]:
            if group.empty:
                return (np.nan, np.nan, np.nan)
            values = group["v_px"].to_numpy(dtype=float)
            return (float(np.min(values)), float(np.max(values)), float(np.ptp(values)))

        v_min, v_max, v_span = v_range(frame_all)
        domain_v_min, domain_v_max, domain_v_span = v_range(frame_domain)
        exclusive_edges = [(float(BIN_EDGES[index]), float(BIN_EDGES[index + 1])) for index in exclusive]
        if exclusive_edges:
            exclusive_v_min = min(edge[0] for edge in exclusive_edges)
            exclusive_v_max = max(edge[1] for edge in exclusive_edges)
            exclusive_v_span = exclusive_v_max - exclusive_v_min
        else:
            exclusive_v_min = exclusive_v_max = exclusive_v_span = np.nan

        rows.append(
            {
                "frame_id": frame_id,
                "fit_group": str(frame_all["fit_group"].iloc[0]),
                "point_count_total": int(len(frame_all)),
                "point_count_in_v_domain": int(len(frame_domain)),
                "point_count_outside_v_domain": int(len(frame_all) - len(frame_domain)),
                "v_min_px": v_min,
                "v_max_px": v_max,
                "v_span_px": v_span,
                "domain_v_min_px": domain_v_min,
                "domain_v_max_px": domain_v_max,
                "domain_v_span_px": domain_v_span,
                "occupied_v_bin_count": int(len(occupied)),
                "exclusive_v_bin_count": int(len(exclusive)),
                "shared_v_bin_count": int(len(shared)),
                "exclusive_v_coverage_rate": float(len(exclusive) / len(occupied)) if occupied else np.nan,
                "overlap_rate": overlap_rate,
                "exclusive_v_min_px": exclusive_v_min,
                "exclusive_v_max_px": exclusive_v_max,
                "exclusive_v_span_px": exclusive_v_span,
                "exclusive_bin_ids": ";".join(
                    f"v_{int(BIN_EDGES[index]):04d}_{int(BIN_EDGES[index + 1]):04d}" for index in sorted(exclusive)
                ),
                "mean_frame_count_in_occupied_bins": float(np.mean([len(bin_frame_sets[index]) for index in occupied])) if occupied else np.nan,
                "max_frame_count_in_occupied_bins": int(max([len(bin_frame_sets[index]) for index in occupied], default=0)),
                "mean_other_pose_count_in_occupied_bins": float(np.mean(other_counts)) if other_counts else np.nan,
                "overlapping_pose_count": int(sum(1 for rate in pair_rates if rate > 0)),
                "mean_pairwise_overlap_rate": float(np.mean(pair_rates)) if pair_rates else np.nan,
                "max_pairwise_overlap_rate": float(np.max(pair_rates)) if pair_rates else np.nan,
                "redundancy_class": redundancy_class,
            }
        )
    return pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)


def classify_overall_coverage(coverage: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    unsupported = coverage[coverage["support_class"] == "unsupported"]
    populated = coverage[coverage["support_class"] != "unsupported"]
    single = coverage[coverage["support_class"] == "single-frame"]
    sparse = coverage[coverage["support_class"] == "sparse"]
    robust = coverage[coverage["support_class"].isin(["well-supported", "highly-redundant"])]
    unsupported_span = float(unsupported["v_bin_hi_px"].max() - unsupported["v_bin_lo_px"].min()) if not unsupported.empty else 0.0
    populated_fraction = float(len(populated) / len(coverage))
    robust_fraction = float(len(robust) / len(coverage))
    if len(unsupported) == 0 and len(single) == 0 and len(sparse) == 0 and robust_fraction >= 0.80:
        label = "SUFFICIENT"
    elif len(unsupported) <= 2 and unsupported_span <= 200.0 and populated_fraction >= 0.80:
        label = "PARTIAL"
    else:
        label = "INSUFFICIENT"
    return label, {
        "unsupported_bin_count": int(len(unsupported)),
        "single_frame_bin_count": int(len(single)),
        "sparse_bin_count": int(len(sparse)),
        "robust_bin_count": int(len(robust)),
        "populated_fraction": populated_fraction,
        "robust_fraction": robust_fraction,
        "unsupported_span_px": unsupported_span,
    }


def classify_overall_redundancy(pose: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    overlap = pose["overlap_rate"].to_numpy(dtype=float)
    overlap = overlap[np.isfinite(overlap)]
    median_overlap = float(np.median(overlap)) if overlap.size else np.nan
    high_count = int(np.count_nonzero(pose["redundancy_class"] == "HIGH"))
    no_unique_count = int(np.count_nonzero(pose["exclusive_v_bin_count"] == 0))
    if median_overlap >= 0.85 and high_count >= max(1, int(np.ceil(len(pose) * 0.50))):
        label = "HIGH"
    elif median_overlap <= 0.55 and high_count < int(np.ceil(len(pose) * 0.25)):
        label = "LOW"
    else:
        label = "MODERATE"
    return label, {
        "median_pose_overlap_rate": median_overlap,
        "mean_pose_overlap_rate": float(np.mean(overlap)) if overlap.size else np.nan,
        "high_redundancy_pose_count": high_count,
        "no_unique_v_coverage_pose_count": no_unique_count,
        "pose_count": int(len(pose)),
    }


def plot_coverage(output: Path, coverage: pd.DataFrame, pose: pd.DataFrame, points: pd.DataFrame) -> None:
    centers = (coverage["v_bin_lo_px"] + coverage["v_bin_hi_px"]) / 2.0
    status_colors = {
        "unsupported": "#bdbdbd",
        "single-frame": "#d62728",
        "sparse": "#ff7f0e",
        "well-supported": "#2ca02c",
        "highly-redundant": "#1f77b4",
    }
    fig = plt.figure(figsize=(14, 15))
    grid = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.0, 1.0, 1.25], hspace=0.38)

    ax = fig.add_subplot(grid[0])
    colors = [status_colors[str(value)] for value in coverage["support_class"]]
    ax.bar(centers, coverage["point_count"], width=92, color=colors, alpha=0.8, label="point count")
    ax.set_ylabel("points / 100 px bin")
    ax.set_xlim(V_MIN_PX, V_MAX_PX)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(centers, coverage["unique_frame_count"], color="black", marker="o", markersize=3, linewidth=1.5, label="unique frames")
    ax2.set_ylabel("unique frames")
    ax2.set_ylim(bottom=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, color in status_colors.items()]
    handles.append(plt.Line2D([], [], color="black", marker="o", label="unique frames"))
    ax.legend(handles=handles, fontsize=8, ncol=3, loc="upper left")
    ax.set_title("FIT v coverage: points, frame multiplicity and support class")

    ax = fig.add_subplot(grid[1], sharex=ax)
    ax.fill_between(centers, coverage["Z_min_mm"], coverage["Z_max_mm"], color="#9467bd", alpha=0.22, label="Z range")
    ax.plot(centers, coverage["Z_min_mm"], color="#9467bd", linewidth=1)
    ax.plot(centers, coverage["Z_max_mm"], color="#9467bd", linewidth=1)
    ax.plot(centers, coverage["Z_median_mm"], color="#4b2e83", linewidth=1.8, label="Z median")
    ax.set_ylabel("Z depth / mm")
    ax.set_xlim(V_MIN_PX, V_MAX_PX)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Per-bin checkerboard truth depth span")

    ax = fig.add_subplot(grid[2], sharex=ax)
    ax.fill_between(centers, coverage["lambda_truth_min_mm"], coverage["lambda_truth_max_mm"], color="#17becf", alpha=0.22, label="lambda truth range")
    ax.plot(centers, coverage["lambda_truth_median_mm"], color="#087f8c", linewidth=1.8, label="lambda truth median")
    ax.set_ylabel("lambda truth / mm")
    ax.set_xlim(V_MIN_PX, V_MAX_PX)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Per-bin ray-domain truth depth span")

    ax = fig.add_subplot(grid[3])
    frame_ids = sorted(pose["frame_id"].astype(str).unique())
    frame_index = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    matrix = np.zeros((len(frame_ids), BIN_COUNT), dtype=float)
    for _, row in points[points["in_v_domain"]].drop_duplicates(["frame_id", "v_bin_index"]).iterrows():
        matrix[frame_index[str(row["frame_id"])], int(row["v_bin_index"])] = 1.0
    ax.imshow(matrix, aspect="auto", interpolation="nearest", extent=(V_MIN_PX, V_MAX_PX, len(frame_ids) - 0.5, -0.5), cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(frame_ids)))
    ax.set_yticklabels(frame_ids, fontsize=7)
    ax.set_xlabel("image v / px")
    ax.set_ylabel("pose")
    ax.set_title("Pose × v-bin occupancy (blue = pose contributes points)")
    ax.set_xlim(V_MIN_PX, V_MAX_PX)

    fig.suptitle("All FIT (001–018, 025–036, 049–054) — full physical-board mask", fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    output: Path,
    config_path: Path,
    intrinsics_path: Path,
    points: pd.DataFrame,
    coverage: pd.DataFrame,
    pose: pd.DataFrame,
    coverage_label: str,
    coverage_diag: Mapping[str, Any],
    redundancy_label: str,
    redundancy_diag: Mapping[str, Any],
    sparse_threshold: float,
) -> None:
    unsupported = coverage[coverage["support_class"] == "unsupported"]
    single = coverage[coverage["support_class"] == "single-frame"]
    sparse = coverage[coverage["support_class"] == "sparse"]
    high_bins = coverage[coverage["support_class"] == "highly-redundant"]
    high_poses = pose[pose["redundancy_class"] == "HIGH"]
    no_unique = pose[pose["exclusive_v_bin_count"] == 0]
    in_domain = points[points["in_v_domain"]]

    def bins_text(frame: pd.DataFrame) -> str:
        return ", ".join(frame["v_bin"].astype(str).tolist()) if not frame.empty else "none"

    z_values = in_domain["Zc_mm"].to_numpy(dtype=float)
    lambda_values = in_domain["lambda_truth_mm"].to_numpy(dtype=float)
    lines = [
        "# 全 FIT v 覆盖与 pose 冗余审核",
        "",
        f"`FULL_V_COVERAGE = {coverage_label}`",
        f"`FIT_REDUNDANCY = {redundancy_label}`",
        "",
        "## Scope",
        "",
        "- 仅读取 FIT 001–018、025–036、049–054 三联图，共 36 pose；未读取 Validation。",
        "- 使用统一 new `full_board_physical` mask：11×8 内角点、20 mm 格距、X=[-20,220] mm、Y=[-20,160] mm、inset=0 mm。",
        f"- 配置：`{config_path}`；内参：`{intrinsics_path}`。",
        "- 仅做 PnP + Steger + per-row single point + continuity + ray/棋盘平面求交；不拟合 Plane/Quadratic/Cone/C1。",
        "- `lambda_truth_mm = Zc_mm / ray_z`，即激光像素射线与对应 PnP 棋盘平面的真实交点参数；`Zc_mm` 为相机坐标深度。",
        "",
        "## Extraction summary",
        "",
        f"- 有效点：{len(points)}；v∈[0,3000) 内点：{len(in_domain)}；域外点：{len(points) - len(in_domain)}。",
        f"- 实际 v 范围：{points['v_px'].min():.2f}–{points['v_px'].max():.2f} px；域内 v 范围：{in_domain['v_px'].min():.2f}–{in_domain['v_px'].max():.2f} px。",
        f"- lambda_truth 范围：{np.nanmin(lambda_values):.3f}–{np.nanmax(lambda_values):.3f} mm，span={np.ptp(lambda_values):.3f} mm。",
        f"- Z 深度范围：{np.nanmin(z_values):.3f}–{np.nanmax(z_values):.3f} mm，span={np.ptp(z_values):.3f} mm。",
        f"- populated bin 的中位点数：{float(coverage.loc[coverage['point_count'] > 0, 'point_count'].median()):.1f}；sparse 阈值：{sparse_threshold:.1f} 点/bin。",
        "",
        "## v-bin support",
        "",
        "| status | bin count | interpretation |",
        "|---|---:|---|",
        f"| unsupported | {coverage_diag['unsupported_bin_count']} | no FIT point in the 100 px bin |",
        f"| single-frame | {coverage_diag['single_frame_bin_count']} | only one pose contributes |",
        f"| sparse | {coverage_diag['sparse_bin_count']} | low point count or ≤2 contributing poses |",
        f"| well-supported | {int(np.count_nonzero(coverage['support_class'] == 'well-supported'))} | multi-pose, non-sparse support |",
        f"| highly-redundant | {int(np.count_nonzero(coverage['support_class'] == 'highly-redundant'))} | ≥18 contributing poses |",
        "",
        f"- unsupported bins: {bins_text(unsupported)}",
        f"- single-frame bins: {bins_text(single)}",
        f"- sparse bins: {bins_text(sparse)}",
        f"- highly-redundant bins: {bins_text(high_bins)}",
        "",
        "Coverage decision rules：",
        "- `SUFFICIENT`：30 个 bin 全部有点，无 single-frame/sparse bin，且 ≥80% bin 为 well-supported 或 highly-redundant。",
        "- `PARTIAL`：最多 2 个、总跨度不超过 200 px 的边缘 unsupported bin，且至少 80% bin 有点；或存在 single/sparse 但未达到 insufficient 条件。",
        "- `INSUFFICIENT`：unsupported 超过上述边界，或有效 v 覆盖比例低于 80%。",
        "",
        "## Pose redundancy",
        "",
        f"- pose overlap rate 中位数：{redundancy_diag['median_pose_overlap_rate']:.3f}；均值：{redundancy_diag['mean_pose_overlap_rate']:.3f}。",
        f"- HIGH pose：{redundancy_diag['high_redundancy_pose_count']} / {redundancy_diag['pose_count']}；没有 exclusive v-bin 的 pose：{redundancy_diag['no_unique_v_coverage_pose_count']}。",
        "- pose 的 `overlap_rate` 定义为：该 pose 所占 v-bin 中，有其他 pose 同时贡献的 bin 比例；exclusive v-bin 是该 pose 在该 bin 中唯一贡献者的覆盖。",
        "- overall redundancy 规则：median overlap≥0.85 且至少一半 pose 为 HIGH → HIGH；median≤0.55 且 HIGH 少于 25% → LOW；其余 → MODERATE。",
        "",
        f"- HIGH redundancy poses：{', '.join(high_poses['frame_id'].tolist()) if not high_poses.empty else 'none'}",
        f"- zero-new-coverage poses：{', '.join(no_unique['frame_id'].tolist()) if not no_unique.empty else 'none'}",
        "",
        "Pose 级别的完整 v 范围、exclusive coverage、overlap rate、pairwise overlap 和 redundancy class 见 `pose_redundancy.csv`。",
        "",
        "## Interpretation",
        "",
        "- 本审核只描述当前 FIT 采样支持和姿态冗余，不自动删除任何 pose。",
        "- highly-redundant 表示统计上存在大量重叠，不等同于数据无效；是否减少 pose 仍需结合 PnP 姿态差异、残差和独立 Validation 决定。",
        "- 每个 v-bin 的点数、frame IDs、frame contribution ratio、lambda_truth/Z 深度跨度见 `full_fit_v_coverage.csv`。",
        "",
        f"结论：`FULL_V_COVERAGE = {coverage_label}`；`FIT_REDUNDANCY = {redundancy_label}`。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit full FIT v coverage and pose redundancy")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--intrinsics", default=None, help="intrinsics 文件或目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = triplets.safe_yaml_load(config_path)
    intrinsic_value = Path(args.intrinsics).resolve() if args.intrinsics else resolve_path(str(cfg["intrinsics"]), config_path.parent)
    intrinsics_path = resolve_intrinsics(intrinsic_value)
    k, d, image_size = triplets.load_intrinsics(intrinsics_path)

    print(f"Config: {config_path}")
    print(f"Intrinsics: {intrinsics_path}")
    print("FIT groups: 001-018, 025-036, 049-054")
    points = extract_all_fit(cfg, k, d, image_size, output_dir)
    coverage, bin_frame_sets, sparse_threshold = bin_rows(points)
    pose = pose_rows(points, bin_frame_sets)
    coverage_label, coverage_diag = classify_overall_coverage(coverage)
    redundancy_label, redundancy_diag = classify_overall_redundancy(pose)

    coverage.to_csv(output_dir / "full_fit_v_coverage.csv", index=False, encoding="utf-8-sig")
    pose.to_csv(output_dir / "pose_redundancy.csv", index=False, encoding="utf-8-sig")
    plot_coverage(output_dir / "v_coverage.png", coverage, pose, points)
    generate_report(
        output_dir / "report.md",
        config_path,
        intrinsics_path,
        points,
        coverage,
        pose,
        coverage_label,
        coverage_diag,
        redundancy_label,
        redundancy_diag,
        sparse_threshold,
    )
    print(f"FULL_V_COVERAGE = {coverage_label}")
    print(f"FIT_REDUNDANCY = {redundancy_label}")
    print(f"Output: {output_dir}")
    print(coverage[["v_bin", "point_count", "unique_frame_count", "support_class"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

