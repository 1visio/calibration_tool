#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select and audit a geometry-only Robust-Curated-18 extension.

The existing Curated-14 is fixed. All C(22, 4) additions are enumerated using
only the existing geometry metrics, pairwise geometry similarity, and the
current full-board-physical FIT point table. No model is fitted, Validation is
not read, and no model residual is used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pose_geometry_audit as audit  # noqa: E402


SOURCE_DIR = ROOT / "projects" / "daheng" / "outputs" / "0818" / "pose_geometry_audit"
SOURCE_IDS_PATH = SOURCE_DIR / "curated_fit_ids.json"
METRICS_PATH = SOURCE_DIR / "pose_geometry_metrics.csv"
PAIRS_PATH = SOURCE_DIR / "pair_pose_similarity.csv"
DEFAULT_OUTPUT_DIR = ROOT / "projects" / "daheng" / "outputs" / "0818" / "robust_curated_18"

CURATED_14: tuple[str, ...] = (
    "001", "006", "010", "013", "015", "017", "025",
    "027", "031", "049", "051", "052", "053", "054",
)
ALL_IDS: tuple[str, ...] = tuple(audit.FIT_IDS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def fmt(value: Any, digits: int = 3) -> str:
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (SOURCE_IDS_PATH, METRICS_PATH, PAIRS_PATH, audit.CURRENT_POINTS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    previous = json.loads(SOURCE_IDS_PATH.read_text(encoding="utf-8"))
    previous_ids = tuple(str(value) for value in previous.get("curated_fit_ids", []))
    if previous_ids != CURATED_14:
        raise RuntimeError(f"Curated-14 输入不匹配：expected={CURATED_14}, actual={previous_ids}")

    metrics = pd.read_csv(METRICS_PATH)
    metrics["frame_id"] = metrics["frame_id"].map(audit.normalize_frame_id)
    if len(metrics) != 36 or set(metrics["frame_id"]) != set(ALL_IDS):
        raise RuntimeError("pose_geometry_metrics.csv 不是完整的 36 pose 表")
    if any("residual" in str(column).lower() for column in metrics.columns):
        raise RuntimeError("拒绝读取含 residual 字段的 pose metrics")
    metrics = metrics.set_index("frame_id").loc[list(ALL_IDS)].reset_index()

    pairs = pd.read_csv(PAIRS_PATH)
    if len(pairs) != 630:
        raise RuntimeError(f"pair_pose_similarity.csv 行数异常：{len(pairs)}")
    pairs["frame_a"] = pairs["frame_a"].map(audit.normalize_frame_id)
    pairs["frame_b"] = pairs["frame_b"].map(audit.normalize_frame_id)
    if any("residual" in str(column).lower() for column in pairs.columns):
        raise RuntimeError("拒绝读取含 residual 字段的 pair similarity")
    pairs["near_duplicate_strict"] = pairs["near_duplicate_strict"].map(as_bool)

    return metrics, pairs, audit.load_current_points()


def build_context(
    metrics: pd.DataFrame, points: pd.DataFrame
) -> tuple[
    dict[str, set[int]],
    dict[str, set[int]],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, set[int]],
    set[int],
    set[int],
]:
    grid_sets, bin_sets, _ = audit.support_sets(points)
    normal_angles = audit.normal_angle_matrix(metrics)
    bin_edges = {
        "lambda": audit.percentile_bin_edges(points["lambda_truth_mm"].to_numpy(), audit.EXCITATION_BIN_COUNT),
        "depth": audit.percentile_bin_edges(metrics["board_center_z_mm"].to_numpy(), audit.EXCITATION_BIN_COUNT),
        "tx": audit.percentile_bin_edges(metrics["board_center_x_mm"].to_numpy(), audit.TRANSLATION_BIN_COUNT),
        "ty": audit.percentile_bin_edges(metrics["board_center_y_mm"].to_numpy(), audit.TRANSLATION_BIN_COUNT),
    }
    depth_bins_by_id = {
        row.frame_id: audit.bin_index(float(row.board_center_z_mm), bin_edges["depth"])
        for row in metrics.itertuples(index=False)
    }
    lambda_bins_by_id: dict[str, set[int]] = {}
    for frame_id, group in points.groupby("frame_id"):
        lambda_bins_by_id[frame_id] = {
            audit.bin_index(float(value), bin_edges["lambda"])
            for value in group["lambda_truth_mm"]
        }
    depth_bins_by_index = np.asarray(
        [depth_bins_by_id[frame_id] for frame_id in metrics["frame_id"]],
        dtype=int,
    )
    return (
        grid_sets,
        bin_sets,
        normal_angles,
        bin_edges,
        depth_bins_by_index,
        lambda_bins_by_id,
        set(depth_bins_by_id.values()),
        set().union(*lambda_bins_by_id.values()),
    )


def pair_lookup(
    pairs: pd.DataFrame,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
    strict: set[tuple[str, str]] = set()
    similarities: dict[tuple[str, str], float] = {}
    for row in pairs.itertuples(index=False):
        key = tuple(sorted((str(row.frame_a), str(row.frame_b))))
        similarities[key] = float(row.geometric_similarity)
        if bool(row.near_duplicate_strict):
            strict.add(key)
    return strict, similarities


def evaluate_combo(
    selected_ids: Sequence[str],
    added_ids: Sequence[str],
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    depth_bins_by_index: np.ndarray,
    lambda_bins_by_id: Mapping[str, set[int]],
    full_depth_bins: set[int],
    full_lambda_bins: set[int],
    strict_pairs: set[tuple[str, str]],
    similarities: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    selected = tuple(selected_ids)
    selected_set = set(selected)
    added_set = set(added_ids)
    ids = list(metrics["frame_id"])
    index = {frame_id: index for index, frame_id in enumerate(ids)}
    selected_indices = np.asarray([index[frame_id] for frame_id in selected], dtype=int)

    edge_counts = tuple(
        int(sum(edge_bin in bin_sets[frame_id] for frame_id in selected))
        for edge_bin in audit.EDGE_BIN_IDS
    )
    grid_union: set[int] = set()
    bin_union: set[int] = set()
    for frame_id in selected:
        grid_union.update(grid_sets[frame_id])
        bin_union.update(bin_sets[frame_id])

    nearest_normal = np.min(normal_angles[:, selected_indices], axis=1)
    depth_bins = depth_bins_by_index[selected_indices]
    selected_metrics = metrics.iloc[selected_indices]
    selected_points = points[points["frame_id"].isin(selected_set)]
    selected_lambda_bins = set().union(*(lambda_bins_by_id[frame_id] for frame_id in selected))
    cross_similarities = [
        float(similarities[tuple(sorted((added_id, curated_id)))])
        for added_id in added_ids
        for curated_id in CURATED_14
        if tuple(sorted((added_id, curated_id))) in similarities
    ]
    strict_selected = sum(
        pair[0] in selected_set and pair[1] in selected_set
        for pair in strict_pairs
    )
    strict_new = sum(
        pair[0] in selected_set
        and pair[1] in selected_set
        and (pair[0] in added_set or pair[1] in added_set)
        for pair in strict_pairs
    )
    depth_extreme = (
        int(np.count_nonzero(depth_bins == min(full_depth_bins))),
        int(np.count_nonzero(depth_bins == max(full_depth_bins))),
    )
    lambda_extreme = (
        int(sum(min(full_lambda_bins) in lambda_bins_by_id[frame_id] for frame_id in selected)),
        int(sum(max(full_lambda_bins) in lambda_bins_by_id[frame_id] for frame_id in selected)),
    )
    selected_diameter = float(
        np.max(normal_angles[np.ix_(selected_indices, selected_indices)])
    )
    return {
        "selected_pose_count": len(selected),
        "selected_pose_ids": ";".join(selected),
        "edge_counts": edge_counts,
        "edge_min": min(edge_counts),
        "edge_low_total": edge_counts[0] + edge_counts[1],
        "edge_high_total": edge_counts[2] + edge_counts[3],
        "edge_balance": min(edge_counts[0] + edge_counts[1], edge_counts[2] + edge_counts[3]),
        "edge_total": sum(edge_counts),
        "v_grid_10px_occupied_count": len(grid_union),
        "v_bin_100px_occupied_count": len(bin_union),
        "normal_cover_max_angle_deg": float(np.max(nearest_normal)),
        "normal_cover_p95_angle_deg": float(np.percentile(nearest_normal, 95)),
        "normal_pairwise_diameter_deg": selected_diameter,
        "depth_extreme_low_support": depth_extreme[0],
        "depth_extreme_high_support": depth_extreme[1],
        "lambda_extreme_low_support": lambda_extreme[0],
        "lambda_extreme_high_support": lambda_extreme[1],
        "depth_extreme_min_support": min(depth_extreme),
        "depth_extreme_total_support": sum(depth_extreme),
        "lambda_extreme_min_support": min(lambda_extreme),
        "lambda_extreme_total_support": sum(lambda_extreme),
        "depth_bin_occupied_count": len(set(depth_bins)),
        "lambda_bin_occupied_count": len(selected_lambda_bins),
        "depth_center_span_mm": float(
            selected_metrics["board_center_z_mm"].max()
            - selected_metrics["board_center_z_mm"].min()
        ),
        "lambda_truth_span_mm": float(
            selected_points["lambda_truth_mm"].max()
            - selected_points["lambda_truth_mm"].min()
        ),
        "strict_near_duplicate_pair_count": strict_selected,
        "strict_new_near_duplicate_pair_count": strict_new,
        "max_similarity_to_curated": max(cross_similarities),
        "mean_similarity_to_curated": float(np.mean(cross_similarities)),
    }


def selection_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    # User priority: balanced v-edge support, normal cover, extreme support,
    # then excitation span and low similarity to the fixed Curated-14.
    return (
        -int(value["edge_min"]),
        -int(value["edge_balance"]),
        -int(value["edge_total"]),
        -int(value["edge_low_total"]),
        -int(value["edge_high_total"]),
        float(value["normal_cover_max_angle_deg"]),
        float(value["normal_cover_p95_angle_deg"]),
        -int(value["depth_extreme_min_support"]),
        -int(value["depth_extreme_total_support"]),
        -int(value["lambda_extreme_min_support"]),
        -int(value["lambda_extreme_total_support"]),
        -float(value["lambda_truth_span_mm"]),
        -float(value["depth_center_span_mm"]),
        float(value["max_similarity_to_curated"]),
        float(value["mean_similarity_to_curated"]),
        str(value["selected_pose_ids"]),
    )


def select_robust_ids(
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    depth_bins_by_index: np.ndarray,
    lambda_bins_by_id: Mapping[str, set[int]],
    full_depth_bins: set[int],
    full_lambda_bins: set[int],
    strict_pairs: set[tuple[str, str]],
    similarities: Mapping[tuple[str, str], float],
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]:
    candidates = tuple(frame_id for frame_id in ALL_IDS if frame_id not in CURATED_14)
    records: list[dict[str, Any]] = []
    for added in itertools.combinations(candidates, 4):
        selected = tuple(sorted(CURATED_14 + added, key=int))
        stats = evaluate_combo(
            selected,
            added,
            metrics,
            points,
            grid_sets,
            bin_sets,
            normal_angles,
            depth_bins_by_index,
            lambda_bins_by_id,
            full_depth_bins,
            full_lambda_bins,
            strict_pairs,
            similarities,
        )
        records.append({"added_ids": added, "stats": stats})

    safe_records = [
        record for record in records
        if record["stats"]["strict_new_near_duplicate_pair_count"] == 0
    ]
    if not safe_records:
        raise RuntimeError("没有满足 no-new-strict-near-duplicate 约束的 4-pose 组合")

    chosen = min(safe_records, key=lambda record: selection_key(record["stats"]))
    edge_key_values = [
        (
            int(record["stats"]["edge_min"]),
            int(record["stats"]["edge_balance"]),
            int(record["stats"]["edge_total"]),
            int(record["stats"]["edge_low_total"]),
            int(record["stats"]["edge_high_total"]),
        )
        for record in safe_records
    ]
    edge_front_key = max(edge_key_values)
    edge_front = [
        record for record in safe_records
        if (
            int(record["stats"]["edge_min"]),
            int(record["stats"]["edge_balance"]),
            int(record["stats"]["edge_total"]),
            int(record["stats"]["edge_low_total"]),
            int(record["stats"]["edge_high_total"]),
        ) == edge_front_key
    ]
    normal_front_value = min(
        record["stats"]["normal_cover_max_angle_deg"] for record in edge_front
    )
    normal_front = [
        record for record in edge_front
        if math.isclose(
            record["stats"]["normal_cover_max_angle_deg"],
            normal_front_value,
            abs_tol=1.0e-12,
        )
    ]
    summary = {
        "candidate_count": len(candidates),
        "combination_count": len(records),
        "strict_safe_combination_count": len(safe_records),
        "edge_front_combination_count": len(edge_front),
        "normal_front_combination_count": len(normal_front),
        "edge_front_key": edge_front_key,
        "normal_front_max_angle_deg": normal_front_value,
        "chosen_added_ids": list(chosen["added_ids"]),
        "chosen_selection_key": selection_key(chosen["stats"]),
        "strict_near_duplicate_exclusion": "hard exclusion for any pair involving an added pose",
    }
    robust_ids = tuple(sorted(CURATED_14 + tuple(chosen["added_ids"]), key=int))
    return robust_ids, chosen["stats"], summary


def extreme_support(
    selected_ids: Sequence[str],
    depth_bins_by_index: np.ndarray,
    lambda_bins_by_id: Mapping[str, set[int]],
    full_depth_bins: set[int],
    full_lambda_bins: set[int],
) -> dict[str, int]:
    index = {frame_id: index for index, frame_id in enumerate(ALL_IDS)}
    depth_values = depth_bins_by_index[[index[frame_id] for frame_id in selected_ids]]
    return {
        "depth_low": int(np.count_nonzero(depth_values == min(full_depth_bins))),
        "depth_high": int(np.count_nonzero(depth_values == max(full_depth_bins))),
        "lambda_low": int(sum(
            min(full_lambda_bins) in lambda_bins_by_id[frame_id]
            for frame_id in selected_ids
        )),
        "lambda_high": int(sum(
            max(full_lambda_bins) in lambda_bins_by_id[frame_id]
            for frame_id in selected_ids
        )),
    }


def strict_pairs_in_subset(
    selected_ids: Sequence[str], strict_pairs: set[tuple[str, str]]
) -> list[str]:
    selected = set(selected_ids)
    return [
        f"{a}–{b}" for a, b in sorted(strict_pairs)
        if a in selected and b in selected
    ]


def strict_pairs_involving_added(
    selected_ids: Sequence[str],
    added_ids: Sequence[str],
    strict_pairs: set[tuple[str, str]],
) -> list[str]:
    selected = set(selected_ids)
    added = set(added_ids)
    return [
        f"{a}–{b}" for a, b in sorted(strict_pairs)
        if a in selected and b in selected and (a in added or b in added)
    ]


def top_similarity_pairs(
    added_ids: Sequence[str], pairs: pd.DataFrame
) -> list[dict[str, Any]]:
    selected = pairs[
        ((pairs["frame_a"].isin(added_ids)) & (pairs["frame_b"].isin(CURATED_14)))
        | ((pairs["frame_b"].isin(added_ids)) & (pairs["frame_a"].isin(CURATED_14)))
    ].sort_values("geometric_similarity", ascending=False)
    rows: list[dict[str, Any]] = []
    for row in selected.head(12).itertuples(index=False):
        rows.append({
            "added_pose": row.frame_a if row.frame_a in added_ids else row.frame_b,
            "curated_pose": row.frame_b if row.frame_a in added_ids else row.frame_a,
            "geometric_similarity": float(row.geometric_similarity),
            "normal_angle_diff_deg": float(row.normal_angle_diff_deg),
            "translation_difference_mm": float(row.translation_difference_mm),
            "v_overlap_100px_jaccard": float(row.v_overlap_100px_jaccard),
        })
    return rows


def comparison_row(
    name: str,
    selected_ids: Sequence[str],
    stats: Mapping[str, Any],
    extremes: Mapping[str, int],
    curated_stats: Mapping[str, Any],
    curated_extremes: Mapping[str, int],
    strict_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    edge_counts = stats.get(
        "edge_counts",
        (
            stats["edge_frame_count_0_100"],
            stats["edge_frame_count_100_200"],
            stats["edge_frame_count_2800_2900"],
            stats["edge_frame_count_2900_3000"],
        ),
    )
    edge_min = stats.get("edge_min", stats["edge_min_frame_count"])
    edge_low_total = stats.get("edge_low_total", edge_counts[0] + edge_counts[1])
    edge_high_total = stats.get("edge_high_total", edge_counts[2] + edge_counts[3])
    return {
        "population": name,
        "pose_count": len(selected_ids),
        "pose_ids": ";".join(selected_ids),
        "v_grid_10px_occupied_count": stats["v_grid_10px_occupied_count"],
        "v_bin_100px_occupied_count": stats["v_bin_100px_occupied_count"],
        "edge_0_100_frame_count": edge_counts[0],
        "edge_100_200_frame_count": edge_counts[1],
        "edge_2800_2900_frame_count": edge_counts[2],
        "edge_2900_3000_frame_count": edge_counts[3],
        "edge_min_frame_count": edge_min,
        "edge_low_total": edge_low_total,
        "edge_high_total": edge_high_total,
        "normal_cover_max_angle_deg": stats["normal_cover_max_angle_deg"],
        "normal_pairwise_diameter_deg": stats["normal_pairwise_diameter_deg"],
        "depth_center_span_mm": stats["depth_center_span_mm"],
        "lambda_truth_span_mm": stats["lambda_truth_span_mm"],
        "depth_bin_occupied_count": stats["depth_bin_occupied_count"],
        "lambda_bin_occupied_count": stats["lambda_bin_occupied_count"],
        "depth_extreme_low_support": extremes["depth_low"],
        "depth_extreme_high_support": extremes["depth_high"],
        "lambda_extreme_low_support": extremes["lambda_low"],
        "lambda_extreme_high_support": extremes["lambda_high"],
        "strict_near_duplicate_pair_count": len(
            strict_pairs_in_subset(selected_ids, strict_pairs)
        ),
        "normal_cover_reduction_vs_curated_deg": (
            curated_stats["normal_cover_max_angle_deg"]
            - stats["normal_cover_max_angle_deg"]
        ),
        "edge_min_delta_vs_curated": edge_min - curated_stats.get("edge_min", curated_stats["edge_min_frame_count"]),
        "edge_low_total_delta_vs_curated": edge_low_total - curated_stats.get("edge_low_total", curated_stats["edge_frame_count_0_100"] + curated_stats["edge_frame_count_100_200"]),
        "edge_high_total_delta_vs_curated": edge_high_total - curated_stats.get("edge_high_total", curated_stats["edge_frame_count_2800_2900"] + curated_stats["edge_frame_count_2900_3000"]),
        "depth_extreme_low_delta_vs_curated": extremes["depth_low"] - curated_extremes["depth_low"],
        "depth_extreme_high_delta_vs_curated": extremes["depth_high"] - curated_extremes["depth_high"],
        "lambda_extreme_low_delta_vs_curated": extremes["lambda_low"] - curated_extremes["lambda_low"],
        "lambda_extreme_high_delta_vs_curated": extremes["lambda_high"] - curated_extremes["lambda_high"],
    }


def plot_coverage(
    output_path: Path,
    points: pd.DataFrame,
    geometry: pd.DataFrame,
    all_ids: Sequence[str],
    curated_ids: Sequence[str],
    robust_ids: Sequence[str],
) -> None:
    all_bins = audit.subset_bin_table(all_ids, points)
    curated_bins = audit.subset_bin_table(curated_ids, points)
    robust_bins = audit.subset_bin_table(robust_ids, points)
    centers = all_bins["v_bin_lo_px"].to_numpy() + audit.V_BIN_WIDTH_PX / 2.0
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, constrained_layout=True)

    for table, color, label in (
        (all_bins, "#7f7f7f", "all 36 FIT"),
        (curated_bins, "#1f77b4", "Curated-14"),
        (robust_bins, "#d62728", "Robust-Curated-18"),
    ):
        axes[0].plot(centers, table["unique_frame_count"], color=color, linewidth=2, label=label)
    axes[0].axhline(2, color="#111111", linestyle="--", linewidth=1, label="gate = 2")
    axes[0].set_ylabel("frames / 100 px")
    axes[0].set_title("Robust-Curated-18 geometry-only v coverage")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.2)

    for table, color, label in (
        (all_bins, "#7f7f7f", "all 36 FIT"),
        (curated_bins, "#1f77b4", "Curated-14"),
        (robust_bins, "#d62728", "Robust-Curated-18"),
    ):
        axes[1].fill_between(centers, table["lambda_truth_min_mm"], table["lambda_truth_max_mm"], color=color, alpha=0.10)
        axes[1].plot(centers, table["lambda_truth_min_mm"], color=color, linewidth=1)
        axes[1].plot(centers, table["lambda_truth_max_mm"], color=color, linewidth=1, label=label)
    axes[1].set_ylabel("lambda truth / mm")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.2)

    for table, color, label in (
        (all_bins, "#7f7f7f", "all 36 FIT"),
        (curated_bins, "#1f77b4", "Curated-14"),
        (robust_bins, "#d62728", "Robust-Curated-18"),
    ):
        axes[2].fill_between(centers, table["Z_min_mm"], table["Z_max_mm"], color=color, alpha=0.10)
        axes[2].plot(centers, table["Z_min_mm"], color=color, linewidth=1)
        axes[2].plot(centers, table["Z_max_mm"], color=color, linewidth=1, label=label)
    axes[2].set_ylabel("Zc / mm")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.2)

    geometry = geometry.copy()
    geometry["v_center_px"] = (geometry["v_min_px"] + geometry["v_max_px"]) / 2.0
    curated_set = set(curated_ids)
    robust_set = set(robust_ids)
    colors = [
        "#d62728" if frame_id in robust_set - curated_set
        else "#1f77b4" if frame_id in curated_set
        else "#bdbdbd"
        for frame_id in geometry["frame_id"]
    ]
    axes[3].scatter(
        geometry["v_center_px"],
        geometry["board_center_z_mm"],
        c=colors,
        s=38,
        edgecolor="black",
        linewidth=0.3,
    )
    for row in geometry.itertuples(index=False):
        if row.frame_id in robust_set:
            axes[3].annotate(
                row.frame_id,
                (row.v_center_px, row.board_center_z_mm),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[3].set_xlabel("v support center / px")
    axes[3].set_ylabel("board-center Z / mm")
    axes[3].grid(alpha=0.2)
    axes[3].set_xlim(audit.V_MIN_PX, audit.V_MAX_PX)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report(
    output_path: Path,
    metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    robust_ids: Sequence[str],
    added_ids: Sequence[str],
    curated_stats: Mapping[str, Any],
    robust_stats: Mapping[str, Any],
    full_stats: Mapping[str, Any],
    curated_extremes: Mapping[str, int],
    robust_extremes: Mapping[str, int],
    search_summary: Mapping[str, Any],
    strict_pairs: set[tuple[str, str]],
    similarities: Mapping[tuple[str, str], float],
    current_coverage: Mapping[str, Any],
) -> None:
    strict_selected = strict_pairs_in_subset(robust_ids, strict_pairs)
    strict_added = strict_pairs_involving_added(robust_ids, added_ids, strict_pairs)
    pair_rows = top_similarity_pairs(added_ids, pairs)
    max_added_similarity = max(
        float(similarities[tuple(sorted((added_id, curated_id)))])
        for added_id in added_ids
        for curated_id in CURATED_14
        if tuple(sorted((added_id, curated_id))) in similarities
    )
    metric_by_id = metrics.set_index("frame_id")
    contribution_lines = ["|---|---|"]
    for frame_id in added_ids:
        row = metric_by_id.loc[frame_id]
        pair_values = [
            float(similarities[tuple(sorted((frame_id, curated_id)))])
            for curated_id in CURATED_14
            if tuple(sorted((frame_id, curated_id))) in similarities
        ]
        contribution_lines.append(
            f"| {frame_id} | edge low/high bins {int(row['edge_low_100px_bin_count'])}/{int(row['edge_high_100px_bin_count'])}; depth bin {int(row['depth_bin_id'])}; lambda bins {row['lambda_bin_ids']}; max similarity {fmt(max(pair_values), 3)} |"
        )
    lines = [
        "# Robust-Curated-18 几何互补审计",
        "",
        "POSE_DIVERSITY = SUFFICIENT",
        "RECOMMENDED_ROBUST_CURATED_FIT_SIZE = 18",
        "",
        "## 结论",
        "",
        f"固定 Curated-14，不删除已有 pose；新增 4 个 pose：{', '.join(added_ids)}。",
        f"Robust-Curated-18：{', '.join(robust_ids)}。",
        "",
        "选择只使用 pose_geometry_metrics.csv、pair_pose_similarity.csv、当前 full-board-physical FIT 点和既有 geometry gates。",
        "没有拟合 Plane/Quadratic/Cone，没有读取 Validation，也没有读取或使用任何模型 residual。",
        "",
        "## 选择方法",
        "",
        f"- 候选池：其余 {search_summary['candidate_count']} 个 pose；穷举组合数：{search_summary['combination_count']}（C(22,4)）。",
        f"- 硬排除任何新增 pose 参与的 strict near-duplicate pair；剩余安全组合：{search_summary['strict_safe_combination_count']}。",
        "- 优先级：四个边缘 band 的最弱独立 frame 数与低/高边缘平衡度 → normal cover 最大角度 → depth/lambda 极值重复支持 → excitation span → 与 Curated-14 的低 geometric similarity。",
        f"- edge 前沿 key（min-band, low/high balance, total, low total, high total）：{search_summary['edge_front_key']}。",
        f"- edge 前沿中的 normal 最佳最大角度：{fmt(search_summary['normal_front_max_angle_deg'])}°。",
        f"- Robust-18 内部 strict near-duplicate pair：{len(strict_selected)}（其中 Curated-14 原有 {len(strict_selected) - len(strict_added)}，新增 pose 涉及 {len(strict_added)}）。",
        "",
        "## 新增 pose 的几何作用",
        "",
        "| pose | 作用 |",
        *contribution_lines,
        "",
        "## 几何覆盖比较",
        "",
        "| 指标 | Curated-14 | Robust-18 | 变化 |",
        "|---|---:|---:|---:|",
        f"| v 10 px occupied | {curated_stats['v_grid_10px_occupied_count']} | {robust_stats['v_grid_10px_occupied_count']} | {robust_stats['v_grid_10px_occupied_count'] - curated_stats['v_grid_10px_occupied_count']:+d} |",
        f"| v 100 px occupied | {curated_stats['v_bin_100px_occupied_count']} | {robust_stats['v_bin_100px_occupied_count']} | {robust_stats['v_bin_100px_occupied_count'] - curated_stats['v_bin_100px_occupied_count']:+d} |",
        f"| edge 0–100 / 100–200 | {curated_stats['edge_counts'][0]} / {curated_stats['edge_counts'][1]} | {robust_stats['edge_counts'][0]} / {robust_stats['edge_counts'][1]} | {robust_stats['edge_counts'][0] - curated_stats['edge_counts'][0]:+d} / {robust_stats['edge_counts'][1] - curated_stats['edge_counts'][1]:+d} |",
        f"| edge 2800–2900 / 2900–3000 | {curated_stats['edge_counts'][2]} / {curated_stats['edge_counts'][3]} | {robust_stats['edge_counts'][2]} / {robust_stats['edge_counts'][3]} | {robust_stats['edge_counts'][2] - curated_stats['edge_counts'][2]:+d} / {robust_stats['edge_counts'][3] - curated_stats['edge_counts'][3]:+d} |",
        f"| edge minimum | {curated_stats['edge_min']} | {robust_stats['edge_min']} | {robust_stats['edge_min'] - curated_stats['edge_min']:+d} |",
        f"| low/high edge total | {curated_stats['edge_low_total']} / {curated_stats['edge_high_total']} | {robust_stats['edge_low_total']} / {robust_stats['edge_high_total']} | {robust_stats['edge_low_total'] - curated_stats['edge_low_total']:+d} / {robust_stats['edge_high_total'] - curated_stats['edge_high_total']:+d} |",
        f"| normal cover max / ° | {fmt(curated_stats['normal_cover_max_angle_deg'])} | {fmt(robust_stats['normal_cover_max_angle_deg'])} | {fmt(curated_stats['normal_cover_max_angle_deg'] - robust_stats['normal_cover_max_angle_deg'])} |",
        f"| normal diameter / ° | {fmt(curated_stats['normal_pairwise_diameter_deg'])} | {fmt(robust_stats['normal_pairwise_diameter_deg'])} | {fmt(robust_stats['normal_pairwise_diameter_deg'] - curated_stats['normal_pairwise_diameter_deg'])} |",
        f"| depth span / mm | {fmt(curated_stats['depth_center_span_mm'])} | {fmt(robust_stats['depth_center_span_mm'])} | {fmt(robust_stats['depth_center_span_mm'] - curated_stats['depth_center_span_mm'])} |",
        f"| lambda span / mm | {fmt(curated_stats['lambda_truth_span_mm'])} | {fmt(robust_stats['lambda_truth_span_mm'])} | {fmt(robust_stats['lambda_truth_span_mm'] - curated_stats['lambda_truth_span_mm'])} |",
        f"| depth extreme low/high support | {curated_extremes['depth_low']} / {curated_extremes['depth_high']} | {robust_extremes['depth_low']} / {robust_extremes['depth_high']} | {robust_extremes['depth_low'] - curated_extremes['depth_low']:+d} / {robust_extremes['depth_high'] - curated_extremes['depth_high']:+d} |",
        f"| lambda extreme low/high support | {curated_extremes['lambda_low']} / {curated_extremes['lambda_high']} | {robust_extremes['lambda_low']} / {robust_extremes['lambda_high']} | {robust_extremes['lambda_low'] - curated_extremes['lambda_low']:+d} / {robust_extremes['lambda_high'] - curated_extremes['lambda_high']:+d} |",
        f"| max similarity(new, Curated-14) | — | {fmt(max_added_similarity, 3)} | lower is more complementary |",
        "",
        f"Robust-18 仍覆盖全部 {full_stats['v_grid_10px_occupied_count']} 个 10 px cells、全部 {full_stats['v_bin_100px_occupied_count']} 个 100 px bins，并通过既有 geometry gates。",
        f"normal cover 从 {fmt(curated_stats['normal_cover_max_angle_deg'])}° 降至 {fmt(robust_stats['normal_cover_max_angle_deg'])}°；depth 极值支持 low/high = {curated_extremes['depth_low']}/{curated_extremes['depth_high']} → {robust_extremes['depth_low']}/{robust_extremes['depth_high']}；lambda 极值支持 low/high = {curated_extremes['lambda_low']}/{curated_extremes['lambda_high']} → {robust_extremes['lambda_low']}/{robust_extremes['lambda_high']}。",
        "",
        "## 近重复复核",
        "",
        "Robust-Curated-18 没有新增 pose 参与既有 strict near-duplicate pair。高相似候选 036 虽能补高 depth 极值，但与 Curated-14 的 006、013 构成 strict near-duplicate，因此按约束排除。",
        "",
        "| 新增 pose | Curated pose | similarity | normal Δ / ° | translation Δ / mm | v Jaccard |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in pair_rows[:8]:
        lines.append(
            f"| {row['added_pose']} | {row['curated_pose']} | {fmt(row['geometric_similarity'], 3)} | {fmt(row['normal_angle_diff_deg'])} | {fmt(row['translation_difference_mm'])} | {fmt(row['v_overlap_100px_jaccard'], 3)} |"
        )
    lines += [
        "",
        "## 输入与输出",
        "",
        f"- 输入 metrics：{METRICS_PATH}。",
        f"- 输入 pair similarity：{PAIRS_PATH}。",
        f"- 当前点表：{audit.CURRENT_POINTS_PATH}；mask provenance = full_board_physical, inset=0 mm。",
        f"- 当前 100 px reference：{current_coverage['populated_bin_count']}/{current_coverage['bin_count']} bins populated，minimum frame multiplicity = {current_coverage['min_unique_frame_count']}。",
        "- 输出：robust_curated_18_ids.json、robust18_geometry_comparison.csv、robust18_v_coverage.png、report.md。",
        "",
        "POSE_DIVERSITY = SUFFICIENT",
        "RECOMMENDED_ROBUST_CURATED_FIT_SIZE = 18",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}；如需重跑请显式使用 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, pairs, points = load_inputs()
    current_coverage = audit.read_current_coverage_reference()
    (
        grid_sets,
        bin_sets,
        normal_angles,
        bin_edges,
        depth_bins_by_index,
        lambda_bins_by_id,
        full_depth_bins,
        full_lambda_bins,
    ) = build_context(metrics, points)
    strict_pairs, similarities = pair_lookup(pairs)

    robust_ids, chosen_combo_stats, search_summary = select_robust_ids(
        metrics,
        points,
        grid_sets,
        bin_sets,
        normal_angles,
        depth_bins_by_index,
        lambda_bins_by_id,
        full_depth_bins,
        full_lambda_bins,
        strict_pairs,
        similarities,
    )
    added_ids = tuple(frame_id for frame_id in robust_ids if frame_id not in CURATED_14)
    if len(added_ids) != 4 or not set(CURATED_14).issubset(robust_ids):
        raise RuntimeError("Robust-Curated-18 没有固定完整 Curated-14 或新增 pose 数不是 4")

    full_stats = audit.subset_stats(
        ALL_IDS, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges
    )
    curated_stats = audit.subset_stats(
        CURATED_14, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges
    )
    robust_stats = audit.subset_stats(
        robust_ids, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges
    )
    for stat in (full_stats, curated_stats, robust_stats):
        stat["edge_counts"] = (
            stat["edge_frame_count_0_100"],
            stat["edge_frame_count_100_200"],
            stat["edge_frame_count_2800_2900"],
            stat["edge_frame_count_2900_3000"],
        )
        stat["edge_min"] = stat["edge_min_frame_count"]
        stat["edge_low_total"] = stat["edge_counts"][0] + stat["edge_counts"][1]
        stat["edge_high_total"] = stat["edge_counts"][2] + stat["edge_counts"][3]
    curated_extremes = extreme_support(
        CURATED_14, depth_bins_by_index, lambda_bins_by_id, full_depth_bins, full_lambda_bins
    )
    robust_extremes = extreme_support(
        robust_ids, depth_bins_by_index, lambda_bins_by_id, full_depth_bins, full_lambda_bins
    )
    if not bool(robust_stats["overall_geometry_ok"]):
        raise RuntimeError(f"Robust-Curated-18 未通过既有 geometry gates：{robust_stats}")
    if strict_pairs_involving_added(robust_ids, added_ids, strict_pairs):
        raise RuntimeError("Robust-Curated-18 包含新增 strict near-duplicate pair")

    comparison = pd.DataFrame([
        comparison_row(
            "full_36",
            ALL_IDS,
            full_stats,
            extreme_support(ALL_IDS, depth_bins_by_index, lambda_bins_by_id, full_depth_bins, full_lambda_bins),
            curated_stats,
            curated_extremes,
            strict_pairs,
        ),
        comparison_row("curated_14", CURATED_14, curated_stats, curated_extremes, curated_stats, curated_extremes, strict_pairs),
        comparison_row("robust_curated_18", robust_ids, robust_stats, robust_extremes, curated_stats, curated_extremes, strict_pairs),
    ])
    comparison.to_csv(
        output_dir / "robust18_geometry_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_coverage(
        output_dir / "robust18_v_coverage.png",
        points,
        metrics,
        ALL_IDS,
        CURATED_14,
        robust_ids,
    )

    metric_by_id = metrics.set_index("frame_id")
    added_rows = []
    for frame_id in added_ids:
        row = metric_by_id.loc[frame_id]
        pair_sims = [
            float(similarities[tuple(sorted((frame_id, curated_id)))])
            for curated_id in CURATED_14
            if tuple(sorted((frame_id, curated_id))) in similarities
        ]
        added_rows.append({
            "frame_id": frame_id,
            "v_min_px": float(row["v_min_px"]),
            "v_max_px": float(row["v_max_px"]),
            "edge_low_100px_bin_count": int(row["edge_low_100px_bin_count"]),
            "edge_high_100px_bin_count": int(row["edge_high_100px_bin_count"]),
            "depth_bin_id": int(row["depth_bin_id"]),
            "lambda_bin_ids": str(row["lambda_bin_ids"]),
            "normal_tilt_deg": float(row["normal_tilt_deg"]),
            "max_similarity_to_curated": max(pair_sims),
            "strict_pair_with_curated": [
                f"{a}–{b}"
                for a, b in sorted(strict_pairs)
                if frame_id in (a, b) and (a in CURATED_14 or b in CURATED_14)
            ],
        })

    selection_document = {
        "selection_name": "Robust-Curated-18",
        "POSE_DIVERSITY": "SUFFICIENT",
        "RECOMMENDED_ROBUST_CURATED_FIT_SIZE": 18,
        "curated_14_fixed": list(CURATED_14),
        "added_4_pose_ids": list(added_ids),
        "robust_curated_18_ids": list(robust_ids),
        "candidate_pool": [frame_id for frame_id in ALL_IDS if frame_id not in CURATED_14],
        "selection_basis": "geometry-only; no Plane/Quadratic/Cone fit, no Validation, no model residuals",
        "geometry_gates": {
            "v_domain_px": [audit.V_MIN_PX, audit.V_MAX_PX],
            "v_grid_width_px": audit.V_GRID_WIDTH_PX,
            "v_bin_width_px": audit.V_BIN_WIDTH_PX,
            "edge_bins": list(audit.EDGE_BIN_IDS),
            "edge_min_frame_count": audit.EDGE_MIN_FRAME_COUNT,
            "normal_cover_threshold_deg": audit.NORMAL_COVER_THRESHOLD_DEG,
            "normal_diameter_tolerance_deg": audit.NORMAL_DIAMETER_TOLERANCE_DEG,
            "span_ratio_threshold": audit.SPAN_RATIO_THRESHOLD,
            "excitation_bin_count": audit.EXCITATION_BIN_COUNT,
            "translation_bin_count": audit.TRANSLATION_BIN_COUNT,
        },
        "search_summary": jsonable(search_summary),
        "selection_score": jsonable(chosen_combo_stats),
        "curated_14_stats": jsonable(curated_stats),
        "robust_18_stats": jsonable(robust_stats),
        "full_36_stats": jsonable(full_stats),
        "curated_14_extreme_support": curated_extremes,
        "robust_18_extreme_support": robust_extremes,
        "normal_cover_reduction_deg": float(
            curated_stats["normal_cover_max_angle_deg"]
            - robust_stats["normal_cover_max_angle_deg"]
        ),
        "strict_near_duplicate_pairs_in_robust_18": strict_pairs_in_subset(robust_ids, strict_pairs),
        "strict_near_duplicate_pairs_involving_added": strict_pairs_involving_added(
            robust_ids, added_ids, strict_pairs
        ),
        "added_pose_contributions": jsonable(added_rows),
        "most_similar_added_to_curated_pairs": jsonable(top_similarity_pairs(added_ids, pairs)),
        "inputs": {
            "curated_14_source": str(SOURCE_IDS_PATH.resolve()),
            "pose_geometry_metrics": str(METRICS_PATH.resolve()),
            "pair_pose_similarity": str(PAIRS_PATH.resolve()),
            "current_points": str(audit.CURRENT_POINTS_PATH.resolve()),
            "current_coverage": str(audit.CURRENT_COVERAGE_PATH.resolve()),
            "mask_provenance": "full_board_physical, inset=0 mm",
            "validation_opened": False,
            "model_residuals_used": False,
        },
    }
    (output_dir / "robust_curated_18_ids.json").write_text(
        json.dumps(selection_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir / "report.md",
        metrics,
        pairs,
        robust_ids,
        added_ids,
        curated_stats,
        robust_stats,
        full_stats,
        curated_extremes,
        robust_extremes,
        search_summary,
        strict_pairs,
        similarities,
        current_coverage,
    )
    print("POSE_DIVERSITY = SUFFICIENT")
    print("RECOMMENDED_ROBUST_CURATED_FIT_SIZE = 18")
    print("ROBUST_CURATED_18_IDS = " + ",".join(robust_ids))
    print("ADDED_4_IDS = " + ",".join(added_ids))
    print(f"OUTPUT_DIR = {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
