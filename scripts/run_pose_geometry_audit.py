#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit FIT pose geometry and select a geometry-only representative subset.

This audit intentionally uses only:

* PnP pose geometry from the existing geometry-only FIT table plus a direct
  solve for the six 049--054 poses absent from that table;
* the existing full-board-physical-mask FIT point table;
* v support and ``lambda_truth`` / Z ranges derived from that point table.

No Plane/Quadratic/Cone fit, model residual, or Validation input is read.
The output is written to a new audit directory and never overwrites source FIT
data or the existing 0817 audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except Exception:  # pragma: no cover - the normal environment has SciPy.
    Bounds = LinearConstraint = milp = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_laser_models_from_triplets as triplets  # noqa: E402


FIT_GROUPS: tuple[tuple[str, Path, tuple[int, ...]], ...] = (
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
FIT_IDS = tuple(f"{value:03d}" for _, _, ids in FIT_GROUPS for value in ids)
HISTORICAL_IDS = tuple(f"{value:03d}" for value in range(1, 19))

INTRINSICS_PATH = ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
CURRENT_AUDIT_DIR = ROOT / "projects" / "daheng" / "outputs" / "0817" / "full_fit_v_coverage_audit"
CURRENT_POINTS_PATH = CURRENT_AUDIT_DIR / "full_fit_points.csv"
CURRENT_COVERAGE_PATH = CURRENT_AUDIT_DIR / "full_fit_v_coverage.csv"
CURRENT_REPORT_PATH = CURRENT_AUDIT_DIR / "report.md"
EXISTING_GEOMETRY_PATH = ROOT / "projects" / "daheng" / "outputs" / "0814" / "board_coordinate_audit" / "frame_board_geometry_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "projects" / "daheng" / "outputs" / "0818" / "pose_geometry_audit"

V_MIN_PX = 0.0
V_MAX_PX = 3000.0
V_GRID_WIDTH_PX = 10.0
V_BIN_WIDTH_PX = 100.0
V_GRID_COUNT = int((V_MAX_PX - V_MIN_PX) / V_GRID_WIDTH_PX)
V_BIN_COUNT = int((V_MAX_PX - V_MIN_PX) / V_BIN_WIDTH_PX)
EDGE_BIN_IDS = (0, 1, 28, 29)

# Geometry-only gates.  These are deliberately declared in one place and are
# reproduced in report.md/curated_fit_ids.json.
NORMAL_COVER_THRESHOLD_DEG = 5.0
NORMAL_DIAMETER_TOLERANCE_DEG = 1.0
SPAN_RATIO_THRESHOLD = 0.95
EDGE_MIN_FRAME_COUNT = 2
EXCITATION_BIN_COUNT = 8
TRANSLATION_BIN_COUNT = 3
MILP_TIME_LIMIT_SECONDS = 30.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def normalize_frame_id(value: Any) -> str:
    try:
        return f"{int(float(value)):03d}"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析 frame_id={value!r}") from exc


def finite_array(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1.0e-12 else 1.0


def interval_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> tuple[float, float]:
    intersection = max(0.0, min(float(hi_a), float(hi_b)) - max(float(lo_a), float(lo_b)))
    union = max(float(hi_a), float(hi_b)) - min(float(lo_a), float(lo_b))
    return intersection, union


def imread_unicode(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"无法读取 FIT chess 图像：{path}")
    return image


def inventory_fit_chess() -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for group_name, root, ids in FIT_GROUPS:
        for image_id in ids:
            frame_id = f"{image_id:03d}"
            path = root / f"chess {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append((frame_id, group_name, path))
    actual = tuple(frame_id for frame_id, _, _ in records)
    if actual != FIT_IDS:
        raise RuntimeError(f"FIT chess inventory mismatch: expected={FIT_IDS}, actual={actual}")
    return records


def load_existing_fit_geometry() -> pd.DataFrame:
    """Load only geometry columns from the existing 30-pose FIT PnP table.

    The source CSV also contains residual-analysis columns.  ``usecols`` is
    intentional: none of those columns are read into this audit.
    """
    usecols = [
        "row_type",
        "frame_id",
        "source_dataset",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "board_origin_x_mm",
        "board_origin_y_mm",
        "board_origin_z_mm",
        "board_center_xc_mm",
        "board_center_yc_mm",
        "board_center_zc_mm",
        "plane_nx",
        "plane_ny",
        "plane_nz",
    ]
    table = pd.read_csv(EXISTING_GEOMETRY_PATH, usecols=usecols)
    table = table[table["row_type"].eq("frame")].copy()
    table["frame_id"] = table["frame_id"].map(normalize_frame_id)
    table = table[table["frame_id"].isin(FIT_IDS)].copy()
    rows: list[dict[str, Any]] = []
    for row in table.to_dict("records"):
        frame_id = str(row["frame_id"])
        normal = np.asarray([row["plane_nx"], row["plane_ny"], row["plane_nz"]], dtype=float)
        normal /= max(float(np.linalg.norm(normal)), np.finfo(float).eps)
        tvec = np.asarray(
            [row["board_origin_x_mm"], row["board_origin_y_mm"], row["board_origin_z_mm"]], dtype=float
        )
        center = np.asarray(
            [row["board_center_xc_mm"], row["board_center_yc_mm"], row["board_center_zc_mm"]], dtype=float
        )
        rows.append(
            {
                "frame_id": frame_id,
                "fit_group": "fit_001_018" if int(frame_id) <= 18 else "fit_025_036",
                "source_chess_path": "existing geometry table",
                "rvec_x": float(row["rvec_x"]),
                "rvec_y": float(row["rvec_y"]),
                "rvec_z": float(row["rvec_z"]),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "translation_norm_mm": float(np.linalg.norm(tvec)),
                "board_center_x_mm": float(center[0]),
                "board_center_y_mm": float(center[1]),
                "board_center_z_mm": float(center[2]),
                "camera_board_distance_mm": float(np.linalg.norm(center)),
                "board_normal_x": float(normal[0]),
                "board_normal_y": float(normal[1]),
                "board_normal_z": float(normal[2]),
                "board_plane_distance_mm": float(-normal @ tvec),
                "board_tilt_deg": math.degrees(math.acos(float(np.clip(-normal[2], -1.0, 1.0)))),
            }
        )
    return pd.DataFrame(rows)


def solve_missing_fit_poses(missing_ids: Sequence[str]) -> pd.DataFrame:
    """Solve only explicit FIT IDs absent from the existing geometry table."""
    records = [record for record in inventory_fit_chess() if record[0] in set(missing_ids)]
    k, distortion, _ = triplets.load_intrinsics(INTRINSICS_PATH)
    rows: list[dict[str, Any]] = []
    center_object = np.asarray([100.0, 70.0, 0.0], dtype=np.float64)

    for frame_id, group_name, path in records:
        image = imread_unicode(path)
        # The RMSE is computed internally by the existing PnP helper as a
        # sanity check, but it is not returned or used as a selection feature.
        pose = triplets.detect_board_pose(
            image,
            k,
            distortion,
            cols=11,
            rows=8,
            square_size_mm=20.0,
            max_rmse_px=float("inf"),
        )
        rotation, _ = cv2.Rodrigues(np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1))
        normal = np.asarray(pose.normal, dtype=np.float64).reshape(3)
        normal /= max(float(np.linalg.norm(normal)), np.finfo(float).eps)
        tvec = np.asarray(pose.tvec, dtype=np.float64).reshape(3)
        board_center = rotation @ center_object + tvec
        board_tilt = math.degrees(math.acos(float(np.clip(-normal[2], -1.0, 1.0))))
        rows.append(
            {
                "frame_id": frame_id,
                "fit_group": group_name,
                "source_chess_path": str(path.resolve()),
                "rvec_x": float(pose.rvec[0]),
                "rvec_y": float(pose.rvec[1]),
                "rvec_z": float(pose.rvec[2]),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "translation_norm_mm": float(np.linalg.norm(tvec)),
                "board_center_x_mm": float(board_center[0]),
                "board_center_y_mm": float(board_center[1]),
                "board_center_z_mm": float(board_center[2]),
                "camera_board_distance_mm": float(np.linalg.norm(board_center)),
                "board_normal_x": float(normal[0]),
                "board_normal_y": float(normal[1]),
                "board_normal_z": float(normal[2]),
                "board_plane_distance_mm": float(pose.d),
                "board_tilt_deg": float(board_tilt),
            }
        )

    return pd.DataFrame(rows)


def solve_fit_poses() -> pd.DataFrame:
    """Assemble all 36 explicit FIT PnP poses from geometry-only inputs."""
    existing = load_existing_fit_geometry()
    missing = [frame_id for frame_id in FIT_IDS if frame_id not in set(existing["frame_id"])]
    solved = solve_missing_fit_poses(missing) if missing else pd.DataFrame()
    result = pd.concat([existing, solved], ignore_index=True)
    result = result.set_index("frame_id").loc[list(FIT_IDS)].reset_index()
    if result["frame_id"].tolist() != list(FIT_IDS):
        raise RuntimeError("PnP geometry result is not ordered as the explicit FIT set")
    return result


def load_current_points() -> pd.DataFrame:
    usecols = ["frame_id", "v_px", "lambda_truth_mm", "Zc_mm", "in_v_domain"]
    points = pd.read_csv(CURRENT_POINTS_PATH, usecols=usecols)
    points["frame_id"] = points["frame_id"].map(normalize_frame_id)
    if set(points["frame_id"]) != set(FIT_IDS):
        raise RuntimeError(
            f"current FIT point table does not contain exactly 36 FIT IDs: {sorted(set(points['frame_id']))}"
        )
    points = points[points["in_v_domain"].astype(bool)].copy()
    for column in ("v_px", "lambda_truth_mm", "Zc_mm"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points = points.dropna(subset=["v_px", "lambda_truth_mm", "Zc_mm"])
    points["v_grid_id"] = np.floor((points["v_px"] - V_MIN_PX) / V_GRID_WIDTH_PX).astype(int)
    points["v_grid_id"] = points["v_grid_id"].clip(0, V_GRID_COUNT - 1)
    points["v_bin_id"] = np.floor((points["v_px"] - V_MIN_PX) / V_BIN_WIDTH_PX).astype(int)
    points["v_bin_id"] = points["v_bin_id"].clip(0, V_BIN_COUNT - 1)
    return points


def read_current_coverage_reference() -> dict[str, Any]:
    """Read only current 100 px coverage aggregates for an input sanity check."""
    usecols = ["v_bin", "unique_frame_count", "lambda_truth_span_mm", "Z_span_mm"]
    coverage = pd.read_csv(CURRENT_COVERAGE_PATH, usecols=usecols)
    return {
        "bin_count": int(len(coverage)),
        "populated_bin_count": int(np.count_nonzero(pd.to_numeric(coverage["unique_frame_count"], errors="coerce") > 0)),
        "min_unique_frame_count": int(pd.to_numeric(coverage["unique_frame_count"], errors="coerce").min()),
        "lambda_truth_span_min_mm": float(pd.to_numeric(coverage["lambda_truth_span_mm"], errors="coerce").min()),
        "lambda_truth_span_max_mm": float(pd.to_numeric(coverage["lambda_truth_span_mm"], errors="coerce").max()),
        "Z_span_min_mm": float(pd.to_numeric(coverage["Z_span_mm"], errors="coerce").min()),
        "Z_span_max_mm": float(pd.to_numeric(coverage["Z_span_mm"], errors="coerce").max()),
    }


def support_sets(points: pd.DataFrame) -> tuple[dict[str, set[int]], dict[str, set[int]], dict[str, tuple[float, float]]]:
    grid: dict[str, set[int]] = {}
    bins: dict[str, set[int]] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for frame_id, group in points.groupby("frame_id", sort=False):
        grid[frame_id] = set(group["v_grid_id"].astype(int))
        bins[frame_id] = set(group["v_bin_id"].astype(int))
        ranges[frame_id] = (float(group["v_px"].min()), float(group["v_px"].max()))
    return grid, bins, ranges


def percentile_bin_edges(values: Sequence[float], count: int) -> np.ndarray:
    finite = finite_array(values)
    if finite.size == 0:
        return np.linspace(0.0, 1.0, count + 1)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo <= 1.0e-12:
        return np.linspace(lo - 0.5, hi + 0.5, count + 1)
    return np.linspace(lo, hi, count + 1)


def bin_index(value: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, float(value), side="right") - 1, 0, len(edges) - 2))


def normal_angle_matrix(metrics: pd.DataFrame) -> np.ndarray:
    normals = metrics[["board_normal_x", "board_normal_y", "board_normal_z"]].to_numpy(dtype=float)
    dots = np.clip(normals @ normals.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def build_pose_metrics(geometry: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[int]], dict[str, set[int]], dict[str, tuple[float, float]], np.ndarray, dict[str, np.ndarray]]:
    grid_sets, bin_sets, v_ranges = support_sets(points)
    normal_angles = normal_angle_matrix(geometry)
    geometry = geometry.copy()
    geometry["normal_tilt_deg"] = geometry["board_tilt_deg"]

    lambda_bins = percentile_bin_edges(points["lambda_truth_mm"].to_numpy(), EXCITATION_BIN_COUNT)
    depth_bins = percentile_bin_edges(geometry["board_center_z_mm"].to_numpy(), EXCITATION_BIN_COUNT)
    tx_bins = percentile_bin_edges(geometry["board_center_x_mm"].to_numpy(), TRANSLATION_BIN_COUNT)
    ty_bins = percentile_bin_edges(geometry["board_center_y_mm"].to_numpy(), TRANSLATION_BIN_COUNT)
    bin_edges = {"lambda": lambda_bins, "depth": depth_bins, "tx": tx_bins, "ty": ty_bins}

    rows: list[dict[str, Any]] = []
    for index, row in geometry.iterrows():
        frame_id = str(row["frame_id"])
        group = points[points["frame_id"] == frame_id]
        lam = group["lambda_truth_mm"].to_numpy(dtype=float)
        z = group["Zc_mm"].to_numpy(dtype=float)
        v_min, v_max = v_ranges[frame_id]
        low_edge_bins = sorted(set(bin_sets[frame_id]) & {0, 1})
        high_edge_bins = sorted(set(bin_sets[frame_id]) & {28, 29})
        rows.append(
            {
                **row.to_dict(),
                "point_count": int(len(group)),
                "v_min_px": v_min,
                "v_max_px": v_max,
                "v_span_px": float(v_max - v_min),
                "v_support_10px_bin_count": int(len(grid_sets[frame_id])),
                "v_support_100px_bin_count": int(len(bin_sets[frame_id])),
                "edge_low_100px_bin_count": int(len(low_edge_bins)),
                "edge_high_100px_bin_count": int(len(high_edge_bins)),
                "edge_low_bin_ids": ";".join(str(value) for value in low_edge_bins),
                "edge_high_bin_ids": ";".join(str(value) for value in high_edge_bins),
                "lambda_truth_min_mm": float(np.min(lam)),
                "lambda_truth_max_mm": float(np.max(lam)),
                "lambda_truth_span_mm": float(np.ptp(lam)),
                "lambda_truth_median_mm": float(np.median(lam)),
                "Z_min_mm": float(np.min(z)),
                "Z_max_mm": float(np.max(z)),
                "Z_span_mm": float(np.ptp(z)),
                "Z_median_mm": float(np.median(z)),
                "lambda_bin_ids": ";".join(str(value) for value in sorted({bin_index(value, lambda_bins) for value in lam})),
                "depth_bin_id": bin_index(float(row["board_center_z_mm"]), depth_bins),
                "translation_x_bin_id": bin_index(float(row["board_center_x_mm"]), tx_bins),
                "translation_y_bin_id": bin_index(float(row["board_center_y_mm"]), ty_bins),
                "nearest_normal_angle_to_other_deg": float(np.min(np.delete(normal_angles[index], index))) if len(geometry) > 1 else np.nan,
                "max_normal_angle_to_other_deg": float(np.max(np.delete(normal_angles[index], index))) if len(geometry) > 1 else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    return result, grid_sets, bin_sets, v_ranges, normal_angles, bin_edges


def pair_similarity(metrics: pd.DataFrame, grid_sets: Mapping[str, set[int]], bin_sets: Mapping[str, set[int]], v_ranges: Mapping[str, tuple[float, float]], normal_angles: np.ndarray) -> pd.DataFrame:
    ids = metrics["frame_id"].tolist()
    by_id = metrics.set_index("frame_id")
    rows: list[dict[str, Any]] = []
    for i, frame_a in enumerate(ids):
        a = by_id.loc[frame_a]
        for j in range(i + 1, len(ids)):
            frame_b = ids[j]
            b = by_id.loc[frame_b]
            shared_grid = len(grid_sets[frame_a] & grid_sets[frame_b])
            union_grid = len(grid_sets[frame_a] | grid_sets[frame_b])
            shared_bins = len(bin_sets[frame_a] & bin_sets[frame_b])
            union_bins = len(bin_sets[frame_a] | bin_sets[frame_b])
            v_intersection, v_union = interval_overlap(
                v_ranges[frame_a][0], v_ranges[frame_a][1], v_ranges[frame_b][0], v_ranges[frame_b][1]
            )
            lambda_intersection, lambda_union = interval_overlap(
                a["lambda_truth_min_mm"], a["lambda_truth_max_mm"], b["lambda_truth_min_mm"], b["lambda_truth_max_mm"]
            )
            normal_delta = float(normal_angles[i, j])
            center_delta = float(
                np.linalg.norm(
                    by_id.loc[frame_a, ["board_center_x_mm", "board_center_y_mm", "board_center_z_mm"]].to_numpy(dtype=float)
                    - by_id.loc[frame_b, ["board_center_x_mm", "board_center_y_mm", "board_center_z_mm"]].to_numpy(dtype=float)
                )
            )
            tvec_delta = float(
                np.linalg.norm(
                    by_id.loc[frame_a, ["tvec_x_mm", "tvec_y_mm", "tvec_z_mm"]].to_numpy(dtype=float)
                    - by_id.loc[frame_b, ["tvec_x_mm", "tvec_y_mm", "tvec_z_mm"]].to_numpy(dtype=float)
                )
            )
            depth_delta = abs(float(a["board_center_z_mm"]) - float(b["board_center_z_mm"]))
            distance_delta = abs(float(a["camera_board_distance_mm"]) - float(b["camera_board_distance_mm"]))
            v_jaccard = safe_ratio(shared_bins, union_bins)
            grid_jaccard = safe_ratio(shared_grid, union_grid)
            lambda_span_delta = abs(float(a["lambda_truth_span_mm"]) - float(b["lambda_truth_span_mm"]))
            # Equal-weight, dimensionless geometry distance.  Scales are
            # declared constants, not estimated from residuals.
            terms = np.asarray(
                [
                    normal_delta / 5.0,
                    center_delta / 50.0,
                    depth_delta / 15.0,
                    (1.0 - v_jaccard) / 0.5,
                    lambda_span_delta / 5.0,
                ],
                dtype=float,
            )
            geo_distance = float(np.sqrt(np.mean(terms * terms)))
            similarity = float(np.exp(-geo_distance))
            strict = bool(
                normal_delta <= 3.0
                and center_delta <= 45.0
                and depth_delta <= 10.0
                and v_jaccard >= 0.75
                and lambda_span_delta <= 6.0
            )
            candidate = bool(
                normal_delta <= 5.0
                and center_delta <= 75.0
                and depth_delta <= 15.0
                and v_jaccard >= 0.65
                and lambda_span_delta <= 12.0
            )
            rows.append(
                {
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "normal_angle_diff_deg": normal_delta,
                    "translation_difference_mm": center_delta,
                    "tvec_difference_mm": tvec_delta,
                    "depth_difference_mm": depth_delta,
                    "camera_board_distance_difference_mm": distance_delta,
                    "v_overlap_100px_jaccard": v_jaccard,
                    "v_overlap_10px_jaccard": grid_jaccard,
                    "v_interval_overlap_px": v_intersection,
                    "v_interval_union_px": v_union,
                    "v_interval_iou": safe_ratio(v_intersection, v_union),
                    "lambda_interval_overlap_mm": lambda_intersection,
                    "lambda_interval_iou": safe_ratio(lambda_intersection, lambda_union),
                    "lambda_span_difference_mm": lambda_span_delta,
                    "geometric_distance": geo_distance,
                    "geometric_similarity": similarity,
                    "near_duplicate_strict": strict,
                    "near_duplicate_candidate": candidate,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["near_duplicate_strict", "geometric_similarity"], ascending=[False, False]
    ).reset_index(drop=True)
    result["similarity_rank"] = np.arange(1, len(result) + 1)
    return result


def normal_cover_stats(selected_ids: Sequence[str], all_metrics: pd.DataFrame, normal_angles: np.ndarray) -> dict[str, Any]:
    ids = all_metrics["frame_id"].tolist()
    selected_indices = [ids.index(frame_id) for frame_id in selected_ids]
    selected_indices = sorted(set(selected_indices))
    if not selected_indices:
        return {
            "normal_cover_max_angle_deg": math.inf,
            "normal_cover_p95_angle_deg": math.inf,
            "normal_pairwise_diameter_deg": 0.0,
            "normal_cover_missing_pose_count": len(ids),
            "normal_cover_missing_pose_ids": ";".join(ids),
        }
    nearest = np.min(normal_angles[:, selected_indices], axis=1)
    selected_matrix = normal_angles[np.ix_(selected_indices, selected_indices)]
    diameter = float(np.max(selected_matrix)) if len(selected_indices) > 1 else 0.0
    missing = [ids[index] for index, angle in enumerate(nearest) if angle > NORMAL_COVER_THRESHOLD_DEG]
    return {
        "normal_cover_max_angle_deg": float(np.max(nearest)),
        "normal_cover_p95_angle_deg": float(np.percentile(nearest, 95)),
        "normal_pairwise_diameter_deg": diameter,
        "normal_cover_missing_pose_count": int(len(missing)),
        "normal_cover_missing_pose_ids": ";".join(missing),
    }


def subset_stats(
    selected_ids: Sequence[str],
    all_metrics: pd.DataFrame,
    points: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    bin_edges: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    selected = [str(value) for value in selected_ids]
    selected_set = set(selected)
    selected_points = points[points["frame_id"].isin(selected_set)]
    selected_metrics = all_metrics[all_metrics["frame_id"].isin(selected_set)]
    grid_union: set[int] = set()
    bin_union: set[int] = set()
    for frame_id in selected:
        grid_union.update(grid_sets[frame_id])
        bin_union.update(bin_sets[frame_id])
    missing_grid = sorted(set(range(V_GRID_COUNT)) - grid_union)
    missing_bins = sorted(set(range(V_BIN_COUNT)) - bin_union)
    edge_counts = {
        str(bin_id): int(sum(bin_id in bin_sets[frame_id] for frame_id in selected))
        for bin_id in EDGE_BIN_IDS
    }
    edge_min = min(edge_counts.values()) if edge_counts else 0

    full_metrics = all_metrics
    full_points = points
    full_depth_range = float(full_metrics["board_center_z_mm"].max() - full_metrics["board_center_z_mm"].min())
    selected_depth_range = float(selected_metrics["board_center_z_mm"].max() - selected_metrics["board_center_z_mm"].min()) if not selected_metrics.empty else 0.0
    full_lambda_range = float(full_points["lambda_truth_mm"].max() - full_points["lambda_truth_mm"].min())
    selected_lambda_range = float(selected_points["lambda_truth_mm"].max() - selected_points["lambda_truth_mm"].min()) if not selected_points.empty else 0.0

    def covered_value_bins(values: Sequence[float], edges: np.ndarray) -> set[int]:
        return {bin_index(float(value), edges) for value in values}

    full_depth_bins = covered_value_bins(full_metrics["board_center_z_mm"].to_numpy(), bin_edges["depth"])
    selected_depth_bins = covered_value_bins(selected_metrics["board_center_z_mm"].to_numpy(), bin_edges["depth"])
    full_lambda_bins = covered_value_bins(full_points["lambda_truth_mm"].to_numpy(), bin_edges["lambda"])
    selected_lambda_bins = covered_value_bins(selected_points["lambda_truth_mm"].to_numpy(), bin_edges["lambda"])
    full_tx_bins = covered_value_bins(full_metrics["board_center_x_mm"].to_numpy(), bin_edges["tx"])
    selected_tx_bins = covered_value_bins(selected_metrics["board_center_x_mm"].to_numpy(), bin_edges["tx"])
    full_ty_bins = covered_value_bins(full_metrics["board_center_y_mm"].to_numpy(), bin_edges["ty"])
    selected_ty_bins = covered_value_bins(selected_metrics["board_center_y_mm"].to_numpy(), bin_edges["ty"])

    normal_stats = normal_cover_stats(selected, all_metrics, normal_angles)
    full_normal_diameter = float(np.max(normal_angles))
    translation_ok = selected_tx_bins == full_tx_bins and selected_ty_bins == full_ty_bins
    excitation_ok = bool(
        safe_ratio(selected_depth_range, full_depth_range) >= SPAN_RATIO_THRESHOLD
        and safe_ratio(selected_lambda_range, full_lambda_range) >= SPAN_RATIO_THRESHOLD
        and selected_depth_bins == full_depth_bins
        and selected_lambda_bins == full_lambda_bins
    )
    normal_ok = bool(
        normal_stats["normal_cover_max_angle_deg"] <= NORMAL_COVER_THRESHOLD_DEG
        and normal_stats["normal_pairwise_diameter_deg"] >= full_normal_diameter - NORMAL_DIAMETER_TOLERANCE_DEG
    )
    v_ok = len(missing_grid) == 0 and len(missing_bins) == 0
    edge_ok = edge_min >= EDGE_MIN_FRAME_COUNT
    result: dict[str, Any] = {
        "selected_pose_count": int(len(selected)),
        "selected_pose_ids": ";".join(selected),
        "v_grid_10px_occupied_count": int(len(grid_union)),
        "v_grid_10px_missing_count": int(len(missing_grid)),
        "v_grid_10px_missing_ids": ";".join(str(value) for value in missing_grid[:100]),
        "v_bin_100px_occupied_count": int(len(bin_union)),
        "v_bin_100px_missing_count": int(len(missing_bins)),
        "v_bin_100px_missing_ids": ";".join(str(value) for value in missing_bins),
        "v_continuous_ok": v_ok,
        "edge_min_frame_count": int(edge_min),
        "edge_frame_count_0_100": int(edge_counts.get("0", 0)),
        "edge_frame_count_100_200": int(edge_counts.get("1", 0)),
        "edge_frame_count_2800_2900": int(edge_counts.get("28", 0)),
        "edge_frame_count_2900_3000": int(edge_counts.get("29", 0)),
        "edge_multiframe_ok": edge_ok,
        "depth_center_min_mm": float(selected_metrics["board_center_z_mm"].min()) if not selected_metrics.empty else math.nan,
        "depth_center_max_mm": float(selected_metrics["board_center_z_mm"].max()) if not selected_metrics.empty else math.nan,
        "depth_center_span_mm": selected_depth_range,
        "depth_span_ratio": safe_ratio(selected_depth_range, full_depth_range),
        "lambda_truth_min_mm": float(selected_points["lambda_truth_mm"].min()) if not selected_points.empty else math.nan,
        "lambda_truth_max_mm": float(selected_points["lambda_truth_mm"].max()) if not selected_points.empty else math.nan,
        "lambda_truth_span_mm": selected_lambda_range,
        "lambda_span_ratio": safe_ratio(selected_lambda_range, full_lambda_range),
        "depth_bin_occupied_count": int(len(selected_depth_bins)),
        "depth_bin_missing_count": int(len(full_depth_bins - selected_depth_bins)),
        "lambda_bin_occupied_count": int(len(selected_lambda_bins)),
        "lambda_bin_missing_count": int(len(full_lambda_bins - selected_lambda_bins)),
        "depth_lambda_excitation_ok": excitation_ok,
        "translation_x_bin_occupied_count": int(len(selected_tx_bins)),
        "translation_y_bin_occupied_count": int(len(selected_ty_bins)),
        "translation_coverage_ok": translation_ok,
        "full_normal_pairwise_diameter_deg": full_normal_diameter,
        **normal_stats,
        "normal_angle_diversity_ok": normal_ok,
    }
    result["overall_geometry_ok"] = bool(
        result["v_continuous_ok"]
        and result["edge_multiframe_ok"]
        and result["depth_lambda_excitation_ok"]
        and result["normal_angle_diversity_ok"]
        and result["translation_coverage_ok"]
    )
    return result


def pose_has_bins(points: pd.DataFrame, ids: Sequence[str], column: str, edges: np.ndarray) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for frame_id in ids:
        values = points.loc[points["frame_id"].eq(frame_id), column].to_numpy(dtype=float)
        result[frame_id] = {bin_index(value, edges) for value in values}
    return result


def minimal_support_sets(support_sets_to_reduce: Sequence[set[str]]) -> list[set[str]]:
    """Remove dominated set-cover rows without changing feasibility.

    For a >=1 (or >=2) coverage constraint, a row with support set T is
    redundant when another row has support set S⊂T: satisfying S also
    satisfies T.  This keeps the MILP small while subset_stats still checks
    every 10 px cell exactly after solving.
    """
    unique = {frozenset(values) for values in support_sets_to_reduce if values}
    kept: list[frozenset[str]] = []
    for candidate in sorted(unique, key=lambda value: (len(value), tuple(sorted(value)))):
        if not any(existing < candidate for existing in kept):
            kept.append(candidate)
    return [set(values) for values in kept]


def constraint_matrix(
    ids: Sequence[str],
    points: pd.DataFrame,
    geometry: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    bin_edges: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {frame_id: i for i, frame_id in enumerate(ids)}
    n = len(ids)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_at_least(values: Sequence[float], bound: float) -> None:
        rows.append(np.asarray(values, dtype=float))
        lower.append(float(bound))
        upper.append(np.inf)

    grid_support = [
        {frame_id for frame_id in ids if grid_id in grid_sets[frame_id]} for grid_id in range(V_GRID_COUNT)
    ]
    for support in minimal_support_sets(grid_support):
        add_at_least([1.0 if frame_id in support else 0.0 for frame_id in ids], 1.0)
    for bin_id in EDGE_BIN_IDS:
        add_at_least([1.0 if bin_id in bin_sets[frame_id] else 0.0 for frame_id in ids], float(EDGE_MIN_FRAME_COUNT))

    geometry_by_id = geometry.set_index("frame_id")
    lambda_pose_bins = pose_has_bins(points, ids, "lambda_truth_mm", bin_edges["lambda"])
    full_lambda_bins = set().union(*(lambda_pose_bins[frame_id] for frame_id in ids)) if ids else set()
    for excitation_id in sorted(full_lambda_bins):
        add_at_least([1.0 if excitation_id in lambda_pose_bins[frame_id] else 0.0 for frame_id in ids], 1.0)
    depth_pose_bins = {
        frame_id: {bin_index(float(geometry_by_id.loc[frame_id, "board_center_z_mm"]), bin_edges["depth"])}
        for frame_id in ids
    }
    full_depth_bins = set().union(*(depth_pose_bins[frame_id] for frame_id in ids)) if ids else set()
    for excitation_id in sorted(full_depth_bins):
        add_at_least([1.0 if excitation_id in depth_pose_bins[frame_id] else 0.0 for frame_id in ids], 1.0)

    tx_pose_bins = {
        frame_id: {bin_index(float(geometry_by_id.loc[frame_id, "board_center_x_mm"]), bin_edges["tx"])}
        for frame_id in ids
    }
    ty_pose_bins = {
        frame_id: {bin_index(float(geometry_by_id.loc[frame_id, "board_center_y_mm"]), bin_edges["ty"])}
        for frame_id in ids
    }
    for translation_bins in (tx_pose_bins, ty_pose_bins):
        for bin_id in range(TRANSLATION_BIN_COUNT):
            add_at_least([1.0 if bin_id in translation_bins[frame_id] else 0.0 for frame_id in ids], 1.0)

    full_ids = geometry["frame_id"].tolist()
    normal_support = [
        {
            frame_id
            for frame_id in ids
            if normal_angles[full_index, index[frame_id]] <= NORMAL_COVER_THRESHOLD_DEG
        }
        for full_index in range(len(full_ids))
    ]
    for support in minimal_support_sets(normal_support):
        add_at_least([1.0 if frame_id in support else 0.0 for frame_id in ids], 1.0)

    return np.vstack(rows), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def greedy_curated_ids(
    ids: Sequence[str],
    all_metrics: pd.DataFrame,
    points: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    bin_edges: Mapping[str, np.ndarray],
) -> tuple[list[str], bool]:
    selected: list[str] = []
    remaining = list(ids)
    by_id = all_metrics.set_index("frame_id")
    id_to_index = {frame_id: index for index, frame_id in enumerate(ids)}
    lambda_pose_bins = pose_has_bins(points, ids, "lambda_truth_mm", bin_edges["lambda"])
    depth_pose_bins = {
        frame_id: {bin_index(float(by_id.loc[frame_id, "board_center_z_mm"]), bin_edges["depth"])} for frame_id in ids
    }
    tx_pose_bins = {
        frame_id: {bin_index(float(by_id.loc[frame_id, "board_center_x_mm"]), bin_edges["tx"])} for frame_id in ids
    }
    ty_pose_bins = {
        frame_id: {bin_index(float(by_id.loc[frame_id, "board_center_y_mm"]), bin_edges["ty"])} for frame_id in ids
    }
    all_grid = set(range(V_GRID_COUNT))
    all_bins = set(range(V_BIN_COUNT))
    all_lambda_excitation = set().union(*(lambda_pose_bins[frame_id] for frame_id in ids)) if ids else set()
    all_depth_excitation = set().union(*(depth_pose_bins[frame_id] for frame_id in ids)) if ids else set()
    all_translation = set(range(TRANSLATION_BIN_COUNT))
    all_normals = set(range(len(ids)))
    covered_grid: set[int] = set()
    covered_bins: set[int] = set()
    covered_lambda: set[int] = set()
    covered_depth: set[int] = set()
    covered_tx: set[int] = set()
    covered_ty: set[int] = set()
    covered_normals: set[int] = set()
    edge_counts = {edge_id: 0 for edge_id in EDGE_BIN_IDS}

    def still_missing() -> bool:
        return bool(
            all_grid - covered_grid
            or all_bins - covered_bins
            or any(edge_counts[edge_id] < EDGE_MIN_FRAME_COUNT for edge_id in EDGE_BIN_IDS)
            or all_lambda_excitation - covered_lambda
            or all_depth_excitation - covered_depth
            or all_translation - covered_tx
            or all_translation - covered_ty
            or all_normals - covered_normals
        )

    while remaining and still_missing():
        scored: list[tuple[float, float, str]] = []
        for candidate in remaining:
            candidate_index = id_to_index[candidate]
            normal_gain = len(
                {index for index in all_normals if index not in covered_normals and normal_angles[index, candidate_index] <= NORMAL_COVER_THRESHOLD_DEG}
            )
            grid_gain = len(grid_sets[candidate] - covered_grid)
            bin_gain = len(bin_sets[candidate] - covered_bins)
            edge_gain = sum(
                1 for edge_id in EDGE_BIN_IDS if edge_counts[edge_id] < EDGE_MIN_FRAME_COUNT and edge_id in bin_sets[candidate]
            )
            lambda_gain = len(lambda_pose_bins[candidate] - covered_lambda)
            depth_gain = len(depth_pose_bins[candidate] - covered_depth)
            tx_gain = len(tx_pose_bins[candidate] - covered_tx)
            ty_gain = len(ty_pose_bins[candidate] - covered_ty)
            score = (
                8.0 * grid_gain
                + 2.0 * bin_gain
                + 25.0 * edge_gain
                + 5.0 * lambda_gain
                + 5.0 * depth_gain
                + 2.0 * (tx_gain + ty_gain)
                + 8.0 * normal_gain
            )
            novelty = float(np.mean(1.0 - np.exp(-normal_angles[candidate_index] / 5.0)))
            scored.append((score, novelty, candidate))
        scored.sort(key=lambda value: (value[0], value[1], -int(value[2])), reverse=True)
        chosen = scored[0][2]
        selected.append(chosen)
        remaining.remove(chosen)
        covered_grid.update(grid_sets[chosen])
        covered_bins.update(bin_sets[chosen])
        covered_lambda.update(lambda_pose_bins[chosen])
        covered_depth.update(depth_pose_bins[chosen])
        covered_tx.update(tx_pose_bins[chosen])
        covered_ty.update(ty_pose_bins[chosen])
        chosen_index = id_to_index[chosen]
        covered_normals.update(
            {index for index in all_normals if normal_angles[index, chosen_index] <= NORMAL_COVER_THRESHOLD_DEG}
        )
        for edge_id in EDGE_BIN_IDS:
            if edge_id in bin_sets[chosen]:
                edge_counts[edge_id] += 1
    # The set-cover loop handles discrete support.  Add the few continuous
    # geometry anchors that cannot be expressed as a simple >=1 row: depth /
    # lambda span ends and the maximum normal-angle pair.
    stats = subset_stats(selected, all_metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    anchors: set[str] = set()
    if stats["depth_span_ratio"] < SPAN_RATIO_THRESHOLD:
        anchors.update(
            all_metrics.loc[
                all_metrics["board_center_z_mm"].isin(
                    [all_metrics["board_center_z_mm"].min(), all_metrics["board_center_z_mm"].max()]
                ),
                "frame_id",
            ].tolist()
        )
    if stats["lambda_span_ratio"] < SPAN_RATIO_THRESHOLD:
        grouped_lambda = points.groupby("frame_id")["lambda_truth_mm"].agg(["min", "max"])
        anchors.update(grouped_lambda.index[grouped_lambda["min"].eq(grouped_lambda["min"].min())].tolist())
        anchors.update(grouped_lambda.index[grouped_lambda["max"].eq(grouped_lambda["max"].max())].tolist())
    if stats["normal_pairwise_diameter_deg"] < stats["full_normal_pairwise_diameter_deg"] - NORMAL_DIAMETER_TOLERANCE_DEG:
        max_pair = np.unravel_index(int(np.argmax(normal_angles)), normal_angles.shape)
        anchors.update([ids[max_pair[0]], ids[max_pair[1]]])
    for anchor in ids:
        if anchor in anchors and anchor not in selected:
            selected.append(anchor)
    stats = subset_stats(selected, all_metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    return selected, bool(stats["overall_geometry_ok"])


def solve_curated_ids(
    ids: Sequence[str],
    all_metrics: pd.DataFrame,
    points: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    bin_edges: Mapping[str, np.ndarray],
) -> tuple[list[str], bool, str]:
    selected, greedy_ok = greedy_curated_ids(ids, all_metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    if not greedy_ok:
        return selected, False, "greedy_failed_geometry_gates"

    # Deterministic local pruning is cheap and guarantees that no selected
    # pose can be removed one at a time while retaining every declared gate.
    changed = True
    while changed:
        changed = False
        for candidate in list(selected):
            trial = [frame_id for frame_id in selected if frame_id != candidate]
            trial_stats = subset_stats(trial, all_metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
            if trial_stats["overall_geometry_ok"]:
                selected = trial
                changed = True
                break

    if milp is None or LinearConstraint is None or Bounds is None or len(selected) <= 1:
        return selected, False, "greedy_local_prune"

    # Exact lower-bound search.  The base rows cover discrete support.  The
    # four extra rows below linearize the two span-ratio gates.  The normal
    # diameter gate is handled by enumerating the finite set of normal pairs
    # that can satisfy it, fixing one pair in each MILP solve.  This avoids a
    # false lower bound from a relaxed solution that has the wrong extremes.
    A, lb, ub = constraint_matrix(ids, points, all_metrics, grid_sets, bin_sets, normal_angles, bin_edges)
    n = len(ids)
    geometry_by_id = all_metrics.set_index("frame_id")
    extra_rows: list[np.ndarray] = []
    extra_lb: list[float] = []
    extra_ub: list[float] = []

    def add_extra_support(support: set[str]) -> None:
        if support:
            extra_rows.append(np.asarray([1.0 if frame_id in support else 0.0 for frame_id in ids], dtype=float))
            extra_lb.append(1.0)
            extra_ub.append(np.inf)

    full_depth_min = float(all_metrics["board_center_z_mm"].min())
    full_depth_max = float(all_metrics["board_center_z_mm"].max())
    depth_margin = (full_depth_max - full_depth_min) * (1.0 - SPAN_RATIO_THRESHOLD)
    add_extra_support(set(geometry_by_id.index[geometry_by_id["board_center_z_mm"] <= full_depth_min + depth_margin + 1.0e-9]))
    add_extra_support(set(geometry_by_id.index[geometry_by_id["board_center_z_mm"] >= full_depth_max - depth_margin - 1.0e-9]))

    lambda_by_pose = points.groupby("frame_id")["lambda_truth_mm"].agg(["min", "max"])
    full_lambda_min = float(points["lambda_truth_mm"].min())
    full_lambda_max = float(points["lambda_truth_mm"].max())
    lambda_margin = (full_lambda_max - full_lambda_min) * (1.0 - SPAN_RATIO_THRESHOLD)
    add_extra_support(set(lambda_by_pose.index[lambda_by_pose["min"] <= full_lambda_min + lambda_margin + 1.0e-9]))
    add_extra_support(set(lambda_by_pose.index[lambda_by_pose["max"] >= full_lambda_max - lambda_margin - 1.0e-9]))

    full_diameter = float(np.max(normal_angles))
    wide_pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if normal_angles[i, j] >= full_diameter - NORMAL_DIAMETER_TOLERANCE_DEG - 1.0e-9
    ]
    if not wide_pairs:
        return selected, False, "greedy_local_prune+no_normal_diameter_pair"

    base_A = np.vstack([A, np.asarray(extra_rows, dtype=float)]) if extra_rows else A
    base_lb = np.concatenate([lb, np.asarray(extra_lb, dtype=float)]) if extra_rows else lb
    base_ub = np.concatenate([ub, np.asarray(extra_ub, dtype=float)]) if extra_rows else ub
    inconclusive = False
    current = selected
    for target_size in range(len(current) - 1, 1, -1):
        found: list[str] | None = None
        for i, j in wide_pairs:
            lower_bounds = np.zeros(n, dtype=float)
            lower_bounds[i] = 1.0
            lower_bounds[j] = 1.0
            size_row = np.ones((1, n), dtype=float)
            solve = milp(
                c=np.zeros(n, dtype=float),
                integrality=np.ones(n, dtype=int),
                bounds=Bounds(lower_bounds, np.ones(n)),
                constraints=[
                    LinearConstraint(base_A, base_lb, base_ub),
                    LinearConstraint(size_row, -np.inf, float(target_size)),
                ],
                options={"time_limit": MILP_TIME_LIMIT_SECONDS},
            )
            status = int(getattr(solve, "status", -1))
            if not solve.success and status != 2:
                inconclusive = True
                continue
            if solve.success and solve.x is not None:
                candidate = [frame_id for frame_id, value in zip(ids, solve.x) if float(value) > 0.5]
                candidate.sort(key=lambda value: ids.index(value))
                candidate_stats = subset_stats(candidate, all_metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
                if candidate_stats["overall_geometry_ok"]:
                    found = candidate
                    break
        if found is None:
            if inconclusive:
                return current, False, "greedy_local_prune+exact_pair_search_inconclusive"
            return current, True, "greedy_local_prune+exact_pair_search_proven"
        current = found
    return current, True, "exact_pair_search_proven"


def attach_loo_and_selection(
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    grid_sets: Mapping[str, set[int]],
    bin_sets: Mapping[str, set[int]],
    normal_angles: np.ndarray,
    bin_edges: Mapping[str, np.ndarray],
    curated_ids: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    all_ids = metrics["frame_id"].tolist()
    full_stats = subset_stats(all_ids, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    loo_by_id: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for frame_id in all_ids:
        selected = [value for value in all_ids if value != frame_id]
        stat = subset_stats(selected, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
        loo_by_id[frame_id] = stat
        row = metrics.loc[metrics["frame_id"].eq(frame_id)].iloc[0].to_dict()
        for key, value in stat.items():
            if key in {"selected_pose_ids", "v_grid_10px_missing_ids", "v_bin_100px_missing_ids", "normal_cover_missing_pose_ids"}:
                continue
            row[f"loo_{key}"] = value
        row["loo_v_coverage_loss"] = not bool(stat["v_continuous_ok"])
        row["loo_edge_multiframe_loss"] = not bool(stat["edge_multiframe_ok"])
        row["loo_depth_lambda_loss"] = not bool(stat["depth_lambda_excitation_ok"])
        row["loo_normal_diversity_loss"] = not bool(stat["normal_angle_diversity_ok"])
        row["loo_translation_coverage_loss"] = not bool(stat["translation_coverage_ok"])
        row["loo_overall_geometry_loss"] = not bool(stat["overall_geometry_ok"])
        row["curated_selected"] = frame_id in set(curated_ids)
        rows.append(row)
    result = pd.DataFrame(rows)
    result["loo_normal_missing_pose_ids"] = [loo_by_id[frame_id]["normal_cover_missing_pose_ids"] for frame_id in all_ids]
    result["loo_v_missing_grid_count"] = [loo_by_id[frame_id]["v_grid_10px_missing_count"] for frame_id in all_ids]
    result["loo_edge_min_frame_count"] = [loo_by_id[frame_id]["edge_min_frame_count"] for frame_id in all_ids]
    return result, full_stats, loo_by_id


def subset_bin_table(selected_ids: Sequence[str], points: pd.DataFrame) -> pd.DataFrame:
    selected_set = set(selected_ids)
    rows: list[dict[str, Any]] = []
    for bin_id in range(V_BIN_COUNT):
        group = points[points["v_bin_id"].eq(bin_id)]
        selected_group = group[group["frame_id"].isin(selected_set)]

        def span(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
            if frame.empty:
                return math.nan, math.nan, math.nan
            lo = float(frame[column].min())
            hi = float(frame[column].max())
            return lo, hi, hi - lo

        lam_lo, lam_hi, lam_span = span(selected_group, "lambda_truth_mm")
        z_lo, z_hi, z_span = span(selected_group, "Zc_mm")
        rows.append(
            {
                "v_bin_id": bin_id,
                "v_bin_lo_px": float(bin_id * V_BIN_WIDTH_PX),
                "v_bin_hi_px": float((bin_id + 1) * V_BIN_WIDTH_PX),
                "point_count": int(len(selected_group)),
                "unique_frame_count": int(selected_group["frame_id"].nunique()),
                "frame_ids": ";".join(sorted(selected_group["frame_id"].unique())),
                "lambda_truth_min_mm": lam_lo,
                "lambda_truth_max_mm": lam_hi,
                "lambda_truth_span_mm": lam_span,
                "Z_min_mm": z_lo,
                "Z_max_mm": z_hi,
                "Z_span_mm": z_span,
            }
        )
    return pd.DataFrame(rows)


def plot_similarity(output_path: Path, ids: Sequence[str], pairs: pd.DataFrame, curated_ids: Sequence[str]) -> None:
    matrix = np.eye(len(ids), dtype=float)
    index = {frame_id: i for i, frame_id in enumerate(ids)}
    for row in pairs.itertuples(index=False):
        matrix[index[row.frame_a], index[row.frame_b]] = float(row.geometric_similarity)
        matrix[index[row.frame_b], index[row.frame_a]] = float(row.geometric_similarity)
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="geometric similarity (1 = closest)")
    ax.set_xticks(np.arange(len(ids)), ids, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(ids)), ids, fontsize=6)
    curated = set(curated_ids)
    for label, frame_id in zip(ax.get_xticklabels(), ids):
        if frame_id in curated:
            label.set_color("#d62728")
            label.set_fontweight("bold")
    for label, frame_id in zip(ax.get_yticklabels(), ids):
        if frame_id in curated:
            label.set_color("#d62728")
            label.set_fontweight("bold")
    ax.set_title("FIT pose pairwise geometric similarity\nred labels = curated representative set")
    ax.set_xlabel("pose B")
    ax.set_ylabel("pose A")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_curated_coverage(output_path: Path, points: pd.DataFrame, all_ids: Sequence[str], curated_ids: Sequence[str], geometry: pd.DataFrame) -> None:
    full_bins = subset_bin_table(all_ids, points)
    selected_bins = subset_bin_table(curated_ids, points)
    centers = full_bins["v_bin_lo_px"].to_numpy() + V_BIN_WIDTH_PX / 2.0
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, constrained_layout=True)

    axes[0].plot(centers, full_bins["unique_frame_count"], color="#7f7f7f", linewidth=2, label="all 36 FIT")
    axes[0].plot(centers, selected_bins["unique_frame_count"], color="#1f77b4", linewidth=2, label=f"curated {len(curated_ids)}")
    axes[0].axhline(EDGE_MIN_FRAME_COUNT, color="#d62728", linestyle="--", linewidth=1, label="edge multi-frame threshold")
    axes[0].set_ylabel("frames / 100 px")
    axes[0].set_title("Curated FIT v coverage and geometric excitation")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.2)

    for table, color, label in (
        (full_bins, "#7f7f7f", "all 36 FIT"),
        (selected_bins, "#2ca02c", f"curated {len(curated_ids)}"),
    ):
        axes[1].fill_between(centers, table["lambda_truth_min_mm"], table["lambda_truth_max_mm"], color=color, alpha=0.18)
        axes[1].plot(centers, table["lambda_truth_min_mm"], color=color, linewidth=1)
        axes[1].plot(centers, table["lambda_truth_max_mm"], color=color, linewidth=1, label=label)
    axes[1].set_ylabel("lambda truth / mm")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.2)

    for table, color, label in (
        (full_bins, "#9467bd", "all 36 FIT"),
        (selected_bins, "#ff7f0e", f"curated {len(curated_ids)}"),
    ):
        axes[2].fill_between(centers, table["Z_min_mm"], table["Z_max_mm"], color=color, alpha=0.18)
        axes[2].plot(centers, table["Z_min_mm"], color=color, linewidth=1)
        axes[2].plot(centers, table["Z_max_mm"], color=color, linewidth=1, label=label)
    axes[2].set_ylabel("Zc / mm")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.2)

    geometry = geometry.copy()
    geometry["v_center_px"] = (geometry["v_min_px"] + geometry["v_max_px"]) / 2.0
    colors = np.where(geometry["frame_id"].isin(set(curated_ids)), "#d62728", "#bdbdbd")
    axes[3].scatter(geometry["v_center_px"], geometry["board_center_z_mm"], c=colors, s=38, edgecolor="black", linewidth=0.3)
    for row in geometry.itertuples(index=False):
        if row.frame_id in set(curated_ids):
            axes[3].annotate(row.frame_id, (row.v_center_px, row.board_center_z_mm), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[3].set_xlabel("v support center / px")
    axes[3].set_ylabel("board-center Z / mm")
    axes[3].grid(alpha=0.2)
    axes[3].set_xlim(V_MIN_PX, V_MAX_PX)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_report(
    output_path: Path,
    geometry: pd.DataFrame,
    pairs: pd.DataFrame,
    full_stats: Mapping[str, Any],
    curated_stats: Mapping[str, Any],
    historical_stats: Mapping[str, Any],
    loo_by_id: Mapping[str, Mapping[str, Any]],
    curated_ids: Sequence[str],
    minimality_proven: bool,
    solver: str,
    current_coverage: Mapping[str, Any],
) -> str:
    diversity = "SUFFICIENT" if curated_stats["overall_geometry_ok"] else "PARTIAL"
    strict_pairs = pairs[pairs["near_duplicate_strict"]].sort_values("geometric_similarity", ascending=False)
    candidate_pairs = pairs[pairs["near_duplicate_candidate"]].sort_values("geometric_similarity", ascending=False)
    loo_rows = []
    for frame_id in geometry["frame_id"].tolist():
        stat = loo_by_id[frame_id]
        losses = [
            name
            for name, key in (
                ("v", "v_continuous_ok"),
                ("edge", "edge_multiframe_ok"),
                ("depth/lambda", "depth_lambda_excitation_ok"),
                ("normal", "normal_angle_diversity_ok"),
                ("translation", "translation_coverage_ok"),
            )
            if not stat[key]
        ]
        if losses:
            loo_rows.append((frame_id, ", ".join(losses)))

    lines = [
        "# FIT pose geometric observability audit",
        "",
        f"`POSE_DIVERSITY = {diversity}`",
        f"`RECOMMENDED_CURATED_FIT_SIZE = {len(curated_ids)}`",
        "",
        "## Scope and guardrails",
        "",
        "- FIT only: 001–018, 025–036, 049–054 (36 poses).",
        "- PnP R/t uses geometry columns only: existing FIT 001–018/025–036 pose records plus direct `SOLVEPNP_ITERATIVE + solvePnPRefineLM` for the six 049–054 chess images absent from that table.",
        f"- Current point input: `{CURRENT_POINTS_PATH}`; current 100 px aggregate: `{CURRENT_COVERAGE_PATH}`.",
        "- The point table is the current `full_board_physical` mask result (inset=0 mm); v support is recomputed from its retained points.",
        "- No Validation image/file was opened. No Plane/Quadratic/Cone was fitted. No model residual was read or used for ranking, deletion, or selection.",
        "",
        "## Geometry-only selection gates",
        "",
        f"- v continuity: every 10 px cell in [0, 3000) occupied, with all 100 px bins populated.",
        f"- Edge support: each of 0–100, 100–200, 2800–2900, 2900–3000 px has at least {EDGE_MIN_FRAME_COUNT} selected poses.",
        f"- Depth/lambda excitation: all occupied full-range depth bins ({full_stats['depth_bin_occupied_count']}) and lambda bins ({full_stats['lambda_bin_occupied_count']}) represented; both spans retain ≥{SPAN_RATIO_THRESHOLD:.0%} of full FIT.",
        f"- Normal diversity: every full-pose normal is within {NORMAL_COVER_THRESHOLD_DEG:g}° of a selected normal and selected normal diameter is within {NORMAL_DIAMETER_TOLERANCE_DEG:g}° of full.",
        f"- Translation diversity: all {TRANSLATION_BIN_COUNT} equal-width board-center X and Y bins represented.",
        "",
        "## Full FIT geometry range",
        "",
        f"- Board-center Z: {fmt(full_stats['depth_center_min_mm'])}–{fmt(full_stats['depth_center_max_mm'])} mm; span {fmt(full_stats['depth_center_span_mm'])} mm.",
        f"- Camera-board distance: {fmt(geometry['camera_board_distance_mm'].min())}–{fmt(geometry['camera_board_distance_mm'].max())} mm.",
        f"- Translation norm: {fmt(geometry['translation_norm_mm'].min())}–{fmt(geometry['translation_norm_mm'].max())} mm.",
        f"- Board tilt: {fmt(geometry['board_tilt_deg'].min())}–{fmt(geometry['board_tilt_deg'].max())}°.",
        f"- Lambda truth: {fmt(full_stats['lambda_truth_min_mm'])}–{fmt(full_stats['lambda_truth_max_mm'])} mm; span {fmt(full_stats['lambda_truth_span_mm'])} mm.",
        f"- Current 0817 100 px reference: {current_coverage['populated_bin_count']}/{current_coverage['bin_count']} populated bins; minimum frame multiplicity {current_coverage['min_unique_frame_count']}.",
        "",
        "## Geometric near duplicates",
        "",
        "Pairwise distance uses normal angle, board-center translation difference, board-center Z difference, 100 px v-support Jaccard, and lambda-span difference. The strict flag is: angle≤3°, translation≤45 mm, depth≤10 mm, v Jaccard≥0.75, lambda-span difference≤6 mm.",
        "",
        "| pair | normal Δ / ° | translation Δ / mm | depth Δ / mm | v Jaccard | lambda-span Δ / mm | similarity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strict_pairs.head(15).itertuples(index=False):
        lines.append(
            f"| {row.frame_a}–{row.frame_b} | {fmt(row.normal_angle_diff_deg)} | {fmt(row.translation_difference_mm)} | {fmt(row.depth_difference_mm)} | {fmt(row.v_overlap_100px_jaccard, 3)} | {fmt(row.lambda_span_difference_mm)} | {fmt(row.geometric_similarity, 3)} |"
        )
    if strict_pairs.empty:
        lines.append("| none under the declared strict gate | | | | | | |")
    lines += [
        "",
        f"Strict near-duplicate pairs: **{len(strict_pairs)}**; broader geometry candidates: **{len(candidate_pairs)}**. Similar geometry does not imply identical v support: the pair CSV retains both 10 px and 100 px support overlap.",
        "",
        "## Leave-one-pose check",
        "",
        f"All 36 leave-one-out cases were evaluated against the gates above. Cases with at least one geometric loss: **{len(loo_rows)}**.",
        "",
        "| deleted pose | lost gate(s) |",
        "|---|---|",
    ]
    if loo_rows:
        lines.extend(f"| {frame_id} | {losses} |" for frame_id, losses in loo_rows)
    else:
        lines.append("| none | none; every single deletion retained the declared full-set gates |")
    lines += [
        "",
        "The per-pose CSV contains the complete LOO metrics, including missing v cells, edge multiplicity, span ratios, normal nearest-cover angle, and translation-bin retention.",
        "",
        "## Curated recommendation",
        "",
        f"- Solver: `{solver}`; minimum-size proof for the declared linear geometry gates: **{minimality_proven}**.",
        f"- Recommended IDs ({len(curated_ids)}): **{', '.join(curated_ids)}**.",
        "",
        "| criterion | full 36 | curated | pass |",
        "|---|---:|---:|:---:|",
        f"| v 10 px occupied cells | {full_stats['v_grid_10px_occupied_count']} | {curated_stats['v_grid_10px_occupied_count']} | {curated_stats['v_continuous_ok']} |",
        f"| v 100 px occupied bins | {full_stats['v_bin_100px_occupied_count']} | {curated_stats['v_bin_100px_occupied_count']} | {curated_stats['v_continuous_ok']} |",
        f"| minimum edge frame count | {full_stats['edge_min_frame_count']} | {curated_stats['edge_min_frame_count']} | {curated_stats['edge_multiframe_ok']} |",
        f"| board-center Z span / mm | {fmt(full_stats['depth_center_span_mm'])} | {fmt(curated_stats['depth_center_span_mm'])} | {curated_stats['depth_lambda_excitation_ok']} |",
        f"| lambda truth span / mm | {fmt(full_stats['lambda_truth_span_mm'])} | {fmt(curated_stats['lambda_truth_span_mm'])} | {curated_stats['depth_lambda_excitation_ok']} |",
        f"| depth span ratio | 1.000 | {fmt(curated_stats['depth_span_ratio'], 3)} | {curated_stats['depth_lambda_excitation_ok']} |",
        f"| lambda span ratio | 1.000 | {fmt(curated_stats['lambda_span_ratio'], 3)} | {curated_stats['depth_lambda_excitation_ok']} |",
        f"| normal cover max angle / ° | 0.000 | {fmt(curated_stats['normal_cover_max_angle_deg'])} | {curated_stats['normal_angle_diversity_ok']} |",
        f"| normal diameter / ° | {fmt(full_stats['normal_pairwise_diameter_deg'])} | {fmt(curated_stats['normal_pairwise_diameter_deg'])} | {curated_stats['normal_angle_diversity_ok']} |",
        f"| translation X/Y bins | {TRANSLATION_BIN_COUNT}/{TRANSLATION_BIN_COUNT} | {curated_stats['translation_x_bin_occupied_count']}/{curated_stats['translation_y_bin_occupied_count']} | {curated_stats['translation_coverage_ok']} |",
        "",
        "## Historical 001–018 comparison",
        "",
        "Historical is treated as the original 001–018 FIT pose set. The comparison is geometric only and uses the same current point table and gates.",
        "",
        "| metric | Historical 001–018 | Full 36 | Curated |",
        "|---|---:|---:|---:|",
        f"| pose count | 18 | 36 | {len(curated_ids)} |",
        f"| v 10 px occupied cells | {historical_stats['v_grid_10px_occupied_count']} | {full_stats['v_grid_10px_occupied_count']} | {curated_stats['v_grid_10px_occupied_count']} |",
        f"| edge minimum frame count | {historical_stats['edge_min_frame_count']} | {full_stats['edge_min_frame_count']} | {curated_stats['edge_min_frame_count']} |",
        f"| board-center Z span / mm | {fmt(historical_stats['depth_center_span_mm'])} | {fmt(full_stats['depth_center_span_mm'])} | {fmt(curated_stats['depth_center_span_mm'])} |",
        f"| lambda truth span / mm | {fmt(historical_stats['lambda_truth_span_mm'])} | {fmt(full_stats['lambda_truth_span_mm'])} | {fmt(curated_stats['lambda_truth_span_mm'])} |",
        f"| normal diameter / ° | {fmt(historical_stats['normal_pairwise_diameter_deg'])} | {fmt(full_stats['normal_pairwise_diameter_deg'])} | {fmt(curated_stats['normal_pairwise_diameter_deg'])} |",
        f"| normal cover max to full / ° | {fmt(historical_stats['normal_cover_max_angle_deg'])} | 0.000 | {fmt(curated_stats['normal_cover_max_angle_deg'])} |",
        "",
        "Historical 001–018 already contains the principal low/high tilt and near/far depth families, but the 025–036 and 049–054 extension poses provide the explicit v-edge and lambda/depth extremes needed by the complete-workdomain gates. The curated set therefore keeps only those extension poses that add a declared geometric bin or normal cover while removing geometry-near repeats.",
        "",
        "## Files",
        "",
        "- `pose_geometry_metrics.csv`: per-pose PnP geometry, v/lambda support and leave-one-out results.",
        "- `pair_pose_similarity.csv`: all 630 pairwise geometry comparisons.",
        "- `pose_similarity_matrix.png`: geometric similarity heatmap.",
        "- `curated_fit_ids.json`: machine-readable selection and gate summary.",
        "- `curated_v_coverage.png`: full-vs-curated support and excitation plot.",
        "",
        f"`POSE_DIVERSITY = {diversity}`",
        f"`RECOMMENDED_CURATED_FIT_SIZE = {len(curated_ids)}`",
        "",
    ]
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return diversity


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空：{output_dir}；如需重跑请显式使用 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (INTRINSICS_PATH, CURRENT_POINTS_PATH, CURRENT_COVERAGE_PATH, CURRENT_REPORT_PATH, EXISTING_GEOMETRY_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)
    current_report = CURRENT_REPORT_PATH.read_text(encoding="utf-8")
    if "full_board_physical" not in current_report or "lambda_truth_mm" not in current_report:
        raise RuntimeError("current audit report does not prove full_board_physical + lambda_truth input")

    current_coverage = read_current_coverage_reference()
    print("[pose-geometry] loaded current coverage", flush=True)
    geometry = solve_fit_poses()
    print("[pose-geometry] assembled PnP geometry", flush=True)
    points = load_current_points()
    print("[pose-geometry] loaded current FIT points", flush=True)
    metrics, grid_sets, bin_sets, v_ranges, normal_angles, bin_edges = build_pose_metrics(geometry, points)
    print("[pose-geometry] built per-pose metrics", flush=True)
    pairs = pair_similarity(metrics, grid_sets, bin_sets, v_ranges, normal_angles)
    print("[pose-geometry] built pair similarity", flush=True)
    curated_ids, minimality_proven, solver = solve_curated_ids(
        FIT_IDS, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges
    )
    curated_ids = sorted(curated_ids, key=lambda value: int(value))
    print(f"[pose-geometry] curated selection via {solver}: {','.join(curated_ids)}", flush=True)
    full_stats = subset_stats(FIT_IDS, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    curated_stats = subset_stats(curated_ids, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    historical_stats = subset_stats(HISTORICAL_IDS, metrics, points, grid_sets, bin_sets, normal_angles, bin_edges)
    metrics_with_loo, full_stats, loo_by_id = attach_loo_and_selection(
        metrics, points, grid_sets, bin_sets, normal_angles, bin_edges, curated_ids
    )
    print("[pose-geometry] completed leave-one-out", flush=True)

    pair_counts = pairs.groupby("frame_a").size().to_dict()
    strict_counts: dict[str, int] = {frame_id: 0 for frame_id in FIT_IDS}
    for row in pairs[pairs["near_duplicate_strict"]].itertuples(index=False):
        strict_counts[row.frame_a] += 1
        strict_counts[row.frame_b] += 1
    metrics_with_loo["near_duplicate_strict_pair_count"] = metrics_with_loo["frame_id"].map(strict_counts).fillna(0).astype(int)
    metrics_with_loo["pair_count"] = metrics_with_loo["frame_id"].map(pair_counts).fillna(0).astype(int)

    metrics_with_loo.to_csv(output_dir / "pose_geometry_metrics.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output_dir / "pair_pose_similarity.csv", index=False, encoding="utf-8-sig")
    plot_similarity(output_dir / "pose_similarity_matrix.png", FIT_IDS, pairs, curated_ids)
    plot_curated_coverage(output_dir / "curated_v_coverage.png", points, FIT_IDS, curated_ids, metrics_with_loo)
    print("[pose-geometry] wrote plots", flush=True)

    selection_document = {
        "POSE_DIVERSITY": "SUFFICIENT" if curated_stats["overall_geometry_ok"] else "PARTIAL",
        "RECOMMENDED_CURATED_FIT_SIZE": len(curated_ids),
        "curated_fit_ids": curated_ids,
        "full_fit_ids": list(FIT_IDS),
        "historical_fit_ids": list(HISTORICAL_IDS),
        "minimality_proven_under_declared_gates": minimality_proven,
        "solver": solver,
        "selection_basis": "geometry-only set cover; no model residuals",
        "geometry_gates": {
            "v_grid_width_px": V_GRID_WIDTH_PX,
            "v_bin_width_px": V_BIN_WIDTH_PX,
            "edge_min_frame_count": EDGE_MIN_FRAME_COUNT,
            "normal_cover_threshold_deg": NORMAL_COVER_THRESHOLD_DEG,
            "normal_diameter_tolerance_deg": NORMAL_DIAMETER_TOLERANCE_DEG,
            "span_ratio_threshold": SPAN_RATIO_THRESHOLD,
            "excitation_bin_count": EXCITATION_BIN_COUNT,
            "full_depth_bin_count": full_stats["depth_bin_occupied_count"],
            "full_lambda_bin_count": full_stats["lambda_bin_occupied_count"],
            "translation_bin_count": TRANSLATION_BIN_COUNT,
        },
        "full_stats": jsonable(full_stats),
        "curated_stats": jsonable(curated_stats),
        "historical_stats": jsonable(historical_stats),
        "loo_loss_pose_ids": {
            "v": [frame_id for frame_id, stat in loo_by_id.items() if not stat["v_continuous_ok"]],
            "edge": [frame_id for frame_id, stat in loo_by_id.items() if not stat["edge_multiframe_ok"]],
            "depth_lambda": [frame_id for frame_id, stat in loo_by_id.items() if not stat["depth_lambda_excitation_ok"]],
            "normal": [frame_id for frame_id, stat in loo_by_id.items() if not stat["normal_angle_diversity_ok"]],
            "translation": [frame_id for frame_id, stat in loo_by_id.items() if not stat["translation_coverage_ok"]],
        },
        "strict_near_duplicate_pairs": pairs.loc[pairs["near_duplicate_strict"], ["frame_a", "frame_b"]].to_dict("records"),
        "inputs": {
            "intrinsics": str(INTRINSICS_PATH.resolve()),
            "existing_pnp_geometry": str(EXISTING_GEOMETRY_PATH.resolve()),
            "current_points": str(CURRENT_POINTS_PATH.resolve()),
            "current_coverage": str(CURRENT_COVERAGE_PATH.resolve()),
            "mask_provenance": "full_board_physical, inset=0 mm",
            "validation_opened": False,
            "model_residuals_used": False,
        },
    }
    (output_dir / "curated_fit_ids.json").write_text(
        json.dumps(selection_document, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8"
    )
    diversity = write_report(
        output_dir / "report.md",
        metrics_with_loo,
        pairs,
        full_stats,
        curated_stats,
        historical_stats,
        loo_by_id,
        curated_ids,
        minimality_proven,
        solver,
        current_coverage,
    )
    print(f"POSE_DIVERSITY = {diversity}")
    print(f"RECOMMENDED_CURATED_FIT_SIZE = {len(curated_ids)}")
    print("CURATED_FIT_IDS = " + ",".join(curated_ids))
    print(f"OUTPUT_DIR = {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
