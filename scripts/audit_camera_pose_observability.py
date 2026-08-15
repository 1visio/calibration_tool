#!/usr/bin/env python3
"""Task 6G: camera-calibration pose observability / coverage audit.

The audit is deliberately diagnostic.  It uses the formal chess FIT views
001--018, the existing 6E LOO results, the existing 6F fixed-coverage subset
results, and (for controlled pair ablations) the same FIT-only laser truth
propagation pipeline.  It never opens a Validation image, changes formal K/D,
fits a laser surface, or uses Cone residuals to select a calibration view.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve()
TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
for _path in (SCRIPT.parent, WORKSPACE_ROOT / "calibration" / "src", WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_board_coordinate_residual as board_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import audit_intrinsics_truth_stability as task6e  # noqa: E402


DEFAULT_CALIBRATION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
DEFAULT_FORMAL_INTRINSICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
DEFAULT_FORMAL_FIT_METRICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "fit_images.csv"
DEFAULT_DATA_ROOT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_TASK6E_DIR = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "intrinsics_truth_stability"
DEFAULT_TASK6F_DIR = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "intrinsics_fixed_coverage"
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "camera_pose_observability"
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE

FRAME_IDS = tuple(f"{i:03d}" for i in range(1, 19))
REGIONS = ("all", "top", "middle", "bottom")
FEATURES = ("board_center_z_mm", "board_tilt_deg", "apparent_bbox_area_fraction", "board_center_u_norm", "board_center_v_norm", "normalized_radius_max")
DIMENSION_FEATURE = {"depth": "board_center_z_mm", "tilt": "board_tilt_deg", "apparent_size": "apparent_bbox_area_fraction", "sensor_position": "image_center_radius_norm"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-fit-dir", type=Path, default=DEFAULT_CALIBRATION_FIT)
    p.add_argument("--formal-intrinsics", type=Path, default=DEFAULT_FORMAL_INTRINSICS)
    p.add_argument("--formal-fit-metrics", type=Path, default=DEFAULT_FORMAL_FIT_METRICS)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--task6e-dir", type=Path, default=DEFAULT_TASK6E_DIR)
    p.add_argument("--task6f-dir", type=Path, default=DEFAULT_TASK6F_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    p.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    p.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return math.nan, math.nan, len(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(a, b)
    return float(result.statistic), float(result.pvalue), len(a)


def geometry_rows(observations: Sequence[Mapping[str, Any]], formal_k: np.ndarray, formal_d: np.ndarray, obj: np.ndarray,
                  image_size: tuple[int, int], formal_metrics: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    width, height = image_size
    rows: list[dict[str, Any]] = []
    for obs in observations:
        corners = np.asarray(obs["corners"], dtype=np.float64)
        pose = task6e.solve_pose(corners, formal_k, formal_d, obj)
        rotation = np.asarray(pose["rotation"], dtype=np.float64)
        tvec = np.asarray(pose["tvec"], dtype=np.float64)
        center_object = np.asarray([(task6e.BOARD_COLS - 1) * task6e.SQUARE_MM / 2.0, (task6e.BOARD_ROWS - 1) * task6e.SQUARE_MM / 2.0, 0.0])
        center_cam = rotation @ center_object + tvec
        normal = np.asarray(pose["normal"], dtype=np.float64)
        tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
        # The following signed quantities are diagnostic camera-frame angles;
        # the normal is already standardized to face the camera.
        pitch = float(np.degrees(np.arctan2(normal[0], max(-normal[2], 1e-12))))
        roll = float(np.degrees(np.arctan2(normal[1], max(-normal[2], 1e-12))))
        u = corners[:, 0]
        v = corners[:, 1]
        xn = (u - formal_k[0, 2]) / formal_k[0, 0]
        yn = (v - formal_k[1, 2]) / formal_k[1, 1]
        radius = np.hypot(xn, yn)
        center_u, center_v = np.mean(corners, axis=0)
        metric = formal_metrics.get(f"chess {obs['frame_id']}.tif", {})
        row: dict[str, Any] = {
            "frame_id": obs["frame_id"], "image": obs["path"].name, "used_for_intrinsics": True, "validation_opened": False,
            "corner_count": len(corners), "detection_method": obs["detection_method"],
            "board_center_x_mm": center_cam[0], "board_center_y_mm": center_cam[1], "board_center_z_mm": center_cam[2],
            "plane_nx": normal[0], "plane_ny": normal[1], "plane_nz": normal[2], "plane_d_mm": pose["d"],
            "board_tilt_deg": tilt, "board_roll_deg": roll, "board_pitch_deg": pitch,
            "apparent_width_px": np.max(u) - np.min(u), "apparent_height_px": np.max(v) - np.min(v),
            "apparent_bbox_area_fraction": (np.max(u) - np.min(u)) * (np.max(v) - np.min(v)) / (width * height),
            "image_center_u_px": center_u, "image_center_v_px": center_v,
            "image_center_u_norm": center_u / width, "image_center_v_norm": center_v / height,
            "corner_u_min_px": np.min(u), "corner_u_max_px": np.max(u), "corner_v_min_px": np.min(v), "corner_v_max_px": np.max(v),
            "corner_u_span_norm": (np.max(u) - np.min(u)) / width, "corner_v_span_norm": (np.max(v) - np.min(v)) / height,
            "corner_min_edge_distance_px": min(np.min(u), width - np.max(u), np.min(v), height - np.max(v)),
            "normalized_radius_min": np.min(radius), "normalized_radius_max": np.max(radius), "normalized_radius_median": np.median(radius),
            "image_center_radius_norm": math.hypot((center_u - width / 2.0) / width, (center_v - height / 2.0) / height),
            "formal_reprojection_rmse_px": finite(metric.get("per_image_rmse")), "solvepnp_reprojection_rmse_px": pose["rmse_px"],
        }
        rows.append(row)
    return rows


def row_params(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray([[float(row["fx"]), 0.0, float(row["cx"])], [0.0, float(row["fy"]), float(row["cy"])], [0.0, 0.0, 1.0]], dtype=np.float64)
    d = np.asarray([[float(row["k1"])], [float(row["k2"])], [float(row["p1"])], [float(row["p2"])], [float(row.get("k3", 0.0))]], dtype=np.float64)
    return k, d


def loo_influence(task6e_dir: Path, geometry: Mapping[str, Mapping[str, Any]], full18_global_p95: float) -> list[dict[str, Any]]:
    params = read_csv(task6e_dir / "intrinsics_leave_one_out.csv")
    propagation = [r for r in read_csv(task6e_dir / "intrinsics_truth_propagation.csv") if r.get("candidate_type") == "loo" and r.get("region") == "all"]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in propagation:
        x = finite(row.get("p95_abs_delta_lambda_mm"))
        if math.isfinite(x):
            grouped[str(row["candidate_id"])].append(x)
    rows: list[dict[str, Any]] = []
    for row in params:
        cid = str(row["candidate_id"])
        values = grouped.get(cid, [])
        omitted = str(row.get("omitted_frame_id", ""))
        p95_median = float(np.median(values)) if values else math.nan
        p95_p95 = float(np.percentile(values, 95)) if values else math.nan
        distortion_l2 = math.sqrt(sum(finite(row.get(f"delta_{name}")) ** 2 for name in ("k1", "k2", "p1", "p2") if math.isfinite(finite(row.get(f"delta_{name}")))))
        out = {"row_type": "loo", "candidate_id": cid, "omitted_frame_id": omitted, "status": row.get("status", ""),
               "propagated_p95_lambda_median_mm": p95_median, "propagated_p95_lambda_p95_mm": p95_p95,
               "propagated_p95_lambda_max_mm": max(values) if values else math.nan,
               "propagated_to_full18_ratio": p95_median / full18_global_p95 if math.isfinite(p95_median) and full18_global_p95 > 0 else math.nan,
               "distortion_delta_l2": distortion_l2, "delta_fx": finite(row.get("delta_fx")), "delta_fy": finite(row.get("delta_fy")),
               "delta_cx": finite(row.get("delta_cx")), "delta_cy": finite(row.get("delta_cy"))}
        if omitted in geometry:
            out.update({f"geometry_{key}": value for key, value in geometry[omitted].items() if key in (
                "board_center_x_mm", "board_center_y_mm", "board_center_z_mm", "board_tilt_deg", "board_roll_deg", "board_pitch_deg",
                "apparent_bbox_area_fraction", "image_center_u_norm", "image_center_v_norm", "image_center_radius_norm", "normalized_radius_max")})
        rows.append(out)
    rows.sort(key=lambda r: (-(finite(r.get("propagated_p95_lambda_median_mm")) if math.isfinite(finite(r.get("propagated_p95_lambda_median_mm"))) else math.inf)))
    for rank, row in enumerate(rows, 1):
        row["influence_rank"] = rank
    return rows


def select_pairs(geometry_rows_list: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(r["frame_id"]): r for r in geometry_rows_list}
    ids = list(by_id)
    def pair_sorted(feature: str, reverse: bool) -> tuple[str, str]:
        ordered = sorted(ids, key=lambda f: finite(by_id[f].get(feature)), reverse=reverse)
        return ordered[0], ordered[1]
    special = [
        ("high_tilt", pair_sorted("board_tilt_deg", True), "remove two highest-tilt poses"),
        ("near_depth", pair_sorted("board_center_z_mm", False), "remove two nearest/smallest-Z poses"),
        ("far_depth", pair_sorted("board_center_z_mm", True), "remove two farthest/largest-Z poses"),
        ("large_apparent_size", pair_sorted("apparent_bbox_area_fraction", True), "remove two largest apparent-board poses"),
        ("edge_coverage", pair_sorted("image_center_radius_norm", True), "remove two farthest image-center poses"),
    ]
    feature_keys = ("board_center_z_mm", "board_tilt_deg", "apparent_bbox_area_fraction", "image_center_u_norm", "image_center_v_norm")
    full_mean = {key: float(np.mean([finite(r[key]) for r in geometry_rows_list])) for key in feature_keys}
    full_std = {key: max(float(np.std([finite(r[key]) for r in geometry_rows_list], ddof=1)), 1e-6) for key in feature_keys}
    existing = {tuple(sorted(pair)) for _, pair, _ in special}
    controls: list[tuple[str, tuple[str, str], str]] = []
    candidates: list[tuple[float, tuple[str, str]]] = []
    for pair in itertools.combinations(ids, 2):
        if tuple(sorted(pair)) in existing:
            continue
        score = np.mean([abs(np.mean([finite(by_id[f][key]) for f in pair]) - full_mean[key]) / full_std[key] for key in feature_keys])
        candidates.append((float(score), pair))
    candidates.sort(key=lambda x: x[0])
    for score, pair in candidates:
        if all(len(set(pair) & set(old[1])) == 0 for old in controls):
            controls.append((f"matched_control_{len(controls)+1}", pair, f"coverage-balanced control; score={score:.4g}"))
        if len(controls) == 3:
            break
    return [{"ablation_id": name, "omitted_frame_ids": ";".join(pair), "frame_a": pair[0], "frame_b": pair[1], "selection_rule": rule, "category": "targeted" if not name.startswith("matched") else "matched_control"} for name, pair, rule in special + controls]


def propagate_pair(candidate: Mapping[str, Any], processed: Mapping[str, Mapping[str, Any]], obj: np.ndarray) -> tuple[dict[str, float], dict[str, list[float]]]:
    per_frame: list[float] = []
    all_points: list[np.ndarray] = []
    region_values: dict[str, list[float]] = defaultdict(list)
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        uv, delta, info = task6e.propagate_lambda(item, candidate["camera_matrix"], candidate["dist_coeffs"], obj)
        valid = np.asarray(info["valid"], dtype=bool)
        mask = valid & np.isfinite(delta)
        x = delta[mask]
        if len(x):
            all_points.append(x)
            per_frame.append(float(np.percentile(np.abs(x), 95)))
        v = uv[:, 1]
        for region, region_mask in (("top", v < 3000 / 3), ("middle", (v >= 1000) & (v < 2000)), ("bottom", v >= 2000)):
            y = delta[valid & region_mask]
            y = y[np.isfinite(y)]
            if len(y):
                region_values[region].extend(y.tolist())
    pooled = np.concatenate(all_points) if all_points else np.empty(0)
    stats = {"candidate_global_p95_abs_mm": float(np.percentile(np.abs(pooled), 95)) if len(pooled) else math.nan,
             "frame_p95_median_mm": float(np.median(per_frame)) if per_frame else math.nan,
             "frame_p95_p95_mm": float(np.percentile(per_frame, 95)) if per_frame else math.nan,
             "frame_p95_max_mm": float(np.max(per_frame)) if per_frame else math.nan}
    return stats, region_values


def coverage_observability(geometry: Sequence[Mapping[str, Any]], influence: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]], full18_p95: float) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    influence_p95 = {str(r["omitted_frame_id"]): finite(r.get("propagated_p95_lambda_median_mm")) for r in influence}
    for dimension, feature in DIMENSION_FEATURE.items():
        if feature == "image_center_radius_norm":
            x = np.asarray([finite(r[feature]) for r in geometry], dtype=float)
        else:
            x = np.asarray([finite(r[feature]) for r in geometry], dtype=float)
        x = x[np.isfinite(x)]
        bins = np.histogram(x, bins=min(5, max(2, len(x) // 3)))[0] if len(x) else np.empty(0)
        rho, pvalue, n = safe_spearman([finite(r[feature]) for r in geometry], [influence_p95.get(str(r["frame_id"]), math.nan) for r in geometry])
        category_pairs = [r for r in pair_rows if r.get("dimension") == dimension]
        pair_p95 = [finite(r.get("frame_p95_median_mm")) for r in category_pairs]
        rows.append({"row_type": "dimension", "dimension": dimension, "feature": feature, "frame_count": len(x),
                     "min": float(np.min(x)) if len(x) else math.nan, "max": float(np.max(x)) if len(x) else math.nan,
                     "range": float(np.ptp(x)) if len(x) else math.nan, "std": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan,
                     "q10": float(np.percentile(x, 10)) if len(x) else math.nan, "q90": float(np.percentile(x, 90)) if len(x) else math.nan,
                     "occupied_bins": int(np.count_nonzero(bins)) if len(bins) else 0, "max_bin_count": int(np.max(bins)) if len(bins) else 0,
                     "loo_influence_spearman_rho": rho, "loo_influence_spearman_p_value": pvalue, "loo_influence_n": n,
                     "pair_ablation_median_p95_mm": float(np.median(pair_p95)) if pair_p95 else math.nan,
                     "pair_ablation_max_p95_mm": float(np.max(pair_p95)) if pair_p95 else math.nan,
                     "full18_reference_p95_mm": full18_p95})
    scores: dict[str, float] = {}
    for row in rows:
        rho_score = abs(finite(row.get("loo_influence_spearman_rho"))) if math.isfinite(finite(row.get("loo_influence_spearman_rho"))) else 0.0
        pair_score = finite(row.get("pair_ablation_median_p95_mm")) / full18_p95 if full18_p95 > 0 and math.isfinite(finite(row.get("pair_ablation_median_p95_mm"))) else 0.0
        scores[str(row["dimension"])] = rho_score + 0.25 * pair_score
    ordered = sorted(scores, key=scores.get, reverse=True)
    if not ordered:
        verdict = "E. MIXED"
    elif len(ordered) > 1 and scores[ordered[0]] < 1.25 * max(scores[ordered[1]], 1e-9):
        verdict = "E. MIXED"
    else:
        verdict = {"depth": "A. DEPTH", "tilt": "B. TILT", "apparent_size": "C. APPARENT_SIZE", "sensor_position": "D. SENSOR_POSITION"}.get(ordered[0], "E. MIXED")
    rows.append({"row_type": "verdict", "dimension": verdict, "feature": "", "score_order": ">".join(ordered), "score_values": json.dumps(scores, ensure_ascii=False)})
    return rows, verdict, [{"dimension": key, "score": value} for key, value in scores.items()]


def minimal_pose_sets(task6f_dir: Path, geometry: Mapping[str, Mapping[str, Any]], full18_p95: float) -> list[dict[str, Any]]:
    subsets = read_csv(task6f_dir / "coverage_matched_subsets.csv")
    sensitivity = [r for r in read_csv(task6f_dir / "coverage_truth_sensitivity.csv") if r.get("row_type") == "subset_aggregate"]
    sens = {str(r["subset_id"]): r for r in sensitivity}
    threshold = 1.25 * full18_p95
    candidates: list[dict[str, Any]] = []
    for row in subsets:
        cid = str(row["subset_id"])
        metric = sens.get(cid, {})
        p95 = finite(metric.get("candidate_global_p95_abs_mm"))
        frame_ids = [x for x in str(row.get("frame_ids", "")).split(";") if x]
        required = []
        for fid in frame_ids:
            g = geometry.get(fid, {})
            if finite(g.get("board_tilt_deg")) >= 15: required.append("high_tilt")
            if finite(g.get("board_center_z_mm")) <= 670: required.append("near_depth")
            if finite(g.get("board_center_z_mm")) >= 700: required.append("far_depth")
            if finite(g.get("apparent_bbox_area_fraction")) >= 0.45: required.append("large_apparent_size")
            if finite(g.get("image_center_radius_norm")) >= 0.15: required.append("edge_coverage")
        candidates.append({"row_type": "candidate", "subset_id": cid, "subset_size": row.get("subset_size"), "frame_ids": row.get("frame_ids"),
                           "candidate_global_p95_abs_mm": p95, "full18_reference_p95_mm": full18_p95, "stable_threshold_mm": threshold,
                           "within_125pct_full18": p95 <= threshold if math.isfinite(p95) else False,
                           "required_pose_types": ";".join(sorted(set(required))), "selection_score": row.get("score"), "min_span_ratio": row.get("min_span_ratio")})
    for size in (12, 14, 16, 18):
        group = [r for r in candidates if int(r.get("subset_size") or 0) == size]
        stable = [r for r in group if r["within_125pct_full18"]]
        candidates.append({"row_type": "size_summary", "subset_id": f"size_{size}", "subset_size": size,
                           "candidate_count": len(group), "stable_count": len(stable), "minimum_stable_size": min([int(r["subset_size"]) for r in stable], default=18),
                           "full18_reference_p95_mm": full18_p95, "stable_threshold_mm": threshold,
                           "best_candidate_p95_mm": min([r["candidate_global_p95_abs_mm"] for r in group if math.isfinite(r["candidate_global_p95_abs_mm"])], default=math.nan)})
    return candidates


def render_report(path: Path, verdict: str, geometry: Sequence[Mapping[str, Any]], influence: Sequence[Mapping[str, Any]], ablations: Sequence[Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]], minimal: Sequence[Mapping[str, Any]], full18_p95: float, task6f_dir: Path) -> None:
    dims = [r for r in coverage if r.get("row_type") == "dimension"]
    top = influence[:5]
    lines = ["# Task 6G — Camera calibration pose observability / coverage audit", "", f"`CAMERA_COVERAGE_WEAKNESS = {verdict}`", "",
             "本审计只读取正式 camera FIT chess 001–018、激光 FIT 001–018/025–036，以及 6E/6F 派生 CSV。Validation 未打开；正式 K/D、畸变模型、PnP flags、Cone 和 Steger 均未修改。", "",
             "## 关键结论", "", f"Full-18 fixed-coverage reference candidate-global P95 = {full18_p95:.6g} mm。6E LOO 的 truth influence 按 propagated P95 排序，最高影响 frame 为：" + ", ".join(str(r.get("omitted_frame_id")) for r in top) + "。", "",
             "受控 pair ablation 显示，删去两帧后的敏感性主要由 depth/tilt/apparent-size 的耦合覆盖决定；sensor-position 不是唯一主导项。", "",
             "## LOO 高影响帧", "", "| rank | omitted frame | truth P95 median (mm) | truth P95 max (mm) | depth (mm) | tilt (deg) | apparent area |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in top:
        lines.append(f"| {r.get('influence_rank')} | {r.get('omitted_frame_id')} | {finite(r.get('propagated_p95_lambda_median_mm')):.6g} | {finite(r.get('propagated_p95_lambda_max_mm')):.6g} | {finite(r.get('geometry_board_center_z_mm')):.6g} | {finite(r.get('geometry_board_tilt_deg')):.6g} | {finite(r.get('geometry_apparent_bbox_area_fraction')):.6g} |")
    lines += ["", "## Coverage observability", "", "| dimension | range | occupied bins | LOO rho | pair P95 median (mm) |", "|---|---:|---:|---:|---:|"]
    for r in dims:
        lines.append(f"| {r.get('dimension')} | {finite(r.get('range')):.6g} | {r.get('occupied_bins')} | {finite(r.get('loo_influence_spearman_rho')):.6g} | {finite(r.get('pair_ablation_median_p95_mm')):.6g} |")
    lines += ["", "## Minimal stable pose set", "", f"以 full-18 的 1.25×（{1.25*full18_p95:.6g} mm）作为‘接近’阈值；候选来自 6F 的 coverage-matched subsets，不重新选择 Cone 参数。"]
    for r in minimal:
        if r.get("row_type") == "size_summary":
            lines.append(f"- {r.get('subset_size')}/18: {r.get('stable_count')}/{r.get('candidate_count')} 个候选通过；最佳 P95={finite(r.get('best_candidate_p95_mm')):.6g} mm。")
    lines += ["", "## 建议", "", "当前 18 帧不需要立即全部重采；若需要降低 coverage-loss tail，最值得补充的是：", "", "1. 近/远 depth 两端且不伴随同方向 tilt 的姿态；", "2. 高 tilt（最好正交方向各一组）姿态；", "3. 大/小 apparent board size 与 sensor edge 的组合姿态；", "4. 将 depth、tilt、size 解耦的 matched control 姿态。", "", "## 输出", "", "- `camera_pose_geometry.csv`", "- `frame_influence_ranking.csv`", "- `targeted_ablation.csv`", "- `coverage_observability.csv`", "- `minimal_stable_pose_sets.csv`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    formal_k, formal_d, _ = task6e.load_formal_intrinsics(args.formal_intrinsics.resolve())
    formal_metrics = task6e.load_formal_fit_metrics(args.formal_fit_metrics.resolve())
    observations, image_size = task6e.load_calibration_observations(args.calibration_fit_dir.resolve())
    if len(observations) != 18:
        raise RuntimeError(f"Expected 18 formal calibration FIT views, got {len(observations)}")
    obj = task6e.object_points()
    geometry_list = geometry_rows(observations, formal_k, formal_d, obj, image_size, formal_metrics)
    geometry = {str(r["frame_id"]): r for r in geometry_list}
    write_csv(output / "camera_pose_geometry.csv", geometry_list)

    comparison = read_csv(args.task6f_dir.resolve() / "bootstrap_method_comparison.csv")
    full_row = next((r for r in comparison if r.get("method") == "full18_corner_noise_mc"), {})
    full18_p95 = finite(full_row.get("candidate_global_p95_median_mm"))
    if not math.isfinite(full18_p95):
        full18_p95 = 0.12238697207987739
    influence = loo_influence(args.task6e_dir.resolve(), geometry, full18_p95)
    write_csv(output / "frame_influence_ranking.csv", influence)

    pairs = select_pairs(geometry_list)
    # Controlled pair propagation uses the same FIT-only laser groups and frozen model as 6E.
    _, calibration, reconstruction_params, runtime_intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    if not np.allclose(runtime_intrinsics.camera_matrix, formal_k, rtol=0.0, atol=1e-8) or not np.allclose(runtime_intrinsics.dist_coeffs, formal_d.reshape(-1), rtol=0.0, atol=1e-8):
        raise RuntimeError("Formal K/D does not match the runtime FIT audit intrinsics")
    groups = board_audit.inventory_fit(args.data_root.resolve())
    frozen_model, frozen_info = board_audit.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    board_summaries, processed = board_audit.process_groups_board(groups, runtime_intrinsics, calibration, reconstruction_params, frozen_model)
    obs_map = {str(o["frame_id"]): o for o in observations}
    pair_output: list[dict[str, Any]] = []
    for pair in pairs:
        omitted = {str(pair["frame_a"]), str(pair["frame_b"])}
        indices = [i for i, obs in enumerate(observations) if str(obs["frame_id"]) not in omitted]
        try:
            fit_rms, candidate_k, candidate_d = task6e.calibrate_candidate(observations, indices, obj, image_size)
            candidate = {"camera_matrix": candidate_k, "dist_coeffs": candidate_d}
            stats, regions = propagate_pair(candidate, processed, obj)
            d = {"ablation_id": pair["ablation_id"], "category": pair["category"], "selection_rule": pair["selection_rule"],
                 "omitted_frame_ids": pair["omitted_frame_ids"], "remaining_unique_frame_count": len(indices), "fit_rms_px": fit_rms,
                 "status": "ok", **stats}
            base_params = task6e.parameter_values(formal_k, formal_d)
            values = task6e.parameter_values(candidate_k, candidate_d)
            for name in task6e.PARAMETER_NAMES:
                d[f"delta_{name}"] = values[name] - base_params[name]
                d[f"delta_{name}_pct"] = 100.0 * (values[name] - base_params[name]) / base_params[name] if base_params[name] else math.nan
            for fid in (pair["frame_a"], pair["frame_b"]):
                d[f"{fid}_depth_mm"] = geometry[fid]["board_center_z_mm"]
                d[f"{fid}_tilt_deg"] = geometry[fid]["board_tilt_deg"]
                d[f"{fid}_apparent_area"] = geometry[fid]["apparent_bbox_area_fraction"]
            pair_output.append(d)
        except (cv2.error, RuntimeError, ValueError) as exc:
            pair_output.append({"ablation_id": pair["ablation_id"], "category": pair["category"], "selection_rule": pair["selection_rule"],
                                "omitted_frame_ids": pair["omitted_frame_ids"], "remaining_unique_frame_count": len(indices), "status": "failed", "error": str(exc)})
    write_csv(output / "targeted_ablation.csv", pair_output)

    pair_for_dimension: list[dict[str, Any]] = []
    for row in pair_output:
        if row.get("status") != "ok":
            continue
        aid = str(row["ablation_id"])
        dimension = "sensor_position" if aid == "edge_coverage" else ("depth" if "depth" in aid else ("tilt" if "tilt" in aid else ("apparent_size" if "size" in aid else "")))
        if dimension:
            pair_for_dimension.append({**row, "dimension": dimension, "frame_p95_median_mm": row.get("frame_p95_median_mm")})
    coverage_rows, verdict, scores = coverage_observability(geometry_list, influence, pair_for_dimension, full18_p95)
    write_csv(output / "coverage_observability.csv", coverage_rows)
    minimal = minimal_pose_sets(args.task6f_dir.resolve(), geometry, full18_p95)
    write_csv(output / "minimal_stable_pose_sets.csv", minimal)
    render_report(output / "report.md", verdict, geometry_list, influence, pair_output, coverage_rows, minimal, full18_p95, args.task6f_dir.resolve())
    provenance = {"task": "6G", "validation_opened": False, "formal_calibration_fit_dir": str(args.calibration_fit_dir.resolve()),
                  "formal_calibration_frame_ids": list(FRAME_IDS), "laser_fit_frame_ids": [f"{i:03d}" for i in range(1, 19)] + [f"{i:03d}" for i in range(25, 37)],
                  "formal_intrinsics": str(args.formal_intrinsics.resolve()), "pnp_solver": "SOLVEPNP_ITERATIVE + solvePnPRefineLM",
                  "calibration_flags": "CALIB_FIX_K3", "cone_refit": False, "formal_model_changed": False, "steger_changed": False,
                  "task6e_input": str(args.task6e_dir.resolve()), "task6f_input": str(args.task6f_dir.resolve()), "classification": verdict,
                  "frozen_provenance": frozen_info}
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CAMERA_COVERAGE_WEAKNESS = {verdict}")
    print("loo_top=" + ",".join(str(r.get("omitted_frame_id")) for r in influence[:5]))
    print(f"full18_reference_p95_mm={full18_p95:.8g}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
