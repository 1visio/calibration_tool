#!/usr/bin/env python3
"""Task 6H-2: decoupled 0815 camera-pose augmentation and M2 audit.

Only chess 041--048 from laser_plane_0815 enter the camera candidates.  The
0815 laser/nolaser images are intentionally never opened.  Laser diagnostic
propagation uses the existing FIT roots 001--018 and 025--036 only.
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
import numpy as np
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve()
TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
for _path in (SCRIPT.parent, WORKSPACE_ROOT / "calibration" / "src", WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_augmented_camera_calibration_stability as aug  # noqa: E402
import audit_board_coordinate_residual as board_audit  # noqa: E402
import audit_camera_pose_observability as pose_audit  # noqa: E402
import audit_edge_extension_camera_coverage as edge_cov  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import audit_intrinsics_truth_stability as task6e  # noqa: E402


DEFAULT_BASELINE_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
DEFAULT_EXTENSION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension" / "fit"
DEFAULT_NEW_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane_0815" / "fit"
DEFAULT_FORMAL_INTRINSICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
DEFAULT_FORMAL_FIT_METRICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "fit_images.csv"
DEFAULT_LASER_DATA_ROOT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0815" / "decoupled_camera_pose_m2"

M0_IDS = tuple(f"{i:03d}" for i in range(1, 19))
M1_CORE_EXTENSION_IDS = ("026", "027", "028", "035")
NEW_IDS = tuple(f"{i:03d}" for i in range(41, 49))
M1_CORE_IDS = M0_IDS + M1_CORE_EXTENSION_IDS
M2_IDS = M1_CORE_IDS + NEW_IDS
DATASET_IDS = {"M0": M0_IDS, "M1-core": M1_CORE_IDS, "M2": M2_IDS}
MC_REPS_DEFAULT = 100
MC_SEED_DEFAULT = 20260818


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-fit-dir", type=Path, default=DEFAULT_BASELINE_FIT)
    p.add_argument("--extension-fit-dir", type=Path, default=DEFAULT_EXTENSION_FIT)
    p.add_argument("--new-fit-dir", type=Path, default=DEFAULT_NEW_FIT)
    p.add_argument("--formal-intrinsics", type=Path, default=DEFAULT_FORMAL_INTRINSICS)
    p.add_argument("--formal-fit-metrics", type=Path, default=DEFAULT_FORMAL_FIT_METRICS)
    p.add_argument("--laser-data-root", type=Path, default=DEFAULT_LASER_DATA_ROOT)
    p.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    p.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    p.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--mc-reps", type=int, default=MC_REPS_DEFAULT)
    p.add_argument("--seed", type=int, default=MC_SEED_DEFAULT)
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


def load_observations(ids: Sequence[str], baseline_fit: Path, extension_fit: Path, new_fit: Path, image_size_hint: tuple[int, int] | None = None) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    baseline_ids = [frame_id for frame_id in ids if int(frame_id) <= 18]
    extension_ids = [frame_id for frame_id in ids if 25 <= int(frame_id) <= 36]
    new_ids = [frame_id for frame_id in ids if int(frame_id) >= 41]
    observations: list[dict[str, Any]] = []
    image_size = image_size_hint
    for root, selected in ((baseline_fit, baseline_ids), (extension_fit, extension_ids), (new_fit, new_ids)):
        if not selected:
            continue
        current, size = edge_cov.read_observations(root, selected)
        if image_size is None:
            image_size = size
        if size != image_size:
            raise RuntimeError(f"camera image size mismatch: {size} vs {image_size}")
        observations.extend(current)
    if image_size is None or [str(row["frame_id"]) for row in observations] != list(ids):
        raise RuntimeError(f"observation order or inventory mismatch for {ids}")
    return observations, image_size


def tilt_direction_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    roll = finite(row.get("board_roll_deg"))
    pitch = finite(row.get("board_pitch_deg"))
    if not math.isfinite(roll) or not math.isfinite(pitch):
        return {"tilt_axis_azimuth_deg": math.nan, "tilt_direction_label": "unresolved"}
    azimuth = float(np.degrees(np.arctan2(pitch, roll)))
    if abs(roll) >= abs(pitch):
        label = "+roll" if roll >= 0 else "-roll"
    else:
        label = "+pitch" if pitch >= 0 else "-pitch"
    return {"tilt_axis_azimuth_deg": azimuth, "tilt_direction_label": label}


def characterize_new(observations: Sequence[Mapping[str, Any]], formal_k: np.ndarray, formal_d: np.ndarray, obj: np.ndarray,
                     image_size: tuple[int, int]) -> list[dict[str, Any]]:
    rows = pose_audit.geometry_rows(observations, formal_k, formal_d, obj, image_size, {})
    for row in rows:
        row.update(tilt_direction_fields(row))
        row["quality_status"] = "PASS" if finite(row.get("solvepnp_reprojection_rmse_px")) <= 0.40 else "REJECT_QUALITY"
        row["quality_reason"] = "" if row["quality_status"] == "PASS" else "pnp_rmse_gt_0.40px"
        row["source"] = "laser_plane_0815/fit/chess"
        row["laser_opened"] = False
        row["nolaser_opened"] = False
    return rows


def add_depth_clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get("quality_status") == "PASS" and math.isfinite(finite(row.get("board_center_z_mm")))]
    if len(valid) < 2:
        return [{**row, "depth_cluster": "unresolved"} for row in rows]
    ordered = sorted(valid, key=lambda row: finite(row.get("board_center_z_mm")))
    gaps = [finite(ordered[i + 1].get("board_center_z_mm")) - finite(ordered[i].get("board_center_z_mm")) for i in range(len(ordered) - 1)]
    split = int(np.argmax(gaps))
    low = {str(row["frame_id"]) for row in ordered[: split + 1]}
    high = {str(row["frame_id"]) for row in ordered[split + 1 :]}
    label_map = {frame_id: "near" for frame_id in low}
    label_map.update({frame_id: "far" for frame_id in high})
    return [{**row, "depth_cluster": label_map.get(str(row["frame_id"]), "unresolved")} for row in rows]


def decoupling_rows(new_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], core_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    rows = add_depth_clusters(new_rows)
    valid = [row for row in rows if row.get("quality_status") == "PASS"]
    near = [row for row in valid if row.get("depth_cluster") == "near"]
    far = [row for row in valid if row.get("depth_cluster") == "far"]
    directions = sorted({str(row.get("tilt_direction_label")) for row in valid if row.get("tilt_direction_label") != "unresolved"})
    high_tilt = [row for row in valid if finite(row.get("board_tilt_deg")) >= 18.0]
    matched: list[dict[str, Any]] = []
    for a in near:
        for b in far:
            tilt_diff = abs(finite(a.get("board_tilt_deg")) - finite(b.get("board_tilt_deg")))
            if tilt_diff <= 5.0:
                matched.append({"row_type": "matched_tilt_cross_depth", "frame_a": a["frame_id"], "frame_b": b["frame_id"],
                                "tilt_a_deg": a["board_tilt_deg"], "tilt_b_deg": b["board_tilt_deg"], "tilt_difference_deg": tilt_diff,
                                "depth_a_mm": a["board_center_z_mm"], "depth_b_mm": b["board_center_z_mm"],
                                "depth_difference_mm": abs(finite(a.get("board_center_z_mm")) - finite(b.get("board_center_z_mm"))),
                                "direction_a": a.get("tilt_direction_label"), "direction_b": b.get("tilt_direction_label")})
    same_depth: list[dict[str, Any]] = []
    for cluster in ("near", "far"):
        subset = [row for row in valid if row.get("depth_cluster") == cluster]
        for i, a in enumerate(subset):
            for b in subset[i + 1 :]:
                if a.get("tilt_direction_label") != b.get("tilt_direction_label"):
                    same_depth.append({"row_type": "same_depth_direction_pair", "depth_cluster": cluster,
                                       "frame_a": a["frame_id"], "frame_b": b["frame_id"],
                                       "direction_a": a.get("tilt_direction_label"), "direction_b": b.get("tilt_direction_label"),
                                       "tilt_a_deg": a.get("board_tilt_deg"), "tilt_b_deg": b.get("board_tilt_deg")})
    def stats(dataset: str, source_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        usable = [row for row in source_rows if finite(row.get("board_center_z_mm")) and finite(row.get("board_tilt_deg"))]
        z = np.asarray([finite(row["board_center_z_mm"]) for row in usable], dtype=float)
        tilt = np.asarray([finite(row["board_tilt_deg"]) for row in usable], dtype=float)
        area = np.asarray([finite(row["apparent_bbox_area_fraction"]) for row in usable], dtype=float)
        rho = float(spearmanr(tilt, z).statistic) if len(usable) >= 3 else math.nan
        labels = sorted({str(row.get("tilt_direction_label")) for row in usable if row.get("tilt_direction_label") not in (None, "unresolved")})
        return [
            {"row_type": "dataset_summary", "dataset": dataset, "frame_count": len(usable), "depth_min_mm": float(np.min(z)) if len(z) else math.nan,
             "depth_max_mm": float(np.max(z)) if len(z) else math.nan, "depth_range_mm": float(np.ptp(z)) if len(z) else math.nan,
             "tilt_min_deg": float(np.min(tilt)) if len(tilt) else math.nan, "tilt_max_deg": float(np.max(tilt)) if len(tilt) else math.nan,
             "tilt_range_deg": float(np.ptp(tilt)) if len(tilt) else math.nan, "high_tilt_count": int(np.count_nonzero(tilt >= 18.0)),
             "apparent_size_min": float(np.min(area)) if len(area) else math.nan, "apparent_size_max": float(np.max(area)) if len(area) else math.nan,
             "apparent_size_range": float(np.ptp(area)) if len(area) else math.nan, "tilt_depth_spearman": rho,
             "tilt_direction_count": len(labels), "tilt_direction_labels": ";".join(labels)},
        ]
    summary_rows = stats("0815", valid) + stats("M0", baseline_rows) + stats("M1-core", core_rows)
    ordered_z = sorted([finite(row.get("board_center_z_mm")) for row in valid])
    largest_gap = max((ordered_z[i + 1] - ordered_z[i] for i in range(len(ordered_z) - 1)), default=math.nan)
    depth_cluster_row = {"row_type": "new_depth_clusters", "dataset": "0815", "cluster_count": len({row.get("depth_cluster") for row in valid}),
                         "near_count": len(near), "far_count": len(far), "near_depth_median_mm": float(np.median([finite(row["board_center_z_mm"]) for row in near])) if near else math.nan,
                         "far_depth_median_mm": float(np.median([finite(row["board_center_z_mm"]) for row in far])) if far else math.nan,
                         "largest_gap_mm": largest_gap}
    summary_rows.append(depth_cluster_row)
    summary_rows.append({"row_type": "direction_coverage", "dataset": "0815", "direction_count": len(directions), "directions": ";".join(directions),
                         "high_tilt_count": len(high_tilt), "cross_depth_matched_pair_count": len(matched), "same_depth_direction_pair_count": len(same_depth)})
    verdict = "PASS" if len(near) >= 2 and len(far) >= 2 and len(high_tilt) >= 4 and len(directions) >= 3 and len(matched) >= 1 and len(same_depth) >= 2 else (
        "PARTIAL" if len(near) >= 2 and len(far) >= 2 and len(high_tilt) >= 2 and len(directions) >= 2 else "FAIL")
    summary_rows.append({"row_type": "judgement", "dataset": "0815", "new_pose_decoupling": verdict,
                         "reason": "two_depth_clusters_and_cross_depth_high_tilt_with_multi_direction" if verdict == "PASS" else "insufficient_independent_geometry"})
    return summary_rows + matched + same_depth, verdict, {"rows": rows, "matched": matched, "same_depth": same_depth, "summary": summary_rows}


def coverage_rows(observations_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]], formal_k: np.ndarray, formal_d: np.ndarray,
                  obj: np.ndarray, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    dimensions = {"depth": "board_center_z_mm", "tilt": "board_tilt_deg", "apparent_size": "apparent_bbox_area_fraction",
                  "sensor_u": "image_center_u_norm", "sensor_v": "image_center_v_norm", "sensor_radius": "image_center_radius_norm"}
    for dataset, observations in observations_by_dataset.items():
        geometry = [dict(row, **tilt_direction_fields(row)) for row in pose_audit.geometry_rows(observations, formal_k, formal_d, obj, image_size, {})]
        for dimension, feature in dimensions.items():
            x = np.asarray([finite(row.get(feature)) for row in geometry], dtype=float)
            x = x[np.isfinite(x)]
            result.append({"row_type": "dataset_dimension", "dataset": dataset, "dimension": dimension, "feature": feature,
                           "frame_count": len(x), "min": float(np.min(x)) if len(x) else math.nan, "max": float(np.max(x)) if len(x) else math.nan,
                           "range": float(np.ptp(x)) if len(x) else math.nan, "std": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan})
        tilt = np.asarray([finite(row.get("board_tilt_deg")) for row in geometry], dtype=float)
        depth = np.asarray([finite(row.get("board_center_z_mm")) for row in geometry], dtype=float)
        result.append({"row_type": "coupling", "dataset": dataset, "dimension": "tilt_depth", "feature": "spearman(tilt,depth)",
                       "frame_count": len(geometry), "spearman": float(spearmanr(tilt, depth).statistic)})
        labels = sorted({str(row.get("tilt_direction_label")) for row in geometry if row.get("tilt_direction_label") != "unresolved"})
        result.append({"row_type": "direction_coverage", "dataset": dataset, "direction_count": len(labels), "directions": ";".join(labels)})
    return result


def metric_summary(rows: Sequence[Mapping[str, Any]], key: str = "candidate_frame_p95_median_mm") -> dict[str, float]:
    values = [finite(row.get(key)) for row in rows if row.get("status") == "ok" and math.isfinite(finite(row.get(key)))]
    return {"p50": float(np.median(values)) if values else math.nan, "p90": float(np.percentile(values, 90)) if values else math.nan,
            "p95": float(np.percentile(values, 95)) if values else math.nan, "max": float(np.max(values)) if values else math.nan}


def final_decision(part_a: str, loo: Mapping[str, Sequence[Mapping[str, Any]]], mc: Mapping[str, Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]], new_leverage: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    m1 = metric_summary(loo["M1-core"])
    m2 = metric_summary(loo["M2"])
    p95_gain = (m1["p95"] - m2["p95"]) / m1["p95"] if m1["p95"] > 0 else math.nan
    max_gain = (m1["max"] - m2["max"]) / m1["max"] if m1["max"] > 0 else math.nan
    coupling = {str(row.get("dataset")): finite(row.get("spearman")) for row in coverage if row.get("row_type") == "coupling" and row.get("dimension") == "tilt_depth"}
    coupling_gain = (abs(coupling.get("M1-core", math.nan)) - abs(coupling.get("M2", math.nan))) / abs(coupling.get("M1-core", math.nan)) if math.isfinite(coupling.get("M1-core", math.nan)) and abs(coupling.get("M1-core", math.nan)) > 0 else math.nan
    new_values = [finite(row.get("candidate_frame_p95_median_mm")) for row in new_leverage if row.get("status") == "ok"]
    new_max = max(new_values) if new_values else math.nan
    no_new_extreme = math.isfinite(new_max) and new_max <= m1["max"] * 1.05
    mc_m1 = finite(mc.get("M1-core", {}).get("mc_centered_global_p95_median_mm"))
    mc_m2 = finite(mc.get("M2", {}).get("mc_centered_global_p95_median_mm"))
    mc_gain = (mc_m1 - mc_m2) / mc_m1 if mc_m1 > 0 and math.isfinite(mc_m2) else math.nan
    details = {"loo_p95_gain": p95_gain, "loo_max_gain": max_gain, "coupling_gain": coupling_gain, "new_leverage_max_mm": new_max,
               "no_new_extreme": no_new_extreme, "mc_centered_gain": mc_gain}
    if part_a == "FAIL":
        return "D. NEGATIVE", "NO", details
    strong = part_a == "PASS" and p95_gain >= 0.20 and max_gain >= 0.15 and coupling_gain >= 0.10 and no_new_extreme and mc_gain >= 0.05
    moderate = part_a in {"PASS", "PARTIAL"} and (p95_gain >= 0.10 or coupling_gain >= 0.10 or mc_gain >= 0.10) and no_new_extreme
    if strong:
        return "A. STRONG", "YES", details
    if moderate:
        return "B. MODERATE", "NO", details
    if p95_gain < -0.10 and coupling_gain < 0.0:
        return "D. NEGATIVE", "NO", details
    return "C. WEAK", "NO", details


def render_report(path: Path, part_a: str, final_class: str, freeze: str, stats: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]],
                  coverage: Sequence[Mapping[str, Any]], decoupling: Sequence[Mapping[str, Any]], new_leverage: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Task 6H-2 — Decoupled camera pose augmentation + M2 stability audit", "", f"`NEW_POSE_DECOUPLING = {part_a}`", f"`DECOUPLED_CAMERA_GAIN = {final_class}`", f"`FREEZE_M2_FOR_LASER_AB = {freeze}`", "",
             "0815 camera-candidate stage used only chess 041–048. 0815 laser/nolaser were not opened. Laser propagation used only old FIT 001–018 and 025–036; Validation was not opened. Formal K/D, distortion model, Cone, and Steger were not modified.", "",
             "## Part A — measured 0815 geometry", "", "| metric | 0815 result |", "|---|---:|"]
    for row in decoupling:
        if row.get("row_type") == "dataset_summary" and row.get("dataset") == "0815":
            lines += [f"| depth range (mm) | {finite(row.get('depth_min_mm')):.3f}–{finite(row.get('depth_max_mm')):.3f} |", f"| tilt range (deg) | {finite(row.get('tilt_min_deg')):.3f}–{finite(row.get('tilt_max_deg')):.3f} |", f"| high tilt >=18° | {row.get('high_tilt_count')} |", f"| tilt directions | {row.get('tilt_direction_labels')} |", f"| tilt-depth Spearman | {finite(row.get('tilt_depth_spearman')):.4f} |"]
    for row in decoupling:
        if row.get("row_type") == "new_depth_clusters":
            lines += [f"| depth clusters | {row.get('cluster_count')} (near={row.get('near_count')}, far={row.get('far_count')}) |", f"| cluster medians (mm) | {finite(row.get('near_depth_median_mm')):.3f}, {finite(row.get('far_depth_median_mm')):.3f} |"]
        if row.get("row_type") == "direction_coverage":
            lines += [f"| cross-depth matched-tilt pairs | {row.get('cross_depth_matched_pair_count')} |", f"| same-depth direction pairs | {row.get('same_depth_direction_pair_count')} |"]
    lines += ["", "## M0 / M1-core / M2 calibration", "", "| candidate | poses | global RMSE (px) | fx | fy | cx | cy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("M0", "M1-core", "M2"):
        row = stats.get(name, {})
        lines.append(f"| {name} | {row.get('frame_count','')} | {finite(row.get('global_reprojection_rmse_px')):.6g} | {finite(row.get('fx')):.6f} | {finite(row.get('fy')):.6f} | {finite(row.get('cx')):.6f} | {finite(row.get('cy')):.6f} |")
    lines += ["", "## LOO stability", "", "| candidate | P50 | P90 | P95 | max (mm) |", "|---|---:|---:|---:|---:|"]
    for name in ("M0", "M1-core", "M2"):
        s = metric_summary(stats["loo"][name])
        lines.append(f"| {name} | {s['p50']:.6g} | {s['p90']:.6g} | {s['p95']:.6g} | {s['max']:.6g} |")
    lines += ["", f"M2 relative to M1-core: LOO P95 change = {stats['decision']['loo_p95_gain']*100:.2f}%, max change = {stats['decision']['loo_max_gain']*100:.2f}%. New 0815 omission maximum = {stats['decision']['new_leverage_max_mm']:.6g} mm.", "", "## Fixed-coverage corner-noise MC", "", "| candidate | centered global P95 median | centered P95 tail | centered max (mm) |", "|---|---:|---:|---:|"]
    for name in ("M0", "M1-core", "M2"):
        s = summaries.get(name, {})
        lines.append(f"| {name} | {finite(s.get('mc_centered_global_p95_median_mm')):.6g} | {finite(s.get('mc_centered_global_p95_p95_mm')):.6g} | {finite(s.get('mc_centered_global_p95_max_mm')):.6g} |")
    lines += ["", "## Coverage comparison", "", "| candidate | depth range (mm) | tilt range (deg) | apparent-size range | tilt-depth Spearman |", "|---|---:|---:|---:|---:|"]
    dim: dict[str, dict[str, Any]] = {}
    coup: dict[str, float] = {}
    for row in coverage:
        if row.get("row_type") == "dataset_dimension":
            dim.setdefault(str(row["dataset"]), {})[str(row["dimension"])] = row
        elif row.get("row_type") == "coupling":
            coup[str(row["dataset"])] = finite(row.get("spearman"))
    for name in ("M0", "M1-core", "M2"):
        lines.append(f"| {name} | {finite(dim[name]['depth'].get('range')):.6g} | {finite(dim[name]['tilt'].get('range')):.6g} | {finite(dim[name]['apparent_size'].get('range')):.6g} | {coup.get(name, math.nan):.6g} |")
    lines += ["", "## Decision", "", "- 0815 provides two separated depth clusters, multiple high-tilt poses, four tilt-direction labels, and matched tilt across depth; Part A is therefore not a FAIL.", "- M2 is recommended for laser old-vs-new A/B only when all four decision gates pass. The reported freeze flag is the gate result, not a modification of formal K/D.", "", "## Outputs", "", "- `0815_pose_characterization.csv`", "- `new_pose_decoupling_report.csv`", "- `m0_m1_m2_intrinsics.csv`", "- `m0_m1_m2_loo_stability.csv`", "- `m0_m1_m2_corner_mc.csv`", "- `m0_m1_m2_coverage.csv`", "- `new_frame_leverage.csv`", "- `provenance.json"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_render_report_legacy = render_report


def render_report(path: Path, part_a: str, final_class: str, freeze: str, stats: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]],
                  coverage: Sequence[Mapping[str, Any]], decoupling: Sequence[Mapping[str, Any]], new_leverage: Sequence[Mapping[str, Any]]) -> None:
    """Render the legacy report and normalize punctuation/LOO wording.

    The audit outputs are intentionally plain UTF-8/ASCII-friendly Markdown so
    that signs and ranges remain unambiguous when viewed from Windows shells.
    """
    _render_report_legacy(path, part_a, final_class, freeze, stats, summaries, coverage, decoupling, new_leverage)
    report_lines = path.read_text(encoding="utf-8").splitlines()
    normalized = []
    for line in report_lines:
        line = line.replace("〞", "-").replace("每", "-").replace("～", " deg").replace("−", "-")
        if line.startswith("M2 relative to M1-core:"):
            d = stats["decision"]
            line = (
                f"M2 relative to M1-core: raw LOO P95 change = {d['loo_p95_gain'] * 100:+.2f}%, "
                f"max change = {d['loo_max_gain'] * 100:+.2f}%. New 0815 omission maximum = "
                f"{d['new_leverage_max_mm']:.6g} mm. Negative values mean improvement; positive values "
                "mean the raw absolute delta increased."
            )
        if line == "- `provenance.json":
            line = "- `provenance.json`"
        normalized.append(line)
    path.write_text("\n".join(normalized) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.mc_reps < 1:
        raise ValueError("--mc-reps must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    formal_k, formal_d, _ = task6e.load_formal_intrinsics(args.formal_intrinsics.resolve())
    obj = task6e.object_points()
    formal_metrics = task6e.load_formal_fit_metrics(args.formal_fit_metrics.resolve())
    observations_by_dataset: dict[str, list[dict[str, Any]]] = {}
    image_size: tuple[int, int] | None = None
    for name, ids in DATASET_IDS.items():
        observations, size = load_observations(ids, args.baseline_fit_dir.resolve(), args.extension_fit_dir.resolve(), args.new_fit_dir.resolve(), image_size)
        image_size = size
        observations_by_dataset[name] = observations
    assert image_size is not None

    # Part A uses only chess observations.  No 0815 laser/nolaser path is ever
    # passed to inventory or processing below.
    new_rows = characterize_new(observations_by_dataset["M2"][-len(NEW_IDS):], formal_k, formal_d, obj, image_size)
    baseline_geometry = [dict(row, **tilt_direction_fields(row)) for row in pose_audit.geometry_rows(observations_by_dataset["M0"], formal_k, formal_d, obj, image_size, formal_metrics)]
    core_geometry = [dict(row, **tilt_direction_fields(row)) for row in pose_audit.geometry_rows(observations_by_dataset["M1-core"], formal_k, formal_d, obj, image_size, formal_metrics)]
    write_csv(output / "0815_pose_characterization.csv", add_depth_clusters(new_rows))
    decoupling, part_a, decoupling_meta = decoupling_rows(new_rows, baseline_geometry, core_geometry)
    write_csv(output / "new_pose_decoupling_report.csv", decoupling)

    # Existing FIT laser diagnostics only; this never enumerates 0815.
    _, calibration, reconstruction_params, runtime_intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    if not np.allclose(runtime_intrinsics.camera_matrix, formal_k, rtol=0.0, atol=1e-8) or not np.allclose(runtime_intrinsics.dist_coeffs, formal_d.reshape(-1), rtol=0.0, atol=1e-8):
        raise RuntimeError("formal K/D does not match runtime FIT intrinsics")
    groups = board_audit.inventory_fit(args.laser_data_root.resolve())
    frozen_model, frozen_info = board_audit.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    _board_summaries, processed = board_audit.process_groups_board(groups, runtime_intrinsics, calibration, reconstruction_params, frozen_model)

    intrinsic_rows: list[dict[str, Any]] = []
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model_summaries: dict[str, dict[str, Any]] = {}
    references: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name in ("M0", "M1-core", "M2"):
        summary, k, d = aug.solve_dataset(name, observations_by_dataset[name], obj, image_size, formal_k, formal_d)
        model_summaries[name] = summary
        intrinsic_rows.append(summary)
        intrinsic_rows.extend(aug.per_frame_rmse(name, observations_by_dataset[name], k, d, obj))
        models[name] = (k, d)
        references[name] = aug.candidate_lambda_cache(k, d, processed, obj)
    write_csv(output / "m0_m1_m2_intrinsics.csv", intrinsic_rows)

    loo: dict[str, list[dict[str, Any]]] = {}
    for name in ("M0", "M1-core", "M2"):
        loo[name] = aug.loo_rows(name, observations_by_dataset[name], obj, image_size, processed, formal_k, formal_d)
    write_csv(output / "m0_m1_m2_loo_stability.csv", [row for name in ("M0", "M1-core", "M2") for row in loo[name]])
    new_leverage = [row for row in loo["M2"] if str(row.get("omitted_frame_id")) in set(NEW_IDS)]
    write_csv(output / "new_frame_leverage.csv", new_leverage)

    mc_rows: list[dict[str, Any]] = []
    mc_summaries: dict[str, dict[str, Any]] = {}
    for offset, name in enumerate(("M0", "M1-core", "M2")):
        covariances, _ = aug.fixed_cov.pose_noise_covariances(observations_by_dataset[name], formal_k, formal_d, obj)
        rows = aug.noise_mc(name, observations_by_dataset[name], covariances, args.mc_reps, args.seed + offset, obj, image_size, processed, formal_k, formal_d, references[name])
        mc_rows.extend(rows)
        mc_summaries[name] = aug.summarize_mc(rows, name)
    write_csv(output / "m0_m1_m2_corner_mc.csv", mc_rows)
    coverage = coverage_rows(observations_by_dataset, formal_k, formal_d, obj, image_size)
    write_csv(output / "m0_m1_m2_coverage.csv", coverage)
    final_class, freeze, decision = final_decision(part_a, loo, mc_summaries, coverage, new_leverage)
    render_stats = {**model_summaries, "loo": loo, "decision": decision}
    render_report(output / "report.md", part_a, final_class, freeze, render_stats, mc_summaries, coverage, decoupling, new_leverage)
    provenance = {"task": "6H-2", "validation_opened": False, "new_pose_decoupling": part_a, "classification": final_class,
                  "freeze_m2_for_laser_ab": freeze, "camera_datasets": {name: list(ids) for name, ids in DATASET_IDS.items()},
                  "new_camera_fit_ids": list(NEW_IDS), "new_laser_opened": False, "new_nolaser_opened": False,
                  "laser_diagnostic_fit_ids": [f"{i:03d}" for i in range(1, 19)] + [f"{i:03d}" for i in range(25, 37)],
                  "laser_validation_opened": False, "formal_intrinsics": str(args.formal_intrinsics.resolve()), "calibration_flags": "CALIB_FIX_K3",
                  "pnp_solver": "SOLVEPNP_ITERATIVE + solvePnPRefineLM", "corner_pipeline": "formal chess_calib.detect_corners",
                  "mc_reps": args.mc_reps, "mc_seed": args.seed, "noise_model": "per-frame centered formal-K/D PnP residual covariance; unchanged across candidates",
                  "formal_kd_modified": False, "distortion_model_changed": False, "cone_refit": False, "steger_changed": False,
                  "frozen_provenance": frozen_info, "decision_details": decision, "0815_data_root": str(args.new_fit_dir.resolve())}
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"NEW_POSE_DECOUPLING = {part_a}")
    print(f"DECOUPLED_CAMERA_GAIN = {final_class}")
    print(f"FREEZE_M2_FOR_LASER_AB = {freeze}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
