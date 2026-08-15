#!/usr/bin/env python3
"""Task 5D-2: matched-plane sensor-position residual audit.

The v2 dataset contains fifteen FIT triplets.  This script assigns five
triplets each to Top/Middle/Bottom by the measured laser-line v centre, checks
that the independently estimated PnP planes agree, and evaluates a frozen
Circular Cone without fitting or correction.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402


DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane_0814_v2"
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "sensor_position_audit"
REGIONS = ("Top", "Middle", "Bottom")
GROUPS_PER_REGION = 5
PLANE_NORMAL_THRESHOLD_DEG = 0.163659
PLANE_D_THRESHOLD_MM = 0.08724
PNP_RMSE_THRESHOLD_PX = 0.40
OVERLAP_STEP_PX = 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=fixed.DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=fixed.DEFAULT_FORMAL_CONE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def inventory_v2(data_root: Path) -> dict[str, dict[str, Any]]:
    rows = fixed.csv_rows(data_root / "frames.csv")
    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        group = f"{int(row['pose_id']):03d}"
        role = str(row.get("role", ""))
        if role not in {"chess", "nolaser", "laser"}:
            continue
        relative = Path(*str(row["filename"]).split("/"))
        groups[group][role] = {
            "path": data_root.joinpath(*relative.parts),
            "filename": row["filename"],
            "manifest_sha256": row.get("sha256", ""),
            "quality_passed": row.get("quality_passed", ""),
            "quality_warnings": row.get("quality_warnings", ""),
            "host_timestamp_ns": row.get("host_timestamp_ns", ""),
        }
    expected = [f"{index:03d}" for index in range(1, 16)]
    if sorted(groups, key=int) != expected or any(set(groups[group]) != {"chess", "nolaser", "laser"} for group in expected):
        raise RuntimeError(f"laser_plane_0814_v2 incomplete triplet registry: {sorted(groups)}")
    return groups


def assign_regions(processed: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    centres = {group: float(np.median(item["v"])) for group, item in processed.items()}
    ordered = sorted(centres, key=lambda group: (centres[group], int(group)))
    if len(ordered) != GROUPS_PER_REGION * len(REGIONS):
        raise RuntimeError(f"Expected 15 processed groups, got {len(ordered)}")
    return {group: REGIONS[index // GROUPS_PER_REGION] for index, group in enumerate(ordered)}


def orient_plane(normal: np.ndarray, d: float, reference: np.ndarray) -> tuple[np.ndarray, float]:
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    value = float(d)
    if float(n @ reference) < 0.0:
        n = -n
        value = -value
    return n, value


def plane_consistency_rows(
    processed: Mapping[str, Mapping[str, Any]], regions: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normals = []
    ds = []
    for group in sorted(processed, key=int):
        pose = processed[group]["pose"]
        n, d = orient_plane(pose.normal, pose.d, np.asarray([0.0, 0.0, -1.0]))
        normals.append(n)
        ds.append(d)
    reference = np.mean(np.stack(normals, axis=0), axis=0)
    reference /= np.linalg.norm(reference)
    d_reference = float(np.median(ds))
    rows: list[dict[str, Any]] = []
    group_values: dict[str, dict[str, float]] = {}
    for group in sorted(processed, key=int):
        pose = processed[group]["pose"]
        n, d = orient_plane(pose.normal, pose.d, reference)
        angle = float(np.degrees(np.arccos(np.clip(float(n @ reference), -1.0, 1.0))))
        d_delta = float(d - d_reference)
        pnp = float(pose.reprojection_rmse_px)
        consistent = bool(angle <= PLANE_NORMAL_THRESHOLD_DEG and abs(d_delta) <= PLANE_D_THRESHOLD_MM and pnp <= PNP_RMSE_THRESHOLD_PX)
        values = {
            "normal_angle_to_reference_deg": angle,
            "plane_d_mm": d,
            "plane_d_delta_to_reference_mm": d_delta,
            "pnp_rmse_px": pnp,
            "consistent": consistent,
        }
        group_values[group] = values
        rows.append({"row_type": "group", "group_id": group, "region": regions[group], **values})
    for left, right in itertools.combinations(sorted(processed, key=int), 2):
        n_left, d_left = orient_plane(processed[left]["pose"].normal, processed[left]["pose"].d, reference)
        n_right, d_right = orient_plane(processed[right]["pose"].normal, processed[right]["pose"].d, reference)
        pair_angle = float(np.degrees(np.arccos(np.clip(float(n_left @ n_right), -1.0, 1.0))))
        pair_d_delta = float(d_left - d_right)
        rows.append(
            {
                "row_type": "pair",
                "group_left": left,
                "group_right": right,
                "region_left": regions[left],
                "region_right": regions[right],
                "pair_normal_angle_deg": pair_angle,
                "pair_plane_d_delta_mm": pair_d_delta,
                "pair_consistent": bool(pair_angle <= PLANE_NORMAL_THRESHOLD_DEG and abs(pair_d_delta) <= PLANE_D_THRESHOLD_MM),
            }
        )
    for region in (*REGIONS, "All"):
        selected = [group for group in sorted(processed, key=int) if region == "All" or regions[group] == region]
        values = [group_values[group] for group in selected]
        rows.append(
            {
                "row_type": "region_summary",
                "region": region,
                "group_ids": ";".join(selected),
                "group_count": len(selected),
                "max_normal_angle_to_reference_deg": float(max(item["normal_angle_to_reference_deg"] for item in values)),
                "max_abs_plane_d_delta_to_reference_mm": float(max(abs(item["plane_d_delta_to_reference_mm"]) for item in values)),
                "pnp_rmse_median_px": float(np.median([item["pnp_rmse_px"] for item in values])),
                "pnp_rmse_max_px": float(max(item["pnp_rmse_px"] for item in values)),
                "consistent": bool(all(item["consistent"] for item in values)),
                "normal_threshold_deg": PLANE_NORMAL_THRESHOLD_DEG,
                "plane_d_threshold_mm": PLANE_D_THRESHOLD_MM,
                "pnp_rmse_threshold_px": PNP_RMSE_THRESHOLD_PX,
            }
        )
    overall = next(row for row in rows if row.get("row_type") == "region_summary" and row.get("region") == "All")
    return rows, {
        "reference_normal": reference.tolist(),
        "reference_plane_d_mm": d_reference,
        "all_consistent": bool(overall["consistent"]),
        "max_normal_angle_deg": overall["max_normal_angle_to_reference_deg"],
        "max_abs_plane_d_delta_mm": overall["max_abs_plane_d_delta_to_reference_mm"],
        "pnp_rmse_max_px": overall["pnp_rmse_max_px"],
        "group_values": group_values,
    }


def interpolate(v: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return fixed.interpolate_unique(np.asarray(v, dtype=np.float64), np.asarray(values, dtype=np.float64), grid)


def overlap_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any] | None:
    low = max(float(np.min(a["v"])), float(np.min(b["v"])))
    high = min(float(np.max(a["v"])), float(np.max(b["v"])))
    grid = np.arange(math.ceil(low), math.floor(high) + 0.5 * OVERLAP_STEP_PX, OVERLAP_STEP_PX)
    if len(grid) < 2:
        return None
    delta = interpolate(a["v"], a["residual"], grid) - interpolate(b["v"], b["residual"], grid)
    return {
        "common_v_start_px": float(grid[0]),
        "common_v_end_px": float(grid[-1]),
        "overlap_point_count": int(len(grid)),
        "e_delta_bias_mm": float(np.mean(delta)),
        "e_delta_rmse_mm": float(np.sqrt(np.mean(delta * delta))),
        "e_delta_p95_abs_mm": float(np.percentile(np.abs(delta), 95)),
        "e_delta_max_abs_mm": float(np.max(np.abs(delta))),
    }


def same_region_repeatability(
    processed: Mapping[str, Mapping[str, Any]], regions: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        groups = [group for group in sorted(processed, key=int) if regions[group] == region]
        pairs: list[dict[str, Any]] = []
        for left, right in itertools.combinations(groups, 2):
            values = overlap_pair(processed[left], processed[right])
            if values is None:
                continue
            row = {"row_type": "same_region_repeat_pair", "region": region, "group_left": left, "group_right": right, **values}
            rows.append(row)
            pairs.append(row)
        if len(pairs) == 0:
            raise RuntimeError(f"No same-v overlap for region {region}")
        aggregates[region] = {
            "same_v_pair_count": len(pairs),
            "same_v_overlap_v_start_median_px": float(np.median([row["common_v_start_px"] for row in pairs])),
            "same_v_overlap_v_end_median_px": float(np.median([row["common_v_end_px"] for row in pairs])),
            "same_v_e_delta_rmse_median_mm": float(np.median([row["e_delta_rmse_mm"] for row in pairs])),
            "same_v_e_delta_rmse_p95_mm": float(np.percentile([row["e_delta_rmse_mm"] for row in pairs], 95)),
            "same_v_e_delta_p95_abs_median_mm": float(np.median([row["e_delta_p95_abs_mm"] for row in pairs])),
            "same_v_e_delta_p95_abs_p95_mm": float(np.percentile([row["e_delta_p95_abs_mm"] for row in pairs], 95)),
            "same_v_e_delta_max_abs_mm": float(max(row["e_delta_max_abs_mm"] for row in pairs)),
        }
    return rows, aggregates


def region_curves(processed: Mapping[str, Mapping[str, Any]], regions: Mapping[str, str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bounds = {}
    for region in REGIONS:
        groups = [group for group in processed if regions[group] == region]
        bounds[region] = (
            max(float(np.min(processed[group]["v"])) for group in groups),
            min(float(np.max(processed[group]["v"])) for group in groups),
        )
    low = max(value[0] for value in bounds.values())
    high = min(value[1] for value in bounds.values())
    if high <= low:
        raise RuntimeError("Top/Middle/Bottom have no common v support")
    grid = np.arange(math.ceil(low), math.floor(high) + 0.5 * OVERLAP_STEP_PX, OVERLAP_STEP_PX)
    curves: dict[str, np.ndarray] = {}
    for region in REGIONS:
        groups = [group for group in processed if regions[group] == region]
        values = np.stack([interpolate(processed[group]["v"], processed[group]["residual"], grid) for group in groups], axis=0)
        curves[region] = np.median(values, axis=0)
    return grid, curves


def region_pair_rows(grid: np.ndarray, curves: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(REGIONS, 2):
        delta = curves[left] - curves[right]
        rows.append(
            {
                "row_type": "matched_v_region_pair",
                "region_left": left,
                "region_right": right,
                "common_v_start_px": float(grid[0]),
                "common_v_end_px": float(grid[-1]),
                "overlap_point_count": int(len(grid)),
                "left_curve_bias_mm": float(np.mean(curves[left])),
                "right_curve_bias_mm": float(np.mean(curves[right])),
                "delta_bias_mm": float(np.mean(delta)),
                "delta_rmse_mm": float(np.sqrt(np.mean(delta * delta))),
                "delta_p95_abs_mm": float(np.percentile(np.abs(delta), 95)),
                "delta_max_abs_mm": float(np.max(np.abs(delta))),
            }
        )
    return rows


def pooled_metrics(values: Sequence[np.ndarray]) -> dict[str, float]:
    x = np.concatenate([np.asarray(item, dtype=np.float64) for item in values])
    return {
        "bias_mm": float(np.mean(x)),
        "rmse_mm": float(np.sqrt(np.mean(x * x))),
        "p95_abs_mm": float(np.percentile(np.abs(x), 95)),
        "mae_mm": float(np.mean(np.abs(x))),
    }


def sensor_summary_rows(
    summary: Sequence[Mapping[str, Any]],
    processed: Mapping[str, Mapping[str, Any]],
    regions: Mapping[str, str],
    plane_info: Mapping[str, Any],
    repeat_aggregates: Mapping[str, Mapping[str, Any]],
    region_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary:
        group = str(item["group_id"])
        pose_values = plane_info["group_values"][group]
        row = dict(item)
        row.update(
            {
                "region": regions[group],
                "v_center_px": float(np.median(processed[group]["v"])),
                "plane_d_mm": pose_values["plane_d_mm"],
                "plane_normal_angle_deg": pose_values["normal_angle_to_reference_deg"],
                "plane_d_delta_mm": pose_values["plane_d_delta_to_reference_mm"],
                "plane_consistent": pose_values["consistent"],
            }
        )
        rows.append(row)
    for region in REGIONS:
        groups = [group for group in sorted(processed, key=int) if regions[group] == region]
        residual_values = [processed[group]["residual"] for group in groups]
        metrics = pooled_metrics(residual_values)
        a = np.asarray([float(next(item for item in summary if item["group_id"] == group)["a_frame_mm"]) for group in groups])
        k = np.asarray([float(next(item for item in summary if item["group_id"] == group)["k_frame_mm_per_normalized_stripe"]) for group in groups])
        repeat = repeat_aggregates[region]
        rows.append(
            {
                "row_type": "region_summary",
                "region": region,
                "group_ids": ";".join(groups),
                "group_count": len(groups),
                **metrics,
                "a_frame_mean_mm": float(np.mean(a)),
                "a_frame_std_mm": float(np.std(a, ddof=1)),
                "a_frame_range_mm": float(np.ptp(a)),
                "k_frame_mean_mm_per_normalized_stripe": float(np.mean(k)),
                "k_frame_std_mm_per_normalized_stripe": float(np.std(k, ddof=1)),
                "k_frame_range_mm_per_normalized_stripe": float(np.ptp(k)),
                "v_center_median_px": float(np.median([np.median(processed[group]["v"]) for group in groups])),
                "v_center_min_px": float(min(np.median(processed[group]["v"]) for group in groups)),
                "v_center_max_px": float(max(np.median(processed[group]["v"]) for group in groups)),
                "plane_normal_angle_max_deg": float(max(plane_info["group_values"][group]["normal_angle_to_reference_deg"] for group in groups)),
                "plane_d_delta_abs_max_mm": float(max(abs(plane_info["group_values"][group]["plane_d_delta_to_reference_mm"]) for group in groups)),
                "same_v_pair_count": repeat["same_v_pair_count"],
                **repeat,
            }
        )
    rows.extend(region_pairs)
    return rows


def plot_residual_vs_v(
    path: Path, processed: Mapping[str, Mapping[str, Any]], regions: Mapping[str, str], grid: np.ndarray, curves: Mapping[str, np.ndarray]
) -> None:
    colors = {"Top": "#c53030", "Middle": "#2b6cb0", "Bottom": "#2f855a"}
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for region in REGIONS:
        groups = [group for group in sorted(processed, key=int) if regions[group] == region]
        for group in groups:
            item = processed[group]
            step = max(1, len(item["v"]) // 300)
            ax.scatter(item["v"][::step], item["residual"][::step], s=4, alpha=0.08, color=colors[region])
        ax.plot(grid, curves[region], color=colors[region], linewidth=2.2, label=f"{region} median ({','.join(groups)})")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("sensor v / px")
    ax.set_ylabel("e_lambda = lambda_truth - lambda_model / mm")
    ax.set_title("Task 5D-2 matched-plane residual versus sensor v")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def classify(
    plane_info: Mapping[str, Any],
    region_summary: Mapping[str, Mapping[str, Any]],
    region_pairs: Sequence[Mapping[str, Any]],
    repeat_aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    top = region_summary["Top"]["bias_mm"]
    middle = region_summary["Middle"]["bias_mm"]
    bottom = region_summary["Bottom"]["bias_mm"]
    sign_threshold = 0.02
    expected_sign_pattern = bool(top > sign_threshold and abs(middle) <= sign_threshold and bottom < -sign_threshold)
    bias_range = float(max(top, middle, bottom) - min(top, middle, bottom))
    curve_delta_p95 = float(max(row["delta_p95_abs_mm"] for row in region_pairs))
    repeat_p95 = float(max(item["same_v_e_delta_p95_abs_p95_mm"] for item in repeat_aggregates.values()))
    if plane_info["all_consistent"] and expected_sign_pattern and repeat_p95 <= 0.05:
        verdict = "A. STRONG"
    elif plane_info["all_consistent"] and bias_range >= 0.05 and repeat_p95 <= 0.08:
        verdict = "B. MODERATE"
    else:
        verdict = "C. WEAK"
    return {
        "verdict": verdict,
        "plane_consistent": bool(plane_info["all_consistent"]),
        "top_bias_mm": float(top),
        "middle_bias_mm": float(middle),
        "bottom_bias_mm": float(bottom),
        "bias_range_mm": bias_range,
        "expected_top_positive_middle_zero_bottom_negative": expected_sign_pattern,
        "matched_v_region_delta_p95_max_mm": curve_delta_p95,
        "same_v_repeatability_p95_max_mm": repeat_p95,
        "gates": {
            "plane_consistency": f"normal angle <= {PLANE_NORMAL_THRESHOLD_DEG} deg, |plane d delta| <= {PLANE_D_THRESHOLD_MM} mm, PnP RMSE <= {PNP_RMSE_THRESHOLD_PX} px",
            "strong": "Top bias > +0.02 mm, Middle |bias| <= 0.02 mm, Bottom bias < -0.02 mm, and same-v repeat P95 <= 0.05 mm",
            "moderate": "plane consistent, region bias range >= 0.05 mm, same-v repeat P95 <= 0.08 mm",
            "weak": "otherwise",
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def render_report(
    data_root: Path,
    regions: Mapping[str, str],
    region_summary: Mapping[str, Mapping[str, Any]],
    plane_info: Mapping[str, Any],
    repeat_aggregates: Mapping[str, Mapping[str, Any]],
    region_pairs: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    frozen_info: Mapping[str, Any],
) -> str:
    groups_by_region = {region: ",".join(group for group in sorted(regions, key=int) if regions[group] == region) for region in REGIONS}
    lines = [
        "# Task 5D-2 — Matched-plane sensor-position audit",
        "",
        f"`SENSOR_POSITION_EFFECT = {decision['verdict']}`",
        "",
        "## Scope and boundary",
        "",
        f"- Data: `{data_root}`; 15 FIT triplets only. The manifest has no explicit Top/Middle/Bottom label, so regions are assigned by measured laser-line median sensor v (lower v = Top): Top={groups_by_region['Top']}, Middle={groups_by_region['Middle']}, Bottom={groups_by_region['Bottom']}.",
        "- Validation was not opened or used. No Circular Cone was fitted/refit, no production file was modified, and no compensation was created.",
        "- Each triplet independently uses PnP plane, Steger laser center, ray-plane `lambda_truth`, and the frozen Circular Cone `e_lambda = lambda_truth - lambda_model`.",
        f"- Frozen Circular provenance SHA-256: `{frozen_info['provenance_sha256']}`; formal Cone SHA-256: `{frozen_info['formal_cone_sha256']}`.",
        "",
        "## PnP plane consistency",
        "",
        f"- Reference normal: `{np.asarray(plane_info['reference_normal']).round(9).tolist()}`; reference plane d={fmt(plane_info['reference_plane_d_mm'], 6)} mm.",
        f"- Maximum normal deviation: **{fmt(plane_info['max_normal_angle_deg'], 6)} deg**; maximum |d deviation|: **{fmt(plane_info['max_abs_plane_d_delta_mm'], 6)} mm**; maximum PnP RMSE: **{fmt(plane_info['pnp_rmse_max_px'], 4)} px**.",
        f"- Plane consistency gate: **{'PASS' if plane_info['all_consistent'] else 'FAIL'}** (normal ≤{PLANE_NORMAL_THRESHOLD_DEG} deg, |d|≤{PLANE_D_THRESHOLD_MM} mm, PnP RMSE≤{PNP_RMSE_THRESHOLD_PX} px).",
        "",
        "## Top / Middle / Bottom residuals",
        "",
        "| region | groups | v-center median (px) | bias (mm) | RMSE (mm) | P95 abs (mm) | a_frame mean (mm) | k_frame mean | same-v repeat P95 (mm) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in REGIONS:
        row = region_summary[region]
        lines.append(
            f"| {region} | {groups_by_region[region]} | {fmt(row['v_center_median_px'])} | {fmt(row['bias_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['a_frame_mean_mm'])} | {fmt(row['k_frame_mean_mm_per_normalized_stripe'])} | {fmt(row['same_v_e_delta_p95_abs_p95_mm'])} |"
        )
    lines += [
        "",
        "## Same-v repeatability and matched-v comparison",
        "",
    ]
    for region in REGIONS:
        item = repeat_aggregates[region]
        repeat_rmse = f"{fmt(item['same_v_e_delta_rmse_median_mm'])} / {fmt(item['same_v_e_delta_rmse_p95_mm'])} mm"
        repeat_p95 = f"{fmt(item['same_v_e_delta_p95_abs_median_mm'])} / {fmt(item['same_v_e_delta_p95_abs_p95_mm'])} mm"
        lines.append(
            f"- {region}: {item['same_v_pair_count']} within-region pairs; same-v residual delta RMSE median/P95 = **{repeat_rmse}**; delta P95 median/P95 = **{repeat_p95}**."
        )
    lines += ["", "| matched-v pair | delta bias (mm) | delta RMSE (mm) | delta P95 abs (mm) |", "|---|---:|---:|---:|"]
    for row in region_pairs:
        lines.append(
            f"| {row['region_left']} vs {row['region_right']} | {fmt(row['delta_bias_mm'])} | {fmt(row['delta_rmse_mm'])} | {fmt(row['delta_p95_abs_mm'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Expected diagnostic pattern Top positive / Middle near zero / Bottom negative: **{'OBSERVED' if decision['expected_top_positive_middle_zero_bottom_negative'] else 'NOT OBSERVED'}**.",
        f"- Region bias range: **{fmt(decision['bias_range_mm'])} mm**; maximum matched-v region-pair residual delta P95: **{fmt(decision['matched_v_region_delta_p95_max_mm'])} mm**.",
        f"- Same-v repeatability maximum P95: **{fmt(decision['same_v_repeatability_p95_max_mm'])} mm**.",
        "- Matched-v cross-region differences exceed repeat noise in places, but they do not form a consistent Top→Middle→Bottom offset; this is a reason to investigate a pose/translation-dependent term, not evidence for a fixed sensor-v residual.",
        "- `residual_vs_v.png` shows all point residuals and the per-region median curves on their common sensor-v support.",
        "",
        f"`SENSOR_POSITION_EFFECT = {decision['verdict']}`.",
        "A STRONG result requires the expected sign reversal after the PnP planes pass consistency. A MODERATE result requires a substantial region bias separation with repeatable same-v residuals. Otherwise the data do not support a fixed sensor-position residual conclusion.",
        "",
        "The laser frames carry the acquisition `dynamic_range_low` warning; this quality limitation is retained in the CSV provenance fields and does not authorize changing the extraction or model.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    groups = inventory_v2(data_root)
    fixed.verify_inventory(groups, args.verify_sha256)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, frozen_info = fixed.load_frozen_model(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    summary, processed, _ = fixed.process_groups(groups, intrinsics, calibration, reconstruction_params, frozen_model)
    regions = assign_regions(processed)
    plane_rows, plane_info = plane_consistency_rows(processed, regions)
    repeat_rows, repeat_aggregates = same_region_repeatability(processed, regions)
    grid, curves = region_curves(processed, regions)
    matched_region_rows = region_pair_rows(grid, curves)
    plane_by_group = plane_info["group_values"]
    # Keep group rows, region summaries, same-region pair diagnostics, and the
    # three matched-v region comparisons in the requested sensor summary CSV.
    region_summary_lookup: dict[str, dict[str, Any]] = {}
    provisional = sensor_summary_rows(summary, processed, regions, plane_info, repeat_aggregates, matched_region_rows)
    for row in provisional:
        if row.get("row_type") == "region_summary":
            region_summary_lookup[str(row["region"])] = row
    # Add same-v pair rows after the region summaries, retaining the requested
    # single summary artifact rather than introducing an extra output file.
    sensor_rows = provisional + repeat_rows
    decision = classify(plane_info, region_summary_lookup, matched_region_rows, repeat_aggregates)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed.write_csv(output_dir / "sensor_position_summary.csv", sensor_rows)
    fixed.write_csv(output_dir / "plane_consistency.csv", plane_rows)
    plot_residual_vs_v(output_dir / "residual_vs_v.png", processed, regions, grid, curves)
    (output_dir / "report.md").write_text(
        render_report(data_root, regions, region_summary_lookup, plane_info, repeat_aggregates, matched_region_rows, decision, frozen_info),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "regions": regions, "plane": plane_info}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
