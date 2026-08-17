#!/usr/bin/env python3
"""Compare frozen C1 boundary policies on the 20 mm and 50 mm standards.

Policies are evaluated on the same existing laser-center/height/baseline UV
rows.  The C1 parameters remain byte-for-byte frozen; only the handling of
``s`` outside the frozen PCA domain changes:

* raw_extrapolation: the frozen BSpline ``extrapolate=True`` behavior;
* boundary_clamp: evaluate F at the nearest frozen domain boundary;
* fallback_c0: use no correction outside the frozen domain.

The C0 baseline is also emitted for reference, but is not one of the three C1
policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


ACCURACY_TARGET_MM = 0.2
DATASETS = {
    "20mm": {
        "nominal_height_mm": 20.0,
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
        "nominal_height_mm": 50.0,
        "directories": (
            "frame_061303_measure",
            "frame_065292_measure",
            "frame_063995_measure",
            "frame_062878_measure",
        ),
    },
}
POLICIES = ("C0_baseline", "raw_extrapolation", "boundary_clamp", "fallback_c0")
C1_POLICIES = POLICIES[1:]
DEFAULT_STANDARD_ROOT = base.DEFAULT_STANDARD_ROOT
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_PROVENANCE = base.DEFAULT_PROVENANCE
DEFAULT_FORMAL_CONE = base.DEFAULT_FORMAL_CONE
DEFAULT_MEASUREMENT_CONFIG = base.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0817" / "c1_boundary_policy"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-root", type=Path, default=DEFAULT_STANDARD_ROOT)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def finite(value: Any) -> float:
    return base.finite(value)


def fmt(value: Any, digits: int = 6) -> str:
    return base.fmt(value, digits)


def pct_change(before: float, after: float) -> float:
    return base.pct_change(before, after)


def policy_correction(
    s: np.ndarray,
    spline: Any,
    domain_min: float,
    domain_max: float,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(s, dtype=np.float64)
    inside = (values >= domain_min) & (values <= domain_max)
    raw = np.asarray(spline(values), dtype=np.float64)
    if policy == "C0_baseline":
        applied = np.zeros_like(values)
    elif policy == "raw_extrapolation":
        applied = raw
    elif policy == "boundary_clamp":
        applied = np.asarray(spline(np.clip(values, domain_min, domain_max)), dtype=np.float64)
    elif policy == "fallback_c0":
        applied = np.where(inside, raw, 0.0)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return applied, raw, inside


def reconstruct_policy(
    uv: np.ndarray,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
    spline: Any,
    policy: str,
) -> dict[str, Any]:
    lambda_c0, valid_c0 = fixed.lambda_by_input(uv, calibration, reconstruction_params)
    _, _, normalized = base.ground_points_from_lambda(uv, lambda_c0, valid_c0, calibration)
    s = base.pca_s_values(normalized, c1_model)
    pca = c1_model["pca_s"]
    domain_min = float(pca["domain_min"])
    domain_max = float(pca["domain_max"])
    correction, raw_correction, inside = policy_correction(
        s, spline, domain_min, domain_max, policy
    )
    lambdas = np.asarray(lambda_c0, dtype=np.float64) + correction
    min_depth = float(reconstruction_params.min_camera_depth_mm)
    max_depth = float(reconstruction_params.max_camera_depth_mm)
    valid = (
        np.asarray(valid_c0, dtype=bool)
        & np.isfinite(lambdas)
        & (lambdas > 0.0)
        & (lambdas >= min_depth)
        & (lambdas <= max_depth)
    )
    ground, valid_final, _ = base.ground_points_from_lambda(
        uv, lambdas, valid, calibration
    )
    return {
        "ground": ground,
        "valid": valid_final,
        "s": s,
        "inside": inside,
        "correction": correction,
        "raw_correction": raw_correction,
        "lambda_c0": np.asarray(lambda_c0, dtype=np.float64),
        "lambda": lambdas,
    }


def input_items(standard_root: Path, dataset_name: str, directories: Sequence[str]) -> list[dict[str, Any]]:
    probes = []
    for directory in directories:
        item = base.input_record(standard_root, directory, directory, "middle")
        item["probe_v_median"] = float(np.median(item["height_uv"][:, 1]))
        probes.append(item)
    probes.sort(key=lambda item: item["probe_v_median"])
    items = []
    for index, item in enumerate(probes, start=1):
        v_median = item["probe_v_median"]
        region = "top" if v_median < 300.0 else "bottom" if v_median >= 2700.0 else "middle"
        item["dataset"] = dataset_name
        item["label"] = f"P{index:02d}_v{v_median:.1f}"
        item["region"] = region
        items.append(item)
    return items


def evaluate_item(
    item: Mapping[str, Any],
    nominal_height_mm: float,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
    spline: Any,
) -> list[dict[str, Any]]:
    baseline_uv = np.asarray(item["baseline_uv"], dtype=np.float64)
    height_uv = np.asarray(item["height_uv"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        baseline = reconstruct_policy(
            baseline_uv, calibration, reconstruction_params, c1_model, spline, policy
        )
        height = reconstruct_policy(
            height_uv, calibration, reconstruction_params, c1_model, spline, policy
        )
        if int(np.count_nonzero(baseline["valid"])) < 30 or int(np.count_nonzero(height["valid"])) < 30:
            raise RuntimeError(
                f"{item['dataset']} {item['directory']} {policy}: insufficient valid points "
                f"(baseline={baseline['valid'].sum()}, height={height['valid'].sum()})"
            )
        measured = base.relative_height_result(
            baseline["ground"][baseline["valid"]],
            height["ground"][height["valid"]],
            nominal_height_mm,
        )
        errors = np.asarray(measured.pop("errors"), dtype=np.float64)
        metrics = base.metric(errors)
        correction = np.asarray(height["correction"], dtype=np.float64)
        raw_correction = np.asarray(height["raw_correction"], dtype=np.float64)
        inside = np.asarray(height["inside"], dtype=bool)
        rows.append(
            {
                "record_type": "position",
                "dataset": item["dataset"],
                "nominal_height_mm": nominal_height_mm,
                "position": item["label"],
                "directory": item["directory"],
                "region": item["region"],
                "v_median_px": item["probe_v_median"],
                "policy": policy,
                **metrics,
                "height_difference_mae_mm": measured["height_difference_mae_mm"],
                "height_difference_bias_mm": measured["height_difference_bias_mm"],
                "height_difference_rmse_mm": measured["height_difference_rmse_mm"],
                "measured_height_mean_mm": measured["measured_height_mean_mm"],
                "measured_height_median_mm": measured["measured_height_median_mm"],
                "measured_height_std_mm": measured["measured_height_std_mm"],
                "valid_baseline_count": int(np.count_nonzero(baseline["valid"])),
                "valid_height_count": int(np.count_nonzero(height["valid"])),
                "baseline_inlier_count": measured["baseline_inlier_count"],
                "height_inlier_count": measured["height_inlier_count"],
                "s_min": float(np.nanmin(height["s"])),
                "s_max": float(np.nanmax(height["s"])),
                "s_domain_min": float(c1_model["pca_s"]["domain_min"]),
                "s_domain_max": float(c1_model["pca_s"]["domain_max"]),
                "s_extrapolated_count": int(np.count_nonzero(~inside)),
                "s_inside_count": int(np.count_nonzero(inside)),
                "correction_mean_signed_mm": float(np.nanmean(correction)),
                "correction_mean_abs_mm": float(np.nanmean(np.abs(correction))),
                "correction_max_abs_mm": float(np.nanmax(np.abs(correction))),
                "raw_extrapolated_correction_mean_abs_mm": float(
                    np.nanmean(np.abs(raw_correction[~inside]))
                )
                if np.any(~inside)
                else 0.0,
                "raw_extrapolated_correction_max_abs_mm": float(
                    np.nanmax(np.abs(raw_correction[~inside]))
                )
                if np.any(~inside)
                else 0.0,
                "within_0_2mm": bool(metrics["max_abs_mm"] <= ACCURACY_TARGET_MM),
                "center_sha256": item["center_sha256"],
                "baseline_sha256": item["baseline_sha256"],
                "height_sha256": item["height_sha256"],
                "_errors": errors,
            }
        )
    return rows


def aggregate_rows(position_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in position_rows:
        groups[(str(row["dataset"]), str(row["policy"]))].append(row)
    output = list(position_rows)
    for (dataset, policy), rows in groups.items():
        errors = np.concatenate([np.asarray(row["_errors"], dtype=np.float64) for row in rows])
        metrics = base.metric(errors)
        rmses = np.asarray([finite(row["rmse_mm"]) for row in rows], dtype=np.float64)
        p95s = np.asarray([finite(row["p95_abs_mm"]) for row in rows], dtype=np.float64)
        biases = np.asarray([finite(row["bias_mm"]) for row in rows], dtype=np.float64)
        output.append(
            {
                "record_type": "overall",
                "dataset": dataset,
                "nominal_height_mm": rows[0]["nominal_height_mm"],
                "position": "All_positions",
                "directory": "",
                "region": "global",
                "v_median_px": "",
                "policy": policy,
                **metrics,
                "worst_position_rmse_mm": float(np.nanmax(rmses)),
                "worst_position_p95_abs_mm": float(np.nanmax(p95s)),
                "position_bias_range_mm": float(np.nanmax(biases) - np.nanmin(biases)),
                "position_rmse_range_mm": float(np.nanmax(rmses) - np.nanmin(rmses)),
                "all_positions_within_0_2mm": bool(
                    all(bool(row["within_0_2mm"]) for row in rows)
                ),
                "s_extrapolated_count": int(sum(int(row["s_extrapolated_count"]) for row in rows)),
                "s_inside_count": int(sum(int(row["s_inside_count"]) for row in rows)),
                "correction_mean_abs_mm": float(
                    np.nanmean([finite(row["correction_mean_abs_mm"]) for row in rows])
                ),
                "correction_max_abs_mm": float(
                    np.nanmax([finite(row["correction_max_abs_mm"]) for row in rows])
                ),
                "_errors": errors,
            }
        )
    return output


def row_lookup(rows: Sequence[Mapping[str, Any]], dataset: str, policy: str, record_type: str = "overall") -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row.get("record_type") == record_type
        and row.get("dataset") == dataset
        and row.get("policy") == policy
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    value = ""
                elif isinstance(value, (bool, np.bool_)):
                    value = "true" if bool(value) else "false"
                output[key] = value
            writer.writerow(output)


def report_text(
    standard_root: Path,
    output_dir: Path,
    c1_model_path: Path,
    c1_model_sha256: str,
    parameter_sha256: str,
    c0_info: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> str:
    lines = [
        "# Frozen C1_4k boundary-policy comparison",
        "",
        f"`C1_BOUNDARY_POLICY = {decision['C1_BOUNDARY_POLICY']}`",
        f"`RECOMMENDED_POLICY = {decision.get('recommended_policy', 'NONE')}`",
        "",
        "## Scope and frozen boundary",
        "",
        f"- 仅读取 20 mm 与 50 mm 两组标准件目录，输入根目录：`{standard_root}`。",
        "- C0/C1 使用完全相同的既有 `laser_center.csv` 及 baseline/height `u,v` 子集。",
        "- 未读取 Validation（019–024、037–040），未重新拟合 C1、K/D 或 Cone，未修改 knots/penalty，未做 C2。",
        f"- Frozen s-domain：`[{decision['domain_min']:.12g}, {decision['domain_max']:.12g}]`。",
        "- C1 三种策略：`raw_extrapolation`、`boundary_clamp`、`fallback_c0`；另列 `C0_baseline` 作为参考。",
        "",
        "## Frozen provenance",
        "",
        f"- C1 model: `{c1_model_path}`",
        f"- C1 model SHA-256: `{c1_model_sha256}`",
        f"- C1 parameter SHA-256: `{parameter_sha256}`",
        f"- Frozen C0 provenance SHA-256: `{c0_info.get('provenance_sha256', '')}`",
        f"- Frozen formal Cone SHA-256: `{c0_info.get('formal_cone_sha256', '')}`",
        "",
        "## Dataset-level policy summary",
        "",
        "| dataset | policy | global RMSE | global P95 | global Max | worst-position RMSE | worst-position P95 | position Bias range | position RMSE range | all positions <=0.2 mm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for dataset in DATASETS:
        for policy in POLICIES:
            row = row_lookup(rows, dataset, policy)
            lines.append(
                f"| {dataset} | {policy} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} | "
                f"{fmt(row['worst_position_rmse_mm'])} | {fmt(row['worst_position_p95_abs_mm'])} | {fmt(row['position_bias_range_mm'])} | "
                f"{fmt(row['position_rmse_range_mm'])} | {row.get('all_positions_within_0_2mm', '')} |"
            )
    lines.extend(
        [
            "",
            "## Position-level policy results",
            "",
            "详细 CSV 对每个位置给出 Bias、MAE、RMSE、P95、Max abs、超域点数、实际修正幅度和 0.2 mm 判定。下表列出 RMSE/P95/Max abs 及超域点数。",
            "",
            "| dataset | position | directory | v median | policy | RMSE | P95 | Max abs | s outside count | correction mean abs | correction max abs | <=0.2 mm |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        if row.get("record_type") != "position":
            continue
        lines.append(
            f"| {row['dataset']} | {row['position']} | {row['directory']} | {fmt(row['v_median_px'], 1)} | {row['policy']} | "
            f"{fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} | {row['s_extrapolated_count']} | "
            f"{fmt(row['correction_mean_abs_mm'])} | {fmt(row['correction_max_abs_mm'])} | {row['within_0_2mm']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `C1_BOUNDARY_POLICY = {decision['C1_BOUNDARY_POLICY']}`。",
            f"- 推荐策略：`{decision.get('recommended_policy', 'NONE')}`。",
            f"- 同时满足两组全部位置 0.2 mm、安全性门槛的策略：{', '.join(decision.get('safe_policies', [])) or '无'}。",
            f"- 原始 extrapolation 的超域点总数：20 mm = {decision['raw_extrapolated_20mm']}, 50 mm = {decision['raw_extrapolated_50mm']}；它在 20 mm Top 的 worst Max abs = {fmt(decision['raw_top_max_20mm'])} mm。",
            f"- boundary clamp 的两组 worst Max abs：20 mm = {fmt(decision['clamp_worst_max_20mm'])} mm，50 mm = {fmt(decision['clamp_worst_max_50mm'])} mm。",
            f"- fallback C0 的两组 all-position gate：20 mm = {decision['fallback_all_20mm']}，50 mm = {decision['fallback_all_50mm']}。",
            "",
            "该结论只评价边界策略，不代表重新校准或放宽 C1 的 frozen 参数；若没有策略同时通过两组门槛，则说明当前 calibration domain 不足。",
            "",
            "## Artifacts",
            "",
            f"- `boundary_policy_comparison.csv`：`{output_dir}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    standard_root = base.resolve_existing(
        args.standard_root,
        Path(str(args.standard_root).replace("linelascan", "linelaserscan")),
    )
    c1_model_path = base.resolve_existing(args.c1_model)
    provenance_path = base.resolve_existing(args.frozen_provenance)
    formal_cone_path = base.resolve_existing(args.formal_cone)
    measurement_config = base.resolve_existing(args.measurement_config)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    c1_model, c1_model_sha256 = frozen.load_frozen_json(c1_model_path)
    spline = frozen.frozen_spline(c1_model)
    c0_model, c0_info = board.load_frozen_model_checked(provenance_path, formal_cone_path)
    for key in ("axis_unit_camera", "apex_camera_mm", "half_apex_angle_deg", "z_valid_range_mm"):
        if not np.allclose(np.asarray(c0_model[key], dtype=np.float64), np.asarray(c1_model["frozen_cone"]["runtime_model"][key], dtype=np.float64), atol=1.0e-10):
            raise RuntimeError(f"Frozen C0 runtime mismatch in {key}")
    _, calibration, reconstruction_params, _ = fixed.load_runtime(measurement_config)
    calibration = dict(calibration)
    calibration["laser_model"] = c0_model
    if calibration.get("ground_u_compensation") is not None:
        raise RuntimeError("Frozen C1 model declares no ground_u_compensation, but runtime has one")

    position_rows: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []
    for dataset_name, spec in DATASETS.items():
        all_items.extend(input_items(standard_root, dataset_name, spec["directories"]))
    for item in all_items:
        position_rows.extend(
            evaluate_item(
                item,
                float(DATASETS[item["dataset"]]["nominal_height_mm"]),
                calibration,
                reconstruction_params,
                c1_model,
                spline,
            )
        )
    rows = aggregate_rows(position_rows)
    domain_min = float(c1_model["pca_s"]["domain_min"])
    domain_max = float(c1_model["pca_s"]["domain_max"])

    safe_policies: list[str] = []
    for policy in C1_POLICIES:
        valid = True
        for dataset in DATASETS:
            candidate = row_lookup(rows, dataset, policy)
            baseline = row_lookup(rows, dataset, "C0_baseline")
            valid &= bool(candidate["all_positions_within_0_2mm"])
            valid &= bool(candidate["rmse_mm"] <= baseline["rmse_mm"])
            valid &= bool(candidate["worst_position_rmse_mm"] <= baseline["worst_position_rmse_mm"])
            valid &= bool(candidate["position_bias_range_mm"] <= baseline["position_bias_range_mm"])
            valid &= bool(candidate["position_rmse_range_mm"] <= baseline["position_rmse_range_mm"])
        if valid:
            safe_policies.append(policy)
    boundary_decision = "SAFE_GUARD_AVAILABLE" if safe_policies else "CALIBRATION_DOMAIN_INSUFFICIENT"
    recommended = "boundary_clamp" if "boundary_clamp" in safe_policies else (safe_policies[0] if safe_policies else "NONE")
    raw_20 = row_lookup(rows, "20mm", "raw_extrapolation")
    raw_50 = row_lookup(rows, "50mm", "raw_extrapolation")
    clamp_20 = row_lookup(rows, "20mm", "boundary_clamp")
    clamp_50 = row_lookup(rows, "50mm", "boundary_clamp")
    fallback_20 = row_lookup(rows, "20mm", "fallback_c0")
    fallback_50 = row_lookup(rows, "50mm", "fallback_c0")
    raw_top_20 = next(
        row for row in rows if row.get("record_type") == "position" and row.get("dataset") == "20mm" and row.get("policy") == "raw_extrapolation" and row.get("region") == "top"
    )
    decision = {
        "C1_BOUNDARY_POLICY": boundary_decision,
        "recommended_policy": recommended,
        "safe_policies": safe_policies,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "raw_extrapolated_20mm": raw_20["s_extrapolated_count"],
        "raw_extrapolated_50mm": raw_50["s_extrapolated_count"],
        "raw_top_max_20mm": raw_top_20["max_abs_mm"],
        "clamp_worst_max_20mm": clamp_20["max_abs_mm"],
        "clamp_worst_max_50mm": clamp_50["max_abs_mm"],
        "fallback_all_20mm": fallback_20["all_positions_within_0_2mm"],
        "fallback_all_50mm": fallback_50["all_positions_within_0_2mm"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "boundary_policy_comparison.csv", rows)
    (output_dir / "report.md").write_text(
        report_text(
            standard_root,
            output_dir,
            c1_model_path,
            c1_model_sha256,
            str(c1_model["parameter_sha256"]),
            c0_info,
            rows,
            decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "C1_BOUNDARY_POLICY": boundary_decision,
                "recommended_policy": recommended,
                "safe_policies": safe_policies,
                "c1_model_sha256": c1_model_sha256,
                "c1_parameter_sha256": c1_model["parameter_sha256"],
                "validation_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run(parse_args())
