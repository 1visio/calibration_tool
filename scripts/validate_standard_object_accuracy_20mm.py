#!/usr/bin/env python3
"""Run the frozen C0/C1 standard-object acceptance for the 20 mm set.

The calculation is intentionally a thin wrapper around
``validate_standard_object_accuracy.py`` so that the reconstruction, frozen
PCA/spline, point selection, and error definitions stay identical to the
50 mm acceptance.  Only the eight input directories, nominal height, position
labels, and output directory differ.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import validate_standard_object_accuracy as base  # noqa: E402
import audit_board_coordinate_residual as board  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402


STANDARD_DIRECTORIES = (
    "frame_012686_measure",
    "frame_011317_measure",
    "frame_009614_measure",
    "frame_008310_measure",
    "frame_007020_measure",
    "frame_005772_measure",
    "frame_004021_measure",
    "frame_000974_measure",
)
NOMINAL_HEIGHT_MM = 20.0
ACCURACY_TARGET_MM = base.ACCURACY_TARGET_MM
DEFAULT_STANDARD_ROOT = base.DEFAULT_STANDARD_ROOT
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_PROVENANCE = base.DEFAULT_PROVENANCE
DEFAULT_FORMAL_CONE = base.DEFAULT_FORMAL_CONE
DEFAULT_MEASUREMENT_CONFIG = base.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0817" / "standard_object_accuracy_20mm"


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


def cache_point_errors(
    result: dict[str, Any],
    item: Mapping[str, Any],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
    spline: Any,
    nominal_height_mm: float,
) -> None:
    for model_id in ("C0", "C1_4k"):
        height_uv = np.asarray(item["height_uv"], dtype=np.float64)
        baseline_uv = np.asarray(item["baseline_uv"], dtype=np.float64)
        height_pair = base.reconstruct_pair(
            height_uv, calibration, reconstruction_params, c1_model, spline
        )
        baseline_pair = base.reconstruct_pair(
            baseline_uv, calibration, reconstruction_params, c1_model, spline
        )
        suffix = "c0" if model_id == "C0" else "c1"
        height_valid = height_pair[f"valid_{suffix}"]
        baseline_valid = baseline_pair[f"valid_{suffix}"]
        measured = base.relative_height_result(
            baseline_pair[f"ground_{suffix}"][baseline_valid],
            height_pair[f"ground_{suffix}"][height_valid],
            nominal_height_mm,
        )
        result["models"][model_id]["_errors"] = measured["errors"]


def region_row(regional: Sequence[Mapping[str, Any]], model: str, name: str) -> Mapping[str, Any]:
    return next(row for row in regional if row.get("record_type") == "region" and row.get("model") == model and row.get("name") == name)


def render_report(
    standard_root: Path,
    output_dir: Path,
    c1_model_path: Path,
    c1_model_sha256: str,
    parameter_sha256: str,
    c0_info: Mapping[str, Any],
    position_rows: Sequence[Mapping[str, Any]],
    regional: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    full_fov: str,
    production: str,
    notes: Sequence[str],
) -> str:
    c0 = summary["C0"]
    c1 = summary["C1_4k"]
    lines = [
        "# Frozen C0 / C1_4k 20 mm standard-object full-FOV acceptance",
        "",
        f"`FULL_FOV_ACCURACY = {full_fov}`",
        f"`C1_PRODUCTION = {production}`",
        "",
        "## Scope and frozen boundary",
        "",
        f"- 仅读取 `{standard_root}` 下明确指定的 8 个标准件目录；保留 8 个独立位置，按 height_points 的 v 中位数排序。",
        "- 未读取 laser-plane Validation（019–024、037–040），未重新拟合 K/D 或 Cone，未训练新 correction。",
        "- C0 = Frozen Circular Cone；C1 = Frozen `C1_4k`，即 `lambda_cone + F(s)`；PCA s、knots、penalty 和 region definition 均未修改。",
        "- C0/C1 使用完全相同的既有 `laser_center.csv` 及对应 baseline/height `u,v` 子集；本轮没有重新提取 laser center。",
        "- 工程真值固定为 **20.000 mm**，由用户提供；四个目录的 `result.json` 均未包含 nominal 字段。",
        "",
        "## Frozen provenance",
        "",
        f"- C1 model: `{c1_model_path}`",
        f"- C1 model SHA-256: `{c1_model_sha256}`",
        f"- C1 parameter SHA-256: `{parameter_sha256}`",
        f"- Frozen C0 provenance SHA-256: `{c0_info.get('provenance_sha256', '')}`",
        f"- Frozen formal Cone SHA-256: `{c0_info.get('formal_cone_sha256', '')}`",
        "",
        "## Eight position results",
        "",
        "误差定义：`(height point Zg - fitted local baseline Zg) - 20.000 mm`；Bias 为带符号均值，P95/Max 为绝对误差。",
        "",
        "| position | directory | region | v median | model | n | Bias | MAE | RMSE | P95 | Max abs | <=0.2 mm |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in position_rows:
        lines.append(
            f"| {row['position']} | {row['directory']} | {row['region']} | {fmt(row['v_median_px'], 1)} | {row['model']} | {row['point_count']} | "
            f"{fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | {fmt(row['p95_abs_mm'])} | {fmt(row['max_abs_mm'])} | {row['within_0_2mm']} |"
        )
    lines.extend(
        [
            "",
            "## Region and full-field consistency",
            "",
            "| model | global RMSE | global MAE | worst-position RMSE | worst-position P95 | position Bias range | position RMSE range | all positions <=0.2 mm |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
            f"| C0 | {fmt(c0['rmse_mm'])} | {fmt(c0['mae_mm'])} | {fmt(c0['worst_position_rmse_mm'])} | {fmt(c0['worst_position_p95_abs_mm'])} | {fmt(c0['position_bias_range_mm'])} | {fmt(c0['position_rmse_range_mm'])} | {c0['all_positions_within_0_2mm']} |",
            f"| C1_4k | {fmt(c1['rmse_mm'])} | {fmt(c1['mae_mm'])} | {fmt(c1['worst_position_rmse_mm'])} | {fmt(c1['worst_position_p95_abs_mm'])} | {fmt(c1['position_bias_range_mm'])} | {fmt(c1['position_rmse_range_mm'])} | {c1['all_positions_within_0_2mm']} |",
            "",
            "| region | C0 RMSE | C1 RMSE | C0 P95 | C1 P95 | C1 RMSE change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("Top", "Middle", "Bottom"):
        r0 = region_row(regional, "C0", name)
        r1 = region_row(regional, "C1_4k", name)
        lines.append(
            f"| {name} | {fmt(r0['rmse_mm'])} | {fmt(r1['rmse_mm'])} | {fmt(r0['p95_abs_mm'])} | {fmt(r1['p95_abs_mm'])} | {fmt(pct_change(r0['rmse_mm'], r1['rmse_mm']), 3)}% |"
        )
    lines.extend(
        [
            "",
            f"- C1 global RMSE change vs C0: **{fmt(pct_change(c0['rmse_mm'], c1['rmse_mm']), 3)}%**；global MAE change: **{fmt(pct_change(c0['mae_mm'], c1['mae_mm']), 3)}%**。负值表示改善。",
            f"- C1 worst-position RMSE change: **{fmt(pct_change(c0['worst_position_rmse_mm'], c1['worst_position_rmse_mm']), 3)}%**；position Bias range change: **{fmt(pct_change(c0['position_bias_range_mm'], c1['position_bias_range_mm']), 3)}%**；position RMSE range change: **{fmt(pct_change(c0['position_rmse_range_mm'], c1['position_rmse_range_mm']), 3)}%**。",
            "",
            "## Decision notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "C1 的生产建议同时考虑所有位置的 Max abs error、最坏位置 RMSE、位置间 Bias/RMSE range 和 Middle 区域是否明显退化，而不是只看 pooled global RMSE。",
            "",
            "## Artifacts",
            "",
            f"- `standard_object_accuracy.csv`、`regional_consistency.csv`：`{output_dir}`",
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

    probes = []
    for directory in STANDARD_DIRECTORIES:
        probe = base.input_record(standard_root, directory, directory, "middle")
        probe["probe_v_median"] = float(np.median(probe["height_uv"][:, 1]))
        probes.append(probe)
    probes.sort(key=lambda item: item["probe_v_median"])
    items = []
    for index, item in enumerate(probes, start=1):
        v_median = item["probe_v_median"]
        region = "top" if v_median < 300.0 else "bottom" if v_median >= 2700.0 else "middle"
        item["label"] = f"P{index:02d}_v{v_median:.1f}"
        item["region"] = region
        items.append(item)

    _, calibration, reconstruction_params, _ = fixed.load_runtime(measurement_config)
    calibration = dict(calibration)
    calibration["laser_model"] = c0_model
    if calibration.get("ground_u_compensation") is not None:
        raise RuntimeError("Frozen C1 model declares no ground_u_compensation, but runtime has one")

    position_results = []
    for item in items:
        result = base.evaluate_position(
            item, calibration, reconstruction_params, c1_model, spline, NOMINAL_HEIGHT_MM
        )
        cache_point_errors(
            result, item, calibration, reconstruction_params, c1_model, spline, NOMINAL_HEIGHT_MM
        )
        position_results.append(result)

    position_rows = base.serialise_position_rows(position_results)
    for row in position_rows:
        row["nominal_height_mm"] = NOMINAL_HEIGHT_MM
    regional, summary = base.regional_rows(position_results)
    c0 = summary["C0"]
    c1 = summary["C1_4k"]
    c1_positions = [row for row in position_rows if row["model"] == "C1_4k"]
    c0_positions = [row for row in position_rows if row["model"] == "C0"]
    c1_all_pass = bool(c1["all_positions_within_0_2mm"])
    c1_global_not_worse = bool(c1["rmse_mm"] <= c0["rmse_mm"] and c1["mae_mm"] <= c0["mae_mm"])
    consistency_improved = bool(
        c1["worst_position_rmse_mm"] < c0["worst_position_rmse_mm"]
        and c1["position_bias_range_mm"] < c0["position_bias_range_mm"]
        and c1["position_rmse_range_mm"] < c0["position_rmse_range_mm"]
    )
    middle_c0 = region_row(regional, "C0", "Middle")
    middle_c1 = region_row(regional, "C1_4k", "Middle")
    middle_not_sacrificed = bool(
        middle_c1["rmse_mm"] <= max(middle_c0["rmse_mm"] * 1.02, middle_c0["rmse_mm"] + 0.005)
    )
    if c1_all_pass and c1_global_not_worse and consistency_improved and middle_not_sacrificed:
        full_fov, production = "PASS", "YES"
    elif c1_all_pass:
        full_fov, production = "PARTIAL", "CONDITIONAL"
    else:
        full_fov, production = "FAIL", "NO"

    worst_c0 = max(c0_positions, key=lambda row: float(row["rmse_mm"]))
    worst_c1 = max(c1_positions, key=lambda row: float(row["rmse_mm"]))
    extrapolated = sum(int(row.get("s_extrapolated_count") or 0) for row in c1_positions)
    notes = [
        f"C1 的 8 个位置中 {sum(bool(row['within_0_2mm']) for row in c1_positions)}/8 个满足 {ACCURACY_TARGET_MM:.1f} mm Max abs error 目标；C0 为 {sum(bool(row['within_0_2mm']) for row in c0_positions)}/8。",
        f"C1 最坏位置为 {worst_c1['position']}（{worst_c1['directory']}），RMSE {fmt(worst_c1['rmse_mm'])} mm；C0 最坏位置为 {worst_c0['position']}，RMSE {fmt(worst_c0['rmse_mm'])} mm。",
        f"最坏位置 RMSE、位置 Bias range 和位置 RMSE range 均{'改善' if consistency_improved else '未同时改善'}；Middle 区域 {'未出现明显退化' if middle_not_sacrificed else '出现明显退化'}。",
        f"8 个位置中 C1 的 frozen s domain 外推点数合计为 {extrapolated}；外推由冻结模型定义允许，未在本轮调整。",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(output_dir / "standard_object_accuracy.csv", position_rows)
    base.write_csv(output_dir / "regional_consistency.csv", regional)
    (output_dir / "report.md").write_text(
        render_report(
            standard_root,
            output_dir,
            c1_model_path,
            c1_model_sha256,
            str(c1_model["parameter_sha256"]),
            c0_info,
            position_rows,
            regional,
            summary,
            full_fov,
            production,
            notes,
        ),
        encoding="utf-8",
    )
    print(
        base.json.dumps(
            {
                "output_dir": str(output_dir),
                "FULL_FOV_ACCURACY": full_fov,
                "C1_PRODUCTION": production,
                "nominal_height_mm": NOMINAL_HEIGHT_MM,
                "c1_model_sha256": c1_model_sha256,
                "c1_parameter_sha256": c1_model["parameter_sha256"],
                "validation_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run(parse_args())
