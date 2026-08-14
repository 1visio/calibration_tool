#!/usr/bin/env python3
"""Evaluate frozen FIT-derived Circular Cone candidates on VALIDATION only.

No parameter, weight, threshold, or candidate is selected using validation.
Candidates are read from the FIT-only trust-region output and evaluated through
the public production reconstruction path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyze_circular_cone_parameter_sensitivity as sensitivity  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_FIT_OUTPUT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_nonlinear_fit_trust_region"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_validation_final_evaluation"
)
OUTPUT_NAMES = (
    "cone_validation_candidates.csv",
    "cone_validation_regions.csv",
    "cone_validation_prediction.png",
    "cone_validation_report.md",
    "OUTPUT_FILES.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen FIT-derived candidate evaluation on VALIDATION only."
    )
    parser.add_argument("--data-root", type=Path, default=sensitivity.paired.DEFAULT_DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=sensitivity.paired.DEFAULT_PNP_AUDIT)
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=sensitivity.paired.DEFAULT_MEASUREMENT_CONFIG,
    )
    parser.add_argument("--fit-output", type=Path, default=DEFAULT_FIT_OUTPUT)
    parser.add_argument("--stability-output", type=Path, default=(CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0813" / "cone_fit_stability"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_candidates(path: Path, theta0: np.ndarray) -> dict[str, np.ndarray]:
    rows: dict[str, Mapping[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["parameter"]] = row
    return {
        weighting: np.asarray(
            [float(rows[spec.name][f"final_{weighting}"]) for spec in sensitivity.PARAMETERS],
            dtype=np.float64,
        )
        for weighting in sensitivity.WEIGHTINGS
    }


def add_metrics(row: dict[str, Any], prefix: str, metrics: sensitivity.MetricSet) -> None:
    row[f"{prefix}_bias_mm"] = metrics.bias_mm
    row[f"{prefix}_mae_mm"] = metrics.mae_mm
    row[f"{prefix}_rmse_mm"] = metrics.rmse_mm
    row[f"{prefix}_p95_abs_mm"] = metrics.p95_abs_mm
    row[f"{prefix}_residual_energy"] = metrics.residual_energy


def metric_rows(
    data: sensitivity.PreparedData,
    candidates: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline, v_px, frame_ids = sensitivity.split_arrays(data, "validation")
    global_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for candidate_name, theta in candidates.items():
        exact = sensitivity.evaluate_candidate(theta, data, "validation")
        invalid_count = int(np.count_nonzero(exact.invalid_mask))
        if invalid_count:
            raise RuntimeError(f"Validation candidate has invalid intersections: {candidate_name}")
        for metric_weighting in sensitivity.WEIGHTINGS:
            weights = sensitivity.weights_for(v_px, frame_ids, metric_weighting)
            before = sensitivity.calculate_metrics(baseline, weights)
            after = sensitivity.calculate_metrics(exact.residual_mm, weights)
            row: dict[str, Any] = {
                "candidate": candidate_name,
                "metric_weighting": metric_weighting,
                "split": "validation",
                "region": "global",
                "sample_count": len(baseline),
                "invalid_count": invalid_count,
            }
            add_metrics(row, "before", before)
            add_metrics(row, "after", after)
            row["explained_fraction"] = (
                before.residual_energy - after.residual_energy
            ) / before.residual_energy
            global_rows.append(row)
            for region, v_min, v_max in sensitivity.region_definitions()[1:]:
                assert v_min is not None and v_max is not None
                mask = (v_px >= v_min) & (v_px < v_max)
                if not np.any(mask):
                    continue
                before_region = sensitivity.calculate_metrics(baseline[mask], weights[mask])
                after_region = sensitivity.calculate_metrics(exact.residual_mm[mask], weights[mask])
                region_row: dict[str, Any] = {
                    "candidate": candidate_name,
                    "metric_weighting": metric_weighting,
                    "split": "validation",
                    "region": region,
                    "v_min_px": v_min,
                    "v_max_px": v_max,
                    "sample_count": int(np.count_nonzero(mask)),
                    "invalid_count": int(np.count_nonzero(exact.invalid_mask[mask])),
                }
                add_metrics(region_row, "before", before_region)
                add_metrics(region_row, "after", after_region)
                region_row["explained_fraction"] = (
                    before_region.residual_energy - after_region.residual_energy
                ) / before_region.residual_energy
                region_rows.append(region_row)
    return global_rows, region_rows


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: sensitivity.csv_value(row[field]) for field in fields})


def save_plot(
    path: Path,
    data: sensitivity.PreparedData,
    candidates: Mapping[str, np.ndarray],
) -> None:
    baseline, v_px, _ = sensitivity.split_arrays(data, "validation")
    colors = {"point_equal": "#2b6cb0", "frame_equal": "#2f855a", "v_region_equal": "#c05621"}
    figure, axis = plt.subplots(figsize=(10.0, 5.4))
    x, y = [], []
    bins = np.floor(v_px / 30.0).astype(int)
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        x.append((bin_id + 0.5) * 30.0)
        y.append(float(np.median(baseline[mask])))
    axis.plot(x, y, color="#222222", linewidth=2.2, label="baseline")
    for candidate_name, theta in candidates.items():
        exact = sensitivity.evaluate_candidate(theta, data, "validation")
        values = []
        for bin_id in np.unique(bins):
            mask = bins == bin_id
            values.append(float(np.median(exact.residual_mm[mask])))
        axis.plot(x, values, color=colors[candidate_name], linewidth=1.7, label=candidate_name)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axvline(300.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axis.axvline(2700.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axis.set_xlabel("laser stripe row v / px")
    axis.set_ylabel("validation vertical residual median / mm")
    axis.set_title("Frozen FIT-derived candidates on VALIDATION 011–013")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    data: sensitivity.PreparedData,
    candidates: Mapping[str, np.ndarray],
    global_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    stability_path: Path,
    cone_path: Path,
    cone_hash: str,
) -> str:
    lines = [
        "# Circular Cone frozen validation evaluation",
        "",
        "**VALIDATION_ONLY = TRUE**",
        "",
        "本报告只评价 FIT-only 产生的三个冻结 candidate；VALIDATION 没有参与 candidate、weight、step、阈值或任何选择。",
        "没有执行 validation reoptimization，也没有写回正式 Cone。",
        "",
        "## Candidate and global result",
        "",
        "| candidate | metric weighting | before RMSE | after RMSE | before P95 | after P95 | explained fraction | invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['candidate']} | {row['metric_weighting']} | {float(row['before_rmse_mm']):.6f} | "
            f"{float(row['after_rmse_mm']):.6f} | {float(row['before_p95_abs_mm']):.6f} | "
            f"{float(row['after_p95_abs_mm']):.6f} | {float(row['explained_fraction']):.6f} | "
            f"{int(row['invalid_count'])} |"
        )
    lines += [
        "",
        "## Top / middle / bottom",
        "",
        "| candidate | metric weighting | region | before RMSE | after RMSE | explained fraction |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in region_rows:
        if row["region"] not in ("top_0_299", "middle_300_2699", "bottom_2700_2999"):
            continue
        lines.append(
            f"| {row['candidate']} | {row['metric_weighting']} | {row['region']} | "
            f"{float(row['before_rmse_mm']):.6f} | {float(row['after_rmse_mm']):.6f} | "
            f"{float(row['explained_fraction']):.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- 三个 candidate 都来自 FIT；本表只回答它们在冻结 VALIDATION 上是否保持改善。",
        "- 不根据 validation 表现挑选 candidate；point/frame/v_region 三者分别报告。",
        "- 即使 validation 仍改善，也不能消除 FIT 中已经发现的 apex/alpha 参数耦合和 A_z 边界问题。",
        "- stability analysis：`" + str(stability_path) + "`。",
        "",
        "## Provenance / 不变项",
        "",
        f"- VALIDATION frozen-pixel SHA-256：`{data.used_pixel_sha256}`。",
        f"- Formal Cone：`{cone_path}`",
        f"- Formal Cone SHA-256（运行前后相同）：`{cone_hash}`",
        "- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 与正式 Cone 均未修改。",
        "",
    ]
    return "\n".join(lines)


def render_output_files() -> str:
    return "\n".join(
        [
            "# Circular Cone validation outputs",
            "",
            "| 文件 | 内容 | 边界 |",
            "|---|---|---|",
            "| cone_validation_candidates.csv | 三个 FIT candidate 的 global validation 指标 | 不用于选择 candidate |",
            "| cone_validation_regions.csv | validation top/middle/bottom、300px bins、外推区 | 不做 validation 优化 |",
            "| cone_validation_prediction.png | baseline 与冻结 candidate 的 residual-v 曲线 | 只是一阶候选评价图 |",
            "| cone_validation_report.md | 严格隔离与最终 validation 结果 | 不授权写回正式参数 |",
            "| OUTPUT_FILES.md | 输出索引 | 不增加科学证据 |",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    pnp_audit = args.pnp_audit.resolve()
    measurement_config = args.measurement_config.resolve()
    fit_output = args.fit_output.resolve()
    stability_output = args.stability_output.resolve()
    output_dir = args.output_dir.resolve()
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite")

    app_config = sensitivity.load_app_config(measurement_config)
    cone_path = Path(app_config.calibration.laser_model).resolve()
    cone_hash_before = sensitivity.sha256_file(cone_path)
    data, _ = sensitivity.prepare_data(
        data_root,
        pnp_audit,
        measurement_config,
        include_splits=("validation",),
    )
    expected_frames = {f"{index:03d}" for index in range(11, 14)}
    if {frame.frame_id for frame in data.frames} != expected_frames or set(data.split) != {"validation"}:
        raise RuntimeError("Validation evaluation did not load exactly 011-013")
    candidates = read_candidates(fit_output / "cone_nonlinear_fit_candidates.csv", data.theta0)
    global_rows, region_rows = metric_rows(data, candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_paths["cone_validation_candidates.csv"], global_rows)
    write_rows(output_paths["cone_validation_regions.csv"], region_rows)
    save_plot(output_paths["cone_validation_prediction.png"], data, candidates)
    output_paths["cone_validation_report.md"].write_text(
        render_report(data, candidates, global_rows, region_rows, stability_output, cone_path, cone_hash_before),
        encoding="utf-8",
    )
    output_paths["OUTPUT_FILES.md"].write_text(render_output_files(), encoding="utf-8")
    cone_hash_after = sensitivity.sha256_file(cone_path)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Cone changed during validation evaluation")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output files: {sorted(actual_names)}")
    for row in global_rows:
        if row["metric_weighting"] == row["candidate"]:
            print(
                f"{row['candidate']}: validation_rmse={float(row['after_rmse_mm']):.9g}, "
                f"explained={float(row['explained_fraction']):.9g}, invalid={int(row['invalid_count'])}"
            )
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
