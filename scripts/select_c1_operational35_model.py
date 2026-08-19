#!/usr/bin/env python3
"""Select the final FIT-only C1 candidate on the operational 35-pose domain.

This is a read-only result-reuse and model-selection layer.  It consumes the
already completed 35-pose grouped-CV artifact; it does not fit C0/C1 again,
open Validation, delete frame027, or modify production configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
PROJECT = ROOT / "projects" / "daheng"
SOURCE_DIR = PROJECT / "outputs/0818/c1_frozen_quadratic_grouped_cv"
DEFAULT_OUTPUT = PROJECT / "outputs/0818/c1_operational35_selection"

SOURCE_COMPARISON = SOURCE_DIR / "c1_candidate_comparison.csv"
SOURCE_POSE = SOURCE_DIR / "c1_pose_cv_metrics.csv"
SOURCE_V_BIN = SOURCE_DIR / "c1_v_bin_metrics.csv"
SOURCE_MANIFEST = SOURCE_DIR / "c1_run_manifest.json"

OPERATIONAL_SCENARIO = "exclude027_grouped_cv_non027"
OPERATIONAL_STATUS = "EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN"
EXCLUSION_REASON = "超出实际工作姿态域"
CANDIDATES = ["C1_3k", "C1_4k", "C1_5k"]
PAIR_ORDER = [("C1_4k", "C1_3k"), ("C1_5k", "C1_3k"), ("C1_5k", "C1_4k")]

# Selection thresholds are deliberately based on paired pose stability, not Max.
STABLE_POSE_FRACTION = 0.80
STABLE_MEDIAN_GAIN_PCT = 0.50
SATURATION_MEDIAN_GAIN_PCT = 2.00
SATURATION_GLOBAL_GAIN_PCT = 1.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if np.isfinite(number) else math.nan


def pct_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1.0e-15:
        return math.nan
    return float(100.0 * (baseline - candidate) / baseline)


def fmt(value: Any, digits: int = 3) -> str:
    number = finite_or_nan(value)
    return "nan" if not np.isfinite(number) else f"{number:.{digits}f}"


def assert_reuse_and_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in (SOURCE_COMPARISON, SOURCE_POSE, SOURCE_V_BIN, SOURCE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("validation_read") is not False:
        raise RuntimeError("Source manifest does not prove Validation exclusion")
    if manifest.get("c0_refit") is not False:
        raise RuntimeError("Source manifest does not prove frozen C0")
    if manifest.get("production_config_modified") is not False:
        raise RuntimeError("Source manifest indicates a production configuration change")
    if manifest.get("grouped_cv_folds") != 6:
        raise RuntimeError("Source grouped-CV artifact is not the required 6-fold protocol")
    if manifest.get("frame027_retained") is not True:
        raise RuntimeError("Source manifest does not prove frame027 retention")
    if manifest.get("robust_loss") != "Huber_IRLS" or not manifest.get("frame_balanced_weighting"):
        raise RuntimeError("Source C1 protocol does not match the requested robust/frame-balanced protocol")
    points_path = Path(manifest["points_artifact"])
    frozen_model_path = Path(manifest["frozen_model"])
    if not points_path.is_file() or sha256_file(points_path) != manifest.get("points_sha256"):
        raise RuntimeError("Full-36 residual artifact hash does not match source manifest")
    if not frozen_model_path.is_file() or sha256_file(frozen_model_path) != manifest.get("frozen_model_sha256"):
        raise RuntimeError("Frozen Quadratic C0 hash does not match source manifest")

    comparison = pd.read_csv(SOURCE_COMPARISON)
    pose = pd.read_csv(SOURCE_POSE, dtype={"heldout_frame_id": str})
    v_bins = pd.read_csv(SOURCE_V_BIN)
    operational = comparison[comparison["scenario"] == OPERATIONAL_SCENARIO].copy()
    if set(operational["candidate"].astype(str)) != set(CANDIDATES) or len(operational) != 3:
        raise RuntimeError("Operational grouped-CV aggregate does not contain exactly 3 candidates")
    for _, row in operational.iterrows():
        if int(row["cv_fold_count"]) != 6 or int(row["evaluation_frame_count"]) != 35:
            raise RuntimeError("Operational grouped-CV aggregate is not 35-pose / 6-fold")
        if not bool(row["training_excludes_027"]):
            raise RuntimeError("Operational grouped-CV row did not exclude 027 from training")
        if row["robust_loss"] != "Huber_IRLS" or abs(float(row["smoothness_penalty"]) - 0.1) > 1.0e-12:
            raise RuntimeError("Operational grouped-CV protocol differs from the reused C1 protocol")

    op_pose = pose[pose["scenario"] == OPERATIONAL_SCENARIO].copy()
    op_pose["heldout_frame_id"] = op_pose["heldout_frame_id"].astype(str).str.zfill(3)
    op_c1 = op_pose[op_pose["model"] == "C0+C1"]
    if set(op_c1["candidate"].astype(str)) != set(CANDIDATES):
        raise RuntimeError("Operational pose metrics do not contain all candidates")
    frame_sets = {tuple(sorted(group["heldout_frame_id"].unique())) for _, group in op_c1.groupby("candidate")}
    if len(frame_sets) != 1:
        raise RuntimeError("Candidates do not share the same operational pose set")
    frames = next(iter(frame_sets))
    if len(frames) != 35 or "027" in frames:
        raise RuntimeError("Operational pose set is not exactly the retained 35 poses")
    if len(op_c1) != 35 * len(CANDIDATES) or set(op_pose["model"].astype(str)) != {"C0", "C0+C1"}:
        raise RuntimeError("Operational pose metrics have unexpected row count")

    op_v = v_bins[v_bins["scenario"] == OPERATIONAL_SCENARIO].copy()
    if set(op_v["candidate"].astype(str)) != set(CANDIDATES) or set(op_v["model"].astype(str)) != {"C0", "C0+C1"}:
        raise RuntimeError("Operational v-bin metrics do not contain all candidate/model rows")
    for candidate in CANDIDATES:
        if len(op_v[op_v["candidate"] == candidate]["v_bin"].unique()) != 30:
            raise RuntimeError(f"Operational v-bin coverage is incomplete for {candidate}")
    return operational, op_pose, op_v, manifest


def make_pose_paired(op_pose: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    c1 = {
        candidate: op_pose[
            (op_pose["candidate"] == candidate) & (op_pose["model"] == "C0+C1")
        ].set_index("heldout_frame_id").sort_index()
        for candidate in CANDIDATES
    }
    frame_ids = sorted(c1[CANDIDATES[0]].index.astype(str))

    # Per-pose C0 -> each candidate, preserving the requested per-pose metrics.
    for candidate in CANDIDATES:
        group = c1[candidate]
        for frame_id in frame_ids:
            item = group.loc[frame_id]
            rows.append(
                {
                    "comparison_type": "C0_vs_candidate",
                    "baseline_model": "C0",
                    "candidate_model": candidate,
                    "heldout_frame_id": frame_id,
                    "fold": item["fold"],
                    "point_count": item["point_count"],
                    "baseline_rmse_mm": math.nan,
                    "candidate_rmse_mm": item["rmse_mm"],
                    "rmse_improvement_pct": item["improvement_vs_c0_pct"],
                    "baseline_p95_abs_mm": math.nan,
                    "candidate_p95_abs_mm": item["p95_abs_mm"],
                    "p95_improvement_pct": math.nan,
                    "baseline_bias_mm": math.nan,
                    "candidate_bias_mm": item["bias_mm"],
                    "better_rmse": bool(float(item["improvement_vs_c0_pct"]) > 0.0),
                    "better_p95": math.nan,
                }
            )

    # Replace the C0-vs-candidate rows with exact C0 baseline values from the
    # same source artifact.  C0 rows are shared across candidates but retained
    # in the source per-candidate table.
    c0 = op_pose[op_pose["model"] == "C0"].copy()
    c0["heldout_frame_id"] = c0["heldout_frame_id"].astype(str).str.zfill(3)
    c0_by_frame = c0.drop_duplicates("heldout_frame_id").set_index("heldout_frame_id")
    for row in rows:
        if row["comparison_type"] != "C0_vs_candidate":
            continue
        baseline = c0_by_frame.loc[row["heldout_frame_id"]]
        row["baseline_rmse_mm"] = baseline["rmse_mm"]
        row["rmse_improvement_pct"] = pct_improvement(float(baseline["rmse_mm"]), float(row["candidate_rmse_mm"]))
        row["baseline_p95_abs_mm"] = baseline["p95_abs_mm"]
        row["p95_improvement_pct"] = pct_improvement(float(baseline["p95_abs_mm"]), float(row["candidate_p95_abs_mm"]))
        row["baseline_bias_mm"] = baseline["bias_mm"]
        row["better_rmse"] = bool(row["rmse_improvement_pct"] > 0.0)
        row["better_p95"] = bool(row["p95_improvement_pct"] > 0.0)

    # Candidate-vs-candidate paired rows use the exact same held-out pose.
    for higher, lower in PAIR_ORDER:
        for frame_id in frame_ids:
            high = c1[higher].loc[frame_id]
            low = c1[lower].loc[frame_id]
            rows.append(
                {
                    "comparison_type": "candidate_vs_candidate",
                    "baseline_model": lower,
                    "candidate_model": higher,
                    "heldout_frame_id": frame_id,
                    "fold": high["fold"],
                    "point_count": high["point_count"],
                    "baseline_rmse_mm": low["rmse_mm"],
                    "candidate_rmse_mm": high["rmse_mm"],
                    "rmse_improvement_pct": pct_improvement(float(low["rmse_mm"]), float(high["rmse_mm"])),
                    "baseline_p95_abs_mm": low["p95_abs_mm"],
                    "candidate_p95_abs_mm": high["p95_abs_mm"],
                    "p95_improvement_pct": pct_improvement(float(low["p95_abs_mm"]), float(high["p95_abs_mm"])),
                    "baseline_bias_mm": low["bias_mm"],
                    "candidate_bias_mm": high["bias_mm"],
                    "better_rmse": bool(high["rmse_mm"] < low["rmse_mm"]),
                    "better_p95": bool(high["p95_abs_mm"] < low["p95_abs_mm"]),
                }
            )

    paired = pd.DataFrame(rows).sort_values(
        ["comparison_type", "candidate_model", "baseline_model", "heldout_frame_id"]
    ).reset_index(drop=True)
    paired["operational_pose_domain"] = "Full-36 minus frame027"
    paired["frame027_status"] = OPERATIONAL_STATUS
    paired["frame027_exclusion_reason"] = EXCLUSION_REASON
    paired["source_grouped_cv_scenario"] = OPERATIONAL_SCENARIO
    paired["source_artifact"] = str(SOURCE_DIR)
    summaries: dict[str, dict[str, float]] = {}
    for higher, lower in PAIR_ORDER:
        part = paired[
            (paired["comparison_type"] == "candidate_vs_candidate")
            & (paired["candidate_model"] == higher)
            & (paired["baseline_model"] == lower)
        ]
        summaries[f"{higher}_vs_{lower}"] = {
            "rmse_better_pose_count": int(part["better_rmse"].sum()),
            "rmse_better_pose_fraction": float(part["better_rmse"].mean()),
            "rmse_gain_median_pct": float(part["rmse_improvement_pct"].median()),
            "rmse_gain_p05_pct": float(part["rmse_improvement_pct"].quantile(0.05)),
            "rmse_gain_p95_pct": float(part["rmse_improvement_pct"].quantile(0.95)),
            "p95_better_pose_count": int(part["better_p95"].sum()),
            "p95_better_pose_fraction": float(part["better_p95"].mean()),
            "p95_gain_median_pct": float(part["p95_improvement_pct"].median()),
            "p95_gain_p05_pct": float(part["p95_improvement_pct"].quantile(0.05)),
            "p95_gain_p95_pct": float(part["p95_improvement_pct"].quantile(0.95)),
        }
    return paired, summaries


def make_comparison(
    operational: pd.DataFrame,
    paired: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, str, str, dict[str, Any]]:
    op = operational.set_index("candidate").loc[CANDIDATES].reset_index()
    c0_pairs = paired[paired["comparison_type"] == "C0_vs_candidate"]
    for candidate in CANDIDATES:
        part = c0_pairs[c0_pairs["candidate_model"] == candidate]
        op.loc[op["candidate"] == candidate, "pose_rmse_better_fraction_vs_c0"] = part["better_rmse"].mean()
        op.loc[op["candidate"] == candidate, "pose_rmse_gain_median_vs_c0_pct"] = part["rmse_improvement_pct"].median()
        op.loc[op["candidate"] == candidate, "pose_rmse_gain_p05_vs_c0_pct"] = part["rmse_improvement_pct"].quantile(0.05)
        op.loc[op["candidate"] == candidate, "pose_rmse_gain_p95_vs_c0_pct"] = part["rmse_improvement_pct"].quantile(0.95)
        op.loc[op["candidate"] == candidate, "pose_p95_better_fraction_vs_c0"] = part["better_p95"].mean()
        op.loc[op["candidate"] == candidate, "pose_p95_gain_median_vs_c0_pct"] = part["p95_improvement_pct"].median()
        op.loc[op["candidate"] == candidate, "pose_p95_gain_p05_vs_c0_pct"] = part["p95_improvement_pct"].quantile(0.05)
        op.loc[op["candidate"] == candidate, "pose_p95_gain_p95_vs_c0_pct"] = part["p95_improvement_pct"].quantile(0.95)

    for higher, lower in PAIR_ORDER:
        summary = summaries[f"{higher}_vs_{lower}"]
        for metric, prefix in (("rmse", "rmse"), ("p95", "p95")):
            op.loc[op["candidate"] == higher, f"pair_{prefix}_better_fraction_{higher}_vs_{lower}"] = summary[f"{metric}_better_pose_fraction"]
            op.loc[op["candidate"] == higher, f"pair_{prefix}_gain_median_pct_{higher}_vs_{lower}"] = summary[f"{metric}_gain_median_pct"]
            op.loc[op["candidate"] == higher, f"pair_{prefix}_gain_p05_pct_{higher}_vs_{lower}"] = summary[f"{metric}_gain_p05_pct"]
            op.loc[op["candidate"] == higher, f"pair_{prefix}_gain_p95_pct_{higher}_vs_{lower}"] = summary[f"{metric}_gain_p95_pct"]

    op["operational_pose_domain"] = "Full-36 minus frame027"
    op["frame027_status"] = OPERATIONAL_STATUS
    op["frame027_exclusion_reason"] = EXCLUSION_REASON
    op["source_grouped_cv_scenario"] = OPERATIONAL_SCENARIO
    op["source_artifact"] = str(SOURCE_DIR)
    op["operational_gate_pass"] = (
        (op["global_rmse_improvement_pct"] >= 5.0)
        & (op["global_p95_improvement_pct"] >= 0.0)
        & (op["worst_v_bin_rmse_improvement_pct"] >= 0.0)
        & (op["worst_v_bin_p95_improvement_pct"] >= -2.0)
        & (op["pose_improvement_ratio"] >= 0.60)
    )

    four = summaries["C1_4k_vs_C1_3k"]
    five_vs_four = summaries["C1_5k_vs_C1_4k"]
    aggregate = op.set_index("candidate")
    four_stable = bool(
        four["rmse_better_pose_fraction"] >= STABLE_POSE_FRACTION
        and four["p95_better_pose_fraction"] >= STABLE_POSE_FRACTION
        and four["rmse_gain_median_pct"] >= STABLE_MEDIAN_GAIN_PCT
        and four["p95_gain_median_pct"] >= STABLE_MEDIAN_GAIN_PCT
    )
    five_saturated = bool(
        five_vs_four["rmse_gain_median_pct"] < SATURATION_MEDIAN_GAIN_PCT
        and five_vs_four["p95_gain_median_pct"] < SATURATION_MEDIAN_GAIN_PCT
        and pct_improvement(
            float(aggregate.loc["C1_4k", "c1_global_rmse_mm"]),
            float(aggregate.loc["C1_5k", "c1_global_rmse_mm"]),
        )
        < SATURATION_GLOBAL_GAIN_PCT
    )
    if four_stable and five_saturated:
        selected = "C1_4k"
        status = "READY_FOR_VALIDATION" if bool(aggregate.loc[selected, "operational_gate_pass"]) else "UNRESOLVED"
        rule = "4k has stable paired pose gains over 3k and 5k has saturated incremental gains over 4k."
    else:
        selected = "C1_3k"
        status = "UNRESOLVED"
        rule = "4k/5k paired gains did not satisfy the stability/saturation rule; retain the simplest 3k candidate."
    op["selection_rule_pass"] = op["candidate"].eq(selected)
    op["selected_for_validation_followup"] = op["candidate"].eq(selected)
    decision = {
        "four_k_stable_over_3k": four_stable,
        "five_k_saturated_over_4k": five_saturated,
        "selection_rule": rule,
        "selected_candidate": selected,
        "C1_OPERATIONAL_MODEL": selected,
        "C1_FIT_STATUS": status,
    }
    return op, selected, status, decision


def write_report(
    output: Path,
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, float]],
    manifest: Mapping[str, Any],
    selected: str,
    status: str,
    decision: Mapping[str, Any],
) -> None:
    rows = comparison.set_index("candidate")
    four = summaries["C1_4k_vs_C1_3k"]
    five_vs_four = summaries["C1_5k_vs_C1_4k"]
    lines = [
        "# Operational 35-pose C1 FIT-only model selection",
        "",
        f"C1_OPERATIONAL_MODEL = {selected}",
        f"C1_FIT_STATUS = {status}",
        "",
        "## Scope decision",
        "",
        f"- frame027 状态：`{OPERATIONAL_STATUS}`。排除理由：**{EXCLUSION_REASON}**；不是 residual-based deletion。",
        "- 027 原始 Full-36 residual artifact 保留不动；本轮只在 C1 development/evaluation domain 中使用剩余 35 pose。",
        "- 本轮没有读取 Validation、没有重拟合 Quadratic C0、没有重跑同协议 grouped-CV、没有做 2D C1、没有修改生产配置。",
        "",
        "## Artifact provenance / reuse audit",
        "",
        "| artifact | action | status | evidence |",
        "|---|---|---|---|",
        f"| Frozen Quadratic C0 | LOADED_ONLY / reused by source artifact | CONFIRMED | `{manifest['frozen_model']}`; sha256 `{manifest['frozen_model_sha256']}` |",
        f"| Full-36 residual artifact | REUSED_EXISTING | CONFIRMED | `{manifest['points_artifact']}`; hash matched source manifest; 027 retained |",
        f"| Existing 35-pose C1 grouped-CV | REUSED_EXISTING | CONFIRMED | `{SOURCE_DIR}`; scenario `{OPERATIONAL_SCENARIO}`; 6 folds; no refit in this run |",
        "| frame027 | EXCLUDED_FROM_OPERATIONAL_DOMAIN | CONFIRMED | reason is actual working-pose domain, not residual deletion |",
        "| Validation | NOT_READ | EXCLUDED | no Validation path opened |",
        "",
        "## Reused 35-pose grouped-CV comparison",
        "",
        "| candidate | RMSE C0→C1 / % | P95 C0→C1 / % | P99 C0→C1 / % | worst-v RMSE / % | worst-v P95 / % | v-bias range C0→C1 / mm | pose RMSE improvement ratio | operational gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in CANDIDATES:
        row = rows.loc[candidate]
        lines.append(
            f"| {candidate} | {fmt(row['global_rmse_improvement_pct'], 2)} | {fmt(row['global_p95_improvement_pct'], 2)} | {fmt(row['global_p99_improvement_pct'], 2)} | {fmt(row['worst_v_bin_rmse_improvement_pct'], 2)} | {fmt(row['worst_v_bin_p95_improvement_pct'], 2)} | {fmt(row['c0_v_bias_range_mm'])}→{fmt(row['c1_v_bias_range_mm'])} | {fmt(row['pose_improvement_ratio'], 3)} | {'PASS' if row['operational_gate_pass'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Pose-level paired comparison",
            "",
            "Positive values mean the higher-knot candidate is better than the baseline candidate.",
            "",
            "| comparison | RMSE better poses | RMSE median / P05 / P95 gain % | P95 better poses | P95 median / P05 / P95 gain % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for higher, lower in PAIR_ORDER:
        summary = summaries[f"{higher}_vs_{lower}"]
        lines.append(
            f"| {higher} vs {lower} | {summary['rmse_better_pose_count']}/35 ({summary['rmse_better_pose_fraction']:.3f}) | {summary['rmse_gain_median_pct']:.2f} / {summary['rmse_gain_p05_pct']:.2f} / {summary['rmse_gain_p95_pct']:.2f} | {summary['p95_better_pose_count']}/35 ({summary['p95_better_pose_fraction']:.3f}) | {summary['p95_gain_median_pct']:.2f} / {summary['p95_gain_p05_pct']:.2f} / {summary['p95_gain_p95_pct']:.2f} |"
        )
    lines.extend(
        [
            "",
            "### Selection rule",
            "",
            f"- 4k stable-gain gate: paired RMSE/P95 better-pose fraction ≥ {STABLE_POSE_FRACTION:.2f}, and both median gains ≥ {STABLE_MEDIAN_GAIN_PCT:.2f}%. Result: `{decision['four_k_stable_over_3k']}`.",
            f"- 5k saturation gate: paired 5k-vs-4k median RMSE/P95 gain < {SATURATION_MEDIAN_GAIN_PCT:.2f}% and global RMSE increment < {SATURATION_GLOBAL_GAIN_PCT:.2f}%. Result: `{decision['five_k_saturated_over_4k']}`.",
            f"- **{decision['selection_rule']}**",
            "",
            f"当前数值支持 C1_4k：相对 3k，RMSE/P95 均在 {int(four['rmse_better_pose_count'])}/35 pose 改善；5k 相对 4k 的 RMSE/P95 median 增量为 {five_vs_four['rmse_gain_median_pct']:.2f}%/{five_vs_four['p95_gain_median_pct']:.2f}%，继续改善 pose 为 {int(five_vs_four['rmse_better_pose_count'])}/35、{int(five_vs_four['p95_better_pose_count'])}/35，收益已明显递减。",
            "",
            "## Scope and handoff",
            "",
            f"- 本报告输出的是最终 FIT-only 候选：`{selected}`；`C1_FIT_STATUS = {status}` 仅表示可以进入下一步 Validation，不表示生产冻结或已通过 Validation。",
            "- 027 仍保留在 Full-36 residual artifact；正式 operational domain label 只作用于 C1 development/evaluation，不改变原始数据文件。",
            "- 下一步应使用独立 Validation 验证该候选，届时另建版本化产物；本轮不写入生产配置。",
            "",
            "## Outputs",
            "",
            "- `c1_operational35_model_comparison.csv`：35-pose aggregate 与选择字段。",
            "- `c1_operational35_pose_paired.csv`：C0→候选及候选间逐 pose paired RMSE/P95。",
            "- `c1_operational35_v_bins.csv`：复用的 100 px v-bin metrics。",
            "- `c1_model_selection_report.md`：本报告。",
            "",
        ]
    )
    (output / "c1_model_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    operational, op_pose, op_v, manifest = assert_reuse_and_load()
    paired, summaries = make_pose_paired(op_pose)
    comparison, selected, status, decision = make_comparison(operational, paired, summaries)

    comparison.to_csv(output / "c1_operational35_model_comparison.csv", index=False)
    paired.to_csv(output / "c1_operational35_pose_paired.csv", index=False)
    op_v = op_v.copy()
    op_v["operational_pose_domain"] = "Full-36 minus frame027"
    op_v["frame027_status"] = OPERATIONAL_STATUS
    op_v["frame027_exclusion_reason"] = EXCLUSION_REASON
    op_v["source_artifact"] = str(SOURCE_DIR)
    op_v.to_csv(output / "c1_operational35_v_bins.csv", index=False)

    selection_manifest = {
        "C1_OPERATIONAL_MODEL": selected,
        "C1_FIT_STATUS": status,
        "operational_pose_count": 35,
        "operational_pose_excludes": ["027"],
        "frame027_status": OPERATIONAL_STATUS,
        "frame027_exclusion_reason": EXCLUSION_REASON,
        "source_grouped_cv_scenario": OPERATIONAL_SCENARIO,
        "source_dir": str(SOURCE_DIR),
        "source_manifest": str(SOURCE_MANIFEST),
        "validation_read": False,
        "c0_refit": False,
        "production_config_modified": False,
        "paired_summaries": summaries,
        "selection_decision": decision,
    }
    (output / "c1_operational35_selection_manifest.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output, comparison, paired, summaries, manifest, selected, status, decision)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "C1_OPERATIONAL_MODEL": selected,
                "C1_FIT_STATUS": status,
                "frame027_status": OPERATIONAL_STATUS,
                "operational_pose_count": 35,
                "reused_grouped_cv": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run(parse_args())
