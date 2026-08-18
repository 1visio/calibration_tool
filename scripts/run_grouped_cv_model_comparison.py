#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pose-grouped CV comparison of Plane, Quadratic and Circular Cone.

The runner opens only the three FIT roots used by the formal FIT collection.
It reuses the full physical-board extraction path and the existing model
classes.  All points from a pose stay in one CV fold; no point-wise split and
no v-density weighting are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
from run_full_fit_v_coverage_audit import extract_all_fit  # noqa: E402


FIT_FRAME_IDS = tuple(
    [f"{value:03d}" for value in range(1, 19)]
    + [f"{value:03d}" for value in range(25, 37)]
    + [f"{value:03d}" for value in range(49, 55)]
)
MODEL_NAMES = ("global_plane", "quadratic_graph", "circular_cone")
V_MIN_PX = 0.0
V_MAX_PX = 3000.0
BIN_WIDTH_PX = 100.0
BIN_EDGES = np.arange(V_MIN_PX, V_MAX_PX + BIN_WIDTH_PX, BIN_WIDTH_PX)
BIN_COUNT = len(BIN_EDGES) - 1
DEFAULT_CONFIG = ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0817" / "grouped_cv_model_comparison"
DEFAULT_FOLDS = 6


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


def fit_models(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> Tuple[triplets.PlaneModel, triplets.QuadraticGraphModel, triplets.CircularConeModel]:
    points, _, frame_ids = triplets.dataframe_arrays(df)
    plane = triplets.PlaneModel()
    plane.fit(points, frame_ids)
    quadratic_cfg = cfg.get("models", {}).get("quadratic", {})
    quadratic = triplets.QuadraticGraphModel(
        ridge=float(quadratic_cfg.get("ridge", 1.0e-10))
    )
    quadratic.fit(points, frame_ids, plane=plane)
    cone = triplets.CircularConeModel(dict(cfg.get("models", {}).get("cone", {})))
    cone.fit(points, frame_ids, plane=plane)
    return plane, quadratic, cone


def make_fold_assignment(frame_ids: Sequence[str], fold_count: int) -> Dict[str, int]:
    ordered = sorted({str(value) for value in frame_ids})
    if len(ordered) < fold_count:
        raise ValueError(f"pose 数 {len(ordered)} 小于 fold 数 {fold_count}")
    return {frame_id: index % fold_count for index, frame_id in enumerate(ordered)}


def metric_values(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "point_count": 0,
            "bias_mm": np.nan,
            "mae_mm": np.nan,
            "rmse_mm": np.nan,
            "p95_mm": np.nan,
            "max_abs_mm": np.nan,
        }
    absolute = np.abs(array)
    return {
        "point_count": int(array.size),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array ** 2))),
        "p95_mm": float(np.percentile(absolute, 95)),
        "max_abs_mm": float(np.max(absolute)),
    }


def add_detail_context(
    detail: pd.DataFrame,
    source_df: pd.DataFrame,
    fold: int,
    heldout_frames: Sequence[str],
) -> pd.DataFrame:
    result = detail.copy()
    result["fold"] = int(fold)
    result["heldout_frames"] = ",".join(heldout_frames)
    result["frame_id"] = source_df.loc[result.index, "frame_id"].astype(str).to_numpy()
    result["fit_group"] = source_df.loc[result.index, "fit_group"].astype(str).to_numpy()
    return result.reset_index(drop=True)


def fold_metric_row(
    detail: pd.DataFrame,
    model: str,
    fold: int,
    train_frames: Sequence[str],
    heldout_frames: Sequence[str],
) -> Dict[str, Any]:
    values = detail["board_error_mm"].to_numpy(dtype=float)
    metric = metric_values(values)
    valid = detail["valid"].to_numpy(dtype=bool)
    row: Dict[str, Any] = {
        "row_type": "fold",
        "fold": int(fold),
        "model": model,
        "region": "Global",
        "v_bin_lo_px": V_MIN_PX,
        "v_bin_hi_px": V_MAX_PX,
        "train_frame_count": int(len(train_frames)),
        "heldout_frame_count": int(len(heldout_frames)),
        "train_frames": ",".join(train_frames),
        "heldout_frames": ",".join(heldout_frames),
        "valid_intersections": int(np.count_nonzero(valid)),
        "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
        **metric,
    }
    return row


def pooled_frame_metrics(detail: pd.DataFrame, column: str) -> Tuple[float, float]:
    values: List[float] = []
    for _, group in detail.groupby(column):
        metric = metric_values(group["board_error_mm"].to_numpy(dtype=float))
        if np.isfinite(metric["rmse_mm"]):
            values.append(float(metric["rmse_mm"]))
    if not values:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def build_bin_metrics(detail: pd.DataFrame, model: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    v = detail["v_px"].to_numpy(dtype=float)
    in_domain = np.isfinite(v) & (v >= V_MIN_PX) & (v < V_MAX_PX)
    detail = detail.loc[in_domain].copy()
    detail["v_bin_index"] = np.floor((detail["v_px"] - V_MIN_PX) / BIN_WIDTH_PX).astype(int)
    for index in range(BIN_COUNT):
        group = detail[detail["v_bin_index"] == index]
        metric = metric_values(group["board_error_mm"].to_numpy(dtype=float))
        valid = group["valid"].to_numpy(dtype=bool)
        frame_mean_rmse, frame_std_rmse = pooled_frame_metrics(group, "frame_id") if not group.empty else (np.nan, np.nan)
        fold_mean_rmse, fold_std_rmse = pooled_frame_metrics(group, "fold") if not group.empty else (np.nan, np.nan)
        rows.append(
            {
                "model": model,
                "v_bin": f"v_{int(BIN_EDGES[index]):04d}_{int(BIN_EDGES[index + 1]):04d}",
                "v_bin_lo_px": float(BIN_EDGES[index]),
                "v_bin_hi_px": float(BIN_EDGES[index + 1]),
                "point_count": int(len(group)),
                "unique_frame_count": int(group["frame_id"].nunique()),
                "fold_count": int(group["fold"].nunique()),
                "frame_ids": ",".join(sorted(group["frame_id"].astype(str).unique())),
                "valid_intersections": int(np.count_nonzero(valid)),
                "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
                **metric,
                "mean_frame_rmse_mm": frame_mean_rmse,
                "std_frame_rmse_mm": frame_std_rmse,
                "mean_fold_rmse_mm": fold_mean_rmse,
                "std_fold_rmse_mm": fold_std_rmse,
            }
        )
    return pd.DataFrame(rows)


def trend_metrics(bin_metrics: pd.DataFrame) -> Dict[str, float]:
    clean = bin_metrics[np.isfinite(bin_metrics["bias_mm"])].copy()
    if clean.empty:
        return {"v_bias_range_mm": np.nan, "v_bias_slope_mm_per_px": np.nan, "v_bias_rho": np.nan}
    x = ((clean["v_bin_lo_px"] + clean["v_bin_hi_px"]) / 2.0).to_numpy(dtype=float)
    y = clean["bias_mm"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(clean) >= 2 else np.nan
    rho = float(np.corrcoef(x, y)[0, 1]) if len(clean) >= 2 and np.std(y) > 1.0e-12 else 0.0
    return {
        "v_bias_range_mm": float(np.ptp(y)) if y.size else np.nan,
        "v_bias_slope_mm_per_px": slope,
        "v_bias_rho": rho,
    }


def aggregate_rows(
    details: Mapping[str, pd.DataFrame],
    fold_rows: pd.DataFrame,
    bin_metrics: Mapping[str, pd.DataFrame],
    all_frame_count: int,
    fold_count: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in MODEL_NAMES:
        detail = details[model]
        metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
        valid = detail["valid"].to_numpy(dtype=bool)
        frame_mean_rmse, frame_std_rmse = pooled_frame_metrics(detail, "frame_id")
        fold_mean_rmse, fold_std_rmse = pooled_frame_metrics(detail, "fold")
        bins = bin_metrics[model]
        valid_bins = bins[np.isfinite(bins["rmse_mm"])].copy()
        if valid_bins.empty:
            worst = pd.Series(dtype=float)
        else:
            worst = valid_bins.sort_values(["rmse_mm", "p95_mm"], ascending=False).iloc[0]
        trend = trend_metrics(bins)
        row: Dict[str, Any] = {
            "row_type": "pooled_cv",
            "fold": "ALL",
            "model": model,
            "region": "Global",
            "v_bin_lo_px": V_MIN_PX,
            "v_bin_hi_px": V_MAX_PX,
            "train_frame_count": int(all_frame_count - all_frame_count / fold_count),
            "heldout_frame_count": all_frame_count,
            "train_frames": "grouped_cv",
            "heldout_frames": ",".join(sorted(detail["frame_id"].astype(str).unique())),
            "fold_count": fold_count,
            "valid_intersections": int(np.count_nonzero(valid)),
            "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
            **metric,
            "mean_frame_rmse_mm": frame_mean_rmse,
            "std_frame_rmse_mm": frame_std_rmse,
            "mean_fold_rmse_mm": fold_mean_rmse,
            "std_fold_rmse_mm": fold_std_rmse,
            "worst_v_bin": str(worst.get("v_bin", "")),
            "worst_v_bin_rmse_mm": float(worst.get("rmse_mm", np.nan)),
            "worst_v_bin_p95_mm": float(worst.get("p95_mm", np.nan)),
            "worst_v_bin_max_abs_mm": float(worst.get("max_abs_mm", np.nan)),
            **trend,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_selection_scores(aggregate: pd.DataFrame) -> pd.DataFrame:
    result = aggregate.copy()
    score_columns = {
        "rmse_mm": 0.25,
        "worst_v_bin_rmse_mm": 0.30,
        "worst_v_bin_p95_mm": 0.15,
        "v_bias_range_mm": 0.15,
        "std_fold_rmse_mm": 0.10,
        "std_frame_rmse_mm": 0.05,
    }
    score = np.zeros(len(result), dtype=float)
    for column, weight in score_columns.items():
        values = result[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            score += weight
            continue
        lo = float(np.min(values[finite]))
        hi = float(np.max(values[finite]))
        normalized = np.zeros_like(values)
        if hi > lo:
            normalized[finite] = (values[finite] - lo) / (hi - lo)
        score += weight * normalized
    result["selection_score"] = score
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=int)
    ranks[order] = np.arange(1, len(score) + 1)
    result["selection_rank"] = ranks
    return result


def choose_candidate(scored: pd.DataFrame) -> Tuple[str, str]:
    ordered = scored.sort_values("selection_score").reset_index(drop=True)
    if len(ordered) < 2:
        return "UNRESOLVED", "insufficient model rows"
    best = float(ordered.loc[0, "selection_score"])
    second = float(ordered.loc[1, "selection_score"])
    gap = second - best
    if not np.isfinite(gap) or gap < 0.05:
        return "UNRESOLVED", f"top score gap={gap:.4f} < 0.05"
    model = str(ordered.loc[0, "model"])
    label = {"global_plane": "PLANE", "quadratic_graph": "QUADRATIC", "circular_cone": "CONE"}.get(model, "UNRESOLVED")
    return label, f"top score gap={gap:.4f}"


def save_full_models(output_dir: Path, models: Sequence[triplets.LaserModel], fit_df: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    model_dir = output_dir / "candidate_models" / "full_fit"
    model_dir.mkdir(parents=True, exist_ok=True)
    parameters: Dict[str, Any] = {
        "source": "FIT 001-018, 025-036, 049-054",
        "frame_count": int(fit_df["frame_id"].nunique()),
        "point_count": int(len(fit_df)),
        "mask_mode": str(cfg.get("extraction", {}).get("board_mask_mode")),
        "mask_inset_mm": float(cfg.get("extraction", {}).get("board_mask_inset_mm", 0.0)),
        "validation_read": False,
        "c1_trained": False,
        "models": {},
    }
    for model in models:
        params = model.to_dict()
        triplets.save_yaml(model_dir / f"{model.name}.yaml", params)
        parameters["models"][model.name] = params
    (model_dir / "model_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def plot_cv(output: Path, bins: Mapping[str, pd.DataFrame]) -> None:
    colors = {"global_plane": "#d62728", "quadratic_graph": "#1f77b4", "circular_cone": "#2ca02c"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for model in MODEL_NAMES:
        data = bins[model]
        x = (data["v_bin_lo_px"] + data["v_bin_hi_px"]) / 2.0
        axes[0].plot(x, data["bias_mm"], marker="o", markersize=3, linewidth=1.8, color=colors[model], label=model)
        axes[1].plot(x, data["rmse_mm"], marker="o", markersize=3, linewidth=1.8, color=colors[model], label=model)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("CV bias / mm")
    axes[0].set_title("Pose-grouped CV residual-v bias by 100 px bin")
    axes[1].set_ylabel("CV RMSE / mm")
    axes[1].set_xlabel("image v / px")
    axes[1].set_title("Pose-grouped CV residual-v RMSE by 100 px bin")
    for axis in axes:
        axis.set_xlim(V_MIN_PX, V_MAX_PX)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("FIT-only grouped CV: Plane / Quadratic / Circular Cone", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_report(
    output: Path,
    config_path: Path,
    intrinsics_path: Path,
    fit_df: pd.DataFrame,
    fold_map: Mapping[str, int],
    fold_rows: pd.DataFrame,
    bins: Mapping[str, pd.DataFrame],
    aggregate: pd.DataFrame,
    candidate: str,
    candidate_reason: str,
) -> None:
    lines = [
        "# Full-v FIT pose-grouped CV 三模型比较",
        "",
        f"`C0_MODEL_CANDIDATE = {candidate}`",
        "",
        "## Scope",
        "",
        "- 仅读取 FIT 001–018、025–036、049–054，共 36 pose；未读取任何 Validation 图像。",
        f"- FIT 点：{len(fit_df)}；内参：`{intrinsics_path}`；配置：`{config_path}`。",
        "- mask：`full_board_physical`，11×8 内角点、20 mm 格距，X=[-20,220] mm、Y=[-20,160] mm、inset=0 mm。",
        "- Steger、vertical per-row single point、continuity、每帧点数限制和 frame-balanced weighting 与正式流程保持一致。",
        "- 未增加 v-density weighting；未训练 C1；未覆盖历史 Frozen C0。",
        "",
        "## Grouped CV design",
        "",
        f"- 采用 {len(set(fold_map.values()))}-fold pose-grouped CV；每个 fold 完整留出 pose，训练与评价之间没有同一 pose 的点交叉。",
        "- fold assignment 为按 frame ID 排序后的确定性 round-robin；每 fold 6 个 held-out pose。",
        "",
        "| fold | held-out frames | train frames |",
        "|---:|---|---:|",
    ]
    for fold in sorted(set(fold_map.values())):
        heldout = sorted([frame for frame, value in fold_map.items() if value == fold])
        lines.append(f"| {fold} | {', '.join(heldout)} | {36 - len(heldout)} |")

    lines += [
        "",
        "## Pooled grouped-CV comparison",
        "",
        "以下指标来自所有 held-out pose 的 pooled predictions；模型选择同时使用 Global、worst-v-bin、v-bias range 和 fold/frame 稳定性，不使用训练集 global RMSE 单独选择。",
        "",
        "| model | Global bias | Global RMSE | Global P95 | worst v-bin | worst RMSE | worst P95 | v-bias range | fold RMSE std | score |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.sort_values("selection_score").itertuples():
        lines.append(
            f"| {row.model} | {row.bias_mm:.5f} | {row.rmse_mm:.5f} | {row.p95_mm:.5f} | {row.worst_v_bin} | {row.worst_v_bin_rmse_mm:.5f} | {row.worst_v_bin_p95_mm:.5f} | {row.v_bias_range_mm:.5f} | {row.std_fold_rmse_mm:.5f} | {row.selection_score:.5f} |"
        )

    lines += [
        "",
        "## Worst v-bin detail",
        "",
    ]
    for model in MODEL_NAMES:
        data = bins[model].dropna(subset=["rmse_mm"]).sort_values("rmse_mm", ascending=False).head(3)
        lines.append(f"### {model}")
        lines.append("")
        lines.append(data[["v_bin", "point_count", "unique_frame_count", "bias_mm", "rmse_mm", "p95_mm", "max_abs_mm"]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")

    lines += [
        "## Full 36-pose reference",
        "",
        "三模型全量拟合参数保存在 `candidate_models/full_fit/`；该 fit 仅作为最大数据量 reference，不用于 grouped-CV 选择。",
        "- `global_plane.yaml`",
        "- `quadratic_graph.yaml`",
        "- `circular_cone.yaml`",
        "- `model_parameters.json`",
        "",
        "## Selection rule",
        "",
        "score = 0.25×Global RMSE + 0.30×worst-v-bin RMSE + 0.15×worst-v-bin P95 + 0.15×v-bias range + 0.10×fold RMSE std + 0.05×frame RMSE std；各项先在三模型间 min-max 归一化。最高分与第二名差距小于 0.05 时判为 `UNRESOLVED`。",
        f"本次选择判据：{candidate_reason}。",
        "",
        "## Artifacts",
        "",
        "- `grouped_cv_model_comparison.csv`：fold 级与 pooled Global CV 指标及 worst-v-bin/trend 汇总。",
        "- `per_v_bin_cv_metrics.csv`：三模型每个 100 px v-bin 的 Bias、RMSE、P95、Max 和 pose/fold 覆盖。",
        "- `residual_vs_v_cv.png`：pooled grouped-CV 的 residual-v Bias/RMSE 趋势。",
        "",
        f"结论：`C0_MODEL_CANDIDATE = {candidate}`。该结论只基于 FIT pose-grouped CV；进入独立 Validation 或标准件验收前，不冻结生产模型。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pose-grouped CV comparison of three laser surface models")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--intrinsics", default=None, help="intrinsics 文件或目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"输出目录非空；如需重新生成请显式使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = triplets.safe_yaml_load(config_path)
    intrinsic_value = Path(args.intrinsics).resolve() if args.intrinsics else resolve_path(str(cfg["intrinsics"]), config_path.parent)
    intrinsics_path = resolve_intrinsics(intrinsic_value)
    k, d, image_size = triplets.load_intrinsics(intrinsics_path)
    extraction_cfg = cfg.get("extraction", {})
    if str(extraction_cfg.get("board_mask_mode", "")).lower() != triplets.FULL_BOARD_PHYSICAL:
        raise RuntimeError("当前配置不是 full_board_physical，停止以避免误用旧 mask")
    if abs(float(extraction_cfg.get("board_mask_inset_mm", 0.0))) > 1.0e-12:
        raise RuntimeError("当前 full_board_physical inset 不是 0 mm")

    print(f"Config: {config_path}")
    print(f"Intrinsics: {intrinsics_path}")
    print("FIT only: 001-018, 025-036, 049-054")
    print(f"Grouped CV folds: {args.folds}")
    fit_df = extract_all_fit(cfg, k, d, image_size, output_dir)
    actual_frames = sorted(fit_df["frame_id"].astype(str).unique())
    if actual_frames != sorted(FIT_FRAME_IDS):
        raise RuntimeError(f"FIT pose 集合异常：{actual_frames}")
    if len(actual_frames) != 36:
        raise RuntimeError(f"FIT pose 数异常：{len(actual_frames)}")

    full_models = fit_models(fit_df, cfg)
    save_full_models(output_dir, full_models, fit_df, cfg)

    fold_map = make_fold_assignment(actual_frames, args.folds)
    fold_rows: List[Dict[str, Any]] = []
    detail_by_model: Dict[str, List[pd.DataFrame]] = {name: [] for name in MODEL_NAMES}
    fold_parameter_rows: Dict[str, Any] = {}
    for fold in range(args.folds):
        heldout_frames = sorted([frame for frame, assigned in fold_map.items() if assigned == fold])
        train_frames = sorted([frame for frame, assigned in fold_map.items() if assigned != fold])
        train_df = fit_df[~fit_df["frame_id"].isin(heldout_frames)].copy()
        test_df = fit_df[fit_df["frame_id"].isin(heldout_frames)].copy()
        print(f"Fold {fold}: train={len(train_frames)} poses, heldout={','.join(heldout_frames)}")
        plane, quadratic, cone = fit_models(train_df, cfg)
        fold_models: Sequence[triplets.LaserModel] = (plane, quadratic, cone)
        fold_parameter_rows[str(fold)] = {model.name: model.to_dict() for model in fold_models}
        for model in fold_models:
            metrics, raw_detail = triplets.evaluate_model(model, test_df, plane)
            detail = add_detail_context(raw_detail, test_df, fold, heldout_frames)
            detail_by_model[model.name].append(detail)
            fold_rows.append(fold_metric_row(detail, model.name, fold, train_frames, heldout_frames))

    (output_dir / "cv_fold_model_parameters.json").write_text(
        json.dumps(fold_parameter_rows, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pooled_details = {model: pd.concat(chunks, ignore_index=True) for model, chunks in detail_by_model.items()}
    fold_metrics = pd.DataFrame(fold_rows)
    bin_metrics = {model: build_bin_metrics(pooled_details[model], model) for model in MODEL_NAMES}
    per_bin = pd.concat(list(bin_metrics.values()), ignore_index=True)
    aggregate = aggregate_rows(pooled_details, fold_metrics, bin_metrics, len(actual_frames), args.folds)
    aggregate = add_selection_scores(aggregate)
    candidate, candidate_reason = choose_candidate(aggregate)
    aggregate["candidate"] = candidate
    aggregate["candidate_reason"] = candidate_reason

    grouped_output = pd.concat([fold_metrics, aggregate], ignore_index=True, sort=False)
    grouped_output.to_csv(output_dir / "grouped_cv_model_comparison.csv", index=False, encoding="utf-8-sig")
    per_bin = per_bin.merge(
        aggregate[["model", "selection_score", "selection_rank"]], on="model", how="left"
    )
    per_bin.to_csv(output_dir / "per_v_bin_cv_metrics.csv", index=False, encoding="utf-8-sig")
    for model, detail in pooled_details.items():
        detail.to_csv(output_dir / f"cv_pointwise_{model}.csv", index=False, encoding="utf-8-sig")
    plot_cv(output_dir / "residual_vs_v_cv.png", bin_metrics)
    generate_report(
        output_dir / "report.md",
        config_path,
        intrinsics_path,
        fit_df,
        fold_map,
        fold_metrics,
        bin_metrics,
        aggregate,
        candidate,
        candidate_reason,
    )
    print(f"C0_MODEL_CANDIDATE = {candidate}")
    print(f"Output: {output_dir}")
    print(aggregate[["model", "rmse_mm", "worst_v_bin", "worst_v_bin_rmse_mm", "selection_score"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

