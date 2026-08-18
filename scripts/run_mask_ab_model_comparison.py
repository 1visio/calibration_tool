#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B audit of the old and full-physical-board laser extraction masks.

This runner intentionally processes only FIT 001--018.  It reuses the formal
triplet extraction and model implementations, and changes only the board mask
configuration between the two independently fitted branches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_laser_models_from_triplets as triplets  # noqa: E402


FIT_IDS = tuple(range(1, 19))
BIN_WIDTH_PX = 100.0
V_MIN_PX = 0.0
V_MAX_PX = 3000.0
MODEL_NAMES = ("global_plane", "quadratic_graph", "circular_cone")


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def resolve_intrinsics(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "calibration_result.yaml"
        if not candidate.exists():
            matches = sorted(path.glob("*.yaml"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"内参路径是目录，但未能唯一定位 calibration_result.yaml：{path}"
                )
            candidate = matches[0]
        return candidate
    return path


def variant_extraction_configs(base: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    old = dict(base)
    old["board_mask_mode"] = "inner_corner_hull"
    old["board_mask_margin_px"] = -2

    new = dict(base)
    new["board_mask_mode"] = triplets.FULL_BOARD_PHYSICAL
    new["board_mask_inset_mm"] = 0.0
    return {"old": old, "new": new}


def load_config_and_inputs(
    config_path: Path,
    intrinsics_arg: Path | None,
    fit_root_arg: Path | None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Tuple[int, int] | None, Path, Path]:
    cfg = triplets.safe_yaml_load(config_path)
    intrinsics_value = intrinsics_arg or resolve_path(str(cfg["intrinsics"]), config_path.parent)
    intrinsics_path = resolve_intrinsics(intrinsics_value)
    k, dist, image_size = triplets.load_intrinsics(intrinsics_path)

    dataset_cfg = cfg.get("datasets", {}).get("train", {})
    fit_root = fit_root_arg or resolve_path(str(dataset_cfg["root"]), config_path.parent)
    return cfg, k, dist, image_size, intrinsics_path, fit_root


def extract_variant(
    variant: str,
    cfg: Mapping[str, Any],
    k: np.ndarray,
    dist: np.ndarray,
    image_size: Tuple[int, int] | None,
    fit_root: Path,
    preview_dir: Path,
) -> pd.DataFrame:
    patterns = dict(cfg["patterns"])
    board_cfg = dict(cfg["board"])
    extraction_cfg = variant_extraction_configs(cfg.get("extraction", {}))[variant]
    laser_cfg = triplets.parse_laser_config(cfg.get("laser"))
    orientation = triplets.normalize_laser_orientation(laser_cfg.orientation)
    dataset_cfg = {"root": str(fit_root), "ids": list(FIT_IDS)}

    df = triplets.process_dataset(
        variant,
        dataset_cfg,
        patterns,
        k,
        dist,
        image_size,
        board_cfg,
        extraction_cfg,
        orientation,
        preview_dir / variant,
    )
    expected_keys = {f"{variant}:{image_id}" for image_id in FIT_IDS}
    actual_keys = set(df["frame_key"].astype(str).unique())
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise RuntimeError(f"{variant} 分支缺少 FIT 帧：{missing}")
    if len(actual_keys) != len(expected_keys):
        raise RuntimeError(f"{variant} 分支帧数量异常：{len(actual_keys)}")
    return df


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


def finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = float(metrics.get(name, np.nan))
    return value if np.isfinite(value) else float("nan")


def comparison_row(
    variant: str,
    region: str,
    v_lo: float,
    v_hi: float,
    detail: pd.DataFrame,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "mask": "old_inner_hull_margin_-2px" if variant == "old" else "new_full_board_physical_inset_0mm",
        "model": str(metrics["model"]),
        "region": region,
        "v_bin_lo_px": float(v_lo),
        "v_bin_hi_px": float(v_hi),
        "point_count": int(metrics["total_points"]),
        "frame_count": int(detail["frame_key"].nunique()) if not detail.empty else 0,
        "valid_intersections": int(metrics["valid_intersections"]),
        "valid_rate": float(metrics["valid_rate"]),
        "bias_mm": finite_metric(metrics, "board_mean_signed_mm"),
        "mae_mm": finite_metric(metrics, "board_mae_mm"),
        "rmse_mm": finite_metric(metrics, "board_rmse_mm"),
        "p95_mm": finite_metric(metrics, "board_p95_abs_mm"),
        "max_abs_mm": finite_metric(metrics, "board_max_abs_mm"),
        "surface_rmse_mm": finite_metric(metrics, "surface_rmse_mm"),
        "ray_rmse_mm": finite_metric(metrics, "ray_rmse_mm"),
    }


def evaluate_variant(
    variant: str,
    df: pd.DataFrame,
    models: Sequence[triplets.LaserModel],
    plane: triplets.PlaneModel,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict[str, Any]] = []
    details: Dict[str, pd.DataFrame] = {}
    edges = np.arange(V_MIN_PX, V_MAX_PX + BIN_WIDTH_PX, BIN_WIDTH_PX)

    for model in models:
        global_metrics, global_detail = triplets.evaluate_model(model, df, plane)
        details[model.name] = global_detail
        rows.append(
            comparison_row(
                variant,
                "Global",
                V_MIN_PX,
                V_MAX_PX,
                global_detail,
                global_metrics,
            )
        )

        for v_lo, v_hi in zip(edges[:-1], edges[1:]):
            selection = (df["v_px"] >= v_lo) & (df["v_px"] < v_hi)
            subset = df.loc[selection].copy()
            region = f"v_{int(v_lo):04d}_{int(v_hi):04d}"
            if subset.empty:
                rows.append(
                    {
                        "variant": variant,
                        "mask": "old_inner_hull_margin_-2px" if variant == "old" else "new_full_board_physical_inset_0mm",
                        "model": model.name,
                        "region": region,
                        "v_bin_lo_px": float(v_lo),
                        "v_bin_hi_px": float(v_hi),
                        "point_count": 0,
                        "frame_count": 0,
                        "valid_intersections": 0,
                        "valid_rate": np.nan,
                        "bias_mm": np.nan,
                        "mae_mm": np.nan,
                        "rmse_mm": np.nan,
                        "p95_mm": np.nan,
                        "max_abs_mm": np.nan,
                        "surface_rmse_mm": np.nan,
                        "ray_rmse_mm": np.nan,
                    }
                )
                continue
            metrics, detail = triplets.evaluate_model(model, subset, plane)
            rows.append(
                comparison_row(variant, region, v_lo, v_hi, detail, metrics)
            )
    return pd.DataFrame(rows), details


def save_models(
    output_dir: Path,
    variant: str,
    models: Sequence[triplets.LaserModel],
) -> None:
    model_dir = output_dir / "models" / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        triplets.save_yaml(model_dir / f"{model.name}.yaml", model.to_dict())


def plot_residual_vs_v(
    output: Path,
    all_details: Mapping[Tuple[str, str], pd.DataFrame],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    bins = np.arange(V_MIN_PX, V_MAX_PX + BIN_WIDTH_PX, BIN_WIDTH_PX)
    colors = {
        "global_plane": "#d62728",
        "quadratic_graph": "#1f77b4",
        "circular_cone": "#2ca02c",
    }
    for axis, variant in zip(axes, ("old", "new")):
        for model_name in MODEL_NAMES:
            detail = all_details[(variant, model_name)]
            clean = detail[["v_px", "board_error_mm"]].dropna()
            if clean.empty:
                continue
            stride = max(1, len(clean) // 3500)
            sample = clean.iloc[::stride]
            axis.scatter(
                sample["v_px"],
                sample["board_error_mm"],
                s=3,
                alpha=0.10,
                color=colors[model_name],
                label=f"{model_name} points",
            )
            bin_index = np.digitize(clean["v_px"].to_numpy(), bins) - 1
            median_rows = []
            for idx in range(len(bins) - 1):
                values = clean.loc[bin_index == idx, "board_error_mm"].to_numpy()
                values = values[np.isfinite(values)]
                if values.size:
                    median_rows.append((0.5 * (bins[idx] + bins[idx + 1]), np.median(values)))
            if median_rows:
                median_array = np.asarray(median_rows)
                axis.plot(
                    median_array[:, 0],
                    median_array[:, 1],
                    color=colors[model_name],
                    linewidth=2.0,
                    label=f"{model_name} 100px median",
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("board residual / mm")
        axis.set_title(
            "Old mask: inner-corner hull + margin=-2 px"
            if variant == "old"
            else "New mask: full physical board, inset=0 mm"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("image v / px")
    axes[-1].set_xlim(V_MIN_PX, V_MAX_PX)
    fig.suptitle("001–018 mask A/B: residual versus v", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def select_worst_bin(rows: pd.DataFrame, variant: str, model: str) -> pd.Series:
    subset = rows[
        (rows["variant"] == variant)
        & (rows["model"] == model)
        & rows["region"].str.startswith("v_")
        & np.isfinite(rows["rmse_mm"])
    ].copy()
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.sort_values(["rmse_mm", "max_abs_mm"], ascending=False).iloc[0]


def effect_table(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model in MODEL_NAMES:
        old_global = comparison[
            (comparison["variant"] == "old")
            & (comparison["model"] == model)
            & (comparison["region"] == "Global")
        ].iloc[0]
        new_global = comparison[
            (comparison["variant"] == "new")
            & (comparison["model"] == model)
            & (comparison["region"] == "Global")
        ].iloc[0]
        old_worst = select_worst_bin(comparison, "old", model)
        new_worst = select_worst_bin(comparison, "new", model)
        new_at_old = comparison[
            (comparison["variant"] == "new")
            & (comparison["model"] == model)
            & (comparison["region"] == old_worst.get("region", "__none__"))
        ]
        old_at_new = comparison[
            (comparison["variant"] == "old")
            & (comparison["model"] == model)
            & (comparison["region"] == new_worst.get("region", "__none__"))
        ]
        new_at_old_row = new_at_old.iloc[0] if not new_at_old.empty else pd.Series(dtype=float)
        old_at_new_row = old_at_new.iloc[0] if not old_at_new.empty else pd.Series(dtype=float)

        def value(row: Mapping[str, Any], key: str) -> float:
            return float(row.get(key, np.nan)) if row is not None else float("nan")

        old_rmse = value(old_global, "rmse_mm")
        new_rmse = value(new_global, "rmse_mm")
        old_p95 = value(old_global, "p95_mm")
        new_p95 = value(new_global, "p95_mm")
        old_max = value(old_global, "max_abs_mm")
        new_max = value(new_global, "max_abs_mm")
        old_worst_rmse = value(old_worst, "rmse_mm")
        new_worst_rmse = value(new_worst, "rmse_mm")
        new_on_old_worst = value(new_at_old_row, "rmse_mm")
        old_on_new_worst = value(old_at_new_row, "rmse_mm")
        rows.append(
            {
                "model": model,
                "old_global_rmse_mm": old_rmse,
                "new_global_rmse_mm": new_rmse,
                "delta_global_rmse_mm_old_minus_new": old_rmse - new_rmse,
                "old_global_p95_mm": old_p95,
                "new_global_p95_mm": new_p95,
                "delta_global_p95_mm_old_minus_new": old_p95 - new_p95,
                "old_global_max_abs_mm": old_max,
                "new_global_max_abs_mm": new_max,
                "delta_global_max_abs_mm_old_minus_new": old_max - new_max,
                "old_worst_v_bin": str(old_worst.get("region", "")),
                "old_worst_rmse_mm": old_worst_rmse,
                "new_rmse_at_old_worst_v_bin_mm": new_on_old_worst,
                "delta_at_old_worst_v_bin_mm": old_worst_rmse - new_on_old_worst,
                "new_worst_v_bin": str(new_worst.get("region", "")),
                "new_worst_rmse_mm": new_worst_rmse,
                "old_rmse_at_new_worst_v_bin_mm": old_on_new_worst,
                "delta_at_new_worst_v_bin_mm": old_on_new_worst - new_worst_rmse,
            }
        )
    return pd.DataFrame(rows)


def relative_improvement(old: float, new: float) -> float:
    if not np.isfinite(old) or not np.isfinite(new) or old <= 0:
        return float("nan")
    return (old - new) / old


def classify_effect(effects: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    global_rel = np.array(
        [
            relative_improvement(row.old_global_rmse_mm, row.new_global_rmse_mm)
            for row in effects.itertuples()
        ],
        dtype=float,
    )
    edge_rel = np.array(
        [
            relative_improvement(row.old_worst_rmse_mm, row.new_rmse_at_old_worst_v_bin_mm)
            for row in effects.itertuples()
        ],
        dtype=float,
    )
    meaningful_global = np.isfinite(global_rel) & (global_rel >= 0.10)
    meaningful_edge = np.isfinite(edge_rel) & (edge_rel >= 0.10)
    severe_global_degradation = np.isfinite(global_rel) & (global_rel <= -0.10)
    severe_edge_degradation = np.isfinite(edge_rel) & (edge_rel <= -0.20)
    strong_count = int(np.count_nonzero(meaningful_global & meaningful_edge))
    moderate_count = int(np.count_nonzero(meaningful_global | meaningful_edge))
    median_global = float(np.nanmedian(global_rel)) if np.any(np.isfinite(global_rel)) else float("nan")
    median_edge = float(np.nanmedian(edge_rel)) if np.any(np.isfinite(edge_rel)) else float("nan")

    if strong_count >= 2 and not np.any(severe_global_degradation | severe_edge_degradation):
        label = "SIGNIFICANT"
    elif moderate_count >= 1 and not np.all(severe_global_degradation):
        label = "MODERATE"
    else:
        label = "WEAK"
    return label, {
        "global_relative_improvement": global_rel.tolist(),
        "old_worst_relative_improvement": edge_rel.tolist(),
        "strong_count": strong_count,
        "moderate_count": moderate_count,
        "median_global_relative_improvement": median_global,
        "median_old_worst_relative_improvement": median_edge,
    }


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def generate_report(
    path: Path,
    cfg: Mapping[str, Any],
    config_path: Path,
    intrinsics_path: Path,
    fit_root: Path,
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    comparison: pd.DataFrame,
    effects: pd.DataFrame,
    effect_label: str,
    effect_diagnostics: Mapping[str, Any],
) -> None:
    lines = [
        "# 001–018 board mask A/B 三模型拟合比较",
        "",
        f"`MASK_FIX_EFFECT = {effect_label}`",
        "",
        "## Scope",
        "",
        "- 仅读取 FIT 001–018 的 chess / nolaser / laser 三联图；未读取 Validation，也未读取 025–036、049–054。",
        f"- FIT root：`{fit_root}`",
        f"- 内参：`{intrinsics_path}`（来自用户指定的 intrinsics 路径）",
        f"- 配置基线：`{config_path}`",
        "- Old：inner-corner convex hull，`margin_px=-2`。",
        "- New：PnP 投影完整棋盘物理边界，11×8 内角点、20 mm 格距，X=[-20,220] mm、Y=[-20,160] mm，inset=0 mm。",
        "- 两个分支均使用同一 PnP、内参、Steger、vertical 每 row 单点、continuity、每帧最多 900 点和 frame-balanced weighting；仅切换 board mask。",
        "- Old/New 各自从头拟合 global_plane、quadratic_graph、circular_cone；没有训练 C1，也没有覆盖 Frozen C0。",
        "- residual 定义：模型从同一分支激光像素反算的点到对应 PnP 棋盘真实平面的有符号距离（mm）。",
        "",
        "## Extracted FIT support",
        "",
        "| mask | frames | points | min v / px | max v / px | min u / px | max u / px |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Old | {old_df['frame_key'].nunique()} | {len(old_df)} | {fmt(old_df['v_px'].min(), 2)} | {fmt(old_df['v_px'].max(), 2)} | {fmt(old_df['u_px'].min(), 2)} | {fmt(old_df['u_px'].max(), 2)} |",
        f"| New | {new_df['frame_key'].nunique()} | {len(new_df)} | {fmt(new_df['v_px'].min(), 2)} | {fmt(new_df['v_px'].max(), 2)} | {fmt(new_df['u_px'].min(), 2)} | {fmt(new_df['u_px'].max(), 2)} |",
        "",
        "`point_count` 是各 mask 分支自己的重新提取结果；由于 mask 改变了采样点集合，Old/New 的误差比较不是 point-wise 配对比较，而是同一 FIT 帧集合上的独立拟合/评价比较。",
        "",
        "## Global comparison",
        "",
        "| model | Old RMSE | New RMSE | ΔRMSE (Old-New) | Old P95 | New P95 | Old max | New max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in effects.itertuples():
        lines.append(
            f"| {row.model} | {fmt(row.old_global_rmse_mm)} | {fmt(row.new_global_rmse_mm)} | {fmt(row.delta_global_rmse_mm_old_minus_new)} | {fmt(row.old_global_p95_mm)} | {fmt(row.new_global_p95_mm)} | {fmt(row.old_global_max_abs_mm)} | {fmt(row.new_global_max_abs_mm)} |"
        )

    lines += [
        "",
        "## Worst v-bin comparison",
        "",
        "| model | Old worst bin | Old RMSE | New RMSE at Old bin | Δ at Old bin | New worst bin | New worst RMSE |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in effects.itertuples():
        lines.append(
            f"| {row.model} | {row.old_worst_v_bin} | {fmt(row.old_worst_rmse_mm)} | {fmt(row.new_rmse_at_old_worst_v_bin_mm)} | {fmt(row.delta_at_old_worst_v_bin_mm)} | {row.new_worst_v_bin} | {fmt(row.new_worst_rmse_mm)} |"
        )

    lines += [
        "",
        "## 100 px v-bin detail",
        "",
        "完整 Global 与 100 px bin 指标见 `mask_ab_model_comparison.csv`。每个 bin 均记录 Bias、MAE、RMSE、P95、Max abs、有效交点率和点数。",
        "",
        "## Effect rule",
        "",
        "- SIGNIFICANT：至少 2/3 模型同时满足 Global RMSE 和 Old worst-v-bin RMSE 相对改善 ≥10%，且没有 ≥10% 的 Global 或 ≥20% 的 Old worst-bin 明显恶化。",
        "- MODERATE：至少一个模型在 Global 或 Old worst-v-bin 达到 ≥10% 改善，且未出现所有模型 Global 同时明显恶化。",
        "- WEAK：不满足上述条件。该标签描述本次 FIT-only mask A/B 影响，不代表 Validation 泛化结论。",
        "",
        f"- 统计得到的 Global RMSE 中位相对改善：{fmt(effect_diagnostics.get('median_global_relative_improvement', np.nan) * 100, 2)}%。",
        f"- 统计得到的 Old worst-v-bin RMSE 中位相对改善：{fmt(effect_diagnostics.get('median_old_worst_relative_improvement', np.nan) * 100, 2)}%。",
        f"- 同时达到阈值的模型数：{effect_diagnostics.get('strong_count', 0)}；至少一项达到阈值的模型数：{effect_diagnostics.get('moderate_count', 0)}。",
        "",
        "## Artifacts",
        "",
        "- `mask_ab_model_comparison.csv`：Old/New × 三模型的 Global 与 v=0–3000、100 px bins 指标。",
        "- `residual_vs_v_mask_ab.png`：Old/New 两个 mask 分支的 residual-v 散点与 100 px 中位趋势。",
        "- `models/old/`、`models/new/`：各分支独立拟合的三模型参数，仅作本次 A/B 审计记录。",
        "",
        f"结论：`MASK_FIX_EFFECT = {effect_label}`。本报告仅隔离 mask 对 001–018 FIT-only 拟合的影响；是否改善独立数据，仍需单独的冻结 Validation/标准件验证。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="001–018 old/new board mask A/B model comparison")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "laser_model_fit_config.daheng.yaml"),
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        help="内参文件或 intrinsics 目录；目录默认解析 calibration_result.yaml",
    )
    parser.add_argument("--fit-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "projects" / "daheng" / "outputs" / "0817" / "mask_ab_model_comparison"),
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, k, dist, image_size, intrinsics_path, fit_root = load_config_and_inputs(
        config_path,
        Path(args.intrinsics).resolve() if args.intrinsics else None,
        Path(args.fit_root).resolve() if args.fit_root else None,
    )
    if not fit_root.exists():
        raise FileNotFoundError(f"FIT root 不存在：{fit_root}")

    print(f"FIT root: {fit_root}")
    print(f"Intrinsics: {intrinsics_path}")
    print("FIT ids:", ",".join(f"{i:03d}" for i in FIT_IDS))
    print("A/B mask: old inner_corner_hull margin=-2; new full_board_physical inset=0")

    preview_dir = output_dir / "extraction_previews"
    old_df = extract_variant("old", cfg, k, dist, image_size, fit_root, preview_dir)
    old_df.to_csv(output_dir / "points_old_mask.csv", index=False, encoding="utf-8-sig")
    print(f"[OLD] {len(old_df)} points across {old_df['frame_key'].nunique()} frames")

    new_df = extract_variant("new", cfg, k, dist, image_size, fit_root, preview_dir)
    new_df.to_csv(output_dir / "points_new_mask.csv", index=False, encoding="utf-8-sig")
    print(f"[NEW] {len(new_df)} points across {new_df['frame_key'].nunique()} frames")

    all_rows: List[pd.DataFrame] = []
    all_details: Dict[Tuple[str, str], pd.DataFrame] = {}
    for variant, df in (("old", old_df), ("new", new_df)):
        print(f"Fitting {variant} models ...")
        plane, quadratic, cone = fit_models(df, cfg)
        models: List[triplets.LaserModel] = [plane, quadratic, cone]
        save_models(output_dir, variant, models)
        comparison, details = evaluate_variant(variant, df, models, plane)
        all_rows.append(comparison)
        for name, detail in details.items():
            all_details[(variant, name)] = detail
        print(f"[{variant}] fitted {', '.join(model.name for model in models)}")

    comparison = pd.concat(all_rows, ignore_index=True)
    comparison = comparison.sort_values(["variant", "model", "region"]).reset_index(drop=True)
    comparison.to_csv(output_dir / "mask_ab_model_comparison.csv", index=False, encoding="utf-8-sig")

    effects = effect_table(comparison)
    effects.to_csv(output_dir / "mask_ab_effect_summary.csv", index=False, encoding="utf-8-sig")
    effect_label, effect_diagnostics = classify_effect(effects)
    plot_residual_vs_v(output_dir / "residual_vs_v_mask_ab.png", all_details)
    generate_report(
        output_dir / "report.md",
        cfg,
        config_path,
        intrinsics_path,
        fit_root,
        old_df,
        new_df,
        comparison,
        effects,
        effect_label,
        effect_diagnostics,
    )

    print(f"MASK_FIX_EFFECT = {effect_label}")
    print(f"Output: {output_dir}")
    print(effects.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
