#!/usr/bin/env python3
"""FIT-only Plane/Quadratic/Circular-Cone baseline with the full board mask.

This runner deliberately inventories only the three requested FIT roots.  It
does not call the formal train/validation CLI, so no Validation image can be
opened accidentally.  The extraction, frame-balanced model fitting, and
ray/board error definitions are delegated to ``fit_laser_models_from_triplets``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import fit_laser_models_from_triplets as triplets  # noqa: E402


FIT_GROUPS = (
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
ROLES = ("chess", "nolaser", "laser")
PATTERNS = {
    "chess": "chess {id:03d}.tif",
    "background": "nolaser {id:03d}.tif",
    "laser": "laser {id:03d}.tif",
}
REGION_BOUNDS = {
    "Top": (0.0, 300.0),
    "Middle": (300.0, 2700.0),
    "Bottom": (2700.0, 3000.0),
}
REGION_ORDER = ("Global", "Top", "Middle", "Bottom")
MODEL_LABELS = {
    "global_plane": "PLANE",
    "quadratic_graph": "QUADRATIC",
    "circular_cone": "CONE",
}
DEFAULT_CONFIG = ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_PCA = (
    ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "c1_independent_validation"
    / "frozen_c1_model.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "fit_model_baseline_full_board"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pca-model", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def inventory_group(root: Path, ids: Sequence[int]) -> dict[str, dict[str, Any]]:
    """Resolve exactly one requested FIT group; never enumerate Validation."""
    groups: dict[str, dict[str, Any]] = {}
    for image_id in ids:
        frame_id = f"{image_id:03d}"
        groups[frame_id] = {}
        for role in ROLES:
            path = root / f"{role} {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            groups[frame_id][role] = {"path": path}
    return groups


def load_frozen_pca(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pca = data.get("pca_s")
    required = ("center_xn", "center_yn", "axis_s_xn", "axis_s_yn")
    if not isinstance(pca, Mapping) or any(key not in pca for key in required):
        raise ValueError(f"Frozen PCA 文件缺少 pca_s 定义：{path}")
    return {key: float(pca[key]) for key in required}


def add_frozen_coordinates(df: pd.DataFrame, k: np.ndarray, d: np.ndarray, pca: Mapping[str, float]) -> pd.DataFrame:
    result = df.copy()
    u = result["u_px"].to_numpy(dtype=float)
    v = result["v_px"].to_numpy(dtype=float)
    rays = triplets.pixels_to_rays(u, v, k, d)
    centered = rays[:, :2] - np.asarray([pca["center_xn"], pca["center_yn"]], dtype=float)
    axis_s = np.asarray([pca["axis_s_xn"], pca["axis_s_yn"]], dtype=float)
    result["pca_s"] = centered @ axis_s

    v_values = result["v_px"].to_numpy(dtype=float)
    result["region"] = np.select(
        [
            (v_values >= REGION_BOUNDS["Top"][0]) & (v_values < REGION_BOUNDS["Top"][1]),
            (v_values >= REGION_BOUNDS["Middle"][0]) & (v_values < REGION_BOUNDS["Middle"][1]),
            (v_values >= REGION_BOUNDS["Bottom"][0]) & (v_values < REGION_BOUNDS["Bottom"][1]),
        ],
        ["Top", "Middle", "Bottom"],
        default="Outside",
    )
    result["frame_id"] = result["image_id"].map(lambda value: f"{int(value):03d}")
    return result


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def trend_stats(x: np.ndarray, y: np.ndarray, bins: int = 20) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.ptp(x) <= 1.0e-12:
        return {
            "count": int(x.size),
            "slope": math.nan,
            "rho": math.nan,
            "binned_median_range": math.nan,
            "max_adjacent_median_jump": math.nan,
            "within_bin_rmse": math.nan,
        }
    slope = float(np.polyfit(x, y, 1)[0])
    rho = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1.0e-12 else 0.0
    edges = np.linspace(float(np.min(x)), float(np.max(x)), min(bins, max(4, x.size // 50)) + 1)
    bin_index = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)
    medians: list[float] = []
    residuals: list[float] = []
    for index in range(len(edges) - 1):
        mask = bin_index == index
        if not np.any(mask):
            continue
        median = float(np.median(y[mask]))
        medians.append(median)
        residuals.extend((y[mask] - median).tolist())
    median_array = np.asarray(medians, dtype=float)
    jumps = np.abs(np.diff(median_array)) if median_array.size > 1 else np.asarray([], dtype=float)
    return {
        "count": int(x.size),
        "slope": slope,
        "rho": rho,
        "binned_median_range": float(np.ptp(median_array)) if median_array.size else math.nan,
        "max_adjacent_median_jump": float(np.max(jumps)) if jumps.size else math.nan,
        "within_bin_rmse": float(np.sqrt(np.mean(np.asarray(residuals) ** 2))) if residuals else math.nan,
    }


def fit_cone_diagnostic(
    df: pd.DataFrame,
    cone_cfg: Mapping[str, Any],
    label: str,
    global_cone: triplets.CircularConeModel,
) -> dict[str, Any]:
    points, _, frame_ids = triplets.dataframe_arrays(df)
    try:
        plane = triplets.PlaneModel()
        plane.fit(points, frame_ids)
        cone = triplets.CircularConeModel(dict(cone_cfg))
        cone.fit(points, frame_ids, plane=plane)
        params = cone.to_dict()
        dot = float(np.clip(abs(np.dot(cone.axis, global_cone.axis)), -1.0, 1.0))
        return {
            "subset": label,
            "status": "OK",
            "frame_count": int(df["frame_key"].nunique()),
            "point_count": int(len(df)),
            "fit_success": bool(cone.fit_success),
            "optimizer_cost": float(cone.cost),
            "half_apex_angle_deg": float(cone.alpha_deg),
            "axis_x": float(cone.axis[0]),
            "axis_y": float(cone.axis[1]),
            "axis_z": float(cone.axis[2]),
            "apex_x_mm": float(cone.apex[0]),
            "apex_y_mm": float(cone.apex[1]),
            "apex_z_mm": float(cone.apex[2]),
            "axis_angle_to_global_deg": math.degrees(math.acos(dot)),
            "model_parameters": params,
        }
    except Exception as exc:
        return {
            "subset": label,
            "status": "FAIL",
            "frame_count": int(df["frame_key"].nunique()),
            "point_count": int(len(df)),
            "fit_success": False,
            "error": str(exc),
        }


def evaluate_models(
    df: pd.DataFrame,
    models: Sequence[triplets.LaserModel],
    plane: triplets.PlaneModel,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    comparison_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    trend_rows: list[dict[str, Any]] = []
    for model in models:
        for region in REGION_ORDER:
            region_df = df if region == "Global" else df[df["region"] == region]
            if region_df.empty:
                continue
            metrics, detail = triplets.evaluate_model(model, region_df, plane)
            metrics["region"] = region
            metrics["frame_count"] = int(region_df["frame_key"].nunique())
            metrics["v_min_px"] = float(region_df["v_px"].min())
            metrics["v_max_px"] = float(region_df["v_px"].max())
            metrics["pca_s_min"] = float(region_df["pca_s"].min())
            metrics["pca_s_max"] = float(region_df["pca_s"].max())
            metrics["bias_mm"] = metrics["board_mean_signed_mm"]
            metrics["mae_mm"] = metrics["board_mae_mm"]
            metrics["rmse_mm"] = metrics["board_rmse_mm"]
            metrics["p95_mm"] = metrics["board_p95_abs_mm"]
            metrics["max_abs_mm"] = metrics["board_max_abs_mm"]
            if region == "Global":
                v_trend = trend_stats(detail["v_px"].to_numpy(), detail["board_error_mm"].to_numpy())
                detail["pca_s"] = df.loc[detail.index, "pca_s"].to_numpy(dtype=float)
                s_trend = trend_stats(detail["pca_s"].to_numpy(), detail["board_error_mm"].to_numpy())
                trend_rows.extend(
                    [
                        {"model": model.name, "region": region, "variable": "v", **v_trend},
                        {"model": model.name, "region": region, "variable": "s", **s_trend},
                    ]
                )
                metrics.update(
                    {
                        "v_trend_slope": v_trend["slope"],
                        "v_trend_rho": v_trend["rho"],
                        "v_binned_median_range_mm": v_trend["binned_median_range"],
                        "v_within_bin_rmse_mm": v_trend["within_bin_rmse"],
                        "s_trend_slope": s_trend["slope"],
                        "s_trend_rho": s_trend["rho"],
                        "s_binned_median_range_mm": s_trend["binned_median_range"],
                        "s_within_bin_rmse_mm": s_trend["within_bin_rmse"],
                    }
                )
            else:
                detail["pca_s"] = df.loc[detail.index, "pca_s"].to_numpy(dtype=float)
            detail["region"] = region
            detail_rows.append(detail)
            comparison_rows.append(metrics)
    return pd.DataFrame(comparison_rows), pd.concat(detail_rows, ignore_index=True), trend_rows


def plot_residual_trend(detail: pd.DataFrame, variable: str, xlabel: str, output: Path) -> None:
    plt.figure(figsize=(11, 6.5))
    for model, group in detail[detail["region"] == "Global"].groupby("model"):
        clean = group[[variable, "board_error_mm"]].dropna().sort_values(variable)
        if clean.empty:
            continue
        sample = clean.iloc[:: max(1, len(clean) // 4000)]
        plt.scatter(sample[variable], sample["board_error_mm"], s=3, alpha=0.12, label=f"{model} points")
        if len(clean) >= 4 and np.ptp(clean[variable].to_numpy()) > 1.0e-12:
            bins = pd.qcut(clean[variable], q=min(40, max(5, len(clean) // 50)), duplicates="drop")
            median = clean.groupby(bins, observed=True).median(numeric_only=True)
            plt.plot(median[variable], median["board_error_mm"], linewidth=2.0, label=f"{model} median")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("FIT-only board reconstruction error / mm")
    plt.title("FIT-only residual trend under full-board physical mask")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def fmt(value: Any, digits: int = 5) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def selection_scores(comparison: pd.DataFrame, trends: pd.DataFrame, cone_stable: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(comparison["model"].unique()):
        subset = comparison[comparison["model"] == model].set_index("region")
        edge = subset.loc[["Top", "Bottom"]]
        global_row = subset.loc["Global"]
        model_trends = trends[trends["model"] == model]
        trend_penalty = float(model_trends["binned_median_range"].max()) if not model_trends.empty else math.inf
        bias_range = float(edge["bias_mm"].max() - edge["bias_mm"].min())
        score = (
            float(global_row["rmse_mm"])
            + float(edge["rmse_mm"].max())
            + 0.5 * float(edge["p95_mm"].max())
            + 0.5 * bias_range
            + 0.5 * trend_penalty
        )
        if model == "circular_cone" and not cone_stable:
            score += 1.0
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "selection_score": score,
                "global_rmse_mm": float(global_row["rmse_mm"]),
                "edge_rmse_max_mm": float(edge["rmse_mm"].max()),
                "edge_p95_max_mm": float(edge["p95_mm"].max()),
                "edge_bias_range_mm": bias_range,
                "trend_penalty_mm": trend_penalty,
                "cone_stability_penalty": 1.0 if model == "circular_cone" and not cone_stable else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)


def choose_winner(scores: pd.DataFrame, cone_stable: bool) -> str:
    if scores.empty:
        return "UNRESOLVED"
    best = scores.iloc[0]
    if best["model"] == "circular_cone" and not cone_stable:
        return "UNRESOLVED"
    if len(scores) > 1:
        second = float(scores.iloc[1]["selection_score"])
        margin = second - float(best["selection_score"])
        if margin < max(0.005, 0.05 * second):
            return "UNRESOLVED"
    return MODEL_LABELS[str(best["model"])]


def render_report(
    output: Path,
    config_path: Path,
    pca_path: Path,
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    trend_rows: Sequence[Mapping[str, Any]],
    cone_rows: Sequence[Mapping[str, Any]],
    scores: pd.DataFrame,
    winner: str,
) -> None:
    trend_df = pd.DataFrame(trend_rows)
    cone_df = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "model_parameters"}
            for row in cone_rows
        ]
    )
    cone_group_rows = cone_df[cone_df["subset"] != "global"] if not cone_df.empty else cone_df
    cone_success = cone_group_rows[cone_group_rows["status"] == "OK"] if not cone_group_rows.empty else cone_group_rows
    cone_group = cone_success
    cone_alpha_range = (
        float(cone_group["half_apex_angle_deg"].max() - cone_group["half_apex_angle_deg"].min())
        if not cone_group.empty
        else math.nan
    )
    cone_axis_max = float(cone_group["axis_angle_to_global_deg"].max()) if not cone_group.empty else math.nan
    cone_stable = bool(
        len(cone_group_rows) == 3
        and len(cone_success) == len(cone_group_rows)
        and cone_alpha_range <= 2.0
        and cone_axis_max <= 5.0
    )

    lines = [
        "# FIT-only 三模型基线（完整棋盘物理 mask）",
        "",
        f"`FIT_MODEL_WINNER = {winner}`",
        "",
        "## Scope",
        "",
        "- 仅读取 FIT：001–018、025–036、049–054，共 36 帧；没有读取任何 Validation 图像。",
        f"- 配置：`{config_path}`；mask mode 固定为 `full_board_physical`，inset=0.0 mm。",
        "- 棋盘：11×8 内角点、20 mm；物理边界为 X=[-20,220] mm、Y=[-20,160] mm。",
        "- Steger、vertical 每 row 单点、continuity、900 点上限和 frame-balanced weighting 沿用正式流程。",
        f"- 有效 FIT 标定点：{len(df)}；有效 frame：{df['frame_key'].nunique()}。",
        f"- residual 定义：模型射线重建点到该点对应 PnP 棋盘真平面的有符号距离（mm）。",
        f"- residual-v/s 中的 s 只使用冻结 PCA 定义：`{pca_path}`；不应用或训练 C1。",
        "- 当前 Frozen C0 未覆盖、未修改。",
        "",
        "## Region definition",
        "",
        "- Top: `0 <= v < 300`；Middle: `300 <= v < 2700`；Bottom: `2700 <= v < 3000`。",
        "- Global/区域指标均评价同一个“全 FIT 拟合”的模型，不对 Top/Middle/Bottom 单独重新拟合模型。",
        "",
        "## Global / regional metrics",
        "",
        "| model | region | n | frames | bias / mm | MAE / mm | RMSE / mm | P95 / mm | max abs / mm | valid rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {MODEL_LABELS.get(str(row['model']), row['model'])} | {row['region']} | {int(row['total_points'])} | "
            f"{int(row['frame_count'])} | {fmt(row['bias_mm'])} | {fmt(row['mae_mm'])} | {fmt(row['rmse_mm'])} | "
            f"{fmt(row['p95_mm'])} | {fmt(row['max_abs_mm'])} | {fmt(row['valid_rate'], 4)} |"
        )

    lines.extend(["", "## FIT-only residual trends", ""])
    lines.append("| model | variable | slope | rho | binned median range / mm | max adjacent jump / mm | within-bin RMSE / mm |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in trend_rows:
        lines.append(
            f"| {MODEL_LABELS.get(str(row['model']), row['model'])} | {row['variable']} | {fmt(row['slope'], 7)} | "
            f"{fmt(row['rho'], 5)} | {fmt(row['binned_median_range'])} | {fmt(row['max_adjacent_median_jump'])} | "
            f"{fmt(row['within_bin_rmse'])} |"
        )

    lines.extend(["", "## Circular Cone parameter diagnostic", ""])
    lines.append("| subset | status | frames | points | success | cost | half angle / deg | axis angle to global / deg | apex / mm |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---|")
    for row in cone_rows:
        apex = "NA"
        if row.get("status") == "OK":
            apex = f"[{fmt(row.get('apex_x_mm'))}, {fmt(row.get('apex_y_mm'))}, {fmt(row.get('apex_z_mm'))}]"
        lines.append(
            f"| {row.get('subset')} | {row.get('status')} | {row.get('frame_count', 'NA')} | {row.get('point_count', 'NA')} | "
            f"{row.get('fit_success', 'NA')} | {fmt(row.get('optimizer_cost'))} | {fmt(row.get('half_apex_angle_deg'))} | "
            f"{fmt(row.get('axis_angle_to_global_deg'))} | {apex} |"
        )
    lines.extend(
        [
            "",
            f"- Cone stability diagnostic: `{'PASS' if cone_stable else 'FAIL'}`; subset half-angle range={fmt(cone_alpha_range)} deg, maximum axis deviation={fmt(cone_axis_max)} deg.",
            "- 该稳定性检查是 FIT 子集敏感性诊断，不是 Validation 泛化证明。",
            "",
            "## Candidate selection",
            "",
            "评分同时考虑 Global RMSE、Top/Bottom worst RMSE、edge P95、Top/Bottom bias range 和 v/s 分箱趋势幅度；Cone 若子集参数不稳定会增加惩罚。",
            "",
            "| candidate | score | Global RMSE | edge worst RMSE | edge worst P95 | edge bias range | trend penalty |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in scores.iterrows():
        lines.append(
            f"| {row['label']} | {fmt(row['selection_score'])} | {fmt(row['global_rmse_mm'])} | {fmt(row['edge_rmse_max_mm'])} | "
            f"{fmt(row['edge_p95_max_mm'])} | {fmt(row['edge_bias_range_mm'])} | {fmt(row['trend_penalty_mm'])} |"
        )
    lines.extend(
        [
            "",
            f"FIT-only 候选结论：`FIT_MODEL_WINNER = {winner}`。",
            "该结论只用于选择下一步 C0 候选，不能替代独立 Validation；进入实际使用前仍需在冻结 Validation 或标准件数据上复核。",
            "",
            "## Artifacts",
            "",
            "- `models/global_plane.yaml`, `models/quadratic_graph.yaml`, `models/circular_cone.yaml`：三模型参数。",
            "- `model_parameters.json`：三模型参数及 Cone 子集诊断。",
            "- `calibration_points_fit.csv`：新 mask 下的 FIT 标定点。",
            "- `model_comparison_fit.csv`：Global/Top/Middle/Bottom 指标。",
            "- `residual_vs_v.png`、`residual_vs_s.png`：FIT-only 残差趋势。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    pca_path = args.pca_model.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"输出目录非空；如需覆盖本次基线请显式使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    cfg = triplets.safe_yaml_load(config_path)
    intrinsics_path = Path(cfg["intrinsics"])
    if not intrinsics_path.is_absolute():
        intrinsics_path = (config_path.parent / intrinsics_path).resolve()
    k, d, image_size = triplets.load_intrinsics(intrinsics_path)
    board_cfg = cfg["board"]
    extraction_cfg = dict(cfg.get("extraction", {}))
    mask_mode = str(extraction_cfg.get("board_mask_mode", triplets.FULL_BOARD_PHYSICAL)).strip().lower()
    inset_mm = float(extraction_cfg.get("board_mask_inset_mm", 0.0))
    if mask_mode != triplets.FULL_BOARD_PHYSICAL or abs(inset_mm) > 1.0e-12:
        raise RuntimeError(
            "本基线要求 Task 8A-1 的 full_board_physical、0 mm inset；"
            f"当前为 mode={mask_mode!r}, inset_mm={inset_mm}"
        )
    pca = load_frozen_pca(pca_path)

    frames: list[pd.DataFrame] = []
    for group_name, root, ids in FIT_GROUPS:
        groups = inventory_group(root, ids)
        dataset_cfg = {"root": str(root), "ids": list(ids)}
        df = triplets.process_dataset(
            group_name,
            dataset_cfg,
            PATTERNS,
            k,
            d,
            image_size,
            board_cfg,
            extraction_cfg,
            "vertical",
            preview_dir,
        )
        expected = {str(value) for value in ids}
        actual = set(df["image_id"].astype(str))
        if actual != expected:
            raise RuntimeError(f"{group_name} 提取不完整：expected={sorted(expected)}, actual={sorted(actual)}")
        df["fit_group"] = group_name
        df["split"] = "fit"
        frames.append(df)

    fit_df = add_frozen_coordinates(pd.concat(frames, ignore_index=True), k, d, pca)
    if fit_df["region"].eq("Outside").any():
        raise RuntimeError("存在不属于 Top/Middle/Bottom 的 FIT 点，请检查 v 范围")
    fit_df.to_csv(output_dir / "calibration_points_fit.csv", index=False, encoding="utf-8-sig")

    points, _, frame_ids = triplets.dataframe_arrays(fit_df)
    plane = triplets.PlaneModel()
    plane.fit(points, frame_ids)
    quadratic = triplets.QuadraticGraphModel(
        ridge=float(cfg.get("models", {}).get("quadratic", {}).get("ridge", 1.0e-10))
    )
    quadratic.fit(points, frame_ids, plane=plane)
    cone_cfg = cfg.get("models", {}).get("cone", {})
    cone = triplets.CircularConeModel(cone_cfg)
    cone.fit(points, frame_ids, plane=plane)
    models: list[triplets.LaserModel] = [plane, quadratic, cone]

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_parameters: dict[str, Any] = {}
    for model in models:
        params = model.to_dict()
        model_parameters[model.name] = params
        triplets.save_yaml(model_dir / f"{model.name}.yaml", params)

    comparison, detail, trend_rows = evaluate_models(fit_df, models, plane)
    cone_rows: list[dict[str, Any]] = [
        fit_cone_diagnostic(fit_df, cone_cfg, "global", cone)
    ]
    for group_name, _, _ in FIT_GROUPS:
        cone_rows.append(fit_cone_diagnostic(fit_df[fit_df["fit_group"] == group_name], cone_cfg, group_name, cone))
    cone_success = [row for row in cone_rows if row.get("status") == "OK" and row.get("subset") != "global"]
    cone_alpha_range = (
        max(float(row["half_apex_angle_deg"]) for row in cone_success)
        - min(float(row["half_apex_angle_deg"]) for row in cone_success)
        if len(cone_success) == 3
        else math.inf
    )
    cone_axis_max = max((float(row.get("axis_angle_to_global_deg", math.inf)) for row in cone_success), default=math.inf)
    cone_stable = len(cone_success) == 3 and cone_alpha_range <= 2.0 and cone_axis_max <= 5.0

    trend_df = pd.DataFrame(trend_rows)
    scores = selection_scores(comparison, trend_df, cone_stable)
    winner = choose_winner(scores, cone_stable)
    comparison["selection_score"] = np.nan
    for _, row in scores.iterrows():
        comparison.loc[comparison["model"] == row["model"], "selection_score"] = float(row["selection_score"])
    comparison.to_csv(output_dir / "model_comparison_fit.csv", index=False, encoding="utf-8-sig")
    trend_df.to_csv(output_dir / "spatial_trends_fit.csv", index=False, encoding="utf-8-sig")
    plot_residual_trend(detail, "v_px", "v / px", output_dir / "residual_vs_v.png")
    plot_residual_trend(detail, "pca_s", "Frozen PCA s", output_dir / "residual_vs_s.png")

    model_parameters["cone_subset_diagnostics"] = [
        {key: value for key, value in row.items() if key != "model_parameters"}
        for row in cone_rows
    ]
    model_parameters["fit_model_winner"] = winner
    model_parameters["validation_read"] = False
    (output_dir / "model_parameters.json").write_text(
        json.dumps(model_parameters, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    render_report(
        output_dir / "report.md",
        config_path,
        pca_path,
        fit_df,
        comparison.sort_values(["model", "region"], key=lambda col: col.map({name: i for i, name in enumerate(REGION_ORDER)}) if col.name == "region" else col),
        trend_rows,
        cone_rows,
        scores,
        winner,
    )
    print(
        {
            "output_dir": str(output_dir),
            "frames": int(fit_df["frame_key"].nunique()),
            "points": int(len(fit_df)),
            "FIT_MODEL_WINNER": winner,
            "cone_stable": cone_stable,
            "validation_read": False,
        }
    )


if __name__ == "__main__":
    run(parse_args())
