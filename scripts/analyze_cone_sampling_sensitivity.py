#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-36 Circular Cone fit_max_points sensitivity without Q/Plane refits.

The 3000-point result is read from the existing 0817 grouped-CV artifacts.
Only Cone fits at 6000, 12000 and all feasible training points are computed.
No image, Validation, Quadratic, or Plane candidate is run.  The existing
per-fold plane parameters are used as the same fixed Cone orientation/root
hint that the 3000-point artifact used; no new PlaneModel.fit is called.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
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


FULL_IDS = tuple(
    [f"{value:03d}" for value in range(1, 19)]
    + [f"{value:03d}" for value in range(25, 37)]
    + [f"{value:03d}" for value in range(49, 55)]
)
FOLD_COUNT = 6
V_MIN = 0.0
V_MAX = 3000.0
BIN_WIDTH = 100.0
BIN_COUNT = 30
MODEL_NAME = "circular_cone"

DEFAULT_CONFIG = ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_POINTS = ROOT / "projects" / "daheng" / "outputs" / "0817" / "full_fit_v_coverage_audit" / "full_fit_points.csv"
DEFAULT_CV_DIR = ROOT / "projects" / "daheng" / "outputs" / "0817" / "grouped_cv_model_comparison"
DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0818" / "cone_sampling_sensitivity"


@dataclass
class ReusedPlaneHint:
    normal: np.ndarray
    d: float
    z_range: Tuple[float, float]

    def intersect_rays(self, rays: np.ndarray, lambda_hint: np.ndarray | None = None) -> np.ndarray:
        denom = rays @ self.normal
        lam = np.full(rays.shape[0], np.nan, dtype=float)
        valid = np.abs(denom) > 1.0e-12
        lam[valid] = -self.d / denom[valid]
        lam[(lam <= 0) | ~np.isfinite(lam)] = np.nan
        return lam


def normalize_frame_id(value: Any) -> str:
    return f"{int(value):03d}"


def metric_values(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"point_count": 0, "bias_mm": np.nan, "mae_mm": np.nan, "rmse_mm": np.nan, "p95_mm": np.nan, "max_abs_mm": np.nan}
    absolute = np.abs(array)
    return {
        "point_count": int(array.size),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array**2))),
        "p95_mm": float(np.percentile(absolute, 95)),
        "max_abs_mm": float(np.max(absolute)),
    }


def normalize_points(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"frame_id", "frame_key", "Xc_mm", "Yc_mm", "Zc_mm", "ray_x", "ray_y", "ray_z", "v_px", "board_nx", "board_ny", "board_nz", "board_d_mm"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RuntimeError(f"Full-36 point table 缺少字段：{missing}")
    data["frame_id"] = data["frame_id"].map(normalize_frame_id)
    if sorted(data["frame_id"].unique()) != sorted(FULL_IDS):
        raise RuntimeError("Full-36 point table pose 集合异常")
    return data


def make_fold_assignment(frame_ids: Sequence[str]) -> Dict[str, int]:
    ordered = sorted({str(value) for value in frame_ids})
    if len(ordered) != len(FULL_IDS):
        raise RuntimeError("Full-36 fold assignment pose 数异常")
    return {frame_id: index % FOLD_COUNT for index, frame_id in enumerate(ordered)}


def pooled_group_rmse(detail: pd.DataFrame, column: str) -> Tuple[float, float]:
    values: List[float] = []
    for _, group in detail.groupby(column):
        metric = metric_values(group["board_error_mm"].to_numpy(dtype=float))
        if np.isfinite(metric["rmse_mm"]):
            values.append(float(metric["rmse_mm"]))
    if not values:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def build_bin_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    data = detail[np.isfinite(detail["v_px"]) & (detail["v_px"] >= V_MIN) & (detail["v_px"] < V_MAX)].copy()
    data["v_bin_index"] = np.floor((data["v_px"] - V_MIN) / BIN_WIDTH).astype(int)
    for index in range(BIN_COUNT):
        group = data[data["v_bin_index"] == index]
        metric = metric_values(group["board_error_mm"].to_numpy(dtype=float))
        valid = group["valid"].to_numpy(dtype=bool)
        frame_mean, frame_std = pooled_group_rmse(group, "frame_id") if not group.empty else (np.nan, np.nan)
        fold_mean, fold_std = pooled_group_rmse(group, "fold") if not group.empty else (np.nan, np.nan)
        rows.append(
            {
                "v_bin": f"v_{index * 100:04d}_{(index + 1) * 100:04d}",
                "v_bin_lo_px": float(index * 100),
                "v_bin_hi_px": float((index + 1) * 100),
                "point_count": int(len(group)),
                "unique_frame_count": int(group["frame_id"].nunique()),
                "fold_count": int(group["fold"].nunique()),
                "valid_intersections": int(np.count_nonzero(valid)),
                "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
                **metric,
                "mean_frame_rmse_mm": frame_mean,
                "std_frame_rmse_mm": frame_std,
                "mean_fold_rmse_mm": fold_mean,
                "std_fold_rmse_mm": fold_std,
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(detail: pd.DataFrame, bins: pd.DataFrame, train_count: int) -> Dict[str, Any]:
    metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
    valid = detail["valid"].to_numpy(dtype=bool)
    frame_mean, frame_std = pooled_group_rmse(detail, "frame_id")
    fold_mean, fold_std = pooled_group_rmse(detail, "fold")
    valid_bins = bins[np.isfinite(bins["rmse_mm"])].copy()
    worst = valid_bins.sort_values(["rmse_mm", "p95_mm"], ascending=False).iloc[0] if not valid_bins.empty else pd.Series(dtype=float)
    clean_bias = bins[np.isfinite(bins["bias_mm"])]
    return {
        "train_frame_count": int(train_count),
        "heldout_frame_count": int(detail["frame_id"].nunique()),
        "fold_count": int(detail["fold"].nunique()),
        "valid_intersections": int(np.count_nonzero(valid)),
        "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
        "point_count": int(len(detail)),
        "global_bias_mm": metric["bias_mm"],
        "global_mae_mm": metric["mae_mm"],
        "global_rmse_mm": metric["rmse_mm"],
        "global_p95_mm": metric["p95_mm"],
        "global_max_abs_mm": metric["max_abs_mm"],
        "frame_rmse_mean_mm": frame_mean,
        "frame_rmse_std_mm": frame_std,
        "fold_rmse_mean_mm": fold_mean,
        "fold_rmse_std_mm": fold_std,
        "worst_v_bin": str(worst.get("v_bin", "")),
        "worst_v_bin_rmse_mm": float(worst.get("rmse_mm", np.nan)),
        "worst_v_bin_p95_mm": float(worst.get("p95_mm", np.nan)),
        "worst_v_bin_max_abs_mm": float(worst.get("max_abs_mm", np.nan)),
        "v_bias_range_mm": float(np.ptp(clean_bias["bias_mm"])) if not clean_bias.empty else np.nan,
    }


def load_reused_artifacts(cv_dir: Path, points: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[int, ReusedPlaneHint], Dict[str, Any]]:
    grouped_path = cv_dir / "grouped_cv_model_comparison.csv"
    per_bin_path = cv_dir / "per_v_bin_cv_metrics.csv"
    pointwise_path = cv_dir / "cv_pointwise_circular_cone.csv"
    parameter_path = cv_dir / "cv_fold_model_parameters.json"
    grouped = pd.read_csv(grouped_path, encoding="utf-8-sig")
    per_bin = pd.read_csv(per_bin_path, encoding="utf-8-sig")
    pointwise = pd.read_csv(pointwise_path, encoding="utf-8-sig")
    pointwise["frame_id"] = pointwise["frame_id"].map(normalize_frame_id)
    pooled = grouped[(grouped["row_type"] == "pooled_cv") & (grouped["model"] == MODEL_NAME)].copy()
    if len(pooled) != 1:
        raise RuntimeError("Full-36 3000-point Cone pooled artifact 缺失")
    pooled_row = pooled.iloc[0].to_dict()
    if int(pooled_row["point_count"]) != len(points) or len(pointwise) != len(points):
        raise RuntimeError("Full-36 3000-point point count 不一致")
    if sorted(pointwise["frame_id"].unique()) != sorted(FULL_IDS) or int(pointwise["fold"].nunique()) != FOLD_COUNT:
        raise RuntimeError("Full-36 3000-point pointwise pose/fold 集合异常")
    if not pointwise["valid"].astype(bool).all():
        raise RuntimeError("Full-36 3000-point Cone 存在无效预测，停止")
    params = json.loads(parameter_path.read_text(encoding="utf-8"))
    hints: Dict[int, ReusedPlaneHint] = {}
    for fold in range(FOLD_COUNT):
        plane = params[str(fold)]["global_plane"]
        hints[fold] = ReusedPlaneHint(
            normal=np.asarray(plane["normal"], dtype=float),
            d=float(plane["d_mm"]),
            z_range=tuple(float(value) for value in plane["z_valid_range_mm"]),
        )
    reused = {
        "setting": "3000",
        "fit_max_points": 3000,
        "sampling_status": "REUSED_EXISTING",
        "protocol_source": str(grouped_path),
        "pointwise_source": str(pointwise_path),
        "pooled_source": str(grouped_path),
        "per_bin_source": str(per_bin_path),
        "pooled_row": pooled_row,
        "grouped": grouped,
        "per_bin": per_bin,
        "pointwise_rows": int(len(pointwise)),
        "parameters": {str(fold): params[str(fold)][MODEL_NAME] for fold in range(FOLD_COUNT)},
    }
    return reused, hints, {"grouped": grouped, "per_bin": per_bin, "pointwise": pointwise, "parameters": params}


def selected_point_count(train_df: pd.DataFrame, fit_max_points: int) -> int:
    counts = train_df.groupby("frame_id").size().to_numpy(dtype=int)
    per_frame = max(20, fit_max_points // max(len(counts), 1))
    chosen = int(sum(min(int(count), per_frame) for count in counts))
    return min(chosen, int(fit_max_points))


def fit_cone_setting(
    setting: str,
    fit_max_points: int,
    points: pd.DataFrame,
    cfg: Mapping[str, Any],
    fold_map: Mapping[str, int],
    plane_hints: Mapping[int, ReusedPlaneHint],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    detail_chunks: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    fold_params: Dict[str, Any] = {}
    train_counts: List[int] = []
    selected_counts: List[int] = []
    cone_cfg_base = copy.deepcopy(cfg.get("models", {}).get("cone", {}))
    for fold in range(FOLD_COUNT):
        heldout = sorted([frame_id for frame_id in FULL_IDS if fold_map[frame_id] == fold])
        train_frames = sorted([frame_id for frame_id in FULL_IDS if fold_map[frame_id] != fold])
        train_df = points[~points["frame_id"].isin(heldout)].copy()
        test_df = points[points["frame_id"].isin(heldout)].copy()
        train_counts.append(len(train_frames))
        selected_counts.append(selected_point_count(train_df, fit_max_points))
        cone_cfg = copy.deepcopy(cone_cfg_base)
        cone_cfg["fit_max_points"] = int(fit_max_points)
        points_array, _, frame_keys = triplets.dataframe_arrays(train_df)
        cone = triplets.CircularConeModel(cone_cfg)
        cone.fit(points_array, frame_keys, plane=plane_hints[fold])  # fixed existing hint; no PlaneModel.fit
        fold_params[str(fold)] = cone.to_dict()
        _, raw_detail = triplets.evaluate_model(cone, test_df, plane_hints[fold])
        detail = raw_detail.copy()
        detail["fold"] = int(fold)
        detail["frame_id"] = test_df.loc[detail.index, "frame_id"].astype(str).to_numpy()
        detail = detail.reset_index(drop=True)
        detail_chunks.append(detail)
        fold_metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
        fold_metrics.append(
            {
                "fold": int(fold),
                "train_frame_count": len(train_frames),
                "heldout_frame_count": len(heldout),
                "heldout_frames": ",".join(heldout),
                "point_count": len(detail),
                "rmse_mm": fold_metric["rmse_mm"],
                "p95_mm": fold_metric["p95_mm"],
                "valid_rate": float(np.mean(detail["valid"].to_numpy(dtype=bool))),
            }
        )
    pooled_detail = pd.concat(detail_chunks, ignore_index=True)
    bins = build_bin_metrics(pooled_detail)
    metrics = aggregate_metrics(pooled_detail, bins, int(np.mean(train_counts)))
    metrics.update(
        {
            "setting": setting,
            "fit_max_points": int(fit_max_points),
            "sampling_status": "COMPUTED_MISSING",
            "selected_fit_points_mean": float(np.mean(selected_counts)),
            "selected_fit_points_min": int(min(selected_counts)),
            "selected_fit_points_max": int(max(selected_counts)),
            "train_point_count_per_fold": int(len(pooled_detail) * (FOLD_COUNT - 1) / FOLD_COUNT),
        }
    )
    return metrics, {"folds": fold_metrics, "detail": pooled_detail, "bins": bins}, fold_params


def vector_angle_deg(a: Sequence[float], b: Sequence[float], absolute: bool = True) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    dot = float((av / np.linalg.norm(av)) @ (bv / np.linalg.norm(bv)))
    if absolute:
        dot = abs(dot)
    return float(math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))


def parameter_summary_rows(all_params: Mapping[str, Mapping[str, Any]], fit_counts: Mapping[str, Mapping[str, int]]) -> pd.DataFrame:
    baseline = all_params["3000"]
    rows: List[Dict[str, Any]] = []
    for setting, params in all_params.items():
        for fold in range(FOLD_COUNT):
            current = params[str(fold)]
            base = baseline[str(fold)]
            apex_delta = float(np.linalg.norm(np.asarray(current["apex_camera_mm"]) - np.asarray(base["apex_camera_mm"])))
            rows.append(
                {
                    "row_type": "fold",
                    "setting": setting,
                    "fold": int(fold),
                    "fit_max_points": int(fit_counts[setting]["fit_max_points"]),
                    "selected_fit_points": int(fit_counts[setting]["selected_fit_points"]),
                    "fit_success": bool(current.get("fit_success", False)),
                    "optimizer_cost": float(current.get("optimizer_cost", np.nan)),
                    "axis_unit_x": float(current["axis_unit_camera"][0]),
                    "axis_unit_y": float(current["axis_unit_camera"][1]),
                    "axis_unit_z": float(current["axis_unit_camera"][2]),
                    "apex_x_mm": float(current["apex_camera_mm"][0]),
                    "apex_y_mm": float(current["apex_camera_mm"][1]),
                    "apex_z_mm": float(current["apex_camera_mm"][2]),
                    "half_apex_angle_deg": float(current["half_apex_angle_deg"]),
                    "axis_angle_vs_3000_deg": vector_angle_deg(current["axis_unit_camera"], base["axis_unit_camera"]),
                    "apex_delta_vs_3000_mm": apex_delta,
                    "half_angle_delta_vs_3000_deg": float(current["half_apex_angle_deg"] - base["half_apex_angle_deg"]),
                }
            )
        axes = np.asarray([params[str(fold)]["axis_unit_camera"] for fold in range(FOLD_COUNT)], dtype=float)
        apex = np.asarray([params[str(fold)]["apex_camera_mm"] for fold in range(FOLD_COUNT)], dtype=float)
        angles = [vector_angle_deg(axes[i], axes[j]) for i in range(FOLD_COUNT) for j in range(i + 1, FOLD_COUNT)]
        apex_spread = [float(np.linalg.norm(apex[i] - apex[j])) for i in range(FOLD_COUNT) for j in range(i + 1, FOLD_COUNT)]
        alpha = np.asarray([params[str(fold)]["half_apex_angle_deg"] for fold in range(FOLD_COUNT)], dtype=float)
        costs = np.asarray([params[str(fold)].get("optimizer_cost", np.nan) for fold in range(FOLD_COUNT)], dtype=float)
        finite_costs = costs[np.isfinite(costs)]
        axis_changes = [vector_angle_deg(params[str(fold)]["axis_unit_camera"], baseline[str(fold)]["axis_unit_camera"]) for fold in range(FOLD_COUNT)]
        apex_changes = [float(np.linalg.norm(np.asarray(params[str(fold)]["apex_camera_mm"]) - np.asarray(baseline[str(fold)]["apex_camera_mm"]))) for fold in range(FOLD_COUNT)]
        alpha_changes = [abs(float(params[str(fold)]["half_apex_angle_deg"] - baseline[str(fold)]["half_apex_angle_deg"])) for fold in range(FOLD_COUNT)]
        rows.append(
            {
                "row_type": "summary",
                "setting": setting,
                "fold": "ALL",
                "fit_max_points": int(fit_counts[setting]["fit_max_points"]),
                "selected_fit_points": float(fit_counts[setting]["selected_fit_points"]),
                "fit_success": bool(all(bool(params[str(fold)].get("fit_success", False)) for fold in range(FOLD_COUNT))),
                "optimizer_cost": float(np.mean(finite_costs)) if finite_costs.size else np.nan,
                "axis_angle_range_deg": float(max(angles) if angles else 0.0),
                "apex_spread_mm": float(max(apex_spread) if apex_spread else 0.0),
                "half_angle_range_deg": float(np.ptp(alpha)),
                "half_angle_std_deg": float(np.std(alpha)),
                "optimizer_cost_cv_pct": float(np.std(finite_costs) / max(abs(np.mean(finite_costs)), 1.0e-12) * 100.0) if finite_costs.size else np.nan,
                "axis_angle_vs_3000_max_deg": float(max(axis_changes) if axis_changes else 0.0),
                "apex_delta_vs_3000_max_mm": float(max(apex_changes) if apex_changes else 0.0),
                "half_angle_delta_vs_3000_max_deg": float(max(alpha_changes) if alpha_changes else 0.0),
                "fit_success_rate": float(np.mean([bool(params[str(fold)].get("fit_success", False)) for fold in range(FOLD_COUNT)])),
            }
        )
    return pd.DataFrame(rows)


def make_performance_table(reused: Mapping[str, Any], new_metrics: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    source = reused["pooled_row"]
    base = {
        "setting": "3000",
        "fit_max_points": 3000,
        "sampling_status": "REUSED_EXISTING",
        "selected_fit_points_mean": 3000.0,
        "selected_fit_points_min": 3000,
        "selected_fit_points_max": 3000,
        "train_point_count_per_fold": 27000,
        "point_count": int(source["point_count"]),
        "fold_count": int(source["fold_count"]),
        "valid_intersections": int(source["valid_intersections"]),
        "valid_rate": float(source["valid_rate"]),
        "global_bias_mm": float(source["bias_mm"]),
        "global_mae_mm": float(source["mae_mm"]),
        "global_rmse_mm": float(source["rmse_mm"]),
        "global_p95_mm": float(source["p95_mm"]),
        "global_max_abs_mm": float(source["max_abs_mm"]),
        "frame_rmse_mean_mm": float(source["mean_frame_rmse_mm"]),
        "frame_rmse_std_mm": float(source["std_frame_rmse_mm"]),
        "fold_rmse_mean_mm": float(source["mean_fold_rmse_mm"]),
        "fold_rmse_std_mm": float(source["std_fold_rmse_mm"]),
        "worst_v_bin": str(source["worst_v_bin"]),
        "worst_v_bin_rmse_mm": float(source["worst_v_bin_rmse_mm"]),
        "worst_v_bin_p95_mm": float(source["worst_v_bin_p95_mm"]),
        "worst_v_bin_max_abs_mm": float(source["worst_v_bin_max_abs_mm"]),
        "v_bias_range_mm": float(source["v_bias_range_mm"]),
        "protocol_source": "0817 grouped_cv_model_comparison.csv",
    }
    rows.append(base)
    rows.extend(dict(row, protocol_source="new Cone-only CV; same folds/weights/config except fit_max_points") for row in new_metrics)
    result = pd.DataFrame(rows)
    baseline = result[result["setting"] == "3000"].iloc[0]
    for column in ("global_rmse_mm", "global_p95_mm", "worst_v_bin_rmse_mm", "worst_v_bin_p95_mm", "fold_rmse_std_mm", "frame_rmse_std_mm"):
        result[f"delta_{column}_vs_3000"] = result[column] - float(baseline[column])
    return result


def classify_status(performance: pd.DataFrame, stability: pd.DataFrame) -> Tuple[str, str]:
    order = ["3000", "6000", "12000", "all_feasible"]
    base = performance[performance["setting"] == "3000"].iloc[0]
    all_row = performance[performance["setting"] == "all_feasible"].iloc[0]
    improvements = {
        "global_rmse": float(base["global_rmse_mm"] - all_row["global_rmse_mm"]),
        "global_p95": float(base["global_p95_mm"] - all_row["global_p95_mm"]),
        "worst_rmse": float(base["worst_v_bin_rmse_mm"] - all_row["worst_v_bin_rmse_mm"]),
        "worst_p95": float(base["worst_v_bin_p95_mm"] - all_row["worst_v_bin_p95_mm"]),
    }
    new_rows = performance[performance["setting"].isin(order[1:])].sort_values("fit_max_points")
    monotonic_rmse = bool(np.all(np.diff(new_rows["global_rmse_mm"].to_numpy(dtype=float)) <= 1.0e-12))
    monotonic_worst = bool(np.all(np.diff(new_rows["worst_v_bin_rmse_mm"].to_numpy(dtype=float)) <= 1.0e-12))
    all_stability = stability[(stability["row_type"] == "summary") & (stability["setting"] == "all_feasible")].iloc[0]
    unstable = (
        not bool(all_stability["fit_success"])
        or float(all_stability.get("axis_angle_vs_3000_max_deg", 0.0)) > 1.0
        or float(all_stability.get("apex_delta_vs_3000_max_mm", 0.0)) > 100.0
        or float(all_stability.get("half_angle_delta_vs_3000_max_deg", 0.0)) > 1.0
        or not monotonic_rmse
        or not monotonic_worst
    )
    if unstable:
        return "UNSTABLE", f"all-feasible 参数/性能路径不稳定：monotonic_rmse={monotonic_rmse}, monotonic_worst={monotonic_worst}, axis_change={all_stability.get('axis_angle_vs_3000_max_deg', np.nan):.3f} deg, apex_change={all_stability.get('apex_delta_vs_3000_max_mm', np.nan):.3f} mm"
    material = improvements["global_rmse"] >= 0.002 and improvements["global_p95"] >= 0.005 and improvements["worst_rmse"] >= 0.005
    if material:
        return "BENEFITS_FROM_MORE_POINTS", f"3000→all-feasible improvements: Global RMSE {improvements['global_rmse']:.5f} mm, Global P95 {improvements['global_p95']:.5f} mm, worst RMSE {improvements['worst_rmse']:.5f} mm"
    saturated = abs(improvements["global_rmse"]) < 0.001 and abs(improvements["global_p95"]) < 0.003 and abs(improvements["worst_rmse"]) < 0.005 and abs(improvements["worst_p95"]) < 0.01
    if saturated:
        return "SATURATED_AT_3000", f"3000→all-feasible changes below thresholds: RMSE 0.001, P95 0.003, worst RMSE 0.005, worst P95 0.01 mm"
    return "BENEFITS_FROM_MORE_POINTS", f"some 3000→all-feasible metrics improve materially: {improvements}"


def plot_performance(performance: pd.DataFrame, stability: pd.DataFrame, output: Path) -> None:
    order = ["3000", "6000", "12000", "all_feasible"]
    labels = ["3000", "6000", "12000", "all"]
    data = performance.set_index("setting").loc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("global_rmse_mm", "Global RMSE / mm"),
        ("global_p95_mm", "Global P95 / mm"),
        ("worst_v_bin_rmse_mm", "Worst-v-bin RMSE / mm"),
        ("worst_v_bin_p95_mm", "Worst-v-bin P95 / mm"),
        ("fold_rmse_std_mm", "Fold RMSE std / mm"),
        ("v_bias_range_mm", "v-bias range / mm"),
    ]
    for axis, (column, title) in zip(axes.flat, panels):
        axis.plot(x, data[column].to_numpy(dtype=float), marker="o", color="#2c7fb8", linewidth=2)
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    axes[0, 0].set_ylabel("Cone only")
    fig.suptitle("Full-36 Circular Cone sampling sensitivity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 5) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def write_audit(output: Path, config_path: Path, points_path: Path, cv_dir: Path, reused: Mapping[str, Any]) -> None:
    rows = [
        {
            "artifact": "laser_model_fit_config.daheng.yaml",
            "path": str(config_path),
            "scope": "formal model configuration",
            "dataset_ids": "Full-36 IDs supplied by grouped-CV runner/artifacts",
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced; no v-density weighting",
            "cv_protocol": "6-fold pose-grouped inherited",
            "model": "Cone configuration only",
            "fit_max_points": 3000,
            "validation_read": False,
            "action": "read only; not modified",
            "provenance_status": "CONFIRMED",
            "notes": "6000/12000/all are in-memory copies; formal YAML remains unchanged",
        },
        {
            "artifact": "full_fit_points.csv",
            "path": str(points_path),
            "scope": "Full-36 FIT point table",
            "dataset_ids": ",".join(FULL_IDS),
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced applied by Cone fit",
            "cv_protocol": "same 6-fold pose-grouped split",
            "model": "Cone input only",
            "fit_max_points": "6000/12000/all",
            "validation_read": False,
            "action": "reused; no image extraction",
            "provenance_status": "CONFIRMED",
            "notes": "32400 points; 900 points/pose",
        },
        {
            "artifact": "grouped_cv_model_comparison.csv + cv_pointwise_circular_cone.csv",
            "path": str(cv_dir),
            "scope": "Full-36 Cone fit_max_points=3000",
            "dataset_ids": ",".join(FULL_IDS),
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced; no v-density weighting",
            "cv_protocol": "6-fold pose-grouped; sorted frame-id round-robin",
            "model": "Circular Cone only",
            "fit_max_points": 3000,
            "validation_read": False,
            "action": "reused exactly; forbidden to rerun",
            "provenance_status": "CONFIRMED",
            "notes": f"pooled points={reused['pointwise_rows']}; existing baseline metrics and fold parameters reused",
        },
        {
            "artifact": "cv_fold_model_parameters.json",
            "path": str(cv_dir / "cv_fold_model_parameters.json"),
            "scope": "existing per-fold initialization reference",
            "dataset_ids": ",".join(FULL_IDS),
            "mask": "inherited",
            "weighting": "inherited",
            "cv_protocol": "same fold mapping",
            "model": "existing Cone parameters + fixed plane hint only",
            "fit_max_points": "3000 baseline / hint reuse",
            "validation_read": False,
            "action": "reused; no PlaneModel.fit",
            "provenance_status": "CONFIRMED",
            "notes": "No new Plane candidate or Plane fitting is run",
        },
        {
            "artifact": "Validation",
            "path": "excluded by task constraint",
            "scope": "none",
            "dataset_ids": "none",
            "mask": "not read",
            "weighting": "not read",
            "cv_protocol": "not read",
            "model": "not read",
            "fit_max_points": "N/A",
            "validation_read": False,
            "action": "excluded",
            "provenance_status": "N/A",
            "notes": "No Validation artifact opened",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def generate_report(output: Path, config_path: Path, cv_dir: Path, performance: pd.DataFrame, stability: pd.DataFrame, status: str, reason: str) -> None:
    base = performance[performance["setting"] == "3000"].iloc[0]
    all_row = performance[performance["setting"] == "all_feasible"].iloc[0]
    rows = stability[stability["row_type"] == "summary"].copy()
    lines = [
        "# Full-36 Circular Cone fit_max_points 敏感性",
        "",
        f"`CONE_SAMPLING_STATUS = {status}`",
        "",
        "## 结论摘要",
        "",
        f"- 3000 点基线直接复用 0817 artifact，未重跑；新增计算仅为 6000、12000、all feasible。",
        f"- all feasible 使用每个训练 fold 的全部 {int(all_row['train_point_count_per_fold'])} 个训练点；除 `fit_max_points` 外，Cone 配置、fold、frame-balanced weighting、初值策略保持不变。",
        f"- 3000→all feasible：Global RMSE {fmt(base['global_rmse_mm'])}→{fmt(all_row['global_rmse_mm'])} mm，Global P95 {fmt(base['global_p95_mm'])}→{fmt(all_row['global_p95_mm'])} mm，worst-v RMSE {fmt(base['worst_v_bin_rmse_mm'])}→{fmt(all_row['worst_v_bin_rmse_mm'])} mm。",
        f"- 判定：`{status}`。{reason}。",
        "",
        "## Artifact reuse audit",
        "",
        f"- 配置：`{config_path}`，只读；正式配置中的 `fit_max_points: 3000` 未修改。",
        f"- 3000 点结果：复用 `{cv_dir}` 下的 `grouped_cv_model_comparison.csv`、`per_v_bin_cv_metrics.csv`、`cv_pointwise_circular_cone.csv` 和 `cv_fold_model_parameters.json`。",
        "- 新档只调用 Circular Cone fit/evaluate；没有运行 Quadratic、Plane candidate 或 Validation。既有每折 plane 参数仅作为 Cone 所需的固定 orientation/root hint，未调用 PlaneModel.fit。",
        "- 训练数据为 Full-36 FIT、full_board_physical、inset=0 mm、frame-balanced weighting、6-fold pose-grouped CV；没有 v-density weighting。",
        "",
        "## Performance",
        "",
        "| sampling | fit_max_points | selected points/fold | Global RMSE | Global P95 | worst-v-bin | worst RMSE | worst P95 | fold RMSE std | v-bias range | status |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in performance.itertuples():
        lines.append(f"| {row.setting} | {row.fit_max_points} | {fmt(row.selected_fit_points_mean, 0)} | {fmt(row.global_rmse_mm)} | {fmt(row.global_p95_mm)} | {row.worst_v_bin} | {fmt(row.worst_v_bin_rmse_mm)} | {fmt(row.worst_v_bin_p95_mm)} | {fmt(row.fold_rmse_std_mm)} | {fmt(row.v_bias_range_mm)} | {row.sampling_status} |")

    lines += [
        "",
        "## Parameter changes and stability",
        "",
        "`axis_angle_vs_3000` uses the absolute axis dot product; apex change is Euclidean camera-coordinate distance.",
        "",
        "| sampling | axis range / deg | apex spread / mm | half-angle range / deg | max axis change vs 3000 / deg | max apex change / mm | max half-angle change / deg | cost CV / % | success rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples():
        lines.append(f"| {row.setting} | {fmt(row.axis_angle_range_deg, 3)} | {fmt(row.apex_spread_mm, 3)} | {fmt(row.half_angle_range_deg, 3)} | {fmt(row.axis_angle_vs_3000_max_deg, 3)} | {fmt(row.apex_delta_vs_3000_max_mm, 3)} | {fmt(row.half_angle_delta_vs_3000_max_deg, 3)} | {fmt(row.optimizer_cost_cv_pct, 2)} | {fmt(row.fit_success_rate, 2)} |")

    lines += [
        "",
        "## Decision rule",
        "",
        "- `SATURATED_AT_3000`：3000→all feasible 的 Global RMSE、Global P95、worst-v RMSE、worst-v P95 改变量分别小于 0.001、0.003、0.005、0.01 mm，且参数路径稳定。",
        "- `BENEFITS_FROM_MORE_POINTS`：上述前三项主要误差指标达到预设实质改善，且参数/性能路径稳定。",
        "- `UNSTABLE`：拟合失败、性能随采样数明显非单调，或出现大幅 axis/apex/half-angle 漂移。",
        "",
        "## Scope exclusions",
        "",
        "- 未重跑 3000 点；未运行 Quadratic/Plane；未读取 Validation；未训练 C1。",
        "- 未修改正式 YAML 配置；所有新采样档均为内存配置副本。",
        "",
        "## Outputs",
        "",
        "- `cone_sampling_sensitivity.csv`",
        "- `cone_sampling_performance.png`",
        "- `parameter_stability.csv`",
        "- `report.md`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--cv-dir", type=Path, default=DEFAULT_CV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    points_path = args.points.resolve()
    cv_dir = args.cv_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"输出目录非空；请显式使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = triplets.safe_yaml_load(config_path)
    extraction = cfg.get("extraction", {})
    if str(extraction.get("board_mask_mode", "")).lower() != triplets.FULL_BOARD_PHYSICAL:
        raise RuntimeError("配置不是 full_board_physical，停止")
    if abs(float(extraction.get("board_mask_inset_mm", 0.0))) > 1.0e-12:
        raise RuntimeError("full_board_physical inset 不是 0 mm，停止")
    configured_points = int(cfg.get("models", {}).get("cone", {}).get("fit_max_points", 0))
    if configured_points != 3000:
        raise RuntimeError(f"正式配置 fit_max_points 不是 3000：{configured_points}")

    points = normalize_points(points_path)
    reused, plane_hints, existing = load_reused_artifacts(cv_dir, points)
    fold_map = make_fold_assignment(FULL_IDS)
    new_metrics: List[Dict[str, Any]] = []
    new_details: Dict[str, Any] = {}
    new_params: Dict[str, Any] = {}
    for setting, limit in (("6000", 6000), ("12000", 12000), ("all_feasible", int(points.groupby("frame_id").size().sum() * (FOLD_COUNT - 1) / FOLD_COUNT))):
        metrics, details, params = fit_cone_setting(setting, limit, points, cfg, fold_map, plane_hints)
        new_metrics.append(metrics)
        new_details[setting] = details
        new_params[setting] = params

    performance = make_performance_table(reused, new_metrics)
    all_params = {"3000": reused["parameters"], **new_params}
    fit_counts = {
        "3000": {"fit_max_points": 3000, "selected_fit_points": 3000},
        "6000": {"fit_max_points": 6000, "selected_fit_points": int(new_metrics[0]["selected_fit_points_mean"])},
        "12000": {"fit_max_points": 12000, "selected_fit_points": int(new_metrics[1]["selected_fit_points_mean"])},
        "all_feasible": {"fit_max_points": int(new_metrics[2]["fit_max_points"]), "selected_fit_points": int(new_metrics[2]["selected_fit_points_mean"])},
    }
    stability = parameter_summary_rows(all_params, fit_counts)
    status, reason = classify_status(performance, stability)

    write_audit(output_dir / "artifact_reuse_audit.csv", config_path, points_path, cv_dir, reused)
    performance.to_csv(output_dir / "cone_sampling_sensitivity.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output_dir / "parameter_stability.csv", index=False, encoding="utf-8-sig")
    (output_dir / "cone_sampling_fold_parameters.json").write_text(json.dumps(new_params, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot_performance(performance, stability, output_dir / "cone_sampling_performance.png")
    generate_report(output_dir / "report.md", config_path, cv_dir, performance, stability, status, reason)

    print(f"CONE_SAMPLING_STATUS = {status}")
    print(performance[["setting", "global_rmse_mm", "global_p95_mm", "worst_v_bin_rmse_mm", "worst_v_bin_p95_mm", "fold_rmse_std_mm"]].to_string(index=False))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
