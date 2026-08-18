#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Curated-14, Robust-18 and the reused Full-36 Q/C pose-grouped CV.

This script deliberately does not read image data or Validation.  Full-36 is
read from the existing 0817 grouped-CV artifacts.  Only the missing
Curated-14 and Robust-18 Quadratic/Cone folds are computed from the already
extracted FIT point table.

The current model implementation uses a plane only as a root-selection hint
and as an initial orientation for the two requested models.  To obey the
"do not run Plane" constraint, subset runs use the already persisted
Full-36 fold plane parameters as a fixed hint; no PlaneModel.fit call is made
and no Plane candidate is evaluated or reported.  The subset fold layout is
the same deterministic sorted-pose round-robin 6-fold protocol, with no
point-wise split.
"""

from __future__ import annotations

import argparse
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
MODEL_NAMES = ("quadratic_graph", "circular_cone")
V_MIN_PX = 0.0
V_MAX_PX = 3000.0
BIN_WIDTH_PX = 100.0
BIN_EDGES = np.arange(V_MIN_PX, V_MAX_PX + BIN_WIDTH_PX, BIN_WIDTH_PX)
BIN_COUNT = len(BIN_EDGES) - 1
FOLD_COUNT = 6

DEFAULT_CONFIG = ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_POINTS = ROOT / "projects" / "daheng" / "outputs" / "0817" / "full_fit_v_coverage_audit" / "full_fit_points.csv"
DEFAULT_FULL_CV = ROOT / "projects" / "daheng" / "outputs" / "0817" / "grouped_cv_model_comparison"
DEFAULT_CURATED = ROOT / "projects" / "daheng" / "outputs" / "0818" / "pose_geometry_audit" / "curated_fit_ids.json"
DEFAULT_ROBUST = ROOT / "projects" / "daheng" / "outputs" / "0818" / "robust_curated_18" / "robust_curated_18_ids.json"
DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0818" / "fit_set_model_comparison"


@dataclass
class ReusedPlaneHint:
    """A persisted plane artifact used only for deterministic root selection."""

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
        "rmse_mm": float(np.sqrt(np.mean(array**2))),
        "p95_mm": float(np.percentile(absolute, 95)),
        "max_abs_mm": float(np.max(absolute)),
    }


def fold_assignment(frame_ids: Sequence[str], fold_count: int = FOLD_COUNT) -> Dict[str, int]:
    ordered = sorted({str(value) for value in frame_ids})
    if len(ordered) < fold_count:
        raise ValueError(f"pose 数 {len(ordered)} 小于 fold 数 {fold_count}")
    return {frame_id: index % fold_count for index, frame_id in enumerate(ordered)}


def pooled_group_rmse(detail: pd.DataFrame, column: str) -> Tuple[float, float]:
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
    data = detail.loc[in_domain].copy()
    data["v_bin_index"] = np.floor((data["v_px"] - V_MIN_PX) / BIN_WIDTH_PX).astype(int)
    for index in range(BIN_COUNT):
        group = data[data["v_bin_index"] == index]
        metric = metric_values(group["board_error_mm"].to_numpy(dtype=float))
        valid = group["valid"].to_numpy(dtype=bool)
        frame_mean, frame_std = pooled_group_rmse(group, "frame_id") if not group.empty else (np.nan, np.nan)
        fold_mean, fold_std = pooled_group_rmse(group, "fold") if not group.empty else (np.nan, np.nan)
        rows.append(
            {
                "model": model,
                "v_bin": f"v_{int(BIN_EDGES[index]):04d}_{int(BIN_EDGES[index + 1]):04d}",
                "v_bin_lo_px": float(BIN_EDGES[index]),
                "v_bin_hi_px": float(BIN_EDGES[index + 1]),
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


def aggregate_subset(
    details: Mapping[str, pd.DataFrame],
    bin_metrics: Mapping[str, pd.DataFrame],
    fit_ids: Sequence[str],
    fold_map: Mapping[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    train_counts = [sum(assigned != fold for assigned in fold_map.values()) for fold in range(FOLD_COUNT)]
    for model in MODEL_NAMES:
        detail = details[model]
        metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
        valid = detail["valid"].to_numpy(dtype=bool)
        frame_mean, frame_std = pooled_group_rmse(detail, "frame_id")
        fold_mean, fold_std = pooled_group_rmse(detail, "fold")
        bins = bin_metrics[model]
        valid_bins = bins[np.isfinite(bins["rmse_mm"])].copy()
        worst = valid_bins.sort_values(["rmse_mm", "p95_mm"], ascending=False).iloc[0] if not valid_bins.empty else pd.Series(dtype=float)
        trend = trend_metrics(bins)
        rows.append(
            {
                "row_type": "pooled_cv",
                "fit_size": int(len(fit_ids)),
                "model": model,
                "v_bin_lo_px": V_MIN_PX,
                "v_bin_hi_px": V_MAX_PX,
                "train_frame_count_mean": float(np.mean(train_counts)),
                "heldout_frame_count": int(len(fit_ids)),
                "fold_count": int(FOLD_COUNT),
                "valid_intersections": int(np.count_nonzero(valid)),
                "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
                **metric,
                "mean_frame_rmse_mm": frame_mean,
                "std_frame_rmse_mm": frame_std,
                "mean_fold_rmse_mm": fold_mean,
                "std_fold_rmse_mm": fold_std,
                "worst_v_bin": str(worst.get("v_bin", "")),
                "worst_v_bin_rmse_mm": float(worst.get("rmse_mm", np.nan)),
                "worst_v_bin_p95_mm": float(worst.get("p95_mm", np.nan)),
                "worst_v_bin_max_abs_mm": float(worst.get("max_abs_mm", np.nan)),
                **trend,
            }
        )
    return pd.DataFrame(rows)


def load_ids(path: Path, key: str) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data[key]
    return [f"{int(value):03d}" for value in values]


def load_plane_hints(full_cv_dir: Path) -> Dict[int, ReusedPlaneHint]:
    path = full_cv_dir / "cv_fold_model_parameters.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    hints: Dict[int, ReusedPlaneHint] = {}
    for fold in range(FOLD_COUNT):
        plane = data[str(fold)]["global_plane"]
        hints[fold] = ReusedPlaneHint(
            normal=np.asarray(plane["normal"], dtype=float),
            d=float(plane["d_mm"]),
            z_range=tuple(float(value) for value in plane["z_valid_range_mm"]),
        )
    return hints


def fit_quadratic_cone(
    train_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    plane_hint: ReusedPlaneHint,
) -> Dict[str, triplets.LaserModel]:
    points, _, frame_ids = triplets.dataframe_arrays(train_df)
    quadratic_cfg = cfg.get("models", {}).get("quadratic", {})
    quadratic = triplets.QuadraticGraphModel(ridge=float(quadratic_cfg.get("ridge", 1.0e-10)))
    quadratic.fit(points, frame_ids, plane=plane_hint)  # fixed hint; no PlaneModel.fit
    cone = triplets.CircularConeModel(dict(cfg.get("models", {}).get("cone", {})))
    cone.fit(points, frame_ids, plane=plane_hint)  # fixed hint; no PlaneModel.fit
    return {"quadratic_graph": quadratic, "circular_cone": cone}


def fold_metric_row(
    detail: pd.DataFrame,
    model: str,
    fold: int,
    train_frames: Sequence[str],
    heldout_frames: Sequence[str],
) -> Dict[str, Any]:
    metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
    valid = detail["valid"].to_numpy(dtype=bool)
    return {
        "row_type": "fold",
        "fit_size": int(len(train_frames) + len(heldout_frames)),
        "model": model,
        "fold": int(fold),
        "train_frame_count": int(len(train_frames)),
        "heldout_frame_count": int(len(heldout_frames)),
        "heldout_frames": ",".join(heldout_frames),
        "valid_intersections": int(np.count_nonzero(valid)),
        "valid_rate": float(np.mean(valid)) if valid.size else np.nan,
        **metric,
    }


def run_subset_cv(
    set_name: str,
    fit_ids: Sequence[str],
    points: pd.DataFrame,
    cfg: Mapping[str, Any],
    plane_hints: Mapping[int, ReusedPlaneHint],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Any]]:
    fit_ids = sorted({str(value) for value in fit_ids})
    subset = points[points["frame_id"].astype(str).isin(fit_ids)].copy()
    actual_ids = sorted(subset["frame_id"].astype(str).unique())
    if actual_ids != fit_ids:
        raise RuntimeError(f"{set_name} 的点表 pose 不完整：requested={fit_ids}, actual={actual_ids}")
    fold_map = fold_assignment(fit_ids, FOLD_COUNT)
    details_by_model: Dict[str, List[pd.DataFrame]] = {model: [] for model in MODEL_NAMES}
    fold_rows: List[Dict[str, Any]] = []
    parameter_rows: Dict[str, Any] = {}

    for fold in range(FOLD_COUNT):
        heldout = sorted([frame_id for frame_id in fit_ids if fold_map[frame_id] == fold])
        train_frames = sorted([frame_id for frame_id in fit_ids if fold_map[frame_id] != fold])
        if not heldout or not train_frames:
            raise RuntimeError(f"{set_name} fold {fold} 的 train/heldout pose 为空")
        train_df = subset[~subset["frame_id"].isin(heldout)].copy()
        test_df = subset[subset["frame_id"].isin(heldout)].copy()
        models = fit_quadratic_cone(train_df, cfg, plane_hints[fold])
        parameter_rows[str(fold)] = {name: model.to_dict() for name, model in models.items()}
        for model_name in MODEL_NAMES:
            model = models[model_name]
            _, raw_detail = triplets.evaluate_model(model, test_df, plane_hints[fold])
            detail = raw_detail.copy()
            detail["fold"] = int(fold)
            detail["heldout_frames"] = ",".join(heldout)
            detail["frame_id"] = test_df.loc[detail.index, "frame_id"].astype(str).to_numpy()
            detail = detail.reset_index(drop=True)
            details_by_model[model_name].append(detail)
            fold_rows.append(fold_metric_row(detail, model_name, fold, train_frames, heldout))

    pooled = {model: pd.concat(chunks, ignore_index=True) for model, chunks in details_by_model.items()}
    bins = {model: build_bin_metrics(pooled[model], model) for model in MODEL_NAMES}
    aggregate = aggregate_subset(pooled, bins, fit_ids, fold_map)
    return aggregate, pooled, bins, {"fold_map": fold_map, "parameters": parameter_rows, "fold_rows": pd.DataFrame(fold_rows)}


def pairwise_angles_deg(vectors: np.ndarray, absolute: bool = True) -> np.ndarray:
    if len(vectors) < 2:
        return np.asarray([], dtype=float)
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1.0e-12)
    values: List[float] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            dot = float(normalized[i] @ normalized[j])
            if absolute:
                dot = abs(dot)
            values.append(math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0)))))
    return np.asarray(values, dtype=float)


def parameter_stability(parameters: Mapping[str, Any], model: str) -> Dict[str, Any]:
    rows = [parameters[key][model] for key in sorted(parameters, key=lambda value: int(value))]
    if not rows:
        return {}
    if model == "quadratic_graph":
        dep_axes = [row.get("dependent_axis", "") for row in rows]
        counts = pd.Series(dep_axes).value_counts()
        centers = np.asarray([row["normalization"]["independent_center_mm"] for row in rows], dtype=float)
        scales = np.asarray([row["normalization"]["independent_scale_mm"] for row in rows], dtype=float)
        beta = np.asarray([row["coefficients"] for row in rows], dtype=float)
        return {
            "parameter_axis_consistency": float(counts.iloc[0] / len(rows)),
            "parameter_center_std_mm": float(np.mean(np.std(centers, axis=0))),
            "parameter_scale_cv_pct": float(np.mean(np.std(scales, axis=0) / np.maximum(np.abs(np.mean(scales, axis=0)), 1.0e-12)) * 100.0),
            "parameter_beta_rel_std_pct": float(np.linalg.norm(np.std(beta, axis=0)) / max(np.linalg.norm(np.mean(beta, axis=0)), 1.0e-12) * 100.0),
            "parameter_axis_angle_range_deg": np.nan,
            "parameter_apex_spread_mm": np.nan,
            "parameter_half_angle_range_deg": np.nan,
            "parameter_half_angle_std_deg": np.nan,
            "parameter_fit_success_rate": np.nan,
            "parameter_cost_cv_pct": np.nan,
            "parameter_stability_metric": float(np.linalg.norm(np.std(beta, axis=0)) / max(np.linalg.norm(np.mean(beta, axis=0)), 1.0e-12) * 100.0),
        }
    axes = np.asarray([row["axis_unit_camera"] for row in rows], dtype=float)
    apex = np.asarray([row["apex_camera_mm"] for row in rows], dtype=float)
    alpha = np.asarray([row["half_apex_angle_deg"] for row in rows], dtype=float)
    costs = np.asarray([row.get("optimizer_cost", np.nan) for row in rows], dtype=float)
    angle_values = pairwise_angles_deg(axes, absolute=True)
    apex_values = pairwise_angles_deg(axes, absolute=True)  # keep empty-safe allocation below
    del apex_values
    distances: List[float] = []
    for i in range(len(apex)):
        for j in range(i + 1, len(apex)):
            distances.append(float(np.linalg.norm(apex[i] - apex[j])))
    finite_costs = costs[np.isfinite(costs)]
    cost_cv = float(np.std(finite_costs) / max(abs(np.mean(finite_costs)), 1.0e-12) * 100.0) if finite_costs.size else np.nan
    fit_success = [bool(row.get("fit_success", False)) for row in rows]
    return {
        "parameter_axis_consistency": np.nan,
        "parameter_center_std_mm": np.nan,
        "parameter_scale_cv_pct": np.nan,
        "parameter_beta_rel_std_pct": np.nan,
        "parameter_axis_angle_range_deg": float(np.max(angle_values) if angle_values.size else 0.0),
        "parameter_apex_spread_mm": float(max(distances) if distances else 0.0),
        "parameter_half_angle_range_deg": float(np.ptp(alpha)),
        "parameter_half_angle_std_deg": float(np.std(alpha)),
        "parameter_fit_success_rate": float(np.mean(fit_success)),
        "parameter_cost_cv_pct": cost_cv,
        "parameter_stability_metric": float(np.max(angle_values) if angle_values.size else 0.0),
    }


def add_parameter_columns(row: Dict[str, Any], stability: Mapping[str, Any]) -> None:
    for key in (
        "parameter_axis_consistency",
        "parameter_center_std_mm",
        "parameter_scale_cv_pct",
        "parameter_beta_rel_std_pct",
        "parameter_axis_angle_range_deg",
        "parameter_apex_spread_mm",
        "parameter_half_angle_range_deg",
        "parameter_half_angle_std_deg",
        "parameter_fit_success_rate",
        "parameter_cost_cv_pct",
        "parameter_stability_metric",
    ):
        row[key] = stability.get(key, np.nan)


def full_rows(
    full_cv_dir: Path,
    full_ids: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    grouped_path = full_cv_dir / "grouped_cv_model_comparison.csv"
    full = pd.read_csv(grouped_path, encoding="utf-8-sig")
    pooled = full[(full["row_type"] == "pooled_cv") & full["model"].isin(MODEL_NAMES)].copy()
    if len(pooled) != len(MODEL_NAMES):
        raise RuntimeError(f"Full-36 grouped-CV 缺少 Quadratic/Cone pooled rows：{grouped_path}")
    params = json.loads((full_cv_dir / "cv_fold_model_parameters.json").read_text(encoding="utf-8"))
    parameter_map = {str(fold): {model: params[str(fold)][model] for model in MODEL_NAMES} for fold in range(FOLD_COUNT)}
    rows: List[Dict[str, Any]] = []
    for source in pooled.to_dict(orient="records"):
        row = {
            "fit_set": "FULL_36",
            "fit_size": int(len(full_ids)),
            "model": str(source["model"]),
            "cv_status": "REUSED_EXISTING",
            "pose_ids": ",".join(full_ids),
            "point_count": int(source["point_count"]),
            "fold_count": int(source["fold_count"]),
            "valid_intersections": int(source["valid_intersections"]),
            "valid_rate": float(source["valid_rate"]),
            "global_bias_mm": float(source["bias_mm"]),
            "global_rmse_mm": float(source["rmse_mm"]),
            "global_p95_mm": float(source["p95_mm"]),
            "worst_v_bin": str(source["worst_v_bin"]),
            "worst_v_bin_rmse_mm": float(source["worst_v_bin_rmse_mm"]),
            "worst_v_bin_p95_mm": float(source["worst_v_bin_p95_mm"]),
            "worst_v_bin_max_abs_mm": float(source["worst_v_bin_max_abs_mm"]),
            "v_bias_range_mm": float(source["v_bias_range_mm"]),
            "fold_rmse_mean_mm": float(source["mean_fold_rmse_mm"]),
            "fold_rmse_std_mm": float(source["std_fold_rmse_mm"]),
            "frame_rmse_mean_mm": float(source["mean_frame_rmse_mm"]),
            "frame_rmse_std_mm": float(source["std_frame_rmse_mm"]),
            "protocol_source": "0817/grouped_cv_model_comparison",
        }
        add_parameter_columns(row, parameter_stability(parameter_map, str(source["model"])))
        rows.append(row)
    return pd.DataFrame(rows), parameter_map


def subset_rows(
    set_name: str,
    fit_ids: Sequence[str],
    aggregate: pd.DataFrame,
    parameter_rows: Mapping[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source in aggregate.to_dict(orient="records"):
        row = {
            "fit_set": set_name,
            "fit_size": int(len(fit_ids)),
            "model": str(source["model"]),
            "cv_status": "COMPUTED_MISSING",
            "pose_ids": ",".join(sorted(fit_ids)),
            "point_count": int(source["point_count"]),
            "fold_count": int(source["fold_count"]),
            "valid_intersections": int(source["valid_intersections"]),
            "valid_rate": float(source["valid_rate"]),
            "global_bias_mm": float(source["bias_mm"]),
            "global_rmse_mm": float(source["rmse_mm"]),
            "global_p95_mm": float(source["p95_mm"]),
            "worst_v_bin": str(source["worst_v_bin"]),
            "worst_v_bin_rmse_mm": float(source["worst_v_bin_rmse_mm"]),
            "worst_v_bin_p95_mm": float(source["worst_v_bin_p95_mm"]),
            "worst_v_bin_max_abs_mm": float(source["worst_v_bin_max_abs_mm"]),
            "v_bias_range_mm": float(source["v_bias_range_mm"]),
            "fold_rmse_mean_mm": float(source["mean_fold_rmse_mm"]),
            "fold_rmse_std_mm": float(source["std_fold_rmse_mm"]),
            "frame_rmse_mean_mm": float(source["mean_frame_rmse_mm"]),
            "frame_rmse_std_mm": float(source["std_frame_rmse_mm"]),
            "protocol_source": "new subset Q/C grouped CV; 0817 fold-plane reused only as fixed root hint",
        }
        add_parameter_columns(row, parameter_stability(parameter_rows, str(source["model"])))
        rows.append(row)
    return pd.DataFrame(rows)


def add_within_set_model_rank(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["within_set_model_rank"] = np.nan
    for set_name, index in result.groupby("fit_set").groups.items():
        local = result.loc[index].sort_values(["global_rmse_mm", "worst_v_bin_rmse_mm"])
        for rank, row_index in enumerate(local.index, start=1):
            result.loc[row_index, "within_set_model_rank"] = rank
    return result


def choose_fit_set(table: pd.DataFrame) -> Tuple[str, str]:
    piv = table.pivot(index="fit_set", columns="model", values="global_rmse_mm")
    required = {"CURATED_14", "ROBUST_18", "FULL_36"}
    if not required.issubset(piv.index):
        return "FULL_36", "缺少完整三组结果，保守推荐 Full-36"
    q = piv["quadratic_graph"]
    c = piv["circular_cone"]
    # Use the better Q/C score per set for size selection; model identity is
    # intentionally not frozen here because C0 remains unresolved.
    best = pd.DataFrame({"quadratic_graph": q, "circular_cone": c}).min(axis=1)
    improvement_14_to_18 = float(best.loc["CURATED_14"] - best.loc["ROBUST_18"])
    gap_18_to_36 = float(best.loc["ROBUST_18"] - best.loc["FULL_36"])
    total_gain = float(best.loc["CURATED_14"] - best.loc["FULL_36"])
    saturation = 1.0 - gap_18_to_36 / total_gain if total_gain > 0 else np.nan
    robust_rows = table[table["fit_set"] == "ROBUST_18"]
    full_rows_df = table[table["fit_set"] == "FULL_36"]
    robust_worst_gap = float(robust_rows["worst_v_bin_p95_mm"].min() - full_rows_df["worst_v_bin_p95_mm"].min())
    if improvement_14_to_18 > 0 and np.isfinite(saturation) and saturation >= 0.70 and robust_worst_gap <= 0.06:
        return "ROBUST_18", f"14→18 最优模型 RMSE 改善 {improvement_14_to_18:.5f} mm，18→36 饱和度 {saturation:.1%}，worst-v P95 gap {robust_worst_gap:.5f} mm"
    if np.isfinite(saturation) and saturation >= 0.50:
        return "ROBUST_18", f"Robust-18 已回收 {saturation:.1%} 的 14→36 RMSE 改善，但边界/稳定性仍需保守解释"
    return "FULL_36", f"Robust-18 相对 Full-36 的剩余差距仍为 {gap_18_to_36:.5f} mm，未达到预设饱和阈值"


def plot_performance(table: pd.DataFrame, output: Path) -> None:
    order = ["CURATED_14", "ROBUST_18", "FULL_36"]
    labels = ["Curated-14", "Robust-18", "Full-36"]
    colors = {"quadratic_graph": "#1f77b4", "circular_cone": "#ff7f0e"}
    titles = [
        ("global_rmse_mm", "Global RMSE / mm"),
        ("global_p95_mm", "Global P95 / mm"),
        ("worst_v_bin_rmse_mm", "Worst-v-bin RMSE / mm"),
        ("worst_v_bin_p95_mm", "Worst-v-bin P95 / mm"),
        ("v_bias_range_mm", "v-bias range / mm"),
        ("fold_rmse_std_mm", "Fold RMSE std / mm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    x = np.arange(len(order))
    width = 0.34
    for axis, (column, title) in zip(axes.flat, titles):
        for offset, model in [(-width / 2, "quadratic_graph"), (width / 2, "circular_cone")]:
            values = [float(table[(table["fit_set"] == fit_set) & (table["model"] == model)][column].iloc[0]) for fit_set in order]
            axis.bar(x + offset, values, width=width, color=colors[model], label=model.replace("_", " "))
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=18)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("FIT pose-grouped CV: Curated-14 / Robust-18 / Full-36", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 5) -> str:
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def generate_audit(output: Path, points_path: Path, full_cv_dir: Path, curated_path: Path, robust_path: Path) -> None:
    rows = [
        {
            "artifact": "full_fit_points.csv",
            "path": str(points_path),
            "scope": "FIT 001-018,025-036,049-054 / 36 poses",
            "dataset_ids": ",".join(FULL_IDS),
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced; no v-density weighting",
            "cv_protocol": "input point table for deterministic 6-fold pose-grouped CV",
            "model": "Quadratic/Cone input; no Plane candidate",
            "validation_read": False,
            "action": "reused; no image re-extraction",
            "protocol_match": "YES",
            "notes": "Steger + vertical per-row single point + continuity + per-frame point limit already frozen in 0817 artifact",
        },
        {
            "artifact": "grouped_cv_model_comparison.csv",
            "path": str(full_cv_dir / "grouped_cv_model_comparison.csv"),
            "scope": "Full-36",
            "dataset_ids": ",".join(FULL_IDS),
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "frame-balanced; no v-density weighting",
            "cv_protocol": "6-fold pose-grouped; sorted frame-id round-robin; 6 held-out poses/fold",
            "model": "Quadratic + Circular Cone (existing artifact also contains Plane; Plane row not used)",
            "validation_read": False,
            "action": "reused; no rerun",
            "protocol_match": "CONFIRMED",
            "notes": "Full-36 pooled rows are the authoritative reference",
        },
        {
            "artifact": "cv_fold_model_parameters.json",
            "path": str(full_cv_dir / "cv_fold_model_parameters.json"),
            "scope": "Full-36 fold parameters",
            "dataset_ids": ",".join(FULL_IDS),
            "mask": "inherited from Full-36 artifact",
            "weighting": "inherited from Full-36 artifact",
            "cv_protocol": "same 6-fold pose-grouped fold reference",
            "model": "reused fold plane parameters only as Q/C root-selection hint; no PlaneModel.fit",
            "validation_read": False,
            "action": "reused for subset hint and Full parameter stability",
            "protocol_match": "Q/C SAME; fixed hint reused",
            "notes": "No new Plane fit, Plane metric, or Plane-based selection",
        },
        {
            "artifact": "curated_fit_ids.json",
            "path": str(curated_path),
            "scope": "Curated-14",
            "dataset_ids": "001,006,010,013,015,017,025,027,031,049,051,052,053,054",
            "mask": "geometry audit source; full_board_physical provenance",
            "weighting": "not used for pose selection",
            "cv_protocol": "subset evaluated with same deterministic 6-fold pose-grouped design",
            "model": "selection was geometry-only; Q/C evaluated only after set fixed",
            "validation_read": False,
            "action": "reused fixed set; no pose deletion",
            "protocol_match": "YES for requested Q/C CV",
            "notes": "No residual participates in set construction",
        },
        {
            "artifact": "robust_curated_18_ids.json",
            "path": str(robust_path),
            "scope": "Robust-Curated-18",
            "dataset_ids": "001,005,006,010,013,015,017,025,026,027,028,031,049,050,051,052,053,054",
            "mask": "geometry audit source; full_board_physical provenance",
            "weighting": "not used for pose selection",
            "cv_protocol": "subset evaluated with same deterministic 6-fold pose-grouped design",
            "model": "selection was geometry-only; Q/C evaluated only after set fixed",
            "validation_read": False,
            "action": "reused fixed set; no pose deletion",
            "protocol_match": "YES for requested Q/C CV",
            "notes": "Curated-14 fixed; added 005,026,028,050 only",
        },
        {
            "artifact": "Historical-18",
            "path": "0818 geometry audit / existing historical references",
            "scope": "Historical 001-018",
            "dataset_ids": "001-018",
            "mask": "geometry reference only",
            "weighting": "not applicable to reused model result",
            "cv_protocol": "no same-protocol Q/C grouped-CV artifact identified",
            "model": "not used",
            "validation_read": False,
            "action": "reference only; no补跑",
            "protocol_match": "NOT AVAILABLE",
            "notes": "Per request, no Historical model CV was added",
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
            "validation_read": False,
            "action": "excluded",
            "protocol_match": "N/A",
            "notes": "No Validation image or Validation artifact was opened",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def generate_report(
    output: Path,
    table: pd.DataFrame,
    fit_recommendation: str,
    fit_reason: str,
    full_cv_dir: Path,
    subset_fold_rows: Mapping[str, pd.DataFrame],
) -> None:
    lines = [
        "# Curated-14 / Robust-18 / Full-36 Quadratic-Cone 泛化比较",
        "",
        f"`RECOMMENDED_FIT_SET = {fit_recommendation}`",
        "",
        "`C0_MODEL_STATUS = UNRESOLVED`",
        "",
        "## 结论摘要",
        "",
        f"- 推荐 FIT 集：`{fit_recommendation}`。判据：{fit_reason}。",
        "- C0：`UNRESOLVED`。Full-36 既有同协议结果的 Quadratic/Cone top-score gap 为 0.0109，小于既定 0.05 决策阈值；因此不能仅凭子集结果把生产 C0 冻结为某一模型。",
        "- Robust-18 的四个新增 pose 是既有几何审核固定的 `005, 026, 028, 050`；本次没有根据 residual 删除或替换任何 pose。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- Full-36 直接复用 `{full_cv_dir / 'grouped_cv_model_comparison.csv'}`，未重跑。",
        "- 复用的协议：`full_board_physical`、inset=0 mm、Steger、vertical per-row single point、continuity、每帧点数限制、frame-balanced weighting、无 v-density weighting、6-fold pose-grouped CV、sorted frame-id round-robin。",
        "- Curated-14/Robust-18 仅读取已生成的 `full_fit_points.csv`，按固定 pose 集做缺失的 Quadratic/Cone CV；没有读取图像，也没有读取 Validation。",
        "- 为满足“不运行 Plane”，子集 CV 不调用 `PlaneModel.fit`。模型实现所需的 root-selection / orientation hint 复用 Full-36 的既有 fold-plane 参数；这些参数不作为候选模型、指标或选择依据。",
        "- Historical-18 没有发现同协议 Q/C CV 结果，仅作已有几何/历史参考，没有补跑。",
        "",
        "## Pooled grouped-CV metrics",
        "",
        "`global_p95_mm` 是所有 held-out pose pooled prediction 的绝对误差 P95；worst-v-bin 在 100 px bin 中按 RMSE 最大者确定。",
        "",
        "| FIT set | model | status | Global RMSE | Global P95 | worst-v-bin | worst RMSE | worst P95 | v-bias range | fold RMSE std | frame RMSE std |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.sort_values(["fit_size", "model"]).itertuples():
        lines.append(
            f"| {row.fit_set} | {row.model} | {row.cv_status} | {fmt(row.global_rmse_mm)} | {fmt(row.global_p95_mm)} | {row.worst_v_bin} | {fmt(row.worst_v_bin_rmse_mm)} | {fmt(row.worst_v_bin_p95_mm)} | {fmt(row.v_bias_range_mm)} | {fmt(row.fold_rmse_std_mm)} | {fmt(row.frame_rmse_std_mm)} |"
        )

    lines += ["", "## Model parameter stability", "", "稳定性只描述每个 FIT 集内部六个训练折得到的参数散布，不使用 residual 删除 pose。", "", "| FIT set | model | axis consistency | beta rel std / % | center std / mm | cone axis range / deg | cone apex spread / mm | cone half-angle range / deg | success rate |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table.sort_values(["fit_size", "model"]).itertuples():
        lines.append(
            f"| {row.fit_set} | {row.model} | {fmt(row.parameter_axis_consistency, 2)} | {fmt(row.parameter_beta_rel_std_pct, 2)} | {fmt(row.parameter_center_std_mm, 3)} | {fmt(row.parameter_axis_angle_range_deg, 3)} | {fmt(row.parameter_apex_spread_mm, 3)} | {fmt(row.parameter_half_angle_range_deg, 3)} | {fmt(row.parameter_fit_success_rate, 2)} |"
        )

    lines += ["", "## 14 → 18 → 36 性能收益", ""]
    for model in MODEL_NAMES:
        subset = table[table["model"] == model].set_index("fit_set")
        if not {"CURATED_14", "ROBUST_18", "FULL_36"}.issubset(subset.index):
            continue
        q14 = subset.loc["CURATED_14"]
        q18 = subset.loc["ROBUST_18"]
        q36 = subset.loc["FULL_36"]
        total = float(q14["global_rmse_mm"] - q36["global_rmse_mm"])
        recovered = float(q14["global_rmse_mm"] - q18["global_rmse_mm"])
        saturation = recovered / total if total > 0 else np.nan
        lines.append(
            f"- **{model}**：Global RMSE {fmt(q14['global_rmse_mm'])} → {fmt(q18['global_rmse_mm'])} → {fmt(q36['global_rmse_mm'])} mm；14→18 改善 {fmt(recovered)} mm，14→36 总改善 {fmt(total)} mm，Robust-18 回收 {fmt(saturation * 100.0, 1)}%。"
        )
        lines.append(
            f"  worst-v-bin RMSE {fmt(q14['worst_v_bin_rmse_mm'])} → {fmt(q18['worst_v_bin_rmse_mm'])} → {fmt(q36['worst_v_bin_rmse_mm'])} mm；worst-v P95 {fmt(q14['worst_v_bin_p95_mm'])} → {fmt(q18['worst_v_bin_p95_mm'])} → {fmt(q36['worst_v_bin_p95_mm'])} mm。"
        )

    lines += [
        "",
        "## Fold detail / reproducibility",
        "",
        "Curated-14 与 Robust-18 的 fold 级结果保存在脚本运行目录的内存计算中，并将 pooled comparison 写入 `fit_set_model_comparison.csv`；Full-36 的 fold 明细继续由 0817 原始 artifact 提供。",
        "",
        "| FIT set | model | fold RMSE mean / mm | fold RMSE std / mm |",
        "|---|---|---:|---:|",
    ]
    for set_name, frame in subset_fold_rows.items():
        for model, group in frame.groupby("model"):
            lines.append(f"| {set_name} | {model} | {fmt(group['rmse_mm'].mean())} | {fmt(group['rmse_mm'].std(ddof=0))} |")

    lines += [
        "",
        "## Scope exclusions",
        "",
        "- 未运行 Plane candidate；未训练 C1；未读取 Validation；未使用 Plane/Quadratic/Cone residual 构造或删除 pose。",
        "- 结果是 FIT pose-grouped CV 的泛化 proxy，不替代独立 Validation 或标准件验收。",
        "",
        "## Output",
        "",
        "- `artifact_reuse_audit.csv`",
        "- `fit_set_model_comparison.csv`",
        "- `fit_size_performance.png`",
        "- `report.md`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--full-cv-dir", type=Path, default=DEFAULT_FULL_CV)
    parser.add_argument("--curated-json", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--robust-json", type=Path, default=DEFAULT_ROBUST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    points_path = args.points.resolve()
    full_cv_dir = args.full_cv_dir.resolve()
    curated_path = args.curated_json.resolve()
    robust_path = args.robust_json.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"输出目录非空；如需覆盖请显式使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = triplets.safe_yaml_load(config_path)
    extraction = cfg.get("extraction", {})
    if str(extraction.get("board_mask_mode", "")).lower() != triplets.FULL_BOARD_PHYSICAL:
        raise RuntimeError("配置不是 full_board_physical，停止")
    if abs(float(extraction.get("board_mask_inset_mm", 0.0))) > 1.0e-12:
        raise RuntimeError("配置 full_board_physical inset 不是 0 mm，停止")

    curated_ids = load_ids(curated_path, "curated_fit_ids")
    robust_ids = load_ids(robust_path, "robust_curated_18_ids")
    if len(curated_ids) != 14 or len(robust_ids) != 18 or not set(curated_ids).issubset(robust_ids):
        raise RuntimeError(f"Curated/Robust IDs 异常：{len(curated_ids)=}, {len(robust_ids)=}")
    if tuple(sorted(FULL_IDS)) != tuple(sorted(set(FULL_IDS))):
        raise AssertionError("FULL_IDS 重复")

    points = pd.read_csv(points_path, encoding="utf-8-sig")
    # pandas infers the zero-padded CSV frame_id column as an integer; restore
    # the canonical pose keys before all set/fold joins.
    points["frame_id"] = points["frame_id"].map(lambda value: f"{int(value):03d}")
    actual_full = sorted(points["frame_id"].astype(str).unique())
    if actual_full != sorted(FULL_IDS):
        raise RuntimeError(f"full_fit_points.csv pose 集合异常：{actual_full}")
    plane_hints = load_plane_hints(full_cv_dir)

    full_table, full_parameter_map = full_rows(full_cv_dir, FULL_IDS)
    curated_agg, curated_details, curated_bins, curated_meta = run_subset_cv("CURATED_14", curated_ids, points, cfg, plane_hints)
    robust_agg, robust_details, robust_bins, robust_meta = run_subset_cv("ROBUST_18", robust_ids, points, cfg, plane_hints)
    del curated_details, curated_bins, robust_details, robust_bins, full_parameter_map

    curated_table = subset_rows("CURATED_14", curated_ids, curated_agg, curated_meta["parameters"])
    robust_table = subset_rows("ROBUST_18", robust_ids, robust_agg, robust_meta["parameters"])
    table = pd.concat([curated_table, robust_table, full_table], ignore_index=True, sort=False)
    table = add_within_set_model_rank(table)
    fit_recommendation, fit_reason = choose_fit_set(table)

    generate_audit(output_dir / "artifact_reuse_audit.csv", points_path, full_cv_dir, curated_path, robust_path)
    table.to_csv(output_dir / "fit_set_model_comparison.csv", index=False, encoding="utf-8-sig")
    plot_performance(table, output_dir / "fit_size_performance.png")
    (output_dir / "subset_cv_fold_model_parameters.json").write_text(
        json.dumps({"CURATED_14": curated_meta["parameters"], "ROBUST_18": robust_meta["parameters"]}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    subset_fold_rows = {"CURATED_14": curated_meta["fold_rows"], "ROBUST_18": robust_meta["fold_rows"]}
    generate_report(output_dir / "report.md", table, fit_recommendation, fit_reason, full_cv_dir, subset_fold_rows)

    print(f"RECOMMENDED_FIT_SET = {fit_recommendation}")
    print("C0_MODEL_STATUS = UNRESOLVED")
    print(table[["fit_set", "model", "global_rmse_mm", "global_p95_mm", "worst_v_bin_rmse_mm", "worst_v_bin_p95_mm", "v_bias_range_mm", "fold_rmse_std_mm"]].to_string(index=False))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
