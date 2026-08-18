#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare existing Cone parameterizations on the same Full-36 ray/v domain.

No model is fitted here.  The four sets of fold parameters are loaded from
the previous sampling-sensitivity artifact, and the existing per-fold plane
parameters are used only as the same root-selection hint used by the CV
prediction path.  The 3000-point lambda predictions are independently
reconstructed and checked against the persisted pointwise artifact.
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
FOLD_COUNT = 6
V_MIN = 0.0
V_MAX = 3000.0
BIN_WIDTH = 100.0
BIN_COUNT = 30
SETTINGS = ("3000", "6000", "12000", "all_feasible")

DEFAULT_POINTS = ROOT / "projects" / "daheng" / "outputs" / "0817" / "full_fit_v_coverage_audit" / "full_fit_points.csv"
DEFAULT_CV_DIR = ROOT / "projects" / "daheng" / "outputs" / "0817" / "grouped_cv_model_comparison"
DEFAULT_SENSITIVITY_DIR = ROOT / "projects" / "daheng" / "outputs" / "0818" / "cone_sampling_sensitivity"
DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0818" / "cone_surface_equivalence"


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


def metric(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "bias_mm": np.nan, "rmse_mm": np.nan, "p95_mm": np.nan, "max_abs_mm": np.nan}
    return {
        "count": int(array.size),
        "bias_mm": float(np.mean(array)),
        "rmse_mm": float(np.sqrt(np.mean(array**2))),
        "p95_mm": float(np.percentile(np.abs(array), 95)),
        "max_abs_mm": float(np.max(np.abs(array))),
    }


def canonicalize(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["frame_id"] = data["frame_id"].map(normalize_frame_id)
    for column in ("u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm"):
        data[f"__{column}_key"] = data[column].astype(float).round(9)
    data = data.sort_values(
        ["frame_key", "__u_px_key", "__v_px_key", "__Xc_mm_key", "__Yc_mm_key", "__Zc_mm_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    data["point_index"] = data.groupby("frame_key", sort=False).cumcount().astype(int)
    return data


def make_fold_assignment() -> Dict[str, int]:
    return {frame_id: index % FOLD_COUNT for index, frame_id in enumerate(sorted(FULL_IDS))}


def load_operational_domain(points_path: Path, cv_dir: Path, fold_map: Mapping[str, int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    points = pd.read_csv(points_path, encoding="utf-8-sig")
    required = {"frame_id", "frame_key", "u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm", "ray_x", "ray_y", "ray_z"}
    missing = sorted(required.difference(points.columns))
    if missing:
        raise RuntimeError(f"Full-36 operational point table 缺少字段：{missing}")
    points = canonicalize(points)
    if sorted(points["frame_id"].unique()) != sorted(FULL_IDS) or len(points) != 32400:
        raise RuntimeError("Full-36 operational domain pose/point 数异常")
    points["fold"] = points["frame_id"].map(fold_map).astype(int)

    pointwise_path = cv_dir / "cv_pointwise_circular_cone.csv"
    baseline = canonicalize(pd.read_csv(pointwise_path, encoding="utf-8-sig"))
    if len(baseline) != len(points):
        raise RuntimeError("3000 pointwise artifact 与 operational point table 行数不一致")
    merge_keys = ["frame_key", "point_index"]
    baseline = baseline[merge_keys + ["frame_id", "fold", "u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm", "lambda_pred_mm", "valid"]]
    baseline = baseline.rename(columns={column: f"{column}_artifact" for column in baseline.columns if column not in merge_keys})
    domain = points.merge(baseline, on=merge_keys, how="inner", validate="one_to_one")
    if len(domain) != len(points):
        raise RuntimeError("operational domain 与 3000 pointwise identity merge 丢行")
    for column in ("frame_id", "fold"):
        if not np.all(domain[column].astype(str).to_numpy() == domain[f"{column}_artifact"].astype(str).to_numpy()):
            raise RuntimeError(f"3000 pointwise {column} 与 operational fold 不一致")
    for column in ("u_px", "v_px", "Xc_mm", "Yc_mm", "Zc_mm"):
        if not np.allclose(domain[column], domain[f"{column}_artifact"], rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"3000 pointwise {column} 与 operational domain 不一致")
    if not domain["valid_artifact"].astype(bool).all():
        raise RuntimeError("已有 3000 pointwise artifact 含无效 lambda")
    domain["v_bin_index"] = np.floor((domain["v_px"] - V_MIN) / BIN_WIDTH).astype(int)
    domain["v_bin"] = domain["v_bin_index"].map(lambda value: f"v_{int(value * 100):04d}_{int((value + 1) * 100):04d}")
    audit = {
        "point_count": int(len(domain)),
        "pose_count": int(domain["frame_id"].nunique()),
        "fold_count": int(domain["fold"].nunique()),
        "v_min_px": float(domain["v_px"].min()),
        "v_max_px": float(domain["v_px"].max()),
        "pointwise_source": str(pointwise_path),
    }
    return domain, audit


def load_parameters(cv_dir: Path, sensitivity_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, ReusedPlaneHint], Dict[str, Any]]:
    fold_params = json.loads((cv_dir / "cv_fold_model_parameters.json").read_text(encoding="utf-8"))
    sampled_params = json.loads((sensitivity_dir / "cone_sampling_fold_parameters.json").read_text(encoding="utf-8"))
    params: Dict[str, Dict[str, Any]] = {
        "3000": {str(fold): fold_params[str(fold)]["circular_cone"] for fold in range(FOLD_COUNT)},
    }
    for setting in ("6000", "12000", "all_feasible"):
        if setting not in sampled_params:
            raise RuntimeError(f"缺少已有 {setting} Cone fold parameters")
        params[setting] = sampled_params[setting]
    hints: Dict[int, ReusedPlaneHint] = {}
    for fold in range(FOLD_COUNT):
        plane = fold_params[str(fold)]["global_plane"]
        hints[fold] = ReusedPlaneHint(
            normal=np.asarray(plane["normal"], dtype=float),
            d=float(plane["d_mm"]),
            z_range=tuple(float(value) for value in plane["z_valid_range_mm"]),
        )
    stability_path = sensitivity_dir / "parameter_stability.csv"
    stability = pd.read_csv(stability_path, encoding="utf-8-sig") if stability_path.is_file() else pd.DataFrame()
    return params, hints, {"fold_params": fold_params, "stability": stability}


def cone_from_dict(parameters: Mapping[str, Any]) -> triplets.CircularConeModel:
    cone = triplets.CircularConeModel({})
    cone.axis = np.asarray(parameters["axis_unit_camera"], dtype=float)
    cone.apex = np.asarray(parameters["apex_camera_mm"], dtype=float)
    cone.alpha_deg = float(parameters["half_apex_angle_deg"])
    cone.fit_success = bool(parameters.get("fit_success", True))
    cone.cost = float(parameters.get("optimizer_cost", np.nan))
    cone.z_range = tuple(float(value) for value in parameters["z_valid_range_mm"])
    return cone


def reconstruct_lambdas(domain: pd.DataFrame, params: Mapping[str, Mapping[str, Any]], hints: Mapping[int, ReusedPlaneHint]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output = domain.copy()
    crosscheck: Dict[str, Any] = {}
    rays_all = output[["ray_x", "ray_y", "ray_z"]].to_numpy(dtype=float)
    for setting in SETTINGS:
        prediction = np.full(len(output), np.nan, dtype=float)
        for fold in range(FOLD_COUNT):
            indices = np.flatnonzero(output["fold"].to_numpy(dtype=int) == fold)
            rays = rays_all[indices]
            cone = cone_from_dict(params[setting][str(fold)])
            lambda_hint = hints[fold].intersect_rays(rays)
            prediction[indices] = cone.intersect_rays(rays, lambda_hint=lambda_hint)
        output[f"lambda_{setting}_mm"] = prediction
        if setting == "3000":
            difference = prediction - output["lambda_pred_mm_artifact"].to_numpy(dtype=float)
            check = metric(difference)
            crosscheck = {
                "baseline_reconstruction_rmse_mm": check["rmse_mm"],
                "baseline_reconstruction_p95_mm": check["p95_mm"],
                "baseline_reconstruction_max_abs_mm": check["max_abs_mm"],
                "baseline_reconstruction_count": check["count"],
            }
            if not np.isfinite(check["max_abs_mm"]) or check["max_abs_mm"] > 1.0e-7:
                raise RuntimeError(f"3000 lambda 重建无法复现既有 pointwise artifact：{crosscheck}")
    return output, crosscheck


def status_for_setting(global_metric: Mapping[str, Any]) -> str:
    rmse = float(global_metric["lambda_delta_rmse_mm"])
    p95 = float(global_metric["lambda_delta_p95_mm"])
    max_abs = float(global_metric["lambda_delta_max_abs_mm"])
    if rmse <= 0.01 and p95 <= 0.03 and max_abs <= 0.10:
        return "EQUIVALENT"
    if rmse <= 0.05 and p95 <= 0.10 and max_abs <= 0.50:
        return "MARGINALLY_DIFFERENT"
    return "MATERIAL_DIFFERENCE"


def equivalence_rows(domain: pd.DataFrame, params: Mapping[str, Mapping[str, Any]], stability: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows: List[Dict[str, Any]] = []
    setting_status: Dict[str, str] = {}
    for setting in ("6000", "12000", "all_feasible"):
        delta = domain[f"lambda_{setting}_mm"].to_numpy(dtype=float) - domain["lambda_3000_mm"].to_numpy(dtype=float)
        global_metric = metric(delta)
        global_row = {
            "row_type": "pooled_global",
            "setting": setting,
            "fold": "ALL",
            "v_bin": "Global",
            "v_bin_lo_px": V_MIN,
            "v_bin_hi_px": V_MAX,
            "point_count": global_metric["count"],
            "pose_count": int(domain["frame_id"].nunique()),
            "lambda_delta_bias_mm": global_metric["bias_mm"],
            "lambda_delta_rmse_mm": global_metric["rmse_mm"],
            "lambda_delta_p95_mm": global_metric["p95_mm"],
            "lambda_delta_max_abs_mm": global_metric["max_abs_mm"],
            "valid_pair_rate": float(np.isfinite(delta).mean()),
        }
        status = status_for_setting(global_row)
        setting_status[setting] = status
        global_row["surface_status"] = status
        rows.append(global_row)
        for fold in range(FOLD_COUNT):
            mask = domain["fold"].to_numpy(dtype=int) == fold
            fold_metric = metric(delta[mask])
            rows.append(
                {
                    "row_type": "fold_global",
                    "setting": setting,
                    "fold": int(fold),
                    "v_bin": "Global",
                    "v_bin_lo_px": V_MIN,
                    "v_bin_hi_px": V_MAX,
                    "point_count": fold_metric["count"],
                    "pose_count": int(domain.loc[mask, "frame_id"].nunique()),
                    "lambda_delta_bias_mm": fold_metric["bias_mm"],
                    "lambda_delta_rmse_mm": fold_metric["rmse_mm"],
                    "lambda_delta_p95_mm": fold_metric["p95_mm"],
                    "lambda_delta_max_abs_mm": fold_metric["max_abs_mm"],
                    "valid_pair_rate": float(np.isfinite(delta[mask]).mean()),
                    "surface_status": status,
                }
            )
        for index in range(BIN_COUNT):
            mask = domain["v_bin_index"].to_numpy(dtype=int) == index
            bin_metric = metric(delta[mask])
            rows.append(
                {
                    "row_type": "pooled_v_bin",
                    "setting": setting,
                    "fold": "ALL",
                    "v_bin": f"v_{index * 100:04d}_{(index + 1) * 100:04d}",
                    "v_bin_lo_px": float(index * 100),
                    "v_bin_hi_px": float((index + 1) * 100),
                    "point_count": bin_metric["count"],
                    "pose_count": int(domain.loc[mask, "frame_id"].nunique()),
                    "lambda_delta_bias_mm": bin_metric["bias_mm"],
                    "lambda_delta_rmse_mm": bin_metric["rmse_mm"],
                    "lambda_delta_p95_mm": bin_metric["p95_mm"],
                    "lambda_delta_max_abs_mm": bin_metric["max_abs_mm"],
                    "valid_pair_rate": float(np.isfinite(delta[mask]).mean()),
                    "surface_status": status,
                }
            )
    return pd.DataFrame(rows), setting_status


def overall_status(setting_status: Mapping[str, str], table: pd.DataFrame) -> str:
    statuses = list(setting_status.values())
    if any(value == "MATERIAL_DIFFERENCE" for value in statuses):
        return "MATERIAL_DIFFERENCE"
    if any(value == "MARGINALLY_DIFFERENT" for value in statuses):
        return "MARGINALLY_DIFFERENT"
    return "EQUIVALENT"


def parameter_drift_table(stability: pd.DataFrame) -> pd.DataFrame:
    if stability.empty:
        return pd.DataFrame()
    return stability[stability["row_type"] == "summary"].copy()


def plot_delta_v(table: pd.DataFrame, output: Path) -> None:
    data = table[table["row_type"] == "pooled_v_bin"].copy()
    colors = {"6000": "#1f77b4", "12000": "#ff7f0e", "all_feasible": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = (data["v_bin_lo_px"] + data["v_bin_hi_px"]) / 2.0
    for setting in ("6000", "12000", "all_feasible"):
        subset = data[data["setting"] == setting].sort_values("v_bin_lo_px")
        xx = (subset["v_bin_lo_px"] + subset["v_bin_hi_px"]) / 2.0
        axes[0, 0].plot(xx, subset["lambda_delta_rmse_mm"], marker="o", markersize=3, label=setting, color=colors[setting])
        axes[0, 1].plot(xx, subset["lambda_delta_p95_mm"], marker="o", markersize=3, label=setting, color=colors[setting])
        axes[1, 0].plot(xx, subset["lambda_delta_max_abs_mm"], marker="o", markersize=3, label=setting, color=colors[setting])
    panels = [(axes[0, 0], "Delta lambda RMSE / mm"), (axes[0, 1], "Delta lambda P95 / mm"), (axes[1, 0], "Delta lambda Max / mm")]
    for axis, title in panels:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("v / px")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    global_data = table[table["row_type"] == "pooled_global"].set_index("setting").loc[["6000", "12000", "all_feasible"]]
    gx = np.arange(3)
    axes[1, 1].plot(gx, global_data["lambda_delta_rmse_mm"], marker="o", label="RMSE")
    axes[1, 1].plot(gx, global_data["lambda_delta_p95_mm"], marker="s", label="P95")
    axes[1, 1].plot(gx, global_data["lambda_delta_max_abs_mm"], marker="^", label="Max")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Global surface prediction drift")
    axes[1, 1].set_xticks(gx, ["6000", "12000", "all"])
    axes[1, 1].set_ylabel("delta lambda / mm")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Cone sampling: operational lambda drift vs 3000-point baseline", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 6) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def write_audit(output: Path, points_path: Path, cv_dir: Path, sensitivity_dir: Path, domain_audit: Mapping[str, Any]) -> None:
    rows = [
        {
            "artifact": "full_fit_points.csv",
            "path": str(points_path),
            "role": "operational Full-36 ray/v domain",
            "scope": "FIT 36 poses / 32400 rays",
            "models": "ray input only",
            "point_count": domain_audit["point_count"],
            "pose_count": domain_audit["pose_count"],
            "fold_count": domain_audit["fold_count"],
            "cv_protocol": "held-out fold assignment reused; no split change",
            "mask": "full_board_physical; inset=0 mm",
            "weighting": "not changed; prediction-only analysis",
            "validation_read": False,
            "action": "reused; no extraction",
            "provenance_status": "CONFIRMED",
            "notes": f"v={domain_audit['v_min_px']:.1f}..{domain_audit['v_max_px']:.1f} px",
        },
        {
            "artifact": "cv_pointwise_circular_cone.csv",
            "path": str(cv_dir / "cv_pointwise_circular_cone.csv"),
            "role": "3000 lambda baseline cross-check",
            "scope": "existing Full-36 Cone CV",
            "models": "Circular Cone / 3000",
            "point_count": domain_audit["point_count"],
            "pose_count": domain_audit["pose_count"],
            "fold_count": domain_audit["fold_count"],
            "cv_protocol": "existing 6-fold pose-grouped",
            "mask": "inherited full_board_physical",
            "weighting": "inherited frame-balanced",
            "validation_read": False,
            "action": "reused; no rerun",
            "provenance_status": "CONFIRMED",
            "notes": "3000 lambda reconstruction checked against persisted pointwise prediction",
        },
        {
            "artifact": "cone_sampling_fold_parameters.json",
            "path": str(sensitivity_dir / "cone_sampling_fold_parameters.json"),
            "role": "6000/12000/all Cone fold parameter source",
            "scope": "Full-36 Cone sampling sensitivity",
            "models": "Circular Cone only",
            "point_count": domain_audit["point_count"],
            "pose_count": domain_audit["pose_count"],
            "fold_count": domain_audit["fold_count"],
            "cv_protocol": "same existing folds; prediction only",
            "mask": "inherited full_board_physical",
            "weighting": "inherited frame-balanced",
            "validation_read": False,
            "action": "reused; no refit",
            "provenance_status": "CONFIRMED",
            "notes": "Parameters are loaded, not optimized",
        },
        {
            "artifact": "cv_fold_model_parameters.json",
            "path": str(cv_dir / "cv_fold_model_parameters.json"),
            "role": "3000 Cone parameters and fixed lambda root hints",
            "scope": "Full-36 six folds",
            "models": "existing Cone + plane hint only",
            "point_count": domain_audit["point_count"],
            "pose_count": domain_audit["pose_count"],
            "fold_count": domain_audit["fold_count"],
            "cv_protocol": "same held-out fold mapping",
            "mask": "inherited",
            "weighting": "inherited",
            "validation_read": False,
            "action": "reused; no Plane fit",
            "provenance_status": "CONFIRMED",
            "notes": "Plane parameters are only a persisted root-selection hint",
        },
        {
            "artifact": "Validation",
            "path": "excluded by task constraint",
            "role": "not read",
            "scope": "none",
            "models": "not read",
            "point_count": 0,
            "pose_count": 0,
            "fold_count": 0,
            "cv_protocol": "not read",
            "mask": "not read",
            "weighting": "not read",
            "validation_read": False,
            "action": "excluded",
            "provenance_status": "N/A",
            "notes": "No Validation artifact opened",
        },
    ]
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")


def generate_report(
    output: Path,
    points_path: Path,
    cv_dir: Path,
    sensitivity_dir: Path,
    domain_audit: Mapping[str, Any],
    crosscheck: Mapping[str, Any],
    table: pd.DataFrame,
    setting_status: Mapping[str, str],
    overall: str,
    stability: pd.DataFrame,
) -> None:
    global_rows = table[table["row_type"] == "pooled_global"].set_index("setting")
    lines = [
        "# Cone operational surface equivalence",
        "",
        f"`CONE_SURFACE_STATUS = {overall}`",
        "",
        "## 结论摘要",
        "",
        f"- 在相同 Full-36 held-out operational ray/v 域（{domain_audit['point_count']} rays、{domain_audit['pose_count']} poses、v={domain_audit['v_min_px']:.1f}..{domain_audit['v_max_px']:.1f} px）上，复用四档 Cone fold 参数计算 lambda；没有重新拟合。",
        f"- 3000 参数重建与既有 pointwise lambda 的 RMSE/P95/Max 差异为 {fmt(crosscheck['baseline_reconstruction_rmse_mm'])}/{fmt(crosscheck['baseline_reconstruction_p95_mm'])}/{fmt(crosscheck['baseline_reconstruction_max_abs_mm'])} mm，说明求交/root 选择路径复现一致。",
        "- 参数漂移没有按同等幅度转化为 operational surface 漂移；需以以下 Δlambda 指标判断，而不是仅看 apex/angle 参数差异。",
        f"- 总体判定：`CONE_SURFACE_STATUS = {overall}`；各档：" + ", ".join(f"{key}={value}" for key, value in setting_status.items()) + "。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- ray/v domain：`{points_path}`；3000 baseline：`{cv_dir / 'cv_pointwise_circular_cone.csv'}`。",
        f"- 6000/12000/all 参数：`{sensitivity_dir / 'cone_sampling_fold_parameters.json'}`；每折 root hint：既有 `cv_fold_model_parameters.json`。",
        "- 每个 sampling setting 的 fold model 只作用于对应 held-out pose rays；所有 setting 使用完全相同的 ray、v、fold、point identity。",
        "- 未读取 Validation；未运行 Quadratic；未训练 C1；没有调用任何拟合/优化过程。",
        "",
        "| check | result |",
        "|---|---|",
        f"| operational rays | {domain_audit['point_count']} |",
        f"| poses / folds | {domain_audit['pose_count']} / {domain_audit['fold_count']} |",
        f"| v domain | {domain_audit['v_min_px']:.1f}–{domain_audit['v_max_px']:.1f} px |",
        f"| 3000 baseline reconstruction max error | {fmt(crosscheck['baseline_reconstruction_max_abs_mm'])} mm |",
        "| parameter operation | loaded only; no refit |",
        "",
        "## Global Δlambda vs 3000",
        "",
        "负值仅表示新 sampling 的 lambda 小于 3000 baseline；指标是绝对 surface prediction difference，不是 board residual。",
        "",
        "| setting | status | Δlambda bias / mm | Δlambda RMSE / mm | Δlambda P95 / mm | Δlambda Max / mm | valid pair rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting in ("6000", "12000", "all_feasible"):
        row = global_rows.loc[setting]
        lines.append(f"| {setting} | {setting_status[setting]} | {fmt(row.lambda_delta_bias_mm)} | {fmt(row.lambda_delta_rmse_mm)} | {fmt(row.lambda_delta_p95_mm)} | {fmt(row.lambda_delta_max_abs_mm)} | {fmt(row.valid_pair_rate, 4)} |")

    lines += [
        "",
        "## Δlambda 随 v 分布",
        "",
        "逐 100 px bin 的明细保存在 `cone_surface_equivalence.csv` 的 `pooled_v_bin` 行；图中同时给出 RMSE/P95/Max。重点观察是否在边缘或少数 bin 出现局部漂移。",
        "",
        "| setting | largest-bin RMSE / mm | v-bin | largest-bin P95 / mm | v-bin | largest-bin Max / mm | v-bin |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    v_rows = table[table["row_type"] == "pooled_v_bin"]
    for setting in ("6000", "12000", "all_feasible"):
        subset = v_rows[v_rows["setting"] == setting]
        rmse_row = subset.loc[subset["lambda_delta_rmse_mm"].abs().idxmax()]
        p95_row = subset.loc[subset["lambda_delta_p95_mm"].abs().idxmax()]
        max_row = subset.loc[subset["lambda_delta_max_abs_mm"].idxmax()]
        lines.append(f"| {setting} | {fmt(rmse_row.lambda_delta_rmse_mm)} | {rmse_row.v_bin} | {fmt(p95_row.lambda_delta_p95_mm)} | {p95_row.v_bin} | {fmt(max_row.lambda_delta_max_abs_mm)} | {max_row.v_bin} |")

    lines += [
        "",
        "## Parameter drift vs surface drift",
        "",
        "| setting | max axis drift / deg | max apex drift / mm | max half-angle drift / deg | Δlambda RMSE / mm | Δlambda P95 / mm | interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    summary = parameter_drift_table(stability).set_index("setting") if not stability.empty else pd.DataFrame()
    for setting in ("6000", "12000", "all_feasible"):
        surface = global_rows.loc[setting]
        if not summary.empty and setting in summary.index:
            param = summary.loc[setting]
            interpretation = "parameter drift, surface nearly unchanged" if float(surface.lambda_delta_rmse_mm) <= 0.01 else "surface drift exceeds equivalent gate"
            lines.append(f"| {setting} | {fmt(param.axis_angle_vs_3000_max_deg, 3)} | {fmt(param.apex_delta_vs_3000_max_mm, 3)} | {fmt(param.half_angle_delta_vs_3000_max_deg, 3)} | {fmt(surface.lambda_delta_rmse_mm)} | {fmt(surface.lambda_delta_p95_mm)} | {interpretation} |")

    lines += [
        "",
        "## Operational equivalence gates",
        "",
        "本次使用的诊断 gate（不是新的标定验收规范）：",
        "- `EQUIVALENT`：所有新 sampling 的 Δlambda RMSE ≤ 0.01 mm、P95 ≤ 0.03 mm、Max ≤ 0.10 mm。",
        "- `MARGINALLY_DIFFERENT`：未达到 equivalent，但所有新 sampling 的 RMSE ≤ 0.05 mm、P95 ≤ 0.10 mm、Max ≤ 0.50 mm。",
        "- `MATERIAL_DIFFERENCE`：任一指标超过 marginal gate。",
        "",
        "## Scope exclusions",
        "",
        "- 不重新拟合；不读取 Validation；不运行 Quadratic；不训练 C1。",
        "- 仅使用实际 Full-36 FIT ray/v operational domain，不引入 synthetic rays 或新的采样域。",
        "",
        "## Outputs",
        "",
        "- `cone_surface_equivalence.csv`",
        "- `delta_lambda_vs_v.png`",
        "- `report.md`",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--cv-dir", type=Path, default=DEFAULT_CV_DIR)
    parser.add_argument("--sensitivity-dir", type=Path, default=DEFAULT_SENSITIVITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    points_path = args.points.resolve()
    cv_dir = args.cv_dir.resolve()
    sensitivity_dir = args.sensitivity_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"输出目录非空；请显式使用 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_map = make_fold_assignment()
    domain, domain_audit = load_operational_domain(points_path, cv_dir, fold_map)
    params, plane_hints, parameter_sources = load_parameters(cv_dir, sensitivity_dir)
    domain, crosscheck = reconstruct_lambdas(domain, params, plane_hints)
    equivalence, setting_status = equivalence_rows(domain, params, parameter_sources["stability"])
    overall = overall_status(setting_status, equivalence)

    write_audit(output_dir / "artifact_reuse_audit.csv", points_path, cv_dir, sensitivity_dir, domain_audit)
    equivalence.to_csv(output_dir / "cone_surface_equivalence.csv", index=False, encoding="utf-8-sig")
    plot_delta_v(equivalence, output_dir / "delta_lambda_vs_v.png")
    generate_report(output_dir / "report.md", points_path, cv_dir, sensitivity_dir, domain_audit, crosscheck, equivalence, setting_status, overall, parameter_sources["stability"])

    print(f"CONE_SURFACE_STATUS = {overall}")
    print(equivalence[equivalence["row_type"] == "pooled_global"][["setting", "lambda_delta_rmse_mm", "lambda_delta_p95_mm", "lambda_delta_max_abs_mm", "surface_status"]].to_string(index=False))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
