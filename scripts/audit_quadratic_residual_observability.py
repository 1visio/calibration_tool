#!/usr/bin/env python3
"""FIT-only observability and tail audit for the frozen Full-36 Quadratic C0.

The script reuses the existing Full-36 calibration point, ray, and PnP truth
artifact. It loads the frozen Quadratic model and evaluates intersections only.
It does not fit C0/C1, read Validation, re-extract laser points, or delete data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
PROJECT = ROOT / "projects" / "daheng"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fit_laser_models_from_triplets as triplets  # noqa: E402


FIT_IDS = [
    *[f"{i:03d}" for i in range(1, 19)],
    *[f"{i:03d}" for i in range(25, 37)],
    *[f"{i:03d}" for i in range(49, 55)],
]
FULL_POINTS = PROJECT / "outputs/0817/full_fit_v_coverage_audit/full_fit_points.csv"
FULL_METADATA = PROJECT / "outputs/0817/grouped_cv_model_comparison/candidate_models/full_fit/model_parameters.json"
FROZEN_MODEL = PROJECT / "outputs/0818/c0_freeze/quadratic_graph.yaml"
INTRINSICS = PROJECT / "outputs/0811/intrinsics/calibration_result.yaml"
FORMAL_CONFIG = ROOT / "configs/laser_model_fit_config.daheng.yaml"
FRAME_GEOMETRY = PROJECT / "outputs/0814/board_coordinate_audit/frame_board_geometry_summary.csv"
FIT_ROOT = PROJECT / "data/laser_plane/fit"
FIT_ROOT_0817 = PROJECT / "data/laser_plane_0817/fit"
DEFAULT_OUTPUT = PROJECT / "outputs/0818/quadratic_residual_observability"

V_BIN_WIDTH = 100.0
PCA_BIN_COUNT = 12
PNP_UNCERTAINTY_LOW_MM = 0.025
PNP_UNCERTAINTY_HIGH_MM = 0.033


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML root is not a mapping: {path}")
    return value


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def pca_rays(xy: np.ndarray) -> dict[str, Any]:
    center = np.mean(xy, axis=0)
    centered = xy - center
    covariance = np.cov(centered, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    s_axis = np.asarray(eigenvectors[:, order[0]], dtype=float)
    if (abs(s_axis[1]) >= abs(s_axis[0]) and s_axis[1] < 0) or (
        abs(s_axis[1]) < abs(s_axis[0]) and s_axis[0] < 0
    ):
        s_axis = -s_axis
    t_axis = np.asarray([-s_axis[1], s_axis[0]], dtype=float)
    coordinates = centered @ np.column_stack([s_axis, t_axis])
    explained = eigenvalues / np.sum(eigenvalues)
    return {
        "center_xn": float(center[0]),
        "center_yn": float(center[1]),
        "axis_s_xn": float(s_axis[0]),
        "axis_s_yn": float(s_axis[1]),
        "axis_t_xn": float(t_axis[0]),
        "axis_t_yn": float(t_axis[1]),
        "eigenvalue_s": float(eigenvalues[0]),
        "eigenvalue_t": float(eigenvalues[1]),
        "explained_s": float(explained[0]),
        "explained_t": float(explained[1]),
        "s_robust_span": float(np.percentile(coordinates[:, 0], 95) - np.percentile(coordinates[:, 0], 5)),
        "t_robust_span": float(np.percentile(coordinates[:, 1], 95) - np.percentile(coordinates[:, 1], 5)),
        "anisotropy_sqrt_eigenvalue_ratio": float(math.sqrt(eigenvalues[0] / eigenvalues[1])),
        "s": coordinates[:, 0],
        "t": coordinates[:, 1],
    }


def robust_span(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, 95) - np.percentile(values, 5)) if len(values) else math.nan


def trend_stats(x: np.ndarray, y: np.ndarray, bins: int = PCA_BIN_COUNT) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    result: dict[str, Any] = {
        "n_points": int(len(x)),
        "x_min": math.nan,
        "x_max": math.nan,
        "x_span": math.nan,
        "y_mean_mm": math.nan,
        "y_std_mm": math.nan,
        "y_p05_mm": math.nan,
        "y_p95_mm": math.nan,
        "slope_mm_per_x": math.nan,
        "slope_mm_per_x_span": math.nan,
        "pearson_r": math.nan,
        "linear_r2": math.nan,
        "binned_explained_fraction": math.nan,
        "low_frequency_amplitude_mm": math.nan,
        "low_frequency_bin_mean_std_mm": math.nan,
        "low_frequency_bin_count": 0,
    }
    if not len(x):
        return result
    span = float(np.ptp(x))
    mean = float(np.mean(y))
    total_ss = float(np.sum((y - mean) ** 2))
    result.update(
        {
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "x_span": span,
            "y_mean_mm": mean,
            "y_std_mm": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "y_p05_mm": float(np.percentile(y, 5)),
            "y_p95_mm": float(np.percentile(y, 95)),
        }
    )
    if len(x) >= 3 and span > 0:
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ beta
        result["slope_mm_per_x"] = float(beta[1])
        result["slope_mm_per_x_span"] = float(beta[1] * span)
        result["pearson_r"] = float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) else math.nan
        result["linear_r2"] = float(1.0 - np.sum((y - prediction) ** 2) / total_ss) if total_ss else math.nan
    if len(y) >= 3 and total_ss > 0:
        order = np.argsort(x, kind="mergesort")
        chunks = [chunk for chunk in np.array_split(order, min(bins, len(order))) if len(chunk)]
        means = np.asarray([np.mean(y[chunk]) for chunk in chunks])
        within_ss = float(sum(np.sum((y[chunk] - np.mean(y[chunk])) ** 2) for chunk in chunks))
        result.update(
            {
                "binned_explained_fraction": float(1.0 - within_ss / total_ss),
                "low_frequency_amplitude_mm": float(np.ptp(means)),
                "low_frequency_bin_mean_std_mm": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                "low_frequency_bin_count": int(len(means)),
            }
        )
    return result


def load_models() -> tuple[Any, Any, dict[str, Any]]:
    metadata = json.loads(FULL_METADATA.read_text(encoding="utf-8"))
    expected = {
        "source": "FIT 001-018, 025-036, 049-054",
        "frame_count": 36,
        "point_count": 32400,
        "mask_mode": "full_board_physical",
        "mask_inset_mm": 0.0,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if isinstance(expected_value, float):
            if abs(float(actual) - expected_value) > 1.0e-12:
                raise RuntimeError(f"Full-36 metadata mismatch: {key}={actual!r}")
        elif actual != expected_value:
            raise RuntimeError(f"Full-36 metadata mismatch: {key}={actual!r}")
    model_data = load_yaml(FROZEN_MODEL)
    if model_data.get("model_type") != "quadratic_graph":
        raise RuntimeError("Frozen model_type is not quadratic_graph")
    plane_data = metadata["models"]["global_plane"]
    plane = triplets.PlaneModel()
    plane.normal = np.asarray(plane_data["normal"], dtype=float)
    plane.d = float(plane_data["d_mm"])
    plane.z_range = tuple(float(v) for v in plane_data["z_valid_range_mm"])
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    model = triplets.QuadraticGraphModel()
    model.dep_axis = axis_index[str(model_data["dependent_axis"]).upper()]
    model.ind_axes = tuple(axis_index[str(v).upper()] for v in model_data["independent_axes"])
    model.center = np.asarray(model_data["normalization"]["independent_center_mm"], dtype=float)
    model.scale = np.asarray(model_data["normalization"]["independent_scale_mm"], dtype=float)
    model.beta = np.asarray(model_data["coefficients"], dtype=float)
    model.z_range = tuple(float(v) for v in model_data["z_valid_range_mm"])
    return model, plane, model_data


def load_pose_sources(k: np.ndarray, d: np.ndarray, cfg: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    saved = pd.read_csv(FRAME_GEOMETRY, dtype={"frame_id": str})
    saved = saved[saved["row_type"].astype(str) == "frame"]
    poses: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for _, row in saved.iterrows():
        frame_id = str(row["frame_id"]).zfill(3)
        if frame_id in FIT_IDS:
            poses[frame_id] = {
                "rvec": np.asarray([row["rvec_x"], row["rvec_y"], row["rvec_z"]], dtype=float),
                "tvec": np.asarray([row["board_origin_x_mm"], row["board_origin_y_mm"], row["board_origin_z_mm"]], dtype=float),
                "pnp_rmse_px": float(row["pnp_rmse_px"]),
                "source": "reused_frame_board_geometry_summary",
            }
    board = cfg["board"]
    for frame_id in FIT_IDS:
        if frame_id in poses:
            continue
        image_candidates = [
            FIT_ROOT / f"chess {frame_id}.tif",
            FIT_ROOT_0817 / f"chess {frame_id}.tif",
        ]
        image_path = next((path for path in image_candidates if path.is_file()), image_candidates[0])
        image = triplets.imread_unicode(image_path)
        pose = triplets.detect_board_pose(
            image,
            k,
            d,
            cols=int(board["pattern_cols"]),
            rows=int(board["pattern_rows"]),
            square_size_mm=float(board["square_size_mm"]),
            max_rmse_px=float(board.get("max_pnp_rmse_px", 0.4)),
        )
        poses[frame_id] = {
            "rvec": np.asarray(pose.rvec, dtype=float),
            "tvec": np.asarray(pose.tvec, dtype=float),
            "pnp_rmse_px": float(pose.reprojection_rmse_px),
            "source": "supplemental_chessboard_pnp_for_boundary_only",
        }
    for frame_id in FIT_IDS:
        audit.append(
            {
                "frame_id": frame_id,
                "pnp_source": poses[frame_id]["source"],
                "pnp_rmse_px": poses[frame_id]["pnp_rmse_px"],
                "laser_points_reextracted": False,
            }
        )
    return poses, audit


def add_board_geometry(frame: pd.DataFrame, poses: Mapping[str, Mapping[str, Any]], cfg: Mapping[str, Any]) -> None:
    board = cfg["board"]
    extraction = cfg["extraction"]
    square = float(board["square_size_mm"])
    x_min, x_max = -square, int(board["pattern_cols"]) * square
    y_min, y_max = -square, int(board["pattern_rows"]) * square
    inset = float(extraction.get("board_mask_inset_mm", 0.0))
    x_min, x_max, y_min, y_max = x_min + inset, x_max - inset, y_min + inset, y_max - inset
    xb = np.full(len(frame), np.nan)
    yb = np.full(len(frame), np.nan)
    zb = np.full(len(frame), np.nan)
    sources = [""] * len(frame)
    pnp_rmse = np.full(len(frame), np.nan)
    for frame_id, indices in frame.groupby("frame_id").groups.items():
        pose = poses[str(frame_id).zfill(3)]
        rotation, _ = cv2.Rodrigues(pose["rvec"].reshape(3, 1))
        points = frame.loc[indices, ["Xc_mm", "Yc_mm", "Zc_mm"]].to_numpy(float)
        local = (points - pose["tvec"][None, :]) @ rotation
        xb[indices], yb[indices], zb[indices] = local[:, 0], local[:, 1], local[:, 2]
        for index in indices:
            sources[index] = str(pose["source"])
            pnp_rmse[index] = float(pose["pnp_rmse_px"])
    distance = np.minimum.reduce([xb - x_min, x_max - xb, yb - y_min, y_max - yb])
    frame["Xb_mm"], frame["Yb_mm"], frame["Zb_mm"] = xb, yb, zb
    frame["mask_boundary_distance_mm"] = distance
    frame["mask_boundary_band"] = np.select(
        [distance <= 5.0, distance <= 10.0],
        ["edge_0_5mm", "edge_5_10mm"],
        default="interior_gt_10mm",
    )
    frame["pnp_boundary_source"] = sources
    frame["pnp_boundary_rmse_px"] = pnp_rmse


def add_bins(frame: pd.DataFrame, s_min: float, s_max: float) -> None:
    v = frame["v_px"].to_numpy(float)
    v_index = np.floor(v / V_BIN_WIDTH).astype(int)
    frame["v_bin"] = [
        f"v_{i * 100:04d}_{i * 100 + 100:04d}" if 0 <= i < 30 else "out_of_range"
        for i in v_index
    ]
    if s_max > s_min:
        edges = np.linspace(s_min, s_max, PCA_BIN_COUNT + 1)
        s_index = np.full(len(frame), -1, dtype=int)
        finite_s = np.isfinite(frame["pca_s"].to_numpy(float))
        s_index[finite_s] = np.clip(
            np.digitize(frame.loc[finite_s, "pca_s"].to_numpy(float), edges[1:-1]),
            0,
            PCA_BIN_COUNT - 1,
        )
    else:
        edges = np.linspace(s_min - 0.5, s_max + 0.5, PCA_BIN_COUNT + 1)
        s_index = np.zeros(len(frame), dtype=int)
    frame["s_bin_index"] = s_index
    frame["s_bin"] = [
        f"s_{edges[i]:+.5f}_{edges[i + 1]:+.5f}" if i >= 0 else "invalid"
        for i in s_index
    ]
    rank = frame["response"].rank(method="average", pct=True).to_numpy()
    q = np.minimum((rank * 5).astype(int), 4)
    frame["response_quintile"] = [f"response_q{i * 20:02d}_{i * 20 + 20:02d}" for i in q]


def compute_trends(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = {"v": "v_px", "s": "pca_s", "t": "pca_t"}
    residuals = {"raw": "residual_mm", "frame_median_subtracted": "residual_centered_mm"}
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("global", "", frame)]
    scopes.extend(("frame", str(fid), part) for fid, part in frame.groupby("frame_id"))
    for scope, frame_id, part in scopes:
        for centering, residual_column in residuals.items():
            for predictor, column in predictors.items():
                rows.append(
                    {
                        "scope": scope,
                        "frame_id": frame_id,
                        "centering": centering,
                        "predictor": predictor,
                        **trend_stats(part[column].to_numpy(float), part[residual_column].to_numpy(float)),
                    }
                )
    trends = pd.DataFrame(rows)
    aggregates: list[dict[str, Any]] = []
    for centering in residuals:
        for predictor in predictors:
            global_row = trends[
                (trends["scope"] == "global")
                & (trends["centering"] == centering)
                & (trends["predictor"] == predictor)
            ].iloc[0]
            part = trends[
                (trends["scope"] == "frame")
                & (trends["centering"] == centering)
                & (trends["predictor"] == predictor)
            ]
            slopes = part["slope_mm_per_x_span"].to_numpy(float)
            amps = part["low_frequency_amplitude_mm"].to_numpy(float)
            ev = part["binned_explained_fraction"].to_numpy(float)
            slopes, amps, ev = slopes[np.isfinite(slopes)], amps[np.isfinite(amps)], ev[np.isfinite(ev)]
            global_slope = float(global_row["slope_mm_per_x_span"])
            if len(slopes):
                same_sign = float(np.mean(np.sign(slopes) == np.sign(global_slope))) if abs(global_slope) > 1.0e-12 else float(max(np.mean(slopes >= 0), np.mean(slopes <= 0)))
            else:
                same_sign = math.nan
            aggregates.append(
                {
                    "scope": "frame_aggregate",
                    "centering": centering,
                    "predictor": predictor,
                    "frame_count": int(len(part)),
                    "global_slope_mm_per_x_span": global_slope,
                    "frame_slope_median_mm_per_x_span": float(np.median(slopes)) if len(slopes) else math.nan,
                    "frame_slope_p05_mm_per_x_span": float(np.percentile(slopes, 5)) if len(slopes) else math.nan,
                    "frame_slope_p95_mm_per_x_span": float(np.percentile(slopes, 95)) if len(slopes) else math.nan,
                    "frame_same_sign_fraction": same_sign,
                    "frame_binned_ev_median": float(np.median(ev)) if len(ev) else math.nan,
                    "frame_low_frequency_amplitude_median_mm": float(np.median(amps)) if len(amps) else math.nan,
                    "frame_low_frequency_amplitude_p75_mm": float(np.percentile(amps, 75)) if len(amps) else math.nan,
                }
            )
    return trends, pd.DataFrame(aggregates)


def classify_observability(pca: Mapping[str, Any], trends: pd.DataFrame, aggregates: pd.DataFrame) -> tuple[str, str, dict[str, Any]]:
    def row(centering: str, predictor: str) -> pd.Series:
        return aggregates[(aggregates["centering"] == centering) & (aggregates["predictor"] == predictor)].iloc[0]

    def global_row(predictor: str) -> pd.Series:
        return trends[(trends["scope"] == "global") & (trends["centering"] == "frame_median_subtracted") & (trends["predictor"] == predictor)].iloc[0]

    s, t = row("frame_median_subtracted", "s"), row("frame_median_subtracted", "t")
    gs, gt = global_row("s"), global_row("t")
    s_amp = max(float(s["frame_low_frequency_amplitude_median_mm"]), float(gs["low_frequency_amplitude_mm"]))
    t_amp = max(float(t["frame_low_frequency_amplitude_median_mm"]), float(gt["low_frequency_amplitude_mm"]))
    frame_ratios = []
    for _, part in trends[trends["scope"] == "frame"].groupby("frame_id"):
        ss = part[part["predictor"] == "s"]["x_span"].iloc[0]
        tt = part[part["predictor"] == "t"]["x_span"].iloc[0]
        frame_ratios.append(float(tt / ss) if ss > 0 else math.nan)
    ratios = np.asarray(frame_ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    t_to_s = float(pca["t_robust_span"] / pca["s_robust_span"])
    median_ratio = float(np.median(ratios)) if len(ratios) else math.nan
    fraction_010 = float(np.mean(ratios >= 0.10)) if len(ratios) else math.nan
    clear_2d = float(pca["explained_t"]) >= 0.10 and t_to_s >= 0.15 and median_ratio >= 0.10 and fraction_010 >= 0.70
    strong_s = s_amp >= 3.0 * PNP_UNCERTAINTY_HIGH_MM and float(s["frame_same_sign_fraction"]) >= 0.60 and float(s["frame_binned_ev_median"]) >= 0.10
    moderate_s = s_amp >= 1.5 * PNP_UNCERTAINTY_HIGH_MM and float(s["frame_same_sign_fraction"]) >= 0.40 and float(s["frame_binned_ev_median"]) >= 0.03
    t_actionable = t_amp >= 1.5 * PNP_UNCERTAINTY_HIGH_MM and float(t["frame_same_sign_fraction"]) >= 0.40 and float(t["frame_binned_ev_median"]) >= 0.03
    if clear_2d and t_actionable:
        label, next_step = "NEED_2D_CHECK", "CHECK_2D"
    elif strong_s:
        label, next_step = "STRONG_1D", "TRY_1D_FS"
    elif moderate_s:
        label, next_step = "MODERATE_1D", "TRY_1D_FS"
    else:
        label, next_step = "WEAK", "DO_NOT_COMPENSATE_YET"
    evidence = {
        "s_evidence_amplitude_mm": s_amp,
        "t_evidence_amplitude_mm": t_amp,
        "s_frame_same_sign_fraction": float(s["frame_same_sign_fraction"]),
        "t_frame_same_sign_fraction": float(t["frame_same_sign_fraction"]),
        "s_frame_binned_ev_median": float(s["frame_binned_ev_median"]),
        "t_frame_binned_ev_median": float(t["frame_binned_ev_median"]),
        "global_t_to_s_robust_span_ratio": t_to_s,
        "median_frame_t_to_s_robust_span_ratio": median_ratio,
        "frame_fraction_t_to_s_ge_0.10": fraction_010,
        "ray_support_clear_2d": clear_2d,
        "t_actionable": t_actionable,
    }
    return label, next_step, evidence


def make_tails(frame: pd.DataFrame, p95_abs: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [("abs_gt_p95", p95_abs), ("abs_gt_0.30", 0.30), ("abs_gt_0.40", 0.40)]
    parts, summaries = [], []
    for name, threshold in specs:
        part = frame[frame["abs_residual_mm"] > threshold].copy()
        part.insert(0, "threshold", name)
        parts.append(part)
        frame_counts = part["frame_id"].value_counts()
        v_counts = part["v_bin"].value_counts()
        s_counts = part["s_bin"].value_counts()
        boundary_counts = part["mask_boundary_band"].value_counts()
        response_counts = part["response_quintile"].value_counts()
        count = len(part)
        summaries.append(
            {
                "threshold": name,
                "threshold_mm": threshold,
                "tail_count": count,
                "tail_fraction": float(count / len(frame)),
                "unique_frame_count": int(part["frame_id"].nunique()),
                "top_frame": str(frame_counts.index[0]) if len(frame_counts) else "",
                "top_frame_count": int(frame_counts.iloc[0]) if len(frame_counts) else 0,
                "top_frame_share": float(frame_counts.iloc[0] / count) if count else math.nan,
                "top3_frame_share": float(frame_counts.head(3).sum() / count) if count else math.nan,
                "top_v_bin": str(v_counts.index[0]) if len(v_counts) else "",
                "top_v_bin_share": float(v_counts.iloc[0] / count) if count else math.nan,
                "top_s_bin": str(s_counts.index[0]) if len(s_counts) else "",
                "top_s_bin_share": float(s_counts.iloc[0] / count) if count else math.nan,
                "boundary_edge_share": float((boundary_counts.get("edge_0_5mm", 0) + boundary_counts.get("edge_5_10mm", 0)) / count) if count else math.nan,
                "boundary_0_5mm_share": float(boundary_counts.get("edge_0_5mm", 0) / count) if count else math.nan,
                "response_low20_share": float(response_counts.get("response_q00_20", 0) / count) if count else math.nan,
                "v_median_px": float(part["v_px"].median()) if count else math.nan,
                "s_median": float(part["pca_s"].median()) if count else math.nan,
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summaries)


def classify_tail(summary: pd.DataFrame, frame_count: int) -> tuple[str, dict[str, Any]]:
    p95 = summary[summary["threshold"] == "abs_gt_p95"].iloc[0]
    mid = summary[summary["threshold"] == "abs_gt_0.30"].iloc[0]
    high = summary[summary["threshold"] == "abs_gt_0.40"].iloc[0]
    frame_dominated = float(mid["top_frame_share"]) >= 0.45 or float(mid["top3_frame_share"]) >= 0.75 or float(high["top_frame_share"]) >= 0.55
    repeated = float(p95["unique_frame_count"]) >= 0.60 * frame_count and float(p95["top_frame_share"]) <= 0.25 and float(mid["unique_frame_count"]) >= 0.40 * frame_count
    edge_or_low_response = float(high["boundary_edge_share"]) >= 0.60 or float(high["response_low20_share"]) >= 0.60
    spatially_concentrated = float(mid["top_v_bin_share"]) >= 0.35 or float(mid["top_s_bin_share"]) >= 0.35
    if frame_dominated and (edge_or_low_response or repeated):
        label = "MIXED"
    elif frame_dominated:
        label = "FRAME_DOMINATED"
    elif repeated and not edge_or_low_response:
        label = "SYSTEMATIC"
    else:
        label = "OUTLIER_DOMINATED"
    return label, {
        "frame_dominated_gate": frame_dominated,
        "repeated_cross_frame_gate": repeated,
        "edge_or_low_response_gate": edge_or_low_response,
        "spatially_concentrated_gate": spatially_concentrated,
    }


def plot_v(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    sample = frame.sample(min(len(frame), 12000), random_state=20260818)
    axes[0, 0].scatter(sample["v_px"], sample["residual_mm"], s=2, alpha=0.12)
    axes[0, 0].set_title("Raw residual vs v")
    axes[0, 1].scatter(sample["v_px"], sample["residual_centered_mm"], s=2, alpha=0.12, color="#a63603")
    axes[0, 1].set_title("Frame-median-subtracted residual vs v")
    grouped = frame.assign(v_index=np.floor(frame["v_px"] / V_BIN_WIDTH).astype(int)).groupby("v_index").agg(
        v_px=("v_px", "mean"),
        mean=("residual_centered_mm", "mean"),
        p05=("residual_centered_mm", lambda x: np.percentile(x, 5)),
        p95=("residual_centered_mm", lambda x: np.percentile(x, 95)),
    ).reset_index()
    axes[1, 0].plot(grouped["v_px"], grouped["mean"], color="#238b45")
    axes[1, 0].fill_between(grouped["v_px"], grouped["p05"], grouped["p95"], alpha=0.2, color="#238b45")
    axes[1, 0].set_title("Centered residual: 100 px v bins")
    medians = frame.groupby("frame_id")["residual_mm"].median().reindex(FIT_IDS)
    axes[1, 1].bar(np.arange(len(medians)), medians.to_numpy(float), color="#756bb1")
    axes[1, 1].set_title("Frame residual medians")
    axes[1, 1].set_xticks(np.arange(len(medians))[::4], medians.index.to_numpy()[::4], rotation=45)
    for axis in axes.ravel():
        axis.axhline(0, color="black", lw=0.7)
        axis.grid(alpha=0.2)
        axis.set_ylabel("residual / mm")
    axes[0, 0].set_xlabel("v / px")
    axes[0, 1].set_xlabel("v / px")
    axes[1, 0].set_xlabel("v / px")
    axes[1, 1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_s(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    sample = frame.sample(min(len(frame), 12000), random_state=20260818)
    for axis, predictor, column, title, color in [
        (axes[0, 0], "pca_s", "residual_mm", "Raw residual vs PCA s", "#24527a"),
        (axes[0, 1], "pca_s", "residual_centered_mm", "Centered residual vs PCA s", "#a63603"),
        (axes[1, 0], "pca_t", "residual_mm", "Raw residual vs PCA t", "#238b45"),
        (axes[1, 1], "pca_t", "residual_centered_mm", "Centered residual vs PCA t", "#762a83"),
    ]:
        axis.scatter(sample[predictor], sample[column], s=2, alpha=0.12, color=color)
        axis.axhline(0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel(predictor)
        axis.set_ylabel("residual / mm")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_tail(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {"abs_gt_p95": "#3182bd", "abs_gt_0.30": "#e6550d", "abs_gt_0.40": "#a50f15"}
    for name, color in colors.items():
        threshold = float(summary.loc[summary["threshold"] == name, "threshold_mm"].iloc[0])
        part = frame[frame["abs_residual_mm"] > threshold]
        counts = part["frame_id"].value_counts().reindex(FIT_IDS, fill_value=0)
        axes[0, 0].plot(np.arange(len(counts)), counts.to_numpy(), marker=".", color=color, label=name)
        v_counts = part["v_bin"].value_counts()
        axes[0, 1].plot([v_counts.get(f"v_{i:04d}_{i + 100:04d}", 0) for i in range(0, 3000, 100)], color=color, label=name)
        s_counts = part["s_bin_index"].value_counts().reindex(range(PCA_BIN_COUNT), fill_value=0)
        axes[0, 2].plot(s_counts.to_numpy(), marker=".", color=color, label=name)
        b_counts = part["mask_boundary_band"].value_counts()
        axes[1, 0].plot([b_counts.get(v, 0) for v in ["edge_0_5mm", "edge_5_10mm", "interior_gt_10mm"]], marker="o", color=color, label=name)
        r_counts = part["response_quintile"].value_counts().reindex([f"response_q{i:02d}_{i + 20:02d}" for i in range(0, 100, 20)], fill_value=0)
        axes[1, 1].plot(r_counts.to_numpy(), marker="o", color=color, label=name)
    axes[0, 0].set_title("Tail count by frame")
    axes[0, 1].set_title("Tail count by v bin")
    axes[0, 2].set_title("Tail count by PCA s bin")
    axes[1, 0].set_title("Tail count by mask boundary band")
    axes[1, 1].set_title("Tail count by response quintile")
    axes[1, 0].set_xticks(range(3), ["0-5 mm", "5-10 mm", ">10 mm"], rotation=25)
    axes[1, 1].set_xlabel("response quintile, low to high")
    axes[1, 2].hist(frame["abs_residual_mm"], bins=80, color="#9ecae1")
    axes[1, 2].axvline(float(summary.loc[summary["threshold"] == "abs_gt_p95", "threshold_mm"].iloc[0]), color="#3182bd", label="P95")
    axes[1, 2].axvline(0.30, color="#e6550d", label="0.30 mm")
    axes[1, 2].axvline(0.40, color="#a50f15", label="0.40 mm")
    axes[1, 2].set_title("Absolute residual distribution")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)
    axes[1, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def report_text(
    frame: pd.DataFrame,
    pca: Mapping[str, Any],
    trends: pd.DataFrame,
    aggregates: pd.DataFrame,
    tails: pd.DataFrame,
    observability: str,
    next_step: str,
    evidence: Mapping[str, Any],
    tail_structure: str,
    tail_evidence: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
) -> str:
    residual = frame["residual_mm"].to_numpy(float)
    absolute = frame["abs_residual_mm"].to_numpy(float)
    medians = frame.groupby("frame_id")["residual_mm"].median()
    lines = [
        "# Frozen Full-36 Quadratic residual observability and tail audit",
        "",
        f"QUADRATIC_RESIDUAL_OBSERVABILITY = {observability}",
        f"TAIL_STRUCTURE = {tail_structure}",
        f"C1_NEXT_STEP = {next_step}",
        "",
        "## 结论",
        "",
        f"- 使用完整 Full-36 FIT：{len(FIT_IDS)} poses、{len(frame):,} points；Validation 未读取。",
        "- residual 定义为 r = lambda_truth - lambda_quadratic。冻结 YAML 只加载并计算 ray-surface intersection，没有调用 fit。",
        f"- raw bias={np.mean(residual):.6f} mm，RMSE={np.sqrt(np.mean(residual ** 2)):.6f} mm，P95(|r|)={np.percentile(absolute, 95):.6f} mm，Max(|r|)={np.max(absolute):.6f} mm。",
        f"- frame median 范围={medians.min():.6f}–{medians.max():.6f} mm，跨度={np.ptp(medians):.6f} mm；去 frame median 后 P05–P95={np.percentile(frame['residual_centered_mm'], 5):.6f}–{np.percentile(frame['residual_centered_mm'], 95):.6f} mm。",
        f"- observability={observability}；next step={next_step}。这只是进入 C1 实验的门控，不是已经拟合 C1。",
        "",
        "## Artifact provenance / reuse audit",
        "",
        "- 直接复用 Full-36 calibration points、camera rays 和 PnP truth；未重复 Steger 提点。",
        "- 30 帧的 board-local mask boundary 坐标复用既有 frame geometry；049–054 只读取 chess 图补充 PnP 几何，不读取 laser 图。",
        "",
        "| artifact | action | status | notes |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {r['artifact']} | {r['action']} | {r['status']} | {r['notes']} |" for r in provenance)
    gs = trends[(trends["scope"] == "global") & (trends["centering"] == "frame_median_subtracted") & (trends["predictor"] == "s")].iloc[0]
    gt = trends[(trends["scope"] == "global") & (trends["centering"] == "frame_median_subtracted") & (trends["predictor"] == "t")].iloc[0]
    gv = trends[(trends["scope"] == "global") & (trends["centering"] == "frame_median_subtracted") & (trends["predictor"] == "v")].iloc[0]
    lines.extend(
        [
            "",
            "## Residual observability",
            "",
            f"- PCA center=({pca['center_xn']:.8g}, {pca['center_yn']:.8g}); explained variance s={pca['explained_s']:.4f}, t={pca['explained_t']:.4f}; sqrt eigenvalue ratio={pca['anisotropy_sqrt_eigenvalue_ratio']:.3f}.",
            f"- Robust span s={pca['s_robust_span']:.6f}, t={pca['t_robust_span']:.6f}, t/s={evidence['global_t_to_s_robust_span_ratio']:.4f}.",
            f"- Per-frame t/s median={evidence['median_frame_t_to_s_robust_span_ratio']:.4f}; fraction >=0.10={evidence['frame_fraction_t_to_s_ge_0.10']:.3f}.",
            "",
            "| predictor | centered low-frequency amplitude / mm | binned EV | frame same-sign | frame EV median |",
            "|---|---:|---:|---:|---:|",
            f"| s | {gs['low_frequency_amplitude_mm']:.6f} | {gs['binned_explained_fraction']:.4f} | {evidence['s_frame_same_sign_fraction']:.3f} | {evidence['s_frame_binned_ev_median']:.4f} |",
            f"| t | {gt['low_frequency_amplitude_mm']:.6f} | {gt['binned_explained_fraction']:.4f} | {evidence['t_frame_same_sign_fraction']:.3f} | {evidence['t_frame_binned_ev_median']:.4f} |",
            f"| v | {gv['low_frequency_amplitude_mm']:.6f} | {gv['binned_explained_fraction']:.4f} | {float(aggregates[(aggregates['centering']=='frame_median_subtracted') & (aggregates['predictor']=='v')]['frame_same_sign_fraction'].iloc[0]):.3f} | {float(aggregates[(aggregates['centering']=='frame_median_subtracted') & (aggregates['predictor']=='v')]['frame_binned_ev_median'].iloc[0]):.4f} |",
            "",
            "诊断量级参考沿用旧方法：PnP uncertainty 0.025–0.033 mm；strong 需要幅度至少 0.099 mm、同号至少 0.60、frame EV 至少 0.10；moderate 需要幅度至少 0.0495 mm、同号至少 0.40、frame EV 至少 0.03。",
            "",
            "## Tail audit",
            "",
            "| threshold | count | fraction | unique frames | top frame | top frame share | top v-bin | top s-bin | boundary edge share | response low20 share |",
            "|---|---:|---:|---:|---|---:|---|---|---:|---:|",
        ]
    )
    for _, row in tails.iterrows():
        lines.append(
            f"| {row['threshold']} | {int(row['tail_count'])} | {row['tail_fraction']:.4f} | {int(row['unique_frame_count'])} | {row['top_frame']} | {row['top_frame_share']:.3f} | {row['top_v_bin']} | {row['top_s_bin']} | {row['boundary_edge_share']:.3f} | {row['response_low20_share']:.3f} |"
        )
    lines.extend(
        [
            "",
            "tail_points.csv 为长表，按三个 threshold 保留所有 tail points；quadratic_residual_points.csv 保留全部点，不因 residual 大而删除。",
            "",
            "## Classification evidence",
            "",
            f"- clear 2D ray support={evidence['ray_support_clear_2d']}；t actionable={evidence['t_actionable']}。",
            f"- s evidence amplitude={evidence['s_evidence_amplitude_mm']:.6f} mm；t evidence amplitude={evidence['t_evidence_amplitude_mm']:.6f} mm。",
            f"- tail gates: frame_dominated={tail_evidence['frame_dominated_gate']}; repeated_cross_frame={tail_evidence['repeated_cross_frame_gate']}; edge_or_low_response={tail_evidence['edge_or_low_response_gate']}; spatially_concentrated={tail_evidence['spatially_concentrated_gate']}。",
            "",
            "## Scope exclusions",
            "",
            "- 未读取 019–024、037–040、055–060；未使用 Validation residual 调参。",
            "- 未重新拟合 Quadratic C0；未拟合 C1；未修改 Steger、mask、weighting。",
            "- 未因 residual 大而删除 pose 或点。",
            "",
        ]
    )
    return "\n".join(lines)


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
    points = pd.read_csv(FULL_POINTS, dtype={"frame_id": str})
    points["frame_id"] = points["frame_id"].astype(str).str.zfill(3)
    if sorted(points["frame_id"].unique()) != sorted(FIT_IDS) or len(points) != 32400:
        raise RuntimeError("Full-36 point artifact does not match 36 poses / 32400 points")
    cfg = load_yaml(FORMAL_CONFIG)
    extraction = cfg["extraction"]
    if str(extraction.get("board_mask_mode", "")).lower() != "full_board_physical" or abs(float(extraction.get("board_mask_inset_mm", 0))) > 1.0e-12:
        raise RuntimeError("Formal config is not full_board_physical inset=0")
    k, d = load_yaml(INTRINSICS).get("camera_matrix"), load_yaml(INTRINSICS).get("dist_coeffs")
    k, d = np.asarray(k, dtype=float), np.asarray(d, dtype=float)
    model, plane, model_data = load_models()
    rays = points[["ray_x", "ray_y", "ray_z"]].to_numpy(float)
    lambda_hint = plane.intersect_rays(rays)
    lambda_q = model.intersect_rays(rays, lambda_hint=lambda_hint)
    points["lambda_quadratic_mm"] = lambda_q
    points["quadratic_valid"] = np.isfinite(lambda_q)
    points["residual_mm"] = points["lambda_truth_mm"].to_numpy(float) - lambda_q
    points["abs_residual_mm"] = np.abs(points["residual_mm"])
    points["xn"] = rays[:, 0] / np.maximum(rays[:, 2], np.finfo(float).eps)
    points["yn"] = rays[:, 1] / np.maximum(rays[:, 2], np.finfo(float).eps)
    valid = np.isfinite(points["residual_mm"].to_numpy(float))
    pca = pca_rays(points.loc[valid, ["xn", "yn"]].to_numpy(float))
    points["pca_s"], points["pca_t"] = np.nan, np.nan
    points.loc[valid, "pca_s"], points.loc[valid, "pca_t"] = pca["s"], pca["t"]
    frame_medians = points.loc[valid].groupby("frame_id")["residual_mm"].median()
    points["frame_residual_median_mm"] = points["frame_id"].map(frame_medians)
    points["residual_centered_mm"] = points["residual_mm"] - points["frame_residual_median_mm"]
    points["abs_centered_residual_mm"] = np.abs(points["residual_centered_mm"])
    s_finite = points.loc[valid, "pca_s"].to_numpy(float)
    add_bins(points, float(np.min(s_finite)), float(np.max(s_finite)))
    poses, pose_audit = load_pose_sources(k, d, cfg)
    add_board_geometry(points, poses, cfg)
    valid_points = points.loc[valid].copy()
    trends, aggregates = compute_trends(valid_points)
    observability, next_step, evidence = classify_observability(pca, trends, aggregates)
    p95_abs = float(np.percentile(valid_points["abs_residual_mm"], 95))
    tail_points, tail_summary = make_tails(valid_points, p95_abs)
    tail_structure, tail_evidence = classify_tail(tail_summary, len(FIT_IDS))
    provenance = [
        {"artifact": "Full-36 calibration points/ray/PnP truth", "path": str(FULL_POINTS), "action": "REUSED_EXISTING", "status": "CONFIRMED", "notes": "32400 points; no re-extraction"},
        {"artifact": "Frozen Full-36 Quadratic YAML", "path": str(FROZEN_MODEL), "action": "LOADED_ONLY", "status": "CONFIRMED", "notes": f"sha256={sha256_file(FROZEN_MODEL)}; fit() not called"},
        {"artifact": "Full-36 metadata", "path": str(FULL_METADATA), "action": "REUSED_EXISTING", "status": "CONFIRMED", "notes": "source/mask/point-count assertions passed"},
        {"artifact": "Formal extraction config", "path": str(FORMAL_CONFIG), "action": "READ_ONLY_ASSERTED", "status": "CONFIRMED", "notes": "full_board_physical; inset=0; unchanged Steger/continuity"},
        {"artifact": "Intrinsics", "path": str(INTRINSICS), "action": "REUSED_EXISTING", "status": "CONFIRMED", "notes": f"sha256={sha256_file(INTRINSICS)}"},
        {"artifact": "Frame board geometry summary", "path": str(FRAME_GEOMETRY), "action": "REUSED_EXISTING", "status": "CONFIRMED", "notes": "30 frame PnP poses for board-local boundary coordinates"},
        {"artifact": "FIT chess 049-054", "path": f"{FIT_ROOT}; {FIT_ROOT_0817}", "action": "READ_CHESS_ONLY", "status": "SUPPLEMENTAL", "notes": "PnP boundary geometry only; laser points not re-extracted"},
        {"artifact": "Validation datasets", "path": "019-024;037-040;055-060", "action": "NOT_READ", "status": "EXCLUDED", "notes": "FIT-only audit"},
        {"artifact": "Old Cone observability script", "path": str(ROOT / "scripts/audit_spatial_residual_observability.py"), "action": "REFERENCE_ONLY", "status": "REFERENCE", "notes": "method reference only; old output excluded 049-054"},
    ]
    points.to_csv(output / "quadratic_residual_points.csv", index=False)
    tail_points.to_csv(output / "tail_points.csv", index=False)
    pd.concat([trends, aggregates.assign(frame_id="")], ignore_index=True, sort=False).to_csv(output / "residual_trends.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "artifact_provenance.csv", index=False)
    plot_v(valid_points, output / "residual_vs_v.png")
    plot_s(valid_points, output / "residual_vs_s.png")
    plot_tail(valid_points, tail_summary, output / "tail_distribution.png")
    summary = {
        "QUADRATIC_RESIDUAL_OBSERVABILITY": observability,
        "TAIL_STRUCTURE": tail_structure,
        "C1_NEXT_STEP": next_step,
        "fit_ids": FIT_IDS,
        "point_count": len(points),
        "valid_quadratic_count": int(np.count_nonzero(valid)),
        "validation_read": False,
        "c0_refit": False,
        "c1_fit": False,
        "frozen_model_sha256": sha256_file(FROZEN_MODEL),
        "intrinsics_sha256": sha256_file(INTRINSICS),
        "formal_config_sha256": sha256_file(FORMAL_CONFIG),
        "pca": {key: value for key, value in pca.items() if key not in {"s", "t"}},
        "observability_evidence": evidence,
        "tail_summary": tail_summary.to_dict(orient="records"),
        "tail_evidence": tail_evidence,
        "pose_audit": pose_audit,
        "model_type": model_data["model_type"],
    }
    (output / "audit_summary.json").write_text(json.dumps(json_clean(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report_text(valid_points, pca, trends, aggregates, tail_summary, observability, next_step, evidence, tail_structure, tail_evidence, provenance), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "point_count": len(points), "valid_quadratic_count": int(np.count_nonzero(valid)), "QUADRATIC_RESIDUAL_OBSERVABILITY": observability, "TAIL_STRUCTURE": tail_structure, "C1_NEXT_STEP": next_step}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
