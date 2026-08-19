#!/usr/bin/env python3
"""FIT-only grouped-CV for a low-DoF 1D C1 correction on frozen Quadratic C0.

This analysis deliberately consumes the existing residual artifact rather than
re-extracting points or recomputing the frozen C0.  It does not read
Validation, remove frame027, refit C0, or modify a production configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
PROJECT = ROOT / "projects" / "daheng"

FIT_IDS = [
    *[f"{i:03d}" for i in range(1, 19)],
    *[f"{i:03d}" for i in range(25, 37)],
    *[f"{i:03d}" for i in range(49, 55)],
]
FRAME027 = "027"
V_BIN_WIDTH = 100.0
V_BIN_COUNT = 30
DEFAULT_POINTS = PROJECT / "outputs/0818/quadratic_residual_observability/quadratic_residual_points.csv"
DEFAULT_AUDIT = PROJECT / "outputs/0818/quadratic_residual_observability/audit_summary.json"
DEFAULT_FROZEN_MODEL = PROJECT / "outputs/0818/c0_freeze/quadratic_graph.yaml"
DEFAULT_OUTPUT = PROJECT / "outputs/0818/c1_frozen_quadratic_grouped_cv"

# This is the fixed analysis protocol.  The penalty is not tuned per candidate.
SMOOTHNESS_PENALTY = 0.1
HUBER_K = 1.345
ROBUST_MAX_ITER = 30
ROBUST_TOL = 1.0e-9

# Descriptive/gating thresholds.  Max is intentionally absent from all gates.
MEANINGFUL_GAIN_PCT = 2.0
FULL_CV_GAIN_GATE_PCT = 5.0
POSE_RATIO_GATE = 0.60
WORST_P95_DEGRADATION_GATE_PCT = 2.0
CURVE_MATERIAL_MAX_DELTA_MM = 0.025
CURVE_MATERIAL_RMS_DELTA_MM = 0.010


@dataclass(frozen=True)
class Candidate:
    model_id: str
    interior_knot_count: int

    @property
    def basis_count(self) -> int:
        return self.interior_knot_count + 4


CANDIDATES = [Candidate(f"C1_{k}k", k) for k in (3, 4, 5)]


@dataclass
class SplineFit:
    candidate: Candidate
    domain_min: float
    domain_max: float
    knots: np.ndarray
    coefficients: np.ndarray
    robust_scale_mm: float
    robust_iterations: int
    training_frame_count: int
    training_point_count: int

    def predict(self, s: np.ndarray) -> np.ndarray:
        return spline_design(np.asarray(s, dtype=float), self.knots, self.domain_min, self.domain_max) @ self.coefficients


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def assert_reuse_contract(points_path: Path, audit_path: Path, frozen_model_path: Path) -> pd.DataFrame:
    """Load only the existing FIT residual artifact and verify its provenance."""

    if not points_path.is_file():
        raise FileNotFoundError(points_path)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if not frozen_model_path.is_file():
        raise FileNotFoundError(frozen_model_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("validation_read") is not False:
        raise RuntimeError("Reuse audit does not prove Validation was excluded")
    if audit.get("c0_refit") is not False:
        raise RuntimeError("Reuse audit does not prove C0 was frozen")
    if audit.get("c1_fit") is not False:
        raise RuntimeError("Input residual audit already contains a C1 fit")
    if audit.get("model_type") != "quadratic_graph":
        raise RuntimeError("Input residual audit is not based on Quadratic C0")
    if audit.get("frozen_model_sha256") != sha256_file(frozen_model_path):
        raise RuntimeError("Frozen C0 hash does not match the residual audit")

    points = pd.read_csv(points_path, dtype={"frame_id": str})
    points["frame_id"] = points["frame_id"].astype(str).str.zfill(3)
    if len(points) != 32400:
        raise RuntimeError(f"Expected 32400 reused points, got {len(points)}")
    if sorted(points["frame_id"].unique().tolist()) != sorted(FIT_IDS):
        raise RuntimeError("Reused residual points do not contain exactly the Full-36 FIT poses")
    if set(points["split"].astype(str).str.lower()) != {"fit"}:
        raise RuntimeError("Reused residual artifact is not FIT-only")
    valid = points["quadratic_valid"].astype(bool).to_numpy()
    if not np.all(valid):
        raise RuntimeError("Frozen Quadratic residual artifact contains invalid intersections")
    required = {
        "residual_mm",
        "residual_centered_mm",
        "frame_residual_median_mm",
        "pca_s",
        "v_px",
        "frame_id",
    }
    missing = sorted(required.difference(points.columns))
    if missing:
        raise RuntimeError(f"Reused residual artifact is missing columns: {missing}")
    residual = points["residual_mm"].to_numpy(float)
    centered = points["residual_centered_mm"].to_numpy(float)
    frame_medians = points["frame_residual_median_mm"].to_numpy(float)
    if not np.allclose(centered, residual - frame_medians, atol=1.0e-12, rtol=1.0e-12):
        raise RuntimeError("Stored frame-median-centered target is inconsistent with stored residuals")
    if not np.all(np.isfinite(points[["residual_centered_mm", "pca_s", "v_px"]].to_numpy(float))):
        raise RuntimeError("Reused C1 target or predictor contains non-finite values")
    return points


def frame_balanced_weights(frame_ids: Sequence[str], robust_factor: np.ndarray | None = None) -> np.ndarray:
    frame_ids = np.asarray(frame_ids).astype(str)
    factors = np.ones(len(frame_ids), dtype=float) if robust_factor is None else np.asarray(robust_factor, dtype=float)
    weights = np.zeros(len(frame_ids), dtype=float)
    for frame_id in np.unique(frame_ids):
        indices = np.flatnonzero(frame_ids == frame_id)
        local = np.maximum(factors[indices], 1.0e-12)
        weights[indices] = local / np.sum(local)
    return weights


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return math.nan
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(np.interp(float(quantile) * cumulative[-1], cumulative, values))


def spline_knots(domain_min: float, domain_max: float, interior_knot_count: int) -> np.ndarray:
    if not domain_min < domain_max:
        raise ValueError("PCA s domain must have positive width")
    interior = np.linspace(domain_min, domain_max, interior_knot_count + 2, dtype=float)[1:-1]
    return np.concatenate(
        [
            np.repeat(domain_min, 4),
            interior,
            np.repeat(domain_max, 4),
        ]
    )


def spline_design(s: np.ndarray, knots: np.ndarray, domain_min: float, domain_max: float) -> np.ndarray:
    clipped = np.clip(np.asarray(s, dtype=float), domain_min, domain_max)
    return BSpline.design_matrix(clipped, knots, k=3, extrapolate=False).toarray()


def second_difference_matrix(basis_count: int) -> np.ndarray:
    matrix = np.zeros((max(basis_count - 2, 0), basis_count), dtype=float)
    for row in range(len(matrix)):
        matrix[row, row : row + 3] = (1.0, -2.0, 1.0)
    return matrix


def solve_penalized_wls(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    penalty_matrix: np.ndarray,
    smoothness_penalty: float,
) -> np.ndarray:
    weighted_design = design * weights[:, None]
    normal = design.T @ weighted_design
    if len(penalty_matrix):
        normal = normal + smoothness_penalty * (penalty_matrix.T @ penalty_matrix)
    normal = normal + 1.0e-12 * np.eye(normal.shape[0])
    rhs = design.T @ (weights * target)
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(normal, rhs, rcond=None)[0]


def robust_scale(residual: np.ndarray, weights: np.ndarray) -> float:
    center = weighted_quantile(residual, weights, 0.5)
    mad = weighted_quantile(np.abs(residual - center), weights, 0.5)
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1.0e-5:
        scale = math.sqrt(float(np.average(residual * residual, weights=weights)))
    return max(float(scale), 1.0e-5)


def fit_robust_spline(
    data: pd.DataFrame,
    candidate: Candidate,
    domain_min: float,
    domain_max: float,
    smoothness_penalty: float = SMOOTHNESS_PENALTY,
) -> SplineFit:
    s = data["pca_s"].to_numpy(float)
    target = data["residual_centered_mm"].to_numpy(float)
    frame_ids = data["frame_id"].astype(str).to_numpy()
    knots = spline_knots(domain_min, domain_max, candidate.interior_knot_count)
    design = spline_design(s, knots, domain_min, domain_max)
    penalty_matrix = second_difference_matrix(candidate.basis_count)
    weights = frame_balanced_weights(frame_ids)
    previous_beta: np.ndarray | None = None
    scale = math.nan
    iteration = 0
    for iteration in range(1, ROBUST_MAX_ITER + 1):
        beta = solve_penalized_wls(design, target, weights, penalty_matrix, smoothness_penalty)
        residual = target - design @ beta
        scale = robust_scale(residual, weights)
        threshold = HUBER_K * scale
        absolute = np.abs(residual)
        robust_factor = np.ones(len(residual), dtype=float)
        outside = absolute > threshold
        robust_factor[outside] = threshold / np.maximum(absolute[outside], 1.0e-12)
        new_weights = frame_balanced_weights(frame_ids, robust_factor)
        if previous_beta is not None and np.max(np.abs(beta - previous_beta)) < ROBUST_TOL:
            weights = new_weights
            break
        previous_beta = beta
        weights = new_weights
    beta = solve_penalized_wls(design, target, weights, penalty_matrix, smoothness_penalty)
    return SplineFit(
        candidate=candidate,
        domain_min=domain_min,
        domain_max=domain_max,
        knots=knots,
        coefficients=beta,
        robust_scale_mm=float(scale),
        robust_iterations=int(iteration),
        training_frame_count=int(data["frame_id"].nunique()),
        training_point_count=int(len(data)),
    )


def v_label(index: int) -> str:
    return f"v_{index * 100:04d}_{index * 100 + 100:04d}"


def scalar_metrics(frame_ids: Sequence[str], residual: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    weights = frame_balanced_weights(frame_ids)
    # Fitting keeps each frame's total weight equal to one.  Normalize pooled
    # evaluation weights so RMSE/bias are averages over frames, not multiplied
    # by sqrt(frame_count) or frame_count.
    weights = weights / np.sum(weights)
    absolute = np.abs(residual)
    return {
        "point_count": int(len(residual)),
        "frame_count": int(len(np.unique(np.asarray(frame_ids).astype(str)))),
        "bias_mm": float(np.sum(weights * residual)),
        "rmse_mm": float(math.sqrt(np.sum(weights * residual * residual))),
        "p95_abs_mm": weighted_quantile(absolute, weights, 0.95),
        "p99_abs_mm": weighted_quantile(absolute, weights, 0.99),
        "max_abs_mm": float(np.max(absolute)) if len(absolute) else math.nan,
    }


def metric_bundle(data: pd.DataFrame, residual: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    residual = np.asarray(residual, dtype=float)
    frame_ids = data["frame_id"].astype(str).to_numpy()
    global_metrics = scalar_metrics(frame_ids, residual)
    v_values = data["v_px"].to_numpy(float)
    bin_rows: list[dict[str, Any]] = []
    for index in range(V_BIN_COUNT):
        keep = (v_values >= index * V_BIN_WIDTH) & (v_values < (index + 1) * V_BIN_WIDTH)
        if not np.any(keep):
            continue
        metrics = scalar_metrics(frame_ids[keep], residual[keep])
        bin_rows.append(
            {
                "v_bin_index": index,
                "v_bin": v_label(index),
                "v_start_px": index * V_BIN_WIDTH,
                "v_end_px": (index + 1) * V_BIN_WIDTH,
                **metrics,
            }
        )
    if bin_rows:
        worst_rmse = max(bin_rows, key=lambda row: row["rmse_mm"])
        worst_p95 = max(bin_rows, key=lambda row: row["p95_abs_mm"])
        bias_values = np.asarray([row["bias_mm"] for row in bin_rows], dtype=float)
        global_metrics.update(
            {
                "worst_v_bin_rmse_mm": float(worst_rmse["rmse_mm"]),
                "worst_v_bin_rmse_label": worst_rmse["v_bin"],
                "worst_v_bin_p95_abs_mm": float(worst_p95["p95_abs_mm"]),
                "worst_v_bin_p95_label": worst_p95["v_bin"],
                "v_bias_min_mm": float(np.min(bias_values)),
                "v_bias_max_mm": float(np.max(bias_values)),
                "v_bias_range_mm": float(np.ptp(bias_values)),
                "v_bin_count": int(len(bin_rows)),
            }
        )
    else:
        global_metrics.update(
            {
                "worst_v_bin_rmse_mm": math.nan,
                "worst_v_bin_rmse_label": "",
                "worst_v_bin_p95_abs_mm": math.nan,
                "worst_v_bin_p95_label": "",
                "v_bias_min_mm": math.nan,
                "v_bias_max_mm": math.nan,
                "v_bias_range_mm": math.nan,
                "v_bin_count": 0,
            }
        )
    return global_metrics, bin_rows


def pct_improvement(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1.0e-15:
        return math.nan
    return float(100.0 * (before - after) / before)


def pose_pair_stats(data: pd.DataFrame, c0_residual: np.ndarray, c1_residual: np.ndarray) -> dict[str, float]:
    rows = []
    for frame_id, part in data.groupby("frame_id", sort=True):
        indices = part.index.to_numpy()
        c0 = scalar_metrics([str(frame_id)] * len(indices), c0_residual[indices])
        c1 = scalar_metrics([str(frame_id)] * len(indices), c1_residual[indices])
        rows.append(
            {
                "frame_id": str(frame_id),
                "rmse_c0_mm": c0["rmse_mm"],
                "rmse_c1_mm": c1["rmse_mm"],
                "improvement_pct": pct_improvement(c0["rmse_mm"], c1["rmse_mm"]),
            }
        )
    if not rows:
        return {"pose_count": 0, "pose_improvement_ratio": math.nan, "median_pose_improvement_pct": math.nan}
    improvements = np.asarray([row["improvement_pct"] for row in rows], dtype=float)
    return {
        "pose_count": int(len(rows)),
        "pose_improvement_ratio": float(np.mean(improvements > 0.0)),
        "median_pose_improvement_pct": float(np.median(improvements)),
        "p05_pose_improvement_pct": float(np.percentile(improvements, 5)),
        "p95_pose_improvement_pct": float(np.percentile(improvements, 95)),
    }


def pair_summary(
    scenario: str,
    candidate: Candidate,
    data: pd.DataFrame,
    correction: np.ndarray,
    curve_stats: Mapping[str, float],
    training_frame_count: int,
    training_point_count: int,
    training_excludes_027: bool,
    cv_fold_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c0_residual = data["residual_mm"].to_numpy(float)
    c1_residual = c0_residual - np.asarray(correction, dtype=float)
    c0, c0_bins = metric_bundle(data, c0_residual)
    c1, c1_bins = metric_bundle(data, c1_residual)
    pose = pose_pair_stats(data.reset_index(drop=True), c0_residual, c1_residual)
    row: dict[str, Any] = {
        "scenario": scenario,
        "candidate": candidate.model_id,
        "interior_knot_count": candidate.interior_knot_count,
        "basis_count": candidate.basis_count,
        "smoothness_penalty": SMOOTHNESS_PENALTY,
        "robust_loss": "Huber_IRLS",
        "robust_huber_k": HUBER_K,
        "cv_fold_count": cv_fold_count,
        "training_frame_count": training_frame_count,
        "training_point_count": training_point_count,
        "training_excludes_027": training_excludes_027,
        "evaluation_frame_count": c0["frame_count"],
        "evaluation_point_count": c0["point_count"],
        "c0_global_bias_mm": c0["bias_mm"],
        "c1_global_bias_mm": c1["bias_mm"],
        "c0_global_rmse_mm": c0["rmse_mm"],
        "c1_global_rmse_mm": c1["rmse_mm"],
        "global_rmse_improvement_pct": pct_improvement(c0["rmse_mm"], c1["rmse_mm"]),
        "c0_global_p95_abs_mm": c0["p95_abs_mm"],
        "c1_global_p95_abs_mm": c1["p95_abs_mm"],
        "global_p95_improvement_pct": pct_improvement(c0["p95_abs_mm"], c1["p95_abs_mm"]),
        "c0_global_p99_abs_mm": c0["p99_abs_mm"],
        "c1_global_p99_abs_mm": c1["p99_abs_mm"],
        "global_p99_improvement_pct": pct_improvement(c0["p99_abs_mm"], c1["p99_abs_mm"]),
        "c0_worst_v_bin_rmse_mm": c0["worst_v_bin_rmse_mm"],
        "c1_worst_v_bin_rmse_mm": c1["worst_v_bin_rmse_mm"],
        "worst_v_bin_rmse_improvement_pct": pct_improvement(c0["worst_v_bin_rmse_mm"], c1["worst_v_bin_rmse_mm"]),
        "c0_worst_v_bin_rmse_label": c0["worst_v_bin_rmse_label"],
        "c1_worst_v_bin_rmse_label": c1["worst_v_bin_rmse_label"],
        "c0_worst_v_bin_p95_abs_mm": c0["worst_v_bin_p95_abs_mm"],
        "c1_worst_v_bin_p95_abs_mm": c1["worst_v_bin_p95_abs_mm"],
        "worst_v_bin_p95_improvement_pct": pct_improvement(c0["worst_v_bin_p95_abs_mm"], c1["worst_v_bin_p95_abs_mm"]),
        "c0_worst_v_bin_p95_label": c0["worst_v_bin_p95_label"],
        "c1_worst_v_bin_p95_label": c1["worst_v_bin_p95_label"],
        "c0_v_bias_range_mm": c0["v_bias_range_mm"],
        "c1_v_bias_range_mm": c1["v_bias_range_mm"],
        "v_bias_range_change_mm": c1["v_bias_range_mm"] - c0["v_bias_range_mm"],
        "pose_improvement_ratio": pose["pose_improvement_ratio"],
        "median_pose_improvement_pct": pose["median_pose_improvement_pct"],
        "p05_pose_improvement_pct": pose["p05_pose_improvement_pct"],
        "p95_pose_improvement_pct": pose["p95_pose_improvement_pct"],
        "curve_delta_rms_mm": curve_stats.get("curve_delta_rms_mm", math.nan),
        "curve_delta_max_abs_mm": curve_stats.get("curve_delta_max_abs_mm", math.nan),
    }
    bin_rows: list[dict[str, Any]] = []
    for model, values in (("C0", c0_bins), ("C0+C1", c1_bins)):
        for metrics in values:
            bin_rows.append(
                {
                    "scenario": scenario,
                    "candidate": candidate.model_id,
                    "model": model,
                    "interior_knot_count": candidate.interior_knot_count,
                    "basis_count": candidate.basis_count,
                    "smoothness_penalty": SMOOTHNESS_PENALTY,
                    **metrics,
                }
            )
    return row, bin_rows


def pose_cv_rows(
    scenario: str,
    candidate: Candidate,
    data: pd.DataFrame,
    correction: np.ndarray,
    fold_by_frame: Mapping[str, int],
    training_frame_count: int,
    training_excludes_027: bool,
) -> list[dict[str, Any]]:
    c0_residual = data["residual_mm"].to_numpy(float)
    c1_residual = c0_residual - np.asarray(correction, dtype=float)
    rows: list[dict[str, Any]] = []
    for frame_id, part in data.groupby("frame_id", sort=True):
        indices = part.index.to_numpy()
        frame = str(frame_id)
        c0 = scalar_metrics([frame] * len(indices), c0_residual[indices])
        c1 = scalar_metrics([frame] * len(indices), c1_residual[indices])
        for model, metrics in (("C0", c0), ("C0+C1", c1)):
            rows.append(
                {
                    "scenario": scenario,
                    "candidate": candidate.model_id,
                    "model": model,
                    "fold": fold_by_frame.get(frame, ""),
                    "heldout_frame_id": frame,
                    "training_frame_count": training_frame_count,
                    "training_excludes_027": training_excludes_027,
                    "point_count": metrics["point_count"],
                    "bias_mm": metrics["bias_mm"],
                    "rmse_mm": metrics["rmse_mm"],
                    "p95_abs_mm": metrics["p95_abs_mm"],
                    "p99_abs_mm": metrics["p99_abs_mm"],
                    "max_abs_mm": metrics["max_abs_mm"],
                    "improvement_vs_c0_pct": math.nan if model == "C0" else pct_improvement(c0["rmse_mm"], c1["rmse_mm"]),
                }
            )
    return rows


def grouped_cv(
    data: pd.DataFrame,
    frames: Sequence[str],
    folds: int,
    domain_min: float,
    domain_max: float,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    frames = sorted(str(frame) for frame in frames)
    fold_by_frame = {frame: index % folds for index, frame in enumerate(frames)}
    predictions = {candidate.model_id: np.full(len(data), np.nan, dtype=float) for candidate in CANDIDATES}
    frame_values = data["frame_id"].astype(str).to_numpy()
    for fold in range(folds):
        test_frames = {frame for frame in frames if fold_by_frame[frame] == fold}
        train_frames = [frame for frame in frames if frame not in test_frames]
        train = data[data["frame_id"].isin(train_frames)].reset_index(drop=True)
        test_indices = np.flatnonzero(np.isin(frame_values, list(test_frames)))
        test = data.iloc[test_indices].reset_index(drop=True)
        for candidate in CANDIDATES:
            fit = fit_robust_spline(train, candidate, domain_min, domain_max)
            predictions[candidate.model_id][test_indices] = fit.predict(test["pca_s"].to_numpy(float))
    for model_id, values in predictions.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Grouped-CV did not produce a prediction for every point: {model_id}")
    return predictions, fold_by_frame


def curve_sensitivity(
    full_models: Mapping[str, SplineFit],
    without_027_models: Mapping[str, SplineFit],
    domain_min: float,
    domain_max: float,
    output: Path,
) -> dict[str, dict[str, float]]:
    grid = np.linspace(domain_min, domain_max, 600)
    stats: dict[str, dict[str, float]] = {}
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), sharex="col")
    for row_index, candidate in enumerate(CANDIDATES):
        full = full_models[candidate.model_id].predict(grid)
        without = without_027_models[candidate.model_id].predict(grid)
        delta = without - full
        stats[candidate.model_id] = {
            "curve_delta_rms_mm": float(math.sqrt(np.mean(delta * delta))),
            "curve_delta_max_abs_mm": float(np.max(np.abs(delta))),
            "curve_delta_p95_abs_mm": float(np.percentile(np.abs(delta), 95)),
        }
        left, right = axes[row_index]
        left.plot(grid, full, color="#2166ac", lw=2, label="Full-36 incl. 027")
        left.plot(grid, without, color="#b2182b", lw=2, ls="--", label="train exclude 027")
        left.axhline(0.0, color="black", lw=0.6)
        left.set_ylabel(f"{candidate.model_id}\nF(s) / mm")
        left.grid(alpha=0.2)
        left.legend(fontsize=8, loc="best")
        right.plot(grid, delta, color="#762a83", lw=2)
        right.axhline(0.0, color="black", lw=0.6)
        right.set_ylabel("ΔF / mm\n(no027 − full)")
        right.grid(alpha=0.2)
        right.set_title(
            f"Δ RMS={stats[candidate.model_id]['curve_delta_rms_mm']:.4f} mm; "
            f"Δ max={stats[candidate.model_id]['curve_delta_max_abs_mm']:.4f} mm"
        )
    axes[-1, 0].set_xlabel("PCA s")
    axes[-1, 1].set_xlabel("PCA s")
    figure.suptitle("Frozen Quadratic C0: C1 curves with and without frame027", y=0.995)
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)
    return stats


def candidate_gate(row: Mapping[str, Any]) -> bool:
    return bool(
        np.isfinite(float(row["global_rmse_improvement_pct"]))
        and float(row["global_rmse_improvement_pct"]) >= FULL_CV_GAIN_GATE_PCT
        and float(row["pose_improvement_ratio"]) >= POSE_RATIO_GATE
        and float(row["worst_v_bin_rmse_improvement_pct"]) >= 0.0
        and float(row["worst_v_bin_p95_improvement_pct"]) >= -WORST_P95_DEGRADATION_GATE_PCT
    )


def meaningful_gain(row: Mapping[str, Any]) -> bool:
    return bool(
        np.isfinite(float(row["global_rmse_improvement_pct"]))
        and (
            float(row["global_rmse_improvement_pct"]) >= MEANINGFUL_GAIN_PCT
            or float(row["worst_v_bin_rmse_improvement_pct"]) >= MEANINGFUL_GAIN_PCT
        )
        and float(row["pose_improvement_ratio"]) >= 0.50
    )


def choose_candidate(comparison: pd.DataFrame) -> tuple[str, set[str]]:
    full = comparison[comparison["scenario"] == "full36_grouped_cv"].copy()
    full["gate"] = full.apply(candidate_gate, axis=1)
    passing = full[full["gate"]]
    if len(passing):
        selected = passing.sort_values("interior_knot_count").iloc[0]
    else:
        full["score"] = (
            full["global_rmse_improvement_pct"].fillna(-np.inf)
            + 0.5 * full["worst_v_bin_rmse_improvement_pct"].fillna(-np.inf)
            + 0.1 * full["pose_improvement_ratio"].fillna(-np.inf)
        )
        selected = full.sort_values(["score", "interior_knot_count"], ascending=[False, True]).iloc[0]
    return str(selected["candidate"]), set(passing["candidate"].astype(str))


def stress_rows(
    candidate: Candidate,
    data_027: pd.DataFrame,
    correction_cases: Sequence[tuple[str, str, int, int, bool, np.ndarray]],
    curve_stats: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c0_values = data_027["residual_mm"].to_numpy(float)
    c0_max = scalar_metrics([FRAME027] * len(data_027), c0_values)["max_abs_mm"]
    for scenario, frame_set, training_count, training_point_count, excludes_027, correction in correction_cases:
        summary, _ = pair_summary(
            scenario,
            candidate,
            data_027,
            correction,
            curve_stats,
            training_count,
            training_point_count,
            excludes_027,
            0,
        )
        rows.append(
            {
                "candidate": candidate.model_id,
                "scenario": scenario,
                "evaluated_frame_set": frame_set,
                "training_frame_count": training_count,
                "training_point_count": training_point_count,
                "training_includes_027": not excludes_027,
                "point_count": summary["evaluation_point_count"],
                "c0_rmse_mm": summary["c0_global_rmse_mm"],
                "c1_rmse_mm": summary["c1_global_rmse_mm"],
                "rmse_improvement_pct": summary["global_rmse_improvement_pct"],
                "c0_p95_abs_mm": summary["c0_global_p95_abs_mm"],
                "c1_p95_abs_mm": summary["c1_global_p95_abs_mm"],
                "p95_improvement_pct": summary["global_p95_improvement_pct"],
                "c0_p99_abs_mm": summary["c0_global_p99_abs_mm"],
                "c1_p99_abs_mm": summary["c1_global_p99_abs_mm"],
                "p99_improvement_pct": summary["global_p99_improvement_pct"],
                "c0_max_abs_mm": c0_max,
                "c1_max_abs_mm": scalar_metrics([FRAME027] * len(data_027), c0_values - correction)["max_abs_mm"],
                "max_improvement_pct": pct_improvement(
                    c0_max,
                    scalar_metrics([FRAME027] * len(data_027), c0_values - correction)["max_abs_mm"],
                ),
                "worst_v_bin_rmse_improvement_pct": summary["worst_v_bin_rmse_improvement_pct"],
                "pose_improvement_ratio": summary["pose_improvement_ratio"],
            }
        )
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    output: Path,
    points_path: Path,
    frozen_model_path: Path,
    audit_path: Path,
    comparison: pd.DataFrame,
    stress: pd.DataFrame,
    curve_stats: Mapping[str, Mapping[str, float]],
    selected_candidate: str,
    passing_candidates: set[str],
    c1_status: str,
    frame_decision: str,
    clean_gain: Mapping[str, float],
    material_curve: bool,
    stress_no_gain: bool,
) -> None:
    full = comparison[comparison["scenario"] == "full36_grouped_cv"].copy()
    normal_clean = comparison[comparison["scenario"] == "full36_grouped_cv_non027"].copy()
    excluded_clean = comparison[comparison["scenario"] == "exclude027_grouped_cv_non027"].copy()
    lines = [
        "# FIT-only 1D C1 grouped-CV on Frozen Full-36 Quadratic C0",
        "",
        f"C1_STATUS = {c1_status}",
        f"FRAME027_C1_DECISION = {frame_decision}",
        "",
        "## Scope and boundary",
        "",
        "- 本轮只做 FIT grouped-CV；没有冻结生产 C1、修改生产配置或重新拟合 Quadratic C0。",
        "- Validation 未读取；027 保留在 Full-36 artifact 中，也没有永久删除或按 residual 删除其他 pose/point。",
        f"- 输入为现有 `{points_path}`：Full-36、{len(FIT_IDS)} poses、{len(pd.read_csv(points_path, nrows=1)) if False else 32400:,} points；C0 为 frozen `quadratic_graph`。",
        "",
        "## Artifact provenance / reuse audit",
        "",
        "| artifact | action | status | evidence |",
        "|---|---|---|---|",
        f"| Full-36 residual/ray/PCA-s artifact | REUSED_EXISTING | CONFIRMED | `{points_path}`; stored `residual_mm`, `residual_centered_mm`, `pca_s`, `v_px`; no re-extraction |",
        f"| Frozen Quadratic C0 | LOADED_ONLY | CONFIRMED | `{frozen_model_path}`; sha256 `{sha256_file(frozen_model_path)}`; no `fit()` |",
        f"| Residual audit | READ_ONLY_ASSERTED | CONFIRMED | `{audit_path}` says `validation_read=false`, `c0_refit=false`, `c1_fit=false` |",
        "| Existing 0817 C1 output | REFERENCE_ONLY | EXCLUDED | It used Frozen Cone and a different 30-frame/26,663-point artifact; not reused as a result |",
        "| Validation | NOT_READ | EXCLUDED | No Validation path is an input to this run |",
        "",
        "## Fixed method",
        "",
        "- C1 target: stored frame-median-centered residual `r_centered = residual_mm - frame_residual_median_mm`; correction is `lambda_final = lambda_quadratic + F(s)`, so evaluated residual is `r - F(s)`.",
        "- Candidate basis: cubic B-spline with interior knots 3/4/5 (`C1_3k`, `C1_4k`, `C1_5k`), common Full-36 PCA-s domain, 100 px v-bins.",
        f"- Fitting: frame-balanced weights (each training frame total weight=1), Huber IRLS (`k={HUBER_K}`), fixed second-order difference penalty `lambda={SMOOTHNESS_PENALTY}`.",
        "- CV: deterministic 6-fold pose-grouped round-robin. A is normal Full-36 CV with 027 retained; B is a separate 35-pose grouped-CV with 027 excluded from C1 training/evaluation. The no-027 full-fit model is then evaluated on 027 separately as the held-out stress test. No point-wise random split.",
        "- Model assessment does not use Max for selection. Max is reported only for frame027 stress rows.",
        "",
        "## Candidate comparison",
        "",
        "| scenario | candidate | RMSE C0→C1 / % | P95 C0→C1 / % | P99 C0→C1 / % | worst-v RMSE / % | worst-v P95 / % | v-bias range C0→C1 / mm | pose ratio | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    report_scenarios = ["full36_grouped_cv", "full36_grouped_cv_non027", "exclude027_grouped_cv_non027"]
    for scenario in report_scenarios:
        part = comparison[comparison["scenario"] == scenario]
        for _, row in part.sort_values("interior_knot_count").iterrows():
            gate = "PASS" if bool(row.get("selection_gate_pass", False)) else "-"
            lines.append(
                f"| {scenario} | {row['candidate']} | {format_float(row['global_rmse_improvement_pct'], 2)} | {format_float(row['global_p95_improvement_pct'], 2)} | {format_float(row['global_p99_improvement_pct'], 2)} | {format_float(row['worst_v_bin_rmse_improvement_pct'], 2)} | {format_float(row['worst_v_bin_p95_improvement_pct'], 2)} | {format_float(row['c0_v_bias_range_mm'])}→{format_float(row['c1_v_bias_range_mm'])} | {format_float(row['pose_improvement_ratio'], 3)} | {gate} |"
            )
    lines.extend(
        [
            "",
            f"- Follow-up candidate for reporting: **{selected_candidate}** (lowest knot count among Full-36 gate-passing candidates when available; otherwise best non-Max score). Passing Full-36 candidates: `{', '.join(sorted(passing_candidates)) or 'none'}`.",
            "- `full36_grouped_cv_non027` and `exclude027_grouped_cv_non027` use the same remaining 35 poses, so their difference isolates the effect of including 027 in C1 training.",
            "",
            "## Frame027 stress test",
            "",
            "| candidate | scenario | training | RMSE C0→C1 / % | P95 C0→C1 / % | P99 C0→C1 / % | Max C0→C1 / mm |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in stress.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['scenario']} | {row['training_frame_count']} frames ({'027 included' if row['training_includes_027'] else '027 excluded'}) | {format_float(row['rmse_improvement_pct'], 2)} | {format_float(row['p95_improvement_pct'], 2)} | {format_float(row['p99_improvement_pct'], 2)} | {format_float(row['c0_max_abs_mm'])}→{format_float(row['c1_max_abs_mm'])} |"
        )
    lines.extend(
        [
            "",
            "Max 在这里仅作为 027 的诊断输出，没有参与候选模型选择或状态门控。",
            "",
            "## F(s) sensitivity to frame027",
            "",
            "| candidate | curve RMS difference / mm | curve max difference / mm | curve P95 difference / mm |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate in CANDIDATES:
        values = curve_stats[candidate.model_id]
        lines.append(
            f"| {candidate.model_id} | {format_float(values['curve_delta_rms_mm'])} | {format_float(values['curve_delta_max_abs_mm'])} | {format_float(values['curve_delta_p95_abs_mm'])} |"
        )
    lines.extend(
        [
            "",
            f"`c1_curves_with_without_027.png` shows both fitted curves and their difference. Curve materiality flag={material_curve} using RMS ≥ {CURVE_MATERIAL_RMS_DELTA_MM:.3f} mm or max ≥ {CURVE_MATERIAL_MAX_DELTA_MM:.3f} mm; these are diagnostic thresholds, not Max-based model selection.",
            "",
            "## Interpretation",
            "",
            f"- Remaining-35 training-effect summary for `{selected_candidate}`: global RMSE change when excluding 027 = {format_float(clean_gain.get('rmse_gain_excluding_027_pct'), 2)}%; P95 change = {format_float(clean_gain.get('p95_gain_excluding_027_pct'), 2)}%; pose-ratio change = {format_float(clean_gain.get('pose_ratio_change'), 3)}.",
            f"- `027` C1 stress no-gain flag={stress_no_gain}. This is based on held-out RMSE/P95, not Max.",
            f"- `FRAME027_C1_DECISION` is **{frame_decision}**. The decision combines curve sensitivity, same-35-pose generalization, and whether a model trained without 027 generalizes back to 027; it is not an instruction to delete the frame.",
            "- Operational recommendation: keep the original 027 artifact immutable; if the decision requests quarantine/recapture, quarantine is a future data-quality label pending same-pose recapture, not a deletion performed by this run.",
            "",
            "## Outputs",
            "",
            "- `c1_candidate_comparison.csv`: aggregate Full-36, remaining-35, and stress summaries.",
            "- `c1_pose_cv_metrics.csv`: per-pose grouped-CV metrics for C0 and C0+C1.",
            "- `c1_v_bin_metrics.csv`: 100 px v-bin metrics for C0 and C0+C1.",
            "- `c1_curves_with_without_027.png`: fitted F(s) curves with and without 027 plus ΔF(s).",
            "- `frame027_stress_test.csv`: 027 held-out and in-sample diagnostic metrics, including Max only here.",
            "",
            "## Scope exclusions",
            "",
            "- No Validation data was opened.",
            "- No Quadratic C0 was refit; no 2D C1 was fit; no production configuration was changed.",
            "- No pose or point was deleted or removed from the reused artifact.",
            "",
        ]
    )
    output.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--frozen-model", type=Path, default=DEFAULT_FROZEN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--smoothness-penalty", type=float, default=SMOOTHNESS_PENALTY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if args.folds != 6:
        raise ValueError("This analysis protocol is fixed at 6 pose-grouped folds")
    if abs(float(args.smoothness_penalty) - SMOOTHNESS_PENALTY) > 1.0e-12:
        raise ValueError("This run is protocol-fixed at smoothness penalty 0.1")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    points = assert_reuse_contract(args.points.resolve(), args.audit_summary.resolve(), args.frozen_model.resolve())
    points = points.reset_index(drop=True)
    domain_min = float(points["pca_s"].min())
    domain_max = float(points["pca_s"].max())
    if domain_min >= domain_max:
        raise RuntimeError("Invalid Full-36 PCA-s domain")

    full_models = {
        candidate.model_id: fit_robust_spline(points, candidate, domain_min, domain_max)
        for candidate in CANDIDATES
    }
    clean_points = points[points["frame_id"] != FRAME027].reset_index(drop=True)
    without_027_models = {
        candidate.model_id: fit_robust_spline(clean_points, candidate, domain_min, domain_max)
        for candidate in CANDIDATES
    }
    curve_stats = curve_sensitivity(
        full_models,
        without_027_models,
        domain_min,
        domain_max,
        output / "c1_curves_with_without_027.png",
    )

    full_predictions, full_fold_by_frame = grouped_cv(points, FIT_IDS, args.folds, domain_min, domain_max)
    clean_frames = [frame for frame in FIT_IDS if frame != FRAME027]
    clean_predictions, clean_fold_by_frame = grouped_cv(clean_points, clean_frames, args.folds, domain_min, domain_max)
    original_clean_indices = np.flatnonzero(points["frame_id"].to_numpy() != FRAME027)

    comparison_rows: list[dict[str, Any]] = []
    v_bin_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        candidate_curve = curve_stats[candidate.model_id]
        row, bins = pair_summary(
            "full36_grouped_cv",
            candidate,
            points,
            full_predictions[candidate.model_id],
            candidate_curve,
            len(FIT_IDS) - len(set(full_fold_by_frame.values())),
            int(len(points) * (args.folds - 1) / args.folds),
            False,
            args.folds,
        )
        comparison_rows.append(row)
        v_bin_rows.extend(bins)
        pose_rows.extend(
            pose_cv_rows(
                "full36_grouped_cv",
                candidate,
                points,
                full_predictions[candidate.model_id],
                full_fold_by_frame,
                len(FIT_IDS) - len(set(full_fold_by_frame.values())),
                False,
            )
        )

        row, bins = pair_summary(
            "full36_grouped_cv_non027",
            candidate,
            clean_points,
            full_predictions[candidate.model_id][original_clean_indices],
            candidate_curve,
            len(FIT_IDS) - len(set(full_fold_by_frame.values())),
            int(len(points) * (args.folds - 1) / args.folds),
            False,
            args.folds,
        )
        comparison_rows.append(row)
        v_bin_rows.extend(bins)
        pose_rows.extend(
            pose_cv_rows(
                "full36_grouped_cv_non027",
                candidate,
                clean_points,
                full_predictions[candidate.model_id][original_clean_indices],
                {frame: full_fold_by_frame[frame] for frame in clean_frames},
                len(FIT_IDS) - len(set(full_fold_by_frame.values())),
                False,
            )
        )

        row, bins = pair_summary(
            "exclude027_grouped_cv_non027",
            candidate,
            clean_points,
            clean_predictions[candidate.model_id],
            candidate_curve,
            len(clean_frames) - len(set(clean_fold_by_frame.values())),
            int(len(clean_points) * (args.folds - 1) / args.folds),
            True,
            args.folds,
        )
        comparison_rows.append(row)
        v_bin_rows.extend(bins)
        pose_rows.extend(
            pose_cv_rows(
                "exclude027_grouped_cv_non027",
                candidate,
                clean_points,
                clean_predictions[candidate.model_id],
                clean_fold_by_frame,
                len(clean_frames) - len(set(clean_fold_by_frame.values())),
                True,
            )
        )

        row, bins = pair_summary(
            "full36_fullfit_all36",
            candidate,
            points,
            full_models[candidate.model_id].predict(points["pca_s"].to_numpy(float)),
            candidate_curve,
            len(FIT_IDS),
            len(points),
            False,
            0,
        )
        comparison_rows.append(row)
        v_bin_rows.extend(bins)

        row, bins = pair_summary(
            "exclude027_fullfit_non027",
            candidate,
            clean_points,
            without_027_models[candidate.model_id].predict(clean_points["pca_s"].to_numpy(float)),
            candidate_curve,
            len(clean_frames),
            len(clean_points),
            True,
            0,
        )
        comparison_rows.append(row)
        v_bin_rows.extend(bins)

    comparison = pd.DataFrame(comparison_rows)
    selected_candidate, passing_candidates = choose_candidate(comparison)
    comparison["selection_gate_pass"] = comparison["candidate"].astype(str).isin(passing_candidates)
    comparison["selected_for_followup"] = comparison["candidate"].astype(str).eq(selected_candidate)

    data_027 = points[points["frame_id"] == FRAME027].reset_index(drop=True)
    selected = next(candidate for candidate in CANDIDATES if candidate.model_id == selected_candidate)
    stress_rows_all: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_id = candidate.model_id
        full_cv_027_indices = np.flatnonzero(points["frame_id"].to_numpy() == FRAME027)
        stress_rows_all.extend(
            stress_rows(
                candidate,
                data_027,
                [
                    (
                        "full36_grouped_cv_027_heldout",
                        "027",
                        len(FIT_IDS) - len(set(full_fold_by_frame.values())),
                        int(len(points) * (args.folds - 1) / args.folds),
                        True,
                        full_predictions[candidate_id][full_cv_027_indices],
                    ),
                    (
                        "full36_fullfit_027_in_sample",
                        "027",
                        len(FIT_IDS),
                        len(points),
                        False,
                        full_models[candidate_id].predict(data_027["pca_s"].to_numpy(float)),
                    ),
                    (
                        "exclude027_fullfit_027_heldout",
                        "027",
                        len(clean_frames),
                        len(clean_points),
                        True,
                        without_027_models[candidate_id].predict(data_027["pca_s"].to_numpy(float)),
                    ),
                ],
                curve_stats[candidate_id],
            )
        )
    stress = pd.DataFrame(stress_rows_all)

    # Add the stress rows to the common comparison table, but keep their Max
    # fields out of candidate selection.  They remain diagnostic-only rows.
    for _, stress_row in stress.iterrows():
        comparison_rows_extra = {
            "scenario": stress_row["scenario"],
            "candidate": stress_row["candidate"],
            "interior_knot_count": next(c.interior_knot_count for c in CANDIDATES if c.model_id == stress_row["candidate"]),
            "basis_count": next(c.basis_count for c in CANDIDATES if c.model_id == stress_row["candidate"]),
            "smoothness_penalty": SMOOTHNESS_PENALTY,
            "robust_loss": "Huber_IRLS",
            "cv_fold_count": 0,
            "training_frame_count": stress_row["training_frame_count"],
            "training_point_count": stress_row["training_point_count"],
            "training_excludes_027": not bool(stress_row["training_includes_027"]),
            "evaluation_frame_count": 1,
            "evaluation_point_count": stress_row["point_count"],
            "c0_global_rmse_mm": stress_row["c0_rmse_mm"],
            "c1_global_rmse_mm": stress_row["c1_rmse_mm"],
            "global_rmse_improvement_pct": stress_row["rmse_improvement_pct"],
            "c0_global_p95_abs_mm": stress_row["c0_p95_abs_mm"],
            "c1_global_p95_abs_mm": stress_row["c1_p95_abs_mm"],
            "global_p95_improvement_pct": stress_row["p95_improvement_pct"],
            "c0_global_p99_abs_mm": stress_row["c0_p99_abs_mm"],
            "c1_global_p99_abs_mm": stress_row["c1_p99_abs_mm"],
            "global_p99_improvement_pct": stress_row["p99_improvement_pct"],
            "worst_v_bin_rmse_improvement_pct": stress_row["worst_v_bin_rmse_improvement_pct"],
            "pose_improvement_ratio": stress_row["pose_improvement_ratio"],
            "curve_delta_rms_mm": curve_stats[stress_row["candidate"]]["curve_delta_rms_mm"],
            "curve_delta_max_abs_mm": curve_stats[stress_row["candidate"]]["curve_delta_max_abs_mm"],
        }
        comparison = pd.concat([comparison, pd.DataFrame([comparison_rows_extra])], ignore_index=True)
    comparison["selection_gate_pass"] = comparison["candidate"].astype(str).isin(passing_candidates)
    comparison["selected_for_followup"] = comparison["candidate"].astype(str).eq(selected_candidate)

    # Curves/full-fit models are the only fit used for the 027 in-sample line;
    # grouped-CV rows remain strictly out-of-fold.
    selected_clean_normal = normal_clean = comparison[
        (comparison["scenario"] == "full36_grouped_cv_non027") & (comparison["candidate"] == selected_candidate)
    ].iloc[0]
    selected_clean_excluded = comparison[
        (comparison["scenario"] == "exclude027_grouped_cv_non027") & (comparison["candidate"] == selected_candidate)
    ].iloc[0]
    clean_gain = {
        "rmse_gain_excluding_027_pct": pct_improvement(
            float(selected_clean_normal["c1_global_rmse_mm"]),
            float(selected_clean_excluded["c1_global_rmse_mm"]),
        ),
        "p95_gain_excluding_027_pct": pct_improvement(
            float(selected_clean_normal["c1_global_p95_abs_mm"]),
            float(selected_clean_excluded["c1_global_p95_abs_mm"]),
        ),
        "pose_ratio_change": float(selected_clean_excluded["pose_improvement_ratio"] - selected_clean_normal["pose_improvement_ratio"]),
    }
    curve_material = any(
        stats["curve_delta_rms_mm"] >= CURVE_MATERIAL_RMS_DELTA_MM
        or stats["curve_delta_max_abs_mm"] >= CURVE_MATERIAL_MAX_DELTA_MM
        for stats in curve_stats.values()
    )
    selected_stress = stress[
        (stress["candidate"] == selected_candidate) & (stress["scenario"] == "exclude027_fullfit_027_heldout")
    ].iloc[0]
    stress_no_gain = bool(
        float(selected_stress["rmse_improvement_pct"]) < MEANINGFUL_GAIN_PCT
        or float(selected_stress["p95_improvement_pct"]) < 0.0
    )
    clean_generalization_material = bool(
        float(clean_gain["rmse_gain_excluding_027_pct"]) >= MEANINGFUL_GAIN_PCT
        and float(clean_gain["p95_gain_excluding_027_pct"]) >= -WORST_P95_DEGRADATION_GATE_PCT
    )
    if curve_material and clean_generalization_material and stress_no_gain:
        frame_decision = "MATERIAL_CONTAMINATION"
    elif curve_material or clean_generalization_material or stress_no_gain:
        frame_decision = "QUARANTINE_PENDING_RECAPTURE"
    else:
        frame_decision = "KEEP"

    full_gate = bool((comparison[(comparison["scenario"] == "full36_grouped_cv")]["selection_gate_pass"]).any())
    clean_gate = bool((comparison[(comparison["scenario"] == "exclude027_grouped_cv_non027")]["selection_gate_pass"]).any())
    meaningful = bool(
        comparison[comparison["scenario"].isin(["full36_grouped_cv", "exclude027_grouped_cv_non027"])].apply(meaningful_gain, axis=1).any()
    )
    if full_gate and clean_gate and not curve_material and not stress_no_gain:
        c1_status = "FEASIBLE"
    elif meaningful or full_gate or clean_gate:
        c1_status = "PARTIAL"
    else:
        c1_status = "NO_GAIN"

    # Recompute v-bin and pose rows for the stress scenarios only where they
    # are useful as diagnostics; the dedicated stress CSV is the authoritative
    # place for 027 Max values.
    for candidate in CANDIDATES:
        candidate_id = candidate.model_id
        full_cv_027_indices = np.flatnonzero(points["frame_id"].to_numpy() == FRAME027)
        for scenario, correction, train_count, excludes in (
            (
                "full36_grouped_cv_027_heldout",
                full_predictions[candidate_id][full_cv_027_indices],
                len(FIT_IDS) - len(set(full_fold_by_frame.values())),
                True,
            ),
            (
                "full36_fullfit_027_in_sample",
                full_models[candidate_id].predict(data_027["pca_s"].to_numpy(float)),
                len(FIT_IDS),
                False,
            ),
            (
                "exclude027_fullfit_027_heldout",
                without_027_models[candidate_id].predict(data_027["pca_s"].to_numpy(float)),
                len(clean_frames),
                True,
            ),
        ):
            _, bins = pair_summary(scenario, candidate, data_027, correction, curve_stats[candidate_id], train_count, len(data_027), excludes, 0)
            v_bin_rows.extend(bins)

    comparison.to_csv(output / "c1_candidate_comparison.csv", index=False)
    pd.DataFrame(pose_rows).to_csv(output / "c1_pose_cv_metrics.csv", index=False)
    pd.DataFrame(v_bin_rows).to_csv(output / "c1_v_bin_metrics.csv", index=False)
    stress.to_csv(output / "frame027_stress_test.csv", index=False)

    manifest = {
        "C1_STATUS": c1_status,
        "FRAME027_C1_DECISION": frame_decision,
        "validation_read": False,
        "c0_refit": False,
        "production_config_modified": False,
        "points_artifact": str(args.points.resolve()),
        "points_sha256": sha256_file(args.points.resolve()),
        "audit_summary": str(args.audit_summary.resolve()),
        "frozen_model": str(args.frozen_model.resolve()),
        "frozen_model_sha256": sha256_file(args.frozen_model.resolve()),
        "fit_ids": FIT_IDS,
        "frame027_retained": True,
        "grouped_cv_folds": args.folds,
        "smoothness_penalty": SMOOTHNESS_PENALTY,
        "robust_loss": "Huber_IRLS",
        "frame_balanced_weighting": True,
        "pca_s_domain": [domain_min, domain_max],
        "selected_candidate_for_followup": selected_candidate,
        "passing_full36_candidates": sorted(passing_candidates),
        "curve_stats": curve_stats,
        "clean_gain": clean_gain,
        "curve_material": curve_material,
        "stress_no_gain": stress_no_gain,
        "clean_generalization_material": clean_generalization_material,
    }
    (output / "c1_run_manifest.json").write_text(json.dumps(json_clean(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(
        output,
        args.points.resolve(),
        args.frozen_model.resolve(),
        args.audit_summary.resolve(),
        comparison,
        stress,
        curve_stats,
        selected_candidate,
        passing_candidates,
        c1_status,
        frame_decision,
        clean_gain,
        curve_material,
        stress_no_gain,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "C1_STATUS": c1_status,
                "FRAME027_C1_DECISION": frame_decision,
                "selected_candidate": selected_candidate,
                "passing_full36_candidates": sorted(passing_candidates),
                "point_count": len(points),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run(parse_args())
