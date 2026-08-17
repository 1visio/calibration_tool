#!/usr/bin/env python3
"""FIT-only spatial observability audit for the frozen Circular Cone residual.

This diagnostic deliberately does not fit K/D, a Cone, or a correction.  It
opens only the explicitly declared FIT triplets (001--018 and 025--036),
reuses the existing frozen-M0/Frozen-Circular processing path, and reports
whether the residual has a stable one- or two-dimensional spatial signature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402


FIT_IDS = [f"{value:03d}" for value in list(range(1, 19)) + list(range(25, 37))]
SENSITIVITY_FRAME = "027"
PREDICTORS = ("s", "t", "u", "v")
CENTERINGS = ("raw", "frame_median_subtracted")
LOW_FREQ_BINS = 12
PNP_UNCERTAINTY_LOW_MM = 0.025
PNP_UNCERTAINTY_HIGH_MM = 0.033
DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_FROZEN_PROVENANCE = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "circular_vs_elliptical_cone"
    / "provenance.json"
)
DEFAULT_FORMAL_CONE = MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0817" / "spatial_residual_observability"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def clean_json(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def robust_span(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) else math.nan


def pca_rays(xy: np.ndarray) -> dict[str, Any]:
    xy = np.asarray(xy, dtype=np.float64)
    center = np.mean(xy, axis=0)
    centered = xy - center
    covariance = np.cov(centered, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    s_axis = np.asarray(eigenvectors[:, order[0]], dtype=np.float64)
    # Make the principal direction deterministic.  The ray lines are mostly
    # vertical in the sensor, so prefer positive y when that is the dominant
    # component; otherwise prefer positive x.
    if (abs(s_axis[1]) >= abs(s_axis[0]) and s_axis[1] < 0.0) or (
        abs(s_axis[1]) < abs(s_axis[0]) and s_axis[0] < 0.0
    ):
        s_axis = -s_axis
    t_axis = np.asarray([-s_axis[1], s_axis[0]], dtype=np.float64)
    coordinates = centered @ np.column_stack([s_axis, t_axis])
    total = float(np.sum(eigenvalues))
    explained = eigenvalues / total if total > 0.0 else np.asarray([math.nan, math.nan])
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
        "s_robust_span": robust_span(coordinates[:, 0]),
        "t_robust_span": robust_span(coordinates[:, 1]),
        "anisotropy_sqrt_eigenvalue_ratio": float(math.sqrt(eigenvalues[0] / eigenvalues[1]))
        if eigenvalues[1] > 0.0
        else math.inf,
        "s": coordinates[:, 0],
        "t": coordinates[:, 1],
    }


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-tolerant enough Spearman rho without adding a stats dependency."""
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    # Average ranks for ties, using a compact implementation.
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
            start = stop
        return result

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def trend_stats(x: np.ndarray, y: np.ndarray, bins: int = LOW_FREQ_BINS) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    empty = {
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
        "intercept_mm": math.nan,
        "pearson_r": math.nan,
        "spearman_rho": math.nan,
        "linear_r2": math.nan,
        "binned_explained_fraction": math.nan,
        "low_frequency_amplitude_mm": math.nan,
        "low_frequency_bin_mean_std_mm": math.nan,
        "low_frequency_bin_count": 0,
    }
    if len(x) == 0:
        return empty
    x_span = float(np.ptp(x))
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    total_ss = float(np.sum(y_centered * y_centered))
    result = dict(empty)
    result.update(
        {
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "x_span": x_span,
            "y_mean_mm": y_mean,
            "y_std_mm": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "y_p05_mm": float(np.percentile(y, 5)),
            "y_p95_mm": float(np.percentile(y, 95)),
        }
    )
    if len(x) >= 3 and x_span > 0.0:
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ beta
        slope = float(beta[1])
        result.update(
            {
                "slope_mm_per_x": slope,
                "slope_mm_per_x_span": float(slope * x_span),
                "intercept_mm": float(beta[0]),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) > 0.0 else math.nan,
                "spearman_rho": rank_correlation(x, y),
                "linear_r2": float(1.0 - np.sum((y - prediction) ** 2) / total_ss) if total_ss > 0.0 else math.nan,
            }
        )
    if len(y) >= 3 and total_ss > 0.0:
        order = np.argsort(x, kind="mergesort")
        chunks = [chunk for chunk in np.array_split(order, min(int(bins), len(order))) if len(chunk)]
        bin_means = np.asarray([np.mean(y[chunk]) for chunk in chunks], dtype=np.float64)
        within_ss = float(sum(np.sum((y[chunk] - np.mean(y[chunk])) ** 2) for chunk in chunks))
        result.update(
            {
                "binned_explained_fraction": float(1.0 - within_ss / total_ss),
                "low_frequency_amplitude_mm": float(np.ptp(bin_means)),
                "low_frequency_bin_mean_std_mm": float(np.std(bin_means, ddof=1)) if len(bin_means) > 1 else 0.0,
                "low_frequency_bin_count": int(len(bin_means)),
            }
        )
    return result


def make_point_rows(processed: Mapping[str, Mapping[str, Any]], intrinsics: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    point_rows: list[dict[str, Any]] = []
    for frame_id in FIT_IDS:
        item = processed[frame_id]
        uv = np.asarray(item["uv"], dtype=np.float64)
        normalized = cv2.undistortPoints(
            uv.reshape(-1, 1, 2), intrinsics.camera_matrix, intrinsics.dist_coeffs
        ).reshape(-1, 2)
        for index in range(len(uv)):
            point_rows.append(
                {
                    "frame_id": frame_id,
                    "source_dataset": str(item["source_dataset"]),
                    "point_index_valid": int(index),
                    "u_px": float(uv[index, 0]),
                    "v_px": float(uv[index, 1]),
                    "xn": float(normalized[index, 0]),
                    "yn": float(normalized[index, 1]),
                    "lambda_cone_mm": float(item["lambda_model"][index]),
                    "lambda_truth_mm": float(item["lambda_truth"][index]),
                    "delta_lambda_mm": float(item["residual"][index]),
                }
            )
    pca = pca_rays(np.asarray([[row["xn"], row["yn"]] for row in point_rows], dtype=np.float64))
    frame_medians: dict[str, float] = {}
    for frame_id in FIT_IDS:
        values = np.asarray([row["delta_lambda_mm"] for row in point_rows if row["frame_id"] == frame_id])
        frame_medians[frame_id] = float(np.median(values))
    for index, row in enumerate(point_rows):
        row["pca_s"] = float(pca["s"][index])
        row["pca_t"] = float(pca["t"][index])
        row["frame_residual_median_mm"] = frame_medians[row["frame_id"]]
        row["residual_centered_mm"] = float(row["delta_lambda_mm"] - frame_medians[row["frame_id"]])
    pca["frame_medians"] = frame_medians
    return point_rows, pca


def frame_support_rows(point_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_frame: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in point_rows:
        by_frame[str(row["frame_id"])].append(row)
    result: list[dict[str, Any]] = []
    for frame_id in FIT_IDS:
        rows = by_frame[frame_id]
        def col(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        s, t = col("pca_s"), col("pca_t")
        xn, yn = col("xn"), col("yn")
        delta = col("delta_lambda_mm")
        result.append(
            {
                "frame_id": frame_id,
                "source_dataset": str(rows[0]["source_dataset"]),
                "point_count": len(rows),
                "xn_min": float(np.min(xn)),
                "xn_max": float(np.max(xn)),
                "xn_span": float(np.ptp(xn)),
                "yn_min": float(np.min(yn)),
                "yn_max": float(np.max(yn)),
                "yn_span": float(np.ptp(yn)),
                "s_min": float(np.min(s)),
                "s_max": float(np.max(s)),
                "s_span": float(np.ptp(s)),
                "s_robust_span": robust_span(s),
                "t_min": float(np.min(t)),
                "t_max": float(np.max(t)),
                "t_span": float(np.ptp(t)),
                "t_robust_span": robust_span(t),
                "t_to_s_robust_span_ratio": float(robust_span(t) / robust_span(s))
                if robust_span(s) > 0.0
                else math.nan,
                "delta_median_mm": float(np.median(delta)),
                "delta_p05_mm": float(np.percentile(delta, 5)),
                "delta_p95_mm": float(np.percentile(delta, 95)),
                "delta_p95_p05_span_mm": float(np.percentile(delta, 95) - np.percentile(delta, 5)),
            }
        )
    return result


def trend_rows(point_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame_values = np.asarray([str(row["frame_id"]) for row in point_rows])
    predictor_values = {
        "s": np.asarray([float(row["pca_s"]) for row in point_rows], dtype=np.float64),
        "t": np.asarray([float(row["pca_t"]) for row in point_rows], dtype=np.float64),
        "u": np.asarray([float(row["u_px"]) for row in point_rows], dtype=np.float64),
        "v": np.asarray([float(row["v_px"]) for row in point_rows], dtype=np.float64),
    }
    residual_values = {
        "raw": np.asarray([float(row["delta_lambda_mm"]) for row in point_rows], dtype=np.float64),
        "frame_median_subtracted": np.asarray([float(row["residual_centered_mm"]) for row in point_rows], dtype=np.float64),
    }
    output: list[dict[str, Any]] = []

    def add(scope: str, frame_id: str, mask: np.ndarray, centering: str, predictor: str) -> None:
        stats = trend_stats(predictor_values[predictor][mask], residual_values[centering][mask])
        output.append(
            {
                "scope": scope,
                "frame_id": frame_id,
                "centering": centering,
                "predictor": predictor,
                **stats,
            }
        )

    all_mask = np.ones(len(point_rows), dtype=bool)
    no_027_mask = frame_values != SENSITIVITY_FRAME
    for scope, mask in (("global", all_mask), ("global_without_027", no_027_mask)):
        for centering in CENTERINGS:
            for predictor in PREDICTORS:
                add(scope, "", mask, centering, predictor)
    for frame_id in FIT_IDS:
        mask = frame_values == frame_id
        for centering in CENTERINGS:
            for predictor in PREDICTORS:
                add("frame", frame_id, mask, centering, predictor)
    return output


def aggregate_frame_trends(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for centering in CENTERINGS:
        for predictor in PREDICTORS:
            frame_rows = [
                row
                for row in rows
                if row["scope"] == "frame" and row["centering"] == centering and row["predictor"] == predictor
            ]
            global_row = next(
                row
                for row in rows
                if row["scope"] == "global" and row["centering"] == centering and row["predictor"] == predictor
            )
            slopes = np.asarray([finite(row["slope_mm_per_x_span"]) for row in frame_rows], dtype=np.float64)
            rho = np.asarray([finite(row["spearman_rho"]) for row in frame_rows], dtype=np.float64)
            ev = np.asarray([finite(row["binned_explained_fraction"]) for row in frame_rows], dtype=np.float64)
            amp = np.asarray([finite(row["low_frequency_amplitude_mm"]) for row in frame_rows], dtype=np.float64)
            finite_slopes = slopes[np.isfinite(slopes)]
            finite_rho = rho[np.isfinite(rho)]
            finite_ev = ev[np.isfinite(ev)]
            finite_amp = amp[np.isfinite(amp)]
            global_slope = finite(global_row["slope_mm_per_x_span"])
            if math.isfinite(global_slope) and abs(global_slope) > 1.0e-12 and len(finite_slopes):
                same_sign = float(np.mean(np.sign(finite_slopes) == np.sign(global_slope)))
            elif len(finite_slopes):
                same_sign = float(max(np.mean(finite_slopes >= 0.0), np.mean(finite_slopes <= 0.0)))
            else:
                same_sign = math.nan
            output.append(
                {
                    "scope": "frame_aggregate",
                    "frame_id": "",
                    "centering": centering,
                    "predictor": predictor,
                    "n_points": int(sum(int(row["n_points"]) for row in frame_rows)),
                    "frame_count": int(len(frame_rows)),
                    "global_slope_mm_per_x_span": global_slope,
                    "frame_slope_median_mm_per_x_span": float(np.median(finite_slopes)) if len(finite_slopes) else math.nan,
                    "frame_slope_p05_mm_per_x_span": float(np.percentile(finite_slopes, 5)) if len(finite_slopes) else math.nan,
                    "frame_slope_p95_mm_per_x_span": float(np.percentile(finite_slopes, 95)) if len(finite_slopes) else math.nan,
                    "frame_same_sign_fraction": same_sign,
                    "frame_abs_spearman_median": float(np.median(np.abs(finite_rho))) if len(finite_rho) else math.nan,
                    "frame_binned_ev_median": float(np.median(finite_ev)) if len(finite_ev) else math.nan,
                    "frame_binned_ev_p75": float(np.percentile(finite_ev, 75)) if len(finite_ev) else math.nan,
                    "frame_low_frequency_amplitude_median_mm": float(np.median(finite_amp)) if len(finite_amp) else math.nan,
                    "frame_low_frequency_amplitude_p75_mm": float(np.percentile(finite_amp, 75)) if len(finite_amp) else math.nan,
                    "frame_count_amplitude_over_pnp_high": int(np.count_nonzero(finite_amp > PNP_UNCERTAINTY_HIGH_MM)),
                }
            )
    return output


def lookup_trend(rows: Sequence[Mapping[str, Any]], scope: str, centering: str, predictor: str) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["scope"] == scope and row["centering"] == centering and row["predictor"] == predictor
    )


def classify_ray_support(pca: Mapping[str, Any], support: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    s_robust = finite(pca["s_robust_span"])
    t_robust = finite(pca["t_robust_span"])
    ratios = np.asarray([finite(row["t_to_s_robust_span_ratio"]) for row in support], dtype=np.float64)
    ratios = ratios[np.isfinite(ratios)]
    median_ratio = float(np.median(ratios)) if len(ratios) else math.nan
    frame_weak_2d_fraction = float(np.mean(ratios >= 0.05)) if len(ratios) else math.nan
    frame_clear_2d_fraction = float(np.mean(ratios >= 0.10)) if len(ratios) else math.nan
    explained_t = finite(pca["explained_t"])
    robust_ratio = float(t_robust / s_robust) if s_robust > 0.0 else math.nan
    if explained_t >= 0.10 and robust_ratio >= 0.15 and median_ratio >= 0.10 and frame_clear_2d_fraction >= 0.70:
        label = "CLEAR_2D"
    elif explained_t >= 0.02 or robust_ratio >= 0.05 or median_ratio >= 0.03:
        label = "WEAK_2D"
    else:
        label = "1D"
    return {
        "label": label,
        "global_s_robust_span": s_robust,
        "global_t_robust_span": t_robust,
        "global_t_to_s_robust_span_ratio": robust_ratio,
        "median_frame_t_to_s_robust_span_ratio": median_ratio,
        "frame_fraction_t_to_s_ge_0.05": frame_weak_2d_fraction,
        "frame_fraction_t_to_s_ge_0.10": frame_clear_2d_fraction,
        "gates": {
            "clear_2d_explained_t_min": 0.10,
            "clear_2d_global_t_to_s_robust_span_min": 0.15,
            "clear_2d_median_frame_t_to_s_robust_span_min": 0.10,
            "clear_2d_frame_fraction_min": 0.70,
            "weak_2d_explained_t_min": 0.02,
            "weak_2d_global_t_to_s_robust_span_min": 0.05,
            "weak_2d_median_frame_t_to_s_robust_span_min": 0.03,
        },
    }


def classify_spatial_residual(
    trend_rows_all: Sequence[Mapping[str, Any]], aggregate_rows: Sequence[Mapping[str, Any]], ray_support: Mapping[str, Any]
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for predictor in ("s", "t"):
        global_row = lookup_trend(trend_rows_all, "global", "frame_median_subtracted", predictor)
        aggregate = next(
            row
            for row in aggregate_rows
            if row["centering"] == "frame_median_subtracted" and row["predictor"] == predictor
        )
        global_amp = finite(global_row["low_frequency_amplitude_mm"])
        frame_amp = finite(aggregate.get("frame_low_frequency_amplitude_median_mm"))
        evidence_amp = max(value for value in (global_amp, frame_amp) if math.isfinite(value)) if (
            math.isfinite(global_amp) or math.isfinite(frame_amp)
        ) else math.nan
        candidates.append(
            {
                "predictor": predictor,
                "global_low_frequency_amplitude_mm": global_amp,
                "median_frame_low_frequency_amplitude_mm": frame_amp,
                "evidence_amplitude_mm": evidence_amp,
                "evidence_amplitude_over_pnp_high": evidence_amp / PNP_UNCERTAINTY_HIGH_MM
                if math.isfinite(evidence_amp)
                else math.nan,
                "frame_same_sign_fraction": finite(aggregate.get("frame_same_sign_fraction")),
                "frame_binned_ev_median": finite(aggregate.get("frame_binned_ev_median")),
                "frame_abs_spearman_median": finite(aggregate.get("frame_abs_spearman_median")),
            }
        )
    candidates.sort(key=lambda row: finite(row["evidence_amplitude_mm"]), reverse=True)
    best = candidates[0]
    amp = finite(best["evidence_amplitude_mm"])
    consistency = finite(best["frame_same_sign_fraction"])
    ev = finite(best["frame_binned_ev_median"])
    # The post-median signal must exceed the supplied PnP truth-uncertainty
    # band and recur in multiple frames; otherwise it is not actionable.
    if amp >= 3.0 * PNP_UNCERTAINTY_HIGH_MM and consistency >= 0.60 and ev >= 0.10:
        label = "STRONG"
    elif amp >= 1.5 * PNP_UNCERTAINTY_HIGH_MM and consistency >= 0.40 and ev >= 0.03:
        label = "MODERATE"
    else:
        label = "WEAK"
    if label == "WEAK":
        next_step = "STOP"
    elif ray_support["label"] == "CLEAR_2D" and best["predictor"] == "t" and finite(best["evidence_amplitude_mm"]) >= PNP_UNCERTAINTY_HIGH_MM:
        next_step = "C1 + C2"
    else:
        next_step = "C1 only"
    return {
        "label": label,
        "best_predictor": best["predictor"],
        "candidate_evidence": candidates,
        "next_step": next_step,
        "gates": {
            "strong_amplitude_min_mm": 3.0 * PNP_UNCERTAINTY_HIGH_MM,
            "strong_frame_same_sign_min": 0.60,
            "strong_frame_binned_ev_min": 0.10,
            "moderate_amplitude_min_mm": 1.5 * PNP_UNCERTAINTY_HIGH_MM,
            "moderate_frame_same_sign_min": 0.40,
            "moderate_frame_binned_ev_min": 0.03,
        },
    }


def frame_trend_summary(aggregate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for centering in CENTERINGS:
        result[centering] = {}
        for predictor in PREDICTORS:
            row = next(
                row
                for row in aggregate_rows
                if row["centering"] == centering and row["predictor"] == predictor
            )
            result[centering][predictor] = dict(row)
    return result


def render_report(
    point_rows: Sequence[Mapping[str, Any]],
    pca: Mapping[str, Any],
    support_rows: Sequence[Mapping[str, Any]],
    trend_rows_all: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    ray_support: Mapping[str, Any],
    spatial: Mapping[str, Any],
    paths: Mapping[str, Any],
) -> str:
    raw_s = lookup_trend(trend_rows_all, "global", "raw", "s")
    raw_t = lookup_trend(trend_rows_all, "global", "raw", "t")
    centered_s = lookup_trend(trend_rows_all, "global", "frame_median_subtracted", "s")
    centered_t = lookup_trend(trend_rows_all, "global", "frame_median_subtracted", "t")
    raw_delta = np.asarray([float(row["delta_lambda_mm"]) for row in point_rows])
    centered_delta = np.asarray([float(row["residual_centered_mm"]) for row in point_rows])
    frame_median_values = np.asarray([float(row["frame_residual_median_mm"]) for row in point_rows])
    unique_frame_medians = np.asarray([float(row["delta_median_mm"]) for row in support_rows])
    no027_s = lookup_trend(trend_rows_all, "global_without_027", "frame_median_subtracted", "s")
    no027_t = lookup_trend(trend_rows_all, "global_without_027", "frame_median_subtracted", "t")
    return "\n".join(
        [
            "# FIT ray residual spatial observability",
            "",
            "## 结论",
            "",
            f"- `RAY_SUPPORT = {ray_support['label']}`",
            f"- `SPATIAL_RESIDUAL = {spatial['label']}`",
            f"- 下一步建议：`{spatial['next_step']}`",
            "",
            f"本轮使用 {len(FIT_IDS)} 个 FIT 帧（001–018、025–036），有效 ray 共 **{len(point_rows):,}** 个；frame 027 保留在统计中。未打开、未读取、未评分 Validation 019–024、037–040。",
            "",
            "## 数据与计算边界",
            "",
            "- K/D：冻结 M0 runtime K/D；`(u,v) → (xn,yn)` 使用 `cv2.undistortPoints`。",
            "- Truth：当前 FIT 棋盘 PnP plane，`lambda_truth = -d / (ray · normal)`，取 camera-Z/lambda。",
            "- Cone：冻结 Circular Cone production path；`lambda_cone` 只做重放，不重新拟合。",
            "- 残差定义：`delta_lambda = lambda_truth - lambda_cone`。",
            "- 未训练 spline、未生成 `F(s)`/`F(s,t)`、未修改 K/D 或 Cone；约 300 mm 仅作为当前水平物理视场约束，不创建新的 `ROI_300`。",
            "",
            "## Ray support / PCA",
            "",
            f"PCA center = ({pca['center_xn']:.8g}, {pca['center_yn']:.8g}); s-axis = ({pca['axis_s_xn']:.8g}, {pca['axis_s_yn']:.8g}); t-axis = ({pca['axis_t_xn']:.8g}, {pca['axis_t_yn']:.8g}).",
            f"主方向解释方差 **{pca['explained_s']:.4f}**，次方向 **{pca['explained_t']:.4f}**，sqrt eigenvalue ratio **{pca['anisotropy_sqrt_eigenvalue_ratio']:.3g}**。",
            f"全局 robust span（P05–P95）：s = **{ray_support['global_s_robust_span']:.8g}**，t = **{ray_support['global_t_robust_span']:.8g}**，t/s = **{ray_support['global_t_to_s_robust_span_ratio']:.4f}**。",
            f"逐帧 t/s robust-span ratio 中位数 **{ray_support['median_frame_t_to_s_robust_span_ratio']:.4f}**；达到 0.05/0.10 的帧比例为 **{ray_support['frame_fraction_t_to_s_ge_0.05']:.3f} / {ray_support['frame_fraction_t_to_s_ge_0.10']:.3f}**。",
            "",
            "判据：`CLEAR_2D` 需要次方向解释方差 ≥0.10、全局 t/s ≥0.15、逐帧 t/s 中位数 ≥0.10 且 ≥70% 帧达到 0.10；较弱但非退化的 2D 支持归为 `WEAK_2D`，其余为 `1D`。",
            "",
            "## Residual low-frequency trends",
            "",
            "低频趋势使用每个 predictor 的 12 个等样本 bin 的 bin-mean explained fraction 和 bin-mean peak-to-peak；它们是描述性统计，不是 correction fit。",
            "",
            "| residual | predictor | global low-frequency amplitude (mm) | binned EV | linear slope over span (mm) | frame sign consistency | median frame EV |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| raw | {predictor} | {finite(lookup_trend(trend_rows_all, 'global', 'raw', predictor)['low_frequency_amplitude_mm']):.6g} | {finite(lookup_trend(trend_rows_all, 'global', 'raw', predictor)['binned_explained_fraction']):.4f} | {finite(lookup_trend(trend_rows_all, 'global', 'raw', predictor)['slope_mm_per_x_span']):.6g} | {finite(next(row for row in aggregate_rows if row['centering']=='raw' and row['predictor']==predictor).get('frame_same_sign_fraction')):.3f} | {finite(next(row for row in aggregate_rows if row['centering']=='raw' and row['predictor']==predictor).get('frame_binned_ev_median')):.4f} |"
                for predictor in PREDICTORS
            ],
            *[
                f"| frame median subtracted | {predictor} | {finite(lookup_trend(trend_rows_all, 'global', 'frame_median_subtracted', predictor)['low_frequency_amplitude_mm']):.6g} | {finite(lookup_trend(trend_rows_all, 'global', 'frame_median_subtracted', predictor)['binned_explained_fraction']):.4f} | {finite(lookup_trend(trend_rows_all, 'global', 'frame_median_subtracted', predictor)['slope_mm_per_x_span']):.6g} | {finite(next(row for row in aggregate_rows if row['centering']=='frame_median_subtracted' and row['predictor']==predictor).get('frame_same_sign_fraction')):.3f} | {finite(next(row for row in aggregate_rows if row['centering']=='frame_median_subtracted' and row['predictor']==predictor).get('frame_binned_ev_median')):.4f} |"
                for predictor in PREDICTORS
            ],
            "",
            f"frame median 的跨帧范围为 **{np.ptp(unique_frame_medians):.6g} mm**；全体 raw residual P05–P95 为 **{np.percentile(raw_delta, 5):.6g}–{np.percentile(raw_delta, 95):.6g} mm**，去 frame median 后为 **{np.percentile(centered_delta, 5):.6g}–{np.percentile(centered_delta, 95):.6g} mm**。",
            f"frame 027 的 residual median = **{float(np.median([float(row['delta_lambda_mm']) for row in point_rows if row['frame_id']=='027'])):.6g} mm**；保留它用于最终结论，同时给出去 027 的 centered s/t amplitude：s **{finite(no027_s['low_frequency_amplitude_mm']):.6g} mm**，t **{finite(no027_t['low_frequency_amplitude_mm']):.6g} mm**。",
            "",
            "## PnP truth uncertainty comparison",
            "",
            f"采用已有 PnP truth uncertainty 参考带 **{PNP_UNCERTAINTY_LOW_MM:.3f}–{PNP_UNCERTAINTY_HIGH_MM:.3f} mm**。去 frame median 后，候选空间结构的最大证据方向为 **{spatial['best_predictor']}**，低频幅度 **{finite(spatial['candidate_evidence'][0]['evidence_amplitude_mm']):.6g} mm**，约为上限的 **{finite(spatial['candidate_evidence'][0]['evidence_amplitude_over_pnp_high']):.3g}×**；逐帧同号比例 **{finite(spatial['candidate_evidence'][0]['frame_same_sign_fraction']):.3f}**。",
            "",
            "这里的比较只用于量级判断：若去中位数后的重复空间变化不超过约 0.025–0.033 mm，不能把它稳健地解释为 Cone 的独立空间误差；超过该带且跨帧同向重复，才进入 STRONG/MODERATE。",
            "",
            "## Decision gates",
            "",
            f"- `SPATIAL_RESIDUAL = STRONG`：低频幅度 ≥ {3.0 * PNP_UNCERTAINTY_HIGH_MM:.3f} mm、逐帧同号 ≥ 0.60、逐帧 median binned EV ≥ 0.10。",
            f"- `SPATIAL_RESIDUAL = MODERATE`：低频幅度 ≥ {1.5 * PNP_UNCERTAINTY_HIGH_MM:.3f} mm、逐帧同号 ≥ 0.40、逐帧 median binned EV ≥ 0.03。",
            "- 否则为 `WEAK`，建议 `STOP`；即便 residual 有信号，只有 `CLEAR_2D` ray support 且 t 方向超过 uncertainty 才建议 `C1 + C2`，否则先做 `C1 only`。",
            "",
            "## Reproducibility",
            "",
            f"- FIT data root: `{paths['data_root']}`",
            f"- measurement config: `{paths['measurement_config']}`",
            f"- frozen Circular provenance: `{paths['frozen_provenance']}`",
            f"- formal Cone range file: `{paths['formal_cone']}`",
            "- 详细逐点数据见 `fit_ray_residual_points.csv`；全局、去 027、逐帧及 frame aggregate 趋势见 `spatial_residual_observability.csv`；PCA/support 与分类证据见 `ray_support_summary.json`。",
            "",
        ]
    )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    data_root = args.data_root.resolve()
    measurement_config = args.measurement_config.resolve()
    frozen_provenance = args.frozen_provenance.resolve()
    formal_cone = args.formal_cone.resolve()

    # inventory_fit resolves only the two declared FIT roots.  In particular,
    # it never enumerates the laser_plane/validation directory.
    groups = board.inventory_fit(data_root)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(measurement_config)
    frozen_model, frozen_info = board.load_frozen_model_checked(frozen_provenance, formal_cone)
    summaries, processed = board.process_groups_board(
        groups, intrinsics, calibration, reconstruction_params, frozen_model
    )
    if [str(frame_id) for frame_id in sorted(processed, key=int)] != FIT_IDS:
        raise RuntimeError("FIT frame registry mismatch")

    point_rows, pca = make_point_rows(processed, intrinsics)
    support_rows = frame_support_rows(point_rows)
    trend = trend_rows(point_rows)
    aggregates = aggregate_frame_trends(trend)
    ray_support = classify_ray_support(pca, support_rows)
    spatial = classify_spatial_residual(trend, aggregates, ray_support)

    raw_delta = np.asarray([float(row["delta_lambda_mm"]) for row in point_rows], dtype=np.float64)
    centered_delta = np.asarray([float(row["residual_centered_mm"]) for row in point_rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "task": "FIT ray residual spatial observability",
        "fit_ids": FIT_IDS,
        "validation_ids_not_read": [f"{value:03d}" for value in list(range(19, 25)) + list(range(37, 41))],
        "validation_opened": False,
        "frame_027_retained": True,
        "point_count": len(point_rows),
        "frame_count": len(FIT_IDS),
        "valid_points_by_frame": {row["frame_id"]: int(row["point_count"]) for row in support_rows},
        "frozen_inputs": {
            "measurement_config": str(measurement_config),
            "measurement_config_sha256": sha256_file(measurement_config),
            "frozen_provenance": str(frozen_provenance),
            "frozen_provenance_sha256": sha256_file(frozen_provenance),
            "formal_cone": str(formal_cone),
            "formal_cone_sha256": sha256_file(formal_cone),
            "data_root": str(data_root),
            "frozen_model_info": frozen_info,
            "camera_matrix": np.asarray(intrinsics.camera_matrix, dtype=np.float64).tolist(),
            "dist_coeffs": np.asarray(intrinsics.dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
        },
        "pca": {key: value for key, value in pca.items() if key not in {"s", "t", "frame_medians"}},
        "ray_support": ray_support,
        "per_frame_support": support_rows,
        "frame_trend_summary": frame_trend_summary(aggregates),
        "residual_distribution": {
            "raw_p05_mm": float(np.percentile(raw_delta, 5)),
            "raw_p95_mm": float(np.percentile(raw_delta, 95)),
            "raw_p95_p05_span_mm": float(np.percentile(raw_delta, 95) - np.percentile(raw_delta, 5)),
            "frame_median_span_mm": float(np.ptp(np.asarray([row["delta_median_mm"] for row in support_rows]))),
            "frame_median_subtracted_p05_mm": float(np.percentile(centered_delta, 5)),
            "frame_median_subtracted_p95_mm": float(np.percentile(centered_delta, 95)),
            "frame_median_subtracted_p95_p05_span_mm": float(
                np.percentile(centered_delta, 95) - np.percentile(centered_delta, 5)
            ),
        },
        "pnp_truth_uncertainty_reference": {
            "low_mm": PNP_UNCERTAINTY_LOW_MM,
            "high_mm": PNP_UNCERTAINTY_HIGH_MM,
            "source": "existing PnP truth uncertainty reference supplied for this audit; no Validation data used",
        },
        "classification": {
            "RAY_SUPPORT": ray_support["label"],
            "SPATIAL_RESIDUAL": spatial["label"],
            "next_step": spatial["next_step"],
            "spatial": spatial,
        },
        "implementation_constraints": {
            "kd_refit": False,
            "cone_refit": False,
            "spline_trained": False,
            "residual_correction_written": False,
            "new_roi_300_created": False,
        },
    }
    paths = {
        "data_root": str(data_root),
        "measurement_config": str(measurement_config),
        "frozen_provenance": str(frozen_provenance),
        "formal_cone": str(formal_cone),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fit_ray_residual_points.csv", point_rows)
    write_csv(output_dir / "spatial_residual_observability.csv", list(trend) + list(aggregates))
    (output_dir / "ray_support_summary.json").write_text(
        json.dumps(clean_json(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(point_rows, pca, support_rows, trend, aggregates, ray_support, spatial, paths),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "point_count": len(point_rows),
                "frame_count": len(FIT_IDS),
                "RAY_SUPPORT": ray_support["label"],
                "SPATIAL_RESIDUAL": spatial["label"],
                "next_step": spatial["next_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run(parse_args())
