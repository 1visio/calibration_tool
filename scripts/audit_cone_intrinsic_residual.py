#!/usr/bin/env python3
"""Task 6B: audit frozen-Circular-Cone residuals in intrinsic coordinates.

FIT-only (001--018 and 025--036, including 027).  This module reuses the
existing PnP/ray-plane/Steger pipeline and a frozen Circular Cone; it never
fits a Cone, opens Validation data, changes extraction, or writes a correction.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT.parents[1]
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import audit_board_coordinate_residual as board_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402


DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "cone_intrinsic_audit"
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE
FIT_IDS = [f"{i:03d}" for i in range(1, 19)] + [f"{i:03d}" for i in range(25, 37)]
TARGET_FRAME = "027"
BOOTSTRAP_REPS = 300
BOOTSTRAP_SEED = 20260815

# Several bin scales are deliberately reported.  No single scale is used to
# declare a surface effect.
BIN_SPECS = (("fine", 10.0, 10.0), ("medium", 20.0, 20.0), ("coarse", 30.0, 40.0))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan, math.nan, int(len(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), int(len(x))


def basic(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0, "mean": math.nan, "std": math.nan, "rmse": math.nan, "p95_abs": math.nan, "min": math.nan, "max": math.nan}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p95_abs": float(np.percentile(np.abs(x), 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def build_axis_basis(axis: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(axis, dtype=np.float64)
    d /= np.linalg.norm(d)
    reference = np.array([0.0, 0.0, 1.0]) if abs(float(d[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    e2 /= np.linalg.norm(e2)
    return d, e1, e2


def cone_intrinsic_for_points(points: np.ndarray, model: Mapping[str, Any], basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    d, e1, e2 = basis
    apex = np.asarray(model["apex_camera_mm"], dtype=np.float64)
    q = np.asarray(points, dtype=np.float64) - apex[None, :]
    a = q @ d
    radial_vector = q - a[:, None] * d[None, :]
    r = np.linalg.norm(radial_vector, axis=1)
    c1 = radial_vector @ e1
    c2 = radial_vector @ e2
    phi_deg = np.degrees(np.arctan2(c2, c1))
    alpha_truth_deg = np.degrees(np.arctan2(r, a))
    alpha_model_deg = float(model["half_apex_angle_deg"])
    delta_alpha_deg = alpha_truth_deg - alpha_model_deg
    # Exact signed normal-distance convention used by CircularConeModel.
    cos2 = math.cos(math.radians(alpha_model_deg)) ** 2
    f = a * a - cos2 * np.sum(q * q, axis=1)
    gradient = 2.0 * (a[:, None] * d[None, :] - cos2 * q)
    e_surface = f / np.maximum(np.linalg.norm(gradient, axis=1), 1.0e-12)
    return {
        "a_mm": a,
        "r_mm": r,
        "phi_deg": phi_deg,
        "alpha_truth_deg": alpha_truth_deg,
        "alpha_model_deg": np.repeat(alpha_model_deg, len(a)),
        "delta_alpha_deg": delta_alpha_deg,
        "e_surface_mm": e_surface,
        "radial_e1_mm": c1,
        "radial_e2_mm": c2,
    }


def add_intrinsic_coordinates(processed: Mapping[str, Mapping[str, Any]], model: Mapping[str, Any]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    basis = build_axis_basis(model["axis_unit_camera"])
    for item in processed.values():
        item["intrinsic"] = cone_intrinsic_for_points(item["p_cam"], model, basis)
    return np.asarray(model["axis_unit_camera"], dtype=np.float64), basis


def concat_arrays(processed: Mapping[str, Mapping[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(processed[f]["intrinsic"][key], dtype=np.float64) for f in sorted(processed, key=int)])


def frame_outcome(item: Mapping[str, Any], outcome: str) -> np.ndarray:
    if outcome == "e_lambda_mm":
        return np.asarray(item["residual"], dtype=np.float64)
    return np.asarray(item["intrinsic"][outcome], dtype=np.float64)


def frame_predictor(item: Mapping[str, Any], predictor: str) -> np.ndarray:
    return np.asarray(item["intrinsic"][predictor], dtype=np.float64)


def predictor_edges(predictor: str, values: np.ndarray, bins: int = 24) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if predictor == "phi_deg":
        return np.linspace(-180.0, 180.0, bins + 1)
    lo = math.floor(float(np.min(x)) / 20.0) * 20.0 if len(x) else 0.0
    hi = math.ceil(float(np.max(x)) / 20.0) * 20.0 if len(x) else 20.0
    if hi <= lo:
        hi = lo + 20.0
    return np.linspace(lo, hi, bins + 1)


def binned_ev(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) < 3:
        return math.nan
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return math.nan
    index = np.digitize(x, edges[1:-1], right=False)
    pred = np.full(len(y), np.mean(y), dtype=np.float64)
    for i in range(len(edges) - 1):
        selected = index == i
        if np.any(selected):
            pred[selected] = np.mean(y[selected])
    return float(1.0 - np.sum((y - pred) ** 2) / total)


def linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(y) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    xx = (x - np.mean(x)) / max(float(np.std(x)), 1.0e-12)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(xx)), xx]), y, rcond=None)[0]
    total = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - np.column_stack([np.ones(len(xx)), xx]) @ beta) ** 2) / total)


def harmonic_fit(phi_deg: np.ndarray, y: np.ndarray) -> dict[str, float]:
    phi = np.radians(np.asarray(phi_deg, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(phi) & np.isfinite(y)
    phi, y = phi[mask], y[mask]
    phi_span_deg = float(np.ptp(np.degrees(phi))) if len(phi) else math.nan
    if len(y) < 6:
        return {"count": int(len(y)), "phi_span_deg": phi_span_deg, "design_condition_number": math.nan, "harmonic_identifiable": False, "r2": math.nan, "amplitude_1": math.nan, "amplitude_2": math.nan, "phase_1_deg": math.nan, "phase_2_deg": math.nan}
    design = np.column_stack([np.ones(len(phi)), np.cos(phi), np.sin(phi), np.cos(2 * phi), np.sin(2 * phi)])
    condition_number = float(np.linalg.cond(design))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    pred = design @ beta
    total = float(np.sum((y - np.mean(y)) ** 2))
    amp1 = float(np.hypot(beta[1], beta[2]))
    amp2 = float(np.hypot(beta[3], beta[4]))
    return {
        "count": int(len(y)),
        "phi_span_deg": phi_span_deg,
        "design_condition_number": condition_number,
        "harmonic_identifiable": bool(phi_span_deg >= 90.0 and condition_number < 1.0e3),
        "r2": float(1.0 - np.sum((y - pred) ** 2) / total) if total > 0 else math.nan,
        "amplitude_1": amp1,
        "amplitude_2": amp2,
        "phase_1_deg": float(np.degrees(np.arctan2(beta[2], beta[1]))),
        "phase_2_deg": float(np.degrees(np.arctan2(beta[4], beta[3])) / 2.0),
    }


def bootstrap_relationship(
    processed: Mapping[str, Mapping[str, Any]], predictor: str, outcome: str, edges: np.ndarray, reps: int, seed: int
) -> dict[str, float]:
    frames = sorted(processed, key=int)
    rng = np.random.default_rng(seed + sum(ord(c) for c in predictor + outcome))
    rhos: list[float] = []
    evs: list[float] = []
    r2s: list[float] = []
    for _ in range(int(reps)):
        sampled = rng.integers(0, len(frames), size=len(frames))
        xs = [frame_predictor(processed[frames[i]], predictor) for i in sampled]
        ys = [frame_outcome(processed[frames[i]], outcome) for i in sampled]
        x, y = np.concatenate(xs), np.concatenate(ys)
        rhos.append(safe_spearman(x, y)[0])
        evs.append(binned_ev(x, y, edges))
        r2s.append(linear_r2(x, y))
    def ci(values: Sequence[float], q: float) -> tuple[float, float, float]:
        finite_values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if not len(finite_values):
            return math.nan, math.nan, math.nan
        return float(np.mean(finite_values)), float(np.percentile(finite_values, 2.5)), float(np.percentile(finite_values, 97.5))
    rho_mean, rho_low, rho_high = ci(rhos, 0.95)
    ev_mean, ev_low, ev_high = ci(evs, 0.95)
    r2_mean, r2_low, r2_high = ci(r2s, 0.95)
    return {"bootstrap_rho_mean": rho_mean, "bootstrap_rho_ci_low": rho_low, "bootstrap_rho_ci_high": rho_high, "bootstrap_ev_mean": ev_mean, "bootstrap_ev_ci_low": ev_low, "bootstrap_ev_ci_high": ev_high, "bootstrap_r2_mean": r2_mean, "bootstrap_r2_ci_low": r2_low, "bootstrap_r2_ci_high": r2_high}


def relationship_rows(processed: Mapping[str, Mapping[str, Any]], reps: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predictor in ("phi_deg", "a_mm"):
        x = concat_arrays(processed, predictor)
        edges = predictor_edges(predictor, x)
        for outcome in ("alpha_truth_deg", "delta_alpha_deg", "e_lambda_mm", "e_surface_mm"):
            y = concat_arrays(processed, outcome) if outcome != "e_lambda_mm" else np.concatenate([np.asarray(processed[f]["residual"], dtype=np.float64) for f in sorted(processed, key=int)])
            rho, pvalue, n = safe_spearman(x, y)
            harmonic = harmonic_fit(x, y) if predictor == "phi_deg" else {}
            boot = bootstrap_relationship(processed, predictor, outcome, edges, reps, seed)
            rows.append({"row_type": "relationship", "scope": "all30", "predictor": predictor, "outcome": outcome, "point_count": n, "frame_count": len(processed), "spearman_rho": rho, "spearman_p_value": pvalue, "binned_explained_fraction": binned_ev(x, y, edges), "simple_linear_r2": linear_r2(x, y), **harmonic, "bootstrap_reps": int(reps), "bootstrap_seed": int(seed), **boot})
    return rows


def make_frame_summary(board_summaries: Sequence[Mapping[str, Any]], processed: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for base in board_summaries:
        row = dict(base)
        item = processed[str(row["frame_id"])]
        for key in ("a_mm", "r_mm", "alpha_truth_deg", "delta_alpha_deg", "e_surface_mm"):
            values = item["intrinsic"][key]
            stats = basic(values)
            prefix = {"a_mm": "cone_a", "r_mm": "cone_r", "alpha_truth_deg": "alpha_truth", "delta_alpha_deg": "delta_alpha", "e_surface_mm": "e_surface"}[key]
            row[f"{prefix}_mean"] = stats["mean"]
            row[f"{prefix}_std"] = stats["std"]
            row[f"{prefix}_min"] = stats["min"]
            row[f"{prefix}_max"] = stats["max"]
            row[f"{prefix}_rmse"] = stats["rmse"]
            row[f"{prefix}_p95_abs"] = stats["p95_abs"]
        phi = item["intrinsic"]["phi_deg"]
        for outcome in ("alpha_truth_deg", "delta_alpha_deg", "e_surface_mm"):
            y = frame_outcome(item, outcome)
            rho, pvalue, n = safe_spearman(phi, y)
            row[f"spearman_{outcome}_vs_phi_rho"] = rho
            row[f"spearman_{outcome}_vs_phi_p_value"] = pvalue
            row[f"spearman_{outcome}_vs_phi_n"] = n
        for outcome in ("alpha_truth_deg", "delta_alpha_deg", "e_surface_mm"):
            y = frame_outcome(item, outcome)
            rho, pvalue, n = safe_spearman(item["intrinsic"]["a_mm"], y)
            row[f"spearman_{outcome}_vs_a_rho"] = rho
            row[f"spearman_{outcome}_vs_a_p_value"] = pvalue
            row[f"spearman_{outcome}_vs_a_n"] = n
        output.append(row)
    return output


def make_point_rows(processed: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        intrinsic = item["intrinsic"]
        pose = item["pose"]
        for i in range(len(item["residual"])):
            row: dict[str, Any] = {
                "frame_id": frame_id,
                "is_frame027": item["is_frame027"],
                "source_dataset": item["source_dataset"],
                "point_index": i,
                "u_px": float(item["u"][i]),
                "v_px": float(item["v"][i]),
                "lambda_truth_mm": float(item["lambda_truth"][i]),
                "lambda_model_mm": float(item["lambda_model"][i]),
                "e_lambda_mm": float(item["residual"][i]),
                "Xc_mm": float(item["p_cam"][i, 0]),
                "Yc_mm": float(item["p_cam"][i, 1]),
                "Zc_mm": float(item["p_cam"][i, 2]),
                "a_mm": float(intrinsic["a_mm"][i]),
                "r_mm": float(intrinsic["r_mm"][i]),
                "phi_deg": float(intrinsic["phi_deg"][i]),
                "alpha_truth_deg": float(intrinsic["alpha_truth_deg"][i]),
                "alpha_model_deg": float(intrinsic["alpha_model_deg"][i]),
                "delta_alpha_deg": float(intrinsic["delta_alpha_deg"][i]),
                "e_surface_mm": float(intrinsic["e_surface_mm"][i]),
                "radial_e1_mm": float(intrinsic["radial_e1_mm"][i]),
                "radial_e2_mm": float(intrinsic["radial_e2_mm"][i]),
                "pnp_rmse_px": float(pose.reprojection_rmse_px),
                "board_Xb_mm": float(item["p_board"][i, 0]),
                "board_Yb_mm": float(item["p_board"][i, 1]),
                "steger_response": float(item["response"][i]),
                "laser_local_intensity_dn": float(item["local"]["laser_intensity_dn"][i]),
                "stripe_contrast_dn": float(item["local"]["stripe_contrast_dn"][i]),
                "fwhm_px": float(item["local"]["fwhm_px"][i]),
            }
            rows.append(row)
    return rows


def intrinsic_bins(processed: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    all_a = concat_arrays(processed, "a_mm")
    rows: list[dict[str, Any]] = []
    meta: dict[str, dict[str, Any]] = {}
    for name, phi_step, a_step in BIN_SPECS:
        phi_edges = np.arange(-180.0, 180.0 + phi_step * 0.5, phi_step)
        a_low = math.floor(float(np.nanmin(all_a)) / a_step) * a_step
        a_high = math.ceil(float(np.nanmax(all_a)) / a_step) * a_step
        if a_high <= a_low:
            a_high = a_low + a_step
        a_edges = np.arange(a_low, a_high + a_step * 0.5, a_step)
        meta[name] = {"phi_edges": phi_edges, "a_edges": a_edges, "phi_step_deg": phi_step, "a_step_mm": a_step}
        for p in sorted(processed, key=int):
            intrinsic = processed[p]["intrinsic"]
            pbin = np.digitize(intrinsic["phi_deg"], phi_edges[1:-1], right=False)
            abin = np.digitize(intrinsic["a_mm"], a_edges[1:-1], right=False)
            processed[p][f"{name}_bin"] = (pbin, abin)
        for pi in range(len(phi_edges) - 1):
            for ai in range(len(a_edges) - 1):
                selected_values: dict[str, np.ndarray] = {}
                all_values: list[np.ndarray] = []
                all_delta_values: list[np.ndarray] = []
                all_surface_values: list[np.ndarray] = []
                frame_means: list[float] = []
                within_vars: list[float] = []
                frame_count = 0
                point_count = 0
                frame027_count = 0
                for frame_id in sorted(processed, key=int):
                    pbin, abin = processed[frame_id][f"{name}_bin"]
                    mask = (pbin == pi) & (abin == ai)
                    if not np.any(mask):
                        continue
                    frame_count += 1
                    point_count += int(np.count_nonzero(mask))
                    if frame_id == TARGET_FRAME:
                        frame027_count += int(np.count_nonzero(mask))
                    values = processed[frame_id]["residual"][mask]
                    selected_values[frame_id] = values
                    frame_means.append(float(np.mean(values)))
                    if len(values) > 1:
                        within_vars.append(float(np.var(values, ddof=1)))
                    all_values.append(values)
                    all_delta_values.append(processed[frame_id]["intrinsic"]["delta_alpha_deg"][mask])
                    all_surface_values.append(processed[frame_id]["intrinsic"]["e_surface_mm"][mask])
                if not all_values:
                    continue
                y = np.concatenate(all_values)
                delta_values = np.concatenate(all_delta_values)
                surface_values = np.concatenate(all_surface_values)
                rows.append({"row_type": "intrinsic_bin", "bin_scale": name, "phi_bin_deg": phi_step, "a_bin_mm": a_step, "phi_bin_index": pi, "a_bin_index": ai, "phi_start_deg": float(phi_edges[pi]), "phi_end_deg": float(phi_edges[pi + 1]), "a_start_mm": float(a_edges[ai]), "a_end_mm": float(a_edges[ai + 1]), "phi_center_deg": float((phi_edges[pi] + phi_edges[pi + 1]) / 2.0), "a_center_mm": float((a_edges[ai] + a_edges[ai + 1]) / 2.0), "point_count": point_count, "unique_frame_count": frame_count, "frame027_point_count": frame027_count, "e_lambda_mean_mm": float(np.mean(y)), "e_lambda_std_mm": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0, "e_lambda_rmse_mm": float(np.sqrt(np.mean(y * y))), "e_lambda_p95_abs_mm": float(np.percentile(np.abs(y), 95)), "delta_alpha_mean_deg": float(np.mean(delta_values)), "delta_alpha_std_deg": float(np.std(delta_values, ddof=1)) if len(delta_values) > 1 else 0.0, "delta_alpha_rmse_deg": float(np.sqrt(np.mean(delta_values * delta_values))), "delta_alpha_p95_abs_deg": float(np.percentile(np.abs(delta_values), 95)), "e_surface_mean_mm": float(np.mean(surface_values)), "e_surface_std_mm": float(np.std(surface_values, ddof=1)) if len(surface_values) > 1 else 0.0, "e_surface_rmse_mm": float(np.sqrt(np.mean(surface_values * surface_values))), "cross_frame_variance_mm2": float(np.var(frame_means, ddof=1)) if len(frame_means) > 1 else math.nan, "within_frame_variance_mm2": float(np.mean(within_vars)) if within_vars else math.nan, "cross_frame_delta_rmse_mm": float(np.sqrt(np.mean((np.asarray(frame_means) - np.mean(frame_means)) ** 2))) if len(frame_means) > 1 else math.nan})
    return rows, meta


def intrinsic_pair_rows(processed: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    frames = sorted(processed, key=int)
    for scale, info in meta.items():
        by_bin: dict[tuple[int, int], dict[str, tuple[float, float, int, float]]] = defaultdict(dict)
        for frame_id in frames:
            pbin, abin = processed[frame_id][f"{scale}_bin"]
            for key in set(zip(pbin.tolist(), abin.tolist())):
                mask = (pbin == key[0]) & (abin == key[1])
                values = processed[frame_id]["residual"][mask]
                if len(values):
                    by_bin[key][frame_id] = (float(np.mean(values)), float(np.var(values, ddof=1)) if len(values) > 1 else 0.0, int(len(values)), float(np.mean(processed[frame_id]["intrinsic"]["a_mm"][mask])))
        pair_data: dict[tuple[str, str], list[tuple[float, float, float, float, int]]] = defaultdict(list)
        for values in by_bin.values():
            present = sorted(values)
            for left, right in itertools.combinations(present, 2):
                ml, vl, nl, _ = values[left]
                mr, vr, nr, _ = values[right]
                pair_data[(left, right)].append((ml, mr, vl, vr, min(nl, nr)))
        for (left, right), entries in sorted(pair_data.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
            if len(entries) < 3:
                continue
            ml = np.asarray([e[0] for e in entries])
            mr = np.asarray([e[1] for e in entries])
            delta = ml - mr
            within = np.asarray([0.5 * (e[2] + e[3]) for e in entries])
            means = np.concatenate([ml, mr])
            output.append({"row_type": "pair", "coordinate_system": "cone_intrinsic", "bin_scale": scale, "frame_left": left, "frame_right": right, "matched_bin_count": len(entries), "matched_point_count": int(np.sum([e[4] for e in entries])), "residual_delta_bias_mm": float(np.mean(delta)), "residual_delta_rmse_mm": float(np.sqrt(np.mean(delta * delta))), "residual_delta_p95_abs_mm": float(np.percentile(np.abs(delta), 95)), "cross_frame_variance_mm2": float(np.var(means, ddof=1)) if len(means) > 1 else math.nan, "within_frame_variance_mm2": float(np.mean(within)), "cross_to_within_variance_ratio": float(np.var(means, ddof=1) / max(np.mean(within), 1.0e-12)) if len(means) > 1 else math.nan})
        pair_rows = [r for r in output if r.get("coordinate_system") == "cone_intrinsic" and r.get("bin_scale") == scale]
        if pair_rows:
            output.append({"row_type": "aggregate", "coordinate_system": "cone_intrinsic", "bin_scale": scale, "pair_count": len(pair_rows), "matched_bin_count_median": float(np.median([r["matched_bin_count"] for r in pair_rows])), "matched_pair_point_count_median": float(np.median([r["matched_point_count"] for r in pair_rows])), "residual_delta_rmse_median_mm": float(np.median([r["residual_delta_rmse_mm"] for r in pair_rows])), "residual_delta_rmse_p95_mm": float(np.percentile([r["residual_delta_rmse_mm"] for r in pair_rows], 95)), "cross_frame_variance_mm2_median": float(np.median([r["cross_frame_variance_mm2"] for r in pair_rows])), "within_frame_variance_mm2_median": float(np.median([r["within_frame_variance_mm2"] for r in pair_rows])), "cross_to_within_variance_ratio_median": float(np.median([r["cross_to_within_variance_ratio"] for r in pair_rows]))})
    return output


def add_sensor_overlap_rows(processed: Mapping[str, Mapping[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Build the 30-frame sensor-v baseline without the 5D-1 ten-pair limit."""
    pair_rows: list[dict[str, Any]] = []
    frames = sorted(processed, key=int)
    for left, right in itertools.combinations(frames, 2):
        a, b = processed[left], processed[right]
        lo = max(float(np.min(a["v"])), float(np.min(b["v"])))
        hi = min(float(np.max(a["v"])), float(np.max(b["v"])))
        grid = np.arange(math.ceil(lo), math.floor(hi) + 0.5, 1.0)
        if len(grid) < 2:
            continue
        def interp(values: np.ndarray, item: Mapping[str, Any]) -> np.ndarray:
            order = np.argsort(item["v"])
            vv = np.asarray(item["v"], dtype=np.float64)[order]
            uu, keep = np.unique(vv, return_index=True)
            return np.interp(grid, uu, np.asarray(values, dtype=np.float64)[order][keep])
        ea, eb = interp(a["residual"], a), interp(b["residual"], b)
        delta = ea - eb
        row = {"row_type": "pair", "coordinate_system": "sensor_v_overlap", "bin_scale": "1px_v", "frame_left": left, "frame_right": right, "matched_bin_count": int(len(grid)), "matched_point_count": int(len(grid)), "residual_delta_bias_mm": float(np.mean(delta)), "residual_delta_rmse_mm": float(np.sqrt(np.mean(delta * delta))), "residual_delta_p95_abs_mm": float(np.percentile(np.abs(delta), 95)), "cross_frame_variance_mm2": float(np.mean(delta * delta) / 2.0), "within_frame_variance_mm2": math.nan, "cross_to_within_variance_ratio": math.nan}
        rows.append(row)
        pair_rows.append(row)
    if not pair_rows:
        raise RuntimeError("No sensor-v overlaps were available across FIT frames")
    rows.append({"row_type": "aggregate", "coordinate_system": "sensor_v_overlap", "bin_scale": "1px_v", "pair_count": len(pair_rows), "matched_bin_count_median": float(np.median([r["matched_bin_count"] for r in pair_rows])), "matched_pair_point_count_median": float(np.median([r["matched_point_count"] for r in pair_rows])), "residual_delta_rmse_median_mm": float(np.median([r["residual_delta_rmse_mm"] for r in pair_rows])), "residual_delta_rmse_p95_mm": float(np.percentile([r["residual_delta_rmse_mm"] for r in pair_rows], 95)), "cross_frame_variance_mm2_median": float(np.median([r["cross_frame_variance_mm2"] for r in pair_rows])), "within_frame_variance_mm2_median": math.nan, "cross_to_within_variance_ratio_median": math.nan})


def symmetry_rows(processed: Mapping[str, Mapping[str, Any]], reps: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha = concat_arrays(processed, "alpha_truth_deg")
    delta = concat_arrays(processed, "delta_alpha_deg")
    surface = concat_arrays(processed, "e_surface_mm")
    lam = np.concatenate([np.asarray(processed[f]["residual"], dtype=np.float64) for f in sorted(processed, key=int)])
    for name, values, unit in (("alpha_truth_deg", alpha, "deg"), ("delta_alpha_deg", delta, "deg"), ("e_surface_mm", surface, "mm"), ("e_lambda_mm", lam, "mm")):
        stats = basic(values)
        rows.append({"row_type": "global_metric", "scope": "all_points", "metric": name, "unit": unit, "point_count": stats["count"], "value_mean": stats["mean"], "value_std": stats["std"], "value_rmse": stats["rmse"], "value_p95_abs": stats["p95_abs"], "value_min": stats["min"], "value_max": stats["max"]})
    frame_means = {f: float(np.mean(processed[f]["intrinsic"]["alpha_truth_deg"])) for f in processed}
    frame_delta_means = {f: float(np.mean(processed[f]["intrinsic"]["delta_alpha_deg"])) for f in processed}
    rows.append({"row_type": "global_metric", "scope": "frame_balanced", "metric": "alpha_truth_frame_mean", "unit": "deg", "frame_count": len(frame_means), "value_mean": float(np.mean(list(frame_means.values()))), "value_std": float(np.std(list(frame_means.values()), ddof=1)), "value_min": float(np.min(list(frame_means.values()))), "value_max": float(np.max(list(frame_means.values())))})
    rows.append({"row_type": "global_metric", "scope": "frame_balanced", "metric": "delta_alpha_frame_mean", "unit": "deg", "frame_count": len(frame_delta_means), "value_mean": float(np.mean(list(frame_delta_means.values()))), "value_std": float(np.std(list(frame_delta_means.values()), ddof=1)), "value_min": float(np.min(list(frame_delta_means.values()))), "value_max": float(np.max(list(frame_delta_means.values())))})
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        alpha_stats = basic(item["intrinsic"]["alpha_truth_deg"])
        delta_stats = basic(item["intrinsic"]["delta_alpha_deg"])
        lambda_stats = basic(item["residual"])
        surface_stats = basic(item["intrinsic"]["e_surface_mm"])
        rows.append({"row_type": "frame_summary", "scope": "frame", "frame_id": frame_id, "is_frame027": bool(item["is_frame027"]), "point_count": int(len(item["residual"])), "alpha_truth_mean_deg": alpha_stats["mean"], "alpha_truth_std_deg": alpha_stats["std"], "delta_alpha_mean_deg": delta_stats["mean"], "delta_alpha_std_deg": delta_stats["std"], "delta_alpha_p95_abs_deg": delta_stats["p95_abs"], "e_surface_mean_mm": surface_stats["mean"], "e_surface_rmse_mm": surface_stats["rmse"], "e_lambda_bias_mm": lambda_stats["mean"], "e_lambda_rmse_mm": lambda_stats["rmse"], "e_lambda_p95_abs_mm": lambda_stats["p95_abs"]})
    for predictor in ("phi_deg", "a_mm"):
        x = concat_arrays(processed, predictor)
        edges = predictor_edges(predictor, x)
        for outcome, unit in (("alpha_truth_deg", "deg"), ("delta_alpha_deg", "deg"), ("e_surface_mm", "mm"), ("e_lambda_mm", "mm")):
            y = concat_arrays(processed, outcome) if outcome != "e_lambda_mm" else lam
            rho, pvalue, n = safe_spearman(x, y)
            row: dict[str, Any] = {"row_type": "relationship", "scope": "all_points", "metric": "residual_vs_intrinsic", "predictor": predictor, "outcome": outcome, "unit": unit, "point_count": n, "spearman_rho": rho, "spearman_p_value": pvalue, "binned_explained_fraction": binned_ev(x, y, edges), "simple_linear_r2": linear_r2(x, y), "bootstrap_reps": reps, "bootstrap_seed": seed, **bootstrap_relationship(processed, predictor, outcome, edges, reps, seed)}
            rows.append(row)
    for outcome, unit in (("delta_alpha_deg", "deg"), ("e_lambda_mm", "mm")):
        y = delta if outcome == "delta_alpha_deg" else lam
        h = harmonic_fit(concat_arrays(processed, "phi_deg"), y)
        rows.append({"row_type": "harmonic", "scope": "all_points", "metric": "phi_harmonic", "predictor": "phi_deg", "outcome": outcome, "unit": unit, **h})
    # Frame-balanced bootstrap for global alpha mean/std and the second-harmonic fit.
    rng = np.random.default_rng(seed + 991)
    boot_mean: list[float] = []
    boot_std: list[float] = []
    boot_h2: list[float] = []
    frames = sorted(processed, key=int)
    for _ in range(int(reps)):
        sampled = rng.integers(0, len(frames), size=len(frames))
        av = np.concatenate([processed[frames[i]]["intrinsic"]["alpha_truth_deg"] for i in sampled])
        pv = np.concatenate([processed[frames[i]]["intrinsic"]["phi_deg"] for i in sampled])
        dv = np.concatenate([processed[frames[i]]["intrinsic"]["delta_alpha_deg"] for i in sampled])
        boot_mean.append(float(np.mean(av)))
        boot_std.append(float(np.std(av, ddof=1)))
        boot_h2.append(harmonic_fit(pv, dv)["amplitude_2"])
    for metric, values, unit in (("alpha_truth_bootstrap_mean", boot_mean, "deg"), ("alpha_truth_bootstrap_std", boot_std, "deg"), ("delta_alpha_phi_second_harmonic_amplitude", boot_h2, "deg")):
        arr = np.asarray(values, dtype=np.float64)
        rows.append({"row_type": "bootstrap", "scope": "frame_resampled", "metric": metric, "unit": unit, "bootstrap_reps": reps, "bootstrap_seed": seed, "value_mean": float(np.mean(arr)), "value_ci_low": float(np.percentile(arr, 2.5)), "value_ci_high": float(np.percentile(arr, 97.5))})
    return rows


def plots(output: Path, processed: Mapping[str, Mapping[str, Any]], bins: Sequence[Mapping[str, Any]], consistency: Sequence[Mapping[str, Any]]) -> None:
    arrays = {key: concat_arrays(processed, key) for key in ("phi_deg", "a_mm", "delta_alpha_deg", "e_surface_mm")}
    arrays["e_lambda_mm"] = np.concatenate([np.asarray(processed[f]["residual"], dtype=np.float64) for f in sorted(processed, key=int)])
    n = len(arrays["phi_deg"])
    rng = np.random.default_rng(6)
    idx = np.arange(n) if n <= 14000 else np.sort(rng.choice(n, 14000, replace=False))
    is027 = np.concatenate([np.repeat(processed[f]["is_frame027"], len(processed[f]["residual"])) for f in sorted(processed, key=int)])[idx]
    colors = np.where(is027, "#c53030", "#2b6cb0")
    for name, xlabel, ylabel, fname in (("phi_deg", "cone azimuth phi / deg", "delta_alpha / deg", "delta_alpha_vs_phi.png"), ("a_mm", "cone axial coordinate a / mm", "delta_alpha / deg", "delta_alpha_vs_a.png")):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(arrays[name][idx], arrays["delta_alpha_deg"][idx], s=4, alpha=0.16, c=colors)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output / fname, dpi=180); plt.close(fig)
    # Heatmaps use the medium scale and show bin means only; no surface correction is fitted.
    medium = [r for r in bins if r.get("bin_scale") == "medium"]
    for outcome, field, fname, title in (("delta_alpha", "e_lambda_mean_mm", "e_lambda_heatmap_phi_a.png", "e_lambda mean in cone intrinsic bins"), ("delta_alpha", "e_lambda_mean_mm", "delta_alpha_heatmap_phi_a.png", "delta_alpha / phi,a diagnostic (e_lambda proxy)")):
        # The first plot is e_lambda; the second is rebuilt from point data below.
        if fname.startswith("delta_alpha"):
            values = concat_arrays(processed, "delta_alpha_deg")
            zfield = "delta_alpha"
        else:
            values = arrays["e_lambda_mm"]
            zfield = "e_lambda"
        phi_edges = np.linspace(-180, 180, 19)
        a_edges = predictor_edges("a_mm", arrays["a_mm"], 18)
        pbin = np.digitize(arrays["phi_deg"], phi_edges[1:-1]); abin = np.digitize(arrays["a_mm"], a_edges[1:-1])
        grid = np.full((len(a_edges)-1, len(phi_edges)-1), np.nan)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                m = (abin == i) & (pbin == j)
                if np.any(m): grid[i, j] = float(np.mean(values[m]))
        fig, ax = plt.subplots(figsize=(11, 5.5)); im = ax.imshow(grid, origin="lower", aspect="auto", extent=[-180,180,a_edges[0],a_edges[-1]], cmap="coolwarm"); fig.colorbar(im, ax=ax, label=("delta_alpha / deg" if zfield == "delta_alpha" else "e_lambda / mm")); ax.set(xlabel="phi / deg", ylabel="a / mm", title=title); fig.tight_layout(); fig.savefig(output / fname, dpi=180); plt.close(fig)
    labels, values = [], []
    for row in consistency:
        if row.get("row_type") != "aggregate": continue
        labels.append(f"{row.get('coordinate_system')}\n{row.get('bin_scale')}")
        values.append(float(row.get("residual_delta_rmse_median_mm", math.nan)))
    fig, ax = plt.subplots(figsize=(9,5)); ax.bar(np.arange(len(values)), values, color="#4a5568"); ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right"); ax.set_ylabel("median matched residual delta RMSE / mm"); ax.set_title("Cross-frame consistency: sensor overlap vs cone intrinsic"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(output / "cross_frame_consistency_sensor_vs_intrinsic.png", dpi=180); plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    x = finite(value)
    return "n/a" if not math.isfinite(x) else f"{x:.{digits}f}"


def classify(
    frame_rows: Sequence[Mapping[str, Any]], relationship: Sequence[Mapping[str, Any]], symmetry: Sequence[Mapping[str, Any]], consistency: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pnp_max = max(float(r["pnp_rmse_px"]) for r in frame_rows)
    z_max = max(float(r["board_z_rmse_mm"]) for r in frame_rows)
    sensor_aggregate = next((r for r in consistency if r.get("coordinate_system") == "sensor_v_overlap" and r.get("row_type") == "aggregate"), None)
    intrinsic_aggs = [r for r in consistency if r.get("coordinate_system") == "cone_intrinsic" and r.get("row_type") == "aggregate"]
    best_intrinsic = min(intrinsic_aggs, key=lambda r: float(r["residual_delta_rmse_median_mm"])) if intrinsic_aggs else None
    sensor_rmse = float(sensor_aggregate["residual_delta_rmse_median_mm"]) if sensor_aggregate else math.nan
    intrinsic_rmse = float(best_intrinsic["residual_delta_rmse_median_mm"]) if best_intrinsic else math.nan
    improvement = 1.0 - intrinsic_rmse / sensor_rmse if math.isfinite(sensor_rmse) and sensor_rmse > 0 and math.isfinite(intrinsic_rmse) else math.nan
    rel_delta_phi = next((r for r in relationship if r.get("predictor") == "phi_deg" and r.get("outcome") == "delta_alpha_deg"), {})
    rel_delta_a = next((r for r in relationship if r.get("predictor") == "a_mm" and r.get("outcome") == "delta_alpha_deg"), {})
    stable_phi = bool(abs(float(rel_delta_phi.get("spearman_rho", math.nan))) >= .20 and float(rel_delta_phi.get("bootstrap_rho_ci_low", math.nan)) * float(rel_delta_phi.get("bootstrap_rho_ci_high", math.nan)) > 0 and float(rel_delta_phi.get("bootstrap_ev_ci_low", math.nan)) > .02)
    stable_a = bool(abs(float(rel_delta_a.get("spearman_rho", math.nan))) >= .20 and float(rel_delta_a.get("bootstrap_rho_ci_low", math.nan)) * float(rel_delta_a.get("bootstrap_rho_ci_high", math.nan)) > 0 and float(rel_delta_a.get("bootstrap_ev_ci_low", math.nan)) > .02)
    intrinsic_better = bool(math.isfinite(improvement) and improvement >= .20)
    geometry_ok = bool(pnp_max <= .40 and z_max <= .20 and len(frame_rows) >= 20 and best_intrinsic is not None and int(best_intrinsic.get("pair_count", 0)) >= 5)
    if not geometry_ok:
        verdict = "D. INSUFFICIENT"
    elif intrinsic_better and (stable_phi or stable_a):
        verdict = "A. STRONG"
    elif intrinsic_better or stable_phi or stable_a:
        verdict = "B. MODERATE"
    else:
        verdict = "C. WEAK"
    harmonic = next((r for r in symmetry if r.get("row_type") == "harmonic" and r.get("outcome") == "delta_alpha_deg"), {})
    alpha_global = next((r for r in symmetry if r.get("row_type") == "global_metric" and r.get("metric") == "alpha_truth_deg" and r.get("scope") == "all_points"), {})
    alpha_std = float(alpha_global.get("value_std", math.nan))
    harmonic_r2 = float(harmonic.get("r2", math.nan))
    harmonic_amp2 = float(harmonic.get("amplitude_2", math.nan))
    harmonic_identifiable = bool(harmonic.get("harmonic_identifiable", False))
    if harmonic_identifiable and math.isfinite(harmonic_r2) and harmonic_r2 >= .10 and math.isfinite(harmonic_amp2) and harmonic_amp2 > .25 * max(alpha_std, 1.0e-12):
        shape = "elliptical-like"
    elif (stable_phi or stable_a) and math.isfinite(harmonic_r2) and harmonic_r2 < .10:
        shape = "non-elliptical structured surface"
    elif math.isfinite(alpha_std) and alpha_std < .02:
        shape = "circular"
    else:
        shape = "inconsistent truth"
    if shape == "inconsistent truth" and geometry_ok:
        shape = "non-elliptical structured surface" if (stable_phi or stable_a or intrinsic_better) else "circular"
    if not geometry_ok:
        next_step = "re-audit truth construction"
    elif intrinsic_better or stable_phi or stable_a:
        next_step = "physical surface model"
    else:
        next_step = "re-audit truth construction"
    return {"verdict": verdict, "geometry_ok": geometry_ok, "max_pnp_rmse_px": pnp_max, "max_board_z_rmse_mm": z_max, "sensor_overlap_median_delta_rmse_mm": sensor_rmse, "best_intrinsic_scale": best_intrinsic.get("bin_scale") if best_intrinsic else "", "best_intrinsic_median_delta_rmse_mm": intrinsic_rmse, "intrinsic_improvement_vs_sensor": improvement, "stable_azimuth_delta_alpha": stable_phi, "stable_axial_delta_alpha": stable_a, "delta_alpha_phi_rho": float(rel_delta_phi.get("spearman_rho", math.nan)), "delta_alpha_a_rho": float(rel_delta_a.get("spearman_rho", math.nan)), "harmonic_phi_identifiable": harmonic_identifiable, "harmonic_phi_span_deg": float(harmonic.get("phi_span_deg", math.nan)), "harmonic_phi_condition_number": float(harmonic.get("design_condition_number", math.nan)), "shape_hypothesis": shape, "frame_sampling_explains_effect": intrinsic_better, "next_step": next_step}


def render_report(data_root: Path, frame_rows: Sequence[Mapping[str, Any]], relationship: Sequence[Mapping[str, Any]], symmetry: Sequence[Mapping[str, Any]], consistency: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], frozen_info: Mapping[str, Any], point_count: int, reps: int, basis: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    d, e1, e2 = basis
    alpha_model = finite(frozen_info.get("half_apex_angle_deg", math.nan))
    alpha = next(r for r in symmetry if r.get("row_type") == "global_metric" and r.get("metric") == "alpha_truth_deg" and r.get("scope") == "all_points")
    delta = next(r for r in symmetry if r.get("row_type") == "global_metric" and r.get("metric") == "delta_alpha_deg" and r.get("scope") == "all_points")
    phi = next(r for r in relationship if r.get("predictor") == "phi_deg" and r.get("outcome") == "delta_alpha_deg")
    aa = next(r for r in relationship if r.get("predictor") == "a_mm" and r.get("outcome") == "delta_alpha_deg")
    lambda_phi = next(r for r in relationship if r.get("predictor") == "phi_deg" and r.get("outcome") == "e_lambda_mm")
    lambda_a = next(r for r in relationship if r.get("predictor") == "a_mm" and r.get("outcome") == "e_lambda_mm")
    harmonic = next(r for r in symmetry if r.get("row_type") == "harmonic" and r.get("outcome") == "delta_alpha_deg")
    target_frame = next(r for r in symmetry if r.get("row_type") == "frame_summary" and r.get("frame_id") == TARGET_FRAME)
    lines = ["# Task 6B — Cone-intrinsic residual audit", "", f"`INTRINSIC_SURFACE_STRUCTURE = {decision['verdict']}`", "", "## Scope and frozen-data boundary", "", f"- FIT-only frames: `001–018`, `025–036` ({len(frame_rows)} frames, {point_count} valid points); frame `027` retained and marked.", f"- Only explicit FIT triplets under `{data_root}/fit` and `{data_root}/fit_edge_extension/fit` were opened. Validation 019–024 and 037–040 were not read.", "- No Cone was fitted/refit, no Elliptical Cone was fitted, no correction/LUT was built, no frame was deleted, and the existing Steger/PnP path was unchanged.", f"- Frozen provenance SHA-256: `{frozen_info['provenance_sha256']}`; formal Cone SHA-256: `{frozen_info['formal_cone_sha256']}`; frame bootstrap B={reps}, seed={BOOTSTRAP_SEED}.", "", "## Cone-intrinsic convention", "", "- `q = P_truth − apex`, `a = q·d`, `r = ||q − a d||`, `phi = atan2(q·e2,q·e1)` in degrees, and `alpha_truth = atan2(r,a)`.", f"- Frozen model half-angle: **{fmt(alpha_model,5)}°**; the exact per-point `alpha_model_deg` is in `cone_intrinsic_points.csv`.", f"- Fixed basis (camera coordinates): d=({fmt(d[0],6)}, {fmt(d[1],6)}, {fmt(d[2],6)}), e1=({fmt(e1[0],6)}, {fmt(e1[1],6)}, {fmt(e1[2],6)}), e2=({fmt(e2[0],6)}, {fmt(e2[1],6)}, {fmt(e2[2],6)}).", "- `delta_alpha = alpha_truth − alpha_model`; `e_surface` uses the CircularConeModel signed normal-distance formula; `e_lambda = lambda_truth − lambda_model`.", "", "## Plane/PnP sanity", "", f"- Maximum PnP RMSE: **{fmt(decision['max_pnp_rmse_px'])} px**; maximum board-Z RMSE: **{fmt(decision['max_board_z_rmse_mm'],6)} mm**; geometry gate: **{decision['geometry_ok']}**.", f"- Global alpha_truth: mean **{fmt(alpha['value_mean'],5)}°**, std **{fmt(alpha['value_std'],5)}°**, range **{fmt(float(alpha['value_max'])-float(alpha['value_min']),5)}°**; delta_alpha std **{fmt(delta['value_std'],5)}°**.", "", "## Intrinsic residual relationships", "", "| predictor → outcome | Spearman rho | binned EV | bootstrap rho 95% CI | bootstrap EV 95% CI |", "|---|---:|---:|---:|---:|"]
    for row in relationship:
        if row.get("outcome") == "delta_alpha_deg":
            lines.append(f"| {row['predictor']} → delta_alpha | {fmt(row.get('spearman_rho'),5)} | {fmt(row.get('binned_explained_fraction'),5)} | [{fmt(row.get('bootstrap_rho_ci_low'),5)}, {fmt(row.get('bootstrap_rho_ci_high'),5)}] | [{fmt(row.get('bootstrap_ev_ci_low'),5)}, {fmt(row.get('bootstrap_ev_ci_high'),5)}] |")
    lines += ["", f"- e_lambda vs phi: rho **{fmt(lambda_phi.get('spearman_rho'),5)}**, binned EV **{fmt(lambda_phi.get('binned_explained_fraction'),5)}**; e_lambda vs a: rho **{fmt(lambda_a.get('spearman_rho'),5)}**, binned EV **{fmt(lambda_a.get('binned_explained_fraction'),5)}**.", f"- Harmonic diagnostic delta_alpha(phi): R² **{fmt(harmonic.get('r2'),5)}**, first-harmonic amplitude **{fmt(harmonic.get('amplitude_1'),5)}°**, second-harmonic amplitude **{fmt(harmonic.get('amplitude_2'),5)}°**; observed phi span **{fmt(harmonic.get('phi_span_deg'),3)}°**, design condition **{fmt(harmonic.get('design_condition_number'),1)}**, identifiable **{harmonic.get('harmonic_identifiable', False)}**. Because the sampled azimuth span is limited, this harmonic fit is diagnostic only and is not interpreted as an elliptical structure or correction.", "", "## Cross-frame consistency", "", "| coordinate system | scale | pair count | median matched delta RMSE / mm | median cross/within variance ratio |", "|---|---|---:|---:|---:|"]
    for row in consistency:
        if row.get("row_type") == "aggregate":
            lines.append(f"| {row.get('coordinate_system')} | {row.get('bin_scale')} | {row.get('pair_count')} | {fmt(row.get('residual_delta_rmse_median_mm'))} | {fmt(row.get('cross_to_within_variance_ratio_median'))} |")
    lines += ["", f"- Frame 027 (retained separately): alpha_truth mean **{fmt(target_frame.get('alpha_truth_mean_deg'),5)}°**, delta_alpha mean **{fmt(target_frame.get('delta_alpha_mean_deg'),5)}°**, e_lambda bias **{fmt(target_frame.get('e_lambda_bias_mm'))} mm**, e_lambda RMSE **{fmt(target_frame.get('e_lambda_rmse_mm'))} mm**.", f"- Best intrinsic scale: **{decision['best_intrinsic_scale']}**, median matched delta RMSE **{fmt(decision['best_intrinsic_median_delta_rmse_mm'])} mm**; sensor-v overlap baseline **{fmt(decision['sensor_overlap_median_delta_rmse_mm'])} mm**; relative improvement **{fmt(decision['intrinsic_improvement_vs_sensor'] * 100,1)}%**.", f"- Cross-frame sampling explanation gate: **{decision['frame_sampling_explains_effect']}**. Intrinsic matching is considered an improvement only when the best scale reduces the sensor-v overlap delta by at least 20%; all reported scales remain visible in the CSV.", "", "## Answers", "", f"1. Residual is more stable in Cone intrinsic coordinates than sensor-v: **{decision['frame_sampling_explains_effect']}**.", f"2. Stable azimuth-dependent cone-angle deviation: **{decision['stable_azimuth_delta_alpha']}** (rho={fmt(decision['delta_alpha_phi_rho'],5)}; frame-balanced bootstrap and binned checks are in `circular_symmetry_audit.csv`).", f"3. Frame effect is mainly different-frame sampling of distinct Cone surface regions: **{decision['frame_sampling_explains_effect']}**.", f"4. Observed deviation is classified as **{decision['shape_hypothesis']}**; no Elliptical Cone was fitted.", f"5. Next step: **{decision['next_step']}**.", "", "## Conclusion", "", f"`INTRINSIC_SURFACE_STRUCTURE = {decision['verdict']}`.", "The classification is descriptive: it combines multiple intrinsic bin scales, point-level Spearman/binned statistics, frame-resampled bootstrap intervals, and pairwise cross-frame matching. It is not a correction model.", "", "Generated figures: `delta_alpha_vs_phi.png`, `delta_alpha_vs_a.png`, `delta_alpha_heatmap_phi_a.png`, `e_lambda_heatmap_phi_a.png`, and `cross_frame_consistency_sensor_vs_intrinsic.png`.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root, output_dir = args.data_root.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    groups = board_audit.inventory_fit(data_root)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, frozen_info = board_audit.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    frozen_info["half_apex_angle_deg"] = float(frozen_model["half_apex_angle_deg"])
    board_summaries, processed = board_audit.process_groups_board(groups, intrinsics, calibration, reconstruction_params, frozen_model)
    _, basis = add_intrinsic_coordinates(processed, frozen_model)
    frame_rows = make_frame_summary(board_summaries, processed)
    point_rows = make_point_rows(processed)
    bins, meta = intrinsic_bins(processed)
    consistency = intrinsic_pair_rows(processed, meta)
    add_sensor_overlap_rows(processed, consistency)
    relationship = relationship_rows(processed, int(args.bootstrap_reps), int(args.seed))
    symmetry = symmetry_rows(processed, int(args.bootstrap_reps), int(args.seed))
    decision = classify(frame_rows, relationship, symmetry, consistency)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed.write_csv(output_dir / "cone_intrinsic_points.csv", point_rows)
    fixed.write_csv(output_dir / "intrinsic_bin_summary.csv", bins)
    fixed.write_csv(output_dir / "intrinsic_cross_frame_consistency.csv", consistency)
    fixed.write_csv(output_dir / "circular_symmetry_audit.csv", list(symmetry) + [{"row_type": "decision", **decision}])
    plots(output_dir, processed, bins, consistency)
    (output_dir / "report.md").write_text(render_report(data_root, frame_rows, relationship, symmetry, consistency, decision, frozen_info, len(point_rows), int(args.bootstrap_reps), basis), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "frame_count": len(frame_rows), "point_count": len(point_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
