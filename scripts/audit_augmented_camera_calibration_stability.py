#!/usr/bin/env python3
"""Task 6H-1: M0/M1 camera-calibration stability A/B audit.

M0 is chess 001--018.  M1-core adds 026/027/028/035 and M1-full adds
031/033/032/030.  Each candidate is calibrated with the formal model and
flags, but the formal K/D file is never overwritten.  Laser propagation uses
only the existing FIT triplets 001--018 and 025--036; Validation is never
enumerated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve()
TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
for _path in (SCRIPT.parent, WORKSPACE_ROOT / "calibration" / "src", WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_board_coordinate_residual as board_audit  # noqa: E402
import audit_camera_calibration_fixed_coverage as fixed_cov  # noqa: E402
import audit_camera_pose_observability as pose_audit  # noqa: E402
import audit_edge_extension_camera_coverage as edge_cov  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import audit_intrinsics_truth_stability as task6e  # noqa: E402


DEFAULT_CALIBRATION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
DEFAULT_EXTENSION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension" / "fit"
DEFAULT_FORMAL_INTRINSICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
DEFAULT_FORMAL_FIT_METRICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "fit_images.csv"
DEFAULT_DATA_ROOT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "augmented_camera_calibration_stability"
DEFAULT_MEASUREMENT_CONFIG = fixed.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FROZEN_PROVENANCE = fixed.DEFAULT_FROZEN_PROVENANCE
DEFAULT_FORMAL_CONE = fixed.DEFAULT_FORMAL_CONE

M0_IDS = tuple(f"{i:03d}" for i in range(1, 19))
CORE_EXTENSION_IDS = ("026", "027", "028", "035")
FULL_EXTENSION_IDS = ("031", "033", "032", "030")
M1_CORE_IDS = M0_IDS + CORE_EXTENSION_IDS
M1_FULL_IDS = M1_CORE_IDS + FULL_EXTENSION_IDS
DATASET_IDS = {"M0": M0_IDS, "M1-core": M1_CORE_IDS, "M1-full": M1_FULL_IDS}
REGIONS = ("all", "top", "middle", "bottom")
MC_REPS_DEFAULT = 1000
MC_SEED_DEFAULT = 20260817


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-fit-dir", type=Path, default=DEFAULT_CALIBRATION_FIT)
    p.add_argument("--extension-fit-dir", type=Path, default=DEFAULT_EXTENSION_FIT)
    p.add_argument("--formal-intrinsics", type=Path, default=DEFAULT_FORMAL_INTRINSICS)
    p.add_argument("--formal-fit-metrics", type=Path, default=DEFAULT_FORMAL_FIT_METRICS)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    p.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    p.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    p.add_argument("--mc-reps", type=int, default=MC_REPS_DEFAULT)
    p.add_argument("--seed", type=int, default=MC_SEED_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dataset_observations(dataset_ids: Sequence[str], baseline_fit: Path, extension_fit: Path, formal_metrics: Mapping[str, Mapping[str, str]]) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    baseline_obs, image_size = edge_cov.read_observations(baseline_fit, [x for x in dataset_ids if int(x) <= 18])
    extension_ids = [x for x in dataset_ids if int(x) >= 25]
    extension_obs, extension_size = edge_cov.read_observations(extension_fit, extension_ids) if extension_ids else ([], image_size)
    if image_size != extension_size:
        raise RuntimeError(f"Camera image size mismatch baseline={image_size}, extension={extension_size}")
    observations = baseline_obs + extension_obs
    if [str(obs["frame_id"]) for obs in observations] != list(dataset_ids):
        raise RuntimeError(f"Dataset observation order mismatch: {dataset_ids}")
    return observations, image_size


def parameter_values(k: np.ndarray, d: np.ndarray) -> dict[str, float]:
    x = np.asarray(d, dtype=np.float64).reshape(-1)
    return {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2]),
            "k1": float(x[0]), "k2": float(x[1]), "p1": float(x[2]), "p2": float(x[3]), "k3": float(x[4]) if len(x) > 4 else 0.0}


def candidate_from_row(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray([[float(row["fx"]), 0.0, float(row["cx"])], [0.0, float(row["fy"]), float(row["cy"])], [0.0, 0.0, 1.0]], dtype=np.float64)
    d = np.asarray([[float(row["k1"])], [float(row["k2"])], [float(row["p1"])], [float(row["p2"])], [float(row.get("k3", 0.0))]], dtype=np.float64)
    return k, d


def solve_dataset(dataset: str, observations: Sequence[Mapping[str, Any]], obj: np.ndarray, image_size: tuple[int, int], formal_k: np.ndarray, formal_d: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    indices = list(range(len(observations)))
    fit_rms, k, d = task6e.calibrate_candidate(observations, indices, obj, image_size)
    base = parameter_values(formal_k, formal_d)
    values = parameter_values(k, d)
    summary: dict[str, Any] = {"row_type": "dataset_summary", "dataset": dataset, "frame_count": len(observations), "frame_ids": ";".join(str(x["frame_id"]) for x in observations),
                               "global_reprojection_rmse_px": fit_rms, "status": "ok"}
    for name in task6e.PARAMETER_NAMES:
        summary[name] = values[name]
        summary[f"delta_{name}"] = values[name] - base[name]
        summary[f"delta_{name}_pct"] = 100.0 * (values[name] - base[name]) / base[name] if base[name] else math.nan
    return summary, k, d


def per_frame_rmse(dataset: str, observations: Sequence[Mapping[str, Any]], k: np.ndarray, d: np.ndarray, obj: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        pose = task6e.solve_pose(np.asarray(obs["corners"], dtype=np.float64), k, d, obj)
        rows.append({"row_type": "per_frame_rmse", "dataset": dataset, "frame_id": obs["frame_id"], "pnp_reprojection_rmse_px": pose["rmse_px"], "corner_count": len(obs["corners"]), "detection_method": obs["detection_method"], "validation_opened": False})
    return rows


def propagate_summary(k: np.ndarray, d: np.ndarray, processed: Mapping[str, Mapping[str, Any]], obj: np.ndarray) -> dict[str, float]:
    per_frame: list[float] = []
    pooled: list[np.ndarray] = []
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        uv, delta, info = task6e.propagate_lambda(item, k, d, obj)
        valid = np.asarray(info["valid"], dtype=bool) & np.isfinite(delta)
        values = delta[valid]
        if len(values):
            pooled.append(values)
            per_frame.append(float(np.percentile(np.abs(values), 95)))
    pooled_values = np.concatenate(pooled) if pooled else np.empty(0)
    return {"candidate_global_p95_abs_delta_lambda_mm": float(np.percentile(np.abs(pooled_values), 95)) if len(pooled_values) else math.nan,
            "candidate_frame_p95_median_mm": float(np.median(per_frame)) if per_frame else math.nan,
            "candidate_frame_p95_p90_mm": float(np.percentile(per_frame, 90)) if per_frame else math.nan,
            "candidate_frame_p95_p95_mm": float(np.percentile(per_frame, 95)) if per_frame else math.nan,
            "candidate_frame_p95_max_mm": float(np.max(per_frame)) if per_frame else math.nan,
            "candidate_frame_count": len(per_frame)}


def candidate_lambda_cache(k: np.ndarray, d: np.ndarray, processed: Mapping[str, Mapping[str, Any]], obj: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Cache a candidate's own ray-plane lambda for centered MC uncertainty.

    The ordinary candidate metrics are deliberately referenced to the formal
    M0 truth.  For corner-noise MC, however, uncertainty must be centered on
    the corresponding full-data candidate so a fixed M1 K/D shift is not
    mistaken for random calibration uncertainty.
    """
    cache: dict[str, dict[str, np.ndarray]] = {}
    for frame_id, item in processed.items():
        _uv, _delta, info = task6e.propagate_lambda(item, k, d, obj)
        cache[str(frame_id)] = {
            "lambda": np.asarray(info["candidate_lambda"], dtype=np.float64),
            "valid": np.asarray(info["valid"], dtype=bool),
        }
    return cache


def centered_propagate_summary(k: np.ndarray, d: np.ndarray, processed: Mapping[str, Mapping[str, Any]], obj: np.ndarray,
                               reference: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, float]:
    """Summarize candidate lambda changes around its own full-data solution."""
    per_frame: list[float] = []
    pooled: list[np.ndarray] = []
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        uv, _delta, info = task6e.propagate_lambda(item, k, d, obj)
        ref = reference[str(frame_id)]
        valid = np.asarray(info["valid"], dtype=bool) & np.asarray(ref["valid"], dtype=bool)
        candidate = np.asarray(info["candidate_lambda"], dtype=np.float64)
        delta = candidate - np.asarray(ref["lambda"], dtype=np.float64)
        valid &= np.isfinite(delta) & np.isfinite(uv[:, 0])
        values = delta[valid]
        if len(values):
            pooled.append(values)
            per_frame.append(float(np.percentile(np.abs(values), 95)))
    pooled_values = np.concatenate(pooled) if pooled else np.empty(0)
    return {
        "mc_centered_global_p95_abs_delta_lambda_mm": float(np.percentile(np.abs(pooled_values), 95)) if len(pooled_values) else math.nan,
        "mc_centered_frame_p95_median_mm": float(np.median(per_frame)) if per_frame else math.nan,
        "mc_centered_frame_p95_p90_mm": float(np.percentile(per_frame, 90)) if per_frame else math.nan,
        "mc_centered_frame_p95_p95_mm": float(np.percentile(per_frame, 95)) if per_frame else math.nan,
        "mc_centered_frame_p95_max_mm": float(np.max(per_frame)) if per_frame else math.nan,
    }


def dual_propagate_summary(k: np.ndarray, d: np.ndarray, processed: Mapping[str, Mapping[str, Any]], obj: np.ndarray,
                           reference: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, float]:
    """Compute formal-referenced and candidate-centered metrics in one pass."""
    raw_per_frame: list[float] = []
    raw_pooled: list[np.ndarray] = []
    centered_per_frame: list[float] = []
    centered_pooled: list[np.ndarray] = []
    for frame_id in sorted(processed, key=int):
        item = processed[frame_id]
        _uv, _delta, info = task6e.propagate_lambda(item, k, d, obj)
        valid = np.asarray(info["valid"], dtype=bool)
        candidate = np.asarray(info["candidate_lambda"], dtype=np.float64)
        formal = np.asarray(item["truth_points_raw"], dtype=np.float64)[:, 2]
        raw_delta = candidate - formal
        raw_valid = valid & np.isfinite(raw_delta)
        raw_values = raw_delta[raw_valid]
        if len(raw_values):
            raw_pooled.append(raw_values)
            raw_per_frame.append(float(np.percentile(np.abs(raw_values), 95)))
        ref = reference[str(frame_id)]
        centered_delta = candidate - np.asarray(ref["lambda"], dtype=np.float64)
        centered_valid = valid & np.asarray(ref["valid"], dtype=bool) & np.isfinite(centered_delta)
        centered_values = centered_delta[centered_valid]
        if len(centered_values):
            centered_pooled.append(centered_values)
            centered_per_frame.append(float(np.percentile(np.abs(centered_values), 95)))
    raw_values = np.concatenate(raw_pooled) if raw_pooled else np.empty(0)
    centered_values = np.concatenate(centered_pooled) if centered_pooled else np.empty(0)
    return {
        "candidate_global_p95_abs_delta_lambda_mm": float(np.percentile(np.abs(raw_values), 95)) if len(raw_values) else math.nan,
        "candidate_frame_p95_median_mm": float(np.median(raw_per_frame)) if raw_per_frame else math.nan,
        "candidate_frame_p95_p90_mm": float(np.percentile(raw_per_frame, 90)) if raw_per_frame else math.nan,
        "candidate_frame_p95_p95_mm": float(np.percentile(raw_per_frame, 95)) if raw_per_frame else math.nan,
        "candidate_frame_p95_max_mm": float(np.max(raw_per_frame)) if raw_per_frame else math.nan,
        "candidate_frame_count": len(raw_per_frame),
        "mc_centered_global_p95_abs_delta_lambda_mm": float(np.percentile(np.abs(centered_values), 95)) if len(centered_values) else math.nan,
        "mc_centered_frame_p95_median_mm": float(np.median(centered_per_frame)) if centered_per_frame else math.nan,
        "mc_centered_frame_p95_p90_mm": float(np.percentile(centered_per_frame, 90)) if centered_per_frame else math.nan,
        "mc_centered_frame_p95_p95_mm": float(np.percentile(centered_per_frame, 95)) if centered_per_frame else math.nan,
        "mc_centered_frame_p95_max_mm": float(np.max(centered_per_frame)) if centered_per_frame else math.nan,
    }


def loo_rows(dataset: str, observations: Sequence[Mapping[str, Any]], obj: np.ndarray, image_size: tuple[int, int], processed: Mapping[str, Mapping[str, Any]], formal_k: np.ndarray, formal_d: np.ndarray) -> list[dict[str, Any]]:
    base = parameter_values(formal_k, formal_d)
    rows: list[dict[str, Any]] = []
    for omit_index, omit_obs in enumerate(observations):
        indices = [i for i in range(len(observations)) if i != omit_index]
        row: dict[str, Any] = {"row_type": "loo", "dataset": dataset, "omitted_frame_id": omit_obs["frame_id"], "remaining_frame_count": len(indices), "status": "ok"}
        try:
            fit_rms, k, d = task6e.calibrate_candidate(observations, indices, obj, image_size)
            values = parameter_values(k, d)
            row["global_reprojection_rmse_px"] = fit_rms
            for name in task6e.PARAMETER_NAMES:
                row[name] = values[name]
                row[f"delta_{name}"] = values[name] - base[name]
                row[f"delta_{name}_pct"] = 100.0 * (values[name] - base[name]) / base[name] if base[name] else math.nan
            row.update(propagate_summary(k, d, processed, obj))
        except (cv2.error, RuntimeError, ValueError) as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)
    return rows


def noise_mc(dataset: str, observations: Sequence[Mapping[str, Any]], covariances: Sequence[Mapping[str, Any]], reps: int, seed: int,
             obj: np.ndarray, image_size: tuple[int, int], processed: Mapping[str, Mapping[str, Any]], formal_k: np.ndarray, formal_d: np.ndarray,
             reference: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    base = parameter_values(formal_k, formal_d)
    rows: list[dict[str, Any]] = []
    for rep in range(reps):
        candidate_id = f"{dataset}_mc_{rep + 1:04d}"
        noisy = fixed_cov.noisy_observations(observations, covariances, rng)
        row: dict[str, Any] = {"row_type": "corner_noise_mc", "dataset": dataset, "rep": rep + 1, "candidate_id": candidate_id, "frame_count": len(observations), "status": "ok"}
        try:
            fit_rms, k, d = task6e.calibrate_candidate(noisy, list(range(len(noisy))), obj, image_size)
            values = parameter_values(k, d)
            row["global_reprojection_rmse_px"] = fit_rms
            for name in task6e.PARAMETER_NAMES:
                row[name] = values[name]
                row[f"delta_{name}"] = values[name] - base[name]
                row[f"delta_{name}_pct"] = 100.0 * (values[name] - base[name]) / base[name] if base[name] else math.nan
            row.update(dual_propagate_summary(k, d, processed, obj, reference))
        except (cv2.error, RuntimeError, ValueError) as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)
    return rows


def coverage_comparison(observations_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]], formal_k: np.ndarray, formal_d: np.ndarray, obj: np.ndarray, image_size: tuple[int, int], formal_metrics: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    geometry: dict[str, list[dict[str, Any]]] = {}
    for dataset, observations in observations_by_dataset.items():
        geometry[dataset] = pose_audit.geometry_rows(observations, formal_k, formal_d, obj, image_size, formal_metrics)
    dimensions = {"depth": "board_center_z_mm", "tilt": "board_tilt_deg", "apparent_size": "apparent_bbox_area_fraction", "sensor_u": "image_center_u_norm", "sensor_v": "image_center_v_norm"}
    rows: list[dict[str, Any]] = []
    for dataset, values in geometry.items():
        valid = [r for r in values if finite(r.get("solvepnp_reprojection_rmse_px")) <= 0.40]
        for dimension, feature in dimensions.items():
            x = np.asarray([finite(r.get(feature)) for r in valid], dtype=float)
            rows.append({"row_type": "dataset_dimension", "dataset": dataset, "dimension": dimension, "feature": feature, "frame_count": len(x),
                         "min": float(np.min(x)), "max": float(np.max(x)), "range": float(np.ptp(x)), "std": float(np.std(x, ddof=1)),
                         "q10": float(np.percentile(x, 10)), "q90": float(np.percentile(x, 90))})
        z = np.asarray([finite(r.get("board_center_z_mm")) for r in valid], dtype=float)
        tilt = np.asarray([finite(r.get("board_tilt_deg")) for r in valid], dtype=float)
        area = np.asarray([finite(r.get("apparent_bbox_area_fraction")) for r in valid], dtype=float)
        rows.append({"row_type": "coupling", "dataset": dataset, "dimension": "tilt_depth", "feature": "spearman(tilt,depth)", "frame_count": len(valid), "spearman": float(spearmanr(tilt, z).statistic)})
        rows.append({"row_type": "coupling", "dataset": dataset, "dimension": "tilt_apparent_size", "feature": "spearman(tilt,apparent_size)", "frame_count": len(valid), "spearman": float(spearmanr(tilt, area).statistic)})
    return rows


def summarize_mc(rows: Sequence[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    values = [finite(r.get("candidate_global_p95_abs_delta_lambda_mm")) for r in rows if r.get("dataset") == dataset and r.get("status") == "ok"]
    centered_values = [finite(r.get("mc_centered_global_p95_abs_delta_lambda_mm")) for r in rows if r.get("dataset") == dataset and r.get("status") == "ok"]
    kstd = {name: float(np.std([finite(r.get(name)) for r in rows if r.get("dataset") == dataset and r.get("status") == "ok"], ddof=1)) for name in task6e.PARAMETER_NAMES}
    return {"dataset": dataset, "mc_success_count": len(values), "mc_global_p95_median_mm": float(np.median(values)) if values else math.nan,
            "mc_global_p95_p90_mm": float(np.percentile(values, 90)) if values else math.nan,
            "mc_global_p95_p95_mm": float(np.percentile(values, 95)) if values else math.nan,
            "mc_global_p95_max_mm": float(np.max(values)) if values else math.nan,
            "mc_centered_global_p95_median_mm": float(np.median(centered_values)) if centered_values else math.nan,
            "mc_centered_global_p95_p90_mm": float(np.percentile(centered_values, 90)) if centered_values else math.nan,
            "mc_centered_global_p95_p95_mm": float(np.percentile(centered_values, 95)) if centered_values else math.nan,
            "mc_centered_global_p95_max_mm": float(np.max(centered_values)) if centered_values else math.nan,
            **{f"mc_std_{name}": value for name, value in kstd.items()}}


def classify(m0_loo: Sequence[Mapping[str, Any]], core_loo: Sequence[Mapping[str, Any]], full_loo: Sequence[Mapping[str, Any]], mc_summary: Mapping[str, Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    def loo_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        x = [finite(r.get("candidate_frame_p95_median_mm")) for r in rows if r.get("status") == "ok"]
        return {"median": float(np.median(x)) if x else math.nan,
                "p90": float(np.percentile(x, 90)) if x else math.nan,
                "p95": float(np.percentile(x, 95)) if x else math.nan,
                "max": float(np.max(x)) if x else math.nan,
                "leverage_ratio": float(np.max(x) / np.median(x)) if x and np.median(x) > 0 else math.nan}
    s0 = loo_stats(m0_loo)
    sc = loo_stats(core_loo)
    sf = loo_stats(full_loo)
    m0 = s0["p95"]
    core = sc["p95"]
    full = sf["p95"]
    m0_mc = finite(mc_summary.get("M0", {}).get("mc_centered_global_p95_median_mm"))
    core_mc = finite(mc_summary.get("M1-core", {}).get("mc_centered_global_p95_median_mm"))
    full_mc = finite(mc_summary.get("M1-full", {}).get("mc_centered_global_p95_median_mm"))
    coupling = {str(r["dataset"]): finite(r.get("spearman")) for r in coverage if r.get("row_type") == "coupling" and r.get("dimension") == "tilt_depth"}
    improvement_core = (m0 - core) / m0 if m0 > 0 and math.isfinite(core) else math.nan
    improvement_full = (m0 - full) / m0 if m0 > 0 and math.isfinite(full) else math.nan
    if not math.isfinite(improvement_core) or not math.isfinite(improvement_full) or not math.isfinite(m0_mc):
        return "D. NEGATIVE", "none"
    # A negative result requires both a worse LOO tail and a worse centered
    # fixed-coverage MC uncertainty; a fixed M1-vs-M0 shift alone is reported
    # but is not itself random stability evidence.
    mc_worse_core = math.isfinite(core_mc) and core_mc > 1.10 * m0_mc
    mc_worse_full = math.isfinite(full_mc) and full_mc > 1.10 * m0_mc
    if improvement_core < -0.10 and improvement_full < -0.10 and mc_worse_core and mc_worse_full:
        return "D. NEGATIVE", "M0"
    coupling_improved = abs(coupling.get("M1-full", math.inf)) < abs(coupling.get("M0", math.inf)) * 0.9
    if improvement_core >= 0.25 and improvement_full >= 0.25 and not mc_worse_core and not mc_worse_full and coupling_improved:
        return "A. STRONG", "M1-full"
    if improvement_core >= 0.10 or improvement_full >= 0.10:
        recommendation = "M1-full" if math.isfinite(full) and math.isfinite(core) and full < core else "M1-core"
        return "B. MODERATE", recommendation
    return "C. WEAK", "M1-core" if math.isfinite(core) and core < m0 else "M0"


def render_report(path: Path, classification: str, recommendation: str, summaries: Mapping[str, Mapping[str, Any]], loo: Mapping[str, Sequence[Mapping[str, Any]]], coverage: Sequence[Mapping[str, Any]], extension_leverage: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Task 6H-1 — Augmented camera calibration stability A/B", "", f"`EDGE_EXTENSION_CAMERA_GAIN = {classification}`", f"推荐冻结 candidate：`{recommendation}`", "",
             "本审计只读取 M0 chess 001–018、M1 extension chess 026/027/028/035/031/033/032/030，以及激光 FIT 001–018/025–036。Validation 未打开；正式 K/D 文件未修改，未更换 distortion model，未拟合 Cone。", "",
             "## Fixed-coverage corner-noise MC", "", "Raw values are relative to formal M0 truth; centered values are relative to each dataset's own full-data candidate and are the stability comparison.", "", "| dataset | MC success | raw global P95 median (mm) | centered global P95 median (mm) | centered P95 tail (mm) | centered max (mm) | fx std (px) | k2 std |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("M0", "M1-core", "M1-full"):
        s = summaries.get(dataset, {})
        lines.append(f"| {dataset} | {s.get('mc_success_count','')} | {finite(s.get('mc_global_p95_median_mm')):.6g} | {finite(s.get('mc_centered_global_p95_median_mm')):.6g} | {finite(s.get('mc_centered_global_p95_p95_mm')):.6g} | {finite(s.get('mc_centered_global_p95_max_mm')):.6g} | {finite(s.get('mc_std_fx')):.6g} | {finite(s.get('mc_std_k2')):.6g} |")
    lines += ["", "## Camera-side LOO stability", "", "| dataset | LOO frame-P95 median across omissions (mm) | P90 | P95 | max |", "|---|---:|---:|---:|---:|"]
    for dataset in ("M0", "M1-core", "M1-full"):
        values = [finite(r.get("candidate_frame_p95_median_mm")) for r in loo.get(dataset, []) if r.get("status") == "ok"]
        lines.append(f"| {dataset} | {np.median(values) if values else math.nan:.6g} | {np.percentile(values,90) if values else math.nan:.6g} | {np.percentile(values,95) if values else math.nan:.6g} | {np.max(values) if values else math.nan:.6g} |")
    lines += ["", "The LOO tail and max quantify frame dependence; the raw M1-vs-M0 shift is not treated as MC uncertainty. M0 high-leverage omissions 002/001/010/003/017 and extension omissions are in `m0_m1_loo_stability.csv` and `extension_frame_leverage.csv`.", "", "## Coverage and coupling", "", "| dataset | dimension | range | min–max / Spearman |", "|---|---|---:|---:|"]
    for row in coverage:
        if row.get("row_type") == "dataset_dimension":
            lines.append(f"| {row.get('dataset')} | {row.get('dimension')} | {finite(row.get('range')):.6g} | {finite(row.get('min')):.6g}–{finite(row.get('max')):.6g} |")
        elif row.get("row_type") == "coupling":
            lines.append(f"| {row.get('dataset')} | {row.get('dimension')} | — | Spearman={finite(row.get('spearman')):.6g} |")
    lines += ["", "## 结论", "", "M1 的判断同时考虑 camera-side LOO、fixed-coverage corner-noise MC 和覆盖耦合；training reprojection RMSE 不是选择依据。若 M1 降低了单帧 leverage 但 tilt-depth 相关仍高，则 observability 仅部分改善，不能宣称 depth/tilt 已解耦。", "", "## 输出", "", "- `m0_m1_intrinsics_comparison.csv`", "- `m0_m1_loo_stability.csv`", "- `m0_m1_corner_mc.csv`", "- `extension_frame_leverage.csv`", "- `m0_m1_coverage_comparison.csv`", "- `provenance.json"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.mc_reps < 1:
        raise ValueError("--mc-reps must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    formal_k, formal_d, _ = task6e.load_formal_intrinsics(args.formal_intrinsics.resolve())
    formal_metrics = task6e.load_formal_fit_metrics(args.formal_fit_metrics.resolve())
    obj = task6e.object_points()
    observations_by_dataset: dict[str, list[dict[str, Any]]] = {}
    image_size: tuple[int, int] | None = None
    for dataset, ids in DATASET_IDS.items():
        observations, current_size = load_dataset_observations(ids, args.calibration_fit_dir.resolve(), args.extension_fit_dir.resolve(), formal_metrics)
        image_size = current_size if image_size is None else image_size
        if current_size != image_size:
            raise RuntimeError(f"Image size mismatch for {dataset}: {current_size} vs {image_size}")
        observations_by_dataset[dataset] = observations
    assert image_size is not None

    # Laser diagnostic FIT is processed exactly once with the formal runtime
    # intrinsics.  This opens only explicit FIT roots, never Validation.
    _, calibration, reconstruction_params, runtime_intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    if not np.allclose(runtime_intrinsics.camera_matrix, formal_k, rtol=0.0, atol=1e-8) or not np.allclose(runtime_intrinsics.dist_coeffs, formal_d.reshape(-1), rtol=0.0, atol=1e-8):
        raise RuntimeError("Formal K/D does not match the runtime FIT intrinsics")
    groups = board_audit.inventory_fit(args.data_root.resolve())
    frozen_model, frozen_info = board_audit.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    board_summaries, processed = board_audit.process_groups_board(groups, runtime_intrinsics, calibration, reconstruction_params, frozen_model)

    intrinsic_rows: list[dict[str, Any]] = []
    dataset_models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dataset_reference_lambdas: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for dataset, observations in observations_by_dataset.items():
        summary, k, d = solve_dataset(dataset, observations, obj, image_size, formal_k, formal_d)
        intrinsic_rows.append(summary)
        intrinsic_rows.extend(per_frame_rmse(dataset, observations, k, d, obj))
        dataset_models[dataset] = (k, d)
        dataset_reference_lambdas[dataset] = candidate_lambda_cache(k, d, processed, obj)
    write_csv(output / "m0_m1_intrinsics_comparison.csv", intrinsic_rows)

    loo_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, observations in observations_by_dataset.items():
        loo_by_dataset[dataset] = loo_rows(dataset, observations, obj, image_size, processed, formal_k, formal_d)
    all_loo = [row for rows in loo_by_dataset.values() for row in rows]
    write_csv(output / "m0_m1_loo_stability.csv", all_loo)
    extension_leverage = [row for row in all_loo if row.get("dataset") in {"M1-core", "M1-full"} and str(row.get("omitted_frame_id")) in set(CORE_EXTENSION_IDS + FULL_EXTENSION_IDS)]
    write_csv(output / "extension_frame_leverage.csv", extension_leverage)

    mc_rows: list[dict[str, Any]] = []
    mc_summaries: dict[str, dict[str, Any]] = {}
    for offset, (dataset, observations) in enumerate(observations_by_dataset.items()):
        covariances, _noise_rows = fixed_cov.pose_noise_covariances(observations, formal_k, formal_d, obj)
        rows = noise_mc(dataset, observations, covariances, args.mc_reps, args.seed + offset, obj, image_size, processed, formal_k, formal_d,
                        dataset_reference_lambdas[dataset])
        mc_rows.extend(rows)
        mc_summaries[dataset] = summarize_mc(rows, dataset)
    write_csv(output / "m0_m1_corner_mc.csv", mc_rows)
    coverage = coverage_comparison(observations_by_dataset, formal_k, formal_d, obj, image_size, formal_metrics)
    write_csv(output / "m0_m1_coverage_comparison.csv", coverage)
    classification, recommendation = classify(loo_by_dataset["M0"], loo_by_dataset["M1-core"], loo_by_dataset["M1-full"], mc_summaries, coverage)
    render_report(output / "report.md", classification, recommendation, mc_summaries, loo_by_dataset, coverage, extension_leverage)
    provenance = {"task": "6H-1", "validation_opened": False, "datasets": {key: list(value) for key, value in DATASET_IDS.items()},
                  "extension_not_used": ["025", "029", "034", "036"], "formal_intrinsics": str(args.formal_intrinsics.resolve()),
                  "calibration_flags": "CALIB_FIX_K3", "pnp_solver": "SOLVEPNP_ITERATIVE + solvePnPRefineLM", "corner_pipeline": "formal chess_calib.detect_corners",
                  "mc_reps": args.mc_reps, "mc_seed": args.seed, "noise_model": "per-frame centered formal-K/D PnP residual covariance; unchanged across M0/M1",
                  "laser_fit_frame_ids": [f"{i:03d}" for i in range(1, 19)] + [f"{i:03d}" for i in range(25, 37)], "laser_validation_opened": False,
                  "formal_kd_modified": False, "distortion_model_changed": False, "cone_refit": False, "steger_changed": False,
                  "classification": classification, "recommendation": recommendation, "frozen_provenance": frozen_info}
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EDGE_EXTENSION_CAMERA_GAIN = {classification}")
    print(f"RECOMMENDATION = {recommendation}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
