#!/usr/bin/env python3
"""Independent Validation for the frozen Operational-35 C1_4k model.

This script is deliberately evaluation-only.  It reuses the existing point-level
Validation artifact and the already frozen C0 prediction stored in that artifact;
it loads the frozen C1 JSON/LUT and never calls fit(), PCA, or any parameter
adjustment routine.  The C1 spline is evaluated after clipping raw ``s`` to the
persisted frozen domain, as required by the frozen model protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import BSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
PROJECT = ROOT / "projects" / "daheng"

DEFAULT_C1 = PROJECT / "outputs/0818/c1_4k_freeze/frozen_c1_4k.json"
DEFAULT_LUT = PROJECT / "outputs/0818/c1_4k_freeze/c1_4k_lut.csv"
DEFAULT_C1_MANIFEST = PROJECT / "outputs/0818/c1_4k_freeze/c1_freeze_manifest.json"
DEFAULT_C0 = PROJECT / "outputs/0818/c0_freeze/quadratic_graph.yaml"
DEFAULT_VALIDATION = PROJECT / "outputs/0818/final_validation_qc/validation_points_used.csv"
DEFAULT_AUDIT = PROJECT / "outputs/0818/final_validation_qc/validation_artifact_reuse_audit.csv"
DEFAULT_OUTPUT = PROJECT / "outputs/0818/c1_validation_c1_4k"

EXPECTED_POSES = [
    *[f"{i:03d}" for i in range(19, 25)],
    *[f"{i:03d}" for i in range(37, 41)],
    *[f"{i:03d}" for i in range(55, 61)],
]
GROUPS: dict[str, list[str]] = {
    "validation_019_024": [f"{i:03d}" for i in range(19, 25)],
    "validation_037_040": [f"{i:03d}" for i in range(37, 41)],
    "validation_055_060": [f"{i:03d}" for i in range(55, 61)],
    "pooled_16": EXPECTED_POSES,
}
REGIONS: dict[str, tuple[float, float]] = {
    "top": (0.0, 300.0),
    "middle": (300.0, 2700.0),
    "bottom": (2700.0, 3000.0),
}
V_BIN_WIDTH = 100.0
V_BIN_COUNT = 30
METRIC_COLUMNS = ("bias_mm", "mae_mm", "rmse_mm", "p95_abs_mm", "p99_abs_mm", "max_abs_mm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-c1", type=Path, default=DEFAULT_C1)
    parser.add_argument("--lut", type=Path, default=DEFAULT_LUT)
    parser.add_argument("--c1-manifest", type=Path, default=DEFAULT_C1_MANIFEST)
    parser.add_argument("--frozen-c0", type=Path, default=DEFAULT_C0)
    parser.add_argument("--validation-points", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--validation-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def frame_balanced_weights(frame_ids: Sequence[str]) -> np.ndarray:
    frame_ids = np.asarray(frame_ids).astype(str)
    weights = np.zeros(len(frame_ids), dtype=float)
    for frame_id in np.unique(frame_ids):
        indices = np.flatnonzero(frame_ids == frame_id)
        weights[indices] = 1.0 / len(indices)
    return weights


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[keep], weights[keep]
    if not len(values):
        return math.nan
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(np.interp(float(quantile) * cumulative[-1], cumulative, values))


def scalar_metrics(frame_ids: Sequence[str], residual: np.ndarray) -> dict[str, float | int]:
    residual = np.asarray(residual, dtype=float)
    if not len(residual):
        return {
            "point_count": 0,
            "pose_count": 0,
            "bias_mm": math.nan,
            "mae_mm": math.nan,
            "rmse_mm": math.nan,
            "p95_abs_mm": math.nan,
            "p99_abs_mm": math.nan,
            "max_abs_mm": math.nan,
        }
    weights = frame_balanced_weights(frame_ids)
    weights = weights / np.sum(weights)
    absolute = np.abs(residual)
    return {
        "point_count": int(len(residual)),
        "pose_count": int(len(np.unique(np.asarray(frame_ids).astype(str)))),
        "bias_mm": float(np.sum(weights * residual)),
        "mae_mm": float(np.sum(weights * absolute)),
        "rmse_mm": float(math.sqrt(np.sum(weights * residual * residual))),
        "p95_abs_mm": weighted_quantile(absolute, weights, 0.95),
        "p99_abs_mm": weighted_quantile(absolute, weights, 0.99),
        "max_abs_mm": float(np.max(absolute)),
    }


def pct_improvement(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1.0e-15:
        return math.nan
    return float(100.0 * (before - after) / before)


def load_frozen_inputs(
    c1_path: Path,
    lut_path: Path,
    c1_manifest_path: Path,
    c0_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, str]]:
    c1_path = require_file(c1_path)
    lut_path = require_file(lut_path)
    c1_manifest_path = require_file(c1_manifest_path)
    c0_path = require_file(c0_path)

    model = json.loads(c1_path.read_text(encoding="utf-8"))
    manifest = json.loads(c1_manifest_path.read_text(encoding="utf-8"))
    c0 = yaml.safe_load(c0_path.read_text(encoding="utf-8"))

    if model.get("model_id") != "C1_4k" or model.get("operational_model") != "C1_4k":
        raise RuntimeError("Frozen model is not C1_4k")
    if model.get("frozen") is not True or model.get("freeze_status") != "FROZEN_FOR_VALIDATION":
        raise RuntimeError("C1 model is not marked FROZEN_FOR_VALIDATION")
    if model.get("frame027_status") != "EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN":
        raise RuntimeError("frame027 status is not the required operational-domain exclusion")
    if model.get("frame027_exclusion_reason") != "超出实际工作姿态域":
        raise RuntimeError("frame027 exclusion reason is not the required non-residual reason")
    if model.get("provenance", {}).get("validation_read") is not False:
        raise RuntimeError("Frozen C1 provenance does not prove Validation was excluded from fitting")
    if model.get("provenance", {}).get("c0_refit") is not False:
        raise RuntimeError("Frozen C1 provenance does not prove C0 was frozen")
    if model.get("provenance", {}).get("production_config_modified") is not False:
        raise RuntimeError("Frozen C1 provenance reports a production configuration change")
    if model.get("provenance", {}).get("c1_candidates_fit") != ["C1_4k"]:
        raise RuntimeError("Frozen C1 provenance is not the single-candidate C1_4k freeze")

    if c0.get("model_type") != "quadratic_graph":
        raise RuntimeError("Frozen C0 is not quadratic_graph")
    actual_c0_sha = sha256_file(c0_path)
    expected_c0_sha = model.get("provenance", {}).get("frozen_c0_sha256")
    if expected_c0_sha != actual_c0_sha:
        raise RuntimeError(f"Frozen C0 SHA mismatch: expected {expected_c0_sha}, got {actual_c0_sha}")
    if manifest.get("source_hashes", {}).get("frozen_c0_sha256") != actual_c0_sha:
        raise RuntimeError("Freeze manifest and C0 artifact SHA disagree")

    pca = model["pca_s"]
    spline = model["spline"]
    protocol = model["protocol"]
    fit = model["fit"]
    parameter_payload = {"pca_s": pca, "spline": spline, "protocol": protocol, "fit": fit}
    actual_parameter_sha = canonical_sha256(parameter_payload)
    if actual_parameter_sha != model.get("parameter_sha256"):
        raise RuntimeError("Frozen C1 parameter SHA mismatch")
    for key, payload in (("pca_sha256", pca), ("spline_sha256", spline), ("protocol_sha256", protocol), ("fit_sha256", fit)):
        if canonical_sha256(payload) != model.get("parameter_hashes", {}).get(key):
            raise RuntimeError(f"Frozen C1 {key} mismatch")
    if int(spline["degree"]) != 3 or int(spline["interior_knot_count"]) != 4:
        raise RuntimeError("Frozen C1 is not the requested cubic 4-interior-knot spline")
    if protocol.get("extrapolation") != "clip_to_pca_s_domain":
        raise RuntimeError("Frozen C1 extrapolation policy is not clip_to_pca_s_domain")
    if protocol.get("pca_definition") != "Full-36 PCA from xn/yn; PCA includes frame027 and is not recomputed on Operational-35":
        raise RuntimeError("Frozen C1 PCA provenance does not match the original Full-36 protocol")
    if fit.get("training_excludes_027") is not True or len(fit.get("training_pose_ids", [])) != 35:
        raise RuntimeError("Frozen C1 training pose provenance is not Operational-35")

    lut = pd.read_csv(lut_path)
    if list(lut.columns) != ["s", "delta_lambda_mm"] or len(lut) != 2049:
        raise RuntimeError("C1 LUT is not the persisted 2049-point s->delta_lambda artifact")
    lut_s = lut["s"].to_numpy(float)
    lut_delta = lut["delta_lambda_mm"].to_numpy(float)
    if not np.all(np.isfinite(lut[["s", "delta_lambda_mm"]].to_numpy(float))):
        raise RuntimeError("C1 LUT contains non-finite values")
    domain_min, domain_max = float(pca["domain_min"]), float(pca["domain_max"])
    if not np.isclose(lut_s[0], domain_min) or not np.isclose(lut_s[-1], domain_max):
        raise RuntimeError("C1 LUT endpoints do not match the frozen PCA-s domain")
    knots = np.asarray(spline["knots"], dtype=float)
    coefficients = np.asarray(spline["coefficients_mm"], dtype=float)
    exact_lut = BSpline.design_matrix(lut_s, knots, k=int(spline["degree"]), extrapolate=False).toarray() @ coefficients
    lut_max_error = float(np.max(np.abs(exact_lut - lut_delta)))
    if lut_max_error > 1.0e-3:
        raise RuntimeError(f"Frozen C1 LUT does not match exact spline within 0.001 mm: {lut_max_error}")

    hashes = {
        "frozen_c1_sha256": sha256_file(c1_path),
        "c1_lut_sha256": sha256_file(lut_path),
        "c1_manifest_sha256": sha256_file(c1_manifest_path),
        "frozen_c0_sha256": actual_c0_sha,
        "parameter_sha256": actual_parameter_sha,
        "validation_audit_sha256": "",
        "validation_points_sha256": "",
    }
    expected_lut_sha = manifest.get("lut_sha256")
    if expected_lut_sha and expected_lut_sha != hashes["c1_lut_sha256"]:
        raise RuntimeError("C1 freeze manifest and LUT SHA disagree")
    model_expected_c1_sha = manifest.get("frozen_model_sha256")
    if model_expected_c1_sha and model_expected_c1_sha != hashes["frozen_c1_sha256"]:
        raise RuntimeError("C1 freeze manifest and frozen model SHA disagree")
    model["_validation_lut_max_error_mm"] = lut_max_error
    return model, lut, manifest, c0, hashes


def load_validation_points(path: Path, audit_path: Path, hashes: dict[str, str]) -> pd.DataFrame:
    path = require_file(path)
    audit_path = require_file(audit_path)
    hashes["validation_points_sha256"] = sha256_file(path)
    hashes["validation_audit_sha256"] = sha256_file(audit_path)

    audit_text = audit_path.read_text(encoding="utf-8")
    if "Validation 0817 C1 artifacts" not in audit_text or "EXCLUDED" not in audit_text:
        raise RuntimeError("Validation reuse audit does not document exclusion of legacy C1 artifacts")
    points = pd.read_csv(path, dtype={"frame_id": str, "pose_id": str})
    required = {
        "split", "pose_id", "frame_id", "v_px", "ray_x", "ray_y", "ray_z",
        "lambda_pred_quadratic_graph_mm", "lambda_truth_mm", "valid_quadratic_graph",
    }
    missing = sorted(required.difference(points.columns))
    if missing:
        raise RuntimeError(f"Validation point artifact is missing columns: {missing}")
    points["pose_id"] = points["pose_id"].astype(str).str.zfill(3)
    points["frame_id"] = points["frame_id"].astype(str).str.zfill(3)
    if set(points["split"].astype(str).str.lower()) != {"validation"}:
        raise RuntimeError("Validation point artifact is not Validation-only")
    if sorted(points["pose_id"].unique().tolist()) != EXPECTED_POSES:
        raise RuntimeError("Validation point artifact does not contain exactly the requested 16 poses")
    counts = points.groupby("pose_id", sort=True).size()
    if not np.all(counts.to_numpy() == 900) or len(points) != 14400:
        raise RuntimeError(f"Expected 14400 Validation points / 900 per pose, got {len(points)}")
    if not np.all(points["frame_id"].to_numpy() == points["pose_id"].to_numpy()):
        raise RuntimeError("Validation pose_id and frame_id differ")
    if not np.all(points["valid_quadratic_graph"].astype(bool).to_numpy()):
        raise RuntimeError("Validation artifact contains invalid frozen-C0 intersections")
    numeric = [
        "v_px", "ray_x", "ray_y", "ray_z", "lambda_pred_quadratic_graph_mm", "lambda_truth_mm",
    ]
    if not np.all(np.isfinite(points[numeric].to_numpy(float))):
        raise RuntimeError("Validation artifact contains non-finite evaluation values")
    c0 = points["lambda_pred_quadratic_graph_mm"].to_numpy(float)
    truth = points["lambda_truth_mm"].to_numpy(float)
    return points


def evaluate_frozen_c1(points: pd.DataFrame, model: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    pca = model["pca_s"]
    spline = model["spline"]
    degree = int(spline["degree"])
    knots = np.asarray(spline["knots"], dtype=float)
    coefficients = np.asarray(spline["coefficients_mm"], dtype=float)
    domain_min, domain_max = float(pca["domain_min"]), float(pca["domain_max"])
    ray = points[["ray_x", "ray_y", "ray_z"]].to_numpy(float)
    if np.any(np.abs(ray[:, 2]) < 1.0e-15):
        raise RuntimeError("Validation ray_z contains a zero; cannot reproduce frozen xn/yn projection")
    xn = ray[:, 0] / ray[:, 2]
    yn = ray[:, 1] / ray[:, 2]
    center = np.array([float(pca["center_xn"]), float(pca["center_yn"])])
    axis_s = np.array([float(pca["axis_s_xn"]), float(pca["axis_s_yn"])])
    s_raw = (np.column_stack([xn, yn]) - center) @ axis_s
    clamp = (s_raw < domain_min) | (s_raw > domain_max)
    s_eval = np.clip(s_raw, domain_min, domain_max)
    design = BSpline.design_matrix(s_eval, knots, k=degree, extrapolate=False).toarray()
    correction = design @ coefficients
    lambda_c0 = points["lambda_pred_quadratic_graph_mm"].to_numpy(float)
    truth = points["lambda_truth_mm"].to_numpy(float)
    lambda_c1 = lambda_c0 + correction
    result = points.copy()
    result["xn_frozen_projection"] = xn
    result["yn_frozen_projection"] = yn
    result["s_raw"] = s_raw
    result["s_eval_clipped"] = s_eval
    result["c1_clamp"] = clamp
    result["delta_lambda_c1_mm"] = correction
    result["lambda_c0_mm"] = lambda_c0
    result["lambda_c1_mm"] = lambda_c1
    result["residual_c0_mm"] = truth - lambda_c0
    result["residual_c1_mm"] = truth - lambda_c1
    audit = {
        "pca_center_xn": center[0],
        "pca_center_yn": center[1],
        "axis_s_xn": axis_s[0],
        "axis_s_yn": axis_s[1],
        "s_domain_min": domain_min,
        "s_domain_max": domain_max,
        "clamp_count": int(np.sum(clamp)),
        "clamp_ratio": float(np.mean(clamp)),
        "lut_max_error_mm": float(model["_validation_lut_max_error_mm"]),
        "evaluation": "exact frozen cubic B-spline after np.clip(s_raw, domain_min, domain_max); no extrapolation",
    }
    return result, audit


def add_metric_pair(row: dict[str, Any], c0: Mapping[str, Any], c1: Mapping[str, Any]) -> None:
    for name in METRIC_COLUMNS:
        row[f"c0_{name}"] = c0[name]
        row[f"c1_{name}"] = c1[name]
        row[f"{name}_change"] = float(c0[name] - c1[name]) if np.isfinite(c0[name]) and np.isfinite(c1[name]) else math.nan
        improvement_names = {
            "mae_mm": "mae_improvement_pct",
            "rmse_mm": "rmse_improvement_pct",
            "p95_abs_mm": "p95_abs_mm_improvement_pct",
            "p99_abs_mm": "p99_abs_mm_improvement_pct",
        }
        if name in improvement_names:
            row[improvement_names[name]] = pct_improvement(float(c0[name]), float(c1[name]))


def v_bin_label(index: int) -> str:
    return f"v_{index * 100:04d}_{index * 100 + 100:04d}"


def v_bin_metrics(data: pd.DataFrame, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v = data["v_px"].to_numpy(float)
    for index in range(V_BIN_COUNT):
        keep = (v >= index * V_BIN_WIDTH) & (v < (index + 1) * V_BIN_WIDTH)
        if not np.any(keep):
            continue
        part = data.loc[keep]
        c0 = scalar_metrics(part["pose_id"].to_numpy(), part["residual_c0_mm"].to_numpy(float))
        c1 = scalar_metrics(part["pose_id"].to_numpy(), part["residual_c1_mm"].to_numpy(float))
        row: dict[str, Any] = {
            "scope": scope,
            "v_bin_index": index,
            "v_bin": v_bin_label(index),
            "v_start_px": index * V_BIN_WIDTH,
            "v_end_px": (index + 1) * V_BIN_WIDTH,
            "in_v_domain": True,
            "clamp_count": int(part["c1_clamp"].sum()),
            "clamp_ratio": float(part["c1_clamp"].mean()),
        }
        add_metric_pair(row, c0, c1)
        rows.append(row)
    out = (v < 0.0) | (v >= V_BIN_COUNT * V_BIN_WIDTH)
    if np.any(out):
        part = data.loc[out]
        c0 = scalar_metrics(part["pose_id"].to_numpy(), part["residual_c0_mm"].to_numpy(float))
        c1 = scalar_metrics(part["pose_id"].to_numpy(), part["residual_c1_mm"].to_numpy(float))
        row = {
            "scope": scope,
            "v_bin_index": -1,
            "v_bin": "out_of_range",
            "v_start_px": math.nan,
            "v_end_px": math.nan,
            "in_v_domain": False,
            "clamp_count": int(part["c1_clamp"].sum()),
            "clamp_ratio": float(part["c1_clamp"].mean()),
        }
        add_metric_pair(row, c0, c1)
        rows.append(row)
    return rows


def pose_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pose_id, part in data.groupby("pose_id", sort=True):
        c0 = scalar_metrics([pose_id] * len(part), part["residual_c0_mm"].to_numpy(float))
        c1 = scalar_metrics([pose_id] * len(part), part["residual_c1_mm"].to_numpy(float))
        row: dict[str, Any] = {"pose_id": pose_id, "group": next(name for name, ids in GROUPS.items() if pose_id in ids and name != "pooled_16")}
        row["point_count"] = len(part)
        row["clamp_count"] = int(part["c1_clamp"].sum())
        row["clamp_ratio"] = float(part["c1_clamp"].mean())
        add_metric_pair(row, c0, c1)
        row["rmse_improved"] = bool(c1["rmse_mm"] < c0["rmse_mm"])
        row["p95_improved"] = bool(c1["p95_abs_mm"] < c0["p95_abs_mm"])
        row["bias_abs_change_mm"] = float(abs(c0["bias_mm"]) - abs(c1["bias_mm"]))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_row(data: pd.DataFrame, scope: str, scope_type: str) -> dict[str, Any]:
    c0 = scalar_metrics(data["pose_id"].to_numpy(), data["residual_c0_mm"].to_numpy(float))
    c1 = scalar_metrics(data["pose_id"].to_numpy(), data["residual_c1_mm"].to_numpy(float))
    bins = v_bin_metrics(data, scope)
    in_bins = [row for row in bins if bool(row["in_v_domain"])]
    row: dict[str, Any] = {
        "scope": scope,
        "scope_type": scope_type,
        "point_count": int(len(data)),
        "pose_count": int(data["pose_id"].nunique()),
        "clamp_count": int(data["c1_clamp"].sum()),
        "clamp_ratio": float(data["c1_clamp"].mean()) if len(data) else math.nan,
        "v_out_of_range_count": int((~data["v_px"].between(0.0, 3000.0, inclusive="left")).sum()),
    }
    add_metric_pair(row, c0, c1)
    if in_bins:
        worst_rmse_c0 = max(in_bins, key=lambda item: item["c0_rmse_mm"])
        worst_rmse_c1 = max(in_bins, key=lambda item: item["c1_rmse_mm"])
        worst_p95_c0 = max(in_bins, key=lambda item: item["c0_p95_abs_mm"])
        worst_p95_c1 = max(in_bins, key=lambda item: item["c1_p95_abs_mm"])
        c0_bias = np.asarray([item["c0_bias_mm"] for item in in_bins], dtype=float)
        c1_bias = np.asarray([item["c1_bias_mm"] for item in in_bins], dtype=float)
        row.update(
            {
                "c0_worst_v_rmse_mm": worst_rmse_c0["c0_rmse_mm"],
                "c1_worst_v_rmse_mm": worst_rmse_c1["c1_rmse_mm"],
                "worst_v_rmse_change_mm": worst_rmse_c0["c0_rmse_mm"] - worst_rmse_c1["c1_rmse_mm"],
                "c0_worst_v_rmse_bin": worst_rmse_c0["v_bin"],
                "c1_worst_v_rmse_bin": worst_rmse_c1["v_bin"],
                "c0_worst_v_p95_abs_mm": worst_p95_c0["c0_p95_abs_mm"],
                "c1_worst_v_p95_abs_mm": worst_p95_c1["c1_p95_abs_mm"],
                "worst_v_p95_change_mm": worst_p95_c0["c0_p95_abs_mm"] - worst_p95_c1["c1_p95_abs_mm"],
                "c0_worst_v_p95_bin": worst_p95_c0["v_bin"],
                "c1_worst_v_p95_bin": worst_p95_c1["v_bin"],
                "c0_v_bias_range_mm": float(np.ptp(c0_bias)),
                "c1_v_bias_range_mm": float(np.ptp(c1_bias)),
                "v_bias_range_change_mm": float(np.ptp(c0_bias) - np.ptp(c1_bias)),
                "v_bin_count": len(in_bins),
            }
        )
    else:
        for key in (
            "c0_worst_v_rmse_mm", "c1_worst_v_rmse_mm", "worst_v_rmse_change_mm",
            "c0_worst_v_p95_abs_mm", "c1_worst_v_p95_abs_mm", "worst_v_p95_change_mm",
            "c0_v_bias_range_mm", "c1_v_bias_range_mm", "v_bias_range_change_mm",
        ):
            row[key] = math.nan
        row["c0_worst_v_rmse_bin"] = ""
        row["c1_worst_v_rmse_bin"] = ""
        row["c0_worst_v_p95_bin"] = ""
        row["c1_worst_v_p95_bin"] = ""
        row["v_bin_count"] = 0
    return row


def add_pose_ratio(summary: pd.DataFrame, poses: pd.DataFrame) -> pd.DataFrame:
    ratios = []
    for _, row in summary.iterrows():
        scope = row["scope"]
        if scope in GROUPS:
            ids = set(GROUPS[scope])
        elif scope in REGIONS:
            lo, hi = REGIONS[scope]
            ids = set(poses.loc[(poses["pose_id"].isin(EXPECTED_POSES)), "pose_id"])
            # Region-specific pose improvement is calculated from the same region below.
            ids = set(ids)
        else:
            ids = set()
        selected = poses[poses["pose_id"].isin(ids)]
        ratios.append(float(selected["rmse_improved"].mean()) if len(selected) else math.nan)
    summary = summary.copy()
    summary["pose_improvement_ratio"] = ratios
    return summary


def build_region_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (lo, hi) in REGIONS.items():
        keep = (data["v_px"] >= lo) & (data["v_px"] < hi)
        rows.append(summary_row(data.loc[keep], name, "v_region"))
    return pd.DataFrame(rows)


def build_clamp_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, keep in (
        ("clamped_only", data["c1_clamp"].to_numpy(bool)),
        ("in_domain_only", ~data["c1_clamp"].to_numpy(bool)),
    ):
        rows.append(summary_row(data.loc[keep], name, "clamp_partition"))
    return pd.DataFrame(rows)


def make_summary(data: pd.DataFrame, poses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [
        summary_row(data.loc[data["pose_id"].isin(ids)], name, "validation_group" if name != "pooled_16" else "pooled")
        for name, ids in GROUPS.items()
    ]
    summary = pd.DataFrame(rows)
    # Region rows use their own paired region metrics and a region-specific pose ratio.
    region = build_region_summary(data)
    clamp = build_clamp_summary(data)

    region_pose_rows = []
    for name, (lo, hi) in REGIONS.items():
        part = data.loc[(data["v_px"] >= lo) & (data["v_px"] < hi)]
        for pose_id, pose_part in part.groupby("pose_id", sort=True):
            c0 = scalar_metrics([pose_id] * len(pose_part), pose_part["residual_c0_mm"].to_numpy(float))
            c1 = scalar_metrics([pose_id] * len(pose_part), pose_part["residual_c1_mm"].to_numpy(float))
            region_pose_rows.append({"region": name, "pose_id": pose_id, "rmse_improved": c1["rmse_mm"] < c0["rmse_mm"]})
    region_pose = pd.DataFrame(region_pose_rows)
    region_ratios = region_pose.groupby("region")["rmse_improved"].mean().to_dict() if len(region_pose) else {}
    region["pose_improvement_ratio"] = region["scope"].map(region_ratios)

    pooled_pose_ratio = float(poses["rmse_improved"].mean())
    summary["pose_improvement_ratio"] = summary["scope"].map(
        {"pooled_16": pooled_pose_ratio, **{name: float(poses.loc[poses["group"] == name, "rmse_improved"].mean()) for name in GROUPS if name != "pooled_16"}}
    )
    clamp["pose_improvement_ratio"] = math.nan
    return pd.concat([summary, region, clamp], ignore_index=True, sort=False), region, clamp


def make_v_bins(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, ids in GROUPS.items():
        rows.extend(v_bin_metrics(data.loc[data["pose_id"].isin(ids)], scope))
    for scope, (lo, hi) in REGIONS.items():
        rows.extend(v_bin_metrics(data.loc[(data["v_px"] >= lo) & (data["v_px"] < hi)], scope))
    return pd.DataFrame(rows)


def status_from_metrics(summary: pd.DataFrame, region: pd.DataFrame, poses: pd.DataFrame) -> str:
    pooled = summary.loc[summary["scope"] == "pooled_16"].iloc[0]
    groups = summary[summary["scope"].isin(["validation_019_024", "validation_037_040", "validation_055_060"])]
    pooled_good = bool(pooled["rmse_improvement_pct"] > 0.0 and pooled["p95_abs_mm_improvement_pct"] > 0.0)
    groups_good = bool(np.all(groups["rmse_improvement_pct"].to_numpy(float) > 0.0) and np.all(groups["p95_abs_mm_improvement_pct"].to_numpy(float) > 0.0))
    regions_good = bool(np.all(region["rmse_improvement_pct"].to_numpy(float) > 0.0) and np.all(region["p95_abs_mm_improvement_pct"].to_numpy(float) > 0.0))
    pose_good = bool(float(poses["rmse_improved"].mean()) >= 0.5)
    if not pooled_good:
        return "FAIL"
    if groups_good and regions_good and pose_good:
        return "PASS"
    return "PARTIAL"


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 6) -> str:
    if frame.empty:
        return "(none)"
    table = frame.loc[:, list(columns)].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if not np.isfinite(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for values in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def plot_residual_vs_v(v_bins: pd.DataFrame, output: Path) -> None:
    pooled = v_bins[(v_bins["scope"] == "pooled_16") & v_bins["in_v_domain"]].sort_values("v_bin_index")
    x = (pooled["v_start_px"].to_numpy(float) + pooled["v_end_px"].to_numpy(float)) / 2.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [
        (axes[0, 0], "c0_bias_mm", "c1_bias_mm", "Bias vs v", "Bias (mm)"),
        (axes[0, 1], "c0_rmse_mm", "c1_rmse_mm", "RMSE vs v", "RMSE (mm)"),
        (axes[1, 0], "c0_p95_abs_mm", "c1_p95_abs_mm", "P95 absolute residual vs v", "P95 (mm)"),
        (axes[1, 1], "clamp_ratio", None, "C1 s-domain clamp ratio vs v", "Clamp ratio"),
    ]
    for ax, c0_col, c1_col, title, ylabel in specs:
        if c1_col is None:
            ax.plot(x, pooled[c0_col].to_numpy(float), "o-", color="#b23a48", label="clamp ratio")
            ax.set_ylim(bottom=0.0)
            ax.legend()
        else:
            ax.plot(x, pooled[c0_col].to_numpy(float), "o-", label="C0")
            ax.plot(x, pooled[c1_col].to_numpy(float), "o-", label="C0+C1")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("v (px)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("Independent Validation: frozen Operational-35 C1_4k", fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_residual_vs_s(data: pd.DataFrame, model: Mapping[str, Any], output: Path) -> None:
    pca = model["pca_s"]
    s_min, s_max = float(pca["domain_min"]), float(pca["domain_max"])
    raw_s = data["s_raw"].to_numpy(float)
    clamp = data["c1_clamp"].to_numpy(bool)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].scatter(raw_s[~clamp], data.loc[~clamp, "residual_c0_mm"], s=3, alpha=0.18, label="C0 in-domain")
    axes[0, 0].scatter(raw_s[~clamp], data.loc[~clamp, "residual_c1_mm"], s=3, alpha=0.18, label="C0+C1 in-domain")
    if np.any(clamp):
        axes[0, 0].scatter(raw_s[clamp], data.loc[clamp, "residual_c0_mm"], s=10, marker="x", color="#b23a48", label="C0 clamped")
        axes[0, 0].scatter(raw_s[clamp], data.loc[clamp, "residual_c1_mm"], s=10, marker="x", color="#1d6fa5", label="C0+C1 clamped")
    axes[0, 0].axvspan(s_min, s_max, color="grey", alpha=0.08, label="frozen s domain")
    axes[0, 0].set_title("Residual vs raw frozen PCA s")
    axes[0, 0].set_xlabel("raw s")
    axes[0, 0].set_ylabel("residual (mm)")
    axes[0, 0].legend(markerscale=2, fontsize=8)

    bins = np.linspace(min(float(np.min(raw_s)), s_min), max(float(np.max(raw_s)), s_max), 41)
    centers = (bins[:-1] + bins[1:]) / 2.0
    for column, label, color in (("residual_c0_mm", "C0", "#e07a5f"), ("residual_c1_mm", "C0+C1", "#277da1")):
        values = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            keep = (raw_s >= lo) & (raw_s < hi)
            values.append(float(np.mean(np.abs(data.loc[keep, column]))) if np.any(keep) else math.nan)
        axes[0, 1].plot(centers, values, "o-", ms=3, label=label, color=color)
    axes[0, 1].axvspan(s_min, s_max, color="grey", alpha=0.08)
    axes[0, 1].set_title("Mean absolute residual vs raw s")
    axes[0, 1].set_xlabel("raw s")
    axes[0, 1].set_ylabel("mean |residual| (mm)")
    axes[0, 1].legend()

    spline = model["spline"]
    grid = np.linspace(s_min, s_max, 512)
    correction = BSpline.design_matrix(grid, np.asarray(spline["knots"], float), k=int(spline["degree"]), extrapolate=False).toarray() @ np.asarray(spline["coefficients_mm"], float)
    axes[1, 0].plot(grid, correction, color="#6a4c93")
    axes[1, 0].set_title("Frozen C1 correction F(s)")
    axes[1, 0].set_xlabel("clipped s")
    axes[1, 0].set_ylabel("delta lambda (mm)")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].scatter(raw_s, data["delta_lambda_c1_mm"], s=3, alpha=0.18, c=np.where(clamp, "#b23a48", "#277da1"))
    axes[1, 1].axvspan(s_min, s_max, color="grey", alpha=0.08)
    axes[1, 1].set_title("Applied correction vs raw s (clamped points marked red)")
    axes[1, 1].set_xlabel("raw s")
    axes[1, 1].set_ylabel("delta lambda (mm)")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("Independent Validation: frozen Operational-35 C1_4k", fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_report(
    output: Path,
    model: Mapping[str, Any],
    manifest: Mapping[str, Any],
    hashes: Mapping[str, str],
    points: pd.DataFrame,
    summary: pd.DataFrame,
    region: pd.DataFrame,
    clamp: pd.DataFrame,
    poses: pd.DataFrame,
    v_bins: pd.DataFrame,
    status: str,
    c1_path: Path,
    lut_path: Path,
    c0_path: Path,
    validation_path: Path,
    audit_path: Path,
) -> None:
    pooled = summary.loc[summary["scope"] == "pooled_16"].iloc[0]
    group_summary = summary[summary["scope"].isin(list(GROUPS))]
    region_view = region[["scope", "point_count", "c0_rmse_mm", "c1_rmse_mm", "rmse_improvement_pct", "c0_p95_abs_mm", "c1_p95_abs_mm", "p95_abs_mm_improvement_pct", "pose_improvement_ratio", "clamp_ratio"]]
    group_view = group_summary[["scope", "point_count", "pose_count", "c0_bias_mm", "c1_bias_mm", "c0_mae_mm", "c1_mae_mm", "c0_rmse_mm", "c1_rmse_mm", "rmse_improvement_pct", "c0_p95_abs_mm", "c1_p95_abs_mm", "p95_abs_mm_improvement_pct", "c0_p99_abs_mm", "c1_p99_abs_mm", "p99_abs_mm_improvement_pct", "c0_worst_v_rmse_mm", "c1_worst_v_rmse_mm", "c0_worst_v_p95_abs_mm", "c1_worst_v_p95_abs_mm", "c0_v_bias_range_mm", "c1_v_bias_range_mm", "pose_improvement_ratio", "clamp_ratio"]]
    pose_view = poses[["pose_id", "group", "c0_rmse_mm", "c1_rmse_mm", "rmse_improvement_pct", "c0_p95_abs_mm", "c1_p95_abs_mm", "p95_abs_mm_improvement_pct", "rmse_improved", "p95_improved", "clamp_count", "clamp_ratio"]]
    clamp_view = clamp[["scope", "point_count", "pose_count", "c0_rmse_mm", "c1_rmse_mm", "rmse_improvement_pct", "c0_p95_abs_mm", "c1_p95_abs_mm", "p95_abs_mm_improvement_pct", "clamp_ratio"]]
    all_regions_rmse = bool(np.all(region["rmse_improvement_pct"].to_numpy(float) > 0.0))
    all_regions_p95 = bool(np.all(region["p95_abs_mm_improvement_pct"].to_numpy(float) > 0.0))
    c1_manifest_display = c1_path.parent / "c1_freeze_manifest.json"
    report = f"""# Frozen Operational-35 C1_4k independent Validation

`C1_VALIDATION_STATUS = {status}`

## Scope and controls

本轮仅评估冻结模型，不进行 `fit()`、PCA、knots/coefficients/penalty 调整，也不读取 Validation 之外的训练数据来改变模型。frame027 不在本 Validation 集中，也没有被删除；其冻结状态仍为 `EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN`，理由为“超出实际工作姿态域”。本轮没有修改生产配置。

- Validation poses: `{", ".join(EXPECTED_POSES)}`；共 `{len(points)}` 点，16 pose，900 点/pose。
- 分组：019–024、037–040、055–060；pooled-16。
- C0：复用 `{validation_path.name}` 中已由 Frozen Quadratic C0 产生的 `lambda_pred_quadratic_graph_mm`，没有在 Validation 上重新拟合或重新估计 C0。
- C1：直接加载冻结 JSON 的 Full-36 PCA center/axis/domain、cubic B-spline knots/coefficients；按冻结协议先将 raw `s` clamp 到 `[domain_min, domain_max]`，再精确求值，禁止 spline 外推。
- 评价权重：frame-balanced；每个 pose 总权重相等。P95/P99 是对应 frame-balanced 权重下的 absolute residual quantile。
- v-bin：沿用既有 QC 的固定 100 px bins `[0, 3000)`；out-of-range 点保留在 global/pose 汇总并单独列出。
- Top/Middle/Bottom：固定 reporting regions 分别为 `[0,300)`、`[300,2700)`、`[2700,3000)` px；不是基于 Validation 调整的选择。

## Artifact provenance / reuse audit

本轮复用现有点级 Validation artifact 和冻结模型；不复用旧的 C1 Validation 数值结果。Validation reuse audit 已确认旧的 0817 C1 artifacts excluded，当前点 artifact 为正式 16-pose Validation 输入。Frozen C1 的 `validation_read=false`、`c0_refit=false`、`production_config_modified=false`，且其 PCA 明确为 Full-36（包含027），没有重算 Operational-35 PCA。

| artifact | path | SHA-256 |
| --- | --- | --- |
| Frozen C1 JSON | `{c1_path}` | `{hashes['frozen_c1_sha256']}` |
| C1 LUT | `{lut_path}` | `{hashes['c1_lut_sha256']}` |
| C1 freeze manifest | `{c1_manifest_display}` | `{hashes['c1_manifest_sha256']}` |
| Frozen C0 YAML | `{c0_path}` | `{hashes['frozen_c0_sha256']}` |
| Validation points | `{validation_path}` | `{hashes['validation_points_sha256']}` |
| Validation reuse audit | `{audit_path}` | `{hashes['validation_audit_sha256']}` |

Frozen C1 parameter SHA: `{hashes['parameter_sha256']}`. LUT 与精确 frozen spline 的 2049 个网格点最大绝对差：`{model['_validation_lut_max_error_mm']:.12g} mm`（阈值 0.001 mm）。

## Pooled and validation-group metrics

“improvement”定义为 `100 * (C0 - C0+C1) / C0`；Max 只作诊断，不参与 status 判断。

{markdown_table(group_view, group_view.columns, 5)}

Pooled-16 关键结果：RMSE `{pooled['c0_rmse_mm']:.6f} -> {pooled['c1_rmse_mm']:.6f} mm`（`{pooled['rmse_improvement_pct']:.3f}%`），P95 `{pooled['c0_p95_abs_mm']:.6f} -> {pooled['c1_p95_abs_mm']:.6f} mm`（`{pooled['p95_abs_mm_improvement_pct']:.3f}%`），P99 `{pooled['c0_p99_abs_mm']:.6f} -> {pooled['c1_p99_abs_mm']:.6f} mm`（`{pooled['p99_abs_mm_improvement_pct']:.3f}%`）。Pooled pose RMSE improvement ratio 为 `{pooled['pose_improvement_ratio']:.3f}`。

## Top / Middle / Bottom

三段是否同时受益：RMSE = `{all_regions_rmse}`，P95 = `{all_regions_p95}`。

{markdown_table(region_view, region_view.columns, 5)}

## Pose-level paired C0 -> C0+C1

逐 pose 的 RMSE/P95 paired 结果如下；`rmse_improved` 和 `p95_improved` 是逐 pose 的布尔判断，不以 Max 选模。

{markdown_table(pose_view, pose_view.columns, 5)}

## s-domain clamp

全体 clamp `{int(points['c1_clamp'].sum())}/{len(points)}` = `{float(points['c1_clamp'].mean()):.6%}`。这些点使用 domain edge 的 frozen spline 值，不做外推；clamp 点与非 clamp 点单独统计如下。Global/pose 指标仍包含全部 Validation 点。

{markdown_table(clamp_view, clamp_view.columns, 5)}

## Status rule and conclusion

本轮 status 规则固定写明如下：`FAIL` 若 pooled-16 RMSE 或 P95 没有改善；若 pooled-16 两者均改善但任一 validation group、Top/Middle/Bottom 稳定性条件或 pooled pose improvement ratio（至少 0.5）不满足，则为 `PARTIAL`；只有 pooled、三个 Validation group、三个 v regions 的 RMSE/P95 均改善且至少一半 pose 的 RMSE 改善，才为 `PASS`。Clamp 不被隐式当作失败，但按上节单独报告。

因此本次结论为 **`C1_VALIDATION_STATUS = {status}`**。这只是独立 Validation 结论，不会自动写入生产配置。

## Generated outputs

- `c1_validation_summary.csv`
- `c1_validation_pose_metrics.csv`
- `c1_validation_v_bins.csv`
- `c1_validation_residual_vs_v.png`
- `c1_validation_residual_vs_s.png`
- `c1_validation_report.md`
"""
    (output / "c1_validation_report.md").write_text(report, encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    c1_path = require_file(args.frozen_c1)
    lut_path = require_file(args.lut)
    c1_manifest_path = require_file(args.c1_manifest)
    c0_path = require_file(args.frozen_c0)
    validation_path = require_file(args.validation_points)
    audit_path = require_file(args.validation_audit)

    model, _lut, manifest, _c0, hashes = load_frozen_inputs(c1_path, lut_path, c1_manifest_path, c0_path)
    points = load_validation_points(validation_path, audit_path, hashes)
    evaluated, evaluation_audit = evaluate_frozen_c1(points, model)
    poses = pose_metrics(evaluated)
    summary, region, clamp = make_summary(evaluated, poses)
    v_bins = make_v_bins(evaluated)
    status = status_from_metrics(summary, region, poses)

    summary.to_csv(output / "c1_validation_summary.csv", index=False)
    poses.to_csv(output / "c1_validation_pose_metrics.csv", index=False)
    v_bins.to_csv(output / "c1_validation_v_bins.csv", index=False)
    plot_residual_vs_v(v_bins, output / "c1_validation_residual_vs_v.png")
    plot_residual_vs_s(evaluated, model, output / "c1_validation_residual_vs_s.png")
    write_report(output, model, manifest, hashes, evaluated, summary, region, clamp, poses, v_bins, status, c1_path, lut_path, c0_path, validation_path, audit_path)

    manifest_out = {
        "C1_VALIDATION_STATUS": status,
        "validation_read_for_evaluation": True,
        "c1_fit_called": False,
        "pca_fit_called": False,
        "parameter_adjustment": False,
        "c0_refit": False,
        "production_config_modified": False,
        "frozen_c1_path": str(c1_path),
        "frozen_c1_sha256": hashes["frozen_c1_sha256"],
        "frozen_c1_parameter_sha256": hashes["parameter_sha256"],
        "c1_lut_path": str(lut_path),
        "c1_lut_sha256": hashes["c1_lut_sha256"],
        "c1_lut_max_exact_spline_error_mm": evaluation_audit["lut_max_error_mm"],
        "frozen_c0_path": str(c0_path),
        "frozen_c0_sha256": hashes["frozen_c0_sha256"],
        "validation_points_path": str(validation_path),
        "validation_points_sha256": hashes["validation_points_sha256"],
        "validation_reuse_audit_path": str(audit_path),
        "validation_reuse_audit_sha256": hashes["validation_audit_sha256"],
        "validation_pose_ids": EXPECTED_POSES,
        "validation_point_count": int(len(evaluated)),
        "evaluation_protocol": evaluation_audit,
        "frame027_status": model["frame027_status"],
        "frame027_exclusion_reason": model["frame027_exclusion_reason"],
        "generated_files": [
            "c1_validation_summary.csv",
            "c1_validation_pose_metrics.csv",
            "c1_validation_v_bins.csv",
            "c1_validation_residual_vs_v.png",
            "c1_validation_residual_vs_s.png",
            "c1_validation_report.md",
        ],
    }
    (output / "c1_validation_manifest.json").write_text(json.dumps(json_clean(manifest_out), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "C1_VALIDATION_STATUS": status, **evaluation_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(parse_args())
