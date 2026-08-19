#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled FIT-only Operational-35 vs Full-36 C0 audit.

This runner is deliberately separate from the frozen C0 directory.  It keeps
the current Full-36 Quadratic YAML as an immutable baseline, fits one
Operational-35 Quadratic candidate from the raw Full-36 FIT point table with
the existing QuadraticGraphModel protocol, and evaluates both training arms on
the same 35 operational poses with deterministic pose-grouped CV.

The runner does not open Validation data and does not import or fit C1.  The
geometry audit is also independent of model residuals: it re-ranges all
coverage gates to the 35-pose operational domain after removing frame027.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_laser_models_from_triplets as triplets  # noqa: E402


FIT_IDS = tuple(
    [f"{value:03d}" for value in range(1, 19)]
    + [f"{value:03d}" for value in range(25, 37)]
    + [f"{value:03d}" for value in range(49, 55)]
)
FRAME027 = "027"
OPERATIONAL_IDS = tuple(frame_id for frame_id in FIT_IDS if frame_id != FRAME027)
FOLD_COUNT = 6
V_MIN_PX = 0.0
V_MAX_PX = 3000.0
V_GRID_WIDTH_PX = 10.0
V_BIN_WIDTH_PX = 100.0
V_GRID_COUNT = int((V_MAX_PX - V_MIN_PX) / V_GRID_WIDTH_PX)
V_BIN_COUNT = int((V_MAX_PX - V_MIN_PX) / V_BIN_WIDTH_PX)
EDGE_BIN_IDS = (0, 1, 28, 29)
NORMAL_COVER_THRESHOLD_DEG = 5.0
NORMAL_DIAMETER_TOLERANCE_DEG = 1.0
SPAN_RATIO_THRESHOLD = 0.95
EDGE_MIN_FRAME_COUNT = 2
EXCITATION_BIN_COUNT = 8
TRANSLATION_BIN_COUNT = 3

# The surface thresholds are the existing operational-surface diagnostic
# margins used in the repository's surface-equivalence audits.  They are not
# used to select or replace a production model.
SURFACE_NEGLIGIBLE_RMSE_MM = 0.010
SURFACE_NEGLIGIBLE_P95_MM = 0.030
SURFACE_NEGLIGIBLE_MAX_MM = 0.100
SURFACE_MATERIAL_RMSE_MM = 0.050
SURFACE_MATERIAL_P95_MM = 0.100
SURFACE_MATERIAL_MAX_MM = 0.500
CV_CHANGE_MATERIAL_PCT = 2.0

PROJECT = ROOT / "projects" / "daheng"
DEFAULT_POINTS = PROJECT / "outputs/0817/grouped_cv_model_comparison/full_fit_points.csv"
DEFAULT_GROUPED_CV_DIR = PROJECT / "outputs/0817/grouped_cv_model_comparison"
DEFAULT_CONFIG = ROOT / "configs/laser_model_fit_config.daheng.yaml"
DEFAULT_FROZEN_MODEL = PROJECT / "outputs/0818/c0_freeze/quadratic_graph.yaml"
DEFAULT_FREEZE_MANIFEST = PROJECT / "outputs/0818/c0_freeze/c0_freeze_manifest.json"
DEFAULT_AUDIT_SUMMARY = PROJECT / "outputs/0818/quadratic_residual_observability/audit_summary.json"
DEFAULT_GEOMETRY = PROJECT / "outputs/0818/pose_geometry_audit/pose_geometry_metrics.csv"
DEFAULT_OUTPUT = PROJECT / "outputs/0818/c0_027_effect"


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


def normalize_frame_id(value: Any) -> str:
    try:
        return f"{int(float(value)):03d}"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid frame_id={value!r}") from exc


def metric_values(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "point_count": 0,
            "bias_mm": math.nan,
            "mae_mm": math.nan,
            "rmse_mm": math.nan,
            "p95_abs_mm": math.nan,
            "p99_abs_mm": math.nan,
            "max_abs_mm": math.nan,
        }
    absolute = np.abs(array)
    return {
        "point_count": int(array.size),
        "bias_mm": float(np.mean(array)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(array * array))),
        "p95_abs_mm": float(np.percentile(absolute, 95)),
        "p99_abs_mm": float(np.percentile(absolute, 99)),
        "max_abs_mm": float(np.max(absolute)),
    }


def pct_change(before: float, after: float) -> float:
    if not np.isfinite(before) or abs(before) < 1.0e-15 or not np.isfinite(after):
        return math.nan
    return float(100.0 * (before - after) / before)


def load_inputs(
    points_path: Path,
    frozen_model_path: Path,
    freeze_manifest_path: Path,
    audit_summary_path: Path,
    config_path: Path,
    grouped_cv_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Perform the provenance/reuse audit and load only FIT/model metadata."""

    for path in (points_path, frozen_model_path, freeze_manifest_path, audit_summary_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    grouped_cv_report = grouped_cv_dir / "report.md"
    grouped_cv_pointwise = grouped_cv_dir / "cv_pointwise_quadratic_graph.csv"
    if not grouped_cv_report.is_file() or not grouped_cv_pointwise.is_file():
        raise FileNotFoundError("Existing Full-36 grouped-CV artifacts are incomplete")

    manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    frozen_hash = sha256_file(frozen_model_path)
    if frozen_hash != manifest["model"]["frozen_yaml_sha256"]:
        raise RuntimeError("Frozen C0 YAML hash does not match c0_freeze_manifest.json")
    if frozen_hash != audit.get("frozen_model_sha256"):
        raise RuntimeError("Frozen C0 YAML hash does not match the existing audit summary")
    if manifest.get("freeze_status") != "FROZEN" or manifest.get("final_c0_status") != "QUADRATIC":
        raise RuntimeError("Current C0 freeze manifest is not the expected Frozen Quadratic baseline")
    if manifest["fit_dataset"].get("pose_count") != 36 or manifest["fit_dataset"].get("point_count") != 32400:
        raise RuntimeError("Current Frozen C0 provenance is not Full-36/32400")
    if manifest["protocol"].get("mask") != "full_board_physical" or abs(float(manifest["protocol"].get("mask_inset_mm", 1.0))) > 1.0e-12:
        raise RuntimeError("Current Frozen C0 provenance is not the formal full_board_physical inset=0 protocol")
    if audit.get("validation_read") is not False or audit.get("c0_refit") is not False or audit.get("c1_fit") is not False:
        raise RuntimeError("Existing provenance audit does not prove the required FIT-only boundary")

    cfg = triplets.safe_yaml_load(config_path)
    extraction = cfg.get("extraction", {})
    if str(extraction.get("board_mask_mode", "")).lower() != "full_board_physical":
        raise RuntimeError("Formal extraction config is not full_board_physical")
    if abs(float(extraction.get("board_mask_inset_mm", 0.0))) > 1.0e-12:
        raise RuntimeError("Formal extraction config does not use inset=0")
    quadratic_cfg = cfg.get("models", {}).get("quadratic", {})
    if abs(float(quadratic_cfg.get("ridge", 1.0e-10)) - 1.0e-10) > 1.0e-20:
        raise RuntimeError("Quadratic ridge differs from the existing protocol")

    required = {
        "split",
        "frame_id",
        "frame_key",
        "image_id",
        "u_px",
        "v_px",
        "ray_x",
        "ray_y",
        "ray_z",
        "Xc_mm",
        "Yc_mm",
        "Zc_mm",
        "board_nx",
        "board_ny",
        "board_nz",
        "board_d_mm",
        "in_v_domain",
        "lambda_truth_mm",
    }
    points = pd.read_csv(points_path, encoding="utf-8-sig", dtype={"frame_id": str})
    missing = sorted(required.difference(points.columns))
    if missing:
        raise RuntimeError(f"Raw Full-36 FIT table missing fields: {missing}")
    points["frame_id"] = points["frame_id"].map(normalize_frame_id)
    if len(points) != 32400:
        raise RuntimeError(f"Expected 32400 Full-36 FIT points, got {len(points)}")
    if set(points["split"].astype(str).str.lower()) != {"fit"}:
        raise RuntimeError("Raw point table is not FIT-only")
    if tuple(sorted(points["frame_id"].unique())) != tuple(sorted(FIT_IDS)):
        raise RuntimeError("Raw point table does not contain exactly the Full-36 FIT poses")
    point_counts = points.groupby("frame_id").size()
    if not bool((point_counts == 900).all()):
        raise RuntimeError(f"Raw Full-36 point counts are not 900 per pose: {point_counts.to_dict()}")
    if not bool(points["in_v_domain"].astype(bool).all()):
        raise RuntimeError("Raw Full-36 FIT table contains points outside the formal v domain")
    numeric = ["u_px", "v_px", "ray_x", "ray_y", "ray_z", "Xc_mm", "Yc_mm", "Zc_mm", "lambda_truth_mm"]
    if not np.isfinite(points[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Raw Full-36 FIT point table contains non-finite ray/geometry values")
    points = points.reset_index(drop=True)

    provenance = {
        "points_path": str(points_path.resolve()),
        "points_sha256": sha256_file(points_path),
        "frozen_model_path": str(frozen_model_path.resolve()),
        "frozen_model_sha256": frozen_hash,
        "freeze_manifest_path": str(freeze_manifest_path.resolve()),
        "audit_summary_path": str(audit_summary_path.resolve()),
        "formal_config_path": str(config_path.resolve()),
        "formal_config_sha256": sha256_file(config_path),
        "grouped_cv_report": str(grouped_cv_report.resolve()),
        "grouped_cv_pointwise_reference": str(grouped_cv_pointwise.resolve()),
        "grouped_cv_pointwise_reference_sha256": sha256_file(grouped_cv_pointwise),
        "validation_read": False,
        "c1_trained": False,
        "frozen_c0_refit": False,
        "raw_fit_points_reused": True,
    }
    return points, cfg, {"manifest": manifest, "audit": audit, "provenance": provenance}


def fit_quadratic(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[triplets.PlaneModel, triplets.QuadraticGraphModel]:
    points, _, frame_keys = triplets.dataframe_arrays(df)
    plane = triplets.PlaneModel()
    plane.fit(points, frame_keys)
    ridge = float(cfg.get("models", {}).get("quadratic", {}).get("ridge", 1.0e-10))
    quadratic = triplets.QuadraticGraphModel(ridge=ridge)
    quadratic.fit(points, frame_keys, plane=plane)
    return plane, quadratic


def quadratic_from_dict(parameters: Mapping[str, Any]) -> triplets.QuadraticGraphModel:
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    model = triplets.QuadraticGraphModel(ridge=1.0e-10)
    model.dep_axis = axis_index[str(parameters["dependent_axis"])]
    model.ind_axes = tuple(axis_index[value] for value in parameters["independent_axes"])  # type: ignore[assignment]
    normalization = parameters["normalization"]
    model.center = np.asarray(normalization["independent_center_mm"], dtype=float)
    model.scale = np.asarray(normalization["independent_scale_mm"], dtype=float)
    model.beta = np.asarray(parameters["coefficients"], dtype=float)
    model.z_range = tuple(float(value) for value in parameters["z_valid_range_mm"])
    return model


def fold_assignment(frames: Sequence[str]) -> dict[str, int]:
    ordered = sorted(str(frame) for frame in frames)
    return {frame_id: index % FOLD_COUNT for index, frame_id in enumerate(ordered)}


def detail_with_context(detail: pd.DataFrame, test_df: pd.DataFrame, fold: int, heldout: Sequence[str]) -> pd.DataFrame:
    result = detail.copy().reset_index(drop=True)
    result["frame_id"] = test_df["frame_id"].astype(str).to_numpy()
    result["fold"] = int(fold)
    result["heldout_frames"] = ",".join(sorted(heldout))
    return result


def run_controlled_cv(points: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run both C0 arms against exactly the same operational test rows."""

    frame_map = fold_assignment(OPERATIONAL_IDS)
    full_details: list[pd.DataFrame] = []
    clean_details: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold_parameters: dict[str, Any] = {}
    point_identity_parts: list[str] = []

    for fold in range(FOLD_COUNT):
        heldout = sorted(frame for frame in OPERATIONAL_IDS if frame_map[frame] == fold)
        train_operational = [frame for frame in OPERATIONAL_IDS if frame not in heldout]
        train_full = [frame for frame in FIT_IDS if frame not in heldout]
        test_df = points[points["frame_id"].isin(heldout)].copy().reset_index(drop=True)
        train_full_df = points[points["frame_id"].isin(train_full)].copy()
        train_operational_df = points[points["frame_id"].isin(train_operational)].copy()
        if sorted(test_df["frame_id"].unique()) != heldout:
            raise RuntimeError(f"Fold {fold} test pose set is not the declared operational set")
        point_identity_parts.extend(
            f"{row.frame_id}|{row.frame_key}|{row.image_id}|{row.u_px:.12f}|{row.v_px:.12f}"
            for row in test_df.itertuples(index=False)
        )

        plane36, model36 = fit_quadratic(train_full_df, cfg)
        plane35, model35 = fit_quadratic(train_operational_df, cfg)
        _, raw36 = triplets.evaluate_model(model36, test_df, plane36)
        _, raw35 = triplets.evaluate_model(model35, test_df, plane35)
        detail36 = detail_with_context(raw36, test_df, fold, heldout)
        detail35 = detail_with_context(raw35, test_df, fold, heldout)
        full_details.append(detail36)
        clean_details.append(detail35)
        fold_parameters[str(fold)] = {
            "heldout_frames": heldout,
            "full36_training_frames": train_full,
            "operational35_training_frames": train_operational,
            "full36_model": model36.to_dict(),
            "operational35_model": model35.to_dict(),
            "full36_valid_rate": float(np.mean(detail36["valid"].to_numpy(dtype=bool))),
            "operational35_valid_rate": float(np.mean(detail35["valid"].to_numpy(dtype=bool))),
        }
        for label, detail, train_frames, excludes_027 in (
            ("C0-36", detail36, train_full, False),
            ("C0-35", detail35, train_operational, True),
        ):
            metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
            fold_rows.append(
                {
                    "row_type": "fold",
                    "scenario": "same35_operational_grouped_cv",
                    "model": label,
                    "fold": fold,
                    "heldout_frames": ",".join(heldout),
                    "training_frame_count": len(train_frames),
                    "training_point_count": int(len(points[points["frame_id"].isin(train_frames)])),
                    "training_excludes_027": excludes_027,
                    "evaluation_frame_count": len(heldout),
                    "evaluation_point_count": int(len(detail)),
                    "valid_rate": float(np.mean(detail["valid"].to_numpy(dtype=bool))),
                    **metric,
                }
            )

    full = pd.concat(full_details, ignore_index=True)
    clean = pd.concat(clean_details, ignore_index=True)
    if len(full) != len(clean) or len(full) != 35 * 900:
        raise RuntimeError("Controlled CV arms do not have the same 35-pose test point count")
    identity_full = full[["frame_id", "frame_key", "image_id", "u_px", "v_px"]].copy()
    identity_clean = clean[["frame_id", "frame_key", "image_id", "u_px", "v_px"]].copy()
    if not identity_full.equals(identity_clean):
        raise RuntimeError("Controlled CV arms do not have identical test point identity")
    identity_hash = hashlib.sha256("\n".join(point_identity_parts).encode("utf-8")).hexdigest()
    metadata = {
        "fold_by_frame": frame_map,
        "fold_parameters": fold_parameters,
        "test_point_identity_sha256": identity_hash,
        "test_point_count": len(full),
        "test_pose_count": int(full["frame_id"].nunique()),
    }
    return full, clean, pd.DataFrame(fold_rows), metadata


def bin_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    v = detail["v_px"].to_numpy(dtype=float)
    for index in range(V_BIN_COUNT):
        keep = (v >= index * V_BIN_WIDTH_PX) & (v < (index + 1) * V_BIN_WIDTH_PX)
        part = detail.loc[keep]
        metric = metric_values(part["board_error_mm"].to_numpy(dtype=float))
        rows.append(
            {
                "v_bin_index": index,
                "v_bin": f"v_{index * 100:04d}_{(index + 1) * 100:04d}",
                "v_bin_lo_px": float(index * 100),
                "v_bin_hi_px": float((index + 1) * 100),
                "point_count": int(len(part)),
                "unique_frame_count": int(part["frame_id"].nunique()),
                **metric,
            }
        )
    return pd.DataFrame(rows)


def aggregate_cv_row(model: str, detail: pd.DataFrame, training_counts: Sequence[int], excludes_027: bool) -> dict[str, Any]:
    metric = metric_values(detail["board_error_mm"].to_numpy(dtype=float))
    bins = bin_metrics(detail)
    valid_bins = bins[bins["point_count"] > 0]
    worst_rmse = valid_bins.loc[valid_bins["rmse_mm"].idxmax()] if not valid_bins.empty else pd.Series(dtype=float)
    worst_p95 = valid_bins.loc[valid_bins["p95_abs_mm"].idxmax()] if not valid_bins.empty else pd.Series(dtype=float)
    bias = valid_bins["bias_mm"].to_numpy(dtype=float) if not valid_bins.empty else np.asarray([], dtype=float)
    return {
        "row_type": "pooled",
        "scenario": "same35_operational_grouped_cv",
        "model": model,
        "fold": "ALL",
        "training_frame_count_min": int(min(training_counts)),
        "training_frame_count_max": int(max(training_counts)),
        "training_excludes_027": excludes_027,
        "evaluation_frame_count": int(detail["frame_id"].nunique()),
        "evaluation_point_count": int(len(detail)),
        "valid_rate": float(np.mean(detail["valid"].to_numpy(dtype=bool))),
        **metric,
        "v_bin_count": int(np.count_nonzero(bins["point_count"].to_numpy() > 0)),
        "worst_v_bin_rmse": str(worst_rmse.get("v_bin", "")),
        "worst_v_bin_rmse_mm": float(worst_rmse.get("rmse_mm", math.nan)),
        "worst_v_bin_rmse_p95_abs_mm": float(worst_rmse.get("p95_abs_mm", math.nan)),
        "worst_v_bin_p95": str(worst_p95.get("v_bin", "")),
        "worst_v_bin_p95_abs_mm": float(worst_p95.get("p95_abs_mm", math.nan)),
        "worst_v_bin_p95_rmse_mm": float(worst_p95.get("rmse_mm", math.nan)),
        "v_bias_min_mm": float(np.min(bias)) if bias.size else math.nan,
        "v_bias_max_mm": float(np.max(bias)) if bias.size else math.nan,
        "v_bias_range_mm": float(np.ptp(bias)) if bias.size else math.nan,
    }


def paired_cv_row(full: pd.DataFrame, clean: pd.DataFrame, pose_table: pd.DataFrame, v_table: pd.DataFrame) -> dict[str, Any]:
    full_metric = metric_values(full["board_error_mm"].to_numpy(dtype=float))
    clean_metric = metric_values(clean["board_error_mm"].to_numpy(dtype=float))
    full_bins = bin_metrics(full).set_index("v_bin_index")
    clean_bins = bin_metrics(clean).set_index("v_bin_index")
    full_bias = full_bins["bias_mm"].to_numpy(dtype=float)
    clean_bias = clean_bins["bias_mm"].to_numpy(dtype=float)
    pose_ratio = float(np.mean(pose_table["c0_35_better"].to_numpy(dtype=bool)))
    rmse_delta = float(clean_metric["rmse_mm"] - full_metric["rmse_mm"])
    p95_delta = float(clean_metric["p95_abs_mm"] - full_metric["p95_abs_mm"])
    return {
        "row_type": "paired_comparison",
        "scenario": "same35_operational_grouped_cv",
        "model": "C0-35_MINUS_C0-36",
        "fold": "ALL",
        "c0_36_bias_mm": full_metric["bias_mm"],
        "c0_35_bias_mm": clean_metric["bias_mm"],
        "delta_bias_mm": clean_metric["bias_mm"] - full_metric["bias_mm"],
        "c0_36_mae_mm": full_metric["mae_mm"],
        "c0_35_mae_mm": clean_metric["mae_mm"],
        "delta_mae_mm": clean_metric["mae_mm"] - full_metric["mae_mm"],
        "c0_36_rmse_mm": full_metric["rmse_mm"],
        "c0_35_rmse_mm": clean_metric["rmse_mm"],
        "delta_rmse_mm": rmse_delta,
        "rmse_improvement_pct": pct_change(full_metric["rmse_mm"], clean_metric["rmse_mm"]),
        "c0_36_p95_abs_mm": full_metric["p95_abs_mm"],
        "c0_35_p95_abs_mm": clean_metric["p95_abs_mm"],
        "delta_p95_abs_mm": p95_delta,
        "p95_improvement_pct": pct_change(full_metric["p95_abs_mm"], clean_metric["p95_abs_mm"]),
        "c0_36_p99_abs_mm": full_metric["p99_abs_mm"],
        "c0_35_p99_abs_mm": clean_metric["p99_abs_mm"],
        "delta_p99_abs_mm": clean_metric["p99_abs_mm"] - full_metric["p99_abs_mm"],
        "p99_improvement_pct": pct_change(full_metric["p99_abs_mm"], clean_metric["p99_abs_mm"]),
        "c0_36_worst_v_bin_rmse_mm": float(full_bins["rmse_mm"].max()),
        "c0_35_worst_v_bin_rmse_mm": float(clean_bins["rmse_mm"].max()),
        "delta_worst_v_bin_rmse_mm": float(clean_bins["rmse_mm"].max() - full_bins["rmse_mm"].max()),
        "worst_v_bin_rmse_improvement_pct": pct_change(float(full_bins["rmse_mm"].max()), float(clean_bins["rmse_mm"].max())),
        "c0_36_worst_v_bin_p95_abs_mm": float(full_bins["p95_abs_mm"].max()),
        "c0_35_worst_v_bin_p95_abs_mm": float(clean_bins["p95_abs_mm"].max()),
        "delta_worst_v_bin_p95_abs_mm": float(clean_bins["p95_abs_mm"].max() - full_bins["p95_abs_mm"].max()),
        "worst_v_bin_p95_improvement_pct": pct_change(float(full_bins["p95_abs_mm"].max()), float(clean_bins["p95_abs_mm"].max())),
        "c0_36_v_bias_range_mm": float(np.ptp(full_bias)),
        "c0_35_v_bias_range_mm": float(np.ptp(clean_bias)),
        "delta_v_bias_range_mm": float(np.ptp(clean_bias) - np.ptp(full_bias)),
        "pose_improvement_ratio": pose_ratio,
        "pose_better_count": int(np.count_nonzero(pose_table["c0_35_better"].to_numpy(dtype=bool))),
        "pose_count": int(len(pose_table)),
        "v_bin_count": int(len(v_table)),
        "test_point_identity_equal": True,
    }


def make_pose_table(full: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame_id in OPERATIONAL_IDS:
        part36 = full[full["frame_id"] == frame_id]
        part35 = clean[clean["frame_id"] == frame_id]
        m36 = metric_values(part36["board_error_mm"].to_numpy(dtype=float))
        m35 = metric_values(part35["board_error_mm"].to_numpy(dtype=float))
        rows.append(
            {
                "frame_id": frame_id,
                "fold": int(full.loc[full["frame_id"].eq(frame_id), "fold"].iloc[0]),
                "point_count": int(len(part36)),
                "c0_36_bias_mm": m36["bias_mm"],
                "c0_35_bias_mm": m35["bias_mm"],
                "delta_bias_mm": m35["bias_mm"] - m36["bias_mm"],
                "c0_36_mae_mm": m36["mae_mm"],
                "c0_35_mae_mm": m35["mae_mm"],
                "delta_mae_mm": m35["mae_mm"] - m36["mae_mm"],
                "c0_36_rmse_mm": m36["rmse_mm"],
                "c0_35_rmse_mm": m35["rmse_mm"],
                "delta_rmse_mm": m35["rmse_mm"] - m36["rmse_mm"],
                "rmse_improvement_pct": pct_change(m36["rmse_mm"], m35["rmse_mm"]),
                "c0_36_p95_abs_mm": m36["p95_abs_mm"],
                "c0_35_p95_abs_mm": m35["p95_abs_mm"],
                "delta_p95_abs_mm": m35["p95_abs_mm"] - m36["p95_abs_mm"],
                "p95_improvement_pct": pct_change(m36["p95_abs_mm"], m35["p95_abs_mm"]),
                "c0_36_p99_abs_mm": m36["p99_abs_mm"],
                "c0_35_p99_abs_mm": m35["p99_abs_mm"],
                "delta_p99_abs_mm": m35["p99_abs_mm"] - m36["p99_abs_mm"],
                "p99_improvement_pct": pct_change(m36["p99_abs_mm"], m35["p99_abs_mm"]),
                "c0_35_better": bool(m35["rmse_mm"] < m36["rmse_mm"]),
            }
        )
    return pd.DataFrame(rows)


def make_v_table(full: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    b36 = bin_metrics(full).set_index("v_bin_index")
    b35 = bin_metrics(clean).set_index("v_bin_index")
    rows: list[dict[str, Any]] = []
    for index in range(V_BIN_COUNT):
        a = b36.loc[index]
        b = b35.loc[index]
        rows.append(
            {
                "v_bin_index": index,
                "v_bin": a["v_bin"],
                "v_bin_lo_px": a["v_bin_lo_px"],
                "v_bin_hi_px": a["v_bin_hi_px"],
                "point_count": int(a["point_count"]),
                "unique_frame_count": int(a["unique_frame_count"]),
                "c0_36_bias_mm": a["bias_mm"],
                "c0_35_bias_mm": b["bias_mm"],
                "delta_bias_mm": b["bias_mm"] - a["bias_mm"],
                "c0_36_mae_mm": a["mae_mm"],
                "c0_35_mae_mm": b["mae_mm"],
                "delta_mae_mm": b["mae_mm"] - a["mae_mm"],
                "c0_36_rmse_mm": a["rmse_mm"],
                "c0_35_rmse_mm": b["rmse_mm"],
                "delta_rmse_mm": b["rmse_mm"] - a["rmse_mm"],
                "c0_36_p95_abs_mm": a["p95_abs_mm"],
                "c0_35_p95_abs_mm": b["p95_abs_mm"],
                "delta_p95_abs_mm": b["p95_abs_mm"] - a["p95_abs_mm"],
                "c0_36_p99_abs_mm": a["p99_abs_mm"],
                "c0_35_p99_abs_mm": b["p99_abs_mm"],
                "delta_p99_abs_mm": b["p99_abs_mm"] - a["p99_abs_mm"],
            }
        )
    return pd.DataFrame(rows)


def write_cv_outputs(
    output: Path,
    full: pd.DataFrame,
    clean: pd.DataFrame,
    fold_rows: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pose_table = make_pose_table(full, clean)
    v_table = make_v_table(full, clean)
    aggregate36 = aggregate_cv_row("C0-36", full, [30, 30, 30, 30, 30, 31], False)
    aggregate35 = aggregate_cv_row("C0-35", clean, [29, 29, 29, 29, 29, 30], True)
    paired = paired_cv_row(full, clean, pose_table, v_table)
    cv_table = pd.concat([pd.DataFrame(fold_rows), pd.DataFrame([aggregate36, aggregate35, paired])], ignore_index=True, sort=False)
    cv_table.to_csv(output / "c0_35_vs_36_cv.csv", index=False, encoding="utf-8-sig")
    pose_table.to_csv(output / "c0_35_vs_36_pose_metrics.csv", index=False, encoding="utf-8-sig")
    v_table.to_csv(output / "c0_35_vs_36_v_bins.csv", index=False, encoding="utf-8-sig")
    (output / "c0_35_vs_36_cv_folds.json").write_text(json.dumps(json_clean(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cv_table, pose_table, v_table


def equal_bin_edges(values: Sequence[float], count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi - lo <= 1.0e-12:
        return np.linspace(lo - 0.5, hi + 0.5, count + 1)
    return np.linspace(lo, hi, count + 1)


def value_bin(value: float, edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(edges, float(value), side="right") - 1, 0, len(edges) - 2))


def normal_angles(geometry: pd.DataFrame) -> np.ndarray:
    normals = geometry[["board_normal_x", "board_normal_y", "board_normal_z"]].to_numpy(dtype=float)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
    return np.degrees(np.arccos(np.clip(normals @ normals.T, -1.0, 1.0)))


def geometry_reference(geometry_path: Path, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "frame_id",
        "fit_group",
        "tvec_x_mm",
        "tvec_y_mm",
        "tvec_z_mm",
        "translation_norm_mm",
        "board_center_x_mm",
        "board_center_y_mm",
        "board_center_z_mm",
        "camera_board_distance_mm",
        "board_normal_x",
        "board_normal_y",
        "board_normal_z",
        "board_plane_distance_mm",
        "board_tilt_deg",
    ]
    if not geometry_path.is_file():
        raise FileNotFoundError(geometry_path)
    geometry = pd.read_csv(geometry_path, usecols=usecols)
    geometry["frame_id"] = geometry["frame_id"].map(normalize_frame_id)
    if set(geometry["frame_id"]) != set(FIT_IDS):
        raise RuntimeError("Existing geometry artifact does not contain exactly Full-36 poses")
    support = points[["frame_id", "v_px", "lambda_truth_mm", "Zc_mm", "in_v_domain"]].copy()
    support["frame_id"] = support["frame_id"].map(normalize_frame_id)
    support = support[support["in_v_domain"].astype(bool)].copy()
    support["v_grid_id"] = np.floor((support["v_px"] - V_MIN_PX) / V_GRID_WIDTH_PX).astype(int).clip(0, V_GRID_COUNT - 1)
    support["v_bin_id"] = np.floor((support["v_px"] - V_MIN_PX) / V_BIN_WIDTH_PX).astype(int).clip(0, V_BIN_COUNT - 1)
    return geometry, support


def geometry_stats(
    selected_ids: Sequence[str],
    reference_ids: Sequence[str],
    geometry: pd.DataFrame,
    support: pd.DataFrame,
    edges: Mapping[str, np.ndarray],
    normal_matrix: np.ndarray,
) -> dict[str, Any]:
    selected = list(selected_ids)
    reference = list(reference_ids)
    selected_set = set(selected)
    reference_set = set(reference)
    geom_by_id = geometry.set_index("frame_id")
    reference_geom = geometry[geometry["frame_id"].isin(reference_set)]
    selected_geom = geometry[geometry["frame_id"].isin(selected_set)]
    reference_points = support[support["frame_id"].isin(reference_set)]
    selected_points = support[support["frame_id"].isin(selected_set)]
    grid_sets = {frame: set(support.loc[support["frame_id"].eq(frame), "v_grid_id"].astype(int)) for frame in reference}
    bin_sets = {frame: set(support.loc[support["frame_id"].eq(frame), "v_bin_id"].astype(int)) for frame in reference}
    grid_union = set().union(*(grid_sets[frame] for frame in selected)) if selected else set()
    bin_union = set().union(*(bin_sets[frame] for frame in selected)) if selected else set()
    missing_grid = sorted(set(range(V_GRID_COUNT)) - grid_union)
    missing_bins = sorted(set(range(V_BIN_COUNT)) - bin_union)
    edge_counts = {str(bin_id): int(sum(bin_id in bin_sets[frame] for frame in selected)) for bin_id in EDGE_BIN_IDS}
    edge_min = min(edge_counts.values()) if edge_counts else 0

    reference_depth_range = float(reference_geom["board_center_z_mm"].max() - reference_geom["board_center_z_mm"].min())
    selected_depth_range = float(selected_geom["board_center_z_mm"].max() - selected_geom["board_center_z_mm"].min()) if not selected_geom.empty else 0.0
    reference_lambda_range = float(reference_points["lambda_truth_mm"].max() - reference_points["lambda_truth_mm"].min())
    selected_lambda_range = float(selected_points["lambda_truth_mm"].max() - selected_points["lambda_truth_mm"].min()) if not selected_points.empty else 0.0

    def covered(values: Sequence[float], bin_edges: np.ndarray) -> set[int]:
        return {value_bin(value, bin_edges) for value in values}

    reference_depth_bins = covered(reference_geom["board_center_z_mm"].to_numpy(), edges["depth"])
    selected_depth_bins = covered(selected_geom["board_center_z_mm"].to_numpy(), edges["depth"])
    reference_lambda_bins = covered(reference_points["lambda_truth_mm"].to_numpy(), edges["lambda"])
    selected_lambda_bins = covered(selected_points["lambda_truth_mm"].to_numpy(), edges["lambda"])
    reference_tx_bins = covered(reference_geom["board_center_x_mm"].to_numpy(), edges["tx"])
    selected_tx_bins = covered(selected_geom["board_center_x_mm"].to_numpy(), edges["tx"])
    reference_ty_bins = covered(reference_geom["board_center_y_mm"].to_numpy(), edges["ty"])
    selected_ty_bins = covered(selected_geom["board_center_y_mm"].to_numpy(), edges["ty"])

    ref_indices = [reference.index(frame) for frame in reference]
    selected_indices = [reference.index(frame) for frame in selected]
    nearest = np.min(normal_matrix[np.ix_(ref_indices, selected_indices)], axis=1) if selected_indices else np.full(len(reference), math.inf)
    selected_diameter = float(np.max(normal_matrix[np.ix_(selected_indices, selected_indices)])) if len(selected_indices) > 1 else 0.0
    reference_diameter = float(np.max(normal_matrix[np.ix_(ref_indices, ref_indices)])) if len(ref_indices) > 1 else 0.0
    missing_normals = [reference[index] for index, angle in enumerate(nearest) if angle > NORMAL_COVER_THRESHOLD_DEG]
    depth_ratio = selected_depth_range / reference_depth_range if reference_depth_range > 1.0e-12 else 1.0
    lambda_ratio = selected_lambda_range / reference_lambda_range if reference_lambda_range > 1.0e-12 else 1.0
    v_ok = not missing_grid and not missing_bins
    edge_ok = edge_min >= EDGE_MIN_FRAME_COUNT
    excitation_ok = bool(
        depth_ratio >= SPAN_RATIO_THRESHOLD
        and lambda_ratio >= SPAN_RATIO_THRESHOLD
        and selected_depth_bins == reference_depth_bins
        and selected_lambda_bins == reference_lambda_bins
    )
    normal_ok = bool(
        float(np.max(nearest)) <= NORMAL_COVER_THRESHOLD_DEG
        and selected_diameter >= reference_diameter - NORMAL_DIAMETER_TOLERANCE_DEG
    )
    translation_ok = selected_tx_bins == reference_tx_bins and selected_ty_bins == reference_ty_bins
    result = {
        "reference_pose_count": len(reference),
        "selected_pose_count": len(selected),
        "selected_pose_ids": ";".join(selected),
        "point_count": int(len(selected_points)),
        "v_grid_10px_occupied_count": len(grid_union),
        "v_grid_10px_missing_count": len(missing_grid),
        "v_grid_10px_missing_ids": ";".join(str(value) for value in missing_grid),
        "v_bin_100px_occupied_count": len(bin_union),
        "v_bin_100px_missing_count": len(missing_bins),
        "v_bin_100px_missing_ids": ";".join(str(value) for value in missing_bins),
        "v_continuous_ok": v_ok,
        "edge_frame_count_0_100": edge_counts.get("0", 0),
        "edge_frame_count_100_200": edge_counts.get("1", 0),
        "edge_frame_count_2800_2900": edge_counts.get("28", 0),
        "edge_frame_count_2900_3000": edge_counts.get("29", 0),
        "edge_min_frame_count": edge_min,
        "edge_multiframe_ok": edge_ok,
        "depth_center_min_mm": float(selected_geom["board_center_z_mm"].min()) if not selected_geom.empty else math.nan,
        "depth_center_max_mm": float(selected_geom["board_center_z_mm"].max()) if not selected_geom.empty else math.nan,
        "depth_center_span_mm": selected_depth_range,
        "depth_span_ratio": depth_ratio,
        "lambda_truth_min_mm": float(selected_points["lambda_truth_mm"].min()) if not selected_points.empty else math.nan,
        "lambda_truth_max_mm": float(selected_points["lambda_truth_mm"].max()) if not selected_points.empty else math.nan,
        "lambda_truth_span_mm": selected_lambda_range,
        "lambda_span_ratio": lambda_ratio,
        "depth_bin_occupied_count": len(selected_depth_bins),
        "depth_bin_missing_count": len(reference_depth_bins - selected_depth_bins),
        "lambda_bin_occupied_count": len(selected_lambda_bins),
        "lambda_bin_missing_count": len(reference_lambda_bins - selected_lambda_bins),
        "depth_lambda_excitation_ok": excitation_ok,
        "translation_x_bin_occupied_count": len(selected_tx_bins),
        "translation_y_bin_occupied_count": len(selected_ty_bins),
        "translation_coverage_ok": translation_ok,
        "reference_normal_pairwise_diameter_deg": reference_diameter,
        "selected_normal_pairwise_diameter_deg": selected_diameter,
        "normal_cover_max_angle_deg": float(np.max(nearest)) if nearest.size else math.inf,
        "normal_cover_p95_angle_deg": float(np.percentile(nearest, 95)) if nearest.size else math.inf,
        "normal_cover_missing_pose_count": len(missing_normals),
        "normal_cover_missing_pose_ids": ";".join(missing_normals),
        "normal_angle_diversity_ok": normal_ok,
    }
    result["overall_geometry_ok"] = bool(v_ok and edge_ok and excitation_ok and normal_ok and translation_ok)
    return result


def run_geometry_audit(geometry_path: Path, points: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    geometry, support = geometry_reference(geometry_path, points)
    op_geometry = geometry[geometry["frame_id"].isin(OPERATIONAL_IDS)].copy()
    op_support = support[support["frame_id"].isin(OPERATIONAL_IDS)].copy()
    ordered_geometry = op_geometry.set_index("frame_id").loc[list(OPERATIONAL_IDS)].reset_index()
    ordered_support = op_support
    edges = {
        "depth": equal_bin_edges(ordered_geometry["board_center_z_mm"].to_numpy(), EXCITATION_BIN_COUNT),
        "lambda": equal_bin_edges(ordered_support["lambda_truth_mm"].to_numpy(), EXCITATION_BIN_COUNT),
        "tx": equal_bin_edges(ordered_geometry["board_center_x_mm"].to_numpy(), TRANSLATION_BIN_COUNT),
        "ty": equal_bin_edges(ordered_geometry["board_center_y_mm"].to_numpy(), TRANSLATION_BIN_COUNT),
    }
    matrix = normal_angles(ordered_geometry)
    full_stats = geometry_stats(OPERATIONAL_IDS, OPERATIONAL_IDS, ordered_geometry, ordered_support, edges, matrix)
    loo_rows: list[dict[str, Any]] = []
    for removed in OPERATIONAL_IDS:
        remaining = [frame for frame in OPERATIONAL_IDS if frame != removed]
        stats = geometry_stats(remaining, OPERATIONAL_IDS, ordered_geometry, ordered_support, edges, matrix)
        losses = [
            name
            for name, ok in (
                ("v", stats["v_continuous_ok"]),
                ("edge", stats["edge_multiframe_ok"]),
                ("depth_lambda", stats["depth_lambda_excitation_ok"]),
                ("normal", stats["normal_angle_diversity_ok"]),
                ("translation", stats["translation_coverage_ok"]),
            )
            if not ok
        ]
        loo_rows.append({"removed_frame_id": removed, "lost_gates": ";".join(losses), "overall_geometry_ok": stats["overall_geometry_ok"], **stats})
    loo = pd.DataFrame(loo_rows)
    summary = {
        "geometry_source": str(geometry_path.resolve()),
        "point_source": "raw Full-36 FIT points filtered to frame027 excluded",
        "reference_pose_ids": list(OPERATIONAL_IDS),
        "excluded_frame_id": FRAME027,
        "excluded_frame027_tilt_deg": float(geometry.loc[geometry["frame_id"].eq(FRAME027), "board_tilt_deg"].iloc[0]),
        "operational_tilt_min_deg": float(ordered_geometry["board_tilt_deg"].min()),
        "operational_tilt_max_deg": float(ordered_geometry["board_tilt_deg"].max()),
        "operational_normal_pairwise_diameter_deg": full_stats["reference_normal_pairwise_diameter_deg"],
        "full_stats": full_stats,
        "loo_failure_count": int(np.count_nonzero(~loo["overall_geometry_ok"].to_numpy(dtype=bool))),
        "loo_loss_counts": {
            gate: int(sum(gate in value.split(";") for value in loo["lost_gates"].astype(str)))
            for gate in ("v", "edge", "depth_lambda", "normal", "translation")
        },
        "edges": {key: value.tolist() for key, value in edges.items()},
    }
    return summary, loo


def surface_metrics(values: np.ndarray) -> dict[str, Any]:
    return metric_values(values)


def run_surface_delta(
    points: pd.DataFrame,
    frozen_model_path: Path,
    candidate_model: triplets.QuadraticGraphModel,
    full_plane: triplets.PlaneModel,
) -> pd.DataFrame:
    frozen = quadratic_from_dict(triplets.safe_yaml_load(frozen_model_path))
    operational = points[points["frame_id"].isin(OPERATIONAL_IDS)].copy().reset_index(drop=True)
    rays = operational[["ray_x", "ray_y", "ray_z"]].to_numpy(dtype=float)
    plane_hint = full_plane.intersect_rays(rays)
    lambda36 = frozen.intersect_rays(rays, lambda_hint=plane_hint)
    lambda35 = candidate_model.intersect_rays(rays, lambda_hint=plane_hint)
    delta = lambda35 - lambda36
    valid = np.isfinite(lambda35) & np.isfinite(lambda36) & np.isfinite(delta)
    operational["lambda_c0_36_mm"] = lambda36
    operational["lambda_c0_35_mm"] = lambda35
    operational["delta_lambda_mm"] = delta
    rows: list[dict[str, Any]] = []
    for label, keep in [("Global", np.ones(len(operational), dtype=bool))] + [
        (f"v_{index * 100:04d}_{(index + 1) * 100:04d}", (operational["v_px"].to_numpy() >= index * 100) & (operational["v_px"].to_numpy() < (index + 1) * 100))
        for index in range(V_BIN_COUNT)
    ]:
        use = keep & valid
        metric = surface_metrics(delta[use])
        rows.append(
            {
                "row_type": "global" if label == "Global" else "v_bin",
                "delta_direction": "C0-35 minus C0-36",
                "v_bin": label,
                "v_bin_lo_px": 0.0 if label == "Global" else float(int(label[2:6])),
                "v_bin_hi_px": 3000.0 if label == "Global" else float(int(label[7:11])),
                "point_count_grid": int(np.count_nonzero(keep)),
                "point_count_valid_pair": int(np.count_nonzero(use)),
                "pose_count": int(operational.loc[keep, "frame_id"].nunique()),
                "valid_pair_rate": float(np.mean(valid[keep])) if np.any(keep) else math.nan,
                "delta_lambda_bias_mm": metric["bias_mm"],
                "delta_lambda_mae_mm": metric["mae_mm"],
                "delta_lambda_rmse_mm": metric["rmse_mm"],
                "delta_lambda_p95_abs_mm": metric["p95_abs_mm"],
                "delta_lambda_p99_abs_mm": metric["p99_abs_mm"],
                "delta_lambda_max_abs_mm": metric["max_abs_mm"],
            }
        )
    if not np.all(valid):
        raise RuntimeError("Full-fit surface delta has invalid lambda on the common operational ray grid")
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def classify_effect(cv_table: pd.DataFrame, surface: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    paired = cv_table[cv_table["row_type"].eq("paired_comparison")].iloc[0]
    surface_global = surface[surface["row_type"].eq("global")].iloc[0]
    changes = {
        "rmse_improvement_pct": float(paired["rmse_improvement_pct"]),
        "p95_improvement_pct": float(paired["p95_improvement_pct"]),
        "p99_improvement_pct": float(paired["p99_improvement_pct"]),
        "worst_v_bin_rmse_improvement_pct": float(paired["worst_v_bin_rmse_improvement_pct"]),
        "worst_v_bin_p95_improvement_pct": float(paired["worst_v_bin_p95_improvement_pct"]),
        "v_bias_range_delta_mm": float(paired["delta_v_bias_range_mm"]),
    }
    global_percent_changes = [
        changes["rmse_improvement_pct"],
        changes["p95_improvement_pct"],
        changes["p99_improvement_pct"],
    ]
    tail_percent_changes = [
        changes["worst_v_bin_rmse_improvement_pct"],
        changes["worst_v_bin_p95_improvement_pct"],
    ]
    global_cv_material = any(abs(value) >= CV_CHANGE_MATERIAL_PCT for value in global_percent_changes if np.isfinite(value))
    tail_cv_material = any(abs(value) >= CV_CHANGE_MATERIAL_PCT for value in tail_percent_changes if np.isfinite(value))
    meaningful_cv = global_cv_material or tail_cv_material
    # Bias range is compared against its baseline in a relative way below;
    # retain it as a diagnostic rather than letting a near-zero denominator
    # decide the classification.
    baseline_range = float(paired["c0_36_v_bias_range_mm"])
    range_change_pct = 100.0 * float(paired["delta_v_bias_range_mm"]) / baseline_range if abs(baseline_range) > 1.0e-15 else math.nan
    meaningful_cv = meaningful_cv or (np.isfinite(range_change_pct) and abs(range_change_pct) >= CV_CHANGE_MATERIAL_PCT)
    surface_rmse = float(surface_global["delta_lambda_rmse_mm"])
    surface_p95 = float(surface_global["delta_lambda_p95_abs_mm"])
    surface_max = float(surface_global["delta_lambda_max_abs_mm"])
    surface_negligible = bool(
        surface_rmse <= SURFACE_NEGLIGIBLE_RMSE_MM
        and surface_p95 <= SURFACE_NEGLIGIBLE_P95_MM
        and surface_max <= SURFACE_NEGLIGIBLE_MAX_MM
    )
    surface_material = bool(
        surface_rmse >= SURFACE_MATERIAL_RMSE_MM
        or surface_p95 >= SURFACE_MATERIAL_P95_MM
        or surface_max >= SURFACE_MATERIAL_MAX_MM
    )
    signs = [value > 0.0 for value in changes.values() if np.isfinite(value) and abs(value) >= 0.25]
    mixed_signs = bool(signs and any(value != signs[0] for value in signs))
    if surface_negligible and not meaningful_cv:
        effect = "NEGLIGIBLE"
    elif surface_material or global_cv_material:
        effect = "MATERIAL"
    else:
        effect = "MIXED"
    diagnostics = {
        "cv_change_material": meaningful_cv,
        "global_cv_material": global_cv_material,
        "tail_cv_material": tail_cv_material,
        "v_bias_range_change_pct": range_change_pct,
        "surface_negligible": surface_negligible,
        "surface_material": surface_material,
        "mixed_cv_signs": mixed_signs,
        "surface_rmse_mm": surface_rmse,
        "surface_p95_abs_mm": surface_p95,
        "surface_max_abs_mm": surface_max,
        "cv_changes": changes,
    }
    return effect, diagnostics


def write_geometry_audit(output: Path, summary: Mapping[str, Any], loo: pd.DataFrame) -> None:
    stats = summary["full_stats"]
    lines = [
        "# Operational-35 geometry-only coverage audit",
        "",
        "`AUDIT_SCOPE = OPERATIONAL_35`",
        "",
        "## Boundary",
        "",
        "- FIT-only geometry audit；只使用现有 Full-36 PnP geometry artifact 与正式 `full_board_physical` mask 的 raw FIT point support。",
        "- `frame027` 状态：`EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN`；原因是超出实际工作姿态域，不是 residual-based deletion。",
        "- Operational reference domain = Full-36 minus 027，共 35 poses；027 保留在原始 Full-36 artifacts 中，本 audit 不修改它们。",
        "- Normal coverage、depth/lambda span、translation bins 的 reference 全部重新由这 35 个 operational poses 建立；不再要求覆盖 027 的超大倾角法向。",
        "- 不读取 Validation；不读取模型 residual；不拟合 Plane/Quadratic/Cone。",
        "",
        "## Operational-35 reference range",
        "",
        f"- Pose count / points: **{stats['selected_pose_count']} / {stats['point_count']}**",
        f"- Board tilt: **{summary['operational_tilt_min_deg']:.3f}–{summary['operational_tilt_max_deg']:.3f}°**; excluded 027: **{summary['excluded_frame027_tilt_deg']:.3f}°**",
        f"- Operational normal pairwise diameter: **{summary['operational_normal_pairwise_diameter_deg']:.3f}°**",
        f"- Board-center Z: **{stats['depth_center_min_mm']:.3f}–{stats['depth_center_max_mm']:.3f} mm**, span **{stats['depth_center_span_mm']:.3f} mm**",
        f"- Lambda truth: **{stats['lambda_truth_min_mm']:.3f}–{stats['lambda_truth_max_mm']:.3f} mm**, span **{stats['lambda_truth_span_mm']:.3f} mm**",
        "",
        "## Coverage gates evaluated against the 35-pose domain",
        "",
        "| gate | Operational-35 result | pass |",
        "|---|---:|:---:|",
        f"| v 10 px occupied cells | {stats['v_grid_10px_occupied_count']}/{V_GRID_COUNT} | {stats['v_continuous_ok']} |",
        f"| v 100 px occupied bins | {stats['v_bin_100px_occupied_count']}/{V_BIN_COUNT} | {stats['v_continuous_ok']} |",
        f"| edge frame count 0–100 / 100–200 / 2800–2900 / 2900–3000 | {stats['edge_frame_count_0_100']} / {stats['edge_frame_count_100_200']} / {stats['edge_frame_count_2800_2900']} / {stats['edge_frame_count_2900_3000']} | {stats['edge_multiframe_ok']} |",
        f"| depth span ratio to operational reference | {stats['depth_span_ratio']:.4f} | {stats['depth_lambda_excitation_ok']} |",
        f"| lambda span ratio to operational reference | {stats['lambda_span_ratio']:.4f} | {stats['depth_lambda_excitation_ok']} |",
        f"| depth/lambda bins | {stats['depth_bin_occupied_count']}/{EXCITATION_BIN_COUNT}; {stats['lambda_bin_occupied_count']}/{EXCITATION_BIN_COUNT} | {stats['depth_lambda_excitation_ok']} |",
        f"| normal cover max angle to operational reference | {stats['normal_cover_max_angle_deg']:.4f}° | {stats['normal_angle_diversity_ok']} |",
        f"| normal diameter | {stats['selected_normal_pairwise_diameter_deg']:.4f}° / reference {stats['reference_normal_pairwise_diameter_deg']:.4f}° | {stats['normal_angle_diversity_ok']} |",
        f"| translation X/Y bins | {stats['translation_x_bin_occupied_count']}/{TRANSLATION_BIN_COUNT}; {stats['translation_y_bin_occupied_count']}/{TRANSLATION_BIN_COUNT} | {stats['translation_coverage_ok']} |",
        f"| overall geometry | — | **{stats['overall_geometry_ok']}** |",
        "",
        "## Leave-one-operational-pose diagnostic",
        "",
        f"- Removing one of the 35 poses was checked against the same Operational-35 reference gates. Failed cases: **{summary['loo_failure_count']}/35**.",
        "",
        "| removed pose | lost gate(s) |",
        "|---|---|",
    ]
    failed = loo[~loo["overall_geometry_ok"]]
    if failed.empty:
        lines.append("| none | none |")
    else:
        for row in failed.itertuples(index=False):
            lines.append(f"| {row.removed_frame_id} | {row.lost_gates or 'none'} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "本 audit 的 `normal_angle_diversity_ok` 只回答 Operational-35 是否覆盖自己的实际法向域；它不回答 027 所代表的超大倾角域是否仍被覆盖。Full-36 的原始 normal diameter 与原 audit 结论保持不变，未被本文件改写。",
            "",
            "`GEOMETRY_ONLY_OPERATIONAL35 = " + ("PASS" if stats["overall_geometry_ok"] else "FAIL") + "`",
            "",
            "## Reuse/new calculation boundary",
            "",
            "- Reused: existing Full-36 PnP geometry rows and raw Full-36 `full_board_physical` FIT point support.",
            "- Newly calculated: 027-filtered support unions, operational re-ranged bins, normal matrix/reference, and leave-one-operational-pose checks.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output: Path,
    provenance: Mapping[str, Any],
    cv_table: pd.DataFrame,
    pose_table: pd.DataFrame,
    v_table: pd.DataFrame,
    surface: pd.DataFrame,
    geometry_summary: Mapping[str, Any],
    effect: str,
    diagnostics: Mapping[str, Any],
) -> None:
    paired = cv_table[cv_table["row_type"].eq("paired_comparison")].iloc[0]
    surface_global = surface[surface["row_type"].eq("global")].iloc[0]
    g = geometry_summary["full_stats"]
    frame_wins = int(paired["pose_better_count"])
    lines = [
        "# C0 frame027 effect: Operational-35 vs Full-36 controlled FIT-only A/B",
        "",
        f"`C0_027_EFFECT = {effect}`",
        "",
        "## Decision summary",
        "",
        f"- Same-35-pose grouped-CV pose improvement ratio (C0-35 RMSE better than C0-36): **{paired['pose_improvement_ratio']:.3f} ({frame_wins}/{int(paired['pose_count'])})**.",
        f"- Global CV RMSE: C0-36 **{paired['c0_36_rmse_mm']:.6f} mm** → C0-35 **{paired['c0_35_rmse_mm']:.6f} mm**; improvement **{paired['rmse_improvement_pct']:.3f}%**.",
        f"- Global CV P95: C0-36 **{paired['c0_36_p95_abs_mm']:.6f} mm** → C0-35 **{paired['c0_35_p95_abs_mm']:.6f} mm**; improvement **{paired['p95_improvement_pct']:.3f}%**.",
        f"- Global CV P99: C0-36 **{paired['c0_36_p99_abs_mm']:.6f} mm** → C0-35 **{paired['c0_35_p99_abs_mm']:.6f} mm**; improvement **{paired['p99_improvement_pct']:.3f}%**.",
        f"- Full-fit surface delta direction is `lambda(C0-35) - lambda(C0-36)` on the same 31,500 operational raw rays: RMSE **{surface_global['delta_lambda_rmse_mm']:.6f} mm**, P95 **{surface_global['delta_lambda_p95_abs_mm']:.6f} mm**, Max **{surface_global['delta_lambda_max_abs_mm']:.6f} mm**.",
        f"- Operational-35 geometry-only audit: **{'PASS' if g['overall_geometry_ok'] else 'FAIL'}** against its own 35-pose reference domain; 027's excluded normal is not a required coverage target.",
        "",
        "## Scope and hard constraints",
        "",
        "- Current Frozen Full-36 Quadratic C0 remains byte-for-byte untouched and is loaded only as baseline.",
        "- C0-35 candidate is the only newly fitted production-like surface; only frame027 is excluded from the raw Full-36 FIT table.",
        "- CV is FIT-only on the same 35 operational poses with deterministic 6-fold pose grouping; the Full-36 arm keeps 027 in each fold's training set, while the Operational-35 arm excludes it.",
        "- Validation is not read. C1 is not imported or trained. No production config is modified.",
        "- This result is an A/B audit only; it does not replace Frozen C0-36.",
        "",
        "## Artifact provenance / reuse audit",
        "",
        "| artifact | action | status |",
        "|---|---|---|",
        f"| Frozen Full-36 C0 YAML | LOADED_ONLY | SHA256 `{provenance['frozen_model_sha256']}`; no frozen fit/write |",
        f"| Raw Full-36 FIT points | REUSED_EXISTING | `{provenance['points_sha256']}`; 36 poses / 32,400 points; formal FIT-only table |",
        f"| Formal extraction config | REUSED_EXISTING | SHA256 `{provenance['formal_config_sha256']}`; `full_board_physical`, inset 0 |",
        f"| Existing Full-36 grouped-CV artifacts | PROTOCOL_REFERENCE_ONLY | `{provenance['grouped_cv_report']}` and pointwise reference; not substituted for same-35 controlled CV because their held-out domain includes 027 |",
        f"| Existing PnP geometry artifact | REUSED_EXISTING | `{geometry_summary['geometry_source']}`; geometry columns only |",
        "| C0-35 quadratic candidate | NEW_FIT | raw FIT points excluding only 027; same QuadraticGraphModel.fit protocol |",
        "| Operational-35 CV / surface delta / geometry re-range | NEW_CALCULATION | generated in this directory |",
        "| Validation / C1 | NOT_READ / NOT_TRAINED | excluded by design |",
        "",
        "## Fixed fitting protocol",
        "",
        "- Model: `quadratic_graph`, dependent axis X, independent axes Y/Z.",
        "- Frame-balanced weights, existing robust 10-iteration QuadraticGraphModel fit, ridge `1e-10`; no new tuning or weighting.",
        "- Formal mask: `full_board_physical`, inset `0 mm`; v domain `[0,3000)` and 100 px reporting bins.",
        "- The C0-35 full-fit YAML is written as a candidate artifact only; it is not copied into the Frozen C0 directory.",
        "",
        "## Controlled grouped-CV comparison",
        "",
        "| metric | C0-36 (027 kept in training) | C0-35 (027 excluded) | C0-35 minus C0-36 / improvement |",
        "|---|---:|---:|---:|",
        f"| Bias / mm | {fmt(paired['c0_36_bias_mm'])} | {fmt(paired['c0_35_bias_mm'])} | Δ {fmt(paired['delta_bias_mm'])} |",
        f"| MAE / mm | {fmt(paired['c0_36_mae_mm'])} | {fmt(paired['c0_35_mae_mm'])} | Δ {fmt(paired['delta_mae_mm'])} |",
        f"| RMSE / mm | {fmt(paired['c0_36_rmse_mm'])} | {fmt(paired['c0_35_rmse_mm'])} | Δ {fmt(paired['delta_rmse_mm'])}; {fmt(paired['rmse_improvement_pct'], 3)}% |",
        f"| P95 abs / mm | {fmt(paired['c0_36_p95_abs_mm'])} | {fmt(paired['c0_35_p95_abs_mm'])} | Δ {fmt(paired['delta_p95_abs_mm'])}; {fmt(paired['p95_improvement_pct'], 3)}% |",
        f"| P99 abs / mm | {fmt(paired['c0_36_p99_abs_mm'])} | {fmt(paired['c0_35_p99_abs_mm'])} | Δ {fmt(paired['delta_p99_abs_mm'])}; {fmt(paired['p99_improvement_pct'], 3)}% |",
        f"| worst-v RMSE / mm | {fmt(paired['c0_36_worst_v_bin_rmse_mm'])} | {fmt(paired['c0_35_worst_v_bin_rmse_mm'])} | Δ {fmt(paired['delta_worst_v_bin_rmse_mm'])}; {fmt(paired['worst_v_bin_rmse_improvement_pct'], 3)}% |",
        f"| worst-v P95 abs / mm | {fmt(paired['c0_36_worst_v_bin_p95_abs_mm'])} | {fmt(paired['c0_35_worst_v_bin_p95_abs_mm'])} | Δ {fmt(paired['delta_worst_v_bin_p95_abs_mm'])}; {fmt(paired['worst_v_bin_p95_improvement_pct'], 3)}% |",
        f"| v-bias range / mm | {fmt(paired['c0_36_v_bias_range_mm'])} | {fmt(paired['c0_35_v_bias_range_mm'])} | Δ {fmt(paired['delta_v_bias_range_mm'])} ({fmt(diagnostics['v_bias_range_change_pct'], 3)}%) |",
        f"| pose improvement ratio | — | — | **{paired['pose_improvement_ratio']:.3f} ({frame_wins}/{int(paired['pose_count'])})** |",
        "",
        "P95/P99/Max use absolute error; CV Bias/MAE/RMSE are pooled over the identical held-out operational point set. The point identity equality is checked in code and recorded by hash in `c0_35_vs_36_cv_folds.json`.",
        "",
        "## Full-fit surface delta on unified Operational-35 ray grid",
        "",
        "- Grid: raw `full_fit_points.csv` rays for the same 35 poses, 900 rows per pose, 31,500 rows total.",
        "- Both full-fit surfaces use the same rays and the same plane root-selection hint; invalid pairs are not silently dropped (this run requires 100% valid pair rate).",
        "- `c0_surface_delta.csv` reports global and each 100 px v-bin metrics.",
        "",
        f"| global delta metric | value |",
        f"|---|---:|",
        f"| Bias / mm | {fmt(surface_global['delta_lambda_bias_mm'])} |",
        f"| MAE / mm | {fmt(surface_global['delta_lambda_mae_mm'])} |",
        f"| RMSE / mm | {fmt(surface_global['delta_lambda_rmse_mm'])} |",
        f"| P95 abs / mm | {fmt(surface_global['delta_lambda_p95_abs_mm'])} |",
        f"| P99 abs / mm | {fmt(surface_global['delta_lambda_p99_abs_mm'])} |",
        f"| Max abs / mm | {fmt(surface_global['delta_lambda_max_abs_mm'])} |",
        "",
        "## Operational-35 geometry-only audit",
        "",
        f"- 35-pose reference normal diameter: **{geometry_summary['operational_normal_pairwise_diameter_deg']:.3f}°**; excluded 027 tilt: **{geometry_summary['excluded_frame027_tilt_deg']:.3f}°**.",
        f"- v coverage: **{g['v_grid_10px_occupied_count']}/{V_GRID_COUNT}** 10 px cells and **{g['v_bin_100px_occupied_count']}/{V_BIN_COUNT}** 100 px bins; edge minimum **{g['edge_min_frame_count']}** poses.",
        f"- depth/lambda span ratios: **{g['depth_span_ratio']:.4f} / {g['lambda_span_ratio']:.4f}** relative to Operational-35 itself.",
        f"- LOO failures against the Operational-35 reference: **{geometry_summary['loo_failure_count']}/35**; this is a redundancy diagnostic, not a reason to delete a pose in this A/B.",
        "",
        "## Classification rule and interpretation",
        "",
        "- `NEGLIGIBLE`: surface delta is within RMSE/P95/Max ≤ 0.010/0.030/0.100 mm and no primary CV change reaches 2%.",
        "- `MATERIAL`: surface delta reaches the declared material diagnostic margin or a primary global CV metric reaches 2%.",
        "- `MIXED`: global CV and surface are negligible, but tail-v/pose evidence still changes at the diagnostic level (or dimensions disagree).",
        f"- This run: surface-negligible={diagnostics['surface_negligible']}, surface-material={diagnostics['surface_material']}, CV-change-material={diagnostics['cv_change_material']}, mixed-signs={diagnostics['mixed_cv_signs']}.",
        f"- **Conclusion: `C0_027_EFFECT = {effect}`.** This is an audit classification only; the current Frozen C0-36 remains the production baseline and is not replaced by C0-35.",
        "",
        "## Outputs",
        "",
        "- `c0_35_vs_36_cv.csv`",
        "- `c0_35_vs_36_pose_metrics.csv`",
        "- `c0_35_vs_36_v_bins.csv`",
        "- `c0_surface_delta.csv`",
        "- `operational35_geometry_audit.md`",
        "- `report.md`",
    ]
    output.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--grouped-cv-dir", type=Path, default=DEFAULT_GROUPED_CV_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frozen-model", type=Path, default=DEFAULT_FROZEN_MODEL)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    frozen_model = args.frozen_model.resolve()
    if output == frozen_model.parent or output in frozen_model.parents:
        raise RuntimeError("Refusing to write the Frozen C0 directory or its parents")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    points, cfg, audit_context = load_inputs(
        args.points.resolve(),
        frozen_model,
        args.freeze_manifest.resolve(),
        args.audit_summary.resolve(),
        args.config.resolve(),
        args.grouped_cv_dir.resolve(),
    )
    provenance = audit_context["provenance"]
    clean_points = points[points["frame_id"] != FRAME027].copy().reset_index(drop=True)
    full_points, _, full_frame_keys = triplets.dataframe_arrays(points)
    full_plane = triplets.PlaneModel()
    full_plane.fit(full_points, full_frame_keys)
    _, candidate_model = fit_quadratic(clean_points, cfg)
    candidate_yaml = output / "candidate_c0_35_quadratic.yaml"
    candidate_yaml.write_text(
        yaml.safe_dump(candidate_model.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    cv_full, cv_clean, fold_rows, cv_metadata = run_controlled_cv(points, cfg)
    cv_table, pose_table, v_table = write_cv_outputs(output, cv_full, cv_clean, fold_rows, cv_metadata)
    surface = run_surface_delta(points, frozen_model, candidate_model, full_plane)
    surface.to_csv(output / "c0_surface_delta.csv", index=False, encoding="utf-8-sig")
    geometry_summary, loo = run_geometry_audit(args.geometry.resolve(), points)
    write_geometry_audit(output / "operational35_geometry_audit.md", geometry_summary, loo)
    effect, diagnostics = classify_effect(cv_table, surface)

    manifest = {
        "C0_027_EFFECT": effect,
        "validation_read": False,
        "c1_trained": False,
        "frozen_c0_modified": False,
        "frame027_status": "EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN",
        "frame027_exclusion_reason": "超出实际工作姿态域",
        "fit_ids": list(FIT_IDS),
        "operational_ids": list(OPERATIONAL_IDS),
        "candidate_fit_ids": list(OPERATIONAL_IDS),
        "candidate_point_count": len(clean_points),
        "grouped_cv_folds": FOLD_COUNT,
        "cv_test_point_identity_sha256": cv_metadata["test_point_identity_sha256"],
        "cv_test_point_count": cv_metadata["test_point_count"],
        "candidate_yaml": str(candidate_yaml.resolve()),
        "candidate_yaml_sha256": sha256_file(candidate_yaml),
        "provenance": provenance,
        "geometry_summary": geometry_summary,
        "classification_diagnostics": diagnostics,
        "surface_grid": "raw Full-36 FIT points filtered to the same 35 operational poses; 31500 rays",
    }
    (output / "c0_027_effect_manifest.json").write_text(json.dumps(json_clean(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output, provenance, cv_table, pose_table, v_table, surface, geometry_summary, effect, diagnostics)
    print(json.dumps({"output_dir": str(output), "C0_027_EFFECT": effect, "cv_test_points": len(cv_full), "surface_rows": len(surface)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
