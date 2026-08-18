#!/usr/bin/env python3
"""Final independent Validation comparison for frozen Full-36 Q/C models.

This script deliberately separates data preparation from model evaluation:

* Validation 019--024 reuses the existing formal calibration-point artifact.
* Validation 037--040 and 055--060 are extracted with the unchanged formal
  full_board_physical/Steger/continuity protocol.
* Full-36 Quadratic and Circular Cone YAMLs are loaded into model objects, but
  no ``fit`` method, optimizer, parameter search, or C1 training is called.
* All Validation poses are retained.  Residuals are used only to compare the
  two frozen models, never to remove a pose or alter the extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng"
DATA_ROOT = PROJECT_ROOT / "data"
INTRINSICS_PATH = PROJECT_ROOT / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
FORMAL_CONFIG_PATH = CALIBRATION_TOOL_ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
REUSED_019_024_POINTS = PROJECT_ROOT / "outputs" / "0811" / "laser_model" / "calibration_points.csv"
FULL36_MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "0817"
    / "grouped_cv_model_comparison"
    / "candidate_models"
    / "full_fit"
)
FULL36_METADATA = FULL36_MODEL_DIR / "model_parameters.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "0818" / "final_validation_qc"

MODEL_KEYS = ("quadratic_graph", "circular_cone")
MODEL_LABELS = {"quadratic_graph": "Quadratic", "circular_cone": "Cone"}
V_MIN = 0.0
V_MAX = 3000.0
V_BIN_WIDTH = 100.0
V_BIN_COUNT = int((V_MAX - V_MIN) / V_BIN_WIDTH)

VALIDATION_BATCHES = {
    "019_024": {
        "ids": tuple(range(19, 25)),
        "root": DATA_ROOT / "laser_plane" / "validation",
        "manifest": DATA_ROOT / "laser_plane" / "frames.csv",
        "source": "REUSED_0811_CALIBRATION_POINTS",
    },
    "037_040": {
        "ids": tuple(range(37, 41)),
        "root": DATA_ROOT / "laser_plane" / "validation_edge_holdout" / "validation",
        "manifest": DATA_ROOT / "laser_plane" / "validation_edge_holdout" / "frames.csv",
        "source": "NEW_FORMAL_PROTOCOL_EXTRACTION",
    },
    "055_060": {
        "ids": tuple(range(55, 61)),
        "root": DATA_ROOT / "laser_plane_0817" / "validation",
        "manifest": DATA_ROOT / "laser_plane_0817" / "frames.csv",
        "source": "NEW_FORMAL_PROTOCOL_EXTRACTION",
    },
}

SCOPE_IDS = {
    "primary_055_060": tuple(range(55, 61)),
    "historical_019_024": tuple(range(19, 25)),
    "historical_037_040": tuple(range(37, 41)),
    "historical_regression": tuple(range(19, 25)) + tuple(range(37, 41)),
    "all_validation": tuple(range(19, 25)) + tuple(range(37, 41)) + tuple(range(55, 61)),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Full-36 Q/C independent Validation comparison")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def zero_pad(value: Any) -> str:
    return f"{int(str(value)):03d}"


def image_paths(root: Path, pose_id: str) -> dict[str, Path]:
    return {
        role: root / f"{role} {pose_id}.tif"
        for role in ("chess", "nolaser", "laser")
    }


def read_manifest(path: Path) -> dict[tuple[str, str], str]:
    manifest = pd.read_csv(path, dtype=str)
    required = {"pose_id", "role", "filename", "sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest {path} misses columns {sorted(missing)}")
    result: dict[tuple[str, str], str] = {}
    for row in manifest.to_dict("records"):
        result[(zero_pad(row["pose_id"]), str(row["role"]).strip())] = str(row["sha256"]).strip()
    return result


def verify_images(batch_name: str, batch: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(Path(batch["manifest"]))
    checked = 0
    mismatches: list[str] = []
    for pose_num in batch["ids"]:
        pose_id = f"{pose_num:03d}"
        paths = image_paths(Path(batch["root"]), pose_id)
        for role, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(path)
            expected = manifest.get((pose_id, role))
            if expected is None:
                raise RuntimeError(f"Manifest has no {batch_name} {pose_id} {role} row")
            actual = sha256_file(path)
            checked += 1
            if actual != expected:
                mismatches.append(f"{pose_id}/{role}: {actual} != {expected}")
    if mismatches:
        raise RuntimeError("Image manifest hash mismatch: " + "; ".join(mismatches))
    return {
        "batch": batch_name,
        "manifest": str(batch["manifest"]),
        "image_count": checked,
        "hash_status": "CONFIRMED",
    }


def protocol_config() -> tuple[dict[str, Any], np.ndarray, np.ndarray, tuple[int, int]]:
    cfg = load_yaml(FORMAL_CONFIG_PATH)
    k, dist, image_size = load_intrinsics(INTRINSICS_PATH)
    if image_size is None:
        raise RuntimeError("Intrinsics has no image size")
    extraction = cfg.get("extraction", {})
    expected = {
        "method": "steger",
        "sigma": 1.5,
        "min_intensity": 8.0,
        "min_response": 0.8,
        "max_subpixel_offset": 0.60,
        "continuity_poly_degree": 2,
        "continuity_threshold_px": 2.0,
        "board_mask_mode": "full_board_physical",
        "board_mask_inset_mm": 0.0,
        "max_points_per_image": 900,
    }
    for key, value in expected.items():
        actual = extraction.get(key)
        if isinstance(value, float):
            if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(f"Formal extraction protocol changed for {key}: {actual!r} != {value!r}")
        elif str(actual).lower() != str(value).lower():
            raise RuntimeError(f"Formal extraction protocol changed for {key}: {actual!r} != {value!r}")
    if str(cfg.get("laser", {}).get("orientation", "")).lower() != "vertical":
        raise RuntimeError("Formal laser orientation is not vertical")
    return cfg, k, dist, image_size


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    cfg = load_yaml(path)
    k = np.asarray(cfg["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(cfg.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
    size = None
    if "image_width" in cfg and "image_height" in cfg:
        size = (int(cfg["image_width"]), int(cfg["image_height"]))
    return k, dist, size


def load_triplets_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import fit_laser_models_from_triplets as triplets  # type: ignore

    return triplets


def normalize_reused_points(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path)
    required = {
        "split", "image_id", "frame_key", "u_px", "v_px", "response",
        "ray_x", "ray_y", "ray_z", "Xc_mm", "Yc_mm", "Zc_mm",
        "board_nx", "board_ny", "board_nz", "board_d_mm", "pnp_rmse_px",
    }
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"Reused calibration points miss columns {sorted(missing)}")
    points = points[points["split"].astype(str).eq("validation")].copy()
    points["pose_id"] = points["image_id"].map(zero_pad)
    expected = {f"{i:03d}" for i in range(19, 25)}
    actual = set(points["pose_id"])
    if actual != expected:
        raise RuntimeError(f"Reused 019-024 artifact has pose IDs {sorted(actual)}")
    counts = points.groupby("pose_id").size()
    if set(counts.tolist()) != {900}:
        raise RuntimeError(f"Reused 019-024 point counts are not 900/frame: {counts.to_dict()}")
    points["batch"] = "019_024"
    points["point_source"] = "REUSED_0811_CALIBRATION_POINTS"
    points["frame_id"] = points["pose_id"]
    return points.reset_index(drop=True)


def extract_missing_points(
    triplets: Any,
    batch_name: str,
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    k: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
) -> pd.DataFrame:
    board_cfg = dict(cfg["board"])
    extraction_cfg = dict(cfg["extraction"])
    orientation = str(cfg["laser"]["orientation"])
    rows: list[dict[str, Any]] = []
    for pose_num in batch["ids"]:
        pose_id = f"{pose_num:03d}"
        paths = image_paths(Path(batch["root"]), pose_id)
        chess = triplets.imread_unicode(paths["chess"])
        background = triplets.imread_unicode(paths["nolaser"])
        laser = triplets.imread_unicode(paths["laser"])
        h, w = triplets.to_gray_float(chess).shape
        if (w, h) != image_size:
            raise RuntimeError(f"{batch_name}/{pose_id} image size {(w, h)} != {image_size}")
        pose = triplets.detect_board_pose(
            chess,
            k,
            dist,
            cols=int(board_cfg["pattern_cols"]),
            rows=int(board_cfg["pattern_rows"]),
            square_size_mm=float(board_cfg["square_size_mm"]),
            max_rmse_px=float(board_cfg.get("max_pnp_rmse_px", 0.4)),
        )
        mask = triplets.board_mask_for_pose(
            (h, w), pose, k, dist, board_cfg, extraction_cfg
        )
        u, v, response, _ = triplets.extract_laser_centers(
            laser, background, mask, extraction_cfg, orientation
        )
        min_points = int(extraction_cfg.get("min_points_per_image", 80))
        if len(u) < min_points:
            raise RuntimeError(f"{batch_name}/{pose_id}: only {len(u)} extracted points")
        rays = triplets.pixels_to_rays(u, v, k, dist)
        points, valid = triplets.intersect_rays_with_plane(rays, pose.normal, pose.d)
        u, v, response, rays, points = (
            u[valid], v[valid], response[valid], rays[valid], points[valid]
        )
        if len(points) < min_points:
            raise RuntimeError(f"{batch_name}/{pose_id}: only {len(points)} valid PnP intersections")
        for index in range(len(points)):
            rows.append(
                {
                    "split": "validation",
                    "image_id": pose_id,
                    "frame_key": f"validation:{pose_id}",
                    "u_px": float(u[index]),
                    "v_px": float(v[index]),
                    "response": float(response[index]),
                    "ray_x": float(rays[index, 0]),
                    "ray_y": float(rays[index, 1]),
                    "ray_z": float(rays[index, 2]),
                    "Xc_mm": float(points[index, 0]),
                    "Yc_mm": float(points[index, 1]),
                    "Zc_mm": float(points[index, 2]),
                    "board_nx": float(pose.normal[0]),
                    "board_ny": float(pose.normal[1]),
                    "board_nz": float(pose.normal[2]),
                    "board_d_mm": float(pose.d),
                    "pnp_rmse_px": float(pose.reprojection_rmse_px),
                    "pose_id": pose_id,
                    "batch": batch_name,
                    "point_source": "NEW_FORMAL_PROTOCOL_EXTRACTION",
                    "frame_id": pose_id,
                }
            )
        print(f"[OK] {batch_name}/{pose_id}: {len(points)} points, PnP RMSE={pose.reprojection_rmse_px:.5f}px")
    result = pd.DataFrame.from_records(rows)
    counts = result.groupby("pose_id").size()
    if set(counts.index) != {f"{i:03d}" for i in batch["ids"]}:
        raise RuntimeError(f"Missing extracted pose in {batch_name}: {counts.to_dict()}")
    return result


def instantiate_frozen_models() -> tuple[dict[str, Any], Any]:
    triplets = load_triplets_module()
    metadata = json.loads(FULL36_METADATA.read_text(encoding="utf-8"))
    if metadata.get("source") != "FIT 001-018, 025-036, 049-054":
        raise RuntimeError(f"Unexpected Full-36 model source: {metadata.get('source')!r}")
    if metadata.get("frame_count") != 36 or metadata.get("point_count") != 32400:
        raise RuntimeError("Full-36 metadata does not describe 36 poses/32400 points")
    if metadata.get("mask_mode") != "full_board_physical" or float(metadata.get("mask_inset_mm", -1)) != 0.0:
        raise RuntimeError("Full-36 metadata mask is not formal full_board_physical/inset=0")

    plane_data = load_yaml(FULL36_MODEL_DIR / "global_plane.yaml")
    plane = triplets.PlaneModel()
    plane.normal = np.asarray(plane_data["normal"], dtype=np.float64)
    plane.d = float(plane_data["d_mm"])
    plane.z_range = tuple(float(v) for v in plane_data["z_valid_range_mm"])

    q_data = load_yaml(FULL36_MODEL_DIR / "quadratic_graph.yaml")
    quadratic = triplets.QuadraticGraphModel()
    axis_index = {"X": 0, "Y": 1, "Z": 2}
    quadratic.dep_axis = axis_index[str(q_data["dependent_axis"])]
    quadratic.ind_axes = tuple(axis_index[v] for v in q_data["independent_axes"])
    quadratic.center = np.asarray(q_data["normalization"]["independent_center_mm"], dtype=np.float64)
    quadratic.scale = np.asarray(q_data["normalization"]["independent_scale_mm"], dtype=np.float64)
    quadratic.beta = np.asarray(q_data["coefficients"], dtype=np.float64)
    quadratic.z_range = tuple(float(v) for v in q_data["z_valid_range_mm"])
    quadratic.plane_hint = plane

    c_data = load_yaml(FULL36_MODEL_DIR / "circular_cone.yaml")
    cone = triplets.CircularConeModel({})
    cone.axis = np.asarray(c_data["axis_unit_camera"], dtype=np.float64)
    cone.axis /= np.linalg.norm(cone.axis)
    cone.apex = np.asarray(c_data["apex_camera_mm"], dtype=np.float64)
    cone.alpha_deg = float(c_data["half_apex_angle_deg"])
    cone.fit_success = bool(c_data.get("fit_success", True))
    cone.cost = float(c_data.get("optimizer_cost", float("nan")))
    cone.z_range = tuple(float(v) for v in c_data["z_valid_range_mm"])
    models = {"quadratic_graph": quadratic, "circular_cone": cone}
    return models, plane


def add_predictions(points: pd.DataFrame, models: Mapping[str, Any], plane: Any) -> pd.DataFrame:
    result = points.copy()
    rays = result[["ray_x", "ray_y", "ray_z"]].to_numpy(dtype=np.float64)
    board_normals = result[["board_nx", "board_ny", "board_nz"]].to_numpy(dtype=np.float64)
    board_d = result["board_d_mm"].to_numpy(dtype=np.float64)
    lambda_hint = plane.intersect_rays(rays)
    for key, model in models.items():
        lam = model.intersect_rays(rays, lambda_hint=lambda_hint)
        prediction = rays * lam[:, None]
        error = np.sum(prediction * board_normals, axis=1) + board_d
        valid = np.isfinite(lam) & np.all(np.isfinite(prediction), axis=1)
        error[~valid] = np.nan
        result[f"lambda_pred_{key}_mm"] = lam
        result[f"board_error_{key}_mm"] = error
        result[f"valid_{key}"] = valid
    result["lambda_truth_mm"] = result["Zc_mm"].to_numpy(dtype=float) / np.maximum(
        result["ray_z"].to_numpy(dtype=float), np.finfo(float).eps
    )
    return result


def frame_weights(frame_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(frame_ids.astype(str), return_counts=True)
    lookup = dict(zip(ids, counts))
    return np.asarray([1.0 / lookup[str(value)] for value in frame_ids], dtype=np.float64)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values[min(index, len(values) - 1)])


def metric_dict(values: np.ndarray, frame_ids: np.ndarray, total_count: int | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    clean = values[finite]
    count = int(len(values) if total_count is None else total_count)
    if clean.size == 0:
        return {
            "point_count": count,
            "valid_count": 0,
            "valid_rate": 0.0 if count else float("nan"),
            "bias_mm": float("nan"),
            "mae_mm": float("nan"),
            "rmse_mm": float("nan"),
            "p95_abs_mm": float("nan"),
            "max_abs_mm": float("nan"),
        }
    weights = frame_weights(np.asarray(frame_ids)[finite])
    abs_values = np.abs(clean)
    denom = float(np.sum(weights))
    return {
        "point_count": count,
        "valid_count": int(clean.size),
        "valid_rate": float(clean.size / count) if count else float("nan"),
        "bias_mm": float(np.sum(weights * clean) / denom),
        "mae_mm": float(np.sum(weights * abs_values) / denom),
        "rmse_mm": float(np.sqrt(np.sum(weights * clean**2) / denom)),
        "p95_abs_mm": weighted_quantile(abs_values, weights, 0.95),
        "max_abs_mm": float(np.max(abs_values)),
    }


def bin_label(index: int) -> str:
    return f"v_{index * 100:04d}_{(index + 1) * 100:04d}"


def scope_frame_ids(scope: str, points: pd.DataFrame) -> np.ndarray:
    wanted = {f"{value:03d}" for value in SCOPE_IDS[scope]}
    return points["pose_id"].astype(str).isin(wanted).to_numpy()


def model_metric_row(scope: str, model_key: str, subset: pd.DataFrame, level: str, v_bin: str) -> dict[str, Any]:
    values = subset[f"board_error_{model_key}_mm"].to_numpy(dtype=float)
    metrics = metric_dict(values, subset["frame_id"].to_numpy(dtype=str), len(subset))
    if v_bin == "Global":
        v_lo, v_hi = V_MIN, V_MAX
    elif v_bin.startswith("v_"):
        _, lo, hi = v_bin.split("_")
        v_lo, v_hi = float(lo), float(hi)
    else:
        v_lo, v_hi = float("nan"), float("nan")
    return {
        "scope": scope,
        "level": level,
        "model": model_key,
        "model_label": MODEL_LABELS[model_key],
        "v_bin": v_bin,
        "v_bin_lo_px": v_lo,
        "v_bin_hi_px": v_hi,
        "metric_weighting": "frame_equal",
        "pose_count": int(subset["pose_id"].nunique()),
        **metrics,
    }


def make_v_bin_metrics(points: pd.DataFrame, scopes: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        scope_mask = scope_frame_ids(scope, points)
        scoped = points.loc[scope_mask]
        v = scoped["v_px"].to_numpy(dtype=float)
        bin_index = np.floor(v / V_BIN_WIDTH).astype(int)
        for index in range(V_BIN_COUNT):
            subset = scoped.loc[bin_index == index]
            for model_key in MODEL_KEYS:
                rows.append(model_metric_row(scope, model_key, subset, "v_bin", bin_label(index)))
        out_of_range = scoped.loc[(v < V_MIN) | (v >= V_MAX)]
        for model_key in MODEL_KEYS:
            rows.append(model_metric_row(scope, model_key, out_of_range, "v_out_of_range", "out_of_range"))
    return pd.DataFrame(rows)


def global_rows(points: pd.DataFrame, scopes: Sequence[str], v_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        mask = scope_frame_ids(scope, points)
        subset = points.loc[mask]
        for model_key in MODEL_KEYS:
            row = model_metric_row(scope, model_key, subset, "global", "Global")
            bins = v_metrics[
                (v_metrics["scope"] == scope)
                & (v_metrics["model"] == model_key)
                & (v_metrics["level"] == "v_bin")
            ]
            bins = bins[bins["point_count"] > 0].sort_values("rmse_mm", ascending=False)
            if bins.empty:
                row.update({"worst_v_bin": "", "worst_v_bin_rmse_mm": np.nan, "worst_v_bin_p95_abs_mm": np.nan, "worst_v_bin_max_abs_mm": np.nan, "worst_v_bin_point_count": 0})
            else:
                worst = bins.iloc[0]
                row.update({
                    "worst_v_bin": worst["v_bin"],
                    "worst_v_bin_rmse_mm": worst["rmse_mm"],
                    "worst_v_bin_p95_abs_mm": worst["p95_abs_mm"],
                    "worst_v_bin_max_abs_mm": worst["max_abs_mm"],
                    "worst_v_bin_point_count": int(worst["point_count"]),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def paired_metrics(subset: pd.DataFrame, scope: str, pose_id: str, row_type: str) -> dict[str, Any]:
    q = subset["board_error_quadratic_graph_mm"].to_numpy(dtype=float)
    c = subset["board_error_circular_cone_mm"].to_numpy(dtype=float)
    valid = np.isfinite(q) & np.isfinite(c)
    q = q[valid]
    c = c[valid]
    frame_ids = subset.loc[valid, "frame_id"].to_numpy(dtype=str)
    q_metric = metric_dict(q, frame_ids, len(q))
    c_metric = metric_dict(c, frame_ids, len(c))
    if len(q) == 0:
        better = float("nan")
        mean_abs_delta = float("nan")
        signed_delta_rmse = float("nan")
    else:
        abs_q = np.abs(q)
        abs_c = np.abs(c)
        better = float(np.mean(abs_q < abs_c))
        mean_abs_delta = float(np.mean(abs_q - abs_c))
        signed_delta_rmse = float(np.sqrt(np.mean((q - c) ** 2)))
    return {
        "row_type": row_type,
        "scope": scope,
        "pose_id": pose_id,
        "batch": str(subset["batch"].iloc[0]) if not subset.empty else "",
        "metric_weighting": "frame_equal",
        "pair_count": int(len(q)),
        "q_bias_mm": q_metric["bias_mm"],
        "q_rmse_mm": q_metric["rmse_mm"],
        "q_p95_abs_mm": q_metric["p95_abs_mm"],
        "q_max_abs_mm": q_metric["max_abs_mm"],
        "c_bias_mm": c_metric["bias_mm"],
        "c_rmse_mm": c_metric["rmse_mm"],
        "c_p95_abs_mm": c_metric["p95_abs_mm"],
        "c_max_abs_mm": c_metric["max_abs_mm"],
        "delta_bias_q_minus_c_mm": q_metric["bias_mm"] - c_metric["bias_mm"],
        "delta_rmse_q_minus_c_mm": q_metric["rmse_mm"] - c_metric["rmse_mm"],
        "delta_p95_q_minus_c_mm": q_metric["p95_abs_mm"] - c_metric["p95_abs_mm"],
        "delta_max_q_minus_c_mm": q_metric["max_abs_mm"] - c_metric["max_abs_mm"],
        "paired_abs_error_delta_mean_q_minus_c_mm": mean_abs_delta,
        "paired_signed_delta_rmse_mm": signed_delta_rmse,
        "q_abs_better_fraction": better,
        "rmse_winner": "Quadratic" if q_metric["rmse_mm"] < c_metric["rmse_mm"] else "Cone" if c_metric["rmse_mm"] < q_metric["rmse_mm"] else "TIE",
        "p95_winner": "Quadratic" if q_metric["p95_abs_mm"] < c_metric["p95_abs_mm"] else "Cone" if c_metric["p95_abs_mm"] < q_metric["p95_abs_mm"] else "TIE",
    }


def make_pose_comparison(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pose_id, subset in points.groupby("pose_id", sort=True):
        rows.append(paired_metrics(subset, "pose", str(pose_id), "pose"))
    for scope in SCOPE_IDS:
        subset = points.loc[scope_frame_ids(scope, points)]
        rows.append(paired_metrics(subset, scope, "ALL", "scope_paired"))
    return pd.DataFrame(rows)


def selection_status(primary: pd.DataFrame, pose_comparison: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    rows = primary.set_index("model")
    metric_pairs = (
        ("abs_bias", lambda row: abs(float(row["bias_mm"]))),
        ("global_rmse", lambda row: float(row["rmse_mm"])),
        ("global_p95", lambda row: float(row["p95_abs_mm"])),
        ("global_max", lambda row: float(row["max_abs_mm"])),
        ("worst_rmse", lambda row: float(row["worst_v_bin_rmse_mm"])),
        ("worst_p95", lambda row: float(row["worst_v_bin_p95_abs_mm"])),
    )
    votes = {"quadratic_graph": 0, "circular_cone": 0}
    metric_winners: dict[str, str] = {}
    for name, getter in metric_pairs:
        q_value = getter(rows.loc["quadratic_graph"])
        c_value = getter(rows.loc["circular_cone"])
        if q_value < c_value:
            votes["quadratic_graph"] += 1
            metric_winners[name] = "Quadratic"
        elif c_value < q_value:
            votes["circular_cone"] += 1
            metric_winners[name] = "Cone"
        else:
            metric_winners[name] = "TIE"
    pose_rows = pose_comparison[
        (pose_comparison["row_type"] == "pose")
        & pose_comparison["pose_id"].isin([f"{v:03d}" for v in SCOPE_IDS["primary_055_060"]])
    ]
    q_pose_wins = int((pose_rows["rmse_winner"] == "Quadratic").sum())
    c_pose_wins = int((pose_rows["rmse_winner"] == "Cone").sum())
    if votes["quadratic_graph"] >= 4 and q_pose_wins >= c_pose_wins:
        status = "QUADRATIC"
    elif votes["circular_cone"] >= 4 and c_pose_wins >= q_pose_wins:
        status = "CONE"
    else:
        status = "UNRESOLVED"
    return status, {
        "primary_metric_winners": metric_winners,
        "primary_metric_votes": votes,
        "primary_pose_rmse_wins": {"Quadratic": q_pose_wins, "Cone": c_pose_wins},
        "rule": "Primary 055-060 only: at least 4/6 lower-error metrics and no fewer primary pose RMSE wins; Historical/pooled rows are not inputs.",
    }


def save_plot(path: Path, points: pd.DataFrame, v_metrics: pd.DataFrame) -> None:
    scope_titles = [
        ("primary_055_060", "Primary 055–060"),
        ("historical_regression", "Historical 019–024 + 037–040"),
        ("all_validation", "All Validation"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=False)
    for axis, (scope, title) in zip(axes.flat[:3], scope_titles):
        rows = v_metrics[v_metrics["scope"] == scope]
        for model_key, color in (("quadratic_graph", "#1f77b4"), ("circular_cone", "#d95f02")):
            part = rows[
                (rows["model"] == model_key)
                & (rows["level"] == "v_bin")
                & (rows["point_count"] > 0)
            ].sort_values("v_bin_lo_px")
            x = (part["v_bin_lo_px"].to_numpy(float) + part["v_bin_hi_px"].to_numpy(float)) / 2.0
            axis.plot(x, part["bias_mm"], color=color, linewidth=1.8, label=f"{MODEL_LABELS[model_key]} bias")
            axis.plot(x, part["rmse_mm"], color=color, linewidth=1.0, linestyle="--", label=f"{MODEL_LABELS[model_key]} RMSE")
        axis.axhline(0.0, color="#666666", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("v / px")
        axis.set_ylabel("mm")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
    rows = v_metrics[(v_metrics["scope"] == "primary_055_060") & (v_metrics["level"] == "v_bin")]
    q = rows[(rows["model"] == "quadratic_graph") & (rows["point_count"] > 0)].sort_values("v_bin_lo_px")
    c = rows[(rows["model"] == "circular_cone") & (rows["point_count"] > 0)].sort_values("v_bin_lo_px")
    x = (q["v_bin_lo_px"].to_numpy(float) + q["v_bin_hi_px"].to_numpy(float)) / 2.0
    axis = axes.flat[3]
    axis.plot(x, q["rmse_mm"].to_numpy(float) - c["rmse_mm"].to_numpy(float), color="#7b3294", linewidth=2, label="ΔRMSE Q−C")
    axis.plot(x, q["p95_abs_mm"].to_numpy(float) - c["p95_abs_mm"].to_numpy(float), color="#008837", linewidth=2, label="ΔP95 Q−C")
    axis.axhline(0.0, color="#666666", linewidth=0.7)
    axis.set_title("Primary paired v-bin difference")
    axis.set_xlabel("v / px")
    axis.set_ylabel("Q − C / mm")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.suptitle("Frozen Full-36 Quadratic vs Circular Cone: independent Validation residuals")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def audit_rows(
    image_audits: Sequence[Mapping[str, Any]],
    points: pd.DataFrame,
    models: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "artifact": "Full-36 quadratic_graph.yaml",
            "path": str(FULL36_MODEL_DIR / "quadratic_graph.yaml"),
            "role": "frozen Full-36 Quadratic candidate",
            "scope": "FIT 001-018,025-036,049-054",
            "model": "quadratic_graph",
            "point_count": 32400,
            "pose_count": 36,
            "mask": "full_board_physical; inset=0 mm",
            "extraction": "FIT artifact; unchanged for evaluation",
            "weighting": "Full-36 frozen model source",
            "cv_protocol": "Full-36 FIT reference; no Validation refit",
            "action": "reused; instantiated only",
            "provenance_status": "CONFIRMED",
            "notes": f"sha256={sha256_file(FULL36_MODEL_DIR / 'quadratic_graph.yaml')}; fit() not called",
        },
        {
            "artifact": "Full-36 circular_cone.yaml",
            "path": str(FULL36_MODEL_DIR / "circular_cone.yaml"),
            "role": "frozen Full-36 Circular Cone candidate",
            "scope": "FIT 001-018,025-036,049-054",
            "model": "circular_cone",
            "point_count": 32400,
            "pose_count": 36,
            "mask": "full_board_physical; inset=0 mm",
            "extraction": "FIT artifact; unchanged for evaluation",
            "weighting": "Full-36 frozen model source",
            "cv_protocol": "Full-36 FIT reference; no Validation refit",
            "action": "reused; instantiated only",
            "provenance_status": "CONFIRMED",
            "notes": f"sha256={sha256_file(FULL36_MODEL_DIR / 'circular_cone.yaml')}; fit() not called",
        },
        {
            "artifact": "calibration_result.yaml",
            "path": str(INTRINSICS_PATH),
            "role": "camera intrinsics/distortion",
            "scope": "all Validation",
            "model": "Q/C",
            "point_count": "",
            "pose_count": 16,
            "mask": "n/a",
            "extraction": "undistortPoints input",
            "weighting": "n/a",
            "cv_protocol": "n/a",
            "action": "reused",
            "provenance_status": "CONFIRMED",
            "notes": f"sha256={sha256_file(INTRINSICS_PATH)}",
        },
        {
            "artifact": "laser_model_fit_config.daheng.yaml",
            "path": str(FORMAL_CONFIG_PATH),
            "role": "formal Validation extraction protocol",
            "scope": "037-040,055-060; provenance for 019-024 reuse",
            "model": "Q/C",
            "point_count": "",
            "pose_count": 16,
            "mask": "full_board_physical; inset=0 mm",
            "extraction": "Steger; sigma=1.5; continuity degree=2; threshold=2.0 px; max=900/frame",
            "weighting": "frame-balanced evaluation",
            "cv_protocol": "frozen prediction only",
            "action": "reused; protocol asserted",
            "provenance_status": "CONFIRMED",
            "notes": f"sha256={sha256_file(FORMAL_CONFIG_PATH)}",
        },
        {
            "artifact": "calibration_points.csv validation rows",
            "path": str(REUSED_019_024_POINTS),
            "role": "formal extracted rays/PnP truth for 019-024",
            "scope": "historical 019-024",
            "model": "Q/C prediction recomputed from reused points",
            "point_count": int(len(points[points["batch"] == "019_024"])),
            "pose_count": 6,
            "mask": "full_board_physical per 0811 stage/config provenance",
            "extraction": "reused; no re-extraction",
            "weighting": "frame-balanced evaluation",
            "cv_protocol": "same pointwise domain; frozen prediction",
            "action": "reused",
            "provenance_status": "CONFIRMED",
            "notes": "900 points/pose; image hashes checked against frames.csv",
        },
    ]
    for item in image_audits:
        batch = str(item["batch"])
        source = VALIDATION_BATCHES[batch]["source"]
        rows.append(
            {
                "artifact": f"{batch} triplet images + frames.csv",
                "path": str(VALIDATION_BATCHES[batch]["root"]),
                "role": "Validation triplet input",
                "scope": batch,
                "model": "Q/C",
                "point_count": int(points[points["batch"] == batch].shape[0]),
                "pose_count": len(VALIDATION_BATCHES[batch]["ids"]),
                "mask": "full_board_physical; inset=0 mm",
                "extraction": "reused formal protocol" if source.startswith("REUSED") else "new formal protocol extraction",
                "weighting": "frame-balanced evaluation",
                "cv_protocol": "frozen prediction only",
                "action": "reused images; extract only missing points" if source.startswith("NEW") else "reused points and images",
                "provenance_status": item["hash_status"],
                "notes": f"image_count={item['image_count']}; manifest={item['manifest']}",
            }
        )
    rows.append(
        {
            "artifact": "Validation 0817 C1 artifacts",
            "path": str(PROJECT_ROOT / "outputs" / "0817" / "c1_independent_validation"),
            "role": "historical reference only",
            "scope": "019-024,037-040",
            "model": "C1/Cone legacy",
            "point_count": "",
            "pose_count": 10,
            "mask": "not used as current Q/C source",
            "extraction": "not used for current metrics",
            "weighting": "not used",
            "cv_protocol": "not same frozen Full-36 Q/C comparison",
            "action": "excluded from model metrics",
            "provenance_status": "EXCLUDED",
            "notes": "read only as provenance context; no C1 training or residual-based pose selection",
        }
    )
    return pd.DataFrame(rows)


def render_report(
    points: pd.DataFrame,
    global_df: pd.DataFrame,
    v_metrics: pd.DataFrame,
    pose_comparison: pd.DataFrame,
    status: str,
    selection: Mapping[str, Any],
    audit: pd.DataFrame,
) -> str:
    primary = global_df[global_df["scope"] == "primary_055_060"].copy()
    historical = global_df[global_df["scope"] == "historical_regression"].copy()
    all_rows = global_df[global_df["scope"] == "all_validation"].copy()
    primary_ids = {f"{value:03d}" for value in SCOPE_IDS["primary_055_060"]}
    primary_count = int(points[points["pose_id"].isin(primary_ids)].shape[0])
    out_of_domain_count = int(((points["v_px"] < V_MIN) | (points["v_px"] >= V_MAX)).sum())
    lines = [
        "# Frozen Full-36 Quadratic / Circular Cone independent Validation",
        "",
        f"`FINAL_C0_STATUS = {status}`",
        "",
        "## 结论",
        "",
        "- 模型选择只使用 Primary Validation 055–060；Historical 与 All Validation 不会回写或改变 Primary 结论。",
        f"- Primary 选择结果：`FINAL_C0_STATUS = {status}`。判定规则：{selection['rule']}",
        f"- Primary 数据：{primary_count} points / 6 poses；全 Validation：{len(points)} points / 16 poses。",
        "- Q/C 参数均从冻结 Full-36 YAML 读取；没有根据 Validation 重新拟合、调参或训练 C1。",
        "",
        "## Primary 055–060",
        "",
        "| model | bias / mm | RMSE / mm | P95 / mm | Max / mm | worst v-bin | worst RMSE / mm | worst P95 / mm | pose RMSE wins |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    primary_pose = pose_comparison[
        (pose_comparison["row_type"] == "pose")
        & pose_comparison["pose_id"].isin([f"{i:03d}" for i in SCOPE_IDS["primary_055_060"]])
    ]
    for _, row in primary.iterrows():
        model = row["model"]
        label = MODEL_LABELS[model]
        wins = int((primary_pose["rmse_winner"] == label).sum())
        lines.append(
            f"| {label} | {float(row['bias_mm']):.6f} | {float(row['rmse_mm']):.6f} | {float(row['p95_abs_mm']):.6f} | "
            f"{float(row['max_abs_mm']):.6f} | {row['worst_v_bin']} | {float(row['worst_v_bin_rmse_mm']):.6f} | "
            f"{float(row['worst_v_bin_p95_abs_mm']):.6f} | {wins}/6 |"
        )
    primary_pair = pose_comparison[(pose_comparison["row_type"] == "scope_paired") & (pose_comparison["scope"] == "primary_055_060")].iloc[0]
    lines += [
        "",
        f"Primary same-point paired RMSE delta (Q−C) = {float(primary_pair['delta_rmse_q_minus_c_mm']):.6f} mm; P95 delta = {float(primary_pair['delta_p95_q_minus_c_mm']):.6f} mm; Q absolute-error-better fraction = {float(primary_pair['q_abs_better_fraction']):.3f}.",
        f"Primary metric winners: {json.dumps(selection['primary_metric_winners'], ensure_ascii=False)}; metric votes: {json.dumps(selection['primary_metric_votes'], ensure_ascii=False)}.",
        "",
        "## Historical regression / pooled consistency",
        "",
        "| scope | model | bias / mm | RMSE / mm | P95 / mm | Max / mm | worst v-bin | worst RMSE / mm |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for scope_name, scope_title in (("historical_019_024", "019–024"), ("historical_037_040", "037–040"), ("historical_regression", "Historical pooled"), ("all_validation", "All Validation")):
        rows = global_df[global_df["scope"] == scope_name]
        for _, row in rows.iterrows():
            lines.append(
                f"| {scope_title} | {MODEL_LABELS[row['model']]} | {float(row['bias_mm']):.6f} | {float(row['rmse_mm']):.6f} | "
                f"{float(row['p95_abs_mm']):.6f} | {float(row['max_abs_mm']):.6f} | {row['worst_v_bin']} | {float(row['worst_v_bin_rmse_mm']):.6f} |"
            )
    lines += [
        "",
        "Historical/pooled 数值只用于检查一致性，不参与 `FINAL_C0_STATUS` 的决策。",
        "",
        "## Pose-level paired comparison",
        "",
        f"Primary pose RMSE wins：Quadratic={selection['primary_pose_rmse_wins']['Quadratic']}, Cone={selection['primary_pose_rmse_wins']['Cone']}。完整逐 pose 同点比较见 `validation_pose_comparison.csv`。",
        "",
        "## v-bin 与 edge coverage",
        "",
        "- v-bin 固定为 100 px，范围 [0,3000)。每个 bin 的 Bias/RMSE/P95/Max 与有效率见 `validation_v_bin_metrics.csv`。",
        f"- 严格 v 域外点数：{out_of_domain_count}；它们保留在 global/pose 指标，并在 v-bin 表中单列为 `out_of_range`，没有静默丢弃。",
        "- 图中分别显示 Primary、Historical pooled、All Validation 的有符号 bias 与 RMSE；第四 panel 显示 Primary 的 Q−C RMSE/P95。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- Intrinsics：`{INTRINSICS_PATH}`，SHA-256 `{sha256_file(INTRINSICS_PATH)}`。",
        f"- Formal extraction config：`{FORMAL_CONFIG_PATH}`，SHA-256 `{sha256_file(FORMAL_CONFIG_PATH)}`。",
        f"- Full-36 source metadata：`{FULL36_METADATA}`，source=FIT 001-018,025-036,049-054；mask=full_board_physical；36 poses/32400 points。",
        "- 019–024：复用 0811 正式 calibration_points.csv validation rows；037–040、055–060：仅补取缺失 points，三联图和配置不修改。",
        "- 每批三联图均与对应 frames.csv 做 SHA-256 校验；PnP truth 为同 pose chessboard solvePnP 平面与 camera ray 的交点。",
        "",
        audit.to_markdown(index=False),
        "",
        "## Constraints",
        "",
        "- 不修改 FIT、原始 Validation 图像、mask、sampling、weighting 或正式配置。",
        "- 不重新拟合 Quadratic/Circular Cone；只加载 frozen YAML 并计算 intersection/prediction。",
        "- 不训练 C1；不以 residual 删除 pose；所有 16 个 Validation pose 均保留。",
        "- 0817 C1 artifact 仅作历史 provenance context，不作为当前 Q/C 指标来源。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        existing = [p for p in output_dir.iterdir() if p.is_file()]
        if existing:
            raise FileExistsError(f"Outputs already exist; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, k, dist, image_size = protocol_config()
    image_audits = [verify_images(name, batch) for name, batch in VALIDATION_BATCHES.items()]
    reused = normalize_reused_points(REUSED_019_024_POINTS)
    triplets = load_triplets_module()
    generated = pd.concat(
        [
            extract_missing_points(triplets, "037_040", VALIDATION_BATCHES["037_040"], cfg, k, dist, image_size),
            extract_missing_points(triplets, "055_060", VALIDATION_BATCHES["055_060"], cfg, k, dist, image_size),
        ],
        ignore_index=True,
    )
    points = pd.concat([reused, generated], ignore_index=True)
    expected_ids = {f"{i:03d}" for ids in SCOPE_IDS.values() for i in ids}
    if set(points["pose_id"]) != expected_ids:
        raise RuntimeError("Validation point table does not cover exactly the requested 16 poses")
    models, plane = instantiate_frozen_models()
    points = add_predictions(points, models, plane)
    for key in MODEL_KEYS:
        invalid = int((~points[f"valid_{key}"]).sum())
        if invalid:
            raise RuntimeError(f"Frozen {key} has {invalid} invalid intersections")

    scopes = ("primary_055_060", "historical_019_024", "historical_037_040", "historical_regression", "all_validation")
    v_metrics = make_v_bin_metrics(points, scopes)
    global_df = global_rows(points, scopes, v_metrics)
    pose_comparison = make_pose_comparison(points)
    primary_df = global_df[global_df["scope"] == "primary_055_060"]
    status, selection = selection_status(primary_df, pose_comparison)
    audit = audit_rows(image_audits, points, models)

    audit.to_csv(output_dir / "validation_artifact_reuse_audit.csv", index=False, encoding="utf-8-sig")
    global_df[global_df["scope"] == "primary_055_060"].to_csv(output_dir / "primary_055_060_model_comparison.csv", index=False, encoding="utf-8-sig")
    global_df[global_df["scope"].isin(["historical_019_024", "historical_037_040", "historical_regression"])].to_csv(output_dir / "historical_validation_comparison.csv", index=False, encoding="utf-8-sig")
    global_df[global_df["scope"] == "all_validation"].to_csv(output_dir / "all_validation_comparison.csv", index=False, encoding="utf-8-sig")
    pose_comparison.to_csv(output_dir / "validation_pose_comparison.csv", index=False, encoding="utf-8-sig")
    v_metrics.to_csv(output_dir / "validation_v_bin_metrics.csv", index=False, encoding="utf-8-sig")
    points.to_csv(output_dir / "validation_points_used.csv", index=False, encoding="utf-8-sig")
    save_plot(output_dir / "validation_residual_vs_v.png", points, v_metrics)
    (output_dir / "report.md").write_text(
        render_report(points, global_df, v_metrics, pose_comparison, status, selection, audit),
        encoding="utf-8",
    )

    print(f"FINAL_C0_STATUS = {status}")
    print(global_df[global_df["scope"].isin(["primary_055_060", "historical_regression", "all_validation"])]
          [["scope", "model", "bias_mm", "rmse_mm", "p95_abs_mm", "max_abs_mm", "worst_v_bin", "worst_v_bin_rmse_mm"]]
          .to_string(index=False))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
