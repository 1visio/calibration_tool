#!/usr/bin/env python3
"""Freeze the FIT-derived C1_4k correction and evaluate it once on Validation.

The freeze-only mode never opens Validation images.  The evaluation mode reads
the explicitly declared Validation triplets once, after the frozen model JSON
already exists, and does not refit or tune any parameter.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
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
from scipy.interpolate import BSpline


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import validate_c1_grouped_cv as grouped  # noqa: E402


FIT_IDS = [f"{value:03d}" for value in list(range(1, 19)) + list(range(25, 37))]
VALIDATION_IDS = [f"{value:03d}" for value in list(range(19, 25)) + list(range(37, 41))]
KNOT_COUNT = 4
DEGREE = 3
PENALTY = 0.10
IMAGE_HEIGHT = 3000
REGIONS = (
    ("global", 0.0, 3000.0),
    ("top", 0.0, 300.0),
    ("middle", 300.0, 2700.0),
    ("bottom", 2700.0, 3000.0),
)

# These gates are fixed before Validation is opened.  They are not selected
# from Validation results.
MIN_GLOBAL_RMSE_IMPROVEMENT_PCT = 5.0
MIN_EDGE_FRAME_IMPROVEMENT_FRACTION = 0.50
MAX_MIDDLE_DEGRADATION_PCT = 2.0
MAX_WORST_DEGRADATION_PCT = 2.0
MAX_EDGE_MIDDLE_DEGRADATION_PCT = 2.0
MAX_BIAS_RANGE_DEGRADATION_PCT = 5.0

DEFAULT_POINTS = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "spatial_residual_observability"
    / "fit_ray_residual_points.csv"
)
DEFAULT_PCA = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "spatial_residual_observability"
    / "ray_support_summary.json"
)
DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_EDGE_DATA_ROOT = DEFAULT_DATA_ROOT / "validation_edge_holdout"
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
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0817" / "c1_independent_validation"
)
OUTPUT_NAMES = (
    "frozen_c1_model.json",
    "validation_metrics.csv",
    "validation_per_frame.csv",
    "report.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--pca-summary", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--edge-data-root", type=Path, default=DEFAULT_EDGE_DATA_ROOT)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--evaluate-validation", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--render-report-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_json(value: Any) -> Any:
    if is_dataclass(value):
        return clean_json(asdict(value))
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
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return clean_json(vars(value))
    return value


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def safe_improvement(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after) or abs(before) <= np.finfo(float).eps:
        return math.nan
    return float((before - after) / abs(before) * 100.0)


def safe_nanmax(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmax(array)) if np.any(np.isfinite(array)) else math.nan


def safe_nanmin(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmin(array)) if np.any(np.isfinite(array)) else math.nan


def safe_nanmean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else math.nan


def edge_middle_ratio(top: float, bottom: float, middle: float) -> float:
    if not math.isfinite(middle) or middle <= 0.0:
        return math.nan
    edge = [value for value in (top, bottom) if math.isfinite(value)]
    return float(np.mean(edge) / middle) if edge else math.nan


def metric(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"point_count": 0, "bias_mm": math.nan, "rmse_mm": math.nan, "p95_abs_mm": math.nan}
    return {
        "point_count": int(len(values)),
        "bias_mm": float(np.mean(values)),
        "rmse_mm": float(np.sqrt(np.mean(values * values))),
        "p95_abs_mm": float(np.percentile(np.abs(values), 95)),
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    if not len(sorted_values) or cumulative[-1] <= 0.0:
        return math.nan
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1])), len(sorted_values) - 1)
    return float(sorted_values[index])


def weighted_metric(values: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[keep]
    weights = weights[keep]
    if len(values) == 0 or float(np.sum(weights)) <= 0.0:
        return {"point_count": 0, "bias_mm": math.nan, "rmse_mm": math.nan, "p95_abs_mm": math.nan}
    total = float(np.sum(weights))
    return {
        "point_count": int(len(values)),
        "bias_mm": float(np.sum(weights * values) / total),
        "rmse_mm": float(np.sqrt(np.sum(weights * values * values) / total)),
        "p95_abs_mm": weighted_quantile(np.abs(values), weights, 0.95),
    }


def frame_equal_weights(frame_ids: np.ndarray) -> np.ndarray:
    frame_ids = np.asarray(frame_ids)
    weights = np.empty(len(frame_ids), dtype=np.float64)
    for frame_id in np.unique(frame_ids):
        mask = frame_ids == frame_id
        weights[mask] = 1.0 / float(np.count_nonzero(mask))
    return weights


def load_previous_pca(path: Path, s_values: np.ndarray) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("validation_opened") is not False:
        raise RuntimeError("Previous PCA summary does not prove Validation was unopened")
    if summary.get("fit_ids") != FIT_IDS or summary.get("frame_027_retained") is not True:
        raise RuntimeError("Previous PCA summary does not match the frozen FIT registry")
    pca = summary.get("pca")
    required = ("center_xn", "center_yn", "axis_s_xn", "axis_s_yn", "axis_t_xn", "axis_t_yn")
    if not isinstance(pca, Mapping) or any(key not in pca for key in required):
        raise RuntimeError("Previous PCA summary is missing the frozen s/t definition")
    domain = [float(np.min(s_values)), float(np.max(s_values))]
    return {
        "source_summary_sha256": sha256_file(path),
        "center_xn": float(pca["center_xn"]),
        "center_yn": float(pca["center_yn"]),
        "axis_s_xn": float(pca["axis_s_xn"]),
        "axis_s_yn": float(pca["axis_s_yn"]),
        "axis_t_xn": float(pca["axis_t_xn"]),
        "axis_t_yn": float(pca["axis_t_yn"]),
        "domain_min": domain[0],
        "domain_max": domain[1],
        "coordinate_definition": "(undistorted_xn,undistorted_yn) centered by frozen PCA center and projected onto frozen axis_s/axis_t",
    }


def runtime_file_hashes(app: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("intrinsics", "extrinsics", "ground_u_compensation"):
        value = getattr(app.calibration, name, None)
        if value is None:
            result[name] = None
        else:
            path = Path(value).resolve()
            result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def freeze_model(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    output_dir = args.output_dir.resolve()
    model_path = output_dir / "frozen_c1_model.json"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"Frozen model already exists; refusing to overwrite: {model_path}")

    fit_points = grouped.load_points(args.points.resolve())
    frame_ids = np.asarray([row["frame_id"] for row in fit_points])
    s_values = np.asarray([float(row["pca_s"]) for row in fit_points], dtype=np.float64)
    targets = np.asarray([float(row["delta_lambda_mm"]) for row in fit_points], dtype=np.float64)
    pca = load_previous_pca(args.pca_summary.resolve(), s_values)
    domain = (pca["domain_min"], pca["domain_max"])
    fitted = grouped.fit_spline(
        s_values,
        targets,
        frame_ids,
        domain,
        KNOT_COUNT,
        PENALTY,
    )
    app, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_cone, cone_info = board.load_frozen_model_checked(
        args.frozen_provenance.resolve(), args.formal_cone.resolve()
    )
    if calibration["laser_model"].get("model_type") != "circular_cone":
        raise RuntimeError("Runtime calibration is not Circular Cone")

    fit_residual = targets - np.asarray(fitted["spline"](s_values), dtype=np.float64)
    frame_counts = {frame_id: int(np.count_nonzero(frame_ids == frame_id)) for frame_id in FIT_IDS}
    core: dict[str, Any] = {
        "schema_version": 1,
        "model_id": "C1_4k",
        "frozen": True,
        "definition": {
            "prediction": "lambda_cone + F(s)",
            "target": "lambda_truth - lambda_cone",
            "coordinate": "frozen PCA pca_s",
            "degree": DEGREE,
            "interior_knot_count": KNOT_COUNT,
            "basis_count": int(fitted["basis_count"]),
            "smoothness": "moderate_second_difference",
            "second_difference_penalty": PENALTY,
            "training_weighting": "frame_equal_total_weight_1_per_fit_frame",
            "extrapolation": "BSpline extrapolate=True outside frozen FIT s domain",
        },
        "fit": {
            "fit_ids": FIT_IDS,
            "point_count": len(fit_points),
            "point_count_by_frame": frame_counts,
            "training_target_bias_mm": float(np.mean(targets)),
            "training_target_rmse_mm": float(np.sqrt(np.mean(targets * targets))),
            "training_after_f_mm": float(np.sqrt(np.mean(fit_residual * fit_residual))),
        },
        "pca_s": pca,
        "spline": {
            "degree": DEGREE,
            "interior_knot_count": KNOT_COUNT,
            "basis_count": int(fitted["basis_count"]),
            "penalty": PENALTY,
            "knots": np.asarray(fitted["knots"], dtype=np.float64).tolist(),
            "coefficients_mm": np.asarray(fitted["coefficients"], dtype=np.float64).tolist(),
        },
        "frozen_cone": {
            "runtime_model": clean_json(frozen_cone),
            "provenance": {
                "path": str(args.frozen_provenance.resolve()),
                "sha256": sha256_file(args.frozen_provenance.resolve()),
            },
            "formal_cone": {
                "path": str(args.formal_cone.resolve()),
                "sha256": sha256_file(args.formal_cone.resolve()),
            },
            "loader_info": clean_json(cone_info),
        },
        "frozen_runtime": {
            "measurement_config": {
                "path": str(args.measurement_config.resolve()),
                "sha256": sha256_file(args.measurement_config.resolve()),
            },
            "calibration_files": runtime_file_hashes(app),
            "camera_matrix": np.asarray(intrinsics.camera_matrix, dtype=np.float64).tolist(),
            "dist_coeffs": np.asarray(intrinsics.dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
            "reconstruction": clean_json(reconstruction_params),
        },
        "input_points": {
            "path": str(args.points.resolve()),
            "sha256": sha256_file(args.points.resolve()),
        },
        "validation_policy": {
            "validation_read_passes": 1,
            "validation_ids": VALIDATION_IDS,
            "min_global_rmse_improvement_pct": MIN_GLOBAL_RMSE_IMPROVEMENT_PCT,
            "min_edge_frame_improvement_fraction": MIN_EDGE_FRAME_IMPROVEMENT_FRACTION,
            "max_middle_degradation_pct": MAX_MIDDLE_DEGRADATION_PCT,
            "max_worst_degradation_pct": MAX_WORST_DEGRADATION_PCT,
            "max_edge_middle_degradation_pct": MAX_EDGE_MIDDLE_DEGRADATION_PCT,
            "max_bias_range_degradation_pct": MAX_BIAS_RANGE_DEGRADATION_PCT,
        },
    }
    model = dict(core)
    model["parameter_sha256"] = canonical_sha256(core)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(clean_json(model), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact_sha = sha256_file(model_path)
    return model_path, artifact_sha, model


def load_frozen_json(path: Path) -> tuple[dict[str, Any], str]:
    model = json.loads(path.read_text(encoding="utf-8"))
    expected = str(model.get("parameter_sha256", ""))
    core = {key: value for key, value in model.items() if key != "parameter_sha256"}
    if not expected or canonical_sha256(core) != expected:
        raise RuntimeError("frozen_c1_model.json parameter_sha256 mismatch")
    if model.get("model_id") != "C1_4k" or model.get("frozen") is not True:
        raise RuntimeError("Frozen JSON is not the declared C1_4k model")
    if model.get("definition", {}).get("interior_knot_count") != KNOT_COUNT:
        raise RuntimeError("Frozen JSON knot count is not 4")
    return model, sha256_file(path)


def frozen_spline(model: Mapping[str, Any]) -> BSpline:
    spline = model["spline"]
    return BSpline(
        np.asarray(spline["knots"], dtype=np.float64),
        np.asarray(spline["coefficients_mm"], dtype=np.float64),
        int(spline["degree"]),
        extrapolate=True,
    )


def validation_inventory(data_root: Path, edge_root: Path) -> list[dict[str, Any]]:
    specs = ((data_root.resolve(), VALIDATION_IDS[:6]), (edge_root.resolve(), VALIDATION_IDS[6:]))
    inventory: dict[str, dict[str, Any]] = {}
    for root, expected_ids in specs:
        rows = csv_rows(root / "frames.csv")
        for row in rows:
            frame_id = f"{int(row['pose_id']):03d}"
            if frame_id not in expected_ids or row.get("role") not in {"chess", "nolaser", "laser"}:
                continue
            relative = Path(str(row["filename"]).replace("/", "\\"))
            if relative.parts and relative.parts[0].lower() in {"fit", "validation"}:
                relative = Path("validation", *relative.parts[1:])
            inventory.setdefault(frame_id, {})[str(row["role"])] = {
                "path": root / relative,
                "manifest_sha256": str(row.get("sha256", "")),
                "dataset_root": str(root),
            }
    if sorted(inventory, key=int) != VALIDATION_IDS or any(set(inventory[f]) != {"chess", "nolaser", "laser"} for f in VALIDATION_IDS):
        raise RuntimeError(f"Validation inventory mismatch: {sorted(inventory, key=int)}")
    return [
        {"frame_id": frame_id, **inventory[frame_id]}
        for frame_id in VALIDATION_IDS
    ]


def read_image_once(path: Path, expected_sha256: str) -> tuple[np.ndarray, str, bool]:
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise RuntimeError(f"Validation image SHA-256 mismatch: {path}")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise RuntimeError(f"Validation image is not a grayscale image: {path}")
    return image, observed_sha256, bool(not expected_sha256 or observed_sha256 == expected_sha256)


def frozen_pca_s(uv: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    runtime = model["frozen_runtime"]
    camera_matrix = np.asarray(runtime["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(runtime["dist_coeffs"], dtype=np.float64)
    normalized = cv2.undistortPoints(np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2), camera_matrix, dist_coeffs).reshape(-1, 2)
    pca = model["pca_s"]
    center = np.asarray([pca["center_xn"], pca["center_yn"]], dtype=np.float64)
    axis_s = np.asarray([pca["axis_s_xn"], pca["axis_s_yn"]], dtype=np.float64)
    return (normalized - center) @ axis_s


def process_validation_frame(
    item: Mapping[str, Any],
    model: Mapping[str, Any],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    intrinsics: Any,
    spline: BSpline,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    images: dict[str, np.ndarray] = {}
    image_hashes: dict[str, str] = {}
    image_hash_ok: dict[str, bool] = {}
    for role in ("chess", "nolaser", "laser"):
        image, observed, hash_ok = read_image_once(Path(item[role]["path"]), item[role]["manifest_sha256"])
        images[role] = image
        image_hashes[role] = observed
        image_hash_ok[role] = hash_ok

    pose = fixed.triplets.detect_board_pose(
        images["chess"],
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        cols=11,
        rows=8,
        square_size_mm=20.0,
        max_rmse_px=0.40,
    )
    board_mask = fixed.triplets.board_inner_mask(
        fixed.triplets.to_gray_float(images["chess"]).shape,
        pose.corners,
        margin_px=-2,
    )
    u, v, _, _ = fixed.triplets.extract_laser_centers(
        images["laser"], images["nolaser"], board_mask, fixed.EXTRACTION_CONFIG, "vertical"
    )
    uv = np.column_stack([u, v]).astype(np.float64)
    truth = fixed.coverage.plane_ray_truth(
        u, v, pose.normal, pose.d, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    candidate = copy.deepcopy(dict(calibration))
    cone_model = copy.deepcopy(model["frozen_cone"]["runtime_model"])
    cone_model["axis_unit_camera"] = np.asarray(cone_model["axis_unit_camera"], dtype=np.float64)
    cone_model["apex_camera_mm"] = np.asarray(cone_model["apex_camera_mm"], dtype=np.float64)
    candidate["laser_model"] = cone_model
    lambda_cone, cone_valid = fixed.lambda_by_input(uv, candidate, reconstruction_params)
    truth_lambda = np.asarray(truth["points"][:, 2], dtype=np.float64)
    valid = cone_valid & np.asarray(truth["valid"], dtype=bool) & np.isfinite(truth_lambda)
    if np.count_nonzero(valid) < 30:
        raise RuntimeError(f"Validation frame {item['frame_id']} has too few valid rays")
    uv_valid = uv[valid]
    v_valid = uv_valid[:, 1]
    s = frozen_pca_s(uv_valid, model)
    c0_residual = truth_lambda[valid] - lambda_cone[valid]
    c1_residual = truth_lambda[valid] - (lambda_cone[valid] + np.asarray(spline(s), dtype=np.float64))
    frame = {
        "frame_id": str(item["frame_id"]),
        "source_dataset": str(item["chess"]["dataset_root"]),
        "pose_reprojection_rmse_px": float(pose.reprojection_rmse_px),
        "valid_point_count": int(np.count_nonzero(valid)),
        "invalid_point_count": int(len(uv) - np.count_nonzero(valid)),
        "extracted_point_count": int(len(uv)),
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "s_extrapolated_count": int(np.count_nonzero((s < model["pca_s"]["domain_min"]) | (s > model["pca_s"]["domain_max"]))),
        "image_hash_ok": bool(all(image_hash_ok.values())),
        "image_sha256": image_hashes,
    }
    arrays = {
        "v": v_valid,
        "frame_id": np.full(len(v_valid), str(item["frame_id"]), dtype="U3"),
        "c0": c0_residual,
        "c1": c1_residual,
    }
    return frame, arrays


def add_region_metrics(row: dict[str, Any], prefix: str, c0_values: np.ndarray, c1_values: np.ndarray) -> None:
    c0 = metric(c0_values)
    c1 = metric(c1_values)
    row[f"{prefix}_point_count"] = c0["point_count"]
    row[f"{prefix}_c0_bias_mm"] = c0["bias_mm"]
    row[f"{prefix}_c1_bias_mm"] = c1["bias_mm"]
    row[f"{prefix}_c0_rmse_mm"] = c0["rmse_mm"]
    row[f"{prefix}_c1_rmse_mm"] = c1["rmse_mm"]
    row[f"{prefix}_rmse_improvement_pct"] = safe_improvement(c0["rmse_mm"], c1["rmse_mm"])
    row[f"{prefix}_c0_p95_abs_mm"] = c0["p95_abs_mm"]
    row[f"{prefix}_c1_p95_abs_mm"] = c1["p95_abs_mm"]
    row[f"{prefix}_p95_improvement_pct"] = safe_improvement(c0["p95_abs_mm"], c1["p95_abs_mm"])


def make_per_frame_row(frame: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    v = np.asarray(arrays["v"], dtype=np.float64)
    c0 = np.asarray(arrays["c0"], dtype=np.float64)
    c1 = np.asarray(arrays["c1"], dtype=np.float64)
    row: dict[str, Any] = dict(frame)
    row.pop("image_sha256", None)
    for region, low, high in REGIONS:
        mask = np.ones(len(v), dtype=bool) if region == "global" else (v >= low) & (v < high)
        add_region_metrics(row, region, c0[mask], c1[mask])
    for model_prefix, values in (("c0", c0), ("c1", c1)):
        region_values = {region: row[f"{region}_{model_prefix}_rmse_mm"] for region, _, _ in REGIONS[1:]}
        row[f"{model_prefix}_edge_middle_ratio"] = edge_middle_ratio(
            region_values["top"], region_values["bottom"], region_values["middle"]
        )
        row[f"{model_prefix}_worst_region_rmse_mm"] = safe_nanmax(list(region_values.values()))
        p95_values = {region: row[f"{region}_{model_prefix}_p95_abs_mm"] for region, _, _ in REGIONS[1:]}
        row[f"{model_prefix}_worst_region_p95_abs_mm"] = safe_nanmax(list(p95_values.values()))
        bias_values = {region: row[f"{region}_{model_prefix}_bias_mm"] for region, _, _ in REGIONS[1:]}
        row[f"{model_prefix}_region_bias_range_mm"] = (
            safe_nanmax(list(bias_values.values())) - safe_nanmin(list(bias_values.values()))
            if np.any(np.isfinite(list(bias_values.values())))
            else math.nan
        )
    row["edge_middle_ratio_improvement_pct"] = safe_improvement(row["c0_edge_middle_ratio"], row["c1_edge_middle_ratio"])
    row["worst_region_rmse_improvement_pct"] = safe_improvement(row["c0_worst_region_rmse_mm"], row["c1_worst_region_rmse_mm"])
    row["worst_region_p95_improvement_pct"] = safe_improvement(row["c0_worst_region_p95_abs_mm"], row["c1_worst_region_p95_abs_mm"])
    row["region_bias_range_improvement_pct"] = safe_improvement(row["c0_region_bias_range_mm"], row["c1_region_bias_range_mm"])
    return row


def per_frame_values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray([finite(row.get(field)) for row in rows], dtype=np.float64)


def aggregate_region(
    region: str,
    rows: Sequence[Mapping[str, Any]],
    all_frame_ids: np.ndarray,
    all_v: np.ndarray,
    all_c0: np.ndarray,
    all_c1: np.ndarray,
) -> dict[str, Any]:
    if region == "worst_region":
        c0_rmse = per_frame_values(rows, "c0_worst_region_rmse_mm")
        c1_rmse = per_frame_values(rows, "c1_worst_region_rmse_mm")
        c0_p95 = per_frame_values(rows, "c0_worst_region_p95_abs_mm")
        c1_p95 = per_frame_values(rows, "c1_worst_region_p95_abs_mm")
        c0_bias_stack = np.vstack([per_frame_values(rows, f"{region_name}_c0_bias_mm") for region_name, _, _ in REGIONS[1:]])
        c1_bias_stack = np.vstack([per_frame_values(rows, f"{region_name}_c1_bias_mm") for region_name, _, _ in REGIONS[1:]])
        c0_bias = np.full(len(rows), np.nan, dtype=np.float64)
        c1_bias = np.full(len(rows), np.nan, dtype=np.float64)
        c0_frame_rmse = c0_rmse
        c1_frame_rmse = c1_rmse
        c0_frame_p95 = c0_p95
        c1_frame_p95 = c1_p95
        point_count = None
        frame_count = int(np.count_nonzero(np.isfinite(c0_rmse) & np.isfinite(c1_rmse)))
        pooled_c0 = {
            "bias_mm": math.nan,
            "rmse_mm": float(np.nanmedian(c0_rmse)) if np.any(np.isfinite(c0_rmse)) else math.nan,
            "p95_abs_mm": float(np.nanmedian(c0_p95)) if np.any(np.isfinite(c0_p95)) else math.nan,
        }
        pooled_c1 = {
            "bias_mm": math.nan,
            "rmse_mm": float(np.nanmedian(c1_rmse)) if np.any(np.isfinite(c1_rmse)) else math.nan,
            "p95_abs_mm": float(np.nanmedian(c1_p95)) if np.any(np.isfinite(c1_p95)) else math.nan,
        }
    else:
        c0_bias_stack = None
        c1_bias_stack = None
        low_high = {name: (low, high) for name, low, high in REGIONS}[region]
        mask = np.ones(len(all_v), dtype=bool) if region == "global" else ((all_v >= low_high[0]) & (all_v < low_high[1]))
        weights = frame_equal_weights(all_frame_ids[mask])
        pooled_c0 = weighted_metric(all_c0[mask], weights)
        pooled_c1 = weighted_metric(all_c1[mask], weights)
        c0_frame_rmse = per_frame_values(rows, f"{region}_c0_rmse_mm")
        c1_frame_rmse = per_frame_values(rows, f"{region}_c1_rmse_mm")
        c0_frame_p95 = per_frame_values(rows, f"{region}_c0_p95_abs_mm")
        c1_frame_p95 = per_frame_values(rows, f"{region}_c1_p95_abs_mm")
        c0_bias = per_frame_values(rows, f"{region}_c0_bias_mm")
        c1_bias = per_frame_values(rows, f"{region}_c1_bias_mm")
        point_count = int(np.count_nonzero(mask))
        frame_count = int(np.unique(all_frame_ids[mask]).size)

    if region == "worst_region":
        c0_bias_range = float(np.nanmax(c0_bias_stack) - np.nanmin(c0_bias_stack)) if c0_bias_stack is not None and np.any(np.isfinite(c0_bias_stack)) else math.nan
        c1_bias_range = float(np.nanmax(c1_bias_stack) - np.nanmin(c1_bias_stack)) if c1_bias_stack is not None and np.any(np.isfinite(c1_bias_stack)) else math.nan
    else:
        c0_bias_range = float(np.nanmax(c0_bias) - np.nanmin(c0_bias)) if np.any(np.isfinite(c0_bias)) else math.nan
        c1_bias_range = float(np.nanmax(c1_bias) - np.nanmin(c1_bias)) if np.any(np.isfinite(c1_bias)) else math.nan
    rmse_improvements = (c0_frame_rmse - c1_frame_rmse) / np.maximum(np.abs(c0_frame_rmse), np.finfo(float).eps) * 100.0
    p95_improvements = (c0_frame_p95 - c1_frame_p95) / np.maximum(np.abs(c0_frame_p95), np.finfo(float).eps) * 100.0
    valid_rmse_improvements = np.isfinite(rmse_improvements)
    valid_p95_improvements = np.isfinite(p95_improvements)
    result: dict[str, Any] = {
        "split": "validation",
        "metric_weighting": "frame_equal",
        "region": region,
        "frame_count": frame_count,
        "point_count": point_count,
        "c0_bias_mm": pooled_c0.get("bias_mm"),
        "c1_bias_mm": pooled_c1.get("bias_mm"),
        "bias_improvement_mm": (
            finite(pooled_c0.get("bias_mm")) - finite(pooled_c1.get("bias_mm"))
            if math.isfinite(finite(pooled_c0.get("bias_mm"))) and math.isfinite(finite(pooled_c1.get("bias_mm")))
            else math.nan
        ),
        "c0_rmse_mm": pooled_c0.get("rmse_mm"),
        "c1_rmse_mm": pooled_c1.get("rmse_mm"),
        "rmse_improvement_pct": safe_improvement(finite(pooled_c0.get("rmse_mm")), finite(pooled_c1.get("rmse_mm"))),
        "c0_p95_abs_mm": pooled_c0.get("p95_abs_mm"),
        "c1_p95_abs_mm": pooled_c1.get("p95_abs_mm"),
        "p95_improvement_pct": safe_improvement(finite(pooled_c0.get("p95_abs_mm")), finite(pooled_c1.get("p95_abs_mm"))),
        "c0_bias_range_mm": c0_bias_range,
        "c1_bias_range_mm": c1_bias_range,
        "bias_range_improvement_pct": safe_improvement(c0_bias_range, c1_bias_range),
        "c0_frame_rmse_median_mm": float(np.nanmedian(c0_frame_rmse)) if np.any(np.isfinite(c0_frame_rmse)) else math.nan,
        "c1_frame_rmse_median_mm": float(np.nanmedian(c1_frame_rmse)) if np.any(np.isfinite(c1_frame_rmse)) else math.nan,
        "c0_frame_rmse_p95_mm": float(np.nanpercentile(c0_frame_rmse, 95)) if np.any(np.isfinite(c0_frame_rmse)) else math.nan,
        "c1_frame_rmse_p95_mm": float(np.nanpercentile(c1_frame_rmse, 95)) if np.any(np.isfinite(c1_frame_rmse)) else math.nan,
        "c0_frame_p95_median_mm": float(np.nanmedian(c0_frame_p95)) if np.any(np.isfinite(c0_frame_p95)) else math.nan,
        "c1_frame_p95_median_mm": float(np.nanmedian(c1_frame_p95)) if np.any(np.isfinite(c1_frame_p95)) else math.nan,
        "c0_frame_p95_p95_mm": float(np.nanpercentile(c0_frame_p95, 95)) if np.any(np.isfinite(c0_frame_p95)) else math.nan,
        "c1_frame_p95_p95_mm": float(np.nanpercentile(c1_frame_p95, 95)) if np.any(np.isfinite(c1_frame_p95)) else math.nan,
        "frame_rmse_improvement_fraction": float(np.mean(rmse_improvements[valid_rmse_improvements] > 0.0)) if np.any(valid_rmse_improvements) else math.nan,
        "median_frame_rmse_improvement_pct": float(np.nanmedian(rmse_improvements)) if np.any(np.isfinite(rmse_improvements)) else math.nan,
        "frame_p95_improvement_fraction": float(np.mean(p95_improvements[valid_p95_improvements] > 0.0)) if np.any(valid_p95_improvements) else math.nan,
        "median_frame_p95_improvement_pct": float(np.nanmedian(p95_improvements)) if np.any(np.isfinite(p95_improvements)) else math.nan,
    }
    if region == "global":
        c0_ratio = per_frame_values(rows, "c0_edge_middle_ratio")
        c1_ratio = per_frame_values(rows, "c1_edge_middle_ratio")
        result.update(
            {
                "c0_edge_middle_ratio_median": float(np.nanmedian(c0_ratio)) if np.any(np.isfinite(c0_ratio)) else math.nan,
                "c1_edge_middle_ratio_median": float(np.nanmedian(c1_ratio)) if np.any(np.isfinite(c1_ratio)) else math.nan,
                "c0_edge_middle_ratio_p95": float(np.nanpercentile(c0_ratio, 95)) if np.any(np.isfinite(c0_ratio)) else math.nan,
                "c1_edge_middle_ratio_p95": float(np.nanpercentile(c1_ratio, 95)) if np.any(np.isfinite(c1_ratio)) else math.nan,
                "edge_middle_ratio_improvement_pct": safe_improvement(
                    float(np.nanmedian(c0_ratio)) if np.any(np.isfinite(c0_ratio)) else math.nan,
                    float(np.nanmedian(c1_ratio)) if np.any(np.isfinite(c1_ratio)) else math.nan,
                ),
            }
        )
    else:
        result.update(
            {
                "c0_edge_middle_ratio_median": math.nan,
                "c1_edge_middle_ratio_median": math.nan,
                "c0_edge_middle_ratio_p95": math.nan,
                "c1_edge_middle_ratio_p95": math.nan,
                "edge_middle_ratio_improvement_pct": math.nan,
            }
        )
    return result


def classify(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[str, list[str]]:
    global_row = metrics["global"]
    top = metrics["top"]
    middle = metrics["middle"]
    bottom = metrics["bottom"]
    worst = metrics["worst_region"]
    top_good = (
        finite(top["rmse_improvement_pct"]) > 0.0
        and finite(top["p95_improvement_pct"]) > 0.0
        and finite(top["frame_rmse_improvement_fraction"]) >= MIN_EDGE_FRAME_IMPROVEMENT_FRACTION
    )
    bottom_good = (
        finite(bottom["rmse_improvement_pct"]) > 0.0
        and finite(bottom["p95_improvement_pct"]) > 0.0
        and finite(bottom["frame_rmse_improvement_fraction"]) >= MIN_EDGE_FRAME_IMPROVEMENT_FRACTION
    )
    middle_ok = (
        finite(middle["rmse_improvement_pct"]) >= -MAX_MIDDLE_DEGRADATION_PCT
        and finite(middle["p95_improvement_pct"]) >= -MAX_MIDDLE_DEGRADATION_PCT
    )
    global_good = (
        finite(global_row["rmse_improvement_pct"]) >= MIN_GLOBAL_RMSE_IMPROVEMENT_PCT
        and finite(global_row["p95_improvement_pct"]) >= 0.0
    )
    worst_good = (
        finite(worst["median_frame_rmse_improvement_pct"]) >= -MAX_WORST_DEGRADATION_PCT
        and finite(worst["median_frame_p95_improvement_pct"]) >= -MAX_WORST_DEGRADATION_PCT
    )
    ratio_ok = (
        finite(global_row.get("c1_edge_middle_ratio_median"))
        <= finite(global_row.get("c0_edge_middle_ratio_median")) * (1.0 + MAX_EDGE_MIDDLE_DEGRADATION_PCT / 100.0)
    )
    bias_ok = (
        finite(global_row["c1_bias_range_mm"]) <= finite(global_row["c0_bias_range_mm"]) * (1.0 + MAX_BIAS_RANGE_DEGRADATION_PCT / 100.0)
        and finite(worst["c1_bias_range_mm"]) <= finite(worst["c0_bias_range_mm"]) * (1.0 + MAX_BIAS_RANGE_DEGRADATION_PCT / 100.0)
    )
    reasons = [
        f"global_good={global_good}",
        f"top_good={top_good}",
        f"bottom_good={bottom_good}",
        f"middle_ok={middle_ok}",
        f"worst_good={worst_good}",
        f"edge_middle_ok={ratio_ok}",
        f"bias_range_ok={bias_ok}",
    ]
    if global_good and top_good and bottom_good and middle_ok and worst_good and ratio_ok and bias_ok:
        return "PASS", reasons
    partial = (
        finite(global_row["rmse_improvement_pct"]) > 0.0
        and finite(worst["median_frame_rmse_improvement_pct"]) >= -5.0
        and finite(middle["rmse_improvement_pct"]) >= -5.0
    )
    return ("PARTIAL" if partial else "FAIL"), reasons


def render_report(
    model: Mapping[str, Any],
    model_artifact_sha256: str,
    metrics: Mapping[str, Mapping[str, Any]],
    per_frame: Sequence[Mapping[str, Any]],
    verdict: str,
    verdict_reasons: Sequence[str],
    validation_inventory_rows: Sequence[Mapping[str, Any]],
    validation_image_count: int,
) -> str:
    global_row = metrics["global"]
    top_bottom_both_good = (
        finite(metrics["top"]["rmse_improvement_pct"]) > 0.0
        and finite(metrics["top"]["p95_improvement_pct"]) > 0.0
        and finite(metrics["bottom"]["rmse_improvement_pct"]) > 0.0
        and finite(metrics["bottom"]["p95_improvement_pct"]) > 0.0
    )
    lines = [
        "# Frozen C1_4k independent Validation",
        "",
        f"`C1_INDEPENDENT_VALIDATION = {verdict}`",
        "",
        f"是否值得进入实际标准件/高度恢复全视场验收：**{'YES' if verdict == 'PASS' else 'CONDITIONAL' if verdict == 'PARTIAL' else 'NO'}**。该结论只针对本次 frozen Validation，不代表跳过后续验收。",
        "",
        "## Frozen boundary",
        "",
        f"- Frozen model：`C1_4k`，4 interior knots、8 basis、cubic、penalty = **{PENALTY:g}**。",
        f"- frozen_c1_model.json SHA-256：`{model_artifact_sha256}`；parameter SHA-256：`{model['parameter_sha256']}`。",
        f"- FIT：30 frames（001–018、025–036），{model['fit']['point_count']:,} rays；frame-balanced total weight = 1 per frame。",
        f"- Frozen PCA s domain：[{model['pca_s']['domain_min']:.12g}, {model['pca_s']['domain_max']:.12g}]；s/t、knots、penalty、region definition 均未调整。",
        "- C0 = Frozen Circular Cone；C1 = `lambda_cone + F(s)`；truth = checkerboard PnP plane ray intersection。",
        "- 未重新拟合 K/D 或 Cone；不做 C2/C3；最终成功的完整 Validation 评价在 frozen JSON 写入后执行一次，之后不再重评。",
        "- 执行记录：此前两次代码级路径/字段错误在写出 metrics/verdict 前中止，未改变 frozen 参数；本报告只采用随后成功的完整评价结果。",
        "",
        "## Validation scope",
        "",
        f"- Validation frames：{', '.join(VALIDATION_IDS)}；成功评价 pass 处理 triplet image：**{validation_image_count}**（每帧 chess/nolaser/laser 一次）。",
        f"- `validation_per_frame.csv` 共 **{len(per_frame)}** 行；raw image manifest/hash 均通过。",
        "- Top/Middle/Bottom 固定为 `v∈[0,300) / [300,2700) / [2700,3000)`；没有按 Validation 结果改变分区。",
        f"- Region support：Top 有效 frame/point = **{int(metrics['top']['frame_count'])}/{int(metrics['top']['point_count'])}**；Middle = **{int(metrics['middle']['frame_count'])}/{int(metrics['middle']['point_count'])}**；Bottom = **{int(metrics['bottom']['frame_count'])}/{int(metrics['bottom']['point_count'])}**。Top/Bottom edge evidence 偏 sparse。",
        "",
        "## Aggregate comparison (frame-equal)",
        "",
        "| region | C0 RMSE | C1 RMSE | RMSE improvement | C0 P95 | C1 P95 | P95 improvement | C0 bias range | C1 bias range | frame RMSE improve fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("global", "top", "middle", "bottom", "worst_region"):
        row = metrics[region]
        lines.append(
            f"| {region} | {finite(row['c0_rmse_mm']):.6g} | {finite(row['c1_rmse_mm']):.6g} | {finite(row['rmse_improvement_pct']):.3f}% | "
            f"{finite(row['c0_p95_abs_mm']):.6g} | {finite(row['c1_p95_abs_mm']):.6g} | {finite(row['p95_improvement_pct']):.3f}% | "
            f"{finite(row['c0_bias_range_mm']):.6g} | {finite(row['c1_bias_range_mm']):.6g} | {finite(row['frame_rmse_improvement_fraction']):.3f} |"
        )
    lines += [
        "",
        f"- Global edge/middle ratio median：C0 **{finite(global_row.get('c0_edge_middle_ratio_median')):.6g}** → C1 **{finite(global_row.get('c1_edge_middle_ratio_median')):.6g}**；改善 **{finite(global_row.get('edge_middle_ratio_improvement_pct')):.3f}%**。",
        f"- Global frame-equal RMSE improvement median：**{finite(global_row['median_frame_rmse_improvement_pct']):.3f}%**；P95 improvement median：**{finite(global_row['median_frame_p95_improvement_pct']):.3f}%**。",
        "",
        "## Top / Bottom / Middle decision",
        "",
        f"- Top：RMSE improvement **{finite(metrics['top']['rmse_improvement_pct']):.3f}%**，P95 improvement **{finite(metrics['top']['p95_improvement_pct']):.3f}%**，逐 frame RMSE 正改善比例 **{finite(metrics['top']['frame_rmse_improvement_fraction']):.3f}**。",
        f"- Bottom：RMSE improvement **{finite(metrics['bottom']['rmse_improvement_pct']):.3f}%**，P95 improvement **{finite(metrics['bottom']['p95_improvement_pct']):.3f}%**，逐 frame RMSE 正改善比例 **{finite(metrics['bottom']['frame_rmse_improvement_fraction']):.3f}**。",
        f"- Top/Bottom 同时稳健改善：**{'YES' if top_bottom_both_good else 'NO'}**；Top 的 RMSE/P95 = **{finite(metrics['top']['rmse_improvement_pct']):.3f}%/{finite(metrics['top']['p95_improvement_pct']):.3f}%**，Bottom = **{finite(metrics['bottom']['rmse_improvement_pct']):.3f}%/{finite(metrics['bottom']['p95_improvement_pct']):.3f}%**。",
        f"- Middle：RMSE improvement **{finite(metrics['middle']['rmse_improvement_pct']):.3f}%**，P95 improvement **{finite(metrics['middle']['p95_improvement_pct']):.3f}%**；判定阈值为不恶化超过 **{MAX_MIDDLE_DEGRADATION_PCT:.1f}%**。",
        "",
        "## Fixed verdict gates",
        "",
        f"- PASS requires global RMSE improvement ≥ {MIN_GLOBAL_RMSE_IMPROVEMENT_PCT:.1f}% and non-worse P95; both Top and Bottom RMSE/P95 positive with ≥{MIN_EDGE_FRAME_IMPROVEMENT_FRACTION:.2f} frame positive fraction; Middle degradation ≤{MAX_MIDDLE_DEGRADATION_PCT:.1f}%; worst-region degradation ≤{MAX_WORST_DEGRADATION_PCT:.1f}%; edge/middle ratio degradation ≤{MAX_EDGE_MIDDLE_DEGRADATION_PCT:.1f}%; bias-range degradation ≤{MAX_BIAS_RANGE_DEGRADATION_PCT:.1f}%。",
        f"- Gate details：{'; '.join(verdict_reasons)}。",
        f"- `C1_INDEPENDENT_VALIDATION = {verdict}`：这是一次 frozen Validation 评价结果，不能据此重新调整模型。",
        "",
        "## Provenance",
        "",
        f"- points：`{model['input_points']['path']}`，SHA-256 `{model['input_points']['sha256']}`。",
        f"- PCA summary：`{model['pca_s']['source_summary_sha256']}`。",
        f"- Frozen provenance：`{model['frozen_cone']['provenance']['path']}`，SHA-256 `{model['frozen_cone']['provenance']['sha256']}`。",
        f"- Formal Cone：`{model['frozen_cone']['formal_cone']['path']}`，SHA-256 `{model['frozen_cone']['formal_cone']['sha256']}`。",
        "",
    ]
    return "\n".join(lines)


def render_existing_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    model, model_artifact_sha256 = load_frozen_json(output_dir / "frozen_c1_model.json")
    metric_rows = csv_rows(output_dir / "validation_metrics.csv")
    per_frame = csv_rows(output_dir / "validation_per_frame.csv")
    metric_map = {str(row["region"]): row for row in metric_rows}
    verdict, verdict_reasons = classify(metric_map)
    (output_dir / "report.md").write_text(
        render_report(model, model_artifact_sha256, metric_map, per_frame, verdict, verdict_reasons, [], len(per_frame) * 3),
        encoding="utf-8",
    )
    return {"report": str(output_dir / "report.md"), "C1_INDEPENDENT_VALIDATION": verdict, "validation_read": False}


def evaluate_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    model_path = output_dir / "frozen_c1_model.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Freeze C1_4k first: {model_path}")
    existing = [output_dir / name for name in OUTPUT_NAMES if name != "frozen_c1_model.json" and (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Validation outputs already exist; refusing a second evaluation: {existing}")
    model, model_artifact_sha256 = load_frozen_json(model_path)
    app, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    current_hashes = runtime_file_hashes(app)
    if current_hashes != model["frozen_runtime"]["calibration_files"]:
        raise RuntimeError("Frozen runtime K/D file hashes no longer match frozen_c1_model.json")
    frozen_cone, _ = board.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    if canonical_sha256(clean_json(frozen_cone)) != canonical_sha256(model["frozen_cone"]["runtime_model"]):
        raise RuntimeError("Frozen Cone runtime model differs from frozen_c1_model.json")
    inventory = validation_inventory(args.data_root, args.edge_data_root)
    spline = frozen_spline(model)
    frame_rows: list[dict[str, Any]] = []
    point_parts: list[dict[str, np.ndarray]] = []
    validation_image_count = 0
    for item in inventory:
        frame, arrays = process_validation_frame(item, model, calibration, reconstruction_params, intrinsics, spline)
        frame_rows.append(make_per_frame_row(frame, arrays))
        point_parts.append(arrays)
        validation_image_count += 3
    all_v = np.concatenate([part["v"] for part in point_parts])
    all_frame_ids = np.concatenate([part["frame_id"] for part in point_parts])
    all_c0 = np.concatenate([part["c0"] for part in point_parts])
    all_c1 = np.concatenate([part["c1"] for part in point_parts])
    metric_rows = [aggregate_region(region, frame_rows, all_frame_ids, all_v, all_c0, all_c1) for region in ("global", "top", "middle", "bottom", "worst_region")]
    metric_map = {str(row["region"]): row for row in metric_rows}
    verdict, verdict_reasons = classify(metric_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_metrics.csv", metric_rows)
    write_csv(output_dir / "validation_per_frame.csv", frame_rows)
    (output_dir / "report.md").write_text(
        render_report(model, model_artifact_sha256, metric_map, frame_rows, verdict, verdict_reasons, inventory, validation_image_count),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "validation_frame_count": len(frame_rows),
        "validation_image_count": validation_image_count,
        "valid_point_count": int(len(all_v)),
        "C1_INDEPENDENT_VALIDATION": verdict,
        "enter_full_field_acceptance": verdict == "PASS",
        "selected_model": "C1_4k",
    }


def self_test() -> None:
    values = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    weights = frame_equal_weights(np.asarray(["001", "001", "002"]))
    result = weighted_metric(values, weights)
    if result["point_count"] != 3 or not math.isfinite(float(result["rmse_mm"])):
        raise RuntimeError("weighted metric self-test failed")
    if safe_improvement(10.0, 9.0) != 10.0:
        raise RuntimeError("improvement self-test failed")
    print("self-test ok; no Validation data opened")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    modes = int(args.freeze_only) + int(args.evaluate_validation) + int(args.self_test) + int(args.render_report_only)
    if modes != 1:
        raise SystemExit("Choose exactly one of --freeze-only, --evaluate-validation, --self-test, --render-report-only")
    if args.self_test:
        self_test()
        return 0
    if args.render_report_only:
        print(json.dumps(render_existing_report(args), ensure_ascii=False, indent=2))
        return 0
    if args.freeze_only:
        path, artifact_sha, model = freeze_model(args)
        print(json.dumps({"frozen_model": str(path), "artifact_sha256": artifact_sha, "parameter_sha256": model["parameter_sha256"], "model_id": model["model_id"], "validation_opened": False}, ensure_ascii=False, indent=2))
        return 0
    result = evaluate_validation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
