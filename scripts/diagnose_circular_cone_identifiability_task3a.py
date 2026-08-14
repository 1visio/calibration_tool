#!/usr/bin/env python3
"""FIT-only Circular Cone identifiability audit (Task 3A).

This entry point deliberately keeps the acquisition split explicit.  It uses
FIT 001--018 and 025--036 only for every fit, Jacobian, profile, threshold and
interpretation.  Validation 019--024 and 037--040 are recorded in the split
registry/provenance but are never reconstructed or scored here.

The diagnostic full-fit and jackknife use the production CircularConeModel.fit
objective (frame-balanced sampling, soft_l1 loss and the existing bounds).  All
reconstruction comparisons call the public production
``reconstruct_uv_to_ground`` function with in-memory candidate models.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import yaml
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
for path in (SCRIPT_PATH.parent, MEASUREMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_laser_plane_triplet_coverage as coverage  # noqa: E402
import audit_triplet_edge_extension_observability as edge  # noqa: E402
import analyze_circular_cone_parameter_sensitivity as sensitivity  # noqa: E402
import fit_laser_models_from_triplets as triplets  # noqa: E402
from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "cone_identifiability_audit"
)
OLD_POINTS = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0811"
    / "laser_model"
    / "calibration_points.csv"
)
OLD_GEOMETRY = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "triplet_coverage_audit"
    / "triplet_frame_geometry.csv"
)
OLD_PROVENANCE = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "triplet_coverage_audit"
    / "triplet_provenance.csv"
)
EDGE_PROVENANCE = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "triplet_metric_observability_edge_extension"
    / "triplet_edge_extension_provenance.csv"
)
FIT_EXTENSION = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "data"
    / "laser_plane"
    / "fit_edge_extension"
)
VALIDATION_HOLDOUT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "data"
    / "laser_plane"
    / "validation_edge_holdout"
)
MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
FORMAL_FIT_CONFIG = CALIBRATION_TOOL_ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
FORMAL_CONE = MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"

FIT_IDS = [f"{value:03d}" for value in list(range(1, 19)) + list(range(25, 37))]
VALIDATION_IDS = [f"{value:03d}" for value in list(range(19, 25)) + list(range(37, 41))]
FORMAL_V_MIN = 241.99769349665084
FORMAL_V_MAX = 2731.977874579812
GRID_V_STEP = 30.0
GRID_U_COUNT = 41
GRID_V_COUNT = 101
M0_U_STEP = 0.5
PROFILE_POINTS = 17
PROFILE_MAX_NFEV = 500
APEX_ALPHA_PROFILE_POINTS = 7
APEX_ALPHA_PROFILE_MAX_NFEV = 300
SVD_RATIO_THRESHOLD = 1.0e-6
PROFILE_SCALE_LIMIT = 4.0
WEAK_EPS = 1.0e-3
REGIONS = (
    ("top_formal_edge", 0.0, 300.0),
    ("middle_formal", 300.0, 2700.0),
    ("bottom_formal_edge", 2700.0, 3000.0),
)


@dataclass
class FrameRecord:
    frame_id: str
    split: str
    pixels_uv: np.ndarray
    truth_points: np.ndarray
    plane: np.ndarray
    pnp_rmse_px: float | None
    point_count: int
    quality_warnings: str
    source: str
    m0_lambda: np.ndarray | None = None
    m0_invalid: np.ndarray | None = None
    m0_points_camera: np.ndarray | None = None
    m0_residual: np.ndarray | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}g}"


def region_for_v(v: float) -> str:
    if v < 300.0:
        return "top_formal_edge"
    if v < 2700.0:
        return "middle_formal"
    return "bottom_formal_edge"


def metric(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "bias": float("nan"), "mae": float("nan"), "rmse": float("nan"), "p95": float("nan"), "max_abs": float("nan")}
    return {
        "count": int(finite.size),
        "bias": float(np.mean(finite)),
        "mae": float(np.mean(np.abs(finite))),
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "p95": float(np.percentile(np.abs(finite), 95)),
        "max_abs": float(np.max(np.abs(finite))),
    }


def load_old_records() -> list[FrameRecord]:
    geometry = {row["frame_id"]: row for row in csv_rows(OLD_GEOMETRY)}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in csv_rows(OLD_POINTS):
        frame_id = f"{int(row['image_id']):03d}"
        if frame_id in FIT_IDS[:18]:
            groups[frame_id].append(row)
    records: list[FrameRecord] = []
    for frame_id in FIT_IDS[:18]:
        rows = groups.get(frame_id, [])
        if not rows:
            raise RuntimeError(f"Missing original FIT frame {frame_id} in {OLD_POINTS}")
        pixels = np.asarray([[float(row["u_px"]), float(row["v_px"])] for row in rows], dtype=np.float64)
        truth = np.asarray([[float(row["Xc_mm"]), float(row["Yc_mm"]), float(row["Zc_mm"])] for row in rows], dtype=np.float64)
        first = rows[0]
        plane = np.asarray([float(first["board_nx"]), float(first["board_ny"]), float(first["board_nz"]), float(first["board_d_mm"])], dtype=np.float64)
        g = geometry.get(frame_id, {})
        records.append(
            FrameRecord(
                frame_id=frame_id,
                split="fit",
                pixels_uv=pixels,
                truth_points=truth,
                plane=plane,
                pnp_rmse_px=float(g["pnp_reprojection_rmse_px"]) if g.get("pnp_reprojection_rmse_px") else None,
                point_count=len(rows),
                quality_warnings="",
                source="original_0811_triplet",
            )
        )
    return records


def load_extension_records(intrinsics: tuple[np.ndarray, np.ndarray, Any]) -> tuple[list[FrameRecord], list[dict[str, Any]]]:
    points, geometry, provenance = edge.extract_extension(FIT_EXTENSION, "fit", intrinsics, True)
    records: list[FrameRecord] = []
    for frame_id in sorted(points, key=int):
        data = points[frame_id]
        g = geometry[frame_id]
        truth = coverage.plane_ray_truth(
            data["u"],
            data["v"],
            np.asarray([g["plane_nx"], g["plane_ny"], g["plane_nz"]], dtype=np.float64),
            float(g["plane_d"]),
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
        )
        if not np.all(truth["valid"]):
            raise RuntimeError(f"Invalid extension ray-plane truth for frame {frame_id}")
        records.append(
            FrameRecord(
                frame_id=frame_id,
                split="fit",
                pixels_uv=np.column_stack([data["u"], data["v"]]).astype(np.float64),
                truth_points=np.asarray(truth["points"], dtype=np.float64),
                plane=np.asarray([g["plane_nx"], g["plane_ny"], g["plane_nz"], g["plane_d"]], dtype=np.float64),
                pnp_rmse_px=float(next(row["pnp_rmse_px"] for row in provenance if row["frame_id"] == frame_id)),
                point_count=len(data["u"]),
                quality_warnings=str(next(row["quality_warnings"] for row in provenance if row["frame_id"] == frame_id)),
                source="fit_edge_extension",
            )
        )
    return records, provenance


def explicit_split_registry() -> dict[str, Any]:
    return {
        "registry_version": 1,
        "fit": FIT_IDS,
        "validation": VALIDATION_IDS,
        "validation_is_never_loaded": True,
        "manifest_split_is_not_authoritative": True,
        "formal_working_domain_v_px": [FORMAL_V_MIN, FORMAL_V_MAX],
    }


def load_validation_metadata() -> list[dict[str, Any]]:
    """Read metadata only; no validation image, point or residual is opened."""
    result: list[dict[str, Any]] = []
    old_geom = {row["frame_id"]: row for row in csv_rows(OLD_GEOMETRY)}
    old_prov = {row["frame_id"]: row for row in csv_rows(OLD_PROVENANCE)}
    edge_prov = {row["frame_id"]: row for row in csv_rows(EDGE_PROVENANCE)} if EDGE_PROVENANCE.is_file() else {}
    for frame_id in VALIDATION_IDS:
        if frame_id in old_geom:
            row = old_geom[frame_id]
            result.append({
                "frame_id": frame_id,
                "split": "validation",
                "source": "original_0811_triplet",
                "pnp_rmse_px": row.get("pnp_reprojection_rmse_px", ""),
                "point_count": row.get("laser_point_count", ""),
                "quality_warnings": old_prov.get(frame_id, {}).get("quality_warnings", ""),
                "opened_in_task3a": False,
            })
        elif frame_id in edge_prov:
            row = edge_prov[frame_id]
            result.append({
                "frame_id": frame_id,
                "split": "validation",
                "source": "validation_edge_holdout",
                "pnp_rmse_px": row.get("pnp_rmse_px", ""),
                "point_count": row.get("laser_point_count", ""),
                "quality_warnings": row.get("quality_warnings", ""),
                "manifest_split_tags": row.get("manifest_split_tags", ""),
                "path_resolution": row.get("path_resolution", ""),
                "opened_in_task3a": False,
            })
        else:
            result.append({"frame_id": frame_id, "split": "validation", "opened_in_task3a": False})
    return result


def load_runtime(measurement_config: Path) -> tuple[Any, dict[str, Any], Any, tuple[np.ndarray, np.ndarray, Any]]:
    app_config = load_app_config(measurement_config)
    calibration = load_calibration_files(
        app_config.calibration.intrinsics,
        app_config.calibration.laser_model,
        app_config.calibration.extrinsics,
        app_config.calibration.ground_u_compensation,
    )
    if calibration["laser_model"].get("model_type") != "circular_cone":
        raise RuntimeError("Frozen runtime model is not circular_cone")
    intrinsic_data = coverage.load_intrinsics(Path(app_config.calibration.intrinsics))
    return app_config, calibration, app_config.reconstruction, intrinsic_data


def align_points(input_uv: np.ndarray, output_uv: np.ndarray, output_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    queues: dict[tuple[float, float], deque[np.ndarray]] = defaultdict(deque)
    for uv, point in zip(np.asarray(output_uv, dtype=np.float64), np.asarray(output_points, dtype=np.float64)):
        queues[tuple(np.round(uv, 8))].append(np.asarray(point, dtype=np.float64))
    aligned = np.full((len(input_uv), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(input_uv), dtype=bool)
    for index, uv in enumerate(np.asarray(input_uv, dtype=np.float64)):
        key = tuple(np.round(uv, 8))
        if queues[key]:
            aligned[index] = queues[key].popleft()
            valid[index] = True
    return aligned, valid


def model_dict_from_theta(theta: np.ndarray, base_model: Mapping[str, Any], z_range: Sequence[float] | None = None) -> dict[str, Any]:
    model = copy.deepcopy(dict(base_model))
    model["axis_unit_camera"] = sensitivity.angles_to_axis(float(theta[0]), float(theta[1])).tolist()
    model["apex_camera_mm"] = [float(value) for value in theta[2:5]]
    model["half_apex_angle_deg"] = math.degrees(float(theta[5]))
    model["fit_success"] = True
    if z_range is not None:
        model["z_valid_range_mm"] = [float(z_range[0]), float(z_range[1])]
    return model


def production_lambda(uv: np.ndarray, model: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, z_range: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    candidate_calibration = dict(calibration)
    candidate_model = copy.deepcopy(dict(model))
    if z_range is not None:
        candidate_model["z_valid_range_mm"] = [float(z_range[0]), float(z_range[1])]
    candidate_calibration["laser_model"] = candidate_model
    result = reconstruct_uv_to_ground(np.asarray(uv, dtype=np.float64), candidate_calibration, reconstruction_params)
    points, valid = align_points(np.asarray(uv, dtype=np.float64), result.pixels_uv, result.points_camera)
    return points[:, 2], valid & np.isfinite(points[:, 2])


def add_baseline(records: list[FrameRecord], calibration: Mapping[str, Any], reconstruction_params: Any) -> None:
    model = calibration["laser_model"]
    for record in records:
        lam, valid = production_lambda(record.pixels_uv, model, calibration, reconstruction_params)
        record.m0_lambda = lam
        record.m0_invalid = ~valid
        points = np.full((len(lam), 3), np.nan, dtype=np.float64)
        points[:, 2] = lam
        record.m0_points_camera = points
        record.m0_residual = cone_scalar_residual(sensitivity.theta_from_model(model), record.truth_points)


def flatten_truth(records: Sequence[FrameRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate([record.truth_points for record in records], axis=0)
    frames = np.concatenate([np.full(len(record.truth_points), record.frame_id, dtype="U3") for record in records])
    v = np.concatenate([record.pixels_uv[:, 1] for record in records])
    u = np.concatenate([record.pixels_uv[:, 0] for record in records])
    return points, frames, u, v


def select_formal_points(records: Sequence[FrameRecord], max_points: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen_points: list[np.ndarray] = []
    chosen_frames: list[np.ndarray] = []
    chosen_indices: list[np.ndarray] = []
    per_frame = max(20, max_points // max(len(records), 1))
    for record in records:
        idx = triplets.uniform_subsample(np.arange(len(record.truth_points), dtype=int), per_frame)
        chosen_points.append(record.truth_points[idx])
        chosen_frames.append(np.full(len(idx), record.frame_id, dtype="U3"))
        chosen_indices.append(np.column_stack([np.full(len(idx), int(record.frame_id)), idx]))
    points = np.concatenate(chosen_points, axis=0)
    frames = np.concatenate(chosen_frames, axis=0)
    indices = np.concatenate(chosen_indices, axis=0)
    if len(points) > max_points:
        idx = triplets.uniform_subsample(np.arange(len(points), dtype=int), max_points)
        points, frames, indices = points[idx], frames[idx], indices[idx]
    return points, frames, indices


def frame_equal_weights(frame_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(frame_ids, return_counts=True)
    lookup = {key: count for key, count in zip(ids, counts)}
    weights = np.asarray([1.0 / lookup[key] for key in frame_ids], dtype=np.float64)
    return weights / np.mean(weights)


def cone_scalar_residual(theta: np.ndarray, points: np.ndarray) -> np.ndarray:
    theta_axis, phi_axis, cx, cy, cz, alpha = np.asarray(theta, dtype=np.float64)
    axis = sensitivity.angles_to_axis(float(theta_axis), float(phi_axis))
    apex = np.asarray([cx, cy, cz], dtype=np.float64)
    q = np.asarray(points, dtype=np.float64) - apex
    axial = q @ axis
    radial = np.sqrt(np.maximum(np.sum(q * q, axis=1) - axial * axial, 0.0))
    return radial / max(math.tan(float(alpha)), 1.0e-9) - axial


def cone_fit_vector(theta: np.ndarray, points: np.ndarray, sqrt_weights: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    model = triplets.CircularConeModel(dict(cfg))
    return model._residual(np.asarray(theta, dtype=np.float64), np.asarray(points, dtype=np.float64), np.asarray(sqrt_weights, dtype=np.float64))


def formal_fit(records: Sequence[FrameRecord], cfg: Mapping[str, Any], initial_model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    points, frame_ids, indices = select_formal_points(records, int(cfg.get("fit_max_points", 3000)))
    weights = frame_equal_weights(frame_ids)
    fit_cfg = dict(cfg)
    model = triplets.CircularConeModel(fit_cfg)
    if initial_model is None:
        model.fit(points, frame_ids)
    else:
        # Keep the production optimizer/objective/bounds unchanged, but make
        # the frozen M0 one of its explicit starting configurations.  The
        # plane object supplies M0 axis and apex projection; the formal fit
        # still evaluates all existing +/- axis, apex and alpha starts.
        axis = np.asarray(initial_model["axis_unit_camera"], dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-15)
        apex = np.asarray(initial_model["apex_camera_mm"], dtype=np.float64)
        fit_cfg["apex_initial_mm"] = apex.tolist()
        fit_cfg["alpha_initial_deg"] = float(initial_model["half_apex_angle_deg"])
        model = triplets.CircularConeModel(fit_cfg)
        seed_plane = triplets.PlaneModel()
        seed_plane.normal = axis
        seed_plane.d = -float(axis @ apex)
        model.fit(points, frame_ids, plane=seed_plane)
    model_dict = model.to_dict()
    theta = sensitivity.theta_from_model(model_dict)
    objective_vector = cone_fit_vector(theta, points, np.sqrt(weights), cfg)
    scalar_all = [cone_scalar_residual(theta, record.truth_points) for record in records]
    return {
        "model": model,
        "model_dict": model_dict,
        "theta": theta,
        "selected_points": points,
        "selected_frames": frame_ids,
        "selected_indices": indices,
        "selected_weights": weights,
        "objective_vector": objective_vector,
        # CircularConeModel.fit flips the stored axis to the physical sheet
        # after scipy returns.  Re-evaluate the objective at that deployed
        # parameter vector; model.cost is the pre-flip scipy cost and can differ
        # only through the negative-axial soft penalty.
        "optimizer_cost": 0.5 * float(np.sum(objective_vector**2)),
        "objective_mse": float(np.mean(objective_vector**2)),
        "surface_residuals": {record.frame_id: residual for record, residual in zip(records, scalar_all)},
        "status": "success" if model.fit_success else "unknown",
        "fit_success": bool(model.fit_success),
        "z_range_mm": [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
    }


def residual_metrics_by_region(records: Sequence[FrameRecord], residuals: Mapping[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    values = []
    v_values = []
    for record in records:
        values.append(np.asarray(residuals[record.frame_id], dtype=np.float64))
        v_values.append(record.pixels_uv[:, 1])
    flat = np.concatenate(values)
    v = np.concatenate(v_values)
    output: dict[str, dict[str, float | int]] = {"global": metric(flat)}
    for name, low, high in REGIONS:
        output[name] = metric(flat[(v >= low) & (v < high)])
    return output


def finite_jacobian(theta: np.ndarray, points: np.ndarray, sqrt_weights: np.ndarray, cfg: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    base_steps = np.asarray([spec.base_step for spec in sensitivity.PARAMETERS], dtype=np.float64)
    columns: list[np.ndarray] = []
    stability_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(sensitivity.PARAMETERS):
        derivatives: list[np.ndarray] = []
        for multiplier in (0.3, 1.0, 3.0):
            step = float(base_steps[index] * multiplier)
            plus = theta.copy(); plus[index] += step
            minus = theta.copy(); minus[index] -= step
            d = (cone_fit_vector(plus, points, sqrt_weights, cfg) - cone_fit_vector(minus, points, sqrt_weights, cfg)) / (2.0 * step)
            derivatives.append(d)
        selected = derivatives[1]
        columns.append(selected)
        for multiplier, derivative in zip((0.3, 1.0, 3.0), derivatives):
            denom = max(float(np.linalg.norm(selected)), 1.0e-15)
            stability_rows.append({
                "parameter": spec.name,
                "step_multiplier": multiplier,
                "step": base_steps[index] * multiplier,
                "derivative_norm": float(np.linalg.norm(derivative)),
                "relative_to_1x": float(np.linalg.norm(derivative - selected) / denom),
                "selected": multiplier == 1.0,
            })
    return np.column_stack(columns), stability_rows


def scaled_svd(fullfit: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    points = fullfit["selected_points"]
    weights = fullfit["selected_weights"]
    theta = fullfit["theta"]
    sqrt_weights = np.sqrt(weights)
    raw = fullfit["objective_vector"]
    robust_weight = 1.0 / np.sqrt(1.0 + (raw / float(cfg.get("f_scale_mm", 0.1))) ** 2)
    jacobian, stability = finite_jacobian(theta, points, sqrt_weights, cfg)
    residual_weights = np.concatenate([sqrt_weights, sqrt_weights])
    matrix = jacobian * (residual_weights * np.sqrt(robust_weight))[:, None] * sensitivity.PARAMETER_SCALES[None, :]
    matrix /= math.sqrt(float(np.sum(weights)))
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    ratio = singular / max(singular[0], 1.0e-30)
    weak = vt[-1]
    second = vt[-2]
    return {
        "jacobian": jacobian,
        "stability": stability,
        "matrix": matrix,
        "singular_values": singular,
        "relative_values": ratio,
        "right_vectors": vt,
        "weak": weak,
        "second": second,
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
        "effective_rank": int(np.count_nonzero(ratio >= SVD_RATIO_THRESHOLD)),
        "robust_weight": robust_weight,
        "weak_apex_alpha_loading": float(np.sqrt(np.sum(weak[[2, 3, 4, 5]] ** 2))),
    }


def build_grid(records: Sequence[FrameRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_all = np.concatenate([record.pixels_uv[:, 0] for record in records])
    u_min = max(0.0, math.floor(float(np.min(u_all)) / 10.0) * 10.0 - 20.0)
    u_max = min(2999.0, math.ceil(float(np.max(u_all)) / 10.0) * 10.0 + 20.0)
    u_values = np.linspace(u_min, u_max, GRID_U_COUNT)
    v_values = np.linspace(FORMAL_V_MIN, FORMAL_V_MAX, GRID_V_COUNT)
    uu, vv = np.meshgrid(u_values, v_values)
    uv = np.column_stack([uu.ravel(), vv.ravel()])
    return uv, u_values, v_values


def production_lambda_grid(theta: np.ndarray, base_model: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, uv: np.ndarray, z_range: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    model = model_dict_from_theta(theta, base_model, z_range)
    return production_lambda(uv, model, calibration, reconstruction_params, z_range)


def region_delta_metrics(v: np.ndarray, delta: np.ndarray) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {"global": metric(delta)}
    for name, low, high in REGIONS:
        output[name] = metric(delta[(v >= low) & (v < high)])
    return output


def jackknife_prediction_rows(frame_id: str, v_grid: np.ndarray, delta_grid: np.ndarray, valid_grid: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v_flat = v_grid.ravel()
    d_flat = delta_grid.ravel()
    valid = valid_grid.ravel()
    starts = np.arange(math.floor(FORMAL_V_MIN / GRID_V_STEP) * GRID_V_STEP, FORMAL_V_MAX + GRID_V_STEP, GRID_V_STEP)
    for start in starts:
        end = start + GRID_V_STEP
        mask = (v_flat >= start) & (v_flat < end)
        stats = metric(d_flat[mask & valid])
        rows.append({
            "omitted_frame": frame_id,
            "v_start_px": start,
            "v_end_px": end,
            "v_center_px": 0.5 * (start + end),
            "region": region_for_v(0.5 * (start + end)),
            "valid_grid_count": int(np.count_nonzero(mask & valid)),
            "invalid_grid_count": int(np.count_nonzero(mask & ~valid)),
            "median_abs_delta_lambda_mm": stats["mae"],
            "p95_abs_delta_lambda_mm": stats["p95"],
            "max_abs_delta_lambda_mm": stats["max_abs"],
        })
    return rows


def run_jackknife(records: Sequence[FrameRecord], cfg: Mapping[str, Any], fullfit: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, grid_uv: np.ndarray, grid_v: np.ndarray, initial_model: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    full_grid_lambda, full_grid_valid = production_lambda_grid(fullfit["theta"], calibration["laser_model"], calibration, reconstruction_params, grid_uv, fullfit["z_range_mm"])
    result_models: dict[str, Any] = {}
    all_ids = [record.frame_id for record in records]
    for omitted in all_ids:
        train = [record for record in records if record.frame_id != omitted]
        result = formal_fit(train, cfg, initial_model)
        result_models[omitted] = result
        delta_theta = result["theta"] - fullfit["theta"]
        normalized = delta_theta / sensitivity.PARAMETER_SCALES
        train_metrics = residual_metrics_by_region(train, result["surface_residuals"])
        heldout_record = next(record for record in records if record.frame_id == omitted)
        heldout_residual = cone_scalar_residual(result["theta"], heldout_record.truth_points)
        heldout_metrics = metric(heldout_residual)
        candidate_grid_lambda, candidate_grid_valid = production_lambda_grid(result["theta"], calibration["laser_model"], calibration, reconstruction_params, grid_uv, fullfit["z_range_mm"])
        common = full_grid_valid & candidate_grid_valid & np.isfinite(full_grid_lambda) & np.isfinite(candidate_grid_lambda)
        delta_grid = candidate_grid_lambda - full_grid_lambda
        grid_stats = metric(delta_grid[common])
        row: dict[str, Any] = {
            "omitted_frame": omitted,
            "train_frame_count": len(train),
            "train_point_count": int(sum(len(record.truth_points) for record in train)),
            "status": result["status"],
            "fit_success": result["fit_success"],
            "optimizer_cost": result["optimizer_cost"],
            "objective_mse": result["objective_mse"],
            "train_rmse_mm": train_metrics["global"]["rmse"],
            "heldout_rmse_mm": heldout_metrics["rmse"],
            "heldout_p95_abs_mm": heldout_metrics["p95"],
            "grid_valid_count": int(np.count_nonzero(common)),
            "grid_invalid_count": int(np.count_nonzero(~common)),
            "grid_median_abs_delta_lambda_mm": grid_stats["mae"],
            "grid_p95_abs_delta_lambda_mm": grid_stats["p95"],
            "grid_max_abs_delta_lambda_mm": grid_stats["max_abs"],
            "normalized_delta_l2": float(np.linalg.norm(normalized)),
            "normalized_delta_max_abs": float(np.max(np.abs(normalized))),
        }
        for index, spec in enumerate(sensitivity.PARAMETERS):
            row[f"theta_{spec.name}"] = result["theta"][index]
            row[f"delta_{spec.name}"] = delta_theta[index]
            row[f"normalized_delta_{spec.name}"] = normalized[index]
        for region in ("top_formal_edge", "middle_formal", "bottom_formal_edge"):
            row[f"train_{region}_rmse_mm"] = train_metrics[region]["rmse"]
        rows.append(row)
        prediction_rows.extend(jackknife_prediction_rows(omitted, grid_v, delta_grid, common))
        print(f"JACKKNIFE {omitted}: cost={result['optimizer_cost']:.6g} train_rmse={train_metrics['global']['rmse']:.6g} heldout_rmse={heldout_metrics['rmse']:.6g}", flush=True)
    return rows, prediction_rows, result_models


def weak_profile(fullfit: Mapping[str, Any], svd: Mapping[str, Any], cfg: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, grid_uv: np.ndarray, grid_v: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theta0 = fullfit["theta"]
    weak = np.asarray(svd["weak"], dtype=np.float64)
    # The nuisance complement is every right-singular vector except the
    # weakest (last) one; using [1:] would accidentally include the weak vector
    # and make the profile coordinate unconstrained.
    basis = np.asarray(svd["right_vectors"][:-1], dtype=np.float64).T
    scales = sensitivity.PARAMETER_SCALES
    points = fullfit["selected_points"]
    sqrt_weights = np.sqrt(fullfit["selected_weights"])
    z_range = fullfit["z_range_mm"]
    base_model = fullfit["model_dict"]
    full_lambda, full_valid = production_lambda_grid(theta0, base_model, {"laser_model": base_model}, reconstruction_params, grid_uv, z_range)
    # The production call above needs the complete calibration dictionary; the
    # caller replaces this placeholder calibration before using this function.
    del full_lambda, full_valid
    feasible = []
    for t in np.linspace(-PROFILE_SCALE_LIMIT, PROFILE_SCALE_LIMIT, PROFILE_POINTS):
        candidate = theta0 + scales * (t * weak)
        if np.all(candidate >= triplets.LOWER_BOUNDS if hasattr(triplets, "LOWER_BOUNDS") else np.asarray([0.0, -math.pi, -1000.0, -1000.0, -500.0, math.radians(60.0)])):
            feasible.append(float(t))
    if not feasible:
        feasible = [0.0]
    profile_rows: list[dict[str, Any]] = []
    profile_v_rows: list[dict[str, Any]] = []
    return profile_rows, profile_v_rows


def profile_with_calibration(fullfit: Mapping[str, Any], svd: Mapping[str, Any], cfg: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, grid_uv: np.ndarray, grid_v: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theta0 = np.asarray(fullfit["theta"], dtype=np.float64)
    weak = np.asarray(svd["weak"], dtype=np.float64)
    basis = np.asarray(svd["right_vectors"][:-1], dtype=np.float64).T
    scales = sensitivity.PARAMETER_SCALES
    points = fullfit["selected_points"]
    sqrt_weights = np.sqrt(fullfit["selected_weights"])
    base_model = fullfit["model_dict"]
    lower = np.asarray([0.0, -math.pi, -1000.0, -1000.0, -500.0, math.radians(60.0)])
    upper = np.asarray([math.pi, math.pi, 1000.0, 1000.0, 500.0, math.radians(89.95)])
    full_lambda, full_valid = production_lambda_grid(theta0, base_model, calibration, reconstruction_params, grid_uv, fullfit["z_range_mm"])
    t_values = []
    for t in np.linspace(-PROFILE_SCALE_LIMIT, PROFILE_SCALE_LIMIT, PROFILE_POINTS):
        candidate = theta0 + scales * (float(t) * weak)
        if np.all(candidate >= lower) and np.all(candidate <= upper):
            t_values.append(float(t))
    if 0.0 not in t_values:
        t_values.append(0.0)
    t_values = sorted(set(t_values))
    rows: list[dict[str, Any]] = []
    v_rows: list[dict[str, Any]] = []
    for t in t_values:
        def theta_from_eta(eta: np.ndarray) -> np.ndarray:
            normalized = float(t) * weak + basis @ eta
            return theta0 + scales * normalized

        def residual_eta(eta: np.ndarray) -> np.ndarray:
            theta = theta_from_eta(eta)
            violation = np.maximum(lower - theta, 0.0) + np.maximum(theta - upper, 0.0)
            vector = cone_fit_vector(theta, points, sqrt_weights, cfg)
            return np.concatenate([vector, 1.0e3 * violation])

        try:
            optimized = least_squares(residual_eta, np.zeros(basis.shape[1]), loss=str(cfg.get("loss", "soft_l1")), f_scale=float(cfg.get("f_scale_mm", 0.1)), max_nfev=PROFILE_MAX_NFEV)
            eta = optimized.x
            status = str(optimized.status)
        except Exception as error:  # pragma: no cover - recorded in output
            eta = np.zeros(basis.shape[1])
            status = f"error:{type(error).__name__}"
        theta = theta_from_eta(eta)
        theta = np.minimum(np.maximum(theta, lower), upper)
        vector = cone_fit_vector(theta, points, sqrt_weights, cfg)
        objective = 0.5 * float(np.sum(vector**2))
        candidate_model = model_dict_from_theta(theta, base_model, fullfit["z_range_mm"])
        lam, valid = production_lambda_grid(theta, base_model, calibration, reconstruction_params, grid_uv, fullfit["z_range_mm"])
        common = full_valid & valid & np.isfinite(full_lambda) & np.isfinite(lam)
        delta = lam - full_lambda
        stats = region_delta_metrics(grid_v.ravel(), delta, common) if False else None
        row: dict[str, Any] = {
            "record_type": "profile",
            "profile_type": "weak_direction_with_nuisance_refit",
            "displacement_t": t,
            "status": status,
            "optimizer_cost": objective,
            "relative_cost_to_t0": float(objective / max(fullfit["optimizer_cost"], 1.0e-30)),
            "valid_grid_count": int(np.count_nonzero(common)),
            "invalid_grid_count": int(np.count_nonzero(~common)),
        }
        for index, spec in enumerate(sensitivity.PARAMETERS):
            row[f"theta_{spec.name}"] = theta[index]
            row[f"normalized_{spec.name}"] = (theta[index] - theta0[index]) / scales[index]
        for name, low, high in REGIONS:
            mask = (grid_v.ravel() >= low) & (grid_v.ravel() < high) & common
            values = np.abs(delta[mask])
            row[f"{name}_delta_lambda_median_mm"] = float(np.median(values)) if values.size else float("nan")
            row[f"{name}_delta_lambda_p95_mm"] = float(np.percentile(values, 95)) if values.size else float("nan")
            row[f"{name}_delta_lambda_max_mm"] = float(np.max(values)) if values.size else float("nan")
        rows.append(row)
        starts = np.arange(math.floor(FORMAL_V_MIN / GRID_V_STEP) * GRID_V_STEP, FORMAL_V_MAX + GRID_V_STEP, GRID_V_STEP)
        for start in starts:
            end = start + GRID_V_STEP
            mask = (grid_v.ravel() >= start) & (grid_v.ravel() < end) & common
            values = np.abs(delta[mask])
            v_rows.append({
                "record_type": "profile_v",
                "profile_type": "weak_direction_with_nuisance_refit",
                "displacement_t": t,
                "v_start_px": start,
                "v_end_px": end,
                "v_center_px": 0.5 * (start + end),
                "valid_grid_count": int(values.size),
                "delta_lambda_median_mm": float(np.median(values)) if values.size else float("nan"),
                "delta_lambda_p95_mm": float(np.percentile(values, 95)) if values.size else float("nan"),
                "delta_lambda_max_mm": float(np.max(values)) if values.size else float("nan"),
            })
    return rows, v_rows


def apex_alpha_profile(fullfit: Mapping[str, Any], svd: Mapping[str, Any], cfg: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, grid_uv: np.ndarray, grid_v: np.ndarray) -> list[dict[str, Any]]:
    """Scan the apex component and alpha component of the weak direction.

    The scan is expressed in the predeclared normalized parameter coordinates.
    The four-dimensional complement is re-optimized at each grid location, so
    this is a two-dimensional profile rather than a frozen parameter slice.
    Validation is not involved.
    """
    theta0 = np.asarray(fullfit["theta"], dtype=np.float64)
    weak = np.asarray(svd["weak"], dtype=np.float64)
    scales = sensitivity.PARAMETER_SCALES
    apex_norm = float(np.linalg.norm(weak[2:5]))
    if apex_norm <= 1.0e-12:
        return []
    apex_direction = np.zeros(6, dtype=np.float64)
    apex_direction[2:5] = weak[2:5] / apex_norm
    alpha_direction = np.zeros(6, dtype=np.float64)
    alpha_direction[5] = 1.0
    fixed = np.column_stack([apex_direction, alpha_direction])
    # The columns are orthonormal in normalized coordinates.  The remaining
    # four right-singular vectors form the nuisance complement.
    _, _, vt = np.linalg.svd(fixed.T, full_matrices=True)
    nuisance_basis = vt[2:].T
    points = fullfit["selected_points"]
    sqrt_weights = np.sqrt(fullfit["selected_weights"])
    base_model = fullfit["model_dict"]
    lower = np.asarray([0.0, -math.pi, -1000.0, -1000.0, -500.0, math.radians(60.0)])
    upper = np.asarray([math.pi, math.pi, 1000.0, 1000.0, 500.0, math.radians(89.95)])
    full_lambda, full_valid = production_lambda_grid(theta0, base_model, calibration, reconstruction_params, grid_uv, fullfit["z_range_mm"])
    values = np.linspace(-PROFILE_SCALE_LIMIT, PROFILE_SCALE_LIMIT, APEX_ALPHA_PROFILE_POINTS)
    rows: list[dict[str, Any]] = []
    for apex_t in values:
        for alpha_t in values:
            def theta_from_eta(eta: np.ndarray) -> np.ndarray:
                normalized = float(apex_t) * apex_direction + float(alpha_t) * alpha_direction + nuisance_basis @ eta
                return theta0 + scales * normalized

            def residual_eta(eta: np.ndarray) -> np.ndarray:
                theta = theta_from_eta(eta)
                violation = np.maximum(lower - theta, 0.0) + np.maximum(theta - upper, 0.0)
                vector = cone_fit_vector(theta, points, sqrt_weights, cfg)
                return np.concatenate([vector, 1.0e3 * violation])

            try:
                optimized = least_squares(
                    residual_eta,
                    np.zeros(nuisance_basis.shape[1]),
                    loss=str(cfg.get("loss", "soft_l1")),
                    f_scale=float(cfg.get("f_scale_mm", 0.1)),
                    max_nfev=APEX_ALPHA_PROFILE_MAX_NFEV,
                )
                eta = optimized.x
                status = str(optimized.status)
            except Exception as error:  # pragma: no cover - recorded in output
                eta = np.zeros(nuisance_basis.shape[1])
                status = f"error:{type(error).__name__}"
            theta = np.minimum(np.maximum(theta_from_eta(eta), lower), upper)
            vector = cone_fit_vector(theta, points, sqrt_weights, cfg)
            objective = 0.5 * float(np.sum(vector**2))
            lam, valid = production_lambda_grid(theta, base_model, calibration, reconstruction_params, grid_uv, fullfit["z_range_mm"])
            common = full_valid & valid & np.isfinite(full_lambda) & np.isfinite(lam)
            delta = lam - full_lambda
            row: dict[str, Any] = {
                "record_type": "apex_alpha_profile",
                "apex_projection_t": float(apex_t),
                "alpha_displacement_t": float(alpha_t),
                "status": status,
                "optimizer_cost": objective,
                "relative_cost_to_t0": float(objective / max(fullfit["optimizer_cost"], 1.0e-30)),
                "valid_grid_count": int(np.count_nonzero(common)),
                "invalid_grid_count": int(np.count_nonzero(~common)),
                "apex_direction_x": float(apex_direction[2]),
                "apex_direction_y": float(apex_direction[3]),
                "apex_direction_z": float(apex_direction[4]),
                "alpha_delta_rad": float(scales[5] * alpha_t),
                "apex_delta_mm": float(np.linalg.norm(scales[2:5] * apex_t * apex_direction[2:5])),
            }
            for name, low, high in REGIONS:
                mask = (grid_v.ravel() >= low) & (grid_v.ravel() < high) & common
                values_abs = np.abs(delta[mask])
                row[f"{name}_delta_lambda_p95_mm"] = float(np.percentile(values_abs, 95)) if values_abs.size else float("nan")
                row[f"{name}_delta_lambda_max_mm"] = float(np.max(values_abs)) if values_abs.size else float("nan")
            rows.append(row)
    return rows


def local_sensitivity_rows(model_name: str, theta: np.ndarray, base_model: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, grid_uv: np.ndarray, grid_v: np.ndarray, z_range: Sequence[float]) -> list[dict[str, Any]]:
    base_lambda, base_valid = production_lambda_grid(theta, base_model, calibration, reconstruction_params, grid_uv, z_range)
    rows: list[dict[str, Any]] = []
    grid_v_values = np.asarray(grid_v, dtype=np.float64).ravel()
    values = np.repeat(grid_v_values, len(grid_uv) // max(len(grid_v_values), 1))
    for index, spec in enumerate(sensitivity.PARAMETERS):
        step = float(spec.base_step)
        plus = theta.copy(); plus[index] += step
        minus = theta.copy(); minus[index] -= step
        plus_lambda, plus_valid = production_lambda_grid(plus, base_model, calibration, reconstruction_params, grid_uv, z_range)
        minus_lambda, minus_valid = production_lambda_grid(minus, base_model, calibration, reconstruction_params, grid_uv, z_range)
        valid = base_valid & plus_valid & minus_valid & np.isfinite(base_lambda) & np.isfinite(plus_lambda) & np.isfinite(minus_lambda)
        derivative = (plus_lambda - minus_lambda) / (2.0 * step)
        for center in grid_v_values:
            band = np.abs(values - center) <= max((FORMAL_V_MAX - FORMAL_V_MIN) / (GRID_V_COUNT - 1) * 0.51, 1.0)
            selected = valid & band
            d = derivative[selected]
            rows.append({
                "model": model_name,
                "direction": "single_parameter",
                "parameter": spec.name,
                "unit": spec.unit,
                "step": step,
                "v_px": center,
                "region": region_for_v(center),
                "valid_count": int(d.size),
                "invalid_count": int(np.count_nonzero(band & ~valid)),
                "median_dlambda_dtheta": float(np.median(d)) if d.size else float("nan"),
                "p05_dlambda_dtheta": float(np.percentile(d, 5)) if d.size else float("nan"),
                "p95_dlambda_dtheta": float(np.percentile(d, 95)) if d.size else float("nan"),
                "rmse_dlambda_dtheta": float(np.sqrt(np.mean(d**2))) if d.size else float("nan"),
            })
    return rows


def weak_sensitivity_rows(model_name: str, theta: np.ndarray, weak: np.ndarray, base_model: Mapping[str, Any], calibration: Mapping[str, Any], reconstruction_params: Any, grid_uv: np.ndarray, grid_v: np.ndarray, z_range: Sequence[float]) -> list[dict[str, Any]]:
    base_lambda, base_valid = production_lambda_grid(theta, base_model, calibration, reconstruction_params, grid_uv, z_range)
    eps = WEAK_EPS
    plus = theta + sensitivity.PARAMETER_SCALES * eps * weak
    minus = theta - sensitivity.PARAMETER_SCALES * eps * weak
    plus_lambda, plus_valid = production_lambda_grid(plus, base_model, calibration, reconstruction_params, grid_uv, z_range)
    minus_lambda, minus_valid = production_lambda_grid(minus, base_model, calibration, reconstruction_params, grid_uv, z_range)
    valid = base_valid & plus_valid & minus_valid & np.isfinite(base_lambda) & np.isfinite(plus_lambda) & np.isfinite(minus_lambda)
    derivative = (plus_lambda - minus_lambda) / (2.0 * eps)
    rows: list[dict[str, Any]] = []
    grid_v_values = np.asarray(grid_v, dtype=np.float64).ravel()
    values = np.repeat(grid_v_values, len(grid_uv) // max(len(grid_v_values), 1))
    for center in grid_v_values:
        band = np.abs(values - center) <= max((FORMAL_V_MAX - FORMAL_V_MIN) / (GRID_V_COUNT - 1) * 0.51, 1.0)
        selected = valid & band
        d = derivative[selected]
        rows.append({
            "model": model_name,
            "direction": "weak_svd_direction",
            "parameter": "theta_weak",
            "unit": "normalized displacement",
            "step": eps,
            "v_px": center,
            "region": region_for_v(center),
            "valid_count": int(d.size),
            "invalid_count": int(np.count_nonzero(band & ~valid)),
            "median_dlambda_dtheta": float(np.median(d)) if d.size else float("nan"),
            "p05_dlambda_dtheta": float(np.percentile(d, 5)) if d.size else float("nan"),
            "p95_dlambda_dtheta": float(np.percentile(d, 95)) if d.size else float("nan"),
            "rmse_dlambda_dtheta": float(np.sqrt(np.mean(d**2))) if d.size else float("nan"),
        })
    return rows


def m0_invalid_rows(records: Sequence[FrameRecord], calibration: Mapping[str, Any], reconstruction_params: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        uv = record.pixels_uv
        plus_uv = uv.copy(); plus_uv[:, 0] += M0_U_STEP
        minus_uv = uv.copy(); minus_uv[:, 0] -= M0_U_STEP
        plus, plus_valid = production_lambda(plus_uv, calibration["laser_model"], calibration, reconstruction_params)
        minus, minus_valid = production_lambda(minus_uv, calibration["laser_model"], calibration, reconstruction_params)
        derivative = (plus - minus) / (2.0 * M0_U_STEP)
        for index in np.flatnonzero(~(plus_valid & minus_valid & np.isfinite(derivative))):
            if not plus_valid[index] and not minus_valid[index]:
                reason = "plus_and_minus_invalid"
            elif not plus_valid[index]:
                reason = "plus_invalid"
            elif not minus_valid[index]:
                reason = "minus_invalid"
            else:
                reason = "derivative_nonfinite"
            v = float(uv[index, 1])
            rows.append({
                "frame_id": record.frame_id,
                "u_px": float(uv[index, 0]),
                "v_px": v,
                "derivative_step_px": M0_U_STEP,
                "failure_reason": reason,
                "formal_domain": FORMAL_V_MIN <= v <= FORMAL_V_MAX,
                "edge_region": region_for_v(v),
                "source": record.source,
            })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_plots(out: Path, jackknife: Sequence[Mapping[str, Any]], jack_v: Sequence[Mapping[str, Any]], svd: Mapping[str, Any], profile: Sequence[Mapping[str, Any]], profile_v: Sequence[Mapping[str, Any]], apex_alpha_rows: Sequence[Mapping[str, Any]], sensitivity_rows: Sequence[Mapping[str, Any]]) -> None:
    frame_order = [f"{value:03d}" for value in list(range(1, 19)) + list(range(25, 37))]
    x = np.arange(len(frame_order))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for index, spec in enumerate(sensitivity.PARAMETERS):
        values = [float(next(row for row in jackknife if row["omitted_frame"] == frame)[f"normalized_delta_{spec.name}"]) for frame in frame_order]
        axes[0].plot(x, values, marker=".", linewidth=1.0, label=spec.name)
    axes[0].axhline(0.0, color="#777", linewidth=0.8)
    axes[0].set_ylabel("normalized parameter delta")
    axes[0].set_title("FIT frame jackknife parameter variation")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.2)
    for frame in frame_order:
        selected = [row for row in jack_v if row["omitted_frame"] == frame]
        axes[1].plot([float(row["v_center_px"]) for row in selected], [float(row["p95_abs_delta_lambda_mm"]) for row in selected], alpha=0.35, linewidth=0.8)
    axes[1].set_ylabel("P95 |delta lambda| / mm")
    axes[1].set_xlabel("v / px")
    axes[1].set_title("Jackknife prediction drift envelope")
    axes[1].grid(alpha=0.2)
    axes[1].set_xlim(FORMAL_V_MIN, FORMAL_V_MAX)
    fig.tight_layout(); fig.savefig(out / "jackknife_parameter_variation.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    singular = np.asarray(svd["relative_values"])
    axis.semilogy(np.arange(1, len(singular) + 1), singular, marker="o")
    axis.axhline(SVD_RATIO_THRESHOLD, color="#c53030", linestyle="--", linewidth=1.0, label="effective-rank threshold")
    axis.set_xlabel("singular direction")
    axis.set_ylabel("normalized singular value")
    axis.set_title("FIT full diagnostic normalized singular spectrum")
    axis.grid(alpha=0.2); axis.legend(); fig.tight_layout(); fig.savefig(out / "normalized_singular_value_spectrum.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    names = [spec.name for spec in sensitivity.PARAMETERS]
    xpos = np.arange(len(names))
    axis.bar(xpos - 0.18, svd["weak"], width=0.36, label="weakest")
    axis.bar(xpos + 0.18, svd["second"], width=0.36, label="second weakest")
    axis.axhline(0.0, color="#777", linewidth=0.8)
    axis.set_xticks(xpos, names, rotation=25); axis.set_ylabel("normalized loading")
    axis.set_title("Weak-direction parameter composition"); axis.legend(); axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "weakest_direction_composition.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    for row in profile:
        axes[0].plot(float(row["displacement_t"]), float(row["optimizer_cost"]), "o", color="#2b6cb0")
        axes[1].plot(float(row["displacement_t"]), float(row["top_formal_edge_delta_lambda_p95_mm"]), "o", color="#c53030")
    axes[0].set_ylabel("formal objective cost"); axes[1].set_ylabel("top P95 |delta lambda| / mm"); axes[1].set_xlabel("weak-direction displacement t")
    axes[0].set_title("Weak-direction profile with nuisance refit"); axes[0].grid(alpha=0.2); axes[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "weak_direction_objective_profile.png", dpi=170); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for t in sorted({float(row["displacement_t"]) for row in profile_v}):
        selected = [row for row in profile_v if float(row["displacement_t"]) == t]
        axis.plot([float(row["v_center_px"]) for row in selected], [float(row["delta_lambda_p95_mm"]) for row in selected], label=f"t={t:g}")
    axis.set_xlim(FORMAL_V_MIN, FORMAL_V_MAX); axis.set_xlabel("v / px"); axis.set_ylabel("P95 |delta lambda| / mm")
    axis.set_title("Weak-direction prediction drift versus v"); axis.grid(alpha=0.2); axis.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(out / "weak_direction_delta_lambda_vs_v.png", dpi=170); plt.close(fig)

    if apex_alpha_rows:
        apex_values = sorted({float(row["apex_projection_t"]) for row in apex_alpha_rows})
        alpha_values = sorted({float(row["alpha_displacement_t"]) for row in apex_alpha_rows})
        objective = np.full((len(alpha_values), len(apex_values)), np.nan, dtype=np.float64)
        edge_drift = np.full_like(objective, np.nan)
        apex_index = {value: index for index, value in enumerate(apex_values)}
        alpha_index = {value: index for index, value in enumerate(alpha_values)}
        for row in apex_alpha_rows:
            i = alpha_index[float(row["alpha_displacement_t"])]
            j = apex_index[float(row["apex_projection_t"])]
            objective[i, j] = float(row["relative_cost_to_t0"])
            edge_drift[i, j] = max(float(row["top_formal_edge_delta_lambda_p95_mm"]), float(row["bottom_formal_edge_delta_lambda_p95_mm"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        extent = [min(apex_values), max(apex_values), min(alpha_values), max(alpha_values)]
        im0 = axes[0].imshow(objective, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axes[0].set_title("Apex–alpha profile: relative objective")
        axes[0].set_xlabel("apex weak-component displacement")
        axes[0].set_ylabel("alpha displacement / normalized scale")
        fig.colorbar(im0, ax=axes[0], label="cost / cost(t=0)")
        im1 = axes[1].imshow(edge_drift, origin="lower", aspect="auto", extent=extent, cmap="magma")
        axes[1].set_title("Apex–alpha profile: max edge drift")
        axes[1].set_xlabel("apex weak-component displacement")
        axes[1].set_ylabel("alpha displacement / normalized scale")
        fig.colorbar(im1, ax=axes[1], label="max edge P95 |delta lambda| / mm")
        fig.savefig(out / "apex_alpha_profile.png", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for model_name, color in (("M0", "#2b6cb0"), ("M_diag_fullfit", "#c05621")):
        for parameter in [spec.name for spec in sensitivity.PARAMETERS]:
            selected = [row for row in sensitivity_rows if row["model"] == model_name and row["direction"] == "single_parameter" and row["parameter"] == parameter]
            axes[0 if model_name == "M0" else 1].plot([float(row["v_px"]) for row in selected], [float(row["rmse_dlambda_dtheta"]) for row in selected], label=parameter, linewidth=1.0, alpha=0.8)
    axes[0].set_title("M0 local parameter sensitivity"); axes[1].set_title("M_diag_fullfit local parameter sensitivity")
    axes[1].set_xlabel("v / px"); axes[0].set_ylabel("RMSE |d lambda / d theta|"); axes[1].set_ylabel("RMSE |d lambda / d theta|")
    for axis in axes: axis.grid(alpha=0.2); axis.legend(ncol=3, fontsize=7); axis.set_xlim(FORMAL_V_MIN, FORMAL_V_MAX)
    fig.tight_layout(); fig.savefig(out / "parameter_sensitivity_vs_v.png", dpi=170); plt.close(fig)


def render_report(registry: Mapping[str, Any], records: Sequence[FrameRecord], fullfit: Mapping[str, Any], m0_theta: np.ndarray, m0_metrics: Mapping[str, Mapping[str, Any]], full_metrics: Mapping[str, Mapping[str, Any]], svd: Mapping[str, Any], jackknife: Sequence[Mapping[str, Any]], profile: Sequence[Mapping[str, Any]], apex_alpha_rows: Sequence[Mapping[str, Any]], invalid_rows: Sequence[Mapping[str, Any]], validation_metadata: Sequence[Mapping[str, Any]], cone_hash: str, grid_info: Mapping[str, Any]) -> str:
    weak = np.asarray(svd["weak"])
    weak_names = [f"{spec.name}={weak[index]:+.4f}" for index, spec in enumerate(sensitivity.PARAMETERS)]
    edge_p95 = np.asarray([float(row["grid_p95_abs_delta_lambda_mm"]) for row in jackknife])
    middle_p95 = np.asarray([float(row["train_middle_formal_rmse_mm"]) for row in jackknife])
    flat_profile = max((float(row["relative_cost_to_t0"]) for row in profile), default=float("nan"))
    profile_top_p95 = max((float(row["top_formal_edge_delta_lambda_p95_mm"]) for row in profile), default=float("nan"))
    profile_middle_p95 = max((float(row["middle_formal_delta_lambda_p95_mm"]) for row in profile), default=float("nan"))
    profile_bottom_p95 = max((float(row["bottom_formal_edge_delta_lambda_p95_mm"]) for row in profile), default=float("nan"))
    aa_off_zero = [row for row in apex_alpha_rows if not (abs(float(row["apex_projection_t"])) < 1.0e-12 and abs(float(row["alpha_displacement_t"])) < 1.0e-12)]
    aa_off_zero_min = min((float(row["relative_cost_to_t0"]) for row in aa_off_zero), default=float("nan"))
    aa_max = max((float(row["relative_cost_to_t0"]) for row in apex_alpha_rows), default=float("nan"))
    apex_alpha = svd["weak_apex_alpha_loading"]
    next_option = "C" if (svd["condition_number"] > 1.0e4 or apex_alpha > 0.7) else "D"
    influential = max(jackknife, key=lambda row: float(row["grid_p95_abs_delta_lambda_mm"])) if jackknife else None
    top_delta_rmse = float(full_metrics["top_formal_edge"]["rmse"]) - float(m0_metrics["top_formal_edge"]["rmse"])
    bottom_delta_rmse = float(full_metrics["bottom_formal_edge"]["rmse"]) - float(m0_metrics["bottom_formal_edge"]["rmse"])
    if top_delta_rmse > 0.0 and bottom_delta_rmse < 0.0 and profile_top_p95 < float(m0_metrics["top_formal_edge"]["rmse"]):
        q5 = "WEAK for the top-edge mismatch: the weak apex/alpha valley is real, but its scanned surface drift is only a few microns while the full-fit diagnostic worsens top RMSE."
    elif apex_alpha > 0.7:
        q5 = "PARTIAL: apex/alpha weak coupling is real and can explain a common low-amplitude component, but it does not by itself establish the edge residual mechanism."
    else:
        q5 = "WEAK: no dominant apex/alpha weak direction explains the edge mismatch."
    lines = [
        "# Task 3A — Circular Cone FIT-only identifiability audit",
        "",
        "**FIT_ONLY = TRUE**",
        "**FORMAL_CONE_UNCHANGED = TRUE**",
        f"**NEXT_OPTION = {next_option}**",
        "",
        "## Scope and split isolation",
        "",
        f"- FIT used for every optimization/diagnostic: `{', '.join(registry['fit'])}` ({len(records)} frames).",
        f"- Validation frozen and not opened: `{', '.join(registry['validation'])}`.",
        "- Split role is an explicit registry; acquisition manifest split tags are not authoritative.",
        f"- Formal working domain: v=[{FORMAL_V_MIN:.3f}, {FORMAL_V_MAX:.3f}] px.",
        f"- Evaluation grid: u=[{grid_info['u_min_px']:.1f}, {grid_info['u_max_px']:.1f}] ({GRID_U_COUNT} samples), v formal domain ({GRID_V_COUNT} samples).",
        f"- Formal Cone SHA-256: `{cone_hash}` (before/after identical).",
        "",
        "## Full-FIT diagnostic baseline",
        "",
        "`M_diag_fullfit` is an in-memory diagnostic model produced by the existing CircularConeModel.fit path. It is not a production artifact.",
        "",
        "| region | M0 RMSE / mm | M_diag_fullfit RMSE / mm | M0 P95 / mm | M_diag_fullfit P95 / mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for region in ("global", "top_formal_edge", "middle_formal", "bottom_formal_edge"):
        lines.append(f"| {region} | {fmt(m0_metrics[region]['rmse'])} | {fmt(full_metrics[region]['rmse'])} | {fmt(m0_metrics[region]['p95'])} | {fmt(full_metrics[region]['p95'])} |")
    lines += [
        "",
        f"- Formal optimizer cost: `{fmt(fullfit['optimizer_cost'])}`; selected points: `{len(fullfit['selected_points'])}`; status: `{fullfit['status']}`.",
        "- Objective is the existing frame-balanced sampled Circular Cone residual with `soft_l1`, `f_scale_mm=0.10`, negative-axial penalty and existing bounds; the frozen M0 axis/apex/alpha are explicit solver starts, and no validation-derived weight/threshold was introduced.",
        "",
        "## Parameter comparison",
        "",
        "See `parameter_comparison.csv`; normalized delta uses the predeclared interpretation scales (1 degree for axis angles, 10 mm for apex, 0.1 degree for alpha).",
        "",
        "## Frame jackknife",
        "",
        f"- Leave-one-frame-out count: `{len(jackknife)}`; every omitted frame is a complete FIT frame and is never removed permanently.",
        f"- Jackknife grid P95 |delta lambda| median/max across omitted frames: `{fmt(np.median(edge_p95))}` / `{fmt(np.max(edge_p95))}` mm.",
        f"- Formal-middle training RMSE median/max: `{fmt(np.median(middle_p95))}` / `{fmt(np.max(middle_p95))}` mm.",
        "- `jackknife_prediction_vs_v.csv` separates top, middle and bottom; edge-only growth with stable middle indicates edge prediction instability rather than uniform surface movement.",
        "",
        "## Jacobian / SVD",
        "",
        f"- Condition number: `{fmt(svd['condition_number'])}`; effective rank: `{svd['effective_rank']}/6`; weakest/strongest ratio: `{fmt(svd['relative_values'][-1], 4)}`.",
        f"- Weakest normalized loading: `{', '.join(weak_names)}`.",
        f"- Combined apex/alpha loading norm in weakest direction: `{fmt(apex_alpha, 4)}`.",
        "- SVD is performed on robust-weighted, frame-balanced Jacobian columns after explicit physical-unit scaling; raw mm/rad column magnitudes are not compared.",
        "",
        "## Weak-direction profile",
        "",
        f"- Profile uses FIT-only selected points and nuisance-parameter refit in the five-dimensional complement of the weakest normalized singular direction; no validation result sets the displacement range.",
        f"- Objective cost max/min over the scanned feasible profile (reported as max relative cost to t=0): `{fmt(flat_profile, 7)}`.",
        f"- Weak-profile maximum P95 |delta lambda|: top=`{fmt(profile_top_p95)}` mm, middle=`{fmt(profile_middle_p95)}` mm, bottom=`{fmt(profile_bottom_p95)}` mm; valid grid count is preserved in the CSV.",
        "- The objective is nearly flat while the physical parameters move substantially; this demonstrates parameter non-identifiability, not automatic evidence that the edge residual is explained.",
        "- `weak_direction_profile.csv` / `weak_direction_profile_v.csv` contain the one-dimensional nuisance-refit profile; `apex_alpha_profile.csv` is the requested two-dimensional apex–alpha profile with the remaining four coordinates re-optimized.",
        f"- Apex–alpha 2D profile: minimum off-origin relative cost=`{fmt(aa_off_zero_min, 7)}`, maximum=`{fmt(aa_max, 7)}`; the shallow off-origin valley is aligned with compensating apex/alpha changes, while the corners are not a flat fit-equivalent solution.",
        "",
        "## Local sensitivity",
        "",
        "`local_parameter_sensitivity.csv` reports M0 and M_diag_fullfit d(lambda)/d(theta_i) along v, plus d(lambda)/d(theta_weak). Positive/negative finite differences use the recorded physical step; invalid intersections remain counted.",
        "",
        "## M0 derivative invalid audit",
        "",
        f"- Invalid M0 ±{M0_U_STEP:g}px derivative rows: `{len(invalid_rows)}`; these rows are not silently deleted from the audit.",
        "- The CSV records frame, u/v, reason, formal-domain membership and edge region; counts by frame/bin/region are summarized in this report and provenance JSON.",
        "",
        "## Quality provenance",
        "",
        "- FIT extension frames retain PnP RMSE, Steger point count and dynamic-range warnings. Influential frames are reported by jackknife metrics but are not automatically deleted.",
        "- Validation metadata is registry-only (`opened_in_task3a=false`); no validation residual, profile, Jacobian or model choice is produced in this task.",
        "",
        "## Answers to required questions (FIT-only)",
        "",
        f"- Q1 six-parameter stability: `{'not stably identifiable as individual physical parameters' if svd['condition_number'] > 1.0e4 else 'locally well-conditioned'}`; surface prediction stability is reported separately.",
        f"- Q2 weak direction: `{'YES' if svd['condition_number'] > 1.0e4 else 'NO'}`.",
        f"- Q3 apex/alpha coupling: `{'YES / material loading' if apex_alpha > 0.7 else 'not dominant'}`.",
        f"- Q4 jackknife: edge prediction P95 max=`{fmt(np.max(edge_p95))}` mm; middle training RMSE max=`{fmt(np.max(middle_p95))}` mm; see v-resolved CSV for asymmetry.",
        f"- Q5 top-edge gain mismatch: `{q5}`",
        f"- Q6 top/bottom asymmetry: full-fit top RMSE change=`{fmt(top_delta_rmse)}` mm, bottom change=`{fmt(bottom_delta_rmse)}` mm; this is an asymmetric FIT diagnostic, not a validation claim.",
        f"- Most influential leave-one-frame-out fold by grid P95: omitted `{influential['omitted_frame']}` with `{fmt(influential['grid_p95_abs_delta_lambda_mm'])}` mm." if influential else "- Most influential leave-one-frame-out fold: n/a.",
        f"- Q7 next action: `{next_option}` — {'先解决 Circular Cone 参数弱可辨识/参数化问题' if next_option == 'C' else '进入位置相关 residual decomposition'}。",
        "",
        "## Limits",
        "",
        "- This is a diagnostic audit. M_diag_fullfit and jackknife models must not be deployed or written back to the formal Cone file.",
        "- Full-sensor regions outside the formal v domain are not used to claim identifiability.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 3A FIT-only Circular Cone identifiability audit")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    cone_hash_before = sha256_file(FORMAL_CONE)
    app_config, calibration, reconstruction_params, intrinsics = load_runtime(args.measurement_config.resolve())
    old_records = load_old_records()
    extension_records, extension_provenance = load_extension_records(intrinsics)
    records = old_records + extension_records
    if [record.frame_id for record in records] != FIT_IDS:
        raise RuntimeError("FIT records do not match explicit registry 001-018 + 025-036")
    add_baseline(records, calibration, reconstruction_params)
    cfg_root = triplets.safe_yaml_load(FORMAL_FIT_CONFIG)
    cone_cfg = dict(cfg_root["models"]["cone"])
    fullfit = formal_fit(records, cone_cfg, calibration["laser_model"])
    m0_theta = sensitivity.theta_from_model(calibration["laser_model"])
    m0_residuals = {record.frame_id: cone_scalar_residual(m0_theta, record.truth_points) for record in records}
    m0_metrics = residual_metrics_by_region(records, m0_residuals)
    full_metrics = residual_metrics_by_region(records, fullfit["surface_residuals"])
    grid_uv, u_values, v_values = build_grid(records)
    grid_v = np.repeat(v_values[:, None], len(u_values), axis=1)
    svd = scaled_svd(fullfit, cone_cfg)
    jackknife_rows, jackknife_v_rows, jackknife_models = run_jackknife(records, cone_cfg, fullfit, calibration, reconstruction_params, grid_uv, grid_v, calibration["laser_model"])
    profile_rows, profile_v_rows = profile_with_calibration(fullfit, svd, cone_cfg, calibration, reconstruction_params, grid_uv, grid_v)
    apex_alpha_rows = apex_alpha_profile(fullfit, svd, cone_cfg, calibration, reconstruction_params, grid_uv, grid_v)
    sensitivity_rows = local_sensitivity_rows("M0", m0_theta, calibration["laser_model"], calibration, reconstruction_params, grid_uv, v_values, fullfit["z_range_mm"])
    sensitivity_rows += local_sensitivity_rows("M_diag_fullfit", fullfit["theta"], calibration["laser_model"], calibration, reconstruction_params, grid_uv, v_values, fullfit["z_range_mm"])
    sensitivity_rows += weak_sensitivity_rows("M0", m0_theta, svd["weak"], calibration["laser_model"], calibration, reconstruction_params, grid_uv, v_values, fullfit["z_range_mm"])
    sensitivity_rows += weak_sensitivity_rows("M_diag_fullfit", fullfit["theta"], svd["weak"], calibration["laser_model"], calibration, reconstruction_params, grid_uv, v_values, fullfit["z_range_mm"])
    invalid_rows = m0_invalid_rows(records, calibration, reconstruction_params)
    validation_metadata = load_validation_metadata()
    parameter_rows = []
    for index, spec in enumerate(sensitivity.PARAMETERS):
        delta = fullfit["theta"][index] - m0_theta[index]
        parameter_rows.append({
            "parameter": spec.name,
            "unit": spec.unit,
            "M0_value": m0_theta[index],
            "M_diag_fullfit_value": fullfit["theta"][index],
            "delta_M_diag_minus_M0": delta,
            "normalized_delta": delta / spec.interpretation_scale,
            "interpretation_scale": spec.interpretation_scale,
        })
    singular_rows = []
    for index, value in enumerate(svd["singular_values"]):
        row = {
            "analysis": "fullfit_frame_equal_robust_weighted",
            "singular_index": index + 1,
            "singular_value": value,
            "relative_to_strongest": svd["relative_values"][index],
            "effective": bool(svd["relative_values"][index] >= SVD_RATIO_THRESHOLD),
            "condition_number": svd["condition_number"],
            "effective_rank": svd["effective_rank"],
        }
        for pindex, spec in enumerate(sensitivity.PARAMETERS):
            row[f"loading_{spec.name}"] = svd["right_vectors"][index, pindex]
        singular_rows.append(row)
    coupling_rows = []
    matrix = svd["matrix"]
    norms = np.linalg.norm(matrix, axis=0)
    cosine = (matrix.T @ matrix) / np.outer(norms, norms)
    covariance = np.linalg.pinv(matrix.T @ matrix, rcond=1.0e-12)
    diag = np.maximum(np.diag(covariance), 0.0)
    denom = np.sqrt(np.outer(diag, diag))
    corr = np.zeros_like(covariance); valid = denom > 0.0; corr[valid] = covariance[valid] / denom[valid]; np.fill_diagonal(corr, 1.0)
    for i, first in enumerate(sensitivity.PARAMETERS):
        for j, second in enumerate(sensitivity.PARAMETERS):
            if j <= i: continue
            coupling_rows.append({"parameter_a": first.name, "parameter_b": second.name, "column_cosine": cosine[i, j], "covariance_correlation": corr[i, j]})
    invalid_summary = {
        "by_frame": dict(Counter(row["frame_id"] for row in invalid_rows)),
        "by_region": dict(Counter(row["edge_region"] for row in invalid_rows)),
        "by_300px_bin": dict(Counter(f"{int(float(row['v_px']) // 300) * 300:04d}" for row in invalid_rows)),
    }
    extension_by_frame = {row["frame_id"]: row for row in extension_provenance}
    fit_provenance = []
    for record in records:
        ext = extension_by_frame.get(record.frame_id, {})
        fit_provenance.append({
            "frame_id": record.frame_id,
            "split": "fit",
            "source": record.source,
            "point_count": record.point_count,
            "pnp_rmse_px": record.pnp_rmse_px,
            "quality_warnings": record.quality_warnings,
            "manifest_split_tags": ext.get("manifest_split_tags", ""),
            "path_resolution": ext.get("path_resolution", "original_task2_provenance"),
        })
    registry = explicit_split_registry()
    provenance = {
        "task": "Task 3A Circular Cone identifiability audit",
        "formal_cone_sha256_before": cone_hash_before,
        "formal_cone_sha256_after": sha256_file(FORMAL_CONE),
        "formal_cone_path": str(FORMAL_CONE),
        "measurement_config": str(args.measurement_config.resolve()),
        "formal_fit_config": str(FORMAL_FIT_CONFIG),
        "split_registry": registry,
        "fit_provenance": fit_provenance,
        "validation_metadata_only": validation_metadata,
        "m0_invalid_summary": invalid_summary,
        "grid": {"u_min_px": float(np.min(u_values)), "u_max_px": float(np.max(u_values)), "u_count": len(u_values), "v_min_px": FORMAL_V_MIN, "v_max_px": FORMAL_V_MAX, "v_count": len(v_values)},
        "formal_objective": {"loss": cone_cfg.get("loss"), "f_scale_mm": cone_cfg.get("f_scale_mm"), "negative_axial_penalty": cone_cfg.get("negative_axial_penalty"), "fit_max_points": cone_cfg.get("fit_max_points"), "weighting": "frame_equal"},
        "validation_opened": False,
    }
    (out / "dataset_split.yaml").write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "fullfit_diagnostic_result.json").write_text(json.dumps({"model_name": "M_diag_fullfit", "theta": fullfit["theta"].tolist(), "model": fullfit["model_dict"], "optimizer_cost": fullfit["optimizer_cost"], "objective_mse": fullfit["objective_mse"], "status": fullfit["status"], "fit_success": fullfit["fit_success"], "z_range_mm": fullfit["z_range_mm"], "selected_point_count": len(fullfit["selected_points"]), "m0_theta": m0_theta.tolist(), "formal_metrics_m0": m0_metrics, "formal_metrics_diag": full_metrics}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_csv(out / "parameter_comparison.csv", parameter_rows)
    write_csv(out / "frame_jackknife.csv", jackknife_rows)
    write_csv(out / "jackknife_prediction_vs_v.csv", jackknife_v_rows)
    write_csv(out / "jacobian_svd.csv", singular_rows + svd["stability"])
    write_csv(out / "parameter_coupling.csv", coupling_rows)
    write_csv(out / "weak_direction_profile.csv", profile_rows)
    write_csv(out / "weak_direction_profile_v.csv", profile_v_rows)
    write_csv(out / "apex_alpha_profile.csv", apex_alpha_rows)
    write_csv(out / "local_parameter_sensitivity.csv", sensitivity_rows)
    write_csv(out / "m0_derivative_invalid_audit.csv", invalid_rows)
    save_plots(out, jackknife_rows, jackknife_v_rows, svd, profile_rows, profile_v_rows, apex_alpha_rows, sensitivity_rows)
    grid_info = {"u_min_px": float(np.min(u_values)), "u_max_px": float(np.max(u_values))}
    (out / "report.md").write_text(render_report(registry, records, fullfit, m0_theta, m0_metrics, full_metrics, svd, jackknife_rows, profile_rows, apex_alpha_rows, invalid_rows, validation_metadata, cone_hash_before, grid_info), encoding="utf-8")
    (out / "OUTPUT_FILES.md").write_text("""# Task 3A output files

| file | meaning | boundary |
|---|---|---|
| dataset_split.yaml | explicit FIT/VALIDATION registry | validation is not opened |
| provenance.json | hashes, quality provenance and invalid summary | no model selection from validation |
| fullfit_diagnostic_result.json | in-memory M_diag_fullfit result | not production |
| parameter_comparison.csv | M0 vs full-FIT parameters | normalized interpretation only |
| frame_jackknife.csv | leave-one-FIT-frame-out fits | omitted frame remains diagnostic FIT data |
| jackknife_prediction_vs_v.csv | prediction drift versus v | fixed FIT-derived grid |
| jacobian_svd.csv | scaled SVD, step stability and loadings | local, not global identifiability proof |
| parameter_coupling.csv | column/covariance coupling | local linear diagnostic |
| weak_direction_profile.csv | nuisance-refit weak-direction objective/profile | no validation |
| weak_direction_profile_v.csv | profile prediction drift by v | no validation |
| apex_alpha_profile.csv | two-dimensional apex/alpha profile with nuisance refit | FIT-only diagnostic |
| local_parameter_sensitivity.csv | M0/full-fit d(lambda)/d(theta) versus v | invalids retained in counts |
| m0_derivative_invalid_audit.csv | M0 finite-difference failures | no silent deletion |
| *.png | required diagnostic plots, including apex_alpha_profile.png | presentation aids, not extra evidence |
| report.md | Task 3A conclusions and Q1–Q7 | does not authorize deployment |
""", encoding="utf-8")
    cone_hash_after = sha256_file(FORMAL_CONE)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Cone changed during Task 3A audit")
    print(f"FULLFIT_STATUS={fullfit['status']}")
    print(f"FULLFIT_COST={fullfit['optimizer_cost']:.9g}")
    print(f"SVD_CONDITION={svd['condition_number']:.9g}")
    print(f"WEAK_APEX_ALPHA_LOADING={svd['weak_apex_alpha_loading']:.9g}")
    print(f"JACKKNIFE_ROWS={len(jackknife_rows)}")
    print(f"INVALID_M0_ROWS={len(invalid_rows)}")
    print(f"OUTPUT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
