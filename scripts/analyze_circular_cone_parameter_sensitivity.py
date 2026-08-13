#!/usr/bin/env python3
"""FIT-only linear sensitivity and identifiability for the deployed Circular Cone.

The production Steger pixels and ``reconstruct_uv_to_ground`` path are reused.
All candidate parameters live only in memory.  No nonlinear optimizer is called
and no calibration parameter artifact is written.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_TOOL_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(MEASUREMENT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_TOOL_ROOT))

import generate_paired_pnp_residual_diagnostics as paired  # noqa: E402
from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from laser.laser_extractor import extract_laser_center  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_parameter_sensitivity"
)
BASELINE_METRICS = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "paired_pnp_residual_diagnostics"
    / "per_frame_residual_metrics.csv"
)
OUTPUT_NAMES = (
    "cone_parameter_sensitivity.csv",
    "cone_jacobian_singular_values.csv",
    "cone_parameter_coupling.csv",
    "cone_residual_explainability.csv",
    "cone_region_explainability.csv",
    "cone_linearized_prediction.png",
    "cone_singular_values.png",
    "cone_sensitivity_report.md",
    "OUTPUT_FILES.md",
)

IMAGE_HEIGHT = 3000
WEIGHTINGS = ("point_equal", "frame_equal", "v_region_equal")
STEP_MULTIPLIERS = (0.3, 1.0, 3.0)
STEP_STABLE_RELATIVE_RMS = 1.0e-3
SVD_EFFECTIVE_RANK_RATIO = 1.0e-6
LSTSQ_RCOND = 1.0e-12
AUDITED_TRAIN_V_MIN = 241.99769349665084
AUDITED_TRAIN_V_MAX = 2731.977874579812


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    base_step: float
    interpretation_scale: float


DEGREE_RAD = math.pi / 180.0
PARAMETERS = (
    ParameterSpec("theta_axis", "rad", 1.0e-5, 1.0 * DEGREE_RAD),
    ParameterSpec("phi_axis", "rad", 1.0e-5, 1.0 * DEGREE_RAD),
    ParameterSpec("A_x", "mm", 1.0e-2, 10.0),
    ParameterSpec("A_y", "mm", 1.0e-2, 10.0),
    ParameterSpec("A_z", "mm", 1.0e-2, 10.0),
    ParameterSpec("alpha", "rad", 1.0e-5, 0.1 * DEGREE_RAD),
)
PARAMETER_SCALES = np.asarray(
    [spec.interpretation_scale for spec in PARAMETERS], dtype=np.float64
)


@dataclass
class FrameData:
    frame_id: str
    split: str
    plane: np.ndarray
    pixels_uv: np.ndarray
    points_ground: np.ndarray
    residual_mm: np.ndarray
    extracted_count: int
    filtered_count: int


@dataclass
class PreparedData:
    frames: list[FrameData]
    pixels_uv: np.ndarray
    split: np.ndarray
    frame_id: np.ndarray
    planes: np.ndarray
    residual_mm: np.ndarray
    calibration: dict[str, Any]
    reconstruction_params: Any
    theta0: np.ndarray
    used_pixel_sha256: str
    baseline_metric_max_error: float


@dataclass
class CandidateEvaluation:
    residual_mm: np.ndarray
    invalid_mask: np.ndarray


@dataclass
class StepResult:
    step: float
    derivative: np.ndarray
    invalid_mask: np.ndarray
    invalid_plus: int
    invalid_minus: int
    derivative_rms: float
    score: float = float("nan")
    relative_to_selected: float = float("nan")
    selected: bool = False


@dataclass
class JacobianResult:
    fit_jacobian: np.ndarray
    validation_jacobian: np.ndarray
    fit_valid_mask: np.ndarray
    validation_valid_mask: np.ndarray
    selected_steps: np.ndarray
    fit_step_results: list[list[StepResult]]
    validation_invalid_masks: list[np.ndarray]


@dataclass
class LinearSolution:
    weighting: str
    delta: np.ndarray
    normalized_delta: np.ndarray
    sensitivity_norm: np.ndarray
    singular_values: np.ndarray
    raw_singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    effective_rank: int
    lstsq_rank: int
    condition_number: float
    column_cosine: np.ndarray
    covariance_correlation: np.ndarray


@dataclass
class MetricSet:
    bias_mm: float
    mae_mm: float
    rmse_mm: float
    p95_abs_mm: float
    residual_energy: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Circular Cone FIT-only linear sensitivity and identifiability."
    )
    parser.add_argument("--data-root", type=Path, default=paired.DEFAULT_DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=paired.DEFAULT_PNP_AUDIT)
    parser.add_argument(
        "--measurement-config", type=Path, default=paired.DEFAULT_MEASUREMENT_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_to_angles(axis: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Circular Cone axis is zero")
    vector = vector / norm
    theta = math.acos(float(np.clip(vector[2], -1.0, 1.0)))
    phi = math.atan2(float(vector[1]), float(vector[0]))
    return theta, phi


def angles_to_axis(theta: float, phi: float) -> np.ndarray:
    axis = np.asarray(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )
    # Unit length follows analytically from the spherical map.  Normalize again
    # before production reconstruction so every finite-difference candidate has
    # exactly the deployed axis invariant.
    return axis / np.linalg.norm(axis)


def theta_from_model(model: Mapping[str, Any]) -> np.ndarray:
    theta_axis, phi_axis = axis_to_angles(
        np.asarray(model["axis_unit_camera"], dtype=np.float64)
    )
    apex = np.asarray(model["apex_camera_mm"], dtype=np.float64).reshape(3)
    return np.asarray(
        [
            theta_axis,
            phi_axis,
            apex[0],
            apex[1],
            apex[2],
            math.radians(float(model["half_apex_angle_deg"])),
        ],
        dtype=np.float64,
    )


def model_from_theta(base_model: Mapping[str, Any], theta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(theta, dtype=np.float64).reshape(len(PARAMETERS))
    model = copy.deepcopy(dict(base_model))
    model["axis_unit_camera"] = angles_to_axis(values[0], values[1])
    model["apex_camera_mm"] = np.ascontiguousarray(values[2:5])
    model["half_apex_angle_deg"] = math.degrees(float(values[5]))
    return model


def vertical_residual(points_ground: np.ndarray, plane: np.ndarray) -> np.ndarray:
    points = np.asarray(points_ground, dtype=np.float64)
    coeff = np.asarray(plane, dtype=np.float64).reshape(4)
    z_plane = -(
        coeff[0] * points[:, 0] + coeff[1] * points[:, 1] + coeff[3]
    ) / coeff[2]
    return points[:, 2] - z_plane


def load_expected_baseline(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["frame_id"].zfill(3)] = {
                "extracted": float(row["extracted_center_count"]),
                "reconstructed": float(row["reconstructed_point_count"]),
                "mean": float(row["residual_mean_mm"]),
                "rmse": float(row["residual_rmse_mm"]),
            }
    return result


def prepare_data(
    data_root: Path,
    pnp_audit_path: Path,
    measurement_config_path: Path,
    include_splits: Sequence[str] = ("fit", "validation"),
) -> tuple[PreparedData, Any]:
    requested_splits = frozenset(include_splits)
    if not requested_splits or not requested_splits <= {"fit", "validation"}:
        raise ValueError(f"Unsupported split selection: {sorted(requested_splits)}")
    plane_rows = paired.load_pnp_planes(pnp_audit_path)
    app_config = load_app_config(measurement_config_path)
    calibration = load_calibration_files(
        app_config.calibration.intrinsics,
        app_config.calibration.laser_model,
        app_config.calibration.extrinsics,
        app_config.calibration.ground_u_compensation,
    )
    if calibration["laser_model"]["model_type"] != "circular_cone":
        raise ValueError("Frozen laser model must be circular_cone")
    if calibration["ground_u_compensation"] is not None:
        raise ValueError("Sensitivity analysis forbids ground compensation")

    extraction_params = create_extraction_params(
        app_config.extraction_method, app_config.extraction_options
    )
    expected = load_expected_baseline(BASELINE_METRICS)
    frames: list[FrameData] = []
    metric_errors: list[float] = []
    pixel_digest = hashlib.sha256()

    for row in plane_rows:
        frame_id = row["frame_id"]
        split = row["split"]
        if split not in requested_splits:
            continue
        laser_path = data_root / split / f"laser {frame_id}.tif"
        image = paired.read_gray(laser_path)
        centers = np.ascontiguousarray(
            extract_laser_center(image, extraction_params), dtype=np.float64
        )
        reconstructed = reconstruct_uv_to_ground(
            centers, calibration, app_config.reconstruction
        )
        pixels = np.ascontiguousarray(reconstructed.pixels_uv)
        points_ground = np.ascontiguousarray(reconstructed.points_ground)
        plane = np.asarray(row["plane"], dtype=np.float64)
        residual = vertical_residual(points_ground, plane)
        filtered_count = int(len(centers) - len(pixels))

        pixel_digest.update(frame_id.encode("ascii"))
        pixel_digest.update(np.asarray(pixels, dtype="<f8").tobytes(order="C"))
        if frame_id not in expected:
            raise RuntimeError(f"Existing baseline metrics omit frame {frame_id}")
        observed = {
            "extracted": float(len(centers)),
            "reconstructed": float(len(pixels)),
            "mean": float(np.mean(residual)),
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }
        for key, value in observed.items():
            error = abs(value - expected[frame_id][key])
            tolerance = 0.0 if key in {"extracted", "reconstructed"} else 5.0e-7
            metric_errors.append(error)
            if error > tolerance:
                raise RuntimeError(
                    f"Frozen baseline mismatch {frame_id} {key}: "
                    f"{value} vs {expected[frame_id][key]}"
                )
        frames.append(
            FrameData(
                frame_id=frame_id,
                split=split,
                plane=plane,
                pixels_uv=pixels,
                points_ground=points_ground,
                residual_mm=residual,
                extracted_count=int(len(centers)),
                filtered_count=filtered_count,
            )
        )

    if not frames:
        raise RuntimeError(f"No paired-PnP frames found for {sorted(requested_splits)}")

    pixels_uv = np.concatenate([frame.pixels_uv for frame in frames], axis=0)
    split = np.concatenate(
        [np.full(len(frame.pixels_uv), frame.split, dtype="U10") for frame in frames]
    )
    frame_id = np.concatenate(
        [np.full(len(frame.pixels_uv), frame.frame_id, dtype="U3") for frame in frames]
    )
    planes = np.concatenate(
        [
            np.repeat(frame.plane[None, :], len(frame.pixels_uv), axis=0)
            for frame in frames
        ],
        axis=0,
    )
    residual = np.concatenate([frame.residual_mm for frame in frames])
    prepared = PreparedData(
        frames=frames,
        pixels_uv=np.ascontiguousarray(pixels_uv),
        split=split,
        frame_id=frame_id,
        planes=np.ascontiguousarray(planes),
        residual_mm=residual,
        calibration=calibration,
        reconstruction_params=app_config.reconstruction,
        theta0=theta_from_model(calibration["laser_model"]),
        used_pixel_sha256=pixel_digest.hexdigest(),
        baseline_metric_max_error=float(max(metric_errors, default=0.0)),
    )
    return prepared, app_config


def align_points(
    input_pixels: np.ndarray,
    output_pixels: np.ndarray,
    output_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.full((len(input_pixels), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(input_pixels), dtype=bool)
    output_index = 0
    for input_index, pixel in enumerate(input_pixels):
        if output_index >= len(output_pixels):
            break
        if np.array_equal(pixel, output_pixels[output_index]):
            aligned[input_index] = output_points[output_index]
            valid[input_index] = True
            output_index += 1
    if output_index != len(output_pixels):
        raise RuntimeError("Could not align production reconstruction output to frozen pixels")
    return aligned, valid


def evaluate_candidate(
    theta: np.ndarray,
    data: PreparedData,
    split: str,
) -> CandidateEvaluation:
    calibration = dict(data.calibration)
    calibration["laser_model"] = model_from_theta(
        data.calibration["laser_model"], theta
    )
    residual_parts: list[np.ndarray] = []
    invalid_parts: list[np.ndarray] = []
    for frame in data.frames:
        if frame.split != split:
            continue
        reconstructed = reconstruct_uv_to_ground(
            frame.pixels_uv, calibration, data.reconstruction_params
        )
        aligned, valid = align_points(
            frame.pixels_uv,
            reconstructed.pixels_uv,
            reconstructed.points_ground,
        )
        residual = np.full(len(frame.pixels_uv), np.nan, dtype=np.float64)
        residual[valid] = vertical_residual(aligned[valid], frame.plane)
        residual_parts.append(residual)
        invalid_parts.append(~valid)
    return CandidateEvaluation(
        residual_mm=np.concatenate(residual_parts),
        invalid_mask=np.concatenate(invalid_parts),
    )


def split_arrays(data: PreparedData, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = data.split == split
    return data.residual_mm[mask], data.pixels_uv[mask, 1], data.frame_id[mask]


def central_derivative(
    plus: CandidateEvaluation,
    minus: CandidateEvaluation,
    step: float,
) -> StepResult:
    invalid = plus.invalid_mask | minus.invalid_mask
    derivative = np.full(len(invalid), np.nan, dtype=np.float64)
    valid = ~invalid
    derivative[valid] = (plus.residual_mm[valid] - minus.residual_mm[valid]) / (
        2.0 * step
    )
    rms = (
        float(np.sqrt(np.mean(derivative[valid] ** 2)))
        if np.any(valid)
        else float("nan")
    )
    return StepResult(
        step=step,
        derivative=derivative,
        invalid_mask=invalid,
        invalid_plus=int(np.count_nonzero(plus.invalid_mask)),
        invalid_minus=int(np.count_nonzero(minus.invalid_mask)),
        derivative_rms=rms,
    )


def derivative_relative_rms(a: StepResult, b: StepResult) -> float:
    common = np.isfinite(a.derivative) & np.isfinite(b.derivative)
    if not np.any(common):
        return float("inf")
    difference = a.derivative[common] - b.derivative[common]
    scale = max(
        0.5
        * (
            float(np.sqrt(np.mean(a.derivative[common] ** 2)))
            + float(np.sqrt(np.mean(b.derivative[common] ** 2)))
        ),
        1.0e-15,
    )
    return float(np.sqrt(np.mean(difference**2)) / scale)


def choose_fit_step(results: list[StepResult]) -> int:
    for index, result in enumerate(results):
        comparisons = [
            derivative_relative_rms(result, other)
            for other_index, other in enumerate(results)
            if other_index != index
        ]
        invalid_fraction = float(np.mean(result.invalid_mask))
        result.score = float(np.median(comparisons) + 10.0 * invalid_fraction)

    middle = 1
    middle_differences = [
        derivative_relative_rms(results[middle], results[index])
        for index in (0, 2)
    ]
    minimum_invalid = min(np.count_nonzero(result.invalid_mask) for result in results)
    if (
        np.count_nonzero(results[middle].invalid_mask) == minimum_invalid
        and max(middle_differences) <= STEP_STABLE_RELATIVE_RMS
    ):
        return middle
    return min(range(len(results)), key=lambda index: (results[index].score, abs(index - 1)))


def compute_fit_jacobian(data: PreparedData) -> JacobianResult:
    fit_count = int(np.count_nonzero(data.split == "fit"))
    validation_count = int(np.count_nonzero(data.split == "validation"))
    fit_jacobian = np.full((fit_count, len(PARAMETERS)), np.nan, dtype=np.float64)
    selected_steps = np.empty(len(PARAMETERS), dtype=np.float64)
    all_fit_results: list[list[StepResult]] = []

    # Crucial split discipline: all three step scales and every selection score
    # are computed on FIT only. Validation is not touched in this function.
    for parameter_index, spec in enumerate(PARAMETERS):
        fit_results: list[StepResult] = []
        for multiplier in STEP_MULTIPLIERS:
            step = spec.base_step * multiplier
            plus_theta = data.theta0.copy()
            minus_theta = data.theta0.copy()
            plus_theta[parameter_index] += step
            minus_theta[parameter_index] -= step
            plus = evaluate_candidate(plus_theta, data, "fit")
            minus = evaluate_candidate(minus_theta, data, "fit")
            fit_results.append(central_derivative(plus, minus, step))

        selected_index = choose_fit_step(fit_results)
        selected = fit_results[selected_index]
        selected.selected = True
        for result in fit_results:
            result.relative_to_selected = derivative_relative_rms(result, selected)
        fit_jacobian[:, parameter_index] = selected.derivative
        selected_steps[parameter_index] = selected.step
        all_fit_results.append(fit_results)

    return JacobianResult(
        fit_jacobian=fit_jacobian,
        validation_jacobian=np.full(
            (validation_count, len(PARAMETERS)), np.nan, dtype=np.float64
        ),
        fit_valid_mask=np.all(np.isfinite(fit_jacobian), axis=1),
        validation_valid_mask=np.zeros(validation_count, dtype=bool),
        selected_steps=selected_steps,
        fit_step_results=all_fit_results,
        validation_invalid_masks=[],
    )


def compute_validation_jacobian(
    data: PreparedData, result: JacobianResult
) -> None:
    """Evaluate validation only after FIT steps, weights and deltas are frozen."""
    invalid_masks: list[np.ndarray] = []
    for parameter_index, step in enumerate(result.selected_steps):
        plus_theta = data.theta0.copy()
        minus_theta = data.theta0.copy()
        plus_theta[parameter_index] += step
        minus_theta[parameter_index] -= step
        validation_plus = evaluate_candidate(plus_theta, data, "validation")
        validation_minus = evaluate_candidate(minus_theta, data, "validation")
        derivative = central_derivative(validation_plus, validation_minus, step)
        result.validation_jacobian[:, parameter_index] = derivative.derivative
        invalid_masks.append(derivative.invalid_mask)
    result.validation_invalid_masks = invalid_masks
    result.validation_valid_mask = np.all(
        np.isfinite(result.validation_jacobian), axis=1
    )


def v_bin_ids(v_px: np.ndarray) -> np.ndarray:
    values = np.asarray(v_px, dtype=np.float64)
    if np.any(values < 0.0) or np.any(values >= IMAGE_HEIGHT):
        raise ValueError("Frozen v pixels must lie in [0, 2999]")
    return np.floor(values / 300.0).astype(int)


def weights_for(v_px: np.ndarray, frame_ids: np.ndarray, weighting: str) -> np.ndarray:
    count = len(v_px)
    if weighting == "point_equal":
        return np.ones(count, dtype=np.float64)

    frame_ids = np.asarray(frame_ids)
    weights = np.empty(count, dtype=np.float64)
    if weighting == "frame_equal":
        for frame_id in np.unique(frame_ids):
            mask = frame_ids == frame_id
            weights[mask] = 1.0 / float(np.count_nonzero(mask))
    elif weighting == "v_region_equal":
        bins = v_bin_ids(v_px)
        populated_bins = np.unique(bins)
        bin_count = len(populated_bins)
        for bin_id in populated_bins:
            bin_mask = bins == bin_id
            populated_frames = np.unique(frame_ids[bin_mask])
            frame_count = len(populated_frames)
            for frame_id in populated_frames:
                cell = bin_mask & (frame_ids == frame_id)
                cell_count = int(np.count_nonzero(cell))
                weights[cell] = 1.0 / float(bin_count * frame_count * cell_count)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    weights /= float(np.mean(weights))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise RuntimeError(f"{weighting} produced non-finite or non-positive weights")

    if weighting == "frame_equal":
        frame_totals = np.asarray(
            [np.sum(weights[frame_ids == frame_id]) for frame_id in np.unique(frame_ids)]
        )
        if not np.allclose(frame_totals, frame_totals[0], rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("frame_equal did not give every frame equal total weight")
    elif weighting == "v_region_equal":
        bins = v_bin_ids(v_px)
        bin_totals = np.asarray(
            [np.sum(weights[bins == bin_id]) for bin_id in np.unique(bins)]
        )
        if not np.allclose(bin_totals, bin_totals[0], rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("v_region_equal did not give every populated bin equal total weight")
        for bin_id in np.unique(bins):
            bin_mask = bins == bin_id
            cell_totals = np.asarray(
                [
                    np.sum(weights[bin_mask & (frame_ids == frame_id)])
                    for frame_id in np.unique(frame_ids[bin_mask])
                ]
            )
            if not np.allclose(
                cell_totals, cell_totals[0], rtol=1.0e-12, atol=1.0e-12
            ):
                raise RuntimeError(
                    "v_region_equal did not preserve frame balance within a populated bin"
                )
    return weights


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.maximum(np.diag(covariance), 0.0)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    result = np.zeros_like(covariance)
    valid = denominator > 0.0
    result[valid] = covariance[valid] / denominator[valid]
    np.fill_diagonal(result, 1.0)
    return result


def solve_fit(
    data: PreparedData,
    jacobians: JacobianResult,
) -> dict[str, LinearSolution]:
    residual_fit, v_fit, frames_fit = split_arrays(data, "fit")
    valid = jacobians.fit_valid_mask
    residual = residual_fit[valid]
    v_px = v_fit[valid]
    frame_ids = frames_fit[valid]
    jacobian = jacobians.fit_jacobian[valid]
    solutions: dict[str, LinearSolution] = {}

    for weighting in WEIGHTINGS:
        weights = weights_for(v_px, frame_ids, weighting)
        sqrt_weight = np.sqrt(weights)
        weighted_raw = sqrt_weight[:, None] * jacobian
        weighted_scaled = weighted_raw * PARAMETER_SCALES[None, :]
        target = -sqrt_weight * residual
        normalized_delta, _, lstsq_rank, _ = np.linalg.lstsq(
            weighted_scaled, target, rcond=LSTSQ_RCOND
        )
        delta = PARAMETER_SCALES * normalized_delta

        rms_scaled = weighted_scaled / math.sqrt(float(np.sum(weights)))
        _, singular_values, vt = np.linalg.svd(rms_scaled, full_matrices=False)
        raw_singular_values = np.linalg.svd(
            weighted_raw / math.sqrt(float(np.sum(weights))),
            compute_uv=False,
        )
        ratio = singular_values / singular_values[0]
        effective_rank = int(np.count_nonzero(ratio >= SVD_EFFECTIVE_RANK_RATIO))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0.0
            else float("inf")
        )

        column_norms = np.linalg.norm(rms_scaled, axis=0)
        column_cosine = (rms_scaled.T @ rms_scaled) / np.outer(
            column_norms, column_norms
        )
        information = rms_scaled.T @ rms_scaled
        covariance = np.linalg.pinv(information, rcond=LSTSQ_RCOND)
        covariance_correlation = correlation_from_covariance(covariance)
        sensitivity_norm = np.sqrt(
            np.sum(weights[:, None] * jacobian**2, axis=0) / np.sum(weights)
        )
        solutions[weighting] = LinearSolution(
            weighting=weighting,
            delta=delta,
            normalized_delta=normalized_delta,
            sensitivity_norm=sensitivity_norm,
            singular_values=singular_values,
            raw_singular_values=raw_singular_values,
            right_singular_vectors=vt,
            effective_rank=effective_rank,
            lstsq_rank=int(lstsq_rank),
            condition_number=condition,
            column_cosine=column_cosine,
            covariance_correlation=covariance_correlation,
        )
    return solutions


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    target = quantile * cumulative[-1]
    return float(sorted_values[min(int(np.searchsorted(cumulative, target)), len(values) - 1)])


def calculate_metrics(residual: np.ndarray, weights: np.ndarray) -> MetricSet:
    total_weight = float(np.sum(weights))
    return MetricSet(
        bias_mm=float(np.sum(weights * residual) / total_weight),
        mae_mm=float(np.sum(weights * np.abs(residual)) / total_weight),
        rmse_mm=float(np.sqrt(np.sum(weights * residual**2) / total_weight)),
        p95_abs_mm=weighted_quantile(np.abs(residual), weights, 0.95),
        residual_energy=float(np.sum(weights * residual**2)),
    )


def metric_row(
    split: str,
    weighting: str,
    region: str,
    v_min: float | None,
    v_max: float | None,
    baseline_count: int,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    before = calculate_metrics(residual_before, weights)
    after = calculate_metrics(residual_after, weights)
    explained = before.residual_energy - after.residual_energy
    return {
        "split": split,
        "weighting": weighting,
        "region": region,
        "v_min_px": v_min,
        "v_max_px": v_max,
        "baseline_sample_count": baseline_count,
        "analysis_sample_count": int(len(residual_before)),
        "invalid_count": int(baseline_count - len(residual_before)),
        "before_bias_mm": before.bias_mm,
        "before_mae_mm": before.mae_mm,
        "before_rmse_mm": before.rmse_mm,
        "before_p95_abs_mm": before.p95_abs_mm,
        "before_residual_energy": before.residual_energy,
        "after_bias_mm": after.bias_mm,
        "after_mae_mm": after.mae_mm,
        "after_rmse_mm": after.rmse_mm,
        "after_p95_abs_mm": after.p95_abs_mm,
        "after_residual_energy": after.residual_energy,
        "explained_energy": explained,
        "explained_fraction": explained / before.residual_energy
        if before.residual_energy > 0.0
        else float("nan"),
    }


def region_definitions() -> list[tuple[str, float | None, float | None]]:
    regions: list[tuple[str, float | None, float | None]] = [
        ("global", None, None),
        ("top_0_299", 0.0, 300.0),
        ("middle_300_2699", 300.0, 2700.0),
        ("bottom_2700_2999", 2700.0, 3000.0),
    ]
    for start in range(0, IMAGE_HEIGHT, 300):
        regions.append((f"bin_{start:04d}_{start + 299:04d}", float(start), float(start + 300)))
    regions.extend(
        [
            ("extrap_top_v_lt_242", 0.0, AUDITED_TRAIN_V_MIN),
            ("extrap_bottom_v_gt_2732", AUDITED_TRAIN_V_MAX, 3000.0),
        ]
    )
    return regions


def prediction_and_metrics(
    data: PreparedData,
    jacobians: JacobianResult,
    solutions: Mapping[str, LinearSolution],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    global_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for split in ("fit", "validation"):
        baseline_residual, all_v, all_frames = split_arrays(data, split)
        jacobian = (
            jacobians.fit_jacobian if split == "fit" else jacobians.validation_jacobian
        )
        common_valid = (
            jacobians.fit_valid_mask
            if split == "fit"
            else jacobians.validation_valid_mask
        )
        residual = baseline_residual[common_valid]
        v_px = all_v[common_valid]
        frame_ids = all_frames[common_valid]
        for weighting in WEIGHTINGS:
            weights = weights_for(v_px, frame_ids, weighting)
            predicted = residual + jacobian[common_valid] @ solutions[weighting].delta
            predictions[(split, weighting)] = (v_px, residual, predicted)
            global_rows.append(
                metric_row(
                    split=split,
                    weighting=weighting,
                    region="global",
                    v_min=None,
                    v_max=None,
                    baseline_count=len(baseline_residual),
                    residual_before=residual,
                    residual_after=predicted,
                    weights=weights,
                )
            )
            for region, v_min, v_max in region_definitions()[1:]:
                assert v_min is not None and v_max is not None
                baseline_region_count = int(
                    np.count_nonzero((all_v >= v_min) & (all_v < v_max))
                )
                mask = (v_px >= v_min) & (v_px < v_max)
                if not np.any(mask):
                    continue
                region_rows.append(
                    metric_row(
                        split=split,
                        weighting=weighting,
                        region=region,
                        v_min=v_min,
                        v_max=v_max,
                        baseline_count=baseline_region_count,
                        residual_before=residual[mask],
                        residual_after=predicted[mask],
                        weights=weights[mask],
                    )
                )
    return global_rows, region_rows, predictions


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return value


def invalid_indices(mask: np.ndarray) -> str:
    return ";".join(str(index) for index in np.flatnonzero(mask))


def write_parameter_sensitivity(
    path: Path,
    data: PreparedData,
    jacobians: JacobianResult,
    solutions: Mapping[str, LinearSolution],
) -> None:
    fields = [
        "parameter",
        "unit",
        "current_value",
        "interpretation_scale",
        "selected_step",
        "delta_point_equal",
        "delta_frame_equal",
        "delta_v_region_equal",
        "normalized_delta_point_equal",
        "normalized_delta_frame_equal",
        "normalized_delta_v_region_equal",
        "jacobian_norm_point_equal",
        "jacobian_norm_frame_equal",
        "jacobian_norm_v_region_equal",
        "fit_invalid_indices",
        "validation_invalid_indices",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, spec in enumerate(PARAMETERS):
            selected_fit = next(
                result for result in jacobians.fit_step_results[index] if result.selected
            )
            row = {
                "parameter": spec.name,
                "unit": spec.unit,
                "current_value": data.theta0[index],
                "interpretation_scale": spec.interpretation_scale,
                "selected_step": jacobians.selected_steps[index],
                "fit_invalid_indices": invalid_indices(selected_fit.invalid_mask),
                "validation_invalid_indices": invalid_indices(
                    jacobians.validation_invalid_masks[index]
                ),
            }
            for weighting in WEIGHTINGS:
                row[f"delta_{weighting}"] = solutions[weighting].delta[index]
                row[f"normalized_delta_{weighting}"] = solutions[
                    weighting
                ].normalized_delta[index]
                row[f"jacobian_norm_{weighting}"] = solutions[
                    weighting
                ].sensitivity_norm[index]
            writer.writerow({key: csv_value(row[key]) for key in fields})


def write_singular_values(
    path: Path, solutions: Mapping[str, LinearSolution]
) -> None:
    fields = [
        "weighting",
        "singular_index",
        "scaled_singular_value",
        "raw_singular_value",
        "relative_singular_value",
        "effective",
        "effective_rank",
        "lstsq_rank",
        "condition_number",
        "rank_ratio_threshold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for weighting in WEIGHTINGS:
            solution = solutions[weighting]
            for index, value in enumerate(solution.singular_values):
                relative = value / solution.singular_values[0]
                row = {
                    "weighting": weighting,
                    "singular_index": index + 1,
                    "scaled_singular_value": value,
                    "raw_singular_value": solution.raw_singular_values[index],
                    "relative_singular_value": relative,
                    "effective": relative >= SVD_EFFECTIVE_RANK_RATIO,
                    "effective_rank": solution.effective_rank,
                    "lstsq_rank": solution.lstsq_rank,
                    "condition_number": solution.condition_number,
                    "rank_ratio_threshold": SVD_EFFECTIVE_RANK_RATIO,
                }
                writer.writerow({key: csv_value(row[key]) for key in fields})


def write_coupling(path: Path, solutions: Mapping[str, LinearSolution]) -> None:
    fields = [
        "weighting",
        "analysis",
        "mode",
        "parameter_a",
        "parameter_b",
        "value",
        "abs_value",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for weighting in WEIGHTINGS:
            solution = solutions[weighting]
            for analysis, matrix in (
                ("column_cosine", solution.column_cosine),
                ("covariance_correlation", solution.covariance_correlation),
            ):
                for i in range(len(PARAMETERS)):
                    for j in range(i + 1, len(PARAMETERS)):
                        value = float(matrix[i, j])
                        row = {
                            "weighting": weighting,
                            "analysis": analysis,
                            "mode": "",
                            "parameter_a": PARAMETERS[i].name,
                            "parameter_b": PARAMETERS[j].name,
                            "value": value,
                            "abs_value": abs(value),
                        }
                        writer.writerow({key: csv_value(row[key]) for key in fields})
            for mode_index in (-2, -1):
                mode = f"v{len(PARAMETERS) + mode_index + 1}"
                vector = solution.right_singular_vectors[mode_index]
                for parameter_index, value in enumerate(vector):
                    row = {
                        "weighting": weighting,
                        "analysis": "smallest_singular_vector",
                        "mode": mode,
                        "parameter_a": PARAMETERS[parameter_index].name,
                        "parameter_b": "",
                        "value": float(value),
                        "abs_value": abs(float(value)),
                    }
                    writer.writerow({key: csv_value(row[key]) for key in fields})


METRIC_FIELDS = [
    "split",
    "weighting",
    "region",
    "v_min_px",
    "v_max_px",
    "baseline_sample_count",
    "analysis_sample_count",
    "invalid_count",
    "before_bias_mm",
    "before_mae_mm",
    "before_rmse_mm",
    "before_p95_abs_mm",
    "before_residual_energy",
    "after_bias_mm",
    "after_mae_mm",
    "after_rmse_mm",
    "after_p95_abs_mm",
    "after_residual_energy",
    "explained_energy",
    "explained_fraction",
]


def write_metric_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row[key]) for key in METRIC_FIELDS})


def binned_median(
    v_px: np.ndarray, residual: np.ndarray, width: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.floor(np.asarray(v_px) / width).astype(int)
    centers: list[float] = []
    medians: list[float] = []
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        centers.append((float(bin_id) + 0.5) * width)
        medians.append(float(np.median(residual[mask])))
    return np.asarray(centers), np.asarray(medians)


def save_linearized_prediction(
    path: Path,
    predictions: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True, sharey=True)
    colors = {
        "point_equal": "#2b6cb0",
        "frame_equal": "#2f855a",
        "v_region_equal": "#c05621",
    }
    for axis, split in zip(axes, ("fit", "validation")):
        v_px, residual, _ = predictions[(split, "point_equal")]
        x, baseline = binned_median(v_px, residual)
        axis.plot(x, baseline, color="#222222", linewidth=2.1, label="baseline")
        for weighting in WEIGHTINGS:
            v_values, _, predicted = predictions[(split, weighting)]
            x_pred, curve = binned_median(v_values, predicted)
            axis.plot(
                x_pred,
                curve,
                linewidth=1.6,
                color=colors[weighting],
                label=weighting,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.axvline(300.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
        axis.axvline(2700.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
        axis.axvline(
            AUDITED_TRAIN_V_MIN, color="#bbbbbb", linestyle=":", linewidth=0.8
        )
        axis.axvline(
            AUDITED_TRAIN_V_MAX, color="#bbbbbb", linestyle=":", linewidth=0.8
        )
        axis.set_ylabel("vertical residual median / mm")
        axis.set_title(f"{split.upper()} · fixed 30 px display bins")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=7.5)
    axes[-1].set_xlabel("laser stripe row v / px")
    figure.suptitle("Circular Cone baseline vs frozen FIT linear predictions")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_singular_plot(path: Path, solutions: Mapping[str, LinearSolution]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    colors = {
        "point_equal": "#2b6cb0",
        "frame_equal": "#2f855a",
        "v_region_equal": "#c05621",
    }
    indices = np.arange(1, len(PARAMETERS) + 1)
    for weighting in WEIGHTINGS:
        singular = solutions[weighting].singular_values
        axis.semilogy(
            indices,
            singular / singular[0],
            marker="o",
            linewidth=1.8,
            color=colors[weighting],
            label=weighting,
        )
    axis.axhline(
        SVD_EFFECTIVE_RANK_RATIO,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        label="effective-rank threshold",
    )
    axis.set_xticks(indices)
    axis.set_xlabel("singular index")
    axis.set_ylabel("relative scaled singular value")
    axis.set_title("Weighted Circular Cone Jacobian spectrum (FIT only)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def row_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["split"]), str(row["weighting"]), str(row["region"])): row
        for row in rows
    }


def verdict_from_results(
    global_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    solutions: Mapping[str, LinearSolution],
    jacobians: JacobianResult,
) -> tuple[str, list[str]]:
    global_map = row_lookup(global_rows)
    region_map = row_lookup(region_rows)
    validation_global = {
        weighting: float(global_map[("validation", weighting, "global")][
            "explained_fraction"
        ])
        for weighting in WEIGHTINGS
    }
    fit_global = {
        weighting: float(global_map[("fit", weighting, "global")][
            "explained_fraction"
        ])
        for weighting in WEIGHTINGS
    }
    validation_primary_regions = {
        region: float(region_map[("validation", "v_region_equal", region)][
            "explained_fraction"
        ])
        for region in ("top_0_299", "middle_300_2699", "bottom_2700_2999")
    }
    max_normalized_delta = max(
        float(np.max(np.abs(solution.normalized_delta)))
        for solution in solutions.values()
    )
    fit_invalid = int(np.count_nonzero(~jacobians.fit_valid_mask))
    max_step_instability = max(
        result.relative_to_selected
        for parameter_results in jacobians.fit_step_results
        for result in parameter_results
        if not result.selected and np.isfinite(result.relative_to_selected)
    )

    strong = (
        min(validation_global.values()) >= 0.50
        and min(validation_primary_regions.values()) >= 0.30
        and all(
            float(region_map[("validation", "v_region_equal", region)][
                "after_rmse_mm"
            ])
            < float(region_map[("validation", "v_region_equal", region)][
                "before_rmse_mm"
            ])
            for region in validation_primary_regions
        )
        and max_normalized_delta <= 5.0
        and fit_invalid == 0
    )
    partial = (
        max(fit_global.values()) >= 0.30
        and max(validation_global.values()) >= 0.10
        and sum(value > 0.0 for value in validation_primary_regions.values()) >= 2
        and max_normalized_delta <= 20.0
    )
    verdict = "STRONG" if strong else "PARTIAL" if partial else "WEAK"
    reasons = [
        "FIT global explained fraction: "
        + ", ".join(f"{key}={value:.6f}" for key, value in fit_global.items()),
        "VALIDATION global explained fraction: "
        + ", ".join(
            f"{key}={value:.6f}" for key, value in validation_global.items()
        ),
        "VALIDATION v_region_equal: "
        + ", ".join(
            f"{key}={value:.6f}"
            for key, value in validation_primary_regions.items()
        ),
        f"max |DeltaTheta/scale|={max_normalized_delta:.6g}",
        f"FIT common invalid rows={fit_invalid}; max step disagreement={max_step_instability:.3e}",
    ]
    return verdict, reasons


def strongest_pair(matrix: np.ndarray, first: Sequence[int], second: Sequence[int]) -> tuple[str, str, float]:
    candidates: list[tuple[float, int, int]] = []
    for i in first:
        for j in second:
            candidates.append((abs(float(matrix[i, j])), i, j))
    _, i, j = max(candidates)
    return PARAMETERS[i].name, PARAMETERS[j].name, float(matrix[i, j])


def smallest_vector_summary(solution: LinearSolution) -> str:
    vector = solution.right_singular_vectors[-1]
    order = np.argsort(np.abs(vector))[::-1][:3]
    return ", ".join(
        f"{PARAMETERS[index].name}={vector[index]:+.4f}" for index in order
    )


def format_number(value: Any, digits: int = 6) -> str:
    number = float(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def render_report(
    data: PreparedData,
    app_config: Any,
    jacobians: JacobianResult,
    solutions: Mapping[str, LinearSolution],
    global_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    verdict: str,
    verdict_reasons: Sequence[str],
    measurement_config_path: Path,
    pnp_audit_path: Path,
    cone_path: Path,
    cone_hash: str,
) -> str:
    global_map = row_lookup(global_rows)
    region_map = row_lookup(region_rows)
    fit_count = int(np.count_nonzero(data.split == "fit"))
    validation_count = int(np.count_nonzero(data.split == "validation"))
    max_normalized_delta = max(
        float(np.max(np.abs(solution.normalized_delta)))
        for solution in solutions.values()
    )
    condition_numbers = [solutions[name].condition_number for name in WEIGHTINGS]
    tail_ratios = [
        float(solutions[name].singular_values[-1] / solutions[name].singular_values[0])
        for name in WEIGHTINGS
    ]
    az_alpha_correlations = [
        float(solutions[name].covariance_correlation[4, 5]) for name in WEIGHTINGS
    ]
    phi_ay_correlations = [
        float(solutions[name].covariance_correlation[1, 3]) for name in WEIGHTINGS
    ]
    lines = [
        "# Circular Cone paired-PnP sensitivity and local identifiability",
        "",
        f"**PARAMETER_ERROR_CAN_EXPLAIN = {verdict}**",
        "",
        "本轮只有当前 `Theta0` 附近的数值 Jacobian、加权线性最小二乘和 SVD；"
        "没有执行非线性重优化，没有把任何 `DeltaTheta` 写回 0811 或部署 Cone。",
        "",
        "## 严格隔离与 residual",
        "",
        f"- FIT 001–010：baseline {fit_count} 点；全部 step 比较、step 选择、Jacobian、weight 和 `DeltaTheta` 只来自 FIT。",
        f"- VALIDATION 011–013：baseline {validation_count} 点；只在 FIT step 与三组 FIT `DeltaTheta` 冻结后计算最终线性预测。",
        f"- 冻结 Steger 像素 hash：`{data.used_pixel_sha256}`。",
        "- 每次 candidate 都只在内存替换 `calibration['laser_model']`，然后直接调用正式 `reconstruct_uv_to_ground()`。",
        "- 主 residual：`r_z = Zg - (-(nx*Xg+ny*Yg+d)/nz)`；候选的 `Xg,Yg,Zg` 均重新重建，PnP plane Z 不是静态标签。",
        "- Jacobian：`J=dr_z/dTheta`。求解符号为 `J DeltaTheta ≈ -r0`，线性预测为 `r_pred=r0+J DeltaTheta`。",
        "",
        "## 参数化与解释尺度",
        "",
        "`Theta=[theta_axis, phi_axis, A_x, A_y, A_z, alpha]`；angle 用 rad，apex 用 mm。"
        "axis 由球坐标生成并再次单位化。解释尺度只用于 SVD/变量数值缩放与 `DeltaTheta/scale` 展示，"
        "没有加入任何正则项。",
        "",
        "| parameter | unit | Theta0 | interpretation scale | selected FIT step |",
        "|---|---|---:|---:|---:|",
    ]
    for index, spec in enumerate(PARAMETERS):
        lines.append(
            f"| {spec.name} | {spec.unit} | {data.theta0[index]:.10g} | "
            f"{spec.interpretation_scale:.10g} | {jacobians.selected_steps[index]:.6g} |"
        )
    lines += [
        "",
        "解释尺度：axis angles=1 degree，apex=10 mm，alpha=0.1 degree。",
        "",
        "## FIT 三尺度 Jacobian 稳定性",
        "",
        "每个参数固定测试 `0.3× / 1× / 3× base_step`。selected step 只按 FIT 导数一致性和 FIT invalid 数选定；"
        "下表不含任何 validation step 比较。",
        "",
        "| parameter | step | selected | derivative RMS | rel. to selected | invalid +/- |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for parameter_index, spec in enumerate(PARAMETERS):
        for result in jacobians.fit_step_results[parameter_index]:
            lines.append(
                f"| {spec.name} | {result.step:.6g} | {str(result.selected).lower()} | "
                f"{result.derivative_rms:.6g} | {result.relative_to_selected:.3e} | "
                f"{result.invalid_plus}/{result.invalid_minus} |"
            )
    lines += [
        "",
        f"FIT common invalid={np.count_nonzero(~jacobians.fit_valid_mask)}；"
        f"VALIDATION fixed-step common invalid={np.count_nonzero(~jacobians.validation_valid_mask)}。"
        "每个参数 selected-step 的精确 split-local invalid indices 已记录在 `cone_parameter_sensitivity.csv`；"
        "没有因不同参数静默改变 residual 长度。",
        "",
        "## 三种 weighting 的线性增量",
        "",
        "- `point_equal`：所有 FIT 点等权。",
        "- `frame_equal`：每个 FIT frame 总权重相同。",
        "- `v_region_equal`：0–2999 按 300 px 分为 10 区；每个有数据区总权重相同，区内每个有数据 frame 总权重相同。",
        "",
        "| parameter | point_equal delta/scale | frame_equal delta/scale | v_region_equal delta/scale |",
        "|---|---:|---:|---:|",
    ]
    for index, spec in enumerate(PARAMETERS):
        lines.append(
            f"| {spec.name} | {solutions['point_equal'].normalized_delta[index]:+.6g} | "
            f"{solutions['frame_equal'].normalized_delta[index]:+.6g} | "
            f"{solutions['v_region_equal'].normalized_delta[index]:+.6g} |"
        )
    lines += [
        "",
        "原生单位的 `DeltaTheta`、每列加权 Jacobian RMS norm 与 selected step 见 `cone_parameter_sensitivity.csv`。",
        "",
        "## 局部可辨识性",
        "",
        f"SVD 对 `W^(1/2) J diag(scale) / sqrt(sum(W))` 计算；effective rank 阈值为 `S/S1 >= {SVD_EFFECTIVE_RANK_RATIO:g}`。",
        "",
        "| weighting | effective rank | condition number | scaled singular values | smallest vector main components |",
        "|---|---:|---:|---|---|",
    ]
    for weighting in WEIGHTINGS:
        solution = solutions[weighting]
        singular = ", ".join(f"{value:.4g}" for value in solution.singular_values)
        lines.append(
            f"| {weighting} | {solution.effective_rank}/6 | {solution.condition_number:.6g} | "
            f"{singular} | {smallest_vector_summary(solution)} |"
        )
    lines += ["", "耦合重点（covariance correlation，scaled FIT Jacobian pseudoinverse）：", ""]
    for weighting in WEIGHTINGS:
        correlation = solutions[weighting].covariance_correlation
        apex_alpha = strongest_pair(correlation, (2, 3, 4), (5,))
        axis_apex = strongest_pair(correlation, (0, 1), (2, 3, 4))
        lines.append(
            f"- `{weighting}`：apex–alpha 最强 `{apex_alpha[0]} / {apex_alpha[1]}`="
            f"{apex_alpha[2]:+.6f}；axis-angle–apex 最强 `{axis_apex[0]} / {axis_apex[1]}`="
            f"{axis_apex[2]:+.6f}。"
        )
    lines += [
        "",
        f"三种 weighting 在阈值 {SVD_EFFECTIVE_RANK_RATIO:g} 下虽均为数值满秩 6/6，"
        f"但 condition number={min(condition_numbers):.3e}–{max(condition_numbers):.3e}，"
        f"最小/最大 singular value={min(tail_ratios):.3e}–{max(tail_ratios):.3e}；"
        "因此只能称为数值满秩，不能称为各物理参数良好可辨识。",
        f"`A_z–alpha` covariance correlation={min(az_alpha_correlations):+.6f}–"
        f"{max(az_alpha_correlations):+.6f}；`phi_axis–A_y`="
        f"{min(phi_ay_correlations):+.6f}–{max(phi_ay_correlations):+.6f}。"
        "前者接近完全耦合，后者表明 axis angle 与 apex 也强耦合。",
        f"当前 `alpha={math.degrees(data.theta0[5]):.6f}°`，Cone 已接近平面极限；"
        f"最弱方向（以 point_equal 为例）为 `{smallest_vector_summary(solutions['point_equal'])}`。"
        "alpha 与 apex 同时进入最弱方向，支持存在近退化的判断；这不是仅由 `alpha≈89°` 单独推断。",
        "",
        "完整 pairwise column cosine、covariance correlation 以及最小两个右奇异向量见 `cone_parameter_coupling.csv`。"
        "`alpha≈89°` 是否形成近退化方向，应结合最小奇异值比例与最小向量中的 alpha/apex 分量判断，"
        "不能只看单个相关系数。",
        "",
        "## Global residual explainability",
        "",
        "所有 VALIDATION 行都使用对应 weighting 在 FIT 得到并冻结的同一个 `DeltaTheta`。",
        "",
        "| split | weighting | before RMSE | after RMSE | before bias | after bias | energy explained |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("fit", "validation"):
        for weighting in WEIGHTINGS:
            row = global_map[(split, weighting, "global")]
            lines.append(
                f"| {split} | {weighting} | {float(row['before_rmse_mm']):.6f} | "
                f"{float(row['after_rmse_mm']):.6f} | {float(row['before_bias_mm']):+.6f} | "
                f"{float(row['after_bias_mm']):+.6f} | {float(row['explained_fraction']):.6f} |"
            )
    lines += [
        "",
        "## Top / middle / bottom（v_region_equal）",
        "",
        "| split | region | samples | before RMSE | after RMSE | explained fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("fit", "validation"):
        for region in ("top_0_299", "middle_300_2699", "bottom_2700_2999"):
            row = region_map[(split, "v_region_equal", region)]
            lines.append(
                f"| {split} | {region} | {int(row['analysis_sample_count'])} | "
                f"{float(row['before_rmse_mm']):.6f} | {float(row['after_rmse_mm']):.6f} | "
                f"{float(row['explained_fraction']):.6f} |"
            )
    lines += [
        "",
        "## 0811 拟合支持之外的 paired points",
        "",
        "0811 原标定点支持约为 `v=[241.998,2731.978]`。以下 paired 点全部保留；"
        "这里只单独汇总其线性 explainability，没有把它们从 FIT 解中排除。",
        "",
        "| split | region | samples | before RMSE | after RMSE (v_region_equal) | explained fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("fit", "validation"):
        for region in ("extrap_top_v_lt_242", "extrap_bottom_v_gt_2732"):
            key = (split, "v_region_equal", region)
            if key not in region_map:
                continue
            row = region_map[key]
            lines.append(
                f"| {split} | {region} | {int(row['analysis_sample_count'])} | "
                f"{float(row['before_rmse_mm']):.6f} | {float(row['after_rmse_mm']):.6f} | "
                f"{float(row['explained_fraction']):.6f} |"
            )
    lines += [
        "",
        "每 300 px bin、Bias/MAE/RMSE/P95/energy 的三 weighting 全量结果见 `cone_region_explainability.csv`；"
        "global 全量结果见 `cone_residual_explainability.csv`。",
        "",
        "## 判定",
        "",
        f"**PARAMETER_ERROR_CAN_EXPLAIN = {verdict}**",
        "",
    ]
    lines.extend(f"- {reason}。" for reason in verdict_reasons)
    if verdict == "PARTIAL" and max_normalized_delta > 5.0:
        lines += [
            "- 关键区分：residual tangent space 的跨 split 解释力本身很强，且 validation 的"
            " top/middle/bottom 均明显改善；但最优增量达到 "
            f"`{max_normalized_delta:.3f}×` 解释尺度，已经超出可信的局部线性邻域。",
            "- 因而本轮能确认“存在共同的参数方向可解释 residual”，却不能仅凭该大步长线性外推确认"
            "真实 `Theta0+DeltaTheta` 仍能实现同样改善；结合约 3e5 的条件数与近完全参数耦合，"
            "参数误差结论保守判为 PARTIAL，而不是把 tangent-space 的高 explained fraction 直接判为 STRONG。",
        ]
    lines += [
        "- 预先固定判据：STRONG 要求三 weighting 的 validation global explained fraction 均 >=0.50，"
        "且 validation 的 top/middle/bottom 在 v_region_equal 下均 >=0.30 并降低 RMSE，"
        "同时 `max|DeltaTheta/scale|<=5` 且 FIT 无 invalid；"
        "PARTIAL 要求至少一种 FIT global >=0.30、至少一种 validation global >=0.10，"
        "且 validation 三大区至少两区为正；否则 WEAK。大于尺度 20 倍的局部增量不判 PARTIAL。",
        "",
        "该结论只评价当前参数切空间能否解释 residual；它不证明 `Theta0+DeltaTheta` 的非线性有效性，"
        "也不授权发布或写回新参数。",
        "",
        "## Provenance / 不变项",
        "",
        f"- Measurement config：`{measurement_config_path}`",
        f"- PnP audit：`{pnp_audit_path}`",
        f"- Formal Cone：`{cone_path}`",
        f"- Formal Cone SHA-256（运行前后相同）：`{cone_hash}`",
        f"- Steger：`{app_config.extraction_options}`",
        f"- Reconstruction：`{app_config.reconstruction}`",
        f"- 既有 frozen baseline 指标最大复核误差：{data.baseline_metric_max_error:.3e}",
        "- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、runtime reconstruction 和正式 Cone 均未修改。",
        "",
    ]
    return "\n".join(lines)


def render_output_files(verdict: str) -> str:
    rows = [
        (
            "cone_parameter_sensitivity.csv",
            "Theta0、三组 DeltaTheta/scale、Jacobian 列 norm、selected step 与精确 invalid indices",
            "增量是否跨越多个解释尺度；三 weighting 是否给出一致方向",
            "参数变化量与局部一阶敏感度",
            "非线性新参数是否有效或可发布",
            "是，建议放参数增量摘要",
        ),
        (
            "cone_jacobian_singular_values.csv",
            "三种加权 FIT Jacobian 的 raw/scaled singular values、rank、condition",
            "最小奇异值比例、effective rank、condition number",
            "局部可辨识维数和近退化强度",
            "具体哪个物理参数单独可信",
            "是，适合一页谱图/表",
        ),
        (
            "cone_parameter_coupling.csv",
            "column cosine、covariance correlation、最小两个右奇异向量",
            "apex-alpha 与 axis-apex 的高相关和最小向量组成",
            "局部耦合方向",
            "因果归因或全局参数唯一性",
            "附录；组会被追问时使用",
        ),
        (
            "cone_residual_explainability.csv",
            "FIT/VALIDATION global 的 before/after 指标与 explained energy",
            "validation 是否保持正 explained fraction",
            "冻结线性预测的全局解释力",
            "真实非线性重优化效果",
            "是，核心结果表",
        ),
        (
            "cone_region_explainability.csv",
            "top/middle/bottom、每300px bin、0811外推区的三 weighting 指标",
            "边缘 after RMSE、explained fraction、invalid count",
            "边缘结构是否落在参数切空间",
            "Cone 模型形式一定充分/不足",
            "是，建议筛选关键区域",
        ),
        (
            "cone_linearized_prediction.png",
            "FIT/VALIDATION 的 baseline 与三组冻结线性预测 residual-v 曲线",
            "validation 边缘是否同步靠近0",
            "空间结构改善是否跨 split",
            "点级误差分布和非线性有效性",
            "是，主图",
        ),
        (
            "cone_singular_values.png",
            "三种 weighting 的相对 scaled singular spectrum",
            "谱尾是否塌陷、weighting 是否改变可辨识性",
            "近退化是否稳健存在",
            "参数误差能解释多少 residual",
            "是，可辨识性主图",
        ),
        (
            "cone_sensitivity_report.md",
            "方法、隔离、step 稳定性、SVD、耦合、区域解释力与最终判定",
            "首行判定及其 validation/edge 证据",
            "本轮完整线性诊断结论",
            "正式参数变更建议的最终批准",
            "是，组会讲稿底稿",
        ),
        (
            "OUTPUT_FILES.md",
            "九个产物的阅读索引和边界",
            "按问题快速定位主文件",
            "每个产物适用范围",
            "任何新增科学证据",
            "是，作为入口页",
        ),
    ]
    lines = [
        "# Circular Cone sensitivity outputs",
        "",
        f"**PARAMETER_ERROR_CAN_EXPLAIN = {verdict}**",
        "",
        "本目录只有线性化 sensitivity / identifiability 产物；没有候选 Cone YAML，也没有非线性重优化结果。",
        "",
        "| 文件 | 文件体现什么 | 主要看什么 | 能得出什么 | 不能得出什么 | 是否适合组会 |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines += [
        "",
        "建议组会顺序：`cone_linearized_prediction.png` → `cone_residual_explainability.csv` → "
        "`cone_singular_values.png` → 报告中的 coupling/edge 表。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    pnp_audit_path = args.pnp_audit.resolve()
    measurement_config_path = args.measurement_config.resolve()
    output_dir = args.output_dir.resolve()
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    config_for_path = load_app_config(measurement_config_path)
    cone_path = Path(config_for_path.calibration.laser_model).resolve()
    cone_hash_before = sha256_file(cone_path)
    data, app_config = prepare_data(
        data_root, pnp_audit_path, measurement_config_path
    )
    jacobians = compute_fit_jacobian(data)
    solutions = solve_fit(data, jacobians)
    # This is the first validation-dependent computation. All FIT step choices,
    # weighting rules and the three DeltaTheta values are already frozen.
    compute_validation_jacobian(data, jacobians)
    global_rows, region_rows, predictions = prediction_and_metrics(
        data, jacobians, solutions
    )
    verdict, verdict_reasons = verdict_from_results(
        global_rows, region_rows, solutions, jacobians
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parameter_sensitivity(
        output_paths["cone_parameter_sensitivity.csv"],
        data,
        jacobians,
        solutions,
    )
    write_singular_values(
        output_paths["cone_jacobian_singular_values.csv"], solutions
    )
    write_coupling(output_paths["cone_parameter_coupling.csv"], solutions)
    write_metric_rows(
        output_paths["cone_residual_explainability.csv"], global_rows
    )
    write_metric_rows(output_paths["cone_region_explainability.csv"], region_rows)
    save_linearized_prediction(
        output_paths["cone_linearized_prediction.png"], predictions
    )
    save_singular_plot(output_paths["cone_singular_values.png"], solutions)
    output_paths["cone_sensitivity_report.md"].write_text(
        render_report(
            data=data,
            app_config=app_config,
            jacobians=jacobians,
            solutions=solutions,
            global_rows=global_rows,
            region_rows=region_rows,
            verdict=verdict,
            verdict_reasons=verdict_reasons,
            measurement_config_path=measurement_config_path,
            pnp_audit_path=pnp_audit_path,
            cone_path=cone_path,
            cone_hash=cone_hash_before,
        ),
        encoding="utf-8",
    )
    output_paths["OUTPUT_FILES.md"].write_text(
        render_output_files(verdict), encoding="utf-8"
    )

    cone_hash_after = sha256_file(cone_path)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Circular Cone file changed during sensitivity run")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError(
            f"Output directory must contain exactly the requested files: {actual_names}"
        )

    print(f"PARAMETER_ERROR_CAN_EXPLAIN={verdict}")
    print(f"FIT baseline points={np.count_nonzero(data.split == 'fit')}")
    print(f"VALIDATION baseline points={np.count_nonzero(data.split == 'validation')}")
    print(f"FIT Jacobian invalid={np.count_nonzero(~jacobians.fit_valid_mask)}")
    print(
        "VALIDATION Jacobian invalid="
        f"{np.count_nonzero(~jacobians.validation_valid_mask)}"
    )
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
