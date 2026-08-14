#!/usr/bin/env python3
"""FIT-only damped trust-region reoptimization of the deployed Circular Cone.

The measurement residual is evaluated exclusively through the public production
``reconstruct_uv_to_ground`` path.  Candidates remain in memory, validation is
never reconstructed, and the formal calibration artifact is hash-checked before
and after the run.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyze_circular_cone_parameter_sensitivity as sensitivity  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_nonlinear_fit_trust_region"
)
OUTPUT_NAMES = (
    "cone_nonlinear_fit_trace.csv",
    "cone_nonlinear_fit_candidates.csv",
    "cone_nonlinear_fit_metrics.csv",
    "cone_nonlinear_fit_regions.csv",
    "cone_nonlinear_fit_singular_values.csv",
    "cone_nonlinear_fit_coupling.csv",
    "cone_nonlinear_fit_jacobian_stability.csv",
    "cone_candidate_surface_consistency.csv",
    "cone_nonlinear_fit_prediction.png",
    "cone_nonlinear_fit_report.md",
    "OUTPUT_FILES.md",
)

LOWER_BOUNDS = np.asarray(
    [0.0, -math.pi, -1000.0, -1000.0, -500.0, math.radians(60.0)],
    dtype=np.float64,
)
UPPER_BOUNDS = np.asarray(
    [math.pi, math.pi, 1000.0, 1000.0, 500.0, math.radians(89.95)],
    dtype=np.float64,
)
INITIAL_RADIUS = 0.1
MAX_RADIUS = 1.0
MIN_RADIUS = 1.0e-7
ACCEPT_RATIO = 0.10
SHRINK_RATIO = 0.25
EXPAND_RATIO = 0.75
GRADIENT_TOL = 1.0e-9
STEP_TOL = 1.0e-7
RELATIVE_OBJECTIVE_TOL = 1.0e-10
MAX_ACCEPTED_STEPS = 80
MAX_TRIALS = 240
MAX_REJECTIONS_AT_STATE = 30
SVD_RANK_RATIO = 1.0e-6


@dataclass
class JacobianEvaluation:
    jacobian: np.ndarray
    invalid_masks: list[np.ndarray]
    schemes: list[str]


@dataclass
class OptimizationResult:
    weighting: str
    theta: np.ndarray
    residual_mm: np.ndarray
    status: str
    accepted_steps: int
    trial_count: int
    final_radius: float
    trace: list[dict[str, Any]]


@dataclass
class FinalJacobianInfo:
    weighting: str
    jacobian: np.ndarray
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    condition_number: float
    effective_rank: int
    column_cosine: np.ndarray
    covariance_correlation: np.ndarray
    gradient_inf: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FIT-only scaled damped trust-region Circular Cone reoptimization."
    )
    parser.add_argument("--data-root", type=Path, default=sensitivity.paired.DEFAULT_DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=sensitivity.paired.DEFAULT_PNP_AUDIT)
    parser.add_argument(
        "--measurement-config",
        type=Path,
        default=sensitivity.paired.DEFAULT_MEASUREMENT_CONFIG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def bounds_ok(theta: np.ndarray) -> bool:
    values = np.asarray(theta, dtype=np.float64)
    return bool(np.all(values >= LOWER_BOUNDS) and np.all(values <= UPPER_BOUNDS))


def finite_difference_column(
    theta: np.ndarray,
    base_residual: np.ndarray,
    data: sensitivity.PreparedData,
    parameter_index: int,
    step: float,
) -> tuple[sensitivity.StepResult, str]:
    plus_theta = np.asarray(theta, dtype=np.float64).copy()
    minus_theta = np.asarray(theta, dtype=np.float64).copy()
    plus_theta[parameter_index] += step
    minus_theta[parameter_index] -= step
    plus_ok = bounds_ok(plus_theta)
    minus_ok = bounds_ok(minus_theta)
    if plus_ok and minus_ok:
        plus = sensitivity.evaluate_candidate(plus_theta, data, "fit")
        minus = sensitivity.evaluate_candidate(minus_theta, data, "fit")
        return sensitivity.central_derivative(plus, minus, step), "central"

    if not plus_ok and not minus_ok:
        invalid = np.ones(len(base_residual), dtype=bool)
        return (
            sensitivity.StepResult(
                step=step,
                derivative=np.full(len(base_residual), np.nan),
                invalid_mask=invalid,
                invalid_plus=len(base_residual),
                invalid_minus=len(base_residual),
                derivative_rms=float("nan"),
            ),
            "unavailable",
        )

    candidate_theta = plus_theta if plus_ok else minus_theta
    candidate = sensitivity.evaluate_candidate(candidate_theta, data, "fit")
    invalid = candidate.invalid_mask | ~np.isfinite(base_residual)
    derivative = np.full(len(base_residual), np.nan, dtype=np.float64)
    valid = ~invalid
    if plus_ok:
        derivative[valid] = (
            candidate.residual_mm[valid] - base_residual[valid]
        ) / step
        scheme = "forward"
        invalid_plus = int(np.count_nonzero(candidate.invalid_mask))
        invalid_minus = 0
    else:
        derivative[valid] = (
            base_residual[valid] - candidate.residual_mm[valid]
        ) / step
        scheme = "backward"
        invalid_plus = 0
        invalid_minus = int(np.count_nonzero(candidate.invalid_mask))
    rms = (
        float(np.sqrt(np.mean(derivative[valid] ** 2)))
        if np.any(valid)
        else float("nan")
    )
    return (
        sensitivity.StepResult(
            step=step,
            derivative=derivative,
            invalid_mask=invalid,
            invalid_plus=invalid_plus,
            invalid_minus=invalid_minus,
            derivative_rms=rms,
        ),
        scheme,
    )


def evaluate_jacobian(
    theta: np.ndarray,
    data: sensitivity.PreparedData,
    steps: np.ndarray | None = None,
    base_residual: np.ndarray | None = None,
) -> JacobianEvaluation:
    used_steps = (
        np.asarray([spec.base_step for spec in sensitivity.PARAMETERS])
        if steps is None
        else np.asarray(steps, dtype=np.float64)
    )
    count = len(data.residual_mm)
    if base_residual is None:
        base = sensitivity.evaluate_candidate(theta, data, "fit")
        if np.any(base.invalid_mask):
            raise RuntimeError("Jacobian base point contains invalid intersections")
        used_base_residual = base.residual_mm
    else:
        used_base_residual = np.asarray(base_residual, dtype=np.float64)
    jacobian = np.full((count, len(sensitivity.PARAMETERS)), np.nan, dtype=np.float64)
    invalid_masks: list[np.ndarray] = []
    schemes: list[str] = []
    for parameter_index, step in enumerate(used_steps):
        derivative, scheme = finite_difference_column(
            theta,
            used_base_residual,
            data,
            parameter_index,
            float(step),
        )
        jacobian[:, parameter_index] = derivative.derivative
        invalid_masks.append(derivative.invalid_mask)
        schemes.append(scheme)
    return JacobianEvaluation(
        jacobian=jacobian, invalid_masks=invalid_masks, schemes=schemes
    )


def projected_gradient_inf(gradient: np.ndarray, theta: np.ndarray) -> float:
    projected = np.asarray(gradient, dtype=np.float64).copy()
    tolerance = 1.0e-9
    at_lower = theta <= LOWER_BOUNDS + tolerance
    at_upper = theta >= UPPER_BOUNDS - tolerance
    projected[at_lower & (projected > 0.0)] = 0.0
    projected[at_upper & (projected < 0.0)] = 0.0
    return float(np.max(np.abs(projected)))


def weighted_system(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalization = math.sqrt(float(np.sum(weights)))
    sqrt_weight = np.sqrt(weights)
    matrix = (
        sqrt_weight[:, None]
        * jacobian
        * sensitivity.PARAMETER_SCALES[None, :]
        / normalization
    )
    vector = sqrt_weight * residual / normalization
    return matrix, vector


def damped_step(
    matrix: np.ndarray,
    vector: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    projected = u.T @ vector
    cutoff = np.finfo(np.float64).eps * max(matrix.shape) * singular[0]

    def step_at(damping: float) -> np.ndarray:
        factors = np.zeros_like(singular)
        valid = singular > cutoff
        factors[valid] = singular[valid] / (singular[valid] ** 2 + damping)
        return -(vt.T @ (factors * projected))

    full_step = step_at(0.0)
    if float(np.linalg.norm(full_step)) <= radius:
        chosen = full_step
        damping = 0.0
    else:
        lower = 0.0
        upper = max(float(singular[0] ** 2), 1.0e-12)
        while float(np.linalg.norm(step_at(upper))) > radius:
            upper *= 10.0
            if upper > 1.0e30:
                raise RuntimeError("Could not bracket trust-region damping")
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if float(np.linalg.norm(step_at(middle))) > radius:
                lower = middle
            else:
                upper = middle
        damping = upper
        chosen = step_at(damping)
    gradient_inf = float(np.max(np.abs(matrix.T @ vector)))
    return chosen, damping, singular, gradient_inf


def weighted_objective(residual: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * residual**2) / np.sum(weights))


def optimize_weighting(
    weighting: str,
    data: sensitivity.PreparedData,
    initial_jacobian: np.ndarray,
) -> OptimizationResult:
    baseline, v_px, frame_ids = sensitivity.split_arrays(data, "fit")
    weights = sensitivity.weights_for(v_px, frame_ids, weighting)
    theta = data.theta0.copy()
    residual = baseline.copy()
    jacobian = initial_jacobian.copy()
    radius = INITIAL_RADIUS
    accepted_steps = 0
    trial_count = 0
    small_progress_streak = 0
    trace: list[dict[str, Any]] = []
    status = "max_trials"

    while trial_count < MAX_TRIALS and accepted_steps < MAX_ACCEPTED_STEPS:
        if not np.all(np.isfinite(jacobian)):
            status = "jacobian_invalid"
            break
        matrix, vector = weighted_system(jacobian, residual, weights)
        _, _, current_singular, _ = damped_step(matrix, vector, radius)
        gradient_inf = projected_gradient_inf(matrix.T @ vector, theta)
        if gradient_inf <= GRADIENT_TOL:
            status = "converged_gradient"
            break

        rejected_at_state = 0
        accepted_this_state = False
        while trial_count < MAX_TRIALS and rejected_at_state < MAX_REJECTIONS_AT_STATE:
            trial_count += 1
            radius_before = radius
            unprojected_step, damping, singular, _ = damped_step(
                matrix, vector, radius_before
            )
            scaled_lower = (LOWER_BOUNDS - theta) / sensitivity.PARAMETER_SCALES
            scaled_upper = (UPPER_BOUNDS - theta) / sensitivity.PARAMETER_SCALES
            scaled_step = np.clip(unprojected_step, scaled_lower, scaled_upper)
            projected_step = not np.allclose(
                scaled_step, unprojected_step, rtol=0.0, atol=1.0e-14
            )
            step_norm = float(np.linalg.norm(scaled_step))
            raw_step = sensitivity.PARAMETER_SCALES * scaled_step
            trial_theta = theta + raw_step
            trial_theta = np.minimum(np.maximum(trial_theta, LOWER_BOUNDS), UPPER_BOUNDS)
            gradient_inf = projected_gradient_inf(matrix.T @ vector, theta)
            current_objective = weighted_objective(residual, weights)
            linear_residual = residual + jacobian @ raw_step
            predicted_objective = weighted_objective(linear_residual, weights)
            predicted_reduction = current_objective - predicted_objective
            is_in_bounds = bounds_ok(trial_theta)
            invalid_count = 0
            evaluated = False
            trial_objective = float("inf")
            actual_reduction = float("-inf")
            reduction_ratio = float("-inf")
            trial_residual: np.ndarray | None = None
            message = "evaluated"

            if predicted_reduction <= 0.0 or not np.isfinite(predicted_reduction):
                message = "nonpositive_predicted_reduction"
            elif not is_in_bounds:
                message = "bounds_rejected"
            else:
                candidate = sensitivity.evaluate_candidate(trial_theta, data, "fit")
                evaluated = True
                invalid_count = int(np.count_nonzero(candidate.invalid_mask))
                if invalid_count:
                    message = "invalid_intersection_rejected"
                else:
                    trial_residual = candidate.residual_mm
                    trial_objective = weighted_objective(trial_residual, weights)
                    actual_reduction = current_objective - trial_objective
                    reduction_ratio = actual_reduction / predicted_reduction

            accepted = bool(
                trial_residual is not None
                and actual_reduction > 0.0
                and reduction_ratio >= ACCEPT_RATIO
            )
            if reduction_ratio < SHRINK_RATIO or not np.isfinite(reduction_ratio):
                radius = max(MIN_RADIUS, 0.25 * radius_before)
            elif reduction_ratio > EXPAND_RATIO and step_norm >= 0.8 * radius_before:
                radius = min(MAX_RADIUS, 2.0 * radius_before)

            trace.append(
                {
                    "weighting": weighting,
                    "trial_index": trial_count,
                    "accepted_step_before": accepted_steps,
                    "accepted": accepted,
                    "message": "accepted" if accepted else message,
                    "evaluated": evaluated,
                    "radius_before": radius_before,
                    "radius_after": radius,
                    "scaled_step_l2": step_norm,
                    "scaled_step_max_abs": float(np.max(np.abs(scaled_step))),
                    "step_projected_to_bounds": projected_step,
                    "damping_lambda": damping,
                    "gradient_inf": gradient_inf,
                    "condition_number": float(singular[0] / singular[-1]),
                    "current_objective_mse": current_objective,
                    "predicted_objective_mse": predicted_objective,
                    "trial_objective_mse": trial_objective,
                    "current_rmse_mm": math.sqrt(current_objective),
                    "predicted_rmse_mm": math.sqrt(max(predicted_objective, 0.0)),
                    "trial_rmse_mm": math.sqrt(trial_objective)
                    if np.isfinite(trial_objective)
                    else float("nan"),
                    "predicted_reduction": predicted_reduction,
                    "actual_reduction": actual_reduction,
                    "actual_to_predicted_ratio": reduction_ratio,
                    "invalid_count": invalid_count,
                    "bounds_ok": is_in_bounds,
                    "trial_theta_axis_rad": trial_theta[0],
                    "trial_phi_axis_rad": trial_theta[1],
                    "trial_A_x_mm": trial_theta[2],
                    "trial_A_y_mm": trial_theta[3],
                    "trial_A_z_mm": trial_theta[4],
                    "trial_alpha_rad": trial_theta[5],
                }
            )

            if accepted:
                assert trial_residual is not None
                relative_reduction = actual_reduction / max(current_objective, 1.0e-30)
                theta = trial_theta
                residual = trial_residual
                accepted_steps += 1
                accepted_this_state = True
                if relative_reduction <= RELATIVE_OBJECTIVE_TOL:
                    small_progress_streak += 1
                else:
                    small_progress_streak = 0
                if step_norm <= STEP_TOL:
                    status = "converged_active_bounds" if projected_step else "converged_step"
                elif small_progress_streak >= 3:
                    status = "converged_objective"
                else:
                    jacobian_eval = evaluate_jacobian(
                        theta, data, base_residual=residual
                    )
                    if any(np.any(mask) for mask in jacobian_eval.invalid_masks):
                        status = "jacobian_invalid"
                    else:
                        jacobian = jacobian_eval.jacobian
                break

            rejected_at_state += 1
            if radius <= MIN_RADIUS:
                status = "stalled_min_radius"
                break

        if status in {
            "converged_step",
            "converged_active_bounds",
            "converged_objective",
            "jacobian_invalid",
            "stalled_min_radius",
        }:
            break
        if not accepted_this_state:
            status = "stalled_rejections"
            break
    else:
        status = (
            "max_accepted_steps"
            if accepted_steps >= MAX_ACCEPTED_STEPS
            else "max_trials"
        )

    return OptimizationResult(
        weighting=weighting,
        theta=theta,
        residual_mm=residual,
        status=status,
        accepted_steps=accepted_steps,
        trial_count=trial_count,
        final_radius=radius,
        trace=trace,
    )


def final_jacobian_and_stability(
    result: OptimizationResult,
    data: sensitivity.PreparedData,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected_columns: list[np.ndarray] = []
    for parameter_index, spec in enumerate(sensitivity.PARAMETERS):
        derivatives: list[sensitivity.StepResult] = []
        schemes: list[str] = []
        for multiplier in sensitivity.STEP_MULTIPLIERS:
            step = spec.base_step * multiplier
            derivative, scheme = finite_difference_column(
                result.theta,
                result.residual_mm,
                data,
                parameter_index,
                step,
            )
            derivatives.append(derivative)
            schemes.append(scheme)
        selected = derivatives[1]
        if np.any(selected.invalid_mask):
            raise RuntimeError(
                f"Final Jacobian invalid: {result.weighting}/{spec.name}"
            )
        selected_columns.append(selected.derivative)
        for multiplier, derivative, scheme in zip(
            sensitivity.STEP_MULTIPLIERS, derivatives, schemes
        ):
            rows.append(
                {
                    "weighting": result.weighting,
                    "parameter": spec.name,
                    "step_multiplier": multiplier,
                    "step": derivative.step,
                    "selected": multiplier == 1.0,
                    "difference_scheme": scheme,
                    "derivative_rms": derivative.derivative_rms,
                    "relative_to_selected": sensitivity.derivative_relative_rms(
                        derivative, selected
                    ),
                    "invalid_plus": derivative.invalid_plus,
                    "invalid_minus": derivative.invalid_minus,
                }
            )
    return np.column_stack(selected_columns), rows


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.maximum(np.diag(covariance), 0.0)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    correlation = np.zeros_like(covariance)
    valid = denominator > 0.0
    correlation[valid] = covariance[valid] / denominator[valid]
    np.fill_diagonal(correlation, 1.0)
    return correlation


def final_jacobian_info(
    result: OptimizationResult,
    jacobian: np.ndarray,
    data: sensitivity.PreparedData,
) -> FinalJacobianInfo:
    _, v_px, frame_ids = sensitivity.split_arrays(data, "fit")
    weights = sensitivity.weights_for(v_px, frame_ids, result.weighting)
    matrix, vector = weighted_system(jacobian, result.residual_mm, weights)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    column_norm = np.linalg.norm(matrix, axis=0)
    column_cosine = (matrix.T @ matrix) / np.outer(column_norm, column_norm)
    information = matrix.T @ matrix
    covariance = np.linalg.pinv(information, rcond=sensitivity.LSTSQ_RCOND)
    ratio = singular / singular[0]
    return FinalJacobianInfo(
        weighting=result.weighting,
        jacobian=jacobian,
        singular_values=singular,
        right_singular_vectors=vt,
        condition_number=float(singular[0] / singular[-1]),
        effective_rank=int(np.count_nonzero(ratio >= SVD_RANK_RATIO)),
        column_cosine=column_cosine,
        covariance_correlation=correlation_from_covariance(covariance),
        gradient_inf=projected_gradient_inf(matrix.T @ vector, result.theta),
    )


def add_metrics(row: dict[str, Any], prefix: str, metrics: sensitivity.MetricSet) -> None:
    row[f"{prefix}_bias_mm"] = metrics.bias_mm
    row[f"{prefix}_mae_mm"] = metrics.mae_mm
    row[f"{prefix}_rmse_mm"] = metrics.rmse_mm
    row[f"{prefix}_p95_abs_mm"] = metrics.p95_abs_mm
    row[f"{prefix}_residual_energy"] = metrics.residual_energy


def metric_rows(
    results: Mapping[str, OptimizationResult],
    data: sensitivity.PreparedData,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline, v_px, frame_ids = sensitivity.split_arrays(data, "fit")
    global_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for candidate_weighting, result in results.items():
        for metric_weighting in sensitivity.WEIGHTINGS:
            weights = sensitivity.weights_for(v_px, frame_ids, metric_weighting)
            before = sensitivity.calculate_metrics(baseline, weights)
            after = sensitivity.calculate_metrics(result.residual_mm, weights)
            global_row: dict[str, Any] = {
                "candidate_weighting": candidate_weighting,
                "metric_weighting": metric_weighting,
                "region": "global",
                "sample_count": len(baseline),
                "invalid_count": 0,
            }
            add_metrics(global_row, "before", before)
            add_metrics(global_row, "after", after)
            global_row["explained_fraction"] = (
                before.residual_energy - after.residual_energy
            ) / before.residual_energy
            global_rows.append(global_row)

            for region, v_min, v_max in sensitivity.region_definitions()[1:]:
                assert v_min is not None and v_max is not None
                mask = (v_px >= v_min) & (v_px < v_max)
                if not np.any(mask):
                    continue
                region_before = sensitivity.calculate_metrics(
                    baseline[mask], weights[mask]
                )
                region_after = sensitivity.calculate_metrics(
                    result.residual_mm[mask], weights[mask]
                )
                row: dict[str, Any] = {
                    "candidate_weighting": candidate_weighting,
                    "metric_weighting": metric_weighting,
                    "region": region,
                    "v_min_px": v_min,
                    "v_max_px": v_max,
                    "sample_count": int(np.count_nonzero(mask)),
                    "invalid_count": 0,
                }
                add_metrics(row, "before", region_before)
                add_metrics(row, "after", region_after)
                row["explained_fraction"] = (
                    region_before.residual_energy - region_after.residual_energy
                ) / region_before.residual_energy
                region_rows.append(row)
    return global_rows, region_rows


def classify_results(
    results: Mapping[str, OptimizationResult],
    global_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    classifications: dict[str, str] = {}
    for weighting, result in results.items():
        global_row = next(
            row
            for row in global_rows
            if row["candidate_weighting"] == weighting
            and row["metric_weighting"] == weighting
        )
        primary = [
            next(
                row
                for row in region_rows
                if row["candidate_weighting"] == weighting
                and row["metric_weighting"] == weighting
                and row["region"] == region
            )
            for region in ("top_0_299", "middle_300_2699", "bottom_2700_2999")
        ]
        global_explained = float(global_row["explained_fraction"])
        region_explained = [float(row["explained_fraction"]) for row in primary]
        if global_explained >= 0.80 and min(region_explained) >= 0.50:
            classification = "SUCCESS"
        elif global_explained >= 0.30 and min(region_explained) > 0.0:
            classification = "PARTIAL"
        else:
            classification = "FAIL"
        if result.status in {"jacobian_invalid", "stalled_rejections"}:
            classification = "FAIL"
        classifications[weighting] = classification
    return classifications


def surface_consistency_rows(
    results: Mapping[str, OptimizationResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(sensitivity.WEIGHTINGS, 2):
        difference = results[first].residual_mm - results[second].residual_mm
        parameter_distance = (
            results[first].theta - results[second].theta
        ) / sensitivity.PARAMETER_SCALES
        rows.append(
            {
                "candidate_a": first,
                "candidate_b": second,
                "surface_residual_difference_bias_mm": float(np.mean(difference)),
                "surface_residual_difference_rms_mm": float(
                    np.sqrt(np.mean(difference**2))
                ),
                "surface_residual_difference_p95_abs_mm": float(
                    np.percentile(np.abs(difference), 95.0)
                ),
                "surface_residual_difference_max_abs_mm": float(
                    np.max(np.abs(difference))
                ),
                "parameter_distance_scaled_l2": float(
                    np.linalg.norm(parameter_distance)
                ),
                "parameter_distance_scaled_max_abs": float(
                    np.max(np.abs(parameter_distance))
                ),
            }
        )
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: sensitivity.csv_value(row[field]) for field in fields}
            )


def candidate_rows(
    results: Mapping[str, OptimizationResult], data: sensitivity.PreparedData
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(sensitivity.PARAMETERS):
        row: dict[str, Any] = {
            "parameter": spec.name,
            "unit": spec.unit,
            "current_value": data.theta0[index],
            "interpretation_scale": spec.interpretation_scale,
        }
        for weighting in sensitivity.WEIGHTINGS:
            final = results[weighting].theta[index]
            row[f"final_{weighting}"] = final
            row[f"delta_{weighting}"] = final - data.theta0[index]
            row[f"normalized_delta_{weighting}"] = (
                final - data.theta0[index]
            ) / spec.interpretation_scale
        rows.append(row)
    return rows


def singular_rows(
    infos: Mapping[str, FinalJacobianInfo]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weighting in sensitivity.WEIGHTINGS:
        info = infos[weighting]
        for index, value in enumerate(info.singular_values):
            relative = float(value / info.singular_values[0])
            rows.append(
                {
                    "weighting": weighting,
                    "singular_index": index + 1,
                    "scaled_singular_value": value,
                    "relative_singular_value": relative,
                    "effective": relative >= SVD_RANK_RATIO,
                    "effective_rank": info.effective_rank,
                    "condition_number": info.condition_number,
                    "rank_ratio_threshold": SVD_RANK_RATIO,
                    "gradient_inf": info.gradient_inf,
                }
            )
    return rows


def coupling_rows(
    infos: Mapping[str, FinalJacobianInfo]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weighting in sensitivity.WEIGHTINGS:
        info = infos[weighting]
        for kind, matrix in (
            ("column_cosine", info.column_cosine),
            ("covariance_correlation", info.covariance_correlation),
        ):
            for first, second in itertools.combinations(range(len(sensitivity.PARAMETERS)), 2):
                rows.append(
                    {
                        "weighting": weighting,
                        "kind": kind,
                        "vector_rank_from_smallest": "",
                        "parameter_a": sensitivity.PARAMETERS[first].name,
                        "parameter_b": sensitivity.PARAMETERS[second].name,
                        "value": matrix[first, second],
                        "absolute_value": abs(float(matrix[first, second])),
                    }
                )
        for vector_rank in (1, 2):
            vector = info.right_singular_vectors[-vector_rank]
            for parameter_index, component in enumerate(vector):
                rows.append(
                    {
                        "weighting": weighting,
                        "kind": "right_singular_vector",
                        "vector_rank_from_smallest": vector_rank,
                        "parameter_a": sensitivity.PARAMETERS[parameter_index].name,
                        "parameter_b": "",
                        "value": component,
                        "absolute_value": abs(float(component)),
                    }
                )
    return rows


def binned_median(v_px: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.floor(v_px / 30.0).astype(int)
    centers: list[float] = []
    values: list[float] = []
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        centers.append((float(bin_id) + 0.5) * 30.0)
        values.append(float(np.median(residual[mask])))
    return np.asarray(centers), np.asarray(values)


def save_prediction_plot(
    path: Path,
    results: Mapping[str, OptimizationResult],
    data: sensitivity.PreparedData,
    region_rows: Sequence[Mapping[str, Any]],
) -> None:
    baseline, v_px, _ = sensitivity.split_arrays(data, "fit")
    colors = {
        "point_equal": "#2b6cb0",
        "frame_equal": "#2f855a",
        "v_region_equal": "#c05621",
    }
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 10.5))
    x, values = binned_median(v_px, baseline)
    axes[0].plot(x, values, color="#222222", linewidth=2.2, label="baseline")
    for weighting in sensitivity.WEIGHTINGS:
        x, values = binned_median(v_px, results[weighting].residual_mm)
        axes[0].plot(
            x,
            values,
            color=colors[weighting],
            linewidth=1.7,
            label=weighting,
        )
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].axvline(300.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axes[0].axvline(2700.0, color="#aaaaaa", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("median vertical residual / mm")
    axes[0].set_title("FIT exact residual after damped relinearized optimization")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)

    for weighting in sensitivity.WEIGHTINGS:
        accepted = [row for row in results[weighting].trace if bool(row["accepted"])]
        x_steps = np.arange(1, len(accepted) + 1)
        y_rmse = [float(row["trial_rmse_mm"]) for row in accepted]
        axes[1].plot(
            x_steps,
            y_rmse,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=colors[weighting],
            label=weighting,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("accepted trust-region step")
    axes[1].set_ylabel("matching weighted RMSE / mm (log)")
    axes[1].set_title("Exact FIT convergence")
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=3, fontsize=8)

    primary_regions = ("top_0_299", "middle_300_2699", "bottom_2700_2999")
    positions = np.arange(len(primary_regions))
    width = 0.24
    for offset, weighting in enumerate(sensitivity.WEIGHTINGS):
        selected = [
            next(
                row
                for row in region_rows
                if row["candidate_weighting"] == weighting
                and row["metric_weighting"] == weighting
                and row["region"] == region
            )
            for region in primary_regions
        ]
        axes[2].bar(
            positions + (offset - 1) * width,
            [float(row["after_rmse_mm"]) for row in selected],
            width=width,
            color=colors[weighting],
            label=weighting,
        )
    baseline_regions = [
        next(
            row
            for row in region_rows
            if row["candidate_weighting"] == "point_equal"
            and row["metric_weighting"] == "point_equal"
            and row["region"] == region
        )
        for region in primary_regions
    ]
    axes[2].scatter(
        positions,
        [float(row["before_rmse_mm"]) for row in baseline_regions],
        color="#111111",
        marker="x",
        s=60,
        linewidth=2,
        label="baseline (point equal)",
        zorder=4,
    )
    axes[2].set_xticks(positions, ["top", "middle", "bottom"])
    axes[2].set_ylabel("regional RMSE / mm")
    axes[2].set_title("Final broad-region FIT error")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def strongest_pair(
    matrix: np.ndarray, first: Sequence[int], second: Sequence[int]
) -> tuple[str, str, float]:
    candidates = [
        (abs(float(matrix[i, j])), i, j) for i in first for j in second
    ]
    _, i, j = max(candidates)
    return (
        sensitivity.PARAMETERS[i].name,
        sensitivity.PARAMETERS[j].name,
        float(matrix[i, j]),
    )


def render_report(
    data: sensitivity.PreparedData,
    results: Mapping[str, OptimizationResult],
    infos: Mapping[str, FinalJacobianInfo],
    classifications: Mapping[str, str],
    global_rows: Sequence[Mapping[str, Any]],
    region_rows: Sequence[Mapping[str, Any]],
    consistency: Sequence[Mapping[str, Any]],
    cone_path: Path,
    cone_hash: str,
) -> str:
    overall = classifications["v_region_equal"]
    parameter_convergence = (
        "CONVERGED"
        if all(
            result.status
            in {
                "converged_gradient",
                "converged_step",
                "converged_objective",
                "converged_active_bounds",
            }
            for result in results.values()
        )
        else "UNRESOLVED_BOUNDARY"
    )
    lines = [
        "# Circular Cone FIT-only damped trust-region reoptimization",
        "",
        f"**FIT_SURFACE_RESULT = {overall}**  ",
        f"**PARAMETER_CONVERGENCE = {parameter_convergence}**",
        "",
        "本步骤只优化 FIT 001–010。VALIDATION 011–013 未打开、未重建、未评分。"
        "所有 candidate 仅在内存和本实验 CSV 中存在，没有写出或覆盖任何正式 Cone YAML。",
        "",
        "## 方法与隔离",
        "",
        "- objective：`mean_w(r_z^2)`，其中每次 candidate 都通过正式 `reconstruct_uv_to_ground()`"
        " 重建，并在新的 `(Xg,Yg)` 上重新评价 paired PnP plane。",
        "- 三种 weighting 独立从同一个正式 `Theta0` 启动；`v_region_equal` 是预先指定的主分支。",
        f"- scaled trust radius：初值 {INITIAL_RADIUS:g}，范围 [{MIN_RADIUS:g},{MAX_RADIUS:g}]；"
        "LM damping 只用于 trust-region step，不进入最终 objective。",
        f"- 接受阈值 actual/predicted >= {ACCEPT_RATIO:g}；invalid 或越界 trial 直接拒绝；"
        "每次接受后重算 Jacobian。",
        "- 主结果使用 L2；没有 robust loss、参数先验或正则化。",
        f"- FIT frozen-pixel SHA-256：`{data.used_pixel_sha256}`。",
        "",
        "## Optimizer outcome",
        "",
        "| weighting | classification | status | accepted/trials | final radius | final gradient inf |",
        "|---|---|---|---:|---:|---:|",
    ]
    for weighting in sensitivity.WEIGHTINGS:
        result = results[weighting]
        info = infos[weighting]
        lines.append(
            f"| {weighting} | {classifications[weighting]} | {result.status} | "
            f"{result.accepted_steps}/{result.trial_count} | {result.final_radius:.3e} | "
            f"{info.gradient_inf:.3e} |"
        )

    lines += [
        "",
        "## Matching-weight global exact residual",
        "",
        "| weighting | before RMSE | after RMSE | before P95 | after P95 | energy explained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weighting in sensitivity.WEIGHTINGS:
        row = next(
            row
            for row in global_rows
            if row["candidate_weighting"] == weighting
            and row["metric_weighting"] == weighting
        )
        lines.append(
            f"| {weighting} | {float(row['before_rmse_mm']):.6f} | "
            f"{float(row['after_rmse_mm']):.6f} | {float(row['before_p95_abs_mm']):.6f} | "
            f"{float(row['after_p95_abs_mm']):.6f} | {float(row['explained_fraction']):.6f} |"
        )

    lines += [
        "",
        "## Matching-weight top / middle / bottom",
        "",
        "| weighting | region | before RMSE | after RMSE | energy explained |",
        "|---|---|---:|---:|---:|",
    ]
    for weighting in sensitivity.WEIGHTINGS:
        for region in ("top_0_299", "middle_300_2699", "bottom_2700_2999"):
            row = next(
                row
                for row in region_rows
                if row["candidate_weighting"] == weighting
                and row["metric_weighting"] == weighting
                and row["region"] == region
            )
            lines.append(
                f"| {weighting} | {region} | {float(row['before_rmse_mm']):.6f} | "
                f"{float(row['after_rmse_mm']):.6f} | {float(row['explained_fraction']):.6f} |"
            )

    lines += [
        "",
        "## Final parameter displacement",
        "",
        "表内为 `(Theta_final-Theta0)/interpretation_scale`；不同分支的弱参数不能直接平均。",
        "",
        "| parameter | point_equal | frame_equal | v_region_equal |",
        "|---|---:|---:|---:|",
    ]
    for index, spec in enumerate(sensitivity.PARAMETERS):
        values = [
            (results[weighting].theta[index] - data.theta0[index])
            / spec.interpretation_scale
            for weighting in sensitivity.WEIGHTINGS
        ]
        lines.append(
            f"| {spec.name} | {values[0]:+.6g} | {values[1]:+.6g} | {values[2]:+.6g} |"
        )

    lines += [
        "",
        "## Final local identifiability",
        "",
        "| weighting | effective rank | condition number | smallest/large singular ratio | strongest apex-alpha | strongest axis-apex |",
        "|---|---:|---:|---:|---|---|",
    ]
    for weighting in sensitivity.WEIGHTINGS:
        info = infos[weighting]
        apex_alpha = strongest_pair(info.covariance_correlation, (2, 3, 4), (5,))
        axis_apex = strongest_pair(info.covariance_correlation, (0, 1), (2, 3, 4))
        lines.append(
            f"| {weighting} | {info.effective_rank}/6 | {info.condition_number:.6g} | "
            f"{info.singular_values[-1]/info.singular_values[0]:.3e} | "
            f"{apex_alpha[0]}/{apex_alpha[1]}={apex_alpha[2]:+.6f} | "
            f"{axis_apex[0]}/{axis_apex[1]}={axis_apex[2]:+.6f} |"
        )

    lines += [
        "",
        "## Candidate surface consistency",
        "",
        "| candidate pair | residual-surface RMS difference | P95 | scaled parameter distance L2 |",
        "|---|---:|---:|---:|",
    ]
    for row in consistency:
        lines.append(
            f"| {row['candidate_a']} / {row['candidate_b']} | "
            f"{float(row['surface_residual_difference_rms_mm']):.6f} | "
            f"{float(row['surface_residual_difference_p95_abs_mm']):.6f} | "
            f"{float(row['parameter_distance_scaled_l2']):.6g} |"
        )

    lines += [
        "",
        "## FIT decision gate",
        "",
        f"**FIT_CONE_RESULT = {overall}**",
        "",
        "- FIT_SURFACE SUCCESS：matching global explained >=0.80 且 top/middle/bottom 各 >=0.50。",
        "- PARTIAL：matching global explained >=0.30 且三个大区均为正改善。",
        "- 其他情况为 FAIL；`v_region_equal` 是总体判定的预注册主分支。",
    ]
    if overall == "SUCCESS":
        lines += [
            "- FIT 表面目标上，Circular Cone 的真实非线性调整已经同时解释全局和上下边缘的主要 residual。",
            "- 但本轮三个分支均为 max_accepted_steps，参数沿 apex/alpha 近退化谷漂移并触及 A_z 上界；"
            "因此 `PARAMETER_CONVERGENCE=UNRESOLVED_BOUNDARY`，不能把 candidate 当作已收敛参数。",
            "- 这使“0811 参数/objective 不匹配”成为更强假设，但还不能查看 validation 后直接发布；"
            "下一步应先做 FIT frame jackknife 与弱方向 profile。",
        ]
    elif overall == "PARTIAL":
        lines += [
            "- FIT 上存在真实、可累计的参数改善，但仍留下明显区域结构；进入 validation 前应先做"
            "弱奇异方向 profile，并检查是否需要同协议 quadratic graph 对照。",
        ]
    else:
        lines += [
            "- FIT-only 的受控非线性优化仍不能同时改善全局和边缘；不应继续消耗 validation，"
            "下一步应转入同协议模型形式比较。",
        ]
    lines += [
        "- 按计划在 FIT 收敛后、读取 VALIDATION 前停止。",
        "",
        "## Provenance / 不变项",
        "",
        f"- Formal Cone：`{cone_path}`",
        f"- Formal Cone SHA-256（运行前后相同）：`{cone_hash}`",
        f"- Frozen baseline 最大复核误差：{data.baseline_metric_max_error:.3e}",
        "- camera intrinsics、distortion、Steger pixels、ground extrinsics、paired PnP poses、"
        "runtime reconstruction 与正式 Cone 均未修改。",
        "",
    ]
    return "\n".join(lines)


def render_output_files(overall: str, parameter_convergence: str) -> str:
    rows = [
        ("cone_nonlinear_fit_trace.csv", "每个 exact trial 的半径、阻尼、预测/实际下降、接受状态", "检查 optimizer 是否真实收敛", "不含 validation"),
        ("cone_nonlinear_fit_candidates.csv", "Theta0 与三组实验 candidate/delta/scale", "查看参数移动和 weighting 差异", "不是可部署 YAML"),
        ("cone_nonlinear_fit_metrics.csv", "三 candidate × 三 metric weighting 的 global 指标", "查看全局改善及交叉稳健性", "不能证明 validation"),
        ("cone_nonlinear_fit_regions.csv", "top/middle/bottom、300px bins、外推区指标", "查看边缘是否改善", "仅 FIT"),
        ("cone_nonlinear_fit_singular_values.csv", "最终 scaled Jacobian singular spectrum", "查看最终条件数和有效秩", "不能单独解释物理参数"),
        ("cone_nonlinear_fit_coupling.csv", "最终 column cosine、covariance correlation、弱向量", "查看 apex-alpha/axis-apex 耦合", "不是参数置信区间"),
        ("cone_nonlinear_fit_jacobian_stability.csv", "最终点三尺度 finite-difference 复核", "确认最终 Jacobian 数值稳定", "不是全局模型验证"),
        ("cone_candidate_surface_consistency.csv", "不同 weighting candidates 的 FIT surface residual 差异", "区分表面稳定与参数漂移", "不含新姿态"),
        ("cone_nonlinear_fit_prediction.png", "残差-v、收敛轨迹、区域 RMSE", "组会主图", "不含 validation"),
        ("cone_nonlinear_fit_report.md", "方法、结果、辨识性与 FIT 决策门", "本阶段主报告", "不授权写回正式参数"),
        ("OUTPUT_FILES.md", "输出索引", "快速导航", "不增加证据"),
    ]
    lines = [
        "# Circular Cone FIT nonlinear outputs",
        "",
        f"**FIT_SURFACE_RESULT = {overall}**  ",
        f"**PARAMETER_CONVERGENCE = {parameter_convergence}**",
        "",
        "| 文件 | 文件体现什么 | 主要看什么 | 不能得出什么 |",
        "|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    pnp_audit = args.pnp_audit.resolve()
    measurement_config = args.measurement_config.resolve()
    output_dir = args.output_dir.resolve()
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    app_config = sensitivity.load_app_config(measurement_config)
    cone_path = Path(app_config.calibration.laser_model).resolve()
    cone_hash_before = sensitivity.sha256_file(cone_path)
    data, _ = sensitivity.prepare_data(
        data_root,
        pnp_audit,
        measurement_config,
        include_splits=("fit",),
    )
    expected_frames = {f"{index:03d}" for index in range(1, 11)}
    observed_frames = {frame.frame_id for frame in data.frames}
    if observed_frames != expected_frames or set(data.split) != {"fit"}:
        raise RuntimeError(
            f"FIT-only isolation failed: frames={sorted(observed_frames)}, splits={sorted(set(data.split))}"
        )

    initial_jacobian_eval = evaluate_jacobian(
        data.theta0, data, base_residual=data.residual_mm
    )
    if any(np.any(mask) for mask in initial_jacobian_eval.invalid_masks):
        raise RuntimeError("Initial FIT Jacobian contains invalid intersections")

    results: dict[str, OptimizationResult] = {}
    for weighting in sensitivity.WEIGHTINGS:
        result = optimize_weighting(
            weighting, data, initial_jacobian_eval.jacobian
        )
        results[weighting] = result
        _, fit_v, fit_frames = sensitivity.split_arrays(data, "fit")
        fit_weights = sensitivity.weights_for(fit_v, fit_frames, weighting)
        final_rmse = math.sqrt(weighted_objective(result.residual_mm, fit_weights))
        print(
            f"OPTIMIZED {weighting}: status={result.status}, "
            f"accepted={result.accepted_steps}, trials={result.trial_count}, "
            f"rmse={final_rmse:.9g}, "
            f"theta_delta_scaled_l2={np.linalg.norm((result.theta-data.theta0)/sensitivity.PARAMETER_SCALES):.6g}, "
            f"theta={np.array2string(result.theta, precision=10, separator=',')}",
            flush=True,
        )

    stability_rows: list[dict[str, Any]] = []
    infos: dict[str, FinalJacobianInfo] = {}
    for weighting in sensitivity.WEIGHTINGS:
        jacobian, rows = final_jacobian_and_stability(results[weighting], data)
        stability_rows.extend(rows)
        infos[weighting] = final_jacobian_info(results[weighting], jacobian, data)

    global_rows, region_rows = metric_rows(results, data)
    classifications = classify_results(results, global_rows, region_rows)
    consistency = surface_consistency_rows(results)
    overall = classifications["v_region_equal"]
    parameter_convergence = (
        "CONVERGED"
        if all(
            result.status
            in {
                "converged_gradient",
                "converged_step",
                "converged_objective",
                "converged_active_bounds",
            }
            for result in results.values()
        )
        else "UNRESOLVED_BOUNDARY"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        output_paths["cone_nonlinear_fit_trace.csv"],
        [row for weighting in sensitivity.WEIGHTINGS for row in results[weighting].trace],
    )
    write_rows(
        output_paths["cone_nonlinear_fit_candidates.csv"], candidate_rows(results, data)
    )
    write_rows(output_paths["cone_nonlinear_fit_metrics.csv"], global_rows)
    write_rows(output_paths["cone_nonlinear_fit_regions.csv"], region_rows)
    write_rows(
        output_paths["cone_nonlinear_fit_singular_values.csv"], singular_rows(infos)
    )
    write_rows(
        output_paths["cone_nonlinear_fit_coupling.csv"], coupling_rows(infos)
    )
    write_rows(
        output_paths["cone_nonlinear_fit_jacobian_stability.csv"], stability_rows
    )
    write_rows(
        output_paths["cone_candidate_surface_consistency.csv"], consistency
    )
    save_prediction_plot(
        output_paths["cone_nonlinear_fit_prediction.png"], results, data, region_rows
    )
    output_paths["cone_nonlinear_fit_report.md"].write_text(
        render_report(
            data,
            results,
            infos,
            classifications,
            global_rows,
            region_rows,
            consistency,
            cone_path,
            cone_hash_before,
        ),
        encoding="utf-8",
    )
    output_paths["OUTPUT_FILES.md"].write_text(
        render_output_files(overall, parameter_convergence), encoding="utf-8"
    )

    cone_hash_after = sensitivity.sha256_file(cone_path)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Circular Cone changed during FIT optimization")
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output files: {sorted(actual_names)}")

    print(f"FIT_SURFACE_RESULT={overall}")
    print(f"PARAMETER_CONVERGENCE={parameter_convergence}")
    for weighting in sensitivity.WEIGHTINGS:
        row = next(
            row
            for row in global_rows
            if row["candidate_weighting"] == weighting
            and row["metric_weighting"] == weighting
        )
        print(
            f"{weighting}: class={classifications[weighting]}, "
            f"rmse={float(row['after_rmse_mm']):.9g}, "
            f"explained={float(row['explained_fraction']):.9g}, "
            f"condition={infos[weighting].condition_number:.9g}"
        )
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
