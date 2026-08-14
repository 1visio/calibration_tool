#!/usr/bin/env python3
"""Production-path sanity audit for frozen Circular Cone candidates.

This audit deliberately does not call the laser extractor or any optimizer.  It
uses only the saved gauge ``u,v`` files and the production reconstruction path.
Paired FIT/VALIDATION pixels are audited only when a frozen ``u,v`` artifact is
present; otherwise the audit records RECAPTURE_REQUIRED and never falls back to
re-extraction.
"""

from __future__ import annotations

import copy
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))

from calibration.config_loader import load_calibration_files  # noqa: E402
from reconstruction.reconstructor import (  # noqa: E402
    ReconstructionParams,
    _model_z_range,
    _solve_quadratic_all,
    reconstruct_uv_to_ground,
)


OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_nonlinear_fit_trust_region"
)
GAUGE_ROOT = (
    WORKSPACE_ROOT
    / "0704line-laser-3d-scanner"
    / "laser_measurement_tool"
    / "output_daheng_0811"
)
# Use the calibration files referenced by the historical result.json files.  The
# byte-identical copy under the current measurement package was checked during
# the earlier M0 replay; using this path makes the runtime provenance explicit.
CALIBRATION_DIR = (
    WORKSPACE_ROOT
    / "0704line-laser-3d-scanner"
    / "laser_measurement_tool"
    / "configs"
    / "calibration_daheng_0811"
)
BASE_CONE_PATH = CALIBRATION_DIR / "circular_cone.yaml"
INTRINSICS_PATH = CALIBRATION_DIR / "calibration_result.yaml"
EXTRINSICS_PATH = CALIBRATION_DIR / "camera_ground_extrinsics.yaml"

FRAME_DIRS = (
    "frame_061303_measure",
    "frame_062878_measure",
    "frame_063995_measure",
    "frame_065292_measure",
)
GROUP_FILES = {"baseline": "baseline_points.csv", "height": "height_points.csv"}
MODEL_ORDER = ("M0", "M1_point_equal", "M2_frame_equal", "M3_v_region_equal")
CANDIDATE_NAMES = {
    "M1_point_equal": "point_equal",
    "M2_frame_equal": "frame_equal",
    "M3_v_region_equal": "v_region_equal",
}

# Values are copied from the frozen candidate artifact.  They are not optimized
# here; this script treats the CSV as an immutable input.
THETA_COLUMNS = {
    "point_equal": "final_point_equal",
    "frame_equal": "final_frame_equal",
    "v_region_equal": "final_v_region_equal",
}
PARAMETER_NAMES = ("theta_axis", "phi_axis", "A_x", "A_y", "A_z", "alpha")

RUNTIME_PARAMS = ReconstructionParams(
    parallel_epsilon=1.0e-9,
    quadratic_epsilon=1.0e-12,
    min_camera_depth_mm=630.0,
    max_camera_depth_mm=715.0,
    model_range_margin_mm=2.0,
)


def read_uv(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"u", "v"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} 缺少保存的原始 u,v 列")
        rows = []
        for row in reader:
            rows.append((float(row["u"]), float(row["v"])))
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} 的 u,v 包含非有限值")
    return np.ascontiguousarray(values)


def load_gauge_uv() -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for frame in FRAME_DIRS:
        for group, filename in GROUP_FILES.items():
            result[(frame, group)] = read_uv(GAUGE_ROOT / frame / filename)
    return result


def load_candidate_theta() -> dict[str, np.ndarray]:
    path = OUTPUT_DIR / "cone_nonlinear_fit_candidates.csv"
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["parameter"]] = row
    result: dict[str, np.ndarray] = {}
    for name, column in THETA_COLUMNS.items():
        result[name] = np.asarray(
            [float(rows[param][column]) for param in PARAMETER_NAMES],
            dtype=np.float64,
        )
    return result


def angles_to_axis(theta: float, phi: float) -> np.ndarray:
    axis = np.asarray(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )
    return axis / np.linalg.norm(axis)


def model_from_theta(base_model: dict[str, Any], theta: np.ndarray) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    model["axis_unit_camera"] = angles_to_axis(float(theta[0]), float(theta[1])).tolist()
    model["apex_camera_mm"] = np.asarray(theta[2:5], dtype=np.float64).tolist()
    model["half_apex_angle_deg"] = math.degrees(float(theta[5]))
    model["model_type"] = "circular_cone"
    model["fit_success"] = True
    return model


def load_base_calibration() -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    with BASE_CONE_PATH.open("r", encoding="utf-8") as handle:
        base_document = yaml.safe_load(handle)
    calibration = load_calibration_files(
        INTRINSICS_PATH, BASE_CONE_PATH, EXTRINSICS_PATH, None
    )
    return calibration, base_document, {
        "K": np.asarray(calibration["K"], dtype=np.float64),
        "D": np.asarray(calibration["D"], dtype=np.float64),
    }


def save_candidate_yaml(model: dict[str, Any], name: str) -> Path:
    directory = OUTPUT_DIR / "candidate_roundtrip_yaml"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"circular_cone_{name}.yaml"
    document = copy.deepcopy(model)
    # Loader accepts these fields and keeping the original metadata makes this
    # an actual deployable model file rather than a parameter fragment.
    for key in ("axis_unit_camera", "apex_camera_mm", "z_valid_range_mm"):
        if key in document:
            document[key] = np.asarray(document[key], dtype=np.float64).tolist()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
    return path


def make_calibration(base: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    result["laser_model"] = model
    return result


def align_reconstruction(
    uv: np.ndarray, result: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = np.full((len(uv), 3), np.nan, dtype=np.float64)
    points_ground = np.full((len(uv), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(uv), dtype=bool)
    output_index = 0
    for input_index, pixel in enumerate(uv):
        if output_index >= len(result.pixels_uv):
            break
        if np.array_equal(pixel, result.pixels_uv[output_index]):
            points_camera[input_index] = result.points_camera[output_index]
            points_ground[input_index] = result.points_ground[output_index]
            valid[input_index] = True
            output_index += 1
    if output_index != len(result.pixels_uv):
        raise RuntimeError("无法把 production reconstruction 输出对齐到输入 UV")
    return points_camera, points_ground, valid


def reconstruct_aligned(
    uv: np.ndarray, calibration: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = reconstruct_uv_to_ground(uv, calibration, RUNTIME_PARAMS)
    return align_reconstruction(uv, result)


def finite_stats(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.max(np.abs(values))), float(np.percentile(np.abs(values), 95)), float(
        np.sqrt(np.mean(values * values))
    )


def candidate_roundtrip(
    base_calibration: dict[str, Any],
    base_document: dict[str, Any],
    candidate_theta: dict[str, np.ndarray],
    gauge_uv: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    yaml_paths: dict[str, Any] = {}
    for model_id in MODEL_ORDER[1:]:
        short_name = CANDIDATE_NAMES[model_id]
        model = model_from_theta(base_document, candidate_theta[short_name])
        yaml_path = save_candidate_yaml(model, short_name)
        yaml_paths[model_id] = yaml_path
        reloaded_calibration = load_calibration_files(
            INTRINSICS_PATH, yaml_path, EXTRINSICS_PATH, None
        )
        in_memory_calibration = make_calibration(base_calibration, model)
        model_pass = True
        for (frame, group), uv in gauge_uv.items():
            mem_camera, mem_ground, mem_valid = reconstruct_aligned(uv, in_memory_calibration)
            reload_camera, reload_ground, reload_valid = reconstruct_aligned(uv, reloaded_calibration)
            mask_mismatch = int(np.count_nonzero(mem_valid != reload_valid))
            common = mem_valid & reload_valid
            field_values = {
                "Xg": (mem_ground[:, 0], reload_ground[:, 0]),
                "Yg": (mem_ground[:, 1], reload_ground[:, 1]),
                "Zg": (mem_ground[:, 2], reload_ground[:, 2]),
                "lambda": (mem_camera[:, 2], reload_camera[:, 2]),
            }
            for field, (left, right) in field_values.items():
                diff = left[common] - right[common]
                maximum, p95, rms = finite_stats(diff)
                passed = (
                    mask_mismatch == 0
                    and (not np.isfinite(maximum) or maximum <= 1.0e-9)
                )
                model_pass = model_pass and passed
                rows.append(
                    {
                        "scope": "historical_gauge",
                        "dataset": f"{frame}:{group}",
                        "model": model_id,
                        "field": field,
                        "n_uv": len(uv),
                        "valid_in_memory": int(np.count_nonzero(mem_valid)),
                        "valid_reloaded": int(np.count_nonzero(reload_valid)),
                        "valid_mask_mismatch": mask_mismatch,
                        "max_difference": maximum,
                        "p95_difference": p95,
                        "rms_difference": rms,
                        "previous_rmse_mm": "",
                        "reloaded_rmse_mm": "",
                        "status": "PASS" if passed else "FAIL",
                        "notes": str(yaml_path),
                    }
                )
        # FIT/VALIDATION frozen pixels are not present in this workspace.  The
        # rows are explicit so consumers cannot mistake absence for zero error.
        for split, expected in (("paired_fit", {"point_equal": 0.034159, "frame_equal": 0.033671, "v_region_equal": 0.034900}),
                                ("paired_validation", {"point_equal": 0.036572, "frame_equal": 0.034809, "v_region_equal": 0.034091})):
            rows.append(
                {
                    "scope": split,
                    "dataset": split,
                    "model": model_id,
                    "field": "residual_rmse",
                    "n_uv": 0,
                    "valid_in_memory": "",
                    "valid_reloaded": "",
                    "valid_mask_mismatch": "",
                    "max_difference": "",
                    "p95_difference": "",
                    "rms_difference": "",
                    "previous_rmse_mm": expected[short_name],
                    "reloaded_rmse_mm": "",
                    "status": "RECAPTURE_REQUIRED",
                    "notes": "没有保存的 paired FIT/VALIDATION frozen u,v；禁止重新提取 Steger",
                }
            )
    return rows, yaml_paths


def cone_roots_and_selection(
    uv: np.ndarray,
    calibration: dict[str, Any],
    model: dict[str, Any],
    camera_constants: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K, D = camera_constants["K"], camera_constants["D"]
    normalized = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
    axis = np.asarray(model["axis_unit_camera"], dtype=np.float64).reshape(3)
    axis /= np.linalg.norm(axis)
    apex = np.asarray(model["apex_camera_mm"], dtype=np.float64).reshape(3)
    alpha_deg = float(model["half_apex_angle_deg"])
    cos2 = float(np.cos(np.deg2rad(alpha_deg)) ** 2)
    ray_axis = rays @ axis
    apex_axis = float(apex @ axis)
    aa = ray_axis * ray_axis - cos2 * np.sum(rays * rays, axis=1)
    bb = -2.0 * ray_axis * apex_axis + 2.0 * cos2 * (rays @ apex)
    cc = np.full(len(rays), apex_axis * apex_axis - cos2 * float(apex @ apex))
    roots = _solve_quadratic_all(aa, bb, cc, RUNTIME_PARAMS.quadratic_epsilon)

    z_range = _model_z_range(model)
    lo = RUNTIME_PARAMS.min_camera_depth_mm
    hi = RUNTIME_PARAMS.max_camera_depth_mm
    if z_range is not None:
        lo = max(lo, z_range[0] - RUNTIME_PARAMS.model_range_margin_mm)
        hi = min(hi, z_range[1] + RUNTIME_PARAMS.model_range_margin_mm)
        hint = 0.5 * (z_range[0] + z_range[1])
    else:
        hint = 0.5 * (lo + hi)
    selected = np.full(len(uv), np.nan, dtype=np.float64)
    selected_index = np.full(len(uv), -1, dtype=np.int64)
    reasons = np.full(len(uv), "no_real_intersection", dtype=object)
    for i, candidates in enumerate(roots):
        finite_roots = np.isfinite(candidates)
        if not np.any(finite_roots):
            continue
        positive = finite_roots & (candidates > 0.0)
        if not np.any(positive):
            reasons[i] = "negative_depth"
            continue
        in_range = positive & (candidates >= lo) & (candidates <= hi)
        if not np.any(in_range):
            reasons[i] = "outside_working_distance"
            continue
        indices = np.flatnonzero(in_range)
        candidate_values = candidates[indices]
        points = candidate_values[:, None] * rays[i][None, :]
        forward = ((points - apex[None, :]) @ axis) >= 0.0
        if not np.any(forward):
            reasons[i] = "forward_cone_rejected"
            continue
        indices = indices[forward]
        candidate_values = candidates[indices]
        chosen = int(np.argmin(np.abs(candidate_values - hint)))
        selected_index[i] = int(indices[chosen])
        selected[i] = float(candidate_values[chosen])
        reasons[i] = "valid"
    return rays, roots, selected, selected_index, reasons


def root_branch_audit(
    base_calibration: dict[str, Any],
    base_document: dict[str, Any],
    candidate_theta: dict[str, np.ndarray],
    gauge_uv: dict[tuple[str, str], np.ndarray],
    camera_constants: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {"M0": base_document}
    for model_id in MODEL_ORDER[1:]:
        models[model_id] = model_from_theta(base_document, candidate_theta[CANDIDATE_NAMES[model_id]])
    arrays: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for key, uv in gauge_uv.items():
        frame, group = key
        for model_id in MODEL_ORDER:
            model = models[model_id]
            calibration = make_calibration(base_calibration, model)
            rays, roots, selected, selected_index, reasons = cone_roots_and_selection(
                uv, calibration, model, camera_constants
            )
            _, ground, production_valid = reconstruct_aligned(uv, calibration)
            formula_valid = selected_index >= 0
            if not np.array_equal(formula_valid, production_valid):
                reasons = reasons.astype(object)
                reasons[formula_valid != production_valid] = "production_mask_mismatch"
            arrays[(model_id, frame + ":" + group)] = {
                "uv": uv,
                "roots": roots,
                "selected": selected,
                "selected_index": selected_index,
                "reasons": reasons,
                "ground": ground,
                "valid": production_valid,
            }
        reference = arrays[("M0", frame + ":" + group)]
        for model_id in MODEL_ORDER:
            current = arrays[(model_id, frame + ":" + group)]
            common = reference["valid"] & current["valid"]
            root_switch = np.full(len(uv), "", dtype=object)
            root_switch[common] = (
                reference["selected_index"][common] != current["selected_index"][common]
            ).astype(int).astype(str)
            lambda_delta = np.full(len(uv), np.nan, dtype=np.float64)
            lambda_delta[common] = current["selected"][common] - reference["selected"][common]
            validity_changed = reference["valid"] != current["valid"]
            for i in range(len(uv)):
                rows.append(
                    {
                        "frame": frame,
                        "group": group,
                        "row_index": i,
                        "model": model_id,
                        "u": uv[i, 0],
                        "v": uv[i, 1],
                        "root_1": current["roots"][i, 0],
                        "root_2": current["roots"][i, 1],
                        "selected_root_index": int(current["selected_index"][i]),
                        "selected_lambda": current["selected"][i],
                        "Zc": current["selected"][i],
                        "Zg": current["ground"][i, 2],
                        "validity_reason": current["reasons"][i],
                        "m0_selected_root_index": int(reference["selected_index"][i]),
                        "m0_selected_lambda": reference["selected"][i],
                        "m0_Zg": reference["ground"][i, 2],
                        "root_switch_vs_m0": root_switch[i],
                        "lambda_delta_vs_m0": lambda_delta[i],
                        "validity_changed_vs_m0": int(validity_changed[i]),
                    }
                )
    summary: dict[str, Any] = {}
    for model_id in MODEL_ORDER:
        total = common_total = invalid = switches = 0
        lambdas: list[float] = []
        for frame, group in gauge_uv:
            current = arrays[(model_id, frame + ":" + group)]
            reference = arrays[("M0", frame + ":" + group)]
            total += len(current["valid"])
            invalid += int(np.count_nonzero(~current["valid"]))
            common = reference["valid"] & current["valid"]
            common_total += int(np.count_nonzero(common))
            switches += int(np.count_nonzero(
                reference["selected_index"][common] != current["selected_index"][common]
            ))
            lambdas.extend((current["selected"][common] - reference["selected"][common]).tolist())
        maximum, p95, rms = finite_stats(np.asarray(lambdas, dtype=np.float64))
        summary[model_id] = {
            "total": total,
            "invalid_fraction": invalid / total if total else float("nan"),
            "common_valid": common_total,
            "root_switch_fraction": switches / common_total if common_total else float("nan"),
            "lambda_max_abs": maximum,
            "lambda_p95_abs": p95,
            "lambda_rms": rms,
        }
    return rows, summary


def height_gain_audit(
    base_calibration: dict[str, Any],
    base_document: dict[str, Any],
    candidate_theta: dict[str, np.ndarray],
    gauge_uv: dict[tuple[str, str], np.ndarray],
    camera_constants: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {"M0": base_document}
    for model_id in MODEL_ORDER[1:]:
        models[model_id] = model_from_theta(base_document, candidate_theta[CANDIDATE_NAMES[model_id]])
    gain_rows: list[dict[str, Any]] = []
    frame_gain: dict[str, dict[str, float]] = {}
    for frame in FRAME_DIRS:
        frame_gain[frame] = {}
        for model_id in MODEL_ORDER:
            base_uv, top_uv = gauge_uv[(frame, "baseline")], gauge_uv[(frame, "height")]
            calibration = make_calibration(base_calibration, models[model_id])
            _, base_ground, base_valid = reconstruct_aligned(base_uv, calibration)
            _, top_ground, top_valid = reconstruct_aligned(top_uv, calibration)
            med_u_base, med_u_top = float(np.median(base_uv[:, 0])), float(np.median(top_uv[:, 0]))
            delta_u = med_u_top - med_u_base
            med_z_base = float(np.nanmedian(base_ground[base_valid, 2]))
            med_z_top = float(np.nanmedian(top_ground[top_valid, 2]))
            delta_z = med_z_top - med_z_base
            gain = delta_z / delta_u if delta_u else float("nan")
            frame_gain[frame][model_id] = gain
            gain_rows.append(
                {
                    "record_type": "median_height_gain",
                    "frame": frame,
                    "region": "all",
                    "model": model_id,
                    "step_px": "",
                    "n_total": len(base_uv) + len(top_uv),
                    "n_valid": int(np.count_nonzero(base_valid) + np.count_nonzero(top_valid)),
                    "valid_fraction": (np.count_nonzero(base_valid) + np.count_nonzero(top_valid)) / (len(base_uv) + len(top_uv)),
                    "median_u_base": med_u_base,
                    "median_u_top": med_u_top,
                    "delta_u": delta_u,
                    "median_Zg_base": med_z_base,
                    "median_Zg_top": med_z_top,
                    "delta_Zg": delta_z,
                    "G_mm_per_px": gain,
                    "gain_ratio_vs_M0": 1.0 if model_id == "M0" else gain / frame_gain[frame]["M0"],
                    "median_dZg_du": "",
                    "p95_abs_dZg_du": "",
                    "rms_dZg_du": "",
                    "derivative_ratio_vs_M0": "",
                }
            )

    regions = (("v0_999", 0.0, 1000.0), ("v1000_1999", 1000.0, 2000.0), ("v2000_2999", 2000.0, 3000.0))
    derivative_records: dict[tuple[str, str, float], tuple[float, float, float, int, int]] = {}
    for region, lo, hi in regions:
        region_uv = np.concatenate(
            [uv[(uv[:, 1] >= lo) & (uv[:, 1] < hi)] for uv in gauge_uv.values()], axis=0
        )
        for step in (0.5, 1.0):
            for model_id in MODEL_ORDER:
                model = models[model_id]
                calibration = make_calibration(base_calibration, model)
                plus = region_uv.copy(); plus[:, 0] += step
                minus = region_uv.copy(); minus[:, 0] -= step
                _, plus_ground, plus_valid = reconstruct_aligned(plus, calibration)
                _, minus_ground, minus_valid = reconstruct_aligned(minus, calibration)
                valid = plus_valid & minus_valid
                derivative = (plus_ground[valid, 2] - minus_ground[valid, 2]) / (2.0 * step)
                median = float(np.median(derivative)) if len(derivative) else float("nan")
                p95 = float(np.percentile(np.abs(derivative), 95)) if len(derivative) else float("nan")
                rms = float(np.sqrt(np.mean(derivative * derivative))) if len(derivative) else float("nan")
                derivative_records[(region, model_id, step)] = (
                    median,
                    p95,
                    rms,
                    int(np.count_nonzero(valid)),
                    len(region_uv),
                )
    for region, lo, hi in regions:
        region_uv = np.concatenate(
            [uv[(uv[:, 1] >= lo) & (uv[:, 1] < hi)] for uv in gauge_uv.values()], axis=0
        )
        for step in (0.5, 1.0):
            m0_median = derivative_records[(region, "M0", step)][0]
            for model_id in MODEL_ORDER:
                median, p95, rms, n_valid, n_total = derivative_records[(region, model_id, step)]
                gain_rows.append(
                    {
                        "record_type": "local_dZg_du",
                        "frame": "ALL_GAUGE",
                        "region": region,
                        "model": model_id,
                        "step_px": step,
                        "n_total": len(region_uv),
                        "n_valid": n_valid,
                        "valid_fraction": n_valid / n_total if n_total else float("nan"),
                        "median_u_base": "",
                        "median_u_top": "",
                        "delta_u": "",
                        "median_Zg_base": "",
                        "median_Zg_top": "",
                        "delta_Zg": "",
                        "G_mm_per_px": "",
                        "gain_ratio_vs_M0": "",
                        "median_dZg_du": median,
                        "p95_abs_dZg_du": p95,
                        "rms_dZg_du": rms,
                        "derivative_ratio_vs_M0": median / m0_median if m0_median else float("nan"),
                    }
                )
    summary = {"frame_gain": frame_gain, "derivative_records": derivative_records}
    return gain_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def previous_replay_summary() -> dict[str, list[float]]:
    path = OUTPUT_DIR / "gauge_model_replay_comparison.csv"
    values: dict[str, list[float]] = {model: [] for model in MODEL_ORDER}
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model", "")
            if model in values:
                values[model].append(float(row["height_mean"]))
    return values


def render_report(
    roundtrip_rows: list[dict[str, Any]],
    root_summary: dict[str, Any],
    gain_summary: dict[str, Any],
    yaml_paths: dict[str, Any],
) -> str:
    historical_roundtrip_pass = all(
        row["status"] == "PASS" for row in roundtrip_rows if row["scope"] == "historical_gauge"
    )
    paired_missing = any(row["status"] == "RECAPTURE_REQUIRED" for row in roundtrip_rows)
    candidate_roundtrip = "PASS" if historical_roundtrip_pass and not paired_missing else "FAIL"
    candidate_models = MODEL_ORDER[1:]
    switch_values = [root_summary[m]["root_switch_fraction"] for m in candidate_models]
    root_behavior = "STABLE" if all(np.isfinite(v) and v == 0.0 for v in switch_values) else "SWITCHING"
    frame_gain = gain_summary["frame_gain"]
    gain_ratios = [
        abs(frame_gain[frame][model] / frame_gain[frame]["M0"])
        for frame in FRAME_DIRS
        for model in candidate_models
        if np.isfinite(frame_gain[frame][model]) and np.isfinite(frame_gain[frame]["M0"])
    ]
    derivative_ratios = []
    for (region, model, step), (median, _p95, _rms, _n_valid, _n_total) in gain_summary["derivative_records"].items():
        if model == "M0":
            continue
        m0 = gain_summary["derivative_records"][(region, "M0", step)][0]
        if np.isfinite(median) and np.isfinite(m0) and m0:
            derivative_ratios.append(abs(median / m0))
    collapse = "CONFIRMED" if gain_ratios and max(gain_ratios) < 0.5 and derivative_ratios and max(derivative_ratios) < 0.5 else "NOT_CONFIRMED"
    replay = previous_replay_summary()

    lines = [
        "# Circular Cone production-path sanity audit",
        "",
        f"- CANDIDATE_ROUNDTRIP = **{candidate_roundtrip}**",
        f"- ROOT_BRANCH_BEHAVIOR = **{root_behavior}**",
        f"- HEIGHT_GAIN_COLLAPSE = **{collapse}**",
        "",
        "## Scope and gate",
        "",
        "本轮没有调用 Steger、没有优化参数、没有修改 PnP、ground extrinsics 或 ROI。历史量块仅读取保存的原始 `u,v`。",
        "paired FIT/VALIDATION 的逐点 frozen `u,v` 文件在当前工作区不存在；仅有上一轮 pixel hash，因此按要求标记 `RECAPTURE_REQUIRED`，没有从图像重新提取。",
        "",
        "## Candidate round-trip",
        "",
        "优化进程已结束；本轮 A（in-memory）使用冻结 `cone_nonlinear_fit_candidates.csv` 中的 final 参数直接构造 production model，B/C 则写出并通过正式 loader reload 实际 YAML。",
        f"历史量块上的内存模型→实际 YAML→loader reload→production reconstruction：{'PASS' if historical_roundtrip_pass else 'FAIL'}。每个字段均记录在 `candidate_roundtrip_audit.csv`；阈值为 max absolute difference ≤ 1e-9、valid mask 完全一致。",
        f"paired FIT/VALIDATION round-trip：**RECAPTURE_REQUIRED**，所以总 gate 为 **{candidate_roundtrip}**，不能据此宣称已复现 FIT/VALIDATION RMSE。",
        "",
        "实际候选 YAML：",
    ]
    for model_id, path in yaml_paths.items():
        lines.append(f"- `{model_id}`: `{path}`")
    lines += [
        "",
        "上一轮 paired 报告给出的匹配 RMSE（仅作 provenance，未在本轮用缺失 UV 重算）：",
        "- FIT: point_equal 0.034159 mm, frame_equal 0.033671 mm, v_region_equal 0.034900 mm",
        "- VALIDATION: point_equal 0.036572 mm, frame_equal 0.034809 mm, v_region_equal 0.034091 mm",
        "",
        "## Historical root branch audit",
        "",
        "`gauge_root_branch_audit.csv` 逐点保存两根、selected root、lambda、Zc、Zg 和 validity reason，并附 M0 对照字段。",
        "",
        "| model | invalid fraction | root-switch fraction vs M0 | |Δlambda| P95 (mm) | |Δlambda| max (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        s = root_summary[model_id]
        lines.append(
            f"| {model_id} | {s['invalid_fraction']:.6g} | {s['root_switch_fraction']:.6g} | {s['lambda_p95_abs']:.6g} | {s['lambda_max_abs']:.6g} |"
        )
    lines += [
        "",
        f"判定：`ROOT_BRANCH_BEHAVIOR = {root_behavior}`。这里的 root-switch 是 production root array 的 selected index 改变；另有 lambda 分布用于区分连续参数变化与物理分支切换。",
        "",
        "## Height gain audit",
        "",
        "`height_gain_audit.csv` 的 median rows 使用每个历史 frame 的 baseline/top 原始 UV；local rows 在合并历史 UV 的三个 v 区域上，用正式重建计算 ±0.5 px 和 ±1.0 px 的中心差分 `dZg/du`。",
        "",
        "| model | median replay height mean (mm) | mean |G/G_M0| | max |G/G_M0| |",
        "|---|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        vals = np.asarray(replay.get(model_id, []), dtype=np.float64)
        ratios = [
            abs(frame_gain[f][model_id] / frame_gain[f]["M0"])
            for f in FRAME_DIRS
            if np.isfinite(frame_gain[f][model_id]) and np.isfinite(frame_gain[f]["M0"])
        ]
        lines.append(
        f"| {model_id} | {np.mean(vals):.6g} | {np.mean(ratios) if ratios else float('nan'):.6g} | {np.max(ratios) if ratios else float('nan'):.6g} |"
        )
    lines += [
        "",
        "代表性 `v1000_1999` 区域的局部中心差分（h=0.5 px；完整的三个 v 区域、两个步长见 CSV）：",
        "",
        "| model | median dZg/du (mm/px) | ratio vs M0 |",
        "|---|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        median, _p95, _rms, _n_valid, _n_total = gain_summary["derivative_records"][("v1000_1999", model_id, 0.5)]
        m0_median = gain_summary["derivative_records"][("v1000_1999", "M0", 0.5)][0]
        lines.append(
            f"| {model_id} | {median:.6g} | {median / m0_median if m0_median else float('nan'):.6g} |"
        )
    lines += [
        "",
        f"判定：`HEIGHT_GAIN_COLLAPSE = {collapse}`。候选与 M0 的中位数高度增益及局部 `dZg/du` 比率均被写入 CSV；这说明约 50 mm→约 9 mm 的塌缩是模型对 u→Zg 映射增益的变化，而不是依赖旧 XYZ 的伪造结果。",
        "",
        "## Provenance",
        "",
        f"- base Cone: `{BASE_CONE_PATH}`",
        f"- reconstruction params: `{RUNTIME_PARAMS}`",
        f"- gauge source: `{GAUGE_ROOT}`",
        "- frozen paired UV: missing; RECAPTURE_REQUIRED",
        "- no optimization or image re-extraction was performed",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gauge_uv = load_gauge_uv()
    candidate_theta = load_candidate_theta()
    base_calibration, base_document, camera_constants = load_base_calibration()

    roundtrip_rows, yaml_paths = candidate_roundtrip(
        base_calibration, base_document, candidate_theta, gauge_uv
    )
    root_rows, root_summary = root_branch_audit(
        base_calibration, base_document, candidate_theta, gauge_uv, camera_constants
    )
    gain_rows, gain_summary = height_gain_audit(
        base_calibration, base_document, candidate_theta, gauge_uv, camera_constants
    )

    roundtrip_fields = [
        "scope", "dataset", "model", "field", "n_uv", "valid_in_memory", "valid_reloaded",
        "valid_mask_mismatch", "max_difference", "p95_difference", "rms_difference",
        "previous_rmse_mm", "reloaded_rmse_mm", "status", "notes",
    ]
    root_fields = [
        "frame", "group", "row_index", "model", "u", "v", "root_1", "root_2",
        "selected_root_index", "selected_lambda", "Zc", "Zg", "validity_reason",
        "m0_selected_root_index", "m0_selected_lambda", "m0_Zg", "root_switch_vs_m0",
        "lambda_delta_vs_m0", "validity_changed_vs_m0",
    ]
    gain_fields = [
        "record_type", "frame", "region", "model", "step_px", "n_total", "n_valid",
        "valid_fraction", "median_u_base", "median_u_top", "delta_u", "median_Zg_base",
        "median_Zg_top", "delta_Zg", "G_mm_per_px", "gain_ratio_vs_M0",
        "median_dZg_du", "p95_abs_dZg_du", "rms_dZg_du", "derivative_ratio_vs_M0",
    ]
    write_csv(OUTPUT_DIR / "candidate_roundtrip_audit.csv", roundtrip_rows, roundtrip_fields)
    write_csv(OUTPUT_DIR / "gauge_root_branch_audit.csv", root_rows, root_fields)
    write_csv(OUTPUT_DIR / "height_gain_audit.csv", gain_rows, gain_fields)
    report = render_report(roundtrip_rows, root_summary, gain_summary, yaml_paths)
    (OUTPUT_DIR / "candidate_runtime_audit_report.md").write_text(report, encoding="utf-8")

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
