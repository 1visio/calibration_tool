#!/usr/bin/env python3
"""Task 5C: FIT-only frame-dependent residual source audit.

No laser model is fitted and no correction is produced.  Validation images,
truth points and residuals are never loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

import audit_laser_plane_triplet_coverage as coverage  # noqa: E402
import audit_triplet_edge_extension_observability as edge  # noqa: E402
import compare_circular_vs_elliptical_cone as task5a  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
import run_circular_cone_local_fullfit as task3b2  # noqa: E402
import run_circular_cone_residual_decomposition as task4a  # noqa: E402


OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0814" / "frame_effect_source_audit"
TASK5A_PROVENANCE = ROOT / "projects" / "daheng" / "outputs" / "0814" / "circular_vs_elliptical_cone" / "provenance.json"
ORIGINAL_GEOMETRY = ROOT / "projects" / "daheng" / "outputs" / "0813" / "triplet_coverage_audit" / "triplet_frame_geometry.csv"
ORIGINAL_PROVENANCE = ROOT / "projects" / "daheng" / "outputs" / "0813" / "triplet_coverage_audit" / "triplet_provenance.csv"
ORIGINAL_DATA = ROOT / "projects" / "daheng" / "data" / "laser_plane"
EXTENSION_DATA = ORIGINAL_DATA / "fit_edge_extension"
TARGET_FRAME = "027"
SENSOR_V_BIN_PX = 60
OVERLAP_U_BIN_PX = 32
OVERLAP_V_BIN_PX = 30
OVERLAP_LAMBDA_BIN_MM = 5.0

OUTCOMES = ("bias_mm", "a_frame_mm", "k_frame_mm_per_s", "rmse_mm", "offset_tilt_explained_fraction")
PREDICTORS: tuple[tuple[str, str], ...] = (
    ("board_center_z_mm", "pose_depth"),
    ("board_tilt_deg", "pose_orientation"),
    ("board_roll_deg", "pose_orientation"),
    ("board_pitch_deg", "pose_orientation"),
    ("board_nx", "pose_orientation"),
    ("board_ny", "pose_orientation"),
    ("board_nz", "pose_orientation"),
    ("pnp_rmse_px", "pnp"),
    ("u_center_px", "coverage"),
    ("u_span_px", "coverage"),
    ("v_center_px", "coverage"),
    ("v_span_px", "coverage"),
    ("lambda_span_mm", "coverage"),
    ("chess_to_laser_s", "acquisition_timing"),
    ("nolaser_to_laser_s", "acquisition_timing"),
    ("chess_focus_laplacian", "image_quality"),
    ("chess_mean_dn", "image_quality"),
    ("laser_mean_dn", "image_quality"),
    ("laser_p99_dn", "image_quality"),
    ("laser_dynamic_range_u8", "image_quality"),
    ("laser_coverage", "image_quality"),
    ("quality_warning_count", "image_quality"),
    ("acquisition_order", "acquisition_order"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5C frame-dependent residual source audit")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--measurement-config", type=Path, default=task3b2.MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_fit_records(intrinsics: Any) -> tuple[list[task3a.FrameRecord], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    records = task3a.load_old_records()
    points, geometry, provenance = edge.extract_extension(task3a.FIT_EXTENSION, "fit", intrinsics, True)
    for frame_id in sorted(points, key=int):
        data = points[frame_id]; g = geometry[frame_id]
        truth = coverage.plane_ray_truth(
            data["u"], data["v"], np.asarray([g["plane_nx"], g["plane_ny"], g["plane_nz"]]),
            float(g["plane_d"]), intrinsics.camera_matrix, intrinsics.dist_coeffs,
        )
        if not np.all(truth["valid"]):
            raise RuntimeError(f"Invalid extension truth: {frame_id}")
        provenance_row = next(row for row in provenance if row["frame_id"] == frame_id)
        records.append(task3a.FrameRecord(
            frame_id=frame_id, split="fit", pixels_uv=np.column_stack([data["u"], data["v"]]).astype(np.float64),
            truth_points=np.asarray(truth["points"], dtype=np.float64),
            plane=np.asarray([g["plane_nx"], g["plane_ny"], g["plane_nz"], g["plane_d"]]),
            pnp_rmse_px=float(provenance_row["pnp_rmse_px"]), point_count=len(data["u"]),
            quality_warnings=str(provenance_row["quality_warnings"]), source="fit_edge_extension",
        ))
    if [record.frame_id for record in records] != task3a.FIT_IDS or len(records) != 30:
        raise RuntimeError("FIT registry mismatch")
    return records, geometry, provenance


def current_model(provenance: Mapping[str, Any], records: Sequence[task3a.FrameRecord]) -> dict[str, Any]:
    expected = [frame_id for frame_id in task3a.FIT_IDS if frame_id != TARGET_FRAME]
    if provenance.get("main_fit_ids") != expected or provenance.get("validation_opened") is not False:
        raise RuntimeError("Task 5A provenance boundary mismatch")
    z = np.concatenate([record.truth_points[:, 2] for record in records if record.frame_id != TARGET_FRAME])
    return task5a.circular_params_to_runtime_model(
        np.asarray(provenance["full_fits"]["Circular"]["params"]),
        np.asarray(provenance["reference_anchor_mm"]), [float(np.min(z)), float(np.max(z))],
    )


def collect_residual_points(
    records: Sequence[task3a.FrameRecord], calibration: Mapping[str, Any], reconstruction_params: Any, model: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidate = copy.deepcopy(dict(calibration)); candidate["laser_model"] = copy.deepcopy(dict(model))
    rows: list[dict[str, Any]] = []; invalid: dict[str, int] = {}
    for record in records:
        reconstructed, valid, _ = task4a.lambda_by_input(record.pixels_uv, candidate, reconstruction_params)
        invalid[record.frame_id] = int(np.count_nonzero(~valid))
        uv = np.asarray(record.pixels_uv)
        centered = uv - np.mean(uv, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        if direction[1] < 0:
            direction = -direction
        stripe_coordinate = centered @ direction
        scale = max(float(np.max(np.abs(stripe_coordinate))), 1.0e-12)
        s = stripe_coordinate / scale
        for index, (pixel, truth) in enumerate(zip(uv, record.truth_points)):
            ok = bool(valid[index])
            rows.append({
                "frame_id": record.frame_id, "source": record.source, "is_frame027": record.frame_id == TARGET_FRAME,
                "point_index": index, "u_px": float(pixel[0]), "v_px": float(pixel[1]), "s_normalized": float(s[index]),
                "lambda_truth_mm": float(truth[2]), "lambda_model_mm": float(reconstructed[index]) if ok else math.nan,
                "e_lambda_mm": float(truth[2] - reconstructed[index]) if ok else math.nan, "valid": ok,
            })
    return rows, invalid


def orientation_from_normal(normal: np.ndarray) -> tuple[float, float, float]:
    n = np.asarray(normal, dtype=np.float64); front = -n
    tilt = math.degrees(math.acos(float(np.clip(front[2], -1.0, 1.0))))
    roll = math.degrees(math.atan2(float(front[1]), float(front[2])))
    pitch = math.degrees(math.atan2(float(-front[0]), float(math.hypot(front[1], front[2]))))
    return tilt, roll, pitch


def fit_metadata_rows(extension_geometry: Mapping[str, Mapping[str, Any]], extension_provenance: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    old_geometry = {f"{int(row['frame_id']):03d}": row for row in read_csv(ORIGINAL_GEOMETRY) if int(row["frame_id"]) <= 18}
    old_provenance = {f"{int(row['frame_id']):03d}": row for row in read_csv(ORIGINAL_PROVENANCE) if int(row["frame_id"]) <= 18}
    frame_sources = ((ORIGINAL_DATA / "frames.csv", set(task3a.FIT_IDS[:18])), (EXTENSION_DATA / "frames.csv", set(task3a.FIT_IDS[18:])))
    role_rows: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for path, allowed in frame_sources:
        for row in read_csv(path):
            frame_id = f"{int(row['pose_id']):03d}"
            if frame_id in allowed and row.get("role") in {"chess", "nolaser", "laser"}:
                role_rows[frame_id][str(row["role"])] = row
    if sorted(role_rows, key=int) != task3a.FIT_IDS:
        raise RuntimeError("FIT acquisition metadata registry mismatch")
    chess_times = {fid: int(role_rows[fid]["chess"]["host_timestamp_ns"]) for fid in role_rows}
    order = {fid: index + 1 for index, fid in enumerate(sorted(chess_times, key=chess_times.get))}
    first_time = min(chess_times.values())
    extension_prov = {str(row["frame_id"]): row for row in extension_provenance}
    result: dict[str, dict[str, Any]] = {}
    for frame_id in task3a.FIT_IDS:
        roles = role_rows[frame_id]
        if set(roles) != {"chess", "nolaser", "laser"}:
            raise RuntimeError(f"Incomplete triplet metadata: {frame_id}")
        if frame_id in old_geometry:
            g = old_geometry[frame_id]
            normal = np.asarray([number(g["board_plane_nx"]), number(g["board_plane_ny"]), number(g["board_plane_nz"])])
            center_z = number(g["board_center_zc_mm"]); pnp = number(g["pnp_reprojection_rmse_px"])
            verified = str(old_provenance[frame_id].get("sha256_verified", "")).lower() == "true"
        else:
            g = extension_geometry[frame_id]
            normal = np.asarray([number(g["plane_nx"]), number(g["plane_ny"]), number(g["plane_nz"])])
            center_z = number(g["board_z"]); pnp = number(extension_prov[frame_id]["pnp_rmse_px"])
            verified = bool(extension_prov[frame_id]["sha256_verified"])
        tilt, roll, pitch = orientation_from_normal(normal)
        chess, nolaser, laser = roles["chess"], roles["nolaser"], roles["laser"]
        warning_text = ";".join(str(roles[role].get("quality_warnings", "")) for role in ("chess", "nolaser", "laser"))
        result[frame_id] = {
            "triplet_complete": True, "sha256_verified": verified,
            "board_center_z_mm": center_z, "board_tilt_deg": tilt, "board_roll_deg": roll, "board_pitch_deg": pitch,
            "board_nx": float(normal[0]), "board_ny": float(normal[1]), "board_nz": float(normal[2]), "pnp_rmse_px": pnp,
            "chess_to_laser_s": (int(laser["host_timestamp_ns"]) - int(chess["host_timestamp_ns"])) * 1.0e-9,
            "nolaser_to_laser_s": (int(laser["host_timestamp_ns"]) - int(nolaser["host_timestamp_ns"])) * 1.0e-9,
            "chess_focus_laplacian": number(chess.get("focus_laplacian")), "chess_mean_dn": number(chess.get("mean_dn")),
            "laser_mean_dn": number(laser.get("mean_dn")), "laser_p99_dn": number(laser.get("p99_dn")),
            "laser_dynamic_range_u8": number(laser.get("dynamic_range_u8")), "laser_coverage": number(laser.get("laser_coverage")),
            "quality_warning_count": sum(bool(item) for item in warning_text.split(";") if item), "quality_warnings": warning_text,
            "acquisition_order": order[frame_id], "acquisition_time_from_first_s": (chess_times[frame_id] - first_time) * 1.0e-9,
        }
    return result


def frame_summary(points: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in points:
        if bool(row["valid"]):
            groups[str(row["frame_id"])].append(row)
    output: list[dict[str, Any]] = []
    for frame_id in task3a.FIT_IDS:
        rows = groups[frame_id]
        y = np.asarray([float(row["e_lambda_mm"]) for row in rows]); s = np.asarray([float(row["s_normalized"]) for row in rows])
        design = np.column_stack([np.ones(len(y)), s]); beta = np.linalg.lstsq(design, y, rcond=None)[0]
        prediction = design @ beta; offset_prediction = np.full(len(y), float(np.mean(y)))
        total = float(y @ y); offset_sse = float(np.sum((y - offset_prediction) ** 2)); tilt_sse = float(np.sum((y - prediction) ** 2))
        uv = np.asarray([[float(row["u_px"]), float(row["v_px"])] for row in rows]); lam = np.asarray([float(row["lambda_truth_mm"]) for row in rows])
        output.append({
            "frame_id": frame_id, "is_frame027": frame_id == TARGET_FRAME, "valid_point_count": len(rows),
            "bias_mm": float(np.mean(y)), "mae_mm": float(np.mean(np.abs(y))), "rmse_mm": float(np.sqrt(np.mean(y * y))),
            "p95_abs_mm": float(np.percentile(np.abs(y), 95)), "a_frame_mm": float(beta[0]), "k_frame_mm_per_s": float(beta[1]),
            "k_peak_to_peak_mm": 2.0 * float(beta[1]), "offset_explained_fraction": 1.0 - offset_sse / total if total else math.nan,
            "offset_tilt_explained_fraction": 1.0 - tilt_sse / total if total else math.nan,
            "tilt_incremental_fraction": (offset_sse - tilt_sse) / total if total else math.nan,
            "remaining_rmse_mm": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "u_min_px": float(np.min(uv[:, 0])), "u_max_px": float(np.max(uv[:, 0])), "u_center_px": float(np.mean(uv[:, 0])), "u_span_px": float(np.ptp(uv[:, 0])),
            "v_min_px": float(np.min(uv[:, 1])), "v_max_px": float(np.max(uv[:, 1])), "v_center_px": float(np.mean(uv[:, 1])), "v_span_px": float(np.ptp(uv[:, 1])),
            "lambda_min_mm": float(np.min(lam)), "lambda_max_mm": float(np.max(lam)), "lambda_span_mm": float(np.ptp(lam)),
            **metadata[frame_id],
        })
    return output


def correlation_rows(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, rows in (("all30", list(summary)), ("leave027_out", [row for row in summary if row["frame_id"] != TARGET_FRAME])):
        for outcome in OUTCOMES:
            for predictor, category in PREDICTORS:
                pairs = [(number(row[predictor]), number(row[outcome])) for row in rows]
                pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
                x = np.asarray([p[0] for p in pairs]); y = np.asarray([p[1] for p in pairs])
                if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
                    values = (math.nan,) * 6
                else:
                    pearson = stats.pearsonr(x, y); spearman = stats.spearmanr(x, y)
                    slope, intercept, _, _, _ = stats.linregress(x, y)
                    robust = stats.theilslopes(y, x, 0.90)
                    values = (float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(spearman.pvalue), float(slope), float(robust.slope))
                output.append({
                    "scope": scope, "outcome": outcome, "predictor": predictor, "predictor_category": category, "n": len(x),
                    "pearson_r": values[0], "pearson_p": values[1], "spearman_rho": values[2], "spearman_p": values[3],
                    "ols_slope": values[4], "theil_sen_slope": values[5],
                })
    return output


def overlap_rows(points: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in points:
        if not bool(row["valid"]):
            continue
        key = (int(math.floor(float(row["u_px"]) / OVERLAP_U_BIN_PX)), int(math.floor(float(row["v_px"]) / OVERLAP_V_BIN_PX)), int(math.floor(float(row["lambda_truth_mm"]) / OVERLAP_LAMBDA_BIN_MM)))
        cells[key].append(row)
    output: list[dict[str, Any]] = []
    between_sum = within_sum = weight_sum = 0.0
    covered_frames: set[str] = set()
    for (iu, iv, ilam), rows in sorted(cells.items()):
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[str(row["frame_id"])].append(float(row["e_lambda_mm"]))
        if len(groups) < 2:
            continue
        medians = np.asarray([np.median(values) for values in groups.values()])
        means = np.asarray([np.mean(values) for values in groups.values()])
        within_variances = np.asarray([np.mean((np.asarray(values) - np.mean(values)) ** 2) for values in groups.values()])
        between = float(np.var(means)); within = float(np.mean(within_variances)); weight = float(len(groups))
        between_sum += weight * between; within_sum += weight * within; weight_sum += weight; covered_frames.update(groups)
        output.append({
            "u_start_px": iu * OVERLAP_U_BIN_PX, "u_end_px": (iu + 1) * OVERLAP_U_BIN_PX,
            "v_start_px": iv * OVERLAP_V_BIN_PX, "v_end_px": (iv + 1) * OVERLAP_V_BIN_PX,
            "lambda_start_mm": ilam * OVERLAP_LAMBDA_BIN_MM, "lambda_end_mm": (ilam + 1) * OVERLAP_LAMBDA_BIN_MM,
            "point_count": len(rows), "unique_frame_count": len(groups), "frame_ids": ";".join(sorted(groups)),
            "frame_median_residual_mean_mm": float(np.mean(medians)), "frame_median_residual_std_mm": float(np.std(medians, ddof=1)),
            "frame_median_residual_range_mm": float(np.ptp(medians)), "between_frame_variance_mm2": between,
            "within_frame_variance_mm2": within, "between_fraction": between / (between + within) if between + within > 0 else math.nan,
            "contains_frame027": TARGET_FRAME in groups,
            "frame_median_residuals": ";".join(f"{fid}:{np.median(groups[fid]):.9g}" for fid in sorted(groups)),
        })
    fraction = between_sum / (between_sum + within_sum) if between_sum + within_sum > 0 else math.nan
    summary = {
        "overlap_cell_count": len(output), "covered_unique_frame_count": len(covered_frames),
        "covered_frame_ids": sorted(covered_frames), "weighted_between_frame_variance_mm2": between_sum / weight_sum if weight_sum else math.nan,
        "weighted_within_frame_variance_mm2": within_sum / weight_sum if weight_sum else math.nan,
        "overlap_between_frame_fraction": fraction,
        "median_frame_median_range_mm": float(np.median([row["frame_median_residual_range_mm"] for row in output])) if output else math.nan,
        "p95_frame_median_range_mm": float(np.percentile([row["frame_median_residual_range_mm"] for row in output], 95)) if output else math.nan,
    }
    return output, summary


def weighted_arrays(points: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray]:
    valid = [row for row in points if bool(row["valid"])]
    counts: dict[str, int] = defaultdict(int)
    for row in valid:
        counts[str(row["frame_id"])] += 1
    weights = np.asarray([1.0 / counts[str(row["frame_id"])] for row in valid])
    y = np.asarray([float(row["e_lambda_mm"]) for row in valid])
    return valid, y, weights


def fit_design(design: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    sqrt_w = np.sqrt(weights); beta = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    prediction = design @ beta; sse = float(np.sum(weights * (y - prediction) ** 2))
    return prediction, sse


def variance_decomposition(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows, y, weights = weighted_arrays(points); frame_ids = task3a.FIT_IDS
    active_bins = sorted({int(math.floor(float(row["v_px"]) / SENSOR_V_BIN_PX)) for row in rows})
    bin_index = {value: index for index, value in enumerate(active_bins)}; frame_index = {fid: index for index, fid in enumerate(frame_ids)}
    sensor = np.zeros((len(rows), 2 * len(active_bins))); frame = np.zeros((len(rows), 2 * len(frame_ids)))
    bin_u_refs: dict[int, float] = {}
    for value in active_bins:
        medians = []
        for fid in frame_ids:
            u = [float(row["u_px"]) for row in rows if int(math.floor(float(row["v_px"]) / SENSOR_V_BIN_PX)) == value and row["frame_id"] == fid]
            if u:
                medians.append(float(np.median(u)))
        bin_u_refs[value] = float(np.median(medians)) if medians else 0.0
    for index, row in enumerate(rows):
        b = int(math.floor(float(row["v_px"]) / SENSOR_V_BIN_PX)); bi = bin_index[b]; fi = frame_index[str(row["frame_id"])]
        sensor[index, 2 * bi] = 1.0; sensor[index, 2 * bi + 1] = (float(row["u_px"]) - bin_u_refs[b]) / 100.0
        frame[index, 2 * fi] = 1.0; frame[index, 2 * fi + 1] = float(row["s_normalized"])
    intercept = np.ones((len(rows), 1)); total = float(np.sum(weights * y * y))
    _, sse_intercept = fit_design(intercept, y, weights)
    _, sse_sensor = fit_design(sensor, y, weights); _, sse_frame = fit_design(frame, y, weights)
    _, sse_combined = fit_design(np.column_stack([sensor, frame]), y, weights)
    explained_sensor = 1.0 - sse_sensor / total; explained_frame = 1.0 - sse_frame / total; explained_combined = 1.0 - sse_combined / total
    unique_sensor = explained_combined - explained_frame; unique_frame = explained_combined - explained_sensor
    shared = explained_sensor + explained_frame - explained_combined
    sensor_allocated = unique_sensor + 0.5 * shared; frame_allocated = unique_frame + 0.5 * shared
    return {
        "frame_equal_total_energy": total, "intercept_only_explained_fraction": 1.0 - sse_intercept / total,
        "sensor_only_explained_fraction": explained_sensor, "frame_only_explained_fraction": explained_frame,
        "combined_explained_fraction": explained_combined, "sensor_unique_fraction": unique_sensor, "frame_unique_fraction": unique_frame,
        "shared_sensor_frame_fraction": shared, "sensor_allocated_fraction": sensor_allocated,
        "frame_allocated_fraction": frame_allocated, "unexplained_fraction": 1.0 - explained_combined,
        "sensor_definition": f"fixed {SENSOR_V_BIN_PX}px v-bin offset + within-bin u slope",
        "frame_definition": "per-frame offset + normalized stripe-direction tilt",
    }


def robust_sources(correlations: Sequence[Mapping[str, Any]], category_prefix: str) -> list[dict[str, Any]]:
    all_rows = {(row["outcome"], row["predictor"]): row for row in correlations if row["scope"] == "all30"}
    loo_rows = {(row["outcome"], row["predictor"]): row for row in correlations if row["scope"] == "leave027_out"}
    robust: list[dict[str, Any]] = []
    for key, first in all_rows.items():
        second = loo_rows[key]
        if not str(first["predictor_category"]).startswith(category_prefix):
            continue
        r1, p1, r2, p2 = number(first["spearman_rho"]), number(first["spearman_p"]), number(second["spearman_rho"]), number(second["spearman_p"])
        if all(math.isfinite(value) for value in (r1, p1, r2, p2)) and abs(r1) >= 0.60 and abs(r2) >= 0.60 and p1 < 0.01 and p2 < 0.01 and r1 * r2 > 0:
            robust.append(dict(first))
    return robust


def classify(
    correlations: Sequence[Mapping[str, Any]], overlap: Mapping[str, Any], overlap_leave027: Mapping[str, Any],
    variance: Mapping[str, Any], variance_leave027: Mapping[str, Any],
) -> dict[str, Any]:
    overlap_sufficient = int(overlap_leave027["overlap_cell_count"]) >= 30 and int(overlap_leave027["covered_unique_frame_count"]) >= 10
    pose = robust_sources(correlations, "pose_")
    acquisition = robust_sources(correlations, "acquisition") + robust_sources(correlations, "image_quality") + robust_sources(correlations, "pnp")
    frame_fraction = float(variance_leave027["frame_allocated_fraction"]); sensor_fraction = float(variance_leave027["sensor_allocated_fraction"])
    overlap_inconsistent = overlap_sufficient and min(float(overlap["overlap_between_frame_fraction"]), float(overlap_leave027["overlap_between_frame_fraction"])) >= 0.50
    pose_families = {row["predictor_category"] for row in pose}
    pose_correlated = frame_fraction >= 0.25 and len(pose_families) >= 2
    acquisition_correlated = frame_fraction >= 0.25 and bool(acquisition)
    if not overlap_sufficient:
        verdict = "E. INSUFFICIENT"
    elif pose_correlated and not acquisition_correlated and sensor_fraction < 0.30:
        verdict = "A. POSE_CORRELATED"
    elif sensor_fraction >= 0.60 and frame_fraction < 0.25 and not overlap_inconsistent:
        verdict = "C. MOSTLY_SENSOR_FIXED"
    elif overlap_inconsistent and frame_fraction >= 0.35 and not pose and not acquisition and sensor_fraction < 0.20:
        verdict = "B. CROSS_FRAME_INCONSISTENCY"
    else:
        verdict = "D. MIXED"
    return {
        "verdict": verdict, "overlap_sufficient": overlap_sufficient, "overlap_inconsistent": overlap_inconsistent,
        "robust_pose_correlation_count": len(pose), "robust_acquisition_correlation_count": len(acquisition),
        "robust_pose_correlations": [{"outcome": row["outcome"], "predictor": row["predictor"], "spearman_rho": row["spearman_rho"]} for row in pose],
        "robust_acquisition_correlations": [{"outcome": row["outcome"], "predictor": row["predictor"], "spearman_rho": row["spearman_rho"]} for row in acquisition],
    }


def make_plots(
    out: Path, summary: Sequence[Mapping[str, Any]], overlap: Sequence[Mapping[str, Any]],
    variance: Mapping[str, Any], variance_leave027: Mapping[str, Any],
) -> None:
    ordered = sorted(summary, key=lambda row: float(row["a_frame_mm"])); x = np.arange(len(ordered)); colors = ["#c53030" if row["frame_id"] == TARGET_FRAME else "#2b6cb0" for row in ordered]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x, [row["a_frame_mm"] for row in ordered], color=colors); axes[0].axhline(0, color="black", linewidth=0.7); axes[0].set_ylabel("a_frame / mm"); axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, [row["k_frame_mm_per_s"] for row in ordered], color=colors); axes[1].axhline(0, color="black", linewidth=0.7); axes[1].set_ylabel("k_frame / mm per s"); axes[1].set_xticks(x, [row["frame_id"] for row in ordered], rotation=75); axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "frame_bias_tilt_ranking.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = tuple((axes[row_index, column_index], outcome, predictor) for row_index, outcome in enumerate(("a_frame_mm", "k_frame_mm_per_s")) for column_index, predictor in enumerate(("board_center_z_mm", "board_tilt_deg", "board_roll_deg")))
    for axis, outcome, predictor in panels:
        for row in summary:
            color = "#c53030" if row["frame_id"] == TARGET_FRAME else "#2b6cb0"
            axis.scatter(row[predictor], row[outcome], color=color, s=28); axis.annotate(row["frame_id"], (row[predictor], row[outcome]), fontsize=6, alpha=0.75)
        axis.set_xlabel(predictor); axis.set_ylabel(outcome); axis.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "frame_bias_vs_board_pose.png", dpi=180); plt.close(fig)

    supported = sorted(overlap, key=lambda row: (-int(row["unique_frame_count"]), -int(row["point_count"])))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8)); axes_flat = axes.ravel()
    for axis, cell in zip(axes_flat, supported):
        values = []
        for item in str(cell["frame_median_residuals"]).split(";"):
            fid, value = item.split(":"); values.append((fid, float(value)))
        axis.scatter(np.arange(len(values)), [v for _, v in values], c=["#c53030" if fid == TARGET_FRAME else "#2b6cb0" for fid, _ in values])
        axis.axhline(0, color="black", linewidth=0.7); axis.set_xticks(np.arange(len(values)), [fid for fid, _ in values], rotation=60, fontsize=7)
        axis.set_title(f"u[{cell['u_start_px']},{cell['u_end_px']}), v[{cell['v_start_px']},{cell['v_end_px']}), λ[{cell['lambda_start_mm']},{cell['lambda_end_mm']})", fontsize=8)
        axis.set_ylabel("frame median residual / mm"); axis.grid(axis="y", alpha=0.2)
    for axis in axes_flat[len(supported):]: axis.axis("off")
    fig.tight_layout(); fig.savefig(out / "cross_frame_overlap_residuals.png", dpi=180); plt.close(fig)

    labels = ["sensor-position", "frame offset/tilt", "unexplained"]
    scopes = (("all 30", variance), ("leave 027 out", variance_leave027)); y_positions = np.arange(2)
    fig, axis = plt.subplots(figsize=(8, 4.8)); left = np.zeros(2)
    for label, key in zip(labels, ("sensor_allocated_fraction", "frame_allocated_fraction", "unexplained_fraction")):
        values = np.asarray([scope[key] for _, scope in scopes]); axis.barh(y_positions, values, left=left, label=label); left += values
    axis.set_xlim(0, 1); axis.set_yticks(y_positions, [name for name, _ in scopes]); axis.set_xlabel("allocated fraction"); axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22)); axis.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "residual_variance_decomposition.png", dpi=180); plt.close(fig)


def report_text(
    summary: Sequence[Mapping[str, Any]], correlations: Sequence[Mapping[str, Any]], overlap: Mapping[str, Any], overlap_leave027: Mapping[str, Any],
    variance: Mapping[str, Any], variance_leave027: Mapping[str, Any], decision: Mapping[str, Any], invalid: Mapping[str, int]
) -> str:
    target = next(row for row in summary if row["frame_id"] == TARGET_FRAME)
    non_target = [row for row in summary if row["frame_id"] != TARGET_FRAME]
    rmse_rank = sorted(summary, key=lambda row: float(row["rmse_mm"]), reverse=True)
    top_correlations = sorted(
        [row for row in correlations if row["scope"] == "leave027_out" and math.isfinite(number(row["spearman_rho"]))],
        key=lambda row: abs(float(row["spearman_rho"])), reverse=True,
    )[:10]
    lines = [
        "# Task 5C — Frame-dependent residual source audit", "",
        f"`FRAME_EFFECT_SOURCE = {decision['verdict']}`", "",
        "## Scope and definitions", "",
        "- FIT only: `001–018 + 025–036` (30 frames). Frame 027 is retained and explicitly marked.",
        "- Validation `019–024, 037–040` was not loaded into truth/residual/model evaluation.",
        "- Frozen model: Task 5A 29-frame Circular diagnostic model; no refit or correction.",
        "- Residual: `e_lambda = lambda_truth - lambda_model`.",
        "- Per-frame stripe coordinate is the dominant PCA direction of `(u,v)`, sign-fixed toward increasing v, zero-mean and normalized by maximum absolute projection.",
        f"- Sensor term: fixed {SENSOR_V_BIN_PX}px v-bin offset plus within-bin u slope. Overlap cells: {OVERLAP_U_BIN_PX}px × {OVERLAP_V_BIN_PX}px × {OVERLAP_LAMBDA_BIN_MM:g}mm.", "",
        "## 1. Frame offset / tilt", "",
        f"- 027: bias={target['bias_mm']:.4f} mm, RMSE={target['rmse_mm']:.4f} mm, P95={target['p95_abs_mm']:.4f} mm, a={target['a_frame_mm']:.4f} mm, k={target['k_frame_mm_per_s']:.4f} mm/s, offset+tilt explained={target['offset_tilt_explained_fraction']:.1%}.",
        f"- 027 RMSE rank: {rmse_rank.index(target) + 1}/{len(summary)} (1 is largest). Median non-027 RMSE={np.median([row['rmse_mm'] for row in non_target]):.4f} mm.",
        f"- Valid intersection losses total {sum(invalid.values())}; per-frame counts are retained in provenance.", "",
        "## 2. Pose and acquisition correlation", "",
        "Both Pearson and Spearman are in `frame_pose_correlation.csv`; scatter plots expose leverage and nonlinearity. A correlation is called robust only when |Spearman rho|≥0.60, p<0.01, sign-consistent in all-30 and leave-027-out scopes.", "",
        f"- Robust pose correlations: {decision['robust_pose_correlation_count']}.",
        f"- Robust acquisition/PnP/image-quality correlations: {decision['robust_acquisition_correlation_count']}.",
        "- Roll/pitch are explicitly normal-derived camera-axis tilt components, not an unobservable arbitrary in-plane board rotation.", "",
        "Strongest leave-027-out associations (diagnostic, uncorrected for multiple comparisons):", "",
        "| outcome | predictor | category | Spearman rho | p | Pearson r |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in top_correlations:
        lines.append(f"| {row['outcome']} | {row['predictor']} | {row['predictor_category']} | {float(row['spearman_rho']):.3f} | {float(row['spearman_p']):.3g} | {float(row['pearson_r']):.3f} |")
    lines += [
        "", "## 3. Cross-frame overlap", "",
        f"- Supported overlap cells: {overlap['overlap_cell_count']}, covering {overlap['covered_unique_frame_count']} frames.",
        f"- Weighted between-frame variance fraction within matched cells: {overlap['overlap_between_frame_fraction']:.1%}.",
        f"- Frame-median residual range per cell: median {overlap['median_frame_median_range_mm']:.4f} mm, P95 {overlap['p95_frame_median_range_mm']:.4f} mm.",
        f"- Leave-027-out sensitivity: {overlap_leave027['overlap_cell_count']} cells / {overlap_leave027['covered_unique_frame_count']} frames, between-frame fraction {overlap_leave027['overlap_between_frame_fraction']:.1%}, median/P95 range {overlap_leave027['median_frame_median_range_mm']:.4f}/{overlap_leave027['p95_frame_median_range_mm']:.4f} mm.",
        "- Cells are selected only by fixed coordinates; the comparison plot uses the four most-supported cells, not cells selected by residual magnitude.", "",
        "## 4. Variance decomposition", "",
        "| scope | sensor-only | frame-only | combined | allocated sensor | allocated frame | unexplained |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| all 30 | {variance['sensor_only_explained_fraction']:.1%} | {variance['frame_only_explained_fraction']:.1%} | {variance['combined_explained_fraction']:.1%} | {variance['sensor_allocated_fraction']:.1%} | {variance['frame_allocated_fraction']:.1%} | {variance['unexplained_fraction']:.1%} |",
        f"| leave 027 out | {variance_leave027['sensor_only_explained_fraction']:.1%} | {variance_leave027['frame_only_explained_fraction']:.1%} | {variance_leave027['combined_explained_fraction']:.1%} | {variance_leave027['sensor_allocated_fraction']:.1%} | {variance_leave027['frame_allocated_fraction']:.1%} | {variance_leave027['unexplained_fraction']:.1%} |", "",
        f"All-30 / leave-027 commonality interaction is {variance['shared_sensor_frame_fraction']:.1%} / {variance_leave027['shared_sensor_frame_fraction']:.1%}. A negative value denotes suppressor/synergy from non-orthogonal sensor and frame designs; it is not negative physical energy. Shapley allocation averages both entry orders. This is an in-sample diagnostic partition, not a deployable model.", "",
        "## Conclusion and next step", "",
        f"`FRAME_EFFECT_SOURCE = {decision['verdict']}`", "",
    ]
    if decision["verdict"].startswith("A"):
        lines.append("Next: repeat selected poses and vary one pose variable at a time; verify whether the same pose-residual relation repeats before changing the laser model.")
    elif decision["verdict"].startswith("B"):
        lines.append("Next: acquire short-repeat triplets at identical board poses with an independently trackable board reference during laser exposure. This directly separates inter-image board motion/timing effects from fixed sensor geometry.")
    elif decision["verdict"].startswith("C"):
        lines.append("Next: freeze a low-DOF sensor-coordinate diagnostic candidate and test it once on untouched validation; do not add a frame term because it is unavailable at runtime.")
    elif decision["verdict"].startswith("D"):
        lines.append("Next: run repeated triplets at a small factorial set of board depth/tilt and sensor-edge locations. The same-pose repeats identify acquisition/frame variance, while crossed sensor locations identify the fixed component; keep validation untouched until a single hypothesis is frozen.")
    else:
        lines.append("Next: acquire more repeated overlap at identical `(u,v,lambda)` support before selecting a source hypothesis.")
    lines += ["", "No validation, refit, frame deletion, complex surface, or correction was used.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    formal_hash_before = sha256_file(task3b2.FORMAL_CONE)
    if formal_hash_before != task3b2.EXPECTED_CONE_SHA256:
        raise RuntimeError("Formal M0 hash mismatch")
    provenance5a = json.loads(TASK5A_PROVENANCE.read_text(encoding="utf-8"))
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records, extension_geometry, extension_provenance = load_fit_records(intrinsics)
    model = current_model(provenance5a, records)
    points, invalid = collect_residual_points(records, calibration, reconstruction_params, model)
    metadata = fit_metadata_rows(extension_geometry, extension_provenance)
    summary = frame_summary(points, metadata); correlations = correlation_rows(summary)
    overlap, overlap_summary = overlap_rows(points)
    points_leave027 = [row for row in points if row["frame_id"] != TARGET_FRAME]
    overlap_leave027, overlap_summary_leave027 = overlap_rows(points_leave027)
    variance = variance_decomposition(points); variance_leave027 = variance_decomposition(points_leave027)
    decision = classify(correlations, overlap_summary, overlap_summary_leave027, variance, variance_leave027)
    write_csv(out / "frame_residual_summary.csv", summary)
    write_csv(out / "frame_pose_correlation.csv", correlations)
    write_csv(out / "cross_frame_overlap.csv", [{"scope": "all30", **row} for row in overlap] + [{"scope": "leave027_out", **row} for row in overlap_leave027])
    make_plots(out, summary, overlap, variance, variance_leave027)
    formal_hash_after = sha256_file(task3b2.FORMAL_CONE)
    if formal_hash_after != formal_hash_before:
        raise RuntimeError("Formal M0 changed during read-only audit")
    provenance = {
        "task": "Task 5C frame-dependent residual source audit", "fit_ids": task3a.FIT_IDS,
        "frame027_retained": True, "validation_ids_not_opened": task3a.VALIDATION_IDS, "validation_opened": False,
        "frozen_model_source": str(TASK5A_PROVENANCE), "model_refit": False, "correction_created": False,
        "formal_cone_sha256_before": formal_hash_before, "formal_cone_sha256_after": formal_hash_after,
        "invalid_intersections_by_frame": invalid, "overlap_summary": {"all30": overlap_summary, "leave027_out": overlap_summary_leave027},
        "variance_decomposition": {"all30": variance, "leave027_out": variance_leave027}, "decision": decision,
        "classification_gates": {
            "overlap_sufficient": "at least 30 cells and 10 covered frames", "overlap_inconsistent": "between-frame fraction >= 0.50",
            "robust_correlation": "abs(Spearman)>=0.60, p<0.01, same sign all30 and leave027-out",
            "mostly_sensor_fixed": "sensor allocated >=0.60, frame <0.25, overlap not inconsistent",
            "cross_frame_inconsistency": "both overlap scopes inconsistent, leave027 frame >=0.35, no robust pose/acquisition, leave027 sensor <0.20",
            "mixed": "substantial secondary sensor allocation (>=0.20) or robust pose/acquisition association coexists with cross-frame inconsistency",
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(report_text(summary, correlations, overlap_summary, overlap_summary_leave027, variance, variance_leave027, decision, invalid), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "decision": decision, "overlap": {"all30": overlap_summary, "leave027_out": overlap_summary_leave027}, "variance": {"all30": variance, "leave027_out": variance_leave027}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
