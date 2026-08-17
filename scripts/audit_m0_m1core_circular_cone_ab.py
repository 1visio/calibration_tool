#!/usr/bin/env python3
"""Task 6I: end-to-end M0 versus M1-core Circular Cone A/B.

The script keeps the two camera candidates separate from the formal files.  It
freezes laser-center pixels once (M0 extraction for the extension data),
recomputes full-board PnP/ray-plane truth for each K/D, fits the same
production CircularConeModel objective to FIT, and only then opens the frozen
Validation holdout for a final A/B measurement.
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
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT = Path(__file__).resolve()
TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
for path in (SCRIPT.parent, WORKSPACE_ROOT / "calibration" / "src", MEASUREMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_augmented_camera_calibration_stability as aug  # noqa: E402
import audit_intrinsics_truth_stability as task6e  # noqa: E402
import audit_triplet_edge_extension_observability as edge_ext  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
import fit_laser_models_from_triplets as triplets  # noqa: E402


DEFAULT_OUTPUT = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0817" / "m0_m1core_circular_cone_ab"
DEFAULT_FORMAL_INTRINSICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
DEFAULT_FORMAL_FIT_METRICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "fit_images.csv"
DEFAULT_MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
DEFAULT_FORMAL_FIT_CONFIG = TOOL_ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_OLD_POINTS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "laser_model" / "calibration_points.csv"
DEFAULT_OLD_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
DEFAULT_OLD_VALIDATION = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "validation"
DEFAULT_EXTENSION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension"
DEFAULT_EXTENSION_VALIDATION = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "validation_edge_holdout"
DEFAULT_FORMAL_CONE = MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"

FIT_IDS = tuple(f"{i:03d}" for i in list(range(1, 19)) + list(range(25, 37)))
OLD_FIT_IDS = tuple(f"{i:03d}" for i in range(1, 19))
EXT_FIT_IDS = tuple(f"{i:03d}" for i in range(25, 37))
VALIDATION_IDS = tuple(f"{i:03d}" for i in list(range(19, 25)) + list(range(37, 41)))
OLD_VALIDATION_IDS = tuple(f"{i:03d}" for i in range(19, 25))
EXT_VALIDATION_IDS = tuple(f"{i:03d}" for i in range(37, 41))
REGIONS = (("global", -math.inf, math.inf), ("top", 0.0, 300.0), ("middle", 300.0, 2700.0), ("bottom", 2700.0, 3000.0))
V_BIN_WIDTH = 60.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--formal-intrinsics", type=Path, default=DEFAULT_FORMAL_INTRINSICS)
    p.add_argument("--formal-fit-metrics", type=Path, default=DEFAULT_FORMAL_FIT_METRICS)
    p.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    p.add_argument("--formal-fit-config", type=Path, default=DEFAULT_FORMAL_FIT_CONFIG)
    p.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    p.add_argument("--old-points", type=Path, default=DEFAULT_OLD_POINTS)
    p.add_argument("--old-fit", type=Path, default=DEFAULT_OLD_FIT)
    p.add_argument("--old-validation", type=Path, default=DEFAULT_OLD_VALIDATION)
    p.add_argument("--extension-fit", type=Path, default=DEFAULT_EXTENSION_FIT)
    p.add_argument("--extension-validation", type=Path, default=DEFAULT_EXTENSION_VALIDATION)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def metric(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0, "bias_mm": math.nan, "mae_mm": math.nan, "rmse_mm": math.nan, "p95_abs_mm": math.nan, "max_abs_mm": math.nan}
    return {
        "count": int(len(x)),
        "bias_mm": float(np.mean(x)),
        "mae_mm": float(np.mean(np.abs(x))),
        "rmse_mm": float(np.sqrt(np.mean(x * x))),
        "p95_abs_mm": float(np.percentile(np.abs(x), 95)),
        "max_abs_mm": float(np.max(np.abs(x))),
    }


def region_for_v(v: float) -> str:
    if v < 300.0:
        return "top"
    if v < 2700.0:
        return "middle"
    return "bottom"


def load_old_uv(path: Path, ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Read only the requested persisted UV rows.

    FIT is loaded before model fitting.  Validation IDs are opened by a
    separate call after both candidates have been frozen.
    """
    wanted = {str(x).zfill(3) for x in ids}
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = f"{int(float(row['image_id'])):03d}"
            if frame_id in wanted:
                groups[frame_id].append((float(row["u_px"]), float(row["v_px"])))
    missing = sorted(wanted - set(groups), key=int)
    if missing:
        raise RuntimeError(f"Missing persisted UV for {missing} in {path}")
    return {key: np.asarray(value, dtype=np.float64) for key, value in groups.items()}


def find_image(root: Path, frame_id: str, role: str = "chess") -> Path:
    matches = sorted(root.rglob(f"{role} {frame_id}.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {role} {frame_id}.tif under {root}, found {matches}")
    return matches[0]


def detect_corners(path: Path) -> tuple[np.ndarray, str]:
    image = task6e.chess_calib.read_image(path)
    if image is None:
        raise RuntimeError(f"Could not read chess image {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners, method = task6e.chess_calib.detect_corners(gray, (11, 8))
    if not found or corners is None:
        raise RuntimeError(f"Chessboard detection failed: {path}")
    return np.asarray(corners, dtype=np.float32).reshape(-1, 2), str(method)


def calibration_observation_maps(
    baseline_fit: Path,
    extension_fit: Path,
    ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], tuple[int, int]]:
    baseline_ids = [x for x in ids if int(x) <= 18]
    extension_ids = [x for x in ids if int(x) >= 25]
    # The extension dataset keeps its image files under ``fit/`` while the
    # legacy loader expects the directory containing ``chess NNN.tif``.
    extension_image_root = extension_fit / "fit" if (extension_fit / "fit").is_dir() else extension_fit
    observations, image_size = aug.load_dataset_observations(ids, baseline_fit, extension_image_root, {})
    result = {str(item["frame_id"]): item for item in observations}
    if sorted(result, key=int) != sorted(ids, key=int):
        raise RuntimeError(f"Calibration observation order mismatch: expected={ids}, got={sorted(result, key=int)}")
    return result, image_size


def extract_extension_uv(root: Path, intrinsics: tuple[np.ndarray, np.ndarray], split: str) -> dict[str, np.ndarray]:
    holder = SimpleNamespace(camera_matrix=np.asarray(intrinsics[0], dtype=np.float64), dist_coeffs=np.asarray(intrinsics[1], dtype=np.float64).reshape(-1))
    points, _geometry, _provenance = edge_ext.extract_extension(root, split, holder, True)
    return {frame_id: np.column_stack([item["u"], item["v"]]).astype(np.float64) for frame_id, item in points.items()}


def pnp_truth_records(
    ids: Sequence[str],
    uv_map: Mapping[str, np.ndarray],
    corners_map: Mapping[str, Mapping[str, Any]],
    k: np.ndarray,
    d: np.ndarray,
    obj: np.ndarray,
    image_roots: Mapping[str, Path],
    split: str,
) -> tuple[list[task3a.FrameRecord], dict[str, dict[str, Any]]]:
    records: list[task3a.FrameRecord] = []
    geometry: dict[str, dict[str, Any]] = {}
    for frame_id in ids:
        if frame_id not in uv_map:
            raise RuntimeError(f"No frozen UV for {frame_id}")
        obs = corners_map.get(frame_id)
        if obs is None:
            path = find_image(image_roots[frame_id], frame_id, "chess")
            corners, method = detect_corners(path)
            obs = {"frame_id": frame_id, "path": path, "corners": corners, "detection_method": method}
        corners = np.asarray(obs["corners"], dtype=np.float32).reshape(-1, 2)
        pose = task6e.solve_pose(corners, k, d, obj)
        uv = np.asarray(uv_map[frame_id], dtype=np.float64)
        if len(uv) < 80:
            raise RuntimeError(f"Frame {frame_id} has too few frozen laser centers: {len(uv)}")
        normalized = cv2.undistortPoints(uv.reshape(-1, 1, 2), k, d).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(uv), dtype=np.float64)])
        denom = rays @ np.asarray(pose["normal"], dtype=np.float64)
        lam = -float(pose["d"]) / denom
        valid = np.isfinite(lam) & (lam > 0.0)
        if not np.all(valid):
            raise RuntimeError(f"Invalid ray-plane truth for {frame_id}: {np.count_nonzero(~valid)} invalid")
        truth = rays * lam[:, None]
        records.append(
            task3a.FrameRecord(
                frame_id=frame_id,
                split=split,
                pixels_uv=uv,
                truth_points=truth,
                plane=np.asarray([*pose["normal"], pose["d"]], dtype=np.float64),
                pnp_rmse_px=float(pose["rmse_px"]),
                point_count=len(uv),
                quality_warnings="",
                source=str(obs.get("path", "")),
            )
        )
        normal = np.asarray(pose["normal"], dtype=np.float64)
        geometry[frame_id] = {
            "pnp_rmse_px": float(pose["rmse_px"]),
            "plane_nx": float(normal[0]),
            "plane_ny": float(normal[1]),
            "plane_nz": float(normal[2]),
            "plane_d_mm": float(pose["d"]),
            "board_tilt_deg": float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))),
            "point_count": int(len(uv)),
            "image": str(obs.get("path", "")),
        }
    return records, geometry


def fit_cone(records: Sequence[task3a.FrameRecord], cone_cfg: Mapping[str, Any], initial_model: Mapping[str, Any]) -> dict[str, Any]:
    result = task3a.formal_fit(records, dict(cone_cfg), initial_model)
    if not bool(result.get("fit_success")):
        raise RuntimeError(f"Circular Cone fit failed: {result.get('status')} {result.get('optimizer_message')}")
    return result


def candidate_calibration(base: Mapping[str, Any], k: np.ndarray, d: np.ndarray, model: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    result["K"] = np.asarray(k, dtype=np.float64)
    result["D"] = np.asarray(d, dtype=np.float64).reshape(-1)
    result["laser_model"] = copy.deepcopy(dict(model))
    return result


def evaluate_records(
    records: Sequence[task3a.FrameRecord],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    model: Mapping[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    z_range = model.get("z_valid_range_mm")
    for record in records:
        lambda_model, valid = task3a.production_lambda(record.pixels_uv, model, calibration, reconstruction_params, z_range)
        truth = np.asarray(record.truth_points[:, 2], dtype=np.float64)
        residual = np.full(len(truth), np.nan, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(truth) & np.isfinite(lambda_model)
        residual[valid] = truth[valid] - lambda_model[valid]
        output[record.frame_id] = {"lambda_model": lambda_model, "lambda_truth": truth, "residual": residual, "valid": valid}
    return output


def region_rows(records: Sequence[task3a.FrameRecord], evaluation: Mapping[str, Mapping[str, np.ndarray]], candidate: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region, low, high in REGIONS:
        values: list[np.ndarray] = []
        frame_count = 0
        invalid = 0
        for record in records:
            item = evaluation[record.frame_id]
            v = record.pixels_uv[:, 1]
            mask = (v >= low) & (v < high) & item["valid"]
            region_all = (v >= low) & (v < high)
            invalid += int(np.count_nonzero(region_all & ~item["valid"]))
            if np.any(mask):
                values.append(item["residual"][mask])
                frame_count += 1
        stats = metric(np.concatenate(values) if values else np.empty(0))
        rows.append({"row_type": "aggregate", "split": split, "candidate": candidate, "weighting": "point_equal", "region": region, "frame_count": frame_count, "invalid_count": invalid, **stats})
    return rows


def frame_metric_rows(records: Sequence[task3a.FrameRecord], evaluation: Mapping[str, Mapping[str, np.ndarray]], candidate: str, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    region_rows_out: list[dict[str, Any]] = []
    for record in records:
        item = evaluation[record.frame_id]
        stats = metric(item["residual"][item["valid"]])
        frame_rows.append({"row_type": "frame", "split": split, "candidate": candidate, "frame_id": record.frame_id, "pnp_rmse_px": record.pnp_rmse_px, "point_count": int(np.count_nonzero(item["valid"])), **stats})
        for region, low, high in REGIONS[1:]:
            v = record.pixels_uv[:, 1]
            mask = (v >= low) & (v < high) & item["valid"]
            rstats = metric(item["residual"][mask])
            region_rows_out.append({"row_type": "frame_region", "split": split, "candidate": candidate, "frame_id": record.frame_id, "region": region, **rstats})
    return frame_rows, region_rows_out


def truth_difference_rows(
    m0: Sequence[task3a.FrameRecord],
    m1: Sequence[task3a.FrameRecord],
    geom0: Mapping[str, Mapping[str, Any]],
    geom1: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by1 = {r.frame_id: r for r in m1}
    rows: list[dict[str, Any]] = []
    summary_inputs: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    for r0 in m0:
        r1 = by1[r0.frame_id]
        if len(r0.pixels_uv) != len(r1.pixels_uv) or not np.allclose(r0.pixels_uv, r1.pixels_uv, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Frozen UV mismatch for {r0.frame_id}")
        delta = r1.truth_points - r0.truth_points
        v = r0.pixels_uv[:, 1]
        for idx in range(len(v)):
            region = region_for_v(float(v[idx]))
            bin_start = math.floor(float(v[idx]) / V_BIN_WIDTH) * int(V_BIN_WIDTH)
            rows.append({
                "row_type": "point",
                "frame_id": r0.frame_id,
                "is_frame027": r0.frame_id == "027",
                "u_px": float(r0.pixels_uv[idx, 0]),
                "v_px": float(v[idx]),
                "v_bin_start_px": bin_start,
                "region": region,
                "board_tilt_m0_deg": geom0[r0.frame_id]["board_tilt_deg"],
                "board_tilt_m1_deg": geom1[r0.frame_id]["board_tilt_deg"],
                "lambda_m0_mm": float(r0.truth_points[idx, 2]),
                "lambda_m1_mm": float(r1.truth_points[idx, 2]),
                "delta_lambda_mm": float(delta[idx, 2]),
                "delta_x_mm": float(delta[idx, 0]),
                "delta_y_mm": float(delta[idx, 1]),
                "delta_z_mm": float(delta[idx, 2]),
            })
            summary_inputs[(r0.frame_id, region, str(bin_start))].append(delta[idx])
        for region, low, high in REGIONS:
            mask = (v >= low) & (v < high)
            d = delta[mask]
            if len(d):
                stats = metric(d[:, 2])
                for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                    astats = metric(d[:, index])
                    stats[f"bias_delta_{axis}_mm"] = astats["bias_mm"]
                    stats[f"rmse_delta_{axis}_mm"] = astats["rmse_mm"]
                rows.append({"row_type": "frame_region_summary", "frame_id": r0.frame_id, "is_frame027": r0.frame_id == "027", "region": region, "board_tilt_m0_deg": geom0[r0.frame_id]["board_tilt_deg"], "board_tilt_m1_deg": geom1[r0.frame_id]["board_tilt_deg"], **{f"delta_lambda_{key}": value for key, value in stats.items()}})
    for (frame_id, region, bin_start), values in sorted(summary_inputs.items(), key=lambda item: (int(item[0][0]), item[0][1], float(item[0][2]))):
        d = np.asarray(values, dtype=np.float64)
        stats = metric(d[:, 2])
        rows.append({"row_type": "frame_v_bin_summary", "frame_id": frame_id, "is_frame027": frame_id == "027", "region": region, "v_bin_start_px": bin_start, "sample_count": len(d), "delta_lambda_bias_mm": stats["bias_mm"], "delta_lambda_rmse_mm": stats["rmse_mm"], "delta_lambda_p95_abs_mm": stats["p95_abs_mm"], "delta_lambda_max_abs_mm": stats["max_abs_mm"]})
    return rows


def fit_summary_rows(
    fits: Mapping[str, Mapping[str, Any]],
    fit_evals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    fit_records: Mapping[str, Sequence[task3a.FrameRecord]],
    grid_uv: np.ndarray,
    grid_v: np.ndarray,
    grid_evals: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, fit in fits.items():
        model = fit["model_dict"]
        rows.append({"row_type": "candidate", "candidate": candidate, "fit_success": fit["fit_success"], "fit_status": fit["status"], "optimizer_cost": fit["optimizer_cost"], "objective_mse": fit["objective_mse"], "selected_point_count": len(fit["selected_points"]), "z_min_mm": fit["z_range_mm"][0], "z_max_mm": fit["z_range_mm"][1], "axis_x": model["axis_unit_camera"][0], "axis_y": model["axis_unit_camera"][1], "axis_z": model["axis_unit_camera"][2], "apex_x_mm": model["apex_camera_mm"][0], "apex_y_mm": model["apex_camera_mm"][1], "apex_z_mm": model["apex_camera_mm"][2], "half_apex_angle_deg": model["half_apex_angle_deg"]})
        # ``region_rows`` labels these records as aggregate metrics; retain a
        # distinct fit-region type in the combined summary so it cannot be
        # confused with the later Validation aggregates.
        rows.extend({"candidate": candidate, "split": "fit", **row, "row_type": "fit_region"} for row in region_rows(fit_records[candidate], fit_evals[candidate], candidate, "fit"))
    lambda0, valid0 = grid_evals["M0"]
    lambda1, valid1 = grid_evals["M1-core"]
    common = valid0 & valid1 & np.isfinite(lambda0) & np.isfinite(lambda1)
    delta = lambda1 - lambda0
    for region, low, high in REGIONS:
        mask = common & (grid_v >= low) & (grid_v < high)
        stats = metric(delta[mask])
        rows.append({"row_type": "grid_prediction_difference", "candidate": "M1-core_minus_M0", "region": region, "grid_count": int(np.count_nonzero(mask)), **stats})
    return rows


def cone_parameter_delta(m0: Mapping[str, Any], m1: Mapping[str, Any]) -> dict[str, float]:
    a0 = np.asarray(m0["axis_unit_camera"], dtype=np.float64)
    a1 = np.asarray(m1["axis_unit_camera"], dtype=np.float64)
    apex0 = np.asarray(m0["apex_camera_mm"], dtype=np.float64)
    apex1 = np.asarray(m1["apex_camera_mm"], dtype=np.float64)
    return {"axis_angle_deg": float(np.degrees(np.arccos(np.clip(abs(float(a0 @ a1)), 0.0, 1.0)))), "apex_delta_mm": float(np.linalg.norm(apex1 - apex0)), "half_apex_angle_delta_deg": float(m1["half_apex_angle_deg"] - m0["half_apex_angle_deg"])}


def plot_validation(output: Path, records: Sequence[task3a.FrameRecord], evaluations: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]], frame_rows: Sequence[Mapping[str, Any]], grid_delta: tuple[np.ndarray, np.ndarray]) -> None:
    colors = {"M0": "#2563eb", "M1-core": "#dc2626"}
    plt.figure(figsize=(10, 5))
    for candidate in ("M0", "M1-core"):
        v = np.concatenate([r.pixels_uv[:, 1] for r in records])
        e = np.concatenate([evaluations[candidate][r.frame_id]["residual"] for r in records])
        plt.scatter(v, e, s=3, alpha=0.2, label=candidate, color=colors[candidate])
    plt.axhline(0.0, color="black", lw=0.8); plt.xlabel("sensor v / px"); plt.ylabel("truth - cone lambda / mm"); plt.legend(); plt.tight_layout(); plt.savefig(output / "validation_residual_vs_v.png", dpi=160); plt.close()

    region_names = ["top", "middle", "bottom"]
    by = {(row["candidate"], row["region"]): float(row["rmse_mm"]) for row in frame_rows if row.get("row_type") == "aggregate" and row.get("region") in region_names}
    x = np.arange(len(region_names)); width = 0.36
    plt.figure(figsize=(8, 5)); plt.bar(x - width / 2, [by.get(("M0", r), math.nan) for r in region_names], width, label="M0"); plt.bar(x + width / 2, [by.get(("M1-core", r), math.nan) for r in region_names], width, label="M1-core"); plt.xticks(x, region_names); plt.ylabel("RMSE / mm"); plt.legend(); plt.tight_layout(); plt.savefig(output / "validation_region_rmse.png", dpi=160); plt.close()

    plt.figure(figsize=(10, 5))
    for candidate in ("M0", "M1-core"):
        vals = []
        centers = []
        for start in np.arange(0.0, 3000.0, V_BIN_WIDTH):
            xvals = []
            for record in records:
                mask = (record.pixels_uv[:, 1] >= start) & (record.pixels_uv[:, 1] < start + V_BIN_WIDTH)
                xvals.extend(evaluations[candidate][record.frame_id]["residual"][mask & evaluations[candidate][record.frame_id]["valid"]].tolist())
            centers.append(start + V_BIN_WIDTH / 2); vals.append(float(np.mean(xvals)) if xvals else math.nan)
        plt.plot(centers, vals, marker=".", label=candidate, color=colors[candidate])
    plt.axhline(0.0, color="black", lw=0.8); plt.xlabel("sensor v / px"); plt.ylabel("bias / mm"); plt.legend(); plt.tight_layout(); plt.savefig(output / "validation_bias_vs_v.png", dpi=160); plt.close()

    delta, valid = grid_delta
    vv = np.linspace(task3a.FORMAL_V_MIN, task3a.FORMAL_V_MAX, task3a.GRID_V_COUNT)
    grid_v = np.repeat(vv[:, None], task3a.GRID_U_COUNT, axis=1).ravel()
    plt.figure(figsize=(10, 5)); plt.scatter(grid_v[valid], delta[valid], s=6, alpha=0.45); plt.axhline(0.0, color="black", lw=0.8); plt.xlabel("evaluation-grid v / px"); plt.ylabel("M1-core - M0 cone lambda / mm"); plt.tight_layout(); plt.savefig(output / "cone_prediction_difference_vs_v.png", dpi=160); plt.close()

    frame_ids = sorted({str(row["frame_id"]) for row in frame_rows if row.get("row_type") == "frame"}, key=int)
    x = np.arange(len(frame_ids)); plt.figure(figsize=(12, 5))
    for candidate, offset, color in (("M0", -0.18, colors["M0"]), ("M1-core", 0.18, colors["M1-core"])):
        vals = {str(row["frame_id"]): float(row["rmse_mm"]) for row in frame_rows if row.get("row_type") == "frame" and row.get("candidate") == candidate}
        plt.scatter(x + offset, [vals.get(fid, math.nan) for fid in frame_ids], label=candidate, color=color)
    plt.xticks(x, frame_ids, rotation=60); plt.ylabel("frame RMSE / mm"); plt.legend(); plt.tight_layout(); plt.savefig(output / "validation_frame_rmse.png", dpi=160); plt.close()


def classify_camera_effect(fit_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], frame_rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    lookup = {(str(r.get("candidate")), str(r.get("region"))): r for r in validation_rows if r.get("row_type") == "aggregate"}
    def value(candidate: str, region: str, key: str) -> float:
        return finite(lookup.get((candidate, region), {}).get(key))
    improvements = {}
    for region in ("global", "top", "middle", "bottom"):
        a, b = value("M0", region, "rmse_mm"), value("M1-core", region, "rmse_mm")
        improvements[f"{region}_rmse_improvement_fraction"] = (a - b) / a if math.isfinite(a) and a > 0 else math.nan
    edge0 = np.nanmean([value("M0", "top", "rmse_mm"), value("M0", "bottom", "rmse_mm")])
    edge1 = np.nanmean([value("M1-core", "top", "rmse_mm"), value("M1-core", "bottom", "rmse_mm")])
    middle0, middle1 = value("M0", "middle", "rmse_mm"), value("M1-core", "middle", "rmse_mm")
    ratio0, ratio1 = edge0 / middle0 if middle0 > 0 else math.nan, edge1 / middle1 if middle1 > 0 else math.nan
    asym0 = abs(value("M0", "top", "bias_mm") - value("M0", "bottom", "bias_mm")); asym1 = abs(value("M1-core", "top", "bias_mm") - value("M1-core", "bottom", "bias_mm"))
    details = {**improvements, "edge_rmse_improvement_fraction": (edge0 - edge1) / edge0 if edge0 > 0 else math.nan, "edge_middle_ratio_m0": ratio0, "edge_middle_ratio_m1": ratio1, "edge_middle_ratio_change_fraction": (ratio0 - ratio1) / ratio0 if ratio0 > 0 else math.nan, "bias_asymmetry_m0_mm": asym0, "bias_asymmetry_m1_mm": asym1, "bias_asymmetry_reduction_fraction": (asym0 - asym1) / asym0 if asym0 > 0 else math.nan}
    top = details["top_rmse_improvement_fraction"]; bottom = details["bottom_rmse_improvement_fraction"]; global_imp = details["global_rmse_improvement_fraction"]; edge_imp = details["edge_rmse_improvement_fraction"]
    if all(math.isfinite(x) for x in (top, bottom, global_imp, edge_imp)) and top >= 0.20 and bottom >= 0.10 and global_imp >= 0.10 and details["edge_middle_ratio_change_fraction"] >= 0.15 and details["bias_asymmetry_reduction_fraction"] >= 0.20:
        verdict = "A. STRONG"
    elif any(math.isfinite(x) and x >= 0.05 for x in (global_imp, top, bottom, edge_imp)) and not (math.isfinite(global_imp) and global_imp < -0.05):
        verdict = "B. MODERATE"
    elif math.isfinite(global_imp) and global_imp < -0.05:
        verdict = "D. NEGATIVE"
    else:
        verdict = "C. WEAK"
    details["verdict"] = verdict
    details["frame_rmse_median_m0_mm"] = float(np.median([float(r["rmse_mm"]) for r in frame_rows if r.get("row_type") == "frame" and r.get("candidate") == "M0"]))
    details["frame_rmse_median_m1_mm"] = float(np.median([float(r["rmse_mm"]) for r in frame_rows if r.get("row_type") == "frame" and r.get("candidate") == "M1-core"]))
    return verdict, details


def render_report(
    output: Path,
    fit_summary: Sequence[Mapping[str, Any]],
    fit_regions: Sequence[Mapping[str, Any]],
    val_regions: Sequence[Mapping[str, Any]],
    val_frames: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    parameter_delta: Mapping[str, Any],
    classification: str,
    attribution: Mapping[str, Any],
) -> None:
    def fmt(v: Any) -> str:
        x = finite(v)
        return "n/a" if not math.isfinite(x) else f"{x:.6g}"
    lines = [
        "# Task 6I - M0 vs M1-core Circular Cone end-to-end A/B",
        "",
        f"`CAMERA_CALIBRATION_EDGE_CAUSAL_EFFECT = {classification}`",
        "",
        "FIT candidates were fitted independently with the same production CircularConeModel path. M0/M1-core are diagnostic candidates only; the formal K/D and Cone files were not overwritten.",
        "Validation was opened only after both FIT candidates completed. No 0815 laser/nolaser data were used.",
        "Residual convention: `e_lambda = lambda_truth - lambda_model`.",
        "",
        "## FIT Cone candidates",
        "",
        "| candidate | cost | objective MSE | selected points | alpha deg | apex delta vs M0 | axis delta deg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fit_summary:
        if row.get("row_type") == "candidate":
            candidate = str(row["candidate"])
            lines.append(f"| {candidate} | {fmt(row.get('optimizer_cost'))} | {fmt(row.get('objective_mse'))} | {row.get('selected_point_count')} | {fmt(row.get('half_apex_angle_deg'))} | {fmt(parameter_delta.get('apex_delta_mm') if candidate == 'M1-core' else 0.0)} | {fmt(parameter_delta.get('axis_angle_deg') if candidate == 'M1-core' else 0.0)} |")
    lines += ["", "## FIT region metrics", "", "| candidate | region | bias mm | RMSE mm | P95 mm | max mm |", "|---|---|---:|---:|---:|---:|"]
    for row in fit_regions:
        if row.get("row_type") in ("aggregate", "fit_region"):
            lines.append(f"| {row.get('candidate')} | {row.get('region')} | {fmt(row.get('bias_mm'))} | {fmt(row.get('rmse_mm'))} | {fmt(row.get('p95_abs_mm'))} | {fmt(row.get('max_abs_mm'))} |")
    lines += ["", "## Frozen Validation A/B", "", "| candidate | region | bias mm | MAE mm | RMSE mm | P95 mm | max mm |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in val_regions:
        if row.get("row_type") == "aggregate":
            lines.append(f"| {row.get('candidate')} | {row.get('region')} | {fmt(row.get('bias_mm'))} | {fmt(row.get('mae_mm'))} | {fmt(row.get('rmse_mm'))} | {fmt(row.get('p95_abs_mm'))} | {fmt(row.get('max_abs_mm'))} |")
    lines += ["", "## Camera-change attribution", "", f"- global RMSE improvement: `{fmt(attribution.get('global_rmse_improvement_fraction'))}`", f"- top RMSE improvement: `{fmt(attribution.get('top_rmse_improvement_fraction'))}`", f"- bottom RMSE improvement: `{fmt(attribution.get('bottom_rmse_improvement_fraction'))}`", f"- edge/middle ratio: `{fmt(attribution.get('edge_middle_ratio_m0'))}` -> `{fmt(attribution.get('edge_middle_ratio_m1'))}`", f"- top-bottom bias asymmetry: `{fmt(attribution.get('bias_asymmetry_m0_mm'))}` -> `{fmt(attribution.get('bias_asymmetry_m1_mm'))}` mm", "", "## 027 diagnostic", "", "027 remains in FIT and is reported separately. See `frame027_camera_ab.csv`; no frame was deleted or reweighted after seeing Validation.", "", "## Controls", "", "- Same Steger extraction settings and frozen UV per frame for both K/D candidates.", "- Same full-board PnP solver, Circular Cone parameterization, frame balancing, soft_l1 loss, bounds, optimizer and formal v-domain.", "- No Elliptical Cone, quadric, correction, LUT, v compensation, or production writeback.", "", "## Outputs", "", "- `m0_m1core_truth_difference.csv`", "- `m0_m1core_cone_fit_summary.csv`", "- `m0_m1core_fit_region_metrics.csv`", "- `m0_m1core_validation_metrics.csv`", "- `m0_m1core_validation_by_frame.csv`", "- `m0_m1core_validation_by_region.csv`", "- `frame027_camera_ab.csv`", "- `Cone_M0_AB.yaml`", "- `Cone_M1core_AB.yaml`"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    formal_k, formal_d, formal_document = task6e.load_formal_intrinsics(args.formal_intrinsics.resolve())
    obj = task6e.object_points()
    formal_fit_metrics = task6e.load_formal_fit_metrics(args.formal_fit_metrics.resolve())
    _app, formal_calibration, reconstruction_params, runtime_intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    if not np.allclose(runtime_intrinsics.camera_matrix, formal_k, rtol=0.0, atol=1.0e-8):
        raise RuntimeError("Formal runtime K differs from formal intrinsics YAML")
    if not np.allclose(np.asarray(runtime_intrinsics.dist_coeffs).reshape(-1), formal_d.reshape(-1), rtol=0.0, atol=1.0e-8):
        raise RuntimeError("Formal runtime D differs from formal intrinsics YAML")

    # FIT-only stage: no Validation image or validation UV is touched here.
    calibration_obs, image_size = calibration_observation_maps(args.old_fit.resolve(), args.extension_fit.resolve(), tuple(OLD_FIT_IDS + ("026", "027", "028", "035")))
    m1_summary, m1_k, m1_d = aug.solve_dataset("M1-core", list(calibration_obs.values()), obj, image_size, formal_k, formal_d)
    m1_frame_rows = aug.per_frame_rmse("M1-core", list(calibration_obs.values()), m1_k, m1_d, obj)
    old_fit_uv = load_old_uv(args.old_points.resolve(), OLD_FIT_IDS)
    ext_fit_uv = extract_extension_uv(args.extension_fit.resolve(), (formal_k, formal_d), "fit")
    uv_fit = {**old_fit_uv, **ext_fit_uv}
    corner_fit = {key: value for key, value in calibration_obs.items()}
    fit_roots = {frame_id: (args.old_fit.resolve() if int(frame_id) <= 18 else args.extension_fit.resolve()) for frame_id in FIT_IDS}
    m0_records, geom0 = pnp_truth_records(FIT_IDS, uv_fit, corner_fit, formal_k, formal_d, obj, fit_roots, "fit")
    m1_records, geom1 = pnp_truth_records(FIT_IDS, uv_fit, corner_fit, m1_k, m1_d, obj, fit_roots, "fit")

    cfg = dict(triplets.safe_yaml_load(args.formal_fit_config.resolve())["models"]["cone"])
    formal_model = copy.deepcopy(formal_calibration["laser_model"])
    fit_m0 = fit_cone(m0_records, cfg, formal_model)
    fit_m1 = fit_cone(m1_records, cfg, formal_model)
    fits = {"M0": fit_m0, "M1-core": fit_m1}
    for name, fit in fits.items():
        (output / ("Cone_M0_AB.yaml" if name == "M0" else "Cone_M1core_AB.yaml")).write_text(yaml.safe_dump(fit["model_dict"], sort_keys=False, allow_unicode=True), encoding="utf-8")

    base_m0_cal = candidate_calibration(formal_calibration, formal_k, formal_d, fit_m0["model_dict"])
    base_m1_cal = candidate_calibration(formal_calibration, m1_k, m1_d, fit_m1["model_dict"])
    fit_evals = {"M0": evaluate_records(m0_records, base_m0_cal, reconstruction_params, fit_m0["model_dict"]), "M1-core": evaluate_records(m1_records, base_m1_cal, reconstruction_params, fit_m1["model_dict"])}
    fit_records_map = {"M0": m0_records, "M1-core": m1_records}

    truth_rows = truth_difference_rows(m0_records, m1_records, geom0, geom1)
    write_csv(output / "m0_m1core_truth_difference.csv", truth_rows)
    grid_uv, _u_values, grid_v_values = task3a.build_grid(m0_records)
    grid_v = np.repeat(grid_v_values[:, None], task3a.GRID_U_COUNT, axis=1).ravel()
    grid_evals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, cal, fit in (("M0", base_m0_cal, fit_m0), ("M1-core", base_m1_cal, fit_m1)):
        lam, valid = task3a.production_lambda(grid_uv, fit["model_dict"], cal, reconstruction_params, fit["z_range_mm"])
        grid_evals[name] = (lam, valid)
    parameter_delta = cone_parameter_delta(fit_m0["model_dict"], fit_m1["model_dict"])
    fit_summary = fit_summary_rows(fits, fit_evals, fit_records_map, grid_uv, grid_v, grid_evals)
    write_csv(output / "m0_m1core_cone_fit_summary.csv", fit_summary)
    fit_region_rows = [row for row in fit_summary if row.get("row_type") == "fit_region"]
    write_csv(output / "m0_m1core_fit_region_metrics.csv", fit_region_rows)

    # Only after both cone candidates have completed do we open Validation.
    old_val_uv = load_old_uv(args.old_points.resolve(), OLD_VALIDATION_IDS)
    ext_val_uv = extract_extension_uv(args.extension_validation.resolve(), (formal_k, formal_d), "validation")
    uv_val = {**old_val_uv, **ext_val_uv}
    val_roots = {frame_id: (args.old_validation.resolve() if int(frame_id) <= 24 else args.extension_validation.resolve()) for frame_id in VALIDATION_IDS}
    val_corners: dict[str, dict[str, Any]] = {}
    val_m0_records, val_geom0 = pnp_truth_records(VALIDATION_IDS, uv_val, val_corners, formal_k, formal_d, obj, val_roots, "validation")
    val_m1_records, val_geom1 = pnp_truth_records(VALIDATION_IDS, uv_val, val_corners, m1_k, m1_d, obj, val_roots, "validation")
    val_evals = {"M0": evaluate_records(val_m0_records, base_m0_cal, reconstruction_params, fit_m0["model_dict"]), "M1-core": evaluate_records(val_m1_records, base_m1_cal, reconstruction_params, fit_m1["model_dict"])}
    val_region_rows = region_rows(val_m0_records, val_evals["M0"], "M0", "validation") + region_rows(val_m1_records, val_evals["M1-core"], "M1-core", "validation")
    val_frame_rows: list[dict[str, Any]] = []
    val_by_region_rows: list[dict[str, Any]] = []
    for name, records in (("M0", val_m0_records), ("M1-core", val_m1_records)):
        frames, regions = frame_metric_rows(records, val_evals[name], name, "validation")
        val_frame_rows.extend(frames); val_by_region_rows.extend(regions)
    write_csv(output / "m0_m1core_validation_metrics.csv", val_region_rows)
    write_csv(output / "m0_m1core_validation_by_frame.csv", val_frame_rows)
    write_csv(output / "m0_m1core_validation_by_region.csv", val_by_region_rows)

    frame027_truth = [row for row in truth_rows if row.get("frame_id") == "027"]
    frame027_rows: list[dict[str, Any]] = [{"row_type": "truth_difference", "candidate": "M1-core_minus_M0", "frame_id": "027", "sample_count": len([r for r in frame027_truth if r.get("row_type") == "point"]), "delta_lambda_bias_mm": metric([r["delta_lambda_mm"] for r in frame027_truth if r.get("row_type") == "point"])["bias_mm"], "delta_lambda_rmse_mm": metric([r["delta_lambda_mm"] for r in frame027_truth if r.get("row_type") == "point"])["rmse_mm"], "delta_lambda_p95_abs_mm": metric([r["delta_lambda_mm"] for r in frame027_truth if r.get("row_type") == "point"])["p95_abs_mm"], "delta_lambda_max_abs_mm": metric([r["delta_lambda_mm"] for r in frame027_truth if r.get("row_type") == "point"])["max_abs_mm"]}]
    for name, records, evaluation in (("M0", m0_records, fit_evals["M0"]), ("M1-core", m1_records, fit_evals["M1-core"])):
        rec = next(r for r in records if r.frame_id == "027")
        stats = metric(evaluation["027"]["residual"][evaluation["027"]["valid"]])
        frame027_rows.append({"row_type": "cone_residual", "candidate": name, "frame_id": "027", **stats})
    write_csv(output / "frame027_camera_ab.csv", frame027_rows)

    classification, attribution = classify_camera_effect(fit_region_rows, val_region_rows, val_frame_rows)
    plot_validation(output, val_m0_records, val_evals, val_region_rows + val_frame_rows, (grid_evals["M1-core"][0] - grid_evals["M0"][0], grid_evals["M1-core"][1] & grid_evals["M0"][1]))
    provenance = {
        "task": "6I",
        "classification": classification,
        "validation_opened": True,
        "validation_ids": list(VALIDATION_IDS),
        "fit_ids": list(FIT_IDS),
        "frame027_retained": True,
        "formal_intrinsics": str(args.formal_intrinsics.resolve()),
        "formal_intrinsics_sha256": sha256_file(args.formal_intrinsics.resolve()),
        "formal_cone": str(args.formal_cone.resolve()),
        "formal_cone_sha256_before": sha256_file(args.formal_cone.resolve()),
        "formal_fit_config": str(args.formal_fit_config.resolve()),
        "m1_core_camera": {"fx": float(m1_k[0, 0]), "fy": float(m1_k[1, 1]), "cx": float(m1_k[0, 2]), "cy": float(m1_k[1, 2]), "dist_coeffs": np.asarray(m1_d).reshape(-1).tolist(), "calibration_summary": m1_summary, "per_frame_rmse": m1_frame_rows},
        "frozen_uv": {"old_fit_ids": list(OLD_FIT_IDS), "extension_fit_ids": list(EXT_FIT_IDS), "old_validation_ids": list(OLD_VALIDATION_IDS), "extension_validation_ids": list(EXT_VALIDATION_IDS), "extension_fit_extraction": "formal M0 K/D, existing Steger config, frozen and reused for M0/M1-core", "extension_validation_extraction": "formal M0 K/D, existing Steger config, frozen and reused for M0/M1-core"},
        "cone_fit": {name: {"fit_success": fit["fit_success"], "status": fit["status"], "optimizer_cost": fit["optimizer_cost"], "objective_mse": fit["objective_mse"], "model": fit["model_dict"]} for name, fit in fits.items()},
        "cone_parameter_delta_m1_minus_m0": parameter_delta,
        "attribution": attribution,
        "formal_kd_modified": False,
        "formal_cone_modified": False,
        "steger_changed": False,
        "elliptical_or_correction_used": False,
        "production_writeback": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")
    render_report(output, fit_summary, fit_region_rows, val_region_rows, val_frame_rows, truth_rows, parameter_delta, classification, attribution)
    if sha256_file(args.formal_cone.resolve()) != provenance["formal_cone_sha256_before"]:
        raise RuntimeError("Formal Cone changed during Task 6I")
    print(f"CAMERA_CALIBRATION_EDGE_CAUSAL_EFFECT = {classification}")
    print(f"OUTPUT = {output}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, cv2.error, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
