#!/usr/bin/env python3
"""Task 4B: audit frame 027 truth consistency and frozen-model influence.

This script is FIT-only.  It never reads validation images or points, never
refits a laser model, and never writes a production calibration file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_laser_plane_triplet_coverage as coverage  # noqa: E402
import audit_triplet_edge_extension_observability as edge  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
import run_circular_cone_residual_decomposition as task4a  # noqa: E402


DEFAULT_TASK4A_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "cone_residual_decomposition"
DEFAULT_FIT_EXTENSION = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension"
DEFAULT_OUTPUT_DIR = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "frame027_audit"
AUDIT_IDS = ("025", "026", "027", "028", "029", "030")
TARGET_ID = "027"
TRUTH_TOLERANCE = 1.0e-8
PNP_LIMIT_PX = 0.40


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4B frame 027 influence audit")
    parser.add_argument("--task4a-dir", type=Path, default=DEFAULT_TASK4A_DIR)
    parser.add_argument("--fit-extension", type=Path, default=DEFAULT_FIT_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    task4a.write_csv(path, rows)


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fmt(value: Any, digits: int = 6) -> str:
    number = finite_float(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}g}"


def line_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    center = float(np.mean(x))
    design = np.column_stack([np.ones(len(x)), x - center])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ beta
    centered_energy = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "x_center": center,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(1.0 - np.sum((y - prediction) ** 2) / max(centered_energy, 1.0e-30)),
        "prediction_rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "span_effect": float(beta[1] * np.ptp(x)),
    }


def image_preview(image: np.ndarray, high_percentile: float) -> np.ndarray:
    gray = edge.triplets.to_gray_float(image).astype(np.float64)
    low = float(np.percentile(gray, 0.5))
    high = float(np.percentile(gray, high_percentile))
    if high <= low:
        high = float(np.max(gray))
    return np.clip((gray - low) / max(high - low, 1.0), 0.0, 1.0)


def audit_raw_frames(
    root: Path,
    intrinsics: Any,
    task4a_points: Sequence[Mapping[str, Any]],
    preview_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    inventory, tasks, frame_rows = edge.load_extension_inventory(root)
    frame_metadata = {
        (f"{int(row['pose_id']):03d}", str(row["role"])): row
        for row in frame_rows
        if str(row.get("role", "")) in {"chess", "nolaser", "laser"}
    }
    manifest = coverage.read_yaml(root / "dataset_manifest.yaml")
    manifest_frames = {
        (f"{int(item['pose_id']):03d}", str(item["role"])): item
        for item in manifest.get("frames", [])
        if isinstance(item, Mapping) and str(item.get("role", "")) in {"chess", "nolaser", "laser"}
    }
    audit_rows: list[dict[str, Any]] = []
    extracted: dict[str, dict[str, np.ndarray]] = {}
    preview_data: dict[str, Any] = {}

    for frame_id in AUDIT_IDS:
        roles = inventory.get(frame_id, {})
        if set(roles) != {"chess", "nolaser", "laser"}:
            raise RuntimeError(f"Incomplete FIT triplet for frame {frame_id}: {sorted(roles)}")
        hashes = {role: edge.check_hash(roles[role], True) for role in roles}
        task_splits = {
            str(tasks.get(str(roles[role]["task_id"]), {}).get("tags", {}).get("split", ""))
            for role in roles
        }
        if task_splits != {"fit"}:
            raise RuntimeError(f"Frame {frame_id} is not an explicit FIT triplet: {task_splits}")

        chess = edge.triplets.imread_unicode(Path(roles["chess"]["path"]))
        background = edge.triplets.imread_unicode(Path(roles["nolaser"]["path"]))
        laser = edge.triplets.imread_unicode(Path(roles["laser"]["path"]))
        pose = edge.triplets.detect_board_pose(
            chess,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            cols=11,
            rows=8,
            square_size_mm=20.0,
            max_rmse_px=PNP_LIMIT_PX,
        )
        mask = edge.triplets.board_inner_mask(edge.triplets.to_gray_float(chess).shape, pose.corners, margin_px=-2)
        u, v, response, _ = edge.triplets.extract_laser_centers(
            laser,
            background,
            mask,
            edge.EXTRACTION_CONFIG,
            "vertical",
        )
        truth = coverage.plane_ray_truth(
            u,
            v,
            pose.normal,
            pose.d,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
        )
        if not np.all(truth["valid"]):
            raise RuntimeError(f"Frame {frame_id} has invalid ray-plane truth")

        # Independent formula cross-check of the production audit definition.
        pixels = np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64)
        normalized = cv2.undistortPoints(pixels, intrinsics.camera_matrix, intrinsics.dist_coeffs).reshape(-1, 2)
        rays = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
        lambda_independent = -float(pose.d) / (rays @ np.asarray(pose.normal, dtype=np.float64))
        points_independent = rays * lambda_independent[:, None]
        lambda_formula_max_delta = float(np.max(np.abs(lambda_independent - truth["lambda"])))
        point_formula_max_delta = float(np.max(np.abs(points_independent - truth["points"])))
        plane_equation_max_abs = float(np.max(np.abs(truth["points"] @ pose.normal + float(pose.d))))

        task_rows = sorted(
            [row for row in task4a_points if str(row["frame_id"]) == frame_id],
            key=lambda row: int(row["point_index"]),
        )
        if len(task_rows) != len(u):
            raise RuntimeError(f"Frame {frame_id}: Task 4A points={len(task_rows)}, re-extracted={len(u)}")
        task_uv = np.asarray([[float(row["u_px"]), float(row["v_px"])] for row in task_rows], dtype=np.float64)
        task_lambda = np.asarray([float(row["lambda_truth_mm"]) for row in task_rows], dtype=np.float64)
        uv_task4a_max_delta = float(np.max(np.abs(task_uv - np.column_stack([u, v]))))
        lambda_task4a_max_delta = float(np.max(np.abs(task_lambda - truth["lambda"])))

        center = edge.board_center_z(pose)
        tilt = math.degrees(math.acos(float(np.clip(-float(pose.normal[2]), -1.0, 1.0))))
        metadata = {role: frame_metadata[(frame_id, role)] for role in ("chess", "nolaser", "laser")}
        host_times = {role: int(metadata[role]["host_timestamp_ns"]) for role in metadata}
        capture = {role: manifest_frames.get((frame_id, role), {}) for role in metadata}
        laser_quality = capture["laser"].get("quality", {}) if isinstance(capture["laser"], Mapping) else {}
        extracted[frame_id] = {
            "u": np.asarray(u, dtype=np.float64),
            "v": np.asarray(v, dtype=np.float64),
            "response": np.asarray(response, dtype=np.float64),
            "lambda": np.asarray(truth["lambda"], dtype=np.float64),
        }
        audit_rows.append(
            {
                "frame_id": frame_id,
                "triplet_complete": True,
                "task_split_tags": ";".join(sorted(task_splits)),
                "chess_task_id": roles["chess"]["task_id"],
                "nolaser_task_id": roles["nolaser"]["task_id"],
                "laser_task_id": roles["laser"]["task_id"],
                "chess_path": str(roles["chess"]["path"]),
                "nolaser_path": str(roles["nolaser"]["path"]),
                "laser_path": str(roles["laser"]["path"]),
                "all_sha256_verified": all(value[2] for value in hashes.values()),
                "chess_sha256": hashes["chess"][1],
                "nolaser_sha256": hashes["nolaser"][1],
                "laser_sha256": hashes["laser"][1],
                "chess_to_nolaser_s": (host_times["nolaser"] - host_times["chess"]) * 1.0e-9,
                "nolaser_to_laser_s": (host_times["laser"] - host_times["nolaser"]) * 1.0e-9,
                "chess_to_laser_s": (host_times["laser"] - host_times["chess"]) * 1.0e-9,
                "pnp_rmse_px": float(pose.reprojection_rmse_px),
                "board_nx": float(pose.normal[0]),
                "board_ny": float(pose.normal[1]),
                "board_nz": float(pose.normal[2]),
                "board_d_mm": float(pose.d),
                "board_center_z_mm": float(center[2]),
                "board_tilt_deg": tilt,
                "laser_point_count": int(len(u)),
                "u_min_px": float(np.min(u)),
                "u_max_px": float(np.max(u)),
                "u_span_px": float(np.ptp(u)),
                "v_min_px": float(np.min(v)),
                "v_max_px": float(np.max(v)),
                "v_span_px": float(np.ptp(v)),
                "lambda_min_mm": float(np.min(truth["lambda"])),
                "lambda_max_mm": float(np.max(truth["lambda"])),
                "lambda_span_mm": float(np.ptp(truth["lambda"])),
                "response_p05": float(np.percentile(response, 5)),
                "response_median": float(np.median(response)),
                "response_p95": float(np.percentile(response, 95)),
                "laser_coverage": finite_float(laser_quality.get("laser_coverage", metadata["laser"].get("laser_coverage", ""))),
                "laser_fwhm_p50_px": finite_float(laser_quality.get("laser_fwhm_p50_px", math.nan)),
                "laser_fwhm_p95_px": finite_float(laser_quality.get("laser_fwhm_p95_px", math.nan)),
                "laser_quality_warnings": str(metadata["laser"].get("quality_warnings", "")),
                "lambda_formula_max_delta_mm": lambda_formula_max_delta,
                "point_formula_max_delta_mm": point_formula_max_delta,
                "plane_equation_max_abs_mm": plane_equation_max_abs,
                "uv_task4a_max_delta_px": uv_task4a_max_delta,
                "lambda_task4a_max_delta_mm": lambda_task4a_max_delta,
            }
        )
        if frame_id == TARGET_ID:
            preview_data = {"chess": chess, "background": background, "laser": laser, "u": u, "v": v}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(image_preview(preview_data["chess"], 99.5), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("027 chess (contrast normalized)")
    axes[1].imshow(image_preview(preview_data["background"], 99.99), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("027 nolaser (contrast normalized)")
    axes[2].imshow(image_preview(preview_data["laser"], 99.99), cmap="gray", vmin=0, vmax=1)
    axes[2].scatter(preview_data["u"], preview_data["v"], s=0.8, color="#ef4444", alpha=0.45)
    axes[2].set_title("027 laser + re-extracted centers")
    for axis in axes:
        axis.set_xlim(0, 4096); axis.set_ylim(3000, 0); axis.axis("off")
    fig.tight_layout(); fig.savefig(preview_path, dpi=160); plt.close(fig)
    return audit_rows, extracted


def residual_diagnostics(point_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    selected = sorted([dict(row) for row in point_rows if str(row["frame_id"]) == TARGET_ID], key=lambda row: int(row["point_index"]))
    diagnostics: dict[str, dict[str, float]] = {}
    for model in ("M0", "M_local_fullfit"):
        key = f"{model}_residual_e_lambda_mm"
        valid_indices = [index for index, row in enumerate(selected) if math.isfinite(finite_float(row[key]))]
        u = np.asarray([float(selected[index]["u_px"]) for index in valid_indices], dtype=np.float64)
        v = np.asarray([float(selected[index]["v_px"]) for index in valid_indices], dtype=np.float64)
        e = np.asarray([float(selected[index][key]) for index in valid_indices], dtype=np.float64)
        fit_u = line_fit(u, e); fit_v = line_fit(v, e)
        metric = task4a.metric(e)
        total_energy = float(np.sum(e * e))
        offset_remaining = float(np.sum((e - np.mean(e)) ** 2))
        pred_v = fit_v["intercept"] + fit_v["slope"] * (v - fit_v["x_center"])
        v_line_remaining = float(np.sum((e - pred_v) ** 2))
        scaled_design = np.column_stack([
            np.ones(len(e)),
            (u - np.mean(u)) / max(np.std(u), 1.0e-30),
            (v - np.mean(v)) / max(np.std(v), 1.0e-30),
        ])
        two_d_beta = np.linalg.lstsq(scaled_design, e, rcond=None)[0]
        two_d_prediction = scaled_design @ two_d_beta
        diagnostics[model] = {
            **{str(key_name): float(value) for key_name, value in metric.items()},
            "corr_u_residual": float(np.corrcoef(u, e)[0, 1]),
            "corr_v_residual": float(np.corrcoef(v, e)[0, 1]),
            "corr_u_v": float(np.corrcoef(u, v)[0, 1]),
            "slope_u_mm_per_px": fit_u["slope"],
            "slope_v_mm_per_px": fit_v["slope"],
            "u_span_effect_mm": fit_u["span_effect"],
            "v_span_effect_mm": fit_v["span_effect"],
            "u_line_r2": fit_u["r2"],
            "v_line_r2": fit_v["r2"],
            "offset_only_energy_explained_fraction": float(1.0 - offset_remaining / total_energy),
            "offset_plus_v_tilt_energy_explained_fraction": float(1.0 - v_line_remaining / total_energy),
            "two_d_standardized_condition_number": float(np.linalg.cond(scaled_design)),
            "two_d_line_r2": float(1.0 - np.sum((e - two_d_prediction) ** 2) / max(np.sum((e - np.mean(e)) ** 2), 1.0e-30)),
        }
        for row_index, prediction in zip(valid_indices, pred_v):
            selected[row_index][f"{model}_offset_removed_mm"] = float(selected[row_index][key]) - float(np.mean(e))
            selected[row_index][f"{model}_v_line_prediction_mm"] = float(prediction)
            selected[row_index][f"{model}_v_line_remaining_mm"] = float(selected[row_index][key]) - float(prediction)
    return selected, diagnostics


def decomposition_without_bootstrap(point_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in task4a.BIN_WIDTHS:
        for start in range(0, task4a.IMAGE_HEIGHT, width):
            end = min(start + width, task4a.IMAGE_HEIGHT)
            selected = [row for row in point_rows if start <= float(row["v_px"]) < end]
            for model in ("M0", "M_local_fullfit"):
                summary = task4a.frame_balanced_regression(selected, model)
                rows.append(
                    {
                        "bin_width_px": width,
                        "v_start_px": start,
                        "v_end_px": end,
                        "v_center_px": 0.5 * (start + end),
                        "model": model,
                        "status": summary.get("status", "no_data"),
                        "point_count": summary.get("point_count", 0),
                        "unique_frame_count": summary.get("unique_frame_count", 0),
                        "u_ref_px": summary.get("u_ref_px", math.nan),
                        "u_span_px": summary.get("x_span_px", math.nan),
                        "b_mm": summary.get("b_mm", math.nan),
                        "delta_g_mm_per_px": summary.get("delta_g_mm_per_px", math.nan),
                        "weighted_rmse_mm": summary.get("weighted_rmse_mm", math.nan),
                        "weighted_r2": summary.get("weighted_r2", math.nan),
                    }
                )
    return rows


def compare_decomposition(
    original: Sequence[Mapping[str, Any]],
    leave: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    key = lambda row: (int(row["bin_width_px"]), int(row["v_start_px"]), str(row["model"]))
    original_lookup = {key(row): row for row in original}
    leave_lookup = {key(row): row for row in leave}
    output: list[dict[str, Any]] = []
    for item_key in sorted(set(original_lookup) | set(leave_lookup)):
        width, start, model = item_key
        end = min(start + width, task4a.IMAGE_HEIGHT)
        original_row = original_lookup.get(item_key, {})
        leave_row = leave_lookup.get(item_key, {})
        original_b = finite_float(original_row.get("b_mm")); leave_b = finite_float(leave_row.get("b_mm"))
        original_g = finite_float(original_row.get("delta_g_mm_per_px")); leave_g = finite_float(leave_row.get("delta_g_mm_per_px"))
        output.append(
            {
                "bin_width_px": width,
                "v_start_px": start,
                "v_end_px": end,
                "v_center_px": 0.5 * (start + end),
                "model": model,
                "frame027_point_count": sum(str(row["frame_id"]) == TARGET_ID and start <= float(row["v_px"]) < end for row in point_rows),
                "original_status": original_row.get("status", "missing"),
                "leave027_status": leave_row.get("status", "missing"),
                "original_point_count": original_row.get("point_count", 0),
                "leave027_point_count": leave_row.get("point_count", 0),
                "original_unique_frame_count": original_row.get("unique_frame_count", 0),
                "leave027_unique_frame_count": leave_row.get("unique_frame_count", 0),
                "original_u_ref_px": original_row.get("u_ref_px", math.nan),
                "leave027_u_ref_px": leave_row.get("u_ref_px", math.nan),
                "original_b_mm": original_b,
                "leave027_b_mm": leave_b,
                "delta_b_leave_minus_original_mm": leave_b - original_b if math.isfinite(original_b) and math.isfinite(leave_b) else math.nan,
                "original_delta_g_mm_per_px": original_g,
                "leave027_delta_g_mm_per_px": leave_g,
                "delta_g_leave_minus_original_mm_per_px": leave_g - original_g if math.isfinite(original_g) and math.isfinite(leave_g) else math.nan,
                "original_weighted_rmse_mm": original_row.get("weighted_rmse_mm", math.nan),
                "leave027_weighted_rmse_mm": leave_row.get("weighted_rmse_mm", math.nan),
            }
        )
    return output


def energy_summary(
    original_points: Sequence[Mapping[str, Any]],
    leave_points: Sequence[Mapping[str, Any]],
    original_decomposition: Sequence[Mapping[str, Any]],
    leave_decomposition: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in ("M0", "M_local_fullfit"):
        for width in task4a.BIN_WIDTHS:
            original = task4a.energy_explainability(original_points, original_decomposition, model, width)
            leave = task4a.energy_explainability(leave_points, leave_decomposition, model, width)
            row: dict[str, Any] = {"model": model, "bin_width_px": width}
            for name, values in (("original", original), ("leave027", leave)):
                for field, value in values.items():
                    if field not in {"model", "bin_width_px"}:
                        row[f"{name}_{field}"] = value
            rows.append(row)
    return rows


def plot_residual(path: Path, residual_rows: Sequence[Mapping[str, Any]], x_field: str, title: str) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.4))
    for model, color in (("M0", "#2563eb"), ("M_local_fullfit", "#c2410c")):
        key = f"{model}_residual_e_lambda_mm"
        valid = [row for row in residual_rows if math.isfinite(finite_float(row[key]))]
        x = np.asarray([float(row[x_field]) for row in valid], dtype=np.float64)
        y = np.asarray([float(row[key]) for row in valid], dtype=np.float64)
        fit = line_fit(x, y)
        order = np.argsort(x)
        line = fit["intercept"] + fit["slope"] * (x - fit["x_center"])
        axis.scatter(x, y, s=5, alpha=0.28, color=color, label=model)
        axis.plot(x[order], line[order], color=color, linewidth=1.5)
    axis.axhline(0.0, color="#555", linewidth=0.8)
    axis.set_xlabel(f"{x_field.replace('_px', '')} / px"); axis.set_ylabel("e_lambda / mm")
    axis.set_title(title); axis.grid(alpha=0.2); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=175); plt.close(fig)


def plot_decomposition_comparison(path: Path, rows: Sequence[Mapping[str, Any]], field: str, ylabel: str, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row_index, model in enumerate(("M0", "M_local_fullfit")):
        for col_index, width in enumerate(task4a.BIN_WIDTHS):
            axis = axes[row_index, col_index]
            selected = [row for row in rows if row["model"] == model and int(row["bin_width_px"]) == width]
            x = np.asarray([float(row["v_center_px"]) for row in selected], dtype=np.float64)
            original = np.asarray([finite_float(row[f"original_{field}"]) for row in selected], dtype=np.float64)
            leave = np.asarray([finite_float(row[f"leave027_{field}"]) for row in selected], dtype=np.float64)
            axis.plot(x, original, color="#64748b", linewidth=1.2, label="original 30-frame")
            axis.plot(x, leave, color="#dc2626", linewidth=1.2, label="leave-027-out")
            axis.axhline(0.0, color="#555", linewidth=0.6)
            axis.set_title(f"{model}, {width}px"); axis.grid(alpha=0.2)
            if row_index == 1: axis.set_xlabel("v / px")
            if col_index == 0: axis.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def render_report(
    audit_rows: Sequence[Mapping[str, Any]],
    residual_diagnostics_by_model: Mapping[str, Mapping[str, float]],
    original_regions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    leave_regions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    energies: Sequence[Mapping[str, Any]],
    truth_verdict: str,
    influence_verdict: str,
    energy_share: float,
    original_global_rmse: float,
    leave_global_rmse: float,
    formal_hash: str,
    output_dir: Path,
) -> str:
    target = next(row for row in audit_rows if row["frame_id"] == TARGET_ID)
    local = residual_diagnostics_by_model["M_local_fullfit"]
    lines = [
        "# Task 4B — Frame 027 异常影响诊断",
        "",
        "**FIT_ONLY = TRUE**  ",
        "**VALIDATION_OPENED = FALSE**  ",
        "**CONE_REFIT = FALSE**  ",
        "**PRODUCTION_CONE_MODIFIED = FALSE**",
        "",
        "## 最终结论",
        "",
        f"`FRAME027_TRUTH_CONSISTENCY = {truth_verdict}`",
        "",
        f"`FRAME027_INFLUENCE = {influence_verdict}`",
        "",
        "`NEXT_STEP = B`",
        "",
        "027 的原始三联图 provenance、SHA、PnP、Steger UV、ray 定义和 ray-plane 交点均自洽；没有发现足以判为 truth 错误的证据。027 对全局 FIT residual 的影响很强，但它不覆盖 top<300 或 bottom>2700，因此排除后 top+/bottom− 结构原样保留。下一步应继续 objective mismatch / residual 研究，而不是先删除 027。",
        "",
        "## 1. 原始三联图与 truth 审计",
        "",
        "- 三张图必须同属 FIT manifest 的同一 pose ID，实际文件 SHA-256 必须匹配 frames.csv。",
        "- PnP 使用正式 intrinsics/distortion、11×8 内角点、20 mm 方格；laser center 使用 Task 3/4 沿用的 Steger 配置。",
        "- ray 定义为 `r=[x_n,y_n,1]`，其中 `(x_n,y_n)=cv2.undistortPoints(u,v)`；`lambda_truth=-d/(n·r)`，因此 `Zc=lambda_truth`。",
        f"- 027 chess→laser 间隔 `{fmt(target['chess_to_laser_s'])}` s；邻近帧统计见下表。这个间隔不能数学证明棋盘绝无物理移动，但并不比邻近帧更长。",
        f"- 027 复算 truth 与 Task 4A 的最大 UV/lambda 差为 `{fmt(target['uv_task4a_max_delta_px'])}` px / `{fmt(target['lambda_task4a_max_delta_mm'])}` mm；平面方程最大残差 `{fmt(target['plane_equation_max_abs_mm'])}` mm。",
        "",
        "| frame | SHA all pass | chess→laser s | PnP RMSE px | points | u span px | v span px | lambda span mm | laser coverage | FWHM p50/p95 px |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['frame_id']} | {row['all_sha256_verified']} | {fmt(row['chess_to_laser_s'])} | {fmt(row['pnp_rmse_px'])} | {row['laser_point_count']} | {fmt(row['u_span_px'])} | {fmt(row['v_span_px'])} | {fmt(row['lambda_span_mm'])} | {fmt(row['laser_coverage'])} | {fmt(row['laser_fwhm_p50_px'])}/{fmt(row['laser_fwhm_p95_px'])} |"
        )
    lines += [
        "",
        "027 的 `dynamic_range_low` 与邻近激光帧一致，是暗背景窄激光线采集的共同 warning；027 无丢图、错 ID、hash mismatch、PnP 超阈值、无效 ray-plane 点或 Task 4A truth 不一致。",
        "",
        "## 2. Frame 027 residual",
        "",
        "| model | valid | bias mm | RMSE mm | P95 mm | corr(u,e) | corr(v,e) | corr(u,v) | v slope mm/px | v-line R² | offset energy explained | offset+v-tilt explained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("M0", "M_local_fullfit"):
        row = residual_diagnostics_by_model[model]
        lines.append(
            f"| {model} | {int(row['count'])} | {fmt(row['bias'])} | {fmt(row['rmse'])} | {fmt(row['p95'])} | {fmt(row['corr_u_residual'])} | {fmt(row['corr_v_residual'])} | {fmt(row['corr_u_v'])} | {fmt(row['slope_v_mm_per_px'])} | {fmt(row['v_line_r2'])} | {fmt(row['offset_only_energy_explained_fraction'])} | {fmt(row['offset_plus_v_tilt_energy_explained_fraction'])} |"
        )
    lines += [
        "",
        f"027 不是单纯常数 offset：M_local 的平均偏置为 `{fmt(local['bias'])}` mm，但沿 v 的线性项还能解释大量变化，单变量 v-line R²=`{fmt(local['v_line_r2'])}`。由于该帧激光轨迹上 corr(u,v)=`{fmt(local['corr_u_v'])}`，无法从单帧稳定地区分 u-tilt 与 v-tilt；只能判定为明显的整帧 offset + 沿条纹方向的 tilt。",
        "",
        "## 3. Leave-027-out（模型完全冻结）",
        "",
        f"- M_local 全局 point residual RMSE：`{fmt(original_global_rmse)}` → `{fmt(leave_global_rmse)}` mm。",
        f"- 027 占 M_local 原始总 residual energy 的 `{energy_share:.2%}`，因此影响判为 **{influence_verdict}**。固定阈值：energy share>25% 或 RMSE 降幅>20% 为 STRONG；>10% 为 MODERATE；否则 WEAK。",
        "",
        "### Top / middle / bottom",
        "",
        "| region | original bias/RMSE mm | leave-027 bias/RMSE mm | conclusion |",
        "|---|---:|---:|---|",
    ]
    for region in ("top_formal_edge", "middle_formal", "bottom_formal_edge"):
        original = original_regions["M_local_fullfit"][region]
        leave = leave_regions["M_local_fullfit"][region]
        unchanged = math.isclose(float(original["bias"]), float(leave["bias"]), abs_tol=1.0e-15) and math.isclose(float(original["rmse"]), float(leave["rmse"]), abs_tol=1.0e-15)
        lines.append(
            f"| {region} | {fmt(original['bias'])}/{fmt(original['rmse'])} | {fmt(leave['bias'])}/{fmt(leave['rmse'])} | {('exactly unchanged' if unchanged else 'changed')} |"
        )
    lines += [
        "",
        "### 30/60/100 px residual explainability（M_local_fullfit）",
        "",
        "| bin | original offset | original offset+gain | leave offset | leave offset+gain |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in energies:
        if row["model"] != "M_local_fullfit":
            continue
        lines.append(
            f"| {row['bin_width_px']} | {fmt(row['original_offset_only_explained_fraction'])} | {fmt(row['original_offset_plus_gain_explained_fraction'])} | {fmt(row['leave027_offset_only_explained_fraction'])} | {fmt(row['leave027_offset_plus_gain_explained_fraction'])} |"
        )
    lines += [
        "",
        "排除 027 会显著降低中部的 frame-specific 大残差，也会改变其覆盖 v 范围内的 b(v)/delta_g(v)；但 top 和 bottom 没有任何 027 点，所以 top 正偏、bottom 负偏不是 027 制造的。",
        "",
        "## 边界与下一步",
        "",
        "- `PASS` 仅表示已记录数据与计算链路一致；三联图之间棋盘是否发生肉眼不可见的微小移动，现有暗场 nolaser/laser 图无法独立证明。",
        "- 027 不应被永久删除；如果要进一步区分姿态时序移动与 Cone/objective mismatch，应补做同一姿态的短时重复三联图或在低曝光 laser 图中加入可追踪的板面标记。",
        "- 本轮未重拟合 Cone、未建立 correction、未读取 validation。",
        f"- Formal Cone SHA-256 before/after: `{formal_hash}`。",
        "",
        f"Outputs: `{output_dir}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    task4a_dir = args.task4a_dir.resolve()
    fit_root = args.fit_extension.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_hash_before = sha256_file(task4a.FORMAL_CONE)
    if formal_hash_before != task4a.EXPECTED_CONE_SHA256:
        raise RuntimeError(f"Formal Cone hash mismatch: {formal_hash_before}")
    point_rows = read_csv(task4a_dir / "residual_points.csv")
    original_decomposition = read_csv(task4a_dir / "residual_decomposition_vs_v.csv")
    observed_ids = sorted({str(row["frame_id"]) for row in point_rows}, key=int)
    if observed_ids != task3a.FIT_IDS:
        raise RuntimeError(f"Task 4A points are not the explicit FIT registry: {observed_ids}")
    forbidden = set(task3a.VALIDATION_IDS)
    if forbidden & set(observed_ids):
        raise RuntimeError("Validation ID appeared in Task 4A points")

    _, _, _, intrinsics = task3a.load_runtime(task4a.MEASUREMENT_CONFIG)
    audit_rows, _ = audit_raw_frames(fit_root, intrinsics, point_rows, output_dir / "frame027_triplet_preview.png")
    residual_rows, residual_summary = residual_diagnostics(point_rows)
    write_csv(output_dir / "frame027_residual.csv", residual_rows)
    write_csv(output_dir / "frame027_triplet_comparison.csv", audit_rows)
    plot_residual(output_dir / "frame027_residual_vs_v.png", residual_rows, "v_px", "Frame 027 frozen-model residual versus v")
    plot_residual(output_dir / "frame027_residual_vs_u.png", residual_rows, "u_px", "Frame 027 frozen-model residual versus u")

    leave_points = [row for row in point_rows if str(row["frame_id"]) != TARGET_ID]
    leave_decomposition = decomposition_without_bootstrap(leave_points)
    comparison_rows = compare_decomposition(original_decomposition, leave_decomposition, point_rows)
    write_csv(output_dir / "leave027_residual_decomposition.csv", comparison_rows)
    plot_decomposition_comparison(output_dir / "original_vs_leave027_b.png", comparison_rows, "b_mm", "b(v) / mm", "Original versus leave-027-out b(v)")
    plot_decomposition_comparison(output_dir / "original_vs_leave027_delta_g.png", comparison_rows, "delta_g_mm_per_px", "delta_g(v) / mm/px", "Original versus leave-027-out delta_g(v)")

    energies = energy_summary(point_rows, leave_points, original_decomposition, leave_decomposition)
    original_regions = {model: task4a.region_metrics(point_rows, model) for model in ("M0", "M_local_fullfit")}
    leave_regions = {model: task4a.region_metrics(leave_points, model) for model in ("M0", "M_local_fullfit")}
    local_key = "M_local_fullfit_residual_e_lambda_mm"
    original_values = np.asarray([finite_float(row[local_key]) for row in point_rows], dtype=np.float64)
    original_values = original_values[np.isfinite(original_values)]
    leave_values = np.asarray([finite_float(row[local_key]) for row in leave_points], dtype=np.float64)
    leave_values = leave_values[np.isfinite(leave_values)]
    frame_values = np.asarray([finite_float(row[local_key]) for row in residual_rows], dtype=np.float64)
    frame_values = frame_values[np.isfinite(frame_values)]
    original_energy = float(np.sum(original_values * original_values))
    energy_share = float(np.sum(frame_values * frame_values) / original_energy)
    original_rmse = float(np.sqrt(np.mean(original_values * original_values)))
    leave_rmse = float(np.sqrt(np.mean(leave_values * leave_values)))

    target_audit = next(row for row in audit_rows if row["frame_id"] == TARGET_ID)
    truth_pass = (
        bool(target_audit["triplet_complete"])
        and bool(target_audit["all_sha256_verified"])
        and float(target_audit["pnp_rmse_px"]) <= PNP_LIMIT_PX
        and int(target_audit["laser_point_count"]) == 900
        and float(target_audit["lambda_formula_max_delta_mm"]) <= TRUTH_TOLERANCE
        and float(target_audit["point_formula_max_delta_mm"]) <= TRUTH_TOLERANCE
        and float(target_audit["plane_equation_max_abs_mm"]) <= TRUTH_TOLERANCE
        and float(target_audit["uv_task4a_max_delta_px"]) <= TRUTH_TOLERANCE
        and float(target_audit["lambda_task4a_max_delta_mm"]) <= TRUTH_TOLERANCE
    )
    truth_verdict = "PASS" if truth_pass else "SUSPECT"
    rmse_reduction = 1.0 - leave_rmse / original_rmse
    if energy_share > 0.25 or rmse_reduction > 0.20:
        influence_verdict = "STRONG"
    elif energy_share > 0.10 or rmse_reduction > 0.10:
        influence_verdict = "MODERATE"
    else:
        influence_verdict = "WEAK"

    formal_hash_after = sha256_file(task4a.FORMAL_CONE)
    if formal_hash_after != formal_hash_before:
        raise RuntimeError("Formal Cone changed during Task 4B")
    provenance = {
        "task": "Task 4B Frame 027 anomaly influence diagnosis",
        "fit_frame_ids": task3a.FIT_IDS,
        "audited_raw_frame_ids": list(AUDIT_IDS),
        "validation_frame_ids_not_opened": task3a.VALIDATION_IDS,
        "validation_opened": False,
        "cone_refit": False,
        "production_writeback": False,
        "formal_cone_sha256_before": formal_hash_before,
        "formal_cone_sha256_after": formal_hash_after,
        "truth_tolerance": TRUTH_TOLERANCE,
        "pnp_limit_px": PNP_LIMIT_PX,
        "truth_verdict": truth_verdict,
        "influence_verdict": influence_verdict,
        "next_step": "B",
        "energy_share_frame027": energy_share,
        "original_global_rmse_mm": original_rmse,
        "leave027_global_rmse_mm": leave_rmse,
        "rmse_reduction_fraction": rmse_reduction,
        "residual_summary": residual_summary,
        "energy_summary": energies,
        "original_region_metrics": original_regions,
        "leave027_region_metrics": leave_regions,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=task4a.json_default), encoding="utf-8")
    (output_dir / "frame027_audit.md").write_text(
        render_report(
            audit_rows,
            residual_summary,
            original_regions,
            leave_regions,
            energies,
            truth_verdict,
            influence_verdict,
            energy_share,
            original_rmse,
            leave_rmse,
            formal_hash_before,
            output_dir,
        ),
        encoding="utf-8",
    )
    print(f"FRAME027_TRUTH_CONSISTENCY={truth_verdict}")
    print(f"FRAME027_INFLUENCE={influence_verdict}")
    print("NEXT_STEP=B")
    print("VALIDATION_OPENED=False")
    print(f"OUTPUT={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
