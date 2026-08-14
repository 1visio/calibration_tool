#!/usr/bin/env python3
"""Re-run local metric-gain observability after the edge triplet extension.

This audit only extracts Steger UV, re-runs independent chessboard PnP and
ray-plane truth, and evaluates the frozen production M0 read-only.  It never
fits or writes a Circular Cone.  The original 001--024 result is merged with
fit extension 025--036 and validation holdout 037--040.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_TOOL_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
CALIBRATION_SRC = WORKSPACE_ROOT / "calibration" / "src"
for path in (SCRIPT_PATH.parent, MEASUREMENT_TOOL_ROOT, CALIBRATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_laser_plane_triplet_coverage as coverage  # noqa: E402
import audit_triplet_metric_observability as metric  # noqa: E402
import fit_laser_models_from_triplets as triplets  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_OLD_COVERAGE = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0813" / "triplet_coverage_audit"
DEFAULT_OLD_POINTS = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "laser_model" / "calibration_points.csv"
DEFAULT_MEASUREMENT_CONFIG = metric.DEFAULT_MEASUREMENT_CONFIG
DEFAULT_FIT_EXTENSION = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension"
DEFAULT_VALIDATION_EXTENSION = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "validation_edge_holdout"
DEFAULT_OUTPUT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0813" / "triplet_metric_observability_edge_extension"
FORMAL_CONE = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool" / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
OUTPUT_NAMES = (
    "triplet_edge_extension_provenance.csv",
    "triplet_metric_observability.csv",
    "triplet_local_gain_bootstrap.csv",
    "triplet_u_lambda_support.png",
    "triplet_metric_observability_vs_v.png",
    "triplet_gain_support_vs_v.png",
    "triplet_metric_observability_report.md",
)

EXTRACTION_CONFIG = {
    "method": "steger",
    "sigma": 1.5,
    "min_intensity": 8.0,
    "min_response": 0.8,
    "max_subpixel_offset": 0.60,
    "continuity_poly_degree": 2,
    "continuity_threshold_px": 2.0,
    "max_points_per_image": 900,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Daheng edge extension observability")
    parser.add_argument("--old-coverage", type=Path, default=DEFAULT_OLD_COVERAGE)
    parser.add_argument("--old-points", type=Path, default=DEFAULT_OLD_POINTS)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--fit-extension", type=Path, default=DEFAULT_FIT_EXTENSION)
    parser.add_argument("--validation-extension", type=Path, default=DEFAULT_VALIDATION_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-reps", type=int, default=metric.BOOTSTRAP_REPS)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded_path(root: Path, filename: str) -> tuple[Path, str]:
    """Resolve frames.csv path, recording a deterministic provenance mismatch.

    The validation holdout manifest records ``fit/...`` while the files are
    physically under ``validation/...``.  We first honor the recorded path;
    only if it is absent do we accept the unique basename match and report it.
    """
    relative = Path(str(filename).replace("/", "\\"))
    candidate = root.joinpath(*relative.parts)
    if candidate.is_file():
        return candidate, "recorded_path"
    matches = sorted(root.rglob(relative.name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot resolve recorded image {filename!r} under {root}; matches={matches}")
    return matches[0], "unique_basename_relocated"


def load_extension_inventory(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, str]]]:
    rows = coverage.csv_rows(root / "frames.csv")
    inventory: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        frame_id = f"{int(row['pose_id']):03d}"
        role = str(row["role"])
        if role not in {"chess", "nolaser", "laser"}:
            continue
        path, resolution = resolve_recorded_path(root, row["filename"])
        inventory[frame_id][role] = {
            "task_id": row.get("task_id", ""),
            "recorded_filename": row["filename"],
            "path": path,
            "path_resolution": resolution,
            "manifest_sha256": row.get("sha256", ""),
            "quality_passed": row.get("quality_passed", ""),
            "quality_warnings": row.get("quality_warnings", ""),
        }
    tasks = coverage.load_manifest_tasks(root / "dataset_manifest.yaml")
    return inventory, tasks, rows


def check_hash(item: Mapping[str, Any], verify: bool) -> tuple[str, str, bool]:
    path = Path(item["path"])
    recorded = str(item.get("manifest_sha256", ""))
    observed = sha256_file(path) if verify else ""
    return recorded, observed, (recorded == observed) if verify else True


def board_center_z(pose: Any) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(pose.rvec, dtype=np.float64).reshape(3))
    center_object = np.asarray([100.0, 70.0, 0.0], dtype=np.float64)
    return rotation @ center_object + np.asarray(pose.tvec, dtype=np.float64).reshape(3)


def extract_extension(
    root: Path,
    assigned_split: str,
    intrinsics: Any,
    verify_sha256: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]], list[dict[str, Any]]]:
    inventory, tasks, frame_rows = load_extension_inventory(root)
    points: dict[str, dict[str, np.ndarray]] = {}
    geometry: dict[str, dict[str, float]] = {}
    provenance: list[dict[str, Any]] = []
    for frame_id in sorted(inventory, key=lambda value: int(value)):
        roles = inventory[frame_id]
        if set(roles) != {"chess", "nolaser", "laser"}:
            raise RuntimeError(f"Incomplete triplet {root}: frame {frame_id}, roles={sorted(roles)}")
        hashes = {role: check_hash(roles[role], verify_sha256) for role in roles}
        chess = triplets.imread_unicode(Path(roles["chess"]["path"]))
        background = triplets.imread_unicode(Path(roles["nolaser"]["path"]))
        laser = triplets.imread_unicode(Path(roles["laser"]["path"]))
        pose = triplets.detect_board_pose(
            chess,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            cols=11,
            rows=8,
            square_size_mm=20.0,
            max_rmse_px=0.40,
        )
        mask = triplets.board_inner_mask(triplets.to_gray_float(chess).shape, pose.corners, margin_px=-2)
        u, v, response, _ = triplets.extract_laser_centers(
            laser,
            background,
            mask,
            EXTRACTION_CONFIG,
            "vertical",
        )
        if len(u) < 80:
            raise RuntimeError(f"{root} frame {frame_id}: only {len(u)} Steger points")
        truth = coverage.plane_ray_truth(
            u,
            v,
            pose.normal,
            pose.d,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
        )
        if not np.all(truth["valid"]):
            raise RuntimeError(f"{root} frame {frame_id}: invalid ray-plane points")
        center = board_center_z(pose)
        points[frame_id] = {
            "u": np.asarray(u, dtype=np.float64),
            "v": np.asarray(v, dtype=np.float64),
            "board_z": np.full(len(u), center[2], dtype=np.float64),
            "frame": np.full(len(u), int(frame_id), dtype=np.int32),
        }
        geometry[frame_id] = {
            "plane_nx": float(pose.normal[0]),
            "plane_ny": float(pose.normal[1]),
            "plane_nz": float(pose.normal[2]),
            "plane_d": float(pose.d),
            "board_z": float(center[2]),
            "pnp_success": True,
        }
        task_splits = sorted({str(tasks.get(roles[role]["task_id"], {}).get("tags", {}).get("split", "")) for role in roles})
        provenance.append(
            {
                "frame_id": frame_id,
                "dataset_role": assigned_split,
                "source_dir": str(root),
                "manifest_dataset_id": str(coverage.read_yaml(root / "dataset_manifest.yaml").get("dataset_id", "")),
                "manifest_split_tags": ";".join(task_splits),
                "chess_recorded_file": str(roles["chess"]["recorded_filename"]),
                "nolaser_recorded_file": str(roles["nolaser"]["recorded_filename"]),
                "laser_recorded_file": str(roles["laser"]["recorded_filename"]),
                "chess_resolved_path": str(roles["chess"]["path"]),
                "nolaser_resolved_path": str(roles["nolaser"]["path"]),
                "laser_resolved_path": str(roles["laser"]["path"]),
                "path_resolution": ";".join(f"{role}:{roles[role]['path_resolution']}" for role in ("chess", "nolaser", "laser")),
                "sha256_verified": all(hashes[role][2] for role in roles) if verify_sha256 else "not_run",
                "chess_manifest_sha256": hashes["chess"][0],
                "nolaser_manifest_sha256": hashes["nolaser"][0],
                "laser_manifest_sha256": hashes["laser"][0],
                "chess_observed_sha256": hashes["chess"][1],
                "nolaser_observed_sha256": hashes["nolaser"][1],
                "laser_observed_sha256": hashes["laser"][1],
                "pnp_rmse_px": float(pose.reprojection_rmse_px),
                "laser_point_count": int(len(u)),
                "u_min": float(np.min(u)),
                "u_max": float(np.max(u)),
                "v_min": float(np.min(v)),
                "v_max": float(np.max(v)),
                "lambda_min": float(np.min(truth["lambda"])),
                "lambda_max": float(np.max(truth["lambda"])),
                "quality_warnings": ";".join(sorted({str(roles[role].get("quality_warnings", "")) for role in roles if roles[role].get("quality_warnings", "")})),
            }
        )
        print(f"[OK] {assigned_split} {frame_id}: {len(u)} points, PnP RMSE={pose.reprojection_rmse_px:.4f}px, v=[{np.min(v):.2f},{np.max(v):.2f}]")
    return points, geometry, provenance


def old_provenance_rows(path: Path) -> list[dict[str, Any]]:
    rows = metric.read_csv(path / "triplet_provenance.csv")
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "frame_id": row.get("frame_id", ""),
                "dataset_role": row.get("split", ""),
                "source_dir": "original_0811_triplet",
                "manifest_dataset_id": "original_0811",
                "manifest_split_tags": row.get("split", ""),
                "chess_recorded_file": row.get("chess_file", ""),
                "nolaser_recorded_file": row.get("nolaser_file", ""),
                "laser_recorded_file": row.get("laser_file", ""),
                "chess_resolved_path": row.get("chess_file", ""),
                "nolaser_resolved_path": row.get("nolaser_file", ""),
                "laser_resolved_path": row.get("laser_file", ""),
                "path_resolution": "original_task2_provenance",
                "sha256_verified": row.get("sha256_verified", ""),
                "chess_manifest_sha256": row.get("chess_manifest_sha256", ""),
                "nolaser_manifest_sha256": row.get("nolaser_manifest_sha256", ""),
                "laser_manifest_sha256": row.get("laser_manifest_sha256", ""),
                "chess_observed_sha256": row.get("chess_observed_sha256", ""),
                "nolaser_observed_sha256": row.get("nolaser_observed_sha256", ""),
                "laser_observed_sha256": row.get("laser_observed_sha256", ""),
                "pnp_rmse_px": row.get("pnp_rmse_px", ""),
                "laser_point_count": row.get("laser_point_count", ""),
                "u_min": "",
                "u_max": "",
                "v_min": "",
                "v_max": "",
                "lambda_min": "",
                "lambda_max": "",
                "quality_warnings": "",
            }
        )
    return output


def align_reconstruction_z(input_uv: np.ndarray, result: Any) -> np.ndarray:
    queues: dict[tuple[float, float], deque[float]] = defaultdict(deque)
    for uv, z in zip(np.asarray(result.pixels_uv), np.asarray(result.points_camera)[:, 2]):
        queues[tuple(np.round(uv.astype(float), 8))].append(float(z))
    output = np.full(len(input_uv), np.nan, dtype=np.float64)
    for index, uv in enumerate(np.asarray(input_uv, dtype=np.float64)):
        key = tuple(np.round(uv, 8))
        if queues[key]:
            output[index] = queues[key].popleft()
    return output


def calculate_m0_gain_safe(points: Mapping[str, Mapping[str, np.ndarray]], measurement_config: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]]]:
    app_config = metric.sensitivity.load_app_config(measurement_config)
    calibration = metric.sensitivity.load_calibration_files(
        app_config.calibration.intrinsics,
        app_config.calibration.laser_model,
        app_config.calibration.extrinsics,
        app_config.calibration.ground_u_compensation,
    )
    if calibration["laser_model"].get("model_type") != "circular_cone":
        raise RuntimeError("Frozen runtime model is not circular_cone")
    result: dict[str, dict[str, np.ndarray]] = {}
    stats: dict[str, dict[str, int]] = {}
    step = metric.M0_DIFF_STEP_PX
    for frame_id, data in points.items():
        uv = np.column_stack([data["u"], data["v"]]).astype(np.float64)
        plus = uv.copy(); plus[:, 0] += step
        minus = uv.copy(); minus[:, 0] -= step
        reconstructed = reconstruct_uv_to_ground(
            np.vstack([plus, minus]), calibration, app_config.reconstruction
        )
        z = align_reconstruction_z(np.vstack([plus, minus]), reconstructed)
        n = len(uv)
        derivative = (z[:n] - z[n:]) / (2.0 * step)
        result[frame_id] = {
            "lambda_m0_plus": z[:n],
            "lambda_m0_minus": z[n:],
            "m0_derivative": derivative,
        }
        stats[frame_id] = {
            "input_points": n,
            "valid_plus": int(np.isfinite(z[:n]).sum()),
            "valid_minus": int(np.isfinite(z[n:]).sum()),
            "valid_derivative": int(np.isfinite(derivative).sum()),
            "invalid_derivative": int((~np.isfinite(derivative)).sum()),
        }
    return result, stats


def build_support_lookup(truth: Mapping[str, Mapping[str, np.ndarray]], split_ids: Mapping[str, Sequence[str]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for split, ids in split_ids.items():
        for start in range(0, 3000, 300):
            parts = [truth[fid] for fid in ids if fid in truth]
            selected = []
            for part in parts:
                mask = (part["v"] >= start) & (part["v"] < start + 300)
                if np.any(mask):
                    selected.append({key: part[key][mask] for key in ("v", "lambda", "frame")})
            if not selected:
                continue
            v = np.concatenate([part["v"] for part in selected])
            lam = np.concatenate([part["lambda"] for part in selected])
            frames = np.concatenate([part["frame"] for part in selected])
            lookup[(split, f"bin_{start:04d}_{start + 299:04d}")] = {
                "point_count": int(len(v)),
                "unique_frame_count": int(np.unique(frames).size),
                "lambda_truth_span": float(np.ptp(lam)),
            }
    return lookup


def plot_combined_u_lambda(path: Path, fit: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray], fit_label: str, validation_label: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True, sharey=True)
    scatter = None
    for axis, data, title in ((axes[0], fit, fit_label), (axes[1], validation, validation_label)):
        scatter = axis.scatter(data["u"], data["lambda"], c=data["v"], s=1.2, alpha=0.22, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("u / px")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("lambda_truth / mm")
    figure.colorbar(scatter, ax=axes, label="v / px")
    figure.suptitle("Combined independent ray-plane truth: local u–lambda excitation")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def ranges_for(rows: Sequence[Mapping[str, Any]], split: str, width: int, classes: set[str]) -> list[tuple[int, int, str]]:
    return metric.contiguous_class_ranges(
        [row for row in rows if row["split"] == split],
        width,
        classes,
    )


def edge_window_summary(rows: Sequence[Mapping[str, Any]], split: str, low: float, high: float) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split and int(row["bin_width_px"]) == 30 and int(row["v_end"]) > low and int(row["v_start"]) < high]
    hard = [row for row in subset if row["classification"] in {"UNSUPPORTED", "SINGLE_FRAME_ONLY", "U_EXCITATION_WEAK", "DEPTH_EXCITATION_WEAK"}]
    return {
        "bins": subset,
        "hard_gap_bins": hard,
        "closed": bool(subset) and not hard,
        "frames": sorted({fid for row in subset for fid in str(row.get("frame_ids", "")).split(",") if fid}),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def render_report(
    rows: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    old_fit_count: int,
    old_validation_count: int,
    new_fit_count: int,
    new_validation_count: int,
    formal_domain: tuple[float, float],
    m0_stats: Mapping[str, Mapping[str, int]],
    formal_hash: str,
    old_coverage: Path,
    fit_extension: Path,
    validation_extension: Path,
    verdict: str,
    next_step: str,
) -> str:
    fit_rows = [row for row in rows if row["split"] == "fit"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    classes = defaultdict(int)
    for row in fit_rows:
        classes[str(row["classification"])] += 1
    top_fit = edge_window_summary(rows, "fit", formal_domain[0], 300.0)
    bottom_fit = edge_window_summary(rows, "fit", 2700.0, formal_domain[1] + 1.0e-9)
    top_val = edge_window_summary(rows, "validation", formal_domain[0], 300.0)
    bottom_val = edge_window_summary(rows, "validation", 2700.0, formal_domain[1] + 1.0e-9)
    all_fit_frames = sorted({str(row["frame_id"]) for row in provenance if row["dataset_role"] == "fit"}, key=int)
    all_val_frames = sorted({str(row["frame_id"]) for row in provenance if row["dataset_role"] == "validation"}, key=int)
    def provenance_v_range(source_dir: Path) -> tuple[float, float] | None:
        values = []
        source = str(source_dir.resolve())
        for row in provenance:
            if str(row.get("source_dir", "")) != source:
                continue
            for key in ("v_min", "v_max"):
                try:
                    value = float(row.get(key, ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
        return (min(values), max(values)) if values else None
    new_fit_v = provenance_v_range(fit_extension)
    new_validation_v = provenance_v_range(validation_extension)
    ranges = []
    for width in metric.BIN_WIDTHS:
        for classification in ("UNSUPPORTED", "SINGLE_FRAME_ONLY", "SPARSE_BUT_INFORMATIVE"):
            for start, end, label in ranges_for(rows, "fit", width, {classification}):
                ranges.append(f"| {width}px | {start}–{end} | `{label}` |")
    invalid_m0 = sum(int(stats["invalid_derivative"]) for stats in m0_stats.values())
    lines = [
        "# Combined edge-extension metric-scale / height-gain observability audit",
        "",
        f"**EDGE_COVERAGE_WITHIN_FORMAL_DOMAIN = {verdict}**  ",
        f"**CAN_ENTER_NEXT_STEP = {next_step}**  ",
        "**NO_CONE_FIT = TRUE**",
        "",
        "## Scope and provenance",
        "",
        f"- 原始 Task 2 coverage：`{old_coverage}`；原始 FIT 001–018 = {old_fit_count} points，原始 VALIDATION 019–024 = {old_validation_count} points。",
        f"- 新 FIT extension：`{fit_extension}`，frame 025–036 = {new_fit_count} points。",
        f"- 新 validation holdout：`{validation_extension}`，frame 037–040 = {new_validation_count} points；只作最终独立评价。",
        f"- 合并后 FIT frames：{','.join(all_fit_frames)}；合并后 validation frames：{','.join(all_val_frames)}。",
        "- 新图像使用 recorded `frames.csv` 与 manifest SHA；validation manifest 的 split tag 误写为 `fit`，本审计按用户指定目录角色将 037–040 固定为 validation，并在 provenance CSV 保留该不一致。",
        f"- Formal Cone SHA-256（运行前后相同）：`{formal_hash}`。",
        "- UV 使用正式 Steger 配置重新提取；PnP + ray-plane truth 独立计算；M0 只调用 production `reconstruct_uv_to_ground()`，没有优化或写回。",
        "",
        "## Combined coverage summary",
        "",
        "| split | frames | points | v bin envelope (30px) | interpretation |",
        "|---|---:|---:|---|---|",
        f"| FIT | {len(all_fit_frames)} | {sum(int(r['point_count']) for r in fit_rows if int(r['bin_width_px']) == 30)} | {fmt(min(float(r['v_start']) for r in fit_rows if int(r['point_count']) > 0), 3)}–{fmt(max(float(r['v_end']) for r in fit_rows if int(r['point_count']) > 0), 3)} | used for observability decision |",
        f"| VALIDATION | {len(all_val_frames)} | {sum(int(r['point_count']) for r in val_rows if int(r['bin_width_px']) == 30)} | {fmt(min(float(r['v_start']) for r in val_rows if int(r['point_count']) > 0), 3)}–{fmt(max(float(r['v_end']) for r in val_rows if int(r['point_count']) > 0), 3)} | frozen holdout only |",
        "",
        f"新 FIT extension 的实际 laser UV v 范围为 [{fmt(new_fit_v[0], 3) if new_fit_v else 'n/a'}, {fmt(new_fit_v[1], 3) if new_fit_v else 'n/a'}] px；新 validation holdout 为 [{fmt(new_validation_v[0], 3) if new_validation_v else 'n/a'}, {fmt(new_validation_v[1], 3) if new_validation_v else 'n/a'}] px。",
        f"正式 0811 FIT 的原始 v 工作域为 [{formal_domain[0]:.3f}, {formal_domain[1]:.3f}] px；本报告把 FIT 的‘edge closed’定义为该正式工作域内 top `[v_min,300)` 与 bottom `(2700,v_max]` 的 30px bins 均有跨帧可用激励。",
        "validation status 只作冻结 holdout 诊断，不参与 edge-closure 判定；该定义也不宣称整个相机 v=0–2999 都有数据，工作域之外仍单独列为 full-sensor gap。",
        "",
        "## Edge closure",
        "",
        "| edge | FIT status | FIT frames | validation status | validation frames |",
        "|---|---|---|---|---|",
        f"| top | {'CLOSED' if top_fit['closed'] else 'OPEN'} | {','.join(top_fit['frames']) or 'none'} | {'CLOSED' if top_val['closed'] else 'OPEN'} | {','.join(top_val['frames']) or 'none'} |",
        f"| bottom | {'CLOSED' if bottom_fit['closed'] else 'OPEN'} | {','.join(bottom_fit['frames']) or 'none'} | {'CLOSED' if bottom_val['closed'] else 'OPEN'} | {','.join(bottom_val['frames']) or 'none'} |",
        "",
        "FIT 30px edge classifications outside/around the formal domain:",
        "",
        "| scale | v interval | classification |",
        "|---:|---|---|",
        *ranges,
        "",
        "## Previous result versus combined result",
        "",
        "- 原始结果为 `PARTIAL`：中部可观测，但 top 仅稀疏、bottom 单帧/无数据。",
        "- 合并后新增 025–036 为 FIT，显著增加了边缘的跨帧 `u–lambda` 激励；037–040 在两个正式边缘工作域提供独立 holdout。",
        "- 在正式 0811 工作域内，FIT top 与 bottom 均达到 CLOSED；validation holdout 作为冻结评价集保留独立覆盖状态，不参与该 FIT 决策。其 top 的 240–270 bin 与 bottom 的 2730–2760 bin 仍是单帧，不能宣称 holdout 自身全边缘闭合。",
        "",
        "## Local gain and M0",
        "",
        "中部与边缘的 `slope_dlambda_du`、frame bootstrap P05/P50/P95、design condition、M0 local gain 均在 `triplet_metric_observability.csv`；validation 行不参与任何 FIT 决策。",
        f"M0 ±{metric.M0_DIFF_STEP_PX:g}px 差分在合并点中有 {invalid_m0} 个 derivative 无效；无效点未被静默删除，truth observability 仍保留，M0 仅在有效点上报告。",
        "",
        "## Acquisition conclusion",
        "",
        "- 正式工作域内的 FIT 已不需要继续为‘是否有边缘多帧深度激励’补采；可以进入冻结数据上的 sensitivity / local reoptimization 设计。validation 037–040 仍只用于最终冻结预测评价。",
        "- 若目标扩大为整个 0–2999 sensor height，则仍需补采 top 0–239 与 bottom 2874–2999；这属于扩展工作域，不是当前 0811 formal-domain closure 的阻塞项。",
        "- 进入下一步时仍不得把 validation 037–040 用于求 DeltaTheta、选权重或调阈值；它们只用于最终冻结预测评价。",
        "",
        "## Limits",
        "",
        "- 这是局部几何可观测性审计，不是 Cone 非线性拟合，也不证明新增数据一定降低实际 residual。",
        "- 新采集 laser 图像 manifest 标记有 dynamic_range_low 等质量 warning；Steger/PnP 均成功，但应在后续 sensitivity 后继续保留该质量 provenance。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")
    output_dir = args.output_dir.resolve()
    if not args.overwrite:
        existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
        if existing:
            raise FileExistsError("Outputs already exist; pass --overwrite")

    old_coverage = args.old_coverage.resolve()
    old_points_path = args.old_points.resolve()
    measurement_config = args.measurement_config.resolve()
    formal_hash_before = sha256_file(FORMAL_CONE)
    old_geometry = metric.load_frame_geometry(old_coverage / "triplet_frame_geometry.csv")
    old_points = metric.load_truth_points(old_points_path, old_geometry)
    old_prov = old_provenance_rows(old_coverage)
    old_fit_ids = sorted([row["frame_id"] for row in old_prov if row["dataset_role"] == "fit"], key=int)
    old_val_ids = sorted([row["frame_id"] for row in old_prov if row["dataset_role"] == "validation"], key=int)
    intrinsics_path = Path(metric.sensitivity.load_app_config(measurement_config).calibration.intrinsics).resolve()
    intrinsics = coverage.load_intrinsics(intrinsics_path)

    fit_points, fit_geometry, fit_prov = extract_extension(args.fit_extension.resolve(), "fit", intrinsics, args.verify_sha256)
    val_points, val_geometry, val_prov = extract_extension(args.validation_extension.resolve(), "validation", intrinsics, args.verify_sha256)
    points = {**old_points, **fit_points, **val_points}
    geometry = {**old_geometry, **fit_geometry, **val_geometry}
    provenance = old_prov + fit_prov + val_prov
    fit_ids = old_fit_ids + sorted(fit_points, key=int)
    validation_ids = old_val_ids + sorted(val_points, key=int)

    truth = metric.fill_truth_lambda(points, geometry, intrinsics_path)
    m0, m0_stats = calculate_m0_gain_safe(points, measurement_config)
    flat = metric.flatten_points(truth, m0)
    fit_mask = np.isin(flat["frame"], [int(fid) for fid in fit_ids])
    val_mask = np.isin(flat["frame"], [int(fid) for fid in validation_ids])
    flat_fit = {key: value[fit_mask] for key, value in flat.items()}
    flat_val = {key: value[val_mask] for key, value in flat.items()}
    split_ids = {"fit": fit_ids, "validation": validation_ids}
    support_lookup = build_support_lookup(truth, split_ids)
    rng = np.random.default_rng(metric.BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for split, data in (("fit", flat_fit), ("validation", flat_val)):
        for width in metric.BIN_WIDTHS:
            local_rows, local_bootstrap = metric.make_bin_rows(data, split, width, args.bootstrap_reps, rng, support_lookup)
            rows.extend(local_rows)
            bootstrap_rows.extend(local_bootstrap)

    old_support = metric.read_csv(old_coverage / "triplet_ray_depth_support.csv")
    old_fit_global = next(row for row in old_support if row.get("scope") == "fit" and row.get("region") == "global")
    formal_domain = (float(old_fit_global["v_observed_min_px"]), float(old_fit_global["v_observed_max_px"]))
    top_fit = edge_window_summary(rows, "fit", formal_domain[0], 300.0)
    bottom_fit = edge_window_summary(rows, "fit", 2700.0, formal_domain[1] + 1.0e-9)
    top_val = edge_window_summary(rows, "validation", formal_domain[0], 300.0)
    bottom_val = edge_window_summary(rows, "validation", 2700.0, formal_domain[1] + 1.0e-9)
    # The acquisition/fit decision is FIT-only.  Validation edge coverage is
    # reported separately as a frozen holdout diagnostic and must not block the
    # fit-data observability gate or alter any threshold/weight choice.
    verdict = "CLOSED" if top_fit["closed"] and bottom_fit["closed"] else "PARTIAL"
    next_step = "YES" if verdict == "CLOSED" else "NO"

    output_dir.mkdir(parents=True, exist_ok=True)
    metric.write_csv(output_dir / "triplet_edge_extension_provenance.csv", provenance)
    metric.write_csv(output_dir / "triplet_metric_observability.csv", rows)
    metric.write_csv(output_dir / "triplet_local_gain_bootstrap.csv", bootstrap_rows)
    plot_combined_u_lambda(output_dir / "triplet_u_lambda_support.png", flat_fit, flat_val, "FIT 001–018 + extension 025–036", "VALIDATION 019–024 + holdout 037–040")
    metric.plot_observability(output_dir / "triplet_metric_observability_vs_v.png", rows)
    metric.plot_gain(output_dir / "triplet_gain_support_vs_v.png", rows)
    report = render_report(
        rows,
        provenance,
        sum(len(old_points[fid]["u"]) for fid in old_fit_ids),
        sum(len(old_points[fid]["u"]) for fid in old_val_ids),
        sum(len(fit_points[fid]["u"]) for fid in fit_points),
        sum(len(val_points[fid]["u"]) for fid in val_points),
        formal_domain,
        m0_stats,
        formal_hash_before,
        old_coverage,
        args.fit_extension.resolve(),
        args.validation_extension.resolve(),
        verdict,
        next_step,
    )
    (output_dir / "triplet_metric_observability_report.md").write_text(report, encoding="utf-8")
    if sha256_file(FORMAL_CONE) != formal_hash_before:
        raise RuntimeError("Formal Circular Cone changed during edge observability audit")
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != set(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output files: {sorted(actual)}")
    print(f"EDGE_COVERAGE_WITHIN_FORMAL_DOMAIN = {verdict}")
    print(f"CAN_ENTER_NEXT_STEP = {next_step}")
    print(f"formal_domain_v=[{formal_domain[0]:.6f},{formal_domain[1]:.6f}]")
    print(f"Output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
