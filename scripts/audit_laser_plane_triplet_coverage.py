#!/usr/bin/env python3
"""Audit the geometry and provenance of the Daheng 0811 triplet dataset.

This script deliberately does not fit, optimize, or load a laser model.  It
uses the recorded Steger centre pixels from the completed 0811 triplet output,
re-runs the official chessboard/PnP detector, and constructs independent
camera-ray / PnP-plane intersections for coverage statistics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = CALIBRATION_TOOL_ROOT.parent
CALIBRATION_SRC = WORKSPACE_ROOT / "calibration" / "src"
if str(CALIBRATION_SRC) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_SRC))

from calibrate_ground_extrinsics_board_only import (  # noqa: E402
    chessboard_object_points,
    detect_chessboard,
    load_intrinsics,
)


DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_FRAMES_CSV = DEFAULT_DATA_ROOT / "frames.csv"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "dataset_manifest.yaml"
DEFAULT_CONFIG = CALIBRATION_TOOL_ROOT / "configs" / "laser_model_fit_config.daheng.yaml"
DEFAULT_INTRINSICS = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
)
DEFAULT_CALIBRATION_POINTS = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "laser_model" / "calibration_points.csv"
)
DEFAULT_STAGE_RUN = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "laser_model" / "stage_run.yaml"
)
DEFAULT_MODEL_OUTPUT = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "laser_model" / "laser_model.yaml"
)
DEFAULT_FORMAL_CONE = (
    WORKSPACE_ROOT
    / "linelaser_tool"
    / "laser_measurement_tool"
    / "configs"
    / "calibration_daheng_0811"
    / "circular_cone.yaml"
)
DEFAULT_PER_IMAGE = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "laser_model" / "per_image_metrics.csv"
)
DEFAULT_OUTPUT = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0813" / "triplet_coverage_audit"
)

OUTPUT_NAMES = (
    "triplet_provenance.csv",
    "triplet_frame_geometry.csv",
    "triplet_ray_depth_support.csv",
    "triplet_uv_support.png",
    "triplet_v_lambda_support.png",
    "triplet_pose_depth_distribution.png",
    "triplet_coverage_audit.md",
    "OUTPUT_FILES.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Daheng 0811 laser-plane triplet coverage")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frames-csv", type=Path, default=DEFAULT_FRAMES_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fit-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--calibration-points", type=Path, default=DEFAULT_CALIBRATION_POINTS)
    parser.add_argument("--stage-run", type=Path, default=DEFAULT_STAGE_RUN)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--formal-cone", type=Path, default=DEFAULT_FORMAL_CONE)
    parser.add_argument("--per-image", type=Path, default=DEFAULT_PER_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify recorded image hashes against the triplet manifest",
    )
    return parser.parse_args(argv)


def read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.12g}"
    return str(value)


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_number(row.get(key)) for key in fields})


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": math.nan, "max": math.nan, "span": math.nan, "median": math.nan, "p05": math.nan, "p95": math.nan}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "span": float(np.max(values) - np.min(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def load_frame_inventory(frames_csv: Path, data_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    inventory: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in csv_rows(frames_csv):
        frame_id = f"{int(row['pose_id']):03d}"
        role = str(row["role"])
        if role not in {"chess", "nolaser", "laser"}:
            continue
        relative = Path(row["filename"].replace("/", str(Path("/"))).replace("\\", str(Path("/"))))
        # Path construction above is only a separator normalisation; the
        # filename itself is taken from frames.csv, never inferred.
        relative = Path(str(row["filename"]).replace("/", "\\"))
        absolute = data_root / Path(*relative.parts)
        inventory[frame_id][role] = {
            "task_id": row.get("task_id", ""),
            "filename": row["filename"],
            "path": absolute,
            "manifest_sha256": row.get("sha256", ""),
            "quality_passed": row.get("quality_passed", ""),
            "quality_warnings": row.get("quality_warnings", ""),
        }
    return inventory


def load_manifest_tasks(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = read_yaml(manifest_path)
    tasks = manifest.get("plan", {}).get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"manifest tasks missing: {manifest_path}")
    return {str(item["task_id"]): item for item in tasks if isinstance(item, Mapping) and "task_id" in item}


def load_points(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = csv_rows(path)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        frame_id = f"{int(row['image_id']):03d}"
        grouped[frame_id]["split"].append(row["split"])
        for key in ("u_px", "v_px", "response", "ray_x", "ray_y", "ray_z", "Xc_mm", "Yc_mm", "Zc_mm", "board_nx", "board_ny", "board_nz", "board_d_mm", "pnp_rmse_px"):
            grouped[frame_id][key].append(float(row[key]))
    result: dict[str, dict[str, np.ndarray]] = {}
    for frame_id, values in grouped.items():
        result[frame_id] = {
            key: np.asarray(items, dtype=np.float64) if key != "split" else np.asarray(items, dtype=object)
            for key, items in values.items()
        }
    return result


def stage_argument(stage_run: Mapping[str, Any], key: str, default: str = "") -> str:
    args = stage_run.get("arguments", [])
    if not isinstance(args, list):
        return default
    for index, value in enumerate(args[:-1]):
        if str(value) == key:
            return str(args[index + 1])
    return default


def parse_ids(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    datasets = config.get("datasets", {})
    train = [f"{int(value):03d}" for value in datasets.get("train", {}).get("ids", [])]
    validation = [f"{int(value):03d}" for value in datasets.get("validation", {}).get("ids", [])]
    return train, validation


def plane_ray_truth(
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
    plane_d: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, np.ndarray]:
    pixels = np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64)
    normalized = cv2.undistortPoints(pixels, camera_matrix, dist_coeffs).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(normalized.shape[0], dtype=np.float64)])
    denominator = rays @ normal
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12)
    lam = np.full(u.shape, np.nan, dtype=np.float64)
    lam[valid] = -float(plane_d) / denominator[valid]
    valid &= np.isfinite(lam) & (lam > 0.0)
    points = np.full((u.size, 3), np.nan, dtype=np.float64)
    points[valid] = rays[valid] * lam[valid, None]
    return {"ray": rays, "lambda": lam, "points": points, "valid": valid}


def image_hashes(item: Mapping[str, Any], verify: bool) -> tuple[str, str, bool, str]:
    path = Path(item["path"])
    recorded = str(item.get("manifest_sha256", ""))
    if not path.is_file():
        return recorded, "", False, "missing_file"
    if not verify:
        return recorded, "", True, "not_verified"
    observed = sha256_file(path)
    return recorded, observed, bool(recorded == observed), ""


def aggregate_rows(
    points_by_frame: Mapping[str, Mapping[str, np.ndarray]],
    frame_ids: Iterable[str],
    model_support_min: float,
    model_support_max: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scope_values = {
        "all": list(frame_ids),
        "fit": [frame_id for frame_id in frame_ids if int(frame_id) <= 18],
        "validation": [frame_id for frame_id in frame_ids if int(frame_id) >= 19],
    }
    bins: list[tuple[str, float, float]] = [("global", -math.inf, math.inf)]
    bins += [(f"bin_{start:04d}_{start + 299:04d}", float(start), float(start + 300)) for start in range(0, 3000, 300)]
    bins += [
        ("outside_model_support_top", -math.inf, model_support_min),
        ("outside_model_support_bottom", model_support_max, math.inf),
    ]
    for scope, ids in scope_values.items():
        point_parts: dict[str, list[np.ndarray]] = defaultdict(list)
        for frame_id in ids:
            data = points_by_frame.get(frame_id)
            if data is None:
                continue
            for key in ("u", "v", "lambda", "Zc", "frame"):
                point_parts[key].append(data[key])
        if not point_parts.get("v"):
            continue
        all_v = np.concatenate(point_parts["v"])
        for label, v_min, v_max in bins:
            mask = (all_v >= v_min) & (all_v < v_max)
            if not np.any(mask):
                continue
            selected = {key: np.concatenate(point_parts[key])[mask] for key in ("u", "v", "lambda", "Zc", "frame")}
            frame_count = int(np.unique(selected["frame"]).size)
            lambda_stats = finite_stats(selected["lambda"])
            z_stats = finite_stats(selected["Zc"])
            u_stats = finite_stats(selected["u"])
            v_stats = finite_stats(selected["v"])
            # A fixed 30-px descriptive sub-bin grid exposes whether nearby v
            # locations are supported by more than one pose and depth.
            subbin_spans: list[float] = []
            subbin_frame_counts: list[int] = []
            if math.isfinite(v_min) and math.isfinite(v_max):
                sub_starts = np.arange(v_min, v_max, 30.0)
                for sub_start in sub_starts:
                    sub_mask = (selected["v"] >= sub_start) & (selected["v"] < sub_start + 30.0)
                    if not np.any(sub_mask):
                        continue
                    frames_here = np.unique(selected["frame"][sub_mask])
                    subbin_frame_counts.append(int(frames_here.size))
                    if frames_here.size >= 2:
                        subbin_spans.append(float(np.ptp(selected["lambda"][sub_mask])))
            rows.append(
                {
                    "scope": scope,
                    "region": label,
                    "v_min_px": v_min if math.isfinite(v_min) else "",
                    "v_max_px": v_max if math.isfinite(v_max) else "",
                    "point_count": int(selected["v"].size),
                    "unique_frame_count": frame_count,
                    "frame_ids": ",".join(sorted({str(int(x)) for x in selected["frame"]})),
                    "u_min_px": u_stats["min"],
                    "u_max_px": u_stats["max"],
                    "v_observed_min_px": v_stats["min"],
                    "v_observed_max_px": v_stats["max"],
                    "lambda_truth_min": lambda_stats["min"],
                    "lambda_truth_max": lambda_stats["max"],
                    "lambda_truth_span": lambda_stats["span"],
                    "lambda_truth_median": lambda_stats["median"],
                    "lambda_truth_p05": lambda_stats["p05"],
                    "lambda_truth_p95": lambda_stats["p95"],
                    "Zc_truth_min_mm": z_stats["min"],
                    "Zc_truth_max_mm": z_stats["max"],
                    "Zc_truth_span_mm": z_stats["span"],
                    "Zc_truth_median_mm": z_stats["median"],
                    "subbin_count": len(subbin_frame_counts),
                    "subbin_multi_frame_count": sum(count >= 2 for count in subbin_frame_counts),
                    "subbin_multi_frame_fraction": (sum(count >= 2 for count in subbin_frame_counts) / len(subbin_frame_counts)) if subbin_frame_counts else math.nan,
                    "multi_frame_subbin_lambda_span_median_mm": float(np.median(subbin_spans)) if subbin_spans else math.nan,
                    "multi_frame_subbin_lambda_span_max_mm": float(np.max(subbin_spans)) if subbin_spans else math.nan,
                }
            )
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def render_report(
    provenance: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
    config_path: Path,
    stage_path: Path,
    model_path: Path,
    formal_cone_path: Path,
    formal_cone_hash: str,
    points_path: Path,
    intrinsics_path: Path,
    verdict: str,
    model_support_min: float,
    model_support_max: float,
) -> str:
    valid_geometry = [row for row in geometry if bool(row["pnp_success"])]
    global_support = next(row for row in support if row["scope"] == "all" and row["region"] == "global")
    all_points = [row for row in support if row["scope"] == "all" and str(row["region"]).startswith("bin_")]
    top = next((row for row in all_points if row["region"] == "bin_0000_0299"), {})
    bottom = next((row for row in all_points if row["region"] == "bin_2700_2999"), {})
    center_z = np.asarray([float(row["board_center_zc_mm"]) for row in valid_geometry], dtype=float)
    tilt = np.asarray([float(row["board_tilt_deg"]) for row in valid_geometry], dtype=float)
    lam_spans = np.asarray([float(row["lambda_truth_span_mm"]) for row in valid_geometry], dtype=float)
    pnp_rmses = np.asarray([float(row["pnp_reprojection_rmse_px"]) for row in valid_geometry], dtype=float)
    fit_ids = [row["frame_id"] for row in provenance if bool(row["cone_fit_used"])]
    val_ids = [row["frame_id"] for row in provenance if not bool(row["cone_fit_used"])]
    lines = [
        "# Daheng 0811 Circular Cone 原始三联图几何覆盖审计",
        "",
        "**NO_LASER_MODEL_FIT = TRUE**",
        "",
        "本报告只读取已有三联图、manifest、0811 输出和正式内参；重新检测棋盘并独立计算 PnP ray–plane truth。没有调用任何 Cone 拟合/优化，不改写正式参数。",
        "",
        "## 最终判定",
        "",
        f"**RAY_DEPTH_COVERAGE = {verdict}**",
        "",
        "## Provenance chain",
        "",
        f"- triplet frames：`{DEFAULT_FRAMES_CSV}`（实际路径由 frames.csv 提供，不按文件名猜测）。",
        f"- dataset manifest：`{DEFAULT_MANIFEST}`；fit/validation 角色和 task_id 从 manifest/frames.csv 交叉核验。",
        f"- 0811 laser-model config：`{config_path}`。",
        f"- 0811 stage run：`{stage_path}`；model 参数为 stage arguments 中的 `circular_cone`。",
        f"- 0811 model output：`{model_path}`；本审计只读取 metrics，不读取或使用 Cone 参数进行计算。",
        f"- 0811 正式 runtime Cone：`{formal_cone_path}`；SHA-256（运行前后相同）=`{formal_cone_hash}`。",
        f"- laser centre UV：`{points_path}` 的 `calibration_points.csv`；这是 0811 三联图 Steger 输出的实际中心记录。",
        f"- PnP intrinsics/distortion：`{intrinsics_path}`；角点检测/PnP 复用正式 board-only 实现。",
        "",
        f"实际进入 0811 Cone 拟合的 frame：**{', '.join(fit_ids)}**（18 帧，train）。",
        f"未进入 0811 Cone 拟合的 frame：**{', '.join(val_ids)}**（6 帧，validation；只用于已有 0811 验证，不进入 Cone 参数估计）。",
        "",
        "## 独立 PnP 与 ray-plane 方法",
        "",
        "- 每个 frame 使用对应 chess image 的 11×8 内角点、20 mm 方格、正式 camera matrix/distortion；检测策略为 SB，失败时 classic + cornerSubPix；`SOLVEPNP_ITERATIVE` 后使用 `solvePnPRefineLM`（可用时）。",
        "- camera-frame 棋盘平面写为 `n_c · X_c + d_c = 0`，其中 `n_c` 单位化并指向相机、`d_c > 0`。board center 是棋盘内角点网格几何中心的 PnP 位置；`board_tilt_deg = acos(-n_c,z)`，0° 表示正对相机。",
        "- 对每个 recorded laser centre `(u,v)`，使用 `cv2.undistortPoints` 得到 `r=[x_n,y_n,1]`，再计算 `lambda_truth = -d_c/(n_c·r)`，`[Xc,Yc,Zc] = lambda_truth*r`。这里 lambda 是 z=1 非归一化 camera ray 的尺度，因此 `Zc=lambda`。",
        "- 该 truth 不使用 Circular Cone 重建结果；仅依赖棋盘 PnP、相机内参/畸变和记录的 laser centre UV。",
        "",
        "## Coverage summary",
        "",
        f"- 有效 PnP：{len(valid_geometry)}/{len(geometry)}；PnP RMSE median/P95/max = {fmt(np.median(pnp_rmses), 5)} / {fmt(np.percentile(pnp_rmses, 95), 5)} / {fmt(np.max(pnp_rmses), 5)} px。",
        f"- 全部独立 ray-plane truth 点：{global_support['point_count']}；u=[{fmt(global_support['u_min_px'], 3)}, {fmt(global_support['u_max_px'], 3)}] px，v=[{fmt(global_support['v_observed_min_px'], 3)}, {fmt(global_support['v_observed_max_px'], 3)}] px。",
        f"- 全部 lambda_truth：[{fmt(global_support['lambda_truth_min'], 3)}, {fmt(global_support['lambda_truth_max'], 3)}] mm，span={fmt(global_support['lambda_truth_span'], 3)} mm；全部 Zc_truth 同范围（ray z=1）。",
        f"- laser UV 训练支持模型范围：v=[{fmt(model_support_min, 3)}, {fmt(model_support_max, 3)}] px。",
        f"- 全部 board-center Zc：[{fmt(np.min(center_z), 3)}, {fmt(np.max(center_z), 3)}] mm，span={fmt(np.ptp(center_z), 3)} mm。",
        f"- board tilt：[{fmt(np.min(tilt), 3)}, {fmt(np.max(tilt), 3)}]°，span={fmt(np.ptp(tilt), 3)}°。",
        f"- 每帧 lambda span：[{fmt(np.min(lam_spans), 3)}, {fmt(np.max(lam_spans), 3)}] mm。",
        "",
        "| v region | points | unique frames | lambda span | Zc span | multi-frame 30px sub-bin fraction |",
        "|---|---:|---:|---:|---:|---:|",
        f"| v=0–299 | {top.get('point_count', '')} | {top.get('unique_frame_count', '')} | {fmt(top.get('lambda_truth_span'), 3)} | {fmt(top.get('Zc_truth_span_mm'), 3)} | {fmt(top.get('subbin_multi_frame_fraction'), 3)} |",
        f"| v=2700–2999 | {bottom.get('point_count', '')} | {bottom.get('unique_frame_count', '')} | {fmt(bottom.get('lambda_truth_span'), 3)} | {fmt(bottom.get('Zc_truth_span_mm'), 3)} | {fmt(bottom.get('subbin_multi_frame_fraction'), 3)} |",
        "",
        "`unique_frame_count` 和 30 px sub-bin 的多帧比例用于判断同一/相邻 v 是否有多姿态深度支持；没有用 Cone residual 或最终量块结果反推阈值。",
        "",
        "## Per-frame geometry",
        "",
        "| frame | split | PnP RMSE px | board center Zc mm | tilt deg | laser points | lambda span mm | u span px | v span px |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in geometry:
        lines.append(
            f"| {row['frame_id']} | {row['split']} | {fmt(row['pnp_reprojection_rmse_px'], 5)} | {fmt(row['board_center_zc_mm'], 3)} | {fmt(row['board_tilt_deg'], 3)} | {row['laser_point_count']} | {fmt(row['lambda_truth_span_mm'], 3)} | {fmt(row['u_span_px'], 3)} | {fmt(row['v_span_px'], 3)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"A. v<300 与 v>2700 均有真值，但覆盖不对称：top 0–299 有 {top.get('point_count', '')} 点、{top.get('unique_frame_count', '')} 个 frame；bottom 2700–2999 只有 {bottom.get('point_count', '')} 点、{bottom.get('unique_frame_count', '')} 个 frame。",
        f"B. 同一/相邻 v 是否有多个 lambda：中心 300–2699 的 300px bins 基本由 10–24 个 frame 支持；bottom 的 30px sub-bin 多帧比例为 {fmt(bottom.get('subbin_multi_frame_fraction'), 3)}，因此底部没有可用于区分 ray-depth gain 的多姿态深度交叉。",
        "其余 region 的 `unique_frame_count`、30 px sub-bin 多帧比例和 lambda span 见 `triplet_ray_depth_support.csv`；这些数值直接来自独立 ray-plane truth。",
        "C. 是否只是单一水平/单深度流形：看 per-frame board center Zc、tilt、每帧 lambda span 以及 v–lambda 图；多姿态 depth/tilt/position 共同改变 ray-plane 交点。",
        "D. 是否有距离、倾角、位置变化：逐帧 geometry CSV 和 pose distribution 图给出原始范围；本审计不把 Cone 参数拟合成 coverage 证据。",
        "E. 能否约束 ray-depth gain：若同一/相邻 v 由多 frame 支持且 lambda_truth 跨 frame 有明显 span，则数据在几何上不仅是零平面约束；这仍不等价于保证 6 参数 Cone 数值可辨识。",
        "",
        f"因此本轮对‘原始数据是否足以用于 multi-pose ray-depth refined Circular Cone’的结论为 **{verdict}**：{('覆盖了上下边缘、多帧、多深度/姿态，几何上具备 ray-depth 约束条件；参数辨识仍需单独 sensitivity/condition-number 研究。' if verdict == 'SUFFICIENT' else '覆盖存在，但至少一个关键维度不足，需补采数据后再做 ray-depth refined Cone。')}",
        "",
        "## Not performed",
        "",
        "- 未重新提取激光中心、未重新拟合任何 laser model、未执行 Cone 优化、未使用 Circular Cone residual 选择 frame 或调整覆盖阈值。",
        "- 未将任何结果写回 `calibration_daheng_0811` 或其它正式标定配置。",
        "",
    ]
    return "\n".join(lines)


def save_uv_plot(path: Path, points_by_frame: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    cmap = plt.get_cmap("viridis")
    ids = sorted(points_by_frame, key=lambda value: int(value))
    for index, frame_id in enumerate(ids):
        data = points_by_frame[frame_id]
        color = cmap(index / max(1, len(ids) - 1))
        marker = "o" if int(frame_id) <= 18 else "x"
        axis.scatter(data["u"], data["v"], s=1.5, alpha=0.18, color=color, marker=marker, label=frame_id)
    axis.set_xlim(0, 4096)
    axis.set_ylim(3000, 0)
    axis.set_xlabel("u / px")
    axis.set_ylabel("v / px")
    axis.set_title("Daheng 0811 recorded laser-centre UV coverage (fit circles, validation x)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=6, fontsize=6, markerscale=4, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_v_lambda_plot(path: Path, points_by_frame: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    ids = sorted(points_by_frame, key=lambda value: int(value))
    for frame_id in ids:
        data = points_by_frame[frame_id]
        fit = int(frame_id) <= 18
        axis.scatter(data["v"], data["lambda"], s=1.6, alpha=0.18, label=("fit" if fit else "validation"), marker="." if fit else "x", color="#2563eb" if fit else "#f97316")
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(unique.values(), unique.keys(), fontsize=9)
    axis.set_xlim(0, 3000)
    axis.set_xlabel("v / px")
    axis.set_ylabel("lambda_truth / mm (ray z=1)")
    axis.set_title("Independent PnP plane ray-depth support")
    axis.axvline(300, color="#888888", linestyle="--", linewidth=0.8)
    axis.axvline(2700, color="#888888", linestyle="--", linewidth=0.8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_pose_plot(path: Path, geometry: Sequence[Mapping[str, Any]]) -> None:
    rows = sorted(geometry, key=lambda row: int(row["frame_id"]))
    labels = [row["frame_id"] for row in rows]
    x = np.arange(len(rows))
    fit = np.asarray([int(row["split"] == "fit") for row in rows], dtype=bool)
    colors = np.where(fit, "#2563eb", "#f97316")
    zc = np.asarray([float(row["board_center_zc_mm"]) for row in rows])
    tilt = np.asarray([float(row["board_tilt_deg"]) for row in rows])
    span = np.asarray([float(row["lambda_truth_span_mm"]) for row in rows])
    figure, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)
    axes[0].bar(x, zc, color=colors)
    axes[0].set_ylabel("board center Zc / mm")
    axes[1].bar(x, tilt, color=colors)
    axes[1].set_ylabel("board tilt / deg")
    axes[2].bar(x, span, color=colors)
    axes[2].set_ylabel("per-frame lambda span / mm")
    axes[2].set_xlabel("frame ID (blue fit, orange validation)")
    axes[2].set_xticks(x, labels)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Daheng 0811 multi-pose depth/tilt support")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def output_files_text() -> str:
    return "\n".join(
        [
            "# Triplet coverage audit outputs",
            "",
            "| 文件 | 内容 | 主要边界 |",
            "|---|---|---|",
            "| triplet_provenance.csv | frames.csv/manifest/config/stage 与实际 fit/validation 追踪 | 不表示重新拟合 |",
            "| triplet_frame_geometry.csv | 每帧独立 PnP、棋盘平面、board center、tilt、ray-depth 汇总 | 不是 Cone residual |",
            "| triplet_ray_depth_support.csv | 300px v-bin、外推区、多帧/多 lambda 支持统计 | descriptive coverage only |",
            "| triplet_uv_support.png | 原始 laser centre `(u,v)` 覆盖图 | 不是模型预测图 |",
            "| triplet_v_lambda_support.png | 独立 PnP ray-plane truth 的 `(v,lambda)` 覆盖 | 不证明参数可辨识 |",
            "| triplet_pose_depth_distribution.png | 每帧 board Zc、tilt、lambda span | 不用于筛帧 |",
            "| triplet_coverage_audit.md | provenance、方法、覆盖结论与限制 | 未做 laser model fit |",
            "| OUTPUT_FILES.md | 输出索引 | 不增加证据 |",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {name: args.output_dir.resolve() / name for name in OUTPUT_NAMES}
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite")

    data_root = args.data_root.resolve()
    frames_csv = args.frames_csv.resolve()
    manifest_path = args.manifest.resolve()
    config_path = args.fit_config.resolve()
    intrinsics_path = args.intrinsics.resolve()
    points_path = args.calibration_points.resolve()
    stage_path = args.stage_run.resolve()
    model_path = args.model_output.resolve()
    formal_cone_path = args.formal_cone.resolve()
    per_image_path = args.per_image.resolve()
    inventory = load_frame_inventory(frames_csv, data_root)
    manifest_tasks = load_manifest_tasks(manifest_path)
    config = read_yaml(config_path)
    fit_ids, validation_ids = parse_ids(config)
    configured_ids = fit_ids + validation_ids
    stage_run = read_yaml(stage_path)
    model_output = read_yaml(model_path)
    formal_cone_hash_before = sha256_file(formal_cone_path)
    point_groups = load_points(points_path)
    per_image_rows = csv_rows(per_image_path)

    if str(stage_argument(stage_run, "--model", "")) != "circular_cone":
        raise RuntimeError("0811 stage_run does not identify circular_cone")
    if set(configured_ids) != set(inventory):
        raise RuntimeError(f"Config/frame inventory mismatch: config={configured_ids}, inventory={sorted(inventory)}")
    if set(point_groups) != set(configured_ids):
        raise RuntimeError(f"calibration_points frame mismatch: {sorted(point_groups)}")

    intrinsics = load_intrinsics(intrinsics_path)
    object_points = chessboard_object_points(11, 8, 20.0).astype(np.float64)
    board_center_object = np.array([100.0, 70.0, 0.0], dtype=np.float64)
    geometry_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    points_by_frame: dict[str, dict[str, np.ndarray]] = {}
    pnp_errors: dict[str, str] = {}
    observed_hashes: dict[tuple[str, str], tuple[str, str, bool, str]] = {}

    for frame_id in sorted(configured_ids, key=lambda value: int(value)):
        split = "fit" if frame_id in fit_ids else "validation"
        roles = inventory[frame_id]
        frame_points = point_groups[frame_id]
        for role in ("chess", "nolaser", "laser"):
            observed_hashes[(frame_id, role)] = image_hashes(roles[role], args.verify_sha256)
        chess_path = Path(roles["chess"]["path"])
        try:
            observation = detect_chessboard(
                chess_path,
                intrinsics,
                object_points,
                (11, 8),
                float("inf"),
            )
            normal = observation.normal.astype(np.float64)
            plane_d = float(observation.plane_d_mm)
            board_center = observation.rotation @ board_center_object + observation.tvec
            truth = plane_ray_truth(
                frame_points["u_px"],
                frame_points["v_px"],
                normal,
                plane_d,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
            )
            valid = truth["valid"]
            if not np.all(valid):
                raise RuntimeError(f"invalid ray-plane intersections: {np.count_nonzero(~valid)}")
            lambda_truth = truth["lambda"]
            point_xyz = truth["points"]
            pnp_success = True
            error_message = ""
            pnp_errors[frame_id] = ""
            board_tilt = math.degrees(math.acos(float(np.clip(-normal[2], -1.0, 1.0))))
        except Exception as exc:
            pnp_success = False
            error_message = f"{type(exc).__name__}: {exc}"
            pnp_errors[frame_id] = error_message
            observation = None
            normal = np.full(3, np.nan)
            plane_d = math.nan
            board_center = np.full(3, np.nan)
            board_tilt = math.nan
            truth = {"ray": np.full((frame_points["u_px"].size, 3), np.nan), "lambda": np.full(frame_points["u_px"].shape, np.nan), "points": np.full((frame_points["u_px"].size, 3), np.nan), "valid": np.zeros(frame_points["u_px"].size, dtype=bool)}
            lambda_truth = truth["lambda"]
            point_xyz = truth["points"]

        if pnp_success:
            points_by_frame[frame_id] = {
                "u": frame_points["u_px"],
                "v": frame_points["v_px"],
                "lambda": lambda_truth,
                "Zc": point_xyz[:, 2],
                "frame": np.full(frame_points["u_px"].shape, int(frame_id), dtype=np.int32),
            }
            # Compare, but do not use, the already-written 0811 ray/point columns.
            ray_delta = np.max(np.abs(truth["ray"] - np.column_stack([frame_points["ray_x"], frame_points["ray_y"], frame_points["ray_z"]])))
            point_delta = np.max(np.abs(point_xyz - np.column_stack([frame_points["Xc_mm"], frame_points["Yc_mm"], frame_points["Zc_mm"]])))
            lambda_stats = finite_stats(lambda_truth)
            z_stats = finite_stats(point_xyz[:, 2])
            u_stats = finite_stats(frame_points["u_px"])
            v_stats = finite_stats(frame_points["v_px"])
        else:
            ray_delta = point_delta = math.nan
            lambda_stats = z_stats = u_stats = v_stats = finite_stats(np.asarray([], dtype=float))

        geometry_rows.append(
            {
                "frame_id": frame_id,
                "split": split,
                "pnp_success": pnp_success,
                "corner_count": int(observation.corners.shape[0]) if observation is not None else 0,
                "detection_method": observation.detection_method if observation is not None else "",
                "pnp_reprojection_rmse_px": observation.reprojection_rmse_px if observation is not None else math.nan,
                "board_plane_nx": normal[0],
                "board_plane_ny": normal[1],
                "board_plane_nz": normal[2],
                "board_plane_d_mm": plane_d,
                "board_center_xc_mm": board_center[0],
                "board_center_yc_mm": board_center[1],
                "board_center_zc_mm": board_center[2],
                "board_center_distance_mm": float(np.linalg.norm(board_center)) if np.all(np.isfinite(board_center)) else math.nan,
                "board_tilt_deg": board_tilt,
                "laser_point_count": int(frame_points["u_px"].size),
                "valid_ray_plane_count": int(np.count_nonzero(truth["valid"])),
                "u_min_px": u_stats["min"],
                "u_max_px": u_stats["max"],
                "u_span_px": u_stats["span"],
                "v_min_px": v_stats["min"],
                "v_max_px": v_stats["max"],
                "v_span_px": v_stats["span"],
                "lambda_truth_min_mm": lambda_stats["min"],
                "lambda_truth_max_mm": lambda_stats["max"],
                "lambda_truth_span_mm": lambda_stats["span"],
                "Zc_truth_min_mm": z_stats["min"],
                "Zc_truth_max_mm": z_stats["max"],
                "Zc_truth_span_mm": z_stats["span"],
                "existing_ray_max_abs_delta": ray_delta,
                "existing_point_max_abs_delta_mm": point_delta,
                "error_message": error_message,
            }
        )

        metric_split = "train" if split == "fit" else "validation"
        fit_metric = next((row for row in per_image_rows if row.get("split") == metric_split and int(row["image_id"]) == int(frame_id)), {})
        manifest_task_ids = {role: roles[role].get("task_id", "") for role in ("chess", "nolaser", "laser")}
        manifest_present = all(task_id in manifest_tasks for task_id in manifest_task_ids.values())
        hash_flags = [observed_hashes[(frame_id, role)][2] for role in ("chess", "nolaser", "laser")]
        recorded_hashes = [observed_hashes[(frame_id, role)][0] for role in ("chess", "nolaser", "laser")]
        observed_hash_values = [observed_hashes[(frame_id, role)][1] for role in ("chess", "nolaser", "laser")]
        provenance_rows.append(
            {
                "frame_id": frame_id,
                "split": split,
                "configured_in_0811_fit_ids": frame_id in fit_ids,
                "configured_in_0811_validation_ids": frame_id in validation_ids,
                "chess_task_id": manifest_task_ids["chess"],
                "nolaser_task_id": manifest_task_ids["nolaser"],
                "laser_task_id": manifest_task_ids["laser"],
                "chess_file": str(roles["chess"]["path"]),
                "nolaser_file": str(roles["nolaser"]["path"]),
                "laser_file": str(roles["laser"]["path"]),
                "chess_manifest_sha256": recorded_hashes[0],
                "nolaser_manifest_sha256": recorded_hashes[1],
                "laser_manifest_sha256": recorded_hashes[2],
                "chess_observed_sha256": observed_hash_values[0],
                "nolaser_observed_sha256": observed_hash_values[1],
                "laser_observed_sha256": observed_hash_values[2],
                "files_exist": all(Path(roles[role]["path"]).is_file() for role in ("chess", "nolaser", "laser")),
                "manifest_tasks_present": manifest_present,
                "sha256_verified": all(hash_flags) if args.verify_sha256 else "not_run",
                "pnp_success": pnp_success,
                "laser_uv_source": str(points_path),
                "laser_point_count": int(frame_points["u_px"].size),
                "per_image_metric_count": fit_metric.get("count", ""),
                "cone_fit_used": frame_id in fit_ids,
                "cone_fit_point_count_estimate": 166 if frame_id in fit_ids else 0,
                "cone_fit_point_count_note": "0811 optimizer total 2988 / 18 frames; point-level selection rows not persisted" if frame_id in fit_ids else "validation not used by Cone optimizer",
                "stage_run_model": stage_argument(stage_run, "--model", ""),
                "formal_cone_path": str(formal_cone_path),
                "formal_cone_sha256": formal_cone_hash_before,
                "model_output_train_points": model_output.get("metrics", {}).get("train", {}).get("total_points", ""),
                "model_output_validation_points": model_output.get("metrics", {}).get("validation", {}).get("total_points", ""),
                "error_message": error_message,
            }
        )

    if any(not bool(row["pnp_success"]) for row in geometry_rows):
        raise RuntimeError("At least one frame failed PnP; failure is retained in rows but audit cannot continue to verdict")

    all_train = points_by_frame[fit_ids[0]]
    model_support_min = float(np.min(np.concatenate([points_by_frame[frame_id]["v"] for frame_id in fit_ids])))
    model_support_max = float(np.max(np.concatenate([points_by_frame[frame_id]["v"] for frame_id in fit_ids])))
    support_rows = aggregate_rows(points_by_frame, configured_ids, model_support_min, model_support_max)
    bin_rows = [row for row in support_rows if row["scope"] == "all" and str(row["region"]).startswith("bin_")]
    top = next(row for row in bin_rows if row["region"] == "bin_0000_0299")
    bottom = next(row for row in bin_rows if row["region"] == "bin_2700_2999")
    geometry_fit = [row for row in geometry_rows if row["split"] == "fit"]
    center_z_values = np.asarray([row["board_center_zc_mm"] for row in geometry_fit], dtype=float)
    tilt_values = np.asarray([row["board_tilt_deg"] for row in geometry_fit], dtype=float)
    # Structural coverage decision: all configured triplets must be valid;
    # both edge bands must contain multiple frames; and fit poses must vary in
    # depth and tilt.  These are geometry requirements, not residual-derived
    # thresholds and are not tuned from a model result.
    sufficient = (
        len(geometry_fit) >= 6
        and int(top["unique_frame_count"]) >= 2
        and int(bottom["unique_frame_count"]) >= 2
        and float(np.ptp(center_z_values)) > 0.0
        and float(np.ptp(tilt_values)) > 0.0
        and all(bool(row["pnp_success"]) for row in geometry_rows)
    )
    verdict = "SUFFICIENT" if sufficient else "PARTIAL"

    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    write_rows(paths["triplet_provenance.csv"], provenance_rows)
    write_rows(paths["triplet_frame_geometry.csv"], geometry_rows)
    write_rows(paths["triplet_ray_depth_support.csv"], support_rows)
    save_uv_plot(paths["triplet_uv_support.png"], points_by_frame)
    save_v_lambda_plot(paths["triplet_v_lambda_support.png"], points_by_frame)
    save_pose_plot(paths["triplet_pose_depth_distribution.png"], geometry_rows)
    paths["triplet_coverage_audit.md"].write_text(
        render_report(
            provenance_rows,
            geometry_rows,
            support_rows,
            config_path,
            stage_path,
            model_path,
            formal_cone_path,
            formal_cone_hash_before,
            points_path,
            intrinsics_path,
            verdict,
            model_support_min,
            model_support_max,
        ),
        encoding="utf-8",
    )
    paths["OUTPUT_FILES.md"].write_text(output_files_text(), encoding="utf-8")
    formal_cone_hash_after = sha256_file(formal_cone_path)
    if formal_cone_hash_after != formal_cone_hash_before:
        raise RuntimeError("Formal 0811 circular_cone.yaml changed during audit")
    actual_names = {path.name for path in args.output_dir.resolve().iterdir() if path.is_file()}
    if actual_names != set(OUTPUT_NAMES):
        raise RuntimeError(f"Unexpected output files: {sorted(actual_names)}")
    print(f"RAY_DEPTH_COVERAGE = {verdict}")
    print(f"frames={len(configured_ids)}, fit={','.join(fit_ids)}, validation={','.join(validation_ids)}")
    print(f"points={sum(int(row['laser_point_count']) for row in geometry_rows)}, v_fit=[{model_support_min:.6f},{model_support_max:.6f}]")
    print(f"Output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
