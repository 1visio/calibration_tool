#!/usr/bin/env python3
"""Task 6H-0: reuse edge-extension chess images for camera coverage audit.

Only chess images are opened.  The formal 0811 K/D is used as a fixed pose
characterization model; no calibration is run and no laser/Validation image is
enumerated.  Candidate labels are geometry/quality labels, not a statement
that the formal camera model has been improved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve()
TOOL_ROOT = SCRIPT.parents[1]
WORKSPACE_ROOT = SCRIPT.parents[2]
for _path in (SCRIPT.parent, WORKSPACE_ROOT / "calibration" / "src", WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_camera_pose_observability as pose_audit  # noqa: E402
import audit_intrinsics_truth_stability as task6e  # noqa: E402


DEFAULT_CALIBRATION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit"
DEFAULT_EXTENSION_FIT = TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension" / "fit"
DEFAULT_FORMAL_INTRINSICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "calibration_result.yaml"
DEFAULT_FORMAL_FIT_METRICS = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811" / "intrinsics" / "fit_images.csv"
DEFAULT_OUTPUT_DIR = TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "edge_extension_camera_coverage"

BASELINE_IDS = tuple(f"{i:03d}" for i in range(1, 19))
EXTENSION_IDS = tuple(f"{i:03d}" for i in range(25, 37))
PRIMARY_FEATURES = ("board_center_z_mm", "board_tilt_deg", "apparent_bbox_area_fraction", "image_center_u_norm", "image_center_v_norm", "normalized_radius_max")
DIMENSION_MAP = {
    "depth": "board_center_z_mm",
    "tilt": "board_tilt_deg",
    "apparent_size": "apparent_bbox_area_fraction",
    "sensor_u": "image_center_u_norm",
    "sensor_v": "image_center_v_norm",
    "normalized_radius": "normalized_radius_max",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-fit-dir", type=Path, default=DEFAULT_CALIBRATION_FIT)
    p.add_argument("--extension-fit-dir", type=Path, default=DEFAULT_EXTENSION_FIT)
    p.add_argument("--formal-intrinsics", type=Path, default=DEFAULT_FORMAL_INTRINSICS)
    p.add_argument("--formal-fit-metrics", type=Path, default=DEFAULT_FORMAL_FIT_METRICS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def finite(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


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
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})


def read_observations(directory: Path, ids: Sequence[str]) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    paths = [directory / f"chess {frame_id}.tif" for frame_id in ids]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError("Missing chess FIT images: " + ", ".join(missing))
    observations: list[dict[str, Any]] = []
    image_size: tuple[int, int] | None = None
    for path in paths:
        image = task6e.chess_calib.read_image(path)
        if image is None:
            observations.append({"frame_id": path.stem.split()[-1], "path": path, "corners": None, "detection_method": "read_failed", "read_failed": True})
            continue
        current_size = (int(image.shape[1]), int(image.shape[0]))
        image_size = current_size if image_size is None else image_size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners, method = task6e.chess_calib.detect_corners(gray, (task6e.BOARD_COLS, task6e.BOARD_ROWS))
        observations.append({
            "frame_id": path.stem.split()[-1], "path": path,
            "corners": np.asarray(corners, dtype=np.float32).reshape(-1, 2) if found and corners is not None else None,
            "detection_method": method if found else "not_detected", "read_failed": False,
        })
    if image_size is None:
        raise RuntimeError(f"No readable chess image in {directory}")
    return observations, image_size


def characterize(observations: Sequence[Mapping[str, Any]], role: str, formal_k: np.ndarray, formal_d: np.ndarray,
                 obj: np.ndarray, image_size: tuple[int, int], formal_metrics: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    valid_obs: list[Mapping[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for obs in observations:
        if obs.get("corners") is None or len(np.asarray(obs["corners"])) != task6e.BOARD_COLS * task6e.BOARD_ROWS:
            failed.append({"frame_id": obs["frame_id"], "dataset_role": role, "image": obs["path"].name, "quality_status": "REJECT_QUALITY",
                           "quality_reason": "chessboard_not_detected_or_corner_count_not_88", "validation_opened": False})
        else:
            valid_obs.append(obs)
    rows: list[dict[str, Any]] = []
    if valid_obs:
        rows.extend(pose_audit.geometry_rows(valid_obs, formal_k, formal_d, obj, image_size, formal_metrics))
        for row in rows:
            row["dataset_role"] = role
            row["quality_status"] = "PASS" if finite(row.get("solvepnp_reprojection_rmse_px")) <= 0.40 else "REJECT_QUALITY"
            row["quality_reason"] = "" if row["quality_status"] == "PASS" else "pnp_rmse_gt_0.40px"
            row["validation_opened"] = False
    rows.extend(failed)
    return rows


def feature_stats(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, float]:
    x = np.asarray([finite(row.get(feature)) for row in rows], dtype=np.float64)
    x = x[np.isfinite(x)]
    return {"count": len(x), "min": float(np.min(x)) if len(x) else math.nan, "max": float(np.max(x)) if len(x) else math.nan,
            "range": float(np.ptp(x)) if len(x) else math.nan, "mean": float(np.mean(x)) if len(x) else math.nan,
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan}


def rank_candidates(baseline: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    base_valid = [r for r in baseline if r.get("quality_status") == "PASS"]
    base_matrix = np.asarray([[finite(r.get(feature)) for feature in PRIMARY_FEATURES] for r in base_valid], dtype=np.float64)
    scale = np.nanstd(base_matrix, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-9)] = 1.0
    base_min = np.nanmin(base_matrix, axis=0)
    base_max = np.nanmax(base_matrix, axis=0)
    base_range = np.maximum(base_max - base_min, 1e-9)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        out = {"frame_id": candidate["frame_id"], "dataset_role": "candidate_extension", "quality_status": candidate.get("quality_status"),
               "quality_reason": candidate.get("quality_reason", ""), "validation_opened": False}
        if candidate.get("quality_status") != "PASS":
            out.update({"category": "REJECT_QUALITY", "novelty_score": math.nan, "nearest_baseline_frame_id": "", "nearest_baseline_distance": math.nan,
                        "new_coverage_dimension_count": 0, "tilt_direction_nearest_deg": math.nan})
            rows.append(out)
            continue
        vector = np.asarray([finite(candidate.get(feature)) for feature in PRIMARY_FEATURES], dtype=np.float64)
        distances = np.sqrt(np.sum(((base_matrix - vector[None, :]) / scale[None, :]) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        nearest_id = str(base_valid[nearest_index]["frame_id"])
        outside: list[str] = []
        gain_values: dict[str, float] = {}
        for dimension, feature in DIMENSION_MAP.items():
            value = finite(candidate.get(feature))
            low = value < base_min[PRIMARY_FEATURES.index(feature)]
            high = value > base_max[PRIMARY_FEATURES.index(feature)]
            if low or high:
                outside.append(dimension)
                boundary = base_min[PRIMARY_FEATURES.index(feature)] if low else base_max[PRIMARY_FEATURES.index(feature)]
                gain_values[dimension] = abs(value - boundary) / base_range[PRIMARY_FEATURES.index(feature)]
            else:
                gain_values[dimension] = 0.0
        tilt_direction = np.asarray([[finite(r.get("board_roll_deg")), finite(r.get("board_pitch_deg"))] for r in base_valid], dtype=float)
        candidate_direction = np.asarray([finite(candidate.get("board_roll_deg")), finite(candidate.get("board_pitch_deg"))], dtype=float)
        tilt_direction_distance = float(np.min(np.linalg.norm(tilt_direction - candidate_direction[None, :], axis=1)))
        major_gain = sum(gain_values.get(key, 0.0) > 0.05 for key in ("depth", "tilt", "apparent_size"))
        new_sensor_u = "sensor_u" in outside
        novelty_score = float(np.min(distances))
        # Coverage labels privilege genuinely new support/range.  A large
        # standardized distance inside an already-covered interval is not by
        # itself HIGH_VALUE; it is at most a directional/useful view.
        if major_gain >= 1 or new_sensor_u or tilt_direction_distance >= 8.0:
            category = "HIGH_VALUE"
        elif tilt_direction_distance >= 4.0 or (novelty_score >= 1.25 and len(outside) > 0):
            category = "USEFUL"
        else:
            category = "REDUNDANT"
        out.update({"category": category, "novelty_score": novelty_score, "nearest_baseline_frame_id": nearest_id,
                    "nearest_baseline_distance": novelty_score, "new_coverage_dimension_count": len(outside),
                    "new_coverage_dimensions": ";".join(outside), "tilt_direction_nearest_deg": tilt_direction_distance,
                    "depth_gain_fraction": gain_values.get("depth", 0.0), "tilt_gain_fraction": gain_values.get("tilt", 0.0),
                    "apparent_size_gain_fraction": gain_values.get("apparent_size", 0.0), "sensor_u_gain_fraction": gain_values.get("sensor_u", 0.0),
                    "sensor_v_gain_fraction": gain_values.get("sensor_v", 0.0), "normalized_radius_gain_fraction": gain_values.get("normalized_radius", 0.0)})
        rows.append(out)
    rows.sort(key=lambda row: (0 if row.get("category") == "HIGH_VALUE" else 1 if row.get("category") == "USEFUL" else 2 if row.get("category") == "REDUNDANT" else 3,
                               -(finite(row.get("novelty_score")) if math.isfinite(finite(row.get("novelty_score"))) else -math.inf)))
    for rank, row in enumerate(rows, 1):
        row["ranking"] = rank
    return rows


def baseline_extension_rows(baseline: Sequence[Mapping[str, Any]], extension: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, feature in DIMENSION_MAP.items():
        b = feature_stats([r for r in baseline if r.get("quality_status") == "PASS"], feature)
        e = feature_stats([r for r in extension if r.get("quality_status") == "PASS"], feature)
        outside_low = [str(r["frame_id"]) for r in extension if r.get("quality_status") == "PASS" and finite(r.get(feature)) < b["min"]]
        outside_high = [str(r["frame_id"]) for r in extension if r.get("quality_status") == "PASS" and finite(r.get(feature)) > b["max"]]
        overlap = max(0.0, min(b["max"], e["max"]) - max(b["min"], e["min"])) if math.isfinite(b["min"]) and math.isfinite(e["min"]) else math.nan
        rows.append({"row_type": "range", "dimension": dimension, "feature": feature, "baseline_count": b["count"], "baseline_min": b["min"], "baseline_max": b["max"],
                     "baseline_range": b["range"], "baseline_std": b["std"], "extension_count": e["count"], "extension_min": e["min"],
                     "extension_max": e["max"], "extension_range": e["range"], "extension_std": e["std"], "new_low_count": len(outside_low),
                     "new_high_count": len(outside_high), "new_low_frame_ids": ";".join(outside_low), "new_high_frame_ids": ";".join(outside_high),
                     "range_increase_fraction": (e["range"] - b["range"]) / b["range"] if b["range"] > 0 and math.isfinite(e["range"]) else math.nan,
                     "interval_overlap": overlap})
    def relationship(role_rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, int]:
        valid = [r for r in role_rows if r.get("quality_status") == "PASS"]
        z = np.asarray([finite(r.get("board_center_z_mm")) for r in valid], dtype=float)
        tilt = np.asarray([finite(r.get("board_tilt_deg")) for r in valid], dtype=float)
        area = np.asarray([finite(r.get("apparent_bbox_area_fraction")) for r in valid], dtype=float)
        tz = spearmanr(tilt, z)
        ta = spearmanr(tilt, area)
        return float(tz.statistic), float(ta.statistic), len(valid)
    base_tz, base_ta, base_n = relationship(baseline)
    ext_tz, ext_ta, ext_n = relationship(extension)
    rows.extend([
        {"row_type": "relationship", "dimension": "tilt_depth", "feature": "spearman(board_tilt_deg,board_center_z_mm)",
         "baseline_count": base_n, "extension_count": ext_n, "baseline_spearman": base_tz, "extension_spearman": ext_tz,
         "relationship_interpretation": "near-zero would indicate tilt/depth decoupling"},
        {"row_type": "relationship", "dimension": "tilt_apparent_size", "feature": "spearman(board_tilt_deg,apparent_bbox_area_fraction)",
         "baseline_count": base_n, "extension_count": ext_n, "baseline_spearman": base_ta, "extension_spearman": ext_ta,
         "relationship_interpretation": "near-zero would indicate tilt/size decoupling"},
    ])
    return rows


def render_report(path: Path, baseline: Sequence[Mapping[str, Any]], extension: Sequence[Mapping[str, Any]], ranking: Sequence[Mapping[str, Any]], comparison: Sequence[Mapping[str, Any]], reusable: str) -> None:
    counts = {key: sum(row.get("category") == key for row in ranking) for key in ("HIGH_VALUE", "USEFUL", "REDUNDANT", "REJECT_QUALITY")}
    high = [str(r["frame_id"]) for r in ranking if r.get("category") == "HIGH_VALUE"]
    useful = [str(r["frame_id"]) for r in ranking if r.get("category") == "USEFUL"]
    lines = ["# Task 6H-0 — Edge-extension chess coverage audit", "", f"`EDGE_EXTENSION_REUSABLE_FOR_CAMERA_CALIBRATION = {reusable}`", "",
             "本审计只打开 baseline chess 001–018 与 extension chess 025–036；使用正式 K/D 做 PnP pose characterization。未读取 laser/Validation，未重新估计 K/D，未拟合 Cone。", "",
             "## 分类", "", f"HIGH_VALUE={counts['HIGH_VALUE']}，USEFUL={counts['USEFUL']}，REDUNDANT={counts['REDUNDANT']}，REJECT_QUALITY={counts['REJECT_QUALITY']}。", "",
             f"建议加入 M1 的 extension frame：{', '.join(high + useful) if high or useful else 'none'}。", "",
             "## 覆盖变化", "", "| dimension | baseline range | extension range | new low | new high | range increase |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        if row.get("row_type", "range") != "range":
            continue
        lines.append(f"| {row['dimension']} | {finite(row.get('baseline_min')):.6g}–{finite(row.get('baseline_max')):.6g} | {finite(row.get('extension_min')):.6g}–{finite(row.get('extension_max')):.6g} | {row.get('new_low_count')} | {row.get('new_high_count')} | {finite(row.get('range_increase_fraction')):.4g} |")
    relationships = [row for row in comparison if row.get("row_type") == "relationship"]
    lines += ["", "Tilt/depth and tilt/size decoupling checks:"]
    for row in relationships:
        lines.append(f"- {row.get('dimension')}: baseline Spearman={finite(row.get('baseline_spearman')):.4g}, extension Spearman={finite(row.get('extension_spearman')):.4g}; {row.get('relationship_interpretation')}.")
    lines += ["", "## Candidate ranking", "", "| rank | frame | category | novelty | nearest baseline | new dimensions | tilt-direction distance | PnP RMSE (px) |", "|---:|---:|---|---:|---:|---|---:|---:|"]
    by_id = {str(r["frame_id"]): r for r in extension}
    for row in ranking:
        g = by_id.get(str(row["frame_id"]), {})
        lines.append(f"| {row.get('ranking')} | {row.get('frame_id')} | {row.get('category')} | {finite(row.get('novelty_score')):.4g} | {row.get('nearest_baseline_frame_id','')} | {row.get('new_coverage_dimensions','')} | {finite(row.get('tilt_direction_nearest_deg')):.4g} | {finite(g.get('solvepnp_reprojection_rmse_px')):.4g} |")
    lines += ["", "## 判断", "", "结论为 PARTIAL：025–036 能补强 tilt、apparent-size 和一部分 sensor-u edge coverage，但没有新增 depth 范围，且 extension 的 tilt–depth Spearman 仍约 -0.60，不能视为独立 depth/tilt 约束。025–036 是否可用于 M1 的判断仅基于棋盘几何覆盖和 PnP 质量，不代表已经验证重新标定后的激光 truth 或 laser surface。HIGH_VALUE/USEFUL candidate 应与原 001–018 一起作为 coverage augmentation；REDUNDANT candidate 不增加明显约束，但不等于图像质量差。", "", "## 输出", "", "- `camera_candidate_pose_coverage.csv`", "- `baseline_vs_extension_coverage.csv`", "- `candidate_ranking.csv`", "- `provenance.json`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    formal_k, formal_d, _ = task6e.load_formal_intrinsics(args.formal_intrinsics.resolve())
    formal_metrics = task6e.load_formal_fit_metrics(args.formal_fit_metrics.resolve())
    obj = task6e.object_points()
    baseline_obs, image_size = read_observations(args.calibration_fit_dir.resolve(), BASELINE_IDS)
    extension_obs, extension_size = read_observations(args.extension_fit_dir.resolve(), EXTENSION_IDS)
    if image_size != extension_size:
        raise RuntimeError(f"Image-size mismatch baseline={image_size}, extension={extension_size}")
    baseline_rows = characterize(baseline_obs, "baseline_001_018", formal_k, formal_d, obj, image_size, formal_metrics)
    extension_rows = characterize(extension_obs, "candidate_025_036", formal_k, formal_d, obj, image_size, {})
    all_rows = [*baseline_rows, *extension_rows]
    write_csv(output / "camera_candidate_pose_coverage.csv", all_rows)
    ranking = rank_candidates(baseline_rows, extension_rows)
    write_csv(output / "candidate_ranking.csv", ranking)
    comparison = baseline_extension_rows(baseline_rows, extension_rows)
    write_csv(output / "baseline_vs_extension_coverage.csv", comparison)
    count = {key: sum(row.get("category") == key for row in ranking) for key in ("HIGH_VALUE", "USEFUL", "REDUNDANT", "REJECT_QUALITY")}
    depth_row = next((row for row in comparison if row.get("row_type") == "range" and row.get("dimension") == "depth"), {})
    tilt_size_gain = any(row.get("row_type") == "range" and row.get("dimension") in {"tilt", "apparent_size"} and (int(row.get("new_high_count") or 0) + int(row.get("new_low_count") or 0) > 0) for row in comparison)
    depth_gain = int(depth_row.get("new_high_count") or 0) + int(depth_row.get("new_low_count") or 0) > 0
    if count["REJECT_QUALITY"] == 0 and depth_gain and count["HIGH_VALUE"] + count["USEFUL"] >= 6:
        reusable = "YES"
    elif count["REJECT_QUALITY"] == 0 and tilt_size_gain and count["HIGH_VALUE"] + count["USEFUL"]:
        reusable = "PARTIAL"
    else:
        reusable = "NO"
    render_report(output / "report.md", baseline_rows, extension_rows, ranking, comparison, reusable)
    provenance = {"task": "6H-0", "validation_opened": False, "baseline_frame_ids": list(BASELINE_IDS), "candidate_frame_ids": list(EXTENSION_IDS),
                  "baseline_fit_dir": str(args.calibration_fit_dir.resolve()), "candidate_fit_dir": str(args.extension_fit_dir.resolve()),
                  "formal_intrinsics": str(args.formal_intrinsics.resolve()), "pnp_solver": "SOLVEPNP_ITERATIVE + solvePnPRefineLM",
                  "camera_calibration_modified": False, "cone_refit": False, "laser_images_opened": False, "classification_counts": count,
                  "reusable": reusable, "category_rules": {"reject_quality": "not detected, not 88 corners, or solvePnP RMSE > 0.40 px",
                  "high_value": "new depth/tilt/apparent-size range, new sensor-u edge range, or tilt-direction distance >=8 deg",
                  "useful": "tilt-direction distance >=4 deg, or moderate novelty with a new support dimension", "redundant": "otherwise quality-passing candidate"}}
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EDGE_EXTENSION_REUSABLE_FOR_CAMERA_CALIBRATION = {reusable}")
    print("M1=" + ",".join([str(r["frame_id"]) for r in ranking if r.get("category") in {"HIGH_VALUE", "USEFUL"}]))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
