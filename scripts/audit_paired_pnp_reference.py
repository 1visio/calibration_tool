#!/usr/bin/env python3
"""Audit strictly paired chess/laser captures and build a PnP plane reference.

This script intentionally does not inspect laser pixels, fit a laser plane, build
``b(v)``, or perform compensation.  Chessboard detection and PnP are delegated
to the existing board-only ground-extrinsics implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
BOARD_ONLY_SRC = WORKSPACE_ROOT / "calibration" / "src"
if str(BOARD_ONLY_SRC) not in sys.path:
    sys.path.insert(0, str(BOARD_ONLY_SRC))

board_only = importlib.import_module("calibrate_ground_extrinsics_board_only")

DEFAULT_DATA_ROOT = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "extrinsics0813"
)
DEFAULT_CALIBRATION_ROOT = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0811"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "pnp_reference_audit"
)
EXPECTED_IDS = {
    "fit": tuple(range(1, 11)),
    "validation": tuple(range(11, 14)),
}


@dataclass(frozen=True)
class PairEvidence:
    files_ok: bool
    manifest_ok: bool
    chess_sha256_ok: bool
    laser_sha256_ok: bool
    capture_gap_s: float
    errors: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired Daheng captures and produce a PnP reference only."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the three named audit outputs if they already exist.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_rows(path: Path) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    indexed: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = Path(row["filename"])
            split = filename.parts[0] if filename.parts else ""
            key = (split, row.get("pose_id", ""), row.get("role", ""))
            indexed.setdefault(key, []).append(row)
    return indexed


def validate_pair(
    data_root: Path,
    manifest: dict[tuple[str, str, str], list[dict[str, str]]],
    split: str,
    frame_id: str,
) -> PairEvidence:
    chess_path = data_root / split / f"chess {frame_id}.tif"
    laser_path = data_root / split / f"laser {frame_id}.tif"
    errors: list[str] = []
    files_ok = chess_path.is_file() and laser_path.is_file()
    if not chess_path.is_file():
        errors.append(f"missing {chess_path.relative_to(data_root)}")
    if not laser_path.is_file():
        errors.append(f"missing {laser_path.relative_to(data_root)}")

    chess_rows = manifest.get((split, frame_id, "chess"), [])
    laser_rows = manifest.get((split, frame_id, "laser"), [])
    manifest_ok = len(chess_rows) == 1 and len(laser_rows) == 1
    if len(chess_rows) != 1:
        errors.append(f"frames.csv chess row count={len(chess_rows)}")
    if len(laser_rows) != 1:
        errors.append(f"frames.csv laser row count={len(laser_rows)}")

    chess_sha_ok = False
    laser_sha_ok = False
    capture_gap_s = float("nan")
    if len(chess_rows) == 1:
        row = chess_rows[0]
        expected = f"{split}/chess {frame_id}.tif"
        if row.get("filename", "").replace("\\", "/") != expected:
            manifest_ok = False
            errors.append("frames.csv chess filename mismatch")
        if chess_path.is_file():
            chess_sha_ok = sha256_file(chess_path) == row.get("sha256", "")
            if not chess_sha_ok:
                errors.append("chess SHA-256 mismatch")
    if len(laser_rows) == 1:
        row = laser_rows[0]
        expected = f"{split}/laser {frame_id}.tif"
        if row.get("filename", "").replace("\\", "/") != expected:
            manifest_ok = False
            errors.append("frames.csv laser filename mismatch")
        if laser_path.is_file():
            laser_sha_ok = sha256_file(laser_path) == row.get("sha256", "")
            if not laser_sha_ok:
                errors.append("laser SHA-256 mismatch")
    if len(chess_rows) == 1 and len(laser_rows) == 1:
        chess_ns = int(chess_rows[0]["host_timestamp_ns"])
        laser_ns = int(laser_rows[0]["host_timestamp_ns"])
        capture_gap_s = (laser_ns - chess_ns) / 1.0e9
        if capture_gap_s < 0.0:
            manifest_ok = False
            errors.append("laser capture precedes chess capture")

    return PairEvidence(
        files_ok=files_ok,
        manifest_ok=manifest_ok,
        chess_sha256_ok=chess_sha_ok,
        laser_sha256_ok=laser_sha_ok,
        capture_gap_s=capture_gap_s,
        errors=tuple(errors),
    )


def strict_inventory_errors(data_root: Path) -> list[str]:
    errors: list[str] = []
    for split, numeric_ids in EXPECTED_IDS.items():
        expected = {
            f"{role} {frame_id:03d}.tif"
            for frame_id in numeric_ids
            for role in ("chess", "laser")
        }
        actual = {
            path.name
            for path in (data_root / split).glob("*.tif")
            if path.is_file()
        }
        for name in sorted(expected - actual):
            errors.append(f"{split}: missing {name}")
        for name in sorted(actual - expected):
            errors.append(f"{split}: unexpected {name}")
    return errors


def transform_plane_to_ground(
    normal_camera: np.ndarray,
    d_camera: float,
    transform_ground_from_camera: np.ndarray,
) -> np.ndarray:
    plane_camera = np.r_[normal_camera, float(d_camera)]
    plane_ground = np.linalg.inv(transform_ground_from_camera).T @ plane_camera
    norm = float(np.linalg.norm(plane_ground[:3]))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("transformed board plane is degenerate")
    plane_ground /= norm
    if plane_ground[2] < 0.0:
        plane_ground = -plane_ground
    return plane_ground


def transform_point_to_ground(
    point_camera: np.ndarray, transform_ground_from_camera: np.ndarray
) -> np.ndarray:
    point_h = np.r_[np.asarray(point_camera, dtype=np.float64), 1.0]
    result = transform_ground_from_camera @ point_h
    return result[:3] / result[3]


def finite_values(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def metric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("median", "p95", "max")}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def height_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            key: float("nan")
            for key in ("median", "min", "max", "range", "std", "p95_abs_from_median")
        }
    median = float(np.median(values))
    return {
        "median": median,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.ptp(values)),
        "std": float(np.std(values)),
        "p95_abs_from_median": float(np.percentile(np.abs(values - median), 95)),
    }


def format_number(value: Any, digits: int = 6) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_id",
        "split",
        "corner_count",
        "pnp_success",
        "reprojection_rmse_px",
        "plane_nx",
        "plane_ny",
        "plane_nz",
        "plane_d",
        "tilt_deg",
        "board_height_at_center_mm",
        "detection_method",
        "pair_files_ok",
        "pair_manifest_ok",
        "chess_sha256_ok",
        "laser_sha256_ok",
        "capture_gap_s",
        "chess_file",
        "laser_file",
        "error_message",
    ]
    numeric_fields = {
        "reprojection_rmse_px",
        "plane_nx",
        "plane_ny",
        "plane_nz",
        "plane_d",
        "tilt_deg",
        "board_height_at_center_mm",
        "capture_gap_s",
    }
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            for key in numeric_fields:
                rendered[key] = format_number(row.get(key))
            writer.writerow({key: rendered.get(key, "") for key in fieldnames})


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows))
    labels = [row["frame_id"] for row in rows]
    colors = ["#2563eb" if row["split"] == "fit" else "#f97316" for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    panels = (
        ("reprojection_rmse_px", "Reprojection RMSE (px)"),
        ("tilt_deg", "Board tilt to ground +Z (deg)"),
        ("board_height_at_center_mm", "Board center height Zg (mm)"),
    )
    for axis, (key, ylabel) in zip(axes, panels):
        values = np.asarray([row.get(key, float("nan")) for row in rows], dtype=float)
        good = np.isfinite(values)
        axis.plot(x[good], values[good], color="#94a3b8", linewidth=1.0, zorder=1)
        axis.scatter(x[good], values[good], c=np.asarray(colors)[good], s=42, zorder=2)
        failed = ~good
        if failed.any():
            axis.scatter(
                x[failed],
                np.zeros(int(failed.sum())),
                marker="x",
                color="#dc2626",
                s=60,
                label="PnP failed",
            )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=0)
    axes[-1].set_xlabel("Frame ID (001-010 fit; 011-013 validation)")
    fig.suptitle("Daheng paired PnP reference audit — 2026-08-13 captures")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scope_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row["pnp_success"]]
    return {
        "total": len(rows),
        "success": len(success),
        "rmse": metric_summary(finite_values(success, "reprojection_rmse_px")),
        "tilt": metric_summary(finite_values(success, "tilt_deg")),
        "height": height_summary(finite_values(success, "board_height_at_center_mm")),
    }


def render_report(
    rows: list[dict[str, Any]],
    inventory_errors: list[str],
    intrinsics_path: Path,
    extrinsics_path: Path,
    data_root: Path,
) -> str:
    by_scope = {
        "overall": scope_stats(rows),
        "fit": scope_stats([row for row in rows if row["split"] == "fit"]),
        "validation": scope_stats(
            [row for row in rows if row["split"] == "validation"]
        ),
    }
    pair_failures = [
        row
        for row in rows
        if not (
            row["pair_files_ok"]
            and row["pair_manifest_ok"]
            and row["chess_sha256_ok"]
            and row["laser_sha256_ok"]
        )
    ]
    pnp_failures = [row for row in rows if not row["pnp_success"]]
    strict_pairing = not inventory_errors and not pair_failures

    lines = [
        "# Paired PnP reference audit",
        "",
        "## 结论",
        "",
        f"- 严格配对核验：**{'通过' if strict_pairing else '失败'}**。",
        f"- PnP：**{by_scope['overall']['success']}/{by_scope['overall']['total']} 成功 "
        f"({100.0 * by_scope['overall']['success'] / by_scope['overall']['total']:.1f}%)**。",
        "- 本审计只读取棋盘图建立 PnP 真值；laser 图只用于配对、清单与 SHA-256 核验。",
        "- 未读取激光像素建立参考平面，未生成 `b(v)`，未执行 compensation。",
        "",
        "## 关键统计",
        "",
        "| split | PnP success | reprojection RMSE median / P95 / max (px) | tilt median / P95 / max (deg) | board center height median / min / max (mm) | height range / std / P95 abs dev (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("fit", "validation", "overall"):
        stats = by_scope[name]
        rmse = stats["rmse"]
        tilt = stats["tilt"]
        height = stats["height"]
        lines.append(
            f"| {name} | {stats['success']}/{stats['total']} | "
            f"{format_number(rmse['median'])} / {format_number(rmse['p95'])} / {format_number(rmse['max'])} | "
            f"{format_number(tilt['median'])} / {format_number(tilt['p95'])} / {format_number(tilt['max'])} | "
            f"{format_number(height['median'])} / {format_number(height['min'])} / {format_number(height['max'])} | "
            f"{format_number(height['range'])} / {format_number(height['std'])} / {format_number(height['p95_abs_from_median'])} |"
        )

    lines.extend(
        [
            "",
            "`height variation` 在这里同时报告 range、population std 和相对中位数的 P95 绝对偏差；"
            "`board_height_at_center_mm` 是棋盘内角点网格几何中心经 PnP 后转换到 ground frame 的 `Zg`。",
            "",
            "## 逐帧结果",
            "",
            "| frame | split | pair | corners | PnP | RMSE px | tilt deg | center Zg mm | method |",
            "|---:|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        pair_ok = (
            row["pair_files_ok"]
            and row["pair_manifest_ok"]
            and row["chess_sha256_ok"]
            and row["laser_sha256_ok"]
        )
        lines.append(
            f"| {row['frame_id']} | {row['split']} | {'OK' if pair_ok else 'FAIL'} | "
            f"{row['corner_count']} | {'OK' if row['pnp_success'] else 'FAIL'} | "
            f"{format_number(row['reprojection_rmse_px'])} | {format_number(row['tilt_deg'])} | "
            f"{format_number(row['board_height_at_center_mm'])} | {row['detection_method']} |"
        )

    lines.extend(["", "## 失败与异常", ""])
    if not inventory_errors and not pair_failures and not pnp_failures:
        lines.append("无。13 组严格配对，13 帧 PnP 全部成功。")
    else:
        for error in inventory_errors:
            lines.append(f"- inventory: {error}")
        for row in pair_failures:
            lines.append(f"- frame {row['frame_id']} pairing: {row['error_message']}")
        for row in pnp_failures:
            lines.append(f"- frame {row['frame_id']} PnP: {row['error_message']}")

    lines.extend(
        [
            "",
            "## 配对证据",
            "",
            "每个编号要求 chess/laser 文件各一份，并要求 `frames.csv` 中相同 `pose_id` 下 "
            "`role=chess` 与 `role=laser` 各恰好一行、记录的相对文件名一致、两份文件 SHA-256 "
            "与采集记录一致，且 laser 的采集时间不早于 chess。详细布尔值与 capture gap 见 CSV。",
            "",
            "## 方法与坐标定义",
            "",
            f"- 数据：`{data_root}`",
            f"- 冻结内参：`{intrinsics_path}`",
            f"- 冻结 ground 变换：`{extrinsics_path}`",
            "- 棋盘：11 × 8 内角点，20.0 mm 方格。",
            "- 复用实现：`calibration/src/calibrate_ground_extrinsics_board_only.py` 的 "
            "`load_intrinsics`、`chessboard_object_points`、`detect_chessboard`；检测策略为 "
            "SB，失败时 classic + cornerSubPix；PnP 为 `SOLVEPNP_ITERATIVE`，可用时再 "
            "`solvePnPRefineLM`。",
            "- camera-frame 棋盘平面为 `n_c · X_c + d_c = 0`；通过冻结的 "
            "`T_ground_from_camera` 作平面协向量变换 `pi_g = T^{-T} pi_c`，归一化并令 "
            "`n_g · +Zg >= 0`。CSV 的 `plane_nx..plane_d` 均为 ground-frame 系数，单位法向，"
            "`plane_d` 单位 mm。",
            "- `tilt_deg = acos(clamp(n_board_ground · [0,0,1], -1, 1))`。",
            "",
            "## 输出",
            "",
            "- `paired_pnp_reference_audit.csv`",
            "- `paired_pnp_reference_report.md`",
            "- `pnp_pose_summary.png`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    calibration_root = args.calibration_root.resolve()
    output_dir = args.output_dir.resolve()
    intrinsics_path = calibration_root / "intrinsics" / "calibration_result.yaml"
    extrinsics_path = (
        calibration_root / "ground_extrinsics" / "camera_ground_extrinsics.yaml"
    )
    frames_csv = data_root / "frames.csv"
    outputs = (
        output_dir / "paired_pnp_reference_audit.csv",
        output_dir / "paired_pnp_reference_report.md",
        output_dir / "pnp_pose_summary.png",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite: " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_document = load_yaml(intrinsics_path)
    cols = int(intrinsics_document["pattern_cols"])
    rows_count = int(intrinsics_document["pattern_rows"])
    square_size_mm = float(intrinsics_document["square_size_mm"])
    intrinsics = board_only.load_intrinsics(intrinsics_path)
    object_points = board_only.chessboard_object_points(cols, rows_count, square_size_mm)

    extrinsics_document = load_yaml(extrinsics_path)
    transform = np.asarray(
        extrinsics_document["T_ground_from_camera"], dtype=np.float64
    )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_ground_from_camera must be a finite 4x4 matrix")

    manifest = load_manifest_rows(frames_csv)
    inventory_errors = strict_inventory_errors(data_root)
    audit_rows: list[dict[str, Any]] = []
    board_center_object = np.asarray(
        [
            0.5 * (cols - 1) * square_size_mm,
            0.5 * (rows_count - 1) * square_size_mm,
            0.0,
        ],
        dtype=np.float64,
    )

    for split, numeric_ids in EXPECTED_IDS.items():
        for numeric_id in numeric_ids:
            frame_id = f"{numeric_id:03d}"
            chess_path = data_root / split / f"chess {frame_id}.tif"
            laser_path = data_root / split / f"laser {frame_id}.tif"
            evidence = validate_pair(data_root, manifest, split, frame_id)
            row: dict[str, Any] = {
                "frame_id": frame_id,
                "split": split,
                "corner_count": 0,
                "pnp_success": False,
                "reprojection_rmse_px": float("nan"),
                "plane_nx": float("nan"),
                "plane_ny": float("nan"),
                "plane_nz": float("nan"),
                "plane_d": float("nan"),
                "tilt_deg": float("nan"),
                "board_height_at_center_mm": float("nan"),
                "detection_method": "",
                "pair_files_ok": evidence.files_ok,
                "pair_manifest_ok": evidence.manifest_ok,
                "chess_sha256_ok": evidence.chess_sha256_ok,
                "laser_sha256_ok": evidence.laser_sha256_ok,
                "capture_gap_s": evidence.capture_gap_s,
                "chess_file": str(chess_path),
                "laser_file": str(laser_path),
                "error_message": "; ".join(evidence.errors),
            }
            if chess_path.is_file():
                try:
                    observation = board_only.detect_chessboard(
                        chess_path,
                        intrinsics,
                        object_points,
                        (cols, rows_count),
                        float("inf"),
                    )
                    plane_ground = transform_plane_to_ground(
                        observation.normal, observation.plane_d_mm, transform
                    )
                    center_camera = (
                        observation.rotation @ board_center_object + observation.tvec
                    )
                    center_ground = transform_point_to_ground(center_camera, transform)
                    tilt_deg = math.degrees(
                        math.acos(float(np.clip(plane_ground[2], -1.0, 1.0)))
                    )
                    row.update(
                        {
                            "corner_count": int(observation.corners.shape[0]),
                            "pnp_success": True,
                            "reprojection_rmse_px": observation.reprojection_rmse_px,
                            "plane_nx": float(plane_ground[0]),
                            "plane_ny": float(plane_ground[1]),
                            "plane_nz": float(plane_ground[2]),
                            "plane_d": float(plane_ground[3]),
                            "tilt_deg": tilt_deg,
                            "board_height_at_center_mm": float(center_ground[2]),
                            "detection_method": observation.detection_method,
                        }
                    )
                except Exception as exc:  # Per-frame failure must remain in the audit.
                    message = f"PnP failed: {type(exc).__name__}: {exc}"
                    row["error_message"] = "; ".join(
                        item for item in (row["error_message"], message) if item
                    )
            audit_rows.append(row)

    write_csv(outputs[0], audit_rows)
    outputs[1].write_text(
        render_report(
            audit_rows,
            inventory_errors,
            intrinsics_path,
            extrinsics_path,
            data_root,
        ),
        encoding="utf-8",
    )
    plot_summary(outputs[2], audit_rows)

    successes = sum(bool(row["pnp_success"]) for row in audit_rows)
    strict_pairs = sum(
        bool(
            row["pair_files_ok"]
            and row["pair_manifest_ok"]
            and row["chess_sha256_ok"]
            and row["laser_sha256_ok"]
        )
        for row in audit_rows
    )
    print(f"strict pairs: {strict_pairs}/{len(audit_rows)}")
    print(f"PnP success: {successes}/{len(audit_rows)}")
    for path in outputs:
        print(path)
    return 0 if strict_pairs == len(audit_rows) and successes == len(audit_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
