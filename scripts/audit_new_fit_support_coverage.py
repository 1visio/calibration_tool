#!/usr/bin/env python3
"""Audit Frozen C1 v/s support for the new FIT poses 049--054.

Only ``fit/049--054`` is opened.  The script uses the existing PnP/Steger
extraction and Frozen Circular Cone validity path to obtain laser-center
points, then maps them to the already frozen C1 PCA ``s`` coordinate.  It
does not fit or modify K/D, the Cone, PCA, or C1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


NEW_FIT_IDS = tuple(f"{value:03d}" for value in range(49, 55))
TOP_IDS = ("049", "050", "051")
BOTTOM_IDS = ("052", "053", "054")
EDGE_BY_FRAME = {frame_id: "Top" for frame_id in TOP_IDS} | {
    frame_id: "Bottom" for frame_id in BOTTOM_IDS
}
ROLES = ("chess", "nolaser", "laser")

# These are the previously established 20 mm / 50 mm height-position domains
# plus the acquisition guard.  They are deliberately kept as audit inputs;
# no standard-object image is opened in this run.
EDGE_TARGETS: dict[str, dict[str, list[float]]] = {
    "Top": {
        "observed_v_px": [86.0, 467.0],
        "observed_s": [-0.18500106019725174, -0.13286152907480264],
        "safe_v_px": [30.0, 520.0],
        "safe_s": [-0.191, -0.127],
    },
    "Bottom": {
        "observed_v_px": [2810.0, 2938.0],
        "observed_s": [0.1863643564578584, 0.2037112277937999],
        "safe_v_px": [2760.0, 2990.0],
        "safe_s": [0.181, 0.209],
    },
}

DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane_0817"
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_FIT_SUPPORT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "c1_support_comparison"
    / "c1_support_comparison.csv"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "new_fit_support_coverage_049_054"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument("--fit-support", type=Path, default=DEFAULT_FIT_SUPPORT)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=base.DEFAULT_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=base.DEFAULT_FORMAL_CONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def inventory_new_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve only the six requested FIT triplets; never enumerate validation."""
    fit_root = data_root / "fit"
    groups: dict[str, dict[str, Any]] = {}
    for frame_id in NEW_FIT_IDS:
        groups[frame_id] = {}
        for role in ROLES:
            path = fit_root / f"{role} {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            groups[frame_id][role] = {
                "path": path,
                "filename": str(path.relative_to(data_root)).replace("\\", "/"),
                "quality_warnings": "",
                "manifest_sha256": "",
            }
    return groups


def read_fit_quality_metadata(data_root: Path) -> dict[str, Any]:
    """Read exactly the first 18 data rows, which are FIT 049--054.

    The acquisition ``frames.csv`` is followed by validation rows.  Reading
    line-by-line and stopping after the known FIT row count ensures this audit
    does not inspect those rows.
    """
    path = data_root / "frames.csv"
    if not path.is_file():
        return {"available": False, "rows_read": 0, "validation_rows_read": False, "rows": []}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            return {"available": False, "rows_read": 0, "validation_rows_read": False, "rows": []}
        headers = next(csv.reader([header_line]))
        expected_count = len(NEW_FIT_IDS) * len(ROLES)
        while len(rows) < expected_count:
            line = handle.readline()
            if not line:
                break
            values = next(csv.reader([line]))
            if not values or not any(str(value).strip() for value in values):
                continue
            rows.append({key: values[index] if index < len(values) else "" for index, key in enumerate(headers)})
    expected_keys = {(frame_id, role) for frame_id in NEW_FIT_IDS for role in ROLES}
    actual_keys = {(row.get("pose_id", ""), row.get("role", "")) for row in rows}
    if actual_keys != expected_keys:
        raise RuntimeError(f"FIT quality metadata does not contain exactly 049--054: {sorted(actual_keys)}")
    per_frame: dict[str, dict[str, Any]] = {}
    warning_rows = 0
    for row in rows:
        frame_id = row["pose_id"]
        passed = row.get("quality_passed", "").strip().lower() == "true"
        warning = row.get("quality_warnings", "").strip()
        if not passed or warning:
            warning_rows += 1
        frame = per_frame.setdefault(frame_id, {"quality_passed": True, "warnings": [], "roles": {}})
        frame["quality_passed"] = bool(frame["quality_passed"] and passed and not warning)
        if warning:
            frame["warnings"].append(f"{row.get('role', '')}:{warning}")
        frame["roles"][row.get("role", "")] = {
            "quality_passed": passed,
            "quality_warnings": warning,
            "laser_coverage": row.get("laser_coverage", ""),
            "dynamic_range_u8": row.get("dynamic_range_u8", ""),
            "mean_dn": row.get("mean_dn", ""),
        }
    return {
        "available": True,
        "rows_read": len(rows),
        "validation_rows_read": False,
        "warning_row_count": warning_rows,
        "per_frame": per_frame,
        "rows": rows,
    }


def load_legacy_fit_support(path: Path) -> dict[str, Any]:
    """Load only old FIT point rows to establish the pre-extension support."""
    values_v: list[float] = []
    values_s: list[float] = []
    frame_ids: set[str] = set()
    old_fit_ids = {f"{value:03d}" for value in range(1, 19)} | {f"{value:03d}" for value in range(25, 37)}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        v_key = "v_px" if "v_px" in fields else "v"
        s_key = "pca_s" if "pca_s" in fields else "s"
        if v_key not in fields or s_key not in fields:
            raise RuntimeError(f"Old FIT support CSV lacks v/s fields: {path}")
        for row in reader:
            if row.get("record_type") != "fit_point":
                continue
            frame_id = str(row.get("frame_id", "")).zfill(3)
            if frame_id not in old_fit_ids:
                raise RuntimeError(f"Unexpected/non-FIT frame in old support CSV: {frame_id}")
            frame_ids.add(frame_id)
            values_v.append(float(row[v_key]))
            values_s.append(float(row[s_key]))
    if not values_v or frame_ids != old_fit_ids:
        raise RuntimeError("Old FIT support CSV is incomplete")
    return {
        "source": str(path.resolve()),
        "sha256": sha256_file(path),
        "frame_count": len(frame_ids),
        "point_count": len(values_v),
        "v_px": [float(np.min(values_v)), float(np.max(values_v))],
        "s": [float(np.min(values_s)), float(np.max(values_s))],
    }


def array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "point_count": 0,
            "min": math.nan,
            "max": math.nan,
            "p01": math.nan,
            "p50": math.nan,
            "p99": math.nan,
        }
    p01, p50, p99 = np.percentile(values, [1.0, 50.0, 99.0])
    return {
        "point_count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
    }


def covers_range(values: np.ndarray, target: Sequence[float]) -> bool:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return bool(len(values) and np.min(values) <= float(target[0]) and np.max(values) >= float(target[1]))


def outside_distance(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.maximum(np.maximum(lower - values, values - upper), 0.0)


def support_row(
    *,
    record_type: str,
    edge: str,
    frame_ids: Sequence[str],
    v: np.ndarray,
    s: np.ndarray,
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    v = np.asarray(v, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    valid = np.isfinite(v) & np.isfinite(s)
    v = v[valid]
    s = s[valid]
    v_stats = array_stats(v)
    s_stats = array_stats(s)
    row: dict[str, Any] = {
        "record_type": record_type,
        "edge": edge,
        "frame_id": frame_ids[0] if len(frame_ids) == 1 else "",
        "frame_ids": "|".join(frame_ids),
        "pose_count": len(frame_ids),
        "point_count": int(len(v)),
        "v_min_px": v_stats["min"],
        "v_max_px": v_stats["max"],
        "v_p01_px": v_stats["p01"],
        "v_median_px": v_stats["p50"],
        "v_p99_px": v_stats["p99"],
        "s_min": s_stats["min"],
        "s_max": s_stats["max"],
        "s_p01": s_stats["p01"],
        "s_median": s_stats["p50"],
        "s_p99": s_stats["p99"],
        "legacy_v_min_px": legacy["v_px"][0],
        "legacy_v_max_px": legacy["v_px"][1],
        "legacy_s_min": legacy["s"][0],
        "legacy_s_max": legacy["s"][1],
    }
    if len(v) == 0 or edge not in EDGE_TARGETS:
        row.update(
            {
                "legacy_v_extrapolated_count": None,
                "legacy_s_extrapolated_count": None,
                "legacy_v_extrapolated_fraction": None,
                "legacy_s_extrapolated_fraction": None,
                "observed_v_covered": None,
                "observed_s_covered": None,
                "observed_vs_covered": None,
                "safe_v_covered": None,
                "safe_s_covered": None,
                "safe_vs_covered": None,
                "safe_v_lower_gap_px": None,
                "safe_v_upper_gap_px": None,
                "safe_s_lower_gap": None,
                "safe_s_upper_gap": None,
            }
        )
        return row
    target = EDGE_TARGETS[edge]
    legacy_v_out = (v < float(legacy["v_px"][0])) | (v > float(legacy["v_px"][1]))
    legacy_s_out = (s < float(legacy["s"][0])) | (s > float(legacy["s"][1]))
    observed_v_covered = covers_range(v, target["observed_v_px"])
    observed_s_covered = covers_range(s, target["observed_s"])
    safe_v_covered = covers_range(v, target["safe_v_px"])
    safe_s_covered = covers_range(s, target["safe_s"])
    row.update(
        {
            "legacy_v_extrapolated_count": int(np.count_nonzero(legacy_v_out)),
            "legacy_s_extrapolated_count": int(np.count_nonzero(legacy_s_out)),
            "legacy_v_extrapolated_fraction": float(np.mean(legacy_v_out)),
            "legacy_s_extrapolated_fraction": float(np.mean(legacy_s_out)),
            "observed_v_covered": observed_v_covered,
            "observed_s_covered": observed_s_covered,
            "observed_vs_covered": bool(observed_v_covered and observed_s_covered),
            "safe_v_covered": safe_v_covered,
            "safe_s_covered": safe_s_covered,
            "safe_vs_covered": bool(safe_v_covered and safe_s_covered),
            "safe_v_lower_gap_px": max(float(target["safe_v_px"][0]) - float(np.min(v)), 0.0),
            "safe_v_upper_gap_px": max(float(np.max(v)) - float(target["safe_v_px"][1]), 0.0),
            "safe_s_lower_gap": max(float(target["safe_s"][0]) - float(np.min(s)), 0.0),
            "safe_s_upper_gap": max(float(np.max(s)) - float(target["safe_s"][1]), 0.0),
        }
    )
    return row


def point_rows(
    frame_id: str,
    uv: np.ndarray,
    s: np.ndarray,
    legacy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    edge = EDGE_BY_FRAME[frame_id]
    target = EDGE_TARGETS[edge]
    rows: list[dict[str, Any]] = []
    for index, (pixel, coordinate) in enumerate(zip(np.asarray(uv), np.asarray(s))):
        u_value, v_value = float(pixel[0]), float(pixel[1])
        s_value = float(coordinate)
        legacy_v = float(legacy["v_px"][0]) <= v_value <= float(legacy["v_px"][1])
        legacy_s = float(legacy["s"][0]) <= s_value <= float(legacy["s"][1])
        safe_v = float(target["safe_v_px"][0]) <= v_value <= float(target["safe_v_px"][1])
        safe_s = float(target["safe_s"][0]) <= s_value <= float(target["safe_s"][1])
        rows.append(
            {
                "record_type": "point",
                "frame_id": frame_id,
                "edge": edge,
                "point_index": index,
                "u_px": u_value,
                "v_px": v_value,
                "s": s_value,
                "legacy_v_inside": legacy_v,
                "legacy_s_inside": legacy_s,
                "legacy_coverage": "interpolation" if legacy_v and legacy_s else "extrapolation",
                "safe_v_inside": safe_v,
                "safe_s_inside": safe_s,
                "safe_coverage": "interpolation" if safe_v and safe_s else "outside_target",
                "legacy_v_distance_px": float(outside_distance(np.asarray([v_value]), *legacy["v_px"])[0]),
                "legacy_s_distance": float(outside_distance(np.asarray([s_value]), *legacy["s"])[0]),
                "safe_v_distance_px": float(outside_distance(np.asarray([v_value]), *target["safe_v_px"])[0]),
                "safe_s_distance": float(outside_distance(np.asarray([s_value]), *target["safe_s"])[0]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_json(row.get(key, "")) for key in fields})


def make_plot(path: Path, points: Mapping[str, tuple[np.ndarray, np.ndarray]], legacy: Mapping[str, Any]) -> None:
    colors = {"Top": "#d95f02", "Bottom": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for edge, (s, v) in points.items():
        if len(s) > 6000:
            indices = np.linspace(0, len(s) - 1, 6000, dtype=int)
            s, v = s[indices], v[indices]
        ax.scatter(s, v, s=3, alpha=0.28, color=colors[edge], label=f"new FIT {edge}")
        target = EDGE_TARGETS[edge]
        ax.add_patch(
            Rectangle(
                (target["safe_s"][0], target["safe_v_px"][0]),
                target["safe_s"][1] - target["safe_s"][0],
                target["safe_v_px"][1] - target["safe_v_px"][0],
                fill=False,
                linewidth=2.0,
                edgecolor=colors[edge],
                linestyle="-",
                label=f"{edge} safe target",
            )
        )
    ax.add_patch(
        Rectangle(
            (legacy["s"][0], legacy["v_px"][0]),
            legacy["s"][1] - legacy["s"][0],
            legacy["v_px"][1] - legacy["v_px"][0],
            fill=False,
            linewidth=1.5,
            edgecolor="black",
            linestyle="--",
            label="legacy Frozen C1 FIT support",
        )
    )
    ax.invert_yaxis()
    ax.set_xlabel("Frozen PCA s")
    ax.set_ylabel("sensor v / px")
    ax.set_title("New FIT 049–054 support vs operational edge targets")
    ax.grid(True, alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_text(
    *,
    output_dir: Path,
    data_root: Path,
    c1_model_path: Path,
    c1_model_sha256: str,
    c1_model: Mapping[str, Any],
    legacy: Mapping[str, Any],
    frame_rows: Sequence[Mapping[str, Any]],
    edge_rows: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    quality: Mapping[str, Any],
    processing: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# New FIT 049–054 coverage audit",
        "",
        f"`NEW_FIT_SUPPORT_COVERAGE = {decision['NEW_FIT_SUPPORT_COVERAGE']}`",
        f"`QUALITY_GATE = {decision['QUALITY_GATE']}`",
        "",
        "## Scope",
        "",
        f"- 只打开：`{data_root / 'fit'}` 下的 FIT `049–054`，共 6 个 pose、18 张图；Top/Bottom 分组来自用户指令：049–051 = Top，052–054 = Bottom。",
        "- 本轮没有打开 `validation/055–060`，也没有读取旧 Validation 019–024、037–040；`validation_read = false`。",
        "- 使用 Frozen M0 K/D、Frozen Circular Cone 和既有 Frozen C1_4k PCA `s`；没有重新拟合 K/D、Cone、PCA 或 C1。",
        f"- Frozen C1：`{c1_model_path}`；artifact SHA-256 = `{c1_model_sha256}`；parameter SHA-256 = `{c1_model.get('parameter_sha256', '')}`。",
        "- 真实 operational domain 取上一轮 20 mm / 50 mm height-position union；本轮只复用其数值，不重新打开标准件图像。",
        "",
        "## Frozen and target domains",
        "",
        f"- 旧 FIT support：v = `[{fmt(legacy['v_px'][0])}, {fmt(legacy['v_px'][1])}] px`；s = `[{fmt(legacy['s'][0], 6)}, {fmt(legacy['s'][1], 6)}]`。",
        f"- Frozen C1 s domain：`[{fmt(c1_model['pca_s']['domain_min'], 9)}, {fmt(c1_model['pca_s']['domain_max'], 9)}]`。",
        "",
        "| edge | observed v | observed s | safe v (+margin) | safe s (+margin) | new FIT v | new FIT s | safe v/s covered |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for edge in ("Top", "Bottom"):
        target = EDGE_TARGETS[edge]
        row = edge_rows[edge]
        lines.append(
            f"| {edge} | [{fmt(target['observed_v_px'][0], 1)}, {fmt(target['observed_v_px'][1], 1)}] | "
            f"[{fmt(target['observed_s'][0], 6)}, {fmt(target['observed_s'][1], 6)}] | "
            f"[{fmt(target['safe_v_px'][0], 1)}, {fmt(target['safe_v_px'][1], 1)}] | "
            f"[{fmt(target['safe_s'][0], 3)}, {fmt(target['safe_s'][1], 3)}] | "
            f"[{fmt(row['v_min_px'], 1)}, {fmt(row['v_max_px'], 1)}] | "
            f"[{fmt(row['s_min'], 6)}, {fmt(row['s_max'], 6)}] | "
            f"{str(row['safe_vs_covered']).lower()} |"
        )
    lines.extend(["", "## Per-pose support", "", "| frame | edge | points | v range / px | s range | old support v extrapolation | old support s extrapolation |", "|---:|---|---:|---:|---:|---:|---:|"])
    for row in frame_rows:
        lines.append(
            f"| {row['frame_id']} | {row['edge']} | {row['point_count']} | [{fmt(row['v_min_px'], 1)}, {fmt(row['v_max_px'], 1)}] | "
            f"[{fmt(row['s_min'], 6)}, {fmt(row['s_max'], 6)}] | "
            f"{row['legacy_v_extrapolated_count']} ({100.0 * float(row['legacy_v_extrapolated_fraction']):.1f}%) | "
            f"{row['legacy_s_extrapolated_count']} ({100.0 * float(row['legacy_s_extrapolated_fraction']):.1f}%) |"
        )
    lines.extend(["", "## Acquisition quality", ""])
    if not quality.get("available"):
        lines.append("- `frames.csv` FIT quality metadata unavailable; geometry was audited from the requested FIT images only.")
    else:
        lines.append(
            f"- 仅读取 `frames.csv` 的前 18 个 FIT 数据行（049–054）；读取行数 = {quality['rows_read']}，Validation 行读取 = `false`。"
        )
        lines.append(
            f"- 质量警告行数：{quality['warning_row_count']} / {quality['rows_read']}；因此 `QUALITY_GATE = {decision['QUALITY_GATE']}`。"
        )
        for frame_id in NEW_FIT_IDS:
            item = quality["per_frame"][frame_id]
            text = "; ".join(item["warnings"]) if item["warnings"] else "none"
            lines.append(f"- frame {frame_id}: `{text}`")
    lines.extend(["", "## Processing sanity", "", "| frame | valid points | PnP RMSE / px | laser intensity mean / DN | stripe contrast mean / DN |", "|---:|---:|---:|---:|---:|"])
    for row in processing:
        lines.append(
            f"| {row['frame_id']} | {row['valid_point_count']} | {fmt(row['pnp_rmse_px'], 4)} | {fmt(row['laser_intensity_mean_dn'], 4)} | {fmt(row['stripe_contrast_mean_dn'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `NEW_FIT_SUPPORT_COVERAGE = {decision['NEW_FIT_SUPPORT_COVERAGE']}`：{decision['coverage_reason']}",
            f"- Top safe target covered = `{str(decision['Top']['safe_vs_covered']).lower()}`；Bottom safe target covered = `{str(decision['Bottom']['safe_vs_covered']).lower()}`。",
            "- 几何 support 若为 SUFFICIENT，表示新 FIT 的 v/s 点集范围已覆盖 operational domain 及安全余量；这不等于新图像已经通过激光信号质量验收。",
            f"- `QUALITY_GATE = {decision['QUALITY_GATE']}`：{decision['quality_reason']}",
            "- 下一步：只有在采集质量问题得到确认/修正后，才建议将 049–054 作为新的 FIT 输入重新冻结 C1；本轮没有执行该拟合。",
            "",
            "## Artifacts",
            "",
            f"- `new_fit_support_coverage.csv`: `{output_dir / 'new_fit_support_coverage.csv'}`",
            f"- `new_fit_support_points.csv`: `{output_dir / 'new_fit_support_points.csv'}`",
            f"- `new_fit_support_summary.json`: `{output_dir / 'new_fit_support_summary.json'}`",
            f"- `new_fit_support_coverage.png`: `{output_dir / 'new_fit_support_coverage.png'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    c1_model_path = args.c1_model.resolve()
    fit_support_path = args.fit_support.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    groups = inventory_new_fit(data_root)
    c1_model, c1_model_sha256 = frozen.load_frozen_json(c1_model_path)
    legacy = load_legacy_fit_support(fit_support_path)
    quality = read_fit_quality_metadata(data_root)

    # This loads only the frozen runtime model and the requested FIT images.
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, frozen_info = board.load_frozen_model_checked(
        args.frozen_provenance.resolve(), args.formal_cone.resolve()
    )
    processing_summaries, processed = board.process_groups_board(
        groups, intrinsics, calibration, reconstruction_params, frozen_model
    )
    if set(processed) != set(NEW_FIT_IDS):
        raise RuntimeError(f"Unexpected processed frame set: {sorted(processed)}")

    all_points: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    point_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for frame_id in NEW_FIT_IDS:
        uv = np.asarray(processed[frame_id]["uv"], dtype=np.float64)
        normalized = cv2.undistortPoints(
            uv.reshape(-1, 1, 2), intrinsics.camera_matrix, intrinsics.dist_coeffs
        ).reshape(-1, 2)
        s = base.pca_s_values(normalized, c1_model)
        edge = EDGE_BY_FRAME[frame_id]
        point_arrays.setdefault(edge, ([], []))
        point_arrays[edge][0].append(s)  # type: ignore[union-attr]
        point_arrays[edge][1].append(uv[:, 1])  # type: ignore[union-attr]
        frame_rows.append(
            support_row(
                record_type="frame_summary",
                edge=edge,
                frame_ids=[frame_id],
                v=uv[:, 1],
                s=s,
                legacy=legacy,
            )
        )
        all_points.extend(point_rows(frame_id, uv, s, legacy))

    edge_rows: dict[str, dict[str, Any]] = {}
    for edge, frame_ids in (("Top", TOP_IDS), ("Bottom", BOTTOM_IDS)):
        s_parts = [np.asarray(part, dtype=np.float64) for part in point_arrays[edge][0]]
        v_parts = [np.asarray(part, dtype=np.float64) for part in point_arrays[edge][1]]
        s_values = np.concatenate(s_parts) if s_parts else np.empty(0)
        v_values = np.concatenate(v_parts) if v_parts else np.empty(0)
        edge_rows[edge] = support_row(
            record_type="edge_union_summary",
            edge=edge,
            frame_ids=list(frame_ids),
            v=v_values,
            s=s_values,
            legacy=legacy,
        )

    all_s = np.concatenate([part for values in point_arrays.values() for part in values[0]])
    all_v = np.concatenate([part for values in point_arrays.values() for part in values[1]])
    all_row = support_row(
        record_type="new_fit_union_summary",
        edge="All",
        frame_ids=list(NEW_FIT_IDS),
        v=all_v,
        s=all_s,
        legacy=legacy,
    )
    summary_rows = [all_row] + frame_rows + list(edge_rows.values())

    safe_pass = all(bool(edge_rows[edge]["safe_vs_covered"]) for edge in ("Top", "Bottom"))
    observed_pass = all(bool(edge_rows[edge]["observed_vs_covered"]) for edge in ("Top", "Bottom"))
    if safe_pass:
        coverage = "SUFFICIENT"
        coverage_reason = "Top 和 Bottom 的新 FIT 三 pose union 均同时覆盖 observed operational domain 与推荐安全余量 domain。"
    elif observed_pass:
        coverage = "PARTIAL"
        coverage_reason = "Top 和 Bottom 均覆盖 observed operational domain，但至少一个 edge 未覆盖完整安全余量 domain。"
    else:
        coverage = "INSUFFICIENT"
        coverage_reason = "至少一个 edge 未覆盖 observed operational domain，或新 FIT 没有足够有效点。"
    quality_gate = "PASS" if quality.get("available") and quality.get("warning_row_count", 0) == 0 else "WARNING"
    quality_reason = (
        "18 个 FIT metadata 行全部通过 quality check。"
        if quality_gate == "PASS"
        else "新 FIT metadata 存在 dynamic_range_low / image_too_dark 警告；几何覆盖结果可用作 support audit，但不应视为采集质量通过。"
    )
    decision = {
        "NEW_FIT_SUPPORT_COVERAGE": coverage,
        "QUALITY_GATE": quality_gate,
        "coverage_reason": coverage_reason,
        "quality_reason": quality_reason,
        "Top": edge_rows["Top"],
        "Bottom": edge_rows["Bottom"],
        "validation_read": False,
        "c1_refit": False,
    }

    processing = []
    for item in processing_summaries:
        processing.append(
            {
                "frame_id": item["frame_id"],
                "valid_point_count": item["valid_point_count"],
                "used_point_count": item["used_point_count"],
                "pnp_rmse_px": item["pnp_rmse_px"],
                "laser_intensity_mean_dn": item["laser_intensity_mean_dn"],
                "stripe_contrast_mean_dn": item["stripe_contrast_mean_dn"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "new_fit_support_coverage.csv", summary_rows)
    write_csv(output_dir / "new_fit_support_points.csv", all_points)
    make_plot(output_dir / "new_fit_support_coverage.png", {
        edge: (np.concatenate(point_arrays[edge][0]), np.concatenate(point_arrays[edge][1]))
        for edge in ("Top", "Bottom")
    }, legacy)

    summary = {
        "schema_version": 1,
        "audit_id": "new_fit_support_coverage_049_054",
        "scope": {
            "data_root": str(data_root),
            "fit_ids": list(NEW_FIT_IDS),
            "top_ids": list(TOP_IDS),
            "bottom_ids": list(BOTTOM_IDS),
            "grouping_source": "user_instruction",
            "validation_read": False,
            "opened_directories": [str(data_root / "fit")],
        },
        "frozen_models": {
            "c1_model": str(c1_model_path),
            "c1_model_sha256": c1_model_sha256,
            "c1_parameter_sha256": c1_model.get("parameter_sha256"),
            "c1_pca_s": c1_model["pca_s"],
            "frozen_cone_provenance": frozen_info,
            "measurement_config": str(args.measurement_config.resolve()),
            "formal_cone": str(args.formal_cone.resolve()),
        },
        "legacy_fit_support": legacy,
        "operational_domain": {
            "source": "previous 20 mm / 50 mm height-position union and prior safety-margin plan",
            "standard_images_opened_this_run": False,
            "targets": EDGE_TARGETS,
        },
        "quality": quality,
        "processing": processing,
        "summary_rows": summary_rows,
        "decision": decision,
    }
    (output_dir / "new_fit_support_summary.json").write_text(
        json.dumps(clean_json(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            data_root=data_root,
            c1_model_path=c1_model_path,
            c1_model_sha256=c1_model_sha256,
            c1_model=c1_model,
            legacy=legacy,
            frame_rows=frame_rows,
            edge_rows=edge_rows,
            decision=decision,
            quality=quality,
            processing=processing,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "NEW_FIT_SUPPORT_COVERAGE": coverage,
                "QUALITY_GATE": quality_gate,
                "Top_safe_vs_covered": edge_rows["Top"]["safe_vs_covered"],
                "Bottom_safe_vs_covered": edge_rows["Bottom"]["safe_vs_covered"],
                "validation_read": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run(parse_args())
