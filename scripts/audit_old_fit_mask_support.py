#!/usr/bin/env python3
"""Compare old and full-physical-board masks on FIT 001--018, 025--036.

This is a FIT-only audit.  It opens the explicitly listed old FIT triplets,
uses the existing Steger/continuity path and frozen C0 validity filtering, and
never enumerates or opens Validation data.  Top/Bottom are evaluated as frozen
sensor-domain targets on the FIT union; old FIT frame IDs are not assigned to
an artificial Top/Bottom pose group.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_board_mask_exclusion_049_054 as mask_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import audit_mask_fix_coverage_049_054 as mask_fix  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


FIT_IDS = tuple(f"{index:03d}" for index in range(1, 19)) + tuple(f"{index:03d}" for index in range(25, 37))
ROLES = ("chess", "nolaser", "laser")
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE_MM = 20.0
CURRENT_MASK_MARGIN_PX = -2
LASER_ORIENTATION = "vertical"
JOINT_FULL_THRESHOLD = 0.95

DEFAULT_DATA_ROOT = ROOT / "projects" / "daheng" / "data" / "laser_plane"
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_OUTPUT_DIR = ROOT / "projects" / "daheng" / "outputs" / "0817" / "old_fit_mask_support_comparison"

TARGETS = {
    "Top": {
        "observed": {"v": [86.0, 467.0], "s": [-0.18500106019725174, -0.13286152907480264]},
        "safe": {"v": [30.0, 520.0], "s": [-0.191, -0.127]},
    },
    "Bottom": {
        "observed": {"v": [2810.0, 2938.0], "s": [0.1863643564578584, 0.2037112277937999]},
        "safe": {"v": [2760.0, 2990.0], "s": [0.181, 0.209]},
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=fixed.DEFAULT_FROZEN_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=fixed.DEFAULT_FORMAL_CONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def inventory_old_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve only FIT 001--018 and 025--036; never enumerate Validation."""
    groups: dict[str, dict[str, Any]] = {}
    for frame_id in FIT_IDS:
        root = data_root if int(frame_id) <= 18 else data_root / "fit_edge_extension"
        groups[frame_id] = {}
        for role in ROLES:
            path = root / "fit" / f"{role} {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            groups[frame_id][role] = {"path": path}
    return groups


def finite_range(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    return float(np.min(values)), float(np.max(values))


def clean(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def overlap_fraction(values: np.ndarray, target: Sequence[float]) -> float:
    lo, hi = finite_range(values)
    target_lo, target_hi = map(float, target)
    if not (math.isfinite(lo) and math.isfinite(hi)) or target_hi <= target_lo:
        return 0.0
    overlap = max(0.0, min(hi, target_hi) - max(lo, target_lo))
    return float(min(1.0, overlap / (target_hi - target_lo)))


def domain_stats(s: np.ndarray, v: np.ndarray, target: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Compute same-point joint support and conditional spans."""
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(v)
    s = s[finite]
    v = v[finite]
    v_target = np.asarray(target["v"], dtype=np.float64)
    s_target = np.asarray(target["s"], dtype=np.float64)

    joint = (v >= v_target[0]) & (v <= v_target[1]) & (s >= s_target[0]) & (s <= s_target[1])
    v_given_s = (s >= s_target[0]) & (s <= s_target[1])
    s_given_v = (v >= v_target[0]) & (v <= v_target[1])
    v_cond_min, v_cond_max = finite_range(v[v_given_s])
    s_cond_min, s_cond_max = finite_range(s[s_given_v])
    v_overlap = overlap_fraction(v[v_given_s], v_target)
    s_overlap = overlap_fraction(s[s_given_v], s_target)
    joint_full = bool(
        len(v[v_given_s])
        and len(s[s_given_v])
        and v_overlap >= JOINT_FULL_THRESHOLD
        and s_overlap >= JOINT_FULL_THRESHOLD
    )
    joint_present = bool(np.count_nonzero(joint))
    status = "FULL" if joint_full else ("PARTIAL" if joint_present else "NONE")
    joint_v_min, joint_v_max = finite_range(v[joint])
    joint_s_min, joint_s_max = finite_range(s[joint])
    return {
        "target_v_min": float(v_target[0]),
        "target_v_max": float(v_target[1]),
        "target_s_min": float(s_target[0]),
        "target_s_max": float(s_target[1]),
        "joint_count": int(np.count_nonzero(joint)),
        "joint_fraction": float(np.mean(joint)) if len(joint) else math.nan,
        "joint_v_min": joint_v_min,
        "joint_v_max": joint_v_max,
        "joint_s_min": joint_s_min,
        "joint_s_max": joint_s_max,
        "v_given_s_count": int(np.count_nonzero(v_given_s)),
        "v_given_s_min": v_cond_min,
        "v_given_s_max": v_cond_max,
        "v_given_s_overlap_fraction": v_overlap,
        "s_given_v_count": int(np.count_nonzero(s_given_v)),
        "s_given_v_min": s_cond_min,
        "s_given_v_max": s_cond_max,
        "s_given_v_overlap_fraction": s_overlap,
        "joint_present": joint_present,
        "joint_full": joint_full,
        "status": status,
    }


def basic_row(prefix: str, result: Mapping[str, Any]) -> dict[str, Any]:
    v_min, v_max = finite_range(np.asarray(result["v"], dtype=np.float64))
    s_min, s_max = finite_range(np.asarray(result["s"], dtype=np.float64))
    return {
        f"{prefix}_steger_count": int(result["steger_count"]),
        f"{prefix}_effective_count": int(result["effective_count"]),
        f"{prefix}_v_min_px": v_min,
        f"{prefix}_v_max_px": v_max,
        f"{prefix}_s_min": s_min,
        f"{prefix}_s_max": s_max,
    }


def add_domain_rows(row: dict[str, Any], prefix: str, s: np.ndarray, v: np.ndarray) -> None:
    for edge in ("Top", "Bottom"):
        for domain in ("observed", "safe"):
            stats = domain_stats(s, v, TARGETS[edge][domain])
            short = f"{prefix}_{edge.lower()}_{domain}"
            for key, value in stats.items():
                row[f"{short}_{key}"] = value


def recovery_counts(row: dict[str, Any], s: np.ndarray, v: np.ndarray) -> None:
    for edge in ("Top", "Bottom"):
        for domain in ("observed", "safe"):
            stats = domain_stats(s, v, TARGETS[edge][domain])
            row[f"recovered_{edge.lower()}_{domain}_count"] = stats["joint_count"]


def summarize_record(
    *,
    record_type: str,
    frame_ids: Sequence[str],
    old_result: Mapping[str, Any],
    new_result: Mapping[str, Any],
    recovered_s: np.ndarray,
    recovered_v: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": record_type,
        "frame_ids": "|".join(frame_ids),
        "pose_count": len(frame_ids),
    }
    if len(frame_ids) == 1:
        row["frame_id"] = frame_ids[0]
    else:
        row["frame_id"] = ""
    row.update(basic_row("old", old_result))
    row.update(basic_row("new", new_result))
    row["new_effective_outside_old_mask_count"] = int(len(recovered_s))
    add_domain_rows(row, "old", np.asarray(old_result["s"]), np.asarray(old_result["v"]))
    add_domain_rows(row, "new", np.asarray(new_result["s"]), np.asarray(new_result["v"]))
    recovery_counts(row, recovered_s, recovered_v)
    return row


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
            writer.writerow({key: clean(row.get(key, "")) for key in fields})


def decision_from_union(union_rows: Mapping[str, Mapping[str, Any]], prefix: str) -> str:
    safe_full = all(union_rows[edge][f"{prefix}_{edge.lower()}_safe_joint_full"] for edge in ("Top", "Bottom"))
    observed_present = all(union_rows[edge][f"{prefix}_{edge.lower()}_observed_joint_count"] > 0 for edge in ("Top", "Bottom"))
    safe_present = all(union_rows[edge][f"{prefix}_{edge.lower()}_safe_joint_count"] > 0 for edge in ("Top", "Bottom"))
    if safe_full:
        return "SUFFICIENT"
    if observed_present and safe_present:
        return "PARTIAL"
    return "INSUFFICIENT"


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def make_plot(path: Path, points: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    colors = {"Top": "#d95f02", "Bottom": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    for axis, variant in zip(axes, ("old", "new")):
        s = points[variant]["s"]
        v = points[variant]["v"]
        axis.scatter(s, v, s=2, alpha=0.16, color="#555555", label="all FIT effective points")
        for edge in ("Top", "Bottom"):
            for domain, linestyle, linewidth in (("observed", "--", 1.5), ("safe", "-", 2.0)):
                target = TARGETS[edge][domain]
                axis.add_patch(
                    Rectangle(
                        (target["s"][0], target["v"][0]),
                        target["s"][1] - target["s"][0],
                        target["v"][1] - target["v"][0],
                        fill=False,
                        edgecolor=colors[edge],
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=f"{edge} {domain}",
                    )
                )
        axis.set_title("old: inner-corner hull + margin -2" if variant == "old" else "new: full physical board, 0 mm inset")
        axis.set_xlabel("Frozen PCA s")
        axis.grid(True, alpha=0.2)
        axis.invert_yaxis()
    axes[0].set_ylabel("sensor v / px")
    handles, labels = axes[1].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axes[1].legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    fig.suptitle("Old FIT 001–018, 025–036: mask support comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_text(
    *,
    output_dir: Path,
    data_root: Path,
    c1_path: Path,
    c1_sha256: str,
    union_rows: Mapping[str, Mapping[str, Any]],
    old_decision: str,
    new_decision: str,
    recovered_union: Mapping[str, int],
) -> str:
    lines = [
        "# Old FIT mask support comparison",
        "",
        f"`OLD_FIT_EDGE_SUPPORT_AFTER_MASK_FIX = {new_decision}`",
        "",
        "## Scope and frozen processing",
        "",
        f"- 只打开 `{data_root}` 下 FIT `001–018` 和 `fit_edge_extension/fit` 下 FIT `025–036` 的 90 张图；没有枚举或打开 Validation。",
        "- Old mask：inner-corner convex hull + `margin_px=-2`。",
        "- New mask：PnP 投影完整棋盘物理边界 `X=[-20,220] mm`、`Y=[-20,160] mm`，0 mm inset、无额外腐蚀。",
        "- 两种 mask 均使用相同 Steger、`vertical` 每 row 单点、continuity 和 900 点上限；有效点使用相同 Frozen Circular Cone validity filter。",
        f"- Frozen C1 artifact：`{c1_path}`；SHA-256 = `{c1_sha256}`。C1 仅用于固定 PCA `s`，没有重新拟合。",
        "- 旧 FIT 没有可靠的 frame-to-Top/Bottom 映射；Top/Bottom 作为冻结 sensor-domain target，在全部 FIT union 和每帧上按同点 `v/s` 联合条件统计。",
        "",
        "## Frozen target domains",
        "",
        "| edge | observed v | observed s | safety v | safety s |",
        "|---|---:|---:|---:|---:|",
        "| Top | [86, 467] | [-0.185001, -0.132862] | [30, 520] | [-0.191, -0.127] |",
        "| Bottom | [2810, 2938] | [0.186364, 0.203712] | [2760, 2990] | [0.181, 0.209] |",
        "",
        "`joint` 表示同一个有效点同时落入目标 v 和目标 s 区间；FULL 使用同条件集合的 v/s conditional span overlap ≥ 95%，避免用互不对应的独立 min/max 误判。",
        "",
        "## All FIT union",
        "",
        "| target edge | old effective | new effective | old v range | new v range | old s range | new s range | old observed joint/status | new observed joint/status | old safety joint/status | new safety joint/status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for edge in ("Top", "Bottom"):
        row = union_rows[edge]
        lines.append(
            f"| {edge} | {row['old_effective_count']} | {row['new_effective_count']} | "
            f"[{fmt(row['old_v_min_px'],1)}, {fmt(row['old_v_max_px'],1)}] | [{fmt(row['new_v_min_px'],1)}, {fmt(row['new_v_max_px'],1)}] | "
            f"[{fmt(row['old_s_min'],6)}, {fmt(row['old_s_max'],6)}] | [{fmt(row['new_s_min'],6)}, {fmt(row['new_s_max'],6)}] | "
            f"{row['old_top_observed_joint_count'] if edge == 'Top' else row['old_bottom_observed_joint_count']} / {row[f'old_{edge.lower()}_observed_status']} | "
            f"{row['new_top_observed_joint_count'] if edge == 'Top' else row['new_bottom_observed_joint_count']} / {row[f'new_{edge.lower()}_observed_status']} | "
            f"{row[f'old_{edge.lower()}_safe_joint_count']} / {row[f'old_{edge.lower()}_safe_status']} | "
            f"{row[f'new_{edge.lower()}_safe_joint_count']} / {row[f'new_{edge.lower()}_safe_status']} |"
        )
    lines.extend(
        [
            "",
            "## Points recoverable by replacing the old mask",
            "",
            "恢复点定义为：new-mask effective point 的像素坐标落在 old mask 外；这比单纯比较 old/new 总点数更直接地隔离 mask 排除效应。",
            "",
            "| target edge | observed recovered points | safety recovered points |",
            "|---|---:|---:|",
            f"| Top | {recovered_union['top_observed']} | {recovered_union['top_safe']} |",
            f"| Bottom | {recovered_union['bottom_observed']} | {recovered_union['bottom_safe']} |",
            "",
            "## Interpretation",
            "",
            f"- Old mask union decision = `{old_decision}`；new full-board mask union decision = `{new_decision}`。",
            f"- `OLD_FIT_EDGE_SUPPORT_AFTER_MASK_FIX = {new_decision}`。",
        ]
    )
    if old_decision == "INSUFFICIENT" and new_decision in {"PARTIAL", "SUFFICIENT"}:
        lines.append("- 结论：此前 support 不足至少部分由 inner-hull + margin mask 造成；新 mask 恢复了边缘 support，但最终是否达到 safety 全覆盖由上表决定。")
    elif old_decision == new_decision == "PARTIAL":
        lines.append("- 结论：旧 mask 确实排除了部分边缘点（见 recovered counts），新 mask 将 Top 提升到 FULL、Bottom 提升到 observed FULL，但 Bottom safety 仍为 PARTIAL；因此旧 mask 是部分原因，不是唯一原因。")
    elif old_decision == new_decision:
        lines.append("- 结论：更换 mask 没有改变 support 等级；此前不足不能主要归因于旧 mask。")
    else:
        lines.append("- 结论：mask 更换改变了 support，但仍需结合联合覆盖结果判断是否已达到完整边缘域。")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `old_fit_mask_support_comparison.csv`: `{output_dir / 'old_fit_mask_support_comparison.csv'}`",
            f"- `old_fit_support_coverage.png`: `{output_dir / 'old_fit_support_coverage.png'}`",
            f"- `report.md`: `{output_dir / 'report.md'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    c1_model_path = args.c1_model.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    groups = inventory_old_fit(data_root)
    c1_model, c1_sha256 = frozen.load_frozen_json(c1_model_path)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, _ = board.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    frozen_calibration = copy.deepcopy(dict(calibration))
    frozen_calibration["laser_model"] = copy.deepcopy(dict(frozen_model))

    all_points = {
        "old": {"s": [], "v": []},
        "new": {"s": [], "v": []},
        "recovered": {"s": [], "v": []},
    }
    frame_rows: list[dict[str, Any]] = []
    old_steger_total = 0
    new_steger_total = 0

    for frame_id in FIT_IDS:
        chess = fixed.triplets.imread_unicode(Path(groups[frame_id]["chess"]["path"]))
        background = fixed.triplets.imread_unicode(Path(groups[frame_id]["nolaser"]["path"]))
        laser = fixed.triplets.imread_unicode(Path(groups[frame_id]["laser"]["path"]))
        pose = fixed.triplets.detect_board_pose(
            chess,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            cols=BOARD_COLS,
            rows=BOARD_ROWS,
            square_size_mm=SQUARE_SIZE_MM,
            max_rmse_px=0.40,
        )
        shape = fixed.triplets.to_gray_float(chess).shape
        old_mask = fixed.triplets.board_inner_mask(shape, pose.corners, margin_px=CURRENT_MASK_MARGIN_PX)
        new_polygon = mask_audit.projected_outer_boundary(pose, intrinsics)
        new_mask = mask_audit.polygon_mask(shape, new_polygon)
        common = {
            "laser": laser,
            "background": background,
            "cfg": fixed.EXTRACTION_CONFIG,
            "pose": pose,
            "intrinsics": intrinsics,
            "frozen_calibration": frozen_calibration,
            "reconstruction_params": reconstruction_params,
            "c1_model": c1_model,
        }
        old_result = mask_fix.extract_effective(mask=old_mask, **common)
        new_result = mask_fix.extract_effective(mask=new_mask, **common)
        old_steger_total += int(old_result["steger_count"])
        new_steger_total += int(new_result["steger_count"])

        new_uv = np.asarray(new_result["uv_effective"], dtype=np.float64)
        inside_old = mask_audit.points_in_mask(new_uv[:, 0], new_uv[:, 1], old_mask)
        recovered = ~inside_old
        recovered_s = np.asarray(new_result["s"], dtype=np.float64)[recovered]
        recovered_v = np.asarray(new_result["v"], dtype=np.float64)[recovered]

        for variant, result in (("old", old_result), ("new", new_result)):
            all_points[variant]["s"].append(np.asarray(result["s"], dtype=np.float64))
            all_points[variant]["v"].append(np.asarray(result["v"], dtype=np.float64))
        all_points["recovered"]["s"].append(recovered_s)
        all_points["recovered"]["v"].append(recovered_v)

        frame_rows.append(
            summarize_record(
                record_type="frame_summary",
                frame_ids=[frame_id],
                old_result=old_result,
                new_result=new_result,
                recovered_s=recovered_s,
                recovered_v=recovered_v,
            )
        )

    def concatenate(variant: str) -> tuple[np.ndarray, np.ndarray]:
        s = np.concatenate(all_points[variant]["s"]) if all_points[variant]["s"] else np.empty(0)
        v = np.concatenate(all_points[variant]["v"]) if all_points[variant]["v"] else np.empty(0)
        return s, v

    old_s, old_v = concatenate("old")
    new_s, new_v = concatenate("new")
    recovered_s, recovered_v = concatenate("recovered")
    old_union_result = {"steger_count": old_steger_total, "effective_count": len(old_s), "s": old_s, "v": old_v}
    new_union_result = {"steger_count": new_steger_total, "effective_count": len(new_s), "s": new_s, "v": new_v}

    union_rows: dict[str, dict[str, Any]] = {}
    csv_rows = list(frame_rows)
    for edge in ("Top", "Bottom"):
        row = summarize_record(
            record_type="fit_union",
            frame_ids=FIT_IDS,
            old_result=old_union_result,
            new_result=new_union_result,
            recovered_s=recovered_s,
            recovered_v=recovered_v,
        )
        union_rows[edge] = row
        csv_rows.append(row)

    # The same union row is repeated for each target edge so the CSV has one
    # directly addressable record per Top/Bottom audit target.
    for row, edge in zip(csv_rows[-2:], ("Top", "Bottom")):
        row["target_edge"] = edge

    old_decision = decision_from_union(union_rows, "old")
    new_decision = decision_from_union(union_rows, "new")
    recovered_union = {
        "top_observed": int(domain_stats(recovered_s, recovered_v, TARGETS["Top"]["observed"])["joint_count"]),
        "top_safe": int(domain_stats(recovered_s, recovered_v, TARGETS["Top"]["safe"])["joint_count"]),
        "bottom_observed": int(domain_stats(recovered_s, recovered_v, TARGETS["Bottom"]["observed"])["joint_count"]),
        "bottom_safe": int(domain_stats(recovered_s, recovered_v, TARGETS["Bottom"]["safe"])["joint_count"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "old_fit_mask_support_comparison.csv", csv_rows)
    make_plot(
        output_dir / "old_fit_support_coverage.png",
        {
            "old": {"s": old_s, "v": old_v},
            "new": {"s": new_s, "v": new_v},
        },
    )
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            data_root=data_root,
            c1_path=c1_model_path,
            c1_sha256=c1_sha256,
            union_rows=union_rows,
            old_decision=old_decision,
            new_decision=new_decision,
            recovered_union=recovered_union,
        ),
        encoding="utf-8",
    )
    print(
        {
            "output_dir": str(output_dir),
            "old_union_decision": old_decision,
            "OLD_FIT_EDGE_SUPPORT_AFTER_MASK_FIX": new_decision,
            "old_effective_count": len(old_s),
            "new_effective_count": len(new_s),
            "recovered_effective_count": len(recovered_s),
            "validation_read": False,
        }
    )


if __name__ == "__main__":
    run(parse_args())
