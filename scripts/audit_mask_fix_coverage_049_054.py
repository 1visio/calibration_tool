#!/usr/bin/env python3
"""Compare the current and full physical-board masks for FIT 049--054.

The ``after`` variant uses the PnP projection of X=[-20,220] mm and
Y=[-20,160] mm, with no erosion.  It is an audit-only mask variant: the
existing production helper is not changed.  Steger, continuity, Frozen C0
validity and Frozen C1 PCA are kept fixed.  Only FIT 049--054 is opened.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_board_mask_exclusion_049_054 as mask_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


NEW_FIT_IDS = tuple(f"{value:03d}" for value in range(49, 55))
TOP_IDS = ("049", "050", "051")
BOTTOM_IDS = ("052", "053", "054")
EDGE_BY_FRAME = {frame_id: "Top" for frame_id in TOP_IDS} | {
    frame_id: "Bottom" for frame_id in BOTTOM_IDS
}
MASK_VARIANTS = ("before", "after")
DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane_0817"
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "mask_fix_coverage_049_054"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--c1-model", type=Path, default=DEFAULT_C1_MODEL)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--frozen-provenance", type=Path, default=base.DEFAULT_PROVENANCE)
    parser.add_argument("--formal-cone", type=Path, default=base.DEFAULT_FORMAL_CONE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "min": math.nan, "max": math.nan, "median": math.nan}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def covers_range(values: np.ndarray, target: Sequence[float]) -> bool:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return bool(len(values) and np.min(values) <= float(target[0]) and np.max(values) >= float(target[1]))


def extract_effective(
    *,
    laser: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    cfg: Mapping[str, Any],
    pose: Any,
    intrinsics: Any,
    frozen_calibration: Mapping[str, Any],
    reconstruction_params: Any,
    c1_model: Mapping[str, Any],
) -> dict[str, Any]:
    u, v, _, _ = fixed.triplets.extract_laser_centers(laser, background, mask, cfg, "vertical")
    uv = np.column_stack([u, v]).astype(np.float64)
    truth = fixed.coverage.plane_ray_truth(
        u, v, pose.normal, pose.d, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    lambda_model, model_valid = fixed.lambda_by_input(uv, frozen_calibration, reconstruction_params)
    effective = (
        np.asarray(model_valid, dtype=bool)
        & np.asarray(truth["valid"], dtype=bool)
        & np.isfinite(np.asarray(truth["points"], dtype=np.float64)[:, 2])
    )
    uv_effective = uv[effective]
    normalized = cv2.undistortPoints(
        uv_effective.reshape(-1, 1, 2), intrinsics.camera_matrix, intrinsics.dist_coeffs
    ).reshape(-1, 2)
    s = base.pca_s_values(normalized, c1_model)
    return {
        "steger_count": int(len(uv)),
        "effective_count": int(np.count_nonzero(effective)),
        "effective_fraction": float(np.mean(effective)) if len(effective) else math.nan,
        "uv_effective": uv_effective,
        "v": uv_effective[:, 1],
        "s": s,
        "lambda_model": np.asarray(lambda_model, dtype=np.float64)[effective],
    }


def summarize_variant(
    *,
    frame_id: str,
    edge: str,
    variant: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    v_stats = array_stats(np.asarray(result["v"], dtype=np.float64))
    s_stats = array_stats(np.asarray(result["s"], dtype=np.float64))
    target = base_targets()[edge]
    return {
        "frame_id": frame_id,
        "edge": edge,
        "mask_variant": variant,
        "steger_count": result["steger_count"],
        "effective_count": result["effective_count"],
        "effective_fraction": result["effective_fraction"],
        "v_min_px": v_stats["min"],
        "v_max_px": v_stats["max"],
        "v_median_px": v_stats["median"],
        "s_min": s_stats["min"],
        "s_max": s_stats["max"],
        "s_median": s_stats["median"],
        "observed_v_covered": covers_range(result["v"], target["observed_v_px"]),
        "observed_s_covered": covers_range(result["s"], target["observed_s"]),
        "observed_vs_covered": bool(
            covers_range(result["v"], target["observed_v_px"])
            and covers_range(result["s"], target["observed_s"])
        ),
        "safe_v_covered": covers_range(result["v"], target["safe_v_px"]),
        "safe_s_covered": covers_range(result["s"], target["safe_s"]),
        "safe_vs_covered": bool(
            covers_range(result["v"], target["safe_v_px"])
            and covers_range(result["s"], target["safe_s"])
        ),
    }


def base_targets() -> dict[str, dict[str, list[float]]]:
    # Reuse the established height-position operational domains and safety
    # margins from the previous coverage audit without reading standard images.
    return {
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


def comparison_row(
    *,
    record_type: str,
    edge: str,
    frame_ids: Sequence[str],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": record_type,
        "edge": edge,
        "frame_id": frame_ids[0] if len(frame_ids) == 1 else "",
        "frame_ids": "|".join(frame_ids),
        "pose_count": len(frame_ids),
    }
    for variant, values in (("before", before), ("after", after)):
        for key in (
            "steger_count",
            "effective_count",
            "effective_fraction",
            "v_min_px",
            "v_max_px",
            "v_median_px",
            "s_min",
            "s_max",
            "s_median",
            "observed_v_covered",
            "observed_s_covered",
            "observed_vs_covered",
            "safe_v_covered",
            "safe_s_covered",
            "safe_vs_covered",
        ):
            row[f"{variant}_{key}"] = values.get(key)
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
            writer.writerow({key: clean_value(row.get(key, "")) for key in fields})


def make_plot(
    path: Path,
    points: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    colors = {"Top": "#d95f02", "Bottom": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    targets = base_targets()
    for axis, variant in zip(axes, ("before", "after")):
        for edge in ("Top", "Bottom"):
            s, v = points[variant][edge]
            axis.scatter(s, v, s=3, alpha=0.25, color=colors[edge], label=f"{edge} effective points")
            target = targets[edge]
            axis.add_patch(
                Rectangle(
                    (target["safe_s"][0], target["safe_v_px"][0]),
                    target["safe_s"][1] - target["safe_s"][0],
                    target["safe_v_px"][1] - target["safe_v_px"][0],
                    fill=False,
                    linewidth=2.0,
                    edgecolor=colors[edge],
                    label=f"{edge} safe target",
                )
            )
        axis.set_title("before: inner hull + margin -2" if variant == "before" else "after: full physical board, 0 mm inset")
        axis.set_xlabel("Frozen PCA s")
        axis.grid(True, alpha=0.2)
        axis.invert_yaxis()
    axes[0].set_ylabel("sensor v / px")
    handles, labels = axes[1].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axes[1].legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    fig.suptitle("FIT 049–054 effective support: mask before vs after")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def report_text(
    *,
    output_dir: Path,
    data_root: Path,
    c1_model_path: Path,
    c1_model_sha256: str,
    edge_rows: Mapping[str, Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    decision: str,
) -> str:
    lines = [
        "# Laser-plane mask fix coverage audit for FIT 049–054",
        "",
        f"`EDGE_SUPPORT_AFTER_MASK_FIX = {decision}`",
        "",
        "## Scope and mask definitions",
        "",
        f"- 只打开 `{data_root / 'fit'}` 下 FIT `049–054` 的 18 张图；没有打开 `validation/055–060` 或旧 Validation。",
        "- Before：当前 `board_inner_mask(inner-corner hull, margin_px=-2)`。",
        "- After：PnP 投影的完整 12×9 方格物理边界 `X=[-20,220] mm`、`Y=[-20,160] mm`，不做任何像素腐蚀。该 polygon 不扩展到棋盘印刷区之外的白边或铝框。",
        "- 两种 mask 均使用相同 Steger、`vertical` 每 row 单点、continuity、900 点上限，并使用相同 Frozen Circular Cone 有效性筛选；`s` 使用 Frozen C1 PCA 定义。",
        "- 未重新拟合或修改 K/D、Cone、C1；C1 仅用于固定 PCA `s` 坐标和 frozen domain 对照。",
        f"- Frozen C1 artifact：`{c1_model_path}`；SHA-256 = `{c1_model_sha256}`。",
        "",
        "安全目标域（沿用上一轮 coverage plan，仅用于覆盖判定）：",
        "",
        "| edge | safe v range (px) | safe s range |",
        "|---|---:|---:|",
        "| Top | [30, 520] | [-0.191, -0.127] |",
        "| Bottom | [2760, 2990] | [0.181, 0.209] |",
        "",
        "## Top/Bottom effective support",
        "",
        "| edge | before effective / Steger | after effective / Steger | before v range | after v range | before s range | after s range | after safe v/s |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for edge in ("Top", "Bottom"):
        row = edge_rows[edge]
        lines.append(
            f"| {edge} | {row['before_effective_count']} / {row['before_steger_count']} | "
            f"{row['after_effective_count']} / {row['after_steger_count']} | "
            f"[{fmt(row['before_v_min_px'], 1)}, {fmt(row['before_v_max_px'], 1)}] | "
            f"[{fmt(row['after_v_min_px'], 1)}, {fmt(row['after_v_max_px'], 1)}] | "
            f"[{fmt(row['before_s_min'], 6)}, {fmt(row['before_s_max'], 6)}] | "
            f"[{fmt(row['after_s_min'], 6)}, {fmt(row['after_s_max'], 6)}] | "
            f"{str(row['after_safe_vs_covered']).lower()} |"
        )
    lines.extend(["", "## Per-pose effective counts", "", "| frame | edge | before effective | after effective | delta | before safe v/s | after safe v/s |", "|---:|---|---:|---:|---:|---|---|"])
    for row in frame_rows:
        lines.append(
            f"| {row['frame_id']} | {row['edge']} | {row['before_effective_count']} | {row['after_effective_count']} | "
            f"{int(row['after_effective_count']) - int(row['before_effective_count'])} | "
            f"{str(row['before_safe_vs_covered']).lower()} | {str(row['after_safe_vs_covered']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `EDGE_SUPPORT_AFTER_MASK_FIX = {decision}`。判定以 Top/Bottom edge union 的 effective points 同时覆盖 v 和 s safe target 为准。",
            f"- After Top safe target covered = `{str(edge_rows['Top']['after_safe_vs_covered']).lower()}`；After Bottom safe target covered = `{str(edge_rows['Bottom']['after_safe_vs_covered']).lower()}`。",
            "- 若 after 仍为 INSUFFICIENT，说明仅扩大棋盘有效 mask 不能补足真实 operational Top/Bottom 位置；问题还包括采集 pose/domain 覆盖不足。",
            "",
            "## Artifacts",
            "",
            f"- `mask_before_after.csv`: `{output_dir / 'mask_before_after.csv'}`",
            f"- `new_fit_support_coverage.png`: `{output_dir / 'new_fit_support_coverage.png'}`",
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

    groups = mask_audit.inventory_new_fit(data_root)
    c1_model, c1_model_sha256 = frozen.load_frozen_json(c1_model_path)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, _ = board.load_frozen_model_checked(
        args.frozen_provenance.resolve(), args.formal_cone.resolve()
    )
    frozen_calibration = copy.deepcopy(dict(calibration))
    frozen_calibration["laser_model"] = copy.deepcopy(dict(frozen_model))

    frame_results: dict[str, dict[str, dict[str, Any]]] = {}
    point_arrays: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "before": {"Top": (np.empty(0), np.empty(0)), "Bottom": (np.empty(0), np.empty(0))},
        "after": {"Top": (np.empty(0), np.empty(0)), "Bottom": (np.empty(0), np.empty(0))},
    }
    frame_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for frame_id in NEW_FIT_IDS:
        chess = fixed.triplets.imread_unicode(Path(groups[frame_id]["chess"]["path"]))
        background = fixed.triplets.imread_unicode(Path(groups[frame_id]["nolaser"]["path"]))
        laser = fixed.triplets.imread_unicode(Path(groups[frame_id]["laser"]["path"]))
        pose = fixed.triplets.detect_board_pose(
            chess,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            cols=11,
            rows=8,
            square_size_mm=20.0,
            max_rmse_px=0.40,
        )
        shape = fixed.triplets.to_gray_float(chess).shape
        current_mask = fixed.triplets.board_inner_mask(shape, pose.corners, margin_px=-2)
        outer_polygon = mask_audit.projected_outer_boundary(pose, intrinsics)
        full_mask = mask_audit.polygon_mask(shape, outer_polygon)
        masks = {"before": current_mask, "after": full_mask}
        frame_results[frame_id] = {}
        for variant in MASK_VARIANTS:
            result = extract_effective(
                laser=laser,
                background=background,
                mask=masks[variant],
                cfg=fixed.EXTRACTION_CONFIG,
                pose=pose,
                intrinsics=intrinsics,
                frozen_calibration=frozen_calibration,
                reconstruction_params=reconstruction_params,
                c1_model=c1_model,
            )
            frame_results[frame_id][variant] = result
            edge = EDGE_BY_FRAME[frame_id]
            existing_s, existing_v = point_arrays[variant][edge]
            point_arrays[variant][edge] = (
                np.concatenate([existing_s, np.asarray(result["s"], dtype=np.float64)]),
                np.concatenate([existing_v, np.asarray(result["v"], dtype=np.float64)]),
            )
        before_summary = summarize_variant(frame_id=frame_id, edge=EDGE_BY_FRAME[frame_id], variant="before", result=frame_results[frame_id]["before"])
        after_summary = summarize_variant(frame_id=frame_id, edge=EDGE_BY_FRAME[frame_id], variant="after", result=frame_results[frame_id]["after"])
        frame_rows.append(
            {
                "frame_id": frame_id,
                "edge": EDGE_BY_FRAME[frame_id],
                **{f"before_{key}": value for key, value in before_summary.items() if key not in {"frame_id", "edge", "mask_variant"}},
                **{f"after_{key}": value for key, value in after_summary.items() if key not in {"frame_id", "edge", "mask_variant"}},
            }
        )

    comparison_rows.extend(
        [
            comparison_row(
                record_type="frame_summary",
                edge=row["edge"],
                frame_ids=[row["frame_id"]],
                before={key.removeprefix("before_"): value for key, value in row.items() if key.startswith("before_")},
                after={key.removeprefix("after_"): value for key, value in row.items() if key.startswith("after_")},
            )
            for row in frame_rows
        ]
    )
    edge_rows: dict[str, dict[str, Any]] = {}
    for edge, frame_ids in (("Top", TOP_IDS), ("Bottom", BOTTOM_IDS)):
        edge_before_result = {
            "steger_count": sum(int(frame_results[frame_id]["before"]["steger_count"]) for frame_id in frame_ids),
            "effective_count": int(len(point_arrays["before"][edge][0])),
            "effective_fraction": math.nan,
            "v": point_arrays["before"][edge][1],
            "s": point_arrays["before"][edge][0],
        }
        edge_after_result = {
            "steger_count": sum(int(frame_results[frame_id]["after"]["steger_count"]) for frame_id in frame_ids),
            "effective_count": int(len(point_arrays["after"][edge][0])),
            "effective_fraction": math.nan,
            "v": point_arrays["after"][edge][1],
            "s": point_arrays["after"][edge][0],
        }
        before_summary = summarize_variant(frame_id="", edge=edge, variant="before", result=edge_before_result)
        after_summary = summarize_variant(frame_id="", edge=edge, variant="after", result=edge_after_result)
        edge_rows[edge] = {
            **{f"before_{key}": value for key, value in before_summary.items() if key not in {"frame_id", "edge", "mask_variant"}},
            **{f"after_{key}": value for key, value in after_summary.items() if key not in {"frame_id", "edge", "mask_variant"}},
        }
        comparison_rows.append(
            comparison_row(
                record_type="edge_union_summary",
                edge=edge,
                frame_ids=list(frame_ids),
                before=before_summary,
                after=after_summary,
            )
        )

    observed_pass = all(bool(edge_rows[edge]["after_observed_vs_covered"]) for edge in ("Top", "Bottom"))
    safe_pass = all(bool(edge_rows[edge]["after_safe_vs_covered"]) for edge in ("Top", "Bottom"))
    if safe_pass:
        decision = "SUFFICIENT"
    elif observed_pass:
        decision = "PARTIAL"
    else:
        decision = "INSUFFICIENT"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "mask_before_after.csv", comparison_rows)
    make_plot(output_dir / "new_fit_support_coverage.png", point_arrays)
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            data_root=data_root,
            c1_model_path=c1_model_path,
            c1_model_sha256=c1_model_sha256,
            edge_rows=edge_rows,
            frame_rows=frame_rows,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        {
            "output_dir": str(output_dir),
            "EDGE_SUPPORT_AFTER_MASK_FIX": decision,
            "Top_after_safe_vs_covered": edge_rows["Top"]["after_safe_vs_covered"],
            "Bottom_after_safe_vs_covered": edge_rows["Bottom"]["after_safe_vs_covered"],
            "validation_read": False,
        }
    )


if __name__ == "__main__":
    run(parse_args())
