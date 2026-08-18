#!/usr/bin/env python3
"""Audit whether the current board mask removes edge laser candidates.

Only FIT 049--054 is opened.  The official board mask and Steger settings are
not modified; a full-image ``True`` mask is used only for this diagnostic
comparison.  No Cone/C1 model is fitted or modified, and no Validation image
is opened.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402


NEW_FIT_IDS = tuple(f"{value:03d}" for value in range(49, 55))
ROLES = ("chess", "nolaser", "laser")
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE_MM = 20.0
CURRENT_MASK_MARGIN_PX = -2
LASER_ORIENTATION = "vertical"
DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane_0817"
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "mask_exclusion_audit_049_054"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    return value


def inventory_new_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve exactly the requested FIT triplets; never enumerate validation."""
    fit_root = data_root / "fit"
    groups: dict[str, dict[str, Any]] = {}
    for frame_id in NEW_FIT_IDS:
        groups[frame_id] = {}
        for role in ROLES:
            path = fit_root / f"{role} {frame_id}.tif"
            if not path.is_file():
                raise FileNotFoundError(path)
            groups[frame_id][role] = {"path": path}
    return groups


def steger_params(cfg: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "sigma": float(cfg.get("sigma", 1.5)),
        "min_intensity": float(cfg.get("min_intensity", 8.0)),
        "min_response": float(cfg.get("min_response", 0.8)),
        "max_subpixel_offset": float(cfg.get("max_subpixel_offset", 0.6)),
    }


def select_params(cfg: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "poly_degree": int(cfg.get("continuity_poly_degree", 2)),
        "outlier_threshold_px": float(cfg.get("continuity_threshold_px", 2.0)),
        "max_points": int(cfg.get("max_points_per_image", 900)),
    }


def steger_candidates_with_mask(
    diff: np.ndarray, mask: np.ndarray, cfg: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = steger_params(cfg)
    return fixed.triplets.steger_candidates(
        diff=diff,
        mask=mask,
        sigma=float(params["sigma"]),
        min_intensity=float(params["min_intensity"]),
        min_response=float(params["min_response"]),
        max_subpixel_offset=float(params["max_subpixel_offset"]),
    )


def select_one_per_scanline(
    x: np.ndarray, y: np.ndarray, response: np.ndarray, cfg: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = select_params(cfg)
    return fixed.triplets.select_one_per_scanline(
        x,
        y,
        response,
        poly_degree=int(params["poly_degree"]),
        outlier_threshold_px=float(params["outlier_threshold_px"]),
        max_points=int(params["max_points"]),
        orientation=LASER_ORIENTATION,
    )


def projected_outer_boundary(pose: Any, intrinsics: Any) -> np.ndarray:
    """兼容旧审计调用；几何实现统一委托正式 mask helper。"""
    return fixed.triplets.projected_board_boundary(
        pose.rvec,
        pose.tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        pattern_cols=BOARD_COLS,
        pattern_rows=BOARD_ROWS,
        square_size_mm=SQUARE_SIZE_MM,
        inset_mm=0.0,
    )


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, points, 255)
    return mask > 0


def points_in_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    h, w = mask.shape[:2]
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    result = np.zeros(len(x), dtype=bool)
    result[valid] = mask[yi[valid], xi[valid]]
    return result


def range_for(values: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if len(selected) == 0:
        return None, None
    return float(np.min(selected)), float(np.max(selected))


def range_text(values: Sequence[Any], digits: int = 1) -> str:
    if len(values) != 2 or values[0] is None or values[1] is None:
        return "NA"
    return f"[{float(values[0]):.{digits}f}, {float(values[1]):.{digits}f}]"


def point_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def summarize_frame(
    *,
    frame_id: str,
    pose: Any,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    masked_x: np.ndarray,
    masked_y: np.ndarray,
    final_x: np.ndarray,
    final_y: np.ndarray,
    mask_free_selected_x: np.ndarray,
    mask_free_selected_y: np.ndarray,
    complete_mask: np.ndarray,
    hull_mask: np.ndarray,
    formal_mask: np.ndarray,
) -> dict[str, Any]:
    raw_complete = points_in_mask(raw_x, raw_y, complete_mask)
    raw_hull = points_in_mask(raw_x, raw_y, hull_mask)
    raw_formal = points_in_mask(raw_x, raw_y, formal_mask)
    mask_free_complete = points_in_mask(mask_free_selected_x, mask_free_selected_y, complete_mask)
    mask_free_hull = points_in_mask(mask_free_selected_x, mask_free_selected_y, hull_mask)
    final_complete = points_in_mask(final_x, final_y, complete_mask)
    raw_complete_outside_hull = raw_complete & ~raw_hull
    raw_complete_outside_formal = raw_complete & ~raw_formal
    raw_hull_but_eroded_out = raw_complete & raw_hull & ~raw_formal
    mask_free_complete_outside_hull = mask_free_complete & ~mask_free_hull

    raw_lost_v = range_for(raw_y, raw_complete_outside_hull)
    raw_lost_u = range_for(raw_x, raw_complete_outside_hull)
    selected_lost_v = range_for(mask_free_selected_y, mask_free_complete_outside_hull)
    final_v = range_for(final_y, np.ones(len(final_y), dtype=bool))
    return {
        "frame_id": frame_id,
        "pnp_rmse_px": float(pose.reprojection_rmse_px),
        "raw_candidate_count": int(len(raw_x)),
        "raw_candidate_inside_complete_count": point_count(raw_complete),
        "raw_candidate_inside_hull_count": point_count(raw_hull),
        "raw_candidate_inside_formal_mask_count": point_count(raw_formal),
        "masked_candidate_count_before_selection": int(len(masked_x)),
        "mask_free_selected_count": int(len(mask_free_selected_x)),
        "final_masked_point_count": int(len(final_x)),
        "raw_complete_outside_hull_count": point_count(raw_complete_outside_hull),
        "raw_complete_outside_formal_mask_count": point_count(raw_complete_outside_formal),
        "raw_complete_inside_hull_but_eroded_out_count": point_count(raw_hull_but_eroded_out),
        "mask_free_selected_complete_outside_hull_count": point_count(mask_free_complete_outside_hull),
        "final_inside_complete_count": point_count(final_complete),
        "raw_complete_outside_hull_v_min_px": raw_lost_v[0],
        "raw_complete_outside_hull_v_max_px": raw_lost_v[1],
        "raw_complete_outside_hull_u_min_px": raw_lost_u[0],
        "raw_complete_outside_hull_u_max_px": raw_lost_u[1],
        "mask_free_selected_outside_hull_v_min_px": selected_lost_v[0],
        "mask_free_selected_outside_hull_v_max_px": selected_lost_v[1],
        "final_v_min_px": final_v[0],
        "final_v_max_px": final_v[1],
    }


def draw_points(image: np.ndarray, x: np.ndarray, y: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    height, width = image.shape[:2]
    for px, py in zip(np.asarray(x), np.asarray(y)):
        ix, iy = int(round(float(px))), int(round(float(py)))
        if 0 <= ix < width and 0 <= iy < height:
            cv2.circle(image, (ix, iy), radius, color, -1, lineType=cv2.LINE_AA)


def make_overlay(
    chess: np.ndarray,
    frame_id: str,
    corners: np.ndarray,
    outer_boundary: np.ndarray,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    final_x: np.ndarray,
    final_y: np.ndarray,
    summary: Mapping[str, Any],
    output: Path,
) -> None:
    gray = fixed.triplets.to_gray_float(chess)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    base_image = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    points_image = base_image.copy()
    # BGR: green raw candidates, blue current final points.
    draw_points(points_image, raw_x, raw_y, (0, 220, 0), 1)
    draw_points(points_image, final_x, final_y, (255, 0, 0), 2)
    vis = cv2.addWeighted(points_image, 0.72, base_image, 0.28, 0.0)

    hull = cv2.convexHull(np.rint(np.asarray(corners)).astype(np.int32)).reshape(-1, 1, 2)
    outer = np.rint(np.asarray(outer_boundary)).astype(np.int32).reshape(-1, 1, 2)
    # BGR: red inner-corner hull, yellow theoretical outer boundary.
    cv2.polylines(vis, [hull], True, (0, 0, 255), 5, lineType=cv2.LINE_AA)
    cv2.polylines(vis, [outer], True, (0, 255, 255), 5, lineType=cv2.LINE_AA)
    cv2.putText(vis, f"frame {frame_id}  PnP RMSE {summary['pnp_rmse_px']:.3f}px", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, f"raw={summary['raw_candidate_count']} final={summary['final_masked_point_count']}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, f"raw inside outer & outside hull={summary['raw_complete_outside_hull_count']}", (24, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    fixed.triplets.imwrite_unicode(output, vis)


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


def report_text(
    *,
    output_dir: Path,
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    conclusion: str,
    raw_lost_total: int,
    selected_lost_total: int,
    formal_lost_total: int,
    eroded_lost_total: int,
) -> str:
    lines = [
        "# Board-mask edge exclusion audit for FIT 049–054",
        "",
        f"`EDGE_POINTS_LOST_BY_BOARD_MASK = {conclusion}`",
        "",
        "## Scope and method",
        "",
        f"- 只打开 `{data_root / 'fit'}` 下的 `chess/laser/nolaser 049–054.tif`；没有打开任何 Validation 图像。",
        "- 棋盘 PnP：11×8 内角点，格距 20 mm；理论外边界使用物理坐标 `X=[-20,220] mm`、`Y=[-20,160] mm`，再用 PnP 投影回图像。",
        "- 红色 = 检测内角点的原始 convex hull；黄色 = 理论完整棋盘外边界；绿色 = 全图 mask=True 时的 Steger 原始候选；蓝色 = 当前 `board_inner_mask(margin_px=-2)` 后沿用正式连续性筛选/900 点上限得到的最终点。",
        "- 当前正式 mask 未修改；Cone/C1 未用于本审计计算、未拟合或修改；Steger 参数和 `vertical` 方向沿用现有流程。",
        "- `raw_complete_outside_hull` 是“理论完整棋盘内、但当前 inner-corner hull 外”的原始 Steger 候选；同时报告 mask-free 连续性筛选后的同类点，避免把单纯噪声候选误解为稳定激光线。",
        "",
        "## Per-frame result",
        "",
        "| frame | PnP RMSE / px | raw candidates | masked candidates before selection | final points | complete∩outside hull raw | v range / px | complete∩outside hull mask-free selected | v range / px |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['frame_id']} | {float(row['pnp_rmse_px']):.4f} | {row['raw_candidate_count']} | "
            f"{row['masked_candidate_count_before_selection']} | {row['final_masked_point_count']} | "
            f"{row['raw_complete_outside_hull_count']} | "
            f"{range_text((row['raw_complete_outside_hull_v_min_px'], row['raw_complete_outside_hull_v_max_px']))} | "
            f"{row['mask_free_selected_complete_outside_hull_count']} | "
            f"{range_text((row['mask_free_selected_outside_hull_v_min_px'], row['mask_free_selected_outside_hull_v_max_px']))} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- raw Steger 候选中落在完整棋盘内、但当前 inner-corner hull 外的总数：`{raw_lost_total}`。",
            f"- 落在完整棋盘内、但当前正式 eroded mask 外的总数：`{formal_lost_total}`；其中仅由 `margin_px=-2` 腐蚀额外排除、但仍在原始 hull 内的数量：`{eroded_lost_total}`。",
            f"- 经过 mask-free 的同一连续性筛选后，仍落在完整棋盘内、但 hull 外的总数：`{selected_lost_total}`。",
            f"- `EDGE_POINTS_LOST_BY_BOARD_MASK = {conclusion}`。",
        ]
    )
    if conclusion == "YES":
        lines.append("- 这说明当前 mask 确实排除了至少一部分理论完整棋盘区域内的 Steger 候选；是否为真实可见激光点，应结合 overlay 中绿色点是否形成连续激光线判断。")
    else:
        lines.append("- 未发现完整棋盘内、inner-corner hull 外的 Steger 候选；当前边缘点缺失不能归因于该 hull mask。")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `mask_exclusion_summary.csv`: `{output_dir / 'mask_exclusion_summary.csv'}`",
            f"- `mask_overlay_049.png` … `mask_overlay_054.png`: `{output_dir}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")

    groups = inventory_new_fit(data_root)
    _, _, _, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    cfg = fixed.EXTRACTION_CONFIG
    rows: list[dict[str, Any]] = []
    raw_lost_total = 0
    selected_lost_total = 0
    formal_lost_total = 0
    eroded_lost_total = 0

    for frame_id in NEW_FIT_IDS:
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
        current_mask = fixed.triplets.board_inner_mask(shape, pose.corners, margin_px=CURRENT_MASK_MARGIN_PX)
        hull_polygon = cv2.convexHull(np.rint(np.asarray(pose.corners)).astype(np.int32)).reshape(-1, 2)
        outer_polygon = projected_outer_boundary(pose, intrinsics)
        hull_mask = polygon_mask(shape, hull_polygon)
        complete_mask = polygon_mask(shape, outer_polygon)

        diff = fixed.triplets.positive_difference(laser, background)
        full_mask = np.ones(diff.shape, dtype=bool)
        raw_x, raw_y, raw_response = steger_candidates_with_mask(diff, full_mask, cfg)
        masked_x, masked_y, masked_response = steger_candidates_with_mask(diff, current_mask, cfg)
        mask_free_selected_x, mask_free_selected_y, _ = select_one_per_scanline(raw_x, raw_y, raw_response, cfg)
        final_x, final_y, _, _ = fixed.triplets.extract_laser_centers(
            laser, background, current_mask, cfg, LASER_ORIENTATION
        )

        row = summarize_frame(
            frame_id=frame_id,
            pose=pose,
            raw_x=raw_x,
            raw_y=raw_y,
            masked_x=masked_x,
            masked_y=masked_y,
            final_x=final_x,
            final_y=final_y,
            mask_free_selected_x=mask_free_selected_x,
            mask_free_selected_y=mask_free_selected_y,
            complete_mask=complete_mask,
            hull_mask=hull_mask,
            formal_mask=current_mask,
        )
        rows.append(row)
        raw_lost_total += int(row["raw_complete_outside_hull_count"])
        selected_lost_total += int(row["mask_free_selected_complete_outside_hull_count"])
        formal_lost_total += int(row["raw_complete_outside_formal_mask_count"])
        eroded_lost_total += int(row["raw_complete_inside_hull_but_eroded_out_count"])
        make_overlay(
            chess=chess,
            frame_id=frame_id,
            corners=pose.corners,
            outer_boundary=outer_polygon,
            raw_x=raw_x,
            raw_y=raw_y,
            final_x=final_x,
            final_y=final_y,
            summary=row,
            output=output_dir / f"mask_overlay_{frame_id}.png",
        )

    conclusion = "YES" if raw_lost_total > 0 else "NO"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "mask_exclusion_summary.csv", rows)
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            data_root=data_root,
            rows=rows,
            conclusion=conclusion,
            raw_lost_total=raw_lost_total,
            selected_lost_total=selected_lost_total,
            formal_lost_total=formal_lost_total,
            eroded_lost_total=eroded_lost_total,
        ),
        encoding="utf-8",
    )
    print(
        {
            "output_dir": str(output_dir),
            "EDGE_POINTS_LOST_BY_BOARD_MASK": conclusion,
            "raw_complete_outside_hull_total": raw_lost_total,
            "raw_complete_outside_formal_mask_total": formal_lost_total,
            "raw_complete_inside_hull_but_eroded_out_total": eroded_lost_total,
            "mask_free_selected_outside_hull_total": selected_lost_total,
            "validation_read": False,
        }
    )


if __name__ == "__main__":
    run(parse_args())
