#!/usr/bin/env python3
"""Verify full-board final-line continuity for FIT 049--054.

The diagnostic uses the theoretical full-board polygon only for Steger
selection.  It compares that final centerline with the existing formal
inner-hull result, without drawing raw candidates and without fitting or
modifying any Cone/C1 model.  Only ``fit/049--054`` is opened.
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

import audit_board_mask_exclusion_049_054 as mask_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402


NEW_FIT_IDS = tuple(f"{value:03d}" for value in range(49, 55))
ROLES = ("chess", "nolaser", "laser")
BOARD_COLS = 11
BOARD_ROWS = 8
SQUARE_SIZE_MM = 20.0
CURRENT_MASK_MARGIN_PX = -2
LASER_ORIENTATION = "vertical"
CLUSTER_GAP_PX = 5.0
TOP_DUAL_SEGMENT_BREAK_PX = 100.0
BRANCH_AMBIGUITY_MARGIN_PX = 0.5
LOCAL_WINDOW_POINTS = 8
DEFAULT_DATA_ROOT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane_0817"
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0817"
    / "final_line_continuity_049_054"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--measurement-config", type=Path, default=fixed.DEFAULT_MEASUREMENT_CONFIG)
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


def inventory_new_fit(data_root: Path) -> dict[str, dict[str, Any]]:
    """Resolve exactly FIT 049--054; do not enumerate validation."""
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


def line_sorted(x: np.ndarray, y: np.ndarray, response: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if response is None:
        response_sorted = None
    else:
        response_sorted = np.asarray(response, dtype=np.float64)
    order = np.argsort(y, kind="stable")
    return x[order], y[order], None if response_sorted is None else response_sorted[order]


def line_statistics(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x, y, _ = line_sorted(x, y)
    if len(x) < 2:
        return {
            "count": int(len(x)),
            "v_min_px": float(np.min(y)) if len(y) else math.nan,
            "v_max_px": float(np.max(y)) if len(y) else math.nan,
            "max_dv_px": math.nan,
            "p95_dv_px": math.nan,
            "max_abs_du_px": math.nan,
            "p95_abs_du_px": math.nan,
            "max_abs_du_dv": math.nan,
            "p95_abs_du_dv": math.nan,
        }
    dx = np.diff(x)
    dy = np.diff(y)
    valid = np.abs(dy) > 1.0e-9
    slopes = dx[valid] / dy[valid]
    return {
        "count": int(len(x)),
        "v_min_px": float(np.min(y)),
        "v_max_px": float(np.max(y)),
        "max_dv_px": float(np.max(np.abs(dy))),
        "p95_dv_px": float(np.percentile(np.abs(dy), 95)),
        "max_abs_du_px": float(np.max(np.abs(dx))),
        "p95_abs_du_px": float(np.percentile(np.abs(dx), 95)),
        "max_abs_du_dv": float(np.max(np.abs(slopes))) if len(slopes) else math.nan,
        "p95_abs_du_dv": float(np.percentile(np.abs(slopes), 95)) if len(slopes) else math.nan,
    }


def local_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(y) <= 1.0e-9:
        return math.nan
    return float(np.polyfit(y, x, deg=1)[0])


def interpolate_line(x: np.ndarray, y: np.ndarray, target_y: float) -> float | None:
    x, y, _ = line_sorted(x, y)
    if len(y) < 2 or target_y < y[0] or target_y > y[-1]:
        return None
    return float(np.interp(target_y, y, x))


def boundary_events(
    x: np.ndarray,
    y: np.ndarray,
    hull_mask: np.ndarray,
    current_x: np.ndarray,
    current_y: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, y, _ = line_sorted(x, y)
    inside = mask_audit.points_in_mask(x, y, hull_mask)
    transitions = np.flatnonzero(inside[1:] != inside[:-1])
    dx = np.diff(x)
    dy = np.diff(y)
    global_abs_du = np.abs(dx)
    global_p95_du = float(np.percentile(global_abs_du, 95)) if len(global_abs_du) else math.nan
    events: list[dict[str, Any]] = []
    for index in transitions:
        before_start = max(0, int(index) - LOCAL_WINDOW_POINTS + 1)
        before_x = x[before_start : int(index) + 1]
        before_y = y[before_start : int(index) + 1]
        after_end = min(len(x), int(index) + 1 + LOCAL_WINDOW_POINTS)
        after_x = x[int(index) + 1 : after_end]
        after_y = y[int(index) + 1 : after_end]
        slope_before = local_slope(before_x, before_y)
        slope_after = local_slope(after_x, after_y)
        u_jump = float(abs(x[index + 1] - x[index]))
        v_jump = float(abs(y[index + 1] - y[index]))
        local_du = np.abs(np.r_[np.diff(before_x), np.diff(after_x)])
        local_du = local_du[np.isfinite(local_du)]
        local_median_du = float(np.median(local_du)) if len(local_du) else math.nan
        jump_limit = max(3.0, 4.0 * local_median_du) if math.isfinite(local_median_du) else 3.0
        slope_delta = abs(slope_after - slope_before) if math.isfinite(slope_before) and math.isfinite(slope_after) else math.nan
        events.append(
            {
                "record_type": "boundary_event",
                "transition_index": int(index),
                "transition_direction": "outside_to_inside" if not inside[index] else "inside_to_outside",
                "v_before_px": float(y[index]),
                "v_after_px": float(y[index + 1]),
                "u_before_px": float(x[index]),
                "u_after_px": float(x[index + 1]),
                "v_jump_px": v_jump,
                "u_jump_px": u_jump,
                "slope_before_du_dv": slope_before,
                "slope_after_du_dv": slope_after,
                "slope_delta_abs": slope_delta,
                "local_median_abs_du_px": local_median_du,
                "global_p95_abs_du_px": global_p95_du,
                "u_jump_limit_px": jump_limit,
                "u_jump_is_outlier": bool(u_jump > jump_limit),
                "current_u_at_v_before_px": interpolate_line(current_x, current_y, float(y[index])),
                "current_u_at_v_after_px": interpolate_line(current_x, current_y, float(y[index + 1])),
            }
        )
    summary = {
        "boundary_transition_count": int(len(events)),
        "boundary_u_jump_max_px": max((float(event["u_jump_px"]) for event in events), default=math.nan),
        "boundary_slope_delta_max_abs": max((float(event["slope_delta_abs"]) for event in events if math.isfinite(float(event["slope_delta_abs"]))), default=math.nan),
        "boundary_u_jump_outlier_count": int(sum(bool(event["u_jump_is_outlier"]) for event in events)),
        "boundary_continuity_ok": bool(len(events) >= 2 and all(not bool(event["u_jump_is_outlier"]) for event in events)),
    }
    return events, summary


def row_clusters(x_values: Sequence[float], response_values: Sequence[float]) -> list[dict[str, float]]:
    if not x_values:
        return []
    order = np.argsort(np.asarray(x_values, dtype=np.float64))
    x = np.asarray(x_values, dtype=np.float64)[order]
    response = np.asarray(response_values, dtype=np.float64)[order]
    cuts = np.flatnonzero(np.diff(x) > CLUSTER_GAP_PX) + 1
    groups = np.split(np.arange(len(x)), cuts)
    clusters: list[dict[str, float]] = []
    for group in groups:
        xx = x[group]
        rr = response[group]
        weights = np.maximum(rr, 1.0e-9)
        clusters.append(
            {
                "x_center": float(np.average(xx, weights=weights)),
                "x_min": float(np.min(xx)),
                "x_max": float(np.max(xx)),
                "response_sum": float(np.sum(rr)),
                "response_peak": float(np.max(rr)),
                "count": int(len(xx)),
            }
        )
    return clusters


def polynomial_prediction(y_values: np.ndarray, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    y_values = np.asarray(y_values, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)
    center = float(np.mean(y_values))
    scale = max(float(np.std(y_values)), 1.0)
    z = (y_values - center) / scale
    degree = min(2, len(y_values) - 1)
    coefficients = np.polyfit(z, x_values, deg=degree)
    predicted = np.polyval(coefficients, z)
    residual = x_values - predicted
    return coefficients, np.asarray([center, scale], dtype=np.float64), float(np.sqrt(np.mean(residual * residual))), float(np.percentile(np.abs(residual), 95))


def predict_polynomial(coefficients: np.ndarray, norm: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    z = (np.asarray(y_values, dtype=np.float64) - float(norm[0])) / float(norm[1])
    return np.polyval(coefficients, z)


def branch_analysis(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    raw_response: np.ndarray,
    full_x: np.ndarray,
    full_y: np.ndarray,
) -> dict[str, Any]:
    rows: dict[int, list[tuple[float, float]]] = {}
    for x_value, y_value, response in zip(raw_x, raw_y, raw_response):
        rows.setdefault(int(round(float(y_value))), []).append((float(x_value), float(response)))
    dual: list[tuple[int, list[dict[str, float]]]] = []
    for scanline, values in rows.items():
        clusters = row_clusters([item[0] for item in values], [item[1] for item in values])
        if len(clusters) >= 2 and clusters[-1]["x_center"] - clusters[0]["x_center"] > CLUSTER_GAP_PX:
            dual.append((scanline, clusters))
    dual.sort(key=lambda item: item[0])
    if not dual:
        return {
            "top_dual_region_y_min_px": math.nan,
            "top_dual_region_y_max_px": math.nan,
            "dual_row_count": 0,
            "branch_a_label": "A_left",
            "branch_b_label": "B_right",
            "branch_a_fit_rmse_px": math.nan,
            "branch_b_fit_rmse_px": math.nan,
            "branch_a_fit_p95_px": math.nan,
            "branch_b_fit_p95_px": math.nan,
            "branch_separation_min_px": math.nan,
            "branch_separation_median_px": math.nan,
            "selected_branch": "NO_DUAL_REGION",
            "selected_branch_a_count": 0,
            "selected_branch_b_count": 0,
            "selected_branch_ambiguous_count": 0,
            "selected_branch_switch_count": 0,
            "selected_branch_longest_run": 0,
            "selected_branch_a_fraction": math.nan,
            "selected_branch_b_fraction": math.nan,
            "selected_branch_ambiguous_fraction": math.nan,
        }

    segments: list[list[tuple[int, list[dict[str, float]]]]] = [[dual[0]]]
    for item in dual[1:]:
        if item[0] - segments[-1][-1][0] > TOP_DUAL_SEGMENT_BREAK_PX:
            segments.append([item])
        else:
            segments[-1].append(item)
    top_segment = segments[0]
    y_branch = np.asarray([item[0] for item in top_segment], dtype=np.float64)
    x_a = np.asarray([item[1][0]["x_center"] for item in top_segment], dtype=np.float64)
    x_b = np.asarray([item[1][-1]["x_center"] for item in top_segment], dtype=np.float64)
    coef_a, norm_a, rmse_a, p95_a = polynomial_prediction(y_branch, x_a)
    coef_b, norm_b, rmse_b, p95_b = polynomial_prediction(y_branch, x_b)
    y_eval = np.linspace(float(np.min(y_branch)), float(np.max(y_branch)), 100)
    separation = predict_polynomial(coef_b, norm_b, y_eval) - predict_polynomial(coef_a, norm_a, y_eval)

    full_x_sorted, full_y_sorted, _ = line_sorted(full_x, full_y)
    selected_region = (full_y_sorted >= np.min(y_branch)) & (full_y_sorted <= np.max(y_branch))
    selected_y = full_y_sorted[selected_region]
    selected_x = full_x_sorted[selected_region]
    pred_a = predict_polynomial(coef_a, norm_a, selected_y) if len(selected_y) else np.empty(0)
    pred_b = predict_polynomial(coef_b, norm_b, selected_y) if len(selected_y) else np.empty(0)
    distance_a = np.abs(selected_x - pred_a)
    distance_b = np.abs(selected_x - pred_b)
    labels = np.full(len(selected_x), "ambiguous", dtype=object)
    labels[(distance_a + BRANCH_AMBIGUITY_MARGIN_PX) < distance_b] = "A_left"
    labels[(distance_b + BRANCH_AMBIGUITY_MARGIN_PX) < distance_a] = "B_right"
    label_list = [str(label) for label in labels]
    a_count = label_list.count("A_left")
    b_count = label_list.count("B_right")
    ambiguous_count = label_list.count("ambiguous")
    unambiguous = [label for label in label_list if label != "ambiguous"]
    switch_count = sum(unambiguous[index] != unambiguous[index - 1] for index in range(1, len(unambiguous)))
    longest_run = 0
    current_run = 0
    previous = None
    for label in unambiguous:
        if label == previous:
            current_run += 1
        else:
            current_run = 1
            previous = label
        longest_run = max(longest_run, current_run)
    if not unambiguous:
        selected_branch = "AMBIGUOUS"
    elif a_count / len(selected_x) >= 0.8 and b_count == 0:
        selected_branch = "A_left"
    elif b_count / len(selected_x) >= 0.8 and a_count == 0:
        selected_branch = "B_right"
    elif switch_count == 0 and a_count >= b_count:
        selected_branch = "A_left_MOSTLY"
    elif switch_count == 0:
        selected_branch = "B_right_MOSTLY"
    else:
        selected_branch = "MIXED"
    return {
        "top_dual_region_y_min_px": int(np.min(y_branch)),
        "top_dual_region_y_max_px": int(np.max(y_branch)),
        "dual_row_count": int(len(y_branch)),
        "branch_a_label": "A_left",
        "branch_b_label": "B_right",
        "branch_a_fit_rmse_px": rmse_a,
        "branch_b_fit_rmse_px": rmse_b,
        "branch_a_fit_p95_px": p95_a,
        "branch_b_fit_p95_px": p95_b,
        "branch_separation_min_px": float(np.min(separation)),
        "branch_separation_median_px": float(np.median(separation)),
        "selected_branch": selected_branch,
        "selected_branch_a_count": int(a_count),
        "selected_branch_b_count": int(b_count),
        "selected_branch_ambiguous_count": int(ambiguous_count),
        "selected_branch_switch_count": int(switch_count),
        "selected_branch_longest_run": int(longest_run),
        "selected_branch_a_fraction": float(a_count / len(selected_x)) if len(selected_x) else math.nan,
        "selected_branch_b_fraction": float(b_count / len(selected_x)) if len(selected_x) else math.nan,
        "selected_branch_ambiguous_fraction": float(ambiguous_count / len(selected_x)) if len(selected_x) else math.nan,
    }


def line_panel(
    chess: np.ndarray,
    frame_id: str,
    pose: Any,
    outer_boundary: np.ndarray,
    full_x: np.ndarray,
    full_y: np.ndarray,
    current_x: np.ndarray,
    current_y: np.ndarray,
    hull_mask: np.ndarray,
    summary: Mapping[str, Any],
    size: tuple[int, int] = (1024, 750),
) -> np.ndarray:
    gray = fixed.triplets.to_gray_float(chess)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    height, width = base.shape[:2]
    out_w, out_h = size
    panel = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
    sx, sy = out_w / width, out_h / height

    def scaled_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.column_stack([np.asarray(x) * sx, np.asarray(y) * sy]).astype(np.int32).reshape(-1, 1, 2)

    full_x_sorted, full_y_sorted, _ = line_sorted(full_x, full_y)
    current_x_sorted, current_y_sorted, _ = line_sorted(current_x, current_y)
    # Full-board final centerline: blue. Current formal centerline: magenta.
    if len(full_x_sorted) >= 2:
        cv2.polylines(panel, [scaled_points(full_x_sorted, full_y_sorted)], False, (255, 0, 0), 3, cv2.LINE_AA)
    if len(current_x_sorted) >= 2:
        cv2.polylines(panel, [scaled_points(current_x_sorted, current_y_sorted)], False, (255, 0, 255), 4, cv2.LINE_AA)
    # Draw only geometric boundaries, never raw candidates.
    hull = cv2.convexHull(np.rint(np.asarray(pose.corners)).astype(np.int32)).reshape(-1, 1, 2)
    hull_scaled = np.rint(hull.astype(np.float64) * np.asarray([sx, sy])).astype(np.int32)
    outer_scaled = np.rint(np.asarray(outer_boundary) * np.asarray([sx, sy])).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(panel, [hull_scaled], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.polylines(panel, [outer_scaled], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"{frame_id}  full={len(full_x_sorted)} current={len(current_x_sorted)}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"branch={summary.get('selected_branch', 'NA')}", (14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


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
    frame_rows: Sequence[Mapping[str, Any]],
    conclusion: str,
    same_branch: str,
) -> str:
    lines = [
        "# Full-board final centerline continuity audit",
        "",
        f"`OUTER_BOARD_LINE_CONTINUITY = {conclusion}`",
        "",
        "## Scope and method",
        "",
        f"- 只打开 `{data_root / 'fit'}` 下的 FIT `049–054` 三联图；没有打开任何 Validation 图像。",
        "- 使用理论完整棋盘边界 `X=[-20,220] mm`、`Y=[-20,160] mm` 的投影 mask，重新执行现有 Steger、`vertical` 每 row 单点、continuity 和 900 点上限。",
        "- 输出 overlay 只画最终中心线：蓝色 = 完整棋盘 mask 最终线，洋红色 = 当前正式 inner-hull mask 最终线；红色 = inner-corner hull，黄色 = 理论外边界；不画 raw candidates。",
        "- 未拟合或修改 Cone/C1/K/D；正式 mask 未修改。",
        "",
        "## Boundary continuity",
        "",
        "| frame | full line v range | current line v range | boundary transitions | max u jump / px | max slope delta | jump outliers | boundary status |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame_rows:
        lines.append(
            f"| {row['frame_id']} | [{fmt(row['full_v_min_px'], 1)}, {fmt(row['full_v_max_px'], 1)}] | "
            f"[{fmt(row['current_v_min_px'], 1)}, {fmt(row['current_v_max_px'], 1)}] | "
            f"{row['boundary_transition_count']} | {fmt(row['boundary_u_jump_max_px'], 3)} | "
            f"{fmt(row['boundary_slope_delta_max_abs'], 5)} | {row['boundary_u_jump_outlier_count']} | "
            f"{str(row['boundary_continuity_ok']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## 052–054 top dual-candidate branch",
            "",
            "Branch A/B is defined geometrically per row: A = smaller u (left branch), B = larger u (right branch). The dual region is the first persistent raw-candidate dual band from the top, with a >100 px gap separating later isolated bands.",
            "",
            "| frame | dual-region v | dual rows | branch A fit RMSE/P95 | branch B fit RMSE/P95 | selected A | selected B | ambiguous | switches | final choice |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in frame_rows:
        if row["frame_id"] not in {"052", "053", "054"}:
            continue
        lines.append(
            f"| {row['frame_id']} | [{fmt(row['top_dual_region_y_min_px'], 1)}, {fmt(row['top_dual_region_y_max_px'], 1)}] | "
            f"{row['dual_row_count']} | {fmt(row['branch_a_fit_rmse_px'], 3)}/{fmt(row['branch_a_fit_p95_px'], 3)} | "
            f"{fmt(row['branch_b_fit_rmse_px'], 3)}/{fmt(row['branch_b_fit_p95_px'], 3)} | "
            f"{row['selected_branch_a_count']} ({100.0 * float(row['selected_branch_a_fraction']):.1f}%) | "
            f"{row['selected_branch_b_count']} ({100.0 * float(row['selected_branch_b_fraction']):.1f}%) | "
            f"{row['selected_branch_ambiguous_count']} ({100.0 * float(row['selected_branch_ambiguous_fraction']):.1f}%) | "
            f"{row['selected_branch_switch_count']} | `{row['selected_branch']}` |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Inner-hull boundary continuity overall: `{conclusion}`。",
            f"- 052–054 top dual region final selection consistency: `{same_branch}`；报告中的 B_right 表示较大 u 的右侧分支。",
            "- 如果 selected branch 在同一帧发生切换、或 boundary u jump 成为局部异常值，则不能把外圈点与正式中心线视为同一条稳定中心线。",
            "",
            "## Artifacts",
            "",
            f"- `final_line_overlay_049_054.png`: `{output_dir / 'final_line_overlay_049_054.png'}`",
            f"- `boundary_continuity.csv`: `{output_dir / 'boundary_continuity.csv'}`",
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
    frame_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []

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
        hull_mask = mask_audit.polygon_mask(shape, hull_polygon)
        outer_polygon = mask_audit.projected_outer_boundary(pose, intrinsics)
        complete_mask = mask_audit.polygon_mask(shape, outer_polygon)
        diff = fixed.triplets.positive_difference(laser, background)
        raw_x, raw_y, raw_response = mask_audit.steger_candidates_with_mask(diff, complete_mask, cfg)
        full_x, full_y, full_response = mask_audit.select_one_per_scanline(raw_x, raw_y, raw_response, cfg)
        current_x, current_y, current_response, _ = fixed.triplets.extract_laser_centers(
            laser, background, current_mask, cfg, LASER_ORIENTATION
        )
        full_x, full_y, full_response = line_sorted(full_x, full_y, full_response)
        current_x, current_y, current_response = line_sorted(current_x, current_y, current_response)
        full_stats = line_statistics(full_x, full_y)
        current_stats = line_statistics(current_x, current_y)
        events, boundary_summary = boundary_events(full_x, full_y, hull_mask, current_x, current_y)
        branch = branch_analysis(raw_x, raw_y, raw_response, full_x, full_y)
        row: dict[str, Any] = {
            "record_type": "frame_summary",
            "frame_id": frame_id,
            "pnp_rmse_px": float(pose.reprojection_rmse_px),
            "full_final_count": len(full_x),
            "current_final_count": len(current_x),
            "full_v_min_px": full_stats["v_min_px"],
            "full_v_max_px": full_stats["v_max_px"],
            "current_v_min_px": current_stats["v_min_px"],
            "current_v_max_px": current_stats["v_max_px"],
            "full_max_dv_px": full_stats["max_dv_px"],
            "full_p95_dv_px": full_stats["p95_dv_px"],
            "full_max_abs_du_px": full_stats["max_abs_du_px"],
            "full_p95_abs_du_px": full_stats["p95_abs_du_px"],
            "full_max_abs_du_dv": full_stats["max_abs_du_dv"],
            "full_p95_abs_du_dv": full_stats["p95_abs_du_dv"],
            "current_max_abs_du_px": current_stats["max_abs_du_px"],
            "current_p95_abs_du_px": current_stats["p95_abs_du_px"],
            **boundary_summary,
            **branch,
        }
        frame_rows.append(row)
        for event_index, event in enumerate(events):
            event_row = {"record_type": "boundary_event", "frame_id": frame_id, "event_index": event_index, **event}
            event_rows.append(event_row)
        panels.append(
            line_panel(
                chess=chess,
                frame_id=frame_id,
                pose=pose,
                outer_boundary=outer_polygon,
                full_x=full_x,
                full_y=full_y,
                current_x=current_x,
                current_y=current_y,
                hull_mask=hull_mask,
                summary=row,
            )
        )

    boundary_ok = all(bool(row["boundary_continuity_ok"]) for row in frame_rows)
    focus_rows = [row for row in frame_rows if row["frame_id"] in {"052", "053", "054"}]
    focus_choices = [row["selected_branch"] for row in focus_rows]
    focus_stable = bool(
        len(focus_rows) == 3
        and all(choice in {"A_left", "B_right"} for choice in focus_choices)
        and len(set(focus_choices)) == 1
        and all(int(row["selected_branch_switch_count"]) == 0 for row in focus_rows)
        and all(float(row["selected_branch_ambiguous_fraction"]) <= 0.10 for row in focus_rows)
    )
    if boundary_ok and focus_stable:
        conclusion = "PASS"
    elif all(int(row["boundary_u_jump_outlier_count"]) == 0 for row in frame_rows) and all(choice != "MIXED" for choice in focus_choices):
        conclusion = "PARTIAL"
    else:
        conclusion = "FAIL"
    same_branch = "STABLE_" + focus_choices[0] if focus_stable else "NOT_STABLE"

    output_dir.mkdir(parents=True, exist_ok=True)
    # 2x3 montage, final lines only; no raw candidates are drawn.
    row_panels = [cv2.hconcat(panels[index : index + 3]) for index in (0, 3)]
    montage = cv2.vconcat(row_panels)
    cv2.putText(montage, "blue=full-board final line  magenta=current formal line  red=inner hull  yellow=outer boundary", (18, montage.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    fixed.triplets.imwrite_unicode(output_dir / "final_line_overlay_049_054.png", montage)
    write_csv(output_dir / "boundary_continuity.csv", frame_rows + event_rows)
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            data_root=data_root,
            frame_rows=frame_rows,
            conclusion=conclusion,
            same_branch=same_branch,
        ),
        encoding="utf-8",
    )
    print(
        {
            "output_dir": str(output_dir),
            "OUTER_BOARD_LINE_CONTINUITY": conclusion,
            "top_dual_selected_branch": same_branch,
            "validation_read": False,
        }
    )


if __name__ == "__main__":
    run(parse_args())
