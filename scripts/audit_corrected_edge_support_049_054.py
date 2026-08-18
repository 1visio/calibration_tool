#!/usr/bin/env python3
"""Re-evaluate FIT 049--054 after correcting the Top/Bottom assignment.

Only the after-mask extraction is repeated.  The mapping is intentionally
corrected to 052--054=Top and 049--051=Bottom; no Validation data is opened.
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
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import audit_board_coordinate_residual as board  # noqa: E402
import audit_board_mask_exclusion_049_054 as mask_audit  # noqa: E402
import audit_fixed_pose_frame_repeatability as fixed  # noqa: E402
import freeze_and_validate_c1_4k as frozen  # noqa: E402
import validate_standard_object_accuracy as base  # noqa: E402


FRAME_IDS = tuple(f"{value:03d}" for value in range(49, 55))
EDGE_BY_FRAME = {
    "052": "Top",
    "053": "Top",
    "054": "Top",
    "049": "Bottom",
    "050": "Bottom",
    "051": "Bottom",
}
EDGE_FRAME_IDS = {"Top": ("052", "053", "054"), "Bottom": ("049", "050", "051")}
JOINT_FULL_THRESHOLD = 0.95
DEFAULT_DATA_ROOT = ROOT / "projects" / "daheng" / "data" / "laser_plane_0817"
DEFAULT_C1_MODEL = base.DEFAULT_C1_MODEL
DEFAULT_OUTPUT_DIR = ROOT / "projects" / "daheng" / "outputs" / "0817" / "corrected_edge_support_049_054"


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
    joint_v_min, joint_v_max = finite_range(v[joint])
    joint_s_min, joint_s_max = finite_range(s[joint])
    v_cond_min, v_cond_max = finite_range(v[v_given_s])
    s_cond_min, s_cond_max = finite_range(s[s_given_v])

    # The conditional spans require the same point set to satisfy the other
    # coordinate's target interval. This avoids a false pass from unrelated
    # extrema in v and s. A 95% span threshold avoids treating sub-pixel
    # sampling of a target endpoint as a missing domain.
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


def target_domains() -> dict[str, dict[str, dict[str, list[float]]]]:
    targets = base_targets_from_previous()
    return targets


def base_targets_from_previous() -> dict[str, dict[str, dict[str, list[float]]]]:
    # The actual observed and safety domains are frozen from the prior
    # coverage plan; this audit does not read standard-object or Validation data.
    return {
        "Top": {
            "observed": {"v": [86.0, 467.0], "s": [-0.18500106019725174, -0.13286152907480264]},
            "safe": {"v": [30.0, 520.0], "s": [-0.191, -0.127]},
        },
        "Bottom": {
            "observed": {"v": [2810.0, 2938.0], "s": [0.1863643564578584, 0.2037112277937999]},
            "safe": {"v": [2760.0, 2990.0], "s": [0.181, 0.209]},
        },
    }


def summarize_points(
    *,
    edge: str,
    frame_ids: Sequence[str],
    s: np.ndarray,
    v: np.ndarray,
    steger_count: int,
    targets: Mapping[str, Mapping[str, Sequence[float]]],
    record_type: str,
) -> dict[str, Any]:
    v_min, v_max = finite_range(v)
    s_min, s_max = finite_range(s)
    row: dict[str, Any] = {
        "record_type": record_type,
        "edge": edge,
        "frame_ids": "|".join(frame_ids),
        "pose_count": len(frame_ids),
        "steger_count": int(steger_count),
        "effective_count": int(len(s)),
        "v_min_px": v_min,
        "v_max_px": v_max,
        "s_min": s_min,
        "s_max": s_max,
    }
    for domain_name in ("observed", "safe"):
        stats = domain_stats(s, v, targets[domain_name])
        for key, value in stats.items():
            row[f"{domain_name}_{key}"] = value
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


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def report_text(
    *,
    output_dir: Path,
    data_root: Path,
    c1_model_path: Path,
    c1_sha256: str,
    group_rows: Mapping[str, Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    decision: str,
) -> str:
    lines = [
        "# Corrected Top/Bottom edge support audit for FIT 049–054",
        "",
        f"`EDGE_SUPPORT_CORRECTED = {decision}`",
        "",
        "## Scope and mapping",
        "",
        f"- 只打开 `{data_root / 'fit'}` 下 FIT `049–054` 的 18 张图；没有打开 Validation。",
        "- 使用上一轮相同的 after-mask：PnP 投影完整棋盘物理边界 `X=[-20,220] mm`、`Y=[-20,160] mm`，0 mm inset、无像素腐蚀。",
        "- 保持 Steger、continuity、Frozen Circular Cone 有效性筛选和 Frozen C1 PCA `s` 不变；未重新拟合 Cone/C1。",
        f"- Frozen C1 artifact SHA-256 = `{c1_sha256}`。",
        "",
        "| corrected edge | frame IDs | actual assignment |",
        "|---|---|---|",
        "| Top | 052–054 | v=[30,520], s=[-0.191,-0.127] safety target |",
        "| Bottom | 049–051 | v=[2760,2990], s=[0.181,0.209] safety target |",
        "",
        "## Group-level corrected support",
        "",
        "这里的 joint 统计要求同一个有效点同时落在目标 v 区间和目标 s 区间；conditional span 则要求用于覆盖 v 的点同时满足 s 目标，反之亦然。",
        "",
        "| edge | effective | v range | s range | observed joint n / fraction / status | safety joint n / fraction / status |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for edge in ("Top", "Bottom"):
        row = group_rows[edge]
        lines.append(
            f"| {edge} | {row['effective_count']} | [{fmt(row['v_min_px'],1)}, {fmt(row['v_max_px'],1)}] | "
            f"[{fmt(row['s_min'],6)}, {fmt(row['s_max'],6)}] | "
            f"{row['observed_joint_count']} / {fmt(row['observed_joint_fraction'],3)} / {row['observed_status']} | "
            f"{row['safe_joint_count']} / {fmt(row['safe_joint_fraction'],3)} / {row['safe_status']} |"
        )
    lines.extend(
        [
            "",
            "### Joint conditional spans",
            "",
            "| edge | domain | v span among s-in-domain | s span among v-in-domain | overlap fractions (v, s) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for edge in ("Top", "Bottom"):
        row = group_rows[edge]
        for domain in ("observed", "safe"):
            lines.append(
                f"| {edge} | {domain} | [{fmt(row[f'{domain}_v_given_s_min'],1)}, {fmt(row[f'{domain}_v_given_s_max'],1)}] | "
                f"[{fmt(row[f'{domain}_s_given_v_min'],6)}, {fmt(row[f'{domain}_s_given_v_max'],6)}] | "
                f"({fmt(row[f'{domain}_v_given_s_overlap_fraction'],3)}, {fmt(row[f'{domain}_s_given_v_overlap_fraction'],3)}) |"
            )
    lines.extend(
        [
            "",
            "## Per-frame joint counts",
            "",
            "| frame | corrected edge | effective | observed joint n/status | safety joint n/status |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in frame_rows:
        lines.append(
            f"| {row['frame_id']} | {row['edge']} | {row['effective_count']} | "
            f"{row['observed_joint_count']} / {row['observed_status']} | "
            f"{row['safe_joint_count']} / {row['safe_status']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `EDGE_SUPPORT_CORRECTED = {decision}`。判定使用 corrected group 的同点 joint support，不使用独立 v/s extrema 通过。",
        f"- FULL：同一条件集合下，v 与 s 的 conditional span overlap 均 ≥ {JOINT_FULL_THRESHOLD:.0%}；PARTIAL：存在目标矩形内有效点但未达到该联合覆盖率；NONE：目标矩形内无有效点。",
            "- 本轮没有重新采图、没有读取 Validation、没有拟合或修改 Cone/C1。",
            "",
            "## Artifacts",
            "",
            f"- `corrected_edge_support.csv`: `{output_dir / 'corrected_edge_support.csv'}`",
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
    c1_model, c1_sha256 = frozen.load_frozen_json(c1_model_path)
    _, calibration, reconstruction_params, intrinsics = fixed.load_runtime(args.measurement_config.resolve())
    frozen_model, _ = board.load_frozen_model_checked(args.frozen_provenance.resolve(), args.formal_cone.resolve())
    frozen_calibration = copy.deepcopy(dict(calibration))
    frozen_calibration["laser_model"] = copy.deepcopy(dict(frozen_model))
    targets = target_domains()

    frame_results: dict[str, dict[str, Any]] = {}
    frame_rows: list[dict[str, Any]] = []
    for frame_id in FRAME_IDS:
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
        outer_polygon = mask_audit.projected_outer_boundary(pose, intrinsics)
        full_mask = mask_audit.polygon_mask(shape, outer_polygon)
        result = mask_audit_module_extract(
            laser=laser,
            background=background,
            mask=full_mask,
            pose=pose,
            intrinsics=intrinsics,
            frozen_calibration=frozen_calibration,
            reconstruction_params=reconstruction_params,
            c1_model=c1_model,
        )
        frame_results[frame_id] = result
        frame_row = summarize_points(
            edge=EDGE_BY_FRAME[frame_id],
            frame_ids=[frame_id],
            s=result["s"],
            v=result["v"],
            steger_count=int(result["steger_count"]),
            targets=targets[EDGE_BY_FRAME[frame_id]],
            record_type="frame_summary",
        )
        frame_row["frame_id"] = frame_id
        frame_rows.append(frame_row)

    group_rows: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = list(frame_rows)
    for edge, frame_ids in EDGE_FRAME_IDS.items():
        s = np.concatenate([np.asarray(frame_results[frame_id]["s"], dtype=np.float64) for frame_id in frame_ids])
        v = np.concatenate([np.asarray(frame_results[frame_id]["v"], dtype=np.float64) for frame_id in frame_ids])
        row = summarize_points(
            edge=edge,
            frame_ids=frame_ids,
            s=s,
            v=v,
            steger_count=sum(int(frame_results[frame_id]["steger_count"]) for frame_id in frame_ids),
            targets=targets[edge],
            record_type="edge_union_summary",
        )
        group_rows[edge] = row
        csv_rows.append(row)

    observed_full = all(group_rows[edge]["observed_status"] == "FULL" for edge in ("Top", "Bottom"))
    safety_full = all(group_rows[edge]["safe_status"] == "FULL" for edge in ("Top", "Bottom"))
    safety_present = all(group_rows[edge]["safe_joint_count"] > 0 for edge in ("Top", "Bottom"))
    if safety_full:
        decision = "SUFFICIENT"
    elif all(group_rows[edge]["observed_joint_count"] > 0 for edge in ("Top", "Bottom")) and safety_present:
        decision = "PARTIAL"
    else:
        decision = "INSUFFICIENT"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "corrected_edge_support.csv", csv_rows)
    (output_dir / "report.md").write_text(
        report_text(
            output_dir=output_dir,
            data_root=data_root,
            c1_model_path=c1_model_path,
            c1_sha256=c1_sha256,
            group_rows=group_rows,
            frame_rows=frame_rows,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        {
            "output_dir": str(output_dir),
            "EDGE_SUPPORT_CORRECTED": decision,
            "observed_status": {edge: group_rows[edge]["observed_status"] for edge in ("Top", "Bottom")},
            "safe_status": {edge: group_rows[edge]["safe_status"] for edge in ("Top", "Bottom")},
            "validation_read": False,
        }
    )


def mask_audit_module_extract(**kwargs: Any) -> dict[str, Any]:
    # Keep the exact extraction/effective filtering implementation from the
    # previous after-mask audit; only the frame-to-edge mapping changes here.
    return mask_fix_extract_effective(**kwargs)


def mask_fix_extract_effective(**kwargs: Any) -> dict[str, Any]:
    u, v, _, _ = fixed.triplets.extract_laser_centers(
        kwargs["laser"], kwargs["background"], kwargs["mask"], fixed.EXTRACTION_CONFIG, "vertical"
    )
    uv = np.column_stack([u, v]).astype(np.float64)
    pose = kwargs["pose"]
    intrinsics = kwargs["intrinsics"]
    truth = fixed.coverage.plane_ray_truth(
        u, v, pose.normal, pose.d, intrinsics.camera_matrix, intrinsics.dist_coeffs
    )
    _, model_valid = fixed.lambda_by_input(
        uv, kwargs["frozen_calibration"], kwargs["reconstruction_params"]
    )
    effective = (
        np.asarray(model_valid, dtype=bool)
        & np.asarray(truth["valid"], dtype=bool)
        & np.isfinite(np.asarray(truth["points"], dtype=np.float64)[:, 2])
    )
    uv_effective = uv[effective]
    normalized = cv2.undistortPoints(
        uv_effective.reshape(-1, 1, 2), intrinsics.camera_matrix, intrinsics.dist_coeffs
    ).reshape(-1, 2)
    s = base.pca_s_values(normalized, kwargs["c1_model"])
    return {
        "steger_count": int(len(uv)),
        "effective_count": int(np.count_nonzero(effective)),
        "v": uv_effective[:, 1],
        "s": s,
    }


if __name__ == "__main__":
    run(parse_args())
