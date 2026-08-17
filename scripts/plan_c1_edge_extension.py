#!/usr/bin/env python3
"""Plan separated Top/Bottom FIT and extreme-edge Validation extensions.

This is a planning-only audit.  It consumes the previous support-comparison
summary and never trains C1 or uses standard-object points as training data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_SUPPORT_CSV = (
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
    / "edge_extension_plan"
)
CURRENT_FIT_IDS = [f"{i:03d}" for i in range(1, 19)] + [f"{i:03d}" for i in range(25, 37)]
CURRENT_VALIDATION_IDS = [f"{i:03d}" for i in range(19, 25)] + [f"{i:03d}" for i in range(37, 41)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: Any) -> float:
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    return f"{number(value):.{digits}f}"


def outward_floor(value: float, grid: float) -> float:
    return math.floor(value / grid + 1.0e-12) * grid


def outward_ceil(value: float, grid: float) -> float:
    return math.ceil(value / grid - 1.0e-12) * grid


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def position_record(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "position": row["position"],
        "region": row["region"],
        "directory": row["directory"],
        "v_range_px": [number(row["v_min_px"]), number(row["v_max_px"])],
        "s_range": [number(row["s_min"]), number(row["s_max"])],
        "v_median_px": number(row["v_median_px"]),
        "s_median": number(row["s_median"]),
        "v_extrapolated_count": int(row["v_extrapolated_count"]),
        "v_extrapolated_fraction": number(row["v_extrapolated_fraction"]),
        "s_extrapolated_count": int(row["s_extrapolated_count"]),
        "s_extrapolated_fraction": number(row["s_extrapolated_fraction"]),
        "v_outside_distance_max_px": number(row["v_outside_distance_max_px"]),
        "s_outside_distance_max": number(row["s_outside_distance_max"]),
        "coverage_status": row["coverage_status"],
    }


def union_domain(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    return {
        "v_px": [
            min(float(row["v_range_px"][0]) for row in rows),
            max(float(row["v_range_px"][1]) for row in rows),
        ],
        "s": [
            min(float(row["s_range"][0]) for row in rows),
            max(float(row["s_range"][1]) for row in rows),
        ],
    }


def plan_pose(
    pose_id: str,
    edge: str,
    role: str,
    reference: Mapping[str, Any],
    target_domain: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        "pose_id": pose_id,
        "edge": edge,
        "role": role,
        "reference_standard_position": f"{reference['dataset']} {reference['position']}",
        "target_center_v_px": reference["v_median_px"],
        "target_center_s": reference["s_median"],
        "edge_target_domain_v_px": list(target_domain["v_px"]),
        "edge_target_domain_s": list(target_domain["s"]),
        "acquisition_rule": "调节棋盘姿态，使 laser-center/height 对应的实际 v/s 落入该 edge target domain；最终以采后实际点覆盖验收。",
    }


def task_records(dataset_id: str, split: str, pose_ids: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    folder = "fit" if split == "fit" else "validation"
    for pose_id in pose_ids:
        for index, role in enumerate(("chess", "nolaser", "laser"), start=1):
            records.append(
                {
                    "task_id": f"{split}_{pose_id}_{index:02d}_{role}",
                    "pose_id": pose_id,
                    "role": role,
                    "split": split,
                    "frames": 1,
                    "relative_path": f"{folder}/{role} {pose_id}.tif",
                    "dataset_id": dataset_id,
                }
            )
    return records


def build_plan(support_csv: Path) -> dict[str, Any]:
    rows = read_rows(support_csv)
    height_rows = [
        position_record(row)
        for row in rows
        if row.get("record_type") == "standard_position_summary"
        and row.get("point_source") == "height"
    ]
    center_rows = [
        position_record(row)
        for row in rows
        if row.get("record_type") == "standard_position_summary"
        and row.get("point_source") == "laser_center"
    ]
    if len(height_rows) != 12 or len(center_rows) != 12:
        raise RuntimeError(f"Expected 12 height and 12 laser-center summaries, got {len(height_rows)} / {len(center_rows)}")

    fit_summary = next(row for row in rows if row.get("record_type") == "fit_summary")
    current_v_domain = [number(fit_summary["fit_v_min_px"]), number(fit_summary["fit_v_max_px"])]
    current_s_domain = [number(fit_summary["frozen_s_domain_min"]), number(fit_summary["frozen_s_domain_max"])]
    edge_rows = {
        edge: [row for row in height_rows if row["region"] == edge]
        for edge in ("top", "bottom")
    }
    required = {edge: union_domain(edge_rows[edge]) for edge in edge_rows}
    center_domain = union_domain(center_rows)

    safety_margin: dict[str, Any] = {
        "method": "max(10% of observed edge span, absolute acquisition guard)",
        "relative_span_fraction": 0.10,
        "absolute_v_guard_px": 50.0,
        "absolute_s_guard": 0.005,
        "rounding": "v rounded outward to 10 px; s rounded outward to 0.001",
        "basis": "height subset defines target standard-object usage; laser-center union is reported separately and is not used as the target-position domain",
    }
    recommended: dict[str, Any] = {}
    for edge, domain in required.items():
        v_span = domain["v_px"][1] - domain["v_px"][0]
        s_span = domain["s"][1] - domain["s"][0]
        v_margin = max(safety_margin["absolute_v_guard_px"], 0.10 * v_span)
        s_margin = max(safety_margin["absolute_s_guard"], 0.10 * s_span)
        exact_v = [domain["v_px"][0] - v_margin, domain["v_px"][1] + v_margin]
        exact_s = [domain["s"][0] - s_margin, domain["s"][1] + s_margin]
        rounded_v = [max(0.0, outward_floor(exact_v[0], 10.0)), min(3000.0, outward_ceil(exact_v[1], 10.0))]
        rounded_s = [outward_floor(exact_s[0], 0.001), outward_ceil(exact_s[1], 0.001)]
        recommended[edge] = {
            "observed_required_domain": domain,
            "observed_v_span_px": v_span,
            "observed_s_span": s_span,
            "v_margin_px": v_margin,
            "s_margin": s_margin,
            "exact_domain_with_margin": {"v_px": exact_v, "s": exact_s},
            "recommended_rounded_domain": {"v_px": rounded_v, "s": rounded_s},
        }

    def find(dataset: str, position: str) -> dict[str, Any]:
        return next(row for row in height_rows if row["dataset"] == dataset and row["position"] == position)

    top_outer = find("20mm", "P01_v125.0")
    top_inner = find("50mm", "P01_v429.5")
    bottom_inner = find("50mm", "P04_v2846.5")
    bottom_outer = find("20mm", "P08_v2905.0")
    fit_pose_ids = ["041", "042", "043", "044"]
    validation_pose_ids = ["045", "046", "047", "048"]
    fit_root = str(CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "fit_edge_extension_v2")
    validation_root = str(CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "laser_plane" / "validation_edge_holdout_v2")
    fit_poses = [
        plan_pose("041", "Top", "outer_guard", top_outer, recommended["top"]["recommended_rounded_domain"]),
        plan_pose("042", "Top", "inner_bridge", top_inner, recommended["top"]["recommended_rounded_domain"]),
        plan_pose("043", "Bottom", "inner_bridge", bottom_inner, recommended["bottom"]["recommended_rounded_domain"]),
        plan_pose("044", "Bottom", "outer_guard", bottom_outer, recommended["bottom"]["recommended_rounded_domain"]),
    ]
    validation_poses = [
        plan_pose("045", "Top", "outer_edge_holdout", top_outer, recommended["top"]["recommended_rounded_domain"]),
        plan_pose("046", "Top", "inner_edge_holdout", top_inner, recommended["top"]["recommended_rounded_domain"]),
        plan_pose("047", "Bottom", "inner_edge_holdout", bottom_inner, recommended["bottom"]["recommended_rounded_domain"]),
        plan_pose("048", "Bottom", "outer_edge_holdout", bottom_outer, recommended["bottom"]["recommended_rounded_domain"]),
    ]

    return {
        "schema_version": 1,
        "plan_id": "c1_edge_extension_v1",
        "status": "planning_only",
        "source": {
            "support_csv": str(support_csv),
            "support_csv_sha256": sha256_file(support_csv),
            "standard_usage_basis": "height subset for the 20 mm / 50 mm target positions",
            "laser_center_domain_reported": center_domain,
            "validation_data_read": False,
            "standard_objects_used_for_training": False,
            "c1_refit": False,
        },
        "current_coverage": {
            "fit_ids": CURRENT_FIT_IDS,
            "validation_ids": CURRENT_VALIDATION_IDS,
            "fit_point_count": int(fit_summary["valid_point_count"]),
            "fit_v_support_px": current_v_domain,
            "frozen_s_support": current_s_domain,
            "frame_027_retained": True,
            "coverage_status": "INSUFFICIENT",
        },
        "standard_usage": {
            "height_position_summaries": height_rows,
            "edge_union": required,
            "laser_center_union_not_primary_target": center_domain,
        },
        "safety_margin": safety_margin,
        "recommended_edge_domains": recommended,
        "fit_extension": {
            "dataset_id": "fit_edge_extension_v2",
            "split": "fit",
            "root": fit_root,
            "pose_ids": fit_pose_ids,
            "minimum_pose_count_total": 4,
            "minimum_pose_count_by_edge": {"Top": 2, "Bottom": 2},
            "reason_for_two_per_edge": "one outer guard plus one inner bridge; a single pose would make the new edge support frame-specific",
            "poses": fit_poses,
            "tasks": task_records("fit_edge_extension_v2", "fit", fit_pose_ids),
            "directory_tree": [
                f"{fit_root}/dataset_manifest.yaml",
                f"{fit_root}/fit/chess 041.tif ... fit/chess 044.tif",
                f"{fit_root}/fit/nolaser 041.tif ... fit/nolaser 044.tif",
                f"{fit_root}/fit/laser 041.tif ... fit/laser 044.tif",
            ],
        },
        "validation_extension": {
            "dataset_id": "validation_edge_holdout_v2",
            "split": "validation",
            "root": validation_root,
            "pose_ids": validation_pose_ids,
            "minimum_pose_count_total": 2,
            "recommended_pose_count_total": 4,
            "recommended_pose_count_by_edge": {"Top": 2, "Bottom": 2},
            "reason_for_four_recommended": "independently checks both outer and inner edge clusters seen in 20 mm / 50 mm standards",
            "poses": validation_poses,
            "tasks": task_records("validation_edge_holdout_v2", "validation", validation_pose_ids),
            "directory_tree": [
                f"{validation_root}/dataset_manifest.yaml",
                f"{validation_root}/validation/chess 045.tif ... validation/chess 048.tif",
                f"{validation_root}/validation/nolaser 045.tif ... validation/nolaser 048.tif",
                f"{validation_root}/validation/laser 045.tif ... validation/laser 048.tif",
            ],
        },
        "strict_separation": {
            "fit_pose_ids": fit_pose_ids,
            "validation_pose_ids": validation_pose_ids,
            "pose_id_intersection": [],
            "raw_file_reuse": False,
            "acquisition_session_reuse": False,
            "checksum_intersection_allowed": False,
            "validation_manifest_split": "validation",
            "validation_manifest_dataset_id": "validation_edge_holdout_v2",
            "validation_must_not_enter_c1_fit_csv": True,
            "spatial_domain_note": "FIT and Validation intentionally target the same required edge domain; separation is by pose/session/raw files. Disjoint v/s ranges would turn Validation into an extrapolation test rather than independent in-domain generalization.",
        },
        "acceptance_gates_after_acquisition": {
            "fit": [
                "actual FIT union covers each recommended rounded Top/Bottom v and s domain",
                "each edge has at least two distinct contributing FIT pose_ids",
                "all three tasks per pose pass chessboard/laser quality checks",
                "new FIT is the only new data allowed into a future C1 refit; 045–048 remain excluded",
            ],
            "validation": [
                "045–048 are acquired independently and pass quality checks",
                "all Validation raw files and hashes are disjoint from FIT",
                "Validation is read only after the future model is frozen, once, with no parameter tuning",
                "report actual v/s support and extrapolation counts before any accuracy metric is interpreted",
            ],
        },
        "number_conflict_check": {
            "new_fit_ids": fit_pose_ids,
            "new_validation_ids": validation_pose_ids,
            "overlap_with_current_ids": sorted(set(fit_pose_ids + validation_pose_ids) & set(CURRENT_FIT_IDS + CURRENT_VALIDATION_IDS)),
        },
    }


def report_text(plan: Mapping[str, Any]) -> str:
    current = plan["current_coverage"]
    standard = plan["standard_usage"]
    margin = plan["safety_margin"]
    domains = plan["recommended_edge_domains"]
    fit = plan["fit_extension"]
    validation = plan["validation_extension"]
    lines = [
        "# C1_4k Top/Bottom edge-extension coverage plan",
        "",
        "`EDGE_EXTENSION_PLAN = READY`",
        "",
        "## Scope",
        "",
        "- This is a planning-only result; C1 is not refit and no model parameter is changed.",
        "- The 20 mm / 50 mm standard-object points are used only to define the real operating support domain, never as C1 training points.",
        f"- Source support CSV: `{plan['source']['support_csv']}`; SHA-256 = `{plan['source']['support_csv_sha256']}`.",
        f"- Current FIT: `{current['fit_ids'][0]}–{current['fit_ids'][17]}`, `{current['fit_ids'][18]}–{current['fit_ids'][-1]}`; current Validation: `019–024`, `037–040`; frame 027 retained = `{str(current['frame_027_retained']).lower()}`.",
        "- Existing Validation raw data was not read in this planning run.",
        "",
        "## Real standard-object operating range",
        "",
        "The primary range is the height subset that defines each accepted standard-object position. The full laser-center cloud is reported in JSON for traceability but is not used as the target-position domain because it spans the complete visible scan line.",
        "",
        "| edge | observed v range / px | observed s range | positions represented |",
        "|---|---:|---:|---|",
    ]
    for edge in ("top", "bottom"):
        d = domains[edge]
        positions = [f"{r['dataset']} {r['position']}" for r in standard["height_position_summaries"] if r["region"] == edge]
        lines.append(
            f"| {edge.title()} | [{fmt(d['observed_required_domain']['v_px'][0], 1)}, {fmt(d['observed_required_domain']['v_px'][1], 1)}] | "
            f"[{fmt(d['observed_required_domain']['s'][0], 6)}, {fmt(d['observed_required_domain']['s'][1], 6)}] | {', '.join(positions)} |"
        )
    lines.extend(
        [
            "",
            f"- Current FIT support is v=[{fmt(current['fit_v_support_px'][0], 3)}, {fmt(current['fit_v_support_px'][1], 3)}] px and s=[{fmt(current['frozen_s_support'][0], 9)}, {fmt(current['frozen_s_support'][1], 9)}].",
            f"- Observed full laser-center union, for reference only: v=[{fmt(standard['laser_center_union_not_primary_target']['v_px'][0], 1)}, {fmt(standard['laser_center_union_not_primary_target']['v_px'][1], 1)}] px and s=[{fmt(standard['laser_center_union_not_primary_target']['s'][0], 6)}, {fmt(standard['laser_center_union_not_primary_target']['s'][1], 6)}].",
            "",
            "## Safety margin",
            "",
            f"- Margin rule: max(10% of each edge span, {fmt(margin['absolute_v_guard_px'], 0)} px v guard, {fmt(margin['absolute_s_guard'], 3)} s guard). Final targets are rounded outward to 10 px / 0.001 s.",
            "",
            "| edge | exact domain with margin (v px) | exact domain with margin (s) | recommended rounded target (v px) | recommended rounded target (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for edge in ("top", "bottom"):
        d = domains[edge]
        exact = d["exact_domain_with_margin"]
        rounded = d["recommended_rounded_domain"]
        lines.append(
            f"| {edge.title()} | [{fmt(exact['v_px'][0], 1)}, {fmt(exact['v_px'][1], 1)}] | [{fmt(exact['s'][0], 6)}, {fmt(exact['s'][1], 6)}] | "
            f"[{fmt(rounded['v_px'][0], 0)}, {fmt(rounded['v_px'][1], 0)}] | [{fmt(rounded['s'][0], 3)}, {fmt(rounded['s'][1], 3)}] |"
        )
    lines.extend(
        [
            "",
            "## New FIT extension",
            "",
            f"- Dataset: `{fit['dataset_id']}`; split=`{fit['split']}`; pose IDs `{fit['pose_ids'][0]}–{fit['pose_ids'][-1]}`.",
            f"- Minimum: **{fit['minimum_pose_count_total']} poses total**, Top={fit['minimum_pose_count_by_edge']['Top']}, Bottom={fit['minimum_pose_count_by_edge']['Bottom']}.",
            "- Top: 041 outer guard near 20 mm Top (v≈125, s≈−0.180), 042 inner bridge near 50 mm Top (v≈430, s≈−0.138).",
            "- Bottom: 043 inner bridge near 50 mm Bottom (v≈2847, s≈+0.191), 044 outer guard near 20 mm Bottom (v≈2905, s≈+0.199).",
            "",
            "| pose | edge | role | reference center v / px | reference center s |",
            "|---|---|---|---:|---:|",
        ]
    )
    for pose in fit["poses"]:
        lines.append(f"| {pose['pose_id']} | {pose['edge']} | {pose['role']} | {fmt(pose['target_center_v_px'], 1)} | {fmt(pose['target_center_s'], 6)} |")
    lines.extend(
        [
            "",
            "## New extreme-edge Validation",
            "",
            f"- Dataset: `{validation['dataset_id']}`; split=`{validation['split']}`; pose IDs `{validation['pose_ids'][0]}–{validation['pose_ids'][-1]}`.",
            f"- Bare minimum: {validation['minimum_pose_count_total']} poses (one outer Top + one outer Bottom). Recommended: **{validation['recommended_pose_count_total']} poses**, Top=2, Bottom=2.",
            "- 045/046 independently repeat the outer/inner Top target centers; 047/048 independently repeat the inner/outer Bottom target centers.",
            "",
            "| pose | edge | role | reference center v / px | reference center s |",
            "|---|---|---|---:|---:|",
        ]
    )
    for pose in validation["poses"]:
        lines.append(f"| {pose['pose_id']} | {pose['edge']} | {pose['role']} | {fmt(pose['target_center_v_px'], 1)} | {fmt(pose['target_center_s'], 6)} |")
    lines.extend(
        [
            "",
            "## Strict FIT / Validation separation",
            "",
            "- FIT IDs 041–044 and Validation IDs 045–048 are disjoint and outside the current 001–040 registry.",
            "- Do not reuse a raw image, camera frame, board pose, acquisition session, or file hash between the two sets.",
            "- New Validation manifest must explicitly use `dataset_id: validation_edge_holdout_v2` and `split: validation`; do not copy the old `validation_edge_holdout` metadata inconsistency where the split remained `fit`.",
            "- FIT and Validation intentionally target the same physical edge domain, but use independent poses/sessions. This is the correct separation for in-domain independent generalization; forcing disjoint v/s ranges would make Validation an extrapolation test instead.",
            "- 045–048 must never enter the C1 fit CSV or a future C1 refit. Freeze the future model before reading them once.",
            "",
            "## Directory structure",
            "",
            "```text",
            f"{fit['root']}/",
            "  dataset_manifest.yaml          # dataset_id=fit_edge_extension_v2, split=fit",
            "  fit/",
            "    chess 041.tif ... chess 044.tif",
            "    nolaser 041.tif ... nolaser 044.tif",
            "    laser 041.tif ... laser 044.tif",
            f"{validation['root']}/",
            "  dataset_manifest.yaml          # dataset_id=validation_edge_holdout_v2, split=validation",
            "  validation/",
            "    chess 045.tif ... chess 048.tif",
            "    nolaser 045.tif ... nolaser 048.tif",
            "    laser 045.tif ... laser 048.tif",
            "```",
            "",
            "## Post-acquisition acceptance gates",
            "",
            "1. FIT 041–044 的实际 union 必须覆盖 Top/Bottom 推荐 rounded v/s domain，且每个 edge 至少由两个不同 pose 提供支持。",
            "2. Validation 045–048 必须独立采集、质量通过、与 FIT 文件/hash 完全不交集。",
            "3. 先冻结未来 C1，再一次性读取新 Validation；不得用 Validation 调 knots、penalty、PCA 或边界策略。",
            "",
            "## Artifacts",
            "",
            f"- `edge_extension_plan.json`: `{plan['output_dir']}`",
            f"- `report.md`: `{plan['output_dir']}`",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    support_csv = args.support_csv.resolve()
    output_dir = args.output_dir.resolve()
    if not support_csv.is_file():
        raise FileNotFoundError(support_csv)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty; use --overwrite: {output_dir}")
    plan = build_plan(support_csv)
    plan["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "edge_extension_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(report_text(plan), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "fit_pose_ids": plan["fit_extension"]["pose_ids"],
        "validation_pose_ids": plan["validation_extension"]["pose_ids"],
        "C1_SUPPORT_COVERAGE": plan["current_coverage"]["coverage_status"],
        "validation_data_read": plan["source"]["validation_data_read"],
        "standard_objects_used_for_training": plan["source"]["standard_objects_used_for_training"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())
