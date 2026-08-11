#!/usr/bin/env python3
"""Stage 3-3：真实数据离线验证 production search-region shadow proposal。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from search_region_expansion_characterization import (
    DEFAULT_GIT_REF,
    DATA_ROOT,
    CaseSpec,
    FrameLevel,
    git_tiff_paths,
    load_realtime_steger,
    pair_samples,
    read_git_image,
    select_frame_level,
)


STABLE_REFERENCE_EXPANSION_EACH_SIDE_PX = 48
MATCH_P95_MAX_PX = 0.01
MATCH_VALID_FRACTION_DELTA_MAX = 0.01
MATCH_SAME_FLOOR_FRACTION_MIN = 0.99
ALTERNATE_SHIFT_P95_MIN_PX = 0.5
ALTERNATE_FRACTION_MIN = 0.01
REGION_BASICALLY_SAME_TOLERANCE_PX = 1


CASES = (
    CaseSpec(
        "B05_A10_boundary_sensitive_h10",
        "B05_A10",
        "multiheight",
        1857,
        1899,
        "boundary-sensitive positive",
    ),
    CaseSpec(
        "B05_A10_h1_truncation",
        "B05_A10",
        "multiheight",
        1800,
        1833,
        "known truncation positive",
    ),
    CaseSpec(
        "B05_A10_full_scanlines",
        "B05_A10",
        "multiheight",
        None,
        None,
        "full-frame alternate-ridge audit",
    ),
    CaseSpec(
        "B12p5_A10_normal_reference",
        "B12p5_A10",
        "reference",
        None,
        None,
        "normal negative control",
    ),
)


@dataclass(frozen=True, slots=True)
class FrameResolverRecord:
    case_id: str
    source_file: str
    current_search_region: tuple[int, int] | None
    proposed_search_region: tuple[int, int] | None
    would_expand: bool
    reason: str
    outside_active_intervals: tuple[tuple[int, int], ...]
    outside_peak_intervals: tuple[tuple[int, int], ...]
    outside_active_position_count: int
    outside_peak_position_count: int
    outside_peak_max_intensity_dn: float | None
    active_intervals: tuple[tuple[int, int], ...]
    seed_interval: tuple[int, int] | None
    seed: int | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-ref", default=DEFAULT_GIT_REF)
    parser.add_argument("--repo", type=Path, default=root)
    parser.add_argument(
        "--calibration-src",
        type=Path,
        default=root.parent / "calibration" / "src",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments" / "production_search_region_replay",
    )
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames 必须为正数")
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    realtime = load_realtime_steger(args.calibration_src)
    options = realtime.load_steger_options()
    if options["scan_axis"] != "column":
        raise ValueError("Stage 3-3 真实数据冻结配置必须是 scan_axis=column")

    grouped: dict[tuple[str, str], list[CaseSpec]] = {}
    for case in CASES:
        grouped.setdefault((case.dataset_id, case.series), []).append(case)
    samples: dict[str, dict[str, list[FrameLevel]]] = {
        case.case_id: {mode: [] for mode in ("current", "proposed", "stable_reference")}
        for case in CASES
    }
    resolver_records: list[FrameResolverRecord] = []
    source_files: dict[str, list[str]] = {case.case_id: [] for case in CASES}

    for (dataset_id, series), cases in grouped.items():
        prefix = f"{DATA_ROOT}/{dataset_id}/images/{series}"
        paths = git_tiff_paths(repo, args.git_ref, prefix)
        if args.max_frames is not None:
            paths = paths[: args.max_frames]
        if not paths:
            raise FileNotFoundError(f"Git ref 中没有 TIFF：{args.git_ref}:{prefix}")
        for path in paths:
            image = read_git_image(repo, args.git_ref, path)
            current = realtime.extract_steger(image, options)
            summary = current.detector_summary
            if summary is None:
                raise RuntimeError("正式 extraction 没有 DetectorSummary")
            current_region = _region_from_metadata(current.metadata)
            resolution = realtime.resolve_production_search_region(
                summary,
                current_region,
            )
            proposed_region = resolution.proposed_search_region
            proposed = realtime.extract_steger(
                image,
                options,
                search_region=proposed_region,
            )
            stable_region = realtime.LaserSearchRegion(
                current_region.start_px - STABLE_REFERENCE_EXPANSION_EACH_SIDE_PX,
                current_region.end_px + STABLE_REFERENCE_EXPANSION_EACH_SIDE_PX,
                source="stage2b_stable_plus_48px_each_side",
            )
            stable = realtime.extract_steger(
                image,
                options,
                search_region=stable_region,
            )
            outside_active = _intervals(
                current.metadata["outside_region_active_intervals_px"]
            )
            outside_peak = _intervals(
                current.metadata["outside_region_peak_intervals_px"]
            )
            for case in cases:
                source_files[case.case_id].append(path)
                samples[case.case_id]["current"].append(
                    select_frame_level(current, case)
                )
                samples[case.case_id]["proposed"].append(
                    select_frame_level(proposed, case)
                )
                samples[case.case_id]["stable_reference"].append(
                    select_frame_level(stable, case)
                )
                resolver_records.append(
                    FrameResolverRecord(
                        case_id=case.case_id,
                        source_file=path,
                        current_search_region=(
                            current_region.start_px,
                            current_region.end_px,
                        ),
                        proposed_search_region=(
                            (proposed_region.start_px, proposed_region.end_px)
                            if proposed_region is not None
                            else None
                        ),
                        would_expand=resolution.would_expand,
                        reason=resolution.reason,
                        outside_active_intervals=outside_active,
                        outside_peak_intervals=outside_peak,
                        outside_active_position_count=int(
                            current.metadata["outside_region_active_position_count"]
                        ),
                        outside_peak_position_count=int(
                            current.metadata["outside_region_peak_position_count"]
                        ),
                        outside_peak_max_intensity_dn=_optional_float(
                            current.metadata["outside_region_peak_max_intensity_dn"]
                        ),
                        active_intervals=summary.active_intervals,
                        seed_interval=summary.seed_active_interval,
                        seed=summary.seed,
                    )
                )

    metric_rows = [
        summarize_mode(
            case,
            mode,
            samples[case.case_id][mode],
            samples[case.case_id]["stable_reference"],
        )
        for case in CASES
        for mode in ("current", "proposed", "stable_reference")
    ]
    comparison_rows = [
        summarize_case_comparison(
            case,
            samples[case.case_id],
            [record for record in resolver_records if record.case_id == case.case_id],
        )
        for case in CASES
    ]
    outcomes = infer_outcomes(metric_rows, comparison_rows)

    write_csv(
        output / "resolver_frames.csv",
        [_resolver_record_row(record) for record in resolver_records],
    )
    write_csv(output / "extraction_metrics.csv", metric_rows)
    write_csv(output / "case_comparisons.csv", comparison_rows)
    summary_payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_ref": args.git_ref,
        "repo": str(repo),
        "calibration_src": str(args.calibration_src.expanduser().resolve()),
        "frozen_steger_options": options,
        "stable_reference": {
            "definition": "current region expanded symmetrically by 48 px each side",
            "expansion_each_side_px": STABLE_REFERENCE_EXPANSION_EACH_SIDE_PX,
        },
        "matching_criteria": {
            "center_shift_p95_max_px": MATCH_P95_MAX_PX,
            "valid_fraction_delta_max": MATCH_VALID_FRACTION_DELTA_MAX,
            "same_floor_candidate_fraction_min": MATCH_SAME_FLOOR_FRACTION_MIN,
        },
        "alternate_ridge_criteria": {
            "center_shift_p95_min_px": ALTERNATE_SHIFT_P95_MIN_PX,
            "alternate_or_proposed_only_fraction_min": ALTERNATE_FRACTION_MIN,
        },
        "cases": [asdict(case) for case in CASES],
        "source_files": source_files,
        "resolver_frames": [asdict(record) for record in resolver_records],
        "extraction_metrics": metric_rows,
        "case_comparisons": comparison_rows,
        **outcomes,
        "behavior_changed": False,
        "formal_steger_result_changed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outcomes, ensure_ascii=False))
    return 0


def _region_from_metadata(metadata: dict[str, Any]) -> Any:
    import realtime_steger as realtime

    start = metadata.get("final_search_region_start_px")
    end = metadata.get("final_search_region_end_px")
    if start is None or end is None:
        raise RuntimeError("正式 extraction 没有 current search region")
    return realtime.LaserSearchRegion(
        int(round(float(start))),
        int(round(float(end))),
        "stage3_3_current_formal_region",
    )


def summarize_mode(
    case: CaseSpec,
    mode: str,
    frames: list[FrameLevel],
    stable_frames: list[FrameLevel],
) -> dict[str, Any]:
    opportunity_count = sum(frame.valid.size for frame in frames)
    valid_count = sum(int(np.count_nonzero(frame.valid)) for frame in frames)
    responses = _concat(
        frame.response[frame.valid & np.isfinite(frame.response)] for frame in frames
    )
    clearances = _concat(
        np.minimum(
            frame.center_px[frame.valid & np.isfinite(frame.center_px)]
            - frame.region_start_px,
            frame.region_end_px
            - frame.center_px[frame.valid & np.isfinite(frame.center_px)],
        )
        for frame in frames
    )
    paired = pair_samples(frames, stable_frames)
    starts = np.asarray([frame.region_start_px for frame in frames], dtype=np.float64)
    ends = np.asarray([frame.region_end_px for frame in frames], dtype=np.float64)
    return {
        "case_id": case.case_id,
        "role": case.role,
        "mode": mode,
        "frame_count": len(frames),
        "opportunity_count": opportunity_count,
        "valid_count": valid_count,
        "valid_fraction": valid_count / opportunity_count if opportunity_count else 0.0,
        "region_start_p50_px": _percentile(starts, 50),
        "region_end_p50_px": _percentile(ends, 50),
        "region_size_p50_px": _percentile(ends - starts, 50),
        "boundary_clearance_p05_px": _percentile(clearances, 5),
        "response_mean": _mean(responses),
        "response_p50": _percentile(responses, 50),
        "response_p95": _percentile(responses, 95),
        "paired_vs_stable_count": paired["paired_count"],
        "center_shift_vs_stable_p50_px": paired["shift_p50_px"],
        "center_shift_vs_stable_p95_px": paired["shift_p95_px"],
        "center_shift_vs_stable_max_px": paired["shift_max_px"],
        "same_floor_candidate_vs_stable_fraction": paired["same_floor_fraction"],
    }


def summarize_case_comparison(
    case: CaseSpec,
    modes: dict[str, list[FrameLevel]],
    records: list[FrameResolverRecord],
) -> dict[str, Any]:
    current = modes["current"]
    proposed = modes["proposed"]
    stable = modes["stable_reference"]
    proposed_vs_stable = pair_samples(proposed, stable)
    proposed_vs_current = pair_samples(current, proposed)
    opportunity_count = sum(frame.valid.size for frame in current)
    current_valid = sum(int(np.count_nonzero(frame.valid)) for frame in current)
    proposed_valid = sum(int(np.count_nonzero(frame.valid)) for frame in proposed)
    stable_valid = sum(int(np.count_nonzero(frame.valid)) for frame in stable)
    proposed_only_stable_count = _one_sided_valid_count(proposed, stable)
    stable_only_proposed_count = _one_sided_valid_count(stable, proposed)
    alternate_fraction = (
        1.0 - float(proposed_vs_stable["same_floor_fraction"])
        if proposed_vs_stable["same_floor_fraction"] is not None
        else 0.0
    )
    proposed_only_fraction = (
        proposed_only_stable_count / opportunity_count if opportunity_count else 0.0
    )
    shift_p95 = proposed_vs_stable["shift_p95_px"]
    alternate_detected = bool(
        (
            shift_p95 is not None
            and shift_p95 >= ALTERNATE_SHIFT_P95_MIN_PX
            and alternate_fraction >= ALTERNATE_FRACTION_MIN
        )
        or proposed_only_fraction >= ALTERNATE_FRACTION_MIN
    )
    reason_counts = Counter(record.reason for record in records)
    current_regions = Counter(record.current_search_region for record in records)
    proposed_regions = Counter(record.proposed_search_region for record in records)
    active_intervals = Counter(record.active_intervals for record in records)
    seed_intervals = Counter(record.seed_interval for record in records)
    outside_active_intervals = Counter(
        record.outside_active_intervals for record in records
    )
    outside_peak_intervals = Counter(
        record.outside_peak_intervals for record in records
    )
    return {
        "case_id": case.case_id,
        "role": case.role,
        "frame_count": len(records),
        "would_expand_fraction": float(np.mean([record.would_expand for record in records])),
        "reason_counts": json.dumps(reason_counts, sort_keys=True),
        "current_region_counts": json.dumps(
            {_region_text(key): value for key, value in current_regions.items()},
            sort_keys=True,
        ),
        "proposed_region_counts": json.dumps(
            {_region_text(key): value for key, value in proposed_regions.items()},
            sort_keys=True,
        ),
        "active_interval_counts": _counter_json(active_intervals),
        "seed_interval_counts": _counter_json(seed_intervals),
        "outside_active_interval_counts": _counter_json(outside_active_intervals),
        "outside_peak_interval_counts": _counter_json(outside_peak_intervals),
        "representative_active_intervals": json.dumps(records[0].active_intervals),
        "representative_seed_interval": json.dumps(records[0].seed_interval),
        "representative_outside_active_intervals": json.dumps(
            records[0].outside_active_intervals
        ),
        "representative_outside_peak_intervals": json.dumps(
            records[0].outside_peak_intervals
        ),
        "current_valid_fraction": current_valid / opportunity_count,
        "proposed_valid_fraction": proposed_valid / opportunity_count,
        "stable_valid_fraction": stable_valid / opportunity_count,
        "proposed_valid_fraction_delta_vs_stable": (
            proposed_valid - stable_valid
        ) / opportunity_count,
        "proposed_vs_current_center_shift_p95_px": proposed_vs_current["shift_p95_px"],
        "proposed_vs_stable_center_shift_p95_px": shift_p95,
        "proposed_vs_stable_center_shift_max_px": proposed_vs_stable["shift_max_px"],
        "same_floor_candidate_vs_stable_fraction": proposed_vs_stable["same_floor_fraction"],
        "alternate_candidate_fraction_vs_stable": alternate_fraction,
        "proposed_only_vs_stable_count": proposed_only_stable_count,
        "proposed_only_vs_stable_fraction": proposed_only_fraction,
        "stable_only_vs_proposed_count": stable_only_proposed_count,
        "alternate_ridge_detected": alternate_detected,
    }


def infer_outcomes(
    metric_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, bool]:
    metrics = {(row["case_id"], row["mode"]): row for row in metric_rows}
    comparisons = {row["case_id"]: row for row in comparison_rows}

    def matches_stable(case_id: str) -> bool:
        proposed = metrics[(case_id, "proposed")]
        stable = metrics[(case_id, "stable_reference")]
        shift = proposed["center_shift_vs_stable_p95_px"]
        same = proposed["same_floor_candidate_vs_stable_fraction"]
        return bool(
            abs(proposed["valid_fraction"] - stable["valid_fraction"])
            <= MATCH_VALID_FRACTION_DELTA_MAX
            and shift is not None
            and shift <= MATCH_P95_MAX_PX
            and same is not None
            and same >= MATCH_SAME_FLOOR_FRACTION_MIN
        )

    problem_ids = (
        "B05_A10_boundary_sensitive_h10",
        "B05_A10_h1_truncation",
    )
    problem_cases_recovered = all(
        comparisons[case_id]["would_expand_fraction"] == 1.0
        and matches_stable(case_id)
        for case_id in problem_ids
    )

    normal_id = "B12p5_A10_normal_reference"
    normal_comparison = comparisons[normal_id]
    normal_expansion_avoided = _region_counts_basically_same(
        normal_comparison["current_region_counts"],
        normal_comparison["proposed_region_counts"],
    )
    normal_shift = normal_comparison["proposed_vs_current_center_shift_p95_px"]
    normal_cases_unchanged = bool(
        matches_stable(normal_id)
        and normal_shift is not None
        and normal_shift <= MATCH_P95_MAX_PX
        and abs(
            normal_comparison["proposed_valid_fraction"]
            - normal_comparison["current_valid_fraction"]
        )
        <= MATCH_VALID_FRACTION_DELTA_MAX
    )
    alternate_ridge_detected = any(
        bool(row["alternate_ridge_detected"]) for row in comparison_rows
    )
    return {
        "shadow_strategy_validated": bool(
            problem_cases_recovered
            and normal_cases_unchanged
            and normal_expansion_avoided
            and not alternate_ridge_detected
        ),
        "problem_cases_recovered": problem_cases_recovered,
        "normal_cases_unchanged": normal_cases_unchanged,
        "normal_unnecessary_expansion_avoided": normal_expansion_avoided,
        "alternate_ridge_detected": alternate_ridge_detected,
    }


def _region_counts_basically_same(current_json: str, proposed_json: str) -> bool:
    current = json.loads(current_json)
    proposed = json.loads(proposed_json)
    if current == proposed:
        return True
    if len(current) != 1 or len(proposed) != 1:
        return False
    current_region = _parse_region_text(next(iter(current)))
    proposed_region = _parse_region_text(next(iter(proposed)))
    if current_region is None or proposed_region is None:
        return current_region == proposed_region
    return bool(
        abs(current_region[0] - proposed_region[0])
        <= REGION_BASICALLY_SAME_TOLERANCE_PX
        and abs(current_region[1] - proposed_region[1])
        <= REGION_BASICALLY_SAME_TOLERANCE_PX
    )


def _one_sided_valid_count(
    first_frames: list[FrameLevel],
    second_frames: list[FrameLevel],
) -> int:
    if len(first_frames) != len(second_frames):
        raise ValueError("frame 数量不一致")
    return sum(
        int(np.count_nonzero(first.valid & ~second.valid))
        for first, second in zip(first_frames, second_frames, strict=True)
    )


def _resolver_record_row(record: FrameResolverRecord) -> dict[str, Any]:
    row = asdict(record)
    for key in (
        "current_search_region",
        "proposed_search_region",
        "outside_active_intervals",
        "outside_peak_intervals",
        "active_intervals",
        "seed_interval",
    ):
        row[key] = json.dumps(row[key])
    return row


def _intervals(value: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(start), int(end)) for start, end in value)


def _region_text(region: tuple[int, int] | None) -> str:
    return "none" if region is None else f"[{region[0]},{region[1]})"


def _counter_json(counter: Counter[Any]) -> str:
    return json.dumps(
        {json.dumps(key): value for key, value in counter.items()},
        sort_keys=True,
    )


def _parse_region_text(value: str) -> tuple[int, int] | None:
    if value == "none":
        return None
    start, end = value.removeprefix("[").removesuffix(")").split(",")
    return int(start), int(end)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _concat(values: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    arrays = [value for value in arrays if value.size]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows 不能为空")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
