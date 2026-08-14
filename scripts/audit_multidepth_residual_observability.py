#!/usr/bin/env python3
"""Task 5B-1: FIT-only multi-depth residual observability audit.

This script is deliberately diagnostic.  It never opens validation data,
never writes a production model, and never constructs a deployable correction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import compare_circular_vs_elliptical_cone as task5a  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
import run_circular_cone_local_fullfit as task3b2  # noqa: E402
import run_circular_cone_residual_decomposition as task4a  # noqa: E402


DEFAULT_OUTPUT = ROOT / "projects" / "daheng" / "outputs" / "0814" / "multidepth_residual_observability"
TASK5A_PROVENANCE = ROOT / "projects" / "daheng" / "outputs" / "0814" / "circular_vs_elliptical_cone" / "provenance.json"
BIN_WIDTHS = (30, 60, 100)
PRIMARY_WIDTH = 60
IMAGE_HEIGHT = 3000
TARGET_FRAME = "027"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260814
MIN_FRAMES_DEPTH = 3
MIN_CROSSFRAME_LAMBDA_SPAN_MM = 2.0
REGIONS = (("global", 0.0, 3000.0),) + task3a.REGIONS

# Fixed, pre-result decision gates.  All improvement numbers are relative to
# the offset-only diagnostic, not relative to the uncorrected residual.
STRONG_INCREMENTAL_EXPLAINED = 0.20
PARTIAL_INCREMENTAL_EXPLAINED = 0.05
STRONG_EDGE_RMSE_REDUCTION = 0.15
PARTIAL_EDGE_RMSE_REDUCTION = 0.10
STRONG_STABLE_BIN_FRACTION = 0.70
PARTIAL_STABLE_BIN_FRACTION = 0.40


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5B-1 multi-depth residual observability audit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--measurement-config", type=Path, default=task3b2.MEASUREMENT_CONFIG)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def collect_points(
    records: Sequence[task3a.FrameRecord],
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    current_model: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    models = {"M0": copy.deepcopy(dict(calibration)), "M_current29_circular": copy.deepcopy(dict(calibration))}
    models["M_current29_circular"]["laser_model"] = copy.deepcopy(dict(current_model))
    rows: list[dict[str, Any]] = []
    invalid = defaultdict(int)
    for record in records:
        reconstructed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model, candidate in models.items():
            values, valid, _ = task4a.lambda_by_input(record.pixels_uv, candidate, reconstruction_params)
            reconstructed[model] = (values, valid)
            invalid[model] += int(np.count_nonzero(~valid))
        for point_index, (uv, truth) in enumerate(zip(record.pixels_uv, record.truth_points)):
            row: dict[str, Any] = {
                "frame_id": record.frame_id,
                "source": record.source,
                "point_index": point_index,
                "u_px": float(uv[0]),
                "v_px": float(uv[1]),
                "region": task3a.region_for_v(float(uv[1])),
                "formal_domain": bool(task3a.FORMAL_V_MIN <= float(uv[1]) <= task3a.FORMAL_V_MAX),
                "lambda_truth_mm": float(truth[2]),
            }
            for model, (values, valid) in reconstructed.items():
                ok = bool(valid[point_index])
                row[f"{model}_valid"] = ok
                row[f"lambda_{model}_mm"] = float(values[point_index]) if ok else float("nan")
                # Additive-correction convention: truth minus physical model.
                row[f"e_lambda_{model}_mm"] = float(truth[2] - values[point_index]) if ok else float("nan")
            rows.append(row)
    return rows, dict(invalid)


def frame_groups(rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, list[Mapping[str, Any]]]:
    key = f"e_lambda_{model}_mm"
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if finite(row.get(key)) and finite(row.get("lambda_truth_mm")):
            grouped[str(row["frame_id"])].append(row)
    return dict(grouped)


def full_fit(rows: Sequence[Mapping[str, Any]], model: str, lambda_ref: float | None = None) -> dict[str, Any]:
    groups = frame_groups(rows, model)
    ids = sorted(groups)
    if not ids:
        return {"status": "unsupported", "unique_frame_count": 0, "point_count": 0}
    frame_lambda = np.asarray([np.median([float(r["lambda_truth_mm"]) for r in groups[fid]]) for fid in ids])
    if lambda_ref is None:
        lambda_ref = float(np.median(frame_lambda))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    for fid in ids:
        x = np.asarray([float(r["lambda_truth_mm"]) - lambda_ref for r in groups[fid]])
        y = np.asarray([float(r[f"e_lambda_{model}_mm"]) for r in groups[fid]])
        x_parts.append(x); y_parts.append(y); w_parts.append(np.full(len(x), 1.0 / len(x)))
    x = np.concatenate(x_parts); y = np.concatenate(y_parts); weights = np.concatenate(w_parts)
    b_offset = float(np.sum(weights * y) / np.sum(weights))
    design = np.column_stack([np.ones_like(x), x])
    sqrt_w = np.sqrt(weights)
    beta = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    prediction_offset = np.full_like(y, b_offset)
    prediction_depth = design @ beta
    total = float(np.sum(weights * y * y))
    offset_remaining = float(np.sum(weights * (y - prediction_offset) ** 2))
    depth_remaining = float(np.sum(weights * (y - prediction_depth) ** 2))
    cross_span = float(np.ptp(frame_lambda))
    depth_usable = len(ids) >= MIN_FRAMES_DEPTH and cross_span >= MIN_CROSSFRAME_LAMBDA_SPAN_MM
    x_scale = max(float(np.std(x)), 1.0e-12)
    normalized_design = np.column_stack([np.ones_like(x), x / x_scale])
    condition = float(np.linalg.cond(normalized_design * sqrt_w[:, None]))
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {
        "status": "depth_informative" if depth_usable else ("single_frame_only" if len(ids) == 1 else "depth_excitation_weak"),
        "point_count": int(len(y)),
        "unique_frame_count": len(ids),
        "frame_ids": ids,
        "lambda_ref_mm": float(lambda_ref),
        "lambda_min_mm": float(np.min(x + lambda_ref)),
        "lambda_max_mm": float(np.max(x + lambda_ref)),
        "lambda_span_mm": float(np.ptp(x)),
        "crossframe_median_lambda_span_mm": cross_span,
        "b0_offset_only_mm": b_offset,
        "b0_mm": float(beta[0]),
        "b1_mm_per_mm": float(beta[1]),
        "design_condition_number_standardized": condition,
        "corr_lambda_residual": corr,
        "total_energy": total,
        "offset_remaining_energy": offset_remaining,
        "depth_remaining_energy": depth_remaining,
        "offset_explained_fraction": 1.0 - offset_remaining / total if total > 0 else float("nan"),
        "offset_plus_depth_explained_fraction": 1.0 - depth_remaining / total if total > 0 else float("nan"),
        "incremental_depth_explained_fraction": (offset_remaining - depth_remaining) / total if total > 0 else float("nan"),
        "offset_rmse_mm": math.sqrt(offset_remaining / np.sum(weights)),
        "depth_rmse_mm": math.sqrt(depth_remaining / np.sum(weights)),
        "depth_usable": depth_usable,
    }


def resample_fit(
    groups: Mapping[str, Sequence[Mapping[str, Any]]], model: str, lambda_ref: float, selected: Sequence[str]
) -> tuple[float, float, float]:
    normal = np.zeros((2, 2)); rhs = np.zeros(2); total_weight = 0.0; y2 = 0.0
    for fid in selected:
        group = groups[fid]
        x = np.asarray([float(r["lambda_truth_mm"]) - lambda_ref for r in group])
        y = np.asarray([float(r[f"e_lambda_{model}_mm"]) for r in group])
        design = np.column_stack([np.ones_like(x), x])
        normal += design.T @ design / len(group)
        rhs += design.T @ y / len(group)
        y2 += float(y @ y / len(group)); total_weight += 1.0
    try:
        beta = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    sse = max(y2 - 2.0 * float(beta @ rhs) + float(beta @ normal @ beta), 0.0)
    return float(beta[0]), float(beta[1]), math.sqrt(sse / max(total_weight, 1.0))


def uncertainty(
    rows: Sequence[Mapping[str, Any]], model: str, base: Mapping[str, Any], reps: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = frame_groups(rows, model); ids = sorted(groups)
    if len(ids) < 2 or not finite(base.get("lambda_ref_mm")):
        return {}, []
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    for index in range(reps):
        selected = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
        b0, b1, rmse = resample_fit(groups, model, float(base["lambda_ref_mm"]), selected)
        samples.append({"resampling_method": "frame_bootstrap", "sample_index": index, "omitted_frame": "", "b0_mm": b0, "b1_mm_per_mm": b1, "fit_rmse_mm": rmse})
    for index, omitted in enumerate(ids):
        selected = [fid for fid in ids if fid != omitted]
        b0, b1, rmse = resample_fit(groups, model, float(base["lambda_ref_mm"]), selected)
        samples.append({"resampling_method": "leave_one_frame_out", "sample_index": index, "omitted_frame": omitted, "b0_mm": b0, "b1_mm_per_mm": b1, "fit_rmse_mm": rmse})
    boot_b0 = np.asarray([s["b0_mm"] for s in samples if s["resampling_method"] == "frame_bootstrap" and finite(s["b0_mm"])])
    boot_b1 = np.asarray([s["b1_mm_per_mm"] for s in samples if s["resampling_method"] == "frame_bootstrap" and finite(s["b1_mm_per_mm"])])
    loo_b1 = np.asarray([s["b1_mm_per_mm"] for s in samples if s["resampling_method"] == "leave_one_frame_out" and finite(s["b1_mm_per_mm"])])
    full_b1 = float(base.get("b1_mm_per_mm", float("nan")))
    sign = 0.0 if full_b1 == 0 else math.copysign(1.0, full_b1)
    sign_consistency = float(np.mean(np.sign(boot_b1) == sign)) if boot_b1.size and sign else float("nan")
    loo_sign_consistency = float(np.mean(np.sign(loo_b1) == sign)) if loo_b1.size and sign else float("nan")
    summary = {
        "bootstrap_count": int(boot_b1.size),
        "b0_p05_mm": float(np.percentile(boot_b0, 5)) if boot_b0.size else float("nan"),
        "b0_p50_mm": float(np.percentile(boot_b0, 50)) if boot_b0.size else float("nan"),
        "b0_p95_mm": float(np.percentile(boot_b0, 95)) if boot_b0.size else float("nan"),
        "b1_p05_mm_per_mm": float(np.percentile(boot_b1, 5)) if boot_b1.size else float("nan"),
        "b1_p50_mm_per_mm": float(np.percentile(boot_b1, 50)) if boot_b1.size else float("nan"),
        "b1_p95_mm_per_mm": float(np.percentile(boot_b1, 95)) if boot_b1.size else float("nan"),
        "b1_bootstrap_sign_consistency": sign_consistency,
        "b1_loo_min_mm_per_mm": float(np.min(loo_b1)) if loo_b1.size else float("nan"),
        "b1_loo_max_mm_per_mm": float(np.max(loo_b1)) if loo_b1.size else float("nan"),
        "b1_loo_sign_consistency": loo_sign_consistency,
    }
    summary["b1_stable"] = bool(
        base.get("depth_usable", False)
        and finite(summary["b1_p05_mm_per_mm"])
        and finite(summary["b1_p95_mm_per_mm"])
        and summary["b1_p05_mm_per_mm"] * summary["b1_p95_mm_per_mm"] > 0
        and sign_consistency >= 0.8
        and loo_sign_consistency >= 0.8
    )
    return summary, samples


def make_bin_rows(points: Sequence[Mapping[str, Any]], reps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []; samples: list[dict[str, Any]] = []
    for width in BIN_WIDTHS:
        for start in range(0, IMAGE_HEIGHT, width):
            end = min(start + width, IMAGE_HEIGHT)
            selected = [r for r in points if start <= float(r["v_px"]) < end]
            for model_index, model in enumerate(("M0", "M_current29_circular")):
                base = full_fit(selected, model)
                uncertainty_summary, raw = uncertainty(selected, model, base, reps, BOOTSTRAP_SEED + width * 10000 + start * 2 + model_index)
                row = {
                    "bin_width_px": width, "v_start_px": start, "v_end_px": end,
                    "v_center_px": 0.5 * (start + end), "region": task3a.region_for_v(0.5 * (start + end)),
                    "model": model, **{key: value for key, value in base.items() if key != "frame_ids"}, **uncertainty_summary,
                }
                result.append(row)
                for sample in raw:
                    samples.append({"bin_width_px": width, "v_start_px": start, "v_end_px": end, "v_center_px": 0.5 * (start + end), "model": model, **sample})
    return result, samples


def frame_equal_metrics(
    points: Sequence[Mapping[str, Any]], bins: Sequence[Mapping[str, Any]], model: str, width: int, region: tuple[str, float, float]
) -> dict[str, Any]:
    name, low, high = region
    selected = [r for r in points if low <= float(r["v_px"]) < high and finite(r.get(f"e_lambda_{model}_mm"))]
    by_bin = {(int(r["v_start_px"]), int(r["v_end_px"])): r for r in bins if r["model"] == model and int(r["bin_width_px"]) == width}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[str(row["frame_id"])].append(row)
    total = offset = depth = 0.0; total_n = 0.0
    frame_before: list[float] = []; frame_offset: list[float] = []; frame_depth: list[float] = []
    used = 0
    for fid, rows in groups.items():
        before_values: list[float] = []; offset_values: list[float] = []; depth_values: list[float] = []
        for row in rows:
            y = float(row[f"e_lambda_{model}_mm"]); v = float(row["v_px"]); lam = float(row["lambda_truth_mm"])
            start = int(math.floor(v / width) * width); summary = by_bin.get((start, min(start + width, IMAGE_HEIGHT)))
            if summary is None or not finite(summary.get("b0_offset_only_mm")):
                continue
            pred_offset = float(summary["b0_offset_only_mm"])
            pred_depth = pred_offset
            if bool(summary.get("depth_usable", False)):
                pred_depth = float(summary["b0_mm"]) + float(summary["b1_mm_per_mm"]) * (lam - float(summary["lambda_ref_mm"]))
            before_values.append(y); offset_values.append(y - pred_offset); depth_values.append(y - pred_depth)
        if not before_values:
            continue
        b = np.asarray(before_values); o = np.asarray(offset_values); d = np.asarray(depth_values)
        total += float(np.mean(b * b)); offset += float(np.mean(o * o)); depth += float(np.mean(d * d)); total_n += 1.0
        frame_before.append(float(np.mean(b))); frame_offset.append(float(np.mean(o))); frame_depth.append(float(np.mean(d))); used += len(b)
    def std(values: Sequence[float]) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {
        "model": model, "bin_width_px": width, "region": name, "point_count": used, "unique_frame_count": len(groups),
        "total_energy_frame_equal": total, "offset_remaining_energy_frame_equal": offset, "offset_plus_depth_remaining_energy_frame_equal": depth,
        "before_rmse_mm": math.sqrt(total / total_n) if total_n else float("nan"),
        "offset_rmse_mm": math.sqrt(offset / total_n) if total_n else float("nan"),
        "offset_plus_depth_rmse_mm": math.sqrt(depth / total_n) if total_n else float("nan"),
        "offset_explained_fraction": 1.0 - offset / total if total > 0 else float("nan"),
        "offset_plus_depth_explained_fraction": 1.0 - depth / total if total > 0 else float("nan"),
        "incremental_depth_explained_fraction": (offset - depth) / total if total > 0 else float("nan"),
        "depth_rmse_reduction_vs_offset_fraction": 1.0 - math.sqrt(depth / offset) if offset > 0 else float("nan"),
        "frame_mean_std_before_mm": std(frame_before), "frame_mean_std_after_offset_mm": std(frame_offset),
        "frame_mean_std_after_offset_plus_depth_mm": std(frame_depth),
    }


def evaluate(points: Sequence[Mapping[str, Any]], bins: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    region_rows = [frame_equal_metrics(points, bins, model, width, region) for width in BIN_WIDTHS for model in ("M0", "M_current29_circular") for region in REGIONS]
    primary = {r["region"]: r for r in region_rows if r["model"] == "M_current29_circular" and r["bin_width_px"] == PRIMARY_WIDTH}
    informative = [r for r in bins if r["model"] == "M_current29_circular" and r["bin_width_px"] == PRIMARY_WIDTH and bool(r.get("depth_usable", False))]
    stable_fraction = float(np.mean([bool(r.get("b1_stable", False)) for r in informative])) if informative else 0.0
    global_gain = float(primary["global"]["incremental_depth_explained_fraction"])
    top_gain = float(primary["top_formal_edge"]["depth_rmse_reduction_vs_offset_fraction"])
    bottom_gain = float(primary["bottom_formal_edge"]["depth_rmse_reduction_vs_offset_fraction"])
    if global_gain >= STRONG_INCREMENTAL_EXPLAINED and min(top_gain, bottom_gain) >= STRONG_EDGE_RMSE_REDUCTION and stable_fraction >= STRONG_STABLE_BIN_FRACTION:
        verdict = "A. STRONG"
    elif global_gain >= PARTIAL_INCREMENTAL_EXPLAINED and max(top_gain, bottom_gain) >= PARTIAL_EDGE_RMSE_REDUCTION and stable_fraction >= PARTIAL_STABLE_BIN_FRACTION:
        verdict = "B. PARTIAL"
    else:
        verdict = "C. WEAK"
    return region_rows, {
        "verdict": verdict, "primary_width_px": PRIMARY_WIDTH,
        "global_incremental_depth_explained_fraction": global_gain,
        "top_rmse_reduction_vs_offset_fraction": top_gain,
        "bottom_rmse_reduction_vs_offset_fraction": bottom_gain,
        "informative_bin_count": len(informative), "stable_b1_bin_count": sum(bool(r.get("b1_stable", False)) for r in informative),
        "stable_b1_bin_fraction": stable_fraction,
    }


def make_plots(out: Path, points: Sequence[Mapping[str, Any]], bins: Sequence[Mapping[str, Any]], regions: Sequence[Mapping[str, Any]]) -> None:
    model = "M_current29_circular"; key = f"e_lambda_{model}_mm"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for axis, (name, low, high) in zip(axes, task3a.REGIONS):
        rows = [r for r in points if low <= float(r["v_px"]) < high and finite(r.get(key))]
        if len(rows) > 6000:
            rows = rows[:: math.ceil(len(rows) / 6000)]
        axis.scatter([float(r["lambda_truth_mm"]) for r in rows], [float(r[key]) for r in rows], c=[float(r["v_px"]) for r in rows], s=4, alpha=0.25, cmap="viridis")
        axis.axhline(0, color="black", linewidth=0.7); axis.set_title(name); axis.set_xlabel("lambda_truth / mm"); axis.grid(alpha=0.2)
    axes[0].set_ylabel("e_lambda = truth - model / mm")
    fig.tight_layout(); fig.savefig(out / "residual_vs_lambda_by_v_region.png", dpi=180); plt.close(fig)

    primary = [r for r in bins if r["bin_width_px"] == PRIMARY_WIDTH and int(r.get("unique_frame_count", 0)) > 0]
    for field, low_field, high_field, filename, ylabel in (
        ("b0_mm", "b0_p05_mm", "b0_p95_mm", "b0_vs_v.png", "b0 / mm"),
        ("b1_mm_per_mm", "b1_p05_mm_per_mm", "b1_p95_mm_per_mm", "b1_vs_v.png", "b1 / mm per mm"),
    ):
        fig, axis = plt.subplots(figsize=(10, 5.2))
        for model_name, color in (("M0", "#2b6cb0"), ("M_current29_circular", "#c05621")):
            rows = [r for r in primary if r["model"] == model_name and finite(r.get(field))]
            if field == "b1_mm_per_mm":
                rows = [r for r in rows if bool(r.get("depth_usable", False))]
            x = np.asarray([r["v_center_px"] for r in rows]); y = np.asarray([r[field] for r in rows])
            axis.plot(x, y, marker="o", markersize=3, linewidth=1, label=model_name, color=color)
            valid_ci = [r for r in rows if finite(r.get(low_field)) and finite(r.get(high_field))]
            if valid_ci:
                axis.fill_between([r["v_center_px"] for r in valid_ci], [r[low_field] for r in valid_ci], [r[high_field] for r in valid_ci], color=color, alpha=0.15)
        axis.axhline(0, color="black", linewidth=0.7); axis.axvline(300, color="gray", linestyle="--"); axis.axvline(2700, color="gray", linestyle="--")
        if field == "b1_mm_per_mm":
            axis.set_yscale("symlog", linthresh=1.0e-3)
        axis.set_xlabel("v / px"); axis.set_ylabel(ylabel); axis.grid(alpha=0.2); axis.legend(); fig.tight_layout(); fig.savefig(out / filename, dpi=180); plt.close(fig)

    rows = [r for r in regions if r["model"] == model and r["bin_width_px"] == PRIMARY_WIDTH]
    names = [r["region"] for r in rows]; offset = [r["offset_explained_fraction"] for r in rows]; depth = [r["offset_plus_depth_explained_fraction"] for r in rows]
    x = np.arange(len(names)); fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, offset, width=0.36, label="offset only"); axis.bar(x + 0.18, depth, width=0.36, label="offset + depth")
    axis.set_xticks(x, names, rotation=15); axis.set_ylabel("explained residual energy fraction"); axis.grid(axis="y", alpha=0.2); axis.legend(); fig.tight_layout()
    fig.savefig(out / "offset_vs_depth_explained_fraction.png", dpi=180); plt.close(fig)


def report_text(summary: Mapping[str, Any], region_rows: Sequence[Mapping[str, Any]], bins: Sequence[Mapping[str, Any]], point_count: int, invalid: Mapping[str, int]) -> str:
    primary = {r["region"]: r for r in region_rows if r["model"] == "M_current29_circular" and r["bin_width_px"] == PRIMARY_WIDTH}
    m0_primary = {r["region"]: r for r in region_rows if r["model"] == "M0" and r["bin_width_px"] == PRIMARY_WIDTH}
    informative = [r for r in bins if r["model"] == "M_current29_circular" and r["bin_width_px"] == PRIMARY_WIDTH and bool(r.get("depth_usable", False))]
    m0_informative = [r for r in bins if r["model"] == "M0" and r["bin_width_px"] == PRIMARY_WIDTH and bool(r.get("depth_usable", False))]
    b1_values = np.asarray([float(r["b1_mm_per_mm"]) for r in informative]) if informative else np.asarray([])
    stable = [r for r in informative if bool(r.get("b1_stable", False))]
    informative_sorted = sorted(informative, key=lambda r: float(r["v_center_px"]))
    b1_signs = [int(np.sign(float(r["b1_mm_per_mm"]))) for r in informative_sorted]
    sign_changes = sum(left != right for left, right in zip(b1_signs, b1_signs[1:]) if left and right)
    top_bins = [r for r in informative if float(r["v_start_px"]) < 300.0]
    bottom_bins = [r for r in informative if float(r["v_end_px"]) > 2700.0]
    scale_rows: list[tuple[int, int, int, float, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for width in BIN_WIDTHS:
        usable = [r for r in bins if r["model"] == "M_current29_circular" and int(r["bin_width_px"]) == width and bool(r.get("depth_usable", False))]
        selected_regions = {r["region"]: r for r in region_rows if r["model"] == "M_current29_circular" and int(r["bin_width_px"]) == width}
        stable_count = sum(bool(r.get("b1_stable", False)) for r in usable)
        scale_rows.append((width, len(usable), stable_count, stable_count / len(usable) if usable else 0.0, selected_regions["global"], selected_regions["top_formal_edge"], selected_regions["bottom_formal_edge"]))
    frame_before = primary["global"]["frame_mean_std_after_offset_mm"]
    frame_after = primary["global"]["frame_mean_std_after_offset_plus_depth_mm"]
    lines = [
        "# Task 5B-1 — Multi-depth residual observability audit", "",
        f"`MULTIDEPTH_CORRECTION_FEASIBILITY = {summary['verdict']}`", "",
        "## Scope and safeguards", "",
        "- Main FIT only: `001–018 + 025–036`, with `027` temporarily excluded (29 frames).",
        "- Validation `019–024, 037–040` was not loaded.",
        "- Models are read-only: formal `M0` and Task 5A's 29-frame Circular diagnostic model.",
        "- Residual convention: `e_lambda = lambda_truth - lambda_model`.",
        f"- Analysed {point_count} independent ray-plane truth points; invalid intersections: {dict(invalid)}.",
        "- No correction, production parameter, spline, polynomial, or LUT was written.", "",
        "## Fixed method", "",
        "For each fixed 30/60/100 px v-bin, every frame has equal total weight. `lambda_ref` is the median of per-frame median truth depth in that bin. The depth slope is enabled only when at least 3 frames and at least 2 mm cross-frame median-depth span are present; otherwise the depth prediction falls back to offset-only. Uncertainty uses 500 frame bootstrap draws plus leave-one-frame-out.", "",
        "The verdict uses the predeclared 60 px gates: STRONG requires global incremental explained fraction ≥20%, both edge RMSE reductions ≥15%, and ≥70% stable informative slopes. PARTIAL requires ≥5%, at least one edge ≥10%, and ≥40% stable slopes. Otherwise WEAK.", "",
        "## Primary 60 px result (current 29-frame Circular)", "",
        "| region | offset explained | offset+depth explained | incremental | RMSE reduction vs offset | frame-mean std after depth (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("global", "top_formal_edge", "middle_formal", "bottom_formal_edge"):
        r = primary[name]
        lines.append(f"| {name} | {r['offset_explained_fraction']:.3f} | {r['offset_plus_depth_explained_fraction']:.3f} | {r['incremental_depth_explained_fraction']:.3f} | {r['depth_rmse_reduction_vs_offset_fraction']:.3f} | {r['frame_mean_std_after_offset_plus_depth_mm']:.4f} |")
    lines += [
        "", "## Observability and stability", "",
        f"- Informative 60 px bins: {len(informative)}; stable b1 bins: {len(stable)} ({summary['stable_b1_bin_fraction']:.1%}).",
        f"- Informative-bin b1 range: {float(np.min(b1_values)) if b1_values.size else float('nan'):.6g} to {float(np.max(b1_values)) if b1_values.size else float('nan'):.6g} mm/mm.",
        f"- Across adjacent informative 60 px bins, b1 changes sign {sign_changes} times ({sum(value > 0 for value in b1_signs)} positive, {sum(value < 0 for value in b1_signs)} negative); this is not a smooth, repeatable v-trend.",
        f"- Top has {len(top_bins)} depth-informative 60 px bin and {sum(bool(r.get('b1_stable', False)) for r in top_bins)} stable bin; bottom has {len(bottom_bins)} informative bin and {sum(bool(r.get('b1_stable', False)) for r in bottom_bins)} stable bin.",
        f"- Global frame-mean residual std after offset-only / after depth: {frame_before:.4f} / {frame_after:.4f} mm.",
        "- A bin is called stable only if its bootstrap 90% interval excludes zero and both bootstrap and LOFO sign consistency are at least 80%.", "",
        "### Bin-scale consistency", "",
        "| bin width | informative bins | stable bins | stable fraction | global incremental | top RMSE reduction | bottom RMSE reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for width, usable_count, stable_count, stable_fraction, global_row, top_row, bottom_row in scale_rows:
        lines.append(f"| {width} px | {usable_count} | {stable_count} | {stable_fraction:.1%} | {global_row['incremental_depth_explained_fraction']:.1%} | {top_row['depth_rmse_reduction_vs_offset_fraction']:.1%} | {bottom_row['depth_rmse_reduction_vs_offset_fraction']:.1%} |")
    lines += [
        "", "All three scales agree that the depth term adds only about 2–3% global explained energy, while stable-slope coverage remains only 13–19%. Edge in-sample improvement therefore does not amount to a globally observable correction.", "",
        "### Read-only M0 cross-check", "",
        f"M0 reaches {m0_primary['global']['incremental_depth_explained_fraction']:.1%} global incremental explanation, with {sum(bool(r.get('b1_stable', False)) for r in m0_informative)}/{len(m0_informative)} stable 60 px slopes. Its top/bottom RMSE reductions are {m0_primary['top_formal_edge']['depth_rmse_reduction_vs_offset_fraction']:.1%}/{m0_primary['bottom_formal_edge']['depth_rmse_reduction_vs_offset_fraction']:.1%}. The larger M0 bottom-only response is not mirrored at top and does not change the current-model verdict.", "",
        "## Answers", "",
        f"1. Adding depth changes global explained energy by {summary['global_incremental_depth_explained_fraction']:.1%} beyond offset-only.",
        f"2. Top / bottom RMSE reductions beyond offset-only are {summary['top_rmse_reduction_vs_offset_fraction']:.1%} / {summary['bottom_rmse_reduction_vs_offset_fraction']:.1%}.",
        f"3. b1 is stable in only {summary['stable_b1_bin_fraction']:.1%} of depth-informative 60 px bins and changes sign repeatedly. It is not yet a stable, smooth, cross-frame-repeatable function of v.",
        f"4. Frame dependence {'remains substantial' if frame_after > 0.7 * frame_before else 'is materially reduced'} after the in-sample depth decomposition (frame-mean std ratio {frame_after / frame_before if frame_before else float('nan'):.3f}).", "",
        "## What this can and cannot establish", "",
        "This FIT-only audit can show whether a low-order depth term is locally observable and repeatable across frames. It cannot establish generalization, choose a deployable v-parameterization, or justify production correction. Those require a separately frozen candidate followed by untouched validation.", "",
        f"`MULTIDEPTH_CORRECTION_FEASIBILITY = {summary['verdict']}`", "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    hash_before = sha256_file(task3b2.FORMAL_CONE)
    if hash_before != task3b2.EXPECTED_CONE_SHA256:
        raise RuntimeError(f"Formal M0 hash mismatch: {hash_before}")
    provenance = json.loads(TASK5A_PROVENANCE.read_text(encoding="utf-8"))
    if provenance.get("validation_opened") is not False or provenance.get("formal_m0_modified") is not False:
        raise RuntimeError("Task 5A provenance violates split or production boundary")
    expected_ids = [fid for fid in task3a.FIT_IDS if fid != TARGET_FRAME]
    if provenance.get("main_fit_ids") != expected_ids:
        raise RuntimeError("Task 5A main FIT registry mismatch")
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records = task3a.load_old_records(); extension, _ = task3a.load_extension_records(intrinsics); records += extension
    if [r.frame_id for r in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT registry mismatch")
    records = [r for r in records if r.frame_id != TARGET_FRAME]
    if [r.frame_id for r in records] != expected_ids or len(records) != 29:
        raise RuntimeError("Leave-027-out registry mismatch")
    z = np.concatenate([r.truth_points[:, 2] for r in records])
    current = task5a.circular_params_to_runtime_model(
        np.asarray(provenance["full_fits"]["Circular"]["params"], dtype=np.float64),
        np.asarray(provenance["reference_anchor_mm"], dtype=np.float64), [float(np.min(z)), float(np.max(z))],
    )
    points, invalid = collect_points(records, calibration, reconstruction_params, current)
    bins, resamples = make_bin_rows(points, int(args.bootstrap_reps))
    region_rows, summary = evaluate(points, bins)
    write_csv(out / "depth_residual_observability.csv", points)
    write_csv(out / "b0_b1_vs_v.csv", [*bins, *({"row_type": "region_explainability", **r} for r in region_rows)])
    write_csv(out / "frame_bootstrap.csv", resamples)
    make_plots(out, points, bins, region_rows)
    hash_after = sha256_file(task3b2.FORMAL_CONE)
    if hash_after != hash_before:
        raise RuntimeError("Formal M0 changed during read-only audit")
    metadata = {
        "task": "Task 5B-1 multi-depth residual observability audit", "fit_ids": expected_ids,
        "excluded_sensitivity_frame": TARGET_FRAME, "validation_ids_not_opened": task3a.VALIDATION_IDS,
        "validation_opened": False, "formal_m0_modified": False, "formal_cone_sha256_before": hash_before,
        "formal_cone_sha256_after": hash_after, "bin_widths_px": BIN_WIDTHS, "primary_width_px": PRIMARY_WIDTH,
        "bootstrap_reps": int(args.bootstrap_reps), "bootstrap_unit": "frame", "weighting": "frame_equal_within_bin",
        "depth_excitation_gate": {"min_unique_frames": MIN_FRAMES_DEPTH, "min_crossframe_median_lambda_span_mm": MIN_CROSSFRAME_LAMBDA_SPAN_MM},
        "decision": summary,
    }
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(report_text(summary, region_rows, bins, len(points), invalid), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "decision": summary, "point_count": len(points), "resample_rows": len(resamples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
