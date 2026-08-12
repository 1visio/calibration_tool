from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "calibration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_ground_bias_compensation import (  # noqa: E402
    bin_frame,
    frame_plane_residual,
    load_csv_or_txt,
)


INPUT_DIR = (
    ROOT
    / "calibration_tool/projects/daheng/outputs/ground_bias_v_experiment_0812/input"
)
OUTPUT_DIR = Path(__file__).resolve().parent
FRAME_COUNT = 31


def calculate_statistics(
    matrix: np.ndarray,
) -> tuple[np.ndarray, ...]:
    width = matrix.shape[1]
    count = np.sum(np.isfinite(matrix), axis=0)
    median = np.full(width, np.nan)
    mean = np.full(width, np.nan)
    std = np.full(width, np.nan)
    mad = np.full(width, np.nan)
    p95 = np.full(width, np.nan)
    positive_fraction = np.full(width, np.nan)
    sign_consistency = np.full(width, np.nan)
    for column in range(width):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        median[column] = np.median(values)
        mean[column] = np.mean(values)
        std[column] = np.std(values, ddof=1) if values.size > 1 else 0.0
        mad[column] = np.median(np.abs(values - median[column]))
        p95[column] = np.percentile(np.abs(values), 95)
        positive_fraction[column] = np.mean(values > 0.0)
        sign_consistency[column] = max(
            positive_fraction[column], 1.0 - positive_fraction[column]
        )
    return count, median, mean, std, mad, p95, positive_fraction, sign_consistency


def write_statistics(
    v: np.ndarray,
    count: np.ndarray,
    median: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mad: np.ndarray,
    p95: np.ndarray,
) -> None:
    with (OUTPUT_DIR / "residual_v_statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "v",
                "sample_count",
                "residual_median_mm",
                "residual_mean_mm",
                "residual_std_mm",
                "residual_mad_mm",
                "residual_p95_abs_mm",
            ]
        )
        writer.writerows(zip(v.astype(int), count, median, mean, std, mad, p95))


def write_correlations(
    frame_ids: list[str], matrix: np.ndarray
) -> np.ndarray:
    rows: list[tuple[str, str, int, float]] = []
    retained: list[float] = []
    for i, frame_i in enumerate(frame_ids):
        for j, frame_j in enumerate(frame_ids):
            common = np.isfinite(matrix[i]) & np.isfinite(matrix[j])
            common_count = int(np.count_nonzero(common))
            correlation = float("nan")
            if common_count >= 3:
                left = matrix[i, common]
                right = matrix[j, common]
                if np.std(left) > 0.0 and np.std(right) > 0.0:
                    correlation = float(np.corrcoef(left, right)[0, 1])
            rows.append((frame_i, frame_j, common_count, correlation))
            if i != j and common_count >= 100 and np.isfinite(correlation):
                retained.append(correlation)
    with (OUTPUT_DIR / "frame_residual_correlation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_i",
                "frame_j",
                "common_sample_count",
                "correlation_coefficient",
            ]
        )
        writer.writerows(rows)
    return np.asarray(retained, dtype=np.float64)


def save_heatmap(
    frame_ids: list[str], matrix: np.ndarray, v_min: int, v_max: int
) -> float:
    absolute = np.abs(matrix[np.isfinite(matrix)])
    color_limit = float(np.percentile(absolute, 99))
    figure, axis = plt.subplots(figsize=(15, 8))
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        extent=[v_min - 0.5, v_max + 0.5, len(frame_ids) + 0.5, 0.5],
    )
    axis.set_xlabel("Image row v (px)")
    axis.set_ylabel("Frame ID")
    axis.set_title("Per-frame signed vertical residual to each frame plane")
    axis.set_yticks(np.arange(1, len(frame_ids) + 1))
    axis.set_yticklabels(frame_ids, fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label(
        "Signed residual (mm), clipped at 99th |residual| percentile"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "residual_frame_v_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)
    return color_limit


def save_statistics_plot(
    v: np.ndarray,
    count: np.ndarray,
    median: np.ndarray,
    std: np.ndarray,
) -> None:
    valid = count > 0
    figure, residual_axis = plt.subplots(figsize=(15, 6.5))
    residual_axis.plot(
        v[valid], median[valid], color="#174a7e", linewidth=1.0, label="Median b(v)"
    )
    residual_axis.fill_between(
        v[valid],
        median[valid] - std[valid],
        median[valid] + std[valid],
        color="#4c9f70",
        alpha=0.24,
        label="Median ± 1 sigma",
    )
    residual_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    residual_axis.set_xlabel("Image row v (px)")
    residual_axis.set_ylabel("Signed residual (mm)")
    residual_axis.grid(True, alpha=0.22)
    count_axis = residual_axis.twinx()
    count_axis.plot(
        v[valid], count[valid], color="#d65f5f", linewidth=0.85, label="Sample count"
    )
    count_axis.set_ylabel("Frame sample count")
    count_axis.set_ylim(0, FRAME_COUNT + 2)
    lines_a, labels_a = residual_axis.get_legend_handles_labels()
    lines_b, labels_b = count_axis.get_legend_handles_labels()
    residual_axis.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", ncol=3)
    residual_axis.set_title("Across-frame median residual, dispersion, and support")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "residual_v_median_sigma.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def contiguous_ranges(mask: np.ndarray, v: np.ndarray, min_length: int = 20) -> list[list[int]]:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, len(indices) - 1]
    result: list[list[int]] = []
    for start, end in zip(starts, ends):
        left = int(v[indices[start]])
        right = int(v[indices[end]])
        if right - left + 1 >= min_length:
            result.append([left, right])
    return result


def main() -> None:
    paths = sorted(INPUT_DIR.glob("laser *.csv"))
    if len(paths) != FRAME_COUNT:
        raise RuntimeError(f"Expected {FRAME_COUNT} input frames, found {len(paths)}")

    fit_args = SimpleNamespace(
        plane_fit_mad_threshold=3.5, plane_fit_max_iterations=8
    )
    frames = []
    plane_diagnostics = []
    for path in paths:
        frame = load_csv_or_txt(path, ("v", "x", "y", "z"), compensation_axis="v")
        residual_frame, diagnostic = frame_plane_residual(frame, fit_args)
        frame_id = path.stem.removeprefix("laser ").strip()
        diagnostic["frame_id"] = frame_id
        frames.append((frame_id, residual_frame))
        plane_diagnostics.append(diagnostic)

    v_min = int(min(np.min(frame.u) for _, frame in frames))
    v_max = int(max(np.max(frame.u) for _, frame in frames))
    v = np.arange(v_min, v_max + 1, dtype=np.float64)
    matrix = np.full((len(frames), len(v)), np.nan, dtype=np.float64)
    for row, (_, frame) in enumerate(frames):
        bins, _, residual = bin_frame(frame, 1.0)
        indices = np.rint(bins - v_min).astype(int)
        valid = (indices >= 0) & (indices < len(v))
        matrix[row, indices[valid]] = residual[valid]

    (
        count,
        median,
        mean,
        std,
        mad,
        p95,
        positive_fraction,
        sign_consistency,
    ) = calculate_statistics(matrix)
    write_statistics(v, count, median, mean, std, mad, p95)
    pairwise_correlation = write_correlations(
        [frame_id for frame_id, _ in frames], matrix
    )
    color_limit = save_heatmap(
        [frame_id for frame_id, _ in frames], matrix, v_min, v_max
    )
    save_statistics_plot(v, count, median, std)

    diagnostic_keys = [
        "frame_id",
        "source_file",
        "a",
        "b",
        "c_mm",
        "design_rank",
        "design_condition_number",
        "centered_XY_aspect_ratio",
        "input_point_count",
        "finite_point_count",
        "inlier_point_count",
        "iteration_count",
    ]
    with (OUTPUT_DIR / "per_frame_plane_fit_diagnostics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_keys)
        writer.writeheader()
        for item in plane_diagnostics:
            writer.writerow({key: item.get(key) for key in diagnostic_keys})

    observed = np.isfinite(matrix)
    predicted = np.broadcast_to(median, matrix.shape)
    comparable = observed & np.isfinite(predicted)
    total_energy = float(np.sum(matrix[comparable] ** 2))
    unexplained_energy = float(np.sum((matrix[comparable] - predicted[comparable]) ** 2))
    explained_fraction = (
        1.0 - unexplained_energy / total_energy if total_energy > 0.0 else float("nan")
    )

    region_limits = {
        "top_0_299": (0, 299),
        "center_300_2699": (300, 2699),
        "bottom_2700_2999": (2700, 2999),
    }
    region_summary = {}
    for name, (low, high) in region_limits.items():
        selected = (v >= low) & (v <= high) & (count >= 2)
        region_summary[name] = {
            "v_min": low,
            "v_max": high,
            "row_count_with_two_or_more_samples": int(np.count_nonzero(selected)),
            "median_sample_count": float(np.median(count[selected])) if np.any(selected) else None,
            "median_residual_std_mm": float(np.nanmedian(std[selected])) if np.any(selected) else None,
            "median_residual_mad_mm": float(np.nanmedian(mad[selected])) if np.any(selected) else None,
            "median_sign_consistency": float(np.nanmedian(sign_consistency[selected])) if np.any(selected) else None,
            "median_abs_common_bias_mm": float(np.nanmedian(np.abs(median[selected]))) if np.any(selected) else None,
        }

    well_supported = count >= 10
    std_threshold = float(np.nanpercentile(std[well_supported], 25))
    high_consistency = (
        well_supported & (sign_consistency >= 0.8) & (std <= std_threshold)
    )
    sign_flip = well_supported & (positive_fraction >= 0.35) & (positive_fraction <= 0.65)
    block_summary = []
    for low in range(v_min, v_max + 1, 100):
        high = min(low + 99, v_max)
        selected = (v >= low) & (v <= high) & well_supported
        if not np.any(selected):
            continue
        block_summary.append(
            {
                "v_range": [low, high],
                "well_supported_row_count": int(np.count_nonzero(selected)),
                "median_sample_count": float(np.median(count[selected])),
                "median_std_mm": float(np.nanmedian(std[selected])),
                "median_abs_bias_mm": float(np.nanmedian(np.abs(median[selected]))),
                "median_sign_consistency": float(np.nanmedian(sign_consistency[selected])),
                "sign_flip_row_fraction": float(np.mean(sign_flip[selected])),
            }
        )
    summary = {
        "input_dir": str(INPUT_DIR),
        "frame_count": len(frames),
        "frame_ids": [frame_id for frame_id, _ in frames],
        "compensation_applied": False,
        "smooth_window_applied": False,
        "residual_definition": "signed vertical residual Zg-(a*Xg+b*Yg+c), fitted independently per frame with iterative MAD rejection",
        "v_range": [v_min, v_max],
        "observed_point_bin_count": int(np.count_nonzero(observed)),
        "support": {
            "rows_observed": int(np.count_nonzero(count > 0)),
            "rows_with_at_least_5_frames": int(np.count_nonzero(count >= 5)),
            "rows_with_at_least_10_frames": int(np.count_nonzero(well_supported)),
            "rows_with_at_least_20_frames": int(np.count_nonzero(count >= 20)),
            "max_sample_count": int(np.max(count)),
            "global_all_31_frame_common_rows": int(np.count_nonzero(count == FRAME_COUNT)),
        },
        "pairwise_correlation_for_overlap_ge_100": {
            "pair_entry_count_including_both_directions": int(pairwise_correlation.size),
            "median": float(np.median(pairwise_correlation)),
            "p10": float(np.percentile(pairwise_correlation, 10)),
            "p90": float(np.percentile(pairwise_correlation, 90)),
            "fraction_positive": float(np.mean(pairwise_correlation > 0.0)),
            "fraction_above_0_5": float(np.mean(pairwise_correlation >= 0.5)),
            "fraction_below_minus_0_2": float(np.mean(pairwise_correlation <= -0.2)),
        },
        "common_profile_explained_energy_fraction": float(explained_fraction),
        "rows_with_sample_ge_10": {
            "count": int(np.count_nonzero(well_supported)),
            "median_std_mm": float(np.nanmedian(std[well_supported])),
            "median_mad_mm": float(np.nanmedian(mad[well_supported])),
            "median_abs_bias_mm": float(np.nanmedian(np.abs(median[well_supported]))),
            "median_sign_consistency": float(np.nanmedian(sign_consistency[well_supported])),
            "fraction_sign_consistency_ge_0_8": float(np.mean(sign_consistency[well_supported] >= 0.8)),
            "fraction_sign_consistency_le_0_6": float(np.mean(sign_consistency[well_supported] <= 0.6)),
        },
        "region_definition": "top and bottom are fixed 10% image-height bands; center is the remaining 80%",
        "regions": region_summary,
        "high_consistency_definition": f"sample_count>=10, sign_consistency>=0.8, std<=well-supported 25th percentile ({std_threshold:.6f} mm)",
        "high_consistency_ranges_min_20_rows": contiguous_ranges(high_consistency, v),
        "sign_flip_definition": "sample_count>=10 and positive residual fraction in [0.35,0.65]",
        "sign_flip_ranges_min_20_rows": contiguous_ranges(sign_flip, v),
        "sign_flip_ranges_min_5_rows": contiguous_ranges(sign_flip, v, min_length=5),
        "blocks_100px_well_supported": block_summary,
        "heatmap_color_limit_mm": color_limit,
        "plane_fit": {
            "min_inlier_fraction": float(min(item["inlier_point_count"] / item["input_point_count"] for item in plane_diagnostics)),
            "median_inlier_fraction": float(np.median([item["inlier_point_count"] / item["input_point_count"] for item in plane_diagnostics])),
            "max_condition_number": float(max(item["design_condition_number"] for item in plane_diagnostics)),
            "median_condition_number": float(np.median([item["design_condition_number"] for item in plane_diagnostics])),
        },
    }
    (OUTPUT_DIR / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
