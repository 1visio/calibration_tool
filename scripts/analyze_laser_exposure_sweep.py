from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_tool.camera.quality import laser_scanline_metrics
from calibration_tool.io_utils import load_document


def _percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _peak_width_metrics(image: np.ndarray, sensor_max_value: float) -> dict[str, np.ndarray]:
    metrics = laser_scanline_metrics(
        image,
        sensor_max_value=sensor_max_value,
        scan_axis="row",
    )
    values = image.astype(np.float32, copy=False)
    background = np.asarray(metrics["background_dn"], dtype=np.float64)
    peak = np.asarray(metrics["peak_dn"], dtype=np.float64)
    active = np.asarray(metrics["active"], dtype=bool)
    half_max = background + (peak - background) * 0.5
    peak_index = np.argmax(values, axis=1)
    above = values >= half_max[:, None]
    component_count = above[:, 0].astype(np.int32) + np.sum(
        above[:, 1:] & ~above[:, :-1],
        axis=1,
    )
    contiguous = np.full(image.shape[0], np.nan, dtype=np.float64)
    interpolated = np.full(image.shape[0], np.nan, dtype=np.float64)

    for row in np.flatnonzero(active):
        centre = int(peak_index[row])
        left = centre
        while left > 0 and above[row, left - 1]:
            left -= 1
        right = centre
        while right + 1 < image.shape[1] and above[row, right + 1]:
            right += 1
        contiguous[row] = float(right - left + 1)

        if left == 0 or right + 1 >= image.shape[1]:
            continue
        level = float(half_max[row])
        left_low = float(values[row, left - 1])
        left_high = float(values[row, left])
        right_high = float(values[row, right])
        right_low = float(values[row, right + 1])
        if left_high == left_low or right_high == right_low:
            continue
        left_crossing = (left - 1) + (level - left_low) / (left_high - left_low)
        right_crossing = right + (right_high - level) / (right_high - right_low)
        interpolated[row] = right_crossing - left_crossing

    return {
        "active": active,
        "background": background,
        "peak_contrast": peak - background,
        "global_fwhm": np.asarray(metrics["fwhm_px"], dtype=np.float64),
        "contiguous_fwhm": contiguous,
        "interpolated_fwhm": interpolated,
        "component_count": component_count,
        "peak_saturated": np.asarray(metrics["peak_saturated"], dtype=bool),
        "peak_near_saturated": np.asarray(metrics["peak_near_saturated"], dtype=bool),
    }


def _steger_aligned_widths(
    image: np.ndarray,
    centres_u: np.ndarray,
    valid: np.ndarray,
    background: np.ndarray,
    *,
    half_window_px: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """只在正式 Steger 中心附近测量连续、插值半高宽。"""

    contiguous = np.full(image.shape[0], np.nan, dtype=np.float64)
    interpolated = np.full(image.shape[0], np.nan, dtype=np.float64)
    for row in np.flatnonzero(valid):
        centre = int(round(float(centres_u[row])))
        start = max(0, centre - half_window_px)
        end = min(image.shape[1], centre + half_window_px + 1)
        profile = image[row, start:end].astype(np.float64, copy=False)
        if profile.size < 3:
            continue
        peak_local = int(np.argmax(profile))
        peak = start + peak_local
        level = float(background[row]) + (
            float(image[row, peak]) - float(background[row])
        ) * 0.5
        left = peak
        while left > 0 and float(image[row, left - 1]) >= level:
            left -= 1
        right = peak
        while right + 1 < image.shape[1] and float(image[row, right + 1]) >= level:
            right += 1
        contiguous[row] = float(right - left + 1)
        if left == 0 or right + 1 >= image.shape[1]:
            continue
        left_low = float(image[row, left - 1])
        left_high = float(image[row, left])
        right_high = float(image[row, right])
        right_low = float(image[row, right + 1])
        if left_high == left_low or right_high == right_low:
            continue
        left_crossing = (left - 1) + (level - left_low) / (left_high - left_low)
        right_crossing = right + (right_high - level) / (right_high - right_low)
        interpolated[row] = right_crossing - left_crossing
    return contiguous, interpolated


def _load_steger(calibration_src: Path) -> tuple[Any, dict[str, Any]]:
    source = calibration_src.expanduser().resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("realtime_steger")
    options = dict(module.load_steger_options())
    options["scan_axis"] = "row"
    return module, module.merge_options(options)


def analyze(dataset: Path, calibration_src: Path, output_dir: Path) -> list[dict[str, float]]:
    dataset = dataset.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_document(dataset / "dataset_manifest.yaml")
    steger, steger_options = _load_steger(calibration_src)
    by_exposure: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for record in manifest["frames"]:
        exposure = int(round(float(record["tags"]["requested_exposure_us"])))
        image_path = dataset / record["filename"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"无法读取 {image_path}")
        widths = _peak_width_metrics(image, sensor_max_value=255.0)
        try:
            extraction = steger.extract_steger(image, steger_options, diagnostic=True)
        except (RuntimeError, ValueError) as error:
            valid = np.zeros(image.shape[0], dtype=bool)
            centres_u = np.full(image.shape[0], np.nan, dtype=np.float64)
            response = np.full(image.shape[0], np.nan, dtype=np.float64)
            steger_error = str(error)
        else:
            valid = np.asarray(extraction.valid, dtype=bool)
            centres_u = np.asarray(extraction.u_px, dtype=np.float64)
            response = np.asarray(extraction.response, dtype=np.float64)
            steger_error = None
        aligned_contiguous, aligned_interpolated = _steger_aligned_widths(
            image,
            centres_u,
            valid,
            widths["background"],
        )
        by_exposure[exposure].append(
            {
                "widths": widths,
                "valid": valid,
                "u_px": centres_u,
                "response": response,
                "steger_error": steger_error,
                "aligned_contiguous_fwhm": aligned_contiguous,
                "aligned_interpolated_fwhm": aligned_interpolated,
                "quality": record["quality"],
            }
        )

    rows: list[dict[str, float]] = []
    for exposure, frames in sorted(by_exposure.items()):
        active_arrays = [frame["widths"]["active"] for frame in frames]

        def pooled(name: str) -> np.ndarray:
            return np.concatenate([
                np.asarray(frame["widths"][name])[active]
                for frame, active in zip(frames, active_arrays)
            ])

        global_width = pooled("global_fwhm")
        contiguous_width = pooled("contiguous_fwhm")
        interpolated_width = pooled("interpolated_fwhm")
        components = pooled("component_count")
        contrast = pooled("peak_contrast")
        peak_saturated = pooled("peak_saturated").astype(bool)
        peak_near_saturated = pooled("peak_near_saturated").astype(bool)
        valid_stack = np.stack([frame["valid"] for frame in frames])
        centre_stack = np.stack([frame["u_px"] for frame in frames])
        common = np.all(valid_stack, axis=0)
        jitter = (
            np.std(centre_stack[:, common], axis=0)
            if np.any(common)
            else np.empty(0, dtype=np.float64)
        )
        responses = np.concatenate([
            frame["response"][frame["valid"]]
            for frame in frames
        ])
        aligned_contiguous = np.concatenate([
            frame["aligned_contiguous_fwhm"][frame["valid"]]
            for frame in frames
        ])
        aligned_interpolated = np.concatenate([
            frame["aligned_interpolated_fwhm"][frame["valid"]]
            for frame in frames
        ])
        quality_values = [frame["quality"] for frame in frames]
        row = {
            "exposure_us": float(exposure),
            "dynamic_range_u8_p50": float(np.median([
                item["dynamic_range_u8"] for item in quality_values
            ])),
            "intensity_coverage_fraction": float(np.mean([
                np.mean(active) for active in active_arrays
            ])),
            "peak_contrast_p50_dn": _percentile(contrast, 50),
            "peak_contrast_p05_dn": _percentile(contrast, 5),
            "global_fwhm_p50_px": _percentile(global_width, 50),
            "global_fwhm_p95_px": _percentile(global_width, 95),
            "contiguous_fwhm_p50_px": _percentile(contiguous_width, 50),
            "contiguous_fwhm_p95_px": _percentile(contiguous_width, 95),
            "interpolated_fwhm_p50_px": _percentile(interpolated_width, 50),
            "interpolated_fwhm_p95_px": _percentile(interpolated_width, 95),
            "steger_aligned_contiguous_fwhm_p50_px": _percentile(aligned_contiguous, 50),
            "steger_aligned_contiguous_fwhm_p95_px": _percentile(aligned_contiguous, 95),
            "steger_aligned_interpolated_fwhm_p50_px": _percentile(aligned_interpolated, 50),
            "steger_aligned_interpolated_fwhm_p95_px": _percentile(aligned_interpolated, 95),
            "multiple_halfmax_components_fraction": float(np.mean(components > 1)),
            "global_exceeds_contiguous_fraction": float(np.mean(global_width > contiguous_width)),
            "peak_saturated_fraction": float(np.mean(peak_saturated)),
            "peak_near_saturated_fraction": float(np.mean(peak_near_saturated)),
            "steger_success_fraction": float(np.mean([
                frame["steger_error"] is None for frame in frames
            ])),
            "steger_valid_fraction": float(np.mean(valid_stack)),
            "steger_common_valid_fraction": float(np.mean(common)),
            "steger_response_p50": _percentile(responses, 50),
            "steger_response_p95": _percentile(responses, 95),
            "centre_jitter_p50_px": _percentile(jitter, 50),
            "centre_jitter_p95_px": _percentile(jitter, 95),
        }
        rows.append(row)

    csv_path = output_dir / "exposure_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    exposure = np.asarray([row["exposure_us"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(exposure, [row["intensity_coverage_fraction"] * 100 for row in rows], "o-", label="intensity coverage")
    axes[0, 0].plot(exposure, [row["steger_valid_fraction"] * 100 for row in rows], "s-", label="Steger valid")
    axes[0, 0].set_ylabel("fraction (%)"); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(exposure, [row["global_fwhm_p50_px"] for row in rows], "o-", label="global count P50")
    axes[0, 1].plot(exposure, [row["contiguous_fwhm_p50_px"] for row in rows], "s-", label="contiguous P50")
    axes[0, 1].plot(exposure, [row["interpolated_fwhm_p50_px"] for row in rows], "^-", label="interpolated P50")
    axes[0, 1].plot(exposure, [row["steger_aligned_interpolated_fwhm_p50_px"] for row in rows], "d-", label="Steger-aligned P50")
    axes[0, 1].set_ylabel("FWHM (px)"); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(exposure, [row["peak_saturated_fraction"] * 100 for row in rows], "o-", label="saturated peaks")
    axes[1, 0].plot(exposure, [row["peak_near_saturated_fraction"] * 100 for row in rows], "s-", label="near-saturated peaks")
    axes[1, 0].set_ylabel("active rows (%)"); axes[1, 0].set_xlabel("exposure (us)"); axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].plot(exposure, [row["centre_jitter_p95_px"] for row in rows], "o-", label="centre jitter P95")
    axes[1, 1].set_ylabel("frame-to-frame jitter (px)"); axes[1, 1].set_xlabel("exposure (us)"); axes[1, 1].grid(True)
    figure.suptitle("Daheng vertical laser exposure characterization")
    figure.savefig(output_dir / "exposure_tradeoff.png", dpi=160)
    plt.close(figure)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--calibration-src", type=Path, default=Path("../calibration/src"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.dataset / "analysis"
    rows = analyze(args.dataset, args.calibration_src, output_dir)
    for row in rows:
        print(
            f"{row['exposure_us']:.0f} us: coverage={row['intensity_coverage_fraction']:.1%}, "
            f"FWHM global/contiguous/interpolated={row['global_fwhm_p50_px']:.2f}/"
            f"{row['contiguous_fwhm_p50_px']:.2f}/{row['interpolated_fwhm_p50_px']:.2f} px, "
            f"Steger-aligned={row['steger_aligned_interpolated_fwhm_p50_px']:.2f} px, "
            f"Steger success/valid={row['steger_success_fraction']:.1%}/"
            f"{row['steger_valid_fraction']:.1%}, "
            f"jitter P95={row['centre_jitter_p95_px']:.4f} px, "
            f"peak saturated={row['peak_saturated_fraction']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
