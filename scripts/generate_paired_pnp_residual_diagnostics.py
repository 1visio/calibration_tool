#!/usr/bin/env python3
"""Generate laser residual diagnostics against paired independent PnP planes.

The production extraction/reconstruction path is reused unchanged.  The only
reference surface for each laser frame is its same-ID chessboard PnP plane.
No laser-point plane fit, compensation, smoothing, interpolation, or formal LUT
is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_TOOL_ROOT = (
    WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
)
if str(MEASUREMENT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_TOOL_ROOT))

from app_config import load_app_config  # noqa: E402
from calibration.config_loader import load_calibration_files  # noqa: E402
from laser.backends import create_extraction_params  # noqa: E402
from laser.laser_extractor import extract_laser_center  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_DATA_ROOT = (
    CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "data" / "extrinsics0813"
)
DEFAULT_PNP_AUDIT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "pnp_reference_audit"
    / "paired_pnp_reference_audit.csv"
)
DEFAULT_MEASUREMENT_CONFIG = (
    MEASUREMENT_TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
)
DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "paired_pnp_residual_diagnostics"
)
OUTPUT_NAMES = (
    "residual_frame_v_heatmap.png",
    "residual_v_median_sigma.png",
    "residual_v_statistics.csv",
    "frame_residual_correlation.csv",
    "per_frame_residual_metrics.csv",
    "diagnostics_summary.json",
    "diagnostics_report.md",
    "OUTPUT_FILES.md",
)
IMAGE_HEIGHT = 3000
FIT_FRAME_COUNT = 10
MIN_CORRELATION_OVERLAP = 100
REGIONS = {
    "top_0_299": (0, 299),
    "middle_300_2699": (300, 2699),
    "bottom_2700_2999": (2700, 2999),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct paired lasers and diagnose residuals to PnP planes."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pnp-audit", type=Path, default=DEFAULT_PNP_AUDIT)
    parser.add_argument(
        "--measurement-config", type=Path, default=DEFAULT_MEASUREMENT_CONFIG
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"OpenCV cannot decode {path}")
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got {image.shape}: {path}")
    return image


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_pnp_planes(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            frame_id = source["frame_id"].zfill(3)
            split = source["split"].strip()
            if split not in {"fit", "validation"}:
                raise ValueError(f"Unknown split {split!r} for frame {frame_id}")
            required_flags = (
                "pnp_success",
                "pair_files_ok",
                "pair_manifest_ok",
                "chess_sha256_ok",
                "laser_sha256_ok",
            )
            failed = [name for name in required_flags if not parse_bool(source[name])]
            if failed:
                raise ValueError(f"Frame {frame_id} failed PnP/pair audit: {failed}")
            plane = np.asarray(
                [
                    float(source["plane_nx"]),
                    float(source["plane_ny"]),
                    float(source["plane_nz"]),
                    float(source["plane_d"]),
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(plane[:3]))
            if not np.all(np.isfinite(plane)) or abs(norm - 1.0) > 2.0e-5:
                raise ValueError(f"Invalid ground-frame PnP plane for frame {frame_id}")
            if abs(float(plane[2])) <= 1.0e-6:
                raise ValueError(f"PnP plane is vertical for frame {frame_id}")
            result.append(
                {
                    "frame_id": frame_id,
                    "split": split,
                    "plane": plane,
                    "tilt_deg": float(source["tilt_deg"]),
                    "pnp_rmse_px": float(source["reprojection_rmse_px"]),
                }
            )
    expected = [(f"{value:03d}", "fit" if value <= 10 else "validation") for value in range(1, 14)]
    actual = [(row["frame_id"], row["split"]) for row in result]
    if actual != expected:
        raise ValueError(f"PnP audit rows must be ordered 001..013 with fixed splits: {actual}")
    return result


def basic_metrics(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "sample_count": 0,
            "mean_mm": None,
            "median_mm": None,
            "std_mm": None,
            "mae_mm": None,
            "rmse_mm": None,
            "p95_abs_mm": None,
            "max_abs_mm": None,
            "positive_fraction": None,
        }
    return {
        "sample_count": int(finite.size),
        "mean_mm": float(np.mean(finite)),
        "median_mm": float(np.median(finite)),
        "std_mm": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "mae_mm": float(np.mean(np.abs(finite))),
        "rmse_mm": float(np.sqrt(np.mean(finite**2))),
        "p95_abs_mm": float(np.percentile(np.abs(finite), 95)),
        "max_abs_mm": float(np.max(np.abs(finite))),
        "positive_fraction": float(np.mean(finite > 0.0)),
    }


def bin_frame_by_v(v_px: np.ndarray, residual: np.ndarray) -> np.ndarray:
    matrix_row = np.full(IMAGE_HEIGHT, np.nan, dtype=np.float64)
    rounded = np.rint(v_px).astype(int)
    valid = (
        np.isfinite(v_px)
        & np.isfinite(residual)
        & (rounded >= 0)
        & (rounded < IMAGE_HEIGHT)
    )
    for row_index in np.unique(rounded[valid]):
        selected = valid & (rounded == row_index)
        matrix_row[row_index] = float(np.median(residual[selected]))
    return matrix_row


def calculate_v_statistics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    width = matrix.shape[1]
    result = {
        "count": np.sum(np.isfinite(matrix), axis=0).astype(int),
        "median": np.full(width, np.nan),
        "mean": np.full(width, np.nan),
        "std": np.full(width, np.nan),
        "mad": np.full(width, np.nan),
        "p95_abs": np.full(width, np.nan),
        "positive_fraction": np.full(width, np.nan),
        "sign_consistency": np.full(width, np.nan),
    }
    for column in range(width):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        median = float(np.median(values))
        positive_fraction = float(np.mean(values > 0.0))
        result["median"][column] = median
        result["mean"][column] = float(np.mean(values))
        result["std"][column] = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        result["mad"][column] = float(np.median(np.abs(values - median)))
        result["p95_abs"][column] = float(np.percentile(np.abs(values), 95))
        result["positive_fraction"][column] = positive_fraction
        result["sign_consistency"][column] = max(
            positive_fraction, 1.0 - positive_fraction
        )
    return result


def explained_energy(matrix: np.ndarray, profile: np.ndarray) -> dict[str, Any]:
    predicted = np.broadcast_to(profile, matrix.shape)
    comparable = np.isfinite(matrix) & np.isfinite(predicted)
    count = int(np.count_nonzero(comparable))
    if count == 0:
        return {
            "comparable_bin_count": 0,
            "total_residual_energy_mm2": None,
            "unexplained_energy_mm2": None,
            "explained_fraction": None,
        }
    total = float(np.sum(matrix[comparable] ** 2))
    unexplained = float(np.sum((matrix[comparable] - predicted[comparable]) ** 2))
    return {
        "comparable_bin_count": count,
        "total_residual_energy_mm2": total,
        "unexplained_energy_mm2": unexplained,
        "explained_fraction": 1.0 - unexplained / total if total > 0.0 else None,
    }


def correlations(
    frame_ids: list[str], split: str, matrix: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    retained: list[float] = []
    for left_index in range(len(frame_ids)):
        for right_index in range(left_index + 1, len(frame_ids)):
            common = np.isfinite(matrix[left_index]) & np.isfinite(matrix[right_index])
            common_count = int(np.count_nonzero(common))
            coefficient = float("nan")
            if common_count >= 3:
                left = matrix[left_index, common]
                right = matrix[right_index, common]
                if np.std(left) > 0.0 and np.std(right) > 0.0:
                    coefficient = float(np.corrcoef(left, right)[0, 1])
            eligible = common_count >= MIN_CORRELATION_OVERLAP and math.isfinite(coefficient)
            if eligible:
                retained.append(coefficient)
            rows.append(
                {
                    "split": split,
                    "frame_i": frame_ids[left_index],
                    "frame_j": frame_ids[right_index],
                    "common_sample_count": common_count,
                    "correlation_coefficient": coefficient,
                    "eligible_overlap_ge_100": eligible,
                }
            )
    return rows, np.asarray(retained, dtype=np.float64)


def correlation_summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "eligible_pair_count": 0,
            "median": None,
            "p10": None,
            "p90": None,
            "fraction_positive": None,
            "fraction_ge_0_5": None,
        }
    return {
        "eligible_pair_count": int(finite.size),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "fraction_positive": float(np.mean(finite > 0.0)),
        "fraction_ge_0_5": float(np.mean(finite >= 0.5)),
    }


def support_summary(statistics: Mapping[str, np.ndarray], frame_count: int) -> dict[str, Any]:
    count = statistics["count"]
    observed = count > 0
    at_least_half = count >= math.ceil(frame_count / 2)
    all_frames = count == frame_count
    std = statistics["std"]
    sign = statistics["sign_consistency"]
    return {
        "rows_observed": int(np.count_nonzero(observed)),
        "rows_with_at_least_2_frames": int(np.count_nonzero(count >= 2)),
        "rows_with_at_least_half_frames": int(np.count_nonzero(at_least_half)),
        "rows_with_all_frames": int(np.count_nonzero(all_frames)),
        "max_sample_count": int(np.max(count)),
        "image_fraction_with_at_least_half_frames": float(
            np.count_nonzero(at_least_half) / IMAGE_HEIGHT
        ),
        "median_std_mm_at_least_half_frames": (
            float(np.nanmedian(std[at_least_half])) if np.any(at_least_half) else None
        ),
        "median_sign_consistency_at_least_half_frames": (
            float(np.nanmedian(sign[at_least_half])) if np.any(at_least_half) else None
        ),
        "fraction_sign_consistency_ge_0_8_at_least_half_frames": (
            float(np.mean(sign[at_least_half] >= 0.8)) if np.any(at_least_half) else None
        ),
    }


def region_summaries(
    matrix: np.ndarray, statistics: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    v = np.arange(IMAGE_HEIGHT)
    for name, (low, high) in REGIONS.items():
        columns = (v >= low) & (v <= high)
        row_supported = columns & (statistics["count"] >= 2)
        point_values = matrix[:, columns]
        point_values = point_values[np.isfinite(point_values)]
        result[name] = {
            "v_range": [low, high],
            "residual_bins": basic_metrics(point_values),
            "row_count_with_at_least_2_frames": int(np.count_nonzero(row_supported)),
            "median_row_sample_count": (
                float(np.median(statistics["count"][row_supported]))
                if np.any(row_supported)
                else None
            ),
            "median_residual_std_v_mm": (
                float(np.nanmedian(statistics["std"][row_supported]))
                if np.any(row_supported)
                else None
            ),
            "median_residual_mad_v_mm": (
                float(np.nanmedian(statistics["mad"][row_supported]))
                if np.any(row_supported)
                else None
            ),
            "median_sign_consistency": (
                float(np.nanmedian(statistics["sign_consistency"][row_supported]))
                if np.any(row_supported)
                else None
            ),
            "median_abs_diagnostic_bias_mm": (
                float(np.nanmedian(np.abs(statistics["median"][row_supported])))
                if np.any(row_supported)
                else None
            ),
        }
    return result


def classify_one_dimensional_bv(
    fit_corr: Mapping[str, Any],
    fit_energy: Mapping[str, Any],
    validation_energy: Mapping[str, Any],
    fit_support: Mapping[str, Any],
) -> tuple[str, list[str]]:
    corr = fit_corr.get("median")
    explained = fit_energy.get("explained_fraction")
    validation_explained = validation_energy.get("explained_fraction")
    coverage = fit_support.get("image_fraction_with_at_least_half_frames")
    sign = fit_support.get("median_sign_consistency_at_least_half_frames")
    strong_checks = {
        "fit median pair correlation >= 0.7": corr is not None and corr >= 0.7,
        "fit explained energy >= 0.7": explained is not None and explained >= 0.7,
        "validation explained energy using fit profile >= 0.5": (
            validation_explained is not None and validation_explained >= 0.5
        ),
        "fit >=half-frame coverage >= 50% of image rows": (
            coverage is not None and coverage >= 0.5
        ),
        "fit median sign consistency >= 0.8": sign is not None and sign >= 0.8,
    }
    if all(strong_checks.values()):
        verdict = "SUPPORTED"
    elif (
        corr is not None
        and corr >= 0.3
        and explained is not None
        and explained >= 0.3
        and validation_explained is not None
        and validation_explained > 0.0
        and coverage is not None
        and coverage >= 0.25
    ):
        verdict = "PARTIAL"
    else:
        verdict = "NOT_SUPPORTED"
    reasons = [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in strong_checks.items()]
    return verdict, reasons


def csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.9f}"
    return value


def write_v_statistics(
    path: Path,
    fit: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
) -> None:
    fields = ["v_px"]
    keys = (
        "count",
        "median",
        "mean",
        "std",
        "mad",
        "p95_abs",
        "positive_fraction",
        "sign_consistency",
    )
    labels = {
        "count": "sample_count",
        "median": "residual_median_mm",
        "mean": "residual_mean_mm",
        "std": "residual_std_mm",
        "mad": "residual_mad_mm",
        "p95_abs": "residual_p95_abs_mm",
        "positive_fraction": "positive_fraction",
        "sign_consistency": "sign_consistency",
    }
    for split in ("fit", "validation"):
        fields.extend(f"{split}_{labels[key]}" for key in keys)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index in range(IMAGE_HEIGHT):
            row: dict[str, Any] = {"v_px": row_index}
            for split, stats in (("fit", fit), ("validation", validation)):
                for key in keys:
                    row[f"{split}_{labels[key]}"] = csv_value(stats[key][row_index])
            writer.writerow(row)


def write_correlations(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "split",
        "frame_i",
        "frame_j",
        "common_sample_count",
        "correlation_coefficient",
        "eligible_overlap_ge_100",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row[key]) for key in fields})


def write_per_frame_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "frame_id",
        "split",
        "laser_file",
        "extracted_center_count",
        "reconstructed_point_count",
        "residual_sample_count",
        "residual_mean_mm",
        "residual_median_mm",
        "residual_std_mm",
        "residual_mae_mm",
        "residual_rmse_mm",
        "residual_p95_abs_mm",
        "residual_max_abs_mm",
        "positive_fraction",
        "pnp_reprojection_rmse_px",
        "pnp_tilt_deg",
        "plane_nx",
        "plane_ny",
        "plane_nz",
        "plane_d_mm",
        "filtered_no_valid_intersection",
        "filtered_outside_working_distance",
        "filtered_non_finite",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def save_heatmap(path: Path, frame_ids: list[str], matrix: np.ndarray) -> float:
    finite = np.abs(matrix[np.isfinite(matrix)])
    color_limit = float(np.percentile(finite, 99))
    figure, axis = plt.subplots(figsize=(15, 6.5))
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        extent=[-0.5, IMAGE_HEIGHT - 0.5, len(frame_ids) + 0.5, 0.5],
    )
    axis.axhline(FIT_FRAME_COUNT + 0.5, color="black", linewidth=1.3)
    axis.text(
        IMAGE_HEIGHT - 8,
        FIT_FRAME_COUNT + 0.25,
        "fit / validation",
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
    )
    axis.set_xlabel("Image row v (px)")
    axis.set_ylabel("Paired frame ID")
    axis.set_title("Signed vertical residual to same-ID independent PnP plane")
    axis.set_yticks(np.arange(1, len(frame_ids) + 1))
    axis.set_yticklabels(frame_ids)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("residual_z (mm), color clipped at P99 |residual|")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return color_limit


def save_median_sigma(
    path: Path,
    fit: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
) -> None:
    v = np.arange(IMAGE_HEIGHT)
    fit_valid = fit["count"] > 0
    validation_valid = validation["count"] > 0
    figure, residual_axis = plt.subplots(figsize=(15, 6.5))
    residual_axis.plot(
        v[fit_valid],
        fit["median"][fit_valid],
        color="#174a7e",
        linewidth=1.0,
        label="fit-only diagnostic median b(v)",
    )
    residual_axis.fill_between(
        v[fit_valid],
        fit["median"][fit_valid] - fit["std"][fit_valid],
        fit["median"][fit_valid] + fit["std"][fit_valid],
        color="#4c9f70",
        alpha=0.24,
        label="fit median ± residual std(v)",
    )
    residual_axis.plot(
        v[validation_valid],
        validation["median"][validation_valid],
        color="#f97316",
        linewidth=0.9,
        alpha=0.85,
        label="validation median (observation only)",
    )
    residual_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    residual_axis.set_xlabel("Image row v (px)")
    residual_axis.set_ylabel("Signed residual_z (mm)")
    residual_axis.grid(True, alpha=0.22)
    count_axis = residual_axis.twinx()
    count_axis.plot(
        v[fit_valid],
        fit["count"][fit_valid],
        color="#dc2626",
        linewidth=0.7,
        alpha=0.65,
        label="fit frame count",
    )
    count_axis.plot(
        v[validation_valid],
        validation["count"][validation_valid],
        color="#7c3aed",
        linewidth=0.7,
        alpha=0.65,
        label="validation frame count",
    )
    count_axis.set_ylabel("Frame sample count")
    count_axis.set_ylim(0, FIT_FRAME_COUNT + 1)
    lines_a, labels_a = residual_axis.get_legend_handles_labels()
    lines_b, labels_b = count_axis.get_legend_handles_labels()
    residual_axis.legend(
        lines_a + lines_b, labels_a + labels_b, loc="upper right", ncol=2
    )
    residual_axis.set_title(
        "Fit-only diagnostic median profile, residual dispersion, and validation observation"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "N/A"


def render_report(summary: Mapping[str, Any], per_frame: list[dict[str, Any]]) -> str:
    fit = summary["splits"]["fit"]
    validation = summary["splits"]["validation"]
    verdict = summary["ONE_DIMENSIONAL_BV"]
    lines = [
        "# Paired PnP laser residual diagnostics",
        "",
        "## 结论",
        "",
        f"**ONE_DIMENSIONAL_BV = {verdict}**",
        "",
        "本结论只评价一维 `v` 相关残差是否具有重复性；没有构建正式 LUT，也没有应用 compensation。",
        "每帧 reference plane 唯一来自同编号 chess 的独立 PnP，禁止且未执行激光点自拟合平面。",
        "",
        "## 关键统计",
        "",
        f"- fit frame pair correlation median（overlap ≥ {MIN_CORRELATION_OVERLAP}）："
        f"{fmt(fit['pair_correlation_overlap_ge_100']['median'])} "
        f"（{fit['pair_correlation_overlap_ge_100']['eligible_pair_count']} 对）。",
        f"- fit median `b(v)` explained residual energy："
        f"{fmt(fit['median_bv_explained_energy']['explained_fraction'])}。",
        f"- validation 对冻结 fit median profile 的 observed explained energy："
        f"{fmt(validation['fit_profile_observed_explained_energy']['explained_fraction'])}；"
        "validation 未参与 profile 计算。",
        f"- fit residual std(v) median（sample count ≥ 5）："
        f"{fmt(fit['support']['median_std_mm_at_least_half_frames'])} mm。",
        f"- fit sign consistency median（sample count ≥ 5）："
        f"{fmt(fit['support']['median_sign_consistency_at_least_half_frames'])}。",
        f"- residual bin sample count：fit={fit['residual_bins']['sample_count']}，"
        f"validation={validation['residual_bins']['sample_count']}。",
        "",
        "## 判定规则",
        "",
        "`SUPPORTED` 要求 fit median pair correlation ≥0.7、fit explained energy ≥0.7、"
        "validation observed explained energy ≥0.5、fit 至少半数帧覆盖 ≥50% 图像行、"
        "且对应行的 median sign consistency ≥0.8。若未全部满足，但 fit correlation 与 "
        "explained energy 均 ≥0.3、validation explained energy 为正且覆盖 ≥25%，判为 "
        "`PARTIAL`；否则 `NOT_SUPPORTED`。",
    ]
    lines.extend(f"- {reason}" for reason in summary["verdict_checks"])
    lines.extend(
        [
            "",
            "当前数据存在明确的一维共同成分，但强支持门槛未通过：fit profile 只解释约56% "
            "energy，validation 只保留约36%；且 fit top/bottom 的 median std(v) 明显高于 middle，"
            "帧间整体偏置也在变化。因此 `PARTIAL` 表示可继续研究的一维成分，不表示足以发布 LUT。",
        ]
    )
    lines.extend(
        [
            "",
            "## 分 split 统计",
            "",
            "| split | frames | residual samples | mean (mm) | median (mm) | std (mm) | RMSE (mm) | P95 abs (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in ("fit", "validation"):
        item = summary["splits"][split_name]
        metrics = item["residual_bins"]
        lines.append(
            f"| {split_name} | {item['frame_count']} | {metrics['sample_count']} | "
            f"{fmt(metrics['mean_mm'])} | {fmt(metrics['median_mm'])} | "
            f"{fmt(metrics['std_mm'])} | {fmt(metrics['rmse_mm'])} | "
            f"{fmt(metrics['p95_abs_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## Top / middle / bottom",
            "",
            "以下按 3000 px 图像高度固定划分 top 10%、middle 80%、bottom 10%。"
            "`std(v)`、sign consistency 与 diagnostic bias 只汇总至少有2帧样本的行。",
            "",
            "| split | region | residual bins | row count ≥2 | median samples/row | median std(v) mm | median sign consistency | median abs bias mm |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in ("fit", "validation"):
        for region_name, region in summary["splits"][split_name]["regions"].items():
            lines.append(
                f"| {split_name} | {region_name} | {region['residual_bins']['sample_count']} | "
                f"{region['row_count_with_at_least_2_frames']} | "
                f"{fmt(region['median_row_sample_count'], 2)} | "
                f"{fmt(region['median_residual_std_v_mm'])} | "
                f"{fmt(region['median_sign_consistency'])} | "
                f"{fmt(region['median_abs_diagnostic_bias_mm'])} |"
            )
    lines.extend(
        [
            "",
            "## 逐帧 residual",
            "",
            "| frame | split | extracted | reconstructed | mean mm | std mm | RMSE mm | P95 abs mm |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_frame:
        lines.append(
            f"| {row['frame_id']} | {row['split']} | {row['extracted_center_count']} | "
            f"{row['reconstructed_point_count']} | {fmt(row['residual_mean_mm'])} | "
            f"{fmt(row['residual_std_mm'])} | {fmt(row['residual_rmse_mm'])} | "
            f"{fmt(row['residual_p95_abs_mm'])} |"
        )
    lines.extend(
        [
            "",
            "## 冻结链路与 residual 定义",
            "",
            f"- Measurement config：`{summary['provenance']['measurement_config']['path']}`",
            f"- PnP audit：`{summary['provenance']['pnp_audit']['path']}`",
            f"- Laser model：`{summary['frozen_settings']['laser_model_type']}`。",
            f"- Steger：`{json.dumps(summary['frozen_settings']['steger'], ensure_ascii=False)}`。",
            f"- Reconstruction：`{json.dumps(summary['frozen_settings']['reconstruction'], ensure_ascii=False)}`。",
            "- Ground compensation 配置为 `null`，本轮没有 compensation。",
            "- 对 ground-frame PnP 平面 `nx Xg + ny Yg + nz Zg + d = 0`，"
            "逐点计算 `Z_plane=-(nx Xg+ny Yg+d)/nz`，再计算 "
            "`residual_z=Zg-Z_plane`。这是 ground Z 方向的有符号误差，不是正交点面距离。",
            "- `residual_v_statistics.csv` 中的 fit median 是未平滑、未插值的诊断统计，"
            "不得直接当作正式 LUT。",
            "",
        ]
    )
    return "\n".join(lines)


def render_output_files() -> str:
    return """# OUTPUT_FILES

本目录只包含 paired-PnP residual 诊断，不包含正式 LUT 或任何 compensation 产物。

- `residual_frame_v_heatmap.png`：13帧 residual_z 的 frame×v 热图；fit/validation 由横线分隔。
- `residual_v_median_sigma.png`：fit-only 诊断 median b(v)、fit std(v)、validation 独立观察和样本数。
- `residual_v_statistics.csv`：每个 v 的 fit/validation 分离统计；fit median 不是正式 LUT。
- `frame_residual_correlation.csv`：split 内无重复帧对相关性及共同样本数。
- `per_frame_residual_metrics.csv`：逐帧提取、重建、PnP plane 与 residual 指标。
- `diagnostics_summary.json`：机器可读的冻结参数、provenance、支持度、相关性、能量和区域统计。
- `diagnostics_report.md`：结论、判定规则和关键表格。
- `OUTPUT_FILES.md`：本文件。

明确未执行：激光点 reference-plane 自拟合、median profile 平滑/插值、正式 LUT 构建、compensation。
"""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    pnp_audit_path = args.pnp_audit.resolve()
    measurement_config_path = args.measurement_config.resolve()
    output_dir = args.output_dir.resolve()
    outputs = [output_dir / name for name in OUTPUT_NAMES]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite: " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    planes = load_pnp_planes(pnp_audit_path)
    app_config = load_app_config(measurement_config_path)
    calibration = load_calibration_files(
        app_config.calibration.intrinsics,
        app_config.calibration.laser_model,
        app_config.calibration.extrinsics,
        app_config.calibration.ground_u_compensation,
    )
    if calibration["laser_model"]["model_type"] != "circular_cone":
        raise ValueError("Frozen laser model must be circular_cone")
    if calibration["ground_u_compensation"] is not None:
        raise ValueError("This diagnostic forbids compensation")
    extraction_params = create_extraction_params(
        app_config.extraction_method, app_config.extraction_options
    )

    matrix_rows: list[np.ndarray] = []
    raw_residuals: dict[str, list[np.ndarray]] = {"fit": [], "validation": []}
    per_frame: list[dict[str, Any]] = []
    for plane_row in planes:
        frame_id = plane_row["frame_id"]
        split = plane_row["split"]
        laser_path = data_root / split / f"laser {frame_id}.tif"
        image = read_gray(laser_path)
        centers = extract_laser_center(image, extraction_params)
        reconstructed = reconstruct_uv_to_ground(
            centers, calibration, app_config.reconstruction
        )
        ground = reconstructed.points_ground
        plane = plane_row["plane"]
        z_plane = -(
            plane[0] * ground[:, 0] + plane[1] * ground[:, 1] + plane[3]
        ) / plane[2]
        residual = ground[:, 2] - z_plane
        if not np.all(np.isfinite(residual)):
            raise ValueError(f"Non-finite residual for frame {frame_id}")
        matrix_rows.append(bin_frame_by_v(reconstructed.pixels_uv[:, 1], residual))
        raw_residuals[split].append(residual)
        metrics = basic_metrics(residual)
        filtered = reconstructed.filtered
        per_frame.append(
            {
                "frame_id": frame_id,
                "split": split,
                "laser_file": str(laser_path),
                "extracted_center_count": int(len(centers)),
                "reconstructed_point_count": reconstructed.point_count,
                "residual_sample_count": metrics["sample_count"],
                "residual_mean_mm": metrics["mean_mm"],
                "residual_median_mm": metrics["median_mm"],
                "residual_std_mm": metrics["std_mm"],
                "residual_mae_mm": metrics["mae_mm"],
                "residual_rmse_mm": metrics["rmse_mm"],
                "residual_p95_abs_mm": metrics["p95_abs_mm"],
                "residual_max_abs_mm": metrics["max_abs_mm"],
                "positive_fraction": metrics["positive_fraction"],
                "pnp_reprojection_rmse_px": plane_row["pnp_rmse_px"],
                "pnp_tilt_deg": plane_row["tilt_deg"],
                "plane_nx": float(plane[0]),
                "plane_ny": float(plane[1]),
                "plane_nz": float(plane[2]),
                "plane_d_mm": float(plane[3]),
                "filtered_no_valid_intersection": filtered["no_valid_intersection"],
                "filtered_outside_working_distance": filtered[
                    "outside_working_distance"
                ],
                "filtered_non_finite": filtered["non_finite"],
            }
        )

    matrix = np.vstack(matrix_rows)
    fit_matrix = matrix[:FIT_FRAME_COUNT]
    validation_matrix = matrix[FIT_FRAME_COUNT:]
    fit_stats = calculate_v_statistics(fit_matrix)
    validation_stats = calculate_v_statistics(validation_matrix)
    diagnostic_bv = fit_stats["median"].copy()
    fit_energy = explained_energy(fit_matrix, diagnostic_bv)
    validation_energy = explained_energy(validation_matrix, diagnostic_bv)

    fit_corr_rows, fit_corr_values = correlations(
        [row["frame_id"] for row in planes[:FIT_FRAME_COUNT]], "fit", fit_matrix
    )
    validation_corr_rows, validation_corr_values = correlations(
        [row["frame_id"] for row in planes[FIT_FRAME_COUNT:]],
        "validation",
        validation_matrix,
    )
    fit_corr_summary = correlation_summary(fit_corr_values)
    validation_corr_summary = correlation_summary(validation_corr_values)
    fit_support = support_summary(fit_stats, FIT_FRAME_COUNT)
    validation_support = support_summary(
        validation_stats, len(planes) - FIT_FRAME_COUNT
    )
    verdict, verdict_checks = classify_one_dimensional_bv(
        fit_corr_summary, fit_energy, validation_energy, fit_support
    )

    color_limit = save_heatmap(outputs[0], [row["frame_id"] for row in planes], matrix)
    save_median_sigma(outputs[1], fit_stats, validation_stats)
    write_v_statistics(outputs[2], fit_stats, validation_stats)
    write_correlations(outputs[3], fit_corr_rows + validation_corr_rows)
    write_per_frame_metrics(outputs[4], per_frame)

    fit_all = np.concatenate(raw_residuals["fit"])
    validation_all = np.concatenate(raw_residuals["validation"])
    summary = {
        "schema_version": 1,
        "ONE_DIMENSIONAL_BV": verdict,
        "verdict_checks": verdict_checks,
        "scope": {
            "reference_plane": "same-ID independent chess PnP plane in ground frame",
            "laser_point_plane_fit_used": False,
            "compensation_applied": False,
            "formal_lut_built": False,
            "median_profile_smoothed": False,
            "median_profile_interpolated": False,
            "validation_participates_in_fit_median_bv": False,
        },
        "residual_definition": (
            "Z_plane=-(plane_nx*Xg+plane_ny*Yg+plane_d)/plane_nz; "
            "residual_z=Zg-Z_plane"
        ),
        "binning": {
            "axis": "v",
            "bin_width_px": 1,
            "within_frame_aggregation": "median",
            "v_range_px": [0, IMAGE_HEIGHT - 1],
        },
        "frozen_settings": {
            "camera_intrinsics": str(app_config.calibration.intrinsics),
            "laser_model": str(app_config.calibration.laser_model),
            "laser_model_type": calibration["laser_model"]["model_type"],
            "ground_extrinsics": str(app_config.calibration.extrinsics),
            "ground_u_compensation": None,
            "steger_method": app_config.extraction_method,
            "steger": dict(app_config.extraction_options),
            "reconstruction": asdict(app_config.reconstruction),
        },
        "provenance": {
            "data_root": str(data_root),
            "measurement_config": {
                "path": str(measurement_config_path),
                "sha256": sha256_file(measurement_config_path),
            },
            "pnp_audit": {
                "path": str(pnp_audit_path),
                "sha256": sha256_file(pnp_audit_path),
            },
            "intrinsics_sha256": sha256_file(app_config.calibration.intrinsics),
            "laser_model_sha256": sha256_file(app_config.calibration.laser_model),
            "ground_extrinsics_sha256": sha256_file(app_config.calibration.extrinsics),
        },
        "splits": {
            "fit": {
                "frame_count": FIT_FRAME_COUNT,
                "frame_ids": [row["frame_id"] for row in planes[:FIT_FRAME_COUNT]],
                "residual_bins": basic_metrics(fit_all),
                "pair_correlation_overlap_ge_100": fit_corr_summary,
                "median_bv_explained_energy": fit_energy,
                "support": fit_support,
                "regions": region_summaries(fit_matrix, fit_stats),
            },
            "validation": {
                "frame_count": len(planes) - FIT_FRAME_COUNT,
                "frame_ids": [row["frame_id"] for row in planes[FIT_FRAME_COUNT:]],
                "residual_bins": basic_metrics(validation_all),
                "pair_correlation_overlap_ge_100": validation_corr_summary,
                "fit_profile_observed_explained_energy": validation_energy,
                "support": validation_support,
                "regions": region_summaries(validation_matrix, validation_stats),
            },
        },
        "heatmap_color_limit_p99_abs_mm": color_limit,
    }
    safe_summary = json_safe(summary)
    outputs[5].write_text(
        json.dumps(safe_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    outputs[6].write_text(render_report(safe_summary, per_frame), encoding="utf-8")
    outputs[7].write_text(render_output_files(), encoding="utf-8")

    print(f"ONE_DIMENSIONAL_BV={verdict}")
    print(
        "fit median pair correlation="
        f"{fmt(fit_corr_summary['median'])}, fit explained energy="
        f"{fmt(fit_energy['explained_fraction'])}"
    )
    print(
        "validation observed explained energy="
        f"{fmt(validation_energy['explained_fraction'])}"
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
