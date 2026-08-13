from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate_diagnostics as base


OUTPUT_DIR = Path(__file__).resolve().parent
MODE_OUTPUT_ROOT = OUTPUT_DIR / "reference_modes"
MIN_CORRELATION_OVERLAP = 100
MIN_WELL_SUPPORTED_FRAMES = 10
REGIONS = {
    "top": (0, 299),
    "middle": (300, 2699),
    "bottom": (2700, 2999),
}
MODE_LABELS = {
    "self_fitted": "A  self_fitted",
    "fixed_normal_per_frame_offset": "B  fixed_normal_per_frame_offset",
    "fixed_ground_plane": "C  fixed_ground_plane",
}
MODE_FORMULAS = {
    "self_fitted": "r_i = Zg - (a_i*Xg + b_i*Yg + c_i)",
    "fixed_normal_per_frame_offset": "r_i = Zg - median(Zg_i)",
    "fixed_ground_plane": "r_i = Zg - Z0",
}


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def write_correlations(
    output_dir: Path, frame_ids: list[str], matrix: np.ndarray
) -> np.ndarray:
    rows: list[tuple[str, str, int, float]] = []
    retained_unique: list[float] = []
    for left_index, left_id in enumerate(frame_ids):
        for right_index, right_id in enumerate(frame_ids):
            common = np.isfinite(matrix[left_index]) & np.isfinite(matrix[right_index])
            common_count = int(np.count_nonzero(common))
            correlation = _finite_correlation(
                matrix[left_index, common], matrix[right_index, common]
            )
            rows.append((left_id, right_id, common_count, correlation))
            if (
                left_index < right_index
                and common_count >= MIN_CORRELATION_OVERLAP
                and np.isfinite(correlation)
            ):
                retained_unique.append(correlation)
    with (output_dir / "frame_residual_correlation.csv").open(
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
    return np.asarray(retained_unique, dtype=np.float64)


def write_statistics(
    output_dir: Path,
    v: np.ndarray,
    statistics: tuple[np.ndarray, ...],
) -> None:
    count, median, mean, std, mad, p95, _, _ = statistics
    with (output_dir / "residual_v_statistics.csv").open(
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


def save_heatmap(
    output_dir: Path,
    mode: str,
    frame_ids: list[str],
    matrix: np.ndarray,
    v_min: int,
    v_max: int,
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
    axis.set_title(f"Signed residual(v) by frame — {mode}")
    axis.set_yticks(np.arange(1, len(frame_ids) + 1))
    axis.set_yticklabels(frame_ids, fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label(
        "Signed vertical residual (mm), clipped at this mode's 99th |residual| percentile"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "residual_frame_v_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    return color_limit


def save_median_sigma(
    output_dir: Path,
    mode: str,
    v: np.ndarray,
    statistics: tuple[np.ndarray, ...],
) -> None:
    count, median, _, std, _, _, _, _ = statistics
    valid = count > 0
    figure, residual_axis = plt.subplots(figsize=(15, 6.5))
    residual_axis.plot(
        v[valid], median[valid], color="#174a7e", linewidth=1.0, label="Median residual(v)"
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
    residual_axis.set_ylabel("Signed vertical residual (mm)")
    residual_axis.grid(True, alpha=0.22)
    count_axis = residual_axis.twinx()
    count_axis.plot(v[valid], count[valid], color="#d65f5f", linewidth=0.85, label="Sample count")
    count_axis.set_ylabel("Frame sample count")
    count_axis.set_ylim(0, base.FRAME_COUNT + 2)
    lines_a, labels_a = residual_axis.get_legend_handles_labels()
    lines_b, labels_b = count_axis.get_legend_handles_labels()
    residual_axis.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", ncol=3)
    residual_axis.set_title(f"Across-frame residual(v), dispersion, and support — {mode}")
    figure.tight_layout()
    figure.savefig(output_dir / "residual_v_median_sigma.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def comparison_metrics(
    matrix: np.ndarray,
    v: np.ndarray,
    statistics: tuple[np.ndarray, ...],
    pair_correlations: np.ndarray,
) -> dict[str, float | int]:
    count, median, _, std, _, _, positive_fraction, _ = statistics
    well_supported = count >= MIN_WELL_SUPPORTED_FRAMES
    predicted = np.broadcast_to(median, matrix.shape)
    comparable = np.isfinite(matrix) & np.isfinite(predicted)
    total_energy = float(np.sum(matrix[comparable] ** 2))
    unexplained_energy = float(np.sum((matrix[comparable] - predicted[comparable]) ** 2))
    sign_mixing = well_supported & (positive_fraction >= 0.35) & (positive_fraction <= 0.65)
    result: dict[str, float | int] = {
        "median_pair_correlation": float(np.median(pair_correlations)),
        "fraction_pair_corr_ge_0.5": float(np.mean(pair_correlations >= 0.5)),
        "explained_residual_energy": (
            float(1.0 - unexplained_energy / total_energy)
            if total_energy > 0.0
            else float("nan")
        ),
        "median_residual_std_mm": float(np.nanmedian(std[well_supported])),
        "median_abs_bias_mm": float(np.nanmedian(np.abs(median[well_supported]))),
        "sign_mixing_fraction": float(np.mean(sign_mixing[well_supported])),
    }
    for region_name, (low, high) in REGIONS.items():
        selected = (v >= low) & (v <= high) & (count >= 2)
        result[f"{region_name}_region_std"] = float(np.nanmedian(std[selected]))
    result["well_supported_row_count"] = int(np.count_nonzero(well_supported))
    result["retained_pair_count"] = int(pair_correlations.size)
    return result


def analyze_mode(
    mode: str,
    frame_ids: list[str],
    matrix: np.ndarray,
    v: np.ndarray,
    v_min: int,
    v_max: int,
    z0_mm: float,
    z0_source: dict[str, object],
) -> dict[str, Any]:
    output_dir = MODE_OUTPUT_ROOT / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = base.calculate_statistics(matrix)
    write_statistics(output_dir, v, statistics)
    pair_correlations = write_correlations(output_dir, frame_ids, matrix)
    color_limit = save_heatmap(output_dir, mode, frame_ids, matrix, v_min, v_max)
    save_median_sigma(output_dir, mode, v, statistics)
    metrics = comparison_metrics(matrix, v, statistics, pair_correlations)
    count = statistics[0]
    summary: dict[str, Any] = {
        "reference_plane_mode": mode,
        "residual_formula": MODE_FORMULAS[mode],
        "input_dir": str(base.INPUT_DIR),
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "compensation_applied": False,
        "smooth_window_applied": False,
        "signed_residual_definition": "vertical ground-frame residual in mm, not orthogonal point-to-plane distance",
        "v_range": [v_min, v_max],
        "support": {
            "rows_observed": int(np.count_nonzero(count > 0)),
            "rows_with_at_least_10_frames": int(np.count_nonzero(count >= 10)),
            "max_sample_count": int(np.max(count)),
        },
        "metric_definitions": {
            "pair_correlation": f"unique frame pairs with common v support >= {MIN_CORRELATION_OVERLAP}",
            "explained_residual_energy": "1 - sum((r-median_profile)^2)/sum(r^2) over observed bins",
            "median_residual_std_mm": f"median per-v cross-frame std for sample_count >= {MIN_WELL_SUPPORTED_FRAMES}",
            "median_abs_bias_mm": f"median |per-v residual median| for sample_count >= {MIN_WELL_SUPPORTED_FRAMES}",
            "sign_mixing_fraction": "fraction of well-supported rows whose positive-residual fraction is in [0.35, 0.65]",
            "region_std": "median per-v cross-frame std in fixed image bands for sample_count >= 2",
        },
        "comparison_metrics": metrics,
        "heatmap_color_limit_mm": color_limit,
        "apparent_tilt_warning": "self-fit apparent tilt is derived from narrow-band reconstructed points and is not checkerboard mechanical tilt",
    }
    if mode == "fixed_ground_plane":
        summary["fixed_ground_z0_mm"] = z0_mm
        summary["fixed_ground_z0_source"] = z0_source
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def repositioning_summary(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    offsets = np.asarray([float(row["offset_from_global_Z0_mm"]) for row in rows])
    tilts = np.asarray([float(row["apparent_tilt_deg"]) for row in rows])
    conditions = np.asarray([float(row["self_fit_condition_number"]) for row in rows])
    return {
        "offset_min_mm": float(np.min(offsets)),
        "offset_max_mm": float(np.max(offsets)),
        "offset_range_mm": float(np.ptp(offsets)),
        "offset_std_mm": float(np.std(offsets, ddof=1)),
        "offset_p95_abs_mm": float(np.percentile(np.abs(offsets), 95)),
        "apparent_tilt_median_deg": float(np.median(tilts)),
        "apparent_tilt_p95_deg": float(np.percentile(tilts, 95)),
        "apparent_tilt_max_deg": float(np.max(tilts)),
        "tilt_condition_pearson_r": _finite_correlation(tilts, conditions),
        "tilt_log10_condition_pearson_r": _finite_correlation(tilts, np.log10(conditions)),
        "tilt_condition_spearman_r": _finite_correlation(_rank(tilts), _rank(conditions)),
    }


def save_comparison_plot(rows: list[dict[str, Any]]) -> None:
    labels = [MODE_LABELS[str(row["mode"])] for row in rows]
    colors = ["#3a6ea5", "#4f9d69", "#c06c4f"]
    panels = [
        ("explained_residual_energy", "Explained residual energy", "fraction"),
        ("median_pair_correlation", "Median pair correlation", "correlation"),
        ("median_residual_std_mm", "Median residual std", "mm"),
        ("sign_mixing_fraction", "Sign mixing", "fraction"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (key, title, unit) in zip(axes.flat, panels):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(np.arange(len(rows)), values, color=colors)
        axis.set_xticks(np.arange(len(rows)))
        axis.set_xticklabels(labels, rotation=13, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Reference-plane mode comparison — no compensation, no smoothing")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "reference_mode_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_comparison_report(
    rows_by_mode: dict[str, dict[str, Any]], repositioning: dict[str, float], z0_source: dict[str, object]
) -> None:
    a = rows_by_mode["self_fitted"]
    b = rows_by_mode["fixed_normal_per_frame_offset"]
    c = rows_by_mode["fixed_ground_plane"]
    pair_delta_ba = float(b["median_pair_correlation"] - a["median_pair_correlation"])
    energy_delta_ba = float(b["explained_residual_energy"] - a["explained_residual_energy"])
    pair_delta_cb = float(c["median_pair_correlation"] - b["median_pair_correlation"])
    energy_delta_cb = float(c["explained_residual_energy"] - b["explained_residual_energy"])
    std_change_ba = float(
        (b["median_residual_std_mm"] / a["median_residual_std_mm"] - 1.0) * 100.0
    )
    std_change_cb = float(
        (c["median_residual_std_mm"] / b["median_residual_std_mm"] - 1.0) * 100.0
    )
    abs_bias_change_cb = float(
        (c["median_abs_bias_mm"] / b["median_abs_bias_mm"] - 1.0) * 100.0
    )
    top_std_change_cb = float((c["top_region_std"] / b["top_region_std"] - 1.0) * 100.0)
    middle_std_change_cb = float(
        (c["middle_region_std"] / b["middle_region_std"] - 1.0) * 100.0
    )
    bottom_std_change_cb = float(
        (c["bottom_region_std"] / b["bottom_region_std"] - 1.0) * 100.0
    )
    relation = max(
        abs(repositioning["tilt_log10_condition_pearson_r"]),
        abs(repositioning["tilt_condition_spearman_r"]),
    )
    relation_text = "明显" if relation >= 0.5 else "不明显"
    report = f"""# Reference-plane mode comparison report

## 范围与统一口径

- 输入：现有 laser 001–031 重建点；三种模式使用同一批 frame 和 image-row `v` bin。
- 未做 compensation；未使用 smooth window。
- residual 均为 ground-frame `Zg` 方向 signed vertical residual，不是正交点面距离。
- pair correlation 只统计共同支持不少于 {MIN_CORRELATION_OVERLAP} 个 `v` 的唯一 frame pair。
- `median_residual_std_mm`、`median_abs_bias_mm` 和 sign mixing 使用 `sample_count >= {MIN_WELL_SUPPORTED_FRAMES}` 的行。
- top/middle/bottom 分别为 `v=0–299`、`300–2699`、`2700–2999`，区域 std 使用至少 2 帧覆盖的行。

## 统一结果

| mode | median pair corr | pair corr >=0.5 | explained energy | median residual std / mm | median abs(bias) / mm | sign mixing |
|---|---:|---:|---:|---:|---:|---:|
| A `self_fitted` | {a['median_pair_correlation']:.4f} | {a['fraction_pair_corr_ge_0.5']:.2%} | {a['explained_residual_energy']:.4f} | {a['median_residual_std_mm']:.5f} | {a['median_abs_bias_mm']:.5f} | {a['sign_mixing_fraction']:.2%} |
| B `fixed_normal_per_frame_offset` | {b['median_pair_correlation']:.4f} | {b['fraction_pair_corr_ge_0.5']:.2%} | {b['explained_residual_energy']:.4f} | {b['median_residual_std_mm']:.5f} | {b['median_abs_bias_mm']:.5f} | {b['sign_mixing_fraction']:.2%} |
| C `fixed_ground_plane` | {c['median_pair_correlation']:.4f} | {c['fraction_pair_corr_ge_0.5']:.2%} | {c['explained_residual_energy']:.4f} | {c['median_residual_std_mm']:.5f} | {c['median_abs_bias_mm']:.5f} | {c['sign_mixing_fraction']:.2%} |

## A. B 相对 A 是否提高跨帧一致性

没有明显提高，反而降低：median pair correlation 变化 {pair_delta_ba:+.4f}，explained residual energy 变化 {energy_delta_ba:+.4f}，median residual std 增加 {std_change_ba:+.2f}%。
因此现有数据不支持“自由平面拟合此前吸收了某个稳定的一维系统 residual，而固定法向后可恢复它”这一解释。

## B. C 相对 B 的恶化与重新摆放 offset

C 相对 B 的 median pair correlation 变化 {pair_delta_cb:+.4f}，explained residual energy 变化 {energy_delta_cb:+.4f}，median residual std 变化 {std_change_cb:+.2f}%，median abs(bias) 增加 {abs_bias_change_cb:+.2f}%。
区域 std 的变化分别为：top {top_std_change_cb:+.2f}%、middle {middle_std_change_cb:+.2f}%、bottom {bottom_std_change_cb:+.2f}%。
固定全局 Z0 会保留每帧整体高度变化，而 B 会移除该变化。31 帧 `median(Zg)-Z0` 的范围为
{repositioning['offset_range_mm']:.5f} mm，样本标准差 {repositioning['offset_std_mm']:.5f} mm，P95 absolute offset
{repositioning['offset_p95_abs_mm']:.5f} mm。这说明重新摆放与重建链共同表现出的逐帧高度 offset 在绝对 residual 中不可忽略；
但没有独立 PnP，不能把它全部归因于棋盘物理重新摆放。
同时，C 与 B 的相关系数理论上对逐帧常数平移不敏感，所以不能仅靠 correlation 判断 offset 大小。
同理，C 的 sign-mixing fraction 下降会受到整体 residual 符号偏移影响，不能单独解释为波形一致性提高。

## C. apparent tilt 与 condition number

- raw Pearson r：{repositioning['tilt_condition_pearson_r']:.4f}
- `log10(condition number)` Pearson r：{repositioning['tilt_log10_condition_pearson_r']:.4f}
- Spearman r：{repositioning['tilt_condition_spearman_r']:.4f}

按 `|r| >= 0.5` 作为明显关系的描述阈值，两者关系{relation_text}。无论相关性强弱，
`apparent_tilt_deg` 都来自激光重建点的窄带自拟合，不能解释为棋盘真实机械倾角。

apparent tilt：median {repositioning['apparent_tilt_median_deg']:.3f}°，P95
{repositioning['apparent_tilt_p95_deg']:.3f}°，max {repositioning['apparent_tilt_max_deg']:.3f}°。

## D. 棋盘真实倾角变化是否可忽略

**INCONCLUSIVE**

这 31 帧没有逐帧独立棋盘 PnP 平面；A 的 `a,b` 来自同一批窄带激光重建点，B/C 又是水平法向假设，
三者都不能独立观测棋盘真实姿态。因此现有数据既不足以支持真实倾角变化可忽略，也不足以证明它不可忽略。

## Z0 来源

`fixed_ground_plane` 使用 `Z0=0.0 mm`。来源为 `{z0_source['path']}`：外参明确规定
`{z0_source['zero_surface']}` 为 ground zero surface；camera-frame 平面经 `T_ground_from_camera`
变换后也数值验证为 `Zg=0`。这不是任意把 Z0 设为零。

## 重要限制

`apparent_tilt_deg = degrees(atan(sqrt(a^2+b^2)))` 只是“窄带自拟合得到的表观倾角”，
不允许把它直接解释成棋盘真实机械倾角。
"""
    (OUTPUT_DIR / "reference_mode_comparison_report.md").write_text(report, encoding="utf-8")


def write_output_files() -> None:
    mode_rows = "\n".join(
        f"| `reference_modes/{mode}/residual_frame_v_heatmap.png` | {mode} 下逐帧 residual(v) 热图 | 重复波形、符号翻转、覆盖空白 | 只能描述该 reference 定义下的 residual；不能给出真实棋盘姿态 | 是 |\n"
        f"| `reference_modes/{mode}/residual_v_median_sigma.png` | {mode} 下跨帧 median、±1 sigma、sample count | median 波形、离散度、支持数 | 不能证明波形来源或补偿有效性 | 是 |\n"
        f"| `reference_modes/{mode}/residual_v_statistics.csv` | {mode} 下逐 v 完整数值统计 | sample count、median、std、MAD、P95 | 不能把 sparse v 行当稳定系统误差 | 否，适合作为备查数据 |\n"
        f"| `reference_modes/{mode}/frame_residual_correlation.csv` | {mode} 下所有 frame pair 共同支持与相关系数 | common_sample_count 和 correlation | 低共同支持相关性不可靠；相关性不反映常数 offset | 否，适合作为备查数据 |\n"
        f"| `reference_modes/{mode}/diagnostics_summary.json` | {mode} 的结构化口径和指标；fixed_ground 另含 Z0 provenance | comparison_metrics、metric_definitions | 不是人工结论，也不是 compensation 配置 | 否，机器复核用 |"
        for mode in base.REFERENCE_PLANE_MODES
    )
    document = f"""# Ground-bias reference-mode diagnostics 输出文件说明

本目录所有结果均基于现有 laser 001–031，未执行补偿、未调整 smooth window。

| 文件 | 文件体现什么 | 应看哪些指标 | 不能得出什么结论 | 推荐放组会报告 |
|---|---|---|---|---|
{mode_rows}
| `reference_mode_comparison.csv` | 三种 reference mode 的统一数值口径 | pair correlation、explained energy、std、sign mixing、三区域 std | 不能把任一模式自动认定为真实棋盘平面 | 是 |
| `reference_mode_comparison.png` | 四个关键一致性指标的并排柱图 | A/B/C 相对变化 | 不能代替逐 v 曲线和覆盖检查 | 是 |
| `repositioning_effects.csv` | 31 帧 offset、表观倾角和 self-fit condition number | offset 范围；tilt 与 condition 的异常帧 | apparent tilt 不是机械倾角 | 可选；报告正文优先放统计摘要 |
| `reference_mode_comparison_report.md` | A–D 问题的定量回答、Z0 来源和限制 | 四个结论段、offset/tilt 统计 | 不是正式补偿验收报告 | 是 |
| `OUTPUT_FILES.md` | 本文件；输出导航和解释边界 | 推荐列和文件用途 | 不包含新的数据分析 | 否 |
| `diagnostics_report.md` | 早先 self_fitted baseline 的详细 A–E 报告，现保留兼容 | self_fitted 的区域、符号翻转、边缘结论 | 不能代表 B/C；统一比较应看新 report | 不推荐单独使用 |
| `diagnostics_summary.json` | 早先 baseline 加初版三模式摘要，现保留兼容 | provenance 和旧 baseline 指标 | 不应替代各模式独立 summary | 否 |
| `residual_frame_v_heatmap.png` | 根目录 legacy self_fitted 热图 | 同 self_fitted 子目录热图 | 不能代表 B/C | 不推荐；使用子目录版本 |
| `residual_v_median_sigma.png` | 根目录 legacy self_fitted 曲线 | 同 self_fitted 子目录曲线 | 不能代表 B/C | 不推荐；使用子目录版本 |
| `residual_v_statistics.csv` | 根目录 legacy self_fitted 统计 | 同 self_fitted 子目录 CSV | 不能代表 B/C | 否 |
| `frame_residual_correlation.csv` | 根目录 legacy self_fitted pair correlation | 同 self_fitted 子目录 CSV | 不能代表 B/C | 否 |
| `per_frame_plane_fit_diagnostics.csv` | self-fit 平面参数、条件数和 inlier 信息 | condition number、inlier count | `a,b` 不能解释成真实机械姿态 | 否 |
| `reference_plane_mode_comparison_per_frame.csv` | 前一轮三模式逐帧初版对照，保留兼容 | self-fit 参数、offset | 新分析应优先使用 `repositioning_effects.csv` | 否 |
| `reference_plane_mode_comparison_summary.csv` | 前一轮三模式初版总体对照，保留兼容 | MAE、RMS、P95 | 指标口径少于新统一比较 | 否 |

组会建议的最小文件组合：

1. `reference_mode_comparison.png`
2. `reference_mode_comparison_report.md`
3. 三张 `reference_modes/*/residual_v_median_sigma.png`
4. 如需展示帧间结构，再选三张 heatmap；不要只展示 self_fitted。
"""
    (OUTPUT_DIR / "OUTPUT_FILES.md").write_text(document, encoding="utf-8")


def main() -> None:
    paths = sorted(base.INPUT_DIR.glob("laser *.csv"))
    if len(paths) != base.FRAME_COUNT:
        raise RuntimeError(f"Expected {base.FRAME_COUNT} input frames, found {len(paths)}")
    fit_args = SimpleNamespace(plane_fit_mad_threshold=3.5, plane_fit_max_iterations=8)
    z0_mm, z0_source = base.load_fixed_ground_z0(base.GROUND_EXTRINSICS)
    raw_frames: list[tuple[str, base.FramePoints]] = []
    for path in paths:
        frame = base.load_csv_or_txt(path, ("v", "x", "y", "z"), compensation_axis="v")
        raw_frames.append((path.stem.removeprefix("laser ").strip(), frame))
    frame_ids = [frame_id for frame_id, _ in raw_frames]
    v_min = int(min(np.min(frame.u) for _, frame in raw_frames))
    v_max = int(max(np.max(frame.u) for _, frame in raw_frames))
    v = np.arange(v_min, v_max + 1, dtype=np.float64)

    mode_frames: dict[str, list[tuple[str, base.FramePoints]]] = {
        mode: [] for mode in base.REFERENCE_PLANE_MODES
    }
    self_diagnostics: dict[str, dict[str, object]] = {}
    offsets: dict[str, float] = {}
    for frame_id, frame in raw_frames:
        for mode in base.REFERENCE_PLANE_MODES:
            residual_frame, diagnostic = base.reference_residual_frame(
                frame, mode, z0_mm=z0_mm, fit_args=fit_args
            )
            mode_frames[mode].append((frame_id, residual_frame))
            if mode == "self_fitted":
                self_diagnostics[frame_id] = diagnostic
            elif mode == "fixed_normal_per_frame_offset":
                offsets[frame_id] = float(diagnostic["reference_z_mm"])

    comparison_rows: list[dict[str, Any]] = []
    for mode in base.REFERENCE_PLANE_MODES:
        matrix = base.residual_matrix(mode_frames[mode], v, v_min)
        metrics = analyze_mode(mode, frame_ids, matrix, v, v_min, v_max, z0_mm, z0_source)
        comparison_rows.append({"mode": mode, **metrics})

    comparison_fields = [
        "mode",
        "median_pair_correlation",
        "fraction_pair_corr_ge_0.5",
        "explained_residual_energy",
        "median_residual_std_mm",
        "median_abs_bias_mm",
        "sign_mixing_fraction",
        "top_region_std",
        "middle_region_std",
        "bottom_region_std",
        "well_supported_row_count",
        "retained_pair_count",
    ]
    with (OUTPUT_DIR / "reference_mode_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    repositioning_rows: list[dict[str, float | int | str]] = []
    for frame_id, frame in raw_frames:
        self_fit = self_diagnostics[frame_id]
        repositioning_rows.append(
            {
                "frame_id": frame_id,
                "fixed_normal_offset_mm": offsets[frame_id],
                "offset_from_global_Z0_mm": float(np.median(frame.xyz[:, 2]) - z0_mm),
                "apparent_tilt_deg": float(self_fit["apparent_tilt_deg"]),
                "self_fit_condition_number": float(self_fit["design_condition_number"]),
            }
        )
    with (OUTPUT_DIR / "repositioning_effects.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(repositioning_rows[0]))
        writer.writeheader()
        writer.writerows(repositioning_rows)

    repositioning = repositioning_summary(repositioning_rows)
    rows_by_mode = {str(row["mode"]): row for row in comparison_rows}
    save_comparison_plot(comparison_rows)
    write_comparison_report(rows_by_mode, repositioning, z0_source)
    write_output_files()
    print(
        json.dumps(
            {
                "comparison": comparison_rows,
                "repositioning": repositioning,
                "output_dir": str(OUTPUT_DIR),
                "compensation_applied": False,
                "smooth_window_applied": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
