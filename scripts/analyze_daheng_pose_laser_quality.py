#!/usr/bin/env python3
"""按 pose 量化大恒 laser-plane 与阶梯障碍观测图的激光质量。"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_tool.io_utils import load_document

from analyze_laser_exposure_sweep import (  # noqa: E402
    _peak_width_metrics,
    _percentile,
    _steger_aligned_widths,
)


HARD_ROI = (1760, 0, 480, 3000)
POSE_FIELDS = (
    "dataset",
    "split",
    "pose",
    "frame_count",
    "exposure_us",
    "search_mode",
    "peak_p50_dn",
    "peak_p95_dn",
    "saturation_fraction",
    "near_saturation_fraction",
    "fwhm_p50_px",
    "fwhm_p95_px",
    "valid_fraction",
    "response_p50",
)


def _load_steger(calibration_src: Path) -> tuple[Any, dict[str, Any]]:
    source = calibration_src.expanduser().resolve()
    module_path = source / "realtime_steger.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"共享 realtime_steger.py 不存在：{module_path}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("realtime_steger")
    if Path(module.__file__).resolve() != module_path:
        raise RuntimeError(
            f"共享 Steger 加载路径错误：期望 {module_path}，实际 {module.__file__}"
        )
    options = dict(module.load_steger_options())
    options["scan_axis"] = "row"
    return module, module.merge_options(options)


def _iter_datasets(data_root: Path) -> Iterable[tuple[str, Path, str]]:
    yield "laser_plane", data_root / "laser_plane", "auto"
    obs_root = data_root / "obs"
    for name in (".chessboard.inprogress", "test", "test_350"):
        yield f"obs/{name}", obs_root / name, "hard_roi"


def _split(record: Mapping[str, Any]) -> str:
    tags = record.get("tags")
    if isinstance(tags, Mapping) and tags.get("split"):
        return str(tags["split"])
    return str(record["task_id"]).split("_", 1)[0]


def _exposure(record: Mapping[str, Any]) -> float:
    camera = record.get("applied_camera") or record.get("requested_camera")
    if not isinstance(camera, Mapping):
        raise ValueError(f"frame 缺少相机参数：{record.get('filename')}")
    return float(camera["exposure_us"])


def _sensor_max(record: Mapping[str, Any]) -> float:
    camera = record.get("applied_camera") or record.get("requested_camera")
    if not isinstance(camera, Mapping):
        raise ValueError(f"frame 缺少相机参数：{record.get('filename')}")
    return 255.0 if camera["pixel_format"] == "Mono8" else 4095.0


def _extract_frame(
    image: np.ndarray,
    *,
    steger: Any,
    options: Mapping[str, Any],
    search_mode: str,
    sensor_max_value: float,
) -> dict[str, Any]:
    if search_mode == "hard_roi":
        roi_x, roi_y, roi_width, roi_height = HARD_ROI
        right = min(image.shape[1], roi_x + roi_width)
        bottom = min(image.shape[0], roi_y + roi_height)
        if right <= roi_x or bottom <= roi_y:
            raise ValueError(f"硬 ROI {HARD_ROI} 与图像 {image.shape[::-1]} 无交集")
        analysis_image = np.ascontiguousarray(image[roi_y:bottom, roi_x:right])
        search_region = steger.LaserSearchRegion(
            0,
            analysis_image.shape[1],
            "daheng_hard_roi",
        )
        extraction = steger.extract_steger(
            analysis_image,
            options,
            search_region=search_region,
            diagnostic=True,
        )
        search_start = roi_x
        search_end = right
    elif search_mode == "auto":
        extraction = steger.extract_steger(image, options, diagnostic=True)
        start_value = extraction.metadata.get("final_search_region_start_px")
        end_value = extraction.metadata.get("final_search_region_end_px")
        if start_value is None or end_value is None:
            analysis_image = image
            search_start = 0
            search_end = image.shape[1]
        else:
            search_start = max(0, int(math.floor(float(start_value))))
            search_end = min(image.shape[1], int(math.ceil(float(end_value))))
            analysis_image = np.ascontiguousarray(image[:, search_start:search_end])
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"未知 search_mode：{search_mode}")

    valid = np.asarray(extraction.valid, dtype=bool)
    centres_u = np.asarray(extraction.u_px, dtype=np.float64)
    if search_mode == "auto":
        centres_u = centres_u - float(search_start)
    response = np.asarray(extraction.response, dtype=np.float64)
    widths = _peak_width_metrics(analysis_image, sensor_max_value)
    _contiguous, interpolated = _steger_aligned_widths(
        analysis_image,
        centres_u,
        valid,
        widths["background"],
    )
    aligned_peak = _steger_aligned_peaks(
        analysis_image,
        centres_u,
        valid,
    )
    return {
        "peak_dn": aligned_peak,
        "peak_saturated": aligned_peak >= sensor_max_value * 0.995,
        "peak_near_saturated": aligned_peak >= sensor_max_value * 0.98,
        "fwhm": interpolated,
        "valid": valid,
        "response": response,
        "search_start_px": float(search_start),
        "search_end_px": float(search_end),
    }


def _steger_aligned_peaks(
    image: np.ndarray,
    centres_u: np.ndarray,
    valid: np.ndarray,
    *,
    half_window_px: int = 24,
) -> np.ndarray:
    """测量正式 Steger 中心附近的局部原始强度峰，排除阶梯亮边。"""

    peak = np.full(image.shape[0], np.nan, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(centres_u)
    for row in np.flatnonzero(usable):
        centre = int(round(float(centres_u[row])))
        start = max(0, centre - half_window_px)
        end = min(image.shape[1], centre + half_window_px + 1)
        if end > start:
            peak[row] = float(np.max(image[row, start:end]))
    return peak


def _pooled(frames: list[dict[str, Any]], name: str, mask_name: str) -> np.ndarray:
    values = [
        np.asarray(frame[name])[np.asarray(frame[mask_name], dtype=bool)]
        for frame in frames
    ]
    nonempty = [value for value in values if value.size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)


def _summarize_pose(
    dataset: str,
    split: str,
    pose: str,
    exposure_us: float,
    search_mode: str,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    peak = _pooled(frames, "peak_dn", "valid")
    saturated = _pooled(frames, "peak_saturated", "valid").astype(bool)
    near_saturated = _pooled(frames, "peak_near_saturated", "valid").astype(bool)
    fwhm = _pooled(frames, "fwhm", "valid")
    response = _pooled(frames, "response", "valid")
    valid_fraction = float(np.mean([
        np.mean(np.asarray(frame["valid"], dtype=bool)) for frame in frames
    ]))
    return {
        "dataset": dataset,
        "split": split,
        "pose": pose,
        "frame_count": len(frames),
        "exposure_us": exposure_us,
        "search_mode": search_mode,
        "peak_p50_dn": _percentile(peak, 50),
        "peak_p95_dn": _percentile(peak, 95),
        "saturation_fraction": float(np.mean(saturated)) if saturated.size else float("nan"),
        "near_saturation_fraction": (
            float(np.mean(near_saturated)) if near_saturated.size else float("nan")
        ),
        "fwhm_p50_px": _percentile(fwhm, 50),
        "fwhm_p95_px": _percentile(fwhm, 95),
        "valid_fraction": valid_fraction,
        "response_p50": _percentile(response, 50),
    }


def analyze(data_root: Path, calibration_src: Path, output_dir: Path) -> list[dict[str, Any]]:
    data_root = data_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steger, options = _load_steger(calibration_src)
    pose_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for dataset_name, dataset_path, search_mode in _iter_datasets(data_root):
        manifest = load_document(dataset_path / "dataset_manifest.yaml")
        grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
        for record in manifest["frames"]:
            if record.get("role") != "laser":
                continue
            image_path = dataset_path / record["filename"]
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"无法读取 {image_path}")
            exposure_us = _exposure(record)
            extracted = _extract_frame(
                image,
                steger=steger,
                options=options,
                search_mode=search_mode,
                sensor_max_value=_sensor_max(record),
            )
            split = _split(record)
            pose = str(record["pose_id"])
            grouped[(split, pose, exposure_us)].append(extracted)
            single = _summarize_pose(
                dataset_name,
                split,
                pose,
                exposure_us,
                search_mode,
                [extracted],
            )
            frame_rows.append({
                **single,
                "frame_index": int(record["index"]),
                "filename": str(record["filename"]),
                "search_start_px": extracted["search_start_px"],
                "search_end_px": extracted["search_end_px"],
            })

        for (split, pose, exposure_us), frames in sorted(grouped.items()):
            pose_rows.append(_summarize_pose(
                dataset_name,
                split,
                pose,
                exposure_us,
                search_mode,
                frames,
            ))

    _write_csv(output_dir / "pose_metrics.csv", pose_rows, POSE_FIELDS)
    frame_fields = tuple(frame_rows[0]) if frame_rows else ()
    _write_csv(output_dir / "frame_metrics.csv", frame_rows, frame_fields)
    _write_report(output_dir / "analysis_report.md", pose_rows, options)
    return pose_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "nan" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _write_report(path: Path, rows: list[dict[str, Any]], options: Mapping[str, Any]) -> None:
    lines = [
        "# 大恒激光图逐 pose Steger 量化分析",
        "",
        "- `laser_plane`：正式共享 Steger 自动搜索带。",
        "- `obs/*`：裁剪到大恒硬 ROI `u=[1760,2240), v=[0,3000)` 后，复用正式共享 Steger。",
        f"- Steger：`sigma={options['sigma']}`、`threshold={options['threshold']}`、"
        f"`deriv_thresh={options['deriv_thresh']}`、`scan_axis={options['scan_axis']}`。",
        "- peak 是 Steger valid 行中，中心 ±24 px 内的局部原始峰值（DN）；FWHM 是同一局部峰的连续插值半高宽。",
        "",
    ]
    for dataset in dict.fromkeys(str(row["dataset"]) for row in rows):
        lines.extend([
            f"## {dataset}",
            "",
            "| pose | exposure | peak P50 | peak P95 | saturation | near-sat | FWHM P50 | FWHM P95 | valid fraction | response P50 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (item for item in rows if item["dataset"] == dataset):
            lines.append(
                f"| {row['pose']} | {_format(row['exposure_us'], 0)} | "
                f"{_format(row['peak_p50_dn'])} | {_format(row['peak_p95_dn'])} | "
                f"{_format(100 * row['saturation_fraction'], 2)}% | "
                f"{_format(100 * row['near_saturation_fraction'], 2)}% | "
                f"{_format(row['fwhm_p50_px'])} | {_format(row['fwhm_p95_px'])} | "
                f"{_format(100 * row['valid_fraction'], 2)}% | "
                f"{_format(row['response_p50'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "projects" / "daheng" / "data",
    )
    parser.add_argument(
        "--calibration-src",
        type=Path,
        default=WORKSPACE / "calibration" / "src",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "projects" / "daheng" / "analysis" / "laser_quality_steger",
    )
    args = parser.parse_args()
    rows = analyze(args.data_root, args.calibration_src, args.output_dir)
    print(f"分析完成：{len(rows)} 个 pose，输出 {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
