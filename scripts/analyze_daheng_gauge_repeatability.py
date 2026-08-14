#!/usr/bin/env python3
"""大恒量块多帧台阶、高度和重复性分析。

该脚本是一个薄适配层：

* TIFF 读取复用 0704 工具的 ``utils.image_io``；
* 中心线复用 ``calibration/src/realtime_steger.py``；
* 三维重建复用 ``reconstruction.reconstructor``；
* 台阶稳定区判断复用 Phase-A ``geometry_experiment`` 的
  ``_detect_stable_plateau`` 和 ``_true_runs``。

当前大恒纵向激光使用 ``scan_axis=row``，因此分析轴是图像 ``v``，每个
扫描位置的激光中心是 ``u``。输出高度单位为 mm。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
DEFAULT_SCANNER_ROOT = WORKSPACE_ROOT / "0704line-laser-3d-scanner"
DEFAULT_CALIBRATION_ROOT = WORKSPACE_ROOT / "calibration"
DEFAULT_CONFIG = (
    DEFAULT_SCANNER_ROOT
    / "laser_measurement_tool"
    / "configs"
    / "measure_tool_daheng_0811.yaml"
)
DEFAULT_INPUT = (
    SCRIPT_ROOT / "projects" / "daheng" / "data" / "obs" / "test_350" / "fit"
)
DEFAULT_OUTPUT = (
    SCRIPT_ROOT / "projects" / "daheng" / "outputs" / "gauge_repeatability"
)


def _bootstrap_imports(scanner_root: Path, calibration_root: Path) -> dict[str, Any]:
    """加载两个仓库中的既有模块，避免复制实现。"""

    module_paths = (
        scanner_root / "laser_measurement_tool",
        calibration_root / "src",
        SCRIPT_ROOT / "scripts",
    )
    for path in module_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"模块目录不存在: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from calibration.config_loader import load_calibration_files
    from geometry_experiment import _detect_stable_plateau, _true_runs
    from reconstruction.reconstructor import (
        ReconstructionParams,
        reconstruct_uv_to_ground,
    )
    from utils.image_io import load_grayscale_image
    import realtime_steger

    return {
        "load_calibration_files": load_calibration_files,
        "detect_stable_plateau": _detect_stable_plateau,
        "true_runs": _true_runs,
        "ReconstructionParams": ReconstructionParams,
        "reconstruct_uv_to_ground": reconstruct_uv_to_ground,
        "load_grayscale_image": load_grayscale_image,
        "realtime_steger": realtime_steger,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 YAML: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点必须是映射: {path}")
    return value


def _resolve_path(value: str | Path, *bases: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [base / path for base in bases]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_profile_options(
    config_path: Path,
    calibration_root: Path,
    extraction: dict[str, Any],
) -> dict[str, Any]:
    profile_value = extraction.get("profile")
    profile: dict[str, Any] = {}
    if profile_value:
        profile_path = _resolve_path(
            profile_value,
            config_path.parent,
            calibration_root / "config",
            WORKSPACE_ROOT / "calibration" / "config",
        )
        if not profile_path.is_file():
            # 0704 配置中的历史相对路径指向仓库外的共享 calibration 目录；
            # 用文件名回退，避免把 Steger 默认值与正式 profile 混用。
            profile_path = calibration_root / "config" / Path(profile_value).name
        if profile_path.is_file():
            document = _read_yaml(profile_path)
            profile_value = document.get("steger", document.get("options", {}))
            if isinstance(profile_value, dict):
                profile = dict(profile_value)
    inline = extraction.get("steger", {})
    if not isinstance(inline, dict):
        raise ValueError("extraction.steger 必须是映射")
    profile.update(inline)
    profile.setdefault("scan_axis", "row")
    return profile


def _load_runtime_config(
    config_path: Path,
    calibration_root: Path,
    modules: dict[str, Any],
) -> tuple[dict[str, Any], Any, Any, Any]:
    document = _read_yaml(config_path)
    extraction = document.get("extraction", {})
    calibration_document = document.get("calibration", {})
    reconstruction_document = document.get("reconstruction", {})
    if not isinstance(extraction, dict) or not isinstance(calibration_document, dict):
        raise ValueError("配置中的 extraction/calibration 必须是映射")

    profile = _load_profile_options(config_path, calibration_root, extraction)
    search_roi = profile.pop("search_roi", None)
    scan_axis = str(profile.get("scan_axis", "row")).lower()
    if scan_axis not in {"row", "column"}:
        raise ValueError(f"scan_axis 必须是 row 或 column，实际为 {scan_axis!r}")

    def calibration_file(key: str, fallback: str) -> Path:
        value = calibration_document.get(key, fallback)
        return _resolve_path(value, config_path.parent, config_path.parent / "configs")

    intrinsics = calibration_file("intrinsics", "calibration_result.yaml")
    laser_model = calibration_file("laser_model", "circular_cone.yaml")
    extrinsics = calibration_file("extrinsics", "camera_ground_extrinsics.yaml")
    ground_u_value = calibration_document.get("ground_u_compensation")
    ground_u = (
        None
        if ground_u_value in (None, "")
        else _resolve_path(ground_u_value, config_path.parent)
    )
    calibration = modules["load_calibration_files"](
        intrinsics=intrinsics,
        laser_plane=laser_model,
        extrinsics=extrinsics,
        ground_u_compensation=ground_u,
        ground_u_optional=True,
    )

    params_type = modules["ReconstructionParams"]
    allowed = {
        "parallel_epsilon",
        "quadratic_epsilon",
        "min_camera_depth_mm",
        "max_camera_depth_mm",
        "model_range_margin_mm",
        "image_roi_polygon",
    }
    reconstruction_kwargs = {
        key: value
        for key, value in reconstruction_document.items()
        if key in allowed
    }
    reconstruction_params = params_type(**reconstruction_kwargs)

    return (
        {
            "document": document,
            "steger_options": profile,
            "scan_axis": scan_axis,
            "search_roi": search_roi,
        },
        calibration,
        reconstruction_params,
        document,
    )


def _parse_range(value: str, name: str) -> tuple[int, int]:
    pieces = value.replace(",", ":").split(":")
    if len(pieces) != 2:
        raise ValueError(f"{name} 应为 start:end，实际为 {value!r}")
    start, end = (int(piece.strip()) for piece in pieces)
    if start < 0 or end <= start:
        raise ValueError(f"{name} 必须满足 0 <= start < end")
    return start, end


def _search_region(
    runtime: dict[str, Any],
    scan_axis: str,
    override: str | None,
    realtime_steger: Any,
) -> Any | None:
    if override:
        start, end = _parse_range(override, "search_roi")
        return realtime_steger.LaserSearchRegion(start, end, "cli")
    configured = runtime.get("search_roi")
    if not isinstance(configured, dict):
        return None
    if scan_axis == "row":
        start = int(configured.get("offset_x", 0))
        length = int(configured.get("width", 0))
    else:
        start = int(configured.get("offset_y", 0))
        length = int(configured.get("height", 0))
    if length <= 0:
        return None
    return realtime_steger.LaserSearchRegion(
        start,
        start + length,
        "config.search_roi",
    )


def _resolve_images(input_path: Path, pattern: str, max_frames: int | None) -> list[Path]:
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            path
            for path in input_path.glob(pattern)
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".bmp"}
        )
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    if max_frames is not None:
        images = images[:max_frames]
    if not images:
        raise FileNotFoundError(f"没有找到输入图像: {input_path} / {pattern}")
    return images


def _profile_arrays(
    extraction: Any,
    image_shape: tuple[int, int],
    scan_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    if scan_axis == "row":
        scan_coordinate = np.asarray(extraction.v_px, dtype=np.float64)
        centre_coordinate = np.asarray(extraction.u_px, dtype=np.float64)
        size = height
    else:
        scan_coordinate = np.asarray(extraction.u_px, dtype=np.float64)
        centre_coordinate = np.asarray(extraction.v_px, dtype=np.float64)
        size = width

    centre = np.full(size, np.nan, dtype=np.float64)
    valid = np.zeros(size, dtype=bool)
    response = np.full(size, -np.inf, dtype=np.float64)
    extracted_valid = np.asarray(extraction.valid, dtype=bool)
    extracted_response = np.asarray(extraction.response, dtype=np.float64)
    for index in np.flatnonzero(
        extracted_valid
        & np.isfinite(scan_coordinate)
        & np.isfinite(centre_coordinate)
    ):
        scan_index = int(np.rint(scan_coordinate[index]))
        if not 0 <= scan_index < size:
            continue
        score = extracted_response[index] if np.isfinite(extracted_response[index]) else 0.0
        if not valid[scan_index] or score > response[scan_index]:
            centre[scan_index] = centre_coordinate[index]
            response[scan_index] = score
            valid[scan_index] = True
    return centre, valid, response


def _reconstruct_profile(
    centre: np.ndarray,
    valid: np.ndarray,
    scan_axis: str,
    calibration: dict[str, Any],
    reconstruction_params: Any,
    modules: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scan_indices = np.flatnonzero(valid & np.isfinite(centre))
    points_uv = (
        np.column_stack([centre[scan_indices], scan_indices])
        if scan_axis == "row"
        else np.column_stack([scan_indices, centre[scan_indices]])
    )
    result = modules["reconstruct_uv_to_ground"](
        points_uv,
        calibration,
        reconstruction_params,
    )
    size = centre.size
    xyz = np.full((size, 3), np.nan, dtype=np.float64)
    reconstructed_valid = np.zeros(size, dtype=bool)
    for pixel, point in zip(result.pixels_uv, result.points_ground):
        scan_index = int(np.rint(pixel[1] if scan_axis == "row" else pixel[0]))
        if 0 <= scan_index < size:
            xyz[scan_index] = point
            reconstructed_valid[scan_index] = True
    return xyz[:, 0], xyz[:, 1], xyz[:, 2], reconstructed_valid


def _nanmedian(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            return np.nanmedian(values, axis=axis)


def _sample_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.nanstd(values, axis=axis, ddof=1)
    return np.where(counts >= 2, result, 0.0)


def _detect_platforms(
    profile: np.ndarray,
    sigma: np.ndarray,
    valid_fraction: np.ndarray,
    scan_range: tuple[int, int],
    args: argparse.Namespace,
    modules: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    start, end = scan_range
    if end > profile.size:
        raise ValueError(f"scan_range 超出剖面长度: {scan_range}, size={profile.size}")
    finite = profile[start:end][np.isfinite(profile[start:end])]
    if finite.size == 0:
        raise ValueError("scan_range 内没有可用的高度剖面")
    config = {
        "multiheight_valid_fraction_min": args.min_valid_fraction,
        "plateau_median_window_px": args.smooth_window,
        "plateau_max_gradient_px_per_column": args.max_gradient,
        "plateau_max_step_px": args.max_step,
        "plateau_sigma_mad_scale": args.sigma_mad_scale,
        "plateau_sigma_floor_limit_px": args.sigma_floor,
        "plateau_erosion_px": args.edge_trim,
        "min_stable_width_px": args.min_platform_width,
    }
    delta = profile - float(np.median(finite))
    detected = modules["detect_stable_plateau"](
        (start, end - 1), delta, sigma, valid_fraction, config
    )
    stable_global = np.zeros(profile.size, dtype=bool)
    stable_global[start:end] = detected["stable_mask"]
    platforms: list[dict[str, Any]] = []
    for run_start, run_end in modules["true_runs"](detected["stable_mask"]):
        stable_start = start + run_start
        stable_end = start + run_end
        analysis_start = stable_start + args.edge_trim
        analysis_end = stable_end - args.edge_trim
        if analysis_end - analysis_start + 1 < args.min_platform_width:
            continue
        platforms.append(
            {
                "platform_id": f"p{len(platforms) + 1:02d}",
                "stable_range": [stable_start, stable_end],
                "analysis_range": [analysis_start, analysis_end],
            }
        )
    if not platforms:
        raise ValueError(
            "没有检测到满足宽度的稳定平台；请检查 scan_range、Steger ROI 或阈值"
        )
    return platforms, stable_global, detected


def _load_frozen_platforms(path: Path, profile_size: int) -> list[dict[str, Any]]:
    """读取首批分析得到的 platform_summary/summary，冻结后续 ROI。"""

    if path.suffix.lower() in {".yaml", ".yml"}:
        document: Any = _read_yaml(path)
    else:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取冻结 ROI: {path}: {error}") from error
    if isinstance(document, dict):
        document = document.get("platforms", document.get("rois", document))
    if isinstance(document, dict):
        document = list(document.values())
    if not isinstance(document, list):
        raise ValueError("冻结 ROI 应为 platforms/rois 列表")

    platforms: list[dict[str, Any]] = []
    for index, item in enumerate(document, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"冻结 ROI 第 {index} 项不是映射")
        stable = item.get("stable_range", item.get("analysis_range"))
        analysis = item.get("analysis_range", stable)
        if not isinstance(stable, (list, tuple)) or not isinstance(analysis, (list, tuple)):
            raise ValueError(f"冻结 ROI 第 {index} 项缺少 stable_range/analysis_range")
        stable_start, stable_end = (int(value) for value in stable)
        analysis_start, analysis_end = (int(value) for value in analysis)
        if not (
            0 <= stable_start <= analysis_start <= analysis_end <= stable_end < profile_size
        ):
            raise ValueError(f"冻结 ROI 第 {index} 项范围无效: {stable}/{analysis}")
        platforms.append(
            {
                "platform_id": str(item.get("platform_id", f"p{index:02d}")),
                "stable_range": [stable_start, stable_end],
                "analysis_range": [analysis_start, analysis_end],
            }
        )
    if not platforms:
        raise ValueError("冻结 ROI 列表为空")
    return platforms


def _platform_stable_mask(platforms: list[dict[str, Any]], size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    for platform in platforms:
        start, end = platform["stable_range"]
        mask[start : end + 1] = True
    return mask


def _frame_stat(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.median(finite)), float(np.std(finite)), int(finite.size)


def _summarize_platforms(
    platforms: list[dict[str, Any]],
    xyz_stack: np.ndarray,
    frame_names: list[str],
    reference_platform: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_count = xyz_stack.shape[0]
    frame_z: list[np.ndarray] = []
    frame_spatial_sigma: list[np.ndarray] = []
    for platform in platforms:
        start, end = platform["analysis_range"]
        values = xyz_stack[:, start : end + 1, 2]
        stats = [_frame_stat(row) for row in values]
        frame_z.append(np.array([item[0] for item in stats]))
        frame_spatial_sigma.append(np.array([item[1] for item in stats]))

    pooled_z = np.array([float(_nanmedian(values)) for values in frame_z])
    if reference_platform is None:
        reference_index = int(np.nanargmin(pooled_z))
    else:
        reference_index = reference_platform - 1
        if not 0 <= reference_index < len(platforms):
            raise ValueError("reference_platform 超出检测到的平台数量")
    baseline_z = frame_z[reference_index]
    frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for platform_index, (platform, values, spatial_sigma) in enumerate(
        zip(platforms, frame_z, frame_spatial_sigma)
    ):
        heights = values - baseline_z
        platform["z_median_mm"] = float(np.nanmedian(values))
        platform["z_sigma_between_frames_mm"] = float(_sample_std(values, axis=0))
        platform["z_spatial_sigma_median_mm"] = float(np.nanmedian(spatial_sigma))
        platform["height_median_mm"] = float(np.nanmedian(heights))
        platform["height_sigma_between_frames_mm"] = float(_sample_std(heights, axis=0))
        platform["height_p95_abs_mm"] = float(
            np.nanpercentile(np.abs(heights - np.nanmedian(heights)), 95)
        )
        platform["is_reference"] = platform_index == reference_index
        summary_rows.append(dict(platform))
        for frame_index, (z_value, height_value, spatial_value) in enumerate(
            zip(values, heights, spatial_sigma)
        ):
            start, end = platform["analysis_range"]
            count = int(np.isfinite(xyz_stack[frame_index, start : end + 1, 2]).sum())
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "filename": frame_names[frame_index],
                    "platform_id": platform["platform_id"],
                    "scan_start": start,
                    "scan_end": end,
                    "point_count": count,
                    "z_median_mm": float(z_value),
                    "z_spatial_sigma_mm": float(spatial_value),
                    "height_relative_mm": float(height_value),
                    "reference_platform_id": platforms[reference_index]["platform_id"],
                }
            )
    return summary_rows, frame_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_profile_plot(
    path: Path,
    scan: np.ndarray,
    profile: np.ndarray,
    sigma: np.ndarray,
    valid_fraction: np.ndarray,
    stable: np.ndarray,
    platforms: list[dict[str, Any]],
    ylabel: str,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; plot skipped"

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axis = axes[0]
    axis.plot(scan, profile, color="#1565c0", linewidth=1.0, label="median")
    axis.fill_between(
        scan,
        profile - sigma,
        profile + sigma,
        color="#90caf9",
        alpha=0.35,
        linewidth=0,
        label="sample sigma",
    )
    axis.plot(scan[stable], profile[stable], ".", color="#2e7d32", markersize=2)
    for platform in platforms:
        start, end = platform["analysis_range"]
        axis.axvspan(start, end, alpha=0.15, label=platform["platform_id"])
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", ncol=3)
    axes[1].plot(scan, valid_fraction, color="#6a1b9a")
    axes[1].axhline(0.8, color="#c62828", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("scan coordinate (px)")
    axes[1].set_ylabel("valid fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return None


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    scanner_root = Path(args.scanner_root).resolve()
    calibration_root = Path(args.calibration_root).resolve()
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() and any(output_path.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空，使用 --overwrite 才会覆盖: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    modules = _bootstrap_imports(scanner_root, calibration_root)
    runtime, calibration, reconstruction_params, config_document = _load_runtime_config(
        config_path, calibration_root, modules
    )
    scan_axis = runtime["scan_axis"]
    if args.scan_axis:
        scan_axis = args.scan_axis
        runtime["steger_options"]["scan_axis"] = scan_axis
    images = _resolve_images(input_path, args.pattern, args.max_frames)
    search_region = _search_region(
        runtime,
        scan_axis,
        args.search_roi,
        modules["realtime_steger"],
    )
    scan_range = _parse_range(args.scan_range, "scan_range")

    centres: list[np.ndarray] = []
    xyz_profiles: list[np.ndarray] = []
    frame_names: list[str] = []
    image_shape: tuple[int, int] | None = None
    extraction_counts: list[int] = []
    reconstruction_counts: list[int] = []
    for image_path in images:
        image = modules["load_grayscale_image"](image_path)
        if image_shape is None:
            image_shape = tuple(int(value) for value in image.shape)
        elif tuple(image.shape) != image_shape:
            raise ValueError(f"图像尺寸不一致: {image_path}: {image.shape} vs {image_shape}")
        extraction = modules["realtime_steger"].extract_steger(
            image,
            runtime["steger_options"],
            search_region=search_region,
            use_auto_band=search_region is None,
        )
        centre, valid, _response = _profile_arrays(extraction, image.shape, scan_axis)
        x, y, z, reconstructed_valid = _reconstruct_profile(
            centre,
            valid,
            scan_axis,
            calibration,
            reconstruction_params,
            modules,
        )
        xyz = np.column_stack([x, y, z])
        xyz[~reconstructed_valid] = np.nan
        centres.append(centre)
        xyz_profiles.append(xyz)
        frame_names.append(image_path.name)
        extraction_counts.append(int(valid.sum()))
        reconstruction_counts.append(int(reconstructed_valid.sum()))

    centre_stack = np.asarray(centres, dtype=np.float64)
    xyz_stack = np.asarray(xyz_profiles, dtype=np.float64)
    centre_median = _nanmedian(centre_stack, axis=0)
    centre_sigma = _sample_std(centre_stack, axis=0)
    xyz_median = _nanmedian(xyz_stack, axis=0)
    xyz_sigma = _sample_std(xyz_stack, axis=0)
    valid_fraction = np.mean(np.isfinite(centre_stack), axis=0)
    z_valid_fraction = np.mean(np.isfinite(xyz_stack[:, :, 2]), axis=0)

    if args.detect_profile == "ground_z":
        detection_profile = xyz_median[:, 2]
        detection_sigma = xyz_sigma[:, 2]
        ylabel = "Zg median (mm)"
    else:
        detection_profile = centre_median
        detection_sigma = centre_sigma
        ylabel = "laser centre median (px)"
    if args.frozen_roi:
        platforms = _load_frozen_platforms(Path(args.frozen_roi), centre_stack.shape[1])
        stable_mask = _platform_stable_mask(platforms, centre_stack.shape[1])
        detector = {"status": "frozen_roi", "sigma_threshold_px": None}
        roi_source = str(Path(args.frozen_roi).resolve())
    else:
        platforms, stable_mask, detector = _detect_platforms(
            detection_profile,
            detection_sigma,
            z_valid_fraction if args.detect_profile == "ground_z" else valid_fraction,
            scan_range,
            args,
            modules,
        )
        roi_source = "auto"
    platform_rows, frame_rows = _summarize_platforms(
        platforms,
        xyz_stack,
        frame_names,
        args.reference_platform,
    )

    scan = np.arange(centre_stack.shape[1], dtype=np.int32)
    profile_rows = []
    for index in range(len(scan)):
        profile_rows.append(
            {
                "scan_px": int(index),
                "center_median_px": float(centre_median[index]),
                "center_sigma_px": float(centre_sigma[index]),
                "z_median_mm": float(xyz_median[index, 2]),
                "z_sigma_mm": float(xyz_sigma[index, 2]),
                "valid_fraction": float(valid_fraction[index]),
                "z_valid_fraction": float(z_valid_fraction[index]),
                "stable": bool(stable_mask[index]),
            }
        )
    _write_csv(output_path / "profile_by_scanline.csv", profile_rows)
    _write_csv(output_path / "platform_summary.csv", platform_rows)
    _write_csv(output_path / "frame_platform_metrics.csv", frame_rows)

    plot_warning = None
    if not args.no_plot:
        plot_warning = _save_profile_plot(
            output_path / "step_profile.png",
            scan,
            detection_profile,
            detection_sigma,
            z_valid_fraction if args.detect_profile == "ground_z" else valid_fraction,
            stable_mask,
            platforms,
            ylabel,
        )

    metadata = {}
    if args.metadata_json:
        metadata = _read_yaml(Path(args.metadata_json)) if str(args.metadata_json).lower().endswith((".yaml", ".yml")) else json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json 根节点必须是映射")
    summary = {
        "script": "analyze_daheng_gauge_repeatability",
        "input": str(input_path),
        "pattern": args.pattern,
        "frames": frame_names,
        "frame_count": len(frame_names),
        "image_shape": list(image_shape or ()),
        "scan_axis": scan_axis,
        "scan_range": list(scan_range),
        "roi_source": roi_source,
        "detect_profile": args.detect_profile,
        "steger_options": runtime["steger_options"],
        "search_region": None
        if search_region is None
        else {
            "start_px": search_region.start_px,
            "end_px": search_region.end_px,
            "source": search_region.source,
        },
        "extracted_valid_count": extraction_counts,
        "reconstructed_valid_count": reconstruction_counts,
        "detector": {
            "sigma_threshold": _to_json_value(detector.get("sigma_threshold_px")),
            "status": detector.get("status"),
        },
        "platforms": platform_rows,
        "metadata": metadata,
        "plot_warning": plot_warning,
        "config": str(config_path),
        "calibration_config": config_document.get("calibration", {}),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def _to_json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return value if np.isfinite(value) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _to_json_value(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化: {type(value).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pattern", default="laser 003*.tif")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scanner-root", type=Path, default=DEFAULT_SCANNER_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-axis", choices=("row", "column"), default=None)
    parser.add_argument("--search-roi", help="沿激光法向轴的半开区间 start:end；默认读取配置")
    parser.add_argument("--scan-range", default="900:2300", help="台阶搜索区间 start:end")
    parser.add_argument(
        "--detect-profile",
        choices=("ground_z", "center"),
        default="ground_z",
        help="默认在重建 Zg 上找台阶；center 仅用于像素域诊断",
    )
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--max-gradient", type=float, default=0.08)
    parser.add_argument("--max-step", type=float, default=0.25)
    parser.add_argument("--sigma-mad-scale", type=float, default=4.0)
    parser.add_argument("--sigma-floor", type=float, default=0.03)
    parser.add_argument("--edge-trim", type=int, default=8)
    parser.add_argument("--min-platform-width", type=int, default=30)
    parser.add_argument("--reference-platform", type=int, default=None, help="1-based 基准平台；默认 Zg 最低的平台")
    parser.add_argument("--frozen-roi", type=Path, default=None, help="复用首批 summary.json/platforms 中的冻结 ROI")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None, help="可选环境/温度元数据 JSON/YAML")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = analyze(args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"frames={summary['frame_count']}")
    print(f"platforms={len(summary['platforms'])}")
    for platform in summary["platforms"]:
        print(
            f"{platform['platform_id']}: "
            f"range={platform['analysis_range']} "
            f"height={platform['height_median_mm']:.4f} mm "
            f"sigma={platform['height_sigma_between_frames_mm']:.4f} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
