#!/usr/bin/env python3
"""Manage and audit the Phase-A baseline/laser-angle screening experiment."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.interpolate import BSpline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "geometry_baseline_angle"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration_tool.camera.models import (  # noqa: E402
    CameraConfig,
    CapturePlan,
    CaptureTask,
)
from calibration_tool.camera.plan_builder import save_generated_capture_plan  # noqa: E402
from calibration_tool.camera.quality import laser_column_metrics  # noqa: E402
from calibration_tool.io_utils import load_document, resolve_relative, sha256_file  # noqa: E402

BASELINE_LEVELS = (
    ("0", "B00"),
    ("5", "B05"),
    ("12.5", "B12p5"),
)
LASER_ANGLES = (5, 10, 15, 20)

# Future analysis code must preserve these columns from the existing CSV.
MANUAL_FIELDS = (
    "baseline_actual_mm",
    "manual_notes",
)

AUTOMATED_FIELDS = (
    "status",
    "capture_complete",
    "exclude_reason",
    "phaseA_selected",
    "data_dir",
    "captured_frame_count",
    "valid_frame_count",
    "laser_coverage",
    "laser_fwhm_p50_px",
    "laser_fwhm_p95_px",
    "laser_saturation_fraction",
    "board_reprojection_rmse_px",
    "geometry_score",
    "screening_status",
    "analyzed_at_utc",
)

CSV_FIELDS = (
    "config_id",
    "baseline_scale_reading",
    "laser_angle_deg",
    *MANUAL_FIELDS,
    *AUTOMATED_FIELDS,
)

EXPERIMENT_TYPE = "geometry_baseline_angle"
EXPERIMENT_PHASE = "phase_A_screening"
WORKING_DISTANCE_NOMINAL_MM = 1000
MEASUREMENT_PIECE_HEIGHTS_MM = (1, 10, 30)
REFERENCE_EXPOSURE_US = 1500.0
MEASUREMENT_EXPOSURE_US = 1500.0
EXPOSURE_OVERRIDES = {"B12p5_A20": 1900.0}
INVALID_FOV_CONFIG_ID = "B00_A05"
REFERENCE_DEVELOPMENT_CONFIG_ID = "B12p5_A20"
EXPECTED_TASK_FRAMES = {"reference": 50, "multiheight": 50}
CAMERA_FIELDS = (
    "exposure_us",
    "gain_db",
    "pixel_format",
    "width",
    "height",
    "offset_x",
    "offset_y",
)
ANALYSIS_NAN_FIELDS = (
    "valid_frame_count",
    "laser_coverage",
    "laser_fwhm_p50_px",
    "laser_fwhm_p95_px",
    "laser_saturation_fraction",
    "board_reprojection_rmse_px",
    "geometry_score",
    "screening_status",
    "analyzed_at_utc",
)
AUDIT_CSV_FIELDS = (
    "config_id",
    "status",
    "capture_complete",
    "exclude_reason",
    "phaseA_selected",
    "dataset_path",
    "dataset_exists",
    "manifest_exists",
    "manifest_status",
    "manifest_frame_count",
    "frames_csv_exists",
    "frames_csv_row_count",
    "reference_manifest_frames",
    "reference_csv_frames",
    "reference_image_count",
    "multiheight_manifest_frames",
    "multiheight_csv_frames",
    "multiheight_image_count",
    "missing_image_count",
    "extra_image_count",
    "exposure_us_values",
    "reference_exposure_us_values",
    "multiheight_exposure_us_values",
    "gain_db_values",
    "pixel_format_values",
    "width_values",
    "height_values",
    "offset_x_values",
    "offset_y_values",
    "camera_parameters_consistent_within_dataset",
    "camera_parameters_match_mode",
    "camera_mismatch_fields",
    "quality_passed_frame_count",
    "quality_warning_frame_count",
    "quality_warning_occurrence_count",
    "quality_warning_counts",
    "transport_warning_occurrence_count",
    "audit_warning_count",
    "audit_warnings",
    "audit_error_count",
    "audit_errors",
)


def build_initial_rows() -> list[dict[str, str]]:
    """Return the fixed 3 x 4 experiment matrix with blank result fields."""
    rows: list[dict[str, str]] = []
    for baseline_value, baseline_id in BASELINE_LEVELS:
        for laser_angle in LASER_ANGLES:
            row = {field: "" for field in CSV_FIELDS}
            row.update(
                config_id=f"{baseline_id}_A{laser_angle:02d}",
                baseline_scale_reading=baseline_value,
                laser_angle_deg=str(laser_angle),
            )
            rows.append(row)
    return rows


def initialize_experiment(experiment_dir: Path = DEFAULT_EXPERIMENT_DIR) -> Path:
    """Create experiment directories and exclusively create geometry_master.csv."""
    experiment_dir = experiment_dir.resolve()
    master_path = experiment_dir / "geometry_master.csv"

    if master_path.exists():
        raise FileExistsError(f"拒绝覆盖已存在的文件：{master_path}")

    for relative_dir in ("configs", "configs/generated", "data", "results"):
        (experiment_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    try:
        with master_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(build_initial_rows())
    except FileExistsError as exc:
        raise FileExistsError(f"拒绝覆盖已存在的文件：{master_path}") from exc

    return master_path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
    _atomic_write_text(path, stream.getvalue())


def _csv_value(value: Any) -> str | int | float:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _read_master_table(master_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not master_path.is_file():
        raise FileNotFoundError(f"geometry_master.csv 不存在，请先执行 init：{master_path}")
    with master_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        required = ("config_id", "baseline_scale_reading", "laser_angle_deg")
        missing = [field for field in required if field not in fieldnames]
        if missing:
            raise ValueError(f"geometry_master.csv 缺少字段：{', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if any(None in row for row in rows):
        raise ValueError("geometry_master.csv 存在超出表头的列")

    expected_ids = [row["config_id"] for row in build_initial_rows()]
    actual_ids = [row["config_id"].strip() for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("geometry_master.csv 的12组 config_id 或顺序与固定实验矩阵不一致")
    for field in CSV_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
        for row in rows:
            row.setdefault(field, "")
    return fieldnames, rows


def _parse_number(value: str, field: str, config_id: str) -> int | float:
    text = value.strip()
    if not text:
        raise ValueError(f"{config_id} 的 {field} 不能为空")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{config_id} 的 {field} 不是有效数字：{text}") from exc
    return int(number) if number.is_integer() else number


def _load_master_rows(master_path: Path) -> list[dict[str, str]]:
    _, rows = _read_master_table(master_path)
    return rows


def _capture_plan_for_row(row: dict[str, str], experiment_dir: Path) -> CapturePlan:
    config_id = row["config_id"].strip()
    baseline_scale = _parse_number(
        row["baseline_scale_reading"], "baseline_scale_reading", config_id
    )
    laser_angle = _parse_number(row["laser_angle_deg"], "laser_angle_deg", config_id)
    baseline_actual_text = row["baseline_actual_mm"].strip()
    baseline_actual = (
        _parse_number(baseline_actual_text, "baseline_actual_mm", config_id)
        if baseline_actual_text
        else None
    )
    measurement_camera = CameraConfig(
        exposure_us=EXPOSURE_OVERRIDES.get(config_id, MEASUREMENT_EXPOSURE_US)
    )
    reference_camera = CameraConfig(
        exposure_us=EXPOSURE_OVERRIDES.get(config_id, REFERENCE_EXPOSURE_US)
    )
    common_tags = {
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_phase": EXPERIMENT_PHASE,
        "config_id": config_id,
        "laser_state": "on",
        "measurement_pieces_fixed": True,
    }
    tasks = (
        CaptureTask(
            task_id="reference",
            pose_id="reference",
            role="reference",
            instruction=(
                "保持激光开启；将棋盘格沿固定导向移动到参考位置；"
                "激光只投射在裸露棋盘格参考面；"
                "1/10/30 mm 测量件保持固定在棋盘格上，不拆卸。"
            ),
            frames=50,
            settle_frames=5,
            image_format="tif",
            quality_mode="laser",
            filename_template="images/reference/{index04}{suffix}",
            config=reference_camera,
            tags={
                **common_tags,
                "measurement_piece_heights_mm": list(MEASUREMENT_PIECE_HEIGHTS_MM),
                "board_position": "reference",
                "laser_target": "bare_chessboard_reference_surface",
            },
        ),
        CaptureTask(
            task_id="multiheight",
            pose_id="measurement",
            role="multiheight",
            instruction=(
                "保持激光开启；将同一块棋盘格沿固定导向移动到 measurement 定位位置；"
                "激光线同时穿过 1 mm、10 mm、30 mm 测量件；"
                "测量件在整个12组实验中保持固定。"
            ),
            frames=50,
            settle_frames=5,
            image_format="tif",
            quality_mode="laser",
            filename_template="images/multiheight/{index04}{suffix}",
            config=measurement_camera,
            tags={
                **common_tags,
                "measurement_piece_heights_mm": list(MEASUREMENT_PIECE_HEIGHTS_MM),
                "board_position": "measurement",
                "laser_target": "multiheight_measurement_pieces",
            },
        ),
    )
    return CapturePlan(
        dataset_id=f"{EXPERIMENT_TYPE}_{config_id}",
        output_dir=(experiment_dir / "data" / config_id).resolve(),
        backend="mvs",
        serial_number="",
        base_config=measurement_camera,
        tasks=tasks,
        metadata={
            "experiment_type": EXPERIMENT_TYPE,
            "experiment_phase": EXPERIMENT_PHASE,
            "config_id": config_id,
            "baseline_scale_reading": baseline_scale,
            "baseline_actual_mm": baseline_actual,
            "laser_angle_deg": laser_angle,
            "working_distance_nominal_mm": WORKING_DISTANCE_NOMINAL_MM,
            "working_distance_calibrated": False,
            "baseline_scale_reading_note": "机械支架刻度，不是实际相机-激光光学基线",
        },
    )


def make_capture_plans(
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Generate one existing-schema capture plan for each matrix row."""
    experiment_dir = experiment_dir.resolve()
    rows = _load_master_rows(experiment_dir / "geometry_master.csv")
    generated_dir = experiment_dir / "configs" / "generated"
    plans = [_capture_plan_for_row(row, experiment_dir) for row in rows]
    targets = [generated_dir / f"{row['config_id'].strip()}.yaml" for row in rows]

    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"拒绝覆盖已存在的采集计划：{names}")

    return [
        save_generated_capture_plan(plan, target, overwrite=overwrite)
        for plan, target in zip(plans, targets)
    ]


def _split_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _camera_values(rows: Sequence[Mapping[str, str]], field: str) -> list[Any]:
    values: set[Any] = set()
    numeric = field != "pixel_format"
    integer = field in {"width", "height", "offset_x", "offset_y"}
    for row in rows:
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        if not numeric:
            values.add(text)
            continue
        try:
            value = float(text)
        except ValueError:
            values.add(text)
            continue
        values.add(int(value) if integer and value.is_integer() else value)
    return sorted(values, key=lambda value: (str(type(value)), str(value)))


def _values_text(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values)


def _new_audit_record(config_id: str, dataset_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {field: "" for field in AUDIT_CSV_FIELDS}
    record.update(
        config_id=config_id,
        status="incomplete",
        capture_complete=False,
        exclude_reason="",
        phaseA_selected=False,
        dataset_path=str(dataset_path),
        dataset_exists=dataset_path.is_dir(),
        manifest_exists=False,
        frames_csv_exists=False,
        camera_parameters_consistent_within_dataset=False,
        camera_parameters_match_mode=False,
        audit_warnings=[],
        audit_errors=[],
        _camera_values={field: [] for field in CAMERA_FIELDS},
    )
    return record


def _finalize_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    record["audit_warning_count"] = len(record["audit_warnings"])
    record["audit_error_count"] = len(record["audit_errors"])
    record["capture_complete"] = not record["audit_errors"]
    record["phaseA_selected"] = bool(record["capture_complete"])
    record["status"] = "captured" if record["capture_complete"] else "incomplete"
    return record


def _audit_dataset(config_id: str, dataset_path: Path) -> dict[str, Any]:
    record = _new_audit_record(config_id, dataset_path)
    errors: list[str] = record["audit_errors"]
    warnings: list[str] = record["audit_warnings"]
    if not dataset_path.is_dir():
        errors.append("dataset_missing")
        return _finalize_audit_record(record)

    manifest_path = dataset_path / "dataset_manifest.yaml"
    frames_csv_path = dataset_path / "frames.csv"
    record["manifest_exists"] = manifest_path.is_file()
    record["frames_csv_exists"] = frames_csv_path.is_file()

    manifest: dict[str, Any] = {}
    manifest_frames: list[Mapping[str, Any]] = []
    if not manifest_path.is_file():
        errors.append("dataset_manifest_missing")
    else:
        try:
            manifest = load_document(manifest_path)
        except Exception as exc:
            errors.append(f"dataset_manifest_invalid:{exc}")
        else:
            record["manifest_status"] = str(manifest.get("status", ""))
            if manifest.get("status") != "completed":
                errors.append(f"manifest_status_not_completed:{manifest.get('status')}")
            plan = manifest.get("plan")
            metadata = plan.get("metadata") if isinstance(plan, Mapping) else None
            manifest_config_id = metadata.get("config_id") if isinstance(metadata, Mapping) else None
            if manifest_config_id != config_id:
                errors.append(f"manifest_config_id_mismatch:{manifest_config_id}")
            raw_frames = manifest.get("frames")
            if isinstance(raw_frames, list) and all(isinstance(item, Mapping) for item in raw_frames):
                manifest_frames = list(raw_frames)
            else:
                errors.append("manifest_frames_invalid")
            record["manifest_frame_count"] = len(manifest_frames)
            task_states = manifest.get("tasks")
            if not isinstance(task_states, Mapping):
                errors.append("manifest_tasks_invalid")
                task_states = {}
            for task_id, expected in EXPECTED_TASK_FRAMES.items():
                state = task_states.get(task_id)
                if not isinstance(state, Mapping):
                    errors.append(f"manifest_task_missing:{task_id}")
                    continue
                captured = int(state.get("frames_captured") or 0)
                expected_in_manifest = int(state.get("frames_expected") or 0)
                if state.get("status") != "completed":
                    errors.append(f"manifest_task_not_completed:{task_id}")
                if captured != expected or expected_in_manifest != expected:
                    errors.append(
                        f"manifest_task_frame_count:{task_id}:{captured}/{expected_in_manifest}"
                    )

    csv_rows: list[dict[str, str]] = []
    if not frames_csv_path.is_file():
        errors.append("frames_csv_missing")
    else:
        try:
            with frames_csv_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                csv_fields = list(reader.fieldnames or ())
                required_fields = (
                    "task_id", "index", "filename", "quality_passed",
                    "quality_warnings", "transport_warnings", *CAMERA_FIELDS,
                )
                missing_fields = [field for field in required_fields if field not in csv_fields]
                if missing_fields:
                    errors.append(f"frames_csv_fields_missing:{'|'.join(missing_fields)}")
                csv_rows = [dict(row) for row in reader]
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"frames_csv_invalid:{exc}")
    record["frames_csv_row_count"] = len(csv_rows)

    manifest_keys = {
        (str(item.get("task_id", "")), str(item.get("index", "")), str(item.get("filename", "")))
        for item in manifest_frames
    }
    csv_keys = {
        (row.get("task_id", ""), row.get("index", ""), row.get("filename", ""))
        for row in csv_rows
    }
    if manifest_frames and csv_rows and manifest_keys != csv_keys:
        errors.append("manifest_frames_csv_mismatch")

    csv_filenames = {row.get("filename", "") for row in csv_rows if row.get("filename")}
    missing_images = {
        filename for filename in csv_filenames if not (dataset_path / Path(filename)).is_file()
    }
    actual_images: set[str] = set()
    image_suffixes = {".tif", ".tiff", ".png"}
    for task_id, expected in EXPECTED_TASK_FRAMES.items():
        task_manifest_count = sum(
            str(item.get("task_id", "")) == task_id for item in manifest_frames
        )
        task_csv_rows = [row for row in csv_rows if row.get("task_id") == task_id]
        image_dir = dataset_path / "images" / task_id
        task_images = {
            path.relative_to(dataset_path).as_posix()
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        } if image_dir.is_dir() else set()
        actual_images.update(task_images)
        record[f"{task_id}_manifest_frames"] = task_manifest_count
        record[f"{task_id}_csv_frames"] = len(task_csv_rows)
        record[f"{task_id}_image_count"] = len(task_images)
        if not image_dir.is_dir():
            errors.append(f"image_directory_missing:{task_id}")
        if task_manifest_count != expected:
            errors.append(f"manifest_frame_count:{task_id}:{task_manifest_count}")
        if len(task_csv_rows) != expected:
            errors.append(f"csv_frame_count:{task_id}:{len(task_csv_rows)}")
        if len(task_images) != expected:
            errors.append(f"image_frame_count:{task_id}:{len(task_images)}")

    extra_images = actual_images - csv_filenames
    record["missing_image_count"] = len(missing_images)
    record["extra_image_count"] = len(extra_images)
    if missing_images:
        errors.append(f"images_missing:{len(missing_images)}")
    if extra_images:
        errors.append(f"images_not_indexed:{len(extra_images)}")

    camera_values = {field: _camera_values(csv_rows, field) for field in CAMERA_FIELDS}
    record["_camera_values"] = camera_values
    for field, values in camera_values.items():
        record[f"{field}_values"] = _values_text(values)
        if not values:
            errors.append(f"camera_field_empty:{field}")
    for task_id in EXPECTED_TASK_FRAMES:
        task_rows = [row for row in csv_rows if row.get("task_id") == task_id]
        record[f"{task_id}_exposure_us_values"] = _values_text(
            _camera_values(task_rows, "exposure_us")
        )
    record["camera_parameters_consistent_within_dataset"] = all(
        len(values) == 1 for values in camera_values.values()
    )
    if csv_rows and not record["camera_parameters_consistent_within_dataset"]:
        warnings.append("camera_parameters_vary_within_dataset")

    quality_warning_counts: Counter[str] = Counter()
    transport_warning_count = 0
    quality_warning_frames = 0
    quality_passed_frames = 0
    for row in csv_rows:
        quality_warnings = _split_warnings(row.get("quality_warnings"))
        quality_warning_counts.update(quality_warnings)
        quality_warning_frames += bool(quality_warnings)
        quality_passed_frames += str(row.get("quality_passed", "")).strip().lower() == "true"
        transport_warning_count += len(_split_warnings(row.get("transport_warnings")))
    record["quality_passed_frame_count"] = quality_passed_frames
    record["quality_warning_frame_count"] = quality_warning_frames
    record["quality_warning_occurrence_count"] = sum(quality_warning_counts.values())
    record["quality_warning_counts"] = dict(sorted(quality_warning_counts.items()))
    record["transport_warning_occurrence_count"] = transport_warning_count
    if quality_warning_frames:
        warnings.append(f"quality_warning_frames:{quality_warning_frames}")
    if transport_warning_count:
        warnings.append(f"transport_warning_occurrences:{transport_warning_count}")
    return _finalize_audit_record(record)


def _apply_cross_dataset_camera_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    captured = [
        record for record in records
        if record["config_id"] != INVALID_FOV_CONFIG_ID and record["dataset_exists"]
    ]
    field_summary: dict[str, Any] = {}
    for field in CAMERA_FIELDS:
        values = sorted(
            {value for record in captured for value in record["_camera_values"][field]},
            key=lambda value: (str(type(value)), str(value)),
        )
        field_summary[field] = {"consistent": len(values) <= 1, "values": values}

    signatures = [
        tuple((field, tuple(record["_camera_values"][field])) for field in CAMERA_FIELDS)
        for record in captured
        if all(len(record["_camera_values"][field]) == 1 for field in CAMERA_FIELDS)
    ]
    mode_signature = Counter(signatures).most_common(1)[0][0] if signatures else tuple()
    mode_values = {field: list(values) for field, values in mode_signature}
    for record in records:
        if record["config_id"] == INVALID_FOV_CONFIG_ID or not record["dataset_exists"]:
            record["camera_parameters_match_mode"] = False
            record["camera_mismatch_fields"] = ""
            continue
        mismatch = [
            field for field in CAMERA_FIELDS
            if record["_camera_values"][field] != mode_values.get(field, [])
        ]
        record["camera_parameters_match_mode"] = not mismatch
        record["camera_mismatch_fields"] = "|".join(mismatch)
        if mismatch:
            record["audit_warnings"].append(
                f"camera_parameters_differ_from_mode:{'|'.join(mismatch)}"
            )
            record["audit_warning_count"] = len(record["audit_warnings"])
    return {
        "consistent": all(item["consistent"] for item in field_summary.values()),
        "fields": field_summary,
        "mode": mode_values,
    }


def _public_audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _update_master_from_audit(
    master_path: Path,
    fieldnames: list[str],
    master_rows: list[dict[str, str]],
    records: Sequence[Mapping[str, Any]],
    root: Path,
) -> None:
    by_id = {str(record["config_id"]): record for record in records}
    try:
        data_root_text = root.relative_to(master_path.parent).as_posix()
    except ValueError:
        data_root_text = str(root)
    for row in master_rows:
        config_id = row["config_id"].strip()
        if config_id == INVALID_FOV_CONFIG_ID:
            row.update(
                status="invalid_fov",
                capture_complete="false",
                exclude_reason="laser_out_of_fov",
                phaseA_selected="false",
                data_dir="",
                captured_frame_count="0",
            )
            for field in ANALYSIS_NAN_FIELDS:
                row[field] = "NaN"
            continue
        record = by_id[config_id]
        complete = bool(record["capture_complete"])
        row.update(
            status="captured" if complete else "incomplete",
            capture_complete="true" if complete else "false",
            exclude_reason="" if complete else "capture_audit_incomplete",
            phaseA_selected="true" if complete else "false",
            data_dir=f"{data_root_text}/{config_id}",
            captured_frame_count=str(record["frames_csv_row_count"] or 0),
        )
    _write_csv(master_path, fieldnames, master_rows)


def audit_captures(root: Path, master_path: Path) -> dict[str, Any]:
    """Audit Phase-A capture provenance without reading or modifying image bytes."""
    root = root.expanduser().resolve()
    master_path = master_path.expanduser().resolve()
    fieldnames, master_rows = _read_master_table(master_path)
    records: list[dict[str, Any]] = []
    for row in master_rows:
        config_id = row["config_id"].strip()
        dataset_path = root / config_id
        if config_id == INVALID_FOV_CONFIG_ID:
            record = _new_audit_record(config_id, dataset_path)
            record.update(
                status="invalid_fov",
                capture_complete=False,
                exclude_reason="laser_out_of_fov",
                phaseA_selected=False,
                dataset_exists=False,
                audit_warning_count=0,
                audit_error_count=0,
            )
        else:
            record = _audit_dataset(config_id, dataset_path)
        records.append(record)

    camera_consistency = _apply_cross_dataset_camera_audit(records)
    _update_master_from_audit(master_path, fieldnames, master_rows, records, root)
    public_records = [_public_audit_record(record) for record in records]
    normal_records = [record for record in records if record["config_id"] != INVALID_FOV_CONFIG_ID]
    summary = {
        "expected_conditions": len(master_rows),
        "captured_conditions": sum(bool(record["dataset_exists"]) for record in normal_records),
        "invalid_fov": 1,
        "complete_datasets": sum(bool(record["capture_complete"]) for record in normal_records),
        "incomplete_datasets": sum(not bool(record["capture_complete"]) for record in normal_records),
    }
    unexpected = sorted(
        path.name for path in root.iterdir()
        if path.is_dir()
        and path.name not in {row["config_id"].strip() for row in master_rows}
    ) if root.is_dir() else []
    result = {
        "schema_version": 1,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_phase": EXPERIMENT_PHASE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "master": str(master_path),
        "summary": summary,
        "camera_consistency": camera_consistency,
        "unexpected_dataset_directories": unexpected,
        "datasets": public_records,
    }
    results_dir = master_path.parent / "results"
    _write_csv(results_dir / "capture_audit.csv", AUDIT_CSV_FIELDS, public_records)
    _atomic_write_text(
        results_dir / "capture_audit.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    return result


def _load_realtime_steger(calibration_src: Path):
    module_path = calibration_src.expanduser().resolve() / "realtime_steger.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"正式 realtime_steger.py 不存在：{module_path}")
    module_name = "_geometry_experiment_realtime_steger"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载正式 Steger 模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_analysis_config(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    document = load_document(source)
    if int(document.get("schema_version") or 0) != 1:
        raise ValueError("analysis.yaml schema_version 必须为 1")
    reference = document.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("analysis.yaml 缺少 reference 映射")
    valid_fraction = float(reference.get("valid_frame_fraction_min", 0.80))
    max_gap = int(reference.get("max_interp_gap_px", -1))
    if not 0.0 < valid_fraction <= 1.0:
        raise ValueError("valid_frame_fraction_min 必须位于 (0, 1]")
    if max_gap < 0:
        raise ValueError("max_interp_gap_px 必须是非负整数")
    reference_surface = document.get("reference_surface")
    if not isinstance(reference_surface, Mapping):
        raise ValueError("analysis.yaml 缺少 reference_surface 映射")
    raw_x_range = reference_surface.get("x_range")
    x_range: tuple[int, int] | None
    if raw_x_range is None:
        x_range = None
    elif (
        isinstance(raw_x_range, Sequence)
        and not isinstance(raw_x_range, (str, bytes))
        and len(raw_x_range) == 2
    ):
        try:
            x_left = int(raw_x_range[0])
            x_right = int(raw_x_range[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_surface.x_range 必须是 [x_left, x_right]") from exc
        if x_left < 0 or x_right < x_left:
            raise ValueError("reference_surface.x_range 必须满足 0 <= x_left <= x_right")
        x_range = (x_left, x_right)
    else:
        raise ValueError("reference_surface.x_range 必须为 null 或 [x_left, x_right]")
    segment_edge_trim_px = int(reference_surface.get("segment_edge_trim_px", 2))
    smooth_spline_basis_count = int(reference_surface.get("smooth_spline_basis_count", 12))
    smooth_spline_penalty = float(reference_surface.get("smooth_spline_penalty", 1.0))
    robust_huber_delta = float(reference_surface.get("robust_huber_delta", 1.5))
    robust_max_iterations = int(reference_surface.get("robust_max_iterations", 15))
    if segment_edge_trim_px < 0:
        raise ValueError("segment_edge_trim_px 必须是非负整数")
    if smooth_spline_basis_count < 4:
        raise ValueError("smooth_spline_basis_count 至少为 4")
    if smooth_spline_penalty < 0.0:
        raise ValueError("smooth_spline_penalty 必须是非负数")
    if robust_huber_delta <= 0.0 or robust_max_iterations < 1:
        raise ValueError("robust Huber 参数无效")
    steger_value = document.get("steger_config")
    if not isinstance(steger_value, str) or not steger_value.strip():
        raise ValueError("analysis.yaml 必须指定正式 steger_config")
    return {
        "source": source,
        "valid_frame_fraction_min": valid_fraction,
        "max_interp_gap_px": max_gap,
        "reference_surface_x_range": x_range,
        "segment_edge_trim_px": segment_edge_trim_px,
        "smooth_spline_basis_count": smooth_spline_basis_count,
        "smooth_spline_penalty": smooth_spline_penalty,
        "robust_huber_delta": robust_huber_delta,
        "robust_max_iterations": robust_max_iterations,
        "steger_config": resolve_relative(source, steger_value),
    }


def _reference_frame_records(dataset: Path) -> list[dict[str, str]]:
    manifest = load_document(dataset / "dataset_manifest.yaml")
    if manifest.get("status") != "completed":
        raise ValueError(f"dataset_manifest status 不是 completed：{manifest.get('status')}")
    plan = manifest.get("plan")
    metadata = plan.get("metadata") if isinstance(plan, Mapping) else None
    config_id = metadata.get("config_id") if isinstance(metadata, Mapping) else None
    if config_id != REFERENCE_DEVELOPMENT_CONFIG_ID:
        raise ValueError(
            f"本阶段只允许分析 {REFERENCE_DEVELOPMENT_CONFIG_ID}，manifest 为 {config_id}"
        )
    frames_path = dataset / "frames.csv"
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames.csv 不存在：{frames_path}")
    with frames_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ("task_id", "index", "filename", "pixel_format")
        missing = [field for field in required if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"frames.csv 缺少字段：{', '.join(missing)}")
        records = [dict(row) for row in reader if row.get("task_id") == "reference"]
    records.sort(key=lambda row: int(row["index"]))
    if len(records) != 50:
        raise ValueError(f"reference 应为50帧，实际 {len(records)}")
    expected_indices = list(range(1, 51))
    actual_indices = [int(row["index"]) for row in records]
    if actual_indices != expected_indices:
        raise ValueError(f"reference 帧索引必须为1..50，实际 {actual_indices}")
    for row in records:
        relative = Path(row["filename"])
        if relative.parts[:2] != ("images", "reference") or ".." in relative.parts:
            raise ValueError(f"拒绝读取 reference 之外的图像：{relative}")
        image_path = dataset / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"reference 图像不存在：{image_path}")
    return records


def _read_gray_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV 无法读取图像：{path}")
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"reference 图像必须是二维 uint8/uint16：{path}")
    return image


def _sensor_max_value(image: np.ndarray, pixel_format: str) -> float:
    if image.dtype == np.uint8 or pixel_format == "Mono8":
        return 255.0
    if pixel_format == "Mono12":
        return 4095.0
    return float(np.iinfo(image.dtype).max)


def _nan_text(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric if np.isfinite(numeric) else "NaN"


def _surface_bounds(config: Mapping[str, Any], width: int, *, required: bool) -> tuple[int, int] | None:
    x_range = config["reference_surface_x_range"]
    if x_range is None:
        if required:
            raise ValueError(
                "reference_surface.x_range 仍为 null；请先运行 preview-reference-roi，"
                "人工确认棋盘格基准面的 [x_left, x_right]"
            )
        return None
    left, right = x_range
    if right >= width:
        raise ValueError(f"reference_surface.x_range 超出图像宽度 {width}：[{left}, {right}]")
    return int(left), int(right)


def _extract_reference_stacks(
    dataset: Path,
    realtime: Any,
    steger_options: Mapping[str, Any],
    *,
    detail_output: Path | None = None,
    collect_median_image: bool = False,
) -> dict[str, Any]:
    """Run the formal extractor; an existing per-frame detail file is never overwritten."""
    records = _reference_frame_records(dataset)
    per_frame_fields = (
        "frame_index", "filename", "u", "y_subpixel_px", "valid",
        "steger_response", "steger_offset_px", "steger_normal_y_abs",
        "fwhm_px", "background_dn", "peak_dn", "peak_contrast_dn",
        "quality_active", "peak_saturated", "peak_near_saturated",
        "saturated_width_px", "column_saturation_fraction",
    )
    temporary_csv: Path | None = None
    detail_stream: Any = None
    writer: csv.DictWriter[str] | None = None
    detail_was_preserved = bool(detail_output and detail_output.exists())
    if detail_output is not None and not detail_was_preserved:
        detail_output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".reference_frame_columns.", suffix=".csv.tmp", dir=str(detail_output.parent)
        )
        os.close(descriptor)
        temporary_csv = Path(temporary_name)
        detail_stream = temporary_csv.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(detail_stream, fieldnames=per_frame_fields, lineterminator="\n")
        writer.writeheader()

    y_stack: np.ndarray | None = None
    valid_stack: np.ndarray | None = None
    fwhm_stack: np.ndarray | None = None
    image_stack: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    completed = False
    try:
        for frame_position, frame_record in enumerate(records):
            image_path = dataset / Path(frame_record["filename"])
            image = _read_gray_image(image_path)
            if image_shape is None:
                image_shape = image.shape
                width = image.shape[1]
                y_stack = np.full((len(records), width), np.nan, dtype=np.float64)
                valid_stack = np.zeros((len(records), width), dtype=bool)
                fwhm_stack = np.full((len(records), width), np.nan, dtype=np.float64)
                if collect_median_image:
                    image_stack = np.empty((len(records), *image.shape), dtype=image.dtype)
            elif image.shape != image_shape:
                raise ValueError(f"reference 图像尺寸不一致：{image_path} {image.shape} != {image_shape}")
            if image_stack is not None:
                image_stack[frame_position] = image
            sensor_max = _sensor_max_value(image, frame_record.get("pixel_format", ""))
            extracted = realtime.extract_steger_columns(image, steger_options)
            metrics = laser_column_metrics(image, sensor_max_value=sensor_max)
            valid = np.asarray(extracted.valid, dtype=bool) & np.isfinite(extracted.v_px)
            assert y_stack is not None and valid_stack is not None and fwhm_stack is not None
            y_stack[frame_position, valid] = extracted.v_px[valid]
            valid_stack[frame_position] = valid
            fwhm = np.asarray(metrics["fwhm_px"], dtype=np.float64)
            fwhm_stack[frame_position] = fwhm
            if writer is not None:
                background = np.asarray(metrics["background_dn"])
                peak = np.asarray(metrics["peak_dn"])
                peak_contrast = np.asarray(metrics["peak_contrast_dn"])
                active = np.asarray(metrics["active"])
                peak_saturated = np.asarray(metrics["peak_saturated"])
                peak_near_saturated = np.asarray(metrics["peak_near_saturated"])
                saturated_width = np.asarray(metrics["saturated_width_px"])
                saturation_fraction = np.asarray(metrics["column_saturation_fraction"])
                for column in range(image.shape[1]):
                    writer.writerow({
                        "frame_index": int(frame_record["index"]),
                        "filename": frame_record["filename"],
                        "u": column,
                        "y_subpixel_px": _nan_text(extracted.v_px[column]),
                        "valid": "true" if valid[column] else "false",
                        "steger_response": _nan_text(extracted.response[column]),
                        "steger_offset_px": _nan_text(extracted.offset_px[column]),
                        "steger_normal_y_abs": _nan_text(extracted.normal_y_abs[column]),
                        "fwhm_px": int(fwhm[column]),
                        "background_dn": float(background[column]),
                        "peak_dn": float(peak[column]),
                        "peak_contrast_dn": float(peak_contrast[column]),
                        "quality_active": "true" if active[column] else "false",
                        "peak_saturated": "true" if peak_saturated[column] else "false",
                        "peak_near_saturated": "true" if peak_near_saturated[column] else "false",
                        "saturated_width_px": int(saturated_width[column]),
                        "column_saturation_fraction": float(saturation_fraction[column]),
                    })
        completed = True
    finally:
        if detail_stream is not None:
            detail_stream.close()
        if completed and temporary_csv is not None and detail_output is not None:
            os.replace(temporary_csv, detail_output)
        if temporary_csv is not None:
            temporary_csv.unlink(missing_ok=True)

    assert image_shape is not None
    assert y_stack is not None and valid_stack is not None and fwhm_stack is not None
    median_image = None
    if image_stack is not None:
        median_image = np.median(image_stack, axis=0, overwrite_input=True)
    return {
        "records": records,
        "image_shape": image_shape,
        "y_stack": y_stack,
        "valid_stack": valid_stack,
        "fwhm_stack": fwhm_stack,
        "median_image": median_image,
        "detail_was_preserved": detail_was_preserved,
    }


def _aggregate_reference_stacks(stacks: Mapping[str, Any]) -> dict[str, np.ndarray]:
    y_stack = np.asarray(stacks["y_stack"], dtype=np.float64)
    valid_stack = np.asarray(stacks["valid_stack"], dtype=bool)
    fwhm_stack = np.asarray(stacks["fwhm_stack"], dtype=np.float64)
    width = y_stack.shape[1]
    valid_count = np.sum(valid_stack, axis=0)
    valid_fraction = valid_count.astype(np.float64) / float(y_stack.shape[0])
    y_median = np.full(width, np.nan, dtype=np.float64)
    sigma = np.full(width, np.nan, dtype=np.float64)
    fwhm_p50 = np.full(width, np.nan, dtype=np.float64)
    for column in range(width):
        mask = valid_stack[:, column]
        if np.any(mask):
            values = y_stack[mask, column]
            y_median[column] = float(np.median(values))
            sigma[column] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            fwhm_p50[column] = float(np.median(fwhm_stack[mask, column]))
    return {
        "u": np.arange(width, dtype=np.float64),
        "y_median": y_median,
        "sigma": sigma,
        "fwhm_p50": fwhm_p50,
        "valid_count": valid_count,
        "valid_fraction": valid_fraction,
    }


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _fit_robust_low_df_spline(
    x: np.ndarray,
    y: np.ndarray,
    domain: tuple[int, int],
    *,
    requested_basis_count: int,
    penalty: float,
    huber_delta: float,
    max_iterations: int,
) -> tuple[BSpline, dict[str, Any]]:
    if x.size < 4:
        raise ValueError(f"reference surface 内经 segment-edge trim 后可靠点不足：{x.size} < 4")
    degree = min(3, int(x.size) - 1)
    basis_count = min(max(degree + 1, requested_basis_count), int(x.size))
    internal_count = basis_count - degree - 1
    left, right = map(float, domain)
    internal = (
        np.linspace(left, right, internal_count + 2, dtype=np.float64)[1:-1]
        if internal_count
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate((np.repeat(left, degree + 1), internal, np.repeat(right, degree + 1)))
    design = BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()
    difference = np.diff(np.eye(basis_count, dtype=np.float64), n=2, axis=0)
    weights = np.ones(x.size, dtype=np.float64)
    coefficients = np.zeros(basis_count, dtype=np.float64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        root_weights = np.sqrt(weights)
        matrix = design * root_weights[:, None]
        target = y * root_weights
        if penalty > 0.0 and difference.size:
            matrix = np.vstack((matrix, np.sqrt(penalty) * difference))
            target = np.concatenate((target, np.zeros(difference.shape[0], dtype=np.float64)))
        updated, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = y - design @ updated
        centered = residual - np.median(residual)
        scale = 1.4826 * float(np.median(np.abs(centered)))
        if scale <= np.finfo(np.float64).eps:
            coefficients = updated
            break
        cutoff = huber_delta * scale
        absolute = np.abs(centered)
        new_weights = np.ones_like(weights)
        outliers = absolute > cutoff
        new_weights[outliers] = cutoff / absolute[outliers]
        converged = np.max(np.abs(updated - coefficients)) <= 1e-9 and np.max(
            np.abs(new_weights - weights)
        ) <= 1e-6
        coefficients = updated
        weights = new_weights
        if converged:
            break
    return BSpline(knots, coefficients, degree, extrapolate=False), {
        "model": "robust_penalized_cubic_bspline" if degree == 3 else "robust_penalized_bspline",
        "degree": degree,
        "basis_count": basis_count,
        "penalty": penalty,
        "huber_delta": huber_delta,
        "iterations": iterations,
    }


def _build_surface_reference(
    aggregates: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    u = aggregates["u"]
    raw_y = aggregates["y_median"]
    valid_fraction = aggregates["valid_fraction"]
    width = u.size
    bounds = _surface_bounds(config, width, required=True)
    assert bounds is not None
    left, right = bounds
    inside = (u >= left) & (u <= right)
    raw_reliable = (
        (valid_fraction >= config["valid_frame_fraction_min"])
        & np.isfinite(raw_y)
    )
    reliable_inside = raw_reliable & inside
    source = np.full(width, "outside_reference_surface", dtype=object)
    source[inside] = "invalid"
    source[reliable_inside] = "observed"

    runs = _true_runs(reliable_inside)
    trim = config["segment_edge_trim_px"]
    for start, end in runs:
        edge_end = min(end, start + trim - 1)
        if edge_end >= start:
            source[start:edge_end + 1] = "segment_edge_excluded"
        edge_start = max(start, end - trim + 1)
        if edge_start <= end:
            source[edge_start:end + 1] = "segment_edge_excluded"

    y_short_gap = np.full(width, np.nan, dtype=np.float64)
    for (left_start, left_end), (right_start, right_end) in zip(runs, runs[1:]):
        gap_start = left_end + 1
        gap_end = right_start - 1
        gap = gap_end - gap_start + 1
        left_anchor = left_end - trim
        right_anchor = right_start + trim
        if (
            gap <= 0
            or gap > config["max_interp_gap_px"]
            or left_anchor < left_start
            or right_anchor > right_end
            or source[left_anchor] != "observed"
            or source[right_anchor] != "observed"
        ):
            continue
        target = np.arange(gap_start, gap_end + 1)
        y_short_gap[target] = np.interp(
            target, (left_anchor, right_anchor), (raw_y[left_anchor], raw_y[right_anchor])
        )
        source[target] = "short_gap_interpolated"

    fit_mask = source == "observed"
    spline, model_info = _fit_robust_low_df_spline(
        u[fit_mask],
        raw_y[fit_mask],
        bounds,
        requested_basis_count=config["smooth_spline_basis_count"],
        penalty=config["smooth_spline_penalty"],
        huber_delta=config["robust_huber_delta"],
        max_iterations=config["robust_max_iterations"],
    )
    y_smooth = np.full(width, np.nan, dtype=np.float64)
    y_smooth[inside] = spline(u[inside])
    source[(source == "invalid") & inside & np.isfinite(y_smooth)] = "smooth_model_filled"
    residual = raw_y[fit_mask] - y_smooth[fit_mask]
    return {
        "bounds": bounds,
        "inside": inside,
        "raw_reliable": raw_reliable,
        "source": source,
        "y_ref_observed": np.where(reliable_inside, raw_y, np.nan),
        "y_ref_short_gap": y_short_gap,
        "y_ref_smooth": y_smooth,
        "fit_mask": fit_mask,
        "residual": residual,
        "model_info": model_info,
    }


def _save_figure(target: Path, figure: Any) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".png", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=160)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _save_reference_roi_preview(
    output_dir: Path,
    median_image: np.ndarray,
    aggregates: Mapping[str, np.ndarray],
    valid_fraction_min: float,
    bounds: tuple[int, int] | None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = aggregates["u"]
    y = aggregates["y_median"]
    observed = (aggregates["valid_fraction"] >= valid_fraction_min) & np.isfinite(y)
    finite = median_image[np.isfinite(median_image)]
    display_min = float(np.percentile(finite, 1.0))
    display_max = float(np.percentile(finite, 99.8))
    figure, axis = plt.subplots(figsize=(18, 11), constrained_layout=True)
    axis.imshow(median_image, cmap="gray", vmin=display_min, vmax=display_max, origin="upper")
    axis.plot(u[observed], y[observed], ".", ms=1.4, color="#00e5ff", label="raw reliable Steger median")
    if bounds is not None:
        axis.axvspan(bounds[0], bounds[1], color="#76ff03", alpha=0.10, label="configured reference surface")
        axis.axvline(bounds[0], color="#76ff03", lw=1.2)
        axis.axvline(bounds[1], color="#76ff03", lw=1.2)
    axis.set_xlim(0, median_image.shape[1] - 1)
    axis.set_ylim(median_image.shape[0] - 1, 0)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("v [px]")
    axis.set_title("B12p5_A20 reference median image + raw Steger centerline (choose x_left / x_right)")
    axis.grid(True, alpha=0.20)
    axis.legend(loc="best")
    target = _save_figure(output_dir / "reference_roi_preview.png", figure)
    plt.close(figure)
    return target


def preview_reference_roi(
    dataset: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render the real 50-frame median image without selecting or building a surface model."""
    dataset = dataset.expanduser().resolve()
    if dataset.name != REFERENCE_DEVELOPMENT_CONFIG_ID:
        raise ValueError(f"本阶段只允许预览 {REFERENCE_DEVELOPMENT_CONFIG_ID}：{dataset}")
    config = _load_reference_analysis_config(analysis_config)
    output_dir = dataset / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / "reference_roi_preview.png", output_dir / "reference_roi_preview.json"]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("ROI 预览已存在，不会静默覆盖：" + ", ".join(path.name for path in existing))
    realtime = _load_realtime_steger(calibration_src)
    steger_options = realtime.load_steger_options(config["steger_config"])
    stacks = _extract_reference_stacks(
        dataset, realtime, steger_options, collect_median_image=True
    )
    aggregates = _aggregate_reference_stacks(stacks)
    width = int(stacks["image_shape"][1])
    bounds = _surface_bounds(config, width, required=False)
    preview_path = _save_reference_roi_preview(
        output_dir,
        np.asarray(stacks["median_image"]),
        aggregates,
        config["valid_frame_fraction_min"],
        bounds,
    )
    summary = {
        "schema_version": 1,
        "config_id": REFERENCE_DEVELOPMENT_CONFIG_ID,
        "frame_count": len(stacks["records"]),
        "image_shape": list(stacks["image_shape"]),
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "steger_config": str(config["steger_config"]),
        "steger_config_sha256": sha256_file(config["steger_config"]),
        "reference_surface_x_range": list(bounds) if bounds is not None else None,
        "reference_model_built": False,
        "multiheight_analyzed": False,
        "output": str(preview_path),
    }
    _atomic_write_text(outputs[1], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _source_spans(axis: Any, source: np.ndarray, colors: Mapping[str, str]) -> None:
    start = 0
    for index in range(1, len(source) + 1):
        if index == len(source) or source[index] != source[start]:
            axis.axvspan(
                start - 0.5,
                index - 0.5,
                color=colors.get(str(source[start]), "#9e9e9e"),
                alpha=0.08,
            )
            start = index


def _save_reference_model_plots(
    output_dir: Path,
    aggregates: Mapping[str, np.ndarray],
    surface: Mapping[str, Any],
    valid_fraction_min: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = aggregates["u"]
    raw_y = aggregates["y_median"]
    sigma = aggregates["sigma"]
    valid_fraction = aggregates["valid_fraction"]
    raw_reliable = surface["raw_reliable"]
    source = surface["source"]
    fit_mask = surface["fit_mask"]
    y_smooth = surface["y_ref_smooth"]
    left, right = surface["bounds"]
    colors = {
        "observed": "#2e7d32",
        "short_gap_interpolated": "#fb8c00",
        "smooth_model_filled": "#8e24aa",
        "segment_edge_excluded": "#d32f2f",
        "outside_reference_surface": "#616161",
        "invalid": "#9e9e9e",
    }

    paths: list[Path] = []
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    raw_source = np.where(raw_reliable, "observed", "invalid")
    _source_spans(axis, raw_source, colors)
    axis.plot(u[raw_reliable], raw_y[raw_reliable], ".", ms=2, color="#1565c0", label="raw observed")
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.invert_yaxis()
    axis.set_xlabel("u [px]")
    axis.set_ylabel("raw median Steger y [px]")
    axis.set_title("B12p5_A20 raw reference centerline (all physical surfaces, no filtering hidden)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "reference_centerline_raw.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    _source_spans(axis, source, colors)
    axis.axvspan(left, right, color="#66bb6a", alpha=0.06)
    axis.axvline(left, color="#1b5e20", ls="--", lw=1.0)
    axis.axvline(right, color="#1b5e20", ls="--", lw=1.0)
    axis.plot(u[fit_mask], raw_y[fit_mask], ".", ms=2.5, color="#1565c0", label="trimmed reliable observed")
    axis.plot(u[surface["inside"]], y_smooth[surface["inside"]], color="#d81b60", lw=1.4, label="y_ref_smooth")
    excluded = source == "segment_edge_excluded"
    axis.plot(u[excluded], raw_y[excluded], "x", ms=3, color="#d32f2f", label="segment-edge excluded")
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.invert_yaxis()
    axis.set_xlabel("u [px]")
    axis.set_ylabel("reference y [px]")
    axis.set_title(f"B12p5_A20 reference surface [{left}, {right}] and robust smooth curve")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "reference_centerline_smooth.png", figure))
    plt.close(figure)

    residual_full = np.full(u.size, np.nan, dtype=np.float64)
    residual_full[fit_mask] = raw_y[fit_mask] - y_smooth[fit_mask]
    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.axhline(0.0, color="#424242", lw=0.8)
    axis.plot(u[fit_mask], residual_full[fit_mask], ".", ms=2.5, color="#6a1b9a")
    axis.axvline(left, color="#1b5e20", ls="--", lw=1.0)
    axis.axvline(right, color="#1b5e20", ls="--", lw=1.0)
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_xlabel("u [px]")
    axis.set_ylabel("observed - y_ref_smooth [px]")
    axis.set_title("B12p5_A20 robust reference model residual")
    axis.grid(True, alpha=0.25)
    paths.append(_save_figure(output_dir / "reference_model_residual.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.plot(u, valid_fraction, color="#1565c0", lw=0.8)
    axis.axhline(valid_fraction_min, color="#d32f2f", ls="--", lw=1.0)
    axis.axvspan(left, right, color="#66bb6a", alpha=0.08)
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("valid frame fraction")
    axis.set_title("B12p5_A20 reference Steger validity across 50 frames")
    axis.grid(True, alpha=0.25)
    paths.append(_save_figure(output_dir / "reference_valid_fraction.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.plot(u[fit_mask], sigma[fit_mask], ".", ms=2.5, color="#6a1b9a")
    axis.axvspan(left, right, color="#66bb6a", alpha=0.08)
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_xlabel("u [px]")
    axis.set_ylabel("sigma_ref [px]")
    axis.set_title("B12p5_A20 repeatability of trimmed surface observations")
    axis.grid(True, alpha=0.25)
    paths.append(_save_figure(output_dir / "reference_repeatability.png", figure))
    plt.close(figure)
    return paths


def analyze_reference(
    dataset: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a surface-bounded B12p5_A20/reference curve; never read multiheight."""
    dataset = dataset.expanduser().resolve()
    if dataset.name != REFERENCE_DEVELOPMENT_CONFIG_ID:
        raise ValueError(f"本阶段只允许分析 {REFERENCE_DEVELOPMENT_CONFIG_ID}：{dataset}")
    config = _load_reference_analysis_config(analysis_config)
    if config["reference_surface_x_range"] is None:
        _surface_bounds(config, 1, required=True)
    output_dir = dataset / "analysis"
    detail_output = output_dir / "reference_frame_columns.csv"
    derived_outputs = [
        output_dir / "reference_by_column.csv",
        output_dir / "reference_centerline_raw.png",
        output_dir / "reference_centerline_smooth.png",
        output_dir / "reference_model_residual.png",
        output_dir / "reference_valid_fraction.png",
        output_dir / "reference_repeatability.png",
        output_dir / "reference_analysis.json",
    ]
    existing = [path for path in derived_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "reference 派生输出已存在，不会静默覆盖：" + ", ".join(path.name for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    realtime = _load_realtime_steger(calibration_src)
    steger_options = realtime.load_steger_options(config["steger_config"])
    stacks = _extract_reference_stacks(
        dataset, realtime, steger_options, detail_output=detail_output
    )
    aggregates = _aggregate_reference_stacks(stacks)
    surface = _build_surface_reference(aggregates, config)
    source = surface["source"]
    fit_mask = surface["fit_mask"]
    sigma = aggregates["sigma"]
    fwhm = aggregates["fwhm_p50"]
    y_observed = surface["y_ref_observed"]
    y_smooth = surface["y_ref_smooth"]
    y_short_gap = surface["y_ref_short_gap"]

    rows = [
        {
            "u": column,
            "y_ref_observed_px": _nan_text(y_observed[column]),
            "y_ref_short_gap_px": _nan_text(y_short_gap[column]),
            "y_ref_smooth_px": _nan_text(y_smooth[column]),
            "sigma_ref_px": _nan_text(sigma[column] if np.isfinite(y_observed[column]) else np.nan),
            "valid_fraction": float(aggregates["valid_fraction"][column]),
            "valid_frame_count": int(aggregates["valid_count"][column]),
            "fwhm_p50_px": _nan_text(fwhm[column] if np.isfinite(y_observed[column]) else np.nan),
            "source": str(source[column]),
        }
        for column in range(len(source))
    ]
    _write_csv(
        output_dir / "reference_by_column.csv",
        (
            "u", "y_ref_observed_px", "y_ref_short_gap_px", "y_ref_smooth_px",
            "sigma_ref_px", "valid_fraction", "valid_frame_count", "fwhm_p50_px", "source",
        ),
        rows,
    )
    plot_paths = _save_reference_model_plots(
        output_dir, aggregates, surface, config["valid_frame_fraction_min"]
    )
    source_names = (
        "observed", "short_gap_interpolated", "smooth_model_filled",
        "segment_edge_excluded", "outside_reference_surface", "invalid",
    )
    source_counts = {name: int(np.count_nonzero(source == name)) for name in source_names}
    surface_width = int(surface["bounds"][1] - surface["bounds"][0] + 1)
    residual = np.asarray(surface["residual"], dtype=np.float64)
    observed_sigma = sigma[fit_mask]
    statistics = {
        "reference_surface_width_px": surface_width,
        "observed_fraction_inside_surface": float(source_counts["observed"] / surface_width),
        "model_filled_fraction": float(source_counts["smooth_model_filled"] / surface_width),
        "sigma_ref_p50_px": float(np.median(observed_sigma)),
        "sigma_ref_p95_px": float(np.percentile(observed_sigma, 95)),
        "reference_model_residual_rmse_px": float(np.sqrt(np.mean(np.square(residual)))),
        "reference_model_residual_p95_px": float(np.percentile(np.abs(residual), 95)),
    }
    summary = {
        "schema_version": 2,
        "config_id": REFERENCE_DEVELOPMENT_CONFIG_ID,
        "task_id": "reference",
        "frame_count": len(stacks["records"]),
        "image_shape": list(stacks["image_shape"]),
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "steger_module": str((calibration_src / "realtime_steger.py").resolve()),
        "steger_config": str(config["steger_config"]),
        "steger_config_sha256": sha256_file(config["steger_config"]),
        "steger_options": steger_options,
        "analysis_config": str(config["source"]),
        "analysis_config_sha256": sha256_file(config["source"]),
        "reference_surface_x_range": list(surface["bounds"]),
        "segment_edge_trim_px": config["segment_edge_trim_px"],
        "valid_frame_fraction_min": config["valid_frame_fraction_min"],
        "max_interp_gap_px": config["max_interp_gap_px"],
        "source_counts": source_counts,
        "statistics": statistics,
        "smooth_model": surface["model_info"],
        "per_frame_steger_output_preserved": bool(stacks["detail_was_preserved"]),
        "global_line_fit_applied": False,
        "model_extrapolated_outside_reference_surface": False,
        "multiheight_analyzed": False,
        "outputs": [str(path) for path in (detail_output, output_dir / "reference_by_column.csv", *plot_paths)],
    }
    _atomic_write_text(
        output_dir / "reference_analysis.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基线刻度—激光倾角 Phase-A 快速筛选实验工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="初始化实验目录和 geometry_master.csv")
    make_plan_parser = subparsers.add_parser(
        "make-plan", help="根据 geometry_master.csv 生成12份采集计划"
    )
    make_plan_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式覆盖 configs/generated 中已有的采集计划",
    )
    audit_parser = subparsers.add_parser(
        "audit-captures", help="审计 Phase-A 数据集、帧索引和实际相机参数"
    )
    audit_parser.add_argument("--root", type=Path, required=True, help="11组已采集 dataset 的根目录")
    audit_parser.add_argument("--master", type=Path, required=True, help="geometry_master.csv 路径")
    preview_parser = subparsers.add_parser(
        "preview-reference-roi",
        help="输出 B12p5_A20 的50帧中位数图和原始 Steger 中心线，供人工选择 x_range",
    )
    preview_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / REFERENCE_DEVELOPMENT_CONFIG_ID,
    )
    preview_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    preview_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    preview_parser.add_argument("--overwrite", action="store_true")
    reference_parser = subparsers.add_parser(
        "analyze-reference",
        help="仅分析 B12p5_A20/reference，不读取 multiheight",
    )
    reference_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / REFERENCE_DEVELOPMENT_CONFIG_ID,
    )
    reference_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    reference_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    reference_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        try:
            master_path = initialize_experiment()
        except FileExistsError as exc:
            print(f"错误：{exc}")
            return 2
        print(f"已创建：{master_path}")
        return 0
    if args.command == "make-plan":
        try:
            paths = make_capture_plans(overwrite=args.overwrite)
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"已生成 {len(paths)} 份采集计划：{paths[0].parent}")
        return 0
    if args.command == "audit-captures":
        try:
            result = audit_captures(args.root, args.master)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        summary = result["summary"]
        print(f"expected conditions = {summary['expected_conditions']}")
        print(f"captured conditions = {summary['captured_conditions']}")
        print(f"invalid_fov = {summary['invalid_fov']}")
        print(f"complete datasets = {summary['complete_datasets']}")
        print(f"incomplete datasets = {summary['incomplete_datasets']}")
        return 0 if summary["incomplete_datasets"] == 0 else 1
    if args.command == "preview-reference-roi":
        try:
            summary = preview_reference_roi(
                args.dataset,
                args.config,
                args.calibration_src,
                overwrite=args.overwrite,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"config_id = {summary['config_id']}")
        print(f"reference frames = {summary['frame_count']}")
        print(f"reference_surface.x_range = {summary['reference_surface_x_range']}")
        print(f"preview = {summary['output']}")
        print("multiheight analyzed = false")
        return 0
    if args.command == "analyze-reference":
        try:
            summary = analyze_reference(
                args.dataset,
                args.config,
                args.calibration_src,
                overwrite=args.overwrite,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        counts = summary["source_counts"]
        print(f"config_id = {summary['config_id']}")
        print(f"reference frames = {summary['frame_count']}")
        print(f"observed columns = {counts['observed']}")
        print(f"short-gap interpolated columns = {counts['short_gap_interpolated']}")
        print(f"smooth-model filled columns = {counts['smooth_model_filled']}")
        print(f"segment-edge excluded columns = {counts['segment_edge_excluded']}")
        print(f"outside reference surface columns = {counts['outside_reference_surface']}")
        print(f"invalid columns = {counts['invalid']}")
        print(f"multiheight analyzed = {str(summary['multiheight_analyzed']).lower()}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
