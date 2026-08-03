from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigError
from ..io_utils import canonical_mapping_hash, load_document, resolve_relative
from .models import CameraConfig, CapturePlan, CaptureTask, QualityThresholds


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是映射")
    return dict(value)


def _camera_config(value: Any, *, base: CameraConfig | None = None) -> CameraConfig:
    try:
        return (base or CameraConfig()).updated(_mapping(value, "camera"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"相机参数无效：{exc}") from exc


def _thresholds(value: Any) -> QualityThresholds:
    try:
        return QualityThresholds(**_mapping(value, "quality"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"质量阈值无效：{exc}") from exc


def _board_pattern(value: Any) -> tuple[int, int] | None:
    board = _mapping(value, "board")
    if not board:
        return None
    try:
        pattern = (int(board["pattern_cols"]), int(board["pattern_rows"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("board 需要正整数 pattern_cols 和 pattern_rows") from exc
    if min(pattern) <= 0:
        raise ConfigError("board pattern_cols/pattern_rows 必须为正整数")
    return pattern


def load_camera_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    document = load_document(source)
    backend = str(document.get("backend", "mvs"))
    if backend not in {"mvs", "synthetic"}:
        raise ConfigError("backend 必须是 mvs 或 synthetic")
    calibration_src = document.get("calibration_src", "../../calibration/src")
    return {
        "source": source,
        "backend": backend,
        "serial_number": str(document.get("serial_number", "")),
        "camera": _camera_config(document.get("camera")),
        "quality_thresholds": _thresholds(document.get("quality")),
        "board_pattern": _board_pattern(document.get("board")),
        "backend_options": _mapping(document.get("backend_options"), "backend_options"),
        "calibration_src": resolve_relative(source, calibration_src),
    }


def load_capture_plan(path: str | Path) -> CapturePlan:
    source = Path(path).expanduser().resolve()
    document = load_document(source)
    base_config = _camera_config(document.get("camera"))
    tasks_value = document.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ConfigError("capture plan 的 tasks 必须是非空列表")
    tasks: list[CaptureTask] = []
    try:
        for index, raw in enumerate(tasks_value, start=1):
            item = _mapping(raw, f"tasks[{index}]")
            task_config = _camera_config(item.pop("camera", None), base=base_config)
            tasks.append(CaptureTask(config=task_config, **item))
        output_value = document.get("output_dir")
        if not output_value:
            raise ConfigError("capture plan 缺少 output_dir")
        return CapturePlan(
            dataset_id=str(document.get("dataset_id", "")),
            output_dir=resolve_relative(source, output_value),
            backend=str(document.get("backend", "mvs")),
            serial_number=str(document.get("serial_number", "")),
            base_config=base_config,
            tasks=tuple(tasks),
            quality_thresholds=_thresholds(document.get("quality")),
            board_pattern=_board_pattern(document.get("board")),
            metadata=_mapping(document.get("metadata"), "metadata"),
            backend_options=_mapping(document.get("backend_options"), "backend_options"),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"采集计划无效：{exc}") from exc


def capture_plan_payload(plan: CapturePlan) -> dict[str, Any]:
    def camera(value: CameraConfig) -> dict[str, Any]:
        return asdict(value)

    return {
        "dataset_id": plan.dataset_id,
        "output_dir": str(plan.output_dir.expanduser().resolve()),
        "backend": plan.backend,
        "serial_number": plan.serial_number,
        "camera": camera(plan.base_config),
        "quality": asdict(plan.quality_thresholds),
        "board_pattern": list(plan.board_pattern) if plan.board_pattern else None,
        "metadata": plan.metadata,
        "backend_options": plan.backend_options,
        "tasks": [
            {
                "task_id": task.task_id,
                "frames": task.frames,
                "filename_template": task.filename_template,
                "camera": camera(task.config),
                "pose_id": task.pose_id,
                "role": task.role,
                "instruction": task.instruction,
                "settle_frames": task.settle_frames,
                "image_format": task.image_format,
                "quality_mode": task.quality_mode,
                "tags": task.tags,
            }
            for task in plan.tasks
        ],
    }


def capture_plan_hash(plan: CapturePlan) -> str:
    return canonical_mapping_hash(capture_plan_payload(plan))
