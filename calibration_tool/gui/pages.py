from __future__ import annotations

import csv
import threading
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
from PySide6.QtCore import QSignalBlocker, QThreadPool, Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..camera import build_camera_provider, load_camera_config, load_capture_plan, run_capture_plan
from ..camera.config import capture_plan_hash
from ..camera.capture import _new_manifest, _save_state, _write_image
from ..camera.plan_builder import (
    CaptureRecipe,
    build_capture_plan_from_recipe,
    capture_plan_summary,
    save_generated_capture_plan,
)
from ..acceptance import build_acceptance_report
from ..camera.models import CameraConfig, CapturePlan, CaptureTask
from ..camera.quality import analyze_frame, quality_to_dict
from ..io_utils import load_document, resolve_relative
from ..io_utils import sha256_file
from ..workflow import run_workflow
from .project import WizardProject
from .result_artifacts import (
    ResultArtifact,
    capture_artifacts_record,
    discover_result_artifacts as discover_artifact_records,
)
from .workflow_inputs import build_workflow_update_preview, save_workflow_update
from .acceptance_plan import (
    default_acceptance_plan_path,
    ensure_default_acceptance_plan,
    update_acceptance_plan_from_workflow,
)
from .widgets import ImagePreview, ResidualPlot
from .workers import FunctionWorker, PreviewThread
from .capture_controller import CaptureTaskGate
from .capture_recipe_widget import CaptureRecipeTable


def _preview_settling_info(quality: Mapping[str, Any]) -> tuple[bool, int]:
    """读取 PreviewThread 的稳定帧元数据，兼容旧的质量字典。"""

    settling = bool(quality.get("settling", False))
    try:
        remaining = max(0, int(quality.get("settle_frames_remaining", 0)))
    except (TypeError, ValueError):
        remaining = 0
    return settling, remaining


class ProjectPage(QWidget):
    project_changed = Signal(object)

    def __init__(self, default_camera_config: Path, parent=None) -> None:
        super().__init__(parent)
        self.default_camera_config = default_camera_config
        self.project: WizardProject | None = None
        layout = QVBoxLayout(self)
        title = QLabel("1. 创建或打开标定项目")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        form = QFormLayout()
        self.project_id = QLineEdit("line-laser-calibration")
        self.workspace = QLineEdit(str(default_camera_config.parent.parent / "projects" / "default"))
        self.camera_config = QLineEdit(str(default_camera_config))
        self.workflow_plan = QLineEdit("")
        self.acceptance_plan = QLineEdit("")
        self.pattern_cols = QSpinBox(); self.pattern_cols.setRange(2, 50); self.pattern_cols.setValue(11)
        self.pattern_rows = QSpinBox(); self.pattern_rows.setRange(2, 50); self.pattern_rows.setValue(8)
        self.square_size = QDoubleSpinBox(); self.square_size.setRange(0.01, 1000); self.square_size.setValue(20); self.square_size.setSuffix(" mm")
        form.addRow("项目 ID", self.project_id)
        form.addRow("项目工作目录", _path_row(self.workspace, self, directory=True))
        form.addRow("相机配置", _path_row(self.camera_config, self, file_filter="YAML (*.yaml *.yml)"))
        form.addRow("标定 workflow", _path_row(self.workflow_plan, self, file_filter="YAML (*.yaml *.yml)"))
        form.addRow("验收计划", _path_row(self.acceptance_plan, self, file_filter="YAML (*.yaml *.yml)"))
        form.addRow("棋盘内角点列数", self.pattern_cols)
        form.addRow("棋盘内角点行数", self.pattern_rows)
        form.addRow("方格尺寸", self.square_size)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("打开项目…")
        self.save_button = QPushButton("保存项目…")
        self.apply_button = QPushButton("应用到向导")
        self.apply_button.setDefault(True)
        buttons.addWidget(self.open_button); buttons.addWidget(self.save_button); buttons.addStretch(); buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)
        layout.addStretch()
        self.open_button.clicked.connect(self._open)
        self.save_button.clicked.connect(self._save)
        self.apply_button.clicked.connect(self.apply)

    def apply(self) -> WizardProject | None:
        try:
            project = self._from_fields()
        except Exception as exc:
            QMessageBox.critical(self, "项目配置无效", str(exc))
            return None
        self.project = project
        self.project_changed.emit(project)
        return project

    def _from_fields(self) -> WizardProject:
        workspace = Path(self.workspace.text()).expanduser().resolve()
        workflow = self.workflow_plan.text().strip()
        capture_output = (
            self.project.capture_output
            if self.project is not None and self.project.capture_output is not None
            else workspace / "data"
        )
        # 默认工作区为 projects/default，因此默认数据集位于
        # projects/default/data；已加载项目仍保留 YAML 中的 capture_output。
        return WizardProject(
            project_id=self.project_id.text().strip(),
            workspace=workspace,
            camera_config=Path(self.camera_config.text()),
            workflow_plan=Path(workflow) if workflow else None,
            acceptance_plan=Path(self.acceptance_plan.text().strip()) if self.acceptance_plan.text().strip() else None,
            capture_output=capture_output,
            pattern_cols=self.pattern_cols.value(),
            pattern_rows=self.pattern_rows.value(),
            square_size_mm=self.square_size.value(),
            source_path=self.project.source_path if self.project else None,
            extra=self.project.extra if self.project else {},
        )

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开标定项目", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.set_project(WizardProject.load(path))
            self.apply()
        except Exception as exc:
            QMessageBox.critical(self, "无法打开项目", str(exc))

    def _save(self) -> None:
        project = self.apply()
        if project is None:
            return
        initial = str(project.source_path or (project.workspace / "wizard_project.yaml"))
        path, _ = QFileDialog.getSaveFileName(self, "保存标定项目", initial, "YAML (*.yaml)")
        if path:
            try:
                project.workspace.mkdir(parents=True, exist_ok=True)
                project.save(path)
            except Exception as exc:
                QMessageBox.critical(self, "无法保存项目", str(exc))

    def set_project(self, project: WizardProject) -> None:
        self.project = project
        self.project_id.setText(project.project_id)
        self.workspace.setText(str(project.workspace))
        self.camera_config.setText(str(project.camera_config))
        self.workflow_plan.setText(str(project.workflow_plan or ""))
        self.acceptance_plan.setText(str(project.acceptance_plan or ""))
        self.pattern_cols.setValue(project.pattern_cols)
        self.pattern_rows.setValue(project.pattern_rows)
        self.square_size.setValue(project.square_size_mm)


class CameraPage(QWidget):
    status_changed = Signal(str)
    frame_ready = Signal(object, object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.runtime: dict[str, Any] | None = None
        self.preview_thread: PreviewThread | None = None
        self.last_frame = None
        self.last_quality: dict[str, Any] | None = None
        self._pending_capture: tuple[int, Callable[[Any, dict[str, Any]], None]] | None = None
        self._workers: set[FunctionWorker] = set()
        layout = QVBoxLayout(self)
        title = QLabel("2. 连接相机并调整采集参数"); title.setObjectName("pageTitle"); layout.addWidget(title)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        controls = QWidget(); form = QFormLayout(controls)
        self.config_path = QLineEdit()
        load_button = QPushButton("加载")
        row = QHBoxLayout(); row.addWidget(self.config_path, 1); row.addWidget(load_button)
        form.addRow("相机配置", row)
        self.backend_label = QLabel("--")
        self.devices = QComboBox()
        self.refresh_button = QPushButton("枚举相机")
        device_row = QHBoxLayout(); device_row.addWidget(self.devices, 1); device_row.addWidget(self.refresh_button)
        form.addRow("后端", self.backend_label); form.addRow("设备", device_row)
        self.exposure = QDoubleSpinBox(); self.exposure.setRange(1, 10_000_000); self.exposure.setDecimals(1); self.exposure.setSuffix(" μs")
        self.gain = QDoubleSpinBox(); self.gain.setRange(-100, 100); self.gain.setDecimals(2); self.gain.setSuffix(" dB")
        self.pixel_format = QComboBox(); self.pixel_format.addItems(["Mono8", "Mono12"])
        self.offset_x = QSpinBox(); self.offset_x.setRange(0, 10000)
        self.offset_y = QSpinBox(); self.offset_y.setRange(0, 10000)
        self.width = QSpinBox(); self.width.setRange(1, 20000)
        self.height = QSpinBox(); self.height.setRange(1, 20000)
        self.quality_mode = QComboBox()
        self.quality_mode.addItem("激光线（激光开启）", "laser")
        self.quality_mode.addItem("棋盘格（内参时关闭激光）", "chessboard")
        self.quality_mode.addItem("通用曝光", "generic")
        for label, widget in (("曝光", self.exposure), ("增益", self.gain), ("像素格式", self.pixel_format),
                              ("Offset X", self.offset_x), ("Offset Y", self.offset_y), ("宽度", self.width),
                              ("高度", self.height), ("质量模式", self.quality_mode)):
            form.addRow(label, widget)
        self.apply_live_button = QPushButton("应用曝光/增益")
        self.auto_stretch = QCheckBox("自动拉伸预览（仅改变显示，不代表实际曝光）")
        self.auto_stretch.setChecked(False)
        form.addRow("在线参数", self.apply_live_button)
        form.addRow("显示映射", self.auto_stretch)
        self.quality_help = QLabel()
        self.quality_help.setWordWrap(True)
        form.addRow("筛查内容", self.quality_help)
        action_row = QHBoxLayout()
        self.preview_button = QPushButton("开始取流")
        self.stop_button = QPushButton("停止")
        self.snapshot_button = QPushButton("保存当前帧…")
        action_row.addWidget(self.preview_button); action_row.addWidget(self.stop_button); action_row.addWidget(self.snapshot_button)
        form.addRow(action_row)
        self.status = QLabel("尚未加载相机配置"); self.status.setWordWrap(True); form.addRow("状态", self.status)
        self.quality = QLabel("--"); self.quality.setWordWrap(True); form.addRow("质量", self.quality)
        splitter.addWidget(controls)
        self.preview = ImagePreview(); splitter.addWidget(self.preview); splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        load_button.clicked.connect(lambda: self.load_config(Path(self.config_path.text())))
        self.refresh_button.clicked.connect(self.enumerate_devices)
        self.preview_button.clicked.connect(self.start_preview)
        self.stop_button.clicked.connect(self.stop_preview)
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.apply_live_button.clicked.connect(self.apply_live_parameters)
        self.exposure.editingFinished.connect(self.apply_live_parameters)
        self.gain.editingFinished.connect(self.apply_live_parameters)
        self.auto_stretch.toggled.connect(
            lambda checked: self.preview.refresh_display(auto_stretch=checked)
        )
        self.quality_mode.currentIndexChanged.connect(self._update_quality_help)
        self._update_quality_help()

    def load_config(self, path: Path) -> bool:
        try:
            runtime = load_camera_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "相机配置无效", str(exc)); return False
        self.runtime = runtime
        self.config_path.setText(str(runtime["source"]))
        self.backend_label.setText(runtime["backend"])
        config = runtime["camera"]
        self.exposure.setValue(config.exposure_us); self.gain.setValue(config.gain_db)
        self.pixel_format.setCurrentText(config.pixel_format)
        self.offset_x.setValue(config.offset_x); self.offset_y.setValue(config.offset_y)
        self.width.setValue(config.width); self.height.setValue(config.height)
        self.status.setText("配置已加载，请枚举相机")
        return True

    def current_config(self):
        if self.runtime is None:
            raise RuntimeError("请先加载相机配置")
        return replace(
            self.runtime["camera"], exposure_us=self.exposure.value(), gain_db=self.gain.value(),
            pixel_format=self.pixel_format.currentText(), offset_x=self.offset_x.value(), offset_y=self.offset_y.value(),
            width=self.width.value(), height=self.height.value(),
        )

    def selected_serial(self) -> str:
        return str(self.devices.currentData() or (self.runtime or {}).get("serial_number", ""))

    def enumerate_devices(self) -> None:
        if self.runtime is None:
            QMessageBox.warning(self, "尚未配置", "请先加载相机配置"); return
        self.refresh_button.setEnabled(False); self.status.setText("正在枚举相机…")
        runtime = self.runtime
        worker = FunctionWorker(lambda _progress: build_camera_provider(
            runtime["backend"], calibration_src=runtime["calibration_src"], backend_options=runtime["backend_options"]
        ).list_devices())
        worker.signals.result.connect(self._set_devices)
        worker.signals.error.connect(lambda message: self._show_error("相机枚举失败", message))
        worker.signals.finished.connect(lambda: self.refresh_button.setEnabled(True))
        self._start_worker(worker)

    def _set_devices(self, devices: list[Any]) -> None:
        self.devices.clear()
        for device in devices:
            self.devices.addItem(device.display_name, device.serial_number)
        self.status.setText(f"找到 {len(devices)} 台相机")

    def start_preview(self, initial_discard_frames: int = 3) -> None:
        if self.runtime is None:
            QMessageBox.warning(self, "尚未配置", "请先加载相机配置"); return
        self.stop_preview()
        try:
            provider = build_camera_provider(
                self.runtime["backend"], calibration_src=self.runtime["calibration_src"], backend_options=self.runtime["backend_options"]
            )
            thread = PreviewThread(
                provider, self.selected_serial(), self.current_config(), str(self.quality_mode.currentData()),
                self.runtime["quality_thresholds"], self.runtime["board_pattern"],
                initial_discard_frames=initial_discard_frames, parent=self,
            )
        except Exception as exc:
            self._show_error("无法开始取流", str(exc)); return
        self.preview_thread = thread
        thread.opened.connect(lambda device, config: self.status.setText(
            f"取流中 · {device.model} · SN {device.serial_number} · {config.width}×{config.height}"
        ))
        thread.frame_ready.connect(self._on_frame)
        thread.failed.connect(lambda message: self._show_error("取流失败", message))
        thread.settings_applied.connect(self._settings_applied)
        thread.parameter_update_failed.connect(
            lambda message: self._show_error("在线参数更新失败", message)
        )
        thread.finished.connect(lambda thread=thread: self._preview_finished(thread))
        self.preview_button.setEnabled(False); self.stop_button.setEnabled(True)
        self._set_restart_controls_enabled(False)
        thread.start()

    def _on_frame(self, frame, quality: dict[str, Any]) -> None:
        self.last_frame = frame
        self.last_quality = quality
        sensor_max = self.preview_thread.config.sensor_max_value if self.preview_thread else self.current_config().sensor_max_value
        self.preview.set_array(
            frame.image,
            auto_stretch=self.auto_stretch.isChecked(),
            sensor_max_value=sensor_max,
        )
        warnings = "、".join(_quality_warning_text(item) for item in quality["warnings"]) or "通过"
        thresholds = self.runtime["quality_thresholds"] if self.runtime else None
        coverage_text = _laser_quality_metrics_text(quality, thresholds)
        chess_hint = quality.get("chessboard_hint")
        chess_text = f" · {chess_hint}" if chess_hint else ""
        settling, settle_remaining = _preview_settling_info(quality)
        settling_text = f" · 正在稳定，剩余 {settle_remaining} 帧" if settling else ""
        self.quality.setText(
            f"{warnings} · 动态范围 {quality['dynamic_range_u8']:.1f} DN8 · "
            f"清晰度 {quality['focus_laplacian']:.1f}{coverage_text}{chess_text}{settling_text}"
        )
        self.frame_ready.emit(frame, quality)
        pending = self._pending_capture
        if pending is not None:
            remaining, callback = pending
            if remaining > 0:
                self._pending_capture = (remaining - 1, callback)
            else:
                self._pending_capture = None
                callback(frame, quality)

    def _update_quality_help(self) -> None:
        descriptions = {
            "generic": "检查过曝、欠曝、全局动态范围；清晰度只显示数值，暂不设统一阈值。",
            "laser": "允许暗背景，检查激光线对比度、横向覆盖率、过曝和动态范围。",
            "chessboard": "检查曝光、动态范围和完整内角点检测。内参棋盘图应关闭激光。",
        }
        self.quality_help.setText(descriptions[str(self.quality_mode.currentData())])

    def stop_preview(self) -> None:
        self.cancel_pending_capture()
        thread = self.preview_thread
        if thread is not None and thread.isRunning():
            self.status.setText("正在停止取流…")
            if not thread.stop():
                self.status.setText("相机停止超时，请检查连接")
                return
            self._preview_finished(thread)

    def capture_after_settle(
        self,
        discard_frames: int,
        callback: Callable[[Any, dict[str, Any]], None],
    ) -> bool:
        """在当前预览流中丢弃指定帧后，把下一帧交给回调。"""
        thread = self.preview_thread
        if thread is None or thread.isFinished() or self._pending_capture is not None:
            return False
        self._pending_capture = (max(0, int(discard_frames)), callback)
        return True

    def request_preview_task(
        self,
        config: CameraConfig,
        quality_mode: str,
        settle_frames: int = 0,
    ) -> bool:
        """让现有 PreviewThread 在线切换任务配置并继续发帧。"""

        thread = self.preview_thread
        if thread is None or not thread.isRunning():
            return False
        return thread.request_task_config(config, quality_mode, settle_frames)

    def cancel_pending_capture(self) -> None:
        self._pending_capture = None

    def apply_live_parameters(self) -> None:
        thread = self.preview_thread
        if thread is None or not thread.isRunning():
            self.status.setText("曝光/增益将在下次开始取流时应用")
            return
        self.status.setText(
            f"正在应用曝光 {self.exposure.value():g} μs、增益 {self.gain.value():g} dB…"
        )
        thread.request_exposure_gain(self.exposure.value(), self.gain.value())

    def _settings_applied(self, config) -> None:
        with QSignalBlocker(self.exposure):
            self.exposure.setValue(config.exposure_us)
        with QSignalBlocker(self.gain):
            self.gain.setValue(config.gain_db)
        self.status.setText(
            f"取流中 · 相机回读：曝光 {config.exposure_us:g} μs，增益 {config.gain_db:g} dB"
        )

    def _set_restart_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.devices, self.refresh_button, self.pixel_format, self.offset_x,
            self.offset_y, self.width, self.height, self.quality_mode,
        ):
            widget.setEnabled(enabled)

    def _preview_finished(self, thread: PreviewThread | None = None) -> None:
        if thread is not None and self.preview_thread is not thread:
            return
        self.cancel_pending_capture()
        self.preview_button.setEnabled(True); self.stop_button.setEnabled(False)
        self._set_restart_controls_enabled(True)
        if not self.status.text().startswith("取流失败"):
            self.status.setText("取流已停止")
        self.preview_thread = None

    def save_snapshot(self) -> None:
        if self.last_frame is None:
            QMessageBox.information(self, "没有图像", "请先开始取流"); return
        default_suffix = ".tif" if self.last_frame.image.dtype.itemsize > 1 else ".png"
        path, _ = QFileDialog.getSaveFileName(self, "保存当前帧", f"snapshot{default_suffix}", "TIFF (*.tif);;PNG (*.png)")
        if not path:
            return
        ok, encoded = cv2.imencode(Path(path).suffix, self.last_frame.image)
        if not ok:
            QMessageBox.critical(self, "保存失败", "OpenCV 无法编码该图像"); return
        Path(path).write_bytes(encoded.tobytes())

    def _show_error(self, title: str, message: str) -> None:
        self.status.setText(f"{title}：{message}")
        QMessageBox.critical(self, title, message)

    def _start_worker(self, worker: FunctionWorker) -> None:
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.thread_pool.start(worker)


class CapturePage(QWidget):
    capture_finished = Signal(object)
    request_camera_page = Signal()

    def __init__(self, thread_pool: QThreadPool, camera_page: CameraPage, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool; self.camera_page = camera_page; self._workers: set[FunctionWorker] = set()
        self.loaded_plan: CapturePlan | None = None
        self.loaded_plan_path: Path | None = None
        self._generated_recipe: CaptureRecipe | None = None
        self._plan_dirty = True
        self._capture_gate: CaptureTaskGate | None = None
        self._capture_cancel_event: threading.Event | None = None
        self._capture_worker: FunctionWorker | None = None
        self._guided_capture_active = False
        self._last_completed_pose: str | None = None
        self.last_capture_artifacts: dict[str, str] | None = None
        self.guided_preview_index: int | None = None
        self._preview_request_token = 0
        self._pending_preview_callback: Callable[[], None] | None = None
        self._capture_in_progress = False
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.scroll_content_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_content)
        page_layout.addWidget(self.scroll_area)
        layout = self.scroll_content_layout
        title = QLabel("3. 批量采集标定图像"); title.setObjectName("pageTitle"); layout.addWidget(title)
        form = QFormLayout()
        self.plan_path = QLineEdit()
        self.load_plan_button = QPushButton("加载计划")
        plan_row = QHBoxLayout(); plan_row.addWidget(_path_row(self.plan_path, self, file_filter="YAML (*.yaml *.yml)"), 1); plan_row.addWidget(self.load_plan_button)
        self.output = QLineEdit(); self.dataset_id = QLineEdit("laser_plane")
        self.image_format = QComboBox(); self.image_format.addItem("TIFF（正式标定默认）", "tif"); self.image_format.addItem("PNG", "png")
        self.fit_groups = QSpinBox(); self.fit_groups.setRange(1, 10000); self.fit_groups.setValue(18)
        self.start_index = QSpinBox(); self.start_index.setRange(0, 1_000_000); self.start_index.setValue(1)
        self.index_digits = QSpinBox(); self.index_digits.setRange(1, 8); self.index_digits.setValue(3)
        self.include_validation = QCheckBox("包含独立验证集"); self.include_validation.setChecked(True)
        self.validation_groups = QSpinBox(); self.validation_groups.setRange(1, 10000); self.validation_groups.setValue(6)
        self.resume = QCheckBox("续采对应的 .inprogress 数据集")
        # 保留语义化别名，便于项目内其它页面/测试读取而不依赖控件布局。
        self.output_dir = self.output
        self.plan_output_path = self.plan_path
        self.fit_group_count = self.fit_groups
        self.validation_group_count = self.validation_groups
        form.addRow("计划 YAML（生成/加载）", plan_row)
        form.addRow("输出数据集", _path_row(self.output, self, directory=True))
        form.addRow("数据集 ID", self.dataset_id); form.addRow("图像格式", self.image_format)
        form.addRow("拟合集组数", self.fit_groups); form.addRow("起始编号", self.start_index)
        form.addRow("编号位数", self.index_digits); form.addRow("验证集", self.include_validation)
        form.addRow("验证集组数", self.validation_groups); form.addRow("异常恢复", self.resume)
        layout.addLayout(form)

        self.recipe_table = CaptureRecipeTable()
        layout.addWidget(QLabel("每组图像配方")); layout.addWidget(self.recipe_table)
        plan_action_row = QHBoxLayout()
        self.generate_plan_button = QPushButton("生成并检查计划")
        self.restore_recipe_button = QPushButton("恢复三联图默认值")
        plan_action_row.addWidget(self.generate_plan_button); plan_action_row.addWidget(self.restore_recipe_button); plan_action_row.addStretch()
        layout.addLayout(plan_action_row)
        self.plan_summary = QLabel("尚未生成计划")
        self.plan_summary.setWordWrap(True)
        layout.addWidget(self.plan_summary)
        self.plan_table = QTableWidget(0, 9)
        self.plan_table.setHorizontalHeaderLabels([
            "split", "pose_id", "task_id", "role", "曝光 μs", "激光状态", "质量模式", "输出文件", "frames",
        ])
        self.plan_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.plan_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.plan_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plan_table.setMaximumHeight(220)
        self.plan_preview = self.plan_table
        layout.addWidget(self.plan_table)

        self.plan_tasks = QListWidget()
        self.plan_tasks.setMaximumHeight(150)
        self.plan_status = QLabel("请先生成并检查计划；也可加载旧版 capture-plan YAML。")
        self.plan_status.setWordWrap(True)
        plan_buttons = QHBoxLayout()
        self.preview_task_button = QPushButton("预览选中任务")
        self.capture_task_button = QPushButton("稳定后保存当前帧")
        self.next_task_button = QPushButton("下一任务")
        plan_buttons.addWidget(self.preview_task_button); plan_buttons.addWidget(self.capture_task_button); plan_buttons.addWidget(self.next_task_button)
        self.start_button = QPushButton("开始采集")
        self.resume_button = QPushButton("续采")
        self.cancel_button = QPushButton("取消")
        # 引导采集期间同一个入口负责“稳定后保存当前帧并切换下一任务”。
        # 旧的后台 run_capture_plan API 仍保留，但不再让实时画面承担 gate 等待。
        self.capture_task_button.setToolTip(
            "稳定后保存当前任务帧；保存成功后自动切换到下一任务。"
        )
        # 兼容旧测试/插件属性，但不再创建第二个可见按钮。
        self.ready_button = self.capture_task_button
        self.resume_button.setEnabled(False); self.cancel_button.setEnabled(False); self.capture_task_button.setEnabled(False)
        capture_buttons = QHBoxLayout()
        capture_buttons.addWidget(self.start_button); capture_buttons.addWidget(self.resume_button)
        capture_buttons.addWidget(self.cancel_button)
        self.progress = QTextEdit(); self.progress.setReadOnly(True)
        self.current_task = QLabel("当前任务：--")
        self.current_task.setWordWrap(True)

        live_panel = QGroupBox("任务实时画面")
        live_layout = QVBoxLayout(live_panel)
        self.live_preview = ImagePreview()
        self.live_auto_stretch = QCheckBox("自动拉伸预览（仅改变显示）")
        self.live_quality = QLabel("尚未取流")
        self.live_quality.setWordWrap(True)
        self.live_camera = QLabel("当前任务：--")
        self.live_camera.setWordWrap(True)
        live_layout.addWidget(self.live_preview, 1)
        live_layout.addWidget(self.live_auto_stretch)
        live_layout.addWidget(self.live_camera)
        live_layout.addWidget(self.live_quality)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.plan_tasks)
        left_layout.addWidget(self.plan_status)
        left_layout.addLayout(plan_buttons)
        left_layout.addWidget(self.current_task)
        left_layout.addLayout(capture_buttons)
        left_layout.addWidget(self.progress, 1)
        content = QSplitter(Qt.Orientation.Horizontal)
        content.addWidget(left)
        content.addWidget(live_panel)
        content.setStretchFactor(0, 1)
        content.setStretchFactor(1, 1)
        layout.addWidget(content, 1)

        # 可见按钮进入 GUI 引导采集；后台 run_capture_plan 入口保留在
        # start_capture()，供旧调用方和兼容测试使用。
        self.start_button.clicked.connect(lambda _checked=False: self.start_guided_capture())
        self.resume_button.clicked.connect(lambda: self.start_guided_capture(resume=True))
        self.cancel_button.clicked.connect(self.cancel_capture)
        self.capture_task_button.clicked.connect(self._capture_task_button_clicked)
        self.generate_plan_button.clicked.connect(self.generate_plan)
        self.restore_recipe_button.clicked.connect(self.recipe_table.reset_defaults)
        self.load_plan_button.clicked.connect(self.load_guided_plan)
        self.preview_task_button.clicked.connect(self.preview_selected_task)
        self.next_task_button.clicked.connect(self.select_next_task)
        self.live_auto_stretch.toggled.connect(
            lambda checked: self.live_preview.refresh_display(auto_stretch=checked)
        )
        self.camera_page.frame_ready.connect(self._on_camera_frame)
        for signal in (
            self.plan_path.textChanged,
            self.output.textChanged,
            self.dataset_id.textChanged,
            self.image_format.currentIndexChanged,
            self.fit_groups.valueChanged,
            self.start_index.valueChanged,
            self.index_digits.valueChanged,
            self.include_validation.toggled,
            self.validation_groups.valueChanged,
        ):
            signal.connect(self._mark_plan_dirty)
        self.recipe_table.changed.connect(self._mark_plan_dirty)

    def set_project(self, project: WizardProject) -> None:
        self.output.setText(str((project.capture_output or project.workspace / "data") / self.dataset_id.text()))
        stored_artifacts = project.extra.get("capture_artifacts") if isinstance(project.extra, dict) else None
        self.last_capture_artifacts = dict(stored_artifacts) if isinstance(stored_artifacts, dict) else None
        if project.workflow_plan and project.workflow_plan.name.startswith("capture_"):
            self.plan_path.setText(str(resolve_relative(project.source_path or project.workspace / "wizard_project.yaml", project.workflow_plan)))

        if not self.plan_path.text().strip():
            self.plan_path.setText(str(project.workspace / "plans" / f"{self.dataset_id.text().strip()}.yaml"))

    def _mark_plan_dirty(self, *_args: Any) -> None:
        if self._capture_worker is not None or self._guided_capture_active:
            return
        self._plan_dirty = True
        self.start_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.capture_task_button.setEnabled(False)
        self.plan_status.setText("配置已改变，需要重新生成并检查计划。")

    def _recipe_from_fields(self) -> CaptureRecipe:
        runtime = self.camera_page.runtime
        if runtime is None:
            raise ValueError("请先在第 2 页加载相机配置")
        image_format = str(self.image_format.currentData())
        return CaptureRecipe(
            dataset_id=self.dataset_id.text().strip(),
            output_dir=Path(self.output.text().strip()),
            plan_output_path=Path(self.plan_path.text().strip()),
            fit_group_count=self.fit_groups.value(),
            include_validation=self.include_validation.isChecked(),
            validation_group_count=self.validation_groups.value(),
            start_index=self.start_index.value(),
            index_digits=self.index_digits.value(),
            camera=self.camera_page.current_config(),
            serial_number=self.camera_page.selected_serial() or str(runtime.get("serial_number", "")),
            backend=str(runtime["backend"]),
            backend_options=dict(runtime.get("backend_options", {})),
            metadata={"created_by": "pyside6_wizard"},
            items=self.recipe_table.recipe_items(image_format),
            board_pattern=runtime.get("board_pattern") or (11, 8),
            quality_thresholds=runtime["quality_thresholds"],
        )

    def generate_plan(self) -> bool:
        try:
            recipe = self._recipe_from_fields()
            plan = build_capture_plan_from_recipe(recipe)
            # “生成并检查”是用户明确的重生成动作；底层 API 仍默认拒绝静默覆盖。
            saved = save_generated_capture_plan(plan, recipe.plan_output_path, overwrite=True)
            loaded = load_capture_plan(saved)
        except Exception as exc:
            self._plan_dirty = True
            self.start_button.setEnabled(False); self.resume_button.setEnabled(False)
            QMessageBox.critical(self, "采集计划无效", str(exc))
            return False
        self.loaded_plan = loaded
        self.loaded_plan_path = saved.resolve()
        self._generated_recipe = recipe
        self._show_plan(loaded)
        self.plan_path.setText(str(saved.resolve()))
        self._plan_dirty = False
        self._set_capture_buttons_ready()
        self.plan_status.setText(f"计划已保存并 round-trip 校验：{saved}")
        return True

    def _show_plan(self, plan: CapturePlan) -> None:
        summary = capture_plan_summary(plan)
        self.plan_summary.setText(
            f"拟合组 {summary['fit_group_count']} · 验证组 {summary['validation_group_count']} · "
            f"任务 {summary['task_count']} · 图像 {summary['image_count']}"
        )
        self.plan_table.setRowCount(len(summary["tasks"]))
        for row, record in enumerate(summary["tasks"]):
            values = (
                record["split"], record["pose_id"], record["task_id"], record["role"],
                f"{record['exposure_us']:g}", record["laser_state"], record["quality_mode"],
                record["relative_output_path"], str(record["frames"]),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.plan_table.setItem(row, column, item)
        self.plan_tasks.clear()
        for index, task in enumerate(plan.tasks, start=1):
            self.plan_tasks.addItem(
                f"{index:02d}. {task.task_id} · pose {task.pose_id} · {task.quality_mode} · "
                f"{task.config.exposure_us:g} μs · {task.instruction}"
            )
        if plan.tasks:
            self.plan_tasks.setCurrentRow(0)
            self.plan_table.selectRow(0)
        self.guided_preview_index = None

    def _set_capture_buttons_ready(self) -> None:
        ready = (
            self.loaded_plan is not None
            and not self._plan_dirty
            and self._capture_worker is None
            and not self._guided_capture_active
        )
        self.start_button.setEnabled(ready)
        self.resume_button.setEnabled(ready)
        self.capture_task_button.setEnabled(ready)

    def _set_guided_running(self, running: bool) -> None:
        """设置 GUI 引导采集状态；引导模式保持 PreviewThread 持续取流。"""

        self._guided_capture_active = running
        for widget in (
            self.plan_path, self.output, self.dataset_id, self.image_format,
            self.fit_groups, self.start_index, self.index_digits,
            self.include_validation, self.validation_groups, self.resume,
            self.recipe_table, self.load_plan_button,
        ):
            widget.setEnabled(not running)
        self.generate_plan_button.setEnabled(not running)
        self.restore_recipe_button.setEnabled(not running)
        self.preview_task_button.setEnabled(not running)
        self.next_task_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.resume_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.cancel_button.setEnabled(running)
        self.capture_task_button.setEnabled(
            running and not self._capture_in_progress and self.camera_page.preview_thread is not None
        )

    def _set_capture_running(self, running: bool) -> None:
        for widget in (
            self.plan_path, self.output, self.dataset_id, self.image_format,
            self.fit_groups, self.start_index, self.index_digits,
            self.include_validation, self.validation_groups, self.resume,
            self.recipe_table, self.load_plan_button,
        ):
            widget.setEnabled(not running)
        self.generate_plan_button.setEnabled(not running)
        self.restore_recipe_button.setEnabled(not running)
        self.preview_task_button.setEnabled(not running)
        self.next_task_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.resume_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.cancel_button.setEnabled(running)
        self.capture_task_button.setEnabled(False if running else self.loaded_plan is not None and not self._plan_dirty)

    def _guided_start_row(self, resume: bool) -> int:
        """返回引导采集要预览的首个任务，并校验续采计划 hash。"""

        if self.loaded_plan is None or not self.loaded_plan.tasks:
            raise ValueError("采集计划没有可执行任务")
        output_dir = self.loaded_plan.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise RuntimeError(f"输出数据集已经存在，不会覆盖：{output_dir}")
        work_dir = output_dir.parent / f".{output_dir.name}.inprogress"
        if not resume:
            if work_dir.exists():
                raise RuntimeError(f"发现未完成数据集；确认后勾选续采：{work_dir}")
            return 0

        manifest_path = work_dir / "dataset_manifest.yaml"
        if not manifest_path.is_file():
            raise RuntimeError(f"没有可续采的 manifest：{manifest_path}")
        manifest = load_document(manifest_path)
        if manifest.get("plan_sha256") != capture_plan_hash(self.loaded_plan):
            raise RuntimeError("当前采集计划与未完成数据集不一致，拒绝续采")
        states = manifest.get("tasks", {})
        for row, task in enumerate(self.loaded_plan.tasks):
            if states.get(task.task_id, {}).get("status") == "completed":
                self._mark_task_row(row)
                continue
            return row
        raise RuntimeError("未完成数据集中没有可继续的任务")

    def start_guided_capture(self, resume: bool | None = None) -> None:
        """启动 GUI 逐任务引导采集，保持实时预览并自动切换任务。"""

        if self._capture_worker is not None:
            QMessageBox.information(self, "采集正在运行", "当前正在执行后台采集计划，请先取消后再启动引导采集。")
            return
        if self.loaded_plan is None or self._plan_dirty:
            QMessageBox.warning(self, "计划未就绪", "请先点击“生成并检查计划”，并保持配置不变")
            return
        if self.camera_page.runtime is None:
            QMessageBox.warning(self, "相机未配置", "请先在相机页面加载配置")
            return
        resume_requested = self.resume.isChecked() if resume is None else bool(resume)
        try:
            row = self._guided_start_row(resume_requested)
        except Exception as exc:
            QMessageBox.critical(self, "无法开始引导采集", str(exc))
            return

        self.camera_page.stop_preview()
        self._capture_in_progress = False
        self._set_guided_running(True)
        self.progress.clear()
        self.progress.append("开始引导采集；实时画面稳定后点击“稳定后保存当前帧”。")
        self.plan_tasks.setCurrentRow(row)
        self.plan_table.selectRow(row)
        self.guided_preview_index = None
        self.plan_status.setText("正在切换到首个未完成任务的相机配置…")
        self.preview_selected_task()

    def start_capture(self, resume: bool | None = None) -> None:
        """兼容旧调用方的后台 run_capture_plan 采集入口。

        GUI 按钮使用 :meth:`start_guided_capture`，以便保存每帧后继续保持
        PreviewThread 取流；该方法保留给旧测试、插件和需要后台计划执行的调用方。
        """

        runtime = self.camera_page.runtime
        if runtime is None:
            QMessageBox.warning(self, "相机未配置", "请先在相机页面加载配置"); return
        if self.loaded_plan is None or self._plan_dirty:
            QMessageBox.warning(self, "计划未就绪", "请先点击“生成并检查计划”，并保持配置不变"); return
        self.camera_page.stop_preview()
        try:
            provider = build_camera_provider(
                self.loaded_plan.backend,
                calibration_src=runtime["calibration_src"],
                backend_options=self.loaded_plan.backend_options,
            )
        except Exception as exc:
            QMessageBox.critical(self, "相机后端无效", str(exc)); return
        resume_requested = self.resume.isChecked() if resume is None else bool(resume)
        # 先把页面切到本次采集的首个任务；工作线程随后会在同一 session
        # 中应用实际相机参数，并在续采时通过 task_requested 选择首个未完成任务。
        if not resume_requested and self.loaded_plan.tasks:
            self._select_batch_task(self.loaded_plan.tasks[0])
        gate = CaptureTaskGate(self)
        cancel_event = threading.Event()
        self._capture_gate = gate
        self._capture_cancel_event = cancel_event
        self._last_completed_pose = None
        self.progress.clear(); self.progress.append("开始采集；每个任务准备好后点击“稳定后保存当前帧”继续…")
        self._set_capture_running(True)
        worker = FunctionWorker(
            lambda report: run_capture_plan(
                self.loaded_plan,
                provider,
                resume=resume_requested,
                progress=report,
                before_task=lambda task: gate.wait_for_task(task, cancel_event),
                cancel_event=cancel_event,
            )
        )
        self._capture_worker = worker
        gate.task_requested.connect(self._on_task_requested)
        worker.signals.progress.connect(self._capture_progress)
        worker.signals.result.connect(self._capture_done)
        worker.signals.error.connect(self._capture_error)
        worker.signals.finished.connect(self._capture_finished)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.thread_pool.start(worker)

    def approve_current_task(self) -> None:
        if self._capture_gate is None:
            return
        self.capture_task_button.setEnabled(False)
        self.plan_status.setText("已确认，工作线程正在采集当前任务…")
        self._capture_gate.approve()

    def _capture_task_button_clicked(self) -> None:
        """统一处理引导保存和批量 gate 确认，避免两个按钮保存同一任务。"""

        if self._capture_worker is not None:
            self.approve_current_task()
        else:
            self.capture_current_task_frame()

    def cancel_capture(self) -> None:
        if self._guided_capture_active:
            self._capture_in_progress = False
            self.camera_page.stop_preview()
            self._set_guided_running(False)
            self.plan_status.setText("引导采集已取消；未完成数据保留在 .inprogress，可勾选续采。")
            return
        if self._capture_cancel_event is None:
            return
        self._capture_cancel_event.set()
        if self._capture_gate is not None:
            self._capture_gate.cancel()
        self.capture_task_button.setEnabled(False)
        self.plan_status.setText("正在取消采集并关闭相机会话…")

    def _on_task_requested(self, task: CaptureTask) -> None:
        if self._capture_worker is None or (
            self._capture_cancel_event is not None and self._capture_cancel_event.is_set()
        ):
            return
        self._select_batch_task(task)
        split = str(task.tags.get("split", ""))
        laser_state = task.tags.get("laser_state", "unchanged")
        output = task.relative_path(1).as_posix()
        self.current_task.setText(
            f"当前组：{split or '--'} · 当前任务：{task.task_id} · pose {task.pose_id} · role {task.role} · "
            f"曝光 {task.config.exposure_us:g} μs · 激光 {laser_state}\n"
            f"提示：{task.instruction or '请按现场要求准备'}\n输出：{output}"
        )
        move_hint = ""
        if self._last_completed_pose and self._last_completed_pose != task.pose_id:
            move_hint = f"上一姿态 {self._last_completed_pose} 已完成，请移动到 {task.pose_id}。\n"
        self.plan_status.setText(move_hint + "请完成当前姿态准备后点击“稳定后保存当前帧”。")
        self.capture_task_button.setEnabled(True)

    def _select_batch_task(self, task: CaptureTask) -> None:
        """批量 worker 进入新 task 时，同步任务选择和相机设置显示。

        批量采集期间不能重新打开预览线程（相机会被 worker 独占），但仍应让
        页面立即显示下一任务的参数；真正的相机 configure 由 run_capture_plan
        在同一 session 中完成。
        """

        if self.loaded_plan is None:
            return
        try:
            row = [item.task_id for item in self.loaded_plan.tasks].index(task.task_id)
        except ValueError:
            return
        self.plan_tasks.setCurrentRow(row)
        self.plan_table.selectRow(row)
        config = task.config
        for widget, value in (
            (self.camera_page.exposure, config.exposure_us),
            (self.camera_page.gain, config.gain_db),
            (self.camera_page.offset_x, config.offset_x),
            (self.camera_page.offset_y, config.offset_y),
            (self.camera_page.width, config.width),
            (self.camera_page.height, config.height),
        ):
            with QSignalBlocker(widget):
                widget.setValue(value)
        with QSignalBlocker(self.camera_page.pixel_format):
            self.camera_page.pixel_format.setCurrentText(config.pixel_format)
        with QSignalBlocker(self.camera_page.quality_mode):
            self.camera_page.quality_mode.setCurrentIndex(
                self.camera_page.quality_mode.findData(task.quality_mode)
            )
        self.guided_preview_index = row
        self.live_camera.setText(
            f"当前任务：{task.task_id} · 曝光 {config.exposure_us:g} μs · "
            f"增益 {config.gain_db:g} dB · 稳定帧 {task.settle_frames}"
        )
        self.live_quality.setText("已切换到当前任务参数，等待采集帧…")

    def _capture_finished(self) -> None:
        self._capture_worker = None
        self._capture_gate = None
        self._capture_cancel_event = None
        self._set_capture_running(False)
        self._set_capture_buttons_ready()

    def load_guided_plan(self) -> None:
        path = Path(self.plan_path.text()).expanduser()
        try:
            plan = load_capture_plan(path)
        except Exception as exc:
            QMessageBox.critical(self, "采集计划无效", str(exc)); return
        self.camera_page.stop_preview()
        self.loaded_plan = plan
        self.loaded_plan_path = path.resolve()
        self.output.setText(str(plan.output_dir))
        self.dataset_id.setText(plan.dataset_id)
        self._generated_recipe = None
        self._plan_dirty = False
        self._show_plan(plan)
        self._set_capture_buttons_ready()
        self.plan_status.setText(
            f"已加载 {plan.dataset_id}：{len(plan.tasks)} 个任务，backend={plan.backend}，输出到 {plan.output_dir}"
        )

    def _selected_plan_task(self, *, silent: bool = False) -> tuple[int, CaptureTask] | None:
        if self.loaded_plan is None:
            if not silent:
                QMessageBox.information(self, "未加载计划", "请先加载 capture-plan YAML")
            return None
        row = self.plan_tasks.currentRow()
        if row < 0 or row >= len(self.loaded_plan.tasks):
            if not silent:
                QMessageBox.information(self, "未选择任务", "请先在任务列表中选择一个 task")
            return None
        return row, self.loaded_plan.tasks[row]

    def _on_camera_frame(self, frame, quality: dict[str, Any]) -> None:
        self.live_preview.set_array(
            frame.image,
            auto_stretch=self.live_auto_stretch.isChecked(),
            sensor_max_value=(
                self.camera_page.preview_thread.config.sensor_max_value
                if self.camera_page.preview_thread is not None
                else None
            ),
        )
        warnings = "、".join(_quality_warning_text(item) for item in quality["warnings"]) or "通过"
        thresholds = (
            self.loaded_plan.quality_thresholds
            if self.loaded_plan is not None
            else (self.camera_page.runtime or {}).get("quality_thresholds")
        )
        coverage_text = _laser_quality_metrics_text(quality, thresholds)
        chess_hint = quality.get("chessboard_hint")
        chess_text = f" · {chess_hint}" if chess_hint else ""
        settling, settle_remaining = _preview_settling_info(quality)
        settling_text = f" · 正在稳定，剩余 {settle_remaining} 帧" if settling else ""
        self.live_quality.setText(
            f"{warnings} · 动态范围 {quality['dynamic_range_u8']:.1f} DN8 · "
            f"清晰度 {quality['focus_laplacian']:.1f}{coverage_text}{chess_text}{settling_text}"
        )
        task_text = "当前任务：--"
        selected = self._selected_plan_task(silent=True)
        if selected is not None:
            _, task = selected
            applied = (
                self.camera_page.preview_thread.config
                if self.camera_page.preview_thread is not None
                else task.config
            )
            task_text = (
                f"当前任务：{task.task_id} · 曝光 {applied.exposure_us:g} μs · "
                f"增益 {applied.gain_db:g} dB · 稳定帧 {task.settle_frames}"
        )
        self.live_camera.setText(task_text)
        if self._guided_capture_active and not self._capture_in_progress:
            if settling:
                self.capture_task_button.setEnabled(False)
            elif self._preview_matches_selected_task():
                self.capture_task_button.setEnabled(True)

    def _preview_matches_selected_task(self) -> bool:
        selected = self._selected_plan_task(silent=True)
        thread = self.camera_page.preview_thread
        if selected is None or thread is None or not thread.isRunning():
            return False
        _, task = selected
        config = thread.config
        return (
            thread.quality_mode == task.quality_mode
            and config.pixel_format == task.config.pixel_format
            and config.offset_x == task.config.offset_x
            and config.offset_y == task.config.offset_y
            and config.width == task.config.width
            and config.height == task.config.height
            and abs(config.exposure_us - task.config.exposure_us) <= max(1.0, task.config.exposure_us * 1e-3)
            and abs(config.gain_db - task.config.gain_db) <= 1e-3
        )

    def preview_selected_task(self, on_started: Callable[[], None] | None = None) -> None:
        selected = self._selected_plan_task()
        if selected is None:
            return
        row, task = selected
        assert self.loaded_plan is not None
        runtime = dict(self.camera_page.runtime or {})
        if "calibration_src" not in runtime:
            QMessageBox.warning(self, "相机配置不足", "请先在第 2 页加载相机配置，以确定 calibration_src"); return
        runtime.update(
            source=self.loaded_plan_path,
            backend=self.loaded_plan.backend,
            serial_number=self.loaded_plan.serial_number,
            camera=task.config,
            quality_thresholds=self.loaded_plan.quality_thresholds,
            board_pattern=self.loaded_plan.board_pattern,
            backend_options=self.loaded_plan.backend_options,
        )
        self._preview_request_token += 1
        token = self._preview_request_token
        self._pending_preview_callback = on_started
        preview_running = (
            self.camera_page.preview_thread is not None
            and self.camera_page.preview_thread.isRunning()
        )
        self.camera_page.runtime = runtime
        self.camera_page.backend_label.setText(self.loaded_plan.backend)
        serial = self.loaded_plan.serial_number
        if self.loaded_plan.backend == "synthetic":
            serial = serial or "SIM-001"
            runtime["serial_number"] = serial
        if not preview_running:
            self.camera_page.devices.clear()
            if self.loaded_plan.backend == "synthetic":
                self.camera_page.devices.addItem(f"模拟相机 · SN {serial}", serial)
            elif serial:
                self.camera_page.devices.addItem(f"计划指定相机 · SN {serial}", serial)
        self.camera_page.exposure.setValue(task.config.exposure_us)
        self.camera_page.gain.setValue(task.config.gain_db)
        self.camera_page.pixel_format.setCurrentText(task.config.pixel_format)
        self.camera_page.offset_x.setValue(task.config.offset_x)
        self.camera_page.offset_y.setValue(task.config.offset_y)
        self.camera_page.width.setValue(task.config.width)
        self.camera_page.height.setValue(task.config.height)
        self.camera_page.quality_mode.setCurrentIndex(
            self.camera_page.quality_mode.findData(task.quality_mode)
        )
        split = str(task.tags.get("split", ""))
        laser_state = task.tags.get("laser_state", "unchanged")
        self.current_task.setText(
            f"当前组：{split or '--'} · 当前任务：{task.task_id} · pose {task.pose_id} · role {task.role} · "
            f"曝光 {task.config.exposure_us:g} μs · 激光 {laser_state}\n"
            f"提示：{task.instruction or '请按现场要求准备'}\n输出：{task.relative_path(1).as_posix()}"
        )
        self.camera_page.last_quality = None
        self.live_camera.setText(
            f"当前任务：{task.task_id} · 曝光 {task.config.exposure_us:g} μs · "
            f"增益 {task.config.gain_db:g} dB · 正在应用任务配置…"
        )
        self.live_quality.setText("实时画面保持连接，正在等待新配置稳定帧…")
        self.guided_preview_index = row
        self.plan_status.setText(f"正在切换到 {task.task_id} 预览：{task.instruction}")
        if self._guided_capture_active:
            self.capture_task_button.setEnabled(False)

        if self.camera_page.request_preview_task(
            task.config, task.quality_mode, task.settle_frames
        ):
            callback = self._pending_preview_callback
            self._pending_preview_callback = None
            if callback is not None:
                QTimer.singleShot(0, callback)
            return

        QTimer.singleShot(
            350,
            lambda: self._start_guided_preview(
                token, row, task.task_id, task.instruction, task.settle_frames
            ),
        )

    def _start_guided_preview(
        self,
        token: int,
        row: int,
        task_id: str,
        instruction: str,
        settle_frames: int,
    ) -> None:
        if token != self._preview_request_token:
            return
        if self.loaded_plan is None or row < 0 or row >= len(self.loaded_plan.tasks):
            return
        self.camera_page.start_preview(initial_discard_frames=settle_frames)
        self.guided_preview_index = row
        self.plan_status.setText(f"正在预览 {task_id}：{instruction}")
        if self._guided_capture_active:
            self.capture_task_button.setEnabled(False)
        callback = self._pending_preview_callback
        self._pending_preview_callback = None
        if callback is not None:
            QTimer.singleShot(0, callback)

    def capture_current_task_frame(self) -> None:
        if self._capture_in_progress:
            self.plan_status.setText("正在等待稳定帧，请稍候…")
            return
        selected = self._selected_plan_task()
        if selected is None:
            return
        row, task = selected
        self._capture_in_progress = True
        self.capture_task_button.setEnabled(False)
        self.preview_task_button.setEnabled(False)
        self.next_task_button.setEnabled(False)

        def save_after_settle(frame, _quality) -> None:
            self._capture_in_progress = False
            self.capture_task_button.setEnabled(not self._guided_capture_active)
            self.preview_task_button.setEnabled(not self._guided_capture_active)
            self.next_task_button.setEnabled(not self._guided_capture_active)
            try:
                completed = self._save_guided_frame(task, frame)
            except Exception as exc:
                QMessageBox.critical(self, "保存任务失败", str(exc))
                return
            if completed:
                self._mark_task_row(row)
                if self._guided_capture_active:
                    self.capture_task_button.setEnabled(False)
                self.select_next_task()
            else:
                self.capture_task_button.setEnabled(True)
                self.plan_status.setText(
                    f"{task.task_id} 尚未采满 {task.frames} 帧；继续点击保存完成该任务。"
                )

        if self.guided_preview_index != row or self.camera_page.preview_thread is None:
            self.preview_selected_task(
                on_started=lambda: self._schedule_guided_capture(task, save_after_settle)
            )
            return
        self._schedule_guided_capture(task, save_after_settle)

    def _schedule_guided_capture(
        self,
        task: CaptureTask,
        callback: Callable[[Any, dict[str, Any]], None],
    ) -> None:
        if self.camera_page.capture_after_settle(task.settle_frames, callback):
            self.plan_status.setText(
                f"{task.task_id}：已应用曝光，丢弃 {task.settle_frames} 帧后保存下一帧…"
            )
            return
        self._capture_in_progress = False
        self.capture_task_button.setEnabled(True)
        self.preview_task_button.setEnabled(not self._guided_capture_active)
        self.next_task_button.setEnabled(not self._guided_capture_active)
        QMessageBox.warning(self, "尚未取流", "任务预览尚未建立，请等待实时画面出现后重试。")

    def _save_guided_frame(self, task: CaptureTask, frame: Any) -> bool:
        assert self.loaded_plan is not None
        output_dir = self.loaded_plan.output_dir.expanduser().resolve()
        work_dir = output_dir.parent / f".{output_dir.name}.inprogress"
        manifest_path = work_dir / "dataset_manifest.yaml"
        if output_dir.exists():
            raise RuntimeError(f"输出数据集已经存在，不会覆盖：{output_dir}")
        if manifest_path.is_file():
            manifest = load_document(manifest_path)
            if manifest.get("plan_sha256") != capture_plan_hash(self.loaded_plan):
                raise RuntimeError("当前采集计划与未完成数据集不一致，拒绝续采")
        elif work_dir.exists() and not self.resume.isChecked():
            raise RuntimeError(f"发现未完成数据集；确认后勾选续采：{work_dir}")
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            manifest = _new_manifest(self.loaded_plan)
        task_state = manifest["tasks"][task.task_id]
        captured = int(task_state.get("frames_captured") or 0)
        if captured >= task.frames:
            raise RuntimeError(f"{task.task_id} 已采满 {task.frames} 帧")
        index = captured + 1
        relative = task.relative_path(index)
        destination = work_dir / relative
        _write_image(destination, frame)
        quality = quality_to_dict(analyze_frame(
            frame.image,
            sensor_max_value=task.config.sensor_max_value,
            mode=task.quality_mode,
            thresholds=self.loaded_plan.quality_thresholds,
            board_pattern=self.loaded_plan.board_pattern,
        ))
        record = {
            "task_id": task.task_id,
            "pose_id": task.pose_id,
            "role": task.role,
            "tags": task.tags,
            "index": index,
            "filename": relative.as_posix(),
            "sha256": sha256_file(destination, normalize_newlines=False),
            "camera_frame_number": frame.camera_frame_number,
            "camera_frame_gap": None,
            "transport_warnings": [],
            "camera_timestamp_ticks": frame.camera_timestamp_ticks,
            "host_timestamp_ns": frame.host_timestamp_ns,
            "host_monotonic_ns": frame.host_monotonic_ns,
            "requested_camera": asdict(task.config),
            "applied_camera": asdict(self.camera_page.preview_thread.config if self.camera_page.preview_thread else task.config),
            "quality": quality,
        }
        manifest["frames"] = [
            item for item in manifest["frames"]
            if not (item["task_id"] == task.task_id and int(item["index"]) == index)
        ]
        manifest["frames"].append(record)
        task_state.update(
            status="completed" if index >= task.frames else "capturing",
            frames_captured=index,
            completed_at=_utc_now_text() if index >= task.frames else None,
        )
        task_completed = index >= task.frames
        if all(item.get("status") == "completed" for item in manifest["tasks"].values()):
            manifest["status"] = "completed"
            manifest["completed_at"] = _utc_now_text()
            _save_state(work_dir, manifest)
            work_dir.replace(output_dir)
            self.camera_page.stop_preview()
            self._set_guided_running(False)
            self.progress.append(f"完成：{output_dir}")
            self.last_capture_artifacts = capture_artifacts_record(self.loaded_plan_path, output_dir)
            self.capture_finished.emit(
                type(
                    "Result",
                    (),
                    {"output_dir": output_dir, "capture_artifacts": self.last_capture_artifacts},
                )()
            )
            return task_completed
        else:
            _save_state(work_dir, manifest)
            self.progress.append(
                f"{task.task_id}  {index}/{task.frames}  {','.join(quality['warnings']) or '通过'}  → {relative.as_posix()}"
            )
            return task_completed

    def _mark_task_row(self, row: int) -> None:
        item = self.plan_tasks.item(row)
        if item and not item.text().startswith("✓ "):
            item.setText("✓ " + item.text())

    def select_next_task(self, *, auto_preview: bool = True) -> None:
        if self.loaded_plan is None:
            return
        row = self.plan_tasks.currentRow()
        for candidate in range(row + 1, len(self.loaded_plan.tasks)):
            item = self.plan_tasks.item(candidate)
            if item and not item.text().startswith("✓ "):
                self.plan_tasks.setCurrentRow(candidate)
                if auto_preview:
                    self.preview_selected_task()
                else:
                    self.plan_status.setText(f"下一任务：{self.loaded_plan.tasks[candidate].instruction}")
                return
        self.plan_status.setText("没有后续未完成任务。")

    def _capture_progress(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name == "task_started":
            self.progress.append(
                f"开始：{event['task_id']} · pose {event.get('pose_id', '')} · "
                f"曝光 {float(event.get('exposure_us', 0)):g} μs · 激光 {event.get('laser_state', 'unchanged')}"
            )
        elif event_name == "frame":
            warnings = ",".join(event["quality"]["warnings"]) or "通过"
            relative = event.get("relative_output_path", "")
            self.progress.append(
                f"{event['task_id']}  {event['index']}/{event['frames']}  {warnings}  → {relative}"
            )
        elif event_name == "quality_warning":
            self.progress.append(
                f"质量告警：{event['task_id']} 帧 {event['index']}：{', '.join(event['warnings'])}"
            )
        elif event_name == "task_completed":
            self.progress.append(f"✓ {event['task_id']} 已提交")
            self._last_completed_pose = str(event.get("pose_id", ""))
            self._display_saved_image(event["task_id"], event.get("relative_output_path", ""))
            if self.loaded_plan is not None:
                task_ids = [task.task_id for task in self.loaded_plan.tasks]
                try:
                    index = task_ids.index(event["task_id"])
                except ValueError:
                    index = -1
                if index >= 0 and index + 1 < len(self.loaded_plan.tasks):
                    next_task = self.loaded_plan.tasks[index + 1]
                    if next_task.pose_id != event.get("pose_id"):
                        self.plan_status.setText(
                            f"姿态 {event.get('pose_id', '')} 的图像已完成，请移动到下一姿态。"
                        )

    def _display_saved_image(self, task_id: str, relative: str) -> None:
        if self.loaded_plan is None or not relative:
            return
        output_dir = self.loaded_plan.output_dir.expanduser().resolve()
        path = output_dir / relative
        if not path.is_file():
            return
        for path in (path,):
            if not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                self.live_preview.set_array(
                    image,
                    auto_stretch=self.live_auto_stretch.isChecked(),
                    sensor_max_value=self.loaded_plan.base_config.sensor_max_value,
                )
            except Exception:
                pass
            return

    def _capture_done(self, result) -> None:
        output_dir = Path(result.output_dir)
        self.last_capture_artifacts = capture_artifacts_record(self.loaded_plan_path, output_dir)
        self.progress.append(f"完成：{result.frame_count} 帧 → {output_dir}")
        self.progress.append(f"manifest：{output_dir / 'dataset_manifest.yaml'}")
        self.progress.append(f"frames：{output_dir / 'frames.csv'}")
        self.capture_finished.emit(result)

    def _capture_error(self, message: str) -> None:
        self.progress.append(f"失败：{message}")
        if "采集已取消" not in message:
            QMessageBox.critical(self, "采集失败", message)


class CalibrationPage(QWidget):
    workflow_finished = Signal(object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool; self._workers: set[FunctionWorker] = set()
        self.capture_artifacts: dict[str, Any] | None = None
        layout = QVBoxLayout(self)
        title = QLabel("4. 一键执行标定 workflow"); title.setObjectName("pageTitle"); layout.addWidget(title)
        self.workflow = QLineEdit(); layout.addWidget(_labeled_path("Workflow YAML", self.workflow, self, "YAML (*.yaml *.yml)"))
        self.refresh_button = QPushButton("检查 Workflow 阶段")
        self.update_workflow_button = QPushButton("从最近采集结果更新 Workflow 输入")
        self.update_workflow_button.setEnabled(False)
        action_row = QHBoxLayout()
        action_row.addWidget(self.refresh_button); action_row.addWidget(self.update_workflow_button); action_row.addStretch()
        layout.addLayout(action_row)
        self.stage_table = QTableWidget(0, 4)
        self.stage_table.setHorizontalHeaderLabels(["启用", "阶段", "输入/配置", "输出"])
        self.stage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.stage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.stage_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.stage_table.setMaximumHeight(220)
        layout.addWidget(self.stage_table)
        self.allow_note = QLabel("向导按 workflow 中启用的阶段顺序执行；每个阶段仍使用阶段 1 的质量门禁。")
        self.allow_note.setWordWrap(True); layout.addWidget(self.allow_note)
        self.capture_artifact_status = QLabel("尚未记录最近采集结果")
        self.capture_artifact_status.setWordWrap(True); layout.addWidget(self.capture_artifact_status)
        self.run_button = QPushButton("一键运行完整标定"); layout.addWidget(self.run_button)
        self.log = QTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log, 1)
        self.run_button.clicked.connect(self.run); self.refresh_button.clicked.connect(lambda: self.refresh_plan())
        self.update_workflow_button.clicked.connect(self.update_workflow_inputs)
        self.workflow.textChanged.connect(
            lambda _text: self.update_workflow_button.setEnabled(
                bool(self.capture_artifacts and self.workflow.text().strip())
            )
        )

    def set_project(self, project: WizardProject) -> None:
        self.workflow.setText(str(project.workflow_plan or ""))
        stored = project.extra.get("capture_artifacts") if isinstance(project.extra, dict) else None
        self.set_capture_artifacts(stored if isinstance(stored, dict) else None)
        self.refresh_plan(silent=True)

    def set_capture_artifacts(self, artifacts: Mapping[str, Any] | None) -> None:
        self.capture_artifacts = dict(artifacts) if artifacts else None
        enabled = bool(self.capture_artifacts and self.workflow.text().strip())
        self.update_workflow_button.setEnabled(enabled)
        if self.capture_artifacts:
            self.capture_artifact_status.setText(
                "最近采集："
                f"\n数据集：{self.capture_artifacts.get('dataset_root', '暂无')}"
                f"\nfit：{self.capture_artifacts.get('fit_dir', '暂无')}"
                f"\nvalidation：{self.capture_artifacts.get('validation_dir', '暂无')}"
            )
        else:
            self.capture_artifact_status.setText("尚未记录最近采集结果")

    def update_workflow_inputs(self) -> None:
        if not self.capture_artifacts:
            QMessageBox.information(self, "没有采集结果", "请先完成一次批量采集。")
            return
        path = Path(self.workflow.text()).expanduser()
        if not path.is_file():
            QMessageBox.warning(self, "Workflow 不存在", str(path))
            return
        try:
            preview = build_workflow_update_preview(path, self.capture_artifacts)
        except Exception as exc:
            QMessageBox.critical(self, "无法生成更新预览", str(exc))
            return
        if not preview.changed:
            QMessageBox.information(self, "无需更新", preview.text())
            return
        answer = QMessageBox.question(
            self,
            "确认更新 Workflow 输入",
            preview.text() + "\n\n确认后会先备份原文件，再保存更新后的 YAML。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            backup = save_workflow_update(preview, backup=True)
            self.refresh_plan(silent=True)
            self.capture_artifact_status.setText(
                self.capture_artifact_status.text()
                + f"\nWorkflow 已更新；备份：{backup or '未生成'}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Workflow 保存失败", str(exc))

    def refresh_plan(self, *, silent: bool = False) -> bool:
        path = Path(self.workflow.text()).expanduser()
        if not path.is_file():
            self.stage_table.setRowCount(0)
            if not silent:
                QMessageBox.warning(self, "Workflow 不存在", str(path))
            return False
        try:
            document = load_document(path)
            stages = document.get("stages", [])
            if not isinstance(stages, list):
                raise ValueError("stages 必须是列表")
            self.stage_table.setRowCount(len(stages))
            for row, stage in enumerate(stages):
                options = stage.get("options", {}) if isinstance(stage, dict) else {}
                enabled = "是" if isinstance(stage, dict) and stage.get("enabled", True) else "否"
                name = str(stage.get("name", "")) if isinstance(stage, dict) else ""
                inputs = [f"{key}={value}" for key, value in options.items() if key not in {"output", "output_dir"}]
                output = options.get("output", options.get("output_dir", ""))
                for column, value in enumerate((enabled, name, "; ".join(inputs), str(output))):
                    item = QTableWidgetItem(value); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.stage_table.setItem(row, column, item)
            return True
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Workflow 无效", str(exc))
            return False

    def run(self) -> None:
        path = Path(self.workflow.text()).expanduser()
        if not self.refresh_plan():
            return
        self.run_button.setEnabled(False); self.log.clear(); self.log.append(f"开始：{path.resolve()}")
        worker = FunctionWorker(lambda progress: run_workflow(path, progress=progress))
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._done)
        worker.signals.error.connect(self._error)
        worker.signals.finished.connect(lambda: self.run_button.setEnabled(True))
        self._workers.add(worker); worker.signals.finished.connect(lambda: self._workers.discard(worker)); self.thread_pool.start(worker)

    def _done(self, result: dict[str, Any]) -> None:
        for stage in result.get("stages", []):
            self.log.append(f"{stage.get('stage')}: {stage.get('status')}")
        self.log.append(f"Workflow: {result.get('status')}")
        if self.capture_artifacts:
            result = dict(result)
            result["capture_artifacts"] = dict(self.capture_artifacts)
        self.workflow_finished.emit(result)

    def _progress(self, event: dict[str, Any]) -> None:
        if event.get("event") == "stage_started":
            self.log.append(f"▶ {event['stage']} 开始")
        elif event.get("event") == "stage_finished":
            self.log.append(f"✓ {event['stage']}：{event['status']}")

    def _error(self, message: str) -> None:
        self.log.append(f"失败：{message}"); QMessageBox.critical(self, "标定失败", message)


class ResultsPage(QWidget):
    acceptance_finished = Signal(object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self._workers: set[FunctionWorker] = set()
        self.current_result: dict[str, Any] | None = None
        self.project: WizardProject | None = None
        self.last_html: Path | None = None
        self.capture_artifacts: dict[str, Any] | None = None
        self._artifact_records: list[ResultArtifact] = []
        self._filtered_artifacts: list[ResultArtifact] = []
        self._current_artifact_index = -1
        layout = QVBoxLayout(self)
        title_row = QHBoxLayout()
        title = QLabel("5. 报告、补偿对比与验收闭环"); title.setObjectName("pageTitle")
        self.load_button = QPushButton("打开报告…")
        title_row.addWidget(title); title_row.addStretch(); title_row.addWidget(self.load_button); layout.addLayout(title_row)
        acceptance_box = QGroupBox("生成正式验收报告")
        acceptance_layout = QGridLayout(acceptance_box)
        self.acceptance_plan = QLineEdit()
        self.acceptance_overwrite = QCheckBox("覆盖已有报告")
        self.acceptance_button = QPushButton("生成并判定")
        self.open_html_button = QPushButton("打开 HTML 报告")
        self.open_html_button.setEnabled(False)
        self.acceptance_status = QLabel("尚未执行验收")
        acceptance_layout.addWidget(QLabel("验收计划"), 0, 0)
        acceptance_layout.addWidget(_path_row(self.acceptance_plan, self, file_filter="YAML (*.yaml *.yml)"), 0, 1, 1, 3)
        acceptance_layout.addWidget(self.acceptance_overwrite, 1, 1)
        acceptance_layout.addWidget(self.acceptance_button, 1, 2)
        acceptance_layout.addWidget(self.open_html_button, 1, 3)
        acceptance_layout.addWidget(self.acceptance_status, 2, 0, 1, 4)
        layout.addWidget(acceptance_box)
        browser_box = QGroupBox("逐图结果浏览")
        browser_layout = QVBoxLayout(browser_box)
        filter_row = QHBoxLayout()
        self.stage_filter = QComboBox(); self.split_filter = QComboBox(); self.pose_filter = QComboBox(); self.status_filter = QComboBox()
        for combo, label in (
            (self.stage_filter, "stage"),
            (self.split_filter, "split"),
            (self.pose_filter, "pose_id"),
            (self.status_filter, "状态"),
        ):
            filter_row.addWidget(QLabel(label)); filter_row.addWidget(combo)
        self.previous_artifact_button = QPushButton("上一张")
        self.next_artifact_button = QPushButton("下一张")
        self.open_artifact_button = QPushButton("在文件夹中打开")
        filter_row.addWidget(self.previous_artifact_button); filter_row.addWidget(self.next_artifact_button); filter_row.addWidget(self.open_artifact_button)
        browser_layout.addLayout(filter_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("报告指标与质量门禁"))
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["项目", "值"]); self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setMaximumHeight(180)
        left_layout.addWidget(self.tree)
        left_layout.addWidget(QLabel("stage / split / pose_id"))
        self.artifact_tree = QTreeWidget(); self.artifact_tree.setHeaderLabels(["结果产物", "状态"])
        self.artifact_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        left_layout.addWidget(self.artifact_tree, 1)
        # 保留旧 QListWidget 属性，兼容旧插件/测试；新界面使用 artifact_tree。
        self.artifacts = QListWidget(); self.artifacts.setVisible(False)

        center = QWidget(); center_layout = QVBoxLayout(center)
        self.image_preview = ImagePreview(); center_layout.addWidget(self.image_preview, 1)
        self.image_status = QLabel("请选择结果图")
        self.image_status.setWordWrap(True); center_layout.addWidget(self.image_status)

        right = QWidget(); right_layout = QVBoxLayout(right)
        self.artifact_info = QLabel("暂无文件信息")
        self.artifact_info.setWordWrap(True); right_layout.addWidget(self.artifact_info)
        self.plot = ResidualPlot(); right_layout.addWidget(self.plot)
        self.table = QTableWidget(); right_layout.addWidget(self.table, 1)
        splitter.addWidget(left); splitter.addWidget(center); splitter.addWidget(right)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 3); splitter.setStretchFactor(2, 2)
        browser_layout.addWidget(splitter, 1)
        layout.addWidget(browser_box, 1)
        self.load_button.clicked.connect(self._open)
        self.acceptance_button.clicked.connect(self.generate_acceptance)
        self.open_html_button.clicked.connect(self.open_html)
        self.artifacts.itemSelectionChanged.connect(self._artifact_selected)
        self.artifact_tree.itemClicked.connect(self._artifact_tree_selected)
        for combo in (self.stage_filter, self.split_filter, self.pose_filter, self.status_filter):
            combo.currentIndexChanged.connect(self._apply_artifact_filters)
        self.previous_artifact_button.clicked.connect(self._previous_artifact)
        self.next_artifact_button.clicked.connect(self._next_artifact)
        self.open_artifact_button.clicked.connect(self._open_artifact_folder)

    def set_project(self, project: WizardProject) -> None:
        self.project = project
        existing = project.acceptance_plan
        plan_path = default_acceptance_plan_path(project.workspace, existing)
        if existing is None:
            project.acceptance_plan = plan_path
        self.acceptance_status.setText(f"当前项目验收计划：{plan_path}")
        self.acceptance_plan.setText(str(plan_path or ""))
        stored = project.extra.get("capture_artifacts") if isinstance(project.extra, dict) else None
        self.set_capture_artifacts(stored if isinstance(stored, dict) else None)

    def prepare_default_plan(self, *, announce: bool = True) -> Path | None:
        """进入第 5 页时创建默认计划；已有显式计划保持不动。"""

        if self.project is None:
            return None
        try:
            default_path = default_acceptance_plan_path(self.project.workspace)
            field_value = self.acceptance_plan.text().strip()
            existing = Path(field_value).expanduser().resolve() if field_value else self.project.acceptance_plan
            # set_project 会先把默认路径显示到控件；该路径仍属于自动计划，
            # 不应被当成“用户显式计划”而跳过首次创建。
            explicit = existing if existing is not None and existing.resolve() != default_path.resolve() else None
            path = ensure_default_acceptance_plan(
                self.project.workspace,
                self.project.project_id,
                existing_path=explicit,
            )
        except Exception as exc:
            self.acceptance_status.setText(f"默认验收计划准备失败：{exc}")
            return None
        self.project.acceptance_plan = path
        self.acceptance_plan.setText(str(path))
        if announce:
            self.acceptance_status.setText(f"当前项目验收计划：{path}")
        return path

    def update_acceptance_from_workflow(self, result: Mapping[str, Any]) -> None:
        """workflow 完成后自动填充当前项目验收计划的空输入项。"""

        self.prepare_default_plan(announce=False)
        plan_value = self.acceptance_plan.text().strip()
        workflow_value = result.get("workflow")
        if not plan_value or not workflow_value:
            return
        try:
            update = update_acceptance_plan_from_workflow(
                plan_value,
                str(workflow_value),
                result,
            )
        except Exception as exc:
            self.acceptance_status.setText(f"Workflow 已完成，但验收计划路径更新失败：{exc}")
            return
        if update.changed:
            self.acceptance_status.setText(
                "已从最近 workflow 自动更新验收输入："
                f"\nworkflow：{update.workflow_report or '暂无'}"
                f"\n补偿指标：{update.compensation_metrics or '暂无'}"
                "\n请核对后点击“生成并判定”。"
            )
        else:
            self.acceptance_status.setText("Workflow 已完成，验收计划未发现需要更新的空路径。")

    def set_capture_artifacts(self, artifacts: Mapping[str, Any] | None) -> None:
        self.capture_artifacts = dict(artifacts) if artifacts else None
        if self.current_result is not None:
            self._set_artifact_records(
                discover_artifact_records(self.current_result, self.capture_artifacts)
            )

    def generate_acceptance(self) -> None:
        path = Path(self.acceptance_plan.text()).expanduser()
        if not path.is_file():
            QMessageBox.warning(self, "验收计划不存在", str(path)); return
        overwrite = self.acceptance_overwrite.isChecked()
        self.acceptance_button.setEnabled(False)
        self.acceptance_status.setText("正在汇总 workflow、补偿指标、golden 和产物哈希…")
        worker = FunctionWorker(lambda _progress: build_acceptance_report(path, overwrite=overwrite))
        worker.signals.result.connect(self._acceptance_done)
        worker.signals.error.connect(self._acceptance_error)
        worker.signals.finished.connect(lambda: self.acceptance_button.setEnabled(True))
        self._workers.add(worker); worker.signals.finished.connect(lambda: self._workers.discard(worker)); self.thread_pool.start(worker)

    def _acceptance_done(self, report: dict[str, Any]) -> None:
        self.show_result(report)
        files = report.get("report_files", {})
        html_value = files.get("html")
        self.last_html = Path(html_value) if html_value else None
        self.open_html_button.setEnabled(bool(self.last_html and self.last_html.is_file()))
        self.acceptance_status.setText(
            f"验收结论：{report.get('decision')} · PASS {report['counts']['pass']} · "
            f"WARN {report['counts']['warn']} · FAIL {report['counts']['fail']} · "
            f"发布：{report.get('release', {}).get('status')}"
        )
        self.acceptance_finished.emit(report)

    def _acceptance_error(self, message: str) -> None:
        self.acceptance_status.setText(f"验收失败：{message}")
        QMessageBox.critical(self, "无法生成验收报告", message)

    def open_html(self) -> None:
        if self.last_html and self.last_html.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_html.resolve())))

    def show_result(self, result: dict[str, Any]) -> None:
        self.current_result = dict(result)
        self.tree.clear()
        _add_tree(self.tree.invisibleRootItem(), self.current_result)
        self.tree.expandToDepth(2)
        capture = self.current_result.get("capture_artifacts")
        if not isinstance(capture, Mapping):
            capture = self.capture_artifacts
        self._set_artifact_records(
            discover_artifact_records(self.current_result, capture if isinstance(capture, Mapping) else None)
        )

    def _set_artifact_records(self, records: list[ResultArtifact]) -> None:
        self._artifact_records = list(records)
        self._filtered_artifacts = []
        self._current_artifact_index = -1
        self.artifacts.clear()
        for record in self._artifact_records:
            self.artifacts.addItem(str(record.display_path))
        self._set_filter_values(self.stage_filter, [record.stage for record in records])
        self._set_filter_values(self.split_filter, [record.split for record in records if record.split])
        self._set_filter_values(self.pose_filter, [record.pose_id for record in records if record.pose_id])
        status_values = [record.status or "暂无关联指标" for record in records]
        self._set_filter_values(self.status_filter, status_values)
        self._apply_artifact_filters()

    @staticmethod
    def _set_filter_values(combo: QComboBox, values: list[str | None]) -> None:
        normalized = sorted({str(value) for value in values if value not in (None, "")})
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem("全部")
            combo.addItems(normalized)
        combo.setCurrentIndex(0)

    def _apply_artifact_filters(self) -> None:
        stage = self.stage_filter.currentText()
        split = self.split_filter.currentText()
        pose = self.pose_filter.currentText()
        status = self.status_filter.currentText()
        self._filtered_artifacts = [
            record
            for record in self._artifact_records
            if (stage in ("", "全部") or record.stage == stage)
            and (split in ("", "全部") or record.split == split)
            and (pose in ("", "全部") or record.pose_id == pose)
            and (
                status in ("", "全部")
                or (record.status or "暂无关联指标") == status
            )
        ]
        self.artifact_tree.clear()
        stage_items: dict[str, QTreeWidgetItem] = {}
        split_items: dict[tuple[str, str], QTreeWidgetItem] = {}
        pose_items: dict[tuple[str, str, str], QTreeWidgetItem] = {}
        index_lookup = {id(record): index for index, record in enumerate(self._artifact_records)}
        for record in self._filtered_artifacts:
            stage_item = stage_items.get(record.stage)
            if stage_item is None:
                stage_item = QTreeWidgetItem([record.stage, ""])
                self.artifact_tree.addTopLevelItem(stage_item)
                stage_items[record.stage] = stage_item
            split_name = (
                "summary_plots"
                if record.pose_id is None and record.artifact_type.startswith("validation_error")
                else record.split or "summary_plots"
            )
            split_key = (record.stage, split_name)
            split_item = split_items.get(split_key)
            if split_item is None:
                split_item = QTreeWidgetItem([split_name, ""])
                stage_item.addChild(split_item)
                split_items[split_key] = split_item
            pose_name = record.pose_id or "汇总"
            pose_key = (record.stage, split_name, pose_name)
            pose_item = pose_items.get(pose_key)
            if pose_item is None:
                pose_item = QTreeWidgetItem([pose_name, ""])
                split_item.addChild(pose_item)
                pose_items[pose_key] = pose_item
            child = QTreeWidgetItem([f"{record.artifact_type} · {record.display_path.name}", record.status or "暂无关联指标"])
            child.setData(0, Qt.ItemDataRole.UserRole, index_lookup[id(record)])
            pose_item.addChild(child)
        if not self._filtered_artifacts:
            self.artifact_tree.addTopLevelItem(QTreeWidgetItem(["没有发现结果图或诊断产物", ""]))
            self.image_preview.clear_image("没有可显示的结果图")
            self.image_status.setText("当前筛选没有结果产物")
            self.artifact_info.setText("暂无文件信息")
        else:
            self.artifact_tree.expandAll()
        self.previous_artifact_button.setEnabled(bool(self._filtered_artifacts))
        self.next_artifact_button.setEnabled(bool(self._filtered_artifacts))
        self.open_artifact_button.setEnabled(bool(self._filtered_artifacts))

    def _artifact_tree_selected(self, item: QTreeWidgetItem, _column: int) -> None:
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(value, int):
            self._display_artifact(value)

    def _display_artifact(self, index: int) -> None:
        if index < 0 or index >= len(self._artifact_records):
            return
        record = self._artifact_records[index]
        self._current_artifact_index = index
        metrics = "暂无关联指标"
        if record.metrics:
            metrics = "\n".join(f"{key}: {value}" for key, value in record.metrics.items())
        self.artifact_info.setText(
            f"stage：{record.stage}\n"
            f"split：{record.split or 'summary'}\n"
            f"pose_id：{record.pose_id or '汇总'}\n"
            f"类型：{record.artifact_type}\n"
            f"状态：{record.status or '暂无关联指标'}\n"
            f"文件：{record.display_path}\n\n{metrics}"
        )
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.plot.set_values([])
        path = record.display_path
        if record.image_path is not None:
            if not path.is_file():
                self.image_preview.clear_image("图像缺失")
                self.image_status.setText(f"图像缺失：{path}")
                return
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.image_preview.clear_image("图像损坏或无法读取")
                self.image_status.setText(f"图像损坏或无法读取：{path}")
                return
            try:
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.image_preview.set_array(image, auto_stretch=True)
                self.image_status.setText(f"已显示：{path}")
            except Exception as exc:
                self.image_preview.clear_image("图像无法显示")
                self.image_status.setText(f"图像无法显示：{exc}")
            return
        self.image_preview.clear_image("当前产物不是图像")
        if path.suffix.lower() == ".csv":
            try:
                headers, rows, values = load_residual_csv(path)
            except Exception as exc:
                self.image_status.setText(f"诊断文件读取失败：{exc}")
                return
            self.table.setColumnCount(len(headers)); self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, value in enumerate(row):
                    self.table.setItem(row_index, column, QTableWidgetItem(value))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.plot.set_values(values)
            self.image_status.setText(f"诊断 CSV：{path}")
        else:
            self.image_status.setText(f"诊断文件：{path}")

    def _previous_artifact(self) -> None:
        if not self._filtered_artifacts:
            return
        current = self._filtered_artifacts.index(self._artifact_records[self._current_artifact_index]) if self._current_artifact_index >= 0 and self._artifact_records[self._current_artifact_index] in self._filtered_artifacts else 0
        record = self._filtered_artifacts[(current - 1) % len(self._filtered_artifacts)]
        self._display_artifact(self._artifact_records.index(record))

    def _next_artifact(self) -> None:
        if not self._filtered_artifacts:
            return
        current = self._filtered_artifacts.index(self._artifact_records[self._current_artifact_index]) if self._current_artifact_index >= 0 and self._artifact_records[self._current_artifact_index] in self._filtered_artifacts else -1
        record = self._filtered_artifacts[(current + 1) % len(self._filtered_artifacts)]
        self._display_artifact(self._artifact_records.index(record))

    def _open_artifact_folder(self) -> None:
        if self._current_artifact_index < 0 or self._current_artifact_index >= len(self._artifact_records):
            return
        path = self._artifact_records[self._current_artifact_index].display_path
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent.resolve())))

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开标定报告", "", "YAML/JSON (*.yaml *.yml *.json)")
        if not path:
            return
        try:
            self.show_result(load_document(path))
            html_path = Path(path).with_suffix(".html")
            self.last_html = html_path if html_path.is_file() else None
            self.open_html_button.setEnabled(self.last_html is not None)
        except Exception as exc:
            QMessageBox.critical(self, "报告读取失败", str(exc))

    def _artifact_selected(self) -> None:
        items = self.artifacts.selectedItems()
        if not items:
            return
        path = Path(items[0].text())
        for index, record in enumerate(self._artifact_records):
            if record.display_path.resolve() == path.resolve():
                self._display_artifact(index)
                return


def discover_result_artifacts(result: dict[str, Any]) -> list[Path]:
    """兼容旧页面/插件的 Path 列表 API。"""

    records = discover_artifact_records(result)
    return sorted({record.display_path.resolve() for record in records}, key=str)


def load_residual_csv(path: Path, limit: int = 500) -> tuple[list[str], list[list[str]], list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        dictionaries = [row for _, row in zip(range(limit), reader)]
    rows = [[str(row.get(header, "")) for header in headers] for row in dictionaries]
    preferred = [header for header in headers if any(key in header.lower() for key in ("residual", "error", "distance", "rmse"))]
    for header in preferred + headers:
        values: list[float] = []
        try:
            values = [float(row[header]) for row in dictionaries if row.get(header, "") not in ("", None)]
        except ValueError:
            continue
        if len(values) >= 2:
            return headers, rows, values
    return headers, rows, []


def _add_tree(parent: QTreeWidgetItem, value: Any, name: str = "root") -> None:
    if isinstance(value, dict):
        item = QTreeWidgetItem([name, ""]); parent.addChild(item)
        for key, child in value.items():
            _add_tree(item, child, str(key))
    elif isinstance(value, list):
        item = QTreeWidgetItem([name, f"{len(value)} 项"]); parent.addChild(item)
        for index, child in enumerate(value):
            _add_tree(item, child, str(index + 1))
    else:
        parent.addChild(QTreeWidgetItem([name, "" if value is None else str(value)]))


def _path_row(line_edit: QLineEdit, parent: QWidget, *, directory: bool = False, file_filter: str = "All files (*)") -> QWidget:
    widget = QWidget(parent); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("浏览…", widget); layout.addWidget(line_edit, 1); layout.addWidget(button)
    def browse() -> None:
        if directory:
            selected = QFileDialog.getExistingDirectory(parent, "选择目录", line_edit.text())
        else:
            selected, _ = QFileDialog.getOpenFileName(parent, "选择文件", line_edit.text(), file_filter)
        if selected:
            line_edit.setText(selected)
    button.clicked.connect(browse)
    return widget


def _labeled_path(label: str, line_edit: QLineEdit, parent: QWidget, file_filter: str) -> QWidget:
    widget = QWidget(parent); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(label)); layout.addWidget(_path_row(line_edit, parent, file_filter=file_filter), 1)
    return widget


def _quality_warning_text(code: str) -> str:
    return {
        "saturation_high": "过曝像素过多",
        "chessboard_saturation_high": "棋盘亮区过曝",
        "chessboard_near_saturation": "棋盘白格接近饱和",
        "laser_saturation_high": "激光饱和像素过多",
        "laser_peak_saturated": "激光峰值已饱和",
        "laser_peak_near_saturation": "激光峰值接近饱和",
        "laser_saturated_line_wide": "激光饱和线宽过大",
        "image_too_dark": "图像整体过暗",
        "dynamic_range_low": "动态范围不足",
        "laser_coverage_low": "激光横向覆盖不足",
        "chessboard_not_found": "未找到完整棋盘",
        "chessboard_pattern_mismatch": "棋盘内角点配置不匹配",
        "chessboard_obscured_by_laser": "激光线遮挡棋盘",
    }.get(code, code)


def _laser_quality_metrics_text(quality: dict[str, Any], thresholds: Any | None) -> str:
    coverage = quality.get("laser_coverage")
    if coverage is None:
        return ""
    parts = [f"覆盖 {coverage:.1%}"]
    saturated = quality.get("laser_peak_saturation_fraction")
    near = quality.get("laser_peak_near_saturation_fraction")
    saturated_width = quality.get("laser_saturated_width_p95_px")
    fwhm_p50 = quality.get("laser_fwhm_p50_px")
    fwhm_p95 = quality.get("laser_fwhm_p95_px")
    if saturated is not None:
        limit = getattr(thresholds, "max_laser_peak_saturation_fraction", None)
        suffix = f"≤{limit:.1%}" if isinstance(limit, (int, float)) else ""
        parts.append(f"峰饱 {saturated:.1%}{suffix}")
    if near is not None:
        limit = getattr(thresholds, "max_laser_peak_near_saturation_fraction", None)
        suffix = f"≤{limit:.1%}" if isinstance(limit, (int, float)) else ""
        parts.append(f"近饱 {near:.1%}{suffix}")
    if saturated_width is not None:
        limit = getattr(thresholds, "max_laser_saturated_width_px", None)
        suffix = f"≤{limit:.1f}px" if isinstance(limit, (int, float)) else ""
        parts.append(f"饱和宽P95 {saturated_width:.1f}px{suffix}")
    if fwhm_p50 is not None and fwhm_p95 is not None:
        parts.append(f"FWHM P50/P95 {fwhm_p50:.1f}/{fwhm_p95:.1f}px")
    return " · 激光" + "，".join(parts)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
