from __future__ import annotations

import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from ..acceptance import build_acceptance_report
from ..camera.models import CapturePlan, CaptureTask
from ..camera.quality import analyze_frame, quality_to_dict
from ..io_utils import load_document, resolve_relative
from ..io_utils import sha256_file
from ..workflow import run_workflow
from .project import WizardProject
from .widgets import ImagePreview, ResidualPlot
from .workers import FunctionWorker, PreviewThread


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
        return WizardProject(
            project_id=self.project_id.text().strip(),
            workspace=workspace,
            camera_config=Path(self.camera_config.text()),
            workflow_plan=Path(workflow) if workflow else None,
            acceptance_plan=Path(self.acceptance_plan.text().strip()) if self.acceptance_plan.text().strip() else None,
            capture_output=workspace / "data",
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
        self.quality.setText(
            f"{warnings} · 动态范围 {quality['dynamic_range_u8']:.1f} DN8 · "
            f"清晰度 {quality['focus_laplacian']:.1f}{coverage_text}{chess_text}"
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
        self.guided_preview_index: int | None = None
        self._preview_request_token = 0
        self._pending_preview_callback: Callable[[], None] | None = None
        self._capture_in_progress = False
        layout = QVBoxLayout(self)
        title = QLabel("3. 批量采集标定图像"); title.setObjectName("pageTitle"); layout.addWidget(title)
        form = QFormLayout()
        self.plan_path = QLineEdit()
        load_plan_button = QPushButton("加载计划")
        plan_row = QHBoxLayout(); plan_row.addWidget(_path_row(self.plan_path, self, file_filter="YAML (*.yaml *.yml)"), 1); plan_row.addWidget(load_plan_button)
        self.output = QLineEdit(); self.dataset_id = QLineEdit("calibration_dataset"); self.pose_id = QLineEdit("pose_01")
        self.role = QComboBox(); self.role.addItem("相机内参棋盘", "intrinsics"); self.role.addItem("激光平面", "laser_plane"); self.role.addItem("地面外参", "ground")
        self.exposures = QLineEdit("1200"); self.exposures.setPlaceholderText("例如 400, 800, 1200")
        self.frames = QSpinBox(); self.frames.setRange(1, 1000); self.frames.setValue(1)
        self.settle = QSpinBox(); self.settle.setRange(0, 100); self.settle.setValue(5)
        self.quality_mode = QComboBox()
        self.quality_mode.addItem("棋盘格（内参时关闭激光）", "chessboard")
        self.quality_mode.addItem("激光线（激光开启）", "laser")
        self.quality_mode.addItem("通用曝光", "generic")
        self.resume = QCheckBox("续采对应的 .inprogress 数据集")
        form.addRow("采集计划 YAML", plan_row)
        form.addRow("输出数据集", _path_row(self.output, self, directory=True))
        form.addRow("数据集 ID", self.dataset_id); form.addRow("姿态 ID", self.pose_id); form.addRow("采集用途", self.role)
        form.addRow("曝光序列 (μs)", self.exposures); form.addRow("每个曝光帧数", self.frames)
        form.addRow("参数切换后丢帧", self.settle); form.addRow("质量模式", self.quality_mode)
        form.addRow("异常恢复", self.resume)
        layout.addLayout(form)

        self.plan_tasks = QListWidget()
        self.plan_tasks.setMaximumHeight(150)
        self.plan_status = QLabel("未加载 capture-plan；上方单姿态采集仍可使用。")
        self.plan_status.setWordWrap(True)
        plan_buttons = QHBoxLayout()
        self.preview_task_button = QPushButton("预览选中任务")
        self.capture_task_button = QPushButton("稳定后保存任务帧")
        self.next_task_button = QPushButton("下一任务")
        plan_buttons.addWidget(self.preview_task_button); plan_buttons.addWidget(self.capture_task_button); plan_buttons.addWidget(self.next_task_button)
        self.start_button = QPushButton("开始批量采集")
        self.progress = QTextEdit(); self.progress.setReadOnly(True)

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
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.progress, 1)
        content = QSplitter(Qt.Orientation.Horizontal)
        content.addWidget(left)
        content.addWidget(live_panel)
        content.setStretchFactor(0, 1)
        content.setStretchFactor(1, 1)
        layout.addWidget(content, 1)

        self.start_button.clicked.connect(self.start_capture); self.role.currentIndexChanged.connect(self._role_changed)
        load_plan_button.clicked.connect(self.load_guided_plan)
        self.preview_task_button.clicked.connect(self.preview_selected_task)
        self.capture_task_button.clicked.connect(self.capture_current_task_frame)
        self.next_task_button.clicked.connect(self.select_next_task)
        self.live_auto_stretch.toggled.connect(
            lambda checked: self.live_preview.refresh_display(auto_stretch=checked)
        )
        self.camera_page.frame_ready.connect(self._on_camera_frame)

    def set_project(self, project: WizardProject) -> None:
        self.output.setText(str((project.capture_output or project.workspace / "data") / self.dataset_id.text()))
        if project.workflow_plan and project.workflow_plan.name.startswith("capture_"):
            self.plan_path.setText(str(resolve_relative(project.source_path or project.workspace / "wizard_project.yaml", project.workflow_plan)))

    def _role_changed(self) -> None:
        mode = "chessboard" if self.role.currentData() == "intrinsics" else "laser"
        self.quality_mode.setCurrentIndex(self.quality_mode.findData(mode))

    def start_capture(self) -> None:
        runtime = self.camera_page.runtime
        if runtime is None:
            QMessageBox.warning(self, "相机未配置", "请先在相机页面加载配置"); return
        self.camera_page.stop_preview()
        try:
            exposures = [float(value.strip()) for value in self.exposures.text().split(",") if value.strip()]
            if not exposures or min(exposures) <= 0 or len(set(exposures)) != len(exposures):
                raise ValueError("曝光序列必须是非重复正数")
            base = self.camera_page.current_config()
            image_format = "tif" if base.pixel_format == "Mono12" else "png"
            tasks = tuple(CaptureTask(
                task_id=f"{self.role.currentData()}_{index:02d}", frames=self.frames.value(),
                filename_template="{role}/{exposure_folder}/{pose_id}{index_suffix}{suffix}",
                config=replace(base, exposure_us=exposure), pose_id=self.pose_id.text().strip(),
                role=str(self.role.currentData()), settle_frames=self.settle.value(), image_format=image_format,
                quality_mode=str(self.quality_mode.currentData()), tags={"requested_exposure_us": exposure},
            ) for index, exposure in enumerate(exposures, 1))
            plan = CapturePlan(
                dataset_id=self.dataset_id.text().strip(), output_dir=Path(self.output.text()), backend=runtime["backend"],
                serial_number=self.camera_page.selected_serial(), base_config=base, tasks=tasks,
                quality_thresholds=runtime["quality_thresholds"], board_pattern=runtime["board_pattern"],
                metadata={"created_by": "pyside6_wizard_mvp"}, backend_options=runtime["backend_options"],
            )
            provider = build_camera_provider(
                plan.backend, calibration_src=runtime["calibration_src"], backend_options=plan.backend_options
            )
        except Exception as exc:
            QMessageBox.critical(self, "采集参数无效", str(exc)); return
        self.start_button.setEnabled(False); self.progress.clear(); self.progress.append("开始采集…")
        resume = self.resume.isChecked()
        worker = FunctionWorker(lambda report: run_capture_plan(
            plan, provider, resume=resume, progress=report
        ))
        worker.signals.progress.connect(self._capture_progress)
        worker.signals.result.connect(self._capture_done)
        worker.signals.error.connect(lambda message: self._capture_error(message))
        worker.signals.finished.connect(lambda: self.start_button.setEnabled(True))
        self._workers.add(worker); worker.signals.finished.connect(lambda: self._workers.discard(worker)); self.thread_pool.start(worker)

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
        self.plan_tasks.clear()
        for index, task in enumerate(plan.tasks, start=1):
            self.plan_tasks.addItem(
                f"{index:02d}. {task.task_id} · pose {task.pose_id} · {task.quality_mode} · {task.config.exposure_us:g} μs · {task.instruction}"
            )
        self.plan_tasks.setCurrentRow(0)
        self.guided_preview_index = None
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
        self.live_quality.setText(
            f"{warnings} · 动态范围 {quality['dynamic_range_u8']:.1f} DN8 · "
            f"清晰度 {quality['focus_laplacian']:.1f}{coverage_text}{chess_text}"
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
        self.camera_page.stop_preview()
        self.camera_page.runtime = runtime
        self.camera_page.backend_label.setText(self.loaded_plan.backend)
        self.camera_page.devices.clear()
        serial = self.loaded_plan.serial_number
        if self.loaded_plan.backend == "synthetic":
            serial = serial or "SIM-001"
            self.camera_page.devices.addItem(f"模拟相机 · SN {serial}", serial)
            runtime["serial_number"] = serial
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
        self.camera_page.last_frame = None
        self.camera_page.last_quality = None
        self.guided_preview_index = row
        self.plan_status.setText(f"正在切换到 {task.task_id} 预览：{task.instruction}")
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
            self.capture_task_button.setEnabled(True)
            self.preview_task_button.setEnabled(True)
            self.next_task_button.setEnabled(True)
            try:
                completed = self._save_guided_frame(task, frame)
            except Exception as exc:
                QMessageBox.critical(self, "保存任务失败", str(exc))
                return
            if completed:
                self._mark_task_row(row)
                self.select_next_task()
            else:
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
        self.preview_task_button.setEnabled(True)
        self.next_task_button.setEnabled(True)
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
            self.progress.append(f"完成：{output_dir}")
            self.capture_finished.emit(type("Result", (), {"output_dir": output_dir})())
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
        if event.get("event") == "frame":
            warnings = ",".join(event["quality"]["warnings"]) or "通过"
            self.progress.append(f"{event['task_id']}  {event['index']}/{event['frames']}  {warnings}")
        elif event.get("event") == "task_completed":
            self.progress.append(f"✓ {event['task_id']} 已提交")

    def _capture_done(self, result) -> None:
        self.progress.append(f"完成：{result.frame_count} 帧 → {result.output_dir}")
        self.capture_finished.emit(result)

    def _capture_error(self, message: str) -> None:
        self.progress.append(f"失败：{message}"); QMessageBox.critical(self, "采集失败", message)


class CalibrationPage(QWidget):
    workflow_finished = Signal(object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool; self._workers: set[FunctionWorker] = set()
        layout = QVBoxLayout(self)
        title = QLabel("4. 一键执行标定 workflow"); title.setObjectName("pageTitle"); layout.addWidget(title)
        self.workflow = QLineEdit(); layout.addWidget(_labeled_path("Workflow YAML", self.workflow, self, "YAML (*.yaml *.yml)"))
        self.refresh_button = QPushButton("检查 Workflow 阶段")
        layout.addWidget(self.refresh_button)
        self.stage_table = QTableWidget(0, 4)
        self.stage_table.setHorizontalHeaderLabels(["启用", "阶段", "输入/配置", "输出"])
        self.stage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.stage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.stage_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.stage_table.setMaximumHeight(220)
        layout.addWidget(self.stage_table)
        self.allow_note = QLabel("向导按 workflow 中启用的阶段顺序执行；每个阶段仍使用阶段 1 的质量门禁。")
        self.allow_note.setWordWrap(True); layout.addWidget(self.allow_note)
        self.run_button = QPushButton("一键运行完整标定"); layout.addWidget(self.run_button)
        self.log = QTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log, 1)
        self.run_button.clicked.connect(self.run); self.refresh_button.clicked.connect(lambda: self.refresh_plan())

    def set_project(self, project: WizardProject) -> None:
        self.workflow.setText(str(project.workflow_plan or ""))
        self.refresh_plan(silent=True)

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
        self.last_html: Path | None = None
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("指标与质量门禁"))
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["项目", "值"]); self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        left_layout.addWidget(self.tree, 2)
        left_layout.addWidget(QLabel("残差/诊断文件")); self.artifacts = QListWidget(); left_layout.addWidget(self.artifacts, 1)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.plot = ResidualPlot(); right_layout.addWidget(self.plot)
        self.table = QTableWidget(); right_layout.addWidget(self.table, 1)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setStretchFactor(1, 1); layout.addWidget(splitter, 1)
        self.load_button.clicked.connect(self._open)
        self.acceptance_button.clicked.connect(self.generate_acceptance)
        self.open_html_button.clicked.connect(self.open_html)
        self.artifacts.itemSelectionChanged.connect(self._artifact_selected)

    def set_project(self, project: WizardProject) -> None:
        self.acceptance_plan.setText(str(project.acceptance_plan or ""))

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
        self.current_result = result; self.tree.clear(); _add_tree(self.tree.invisibleRootItem(), result); self.tree.expandToDepth(2)
        self.artifacts.clear()
        for path in discover_result_artifacts(result):
            self.artifacts.addItem(str(path))

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
        if path.suffix.lower() == ".csv":
            try:
                headers, rows, values = load_residual_csv(path)
            except Exception as exc:
                QMessageBox.critical(self, "残差文件读取失败", str(exc)); return
            self.table.setColumnCount(len(headers)); self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, value in enumerate(row):
                    self.table.setItem(row_index, column, QTableWidgetItem(value))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.plot.set_values(values)


def discover_result_artifacts(result: dict[str, Any]) -> list[Path]:
    stages = result.get("stages", [result])
    roots: set[Path] = set()
    for stage in stages if isinstance(stages, list) else []:
        if isinstance(stage, dict) and stage.get("output_dir"):
            root = Path(stage["output_dir"])
            if root.is_dir():
                roots.add(root)
    keywords = ("residual", "reprojection", "error", "diagnostic", "metric")
    artifacts: set[Path] = set()
    for item in result.get("artifacts", []):
        if isinstance(item, dict) and item.get("path"):
            path = Path(str(item["path"]))
            if path.is_file():
                artifacts.add(path.resolve())
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".yaml", ".yml", ".json", ".png"}:
                if any(keyword in path.name.lower() for keyword in keywords):
                    artifacts.add(path.resolve())
    return sorted(artifacts, key=str)


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
