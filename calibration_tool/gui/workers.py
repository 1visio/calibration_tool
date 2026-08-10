from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThread, Signal, Slot

from ..camera.models import CameraConfig, CameraProvider, QualityThresholds
from ..camera.quality import analyze_frame, quality_to_dict


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class FunctionWorker(QRunnable):
    """在线程池中运行 `fn(progress_callback)`。"""

    def __init__(self, fn: Callable[[Callable[[object], None]], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self.fn(self.signals.progress.emit)
        except Exception as exc:  # Qt 边界统一转换为可展示消息
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class PreviewThread(QThread):
    frame_ready = Signal(object, object)
    opened = Signal(object, object)
    failed = Signal(str)
    settings_applied = Signal(object)
    parameter_update_failed = Signal(str)

    def __init__(
        self,
        provider: CameraProvider,
        serial_number: str,
        config: CameraConfig,
        quality_mode: str,
        thresholds: QualityThresholds,
        board_pattern: tuple[int, int] | None,
        laser_orientation: str = "horizontal",
        initial_discard_frames: int = 3,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.provider = provider
        self.serial_number = serial_number
        self.config = config
        self.quality_mode = quality_mode
        self.thresholds = thresholds
        self.board_pattern = board_pattern
        self.laser_orientation = laser_orientation
        self.initial_discard_frames = max(0, int(initial_discard_frames))
        self._parameter_lock = threading.Lock()
        self._pending_exposure_gain: tuple[float, float] | None = None
        self._pending_task: tuple[CameraConfig, str, int] | None = None

    def request_exposure_gain(self, exposure_us: float, gain_db: float) -> None:
        with self._parameter_lock:
            self._pending_exposure_gain = (float(exposure_us), float(gain_db))

    def request_task_config(
        self,
        config: CameraConfig,
        quality_mode: str,
        settle_frames: int = 0,
    ) -> bool:
        """在线请求下一个任务配置，不中断普通曝光/增益切换的取流。

        默认三联图只改变曝光和质量模式，工作线程会在同一个 session 上
        更新参数并继续发帧；ROI/像素格式变化则由线程内部执行必要的停流重配。
        """

        if not self.isRunning():
            return False
        with self._parameter_lock:
            self._pending_task = (config, str(quality_mode), max(0, int(settle_frames)))
        return True

    def _take_pending_exposure_gain(self) -> tuple[float, float] | None:
        with self._parameter_lock:
            pending = self._pending_exposure_gain
            self._pending_exposure_gain = None
            return pending

    def _take_pending_task(self) -> tuple[CameraConfig, str, int] | None:
        with self._parameter_lock:
            pending = self._pending_task
            self._pending_task = None
            return pending

    @staticmethod
    def _requires_restart(current: CameraConfig, target: CameraConfig) -> bool:
        return any(
            getattr(current, name) != getattr(target, name)
            for name in ("pixel_format", "offset_x", "offset_y", "width", "height")
        )

    def _emit_frame(
        self,
        session: Any,
        *,
        settling: bool = False,
        settle_frames_remaining: int = 0,
    ) -> None:
        """读取并发送一帧预览。

        稳定帧仍然发送给 GUI，避免任务切换时画面停在上一帧；额外的
        ``settling`` 元数据只用于 GUI 禁用保存按钮，不改变质量分析结果。
        """

        frame = session.get_frame(session.config.timeout_ms)
        quality = analyze_frame(
            frame.image,
            sensor_max_value=session.config.sensor_max_value,
            mode=self.quality_mode,
            thresholds=self.thresholds,
            board_pattern=self.board_pattern,
            laser_orientation=self.laser_orientation,
        )
        payload = quality_to_dict(quality)
        payload["settling"] = bool(settling)
        payload["settle_frames_remaining"] = max(0, int(settle_frames_remaining))
        self.frame_ready.emit(frame, payload)

    def run(self) -> None:
        session = None
        try:
            session = self.provider.open(self.serial_number, self.config)
            self.opened.emit(session.device, session.config)
            session.start()
            for remaining in range(self.initial_discard_frames, 0, -1):
                # 保持实时画面；最后一帧稳定帧到达后即可允许保存。
                self._emit_frame(
                    session,
                    settling=remaining > 1,
                    settle_frames_remaining=remaining - 1,
                )
            while not self.isInterruptionRequested():
                pending_task = self._take_pending_task()
                if pending_task is not None:
                    target, quality_mode, settle_frames = pending_task
                    try:
                        if self._requires_restart(self.config, target):
                            # 仅结构性参数需要停流；曝光/增益和质量模式保持连续取流。
                            session.stop()
                            applied = session.configure(target)
                            session.start()
                        else:
                            applied = session.config
                            if (
                                applied.exposure_us != target.exposure_us
                                or applied.gain_db != target.gain_db
                            ):
                                applied = session.update_exposure_gain(
                                    target.exposure_us, target.gain_db
                                )
                            applied = replace(
                                applied,
                                pixel_format=target.pixel_format,
                                offset_x=target.offset_x,
                                offset_y=target.offset_y,
                                width=target.width,
                                height=target.height,
                                timeout_ms=target.timeout_ms,
                            )
                        self.config = applied
                        self.quality_mode = quality_mode
                        self.settings_applied.emit(self.config)
                        for remaining in range(settle_frames, 0, -1):
                            self._emit_frame(
                                session,
                                settling=remaining > 1,
                                settle_frames_remaining=remaining - 1,
                            )
                    except Exception as exc:
                        self.parameter_update_failed.emit(str(exc))

                pending = self._take_pending_exposure_gain()
                if pending is not None:
                    try:
                        self.config = session.update_exposure_gain(*pending)
                    except Exception as exc:
                        self.parameter_update_failed.emit(str(exc))
                    else:
                        self.settings_applied.emit(self.config)
                        # 旧曝光仍可能滞留在传输队列中，但这些帧也继续显示。
                        for remaining in range(2, 0, -1):
                            self._emit_frame(
                                session,
                                settling=remaining > 1,
                                settle_frames_remaining=remaining - 1,
                            )
                self._emit_frame(session)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))
        finally:
            if session is not None:
                try:
                    session.stop()
                finally:
                    session.close()

    def stop(self, timeout_ms: int = 5000) -> bool:
        self.requestInterruption()
        return self.wait(timeout_ms)
