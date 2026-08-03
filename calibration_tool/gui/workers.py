from __future__ import annotations

import threading
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
        self.initial_discard_frames = max(0, int(initial_discard_frames))
        self._parameter_lock = threading.Lock()
        self._pending_exposure_gain: tuple[float, float] | None = None

    def request_exposure_gain(self, exposure_us: float, gain_db: float) -> None:
        with self._parameter_lock:
            self._pending_exposure_gain = (float(exposure_us), float(gain_db))

    def _take_pending_exposure_gain(self) -> tuple[float, float] | None:
        with self._parameter_lock:
            pending = self._pending_exposure_gain
            self._pending_exposure_gain = None
            return pending

    def run(self) -> None:
        session = None
        try:
            session = self.provider.open(self.serial_number, self.config)
            self.opened.emit(session.device, session.config)
            session.start()
            for _ in range(self.initial_discard_frames):
                session.get_frame(session.config.timeout_ms)
            while not self.isInterruptionRequested():
                pending = self._take_pending_exposure_gain()
                if pending is not None:
                    try:
                        self.config = session.update_exposure_gain(*pending)
                    except Exception as exc:
                        self.parameter_update_failed.emit(str(exc))
                    else:
                        self.settings_applied.emit(self.config)
                        # 丢弃旧曝光仍可能滞留在传输队列中的帧。
                        for _ in range(2):
                            session.get_frame(session.config.timeout_ms)
                frame = session.get_frame(session.config.timeout_ms)
                quality = analyze_frame(
                    frame.image,
                    sensor_max_value=session.config.sensor_max_value,
                    mode=self.quality_mode,
                    thresholds=self.thresholds,
                    board_pattern=self.board_pattern,
                )
                self.frame_ready.emit(frame, quality_to_dict(quality))
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
