from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


def to_display_u8(
    image: np.ndarray,
    *,
    auto_stretch: bool = False,
    sensor_max_value: float | None = None,
) -> np.ndarray:
    """转换显示图；默认固定量程，自动拉伸仅用于观察暗部细节。"""
    if not auto_stretch:
        if image.dtype == np.uint8:
            return np.ascontiguousarray(image)
        maximum = float(sensor_max_value or np.iinfo(image.dtype).max)
        return np.ascontiguousarray(
            np.clip(image.astype(np.float32) * 255.0 / maximum, 0, 255).astype(np.uint8)
        )
    values = image.astype(np.float32, copy=False)
    low, high = (float(value) for value in np.percentile(values, (0.5, 99.8)))
    if not math.isfinite(low + high) or high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.ascontiguousarray(np.clip((values - low) * 255.0 / (high - low), 0, 255).astype(np.uint8))


class ImagePreview(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__("尚未取流", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#16191d;color:#aeb5bd;border:1px solid #30353b;")
        self._pixmap: QPixmap | None = None
        self._last_image: np.ndarray | None = None
        self._auto_stretch = False
        self._sensor_max_value: float | None = None

    def set_array(
        self,
        image: np.ndarray,
        *,
        auto_stretch: bool = False,
        sensor_max_value: float | None = None,
    ) -> None:
        self._last_image = image
        self._auto_stretch = auto_stretch
        self._sensor_max_value = sensor_max_value
        display = to_display_u8(
            image,
            auto_stretch=auto_stretch,
            sensor_max_value=sensor_max_value,
        )
        height, width = display.shape
        qimage = QImage(display.data, width, height, display.strides[0], QImage.Format.Format_Grayscale8).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_pixmap()

    def refresh_display(self, *, auto_stretch: bool) -> None:
        if self._last_image is not None:
            self.set_array(
                self._last_image,
                auto_stretch=auto_stretch,
                sensor_max_value=self._sensor_max_value,
            )

    def clear_image(self, text: str = "尚未取流") -> None:
        """清除上一张图，避免 QLabel resize 时恢复旧 pixmap。"""

        self._pixmap = None
        self._last_image = None
        self.clear()
        self.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))


class ResidualPlot(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.values: list[float] = []
        self.setMinimumHeight(220)

    def set_values(self, values: Sequence[float]) -> None:
        self.values = [float(value) for value in values if math.isfinite(float(value))]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        bounds = self.rect().adjusted(45, 18, -18, -32)
        painter.setPen(QPen(QColor("#cbd2da"), 1))
        painter.drawRect(bounds)
        if len(self.values) < 2:
            painter.setPen(QColor("#6b7280"))
            painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, "选择包含数值残差的 CSV 文件")
            return
        minimum, maximum = min(self.values), max(self.values)
        span = maximum - minimum or 1.0
        path = QPainterPath()
        for index, value in enumerate(self.values):
            x = bounds.left() + bounds.width() * index / (len(self.values) - 1)
            y = bounds.bottom() - bounds.height() * (value - minimum) / span
            if index == 0:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#1769aa"), 1.5))
        painter.drawPath(path)
        painter.setPen(QColor("#4b5563"))
        painter.drawText(4, bounds.top() + 5, f"{maximum:.4g}")
        painter.drawText(4, bounds.bottom(), f"{minimum:.4g}")
