from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QTableWidgetItem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration_tool.gui.main_window import CalibrationWizardWindow
from calibration_tool.io_utils import load_document


OUTPUT = ROOT / "docs" / "images" / "user_manual"


def process_events() -> None:
    for _ in range(4):
        QCoreApplication.processEvents()


def save_page(window: CalibrationWizardWindow, index: int, filename: str) -> None:
    window.steps.setCurrentRow(index)
    process_events()
    target = OUTPUT / filename
    if not window.grab().save(str(target), "PNG"):
        raise RuntimeError(f"无法保存截图：{target}")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = CalibrationWizardWindow(
        project_path=ROOT / "configs" / "wizard_project.example.yaml",
        default_camera_config=ROOT / "configs" / "camera.mvs.example.yaml",
    )
    window.resize(1600, 1000)
    window.show()
    process_events()

    save_page(window, 0, "01_project.png")

    camera = window.camera_page
    camera.devices.clear()
    camera.devices.addItem("MV-CS050-60GM (DA7711077)", "DA7711077")
    camera.status.setText("取流中 · MV-CS050-60GM · SN DA7711077 · 2448×2048")
    camera.quality_mode.setCurrentIndex(camera.quality_mode.findData("chessboard"))
    camera.exposure.setValue(50000.0)
    camera.pixel_format.setCurrentText("Mono8")
    camera.quality.setText("通过 · 动态范围 138.0 DN8 · 清晰度 158.2 · 已找到 11×8 完整内角点")
    preview_path = (
        ROOT / "runs" / "smoke_ground_board_20260803" / "diagnostics" / "fit"
        / "Image_20260730165210650.png"
    )
    preview = cv2.imread(str(preview_path), cv2.IMREAD_GRAYSCALE)
    if preview is None:
        raise RuntimeError(f"无法读取预览示例：{preview_path}")
    camera.preview.set_array(preview, sensor_max_value=255)
    save_page(window, 1, "02_camera.png")

    capture = window.capture_page
    capture.output.setText(
        str(ROOT / "projects" / "mv-cs050-60gm-calibration" / "data" / "intrinsics_fit_pose_01")
    )
    capture.dataset_id.setText("intrinsics_fit_pose_01")
    capture.pose_id.setText("pose_01")
    capture.role.setCurrentIndex(capture.role.findData("intrinsics"))
    capture.exposures.setText("30000, 40000, 50000")
    capture.frames.setValue(1)
    capture.progress.setPlainText(
        "开始采集…\n"
        "intrinsics_01  1/1  通过\n✓ intrinsics_01 已提交\n"
        "intrinsics_02  1/1  通过\n✓ intrinsics_02 已提交\n"
        "intrinsics_03  1/1  通过\n✓ intrinsics_03 已提交\n"
        "完成：3 帧；请从 manifest 中选择完整角点稳定通过的曝光。"
    )
    save_page(window, 2, "03_capture.png")

    calibration = window.calibration_page
    calibration.workflow.setText(str(ROOT / "configs" / "workflow.example.yaml"))
    calibration.refresh_plan(silent=True)
    for row in range(calibration.stage_table.rowCount()):
        calibration.stage_table.setItem(row, 0, QTableWidgetItem("是（配置示例）"))
    calibration.log.setPlainText(
        "▶ intrinsics 开始\nintrinsics: completed\n"
        "▶ laser_plane_shared_steger 开始\nlaser_plane_shared_steger: completed\n"
        "▶ ground_extrinsics_board_only 开始\nground_extrinsics_board_only: completed\n"
        "▶ ground_bias 开始\nground_bias: completed\nWorkflow: completed"
    )
    save_page(window, 3, "04_calibration.png")

    results = window.results_page
    report_path = ROOT / "reports" / "current_acceptance" / "acceptance_report.yaml"
    report = load_document(report_path)
    results.show_result(report)
    results.acceptance_status.setText(
        "验收结论：rejected · PASS 15 · WARN 1 · FAIL 12 · 发布：disabled（历史数据示例）"
    )
    results.last_html = report_path.with_suffix(".html")
    results.open_html_button.setEnabled(True)
    residual = ROOT / "runs" / "smoke_ground_board_20260803" / "validation_frames.csv"
    results.artifacts.insertItem(0, str(residual))
    results.artifacts.setCurrentRow(0)
    process_events()
    save_page(window, 4, "05_results.png")

    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
