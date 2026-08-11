import unittest

import numpy as np

from calibration_tool.camera.models import QualityThresholds
from calibration_tool.camera.quality import analyze_frame, laser_column_metrics, summarize_quality


class CameraQualityTests(unittest.TestCase):
    def test_laser_coverage_detects_full_width_stripe(self):
        image = np.full((48, 64), 10, dtype=np.uint8)
        image[23:26, :] = 220
        quality = analyze_frame(
            image,
            sensor_max_value=255,
            mode="laser",
            thresholds=QualityThresholds(min_laser_coverage=0.8),
        )
        self.assertGreaterEqual(quality.laser_coverage, 0.99)
        self.assertNotIn("laser_coverage_low", quality.warnings)

    def test_laser_column_metrics_exposes_existing_width_and_saturation_formula(self):
        image = np.full((20, 8), 10, dtype=np.uint8)
        image[8:11, :] = 200
        image[9, 3] = 255
        metrics = laser_column_metrics(image, sensor_max_value=255)
        self.assertTrue(np.all(metrics["active"]))
        self.assertEqual(metrics["fwhm_px"].tolist(), [3] * 8)
        self.assertTrue(metrics["peak_saturated"][3])
        self.assertEqual(int(metrics["saturated_width_px"][3]), 1)

    def test_flat_dark_frame_is_reported(self):
        image = np.zeros((24, 32), dtype=np.uint8)
        quality = analyze_frame(image, sensor_max_value=255)
        self.assertIn("image_too_dark", quality.warnings)
        self.assertIn("dynamic_range_low", quality.warnings)
        summary = summarize_quality([quality])
        self.assertEqual(summary["warnings"], 1)

    def test_laser_mode_allows_dark_background(self):
        image = np.zeros((48, 64), dtype=np.uint16)
        image[23:26, :] = 1200
        quality = analyze_frame(image, sensor_max_value=4095, mode="laser")
        self.assertNotIn("image_too_dark", quality.warnings)
        self.assertGreater(quality.laser_coverage, 0.9)
        self.assertEqual(quality.laser_fwhm_p50_px, 3.0)
        self.assertEqual(quality.laser_fwhm_p95_px, 3.0)

    def test_laser_mode_detects_local_peak_saturation(self):
        image = np.full((200, 200), 10, dtype=np.uint8)
        image[100, :] = 255
        quality = analyze_frame(image, sensor_max_value=255, mode="laser")
        self.assertLess(quality.saturation_fraction, 0.05)
        self.assertIn("laser_saturation_high", quality.warnings)
        self.assertIn("laser_peak_saturated", quality.warnings)
        self.assertEqual(quality.laser_saturated_width_p95_px, 1.0)
        self.assertEqual(quality.laser_fwhm_p50_px, 1.0)

    def test_laser_mode_warns_before_hard_saturation(self):
        image = np.full((120, 160), 12, dtype=np.uint8)
        image[60, :] = 250
        quality = analyze_frame(image, sensor_max_value=255, mode="laser")
        self.assertEqual(quality.laser_peak_saturation_fraction, 0.0)
        self.assertIn("laser_peak_near_saturation", quality.warnings)

    def test_chessboard_mode_detects_complete_pattern(self):
        square = 36
        rows, columns = 9, 12  # 11 x 8 internal corners
        board = np.zeros((rows * square, columns * square), dtype=np.uint8)
        for row in range(rows):
            for column in range(columns):
                if (row + column) % 2:
                    board[row * square:(row + 1) * square, column * square:(column + 1) * square] = 220
        image = np.full((board.shape[0] + 100, board.shape[1] + 100), 110, dtype=np.uint8)
        image[50:50 + board.shape[0], 50:50 + board.shape[1]] = board
        quality = analyze_frame(
            image, sensor_max_value=255, mode="chessboard", board_pattern=(11, 8)
        )
        self.assertTrue(quality.chessboard_detected)
        self.assertEqual(quality.chessboard_pattern_used, (11, 8))
        self.assertNotIn("chessboard_not_found", quality.warnings)

    def test_chessboard_mode_detects_near_saturated_white_squares(self):
        square = 36
        rows, columns = 9, 12
        board = np.full((rows * square, columns * square), 20, dtype=np.uint8)
        for row in range(rows):
            for column in range(columns):
                if (row + column) % 2:
                    board[
                        row * square:(row + 1) * square,
                        column * square:(column + 1) * square,
                    ] = 250
        image = np.full((board.shape[0] + 100, board.shape[1] + 100), 110, dtype=np.uint8)
        image[50:50 + board.shape[0], 50:50 + board.shape[1]] = board
        quality = analyze_frame(
            image, sensor_max_value=255, mode="chessboard", board_pattern=(11, 8)
        )
        self.assertTrue(quality.chessboard_detected)
        self.assertIn("chessboard_near_saturation", quality.warnings)


if __name__ == "__main__":
    unittest.main()
