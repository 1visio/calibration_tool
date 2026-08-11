import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np

from calibration_tool.cli import main


class CameraCliTests(unittest.TestCase):
    def test_search_region_replay_is_read_only_and_reports_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "laser.png"
            image = np.zeros((96, 160), dtype=np.uint8)
            image[46:50, :] = 180
            self.assertTrue(cv2.imwrite(str(image_path), image))
            calibration_src = Path(__file__).resolve().parents[2] / "calibration" / "src"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main([
                    "search-region-replay",
                    str(image_path),
                    "--calibration-src",
                    str(calibration_src),
                    "--laser-orientation",
                    "horizontal",
                ])

            self.assertEqual(exit_code, 0, stderr.getvalue())
            self.assertIn("search_region_health:", stdout.getvalue())
            self.assertIn("normal_axis: v", stdout.getvalue())
            self.assertTrue(image_path.is_file())

    def test_exposure_series_runs_end_to_end_with_synthetic_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "camera.yaml"
            config.write_text(
                """backend: synthetic
serial_number: SIM-001
camera:
  exposure_us: 800
  pixel_format: Mono8
  width: 64
  height: 48
  timeout_ms: 100
backend_options:
  target_fps: 1000
""",
                encoding="utf-8",
            )
            output = root / "series"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main([
                    "capture-exposure-series",
                    "--config", str(config),
                    "--output", str(output),
                    "--dataset-id", "series",
                    "--pose-id", "pose_01",
                    "--exposures-us", "800", "1200",
                    "--frames-per-exposure", "2",
                    "--settle-frames", "0",
                    "--quality-mode", "laser",
                ])
            self.assertEqual(exit_code, 0, stderr.getvalue())
            self.assertIn("frame_count: 4", stdout.getvalue())
            self.assertTrue((output / "dataset_manifest.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
