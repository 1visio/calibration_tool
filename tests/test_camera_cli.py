import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from calibration_tool.cli import main


class CameraCliTests(unittest.TestCase):
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
