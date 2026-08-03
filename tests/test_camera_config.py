import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_tool.camera.config import load_camera_config, load_capture_plan


ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = ROOT / "calibration_tool"


class CameraConfigTests(unittest.TestCase):
    def test_camera_geometry_matches_both_golden_runtime_configs(self):
        runtime = load_camera_config(TOOL_ROOT / "configs" / "camera.example.yaml")
        expected = (runtime["camera"].width, runtime["camera"].height)
        for config_path in (
            ROOT / "linelaser_tool" / "laser_measurement_tool" / "configs" / "measure_tool.yaml",
            ROOT / "0704line-laser-3d-scanner" / "laser_measurement_tool" / "configs" / "measure_tool.yaml",
        ):
            document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
            intrinsics = config_path.parent / document["calibration"]["intrinsics"]
            calibration = yaml.safe_load(intrinsics.read_text(encoding="utf-8-sig"))
            self.assertEqual(expected, (calibration["image_width"], calibration["image_height"]))

    def test_capture_plan_task_camera_inherits_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.yaml"
            path.write_text(
                """dataset_id: test
output_dir: output
backend: synthetic
camera:
  exposure_us: 1000
  gain_db: 2
  width: 64
  height: 48
tasks:
  - task_id: one
    frames: 1
    filename_template: image{suffix}
    camera:
      exposure_us: 2000
""",
                encoding="utf-8",
            )
            plan = load_capture_plan(path)
            self.assertEqual(plan.tasks[0].config.exposure_us, 2000)
            self.assertEqual(plan.tasks[0].config.gain_db, 2)
            self.assertEqual(plan.output_dir, Path(temporary) / "output")


if __name__ == "__main__":
    unittest.main()
