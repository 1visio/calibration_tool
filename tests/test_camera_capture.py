import tempfile
import unittest
from pathlib import Path

import yaml

from calibration_tool.camera.capture import run_capture_plan
from calibration_tool.camera.capture import _new_manifest
from calibration_tool.camera.models import CameraConfig, CapturePlan, CaptureTask
from calibration_tool.camera.synthetic import SyntheticCameraProvider
from calibration_tool.errors import CaptureError
from calibration_tool.laser import LaserConfig


class _FailOnConfigureSession:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def configure(self, config):
        raise RuntimeError("intentional configure failure")


class _FailOnConfigureProvider(SyntheticCameraProvider):
    def open(self, serial_number, config):
        return _FailOnConfigureSession(super().open(serial_number, config))


class _RecordingProvider(SyntheticCameraProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sessions = []

    def open(self, serial_number, config):
        session = super().open(serial_number, config)
        self.sessions.append(session)
        return session


class CameraCaptureTests(unittest.TestCase):
    def test_manifest_records_actual_laser_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary) / "dataset")
            plan = CapturePlan(
                dataset_id=plan.dataset_id,
                output_dir=plan.output_dir,
                backend=plan.backend,
                serial_number=plan.serial_number,
                base_config=plan.base_config,
                tasks=plan.tasks,
                quality_thresholds=plan.quality_thresholds,
                laser=LaserConfig("vertical"),
            )
            manifest = _new_manifest(plan)
            self.assertEqual(manifest["laser"], {"orientation": "vertical"})
            self.assertEqual(manifest["plan"]["laser"], {"orientation": "vertical"})

    def _plan(self, output: Path) -> CapturePlan:
        base = CameraConfig(exposure_us=800, width=64, height=48, timeout_ms=100)
        tasks = (
            CaptureTask(
                "exposure_01", 2, "800us/frame_{index02}{suffix}", base,
                settle_frames=0, image_format="png", quality_mode="laser",
            ),
            CaptureTask(
                "exposure_02", 1, "1200us/frame_{index02}{suffix}",
                base.updated({"exposure_us": 1200}),
                settle_frames=0, image_format="png", quality_mode="laser",
            ),
        )
        return CapturePlan("test_set", output, "synthetic", "SIM-001", base, tasks)

    def test_capture_writes_manifest_csv_images_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            result = run_capture_plan(
                self._plan(output),
                SyntheticCameraProvider(target_fps=1000),
            )
            self.assertEqual(result.frame_count, 3)
            manifest = yaml.safe_load((output / "dataset_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["laser"], {"orientation": "horizontal"})
            self.assertEqual(len(manifest["frames"]), 3)
            self.assertTrue((output / "frames.csv").is_file())
            self.assertFalse((output / ".task_staging").exists())
            for frame in manifest["frames"]:
                self.assertTrue((output / frame["filename"]).is_file())
                self.assertEqual(len(frame["sha256"]), 64)
                self.assertIn("applied_camera", frame)
                self.assertIn("quality", frame)

    def test_task_config_is_applied_before_before_task_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = _RecordingProvider(target_fps=1000)
            seen = []

            def before_task(task):
                seen.append((task.task_id, provider.sessions[0].config.exposure_us))
                return True

            run_capture_plan(
                self._plan(Path(temporary) / "dataset"),
                provider,
                before_task=before_task,
            )

            self.assertEqual(seen, [("exposure_01", 800.0), ("exposure_02", 1200.0)])

    def test_failed_plan_resumes_without_recapturing_completed_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            plan = self._plan(output)
            with self.assertRaises(CaptureError):
                run_capture_plan(plan, _FailOnConfigureProvider(target_fps=1000))
            work = output.parent / f".{output.name}.inprogress"
            failed = yaml.safe_load((work / "dataset_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(failed["tasks"]["exposure_01"]["status"], "completed")
            self.assertEqual(failed["tasks"]["exposure_02"]["status"], "failed")

            result = run_capture_plan(
                plan,
                SyntheticCameraProvider(target_fps=1000),
                resume=True,
            )
            self.assertEqual(result.frame_count, 3)
            completed = yaml.safe_load((output / "dataset_manifest.yaml").read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(len([x for x in completed["frames"] if x["task_id"] == "exposure_01"]), 2)


if __name__ == "__main__":
    unittest.main()
