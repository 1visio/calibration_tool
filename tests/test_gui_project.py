import tempfile
import unittest
from pathlib import Path

from calibration_tool.gui.project import WizardProject


class WizardProjectTests(unittest.TestCase):
    def test_round_trip_preserves_unknown_project_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera = root / "camera.yaml"; camera.write_text("backend: synthetic\n", encoding="utf-8")
            project = WizardProject(
                project_id="test", workspace=root / "work", camera_config=camera,
                pattern_cols=11, pattern_rows=8, square_size_mm=20,
                extra={"custom_metadata": {"owner": "test"}},
            )
            path = project.save(root / "project.yaml")
            loaded = WizardProject.load(path)
            self.assertEqual(loaded.project_id, "test")
            self.assertEqual(loaded.camera_config, camera.resolve())
            self.assertEqual(loaded.extra["custom_metadata"]["owner"], "test")

    def test_example_project_links_acceptance_plan(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "wizard_project.example.yaml"
        project = WizardProject.load(path)
        self.assertIsNotNone(project.acceptance_plan)
        self.assertTrue(project.acceptance_plan.is_file())

    def test_capture_artifacts_are_persisted_with_backup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera = root / "camera.yaml"
            camera.write_text("backend: synthetic\n", encoding="utf-8")
            project = WizardProject("test", root / "work", camera)
            path = project.save(root / "project.yaml")
            backup = project.record_capture_artifacts(
                {
                    "capture_plan": str(root / "plan.yaml"),
                    "dataset_root": str(root / "dataset"),
                    "fit_dir": str(root / "dataset" / "fit"),
                    "validation_dir": str(root / "dataset" / "validation"),
                    "dataset_manifest": str(root / "dataset" / "dataset_manifest.yaml"),
                    "frames_csv": str(root / "dataset" / "frames.csv"),
                }
            )
            self.assertEqual(backup, path.with_name("project.yaml.bak"))
            self.assertTrue(backup.is_file())
            loaded = WizardProject.load(path)
            self.assertEqual(loaded.extra["capture_artifacts"]["dataset_root"], str(root / "dataset"))


if __name__ == "__main__":
    unittest.main()
