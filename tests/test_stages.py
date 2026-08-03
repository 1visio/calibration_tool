import tempfile
import unittest
from pathlib import Path

from calibration_tool.errors import StageExecutionError
from calibration_tool.stages import STAGES, ComputationService, options_to_argv
from calibration_tool.stages import _evaluate_stage


class StageTests(unittest.TestCase):
    def test_required_stages_are_registered(self) -> None:
        self.assertTrue(
            {
                "intrinsics",
                "laser_plane_shared_steger",
                "ground_extrinsics_board_only",
                "ground_bias",
                "reconstruct_shared_steger",
            }.issubset(STAGES)
        )

    def test_options_to_argv(self) -> None:
        self.assertEqual(
            options_to_argv(
                {"fit_dir": "fit", "free_k3": True, "exclude_fit": ["a.tif", "b.tif"]}
            ),
            [
                "--fit-dir",
                "fit",
                "--free-k3",
                "--exclude-fit",
                "a.tif",
                "b.tif",
            ],
        )

    def test_missing_calibration_src_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StageExecutionError):
                ComputationService(Path(directory) / "missing")

    def test_intrinsics_without_holdout_fails_quality(self) -> None:
        gates = _evaluate_stage(
            "intrinsics",
            {
                "fit_rmse_px": 0.1,
                "test_image_count": 0,
                "test_rmse_px": None,
            },
            None,
        )
        failed = {gate["id"] for gate in gates if gate["status"] == "fail"}
        self.assertIn("intrinsics.independent_test", failed)
        self.assertIn("intrinsics.test_rmse", failed)

    def test_compensation_without_holdout_fails_quality(self) -> None:
        gates = _evaluate_stage(
            "ground_bias",
            {"independent_validation_frame_count": 0},
            None,
        )
        self.assertEqual(gates[0]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
