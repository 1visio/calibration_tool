import unittest
from pathlib import Path

from calibration_tool.golden import check_golden_baseline
from calibration_tool.profiles import load_runtime_profile


ROOT = Path(__file__).resolve().parents[1]


class GoldenSnapshotTests(unittest.TestCase):
    def test_generated_baseline_matches_sources(self) -> None:
        result = check_golden_baseline(ROOT / "golden" / "baseline.yaml")
        self.assertTrue(result["matches"])

    def test_online_snapshot_manifest_is_self_contained(self) -> None:
        profile = load_runtime_profile(
            ROOT / "golden" / "snapshots" / "online_measurement_tool" / "measure_tool.yaml",
            expected_extractor="shared_steger",
        )
        manifest = profile["manifest"]
        self.assertIsNotNone(manifest)
        self.assertTrue(all(item["matches"] for item in manifest["files"].values()))


if __name__ == "__main__":
    unittest.main()
