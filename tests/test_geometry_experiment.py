import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from calibration_tool.camera import load_capture_plan


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "geometry_experiment.py"
SPEC = importlib.util.spec_from_file_location("geometry_experiment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
GEOMETRY_EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEOMETRY_EXPERIMENT)


class GeometryExperimentTests(unittest.TestCase):
    def _write_complete_dataset(self, root: Path, config_id: str, exposure_us: float) -> Path:
        dataset = root / config_id
        frames = []
        csv_rows = []
        csv_fields = (
            "task_id", "index", "filename", "quality_passed", "quality_warnings",
            "transport_warnings", "exposure_us", "gain_db", "pixel_format",
            "width", "height", "offset_x", "offset_y",
        )
        for task_id in ("reference", "multiheight"):
            image_dir = dataset / "images" / task_id
            image_dir.mkdir(parents=True, exist_ok=True)
            for index in range(1, 51):
                relative = f"images/{task_id}/{index:04d}.tif"
                (dataset / relative).write_bytes(b"unchanged-image-bytes")
                frames.append({"task_id": task_id, "index": index, "filename": relative})
                csv_rows.append({
                    "task_id": task_id,
                    "index": index,
                    "filename": relative,
                    "quality_passed": index != 1,
                    "quality_warnings": "dynamic_range_low" if index == 1 else "",
                    "transport_warnings": "",
                    "exposure_us": exposure_us,
                    "gain_db": 0.0,
                    "pixel_format": "Mono8",
                    "width": 2448,
                    "height": 2048,
                    "offset_x": 0,
                    "offset_y": 0,
                })
        manifest = {
            "status": "completed",
            "plan": {"metadata": {"config_id": config_id}},
            "tasks": {
                task_id: {
                    "status": "completed",
                    "frames_expected": 50,
                    "frames_captured": 50,
                }
                for task_id in ("reference", "multiheight")
            },
            "frames": frames,
        }
        (dataset / "dataset_manifest.yaml").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        return dataset

    def test_init_creates_fixed_matrix_and_blank_mutable_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)

            for relative_dir in ("configs", "configs/generated", "data", "results"):
                self.assertTrue((experiment_dir / relative_dir).is_dir())

            with master_path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)

            self.assertEqual(tuple(reader.fieldnames or ()), GEOMETRY_EXPERIMENT.CSV_FIELDS)
            self.assertEqual(len(rows), 12)
            self.assertEqual(
                [row["config_id"] for row in rows],
                [
                    "B00_A05", "B00_A10", "B00_A15", "B00_A20",
                    "B05_A05", "B05_A10", "B05_A15", "B05_A20",
                    "B12p5_A05", "B12p5_A10", "B12p5_A15", "B12p5_A20",
                ],
            )
            self.assertEqual(
                [row["baseline_scale_reading"] for row in rows],
                ["0"] * 4 + ["5"] * 4 + ["12.5"] * 4,
            )
            for row in rows:
                for field in (*GEOMETRY_EXPERIMENT.MANUAL_FIELDS, *GEOMETRY_EXPERIMENT.AUTOMATED_FIELDS):
                    self.assertEqual(row[field], "")

    def test_existing_master_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            original = master_path.read_bytes()

            with self.assertRaises(FileExistsError):
                GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)

            self.assertEqual(master_path.read_bytes(), original)

    def test_make_plan_generates_two_tasks_per_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            original_master = master_path.read_bytes()

            paths = GEOMETRY_EXPERIMENT.make_capture_plans(experiment_dir)

            self.assertEqual(len(paths), 12)
            self.assertEqual(master_path.read_bytes(), original_master)
            first = load_capture_plan(paths[0])
            last = load_capture_plan(paths[-1])
            self.assertEqual([task.task_id for task in first.tasks], ["reference", "multiheight"])
            self.assertEqual([task.frames for task in first.tasks], [50, 50])
            self.assertEqual([task.quality_mode for task in first.tasks], ["laser", "laser"])
            self.assertEqual([task.config.exposure_us for task in first.tasks], [1500.0, 1500.0])
            self.assertEqual(first.tasks[0].relative_path(1).as_posix(), "images/reference/0001.tif")
            self.assertEqual(first.tasks[1].relative_path(50).as_posix(), "images/multiheight/0050.tif")
            self.assertEqual(first.metadata["baseline_scale_reading"], 0)
            self.assertIsNone(first.metadata["baseline_actual_mm"])
            self.assertEqual(first.metadata["laser_angle_deg"], 5)
            self.assertEqual(last.metadata["baseline_scale_reading"], 12.5)
            self.assertEqual(last.metadata["laser_angle_deg"], 20)
            self.assertEqual([task.config.exposure_us for task in last.tasks], [1900.0, 1900.0])
            self.assertEqual(first.metadata["working_distance_nominal_mm"], 1000)
            self.assertFalse(first.metadata["working_distance_calibrated"])

    def test_make_plan_preserves_manual_master_fields_and_refuses_plan_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            rows = GEOMETRY_EXPERIMENT._load_master_rows(master_path)
            rows[0]["baseline_actual_mm"] = "42.75"
            rows[0]["manual_notes"] = "人工测量"
            with master_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=GEOMETRY_EXPERIMENT.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            original_master = master_path.read_bytes()

            paths = GEOMETRY_EXPERIMENT.make_capture_plans(experiment_dir)
            first = load_capture_plan(paths[0])
            self.assertEqual(first.metadata["baseline_actual_mm"], 42.75)
            with self.assertRaises(FileExistsError):
                GEOMETRY_EXPERIMENT.make_capture_plans(experiment_dir)
            self.assertEqual(master_path.read_bytes(), original_master)

    def test_audit_captures_handles_invalid_fov_and_preserves_images_and_manual_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment_dir = Path(temporary) / "geometry_baseline_angle"
            master_path = GEOMETRY_EXPERIMENT.initialize_experiment(experiment_dir)
            fieldnames, master_rows = GEOMETRY_EXPERIMENT._read_master_table(master_path)
            master_rows[0]["baseline_actual_mm"] = "42.5"
            master_rows[0]["manual_notes"] = "视场外，人工确认"
            GEOMETRY_EXPERIMENT._write_csv(master_path, fieldnames, master_rows)

            data_root = experiment_dir / "data"
            expected_ids = [row["config_id"] for row in GEOMETRY_EXPERIMENT.build_initial_rows()]
            for config_id in expected_ids:
                if config_id == "B00_A05":
                    continue
                exposure = 1900.0 if config_id == "B12p5_A20" else 1500.0
                self._write_complete_dataset(data_root, config_id, exposure)
            sample_image = data_root / "B00_A10" / "images" / "reference" / "0001.tif"
            original_image = sample_image.read_bytes()

            result = GEOMETRY_EXPERIMENT.audit_captures(data_root, master_path)

            self.assertEqual(result["summary"], {
                "expected_conditions": 12,
                "captured_conditions": 11,
                "invalid_fov": 1,
                "complete_datasets": 11,
                "incomplete_datasets": 0,
            })
            self.assertFalse(result["camera_consistency"]["consistent"])
            self.assertEqual(sample_image.read_bytes(), original_image)
            self.assertTrue((experiment_dir / "results" / "capture_audit.csv").is_file())
            self.assertTrue((experiment_dir / "results" / "capture_audit.json").is_file())

            _, updated_rows = GEOMETRY_EXPERIMENT._read_master_table(master_path)
            invalid = updated_rows[0]
            self.assertEqual(invalid["status"], "invalid_fov")
            self.assertEqual(invalid["capture_complete"], "false")
            self.assertEqual(invalid["exclude_reason"], "laser_out_of_fov")
            self.assertEqual(invalid["phaseA_selected"], "false")
            self.assertEqual(invalid["baseline_actual_mm"], "42.5")
            self.assertEqual(invalid["manual_notes"], "视场外，人工确认")
            self.assertTrue(all(invalid[field] == "NaN" for field in GEOMETRY_EXPERIMENT.ANALYSIS_NAN_FIELDS))
            self.assertTrue(all(row["status"] == "captured" for row in updated_rows[1:]))
            first_captured = next(
                record for record in result["datasets"] if record["config_id"] == "B00_A10"
            )
            self.assertEqual(first_captured["quality_warning_frame_count"], 2)
            self.assertEqual(first_captured["quality_warning_occurrence_count"], 2)

    def test_reference_analysis_limits_surface_trims_segments_and_preserves_steger_detail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "B12p5_A20"
            image_dir = dataset / "images" / "reference"
            image_dir.mkdir(parents=True)
            frame_rows = []
            for index in range(1, 51):
                image = np.full((24, 16), 5, dtype=np.uint8)
                image[10:13, :] = 180
                ok, encoded = cv2.imencode(".tif", image)
                self.assertTrue(ok)
                filename = f"images/reference/{index:04d}.tif"
                (dataset / filename).write_bytes(encoded.tobytes())
                frame_rows.append({
                    "task_id": "reference",
                    "index": index,
                    "filename": filename,
                    "pixel_format": "Mono8",
                })
            (dataset / "dataset_manifest.yaml").write_text(
                json.dumps({
                    "status": "completed",
                    "plan": {"metadata": {"config_id": "B12p5_A20"}},
                }),
                encoding="utf-8",
            )
            with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("task_id", "index", "filename", "pixel_format")
                )
                writer.writeheader(); writer.writerows(frame_rows)
            steger_config = root / "realtime_steger.yaml"
            steger_config.write_text("steger: {}\n", encoding="utf-8")
            analysis_config = root / "analysis.yaml"
            analysis_config.write_text(
                "schema_version: 1\nsteger_config: realtime_steger.yaml\n"
                "reference:\n  valid_frame_fraction_min: 0.8\n  max_interp_gap_px: 1\n"
                "reference_surface:\n  x_range: [1, 14]\n  segment_edge_trim_px: 1\n"
                "  smooth_spline_basis_count: 12\n  smooth_spline_penalty: 1.0\n"
                "  robust_huber_delta: 1.5\n  robust_max_iterations: 10\n",
                encoding="utf-8",
            )

            valid = np.ones(16, dtype=bool)
            valid[5] = False  # one-column gap: local interpolation
            valid[10:13] = False  # larger gap: smooth model only
            fake_realtime = SimpleNamespace(
                load_steger_options=lambda _path: {"sigma": 1.5},
                extract_steger_columns=lambda _image, _options: SimpleNamespace(
                    u_px=np.arange(16, dtype=np.float64),
                    v_px=np.where(valid, 11.25, np.nan),
                    valid=valid,
                    response=np.where(valid, 2.0, np.nan),
                    offset_px=np.where(valid, 0.1, np.nan),
                    normal_y_abs=np.where(valid, 1.0, np.nan),
                ),
            )
            fake_src = root / "calibration_src"
            fake_src.mkdir()
            (fake_src / "realtime_steger.py").write_text("# formal stub\n", encoding="utf-8")
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                summary = GEOMETRY_EXPERIMENT.analyze_reference(
                    dataset, analysis_config, fake_src
                )

            self.assertEqual(summary["source_counts"], {
                "observed": 4,
                "short_gap_interpolated": 1,
                "smooth_model_filled": 3,
                "segment_edge_excluded": 6,
                "outside_reference_surface": 2,
                "invalid": 0,
            })
            self.assertFalse(summary["global_line_fit_applied"])
            self.assertFalse(summary["model_extrapolated_outside_reference_surface"])
            self.assertFalse(summary["multiheight_analyzed"])
            detail_path = dataset / "analysis" / "reference_frame_columns.csv"
            detail_before = detail_path.read_bytes()
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                second = GEOMETRY_EXPERIMENT.analyze_reference(
                    dataset, analysis_config, fake_src, overwrite=True
                )
            self.assertTrue(second["per_frame_steger_output_preserved"])
            self.assertEqual(detail_path.read_bytes(), detail_before)
            with (dataset / "analysis" / "reference_by_column.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[5]["source"], "short_gap_interpolated")
            self.assertEqual(
                [rows[index]["source"] for index in range(10, 13)],
                ["smooth_model_filled"] * 3,
            )
            self.assertEqual([rows[index]["source"] for index in (0, 15)], ["outside_reference_surface"] * 2)
            self.assertTrue(all(rows[index]["y_ref_smooth_px"] == "NaN" for index in (0, 15)))
            self.assertTrue(all(rows[index]["y_ref_smooth_px"] != "NaN" for index in range(1, 15)))
            self.assertEqual(rows[1]["source"], "segment_edge_excluded")
            self.assertFalse((dataset / "images" / "multiheight").exists())

    def test_preview_reference_roi_accepts_null_surface_without_building_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "B12p5_A20"
            image_dir = dataset / "images" / "reference"
            image_dir.mkdir(parents=True)
            rows = []
            for index in range(1, 51):
                image = np.full((12, 10), 5, dtype=np.uint8)
                image[6, :] = 200
                ok, encoded = cv2.imencode(".tif", image)
                self.assertTrue(ok)
                filename = f"images/reference/{index:04d}.tif"
                (dataset / filename).write_bytes(encoded.tobytes())
                rows.append({
                    "task_id": "reference", "index": index,
                    "filename": filename, "pixel_format": "Mono8",
                })
            (dataset / "dataset_manifest.yaml").write_text(
                json.dumps({
                    "status": "completed",
                    "plan": {"metadata": {"config_id": "B12p5_A20"}},
                }),
                encoding="utf-8",
            )
            with (dataset / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("task_id", "index", "filename", "pixel_format")
                )
                writer.writeheader(); writer.writerows(rows)
            (root / "realtime_steger.yaml").write_text("steger: {}\n", encoding="utf-8")
            config = root / "analysis.yaml"
            config.write_text(
                "schema_version: 1\nsteger_config: realtime_steger.yaml\n"
                "reference:\n  valid_frame_fraction_min: 0.8\n  max_interp_gap_px: 8\n"
                "reference_surface:\n  x_range: null\n  segment_edge_trim_px: 2\n",
                encoding="utf-8",
            )
            valid = np.ones(10, dtype=bool)
            fake_realtime = SimpleNamespace(
                load_steger_options=lambda _path: {"sigma": 1.5},
                extract_steger_columns=lambda _image, _options: SimpleNamespace(
                    u_px=np.arange(10, dtype=np.float64),
                    v_px=np.full(10, 6.0),
                    valid=valid,
                    response=np.ones(10),
                    offset_px=np.zeros(10),
                    normal_y_abs=np.ones(10),
                ),
            )
            fake_src = root / "calibration_src"
            fake_src.mkdir()
            (fake_src / "realtime_steger.py").write_text("# formal stub\n", encoding="utf-8")
            with patch.object(
                GEOMETRY_EXPERIMENT, "_load_realtime_steger", return_value=fake_realtime
            ):
                summary = GEOMETRY_EXPERIMENT.preview_reference_roi(dataset, config, fake_src)
            self.assertIsNone(summary["reference_surface_x_range"])
            self.assertFalse(summary["reference_model_built"])
            self.assertTrue((dataset / "analysis" / "reference_roi_preview.png").is_file())
            with self.assertRaises(ValueError):
                GEOMETRY_EXPERIMENT.analyze_reference(dataset, config, fake_src)
            self.assertFalse((dataset / "images" / "multiheight").exists())


if __name__ == "__main__":
    unittest.main()
