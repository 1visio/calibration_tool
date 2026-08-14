#!/usr/bin/env python3
"""Laser-model-only replay for the frozen Daheng gauge pixels.

The script deliberately reads only the saved ``u,v`` columns from the historical
CSV files.  Stored XYZ columns are never used.  M0 is evaluated first and the
candidate CSV is written only when the M0 replay agrees with every historical
``result.json`` within the declared tolerance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_TOOL_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
HISTORY_ROOT = (
    WORKSPACE_ROOT
    / "0704line-laser-3d-scanner"
    / "laser_measurement_tool"
    / "output_daheng_0811"
)
CANDIDATE_ROOT = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0813"
    / "cone_nonlinear_fit_trust_region"
)
DEFAULT_OUTPUT = CANDIDATE_ROOT / "gauge_model_replay_comparison.csv"
DEFAULT_CANDIDATES = CANDIDATE_ROOT / "cone_nonlinear_fit_candidates.csv"
DEFAULT_CONFIG = MEASUREMENT_TOOL_ROOT / "configs" / "measure_tool_daheng_0811.yaml"

FRAME_IDS = (
    "frame_061303_measure",
    "frame_062878_measure",
    "frame_063995_measure",
    "frame_065292_measure",
)
MODEL_NAMES = ("M0", "M1_point_equal", "M2_frame_equal", "M3_v_region_equal")
PARAMETER_NAMES = ("theta_axis", "phi_axis", "A_x", "A_y", "A_z", "alpha")
NOMINAL_HEIGHT_MM = 50.0

# These are deliberately much looser than the observed M0 differences (~3e-8 mm),
# while still detecting an offset, coordinate-frame, runtime-config, or filtering
# mistake immediately.
M0_TOLERANCE_MM = 5.0e-6


class RecaptureRequired(RuntimeError):
    """A historical result does not contain the required frozen pixels."""


def _add_import_root() -> None:
    path = str(MEASUREMENT_TOOL_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve_recorded_path(value: str | Path, workspace_root: Path) -> Path:
    """Resolve a Windows path recorded on a machine with the old root spelling."""
    raw = str(value)
    candidates = [Path(raw)]
    normalized = raw.replace("\\", "/")
    markers = (
        "/0704line-laser-3d-scanner/",
        "/linelaser_tool/",
        "/calibration_tool/",
    )
    for marker in markers:
        index = normalized.lower().find(marker.lower())
        if index >= 0:
            candidates.append(workspace_root / normalized[index + 1 :])
            break
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Recorded path does not exist: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_uv_csv(path: Path) -> np.ndarray:
    """Read only ``u,v``; never parse or fall back to stored XYZ columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"u", "v"}.issubset(reader.fieldnames):
            raise RecaptureRequired(
                f"RECAPTURE_REQUIRED: {path} lacks saved u,v columns"
            )
        rows: list[tuple[float, float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                u = float(row["u"])
                v = float(row["v"])
            except (TypeError, ValueError) as error:
                raise RecaptureRequired(
                    f"RECAPTURE_REQUIRED: invalid u,v in {path}:{line_number}"
                ) from error
            if not np.isfinite([u, v]).all():
                raise RecaptureRequired(
                    f"RECAPTURE_REQUIRED: non-finite u,v in {path}:{line_number}"
                )
            rows.append((u, v))
    if not rows:
        raise RecaptureRequired(f"RECAPTURE_REQUIRED: {path} has no saved u,v rows")
    return np.asarray(rows, dtype=np.float64)


def _load_runtime(config_path: Path, recorded_config_path: Path | None) -> tuple[Any, Any]:
    """Build the effective runtime dataclasses from the recorded 0811 config."""
    from measurement.height_measure import MeasurementParams
    from reconstruction.reconstructor import ReconstructionParams

    document_path = recorded_config_path or config_path
    document = yaml.safe_load(document_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"Runtime config root must be a mapping: {document_path}")
    reconstruction_document = document.get("reconstruction", {})
    measurement_document = document.get("measurement", {})
    if not isinstance(reconstruction_document, Mapping) or not isinstance(
        measurement_document, Mapping
    ):
        raise ValueError("Runtime config reconstruction/measurement must be mappings")
    reconstruction_fields = {
        field.name for field in ReconstructionParams.__dataclass_fields__.values()
    }
    measurement_fields = {
        field.name for field in MeasurementParams.__dataclass_fields__.values()
    }
    unknown_reconstruction = set(reconstruction_document) - reconstruction_fields
    if unknown_reconstruction and any(
        reconstruction_document[key] is not None for key in unknown_reconstruction
    ):
        raise ValueError(
            "Runtime config contains unsupported non-null reconstruction fields: "
            f"{sorted(unknown_reconstruction)}"
        )
    unknown_measurement = set(measurement_document) - measurement_fields
    if unknown_measurement:
        raise ValueError(
            "Runtime config contains unsupported measurement fields: "
            f"{sorted(unknown_measurement)}"
        )
    reconstruction = ReconstructionParams(
        **{
            key: value
            for key, value in reconstruction_document.items()
            if key in reconstruction_fields
        }
    )
    measurement = MeasurementParams(
        **{
            key: value
            for key, value in measurement_document.items()
            if key in measurement_fields
        }
    )
    return reconstruction, measurement


def _load_calibration_from_result(
    result: Mapping[str, Any],
    fallback_config: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    from calibration.config_loader import load_calibration_files

    calibration_document = result.get("calibration")
    if not isinstance(calibration_document, Mapping):
        raise ValueError("result.json missing calibration mapping")

    def recorded(name: str, fallback: Path) -> Path:
        value = calibration_document.get(name)
        if isinstance(value, str) and value.strip():
            return _resolve_recorded_path(value, WORKSPACE_ROOT)
        return fallback

    config_document = yaml.safe_load(fallback_config.read_text(encoding="utf-8")) or {}
    config_calibration = config_document.get("calibration", {})
    if not isinstance(config_calibration, Mapping):
        raise ValueError(f"Invalid calibration section in {fallback_config}")
    base_dir = fallback_config.parent

    def fallback(name: str) -> Path:
        value = config_calibration.get(name)
        if not isinstance(value, str):
            raise ValueError(f"Runtime config missing calibration.{name}")
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path).resolve()

    intrinsics = recorded("intrinsics", fallback("intrinsics"))
    laser_model = recorded("laser_model", recorded("laser_plane", fallback("laser_model")))
    extrinsics = recorded("extrinsics", fallback("extrinsics"))
    ground_value = calibration_document.get("ground_u_compensation")
    if ground_value in (None, ""):
        ground = None
    else:
        ground = _resolve_recorded_path(str(ground_value), WORKSPACE_ROOT)
    calibration = load_calibration_files(
        intrinsics=intrinsics,
        laser_plane=laser_model,
        extrinsics=extrinsics,
        ground_u_compensation=ground,
        ground_u_optional=True,
    )
    return calibration, intrinsics, laser_model, extrinsics


def _load_candidates(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, np.ndarray] = {}
    for model_name in ("point_equal", "frame_equal", "v_region_equal"):
        values: list[float] = []
        for parameter in PARAMETER_NAMES:
            column = f"final_{model_name}"
            row = next((item for item in rows if item.get("parameter") == parameter), None)
            if row is None or column not in row or row[column] in (None, ""):
                raise ValueError(f"Candidate CSV missing {column}/{parameter}")
            values.append(float(row[column]))
        result[model_name] = np.asarray(values, dtype=np.float64)
    return result


def _model_from_theta(base_model: Mapping[str, Any], theta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(theta, dtype=np.float64).reshape(6)
    theta_axis, phi_axis = values[:2]
    axis = np.asarray(
        [
            math.sin(theta_axis) * math.cos(phi_axis),
            math.sin(theta_axis) * math.sin(phi_axis),
            math.cos(theta_axis),
        ],
        dtype=np.float64,
    )
    axis /= np.linalg.norm(axis)
    model = copy.deepcopy(dict(base_model))
    model["axis_unit_camera"] = axis
    model["apex_camera_mm"] = np.ascontiguousarray(values[2:5])
    model["half_apex_angle_deg"] = math.degrees(float(values[5]))
    return model


def _measure(
    pixels_baseline: np.ndarray,
    pixels_height: np.ndarray,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
    measurement_params: Any,
) -> tuple[Any, Any, Any]:
    from measurement.height_measure import measure_height_line
    from reconstruction.reconstructor import reconstruct_uv_to_ground

    baseline = reconstruct_uv_to_ground(
        pixels_baseline, calibration, reconstruction_params
    )
    height = reconstruct_uv_to_ground(pixels_height, calibration, reconstruction_params)
    measurement = measure_height_line(
        baseline.points_ground, height.points_ground, measurement_params
    )
    return baseline, height, measurement


def _metric_row(
    model_name: str,
    frame_id: str,
    result_path: Path,
    result: Mapping[str, Any],
    baseline: Any,
    height: Any,
    measurement: Any,
    candidate_theta: np.ndarray | None,
    m0_deltas: Mapping[str, float],
    m0_status: str,
) -> dict[str, Any]:
    profile = measurement.ground_profile_fit
    model_values = {
        "theta_axis_rad": "",
        "phi_axis_rad": "",
        "apex_x_mm": "",
        "apex_y_mm": "",
        "apex_z_mm": "",
        "half_apex_angle_rad": "",
    }
    if candidate_theta is not None:
        model_values = {
            "theta_axis_rad": float(candidate_theta[0]),
            "phi_axis_rad": float(candidate_theta[1]),
            "apex_x_mm": float(candidate_theta[2]),
            "apex_y_mm": float(candidate_theta[3]),
            "apex_z_mm": float(candidate_theta[4]),
            "half_apex_angle_rad": float(candidate_theta[5]),
        }
    filters_baseline = json.dumps(baseline.filtered, sort_keys=True, separators=(",", ":"))
    filters_height = json.dumps(height.filtered, sort_keys=True, separators=(",", ":"))
    row: dict[str, Any] = {
        "model": model_name,
        "frame": frame_id,
        "source_result_json": str(result_path),
        "nominal_height_mm": NOMINAL_HEIGHT_MM,
        "height_mean": float(measurement.height_mean_mm),
        "height_std": float(measurement.height_std_mm),
        "signed_error": float(measurement.height_mean_mm - NOMINAL_HEIGHT_MM),
        "absolute_error": float(abs(measurement.height_mean_mm - NOMINAL_HEIGHT_MM)),
        "baseline_z": float(measurement.ground_baseline_zg_mm),
        "baseline_rmse": float(profile.rmse_mm if profile is not None else float("nan")),
        "height_fit_rmse": float(measurement.height_fit.rmse_mm),
        "valid_baseline_points": int(baseline.point_count),
        "valid_height_points": int(height.point_count),
        "baseline_inlier_points": int(measurement.baseline_inlier_count),
        "height_inlier_points": int(measurement.height_inlier_count),
        "input_baseline_points": int(len(baseline.pixels_uv) + sum(baseline.filtered.values())),
        "input_height_points": int(len(height.pixels_uv) + sum(height.filtered.values())),
        "filtered_baseline": filters_baseline,
        "filtered_height": filters_height,
        "m0_replay_status": m0_status,
        "m0_height_mean_delta": m0_deltas["height_mean"],
        "m0_height_std_delta": m0_deltas["height_std"],
        "m0_baseline_z_delta": m0_deltas["baseline_z"],
        "m0_baseline_rmse_delta": m0_deltas["baseline_rmse"],
        "m0_height_fit_rmse_delta": m0_deltas["height_fit_rmse"],
        "m0_point_count_match": m0_deltas["point_count_match"],
    }
    row.update(model_values)
    return row


def _compare_m0(
    result: Mapping[str, Any],
    baseline: Any,
    height: Any,
    measurement: Any,
) -> dict[str, Any]:
    expected = result.get("results_mm")
    expected_counts = result.get("point_counts")
    if not isinstance(expected, Mapping) or not isinstance(expected_counts, Mapping):
        raise ValueError("result.json missing results_mm/point_counts")
    expected_profile = expected.get("ground_profile")
    if not isinstance(expected_profile, Mapping):
        raise ValueError("result.json missing results_mm.ground_profile")
    actual = {
        "height_mean": float(measurement.height_mean_mm),
        "height_std": float(measurement.height_std_mm),
        "baseline_z": float(measurement.ground_baseline_zg_mm),
        "baseline_rmse": float(measurement.ground_profile_fit.rmse_mm),
        "height_fit_rmse": float(measurement.height_fit.rmse_mm),
    }
    expected_values = {
        "height_mean": float(expected["height_mean"]),
        "height_std": float(expected["height_std"]),
        "baseline_z": float(expected["ground_baseline_zg"]),
        "baseline_rmse": float(expected_profile["rmse_mm"]),
        "height_fit_rmse": float(expected["height_line_fit_rmse"]),
    }
    deltas = {
        key: actual[key] - expected_values[key] for key in actual
    }
    expected_counts_tuple = (
        int(expected_counts["baseline_total"]),
        int(expected_counts["height_total"]),
        int(expected_counts["baseline_inliers"]),
        int(expected_counts["height_inliers"]),
    )
    actual_counts_tuple = (
        baseline.point_count,
        height.point_count,
        measurement.baseline_inlier_count,
        measurement.height_inlier_count,
    )
    point_count_match = actual_counts_tuple == expected_counts_tuple
    deltas["point_count_match"] = point_count_match
    failed = [
        f"{key} delta={value:.9g}"
        for key, value in deltas.items()
        if key != "point_count_match" and abs(float(value)) > M0_TOLERANCE_MM
    ]
    if not point_count_match:
        failed.append(f"point_counts actual={actual_counts_tuple} expected={expected_counts_tuple}")
    if failed:
        raise RuntimeError(
            "M0_REPLAY_FAILED; STOP before comparing refined models. "
            "Inspect image_offset, full-image/ROI coordinates, runtime config, "
            "and filtering. "
            + "; ".join(failed)
        )
    return deltas


def replay(
    history_root: Path,
    candidate_path: Path,
    output_path: Path,
    config_path: Path,
) -> list[dict[str, Any]]:
    _add_import_root()

    records: dict[str, dict[str, Any]] = {}
    uv_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for frame_id in FRAME_IDS:
        directory = history_root / frame_id
        required = (
            "laser_center.csv",
            "baseline_points.csv",
            "height_points.csv",
            "result.json",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise RecaptureRequired(
                f"RECAPTURE_REQUIRED: {directory} missing {', '.join(missing)}"
            )
        # Validate all saved-pixel artifacts, while using only the selected groups
        # for measurement.  No XYZ value is read from any CSV.
        _read_uv_csv(directory / "laser_center.csv")
        pixels_baseline = _read_uv_csv(directory / "baseline_points.csv")
        pixels_height = _read_uv_csv(directory / "height_points.csv")
        result_path = directory / "result.json"
        result = _read_json(result_path)
        offset = result.get("image_offset")
        if not isinstance(offset, Mapping) or {"u", "v"} - set(offset):
            raise RuntimeError(
                f"M0_REPLAY_FAILED: {result_path} missing image_offset metadata"
            )
        records[frame_id] = {"result": result, "path": result_path}
        uv_data[frame_id] = (pixels_baseline, pixels_height)

    recorded_config_value = records[FRAME_IDS[0]]["result"].get("config")
    recorded_config_path = None
    if isinstance(recorded_config_value, str) and recorded_config_value.strip():
        recorded_config_path = _resolve_recorded_path(recorded_config_value, WORKSPACE_ROOT)
    reconstruction_params, measurement_params = _load_runtime(
        config_path, recorded_config_path
    )
    calibration, intrinsics_path, cone_path, extrinsics_path = _load_calibration_from_result(
        records[FRAME_IDS[0]]["result"], config_path
    )
    base_model = copy.deepcopy(calibration["laser_model"])
    if base_model.get("model_type") != "circular_cone":
        raise ValueError("M0 calibration model is not circular_cone")

    # Check that every frame records the same 0811 calibration provenance.
    expected_hashes = {
        "intrinsics": _sha256(intrinsics_path),
        "laser_model": _sha256(cone_path),
        "extrinsics": _sha256(extrinsics_path),
    }
    for frame_id in FRAME_IDS:
        result = records[frame_id]["result"]
        frame_calibration = result.get("calibration")
        if not isinstance(frame_calibration, Mapping):
            raise ValueError(f"{frame_id} result.json missing calibration")
        for key, expected_hash in expected_hashes.items():
            raw = frame_calibration.get(key)
            if not isinstance(raw, str):
                raise ValueError(f"{frame_id} result.json missing calibration.{key}")
            path = _resolve_recorded_path(raw, WORKSPACE_ROOT)
            if _sha256(path) != expected_hash:
                raise ValueError(f"{frame_id} calibration.{key} differs from M0 calibration")

    candidates = _load_candidates(candidate_path)
    results: list[dict[str, Any]] = []
    m0_deltas_by_frame: dict[str, dict[str, Any]] = {}

    # M0 is intentionally a complete pass before any refined model is touched.
    for frame_id in FRAME_IDS:
        result = records[frame_id]["result"]
        pixels_baseline, pixels_height = uv_data[frame_id]
        baseline, height, measurement = _measure(
            pixels_baseline,
            pixels_height,
            calibration,
            reconstruction_params,
            measurement_params,
        )
        deltas = _compare_m0(result, baseline, height, measurement)
        m0_deltas_by_frame[frame_id] = deltas
        results.append(
            _metric_row(
                "M0",
                frame_id,
                records[frame_id]["path"],
                result,
                baseline,
                height,
                measurement,
                None,
                deltas,
                "PASS",
            )
        )

    for model_name, candidate_key in zip(
        ("M1_point_equal", "M2_frame_equal", "M3_v_region_equal"),
        ("point_equal", "frame_equal", "v_region_equal"),
    ):
        candidate_calibration = copy.deepcopy(calibration)
        candidate_theta = candidates[candidate_key]
        candidate_calibration["laser_model"] = _model_from_theta(
            base_model, candidate_theta
        )
        for frame_id in FRAME_IDS:
            result = records[frame_id]["result"]
            pixels_baseline, pixels_height = uv_data[frame_id]
            baseline, height, measurement = _measure(
                pixels_baseline,
                pixels_height,
                candidate_calibration,
                reconstruction_params,
                measurement_params,
            )
            results.append(
                _metric_row(
                    model_name,
                    frame_id,
                    records[frame_id]["path"],
                    result,
                    baseline,
                    height,
                    measurement,
                    candidate_theta,
                    m0_deltas_by_frame[frame_id],
                    "PASS",
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = replay(
            args.history_root.resolve(),
            args.candidate_csv.resolve(),
            args.output.resolve(),
            args.config.resolve(),
        )
    except RecaptureRequired as error:
        print(str(error), file=sys.stderr)
        return 3
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"M0_REPLAY=PASS rows={len(rows)} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
