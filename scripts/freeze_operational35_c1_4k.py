#!/usr/bin/env python3
"""Deterministically fit and freeze only the existing Operational-35 C1_4k.

This exporter deliberately reuses the original C1 fitting implementation but
calls it exactly once for C1_4k on the 35-pose set.  It does not run grouped
CV, C1_3k/C1_5k, model selection, Validation, or any C0 fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
PROJECT = ROOT / "projects" / "daheng"
SOURCE_DIR = PROJECT / "outputs/0818/c1_frozen_quadratic_grouped_cv"
SELECTION_DIR = PROJECT / "outputs/0818/c1_operational35_selection"
OUTPUT_DEFAULT = PROJECT / "outputs/0818/c1_4k_freeze"

POINTS = PROJECT / "outputs/0818/quadratic_residual_observability/quadratic_residual_points.csv"
AUDIT = PROJECT / "outputs/0818/quadratic_residual_observability/audit_summary.json"
FROZEN_C0 = PROJECT / "outputs/0818/c0_freeze/quadratic_graph.yaml"
SOURCE_MANIFEST = SOURCE_DIR / "c1_run_manifest.json"
SELECTION_MANIFEST = SELECTION_DIR / "c1_operational35_selection_manifest.json"
HISTORICAL_STRESS = SOURCE_DIR / "frame027_stress_test.csv"

FRAME027 = "027"
CANDIDATE_ID = "C1_4k"
INTERIOR_KNOT_COUNT = 4
SPLINE_DEGREE = 3
METRIC_TOL_MM = 1.0e-9
LUT_TOL_MM = 1.0e-3
PCA_TOL = 1.0e-12


def load_original_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import fit_c1_frozen_quadratic_grouped_cv as original  # type: ignore

    return original


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Non-finite value cannot enter frozen JSON")
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        clean_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "repository": str(ROOT),
        "commit": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def reproduce_full36_pca(points: pd.DataFrame, audit_summary: Mapping[str, Any]) -> dict[str, Any]:
    residual = points["residual_mm"].to_numpy(float)
    valid = np.isfinite(residual)
    xy = points.loc[valid, ["xn", "yn"]].to_numpy(float)
    center = np.mean(xy, axis=0)
    centered = xy - center
    covariance = np.cov(centered, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axis_s = np.asarray(eigenvectors[:, order[0]], dtype=float)
    if (abs(axis_s[1]) >= abs(axis_s[0]) and axis_s[1] < 0) or (
        abs(axis_s[1]) < abs(axis_s[0]) and axis_s[0] < 0
    ):
        axis_s = -axis_s
    axis_t = np.asarray([-axis_s[1], axis_s[0]], dtype=float)
    coordinates = centered @ np.column_stack([axis_s, axis_t])
    explained = eigenvalues / np.sum(eigenvalues)
    stored_s = points.loc[valid, "pca_s"].to_numpy(float)
    max_abs_diff = float(np.max(np.abs(coordinates[:, 0] - stored_s)))
    if max_abs_diff > PCA_TOL:
        raise RuntimeError(f"Reproduced Full-36 PCA-s differs from stored pca_s: {max_abs_diff}")

    expected = audit_summary.get("pca", {})
    pca = {
        "coordinate_definition": "(xn, yn) centered by Full-36 PCA center and projected onto axis_s",
        "source_points_include_frame027": True,
        "center_xn": float(center[0]),
        "center_yn": float(center[1]),
        "axis_s_xn": float(axis_s[0]),
        "axis_s_yn": float(axis_s[1]),
        "axis_t_xn": float(axis_t[0]),
        "axis_t_yn": float(axis_t[1]),
        "eigenvalue_s": float(eigenvalues[0]),
        "eigenvalue_t": float(eigenvalues[1]),
        "explained_s": float(explained[0]),
        "explained_t": float(explained[1]),
        "s_robust_span": float(np.percentile(coordinates[:, 0], 95) - np.percentile(coordinates[:, 0], 5)),
        "t_robust_span": float(np.percentile(coordinates[:, 1], 95) - np.percentile(coordinates[:, 1], 5)),
        "anisotropy_sqrt_eigenvalue_ratio": float(np.sqrt(eigenvalues[0] / eigenvalues[1])),
        "stored_pca_s_max_abs_diff": max_abs_diff,
    }
    for key in (
        "center_xn",
        "center_yn",
        "axis_s_xn",
        "axis_s_yn",
        "axis_t_xn",
        "axis_t_yn",
    ):
        if key in expected and abs(float(expected[key]) - pca[key]) > PCA_TOL:
            raise RuntimeError(f"Reproduced PCA field differs from audit summary: {key}")
    return pca


def metric_row(original: Any, frame_id: str, residual: np.ndarray) -> dict[str, float]:
    metrics = original.scalar_metrics([frame_id] * len(residual), residual)
    return {
        "rmse_mm": float(metrics["rmse_mm"]),
        "p95_abs_mm": float(metrics["p95_abs_mm"]),
        "p99_abs_mm": float(metrics["p99_abs_mm"]),
        "max_abs_mm": float(metrics["max_abs_mm"]),
    }


def write_reproduction_check(
    path: Path,
    historical: Mapping[str, float],
    reproduced: Mapping[str, float],
    metric_pass: Mapping[str, bool],
    reproduction_pass: bool,
) -> None:
    rows: list[dict[str, Any]] = [
        {
            "record_type": "status",
            "metric": "C1_REPRODUCTION",
            "historical_value": "HISTORICAL",
            "reproduced_value": "REPRODUCED",
            "delta_mm": "",
            "tolerance_mm": METRIC_TOL_MM,
            "pass": reproduction_pass,
        }
    ]
    for metric in ("rmse_mm", "p95_abs_mm", "p99_abs_mm", "max_abs_mm"):
        rows.append(
            {
                "record_type": "metric",
                "metric": metric,
                "historical_value": historical[metric],
                "reproduced_value": reproduced[metric],
                "delta_mm": reproduced[metric] - historical[metric],
                "tolerance_mm": METRIC_TOL_MM,
                "pass": metric_pass[metric],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frozen_predict(original: Any, model: Mapping[str, Any], s: np.ndarray) -> np.ndarray:
    pca_s = model["pca_s"]
    spline = model["spline"]
    values = np.asarray(s, dtype=float)
    domain_min = float(pca_s["domain_min"])
    domain_max = float(pca_s["domain_max"])
    knots = np.asarray(spline["knots"], dtype=float)
    coefficients = np.asarray(spline["coefficients_mm"], dtype=float)
    return original.spline_design(values, knots, domain_min, domain_max) @ coefficients


def write_lut(original: Any, model: Mapping[str, Any], path: Path) -> dict[str, float | int | bool]:
    pca_s = model["pca_s"]
    domain_min = float(pca_s["domain_min"])
    domain_max = float(pca_s["domain_max"])
    grid = np.linspace(domain_min, domain_max, 2049, dtype=float)
    exact = frozen_predict(original, model, grid)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["s", "delta_lambda_mm"])
        writer.writerows((f"{s:.17g}", f"{value:.17g}") for s, value in zip(grid, exact))

    lut = pd.read_csv(path)
    lut_s = lut["s"].to_numpy(float)
    lut_values = lut["delta_lambda_mm"].to_numpy(float)
    grid_error = float(np.max(np.abs(lut_values - exact)))
    dense = np.linspace(domain_min, domain_max, 100001, dtype=float)
    dense_exact = frozen_predict(original, model, dense)
    dense_interp = np.interp(dense, lut_s, lut_values)
    interpolation_error = float(np.max(np.abs(dense_interp - dense_exact)))
    return {
        "point_count": int(len(lut)),
        "grid_roundtrip_max_abs_error_mm": grid_error,
        "linear_interp_dense_max_abs_error_mm": interpolation_error,
        "max_abs_error_mm": max(grid_error, interpolation_error),
        "pass": bool(max(grid_error, interpolation_error) <= LUT_TOL_MM),
    }


def run(args: argparse.Namespace) -> None:
    original = load_original_module()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    selection_manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    audit_summary = json.loads(AUDIT.read_text(encoding="utf-8"))
    historical_stress = pd.read_csv(HISTORICAL_STRESS)

    if selection_manifest.get("C1_OPERATIONAL_MODEL") != CANDIDATE_ID:
        raise RuntimeError("Operational model selection is not C1_4k")
    if selection_manifest.get("operational_pose_count") != 35:
        raise RuntimeError("Operational selection does not declare 35 poses")
    if selection_manifest.get("frame027_status") != "EXCLUDED_OUTSIDE_OPERATIONAL_POSE_DOMAIN":
        raise RuntimeError("frame027 operational status is not the required exclusion label")
    if source_manifest.get("validation_read") is not False:
        raise RuntimeError("Source manifest does not prove Validation exclusion")
    if source_manifest.get("c0_refit") is not False:
        raise RuntimeError("Source manifest does not prove frozen C0")
    if source_manifest.get("production_config_modified") is not False:
        raise RuntimeError("Source manifest indicates production configuration was modified")

    points = original.assert_reuse_contract(POINTS.resolve(), AUDIT.resolve(), FROZEN_C0.resolve())
    points = points.reset_index(drop=True)
    pca = reproduce_full36_pca(points, audit_summary)
    domain_min = float(points["pca_s"].min())
    domain_max = float(points["pca_s"].max())
    expected_domain = source_manifest.get("pca_s_domain")
    if expected_domain is None or not np.allclose([domain_min, domain_max], expected_domain, atol=PCA_TOL, rtol=0.0):
        raise RuntimeError("Full-36 PCA-s domain does not match the original C1 run manifest")

    clean_points = points[points["frame_id"].astype(str).str.zfill(3) != FRAME027].reset_index(drop=True)
    data_027 = points[points["frame_id"].astype(str).str.zfill(3) == FRAME027].reset_index(drop=True)
    operational_ids = sorted(clean_points["frame_id"].astype(str).str.zfill(3).unique().tolist())
    if len(operational_ids) != 35 or FRAME027 in operational_ids or len(clean_points) != 31500:
        raise RuntimeError("Operational-35 clean data is not exactly 35 poses / 31500 points")
    if len(data_027) != 900:
        raise RuntimeError("Expected 900 held-out frame027 points")

    # The only model fit in this script: original protocol, C1_4k, all 35 poses.
    candidate = original.Candidate(CANDIDATE_ID, INTERIOR_KNOT_COUNT)
    fitted = original.fit_robust_spline(clean_points, candidate, domain_min, domain_max)

    pca_payload = dict(pca)
    pca_payload.update({"domain_min": domain_min, "domain_max": domain_max})
    spline_payload = {
        "degree": SPLINE_DEGREE,
        "interior_knot_count": INTERIOR_KNOT_COUNT,
        "basis_count": int(candidate.basis_count),
        "knots": np.asarray(fitted.knots, dtype=float).tolist(),
        "coefficients_mm": np.asarray(fitted.coefficients, dtype=float).tolist(),
    }
    protocol_payload = {
        "target": "residual_centered_mm",
        "target_definition": "residual_mm - frame_residual_median_mm",
        "correction_formula": "c1_residual_mm = residual_mm - F(pca_s)",
        "lambda_formula": "lambda_final = lambda_quadratic + F(pca_s)",
        "spline_basis": "cubic_B_spline",
        "spline_degree": SPLINE_DEGREE,
        "interior_knot_count": INTERIOR_KNOT_COUNT,
        "basis_count": int(candidate.basis_count),
        "smoothness_penalty": float(original.SMOOTHNESS_PENALTY),
        "penalty_order": 2,
        "robust_loss": "Huber_IRLS",
        "robust_huber_k": float(original.HUBER_K),
        "robust_max_iter": int(original.ROBUST_MAX_ITER),
        "robust_tol": float(original.ROBUST_TOL),
        "frame_balanced_weighting": True,
        "extrapolation": "clip_to_pca_s_domain",
        "pca_definition": "Full-36 PCA from xn/yn; PCA includes frame027 and is not recomputed on Operational-35",
        "v_bin_width_px": float(original.V_BIN_WIDTH),
        "v_bin_count": int(original.V_BIN_COUNT),
    }
    fit_payload = {
        "training_pose_ids": operational_ids,
        "training_frame_count": int(fitted.training_frame_count),
        "training_point_count": int(fitted.training_point_count),
        "training_excludes_027": True,
        "robust_scale_mm": float(fitted.robust_scale_mm),
        "robust_iterations": int(fitted.robust_iterations),
    }
    parameter_payload = {
        "pca_s": pca_payload,
        "spline": spline_payload,
        "protocol": protocol_payload,
        "fit": fit_payload,
    }
    parameter_hashes = {
        "pca_sha256": canonical_sha256(pca_payload),
        "spline_sha256": canonical_sha256(spline_payload),
        "protocol_sha256": canonical_sha256(protocol_payload),
        "fit_sha256": canonical_sha256(fit_payload),
        "parameter_sha256": canonical_sha256(parameter_payload),
    }

    source_hashes = {
        "frozen_c0_sha256": sha256_file(FROZEN_C0),
        "residual_artifact_sha256": sha256_file(POINTS),
        "audit_summary_sha256": sha256_file(AUDIT),
        "original_fit_script_sha256": sha256_file(Path(original.__file__).resolve()),
        "historical_stress_sha256": sha256_file(HISTORICAL_STRESS),
        "source_run_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "operational_selection_manifest_sha256": sha256_file(SELECTION_MANIFEST),
        "freeze_script_sha256": sha256_file(SCRIPT_PATH),
    }
    git = git_provenance()

    model_core: dict[str, Any] = {
        "schema_version": 1,
        "model_id": CANDIDATE_ID,
        "operational_model": CANDIDATE_ID,
        "frozen": False,
        "freeze_status": "PENDING_REPRODUCTION",
        "frame027_status": selection_manifest["frame027_status"],
        "frame027_exclusion_reason": selection_manifest["frame027_exclusion_reason"],
        "pca_s": pca_payload,
        "spline": spline_payload,
        "protocol": protocol_payload,
        "fit": fit_payload,
        "parameter_sha256": parameter_hashes["parameter_sha256"],
        "parameter_hashes": parameter_hashes,
        "provenance": {
            "frozen_c0_path": str(FROZEN_C0),
            "frozen_c0_sha256": source_hashes["frozen_c0_sha256"],
            "residual_artifact_path": str(POINTS),
            "residual_artifact_sha256": source_hashes["residual_artifact_sha256"],
            "audit_summary_path": str(AUDIT),
            "audit_summary_sha256": source_hashes["audit_summary_sha256"],
            "original_fit_script_path": str(Path(original.__file__).resolve()),
            "original_fit_script_sha256": source_hashes["original_fit_script_sha256"],
            "source_run_manifest": str(SOURCE_MANIFEST),
            "source_run_manifest_sha256": source_hashes["source_run_manifest_sha256"],
            "operational_selection_manifest": str(SELECTION_MANIFEST),
            "operational_selection_manifest_sha256": source_hashes["operational_selection_manifest_sha256"],
            "git": git,
            "validation_read": False,
            "c0_refit": False,
            "production_config_modified": False,
            "c1_fit_call_count": 1,
            "c1_candidates_fit": [CANDIDATE_ID],
            "model_selection_rerun": False,
        },
    }

    model_path = output / "frozen_c1_4k.json"
    model_path.write_text(json.dumps(clean_json(model_core), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    historical = historical_stress[
        (historical_stress["scenario"] == "exclude027_fullfit_027_heldout")
        & (historical_stress["candidate"] == CANDIDATE_ID)
    ]
    if len(historical) != 1:
        raise RuntimeError("Historical exclude027_fullfit_027_heldout/C1_4k stress row is missing or duplicated")
    historical_row = historical.iloc[0]
    historical_metrics = {
        "rmse_mm": float(historical_row["c1_rmse_mm"]),
        "p95_abs_mm": float(historical_row["c1_p95_abs_mm"]),
        "p99_abs_mm": float(historical_row["c1_p99_abs_mm"]),
        "max_abs_mm": float(historical_row["c1_max_abs_mm"]),
    }
    c0_metrics = metric_row(original, FRAME027, data_027["residual_mm"].to_numpy(float))
    correction = fitted.predict(data_027["pca_s"].to_numpy(float))
    reproduced_metrics = metric_row(
        original,
        FRAME027,
        data_027["residual_mm"].to_numpy(float) - correction,
    )
    metric_pass = {
        metric: abs(reproduced_metrics[metric] - historical_metrics[metric]) <= METRIC_TOL_MM
        for metric in historical_metrics
    }
    reproduction_pass = bool(all(metric_pass.values()))
    reproduction_path = output / "c1_reproduction_check.csv"
    write_reproduction_check(
        reproduction_path,
        historical_metrics,
        reproduced_metrics,
        metric_pass,
        reproduction_pass,
    )

    lut_stats: dict[str, Any] | None = None
    status = "REPRODUCTION_MISMATCH"
    lut_path = output / "c1_4k_lut.csv"
    if reproduction_pass:
        lut_model = json.loads(model_path.read_text(encoding="utf-8"))
        lut_stats = write_lut(original, lut_model, lut_path)
        if lut_stats["pass"]:
            status = "FROZEN_FOR_VALIDATION"
        else:
            lut_path.unlink(missing_ok=True)

    model_final = json.loads(model_path.read_text(encoding="utf-8"))
    model_final["frozen"] = status == "FROZEN_FOR_VALIDATION"
    model_final["freeze_status"] = status
    model_final["stress_reproduction"] = {
        "scenario": "exclude027_fullfit_027_heldout",
        "historical": historical_metrics,
        "reproduced": reproduced_metrics,
        "delta_mm": {metric: reproduced_metrics[metric] - historical_metrics[metric] for metric in historical_metrics},
        "tolerance_mm": METRIC_TOL_MM,
        "C1_REPRODUCTION": "PASS" if reproduction_pass else "MISMATCH",
        "metric_pass": metric_pass,
    }
    model_final["lut_validation"] = lut_stats
    model_final["artifact_sha256"] = None
    model_path.write_text(json.dumps(clean_json(model_final), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_artifact_sha = sha256_file(model_path)

    freeze_manifest = {
        "C1_FREEZE_STATUS": status,
        "C1_REPRODUCTION": "PASS" if reproduction_pass else "MISMATCH",
        "C1_OPERATIONAL_MODEL": CANDIDATE_ID,
        "frozen_model_path": str(model_path),
        "frozen_model_sha256": model_artifact_sha,
        "lut_path": str(lut_path) if lut_path.is_file() else None,
        "lut_sha256": sha256_file(lut_path) if lut_path.is_file() else None,
        "c1_reproduction_check_path": str(reproduction_path),
        "parameter_hashes": parameter_hashes,
        "source_hashes": source_hashes,
        "git": git,
        "pca_s_reproduction": {
            "max_abs_diff_to_stored": pca["stored_pca_s_max_abs_diff"],
            "tolerance": PCA_TOL,
            "pass": pca["stored_pca_s_max_abs_diff"] <= PCA_TOL,
            "pca_includes_027": True,
            "operational_pca_refit": False,
        },
        "operational_pose_ids": operational_ids,
        "operational_pose_count": len(operational_ids),
        "operational_point_count": int(len(clean_points)),
        "frame027_status": selection_manifest["frame027_status"],
        "frame027_exclusion_reason": selection_manifest["frame027_exclusion_reason"],
        "training_excludes_027": True,
        "fit_call_count": 1,
        "fit_candidates": [CANDIDATE_ID],
        "model_selection_rerun": False,
        "validation_read": False,
        "c0_refit": False,
        "production_config_modified": False,
        "historical_stress_row": {
            "path": str(HISTORICAL_STRESS),
            "scenario": "exclude027_fullfit_027_heldout",
            "candidate": CANDIDATE_ID,
        },
        "reproduction": {
            "historical": historical_metrics,
            "reproduced": reproduced_metrics,
            "metric_pass": metric_pass,
            "tolerance_mm": METRIC_TOL_MM,
        },
        "lut_validation": lut_stats,
    }
    manifest_path = output / "c1_freeze_manifest.json"
    manifest_path.write_text(json.dumps(clean_json(freeze_manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Frozen Operational-35 C1_4k",
        "",
        f"C1_FREEZE_STATUS = {status}",
        f"C1_REPRODUCTION = {'PASS' if reproduction_pass else 'MISMATCH'}",
        "",
        "## Scope",
        "",
        "- 仅对既定 Operational-35 执行一次 C1_4k full-fit；未运行 C1_3k/C1_5k、grouped-CV 或模型选择。",
        "- Frozen Full-36 Quadratic C0 未重新拟合。",
        "- frame027 未进入 C1 FIT，只用于 `exclude027_fullfit_027_heldout` reproduction stress check。",
        "- Validation 未读取，生产配置未修改。",
        "",
        "## Provenance / protocol",
        "",
        f"- Operational pose count: `{len(operational_ids)}`；point count: `{len(clean_points)}`。",
        f"- Operational IDs: `{', '.join(operational_ids)}`。",
        f"- frame027: `{selection_manifest['frame027_status']}`；理由：**{selection_manifest['frame027_exclusion_reason']}**。",
        f"- C0 SHA256: `{source_hashes['frozen_c0_sha256']}`。",
        f"- residual artifact SHA256: `{source_hashes['residual_artifact_sha256']}`。",
        f"- original fit script SHA256: `{source_hashes['original_fit_script_sha256']}`。",
        f"- git commit: `{git.get('commit')}`；worktree dirty: `{git.get('worktree_dirty')}`。",
        "- Protocol: cubic B-spline, degree 3, 4 interior knots, frame-balanced weighting, Huber IRLS, second-difference penalty 0.1.",
        "- Target: `residual_centered_mm = residual_mm - frame_residual_median_mm`; correction sign: `residual - F(s)`; application: `lambda_final = lambda_quadratic + F(s)`.",
        "",
        "## PCA-s reproduction",
        "",
        f"- PCA 使用 Full-36 `xn/yn`，包括 027；没有重新拟合 Operational-35 PCA。",
        f"- center: `({pca['center_xn']:.17g}, {pca['center_yn']:.17g})`。",
        f"- axis_s: `({pca['axis_s_xn']:.17g}, {pca['axis_s_yn']:.17g})`。",
        f"- s domain: `[{domain_min:.17g}, {domain_max:.17g}]`。",
        f"- recomputed `pca_s` vs stored `pca_s` max abs diff: `{pca['stored_pca_s_max_abs_diff']:.17g}`; tolerance `{PCA_TOL:g}`。",
        "",
        "## Frozen spline",
        "",
        f"- robust scale: `{fitted.robust_scale_mm:.17g} mm`；robust iterations: `{fitted.robust_iterations}`。",
        f"- training frames/points: `{fitted.training_frame_count}` / `{fitted.training_point_count}`。",
        f"- parameter SHA256: `{parameter_hashes['parameter_sha256']}`。",
        "- 完整 knot vector、coefficients、PCA 参数和 protocol 已写入 `frozen_c1_4k.json`。",
        "",
        "## frame027 reproduction",
        "",
        "| metric | historical | reproduced | delta mm | tolerance mm | pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in ("rmse_mm", "p95_abs_mm", "p99_abs_mm", "max_abs_mm"):
        report_lines.append(
            f"| {metric} | {historical_metrics[metric]:.17g} | {reproduced_metrics[metric]:.17g} | {reproduced_metrics[metric] - historical_metrics[metric]:.3g} | {METRIC_TOL_MM:g} | {'PASS' if metric_pass[metric] else 'FAIL'} |"
        )
    report_lines.extend(
        [
            "",
            f"`C1_REPRODUCTION = {'PASS' if reproduction_pass else 'MISMATCH'}`。",
            "",
            "## LUT validation",
            "",
        ]
    )
    if lut_stats is None:
        report_lines.append("由于 reproduction 未通过，LUT 未生成。")
    else:
        report_lines.extend(
            [
                f"- points: `{lut_stats['point_count']}`。",
                f"- grid round-trip max error: `{lut_stats['grid_roundtrip_max_abs_error_mm']:.17g} mm`。",
                f"- dense linear-interpolation max error: `{lut_stats['linear_interp_dense_max_abs_error_mm']:.17g} mm`。",
                f"- tolerance: `{LUT_TOL_MM:g} mm`；result: `{'PASS' if lut_stats['pass'] else 'FAIL'}`。",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `frozen_c1_4k.json`：{model_path}",
            f"- `c1_4k_lut.csv`：{lut_path if lut_path.is_file() else '未生成'}",
            f"- `c1_reproduction_check.csv`：{reproduction_path}",
            f"- `c1_freeze_manifest.json`：{manifest_path}",
            "- `c1_freeze_report.md`：本报告。",
            "",
        ]
    )
    (output / "c1_freeze_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "C1_FREEZE_STATUS": status,
                "C1_REPRODUCTION": "PASS" if reproduction_pass else "MISMATCH",
                "model_path": str(model_path),
                "lut_path": str(lut_path) if lut_path.is_file() else None,
                "fit_call_count": 1,
                "validation_read": False,
                "c0_refit": False,
                "production_config_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
