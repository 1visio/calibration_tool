#!/usr/bin/env python3
"""Task 3B-1: FIT-only equivalent local Circular Cone parameterization audit.

This script performs only coordinate conversion and equivalence checks.  It
does not call a nonlinear optimizer and never loads validation images or
validation point records.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
CALIBRATION_TOOL_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
MEASUREMENT_ROOT = WORKSPACE_ROOT / "linelaser_tool" / "laser_measurement_tool"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(MEASUREMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MEASUREMENT_ROOT))

import circular_cone_local_parameterization as local  # noqa: E402
import diagnose_circular_cone_identifiability_task3a as task3a  # noqa: E402
from reconstruction.reconstructor import reconstruct_uv_to_ground  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    CALIBRATION_TOOL_ROOT
    / "projects"
    / "daheng"
    / "outputs"
    / "0814"
    / "cone_local_parameterization"
)
MEASUREMENT_CONFIG = MEASUREMENT_ROOT / "configs" / "measure_tool_daheng_0811.yaml"
FORMAL_CONE = MEASUREMENT_ROOT / "configs" / "calibration_daheng_0811" / "circular_cone.yaml"
TASK3A_OUTPUT = CALIBRATION_TOOL_ROOT / "projects" / "daheng" / "outputs" / "0814" / "cone_identifiability_audit"
EXPECTED_CONE_SHA256 = "478d11c97c174e75d3167133a050540573a93dc28451e4b961ce12913709feac"
LAMBDA_TOL_MM = 1.0e-6
OBJECTIVE_TOL = 1.0e-10
AXIS_TOL = 1.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_reference_anchor(records: Sequence[task3a.FrameRecord]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Equal-weight the 30 per-frame ray-plane truth centroids."""
    centroids: list[dict[str, Any]] = []
    values: list[np.ndarray] = []
    for record in records:
        centroid = np.mean(record.truth_points, axis=0)
        values.append(centroid)
        centroids.append(
            {
                "frame_id": record.frame_id,
                "split": record.split,
                "source": record.source,
                "point_count": record.point_count,
                "centroid_camera_mm": centroid.tolist(),
                "pnp_rmse_px": record.pnp_rmse_px,
            }
        )
    if len(values) != len(task3a.FIT_IDS):
        raise RuntimeError("reference anchor must use all 30 explicit FIT frames")
    return np.mean(np.stack(values, axis=0), axis=0), centroids


def lambda_by_input(
    pixels_uv: np.ndarray,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Call the public production reconstruction and restore input alignment."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = reconstruct_uv_to_ground(pixels_uv, calibration, reconstruction_params)
    values = np.full(len(pixels_uv), np.nan, dtype=np.float64)
    valid = np.zeros(len(pixels_uv), dtype=bool)
    lookup = {tuple(np.round(uv, 10)): index for index, uv in enumerate(pixels_uv)}
    for uv, point in zip(result.pixels_uv, result.points_camera):
        index = lookup.get(tuple(np.round(uv, 10)))
        if index is not None:
            values[index] = float(point[2])
            valid[index] = True
    return values, valid, dict(result.filtered)


def model_for_theta(theta: np.ndarray, base_model: Mapping[str, Any]) -> dict[str, Any]:
    z_range = base_model.get("z_valid_range_mm")
    return local.theta_to_model(theta, z_range)


def calibration_with_model(calibration: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(calibration))
    result["laser_model"] = copy.deepcopy(dict(model))
    return result


def cone_axis(theta: np.ndarray) -> np.ndarray:
    return local.angles_to_axis(float(theta[0]), float(theta[1]))


def evaluate_model(
    name: str,
    theta: np.ndarray,
    base_model: Mapping[str, Any],
    records: Sequence[task3a.FrameRecord],
    points: np.ndarray,
    sqrt_weights: np.ndarray,
    cfg: Mapping[str, Any],
    p_ref: np.ndarray,
    grid_uv: np.ndarray,
    calibration: Mapping[str, Any],
    reconstruction_params: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local_theta = local.legacy_to_local(theta, p_ref)
    roundtrip_theta = local.local_to_legacy(local_theta, p_ref)
    legacy_model = model_for_theta(theta, base_model)
    roundtrip_model = model_for_theta(roundtrip_theta, base_model)

    objective_legacy = task3a.cone_fit_vector(theta, points, sqrt_weights, cfg)
    objective_roundtrip = task3a.cone_fit_vector(roundtrip_theta, points, sqrt_weights, cfg)
    scalar_legacy = np.concatenate([task3a.cone_scalar_residual(theta, record.truth_points) for record in records])
    scalar_roundtrip = np.concatenate([task3a.cone_scalar_residual(roundtrip_theta, record.truth_points) for record in records])

    legacy_calibration = calibration_with_model(calibration, legacy_model)
    roundtrip_calibration = calibration_with_model(calibration, roundtrip_model)
    lambda_legacy, valid_legacy, filtered_legacy = lambda_by_input(grid_uv, legacy_calibration, reconstruction_params)
    lambda_roundtrip, valid_roundtrip, filtered_roundtrip = lambda_by_input(grid_uv, roundtrip_calibration, reconstruction_params)
    common = valid_legacy & valid_roundtrip
    if np.any(common):
        lambda_diff = np.abs(lambda_legacy[common] - lambda_roundtrip[common])
        max_lambda_diff = float(np.max(lambda_diff))
        p95_lambda_diff = float(np.percentile(lambda_diff, 95))
        mean_lambda_diff = float(np.mean(lambda_diff))
    else:
        max_lambda_diff = p95_lambda_diff = mean_lambda_diff = float("nan")

    apex = theta[2:5]
    axis = cone_axis(theta)
    rt_axis = cone_axis(roundtrip_theta)
    axial_legacy = np.concatenate([(record.truth_points - apex) @ axis for record in records])
    rt_apex = roundtrip_theta[2:5]
    axial_roundtrip = np.concatenate([(record.truth_points - rt_apex) @ rt_axis for record in records])
    theta_diff = np.abs(theta - roundtrip_theta)
    objective_diff = np.abs(objective_legacy - objective_roundtrip)
    scalar_diff = np.abs(scalar_legacy - scalar_roundtrip)

    row = {
        "model": name,
        "legacy_theta_axis": theta[0],
        "legacy_phi_axis": theta[1],
        "legacy_A_x_mm": theta[2],
        "legacy_A_y_mm": theta[3],
        "legacy_A_z_mm": theta[4],
        "legacy_alpha_rad": theta[5],
        "local_theta_axis": local_theta[0],
        "local_phi_axis": local_theta[1],
        "local_c1_mm": local_theta[2],
        "local_c2_mm": local_theta[3],
        "local_rho_ref_mm": local_theta[4],
        "local_q_cot_alpha": local_theta[5],
        "roundtrip_theta_max_abs_diff": float(np.max(theta_diff)),
        "axis_dot_legacy_roundtrip": float(axis @ rt_axis),
        "axis_norm_legacy": float(np.linalg.norm(axis)),
        "axis_norm_roundtrip": float(np.linalg.norm(rt_axis)),
        "median_axial_legacy_mm": float(np.median(axial_legacy)),
        "median_axial_roundtrip_mm": float(np.median(axial_roundtrip)),
        "nappe_sign_preserved": bool(np.sign(np.median(axial_legacy)) == np.sign(np.median(axial_roundtrip))),
        "objective_cost_legacy": 0.5 * float(np.sum(objective_legacy**2)),
        "objective_cost_roundtrip": 0.5 * float(np.sum(objective_roundtrip**2)),
        "max_objective_vector_abs_diff": float(np.max(objective_diff)),
        "max_scalar_residual_abs_diff_mm": float(np.max(scalar_diff)),
        "lambda_legacy_valid_count": int(np.count_nonzero(valid_legacy)),
        "lambda_roundtrip_valid_count": int(np.count_nonzero(valid_roundtrip)),
        "lambda_common_valid_count": int(np.count_nonzero(common)),
        "lambda_max_abs_diff_mm": max_lambda_diff,
        "lambda_p95_abs_diff_mm": p95_lambda_diff,
        "lambda_mean_abs_diff_mm": mean_lambda_diff,
        "filtered_legacy": json.dumps(filtered_legacy, sort_keys=True),
        "filtered_roundtrip": json.dumps(filtered_roundtrip, sort_keys=True),
    }
    detail = {
        "name": name,
        "legacy_theta": theta.tolist(),
        "local_theta": local_theta.tolist(),
        "roundtrip_theta": roundtrip_theta.tolist(),
        "legacy_model": legacy_model,
        "roundtrip_model": roundtrip_model,
        "row": row,
    }
    return row, detail


def render_definition(p_ref: np.ndarray, anchor_count: int) -> str:
    return f"""# Circular Cone 等价局部参数化

## 正式 legacy 模型

正式模型仍为

`Theta_legacy = [theta_axis, phi_axis, A_x, A_y, A_z, alpha]`

`d = spherical(theta_axis, phi_axis)` 为单位轴方向，`A` 为 apex，`alpha` 为 half-apex angle。圆锥方程保持正式实现：

`||(X-A) - ((X-A)·d)d|| / tan(alpha) - (X-A)·d = 0`

求交仍使用正式的 ray-cone quadratic intersection、正深度/工作距离筛选和 `axial >= 0` 的物理 nappe 选择；本参数化没有重新定义模型。

## 固定参考锚点

`P_ref = {p_ref.tolist()} mm`，由 `{anchor_count}` 个 FIT frame 的 ray-plane truth 逐 frame 计算 3D centroid，再对 30 个 centroid 等权平均。Validation 不参与锚点计算。

对每个轴 `d`，用确定性的相机坐标基构造 `e1,e2`：优先投影 camera-Z 到 `d` 的正交平面；若近似平行则使用 camera-Y；`e2 = d × e1`。

令 `C_ref` 为 cone axis 与过 `P_ref` 的法向截面相交点，定义：

`C_ref = P_ref + c1*e1 + c2*e2`

`s_ref = (P_ref-A)·d`

`rho_ref = s_ref*tan(alpha)`

`q = cot(alpha) = 1/tan(alpha)`

因此局部向量为：

`Theta_local = [theta_axis, phi_axis, c1, c2, rho_ref, q]`

## 严格双向转换

`legacy_to_local`：由 `A,d,alpha,P_ref` 计算 `C_ref,s_ref,rho_ref,q`。

`local_to_legacy`：

`s_ref = rho_ref*q`

`A = P_ref + c1*e1 + c2*e2 - s_ref*d`

`alpha = atan2(1,q)`

这个映射在 `0 < alpha < 90°`、`q > 0` 下是严格可逆的。轴向量不取反，因而 nappe convention 不改变。

## 为什么是局部坐标

原始 apex 位于观测区域之外，且 `alpha≈90°` 时 apex displacement 与 alpha 变化会形成很长的弱谷。新坐标把几何量改写为观测区域附近的 axis-line 横向位置、参考截面半径和局部斜率；它只改变坐标，不保证条件数已经改善。条件数和 Full-FIT 稳定性留给 Task 3B-2。
"""


def render_report(
    rows: Sequence[Mapping[str, Any]],
    objective: Mapping[str, Any],
    p_ref: np.ndarray,
    anchor_count: int,
    cone_hash_before: str,
    cone_hash_after: str,
    output_dir: Path,
) -> str:
    passed = bool(objective["equivalence_pass"])
    max_lambda = max(float(row["lambda_max_abs_diff_mm"]) for row in rows)
    max_theta = max(float(row["roundtrip_theta_max_abs_diff"]) for row in rows)
    lines = [
        "# Task 3B-1 — Circular Cone 等价局部参数化与验证",
        "",
        "**VALIDATION_OPENED = FALSE**",
        "**PRODUCTION_CONE_MODIFIED = FALSE**",
        f"**LOCAL_PARAMETERIZATION_EQUIVALENCE = {'PASS' if passed else 'FAIL'}**",
        "",
        "## 数据与冻结项",
        "",
        "- FIT-only: 001–018 + 025–036，共 30 frame；没有读取 019–024、037–040 的图像、点或 residual。",
        f"- Formal working domain: v=[{task3a.FORMAL_V_MIN:.3f}, {task3a.FORMAL_V_MAX:.3f}] px；evaluation grid 与 Task 3A 相同。",
        f"- Formal Cone SHA-256 before: `{cone_hash_before}`；after: `{cone_hash_after}`。",
        f"- P_ref 由 {anchor_count} 个 FIT frame centroid 等权得到：`{p_ref.tolist()}` mm。",
        "",
        "## 1. 数学定义",
        "",
        "详见 `local_parameterization_definition.md`。新参数为 `[theta_axis, phi_axis, c1, c2, rho_ref, q]`，其中 `q=cot(alpha)`，并通过固定 P_ref 的 axis-normal 截面严格恢复 legacy apex 与 alpha。",
        "",
        "## 2. 等价性结果",
        "",
        "| model | max theta roundtrip error | axis dot | nappe preserved | max objective-vector diff | max lambda diff / mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {float(row['roundtrip_theta_max_abs_diff']):.3e} | {float(row['axis_dot_legacy_roundtrip']):.15f} | {row['nappe_sign_preserved']} | {float(row['max_objective_vector_abs_diff']):.3e} | {float(row['lambda_max_abs_diff_mm']):.3e} |"
        )
    lines += [
        "",
        f"- Across M0 and M_diag_fullfit, maximum legacy→local→legacy parameter error: `{max_theta:.3e}` (native units).",
        f"- Across both models, maximum evaluation-grid `|lambda_legacy-lambda_roundtrip|`: `{max_lambda:.3e}` mm; required threshold is `1.0e-6` mm.",
        f"- Objective cost and residual vector equivalence: `{'PASS' if objective['objective_pass'] else 'FAIL'}`; ray intersection equivalence: `{'PASS' if objective['lambda_pass'] else 'FAIL'}`; axis/nappe equivalence: `{'PASS' if objective['axis_nappe_pass'] else 'FAIL'}`.",
        "",
        "## 3. Interpretation",
        "",
        "- The local coordinates are a coordinate change only. They do not add a prior, regularization, v-dependent term, polynomial correction or residual compensation.",
        "- The local coordinates are closer to the finite observed patch because the apex is represented through a nearby axis-line cross-section and local slope. This is a parameterization hypothesis; improved conditioning must be tested separately.",
        "- The top-edge residual is deliberately not analyzed or corrected in this task.",
        "",
        "## 4. Next step",
        "",
        f"- Task 3B-2 local-parameter Full-FIT + SVD + jackknife may proceed: `{'YES' if passed else 'NO — stop and fix equivalence first'}`.",
        "- No diagnostic local parameter vector is written to the production Cone file.",
        "",
        f"Outputs are under `{output_dir}`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 3B-1 FIT-only local Circular Cone equivalence audit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--measurement-config", type=Path, default=MEASUREMENT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    cone_hash_before = sha256_file(FORMAL_CONE)
    if cone_hash_before != EXPECTED_CONE_SHA256:
        raise RuntimeError(f"Formal Cone hash mismatch: {cone_hash_before}")

    # Explicitly load only FIT records.  No validation loader is called.
    _, calibration, reconstruction_params, intrinsics = task3a.load_runtime(args.measurement_config.resolve())
    records = task3a.load_old_records()
    extension_records, extension_provenance = task3a.load_extension_records(intrinsics)
    records += extension_records
    if [record.frame_id for record in records] != task3a.FIT_IDS:
        raise RuntimeError("FIT records do not match explicit 001-018 + 025-036 registry")
    p_ref, frame_centroids = build_reference_anchor(records)

    m0_model = copy.deepcopy(calibration["laser_model"])
    fullfit_path = TASK3A_OUTPUT / "fullfit_diagnostic_result.json"
    if not fullfit_path.is_file():
        raise FileNotFoundError(f"Task 3A diagnostic model missing: {fullfit_path}")
    fullfit_json = json.loads(fullfit_path.read_text(encoding="utf-8"))
    diag_model = copy.deepcopy(fullfit_json["model"])
    m0_theta = local.legacy_model_to_theta(m0_model)
    diag_theta = np.asarray(fullfit_json["theta"], dtype=np.float64).reshape(6)

    cfg_root = task3a.triplets.safe_yaml_load(task3a.FORMAL_FIT_CONFIG)
    cone_cfg = dict(cfg_root["models"]["cone"])
    selected_points, selected_frames, _ = task3a.select_formal_points(records, int(cone_cfg.get("fit_max_points", 3000)))
    sqrt_weights = np.sqrt(task3a.frame_equal_weights(selected_frames))
    grid_uv, u_values, v_values = task3a.build_grid(records)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for name, theta, model in (("M0", m0_theta, m0_model), ("M_diag_fullfit", diag_theta, diag_model)):
        row, detail = evaluate_model(
            name,
            theta,
            model,
            records,
            selected_points,
            sqrt_weights,
            cone_cfg,
            p_ref,
            grid_uv,
            calibration,
            reconstruction_params,
        )
        rows.append(row)
        details[name] = detail

    objective = {
        "task": "Task 3B-1",
        "validation_opened": False,
        "fit_frame_ids": task3a.FIT_IDS,
        "validation_frame_ids": task3a.VALIDATION_IDS,
        "formal_working_domain_v_px": [task3a.FORMAL_V_MIN, task3a.FORMAL_V_MAX],
        "evaluation_grid": {
            "u_min_px": float(np.min(u_values)),
            "u_max_px": float(np.max(u_values)),
            "u_count": len(u_values),
            "v_min_px": float(np.min(v_values)),
            "v_max_px": float(np.max(v_values)),
            "v_count": len(v_values),
            "point_count": len(grid_uv),
        },
        "selected_objective_point_count": len(selected_points),
        "models": details,
        "objective_pass": all(float(row["max_objective_vector_abs_diff"]) <= OBJECTIVE_TOL for row in rows),
        "lambda_pass": all(float(row["lambda_max_abs_diff_mm"]) <= LAMBDA_TOL_MM for row in rows),
        "axis_nappe_pass": all(
            float(row["axis_dot_legacy_roundtrip"]) >= 1.0 - AXIS_TOL
            and bool(row["nappe_sign_preserved"])
            for row in rows
        ),
    }
    objective["equivalence_pass"] = bool(objective["objective_pass"] and objective["lambda_pass"] and objective["axis_nappe_pass"])

    reference_payload = {
        "p_ref_camera_mm": p_ref.tolist(),
        "definition": "equal-weight mean of per-frame independent ray-plane truth centroids",
        "fit_frame_count": len(records),
        "fit_frame_ids": task3a.FIT_IDS,
        "validation_used": False,
        "frame_centroids": frame_centroids,
        "extension_provenance_count": len(extension_provenance),
        "formal_working_domain_v_px": [task3a.FORMAL_V_MIN, task3a.FORMAL_V_MAX],
    }
    provenance = {
        "task": "Task 3B-1 Circular Cone equivalent local parameterization",
        "formal_cone_path": str(FORMAL_CONE),
        "formal_cone_sha256_before": cone_hash_before,
        "formal_cone_sha256_after": sha256_file(FORMAL_CONE),
        "measurement_config": str(args.measurement_config.resolve()),
        "task3a_diagnostic_input": str(fullfit_path),
        "validation_opened": False,
        "production_writeback": False,
        "optimizer_called": False,
        "reference_anchor": reference_payload,
    }

    (out / "reference_anchor.json").write_text(json.dumps(reference_payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    (out / "objective_equivalence.json").write_text(json.dumps(objective, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    write_csv(out / "roundtrip_equivalence.csv", rows)
    (out / "local_parameterization_definition.md").write_text(render_definition(p_ref, len(records)), encoding="utf-8")
    (out / "report.md").write_text(render_report(rows, objective, p_ref, len(records), cone_hash_before, provenance["formal_cone_sha256_after"], out), encoding="utf-8")

    # Make the exact conversion implementation and its unit test discoverable
    # alongside the audit outputs, without making either a production artifact.
    shutil.copy2(SCRIPT_PATH.parent / "circular_cone_local_parameterization.py", out / "circular_cone_local_parameterization.py")
    shutil.copy2(CALIBRATION_TOOL_ROOT / "tests" / "test_circular_cone_local_parameterization.py", out / "test_circular_cone_local_parameterization.py")
    (out / "OUTPUT_FILES.md").write_text(
        """# Task 3B-1 output files

| file | meaning | conclusion boundary |
|---|---|---|
| report.md | final equivalence result and next-step gate | no top-edge conclusion, no deployment authorization |
| local_parameterization_definition.md | exact local coordinates and conversion equations | coordinate change only |
| reference_anchor.json | FIT-only P_ref and per-frame centroid provenance | validation not used |
| roundtrip_equivalence.csv | M0/M_diag legacy→local→legacy and grid lambda checks | no optimization |
| objective_equivalence.json | objective, ray-intersection, axis and nappe equivalence | same physical cone only |
| provenance.json | hash, split isolation and no-optimizer provenance | no validation model selection |
| circular_cone_local_parameterization.py | conversion implementation | not production runtime |
| test_circular_cone_local_parameterization.py | unit tests for inverse mapping/basis/axis sign | does not test Full-FIT |
""",
        encoding="utf-8",
    )

    cone_hash_after = sha256_file(FORMAL_CONE)
    if cone_hash_after != cone_hash_before:
        raise RuntimeError("Formal Cone changed during Task 3B-1")
    print(f"LOCAL_PARAMETERIZATION_EQUIVALENCE={'PASS' if objective['equivalence_pass'] else 'FAIL'}")
    print(f"MAX_LAMBDA_ROUNDTRIP_ERROR_MM={max(float(row['lambda_max_abs_diff_mm']) for row in rows):.9g}")
    print(f"VALIDATION_OPENED={objective['validation_opened']}")
    print(f"OUTPUT={out}")
    return 0 if objective["equivalence_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

