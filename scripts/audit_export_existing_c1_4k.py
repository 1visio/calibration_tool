#!/usr/bin/env python3
"""Audit whether an existing operational-35 C1_4k full-fit is exportable.

This script is intentionally audit-only.  It does not import the C1 fitting
module and must never call fit().  If the full-fit parameters are not already
persisted, it writes a manifest/report with the blocked export status and does
not create a model or LUT placeholder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
PROJECT = ROOT / "projects" / "daheng"
OUTPUT_DEFAULT = PROJECT / "outputs/0818/c1_4k_freeze"

SOURCE_DIR = PROJECT / "outputs/0818/c1_frozen_quadratic_grouped_cv"
SELECTION_DIR = PROJECT / "outputs/0818/c1_operational35_selection"
SOURCE_MANIFEST = SOURCE_DIR / "c1_run_manifest.json"
SELECTION_MANIFEST = SELECTION_DIR / "c1_operational35_selection_manifest.json"
SOURCE_POSE = SOURCE_DIR / "c1_pose_cv_metrics.csv"
SOURCE_V_BINS = SOURCE_DIR / "c1_v_bin_metrics.csv"
SOURCE_STRESS = SOURCE_DIR / "frame027_stress_test.csv"
SOURCE_COMPARISON = SOURCE_DIR / "c1_candidate_comparison.csv"
SOURCE_REPORT = SOURCE_DIR / "report.md"
SOURCE_CURVES = SOURCE_DIR / "c1_curves_with_without_027.png"

SCENARIO = "exclude027_fullfit_027_heldout"
CANDIDATE = "C1_4k"
FRAME027 = "027"
STATUS = "FULLFIT_PARAMETERS_NOT_PERSISTED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def zfill_frame(value: Any) -> str:
    return str(value).strip().zfill(3)


def required_paths() -> tuple[Path, ...]:
    return (
        SOURCE_MANIFEST,
        SELECTION_MANIFEST,
        SOURCE_POSE,
        SOURCE_V_BINS,
        SOURCE_STRESS,
        SOURCE_COMPARISON,
        SOURCE_REPORT,
        SOURCE_CURVES,
    )


def check_hash(path: Path, expected: str | None) -> dict[str, Any]:
    actual = sha256_file(path)
    return {
        "path": str(path),
        "sha256": actual,
        "expected_sha256": expected,
        "matches_expected": expected is None or actual == expected,
    }


def json_keys_recursive(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.add(name)
            keys.update(json_keys_recursive(item, name))
    elif isinstance(value, list):
        for item in value:
            keys.update(json_keys_recursive(item, prefix))
    return keys


def collect_parameter_candidates() -> dict[str, Any]:
    """Inspect only the existing grouped-CV output directory for parameters."""
    candidates: list[str] = []
    json_key_inventory: dict[str, list[str]] = {}
    parameter_tokens = ("knot", "coeff", "center_xn", "center_yn", "axis_s")
    token_hits: dict[str, list[str]] = {token: [] for token in parameter_tokens}
    for path in sorted(SOURCE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".yaml", ".yml", ".npz", ".pkl", ".pickle", ".joblib"}:
            candidates.append(str(path))
            if path.suffix.lower() == ".json":
                try:
                    keys = sorted(json_keys_recursive(read_json(path)))
                except (OSError, json.JSONDecodeError):
                    keys = []
                json_key_inventory[str(path)] = keys
        if path.suffix.lower() in {".json", ".yaml", ".yml", ".csv", ".md"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lowered = text.lower()
            for token in parameter_tokens:
                if token.lower() in lowered:
                    token_hits[token].append(str(path))
    return {
        "parameter_file_candidates_in_source_dir": candidates,
        "json_key_inventory": json_key_inventory,
        "parameter_token_hits": token_hits,
    }


def audit(output: Path) -> dict[str, Any]:
    missing = [str(path) for path in required_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source artifacts: " + ", ".join(missing))

    source_manifest = read_json(SOURCE_MANIFEST)
    selection_manifest = read_json(SELECTION_MANIFEST)
    pose_rows = read_csv(SOURCE_POSE)
    stress_rows = read_csv(SOURCE_STRESS)
    v_rows = read_csv(SOURCE_V_BINS)
    comparison_rows = read_csv(SOURCE_COMPARISON)

    operational_pose_rows = [
        row
        for row in pose_rows
        if row.get("scenario") == "exclude027_grouped_cv_non027"
        and row.get("candidate") == CANDIDATE
        and row.get("model") == "C0+C1"
    ]
    operational_ids = sorted({zfill_frame(row["heldout_frame_id"]) for row in operational_pose_rows})
    if len(operational_ids) != 35 or FRAME027 in operational_ids:
        raise RuntimeError(f"Operational pose set is not exactly 35 poses without 027: {operational_ids}")

    fullfit_stress_rows = [
        row
        for row in stress_rows
        if row.get("scenario") == SCENARIO and row.get("candidate") == CANDIDATE
    ]
    operational_v_rows = [
        row
        for row in v_rows
        if row.get("scenario") == SCENARIO and row.get("candidate") == CANDIDATE
    ]
    operational_comparison_rows = [
        row
        for row in comparison_rows
        if row.get("scenario") == SCENARIO and row.get("candidate") == CANDIDATE
    ]

    points_path = Path(source_manifest["points_artifact"])
    frozen_model_path = Path(source_manifest["frozen_model"])
    source_hashes = {
        "frozen_quadratic_c0": check_hash(
            frozen_model_path, source_manifest.get("frozen_model_sha256")
        ),
        "full36_residual_artifact": check_hash(
            points_path, source_manifest.get("points_sha256")
        ),
        "c1_run_manifest": check_hash(SOURCE_MANIFEST, None),
        "c1_candidate_comparison": check_hash(SOURCE_COMPARISON, None),
        "c1_pose_cv_metrics": check_hash(SOURCE_POSE, None),
        "c1_v_bin_metrics": check_hash(SOURCE_V_BINS, None),
        "frame027_stress_test": check_hash(SOURCE_STRESS, None),
        "c1_operational_selection_manifest": check_hash(SELECTION_MANIFEST, None),
    }

    parameter_presence = {
        "pca_center_axis_s": {
            "status": "MISSING",
            "evidence": "No machine-readable center_xn/center_yn/axis_s_xn/axis_s_yn in the existing 0818 C1 artifacts.",
        },
        "s_domain": {
            "status": "PRESENT",
            "value": source_manifest.get("pca_s_domain"),
            "evidence": f"{SOURCE_MANIFEST}:pca_s_domain",
        },
        "spline_degree": {
            "status": "MISSING",
            "evidence": "Cubic basis is described by source code/protocol, but no degree field is persisted with the full-fit model.",
        },
        "knot_vector": {
            "status": "MISSING",
            "evidence": "No serialized knots array in the existing grouped-CV output directory.",
        },
        "spline_coefficients": {
            "status": "MISSING",
            "evidence": "No serialized coefficients array in the existing grouped-CV output directory.",
        },
        "penalty_and_fitting_protocol": {
            "status": "PARTIAL",
            "value": {
                "smoothness_penalty": source_manifest.get("smoothness_penalty"),
                "robust_loss": source_manifest.get("robust_loss"),
                "frame_balanced_weighting": source_manifest.get("frame_balanced_weighting"),
                "grouped_cv_folds": source_manifest.get("grouped_cv_folds"),
            },
            "evidence": f"{SOURCE_MANIFEST}; full-fit robust scale/iterations and parameter arrays are absent.",
        },
    }

    parameter_complete = all(
        item["status"] == "PRESENT" for item in parameter_presence.values()
    )
    if parameter_complete:
        raise RuntimeError("Unexpected complete parameter set; audit-only exporter must be reviewed before export")

    output.mkdir(parents=True, exist_ok=True)
    exported_model = output / "frozen_c1_4k.json"
    exported_lut = output / "c1_4k_lut.csv"
    if exported_model.exists() or exported_lut.exists():
        raise RuntimeError("Refusing to overwrite a model/LUT placeholder or prior export")

    inventory = collect_parameter_candidates()
    freeze_manifest: dict[str, Any] = {
        "C1_EXPORT_STATUS": STATUS,
        "C1_OPERATIONAL_MODEL": CANDIDATE,
        "exported_model": None,
        "exported_lut": None,
        "operational_pose_ids": operational_ids,
        "operational_pose_count": len(operational_ids),
        "frame027_status": selection_manifest.get("frame027_status"),
        "frame027_exclusion_reason": selection_manifest.get("frame027_exclusion_reason"),
        "fullfit_scenario": SCENARIO,
        "fullfit_stress_rows_found": len(fullfit_stress_rows),
        "operational_v_rows_found": len(operational_v_rows),
        "operational_comparison_rows_found": len(operational_comparison_rows),
        "parameter_complete": False,
        "parameter_presence": parameter_presence,
        "parameter_sha256": None,
        "lut_sha256": None,
        "lut_max_abs_error_mm": None,
        "source_grouped_cv_dir": str(SOURCE_DIR),
        "source_manifest": str(SOURCE_MANIFEST),
        "selection_manifest": str(SELECTION_MANIFEST),
        "source_hashes": source_hashes,
        "source_parameter_inventory": inventory,
        "protocol_assertions": {
            "fit_called": False,
            "c1_refit": False,
            "c0_refit": False,
            "validation_read": False,
            "production_config_modified": False,
            "pca_recomputed": False,
        },
        "blocked_reason": "Existing exclude027 full-fit C1_4k object was in-memory only; required model parameters were not persisted.",
    }
    manifest_path = output / "c1_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(freeze_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# C1_4k frozen export audit",
        "",
        f"C1_EXPORT_STATUS = {STATUS}",
        "",
        "## Decision",
        "",
        "未导出 `frozen_c1_4k.json` 和 `c1_4k_lut.csv`。已有 Operational-35 full-fit C1_4k 的完整参数没有持久化；按约束停止，不调用 `fit()`，不自动重拟合。",
        "",
        "## Provenance / reuse audit",
        "",
        f"- 目标 full-fit 场景：`{SCENARIO}`；模型：`{CANDIDATE}`。",
        f"- 该场景使用 35 个训练 pose（{len(operational_ids)} 个 operational IDs），027 仅作为 held-out stress pose。",
        f"- Operational IDs：`{', '.join(operational_ids)}`。",
        f"- frame027：`{selection_manifest.get('frame027_status')}`；理由：**{selection_manifest.get('frame027_exclusion_reason')}**。原始 artifact 保留。",
        f"- Frozen Quadratic C0：`{frozen_model_path}`；SHA256 `{source_hashes['frozen_quadratic_c0']['sha256']}`。",
        f"- Full-36 residual artifact：`{points_path}`；SHA256 `{source_hashes['full36_residual_artifact']['sha256']}`。",
        f"- Existing grouped-CV artifacts：`{SOURCE_DIR}`；仅复用和审计，没有重新拟合。",
        "- Validation：未读取。生产配置：未修改。",
        "",
        "## Full-fit parameter completeness",
        "",
        "| parameter | status | evidence |",
        "|---|---|---|",
    ]
    for name, item in parameter_presence.items():
        lines.append(f"| `{name}` | **{item['status']}** | {item['evidence']} |")
    lines.extend(
        [
            "",
            "## Evidence of non-persistence",
            "",
            "- `fit_c1_frozen_quadratic_grouped_cv.py:838-841` 在进程内构造 `without_027_models['C1_4k']`。",
            "- `fit_c1_frozen_quadratic_grouped_cv.py:244-287` 的 `SplineFit` 内存对象包含 knots/coefficients，但没有写入输出。",
            "- `fit_c1_frozen_quadratic_grouped_cv.py:998-1004` 只对 027 执行 `.predict()`，保存的是 stress metrics。",
            "- 现有 `c1_run_manifest.json` 仅保存 `pca_s_domain`、协议和 metrics 元数据；没有 PCA center/axis、knot vector 或 coefficients。",
            "- 现有 CSV/PNG 是评估结果，不是可加载的 full-fit 模型。",
            "",
            "## Output state",
            "",
            "- `frozen_c1_4k.json`：**未生成**；没有完整参数可冻结。",
            "- `c1_4k_lut.csv`：**未生成**；没有精确 spline 可用于 LUT 对照，因此没有伪造误差验证。",
            "- `c1_freeze_manifest.json`：本审计结果及 provenance。",
            "- `c1_freeze_report.md`：本报告。",
            "",
            "## Required next action",
            "",
            "需要在下一次允许的 C1 fit/export 运行中显式持久化 `SplineFit` 的 PCA center/axis、完整 knot vector、coefficients、degree、domain、robust scale/iterations 及 protocol；本轮不自动执行该动作。",
            "",
        ]
    )
    (output / "c1_freeze_report.md").write_text("\n".join(lines), encoding="utf-8")
    return freeze_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = audit(args.output_dir.resolve())
    print(
        json.dumps(
            {
                "C1_EXPORT_STATUS": manifest["C1_EXPORT_STATUS"],
                "output_dir": str(args.output_dir.resolve()),
                "model_written": manifest["exported_model"] is not None,
                "lut_written": manifest["exported_lut"] is not None,
                "fit_called": manifest["protocol_assertions"]["fit_called"],
                "validation_read": manifest["protocol_assertions"]["validation_read"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
