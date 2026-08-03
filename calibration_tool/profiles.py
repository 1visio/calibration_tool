from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .io_utils import canonical_mapping_hash, load_document, resolve_relative, sha256_file


CALIBRATION_FILE_KEYS = (
    "intrinsics",
    "laser_plane",
    "extrinsics",
    "ground_u_compensation",
)


def load_runtime_profile(
    config_path: str | Path,
    *,
    expected_extractor: str | None = None,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    document = load_document(path)
    if document.get("schema_version") != 1:
        raise ConfigError(f"只支持 schema_version=1：{path}")

    calibration = _mapping(document.get("calibration"), "calibration")
    extraction = _mapping(document.get("extraction"), "extraction")
    method = _required_text(extraction.get("method"), "extraction.method")
    options = extraction.get(method, {})
    if not isinstance(options, Mapping):
        raise ConfigError(f"extraction.{method} 必须是映射")

    files: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, str]] = []
    for key in CALIBRATION_FILE_KEYS:
        value = calibration.get(key)
        if value in (None, ""):
            if key != "ground_u_compensation":
                findings.append(_finding("missing_calibration_path", "fail", f"缺少 calibration.{key}"))
            continue
        target = resolve_relative(path, _required_text(value, f"calibration.{key}"))
        record: dict[str, Any] = {"path": str(target), "exists": target.is_file()}
        if target.is_file():
            record.update(
                sha256=sha256_file(target),
                raw_sha256=sha256_file(target, normalize_newlines=False),
                size_bytes=target.stat().st_size,
            )
        else:
            findings.append(_finding("missing_calibration_file", "fail", f"标定文件不存在：{target}"))
        files[key] = record

    manifest_info: dict[str, Any] | None = None
    manifest_value = calibration.get("manifest")
    if manifest_value not in (None, ""):
        manifest_path = resolve_relative(path, _required_text(manifest_value, "calibration.manifest"))
        manifest_info = _inspect_manifest(manifest_path)
        findings.extend(manifest_info.pop("findings"))
        manifest_algorithm = manifest_info.get("algorithm")
        if manifest_algorithm and method != manifest_algorithm:
            findings.append(
                _finding(
                    "runtime_manifest_extractor_mismatch",
                    "fail",
                    f"运行 config 选择 {method}，标定 manifest 声明 {manifest_algorithm}",
                )
            )

    if expected_extractor and method != expected_extractor:
        findings.append(
            _finding(
                "expected_extractor_mismatch",
                "fail",
                f"运行 config 选择 {method}，golden 标定链要求 {expected_extractor}",
            )
        )

    camera = _camera_summary(files.get("intrinsics"))
    return {
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "config_raw_sha256": sha256_file(path, normalize_newlines=False),
        "semantic_sha256": canonical_mapping_hash(document),
        "schema_version": 1,
        "selected_extractor": {
            "method": method,
            "options": dict(options),
            "profile_sha256": canonical_mapping_hash({"method": method, "options": dict(options)}),
        },
        "expected_extractor": expected_extractor,
        "camera": camera,
        "calibration_files": files,
        "manifest": manifest_info,
        "findings": findings,
    }


def _inspect_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "findings": [_finding("missing_manifest", "fail", f"manifest 不存在：{path}")],
        }
    document = load_document(path)
    extractor = document.get("extractor")
    algorithm = extractor.get("algorithm") if isinstance(extractor, Mapping) else None
    findings: list[dict[str, str]] = []
    entries: dict[str, Any] = {}
    files = document.get("files")
    if not isinstance(files, Mapping):
        findings.append(_finding("manifest_files_invalid", "fail", "manifest.files 必须是映射"))
        files = {}
    for name, value in files.items():
        if not isinstance(value, Mapping):
            findings.append(_finding("manifest_entry_invalid", "fail", f"manifest.files.{name} 不是映射"))
            continue
        relative = value.get("path")
        expected = str(value.get("sha256", "")).lower()
        if not isinstance(relative, str) or not relative:
            findings.append(_finding("manifest_path_invalid", "fail", f"manifest.files.{name}.path 无效"))
            continue
        target = (path.parent / relative).resolve()
        actual = sha256_file(target) if target.is_file() else None
        matches = actual == expected
        entries[str(name)] = {
            "path": str(target),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            findings.append(_finding("manifest_hash_mismatch", "fail", f"manifest 哈希不匹配：{name}"))
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "package_id": document.get("package_id"),
        "algorithm": algorithm,
        "quality": document.get("quality", {}),
        "files": entries,
        "findings": findings,
    }


def _camera_summary(intrinsics_record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not intrinsics_record or not intrinsics_record.get("exists"):
        return {}
    document = load_document(str(intrinsics_record["path"]))
    return {
        "image_width": document.get("image_width"),
        "image_height": document.get("image_height"),
        "camera_matrix": document.get("camera_matrix"),
        "dist_coeffs": document.get("dist_coeffs"),
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是映射")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须是非空字符串")
    return value.strip()


def _finding(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}

