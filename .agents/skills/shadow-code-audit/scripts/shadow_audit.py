#!/usr/bin/env python3
"""Cross-process Codex shadow auditor.

Modes:
  init   - capture the requirement contract used by success-goal-auditor
  watch  - watch a Git worktree and probabilistically run read-only auditors
  audit  - run an audit immediately (with --final to force both auditors)
  hook   - Codex hook bridge for UserPromptSubmit + Stop
  checkpoint-hook - non-blocking PostToolUse checkpoint for apply_patch

Uses only the Python standard library. Reviewer processes are fresh, ephemeral,
and read-only Codex exec invocations.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "base_ref": "HEAD",
    "poll_seconds": 2.0,
    "debounce_seconds": 8.0,
    "minimum_interval_seconds": 30.0,
    "max_parallel_auditors": 2,
    "final_audit_on_stop": True,
    "block_on_audit_error": False,
    "checkpoint": {
        "enabled": True,
        "debounce_seconds": 8.0,
        "minimum_interval_seconds": 90.0,
        "activation_probability": 0.35,
    },
    "auditors": {
        "code-smell-auditor": {
            "enabled": True,
            "activation_probability": 0.4,
            "timeout_seconds": 120,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "high",
        },
        "success-goal-auditor": {
            "enabled": True,
            "activation_probability": 0.4,
            "timeout_seconds": 180,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "xhigh",
        },
    },
}

PASS_SENTINELS = {
    "code-smell-auditor": "NO_FINDING",
    "success-goal-auditor": "GOAL_ALIGNED",
}

AUDITOR_EXECUTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "code-smell-auditor": {
        "model": "gpt-5.6-luna",
        "reasoning_effort": "high",
        "timeout_seconds": 120,
    },
    "success-goal-auditor": {
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "timeout_seconds": 180,
    },
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run(cmd: list[str], cwd: pathlib.Path, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def repo_root(start: str | pathlib.Path = ".") -> pathlib.Path:
    start_path = pathlib.Path(start).resolve()
    cp = run(["git", "rev-parse", "--show-toplevel"], start_path)
    if cp.returncode != 0:
        raise SystemExit(f"Not inside a Git repository: {start_path}\n{cp.stderr.strip()}")
    return pathlib.Path(cp.stdout.strip()).resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def config_path(root: pathlib.Path) -> pathlib.Path:
    return root / ".codex" / "shadow-audit.json"


def state_dir(root: pathlib.Path) -> pathlib.Path:
    p = root / ".codex-shadow"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(root: pathlib.Path) -> dict[str, Any]:
    path = config_path(root)
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid {path}: {exc}")
    return deep_merge(DEFAULT_CONFIG, data)


def _git_command_failure(command: list[str], completed: subprocess.CompletedProcess[str]) -> RuntimeError:
    stderr = (completed.stderr or "").strip()
    diagnostic = stderr or (completed.stdout or "").strip() or "<no diagnostic>"
    return RuntimeError(
        f"git command failed: command={shlex.join(command)}; returncode={completed.returncode}; stderr={diagnostic}"
    )


def git_fingerprint(root: pathlib.Path, base_ref: str) -> str:
    """Hash tracked diff plus untracked path metadata (excluding runtime state)."""
    h = hashlib.sha256()
    diff_command = ["git", "diff", "--binary", "--no-ext-diff", base_ref, "--", "."]
    cp = run(diff_command, root)
    if cp.returncode != 0:
        raise _git_command_failure(diff_command, cp)
    h.update((cp.stdout or "").encode("utf-8", errors="replace"))

    ls_files_command = ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    untracked = run(ls_files_command, root)
    if untracked.returncode != 0:
        raise _git_command_failure(ls_files_command, untracked)
    for rel in sorted(x for x in untracked.stdout.split("\0") if x):
        if rel == ".codex-shadow" or rel.startswith(".codex-shadow/"):
            continue
        p = root / rel
        stat = p.stat()
        normalized_rel = rel.replace("\\", "/")
        h.update(
            f"untracked\0path={normalized_rel}\0size={stat.st_size}\0mtime_ns={stat.st_mtime_ns}\n".encode(
                "utf-8", errors="replace"
            )
        )
    return h.hexdigest()


def changed_summary(root: pathlib.Path, base_ref: str) -> str:
    status = run(["git", "status", "--short"], root).stdout.strip()
    stat = run(["git", "diff", "--stat", base_ref, "--", "."], root).stdout.strip()
    return f"BASE_REF: {base_ref}\nSTATUS:\n{status or '(clean)'}\nDIFF_STAT:\n{stat or '(no tracked diff)'}"


def requirements_path(root: pathlib.Path) -> pathlib.Path:
    return state_dir(root) / "requirements.md"


@contextlib.contextmanager
def _exclusive_file_lock(path: pathlib.Path):
    """Serialize append/dedup checks across concurrent hook processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write("0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def events_path(root: pathlib.Path) -> pathlib.Path:
    return state_dir(root) / "events.jsonl"


def _short_error(exc: BaseException, limit: int = 240) -> str:
    message = " ".join(str(exc).split())
    if not message:
        message = "(no error message)"
    return message[:limit]


def _error_fields(exc: BaseException) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "error": _short_error(exc)}


def append_event(root: pathlib.Path, data: dict[str, Any]) -> None:
    """Best-effort append of one structured hook diagnostic event."""
    try:
        path = events_path(root)
        lock_path = path.with_name(path.name + ".lock")
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _exclusive_file_lock(lock_path):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception as exc:
        # Diagnostics must never change the hook's behavior. There is no usable
        # persistent fallback if the state directory itself is unavailable, so
        # leave a best-effort stderr breadcrumb as the last resort.
        print(f"shadow audit: failed to write event log: {_short_error(exc)}", file=sys.stderr)


def record_event(
    root: pathlib.Path,
    event_name: str,
    session_id: str,
    turn_id: str,
    **details: Any,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "event": event_name,
        "session_id": session_id,
        "turn_id": turn_id,
    }
    payload.update(details)
    try:
        append_event(root, payload)
    except Exception as exc:
        # Keep callers fail-soft even if tests or an embedding caller replace
        # append_event with an implementation that raises.
        print(f"shadow audit: event {event_name} unavailable: {_short_error(exc)}", file=sys.stderr)


def _requirement_marker(session_id: str | None, turn_id: str | None) -> str | None:
    if not (session_id or turn_id):
        return None
    return f"<!-- session={session_id or ''} turn={turn_id or ''} -->"


def normalize_prompt(prompt: str) -> tuple[str, int]:
    """Make a prompt safe for strict UTF-8 persistence without dropping it."""
    replacement_count = sum(0xD800 <= ord(character) <= 0xDFFF for character in prompt)
    normalized = prompt.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return normalized, replacement_count


def append_requirement(root: pathlib.Path, prompt: str, session_id: str | None = None, turn_id: str | None = None) -> None:
    prompt, _ = normalize_prompt(prompt)
    if not prompt.strip() or prompt.startswith("SHADOW_AUDIT_FEEDBACK:"):
        return
    path = requirements_path(root)
    header = f"\n\n## Requirement update — {now_iso()}"
    marker = _requirement_marker(session_id, turn_id)
    if marker:
        header += f"\n{marker}"
    lock_path = path.with_name(path.name + ".lock")
    with _exclusive_file_lock(lock_path):
        if marker and path.exists() and marker in path.read_text(encoding="utf-8"):
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(header + "\n\n" + prompt.strip() + "\n")


def init_requirements(root: pathlib.Path, text: str | None, source_file: str | None, force: bool) -> pathlib.Path:
    path = requirements_path(root)
    if path.exists() and not force:
        raise SystemExit(f"Requirements already exist: {path}. Use --force to replace them.")
    if source_file:
        content = pathlib.Path(source_file).read_text(encoding="utf-8")
    elif text:
        content = text
    else:
        raise SystemExit("Provide --requirements or --requirements-file.")
    path.write_text(
        f"# Original requirement contract\n\nCaptured: {now_iso()}\n\n{content.strip()}\n",
        encoding="utf-8",
    )
    return path


def codex_executable() -> str:
    exe = shutil.which("codex")
    if not exe:
        raise FileNotFoundError("`codex` CLI was not found on PATH")
    return exe


def auditor_prompt(root: pathlib.Path, auditor: str, base_ref: str, final: bool) -> str:
    req = requirements_path(root)
    scope = changed_summary(root, base_ref)
    final_text = "This is the final completion gate; inspect all material requirements in scope." if final else "This is a checkpoint audit of the current working tree."
    if auditor == "success-goal-auditor":
        requirement_note = (
            f"Read the requirement log at {req.relative_to(root)}. Later explicit entries override conflicting earlier entries."
            if req.exists()
            else "The requirement log is missing. Do not invent requirements; use the skill's AUDIT_BLOCKED output."
        )
    else:
        requirement_note = "Do not perform requirement-completion review; stay within code/project-structure scope."
    return (
        f"${auditor}\n\n"
        f"Audit another process's current implementation in this repository. {final_text}\n"
        f"{requirement_note}\n"
        f"Use the repository as the evidence source. Do not edit files. Return only the skill's contracted output.\n\n"
        f"Current change summary:\n{scope}\n"
    )


def write_run_metadata(report_dir: pathlib.Path, data: dict[str, Any]) -> None:
    (report_dir / "metadata.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def auditor_execution_settings(auditor: str, cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = AUDITOR_EXECUTION_DEFAULTS.get(auditor, {})
    return {
        "model": str(cfg.get("model", defaults.get("model", "gpt-5.6-luna"))),
        "reasoning_effort": str(cfg.get("reasoning_effort", defaults.get("reasoning_effort", "high"))),
        "timeout_seconds": float(cfg.get("timeout_seconds", defaults.get("timeout_seconds", 120))),
    }


def auditor_event_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "auditor": result.get("auditor"),
            "model": result.get("model"),
            "reasoning_effort": result.get("reasoning_effort"),
            "duration_seconds": result.get("duration_seconds"),
            "status": result.get("status"),
        }
        for result in results
    ]


def configured_auditor_summary(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "auditor": name,
            **{
                key: settings[key]
                for key in ("model", "reasoning_effort", "timeout_seconds")
            },
        }
        for name, auditor_cfg in cfg.get("auditors", {}).items()
        if auditor_cfg.get("enabled", True)
        for settings in [auditor_execution_settings(name, auditor_cfg)]
    ]


def run_one_auditor(root: pathlib.Path, auditor: str, cfg: dict[str, Any], base_ref: str, final: bool, report_dir: pathlib.Path) -> dict[str, Any]:
    settings = auditor_execution_settings(auditor, cfg)
    timeout = settings["timeout_seconds"]
    model = settings["model"]
    reasoning_effort = settings["reasoning_effort"]
    output_path = report_dir / f"{auditor}.md"
    stderr_path = report_dir / f"{auditor}.stderr.log"
    prompt = auditor_prompt(root, auditor, base_ref, final)
    started = time.time()
    result: dict[str, Any] = {
        "auditor": auditor,
        "started_at": now_iso(),
        "timeout_seconds": timeout,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }
    try:
        exe = codex_executable()
        env = os.environ.copy()
        env["CODEX_SHADOW_AUDIT_CHILD"] = "1"
        cp = subprocess.run(
            [
                exe,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                model,
                "--config",
                f"model_reasoning_effort={reasoning_effort}",
                "-o",
                str(output_path),
                prompt,
            ],
            cwd=str(root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stderr_path.write_text(cp.stderr, encoding="utf-8")
        text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else cp.stdout.strip()
        result.update({
            "status": "ok" if cp.returncode == 0 else "error",
            "returncode": cp.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "report": text or f"AUDIT_ERROR\nreason: auditor returned no final message (exit {cp.returncode})",
        })
    except subprocess.TimeoutExpired:
        result.update({
            "status": "timeout",
            "duration_seconds": round(time.time() - started, 3),
            "report": f"AUDIT_ERROR\nreason: {auditor} timed out after {timeout:g}s",
        })
    except Exception as exc:
        result.update({
            "status": "error",
            "duration_seconds": round(time.time() - started, 3),
            "report": f"AUDIT_ERROR\nreason: {exc}",
        })
    if not output_path.exists():
        output_path.write_text(result["report"] + "\n", encoding="utf-8")
    return result


def should_activate(auditor_cfg: dict[str, Any], final: bool, rng: random.Random) -> bool:
    if not auditor_cfg.get("enabled", True):
        return False
    if final:
        return True
    p = float(auditor_cfg.get("activation_probability", 0.3))
    return rng.random() < max(0.0, min(1.0, p))


def report_is_issue(auditor: str, report: str, block_on_error: bool) -> bool:
    stripped = report.strip()
    sentinel = PASS_SENTINELS.get(auditor)
    if sentinel and stripped == sentinel:
        return False
    if stripped.startswith("AUDIT_ERROR"):
        return block_on_error
    return True


_FINDING_MARKER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:finding|issue|code[_ -]?smell(?:\s+finding)?)\b",
    re.IGNORECASE,
)
_SEVERITY_SCORES = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _checkpoint_finding_blocks(report: str) -> list[str]:
    lines = report.strip().splitlines()
    markers = [index for index, line in enumerate(lines) if _FINDING_MARKER_RE.match(line)]
    if len(markers) > 1:
        blocks = []
        for index, start in enumerate(markers):
            end = markers[index + 1] if index + 1 < len(markers) else len(lines)
            block = "\n".join(lines[start:end]).strip()
            if block:
                blocks.append(block)
        return blocks
    paragraphs = [paragraph.strip() for paragraph in report.strip().split("\n\n") if paragraph.strip()]
    return paragraphs or ([report.strip()] if report.strip() else [])


def _checkpoint_finding_score(block: str, index: int) -> tuple[int, int, int]:
    lower = block.lower()
    severity = max(
        (score for name, score in _SEVERITY_SCORES.items() if re.search(rf"\b{name}\b", lower)),
        default=0,
    )
    specificity = sum(
        token in lower
        for token in ("file", "path", "line", "impact", "why", "fix", "suggest")
    )
    return severity, specificity, -index


def _checkpoint_finding_lines(block: str) -> list[str]:
    lines: list[str] = []
    for raw_line in block.splitlines():
        line = re.sub(r"^\s*(?:[-*]\s+|#{1,6}\s+)", "", raw_line).strip()
        line = re.sub(
            r"^(?:finding|issue|code[_ -]?smell(?:\s+finding)?)\s*(?:#?\d+)?\s*[:\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if line:
            lines.append(line)
    if not lines:
        return []
    if len(lines) == 1:
        lines.append("scope: code and project structure")
    return lines[:4]


def checkpoint_additional_context(report: str, status: str) -> str | None:
    if status != "ok":
        return None
    stripped = report.strip()
    if not stripped or not report_is_issue(CHECKPOINT_AUDITOR, stripped, False):
        return None
    blocks = _checkpoint_finding_blocks(stripped)
    if not blocks:
        return None
    best = max(enumerate(blocks), key=lambda item: _checkpoint_finding_score(item[1], item[0]))[1]
    lines = _checkpoint_finding_lines(best)
    if not lines:
        return None
    return "🔴 shadow · [代码与项目结构审计者 / code-smell-auditor]\n" + "\n".join(lines)


def audit_cycle(
    root: pathlib.Path,
    final: bool = False,
    seed: int | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    skip_auditors: set[str] | None = None,
    precomputed_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = load_config(root)
    if not cfg.get("enabled", True):
        return {"status": "disabled", "reports": []}
    base_ref = str(cfg.get("base_ref", "HEAD"))
    rng = random.Random(seed)
    audit_cfgs: dict[str, Any] = cfg.get("auditors", {})
    skipped = skip_auditors or set()
    selected = [
        name
        for name, acfg in audit_cfgs.items()
        if name not in skipped and should_activate(acfg, final, rng)
    ]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = state_dir(root) / "reports" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = list(precomputed_results or [])
    max_workers = max(1, min(int(cfg.get("max_parallel_auditors", 2)), len(selected) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_one_auditor, root, name, audit_cfgs[name], base_ref, final, out_dir): name
            for name in selected
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            record_event(
                root,
                "auditor_finished",
                session_id or "unknown-session",
                turn_id or "unknown-turn",
                auditor=result.get("auditor"),
                model=result.get("model"),
                reasoning_effort=result.get("reasoning_effort"),
                duration_seconds=result.get("duration_seconds"),
                status=result.get("status"),
            )
    results.sort(key=lambda r: r["auditor"])

    block_on_error = bool(cfg.get("block_on_audit_error", False))
    issues = [r for r in results if report_is_issue(r["auditor"], r["report"], block_on_error)]
    metadata = {
        "created_at": now_iso(),
        "final": final,
        "base_ref": base_ref,
        "fingerprint": git_fingerprint(root, base_ref),
        "selected": selected,
        "auditors": auditor_event_summary(results),
        "results": [{k: v for k, v in r.items() if k != "report"} for r in results],
        "issue_auditors": [r["auditor"] for r in issues],
    }
    write_run_metadata(out_dir, metadata)
    render_latest(root, out_dir, results, final)
    return {"status": "ok", "report_dir": str(out_dir), "reports": results, "issues": issues}


def render_latest(root: pathlib.Path, out_dir: pathlib.Path, results: list[dict[str, Any]], final: bool) -> None:
    lines = [f"# Shadow audit — {'final' if final else 'checkpoint'}", "", f"Generated: {now_iso()}", f"Report dir: {out_dir}", ""]
    if not results:
        lines += ["No auditor activated in this probabilistic checkpoint.", ""]
    for result in results:
        lines += [f"## {result['auditor']}", "", result["report"].strip(), ""]
    (state_dir(root) / "latest.md").write_text("\n".join(lines), encoding="utf-8")


def print_cycle(cycle: dict[str, Any]) -> None:
    if cycle.get("status") != "ok":
        print(f"shadow audit: {cycle.get('status')}")
        return
    reports = cycle.get("reports", [])
    if not reports:
        print("shadow audit: no auditor activated")
        return
    for result in reports:
        print(f"\n[{result['auditor']}]\n{result['report'].strip()}")
    print(f"\nReports: {cycle.get('report_dir')}")


def watch(root: pathlib.Path, once: bool = False) -> None:
    cfg = load_config(root)
    base_ref = str(cfg.get("base_ref", "HEAD"))
    poll = float(cfg.get("poll_seconds", 2.0))
    debounce = float(cfg.get("debounce_seconds", 8.0))
    min_interval = float(cfg.get("minimum_interval_seconds", 30.0))
    last_fp = git_fingerprint(root, base_ref)
    last_change = time.monotonic()
    last_audit = 0.0
    print(f"Watching {root} (base={base_ref}). Ctrl+C to stop.")
    print(f"Reports: {state_dir(root) / 'reports'}")
    try:
        while True:
            time.sleep(poll)
            fp = git_fingerprint(root, base_ref)
            now = time.monotonic()
            if fp != last_fp:
                last_fp = fp
                last_change = now
                continue
            if now - last_change >= debounce and now - last_audit >= min_interval:
                # Do not audit a clean tree unless explicitly requested with `audit`.
                status = run(["git", "status", "--porcelain"], root).stdout.strip()
                if status:
                    cycle = audit_cycle(root, final=False)
                    last_audit = time.monotonic()
                    print_cycle(cycle)
                    if once:
                        return
                last_change = now
            if once and now - last_change >= debounce:
                return
    except KeyboardInterrupt:
        print("\nStopped.")


CHECKPOINT_AUDITOR = "code-smell-auditor"


def checkpoint_state_file(root: pathlib.Path) -> pathlib.Path:
    return state_dir(root) / "checkpoint-state.json"


def _checkpoint_skip(
    root: pathlib.Path,
    session_id: str,
    turn_id: str,
    reason: str,
    **details: Any,
) -> int:
    record_event(root, "checkpoint_skipped", session_id, turn_id, reason=reason, **details)
    print("{}")
    return 0


def _checkpoint_number(state: dict[str, Any], key: str) -> float | None:
    value = state.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"checkpoint state field {key!r} is not numeric")
    return float(value)


def _checkpoint_reusable_result(
    root: pathlib.Path,
    fingerprint: str,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return a checkpoint PASS only when its durable state proves it is reusable."""
    try:
        state_path = checkpoint_state_file(root)
        lock_path = state_path.with_name(state_path.name + ".lock")
        with _exclusive_file_lock(lock_path):
            if not state_path.exists():
                return None, {"reason": "checkpoint_state_missing"}
            state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("checkpoint state must be a JSON object")
    except Exception as exc:
        return None, {"reason": "checkpoint_state_unavailable", **_error_fields(exc)}

    audited_fingerprint = state.get("audited_fingerprint")
    if audited_fingerprint != fingerprint:
        return None, {
            "reason": "fingerprint_not_reusable",
            "checkpoint_fingerprint": audited_fingerprint,
        }

    auditor = state.get("auditor", state.get("audited_auditor"))
    if auditor != CHECKPOINT_AUDITOR:
        return None, {"reason": "auditor_not_reusable", "checkpoint_auditor": auditor}

    status = state.get("status", state.get("audited_status"))
    if status != "ok":
        return None, {"reason": "checkpoint_not_successful", "checkpoint_status": status}
    if state.get("audited_pass") is not True:
        return None, {"reason": "checkpoint_not_pass", "checkpoint_status": status}

    timestamp = state.get("timestamp", state.get("audited_at"))
    if not isinstance(timestamp, str) or not timestamp:
        return None, {"reason": "checkpoint_timestamp_missing"}

    auditor_cfgs = cfg.get("auditors", {})
    auditor_cfg = auditor_cfgs.get(CHECKPOINT_AUDITOR) if isinstance(auditor_cfgs, dict) else None
    if not isinstance(auditor_cfg, dict) or not auditor_cfg.get("enabled", True):
        return None, {"reason": "auditor_disabled", "auditor": CHECKPOINT_AUDITOR}
    settings = auditor_execution_settings(CHECKPOINT_AUDITOR, auditor_cfg)
    duration = state.get("audited_duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        duration = 0.0
    return {
        "auditor": CHECKPOINT_AUDITOR,
        "started_at": timestamp,
        "finished_at": timestamp,
        "timeout_seconds": settings["timeout_seconds"],
        "model": str(state.get("audited_model") or settings["model"]),
        "reasoning_effort": str(state.get("audited_reasoning_effort") or settings["reasoning_effort"]),
        "duration_seconds": duration,
        "status": "ok",
        "report": PASS_SENTINELS[CHECKPOINT_AUDITOR],
        "reused": True,
        "checkpoint_timestamp": timestamp,
    }, {
        "reason": "valid_pass",
        "fingerprint": fingerprint,
        "auditor": CHECKPOINT_AUDITOR,
        "status": "ok",
        "checkpoint_timestamp": timestamp,
    }


def _checkpoint_process(
    root: pathlib.Path,
    event: dict[str, Any],
    session_id: str,
    turn_id: str,
) -> int:
    state_path = checkpoint_state_file(root)
    lock_path = state_path.with_name(state_path.name + ".lock")
    with _exclusive_file_lock(lock_path):
        return _checkpoint_process_unlocked(root, event, session_id, turn_id)


def _checkpoint_process_unlocked(
    root: pathlib.Path,
    event: dict[str, Any],
    session_id: str,
    turn_id: str,
) -> int:
    try:
        cfg = load_config(root)
    except (Exception, SystemExit) as exc:
        return _checkpoint_skip(root, session_id, turn_id, "config_failed", **_error_fields(exc))
    if not cfg.get("enabled", True):
        return _checkpoint_skip(root, session_id, turn_id, "shadow_audit_disabled")

    checkpoint_cfg = cfg.get("checkpoint", {})
    if not isinstance(checkpoint_cfg, dict) or not checkpoint_cfg.get("enabled", True):
        return _checkpoint_skip(root, session_id, turn_id, "checkpoint_disabled")
    try:
        debounce_seconds = max(0.0, float(checkpoint_cfg.get("debounce_seconds", 8.0)))
        minimum_interval_seconds = max(
            0.0,
            float(checkpoint_cfg.get("minimum_interval_seconds", 90.0)),
        )
        activation_probability = max(
            0.0,
            min(1.0, float(checkpoint_cfg.get("activation_probability", 0.35))),
        )
    except (TypeError, ValueError) as exc:
        return _checkpoint_skip(root, session_id, turn_id, "config_invalid", **_error_fields(exc))

    auditor_cfgs = cfg.get("auditors", {})
    auditor_cfg = auditor_cfgs.get(CHECKPOINT_AUDITOR) if isinstance(auditor_cfgs, dict) else None
    if not isinstance(auditor_cfg, dict) or not auditor_cfg.get("enabled", True):
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "auditor_disabled",
            auditor=CHECKPOINT_AUDITOR,
        )

    try:
        state_path = checkpoint_state_file(root)
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        if not isinstance(state, dict):
            raise ValueError("checkpoint state must be a JSON object")
        for key in ("pending_fingerprint", "last_decided_fingerprint", "last_audited_fingerprint"):
            value = state.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"checkpoint state field {key!r} is not a string")
        pending_since = _checkpoint_number(state, "pending_since")
        last_audit_at = _checkpoint_number(state, "last_audit_at")
    except Exception as exc:
        return _checkpoint_skip(root, session_id, turn_id, "state_invalid", **_error_fields(exc))

    base_ref = str(cfg.get("base_ref", "HEAD"))
    try:
        fingerprint = git_fingerprint(root, base_ref)
    except Exception as exc:
        return _checkpoint_skip(root, session_id, turn_id, "fingerprint_failed", **_error_fields(exc))

    if fingerprint == state.get("last_decided_fingerprint"):
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "fingerprint_unchanged",
            fingerprint=fingerprint,
        )

    now = time.time()
    pending_fingerprint = state.get("pending_fingerprint")
    if pending_fingerprint != fingerprint:
        state["pending_fingerprint"] = fingerprint
        state["pending_since"] = now
        pending_since = now
        record_event(
            root,
            "checkpoint_candidate",
            session_id,
            turn_id,
            fingerprint=fingerprint,
            base_ref=base_ref,
            auditor=CHECKPOINT_AUDITOR,
        )
        try:
            write_json_atomic(state_path, state)
        except Exception as exc:
            return _checkpoint_skip(
                root,
                session_id,
                turn_id,
                "state_write_failed",
                phase="candidate",
                fingerprint=fingerprint,
                **_error_fields(exc),
            )
    elif pending_since is None:
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "state_invalid",
            **_error_fields(ValueError("pending fingerprint has no pending_since timestamp")),
        )

    assert pending_since is not None
    candidate_age = max(0.0, now - pending_since)
    if candidate_age < debounce_seconds:
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "debounce",
            fingerprint=fingerprint,
            elapsed_seconds=round(candidate_age, 3),
            required_seconds=debounce_seconds,
        )

    if last_audit_at is not None:
        since_last_audit = max(0.0, now - last_audit_at)
        if since_last_audit < minimum_interval_seconds:
            return _checkpoint_skip(
                root,
                session_id,
                turn_id,
                "minimum_interval",
                fingerprint=fingerprint,
                elapsed_seconds=round(since_last_audit, 3),
                required_seconds=minimum_interval_seconds,
            )

    sample = random.random()
    if sample >= activation_probability:
        state["last_decided_fingerprint"] = fingerprint
        state.pop("pending_fingerprint", None)
        state.pop("pending_since", None)
        try:
            write_json_atomic(state_path, state)
        except Exception as exc:
            return _checkpoint_skip(
                root,
                session_id,
                turn_id,
                "state_write_failed",
                phase="probability",
                fingerprint=fingerprint,
                **_error_fields(exc),
            )
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "activation_probability",
            fingerprint=fingerprint,
            sample=round(sample, 6),
            probability=activation_probability,
        )

    try:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        report_dir = state_dir(root) / "reports" / f"checkpoint-{stamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        settings = auditor_execution_settings(CHECKPOINT_AUDITOR, auditor_cfg)
        state["last_decided_fingerprint"] = fingerprint
        state["last_audit_started_at"] = now
        state.pop("pending_fingerprint", None)
        state.pop("pending_since", None)
        write_json_atomic(state_path, state)
    except Exception as exc:
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "start_failed",
            fingerprint=fingerprint,
            **_error_fields(exc),
        )

    record_event(
        root,
        "checkpoint_started",
        session_id,
        turn_id,
        fingerprint=fingerprint,
        auditor=CHECKPOINT_AUDITOR,
        model=settings["model"],
        reasoning_effort=settings["reasoning_effort"],
        timeout_seconds=settings["timeout_seconds"],
    )
    started = time.time()
    result: dict[str, Any]
    execution_error: BaseException | None = None
    try:
        result = run_one_auditor(
            root,
            CHECKPOINT_AUDITOR,
            auditor_cfg,
            base_ref,
            False,
            report_dir,
        )
    except Exception as exc:
        execution_error = exc
        result = {
            "auditor": CHECKPOINT_AUDITOR,
            "model": settings["model"],
            "reasoning_effort": settings["reasoning_effort"],
            "status": "error",
            "duration_seconds": round(time.time() - started, 3),
            "report": f"AUDIT_ERROR\nreason: {exc}",
        }
    finished_at = time.time()

    result_status = str(result.get("status") or "error")
    result_report = str(result.get("report") or "")
    audited_pass = result_status == "ok" and result_report.strip() == PASS_SENTINELS[CHECKPOINT_AUDITOR]
    if audited_pass:
        audited_outcome = "pass"
    elif result_status == "ok":
        audited_outcome = "finding"
    else:
        audited_outcome = result_status
    audited_timestamp = now_iso()
    state["last_audited_fingerprint"] = fingerprint
    state["last_audit_at"] = finished_at
    state["audited_fingerprint"] = fingerprint
    state["auditor"] = CHECKPOINT_AUDITOR
    state["status"] = result_status
    state["timestamp"] = audited_timestamp
    state["audited_auditor"] = CHECKPOINT_AUDITOR
    state["audited_status"] = result_status
    state["audited_at"] = audited_timestamp
    state["audited_timestamp"] = finished_at
    state["audited_pass"] = audited_pass
    state["audited_outcome"] = audited_outcome
    state["audited_model"] = result.get("model", settings["model"])
    state["audited_reasoning_effort"] = result.get("reasoning_effort", settings["reasoning_effort"])
    state["audited_duration_seconds"] = result.get("duration_seconds")
    state.pop("last_audit_started_at", None)
    state_saved = True
    state_error: BaseException | None = None
    try:
        write_json_atomic(state_path, state)
    except Exception as exc:
        state_saved = False
        state_error = exc

    metadata_error: BaseException | None = None
    try:
        write_run_metadata(
            report_dir,
            {
                "created_at": now_iso(),
                "checkpoint": True,
                "final": False,
                "base_ref": base_ref,
                "fingerprint": fingerprint,
                "selected": [CHECKPOINT_AUDITOR],
                "auditors": auditor_event_summary([result]),
                "results": [{k: v for k, v in result.items() if k != "report"}],
            },
        )
    except Exception as exc:
        metadata_error = exc

    duration = result.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        duration = round(finished_at - started, 3)
    finished_details: dict[str, Any] = {
        "fingerprint": fingerprint,
        "auditor": result.get("auditor", CHECKPOINT_AUDITOR),
        "model": result.get("model", settings["model"]),
        "reasoning_effort": result.get("reasoning_effort", settings["reasoning_effort"]),
        "duration_seconds": duration,
        "status": result_status,
        "audited_fingerprint": fingerprint,
        "audited_auditor": CHECKPOINT_AUDITOR,
        "audited_status": result_status,
        "audited_pass": audited_pass,
        "audited_outcome": audited_outcome,
        "audited_timestamp": audited_timestamp,
        "state_saved": state_saved,
        "report_dir": str(report_dir),
    }
    if execution_error is not None:
        finished_details.update(_error_fields(execution_error))
    if state_error is not None:
        finished_details["state_error_type"] = type(state_error).__name__
        finished_details["state_error"] = _short_error(state_error)
    if metadata_error is not None:
        finished_details["metadata_error_type"] = type(metadata_error).__name__
        finished_details["metadata_error"] = _short_error(metadata_error)
    record_event(root, "checkpoint_finished", session_id, turn_id, **finished_details)
    context = checkpoint_additional_context(
        str(result.get("report") or ""),
        str(result.get("status") or ""),
    )
    if context:
        record_event(
            root,
            "checkpoint_feedback_emitted",
            session_id,
            turn_id,
            fingerprint=fingerprint,
            auditor=CHECKPOINT_AUDITOR,
            finding_count=1,
            context_line_count=max(0, len(context.splitlines()) - 1),
        )
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": context,
                    }
                },
                ensure_ascii=False,
            )
        )
    else:
        print("{}")
    return 0


def checkpoint_hook_main(root: pathlib.Path) -> int:
    # Auditor child codex processes inherit project hooks. Never recurse.
    if os.environ.get("CODEX_SHADOW_AUDIT_CHILD") == "1":
        print("{}")
        return 0
    try:
        event = json.load(sys.stdin)
    except Exception:
        print("{}")
        return 0

    event_name = event.get("hook_event_name")
    session_id = str(event.get("session_id") or "unknown-session")
    turn_id = str(event.get("turn_id") or "unknown-turn")
    record_event(
        root,
        "hook_received",
        session_id,
        turn_id,
        hook_event_name=event_name,
    )
    tool_name = str(event.get("tool_name") or event.get("toolName") or "")
    if event_name != "PostToolUse" or tool_name != "apply_patch":
        return _checkpoint_skip(
            root,
            session_id,
            turn_id,
            "tool_not_apply_patch",
            tool_name=tool_name,
        )
    try:
        return _checkpoint_process(root, event, session_id, turn_id)
    except (Exception, SystemExit) as exc:
        return _checkpoint_skip(root, session_id, turn_id, "checkpoint_failed", **_error_fields(exc))


def turn_state_file(root: pathlib.Path, session_id: str, turn_id: str) -> pathlib.Path:
    safe = hashlib.sha256(f"{session_id}:{turn_id}".encode()).hexdigest()[:20]
    p = state_dir(root) / "turns"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.json"


def write_json_atomic(path: pathlib.Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def hook_main(root: pathlib.Path) -> int:
    # Auditor child codex processes inherit project hooks. Never recurse.
    if os.environ.get("CODEX_SHADOW_AUDIT_CHILD") == "1":
        print("{}")
        return 0
    try:
        event = json.load(sys.stdin)
    except Exception:
        print("{}")
        return 0

    event_name = event.get("hook_event_name")
    session_id = str(event.get("session_id") or "unknown-session")
    turn_id = str(event.get("turn_id") or "unknown-turn")
    record_event(
        root,
        "hook_received",
        session_id,
        turn_id,
        hook_event_name=event_name,
    )

    cfg = load_config(root)
    if not cfg.get("enabled", True):
        if event_name == "Stop":
            record_event(root, "stop", session_id, turn_id, reason="final_audit_disabled")
        else:
            record_event(root, "hook_disabled", session_id, turn_id)
        print("{}")
        return 0
    base_ref = str(cfg.get("base_ref", "HEAD"))

    if event_name == "UserPromptSubmit":
        prompt = str(event.get("prompt") or "")
        prompt, replacement_count = normalize_prompt(prompt)
        if replacement_count:
            record_event(
                root,
                "prompt_unicode_sanitized",
                session_id,
                turn_id,
                replacement_count=replacement_count,
            )
        try:
            append_requirement(root, prompt, session_id, turn_id)
        except Exception as exc:
            record_event(root, "requirement_failed", session_id, turn_id, **_error_fields(exc))
            print(f"shadow audit: failed to record requirement: {exc}", file=sys.stderr)
            print("{}")
            return 0
        record_event(root, "requirement_saved", session_id, turn_id)

        try:
            state_path = turn_state_file(root, session_id, turn_id)
        except Exception as exc:
            record_event(root, "turn_state_failed", session_id, turn_id, phase="path", **_error_fields(exc))
            print(f"shadow audit: failed to prepare turn state: {exc}", file=sys.stderr)
            print("{}")
            return 0

        try:
            existing = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        except Exception as exc:
            record_event(root, "turn_state_failed", session_id, turn_id, phase="read", **_error_fields(exc))
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("session_id") == session_id
            and existing.get("turn_id") == turn_id
            and isinstance(existing.get("baseline_fingerprint"), str)
            and existing["baseline_fingerprint"]
        ):
            record_event(root, "turn_state_saved", session_id, turn_id, action="reused")
            print("{}")
            return 0

        try:
            baseline_fingerprint = git_fingerprint(root, base_ref)
        except Exception as exc:
            record_event(root, "baseline_failed", session_id, turn_id, **_error_fields(exc))
            print(f"shadow audit: baseline fingerprint unavailable: {exc}", file=sys.stderr)
            print("{}")
            return 0
        record_event(root, "baseline_saved", session_id, turn_id)

        state = {
            "session_id": session_id,
            "turn_id": turn_id,
            "baseline_fingerprint": baseline_fingerprint,
            "captured_at": now_iso(),
        }
        try:
            write_json_atomic(state_path, state)
        except Exception as exc:
            record_event(root, "turn_state_failed", session_id, turn_id, phase="write", **_error_fields(exc))
            print(f"shadow audit: failed to record baseline state: {exc}", file=sys.stderr)
        else:
            record_event(root, "turn_state_saved", session_id, turn_id, action="written")
        print("{}")
        return 0

    if event_name == "Stop":
        if bool(event.get("stop_hook_active")):
            record_event(root, "stop", session_id, turn_id, reason="stop_reentry")
            print("{}")
            return 0
        if not cfg.get("final_audit_on_stop", True):
            record_event(root, "stop", session_id, turn_id, reason="final_audit_disabled")
            print("{}")
            return 0

        try:
            state_path = turn_state_file(root, session_id, turn_id)
            state_exists = state_path.exists()
        except Exception as exc:
            record_event(root, "turn_state_failed", session_id, turn_id, phase="Stop_path", **_error_fields(exc))
            record_event(root, "stop", session_id, turn_id, reason="state_missing", **_error_fields(exc))
            print("{}")
            return 0
        if not state_exists:
            record_event(root, "stop", session_id, turn_id, reason="state_missing")
            print("{}")
            return 0
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            record_event(root, "turn_state_failed", session_id, turn_id, phase="Stop_read", **_error_fields(exc))
            record_event(root, "stop", session_id, turn_id, reason="state_invalid", **_error_fields(exc))
            print("{}")
            return 0
        if (
            not isinstance(state, dict)
            or state.get("session_id") != session_id
            or state.get("turn_id") != turn_id
            or not isinstance(state.get("baseline_fingerprint"), str)
            or not state["baseline_fingerprint"]
        ):
            exc = ValueError("turn state is missing a valid session, turn, or baseline fingerprint")
            record_event(root, "turn_state_failed", session_id, turn_id, phase="Stop_validate", **_error_fields(exc))
            record_event(root, "stop", session_id, turn_id, reason="state_invalid", **_error_fields(exc))
            print("{}")
            return 0
        try:
            current = git_fingerprint(root, base_ref)
        except Exception as exc:
            record_event(root, "stop", session_id, turn_id, reason="fingerprint_failed", **_error_fields(exc))
            print("{}")
            return 0
        if current == state.get("baseline_fingerprint"):
            record_event(root, "stop", session_id, turn_id, reason="fingerprint_unchanged")
            print("{}")
            return 0

        record_event(
            root,
            "audit_started",
            session_id,
            turn_id,
            final=True,
            auditors=configured_auditor_summary(cfg),
        )
        try:
            reusable, reuse_details = _checkpoint_reusable_result(root, current, cfg)
            if reusable is not None:
                record_event(
                    root,
                    "code_smell_reused",
                    session_id,
                    turn_id,
                    fingerprint=current,
                    auditor=CHECKPOINT_AUDITOR,
                    status=reusable.get("status"),
                    checkpoint_timestamp=reusable.get("checkpoint_timestamp"),
                )
                cycle = audit_cycle(
                    root,
                    final=True,
                    session_id=session_id,
                    turn_id=turn_id,
                    skip_auditors={CHECKPOINT_AUDITOR},
                    precomputed_results=[reusable],
                )
            else:
                rerun_details: dict[str, Any] = {
                    "fingerprint": current,
                    "auditor": CHECKPOINT_AUDITOR,
                }
                rerun_details.update(reuse_details)
                record_event(root, "code_smell_rerun", session_id, turn_id, **rerun_details)
                cycle = audit_cycle(root, final=True, session_id=session_id, turn_id=turn_id)
            issues = cycle.get("issues", [])
        except Exception as exc:
            record_event(root, "audit_finished", session_id, turn_id, final=True, status="error", **_error_fields(exc))
            record_event(root, "stop", session_id, turn_id, reason="audit_failed", **_error_fields(exc))
            raise
        record_event(
            root,
            "audit_finished",
            session_id,
            turn_id,
            final=True,
            status=cycle.get("status", "ok"),
            issue_count=len(issues) if isinstance(issues, list) else None,
            auditors=auditor_event_summary(cycle.get("reports", [])),
        )
        if not issues:
            record_event(root, "stop", session_id, turn_id, reason="audit_pass")
            print("{}")
            return 0
        report_text = []
        for item in issues:
            report_text.append(f"[{item['auditor']}]\n{item['report'].strip()}")
        reason = "SHADOW_AUDIT_FEEDBACK:\n" + "\n\n".join(report_text)
        # Stop decision:block asks the main Codex turn to continue with this feedback.
        record_event(root, "stop", session_id, turn_id, reason="audit_block", issue_count=len(issues))
        print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
        return 0

    print("{}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Independent Codex shadow auditors for another code-writing process")
    p.add_argument("--repo", default=".", help="Path inside the Git repository")
    sub = p.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Initialize the requirement contract")
    init_p.add_argument("--requirements", help="Requirement text")
    init_p.add_argument("--requirements-file", help="File containing the original requirements")
    init_p.add_argument("--force", action="store_true", help="Replace existing requirement log")

    watch_p = sub.add_parser("watch", help="Watch Git changes and probabilistically run auditors")
    watch_p.add_argument("--once", action="store_true", help="Exit after the first stable checkpoint")

    audit_p = sub.add_parser("audit", help="Run auditors immediately")
    audit_p.add_argument("--final", action="store_true", help="Force every enabled auditor to run")
    audit_p.add_argument("--seed", type=int, help="Random seed for probabilistic checkpoint selection")

    sub.add_parser("hook", help="Read a Codex hook event JSON from stdin")
    sub.add_parser("checkpoint-hook", help="Run a non-blocking apply_patch checkpoint from hook JSON")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = repo_root(args.repo)
    if args.command == "init":
        path = init_requirements(root, args.requirements, args.requirements_file, args.force)
        print(path)
        return 0
    if args.command == "watch":
        watch(root, args.once)
        return 0
    if args.command == "audit":
        cycle = audit_cycle(root, final=args.final, seed=args.seed)
        print_cycle(cycle)
        return 1 if cycle.get("issues") else 0
    if args.command == "hook":
        return hook_main(root)
    if args.command == "checkpoint-hook":
        return checkpoint_hook_main(root)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
