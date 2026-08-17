#!/usr/bin/env python3
"""Cross-process Codex shadow auditor.

Modes:
  init   - capture the requirement contract used by success-goal-auditor
  watch  - watch a Git worktree and probabilistically run read-only auditors
  audit  - run an audit immediately (with --final to force both auditors)
  hook   - Codex hook bridge for UserPromptSubmit + Stop

Uses only the Python standard library. Reviewer processes are fresh, ephemeral,
and read-only Codex exec invocations.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import random
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
    "auditors": {
        "code-smell-auditor": {
            "enabled": True,
            "activation_probability": 0.4,
            "timeout_seconds": 120,
        },
        "success-goal-auditor": {
            "enabled": True,
            "activation_probability": 0.4,
            "timeout_seconds": 120,
        },
    },
}

PASS_SENTINELS = {
    "code-smell-auditor": "NO_FINDING",
    "success-goal-auditor": "GOAL_ALIGNED",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run(cmd: list[str], cwd: pathlib.Path, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
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


def git_fingerprint(root: pathlib.Path, base_ref: str) -> str:
    """Hash tracked diff plus untracked file contents (excluding runtime state)."""
    h = hashlib.sha256()
    cp = run(["git", "diff", "--binary", "--no-ext-diff", base_ref, "--", "."], root)
    h.update(cp.stdout.encode("utf-8", errors="replace"))
    if cp.returncode != 0:
        h.update(cp.stderr.encode("utf-8", errors="replace"))

    untracked = run(["git", "ls-files", "--others", "--exclude-standard", "-z"], root)
    for rel in sorted(x for x in untracked.stdout.split("\0") if x):
        if rel == ".codex-shadow" or rel.startswith(".codex-shadow/"):
            continue
        h.update(rel.encode("utf-8", errors="replace"))
        p = root / rel
        try:
            if p.is_file():
                with p.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
        except OSError as exc:
            h.update(f"<unreadable:{exc}>".encode())
    return h.hexdigest()


def changed_summary(root: pathlib.Path, base_ref: str) -> str:
    status = run(["git", "status", "--short"], root).stdout.strip()
    stat = run(["git", "diff", "--stat", base_ref, "--", "."], root).stdout.strip()
    return f"BASE_REF: {base_ref}\nSTATUS:\n{status or '(clean)'}\nDIFF_STAT:\n{stat or '(no tracked diff)'}"


def requirements_path(root: pathlib.Path) -> pathlib.Path:
    return state_dir(root) / "requirements.md"


def append_requirement(root: pathlib.Path, prompt: str, session_id: str | None = None, turn_id: str | None = None) -> None:
    if not prompt.strip() or prompt.startswith("SHADOW_AUDIT_FEEDBACK:"):
        return
    path = requirements_path(root)
    header = f"\n\n## Requirement update — {now_iso()}"
    if session_id or turn_id:
        header += f"\n<!-- session={session_id or ''} turn={turn_id or ''} -->"
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


def run_one_auditor(root: pathlib.Path, auditor: str, cfg: dict[str, Any], base_ref: str, final: bool, report_dir: pathlib.Path) -> dict[str, Any]:
    timeout = float(cfg.get("timeout_seconds", 120))
    output_path = report_dir / f"{auditor}.md"
    stderr_path = report_dir / f"{auditor}.stderr.log"
    prompt = auditor_prompt(root, auditor, base_ref, final)
    started = time.time()
    result: dict[str, Any] = {"auditor": auditor, "started_at": now_iso(), "timeout_seconds": timeout}
    try:
        exe = codex_executable()
        env = os.environ.copy()
        env["CODEX_SHADOW_AUDIT_CHILD"] = "1"
        cp = subprocess.run(
            [exe, "exec", "--ephemeral", "--sandbox", "read-only", "-o", str(output_path), prompt],
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


def audit_cycle(root: pathlib.Path, final: bool = False, seed: int | None = None) -> dict[str, Any]:
    cfg = load_config(root)
    if not cfg.get("enabled", True):
        return {"status": "disabled", "reports": []}
    base_ref = str(cfg.get("base_ref", "HEAD"))
    rng = random.Random(seed)
    audit_cfgs: dict[str, Any] = cfg.get("auditors", {})
    selected = [name for name, acfg in audit_cfgs.items() if should_activate(acfg, final, rng)]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = state_dir(root) / "reports" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    max_workers = max(1, min(int(cfg.get("max_parallel_auditors", 2)), len(selected) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_one_auditor, root, name, audit_cfgs[name], base_ref, final, out_dir): name
            for name in selected
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r["auditor"])

    block_on_error = bool(cfg.get("block_on_audit_error", False))
    issues = [r for r in results if report_is_issue(r["auditor"], r["report"], block_on_error)]
    metadata = {
        "created_at": now_iso(),
        "final": final,
        "base_ref": base_ref,
        "fingerprint": git_fingerprint(root, base_ref),
        "selected": selected,
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


def turn_state_file(root: pathlib.Path, session_id: str, turn_id: str) -> pathlib.Path:
    safe = hashlib.sha256(f"{session_id}:{turn_id}".encode()).hexdigest()[:20]
    p = state_dir(root) / "turns"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe}.json"


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

    cfg = load_config(root)
    if not cfg.get("enabled", True):
        print("{}")
        return 0
    event_name = event.get("hook_event_name")
    session_id = str(event.get("session_id") or "unknown-session")
    turn_id = str(event.get("turn_id") or "unknown-turn")
    base_ref = str(cfg.get("base_ref", "HEAD"))

    if event_name == "UserPromptSubmit":
        prompt = str(event.get("prompt") or "")
        append_requirement(root, prompt, session_id, turn_id)
        state = {
            "session_id": session_id,
            "turn_id": turn_id,
            "baseline_fingerprint": git_fingerprint(root, base_ref),
            "captured_at": now_iso(),
        }
        turn_state_file(root, session_id, turn_id).write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("{}")
        return 0

    if event_name == "Stop":
        if bool(event.get("stop_hook_active")):
            print("{}")
            return 0
        if not cfg.get("final_audit_on_stop", True):
            print("{}")
            return 0
        state_path = turn_state_file(root, session_id, turn_id)
        if not state_path.exists():
            print("{}")
            return 0
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            print("{}")
            return 0
        current = git_fingerprint(root, base_ref)
        if current == state.get("baseline_fingerprint"):
            print("{}")
            return 0

        cycle = audit_cycle(root, final=True)
        issues = cycle.get("issues", [])
        if not issues:
            print("{}")
            return 0
        report_text = []
        for item in issues:
            report_text.append(f"[{item['auditor']}]\n{item['report'].strip()}")
        reason = "SHADOW_AUDIT_FEEDBACK:\n" + "\n\n".join(report_text)
        # Stop decision:block asks the main Codex turn to continue with this feedback.
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
