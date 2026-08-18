import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".agents" / "skills" / "shadow-code-audit" / "scripts" / "shadow_audit.py"
SPEC = importlib.util.spec_from_file_location("shadow_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
shadow_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shadow_audit)


class ShadowAuditHookTests(unittest.TestCase):
    @staticmethod
    def _events(root):
        path = root / ".codex-shadow" / "events.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def _run_hook(root, event, stdout=None, stderr=None):
        stdout = stdout or io.StringIO()
        stderr = stderr or io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(event))), redirect_stdout(stdout), redirect_stderr(stderr):
            result = shadow_audit.hook_main(root)
        return result, stdout, stderr

    @staticmethod
    def _run_checkpoint_hook(root, event):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(event))), redirect_stdout(stdout), redirect_stderr(stderr):
            result = shadow_audit.checkpoint_hook_main(root)
        return result, stdout, stderr

    def test_hooks_register_async_apply_patch_checkpoint(self):
        config = json.loads((ROOT / ".codex" / "hooks.json").read_text(encoding="utf-8"))
        groups = config["hooks"]["PostToolUse"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["matcher"], "apply_patch")
        handler = groups[0]["hooks"][0]
        self.assertEqual(handler["type"], "command")
        self.assertTrue(handler["async"])
        self.assertIn("checkpoint-hook", handler["command"])

    def test_checkpoint_obeys_debounce_interval_and_fingerprint_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "PostToolUse",
                "session_id": "session-checkpoint",
                "turn_id": "turn-checkpoint",
                "tool_name": "apply_patch",
            }
            clock = [1000.0]
            fingerprints = ["fp-1", "fp-1", "fp-1", "fp-2", "fp-2", "fp-2"]

            def fake_run_one(root_arg, auditor, cfg, base_ref, final, report_dir):
                self.assertEqual(auditor, "code-smell-auditor")
                self.assertEqual(cfg["model"], "gpt-5.6-luna")
                self.assertEqual(cfg["reasoning_effort"], "high")
                self.assertFalse(final)
                return {
                    "auditor": auditor,
                    "model": cfg["model"],
                    "reasoning_effort": cfg["reasoning_effort"],
                    "duration_seconds": 0.012,
                    "status": "ok",
                    "report": "NO_FINDING",
                }

            with patch.object(shadow_audit, "git_fingerprint", side_effect=fingerprints), patch.object(
                shadow_audit.time, "time", side_effect=lambda: clock[0]
            ), patch.object(shadow_audit.random, "random", side_effect=[0.1, 0.1]), patch.object(
                shadow_audit, "run_one_auditor", side_effect=fake_run_one
            ) as run_one:
                result, stdout, _ = self._run_checkpoint_hook(root, event)
                self.assertEqual((result, stdout.getvalue()), (0, "{}\n"))

                clock[0] = 1008.0
                self._run_checkpoint_hook(root, event)

                clock[0] = 1009.0
                self._run_checkpoint_hook(root, event)

                clock[0] = 1010.0
                self._run_checkpoint_hook(root, event)

                clock[0] = 1018.0
                self._run_checkpoint_hook(root, event)

                clock[0] = 1098.0
                self._run_checkpoint_hook(root, event)

            self.assertEqual(run_one.call_count, 2)
            self.assertEqual(
                [call.args[1] for call in run_one.call_args_list],
                ["code-smell-auditor", "code-smell-auditor"],
            )
            events = self._events(root)
            candidates = [item for item in events if item["event"] == "checkpoint_candidate"]
            self.assertEqual([item["fingerprint"] for item in candidates], ["fp-1", "fp-2"])
            skipped = [item for item in events if item["event"] == "checkpoint_skipped"]
            self.assertEqual(
                [item["reason"] for item in skipped],
                ["debounce", "fingerprint_unchanged", "debounce", "minimum_interval"],
            )
            started = [item for item in events if item["event"] == "checkpoint_started"]
            finished = [item for item in events if item["event"] == "checkpoint_finished"]
            self.assertEqual(len(started), 2)
            self.assertEqual(len(finished), 2)
            self.assertTrue(all(item["model"] == "gpt-5.6-luna" for item in started + finished))
            self.assertTrue(all(item["reasoning_effort"] == "high" for item in started + finished))
            self.assertTrue(all(item["status"] == "ok" for item in finished))
            checkpoint_state = json.loads(
                (root / ".codex-shadow" / "checkpoint-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint_state["audited_fingerprint"], "fp-2")
            self.assertEqual(checkpoint_state["auditor"], "code-smell-auditor")
            self.assertEqual(checkpoint_state["status"], "ok")
            self.assertTrue(checkpoint_state["audited_pass"])
            self.assertIsInstance(checkpoint_state["timestamp"], str)

    def test_checkpoint_probability_skip_is_non_blocking_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "PostToolUse",
                "session_id": "session-probability",
                "turn_id": "turn-probability",
                "tool_name": "apply_patch",
            }
            clock = [2000.0]
            with patch.object(shadow_audit, "git_fingerprint", return_value="fp-prob"), patch.object(
                shadow_audit.time, "time", side_effect=lambda: clock[0]
            ), patch.object(shadow_audit.random, "random", return_value=0.9), patch.object(
                shadow_audit, "run_one_auditor"
            ) as run_one:
                self._run_checkpoint_hook(root, event)
                clock[0] = 2008.0
                result, stdout, _ = self._run_checkpoint_hook(root, event)
                clock[0] = 2009.0
                self._run_checkpoint_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            run_one.assert_not_called()
            skipped = [item for item in self._events(root) if item["event"] == "checkpoint_skipped"]
            self.assertEqual([item["reason"] for item in skipped], ["debounce", "activation_probability", "fingerprint_unchanged"])
            self.assertEqual(skipped[1]["probability"], 0.35)
            self.assertEqual(skipped[1]["sample"], 0.9)

    def test_checkpoint_fingerprint_failure_does_not_block_main_codex(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "PostToolUse",
                "session_id": "session-checkpoint-error",
                "turn_id": "turn-checkpoint-error",
                "tool_name": "apply_patch",
            }
            with patch.object(shadow_audit, "git_fingerprint", side_effect=RuntimeError("git diff unavailable")):
                result, stdout, _ = self._run_checkpoint_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            failures = [item for item in self._events(root) if item["event"] == "checkpoint_skipped"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["reason"], "fingerprint_failed")
            self.assertEqual(failures[0]["error_type"], "RuntimeError")
            self.assertEqual(failures[0]["error"], "git diff unavailable")

    @staticmethod
    def _ready_checkpoint_state(root, fingerprint="fp-ready"):
        path = shadow_audit.checkpoint_state_file(root)
        path.write_text(
            json.dumps({"pending_fingerprint": fingerprint, "pending_since": 1000.0}),
            encoding="utf-8",
        )

    def test_checkpoint_pass_emits_no_additional_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._ready_checkpoint_state(root)
            event = {
                "hook_event_name": "PostToolUse",
                "session_id": "session-pass",
                "turn_id": "turn-pass",
                "tool_name": "apply_patch",
            }
            result_data = {
                "auditor": "code-smell-auditor",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "high",
                "duration_seconds": 0.01,
                "status": "ok",
                "report": "NO_FINDING",
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="fp-ready"), patch.object(
                shadow_audit.time, "time", return_value=1008.0
            ), patch.object(shadow_audit.random, "random", return_value=0.1), patch.object(
                shadow_audit, "run_one_auditor", return_value=result_data
            ):
                result, stdout, _ = self._run_checkpoint_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue()), {})
            self.assertNotIn(
                "checkpoint_feedback_emitted",
                [item["event"] for item in self._events(root)],
            )

    def test_checkpoint_finding_emits_non_blocking_post_tool_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._ready_checkpoint_state(root)
            event = {
                "hook_event_name": "PostToolUse",
                "session_id": "session-finding",
                "turn_id": "turn-finding",
                "tool_name": "apply_patch",
            }
            result_data = {
                "auditor": "code-smell-auditor",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "high",
                "duration_seconds": 0.01,
                "status": "ok",
                "report": "FINDING\nseverity: high\nfile: calibration_tool/a.py:10\nproblem: duplicated validation\nfix: extract the shared helper",
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="fp-ready"), patch.object(
                shadow_audit.time, "time", return_value=1008.0
            ), patch.object(shadow_audit.random, "random", return_value=0.1), patch.object(
                shadow_audit, "run_one_auditor", return_value=result_data
            ):
                result, stdout, _ = self._run_checkpoint_hook(root, event)

            self.assertEqual(result, 0)
            response = json.loads(stdout.getvalue())
            self.assertEqual(response.keys(), {"hookSpecificOutput"})
            output = response["hookSpecificOutput"]
            self.assertEqual(output["hookEventName"], "PostToolUse")
            self.assertIn("additionalContext", output)
            self.assertNotIn("decision", response)
            self.assertNotIn("continue", response)
            context = output["additionalContext"]
            self.assertTrue(context.startswith("🔴 shadow · [代码与项目结构审计者 / code-smell-auditor]\n"))
            self.assertIn("severity: high", context)
            self.assertIn("calibration_tool/a.py:10", context)
            self.assertLessEqual(len(context.splitlines()) - 1, 4)
            feedback = [item for item in self._events(root) if item["event"] == "checkpoint_feedback_emitted"]
            self.assertEqual(len(feedback), 1)
            self.assertEqual(feedback[0]["finding_count"], 1)

    def test_checkpoint_finding_returns_only_highest_value_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._ready_checkpoint_state(root)
            event = {
                "hook_event_name": "PostToolUse",
                "session_id": "session-one-finding",
                "turn_id": "turn-one-finding",
                "tool_name": "apply_patch",
            }
            result_data = {
                "auditor": "code-smell-auditor",
                "model": "gpt-5.6-luna",
                "reasoning_effort": "high",
                "duration_seconds": 0.01,
                "status": "ok",
                "report": (
                    "FINDING 1:\nseverity: low\nfile: low.py:1\nproblem: low value\n\n"
                    "FINDING 2:\nseverity: high\nfile: high.py:2\nproblem: high value\n"
                ),
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="fp-ready"), patch.object(
                shadow_audit.time, "time", return_value=1008.0
            ), patch.object(shadow_audit.random, "random", return_value=0.1), patch.object(
                shadow_audit, "run_one_auditor", return_value=result_data
            ):
                _, stdout, _ = self._run_checkpoint_hook(root, event)

            context = json.loads(stdout.getvalue())["hookSpecificOutput"]["additionalContext"]
            self.assertIn("high.py:2", context)
            self.assertNotIn("low.py:1", context)
            self.assertEqual(context.count("file:"), 1)

    def test_untracked_fingerprint_uses_metadata_without_reading_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            large_file = root / "large.bin"
            large_file.write_bytes(b"x" * (8 * 1024 * 1024))

            def fake_run(command, cwd, timeout=None):
                if command[1] == "diff":
                    return subprocess.CompletedProcess(command, 0, stdout="tracked diff", stderr="")
                if command[1] == "ls-files":
                    return subprocess.CompletedProcess(command, 0, stdout="large.bin\0", stderr="")
                raise AssertionError(command)

            with patch.object(shadow_audit, "run", side_effect=fake_run), patch.object(
                Path, "open", side_effect=AssertionError("untracked content was read")
            ):
                first = shadow_audit.git_fingerprint(root, "HEAD")

            large_file.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
            with patch.object(shadow_audit, "run", side_effect=fake_run), patch.object(
                Path, "open", side_effect=AssertionError("untracked content was read")
            ):
                second = shadow_audit.git_fingerprint(root, "HEAD")

            self.assertNotEqual(first, second)

    def test_git_fingerprint_reports_diff_failure_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_run(command, cwd, timeout=None):
                self.assertEqual(command[1], "diff")
                return subprocess.CompletedProcess(
                    command,
                    128,
                    stdout="",
                    stderr="fatal: bad revision 'HEAD'",
                )

            with patch.object(shadow_audit, "run", side_effect=fake_run) as run_mock:
                with self.assertRaisesRegex(RuntimeError, "returncode=128.*fatal: bad revision 'HEAD'") as raised:
                    shadow_audit.git_fingerprint(root, "HEAD")

            self.assertIn("git diff", str(raised.exception))
            self.assertEqual(run_mock.call_count, 1)

    def test_git_fingerprint_does_not_mask_missing_diff_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_run(command, cwd, timeout=None):
                return subprocess.CompletedProcess(command, 128, stdout="stdout diagnostic", stderr=None)

            with patch.object(shadow_audit, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "returncode=128.*stdout diagnostic") as raised:
                    shadow_audit.git_fingerprint(root, "HEAD")

            self.assertNotIn("NoneType", str(raised.exception))

    def test_git_fingerprint_reports_ls_files_failure_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_run(command, cwd, timeout=None):
                if command[1] == "diff":
                    return subprocess.CompletedProcess(command, 0, stdout="tracked diff", stderr="")
                self.assertEqual(command[1], "ls-files")
                return subprocess.CompletedProcess(command, 2, stdout="", stderr="fatal: repository unavailable")

            with patch.object(shadow_audit, "run", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "git ls-files.*returncode=2.*repository unavailable"):
                    shadow_audit.git_fingerprint(root, "HEAD")

    def test_run_one_auditor_passes_explicit_model_and_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_dir = root / "reports"
            report_dir.mkdir()
            commands = []

            def fake_subprocess_run(command, **kwargs):
                commands.append((command, kwargs))
                output_path = Path(command[command.index("-o") + 1])
                output_path.write_text("NO_FINDING\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            auditor_configs = [
                (
                    "code-smell-auditor",
                    {"model": "gpt-5.6-luna", "reasoning_effort": "high", "timeout_seconds": 120},
                ),
                (
                    "success-goal-auditor",
                    {"model": "gpt-5.6-luna", "reasoning_effort": "xhigh", "timeout_seconds": 180},
                ),
            ]
            with patch.object(shadow_audit, "codex_executable", return_value="codex"), patch.object(
                shadow_audit, "auditor_prompt", return_value="audit prompt"
            ), patch.object(shadow_audit.subprocess, "run", side_effect=fake_subprocess_run):
                results = [
                    shadow_audit.run_one_auditor(root, auditor, cfg, "HEAD", True, report_dir)
                    for auditor, cfg in auditor_configs
                ]

            self.assertEqual(len(commands), 2)
            for (auditor, cfg), result, (command, kwargs) in zip(auditor_configs, results, commands):
                self.assertIn("--ephemeral", command)
                self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
                self.assertEqual(command[command.index("--model") + 1], cfg["model"])
                self.assertEqual(
                    command[command.index("--config") + 1],
                    f"model_reasoning_effort={cfg['reasoning_effort']}",
                )
                self.assertEqual(kwargs["timeout"], cfg["timeout_seconds"])
                self.assertEqual(result["model"], cfg["model"])
                self.assertEqual(result["reasoning_effort"], cfg["reasoning_effort"])
                self.assertEqual(result["duration_seconds"] >= 0, True)
                self.assertEqual(result["status"], "ok")

    def test_final_audit_forces_both_enabled_auditors_and_records_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "enabled": True,
                "base_ref": "HEAD",
                "max_parallel_auditors": 2,
                "block_on_audit_error": False,
                "auditors": {
                    "code-smell-auditor": {
                        "enabled": True,
                        "activation_probability": 0.0,
                        "model": "gpt-5.6-luna",
                        "reasoning_effort": "high",
                        "timeout_seconds": 120,
                    },
                    "success-goal-auditor": {
                        "enabled": True,
                        "activation_probability": 0.0,
                        "model": "gpt-5.6-luna",
                        "reasoning_effort": "xhigh",
                        "timeout_seconds": 180,
                    },
                },
            }

            def fake_run_one(root_arg, auditor, cfg, base_ref, final, report_dir):
                return {
                    "auditor": auditor,
                    "model": cfg["model"],
                    "reasoning_effort": cfg["reasoning_effort"],
                    "timeout_seconds": cfg["timeout_seconds"],
                    "duration_seconds": 0.012,
                    "status": "ok",
                    "report": shadow_audit.PASS_SENTINELS[auditor],
                }

            with patch.object(shadow_audit, "load_config", return_value=config), patch.object(
                shadow_audit, "git_fingerprint", return_value="fingerprint"
            ), patch.object(shadow_audit, "run_one_auditor", side_effect=fake_run_one) as run_one:
                cycle = shadow_audit.audit_cycle(root, final=True, session_id="session", turn_id="turn")

            self.assertEqual(cycle["issues"], [])
            self.assertEqual(
                {call.args[1] for call in run_one.call_args_list},
                {"code-smell-auditor", "success-goal-auditor"},
            )
            self.assertTrue(all(call.args[4] is True for call in run_one.call_args_list))

            metadata = json.loads((Path(cycle["report_dir"]) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {(item["auditor"], item["reasoning_effort"]) for item in metadata["auditors"]},
                {("code-smell-auditor", "high"), ("success-goal-auditor", "xhigh")},
            )
            self.assertTrue(all(item["model"] == "gpt-5.6-luna" for item in metadata["auditors"]))
            self.assertTrue(all(item["status"] == "ok" for item in metadata["auditors"]))
            events = [item for item in self._events(root) if item["event"] == "auditor_finished"]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(item["session_id"] == "session" for item in events))
            self.assertTrue(all(item["turn_id"] == "turn" for item in events))
            self.assertTrue(all("duration_seconds" in item for item in events))

    def test_user_prompt_is_recorded_once_when_baseline_fingerprint_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "prompt": "Record this requirement exactly once",
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.object(shadow_audit, "git_fingerprint", side_effect=OSError("git unavailable")):
                with patch.object(sys, "stdin", io.StringIO(json.dumps(event))), redirect_stdout(stdout), redirect_stderr(stderr):
                    first = shadow_audit.hook_main(root)
                with patch.object(sys, "stdin", io.StringIO(json.dumps(event))), redirect_stdout(stdout), redirect_stderr(stderr):
                    second = shadow_audit.hook_main(root)

            self.assertEqual(first, 0)
            self.assertEqual(second, 0)
            requirements = (root / ".codex-shadow" / "requirements.md").read_text(encoding="utf-8")
            self.assertEqual(requirements.count(event["prompt"]), 1)
            self.assertIn("baseline fingerprint unavailable", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "{}\n{}\n")
            events = self._events(root)
            self.assertEqual([item["event"] for item in events], [
                "hook_received",
                "requirement_saved",
                "baseline_failed",
                "hook_received",
                "requirement_saved",
                "baseline_failed",
            ])
            baseline_failures = [item for item in events if item["event"] == "baseline_failed"]
            self.assertTrue(all(item["error_type"] == "OSError" for item in baseline_failures))
            self.assertTrue(all(item["error"] == "git unavailable" for item in baseline_failures))
            self.assertNotIn(event["prompt"], (root / ".codex-shadow" / "events.jsonl").read_text(encoding="utf-8"))

    def test_baseline_failed_event_keeps_git_error_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-git-error",
                "turn_id": "turn-git-error",
                "prompt": "keep the Git diagnostic",
            }
            git_error = RuntimeError(
                "git command failed: command=git diff HEAD -- .; returncode=128; stderr=fatal: bad revision"
            )
            with patch.object(shadow_audit, "git_fingerprint", side_effect=git_error):
                result, stdout, _ = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            failures = [item for item in self._events(root) if item["event"] == "baseline_failed"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["error_type"], "RuntimeError")
            self.assertIn("returncode=128", failures[0]["error"])
            self.assertIn("fatal: bad revision", failures[0]["error"])

    def test_normal_unicode_prompt_is_written_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-unicode",
                "turn_id": "turn-normal",
                "prompt": "请保留这些中文字符：激光标定",
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="baseline-normal"):
                result, stdout, _ = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            requirements = (root / ".codex-shadow" / "requirements.md").read_text(encoding="utf-8")
            self.assertIn(event["prompt"], requirements)
            events_text = (root / ".codex-shadow" / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("prompt_unicode_sanitized", events_text)

    def test_isolated_surrogate_is_sanitized_and_hook_saves_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = "中文前缀 " + chr(0xDCBA) + " 中文后缀"
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-surrogate",
                "turn_id": "turn-surrogate",
                "prompt": prompt,
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="baseline-surrogate"):
                result, stdout, _ = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            requirements_path = root / ".codex-shadow" / "requirements.md"
            requirements_bytes = requirements_path.read_bytes()
            requirements = requirements_bytes.decode("utf-8")
            self.assertIn("中文前缀 \\udcba 中文后缀", requirements)
            self.assertNotIn(chr(0xDCBA), requirements)
            self.assertEqual(requirements.encode("utf-8"), requirements_bytes)

            state_path = shadow_audit.turn_state_file(root, event["session_id"], event["turn_id"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["baseline_fingerprint"], "baseline-surrogate")

            events = self._events(root)
            sanitized = [item for item in events if item["event"] == "prompt_unicode_sanitized"]
            self.assertEqual(len(sanitized), 1)
            self.assertEqual(sanitized[0]["replacement_count"], 1)
            events_text = (root / ".codex-shadow" / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn(prompt, events_text)
            self.assertIn("turn_state_saved", [item["event"] for item in events])

    def test_user_prompt_requirement_write_failure_is_non_blocking_and_persistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-requirement-failure",
                "turn_id": "turn-1",
                "prompt": "requirement should not block the main turn",
            }
            with patch.object(shadow_audit, "append_requirement", side_effect=OSError("requirements unavailable")) as save_requirement, patch.object(
                shadow_audit, "git_fingerprint"
            ) as fingerprint:
                result, stdout, stderr = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            self.assertIn("failed to record requirement", stderr.getvalue())
            save_requirement.assert_called_once()
            fingerprint.assert_not_called()
            events = self._events(root)
            self.assertEqual([item["event"] for item in events], ["hook_received", "requirement_failed"])
            failure = events[-1]
            self.assertEqual(failure["error_type"], "OSError")
            self.assertEqual(failure["error"], "requirements unavailable")
            self.assertNotIn(event["prompt"], (root / ".codex-shadow" / "events.jsonl").read_text(encoding="utf-8"))

    def test_user_prompt_state_write_failure_is_non_blocking_and_persistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "session-state-failure",
                "turn_id": "turn-1",
                "prompt": "state write failure is diagnostic",
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="baseline-1"), patch.object(
                shadow_audit, "write_json_atomic", side_effect=OSError("state disk full")
            ):
                result, stdout, stderr = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            self.assertIn("failed to record baseline state", stderr.getvalue())
            events = self._events(root)
            self.assertEqual([item["event"] for item in events], [
                "hook_received",
                "requirement_saved",
                "baseline_saved",
                "turn_state_failed",
            ])
            failure = events[-1]
            self.assertEqual(failure["phase"], "write")
            self.assertEqual(failure["error_type"], "OSError")
            self.assertEqual(failure["error"], "state disk full")

    def test_stop_missing_state_is_persistently_explained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = {
                "hook_event_name": "Stop",
                "session_id": "session-missing-state",
                "turn_id": "turn-1",
            }
            with patch.object(shadow_audit, "audit_cycle") as audit:
                result, stdout, _ = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            audit.assert_not_called()
            events = self._events(root)
            self.assertEqual([item["event"] for item in events], ["hook_received", "stop"])
            self.assertEqual(events[-1]["reason"], "state_missing")

    def test_stop_with_change_enters_final_audit_and_preserves_block_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_id = "session-final-audit"
            turn_id = "turn-1"
            state_path = shadow_audit.turn_state_file(root, session_id, turn_id)
            state_path.write_text(
                json.dumps({
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "baseline_fingerprint": "baseline-1",
                }),
                encoding="utf-8",
            )
            event = {
                "hook_event_name": "Stop",
                "session_id": session_id,
                "turn_id": turn_id,
            }
            cycle = {
                "status": "ok",
                "issues": [{"auditor": "code-smell-auditor", "report": "finding text"}],
            }
            with patch.object(shadow_audit, "git_fingerprint", return_value="current-2") as fingerprint, patch.object(
                shadow_audit, "audit_cycle", return_value=cycle
            ) as audit:
                result, stdout, _ = self._run_hook(root, event)

            self.assertEqual(result, 0)
            response = json.loads(stdout.getvalue())
            self.assertEqual(response["decision"], "block")
            self.assertIn("SHADOW_AUDIT_FEEDBACK:", response["reason"])
            self.assertIn("[code-smell-auditor]", response["reason"])
            self.assertIn("finding text", response["reason"])
            fingerprint.assert_called_once_with(root, "HEAD")
            audit.assert_called_once_with(root, final=True, session_id=session_id, turn_id=turn_id)
            events = self._events(root)
            self.assertEqual([item["event"] for item in events], [
                "hook_received",
                "audit_started",
                "code_smell_rerun",
                "audit_finished",
                "stop",
            ])
            self.assertTrue(events[1]["final"])
            self.assertEqual(events[2]["reason"], "checkpoint_state_missing")
            self.assertEqual(events[3]["status"], "ok")
            self.assertEqual(events[3]["issue_count"], 1)
            self.assertEqual(events[4]["reason"], "audit_block")

    def test_stop_reuses_checkpoint_pass_and_still_forces_success_goal_auditor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_id = "session-reuse"
            turn_id = "turn-reuse"
            state_path = shadow_audit.turn_state_file(root, session_id, turn_id)
            state_path.write_text(
                json.dumps({
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "baseline_fingerprint": "baseline-1",
                }),
                encoding="utf-8",
            )
            shadow_audit.checkpoint_state_file(root).write_text(
                json.dumps({
                    "audited_fingerprint": "current-2",
                    "auditor": "code-smell-auditor",
                    "status": "ok",
                    "timestamp": "2026-08-18T12:00:00+08:00",
                    "audited_pass": True,
                    "audited_model": "gpt-5.6-luna",
                    "audited_reasoning_effort": "high",
                    "audited_duration_seconds": 0.25,
                }),
                encoding="utf-8",
            )
            event = {
                "hook_event_name": "Stop",
                "session_id": session_id,
                "turn_id": turn_id,
            }

            def fake_run_one(root_arg, auditor, cfg, base_ref, final, report_dir):
                self.assertEqual(auditor, "success-goal-auditor")
                self.assertTrue(final)
                return {
                    "auditor": auditor,
                    "model": cfg["model"],
                    "reasoning_effort": cfg["reasoning_effort"],
                    "duration_seconds": 0.1,
                    "status": "ok",
                    "report": "GOAL_ALIGNED",
                }

            with patch.object(shadow_audit, "git_fingerprint", return_value="current-2"), patch.object(
                shadow_audit, "run_one_auditor", side_effect=fake_run_one
            ) as run_one:
                result, stdout, _ = self._run_hook(root, event)

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "{}\n")
            self.assertEqual([call.args[1] for call in run_one.call_args_list], ["success-goal-auditor"])
            events = self._events(root)
            self.assertIn("code_smell_reused", [item["event"] for item in events])
            self.assertNotIn("code_smell_rerun", [item["event"] for item in events])
            finished = next(item for item in events if item["event"] == "audit_finished")
            self.assertEqual(
                {item["auditor"] for item in finished["auditors"]},
                {"code-smell-auditor", "success-goal-auditor"},
            )
            self.assertEqual(next(item for item in events if item["event"] == "stop")["reason"], "audit_pass")

    def test_checkpoint_finding_failure_and_timeout_are_never_reused_as_pass(self):
        cases = [
            ("ok", False),
            ("error", False),
            ("timeout", False),
        ]
        for status, audited_pass in cases:
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    session_id = f"session-no-reuse-{status}"
                    turn_id = "turn-1"
                    state_path = shadow_audit.turn_state_file(root, session_id, turn_id)
                    state_path.write_text(
                        json.dumps({
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "baseline_fingerprint": "baseline-1",
                        }),
                        encoding="utf-8",
                    )
                    shadow_audit.checkpoint_state_file(root).write_text(
                        json.dumps({
                            "audited_fingerprint": "current-2",
                            "auditor": "code-smell-auditor",
                            "status": status,
                            "timestamp": "2026-08-18T12:00:00+08:00",
                            "audited_pass": audited_pass,
                        }),
                        encoding="utf-8",
                    )
                    event = {
                        "hook_event_name": "Stop",
                        "session_id": session_id,
                        "turn_id": turn_id,
                    }
                    cycle = {"status": "ok", "reports": [], "issues": []}
                    with patch.object(shadow_audit, "git_fingerprint", return_value="current-2"), patch.object(
                        shadow_audit, "audit_cycle", return_value=cycle
                    ) as audit:
                        result, stdout, _ = self._run_hook(root, event)

                    self.assertEqual(result, 0)
                    self.assertEqual(stdout.getvalue(), "{}\n")
                    audit.assert_called_once_with(root, final=True, session_id=session_id, turn_id=turn_id)
                    events = self._events(root)
                    self.assertNotIn("code_smell_reused", [item["event"] for item in events])
                    rerun = next(item for item in events if item["event"] == "code_smell_rerun")
                    self.assertIn(rerun["reason"], {"checkpoint_not_pass", "checkpoint_not_successful"})


if __name__ == "__main__":
    unittest.main()
