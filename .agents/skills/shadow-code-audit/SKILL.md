---
name: shadow-code-audit
description: Orchestrate two independent read-only Codex reviewers around active code work: code-smell-auditor checks maintainability/project structure, and success-goal-auditor checks the implementation against the original request. Use for checkpoint or final audits while another agent is implementing, or to set up the optional cross-process watcher.
---

# Shadow code audit orchestrator

Use two independent review lanes. Do not collapse them into one generic reviewer:

- `code_smell_auditor`: code internals, module boundaries, directory organization, repository hygiene.
- `success_goal_auditor`: original user goal, explicit follow-up constraints, acceptance criteria, and evidence of completion.

Both reviewers are read-only and must not implement fixes themselves.

## Native Codex checkpoint mode

When this skill is invoked during an implementation task:

1. Identify the current change scope from the parent thread: original request, explicit follow-up constraints, changed paths, baseline/ref if known, and the current implementation status.
2. Preserve the user's wording for hard requirements. Do not summarize away numeric thresholds, defaults, file/path constraints, compatibility promises, or “must / must not” clauses.
3. Spawn **two subagents in parallel**, using the project custom-agent types `code_smell_auditor` and `success_goal_auditor` when available.
4. Give both agents the relevant change scope. Give the success-goal agent the original requirement contract and later explicit changes verbatim or as a clearly delimited requirement log.
5. Tell both agents to inspect the repository independently rather than trust implementation claims from the parent.
6. Wait for both results before deciding whether the checkpoint passes.
7. Treat `NO_FINDING` and `GOAL_ALIGNED` as silence/pass. Surface only concrete findings.
8. If a reviewer reports a material issue and the parent task includes implementation, the **parent agent** owns the fix. Do not delegate the fix back to the auditor.
9. After a material fix, re-run only the auditor whose finding could have changed, unless the fix materially changed both structure and requirements.

## Final-completion gate

Before the parent declares a code-changing task complete, run both reviewers regardless of whether an earlier checkpoint passed.

Do not claim completion while either reviewer has an unresolved `FINDING`, `GOAL_MISMATCH`, or `AUDIT_BLOCKED` that is material to the requested task. Fix or obtain the missing evidence, then re-audit the affected lane.

## Cross-process mode

For a truly separate reviewer process, use `scripts/shadow_audit.py` from this skill. The watcher deliberately does **not** consume another agent's private reasoning or unstable transcript format. It observes:

- the Git working tree and diff,
- a requirement log/file,
- the real repository files,
- optional test/config evidence the auditor can inspect.

Each activation launches a fresh `codex exec --ephemeral --sandbox read-only` process and explicitly invokes one auditor skill. This gives each auditor an independent evidence chain and prevents it from editing the implementation.

Useful commands after installation in a repository:

```text
python .agents/skills/shadow-code-audit/scripts/shadow_audit.py init --requirements-file REQUIREMENTS.md
python .agents/skills/shadow-code-audit/scripts/shadow_audit.py watch
python .agents/skills/shadow-code-audit/scripts/shadow_audit.py audit --final
```

On Windows, `py -3` can replace `python`.

The watcher writes runtime state and reports under `.codex-shadow/`. Keep that directory local/untracked.

## Pi Shadow Mind principles preserved

Preserve these behavioral properties:

- Independent reviewer context, not continuation of the implementer's unfinished work.
- Read-only by default.
- Fresh reviewer session for every activation.
- Relevance/silence behavior: no low-value report just because the reviewer ran.
- At most one highest-value finding from each auditor activation.
- Changed-code-first scope; expand only with evidence.
- Requirement review and code-structure review remain separate responsibilities.
- A final deterministic completion gate may be stricter than probabilistic mid-task checks.

Do not pretend the Skill system itself supplies Pi-style background heartbeats. Use Codex subagents for native parallel review, and use the companion watcher/Hook only when true cross-process triggering is required.
