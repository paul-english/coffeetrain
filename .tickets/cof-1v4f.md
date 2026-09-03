---
id: cof-1v4f
status: closed
deps: []
links: []
created: 2026-09-03T17:50:36Z
type: task
priority: 1
assignee: Paul English
---
# Fix v0.2.0 bugs for v0.2.1 release

**Working directory:** `/mnt/md0/src/python/coffeetrain`

Bugs 1-9 from code review: CLI hyperparams ignored, set_state NameError, coerce NameError, SkipBlock NameError, broken hf_accelerator, early_stopping stub, signal handler restore, stop_training unchecked, bert-hgnc stale v1 example. Fix all, add tests, bump to 0.2.1, commit without pushing.


## Notes

**2026-09-03T18:42:34Z**

Implemented v0.2.1 fixes: typed CLI overrides, state/coercion bugs, safe block overrides, repaired early stopping and signals, completed HuggingFace Accelerate plugin with optional extra, migrated HGNC example to v2, strengthened tests/lint/CI/docs. Validation: 14 pytest tests, Ruff, uv lock --check, compileall, and uv build all pass. Commit prepared locally; not pushed.

**2026-09-03T18:44:44Z**

Final validation passed: 14 tests, full Ruff, compileall, uv lock --check, uv build, and twine check for both 0.2.1 distributions. Changes staged for local commit only; no push or tag.

**2026-09-03T18:48:54Z**

Added required transcript summary at docs/transcript-summaries/2026-09-03-v0-2-1-bug-fix-release-and-deployment.md before deployment.
