# CLAUDE.md

<!-- GENERATED FILE: DO NOT EDIT DIRECTLY -->
<!-- generated_by: scripts/meta/render_agents_md.py -->
<!-- canonical_claude: CLAUDE.md -->
<!-- canonical_relationships: scripts/relationships.yaml -->
<!-- canonical_relationships_sha256: 840b164dcfa4 -->
<!-- sync_check: python scripts/meta/check_agents_sync.py --check -->

This file is a generated Codex-oriented projection of repo governance.
Edit the canonical sources instead of editing this file directly.

Canonical governance sources:
- `CLAUDE.md` — human-readable project rules, workflow, and references
- `scripts/relationships.yaml` — machine-readable ADR, coupling, and required-reading graph

## Purpose

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
# Run process tracing pipeline
python -m pt input_text/your_text.txt --output-dir output/your_run/

# Run deterministic tests
PYTHONPATH=. pytest tests -q

# Multi-document analysis
python -m pt.multi input_text/case_a.txt input_text/case_b.txt --output-dir output/multi/
```

## Operating Rules

This projection keeps the highest-signal rules in always-on Codex context.
For full project structure, detailed terminology, and any rule omitted here,
read `CLAUDE.md` directly.

### Principles

- **LLM-First**: all semantic analysis uses LLM, never rule-based matching or keyword logic
- **Structured output**: all LLM calls return Pydantic models; `if/elif` chains for semantic understanding are prohibited
- **Fail loud**: raise on LLM failure, never return None/0/[] silently
- **Generalist**: no dataset-specific hardcoding; the pipeline must work on any historical text

### Workflow

1. Pass text to `python -m pt` (or `python -m pt.multi` for cross-case)
2. Pipeline runs: Extract → Hypothesize → Test → Bayesian update → Synthesize → (optionally Refine)
3. Output: `result.json` (full data) + `report.html` (visual network)
4. Review HTML report for narrative and verdicts

## Machine-Readable Governance

`scripts/relationships.yaml` is the source of truth for machine-readable governance in this repo: ADR coupling, required-reading edges, and doc-code linkage. This generated file does not inline that graph; it records the canonical path and sync marker, then points operators and validators back to the source graph. Prefer deterministic validators over prompt-only memory when those scripts are available.

## References

- `docs/V2_RECURSIVE_AGENT_MIGRATION_PLAN.md` — Recursive agent design (if present)
- `pt/` — Active codebase (process tracing engine)
- `CLAUDE.md` — This file (canonical operating guidance)
- `AGENTS.md` — Generated mirror for non-Claude agents
