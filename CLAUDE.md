# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## LLM-First Architecture Policy (MANDATORY)

**CORE PRINCIPLE**: This system is **LLM-FIRST** with **ZERO TOLERANCE** for rule-based or keyword-based implementations.

**PROHIBITED**:
- Keyword matching for evidence classification
- Hardcoded probative value assignments
- Rule-based contradiction detection
- Any `if/elif` chains for semantic understanding
- Dataset-specific logic or historical period-specific rules
- Returning None/0/[] on LLM failure (must raise, fail-fast)
- Direct API calls bypassing `pt/llm.py` / `llm_client`

**REQUIRED**:
- LLM semantic analysis for ALL evidence-hypothesis relationships
- Structured Pydantic outputs for ALL LLM calls
- Generalist process tracing — no dataset-specific hardcoding
- Consistent `llm_client` routing for all LLM operations
- Single model configuration across entire pipeline

**Exception**: Pure math (Bayesian updating, normalization, clamping) is deliberately NOT LLM — it lives in `pt/bayesian.py` with deterministic, testable functions.

---

## Project Overview

**Van Evera Process Tracing Pipeline** — a multi-pass LLM pipeline that performs systematic causal inference on historical texts using Stephen Van Evera's methodology.

### North Star

This project is automated process tracing and mixed-methods causal inference, not generic qualitative coding or thematic analysis. The ambition is to automate process-tracing research at PhD / think-tank / academic quality by making the labor-intensive parts of within-case causal inference explicit, structured, auditable, scalable, and interoperable with quantitative causal methods.

The working premise is that expert process-tracing tasks are not protected by an inherent human advantage. Humans are often slower, less reproducible, less exhaustive, and not automatically less biased. The quality target is achieved through architecture: structured prompts and schemas, provenance, adversarial critic roles, independence checks, sensitivity analysis, trace-production modeling, cross-case quantitative integration, and iterative process tracing ↔ quantitative feedback. Human review remains valuable as direction-setting, validation, and accountability, but it is not treated as the only source of methodological rigor or as a permanent bottleneck.

### Future Workbench Alignment

This repo is the causal/process-tracing engine for a future
`mixed_methods_workbench`, alongside `~/projects/qualitative_coding` as the
qualitative evidence engine. Preserve the boundary: this repo owns source
packets, source coverage, rival causal hypotheses, observable predictions,
diagnostic evidence testing, likelihood-vector elicitation, deterministic
comparative support updates, absence checks, trace-production/dependence
modeling, and process-tracing reports. Do **not** rebuild broad qualitative
coding, QDA export, or general codebook review here; instead consume qualitative
claims, patterns, source anchors, and candidate explanations through typed
contracts when the workbench bridge is ready.

Given a text, the pipeline:
1. **Extracts** evidence, actors, events, mechanisms, causal edges
2. **Hypothesizes** competing causal explanations with observable predictions
3. **Tests** evidence with one coherent evidence-by-hypothesis likelihood matrix
3b. **Absence** evaluates what predicted evidence is missing from the text (failed hoop tests)
4. **Updates** comparative support via deterministic Bayesian math in log space
5. **Synthesizes** a written analytical narrative with verdicts (informed by absence findings)
6. **Refines** (optional) re-reads source text with full context, applies delta, re-runs passes 3-5

### Running the Pipeline

```bash
python -m pt input_text/revolutions/french_revolution.txt --output-dir output/french_rev
```

Output: `result.json` (full structured data) + `report.html` (Bootstrap + vis.js network graph).

Options:
- `--model <litellm-model-id>` — override the default model
- `--theories <path>` — inject theoretical frameworks for hypothesis generation (see Theory Injection below)
- `--research-question <text>` — pin the outcome to explain instead of letting the LLM choose it
- `--source-packet <path>` — load a source-packet contract or assistant artifact; pins research question and source-scope metadata
- `--review` — pause after hypothesis generation (and after refinement) for human review/editing
- `--json-only` — skip HTML report generation
- `--refine` — run analytical refinement after initial pipeline, then re-run passes 3+
- `--from-result <path>` — load existing result.json, skip passes 1-2, implies `--refine`
- `--priors <path>` — JSON object mapping hypothesis ids to positive prior weights
- `--max-budget <dollars>` — per-call LLM budget cap

### Knowledge Surfaces And Archive Boundary

- Active repo orientation wiki: `process-tracing-current/` (OKF bundle with
  chronicle at `process-tracing-current/wiki/log.md`). Use it for current-state
  orientation, then verify against active code, tests, and docs before changing
  behavior.
- Archive wiki: `~/archive/process_tracing/` (OKF bundle with chronicle at
  `~/archive/process_tracing/wiki/log.md`). Do not search it during normal
  implementation; use it only for history, recovery, previous-attempt review, or
  explicit archive cleanup.
- Archive policy: `docs/ARCHIVE_POLICY.md`. Read it before moving retired
  material out of this repo or promoting archived material back into active docs,
  plans, code, or tests.

### Multi-Document Cross-Case Analysis

```bash
# Theory-driven: researcher provides the causal model
python -m pt.multi input_text/revolutions/*.txt \
    --causal-model models/skocpol_revolution.yaml \
    -o output/revolutions_cross

# Data-driven: LLM proposes model, researcher reviews
python -m pt.multi input_text/revolutions/*.txt \
    -o output/revolutions_cross --review

# Skip CausalQueries (just binarize)
python -m pt.multi input_text/revolutions/*.txt \
    --causal-model models/skocpol.yaml --skip-cq \
    -o output/revolutions_cross
```

Output: `multi_result.json` + `multi_report.html` (Bootstrap + vis.js DAG + binarization matrix).

Options:
- `--causal-model <path>` — YAML causal model (theory-driven). Without it, LLM proposes one (data-driven).
- `--skip-cq` — skip CausalQueries R bridge (just binarize, useful when R not installed)
- `--review` — pause at checkpoints (model proposal, hypothesis generation)
- Other options same as single-text pipeline (`--model`, `--theories`, `--json-only`)

**Dependencies**: R + CausalQueries package only needed for causal estimation. Pipeline works without R through binarization.

---

## Active Codebase: `pt/`

| File | Purpose | LLM? |
|------|---------|------|
| `pt/schemas.py` | Pydantic models for all pipeline data | No |
| `pt/llm.py` | LLM boundary — delegates to `llm_client.call_llm_structured` | Yes |
| `pt/prompts/*.yaml` | YAML/Jinja2 prompt templates (loaded via `llm_client.render_prompt()`) | — |
| `pt/pass_extract.py` | Pass 1: Evidence extraction | Yes |
| `pt/pass_hypothesize.py` | Pass 2: Hypothesis generation | Yes |
| `pt/pass_test.py` | Pass 3: likelihood-vector elicitation (one matrix call: per-evidence vector across all hypotheses) | Yes |
| `pt/pass_absence.py` | Pass 3b: Absence-of-evidence evaluation (all hypotheses, single call) | Yes |
| `pt/bayesian.py` | Pass 3.5: coherent joint update (log-space softmax, pure math, no LLM) | No |
| `pt/pass_synthesize.py` | Pass 4: Written synthesis | Yes |
| `pt/verdict_calibration.py` | Deterministic calibration of synthesis verdict labels against computed support | No |
| `pt/pass_refine.py` | Pass 5: Analytical refinement (second reading) | Yes |
| `pt/apply_refinement.py` | Apply refinement delta to extraction + hypotheses | No |
| `pt/pipeline.py` | Orchestrator — runs passes sequentially | No |
| `pt/report.py` | HTML report generation, PhD audit, temporal causal network | No |
| `pt/cli.py` | CLI entry point | No |
| `pt/source_packet.py` | Source-packet contract, loader, summary, and prompt context | No |
| `pt/source_coverage.py` | Deterministic packet-source marker coverage against input/evidence | No |
| `pt/schemas_multi.py` | Pydantic models for cross-case analysis | No |
| `pt/pass_binarize.py` | Map extraction → binary variables (per case) | Yes |
| `pt/pass_propose_model.py` | Data-driven: propose causal model from N extractions | Yes |
| `pt/multi_pipeline.py` | Multi-doc orchestrator with caching | No |
| `pt/cli_multi.py` | CLI entry point for `python -m pt.multi` | No |
| `pt/cq_bridge.py` | Python side of CausalQueries R bridge (subprocess + JSON) | No |
| `pt/report_multi.py` | Cross-case HTML report | No |
| `scripts/cq_runner.R` | R side of CQ bridge (~100 lines) | No |
| `models/skocpol_revolution.yaml` | Example theory-driven causal model (Skocpol) | No |

### Inference-core rebuild (current)

The single-text testing/update core was rebuilt to be **coherent**. Canonical spec:
`docs/PROJECT_THEORY_AND_GOALS.md`; methodology: `docs/WHITEPAPER_optimal_automated_process_tracing.md`;
build plan: `docs/BUILDPLAN_pragmatic_process_tracing.md`; historical rebuild log:
`~/archive/process_tracing/raw/repo-archive-2026-06-24/docs/archive/development/REBUILD_SPRINT.md`.

- **Likelihood *vectors*, not two-way LRs**: Pass 3 makes **one matrix call** — for each evidence item it emits a relative-likelihood vector across *all* hypotheses (`pt/schemas.py: EvidenceLikelihood`). Per-hypothesis LRs are derived as `relative_likelihood / geomean(vector)`, so pairwise ratios are coherent by construction. `pass_test` fails loud on incomplete/duplicate/unknown vectors.
- **Coherent joint update**: `pt/bayesian.py` uses a log-space softmax (`post_i = softmax(log prior_i + Σ log LR_i)`) — **order-invariant**, no per-step clamping. (Replaced the earlier per-hypothesis binary-odds-then-normalize, which was order-dependent.)
- **Pairwise LR cap = 20**: bounds a single item's max:min ratio across hypotheses (`LR_CAP`, each centered log-LR clamped to ±0.5·log 20). Does **not** bound aggregate overconfidence — see below.
- **Researcher priors + sensitivity**: CLI `--priors` (JSON, validated fail-loud); `PriorSensitivity` reports robustness to ±2× prior swings. Report shows comparative **support** (not "posterior probability"), support sensitivity ranges, and rank/prior stability.
- **Residual + dependence pooling**: the update includes `H0_residual` by default and partially pools LLM-supplied dependence clusters so correlated evidence is not raw-counted as independent.
- **Known limitation (load-bearing)**: dependence pooling uses one scalar dependence strength per cluster, not per-hypothesis redundancy or a full trace-production model. If the LLM misses a cluster, support can still be overconfident; read high-support fragile results as a ranking until source-lineage and trace-production audits improve.

### Implemented Guardrails and Design Constraints

- **Live E2E required**: Deterministic mocked tests are necessary but never
  sufficient for implementation slices. Every slice that changes pipeline,
  prompt, schema, report, audit, or agent behavior must run a live non-mocked
  end-to-end command on a real process-tracing case, write `result.json` and
  `report.html`, run `make audit-result` on those artifacts, and record the
  command, output directory, model, audit grade, and failures/fixes in the slice
  evidence note before the slice can be considered complete.

- **Relevance gating**: Evidence with `relevance < 0.4` is forced to `LR = 1.0` (uninformative). Above 0.4, soft discount on the log scale of the centered LR.
- **Relevance = min(temporal, causal-domain)**: Not just "how recent" but "how on-topic for this hypothesis."
- **Anti-circularity**: Hypotheses cannot be derived from interpretive evidence. Circular evidence gets `relevance = 0.1`.
- **Anti-tautology**: Hypotheses must specify causal mechanisms, not describe outcomes.
- **Opposite predictions**: Genuine rivals must make opposite predictions about the same observable evidence. Complementary factors (precondition + trigger + framing) must be merged.
- **Mutual exclusion self-check**: "If H_A is decisive, does that make H_B unnecessary or wrong?" If no → merge.
- **Mandatory agency hypothesis**: At least one hypothesis must name specific individuals making deliberate choices.
- **Source fidelity**: Every evidence item must quote or closely paraphrase the input text. No hallucinated evidence.
- **Pairwise discrimination**: Each hypothesis pair must have 3+ evidence items where LRs diverge by 2x+.
- **Steelman verdicts**: Every hypothesis gets a mandatory steelman case, even if eliminated — ensures fair analysis.
- **Verdict calibration**: `pt/verdict_calibration.py` deterministically downgrades synthesis status labels that overstate computed comparative support. LLMs write reasoning and steelman cases; they do not get final authority to call a very-low-posterior hypothesis "supported".
- **Mechanical robustness**: Each hypothesis's posterior is mechanically classified as "robust" (driven by few decisive LRs with |log(LR)| > 1.6), "fragile" (driven by many weak LRs with |log(LR)| < 0.7), or "moderate". No LLM judgment involved.
- **Sensitivity analysis**: Per-hypothesis perturbation of top-N most influential LRs (plus rivals' top drivers) by ±50% on log-LR scale. Reports posterior ranges and rank stability under perturbation.
- **Multi-speaker awareness**: Debate/discussion texts get speaker-attributed evidence, disputed facts classified as interpretive, neutral research questions, and Rule F preventing systematic speaker favoritism in testing.
- **Retry logic**: Up to 3 retries with exponential backoff (jittered, capped 30s) for transient LLM failures (JSON parse, rate limits, timeouts).
- **Review checkpoint**: `--review` flag pauses after hypothesis generation for human review/editing before expensive testing pass.
- **Theory injection**: By default, the hypothesis pass generates at least one theory-derived hypothesis from the LLM's intrinsic knowledge of social science frameworks. Optional `--theories <file>` injects user-provided frameworks; the LLM must generate at least one hypothesis per framework. Example theory file: `input_text/theories/legitimacy_vacuum.txt`.
- **Absence-of-evidence**: After testing, a single LLM call evaluates all hypotheses for missing predicted evidence (failed hoop tests). Findings are qualitative only — they feed into synthesis narrative but NOT into Bayesian updating, avoiding speculative LR assignments for absent evidence. Each finding rated "damaging", "notable", or "minor" with reasoning about whether the text would contain this evidence if it existed.
- **Analytical refinement (second reading)**: `--refine` flag triggers a re-read of the source text after the full pipeline completes. The refinement pass receives source text + condensed first-pass results (~33K tokens), produces a structured delta (new evidence, reinterpretations, spurious removals, hypothesis refinements, missing mechanisms), then re-runs passes 3+ with the updated data. New evidence uses `evi_ref_` prefix. `--from-result <path>` loads an existing result.json to skip passes 1-2. `merge_suggestion` refinements are advisory only (not auto-applied). The `--review` flag adds a checkpoint after the refinement LLM call. Costs ~7 additional LLM calls (~3-5 min).

### Known Gaps vs. PhD-Level Analysis

1. **Source scope and diagnosticity cap academic readiness** — broad overview texts often produce useful exploratory rankings, not PhD-review-ready causal claims. Use source packets and pre-specified discriminators.
2. **Hypothesis partitions still need audit** — broad, overlapping, or complementary hypotheses can make comparative support look cleaner than the actual causal menu. Use `--review` and preserve pairwise discriminators.
3. **Dependence is partial** — scalar dependence clusters reduce double-counting, but per-hypothesis redundancy, source-lineage graphs, and trace-production model averaging remain planned.
4. **Absence remains qualitative** — Pass 3b evaluates missing predicted evidence for synthesis, but observability-weighted absence does not yet enter Bayesian updating.
5. **Cross-case estimation needs real variation** — CausalQueries requires comparable cases with variation in outcomes and covariates; all-positive/all-ones revolution corpora are not enough for effect estimation.

---

## Strategic Position

1. **Text-in, analysis-out** — no manual model specification required
2. **Methodologically grounded** — Van Evera's process tracing (hoop/smoking gun/doubly decisive), not ad-hoc LLM reasoning
3. **Quantified and audited support** — comparative support, sensitivity ranges, robustness, evidence triage, and explicit academic caps
4. **Mixed-methods bridge** — single-text process tracing plus cross-case CausalQueries path, without pretending the two estimands are the same

### Prompt Quality Notes

The prompts in `pt/prompts/pass3_test.yaml` and `pt/prompts/pass2_hypothesize.yaml` are the most critical for output quality. They encode Van Evera's methodology and the anti-bias rules that prevent:
- Compound bias (many slightly-anti LRs crushing narrowly-correct hypotheses)
- Circular reasoning (hypothesis derived from interpretive evidence)
- Tautological hypotheses (descriptions masquerading as explanations)
- Overlap (hypotheses that predict the same evidence)

Changes to these prompts should be validated by running the pipeline on a text and comparing the output against the source material.

---

## Legacy Codebase (Removed 2026-02-15)

The old `core/` graph-based architecture, `universal_llm_kit/`, `tools/`, and all root-level debug scripts (`test_*.py`, `verify_*.py`) have been removed. They imported a non-existent `core/` module. The `tests/` directory was pruned to only files testing active `pt` code. Historical evidence records have been moved to `~/archive/process_tracing/raw/repo-historical-evidence-2026-06-24/`; active evidence in this repo is limited to Plan 003 notes under `evidence/current/`.

---

## Coding Philosophy

- **No lazy implementations**: Complete, working code — no stubs or placeholders
- **Fail-fast**: Surface errors immediately, no silent failures
- **Test before declaring done**: Verify with actual pipeline runs
- **LLM for semantics, math for math**: Bayesian updating is pure Python; everything else is LLM

---

## Test Inputs

Validated datasets in `input_text/`:
- `input_text/revolutions/french_revolution.txt` — Primary test text (Wikipedia, French Revolution)
- Additional texts in `input_text/` subdirectories

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

## Commands

```bash
# Run process tracing pipeline
python -m pt input_text/your_text.txt --output-dir output/your_run/

# Run deterministic tests and repo validators
make check

# Multi-document analysis
python -m pt.multi input_text/case_a.txt input_text/case_b.txt --output-dir output/multi/
```

## Principles

- **Research ambition**: automate expert process tracing and mixed-methods causal research at PhD / think-tank / academic quality; do not frame the system as generic qualitative coding or merely augmenting human analysts.
- **LLM-First**: all semantic analysis uses LLM, never rule-based matching or keyword logic
- **Structured output**: all LLM calls return Pydantic models; `if/elif` chains for semantic understanding are prohibited
- **Fail loud**: raise on LLM failure, never return None/0/[] silently
- **Generalist**: no dataset-specific hardcoding; the pipeline must work on any historical text
- **Workbench boundary**: this repo is the causal/process-tracing engine for a future `mixed_methods_workbench`; do not rebuild broad qualitative coding or QDA review here, and consume `qualitative_coding` outputs only through typed contracts when that bridge is ready.

## Workflow

1. Pass text to `python -m pt` (or `python -m pt.multi` for cross-case)
2. Pipeline runs: Extract → Hypothesize → Test → Bayesian update → Synthesize → (optionally Refine)
3. Output: `result.json` (full data) + `report.html` (visual network)
4. Review HTML report for narrative and verdicts

## References

- `docs/V2_RECURSIVE_AGENT_MIGRATION_PLAN.md` — Recursive agent design (if present)
- `pt/` — Active codebase (process tracing engine)
- `CLAUDE.md` — This file (canonical operating guidance)
- `AGENTS.md` — Generated mirror for non-Claude agents
