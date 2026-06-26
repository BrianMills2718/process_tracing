# Agent Handoff — process_tracing — 2026-06-26

**For**: Fresh agent starting a new session on this repo  
**Start with**: Comprehensive audit and critique before any new work  
**Current HEAD**: 364b4f7 (master)  
**Deterministic tests**: 210 pass (`make check`)

---

## What This Repo Is

A multi-pass LLM pipeline for Van Evera process tracing. Given a historical text it:
1. Extracts evidence, actors, events, mechanisms (Pass 1)
2. Generates competing causal hypotheses (Pass 2)
3. Elicits a likelihood vector per evidence item across all hypotheses (Pass 3)
4. Evaluates absence-of-evidence (Pass 3b)
5. Updates comparative support via deterministic Bayesian math in log space (Pass 3.5)
6. Optionally runs a structural critic (Pass 3.7) — now supports `--critic-model` for independent critique
7. Synthesizes a written analytical narrative (Pass 4)
8. Optionally refines on second reading (Pass 5)

Key constraint: **LLM-first** — no keyword matching, no if/elif semantic logic, no silent fallbacks.

Run: `python -m pt input_text/revolutions/french_revolution.txt --output-dir output/test`  
Audit: `python scripts/audit_result_quality.py output/test/result.json`  
Benchmark: `PYTHONPATH=. python scripts/run_benchmark.py --verbose`

---

## What Was Done This Session (since fd79504)

### C-017: Independent Critic Model — RESOLVED (commits 959c594, 364b4f7)
**Problem**: Same-model structural critic (Pass 3.7) could not detect confounds it shared with Pass 3 — confirmed by live E2E on a planted-confound synthetic text (fiscal crisis + popular unrest always co-vary, explicitly stated in text).

**Fix**: Added `--critic-model` CLI flag. Independent-model E2E (gpt-5.4-mini main + claude-haiku-4-5 critic) on same text → `confound=high` + `too_strong_claim=high×2` + `re_elicitation=True`. Same-model: 0 high-severity, no re-elicitation.

**Files changed**: `pt/cli.py`, `pt/pipeline.py`, `tests/test_pass_critic.py` (+2 tests), `docs/benchmarks/benchmark_config.yaml` (2 optional result_file cases documenting before/after)

**Usage**: `python -m pt text.txt --critic --critic-model openrouter/anthropic/claude-haiku-4-5`

### C-018: Cyclic DAG Prompt — MITIGATED (commit ba0876a)
**Problem**: Data-driven `pt.multi` (no `--causal-model`) proposed cyclic DAGs that exhausted retries.

**Fix**: Added to `pt/prompts/propose_model.yaml`: concrete cycle example (A→B→C→A), FORBIDDEN vs VALID chain examples, before-you-submit self-check section.

**Status**: Prompt strengthened. Live verification (`pt.multi --data-driven`) was run as background task this session — see "C-018 verification" below.

### Post-Audit Corrections (commit cf1da47)
- `docs/benchmarks/last_scorecard.json` removed from git tracking (generated artifact)
- `pyrightconfig.json` added (Pyright module resolution for `scripts.*`)
- `test_benchmark.py` strengthened: `no-report_path` test now asserts `score is not None`
- `benchmark_config.yaml`: adversarial fixture dual-failure mode documented
- `FUTURE_WORK.md`: `--critic-model` added (now implemented, removed from future work)
- `docs/FUTURE_WORK.md` methodology extensions updated

### Benchmark Bug Fixes (commit a1921ab)
- `run_benchmark.py:75`: `relevance_to_hypothesis` → `relevance` (Pydantic extra='ignore' was silently dropping the field; all fixture items had relevance=1.0 instead of intended 0.9)
- `run_benchmark.py:108`: Removed non-existent `hypothesis_link`/`evidence_ids` from `Mechanism` constructor
- `test_multi_contracts.py`: Added `# type: ignore[arg-type]` to int→Literal[0,1] test helpers

---

## C-018 Verification (run this session, result pending)

A background task ran:
```bash
python -m pt.multi \
  input_text/revolutions/french_revolution.txt \
  input_text/revolutions/american_revolution.txt \
  --model openrouter/openai/gpt-5.4-mini \
  --output-dir output/c018_live_verify \
  --skip-cq --json-only
```

**If successful** (output/c018_live_verify/multi_result.json exists with a valid acyclic `CausalModelSpec`): mark C-018 resolved in `docs/plans/sota_plus_concern_register.md`, update handoff.

**If still cyclic** (exception or ValidationError on CausalModelSpec): document failure, try with `--causal-model models/skocpol_revolution.yaml` (theory-driven mode), and note that C-018 requires a stronger acyclicity constraint or retry logic enhancement.

---

## Open Items (prioritized)

| ID | Priority | Status | Summary |
|---|---|---|---|
| C-018-live-verify | low | mitigated | Confirm cyclic DAG prompt fix works in live data-driven run |
| P3 | medium | deferred | Cap-84 source-lineage inflation — needs multi-source corpus |

### Recommended Next Work (in order)

1. **Hypothesis partition audit** (FUTURE_WORK.md #1) — highest leverage for result quality. Broad/overlapping/complementary hypotheses undermine comparative support more than any single evidence issue. Design a partition-quality checklist, add a `--partition-review` mode or extend Pass 2's output, add deterministic tests.

2. **trace_eval integration** — `~/projects/trace_eval` is shared infra that diagnoses which pipeline stage caused a failure (execution vs design vs cascade vs hallucination). process_tracing has no structured way to answer "did this result fail because Pass 3 was incoherent or because Pass 2 gave it bad hypotheses." Integration: define a `PipelineCase` per stage, wire into `make audit-result`.

3. **Dependence / trace-production upgrade** (FUTURE_WORK.md #2) — scalar clusters reduce double-counting but miss per-hypothesis redundancy and source-lineage structure.

4. **C-018-live-verify** — quick check once API quota allows.

---

## Key Architecture Facts

- **Recommended E2E model**: `openrouter/openai/gpt-5.4-mini` (265s full French Rev, 123/123 vectors, no failures)
- **Avoid**: `gemini/gemini-2.5-flash` (20 req/day free cap), `deepseek/deepseek-chat` (returns relative_likelihood=0, violates gt=0.0 schema), `openrouter/openai/gpt-4o-mini` (hard-blocked in llm_client)
- **Pydantic extra='ignore'**: All pt/schemas.py models silently drop unknown field names at construction time — always verify field names against schema definition (sm-64f34fac670c)
- **Bayesian update is log-space softmax** — order-invariant, no per-step clamping. `DECISIVE_COUNT_FOR_ROBUST=3` (need ≥3 items with |log(LR)|>1.6 for "robust" label)
- **Critic**: advisory only — never mutates LR values directly; re-elicitation routes back through Pass 3 with critic summary injected as context
- **Benchmark**: `PYTHONPATH=. python scripts/run_benchmark.py --verbose` — 3 required fixture cases (no live LLM), 3 optional result_file cases. `last_scorecard.json` is gitignored.

## Known Gaps (from CLAUDE.md)

1. Complementary hypotheses treated as rivals
2. Debate genre partially mishandled (multi-speaker awareness partial)
3. Dependence partial — scalar clusters, no per-hypothesis redundancy or trace-production model
4. Absence qualitative only — not in Bayesian update
5. Cross-case estimation needs real variation
6. ~~Same-model critic~~ — **FIXED** via `--critic-model`
7. `missing_pathway` overloaded (hypothesis-level gap vs structural causal-edge gap)

## Sanity Checks

```bash
# Deterministic suite
PYTHONPATH=. pytest tests/test_pt_bayesian.py tests/test_pt_schemas.py \
  tests/test_extraction_quality.py tests/test_pass_diagnostic.py \
  tests/test_pass_critic.py tests/test_multi_contracts.py tests/test_benchmark.py -q
# Expect: 210 passed

# Imports
python3 -c "from pt.pass_critic import run_critic; from pt.pipeline import run_pipeline; print('ok')"

# CLI smoke
python -m pt --help | grep critic-model
# Expect: --critic-model shown in help

# Benchmark
PYTHONPATH=. python scripts/run_benchmark.py
# Expect: 2 passed (fixture cases), 2-3 skipped (optional result_file cases)
```

## Files To Read First

For orientation (in order):
1. `CLAUDE.md` (this repo) — canonical operating guidance, LLM-first policy, methodology
2. `docs/FUTURE_WORK.md` — prioritized roadmap
3. `docs/plans/sota_plus_concern_register.md` — open/resolved concerns with full history
4. `.claude/handoff.yml` — machine-readable current state
5. `pt/schemas.py` — all Pydantic models; ground truth for field names

For implementation:
- `pt/pipeline.py` — orchestrator; `run_pipeline()` is the main entry point
- `pt/pass_critic.py` — `run_critic()` — now accepts `model=` param threaded from `critic_model or model`
- `scripts/audit_result_quality.py` — 100-point audit rubric
- `scripts/run_benchmark.py` — benchmark runner (fixture + result_file cases)
