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
- Direct API calls bypassing LiteLLM

**REQUIRED**:
- LLM semantic analysis for ALL evidence-hypothesis relationships
- Structured Pydantic outputs for ALL LLM calls
- Generalist process tracing — no dataset-specific hardcoding
- Consistent LiteLLM routing for all LLM operations
- Single model configuration across entire pipeline

**Exception**: Pure math (Bayesian updating, normalization, clamping) is deliberately NOT LLM — it lives in `pt/bayesian.py` with deterministic, testable functions.

---

## Project Overview

**Van Evera Process Tracing Pipeline** — a multi-pass LLM pipeline that performs systematic causal inference on historical texts using Stephen Van Evera's methodology.

Given a text, the pipeline:
1. **Extracts** evidence, actors, events, mechanisms, causal edges
2. **Hypothesizes** competing causal explanations with observable predictions
3. **Tests** each hypothesis against every evidence item (diagnostic tests + likelihood ratios)
4. **Updates** posteriors via Bayesian math in odds space
5. **Synthesizes** a written analytical narrative with verdicts

### Running the Pipeline

```bash
python -m pt input_text/revolutions/french_revolution.txt --output-dir output/french_rev
```

Output: `result.json` (full structured data) + `report.html` (Bootstrap + vis.js network graph).

Options: `--model <litellm-model-id>` to override the default model.

---

## Active Codebase: `pt/`

| File | Purpose | LLM? |
|------|---------|------|
| `pt/schemas.py` | Pydantic models for all pipeline data | No |
| `pt/llm.py` | LiteLLM abstraction, structured output parsing | Yes |
| `pt/pass_extract.py` | Pass 1: Evidence extraction prompt | Yes |
| `pt/pass_hypothesize.py` | Pass 2: Hypothesis generation prompt | Yes |
| `pt/pass_test.py` | Pass 3: Diagnostic testing prompt (one call per hypothesis) | Yes |
| `pt/bayesian.py` | Pass 3.5: Bayesian updating (pure math, no LLM) | No |
| `pt/pass_synthesize.py` | Pass 4: Written synthesis prompt | Yes |
| `pt/pipeline.py` | Orchestrator — runs passes sequentially | No |
| `pt/report.py` | HTML report generation | No |
| `pt/cli.py` | CLI entry point | No |

### Key Design Decisions (v5f)

- **LR cap at 20.0**: Prevents single evidence items from dominating. `LR_CAP = 20.0`, `LR_FLOOR = 0.05`.
- **Relevance gating**: Evidence with `relevance < 0.4` is forced to `LR = 1.0` (uninformative). Above 0.4, soft discount via `lr = exp(relevance * log(capped_lr))`.
- **Relevance = min(temporal, causal-domain)**: Not just "how recent" but "how on-topic for this hypothesis."
- **Anti-circularity**: Hypotheses cannot be derived from interpretive evidence. Circular evidence gets `relevance = 0.1`.
- **Anti-tautology**: Hypotheses must specify causal mechanisms, not describe outcomes.
- **Opposite predictions**: Genuine rivals must make opposite predictions about the same observable evidence. Complementary factors (precondition + trigger + framing) must be merged.
- **Mutual exclusion self-check**: "If H_A is decisive, does that make H_B unnecessary or wrong?" If no → merge.
- **Mandatory agency hypothesis**: At least one hypothesis must name specific individuals making deliberate choices.
- **Source fidelity**: Every evidence item must quote or closely paraphrase the input text. No hallucinated evidence.
- **Pairwise discrimination**: Each hypothesis pair must have 3+ evidence items where LRs diverge by 2x+.
- **Steelman verdicts**: Every hypothesis gets a mandatory steelman case, even if eliminated — ensures fair analysis.
- **Posterior robustness**: Each verdict flagged "robust" (driven by decisive tests) or "fragile" (driven by many small LR effects).
- **Multi-speaker awareness**: Debate/discussion texts get speaker-attributed evidence, disputed facts classified as interpretive, neutral research questions, and Rule F preventing systematic speaker favoritism in testing.
- **Retry logic**: Up to 3 retries with exponential backoff (jittered, capped 30s) for transient LLM failures (JSON parse, rate limits, timeouts).
- **Review checkpoint**: `--review` flag pauses after hypothesis generation for human review/editing before expensive testing pass.

### Known Gaps vs. PhD-Level Analysis (v5f)

**Fixable via code (pure math in bayesian.py):**
1. **No sensitivity analysis** — Pipeline produces point-estimate posteriors with no indication of how stable the ranking is under perturbation of uncertain LR assignments. Fix: perturb top-N most extreme LRs by ±50%, report posterior ranges.
2. **LLM-guessed robustness is unreliable** — Synthesis pass guesses "robust"/"fragile" but often marks everything "robust". Fix: compute mechanically from LR distribution (few decisive LRs = robust, many small LRs = fragile).

**Partially fixable via prompts (diminishing returns):**
3. **Complementary hypotheses as rivals** — Every run has ≥1 pair that the synthesis admits are "two sides of the same coin." The mutual exclusion rules help but don't fully solve it. Best mitigation: `--review` checkpoint.
4. **No hypotheses from theoretical frameworks** — Generated hypotheses anchor on what the text says rather than bringing external analytical frameworks (offense-defense theory, selectorate theory, path dependence). The LLM has this knowledge but the prompt doesn't elicit it.
5. **Debate genre still partially mishandled** — Speaker assessments sometimes coded empirical. Cross-speaker agreement not weighted more heavily in practice.
6. **Absence-of-evidence not evaluated** — Pipeline only tests evidence that IS present, not evidence that a hypothesis predicts SHOULD be present but isn't (hoop test failures from missing evidence).
7. **Synthesis is summary, not analysis** — Tends toward restating Bayesian results in prose rather than generating original analytical insights.

### Prompt Quality Notes

The prompts in `pass_test.py` and `pass_hypothesize.py` are the most critical for output quality. They encode Van Evera's methodology and the anti-bias rules that prevent:
- Compound bias (many slightly-anti LRs crushing narrowly-correct hypotheses)
- Circular reasoning (hypothesis derived from interpretive evidence)
- Tautological hypotheses (descriptions masquerading as explanations)
- Overlap (hypotheses that predict the same evidence)

Changes to these prompts should be validated by running the pipeline on a text and comparing the output against the source material.

---

## Legacy Codebase: `core/`

The `core/` directory (~25K lines) is an older graph-based architecture with ontology management, dynamic validation, and plugin systems. It is **not actively developed** — the `pt/` pipeline replaced it. Key components:

- `core/ontology_manager.py` — Centralized ontology queries
- `core/dynamic_ontology_validator.py` — Functional validation
- `core/structured_extractor.py` — LLM extraction
- `analyze_direct.py` — Old pipeline entry point
- `config/ontology_config.json` — Ontology definition

Root-level `debug_*.py`, `test_*.py`, and `verify_*.py` scripts are debugging artifacts from `core/` development.

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
