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
3b. **Absence** evaluates what predicted evidence is missing from the text (failed hoop tests)
4. **Updates** posteriors via Bayesian math in odds space
5. **Synthesizes** a written analytical narrative with verdicts (informed by absence findings)

### Running the Pipeline

```bash
python -m pt input_text/revolutions/french_revolution.txt --output-dir output/french_rev
```

Output: `result.json` (full structured data) + `report.html` (Bootstrap + vis.js network graph).

Options:
- `--model <litellm-model-id>` — override the default model
- `--theories <path>` — inject theoretical frameworks for hypothesis generation (see Theory Injection below)
- `--review` — pause after hypothesis generation for human review/editing
- `--json-only` — skip HTML report generation

---

## Active Codebase: `pt/`

| File | Purpose | LLM? |
|------|---------|------|
| `pt/schemas.py` | Pydantic models for all pipeline data | No |
| `pt/llm.py` | LiteLLM abstraction, structured output parsing | Yes |
| `pt/pass_extract.py` | Pass 1: Evidence extraction prompt | Yes |
| `pt/pass_hypothesize.py` | Pass 2: Hypothesis generation prompt | Yes |
| `pt/pass_test.py` | Pass 3: Diagnostic testing prompt (one call per hypothesis) | Yes |
| `pt/pass_absence.py` | Pass 3b: Absence-of-evidence evaluation (all hypotheses, single call) | Yes |
| `pt/bayesian.py` | Pass 3.5: Bayesian updating (pure math, no LLM) | No |
| `pt/pass_synthesize.py` | Pass 4: Written synthesis prompt | Yes |
| `pt/pipeline.py` | Orchestrator — runs passes sequentially | No |
| `pt/report.py` | HTML report generation | No |
| `pt/cli.py` | CLI entry point | No |

### Key Design Decisions (v5h)

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
- **Mechanical robustness**: Each hypothesis's posterior is mechanically classified as "robust" (driven by few decisive LRs with |log(LR)| > 1.6), "fragile" (driven by many weak LRs with |log(LR)| < 0.7), or "moderate". No LLM judgment involved.
- **Sensitivity analysis**: Per-hypothesis perturbation of top-N most influential LRs (plus rivals' top drivers) by ±50% on log-LR scale. Reports posterior ranges and rank stability under perturbation.
- **Multi-speaker awareness**: Debate/discussion texts get speaker-attributed evidence, disputed facts classified as interpretive, neutral research questions, and Rule F preventing systematic speaker favoritism in testing.
- **Retry logic**: Up to 3 retries with exponential backoff (jittered, capped 30s) for transient LLM failures (JSON parse, rate limits, timeouts).
- **Review checkpoint**: `--review` flag pauses after hypothesis generation for human review/editing before expensive testing pass.
- **Theory injection**: By default, the hypothesis pass generates at least one theory-derived hypothesis from the LLM's intrinsic knowledge of social science frameworks. Optional `--theories <file>` injects user-provided frameworks; the LLM must generate at least one hypothesis per framework. Example theory file: `input_text/theories/legitimacy_vacuum.txt`.
- **Absence-of-evidence**: After testing, a single LLM call evaluates all hypotheses for missing predicted evidence (failed hoop tests). Findings are qualitative only — they feed into synthesis narrative but NOT into Bayesian updating, avoiding speculative LR assignments for absent evidence. Each finding rated "damaging", "notable", or "minor" with reasoning about whether the text would contain this evidence if it existed.

### Known Gaps vs. PhD-Level Analysis

1. **Complementary hypotheses as rivals** — Every run has ≥1 pair that the synthesis admits are "two sides of the same coin." The mutual exclusion rules help but don't fully solve it. Best mitigation: `--review` checkpoint.
2. **Debate genre still partially mishandled** — Speaker assessments sometimes coded empirical. Cross-speaker agreement not weighted more heavily in practice.
3. ~~**Absence-of-evidence not evaluated**~~ FIXED — Pass 3b evaluates missing predicted evidence qualitatively (synthesis only, not Bayesian). Each finding includes severity, reasoning, and whether the text would contain the evidence if it existed.
4. **Synthesis is summary, not analysis** — Tends toward restating Bayesian results in prose rather than generating original analytical insights.

---

## Competitive Landscape (surveyed Feb 2026)

**No existing tool does the full loop: text in → extraction → hypotheses → diagnostic tests → Bayesian posteriors → sensitivity → synthesis.** That end-to-end pipeline is our unique position.

### Direct competitors (same problem space)

| Tool | What it does | What it lacks vs. us |
|------|-------------|---------------------|
| **CausalQueries** (R, Humphreys) | Formal Bayesian process tracing with DAGs. Mathematically rigorous, d-separation aware. | No text input, no LLM, no automation. Analyst must manually specify model structure and data. It's a calculator, not an analyst. |
| **ACH tools** (Burton, Open-Synthesis, ArkhamMirror) | CIA-style consistency matrices. ArkhamMirror adds LLM "devil's advocate." | No Bayesian math — consistency counting only. No diagnostic test classification. No extraction pipeline — human provides hypotheses and evidence. |
| **LLM SATs** (Roberts, SANS) | Streamlit apps using GPT-4 for ACH, Starbursting, Key Assumptions Checks. | Single-pass LLM, no pipeline. Proof-of-concept, no Bayesian updating or sensitivity. |

### Adjacent tools (LLM + causal inference, but not from text)

| Tool | What it does | What it lacks vs. us |
|------|-------------|---------------------|
| **PyWhy-LLM** (DoWhy ecosystem) | LLM suggests confounders, validates causal assumptions for structured data. | Operates on structured data, not text. No Van Evera, no process tracing. |
| **LLM-argumentation** (DAMO-NLP-SG, ACL 2024) | Benchmarks LLMs on argument mining tasks. | Benchmark, not a tool. Evaluates capabilities, doesn't build a pipeline. |

### Adjacent tools (LLM + qualitative research)

| Tool | What it does | What it lacks vs. us |
|------|-------------|---------------------|
| **LLMCode** (Hämäläinen) | LLM-assisted qualitative coding with IoU/Hausdorff alignment metrics. Published. | Coding, not causal inference. No hypotheses, no Bayesian updating. |
| **DeTAILS** | LLM-assisted thematic analysis with researcher agency preserved. | Thematic analysis, not causal reasoning. |

### Our unique value = three things no one else combines

1. **Text-in, analysis-out** — no manual model specification required
2. **Methodologically grounded** — Van Evera's process tracing (hoop/smoking gun/doubly decisive), not ad-hoc LLM reasoning
3. **Quantified uncertainty** — Bayesian posteriors with sensitivity ranges and mechanical robustness, not just "the LLM thinks X"

### Strategic opportunities (not yet implemented)

- **Multi-document analysis** — compare causal claims across multiple texts on the same topic. Qualitative coding tools don't do causal inference; causal tools don't handle multiple texts.
- **CausalQueries bridge** — export extracted causal graphs in a format CausalQueries can import, bridging automated extraction and formal Bayesian methods.
- ~~**Absence-of-evidence**~~ IMPLEMENTED — Pass 3b evaluates missing predicted evidence (qualitative, synthesis only).

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
