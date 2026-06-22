# Automated Process Tracing Pipeline

A multi-pass LLM pipeline for Van-Evera-style process tracing: source text in,
competing causal explanations and auditable support out. This is process tracing
and mixed-methods causal inference infrastructure, not generic qualitative coding.

## What It Does

Given a text, the single-case pipeline:

1. Extracts actors, events, mechanisms, causal edges, and quoted evidence.
2. Generates a research question and competing causal hypotheses with observable
   predictions.
3. Tests the evidence in one coherent evidence-by-hypothesis likelihood matrix,
   with Van Evera diagnostic labels and dependence clusters.
4. Evaluates absence of predicted evidence separately from evidence of absence.
5. Updates comparative support in Bayesian log space with researcher priors,
   an explicit residual hypothesis, interpretive-evidence caps, dependence
   pooling, and sensitivity checks.
6. Synthesizes a written analysis with verdicts, caveats, and further tests.
7. Optionally runs a second-reading refinement pass, then reruns testing,
   updating, and synthesis.

Output: `result.json` (structured data) and `report.html` (Bootstrap dashboard
with support tables, PhD-style audit, temporal timeline, and a vis.js temporal
causal network).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ../llm_client

# Set an API key for the configured provider
export GEMINI_API_KEY=your_key_here

# Run on a sample historical text
python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir output/french_rev \
  --research-question "Why did the French Revolution radicalize?"

# Optional model override
python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir output/french_rev \
  --model gpt-4o
```

Useful options:

- `--research-question <text>` pins the outcome to explain.
- `--priors priors.json` supplies `{hypothesis_id: weight}` prior weights.
- `--review` pauses after hypothesis generation, and after refinement if used.
- `--refine` runs the second-reading refinement pass.
- `--from-result output/run/result.json` reuses extraction and hypotheses from a
  matching prior result, then refines.
- `--max-budget <dollars>` sets the per-call LLM budget cap.

Quality audit:

```bash
make audit-result RESULT=output/french_rev/result.json REPORT=output/french_rev/report.html
```

## Architecture

```text
pt/
  schemas.py           Pydantic contracts for all pipeline data
  llm.py               LLM boundary through llm_client.call_llm_structured
  pass_extract.py      Pass 1: source-grounded extraction
  pass_hypothesize.py  Pass 2: hypothesis generation
  pass_test.py         Pass 3: evidence-by-hypothesis likelihood matrix
  pass_absence.py      Pass 3b: qualitative absence-of-evidence evaluation
  bayesian.py          Pass 3.5: deterministic comparative-support update
  pass_synthesize.py   Pass 4: written synthesis
  pass_refine.py       Pass 5: optional second-reading refinement
  pipeline.py          Single-case orchestrator
  report.py            HTML report and temporal causal network
  cli.py               python -m pt entry point
  multi_pipeline.py    Multi-document cross-case analysis
```

## Documentation Map

- `docs/PROJECT_THEORY_AND_GOALS.md` - canonical project intent and current
  capability ledger.
- `docs/WHITEPAPER_optimal_automated_process_tracing.md` - quality-optimal
  methodology paper.
- `docs/BUILDPLAN_pragmatic_process_tracing.md` - pragmatic implementation plan
  and compromise record.
- `docs/OUTPUT_QUALITY_RUBRIC.md` - audit rubric for generated results/reports.
- `docs/ontology.md` - current analytic ontology and report-network semantics.
- `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md` - external SOTA map and
  SOTA+ opportunity analysis.
- `docs/plans/002_sota_plus_recovery_plan.md` - recovery gates and thin-slice
  operating model.

Historical sprint notes and superseded analyses live under `docs/archive/`.

## Tests

```bash
make check
```

Legacy and live-LLM exploratory tests are skipped by default. Set
`PT_RUN_LIVE_LLM_TESTS=1` to run the live provider smoke tests.

## Sample Input Texts

- `input_text/revolutions/french_revolution.txt` - Wikipedia, French Revolution
- `input_text/american_revolution/american_revolution.txt` - Wikipedia, American Revolution
- `input_text/russia_ukraine_debate/westminister_pirchner_v_bryan.txt` - Westminster debate transcript
