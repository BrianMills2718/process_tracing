# Van Evera Process Tracing Pipeline

A multi-pass LLM pipeline that performs systematic causal inference on historical texts using Stephen Van Evera's process tracing methodology.

## What It Does

Given a text, the pipeline:
1. **Extracts** evidence, actors, events, mechanisms, and causal edges
2. **Hypothesizes** competing causal explanations with observable predictions
3. **Tests** each hypothesis against every evidence item (diagnostic tests + likelihood ratios)
4. **Updates** posteriors via Bayesian math in odds space
5. **Synthesizes** a written analytical narrative with verdicts

Output: `result.json` (structured data) + `report.html` (Bootstrap dashboard with vis.js network graph).

## Quick Start

```bash
# Install dependencies
pip install litellm pydantic

# Set up API key (any LiteLLM-supported provider)
export GEMINI_API_KEY=your_key_here

# Run on the French Revolution text
python -m pt input_text/revolutions/french_revolution.txt --output-dir output/french_rev

# Use a different model
python -m pt input_text/revolutions/french_revolution.txt --output-dir output/french_rev --model gpt-4o
```

## Architecture

```
pt/
  schemas.py           Pydantic models for all pipeline data
  llm.py               LiteLLM abstraction, structured output parsing
  pass_extract.py      Pass 1: Evidence extraction
  pass_hypothesize.py  Pass 2: Hypothesis generation
  pass_test.py         Pass 3: Diagnostic testing (one LLM call per hypothesis)
  bayesian.py          Pass 3.5: Bayesian updating (pure math, no LLM)
  pass_synthesize.py   Pass 4: Written synthesis
  pipeline.py          Orchestrator
  report.py            HTML report generation
  cli.py               CLI entry point
```

## Tests

```bash
python -m pytest tests/test_pt_bayesian.py tests/test_pt_schemas.py -v
```

## Input Texts

Validated datasets in `input_text/`:
- `revolutions/french_revolution.txt` — Wikipedia, French Revolution
- `american_revolution/american_revolution.txt` — Wikipedia, American Revolution
- `russia_ukraine_debate/westminister_pirchner_v_bryan.txt` — Westminster debate transcript
