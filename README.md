# Process Tracing Toolkit (LLM-Enhanced)

## Overview

This toolkit supports advanced, multi-case, hypothesis-driven qualitative analysis using process tracing methodology, now deeply integrated with LLM (Gemini API) for enhanced causal reasoning, evidence assessment, and reporting.

## Key Features

- **LLM-Powered Graph Extraction:**
  - Extracts detailed, causally rich process tracing graphs from case texts using a comprehensive prompt and Gemini API.

- **LLM-Aided Mechanism Elaboration:**
  - After initial extraction, each Causal_Mechanism node is further analyzed by the LLM to:
    - Write a narrative connecting its constituent events.
    - Suggest missing micro-steps (as new Event nodes).
    - Assess internal coherence.
    - Suggest refined properties (confidence, level_of_detail).
  - Results are included in the HTML report for each mechanism.

- **LLM-Aided Evidence Refinement:**
  - For each Hypothesis-Evidence link, the LLM:
    - Refines the Van Evera diagnostic type (hoop, smoking_gun, etc.).
    - Suggests a more nuanced probative value (including Bayesian likelihoods).
    - Provides a textual justification for its assessment.
  - These refinements are shown in the evidence tables and used in balance calculations.

- **LLM-Generated Analytical Narrative Summaries:**
  - The LLM generates concise, analytical summaries for:
    - Causal chains (highlighting triggers and outcomes).
    - Each mechanism and hypothesis evaluation.
    - The overall cross-case synthesis (multi-case studies).
  - These summaries are embedded in the HTML reports for richer, more readable output.

- **LLM-Powered Counterfactual Exploration:**
  - Analyze "what if" scenarios by specifying a counterfactual premise and outcome of interest.
  - The LLM traces consequences through the causal graph and provides a structured analysis.
  - Run via the CLI script (see below).

## Directory Structure

- `process_trace_advanced.py` — Main extraction and pipeline orchestrator.
- `core/analyze.py` — Per-case graph analysis, now with LLM-powered mechanism and evidence enhancement.
- `core/enhance_mechanisms.py` — LLM mechanism elaboration logic.
- `core/enhance_evidence.py` — LLM evidence refinement logic.
- `core/llm_reporting_utils.py` — LLM narrative summary generation.
- `core/counterfactual_analyzer.py` — LLM-powered counterfactual analysis.
- `run_study.py` — Multi-case orchestrator.
- `run_counterfactual_analysis.py` — CLI for counterfactual analysis.

## Usage

### 1. Standard Multi-Case Analysis

Prepare a `study_config.json` specifying your cases, (optional) global hypothesis, and output directory. Then run:

```bash
python run_study.py study_config.json
```

This will:
- Extract and analyze each case, including LLM-powered mechanism and evidence enhancement.
- Generate HTML reports with LLM-generated summaries.
- Perform cross-case synthesis with an LLM summary at the top of the report.

### 2. Single-Case Analysis

You can run the pipeline on a single case file using `process_trace_advanced.py` or analyze an existing graph with:

```bash
python -m core.analyze <case_graph.json> --html
```

### 3. Counterfactual Analysis

To explore counterfactual scenarios:

```bash
python run_counterfactual_analysis.py --graph_json <path_to_case_graph.json> --premise "What if Event X did not happen?" --outcome_id <EventY>
```

- The result will be printed and can be saved with `--output <result.txt>`.

## Requirements

- Python 3.8+
- `google-genai` (Gemini API)
- `networkx`, `matplotlib`, `dotenv`, etc. (see `requirements.txt`)

## LLM API Key

- Place your Gemini API key in a `.env` file as `GOOGLE_API_KEY=your_key_here` or set it in your environment.

## Notes

- LLM calls may incur costs and can be slow for large graphs or many cases.
- All LLM prompts and responses are saved for debugging.
- The toolkit is robust to large outputs and will chunk or save data as needed.

## Advanced

- You can further customize prompts or add new LLM-powered modules by following the structure in `core/enhance_mechanisms.py`, `core/enhance_evidence.py`, and `core/llm_reporting_utils.py`.

---

For more details, see the docstrings in each module and the comments in the main scripts.
