# Tests Directory

pytest test suite organized by test type.

## Structure

```
tests/
├── test_pt_schemas.py           # Pydantic model contract tests (likelihood vectors etc.)
├── test_pt_bayesian.py          # Deterministic math: coherent joint update, cap, priors, sensitivity
├── test_pipeline_integration.py # Pipeline/report integration; vector-completeness fail-loud; labeling
├── test_extraction_quality.py   # Deterministic extraction contract tests
├── test_pt_llm.py               # LLM boundary contract tests + opt-in live structured smoke
└── test_assistant.py            # Agentic assistant harness contract tests + opt-in live agent smoke
```

Live LLM smoke tests are gated behind `PT_RUN_LIVE_LLM_TESTS=1` (currently the
`call_llm_structured` smoke in `test_pt_llm.py`). Live assistant/agent smoke is
gated behind `PT_RUN_LIVE_AGENT_TESTS=1` in `test_assistant.py`. The old
standalone `test_litellm_structured.py` / `test_instructor_approach.py` were
removed when the LLM boundary moved to `llm_client.call_llm_structured`.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Fast default suite
PYTHONPATH=. pytest tests -q --tb=short

# Single test
pytest tests/test_pt_bayesian.py::TestRelevanceGating::test_relevance_threshold_blocks_low_relevance -v

# Opt-in live provider smoke tests
PT_RUN_LIVE_LLM_TESTS=1 pytest tests/test_pt_llm.py -v
PT_RUN_LIVE_AGENT_TESTS=1 pytest tests/test_assistant.py -v
```

## Test Types

| Type | Purpose | Speed |
|------|---------|-------|
| **Contract** | Schema and LLM boundary invariants | Fast |
| **Math** | Deterministic Bayesian update behavior | Fast |
| **Integration** | Multi-pass pipeline and report behavior with deterministic boundaries | Medium |
| **Live smoke** | Provider connectivity and structured-output behavior | Slow, opt-in |

## Conventions

1. **Default suite is deterministic** - no live provider calls unless explicitly gated
2. **Real tests preferred** - mock only LLM/provider boundaries with `mock-ok:` justification
3. **Fast execution** - default suite should run in seconds
4. **Mark with plan** - use `@pytest.mark.plans(N)` when implementing a numbered plan

## Adding Tests

1. Add focused deterministic tests for new logic
2. Add integration coverage when multiple pipeline passes or report output are involved
3. Keep live LLM checks opt-in with `PT_RUN_LIVE_LLM_TESTS=1`
4. Run full default verification before PR: `make check`
