# Tests Directory

pytest test suite organized by test type.

## Structure

```
tests/
├── test_pt_schemas.py          # Pydantic model contract tests
├── test_pt_bayesian.py         # Deterministic Bayesian math tests
├── test_pipeline_integration.py # Deterministic pipeline/report integration tests
├── test_extraction_quality.py  # Deterministic extraction contract tests
├── test_pt_llm.py             # LLM boundary contract tests
├── test_litellm_structured.py # Opt-in live LiteLLM smoke tests
└── test_instructor_approach.py # Opt-in live structured-output smoke tests
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Fast default suite
PYTHONPATH=. pytest tests -q --tb=short

# Single test
pytest tests/test_pt_bayesian.py::TestRelevanceGating::test_relevance_threshold_blocks_low_relevance -v

# Opt-in live provider smoke tests
PT_RUN_LIVE_LLM_TESTS=1 pytest tests/test_litellm_structured.py tests/test_instructor_approach.py -v
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
