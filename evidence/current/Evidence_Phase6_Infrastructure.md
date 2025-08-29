# Evidence Phase 6: LLM-Required Infrastructure

## Date: 2025-01-29

## Task 1: Create LLM-Required Infrastructure

### Files Created

#### 1. core/llm_required.py
**Status**: ✅ CREATED

**Key Features**:
- `LLMRequiredError` exception class for clear error messages
- `require_llm()` function that fails fast if LLM unavailable
- Test call to verify LLM is functional
- Support for `DISABLE_LLM` environment variable (for testing)
- `require_llm_lazy()` for module-level initialization

**Code Verification**:
```python
# Key section showing fail-fast behavior:
if not llm:
    raise LLMRequiredError("LLM interface returned None - LLM is required")

# Test that LLM can actually make calls
try:
    test_result = llm.assess_probative_value(...)
    if not test_result:
        raise LLMRequiredError("LLM test call failed - LLM must be functional")
```

#### 2. core/plugins/van_evera_llm_schemas.py
**Status**: ✅ UPDATED

**New Schemas Added**:

1. **ConfidenceFormulaWeights**:
   - Dynamic weights for confidence calculation
   - Replaces hardcoded 0.4, 0.2, 0.2, 0.2 values
   - Includes reasoning field for justification
   - Validates weights sum to ~1.0

2. **SemanticRelevanceAssessment**:
   - Pure semantic understanding (no word counting)
   - Multiple semantic factors (alignment, fit, distance)
   - Replaces ALL word overlap logic
   - Includes detailed reasoning

**Verification**:
Both schemas properly added with Pydantic validation and comprehensive fields.

### Success Criteria Met

✅ LLM requirement utilities created
✅ Fail-fast behavior implemented
✅ New schemas for replacing hardcoded logic
✅ Clear error messages for LLM unavailability

### Next Steps

Proceed to Task 2: Remove ALL Fallbacks from confidence_calculator.py