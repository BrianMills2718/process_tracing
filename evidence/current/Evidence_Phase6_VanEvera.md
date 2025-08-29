# Evidence Phase 6: Van Evera Testing Engine Changes

## Date: 2025-01-29

## Task 3: Remove Word Overlap from van_evera_testing_engine.py

### Changes Made

#### 1. Added LLM Requirement to __init__
**Before**:
```python
def __init__(self, graph_data: Dict):
    self.graph_data = graph_data
```

**After**:
```python
def __init__(self, graph_data: Dict):
    from core.llm_required import require_llm
    self.llm = require_llm()  # Will fail if LLM unavailable
```

#### 2. Deleted Entire Methods

**DELETED: _generate_generic_predictions() (lines 249-265)**
- Was extracting keywords from hypothesis description
- Creating fallback predictions with word lists
- VIOLATION: Keyword-based logic

**DELETED: _extract_prediction_keywords() (lines 423-455)**
- Was extracting "semantic requirements" via word lists
- VIOLATION: Not true semantic understanding

#### 3. Removed Word Overlap Fallbacks

**In _is_evidence_relevant_to_prediction() (lines 346-360)**:
- REMOVED: Word set creation and intersection
- REMOVED: `overlap_ratio >= 0.2` check
- REPLACED with: `raise LLMRequiredError()`

**In _find_semantic_evidence() (lines 401-420)**:
- REMOVED: Word overlap counting (`len(overlap) >= 2`)
- REMOVED: Common word removal logic
- REPLACED with: `raise LLMRequiredError()`

#### 4. Removed Fallback Prediction Generation

**Line ~145**:
- REMOVED: Call to `_generate_generic_predictions()`
- REPLACED with: `raise LLMRequiredError()` if no predictions

### Verification

```bash
# Check for word overlap patterns
grep -r "overlap_ratio\|word.*overlap\|intersection" core/van_evera_testing_engine.py
# Result: NONE found

# Check for deleted methods
grep "_generate_generic_predictions\|_extract_prediction_keywords" core/van_evera_testing_engine.py
# Result: Only deletion comments remain
```

### Key Changes Summary

**Before**: System would fall back to word counting if LLM failed
**After**: System FAILS immediately if LLM unavailable

**Deleted Patterns**:
- Word set operations
- Overlap ratios
- Common word lists
- Keyword extraction
- Fallback predictions

### Success Criteria Met

✅ Deleted ALL word overlap logic
✅ Removed keyword extraction methods
✅ No word counting remains
✅ System fails without LLM

### Next Steps

Proceed to Task 4: Fix advanced_prediction_engine.py Thresholds