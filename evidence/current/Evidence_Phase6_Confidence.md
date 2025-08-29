# Evidence Phase 6: Confidence Calculator Changes

## Date: 2025-01-29

## Task 2: Remove ALL Fallbacks from confidence_calculator.py

### Changes Made

#### 1. Updated __init__ to Require LLM
**Before**:
```python
self._llm_interface = None  # Lazy load LLM interface
```

**After**:
```python
from .llm_required import require_llm
self.llm = require_llm()  # Will raise LLMRequiredError if unavailable
```

#### 2. Removed ALL Fallback Values

**Causal Mechanism (lines ~303-306)**:
- REMOVED: `mechanism_completeness = 0.7  # Default fallback`
- REMOVED: `temporal_consistency = 0.8  # Default fallback`
- REPLACED with: Direct LLM call without try/except

**Coherence (line ~359)**:
- REMOVED: `base_coherence = 0.8  # Default fallback`
- REPLACED with: Required LLM threshold check

**Independence (line ~389)**:
- REMOVED: `independence_score = 0.8  # Default fallback`
- REPLACED with: Required LLM threshold check

**Posterior Uncertainty (line ~497)**:
- REMOVED: `posterior_uncertainty = 0.1  # Default fallback`
- REPLACED with: Required LLM threshold check

#### 3. Replaced ALL Hardcoded Formula Weights

**Evidential Confidence (lines 273-277)**:
- REMOVED: `0.4 * quality_score + 0.2 * quantity_factor + ...`
- REPLACED with: LLM-determined weights via `determine_confidence_weights()`

**Causal Confidence (lines 309-314)**:
- REMOVED: `0.4 * posterior + 0.3 * ratio + 0.2 * mechanism + 0.1 * temporal`
- REPLACED with: LLM-determined weights via `determine_causal_weights()`

**Robustness Confidence (lines 392-397)**:
- REMOVED: `0.3 * diversity + 0.3 * consistency + 0.2 * balance + 0.2 * independence`
- REPLACED with: LLM-determined weights via `determine_robustness_weights()`

**Overall Confidence (lines 553-559)**:
- REMOVED: `{EVIDENTIAL: 0.30, CAUSAL: 0.25, COHERENCE: 0.20, ...}`
- REPLACED with: LLM-determined weights via `determine_overall_confidence_weights()`

### LLM Interface Methods Added

Added to van_evera_llm_interface.py:
1. `determine_confidence_weights()` - For evidential formula
2. `determine_causal_weights()` - For causal formula
3. `determine_robustness_weights()` - For robustness formula
4. `determine_overall_confidence_weights()` - For overall aggregation

All methods return `ConfidenceFormulaWeights` with dynamic, context-aware values.

### Verification

**Key Changes**:
- NO fallback values remain
- NO try/except that continues without LLM
- ALL formulas use LLM-determined weights
- System will FAIL if LLM unavailable

### Success Criteria Met

✅ Removed ALL fallback values
✅ Made LLM mandatory in __init__
✅ Replaced hardcoded formula weights
✅ No silent failure paths

### Next Steps

Proceed to Task 3: Remove Word Overlap from van_evera_testing_engine.py