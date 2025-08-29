# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 PERMANENT INFORMATION -- DO NOT CHANGE ON UPDATES

### LLM-First Architecture Policy (MANDATORY)

**CORE PRINCIPLE**: This system is **LLM-FIRST** with **ZERO TOLERANCE** for rule-based or keyword-based implementations.

**PROHIBITED IMPLEMENTATIONS**:
- ❌ Keyword matching for evidence classification (`if 'ideological' in text`)
- ❌ Hardcoded probative value assignments (`probative_value = 0.7`)
- ❌ Rule-based contradiction detection (`if 'before' in hypothesis and 'after' in evidence`)
- ❌ Domain classification using keyword lists
- ❌ Confidence thresholds based on hardcoded ranges
- ❌ Any `if/elif` chains for semantic understanding
- ❌ Dataset-specific logic (American Revolution hardcoded rules)
- ❌ Historical period-specific keyword matching
- ❌ Word overlap/counting for semantic decisions
- ❌ Fallback values that hide LLM unavailability

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding
- ✅ FAIL-FAST when LLM unavailable (no silent fallbacks)
- ✅ Dynamic formula weights from LLM (no hardcoded calculations)

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Phase 6 - Complete TRUE LLM-First Migration (Updated 2025-01-29)

**System Status**: **Hybrid Mode (71%)** - LLM optional, not required
**Current Priority**: **REMOVE ALL FALLBACKS** - Make LLM mandatory
**Critical Issue**: **System still operates with fallback logic**

**PHASE 5 PARTIALLY COMPLETED (2025-01-29):**
- ✅ **Infrastructure Built**: LLM assessment schemas and methods created
- ✅ **Integration Added**: Confidence calculator and testing engine augmented
- ⚠️ **Fallbacks Remain**: Hardcoded values still present as "defaults"
- ❌ **Word Overlap**: Semantic decisions still use word counting
- ❌ **Optional LLM**: System works without LLM (violates policy)

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability
- **LLM-REQUIRED**: System must fail immediately if LLM unavailable

### Quality Standards
- **Semantic Understanding**: All classification based on LLM analysis, not keyword matching
- **Generalist System**: No dataset-specific hardcoding - system works across all historical periods
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results in structured evidence files
- **Zero Fallbacks**: No try/except that continues without LLM

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Architecture
- **Plugin System**: 16 registered plugins requiring TRUE LLM-first conversion
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence
- **LLM Integration**: VanEveraLLMInterface with structured Pydantic output (Gemini 2.5 Flash)
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management
- **Universality**: No dataset-specific logic - works across all domains and time periods
- **Fail-Fast**: System must error if LLM unavailable

## 🚀 PHASE 6: Complete TRUE LLM-First Migration

### Critical Context
**What exists**: LLM infrastructure added but not required
**What's wrong**: Fallback logic allows system to run without LLM
**Why it matters**: Violates ZERO TOLERANCE policy for non-LLM decisions

### Task 1: Create LLM-Required Infrastructure

**Objective**: Add utilities that enforce LLM availability

**Required Actions**:

1. Create `core/llm_required.py`:
```python
"""
LLM requirement enforcement utilities.
System MUST fail if LLM is unavailable.
"""

class LLMRequiredError(Exception):
    """Raised when LLM is required but unavailable"""
    pass

def require_llm():
    """
    Ensure LLM is available or fail immediately.
    NO FALLBACKS ALLOWED.
    """
    try:
        from plugins.van_evera_llm_interface import get_van_evera_llm
        llm = get_van_evera_llm()
        if not llm:
            raise LLMRequiredError("LLM interface required but not available")
        return llm
    except Exception as e:
        raise LLMRequiredError(f"Cannot operate without LLM: {e}")
```

2. Add new schemas to `van_evera_llm_schemas.py`:
```python
class ConfidenceFormulaWeights(BaseModel):
    """LLM determines appropriate weights for confidence calculation"""
    quality_weight: float = Field(ge=0.0, le=1.0)
    quantity_weight: float = Field(ge=0.0, le=1.0)
    diversity_weight: float = Field(ge=0.0, le=1.0)
    balance_weight: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Justification for weight selection")
    
class SemanticRelevanceAssessment(BaseModel):
    """Replace ALL word overlap with semantic assessment"""
    is_relevant: bool
    relevance_score: float = Field(ge=0.0, le=1.0)
    semantic_relationship: str
    reasoning: str
```

3. Document in `evidence/current/Evidence_Phase6_Infrastructure.md`

### Task 2: Remove ALL Fallbacks from confidence_calculator.py

**Objective**: Make confidence calculation require LLM

**Required Changes**:

1. Update `__init__` method:
```python
def __init__(self):
    from core.llm_required import require_llm
    self.llm = require_llm()  # Fail immediately if no LLM
    self.evidence_quantifier = EvidenceStrengthQuantifier()
    self.assessment_history = []
```

2. Remove ALL fallback values:
- Line ~303: DELETE `mechanism_completeness = 0.7  # Default fallback`
- Line ~306: DELETE `temporal_consistency = 0.8  # Default fallback`
- Line ~359: DELETE `base_coherence = 0.8  # Default fallback`
- Line ~389: DELETE `independence_score = 0.8  # Default fallback`
- Line ~497: DELETE `posterior_uncertainty = 0.1  # Default fallback`

Replace each with:
```python
# No fallback - LLM required
causal_assessment = self.llm.assess_causal_mechanism(...)
mechanism_completeness = causal_assessment.mechanism_completeness
```

3. Replace hardcoded formula weights (lines ~273-277):
```python
# OLD:
evidential_confidence = (
    0.4 * quality_score +
    0.2 * quantity_factor +
    ...
)

# NEW:
weights = self.llm.determine_confidence_weights(context)
evidential_confidence = (
    weights.quality_weight * quality_score +
    weights.quantity_weight * quantity_factor +
    ...
)
```

4. Document changes in `evidence/current/Evidence_Phase6_Confidence.md`

### Task 3: Remove Word Overlap from van_evera_testing_engine.py

**Objective**: Delete ALL word counting/overlap logic

**Required Deletions**:

1. DELETE entire method `_generate_generic_predictions()` (lines 249-265)

2. DELETE word overlap in `_is_evidence_relevant_to_prediction()` (lines 346-360):
```python
# DELETE THIS ENTIRE SECTION:
# Fallback to basic non-keyword analysis
evidence_words = set(evidence_text.lower().split())
...
return overlap_ratio >= 0.2
```

Replace with:
```python
# No fallback - LLM required
raise LLMRequiredError("LLM assessment required for relevance")
```

3. DELETE word overlap in `_find_semantic_evidence()` (lines 401-420)

4. DELETE entire method `_extract_prediction_keywords()` (lines 423-455)

5. Document in `evidence/current/Evidence_Phase6_VanEvera.md`

### Task 4: Fix advanced_prediction_engine.py Thresholds

**Objective**: Replace 18 hardcoded thresholds

**Required Changes**:

1. Find and replace ALL patterns like:
```python
'quantitative_threshold': 0.70,  # Lines 93, 102, 111, etc.
```

With:
```python
'quantitative_threshold': self.llm.determine_threshold(context),
```

2. Replace weight dictionaries (lines 370-382):
```python
# OLD:
'weight': 0.25,

# NEW:
'weight': self.llm.determine_criterion_weight(criterion_name),
```

3. Document in `evidence/current/Evidence_Phase6_Prediction.md`

### Task 5: Create Strict Validation

**Objective**: Verify TRUE LLM-first compliance

**Create `validate_strict_llm_first.py`**:
```python
#!/usr/bin/env python3
"""
Strict validation for TRUE LLM-first architecture.
System MUST fail without LLM - no fallbacks allowed.
"""

import os
import sys
import re
from pathlib import Path

def test_llm_required():
    """Test that system fails without LLM"""
    # Disable LLM
    os.environ['DISABLE_LLM'] = 'true'
    
    try:
        from core.analyze import run_analysis
        result = run_analysis("test_data/sample.json")
        print("[FAIL] System ran without LLM - fallbacks still exist!")
        return False
    except Exception as e:
        if "LLM" in str(e) or "required" in str(e):
            print("[OK] System correctly failed without LLM")
            return True
        else:
            print(f"[FAIL] Wrong error: {e}")
            return False

def check_no_hardcoded_values():
    """Verify no hardcoded probability/confidence values"""
    files_to_check = [
        "core/confidence_calculator.py",
        "core/van_evera_testing_engine.py",
        "core/plugins/advanced_van_evera_prediction_engine.py"
    ]
    
    pattern = r'= 0\.\d+(?!.*Field)'  # Exclude Pydantic Field defaults
    
    for file_path in files_to_check:
        with open(file_path, 'r') as f:
            content = f.read()
        matches = re.findall(pattern, content)
        if matches:
            print(f"[FAIL] {file_path} has hardcoded values: {matches[:3]}")
            return False
    
    print("[OK] No hardcoded values found")
    return True

def check_no_word_overlap():
    """Verify no word overlap/counting logic"""
    forbidden_patterns = [
        r'overlap_ratio',
        r'len\(overlap\)',
        r'intersection\(',
        r'word.*overlap',
        r'evidence_words.*prediction_words'
    ]
    
    for pattern in forbidden_patterns:
        result = os.popen(f'grep -r "{pattern}" core/').read()
        if result:
            print(f"[FAIL] Found word overlap pattern: {pattern}")
            return False
    
    print("[OK] No word overlap patterns found")
    return True

if __name__ == "__main__":
    tests = [
        test_llm_required(),
        check_no_hardcoded_values(),
        check_no_word_overlap()
    ]
    
    if all(tests):
        print("\n✅ TRUE LLM-FIRST ACHIEVED!")
    else:
        print("\n❌ Violations remain - not LLM-first")
        sys.exit(1)
```

### Success Criteria

**Must demonstrate**:
- ✅ System fails immediately without LLM (no fallbacks)
- ✅ ZERO word overlap/counting patterns
- ✅ ZERO hardcoded values (except Pydantic schema defaults)
- ✅ All formulas use LLM-generated weights
- ✅ No try/except that hides LLM failures

### Expected Outcomes

**After Phase 6**:
- System is 100% LLM-first
- No semantic decisions without LLM
- Clear errors when LLM unavailable
- Dynamic, context-aware calculations throughout

### Testing Commands

```bash
# Run strict validation
python validate_strict_llm_first.py

# Test with LLM disabled (must fail)
DISABLE_LLM=true python -m core.analyze test_data/sample.json

# Search for violations
grep -r "overlap_ratio\|= 0\.[0-9]" core/
```

## Evidence Files Structure

Create these files in `evidence/current/`:
- `Evidence_Phase6_Infrastructure.md` - LLM requirement utilities
- `Evidence_Phase6_Confidence.md` - Confidence calculator changes
- `Evidence_Phase6_VanEvera.md` - Testing engine word overlap removal
- `Evidence_Phase6_Prediction.md` - Prediction engine threshold replacement
- `Evidence_Phase6_Validation.md` - Final validation results

Each evidence file must contain:
- Raw command outputs
- Before/after code snippets
- Error logs showing LLM requirement
- Success/failure determination

## Next Steps After Phase 6

Once TRUE LLM-first is achieved:
1. Enhanced Van Evera test generation
2. Counterfactual analysis implementation
3. Causal mechanism strengthening
4. Multi-hypothesis relationship analysis