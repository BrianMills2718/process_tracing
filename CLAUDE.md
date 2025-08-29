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

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Phase 8 Week 2 - Gateway Integration & First Migrations (Updated 2025-01-29)

**System Status**: **~12% LLM-First** (Week 1 testing revealed lower coverage than estimated)
**Current Priority**: **PROVE GATEWAY WORKS** then migrate simple files
**Critical Issue**: **Gateway exists but integration untested**

**WEEK 1 HONEST ASSESSMENT (2025-01-29):**
- ✅ **Structure Created**: Gateway design, file classification, evidence files
- ⚠️ **Gateway Untested**: Code exists but no integration testing done
- ✅ **Accurate Metrics**: 109 fallback patterns, 6/68 files use require_llm (12% coverage)
- ❌ **No Migrations**: Zero files actually migrated to LLM-first

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Semantic Understanding**: All classification based on LLM analysis, not keyword matching
- **Generalist System**: No dataset-specific hardcoding - system works across all historical periods
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results in structured evidence files

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Architecture
- **Plugin System**: 16 registered plugins requiring LLM-first conversion
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence
- **LLM Integration**: VanEveraLLMInterface with structured Pydantic output (Gemini 2.5 Flash)
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management
- **Universality**: No dataset-specific logic - works across all domains and time periods

### Current Metrics (Validated)
- **Total Python Files**: 68 in core/
- **Fallback Patterns**: 109 (42 return None, 18 return {}/[], 15 hardcoded thresholds, 34 return 0)
- **Files Using require_llm**: 6/68 (8.8%)
- **Files Using semantic_service**: 16/68 (23.5%)
- **TRUE LLM-First Coverage**: ~12%

## 🚀 PHASE 8 WEEK 2: Prove Gateway Works & Start Migration

### Critical Context
**What We Have**: 
- LLM Gateway code (`core/llm_gateway.py`) with 8 methods
- Error handling that properly raises LLMRequiredError
- 109 documented fallback patterns to fix

**What We DON'T Have**:
- Proof the gateway works with existing system
- Any actual migrated files
- Integration tests

### Task 1: Prove Gateway Integration Works

**Objective**: Test if gateway can actually replace semantic_analysis_service calls

**Required Actions**:

1. Create `test_gateway_integration.py`:
```python
#!/usr/bin/env python3
"""
Test that LLM Gateway can replace existing semantic_analysis_service calls.
This is the CRITICAL test - if this fails, the entire approach needs rework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_replace_semantic_service():
    """Find ONE place semantic_service is used and replace with gateway"""
    
    # Step 1: Import both
    from core.semantic_analysis_service import get_semantic_service
    from core.llm_gateway import LLMGateway
    
    # Step 2: Create instances
    semantic_service = get_semantic_service()
    gateway = LLMGateway()
    
    # Step 3: Test same input with both
    test_evidence = "The Boston Tea Party occurred in 1773"
    test_hypothesis = "Colonial protests led to American independence"
    
    # Test semantic service (existing)
    try:
        old_result = semantic_service.assess_probative_value(
            test_evidence, test_hypothesis
        )
        print(f"Semantic service result: {old_result}")
    except Exception as e:
        print(f"Semantic service failed: {e}")
    
    # Test gateway (new)
    try:
        new_result = gateway.calculate_probative_value(
            test_evidence, test_hypothesis, "straw_in_wind"
        )
        print(f"Gateway result: {new_result}")
    except Exception as e:
        print(f"Gateway failed: {e}")
    
    # Compare results
    # Document any incompatibilities

if __name__ == "__main__":
    test_replace_semantic_service()
```

2. Run test and document results in `evidence/current/Evidence_Phase8_Integration_Test.md`

3. If test fails, FIX GATEWAY FIRST before any migrations

**Success Criteria**:
- Gateway method returns comparable results to semantic_service
- Error handling works correctly
- No unexpected exceptions

### Task 2: Fix Critical Gateway Issues

**Objective**: Address issues found in Task 1

**Known Issues to Fix**:

1. **JSON Parsing Assumption**:
```python
# In llm_gateway.py, replace:
result_dict = json.loads(response)

# With:
try:
    result_dict = json.loads(response)
except json.JSONDecodeError:
    # Handle plain text response
    if "supports" in response.lower():
        # Parse as plain text
    else:
        raise LLMRequiredError(f"Invalid LLM response format: {response[:100]}")
```

2. **Rate Limiting Handling**:
Add retry logic to gateway methods

3. Document all fixes in `evidence/current/Evidence_Phase8_Gateway_Fixes.md`

### Task 3: Migrate First Simple File

**Objective**: Complete ONE full migration as proof of concept

**Target File**: `core/enhance_evidence.py` (87 lines, simple structure)

**Migration Steps**:

1. **Analyze Current State**:
```bash
# Find all fallback patterns
grep -n "return None\|except\|if not" core/enhance_evidence.py
```

2. **Create Migrated Version**:
```python
# Replace line 42:
# OLD:
except Exception as e:
    return None

# NEW:
except Exception as e:
    raise LLMRequiredError(f"LLM required for evidence enhancement: {e}")
```

3. **Test Migration**:
```python
# test_enhance_evidence_migration.py
from core.enhance_evidence import enhance_evidence_description
from core.llm_required import LLMRequiredError

def test_migration():
    # Test 1: Normal operation
    result = enhance_evidence_description("Test evidence", {})
    assert result is not None
    
    # Test 2: LLM failure handling
    os.environ['DISABLE_LLM'] = 'true'
    try:
        result = enhance_evidence_description("Test", {})
        assert False, "Should have raised LLMRequiredError"
    except LLMRequiredError:
        print("Correctly raised LLMRequiredError")
```

4. Document in `evidence/current/Evidence_Phase8_First_Migration.md`

### Task 4: Create Migration Validation Script

**Objective**: Automated validation for migrated files

Create `validate_migration.py`:
```python
#!/usr/bin/env python3
"""Validate that a file has been properly migrated to LLM-first"""

import ast
import sys
from pathlib import Path

def validate_file(filepath):
    """Check if file is properly migrated"""
    
    content = Path(filepath).read_text()
    issues = []
    
    # Check 1: No return None patterns
    if "return None" in content:
        issues.append("Still has 'return None' patterns")
    
    # Check 2: No return {} or []
    if "return {}" in content or "return []" in content:
        issues.append("Still has empty return patterns")
    
    # Check 3: Uses LLMRequiredError
    if "LLMRequiredError" not in content and "require_llm" not in content:
        issues.append("Doesn't use LLMRequiredError or require_llm")
    
    # Check 4: No hardcoded thresholds
    import re
    if re.search(r'= 0\.\d+', content) and 'Field' not in content:
        issues.append("Has hardcoded decimal values")
    
    if issues:
        print(f"FAIL - {filepath}:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"PASS - {filepath}")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_file(sys.argv[1])
```

### Task 5: Measure Progress

**Objective**: Track actual progress with evidence

1. Run coverage check:
```bash
python validate_true_llm_coverage.py > evidence/current/Evidence_Phase8_Week2_Coverage.md
```

2. Count migrated files:
```bash
for file in core/*.py core/plugins/*.py; do
    python validate_migration.py "$file"
done | grep PASS | wc -l
```

3. Update metrics in evidence file

### Success Criteria for Week 2

**Minimum Requirements**:
- ✅ Gateway integration test passes
- ✅ At least ONE file fully migrated and validated
- ✅ Migration validation script working
- ✅ Coverage increased from 12% to at least 15%

**Good Progress**:
- 3-5 files migrated
- Coverage at 20%
- No regressions in existing functionality

### Testing Commands

```bash
# Test gateway integration
python test_gateway_integration.py

# Test first migration
python test_enhance_evidence_migration.py

# Validate migration
python validate_migration.py core/enhance_evidence.py

# Check coverage
python validate_true_llm_coverage.py

# Run main system (should still work)
python -m core.analyze sample_data.json
```

## Evidence Files Structure

Create these files in `evidence/current/`:
- `Evidence_Phase8_Integration_Test.md` - Gateway integration test results
- `Evidence_Phase8_Gateway_Fixes.md` - Fixes applied to gateway
- `Evidence_Phase8_First_Migration.md` - First file migration details
- `Evidence_Phase8_Week2_Coverage.md` - Updated coverage metrics

Each evidence file must contain:
- Raw command outputs
- Before/after code snippets
- Test results with actual output
- No false claims - validated data only

## Next Steps (Week 3)

Only proceed to Week 3 after Week 2 success criteria are met:
1. Migrate 5-10 more files using proven process
2. Target plugins with highest impact
3. Remove hardcoded thresholds from advanced_van_evera_prediction_engine.py
4. Aim for 40% coverage

## Critical Implementation Notes

**DO NOT**:
- Claim success without running tests
- Migrate files without validation
- Skip integration testing
- Make assumptions about LLM responses

**ALWAYS**:
- Test with actual data
- Document raw outputs
- Validate with scripts
- Check for regressions

**If Gateway Doesn't Work**: 
Stop migrations and fix gateway first. The entire approach depends on a working gateway.