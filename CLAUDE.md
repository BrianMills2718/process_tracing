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

## 🎯 CURRENT STATUS: Phase 6B - Fix Critical Runtime Errors (Updated 2025-01-29)

**System Status**: **BROKEN (~40% LLM-first)** - Runtime errors prevent execution
**Current Priority**: **FIX RUNTIME ERRORS** - System cannot run due to import and method call failures
**Critical Issue**: **Phase 6 implementation has fundamental errors that will crash at runtime**

**PHASE 6A ATTEMPTED BUT FAILED (2025-01-29):**
- ⚠️ **Infrastructure Created**: llm_required.py has WRONG import path
- ❌ **Method Call Errors**: confidence_calculator.py calls non-existent methods
- ⚠️ **Partial Word Removal**: van_evera_testing_engine.py word overlap deleted
- ❌ **18 Hardcoded Thresholds**: advanced_prediction_engine.py unchanged
- ⚠️ **Fallback Values Remain**: Several 0.5 values still present

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

## 🚀 PHASE 6B: Fix Critical Runtime Errors

### Critical Context
**What's broken**: Phase 6A implementation has fatal runtime errors
**Import Error**: `llm_required.py` uses wrong import path `plugins.` instead of `core.plugins.`
**Method Error**: `confidence_calculator.py` calls methods that don't exist on returned LLM object
**Why it matters**: System will crash immediately when run

### Task 1: Fix Import Path Errors

**Objective**: Correct the import path in llm_required.py

**CRITICAL FIX REQUIRED**:

Fix `core/llm_required.py` line ~13:
```python
# WRONG (current):
from plugins.van_evera_llm_interface import get_van_evera_llm

# CORRECT (must be):
from core.plugins.van_evera_llm_interface import get_van_evera_llm
```

**Validation**: After fix, run:
```bash
python -c "from core.llm_required import require_llm; print('Import successful')"
```

Document fix in `evidence/current/Evidence_Phase6B_ImportFix.md`

### Task 2: Fix Method Call Errors

**Objective**: Fix confidence_calculator.py calling non-existent methods

**CRITICAL PROBLEM**: 
The `require_llm()` function returns a generic VanEveraLLMInterface object, but confidence_calculator.py calls methods that DON'T EXIST on that object:
- `determine_confidence_weights()` - DOESN'T EXIST
- `assess_causal_mechanism()` - DOESN'T EXIST
- `determine_causal_weights()` - DOESN'T EXIST
- `determine_robustness_weights()` - DOESN'T EXIST
- `determine_overall_confidence_weights()` - DOESN'T EXIST

**SOLUTION APPROACH**:

Option 1: Add wrapper class that provides needed methods:
```python
# In core/llm_required.py after imports:
class LLMWithConfidenceMethods:
    """Wrapper that adds confidence-specific methods to base LLM"""
    
    def __init__(self, base_llm):
        self.base_llm = base_llm
        # Expose base methods
        self.__dict__.update(base_llm.__dict__)
    
    def determine_confidence_weights(self, context):
        """Get dynamic weights for confidence formula"""
        # Call base LLM with structured prompt
        prompt = f"Determine confidence formula weights for: {context}"
        response = self.base_llm.generate_structured_output(
            prompt, 
            output_model=ConfidenceFormulaWeights
        )
        return response
    
    def assess_causal_mechanism(self, hypothesis, evidence):
        """Assess causal mechanism strength"""
        # Similar implementation...
```

Option 2: Add methods directly to VanEveraLLMInterface class in `core/plugins/van_evera_llm_interface.py`

**Validation**: After fix, run:
```bash
python -c "from core.confidence_calculator import CausalConfidenceCalculator; c = CausalConfidenceCalculator()"
```

Document fix in `evidence/current/Evidence_Phase6B_MethodFix.md`

### Task 3: Remove Remaining Fallback Values

**Objective**: Remove ALL hardcoded fallback values that remain

**CRITICAL MISSED VALUES**:
Current implementation still has these fallback values that MUST be removed:

1. **confidence_calculator.py line ~375**:
```python
# WRONG (current):
ratio_component = 0.5  # When no ratios exist

# CORRECT (must be):
# Require LLM assessment for neutral state
ratio_component = self.llm.assess_neutral_state("no_ratios_available").value
```

2. **confidence_calculator.py line ~337**:
```python
# WRONG (current):
evidence_balance = 0.5  # Neutral when no evidence

# CORRECT (must be):
evidence_balance = self.llm.assess_evidence_balance(supporting=0, challenging=0).balance
```

3. **confidence_calculator.py line ~541**:
```python
# WRONG (current):
sensitivity = 0.5  # Default sensitivity

# CORRECT (must be):
sensitivity = self.llm.determine_sensitivity(context).value
```

4. **confidence_calculator.py line ~605**:
```python
# WRONG (current):
independence = 0.5  # Moderate independence  

# CORRECT (must be):
independence = self.llm.assess_independence(evidence_items).score
```

**Validation**: After removing ALL fallbacks:
```bash
grep -n "= 0\.[0-9]" core/confidence_calculator.py | grep -v "Field"
# Should return NOTHING
```

Document in `evidence/current/Evidence_Phase6B_FallbackRemoval.md`

### Task 4: Fix advanced_prediction_engine.py Thresholds

**Objective**: Replace 18 hardcoded thresholds with LLM-determined values

**CRITICAL**: This file has NOT been modified yet. All 18 thresholds remain hardcoded.

**Required Refactoring**:

1. Add LLM requirement to `__init__`:
```python
def __init__(self):
    super().__init__()
    from core.llm_required import require_llm
    self.llm = require_llm()
    # Remove static DOMAIN_PREDICTION_STRATEGIES
    self.strategies = self._build_dynamic_strategies()
```

2. Replace static dictionary with dynamic builder:
```python
def _build_dynamic_strategies(self):
    """Build strategies with LLM-determined thresholds"""
    strategies = {}
    for domain in PredictionDomain:
        strategies[domain] = self._get_domain_strategy(domain)
    return strategies

def _get_domain_strategy(self, domain):
    """Get strategy with dynamic thresholds from LLM"""
    # Lines 93, 102, 111, 120, 129, 180, 189, 198, 207, 250, 259, 268, 311, 320, 329
    threshold = self.llm.determine_threshold(
        domain=domain.value,
        context="Van Evera quantitative threshold"
    ).value
    
    # Build strategy with dynamic threshold
    return {
        'quantitative_threshold': threshold,
        # ... rest of strategy
    }
```

**Validation**: After fix:
```bash
grep -n "'quantitative_threshold': 0\." core/plugins/advanced_van_evera_prediction_engine.py
# Should return NOTHING
```

Document in `evidence/current/Evidence_Phase6B_Thresholds.md`

### Task 5: Comprehensive Validation

**Objective**: Create validation script that ACTUALLY tests the fixes

**CRITICAL**: The existing validate_strict_llm_first.py won't catch the runtime errors!

**Create `validate_phase6b_fixes.py`**:
```python
#!/usr/bin/env python3
"""
Validate Phase 6B fixes for runtime errors and LLM-first compliance.
"""

import os
import sys
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import_paths():
    """Test that import paths are correct"""
    print("\n[TEST] Checking import paths...")
    
    # Check llm_required.py
    llm_req_path = Path("core/llm_required.py")
    if llm_req_path.exists():
        content = llm_req_path.read_text()
        if "from plugins." in content:
            print("[FAIL] Wrong import path: 'from plugins.' should be 'from core.plugins.'")
            return False
        elif "from core.plugins." in content:
            print("[OK] Import path is correct")
        else:
            print("[FAIL] No import found in llm_required.py")
            return False
    else:
        print("[FAIL] llm_required.py not found")
        return False
    
    # Try importing
    try:
        from core.llm_required import require_llm
        print("[OK] Import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_method_existence():
    """Test that called methods actually exist"""
    print("\n[TEST] Checking method existence...")
    
    try:
        # Check if confidence calculator can instantiate
        from core.confidence_calculator import CausalConfidenceCalculator
        
        # Check if it tries to call non-existent methods
        calc_content = Path("core/confidence_calculator.py").read_text()
        
        problematic_calls = [
            "self.llm.determine_confidence_weights",
            "self.llm.assess_causal_mechanism",
            "self.llm.determine_causal_weights",
            "self.llm.determine_robustness_weights",
            "self.llm.determine_overall_confidence_weights"
        ]
        
        found_problems = []
        for call in problematic_calls:
            if call in calc_content:
                found_problems.append(call)
        
        if found_problems:
            print(f"[FAIL] Calls to non-existent methods: {found_problems}")
            print("[INFO] These methods don't exist on VanEveraLLMInterface")
            return False
        else:
            print("[OK] No calls to non-existent methods")
            return True
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_fallback_values():
    """Check for remaining fallback values"""
    print("\n[TEST] Checking for fallback values...")
    
    # Check confidence_calculator.py for specific fallbacks
    calc_path = Path("core/confidence_calculator.py")
    if calc_path.exists():
        content = calc_path.read_text()
        lines = content.split('\n')
        
        fallbacks_found = []
        for i, line in enumerate(lines, 1):
            if '= 0.5' in line and 'Field' not in line:
                fallbacks_found.append(f"Line {i}: {line.strip()[:60]}")
        
        if fallbacks_found:
            print(f"[FAIL] Found {len(fallbacks_found)} fallback values:")
            for fb in fallbacks_found[:5]:
                print(f"  {fb}")
            return False
        else:
            print("[OK] No 0.5 fallback values found")
    
    return True

def test_hardcoded_thresholds():
    """Check for hardcoded thresholds in prediction engine"""
    print("\n[TEST] Checking prediction engine thresholds...")
    
    pred_path = Path("core/plugins/advanced_van_evera_prediction_engine.py")
    if pred_path.exists():
        content = pred_path.read_text()
        
        # Count quantitative_threshold occurrences
        threshold_pattern = r"'quantitative_threshold':\s*0\.\d+"
        matches = re.findall(threshold_pattern, content)
        
        if matches:
            print(f"[FAIL] Found {len(matches)} hardcoded thresholds")
            print(f"  First 3: {matches[:3]}")
            return False
        else:
            print("[OK] No hardcoded thresholds found")
            return True
    else:
        print("[SKIP] Prediction engine not found")
        return True

def test_llm_required():
    """Test that system fails without LLM"""
    print("\n[TEST] Checking if system fails without LLM...")
    
    # Set environment to disable LLM
    os.environ['DISABLE_LLM'] = 'true'
    
    try:
        from core.llm_required import require_llm
        llm = require_llm()
        print("[FAIL] require_llm() should have failed but didn't!")
        return False
    except Exception as e:
        if "LLM" in str(e):
            print(f"[OK] System correctly failed: {str(e)[:50]}")
            return True
        else:
            print(f"[FAIL] Wrong error: {e}")
            return False
    finally:
        # Clear the environment variable
        if 'DISABLE_LLM' in os.environ:
            del os.environ['DISABLE_LLM']

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("PHASE 6B VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Import Paths", test_import_paths),
        ("Method Existence", test_method_existence),
        ("Fallback Values", test_fallback_values),
        ("Hardcoded Thresholds", test_hardcoded_thresholds),
        ("LLM Required", test_llm_required)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All fixes validated!")
    else:
        print("[FAIL] Critical issues remain - fix before proceeding")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Success Criteria

**Must fix ALL runtime errors**:
- ✅ Import paths corrected (plugins. → core.plugins.)
- ✅ Method calls match actual available methods
- ✅ System can instantiate without crashing
- ✅ All fallback values removed (no 0.5 defaults)
- ✅ 18 hardcoded thresholds replaced with LLM calls

**Then achieve TRUE LLM-first**:
- ✅ System fails immediately without LLM
- ✅ ZERO word overlap/counting patterns
- ✅ ZERO hardcoded values (except Pydantic Field defaults)
- ✅ All formulas use LLM-generated weights
- ✅ No try/except that hides LLM failures

### Testing Commands

```bash
# Run Phase 6B validation (MUST PASS FIRST)
python validate_phase6b_fixes.py

# Test individual fixes
python -c "from core.llm_required import require_llm; print('Import works')"
python -c "from core.confidence_calculator import CausalConfidenceCalculator; print('No method errors')"

# Check for violations
grep -n "from plugins\." core/llm_required.py  # Should be empty
grep -n "= 0\.5" core/confidence_calculator.py | grep -v Field  # Should be empty
grep -n "'quantitative_threshold': 0\." core/plugins/advanced_van_evera_prediction_engine.py  # Should be empty

# Test with LLM disabled (must fail)
DISABLE_LLM=true python -m core.analyze test_data/sample.json
```

## Evidence Files Structure

Create these files in `evidence/current/`:
- `Evidence_Phase6B_ImportFix.md` - Fixed import path errors
- `Evidence_Phase6B_MethodFix.md` - Fixed method call errors  
- `Evidence_Phase6B_FallbackRemoval.md` - Removed ALL 0.5 values
- `Evidence_Phase6B_Thresholds.md` - Fixed 18 hardcoded thresholds
- `Evidence_Phase6B_Validation.md` - Final validation showing ALL TESTS PASS

Each evidence file must contain:
- Exact line numbers and changes made
- Before/after code snippets showing fixes
- Test output proving fix works
- No false claims - actual working code only

## Critical Implementation Notes

**DO NOT CLAIM SUCCESS WITHOUT EVIDENCE**:
- Every fix must be tested with actual Python execution
- Method calls must be verified to exist on the actual object
- Import statements must successfully import
- No mocking or stubbing - real implementations only

**Common Pitfalls to Avoid**:
1. Don't call methods that don't exist on the returned object type
2. Don't use wrong import paths (always use core.plugins not plugins)
3. Don't leave ANY hardcoded values (search thoroughly)
4. Don't claim completion without running validation script
5. Don't create circular imports when fixing method issues

## Next Phase (Only After 6B Complete)

Phase 7: Enhanced Van Evera Features
- Test generation improvements
- Counterfactual analysis
- Causal mechanism detection
- Multi-hypothesis relationships

**BUT FIRST**: Fix ALL runtime errors in Phase 6B!