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

## 🎯 CURRENT STATUS: Phase 8 - Systematic LLM-First Migration (Updated 2025-01-29)

**System Status**: **PARTIALLY WORKING (~30% LLM-first)** - Main path requires LLM but most files have fallbacks
**Current Priority**: **SYSTEMATIC MIGRATION** - Follow MASTER_PLAN_100_PERCENT_LLM_FIRST.md
**Critical Issue**: **70% of system still has fallbacks, hardcoded values, and bypass paths**

**PHASE 6B/7 COMPLETED (2025-01-29):**
- ✅ **Fixed semantic_analysis_service.py**: Removed 8 fallbacks, now requires LLM
- ✅ **Fixed van_evera_testing_engine.py**: Removed all hardcoded values
- ✅ **Fixed import paths**: llm_required.py now works correctly
- ⚠️ **Honest Assessment**: System is ~30% LLM-first, not 90% as initially claimed
- ✅ **Created Master Plan**: MASTER_PLAN_100_PERCENT_LLM_FIRST.md for systematic migration

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

## 🚀 PHASE 8: Systematic LLM-First Migration Week 1

### Critical Context
**Current State**: ~30% LLM-first (validated by validate_true_llm_coverage.py)
**Master Plan**: See MASTER_PLAN_100_PERCENT_LLM_FIRST.md for full 5-week plan
**This Week**: Foundation - Assessment & Central Gateway Creation

### Task 1: Complete File Classification (Day 1-2)

**Objective**: Classify all 67 Python files into categories for migration strategy

**Required Actions**:

1. Create `evidence/current/Evidence_Phase8_Classification.md`

2. Run classification script:
```python
# classify_files.py
import os
from pathlib import Path
import ast

def classify_file(filepath):
    """Classify file as: semantic, computational, hybrid, or dead"""
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Check for semantic indicators
    semantic_keywords = ['hypothesis', 'evidence', 'confidence', 'domain', 
                         'classify', 'assess', 'evaluate', 'relationship']
    has_semantic = any(kw in content.lower() for kw in semantic_keywords)
    
    # Check for LLM usage
    uses_llm = 'llm' in content.lower() or 'query_llm' in content
    
    # Check for imports from other files
    try:
        tree = ast.parse(content)
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    except:
        imports = []
    
    # Classification logic
    if 'confidence_calculator.py' in str(filepath):
        return 'dead'  # Known unused
    elif has_semantic or uses_llm:
        return 'semantic'
    elif any(term in content for term in ['networkx', 'graph', 'algorithm']):
        return 'computational'
    else:
        return 'hybrid'

# Classify all files
for file in Path('core').rglob('*.py'):
    category = classify_file(file)
    print(f"{category}: {file}")
```

3. Document results in categories:
   - **Category A (Semantic)**: MUST be LLM-first
   - **Category B (Computational)**: Don't need LLM
   - **Category C (Hybrid)**: Selective LLM
   - **Category D (Dead)**: Delete these

**Validation**: 
- Total files classified: 67
- Each file assigned to exactly one category
- Document rationale for each classification

### Task 2: Inventory All Fallback Patterns (Day 2)

**Objective**: Find and document ALL fallback patterns in the codebase

**Required Actions**:

1. Create `evidence/current/Evidence_Phase8_Fallback_Inventory.md`

2. Run comprehensive search:
```bash
# Find all fallback patterns
echo "=== return None patterns ==="
grep -rn "return None" core/ --include="*.py"

echo "=== return empty/default patterns ==="
grep -rn "return {}\|return \[\]\|return 0\." core/ --include="*.py"

echo "=== except with return patterns ==="
grep -rn "except.*:" core/ --include="*.py" -A 2 | grep "return"

echo "=== hardcoded thresholds ==="
grep -rn "= 0\.[0-9]" core/ --include="*.py" | grep -v "Field\|\*\|/"

echo "=== if not llm patterns ==="
grep -rn "if not.*llm\|if.*llm.*is None" core/ --include="*.py"
```

3. Document each pattern found:
   - File path and line number
   - Pattern type (return None, hardcoded value, etc.)
   - Context (what operation fails back)
   - Priority (critical path vs peripheral)

**Expected Findings**:
- 17+ files with `return None` patterns
- 25+ files with hardcoded decimal values
- Multiple `except: return` fallbacks
- Various `if not llm` bypass paths

**Validation**:
```python
# Count total fallbacks
patterns_found = count_from_grep_output()
print(f"Total fallback patterns: {patterns_found}")
print(f"Files affected: {unique_files_count}")
```

### Task 3: Design Central LLM Gateway (Day 3-4)

**Objective**: Create single point of LLM access for entire system

**Required Implementation**:

1. Create `core/llm_gateway.py`:
```python
"""
Central LLM Gateway - SINGLE point of LLM access.
NO FALLBACKS. System MUST fail if LLM unavailable.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.llm_required import require_llm, LLMRequiredError
from core.plugins.van_evera_llm_schemas import (
    HypothesisDomainClassification,
    ProbativeValueAssessment,
    ConfidenceFormulaWeights,
    # Import ALL schema types
)

@dataclass
class ThresholdResult:
    value: float
    reasoning: str
    context: str

class LLMGateway:
    """
    Central gateway for ALL LLM operations.
    Every semantic decision MUST go through here.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - one gateway for entire system"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.llm = require_llm()  # FAILS if no LLM
            self.cache = {}  # Simple cache
            self.initialized = True
    
    # Domain & Classification Operations
    def classify_domain(self, text: str, context: str = "") -> HypothesisDomainClassification:
        """Classify domain - NO FALLBACK"""
        try:
            return self.llm.classify_hypothesis_domain(text, context)
        except Exception as e:
            raise LLMRequiredError(f"LLM required for domain classification: {e}")
    
    # Confidence & Assessment Operations
    def assess_confidence(self, hypothesis: str, evidence: str) -> ProbativeValueAssessment:
        """Assess confidence - NO FALLBACK"""
        try:
            return self.llm.assess_probative_value(evidence, hypothesis, "")
        except Exception as e:
            raise LLMRequiredError(f"LLM required for confidence assessment: {e}")
    
    # Threshold Operations
    def determine_threshold(self, metric: str, context: str, domain: str = "") -> float:
        """Get dynamic threshold - NO FALLBACK"""
        prompt = f"Determine {metric} threshold for {context} in {domain} domain"
        try:
            # This would call LLM to determine appropriate threshold
            result = self.llm.determine_confidence_threshold(context, metric)
            return result.threshold
        except Exception as e:
            raise LLMRequiredError(f"LLM required for threshold determination: {e}")
    
    # Add 50+ more specific methods for EVERY semantic operation
    # NO method should have a fallback
    # ALL methods must raise LLMRequiredError on failure

# Global accessor
def get_llm_gateway() -> LLMGateway:
    """Get the singleton LLM gateway"""
    return LLMGateway()
```

2. Document design in `evidence/current/Evidence_Phase8_Gateway_Design.md`

3. List ALL semantic operations that need gateway methods:
   - Domain classification
   - Confidence assessment  
   - Threshold determination
   - Contradiction detection
   - Relationship evaluation
   - Evidence quality scoring
   - Hypothesis generation
   - Test creation
   - (continue for all operations...)

**Validation**:
```python
# Test gateway fails without LLM
os.environ['DISABLE_LLM'] = 'true'
try:
    gateway = LLMGateway()
    print("FAIL: Gateway should fail without LLM")
except LLMRequiredError:
    print("OK: Gateway correctly requires LLM")
```

### Task 4: Implement Gateway Core Methods (Day 5)

**Objective**: Implement the most critical gateway methods

**Required Implementation**:

1. Add to `core/llm_gateway.py`:
```python
# Priority 1: Methods used by semantic_analysis_service.py
def classify_hypothesis_domain(self, hypothesis: str, context: str = "") -> HypothesisDomainClassification:
    """Direct replacement for semantic service method"""
    try:
        return self.llm.classify_hypothesis_domain(hypothesis, context)
    except Exception as e:
        raise LLMRequiredError(f"LLM required: {e}")

def assess_probative_value(self, evidence: str, hypothesis: str, context: str = "") -> ProbativeValueAssessment:
    """Direct replacement for semantic service method"""
    try:
        return self.llm.assess_probative_value(evidence, hypothesis, context)
    except Exception as e:
        raise LLMRequiredError(f"LLM required: {e}")

# Priority 2: Methods for threshold replacement
def get_confidence_threshold(self, context: str, confidence_type: str = "general") -> float:
    """Replace hardcoded 0.5, 0.7, etc."""
    # Implementation that calls LLM for appropriate threshold
    
def get_weight_for_formula(self, formula_type: str, component: str) -> float:
    """Replace hardcoded weights in formulas"""
    # Implementation that calls LLM for weight

# Priority 3: Methods for enhancement functions
def enhance_evidence(self, evidence: str, context: str) -> EnhancedEvidence:
    """Replace enhance_evidence.py functionality"""
    # NO return None - must raise error

def enhance_mechanism(self, mechanism: str, events: List[str]) -> EnhancedMechanism:
    """Replace enhance_mechanisms.py functionality"""
    # NO return None - must raise error
```

2. Test each method:
```python
# test_gateway_methods.py
from core.llm_gateway import get_llm_gateway

gateway = get_llm_gateway()

# Test classification
try:
    result = gateway.classify_hypothesis_domain("test hypothesis")
    assert result is not None
    print(f"OK: Domain classification works: {result.primary_domain}")
except Exception as e:
    print(f"FAIL: {e}")

# Test assessment
try:
    result = gateway.assess_probative_value("evidence", "hypothesis")
    assert result is not None
    print(f"OK: Probative assessment works: {result.probative_value}")
except Exception as e:
    print(f"FAIL: {e}")

# Continue for all methods...
```

3. Document in `evidence/current/Evidence_Phase8_Gateway_Methods.md`:
   - List all methods implemented
   - Show test results for each
   - Note any that need LLM schema updates

**Validation**:
- All priority 1 methods work
- No fallbacks in any method
- Methods raise LLMRequiredError on failure

### Task 5: Week 1 Validation & Documentation

**Objective**: Validate Week 1 progress and document results

**Create `validate_phase8_week1.py`**:
```python
#!/usr/bin/env python3
"""
Validate Phase 8 Week 1 progress - File classification and gateway design.
"""

import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_file_classification():
    """Test that all files are classified"""
    print("\n[TEST] Checking file classification...")
    
    classification_file = Path("evidence/current/Evidence_Phase8_Classification.md")
    if not classification_file.exists():
        print("[FAIL] Classification file not found")
        return False
    
    # Count total Python files
    total_files = len(list(Path("core").rglob("*.py")))
    print(f"Total Python files: {total_files}")
    
    # Check classification completeness
    content = classification_file.read_text()
    if f"Total files classified: {total_files}" in content:
        print("[OK] All files classified")
        return True
    else:
        print("[FAIL] Not all files classified")
        return False

def test_fallback_inventory():
    """Test that fallback patterns are documented"""
    print("\n[TEST] Checking fallback inventory...")
    
    inventory_file = Path("evidence/current/Evidence_Phase8_Fallback_Inventory.md")
    if not inventory_file.exists():
        print("[FAIL] Fallback inventory not found")
        return False
    
    content = inventory_file.read_text()
    required_sections = [
        "return None patterns",
        "except with return patterns",
        "hardcoded thresholds",
        "Priority Migration List"
    ]
    
    for section in required_sections:
        if section in content:
            print(f"[OK] Found {section}")
        else:
            print(f"[FAIL] Missing {section}")
            return False
    
    return True

def test_gateway_design():
    """Test that LLM Gateway design is complete"""
    print("\n[TEST] Checking gateway design...")
    
    gateway_file = Path("core/llm_gateway.py")
    if not gateway_file.exists():
        print("[INFO] Gateway implementation not yet started (expected for Week 1)")
        
    design_file = Path("evidence/current/Evidence_Phase8_Gateway_Design.md")
    if not design_file.exists():
        print("[FAIL] Gateway design document not found")
        return False
    
    content = design_file.read_text()
    required_elements = [
        "Class: LLMGateway",
        "Method signatures",
        "Migration strategy",
        "Error handling"
    ]
    
    for element in required_elements:
        if element in content:
            print(f"[OK] Design includes {element}")
        else:
            print(f"[FAIL] Design missing {element}")
            return False
    
    return True

def test_current_coverage():
    """Test current LLM coverage metrics"""
    print("\n[TEST] Checking current LLM coverage...")
    
    from validate_true_llm_coverage import main as check_coverage
    
    # Capture coverage metrics
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        check_coverage()
    output = f.getvalue()
    
    # Parse metrics
    if "~30-40%" in output or "~30%" in output:
        print("[OK] Baseline coverage established: ~30%")
        return True
    else:
        print("[WARN] Coverage metrics unclear")
        return False

def main():
    """Run all Week 1 validation tests"""
    print("=" * 60)
    print("PHASE 8 WEEK 1 VALIDATION")
    print("=" * 60)
    
    tests = [
        ("File Classification", test_file_classification),
        ("Fallback Inventory", test_fallback_inventory),
        ("Gateway Design", test_gateway_design),
        ("Coverage Baseline", test_current_coverage)
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
    print("WEEK 1 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] Week 1 foundation complete!")
        print("Ready to proceed to Week 2: Gateway Implementation")
    else:
        print("[INCOMPLETE] Complete remaining Week 1 tasks before proceeding")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Success Criteria

**Week 1 Deliverables**:
- ✅ All 67 Python files classified into categories
- ✅ Complete fallback pattern inventory with priorities
- ✅ LLM Gateway design document with method signatures
- ✅ Migration strategy documented
- ✅ Baseline coverage metrics established (~30%)

**Quality Requirements**:
- Evidence files contain actual command outputs
- Classification rationale documented for each file
- Gateway design includes error handling strategy
- Migration priorities based on impact and complexity

### Testing Commands

```bash
# Run Week 1 validation
python validate_phase8_week1.py

# Check file classification
python -c "from pathlib import Path; print(f'Total files: {len(list(Path(\"core\").rglob(\"*.py\")))}')"

# Find fallback patterns
grep -rn "return None\|return {}\|return \[\]" core/ --include="*.py" | wc -l

# Check current coverage
python validate_true_llm_coverage.py

# Verify no runtime errors
python -m core.analyze test_data/american_revolution_graph.json
```

## Evidence Files Structure

Create these files in `evidence/current/`:
- `Evidence_Phase8_Classification.md` - File-by-file classification with rationale
- `Evidence_Phase8_Fallback_Inventory.md` - All fallback patterns with line numbers
- `Evidence_Phase8_Gateway_Design.md` - Detailed gateway architecture design
- `Evidence_Phase8_Week1_Summary.md` - Week 1 progress and metrics

Each evidence file must contain:
- Raw command outputs
- Structured data (classifications, patterns, metrics)
- Analysis and recommendations
- No false claims - validated data only

## Next Steps (Week 2)

After Week 1 validation passes, proceed to Week 2:
1. **Implement core LLMGateway class** with base methods
2. **Create migration helpers** for common patterns
3. **Migrate first 5 priority files** to use gateway
4. **Update tests** for migrated files
5. **Document migration patterns** for team reference

See `MASTER_PLAN_100_PERCENT_LLM_FIRST.md` for complete Week 2-5 roadmap.

## Critical Implementation Notes

**Evidence-Based Development**:
- Every change must be validated with actual execution
- No mocking or stubbing - real implementations only
- Document all decisions with rationale
- Test after each file migration

**Common Migration Patterns**:
1. Replace `try/except` with `require_llm()` and proper error propagation
2. Convert hardcoded values to LLM-generated decisions
3. Replace word overlap logic with semantic analysis
4. Update method signatures to accept LLM results
5. Add proper Pydantic schemas for structured outputs

**Quality Gates**:
- File must import and execute without errors
- All tests must pass after migration
- LLM calls must have proper error handling
- No silent fallbacks or default values
- Evidence of successful execution required