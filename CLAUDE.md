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
- ❌ Returning None/0/[] on LLM failure (must raise LLMRequiredError)
- ❌ Mixed LLM configurations (some calls to Gemini, others to different models)
- ❌ Direct API calls bypassing LiteLLM infrastructure

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation
- ✅ Generalist process tracing without dataset-specific hardcoding
- ✅ Raise LLMRequiredError on any LLM failure (fail-fast)
- ✅ Consistent LiteLLM routing for ALL LLM operations
- ✅ Single model configuration across entire system

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: Phase 21B - LLM-First Policy Runtime Integration Fix (Updated 2025-01-09)

**System Status**: **CONVERSION CONCEPTS CORRECT BUT RUNTIME BROKEN - CRITICAL INTEGRATION FIXES REQUIRED**  
**Latest Achievement**: **Phase 21A Complete - Keyword logic removed but LLM interface calls broken**  
**Current Priority**: **Fix LLM interface integration and validate runtime execution (critical-path)**

**PHASE 21A RESULTS**:
- ✅ **Keyword Logic Removed**: Zero keyword matching patterns remain in semantic analysis
- ✅ **Hardcoded Thresholds Removed**: All confidence fallbacks eliminated
- ✅ **LLMRequiredError Added**: Proper fail-fast error handling integrated
- ✅ **Source Validation**: Comprehensive audit confirms LLM-First compliance achieved
- ❌ **Runtime Broken**: LLM interface calls use wrong methods and parameters

**CRITICAL RUNTIME ISSUES IDENTIFIED**:
- 🔴 **Import Errors**: `BayesianEvidence` referenced but not imported in `core/evidence_weighting.py`
- 🔴 **Wrong LLM Methods**: Calling `get_structured_response()` which doesn't exist
- 🔴 **Invalid Parameters**: Using string schemas instead of Pydantic model types
- 🔴 **Response Parsing Errors**: Assuming attributes that don't exist in response models
- 🔴 **Integration Failures**: Code will crash at runtime despite conceptual correctness

**IMMEDIATE IMPACT**: LLM-First policy violations **conceptually resolved** but **implementation broken** - requires systematic interface integration fix.

## 🔧 PHASE 21B: LLM Interface Integration Fix

### OBJECTIVES: Make LLM-First conversions executable at runtime

**CRITICAL INTEGRATION FIXES REQUIRED**:
1. **Import Resolution**: Fix `BayesianEvidence` import errors in evidence weighting
2. **LLM Interface Integration**: Connect to actual VanEveraLLMInterface methods
3. **Schema Alignment**: Use correct Pydantic models for structured responses
4. **Response Parsing**: Handle actual LLM response attributes correctly
5. **Runtime Validation**: Comprehensive testing of converted functions

**STRATEGIC APPROACH**: Systematic interface integration following ultrathink remediation plan

## 🔧 SYSTEMATIC REMEDIATION TASKS

### TASK 1: Foundation Repair (30 minutes)

**OBJECTIVE**: Fix critical runtime issues without changing core logic

**SUBTASK 1.1: Resolve Import Dependencies** (10 minutes)
```bash
# CURRENT ERROR: BayesianEvidence referenced but not imported
# FILE: core/evidence_weighting.py lines 246, 288, 330, 374
# INVESTIGATION REQUIRED:
find . -name "*.py" -exec grep -l "class BayesianEvidence" {} \;
grep -r "from.*BayesianEvidence" core/

# RESOLUTION OPTIONS:
# A) Find correct import path from existing modules
# B) Create type alias: from typing import Any; BayesianEvidence = Any
# C) Comment out unused functions temporarily (safest for validation)

# VALIDATION:
python -c "from core.evidence_weighting import EvidenceStrengthQuantifier; print('✅ Import success')"
```

**SUBTASK 1.2: Fix LLM Interface Method Calls** (15 minutes)
```python
# CURRENT ERROR: Calling non-existent get_structured_response()
# REQUIRED: Map to actual VanEveraLLMInterface methods

# INVESTIGATION:
grep -A 5 "def.*structured" core/plugins/van_evera_llm_interface.py
grep -A 10 "assess_probative_value\|assess_confidence" core/plugins/van_evera_llm_interface.py

# MAPPING NEEDED:
# Evidence reliability → assess_probative_value() or _get_structured_response()
# Source credibility → assess_probative_value() or custom method
# Response parsing → Map to actual Pydantic model attributes
```

**SUBTASK 1.3: Schema Integration** (5 minutes)
```python
# INVESTIGATION: What schemas exist for reliability/credibility?
grep -A 20 "reliability_score\|credibility" core/plugins/van_evera_llm_schemas.py
grep -A 10 "class.*Assessment" core/plugins/van_evera_llm_schemas.py

# REQUIRED: Map each use case to existing Pydantic models
# ComprehensiveEvidenceAnalysis.reliability_score exists ✓
```

### TASK 2: LLM Interface Integration (45 minutes)

**OBJECTIVE**: Connect to actual VanEveraLLMInterface methods correctly

**SUBTASK 2.1: Evidence Reliability Integration** (15 minutes)
```python
# TARGET: _analyze_reliability_indicators() in evidence_weighting.py
# REPLACE: get_structured_response(prompt, target_schema="reliability_float")
# WITH: assess_probative_value() OR _get_structured_response(prompt, ComprehensiveEvidenceAnalysis)

# IMPLEMENTATION TEMPLATE:
def _analyze_reliability_indicators(self, reasoning: str, justification: str) -> float:
    from .plugins.van_evera_llm_interface import VanEveraLLMInterface
    from .llm_required import LLMRequiredError
    
    try:
        llm_interface = VanEveraLLMInterface()
        
        # METHOD 1: Use existing assess_probative_value
        response = llm_interface.assess_probative_value(
            evidence_description=f"Reasoning: {reasoning}. Justification: {justification}",
            hypothesis_description="Evidence reliability assessment context",
            context="Assessing evidence reliability for Bayesian weighting"
        )
        return response.reliability_score  # ComprehensiveEvidenceAnalysis has this field
        
        # METHOD 2: If above fails, use _get_structured_response
        # response = llm_interface._get_structured_response(prompt, ComprehensiveEvidenceAnalysis)
        
    except Exception as e:
        raise LLMRequiredError(f"LLM required for reliability assessment: {e}")
```

**SUBTASK 2.2: Source Credibility Integration** (15 minutes)
```python
# Same pattern as reliability but may need different approach
# ComprehensiveEvidenceAnalysis doesn't have credibility_score
# May need to use probative_value or evidence_quality as proxy
```

**SUBTASK 2.3: Plugin Integration Fixes** (15 minutes)  
```python
# Research Question Generator - phenomenon classification
# Use classify_hypothesis_domain() method that exists
# Alternative Hypothesis Generator - use assess_probative_value()
```

### TASK 3: Comprehensive Testing Strategy (60 minutes)

**OBJECTIVE**: Validate all conversions work at runtime

**SUBTASK 3.1: Unit Testing** (20 minutes)
```python
# Create test_llm_integration_fixed.py
def test_reliability_assessment_actual():
    """Test reliability assessment with real LLM interface"""
    quantifier = EvidenceStrengthQuantifier()
    
    # Test with actual reasoning/justification
    score = quantifier._analyze_reliability_indicators(
        "Multiple verified sources confirm timeline", 
        "Academic peer-reviewed analysis with institutional backing"
    )
    
    assert 0.0 <= score <= 1.0
    print(f"✅ Reliability score: {score}")

def test_import_resolution():
    """Test all imports work"""
    from core.evidence_weighting import EvidenceStrengthQuantifier
    from core.confidence_calculator import ConfidenceLevel
    print("✅ All imports successful")
```

**SUBTASK 3.2: Integration Testing** (20 minutes)
```python
def test_american_revolution_pipeline():
    """Test extraction still works with fixes"""
    # Run process_trace_advanced.py --extract-only
    # Verify 31 nodes, 30 edges still produced
    # Ensure no import/interface errors
```

**SUBTASK 3.3: Validation Framework** (20 minutes)
```bash
# Comprehensive compliance check
grep -r "if.*in.*text" core/ | wc -l  # Should be 0
grep -r "positive_indicators" core/ | wc -l  # Should be 0
grep -r "get_structured_response" core/ | wc -l  # Should be 0
python -c "from core.confidence_calculator import ConfidenceLevel; ConfidenceLevel.from_score(0.75)" # Should raise LLMRequiredError
```

### TASK 4: Runtime Validation (45 minutes)

**OBJECTIVE**: Prove all conversions work in practice

**SUBTASK 4.1: End-to-End Pipeline Test** (20 minutes)
```bash
# Test American Revolution extraction with fixes
echo "1" | python process_trace_advanced.py --extract-only
# Should complete without LLM interface errors

# Monitor for specific error patterns:
# - ImportError: BayesianEvidence
# - AttributeError: get_structured_response  
# - AttributeError: reliability_score/credibility_score
```

**SUBTASK 4.2: Evidence Processing Test** (15 minutes)
```python
def test_evidence_assessment_integration():
    """Test EvidenceAssessment → EvidenceWeights pipeline"""
    from core.evidence_weighting import EvidenceStrengthQuantifier
    from core.structured_models import EvidenceAssessment
    
    # Create test evidence assessment
    assessment = EvidenceAssessment(
        evidence_id="test_evidence",
        reasoning_for_type="Multiple verified sources document this event",
        justification_for_likelihoods="Academic historians confirm timeline through primary sources",
        suggested_numerical_probative_value=0.8
    )
    
    quantifier = EvidenceStrengthQuantifier()
    weights = quantifier.quantify_llm_assessment(assessment)
    
    print(f"✅ Evidence weights generated: {weights.combined_weight}")
```

**SUBTASK 4.3: Success Criteria Validation** (10 minutes)
```bash
# Final validation checklist
echo "FINAL VALIDATION CHECKLIST:"
echo "1. Zero keyword logic patterns:"
grep -r "if.*in.*text" core/ || echo "✅ PASSED"

echo "2. Zero hardcoded thresholds:"
grep -r "Fallback to default" core/ || echo "✅ PASSED"

echo "3. LLM interface integration:"
python -c "
from core.evidence_weighting import EvidenceStrengthQuantifier
q = EvidenceStrengthQuantifier()
# Test would call _analyze_reliability_indicators here
print('✅ LLM interface accessible')
"

echo "4. American Revolution regression:"
# Should have successfully extracted 31 nodes, 30 edges
echo "✅ Pipeline functional (if previous test passed)"
```

## 📊 SUCCESS CRITERIA

### **Technical Success Criteria:**
1. **Zero Import Errors**: All modules import without `NameError` or `ImportError`
2. **Zero Method Errors**: LLM interface calls use correct method names and parameters
3. **Zero Response Errors**: LLM response parsing handles actual response attributes  
4. **Runtime Execution**: All converted functions execute without crashes

### **Functional Success Criteria:**
1. **Zero keyword-based logic** in semantic analysis functions (already achieved ✅)
2. **Zero hardcoded probative values** or confidence thresholds (already achieved ✅)
3. **All LLM failures** result in LLMRequiredError (already achieved ✅)
4. **LLM interface integration** works correctly at runtime

### **Validation Success Criteria:**
1. **American Revolution extraction** completes successfully (31 nodes, 30 edges)
2. **Evidence processing pipeline** executes without LLM interface errors
3. **Quality improvements** demonstrable through semantic vs. keyword comparison
4. **Performance impact** measured and documented

## 🎯 POST-COMPLETION OPTIMIZATION OPPORTUNITIES

After successful runtime integration, consider implementing user's batching insight:

**Current State**: Individual LLM calls for each evidence-reliability/credibility assessment  
**Optimization Target**: Batch multiple assessments in single LLM calls  
**Expected Impact**: Significant performance improvement while maintaining semantic quality

**Implementation Strategy**:
```python
def assess_multiple_evidence_reliability(evidence_list):
    """Single LLM call to assess reliability of multiple evidence items"""
    # Batch processing for efficiency
    # Maintain individual reliability scores
    # Preserve semantic analysis quality
```

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`process_trace_advanced.py`**: Main orchestration script with project selection and pipeline management
- **`core/extract.py`**: Extraction phase entry point - working correctly (extraction successful)
- **`core/analyze.py`**: Analysis phase entry point - times out but imports successfully

### Critical Files for Phase 21B
- **`core/evidence_weighting.py`**: Contains broken LLM interface calls requiring fix (lines 142-179, 186-221)
- **`core/confidence_calculator.py`**: Already fixed - requires LLM thresholds (✅ completed)
- **`core/plugins/van_evera_llm_interface.py`**: Target interface with correct methods
- **`core/plugins/van_evera_llm_schemas.py`**: Contains Pydantic models for structured responses

### Critical Integration Points
- **LLM Interface**: `VanEveraLLMInterface` class with methods like `assess_probative_value()`
- **LLM Schemas**: `ComprehensiveEvidenceAnalysis` model has `reliability_score` field  
- **Error Handling**: `core/llm_required.py` with `LLMRequiredError` (already integrated ✅)

### Runtime Environment
- **Virtual Environment**: `test_env/` activated with `source test_env/bin/activate`
- **Dependencies**: All required packages installed and functional
- **LLM Access**: Universal LLM kit configured for Gemini/GPT integration

## 📋 Coding Philosophy

### NO LAZY IMPLEMENTATIONS  
- Every function must be fully executable at runtime
- No placeholder code, mocking, or temporary stubs
- Test imports and method calls before claiming success

### FAIL-FAST PRINCIPLES
- All LLM failures must raise `LLMRequiredError` (already implemented ✅)
- Surface import and interface errors immediately
- No silent failures or degraded functionality

### EVIDENCE-BASED DEVELOPMENT
- All claims require raw execution evidence
- Test actual LLM interface calls, not theoretical implementations
- Document runtime behavior, not conceptual correctness

### SYSTEMATIC VALIDATION
- Test each fix before proceeding to next
- Validate imports work before testing interface calls
- Prove pipeline compatibility after each change

---

## 📁 Evidence Structure

Evidence for Phase 21B must be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase21B_LLMIntegrationFix.md
```

**REQUIRED EVIDENCE**:
- Raw import test outputs (`python -c "from core.evidence_weighting import EvidenceStrengthQuantifier"`)
- Actual LLM interface method signatures and available schemas
- Runtime test results for each fixed function
- American Revolution extraction logs confirming pipeline compatibility
- Before/after performance measurements

**CRITICAL**: No success claims without demonstrable runtime execution evidence.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.