# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: PHASE 1 VALIDATION REQUIRED (Updated 2025-01-27)

**System Status**: **Phase 1 Implementation Complete** - Three academic enhancements implemented with fail-fast error handling  
**Current Priority**: **Critical Validation Testing** - Verify actual functionality before declaring success  
**Academic Quality**: **Implementation Unvalidated** - Claims require empirical testing with raw evidence

**COMPLETED PHASE 1 IMPLEMENTATION (2025-01-27)**:
- ✅ **Van Evera Test Integration**: Integrated test results into hypothesis confidence scoring (`core/analyze.py`)
- ✅ **Hypothesis LLM Enhancement**: Created `core/enhance_hypotheses.py` with VanEveraLLMInterface  
- ✅ **Evidence Balance Correction**: Implemented systematic evidence classification targeting 0.6-0.8 ratios
- ✅ **Fail-Fast Error Handling**: Removed all fallback mechanisms from LLM interface
- ❌ **VALIDATION STATUS**: **ZERO EMPIRICAL TESTING** - No validation of actual functionality

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Academic Functionality**: Target ≥80% Van Evera compliance (current: implementation unvalidated)
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results in structured evidence files

## Project Overview

**LLM-Enhanced Process Tracing Toolkit** - Production-ready system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing.

### Architecture
- **Plugin System**: 16 registered plugins with sophisticated LLM integration
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence
- **LLM Integration**: VanEveraLLMInterface with structured Pydantic output (Gemini 2.5 Flash)
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management

## 🧪 PHASE 1 VALIDATION PROTOCOL (Critical Priority)

### **MANDATORY VALIDATION**: Verify Phase 1 Implementation Before Expansion

**Critical Gap**: Phase 1 tasks were implemented but **never validated**. System behavior under new enhancements is unknown.

**Validation Required**:
1. **Integration Completeness**: Do all three enhancements execute during full analysis?
2. **Evidence Balance Results**: Do we get support ratios other than 1.0 or 0S/0R?  
3. **Van Evera Confidence Mapping**: Do hypothesis confidence scores vary with test outcomes?
4. **LLM Enhancement Execution**: Does `enhance_hypotheses.py` actually run and generate results?
5. **Fail-Fast Error Handling**: Does system fail immediately on validation errors without fallbacks?

### **TASK V1: Integration Validation Test**
**Priority**: **CRITICAL** - Must complete before any other development
**Objective**: Verify all three Phase 1 enhancements execute in full analysis pipeline

**Test Protocol**:
```bash
# Full analysis with comprehensive logging
python -m core.analyze output_data/revolutions/revolutions_20250805_122000_graph.json --html 2>&1 | tee validation_test_$(date +%Y%m%d_%H%M%S).log
```

**Evidence Required**:
- Console logs showing `[HYPOTHESIS_ENHANCEMENT]`, `[VAN_EVERA_INTEGRATION]`, `[EVIDENCE_BALANCE]` execution
- Final HTML output demonstrating all three enhancements visible
- Hypothesis confidence scores populated (not empty/null)
- Evidence balance ratios measured and documented
- Performance impact measurements (execution time, LLM calls, token usage)

**Success Criteria**:
- All three enhancements execute without errors
- Results visible in final output
- No regression in existing mechanism/alternative analysis quality
- Fail-fast behavior confirmed (no silent fallbacks)

**Failure Response**: If validation fails, debug individual components before integration expansion

### **TASK V2: Evidence Balance Validation**
**Priority**: **HIGH** - Core academic quality metric
**Objective**: Confirm evidence balance correction produces academic-standard ratios

**Current Baseline**: 1.00 support ratio (100% supporting evidence, 0% refuting)
**Academic Target**: 0.6-0.8 support ratio (balanced evidence evaluation)

**Test Protocol**:
```python
def validate_evidence_balance(analysis_results):
    """Measure evidence balance improvements"""
    for hyp_id, hyp_data in analysis_results.get('evidence_analysis', {}).items():
        supporting_count = len(hyp_data.get('supporting_evidence', []))
        refuting_count = len(hyp_data.get('refuting_evidence', []))
        
        if supporting_count + refuting_count > 0:
            support_ratio = supporting_count / (supporting_count + refuting_count)
            print(f"Hypothesis {hyp_id}: {support_ratio:.2f} support ratio")
            print(f"  Supporting: {supporting_count}, Refuting: {refuting_count}")
    return evidence_balance_metrics
```

**Evidence Required**:
- Raw evidence counts (supporting vs refuting) for each hypothesis
- Support ratio calculations with comparison to 1.0 baseline  
- Van Evera FAIL results mapped to refuting evidence
- Systematic disconfirming evidence identification working

### **TASK V3: Cross-Dataset Validation**
**Priority**: **MEDIUM** - Generalizability assessment
**Objective**: Confirm improvements work beyond American Revolution dataset

**Simple Test Case Creation**:
```
input_text/validation_case/simple_conflict.txt:
"Economic sanctions were imposed on Country X in January 2020. 
Political protests erupted in March 2020, with widespread demonstrations.
Government officials claimed the protests were due to internal political disputes.
However, protest timing closely followed sanctions implementation.
Some evidence suggests economic hardship from sanctions motivated protesters.
Critics argue domestic political factors were more significant than economic pressure."
```

**Test Protocol**:
```bash
# Generate graph from simple case
python -m core.extract input_text/validation_case/simple_conflict.txt

# Run enhanced analysis  
python -m core.analyze [generated_graph.json] --html
```

**Evidence Required**:
- System processes different case successfully  
- Hypotheses about sanctions→protests causation identified
- Evidence balance shows mixed supporting/refuting evidence
- Van Evera tests execute on different domain
- LLM enhancements work across different subject matter

### **TASK V4: Error Handling Validation**
**Priority**: **MEDIUM** - Fail-fast compliance verification
**Objective**: Confirm removal of fallback mechanisms and proper error propagation

**Test Cases**:
1. **LLM Validation Error**: Force invalid schema response from Gemini
2. **Network Error**: Simulate API timeout/connection failure  
3. **Missing Van Evera Results**: Test with incomplete plugin data
4. **Pydantic Schema Error**: Test with malformed structured output

**Test Protocol**:
```python
# Simulate validation error
def test_fail_fast_validation():
    invalid_llm_response = "This is not valid JSON schema"
    # Should raise ValidationError immediately, no fallback behavior
    with pytest.raises(ValidationError):
        enhance_hypothesis_with_llm(test_hypothesis, [], [])
```

**Evidence Required**:
- System fails immediately on schema validation errors
- No silent fallback behavior observed
- Proper error messages surface to user
- Transient network errors retry appropriately
- Non-recoverable errors fail fast without masking

## Implementation Guidelines

### **Evidence Requirements** 
Create structured evidence files in `evidence/current/` for each validation task:
```
evidence/current/
├── Evidence_INTEGRATION_Validation.md
├── Evidence_EVIDENCE_Balance_Validation.md  
├── Evidence_CROSS_Dataset_Validation.md
└── Evidence_ERROR_Handling_Validation.md
```

**Required Evidence Per Task**:
- Raw console logs with timestamps
- Before/after comparison metrics
- Quantified improvements with specific measurements
- Error case documentation with actual error messages
- Performance impact assessment

### **Validation Protocol**
**For Each Validation Task**:
```bash
# Document before state
python -m core.analyze [test_case] --html > before_validation.log 2>&1

# Run validation test
[execute validation]

# Document after state  
python -m core.analyze [test_case] --html > after_validation.log 2>&1

# Compare results
diff before_validation.log after_validation.log
```

### **Success Criteria**
- **Integration**: All three enhancements execute and produce visible results
- **Evidence Balance**: Support ratios between 0.6-0.8 for at least 50% of hypotheses
- **Cross-Dataset**: System works on different domains without modification
- **Error Handling**: Immediate failure on validation errors, no silent fallbacks
- **Performance**: No >3x execution time increase from baseline

### **Quality Gates**
- **Empirical**: All validation tests pass with documented evidence
- **Academic**: Evidence balance improvements demonstrated with quantified metrics
- **Technical**: No regression in existing LLM mechanism quality
- **Reliability**: Fail-fast error handling confirmed across error types

## Codebase Structure

### Key Entry Points
- `core/analyze.py`: Main analysis orchestration with Phase 1 enhancements (lines 2940-3271)
- `core/enhance_hypotheses.py`: LLM hypothesis enhancement pipeline (new)
- `core/plugins/van_evera_llm_interface.py`: Fail-fast LLM interface (updated)

### Critical Integration Points
- **Hypothesis Enhancement Loop**: Lines 2940-2988 in `core/analyze.py`
- **Van Evera Result Integration**: Lines 3156-3197 in `core/analyze.py`
- **Evidence Balance Correction**: Lines 3252-3271 in `core/analyze.py`
- **Fail-Fast Error Handling**: Removed fallback methods in `van_evera_llm_interface.py`

### Module Organization
- `core/plugins/`: 16 registered plugins with structured LLM integration
- `core/logging_utils.py`: Structured logging utilities for operational context  
- `universal_llm_kit/`: LiteLLM integration with Gemini 2.5 Flash
- `evidence/`: Structured evidence documentation (current/ and completed/ phases)

### Evidence Structure
```
evidence/
├── current/          # Active validation work only
├── completed/        # Archived completed phases  
```

**CRITICAL**: Evidence files must contain ONLY current validation work. Archive completed phases to avoid chronological confusion.

## Next Steps After Validation

**Only After Successful V1-V4 Validation**:
- **Phase 2**: Academic rigor expansion (alternative explanations, causal chains)
- **Multi-Document Architecture**: Corpus handling for large-scale analysis  
- **Performance Optimization**: LLM cost reduction and caching strategies
- **Academic Benchmarking**: Human expert comparison studies

**System Architecture**: Phase 1 implementation complete, awaiting empirical validation to confirm functionality before expansion.