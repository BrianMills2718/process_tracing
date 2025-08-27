# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: ADVANCED LLM INTELLIGENCE INTEGRATION PHASE (Updated 2025-01-27)

**System Status**: **Production-Ready Foundation Complete** - Core quality improvements finished, system fully operational  
**Current Priority**: **CRITICAL - Transform Rule-Based Systems to Pure LLM Intelligence**  
**Academic Quality**: **Van Evera gap identified** - System has solid foundation but lacks academic reasoning engine

**COMPLETED FOUNDATION (2025-01-27)**:
- ✅ **Debug Statement Removal**: Zero debug statements, proper logging implemented
- ✅ **Unit Test Coverage**: 161 tests passing, 81-100% coverage on critical plugins
- ✅ **Structured Error Logging**: Comprehensive logging utilities with operational context
- ✅ **Plugin Architecture**: 16 plugins operational and properly registered
- ✅ **LLM Integration**: Gemini 2.5 Flash with structured Pydantic output
- ✅ **Type Safety**: mypy clean across core source files
- ✅ **Fresh Analysis Report**: 565KB HTML report demonstrates system functionality
- ✅ **Academic Gap Diagnosis**: Van Evera methodology gap comprehensively analyzed

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Academic Functionality**: Target ≥90% Van Evera compliance (current foundation: 67.9-71.5%)
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results

## Project Overview

**LLM-Enhanced Process Tracing Toolkit** - Production-ready system implementing Van Evera academic methodology for qualitative analysis using process tracing. **Current limitation**: Uses 200+ hardcoded keyword patterns instead of leveraging full LLM reasoning capabilities.

### Architecture
- **Plugin System**: 16 registered plugins with proper abstractions
- **Van Evera Workflow**: 8-step academic analysis pipeline (needs LLM enhancement)
- **LLM Integration**: Gemini 2.5 Flash with structured Pydantic output
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management

## 🚀 CURRENT PHASE: CRITICAL LLM INTELLIGENCE INTEGRATION

### **DIAGNOSED PROBLEM**: Academic Rigor Gap
**Root Cause**: System uses inferior rule-based keyword matching instead of sophisticated LLM semantic understanding for Van Evera diagnostic testing.

**Evidence**: Fresh HTML analysis shows:
- 379 causal chains identified correctly
- 160 nodes (events, evidence, hypotheses) properly categorized  
- **BUT**: No Van Evera diagnostic test results, no hypothesis testing, no academic conclusions
- **Missing**: Hoop tests, smoking gun tests, evidence probative values, confidence scores

**Opportunity**: Transform to pure LLM intelligence for 40-60% academic quality improvement.

## 📋 CRITICAL TASKS - LLM INTELLIGENCE TRANSFORMATION

### **PHASE 1: CORE VAN EVERA ENGINE RESTORATION**

#### **TASK 1.1: Enable Disabled LLM Enhancement in Diagnostic Classifier** ⭐ CRITICAL
**File**: `core/plugins/content_based_diagnostic_classifier.py:354-358`  
**Issue**: LLM enhancement commented out: `# Temporarily disable LLM enhancement due to deployment issues`  
**Impact**: Core Van Evera diagnostic classification using crude keyword matching instead of semantic analysis

**Implementation Steps**:
1. **Investigate root cause**: Why was LLM enhancement originally disabled?
2. **Read error logs**: Check for deployment issues mentioned in comments
3. **Re-enable LLM enhancement**: Remove comment blocks, restore LLM integration
4. **Add proper error handling**: Ensure robust LLM failure handling
5. **A/B test accuracy**: Compare LLM vs keyword classification results
6. **Validate diagnostic quality**: Ensure improved Van Evera test classification

**Success Criteria**:
- LLM enhancement functional without deployment issues
- Diagnostic classification accuracy demonstrably improved
- No regression in system stability

#### **TASK 1.2: Replace Semantic Bridge Keywords with LLM Intelligence**
**File**: `core/plugins/evidence_connector_enhancer.py:23-53`  
**Issue**: 52+ hardcoded SEMANTIC_BRIDGES dictionary instead of contextual semantic analysis  
**Impact**: Evidence-hypothesis connections based on keyword matching vs academic reasoning

**Current Problem**:
```python
SEMANTIC_BRIDGES = {
    "therefore": 0.8, "consequently": 0.7, "as_a_result": 0.75,
    # ... 49+ more hardcoded patterns
}
```

**Target Implementation**:
```python
def analyze_evidence_hypothesis_connection_llm(evidence_text, hypothesis_text, llm_interface):
    """Use LLM to evaluate semantic relationship strength and type"""
    return llm_interface.evaluate_semantic_connection(
        evidence_text=evidence_text,
        hypothesis_text=hypothesis_text,
        analysis_depth="academic_semantic_relationship_assessment"
    )
```

**Implementation Steps**:
1. **Analyze current semantic bridge logic**: Understand how 52 patterns are used
2. **Design LLM semantic analysis**: Create contextual relationship evaluation
3. **Implement LLM replacement**: Replace dictionary lookups with LLM calls
4. **Performance optimization**: Cache LLM results for repeated evaluations
5. **A/B test semantic accuracy**: Compare keyword vs LLM relationship detection
6. **Validate academic improvement**: Ensure better evidence-hypothesis connections

#### **TASK 1.3: Upgrade Van Evera Test Confidence to LLM Academic Reasoning**
**File**: `core/plugins/van_evera_testing.py:423-465`  
**Issue**: Fixed confidence thresholds (0.3, 0.8, 0.95) instead of contextual academic evaluation  
**Impact**: Academic assessment using algorithmic scoring vs evidence-based reasoning

**Current Problem**: Hardcoded confidence calculation
**Target**: LLM contextual evaluation of diagnostic test strength

**Implementation Steps**:
1. **Understand current confidence logic**: How thresholds are applied
2. **Design LLM academic reasoning**: Context-aware confidence evaluation
3. **Implement Bayesian-style updating**: Prior beliefs + evidence → posterior confidence
4. **Add uncertainty quantification**: Confidence intervals for all assessments
5. **Validate academic rigor**: Compare fixed vs contextual confidence accuracy
6. **Test with historical cases**: Ensure improved Van Evera compliance

### **PHASE 2: COMPREHENSIVE INTELLIGENCE TRANSFORMATION**

#### **Academic Quality Assessment Intelligence**
**File**: `core/plugins/primary_hypothesis_identifier.py:270-300`  
**Replace**: Keyword counting for theoretical sophistication with LLM academic assessment

#### **Dynamic Prediction Generation**  
**File**: `core/plugins/van_evera_testing.py:423-465`  
**Replace**: Hardcoded prediction templates with LLM context-aware generation

#### **Research Question Intelligence**
**File**: `core/plugins/research_question_generator.py:135-317`  
**Replace**: Pattern-based classification with LLM academic question generation

## 🔧 IMPLEMENTATION GUIDANCE

### **Evidence-First Methodology**
**Create Evidence Structure**:
```
evidence/
├── current/
│   ├── Evidence_LLM_ENHANCEMENT_Diagnostic_Classifier.md
│   ├── Evidence_LLM_ENHANCEMENT_Semantic_Bridges.md
│   ├── Evidence_LLM_ENHANCEMENT_Van_Evera_Confidence.md
└── completed/
    └── Evidence_CODE_QUALITY_Improvements.md  # Archive previous phase
```

**Required Evidence Per Task**:
- **Raw LLM API logs**: All request/response pairs
- **A/B comparison data**: Rule-based vs LLM accuracy metrics
- **Quality metrics**: Before/after Van Evera compliance scores
- **Performance benchmarks**: Speed, token usage, success rates
- **Error analysis**: Any failures and their resolutions

### **Quality Gates - Van Evera Academic Standards**
**Phase 1 Success Criteria**:
- **LLM Enhancement Active**: No commented-out LLM code, all integrations functional
- **Semantic Intelligence**: Zero hardcoded keyword patterns, all contextual evaluation
- **Academic Confidence**: Evidence-based assessment, no fixed thresholds
- **Measurable Improvement**: Van Evera compliance increase documented with evidence

**Overall Success Target**: **Van Evera compliance ≥90%** (vs current ~72%)

### **Testing and Validation Protocol**

**Before Each Implementation**:
```bash
# Capture current baseline
python -c "
from core.analyze import main
result = main('revolutions_20250805_122000_graph.json')
print(f'Van Evera Compliance: {result.get(\"van_evera_compliance\")}')
print(f'Academic Quality Score: {result.get(\"academic_quality_score\")}')
"
```

**After Each Implementation**:
```bash
# Run comprehensive validation
python -m pytest tests/plugins/ -v  # Ensure no regression
python -m mypy core/  # Type safety maintained
# Re-run analysis and compare metrics
# Document improvement in evidence files
```

## Minor Technical Cleanup

### **Cleanup Task: Resolve Pytest Warnings**
**Files**: `core/plugins/van_evera_testing.py` lines 15, 28, 40  
**Issue**: Classes with `__init__` constructors triggering pytest collection warnings  
**Fix**: Rename classes to avoid pytest test discovery or add `__test__ = False`

## Codebase Structure

### Key Entry Points
- `core/analyze.py`: Main analysis orchestration with Van Evera pipeline
- `core/plugins/van_evera_workflow.py`: 8-step academic analysis workflow
- `core/plugins/van_evera_llm_interface.py`: LLM integration with structured output

### Plugin Organization
- `core/plugins/base.py`: Base classes with structured error logging
- `core/plugins/register_plugins.py`: Plugin registration system
- `core/logging_utils.py`: Structured logging utilities for operational context

### Critical Integration Points
- **LLM Interface**: `van_evera_llm_interface.py` - Gemini 2.5 Flash with Pydantic schemas
- **Plugin Architecture**: 16 plugins with proper abstractions and error handling
- **Test Coverage**: `tests/plugins/` - 161 tests with 81-100% coverage on critical components

**Current State**: Production-ready foundation with identified academic methodology gap. LLM intelligence integration will transform rule-based keyword matching into contextual academic reasoning, achieving publication-quality Van Evera process tracing analysis.