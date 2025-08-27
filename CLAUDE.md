# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: PRODUCTION-READY SYSTEM (Updated 2025-01-27)

**System Status**: **Production-Ready** - All core quality improvements completed, system fully operational  
**Current Priority**: **Advanced LLM Intelligence Integration** - Transform rule-based systems to pure LLM reasoning  
**Academic Quality**: **Van Evera compliant** - Complete diagnostic test implementation with structured LLM integration

**COMPLETED FOUNDATION (2025-01-27)**:
- ✅ **Debug Statement Removal**: Zero debug statements in production code, proper logging implemented
- ✅ **Unit Test Coverage**: 161 tests passing, 81-100% coverage on 6 critical plugins
- ✅ **Structured Error Logging**: Comprehensive logging utilities with operational context
- ✅ **Plugin Architecture**: 16 plugins operational and properly registered
- ✅ **LLM Integration**: Gemini 2.5 Flash with structured Pydantic output
- ✅ **Type Safety**: mypy clean - no type errors in core source files
- ✅ **Security**: Proper API key handling via environment variables

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Academic Functionality**: Maintain ≥60% Van Evera compliance (currently 67.9-71.5%)
- **Technical Correctness**: All Pydantic validations must pass
- **Process Adherence**: Run lint/typecheck before marking tasks complete
- **Evidence Documentation**: Document all claims with concrete test results

## Project Overview

**LLM-Enhanced Process Tracing Toolkit** - Production-ready system implementing Van Evera academic methodology for qualitative analysis using process tracing with comprehensive diagnostic tests and structured LLM integration.

### Architecture
- **Plugin System**: 16 registered plugins with proper abstractions
- **Van Evera Workflow**: 8-step academic analysis pipeline
- **LLM Integration**: Gemini 2.5 Flash with structured Pydantic output
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management

## 🚀 NEXT PHASE: ADVANCED LLM INTELLIGENCE INTEGRATION

### **Priority: Transform Rule-Based Systems to Pure LLM Intelligence**

**Current Limitation**: System uses 200+ hardcoded keyword patterns for semantic analysis instead of leveraging full LLM reasoning capabilities.

**Opportunity**: Replace inferior rule-based methods with sophisticated LLM semantic understanding for 40-60% quality improvement.

### **PHASE 1: CRITICAL LLM ENHANCEMENT (High Priority)**

#### **Task 1.1: Enable Disabled LLM Enhancement in Diagnostic Classifier**
**File**: `core/plugins/content_based_diagnostic_classifier.py:354-358`
**Issue**: LLM enhancement commented out: `# Temporarily disable LLM enhancement due to deployment issues`
**Impact**: Core Van Evera diagnostic classification using crude keyword matching instead of LLM semantic analysis
**Priority**: **CRITICAL** - Blocking advanced academic analysis quality

**Implementation**:
1. Investigate root cause of original LLM disable
2. Re-enable LLM enhancement with proper error handling
3. A/B test LLM vs keyword classification accuracy
4. Validate improved diagnostic test quality

#### **Task 1.2: Replace Semantic Bridge Keywords with LLM Intelligence**
**File**: `core/plugins/evidence_connector_enhancer.py:23-53`
**Issue**: 52+ hardcoded semantic bridges instead of true semantic understanding
**Impact**: Evidence-hypothesis connections based on keyword matching vs contextual analysis

**Implementation**:
```python
# Replace SEMANTIC_BRIDGES dictionary with:
def analyze_evidence_hypothesis_connection_llm(evidence_text, hypothesis_text, llm_interface):
    """Use LLM to evaluate semantic relationship strength and type"""
    return llm_interface.evaluate_semantic_connection(
        evidence_text=evidence_text,
        hypothesis_text=hypothesis_text,
        analysis_depth="academic_semantic_relationship_assessment"
    )
```

#### **Task 1.3: Upgrade Van Evera Test Confidence to LLM Academic Reasoning**
**File**: `core/plugins/van_evera_testing.py:423-465`
**Issue**: Fixed confidence thresholds (0.3, 0.8, 0.95) instead of contextual reasoning
**Impact**: Academic assessment using algorithmic scoring vs evidence-based evaluation

### **PHASE 2: COMPREHENSIVE INTELLIGENCE TRANSFORMATION (Medium Priority)**

#### **Task 2.1: Academic Quality Assessment Intelligence**
**File**: `core/plugins/primary_hypothesis_identifier.py:270-300`
**Replace**: Keyword counting for theoretical sophistication with LLM academic assessment

#### **Task 2.2: Dynamic Prediction Generation**
**File**: `core/plugins/van_evera_testing.py:423-465`
**Replace**: Hardcoded prediction templates with LLM context-aware generation

#### **Task 2.3: Research Question Intelligence**
**File**: `core/plugins/research_question_generator.py:135-317`
**Replace**: Pattern-based classification with LLM academic question generation

### **Quality Validation Framework**

**Pre-Transformation Baseline**:
```bash
# Capture current quality metrics for comparison
python -c "
from core.analyze import main
result = main('revolutions_20250805_122000_graph.json')
print(f'Van Evera Compliance: {result.get(\"van_evera_compliance\")}')
print(f'Academic Quality Score: {result.get(\"academic_quality_score\")}')
"
```

**Success Criteria**:
- Van Evera compliance >90% (vs current 71.5%)
- Zero logical contradictions (fix AE_001 type issues)
- Diagnostic accuracy >85% through semantic understanding
- All rule-based keyword matching eliminated

### **Evidence Requirements**

Create structured evidence files:
```
evidence/
├── current/
│   ├── Evidence_LLM_ENHANCEMENT_Diagnostic_Classifier.md
│   ├── Evidence_LLM_ENHANCEMENT_Evidence_Connection.md
│   └── Evidence_LLM_ENHANCEMENT_Van_Evera_Testing.md
└── completed/
    └── Evidence_CODE_QUALITY_Improvements.md
```

**Required Evidence Per Task**:
- Raw LLM API logs and responses
- A/B comparison data (rule-based vs LLM accuracy)
- Quality metrics before/after transformation
- Performance benchmarks (speed, token usage, success rates)

## 🔧 IMPLEMENTATION GUIDANCE

### **Evidence-First Methodology**
1. **Validate Current State**: Understand why LLM was disabled originally
2. **Implement Incrementally**: Enable one LLM enhancement at a time with validation
3. **Measure Improvement**: Quantify quality gains through academic metrics
4. **Document Evidence**: Record all LLM integration results and comparisons

### **Quality Gates**
- **Phase 1**: LLM enhancements functional, measurable quality improvement
- **Phase 2**: All keyword matching eliminated, semantic understanding verified
- **Overall**: Van Evera pipeline maintains functionality with enhanced academic rigor

**Process Compliance**: Follow evidence-based development - validate improvements through measurable academic quality gains.

## Minor Technical Cleanup (Low Priority)

### **Cleanup Task: Resolve Pytest Warnings**
**Files**: `core/plugins/van_evera_testing.py` lines 15, 28, 40
**Issue**: Classes with `__init__` constructors triggering pytest collection warnings
**Fix**: Rename classes to avoid pytest test discovery or add `__test__ = False`

**Validation**:
```bash
python -m pytest tests/plugins/ -v  # Should run without warnings
```

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

**System Architecture**: Production-ready foundation ready for advanced LLM intelligence integration to achieve academic-quality semantic analysis.