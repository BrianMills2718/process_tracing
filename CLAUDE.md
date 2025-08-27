# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: ENHANCED ACADEMIC INTELLIGENCE (Updated 2025-01-27)

**System Status**: **Production-Ready with Advanced LLM Intelligence** - Core semantic understanding implemented  
**Current Priority**: **Academic Quality Enhancement** - Systematic improvement of Van Evera compliance and evidence balance  
**Academic Quality**: **Enhanced Foundation** - Sophisticated LLM mechanisms (0.35-0.7 completeness with detailed reasoning)

**COMPLETED LLM INTELLIGENCE FOUNDATION (2025-01-27)**:
- ✅ **Structured Output Integration**: All three critical plugins use VanEveraLLMInterface with proper Pydantic validation
- ✅ **Mechanism Quality**: 4/4 mechanisms have sophisticated LLM analysis with completeness/plausibility scoring
- ✅ **Semantic Understanding**: Evidence connections use CausalRelationshipAnalysis vs hardcoded keyword bridges
- ✅ **Van Evera Testing**: Diagnostic tests operational with structured confidence calculation methods
- ✅ **End-to-End Validation**: Full American Revolution analysis with 379 causal chains and comprehensive HTML output

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Academic Functionality**: Target ≥80% Van Evera compliance (current foundation: enhanced mechanisms working)
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

## 🚀 NEXT PHASE: ACADEMIC QUALITY ENHANCEMENT

### **Priority: Systematic Van Evera Compliance and Evidence Balance Improvement**

**Current Opportunity**: System has sophisticated LLM mechanisms (working perfectly) but needs systematic improvement in hypothesis evaluation, evidence balance, and Van Evera test integration for full academic rigor.

**Evidence of Need**: End-to-end American Revolution analysis shows:
- ✅ **Mechanism Quality**: 4/4 with detailed LLM analysis (0.35-0.7 completeness, 1500-word academic reasoning)
- ❌ **Hypothesis Confidence**: 5/5 hypotheses missing LLM-generated confidence scores
- ❌ **Evidence Balance**: 1.00 support ratio (academic ideal: 0.6-0.8)  
- ❌ **Van Evera Integration**: Console shows tests running but results not populating final output

### **PHASE 1: SYSTEMATIC ACADEMIC ENHANCEMENT (High Priority)**

#### **Task 1.1: Van Evera Test Integration**
**File**: `core/analyze.py` hypothesis evaluation pipeline
**Issue**: Van Evera tests execute (`[VAN_EVERA_TESTING] hoop test: PASS`) but results don't populate hypothesis confidence scores
**Impact**: Missing critical academic methodology integration despite tests running

**Investigation Required**:
1. Trace Van Evera plugin result storage in `plugin_results['van_evera_testing']`
2. Map plugin execution results to final `hypotheses_evaluation` data flow
3. Identify where test outcomes get disconnected from confidence calculation

**Implementation**:
```python
def integrate_van_evera_results_to_hypotheses(analysis_results):
    """Integrate Van Evera test results into hypothesis confidence scoring"""
    van_evera_results = analysis_results.get('van_evera_assessment', {})
    
    for hyp_id, hyp_data in analysis_results['evidence_analysis'].items():
        # Extract Van Evera test results for this hypothesis
        test_outcomes = extract_van_evera_outcomes(van_evera_results, hyp_id)
        
        # Calculate confidence from test results
        confidence_score = calculate_confidence_from_tests(test_outcomes)
        
        # Add to hypothesis data
        hyp_data['van_evera_confidence'] = confidence_score
        hyp_data['test_details'] = test_outcomes
```

#### **Task 1.2: Hypothesis LLM Enhancement Pipeline** 
**File**: Create `core/enhance_hypotheses.py` (following mechanism pattern)
**Issue**: Hypotheses lack LLM-generated confidence scores and narrative synthesis despite sophisticated mechanism analysis
**Impact**: Academic analysis incomplete - mechanisms enhanced but hypotheses remain rule-based

**Implementation**:
```python
# Create enhance_hypotheses.py following core/enhance_mechanisms.py pattern
from .van_evera_llm_interface import VanEveraLLMInterface
from .van_evera_llm_schemas import VanEveraPredictionEvaluation

def enhance_hypothesis_with_llm(hypothesis_node, supporting_evidence, refuting_evidence, van_evera_tests):
    """Use VanEveraLLMInterface for sophisticated hypothesis evaluation"""
    llm_interface = VanEveraLLMInterface()
    
    structured_result = llm_interface.evaluate_prediction_structured(
        prediction_description=hypothesis_node['description'],
        diagnostic_type=determine_primary_diagnostic_type(van_evera_tests),
        theoretical_mechanism=extract_theoretical_mechanism(hypothesis_node),
        evidence_context=compile_evidence_context(supporting_evidence, refuting_evidence)
    )
    
    return structured_result  # Returns VanEveraPredictionEvaluation with confidence_score, reasoning
```

**Integration into analyze.py**:
Add hypothesis LLM enhancement loop following the mechanism enhancement pattern (lines 2850-2900).

#### **Task 1.3: Evidence Balance Correction**
**File**: Evidence classification pipeline in hypothesis evaluation
**Issue**: 1.00 evidence support ratio violates academic standards (ideal: 0.6-0.8)
**Impact**: System shows confirmation bias - not systematically seeking disconfirming evidence

**Investigation Required**:
1. Identify why no refuting evidence is found in current classification
2. Map how Van Evera FAIL results should contribute to refuting evidence
3. Examine evidence-hypothesis linking algorithms for bias

**Implementation**:
```python  
def systematic_evidence_evaluation(hypothesis, all_evidence_nodes, van_evera_results):
    """Systematically evaluate evidence as supporting/refuting with academic balance"""
    
    # Use Van Evera FAIL results as refuting evidence indicators
    failed_tests = extract_failed_van_evera_tests(van_evera_results, hypothesis['id'])
    
    # Actively seek disconfirming evidence
    refuting_evidence = identify_disconfirming_evidence(hypothesis, all_evidence_nodes, failed_tests)
    
    # Balance evidence discovery (target 0.6-0.8 support ratio)
    balanced_evidence = balance_evidence_classification(supporting_evidence, refuting_evidence)
    
    return balanced_evidence
```

### **PHASE 2: ACADEMIC RIGOR EXPANSION (Medium Priority)**

#### **Task 2.1: Alternative Explanation Enhancement**
**Current**: 2 alternative explanations (1 eliminated, 1 active)
**Target**: 3-5 systematically evaluated alternative explanations
**Implementation**: Expand alternative hypothesis generation and use LLM evaluation

#### **Task 2.2: Causal Chain LLM Assessment**  
**Current**: 379 causal chains identified but no quality scoring
**Target**: LLM-based plausibility and completeness assessment for multi-step chains
**Implementation**: Apply MechanismAssessment approach to causal chain evaluation

## Implementation Guidelines

### **Evidence Requirements**
Create structured evidence files in `evidence/current/` for each task:
```
evidence/current/
├── Evidence_VAN_EVERA_Integration.md
├── Evidence_HYPOTHESIS_Enhancement.md
└── Evidence_EVIDENCE_Balance.md
```

**Required Evidence Per Task**:
- Raw console logs showing before/after behavior
- Van Evera compliance measurements with specific metrics
- Academic quality improvements with quantified evidence balance ratios
- End-to-end test results demonstrating integration success

### **Validation Protocol**
**Before Each Task**:
```bash
# Capture current baseline metrics
python -m core.analyze output_data/revolutions/revolutions_20250805_122000_graph.json --html
# Document: hypothesis confidence scores, evidence ratios, Van Evera test visibility
```

**After Each Task**:
```bash
# Validate improvements
python -m core.analyze output_data/revolutions/revolutions_20250805_122000_graph.json --html
# Compare: academic quality metrics, ensure no mechanism regression
python -m pytest tests/plugins/ -v  # Ensure no regression
```

**Success Criteria**:
- Van Evera compliance >80% (measurable improvement)
- Evidence balance 0.6-0.8 support ratio (from current 1.0)
- All hypotheses have LLM-generated confidence scores and reasoning
- Van Evera test results visible in final analysis output
- No regression in existing mechanism LLM quality

### **Quality Gates**
- **Academic**: Measurable Van Evera compliance improvement
- **Technical**: All existing LLM integrations continue functioning
- **Evidence**: Structured evidence files document all improvements with raw data
- **Integration**: Full end-to-end analysis produces enhanced academic output

## Codebase Structure

### Key Entry Points
- `core/analyze.py`: Main analysis orchestration with Van Evera pipeline and LLM integration
- `core/plugins/van_evera_workflow.py`: 8-step academic analysis workflow
- `core/plugins/van_evera_llm_interface.py`: Professional LLM integration with structured output

### Critical Integration Points
- **LLM Interface**: `van_evera_llm_interface.py` - VanEveraLLMInterface with all structured methods
- **Plugin Architecture**: 16 plugins with proper abstractions and LLM enhancement
- **Structured Output**: `van_evera_llm_schemas.py` - Academic Pydantic models for all Van Evera concepts

### Module Organization
- `core/plugins/`: 16 registered plugins with structured LLM integration
- `core/logging_utils.py`: Structured logging utilities for operational context  
- `universal_llm_kit/`: LiteLLM integration with Gemini 2.5 Flash
- `evidence/`: Structured evidence documentation (current/ and completed/ phases)

**System Architecture**: Enhanced foundation ready for systematic academic quality improvements to achieve professional Van Evera compliance and evidence balance.