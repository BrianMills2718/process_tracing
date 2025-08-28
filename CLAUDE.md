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

**REQUIRED IMPLEMENTATIONS**:
- ✅ LLM semantic analysis for ALL evidence-hypothesis relationships
- ✅ LLM-generated probative values with reasoning
- ✅ LLM-based domain and diagnostic type classification
- ✅ Structured Pydantic outputs for ALL semantic decisions
- ✅ Evidence-based confidence scoring through LLM evaluation

**APPROVAL REQUIRED**: Any rule-based logic must be explicitly approved with academic justification. Default assumption: **USE LLM SEMANTIC UNDERSTANDING**.

**VALIDATION REQUIREMENT**: All semantic decisions must be traceable to LLM reasoning outputs, not hardcoded logic.

---

## 🎯 CURRENT STATUS: LLM-FIRST MIGRATION (Updated 2025-01-28)

**System Status**: **Phase 1 Validation Complete** - Rule-based system identified and documented  
**Current Priority**: **Critical Evidence Classification Fix** - Replace keyword matching with LLM semantic analysis  
**Academic Issue**: **0% Academic Compliance** - Rule-based contradiction detection causes systematic misclassification

**ROOT CAUSE IDENTIFIED (2025-01-28)**:
- ❌ **Rule-Based Evidence Classification**: `_identify_contradiction_patterns()` uses keyword matching
- ❌ **Hardcoded Academic Logic**: `'ideological' in hypothesis and 'economic' in evidence → contradiction`
- ❌ **Systematic Misclassification**: Boston Massacre evidence incorrectly classified as refuting ideological movement
- ❌ **Academic Compliance Failure**: 0/3 hypotheses achieve 0.6-0.8 support ratio target (0% vs ≥50% required)

## Evidence-Based Development Philosophy (Mandatory)

### Core Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **DON'T EDIT GENERATED SYSTEMS**: Fix the autocoder itself, not generated outputs
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability

### Quality Standards
- **Academic Functionality**: Target ≥50% academic compliance (0.6-0.8 support ratios)
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

## 🚨 LLM-FIRST MIGRATION PROTOCOL (Critical Priority)

### **MANDATORY MIGRATION**: Replace Rule-Based Evidence Classification

**Critical Issue**: Evidence classification using keyword matching causes 0% academic compliance. System incorrectly classifies semantically supporting evidence as refuting based on keyword rules.

**Example Failure**:
- **Evidence**: "Boston Massacre turning colonial sentiment against British"
- **Hypothesis**: "American Revolution was ideological and political movement"
- **Current System**: Classified as REFUTING (keyword rule: 'ideological' + economic context → contradiction)
- **Should Be**: Classified as SUPPORTING (anti-British sentiment supports ideological movement)

### **PHASE 1: Evidence Classification Migration (CRITICAL)**

**Objective**: Replace `_identify_contradiction_patterns()` function with LLM semantic analysis
**Impact**: Direct fix for 0% academic compliance issue
**Risk**: High - Core evidence classification logic

#### **TASK 1A: Schema Creation (Safe)**
**File**: `core/plugins/van_evera_llm_schemas.py`
**Action**: Add new Pydantic schema for evidence-hypothesis relationship classification

**Required Schema**:
```python
class EvidenceRelationshipClassification(BaseModel):
    relationship_type: Literal["supporting", "refuting", "irrelevant"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed semantic reasoning for classification")
    probative_value: float = Field(ge=0.0, le=1.0, description="Evidence strength assessment")
    contradiction_indicators: int = Field(ge=0, description="Number of semantic contradictions identified")
    semantic_analysis: str = Field(description="Analysis of evidence-hypothesis semantic relationship")
```

#### **TASK 1B: LLM Interface Extension (Safe)**
**File**: `core/plugins/van_evera_llm_interface.py` 
**Action**: Add semantic evidence classification method

**Required Method**:
```python
def classify_evidence_relationship(self, evidence_description: str, 
                                 hypothesis_description: str) -> EvidenceRelationshipClassification:
    """
    Classify evidence-hypothesis relationship using semantic understanding.
    Replaces keyword-based contradiction detection with LLM semantic analysis.
    
    Args:
        evidence_description: Description of the evidence
        hypothesis_description: Description of the hypothesis
        
    Returns:
        Structured classification with reasoning and probative value
    """
    prompt = f"""
    Analyze the semantic relationship between this evidence and hypothesis using Van Evera academic methodology.
    
    EVIDENCE: {evidence_description}
    HYPOTHESIS: {hypothesis_description}
    
    Determine if the evidence:
    1. SUPPORTS the hypothesis (evidence strengthens or confirms the hypothesis)
    2. REFUTES the hypothesis (evidence weakens or contradicts the hypothesis)  
    3. Is IRRELEVANT to the hypothesis (no clear relationship)
    
    Consider semantic meaning, not keyword matching. Evidence about anti-British sentiment 
    SUPPORTS hypotheses about ideological movements, regardless of economic/political keyword conflicts.
    
    Provide detailed reasoning for your classification and assess the probative value (0.0-1.0).
    """
    
    return self._get_structured_response(prompt, EvidenceRelationshipClassification)
```

#### **TASK 1C: Function Replacement (HIGH RISK)**
**File**: `core/analyze.py` Lines 1070-1097
**Action**: Replace rule-based function with LLM semantic analysis

**Critical Requirements**:
1. **Preserve Function Signature**: Same parameters, same return type (float)
2. **Maintain Compatibility**: Return value must work with existing callers
3. **Error Resilience**: LLM failures cannot crash the system

**Implementation Strategy**:
```python
def _identify_contradiction_patterns(hypothesis_desc, evidence_desc):
    """
    REPLACED: Now uses LLM semantic analysis instead of keyword matching.
    Maintains original function signature for compatibility.
    """
    try:
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
        
        llm_interface = get_van_evera_llm()
        classification = llm_interface.classify_evidence_relationship(
            evidence_description=evidence_desc,
            hypothesis_description=hypothesis_desc
        )
        
        # Convert LLM classification to float for compatibility with existing code
        if classification.relationship_type == "refuting":
            return float(classification.contradiction_indicators)  # Use LLM-assessed contradiction count
        else:
            return 0.0  # No contradiction for supporting/irrelevant evidence
            
    except Exception as e:
        logger.error(f"LLM evidence classification failed: {e}", exc_info=True)
        return 0.0  # Conservative fallback - assume no contradiction
```

### **TASK 1D: Migration Validation (CRITICAL)**

**Objective**: Verify academic compliance improvement and system stability
**Test Dataset**: Use existing American Revolution analysis results

**Validation Protocol**:
```bash
# 1. Run full analysis with LLM classification
python -m core.analyze output_data/revolutions/revolutions_20250805_122000_graph.json --html

# 2. Measure academic compliance improvement
# Before: 0/3 hypotheses in 0.6-0.8 support ratio range (0%)
# Target: ≥50% of hypotheses in academic range

# 3. Verify semantic accuracy on known cases
# Evidence: "Boston Massacre turning sentiment against British" 
# Hypothesis: "Ideological movement"
# Expected: SUPPORTING classification (not refuting)
```

**Success Criteria**:
- ✅ Academic compliance improves from 0% to ≥20%
- ✅ Boston Massacre evidence correctly classified as supporting ideological movement
- ✅ System completes full analysis without crashes
- ✅ LLM cost increase <5x current costs
- ✅ Execution time increase <3x current time

**Failure Criteria (Rollback Required)**:
- ❌ System crashes or fails to generate results
- ❌ Academic compliance doesn't improve
- ❌ LLM classifications are semantically incorrect
- ❌ Cost or performance impact exceeds acceptable bounds

## Codebase Structure

### Key Files for Migration
- `core/analyze.py`: Lines 1070-1097 contain rule-based contradiction detection (TARGET FOR REPLACEMENT)
- `core/plugins/van_evera_llm_interface.py`: LLM interface requiring extension
- `core/plugins/van_evera_llm_schemas.py`: Pydantic schemas requiring new evidence classification schema
- `core/enhance_hypotheses.py`: Uses enhanced LLM interface for hypothesis evaluation

### Critical Integration Points
- **Evidence Classification Pipeline**: Lines 985-1068 in `core/analyze.py`
- **Academic Compliance Calculation**: Evidence balance ratios computed from supporting/refuting classification
- **HTML Output Generation**: Results must display in analysis summary
- **Van Evera Test Integration**: Must work with existing test result processing

### Evidence Files (Phase 1 Validation)
```
evidence/current/
├── Evidence_LLM_FIRST_Migration_Phase1.md     # Migration validation results
└── Evidence_ACADEMIC_Compliance_Improvement.md # Before/after compliance measurements
```

## Implementation Guidelines

### **Git-Based Safety**
- **Current Commit**: `e87815d` - Known working state with 0% academic compliance
- **Rollback Strategy**: `git revert HEAD` if migration fails
- **No Feature Flags**: Direct replacement for clean codebase

### **Error Handling Requirements**
```python
# All LLM calls must handle failures gracefully
try:
    return llm_semantic_analysis()
except Exception as e:
    logger.error(f"LLM classification failed: {e}")
    return conservative_fallback()  # Don't crash the system
```

### **Performance Monitoring**
- **Current**: ~30-40 LLM calls per analysis
- **After Migration**: ~40-55 LLM calls per analysis
- **Acceptable Impact**: <3x execution time, <5x cost increase

### **Validation Requirements**
**Evidence Required for Task Completion**:
- Console logs showing LLM classification execution
- Before/after academic compliance measurements
- Semantic accuracy verification on test cases
- Performance impact assessment (time and cost)
- No regression in existing functionality

### **Success Declaration Requirements**
- **Academic Improvement**: Quantified improvement in 0.6-0.8 support ratio compliance
- **Semantic Accuracy**: Evidence correctly classified based on meaning, not keywords
- **System Stability**: Full analysis completes without errors
- **Evidence Documentation**: All claims supported with raw execution logs

## Next Steps After Phase 1

**Only After Successful Phase 1 Validation**:
- **Phase 2**: Replace hardcoded probative value assignments with LLM assessment
- **Phase 3**: Replace domain classification keyword matching with LLM semantic analysis  
- **Phase 4**: Replace confidence threshold hardcoded ranges with LLM evaluation

**System Architecture**: Phase 1 implementation replaces rule-based evidence classification with LLM semantic understanding to achieve academic compliance standards.