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

## 🎯 CURRENT STATUS: LLM-FIRST COMPREHENSIVE MIGRATION (Updated 2025-01-28)

**System Status**: **Phase 1 Complete, Phase 2 Required** - One function replaced, dozens more remain rule-based  
**Current Priority**: **Complete LLM-First Migration** - Replace ALL keyword matching throughout codebase  
**System Goal**: **Generalist Process Tracing** - Remove all dataset-specific hardcoding for universal applicability

**PHASE 1 COMPLETED (2025-01-28)**:
- ✅ **Evidence Classification**: `_identify_contradiction_patterns()` replaced with LLM semantic analysis
- ✅ **System Validation**: Full analysis completes successfully without crashes
- ✅ **Semantic Accuracy**: Boston Massacre correctly classified as supporting ideological movement

**PHASE 2 REQUIRED - COMPREHENSIVE RULE-BASED ELIMINATION**:
- ❌ **Van Evera Testing**: Still uses keyword matching for hypothesis classification
- ❌ **Domain Classification**: Still uses hardcoded keyword lists (political/economic/ideological)
- ❌ **Probative Value Assignment**: Still uses hardcoded values (0.7, 0.6, etc.)
- ❌ **Diagnostic Type Determination**: Still uses keyword-based Van Evera type assignment
- ❌ **Alternative Hypothesis Generation**: Still uses domain keyword dictionaries
- ❌ **Dataset-Specific Logic**: American Revolution-specific rules throughout codebase

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

## 🚨 PHASE 2: COMPREHENSIVE LLM-FIRST MIGRATION (Critical Priority)

### **MANDATORY MIGRATION**: Eliminate ALL Rule-Based Logic Throughout System

**Scope**: Replace ALL keyword matching, hardcoded values, and dataset-specific logic with LLM semantic understanding

**Files Requiring Complete LLM Migration**:
1. `core/van_evera_testing_engine.py` - Hypothesis classification, test generation
2. `core/plugins/content_based_diagnostic_classifier.py` - Domain classification  
3. `core/alternative_hypothesis_generator.py` - Domain keyword dictionaries
4. `core/plugins/advanced_van_evera_prediction_engine.py` - Domain detection
5. `core/plugins/research_question_generator.py` - Domain keywords
6. Multiple files with hardcoded probative value assignments

### **TASK 2A: Van Evera Testing Engine LLM Migration**
**File**: `core/van_evera_testing_engine.py`
**Critical Issue**: Lines 127, 174, 259, 270 use keyword matching for hypothesis classification

**Current Rule-Based Logic**:
```python
# PROHIBITED - Replace with LLM:
if 'ideological' in hypothesis_desc.lower() and 'political' in hypothesis_desc.lower():
if 'merchant' in alt_desc or 'economic' in alt_desc:
if any(term in desc_lower for term in ['taxation', 'tax', 'representation', 'parliament']):
```

**Required Implementation**:
1. **New Schema**: `HypothesisDomainClassification` in `van_evera_llm_schemas.py`
2. **New LLM Method**: `classify_hypothesis_domain()` in `VanEveraLLMInterface`
3. **Replace All Keyword Logic**: Use LLM semantic understanding for domain detection
4. **Generalist Approach**: Remove American Revolution-specific keywords

**Schema Definition**:
```python
class HypothesisDomainClassification(BaseModel):
    primary_domain: Literal["political", "economic", "ideological", "military", "social", "cultural", "religious", "technological"]
    secondary_domains: List[str] = Field(description="Additional relevant domains")
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Semantic reasoning for domain classification")
    generalizability: str = Field(description="How this applies beyond specific historical contexts")
```

**LLM Method**:
```python
def classify_hypothesis_domain(self, hypothesis_description: str, 
                             context: Optional[str] = None) -> HypothesisDomainClassification:
    """
    Classify hypothesis domain using semantic understanding.
    Replaces keyword matching with universal domain analysis.
    """
    prompt = f"""
    Analyze this hypothesis to determine its primary domain using semantic understanding.
    
    HYPOTHESIS: {hypothesis_description}
    CONTEXT: {context or 'Universal process tracing analysis'}
    
    Classify the hypothesis into domains based on its SEMANTIC CONTENT, not keywords:
    - POLITICAL: Government, authority, power structures, governance, policy
    - ECONOMIC: Trade, resources, financial systems, markets, wealth
    - IDEOLOGICAL: Beliefs, values, worldviews, philosophical positions
    - MILITARY: Armed conflict, strategy, warfare, defense
    - SOCIAL: Community structures, relationships, class, identity
    - CULTURAL: Traditions, customs, arts, shared practices
    - RELIGIOUS: Faith, spiritual beliefs, religious institutions
    - TECHNOLOGICAL: Innovation, technical advancement, tools, methods
    
    Provide semantic reasoning that would apply across ANY historical period or domain.
    Avoid dataset-specific keywords - focus on universal conceptual categories.
    """
    
    return self._get_structured_response(prompt, HypothesisDomainClassification)
```

### **TASK 2B: Content-Based Diagnostic Classifier Migration**
**File**: `core/plugins/content_based_diagnostic_classifier.py`
**Issue**: Lines 109-114 use hardcoded keyword lists for domain classification

**Current Rule-Based Logic**:
```python
# PROHIBITED - Replace with LLM:
'political_keywords': ['constitutional', 'rights', 'representation', 'assembly', 'government'],
'economic_keywords': ['trade', 'taxation', 'revenue', 'commercial', 'merchant', 'profit'],
```

**Required Implementation**:
1. **Replace Keyword Lists**: Use LLM domain classification from Task 2A
2. **Semantic Analysis**: Replace hardcoded matching with semantic understanding
3. **Universal Applicability**: Remove historical period restrictions

### **TASK 2C: Probative Value LLM Generation** 
**Files**: Multiple files with hardcoded `probative_value` assignments
**Issue**: Lines throughout codebase assign arbitrary values (0.7, 0.6, etc.)

**Current Rule-Based Logic**:
```python
# PROHIBITED - Replace with LLM:
'probative_value': 0.7  # High probative value for academic rigor
'probative_value': min(0.6, contradiction_indicators * 0.3)
probative_value *= 0.7  # Reduce probative value when demoting
```

**Required Implementation**:
1. **New Schema**: `ProbativeValueAssessment` in `van_evera_llm_schemas.py` 
2. **New LLM Method**: `assess_probative_value()` in `VanEveraLLMInterface`
3. **Replace All Hardcoded Values**: Use LLM assessment for evidence strength

**Schema Definition**:
```python
class ProbativeValueAssessment(BaseModel):
    probative_value: float = Field(ge=0.0, le=1.0, description="Evidence strength assessment")
    confidence_score: float = Field(ge=0.0, le=1.0) 
    reasoning: str = Field(description="Academic justification for probative value")
    evidence_quality_factors: List[str] = Field(description="Factors contributing to evidence strength")
    reliability_assessment: str = Field(description="Assessment of evidence reliability and credibility")
    van_evera_implications: str = Field(description="Implications for Van Evera diagnostic testing")
```

### **TASK 2D: Alternative Hypothesis Generator Migration**
**File**: `core/alternative_hypothesis_generator.py`
**Issue**: Lines 239-245 use domain keyword dictionaries

**Current Rule-Based Logic**:
```python
# PROHIBITED - Replace with LLM:
'economic': ['merchant', 'trade', 'profit', 'economic', 'commercial', 'business'],
'ideological': ['idea', 'philosophy', 'enlightenment', 'rights', 'liberty', 'freedom'],
```

**Required Implementation**:
1. **Replace Keyword Dictionaries**: Use semantic understanding for alternative generation
2. **LLM Alternative Generation**: Generate context-appropriate competing hypotheses
3. **Universal Scope**: Generate alternatives for any domain/time period

### **TASK 2E: Dataset-Specific Logic Elimination**
**Files**: Multiple files with American Revolution-specific hardcoding
**Issue**: System hardcoded for specific historical period instead of universal process tracing

**Examples to Replace**:
```python
# PROHIBITED - Remove dataset-specific logic:
# Political contradictions (American Revolution specific)
if 'ideological' in hypothesis_desc and 'economic' in evidence_desc:
    contradiction_count += 0.3

# American Revolution keywords
['taxation', 'tax', 'representation', 'parliament', 'stamp', 'townshend']
```

**Required Implementation**:
1. **Generalist Approach**: Replace all historical period-specific logic with universal semantic analysis
2. **LLM Context Awareness**: Let LLM understand context without hardcoded rules
3. **Domain Neutrality**: System works equally well for any historical period or domain

## Implementation Strategy

### Phase 2A: Core LLM Infrastructure (Safe)
1. **Extend Schemas**: Add all required Pydantic models to `van_evera_llm_schemas.py`
2. **Extend LLM Interface**: Add all semantic analysis methods to `VanEveraLLMInterface`
3. **Test Infrastructure**: Validate new LLM methods work correctly

### Phase 2B: Van Evera Testing Engine Migration (High Risk)
1. **Replace Hypothesis Classification**: Use LLM domain analysis instead of keywords
2. **Replace Test Generation**: Use LLM to create context-appropriate tests
3. **Validate Test Quality**: Ensure LLM generates proper Van Evera diagnostic tests

### Phase 2C: System-Wide Keyword Elimination (High Risk)
1. **Search and Replace**: Find all keyword matching throughout codebase
2. **Replace with LLM Calls**: Use appropriate semantic analysis methods
3. **Remove Hardcoded Values**: Replace all probative value assignments with LLM assessment

### Phase 2D: Dataset Generalization (Medium Risk)
1. **Remove Historical Specificity**: Eliminate American Revolution-specific logic
2. **Universal Context**: Make system work for any historical period or domain
3. **Test Across Domains**: Validate system works beyond American Revolution

## Validation Protocol

### Success Criteria
- ✅ **Zero Keyword Matching**: No `if 'keyword' in text` anywhere in codebase
- ✅ **Zero Hardcoded Values**: No `probative_value = 0.X` assignments  
- ✅ **Zero Dataset Specificity**: No American Revolution-specific logic
- ✅ **Universal Applicability**: System works for any historical period/domain
- ✅ **LLM Semantic Understanding**: All classifications based on meaning, not rules

### Test Cases
1. **Cross-Domain Testing**: Test with French Revolution, Industrial Revolution, Cold War datasets
2. **Semantic Accuracy**: Verify evidence classified correctly across different contexts  
3. **Domain Neutrality**: System performs equally well across different historical periods
4. **Alternative Generation**: LLM generates appropriate competing hypotheses for any context

## Codebase Structure

### Files Requiring Complete Migration
- `core/van_evera_testing_engine.py`: Hypothesis classification, test generation
- `core/plugins/content_based_diagnostic_classifier.py`: Domain classification systems
- `core/alternative_hypothesis_generator.py`: Alternative hypothesis generation with keywords
- `core/plugins/advanced_van_evera_prediction_engine.py`: Domain detection logic
- `core/plugins/research_question_generator.py`: Domain-based question generation
- `core/confidence_calculator.py`: Hardcoded confidence thresholds
- Multiple files: Probative value hardcoding throughout system

### Integration Points
- **Van Evera Testing Pipeline**: Must maintain academic rigor while using LLM analysis
- **Domain Classification**: Must work universally across all subject areas
- **Alternative Generation**: Must produce quality competing hypotheses for any context
- **Evidence Assessment**: Must generate appropriate probative values for any evidence type

### Evidence Files (Phase 2 Validation)
```
evidence/current/
├── Evidence_LLM_FIRST_Migration_Phase2.md     # Comprehensive migration results
├── Evidence_KEYWORD_Elimination_Complete.md   # Zero keyword matching validation
├── Evidence_GENERALIST_System_Validation.md   # Cross-domain testing results
└── Evidence_SEMANTIC_Accuracy_Assessment.md   # LLM classification quality
```

## Implementation Guidelines

### **Git-Based Safety**
- **Current Commit**: Known working state after Phase 1
- **Rollback Strategy**: `git revert HEAD` if any phase fails
- **Incremental Commits**: Commit each task separately for granular rollback

### **LLM Integration Requirements**
```python
# All LLM calls must follow this pattern:
try:
    return llm_semantic_analysis()
except Exception as e:
    logger.error(f"LLM analysis failed: {e}")
    return conservative_fallback()  # Don't crash the system
```

### **Performance Considerations**
- **Expected LLM Calls**: ~100-150 per analysis (vs current ~40-55)
- **Batch Processing**: Group similar analyses where possible
- **Caching**: Cache repeated domain/probative value assessments
- **Quality Trade-off**: Accept performance cost for semantic accuracy

### **Validation Requirements**
**Evidence Required for Task Completion**:
- Complete keyword elimination verification (`grep -r "if.*in.*desc"` returns zero matches)
- Cross-domain testing on non-American Revolution datasets
- Semantic accuracy verification on diverse test cases  
- Performance impact assessment with mitigation strategies
- Universal applicability demonstration

### **Success Declaration Requirements**
- **Keyword Elimination**: Zero rule-based logic remaining in codebase
- **Semantic Accuracy**: Evidence classified correctly across different domains
- **Universal System**: Works equally well for any historical period or subject area
- **Academic Quality**: Maintains Van Evera methodology standards across all contexts
- **Evidence Documentation**: All claims supported with cross-domain test results

## Next Steps After Phase 2

**Only After Successful Phase 2 Validation**:
- **Performance Optimization**: Optimize LLM calls for production use
- **Advanced Semantic Features**: Add sophisticated causal reasoning, counterfactual analysis
- **Multi-Language Support**: Extend system to work with non-English sources
- **Real-Time Analysis**: Optimize for live document analysis and streaming updates

**System Architecture**: Universal LLM-first process tracing system with complete semantic understanding, no rule-based logic, and applicability across all historical periods and domains.