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

## 🎯 CURRENT STATUS: Phase 4 - LLM Efficiency Optimization (Updated 2025-01-28)

**System Status**: **LLM-First Architecture Complete (75%)** - Now optimizing for efficiency
**Current Priority**: **Zero-Quality-Loss Optimizations** - Reduce LLM calls by 50-70%
**System Goal**: **Efficient LLM Usage** - Maintain quality while dramatically reducing API costs

**PHASE 3 COMPLETED:**
- ✅ 21 of 28 files migrated to LLM-first (75% complete)
- ✅ All critical semantic decisions use LLM
- ✅ SemanticAnalysisService operational with caching
- ✅ System is functionally LLM-first and production-ready

**PHASE 4 OBJECTIVE:**
Reduce LLM calls from 15-25 to 5-8 per analysis through intelligent batching and caching, with ZERO quality degradation.

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

## 🚀 PHASE 4: Zero-Quality-Loss LLM Efficiency Optimization

### Overview: Efficiency Without Quality Compromise

**Current Problem**: System makes 15-25 separate LLM calls per analysis, causing:
- High API costs ($0.50-1.00 per analysis)
- Slow response times (30-45 seconds)
- Redundant processing of same content
- Fragmented understanding (LLM loses context between calls)

**Solution**: Four zero-quality-loss optimizations that will:
- Reduce LLM calls by 50-70% (from 15-25 to 5-8)
- Improve quality through coherent analysis
- Cut costs by 60-70%
- Speed up analysis by 2-3x

## Task 1: Batch Related Analyses (50-70% Fewer Calls)

### Problem Analysis
**Current Anti-Pattern**: Making multiple separate LLM calls for related analyses
```python
# BAD - Current fragmented approach (3-5 separate calls):
domain = semantic_service.classify_domain(evidence)
probative = semantic_service.assess_probative_value(evidence, hypothesis)
contradiction = semantic_service.detect_contradiction(evidence, hypothesis)
mechanism = semantic_service.detect_mechanism(evidence)
actors = semantic_service.identify_actors(evidence)
```

### Required Implementation

**Step 1.1: Create Comprehensive Analysis Schema**
**File**: `core/plugins/van_evera_llm_schemas.py`
**Action**: Add new comprehensive schema

```python
class ComprehensiveEvidenceAnalysis(BaseModel):
    """Single schema capturing all semantic features of evidence"""
    
    # Domain Analysis
    primary_domain: Literal["political", "economic", "ideological", "military", 
                           "social", "cultural", "religious", "technological"]
    secondary_domains: List[str] = Field(default_factory=list)
    domain_confidence: float = Field(ge=0.0, le=1.0)
    domain_reasoning: str
    
    # Probative Assessment
    probative_value: float = Field(ge=0.0, le=1.0)
    probative_factors: List[str]
    evidence_quality: Literal["high", "medium", "low"]
    reliability_score: float = Field(ge=0.0, le=1.0)
    
    # Hypothesis Relationship
    relationship_type: Literal["supports", "contradicts", "neutral", "ambiguous"]
    relationship_confidence: float = Field(ge=0.0, le=1.0)
    relationship_reasoning: str
    van_evera_diagnostic: Literal["hoop", "smoking_gun", "doubly_decisive", "straw_in_wind"]
    
    # Semantic Features
    causal_mechanisms: List[Dict[str, str]] = Field(default_factory=list)
    temporal_markers: List[Dict[str, str]] = Field(default_factory=list)
    actor_relationships: List[Dict[str, str]] = Field(default_factory=list)
    
    # Meta-Analysis
    key_concepts: List[str]
    contextual_factors: List[str]
    alternative_interpretations: List[str] = Field(default_factory=list)
    confidence_overall: float = Field(ge=0.0, le=1.0)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
```

**Step 1.2: Implement Comprehensive Analysis Method**
**File**: `core/plugins/van_evera_llm_interface.py`
**Action**: Add new unified analysis method

```python
def analyze_evidence_comprehensive(self, 
                                  evidence_description: str,
                                  hypothesis_description: str,
                                  context: Optional[str] = None) -> ComprehensiveEvidenceAnalysis:
    """
    Comprehensive evidence analysis in a single LLM call.
    Replaces 5-10 separate calls with one coherent analysis.
    """
    prompt = f"""
    Perform comprehensive semantic analysis of this evidence-hypothesis relationship.
    Extract ALL features in one coherent analysis.
    
    EVIDENCE: {evidence_description}
    HYPOTHESIS: {hypothesis_description}
    CONTEXT: {context or 'Process tracing analysis'}
    
    Analyze comprehensively:
    1. DOMAIN: Classify primary and secondary domains with reasoning
    2. PROBATIVE VALUE: Assess evidence strength (0.0-1.0) with factors
    3. RELATIONSHIP: Determine if evidence supports/contradicts hypothesis
    4. VAN EVERA TYPE: Classify as hoop/smoking gun/doubly decisive/straw in wind
    5. CAUSAL MECHANISMS: Identify all cause-effect relationships
    6. TEMPORAL MARKERS: Extract time references and sequences
    7. ACTORS: Identify actors and their relationships
    8. KEY CONCEPTS: Extract main conceptual elements
    9. CONTEXT: Note contextual factors affecting interpretation
    
    Provide reasoning that considers relationships between all features.
    """
    
    return self._get_structured_response(prompt, ComprehensiveEvidenceAnalysis)
```

**Step 1.3: Update SemanticAnalysisService**
**File**: `core/semantic_analysis_service.py`
**Action**: Add routing to comprehensive analysis

```python
def analyze_comprehensive(self, evidence: str, hypothesis: str, 
                        context: Optional[str] = None) -> ComprehensiveEvidenceAnalysis:
    """Route to comprehensive analysis with caching"""
    cache_key = self._generate_cache_key(evidence, hypothesis, "comprehensive")
    
    if cache_key in self._cache:
        if not self._is_cache_expired(cache_key):
            self.cache_hits += 1
            return self._cache[cache_key][0]
    
    # Single LLM call replacing multiple calls
    result = self.llm_interface.analyze_evidence_comprehensive(
        evidence, hypothesis, context
    )
    
    self._cache[cache_key] = (result, datetime.now())
    self.call_count += 1
    
    return result

# Backward compatibility methods that extract from comprehensive
def classify_domain(self, text: str, context: Optional[str] = None) -> DomainClassification:
    """Extract domain from comprehensive analysis for backward compatibility"""
    # If we have a cached comprehensive analysis, extract from it
    # Otherwise, do comprehensive and extract
    pass
```

**Validation**: Compare output quality before/after batching. Must show equal or better coherence.

---

## Task 2: Smarter Caching (30-40% Reduction)

### Required Implementation

**Step 2.1: Create Semantic Signature System**
**File**: `core/semantic_signature.py` (NEW)

Create a semantic fingerprinting system that generates cache keys based on meaning rather than exact text. This allows semantically similar queries to hit the cache even with different wording.

**Step 2.2: Implement Multi-Layer Cache**
**File**: `core/semantic_analysis_service.py`

Upgrade the current MD5-based cache to a three-layer system:
- L1 Cache: Exact text matches (instant hit)
- L2 Cache: Semantic signature matches (very fast)
- L3 Cache: Partial results that can be reused

**Validation**: Test with paraphrased inputs. Target 40-60% cache hit rate.

---

## Task 3: Single Document Analysis (60% Reduction)

### Required Implementation

**Step 3.1: Create Evidence Pre-Analysis**
**File**: `core/evidence_document.py` (NEW)

Implement a system where evidence is comprehensively analyzed once, then efficiently evaluated against multiple hypotheses using the cached analysis.

**Step 3.2: Update Processing Flow**
**File**: `core/analyze.py`

Modify the main analysis loop to use pre-analyzed evidence documents instead of re-analyzing for each hypothesis.

**Validation**: Verify consistent interpretation across all hypothesis evaluations.

---

## Task 4: Compound Feature Extraction (40% Reduction)

### Required Implementation

**Step 4.1: Create Multi-Feature Schema**
**File**: `core/plugins/van_evera_llm_schemas.py`

Add `MultiFeatureExtraction` schema that captures all semantic features (actors, mechanisms, temporal markers, concepts) in one structure.

**Step 4.2: Implement Compound Extraction**
**File**: `core/plugins/van_evera_llm_interface.py`

Add `extract_all_features()` method that extracts all features in one LLM call, capturing relationships between features.

**Validation**: Compare feature completeness and relationship detection.
---

## Implementation Strategy

### Phase 1: Schema Development (2 hours)
1. Add `ComprehensiveEvidenceAnalysis` schema to `van_evera_llm_schemas.py`
2. Add `MultiFeatureExtraction` schema for compound extraction
3. Write schema validation tests

### Phase 2: Core Infrastructure (3 hours)
1. Implement comprehensive analysis method in `van_evera_llm_interface.py`
2. Create `semantic_signature.py` for smart caching
3. Create `evidence_document.py` for pre-analysis
4. Upgrade `semantic_analysis_service.py` with multi-layer cache

### Phase 3: Integration (2 hours)
1. Update calling code to use new batched methods
2. Add backward compatibility wrappers
3. Implement feature extraction helpers

### Phase 4: Validation (2 hours)
1. Compare quality metrics before/after
2. Measure performance improvements
3. Validate cache effectiveness
4. Document results in evidence files

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

### Success Criteria (75% Achieved)
- ⚠️ **Keyword Matching Reduction**: ~50 patterns remain (structural only, not semantic)
- ✅ **Critical Semantic Decisions**: All use LLM analysis (100% complete)
- ✅ **Dataset Generalization**: 90% of American Revolution references removed
- ✅ **Universal Applicability**: System functional for any historical period/domain
- ✅ **LLM Semantic Understanding**: All critical classifications based on meaning

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

### Evidence-Based Validation Requirements

#### Required Evidence Files (Phase 4)
```
evidence/current/
├── Evidence_Phase4_Baseline_Performance.md    # Current LLM call counts and timings
├── Evidence_Phase4_Optimization_Results.md    # After optimization metrics
├── Evidence_Phase4_Quality_Comparison.md      # Side-by-side quality validation
├── Evidence_Phase4_Cache_Effectiveness.md     # Cache hit rates and performance
└── validate_phase4_optimization.py            # Script to measure improvements
```

#### Success Metrics
- **Performance**: 50%+ reduction in LLM calls (from 15-25 to 5-8)
- **Quality**: No degradation in analysis accuracy
- **Cache Hit Rate**: 40-60% with semantic signatures
- **Cost Reduction**: 60-70% lower API costs
- **Response Time**: 2-3x faster (10-15s vs 30-45s)

### Phase 3 Evidence Files (Reference)
```
evidence/current/
├── Evidence_Phase3_Final_Report.md           # 75% completion evidence
├── Evidence_Phase3_Implementation_Start.md   # Initial state documentation  
├── Evidence_Phase3_Session_Complete.md       # Mid-point progress (64.3%)
└── validate_phase3.py                        # Validation showing 75% complete
```

## Phase 3 Accomplishments (75% System Migration)

### Successfully Migrated (21/28 files)
**Core Modules (11 files - 100% of critical paths):**
- semantic_analysis_service.py - NEW centralized service (303 lines)
- analyze.py - Actor relevance using LLM
- van_evera_testing_engine.py - Completely LLM-first
- connectivity_analysis.py - All patterns eliminated
- All other critical core modules

**Plugin Files (10/16 files):**
- Advanced prediction engine
- Evidence connector enhancer
- Diagnostic classifier and rebalancer
- Research question generator
- Additional testing plugins

### Remaining Work (Optional - 25%)
**Non-Critical Patterns (~50 remaining):**
- Structural checks: `if 'properties' in data`
- Response validation: `if 'result' in response`
- Minor confidence calculations
- Not affecting semantic understanding

**System Performance:**
- LLM calls: 15-25 per analysis (with caching)
- Cache hit rate: Improving with use
- Response time: Acceptable for research use
- All critical paths: Using LLM semantic analysis

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