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

## 🎯 CURRENT STATUS: Phase 21 - LLM-First Policy Violation Remediation (Updated 2025-01-09)

**System Status**: **CRITICAL QUALITY VIOLATIONS IDENTIFIED - IMMEDIATE REMEDIATION REQUIRED**  
**Latest Achievement**: **Phase 20 Complete - Comprehensive audit reveals 5+ LLM-First policy violations**  
**Current Priority**: **Convert keyword-based logic to LLM semantic analysis (quality-critical)**

**VIOLATION AUDIT RESULTS**:
- ❌ **Evidence Weighting**: Keyword matching for reliability/credibility assessment (`core/evidence_weighting.py:154-179`)
- ❌ **Confidence Calculator**: Hardcoded thresholds instead of LLM-generated (`core/confidence_calculator.py`)
- ❌ **Temporal Graph**: Rule-based node matching vs semantic analysis (`core/temporal_graph.py:487`)
- ❌ **Multiple Files**: Hardcoded probative values violating LLM-First architecture
- ✅ **WSL Migration**: Windows hangs resolved, analysis pipeline operational

**IMMEDIATE IMPACT**: Current system uses **rule-based logic instead of semantic understanding**, directly violating CLAUDE.md LLM-First Architecture Policy and **reducing output quality**.

## 🏆 PHASE 21: LLM-First Policy Violation Remediation

### OBJECTIVES: Convert all keyword/rule-based logic to LLM semantic analysis

**CRITICAL VIOLATIONS TO FIX**:
1. **`core/evidence_weighting.py:154-179`**: Evidence reliability/credibility keyword matching
2. **`core/confidence_calculator.py`**: Hardcoded confidence thresholds  
3. **`core/temporal_graph.py:487`**: Keyword-based node matching
4. **Plugin system audit**: Additional rule-based implementations

**STRATEGIC APPROACH**: Systematic conversion with quality preservation testing

### TASK 1: Convert Evidence Reliability Assessment (45 minutes)

**Target**: `core/evidence_weighting.py` function `_analyze_reliability_indicators()`

**Current Violation** (lines 154-155):
```python
positive_score = sum(1 for indicator in positive_indicators if indicator in combined_text)
negative_score = sum(1 for indicator in negative_indicators if indicator in combined_text)
```

**Required Conversion**:
```python
def _analyze_reliability_indicators(self, reasoning: str, justification: str) -> float:
    """Use LLM semantic analysis for evidence reliability assessment."""
    from .plugins.van_evera_llm_interface import VanEveraLLMInterface
    from .llm_required import LLMRequiredError
    
    try:
        llm_interface = VanEveraLLMInterface()
        
        prompt = f"""
Assess evidence reliability (0.0-1.0) based on these indicators:

Reasoning: {reasoning}
Justification: {justification}

Consider semantic understanding of:
- Verification status and documentation quality
- Source consistency and corroboration
- Multiple vs single source evidence
- Professional vs amateur collection methods

Return only numerical score 0.0-1.0 representing reliability.
"""
        
        response = llm_interface.get_structured_response(prompt, target_schema="reliability_float")
        return float(response.reliability_score)
        
    except Exception as e:
        raise LLMRequiredError(f"Failed to assess evidence reliability with LLM: {e}")
```

**Validation Requirements**:
1. **Unit test**: Compare LLM vs keyword results on test cases
2. **Edge case test**: Cases where keywords fail but LLM succeeds
3. **Integration test**: Full pipeline with converted function

### TASK 2: Convert Source Credibility Assessment (45 minutes)

**Target**: `core/evidence_weighting.py` function `_analyze_credibility_indicators()`

**Current Violation** (lines 178-179):
```python  
high_score = sum(1 for indicator in high_credibility if indicator in combined_text)
low_score = sum(1 for indicator in low_credibility if indicator in combined_text)
```

**Required Conversion**:
```python
def _analyze_credibility_indicators(self, reasoning: str, justification: str) -> float:
    """Use LLM semantic analysis for source credibility assessment."""
    from .plugins.van_evera_llm_interface import VanEveraLLMInterface
    from .llm_required import LLMRequiredError
    
    try:
        llm_interface = VanEveraLLMInterface()
        
        prompt = f"""
Assess source credibility (0.0-1.0) based on semantic analysis:

Reasoning: {reasoning}
Justification: {justification}

Consider semantic understanding of:
- Official vs unofficial source authority
- Academic/expert vs amateur qualifications
- Institutional vs individual source type
- Bias indicators and conflict of interest
- Publication and peer review status

Return only numerical score 0.0-1.0 representing credibility.
"""
        
        response = llm_interface.get_structured_response(prompt, target_schema="credibility_float")
        return float(response.credibility_score)
        
    except Exception as e:
        raise LLMRequiredError(f"Failed to assess source credibility with LLM: {e}")
```

### TASK 3: Remove Hardcoded Confidence Thresholds (30 minutes)

**Target**: `core/confidence_calculator.py` fallback logic

**Current Violation**: Default hardcoded thresholds (lines 37-47, 65-67)

**Required Change**:
1. **Remove all hardcoded threshold fallbacks**
2. **Ensure LLM-generated thresholds are always used**
3. **Raise LLMRequiredError if LLM threshold generation fails**

**Implementation**:
```python
@classmethod
def from_score(cls, score: float, thresholds=None) -> 'ConfidenceLevel':
    """Get confidence level from numerical score with LLM-generated thresholds."""
    if not thresholds:
        raise LLMRequiredError("Confidence thresholds must be LLM-generated - no hardcoded fallbacks allowed")
    
    # Use only LLM-generated thresholds
    if score >= thresholds.very_high_threshold:
        return ConfidenceLevel.VERY_HIGH
    elif score >= thresholds.high_threshold:
        return ConfidenceLevel.HIGH
    # ... etc - no fallback to hardcoded values
```

### TASK 4: Convert Temporal Node Matching (30 minutes)

**Target**: `core/temporal_graph.py:487` keyword matching

**Current Violation**:
```python
# Simple keyword matching - could be enhanced with NLP
```

**Required Conversion**: Already partially implemented with semantic analysis - ensure no keyword fallbacks remain.

### TASK 5: Plugin System Audit (60 minutes)

**Systematic search for additional violations**:

```bash
# Search for prohibited patterns
grep -r "if.*in.*text" core/plugins/
grep -r "keyword" core/plugins/  
grep -r "hardcoded" core/plugins/
grep -r "probative_value.*=" core/plugins/
```

**Document and fix any additional violations found**.

## 🧪 TESTING STRATEGY

### Test Level 1: Unit Testing (Per-Function)
```python
class TestLLMEvidenceWeighting:
    def test_reliability_llm_vs_keywords(self):
        """Compare LLM semantic analysis vs keyword matching"""
        test_cases = [
            # High quality cases
            ("documented verified multiple sources", "official government report"),
            # Edge cases where keywords fail
            ("unreliable hearsay but actually peer-reviewed study", "academic institution"),
            # Ambiguous cases requiring nuance
            ("single anonymous source with detailed corroborating evidence", "expert analysis")
        ]
        
        for reasoning, justification in test_cases:
            llm_score = self.quantifier._analyze_reliability_indicators(reasoning, justification)
            # Validate LLM provides nuanced assessment
            assert 0.0 <= llm_score <= 1.0
            # Add specific assertions about expected semantic understanding
    
    def test_llm_failure_handling(self):
        """Ensure LLMRequiredError on LLM failures"""
        # Mock LLM failure
        with pytest.raises(LLMRequiredError):
            self.quantifier._analyze_reliability_indicators("test", "test")
```

### Test Level 2: Integration Testing
```python
def test_end_to_end_evidence_assessment():
    """Test full EvidenceAssessment → EvidenceWeights pipeline"""
    # Use real EvidenceAssessment data
    # Validate LLM conversions work in full pipeline
    
def test_american_revolution_regression():
    """Compare output quality before/after LLM conversion"""
    # Run analysis on American Revolution
    # Compare evidence relevance scores
    # Validate quality improvements in edge cases
```

### Test Level 3: Performance Testing
```python
def test_llm_call_count_increase():
    """Measure additional LLM calls from conversion"""
    # Count calls before/after conversion
    # Document performance impact
    
def test_batching_opportunities():
    """Identify opportunities for batch LLM assessments"""
    # Multiple reliability assessments → single batch call
```

## 📊 SUCCESS CRITERIA

1. **Zero keyword-based logic** in semantic analysis functions
2. **Zero hardcoded probative values** or confidence thresholds  
3. **All LLM failures** result in LLMRequiredError (fail-fast)
4. **Quality improvements** demonstrated on test cases
5. **Regression tests pass** on American Revolution
6. **Performance impact** documented and acceptable

## 🎯 BATCH OPTIMIZATION STRATEGY (Post-Remediation)

After LLM-First compliance is achieved, implement user's batching insight:

**Current**: 15 evidence × 8 hypotheses = **120 LLM calls** (individual pairs)
**Target**: 8 hypotheses × 1 batch call = **8 LLM calls** (all evidence per hypothesis)

**Implementation**:
```python
def assess_all_evidence_for_hypothesis(hypothesis, all_evidence_items):
    """Single LLM call to assess all evidence against one hypothesis"""
    prompt = f"""
    Hypothesis: {hypothesis.description}
    
    Evidence Items:
    {format_all_evidence_items(all_evidence_items)}
    
    For each evidence item, assess:
    1. Relevance (0.0-1.0)
    2. Support/contradict/neutral
    3. Probative value (0.0-1.0)
    4. Reasoning
    
    Return structured assessment for all evidence items.
    """
```

**Expected Impact**: 93% reduction in LLM calls (120 → 8), 5-12 minutes → <30 seconds

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`process_trace_advanced.py`**: Main orchestration script with project selection and pipeline management
- **`core/analyze.py`**: Analysis phase entry point - contains violations requiring LLM conversion
- **`core/extract.py`**: Extraction phase entry point - working perfectly (132.93s for 39 nodes)

### Critical Violation Locations  
- **`core/evidence_weighting.py`**: Lines 154-179 keyword matching for reliability/credibility
- **`core/confidence_calculator.py`**: Hardcoded threshold fallbacks throughout
- **`core/temporal_graph.py`**: Line 487 keyword-based node matching
- **`core/plugins/`**: Unknown additional violations requiring systematic audit

### Important Integration Points
- **LLM Interface**: `core/plugins/van_evera_llm_interface.py` for semantic analysis
- **LLM Error Handling**: `core/llm_required.py` with LLMRequiredError fail-fast behavior
- **Structured Models**: `core/structured_models.py` for Pydantic validation

### WSL Environment Setup
- **Virtual Environment**: `test_env/` with all dependencies installed
- **Activation Command**: `source test_env/bin/activate`
- **Dependencies**: pandas, litellm, google-generativeai, networkx, pydantic

## 📋 Coding Philosophy

### NO LAZY IMPLEMENTATIONS
- No mocking, stubs, or pseudo-code
- Every LLM conversion must be fully functional
- Test each conversion before moving to next

### FAIL-FAST PRINCIPLES
- All LLM failures must raise LLMRequiredError
- No fallbacks to rule-based logic
- Surface semantic analysis failures immediately

### EVIDENCE-BASED DEVELOPMENT
- All quality claims require test evidence
- Compare semantic vs keyword results
- Document improvements with examples

---

## Evidence Structure

Evidence for this phase should be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase21_LLMFirstRemediation.md
```

Include:
- Before/after code comparisons for each conversion
- Unit test results showing LLM vs keyword differences
- Integration test results on American Revolution
- Performance impact measurements
- Quality improvement demonstrations

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.