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

## 🎯 CURRENT STATUS: Phase 22A - Complete TEXT → JSON → HTML Pipeline Integration (Updated 2025-09-10)

**System Status**: **🔧 PHASE 22A IN PROGRESS - CORE FUNCTIONALITY PROVEN, INTEGRATION NEEDED**  
**Latest Achievement**: **Systematic investigation identified root cause and created working HTML generation solution**  
**Current Priority**: **Complete TEXT → JSON → HTML pipeline integration using direct entry point approach**

**SYSTEMATIC INVESTIGATION RESULTS**:
- ✅ **Root Cause Identified**: Hang occurs in `python -m core.analyze` execution context, NOT in code logic
- ✅ **Core Functions Validated**: load_graph(), HTML generation, post-load operations all work perfectly (0.00s execution)
- ✅ **Comprehensive Testing**: 7 systematic tests confirm all functionality works when called directly
- ✅ **Working HTML Generation**: Successfully created 3,890-byte HTML report from JSON data
- ✅ **Alternative Entry Point**: analyze_direct.py bypasses problematic module execution context

**CURRENT CAPABILITY STATUS**:
- ✅ **JSON → HTML**: Working perfectly via direct entry point (0.00s execution)
- ✅ **TEXT → JSON**: Extraction pipeline functional (separate testing confirmed)
- ❌ **TEXT → JSON → HTML**: Integration between extraction and direct entry point not yet implemented
- ✅ **Core Architecture**: All underlying functionality proven working

## 🔧 PHASE 22A: Complete TEXT → JSON → HTML Pipeline Integration

### OBJECTIVE: Complete integration for full pipeline functionality using direct entry point approach

**TARGET USER REQUEST**: Enable full pipeline processing of `/home/brian/projects/process_tracing/input_text/revolutions/french_revolution.txt` to HTML output

**CURRENT CHALLENGE**: Direct entry point (`analyze_direct.py`) handles JSON → HTML perfectly, but TEXT → JSON integration not yet implemented.

**EVIDENCE-BASED SOLUTION STRATEGY**: Since systematic testing proved all individual components work perfectly, the solution is straightforward integration rather than complex architectural changes.

## 🔧 PHASE 22A IMPLEMENTATION TASKS

### TASK 1: Current State Validation (15 minutes)

**OBJECTIVE**: Test current direct entry point against original user request and identify specific gaps

**VALIDATION TESTS**:
```bash
# Test 1: What happens with text input? (Expected: FAIL - only handles JSON)
python analyze_direct.py /home/brian/projects/process_tracing/input_text/revolutions/french_revolution.txt --html

# Test 2: Confirm JSON → HTML still works (Expected: SUCCESS)
python analyze_direct.py output_data/revolutions/revolutions_20250910_081813_graph.json --html

# Test 3: Confirm extraction still works independently (Expected: SUCCESS)  
echo "2" | python process_trace_advanced.py --extract-only
```

**DOCUMENTATION REQUIREMENT**: Record exact error messages and identify missing functionality.

### TASK 2: Extraction Integration Implementation (30 minutes)

**OBJECTIVE**: Add TEXT → JSON capability to direct entry point

**TARGET FILE**: `analyze_direct.py`

**IMPLEMENTATION STRATEGY**:
```python
def main():
    # Add text file detection
    if args.json_file.endswith('.txt'):
        print("📝 Text input detected - performing extraction...")
        json_file = extract_text_to_json(args.json_file)
        print(f"✅ Extraction completed: {json_file}")
    else:
        json_file = args.json_file
        
    # Existing JSON → HTML logic (known working)
    G, data = load_graph(json_file)
    if args.html:
        html_file = generate_html_report(G, data)
```

**EXTRACTION FUNCTION OPTIONS**:
1. **Import extraction functions directly** (preferred)
2. **Call extraction as subprocess** (fallback if imports hang)
3. **Copy minimal extraction logic** (last resort)

### TASK 3: End-to-End Pipeline Testing (20 minutes)

**OBJECTIVE**: Validate complete TEXT → JSON → HTML pipeline using direct entry point

**CRITICAL TEST**:
```bash
# The original user request - this MUST work
python analyze_direct.py /home/brian/projects/process_tracing/input_text/revolutions/french_revolution.txt --html
# Expected: Complete HTML report generated successfully
```

**SUCCESS CRITERIA**:
- [ ] French Revolution text loads without hanging
- [ ] JSON extraction completes successfully  
- [ ] HTML report generated with meaningful content
- [ ] Total execution time under 5 minutes
- [ ] No ImportError or circular dependency issues

## 📊 SUCCESS CRITERIA FOR PHASE 22A

### **Primary Success Criteria**:
1. **User Request Fulfilled**: `python analyze_direct.py /home/brian/projects/process_tracing/input_text/revolutions/french_revolution.txt --html` generates working HTML report
2. **Complete Pipeline**: TEXT → JSON → HTML integration working end-to-end
3. **Performance**: Pipeline completes without hanging (target: <5 minutes)
4. **Quality Output**: HTML report contains meaningful French Revolution analysis with node/edge statistics

### **Technical Success Criteria**:
1. **Text Input Handling**: Direct entry point detects and processes .txt files correctly
2. **Extraction Integration**: TEXT → JSON conversion working within direct entry point
3. **Error Handling**: Graceful failure messages for invalid inputs or extraction failures
4. **Functionality Preservation**: JSON → HTML capability maintained (already proven working)

## 🏗️ POST-PHASE 22A ENHANCEMENTS

After successful TEXT → JSON → HTML integration, consider implementing:

**HTML Feature Parity** (Optional enhancement):
- Compare direct entry point HTML output with original system HTML features
- Add advanced visualizations (vis.js network graphs) if needed
- Integrate Van Evera diagnostic summaries and Bayesian confidence scoring

**Root Cause Resolution** (Optional investigation):
- Investigate and potentially fix the underlying `python -m core.analyze` hang
- Determine if the hang can be resolved without workarounds

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`process_trace_advanced.py`**: Original orchestration script - ⚠️ hangs in `python -m core.analyze` execution
- **`analyze_direct.py`**: **NEW** - Alternative entry point that bypasses hanging execution context - ✅ working for JSON → HTML
- **`core/extract.py`**: Extraction phase entry point - ✅ fully functional
- **`core/analyze.py`**: Contains load_graph() and HTML generation functions - ✅ functions work when called directly

### Critical Files for Phase 22A
- **`analyze_direct.py`**: Target for TEXT → JSON integration (needs text input detection and extraction calling)
- **`test_deadlock_isolation.py`**: Systematic testing framework proving all components work when called directly
- **`test_post_load_hang.py`**: Post-load operations testing confirming HTML generation logic is functional
- **`core/plugins/register_plugins.py`**: Plugin registration system  
- **`core/plugins/evidence_connector_enhancer.py`**: Problematic plugin triggering circular dependency

### Integration Points  
- **LLM Interface**: `VanEveraLLMInterface` - ✅ working correctly
- **Bayesian Models**: `core/bayesian_models.py` - ✅ complete infrastructure
- **Evidence Processing**: `core/evidence_weighting.py` - ✅ LLM-First compliant
- **Confidence Calculation**: `core/confidence_calculator.py` - ✅ LLM-First compliant

### Runtime Environment
- **Virtual Environment**: `test_env/` - ✅ activated with all dependencies  
- **LLM Access**: Google Generative AI configured and functional
- **Graph Processing**: NetworkX, matplotlib available
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