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

## 🎯 CURRENT STATUS: System Operational with Mixed Routing (Updated 2025-01-05)

**System Status**: **OPERATIONAL WITH MIXED ROUTING**  
**Phase 17 Achievement**: **GPT-5-mini Parameter Integration SUCCESSFUL**
**Current Priority**: **Schema refinement and complete router unification**

**PHASE 17 COMPLETION SUMMARY:**
- ✅ **Extraction Phase**: GPT-5-mini working with router parameters (2-5s processing)
- ✅ **Analysis Phase**: Van Evera LLM interface operational (15-20s processing, using Gemini)
- ✅ **Parameter Integration**: Router parameters successfully integrated into StructuredExtractor
- ⚠️ **Schema Compliance**: ~85% validation success, minor property validation issues
- ⚠️ **Mixed Routing**: Extraction=GPT-5-mini, Analysis=Gemini (documented, functional)

**ROOT CAUSE RESOLUTION**:
- **Parameter Discovery**: Missing router parameters `use_in_pass_through=False`, `use_litellm_proxy=False`, `merge_reasoning_content_in_choices=False`
- **Integration Success**: GPT-5-mini now generates structured output consistently
- **Pipeline Status**: End-to-end processing functional, HTML generation capability restored

---

## 🔧 PHASE 18: Schema Refinement and Router Unification (NEXT PRIORITY)

### OBJECTIVE: Achieve 100% schema compliance and complete GPT-5-mini routing

**RATIONALE**: Phase 17 successfully resolved the critical infrastructure issues (GPT-5-mini connectivity). The system is now operational with mixed routing (GPT-5-mini for extraction, Gemini for analysis). Phase 18 focuses on refinement and complete unification.

**STRATEGIC APPROACH**: Schema-first refinement to achieve perfect validation, followed by complete router unification.

**EXPECTED IMPACT**: Mixed routing → Complete GPT-5-mini unification with 100% schema compliance

## 📋 PHASE 18A: Schema Validation Refinement (30-45 minutes, MEDIUM priority)

### OBJECTIVE: Achieve 100% Pydantic schema validation compliance
**Target**: Resolve minor validation errors preventing perfect pipeline execution
**Scope**: StructuredExtractor prompt engineering and schema alignment

#### TASK 1A: Schema Error Analysis and Resolution (20 minutes)
**Purpose**: Fix the ~15% validation errors identified in Phase 17C

**Current Validation Issues**:
```python
# Actor nodes missing description field (have name instead)
nodes.X.properties.description: Field required (Actor nodes using 'name')

# Array fields generated as strings
nodes.X.properties.key_predictions: Input should be array (generates string)

# Non-standard enum values
edges.X.properties.test_result: Should be 'passed'/'failed'/'ambiguous' (uses 'inconclusive', 'supports')
edges.X.properties.agency: Should be string (generates boolean)
```

**Required Fixes**:
1. Update prompt template to specify Actor nodes must have `description` field
2. Add explicit array formatting examples for `key_predictions` fields
3. Standardize enum values in prompt with exact allowed values
4. Add type specifications for boolean vs string properties

**Validation Command**:
```bash
python -c "
from core.structured_extractor import StructuredProcessTracingExtractor
result = extractor.extract_graph('Test input')
# Expected: Zero validation errors, 100% schema compliance
"
```

#### TASK 1B: Prompt Template Enhancement (15 minutes)
**Purpose**: Enhance schema clarity to prevent validation issues

**Enhancement Areas**:
- Property type specifications (string vs array vs boolean)
- Required field emphasis for all node types
- Enum value lists with exact allowed values
- Example JSON with perfect schema compliance

**Validation**: Multiple extraction tests with different input types to confirm consistent schema compliance

#### TASK 1C: Schema Compliance Testing (10 minutes)
**Purpose**: Comprehensive validation of schema refinement

**Test Cases**:
- Simple text (2-3 events)
- Complex text (multiple actors, alternatives)
- Edge case inputs (minimal text, complex temporal relationships)

**Success Criteria**: 100% schema validation across all test cases

### PHASE 18A VALIDATION CRITERIA
- **Zero Validation Errors**: All Pydantic schema validation passes
- **Consistent Field Types**: Arrays, strings, booleans generated correctly
- **Enum Compliance**: All enum values use standard allowed values
- **Required Field Presence**: All required fields present in generated nodes/edges

## 🔄 PHASE 18B: Complete Router Unification (45-60 minutes, HIGH priority)

### OBJECTIVE: Migrate analysis phase from Gemini to GPT-5-mini
**Target**: Eliminate mixed routing, achieve consistent GPT-5-mini throughout pipeline
**Approach**: Van Evera LLM interface router configuration updates

#### TASK 2A: Van Evera Interface Router Investigation (15 minutes)
**Purpose**: Understand why analysis phase routes to Gemini instead of GPT-5-mini

**Investigation Commands**:
```bash
# Check current Van Evera routing
python -c "
from core.plugins.van_evera_llm_interface import get_van_evera_llm
llm = get_van_evera_llm()
# Trace what router configuration is actually used
"

# Check UniversalLLM router priority
python -c "
from universal_llm_kit.universal_llm import get_llm
router = get_llm()
print(f'Router priority order: {router.router.model_list}')
"
```

**Expected Discovery**: Identify why "smart" model routes to Gemini in analysis phase despite router configuration showing GPT-5-mini

#### TASK 2B: Router Configuration Unification (20 minutes)
**Purpose**: Force all "smart" model calls to use GPT-5-mini consistently

**Potential Solutions**:
1. Update UniversalLLM router to prioritize GPT-5-mini more strongly
2. Modify Van Evera interface to specify model explicitly
3. Remove Gemini from router configuration entirely (force GPT-5-mini only)

**Configuration Update**:
```python
# In universal_llm_kit/universal_llm.py - ensure GPT-5-mini exclusivity
if os.getenv("OPENAI_API_KEY"):
    model_list = [
        {"model_name": "smart", "litellm_params": {"model": "gpt-5-mini", "max_completion_tokens": 16384}},
        {"model_name": "fast", "litellm_params": {"model": "gpt-5-mini", "max_completion_tokens": 16384}},
    ]
    # Remove or comment out Gemini configurations to force GPT-5-mini
```

#### TASK 2C: Unified Routing Validation (20 minutes)  
**Purpose**: Prove complete pipeline uses GPT-5-mini exclusively

**Test Commands**:
```bash
# Test extraction phase routing
python -c "from core.structured_extractor import StructuredProcessTracingExtractor; ..."
# Expected: GPT-5-mini logs

# Test analysis phase routing  
python -c "from core.plugins.van_evera_llm_interface import get_van_evera_llm; ..."
# Expected: GPT-5-mini logs, NO Gemini logs

# Full pipeline test with logging
python process_trace_advanced.py --project test_simple > pipeline_logs.txt 2>&1
grep -E "(gpt-5-mini|gemini)" pipeline_logs.txt
# Expected: Only GPT-5-mini references, zero Gemini references
```

### PHASE 18B VALIDATION CRITERIA
- **Unified Model Calls**: All LLM calls route to GPT-5-mini
- **Zero Mixed Routing**: No Gemini calls detected in pipeline logs
- **Performance Consistency**: Analysis phase maintains sub-20s performance with GPT-5-mini  
- **Function Preservation**: All Van Evera functionality working with GPT-5-mini routing

## 🧪 PHASE 18C: Complete Pipeline HTML Generation (30-45 minutes, VALIDATION)

### OBJECTIVE: Prove unified pipeline generates HTML successfully
**Target**: End-to-end HTML generation with 100% GPT-5-mini routing and perfect schema compliance
**Approach**: Full pipeline execution with comprehensive validation

#### TASK 3A: Schema-Compliant Pipeline Test (15 minutes)
**Purpose**: Test pipeline with refined schema to ensure no validation errors

**Test Execution**:
```bash
# Run complete pipeline with schema refinements
python process_trace_advanced.py --project test_simple_phase18

# Check for validation errors in output
find output_data/test_simple_phase18 -name "*.html" -exec echo "HTML generated: {}" \; -exec ls -la {} \;

# Validate unified routing in logs
grep -E "(gpt-5-mini|gemini)" pipeline_output.log
# Expected: Only gpt-5-mini, zero gemini
```

#### TASK 3B: HTML Quality Validation (15 minutes)
**Purpose**: Verify generated HTML contains complete process tracing analysis

**Quality Checks**:
- HTML file size >100KB (indicates comprehensive analysis)
- Van Evera diagnostic tests present in HTML
- Interactive visualizations rendering correctly
- All process tracing components (nodes, edges, analysis) included

#### TASK 3C: Performance Benchmarking (15 minutes)
**Purpose**: Document complete pipeline performance with unified GPT-5-mini routing

**Benchmark Metrics**:
- Extraction phase timing (target: <5s)
- Analysis phase timing (target: <30s with GPT-5-mini)
- Total pipeline timing (target: <60s)
- HTML generation size and completeness

### PHASE 18C VALIDATION CRITERIA
- **HTML Generated**: Physical HTML file created with complete analysis
- **Unified Routing**: 100% GPT-5-mini routing throughout pipeline
- **Schema Perfect**: Zero validation errors in any pipeline component
- **Performance Acceptable**: Complete pipeline under 60 seconds
- **Quality High**: HTML contains comprehensive Van Evera process tracing analysis

## 📊 COMPREHENSIVE SUCCESS VALIDATION

### FINAL EVIDENCE REQUIREMENTS

**Must Create Evidence Files**:
1. **`evidence/current/Evidence_Phase18A_SchemaRefinement.md`**: Schema validation fixes and testing results
2. **`evidence/current/Evidence_Phase18B_RouterUnification.md`**: Complete GPT-5-mini routing implementation
3. **`evidence/current/Evidence_Phase18C_HTMLGeneration.md`**: End-to-end HTML generation proof
4. **`evidence/current/Evidence_Phase18_Complete.md`**: Comprehensive phase 18 success summary

**Each Evidence File Must Include**:
- **Before/After Configurations**: Show exact changes made
- **Raw Command Outputs**: All validation test results with timestamps
- **Error Resolution**: Document how each issue was fixed
- **Performance Metrics**: Timing and consistency measurements
- **Success Artifacts**: Screenshots, file sizes, browser rendering proof
- **Reproducibility Tests**: Multiple successful runs documented

### CRITICAL SUCCESS CRITERIA (ALL MUST BE MET)

- ✅ **Perfect Schema Compliance**: 100% Pydantic validation success across all components
- ✅ **Unified GPT-5-mini Routing**: All LLM calls use GPT-5-mini (zero Gemini calls)
- ✅ **HTML Report Generated**: Physical HTML file exists and renders completely in browser
- ✅ **Pipeline Reliability**: Multiple successful runs without any errors
- ✅ **Performance Excellent**: End-to-end completion under 60 seconds
- ✅ **Complete Functionality**: All Van Evera process tracing features working

### FAILURE CONDITIONS (ANY MEANS INCOMPLETE)

- ❌ Any Pydantic validation errors in pipeline execution
- ❌ Mixed model calls (any Gemini references in logs)
- ❌ No HTML output generated or incomplete HTML
- ❌ Pipeline failures or hanging processes
- ❌ Performance degradation >60 seconds total
- ❌ Missing Van Evera functionality in final HTML

## ⚡ IMPLEMENTATION TIMELINE

### **Phase 18A: Schema Refinement** (30-45 minutes)
1. **Error Analysis** (20 min) → Fix Actor descriptions, array fields, enum values
2. **Prompt Enhancement** (15 min) → Add detailed type specifications and examples
3. **Compliance Testing** (10 min) → Validate 100% schema success

### **Phase 18B: Router Unification** (45-60 minutes)
1. **Routing Investigation** (15 min) → Understand Gemini vs GPT-5-mini routing
2. **Configuration Updates** (20 min) → Force exclusive GPT-5-mini routing
3. **Unified Validation** (20 min) → Prove complete GPT-5-mini pipeline

### **Phase 18C: HTML Generation** (30-45 minutes)
1. **Pipeline Testing** (15 min) → Schema-compliant full pipeline execution
2. **HTML Validation** (15 min) → Quality and completeness verification
3. **Performance Benchmarking** (15 min) → Document unified pipeline performance

**Total Estimated Time**: 1.75-2.5 hours for complete system unification

## 🎯 EVIDENCE-BASED DEVELOPMENT REQUIREMENTS

### Mandatory Validation Process
1. **No Claims Without Evidence**: Every success statement must be backed by command outputs
2. **Incremental Validation**: Test after each phase before proceeding
3. **Comprehensive Documentation**: All changes and results documented with timestamps
4. **Reproducible Results**: Multiple successful pipeline runs required for success claim
5. **Performance Monitoring**: Track timing and consistency across all phases

### Quality Gates
- **Schema Validation**: 100% compliance required before router unification
- **Unified Routing**: Zero mixed model calls allowed
- **HTML Generation**: Physical file with complete analysis required
- **Performance Standards**: Sub-60-second total pipeline execution

**Remember**: This is SYSTEM OPTIMIZATION after successful infrastructure repair. Phase 17 resolved the critical issues; Phase 18 achieves perfect unification and performance.

---

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Current Architecture Status
- **Plugin System**: 16+ registered plugins (100% LLM-first compliance achieved in Phase 13)
- **Van Evera Workflow**: 8-step academic analysis pipeline with enhanced intelligence  
- **LLM Integration**: OPERATIONAL with mixed routing (GPT-5-mini extraction, Gemini analysis)
- **Validation System**: validate_true_compliance.py for comprehensive compliance checking
- **Type Safety**: Full mypy compliance across core modules
- **Security**: Environment-based API key management (OpenAI configured)
- **Universality**: No dataset-specific logic - works across all domains and time periods

## Testing Commands Reference

```bash
# Schema validation testing
python -c "from core.structured_extractor import StructuredProcessTracingExtractor; extractor.extract_graph('test')"

# Router configuration verification  
python -c "from universal_llm_kit.universal_llm import get_llm; router = get_llm(); print(router.router.model_list)"

# Van Evera interface testing
python -c "from core.plugins.van_evera_llm_interface import get_van_evera_llm; llm.assess_probative_value(...)"

# Complete pipeline test
python process_trace_advanced.py --project test_simple

# HTML generation validation
find output_data -name "*.html" -mmin -10 -exec echo "HTML: {}" \; -exec wc -c {} \;
```

## Critical Success Factors

- **Evidence-First Approach**: All claims require concrete validation with command outputs
- **Systematic Problem Solving**: Complete analysis before implementing changes
- **Incremental Validation**: Test each component before integration
- **Documentation Discipline**: Every change documented with before/after evidence
- **Performance Monitoring**: Measure and validate timing across all pipeline stages
- **Quality Focus**: 100% schema compliance and unified routing as success criteria

**Current Priority**: The system is operational with mixed routing. Phase 18 focuses on achieving perfect unification (complete GPT-5-mini routing) and 100% schema compliance for production-ready quality.
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

      
      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.