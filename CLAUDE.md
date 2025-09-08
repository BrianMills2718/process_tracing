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

## 🎯 CURRENT STATUS: Pipeline Silent Hang Issues (Updated 2025-09-05)

**System Status**: **PIPELINE HANGING - MULTIPLE BLOCKING ISSUES**  
**Latest Achievement**: **Enum Validation Partially Fixed**  
**Current Priority**: **Identify and Fix Silent Hang Root Cause**

**RECENT PROGRESS:**
- ✅ **Enum Validation**: Fixed test_result enum in prompt template (supports/refutes → passed/failed/ambiguous)
- ✅ **Simple Tests**: Basic extraction works with correct enum values on small texts
- ✅ **Non-Interactive Mode**: Identified proper --project flag usage instead of interactive menu

**CURRENT BLOCKING ISSUES (PHASE 19B COMPLETE - NEW PRIORITY):**
- ✅ **Diagnostic Visibility**: RESOLVED - Full pipeline visibility with progress tracking
- ✅ **Silent Hangs**: RESOLVED - Not hanging, completing successfully in ~3-4 minutes  
- ✅ **Timeout Configuration**: RESOLVED - 30-minute timeouts configured
- ❌ **Schema Validation**: LLM generating 'supports'/'refutes' instead of 'passed'/'failed'/'ambiguous'

**INFRASTRUCTURE STATUS:**
- **Simple Cases**: ✅ Work correctly (test_simple, synthetic inputs)
- **Complex Cases**: ✅ LLM processing completes successfully (~3-4 minutes), ❌ Schema validation fails
- **Schema Compliance**: ❌ test_result enum still generating wrong values on complex cases
- **Error Reporting**: ✅ EXCELLENT - Full diagnostic visibility with timestamps and progress
- **Production Readiness**: 🔄 BLOCKED on schema validation fix (estimated 30-60 minutes)

---

## 🔧 PHASE 19B: Silent Hang Investigation (COMPLETE ✅) 

### OBJECTIVE: ✅ COMPLETED - Identified root cause of processing failures

**MAJOR DISCOVERY**: Pipeline was **NOT HANGING** - it was **COMPLETING SUCCESSFULLY** but failing on Pydantic validation.

**ROOT CAUSE IDENTIFIED**: 
- LLM calls complete successfully (~206 seconds for American Revolution)
- LLM generates valid JSON (28,449 characters) 
- **Schema validation fails**: LLM generates `'supports'/'refutes'` instead of required `'passed'/'failed'/'ambiguous'` for test_result field
- This causes silent failure appearance due to exception handling

**DIAGNOSTIC SUCCESS**: Comprehensive instrumentation added reveals complete pipeline flow visibility

---

## 🔧 PHASE 19C: Schema Validation Final Fix (CRITICAL PRIORITY - 30-60 minutes)

### OBJECTIVE: Fix final test_result enum validation to achieve 100% success

**RATIONALE**: Phase 19B revealed the pipeline works perfectly - LLM processing completes in ~3-4 minutes with valid JSON output. Only remaining issue is the test_result field still generating 'supports'/'refutes' instead of required enum values.

**STRATEGIC APPROACH**: Strengthen prompt engineering with explicit enum constraints and examples.

**EXPECTED IMPACT**: Schema validation fix → Complete American Revolution success → French Revolution testing

## 📋 PHASE 19C: Schema Validation Final Fix (30-60 minutes, CRITICAL priority)

### OBJECTIVE: Fix test_result enum validation to achieve pipeline success

**Target**: 100% Pydantic validation success on American Revolution, then test French Revolution  
**Scope**: Prompt engineering enhancement, enum specification strengthening

#### TASK 1: Enhanced Enum Specification (20 minutes)
**Purpose**: Strengthen prompt template to ensure correct enum generation

**Implementation Strategy**:
- Add explicit CRITICAL sections for test_result enum in prompt template
- Provide clear examples showing correct enum usage
- Add validation warnings and enum constraints
- Include JSON examples with proper test_result values

**Expected Impact**: 100% enum compliance for test_result field

#### TASK 2: American Revolution Validation (20 minutes)  
**Purpose**: Validate fix with American Revolution file

**Testing Strategy**:
- Run American Revolution with diagnostic instrumentation
- Verify schema validation success
- Confirm graph.json and HTML output generation
- Document complete success evidence

**Expected Impact**: Complete American Revolution processing pipeline

#### TASK 3: French Revolution Testing (20 minutes)
**Purpose**: Validate fix with original French Revolution target

**Testing Strategy**:
- Test French Revolution (52K chars) with working pipeline
- Monitor with diagnostic instrumentation
- Confirm complete processing chain
- Generate success evidence

**Expected Impact**: French Revolution analysis with full HTML output

### PHASE 19C VALIDATION CRITERIA
- **Schema Compliance**: 100% Pydantic validation success on American Revolution  
- **American Revolution Success**: Complete processing with graph.json and HTML output
- **French Revolution Success**: Complete processing of original 52K character target
- **Evidence Generation**: Proof of complete pipeline functionality

## 📋 PHASE 19B: Silent Hang Investigation (4-6 hours, CRITICAL priority)

### OBJECTIVE: Fix silent hang issues preventing complex text processing

**Target**: Enable successful French Revolution (52K chars) processing with full diagnostic visibility  
**Scope**: Diagnostic instrumentation, timeout configuration, hang point identification

#### TASK 1: Diagnostic Instrumentation (2 hours)
**Purpose**: Add comprehensive logging to identify where hangs occur

**Implementation Areas**:
- Add progress indicators to LLM calls ("Waiting for LLM response...")
- Log each pipeline phase with timestamps ("Starting extraction...", "Extraction complete")
- Add timeout detection with specific failure points
- Implement LLM response inspection and validation logging

**Expected Output**: Clear visibility into hang points and failure modes

#### TASK 2: Timeout Configuration (1 hour)
**Purpose**: Configure proper timeouts for large text processing

**Configuration Strategy**:
- Increase LiteLLM completion timeouts to 30+ minutes
- Add timeout parameters to all subprocess calls
- Configure proper error handling for timeout scenarios
- Test timeout behavior with diagnostic logging

**Expected Impact**: Proper timeout handling instead of silent hangs

#### TASK 3: Pipeline Testing with Diagnostics (2-3 hours)
**Purpose**: Validate fixes with French Revolution processing

**Testing Progression**:
- Level 1: test_phase18c (881 chars) with full diagnostics
- Level 2: French Revolution (52K chars) with monitoring
- Level 3: Full pipeline including HTML generation

**Expected Impact**: Complete processing chain working with transparency

### PHASE 19B VALIDATION CRITERIA
- **No Silent Hangs**: All failures provide clear diagnostic information
- **French Revolution Success**: Complete processing with graph.json and HTML output
- **Timeout Transparency**: Clear timeout messages instead of silent hangs
- **Diagnostic Visibility**: Progress tracking through all pipeline phases

## 📋 PHASE 19C: Documentation Architecture Investigation (2-3 hours)

### OBJECTIVE: Investigate and resolve documentation relevance and system architecture questions

**RATIONALE**: Multiple legacy documentation files exist with uncertain relevance to current system state. Need systematic investigation to determine what's implemented vs. planned.

**CRITICAL QUESTIONS TO INVESTIGATE**:

1. **Connectivity System Status**: 
   - Is graph connectivity issue resolved? 
   - Is `core/connectivity_analysis.py` fully functional?
   - Should `CONNECTIVITY_PLAN.md` be kept or archived?

2. **Diagnostic Rebalancer Implementation**:
   - Is Van Evera diagnostic rebalancing feature complete and working?
   - Are `core/plugins/diagnostic_rebalancer.py` files functional?
   - Should `DIAGNOSTIC_REBALANCER_IMPLEMENTATION.md` be kept?

3. **LLM-First Architecture Reality Check**:
   - **CONFLICT**: CLAUDE.md claims "100% GPT-5-mini routing" but 0 files found with `require_llm`/`LLMRequiredError`
   - Is the system actually LLM-first or using different patterns?
   - Should `MASTER_PLAN_100_PERCENT_LLM_FIRST.md` be archived?
   - Verify: Should be using structured output with LiteLLM + configurable model/provider/API key

4. **Legacy Validation Documents**:
   - Are `critical_assessment_report.md` and `debugging_disconnected_entities.md` still relevant?
   - Should they be moved to `docs/archive/`?

### INVESTIGATION SOURCES:
- Evidence files in `evidence/` directory for historical context
- Core implementation files for current state verification
- Plugin system for feature completeness assessment
- LLM routing configuration validation

### EXPECTED OUTCOMES:
- Clear documentation cleanup plan
- Accurate system architecture understanding  
- Resolved conflicts between claims and implementation
- Updated CLAUDE.md with verified system status

---

## ⚡ IMPLEMENTATION COMMANDS

### **PHASE 19A Start Command**:
```bash
# Begin schema validation fix
echo "9" | python process_trace_advanced.py  # Start with test_simple to validate fixes
```

### **Testing Progression**:
```bash
# Level 0: Minimal synthetic input (direct testing)
python -c "from core.structured_extractor import StructuredProcessTracingExtractor; 
extractor = StructuredProcessTracingExtractor(); 
result = extractor.extract_graph('Economic sanctions were imposed. Business leaders protested.')"

# Level 1: test_simple (215 characters)
echo "9" | python process_trace_advanced.py

# Level 2: test_extended (medium complexity)  
echo "4" | python process_trace_advanced.py

# Level 3: American Revolution (final validation)
echo "1" | python process_trace_advanced.py
```

### **Schema Investigation Commands**:
```bash
# Audit ProcessTracingGraph schema
python -c "from core.ontology import ProcessTracingGraph; print(ProcessTracingGraph.model_json_schema())"

# Check enum field definitions
grep -r "Literal\|Enum" core/ontology.py

# Find current prompt template
cat core/structured_extractor.py | grep -A50 "STRUCTURED_EXTRACTION_PROMPT"
```

### **Error Diagnosis Commands**:
```bash
# Test current extraction with detailed error
python -c "
from core.structured_extractor import StructuredProcessTracingExtractor
import traceback
try:
    extractor = StructuredProcessTracingExtractor()
    result = extractor.extract_graph('Economic sanctions were imposed.')
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
"
```

## 📊 PHASE 19A SUCCESS VALIDATION

### **EVIDENCE REQUIREMENTS**
1. **`evidence/current/Evidence_Phase19A_SchemaEnumFix.md`**: Schema audit, prompt fixes, and validation results
2. **Evidence of successful test progression**: Level 0 → 1 → 2 → 3 all passing
3. **American Revolution HTML output**: Complete analysis with network visualization

### **CRITICAL SUCCESS CRITERIA**
- ✅ **Schema Compliance**: 100% Pydantic validation success across all test levels
- ✅ **Output Generation**: Valid graph.json, analysis summary, and HTML files created
- ✅ **American Revolution Success**: Complete 27,930-character document processing
- ✅ **Error Transparency**: Any failures provide clear, actionable error messages

### **VALIDATION COMMANDS**
```bash
# Verify output files created
ls -la output_data/revolutions/ | head -5

# Validate HTML content
find output_data -name "*.html" -mmin -10 -exec wc -l {} \;

# Check analysis summary 
find output_data -name "*analysis_summary*.json" -mmin -10 -exec head -20 {} \;
```

---

## Project Overview

**Generalist LLM-Enhanced Process Tracing Toolkit** - Universal system implementing Van Evera academic methodology with sophisticated LLM semantic understanding for qualitative analysis using process tracing across any historical period or domain.

### Current Architecture Status
- **Pipeline Structure**: ✅ FUNCTIONAL - execute_single_case_processing restored with proper extraction/analysis flow
- **Error Handling**: ✅ ENHANCED - All silent failures now fail loudly with detailed error messages  
- **LLM Integration**: ✅ UNIFIED - 100% GPT-5-mini routing throughout entire pipeline
- **Schema Validation**: ❌ BLOCKING - Enum validation failures prevent successful completion
- **Van Evera Workflow**: ✅ OPERATIONAL - 8-step academic analysis pipeline functional
- **Plugin System**: ✅ ACTIVE - 16+ registered plugins with LLM-first compliance
- **Security**: ✅ CONFIGURED - Environment-based API key management (OpenAI/GPT-5-mini)
- **Universality**: ✅ MAINTAINED - No dataset-specific logic, works across all domains

## Testing Commands Reference

```bash
# Current pipeline test (enum validation will fail)
echo "9" | python process_trace_advanced.py  # test_simple

# Schema validation debugging
python -c "from core.structured_extractor import StructuredProcessTracingExtractor; 
extractor = StructuredProcessTracingExtractor(); 
extractor.extract_graph('Test text')"

# Router verification (should show GPT-5-mini)
python -c "from universal_llm_kit.universal_llm import get_llm; 
router = get_llm(); print(f'Model: {router.router.model_list}')"

# Error diagnosis for enum issues
python -c "
try:
    from core.structured_extractor import StructuredProcessTracingExtractor
    extractor = StructuredProcessTracingExtractor()
    result = extractor.extract_graph('Economic sanctions were imposed.')
    print('SUCCESS - No enum errors!')
except Exception as e:
    print(f'ENUM ERROR: {e}')
"

# Check for recent output files
find output_data -mmin -30 -type f | head -5
```

## Critical Success Factors

- **Evidence-First Approach**: All claims require concrete validation with command outputs
- **Schema-First Development**: 100% Pydantic validation required before claiming success
- **Incremental Testing**: Level 0 → 1 → 2 → 3 progression with validation at each step
- **Error Transparency**: All failures must provide clear, actionable error messages
- **Output Verification**: Success requires actual files (graph.json, analysis, HTML) created
- **No Silent Failures**: Every error must be caught and reported with troubleshooting guidance

**Current Priority**: Fix enum validation failures through systematic prompt engineering to achieve successful American Revolution processing with complete HTML analysis output.
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

      
      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.