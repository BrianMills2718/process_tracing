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

## 🎯 CURRENT STATUS: Phase 27A Complete - Ontology Evolution Support Added (Updated 2025-09-13)

**System Status**: **✅ EVOLUTION SUPPORT COMPLETE - Ontology changes now supported with --evolution-mode**  
**Latest Achievement**: **Phase 27A Complete - Added configurable ontology evolution support**  
**Current Priority**: **Ready for Phase 27B or future development**

**EVOLUTION CAPABILITIES DELIVERED**:
- ✅ **Perfect Protection**: System still prevents ontology corruption with clear error messages by default
- ✅ **Evolution Support**: --evolution-mode flag allows legitimate ontology updates/improvements
- ✅ **Backward Compatible**: Default behavior unchanged - existing workflows unaffected
- ✅ **Clear Warnings**: Evolution mode provides clear warnings when validation is bypassed
- ✅ **Fail-Fast Preserved**: System still fails fast on configuration problems in normal mode

**EVOLUTION MODE USAGE**: `python analyze_direct.py input.txt --evolution-mode` to bypass strict validation

---

## 🏗️ WHAT IS THE MIGRATION?

### **UNDERSTANDING THE HARDCODED EDGE TYPE MIGRATION**:

**THE PROBLEM**: System contained hardcoded edge type strings (like `'supports'`, `'tests_hypothesis'`) scattered throughout code. When ontology changes, developers had to manually update every hardcoded reference.

**THE SOLUTION**: Replace hardcoded strings with dynamic queries to centralized OntologyManager.

**CRITICAL DISTINCTION**: Not all patterns containing edge type strings should be migrated:
- **PROBLEMATIC**: `if edge_type in ['supports', 'refutes']:` (hardcoded logic)
- **APPROPRIATE**: `{'type': 'supports'}` (test data), `"supports"` (schema), `'supports'` (semantic patterns)

**MIGRATION EXAMPLE**:
```python
# BEFORE (Hardcoded Logic - PROBLEMATIC)
if edge_type in ['supports', 'tests_hypothesis', 'provides_evidence_for']:
    process_evidence_hypothesis_relationship()

# AFTER (Dynamic Query - CORRECT)  
from core.ontology_manager import ontology_manager
if edge_type in ontology_manager.get_evidence_hypothesis_edges():
    process_evidence_hypothesis_relationship()
```

---

## 🔧 PHASE 27A: ONTOLOGY VALIDATION EVOLUTION SUPPORT

### OBJECTIVE: Enable ontology evolution while preserving current robustness and fail-fast behavior

✅ **FOUNDATION FROM PHASE 26C**: 
- **Perfect Validation**: System detects configuration problems immediately with clear error messages
- **No Hangs**: All scenarios handled gracefully with fail-fast behavior
- **Robust Protection**: Users protected from ontology corruption

❌ **EVOLUTION BLOCKER IDENTIFIED**: Current validation prevents legitimate ontology improvements

---

## 📋 PHASE 27A: SYSTEMATIC ONTOLOGY EVOLUTION ENABLEMENT - ✅ COMPLETE

**COMPLETED 2025-09-13**: Successfully added configurable ontology evolution support with comprehensive testing.

### ✅ DELIVERABLES COMPLETED:

**1. Evolution Mode Implementation**:
- Added `--evolution-mode` command line flag to `analyze_direct.py`
- Modified `validate_system_ontology()` function to accept evolution_mode parameter
- Maintains backward compatibility - default behavior unchanged

**2. Validation Logic Updates**:
- **Location**: `analyze_direct.py:20-57` (validate_system_ontology function)
- **Hardcoded Edge Types**: `['tests_hypothesis', 'supports', 'provides_evidence_for']`  
- **Evolution Mode**: Bypasses strict validation with clear warning messages
- **Default Mode**: Preserves original fail-fast behavior

**3. Comprehensive Testing**:
- ✅ Edge type removal testing (supports, tests_hypothesis, provides_evidence_for)
- ✅ Backward compatibility validation (fails fast without --evolution-mode)
- ✅ Clear warning message verification in evolution mode
- ✅ System health check (22/22 tests passing)

**4. Evidence Documentation**:
- `evidence/current/Evidence_Phase27A_ValidationInvestigation.md`
- `evidence/current/Evidence_Phase27A_SolutionDesign.md` 
- `evidence/current/Evidence_Phase27A_Implementation.md`
- `evidence/current/Evidence_Phase27A_EvolutionTesting.md`
- `evidence/current/Evidence_Phase27A_FinalValidation.md`

### 🎯 SUCCESS CRITERIA - ALL ACHIEVED:
- ✅ **Evolution Support**: `--evolution-mode` allows ontology changes with warnings
- ✅ **Backward Compatibility**: Default behavior unchanged (existing workflows unaffected)
- ✅ **Clear Configuration**: Users understand when and how to use evolution mode
- ✅ **Robust Protection**: System still prevents accidental ontology corruption
- ✅ **Performance Maintained**: No degradation in normal operations

---

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline entry point
- **`core/structured_extractor.py`**: LLM extraction with Pydantic validation
- **`core/analyze.py`**: Graph loading and analysis orchestration (✅ migrated to dynamic ontology)

### Critical Files Status (Phase 26A Complete)
- **`core/ontology_manager.py`**: ✅ EXCELLENT - 22 passing tests, perfect fail-fast behavior for ontology changes
- **`config/ontology_config.json`**: ✅ STABLE - Authoritative ontology definition, handles additions gracefully
- **`analyze_direct.py`**: ❌ PIPELINE HANG - Hangs after LLM extraction phase with ontology changes
- **`core/analyze.py`**: ❌ SUSPECT - Graph loading/analysis phase appears to be hang location
- **`core/structured_extractor.py`**: ✅ WORKING - LLM extraction completes successfully even with modified ontology
- **`core/plugins/`**: ⚠️ UNKNOWN - Plugin system resilience to ontology changes untested
- **`core/enhance_*`**: ⚠️ UNKNOWN - Enhancement components resilience untested

### System Status (Phase 26A Findings)
- **Core Resilience**: ✅ EXCELLENT - OntologyManager provides perfect fail-fast behavior
- **Dynamic Architecture**: ✅ SUCCESS - System handles ontology additions gracefully (22/22 tests pass)  
- **Pipeline Resilience**: ❌ CRITICAL ISSUE - End-to-end pipeline hangs after LLM extraction
- **State Management**: ❌ CORRUPTION SUSPECTED - Hangs persist even after ontology restoration

### Runtime Environment
- **Virtual Environment**: `test_env/` activated with `source test_env/bin/activate`
- **Test Inputs**: Multiple validated datasets in `input_text/` (French Revolution, American Revolution, Westminster Debate)
- **Output Structure**: `output_data/direct_extraction/` contains rich HTML reports with network visualizations

---

## 📋 Coding Philosophy

### NO LAZY IMPLEMENTATIONS
- Implement complete, working code - no stubs or placeholders
- Test every component before declaring completion  
- Raw execution logs required for all claims

### FAIL-FAST PRINCIPLES
- Surface data integrity issues immediately
- No silent data loss tolerance
- Clear error reporting with actionable information

### EVIDENCE-BASED DEVELOPMENT
- All implementation progress must be documented in `evidence/current/Evidence_Phase26B_*.md` files
- Include raw timeout test results, component testing logs, and stack traces
- Document each fail-fast implementation with before/after behavior validation
- Validate all claims with command-line evidence and systematic debugging logs

### SYSTEMATIC VALIDATION
- Distinguish appropriate patterns (test data, documentation, semantic processing) from problematic patterns (hardcoded logic)
- Preserve system functionality as highest priority
- Test imports and functionality after each migration
- Maintain comprehensive system integration testing

---

## 📁 Evidence Structure

**CURRENT PRACTICE**: Use structured evidence organization:

```
evidence/
├── current/
│   ├── Evidence_Phase26B_HangLocation.md       # Task 1: Hang isolation investigation
│   ├── Evidence_Phase26B_ComponentTesting.md   # Task 2: Component resilience testing
│   ├── Evidence_Phase26B_StateCorruption.md    # Task 3: State corruption investigation
│   ├── Evidence_Phase26B_FailFast.md          # Task 4: Fail-fast implementation
│   └── Evidence_Phase26B_Complete.md          # Task 5: Final validation results
├── completed/  
│   └── Evidence_Phase26A_Complete.md          # Phase 26A results (archived)
```

**CRITICAL REQUIREMENTS**:
- Evidence files must contain systematic debugging approach with timeout tests
- Raw execution logs, stack traces, and component testing results required
- Exact hang location must be identified with demonstrable evidence
- Document all fail-fast implementations with before/after behavior validation

---

## 🚨 IMMEDIATE NEXT STEPS FOR NEW LLM

### MANDATORY FIRST STEP: Pipeline Hang Validation
```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# 1. Validate core system is still healthy (must show 22/22 tests passing)
python -m pytest tests/test_ontology_manager.py -v

# 2. CRITICAL: Attempt to reproduce pipeline hang issue
echo "Testing if pipeline hang issue persists..."
timeout 60 python analyze_direct.py input_text/revolutions/french_revolution.txt
# Expected result: Either completes successfully OR times out (indicating hang persists)

# 3. Document current system state
python -c "
from core.ontology_manager import ontology_manager
print(f'✅ OntologyManager healthy: {len(ontology_manager.get_all_edge_types())} edge types')
print('🔍 Ready to begin systematic hang investigation')
"
```

### NEXT: Execute Systematic Phase 26B Tasks 1-5
Follow the 5 tasks above (🔍 TASK 1 through 📊 TASK 5) with systematic hang investigation approach.

**CRITICAL REMINDERS FROM Phase 26A LESSONS LEARNED**:
- 🚨 **SYSTEMATIC HANG INVESTIGATION**: Must identify exact hang location, no assumptions allowed
- 🚨 **COMPONENT-BY-COMPONENT TESTING**: Test each system component individually with ontology changes  
- 🚨 **TIMEOUT-BASED VALIDATION**: Use timeouts to catch hangs, never wait indefinitely
- 🚨 **FAIL-FAST IMPLEMENTATION**: Convert all hangs to clear, actionable error messages
- 🚨 **STATE CORRUPTION AWARENESS**: Investigate module caching and singleton patterns causing persistent issues

## 🚀 READY FOR PHASE 27A: Ontology Evolution Support

**CURRENT STATUS**: **Ready for Phase 27A** - System has excellent validation but needs evolution flexibility.

### 📋 IMMEDIATE NEXT STEPS FOR NEW LLM:

**MANDATORY FIRST STEP**: Begin with Task 1 (Validation Logic Investigation)
```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# Start Phase 27A Task 1
mkdir -p evidence/current
echo "=== PHASE 27A TASK 1: VALIDATION LOGIC INVESTIGATION ===" > evidence/current/Evidence_Phase27A_ValidationInvestigation.md
```

**CRITICAL REQUIREMENTS**:
- Follow tasks in exact sequence (Task 1 → Task 2 → Task 3 → Task 4 → Task 5)
- Document all findings in evidence files with raw execution logs
- Test thoroughly before proceeding to next task
- Maintain backward compatibility throughout

### ✅ CURRENT SYSTEM HEALTH STATUS:
- **Ontology Manager**: 22/22 tests passing - EXCELLENT
- **Pipeline Validation**: Perfect fail-fast behavior - EXCELLENT BUT TOO STRICT
- **LLM Extraction**: Working reliably (164.54s, 41 nodes, 43 edges) - EXCELLENT
- **Analysis Phase**: Fast graph loading (0.00s) - EXCELLENT
- **Overall System**: Production-ready but needs evolution flexibility - GOOD

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.