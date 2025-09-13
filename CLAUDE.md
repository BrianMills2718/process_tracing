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

## 🎯 CURRENT STATUS: Phase 26B - Pipeline Hang Investigation & Fail-Fast Implementation (Updated 2025-01-12)

**System Status**: **⚠️ CRITICAL ISSUE IDENTIFIED - Pipeline Hanging After Ontology Changes**  
**Latest Achievement**: **Phase 26A Complete - Core ontology resilience validated, pipeline brittleness discovered**  
**Current Priority**: **Systematic investigation of pipeline hanging and fail-fast implementation**

**PHASE 26A COMPLETE RESULTS** (Evidence-validated 2025-01-12):
- ✅ **Excellent Core Resilience**: OntologyManager shows perfect fail-fast behavior (9/22 test failures with clear errors)
- ✅ **Dynamic Architecture Success**: System handles ontology additions gracefully (22/22 tests pass)  
- ✅ **Rollback Safety**: Commit 75a0f77 established as fallback point
- ❌ **Critical Pipeline Hang**: End-to-end pipeline hangs after LLM extraction phase with ANY ontology changes
- ❌ **Hang Persistence**: Issue persists even after ontology restoration, suggesting state corruption

**CURRENT CRITICAL ISSUE** (Blocking ontology resilience validation):
- **Pipeline Brittleness**: Hangs indefinitely after successful LLM extraction (161.07s, 37 nodes, 37 edges)
- **No Fail-Fast at Pipeline Level**: Should detect ontology issues and fail immediately with clear messages
- **State Corruption Suspected**: Hang persists even after ontology restoration
- **End-to-End Validation Blocked**: Cannot complete aggressive resilience testing

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

## 🔧 PHASE 26B: Pipeline Hang Investigation & Fail-Fast Implementation

### OBJECTIVE: Systematic investigation of pipeline hanging and implementation of robust fail-fast architecture

✅ **FOUNDATION ESTABLISHED**: 
- **Core Resilience Validated**: OntologyManager provides excellent fail-fast behavior with clear error messages
- **Dynamic Architecture Confirmed**: System handles ontology evolution gracefully at core level
- **Systematic Testing Approach**: 5-phase methodical plan developed for hang investigation
- **Safety Measures**: Rollback commit (75a0f77) established for aggressive testing

⚠️ **CRITICAL IMPERATIVE**: Pipeline hanging blocks complete ontology resilience validation - must be resolved systematically

---

## 📋 PHASE 26B: PIPELINE HANG INVESTIGATION & FAIL-FAST IMPLEMENTATION

**CRITICAL REQUIREMENT**: Systematic investigation mandatory to resolve pipeline hanging. NO shortcuts - must identify root cause systematically.

### 🔍 TASK 1: HANG LOCATION ISOLATION (3-4 hours)
*Pinpoint exact hang location in pipeline using systematic debugging*

**OBJECTIVE**: Identify exactly where in the pipeline the hang occurs

```bash
# CRITICAL: Phase 26A discovered pipeline hangs after successful LLM extraction
# Pattern: ✅ LLM extraction (161.07s, 37 nodes, 37 edges) → ❌ HANG in analysis phase

cd /home/brian/projects/process_tracing
source test_env/bin/activate

# 1. Test each pipeline stage independently
echo "=== COMPONENT ISOLATION TESTING ===" > evidence/hang_investigation.md

# Test extraction-only (bypass analysis)
python -c "
from core.structured_extractor import StructuredProcessTracingExtractor
extractor = StructuredProcessTracingExtractor()
print('✅ Extraction import OK')
" 2>&1 | tee -a evidence/hang_investigation.md

# Test graph loading (bypass analysis) 
python -c "
import json
from core.analyze import load_graph
print('✅ Load_graph import OK')
" 2>&1 | tee -a evidence/hang_investigation.md

# Test core imports
python -c "
from core.analyze import *
print('✅ Core.analyze imports OK')
" 2>&1 | tee -a evidence/hang_investigation.md

# 2. Incremental debug checkpoints
timeout 60 python analyze_direct.py input_text/revolutions/french_revolution.txt --debug-checkpoints 2>&1 | tee -a evidence/hang_investigation.md

# 3. Minimal graph testing
echo '{"nodes": [{"id": "test", "type": "Evidence"}], "edges": []}' > minimal_test.json
python -c "
from core.analyze import load_graph
G, data = load_graph('minimal_test.json')
print(f'✅ Minimal graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges')
" 2>&1 | tee -a evidence/hang_investigation.md
```

**DELIVERABLE**: `evidence/current/Evidence_Phase26B_HangLocation.md` with exact hang location identified

### 🧪 TASK 2: COMPONENT-BY-COMPONENT RESILIENCE TESTING (4-5 hours)
*Test each system component with modified ontologies to identify brittle components*

**OBJECTIVE**: Identify which specific components cause hanging or fail to adapt

```bash
# CRITICAL: Phase 26A showed OntologyManager excellent (fail-fast), pipeline hangs
# Focus: Find which component between extraction and final output causes hanging

echo "=== COMPONENT RESILIENCE TESTING ===" > evidence/component_testing.md

# 1. Test each critical component individually
# Van Evera Analysis
python -c "
from core.plugins.van_evera_testing_engine import VanEveraTesting
engine = VanEveraTesting()
print('✅ Van Evera engine created')
" 2>&1 | tee -a evidence/component_testing.md

# Plugin System
python -c "
import os
plugin_files = [f for f in os.listdir('core/plugins/') if f.endswith('.py') and not f.startswith('__')]
for plugin_file in plugin_files[:3]:  # Test first 3 plugins
    try:
        module_name = f'core.plugins.{plugin_file[:-3]}'
        __import__(module_name)
        print(f'✅ Plugin {plugin_file} imported')
    except Exception as e:
        print(f'❌ Plugin {plugin_file} failed: {e}')
" 2>&1 | tee -a evidence/component_testing.md

# Enhancement Components  
python -c "
try:
    from core.enhance_evidence import *
    print('✅ enhance_evidence imported')
except Exception as e:
    print(f'❌ enhance_evidence failed: {e}')
    
try:
    from core.enhance_mechanisms import *
    print('✅ enhance_mechanisms imported')
except Exception as e:
    print(f'❌ enhance_mechanisms failed: {e}')
" 2>&1 | tee -a evidence/component_testing.md

# 2. Test components with ontology modifications
cp config/ontology_config.json config/ontology_config.json.backup

# Add test edge type and retest components
echo "=== TESTING WITH MODIFIED ONTOLOGY ===" >> evidence/component_testing.md
# [Add new edge type to ontology]
# Retest all components above with modified ontology

# 3. Create systematic component failure matrix
echo "Component | Original Ontology | Modified Ontology | Notes" >> evidence/component_testing.md
echo "---------|-------------------|-------------------|-------" >> evidence/component_testing.md
```

**DELIVERABLE**: `evidence/current/Evidence_Phase26B_ComponentTesting.md` with component resilience matrix

### 🛠️ TASK 3: STATE MANAGEMENT & CORRUPTION INVESTIGATION (2-3 hours)
*Identify state corruption issues from ontology modifications*

**OBJECTIVE**: Determine if hanging is caused by cached state or module corruption

```bash
# CRITICAL: Phase 26A hang persists even after ontology restoration
# Hypothesis: State corruption from ontology modifications

echo "=== STATE CORRUPTION INVESTIGATION ===" > evidence/state_investigation.md

# 1. Fresh process vs reused process testing
python3 -c "
import sys
sys.path.insert(0, '.')
from core.ontology_manager import ontology_manager
print('Fresh process edge count:', len(ontology_manager.get_all_edge_types()))
" 2>&1 | tee -a evidence/state_investigation.md

# 2. Module import caching investigation
python3 -c "
import importlib
import core.ontology_manager
initial_edges = len(core.ontology_manager.ontology_manager.get_all_edge_types())
print(f'Initial edge count: {initial_edges}')

# Simulate ontology reload
core.ontology_manager = importlib.reload(core.ontology_manager)
new_edges = len(core.ontology_manager.ontology_manager.get_all_edge_types())
print(f'After reload: {new_edges}')
print(f'State preserved: {initial_edges == new_edges}')
" 2>&1 | tee -a evidence/state_investigation.md

# 3. Singleton and caching pattern detection
echo "=== CACHING PATTERN DETECTION ===" >> evidence/state_investigation.md
grep -r "class.*:" --include="*.py" core/ | grep -E "(Singleton|__new__|_instance)" >> evidence/state_investigation.md
grep -r "global.*ontology" --include="*.py" . >> evidence/state_investigation.md
grep -r "_cache.*ontology" --include="*.py" . >> evidence/state_investigation.md
grep -r "functools.lru_cache" --include="*.py" core/ >> evidence/state_investigation.md

# 4. Fresh virtual environment test
echo "=== FRESH ENVIRONMENT TEST ===" >> evidence/state_investigation.md
echo "Deactivating current environment and creating fresh test environment..."
# deactivate; rm -rf test_env_fresh; python -m venv test_env_fresh; source test_env_fresh/bin/activate
# Test in completely fresh environment
```

**DELIVERABLE**: `evidence/current/Evidence_Phase26B_StateCorruption.md` with state management analysis

### ⚡ TASK 4: FAIL-FAST ARCHITECTURE IMPLEMENTATION (5-6 hours)
*Implement robust fail-fast validation throughout pipeline*

**OBJECTIVE**: Transform pipeline hanging into clear fail-fast errors

```bash
# Based on findings from Tasks 1-3, implement fail-fast validation

echo "=== FAIL-FAST IMPLEMENTATION ===" > evidence/fail_fast_implementation.md

# 1. Pipeline entry validation in analyze_direct.py
echo "Adding ontology validation at pipeline entry point..." >> evidence/fail_fast_implementation.md

# Add to analyze_direct.py:
# def validate_system_ontology():
#     try:
#         from core.ontology_manager import ontology_manager
#         required_edges = ['tests_hypothesis', 'supports', 'provides_evidence_for']
#         missing = [e for e in required_edges if e not in ontology_manager.get_all_edge_types()]
#         if missing:
#             raise ValueError(f"❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: {missing}")
#         print(f"✅ Ontology validation passed: {len(ontology_manager.get_all_edge_types())} edge types")
#     except Exception as e:
#         print(f"❌ ONTOLOGY VALIDATION FAILED: {e}")
#         sys.exit(1)

# 2. Component-level fail-fast validation
echo "Adding component-level validation..." >> evidence/fail_fast_implementation.md

# Van Evera Analysis validation
# def validate_van_evera_ontology_compatibility():
#     required_edges = ['tests_hypothesis', 'supports', 'provides_evidence_for']
#     available_edges = ontology_manager.get_evidence_hypothesis_edges()
#     missing = [e for e in required_edges if e not in available_edges]
#     if missing:
#         raise ComponentValidationError(f"❌ Van Evera requires edge types: {missing}")

# 3. Graceful degradation for non-critical components
echo "Implementing graceful degradation..." >> evidence/fail_fast_implementation.md

# Test each implementation:
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/fail_fast_implementation.md
```

**DELIVERABLE**: `evidence/current/Evidence_Phase26B_FailFast.md` with fail-fast implementation results

### 📊 TASK 5: END-TO-END RESILIENCE VALIDATION (3-4 hours)
*Comprehensive validation across ontology modification scenarios with fail-fast behavior*

**OBJECTIVE**: Validate complete pipeline resilience after implementing fail-fast architecture

```bash
# Final comprehensive testing after implementing fail-fast from Task 4

EVIDENCE_DIR="evidence/current/Evidence_Phase26B_Complete_$(date +%Y%m%d)"
mkdir -p $EVIDENCE_DIR

echo "=== COMPREHENSIVE RESILIENCE VALIDATION ===" > $EVIDENCE_DIR/comprehensive_validation.md

# 1. Systematic ontology stress testing with new fail-fast behavior
for edge in "supports" "tests_hypothesis" "provides_evidence_for"; do
    echo "=== Testing removal of $edge ===" >> $EVIDENCE_DIR/comprehensive_validation.md
    cp config/ontology_config.json config/ontology_config.json.backup
    
    # Remove edge type from ontology (edit ontology_config.json)
    # Test pipeline behavior (should fail fast with clear message, NOT hang)
    timeout 60 python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 >> $EVIDENCE_DIR/comprehensive_validation.md
    
    # Restore ontology
    cp config/ontology_config.json.backup config/ontology_config.json
done

# 2. Recovery validation testing  
echo "=== RECOVERY TESTING ===" >> $EVIDENCE_DIR/comprehensive_validation.md
# Change ontology → test behavior → restore ontology → test recovery
# Validate no state corruption accumulates

# 3. Multi-input validation
echo "=== MULTI-INPUT VALIDATION ===" >> $EVIDENCE_DIR/comprehensive_validation.md
for input in input_text/*/*.txt; do
    echo "Testing resilience with: $input" >> $EVIDENCE_DIR/comprehensive_validation.md
    timeout 60 python analyze_direct.py "$input" 2>&1 >> $EVIDENCE_DIR/comprehensive_validation.md || echo "Pipeline completed (success or fail-fast)" >> $EVIDENCE_DIR/comprehensive_validation.md
done

# 4. Document final system state
python -m pytest tests/test_ontology_manager.py -v > $EVIDENCE_DIR/final_system_health.txt
python analyze_direct.py input_text/revolutions/french_revolution.txt > $EVIDENCE_DIR/final_pipeline_test.txt 2>&1

# Copy all investigation logs
cp evidence/hang_investigation.md $EVIDENCE_DIR/ 2>/dev/null || true
cp evidence/component_testing.md $EVIDENCE_DIR/ 2>/dev/null || true 
cp evidence/state_investigation.md $EVIDENCE_DIR/ 2>/dev/null || true
cp evidence/fail_fast_implementation.md $EVIDENCE_DIR/ 2>/dev/null || true
```

**CRITICAL SUCCESS CRITERIA**:
✅ **Hang Elimination**: Pipeline never hangs - either completes or fails fast with clear errors  
✅ **Root Cause Identified**: Exact cause of hanging documented with evidence  
✅ **Fail-Fast Implementation**: Clear, actionable error messages for ontology issues  
✅ **State Corruption Resolved**: No persistence of corrupted state between runs  
✅ **Component Resilience**: Each component handles ontology changes appropriately

---

## ⚠️ CRITICAL SUCCESS FACTORS

### **Methodology Requirements (Learning from Phase 26A)**:
1. **NO SHORTCUTS**: Systematic investigation mandatory - must identify exact hang location
2. **COMPONENT-BY-COMPONENT**: Test each system component individually to isolate failures
3. **EVIDENCE FIRST**: All hang analysis backed by timeout tests and stack traces
4. **FAIL-FAST IMPLEMENTATION**: Convert hangs to clear error messages with actionable guidance
5. **STATE CORRUPTION PREVENTION**: Eliminate cached state issues causing persistent problems

### **Quality Gates**:
- **Hang Location Identified**: Exact pipeline component causing hang must be found
- **Component Resilience**: Each component tested with ontology modifications
- **Fail-Fast Validation**: All ontology issues produce clear, immediate error messages
- **No Hanging Behavior**: System always terminates with definitive result (success or clear failure)
- **Recovery Capability**: System recovers properly from ontology issues without state corruption

### **Failure Response Protocol**:
- **Hanging continues**: Escalate to deeper debugging with stack traces and memory analysis
- **Component fails inappropriately**: Implement fail-fast validation or graceful degradation
- **State corruption detected**: Clear all caches and implement fresh state loading
- **Performance degradation**: Profile and optimize, but functionality takes priority over performance
- **Any timeout in testing**: Document exact timeout location and implement proper error handling

**ESTIMATED TIME**: 15-20 hours of systematic work over 3-4 focused sessions

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

**CURRENT STATUS**: Phase 26B Ready - Pipeline hang investigation required to resolve critical brittleness blocking ontology resilience validation.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.