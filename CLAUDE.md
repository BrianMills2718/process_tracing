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

## 🎯 CURRENT STATUS: Phase 25C - Complete Systematic Migration (Updated 2025-01-11)

**System Status**: **🔧 SYSTEMATIC MIGRATION - COMPREHENSIVE CLEANUP REQUIRED**  
**Latest Achievement**: **Phase 25B Partial Complete - OntologyManager + 3 critical files migrated**  
**Current Priority**: **Complete systematic migration of remaining 25+ files with hardcoded edge type patterns**

**PHASE 25B RESULTS** (Evidence-validated 2025-01-11):
- ✅ **OntologyManager Created**: Centralized dynamic ontology abstraction layer with 22 passing tests
- ✅ **Infrastructure Complete**: Migration tools, test helpers, systematic tracking created
- ✅ **Critical Files Started**: 3 core system files successfully migrated and validated
- ⚠️ **Incomplete Migration**: 16 hardcoded patterns remain in core/, 109 patterns total across codebase
- ⚠️ **Residual Cleanup**: "Previously migrated" files still contain hardcoded patterns requiring cleanup

**DISCOVERED REALITY** (Double-check validation):
- **Migration Status**: 11% complete (3 new files migrated, not 7+ as initially claimed)
- **Core Files**: 16 hardcoded patterns remain in critical system components
- **Total Patterns**: 109 hardcoded patterns across entire codebase (down from 120+)
- **Architecture Success**: Dynamic ontology system works correctly, foundation established

## 🔧 PHASE 25C: Complete Systematic Migration

### OBJECTIVE: Migrate ALL remaining hardcoded edge type references (25+ files with 109 patterns)

⚠️ **CRITICAL CONTEXT**: 
- **Phase 25B Foundation**: OntologyManager working, 3 files properly migrated, tools created
- **Reality Check**: Previous claims of "100% critical files migrated" were inaccurate
- **Current State**: 16 patterns in core/, 93 patterns in test/documentation files
- **Migration Approach**: Evidence-based systematic cleanup with comprehensive validation

## 📋 IMPLEMENTATION TASKS

⚠️ **EVIDENCE-BASED APPROACH REQUIRED**: All work must be validated with grep commands and documented in evidence files.

### PHASE 1: Comprehensive Audit (4 hours estimated)

**TASK 1A: Complete Pattern Discovery**
```bash
# Execute these commands to get current accurate state:
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env
grep -r "edge.*type.*in \[" --include="*.py" .
grep -r "edge\['type'\].*==" --include="*.py" .
```

**TASK 1B: Categorize by Risk Level**
- **P0 Critical**: Core execution path files (core/*.py main modules)
- **P1 High**: Plugin system files (core/plugins/*.py)
- **P2 Medium**: Test files that affect system validation
- **P3 Low**: Documentation and example files

**TASK 1C: Residual Cleanup Assessment**
Files marked as "already migrated" but containing patterns:
- `core/disconnection_repair.py`: 14 hardcoded patterns (semantic)
- `core/van_evera_testing_engine.py`: 1 residual pattern
- `core/analyze.py`: 1 residual pattern
- `core/streaming_html.py`: 1 residual pattern

**DELIVERABLE**: Updated `tools/migration_inventory.py` with accurate current state

### PHASE 2: Residual Cleanup in "Migrated" Files (3 hours estimated)

**OBJECTIVE**: Clean up hardcoded patterns remaining in files marked as "already migrated"

**APPROACH**: These files likely contain semantic patterns that are appropriate vs actual hardcoded edge types that need migration.

**TASK 2A: Analyze Residual Patterns**
```bash
# Check each "migrated" file for remaining patterns:
grep -n "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/disconnection_repair.py
grep -n "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/van_evera_testing_engine.py
grep -n "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/analyze.py
grep -n "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/streaming_html.py
```

**TASK 2B: Determine Action Needed**
- **Semantic patterns**: Language processing logic (KEEP)
- **Edge type logic**: Hardcoded edge type lists/checks (MIGRATE)
- **Comments/strings**: Documentation references (KEEP)
- **Test data**: Hardcoded test values (KEEP if test-specific)

**TASK 2C: Apply Targeted Migrations**
Only migrate actual hardcoded edge type logic, preserve appropriate semantic patterns.

### PHASE 3: Systematic New File Migration (6 hours estimated)

**OBJECTIVE**: Migrate files that have never been properly migrated to OntologyManager

**COMPLETED INFRASTRUCTURE**: 
- ✅ `tests/ontology_test_helpers.py` - Helper module for test files (created Phase 25B)
- ✅ `tools/migration_inventory.py` - Systematic tracking (created Phase 25B)
- ✅ Migration patterns established and tested

**TASK 3A: Priority-Based Migration Sequence**
1. **P0 Critical (Main Execution Path)**:
   - Files that directly affect core system functionality
   - Must be migrated first to ensure system stability

2. **P1 High (Analysis Modules)**:
   - Plugin system files
   - Analysis and processing modules

3. **P2 Medium (Test Files)**:
   - Use `tests/ontology_test_helpers.py` for systematic migration
   - Focus on tests that validate core functionality

4. **P3 Low (Documentation)**:
   - Example files and documentation
   - Can be migrated incrementally

**TASK 3B: Migration Execution Pattern**

**FOR EACH FILE**:
1. **Pre-Migration Validation**:
```bash
# Grep patterns in specific file
grep -n "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" FILENAME.py
```

2. **Apply Standard Migration Pattern**:
```python
# Add import
from core.ontology_manager import ontology_manager

# Replace hardcoded lists with dynamic calls
# BEFORE:
if edge_type in ['supports', 'tests_hypothesis']:
    process_edge()

# AFTER:
if edge_type in ontology_manager.get_evidence_hypothesis_edges():
    process_edge()
```

3. **Post-Migration Validation**:
```bash
# Verify patterns eliminated
grep -n "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" FILENAME.py
# Should return no matches (or only semantic patterns)

# Test functionality
python -c "import MODULE; print('Migration successful')"
```

**TASK 3C: Batch Processing**
Process files in priority order, validating each before proceeding to next.

### PHASE 4: Comprehensive Testing Framework (4 hours estimated)

**TASK 4A: Regression Test Suite**
```bash
# Core system functionality
python -m pytest tests/test_ontology_manager.py -v

# System integration test
python analyze_direct.py input_text/revolutions/french_revolution.txt

# Plugin system validation
python -m pytest tests/plugins/ -v

# Cross-domain testing
python -m pytest tests/test_cross_domain.py -v
```

**TASK 4B: Performance Validation**
```bash
# Benchmark processing time before/after
time python analyze_direct.py input_text/revolutions/french_revolution.txt
# Compare with baseline from Phase 25B
```

**TASK 4C: Pattern Elimination Verification**
```bash
# Final verification - should return 0 matches
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env | grep -v "# semantic pattern" | wc -l
```

### PHASE 5: Final Validation & Documentation (2 hours estimated)

**TASK 5A: System-Wide Validation**
```bash
# Complete system test across all inputs
for input_file in input_text/*/*.txt; do
    echo "Testing: $input_file"
    python analyze_direct.py "$input_file"
    if [ $? -ne 0 ]; then
        echo "FAILED: $input_file"
        exit 1
    fi
done
```

**TASK 5B: Final Evidence Documentation**
Create `evidence/current/Evidence_Phase25C_Complete.md` with:
- Total patterns eliminated (should be 109 → 0)
- All files migrated with before/after comparisons
- Complete test results
- Performance impact analysis
- Architecture improvements achieved

## 📊 SUCCESS CRITERIA FOR PHASE 25C

### **Implementation Success Metrics:**
1. **Zero Hardcoded Patterns**: `grep` commands return 0 matches for edge type patterns
2. **100% File Migration**: All 25+ identified files successfully migrated
3. **System Functionality**: All test inputs process successfully
4. **Performance Maintained**: <10% degradation from Phase 25B baseline
5. **Test Coverage**: All existing tests pass, OntologyManager tests maintain 22/22

### **Evidence Requirements:**
1. **Before/After Grep Counts**: Document pattern reduction (109 → 0)
2. **File-by-File Migration**: Every migration documented with test results
3. **System Integration**: Full pipeline tests with multiple inputs
4. **Performance Benchmarks**: Execution time comparisons

### **Evidence Documentation Structure:**
```
evidence/current/
├── Evidence_Phase25C_Audit.md           # Comprehensive audit results
├── Evidence_Phase25C_ResidualCleanup.md # Cleanup of "migrated" files
├── Evidence_Phase25C_SystematicMigration.md # New file migrations
├── Evidence_Phase25C_Testing.md         # Comprehensive test results
└── Evidence_Phase25C_Complete.md        # Final validation & summary
```

**REQUIRED EVIDENCE CONTENT**:
- **Grep Results**: Raw output showing pattern counts before/after
- **Migration Details**: Specific lines changed in each file
- **Test Validation**: Command executed + raw output for each migration
- **Performance Data**: Execution time comparisons
- **Integration Tests**: Full pipeline results with multiple inputs

---

## 🏗️ Codebase Structure

### Key Entry Points  
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline with basic HTML fallback
- **`core/structured_extractor.py`**: LLM extraction (Phase 23A: enhanced with raw response capture)
- **`core/analyze.py`**: Contains `load_graph()` (Phase 23A: fixed MultiDiGraph) + hanging `generate_html_report()`

### Critical Files Status (Phase 25C)
- **`core/ontology_manager.py`**: ✅ COMPLETE - 22 passing tests, fully functional
- **`tests/ontology_test_helpers.py`**: ✅ COMPLETE - Centralized test migration helper
- **`tools/migration_inventory.py`**: ✅ COMPLETE - Systematic migration tracking
- **25+ files with hardcoded patterns**: ⚠️ REQUIRES MIGRATION - 109 patterns remaining
- **`config/ontology_config.json`**: ✅ STABLE - Authoritative ontology definition

### Working Components (Phase 24A Investigation Complete)
- **Rich HTML Generation**: `core/html_generator.py` with interactive vis.js network visualizations
- **Van Evera Analytics**: Evidence-hypothesis analysis revealing ontology redundancies
- **Complete Pipeline**: TEXT → JSON → Rich HTML working end-to-end
- **Cross-Input Validation**: Multiple datasets tested with consistent results

### Runtime Environment
- **Virtual Environment**: `test_env/` activated with `source test_env/bin/activate`
- **Test Inputs**: Multiple validated inputs in `input_text/` (French Revolution, American Revolution, Westminster Debate)
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
- All implementation progress must be documented in `evidence/current/Evidence_Phase25C_*.md` files
- Include raw grep results showing pattern elimination
- Document each module migration with before/after analysis
- Validate all claims with command-line evidence

### SYSTEMATIC VALIDATION
- Run regression tests after each module migration
- Compare outputs before/after refactoring
- Benchmark performance to ensure no degradation

---

## 📁 Evidence Structure

⚠️ **PHASE 25C DOCUMENTATION REQUIREMENTS**

Evidence for Phase 25C must be documented in:
```
evidence/
├── current/
│   ├── Evidence_Phase25C_Audit.md           # Complete audit & pattern discovery
│   ├── Evidence_Phase25C_ResidualCleanup.md # "Migrated" file cleanup
│   ├── Evidence_Phase25C_SystematicMigration.md # New file migrations
│   ├── Evidence_Phase25C_Testing.md         # Comprehensive testing
│   └── Evidence_Phase25C_Complete.md        # Final validation
├── completed/
│   ├── Evidence_Phase25B_ValidationResults.md # Archived - partial completion
│   └── Evidence_Phase25A_Refactoring.md     # Archived
```

**EVIDENCE FILE REQUIREMENTS**:
- **RAW SEARCH RESULTS**: Include full grep outputs showing remaining hardcoded patterns
- **MIGRATION LOGS**: Before/after code snippets for every file changed
- **TEST OUTPUTS**: Complete test results after each migration
- **VALIDATION**: Proof that hardcoded patterns are eliminated

**CRITICAL WARNINGS**:
- ⚠️ **Do not claim 100% migration without grep validation**
- ⚠️ **Test every migration immediately - assume nothing works**
- ⚠️ **Include raw execution logs, not just success claims**
- ⚠️ **Validate that system still processes test inputs correctly**

---

## 🚨 IMMEDIATE NEXT STEPS FOR NEW LLM

### STEP 1: Validate Current System State (FIRST PRIORITY)
```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# Verify OntologyManager functionality
python -m pytest tests/test_ontology_manager.py -v
# Expected: 22/22 tests passing

# Verify system integration
python analyze_direct.py input_text/revolutions/french_revolution.txt
# Expected: Complete successfully without errors
```

### STEP 2: Execute Comprehensive Pattern Audit
```bash
# Get accurate current state - save ALL output to Evidence_Phase25C_Audit.md
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env
# Expected: ~109 matches across 25+ files

grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/ --include="*.py" 
# Expected: ~16 matches in core files
```

### STEP 3: Migration Execution Order
1. **PHASE 1**: Audit & inventory (identify all 109 patterns)
2. **PHASE 2**: Residual cleanup (fix "migrated" files with remaining patterns)
3. **PHASE 3**: Systematic migration (process all unmigrated files by priority)
4. **PHASE 4**: Comprehensive testing (validate all migrations)
5. **PHASE 5**: Final validation (achieve 0 hardcoded patterns)

### CURRENT INFRASTRUCTURE (Ready to Use):
- ✅ **OntologyManager**: `core/ontology_manager.py` - 22 passing tests, fully functional
- ✅ **Test Helpers**: `tests/ontology_test_helpers.py` - Migration support for test files
- ✅ **Migration Tracking**: `tools/migration_inventory.py` - Systematic progress tracking
- ✅ **System Validation**: Core pipeline processes inputs successfully

### CRITICAL UNDERSTANDING:
- **Previous claims of completion were inaccurate** - only 3 files properly migrated in Phase 25B
- **"Migrated" files still contain patterns** - require residual cleanup analysis
- **109 total patterns remain** - comprehensive migration still needed
- **System works correctly** - OntologyManager architecture is sound and functional

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.