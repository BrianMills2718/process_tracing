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

## 🎯 CURRENT STATUS: Phase 25D - Complete Optional Migration (Updated 2025-01-11)

**System Status**: **✅ CRITICAL PATH COMPLETE - Optional Cleanup Remaining**  
**Latest Achievement**: **Phase 25C Complete - Core system successfully migrated to dynamic ontology architecture**  
**Current Priority**: **Complete remaining test/documentation file migrations (non-critical for system operation)**

**PHASE 25C RESULTS** (Evidence-validated 2025-01-11):
- ✅ **Critical Path Migration**: 3 core system files fully migrated to OntologyManager dynamic queries
- ✅ **Architecture Transformation**: All inappropriate hardcoded edge type logic eliminated from critical execution path
- ✅ **System Validation**: Full pipeline functional (TEXT→JSON→HTML) with comprehensive testing (22/22 OntologyManager tests)
- ✅ **Pattern Cleanup**: Core patterns reduced 16 → 12 (100% of inappropriate hardcoded logic eliminated)
- ✅ **Evidence Documentation**: Complete validation with before/after comparisons and system integration testing

**ACCURATE STATUS ASSESSMENT** (Double-check validated):
- **Core System**: All critical hardcoded edge type dependencies eliminated - system now ontology-first
- **Remaining Work**: 104 patterns across 26 files (mostly appropriate test data, documentation, semantic processing)
- **System Health**: Complete pipeline operational with no regressions, enhanced with dynamic ontology capabilities
- **Architecture Foundation**: Robust dynamic ontology system ready for future ontology consolidation improvements

---

## 🏗️ WHAT IS THE MIGRATION?

### **UNDERSTANDING THE HARDCODED EDGE TYPE MIGRATION**:

**THE PROBLEM**: System contained hardcoded edge type strings (like `'supports'`, `'tests_hypothesis'`) scattered throughout code. When ontology changes, developers had to manually update every hardcoded reference.

**THE SOLUTION**: Replace hardcoded strings with dynamic queries to centralized OntologyManager.

**MIGRATION EXAMPLE**:
```python
# BEFORE (Hardcoded)
if edge_type in ['supports', 'tests_hypothesis', 'provides_evidence_for']:
    process_evidence_hypothesis_relationship()

# AFTER (Dynamic)  
if edge_type in ontology_manager.get_evidence_hypothesis_edges():
    process_evidence_hypothesis_relationship()
```

**RESULT**: Ontology changes now automatically propagate throughout system without code modifications.

---

## 🔧 PHASE 25D: Complete Optional Test & Documentation Migration

### OBJECTIVE: Migrate remaining test and documentation files (23 files with ~90 patterns)

✅ **CRITICAL PATH COMPLETE**: 
- **Core System**: All inappropriate hardcoded logic eliminated from critical execution path
- **System Validation**: Full functionality maintained with comprehensive testing (no regressions)
- **Architecture Foundation**: Robust OntologyManager integration providing dynamic ontology queries
- **Pattern Status**: Only appropriate test data, documentation examples, and semantic processing patterns remain

⚠️ **REMAINING SCOPE**: Optional cleanup of test files and documentation for consistency (non-critical)

---

## 📋 IMPLEMENTATION TASKS

✅ **CRITICAL PATH COMPLETE**: Core system migration accomplished. Remaining work is optional consistency improvements.

### PHASE 1: Final Scope Assessment (2 hours estimated)

**TASK 1A: Categorize Remaining Patterns**
```bash
# Current state: 104 patterns across 26 files
find . -name "*.py" -exec grep -l "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" {} \; | grep -v test_env | sort

# For each file, classify patterns as:
# - APPROPRIATE: Test data, documentation, semantic processing (DO NOT MIGRATE)
# - PROBLEMATIC: Hardcoded logic requiring migration (MIGRATE)
```

**KNOWN APPROPRIATE PATTERNS** (Do not migrate):
- `tests/test_ontology_manager.py`: Test data validating OntologyManager functionality (34 patterns)
- `tools/migrate_ontology.py`: Migration mapping data (8 patterns) 
- `core/ontology_manager.py`: Documentation example in docstring (1 pattern)
- `core/plugins/content_based_diagnostic_classifier.py`: Semantic language processing pattern (1 pattern)
- `core/disconnection_repair.py`: Semantic patterns and dynamic system fallbacks (appropriate in migrated system)

**TASK 1B: Create Accurate Migration Inventory**
```python
# Update tools/migration_inventory.py to focus only on truly problematic patterns
# Exclude appropriate test data, documentation, and semantic processing patterns
# Focus on files containing hardcoded logic that should use dynamic ontology queries
```

**DELIVERABLE**: `evidence/current/Evidence_Phase25D_FinalScope.md` with exact problematic patterns requiring migration

### PHASE 2: Test File Migration (3 hours estimated)  

**OBJECTIVE**: Migrate test files containing problematic hardcoded logic (not test data)

**INFRASTRUCTURE READY**:
- ✅ `tests/ontology_test_helpers.py`: Helper functions for systematic test file migrations
- ✅ Migration patterns established and tested in Phase 25C
- ✅ Validation framework operational with comprehensive testing

**TASK 2A: Identify Test Logic vs Test Data**
```bash
# Distinguish between:
# - Test data: Hardcoded values used to test system functionality (KEEP)
# - Test logic: Code that makes decisions based on hardcoded edge types (MIGRATE)

# Example of test logic requiring migration:
if edge_type in ['supports', 'tests_hypothesis']:  # This logic should be dynamic
    validate_evidence_hypothesis_relationship()
```

**TARGET FILES** (Containing problematic test logic):
- Test files where code logic depends on hardcoded edge type lists
- Files where hardcoded checks affect test validation logic
- Plugin test files with hardcoded edge type validation

**TASK 2B: Systematic Migration with Test Helper**
```python
# Use ontology_test_helpers.py for consistent test migrations
from tests.ontology_test_helpers import OntologyTestHelper

# BEFORE (in test logic)
if edge_type in ['supports', 'tests_hypothesis']:
    assert_evidence_relationship()

# AFTER (in test logic)  
if edge_type in OntologyTestHelper.get_evidence_hypothesis_edges():
    assert_evidence_relationship()
```

**TASK 2C: Validation After Each Migration**
```bash
# Test functionality after each file migration
python -m pytest MIGRATED_FILE.py -v  
python -c "import MIGRATED_MODULE; print('Import successful')"
```

### PHASE 3: Documentation File Migration (2 hours estimated)

**OBJECTIVE**: Update documentation files containing hardcoded examples that should demonstrate dynamic patterns

**LOW IMPACT SCOPE**: Documentation files where hardcoded examples could mislead developers

**TARGET FILES**:
- `docs/testing/*.py`: Example files demonstrating hardcoded patterns (should show dynamic patterns)
- Documentation where hardcoded examples don't reflect current architecture

**TASK 3A: Documentation Pattern Updates**
```python
# Update examples to demonstrate current dynamic architecture
# BEFORE (in documentation example):
example_edges = ['supports', 'tests_hypothesis']  # Misleading hardcoded example

# AFTER (in documentation example):
from core.ontology_manager import ontology_manager
example_edges = ontology_manager.get_evidence_hypothesis_edges()  # Shows current architecture
```

**TASK 3B: Selective Update Strategy**
- Update examples that should demonstrate dynamic ontology usage
- Preserve appropriate test data and semantic processing patterns  
- Focus on files where hardcoded examples contradict current architecture

### PHASE 4: Final Validation Framework (2 hours estimated)

**TASK 4A: Comprehensive Pattern Analysis**
```bash
# Verify remaining patterns are all appropriate
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env

# Cross-reference against approved appropriate patterns:
# - Test data in test_ontology_manager.py (validates OntologyManager functionality)
# - Migration tool data in migrate_ontology.py (migration mapping data)
# - Documentation examples in ontology_manager.py (docstring examples)
# - Semantic processing in content_based_diagnostic_classifier.py (language analysis)
# - Dynamic system fallbacks in disconnection_repair.py (appropriate in migrated architecture)
```

**TASK 4B: System Integration Validation**
```bash
# Ensure full system functionality maintained throughout optional cleanup
python -m pytest tests/test_ontology_manager.py -v  # Should show 22/22 passing
python analyze_direct.py input_text/revolutions/french_revolution.txt  # Should complete successfully

# Multi-dataset validation
for input_file in input_text/*/*.txt; do
    echo "Testing: $input_file"
    python analyze_direct.py "$input_file"
    [ $? -eq 0 ] || echo "FAILED: $input_file"
done

# Performance validation - should show no significant degradation
time python analyze_direct.py input_text/revolutions/french_revolution.txt
```

**TASK 4C: Migration Quality Assessment**
- All migrated files import successfully without errors
- No new test failures introduced by migrations
- System functionality fully maintained
- Pattern count matches expected appropriate patterns only

### PHASE 5: Final Documentation (1 hour estimated)

**TASK 5A: System Health Confirmation**
```bash
# Comprehensive system validation across all test datasets
for input_file in input_text/*/*.txt; do
    echo "Processing: $input_file"  
    python analyze_direct.py "$input_file"
    [ $? -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED"
done
# Expected: All datasets process successfully with no regressions
```

**TASK 5B: Final Documentation Update**
Create `evidence/current/Evidence_Phase25D_Complete.md` with:
- **Final Pattern Analysis**: Classification of remaining patterns (appropriate vs eliminated)
- **Migration Results**: Documentation of all migrations with validation results
- **System Functionality**: Confirmation of continued full pipeline operation
- **Architecture Summary**: Transformation from hardcoded → dynamic ontology system
- **Future Maintenance**: Guidance for adding new edge types (should automatically propagate)

---

## 📊 SUCCESS CRITERIA FOR PHASE 25D

### **Implementation Success Metrics:**
1. **Pattern Classification Complete**: Clear documentation distinguishing appropriate vs problematic patterns
2. **Problematic Logic Eliminated**: All remaining hardcoded edge type logic replaced with dynamic queries
3. **System Functionality Maintained**: All test inputs continue to process successfully (no regressions)
4. **Test Coverage Preserved**: All existing tests pass, no new test failures introduced
5. **Migration Quality Validated**: All migrated files import correctly and function as expected

### **Evidence Requirements:**
1. **Pattern Analysis**: Comprehensive classification of all 104 remaining patterns with justification
2. **Migration Documentation**: Each migration validated with before/after testing
3. **System Integration**: Continued full pipeline functionality across all test datasets  
4. **Final Assessment**: Complete analysis of remaining patterns with clear rationale

### **Quality Standards:**
- **Evidence-Based**: All claims supported by grep validation and test results
- **System-First**: Functionality preservation prioritized over pattern count reduction
- **Surgical Precision**: Only migrate patterns that represent problematic hardcoded logic
- **Architecture Consistency**: All migrations follow established OntologyManager patterns

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline entry point
- **`core/structured_extractor.py`**: LLM extraction with Pydantic validation
- **`core/analyze.py`**: Graph loading and analysis orchestration (✅ migrated to dynamic ontology)

### Critical Files Status (Phase 25D)
- **`core/ontology_manager.py`**: ✅ COMPLETE - 22 passing tests, centralized dynamic ontology queries
- **`core/disconnection_repair.py`**: ✅ MIGRATED - Dynamic ontology-based connection inference
- **`core/van_evera_testing_engine.py`**: ✅ MIGRATED - Dynamic evidence-hypothesis edge detection
- **`core/analyze.py`**: ✅ MIGRATED - Dynamic Evidence→Mechanism relationship analysis
- **`tests/ontology_test_helpers.py`**: ✅ READY - Helper functions for systematic test file migrations
- **`tools/migration_inventory.py`**: ✅ OPERATIONAL - Migration tracking and progress monitoring
- **26 remaining files**: ⚠️ OPTIONAL CLEANUP - Mostly appropriate test data and documentation patterns
- **`config/ontology_config.json`**: ✅ STABLE - Authoritative ontology definition

### Working Components (Fully Functional)
- **Rich HTML Generation**: Complete TEXT → JSON → HTML pipeline with interactive visualizations
- **Van Evera Analytics**: Evidence-hypothesis analysis using dynamic ontology queries
- **OntologyManager Integration**: Centralized dynamic edge type queries throughout critical system
- **Cross-Input Validation**: Multiple datasets tested with consistent results (no regressions)

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
- All implementation progress must be documented in `evidence/current/Evidence_Phase25D_*.md` files
- Include raw grep results showing pattern classification and elimination
- Document each migration with before/after testing validation
- Validate all claims with command-line evidence

### SYSTEMATIC VALIDATION
- Distinguish appropriate patterns (test data, documentation, semantic processing) from problematic patterns (hardcoded logic)
- Preserve system functionality as highest priority
- Test imports and functionality after each migration
- Maintain comprehensive system integration testing

---

## 📁 Evidence Structure

⚠️ **PHASE 25D DOCUMENTATION REQUIREMENTS**

Evidence for Phase 25D must be documented in:
```
evidence/
├── current/
│   ├── Evidence_Phase25D_FinalScope.md      # Pattern classification and migration targets
│   ├── Evidence_Phase25D_TestMigration.md   # Test file migration results
│   ├── Evidence_Phase25D_DocMigration.md    # Documentation file updates
│   ├── Evidence_Phase25D_Validation.md     # System validation results
│   └── Evidence_Phase25D_Complete.md       # Final comprehensive assessment
├── completed/
│   ├── Evidence_Phase25C_Complete.md       # Archived - Critical path migration complete
│   ├── Evidence_Phase25C_ResidualCleanup.md # Archived - Core file migrations
│   └── Evidence_Phase25C_Audit.md          # Archived - Initial pattern discovery
```

**EVIDENCE FILE REQUIREMENTS**:
- **Pattern Classification**: Clear distinction between appropriate vs problematic patterns
- **Raw Grep Results**: Before/after pattern counts with specific pattern analysis
- **Migration Validation**: Import tests and functionality validation for each migrated file
- **System Integration**: Full pipeline tests demonstrating continued functionality
- **Justification Documentation**: Clear rationale for patterns left unmigrated (appropriate test data, etc.)

**CRITICAL GUIDANCE**:
- ⚠️ **Distinguish appropriate vs problematic patterns** - Not all patterns need migration
- ⚠️ **Preserve test data and semantic processing** - Only migrate hardcoded logic
- ⚠️ **Validate every change with grep and testing** - Evidence-based development required  
- ⚠️ **System functionality is paramount** - Maintain complete pipeline operation

---

## 🚨 IMMEDIATE NEXT STEPS FOR NEW LLM

### STEP 1: Validate System Health (FIRST PRIORITY)
```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# Verify core system remains fully functional after Phase 25C
python -m pytest tests/test_ontology_manager.py -v
# Expected: 22/22 tests passing

# Verify complete pipeline operation
python analyze_direct.py input_text/revolutions/french_revolution.txt
# Expected: Complete TEXT→JSON→HTML generation successfully

# Check system integration across multiple datasets
for input in input_text/revolutions/*.txt; do
    echo "Testing: $input"
    python analyze_direct.py "$input" && echo "✅ SUCCESS" || echo "❌ FAILED"
done
# Expected: All test inputs process successfully with no regressions
```

### STEP 2: Assess Remaining Migration Scope  
```bash
# Current pattern count (should be ~104 patterns across 26 files)
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env | wc -l

# Core patterns (should be ~12 patterns, all appropriate fallbacks/semantic/documentation)
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/ --include="*.py" | wc -l

# List all files containing patterns for classification
find . -name "*.py" -exec grep -l "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" {} \; | grep -v test_env | sort
```

### STEP 3: Pattern Classification Strategy
1. **IDENTIFY APPROPRIATE PATTERNS**: 
   - Test data validating OntologyManager functionality (test_ontology_manager.py)
   - Migration tool mapping data (migrate_ontology.py)  
   - Documentation examples (docstrings, comments)
   - Semantic language processing (content_based_diagnostic_classifier.py)
   - Dynamic system fallbacks (appropriate in migrated architecture)

2. **IDENTIFY PROBLEMATIC PATTERNS**: 
   - Hardcoded logic where code behavior depends on specific edge type strings
   - Validation loops checking against hardcoded edge type lists
   - Assignment logic using hardcoded edge types

3. **CREATE FOCUSED MIGRATION PLAN**: Target only patterns representing hardcoded logic requiring dynamic conversion

4. **VALIDATE SYSTEM CONTINUOUSLY**: Ensure complete functionality maintained throughout optional cleanup process

### CURRENT INFRASTRUCTURE (Fully Operational):
- ✅ **Core System Architecture**: Complete migration to dynamic ontology queries (critical path complete)
- ✅ **OntologyManager**: Fully functional with 22/22 passing tests - handles all dynamic ontology operations
- ✅ **Migration Tools**: `tests/ontology_test_helpers.py` ready for systematic test file migrations
- ✅ **System Validation**: Complete pipeline operational with no regressions across multiple test datasets
- ✅ **Architecture Foundation**: Robust dynamic ontology system ready for future ontology improvements

### CRITICAL UNDERSTANDING:
- **✅ Core system migration COMPLETE**: All inappropriate hardcoded edge type logic eliminated from critical execution path
- **✅ System fully operational**: Complete TEXT→JSON→HTML pipeline with enhanced dynamic ontology capabilities
- **⚠️ Remaining work is OPTIONAL**: Test file and documentation cleanup for consistency (non-critical)
- **✅ Architecture transformation successful**: System now ontology-first with centralized dynamic queries
- **✅ Future-ready foundation**: Ontology changes automatically propagate without code modifications

---

**PHASE 25D FOCUS**: Complete optional test and documentation file cleanup while maintaining the fully functional dynamic ontology architecture achieved in Phase 25C.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.