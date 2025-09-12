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

## 🎯 CURRENT STATUS: Phase 25E - Complete Systematic Hardcoded Edge Type Migration (Updated 2025-01-12)

**System Status**: **✅ CORE FUNCTIONAL - Systematic Analysis Required**  
**Latest Achievement**: **Phase 25E Partial - High-quality targeted fixes with honest assessment of limitations**  
**Current Priority**: **Complete systematic file-by-file analysis of all remaining hardcoded edge type patterns**

**PHASE 25E PARTIAL RESULTS** (Evidence-validated 2025-01-12):
- ✅ **Targeted High-Quality Fixes**: 2 most obvious hardcoded logic violations migrated successfully
  - `tools/migrate_ontology.py`: Dynamic ontology-based diagnostic type inference (lines 90-95)
  - `tests/ontology_test_helpers.py`: Pattern-based hypothesis testing edge selection (line 43)
- ✅ **System Health Maintained**: 22/22 OntologyManager tests passing throughout process
- ✅ **Validation Completed**: Import tests and functional validation for all changes
- ⚠️ **Limited Scope**: Only ~6 of 34 files systematically analyzed (spot-checking approach)
- ❌ **Incomplete Systematic Coverage**: Pattern-by-pattern analysis of all 158 instances still required

**HONEST ASSESSMENT** (Learning from overconfidence):
- **Core System**: Robust dynamic ontology architecture operational with targeted improvements
- **Migration Progress**: ~5-10% complete - only most obvious violations addressed
- **Remaining Work**: ~30+ files require complete systematic analysis with pattern-by-pattern classification
- **Quality Imperative**: Systematic coverage mandatory to avoid missed hardcoded logic patterns

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

## 🔧 PHASE 25E: Systematic Hardcoded Edge Type Pattern Migration

### OBJECTIVE: Complete systematic analysis and migration of all hardcoded edge type patterns

✅ **FOUNDATION ESTABLISHED**: 
- **Core System**: Robust dynamic ontology architecture operational (Phase 25C)
- **Partial Migration**: 2 problematic patterns successfully migrated (Phase 25D)
- **Analysis Framework**: Systematic discovery and classification methodology developed
- **System Health**: Complete functionality maintained throughout process

⚠️ **CRITICAL IMPERATIVE**: Previous analysis revealed overconfident claims - systematic verification essential

---

## 📋 PHASE 25E SYSTEMATIC COMPLETION TASKS

**CRITICAL REQUIREMENT**: Complete systematic analysis with rigorous evidence, learning from overconfidence failures in partial Phase 25E.

### 🔍 TASK 1: BASELINE DISCOVERY VERIFICATION (2-3 hours)
*Establish true, consistent baseline before any further work*

**OBJECTIVE**: Create verifiable baseline of all hardcoded edge type patterns with consistent methodology

```bash
# Create timestamped baseline directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_DIR="evidence/phase25e_baseline_${TIMESTAMP}"
mkdir -p $BASELINE_DIR

# Use identical grep patterns with documented methodology
grep -rn "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" \
  --include="*.py" . | grep -v test_env > $BASELINE_DIR/patterns_primary.txt

# Document methodology and baseline counts
echo "Search methodology:" > $BASELINE_DIR/methodology.md
echo "Date: $(date)" >> $BASELINE_DIR/methodology.md
echo "Files found: $(cat $BASELINE_DIR/patterns_primary.txt | cut -d: -f1 | sort -u | wc -l)" >> $BASELINE_DIR/methodology.md
echo "Total instances: $(cat $BASELINE_DIR/patterns_primary.txt | wc -l)" >> $BASELINE_DIR/methodology.md

# Document already-migrated patterns from Phase 25E partial
echo "Already migrated:" >> $BASELINE_DIR/methodology.md
echo "- tools/migrate_ontology.py: Dynamic diagnostic type inference" >> $BASELINE_DIR/methodology.md  
echo "- tests/ontology_test_helpers.py: Pattern-based edge selection" >> $BASELINE_DIR/methodology.md
```

**DELIVERABLE**: `$BASELINE_DIR/` with consistent methodology and true pattern counts

### 🔎 TASK 2: SYSTEMATIC FILE-BY-FILE ANALYSIS (6-8 hours)
*Complete analysis of every file, every pattern - no exceptions*

**OBJECTIVE**: Classify every pattern instance with documented reasoning and confidence levels

**Classification Framework**:
- **TEST_DATA**: Creates graph/edge for testing → PRESERVE  
- **SCHEMA_CONFIG**: Defines valid values → PRESERVE
- **HARDCODED_LOGIC**: Code behavior depends on string → MIGRATE
- **DYNAMIC_FALLBACK**: Uses after dynamic query → PRESERVE
- **DOCUMENTATION**: Comments/examples → PRESERVE
- **VALIDATION_LOGIC**: Checks against hardcoded list → MIGRATE

```bash
# Extract files requiring analysis and perform complete context analysis
cat $BASELINE_DIR/patterns_primary.txt | cut -d: -f1 | sort -u > $BASELINE_DIR/files_to_analyze.txt

# For each file, extract ALL patterns with full context
while read -r filepath; do
  grep -n -B5 -A5 "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" \
    "$filepath" > $BASELINE_DIR/context_${filepath//\//_}.txt
  pattern_count=$(grep -c "'supports'\|'tests_hypothesis'" "$filepath")
  echo "File: $filepath, Patterns: $pattern_count" >> $BASELINE_DIR/per_file_counts.txt
done < $BASELINE_DIR/files_to_analyze.txt

# Classification summary with confidence levels
echo "HIGH_CONFIDENCE_PRESERVE: [count]" > $BASELINE_DIR/classification_summary.md
echo "HIGH_CONFIDENCE_MIGRATE: [count]" >> $BASELINE_DIR/classification_summary.md  
echo "REVIEW_REQUIRED: [count]" >> $BASELINE_DIR/classification_summary.md
```

**DELIVERABLE**: Complete classification of every pattern with documented reasoning

### 🔧 TASK 3: SELECTIVE MIGRATION IMPLEMENTATION (3-4 hours)
*Migrate ONLY HIGH_CONFIDENCE_MIGRATE patterns with rigorous validation*

**OBJECTIVE**: Implement migrations for genuinely problematic patterns only

```bash
# For each HIGH_CONFIDENCE_MIGRATE pattern:
migrate_pattern() {
  local file=$1
  local description=$2
  
  # 1. Create backup
  cp "$file" "$file.pre_migration_backup"
  
  # 2. Document before state
  echo "BEFORE:" >> $BASELINE_DIR/migration_log.md
  grep -n -A3 -B3 [pattern] "$file" >> $BASELINE_DIR/migration_log.md
  
  # 3. Implement migration (specific to pattern)
  # [Edit command here based on pattern analysis]
  
  # 4. Document after state and validate
  echo "AFTER:" >> $BASELINE_DIR/migration_log.md
  grep -n -A3 -B3 [modified_area] "$file" >> $BASELINE_DIR/migration_log.md
  
  # 5. System integration test after each migration
  python -m pytest tests/test_ontology_manager.py -v  # Must stay 22/22
  python -c "from core.ontology_manager import ontology_manager; print('✅ Core system OK')"
}
```

**DELIVERABLE**: Each migration documented with before/after validation

### ✅ TASK 4: COMPREHENSIVE VALIDATION (2-3 hours)
*Multi-level validation to ensure zero regressions*

**OBJECTIVE**: Rigorous validation at file, system, and integration levels

```bash
# Core system health
python -m pytest tests/test_ontology_manager.py -v  # Must be 22/22

# Key integration points  
python -c "
from core.ontology_manager import ontology_manager
from tests.ontology_test_helpers import OntologyTestHelper
from tools.migrate_ontology import OntologyMigrator

print('Testing core integrations...')
edges = ontology_manager.get_evidence_hypothesis_edges()
print(f'Evidence-hypothesis edges: {len(edges)}')

helper = OntologyTestHelper()
supportive = helper.get_supportive_edges()  
print(f'Supportive edges: {len(supportive)}')

print('✅ All integrations working')
"

# Pattern count verification with consistent methodology
grep -rn "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" \
  --include="*.py" . | grep -v test_env > $BASELINE_DIR/patterns_final.txt

echo "PATTERN COUNT VERIFICATION:" > $BASELINE_DIR/final_verification.md
echo "Baseline patterns: $(cat $BASELINE_DIR/patterns_primary.txt | wc -l)" >> $BASELINE_DIR/final_verification.md
echo "Final patterns: $(cat $BASELINE_DIR/patterns_final.txt | wc -l)" >> $BASELINE_DIR/final_verification.md
```

**DELIVERABLE**: Comprehensive validation results with pattern verification

### 📝 TASK 5: HONEST EVIDENCE DOCUMENTATION (2 hours)
*Document exactly what was done with verifiable evidence*

**OBJECTIVE**: Create complete evidence package with honest assessment

```bash
# Create final evidence directory
EVIDENCE_DIR="evidence/current/Evidence_Phase25E_SystematicComplete_$(date +%Y%m%d)"
mkdir -p $EVIDENCE_DIR
cp -r $BASELINE_DIR/* $EVIDENCE_DIR/

# Create final assessment with:
# - Files systematically analyzed: [X/34] 
# - Patterns classified with high confidence: [X/Y]
# - Patterns migrated: [specific count with locations]
# - System health maintained: [test results]
# - Confidence levels: HIGH/MEDIUM/LOW for each category
# - Limitations acknowledged: [areas not covered, assumptions made]
```

**CRITICAL SUCCESS CRITERIA**:
✅ **Systematic Coverage**: All files analyzed with pattern-by-pattern classification  
✅ **Evidence Quality**: Every claim supported by raw command output  
✅ **Migration Precision**: Only genuinely problematic patterns migrated  
✅ **System Health**: 22/22 OntologyManager tests maintained  
✅ **Honest Assessment**: Confidence levels and limitations documented

---

## ⚠️ CRITICAL SUCCESS FACTORS

### **Methodology Requirements (Learning from Overconfidence)**:
1. **NO SHORTCUTS**: Every file gets complete analysis, no sampling or extrapolation
2. **CONSISTENT MEASUREMENT**: Use identical grep patterns and methodology throughout
3. **EVIDENCE FIRST**: Claims only after verifiable command execution, not speculation
4. **SYSTEMATIC TESTING**: Validation at every step prevents accumulating issues
5. **HONEST CONFIDENCE**: State limitations explicitly, acknowledge areas of uncertainty

### **Quality Gates**:
- **Pattern Analysis**: >90% of patterns classified with HIGH or MEDIUM confidence
- **Migration Quality**: 100% of migrations validated with import + functionality tests
- **System Integration**: Zero regressions in core system functionality  
- **Evidence Traceability**: 100% of claims traceable to raw command output
- **Methodology Consistency**: Same measurement approach for baseline and final verification

### **Failure Response Protocol**:
- **Any import failure**: Immediately restore from backup, analyze issue
- **Test regression**: Stop migration, identify root cause, fix before continuing
- **System integration failure**: Full rollback to last known good state
- **Performance degradation >20%**: Investigate and optimize before continuing

**ESTIMATED TIME**: 15-20 hours of systematic work over 3-4 focused sessions

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline entry point
- **`core/structured_extractor.py`**: LLM extraction with Pydantic validation
- **`core/analyze.py`**: Graph loading and analysis orchestration (✅ migrated to dynamic ontology)

### Critical Files Status (Phase 25E Partial)
- **`core/ontology_manager.py`**: ✅ COMPLETE - 22 passing tests, centralized dynamic ontology queries
- **`tools/migrate_ontology.py`**: ✅ MIGRATED (Phase 25E) - Dynamic ontology-based diagnostic type inference
- **`tests/ontology_test_helpers.py`**: ✅ MIGRATED (Phase 25E) - Pattern-based hypothesis testing edge selection
- **`core/disconnection_repair.py`**: ✅ MIGRATED (Phase 25D) - Dynamic ontology-based connection inference
- **`core/van_evera_testing_engine.py`**: ✅ MIGRATED (Phase 25D) - Dynamic evidence-hypothesis edge detection
- **`core/analyze.py`**: ✅ MIGRATED (Phase 25D) - Dynamic Evidence→Mechanism relationship analysis
- **`config/ontology_config.json`**: ✅ STABLE - Authoritative ontology definition
- **~31 remaining files**: ⚠️ REQUIRE SYSTEMATIC ANALYSIS - Classification needed for all patterns

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
- All implementation progress must be documented in `evidence/current/Evidence_Phase25E_*.md` files
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

**CURRENT PRACTICE**: Use structured evidence organization:

```
evidence/
├── current/
│   └── Evidence_Phase25E_[TaskName].md     # Current phase only
├── completed/  
│   └── Evidence_Phase25E_Complete.md       # Phase 25E partial work (archived)
```

**CRITICAL REQUIREMENTS**:
- Evidence files must contain ONLY current phase work (no historical contradictions)
- Raw execution logs required for all claims
- No success declarations without demonstrable proof
- Archive completed phases to avoid chronological confusion

---

## 🚨 IMMEDIATE NEXT STEPS FOR NEW LLM

### MANDATORY FIRST STEP: System Health Validation
```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# Validate core system (must show 22/22 tests passing)
python -m pytest tests/test_ontology_manager.py -v

# Record baseline state
python -c "
from core.ontology_manager import ontology_manager
from tests.ontology_test_helpers import OntologyTestHelper
print('✅ System health validated - proceed with systematic analysis')
"
```

### NEXT: Execute Systematic Phase 25E Tasks 1-5
Follow the 5 tasks above (🔍 TASK 1 through 📝 TASK 5) with complete systematic coverage.

**CRITICAL REMINDERS FROM LESSONS LEARNED**:
- 🚨 **NO SHORTCUTS**: Complete analysis of every file and pattern required
- 🚨 **EVIDENCE-BASED**: Every claim backed by raw command output
- 🚨 **HONEST ASSESSMENT**: State confidence levels and limitations explicitly
- 🚨 **SYSTEMATIC COVERAGE**: No spot-checking or extrapolation allowed

**CURRENT STATUS**: Phase 25E Partial - High-quality targeted fixes completed, systematic analysis of remaining ~31 files required.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.