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

## 🎯 CURRENT STATUS: Phase 25E - Systematic Hardcoded Edge Type Migration (Updated 2025-01-12)

**System Status**: **✅ CORE FUNCTIONAL - Systematic Migration Required**  
**Latest Achievement**: **Phase 25D Partial - 2 problematic patterns migrated, overconfident initial claims corrected**  
**Current Priority**: **Complete systematic analysis and migration of remaining hardcoded edge type patterns**

**PHASE 25D RESULTS** (Evidence-validated 2025-01-12):
- ✅ **Systematic Analysis Initiated**: Comprehensive pattern discovery and classification methodology developed
- ✅ **Problematic Patterns Migrated**: 2 hardcoded logic patterns converted to dynamic ontology queries
- ✅ **Analysis Lessons**: Corrected overconfident initial claims through systematic verification
- ✅ **System Health Maintained**: Complete pipeline functional with 22/22 OntologyManager tests passing
- ⚠️ **Incomplete Coverage**: Spot-checking revealed need for comprehensive systematic analysis

**CURRENT STATUS ASSESSMENT** (Honest evaluation):
- **Core System**: Robust dynamic ontology architecture operational (Phase 25C achievement preserved)
- **Migration Progress**: ~95% complete based on partial analysis, but systematic verification required
- **Remaining Work**: ~20+ patterns across 25 files require systematic classification and selective migration
- **Quality Imperative**: Previous overconfident claims highlight need for rigorous systematic approach

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

## 📋 IMPLEMENTATION TASKS

**CRITICAL REQUIREMENT**: Systematic approach mandatory after Phase 25D lessons learned. NO shortcuts or confidence without verification.

### PHASE 1: COMPREHENSIVE PATTERN DISCOVERY (2-3 hours)

**TASK 1A: Multi-Strategy Pattern Discovery**
```bash
# Execute comprehensive search using multiple strategies
# Primary patterns
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env > patterns_primary.txt

# Quoted variations
grep -r '"supports"\|"tests_hypothesis"\|"provides_evidence_for"\|"updates_probability"\|"weighs_evidence"' --include="*.py" . | grep -v test_env > patterns_quoted.txt

# Logic patterns (conditional/assignment)
grep -r "== 'supports'\|in \[.*'supports.*'\]\|type.*'supports'" --include="*.py" . | grep -v test_env > patterns_logic.txt

# Combine and deduplicate file list
cat patterns_*.txt | cut -d: -f1 | sort -u > files_containing_patterns.txt

echo "Files found: $(wc -l < files_containing_patterns.txt)"
echo "Total pattern instances: $(cat patterns_*.txt | wc -l)"
```

**TASK 1B: File-by-File Context Analysis**
```bash
# For each file, document complete context
while read -r file; do
    echo "=== ANALYZING: $file ===" >> complete_analysis.md
    echo "Pattern count: $(grep -c "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" "$file")" >> complete_analysis.md
    echo "" >> complete_analysis.md
    grep -n -B3 -A3 "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" "$file" >> complete_analysis.md
    echo "" >> complete_analysis.md
done < files_containing_patterns.txt
```

**DELIVERABLE**: `evidence/current/Evidence_Phase25E_Discovery.md` with complete pattern inventory and context analysis

### PHASE 2: SYSTEMATIC PATTERN CLASSIFICATION (3-4 hours)

**OBJECTIVE**: Classify every pattern instance using rigorous criteria - no assumptions allowed

**CLASSIFICATION FRAMEWORK**:

| Pattern Type | Criteria | Action | Example |
|--------------|----------|--------|---------|
| **TEST_DATA** | Creates graph/edge for testing | PRESERVE | `{'type': 'supports'}` |
| **SCHEMA_CONFIG** | Defines valid values | PRESERVE | `"supports"` in enum |
| **HARDCODED_LOGIC** | Code behavior depends on string | MIGRATE | `if edge_type == 'supports':` |
| **DYNAMIC_FALLBACK** | Uses after dynamic query | PRESERVE | `edges[0] if edges else 'supports'` |
| **DOCUMENTATION** | Comments/docstring examples | PRESERVE | `# Example: 'supports'` |
| **TEMPLATE_DATA** | LLM extraction templates | PRESERVE | Template for LLM |
| **VALIDATION_LOGIC** | Checks against hardcoded list | MIGRATE | `if type in ['supports']:` |

**TASK 2A: Pattern-by-Pattern Classification**
```bash
# For each pattern instance, document:
# 1. File location and line number
# 2. Surrounding context (5 lines before/after)
# 3. Classification using framework above
# 4. Confidence level (HIGH/MEDIUM/LOW)
# 5. Action required (PRESERVE/MIGRATE/REVIEW)

# Create systematic classification log
echo "# Pattern Classification Log" > pattern_classification.md
echo "" >> pattern_classification.md

# Process each pattern with full context documentation
```

**TASK 2B: High-Confidence vs Review-Required**
```bash
# Separate patterns into confidence buckets:
# HIGH_CONFIDENCE_PRESERVE: Obvious test data, schema definitions
# HIGH_CONFIDENCE_MIGRATE: Clear hardcoded logic patterns  
# REQUIRES_REVIEW: Ambiguous cases needing deeper analysis

# Only proceed with HIGH_CONFIDENCE patterns initially
```

### PHASE 3: INCREMENTAL MIGRATION IMPLEMENTATION (2-4 hours)

**OBJECTIVE**: Migrate HIGH_CONFIDENCE_MIGRATE patterns using systematic procedure

**MIGRATION ORDER** (Risk-based approach):
1. **LOW-RISK**: Documentation files (`docs/testing/*.py`)
2. **MEDIUM-RISK**: Test logic files (test assertions)
3. **HIGH-RISK**: Core system files (if any found)

**TASK 3A: Systematic Migration Procedure**
```bash
# For each file requiring migration:
file="path/to/file.py"

# 1. Create backup
cp "$file" "${file}.backup"

# 2. Document current state
grep -n "'supports'\|'tests_hypothesis'" "$file" > "${file}.before.txt"

# 3. Perform migration (add import, replace logic)
# Example migration pattern:
# BEFORE: if edge_type in ['supports', 'refutes']:
# AFTER: 
#   from core.ontology_manager import ontology_manager
#   evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
#   if edge_type in evidence_hypothesis_edges:

# 4. Document changes
grep -n "'supports'\|'tests_hypothesis'" "$file" > "${file}.after.txt"
diff "${file}.before.txt" "${file}.after.txt" > "${file}.changes.txt"

# 5. Validate import
python -c "import $(echo $file | sed 's/\//./g' | sed 's/.py//'); print('✅ Import OK')" || echo "❌ Import FAILED"

# 6. Run tests if applicable
if [[ $file == tests/* ]]; then
    python -m pytest "$file" -v
fi
```

**TASK 3B: Migration Validation Requirements**
```bash
# After each migration, verify:
# 1. File imports successfully
# 2. No syntax errors introduced  
# 3. Relevant tests still pass
# 4. Pattern count reduced appropriately
# 5. System integration maintained
```

### PHASE 4: COMPREHENSIVE SYSTEM VALIDATION (2-3 hours)

**OBJECTIVE**: Rigorous validation at multiple levels to ensure migration quality

**TASK 4A: File-Level Validation**
```bash
# After each migration, immediately verify:
# 1. Import test
python -c "import MODULE; print('✅ Import OK')" || echo "❌ Import FAILED"

# 2. Syntax validation  
python -m py_compile "$file" || echo "❌ Syntax ERROR"

# 3. Individual tests (if test file)
if [[ $file == tests/* ]]; then
    python -m pytest "$file" -v
fi
```

**TASK 4B: System Integration Validation**
```bash
# After every 3-5 migrations, verify system health:
# 1. Core system test
python -m pytest tests/test_ontology_manager.py -v  # Must show 22/22 passing

# 2. Pipeline functionality test
python analyze_direct.py input_text/revolutions/french_revolution.txt

# 3. Multi-dataset validation
for input_file in input_text/*/*.txt; do
    echo "Testing: $input_file"
    python analyze_direct.py "$input_file" > /dev/null 2>&1
    [ $? -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED: $input_file"
done

# 4. Performance benchmark
time python analyze_direct.py input_text/revolutions/french_revolution.txt > /dev/null 2>&1
```

**TASK 4C: Pattern Count Verification**
```bash
# Verify migration impact:
echo "Patterns before migration: [document baseline]"
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env | wc -l
echo "Patterns after migration: [should be reduced]"

# Verify no hardcoded logic patterns remain:
grep -r "if.*type.*in.*\[.*'supports'" --include="*.py" . | grep -v test_env
# Should return only appropriate patterns
```

### PHASE 5: COMPREHENSIVE EVIDENCE DOCUMENTATION (2-3 hours)

**OBJECTIVE**: Document complete migration process with rigorous evidence

**TASK 5A: Phase-by-Phase Evidence Documentation**
```markdown
# Required Evidence Files (all in evidence/current/):

Evidence_Phase25E_Discovery.md
- Complete pattern inventory with raw grep results
- File-by-file context analysis  
- Pattern count documentation with verification commands

Evidence_Phase25E_Classification.md
- Systematic classification of every pattern instance
- Classification framework application with reasoning
- High-confidence vs review-required separation

Evidence_Phase25E_Migration.md
- Each migration documented with before/after diffs
- Import and test validation results
- System integration testing results

Evidence_Phase25E_Validation.md  
- Comprehensive system validation across all levels
- Performance benchmarks and regression testing
- Final pattern count verification

Evidence_Phase25E_Complete.md
- Final honest assessment with lessons learned
- Complete migration statistics with evidence
- Future maintenance guidance
```

**TASK 5B: Evidence Quality Requirements**
```bash
# Each claim must be supported by:
# 1. Raw command output (grep results, test results)
# 2. Before/after comparisons where applicable
# 3. System functionality validation commands
# 4. Specific pattern counts and file locations  
# 5. Error logs if any issues encountered

# NO SUCCESS CLAIMS WITHOUT DEMONSTRABLE PROOF
```

---

## 📊 SUCCESS CRITERIA FOR PHASE 25E

### **Critical Success Requirements:**
1. **Complete Systematic Analysis**: Every pattern instance classified with documented reasoning
2. **Verified Pattern Classification**: No assumptions - every classification backed by context analysis
3. **Surgical Migration Precision**: Only genuinely problematic patterns migrated
4. **Zero Regressions**: All system functionality maintained throughout process
5. **Rigorous Evidence Documentation**: Every claim supported by raw execution logs

### **Quality Validation Metrics:**
1. **Discovery Completeness**: Multi-strategy search captures all pattern variations
2. **Classification Accuracy**: Clear criteria applied consistently to every instance
3. **Migration Quality**: Each migration validated with import/test/integration verification
4. **System Health**: 22/22 OntologyManager tests + multi-dataset pipeline validation
5. **Evidence Quality**: Raw logs, before/after comparisons, specific pattern counts

### **Failure Criteria (Immediate Stop and Reassess):**
- **OntologyManager tests fail**: Core system integrity compromised
- **Pipeline fails on known-good input**: Critical functionality broken
- **More than 2 import errors**: Migration quality insufficient
- **Classification confidence below 90%**: Analysis not systematic enough
- **Performance degradation >20%**: Unacceptable system impact

### **Quality Standards (Non-Negotiable):**
- **Evidence-Based**: Every claim requires raw command output verification
- **Systematic Coverage**: No spot-checking - complete analysis mandatory
- **Honest Assessment**: Confidence levels must reflect actual verification depth
- **Preservation Priority**: System functionality over pattern count reduction

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

⚠️ **PHASE 25E DOCUMENTATION REQUIREMENTS**

Evidence for Phase 25E must be documented with COMPLETE SYSTEMATIC COVERAGE:
```
evidence/
├── current/
│   ├── Evidence_Phase25E_Discovery.md        # Complete pattern inventory & multi-strategy search
│   ├── Evidence_Phase25E_Classification.md   # Systematic classification of EVERY pattern
│   ├── Evidence_Phase25E_Migration.md        # Incremental migration with validation
│   ├── Evidence_Phase25E_Validation.md       # Comprehensive system testing
│   └── Evidence_Phase25E_Complete.md         # Honest final assessment
├── completed/
│   ├── Evidence_Phase25D_Complete.md         # Archived - Partial migration with lessons learned
│   ├── Evidence_Phase25D_*.md                # Archived - Previous phase evidence
│   └── Evidence_Phase25C_*.md                # Archived - Core system migration
```

**EVIDENCE FILE REQUIREMENTS** (Mandatory - No Exceptions):
- **Complete Discovery Documentation**: Raw output from multi-strategy searches
- **Systematic Classification**: EVERY pattern instance classified with full context
- **Pattern-by-Pattern Analysis**: Context analysis (5 lines before/after) for each instance
- **Migration Documentation**: Before/after diffs, import validation, test results
- **System Health Validation**: Raw test output, pipeline validation, performance benchmarks
- **Honest Assessment**: Confidence levels, limitations, areas requiring further review

**CRITICAL REQUIREMENTS** (Learned from Phase 25D):
- 🚨 **NO OVERCONFIDENT CLAIMS** - State confidence levels explicitly
- 🚨 **SYSTEMATIC COVERAGE MANDATORY** - No spot-checking allowed
- 🚨 **RAW EVIDENCE REQUIRED** - Every claim backed by command output
- 🚨 **PRESERVE SYSTEM FUNCTION** - Continuous validation throughout process
- 🚨 **DOCUMENT UNCERTAINTIES** - Acknowledge limitations and review-required items

---

## 🚨 IMMEDIATE NEXT STEPS FOR NEW LLM

### STEP 1: System Health Baseline Validation (MANDATORY FIRST STEP)
```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# 1. Core system validation
python -m pytest tests/test_ontology_manager.py -v
# REQUIREMENT: Must show 22/22 tests passing

# 2. Pipeline functionality validation
python analyze_direct.py input_text/revolutions/french_revolution.txt
# REQUIREMENT: Must complete TEXT→JSON→HTML successfully

# 3. Multi-dataset system integration
for input in input_text/*/*.txt; do
    echo "Processing: $input"
    python analyze_direct.py "$input" > /dev/null 2>&1 && echo "✅ SUCCESS" || echo "❌ FAILED"
done
# REQUIREMENT: ALL inputs must process successfully

# 4. Performance baseline
time python analyze_direct.py input_text/revolutions/french_revolution.txt > /dev/null 2>&1
# DOCUMENT: Record baseline performance for regression detection
```

### STEP 2: Execute PHASE 1 - Comprehensive Pattern Discovery (MANDATORY)
```bash
# SYSTEMATIC DISCOVERY - NO SHORTCUTS ALLOWED

# 1. Multi-strategy pattern search
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env > patterns_primary.txt
grep -r '"supports"\|"tests_hypothesis"\|"provides_evidence_for"\|"updates_probability"\|"weighs_evidence"' --include="*.py" . | grep -v test_env > patterns_quoted.txt
grep -r "== 'supports'\|in \[.*'supports.*'\]\|type.*'supports'" --include="*.py" . | grep -v test_env > patterns_logic.txt

# 2. Document comprehensive baseline
cat patterns_*.txt | cut -d: -f1 | sort -u > files_containing_patterns.txt
echo "Files found: $(wc -l < files_containing_patterns.txt)"
echo "Total patterns: $(cat patterns_*.txt | wc -l)"

# 3. Begin Evidence_Phase25E_Discovery.md documentation
# REQUIREMENT: Document ALL findings with raw output
```

### STEP 3: Execute PHASE 2 - Systematic Pattern Classification (MANDATORY)

**REQUIREMENTS**:
1. **SYSTEMATIC APPROACH**: Use classification framework - NO assumptions allowed
2. **COMPLETE COVERAGE**: Every pattern instance must be classified
3. **EVIDENCE-BASED**: Every classification backed by context analysis
4. **CONFIDENCE LEVELS**: Document HIGH/MEDIUM/LOW confidence for each classification

**CLASSIFICATION WORKFLOW**:
```bash
# For each file containing patterns:
while read -r file; do
    echo "=== ANALYZING: $file ===" >> classification_analysis.md
    echo "Pattern count: $(grep -c "'supports'\|'tests_hypothesis'" "$file")" >> classification_analysis.md
    echo "Context analysis:" >> classification_analysis.md
    grep -n -B3 -A3 "'supports'\|'tests_hypothesis'" "$file" >> classification_analysis.md
    echo "Classification: [TO BE DETERMINED]" >> classification_analysis.md
    echo "" >> classification_analysis.md
done < files_containing_patterns.txt

# REQUIREMENT: Complete Evidence_Phase25E_Classification.md
```

### PHASE 25E REQUIREMENTS (Based on Phase 25D Lessons):
- 🚨 **SYSTEMATIC APPROACH MANDATORY**: No spot-checking - complete analysis required
- 🚨 **EVIDENCE-BASED CLAIMS ONLY**: Every assertion backed by raw command output
- 🚨 **HONEST CONFIDENCE LEVELS**: State limitations and uncertainties explicitly
- 🚨 **PRESERVE SYSTEM HEALTH**: Continuous validation throughout migration process
- 🚨 **RIGOROUS DOCUMENTATION**: Complete evidence files for every phase

### CURRENT INFRASTRUCTURE STATUS:
- ✅ **Core System**: Robust dynamic ontology architecture operational (Phase 25C achievement)
- ✅ **OntologyManager**: 22/22 tests passing - centralized dynamic query system
- ✅ **Pipeline**: Complete TEXT→JSON→HTML functionality across all test datasets
- ⚠️ **Migration Status**: Partial completion (2 patterns migrated in Phase 25D)
- 📝 **Analysis Quality**: Previous overconfident claims corrected - systematic approach required

### CRITICAL LESSONS FROM PHASE 25D:
- **Overconfidence Risk**: Initial "99% accuracy" claims were incorrect
- **Spot-Checking Insufficient**: Missed patterns due to incomplete systematic analysis
- **Verification Essential**: Claims must be backed by complete systematic verification
- **System Priority**: Maintain functionality while achieving migration precision

---

**PHASE 25E MISSION**: Complete systematic hardcoded edge type migration with rigorous evidence-based approach, learning from Phase 25D analysis failures.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.