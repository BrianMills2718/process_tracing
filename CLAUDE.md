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

## 🎯 CURRENT STATUS: Phase 26A - Aggressive Ontology Resilience Testing (Updated 2025-01-12)

**System Status**: **✅ EXCELLENT - Ontology Resilience Validation Required**  
**Latest Achievement**: **Phase 25E Complete - Comprehensive systematic analysis completed successfully**  
**Current Priority**: **Validate system resilience to ontology changes through aggressive end-to-end testing**

**PHASE 25E COMPLETE RESULTS** (Evidence-validated 2025-01-12):
- ✅ **Complete Systematic Analysis**: All 24 files and 97 hardcoded edge type patterns analyzed
- ✅ **Excellent System State Discovered**: Zero hardcoded logic violations requiring migration
- ✅ **High-Quality Codebase Validated**: 97% of patterns are appropriate (test data, configuration, documentation)
- ✅ **Dynamic Architecture Confirmed**: All critical systems properly using ontology_manager queries
- ✅ **System Health Maintained**: 22/22 OntologyManager tests passing throughout analysis
- ✅ **Evidence-Based Assessment**: All claims supported by systematic pattern classification

**CURRENT CHALLENGE** (Identified through analysis):
- **Ontology Resilience Untested**: System uses dynamic queries but resilience to ontology changes unvalidated
- **End-to-End Impact Unknown**: Need to validate full TEXT → JSON → HTML pipeline with ontology modifications
- **Fail-Fast Behavior Unverified**: Error handling when ontology changes break assumptions needs testing

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

## 📋 PHASE 26A: AGGRESSIVE ONTOLOGY RESILIENCE TESTING

**CRITICAL REQUIREMENT**: Validate system can handle arbitrary ontology changes with fail-fast error handling or graceful adaptation.

### 🚨 TASK 1: BASELINE VALIDATION & COMMIT SAFETY (30 minutes)
*Establish rollback point and validate current end-to-end functionality*

**OBJECTIVE**: Create safety checkpoint and establish baseline end-to-end behavior

```bash
# 1. Commit current state for rollback safety
git add -A && git commit -m "Pre-ontology-resilience baseline - Phase 26A start"

# 2. Validate current system works end-to-end (establish baseline)
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# Core system health check
python -m pytest tests/test_ontology_manager.py -v  # Must be 22/22

# End-to-end pipeline test with actual input text
python analyze_direct.py input_text/revolutions/french_revolution.txt
# Note: May hit LLM validation issues - document current behavior

# Test multiple inputs if possible
for input_file in input_text/*/*.txt; do
    echo "Testing: $input_file"
    python analyze_direct.py "$input_file" || echo "Failed: $input_file"
done
```

**DELIVERABLE**: Rollback commit + documented baseline end-to-end behavior

### 🔥 TASK 2: AGGRESSIVE ONTOLOGY STRESS TESTING (3-4 hours)
*Systematically break ontology to find brittleness through end-to-end testing*

**OBJECTIVE**: Discover what breaks when ontology changes through realistic end-to-end scenarios

```bash
# Backup current ontology
cp config/ontology_config.json config/ontology_config.json.backup

# Test 2.1: Remove Critical Edge Type ('supports')
echo "=== REMOVING 'supports' EDGE TYPE ===" > evidence/ontology_stress_tests.md
# [Edit ontology to remove 'supports']
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/ontology_stress_tests.md
# Document: Does it fail fast? Silent failure? Graceful adaptation?

# Restore ontology between tests
cp config/ontology_config.json.backup config/ontology_config.json

# Test 2.2: Rename Edge Type ('tests_hypothesis' → 'validates_hypothesis')
echo "=== RENAMING 'tests_hypothesis' → 'validates_hypothesis' ===" >> evidence/ontology_stress_tests.md
# [Edit ontology to rename edge type]
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/ontology_stress_tests.md

# Test 2.3: Add New Edge Type ('challenges_hypothesis')
echo "=== ADDING 'challenges_hypothesis' EDGE TYPE ===" >> evidence/ontology_stress_tests.md
# [Edit ontology to add new edge type]
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/ontology_stress_tests.md

# Test 2.4: Remove Entire Relationship Category (Evidence→Hypothesis)
echo "=== REMOVING Evidence→Hypothesis RELATIONSHIPS ===" >> evidence/ontology_stress_tests.md
# [Edit ontology to remove relationship category]
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/ontology_stress_tests.md

# Always restore baseline after each test
cp config/ontology_config.json.backup config/ontology_config.json
```

**DELIVERABLE**: Complete stress test log with all failure modes documented

### 🔧 TASK 3: BRITTLENESS ELIMINATION (4-6 hours)
*Fix every system that breaks inappropriately - implement fail-fast behavior*

**OBJECTIVE**: Convert brittle failures into appropriate fail-fast behavior or graceful adaptation

```bash
# For each failure discovered in Task 2:
# 1. Trace failure to specific code location
# 2. Determine if failure is appropriate (should fail-fast) or inappropriate (should adapt)
# 3. Fix inappropriate failures:
#    - Convert hardcoded references → dynamic ontology queries
#    - Add explicit error handling with clear messages
#    - Remove silent fallbacks that hide problems
# 4. Enhance appropriate failures:
#    - Make error messages clearer and more actionable
#    - Fail faster and louder
#    - Point to ontology configuration issues

# Testing loop for each fix:
fix_and_test() {
  local description=$1
  
  echo "=== FIXING: $description ===" >> evidence/fixes_log.md
  
  # [Implement specific fix]
  
  # Validate fix with original failing ontology
  python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/fixes_log.md
  
  # Validate fix doesn't break baseline
  cp config/ontology_config.json.backup config/ontology_config.json
  python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/fixes_log.md
  
  # Validate core system health maintained
  python -m pytest tests/test_ontology_manager.py -v | tee -a evidence/fixes_log.md
}
```

**DELIVERABLE**: All brittleness eliminated with fail-fast or graceful adaptation

### 🏆 TASK 4: ONTOLOGY EVOLUTION VALIDATION (2-3 hours)
*Validate system handles realistic ontology changes gracefully*

**OBJECTIVE**: Confirm system works across multiple ontology configurations

```bash
# Create realistic ontology evolution scenarios
# Scenario 1: Enhanced edge types (add properties, new relationships)
echo "=== ONTOLOGY EVOLUTION SCENARIO 1: Enhanced Types ===" > evidence/evolution_tests.md
# [Create enhanced ontology with additional edge properties]
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/evolution_tests.md
python analyze_direct.py input_text/american_revolution/american_revolution.txt 2>&1 | tee -a evidence/evolution_tests.md

# Scenario 2: Simplified ontology (remove unused types, streamline)  
echo "=== ONTOLOGY EVOLUTION SCENARIO 2: Simplified Types ===" >> evidence/evolution_tests.md
# [Create simplified ontology]
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/evolution_tests.md

# Scenario 3: Domain-specific ontology (specialized for political analysis)
echo "=== ONTOLOGY EVOLUTION SCENARIO 3: Domain-Specific ===" >> evidence/evolution_tests.md
# [Create domain-specific ontology]
python analyze_direct.py input_text/revolutions/french_revolution.txt 2>&1 | tee -a evidence/evolution_tests.md

# Validate all scenarios work or fail appropriately
for input_file in input_text/*/*.txt; do
    echo "Multi-input validation: $input_file" >> evidence/evolution_tests.md
    python analyze_direct.py "$input_file" 2>&1 | head -20 >> evidence/evolution_tests.md
done

# Restore baseline ontology
cp config/ontology_config.json.backup config/ontology_config.json
```

**DELIVERABLE**: Validated ontology-resilient system across multiple realistic configurations

### 📋 TASK 5: EVIDENCE-BASED ASSESSMENT (1-2 hours)
*Document actual ontology resilience achieved with evidence*

**OBJECTIVE**: Provide honest assessment of ontology resilience with supporting evidence

```bash
# Create comprehensive evidence package
EVIDENCE_DIR="evidence/current/Evidence_Phase26A_OntologyResilience_$(date +%Y%m%d)"
mkdir -p $EVIDENCE_DIR

# Copy all test results and logs
cp evidence/ontology_stress_tests.md $EVIDENCE_DIR/
cp evidence/fixes_log.md $EVIDENCE_DIR/
cp evidence/evolution_tests.md $EVIDENCE_DIR/

# Document final system state
python -m pytest tests/test_ontology_manager.py -v > $EVIDENCE_DIR/final_system_health.txt
python analyze_direct.py input_text/revolutions/french_revolution.txt > $EVIDENCE_DIR/final_pipeline_test.txt 2>&1

# Create honest assessment:
# - What ontology changes are now supported
# - What changes cause appropriate failures  
# - What changes cause inappropriate failures (remaining work)
# - System limitations and recommended usage patterns
```

**CRITICAL SUCCESS CRITERIA**:
✅ **End-to-End Resilience**: Can modify ontology without breaking TEXT→JSON→HTML pipeline  
✅ **Fail-Fast Behavior**: Clear errors when ontology changes break assumptions  
✅ **Graceful Adaptation**: System adapts to compatible ontology changes  
✅ **Multi-Input Consistency**: Behavior consistent across different input texts  
✅ **Evidence-Based Claims**: All resilience claims supported by test results

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