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

## 🎯 CURRENT STATUS: Phase 27B COMPLETE - Dynamic Ontology Validation Implemented (Updated 2025-09-14)

**System Status**: **🎉 ARCHITECTURAL IMPROVEMENT COMPLETE - Dynamic validation successfully implemented**  
**Latest Achievement**: **Phase 27B Complete - Hardcoded validation replaced with dynamic validation**  
**Current Priority**: **System ready for ontology evolution - validation adapts automatically to ontology changes**

**ARCHITECTURAL PROBLEMS SOLVED**:
- ✅ **Dynamic Validation**: System validates functional capabilities, not hardcoded edge names
- ✅ **Flexible Architecture**: Validation rules derive from ontology structure automatically
- ✅ **Proper Solution**: Evolution workaround replaced with robust validation architecture
- ✅ **Schema Integration**: Validation discovers requirements from ontology capabilities

**DYNAMIC VALIDATION IMPLEMENTED**: 
- ✅ **Functional Requirements**: Validates Evidence->Hypothesis connectivity (critical) vs specific names
- ✅ **Schema-Driven**: DynamicOntologyValidator automatically adapts to ontology structure
- ✅ **Ontology Agnostic**: Supports any ontology meeting functional requirements
- ✅ **Multi-Mode**: Three validation modes (strict/minimal/schema-only) for different use cases

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

## 📋 PHASE 27B: DYNAMIC ONTOLOGY VALIDATION IMPLEMENTATION

**EVIDENCE-BASED APPROACH**: Replace hardcoded validation assumptions with dynamic functional requirement validation.

### 🔍 TASK 1: FUNCTIONAL REQUIREMENT DISCOVERY (3-4 hours)
*Understand what the system actually needs vs what it assumes it needs*

**OBJECTIVE**: Discover actual system requirements through codebase analysis

```bash
cd /home/brian/projects/process_tracing
source test_env/bin/activate

# 1.1 Create evidence file
mkdir -p evidence/current
echo "=== PHASE 27B TASK 1: FUNCTIONAL REQUIREMENT DISCOVERY ===" > evidence/current/Evidence_Phase27B_RequirementDiscovery.md

# 1.2 Find hardcoded validation assumptions
echo "=== CURRENT HARDCODED VALIDATION ===" >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md
grep -r "required_edges.*=.*\[" --include="*.py" . >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md
echo "" >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md

# 1.3 Analyze what ontology_manager can already tell us
echo "=== ONTOLOGY MANAGER CAPABILITIES ===" >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md
python -c "
from core.ontology_manager import ontology_manager
print('Evidence-Hypothesis edges:', ontology_manager.get_evidence_hypothesis_edges())
print('Van Evera edges:', ontology_manager.get_van_evera_edges())
print('All edge types:', ontology_manager.get_all_edge_types())
print('All node types:', ontology_manager.get_all_node_types())
" >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md

# 1.4 Find actual functional usage patterns
echo "=== FUNCTIONAL USAGE PATTERNS ===" >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md
grep -r "get_evidence_hypothesis_edges\|get_van_evera_edges" --include="*.py" core/ >> evidence/current/Evidence_Phase27B_RequirementDiscovery.md
```

**SUCCESS CRITERIA**: 
- ✅ Identified actual functional requirements (what system needs to operate)
- ✅ Separated naming assumptions from functional needs
- ✅ Documented ontology_manager's current query capabilities
- ✅ Mapped hardcoded validation to actual usage patterns

### 🏗️ TASK 2: DYNAMIC VALIDATOR ARCHITECTURE (2-3 hours) 
*Design validation system that derives requirements from ontology and codebase*

**OBJECTIVE**: Design validator that checks functional capabilities, not hardcoded names

```bash
# 2.1 Document validation architecture design
echo "=== DYNAMIC VALIDATOR ARCHITECTURE ===" > evidence/current/Evidence_Phase27B_ValidatorDesign.md

# 2.2 Design functional requirement categories
echo "=== REQUIREMENT CATEGORIES ===" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "CRITICAL REQUIREMENTS (system cannot function without):" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "- At least one edge type connecting Evidence to Hypothesis" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "- Valid JSON schema structure" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "OPTIONAL REQUIREMENTS (enhanced functionality):" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "- Van Evera diagnostic test capabilities" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "- Probative value properties" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md

# 2.3 Design validation modes
echo "=== VALIDATION MODES ===" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "STRICT MODE: Requires all critical + optional requirements" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "MINIMAL MODE: Requires only critical requirements" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
echo "SCHEMA_ONLY MODE: Validates JSON structure only" >> evidence/current/Evidence_Phase27B_ValidatorDesign.md
```

**SUCCESS CRITERIA**:
- ✅ Designed functional requirement categories (critical vs optional)
- ✅ Defined validation modes for different use cases  
- ✅ Separated schema validation from functional validation
- ✅ Created architecture that adapts to ontology changes

### 🔧 TASK 3: DYNAMIC VALIDATOR IMPLEMENTATION (4-5 hours)
*Build validation system that queries ontology capabilities*

**OBJECTIVE**: Implement DynamicOntologyValidator class

```bash
# 3.1 Document implementation progress
echo "=== DYNAMIC VALIDATOR IMPLEMENTATION ===" > evidence/current/Evidence_Phase27B_ValidatorImplementation.md

# 3.2 Create validator class
echo "Creating DynamicOntologyValidator class..." >> evidence/current/Evidence_Phase27B_ValidatorImplementation.md
# Implementation will create: core/dynamic_ontology_validator.py

# 3.3 Test validator with current ontology
echo "=== BASELINE VALIDATION TESTING ===" >> evidence/current/Evidence_Phase27B_ValidatorImplementation.md
python -c "
from core.dynamic_ontology_validator import DynamicOntologyValidator
from core.ontology_manager import ontology_manager

validator = DynamicOntologyValidator(ontology_manager)
result = validator.validate('strict')
print('Baseline validation result:', result.summary())
" >> evidence/current/Evidence_Phase27B_ValidatorImplementation.md 2>&1

# 3.4 Test functional equivalence
echo "=== FUNCTIONAL EQUIVALENCE TESTING ===" >> evidence/current/Evidence_Phase27B_ValidatorImplementation.md
# Test: ontology with different edge names but same functionality should pass
```

**SUCCESS CRITERIA**:
- ✅ DynamicOntologyValidator class implemented
- ✅ Functional requirements replace hardcoded edge type lists
- ✅ Multiple validation modes supported (strict/minimal/schema-only)
- ✅ Current ontology passes validation with new system

### 🔄 TASK 4: INTEGRATION AND MIGRATION (2-3 hours)
*Replace hardcoded validation throughout system*

**OBJECTIVE**: Migrate analyze_direct.py to use dynamic validation

```bash
# 4.1 Document migration process
echo "=== VALIDATION MIGRATION ===" > evidence/current/Evidence_Phase27B_ValidationMigration.md

# 4.2 Find all validation call sites
echo "=== VALIDATION CALL SITES ===" >> evidence/current/Evidence_Phase27B_ValidationMigration.md
grep -r "validate_system_ontology" --include="*.py" . >> evidence/current/Evidence_Phase27B_ValidationMigration.md

# 4.3 Update analyze_direct.py validation
echo "=== ANALYZE_DIRECT.PY MIGRATION ===" >> evidence/current/Evidence_Phase27B_ValidationMigration.md
# Replace hardcoded validation with dynamic validation

# 4.4 Update command line interface  
echo "=== CLI INTERFACE UPDATES ===" >> evidence/current/Evidence_Phase27B_ValidationMigration.md
# Replace --evolution-mode with --validation-mode [strict|minimal|schema-only]
```

**SUCCESS CRITERIA**:
- ✅ All hardcoded validation replaced with dynamic validation
- ✅ Command line interface improved (--validation-mode option)
- ✅ Backward compatibility maintained (strict mode as default)
- ✅ Error messages are clear and actionable

### 🧪 TASK 5: COMPREHENSIVE FUNCTIONAL TESTING (3-4 hours)
*Verify dynamic validation works with various ontology configurations*

**OBJECTIVE**: Test system with functionally equivalent but structurally different ontologies

```bash
# 5.1 Create comprehensive testing evidence
echo "=== COMPREHENSIVE DYNAMIC VALIDATION TESTING ===" > evidence/current/Evidence_Phase27B_ComprehensiveTesting.md

# 5.2 Test functional equivalence
echo "=== FUNCTIONAL EQUIVALENCE TESTS ===" >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md
# Create ontology with different edge names but same Evidence->Hypothesis functionality
# Should pass validation because functional requirements are met

# 5.3 Test critical requirement violations
echo "=== CRITICAL REQUIREMENT VIOLATION TESTS ===" >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md  
# Remove ALL Evidence->Hypothesis connections
# Should fail validation with clear error message

# 5.4 Test validation modes
echo "=== VALIDATION MODE TESTING ===" >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md
for mode in "strict" "minimal" "schema-only"; do
    echo "Testing --validation-mode $mode" >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md
    timeout 60 python analyze_direct.py input_text/revolutions/french_revolution.txt --validation-mode $mode --extract-only >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md 2>&1
    echo "" >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md
done

# 5.5 Final system health check
python -m pytest tests/test_ontology_manager.py -v >> evidence/current/Evidence_Phase27B_ComprehensiveTesting.md
```

**SUCCESS CRITERIA**: 
- ✅ Functionally equivalent ontologies pass validation
- ✅ Missing critical functionality fails validation appropriately
- ✅ All validation modes work correctly
- ✅ System health maintained (all tests passing)

---

## 🎯 PHASE 27B SUCCESS CRITERIA

**ARCHITECTURAL IMPROVEMENT**:
- ✅ **Dynamic Validation**: System validates functional capabilities, not hardcoded names
- ✅ **Schema-Driven**: Validation derives from ontology structure automatically  
- ✅ **Ontology Agnostic**: Supports any ontology meeting functional requirements
- ✅ **Maintainable**: Requirements discoverable and extensible without hardcoding

**FUNCTIONAL SUCCESS**:
- ✅ **Equivalent Ontologies**: Different edge names with same functionality work
- ✅ **Clear Requirements**: System documents what it needs functionally
- ✅ **Graceful Degradation**: Missing optional features generate warnings, not errors
- ✅ **Better UX**: Users understand functional requirements, not implementation details

**DELIVERABLES**:
1. **DynamicOntologyValidator**: Replaces hardcoded validation with functional requirements
2. **Validation Modes**: strict/minimal/schema-only options for different use cases
3. **Improved CLI**: --validation-mode replaces --evolution-mode workaround
4. **Comprehensive Testing**: Evidence that various ontology designs work correctly
5. **Documentation**: Clear explanation of functional requirements vs naming requirements

---

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline entry point
- **`core/structured_extractor.py`**: LLM extraction with Pydantic validation
- **`core/analyze.py`**: Graph loading and analysis orchestration (✅ migrated to dynamic ontology)

### Critical Files Status (Phase 27B Complete)
- **`core/ontology_manager.py`**: ✅ EXCELLENT - 22 passing tests, perfect fail-fast behavior for ontology changes
- **`config/ontology_config.json`**: ✅ STABLE - Authoritative ontology definition, handles additions gracefully
- **`analyze_direct.py`**: ✅ UPGRADED - Dynamic validation implemented, supports ontology evolution
- **`core/dynamic_ontology_validator.py`**: ✅ NEW - Validates functional capabilities instead of hardcoded names
- **`core/structured_extractor.py`**: ✅ WORKING - LLM extraction completes successfully even with modified ontology
- **`core/plugins/`**: ✅ DYNAMIC - Uses ontology_manager queries, adapts to ontology changes automatically
- **`core/enhance_*`**: ✅ DYNAMIC - Enhancement components use ontology_manager, fully adaptable

### System Status (Phase 27B Complete)
- **Core Resilience**: ✅ EXCELLENT - OntologyManager provides perfect fail-fast behavior
- **Dynamic Architecture**: ✅ SUCCESS - System handles ontology changes gracefully (22/22 tests pass)  
- **Validation Architecture**: ✅ UPGRADED - Dynamic validation replaces hardcoded assumptions
- **Ontology Evolution**: ✅ ENABLED - System adapts automatically to ontology structure changes

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

## 🎉 PHASE 27B COMPLETE - DYNAMIC VALIDATION IMPLEMENTED

### ✅ SYSTEM HEALTH STATUS (Phase 27B Complete):
- **Ontology Manager**: 22/22 tests passing - EXCELLENT
- **Dynamic Validation**: DynamicOntologyValidator implemented - EXCELLENT  
- **Validation Modes**: Three modes (strict/minimal/schema-only) working - EXCELLENT
- **CLI Interface**: --validation-mode argument replacing --evolution-mode - EXCELLENT
- **LLM Extraction**: Working reliably - EXCELLENT
- **Overall System**: FULLY FUNCTIONAL with dynamic ontology validation architecture

### 🏆 ACHIEVEMENTS:
1. **DynamicOntologyValidator**: Validates functional capabilities instead of hardcoded names
2. **Multi-Mode Validation**: strict/minimal/schema-only options for different use cases  
3. **Improved CLI**: --validation-mode replaces evolution workaround
4. **Comprehensive Testing**: Evidence that various ontology designs work correctly
5. **Documentation**: Clear explanation of functional vs naming requirements

### 🚀 READY FOR FUTURE ONTOLOGY EVOLUTION:
System now automatically adapts to ontology changes. Any ontology with Evidence->Hypothesis connectivity will work seamlessly. The architectural improvement is complete and the system is ready for production use with ontology evolution capabilities.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.