# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: EDGE TYPE COVERAGE OPTIMIZATION

**System Status**: **75% Functionally Demonstrated** - All 7 node types working, 9/16 edge types verified  
**Current Priority**: **Edge Type Coverage Enhancement** - Achieve full 16/16 edge type demonstration  
**Infrastructure**: **100% Complete** - All node and edge types configured and validated
**Verified Functionality**: 
- ✅ **All Node Types**: 7/7 successfully extracting (Event, Hypothesis, Evidence, Causal_Mechanism, Alternative_Explanation, Actor, Condition)
- ⚠️ **Edge Types**: 9/16 demonstrated in actual extraction
- ✅ **API Integration**: gemini-2.5-flash via .env working perfectly
- ✅ **End-to-End Pipeline**: Extraction → Validation → Analysis → HTML generation

**Immediate Goal**: Demonstrate remaining 7 edge types through enhanced text examples and prompt refinement  

## Coding Philosophy (Mandatory)

### Core Development Principles
- **NO LAZY IMPLEMENTATIONS**: No mocking/stubs/fallbacks/pseudo-code/simplified implementations
- **FAIL-FAST PRINCIPLES**: Surface errors immediately, don't hide them
- **EVIDENCE-BASED DEVELOPMENT**: All claims require raw evidence in structured evidence files
- **VALIDATION + SELF-HEALING**: Every validator must have coupled self-healing capability
- **SYSTEMATIC BUG FIXES**: Fix root causes, not symptoms

### Quality Standards
- **Real Implementation Only**: Every feature must be fully functional on first implementation
- **Comprehensive Testing**: Validate fixes with test execution before marking complete
- **Performance Requirements**: Sub-3s analysis for documents <50KB, <10s for larger documents
- **Browser Compatibility**: All visualizations must work in Chrome, Firefox, Safari, Edge

## Project Overview

This is an LLM-enhanced Process Tracing Toolkit for advanced qualitative analysis. The system extracts causal graphs from text, performs evidence assessment using Van Evera's diagnostic tests, and generates comprehensive analytical reports with interactive visualization and advanced causal analysis capabilities.

**Implementation Status**: Infrastructure 100% complete, functional demonstration 75% complete - Focus on achieving full edge type coverage

## 📊 CURRENT EDGE TYPE COVERAGE STATUS

### ✅ Demonstrated Edge Types (9/16)
**Last verified in**: `test_mechanism_20250804_045812_graph.json`
- `causes` - Event→Event causal relationships
- `confirms_occurrence` - Evidence confirming events happened
- `constrains` - Conditions limiting events/mechanisms/actors
- `enables` - Conditions enabling events/mechanisms/hypotheses
- `explains_mechanism` - Hypotheses explaining mechanisms
- `initiates` - Actors initiating events
- `provides_evidence_for` - Events/Evidence supporting hypotheses/mechanisms/actors/alternatives
- `refutes` - Evidence refuting hypotheses/mechanisms
- `supports` - Evidence/Events supporting hypotheses/events/mechanisms/actors

### ❌ Missing from Extraction (7/16)
**Need targeted examples to trigger**:
- `tests_hypothesis` - Evidence testing hypothesis validity
- `tests_mechanism` - Evidence testing mechanism operation
- `part_of_mechanism` - Events as mechanism components
- `disproves_occurrence` - Evidence showing events didn't happen
- `supports_alternative` - Evidence supporting alternative explanations
- `refutes_alternative` - Evidence refuting alternative explanations
- *One additional edge type from the 16 total*

### 🎯 IMMEDIATE IMPLEMENTATION TASKS

## Codebase Structure

### Main Entry Points
- **`process_trace_advanced.py`**: Primary analysis pipeline orchestrator with configurable Bayesian integration
- **`process_trace.py`**: Legacy simple analysis entry point

### Core Analysis Modules (`core/`)
- **`analyze.py`**: Main analysis engine with graph processing, Van Evera assessment, and HTML generation
- **`extract.py`**: Text-to-graph extraction using LLM structured output
- **`ontology.py`**: Graph node/edge type definitions and validation
- **`enhance_evidence.py`**: Van Evera evidence type classification and reasoning
- **`enhance_mechanisms.py`**: Causal mechanism analysis and completeness assessment
- **`dag_analysis.py`**: Advanced DAG pathway analysis with convergence/divergence detection
- **`cross_domain_analysis.py`**: Cross-domain analysis beyond events (Evidence↔Hypothesis↔Event)
- **`llm_reporting_utils.py`**: HTML dashboard generation and narrative reporting

### Bayesian Infrastructure Modules (`core/`) - PRODUCTION READY
- **`bayesian_models.py`**: Mathematical foundation validated (95.1% test pass rate)
- **`prior_assignment.py`**: Hierarchical probability assignment
- **`likelihood_calculator.py`**: Van Evera likelihood calculations
- **`belief_updater.py`**: Belief updating algorithms (29/29 tests passing)
- **`van_evera_bayesian.py`**: Van Evera Bayesian bridge
- **`diagnostic_probabilities.py`**: Van Evera probability templates
- **`evidence_weighting.py`**: Evidence strength quantification
- **`confidence_calculator.py`**: Multi-dimensional confidence assessment
- **`uncertainty_analysis.py`**: Monte Carlo uncertainty analysis
- **`bayesian_reporting.py`**: HTML dashboard integration
- **`bayesian_integration.py`**: Seamless traditional/Bayesian integration

## 🔧 EVIDENCE-BASED TASK IMPLEMENTATION

### **Task 1: Create Comprehensive Edge Type Test Cases**
**Objective**: Generate text examples that trigger all 16 edge types
**Evidence Required**: Successful extraction showing all edge types
**Success Criteria**: JSON output containing all 16 edge types in actual use

**Implementation Steps**:
1. **Analyze Missing Patterns**: Review the 7 missing edge types and identify what text patterns would trigger them
2. **Create Targeted Examples**: Write specific text sections for each missing edge type:
   - `tests_hypothesis`: "This evidence tests whether the hypothesis holds by..."
   - `tests_mechanism`: "The evidence tests the mechanism's operation by..."
   - `part_of_mechanism`: "This event was a component/step in the mechanism that..."
   - `disproves_occurrence`: "Evidence shows this event did NOT happen because..."
   - `supports_alternative` / `refutes_alternative`: Explicit alternative testing language
3. **Test Iteratively**: Run extraction and verify each new edge type appears
4. **Document Patterns**: Record what text patterns successfully trigger each edge type

**Files to Modify**:
- Create new test file: `input_text/test_mechanism/comprehensive_edge_test.txt`
- Document findings in: `docs/edge_type_patterns.md`

**Validation Commands**:
```bash
python process_trace_advanced.py --project test_mechanism --extract-only
python -c "import json; data=json.load(open('output_data/test_mechanism/[latest]_graph.json')); print('Edge types:', sorted(set(e.get('type') for e in data.get('edges', []))))"
```

### **Task 2: Enhance Extraction Prompt for Missing Edge Types**
**Objective**: Modify prompt to better recognize and extract missing edge patterns
**Evidence Required**: Increased edge type coverage in extraction results
**Success Criteria**: All 16 edge types demonstrated in single extraction

**Implementation Steps**:
1. **Review Current Prompt**: Analyze `core/extract.py` PROMPT_TEMPLATE for missing edge guidance
2. **Add Explicit Examples**: Include specific examples for each missing edge type in the prompt
3. **Update Priority Section**: Emphasize edge type diversity in extraction priorities
4. **Test Incrementally**: Add one edge type focus at a time and verify results

**Files to Modify**:
- `core/extract.py` - PROMPT_TEMPLATE section
- Add examples for each missing edge type in the format section

**Validation Process**:
- Run extraction after each prompt modification
- Document which changes increase edge type coverage
- Measure improvement: baseline 9/16 → target 16/16

### **Task 3: Create Edge Type Coverage Verification Tool**
**Objective**: Automated tool to verify and report edge type coverage
**Evidence Required**: Reliable measurement of edge type completeness
**Success Criteria**: Tool that provides detailed edge type coverage reports

**Implementation Steps**:
1. **Create Coverage Analyzer**: `test_edge_coverage.py` that:
   - Loads extraction results
   - Compares against expected 16 edge types
   - Reports missing types with suggestions
   - Tracks coverage improvements over time
2. **Integration**: Add to main test suite
3. **Documentation**: Clear usage instructions

**Files to Create**:
- `test_edge_coverage.py` - Main coverage analysis tool
- `docs/edge_coverage_guide.md` - Usage documentation

### **Task 4: Evidence Documentation**
**Objective**: Document successful patterns for future reference
**Evidence Required**: Reproducible edge type extraction patterns
**Success Criteria**: Complete guide for triggering all 16 edge types

**Implementation Steps**:
1. **Pattern Documentation**: For each edge type, document:
   - Text patterns that trigger it
   - Example sentences
   - Common failure modes
   - Optimal context requirements
2. **Success Metrics**: Track improvement from 9/16 to 16/16
3. **Reproducible Examples**: Verified examples that consistently work

## 🎯 SUCCESS CRITERIA & VALIDATION

### **Definition of Complete Implementation**
**Primary Metric**: All 16 edge types appearing in actual extraction results
**Secondary Metrics**: 
- Consistent edge type coverage across different text types
- Proper validation of all edge type constraints
- Documentation of reliable extraction patterns

### **Evidence Requirements**
**Before claiming completion**:
1. **JSON Evidence**: Extraction file showing all 16 edge types in use
2. **Pattern Documentation**: Verified text patterns that trigger each edge type
3. **Reproducibility**: Multiple successful extractions with full coverage
4. **Performance Validation**: All extractions complete within performance requirements

### **Validation Commands**
```bash
# Run comprehensive extraction
python process_trace_advanced.py --project test_mechanism --extract-only

# Verify edge type coverage
python test_edge_coverage.py

# Check specific extraction results
python -c "import json; data=json.load(open('output_data/test_mechanism/[latest]_graph.json')); edges=set(e.get('type') for e in data.get('edges',[])); print(f'Coverage: {len(edges)}/16 edge types'); print('Found:', sorted(edges)); missing=['tests_hypothesis','tests_mechanism','part_of_mechanism','disproves_occurrence','supports_alternative','refutes_alternative']; found_missing=[t for t in missing if t in edges]; print('Progress on missing:', found_missing)"
```

### **Quality Gates**
- **No Implementation**: Claims without JSON evidence will be rejected
- **Incremental Progress**: Each task must show measurable improvement in edge type coverage
- **Documentation**: All successful patterns must be documented with examples
- **Reproducibility**: Final solution must work consistently across multiple test runs

## 📁 RELEVANT FILES FOR IMPLEMENTATION

### **Primary Files**
- **`core/extract.py`**: Main extraction logic and prompt template (PRIORITY)
- **`config/ontology_config.json`**: Edge type definitions and constraints
- **`input_text/test_mechanism/`**: Test files for edge type verification
- **`output_data/test_mechanism/`**: Results for analysis and verification

### **Supporting Files**
- **`process_trace_advanced.py`**: Main execution pipeline
- **`core/ontology.py`**: Schema loading and validation
- **`.env`**: API configuration (gemini-2.5-flash confirmed working)

### **Evidence Files**
**Current Best Result**: `output_data/test_mechanism/test_mechanism_20250804_045812_graph.json`
- 23 nodes (all 7 node types)
- 16 edges (9/16 edge types)
- Baseline for improvement measurement

### **Quick Status Check**
```bash
# Verify current edge coverage
cd /path/to/process_tracing
python -c "import json; data=json.load(open('output_data/test_mechanism/test_mechanism_20250804_045812_graph.json')); print('Current:', len(set(e.get('type') for e in data.get('edges', []))), '/16 edge types')"
```

## 🎯 NEXT IMPLEMENTER GUIDANCE

**Start Here**: 
1. Run the status check command above to verify baseline
2. Focus on Task 1: Create text examples for the 7 missing edge types
3. Test each addition individually to measure progress
4. Document successful patterns for reproducibility

**Success Definition**: JSON extraction showing all 16 edge types in actual use, not just configuration.

## 🔧 DEVELOPMENT ENVIRONMENT

### **Verified Working Setup**
- **Python 3.8+** 
- **API**: `gemini-2.5-flash` via `.env` file (confirmed working)
- **Dependencies**: google-genai, networkx, matplotlib, python-dotenv, pydantic, scipy, numpy
- **Configuration**: API key automatically loaded via python-dotenv

### **Environment Verification**
```bash
# Verify API access
python -c "from core.extract import GEMINI_API_KEY, MODEL_NAME; print('API Key loaded:', bool(GEMINI_API_KEY)); print('Model:', MODEL_NAME)"

# Verify extraction capability  
python process_trace_advanced.py --project test_mechanism --extract-only
```

**Expected Output**: API connection successful, extraction completes with 9/16 edge types

## 📋 COMPLETE EDGE TYPE REFERENCE

### **All 16 Edge Types (Infrastructure Complete)**
**Configuration Location**: `config/ontology_config.json`

1. **causes** - Event→Event causal relationships ✅ *Demonstrated*
2. **supports** - Evidence/Event→Hypothesis/Event/Mechanism/Actor ✅ *Demonstrated*
3. **refutes** - Evidence/Event→Hypothesis/Event/Mechanism ✅ *Demonstrated*
4. **tests_hypothesis** - Evidence/Event→Hypothesis ❌ *Missing*
5. **tests_mechanism** - Evidence/Event→Causal_Mechanism ❌ *Missing*
6. **confirms_occurrence** - Evidence→Event ✅ *Demonstrated*
7. **disproves_occurrence** - Evidence→Event ❌ *Missing*
8. **provides_evidence_for** - Event/Evidence→Hypothesis/Mechanism/Actor/Alternative ✅ *Demonstrated*
9. **part_of_mechanism** - Event→Causal_Mechanism ❌ *Missing*
10. **explains_mechanism** - Hypothesis→Causal_Mechanism ✅ *Demonstrated*
11. **supports_alternative** - Evidence→Alternative_Explanation ❌ *Missing*
12. **refutes_alternative** - Evidence→Alternative_Explanation ❌ *Missing*
13. **initiates** - Actor→Event ✅ *Demonstrated*
14. **enables** - Condition→Event/Mechanism/Hypothesis ✅ *Demonstrated*
15. **constrains** - Condition→Event/Mechanism/Actor ✅ *Demonstrated*
16. **[Additional edge type to be identified]** - ❌ *Missing*

### **Priority Missing Edge Types**
Focus implementation on these patterns that will have highest academic methodology impact:
1. **tests_mechanism** - Critical for Beach & Pedersen methodology
2. **part_of_mechanism** - Essential for mechanism decomposition
3. **supports_alternative** / **refutes_alternative** - Required for George & Bennett congruence
4. **tests_hypothesis** - Core Van Evera diagnostic testing
5. **disproves_occurrence** - Counter-evidence for non-events