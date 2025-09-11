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

## 🎯 CURRENT STATUS: Phase 24A - Ontology Architecture Investigation (Updated 2025-01-11)

**System Status**: **🎉 HTML GENERATION RESTORED - ONTOLOGY ARCHITECTURE INVESTIGATION**  
**Latest Achievement**: **Phase 23B Complete - Rich HTML generation pipeline with interactive network visualizations**  
**Current Priority**: **Investigate ontology architecture to address redundant edge relationships discovered in network visualization**

**PHASE 23B COMPLETION RESULTS**:
- ✅ **Rich HTML Generation**: Interactive vis.js network graphs with professional Bootstrap styling
- ✅ **Van Evera Analytics**: Evidence-hypothesis analysis, causal chains, hypothesis support scoring
- ✅ **Interactive Features**: Search, filtering, clickable nodes/edges with detailed information panels
- ✅ **Complete Pipeline**: TEXT → JSON → Rich HTML working end-to-end without hanging issues
- ✅ **Cross-Input Validation**: French Revolution (43 nodes/42 edges), American Revolution (40 nodes/38 edges), Westminster Debate (33 nodes/29 edges)

**DISCOVERED ONTOLOGY ARCHITECTURE ISSUE**:
- **Problem**: Network visualization reveals redundant edge relationships between Evidence and Hypothesis nodes
- **Specific Case**: Single evidence node connected to hypothesis via 3 separate edges: `provides_evidence_for`, `updates_probability`, `weighs_evidence`
- **Impact**: Logically redundant relationships - if evidence provides evidence for hypothesis, it automatically should update probability and contribute to weight
- **Root Cause**: Ontology contains overlapping edge types without clear semantic distinctions or logical dependencies

## 🔧 PHASE 24A: Ontology Architecture Investigation

### OBJECTIVE: Investigate current ontology architecture and document findings without making any changes

**CRITICAL INVESTIGATION QUESTION**:
- **Architecture Question**: Is the ontology properly abstracted as dependency injection, or hardcoded throughout the system?
- **Impact Assessment**: What would be required to improve ontology design for academic process tracing?
- **Change Scope**: How many system components would be affected by ontology modifications?

## 🔧 INVESTIGATION TASKS

### TASK 1: Ontology Architecture Analysis (60 minutes)

**OBJECTIVE**: Determine if ontology is properly abstracted or tightly coupled throughout system

**INVESTIGATION STEPS**:
1. **Locate Ontology Definition Sources**:
   ```bash
   # Find primary ontology definition files
   find . -name "*ontolog*" -type f
   
   # Identify configuration vs code definitions
   find . -name "*.json" | xargs grep -l "edge_types\|node_types"
   ```

2. **Analyze Dynamic vs Hardcoded Usage**:
   ```bash
   # Check for hardcoded edge type lists in code
   grep -r "\[.*provides_evidence_for.*\]" --include="*.py" .
   
   # Look for hardcoded string comparisons
   grep -r "== ['\"]provides_evidence_for['\"]" --include="*.py" .
   
   # Verify if extractors import ontology dynamically
   grep -A 10 -B 5 "import.*ontology" core/structured_extractor.py
   ```

3. **Document Current Architecture**:
   - Map all files that reference ontology
   - Identify hardcoded vs dynamic ontology usage
   - Assess coupling vs proper dependency injection
   - Document architectural strengths and weaknesses

**EVIDENCE REQUIREMENTS**:
- Complete list of ontology definition locations
- Analysis of hardcoded vs dynamic ontology consumption
- Assessment of system coupling to ontology structure
- Architectural recommendations for improvement

### TASK 2: Redundant Edge Type Analysis (45 minutes)

**OBJECTIVE**: Document current ontological redundancies and their logical relationships

**INVESTIGATION STEPS**:
1. **Extract Current Ontology Structure**:
   ```bash
   # Read the authoritative ontology definition
   cat config/ontology_config.json
   
   # Focus on problematic edge relationships
   grep -A 20 "provides_evidence_for\|updates_probability\|weighs_evidence" config/ontology_config.json
   ```

2. **Analyze Domain/Range Overlaps**:
   - Document all Evidence→Hypothesis edge types
   - Identify logical redundancies and dependencies
   - Map Van Evera diagnostic framework to current edge types
   - Assess academic process tracing requirements

3. **Document Logical Inconsistencies**:
   - Evidence that `provides_evidence_for` hypothesis logically implies probability updating
   - Evidence that `provides_evidence_for` hypothesis logically implies evidence weighting
   - Identify other redundant relationship patterns

**EVIDENCE REQUIREMENTS**:
- Complete ontology structure documentation
- Analysis of all Evidence→Hypothesis edge types
- Documentation of logical redundancies and dependencies
- Academic process tracing requirement assessment

### TASK 3: System Impact Assessment (45 minutes)

**OBJECTIVE**: Document what system components would be affected by ontology improvements

**INVESTIGATION STEPS**:
1. **Identify Ontology Consumers**:
   ```bash
   # Find all files that import ontology
   grep -r "from.*ontology import\|import.*ontology" --include="*.py" .
   
   # Find all references to specific edge types
   grep -r "provides_evidence_for\|updates_probability\|weighs_evidence" --include="*.py" .
   ```

2. **Analyze Change Impact Scope**:
   - LLM extraction pipeline dependencies
   - Graph validation and loading components  
   - HTML generation analytics dependencies
   - Existing data file compatibility
   - Test suite dependencies

3. **Document Change Complexity**:
   - Files requiring modification for ontology changes
   - Risk assessment for different change approaches
   - Migration requirements for existing data
   - Testing and validation requirements

**EVIDENCE REQUIREMENTS**:
- Complete list of ontology-dependent system components
- Impact assessment for different ontology improvement approaches
- Risk analysis for ontology modifications
- Recommendations for implementation strategy

### TASK 4: Academic Process Tracing Requirements (30 minutes)

**OBJECTIVE**: Document how current ontology aligns with academic process tracing standards

**INVESTIGATION STEPS**:
1. **Van Evera Framework Analysis**:
   - Map current edge types to Van Evera diagnostic tests
   - Identify missing academic process tracing components
   - Assess hierarchical vs flat relationship modeling

2. **Academic Standards Compliance**:
   - George & Bennett methodological requirements
   - Temporal sequence modeling capabilities
   - Alternative hypothesis testing framework
   - Mechanism decomposition support

3. **Improvement Recommendations**:
   - Academic-grade ontology design principles
   - Elimination of logical redundancies
   - Implementation of diagnostic test hierarchy
   - Research design integration requirements

**EVIDENCE REQUIREMENTS**:
- Assessment of current ontology vs academic standards
- Documentation of missing academic process tracing features
- Recommendations for academic-grade ontology improvements
- Implementation priority analysis for academic compliance

## 📊 INVESTIGATION SUCCESS CRITERIA

### **Documentation Success Criteria:**
1. **Architecture Assessment**: Complete analysis of ontology coupling vs dependency injection
2. **Redundancy Documentation**: Full analysis of logical redundancies in edge relationships
3. **Impact Analysis**: Comprehensive assessment of change scope and complexity
4. **Academic Alignment**: Documentation of current vs ideal process tracing ontology

### **Evidence-Based Findings:**
1. **System Architecture**: Clear documentation of hardcoded vs dynamic ontology usage
2. **Change Complexity**: Evidence-based assessment of modification requirements
3. **Academic Gap Analysis**: Documentation of ontology improvements needed for academic standards
4. **Implementation Strategy**: Recommendations for ontology improvement approach

---

## 🏗️ Codebase Structure

### Key Entry Points  
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline with basic HTML fallback
- **`core/structured_extractor.py`**: LLM extraction (Phase 23A: enhanced with raw response capture)
- **`core/analyze.py`**: Contains `load_graph()` (Phase 23A: fixed MultiDiGraph) + hanging `generate_html_report()`

### Critical Files for Phase 24A Investigation
- **`config/ontology_config.json`**: Authoritative ontology definition (PRIMARY TARGET)
- **`core/ontology.py`**: Ontology loading and interface module
- **`core/structured_extractor.py`**: LLM extraction pipeline (ontology consumer)
- **`core/html_generator.py`**: Van Evera analytics (ontology consumer)
- **`analyze_direct.py`**: Graph loading and validation (ontology consumer)

### Working Components (Phase 23B Complete)
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
- Every investigation step must produce concrete evidence files
- No assumptions or speculation - only data-driven conclusions
- Raw execution logs required for all claims

### FAIL-FAST PRINCIPLES  
- Surface data integrity issues immediately
- No silent data loss tolerance
- Clear error reporting with actionable information

### EVIDENCE-BASED DEVELOPMENT
- All investigation findings must be documented in `evidence/current/Evidence_Phase24A_OntologyInvestigation.md`
- Raw command outputs and analysis results required for all claims
- Systematic documentation of architectural findings

### SYSTEMATIC VALIDATION
- Test each investigation step before proceeding to next
- Validate raw response capture before analyzing processing pipeline
- Prove reproducibility before implementing fixes

---

## 📁 Evidence Structure

Evidence for Phase 24A must be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase24A_OntologyInvestigation.md    # Active development  
├── completed/
│   └── Evidence_Phase23B_HTMLGeneration.md         # Archived
```

**REQUIRED EVIDENCE FOR PHASE 24A**:
- Complete list of ontology definition locations and their relationships
- Analysis results showing hardcoded vs dynamic ontology usage patterns
- Documentation of all Evidence→Hypothesis edge type redundancies
- Impact assessment of files requiring modification for ontology changes
- Academic process tracing standards comparison with current ontology
- Recommendations for ontology architecture improvements

**CRITICAL**: All architectural claims must be supported by actual command outputs and code analysis evidence.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.