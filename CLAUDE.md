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

## 🎯 CURRENT STATUS: Phase 23A - Data Integrity Investigation & Resolution (Updated 2025-01-09)

**System Status**: **📊 FUNCTIONAL PIPELINE WITH DATA QUALITY ISSUES**  
**Latest Achievement**: **Phase 22A Complete - Full TEXT → JSON → HTML pipeline operational**  
**Current Priority**: **Investigate and resolve node/edge consistency issues in extraction pipeline**

**PHASE 22A COMPLETION RESULTS**:
- ✅ **Complete Pipeline**: TEXT → JSON → HTML fully functional
- ✅ **Performance Validated**: French Revolution (52,160 chars) → 37 nodes, 34 edges → HTML (3min extraction, <1s analysis)
- ✅ **Direct Entry Point**: Successfully bypasses hanging analysis module
- ✅ **Professional HTML Output**: Statistics, tables, formatted reports generated
- ❌ **Data Quality Issue**: 3/34 edges lost due to missing node references

**DATA INTEGRITY PROBLEM IDENTIFIED**:
- **Issue**: LLM extraction creates edges referencing `evidence_flight_to_varennes_1791` that doesn't exist in nodes
- **Impact**: 3 edges dropped during load_graph processing (34 → 29 loaded edges)  
- **Root Cause**: Unknown - requires systematic investigation
- **User Impact**: Incomplete relationship data in final HTML reports

## 🔧 PHASE 23A: Systematic Data Integrity Investigation

### OBJECTIVE: Identify and resolve LLM extraction consistency issues through evidence-based investigation

**CRITICAL DATA QUALITY PROBLEM**:
```
[EXTRACTION] Graph extracted: 37 nodes, 34 edges
[LOAD] Graph loaded successfully: 37 nodes, 29 edges  # ← 5 edges lost
WARNING: Skipping edge because source node not in graph (3x warnings)
```

**SPECIFIC CASE IDENTIFIED**:
- **Missing Node**: `evidence_flight_to_varennes_1791` 
- **Existing Similar**: `event_flight_to_varennes_1791` (exists)
- **Lost Edges**: 3 edges with detailed properties and source quotes
- **Pattern**: Evidence node missing but Event node exists

## 🔧 SYSTEMATIC INVESTIGATION TASKS

### TASK 1: Raw LLM Response Capture & Analysis (45 minutes)

**OBJECTIVE**: Determine if issue occurs during LLM generation or post-processing

**IMPLEMENTATION STEPS**:
1. **Modify StructuredProcessTracingExtractor** to capture raw LLM output:
   ```python
   # In core/structured_extractor.py - add logging before any processing
   def extract_graph(self, text: str, project_name: str = "default") -> StructuredExtractionResult:
       # Add raw response capture
       raw_response = self._call_llm(text)  # Whatever the actual LLM call is
       
       # CRITICAL: Save raw response before any processing
       raw_filepath = f"debug/raw_llm_response_{timestamp}.json"
       with open(raw_filepath, 'w') as f:
           f.write(raw_response)
       print(f"[DEBUG] Raw LLM response saved to: {raw_filepath}")
   ```

2. **Compare raw vs processed output**:
   ```bash
   # Test extraction with debug logging
   python analyze_direct.py input_text/revolutions/french_revolution.txt --extract-only
   
   # Analyze the raw response file
   python -c "
   import json
   with open('debug/raw_llm_response_[timestamp].json') as f:
       raw_data = json.load(f)
   
   # Check if evidence_flight_to_varennes_1791 exists in raw response
   raw_nodes = {node['id'] for node in raw_data.get('nodes', [])}
   print('Missing node in raw response:', 'evidence_flight_to_varennes_1791' not in raw_nodes)
   "
   ```

3. **Map processing pipeline losses**:
   - Document exactly where nodes disappear (if they do)
   - Identify which processing step removes nodes
   - Create timeline of data transformations

**EVIDENCE REQUIREMENTS**:
- Raw LLM response JSON file
- Node count at each processing stage
- Exact location where data loss occurs (if during processing)

### TASK 2: Validation Pipeline Audit (30 minutes)

**OBJECTIVE**: Examine all validation/cleaning steps for node filtering

**INVESTIGATION TARGETS**:
1. **Pydantic Model Validation**:
   ```bash
   # Examine validation rules
   grep -r "evidence_" core/plugins/van_evera_llm_schemas.py
   grep -A 10 -B 10 "class.*Node\|class.*Edge" core/plugins/van_evera_llm_schemas.py
   ```

2. **Processing Functions Audit**:
   ```bash
   # Find all functions that might filter/transform nodes
   grep -r "filter\|remove\|skip" core/structured_extractor.py
   grep -r "validation\|clean" core/structured_extractor.py
   ```

3. **JSON Cleaning Steps**:
   ```bash
   # Check if there are cleaning steps that might remove nodes
   grep -A 5 -B 5 "clean.*json\|sanitize" core/structured_extractor.py
   ```

**EVIDENCE REQUIREMENTS**:
- List of all validation rules that could affect nodes
- Documentation of any filtering/cleaning logic
- Confirmation of processing pipeline integrity

### TASK 3: Reproducibility & Pattern Analysis (30 minutes)

**OBJECTIVE**: Determine if issue is systematic or random

**SYSTEMATIC TESTING**:
1. **Multiple Runs Same Text**:
   ```bash
   # Run extraction 3 times on same input
   for i in {1..3}; do
       echo "Run $i:"
       python analyze_direct.py input_text/revolutions/french_revolution.txt --extract-only
       # Check if same nodes are missing
   done
   ```

2. **Different Input Texts**:
   ```bash
   # Test with American Revolution text
   python analyze_direct.py input_text/american_revolution/american_revolution.txt --extract-only
   
   # Check for similar missing evidence nodes pattern
   ```

3. **Pattern Analysis**:
   ```python
   # Create systematic analysis script
   def analyze_node_edge_consistency(json_file):
       # Load and check all node/edge references
       # Return detailed mismatch report
       # Pattern detection for missing evidence nodes
   ```

**EVIDENCE REQUIREMENTS**:
- Consistency report across multiple runs
- Pattern documentation (is it always evidence nodes?)
- Scope assessment (how common is this issue?)

### TASK 4: Root Cause Determination & Resolution Strategy (30 minutes)

**OBJECTIVE**: Based on investigation results, determine fix strategy

**DECISION TREE**:
- **If issue is in raw LLM response**: Focus on prompt engineering fixes
- **If issue is in processing**: Focus on validation/cleaning pipeline fixes  
- **If issue is random**: Focus on consistency checking and retry mechanisms
- **If issue is systematic**: Focus on structural validation improvements

**RESOLUTION IMPLEMENTATION PLACEHOLDER**:
```python
# Based on investigation results, implement appropriate fix:

# Option A: LLM Prompt Enhancement
def enhance_extraction_prompt_for_consistency():
    # Add explicit node/edge consistency requirements to prompt
    pass

# Option B: Processing Pipeline Fix  
def add_node_edge_validation():
    # Add validation step that ensures all edge references have corresponding nodes
    pass

# Option C: Automatic Healing
def create_missing_nodes_automatically():
    # Generate missing evidence nodes based on edge requirements
    pass
```

## 📊 SUCCESS CRITERIA

### **Technical Success Criteria:**
1. **Root Cause Identified**: Clear determination of where/why nodes disappear
2. **Zero Data Loss**: All extracted edges successfully load into NetworkX graph
3. **Validation Framework**: Systematic detection of node/edge inconsistencies
4. **Reproducible Quality**: Multiple runs produce consistent, complete results

### **Functional Success Criteria:**
1. **Complete Edge Loading**: 34 extracted edges → 34 loaded edges (no loss)
2. **Consistent Extraction**: Multiple runs produce equivalent graph structures
3. **Quality Monitoring**: Automated detection and reporting of data integrity issues
4. **User Experience**: Professional HTML reports with complete relationship data

---

## 🏗️ Codebase Structure

### Key Entry Points
- **`analyze_direct.py`**: Main TEXT → JSON → HTML pipeline (Phase 22A achievement)
- **`core/structured_extractor.py`**: LLM extraction interface requiring investigation
- **`core/plugins/van_evera_llm_schemas.py`**: Pydantic models for validation rules

### Critical Files for Phase 23A Investigation
- **`core/structured_extractor.py`**: Target for raw LLM response capture and processing audit
- **Output location**: `output_data/direct_extraction/` contains problematic JSON files
- **Debug location**: `debug/` directory for raw LLM response logging (to be created)

### Runtime Environment
- **Virtual Environment**: `test_env/` activated with `source test_env/bin/activate`
- **Test Input**: `/home/brian/projects/process_tracing/input_text/revolutions/french_revolution.txt`
- **Working Pipeline**: `analyze_direct.py` fully functional for testing

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
- All investigation findings must be documented in `evidence/current/Evidence_Phase23A_DataIntegrity.md`
- Raw logs and response files required for all claims
- Timeline documentation of all processing steps

### SYSTEMATIC VALIDATION
- Test each investigation step before proceeding to next
- Validate raw response capture before analyzing processing pipeline
- Prove reproducibility before implementing fixes

---

## 📁 Evidence Structure

Evidence for Phase 23A must be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase23A_DataIntegrity.md
```

**REQUIRED EVIDENCE**:
- Raw LLM response JSON files with timestamps
- Node count tracking through all processing stages  
- Reproducibility test results across multiple runs
- Exact identification of data loss location in pipeline
- Processing pipeline audit findings
- Pattern analysis results (systematic vs random occurrence)

**CRITICAL**: No resolution implementation without definitive root cause identification supported by raw evidence files.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.