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

## 🎯 CURRENT STATUS: Phase 23B - HTML Generation Pipeline Restoration (Updated 2025-01-09)

**System Status**: **📊 DATA INTEGRITY RESOLVED - HTML GENERATION BROKEN**  
**Latest Achievement**: **Phase 23A Complete - Zero data loss achieved in extraction pipeline**  
**Current Priority**: **Restore rich HTML generation with network visualizations from working graph data**

**PHASE 23A COMPLETION RESULTS**:
- ✅ **Zero Data Loss**: NetworkX edge collapsing issue resolved (MultiDiGraph + unique keys)
- ✅ **100% Data Integrity**: French Revolution (31 edges), American Revolution (39 edges), Westminster Debate (31 edges) - all preserve complete data
- ✅ **Investigation Framework**: Raw LLM response capture, systematic debugging tools created
- ✅ **Evidence Documentation**: Complete root cause analysis with reproducible validation
- ❌ **HTML Generation**: Only basic tables generated, rich network visualizations missing

**CURRENT HTML GENERATION PROBLEM**:
- **Issue**: `from core.analyze import generate_html_report` fails due to hanging analysis module
- **Impact**: Users get basic HTML tables instead of interactive network graphs and advanced analytics
- **Root Cause**: HTML generation functions trapped in hanging `core.analyze` module
- **User Impact**: No network visualizations, limited analytical insights, poor user experience

## 🔧 PHASE 23B: HTML Generation Pipeline Restoration

### OBJECTIVE: Extract and restore rich HTML generation capabilities without fixing the hanging analysis issue

**CRITICAL HTML GENERATION PROBLEM**:
```
🌐 Generating HTML report...
⚠️  HTML generation functions not available    # ← Rich visualization lost
   Creating basic HTML report...                # ← Fallback to tables only
✅ Basic HTML report generated: [path]         # ← No network graphs
```

**AVAILABLE WORKING DATA**:
- **Complete Graph Data**: NetworkX MultiDiGraph with 100% edge preservation
- **JSON Structure**: All nodes, edges, properties, relationships intact
- **Load Pipeline**: `G, data = load_graph(json_file)` works perfectly
- **Missing Component**: Rich HTML generation with network visualizations

## 🔧 HTML RESTORATION TASKS

### TASK 1: HTML Function Extraction & Analysis (45 minutes)

**OBJECTIVE**: Locate and extract HTML generation functions from hanging analysis module

**IMPLEMENTATION STEPS**:
1. **Locate HTML Generation Code**:
   ```bash
   # Find the generate_html_report function
   grep -n "def generate_html_report" core/analyze.py
   grep -A 50 "def generate_html_report" core/analyze.py
   
   # Find all HTML-related functions
   grep -n "html\|HTML" core/analyze.py
   grep -A 10 "html\|HTML" core/analyze.py
   ```

2. **Identify Dependencies**:
   ```bash
   # Find what generate_html_report imports/uses
   grep -B 20 "def generate_html_report" core/analyze.py
   
   # Look for visualization libraries
   grep "matplotlib\|plotly\|d3\|vis\|networkx.*draw" core/analyze.py
   ```

3. **Extract to Standalone Module**:
   ```python
   # Create core/html_generator.py with extracted functions
   # Copy generate_html_report and all dependencies
   # Test import without hanging analysis module
   ```

**EVIDENCE REQUIREMENTS**:
- Location of generate_html_report function in core/analyze.py
- List of all dependencies and imports required
- Successful extraction to standalone module
- Import test without hanging issues

### TASK 2: Standalone HTML Generator Creation (60 minutes)

**OBJECTIVE**: Create working HTML generator that uses our perfect graph data

**IMPLEMENTATION TARGETS**:
1. **Core HTML Generator Module**:
   ```python
   # core/html_generator.py
   def generate_process_tracing_html(G, data, output_dir):
       # Network visualization (NetworkX + D3.js/vis.js)
       # Node/edge statistics and tables
       # Interactive features
       # Professional styling
       return html_file_path
   ```

2. **Integration with analyze_direct.py**:
   ```python
   # Replace the basic HTML fallback
   from core.html_generator import generate_process_tracing_html
   
   html_file = generate_process_tracing_html(G, data, output_dir)
   ```

3. **Network Visualization Components**:
   ```python
   # Generate interactive network graph
   def create_network_visualization(G):
       # Use networkx layout algorithms
       # Export to D3.js/vis.js format
       # Include node/edge styling based on types
   ```

**EVIDENCE REQUIREMENTS**:
- Working core/html_generator.py module
- Successful integration test with analyze_direct.py
- Generated HTML with network visualization
- No hanging or import issues

### TASK 3: Rich Analytics & Visualization Features (45 minutes)

**OBJECTIVE**: Restore advanced analytics and interactive features for process tracing analysis

**SYSTEMATIC IMPLEMENTATION**:
1. **Network Analysis Features**:
   ```python
   # Add advanced NetworkX analytics
   def analyze_graph_structure(G):
       # Centrality measures (betweenness, closeness, eigenvector)
       # Path analysis (shortest paths, connectivity)
       # Community detection
       # Node importance ranking
   ```

2. **Process Tracing Specific Analytics**:
   ```python
   # Van Evera methodology analytics
   def analyze_evidence_strength(G, data):
       # Evidence type distribution (hoop, smoking gun, etc.)
       # Hypothesis support analysis
       # Causal mechanism completeness
   ```

3. **Interactive HTML Features**:
   ```python
   # Enhanced user experience
   def create_interactive_features():
       # Clickable nodes/edges with detailed info
       # Filtering by node/edge types
       # Search functionality
       # Export capabilities
   ```

**EVIDENCE REQUIREMENTS**:
- Network analysis metrics calculated and displayed
- Process tracing specific insights included
- Interactive features working in browser
- Professional styling and usability

### TASK 4: Integration Testing & Validation (30 minutes)

**OBJECTIVE**: Verify complete HTML generation pipeline works end-to-end

**VALIDATION TESTING**:
1. **Full Pipeline Test**:
   ```bash
   # Test complete TEXT → JSON → HTML with rich output
   python analyze_direct.py input_text/revolutions/french_revolution.txt --extract-only
   python analyze_direct.py output_data/direct_extraction/[latest].json --html
   
   # Verify rich HTML generated (not basic fallback)
   ```

2. **Cross-Input Validation**:
   ```bash
   # Test with different inputs
   python analyze_direct.py input_text/american_revolution/american_revolution.txt --extract-only
   python analyze_direct.py output_data/direct_extraction/[latest].json --html
   
   # Verify consistent rich HTML generation
   ```

3. **HTML Quality Assessment**:
   ```python
   # Verify HTML contains expected components
   def validate_html_output(html_file):
       # Check for network visualization elements
       # Verify analytics tables present
       # Confirm interactive features work
       # Validate professional styling
   ```

**EVIDENCE REQUIREMENTS**:
- Complete TEXT → JSON → HTML pipeline with rich output
- Network visualizations generated for multiple inputs
- Interactive features confirmed working
- No fallback to basic HTML tables

## 📊 SUCCESS CRITERIA

### **Technical Success Criteria:**
1. **HTML Functions Extracted**: generate_html_report successfully isolated from hanging module
2. **Rich Visualization**: Interactive network graphs generated from perfect graph data
3. **Analytics Restored**: Advanced NetworkX and process tracing analytics included
4. **Integration Complete**: analyze_direct.py generates rich HTML without hanging issues

### **Functional Success Criteria:**
1. **Network Visualizations**: Interactive node/edge graphs with professional styling
2. **Advanced Analytics**: Centrality measures, path analysis, evidence strength assessment
3. **User Experience**: Clickable elements, filtering, search, professional interface
4. **Complete Pipeline**: TEXT → JSON → Rich HTML working end-to-end

---

## 🏗️ Codebase Structure

### Key Entry Points  
- **`analyze_direct.py`**: Working TEXT → JSON → HTML pipeline with basic HTML fallback
- **`core/structured_extractor.py`**: LLM extraction (Phase 23A: enhanced with raw response capture)
- **`core/analyze.py`**: Contains `load_graph()` (Phase 23A: fixed MultiDiGraph) + hanging `generate_html_report()`

### Critical Files for Phase 23B Implementation
- **`core/analyze.py`**: Target for HTML function extraction (contains `generate_html_report`)
- **`analyze_direct.py`**: Integration point for new HTML generator (replace basic fallback)
- **`core/html_generator.py`**: To be created - standalone HTML generation module
- **`debug/`**: Contains raw LLM response files for investigation support

### Working Components (Phase 23A Complete)
- **Graph Loading**: `G, data = load_graph(json_file)` works perfectly with 100% data integrity
- **NetworkX Graph**: MultiDiGraph with unique edge keys, zero data loss
- **JSON Structure**: Complete nodes/edges/properties preserved through entire pipeline
- **Extraction Debug**: Raw LLM response capture and systematic analysis tools

### Runtime Environment
- **Virtual Environment**: `test_env/` activated with `source test_env/bin/activate`
- **Test Inputs**: Multiple validated inputs in `input_text/` (French Revolution, American Revolution, Westminster Debate)
- **Output Structure**: `output_data/direct_extraction/` contains complete graph data ready for rich HTML generation

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

Evidence for Phase 23B must be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase23B_HTMLGeneration.md    # Active development  
├── completed/
│   └── Evidence_Phase23A_DataIntegrity.md     # Archived
```

**REQUIRED EVIDENCE FOR PHASE 23B**:
- Location and extraction of generate_html_report function from core/analyze.py
- Successful creation of core/html_generator.py without hanging imports
- Network visualization generation with working graph data
- Integration test results showing rich HTML output (not basic fallback)
- Cross-input validation with multiple text sources
- Interactive feature validation in browser

**CRITICAL**: All HTML generation claims must be validated with actual generated files and browser testing evidence.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.  
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.