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

## 🎯 CURRENT STATUS: WSL Migration Complete - Windows Issues Confirmed Resolved (Updated 2025-01-09)

**System Status**: **WSL MIGRATION SUCCESSFUL - WINDOWS HANGS CONFIRMED RESOLVED**  
**Latest Achievement**: **All Windows import/Unicode hangs eliminated by WSL + Extraction phase verified working**  
**Current Priority**: **Install missing pandas dependency to complete analysis phase testing**

**PIPELINE STATUS**:
- ✅ **Extraction Phase**: **VERIFIED WORKING** - French Revolution extracted in 132.93s (39 nodes, 31 edges)
- ✅ **Pydantic Validation**: **VERIFIED WORKING** - Complete JSON with all required fields validated  
- ✅ **Windows Unicode Hang**: **CONFIRMED RESOLVED** - No `sys.stdout.reconfigure()` hang
- ✅ **Windows Import Hang**: **CONFIRMED RESOLVED** - `llm_reporting_utils.py` imports in 0.0s  
- ✅ **Plugin System**: **VERIFIED WORKING** - All plugins import successfully in 7.7s
- ⚠️ **Analysis Phase**: **BLOCKED BY PANDAS DEPENDENCY** - System ready, just needs `pip install pandas`

**WINDOWS ISSUES RESOLUTION EVIDENCE**:
- **Issue #1**: Unicode reconfiguration hang → **✅ RESOLVED** - Import debug shows normal progression
- **Issue #2**: Import hang in plugin chain → **✅ RESOLVED** - All plugins import without hang
- **Proof**: Complete extraction pipeline runs to completion (132.93s extraction time)
- **Analysis Ready**: Analysis phase reaches pandas import (not hanging), just missing dependency

---

## 🏆 PHASE 20: COMPLETED - Windows Issues Identified and Instrumentation Added

### OBJECTIVES ACHIEVED: ✅ Full instrumentation + Windows hang diagnosis + Pydantic validation fixes

**MAJOR ACCOMPLISHMENTS**:
1. **✅ Windows Unicode Hang Fixed**: Disabled problematic `sys.stdout.reconfigure()` causing infinite hang
2. **✅ Pydantic Validation Fixed**: Updated prompts with complete 8-field JSON examples, verified working
3. **✅ Comprehensive Debugging Added**: Full LLM call logging, progress tracking, diagnostic output
4. **✅ Root Cause Analysis**: Identified exact hang location in `llm_reporting_utils.py` import chain
5. **✅ WSL Migration Strategy**: Switching to Linux environment to bypass Windows-specific issues

**INSTRUMENTATION IMPLEMENTED**:
- Real-time LLM call logging with prompt/response visibility
- Progress tracking with percentage completion
- Diagnostic file generation surviving timeouts  
- Graph complexity analysis with workload prediction
- Import-level debugging with precise hang location identification

**EVIDENCE OF SUCCESS**:
- Individual component test shows perfect Pydantic validation with all 8 fields
- Debug output reveals complete JSON structure and successful schema validation
- System now reaches analysis phase (previously hung during import)
- Comprehensive diagnostic infrastructure ready for WSL testing

### NEXT PHASE: WSL Environment Testing

**RATIONALE**: The analysis subprocess runs as a complete black box. We need visibility into LLM call patterns, progress tracking, and resource usage to understand why it times out and how to optimize it.

**STRATEGIC APPROACH**: Progressive instrumentation with systematic testing at increasing complexity levels.

**EXPECTED IMPACT**: Complete visibility into analysis phase → Informed optimization decisions → Full pipeline success

## 📋 PHASE 20: Implementation Tasks

### TASK 1: Fix Subprocess Output Visibility (15 minutes)
**Purpose**: Enable real-time progress monitoring during analysis phase

**File**: `process_trace_advanced.py`
**Location**: Function `execute_single_case_processing`, around line 395-410

**Current Code** (find this):
```python
# Run analysis
print(f"[INFO] Starting analysis phase...")
analyze_cmd = [
    sys.executable, "-m", "core.analyze",
    str(graph_json_path),
    "--html",
    "--network-data", str(output_dir_for_case / f"{project_name_str}_network_data.json")
]
result = subprocess.run(analyze_cmd, capture_output=True, text=True)
```

**Replace With**:
```python
# Run analysis with real-time output visibility
print(f"[INFO] Starting analysis phase with real-time progress tracking...")
analyze_cmd = [
    sys.executable, "-m", "core.analyze",
    str(graph_json_path),
    "--html",
    "--network-data", str(output_dir_for_case / f"{project_name_str}_network_data.json")
]
# PHASE 20: Show real-time output instead of capturing
process = subprocess.Popen(analyze_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                          text=True, bufsize=1)
# Stream output line by line
for line in iter(process.stdout.readline, ''):
    if line:
        print(f"[ANALYSIS] {line.rstrip()}")
for line in iter(process.stderr.readline, ''):
    if line:
        print(f"[ANALYSIS-ERR] {line.rstrip()}")
process.wait()
result = process
```

**Validation**: Run with American Revolution and confirm you see output during analysis phase.

### TASK 2: Add Progress Logging to Analysis Functions (30 minutes)
**Purpose**: Track progress through major analysis phases

**File**: `core/analyze.py`

**Add at the top of file** (after imports):
```python
import time
from datetime import datetime

# PHASE 20: Progress tracking
class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = []
    
    def checkpoint(self, name, details=""):
        elapsed = time.time() - self.start_time
        self.checkpoints.append((name, elapsed, details))
        print(f"[PROGRESS] {elapsed:.1f}s | {name} | {details}")
        return elapsed

progress = ProgressTracker()
```

**Instrument key functions** (add to beginning of each):

1. **Function `load_graph`** (around line 50):
```python
def load_graph(graph_path):
    progress.checkpoint("load_graph", f"Loading from {graph_path}")
    # ... existing code ...
```

2. **Function `analyze_graph`** (around line 200):
```python
def analyze_graph(graph_path, output_dir=None, ...):
    progress.checkpoint("analyze_graph", f"Starting main analysis")
    # ... existing code ...
    
    # Add progress tracking for major sections
    progress.checkpoint("hypotheses_loaded", f"Found {len(hypotheses)} hypotheses")
    # ... after evidence loading ...
    progress.checkpoint("evidence_loaded", f"Found {len(evidence_edges)} evidence edges")
```

3. **Function `analyze_evidence`** (around line 400):
```python
def analyze_evidence(evidence_edges, hypotheses, graph, output_dir):
    total_pairs = len(evidence_edges) * len(hypotheses)
    progress.checkpoint("analyze_evidence", f"Processing {total_pairs} evidence-hypothesis pairs")
    processed = 0
    
    # In the main loop, add counter:
    for evidence_edge in evidence_edges:
        for hypothesis in hypotheses:
            processed += 1
            if processed % 10 == 0:  # Log every 10 pairs
                progress.checkpoint(f"evidence_progress", f"{processed}/{total_pairs} pairs ({100*processed/total_pairs:.1f}%)")
```

### TASK 3: LLM Call Instrumentation (45 minutes)
**Purpose**: Track every LLM call with timing and context

**File**: `core/analyze.py`

**Add LLM tracker class** (after ProgressTracker):
```python
# PHASE 20: LLM call tracking
class LLMCallTracker:
    def __init__(self):
        self.calls = []
        self.total_time = 0.0
        self.call_count = 0
    
    def start_call(self, function_name, input_size):
        self.call_count += 1
        print(f"[LLM-CALL-{self.call_count}] Starting: {function_name} | Input: {input_size} chars")
        return time.time()
    
    def end_call(self, start_time, function_name, success=True):
        duration = time.time() - start_time
        self.total_time += duration
        self.calls.append({
            "function": function_name,
            "duration": duration,
            "success": success,
            "cumulative_time": self.total_time
        })
        print(f"[LLM-COMPLETE-{self.call_count}] {function_name} | Duration: {duration:.2f}s | Total: {self.total_time:.2f}s")
        return duration

llm_tracker = LLMCallTracker()
```

**Instrument LLM calls** - Find and wrap these functions:

1. **`refine_evidence_assessment_with_llm`** calls:
```python
# Find lines like:
enhanced_assessment = refine_evidence_assessment_with_llm(...)

# Replace with:
llm_start = llm_tracker.start_call("refine_evidence_assessment", len(str(evidence_text)))
try:
    enhanced_assessment = refine_evidence_assessment_with_llm(...)
    llm_tracker.end_call(llm_start, "refine_evidence_assessment", success=True)
except Exception as e:
    llm_tracker.end_call(llm_start, "refine_evidence_assessment", success=False)
    raise
```

2. **`get_comprehensive_analysis`** calls (similar pattern)
3. **`enhance_hypothesis_with_llm`** calls (similar pattern)

### TASK 4: Graph Complexity Analysis (20 minutes)
**Purpose**: Predict workload before starting analysis

**File**: `core/analyze.py`

**Add after loading graph** (in `analyze_graph` function, after `graph = load_graph(...)`):
```python
# PHASE 20: Analyze graph complexity upfront
def analyze_complexity(graph):
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    evidence_nodes = [n for n in nodes if n.get('type') == 'Evidence']
    hypothesis_nodes = [n for n in nodes if n.get('type') == 'Hypothesis']
    evidence_edges = [e for e in edges if 'evidence' in e.get('type', '').lower()]
    
    estimated_llm_calls = len(evidence_nodes) * len(hypothesis_nodes)
    estimated_time = estimated_llm_calls * 3  # 3 seconds per call average
    
    print(f"""
[GRAPH-COMPLEXITY] Workload Analysis:
  Total Nodes: {len(nodes)}
  Evidence Nodes: {len(evidence_nodes)}
  Hypothesis Nodes: {len(hypothesis_nodes)}
  Evidence Edges: {len(evidence_edges)}
  Estimated LLM Calls: {estimated_llm_calls}
  Estimated Time: {estimated_time}s ({estimated_time/60:.1f} minutes)
  WARNING: {'HEAVY WORKLOAD - Consider timeout increase' if estimated_llm_calls > 50 else 'Normal workload'}
""")
    return estimated_llm_calls

estimated_calls = analyze_complexity(graph)
```

### TASK 5: Diagnostic File Output (15 minutes)
**Purpose**: Create persistent diagnostics that survive timeouts

**File**: `core/analyze.py`

**Add diagnostic logger** (after tracker classes):
```python
# PHASE 20: Persistent diagnostics
import json

class DiagnosticLogger:
    def __init__(self, output_dir):
        from pathlib import Path
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.log_file = self.output_dir / f"analysis_diagnostics_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.data = {
            "start_time": datetime.now().isoformat(),
            "progress": [],
            "llm_calls": [],
            "errors": []
        }
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"[DIAGNOSTIC] Saved to {self.log_file}")
    
    def log_progress(self, checkpoint, elapsed, details):
        self.data["progress"].append({
            "checkpoint": checkpoint,
            "elapsed": elapsed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def log_llm_call(self, function, duration, success):
        self.data["llm_calls"].append({
            "function": function,
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

# Initialize in analyze_graph function:
diagnostics = DiagnosticLogger(output_dir)
```

## 📊 TESTING PROGRESSION

### Test Level 0: Minimal Synthetic Test (2 minutes)
Create file `test_data/minimal_graph.json`:
```json
{
  "nodes": [
    {"id": "e1", "type": "Evidence", "properties": {"description": "Test evidence"}},
    {"id": "h1", "type": "Hypothesis", "properties": {"description": "Test hypothesis"}}
  ],
  "edges": [
    {"source": "e1", "target": "h1", "type": "tests_hypothesis"}
  ]
}
```

Run: `python -m core.analyze test_data/minimal_graph.json`
**Expected**: 1-2 LLM calls, completes in <10 seconds

### Test Level 1: American Revolution (5 minutes)
Run: `echo "1" | python process_trace_advanced.py`
**Expected**: See real-time progress, count exact LLM calls before timeout

### Test Level 2: Extended Timeout Test (10 minutes)
Run: `echo "1" | timeout 600 python process_trace_advanced.py`
**Expected**: May complete if given 10 minutes

### Test Level 3: French Revolution (After instrumentation)
Run with revolution file once we understand American Revolution patterns

## 🎯 SUCCESS CRITERIA

1. **See real-time output** during analysis phase (not after timeout)
2. **Count exact LLM calls** made before timeout
3. **Know progress percentage** when timeout occurs
4. **Have diagnostic JSON** file with all metrics after timeout
5. **Understand graph complexity** before analysis starts

## 📈 EXPECTED DISCOVERIES

Based on this instrumentation, we expect to find:
- American Revolution makes 50-100+ LLM calls in analysis phase
- Each call takes 2-5 seconds (sequential bottleneck)
- Specific functions consuming 80% of time
- Progress is steady but too slow for 5-minute timeout

## 🚀 NEXT STEPS AFTER INSTRUMENTATION

Once we have visibility, we can make informed decisions about:
1. **Timeout increases** - If progress is steady
2. **Parallelization** - If many independent LLM calls
3. **Selective analysis** - If some pairs are low-value
4. **Caching improvements** - If duplicate calls exist

---

## Evidence Structure

Evidence for this phase should be documented in:
```
evidence/
├── current/
│   └── Evidence_Phase20_AnalysisInstrumentation.md
```

Include:
- Raw console output showing real-time progress
- LLM call counts and timing data
- Diagnostic JSON file contents
- Graph complexity analysis results
- Specific bottleneck identification

---

## Coding Philosophy

### NO LAZY IMPLEMENTATIONS
- No mocking, stubs, or pseudo-code
- Every change must be fully functional
- Test each change before moving to next

### FAIL-FAST PRINCIPLES
- Surface errors immediately
- Don't hide failures with try/except
- Make problems visible

### EVIDENCE-BASED DEVELOPMENT
- All claims require raw console output
- Save diagnostic files as proof
- No success claims without demonstrable evidence