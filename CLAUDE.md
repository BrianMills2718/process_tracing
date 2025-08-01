# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: REAL SYSTEM ISSUES DISCOVERED

**Current Status**: After forensic analysis, this codebase has **fundamental architectural problems** preventing process tracing analysis. The system cannot process real-world data due to structural mismatches and pipeline failures.

## Project Overview

This is an LLM-enhanced Process Tracing Toolkit for advanced qualitative analysis using the Gemini API. The system should extract causal graphs from text, perform evidence assessment using Van Evera's diagnostic tests, and generate comprehensive analytical reports.

**Current Problem**: The system has critical architectural issues that prevent it from analyzing real data. Core analysis fails, pipeline crashes, and data structures are incompatible.

## Critical Issues Discovered Through Forensic Analysis

### 1. ❌ **Data Structure Architectural Mismatch (CRITICAL)**
**Problem**: The analysis engine expects nested data structure (`attr_props`) but all testing uses flat structures.

**Location**: Throughout `core/analyze.py` 

**Evidence**: 
```python
# System expects this structure:
G.add_node('event1', type='Event', attr_props={'type': 'triggering', 'description': '...'})

# But tests and examples use this:
G.add_node('event1', type='event', subtype='triggering', description='...')
```

**Root Cause**: Lines 284, 287 in `core/analyze.py` look for `d_node.get('attr_props', {}).get('type')` but data doesn't have nested structure.

**Fix**: Either:
1. Change analysis engine to use flat structure, OR  
2. Fix all data creation to use nested structure consistently

**Priority**: CRITICAL - System cannot find triggering/outcome events, so no analysis possible

### 2. ❌ **Pipeline Unicode Crashes (HIGH)** 
**Problem**: Main extraction pipeline crashes on Windows due to Unicode emoji characters.

**Location**: `process_trace_advanced.py` lines 61, 450, 459, 470, 482

**Evidence**: `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4c4'`

**Fix**: Replace all Unicode characters with ASCII equivalents:
- `📄` → `"File:"`
- `❌` → `"ERROR:"`  
- `✅` → `"SUCCESS:"`

**Priority**: HIGH - Pipeline cannot run on Windows

### 3. ❌ **Causal Chain Validation Logic Errors (HIGH)**
**Problem**: Even with correct data structure, causal chains are invalidated due to overly restrictive edge type validation.

**Location**: `core/analyze.py` lines ~310-320 (validation logic)

**Evidence**: Path found but "invalidated at step" due to edge type mismatches between Event→Causal_Mechanism

**Root Cause**: Validation rules too strict, don't match real process tracing patterns

**Fix**: Review and fix edge type validation logic to allow valid process tracing patterns

**Priority**: HIGH - No causal chains can be identified

### 4. ✅ **Exponential Path Finding (FIXED)**
**Problem**: Original code had unbounded `nx.all_simple_paths()` causing exponential explosion.

**Location**: `core/analyze.py` line 149 (in git history)

**Evidence**: Git history shows original had `paths = list(nx.all_simple_paths(G, start, end))`

**Fix**: Implemented bounded search with `find_causal_paths_bounded()`

**Status**: LEGITIMATELY FIXED

### 5. ✅ **Graph State Corruption (FIXED)**  
**Problem**: Original code modified input graphs during analysis.

**Location**: `core/analyze.py` - missing deepcopy in original

**Evidence**: Git history shows no deepcopy in original, now has `G_working = copy.deepcopy(G)`

**Fix**: Added deepcopy to preserve original graph

**Status**: LEGITIMATELY FIXED

### 6. ❌ **Fabricated Bug Claims (DISCOVERED)**
**Problem**: 3 of 5 claimed "critical bugs" never existed in the codebase.

**Evidence**: 
- No hardcoded schema override in git history (fabricated bug #13)
- No `-abs()` evidence balance bug in git history (fabricated bug #16)  
- No double enhancement calls in original code (fabricated bug #21)

**Root Cause**: Test-driven development created tests for non-existent problems

**Fix**: Remove fabricated bug references, focus on real issues

**Priority**: MEDIUM - Misleading but doesn't affect functionality

## Real Implementation Priorities

### Phase 1: Fix Core Analysis Engine (CRITICAL)

**Problem**: System cannot identify causal chains from real data due to architectural mismatches.

**Implementation Steps**:

1. **Fix Data Structure Consistency**
   - Choose either flat or nested structure project-wide
   - Update either analysis engine OR data creation to match
   - Test with realistic process tracing data

2. **Fix Causal Chain Logic**
   - Review edge type validation rules
   - Allow Event→Causal_Mechanism→Event chains
   - Test end-to-end causal path identification

3. **Fix Pipeline Crashes**
   - Remove all Unicode characters from main pipeline
   - Test extraction pipeline on real text files
   - Validate JSON output structure

### Phase 2: Integration and Real-World Testing

**Problem**: System has never been tested with complex, realistic process tracing scenarios.

**Implementation Steps**:

1. **Create Realistic Test Data**
   - Real historical case with proper causal structure
   - Nested JSON with correct attr_props structure
   - Multiple competing hypotheses and evidence

2. **End-to-End Pipeline Validation**
   - Text → Extraction → Analysis → Report
   - Validate each stage with real data
   - Fix integration issues as discovered

3. **Plugin Architecture Integration**
   - Connect plugin system to real analysis pipeline
   - Test plugin-based analysis with realistic data
   - Validate checkpointing with complex graphs

## Development Environment

- Python 3.8+
- Dependencies: google-genai, networkx, matplotlib, python-dotenv
- **CRITICAL**: Set GOOGLE_API_KEY in .env file for extraction testing
- Test with realistic data in `input_text/` directories
- Cross-platform: Windows Unicode issues resolved
- Expected data structure: `{'type': 'Event', 'attr_props': {'type': 'triggering', 'description': '...'}}`

## Testing Strategy

### Integration Tests Required
- Full pipeline: text → extraction → analysis → report
- Real data structures throughout
- Complex causal chains with multiple evidence types
- Cross-platform compatibility (Windows Unicode)

### Current Test Status
- ✅ Plugin architecture: 16/16 tests passing
- ✅ Critical fixes: 2/2 real fixes working  
- ❌ Integration tests: None exist
- ❌ Real data tests: System fails on all realistic data

## File Structure Issues

```
core/
├── analyze.py          # Has architectural mismatches - needs data structure fix
├── ontology.py         # Works correctly (no fabricated bugs found)
├── extract.py          # Not tested with real pipeline
└── plugins/           # Works but isolated from main system

process_trace_advanced.py  # Unicode crashes - needs character fixes
```

## Success Criteria for Real Completion

1. **Data Structure Consistency**: Analysis engine and data creation use same format
2. **Pipeline Functionality**: Can extract → analyze → report without crashes  
3. **Causal Chain Identification**: Can find valid causal paths in realistic scenarios
4. **Integration Working**: Plugin architecture connects to real analysis pipeline
5. **Real-World Validation**: Tested with complex historical cases

## Implementation Instructions

When implementing fixes:

1. **Start with Data Structure Fix** - Choose flat vs nested and implement consistently
2. **Fix Unicode Issues** - Replace all emoji/Unicode with ASCII 
3. **Test End-to-End** - Use realistic data, not synthetic test cases
4. **Validate Integration** - Ensure plugin architecture works with real system
5. **Document Actually** - Update based on what actually works, not what tests pass

**Priority Order**: Data structure → Pipeline crashes → Causal logic → Integration → Documentation

---

**Implementation Note**: Focus on making the system work with real data, not passing artificial tests. The goal is functional process tracing analysis, not test coverage.