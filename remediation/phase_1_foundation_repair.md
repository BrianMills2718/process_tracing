# Phase 1: Foundation Repair
*Fix the 5 most critical bugs preventing basic functionality*

## Critical Fixes Required

### 1. Schema Override Bug (#13)
**Problem**: Hardcoded schema overrides config file  
**Fix**: Delete lines 15-278 in `core/ontology.py`
**Test**: `test_ontology_loads_from_config_only()`

### 2. Evidence Balance Math Error (#16)
**Problem**: `balance_effect = -abs(probative_value)` always negative  
**Fix**: Change to `balance_effect = probative_value`
**Test**: `test_positive_evidence_increases_balance()`

### 3. Graph State Corruption (#34)
**Problem**: Original graph modified during analysis  
**Fix**: `graph_copy = copy.deepcopy(graph)` before modifications
**Test**: `test_graph_immutable_during_analysis()`

### 4. Path Finding Hangs (#18)
**Problem**: `nx.all_simple_paths()` exponential complexity  
**Fix**: Limit to max 100 paths or 10 depth
**Test**: `test_path_finding_completes_quickly()`

### 5. Double Processing Bug (#21)
**Problem**: Enhancement runs twice, corrupting data  
**Fix**: Remove second call at line 1326
**Test**: `test_enhancement_runs_once()`

## Success Criteria
- System can load a graph from JSON
- Basic math calculations are correct
- Analysis completes without hanging
- Original data preserved

## Next Steps
When these 5 fixes pass tests, proceed to remaining Phase 1 issues in the full document.