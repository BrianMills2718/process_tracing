# Evidence Phase 4B: Discovery of Evaluation Points

## Discovery Date: 2025-01-29

## Main Evaluation Loop Found

**Primary Location**: Lines 911-997 in `execute_analysis()` function
- Nested loop structure: 
  - Outer: `for evidence_id, evidence_node_data in evidence_nodes_data.items()` (line 911)
  - Inner: `for u_ev_id, v_hyp_id, edge_data in G.out_edges(evidence_id, data=True)` (line 927)
- This processes existing edges between evidence and hypotheses

## Individual Evaluation Points

### Error Handling Paths
1. **Line 950-957**: Invalid probative value conversion
   - Uses `get_comprehensive_analysis()` for error recovery
   - Context: "Evidence validation - conversion error"

2. **Line 960-967**: Missing probative value
   - Uses `get_comprehensive_analysis()` for missing values
   - Context: "Evidence validation - missing value"

3. **Line 973-980**: Negative probative value clamping
   - Uses `get_comprehensive_analysis()` for negative values
   - Context: "Evidence validation - negative value clamping"

4. **Line 1135-1144**: Evidence demotion adjustment
   - Uses `get_comprehensive_analysis()` for demoted evidence
   - Context: "Evidence demotion due to lack of differentiation"

### Evidence Strength Assessment
5. **Line 1226-1232**: Supporting evidence strength
   - Uses `get_comprehensive_analysis()` for each supporting evidence
   - Context: "Determining evidence strength for hypothesis support"

6. **Line 1239-1245**: Refuting evidence strength
   - Uses `get_comprehensive_analysis()` for each refuting evidence
   - Context: "Determining evidence strength for hypothesis refutation"

### Actor Relevance
7. **Line 1468-1472**: Actor-node relevance scoring
   - Uses `get_comprehensive_analysis()` for actor involvement
   - Context: "Assessing actor relevance to {node_type}"

## Key Findings

**CRITICAL ISSUE**: The main loop at line 927 iterates over EXISTING edges in the graph.
- This means evidence-hypothesis relationships must already exist
- The code is updating existing edges, not creating new ones
- **We need to find where edges are initially created**

## Search for Edge Creation

```bash
grep -n "add_edge" core/analyze.py
```

No results found for edge creation in analyze.py!

This suggests edges are created elsewhere (possibly in plugins or during graph loading).

## Recommendation

The batching integration needs to happen at a different point - where evidence is first evaluated against hypotheses, not where existing edges are processed. We may need to:
1. Find where the graph is initially constructed
2. OR add a new evaluation phase before the main loop
3. OR modify the plugin system that creates these edges