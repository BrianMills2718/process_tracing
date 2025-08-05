# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 CURRENT STATUS: GRAPH CONNECTIVITY OPTIMIZATION

**System Status**: **85% Complete** - Advanced analytical capabilities achieved with connectivity issues  
**Current Priority**: **Graph Connectivity Enhancement** - Resolve disconnected entities problem  
**Infrastructure**: **100% Complete** - All edge/node types configured and validated
**Verified Functionality**: 
- ✅ **Edge Type Coverage**: 18/21 edge types (85.7%) - MAJOR BREAKTHROUGH achieved
- ✅ **Node Type Coverage**: 8/8 node types (100%) - Complete coverage
- ✅ **Advanced Analytics**: Alternative explanations, mechanism integration, evidence weighting
- ✅ **API Integration**: gemini-2.5-flash via .env working perfectly
- ✅ **End-to-End Pipeline**: Extraction → Analysis → HTML generation functional
- ⚠️ **Graph Connectivity**: 14% disconnection rate causing visualization fragmentation

**Current Problem**: Despite sophisticated edge type extraction, graphs have isolated nodes that should connect to main analysis, reducing analytical coherence and visualization quality.

**Immediate Goal**: Implement two-pass connectivity solution to reduce disconnection rate from 14% to <5%

## 📋 IMPLEMENTATION PLAN

### **MANDATORY REFERENCE**: Read `CONNECTIVITY_PLAN.md` for complete implementation roadmap

**Phase 1 Priority**: Implement two-pass LLM extraction to resolve graph fragmentation  

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

**Implementation Status**: Infrastructure 100% complete, analytical capabilities 85% complete - Focus on connectivity optimization

## 🎯 PHASE 1 IMPLEMENTATION TASKS: TWO-PASS CONNECTIVITY

### **Current Graph Analysis Problem**
**Evidence**: `output_data/revolutions/revolutions_20250804_072914_graph.json` shows 14% disconnection rate
**Disconnected Entities**: 
- `condition_geographic_distance` - Should `enables` colonial autonomy
- `condition_enlightenment_ideas` - Should `enables` ideological resistance  
- `actor_thomas_hutchinson` - Should `initiates` British enforcement events
- `event_stamp_act_congress_declaration` - Should connect to main causal chain

### **STEP 1: Create Connectivity Detection System**
**Objective**: Identify disconnected nodes requiring repair
**File to Create**: `core/connectivity_analysis.py`

**IMPLEMENTATION**:
```python
import networkx as nx
from typing import Dict, List, Set, Tuple

def analyze_graph_connectivity(graph_data: dict) -> dict:
    """
    Analyze connectivity issues in process tracing graph
    
    Args:
        graph_data: JSON graph with 'nodes' and 'edges' arrays
    
    Returns:
        dict: Connectivity analysis with repair recommendations
    """
    # Build NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in graph_data.get('nodes', []):
        G.add_node(node['id'], **node)
    
    # Add edges
    for edge in graph_data.get('edges', []):
        G.add_edge(edge['source_id'], edge['target_id'], **edge)
    
    # Analyze connectivity
    components = list(nx.connected_components(G))
    giant_component = max(components, key=len) if components else set()
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    
    disconnection_rate = 1 - (len(giant_component) / len(G.nodes())) if G.nodes() else 0
    
    return {
        'needs_repair': disconnection_rate > 0.1,
        'disconnection_rate': disconnection_rate,
        'total_components': len(components),
        'giant_component_size': len(giant_component),
        'isolated_nodes': isolated_nodes,
        'small_components': [comp for comp in components if len(comp) < 5 and comp != giant_component],
        'isolated_node_details': [
            {
                'id': node_id,
                'type': G.nodes[node_id].get('type', 'unknown'),
                'description': G.nodes[node_id].get('properties', {}).get('description', 'No description')[:100]
            }
            for node_id in isolated_nodes
        ]
    }

def find_isolated_nodes(graph_data: dict) -> List[dict]:
    """Extract detailed information about isolated nodes"""
    analysis = analyze_graph_connectivity(graph_data)
    return analysis['isolated_node_details']
```

### **STEP 2: Create Second-Pass Extraction Prompt**
**Objective**: Design targeted prompt for connectivity repair
**File to Modify**: `core/extract.py`

**ADD THIS FUNCTION**:
```python
def create_connectivity_repair_prompt(original_text: str, isolated_nodes: List[dict], main_graph_summary: str) -> str:
    """
    Create focused prompt for connecting isolated nodes to main graph
    """
    isolated_descriptions = "\n".join([
        f"- {node['id']} ({node['type']}): {node['description']}"
        for node in isolated_nodes
    ])
    
    return f"""CONNECTIVITY REPAIR TASK:

You previously extracted a causal graph but some important nodes are disconnected from the main analysis. Your task is to identify the missing relationships that should connect these isolated nodes to the main graph.

ISOLATED NODES NEEDING CONNECTIONS:
{isolated_descriptions}

MAIN GRAPH CONTEXT:
{main_graph_summary}

ORIGINAL TEXT:
{original_text}

Based on the original text, identify what relationships should connect these isolated nodes to the main graph. Look for:
- causes: Causal relationships between events
- enables/constrains: How conditions affect events or mechanisms  
- initiates: How actors start events or mechanisms
- supports/refutes: How evidence relates to hypotheses
- provides_evidence_for: How events/evidence support other elements
- part_of_mechanism: How events are components of mechanisms

Output ONLY the missing edges as JSON in this format:
{{
  "additional_edges": [
    {{
      "source_id": "isolated_node_id",
      "target_id": "main_graph_node_id", 
      "type": "relationship_type",
      "properties": {{
        "reasoning": "Brief explanation from text",
        "confidence": 0.8,
        "source": "connectivity_repair"
      }}
    }}
  ]
}}

Focus on high-confidence connections clearly supported by the text. Do not create speculative relationships."""

def extract_connectivity_relationships(prompt: str) -> List[dict]:
    """
    Extract additional relationships using connectivity repair prompt
    """
    try:
        response = query_gemini_structured(prompt, {"type": "object", "properties": {"additional_edges": {"type": "array"}}})
        return response.get('additional_edges', [])
    except Exception as e:
        print(f"[WARNING] Connectivity repair failed: {e}")
        return []
```

### **STEP 3: Implement Two-Pass Extraction**
**File to Modify**: `core/extract.py`

**ADD THIS FUNCTION**:
```python
def extract_causal_graph_two_pass(text: str) -> dict:
    """
    Two-pass extraction: standard extraction + connectivity repair
    """
    # Pass 1: Standard extraction
    initial_graph = extract_causal_graph(text)
    
    # Analyze connectivity
    connectivity = analyze_graph_connectivity(initial_graph)
    
    if not connectivity['needs_repair']:
        print(f"[INFO] Graph connectivity acceptable: {connectivity['disconnection_rate']:.1%} disconnection rate")
        return initial_graph
    
    print(f"[INFO] Graph needs connectivity repair: {connectivity['disconnection_rate']:.1%} disconnection rate")
    print(f"[INFO] Found {len(connectivity['isolated_nodes'])} isolated nodes")
    
    # Pass 2: Connectivity repair
    if connectivity['isolated_node_details']:
        main_graph_summary = f"Main graph has {connectivity['giant_component_size']} connected nodes including Events, Hypotheses, Evidence, and Mechanisms"
        
        repair_prompt = create_connectivity_repair_prompt(
            text, 
            connectivity['isolated_node_details'],
            main_graph_summary
        )
        
        additional_edges = extract_connectivity_relationships(repair_prompt)
        
        if additional_edges:
            print(f"[INFO] Found {len(additional_edges)} additional relationships")
            # Merge edges into original graph
            initial_graph['edges'].extend(additional_edges)
            
            # Re-analyze connectivity
            final_connectivity = analyze_graph_connectivity(initial_graph)
            print(f"[INFO] Final disconnection rate: {final_connectivity['disconnection_rate']:.1%}")
        else:
            print("[WARNING] No additional relationships found in connectivity repair")
    
    return initial_graph
```

### **STEP 4: Integration Point**
**File to Modify**: `process_trace_advanced.py`

**FIND AND REPLACE**:
```python
# OLD (around line 740):
graph_data = extract_causal_graph(text)

# NEW:
graph_data = extract_causal_graph_two_pass(text)
```

### **VALIDATION COMMANDS**
```bash
# Test on American Revolution case
python process_trace_advanced.py --project revolutions

# Check connectivity improvement
python -c "
import json
from core.connectivity_analysis import analyze_graph_connectivity
data = json.load(open('output_data/revolutions/[latest]_graph.json'))
analysis = analyze_graph_connectivity(data)
print(f'Disconnection rate: {analysis[\"disconnection_rate\"]:.1%}')
print(f'Isolated nodes: {len(analysis[\"isolated_nodes\"])}')
"
```

### **SUCCESS CRITERIA**
- **Disconnection rate**: <5% (currently 14%)
- **Giant component**: >95% of nodes (currently 85.7%)  
- **Connected components**: 1-2 (currently 7)
- **Academic validity**: All new connections traceable to original text

### **REQUIRED IMPORTS FOR IMPLEMENTATION**
**Add to `core/extract.py`**:
```python
from .connectivity_analysis import analyze_graph_connectivity
```

**Dependencies**: NetworkX (already available in project requirements)

### **QUICK STATUS CHECK**
```bash
# Verify current issue exists
python -c "
import json
data = json.load(open('output_data/revolutions/revolutions_20250804_072914_graph.json'))
print('Current disconnection issue:')
print(f'  Total nodes: {len(data[\"nodes\"])}')  
print(f'  Total edges: {len(data[\"edges\"])}')
print('Isolated node types:')
for node in data['nodes']:
    connected = any(e['source_id'] == node['id'] or e['target_id'] == node['id'] for e in data['edges'])
    if not connected:
        print(f'  - {node[\"id\"]} ({node[\"type\"]})')
"
```

## Codebase Structure

### Key Files for Implementation
- **`core/extract.py`**: Current extraction logic (TO MODIFY)
- **`core/connectivity_analysis.py`**: (TO CREATE)
- **`process_trace_advanced.py`**: Main pipeline (TO MODIFY)
- **`config/ontology_config.json`**: Edge/node type definitions (REFERENCE)

### Test Cases  
- **`input_text/revolutions/american_revolution_enhanced.txt`**: Enhanced analytical text with 18/21 edge types
- **`output_data/revolutions/revolutions_20250804_072914_graph.json`**: Current example with 14% disconnection rate

### API Integration
- **Model**: gemini-2.5-flash via `.env` file
- **Function**: `query_gemini_structured()` available in `core/extract.py`

## 🎯 NEXT IMPLEMENTER GUIDANCE

### **START HERE - Implementation Order**

1. **Verify Current Issue** - Run the quick status check above to confirm 14% disconnection rate
2. **Create `core/connectivity_analysis.py`** - Implement Step 1 code exactly as shown
3. **Modify `core/extract.py`** - Add Step 2 and Step 3 functions 
4. **Update `process_trace_advanced.py`** - Make Step 4 integration change
5. **Test and Validate** - Run validation commands to measure improvement

### **Success Definition**
**Primary**: Disconnection rate reduced from 14% to <5%
**Secondary**: Giant component >95% of nodes, <3 connected components
**Validation**: All new connections traceable to original text with confidence scores

### **If Implementation Fails**
1. Check import statements and dependencies
2. Verify Gemini API access with existing functions
3. Test connectivity analysis functions independently
4. Review debugging_disconnected_entities.md for troubleshooting

**Critical**: This is a post-processing approach that enhances existing 18/21 edge type capability rather than replacing it.

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