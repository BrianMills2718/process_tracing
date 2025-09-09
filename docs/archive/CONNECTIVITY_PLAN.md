# Process Tracing System - Connectivity Enhancement Plan

## Current Status (August 4, 2025)

### ✅ **Achievements Completed**
- **18/21 edge types (85.7% coverage)** - Major breakthrough from baseline 3/21
- **8/8 node types (100% coverage)** - Complete node type extraction
- **Sophisticated analytical framework** - Alternative explanations, mechanism integration, evidence weighting
- **Working system** - Full extraction → analysis → HTML pipeline functional

### ⚠️ **Current Issue: Graph Fragmentation**
- **Disconnection rate**: 14% (5/49 nodes isolated)
- **Connected components**: 7 (should be 1-2 for coherent analysis)
- **Giant component coverage**: 85.7% (should be >95%)

**Specific Disconnected Entities**:
- `condition_geographic_distance` - Should enable/constrain mechanisms
- `condition_enlightenment_ideas` - Should enable ideological resistance
- `actor_thomas_hutchinson` - Should initiate British enforcement events
- `event_stamp_act_congress_declaration` - Should be in main causal chain

## Solution: Two-Pass LLM Approach

### **Phase 1: Two-Pass Extraction Implementation**

#### **Step 1: Connectivity Detection Function**
**File**: `core/connectivity_analysis.py`
**Function**: 
```python
def analyze_connectivity(graph):
    components = find_connected_components(graph)
    disconnection_rate = calculate_disconnection_rate(components)
    isolated_nodes = find_isolated_nodes(graph)
    
    return {
        'needs_repair': disconnection_rate > 0.1,
        'isolated_nodes': isolated_nodes,
        'disconnection_rate': disconnection_rate,
        'giant_component_size': max(len(c) for c in components)
    }
```

#### **Step 2: Second-Pass Prompt Design**
**Target**: Create focused prompt for connectivity repair
**Template**:
```
You previously extracted this causal graph but some nodes are disconnected:

ISOLATED NODES:
{list_isolated_nodes_with_descriptions}

MAIN GRAPH CONTEXT:
{summary_of_main_graph}

ORIGINAL TEXT:
{original_text}

Based on the text, what relationships should connect these isolated nodes to the main graph?
Focus on: causes, enables, constrains, initiates, provides_evidence_for, supports, refutes

Output only the missing edges as JSON: {"additional_edges": [...]}
```

#### **Step 3: Integration Function**
**File**: `core/extract.py` enhancement
**Function**:
```python
def two_pass_extraction(text):
    # Pass 1: Standard extraction
    initial_graph = extract_causal_graph(text)
    
    # Connectivity check
    connectivity = analyze_connectivity(initial_graph)
    
    if connectivity['needs_repair']:
        # Pass 2: Targeted connectivity repair
        repair_prompt = create_connectivity_prompt(text, connectivity['isolated_nodes'], initial_graph)
        additional_edges = extract_relationships_only(repair_prompt)
        enhanced_graph = merge_edges(initial_graph, additional_edges)
        return enhanced_graph
    
    return initial_graph
```

### **Phase 2: Implementation Details**

#### **Priority 1: Core Implementation (Today)**
1. **Create `core/connectivity_analysis.py`** - Detection functions
2. **Enhance `core/extract.py`** - Add two-pass option
3. **Test on American Revolution case** - Validate effectiveness
4. **Measure improvement** - Before/after disconnection rates

#### **Priority 2: Integration (This Week)**  
1. **Update `process_trace_advanced.py`** - Enable two-pass extraction
2. **Add command-line flag** - `--two-pass` option
3. **Performance optimization** - Minimize additional token usage
4. **Error handling** - Graceful fallback if second pass fails

#### **Priority 3: Validation (Next Steps)**
1. **Test on multiple cases** - Revolutions, test_mechanism, etc.
2. **Measure success rates** - Disconnection rate improvements
3. **Academic validation** - Ensure connections are methodologically sound
4. **Documentation** - Update usage guides

### **Success Metrics**

#### **Primary Goals**
- **Disconnection rate**: 14% → <5%
- **Giant component coverage**: 85.7% → >95%
- **Connected components**: 7 → 1-2

#### **Quality Gates**
- **Relationship accuracy**: >90% of new connections should be academically valid
- **Performance**: Second pass adds <30 seconds to processing time
- **Reliability**: Success rate >80% across different text types

#### **Academic Standards**
- **Methodological compliance**: All connections follow process tracing methodology
- **Transparency**: Clear marking of first-pass vs. second-pass extractions
- **Evidence-based**: All connections traceable to original text

### **Implementation Timeline**

#### **Day 1 (Today)**
- [ ] Implement connectivity detection functions
- [ ] Create second-pass prompt template  
- [ ] Test basic two-pass extraction
- [ ] Validate on American Revolution case

#### **Day 2-3**
- [ ] Integrate into main pipeline
- [ ] Add command-line options
- [ ] Performance optimization
- [ ] Error handling and fallbacks

#### **Week 1**
- [ ] Test across multiple cases
- [ ] Measure and document improvements
- [ ] Academic validation of connections
- [ ] Update documentation

### **Fallback Strategy**

If two-pass approach doesn't achieve targets:
1. **Hybrid approach**: Two-pass + simple rule-based validation
2. **Three-pass approach**: Entities → Relationships → Connectivity
3. **Domain-specific prompts**: Different connectivity patterns for different node types

### **Expected Outcomes**

**Technical Success**: Graph connectivity suitable for academic analysis
**User Experience**: Coherent, integrated visualizations  
**Academic Value**: Process tracing methodology properly implemented
**System Maturity**: Production-ready connectivity solution

---

## Current Next Action

**Immediate**: Implement Step 1 (Connectivity Detection Function) and begin Step 2 (Second-Pass Prompt Design)

**File to modify first**: `core/connectivity_analysis.py` (create new)
**File to modify second**: `core/extract.py` (enhance existing)
**Test case**: American Revolution analysis with 14% disconnection rate

**Goal**: Reduce disconnection rate from 14% to <5% while maintaining 18/21 edge type sophistication.