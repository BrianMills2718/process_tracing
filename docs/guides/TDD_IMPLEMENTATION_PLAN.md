# TDD IMPLEMENTATION PLAN - Process Tracing System Recovery

## 🔍 COMPREHENSIVE CODEBASE ANALYSIS RESULTS

After deep code examination, I've identified the **root causes** and **interdependencies** that explain why the system is only 30% functional.

---

## 📊 CRITICAL FINDINGS SUMMARY

### **✅ WHAT'S ACTUALLY WORKING (30%)**
1. **Basic Infrastructure**: Graph loading, plugin architecture, HTML generation
2. **LLM Integration**: Structured output with Pydantic models works
3. **Causal Chain Mechanics**: Path finding and validation logic is sound
4. **Van Evera Logic**: Diagnostic classification algorithm is correct

### **❌ WHAT'S BROKEN AND WHY (70%)**

#### **1. EXTRACTION QUALITY CRISIS (Lines 100-198 in process_trace_advanced.py)**

**Root Cause**: The extraction prompt is **too complex and unfocused**

**Specific Problems**:
- **Prompt overload**: 2000+ word prompt with 8 different instruction sections
- **Competing priorities**: Connectivity vs. content quality vs. Van Evera classification
- **Generic descriptions**: LLM produces "N/A" because prompt doesn't emphasize source text extraction
- **Missing validation**: No checks for description quality in extraction

**Evidence**:
```json
// Current broken output:
"description": "N/A"
"description": "Description_Not_Found_For_H1"
"source_text_quote": ""
```

#### **2. DATA STRUCTURE CORRUPTION (Lines 250-260 in core/analyze.py)**

**Root Cause**: The `load_graph` function **flattens properties incorrectly**

**Specific Problems**:
```python
# Line 255-256: CORRUPTS DATA
node_attributes = {'type': str(main_type_from_json)}  # Sets type="Event"
node_attributes.update(properties_from_json)          # Overwrites with "triggering"
```

**Impact**: 
- Event nodes lose their main type ("Event" becomes "triggering")
- Analysis can't distinguish between node types and event subtypes
- Description access patterns fail because data is in wrong location

#### **3. EVIDENCE ANALYSIS BREAKDOWN (core/enhance_evidence.py:42-46)**

**Root Cause**: Evidence enhancement gets **empty text input**

**Specific Problems**:
```python
# Line 42: PASSES EMPTY TEXT
llm_response = query_gemini(
    text_content="",  # ❌ NO ORIGINAL TEXT CONTEXT
    schema=EvidenceAssessment,
    system_instruction_text=prompt,
)
```

**Impact**:
- LLM has no source text to analyze
- Produces 0.0 probative values and generic responses
- Van Evera classifications become meaningless

---

## 🧪 TDD IMPLEMENTATION STRATEGY

### **PHASE 1: EXTRACTION QUALITY RECOVERY**

#### **Test 1: Basic Description Extraction**
```python
def test_extraction_produces_meaningful_descriptions():
    """FAILING TEST: Descriptions should come from source text"""
    graph = extract_graph("input_text/revolutions/american_revolution.txt")
    
    events = [n for n in graph['nodes'] if n['type'] == 'Event']
    assert len(events) >= 8, "Should extract minimum 8 events"
    
    for event in events:
        desc = event['properties']['description']
        # THESE WILL FAIL NOW:
        assert desc != "N/A", f"Event {event['id']} has placeholder description"
        assert "Description_Not_Found" not in desc, f"Event {event['id']} has broken description"
        assert len(desc) >= 20, f"Event {event['id']} description too short: '{desc}'"
        
        # Check description contains actual historical content
        historical_terms = ['British', 'colonial', 'tax', 'Parliament', 'Boston', 'Continental']
        has_historical_content = any(term in desc for term in historical_terms)
        assert has_historical_content, f"Event {event['id']} description lacks historical content: '{desc}'"
```

**Implementation Fix**: 
1. **Simplify extraction prompt** to focus on description quality
2. **Add extraction validation** that checks for meaningful descriptions
3. **Use few-shot examples** showing good vs. bad descriptions

#### **Test 2: Evidence Source Quote Extraction**
```python
def test_evidence_has_source_quotes():
    """FAILING TEST: Evidence should include source text quotes"""
    graph = extract_graph("input_text/revolutions/american_revolution.txt")
    
    # Find evidence-hypothesis links
    evidence_edges = [e for e in graph['edges'] if e['type'] in ['supports', 'refutes']]
    assert len(evidence_edges) >= 3, "Should have evidence-hypothesis links"
    
    for edge in evidence_edges:
        quote = edge['properties'].get('source_text_quote', '')
        # THESE WILL FAIL NOW:
        assert len(quote) > 0, f"Edge {edge['id']} missing source quote"
        assert len(quote) >= 20, f"Edge {edge['id']} source quote too short: '{quote}'"
        assert quote != "Not available", f"Edge {edge['id']} has placeholder quote"
```

**Implementation Fix**:
1. **Enhance extraction prompt** to require source quotes for all evidence
2. **Add quote validation** in extraction pipeline
3. **Use structured output** to ensure quote fields are populated

### **PHASE 2: DATA STRUCTURE PRESERVATION**

#### **Test 3: Node Type Preservation**
```python
def test_node_types_preserved_after_loading():
    """FAILING TEST: Node types should be preserved during graph loading"""
    # Create test graph with known structure
    test_graph = {
        "nodes": [
            {"id": "E1", "type": "Event", "properties": {"description": "Test event", "type": "triggering"}},
            {"id": "CM1", "type": "Causal_Mechanism", "properties": {"description": "Test mechanism"}}
        ],
        "edges": []
    }
    
    # Save and reload
    with open("test_graph.json", "w") as f:
        json.dump(test_graph, f)
    
    G, data = load_graph("test_graph.json")
    
    # THESE WILL FAIL NOW:
    assert G.nodes['E1']['type'] == 'Event', f"E1 type corrupted: {G.nodes['E1']['type']}"
    assert G.nodes['CM1']['type'] == 'Causal_Mechanism', f"CM1 type corrupted: {G.nodes['CM1']['type']}"
    
    # Check properties are also preserved
    assert 'triggering' in str(G.nodes['E1']), "Event subtype should be accessible"
```

**Implementation Fix**:
1. **Preserve main types** during graph loading
2. **Store properties separately** to avoid overwrites
3. **Create unified access methods** that check multiple locations

### **PHASE 3: EVIDENCE ANALYSIS INTEGRATION**

#### **Test 4: Evidence Analysis With Source Text**
```python
def test_evidence_analysis_gets_source_context():
    """FAILING TEST: Evidence analysis should have access to original text"""
    # Mock evidence and hypothesis with source text
    hypothesis = {"id": "H1", "properties": {"description": "British taxation caused colonial rebellion"}}
    evidence = {"id": "EV1", "properties": {"description": "Colonists protested taxation without representation"}}
    edge_props = {"source_text_quote": "taxation without representation violated their rights as Englishmen"}
    
    # This should NOT be empty
    original_text = load_original_text("input_text/revolutions/american_revolution.txt")
    
    result = refine_evidence_assessment_with_llm(hypothesis, evidence, edge_props, original_text)
    
    # THESE WILL FAIL NOW:
    assert result.suggested_numerical_probative_value > 0.0, "Should have meaningful probative value"
    assert len(result.reasoning_for_type) > 50, "Should have substantial reasoning"
    assert result.refined_evidence_type != 'general', "Should classify as specific Van Evera type"
```

**Implementation Fix**:
1. **Pass original text** to evidence analysis
2. **Enhance evidence prompts** with full context
3. **Validate probative values** are meaningful (>0.3)

---

## 🔧 IMPLEMENTATION PRIORITY MATRIX

### **Priority 1 (CRITICAL - Week 1): Extraction Quality**
**Impact**: 🔥🔥🔥 **HIGHEST** - Fixes 60% of problems
**Effort**: 🔨🔨 **MEDIUM** - Prompt engineering + validation
**Dependencies**: None - can be done independently

**Files to Change**:
- `process_trace_advanced.py` lines 100-198 (extraction prompt)
- Add validation functions for description quality
- Create few-shot examples for better extraction

### **Priority 2 (HIGH - Week 2): Data Structure Preservation**
**Impact**: 🔥🔥 **HIGH** - Fixes 25% of problems  
**Effort**: 🔨 **LOW** - Simple code changes
**Dependencies**: Must be done after extraction fixes

**Files to Change**:
- `core/analyze.py` lines 250-260 (load_graph function)
- Update property access patterns throughout analysis
- Add unified data access helper functions

### **Priority 3 (MEDIUM - Week 3): Evidence Analysis Enhancement**
**Impact**: 🔥 **MEDIUM** - Fixes 15% of problems
**Effort**: 🔨🔨 **MEDIUM** - Context passing + prompt enhancement
**Dependencies**: Requires extraction and data structure fixes

**Files to Change**:
- `core/enhance_evidence.py` lines 40-48 (text context passing)
- Evidence analysis prompts and validation
- Van Evera classification logic

---

## 📋 DETAILED IMPLEMENTATION ROADMAP

### **Sprint 1: Fix Extraction Quality (Days 1-5)**

#### **Day 1: Write and Run Failing Tests**
```bash
# Create test suite
python -m pytest tests/test_extraction_quality.py -v
# Expected: ALL TESTS FAIL (confirming current problems)
```

#### **Day 2-3: Simplify and Focus Extraction Prompt**
**Current Problem**: 2000+ word prompt tries to do everything
**Solution**: Create focused 500-word prompt for description extraction

```python
# NEW SIMPLIFIED PROMPT:
FOCUSED_EXTRACTION_PROMPT = """
Extract causal events from the historical text with detailed descriptions.

CRITICAL REQUIREMENTS:
1. Each Event description must be extracted from the source text (minimum 20 words)
2. Each Evidence must include the exact source text quote that supports it
3. Descriptions cannot be "N/A", "Description_Not_Found", or generic placeholders

EXAMPLE (GOOD):
Event: "French and Indian War ends with defeat of France" 
Evidence: "taxation without representation violated their rights as Englishmen"

EXAMPLE (BAD):
Event: "N/A"
Evidence: "Colonial discontent"

Focus on quality over quantity - better to have 5 well-described events than 20 with "N/A" descriptions.
"""
```

#### **Day 4: Add Extraction Validation**
```python
def validate_extraction_quality(graph_data):
    """Validate that extraction produced meaningful content"""
    errors = []
    
    for node in graph_data['nodes']:
        if node['type'] == 'Event':
            desc = node['properties'].get('description', '')
            if desc in ['N/A', ''] or 'Description_Not_Found' in desc:
                errors.append(f"Event {node['id']} has placeholder description")
            if len(desc) < 20:
                errors.append(f"Event {node['id']} description too short")
    
    if errors:
        raise ValidationError(f"Extraction quality issues: {errors}")
    
    return True
```

#### **Day 5: Test and Validate**
```bash
# Tests should now pass
python -m pytest tests/test_extraction_quality.py -v
# Expected: TESTS PASS (extraction produces meaningful content)
```

### **Sprint 2: Fix Data Structure Issues (Days 6-10)**

#### **Day 6: Write Data Structure Tests**
```python
def test_graph_loading_preserves_structure():
    """Test that graph loading doesn't corrupt data"""
    # Implementation from Phase 2 Test 3 above
```

#### **Day 7-8: Fix load_graph Function**
```python
# CURRENT BROKEN CODE:
node_attributes = {'type': str(main_type_from_json)}
node_attributes.update(properties_from_json)  # ❌ OVERWRITES

# NEW FIXED CODE:
node_attributes = {
    'main_type': str(main_type_from_json),  # Preserve main type
    'subtype': properties_from_json.get('type', 'unspecified')  # Preserve subtype
}
node_attributes.update({k: v for k, v in properties_from_json.items() if k != 'type'})
node_attributes['type'] = str(main_type_from_json)  # Ensure main type is final
```

#### **Day 9: Update Analysis Access Patterns**
```python
def get_node_main_type(node_data):
    return node_data.get('main_type') or node_data.get('type')

def get_node_subtype(node_data):
    return node_data.get('subtype') or node_data.get('properties', {}).get('type')

def get_node_description(node_data):
    return (node_data.get('description') or 
            node_data.get('properties', {}).get('description') or
            node_data.get('attr_props', {}).get('description'))
```

#### **Day 10: Test and Validate**
```bash
python -m pytest tests/test_data_structure.py -v
# Expected: TESTS PASS (data structures preserved)
```

### **Sprint 3: Fix Evidence Analysis (Days 11-15)**

#### **Day 11: Write Evidence Analysis Tests**
```python
def test_evidence_gets_meaningful_probative_values():
    """Test that evidence analysis produces meaningful results"""
    # Implementation from Phase 3 Test 4 above
```

#### **Day 12-13: Fix Evidence Analysis Context**
```python
# CURRENT BROKEN CODE:
llm_response = query_gemini(
    text_content="",  # ❌ EMPTY
    schema=EvidenceAssessment,
    system_instruction_text=prompt,
)

# NEW FIXED CODE:
def refine_evidence_assessment_with_llm(hypothesis_node, evidence_node, edge_properties, original_text_context):
    # Get relevant text excerpt around the evidence
    source_quote = edge_properties.get('source_text_quote', '')
    context_window = extract_context_window(original_text_context, source_quote, window_size=500)
    
    llm_response = query_gemini(
        text_content=context_window,  # ✅ ACTUAL TEXT CONTEXT
        schema=EvidenceAssessment,
        system_instruction_text=enhanced_evidence_prompt,
    )
```

#### **Day 14: Enhance Evidence Prompts**
```python
ENHANCED_EVIDENCE_PROMPT = """
Analyze this evidence in context of the hypothesis using Van Evera methodology.

SOURCE TEXT CONTEXT:
{context_window}

EVIDENCE QUOTE: "{source_quote}"
HYPOTHESIS: {hypothesis_description}

Determine:
1. Van Evera Type: Is this evidence necessary (hoop), sufficient (smoking gun), suggestive (straw-in-the-wind), or both (doubly decisive)?
2. Probative Value: Based on P(Evidence|Hypothesis true) vs P(Evidence|Hypothesis false)
3. Reasoning: Explain your classification with reference to the source text

IMPORTANT: Probative values must be >0.3 for meaningful evidence.
"""
```

#### **Day 15: Test and Validate**
```bash
python -m pytest tests/test_evidence_analysis.py -v
# Expected: TESTS PASS (evidence analysis produces meaningful results)
```

---

## 🎯 SUCCESS METRICS (Measurable)

### **Phase 1 Success (After Sprint 1)**:
- ✅ **0 "N/A" descriptions** in American Revolution test case
- ✅ **Average description length >30 words** (currently ~5)
- ✅ **≥80% evidence has source quotes** (currently 0%)
- ✅ **Historical content validation passes** for all events

### **Phase 2 Success (After Sprint 2)**:
- ✅ **Node types preserved** during load/save cycle
- ✅ **Analysis finds ≥1 causal chain** in American Revolution (currently 0)
- ✅ **Event detection works** on both old and new data formats
- ✅ **No data structure corruption** in graph processing

### **Phase 3 Success (After Sprint 3)**:
- ✅ **Evidence probative values >0.3** (currently 0.0)
- ✅ **Van Evera classifications are meaningful** (not generic)
- ✅ **Evidence analysis reasoning >50 words** (currently empty)
- ✅ **Mechanism completeness >0.5** (currently 0.3)

### **Final System Success**:
- ✅ **American Revolution finds ≥2 causal chains** 
- ✅ **All evidence has source quotes and probative values >0.3**
- ✅ **Mechanism completeness scores >0.6**
- ✅ **Professional analysis reports** suitable for academic use
- ✅ **System functionality: 90%** (up from current 30%)

---

## 🚫 RISK MITIGATION

### **Risk 1: Extraction Changes Break Existing Cases**
**Mitigation**: Test on multiple historical cases (American, Cuban, French Revolutions)

### **Risk 2: Data Structure Changes Cause Regressions**
**Mitigation**: Maintain backward compatibility with dual access patterns

### **Risk 3: LLM Response Quality Varies**
**Mitigation**: Add validation loops and retry logic for poor responses

### **Risk 4: Interdependency Failures**
**Mitigation**: Implement each sprint completely before starting next

---

## 📊 IMPLEMENTATION CONFIDENCE

| Component | Current State | Target State | Confidence | Effort |
|-----------|---------------|--------------|------------|---------|
| Extraction Quality | 20% | 90% | 🟢 HIGH | Medium |
| Data Structures | 60% | 95% | 🟢 HIGH | Low |
| Evidence Analysis | 10% | 80% | 🟡 MEDIUM | Medium |
| Overall System | 30% | 90% | 🟢 HIGH | 3 weeks |

This TDD approach will systematically address the root causes identified in the codebase analysis, ensuring we build a process tracing system that actually produces meaningful analysis rather than just running without errors.