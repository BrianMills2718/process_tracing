# DEBUG_PHASE_1.md - Critical System Analysis & TDD Roadmap

## 🎯 DESIRED END STATE

A **LLM-Enhanced Process Tracing Toolkit** that can:

1. **Extract Meaningful Causal Graphs** from historical text with:
   - Complete event descriptions from source text (not "N/A" or "Description_Not_Found")
   - Connected causal chains from triggering to outcome events
   - Source quotes for all evidence extracted from text
   - Proper temporal sequencing of events

2. **Perform Van Evera Diagnostic Analysis** with:
   - Evidence with meaningful probative values (>0.0)
   - Proper diagnostic classifications (hoop, smoking gun, etc.)
   - Evidence-hypothesis links with source text quotes
   - Confidence assessments based on actual evidence quality

3. **Generate Professional Process Tracing Reports** with:
   - Complete causal chain narratives
   - Evidence assessment tables with source quotes
   - Mechanism completeness analysis
   - Interactive visualizations

4. **Success Metrics for End State**:
   - American Revolution test case finds ≥1 meaningful causal chain
   - Evidence probative values >0.3 (not 0.0)
   - Event descriptions extracted from source text (not generic)
   - Source quotes for ≥80% of evidence
   - Van Evera classifications based on actual evidence content

---

## 🚨 CURRENT PROBLEMS (Critical Validation Results)

### **1. EXTRACTION QUALITY CRISIS (Root Cause)**

**Problem**: System extracts graph structure but loses semantic content

**Evidence**:
```json
"description": "Description_Not_Found_For_H1"
"description": "N/A"  
"probative_value": 0.0
"source_text_quote": ""
```

**Impact**: Makes entire analysis pipeline meaningless

### **2. CAUSAL CHAIN DETECTION FAILURE**

**Problem**: American Revolution test finds 0 causal chains despite connected graph

**Evidence**:
```
DEBUG_CHAINS: Total causal_chains collected: 0
"causal_chains": []
```

**Root Cause**: Chain validation fails even when paths exist

### **3. EVIDENCE ANALYSIS DEGRADATION**

**Problem**: Van Evera analysis operates on garbage data

**Evidence**:
- Evidence with 0.0 probative value
- Missing source text quotes
- "N/A" descriptions making classification meaningless

### **4. MECHANISM ASSESSMENT FAILURE**

**Problem**: Low completeness scores indicate broken mechanisms

**Evidence**:
```json
"completeness_score": 0.3  // 30% complete
"plausibility_score": 0.1  // 10% plausible
"evidence_support_level": "none"
```

### **5. CHERRY-PICKED VALIDATION**

**Problem**: Focus on one working case (Cuban Missile Crisis) while ignoring American Revolution failure

**Impact**: False confidence in system functionality

---

## 🧪 TDD APPROACH: TEST-DRIVEN RESTORATION

### **Phase 1: Define Success Tests (Write Tests First)**

#### **Test 1: Basic Extraction Quality**
```python
def test_american_revolution_extraction_quality():
    """Test that extraction produces meaningful content from American Revolution text"""
    result = extract_graph("input_text/revolutions/american_revolution.txt")
    
    # Events must have real descriptions from text
    events = [n for n in result['nodes'] if n['type'] == 'Event']
    assert len(events) >= 8, "Should extract at least 8 events"
    
    for event in events:
        desc = event['properties']['description']
        assert desc != "N/A", f"Event {event['id']} has N/A description"
        assert "Description_Not_Found" not in desc, f"Event {event['id']} has placeholder description"
        assert len(desc) >= 20, f"Event {event['id']} description too short: {desc}"
        
    # Evidence must have source quotes
    evidence = [n for n in result['nodes'] if n['type'] == 'Evidence']
    for ev in evidence:
        desc = ev['properties']['description']
        assert desc != "N/A", f"Evidence {ev['id']} has N/A description"
        assert len(desc) >= 10, f"Evidence {ev['id']} description too short"
```

#### **Test 2: Causal Chain Detection**
```python
def test_american_revolution_causal_chains():
    """Test that analysis finds meaningful causal chains"""
    graph_file = "test_fixtures/american_revolution_graph.json"
    analysis = analyze_graph_from_file(graph_file)
    
    assert len(analysis['causal_chains']) >= 1, "Should find at least 1 causal chain"
    
    chain = analysis['causal_chains'][0]
    assert len(chain['path']) >= 5, "Chain should have at least 5 events"
    assert chain['trigger'] != chain['outcome'], "Chain should connect different events"
    
    # Chain should have meaningful descriptions
    for desc in chain['path_descriptions']:
        assert desc != "N/A", f"Chain has N/A description: {desc}"
        assert len(desc) >= 15, f"Chain description too short: {desc}"
```

#### **Test 3: Evidence Quality**
```python
def test_evidence_probative_values():
    """Test that evidence has meaningful probative values"""
    analysis = analyze_graph_from_file("test_fixtures/american_revolution_graph.json")
    
    evidence_links = analysis['evidence_hypothesis_links']
    assert len(evidence_links) >= 1, "Should find evidence-hypothesis links"
    
    for link in evidence_links:
        assert link['probative_value'] > 0.0, f"Evidence {link['evidence_id']} has 0.0 probative value"
        assert link['probative_value'] <= 1.0, f"Evidence {link['evidence_id']} has invalid probative value"
        assert len(link['source_text_quote']) > 0, f"Evidence {link['evidence_id']} missing source quote"
```

#### **Test 4: Van Evera Integration**
```python
def test_van_evera_classifications():
    """Test that Van Evera classifications are meaningful"""
    analysis = analyze_graph_from_file("test_fixtures/american_revolution_graph.json")
    
    for link in analysis['evidence_hypothesis_links']:
        # Should have valid Van Evera type
        valid_types = ['hoop', 'smoking_gun', 'straw_in_the_wind', 'doubly_decisive']
        assert link['van_evera_type'] in valid_types, f"Invalid Van Evera type: {link['van_evera_type']}"
        
        # Should have reasoning for classification
        assert len(link['van_evera_reasoning']) > 20, f"Van Evera reasoning too short for {link['evidence_id']}"
```

### **Phase 2: Run Tests (They Should All Fail Now)**

Expected results:
- ❌ All tests fail (confirming current problems)
- This gives us concrete targets to fix

### **Phase 3: Fix Code to Pass Tests (One Test at a Time)**

#### **Fix 1: Extraction Quality**
- Enhance extraction prompt to emphasize description extraction
- Add validation that descriptions come from source text
- Ensure evidence extraction includes source quotes

#### **Fix 2: Causal Chain Detection** 
- Debug why American Revolution chains aren't found
- Fix chain validation logic
- Ensure meaningful event connections

#### **Fix 3: Evidence Analysis**
- Fix evidence-hypothesis linking
- Ensure probative value calculation
- Add source quote extraction

#### **Fix 4: Van Evera Integration**
- Ensure classifications based on real evidence content
- Add reasoning for diagnostic type assignment

---

## 🔍 COMPREHENSIVE CODEBASE REVIEW

### **Pipeline Flow Analysis**

```
TEXT INPUT → EXTRACTION → GRAPH LOADING → ANALYSIS → REPORT OUTPUT
     ↓           ↓            ↓           ↓           ↓
   27KB text  JSON graph   NetworkX   Analysis    HTML report
               ↓            ↓           ↓           ↓
             BROKEN:     BROKEN:    BROKEN:    POOR OUTPUT:
           descriptions  data loss  0 chains   meaningless
           are "N/A"     flattens   detected     content
```

### **Critical Code Sections to Review**

#### **1. Extraction Pipeline (CRITICAL)**
**Files**: `process_trace_advanced.py`, `core/ontology.py`
**Issue**: Produces graph structure but loses semantic content
**Review Focus**:
- LLM prompt effectiveness for description extraction
- JSON schema requirements for content preservation  
- Post-extraction validation

#### **2. Graph Loading (`core/analyze.py:210-300`)**
**Issue**: Data structure flattening loses information
**Review Focus**:
- Node/edge attribute preservation
- Properties vs. main type handling
- Description access patterns

#### **3. Causal Chain Detection (`core/analyze.py:299-410`)**
**Issue**: Path finding works but validation fails
**Review Focus**:
- Event type detection logic
- Chain validation criteria
- Path length limitations

#### **4. Evidence Analysis (`core/enhance_evidence.py`)**
**Issue**: Operates on empty/meaningless data
**Review Focus**:
- Evidence-hypothesis linking
- Probative value calculation
- Source quote extraction

### **Data Structure Audit**

#### **Current Data Flow Problems**:

1. **Extraction Stage**:
   ```json
   // What we get (BROKEN):
   {"description": "N/A", "type": "triggering"}
   
   // What we need (FIXED):
   {"description": "French and Indian War ends with defeat of France", "type": "triggering"}
   ```

2. **Loading Stage**:
   ```python
   # Current (FLATTENS DATA):
   node_attributes.update(properties_from_json)  # Overwrites main type
   
   # Needed (PRESERVES STRUCTURE):
   # Preserve both main type and properties separately
   ```

3. **Analysis Stage**:
   ```python
   # Current (LOOKS IN WRONG PLACES):
   description = node.get('attr_props', {}).get('description')  # Returns None
   
   # Needed (MULTIPLE FALLBACKS):
   description = (node.get('properties', {}).get('description') or 
                 node.get('attr_props', {}).get('description') or
                 node.get('description'))
   ```

---

## 📋 TDD IMPLEMENTATION PLAN

### **Sprint 1: Extraction Quality (Week 1)**
1. ✅ Write extraction quality tests (Day 1)
2. ❌ Run tests - confirm they fail (Day 1)
3. 🔧 Fix extraction prompt and validation (Days 2-3)
4. ✅ Tests pass - extraction produces meaningful content (Day 4)
5. 📝 Document changes and validate (Day 5)

### **Sprint 2: Causal Chain Detection (Week 2)**
1. ✅ Write causal chain detection tests
2. ❌ Run tests - confirm they fail
3. 🔧 Fix chain detection and validation logic
4. ✅ Tests pass - chains found and validated
5. 📝 Document and validate

### **Sprint 3: Evidence & Van Evera Integration (Week 3)**
1. ✅ Write evidence quality tests
2. ❌ Run tests - confirm they fail  
3. 🔧 Fix evidence analysis and Van Evera integration
4. ✅ Tests pass - meaningful evidence analysis
5. 📝 Document and validate

### **Sprint 4: Integration & Validation (Week 4)**
1. ✅ Write end-to-end integration tests
2. 🔧 Fix remaining integration issues
3. ✅ All tests pass - system fully functional
4. 📊 Performance benchmarking
5. 📝 Final documentation and delivery

---

## 🎯 SUCCESS CRITERIA (Measurable)

### **Phase 1 Complete When**:
- ✅ American Revolution test finds ≥1 causal chain (currently 0)
- ✅ Evidence probative values >0.3 (currently 0.0)
- ✅ Event descriptions from source text (currently "N/A")
- ✅ Source quotes for ≥80% of evidence (currently 0%)
- ✅ All TDD tests pass

### **Quality Gates**:
1. **No placeholder content**: No "N/A", "Description_Not_Found" 
2. **Meaningful probative values**: Evidence >0.3, mechanisms >0.5
3. **Source attribution**: ≥80% evidence has source quotes
4. **Connected analysis**: Triggering events connect to outcomes
5. **Professional output**: Reports suitable for academic use

---

## 🚫 ANTI-PATTERNS TO AVOID

1. **Technical Implementation Over Analytical Utility**: Focus on meaningful results, not error-free code
2. **Cherry-Picked Validation**: Test on multiple cases, especially failures
3. **Premature Optimization**: Fix extraction quality before connectivity
4. **Overconfident Claims**: Validate claims with evidence
5. **Infrastructure Focus**: Prioritize analytical output quality

---

## 📊 CURRENT SYSTEM ASSESSMENT (Honest)

| Component | Status | Evidence | Priority |
|-----------|--------|----------|----------|
| Extraction Quality | 🚨 BROKEN | "N/A" descriptions, 0.0 probative values | CRITICAL |
| Causal Chains | 🚨 BROKEN | 0 chains found in American Revolution | CRITICAL |
| Evidence Analysis | 🚨 BROKEN | No source quotes, meaningless values | HIGH |
| Van Evera Integration | ⚠️ DEGRADED | Runs but on garbage data | HIGH |
| Infrastructure | ✅ WORKING | No runtime errors, generates reports | MEDIUM |
| Plugin Architecture | ✅ WORKING | 16/16 tests pass | LOW |

**Overall System Functionality**: ~30% (not 90% as previously claimed)

---

This TDD approach will ensure we build a system that actually works for process tracing analysis rather than just running without errors. The focus shifts from technical implementation to analytical utility with measurable success criteria.