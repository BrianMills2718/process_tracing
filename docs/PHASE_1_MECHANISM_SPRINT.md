# Phase 1: Core Mechanism Analysis Sprint

## 🎯 Sprint Objective
**Goal**: Enable process tracing of causal mechanisms - the core requirement for academic process tracing methodology

**Current Problem**: System only models events and hypotheses, missing the mechanisms that connect them. Process tracing specifically requires tracing HOW causation works, not just THAT it works.

**Success Criteria**: 
- Extract Causal_Mechanism nodes from text
- Link events to mechanisms via part_of_mechanism edges
- Test mechanism operation with evidence via tests_mechanism edges
- Connect hypotheses to mechanisms via explains_mechanism edges

## 📊 Impact Assessment
**Methodology Coverage**: 60% → 85% (+25%)
**Key Academic Methods Enabled**:
- ✅ Beach & Pedersen theory-testing process tracing
- ✅ Mechanism-based causal inference
- ✅ Multi-step causal process analysis
- ✅ Van Evera tests applied to mechanisms (not just hypotheses)

## 🏗️ Implementation Tasks

### 1. Configuration Layer Updates
**File**: `config/ontology_config.json`
- ✅ Causal_Mechanism node type already configured
- ✅ part_of_mechanism, tests_mechanism, explains_mechanism edges configured
- **Action**: Verify configuration completeness

### 2. Extraction Prompt Updates
**File**: `core/extract.py`
- **Current Issue**: Prompt restricts to [Event, Hypothesis, Evidence] node types
- **Required Changes**:
  - Add Causal_Mechanism to allowed node types
  - Add mechanism edges to allowed edge types  
  - Update examples to show mechanism extraction
  - Add guidance on mechanism identification

### 3. Enhanced Mechanism Properties
**Requirements**:
- description: Step-by-step mechanism operation
- confidence: How certain we are mechanism operates
- completeness: How complete our understanding is
- status: hypothetical/supported/refuted/partial
- testable_predictions: What evidence would confirm operation

### 4. New Edge Properties
**part_of_mechanism**:
- role: trigger/intermediate/outcome/facilitating
- sequence_position: Step number in mechanism
- necessity: How necessary event is for mechanism

**tests_mechanism**:
- probative_value: Strength of mechanism test
- test_result: passed/failed/ambiguous/inconclusive
- diagnostic_type: Van Evera types for mechanisms
- mechanism_aspect: existence/operation/completeness

**explains_mechanism**:
- certainty: Confidence in explanation
- type_of_claim: existence/operation/necessity/sufficiency
- scope: Under what conditions explanation applies

## 🧪 Test Cases

### Test Case 1: American Revolution Mechanism
**Expected Mechanism**: "British Taxation → Colonial Economic Pressure → Organized Resistance → Revolutionary War"

**Expected Extraction**:
```json
{
  "id": "mechanism_taxation_resistance",
  "type": "Causal_Mechanism", 
  "properties": {
    "description": "British taxation creates economic pressure on colonists, leading them to organize resistance movements that escalate into armed conflict",
    "confidence": 0.8,
    "status": "supported"
  }
}
```

**Expected Connections**:
- Stamp Act → part_of_mechanism → Taxation Resistance Mechanism (role: trigger)
- Boston Tea Party → part_of_mechanism → Taxation Resistance Mechanism (role: intermediate)
- Lexington & Concord → part_of_mechanism → Taxation Resistance Mechanism (role: outcome)

### Test Case 2: Mechanism Testing
**Expected Evidence Testing**:
```json
{
  "type": "tests_mechanism",
  "source": "evidence_colonial_boycotts", 
  "target": "mechanism_taxation_resistance",
  "properties": {
    "test_result": "passed",
    "diagnostic_type": "smoking_gun",
    "mechanism_aspect": "operation"
  }
}
```

## 🔍 Quality Criteria

### Mechanism Identification Quality
- **Specificity**: Mechanisms describe HOW causation works, not just correlation
- **Testability**: Clear predictions about what evidence would confirm/refute
- **Completeness**: Major steps in causal process identified
- **Temporal Logic**: Proper sequence from trigger to outcome

### Connection Quality  
- **Event-Mechanism Links**: Events properly categorized by mechanism role
- **Evidence-Mechanism Tests**: Evidence tests mechanism operation, not just existence
- **Hypothesis-Mechanism Explanations**: Clear claims about how mechanisms work

### Academic Rigor
- **Van Evera Integration**: Diagnostic tests applied to mechanism components
- **Source Attribution**: Evidence for mechanism operation includes text quotes
- **Uncertainty Quantification**: Confidence levels for mechanism claims

## 📋 Definition of Done

### Technical Completion
- [ ] Causal_Mechanism nodes extracted from text
- [ ] part_of_mechanism edges connecting events to mechanisms
- [ ] tests_mechanism edges connecting evidence to mechanisms  
- [ ] explains_mechanism edges connecting hypotheses to mechanisms
- [ ] All new properties properly populated

### Quality Validation
- [ ] American Revolution case generates 2-3 mechanism nodes
- [ ] Events properly categorized by mechanism role (trigger/intermediate/outcome)
- [ ] Evidence tests mechanism operation with diagnostic types
- [ ] No self-referential or circular mechanism connections
- [ ] Mechanism descriptions explain HOW causation works

### Academic Methodology
- [ ] Beach & Pedersen theory-testing patterns supported
- [ ] Van Evera tests applicable to mechanisms
- [ ] Multi-step causal process analysis functional
- [ ] Mechanism-based hypothesis testing operational

## 🚨 Risk Mitigation

### Risk 1: LLM Struggles with Mechanism Abstraction
**Mitigation**: Provide clear examples and step-by-step guidance in prompt

### Risk 2: Over-complex Mechanism Extraction  
**Mitigation**: Start with 1-3 major mechanisms per case, focus on quality over quantity

### Risk 3: Circular Event-Mechanism References
**Mitigation**: Clear validation that mechanisms describe processes, not just event lists

### Risk 4: Poor Mechanism-Evidence Integration
**Mitigation**: Explicit examples of how evidence tests mechanism operation vs existence

## 📈 Success Metrics

### Quantitative Targets
- **Mechanism Nodes**: 2-3 per American Revolution case
- **part_of_mechanism Edges**: 8-12 (major events linked to mechanisms)
- **tests_mechanism Edges**: 4-6 (evidence testing mechanism operation)
- **explains_mechanism Edges**: 2-4 (hypotheses explaining mechanisms)

### Qualitative Assessment  
- Mechanisms describe causal processes, not just event sequences
- Evidence tests mechanism operation with proper diagnostic types
- Event roles (trigger/intermediate/outcome) properly assigned
- No circular or trivial mechanism definitions

## 🔄 Iteration Plan

### Sprint 1A: Basic Mechanism Extraction
- Add Causal_Mechanism to extraction prompt
- Test with American Revolution case
- Validate basic node generation

### Sprint 1B: Mechanism-Event Integration  
- Implement part_of_mechanism edges
- Test event role assignment
- Validate temporal sequencing

### Sprint 1C: Mechanism Testing
- Implement tests_mechanism edges
- Test evidence-mechanism connections
- Validate diagnostic type application

### Sprint 1D: Hypothesis-Mechanism Integration
- Implement explains_mechanism edges  
- Test hypothesis-mechanism theoretical connections
- Final integration testing and quality assessment