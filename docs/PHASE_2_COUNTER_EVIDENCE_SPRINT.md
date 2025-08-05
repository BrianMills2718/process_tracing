# Phase 2: Counter-Evidence & Alternatives Sprint

## 🎯 Sprint Objective
**Goal**: Enable systematic testing of competing explanations and counter-evidence patterns - essential for rigorous hypothesis testing

**Current Problem**: System only supports confirming evidence. Academic process tracing requires testing alternative explanations and systematically evaluating disconfirming evidence.

**Success Criteria**:
- Extract Alternative_Explanation nodes for competing theories
- Generate refutes edges for counter-evidence
- Generate disproves_occurrence edges for non-events
- Support systematic alternative explanation testing

## 📊 Impact Assessment
**Methodology Coverage**: 85% → 95% (+10%)
**Key Academic Methods Enabled**:
- ✅ George & Bennett congruence method (confirming vs disconfirming evidence)
- ✅ Explaining-outcome process tracing (systematic alternative elimination)
- ✅ Van Evera failed hoop tests (necessary conditions absent)
- ✅ Comparative hypothesis testing with evidence weighting

## 🏗️ Implementation Tasks

### 1. New Node Type: Alternative_Explanation
**Properties**:
- description: Alternative causal explanation
- probability: Estimated likelihood (0.0-1.0)
- status: active/eliminated/supported/undetermined
- key_predictions: Distinguishing implications that separate from main hypotheses

**Purpose**: Represent competing explanations that must be systematically tested against primary hypotheses

### 2. Counter-Evidence Edge Types

**refutes**:
- Domain: [Evidence, Event] → Range: [Hypothesis, Causal_Mechanism, Alternative_Explanation]
- Properties: probative_value, diagnostic_type, refutation_strength
- Purpose: Evidence that contradicts or undermines claims

**disproves_occurrence**:
- Domain: [Evidence] → Range: [Event]  
- Properties: certainty, source_text_quote, diagnostic_type
- Purpose: Evidence showing events did NOT happen as claimed

**supports_alternative / refutes_alternative**:
- Domain: [Evidence] → Range: [Alternative_Explanation]
- Properties: probative_value, diagnostic_type, comparative_strength
- Purpose: Evidence for/against competing explanations

### 3. Enhanced Van Evera Integration
**Failed Hoop Tests**: Evidence that necessary conditions are absent
**Failed Smoking Gun**: Evidence that sufficient conditions don't confirm
**Counter-Diagnostic Types**: Systematic application to refutation patterns

## 🧪 Test Cases

### Test Case 1: American Revolution Alternative Explanations
**Primary Hypothesis**: "Taxation without representation caused revolution"
**Alternative 1**: "Economic interests drove revolution (merchants avoiding taxes)"
**Alternative 2**: "Elite manipulation of popular sentiment"
**Alternative 3**: "External French influence and support"

**Expected Extraction**:
```json
{
  "id": "alt_economic_interests",
  "type": "Alternative_Explanation",
  "properties": {
    "description": "Revolution primarily driven by merchant class economic interests rather than constitutional principles",
    "probability": 0.3,
    "status": "active",
    "key_predictions": ["Merchant leadership disproportionate", "Focus on trade policy over representation"]
  }
}
```

### Test Case 2: Counter-Evidence Testing
**Expected Refutation**:
```json
{
  "type": "refutes",
  "source": "evidence_working_class_participation",
  "target": "alt_economic_interests", 
  "properties": {
    "diagnostic_type": "hoop",
    "probative_value": 0.8,
    "refutation_strength": "strong"
  }
}
```

### Test Case 3: Non-Event Evidence
**Expected Disproof**:
```json
{
  "type": "disproves_occurrence",
  "source": "evidence_no_french_support_early",
  "target": "event_early_french_alliance",
  "properties": {
    "diagnostic_type": "smoking_gun", 
    "certainty": 0.9
  }
}
```

## 🔍 Quality Criteria

### Alternative Explanation Quality
- **Distinctiveness**: Alternatives make different predictions than primary hypotheses
- **Completeness**: Cover major competing explanations in literature
- **Testability**: Clear implications that can be tested with available evidence
- **Plausibility**: Reasonable alternatives, not strawmen

### Counter-Evidence Quality
- **Systematic**: Not just cherry-picked disconfirming evidence
- **Diagnostic**: Proper Van Evera typing for failed tests
- **Proportional**: Refutation strength matches evidence quality
- **Comprehensive**: Tests both existence and operation claims

### Comparative Analysis
- **Evidence Weighting**: Systematic comparison of confirming vs disconfirming evidence
- **Alternative Ranking**: Relative probability assessments
- **Elimination Logic**: Clear reasoning for alternative elimination

## 📋 Definition of Done

### Technical Completion
- [ ] Alternative_Explanation nodes extracted from text
- [ ] refutes edges connecting counter-evidence to claims
- [ ] disproves_occurrence edges for non-events
- [ ] supports_alternative/refutes_alternative edges
- [ ] All counter-evidence properties properly populated

### Quality Validation
- [ ] American Revolution case generates 2-3 alternative explanations
- [ ] Counter-evidence patterns properly typed with Van Evera diagnostics
- [ ] Failed hoop tests and absent necessary conditions identified
- [ ] Alternative explanations systematically tested against evidence
- [ ] No trivial or strawman alternatives

### Academic Methodology
- [ ] George & Bennett congruence method supported
- [ ] Explaining-outcome process tracing functional
- [ ] Comparative hypothesis testing operational
- [ ] Systematic alternative elimination logic

## 🧪 Extended Test Cases

### American Revolution Counter-Evidence Examples

**Failed Hoop Test**:
- Hypothesis: "Pure constitutional principles drove revolution"
- Necessary Condition: "Consistent principled opposition to all taxation"  
- Counter-Evidence: "Colonists accepted internal taxes, only opposed external"
- Result: Hypothesis fails hoop test

**Alternative Support**:
- Alternative: "Economic interests primary"
- Supporting Evidence: "Merchant class leadership in resistance movements"
- Refuting Evidence: "Working class and farmers also participated significantly"
- Result: Mixed support, requires nuanced assessment

**Non-Event Evidence**:
- Claimed Event: "Early coordinated colonial rebellion plan"
- Counter-Evidence: "No evidence of inter-colonial coordination before 1774"
- Result: Event disproved, spontaneous resistance model supported

## 🚨 Risk Mitigation

### Risk 1: Over-Emphasis on Counter-Evidence
**Mitigation**: Balance confirming and disconfirming evidence proportionally

### Risk 2: Trivial Alternative Explanations
**Mitigation**: Focus on alternatives actually discussed in historical literature

### Risk 3: Complex Comparative Logic
**Mitigation**: Start with binary comparisons, build toward multi-alternative assessment

### Risk 4: Evidence Double-Counting
**Mitigation**: Clear validation that evidence tests specific claims, not general support

## 📈 Success Metrics

### Quantitative Targets
- **Alternative_Explanation Nodes**: 2-3 per case
- **refutes Edges**: 3-5 per case (systematic counter-evidence)
- **disproves_occurrence Edges**: 1-2 per case (non-events identified)
- **Alternative Testing Edges**: 4-6 per case (evidence for/against alternatives)

### Qualitative Assessment
- Alternatives represent genuine scholarly debates
- Counter-evidence properly diagnostic-typed
- Failed tests properly identified and reasoned
- Comparative analysis supports systematic alternative evaluation

## 🔄 Implementation Sequence

### Sprint 2A: Alternative Explanation Modeling
- Add Alternative_Explanation nodes to extraction
- Test with American Revolution competing theories
- Validate alternative distinctiveness and plausibility

### Sprint 2B: Basic Counter-Evidence
- Implement refutes edges
- Test failed hoop and smoking gun patterns
- Validate Van Evera diagnostic integration

### Sprint 2C: Non-Event Evidence
- Implement disproves_occurrence edges  
- Test absence of claimed events
- Validate negative evidence patterns

### Sprint 2D: Comparative Analysis Integration
- Implement alternative testing edges
- Test systematic alternative evaluation
- Final integration and comparative analysis validation