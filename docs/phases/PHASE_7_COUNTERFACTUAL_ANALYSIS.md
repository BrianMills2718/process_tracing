# PHASE 7: COUNTERFACTUAL ANALYSIS (MEDIUM JUICE/SQUEEZE)

**Priority**: 4 - Medium-High Impact
**Complexity**: Medium-High
**Timeline**: 3-4 weeks
**Juice/Squeeze Ratio**: 7/10 - Important for causal validity with moderate complexity

## Overview

Implement counterfactual analysis capabilities to explore alternative causal pathways, test necessity and sufficiency conditions, and perform robust causal inference through "what if" scenario analysis. This addresses causal validity by testing whether identified mechanisms are truly necessary or sufficient.

## Core Problem

Current system identifies causal pathways but doesn't test their robustness. Counterfactual analysis enables:
- **Necessity Testing**: Would outcome occur without this cause?
- **Sufficiency Testing**: Is this cause alone enough for the outcome?
- **Alternative Pathways**: What other causal routes were possible?
- **Robustness Assessment**: How sensitive are conclusions to key assumptions?

## Implementation Strategy

### Phase 7A: Counterfactual Infrastructure (Week 1-2)
**Target**: Core counterfactual reasoning and scenario generation

#### Task 1: Counterfactual Data Model
**Files**: `core/counterfactual_models.py` (new)
- Counterfactual scenario definitions
- Alternative pathway structures
- Necessity/sufficiency test frameworks
- INUS condition (Insufficient but Non-redundant parts of Unnecessary but Sufficient) analysis

#### Task 2: Scenario Generation
**Files**: `core/scenario_generator.py` (new)
- LLM-assisted counterfactual scenario creation
- Alternative pathway identification
- Plausible alternative history generation
- Constraint-based scenario validation

#### Task 3: Causal Mechanism Testing
**Files**: `core/mechanism_testing.py` (new)
- Necessity condition testing algorithms
- Sufficiency condition testing algorithms
- INUS condition analysis
- Mechanism robustness assessment

### Phase 7B: Alternative Pathway Analysis (Week 2-3)
**Target**: Systematic analysis of alternative causal routes

#### Task 4: Alternative Pathway Detection
**Files**: `core/alternative_pathways.py` (new)
- Identify potential alternative causal routes
- Pathway plausibility assessment
- Historical alternative analysis
- Critical juncture alternative outcomes

#### Task 5: Pathway Competition Analysis
**Files**: `core/pathway_competition.py` (new)
- Compare actual vs alternative pathways
- Assess relative pathway strength
- Identify critical pathway decision points
- Pathway selection factor analysis

#### Task 6: Scope Condition Testing
**Files**: `core/scope_testing.py` (new)
- Test mechanism performance under different conditions
- Boundary condition identification
- Context sensitivity analysis
- Generalizability assessment through counterfactuals

### Phase 7C: Counterfactual Reasoning & Validation (Week 3-4)
**Target**: Advanced counterfactual inference and validation

#### Task 7: Counterfactual Inference Engine
**Files**: `core/counterfactual_inference.py` (new)
- Systematic counterfactual reasoning algorithms
- Causal strength assessment through counterfactuals
- Multiple counterfactual scenario integration
- Confidence assessment for counterfactual claims

#### Task 8: Historical Plausibility Assessment
**Files**: `core/plausibility_assessment.py` (new)
- Evaluate plausibility of counterfactual scenarios
- Historical constraint checking
- Actor capability and motivation analysis
- Structural constraint evaluation

### Phase 7D: Integration & Visualization (Week 4)
**Target**: Counterfactual visualization and pipeline integration

#### Task 9: Counterfactual Visualization
**Files**: `core/counterfactual_viz.py` (new)
- Alternative pathway visualization
- Scenario comparison charts
- Necessity/sufficiency test results
- Interactive counterfactual exploration

#### Task 10: Pipeline Integration
**Files**: `process_trace_counterfactual.py` (new), modify main pipeline
- Counterfactual analysis workflow
- Integration with temporal and Bayesian analysis
- Counterfactual HTML dashboard generation
- Quality gates for counterfactual validity

## Technical Implementation

### Counterfactual Data Structures
```python
@dataclass
class CounterfactualScenario:
    scenario_id: str
    description: str
    modified_conditions: List[str]
    alternative_pathway: List[CausalStep]
    plausibility_score: float
    evidence_requirements: List[str]
    outcome_prediction: str
    confidence_level: float

@dataclass
class NecessityTest:
    mechanism_id: str
    test_description: str
    removal_scenario: CounterfactualScenario
    outcome_change: bool
    necessity_strength: float
    supporting_evidence: List[str]
    confidence: float

@dataclass
class SufficiencyTest:
    mechanism_id: str
    test_description: str
    isolation_scenario: CounterfactualScenario
    outcome_achieved: bool
    sufficiency_strength: float
    additional_conditions: List[str]
    confidence: float

@dataclass
class INUSAnalysis:
    condition_set: List[str]
    insufficiency_evidence: List[str]
    non_redundancy_evidence: List[str]
    unnecessary_evidence: List[str]
    sufficient_combinations: List[List[str]]
    inus_score: float
```

### LLM Counterfactual Analysis Prompt
```
Perform counterfactual analysis for the following causal mechanism:

1. NECESSITY TESTING:
   - If this mechanism were absent, would the outcome still occur?
   - What alternative pathways might have led to the same outcome?
   - How critical was this mechanism for the outcome?

2. SUFFICIENCY TESTING:
   - Is this mechanism alone sufficient for the outcome?
   - What additional conditions were required?
   - Could this mechanism have failed to produce the outcome?

3. ALTERNATIVE SCENARIOS:
   - What plausible alternative histories were possible?
   - At what critical junctures could events have gone differently?
   - How would outcomes have changed under alternative conditions?

4. ROBUSTNESS ASSESSMENT:
   - How sensitive is the causal conclusion to key assumptions?
   - What evidence would be needed to support alternative explanations?
   - How confident can we be in the identified causal pathway?

Generate structured counterfactual analysis with plausibility assessments.
```

## Success Criteria

### Functional Requirements
- **Scenario Generation**: Create plausible counterfactual scenarios
- **Necessity Testing**: Systematic assessment of mechanism necessity
- **Sufficiency Testing**: Evaluation of mechanism sufficiency
- **Alternative Pathways**: Identification of alternative causal routes
- **Visualization**: Interactive counterfactual exploration dashboards

### Performance Requirements
- **Analysis Speed**: <15s for counterfactual analysis of mechanism with 10 nodes
- **Scenario Generation**: <30s for generating 5 alternative scenarios
- **Memory Usage**: <300MB additional for counterfactual data structures
- **Scalability**: Support counterfactual analysis for graphs up to 100 nodes

### Quality Requirements
- **Plausibility**: Counterfactual scenarios must be historically/logically plausible
- **Completeness**: Cover major alternative pathways and critical junctures
- **Validity**: Necessity/sufficiency tests must be methodologically sound
- **Consistency**: Counterfactual reasoning must be internally consistent

## Testing Strategy

### Unit Tests
- Counterfactual scenario generation logic
- Necessity/sufficiency testing algorithms
- Alternative pathway detection
- Plausibility assessment functions

### Integration Tests
- Full counterfactual pipeline with real cases
- Integration with temporal analysis for historical constraints
- Integration with Bayesian analysis for probability assessment
- HTML dashboard with counterfactual sections

### Validation Tests
- Replication of known counterfactual studies
- Expert validation of scenario plausibility
- Comparison with manual counterfactual analysis
- Cross-validation across different case types

## Expected Benefits

### Research Value
- **Causal Validity**: Rigorous testing of causal necessity and sufficiency
- **Robustness**: Assessment of conclusion sensitivity to assumptions
- **Alternative Awareness**: Systematic consideration of alternative explanations
- **Methodological Completeness**: Addresses key process tracing validation requirements

### User Benefits
- **Confidence**: Understanding of causal mechanism robustness
- **Insight**: Discovery of hidden alternative pathways
- **Validation**: Systematic testing of causal claims
- **Decision Making**: Better understanding of scenario sensitivity

## Integration Points

### Existing System
- Builds on temporal analysis for historically constrained scenarios
- Utilizes Bayesian analysis for counterfactual probability assessment
- Extends Van Evera framework with counterfactual evidence evaluation
- Integrates with comparative analysis for cross-case counterfactual validation

### Future Phases
- **Phase 8**: Quantitative integration uses counterfactual analysis for statistical validation
- **Phase 9**: Network analysis incorporates counterfactual pathway analysis
- **Phase 10**: Mixed methods integration benefits from counterfactual robustness testing

## Risk Assessment

### Technical Risks
- **Complexity**: Counterfactual reasoning is computationally and logically complex
- **Validation**: Difficult to validate counterfactual scenarios empirically
- **Combinatorial Explosion**: Number of possible scenarios grows exponentially

### Methodological Risks
- **Speculation**: Counterfactual analysis may become purely speculative
- **Bias**: Analyst preferences may influence scenario generation
- **Overconfidence**: May provide false certainty about alternative outcomes

### Mitigation Strategies
- Systematic plausibility constraints based on historical/logical limitations
- Multiple independent scenario generation with convergence testing
- Expert validation interfaces for scenario assessment
- Confidence scoring and uncertainty quantification for all counterfactual claims
- Integration with empirical evidence to constrain speculation

## Deliverables

1. **Counterfactual Infrastructure**: Core scenario generation and reasoning engine
2. **Necessity/Sufficiency Testing**: Systematic causal condition assessment
3. **Alternative Pathway Analyzer**: Identification and evaluation of alternative routes
4. **Plausibility Assessment**: Historical and logical constraint validation
5. **Counterfactual Visualizations**: Interactive scenario exploration dashboards
6. **Integrated Pipeline**: End-to-end counterfactual process tracing workflow
7. **Validation Framework**: Counterfactual analysis quality assessment
8. **Test Suite**: Comprehensive counterfactual functionality testing
9. **Documentation**: Counterfactual process tracing methodology guide

This phase enhances our toolkit with robust causal validation capabilities, enabling systematic testing of mechanism necessity, sufficiency, and alternative pathway analysis for more rigorous process tracing conclusions.