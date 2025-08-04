# Phase 2: Scientific Methodology Compliance (Months 5-12)
*Make the system scientifically valid for process tracing*

## Overview
With basic functionality restored in Phase 1 and plugin architecture established in Phase 1.5, this phase implements proper process tracing methodology according to Van Evera standards using the plugin framework. The system currently uses arithmetic scoring instead of logical inference rules.

## Prerequisites
- All Phase 1 fixes complete and tested
- Plugin architecture fully implemented (Phase 1.5)
- Special consideration plugins operational:
  - TokenLimitPlugin enforcing 1M token limit
  - FloatingPointPlugin with ε=1e-9
  - InteractiveVisualizationPlugin with Qt5Agg
  - CrossPlatformIOPlugin for consistent I/O
- Basic mathematical operations correct
- Graph processing functional
- Memory leaks resolved

## Month 5-6: Van Evera Test Logic (Category D1)

### Priority 1: Implement Proper Test Logic as Plugins

#### Issue #2: Van Evera Test Logic Not Implemented
**Current State**: Uses arithmetic balance scores instead of logical operations
**Plugin Solution**: Create diagnostic test plugins with proper logic

```python
# core/plugins/diagnostic_tests/hoop_test.py
class HoopTestPlugin(ProcessTracingPlugin):
    """Implements Van Evera hoop test logic"""
    plugin_id = "hoop_test"
    
    def validate_input(self, data):
        required = ['hypotheses', 'evidence', 'test_conditions']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return True
    
    def execute(self, data):
        results = []
        for hypothesis in data['hypotheses']:
            test_result = self._apply_hoop_test(
                hypothesis, 
                data['evidence'],
                data['test_conditions']
            )
            if test_result == "failed":
                hypothesis['status'] = "eliminated"
                hypothesis['elimination_reason'] = "Failed necessary condition"
                self.logger.info(f"Hypothesis {hypothesis['id']} ELIMINATED by hoop test")
            results.append(hypothesis)
        
        return {'hypotheses': results, 'test_type': 'hoop'}
    
    def _apply_hoop_test(self, hypothesis, evidence, conditions):
        """Apply necessary condition logic"""
        # Real Van Evera logic, not arithmetic
        for condition in conditions:
            if not self._condition_met(hypothesis, evidence, condition):
                return "failed"
        return "passed"
```

**Test**: `test_hoop_test_plugin_elimination_logic()`

#### Issue #68: Van Evera Classification Post-Hoc
**Current State**: Tests classified after seeing evidence
**Location**: `core/enhance_evidence.py` lines 29-30
**Fix**: Require test design BEFORE evidence evaluation
**Test**: `test_test_design_precedes_evaluation()`

#### Issue #69: Bayesian Probability Conflation
**Current State**: Mixes qualitative likelihoods with quantitative Bayes factors
**Location**: `core/enhance_evidence.py` lines 32-35
**Fix**: Separate qualitative assessment from Bayesian calculations
**Test**: `test_bayesian_calculation_separation()`

### Priority 2: Evidence Assessment Reform

#### Issue #70: Evidence Independence Assumption
**Current State**: Treats all evidence as independent
**Fix**: Implement evidence clustering and dependency detection
**Test**: `test_evidence_independence_validation()`

#### Issue #75: Evidence Source Circular Reference
**Current State**: Uses source quotes to validate evidence from same source
**Fix**: Require independent validation sources
**Test**: `test_evidence_validation_independence()`

## Month 7-8: Process Tracing Pipeline (Category D2)

### Priority 3: Complete Workflow Implementation via Plugin DAG

#### Issue #1: Incomplete Process Tracing Pipeline
**Current State**: Missing hypothesis testing phase entirely
**Plugin Solution**: Implement complete workflow as plugin DAG

```python
# core/workflows/van_evera_workflow.py
class VanEveraWorkflow:
    """Complete Van Evera process tracing workflow"""
    
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        self.workflow_definition = {
            "nodes": [
                {"id": "extract", "plugin": "TextExtractionPlugin"},
                {"id": "graph", "plugin": "GraphBuilderPlugin"},
                {"id": "hypotheses", "plugin": "HypothesisGeneratorPlugin"},
                {"id": "test_design", "plugin": "TestDesignPlugin"},
                {"id": "hoop", "plugin": "HoopTestPlugin"},
                {"id": "smoking_gun", "plugin": "SmokingGunPlugin"},
                {"id": "straw", "plugin": "StrawInWindPlugin"},
                {"id": "doubly", "plugin": "DoublyDecisivePlugin"},
                {"id": "bayesian", "plugin": "BayesianUpdatePlugin"},
                {"id": "comparison", "plugin": "HypothesisComparisonPlugin"},
                {"id": "counterfactual", "plugin": "CounterfactualPlugin"},
                {"id": "synthesis", "plugin": "SynthesisPlugin"},
                {"id": "report", "plugin": "ReportGeneratorPlugin"}
            ],
            "edges": [
                {"from": "extract", "to": "graph"},
                {"from": "graph", "to": "hypotheses"},
                {"from": "hypotheses", "to": "test_design"},
                {"from": "test_design", "to": "hoop"},
                {"from": "hoop", "to": "smoking_gun"},
                {"from": "smoking_gun", "to": "straw"},
                {"from": "straw", "to": "doubly"},
                {"from": "doubly", "to": "bayesian"},
                {"from": "bayesian", "to": "comparison"},
                {"from": "comparison", "to": "counterfactual"},
                {"from": "counterfactual", "to": "synthesis"},
                {"from": "synthesis", "to": "report"}
            ]
        }
    
    def execute(self, text, checkpoint_manager):
        """Execute complete workflow with checkpointing"""
        data = {"text": text}
        
        for node in self.workflow_definition["nodes"]:
            plugin_id = node["plugin"]
            
            # Check if we can resume from checkpoint
            if checkpoint_manager.can_resume_from(plugin_id):
                data = checkpoint_manager.load_checkpoint(plugin_id)
                self.logger.info(f"Resumed from checkpoint: {plugin_id}")
                continue
            
            # Execute plugin
            plugin = self.plugin_manager.create_plugin(plugin_id)
            self.logger.info(f"Executing plugin: {plugin_id}")
            
            try:
                plugin.validate_input(data)
                result = plugin.execute(data)
                
                # Save checkpoint
                checkpoint_manager.save_checkpoint(plugin_id, result)
                
                # Update data for next plugin
                data.update(result)
                
            except Exception as e:
                self.logger.error(f"Plugin {plugin_id} failed: {str(e)}")
                raise  # Fail loud!
        
        return data
```

**Test**: `test_complete_workflow_plugin_execution()`

#### Issue #3: Missing Temporal Validation
**Current State**: No verification that causes precede effects
**Fix**: Add temporal ordering validation
**Test**: `test_temporal_causation_validation()`

#### Issue #71: Temporal Causation Logic Missing
**Current State**: Accepts impossible sequences
**Fix**: Implement comprehensive temporal validation
**Test**: `test_impossible_temporal_sequences_rejected()`

### Priority 4: Causal Logic Implementation

#### Issue #72: Hypothesis vs. Belief Confusion
**Current State**: Conflates descriptive beliefs with testable claims
**Fix**: Filter for testable causal hypotheses only
**Test**: `test_only_testable_hypotheses_accepted()`

#### Issue #73: Causal Mechanism Definition Violation
**Current State**: Treats events AS mechanisms
**Fix**: Mechanisms must be processes linking events
**Test**: `test_mechanisms_are_processes_not_events()`

## Month 9-10: Comparative Methodology (Category D3)

### Priority 5: Cross-Case Analysis

#### Issue #11: Missing MSSD/MDSD Analysis
**Current State**: No comparative methodology implementation
**Fix**: Implement Most Similar/Different Systems Design
```python
class ComparativeAnalysis:
    def mssd_analysis(self, cases):
        # Find similar cases with different outcomes
        # Identify crucial differences
    
    def mdsd_analysis(self, cases):
        # Find different cases with similar outcomes
        # Identify crucial similarities
```
**Test**: `test_mssd_methodology_implementation()`

#### Issue #74: Cross-Case Statistical Invalidity
**Current State**: Simple counting instead of proper comparison
**Fix**: Implement case similarity metrics and weighting
**Test**: `test_cross_case_statistical_validity()`

### Priority 6: Scope Conditions

#### Issue #12: Scope Condition Identification Missing
**Current State**: No boundary condition analysis
**Fix**: Identify when/where causal mechanisms apply
**Test**: `test_scope_condition_identification()`

#### Issue #78: Alternative Explanation Testing Missing
**Current State**: No systematic testing of competing explanations
**Fix**: Implement eliminative inference
**Test**: `test_alternative_explanation_elimination()`

## Month 11-12: Algorithmic Correctness (Category E)

### Priority 7: Evidence Assessment Logic

#### Issue #6: Competitive Hypothesis Testing Missing
**Current State**: Each hypothesis analyzed in isolation
**Fix**: Head-to-head hypothesis comparison
**Test**: `test_competitive_hypothesis_evaluation()`

#### Issue #7: Causal Pathway Analysis Missing
**Current State**: No mechanism comparison across cases
**Fix**: Systematic pathway comparison
**Test**: `test_causal_pathway_comparison()`

#### Issue #8: Improper Bayesian Implementation
**Current State**: Uses arithmetic instead of probability theory
**Fix**: Proper Bayesian updating
```python
def bayesian_update(prior, likelihood_ratio):
    odds_prior = prior / (1 - prior)
    odds_posterior = odds_prior * likelihood_ratio
    return odds_posterior / (1 + odds_posterior)
```
**Test**: `test_proper_bayesian_updating()`

### Priority 8: Data Processing Logic

#### Issue #4: Enhancement Module Disconnection
**Current State**: Enhancement insights don't trigger re-evaluation
**Fix**: Create event system for propagation
**Test**: `test_enhancement_propagation()`

#### Issue #5: Cross-Case Architecture Failure
**Current State**: Requires exact ID matching
**Fix**: Semantic similarity matching
**Test**: `test_semantic_hypothesis_matching()`

## Testing Strategy for Phase 2

### Test Organization
```
tests/
  phase_2/
    test_van_evera_logic.py
    test_process_tracing_pipeline.py
    test_comparative_methodology.py
    test_bayesian_inference.py
    fixtures/
      van_evera_test_cases/
      comparative_cases/
      temporal_sequences/
```

### Methodology Validation Tests
- Known process tracing examples from literature
- Van Evera's own case studies
- Validated causal sequences
- Impossible scenarios that should fail

## Success Criteria for Phase 2

- [ ] Hoop test failures eliminate hypotheses
- [ ] Smoking gun successes confirm hypotheses
- [ ] Complete pipeline includes hypothesis testing
- [ ] Temporal validation prevents impossible sequences
- [ ] MSSD/MDSD analysis implemented
- [ ] Bayesian updating mathematically correct
- [ ] All Phase 2 tests passing

## Deliverables

1. New `core/hypothesis_testing.py` module
2. Rewritten `core/analyze.py` with Van Evera logic
3. New `core/comparative_analysis.py` module
4. Updated prompts without methodological bias
5. Comprehensive methodology test suite
6. Validation against published process tracing studies

## Plugin Architecture Benefits for Phase 2

### Why Plugins Make Methodology Compliance Easier

1. **Isolation of Logic**: Each Van Evera test is a separate plugin with clear logic
2. **Testability**: Test each diagnostic method in isolation
3. **Flexibility**: Swap test implementations without changing workflow
4. **Observability**: Plugin lifecycle provides natural logging points
5. **Checkpointing**: Each plugin completion is a resumable checkpoint

### Plugin Testing Strategy

```python
# tests/phase_2/test_diagnostic_plugins.py
class TestDiagnosticPlugins:
    def test_hoop_test_eliminates_on_failure(self):
        """Hoop test must eliminate hypotheses that fail"""
        plugin = HoopTestPlugin("hoop_test", mock_context)
        data = {
            "hypotheses": [{"id": "H1", "claim": "X causes Y"}],
            "evidence": [{"supports": "H2", "refutes": "H1"}],
            "test_conditions": [{"necessary": "evidence supporting H1"}]
        }
        result = plugin.execute(data)
        assert result["hypotheses"][0]["status"] == "eliminated"
    
    def test_smoking_gun_confirms_on_success(self):
        """Smoking gun must confirm hypotheses that pass"""
        plugin = SmokingGunTestPlugin("smoking_gun", mock_context)
        # ... test implementation
```

### Complete Plugin List for Phase 2

```python
# Diagnostic Test Plugins (Van Evera)
- HoopTestPlugin
- SmokingGunTestPlugin  
- StrawInWindPlugin
- DoublyDecisivePlugin

# Methodology Plugins
- TestDesignPlugin (enforces test design before evaluation)
- TemporalValidationPlugin (ensures causes precede effects)
- BayesianUpdatePlugin (proper probability calculations)
- HypothesisComparisonPlugin (competitive evaluation)

# Comparative Analysis Plugins
- MSSDAnalysisPlugin (Most Similar Systems Design)
- MDSDAnalysisPlugin (Most Different Systems Design)
- ScopeConditionPlugin (boundary analysis)
- AlternativeExplanationPlugin (eliminative inference)

# Synthesis Plugins
- CausalPathwayPlugin (mechanism comparison)
- CounterfactualPlugin (alternative histories)
- SynthesisPlugin (final integration)
```

## Next Phase Trigger

Phase 2 is complete when:
- All 18 Category D methodology violations are fixed via plugins
- All 10 Category E algorithmic failures are resolved via plugins
- Plugin-based workflow produces methodologically valid process tracing
- Results align with published examples
- All plugins have >95% test coverage

Once complete, replace Phase 1 instructions in CLAUDE.md with Phase 3: System Optimization.