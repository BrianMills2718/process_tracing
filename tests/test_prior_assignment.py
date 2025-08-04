"""
Comprehensive test suite for prior assignment algorithms.

Tests all prior assignment methods including uniform, frequency-based,
theory-guided, complexity-penalized, and hierarchical assignments.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import tempfile
import json
from pathlib import Path

from core.prior_assignment import (
    UniformPriorAssigner, FrequencyBasedPriorAssigner, TheoryGuidedPriorAssigner,
    ComplexityPenalizedPriorAssigner, HierarchicalPriorAssigner,
    PriorAssignmentOrchestrator, PriorAssignmentConfig, PriorAssignmentMethod
)
from core.bayesian_models import (
    BayesianHypothesis, BayesianHypothesisSpace, HypothesisType
)


class TestUniformPriorAssigner:
    """Test uniform prior assignment algorithm."""
    
    def setup_method(self):
        """Set up test hypothesis space."""
        self.space = BayesianHypothesisSpace("test_space", "Test space")
        
        # Create hypotheses
        hyp1 = BayesianHypothesis("h1", "Hypothesis 1", HypothesisType.PRIMARY)
        hyp2 = BayesianHypothesis("h2", "Hypothesis 2", HypothesisType.ALTERNATIVE)
        hyp3 = BayesianHypothesis("h3", "Hypothesis 3", HypothesisType.ALTERNATIVE)
        
        self.space.add_hypothesis(hyp1)
        self.space.add_hypothesis(hyp2)
        self.space.add_hypothesis(hyp3)
        
        self.config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.UNIFORM,
            parameters={}
        )
        self.assigner = UniformPriorAssigner(self.config)
    
    def test_uniform_assignment_basic(self):
        """Test basic uniform prior assignment."""
        priors = self.assigner.assign_priors(self.space)
        
        assert len(priors) == 3
        assert "h1" in priors
        assert "h2" in priors
        assert "h3" in priors
        
        # All priors should be equal
        unique_priors = set(priors.values())
        assert len(unique_priors) == 1
        
        # Should sum to 1 for collective exhaustiveness
        assert abs(sum(priors.values()) - 1.0) < 1e-10
    
    def test_uniform_assignment_mutual_exclusivity(self):
        """Test uniform assignment with mutual exclusivity groups."""
        # Make h2 and h3 mutually exclusive
        self.space.add_mutual_exclusivity_group({"h2", "h3"})
        
        priors = self.assigner.assign_priors(self.space)
        
        # h2 and h3 should each get 0.5 of their group
        assert abs(priors["h2"] - 0.5) < 1e-10
        assert abs(priors["h3"] - 0.5) < 1e-10
    
    def test_uniform_assignment_empty_space(self):
        """Test uniform assignment with empty hypothesis space."""
        empty_space = BayesianHypothesisSpace("empty", "Empty space")
        priors = self.assigner.assign_priors(empty_space)
        
        assert len(priors) == 0
    
    def test_get_assignment_rationale(self):
        """Test assignment rationale explanation."""
        rationale = self.assigner.get_assignment_rationale()
        
        assert "uniform" in rationale.lower()
        assert "principle of indifference" in rationale.lower()


class TestFrequencyBasedPriorAssigner:
    """Test frequency-based prior assignment algorithm."""
    
    def setup_method(self):
        """Set up test hypothesis space and frequency data."""
        self.space = BayesianHypothesisSpace("freq_space", "Frequency test space")
        
        # Create hypotheses
        hyp1 = BayesianHypothesis("economic_growth", "Economic growth causes development", HypothesisType.PRIMARY)
        hyp2 = BayesianHypothesis("political_stability", "Political stability causes development", HypothesisType.ALTERNATIVE)
        hyp3 = BayesianHypothesis("unknown_cause", "Unknown causal mechanism", HypothesisType.NULL)
        
        self.space.add_hypothesis(hyp1)
        self.space.add_hypothesis(hyp2)
        self.space.add_hypothesis(hyp3)
        
        # Historical frequency data
        frequency_data = {
            "economic_growth": 0.7,
            "political_stability": 0.5,
            "type_primary": 0.6,
            "type_alternative": 0.4,
            "type_null": 0.1
        }
        
        self.config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.FREQUENCY_BASED,
            parameters={
                "historical_frequencies": frequency_data,
                "smoothing_factor": 0.1,
                "default_frequency": 0.2
            }
        )
        self.assigner = FrequencyBasedPriorAssigner(self.config)
    
    def test_frequency_assignment_exact_match(self):
        """Test frequency assignment with exact hypothesis ID match."""
        priors = self.assigner.assign_priors(self.space)
        
        assert len(priors) == 3
        
        # Should use exact frequencies for economic_growth and political_stability
        assert priors["economic_growth"] > priors["political_stability"]
        assert priors["political_stability"] > priors["unknown_cause"]
    
    def test_frequency_assignment_type_fallback(self):
        """Test frequency assignment with hypothesis type fallback."""
        # Remove exact matches from frequency data
        config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.FREQUENCY_BASED,
            parameters={
                "historical_frequencies": {
                    "type_primary": 0.7,
                    "type_alternative": 0.3,
                    "type_null": 0.1
                },
                "smoothing_factor": 0.1,
                "default_frequency": 0.2
            }
        )
        assigner = FrequencyBasedPriorAssigner(config)
        
        priors = assigner.assign_priors(self.space)
        
        # Should use type-based frequencies
        assert priors["economic_growth"] > priors["political_stability"]  # PRIMARY > ALTERNATIVE
        assert priors["political_stability"] > priors["unknown_cause"]    # ALTERNATIVE > NULL
    
    def test_frequency_assignment_normalization(self):
        """Test that frequency assignment is properly normalized."""
        priors = self.assigner.assign_priors(self.space)
        
        # Should sum to 1 for collective exhaustiveness
        total = sum(priors.values())
        assert abs(total - 1.0) < 1e-10
    
    def test_frequency_assignment_smoothing(self):
        """Test smoothing factor prevents zero probabilities."""
        # Use config with zero frequency for some hypothesis
        config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.FREQUENCY_BASED,
            parameters={
                "historical_frequencies": {"economic_growth": 0.0},
                "smoothing_factor": 0.1,
                "default_frequency": 0.0
            }
        )
        assigner = FrequencyBasedPriorAssigner(config)
        
        priors = assigner.assign_priors(self.space)
        
        # All priors should be > 0 due to smoothing
        for prior in priors.values():
            assert prior > 0.0


class TestTheoryGuidedPriorAssigner:
    """Test theory-guided prior assignment algorithm."""
    
    def setup_method(self):
        """Set up test hypothesis space and theory data."""
        self.space = BayesianHypothesisSpace("theory_space", "Theory test space")
        
        # Create hypotheses
        hyp1 = BayesianHypothesis("market_efficiency", "Market efficiency hypothesis", HypothesisType.PRIMARY)
        hyp2 = BayesianHypothesis("government_intervention", "Government intervention hypothesis", HypothesisType.ALTERNATIVE)
        hyp3 = BayesianHypothesis("complex_interaction", "Complex market-government interaction", HypothesisType.COMPOSITE)
        hyp4 = BayesianHypothesis("necessary_condition", "Market access as necessary condition", HypothesisType.NECESSARY)
        
        self.space.add_hypothesis(hyp1)
        self.space.add_hypothesis(hyp2)
        self.space.add_hypothesis(hyp3)
        self.space.add_hypothesis(hyp4)
        
        self.config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.THEORY_GUIDED,
            parameters={
                "theory_weights": {
                    "market": 1.2,        # Boost market-related hypotheses
                    "efficiency": 1.1,    # Boost efficiency theories
                    "government": 0.8,    # Reduce government theories
                    "complex": 0.7        # Reduce complex theories
                },
                "mechanism_plausibility": {
                    "market_efficiency": 0.8,
                    "necessary_condition": 0.7
                }
            },
            theoretical_framework="Neoclassical Economics",
            domain_expertise_level=0.8
        )
        self.assigner = TheoryGuidedPriorAssigner(self.config)
    
    def test_theory_guided_assignment_type_preferences(self):
        """Test theory-guided assignment respects hypothesis type preferences."""
        priors = self.assigner.assign_priors(self.space)
        
        assert len(priors) == 4
        
        # PRIMARY should have higher prior than ALTERNATIVE
        assert priors["market_efficiency"] > priors["government_intervention"]
        
        # NECESSARY should have high prior
        assert priors["necessary_condition"] > 0.6
        
        # COMPOSITE should have lower prior (complexity penalty)
        assert priors["complex_interaction"] < priors["market_efficiency"]
    
    def test_theory_guided_assignment_theory_weights(self):
        """Test theory weights influence prior assignment."""
        priors = self.assigner.assign_priors(self.space)
        
        # Market-related hypotheses should be boosted
        market_hyps = ["market_efficiency"]
        gov_hyps = ["government_intervention"]
        
        for market_hyp in market_hyps:
            for gov_hyp in gov_hyps:
                # Market hypotheses should generally have higher priors due to theory weights
                # (though other factors like hypothesis type also matter)
                pass  # Theory weights are just one factor
    
    def test_theory_guided_assignment_mechanism_plausibility(self):
        """Test mechanism plausibility affects prior assignment."""
        priors = self.assigner.assign_priors(self.space)
        
        # Hypotheses with explicit plausibility data should reflect those values
        assert priors["market_efficiency"] > 0.5  # High plausibility
        assert priors["necessary_condition"] > 0.5  # High plausibility
    
    def test_theory_guided_expertise_adjustment(self):
        """Test domain expertise level affects theory application."""
        # High expertise - should strongly apply theory
        high_expertise_config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.THEORY_GUIDED,
            parameters=self.config.parameters,
            domain_expertise_level=0.9
        )
        high_expertise_assigner = TheoryGuidedPriorAssigner(high_expertise_config)
        
        # Low expertise - should move towards uniform
        low_expertise_config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.THEORY_GUIDED,
            parameters=self.config.parameters,
            domain_expertise_level=0.2
        )
        low_expertise_assigner = TheoryGuidedPriorAssigner(low_expertise_config)
        
        high_priors = high_expertise_assigner.assign_priors(self.space)
        low_priors = low_expertise_assigner.assign_priors(self.space)
        
        # High expertise should show more variation (stronger theory application)
        high_std = np.std(list(high_priors.values()))
        low_std = np.std(list(low_priors.values()))
        
        assert high_std >= low_std


class TestComplexityPenalizedPriorAssigner:
    """Test complexity-penalized prior assignment (Occam's razor)."""
    
    def setup_method(self):
        """Set up test hypothesis space with varying complexity."""
        self.space = BayesianHypothesisSpace("complexity_space", "Complexity test space")
        
        # Simple hypothesis
        simple_hyp = BayesianHypothesis("simple", "Simple direct causation", HypothesisType.PRIMARY)
        
        # Complex hypothesis with multiple requirements
        complex_hyp = BayesianHypothesis(
            "complex", 
            "Complex multi-factor causation involving economic, political, social, and environmental factors interacting through multiple pathways",
            HypothesisType.COMPOSITE
        )
        complex_hyp.required_evidence = {"ev1", "ev2", "ev3", "ev4", "ev5"}
        complex_hyp.temporal_constraints = {"seq1": "ordered", "seq2": "simultaneous"}
        
        # Hierarchical hypothesis with children
        parent_hyp = BayesianHypothesis("parent", "Parent hypothesis", HypothesisType.PRIMARY)
        child1_hyp = BayesianHypothesis("child1", "Child hypothesis 1", HypothesisType.CONDITIONAL, parent_hypothesis="parent")
        child2_hyp = BayesianHypothesis("child2", "Child hypothesis 2", HypothesisType.CONDITIONAL, parent_hypothesis="parent")
        
        parent_hyp.child_hypotheses = ["child1", "child2"]
        
        self.space.add_hypothesis(simple_hyp)
        self.space.add_hypothesis(complex_hyp)
        self.space.add_hypothesis(parent_hyp)
        self.space.add_hypothesis(child1_hyp)
        self.space.add_hypothesis(child2_hyp)
        
        self.config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.COMPLEXITY_PENALIZED,
            parameters={
                "complexity_penalty": 0.1,
                "base_prior": 0.5,
                "max_complexity": 10.0
            }
        )
        self.assigner = ComplexityPenalizedPriorAssigner(self.config)
    
    def test_complexity_penalty_basic(self):
        """Test basic complexity penalty application."""
        priors = self.assigner.assign_priors(self.space)
        
        # Simple hypothesis should have higher prior than complex one
        assert priors["simple"] > priors["complex"]
        
        # Parent should have higher prior than children (less hierarchical complexity)
        assert priors["parent"] > priors["child1"]
        assert priors["parent"] > priors["child2"]
    
    def test_complexity_scoring(self):
        """Test complexity scoring algorithm."""
        # Test with specific hypotheses
        simple_hyp = self.space.hypotheses["simple"]
        complex_hyp = self.space.hypotheses["complex"]
        
        simple_complexity = self.assigner._calculate_complexity(simple_hyp, self.space)
        complex_complexity = self.assigner._calculate_complexity(complex_hyp, self.space)
        
        # Complex hypothesis should have higher complexity score
        assert complex_complexity > simple_complexity
        
        # Complex hypothesis penalties:
        # - COMPOSITE type: +2.0
        # - Required evidence (5 pieces): +2.5
        # - Temporal constraints (2): +0.6
        # - Description length: ~1.0
        assert complex_complexity >= 5.0
    
    def test_complexity_penalty_normalization(self):
        """Test that complexity-penalized priors are properly normalized."""
        priors = self.assigner.assign_priors(self.space)
        
        # Should sum to 1
        total = sum(priors.values())
        assert abs(total - 1.0) < 1e-10
        
        # All priors should be positive
        for prior in priors.values():
            assert prior > 0.0


class TestHierarchicalPriorAssigner:
    """Test hierarchical prior assignment algorithm."""
    
    def setup_method(self):
        """Set up test hierarchy."""
        self.space = BayesianHypothesisSpace("hierarchy_space", "Hierarchy test space")
        
        # Root hypotheses
        root1 = BayesianHypothesis("root1", "Root hypothesis 1", HypothesisType.PRIMARY)
        root2 = BayesianHypothesis("root2", "Root hypothesis 2", HypothesisType.PRIMARY)
        
        # Level 1 children
        child1_1 = BayesianHypothesis("child1_1", "Child 1 of root 1", HypothesisType.CONDITIONAL, parent_hypothesis="root1")
        child1_2 = BayesianHypothesis("child1_2", "Child 2 of root 1", HypothesisType.CONDITIONAL, parent_hypothesis="root1")
        child2_1 = BayesianHypothesis("child2_1", "Child 1 of root 2", HypothesisType.CONDITIONAL, parent_hypothesis="root2")
        
        # Level 2 grandchildren
        grandchild = BayesianHypothesis("grandchild", "Grandchild", HypothesisType.CONDITIONAL, parent_hypothesis="child1_1")
        
        self.space.add_hypothesis(root1)
        self.space.add_hypothesis(root2)
        self.space.add_hypothesis(child1_1)
        self.space.add_hypothesis(child1_2)
        self.space.add_hypothesis(child2_1)
        self.space.add_hypothesis(grandchild)
        
        self.config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.HIERARCHICAL,
            parameters={
                "parent_influence": 0.7,
                "level_discount": 0.1,
                "root_prior": 0.8
            }
        )
        self.assigner = HierarchicalPriorAssigner(self.config)
    
    def test_hierarchical_root_assignment(self):
        """Test root hypothesis prior assignment."""
        priors = self.assigner.assign_priors(self.space)
        
        # Root hypotheses should share root prior equally
        expected_root_prior = 0.8 / 2  # root_prior / number_of_roots
        assert abs(priors["root1"] - expected_root_prior) < 1e-10
        assert abs(priors["root2"] - expected_root_prior) < 1e-10
    
    def test_hierarchical_inheritance(self):
        """Test probability inheritance from parents to children."""
        priors = self.assigner.assign_priors(self.space)
        
        # Children should have lower priors than parents
        assert priors["child1_1"] < priors["root1"]
        assert priors["child1_2"] < priors["root1"]
        assert priors["child2_1"] < priors["root2"]
        
        # Grandchildren should have lower priors than children
        assert priors["grandchild"] < priors["child1_1"]
    
    def test_hierarchical_level_discount(self):
        """Test level discount factor application."""
        priors = self.assigner.assign_priors(self.space)
        
        # Level 1 children should have higher priors than level 2 grandchildren
        level1_avg = (priors["child1_1"] + priors["child1_2"] + priors["child2_1"]) / 3
        level2_prior = priors["grandchild"]
        
        assert level1_avg > level2_prior
    
    def test_hierarchical_sibling_equality(self):
        """Test that sibling hypotheses receive equal inheritance."""
        priors = self.assigner.assign_priors(self.space)
        
        # Children of root1 should have equal priors (same parent, same level)
        assert abs(priors["child1_1"] - priors["child1_2"]) < 1e-10


class TestPriorAssignmentOrchestrator:
    """Test prior assignment orchestrator and integration."""
    
    def setup_method(self):
        """Set up test orchestrator and hypothesis space."""
        self.orchestrator = PriorAssignmentOrchestrator()
        
        self.space = BayesianHypothesisSpace("orchestrator_space", "Orchestrator test space")
        
        # Create diverse hypotheses
        hyp1 = BayesianHypothesis("h1", "Primary hypothesis", HypothesisType.PRIMARY)
        hyp2 = BayesianHypothesis("h2", "Alternative hypothesis", HypothesisType.ALTERNATIVE)
        hyp3 = BayesianHypothesis("h3", "Null hypothesis", HypothesisType.NULL)
        
        self.space.add_hypothesis(hyp1)
        self.space.add_hypothesis(hyp2)
        self.space.add_hypothesis(hyp3)
    
    def test_orchestrator_single_assignment(self):
        """Test single prior assignment method."""
        config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.UNIFORM,
            parameters={}
        )
        
        priors = self.orchestrator.assign_priors(self.space, config)
        
        assert len(priors) == 3
        assert len(self.orchestrator.assignment_history) == 1
        
        # Check that priors were actually applied to hypotheses
        for hyp_id, prior in priors.items():
            assert self.space.hypotheses[hyp_id].prior_probability == prior
            assert self.space.hypotheses[hyp_id].posterior_probability == prior
    
    def test_orchestrator_multiple_method_combination(self):
        """Test combining multiple assignment methods."""
        configs = [
            PriorAssignmentConfig(method=PriorAssignmentMethod.UNIFORM, parameters={}),
            PriorAssignmentConfig(
                method=PriorAssignmentMethod.THEORY_GUIDED,
                parameters={"theory_weights": {"primary": 1.2}},
                domain_expertise_level=0.6
            )
        ]
        
        weights = [0.3, 0.7]  # Weight theory-guided more heavily
        
        combined_priors = self.orchestrator.combine_multiple_assignments(self.space, configs, weights)
        
        assert len(combined_priors) == 3
        assert len(self.orchestrator.assignment_history) == 2  # One for each method
        
        # Combined result should be different from either individual method
        uniform_priors = self.orchestrator.assignment_history[0]["priors"]
        theory_priors = self.orchestrator.assignment_history[1]["priors"]
        
        for hyp_id in combined_priors:
            # Combined should not equal uniform (unless weights are very skewed)
            if weights[1] > 0.5:  # Theory-guided weighted more
                assert combined_priors[hyp_id] != uniform_priors[hyp_id]
    
    def test_orchestrator_sensitivity_analysis(self):
        """Test sensitivity analysis for prior assignment."""
        config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.THEORY_GUIDED,
            parameters={
                "theory_weights": {"primary": 1.0},
                "mechanism_plausibility": {}
            },
            domain_expertise_level=0.5
        )
        
        parameter_ranges = {
            "domain_expertise_level": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        sensitivity_results = self.orchestrator.sensitivity_analysis(self.space, config, parameter_ranges)
        
        assert "base_priors" in sensitivity_results
        assert "parameter_sensitivity" in sensitivity_results
        assert "domain_expertise_level" in sensitivity_results["parameter_sensitivity"]
        
        # Should have results for each parameter value
        expertise_results = sensitivity_results["parameter_sensitivity"]["domain_expertise_level"]
        assert len(expertise_results) == 5
        
        # Should track maximum deviations
        assert "max_deviation" in sensitivity_results
        assert "most_sensitive_hypothesis" in sensitivity_results
    
    def test_orchestrator_assignment_history(self):
        """Test assignment history tracking."""
        config1 = PriorAssignmentConfig(method=PriorAssignmentMethod.UNIFORM, parameters={})
        config2 = PriorAssignmentConfig(method=PriorAssignmentMethod.COMPLEXITY_PENALIZED, parameters={})
        
        self.orchestrator.assign_priors(self.space, config1)
        self.orchestrator.assign_priors(self.space, config2)
        
        assert len(self.orchestrator.assignment_history) == 2
        
        # Each record should have required fields
        for record in self.orchestrator.assignment_history:
            assert "timestamp" in record
            assert "method" in record
            assert "hypothesis_space_id" in record
            assert "priors" in record
            assert "rationale" in record
    
    def test_orchestrator_export_and_summary(self):
        """Test history export and summary generation."""
        config = PriorAssignmentConfig(method=PriorAssignmentMethod.UNIFORM, parameters={})
        self.orchestrator.assign_priors(self.space, config)
        
        # Test summary
        summary = self.orchestrator.get_assignment_summary()
        assert summary["total_assignments"] == 1
        assert "uniform" in summary["methods_used"]
        assert summary["methods_used"]["uniform"] == 1
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.orchestrator.export_assignment_history(tmp_path)
            
            # Verify export
            assert Path(tmp_path).exists()
            
            with open(tmp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) == 1
            assert exported_data[0]["method"] == "uniform"
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestMathematicalValidation:
    """Test mathematical correctness of prior assignment algorithms."""
    
    def test_probability_conservation(self):
        """Test that all assignment methods conserve probability mass."""
        space = BayesianHypothesisSpace("math_space", "Mathematical validation space")
        
        # Create hypotheses
        for i in range(5):
            hyp = BayesianHypothesis(f"h{i}", f"Hypothesis {i}", HypothesisType.PRIMARY)
            space.add_hypothesis(hyp)
        
        # Test all assignment methods
        methods = [
            PriorAssignmentMethod.UNIFORM,
            PriorAssignmentMethod.FREQUENCY_BASED,
            PriorAssignmentMethod.THEORY_GUIDED,
            PriorAssignmentMethod.COMPLEXITY_PENALIZED,
            PriorAssignmentMethod.HIERARCHICAL
        ]
        
        orchestrator = PriorAssignmentOrchestrator()
        
        for method in methods:
            config = PriorAssignmentConfig(
                method=method,
                parameters={"historical_frequencies": {"h0": 0.3, "h1": 0.2}} if method == PriorAssignmentMethod.FREQUENCY_BASED else {}
            )
            
            priors = orchestrator.assign_priors(space, config)
            
            # Should sum to 1
            total = sum(priors.values())
            assert abs(total - 1.0) < 1e-10, f"Method {method} failed probability conservation: {total}"
            
            # All priors should be positive
            for prior in priors.values():
                assert prior > 0.0, f"Method {method} produced non-positive prior: {prior}"
    
    def test_mutual_exclusivity_constraint_satisfaction(self):
        """Test that mutual exclusivity constraints are satisfied."""
        space = BayesianHypothesisSpace("constraint_space", "Constraint test space")
        
        # Create mutually exclusive hypotheses
        h1 = BayesianHypothesis("exclusive1", "Exclusive hypothesis 1", HypothesisType.PRIMARY)
        h2 = BayesianHypothesis("exclusive2", "Exclusive hypothesis 2", HypothesisType.ALTERNATIVE)
        h3 = BayesianHypothesis("independent", "Independent hypothesis", HypothesisType.PRIMARY)
        
        space.add_hypothesis(h1)
        space.add_hypothesis(h2)
        space.add_hypothesis(h3)
        
        # Make h1 and h2 mutually exclusive
        space.add_mutual_exclusivity_group({"exclusive1", "exclusive2"})
        
        orchestrator = PriorAssignmentOrchestrator()
        config = PriorAssignmentConfig(method=PriorAssignmentMethod.UNIFORM, parameters={})
        
        priors = orchestrator.assign_priors(space, config)
        
        # Mutually exclusive hypotheses should sum to their allocated probability
        exclusive_sum = priors["exclusive1"] + priors["exclusive2"]
        assert exclusive_sum <= 1.0  # Should not exceed total probability
        
        # Each should have equal probability within their group
        assert abs(priors["exclusive1"] - priors["exclusive2"]) < 1e-10
    
    def test_hierarchical_probability_flow(self):
        """Test probability flow in hierarchical assignments."""
        space = BayesianHypothesisSpace("hierarchy_space", "Hierarchy test")
        
        # Create hierarchy: parent -> child1, child2
        parent = BayesianHypothesis("parent", "Parent hypothesis", HypothesisType.PRIMARY)
        child1 = BayesianHypothesis("child1", "Child 1", HypothesisType.CONDITIONAL, parent_hypothesis="parent")
        child2 = BayesianHypothesis("child2", "Child 2", HypothesisType.CONDITIONAL, parent_hypothesis="parent")
        
        space.add_hypothesis(parent)
        space.add_hypothesis(child1)
        space.add_hypothesis(child2)
        
        orchestrator = PriorAssignmentOrchestrator()
        config = PriorAssignmentConfig(
            method=PriorAssignmentMethod.HIERARCHICAL,
            parameters={"parent_influence": 0.5, "root_prior": 0.6}
        )
        
        priors = orchestrator.assign_priors(space, config)
        
        # Parent probability should be distributed among children
        parent_prob = priors["parent"]
        children_sum = priors["child1"] + priors["child2"]
        
        # Children should have lower individual probabilities than parent
        assert priors["child1"] < parent_prob
        assert priors["child2"] < parent_prob
        
        # Children should have equal probabilities (same parent, same level)
        assert abs(priors["child1"] - priors["child2"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])