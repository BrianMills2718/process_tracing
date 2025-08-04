"""
Comprehensive test suite for Bayesian models and data structures.

Tests all core Bayesian inference components including hypotheses, evidence,
hypothesis spaces, and the main process tracing model with mathematical validation.
"""

import pytest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import json
from pathlib import Path

from core.bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace,
    BayesianProcessTracingModel, HypothesisType, EvidenceType, PriorType
)


class TestBayesianHypothesis:
    """Test BayesianHypothesis data structure and methods."""
    
    def test_hypothesis_creation_basic(self):
        """Test basic hypothesis creation with required fields."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="test_hyp_1",
            description="Test hypothesis for economic growth",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        assert hypothesis.hypothesis_id == "test_hyp_1"
        assert hypothesis.description == "Test hypothesis for economic growth"
        assert hypothesis.hypothesis_type == HypothesisType.PRIMARY
        assert hypothesis.prior_probability == 0.5
        assert hypothesis.posterior_probability == 0.5
        assert isinstance(hypothesis.last_updated, datetime)
    
    def test_hypothesis_creation_with_custom_priors(self):
        """Test hypothesis creation with custom prior probabilities."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="custom_prior",
            description="Custom prior hypothesis",
            hypothesis_type=HypothesisType.ALTERNATIVE,
            prior_probability=0.3,
            posterior_probability=0.7
        )
        
        assert hypothesis.prior_probability == 0.3
        assert hypothesis.posterior_probability == 0.7
    
    def test_hypothesis_probability_validation(self):
        """Test that invalid probabilities raise ValueError."""
        with pytest.raises(ValueError, match="Prior probability must be between 0 and 1"):
            BayesianHypothesis(
                hypothesis_id="invalid",
                description="Invalid prior",
                hypothesis_type=HypothesisType.PRIMARY,
                prior_probability=1.5
            )
        
        with pytest.raises(ValueError, match="Posterior probability must be between 0 and 1"):
            BayesianHypothesis(
                hypothesis_id="invalid",
                description="Invalid posterior",
                hypothesis_type=HypothesisType.PRIMARY,
                posterior_probability=-0.1
            )
    
    def test_add_child_hypothesis(self):
        """Test adding child hypotheses."""
        parent = BayesianHypothesis(
            hypothesis_id="parent",
            description="Parent hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        parent.add_child_hypothesis("child1")
        parent.add_child_hypothesis("child2")
        parent.add_child_hypothesis("child1")  # Duplicate should not be added
        
        assert len(parent.child_hypotheses) == 2
        assert "child1" in parent.child_hypotheses
        assert "child2" in parent.child_hypotheses
    
    def test_add_evidence(self):
        """Test adding different types of evidence."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="evidence_test",
            description="Test evidence addition",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        hypothesis.add_evidence("evidence1", "supporting")
        hypothesis.add_evidence("evidence2", "required")
        hypothesis.add_evidence("evidence3", "contradicting")
        
        assert "evidence1" in hypothesis.supporting_evidence
        assert "evidence2" in hypothesis.supporting_evidence
        assert "evidence2" in hypothesis.required_evidence
        assert "evidence3" in hypothesis.contradicting_evidence
    
    def test_update_posterior(self):
        """Test posterior probability updating with history tracking."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="update_test",
            description="Test posterior updates",
            hypothesis_type=HypothesisType.PRIMARY,
            posterior_probability=0.5
        )
        
        initial_time = hypothesis.last_updated
        
        # Update posterior
        hypothesis.update_posterior(0.8, "evidence1", 2.0)
        
        assert hypothesis.posterior_probability == 0.8
        assert hypothesis.last_updated > initial_time
        assert len(hypothesis.update_history) == 1
        
        update_record = hypothesis.update_history[0]
        assert update_record["evidence_id"] == "evidence1"
        assert update_record["old_posterior"] == 0.5
        assert update_record["new_posterior"] == 0.8
        assert update_record["likelihood_ratio"] == 2.0
    
    def test_calculate_confidence_no_evidence(self):
        """Test confidence calculation with no evidence."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="no_evidence",
            description="No evidence hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        confidence = hypothesis.calculate_confidence()
        assert confidence == 0.0
        assert hypothesis.confidence_level == 0.0
    
    def test_calculate_confidence_with_evidence(self):
        """Test confidence calculation with supporting evidence."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="with_evidence",
            description="Hypothesis with evidence",
            hypothesis_type=HypothesisType.PRIMARY,
            posterior_probability=0.8
        )
        
        # Add supporting evidence
        for i in range(3):
            hypothesis.add_evidence(f"evidence_{i}", "supporting")
        
        confidence = hypothesis.calculate_confidence()
        
        assert confidence > 0.0
        assert confidence <= 1.0
        assert hypothesis.confidence_level == confidence
    
    def test_calculate_confidence_with_contradicting_evidence(self):
        """Test confidence calculation with contradicting evidence."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="contradicted",
            description="Hypothesis with contradicting evidence",
            hypothesis_type=HypothesisType.PRIMARY,
            posterior_probability=0.8
        )
        
        # Add supporting and contradicting evidence
        hypothesis.add_evidence("support1", "supporting")
        hypothesis.add_evidence("support2", "supporting")
        hypothesis.add_evidence("contradict1", "contradicting")
        hypothesis.add_evidence("contradict2", "contradicting")
        
        confidence = hypothesis.calculate_confidence()
        
        # Should be lower due to contradicting evidence
        assert 0.0 <= confidence <= 1.0
        
        # Compare with hypothesis having only supporting evidence
        clean_hypothesis = BayesianHypothesis(
            hypothesis_id="clean",
            description="Clean hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            posterior_probability=0.8
        )
        clean_hypothesis.add_evidence("support1", "supporting")
        clean_hypothesis.add_evidence("support2", "supporting")
        
        clean_confidence = clean_hypothesis.calculate_confidence()
        assert confidence < clean_confidence


class TestBayesianEvidence:
    """Test BayesianEvidence data structure and methods."""
    
    def test_evidence_creation_basic(self):
        """Test basic evidence creation."""
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence piece",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_1"
        )
        
        assert evidence.evidence_id == "test_evidence"
        assert evidence.description == "Test evidence piece"
        assert evidence.evidence_type == EvidenceType.SMOKING_GUN
        assert evidence.source_node_id == "node_1"
        assert isinstance(evidence.last_updated, datetime)
    
    def test_evidence_probability_validation(self):
        """Test evidence probability validation."""
        with pytest.raises(ValueError):
            BayesianEvidence(
                evidence_id="invalid",
                description="Invalid evidence",
                evidence_type=EvidenceType.HOOP,
                source_node_id="node_1",
                necessity=1.5  # Invalid probability
            )
    
    def test_van_evera_hoop_properties(self):
        """Test Van Evera hoop test properties."""
        evidence = BayesianEvidence(
            evidence_id="hoop_test",
            description="Hoop test evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="node_1"
        )
        
        # Hoop tests should have high necessity, low sufficiency
        assert evidence.necessity >= 0.8
        assert evidence.sufficiency <= 0.4
        assert evidence.likelihood_positive >= 0.8
        assert evidence.likelihood_negative <= 0.2
    
    def test_van_evera_smoking_gun_properties(self):
        """Test Van Evera smoking gun properties."""
        evidence = BayesianEvidence(
            evidence_id="smoking_gun_test",
            description="Smoking gun evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_1"
        )
        
        # Smoking gun should have low necessity, high sufficiency
        assert evidence.necessity <= 0.4
        assert evidence.sufficiency >= 0.8
        assert evidence.likelihood_positive >= 0.8
        assert evidence.likelihood_negative <= 0.1
    
    def test_van_evera_doubly_decisive_properties(self):
        """Test Van Evera doubly decisive properties."""
        evidence = BayesianEvidence(
            evidence_id="doubly_decisive_test",
            description="Doubly decisive evidence",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="node_1"
        )
        
        # Doubly decisive should have high necessity and sufficiency
        assert evidence.necessity >= 0.8
        assert evidence.sufficiency >= 0.8
        assert evidence.likelihood_positive >= 0.9
        assert evidence.likelihood_negative <= 0.1
    
    def test_van_evera_straw_in_wind_properties(self):
        """Test Van Evera straw in the wind properties."""
        evidence = BayesianEvidence(
            evidence_id="straw_test",
            description="Straw in the wind evidence",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="node_1"
        )
        
        # Straw in the wind should have low necessity and sufficiency
        assert evidence.necessity <= 0.5
        assert evidence.sufficiency <= 0.5
    
    def test_likelihood_ratio_calculation(self):
        """Test likelihood ratio calculation."""
        evidence = BayesianEvidence(
            evidence_id="ratio_test",
            description="Likelihood ratio test",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_1",
            likelihood_positive=0.9,
            likelihood_negative=0.1
        )
        
        ratio = evidence.get_likelihood_ratio()
        expected_ratio = 0.9 / 0.1
        assert abs(ratio - expected_ratio) < 0.001
    
    def test_likelihood_ratio_infinite(self):
        """Test likelihood ratio with zero false positive rate."""
        evidence = BayesianEvidence(
            evidence_id="perfect_test",
            description="Perfect evidence",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="node_1",
            likelihood_positive=1.0,
            likelihood_negative=0.0
        )
        
        ratio = evidence.get_likelihood_ratio()
        assert ratio == float('inf')
    
    def test_adjusted_likelihood_ratio(self):
        """Test adjusted likelihood ratio with reliability factors."""
        evidence = BayesianEvidence(
            evidence_id="adjusted_test",
            description="Adjusted evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_1",
            likelihood_positive=0.9,
            likelihood_negative=0.1,
            reliability=0.8,
            strength=0.9,
            source_credibility=0.85
        )
        
        base_ratio = evidence.get_likelihood_ratio()
        adjusted_ratio = evidence.get_adjusted_likelihood_ratio()
        
        # Adjusted ratio should be closer to 1 (neutral) due to adjustment factors
        assert adjusted_ratio != base_ratio
        assert abs(adjusted_ratio - 1.0) < abs(base_ratio - 1.0)
    
    def test_evidence_with_timestamp(self):
        """Test evidence with temporal information."""
        timestamp = datetime.now() - timedelta(days=30)
        
        evidence = BayesianEvidence(
            evidence_id="temporal_test",
            description="Evidence with timestamp",
            evidence_type=EvidenceType.HOOP,
            source_node_id="node_1",
            timestamp=timestamp,
            temporal_order=1
        )
        
        assert evidence.timestamp == timestamp
        assert evidence.temporal_order == 1


class TestBayesianHypothesisSpace:
    """Test BayesianHypothesisSpace management and operations."""
    
    def setup_method(self):
        """Set up test hypothesis space."""
        self.space = BayesianHypothesisSpace(
            hypothesis_space_id="test_space",
            description="Test hypothesis space"
        )
        
        # Create test hypotheses
        self.hyp1 = BayesianHypothesis(
            hypothesis_id="hyp1",
            description="Primary hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.6
        )
        
        self.hyp2 = BayesianHypothesis(
            hypothesis_id="hyp2",
            description="Alternative hypothesis",
            hypothesis_type=HypothesisType.ALTERNATIVE,
            prior_probability=0.4
        )
        
        self.hyp3 = BayesianHypothesis(
            hypothesis_id="hyp3",
            description="Child hypothesis",
            hypothesis_type=HypothesisType.CONDITIONAL,
            parent_hypothesis="hyp1"
        )
    
    def test_add_hypothesis(self):
        """Test adding hypotheses to space."""
        self.space.add_hypothesis(self.hyp1)
        self.space.add_hypothesis(self.hyp2)
        
        assert len(self.space.hypotheses) == 2
        assert "hyp1" in self.space.hypotheses
        assert "hyp2" in self.space.hypotheses
        assert self.space.hypothesis_graph.has_node("hyp1")
        assert self.space.hypothesis_graph.has_node("hyp2")
    
    def test_add_hierarchical_hypotheses(self):
        """Test adding hypotheses with parent-child relationships."""
        self.space.add_hypothesis(self.hyp1)
        self.space.add_hypothesis(self.hyp3)
        
        assert self.space.hypothesis_graph.has_edge("hyp1", "hyp3")
        
        # Test hierarchy level calculation
        level_hyp1 = self.space.get_hierarchy_level("hyp1")
        level_hyp3 = self.space.get_hierarchy_level("hyp3")
        
        assert level_hyp1 == 0  # Root level
        assert level_hyp3 == 1  # Child level
    
    def test_add_evidence(self):
        """Test adding evidence to space."""
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="node_1"
        )
        
        self.space.add_evidence(evidence)
        
        assert len(self.space.evidence) == 1
        assert "test_evidence" in self.space.evidence
    
    def test_mutual_exclusivity_groups(self):
        """Test mutual exclusivity constraints."""
        self.space.add_hypothesis(self.hyp1)
        self.space.add_hypothesis(self.hyp2)
        
        # Make hypotheses mutually exclusive
        self.space.add_mutual_exclusivity_group({"hyp1", "hyp2"})
        
        assert len(self.space.mutual_exclusivity_groups) == 1
        assert {"hyp1", "hyp2"} in self.space.mutual_exclusivity_groups
    
    def test_mutual_exclusivity_normalization(self):
        """Test probability normalization with mutual exclusivity."""
        self.space.add_hypothesis(self.hyp1)
        self.space.add_hypothesis(self.hyp2)
        self.space.add_mutual_exclusivity_group({"hyp1", "hyp2"})
        
        # Set probabilities that don't sum to 1
        self.hyp1.posterior_probability = 0.8
        self.hyp2.posterior_probability = 0.6
        
        # Trigger normalization
        self.space._normalize_probabilities()
        
        # Should now sum to 1
        total_prob = (self.space.hypotheses["hyp1"].posterior_probability + 
                     self.space.hypotheses["hyp2"].posterior_probability)
        assert abs(total_prob - 1.0) < 1e-10
    
    def test_get_competing_hypotheses(self):
        """Test finding competing hypotheses."""
        self.space.add_hypothesis(self.hyp1)
        self.space.add_hypothesis(self.hyp2)
        self.space.add_mutual_exclusivity_group({"hyp1", "hyp2"})
        
        competitors = self.space.get_competing_hypotheses("hyp1")
        
        assert len(competitors) == 1
        assert competitors[0].hypothesis_id == "hyp2"
    
    def test_get_summary_statistics(self):
        """Test summary statistics calculation."""
        self.space.add_hypothesis(self.hyp1)
        self.space.add_hypothesis(self.hyp2)
        
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="node_1"
        )
        self.space.add_evidence(evidence)
        
        stats = self.space.get_summary_statistics()
        
        assert stats["total_hypotheses"] == 2
        assert stats["total_evidence"] == 1
        assert "max_posterior" in stats
        assert "min_posterior" in stats
        assert "mean_posterior" in stats
        assert "std_posterior" in stats
    
    def test_hypothesis_not_found_error(self):
        """Test error when adding mutual exclusivity for non-existent hypothesis."""
        with pytest.raises(ValueError, match="Hypothesis .* not found"):
            self.space.add_mutual_exclusivity_group({"nonexistent", "also_nonexistent"})


class TestBayesianProcessTracingModel:
    """Test complete Bayesian process tracing model."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = BayesianProcessTracingModel(
            model_id="test_model",
            description="Test Bayesian process tracing model"
        )
        
        # Create test hypothesis space
        self.space = BayesianHypothesisSpace(
            hypothesis_space_id="test_space",
            description="Test space"
        )
        
        # Add hypotheses
        hyp1 = BayesianHypothesis(
            hypothesis_id="economic_growth",
            description="Economic growth hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.6
        )
        
        hyp2 = BayesianHypothesis(
            hypothesis_id="political_stability",
            description="Political stability hypothesis",
            hypothesis_type=HypothesisType.ALTERNATIVE,
            prior_probability=0.4
        )
        
        self.space.add_hypothesis(hyp1)
        self.space.add_hypothesis(hyp2)
    
    def test_model_creation(self):
        """Test model creation and initialization."""
        assert self.model.model_id == "test_model"
        assert self.model.description == "Test Bayesian process tracing model"
        assert self.model.prior_type == PriorType.UNIFORM
        assert isinstance(self.model.created_at, datetime)
        assert self.model.analysis_count == 0
    
    def test_add_hypothesis_space(self):
        """Test adding hypothesis spaces to model."""
        self.model.add_hypothesis_space(self.space)
        
        assert len(self.model.hypothesis_spaces) == 1
        assert "test_space" in self.model.hypothesis_spaces
    
    def test_add_global_evidence(self):
        """Test adding global evidence."""
        evidence = BayesianEvidence(
            evidence_id="global_evidence",
            description="Global evidence piece",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="global_node"
        )
        
        self.model.add_global_evidence(evidence)
        
        assert len(self.model.global_evidence) == 1
        assert "global_evidence" in self.model.global_evidence
    
    def test_set_causal_graph(self):
        """Test setting causal graph."""
        graph = nx.DiGraph()
        graph.add_node("cause", type="Event")
        graph.add_node("effect", type="Event")
        graph.add_edge("cause", "effect", strength=0.8)
        
        self.model.set_causal_graph(graph)
        
        assert self.model.causal_graph is not None
        assert self.model.causal_graph.has_node("cause")
        assert self.model.causal_graph.has_node("effect")
        assert self.model.causal_graph.has_edge("cause", "effect")
    
    def test_get_all_hypotheses(self):
        """Test retrieving all hypotheses across spaces."""
        self.model.add_hypothesis_space(self.space)
        
        all_hypotheses = self.model.get_all_hypotheses()
        
        assert len(all_hypotheses) == 2
        assert "economic_growth" in all_hypotheses
        assert "political_stability" in all_hypotheses
    
    def test_get_all_evidence(self):
        """Test retrieving all evidence (global and space-specific)."""
        # Add global evidence
        global_evidence = BayesianEvidence(
            evidence_id="global_evidence",
            description="Global evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="global_node"
        )
        self.model.add_global_evidence(global_evidence)
        
        # Add space-specific evidence
        space_evidence = BayesianEvidence(
            evidence_id="space_evidence",
            description="Space-specific evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="space_node"
        )
        self.space.add_evidence(space_evidence)
        self.model.add_hypothesis_space(self.space)
        
        all_evidence = self.model.get_all_evidence()
        
        assert len(all_evidence) == 2
        assert "global_evidence" in all_evidence
        assert "space_evidence" in all_evidence
    
    def test_find_most_likely_hypothesis(self):
        """Test finding hypothesis with highest posterior."""
        self.model.add_hypothesis_space(self.space)
        
        # Set different posteriors
        self.space.hypotheses["economic_growth"].posterior_probability = 0.8
        self.space.hypotheses["political_stability"].posterior_probability = 0.2
        
        most_likely = self.model.find_most_likely_hypothesis()
        
        assert most_likely is not None
        assert most_likely[0] == "economic_growth"
        assert most_likely[1] == 0.8
        assert self.model.most_likely_hypothesis == "economic_growth"
    
    def test_calculate_model_confidence(self):
        """Test model confidence calculation."""
        self.model.add_hypothesis_space(self.space)
        
        # Set up scenario with clear winner
        self.space.hypotheses["economic_growth"].posterior_probability = 0.9
        self.space.hypotheses["political_stability"].posterior_probability = 0.1
        self.space.hypotheses["economic_growth"].confidence_level = 0.8
        self.space.hypotheses["political_stability"].confidence_level = 0.3
        
        confidence = self.model.calculate_model_confidence()
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to clear separation
        assert self.model.model_confidence == confidence
    
    def test_export_to_dict(self):
        """Test model export to dictionary."""
        self.model.add_hypothesis_space(self.space)
        
        export_dict = self.model.export_to_dict()
        
        assert export_dict["model_id"] == "test_model"
        assert export_dict["description"] == "Test Bayesian process tracing model"
        assert "hypothesis_spaces" in export_dict
        assert "test_space" in export_dict["hypothesis_spaces"]
        assert export_dict["global_evidence_count"] == 0
        assert "created_at" in export_dict
    
    def test_save_and_load_model(self):
        """Test saving and loading model from file."""
        self.model.add_hypothesis_space(self.space)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save model
            self.model.save_to_file(tmp_path)
            
            # Verify file exists and contains valid JSON
            assert Path(tmp_path).exists()
            
            with open(tmp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["model_id"] == "test_model"
            
            # Load model
            loaded_model = BayesianProcessTracingModel.load_from_file(tmp_path)
            
            assert loaded_model.model_id == "test_model"
            assert loaded_model.description == "Test Bayesian process tracing model"
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestMathematicalValidation:
    """Test mathematical correctness of Bayesian operations."""
    
    def test_bayes_theorem_basic(self):
        """Test basic Bayes' theorem calculation."""
        # P(H|E) = P(E|H) * P(H) / P(E)
        # Where P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        
        evidence = BayesianEvidence(
            evidence_id="math_test",
            description="Mathematical validation evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="math_node",
            likelihood_positive=0.9,  # P(E|H)
            likelihood_negative=0.1   # P(E|¬H)
        )
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="math_hyp",
            description="Mathematical validation hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5  # P(H)
        )
        
        # Calculate posterior using Bayes' theorem
        p_h = hypothesis.prior_probability
        p_not_h = 1 - p_h
        p_e_given_h = evidence.likelihood_positive
        p_e_given_not_h = evidence.likelihood_negative
        
        # P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        p_e = p_e_given_h * p_h + p_e_given_not_h * p_not_h
        
        # P(H|E) = P(E|H) * P(H) / P(E)
        expected_posterior = (p_e_given_h * p_h) / p_e
        
        # Calculate using likelihood ratio method
        likelihood_ratio = evidence.get_likelihood_ratio()
        lr_posterior = (likelihood_ratio * p_h) / (likelihood_ratio * p_h + p_not_h)
        
        # Both methods should give same result
        assert abs(expected_posterior - lr_posterior) < 1e-10
        
        # Verify our expected calculation
        expected_value = (0.9 * 0.5) / (0.9 * 0.5 + 0.1 * 0.5)
        assert abs(expected_posterior - expected_value) < 1e-10
        assert abs(expected_posterior - 0.9) < 1e-10
    
    def test_probability_normalization(self):
        """Test probability normalization across hypothesis space."""
        space = BayesianHypothesisSpace("norm_test", "Normalization test")
        
        # Create three hypotheses
        hyp1 = BayesianHypothesis("h1", "Hypothesis 1", HypothesisType.PRIMARY)
        hyp2 = BayesianHypothesis("h2", "Hypothesis 2", HypothesisType.ALTERNATIVE)
        hyp3 = BayesianHypothesis("h3", "Hypothesis 3", HypothesisType.ALTERNATIVE)
        
        # Set arbitrary probabilities
        hyp1.posterior_probability = 0.6
        hyp2.posterior_probability = 0.8
        hyp3.posterior_probability = 0.4
        
        space.add_hypothesis(hyp1)
        space.add_hypothesis(hyp2)
        space.add_hypothesis(hyp3)
        
        # Make them mutually exclusive
        space.add_mutual_exclusivity_group({"h1", "h2", "h3"})
        
        # Normalize
        space._normalize_probabilities()
        
        # Should sum to 1
        total = (space.hypotheses["h1"].posterior_probability +
                space.hypotheses["h2"].posterior_probability +
                space.hypotheses["h3"].posterior_probability)
        
        assert abs(total - 1.0) < 1e-10
        
        # Should preserve relative ratios
        assert space.hypotheses["h2"].posterior_probability > space.hypotheses["h1"].posterior_probability
        assert space.hypotheses["h1"].posterior_probability > space.hypotheses["h3"].posterior_probability
    
    def test_likelihood_ratio_properties(self):
        """Test mathematical properties of likelihood ratios."""
        # Test 1: LR = 1 means evidence is uninformative
        neutral_evidence = BayesianEvidence(
            evidence_id="neutral",
            description="Neutral evidence",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="neutral_node",
            likelihood_positive=0.5,
            likelihood_negative=0.5
        )
        
        assert abs(neutral_evidence.get_likelihood_ratio() - 1.0) < 1e-10
        
        # Test 2: LR > 1 supports hypothesis
        supporting_evidence = BayesianEvidence(
            evidence_id="supporting",
            description="Supporting evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="support_node",
            likelihood_positive=0.9,
            likelihood_negative=0.1
        )
        
        assert supporting_evidence.get_likelihood_ratio() > 1.0
        
        # Test 3: LR < 1 contradicts hypothesis
        contradicting_evidence = BayesianEvidence(
            evidence_id="contradicting",
            description="Contradicting evidence",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="contradict_node",
            likelihood_positive=0.2,
            likelihood_negative=0.8
        )
        
        assert contradicting_evidence.get_likelihood_ratio() < 1.0
    
    def test_van_evera_mathematical_consistency(self):
        """Test mathematical consistency of Van Evera classifications."""
        # Hoop test: high necessity, low sufficiency
        hoop = BayesianEvidence(
            evidence_id="hoop",
            description="Hoop test",
            evidence_type=EvidenceType.HOOP,
            source_node_id="hoop_node"
        )
        
        # For hoop tests: P(E|H) should be high, P(E|¬H) should be moderate
        assert hoop.likelihood_positive >= 0.8  # High necessity
        assert hoop.get_likelihood_ratio() > 1.0  # Should support hypothesis
        
        # Smoking gun: low necessity, high sufficiency
        smoking_gun = BayesianEvidence(
            evidence_id="smoking_gun",
            description="Smoking gun test",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="sg_node"
        )
        
        # For smoking gun: P(E|¬H) should be very low
        assert smoking_gun.likelihood_negative <= 0.1
        assert smoking_gun.get_likelihood_ratio() >= 8.0  # Strong support
        
        # Doubly decisive: high necessity and sufficiency
        doubly_decisive = BayesianEvidence(
            evidence_id="doubly_decisive",
            description="Doubly decisive test",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="dd_node"
        )
        
        # Should be strongest evidence
        assert doubly_decisive.get_likelihood_ratio() >= smoking_gun.get_likelihood_ratio()
        assert doubly_decisive.likelihood_positive >= 0.9
        assert doubly_decisive.likelihood_negative <= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])