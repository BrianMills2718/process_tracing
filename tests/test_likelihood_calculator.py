"""
Comprehensive test suite for likelihood ratio calculations.

Tests Van Evera diagnostic tests, frequency-based calculations,
mechanism-based analysis, and contextual likelihood adjustments.
"""

import pytest
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import json
from pathlib import Path

from core.likelihood_calculator import (
    VanEveraLikelihoodCalculator, FrequencyBasedLikelihoodCalculator,
    MechanismBasedLikelihoodCalculator, ContextualLikelihoodCalculator,
    LikelihoodCalculationOrchestrator, LikelihoodCalculationConfig,
    LikelihoodCalculationMethod
)
from core.bayesian_models import (
    BayesianHypothesis, BayesianEvidence, HypothesisType, EvidenceType
)


class TestVanEveraLikelihoodCalculator:
    """Test Van Evera likelihood ratio calculations."""
    
    def setup_method(self):
        """Set up test data."""
        self.config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.VAN_EVERA,
            van_evera_strictness=0.8,
            uncertainty_factor=0.1
        )
        self.calculator = VanEveraLikelihoodCalculator(self.config)
        
        self.hypothesis = BayesianHypothesis(
            hypothesis_id="test_hyp",
            description="Test hypothesis for likelihood calculation",
            hypothesis_type=HypothesisType.PRIMARY
        )
    
    def test_hoop_test_likelihood_ratio(self):
        """Test likelihood ratio calculation for hoop tests."""
        evidence = BayesianEvidence(
            evidence_id="hoop_evidence",
            description="Hoop test evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="test_node",
            reliability=1.0,
            strength=1.0,
            source_credibility=1.0
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Hoop tests should have moderate likelihood ratios (high necessity, low sufficiency)
        assert ratio > 1.0  # Should support hypothesis
        assert ratio < 10.0  # But not extremely strong support
        
        # P(E|H) should be high for hoop tests
        assert evidence.likelihood_positive >= 0.8
        
        # P(E|¬H) should be moderate (hoop tests can occur with alternative hypotheses)
        assert evidence.likelihood_negative <= 0.5
    
    def test_smoking_gun_likelihood_ratio(self):
        """Test likelihood ratio calculation for smoking gun tests."""
        evidence = BayesianEvidence(
            evidence_id="smoking_gun_evidence",
            description="Smoking gun evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            reliability=1.0,
            strength=1.0,
            source_credibility=1.0
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Smoking gun should have high likelihood ratios
        assert ratio >= 8.0  # Strong support for hypothesis
        
        # P(E|H) should be high
        assert evidence.likelihood_positive >= 0.8
        
        # P(E|¬H) should be very low (smoking gun rarely occurs with alternatives)
        assert evidence.likelihood_negative <= 0.1
    
    def test_doubly_decisive_likelihood_ratio(self):
        """Test likelihood ratio calculation for doubly decisive tests."""
        evidence = BayesianEvidence(
            evidence_id="doubly_decisive_evidence",
            description="Doubly decisive evidence",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="test_node",
            reliability=1.0,
            strength=1.0,
            source_credibility=1.0
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Doubly decisive should have the highest likelihood ratios
        assert ratio >= 9.0  # Very strong support
        
        # Should have both high necessity and sufficiency
        assert evidence.likelihood_positive >= 0.9
        assert evidence.likelihood_negative <= 0.1
    
    def test_straw_in_wind_likelihood_ratio(self):
        """Test likelihood ratio calculation for straw in the wind tests."""
        evidence = BayesianEvidence(
            evidence_id="straw_evidence",
            description="Straw in the wind evidence",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="test_node",
            reliability=1.0,
            strength=0.8,
            source_credibility=1.0
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Straw in the wind should provide weak support
        assert 1.0 < ratio < 3.0  # Weak to moderate support
        
        # Should have low necessity and sufficiency
        assert evidence.likelihood_positive <= 0.8
        assert evidence.likelihood_negative >= 0.3
    
    def test_strictness_factor_effect(self):
        """Test effect of Van Evera strictness factor."""
        evidence = BayesianEvidence(
            evidence_id="strictness_test",
            description="Test strictness effect",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            reliability=1.0,
            strength=1.0,
            source_credibility=1.0
        )
        
        # High strictness
        strict_config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.VAN_EVERA,
            van_evera_strictness=0.9
        )
        strict_calculator = VanEveraLikelihoodCalculator(strict_config)
        
        # Low strictness
        lenient_config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.VAN_EVERA,
            van_evera_strictness=0.3
        )
        lenient_calculator = VanEveraLikelihoodCalculator(lenient_config)
        
        strict_ratio = strict_calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        lenient_ratio = lenient_calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Strict interpretation should give stronger ratios for strong evidence types
        assert strict_ratio >= lenient_ratio
    
    def test_uncertainty_adjustment(self):
        """Test uncertainty factor adjustment of likelihood ratios."""
        evidence = BayesianEvidence(
            evidence_id="uncertainty_test",
            description="Test uncertainty adjustment",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            reliability=0.7,  # Moderate reliability
            strength=0.8,
            source_credibility=0.6  # Low source credibility
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Create perfect evidence for comparison
        perfect_evidence = BayesianEvidence(
            evidence_id="perfect_test",
            description="Perfect evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            reliability=1.0,
            strength=1.0,
            source_credibility=1.0
        )
        
        perfect_ratio = self.calculator.calculate_likelihood_ratio(perfect_evidence, self.hypothesis)
        
        # Uncertain evidence should have ratio closer to neutral (1.0)
        assert abs(ratio - 1.0) < abs(perfect_ratio - 1.0)
    
    def test_zero_false_positive_rate(self):
        """Test handling of zero false positive rate (infinite likelihood ratio)."""
        # Create evidence with zero P(E|¬H)
        evidence = BayesianEvidence(
            evidence_id="perfect_evidence",
            description="Perfect diagnostic evidence",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="test_node",
            likelihood_positive=1.0,
            likelihood_negative=0.0,
            reliability=1.0,
            strength=1.0,
            source_credibility=1.0
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Should handle infinite ratio gracefully
        assert ratio == float('inf') or ratio > 1000  # Very large ratio


class TestFrequencyBasedLikelihoodCalculator:
    """Test frequency-based likelihood calculations."""
    
    def setup_method(self):
        """Set up test data."""
        self.config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.FREQUENCY_BASED
        )
        self.calculator = FrequencyBasedLikelihoodCalculator(self.config)
        
        # Load test frequency data
        frequency_data = {
            "smoking_gun_evidence|primary_hypothesis|True": {"frequency": 0.8},
            "smoking_gun_evidence|primary_hypothesis|False": {"frequency": 0.1},
            "hoop_evidence|alternative_hypothesis|True": {"frequency": 0.9},
            "hoop_evidence|alternative_hypothesis|False": {"frequency": 0.4},
            "smoking_gun|True": {"frequency": 0.7},
            "smoking_gun|False": {"frequency": 0.2},
            "hoop|True": {"frequency": 0.85},
            "hoop|False": {"frequency": 0.3}
        }
        self.calculator.load_frequency_data(frequency_data)
        
        self.hypothesis = BayesianHypothesis(
            hypothesis_id="primary_hypothesis",
            description="Primary test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
    
    def test_frequency_based_exact_match(self):
        """Test frequency-based calculation with exact data match."""
        evidence = BayesianEvidence(
            evidence_id="smoking_gun_evidence",
            description="Evidence with exact frequency match",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            collection_method="direct_observation"
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Should use exact frequency data: P(E|H) = 0.8, P(E|¬H) = 0.1
        # After smoothing: (0.8 + 0.1) / (1 + 0.2) = 0.9/1.2 = 0.75
        # and (0.1 + 0.1) / (1 + 0.2) = 0.2/1.2 = 0.167
        # Ratio ≈ 0.75 / 0.167 ≈ 4.5
        assert 3.0 < ratio < 6.0  # Allow for smoothing effects
    
    def test_frequency_based_type_fallback(self):
        """Test fallback to evidence type when exact match unavailable."""
        evidence = BayesianEvidence(
            evidence_id="unknown_evidence",
            description="Evidence without exact frequency match",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            collection_method="unknown_method"
        )
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="unknown_hypothesis",
            description="Hypothesis without exact match",
            hypothesis_type=HypothesisType.ALTERNATIVE
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, hypothesis)
        
        # Should fall back to general smoking gun frequencies
        assert ratio > 1.0  # Should still support hypothesis
    
    def test_frequency_based_smoothing_prevents_zero(self):
        """Test that smoothing prevents zero probabilities."""
        # Use calculator with zero frequencies
        zero_calc = FrequencyBasedLikelihoodCalculator(self.config)
        zero_data = {
            "test_evidence|test_hypothesis|True": {"frequency": 0.0},
            "test_evidence|test_hypothesis|False": {"frequency": 0.0}
        }
        zero_calc.load_frequency_data(zero_data)
        
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="test_node",
            collection_method="test_method"
        )
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="test_hypothesis",
            description="Test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        ratio = zero_calc.calculate_likelihood_ratio(evidence, hypothesis)
        
        # Should not be zero or infinite due to smoothing
        assert 0.0 < ratio < float('inf')
        assert not math.isnan(ratio)


class TestMechanismBasedLikelihoodCalculator:
    """Test mechanism-based likelihood calculations."""
    
    def setup_method(self):
        """Set up test data."""
        self.config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.MECHANISM_BASED
        )
        self.calculator = MechanismBasedLikelihoodCalculator(self.config)
        
        # Set up mechanism data
        mechanism_strengths = {
            "economic_growth_mechanism": 0.8,
            "political_stability_mechanism": 0.6,
            "weak_mechanism": 0.3
        }
        
        pathway_probabilities = {
            "economic_growth_mechanism": 0.9,
            "political_stability_mechanism": 0.7,
            "weak_mechanism": 0.4
        }
        
        dependencies = {
            "economic_growth_mechanism": ["market_access", "investment_climate"],
            "political_stability_mechanism": ["governance", "rule_of_law"]
        }
        
        self.calculator.set_mechanism_data(mechanism_strengths, pathway_probabilities, dependencies)
        
        self.hypothesis = BayesianHypothesis(
            hypothesis_id="economic_development",
            description="Economic development hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
    
    def test_mechanism_based_with_relevant_mechanisms(self):
        """Test mechanism-based calculation with relevant mechanisms."""
        evidence = BayesianEvidence(
            evidence_id="growth_evidence",
            description="Evidence of economic growth mechanism",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="economic_growth_node",
            strength=0.9,
            reliability=0.8
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Should use mechanism strength and pathway probability
        assert ratio > 1.0  # Should support hypothesis
        
        # Stronger mechanisms should give higher ratios
        assert ratio > 2.0  # Should be substantial given strong mechanism
    
    def test_mechanism_based_weak_mechanisms(self):
        """Test mechanism-based calculation with weak mechanisms."""
        evidence = BayesianEvidence(
            evidence_id="weak_evidence",
            description="Evidence of weak mechanism",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="weak_mechanism_node",
            strength=0.5,
            reliability=0.7
        )
        
        weak_hypothesis = BayesianHypothesis(
            hypothesis_id="weak_hypothesis",
            description="Hypothesis with weak mechanism",
            hypothesis_type=HypothesisType.ALTERNATIVE
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, weak_hypothesis)
        
        # Should provide weak support due to weak mechanism
        assert 0.5 < ratio < 2.0
    
    def test_mechanism_based_no_relevant_mechanisms(self):
        """Test fallback when no relevant mechanisms found."""
        evidence = BayesianEvidence(
            evidence_id="unrelated_evidence",
            description="Evidence with no relevant mechanisms",
            evidence_type=EvidenceType.HOOP,
            source_node_id="unrelated_node",
            strength=0.8,
            reliability=0.9
        )
        
        unrelated_hypothesis = BayesianHypothesis(
            hypothesis_id="unrelated_hypothesis",
            description="Hypothesis with no mechanism connection",
            hypothesis_type=HypothesisType.NULL
        )
        
        ratio = self.calculator.calculate_likelihood_ratio(evidence, unrelated_hypothesis)
        
        # Should fall back to Van Evera calculation
        assert ratio > 0.0  # Should still provide some ratio
        
        # For hoop test, should be moderate support
        assert 1.0 < ratio < 10.0


class TestContextualLikelihoodCalculator:
    """Test contextual likelihood calculations."""
    
    def setup_method(self):
        """Set up test data."""
        self.config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.CONTEXTUAL,
            context_sensitivity=0.5,
            temporal_decay=0.05
        )
        self.calculator = ContextualLikelihoodCalculator(self.config)
        
        # Set context factors
        context_factors = {
            "economic": 0.8,    # Favorable economic context
            "political": 0.3,   # Unfavorable political context
            "social": 0.6,      # Moderate social context
            "crisis": 0.2       # Crisis context
        }
        self.calculator.set_context(context_factors, datetime.now())
        
        # Set interaction effects
        interaction_effects = {
            ("evidence1", "evidence2"): 1.3,  # Positive interaction
            ("evidence1", "evidence3"): 0.7,  # Negative interaction
        }
        self.calculator.set_interaction_effects(interaction_effects)
        
        self.hypothesis = BayesianHypothesis(
            hypothesis_id="contextual_hypothesis",
            description="Economic growth in political crisis",
            hypothesis_type=HypothesisType.PRIMARY
        )
    
    def test_contextual_temporal_decay(self):
        """Test temporal decay factor."""
        # Recent evidence
        recent_evidence = BayesianEvidence(
            evidence_id="recent_evidence",
            description="Recent evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            timestamp=datetime.now() - timedelta(days=30)
        )
        
        # Old evidence
        old_evidence = BayesianEvidence(
            evidence_id="old_evidence",
            description="Old evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            timestamp=datetime.now() - timedelta(days=365*2)  # 2 years old
        )
        
        recent_ratio = self.calculator.calculate_likelihood_ratio(recent_evidence, self.hypothesis)
        old_ratio = self.calculator.calculate_likelihood_ratio(old_evidence, self.hypothesis)
        
        # Recent evidence should have higher effective ratio
        assert recent_ratio >= old_ratio
    
    def test_contextual_factor_adjustment(self):
        """Test context factor adjustment."""
        # Evidence mentioning favorable context
        favorable_evidence = BayesianEvidence(
            evidence_id="favorable_evidence",
            description="Evidence in favorable economic context",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node"
        )
        
        # Evidence mentioning unfavorable context
        unfavorable_evidence = BayesianEvidence(
            evidence_id="unfavorable_evidence",
            description="Evidence in political crisis context",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node"
        )
        
        favorable_ratio = self.calculator.calculate_likelihood_ratio(favorable_evidence, self.hypothesis)
        unfavorable_ratio = self.calculator.calculate_likelihood_ratio(unfavorable_evidence, self.hypothesis)
        
        # Favorable context should boost likelihood ratio
        assert favorable_ratio >= unfavorable_ratio
    
    def test_interaction_effects(self):
        """Test evidence interaction effects."""
        evidence1 = BayesianEvidence(
            evidence_id="evidence1",
            description="First evidence piece",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node"
        )
        
        evidence2 = BayesianEvidence(
            evidence_id="evidence2",
            description="Second evidence piece (positive interaction)",
            evidence_type=EvidenceType.HOOP,
            source_node_id="test_node"
        )
        
        evidence3 = BayesianEvidence(
            evidence_id="evidence3",
            description="Third evidence piece (negative interaction)",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="test_node"
        )
        
        # Calculate with no other evidence
        ratio_alone = self.calculator.calculate_likelihood_ratio(evidence1, self.hypothesis)
        
        # Calculate with positive interaction
        ratio_positive = self.calculator.calculate_likelihood_ratio(evidence1, self.hypothesis, [evidence2])
        
        # Calculate with negative interaction
        ratio_negative = self.calculator.calculate_likelihood_ratio(evidence1, self.hypothesis, [evidence3])
        
        # Positive interaction should boost ratio
        assert ratio_positive >= ratio_alone
        
        # Negative interaction should reduce ratio
        assert ratio_negative <= ratio_alone
    
    def test_contextual_no_context_data(self):
        """Test behavior when no context data is available."""
        no_context_calc = ContextualLikelihoodCalculator(self.config)
        # Don't set any context data
        
        evidence = BayesianEvidence(
            evidence_id="no_context_evidence",
            description="Evidence without context",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node"
        )
        
        ratio = no_context_calc.calculate_likelihood_ratio(evidence, self.hypothesis)
        
        # Should fall back to base Van Evera calculation
        assert ratio > 1.0  # Smoking gun should support hypothesis


class TestLikelihoodCalculationOrchestrator:
    """Test likelihood calculation orchestrator."""
    
    def setup_method(self):
        """Set up test orchestrator."""
        self.orchestrator = LikelihoodCalculationOrchestrator()
        
        self.hypothesis = BayesianHypothesis(
            hypothesis_id="test_hypothesis",
            description="Test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        self.evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node"
        )
    
    def test_orchestrator_van_evera_calculation(self):
        """Test orchestrator with Van Evera method."""
        config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.VAN_EVERA,
            van_evera_strictness=0.8
        )
        
        ratio = self.orchestrator.calculate_likelihood_ratio(self.evidence, self.hypothesis, config)
        
        assert ratio > 0.0
        assert len(self.orchestrator.calculation_history) == 1
        
        # Check calculation record
        record = self.orchestrator.calculation_history[0]
        assert record["method"] == "van_evera"
        assert record["evidence_id"] == "test_evidence"
        assert record["hypothesis_id"] == "test_hypothesis"
        assert record["likelihood_ratio"] == ratio
    
    def test_orchestrator_frequency_based_calculation(self):
        """Test orchestrator with frequency-based method."""
        config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.FREQUENCY_BASED
        )
        
        additional_data = {
            "frequency_data": {
                "test_evidence|test_hypothesis|True": {"frequency": 0.8},
                "test_evidence|test_hypothesis|False": {"frequency": 0.2}
            }
        }
        
        ratio = self.orchestrator.calculate_likelihood_ratio(
            self.evidence, self.hypothesis, config, additional_data
        )
        
        assert ratio > 0.0
        assert len(self.orchestrator.calculation_history) == 1
    
    def test_orchestrator_batch_calculation(self):
        """Test batch likelihood ratio calculation."""
        evidence_list = [
            BayesianEvidence(
                evidence_id=f"evidence_{i}",
                description=f"Test evidence {i}",
                evidence_type=EvidenceType.SMOKING_GUN,
                source_node_id=f"node_{i}"
            )
            for i in range(3)
        ]
        
        config = LikelihoodCalculationConfig(method=LikelihoodCalculationMethod.VAN_EVERA)
        
        ratios = self.orchestrator.batch_calculate_likelihood_ratios(
            evidence_list, self.hypothesis, config
        )
        
        assert len(ratios) == 3
        assert all(ratio > 0.0 for ratio in ratios.values())
        assert len(self.orchestrator.calculation_history) == 3
    
    def test_orchestrator_uncertainty_analysis(self):
        """Test uncertainty analysis."""
        config = LikelihoodCalculationConfig(
            method=LikelihoodCalculationMethod.VAN_EVERA,
            van_evera_strictness=0.5,
            uncertainty_factor=0.1
        )
        
        uncertainty_ranges = {
            "van_evera_strictness": (0.1, 0.9),
            "uncertainty_factor": (0.0, 0.3)
        }
        
        uncertainty_results = self.orchestrator.uncertainty_analysis(
            self.evidence, self.hypothesis, config, uncertainty_ranges
        )
        
        assert "base_likelihood_ratio" in uncertainty_results
        assert "parameter_uncertainty" in uncertainty_results
        assert "confidence_interval" in uncertainty_results
        assert "sensitivity_score" in uncertainty_results
        
        # Should have uncertainty results for each parameter
        assert "van_evera_strictness" in uncertainty_results["parameter_uncertainty"]
        assert "uncertainty_factor" in uncertainty_results["parameter_uncertainty"]
        
        # Confidence interval should be a tuple
        ci = uncertainty_results["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= ci[1]  # Lower bound <= upper bound
    
    def test_orchestrator_calculation_summary(self):
        """Test calculation summary generation."""
        # Perform several calculations
        config = LikelihoodCalculationConfig(method=LikelihoodCalculationMethod.VAN_EVERA)
        
        for i in range(5):
            evidence = BayesianEvidence(
                evidence_id=f"evidence_{i}",
                description=f"Evidence {i}",
                evidence_type=EvidenceType.SMOKING_GUN,
                source_node_id=f"node_{i}"
            )
            self.orchestrator.calculate_likelihood_ratio(evidence, self.hypothesis, config)
        
        summary = self.orchestrator.get_calculation_summary()
        
        assert summary["total_calculations"] == 5
        assert "van_evera" in summary["methods_used"]
        assert summary["methods_used"]["van_evera"] == 5
        assert "likelihood_ratio_stats" in summary
        assert "mean" in summary["likelihood_ratio_stats"]
        assert "std" in summary["likelihood_ratio_stats"]
        assert summary["unique_evidence"] == 5
        assert summary["unique_hypotheses"] == 1
    
    def test_orchestrator_export_history(self):
        """Test calculation history export."""
        config = LikelihoodCalculationConfig(method=LikelihoodCalculationMethod.VAN_EVERA)
        self.orchestrator.calculate_likelihood_ratio(self.evidence, self.hypothesis, config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.orchestrator.export_calculation_history(tmp_path)
            
            # Verify export
            assert Path(tmp_path).exists()
            
            with open(tmp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) == 1
            assert exported_data[0]["method"] == "van_evera"
            assert exported_data[0]["evidence_id"] == "test_evidence"
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestMathematicalValidation:
    """Test mathematical correctness of likelihood calculations."""
    
    def test_likelihood_ratio_properties(self):
        """Test mathematical properties of likelihood ratios."""
        config = LikelihoodCalculationConfig(method=LikelihoodCalculationMethod.VAN_EVERA)
        calculator = VanEveraLikelihoodCalculator(config)
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="math_test",
            description="Mathematical test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        # Test 1: LR = 1 should be neutral
        neutral_evidence = BayesianEvidence(
            evidence_id="neutral",
            description="Neutral evidence",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="neutral_node",
            likelihood_positive=0.5,
            likelihood_negative=0.5
        )
        
        neutral_ratio = calculator.calculate_likelihood_ratio(neutral_evidence, hypothesis)
        assert abs(neutral_ratio - 1.0) < 0.2  # Allow for Van Evera adjustments
        
        # Test 2: Higher P(E|H) should give higher LR
        high_pos_evidence = BayesianEvidence(
            evidence_id="high_pos",
            description="High P(E|H) evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="high_node",
            likelihood_positive=0.9,
            likelihood_negative=0.1
        )
        
        low_pos_evidence = BayesianEvidence(
            evidence_id="low_pos",
            description="Low P(E|H) evidence",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="low_node",
            likelihood_positive=0.6,
            likelihood_negative=0.1
        )
        
        high_ratio = calculator.calculate_likelihood_ratio(high_pos_evidence, hypothesis)
        low_ratio = calculator.calculate_likelihood_ratio(low_pos_evidence, hypothesis)
        
        assert high_ratio > low_ratio
        
        # Test 3: Lower P(E|¬H) should give higher LR
        low_neg_evidence = BayesianEvidence(
            evidence_id="low_neg",
            description="Low P(E|¬H) evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="low_neg_node",
            likelihood_positive=0.8,
            likelihood_negative=0.05
        )
        
        high_neg_evidence = BayesianEvidence(
            evidence_id="high_neg",
            description="High P(E|¬H) evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="high_neg_node",
            likelihood_positive=0.8,
            likelihood_negative=0.4
        )
        
        low_neg_ratio = calculator.calculate_likelihood_ratio(low_neg_evidence, hypothesis)
        high_neg_ratio = calculator.calculate_likelihood_ratio(high_neg_evidence, hypothesis)
        
        assert low_neg_ratio > high_neg_ratio
    
    def test_van_evera_classification_consistency(self):
        """Test consistency of Van Evera classifications."""
        config = LikelihoodCalculationConfig(method=LikelihoodCalculationMethod.VAN_EVERA)
        calculator = VanEveraLikelihoodCalculator(config)
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="consistency_test",
            description="Consistency test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        # Create evidence of each Van Evera type
        evidence_types = [
            (EvidenceType.HOOP, "hoop"),
            (EvidenceType.SMOKING_GUN, "smoking_gun"),
            (EvidenceType.DOUBLY_DECISIVE, "doubly_decisive"),
            (EvidenceType.STRAW_IN_THE_WIND, "straw")
        ]
        
        ratios = {}
        for ev_type, name in evidence_types:
            evidence = BayesianEvidence(
                evidence_id=f"{name}_evidence",
                description=f"{name} evidence",
                evidence_type=ev_type,
                source_node_id=f"{name}_node",
                reliability=1.0,
                strength=1.0,
                source_credibility=1.0
            )
            
            ratios[name] = calculator.calculate_likelihood_ratio(evidence, hypothesis)
        
        # Doubly decisive should have highest ratio
        assert ratios["doubly_decisive"] >= ratios["smoking_gun"]
        assert ratios["doubly_decisive"] >= ratios["hoop"]
        assert ratios["doubly_decisive"] >= ratios["straw"]
        
        # Smoking gun should have higher ratio than hoop
        assert ratios["smoking_gun"] >= ratios["hoop"]
        
        # All should be positive
        assert all(ratio > 0 for ratio in ratios.values())
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = LikelihoodCalculationConfig(method=LikelihoodCalculationMethod.VAN_EVERA)
        calculator = VanEveraLikelihoodCalculator(config)
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="stability_test",
            description="Numerical stability test",
            hypothesis_type=HypothesisType.PRIMARY
        )
        
        # Test with very small probabilities
        tiny_evidence = BayesianEvidence(
            evidence_id="tiny",
            description="Tiny probability evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="tiny_node",
            likelihood_positive=0.001,
            likelihood_negative=0.0001
        )
        
        tiny_ratio = calculator.calculate_likelihood_ratio(tiny_evidence, hypothesis)
        assert not math.isnan(tiny_ratio)
        assert not math.isinf(tiny_ratio)
        assert tiny_ratio > 0
        
        # Test with probabilities close to 1
        large_evidence = BayesianEvidence(
            evidence_id="large",
            description="Large probability evidence",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="large_node",
            likelihood_positive=0.999,
            likelihood_negative=0.001
        )
        
        large_ratio = calculator.calculate_likelihood_ratio(large_evidence, hypothesis)
        assert not math.isnan(large_ratio)
        assert large_ratio > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])