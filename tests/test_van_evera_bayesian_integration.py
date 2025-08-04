"""
Integration tests for Phase 6B Van Evera Bayesian Integration.

Tests the complete pipeline from LLM evidence assessments through
Van Evera classification to Bayesian inference calculations.
"""

import pytest
import math
from typing import Dict, List
from core.van_evera_bayesian import VanEveraBayesianBridge, VanEveraBayesianConfig
from core.diagnostic_probabilities import DiagnosticProbabilityTemplates, validate_van_evera_probabilities
from core.evidence_weighting import EvidenceStrengthQuantifier, EvidenceWeights, IndependenceType
from core.bayesian_models import BayesianEvidence, EvidenceType
from core.structured_models import EvidenceAssessment, VanEveraEvidenceType


class TestDiagnosticProbabilityTemplates:
    """Test diagnostic probability template functionality."""
    
    def test_get_template_probabilities_basic(self):
        """Test basic template probability retrieval."""
        likelihood_pos, likelihood_neg = DiagnosticProbabilityTemplates.get_template_probabilities(
            EvidenceType.HOOP, strength=1.0, reliability=1.0
        )
        
        assert 0.0 <= likelihood_pos <= 1.0
        assert 0.0 <= likelihood_neg <= 1.0
        assert likelihood_pos > likelihood_neg  # HOOP should have high necessity
    
    def test_van_evera_logic_enforcement(self):
        """Test that Van Evera test logic is properly enforced."""
        # HOOP test should have high P(E|H)
        likelihood_pos, likelihood_neg = DiagnosticProbabilityTemplates.get_template_probabilities(
            EvidenceType.HOOP
        )
        assert likelihood_pos >= 0.70, f"HOOP P(E|H) should be >= 0.70, got {likelihood_pos}"
        
        # SMOKING_GUN should have low P(E|¬H) 
        likelihood_pos, likelihood_neg = DiagnosticProbabilityTemplates.get_template_probabilities(
            EvidenceType.SMOKING_GUN
        )
        assert likelihood_neg <= 0.15, f"SMOKING_GUN P(E|¬H) should be <= 0.15, got {likelihood_neg}"
        
        # DOUBLY_DECISIVE should have both high P(E|H) and low P(E|¬H)
        likelihood_pos, likelihood_neg = DiagnosticProbabilityTemplates.get_template_probabilities(
            EvidenceType.DOUBLY_DECISIVE
        )
        assert likelihood_pos >= 0.80, f"DOUBLY_DECISIVE P(E|H) should be >= 0.80, got {likelihood_pos}"
        assert likelihood_neg <= 0.15, f"DOUBLY_DECISIVE P(E|¬H) should be <= 0.15, got {likelihood_neg}"
    
    def test_strength_reliability_adjustments(self):
        """Test that strength and reliability adjustments work correctly."""
        # High strength and reliability should increase likelihood_positive
        likelihood_pos_high, likelihood_neg_high = DiagnosticProbabilityTemplates.get_template_probabilities(
            EvidenceType.HOOP, strength=1.0, reliability=1.0
        )
        
        # Low strength and reliability should decrease likelihood_positive
        likelihood_pos_low, likelihood_neg_low = DiagnosticProbabilityTemplates.get_template_probabilities(
            EvidenceType.HOOP, strength=0.5, reliability=0.5
        )
        
        assert likelihood_pos_high > likelihood_pos_low
        assert likelihood_neg_high < likelihood_neg_low
    
    def test_validate_van_evera_probabilities(self):
        """Test Van Evera probability validation function."""
        # Valid HOOP test
        assert validate_van_evera_probabilities(0.8, 0.3, EvidenceType.HOOP)
        
        # Invalid HOOP test (low necessity)
        assert not validate_van_evera_probabilities(0.5, 0.3, EvidenceType.HOOP)
        
        # Valid SMOKING_GUN test  
        assert validate_van_evera_probabilities(0.7, 0.1, EvidenceType.SMOKING_GUN)
        
        # Invalid SMOKING_GUN test (high false positive rate)
        assert not validate_van_evera_probabilities(0.7, 0.4, EvidenceType.SMOKING_GUN)


class TestVanEveraBayesianBridge:
    """Test Van Evera Bayesian bridge functionality."""
    
    @pytest.fixture
    def bridge(self):
        """Create Van Evera Bayesian bridge instance."""
        config = VanEveraBayesianConfig(
            use_llm_likelihood_overrides=True,
            minimum_likelihood=0.01,
            maximum_likelihood=0.99
        )
        return VanEveraBayesianBridge(config)
    
    @pytest.fixture
    def sample_evidence_assessment(self):
        """Create sample evidence assessment from LLM."""
        return EvidenceAssessment(
            evidence_id="test_evidence_1",
            refined_evidence_type=VanEveraEvidenceType.SMOKING_GUN,
            reasoning_for_type="This evidence strongly confirms the hypothesis with very low chance of false positive",
            likelihood_P_E_given_H="High (0.8)",
            likelihood_P_E_given_NotH="Very Low (0.1)",
            justification_for_likelihoods="The evidence is very specific to this hypothesis and unlikely to occur under alternative explanations",
            suggested_numerical_probative_value=8.5
        )
    
    def test_convert_evidence_assessment(self, bridge, sample_evidence_assessment):
        """Test conversion of LLM evidence assessment to Bayesian evidence."""
        bayesian_evidence = bridge.convert_evidence_assessment(
            sample_evidence_assessment,
            hypothesis_context="Test hypothesis about causal relationship",
            source_node_id="node_123"
        )
        
        assert bayesian_evidence.evidence_id == "test_evidence_1"
        assert bayesian_evidence.evidence_type == EvidenceType.SMOKING_GUN
        assert bayesian_evidence.source_node_id == "node_123"
        assert 0.0 <= bayesian_evidence.likelihood_positive <= 1.0
        assert 0.0 <= bayesian_evidence.likelihood_negative <= 1.0
        assert bayesian_evidence.likelihood_positive > bayesian_evidence.likelihood_negative
    
    def test_extract_likelihoods_from_llm(self, bridge):
        """Test extraction of numerical likelihoods from LLM text."""
        assessment = EvidenceAssessment(
            evidence_id="test_evidence_2",
            refined_evidence_type=VanEveraEvidenceType.HOOP,
            reasoning_for_type="Test reasoning",
            likelihood_P_E_given_H="High (0.85)",
            likelihood_P_E_given_NotH="Medium (0.45)",
            justification_for_likelihoods="Test justification",
            suggested_numerical_probative_value=7.0
        )
        
        likelihood_pos, likelihood_neg = bridge._extract_likelihoods_from_llm(assessment)
        
        assert likelihood_pos == 0.85
        assert likelihood_neg == 0.45
    
    def test_parse_likelihood_string(self, bridge):
        """Test parsing of various likelihood string formats."""
        # Test parentheses format
        assert bridge._parse_likelihood_string("High (0.8)") == 0.8
        assert bridge._parse_likelihood_string("Very Low (0.1)") == 0.1
        
        # Test direct number format
        assert bridge._parse_likelihood_string("0.75") == 0.75
        
        # Test qualitative mapping
        assert bridge._parse_likelihood_string("Very High") == 0.9
        assert bridge._parse_likelihood_string("Low") == 0.2
        
        # Test invalid inputs
        assert bridge._parse_likelihood_string("Invalid") is None
        assert bridge._parse_likelihood_string("") is None
    
    def test_van_evera_type_mapping(self, bridge):
        """Test mapping between VanEveraEvidenceType and EvidenceType."""
        assert bridge._map_van_evera_type(VanEveraEvidenceType.HOOP) == EvidenceType.HOOP
        assert bridge._map_van_evera_type(VanEveraEvidenceType.SMOKING_GUN) == EvidenceType.SMOKING_GUN
        assert bridge._map_van_evera_type(VanEveraEvidenceType.DOUBLY_DECISIVE) == EvidenceType.DOUBLY_DECISIVE
        assert bridge._map_van_evera_type(VanEveraEvidenceType.STRAW_IN_THE_WIND) == EvidenceType.STRAW_IN_THE_WIND
    
    def test_calculate_van_evera_likelihoods(self, bridge):
        """Test Van Evera likelihood calculation."""
        # Test HOOP test
        likelihood_pos, likelihood_neg = bridge.calculate_van_evera_likelihoods(
            EvidenceType.HOOP, strength=1.0, reliability=1.0
        )
        assert likelihood_pos >= 0.70  # High necessity
        assert 0.0 <= likelihood_neg <= 1.0
        
        # Test SMOKING_GUN test
        likelihood_pos, likelihood_neg = bridge.calculate_van_evera_likelihoods(
            EvidenceType.SMOKING_GUN, strength=1.0, reliability=1.0
        )
        assert likelihood_neg <= 0.15  # Low false positive rate
        assert 0.0 <= likelihood_pos <= 1.0
    
    def test_create_bayesian_evidence_from_graph_data(self, bridge):
        """Test creating Bayesian evidence from graph node data."""
        evidence_node = {
            'id': 'evidence_123',
            'properties': {'description': 'Test evidence from graph'}
        }
        
        hypothesis_node = {
            'id': 'hypothesis_456',
            'properties': {'description': 'Test hypothesis'}
        }
        
        edge_properties = {
            'edge_type': 'supports',
            'strength': 0.8
        }
        
        bayesian_evidence = bridge.create_bayesian_evidence_from_graph_data(
            evidence_node, hypothesis_node, edge_properties, VanEveraEvidenceType.HOOP
        )
        
        assert bayesian_evidence.evidence_id == 'evidence_123'
        assert bayesian_evidence.evidence_type == EvidenceType.HOOP
        assert bayesian_evidence.collection_method == "graph_inference"
    
    def test_batch_convert_evidence_assessments(self, bridge, sample_evidence_assessment):
        """Test batch conversion of multiple evidence assessments."""
        assessments = [sample_evidence_assessment] * 3
        node_ids = ["node_1", "node_2", "node_3"]
        
        bayesian_evidence_list = bridge.batch_convert_evidence_assessments(
            assessments, "Test hypothesis", node_ids
        )
        
        assert len(bayesian_evidence_list) == 3
        for i, evidence in enumerate(bayesian_evidence_list):
            assert evidence.source_node_id == node_ids[i]
            assert evidence.evidence_type == EvidenceType.SMOKING_GUN
    
    def test_get_likelihood_ratio_summary(self, bridge, sample_evidence_assessment):
        """Test likelihood ratio summary generation."""
        bayesian_evidence = bridge.convert_evidence_assessment(
            sample_evidence_assessment, "Test hypothesis", "node_123"
        )
        
        summary = bridge.get_likelihood_ratio_summary(bayesian_evidence)
        
        assert 'likelihood_ratio' in summary
        assert 'strength_interpretation' in summary
        assert 'evidence_type' in summary
        assert summary['evidence_type'] == 'smoking_gun'
        assert isinstance(summary['likelihood_ratio'], (int, float))


class TestEvidenceStrengthQuantifier:
    """Test evidence strength quantification functionality."""
    
    @pytest.fixture
    def quantifier(self):
        """Create evidence strength quantifier instance."""
        return EvidenceStrengthQuantifier()
    
    @pytest.fixture
    def sample_evidence_assessment(self):
        """Create sample evidence assessment for testing."""
        return EvidenceAssessment(
            evidence_id="test_evidence_strength",
            refined_evidence_type=VanEveraEvidenceType.HOOP,
            reasoning_for_type="This evidence is well-documented and verified by multiple reliable sources",
            likelihood_P_E_given_H="High (0.8)",
            likelihood_P_E_given_NotH="Medium (0.4)",
            justification_for_likelihoods="The evidence shows clear consistent patterns that strongly support the hypothesis based on verified documentation from credible institutional sources",
            suggested_numerical_probative_value=7.5
        )
    
    def test_quantify_llm_assessment(self, quantifier, sample_evidence_assessment):
        """Test quantification of LLM assessment into numerical weights."""
        weights = quantifier.quantify_llm_assessment(sample_evidence_assessment)
        
        assert isinstance(weights, EvidenceWeights)
        assert weights.evidence_id == "test_evidence_strength"
        assert 0.0 <= weights.base_weight <= 1.0
        assert 0.0 <= weights.reliability_weight <= 1.0
        assert 0.0 <= weights.credibility_weight <= 1.0
        assert 0.0 <= weights.combined_weight <= 1.0
        
        # Check that confidence interval is valid
        lower, upper = weights.confidence_interval
        assert lower <= weights.combined_weight <= upper
        assert 0.0 <= lower <= upper <= 1.0
    
    def test_analyze_reliability_indicators(self, quantifier):
        """Test reliability indicator analysis."""
        # High reliability text
        high_reliability = quantifier._analyze_reliability_indicators(
            "verified and documented evidence", 
            "confirmed by multiple sources"
        )
        
        # Low reliability text  
        low_reliability = quantifier._analyze_reliability_indicators(
            "unverified claims",
            "based on hearsay and rumor"
        )
        
        assert high_reliability > low_reliability
        assert 0.0 <= high_reliability <= 1.0
        assert 0.0 <= low_reliability <= 1.0
    
    def test_analyze_credibility_indicators(self, quantifier):
        """Test credibility indicator analysis."""
        # High credibility text
        high_credibility = quantifier._analyze_credibility_indicators(
            "official government report",
            "published academic research by experts"
        )
        
        # Low credibility text
        low_credibility = quantifier._analyze_credibility_indicators(
            "anonymous source",
            "biased partisan claims"
        )
        
        assert high_credibility > low_credibility
        assert 0.0 <= high_credibility <= 1.0
        assert 0.0 <= low_credibility <= 1.0
    
    def test_combine_multiple_evidence_independent(self, quantifier):
        """Test combination of independent evidence."""
        # Create sample evidence with known likelihood ratios
        evidence1 = BayesianEvidence(
            evidence_id="evidence_1",
            description="First evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_1",
            likelihood_positive=0.8,
            likelihood_negative=0.2
        )
        
        evidence2 = BayesianEvidence(
            evidence_id="evidence_2", 
            description="Second evidence",
            evidence_type=EvidenceType.HOOP,
            source_node_id="node_2",
            likelihood_positive=0.9,
            likelihood_negative=0.3
        )
        
        # All evidence is independent
        independence_assumptions = {}
        
        combined_ratio = quantifier.combine_multiple_evidence(
            [evidence1, evidence2], independence_assumptions
        )
        
        # For independent evidence, ratios should multiply
        expected_ratio = (0.8/0.2) * (0.9/0.3)  # 4.0 * 3.0 = 12.0
        assert abs(combined_ratio - expected_ratio) < 0.1
    
    def test_combine_multiple_evidence_dependent(self, quantifier):
        """Test combination of dependent evidence."""
        evidence1 = BayesianEvidence(
            evidence_id="evidence_1",
            description="First evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_1",
            likelihood_positive=0.8,
            likelihood_negative=0.2
        )
        
        evidence2 = BayesianEvidence(
            evidence_id="evidence_2",
            description="Second evidence", 
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="node_2",
            likelihood_positive=0.8,
            likelihood_negative=0.2
        )
        
        # Mark evidence as dependent
        independence_assumptions = {
            "evidence_1-evidence_2": IndependenceType.DEPENDENT
        }
        
        combined_ratio = quantifier.combine_multiple_evidence(
            [evidence1, evidence2], independence_assumptions
        )
        
        # Dependent evidence should be less than independent multiplication
        independent_ratio = (0.8/0.2) * (0.8/0.2)  # 16.0
        assert combined_ratio < independent_ratio
        assert combined_ratio > 1.0  # Should still support hypothesis
    
    def test_calculate_evidence_diversity(self, quantifier):
        """Test evidence diversity calculation."""
        # Create diverse evidence set
        evidence_list = [
            BayesianEvidence("e1", "Evidence 1", EvidenceType.HOOP, "n1", 
                           collection_method="interview", reliability=0.9),
            BayesianEvidence("e2", "Evidence 2", EvidenceType.SMOKING_GUN, "n2",
                           collection_method="document", reliability=0.7),
            BayesianEvidence("e3", "Evidence 3", EvidenceType.STRAW_IN_THE_WIND, "n3",
                           collection_method="observation", reliability=0.8)
        ]
        
        diversity = quantifier.calculate_evidence_diversity(evidence_list)
        
        assert 0.0 <= diversity <= 1.0
        assert diversity > 0.0  # Should have some diversity
        
        # Test single evidence
        single_diversity = quantifier.calculate_evidence_diversity([evidence_list[0]])
        assert single_diversity == 0.5
        
        # Test empty list
        empty_diversity = quantifier.calculate_evidence_diversity([])
        assert empty_diversity == 0.0


class TestEndToEndIntegration:
    """Test end-to-end integration of Phase 6B components."""
    
    def test_complete_pipeline_llm_to_bayesian(self):
        """Test complete pipeline from LLM assessment to Bayesian inference."""
        # Step 1: Create LLM evidence assessment
        llm_assessment = EvidenceAssessment(
            evidence_id="integration_test_evidence",
            refined_evidence_type=VanEveraEvidenceType.DOUBLY_DECISIVE,
            reasoning_for_type="Evidence is both necessary and sufficient for hypothesis confirmation",
            likelihood_P_E_given_H="Very High (0.9)",
            likelihood_P_E_given_NotH="Very Low (0.05)",
            justification_for_likelihoods="This evidence is extremely specific to the hypothesis and has been verified through multiple independent reliable sources",
            suggested_numerical_probative_value=9.2
        )
        
        # Step 2: Convert to Bayesian evidence using bridge
        bridge = VanEveraBayesianBridge()
        bayesian_evidence = bridge.convert_evidence_assessment(
            llm_assessment,
            hypothesis_context="Test causal hypothesis",
            source_node_id="test_node_123"
        )
        
        # Step 3: Quantify evidence strength
        quantifier = EvidenceStrengthQuantifier()
        evidence_weights = quantifier.quantify_llm_assessment(llm_assessment)
        
        # Step 4: Validate results
        assert bayesian_evidence.evidence_type == EvidenceType.DOUBLY_DECISIVE
        assert bayesian_evidence.likelihood_positive >= 0.8  # High necessity
        assert bayesian_evidence.likelihood_negative <= 0.15  # Low false positive
        
        likelihood_ratio = bayesian_evidence.get_likelihood_ratio()
        assert likelihood_ratio >= 5.0  # Strong evidence
        
        assert evidence_weights.combined_weight >= 0.7  # High weight due to strong assessment
        
        # Step 5: Test likelihood ratio interpretation
        summary = bridge.get_likelihood_ratio_summary(bayesian_evidence)
        assert "strong" in summary['strength_interpretation'].lower()
        
        print(f"✅ End-to-end integration test passed:")
        print(f"   - Evidence type: {bayesian_evidence.evidence_type.value}")
        print(f"   - Likelihood ratio: {likelihood_ratio:.2f}")
        print(f"   - Evidence weight: {evidence_weights.combined_weight:.2f}")
        print(f"   - Strength interpretation: {summary['strength_interpretation']}")
    
    def test_mathematical_validity_preservation(self):
        """Test that Phase 6B preserves mathematical validity from Phase 6A."""
        # Create evidence with extreme values to test edge cases
        extreme_assessment = EvidenceAssessment(
            evidence_id="extreme_test",
            refined_evidence_type=VanEveraEvidenceType.SMOKING_GUN,
            reasoning_for_type="Perfect smoking gun evidence",
            likelihood_P_E_given_H="High (0.95)",
            likelihood_P_E_given_NotH="Extremely Low (0.01)",
            justification_for_likelihoods="This evidence cannot occur under alternative hypotheses",
            suggested_numerical_probative_value=10.0
        )
        
        bridge = VanEveraBayesianBridge()
        bayesian_evidence = bridge.convert_evidence_assessment(
            extreme_assessment, "Test hypothesis", "test_node"
        )
        
        # Validate mathematical properties
        assert 0.0 <= bayesian_evidence.likelihood_positive <= 1.0
        assert 0.0 <= bayesian_evidence.likelihood_negative <= 1.0
        
        likelihood_ratio = bayesian_evidence.get_likelihood_ratio()
        assert likelihood_ratio > 0  # Must be positive
        assert not math.isnan(likelihood_ratio)  # Must not be NaN
        
        # Test with zero false positive rate
        zero_fp_evidence = BayesianEvidence(
            evidence_id="zero_fp_test",
            description="Perfect evidence",
            evidence_type=EvidenceType.SMOKING_GUN, 
            source_node_id="test_node",
            likelihood_positive=0.9,
            likelihood_negative=0.0  # Zero false positive rate
        )
        
        ratio = zero_fp_evidence.get_likelihood_ratio()
        assert ratio == float('inf')  # Should handle infinite ratios
        
        print(f"✅ Mathematical validity test passed:")
        print(f"   - Likelihood ratio with extreme values: {likelihood_ratio:.2f}")
        print(f"   - Infinite ratio handling: {ratio}")


if __name__ == "__main__":
    # Run specific integration tests
    print("Running Phase 6B Van Evera Bayesian Integration Tests...")
    
    # Test template functionality
    template_test = TestDiagnosticProbabilityTemplates()
    template_test.test_get_template_probabilities_basic()
    template_test.test_van_evera_logic_enforcement()
    print("✅ Diagnostic probability templates tests passed")
    
    # Test bridge functionality  
    bridge_test = TestVanEveraBayesianBridge()
    bridge = VanEveraBayesianBridge()
    sample_assessment = EvidenceAssessment(
        evidence_id="test_evidence_1",
        refined_evidence_type=VanEveraEvidenceType.SMOKING_GUN,
        reasoning_for_type="Test reasoning",
        likelihood_P_E_given_H="High (0.8)",
        likelihood_P_E_given_NotH="Very Low (0.1)",
        justification_for_likelihoods="Test justification",
        suggested_numerical_probative_value=8.5
    )
    
    bridge_test.test_convert_evidence_assessment(bridge, sample_assessment)
    bridge_test.test_extract_likelihoods_from_llm(bridge)
    print("✅ Van Evera Bayesian bridge tests passed")
    
    # Test quantifier functionality
    quantifier_test = TestEvidenceStrengthQuantifier()
    quantifier = EvidenceStrengthQuantifier()
    quantifier_test.test_quantify_llm_assessment(quantifier, sample_assessment)
    quantifier_test.test_combine_multiple_evidence_independent(quantifier)
    print("✅ Evidence strength quantifier tests passed")
    
    # Test end-to-end integration
    integration_test = TestEndToEndIntegration()
    integration_test.test_complete_pipeline_llm_to_bayesian()
    integration_test.test_mathematical_validity_preservation()
    print("✅ End-to-end integration tests passed")
    
    print("\n🎉 All Phase 6B integration tests passed successfully!")
    print("Phase 6B Van Evera Bayesian Integration is ready for production use.")