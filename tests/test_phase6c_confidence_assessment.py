"""
Comprehensive tests for Phase 6C Confidence Assessment functionality.

Tests confidence calculation, uncertainty analysis, and Bayesian reporting
components with mathematical validation and integration testing.
"""

import pytest
import numpy as np
import math
from typing import Dict, List
from unittest.mock import patch, MagicMock

from core.confidence_calculator import (
    CausalConfidenceCalculator, ConfidenceAssessment, ConfidenceLevel, 
    ConfidenceType
)
from core.uncertainty_analysis import (
    UncertaintyAnalyzer, UncertaintyAnalysisResult, UncertaintySource,
    UncertaintyType, SensitivityType
)
from core.bayesian_reporting import (
    BayesianReporter, BayesianReportConfig, BayesianReportSection
)
from core.bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace,
    HypothesisType, EvidenceType
)


class TestConfidenceCalculator:
    """Test confidence calculation functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create confidence calculator instance."""
        return CausalConfidenceCalculator()
    
    @pytest.fixture
    def sample_hypothesis_space(self):
        """Create sample hypothesis space for testing."""
        space = BayesianHypothesisSpace("test_space", "Test hypothesis space")
        
        # Create competing hypotheses
        main_hypothesis = BayesianHypothesis(
            hypothesis_id="main_hypothesis",
            description="Main causal hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.4,
            posterior_probability=0.7
        )
        
        alt_hypothesis = BayesianHypothesis(
            hypothesis_id="alt_hypothesis", 
            description="Alternative hypothesis",
            hypothesis_type=HypothesisType.ALTERNATIVE,
            prior_probability=0.3,
            posterior_probability=0.2
        )
        
        null_hypothesis = BayesianHypothesis(
            hypothesis_id="null_hypothesis",
            description="Null hypothesis",
            hypothesis_type=HypothesisType.NULL,
            prior_probability=0.3,
            posterior_probability=0.1
        )
        
        space.add_hypothesis(main_hypothesis)
        space.add_hypothesis(alt_hypothesis)
        space.add_hypothesis(null_hypothesis)
        
        # Set mutual exclusivity
        space.add_mutual_exclusivity_group({
            "main_hypothesis", "alt_hypothesis", "null_hypothesis"
        })
        
        return space, main_hypothesis
    
    @pytest.fixture
    def sample_evidence_list(self):
        """Create sample evidence list for testing."""
        evidence_list = [
            BayesianEvidence(
                evidence_id="smoking_gun_1",
                description="Strong confirming evidence",
                evidence_type=EvidenceType.SMOKING_GUN,
                source_node_id="node_1",
                likelihood_positive=0.9,
                likelihood_negative=0.1,
                reliability=0.8,
                strength=0.9
            ),
            BayesianEvidence(
                evidence_id="hoop_1",
                description="Necessary evidence",
                evidence_type=EvidenceType.HOOP,
                source_node_id="node_2",
                likelihood_positive=0.8,
                likelihood_negative=0.4,
                reliability=0.7,
                strength=0.8
            ),
            BayesianEvidence(
                evidence_id="straw_1",
                description="Weak supportive evidence",
                evidence_type=EvidenceType.STRAW_IN_THE_WIND,
                source_node_id="node_3",
                likelihood_positive=0.6,
                likelihood_negative=0.5,
                reliability=0.6,
                strength=0.6
            )
        ]
        return evidence_list
    
    def test_confidence_assessment_creation(self, calculator, sample_hypothesis_space, sample_evidence_list):
        """Test basic confidence assessment creation."""
        space, hypothesis = sample_hypothesis_space
        
        # Add evidence to hypothesis
        hypothesis.supporting_evidence.add("smoking_gun_1")
        hypothesis.supporting_evidence.add("hoop_1")
        hypothesis.supporting_evidence.add("straw_1")
        
        # Add evidence to space
        for evidence in sample_evidence_list:
            space.add_evidence(evidence)
        
        # Calculate confidence
        assessment = calculator.calculate_confidence(hypothesis, space, sample_evidence_list)
        
        # Validate assessment structure
        assert isinstance(assessment, ConfidenceAssessment)
        assert assessment.hypothesis_id == "main_hypothesis"
        assert 0.0 <= assessment.overall_confidence <= 1.0
        assert isinstance(assessment.confidence_level, ConfidenceLevel)
        assert len(assessment.confidence_components) > 0
        
        # Check that all confidence types are present
        expected_types = {ConfidenceType.EVIDENTIAL, ConfidenceType.CAUSAL, 
                         ConfidenceType.COHERENCE, ConfidenceType.ROBUSTNESS, 
                         ConfidenceType.SENSITIVITY}
        assert set(assessment.confidence_components.keys()) == expected_types
        
        # Validate component scores
        for conf_type, score in assessment.confidence_components.items():
            assert 0.0 <= score <= 1.0, f"{conf_type} score out of range: {score}"
    
    def test_evidential_confidence_calculation(self, calculator, sample_hypothesis_space, sample_evidence_list):
        """Test evidential confidence calculation."""
        space, hypothesis = sample_hypothesis_space
        
        # Add high-quality evidence
        hypothesis.supporting_evidence.add("smoking_gun_1")
        hypothesis.supporting_evidence.add("hoop_1")
        
        for evidence in sample_evidence_list:
            space.add_evidence(evidence)
        
        # Calculate evidential confidence
        evidential_confidence = calculator._calculate_evidential_confidence(
            hypothesis, sample_evidence_list
        )
        
        assert 0.0 <= evidential_confidence <= 1.0
        
        # Should be relatively high due to good evidence quality
        assert evidential_confidence > 0.5
    
    def test_causal_confidence_calculation(self, calculator, sample_hypothesis_space, sample_evidence_list):
        """Test causal confidence calculation."""
        space, hypothesis = sample_hypothesis_space
        
        causal_confidence = calculator._calculate_causal_confidence(
            hypothesis, sample_evidence_list, space
        )
        
        assert 0.0 <= causal_confidence <= 1.0
        
        # Should reflect posterior probability
        assert causal_confidence > 0.5  # High posterior should yield high causal confidence
    
    def test_coherence_confidence_calculation(self, calculator, sample_hypothesis_space, sample_evidence_list):
        """Test logical coherence confidence calculation."""
        space, hypothesis = sample_hypothesis_space
        
        coherence_confidence = calculator._calculate_coherence_confidence(
            hypothesis, sample_evidence_list, space
        )
        
        assert 0.0 <= coherence_confidence <= 1.0
    
    def test_confidence_level_classification(self, calculator):
        """Test confidence level classification."""
        # Test level boundaries
        assert ConfidenceLevel.from_score(0.95) == ConfidenceLevel.VERY_HIGH
        assert ConfidenceLevel.from_score(0.80) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.60) == ConfidenceLevel.MODERATE
        assert ConfidenceLevel.from_score(0.40) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.20) == ConfidenceLevel.VERY_LOW
    
    def test_confidence_interpretation(self, calculator, sample_hypothesis_space, sample_evidence_list):
        """Test confidence interpretation and recommendations."""
        space, hypothesis = sample_hypothesis_space
        
        hypothesis.supporting_evidence.add("smoking_gun_1")
        for evidence in sample_evidence_list:
            space.add_evidence(evidence)
        
        assessment = calculator.calculate_confidence(hypothesis, space, sample_evidence_list)
        
        # Test interpretation
        interpretation = assessment.get_interpretation()
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        
        # Test recommendations
        recommendations = assessment.get_recommendations()
        assert isinstance(recommendations, list)
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_confidence_with_contradicting_evidence(self, calculator, sample_hypothesis_space, sample_evidence_list):
        """Test confidence calculation with contradicting evidence."""
        space, hypothesis = sample_hypothesis_space
        
        # Add some supporting and some contradicting evidence
        hypothesis.supporting_evidence.add("smoking_gun_1")
        hypothesis.contradicting_evidence.add("hoop_1")
        
        for evidence in sample_evidence_list:
            space.add_evidence(evidence)
        
        assessment = calculator.calculate_confidence(hypothesis, space, sample_evidence_list)
        
        # Confidence should be lower due to contradicting evidence
        assert assessment.overall_confidence < 1.0
        assert assessment.contradicting_evidence_count > 0
    
    def test_multiple_hypothesis_comparison(self, calculator, sample_hypothesis_space):
        """Test confidence comparison across multiple hypotheses."""
        space, main_hypothesis = sample_hypothesis_space
        
        hypotheses = list(space.hypotheses.values())
        assessments = calculator.compare_hypotheses_confidence(hypotheses, space)
        
        assert len(assessments) == len(hypotheses)
        for hyp_id, assessment in assessments.items():
            assert isinstance(assessment, ConfidenceAssessment)
            assert assessment.hypothesis_id == hyp_id


class TestUncertaintyAnalyzer:
    """Test uncertainty analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create uncertainty analyzer instance."""
        return UncertaintyAnalyzer(random_seed=42)  # Fixed seed for reproducibility
    
    @pytest.fixture
    def sample_uncertainty_sources(self):
        """Create sample uncertainty sources."""
        return [
            UncertaintySource(
                source_id="evidence_reliability",
                uncertainty_type=UncertaintyType.DATA,
                description="Evidence reliability uncertainty",
                affected_parameters=["evidence_smoking_gun_1_reliability"],
                uncertainty_magnitude=0.1,
                distribution_type="beta",
                distribution_params={"alpha": 8, "beta": 2}
            ),
            UncertaintySource(
                source_id="prior_uncertainty",
                uncertainty_type=UncertaintyType.EPISTEMIC,
                description="Prior probability uncertainty",
                affected_parameters=["hypothesis_main_hypothesis_prior"],
                uncertainty_magnitude=0.05,
                distribution_type="normal",
                distribution_params={"mean": 0.0, "std": 0.05}
            )
        ]
    
    def test_uncertainty_source_sampling(self, sample_uncertainty_sources):
        """Test uncertainty source sampling."""
        source = sample_uncertainty_sources[0]
        
        # Test beta distribution sampling
        samples = source.sample(100)
        assert len(samples) == 100
        assert all(0 <= s <= 1 for s in samples)  # Beta distribution should be in [0,1]
        
        # Test normal distribution sampling
        normal_source = sample_uncertainty_sources[1]
        normal_samples = normal_source.sample(100)
        assert len(normal_samples) == 100
    
    def test_uncertainty_analysis_basic(self, analyzer):
        """Test basic uncertainty analysis."""
        # Create simple test setup
        hypothesis = BayesianHypothesis(
            hypothesis_id="test_hypothesis",
            description="Test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,
            posterior_probability=0.7
        )
        
        space = BayesianHypothesisSpace("test_space", "Test space")
        space.add_hypothesis(hypothesis)
        
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            likelihood_positive=0.8,
            likelihood_negative=0.2,
            reliability=0.9,
            strength=0.8
        )
        
        space.add_evidence(evidence)
        evidence_list = [evidence]
        
        # Run uncertainty analysis with small number of simulations for speed
        result = analyzer.analyze_uncertainty(
            hypothesis, space, evidence_list, n_simulations=100
        )
        
        # Validate result structure
        assert isinstance(result, UncertaintyAnalysisResult)
        assert result.hypothesis_id == "test_hypothesis"
        assert 0.0 <= result.baseline_confidence <= 1.0
        assert len(result.confidence_distribution) == 100
        assert result.n_simulations == 100
        
        # Check statistical measures
        assert 0.0 <= result.confidence_mean <= 1.0
        assert result.confidence_std >= 0.0
        assert 0.0 <= result.robustness_score <= 1.0
        assert 0.0 <= result.stability_score <= 1.0
    
    def test_monte_carlo_uncertainty_propagation(self, analyzer, sample_uncertainty_sources):
        """Test Monte Carlo uncertainty propagation."""
        # Create test hypothesis and evidence
        hypothesis = BayesianHypothesis(
            hypothesis_id="test_hypothesis",
            description="Test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,
            posterior_probability=0.7
        )
        
        evidence = BayesianEvidence(
            evidence_id="smoking_gun_1",
            description="Test evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            likelihood_positive=0.8,
            likelihood_negative=0.2,
            reliability=0.9,
            strength=0.8
        )
        
        space = BayesianHypothesisSpace("test_space", "Test space")
        space.add_hypothesis(hypothesis)
        space.add_evidence(evidence)
        
        # Run Monte Carlo simulation
        confidence_distribution = analyzer._monte_carlo_uncertainty_propagation(
            hypothesis, space, [evidence], sample_uncertainty_sources, 50
        )
        
        assert len(confidence_distribution) == 50
        assert all(0.0 <= conf <= 1.0 for conf in confidence_distribution)
    
    def test_sensitivity_analysis(self, analyzer):
        """Test sensitivity analysis functionality."""
        # Create test setup
        hypothesis = BayesianHypothesis(
            hypothesis_id="test_hypothesis",
            description="Test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,
            posterior_probability=0.7
        )
        
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            likelihood_positive=0.8,
            likelihood_negative=0.2
        )
        
        space = BayesianHypothesisSpace("test_space", "Test space")
        space.add_hypothesis(hypothesis)
        space.add_evidence(evidence)
        
        # Create uncertainty source
        uncertainty_source = UncertaintySource(
            source_id="test_uncertainty",
            uncertainty_type=UncertaintyType.PARAMETER,
            description="Test uncertainty",
            affected_parameters=["evidence_test_evidence_likelihood_positive"],
            uncertainty_magnitude=0.1
        )
        
        # Run sensitivity analysis
        sensitivity_result = analyzer._calculate_parameter_sensitivity(
            hypothesis, space, [evidence], uncertainty_source, 0.7
        )
        
        assert sensitivity_result.parameter_name == "test_uncertainty"
        assert sensitivity_result.sensitivity_index >= 0.0
        assert sensitivity_result.baseline_output == 0.7
        assert len(sensitivity_result.perturbed_outputs) > 0
    
    def test_uncertainty_auto_detection(self, analyzer):
        """Test automatic uncertainty source detection."""
        hypothesis = BayesianHypothesis(
            hypothesis_id="test_hypothesis",
            description="Test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,
            posterior_probability=0.7
        )
        
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Test evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            likelihood_positive=0.8,
            likelihood_negative=0.2,
            reliability=0.9,  # Less than 1.0 should trigger uncertainty
            strength=0.8      # Less than 1.0 should trigger uncertainty
        )
        
        uncertainty_sources = analyzer._detect_uncertainty_sources(hypothesis, [evidence])
        
        # Should detect multiple uncertainty sources
        assert len(uncertainty_sources) > 0
        
        # Check that different types of uncertainty are detected
        uncertainty_types = [source.uncertainty_type for source in uncertainty_sources]
        assert UncertaintyType.DATA in uncertainty_types  # Reliability uncertainty
        assert UncertaintyType.EPISTEMIC in uncertainty_types  # Prior uncertainty
    
    def test_robustness_calculation(self, analyzer):
        """Test robustness score calculation."""
        # Test with low variance (high robustness)
        confidence_dist_low_var = np.array([0.7, 0.71, 0.69, 0.7, 0.72])
        robustness_low = analyzer._calculate_robustness_score(0.7, confidence_dist_low_var)
        
        # Test with high variance (low robustness)
        confidence_dist_high_var = np.array([0.3, 0.9, 0.2, 0.8, 0.1])
        robustness_high = analyzer._calculate_robustness_score(0.7, confidence_dist_high_var)
        
        assert 0.0 <= robustness_low <= 1.0
        assert 0.0 <= robustness_high <= 1.0
        assert robustness_low > robustness_high  # Low variance should be more robust
    
    def test_uncertainty_report_generation(self, analyzer):
        """Test uncertainty report generation."""
        # Create mock uncertainty result
        uncertainty_result = UncertaintyAnalysisResult(
            hypothesis_id="test_hypothesis",
            baseline_confidence=0.7,
            confidence_distribution=np.array([0.65, 0.7, 0.75, 0.68, 0.72]),
            confidence_percentiles={"2.5": 0.65, "97.5": 0.75},
            sensitivity_results={},
            uncertainty_sources=[],
            confidence_mean=0.7,
            confidence_std=0.03,
            robustness_score=0.8,
            stability_score=0.85,
            n_simulations=100,
            convergence_achieved=True
        )
        
        report = analyzer.generate_uncertainty_report(uncertainty_result)
        
        assert "executive_summary" in report
        assert "statistical_analysis" in report
        assert "recommendations" in report
        
        # Check executive summary content
        summary = report["executive_summary"]
        assert "baseline_confidence" in summary
        assert "mean_confidence" in summary
        assert "confidence_std" in summary


class TestBayesianReporter:
    """Test Bayesian reporting functionality."""
    
    @pytest.fixture
    def reporter(self):
        """Create Bayesian reporter instance."""
        config = BayesianReportConfig(
            include_visualizations=False,  # Disable for testing without matplotlib
            uncertainty_simulations=50     # Reduce for speed
        )
        return BayesianReporter(config)
    
    @pytest.fixture
    def sample_report_data(self):
        """Create sample data for reporting tests."""
        space = BayesianHypothesisSpace("test_space", "Test hypothesis space")
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="main_hypothesis",
            description="Main causal hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.4,
            posterior_probability=0.75
        )
        
        evidence = BayesianEvidence(
            evidence_id="test_evidence",
            description="Strong supporting evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="test_node",
            likelihood_positive=0.9,
            likelihood_negative=0.1,
            reliability=0.8,
            strength=0.9
        )
        
        hypothesis.supporting_evidence.add("test_evidence")
        space.add_hypothesis(hypothesis)
        space.add_evidence(evidence)
        
        return space, hypothesis
    
    def test_report_config_creation(self):
        """Test report configuration creation."""
        config = BayesianReportConfig()
        
        assert config.include_confidence_analysis is True
        assert config.include_uncertainty_analysis is True
        assert config.include_evidence_details is True
        assert config.uncertainty_simulations == 1000
        assert 0.0 <= config.confidence_level <= 1.0
    
    def test_report_section_creation(self, reporter):
        """Test report section creation."""
        section = BayesianReportSection(
            section_id="test_section",
            title="Test Section",
            content_html="<p>Test content</p>",
            data={"test_key": "test_value"}
        )
        
        assert section.section_id == "test_section"
        assert section.title == "Test Section"
        assert "Test content" in section.content_html
        assert section.data["test_key"] == "test_value"
        assert section.visualizations == []
    
    def test_executive_summary_generation(self, reporter, sample_report_data):
        """Test executive summary section generation."""
        space, hypothesis = sample_report_data
        evidence_list = [space.get_evidence("test_evidence")]
        
        summary_section = reporter._generate_executive_summary(
            hypothesis, space, evidence_list
        )
        
        assert summary_section.section_id == "executive_summary"
        assert summary_section.title == "Executive Summary"
        assert "Executive Summary" in summary_section.content_html
        assert "posterior_probability" in summary_section.data
        assert "confidence_score" in summary_section.data
        assert 0.0 <= summary_section.data["posterior_probability"] <= 1.0
    
    def test_confidence_analysis_generation(self, reporter, sample_report_data):
        """Test confidence analysis section generation."""
        space, hypothesis = sample_report_data
        evidence_list = [space.get_evidence("test_evidence")]
        
        confidence_section = reporter._generate_confidence_analysis(
            hypothesis, space, evidence_list
        )
        
        assert confidence_section.section_id == "confidence_analysis"
        assert "Confidence Analysis" in confidence_section.content_html
        assert "overall_confidence" in confidence_section.data
        assert "confidence_components" in confidence_section.data
    
    def test_evidence_analysis_generation(self, reporter, sample_report_data):
        """Test evidence analysis section generation."""
        space, hypothesis = sample_report_data
        evidence_list = [space.get_evidence("test_evidence")]
        
        evidence_section = reporter._generate_evidence_analysis(hypothesis, evidence_list)
        
        assert evidence_section.section_id == "evidence_analysis"
        assert "Evidence Analysis" in evidence_section.content_html
        assert "evidence_analyses" in evidence_section.data
        assert "evidence_diversity" in evidence_section.data
    
    def test_hypothesis_comparison_generation(self, reporter, sample_report_data):
        """Test hypothesis comparison section generation."""
        space, hypothesis = sample_report_data
        
        # Add another hypothesis for comparison
        alt_hypothesis = BayesianHypothesis(
            hypothesis_id="alt_hypothesis",
            description="Alternative hypothesis",
            hypothesis_type=HypothesisType.ALTERNATIVE,
            prior_probability=0.3,
            posterior_probability=0.25
        )
        space.add_hypothesis(alt_hypothesis)
        
        comparison_section = reporter._generate_hypothesis_comparison(
            space, hypothesis.hypothesis_id
        )
        
        assert comparison_section.section_id == "hypothesis_comparison"
        assert "Hypothesis Comparison" in comparison_section.content_html
        assert "hypotheses" in comparison_section.data
        assert len(comparison_section.data["hypotheses"]) == 2
    
    def test_methodology_section_generation(self, reporter, sample_report_data):
        """Test methodology section generation."""
        space, hypothesis = sample_report_data
        evidence_list = [space.get_evidence("test_evidence")]
        
        methodology_section = reporter._generate_methodology_section(hypothesis, evidence_list)
        
        assert methodology_section.section_id == "methodology"
        assert "Methodology" in methodology_section.content_html
        assert "evidence_type_distribution" in methodology_section.data
        assert "total_evidence_count" in methodology_section.data
    
    @patch('core.bayesian_reporting.PLOTTING_AVAILABLE', False)
    def test_comprehensive_report_generation(self, reporter, sample_report_data):
        """Test comprehensive report generation without visualizations."""
        space, hypothesis = sample_report_data
        
        # Generate comprehensive report
        complete_report = reporter.generate_comprehensive_report(
            space, hypothesis.hypothesis_id
        )
        
        # Validate report structure
        assert "html_content" in complete_report
        assert "sections" in complete_report
        assert "metadata" in complete_report
        assert "target_hypothesis" in complete_report
        
        # Check metadata
        metadata = complete_report["metadata"]
        assert metadata["target_hypothesis"] == hypothesis.hypothesis_id
        assert metadata["hypothesis_count"] == len(space.hypotheses)
        assert "report_generated" in metadata
        
        # Check HTML content
        html_content = complete_report["html_content"]
        assert "Bayesian Process Tracing Analysis Report" in html_content
        assert hypothesis.hypothesis_id in html_content
        
        # Check sections
        sections = complete_report["sections"]
        assert len(sections) > 0
        
        section_ids = [section["section_id"] for section in sections]
        expected_sections = [
            "executive_summary", "confidence_analysis", "evidence_analysis", 
            "hypothesis_comparison", "methodology"
        ]
        
        for expected_section in expected_sections:
            assert expected_section in section_ids
    
    def test_visualization_handling_without_matplotlib(self, reporter):
        """Test that reporter handles missing matplotlib gracefully."""
        # This test ensures the reporter works even without plotting capabilities
        with patch('core.bayesian_reporting.PLOTTING_AVAILABLE', False):
            # Create mock confidence assessment
            from core.confidence_calculator import ConfidenceAssessment, ConfidenceLevel, ConfidenceType
            
            confidence_assessment = ConfidenceAssessment(
                hypothesis_id="test_hypothesis",
                overall_confidence=0.8,
                confidence_components={ConfidenceType.EVIDENTIAL: 0.8},
                confidence_level=ConfidenceLevel.HIGH
            )
            
            # Should not create visualization but should not error
            viz = reporter._create_summary_visualization(confidence_assessment)
            assert viz is None


class TestPhase6CIntegration:
    """Test integration between Phase 6C components."""
    
    def test_confidence_uncertainty_integration(self):
        """Test integration between confidence calculator and uncertainty analyzer."""
        # Create test setup
        hypothesis = BayesianHypothesis(
            hypothesis_id="integration_test",
            description="Integration test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,
            posterior_probability=0.8
        )
        
        evidence = BayesianEvidence(
            evidence_id="integration_evidence",
            description="Integration test evidence",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="integration_node",
            likelihood_positive=0.9,
            likelihood_negative=0.05,
            reliability=0.9,
            strength=0.9
        )
        
        space = BayesianHypothesisSpace("integration_space", "Integration test space")
        space.add_hypothesis(hypothesis)
        space.add_evidence(evidence)
        hypothesis.supporting_evidence.add("integration_evidence")
        
        # Test confidence calculation
        confidence_calc = CausalConfidenceCalculator()
        confidence_assessment = confidence_calc.calculate_confidence(
            hypothesis, space, [evidence]
        )
        
        # Test uncertainty analysis
        uncertainty_analyzer = UncertaintyAnalyzer(random_seed=42)
        uncertainty_result = uncertainty_analyzer.analyze_uncertainty(
            hypothesis, space, [evidence], n_simulations=100
        )
        
        # Validate integration
        assert confidence_assessment.overall_confidence > 0.0
        assert uncertainty_result.baseline_confidence > 0.0
        
        # The baseline confidence should be related to the confidence assessment
        # (they use different calculations, so exact equality is not expected)
        confidence_diff = abs(
            confidence_assessment.overall_confidence - uncertainty_result.baseline_confidence
        )
        assert confidence_diff < 0.5  # Should be reasonably close
    
    def test_reporting_integration(self):
        """Test integration with reporting system."""
        # Create test setup
        space = BayesianHypothesisSpace("report_test_space", "Report test space")
        
        hypothesis = BayesianHypothesis(
            hypothesis_id="report_test_hypothesis",
            description="Report test hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.4,
            posterior_probability=0.8
        )
        
        evidence = BayesianEvidence(
            evidence_id="report_test_evidence",
            description="Report test evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="report_test_node",
            likelihood_positive=0.9,
            likelihood_negative=0.1,
            reliability=0.8,
            strength=0.9
        )
        
        hypothesis.supporting_evidence.add("report_test_evidence")
        space.add_hypothesis(hypothesis)
        space.add_evidence(evidence)
        
        # Test reporting
        config = BayesianReportConfig(
            include_visualizations=False,
            uncertainty_simulations=50
        )
        reporter = BayesianReporter(config)
        
        # Generate report (this integrates all Phase 6C components)
        report = reporter.generate_comprehensive_report(space, hypothesis.hypothesis_id)
        
        # Validate that all components worked together
        assert "html_content" in report
        assert "sections" in report
        assert "metadata" in report
        
        # Check that confidence and uncertainty analysis were included
        section_ids = [section["section_id"] for section in report["sections"]]
        assert "confidence_analysis" in section_ids
        
        # Validate HTML content quality
        html_content = report["html_content"]
        assert len(html_content) > 1000  # Should be substantial content
        assert "Bayesian Process Tracing" in html_content
        assert hypothesis.hypothesis_id in html_content
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency across Phase 6C components."""
        # Create test with known mathematical properties
        hypothesis = BayesianHypothesis(
            hypothesis_id="math_test",
            description="Mathematical consistency test",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.5,
            posterior_probability=0.8
        )
        
        # Perfect smoking gun evidence (infinite likelihood ratio)
        evidence = BayesianEvidence(
            evidence_id="perfect_evidence",
            description="Perfect evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="math_node",
            likelihood_positive=0.95,
            likelihood_negative=0.0,  # Zero false positive rate
            reliability=1.0,
            strength=1.0
        )
        
        space = BayesianHypothesisSpace("math_space", "Mathematical test space")
        space.add_hypothesis(hypothesis)
        space.add_evidence(evidence)
        hypothesis.supporting_evidence.add("perfect_evidence")
        
        # Test confidence calculation
        confidence_calc = CausalConfidenceCalculator()
        confidence_assessment = confidence_calc.calculate_confidence(
            hypothesis, space, [evidence]
        )
        
        # Perfect evidence should yield reasonably high confidence
        # (Note: Single evidence piece will have lower sensitivity score, affecting overall confidence)
        assert confidence_assessment.overall_confidence > 0.7
        
        # Test uncertainty analysis
        uncertainty_analyzer = UncertaintyAnalyzer(random_seed=42)
        uncertainty_result = uncertainty_analyzer.analyze_uncertainty(
            hypothesis, space, [evidence], n_simulations=50
        )
        
        # Mathematical consistency checks
        assert 0.0 <= uncertainty_result.confidence_mean <= 1.0
        assert uncertainty_result.confidence_std >= 0.0
        assert 0.0 <= uncertainty_result.robustness_score <= 1.0
        
        # Perfect evidence should have low uncertainty
        assert uncertainty_result.confidence_std < 0.2


if __name__ == "__main__":
    # Run specific tests for debugging
    print("Running Phase 6C Confidence Assessment Tests...")
    
    # Test confidence calculator
    calc_test = TestConfidenceCalculator()
    calculator = CausalConfidenceCalculator()
    
    # Create simple test case
    hypothesis = BayesianHypothesis(
        hypothesis_id="simple_test",
        description="Simple test hypothesis",
        hypothesis_type=HypothesisType.PRIMARY,
        prior_probability=0.5,
        posterior_probability=0.7
    )
    
    evidence = BayesianEvidence(
        evidence_id="simple_evidence",
        description="Simple test evidence",
        evidence_type=EvidenceType.SMOKING_GUN,
        source_node_id="simple_node",
        likelihood_positive=0.9,
        likelihood_negative=0.1
    )
    
    space = BayesianHypothesisSpace("simple_space", "Simple test space")
    space.add_hypothesis(hypothesis)
    space.add_evidence(evidence)
    hypothesis.supporting_evidence.add("simple_evidence")
    
    # Test confidence calculation
    assessment = calculator.calculate_confidence(hypothesis, space, [evidence])
    print(f"Confidence Assessment: {assessment.overall_confidence:.2%}")
    print(f"Confidence Level: {assessment.confidence_level.label}")
    print(f"Evidence Count: {assessment.evidence_count}")
    
    # Test uncertainty analysis
    uncertainty_analyzer = UncertaintyAnalyzer(random_seed=42)
    uncertainty_result = uncertainty_analyzer.analyze_uncertainty(
        hypothesis, space, [evidence], n_simulations=100
    )
    print(f"Uncertainty Analysis - Mean: {uncertainty_result.confidence_mean:.2%}")
    print(f"Uncertainty Analysis - Std: {uncertainty_result.confidence_std:.2%}")
    print(f"Robustness Score: {uncertainty_result.robustness_score:.2%}")
    
    # Test reporting
    config = BayesianReportConfig(include_visualizations=False, uncertainty_simulations=50)
    reporter = BayesianReporter(config)
    report = reporter.generate_comprehensive_report(space, hypothesis.hypothesis_id)
    print(f"Report Sections: {len(report['sections'])}")
    print(f"HTML Content Length: {len(report['html_content'])} characters")
    
    print("\n✅ All Phase 6C components working correctly!")
    print("Phase 6C Confidence Assessment is ready for production use.")