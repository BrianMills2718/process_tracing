#!/usr/bin/env python3
"""
End-to-End Validation Test for Phase 6C: Confidence Assessment

Demonstrates the complete Phase 6C pipeline:
1. Multi-dimensional confidence assessment
2. Uncertainty analysis with Monte Carlo simulation
3. Comprehensive Bayesian reporting with HTML generation
4. Integration with existing infrastructure
"""

import sys
import numpy as np
from datetime import datetime

# Import Phase 6C components
from core.confidence_calculator import (
    CausalConfidenceCalculator, ConfidenceAssessment, ConfidenceLevel, ConfidenceType
)
from core.uncertainty_analysis import (
    UncertaintyAnalyzer, UncertaintyAnalysisResult, UncertaintySource, UncertaintyType
)
from core.bayesian_reporting import (
    BayesianReporter, BayesianReportConfig
)

# Import existing infrastructure
from core.bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace,
    HypothesisType, EvidenceType
)

def create_test_scenario():
    """Create a comprehensive test scenario for Phase 6C validation."""
    print("=== Creating Test Scenario ===")
    
    # Create hypothesis space
    space = BayesianHypothesisSpace("phase6c_test", "Phase 6C Validation Test")
    
    # Create competing hypotheses
    main_hypothesis = BayesianHypothesis(
        hypothesis_id="revolutionary_leadership",
        description="Leadership played decisive role in revolutionary success",
        hypothesis_type=HypothesisType.PRIMARY,
        prior_probability=0.4,
        posterior_probability=0.75
    )
    
    alt_hypothesis = BayesianHypothesis(
        hypothesis_id="economic_factors",
        description="Economic conditions were primary driver",
        hypothesis_type=HypothesisType.ALTERNATIVE,
        prior_probability=0.35,
        posterior_probability=0.2
    )
    
    null_hypothesis = BayesianHypothesis(
        hypothesis_id="random_events",
        description="Random events and contingency explain outcome",
        hypothesis_type=HypothesisType.NULL,
        prior_probability=0.25,
        posterior_probability=0.05
    )
    
    # Add hypotheses to space
    space.add_hypothesis(main_hypothesis)
    space.add_hypothesis(alt_hypothesis)
    space.add_hypothesis(null_hypothesis)
    
    # Create diverse evidence supporting main hypothesis
    evidence_list = [
        BayesianEvidence(
            evidence_id="leader_decision_timing",
            description="Leader made critical decisions at key moments",
            evidence_type=EvidenceType.SMOKING_GUN,
            source_node_id="leadership_node",
            likelihood_positive=0.9,
            likelihood_negative=0.1,
            reliability=0.8,
            strength=0.9,
            source_credibility=0.85,
            collection_method="primary_sources"
        ),
        BayesianEvidence(
            evidence_id="organizational_capacity",
            description="Revolutionary organization showed exceptional coordination",
            evidence_type=EvidenceType.HOOP,
            source_node_id="organization_node", 
            likelihood_positive=0.85,
            likelihood_negative=0.4,
            reliability=0.75,
            strength=0.8,
            source_credibility=0.8,
            collection_method="historical_analysis"
        ),
        BayesianEvidence(
            evidence_id="communication_network",
            description="Effective communication network enabled coordination",
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            source_node_id="communication_node",
            likelihood_positive=0.9,
            likelihood_negative=0.05,
            reliability=0.9,
            strength=0.85,
            source_credibility=0.9,
            collection_method="archival_research"
        ),
        BayesianEvidence(
            evidence_id="popular_support",
            description="Leadership gained substantial popular support",
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            source_node_id="support_node",
            likelihood_positive=0.7,
            likelihood_negative=0.6,
            reliability=0.7,
            strength=0.7,
            source_credibility=0.75,
            collection_method="survey_data"
        )
    ]
    
    # Add evidence to space and link to hypothesis
    for evidence in evidence_list:
        space.add_evidence(evidence)
        main_hypothesis.supporting_evidence.add(evidence.evidence_id)
    
    print(f"[OK] Created hypothesis space with {len(space.hypotheses)} hypotheses")
    print(f"[OK] Created {len(evidence_list)} pieces of evidence")
    print(f"[OK] Main hypothesis posterior: {main_hypothesis.posterior_probability:.1%}")
    
    return space, main_hypothesis, evidence_list


def test_confidence_calculation(hypothesis, space, evidence_list):
    """Test Phase 6C confidence calculation functionality."""
    print("\n=== Testing Confidence Calculation ===")
    
    # Initialize confidence calculator
    confidence_calc = CausalConfidenceCalculator()
    
    # Calculate confidence assessment
    assessment = confidence_calc.calculate_confidence(hypothesis, space, evidence_list)
    
    print(f"Overall Confidence: {assessment.overall_confidence:.1%}")
    print(f"Confidence Level: {assessment.confidence_level.label.replace('_', ' ').title()}")
    print(f"Evidence Count: {assessment.evidence_count}")
    print(f"Evidence Quality: {assessment.evidence_quality_score:.1%}")
    
    print("\nConfidence Components:")
    for conf_type, score in assessment.confidence_components.items():
        print(f"  {conf_type.value.replace('_', ' ').title()}: {score:.1%}")
    
    print(f"\nConfidence Interval: {assessment.confidence_interval[0]:.1%} - {assessment.confidence_interval[1]:.1%}")
    
    # Validate confidence assessment
    assert isinstance(assessment, ConfidenceAssessment)
    assert 0.0 <= assessment.overall_confidence <= 1.0
    assert len(assessment.confidence_components) == 5
    assert assessment.evidence_count == len(evidence_list)
    
    print("[OK] Confidence calculation completed successfully")
    return assessment


def test_uncertainty_analysis(hypothesis, space, evidence_list):
    """Test Phase 6C uncertainty analysis functionality."""
    print("\n=== Testing Uncertainty Analysis ===")
    
    # Initialize uncertainty analyzer
    uncertainty_analyzer = UncertaintyAnalyzer(random_seed=42)
    
    # Run uncertainty analysis
    uncertainty_result = uncertainty_analyzer.analyze_uncertainty(
        hypothesis, space, evidence_list, n_simulations=200  # Reduced for speed
    )
    
    print(f"Baseline Confidence: {uncertainty_result.baseline_confidence:.1%}")
    print(f"Mean Confidence: {uncertainty_result.confidence_mean:.1%}")
    print(f"Confidence Std Dev: {uncertainty_result.confidence_std:.1%}")
    print(f"Robustness Score: {uncertainty_result.robustness_score:.1%}")
    print(f"Stability Score: {uncertainty_result.stability_score:.1%}")
    
    print(f"\n95% Confidence Interval:")
    ci_lower = uncertainty_result.confidence_percentiles.get("2.5", 0)
    ci_upper = uncertainty_result.confidence_percentiles.get("97.5", 1)
    print(f"  {ci_lower:.1%} - {ci_upper:.1%}")
    
    print(f"\nTop Sensitive Parameters:")
    sensitive_params = uncertainty_result._get_most_sensitive_parameters()[:3]
    for param, sensitivity in sensitive_params:
        print(f"  {param}: {sensitivity:.1%} sensitivity")
    
    # Validate uncertainty analysis
    assert isinstance(uncertainty_result, UncertaintyAnalysisResult)
    assert 0.0 <= uncertainty_result.baseline_confidence <= 1.0
    assert uncertainty_result.confidence_std >= 0.0
    assert len(uncertainty_result.confidence_distribution) == 200
    assert len(uncertainty_result.uncertainty_sources) > 0
    
    print("[OK] Uncertainty analysis completed successfully")
    return uncertainty_result


def test_bayesian_reporting(space, hypothesis):
    """Test Phase 6C Bayesian reporting functionality."""
    print("\n=== Testing Bayesian Reporting ===")
    
    # Initialize reporter with minimal config for testing
    config = BayesianReportConfig(
        include_visualizations=False,  # Disable for testing
        uncertainty_simulations=50     # Reduce for speed
    )
    reporter = BayesianReporter(config)
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(space, hypothesis.hypothesis_id)
    
    print(f"Report Sections: {len(report['sections'])}")
    print(f"HTML Content Length: {len(report['html_content'])} characters")
    print(f"Target Hypothesis: {report['target_hypothesis']}")
    
    # List report sections
    print("\nReport Sections Generated:")
    for section in report['sections']:
        print(f"  - {section['title']} ({section['section_id']})")
    
    # Validate report structure
    assert "html_content" in report
    assert "sections" in report
    assert "metadata" in report
    assert len(report["sections"]) >= 5  # Should have multiple sections
    assert "Bayesian Process Tracing" in report["html_content"]
    
    # Check that key sections are present
    section_ids = [section["section_id"] for section in report["sections"]]
    expected_sections = ["executive_summary", "confidence_analysis", "evidence_analysis"]
    for expected in expected_sections:
        assert expected in section_ids, f"Missing section: {expected}"
    
    print("[OK] Bayesian reporting completed successfully")
    return report


def test_integration_consistency(confidence_assessment, uncertainty_result, report):
    """Test consistency across Phase 6C components."""
    print("\n=== Testing Integration Consistency ===")
    
    # Check that baseline confidence values are reasonably consistent
    confidence_diff = abs(confidence_assessment.overall_confidence - uncertainty_result.baseline_confidence)
    print(f"Confidence Assessment: {confidence_assessment.overall_confidence:.1%}")
    print(f"Uncertainty Baseline: {uncertainty_result.baseline_confidence:.1%}")
    print(f"Difference: {confidence_diff:.1%}")
    
    # Values should be reasonably close (different calculation methods expected)
    assert confidence_diff < 0.3, f"Confidence values too different: {confidence_diff:.1%}"
    
    # Check that report contains confidence data
    assert "confidence_analysis" in [s["section_id"] for s in report["sections"]]
    
    # Check report metadata consistency
    metadata = report["metadata"]
    assert metadata["target_hypothesis"] == confidence_assessment.hypothesis_id
    
    print("[OK] Integration consistency validated")


def test_mathematical_properties(confidence_assessment, uncertainty_result):
    """Test mathematical properties of Phase 6C results."""
    print("\n=== Testing Mathematical Properties ===")
    
    # Confidence bounds
    assert 0.0 <= confidence_assessment.overall_confidence <= 1.0
    for conf_type, score in confidence_assessment.confidence_components.items():
        assert 0.0 <= score <= 1.0, f"{conf_type} score out of bounds: {score}"
    
    # Uncertainty distribution properties
    assert uncertainty_result.confidence_std >= 0.0
    assert np.all(uncertainty_result.confidence_distribution >= 0.0)
    assert np.all(uncertainty_result.confidence_distribution <= 1.0)
    
    # Percentiles should be ordered
    percentiles = uncertainty_result.confidence_percentiles
    assert percentiles["2.5"] <= percentiles["50"] <= percentiles["97.5"]
    
    # Robustness and stability scores
    assert 0.0 <= uncertainty_result.robustness_score <= 1.0
    assert 0.0 <= uncertainty_result.stability_score <= 1.0
    
    print("[OK] Mathematical properties validated")


def main():
    """Run complete Phase 6C end-to-end validation."""
    print("Starting Phase 6C: Confidence Assessment End-to-End Validation")
    print("=" * 70)
    
    try:
        # Create test scenario
        space, hypothesis, evidence_list = create_test_scenario()
        
        # Test Phase 6C components
        confidence_assessment = test_confidence_calculation(hypothesis, space, evidence_list)
        uncertainty_result = test_uncertainty_analysis(hypothesis, space, evidence_list)
        report = test_bayesian_reporting(space, hypothesis)
        
        # Test integration and consistency
        test_integration_consistency(confidence_assessment, uncertainty_result, report)
        test_mathematical_properties(confidence_assessment, uncertainty_result)
        
        # Summary
        print("\n" + "=" * 70)
        print("PHASE 6C END-TO-END VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\nKey Results:")
        print(f"  Overall Confidence: {confidence_assessment.overall_confidence:.1%}")
        print(f"  Confidence Level: {confidence_assessment.confidence_level.label.replace('_', ' ').title()}")
        print(f"  Uncertainty (Std): {uncertainty_result.confidence_std:.1%}")
        print(f"  Robustness Score: {uncertainty_result.robustness_score:.1%}")
        print(f"  Report Sections: {len(report['sections'])}")
        print(f"  HTML Content: {len(report['html_content']):,} characters")
        
        print("\nPhase 6C Status: COMPLETE")
        print("  [OK] Multi-dimensional confidence assessment")
        print("  [OK] Monte Carlo uncertainty analysis")
        print("  [OK] Comprehensive Bayesian reporting")
        print("  [OK] HTML dashboard integration")
        print("  [OK] Mathematical validation")
        print("  [OK] Integration testing")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        print(f"Phase 6C validation failed: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)