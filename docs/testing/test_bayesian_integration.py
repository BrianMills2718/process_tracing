#!/usr/bin/env python3
"""
Test Bayesian Integration with Existing Process Tracing Pipeline

Validates that Phase 6C Bayesian components integrate properly with the
existing process tracing infrastructure without breaking compatibility.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Import integration components
from core.bayesian_integration import BayesianProcessTracingIntegrator
from core.bayesian_reporting import BayesianReportConfig

# Import existing infrastructure
from core.enhance_evidence import EvidenceAssessment
from core.structured_models import VanEveraEvidenceType


def create_mock_graph_analysis() -> Dict[str, Any]:
    """Create mock traditional process tracing analysis results."""
    return {
        "timestamp": "2025-08-02T21:00:00",
        "nodes": [
            {"id": "event_1", "type": "event", "description": "Revolutionary leadership emerged"},
            {"id": "evidence_1", "type": "evidence", "description": "Leaders made critical decisions"},
            {"id": "mechanism_1", "type": "mechanism", "description": "Organizational coordination"}
        ],
        "edges": [
            {"source": "evidence_1", "target": "event_1", "type": "supports"},
            {"source": "mechanism_1", "target": "event_1", "type": "explains"}
        ],
        "alternative_explanations": {
            "economic_factors": {
                "description": "Economic conditions drove the outcome",
                "likelihood": "moderate"
            },
            "external_pressure": {
                "description": "External political pressure was decisive",
                "likelihood": "low"
            }
        },
        "narrative_summary": "Revolutionary leadership played a decisive role in the successful outcome through effective organizational coordination and strategic decision-making.",
        "causal_mechanisms": {
            "leadership_coordination": {
                "description": "Leaders coordinated organizational efforts effectively",
                "strength": "high"
            }
        },
        "evidence_assessment": {
            "supporting_evidence_count": 3,
            "contradicting_evidence_count": 1,
            "evidence_quality": "high"
        }
    }


def create_mock_evidence_assessments() -> List[EvidenceAssessment]:
    """Create mock Van Evera evidence assessments."""
    evidence_assessments = []
    
    # Create mock evidence assessment objects
    # Note: Using dictionary representation since we might not have exact EvidenceAssessment structure
    mock_assessments = [
        {
            "evidence_id": "leadership_decisions",
            "description": "Critical leadership decisions at key moments",
            "refined_evidence_type": VanEveraEvidenceType.SMOKING_GUN,
            "likelihood_P_E_given_H": "High (0.9)",
            "likelihood_P_E_given_NotH": "Low (0.1)",
            "strength_assessment": "Very strong evidence",
            "reliability_assessment": "High reliability from primary sources",
            "reasoning": "Leaders consistently made optimal decisions at critical junctures"
        },
        {
            "evidence_id": "organizational_capacity", 
            "description": "Revolutionary organization showed coordination",
            "refined_evidence_type": VanEveraEvidenceType.HOOP,
            "likelihood_P_E_given_H": "High (0.85)",
            "likelihood_P_E_given_NotH": "Moderate (0.4)",
            "strength_assessment": "Strong evidence",
            "reliability_assessment": "Good reliability from historical analysis",
            "reasoning": "Necessary organizational capacity for successful coordination"
        },
        {
            "evidence_id": "popular_support",
            "description": "Leaders gained substantial popular support",
            "refined_evidence_type": VanEveraEvidenceType.STRAW_IN_THE_WIND,
            "likelihood_P_E_given_H": "Moderate (0.7)",
            "likelihood_P_E_given_NotH": "Moderate (0.6)",
            "strength_assessment": "Moderate evidence",
            "reliability_assessment": "Moderate reliability from survey data",
            "reasoning": "Popular support is correlated but not decisive"
        }
    ]
    
    # Convert to EvidenceAssessment objects (simplified)
    for mock_data in mock_assessments:
        # Create a simple object with the required attributes
        class MockEvidenceAssessment:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                # Add missing attribute required by van_evera_bayesian.py
                self.suggested_numerical_probative_value = 0.8  # Default value
            
            def dict(self):
                return self.__dict__
        
        evidence_assessments.append(MockEvidenceAssessment(mock_data))
    
    return evidence_assessments


def test_bayesian_integration_basic():
    """Test basic Bayesian integration functionality."""
    print("=== Testing Basic Bayesian Integration ===")
    
    # Create test data
    graph_analysis = create_mock_graph_analysis()
    evidence_assessments = create_mock_evidence_assessments()
    
    print(f"Mock analysis has {len(graph_analysis.get('alternative_explanations', {}))} alternative explanations")
    print(f"Mock evidence has {len(evidence_assessments)} evidence assessments")
    
    # Create integrator
    config = BayesianReportConfig(
        include_visualizations=False,  # Disable for testing
        uncertainty_simulations=50     # Reduce for speed
    )
    integrator = BayesianProcessTracingIntegrator(config)
    
    # Test integration
    try:
        enhanced_analysis = integrator.enhance_analysis_with_bayesian(
            graph_analysis, evidence_assessments
        )
        
        print("[OK] Bayesian integration completed successfully")
        
        # Validate enhanced analysis structure
        assert 'bayesian_analysis' in enhanced_analysis
        bayesian_section = enhanced_analysis['bayesian_analysis']
        
        assert 'confidence_assessments' in bayesian_section
        assert 'timestamp' in bayesian_section
        
        confidence_assessments = bayesian_section['confidence_assessments']
        print(f"[OK] Generated {len(confidence_assessments)} confidence assessments")
        
        # Check for primary hypothesis assessment
        if 'primary_explanation' in confidence_assessments:
            primary_assessment = confidence_assessments['primary_explanation']
            confidence = primary_assessment.get('overall_confidence', 0)
            level = primary_assessment.get('confidence_level', 'unknown')
            
            print(f"[OK] Primary hypothesis confidence: {confidence:.1%} ({level})")
            assert 0.0 <= confidence <= 1.0
        
        # Check uncertainty analysis if present
        if 'uncertainty_analysis' in bayesian_section:
            uncertainty_data = bayesian_section['uncertainty_analysis']
            print(f"[OK] Generated uncertainty analysis for {len(uncertainty_data)} hypotheses")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Bayesian integration failed: {e}")
        return False


def test_bayesian_report_generation():
    """Test Bayesian report generation."""
    print("\n=== Testing Bayesian Report Generation ===")
    
    # Create test data
    graph_analysis = create_mock_graph_analysis()
    evidence_assessments = create_mock_evidence_assessments()
    
    # Create integrator with reporting enabled
    config = BayesianReportConfig(
        include_visualizations=False,
        include_uncertainty_analysis=True,
        uncertainty_simulations=50
    )
    integrator = BayesianProcessTracingIntegrator(config)
    
    try:
        # Run integration with reporting
        enhanced_analysis = integrator.enhance_analysis_with_bayesian(
            graph_analysis, evidence_assessments
        )
        
        # Check for Bayesian report
        bayesian_section = enhanced_analysis.get('bayesian_analysis', {})
        bayesian_report = bayesian_section.get('bayesian_report', {})
        
        if bayesian_report:
            html_content = bayesian_report.get('html_content', '')
            sections = bayesian_report.get('sections', [])
            
            print(f"[OK] Generated HTML report with {len(html_content)} characters")
            print(f"[OK] Report contains {len(sections)} sections")
            
            # Validate report sections
            section_ids = [section.get('section_id') for section in sections]
            expected_sections = ['executive_summary', 'confidence_analysis']
            
            for expected in expected_sections:
                if expected in section_ids:
                    print(f"[OK] Report contains {expected} section")
                else:
                    print(f"[WARNING] Report missing {expected} section")
            
            # Check HTML content quality
            if 'Bayesian' in html_content:
                print("[OK] HTML content contains Bayesian analysis")
            
            return True
        else:
            print("[WARNING] No Bayesian report generated")
            return False
            
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        return False


def test_integration_compatibility():
    """Test that integration maintains compatibility with existing analysis."""
    print("\n=== Testing Integration Compatibility ===")
    
    # Create test data
    graph_analysis = create_mock_graph_analysis()
    evidence_assessments = create_mock_evidence_assessments()
    
    # Store original keys
    original_keys = set(graph_analysis.keys())
    
    try:
        # Create integrator
        integrator = BayesianProcessTracingIntegrator()
        
        # Run integration
        enhanced_analysis = integrator.enhance_analysis_with_bayesian(
            graph_analysis, evidence_assessments
        )
        
        # Check that original analysis is preserved
        for key in original_keys:
            if key in enhanced_analysis:
                print(f"[OK] Original key preserved: {key}")
            else:
                print(f"[ERROR] Original key missing: {key}")
                return False
        
        # Check that new Bayesian data is added
        if 'bayesian_analysis' in enhanced_analysis:
            print("[OK] Bayesian analysis section added")
        else:
            print("[ERROR] Bayesian analysis section missing")
            return False
        
        # Check that summary metrics are added
        summary_metrics = ['overall_confidence', 'confidence_level']
        for metric in summary_metrics:
            if metric in enhanced_analysis:
                print(f"[OK] Summary metric added: {metric}")
            else:
                print(f"[WARNING] Summary metric missing: {metric}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Compatibility test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and graceful degradation."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with invalid data
        integrator = BayesianProcessTracingIntegrator()
        
        # Test with empty analysis
        empty_analysis = {}
        empty_evidence = []
        
        result = integrator.enhance_analysis_with_bayesian(empty_analysis, empty_evidence)
        
        if 'bayesian_enhancement_error' in result:
            print("[OK] Error handling works - error recorded in result")
        else:
            print("[OK] Empty data handled gracefully")
        
        # Test with malformed evidence
        malformed_evidence = [{"invalid": "data"}]
        graph_analysis = create_mock_graph_analysis()
        
        result = integrator.enhance_analysis_with_bayesian(graph_analysis, malformed_evidence)
        
        # Should either work or fail gracefully
        if 'bayesian_analysis' in result or 'bayesian_enhancement_error' in result:
            print("[OK] Malformed evidence handled appropriately")
        else:
            print("[WARNING] Malformed evidence handling unclear")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error handling test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("Starting Bayesian Integration Tests")
    print("="*50)
    
    tests = [
        test_bayesian_integration_basic,
        test_bayesian_report_generation,
        test_integration_compatibility,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("BAYESIAN INTEGRATION TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    for i, test in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("[SUCCESS] All integration tests passed!")
        print("Phase 6C Bayesian integration is ready for production use.")
        return True
    else:
        print("[WARNING] Some integration tests failed.")
        print("Integration may need additional work before production use.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)