#!/usr/bin/env python3
"""
End-to-End Integration Test for Phase 6C Bayesian Process Tracing

Tests the complete pipeline from traditional analysis through Bayesian enhancement
to final integrated reporting using real integration components.
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Import integration components
from core.bayesian_integration import integrate_bayesian_analysis
from core.llm_reporting_utils import (
    generate_enhanced_report_with_bayesian,
    check_bayesian_availability,
    get_bayesian_report_metadata
)
from core.bayesian_reporting import BayesianReportConfig

# Import process_trace_bayesian for full pipeline test
try:
    import process_trace_bayesian
    BAYESIAN_ENTRY_AVAILABLE = True
except ImportError:
    BAYESIAN_ENTRY_AVAILABLE = False


def create_test_analysis_data() -> Dict[str, Any]:
    """Create realistic test analysis data."""
    return {
        "timestamp": "2025-08-02T21:30:00",
        "nodes": [
            {"id": "event_1", "type": "event", "description": "Revolutionary leadership emerged with clear vision"},
            {"id": "evidence_1", "type": "evidence", "description": "Leaders made strategic decisions at critical moments"},
            {"id": "mechanism_1", "type": "mechanism", "description": "Organizational coordination enabled collective action"}
        ],
        "edges": [
            {"source": "evidence_1", "target": "event_1", "type": "supports"},
            {"source": "mechanism_1", "target": "event_1", "type": "explains"}
        ],
        "alternative_explanations": {
            "economic_pressure": {
                "description": "Economic conditions created revolutionary pressures",
                "likelihood": "moderate"
            },
            "external_influence": {
                "description": "Foreign powers influenced the revolutionary outcome", 
                "likelihood": "low"
            }
        },
        "narrative_summary": "Revolutionary leadership emerged as the decisive factor in successful collective action through effective organizational coordination and strategic decision-making at critical junctures.",
        "causal_mechanisms": {
            "leadership_coordination": {
                "description": "Leaders effectively coordinated organizational efforts and strategic planning",
                "strength": "high"
            }
        },
        "evidence_assessment": {
            "supporting_evidence_count": 4,
            "contradicting_evidence_count": 1,
            "evidence_quality": "high"
        }
    }


def create_test_evidence_assessments():
    """Create test Van Evera evidence assessments."""
    from core.structured_models import VanEveraEvidenceType
    
    # Create simple mock evidence objects
    evidence_assessments = []
    
    mock_evidence_data = [
        {
            "evidence_id": "strategic_leadership",
            "description": "Leaders demonstrated strategic thinking and coordination",
            "refined_evidence_type": VanEveraEvidenceType.SMOKING_GUN,
            "likelihood_P_E_given_H": "High (0.9)",
            "likelihood_P_E_given_NotH": "Low (0.1)", 
            "strength_assessment": "Very strong - definitive evidence",
            "reliability_assessment": "High reliability from primary sources",
            "reasoning": "Leadership decisions were consistently optimal and well-documented",
            "suggested_numerical_probative_value": 0.9,
            "justification_for_likelihoods": "High likelihood given hypothesis, low without it"
        },
        {
            "evidence_id": "organizational_capacity",
            "description": "Revolutionary organization showed clear coordination capacity",
            "refined_evidence_type": VanEveraEvidenceType.HOOP,
            "likelihood_P_E_given_H": "High (0.85)",
            "likelihood_P_E_given_NotH": "Moderate (0.4)",
            "strength_assessment": "Strong - necessary condition",
            "reliability_assessment": "Good reliability from historical analysis",
            "reasoning": "Organizational capacity was prerequisite for success",
            "suggested_numerical_probative_value": 0.8,
            "justification_for_likelihoods": "Necessary test - unlikely to succeed without capacity"
        }
    ]
    
    # Convert to mock objects with required attributes
    for data in mock_evidence_data:
        class MockEvidence:
            def __init__(self, data_dict):
                for key, value in data_dict.items():
                    setattr(self, key, value)
            
            def dict(self):
                return self.__dict__
        
        evidence_assessments.append(MockEvidence(data))
    
    return evidence_assessments


def test_bayesian_availability():
    """Test that Bayesian components are available."""
    print("=== Testing Bayesian Component Availability ===")
    
    available = check_bayesian_availability()
    print(f"Bayesian components available: {available}")
    
    if not available:
        print("[ERROR] Bayesian components not available - cannot test integration")
        return False
    
    print("[OK] Bayesian components are available")
    return True


def test_bayesian_integration():
    """Test the Bayesian integration pipeline."""
    print("\n=== Testing Bayesian Integration Pipeline ===")
    
    # Create test data
    analysis_data = create_test_analysis_data()
    evidence_assessments = create_test_evidence_assessments()
    
    print(f"Test analysis data: {len(analysis_data.get('alternative_explanations', {}))} alternatives")
    print(f"Test evidence: {len(evidence_assessments)} assessments")
    
    # Test integration
    try:
        # Create configuration for testing
        config = BayesianReportConfig(
            include_visualizations=False,  # Disable for testing speed
            uncertainty_simulations=100    # Reduce for testing speed
        )
        
        # Run Bayesian integration
        enhanced_analysis = integrate_bayesian_analysis(
            analysis_data, evidence_assessments, config=config
        )
        
        print("[OK] Bayesian integration completed")
        
        # Validate results
        if 'bayesian_analysis' in enhanced_analysis:
            bayesian_section = enhanced_analysis['bayesian_analysis']
            confidence_assessments = bayesian_section.get('confidence_assessments', {})
            
            print(f"[OK] Generated {len(confidence_assessments)} confidence assessments")
            
            # Check for primary hypothesis
            if 'primary_explanation' in confidence_assessments:
                primary = confidence_assessments['primary_explanation']
                confidence = primary.get('overall_confidence', 0)
                level = primary.get('confidence_level', 'unknown')
                print(f"[OK] Primary hypothesis: {confidence:.1%} confidence ({level})")
            
            return enhanced_analysis
        else:
            print("[ERROR] No Bayesian analysis section generated")
            return None
            
    except Exception as e:
        print(f"[ERROR] Bayesian integration failed: {e}")
        return None


def test_enhanced_reporting(enhanced_analysis):
    """Test enhanced HTML report generation."""
    print("\n=== Testing Enhanced HTML Report Generation ===")
    
    if not enhanced_analysis:
        print("[ERROR] No enhanced analysis data to test")
        return False
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate enhanced report
            result = generate_enhanced_report_with_bayesian(
                enhanced_analysis,
                output_dir=temp_dir,
                project_name="test_integration"
            )
            
            # Check if report was generated
            if 'enhanced_report_path' in result:
                report_path = result['enhanced_report_path']
                print(f"[OK] Enhanced report generated: {Path(report_path).name}")
                
                # Check report file exists and has content
                report_file = Path(report_path)
                if report_file.exists():
                    content_size = report_file.stat().st_size
                    print(f"[OK] Report file size: {content_size} bytes")
                    
                    # Read and validate HTML content
                    with open(report_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Basic HTML validation
                    if 'Enhanced Process Tracing Analysis' in html_content:
                        print("[OK] Report contains expected title")
                    if 'Bayesian Confidence Assessment' in html_content:
                        print("[OK] Report contains Bayesian sections")
                    if 'confidence-badge' in html_content:
                        print("[OK] Report contains confidence badges")
                    
                    return True
                else:
                    print("[ERROR] Report file not found")
                    return False
            else:
                print("[ERROR] No enhanced report path in result")
                return False
                
    except Exception as e:
        print(f"[ERROR] Enhanced reporting failed: {e}")
        return False


def test_metadata_extraction(enhanced_analysis):
    """Test Bayesian metadata extraction."""
    print("\n=== Testing Bayesian Metadata Extraction ===")
    
    if not enhanced_analysis:
        print("[ERROR] No enhanced analysis data to test")
        return False
    
    try:
        metadata = get_bayesian_report_metadata(enhanced_analysis)
        
        print(f"Bayesian available: {metadata.get('bayesian_available', False)}")
        print(f"Analysis present: {metadata.get('bayesian_analysis_present', False)}")
        print(f"Hypothesis count: {metadata.get('hypothesis_count', 0)}")
        
        if metadata.get('primary_confidence') is not None:
            confidence = metadata['primary_confidence']
            level = metadata.get('primary_confidence_level', 'unknown')
            evidence_count = metadata.get('primary_evidence_count', 0)
            print(f"Primary hypothesis: {confidence:.1%} ({level}) with {evidence_count} evidence")
        
        print("[OK] Metadata extraction successful")
        return True
        
    except Exception as e:
        print(f"[ERROR] Metadata extraction failed: {e}")
        return False


def test_bayesian_entry_point():
    """Test that the Bayesian entry point module loads correctly."""
    print("\n=== Testing Bayesian Entry Point ===")
    
    if not BAYESIAN_ENTRY_AVAILABLE:
        print("[ERROR] process_trace_bayesian module not available")
        return False
    
    try:
        # Test that key functions exist
        if hasattr(process_trace_bayesian, 'run_complete_bayesian_analysis'):
            print("[OK] run_complete_bayesian_analysis function available")
        else:
            print("[ERROR] Missing run_complete_bayesian_analysis function")
            return False
        
        if hasattr(process_trace_bayesian, 'main'):
            print("[OK] main entry point function available")
        else:
            print("[ERROR] Missing main entry point function")
            return False
        
        print("[OK] Bayesian entry point module loaded successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Bayesian entry point test failed: {e}")
        return False


def main():
    """Run complete end-to-end integration test."""
    print("Starting End-to-End Bayesian Integration Test")
    print("=" * 60)
    
    tests = [
        ("Bayesian Availability", test_bayesian_availability),
        ("Bayesian Integration", test_bayesian_integration),
        ("Enhanced Reporting", lambda: test_enhanced_reporting(enhanced_analysis)),
        ("Metadata Extraction", lambda: test_metadata_extraction(enhanced_analysis)),
        ("Entry Point", test_bayesian_entry_point)
    ]
    
    results = []
    enhanced_analysis = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Bayesian Integration":
                enhanced_analysis = test_func()
                result = enhanced_analysis is not None
            elif test_name in ["Enhanced Reporting", "Metadata Extraction"]:
                result = test_func()
            else:
                result = test_func()
            
            results.append(result)
            
        except Exception as e:
            print(f"[ERROR] Test {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("END-TO-END INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("[SUCCESS] All end-to-end integration tests passed!")
        print("Phase 6C Bayesian integration pipeline is fully functional.")
        return True
    else:
        print("[WARNING] Some integration tests failed.")
        print("Pipeline may need additional work before production use.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)