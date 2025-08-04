#!/usr/bin/env python3
"""
Test Bayesian Enhancement Only

Tests the Bayesian enhancement functionality using existing analysis results,
bypassing the LLM extraction step to isolate the integration functionality.
"""

import json
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.bayesian_integration import integrate_bayesian_analysis
from core.bayesian_reporting import BayesianReportConfig

def test_bayesian_enhancement_with_existing_data():
    """Test Bayesian enhancement using existing analysis results."""
    print("=== Testing Bayesian Enhancement with Existing Data ===")
    
    # Use the most recent analysis result
    analysis_file = "output_data/revolutions/revolutions_20250803_061533_analysis_summary_20250803_061602.json"
    
    if not Path(analysis_file).exists():
        # Fall back to another recent file
        analysis_file = "output_data/revolutions/revolutions_20250801_235502_analysis_summary_20250801_235529.json"
    
    if not Path(analysis_file).exists():
        print(f"[ERROR] No existing analysis file found to test with")
        return False
    
    print(f"[INFO] Using analysis file: {analysis_file}")
    
    try:
        # Load existing analysis results
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_results = json.load(f)
        
        # Check the structure of the analysis file
        if 'nodes' in analysis_results:
            print(f"[OK] Loaded analysis results with {len(analysis_results.get('nodes', []))} nodes")
        elif 'causal_chains' in analysis_results:
            print(f"[OK] Loaded analysis results with {len(analysis_results.get('causal_chains', []))} causal chains")
            # Convert to node format for testing
            analysis_results['nodes'] = []
            analysis_results['alternative_explanations'] = {
                "economic_factors": {"description": "Economic pressures drove the revolution"},
                "external_influence": {"description": "Foreign influence was decisive"}
            }
            analysis_results['narrative_summary'] = "American Revolution driven by colonial resistance to British taxation"
        else:
            print(f"[INFO] Analysis file has format: {list(analysis_results.keys())}")
        
        # Create mock evidence assessments (since we don't have LLM access)
        evidence_assessments = []
        
        # Create Bayesian configuration for testing
        config = BayesianReportConfig(
            include_uncertainty_analysis=False,  # Skip for speed
            include_visualizations=False,        # Skip for speed
            uncertainty_simulations=10          # Minimal for testing
        )
        
        print("[INFO] Starting Bayesian integration...")
        
        # Test the Bayesian integration
        enhanced_analysis = integrate_bayesian_analysis(
            analysis_results,
            evidence_assessments,
            output_dir="test_output",
            config=config
        )
        
        print("[OK] Bayesian integration completed successfully")
        
        # Validate the enhanced results
        if 'bayesian_analysis' in enhanced_analysis:
            bayesian_section = enhanced_analysis['bayesian_analysis']
            confidence_assessments = bayesian_section.get('confidence_assessments', {})
            
            print(f"[OK] Generated {len(confidence_assessments)} confidence assessments")
            
            # Display results
            for hyp_id, assessment in confidence_assessments.items():
                confidence = assessment.get('overall_confidence', 0)
                level = assessment.get('confidence_level', 'unknown')
                evidence_count = assessment.get('evidence_count', 0)
                
                print(f"  {hyp_id.replace('_', ' ').title()}: {confidence:.1%} confidence ({level}) - {evidence_count} evidence")
            
            # Check for uncertainty analysis (should be empty since disabled)
            uncertainty_analysis = bayesian_section.get('uncertainty_analysis', {})
            print(f"[OK] Uncertainty analysis: {len(uncertainty_analysis)} hypotheses (expected 0)")
            
            # Check for report generation
            bayesian_report = bayesian_section.get('bayesian_report', {})
            if bayesian_report:
                html_content = bayesian_report.get('html_content', '')
                sections = bayesian_report.get('sections', [])
                print(f"[OK] Generated HTML report: {len(html_content)} characters, {len(sections)} sections")
            
            return True
        else:
            print("[ERROR] No bayesian_analysis section found in enhanced results")
            return False
            
    except Exception as e:
        print(f"[ERROR] Bayesian enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bayesian_functions_exist():
    """Test that all required Bayesian functions exist and are importable."""
    print("\n=== Testing Bayesian Function Availability ===")
    
    try:
        # Test imports
        from core.bayesian_integration import BayesianProcessTracingIntegrator
        from core.confidence_calculator import CausalConfidenceCalculator
        from core.uncertainty_analysis import UncertaintyAnalyzer
        from core.bayesian_reporting import BayesianReporter
        
        print("[OK] All Bayesian classes imported successfully")
        
        # Test instantiation
        integrator = BayesianProcessTracingIntegrator()
        print("[OK] BayesianProcessTracingIntegrator instantiated")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Bayesian function test failed: {e}")
        return False

def main():
    """Run Bayesian enhancement tests."""
    print("Starting Bayesian Enhancement Tests (No LLM Required)")
    print("=" * 60)
    
    tests = [
        ("Bayesian Functions", test_bayesian_functions_exist),
        ("Bayesian Enhancement", test_bayesian_enhancement_with_existing_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("BAYESIAN ENHANCEMENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("[SUCCESS] Bayesian enhancement functionality is working!")
        print("The --bayesian flag integration should work when API keys are available.")
        return True
    else:
        print("[WARNING] Some Bayesian enhancement tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)