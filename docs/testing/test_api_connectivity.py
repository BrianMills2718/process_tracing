#!/usr/bin/env python3
"""
Test API Connectivity and Bayesian Integration

Quick test to verify API key, model configuration, and basic functionality.
"""

import os
import sys
from pathlib import Path

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_api_key():
    """Test that API key is available."""
    print("=== Testing API Key Configuration ===")
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"[OK] API key found: {api_key[:20]}...{api_key[-4:]}")
        return True
    else:
        print("[ERROR] No API key found")
        return False

def test_gemini_connectivity():
    """Test basic Gemini API connectivity."""
    print("\n=== Testing Gemini API Connectivity ===")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Test gemini-2.5-flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Simple test prompt
        response = model.generate_content("Say 'API test successful' and nothing else.")
        result = response.text.strip()
        
        print(f"[OK] Gemini 2.5 Flash response: {result}")
        return "successful" in result.lower()
        
    except Exception as e:
        print(f"[ERROR] Gemini API test failed: {e}")
        return False

def test_bayesian_imports():
    """Test that Bayesian components can be imported."""
    print("\n=== Testing Bayesian Component Imports ===")
    
    try:
        from core.bayesian_integration import integrate_bayesian_analysis
        from core.bayesian_reporting import BayesianReportConfig
        print("[OK] Bayesian integration components imported")
        
        from core.enhance_evidence import refine_evidence_assessment_with_llm
        print("[OK] Evidence enhancement components imported")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Bayesian import failed: {e}")
        return False

def test_configurable_integration():
    """Test the configurable integration from process_trace_advanced.py"""
    print("\n=== Testing Configurable Integration Functions ===")
    
    try:
        # Add project to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        import process_trace_advanced
        
        # Test function availability
        functions = ['create_bayesian_config_from_args', 'enhance_with_bayesian_analysis']
        for func_name in functions:
            if hasattr(process_trace_advanced, func_name):
                print(f"[OK] Function available: {func_name}")
            else:
                print(f"[ERROR] Function missing: {func_name}")
                return False
        
        # Test argument parsing
        import argparse
        
        # Create mock args for testing
        class MockArgs:
            def __init__(self):
                self.bayesian = True
                self.simulations = 100
                self.confidence_level = 0.95
                self.no_uncertainty = True
                self.no_visualizations = True
        
        args = MockArgs()
        config = process_trace_advanced.create_bayesian_config_from_args(args)
        
        if config:
            print(f"[OK] Bayesian config created: {config.uncertainty_simulations} simulations")
            return True
        else:
            print("[ERROR] Failed to create Bayesian config")
            return False
            
    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        return False

def main():
    """Run connectivity and integration tests."""
    print("API Connectivity and Bayesian Integration Test")
    print("=" * 60)
    
    tests = [
        ("API Key", test_api_key),
        ("Gemini Connectivity", test_gemini_connectivity),
        ("Bayesian Imports", test_bayesian_imports),
        ("Configurable Integration", test_configurable_integration)
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
    print("CONNECTIVITY TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n[SUCCESS] All connectivity tests passed!")
        print("The system is ready for end-to-end Bayesian analysis.")
        return True
    else:
        print(f"\n[WARNING] {total-passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)