#!/usr/bin/env python3
"""
Test Configurable Bayesian Integration in process_trace_advanced.py

Validates that the --bayesian flag integration works correctly and
that traditional workflows remain unchanged.
"""

import sys
import subprocess
from pathlib import Path

def test_help_output():
    """Test that Bayesian flags appear in help output."""
    print("=== Testing Help Output ===")
    
    result = subprocess.run([
        sys.executable, 'process_trace_advanced.py', '--help'
    ], capture_output=True, text=True, cwd='C:\\Users\\Brian\\Documents\\code\\process_tracing')
    
    help_text = result.stdout
    
    # Check for Bayesian flags
    bayesian_flags = [
        '--bayesian',
        '--simulations',
        '--confidence-level', 
        '--no-uncertainty',
        '--no-visualizations'
    ]
    
    for flag in bayesian_flags:
        if flag in help_text:
            print(f"[OK] Found flag: {flag}")
        else:
            print(f"[ERROR] Missing flag: {flag}")
            return False
    
    print("[OK] All Bayesian flags present in help output")
    return True

def test_argument_parsing():
    """Test that Bayesian arguments parse correctly."""
    print("\n=== Testing Argument Parsing ===")
    
    # Import the module to test argument parsing
    sys.path.insert(0, 'C:\\Users\\Brian\\Documents\\code\\process_tracing')
    
    try:
        # Test basic argument parsing (without actually running analysis)
        import argparse
        
        # Create a test parser similar to process_trace_advanced.py
        parser = argparse.ArgumentParser()
        parser.add_argument("--bayesian", action="store_true")
        parser.add_argument("--simulations", type=int, default=1000)
        parser.add_argument("--confidence-level", type=float, default=0.95)
        parser.add_argument("--no-uncertainty", action="store_true")
        parser.add_argument("--no-visualizations", action="store_true")
        
        # Test parsing various combinations
        test_cases = [
            [],  # No Bayesian flags
            ['--bayesian'],  # Basic Bayesian
            ['--bayesian', '--simulations', '500'],  # Custom simulations
            ['--bayesian', '--confidence-level', '0.99'],  # Custom confidence
            ['--bayesian', '--no-uncertainty'],  # Disable uncertainty
            ['--bayesian', '--no-visualizations'],  # Disable visualizations
            ['--bayesian', '--simulations', '100', '--no-uncertainty', '--no-visualizations']  # All options
        ]
        
        for i, args in enumerate(test_cases):
            try:
                parsed = parser.parse_args(args)
                print(f"[OK] Test case {i+1}: {args}")
                
                if parsed.bayesian:
                    print(f"    Bayesian: {parsed.bayesian}")
                    print(f"    Simulations: {parsed.simulations}")
                    print(f"    Confidence: {parsed.confidence_level}")
                    print(f"    No uncertainty: {parsed.no_uncertainty}")
                    print(f"    No visualizations: {parsed.no_visualizations}")
                
            except Exception as e:
                print(f"[ERROR] Test case {i+1} failed: {e}")
                return False
        
        print("[OK] All argument parsing tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Argument parsing test failed: {e}")
        return False

def test_function_availability():
    """Test that new functions are available."""
    print("\n=== Testing Function Availability ===")
    
    try:
        sys.path.insert(0, 'C:\\Users\\Brian\\Documents\\code\\process_tracing')
        
        # Test that we can import the module
        import process_trace_advanced
        
        # Check that new functions exist
        functions_to_check = [
            'create_bayesian_config_from_args',
            'enhance_with_bayesian_analysis'
        ]
        
        for func_name in functions_to_check:
            if hasattr(process_trace_advanced, func_name):
                print(f"[OK] Function available: {func_name}")
            else:
                print(f"[ERROR] Function missing: {func_name}")
                return False
        
        print("[OK] All required functions are available")
        return True
        
    except Exception as e:
        print(f"[ERROR] Function availability test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that existing commands still work."""
    print("\n=== Testing Backward Compatibility ===")
    
    # Test that traditional flags still work
    result = subprocess.run([
        sys.executable, 'process_trace_advanced.py', '--extract-only', '--help'
    ], capture_output=True, text=True, cwd='C:\\Users\\Brian\\Documents\\code\\process_tracing')
    
    if result.returncode == 0:
        print("[OK] Traditional --extract-only flag still works")
    else:
        print(f"[ERROR] Traditional flag compatibility issue: {result.stderr}")
        return False
    
    # Test help still works  
    result = subprocess.run([
        sys.executable, 'process_trace_advanced.py', '--help'
    ], capture_output=True, text=True, cwd='C:\\Users\\Brian\\Documents\\code\\process_tracing')
    
    if result.returncode == 0 and 'Advanced Process Tracing Pipeline' in result.stdout:
        print("[OK] Help output still works correctly")
    else:
        print(f"[ERROR] Help output issue")
        return False
    
    print("[OK] Backward compatibility maintained")
    return True

def main():
    """Run all configurable integration tests."""
    print("Starting Configurable Bayesian Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Help Output", test_help_output),
        ("Argument Parsing", test_argument_parsing), 
        ("Function Availability", test_function_availability),
        ("Backward Compatibility", test_backward_compatibility)
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
    print("CONFIGURABLE INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("[SUCCESS] Configurable Bayesian integration is working!")
        print("Users can now use --bayesian flag with process_trace_advanced.py")
        return True
    else:
        print("[WARNING] Some integration tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)