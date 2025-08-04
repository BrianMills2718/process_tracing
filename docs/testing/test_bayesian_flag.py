#!/usr/bin/env python3
"""
Test --bayesian Flag Integration

Test that the --bayesian flag works correctly with process_trace_advanced.py
"""

import sys
import subprocess
from pathlib import Path

def test_bayesian_flag_parsing():
    """Test that --bayesian flag is parsed correctly."""
    print("=== Testing --bayesian Flag Parsing ===")
    
    # Test help output includes Bayesian options
    result = subprocess.run([
        sys.executable, 'process_trace_advanced.py', '--help'
    ], capture_output=True, text=True, cwd=str(Path(__file__).parent))
    
    if '--bayesian' in result.stdout:
        print("[OK] --bayesian flag found in help")
    else:
        print("[ERROR] --bayesian flag not found in help")
        return False
    
    # Test that we can import and check BAYESIAN_AVAILABLE
    sys.path.insert(0, str(Path(__file__).parent))
    import process_trace_advanced
    
    if process_trace_advanced.BAYESIAN_AVAILABLE:
        print("[OK] BAYESIAN_AVAILABLE is True")
    else:
        print("[ERROR] BAYESIAN_AVAILABLE is False")
        return False
    
    # Test config creation
    class MockArgs:
        def __init__(self):
            self.bayesian = True
            self.simulations = 100
            self.confidence_level = 0.95
            self.no_uncertainty = True
            self.no_visualizations = True
    
    args = MockArgs()
    config = process_trace_advanced.create_bayesian_config_from_args(args)
    
    if config is not None:
        print(f"[OK] Bayesian config created with {config.uncertainty_simulations} simulations")
        return True
    else:
        print("[ERROR] Failed to create Bayesian config")
        return False

def test_extract_only_with_bayesian():
    """Test --extract-only with --bayesian flag (should be much faster)."""
    print("\n=== Testing --extract-only with --bayesian ===")
    
    try:
        result = subprocess.run([
            sys.executable, 'process_trace_advanced.py', 
            '--project', 'test_project',
            '--extract-only',
            '--bayesian', 
            '--simulations', '5',
            '--no-visualizations',
            '--no-uncertainty'
        ], capture_output=True, text=True, timeout=30, 
           cwd=str(Path(__file__).parent))
        
        if result.returncode == 0:
            print("[OK] --extract-only with --bayesian completed successfully")
            if "Bayesian analysis enabled" in result.stdout:
                print("[OK] Bayesian configuration was processed")
            return True
        else:
            print(f"[ERROR] Command failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout[-500:]}")
            print(f"STDERR: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Command timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"[ERROR] Command failed: {e}")
        return False

def main():
    """Run Bayesian flag tests."""
    print("Testing --bayesian Flag Integration")
    print("=" * 50)
    
    tests = [
        ("Bayesian Flag Parsing", test_bayesian_flag_parsing),
        ("Extract-Only with Bayesian", test_extract_only_with_bayesian)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test_name} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("BAYESIAN FLAG TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n[SUCCESS] --bayesian flag integration is working!")
        print("Users can now use --bayesian with process_trace_advanced.py")
        return True
    else:
        print(f"\n[WARNING] {total-passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)