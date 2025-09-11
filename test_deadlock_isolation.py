#!/usr/bin/env python3
"""
Systematic Deadlock Isolation Testing
=====================================

This script systematically tests different aspects of the load_graph function call
to identify exactly what's causing the deadlock in the analysis pipeline.
"""

import sys
import os
import time
import threading
import signal
import subprocess
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/home/brian/projects/process_tracing')

def test_1_direct_import_and_call():
    """Test 1: Direct import and function call (known working)"""
    print("="*60)
    print("TEST 1: Direct import and function call")
    print("="*60)
    
    try:
        from core.analyze import load_graph
        json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
        
        start_time = time.time()
        G, data = load_graph(json_file)
        duration = time.time() - start_time
        
        print(f"✅ SUCCESS: Direct call completed in {duration:.2f}s")
        print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_2_replicate_pipeline_context():
    """Test 2: Replicate the exact calling context from the pipeline"""
    print("="*60)
    print("TEST 2: Replicate pipeline calling context")
    print("="*60)
    
    try:
        # Replicate the imports and setup from main()
        import argparse
        import logging
        from core.analyze import load_graph
        from core.llm_reporting_utils import log_structured_error, create_analysis_context
        
        # Replicate argument parsing
        class MockArgs:
            json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
            html = True
            
        args = MockArgs()
        
        # Replicate logger setup
        logger = logging.getLogger(__name__)
        
        print("Calling load_graph in replicated context...")
        start_time = time.time()
        
        # This is the exact same call that hangs in the pipeline
        G, data = load_graph(args.json_file)
        
        duration = time.time() - start_time
        print(f"✅ SUCCESS: Replicated context call completed in {duration:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_with_timeout_handlers():
    """Test 3: Test with timeout signal handlers installed"""
    print("="*60)
    print("TEST 3: With timeout signal handlers")
    print("="*60)
    
    def timeout_handler(signum, frame):
        print("TIMEOUT HANDLER TRIGGERED!")
        raise TimeoutError("Function call timed out")
    
    try:
        from core.analyze import load_graph
        json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
        
        # Install the same timeout handler used in the main pipeline
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        print("Calling load_graph with timeout handler...")
        start_time = time.time()
        G, data = load_graph(json_file)
        duration = time.time() - start_time
        
        signal.alarm(0)  # Cancel alarm
        print(f"✅ SUCCESS: With timeout handler completed in {duration:.2f}s")
        return True
        
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        print(f"❌ FAILED: {e}")
        return False

def test_4_threading_context():
    """Test 4: Test in different threading contexts"""
    print("="*60)
    print("TEST 4: Different threading contexts")
    print("="*60)
    
    results = {}
    
    def run_in_thread(name):
        try:
            from core.analyze import load_graph
            json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
            
            start_time = time.time()
            G, data = load_graph(json_file)
            duration = time.time() - start_time
            
            results[name] = {'success': True, 'duration': duration}
            print(f"✅ {name}: SUCCESS in {duration:.2f}s")
            
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"❌ {name}: FAILED - {e}")
    
    # Test in main thread
    print("Testing in main thread...")
    run_in_thread("main_thread")
    
    # Test in separate thread
    print("Testing in separate thread...")
    thread = threading.Thread(target=run_in_thread, args=("separate_thread",))
    thread.start()
    thread.join(timeout=15)
    
    if thread.is_alive():
        print("❌ separate_thread: TIMEOUT (thread still alive)")
        results["separate_thread"] = {'success': False, 'error': 'timeout'}
    
    return all(r.get('success', False) for r in results.values())

def test_5_minimal_subprocess():
    """Test 5: Minimal subprocess test"""
    print("="*60)
    print("TEST 5: Minimal subprocess")
    print("="*60)
    
    test_script = '''
import sys
sys.path.insert(0, '/home/brian/projects/process_tracing')

try:
    from core.analyze import load_graph
    json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
    
    print("SUBPROCESS: About to call load_graph")
    G, data = load_graph(json_file)
    print(f"SUBPROCESS: SUCCESS - {G.number_of_nodes()} nodes")
    
except Exception as e:
    print(f"SUBPROCESS: ERROR - {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        result = subprocess.run([
            sys.executable, '-c', test_script
        ], capture_output=True, text=True, timeout=15, 
        cwd='/home/brian/projects/process_tracing')
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ SUCCESS: Subprocess completed")
            return True
        else:
            print(f"❌ FAILED: Return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ FAILED: Subprocess timed out")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_6_exact_pipeline_replication():
    """Test 6: Exact replication of problematic pipeline section"""
    print("="*60)
    print("TEST 6: Exact pipeline replication")
    print("="*60)
    
    try:
        # Import everything exactly as in the main pipeline
        import time
        import os
        import sys
        import argparse
        import logging
        
        # Set up the exact same environment
        sys.path.insert(0, '/home/brian/projects/process_tracing')
        
        # Mock the exact args structure
        class Args:
            json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
            html = True
            
        args = Args()
        
        # Set up logger exactly like the pipeline
        logger = logging.getLogger(__name__)
        
        print("Replicating exact pipeline call sequence...")
        
        # This is the EXACT sequence from the main function
        load_start_main = time.time()
        
        print("[PIPELINE-TEST] About to call load_graph...")
        sys.stdout.flush()
        
        from core.analyze import load_graph
        
        print(f"[PIPELINE-TEST] Calling load_graph('{args.json_file}')")
        sys.stdout.flush()
        
        # THE PROBLEMATIC LINE:
        G, data = load_graph(args.json_file)
        
        print(f"[PIPELINE-TEST] ✅ SUCCESS: load_graph completed in {time.time() - load_start_main:.1f}s")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_import_order_investigation():
    """Test 7: Test different import orders"""
    print("="*60)
    print("TEST 7: Import order investigation")
    print("="*60)
    
    import importlib
    import sys
    
    # Test different import sequences
    test_cases = [
        "analyze_first",
        "logging_first", 
        "argparse_first",
        "all_together"
    ]
    
    results = {}
    
    for test_case in test_cases:
        try:
            print(f"Testing import order: {test_case}")
            
            # Clear any cached imports
            modules_to_clear = [k for k in sys.modules.keys() if k.startswith('core.analyze')]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]
            
            if test_case == "analyze_first":
                from core.analyze import load_graph
                import logging
                import argparse
                
            elif test_case == "logging_first":
                import logging
                import argparse  
                from core.analyze import load_graph
                
            elif test_case == "argparse_first":
                import argparse
                import logging
                from core.analyze import load_graph
                
            elif test_case == "all_together":
                import logging, argparse
                from core.analyze import load_graph
            
            # Test the function call
            json_file = 'output_data/revolutions/revolutions_20250910_081813_graph.json'
            start_time = time.time()
            G, data = load_graph(json_file)
            duration = time.time() - start_time
            
            results[test_case] = {'success': True, 'duration': duration}
            print(f"   ✅ SUCCESS in {duration:.2f}s")
            
        except Exception as e:
            results[test_case] = {'success': False, 'error': str(e)}
            print(f"   ❌ FAILED: {e}")
    
    return all(r.get('success', False) for r in results.values())

def main():
    """Run all systematic tests"""
    print("SYSTEMATIC DEADLOCK ISOLATION TESTING")
    print("=" * 60)
    print()
    
    # Change to project directory
    os.chdir('/home/brian/projects/process_tracing')
    
    tests = [
        ("Direct Import Call", test_1_direct_import_and_call),
        ("Pipeline Context", test_2_replicate_pipeline_context), 
        ("Timeout Handlers", test_3_with_timeout_handlers),
        ("Threading Context", test_4_threading_context),
        ("Subprocess", test_5_minimal_subprocess),
        ("Exact Pipeline", test_6_exact_pipeline_replication),
        ("Import Order", test_7_import_order_investigation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print()
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ TEST CRASHED: {e}")
            results[test_name] = False
        
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed - issue may be environment-specific")
    else:
        failing_tests = [name for name, success in results.items() if not success]
        print(f"🔍 Focus investigation on: {', '.join(failing_tests)}")

if __name__ == "__main__":
    main()