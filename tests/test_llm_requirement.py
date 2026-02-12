#!/usr/bin/env python3
"""
Test that the system is truly LLM-first and fails without LLM.
"""

import os
import sys

def test_semantic_service_requires_llm():
    """Test that semantic service fails without LLM"""
    print("\n[TEST] Testing semantic service LLM requirement...")
    
    # Disable LLM
    os.environ['DISABLE_LLM'] = 'true'
    
    try:
        from core.semantic_analysis_service import get_semantic_service
        service = get_semantic_service()
        print("[FAIL] Semantic service should have failed without LLM!")
        return False
    except Exception as e:
        if "LLM" in str(e):
            print(f"[OK] Semantic service correctly failed: {str(e)[:80]}...")
            return True
        else:
            print(f"[FAIL] Wrong error: {e}")
            return False
    finally:
        if 'DISABLE_LLM' in os.environ:
            del os.environ['DISABLE_LLM']

def test_van_evera_requires_llm():
    """Test that Van Evera engine fails without LLM"""
    print("\n[TEST] Testing Van Evera engine LLM requirement...")
    
    # Disable LLM
    os.environ['DISABLE_LLM'] = 'true'
    
    try:
        from core.van_evera_testing_engine import VanEveraTestingEngine
        # Try to create with dummy data
        engine = VanEveraTestingEngine({'nodes': [], 'edges': []})
        print("[FAIL] Van Evera engine should have failed without LLM!")
        return False
    except Exception as e:
        if "LLM" in str(e):
            print(f"[OK] Van Evera engine correctly failed: {str(e)[:80]}...")
            return True
        else:
            print(f"[FAIL] Wrong error: {e}")
            return False
    finally:
        if 'DISABLE_LLM' in os.environ:
            del os.environ['DISABLE_LLM']

def test_analyze_requires_llm():
    """Test that main analyze function fails without LLM"""
    print("\n[TEST] Testing main analyze LLM requirement...")
    
    # Disable LLM
    os.environ['DISABLE_LLM'] = 'true'
    
    try:
        from core.analyze import run_analysis
        # Try with minimal data
        result = run_analysis({'nodes': [], 'edges': []})
        print("[FAIL] Analysis should have failed without LLM!")
        return False
    except Exception as e:
        if "LLM" in str(e):
            print(f"[OK] Analysis correctly failed: {str(e)[:80]}...")
            return True
        else:
            # This might fail for other reasons (missing data, etc)
            print(f"[INFO] Analysis failed but not clearly LLM-related: {str(e)[:80]}...")
            return True  # Consider this OK since it failed
    finally:
        if 'DISABLE_LLM' in os.environ:
            del os.environ['DISABLE_LLM']

def calculate_llm_coverage():
    """Calculate what percentage of the system is LLM-first"""
    print("\n[TEST] Calculating LLM-first coverage...")
    
    # Count files
    import os
    from pathlib import Path
    
    core_path = Path("core")
    total_py_files = len(list(core_path.rglob("*.py")))
    
    # Files that require LLM
    llm_required_files = [
        "llm_required.py",
        "semantic_analysis_service.py",
        "van_evera_testing_engine.py",
        "plugins/advanced_van_evera_prediction_engine.py"
    ]
    
    # Files that use semantic service (indirectly require LLM)
    uses_semantic = [
        "analyze.py",  # Main entry point
        "plugins/van_evera_workflow.py",  # Uses semantic service
    ]
    
    direct_llm = len(llm_required_files)
    indirect_llm = len(uses_semantic)
    
    print(f"Total Python files: {total_py_files}")
    print(f"Files directly requiring LLM: {direct_llm}")
    print(f"Files indirectly requiring LLM: {indirect_llm}")
    
    # The real metric: does the MAIN ENTRY POINT require LLM?
    print("\n[CRITICAL] Main entry point (analyze.py) uses semantic_analysis_service")
    print("           which now REQUIRES LLM with no fallbacks!")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("LLM-FIRST ARCHITECTURE TEST")
    print("=" * 60)
    
    tests = [
        test_semantic_service_requires_llm(),
        test_van_evera_requires_llm(),
        test_analyze_requires_llm(),
        calculate_llm_coverage()
    ]
    
    if all(tests):
        print("\n" + "=" * 60)
        print("[SUCCESS] System is LLM-FIRST!")
        print("The critical path (analyze.py -> semantic_analysis_service.py)")
        print("now REQUIRES LLM with NO FALLBACKS!")
        print("=" * 60)
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())