#!/usr/bin/env python3
"""
Test individual plugin imports to isolate Pydantic validation error source.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a single module with detailed error reporting."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Module: {module_name}")
    print('='*60)
    
    try:
        print(f"[TEST] Attempting to import {module_name}...")
        module = __import__(module_name, fromlist=[''])
        print(f"[SUCCESS] {module_name} imported successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to import {module_name}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Test plugin imports in isolation to find validation error source."""
    
    print("SYSTEMATIC PLUGIN IMPORT TESTING")
    print("Goal: Find which plugin import triggers Pydantic validation error")
    
    # Test core dependencies first
    tests = [
        ("pathlib", "Standard pathlib"),
        ("json", "Standard json"),
        ("networkx", "NetworkX graph library"),
        ("pydantic", "Pydantic validation library"),
        ("litellm", "LiteLLM interface"),
        
        # Test our core modules
        ("core.graph_schema", "Our graph schema definitions"),
        ("core.structured_extractor", "Our extraction service"),
        
        # Test plugin components individually
        ("core.plugins.van_evera_llm_schemas", "Van Evera LLM schemas"),
        ("core.plugins.van_evera_llm_interface", "Van Evera LLM interface (SUSPECT)"),
        
        # Test the analyze module that triggers the error
        ("core.analyze", "Core analysis module (triggers error during import)")
    ]
    
    results = []
    for module_name, description in tests:
        success = test_import(module_name, description)
        results.append((module_name, description, success))
        
        if not success and "van_evera" in module_name:
            print(f"\n[ANALYSIS] Found failure in Van Evera component: {module_name}")
            print("This may be our validation error source!")
        
        if not success and module_name == "core.analyze":
            print(f"\n[ANALYSIS] Core analyze import failed as expected")
            print("This confirms the error occurs during analyze module import")
    
    print(f"\n{'='*60}")
    print("IMPORT TEST SUMMARY")
    print('='*60)
    
    for module_name, description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {module_name:30} | {description}")
    
    failed_modules = [name for name, _, success in results if not success]
    if failed_modules:
        print(f"\nFAILED MODULES: {failed_modules}")
        print("Next step: Investigate the first failed module in detail")
    else:
        print(f"\nAll modules imported successfully!")
        print("The error may occur later during actual usage, not import")

if __name__ == "__main__":
    main()