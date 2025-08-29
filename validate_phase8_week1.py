#!/usr/bin/env python3
"""
Validate Phase 8 Week 1 progress - File classification and gateway design.
"""

import os
import sys
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_file_classification():
    """Test that all files are classified"""
    print("\n[TEST] Checking file classification...")
    
    classification_file = Path("evidence/current/Evidence_Phase8_Classification.md")
    if not classification_file.exists():
        print("[FAIL] Classification file not found")
        return False
    
    # Count total Python files
    total_files = len(list(Path("core").rglob("*.py")))
    print(f"Total Python files: {total_files}")
    
    # Check classification completeness
    content = classification_file.read_text()
    if f"Total files classified: {total_files}" in content:
        print("[OK] All files classified")
        return True
    else:
        print("[FAIL] Not all files classified")
        return False

def test_fallback_inventory():
    """Test that fallback patterns are documented"""
    print("\n[TEST] Checking fallback inventory...")
    
    inventory_file = Path("evidence/current/Evidence_Phase8_Fallback_Inventory.md")
    if not inventory_file.exists():
        print("[FAIL] Fallback inventory not found")
        return False
    
    content = inventory_file.read_text()
    required_sections = [
        "return None patterns",
        "except with return patterns",
        "hardcoded thresholds",
        "Priority Migration List"
    ]
    
    for section in required_sections:
        if section in content:
            print(f"[OK] Found {section}")
        else:
            print(f"[FAIL] Missing {section}")
            return False
    
    return True

def test_gateway_design():
    """Test that LLM Gateway design is complete"""
    print("\n[TEST] Checking gateway design...")
    
    gateway_file = Path("core/llm_gateway.py")
    if not gateway_file.exists():
        print("[INFO] Gateway implementation not yet started (expected for Week 1)")
        
    design_file = Path("evidence/current/Evidence_Phase8_Gateway_Design.md")
    if not design_file.exists():
        print("[FAIL] Gateway design document not found")
        return False
    
    content = design_file.read_text()
    required_elements = [
        "Class: LLMGateway",
        "Method signatures",
        "Migration strategy",
        "Error handling"
    ]
    
    for element in required_elements:
        if element in content:
            print(f"[OK] Design includes {element}")
        else:
            print(f"[FAIL] Design missing {element}")
            return False
    
    return True

def test_current_coverage():
    """Test current LLM coverage metrics"""
    print("\n[TEST] Checking current LLM coverage...")
    
    from validate_true_llm_coverage import main as check_coverage
    
    # Capture coverage metrics
    import io
    import contextlib
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        check_coverage()
    output = f.getvalue()
    
    # Parse metrics
    if "~30-40%" in output or "~30%" in output:
        print("[OK] Baseline coverage established: ~30%")
        return True
    else:
        print("[WARN] Coverage metrics unclear")
        return False

def main():
    """Run all Week 1 validation tests"""
    print("=" * 60)
    print("PHASE 8 WEEK 1 VALIDATION")
    print("=" * 60)
    
    tests = [
        ("File Classification", test_file_classification),
        ("Fallback Inventory", test_fallback_inventory),
        ("Gateway Design", test_gateway_design),
        ("Coverage Baseline", test_current_coverage)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("WEEK 1 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] Week 1 foundation complete!")
        print("Ready to proceed to Week 2: Gateway Implementation")
    else:
        print("[INCOMPLETE] Complete remaining Week 1 tasks before proceeding")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())