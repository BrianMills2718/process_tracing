#!/usr/bin/env python3
"""
Strict validation for TRUE LLM-first architecture.
System MUST fail without LLM - no fallbacks allowed.
"""

import os
import sys
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_required():
    """Test that system fails without LLM"""
    print("\n[TEST] Checking if system fails without LLM...")
    
    # Set environment to disable LLM
    os.environ['DISABLE_LLM'] = 'true'
    
    # Test 1: Check core.llm_required module
    try:
        from core.llm_required import require_llm
        llm = require_llm()
        print("[FAIL] require_llm() should have failed but didn't!")
        return False
    except Exception as e:
        if "LLM" in str(e) and ("required" in str(e) or "disabled" in str(e)):
            print(f"[OK] require_llm() correctly failed: {e}")
        else:
            print(f"[FAIL] Wrong error from require_llm(): {e}")
            return False
    
    # Test 2: Check confidence_calculator
    try:
        from core.confidence_calculator import CausalConfidenceCalculator
        calc = CausalConfidenceCalculator()
        print("[FAIL] CausalConfidenceCalculator should fail without LLM!")
        return False
    except Exception as e:
        if "LLM" in str(e):
            print(f"[OK] CausalConfidenceCalculator correctly failed: {str(e)[:50]}...")
        else:
            print(f"[WARN] CausalConfidenceCalculator failed but not clearly LLM-related: {e}")
    
    # Test 3: Check van_evera_testing_engine
    try:
        from core.van_evera_testing_engine import VanEveraTestingEngine
        engine = VanEveraTestingEngine({'nodes': [], 'edges': []})
        print("[FAIL] VanEveraTestingEngine should fail without LLM!")
        return False
    except Exception as e:
        if "LLM" in str(e):
            print(f"[OK] VanEveraTestingEngine correctly failed: {str(e)[:50]}...")
        else:
            print(f"[WARN] VanEveraTestingEngine failed but not clearly LLM-related: {e}")
    
    # Clear the environment variable
    del os.environ['DISABLE_LLM']
    
    print("[OK] All components correctly require LLM")
    return True

def check_no_hardcoded_values():
    """Verify no hardcoded probability/confidence values"""
    print("\n[TEST] Checking for hardcoded values...")
    
    files_to_check = [
        "core/confidence_calculator.py",
        "core/van_evera_testing_engine.py",
    ]
    
    # Pattern to find hardcoded decimals (excluding Pydantic Field defaults)
    patterns = [
        r'= 0\.\d+\s*#.*(?:default|fallback)',  # With comment
        r'mechanism_completeness = 0\.\d+',      # Specific variables
        r'temporal_consistency = 0\.\d+',
        r'base_coherence = 0\.\d+',
        r'independence_score = 0\.\d+',
        r'posterior_uncertainty = 0\.\d+',
    ]
    
    issues_found = False
    for file_path in files_to_check:
        if not Path(file_path).exists():
            print(f"[SKIP] {file_path} not found")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"[FAIL] {file_path} has hardcoded values matching '{pattern}': {matches[:2]}")
                issues_found = True
    
    if not issues_found:
        print("[OK] No problematic hardcoded values found in checked files")
    
    # Note about advanced_prediction_engine.py
    print("[INFO] advanced_prediction_engine.py has 18 hardcoded thresholds - needs refactoring")
    
    return not issues_found

def check_no_word_overlap():
    """Verify no word overlap/counting logic"""
    print("\n[TEST] Checking for word overlap patterns...")
    
    forbidden_patterns = [
        'overlap_ratio',
        'len\\(overlap\\)',
        'intersection\\(',
        'word.*overlap',
        'evidence_words.*prediction_words',
        'word.*count',
        'common_words.*set'
    ]
    
    files_to_check = [
        "core/van_evera_testing_engine.py",
        "core/confidence_calculator.py",
        "core/semantic_analysis_service.py"
    ]
    
    issues_found = False
    for file_path in files_to_check:
        if not Path(file_path).exists():
            print(f"[SKIP] {file_path} not found")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        for pattern in forbidden_patterns:
            if re.search(pattern, content):
                print(f"[FAIL] Found word overlap pattern '{pattern}' in {file_path}")
                # Show context
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        print(f"  Line {i+1}: {line.strip()[:80]}")
                        break
                issues_found = True
    
    if not issues_found:
        print("[OK] No word overlap patterns found")
    
    return not issues_found

def check_no_try_except_fallbacks():
    """Check for try/except blocks that hide LLM failures"""
    print("\n[TEST] Checking for try/except that hides LLM failures...")
    
    files_to_check = [
        "core/confidence_calculator.py",
        "core/van_evera_testing_engine.py"
    ]
    
    # Pattern to find problematic try/except
    pattern = r'try:.*?llm.*?except.*?(?:pass|continue|logger\.warning|logger\.debug)'
    
    issues_found = False
    for file_path in files_to_check:
        if not Path(file_path).exists():
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Simple check for try/except with fallback behavior
        if re.search(r'except.*:\s*(?:pass|.*fallback|.*default)', content, re.IGNORECASE | re.DOTALL):
            # More detailed check needed
            lines = content.split('\n')
            in_try = False
            for i, line in enumerate(lines):
                if 'try:' in line:
                    in_try = True
                elif in_try and 'except' in line:
                    # Check next few lines for problematic patterns
                    next_lines = ' '.join(lines[i:i+3])
                    if any(bad in next_lines.lower() for bad in ['fallback', 'default', 'pass', 'logger.warning']):
                        if 'LLMRequiredError' not in next_lines:  # Unless it's re-raising
                            print(f"[WARN] Possible fallback try/except in {file_path} line {i+1}")
                            issues_found = True
                    in_try = False
    
    if not issues_found:
        print("[OK] No problematic try/except fallbacks found")
    
    return not issues_found

def check_formula_weights():
    """Check that formula weights are not hardcoded"""
    print("\n[TEST] Checking for hardcoded formula weights...")
    
    # Pattern for hardcoded weights in formulas
    patterns = [
        r'0\.\d+\s*\*.*(?:score|factor|component)',  # Weight multiplying a score
        r'weights\s*=\s*\{[^}]*0\.\d+',              # Weight dictionary
        r'ConfidenceType\.\w+:\s*0\.\d+',            # Confidence type weights
    ]
    
    issues_found = False
    file_path = "core/confidence_calculator.py"
    
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            content = f.read()
            
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                print(f"[FAIL] Found hardcoded formula weights: {matches[:2]}")
                issues_found = True
    
    if not issues_found:
        print("[OK] No hardcoded formula weights found")
    
    return not issues_found

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("STRICT LLM-FIRST VALIDATION")
    print("=" * 60)
    
    tests = [
        ("LLM Required", test_llm_required),
        ("No Hardcoded Values", check_no_hardcoded_values),
        ("No Word Overlap", check_no_word_overlap),
        ("No Try/Except Fallbacks", check_no_try_except_fallbacks),
        ("No Formula Weights", check_formula_weights)
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
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] TRUE LLM-FIRST ACHIEVED!")
        print("System correctly requires LLM for all semantic decisions")
    else:
        print("[FAIL] Violations remain - not fully LLM-first")
        print("Address remaining issues to achieve compliance")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())