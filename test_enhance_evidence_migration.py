#!/usr/bin/env python3
"""Test that enhance_evidence.py migration to gateway works correctly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_enhance_evidence():
    print("=" * 60)
    print("ENHANCE EVIDENCE MIGRATION TEST")
    print("=" * 60)
    
    # Test import
    print("\n[1] Testing import...")
    try:
        from core.enhance_evidence import refine_evidence_assessment_with_llm
        print("   [OK] Import successful")
    except ImportError as e:
        print(f"   [FAIL] Import failed: {e}")
        return False
    
    # Test basic functionality
    print("\n[2] Testing evidence assessment...")
    test_evidence = "British troops fired on colonists at Lexington in 1775"
    test_context = "The American Revolution was caused by British aggression"
    
    try:
        result = refine_evidence_assessment_with_llm(
            evidence_description=test_evidence,
            text_content="Historical accounts show British troops opening fire",
            context_info=test_context
        )
        
        if result:
            print("   [OK] Assessment successful")
            print(f"       Type: {result.refined_evidence_type}")
            print(f"       Value: {result.suggested_numerical_probative_value:.2f}")
            print(f"       P(E|H): {result.likelihood_P_E_given_H}")
            print(f"       P(E|~H): {result.likelihood_P_E_given_NotH}")
            return True
        else:
            print("   [FAIL] Assessment returned None")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhance_evidence()
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] Migration working correctly!")
    else:
        print("[FAIL] Migration has issues")
    
    sys.exit(0 if success else 1)