#!/usr/bin/env python3
"""
Test the Pydantic validation fix for probative value assessment
"""

import sys
import traceback

def test_probative_value_assessment():
    """Test that assess_probative_value works without Pydantic validation errors."""
    print("TESTING PROBATIVE VALUE ASSESSMENT FIX")
    print("="*50)
    
    try:
        # Import the fixed interface
        print("[TEST] Importing VanEveraLLMInterface...")
        from core.plugins.van_evera_llm_interface import VanEveraLLMInterface
        
        print("[TEST] Creating interface...")
        interface = VanEveraLLMInterface()
        
        print("[TEST] Testing assess_probative_value with simple inputs...")
        
        # Simple test case
        evidence = "The treaty was signed in 1783"
        hypothesis = "The American Revolution ended in 1783"
        
        print(f"[TEST] Evidence: {evidence}")
        print(f"[TEST] Hypothesis: {hypothesis}")
        print("[TEST] Calling assess_probative_value...")
        
        # This should now work without Pydantic validation errors
        result = interface.assess_probative_value(
            evidence_description=evidence,
            hypothesis_description=hypothesis
        )
        
        print("[SUCCESS] assess_probative_value completed successfully!")
        print(f"[RESULT] Probative value: {result.probative_value}")
        print(f"[RESULT] Confidence: {result.confidence_score}")
        print(f"[RESULT] Reasoning: {result.reasoning[:100]}...")
        print(f"[RESULT] Quality factors: {len(result.evidence_quality_factors)} factors")
        print(f"[RESULT] Reliability: {result.reliability_assessment[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {type(e).__name__}: {str(e)}")
        print("[TRACEBACK]")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_probative_value_assessment()
    if success:
        print("\n✅ PYDANTIC VALIDATION FIX SUCCESSFUL")
        sys.exit(0)
    else:
        print("\n❌ PYDANTIC VALIDATION FIX FAILED")  
        sys.exit(1)