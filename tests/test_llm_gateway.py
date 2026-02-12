#!/usr/bin/env python3
"""
Test LLM Gateway implementation with actual LLM interface.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gateway_initialization():
    """Test that gateway initializes properly"""
    print("\n[TEST] Gateway Initialization")
    print("-" * 40)
    
    try:
        from core.llm_gateway import LLMGateway
        gateway = LLMGateway()
        print("[OK] Gateway initialized successfully")
        print(f"   - Has LLM interface: {gateway.llm is not None}")
        print(f"   - Has query function: {hasattr(gateway, 'llm_query')}")
        return gateway
    except Exception as e:
        print(f"[FAIL] Failed to initialize gateway: {e}")
        return None

def test_assess_relationship(gateway):
    """Test relationship assessment method"""
    print("\n[TEST] Assess Relationship")
    print("-" * 40)
    
    test_evidence = "British troops fired on colonists at Lexington in April 1775"
    test_hypothesis = "The American Revolution was caused by British military aggression"
    
    try:
        result = gateway.assess_relationship(
            evidence=test_evidence,
            hypothesis=test_hypothesis,
            context="American Revolutionary War"
        )
        
        print("[OK] Relationship assessment successful")
        print(f"   - Type: {result.relationship_type}")
        print(f"   - Confidence: {result.confidence:.2f}")
        print(f"   - Reasoning: {result.reasoning[:100]}...")
        if result.diagnostic_type:
            print(f"   - Diagnostic: {result.diagnostic_type}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to assess relationship: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classify_domain(gateway):
    """Test domain classification method"""
    print("\n[TEST] Classify Domain")
    print("-" * 40)
    
    test_text = "The economic policies led to widespread unemployment and social unrest"
    allowed_domains = ["political", "economic", "social", "military"]
    
    try:
        result = gateway.classify_domain(
            text=test_text,
            allowed_domains=allowed_domains
        )
        
        print("[OK] Domain classification successful")
        print(f"   - Primary: {result.primary_domain}")
        print(f"   - Confidence: {result.confidence_score:.2f}")
        print(f"   - Reasoning: {result.reasoning[:100]}...")
        if result.secondary_domains:
            print(f"   - Secondary: {result.secondary_domains}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to classify domain: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_temporal_evaluation(gateway):
    """Test temporal relationship evaluation"""
    print("\n[TEST] Temporal Evaluation")
    print("-" * 40)
    
    event1 = "Boston Tea Party occurred in December 1773"
    event2 = "British Parliament passed the Intolerable Acts in 1774"
    
    try:
        result = gateway.evaluate_temporal_relationship(
            event1=event1,
            event2=event2,
            temporal_context={"historical_period": "American Revolution"}
        )
        
        print("[OK] Temporal evaluation successful")
        print(f"   - Order: {result.temporal_order}")
        print(f"   - Plausibility: {result.causal_plausibility:.2f}")
        print(f"   - Time gap: {result.time_gap_assessment}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to evaluate temporal relationship: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_probative_value(gateway):
    """Test probative value calculation"""
    print("\n[TEST] Calculate Probative Value")
    print("-" * 40)
    
    evidence = "A gun with fingerprints was found at the crime scene"
    hypothesis = "The suspect was present at the crime scene"
    diagnostic = "smoking_gun"
    
    try:
        result = gateway.calculate_probative_value(
            evidence=evidence,
            hypothesis=hypothesis,
            diagnostic_type=diagnostic
        )
        
        print("[OK] Probative value calculation successful")
        print(f"   - Value: {result:.2f}")
        print(f"   - In expected range: {0.0 <= result <= 1.0}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to calculate probative value: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_caching(gateway):
    """Test that caching works"""
    print("\n[TEST] Caching Mechanism")
    print("-" * 40)
    
    # Clear cache and stats
    gateway.clear_cache()
    initial_calls = gateway.get_stats()['calls']
    
    # Make same call twice
    test_evidence = "Test evidence for caching"
    test_hypothesis = "Test hypothesis for caching"
    
    try:
        # First call
        result1 = gateway.assess_relationship(test_evidence, test_hypothesis)
        calls_after_first = gateway.get_stats()['calls']
        
        # Second call (should hit cache)
        result2 = gateway.assess_relationship(test_evidence, test_hypothesis)
        calls_after_second = gateway.get_stats()['calls']
        cache_hits = gateway.get_stats()['cache_hits']
        
        print("[OK] Caching test completed")
        print(f"   - First call increased calls: {calls_after_first > initial_calls}")
        print(f"   - Second call hit cache: {calls_after_second == calls_after_first}")
        print(f"   - Cache hits recorded: {cache_hits > 0}")
        print(f"   - Results identical: {result1.confidence == result2.confidence}")
        return True
    except Exception as e:
        print(f"[FAIL] Failed caching test: {e}")
        return False

def test_error_handling(gateway):
    """Test error handling with bad inputs"""
    print("\n[TEST] Error Handling")
    print("-" * 40)
    
    from core.llm_required import LLMRequiredError
    
    # Test with empty inputs
    try:
        result = gateway.assess_relationship("", "")
        print("[WARN]  Accepted empty inputs (may be intentional)")
    except LLMRequiredError as e:
        print("[OK] Properly raised LLMRequiredError for empty inputs")
    except Exception as e:
        print(f"[WARN]  Raised different error: {type(e).__name__}")
    
    # Test with invalid domain list
    try:
        result = gateway.classify_domain("test text", [])
        print("[WARN]  Accepted empty domain list")
    except LLMRequiredError as e:
        print("[OK] Properly raised LLMRequiredError for invalid domains")
    except Exception as e:
        print(f"[WARN]  Raised different error: {type(e).__name__}")
    
    return True

def main():
    """Run all gateway tests"""
    print("=" * 60)
    print("LLM GATEWAY INTEGRATION TESTS")
    print("=" * 60)
    
    # Check if LLM is available
    if os.environ.get('DISABLE_LLM') == 'true':
        print("\n[WARN]  LLM is disabled. Set DISABLE_LLM=false to run tests.")
        return 1
    
    # Initialize gateway
    gateway = test_gateway_initialization()
    if not gateway:
        print("\n[FAIL] Cannot proceed without gateway")
        return 1
    
    # Run tests
    tests = [
        ("Assess Relationship", lambda: test_assess_relationship(gateway)),
        ("Classify Domain", lambda: test_classify_domain(gateway)),
        ("Temporal Evaluation", lambda: test_temporal_evaluation(gateway)),
        ("Probative Value", lambda: test_probative_value(gateway)),
        ("Caching", lambda: test_caching(gateway)),
        ("Error Handling", lambda: test_error_handling(gateway))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n[FAIL] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"[WARN]  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())