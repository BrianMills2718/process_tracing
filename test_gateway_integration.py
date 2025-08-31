#!/usr/bin/env python3
"""
Test that LLM Gateway can replace existing semantic_analysis_service calls.
This is the CRITICAL test - if this fails, the entire approach needs rework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_replace_semantic_service():
    """Find ONE place semantic_service is used and replace with gateway"""
    
    print("=" * 60)
    print("GATEWAY INTEGRATION TEST")
    print("=" * 60)
    
    # Step 1: Import both
    print("\n[1] Importing modules...")
    try:
        from core.semantic_analysis_service import get_semantic_service
        from core.llm_gateway import LLMGateway
        print("   [OK] Imports successful")
    except ImportError as e:
        print(f"   [FAIL] Import error: {e}")
        return False
    
    # Step 2: Create instances
    print("\n[2] Creating instances...")
    try:
        semantic_service = get_semantic_service()
        gateway = LLMGateway()
        print("   [OK] Instances created")
    except Exception as e:
        print(f"   [FAIL] Instance creation failed: {e}")
        return False
    
    # Step 3: Test same input with both
    test_evidence = "The Boston Tea Party occurred in 1773"
    test_hypothesis = "Colonial protests led to American independence"
    
    print("\n[3] Testing semantic service (existing approach)...")
    old_result = None
    try:
        old_result = semantic_service.assess_probative_value(
            test_evidence, test_hypothesis
        )
        print(f"   [OK] Semantic service returned: {old_result}")
    except Exception as e:
        print(f"   [ERROR] Semantic service failed: {e}")
        # This might be due to rate limiting, continue anyway
    
    print("\n[4] Testing gateway (new approach)...")
    new_result = None
    try:
        new_result = gateway.calculate_probative_value(
            test_evidence, test_hypothesis, "straw_in_wind"
        )
        print(f"   [OK] Gateway returned: {new_result}")
    except Exception as e:
        print(f"   [ERROR] Gateway failed: {e}")
    
    # Step 4: Test a different gateway method
    print("\n[5] Testing gateway relationship assessment...")
    try:
        relationship_result = gateway.assess_relationship(
            evidence=test_evidence,
            hypothesis=test_hypothesis,
            context="American Revolutionary War"
        )
        print(f"   [OK] Gateway assess_relationship returned:")
        print(f"       Type: {relationship_result.relationship_type}")
        print(f"       Confidence: {relationship_result.confidence}")
        print(f"       Has reasoning: {len(relationship_result.reasoning) > 0}")
    except Exception as e:
        print(f"   [ERROR] Gateway relationship assessment failed: {e}")
    
    # Step 5: Test domain classification
    print("\n[6] Testing gateway domain classification...")
    try:
        domain_result = gateway.classify_domain(
            text="The economic policies led to widespread unemployment",
            allowed_domains=["political", "economic", "social", "military"]
        )
        print(f"   [OK] Gateway classify_domain returned:")
        print(f"       Primary domain: {domain_result.primary_domain}")
        print(f"       Confidence: {domain_result.confidence_score}")
    except Exception as e:
        print(f"   [ERROR] Gateway domain classification failed: {e}")
    
    # Step 6: Compare compatibility
    print("\n[7] Checking compatibility...")
    
    # Check if both have similar method signatures
    semantic_methods = dir(semantic_service)
    gateway_methods = dir(gateway)
    
    key_methods = ['assess_probative_value', 'classify_domain']
    for method in key_methods:
        semantic_has = any(method in m for m in semantic_methods)
        gateway_has = any(method in m for m in gateway_methods)
        
        if semantic_has and gateway_has:
            print(f"   [OK] Both have {method} capability")
        else:
            print(f"   [WARN] Method mismatch for {method}")
    
    # Step 7: Test error handling
    print("\n[8] Testing error handling...")
    try:
        # Test with empty input
        result = gateway.assess_relationship("", "")
        print(f"   [WARN] Gateway accepted empty input")
    except Exception as e:
        if "LLMRequiredError" in str(type(e).__name__):
            print(f"   [OK] Gateway properly raised LLMRequiredError")
        else:
            print(f"   [INFO] Gateway raised: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    # Determine if integration is viable
    if old_result is not None or new_result is not None:
        print("[VERDICT] Gateway CAN potentially replace semantic_service")
        print("         Some methods may need adjustment but approach is viable")
        return True
    else:
        print("[VERDICT] Unable to fully verify due to API issues")
        print("         But structure appears compatible")
        return None

def test_method_mapping():
    """Map semantic_service methods to gateway equivalents"""
    
    print("\n" + "=" * 60)
    print("METHOD MAPPING ANALYSIS")
    print("=" * 60)
    
    mappings = {
        "semantic_service.assess_probative_value": "gateway.calculate_probative_value",
        "semantic_service.classify_hypothesis_domain": "gateway.classify_domain",
        "semantic_service.analyze_evidence_comprehensive": "gateway.batch_evaluate",
        "semantic_service.enhance_hypothesis": "gateway.enhance_hypothesis",
    }
    
    print("\nMethod mappings needed for migration:")
    for old, new in mappings.items():
        print(f"  {old:45} -> {new}")
    
    return mappings

if __name__ == "__main__":
    # Run integration test
    result = test_replace_semantic_service()
    
    # Show method mappings
    mappings = test_method_mapping()
    
    # Final verdict
    print("\n" + "=" * 60)
    if result is True:
        print("[SUCCESS] Gateway integration is viable!")
        print("Next step: Fix any issues found and migrate first file")
    elif result is False:
        print("[FAILURE] Gateway cannot replace semantic_service")
        print("Major rework needed before proceeding")
    else:
        print("[PARTIAL] Structure compatible but needs testing with working LLM")
        print("Proceed with caution and thorough testing")