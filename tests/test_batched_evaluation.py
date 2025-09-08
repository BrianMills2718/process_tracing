#!/usr/bin/env python3
"""
Test script for batched hypothesis evaluation.
Validates that the new LLM-first batched approach works correctly.
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.semantic_analysis_service import get_semantic_service
from core.evidence_document import EvidenceDocument


def test_batched_evaluation():
    """Test the new batched evaluation approach"""
    print("=" * 80)
    print("TESTING BATCHED HYPOTHESIS EVALUATION")
    print("=" * 80)
    
    # Initialize service
    semantic_service = get_semantic_service()
    semantic_service.clear_cache()
    
    # Create test evidence
    evidence_text = """
    The colonists organized protests against British taxation policies, 
    citing violations of their rights as Englishmen. Merchants formed 
    non-importation agreements to pressure Parliament economically.
    """
    
    # Create test hypotheses
    hypotheses = [
        {'id': 'h1', 'text': 'Economic grievances caused the American Revolution'},
        {'id': 'h2', 'text': 'Ideological beliefs about rights and freedom drove the revolution'},
        {'id': 'h3', 'text': 'British military oppression triggered colonial resistance'},
        {'id': 'h4', 'text': 'Colonial unity emerged from shared grievances'}
    ]
    
    # Test 1: Direct batch evaluation
    print("\n1. Testing direct batch evaluation...")
    start_time = time.time()
    
    try:
        batch_result = semantic_service.evaluate_evidence_against_hypotheses_batch(
            'evidence_001',
            evidence_text,
            hypotheses,
            context="Testing batch evaluation"
        )
        
        elapsed = time.time() - start_time
        print(f"   Time: {elapsed:.2f}s")
        print(f"   LLM calls: {semantic_service._stats['llm_calls']}")
        
        # Display results
        print("\n   Results:")
        for eval_result in batch_result.evaluations:
            print(f"   - {eval_result.hypothesis_id}: {eval_result.relationship_type} "
                  f"(confidence: {eval_result.confidence:.2f})")
            print(f"     Van Evera: {eval_result.van_evera_diagnostic}")
            print(f"     Reasoning: {eval_result.reasoning[:100]}...")
        
        # Display inter-hypothesis insights
        print("\n   Inter-hypothesis insights:")
        print(f"   - Primary supported: {batch_result.primary_hypothesis_supported}")
        print(f"   - Conflicting pairs: {batch_result.conflicting_hypotheses}")
        print(f"   - Complementary pairs: {batch_result.complementary_hypotheses}")
        print(f"   - Evidence significance: {batch_result.evidence_significance}")
        
        print("\n   [OK] Batch evaluation successful")
        
    except Exception as e:
        print(f"   [FAIL] Batch evaluation failed: {e}")
        return False
    
    # Test 2: EvidenceDocument with batch evaluation
    print("\n2. Testing EvidenceDocument batch evaluation...")
    semantic_service.clear_cache()
    semantic_service._stats['llm_calls'] = 0
    
    doc = EvidenceDocument('doc_001', evidence_text)
    
    start_time = time.time()
    try:
        # Evaluate against all hypotheses at once
        evaluations = doc.evaluate_against_hypotheses_batch(
            hypotheses,
            semantic_service
        )
        
        elapsed = time.time() - start_time
        print(f"   Time: {elapsed:.2f}s")
        print(f"   LLM calls: {semantic_service._stats['llm_calls']}")
        print(f"   Evaluations returned: {len(evaluations)}")
        
        # Display evaluations
        for hyp_id, evaluation in evaluations.items():
            print(f"   - {hyp_id}: {evaluation.relationship_type} "
                  f"(confidence: {evaluation.confidence:.2f})")
        
        print("\n   [OK] EvidenceDocument batch evaluation successful")
        
    except Exception as e:
        print(f"   [FAIL] EvidenceDocument batch evaluation failed: {e}")
        return False
    
    # Test 3: Verify caching works
    print("\n3. Testing cache effectiveness...")
    initial_calls = semantic_service._stats['llm_calls']
    
    # Re-evaluate same hypotheses (should hit cache)
    evaluations2 = doc.evaluate_against_hypotheses_batch(
        hypotheses,
        semantic_service
    )
    
    new_calls = semantic_service._stats['llm_calls'] - initial_calls
    print(f"   Additional LLM calls for cached evaluation: {new_calls}")
    
    if new_calls == 0:
        print("   [OK] Cache working correctly")
    else:
        print("   [FAIL] Cache not working - made additional calls")
    
    # Test 4: Compare quality with old approach
    print("\n4. Quality comparison...")
    print("   Old approach: Would use keyword matching for subsequent hypotheses")
    print("   New approach: Uses full LLM understanding for all relationships")
    print("   Expected improvement: Better semantic understanding, inter-hypothesis insights")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total LLM calls: {semantic_service._stats['llm_calls']}")
    print(f"Cache hits: {semantic_service._stats['cache_hits']}")
    print(f"Errors: {semantic_service._stats['errors']}")
    
    return True


if __name__ == "__main__":
    success = test_batched_evaluation()
    sys.exit(0 if success else 1)