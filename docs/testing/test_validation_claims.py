#!/usr/bin/env python3
"""
Critical validation test to verify Phase 6A completion claims.
Tests the specific issues mentioned in the original CLAUDE.md requirements.
"""

from core.bayesian_models import BayesianEvidence, EvidenceType
import pytest

def test_bayesian_evidence_auto_assignment_fix():
    """Test that BayesianEvidence preserves explicit user parameters."""
    print("\n=== Testing BayesianEvidence Auto-Assignment Fix ===")
    
    # Test case from original CLAUDE.md: User explicitly sets likelihood_positive=0.5, likelihood_negative=0.5
    evidence = BayesianEvidence(
        evidence_id="test_evidence",
        description="Test evidence for auto-assignment",
        evidence_type=EvidenceType.HOOP,  # This would normally override to ~0.8/0.2
        source_node_id="test_node",
        likelihood_positive=0.5,  # Explicitly set by user
        likelihood_negative=0.5   # Explicitly set by user
    )
    
    print(f"User set likelihood_positive=0.5, actual value: {evidence.likelihood_positive}")
    print(f"User set likelihood_negative=0.5, actual value: {evidence.likelihood_negative}")
    
    # These must pass for the fix to be valid
    assert evidence.likelihood_positive == 0.5, f"Expected 0.5, got {evidence.likelihood_positive}"
    assert evidence.likelihood_negative == 0.5, f"Expected 0.5, got {evidence.likelihood_negative}"
    
    print("PASS: BayesianEvidence preserves explicit user parameters")

def test_auto_assignment_still_works():
    """Test that auto-assignment still works when user doesn't provide explicit values."""
    print("\n=== Testing Auto-Assignment Still Works ===")
    
    # When user doesn't provide explicit values (uses defaults), Van Evera should apply
    evidence = BayesianEvidence(
        evidence_id="test_evidence_auto",
        description="Test evidence for auto-assignment",
        evidence_type=EvidenceType.HOOP,
        source_node_id="test_node"
        # No explicit likelihood values provided - should use Van Evera defaults
    )
    
    print(f"No explicit values, HOOP evidence likelihood_positive: {evidence.likelihood_positive}")
    print(f"No explicit values, HOOP evidence likelihood_negative: {evidence.likelihood_negative}")
    
    # HOOP evidence should have high likelihood_positive (>=0.8) and low likelihood_negative (<=0.2)
    assert evidence.likelihood_positive >= 0.8, f"HOOP should have high likelihood_positive, got {evidence.likelihood_positive}"
    assert evidence.likelihood_negative <= 0.2, f"HOOP should have low likelihood_negative, got {evidence.likelihood_negative}"
    
    print("PASS: Auto-assignment works when user doesn't provide explicit values")

def test_detection_logic_robustness():
    """Test edge cases for the user-provided detection logic."""
    print("\n=== Testing Detection Logic Robustness ===")
    
    # Edge case: User provides default value (1.0) - should this be considered "user-provided"?
    evidence_edge = BayesianEvidence(
        evidence_id="test_edge",
        description="Edge case test",
        evidence_type=EvidenceType.SMOKING_GUN,
        source_node_id="test_node",
        likelihood_positive=1.0,  # This is the default value
        likelihood_negative=1.0   # This is the default value
    )
    
    print(f"User provided default values (1.0), result: pos={evidence_edge.likelihood_positive}, neg={evidence_edge.likelihood_negative}")
    
    # The current fix checks != 1.0, so providing 1.0 should NOT be considered user-provided
    # This might be a flaw in the detection logic
    
    # For SMOKING_GUN, if defaults are used, should get high likelihood_positive, low likelihood_negative
    # But if user explicitly sets 1.0, should it be preserved?
    print(f"Detection treats 1.0 as: user_provided_pos={evidence_edge._user_provided_likelihood_positive}, user_provided_neg={evidence_edge._user_provided_likelihood_negative}")

if __name__ == "__main__":
    test_bayesian_evidence_auto_assignment_fix()
    test_auto_assignment_still_works() 
    test_detection_logic_robustness()
    print("\n=== All validation tests completed ===")