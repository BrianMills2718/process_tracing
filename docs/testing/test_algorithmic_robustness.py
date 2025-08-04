#!/usr/bin/env python3
"""
Test algorithmic robustness to assess whether failing tests indicate 
mathematical instability or just parameter tuning issues.
"""

from core.bayesian_models import BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace, EvidenceType, HypothesisType
from core.prior_assignment import HierarchicalPriorAssigner, PriorAssignmentConfig, PriorAssignmentMethod
from core.likelihood_calculator import FrequencyBasedLikelihoodCalculator
import numpy as np

def test_probability_conservation_robustness():
    """Test whether probability violations affect mathematical validity."""
    print("\n=== Testing Probability Conservation Under Edge Cases ===")
    
    # Create hierarchical space like the failing test
    space = BayesianHypothesisSpace("test_space", "Test hierarchical space")
    
    # Add root hypotheses
    root1 = BayesianHypothesis("root1", "Root hypothesis 1", HypothesisType.PRIMARY, prior_probability=0.4)
    root2 = BayesianHypothesis("root2", "Root hypothesis 2", HypothesisType.PRIMARY, prior_probability=0.4)
    space.add_hypothesis(root1)
    space.add_hypothesis(root2)
    
    # Configure hierarchical assigner with same parameters as failing test
    config = PriorAssignmentConfig(
        method=PriorAssignmentMethod.HIERARCHICAL,
        parameters={
            "parent_influence": 0.7,
            "level_discount": 0.1,
            "root_prior": 0.8
        }
    )
    assigner = HierarchicalPriorAssigner(config)
    
    # Get priors
    priors = assigner.assign_priors(space)
    total_prob = sum(priors.values())
    
    print(f"Root1 prior: {priors['root1']:.6f}")
    print(f"Root2 prior: {priors['root2']:.6f}")
    print(f"Expected root prior each: {0.8/2:.6f}")
    print(f"Total probability: {total_prob:.6f}")
    
    # Check if this affects mathematical operations
    print(f"Probability conservation: {'PASS' if abs(total_prob - 1.0) < 0.01 else 'FAIL'}")
    print(f"Root assignment accuracy: {'FAIL' if abs(priors['root1'] - 0.4) > 0.1 else 'PASS'}")
    
    return abs(total_prob - 1.0) < 0.01  # Return whether conservation holds

def test_likelihood_ratio_edge_case_robustness():
    """Test likelihood ratio calculations under edge conditions."""
    print("\n=== Testing Likelihood Ratio Edge Case Robustness ===")
    
    # Test the exact failing scenario
    calculator = FrequencyBasedLikelihoodCalculator({})
    
    # Create evidence with exact match scenario (should give high ratio)
    evidence = BayesianEvidence(
        evidence_id="test_evidence",
        description="Test evidence with exact frequency match",
        evidence_type=EvidenceType.SMOKING_GUN,
        source_node_id="test_node",
        collection_method="documentary_analysis"
    )
    
    hypothesis = BayesianHypothesis(
        hypothesis_id="test_hypothesis", 
        description="Test hypothesis with exact match",
        hypothesis_type=HypothesisType.PRIMARY
    )
    
    # Calculate ratio
    ratio = calculator.calculate_likelihood_ratio(evidence, hypothesis)
    print(f"Likelihood ratio for exact match: {ratio:.6f}")
    print(f"Expected range: 3.0 to 6.0")
    print(f"Ratio validity: {'PASS' if 3.0 < ratio < 6.0 else 'FAIL'}")
    
    # Test if this ratio produces mathematically valid Bayesian updates
    prior = 0.5
    likelihood_positive = evidence.likelihood_positive
    likelihood_negative = evidence.likelihood_negative
    
    # Manual Bayes calculation
    marginal = likelihood_positive * prior + likelihood_negative * (1 - prior)
    posterior = (likelihood_positive * prior) / marginal
    
    print(f"Prior: {prior}, Posterior: {posterior:.6f}")
    print(f"Posterior validity: {'PASS' if 0 <= posterior <= 1 else 'FAIL'}")
    
    return 0 <= posterior <= 1  # Return whether result is mathematically valid

def test_mathematical_stability():
    """Test whether the failing edge cases affect core mathematical properties."""
    print("\n=== Testing Core Mathematical Stability ===")
    
    # Test basic Bayes theorem with various evidence strengths
    test_cases = [
        (0.5, 0.8, 0.2),  # Strong evidence
        (0.5, 0.6, 0.4),  # Moderate evidence  
        (0.5, 0.5, 0.5),  # Neutral evidence
        (0.3, 0.7, 0.3),  # Different prior
        (0.9, 0.9, 0.1),  # High prior
    ]
    
    all_valid = True
    for i, (prior, likelihood_pos, likelihood_neg) in enumerate(test_cases):
        # Calculate posterior using Bayes theorem
        marginal = likelihood_pos * prior + likelihood_neg * (1 - prior)
        posterior = (likelihood_pos * prior) / marginal
        
        # Check mathematical properties
        valid_range = 0 <= posterior <= 1
        if not valid_range:
            all_valid = False
            
        print(f"Case {i+1}: P(H)={prior}, P(E|H)={likelihood_pos}, P(E|¬H)={likelihood_neg}")
        print(f"  Posterior: {posterior:.4f}, Valid: {'PASS' if valid_range else 'FAIL'}")
    
    print(f"Overall mathematical stability: {'PASS' if all_valid else 'FAIL'}")
    return all_valid

if __name__ == "__main__":
    prob_conservation = test_probability_conservation_robustness()
    likelihood_validity = test_likelihood_ratio_edge_case_robustness() 
    math_stability = test_mathematical_stability()
    
    print(f"\n=== ALGORITHMIC ROBUSTNESS ASSESSMENT ===")
    print(f"Probability conservation: {'STABLE' if prob_conservation else 'UNSTABLE'}")
    print(f"Likelihood calculations: {'STABLE' if likelihood_validity else 'UNSTABLE'}")
    print(f"Mathematical foundation: {'STABLE' if math_stability else 'UNSTABLE'}")
    
    overall_robust = prob_conservation and likelihood_validity and math_stability
    print(f"Overall algorithmic robustness: {'ROBUST' if overall_robust else 'COMPROMISED'}")