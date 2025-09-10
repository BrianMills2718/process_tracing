#!/usr/bin/env python3
"""
Test that BayesianEvidence functionality has been restored successfully.

Tests all the previously disabled methods to ensure they work correctly
with the new bayesian_models.py implementation.
"""

def test_bayesian_imports():
    """Test that all Bayesian classes can be imported."""
    print("=== Testing Bayesian Imports ===")
    
    try:
        from core.bayesian_models import (
            BayesianEvidence, BayesianHypothesis, BayesianHypothesisSpace,
            EvidenceType, HypothesisType, IndependenceType
        )
        print("✅ All Bayesian classes imported successfully")
        
        from core.evidence_weighting import EvidenceStrengthQuantifier
        print("✅ EvidenceStrengthQuantifier imported successfully")
        
        from core.confidence_calculator import ConfidenceLevel
        print("✅ ConfidenceLevel imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_bayesian_evidence_creation():
    """Test creating BayesianEvidence objects."""
    print("\n=== Testing BayesianEvidence Creation ===")
    
    try:
        from core.bayesian_models import BayesianEvidence, EvidenceType
        
        # Create various types of evidence
        evidence1 = BayesianEvidence(
            evidence_id="e1",
            description="Strong documentary evidence",
            evidence_type=EvidenceType.SMOKING_GUN,
            strength=0.9,
            reliability=0.8,
            source_credibility=0.85
        )
        
        evidence2 = BayesianEvidence(
            evidence_id="e2", 
            description="Necessary precondition observed",
            evidence_type=EvidenceType.HOOP,
            strength=0.7,
            reliability=0.75
        )
        
        print(f"✅ Created smoking gun evidence: {evidence1.evidence_id}")
        print(f"   LR: {evidence1.get_likelihood_ratio():.2f}")
        print(f"   Adjusted LR: {evidence1.get_adjusted_likelihood_ratio():.2f}")
        
        print(f"✅ Created hoop evidence: {evidence2.evidence_id}")
        print(f"   LR: {evidence2.get_likelihood_ratio():.2f}")
        print(f"   Adjusted LR: {evidence2.get_adjusted_likelihood_ratio():.2f}")
        
        return [evidence1, evidence2]
    except Exception as e:
        print(f"❌ Evidence creation failed: {e}")
        return None

def test_previously_disabled_methods(evidence_list):
    """Test the methods that were previously disabled."""
    print("\n=== Testing Previously Disabled Methods ===")
    
    try:
        from core.evidence_weighting import EvidenceStrengthQuantifier
        from core.bayesian_models import IndependenceType
        
        quantifier = EvidenceStrengthQuantifier()
        
        # Test 1: calculate_evidence_diversity
        diversity = quantifier.calculate_evidence_diversity(evidence_list)
        print(f"✅ calculate_evidence_diversity: {diversity:.3f}")
        assert 0.0 <= diversity <= 1.0, f"Diversity out of range: {diversity}"
        
        # Test 2: combine_multiple_evidence
        independence_assumptions = {
            "e1-e2": IndependenceType.INDEPENDENT
        }
        combined_lr = quantifier.combine_multiple_evidence(evidence_list, independence_assumptions)
        print(f"✅ combine_multiple_evidence: {combined_lr:.3f}")
        assert combined_lr > 0, f"Combined LR should be positive: {combined_lr}"
        
        # Test 3: _group_evidence_by_independence
        groups = quantifier._group_evidence_by_independence(evidence_list, independence_assumptions)
        print(f"✅ _group_evidence_by_independence: {len(groups)} groups")
        
        # Test 4: _combine_dependent_evidence
        dependent_lr = quantifier._combine_dependent_evidence(evidence_list)
        print(f"✅ _combine_dependent_evidence: {dependent_lr:.3f}")
        assert dependent_lr > 0, f"Dependent LR should be positive: {dependent_lr}"
        
        print("🎉 ALL PREVIOUSLY DISABLED METHODS NOW WORK!")
        return True
        
    except Exception as e:
        print(f"❌ Previously disabled methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hypothesis_space():
    """Test BayesianHypothesisSpace functionality."""
    print("\n=== Testing BayesianHypothesisSpace ===")
    
    try:
        from core.bayesian_models import (
            BayesianHypothesis, BayesianHypothesisSpace, HypothesisType
        )
        
        # Create hypothesis space
        space = BayesianHypothesisSpace(
            hypothesis_space_id="test_space",
            description="Test hypothesis space"
        )
        
        # Create hypotheses
        hyp1 = BayesianHypothesis(
            hypothesis_id="h1",
            description="Primary causal hypothesis",
            hypothesis_type=HypothesisType.PRIMARY,
            prior_probability=0.6,
            posterior_probability=0.8
        )
        
        hyp2 = BayesianHypothesis(
            hypothesis_id="h2",
            description="Alternative hypothesis", 
            hypothesis_type=HypothesisType.ALTERNATIVE,
            prior_probability=0.4,
            posterior_probability=0.2
        )
        
        # Add to space
        space.add_hypothesis(hyp1)
        space.add_hypothesis(hyp2)
        
        print(f"✅ Created hypothesis space with {len(space.hypotheses)} hypotheses")
        
        # Test competing hypotheses
        competitors = space.get_competing_hypotheses("h1")
        print(f"✅ Found {len(competitors)} competing hypotheses")
        
        # Test summary statistics
        stats = space.get_summary_statistics()
        print(f"✅ Summary stats: {stats['total_hypotheses']} hypotheses, mean posterior: {stats['mean_posterior']:.2f}")
        
        return space
        
    except Exception as e:
        print(f"❌ Hypothesis space test failed: {e}")
        return None

def test_end_to_end_integration(evidence_list, hypothesis_space):
    """Test end-to-end integration with evidence and hypotheses."""
    print("\n=== Testing End-to-End Integration ===")
    
    try:
        from core.evidence_weighting import EvidenceStrengthQuantifier
        from core.structured_models import EvidenceAssessment
        
        # Add evidence to hypothesis space
        for evidence in evidence_list:
            hypothesis_space.add_evidence(evidence)
        
        # Connect evidence to hypothesis
        hypothesis_space.hypotheses["h1"].add_evidence("e1", "supporting")
        hypothesis_space.hypotheses["h1"].add_evidence("e2", "supporting")
        
        print(f"✅ Added {len(hypothesis_space.evidence)} evidence pieces to space")
        print(f"✅ Hypothesis h1 has {len(hypothesis_space.hypotheses['h1'].supporting_evidence)} supporting evidence")
        
        # Test evidence assessment integration
        assessment = EvidenceAssessment(
            evidence_id="integration_test",
            refined_evidence_type="smoking_gun",
            reasoning_for_type="Integration test evidence",
            likelihood_P_E_given_H="High (0.85)",
            likelihood_P_E_given_NotH="Low (0.15)",
            justification_for_likelihoods="Testing evidence assessment integration",
            suggested_numerical_probative_value=0.85
        )
        
        quantifier = EvidenceStrengthQuantifier()
        weights = quantifier.quantify_llm_assessment(assessment)
        
        print(f"✅ Evidence assessment integration successful")
        print(f"   Combined weight: {weights.combined_weight:.3f}")
        print(f"   Reliability weight: {weights.reliability_weight:.3f}")
        print(f"   Credibility weight: {weights.credibility_weight:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all restoration tests."""
    print("🚀 TESTING BAYESIAN EVIDENCE FUNCTIONALITY RESTORATION")
    print("="*60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Imports
    total_tests += 1
    if test_bayesian_imports():
        tests_passed += 1
    
    # Test 2: Evidence creation
    total_tests += 1
    evidence_list = test_bayesian_evidence_creation()
    if evidence_list:
        tests_passed += 1
    else:
        print("💥 Cannot continue without evidence objects")
        return False
    
    # Test 3: Previously disabled methods
    total_tests += 1
    if test_previously_disabled_methods(evidence_list):
        tests_passed += 1
    
    # Test 4: Hypothesis space
    total_tests += 1
    hypothesis_space = test_hypothesis_space()
    if hypothesis_space:
        tests_passed += 1
    else:
        print("💥 Cannot continue without hypothesis space")
        return False
    
    # Test 5: End-to-end integration
    total_tests += 1
    if test_end_to_end_integration(evidence_list, hypothesis_space):
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"🎯 RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 SUCCESS: All BayesianEvidence functionality restored!")
        print("✅ Previously disabled methods now work correctly")
        print("✅ Full evidence analysis capabilities available")
        print("✅ Multi-evidence combination and diversity analysis functional")
        return True
    else:
        print("💥 FAILURE: Some functionality still broken")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)