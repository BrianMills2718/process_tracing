#!/usr/bin/env python3
"""
Reality check: Test if the "restored" functionality actually works as claimed.
"""

def test_signature_mismatch():
    """Test if the method signatures actually match what I claimed."""
    print("=== Testing Signature Compatibility ===")
    
    try:
        from core.bayesian_models import BayesianEvidence, EvidenceType, IndependenceType
        from core.evidence_weighting import EvidenceStrengthQuantifier
        
        # Create evidence
        e1 = BayesianEvidence("e1", "test", EvidenceType.SMOKING_GUN)
        e2 = BayesianEvidence("e2", "test", EvidenceType.HOOP)
        evidence_list = [e1, e2]
        
        # Try the wrapper method
        quantifier = EvidenceStrengthQuantifier()
        independence_assumptions = {"e1-e2": IndependenceType.INDEPENDENT}
        
        result = quantifier.combine_multiple_evidence(evidence_list, independence_assumptions)
        print(f"✅ Wrapper method works: {result}")
        
        # But let's see what it actually does with the assumptions
        result_empty = quantifier.combine_multiple_evidence(evidence_list, {})
        print(f"Result with empty assumptions: {result_empty}")
        print(f"Results are the same? {abs(result - result_empty) < 1e-6}")
        
        return True
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

def test_grouping_actually_works():
    """Test if evidence grouping actually groups anything."""
    print("\n=== Testing Evidence Grouping Reality ===")
    
    try:
        from core.bayesian_models import BayesianEvidence, EvidenceType, IndependenceType
        from core.evidence_weighting import EvidenceStrengthQuantifier
        
        # Create evidence that should be grouped
        e1 = BayesianEvidence("e1", "test1", EvidenceType.SMOKING_GUN)
        e2 = BayesianEvidence("e2", "test2", EvidenceType.HOOP)
        e3 = BayesianEvidence("e3", "test3", EvidenceType.SMOKING_GUN)
        evidence_list = [e1, e2, e3]
        
        quantifier = EvidenceStrengthQuantifier()
        
        # Test 1: All independent (should be 3 groups of 1 each)
        independent_assumptions = {
            "e1-e2": IndependenceType.INDEPENDENT,
            "e1-e3": IndependenceType.INDEPENDENT,
            "e2-e3": IndependenceType.INDEPENDENT
        }
        
        groups_independent = quantifier._group_evidence_by_independence(evidence_list, independent_assumptions)
        print(f"Independent assumptions: {len(groups_independent)} groups")
        print(f"Group sizes: {[len(group) for group in groups_independent]}")
        
        # Test 2: Some dependent (should group dependent evidence)
        dependent_assumptions = {
            "e1-e2": IndependenceType.DEPENDENT,
            "e2-e3": IndependenceType.INDEPENDENT
        }
        
        groups_dependent = quantifier._group_evidence_by_independence(evidence_list, dependent_assumptions)
        print(f"Dependent assumptions: {len(groups_dependent)} groups")
        print(f"Group sizes: {[len(group) for group in groups_dependent]}")
        
        # REALITY CHECK: Are the results actually different?
        if groups_independent == groups_dependent:
            print("❌ CRITICAL: Grouping ignores independence assumptions!")
            return False
        else:
            print("✅ Grouping responds to independence assumptions")
            return True
            
    except Exception as e:
        print(f"❌ Grouping test failed: {e}")
        return False

def test_original_vs_new_behavior():
    """Test if new implementation matches what original disabled code expected."""
    print("\n=== Testing Original vs New Behavior ===")
    
    # The original disabled code expected these method signatures:
    # combine_multiple_evidence(evidence_list, independence_assumptions) 
    # where independence_assumptions was REQUIRED, not optional
    
    try:
        from core.bayesian_models import EvidenceStrengthQuantifier as NewQuantifier
        from core.evidence_weighting import EvidenceStrengthQuantifier as WrapperQuantifier
        from core.bayesian_models import BayesianEvidence, EvidenceType, IndependenceType
        
        evidence_list = [
            BayesianEvidence("e1", "test", EvidenceType.SMOKING_GUN),
            BayesianEvidence("e2", "test", EvidenceType.HOOP)
        ]
        
        independence_assumptions = {"e1-e2": IndependenceType.DEPENDENT}
        
        # Test direct call to new implementation
        new_quantifier = NewQuantifier()
        result_new = new_quantifier.combine_multiple_evidence(evidence_list, independence_assumptions)
        print(f"New implementation result: {result_new}")
        
        # Test wrapper call
        wrapper_quantifier = WrapperQuantifier()
        result_wrapper = wrapper_quantifier.combine_multiple_evidence(evidence_list, independence_assumptions)
        print(f"Wrapper result: {result_wrapper}")
        
        print(f"Results match? {abs(result_new - result_wrapper) < 1e-6}")
        
        return True
    except Exception as e:
        print(f"❌ Behavior comparison failed: {e}")
        return False

def main():
    """Run reality check tests."""
    print("🔍 REALITY CHECK: Testing if BayesianEvidence restoration actually works as claimed")
    print("="*80)
    
    tests = [
        test_signature_mismatch,
        test_grouping_actually_works, 
        test_original_vs_new_behavior
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"🎯 REALITY CHECK RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ Implementation works as claimed")
    else:
        print("❌ SIGNIFICANT GAPS between claims and reality")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)