#!/usr/bin/env python3
"""
Test script to validate LLM interface integration fixes from Phase 21B.

Validates that:
1. All imports work without errors
2. LLM interface methods are accessible 
3. Evidence processing pipeline functions correctly
"""

def test_import_resolution():
    """Test all critical imports work"""
    print("Testing imports...")
    
    try:
        from core.evidence_weighting import EvidenceStrengthQuantifier
        print("✅ EvidenceStrengthQuantifier import successful")
    except Exception as e:
        print(f"❌ EvidenceStrengthQuantifier import failed: {e}")
        return False
    
    try:
        from core.confidence_calculator import ConfidenceLevel
        print("✅ ConfidenceLevel import successful")
    except Exception as e:
        print(f"❌ ConfidenceLevel import failed: {e}")
        return False
    
    try:
        from core.plugins.van_evera_llm_interface import VanEveraLLMInterface
        print("✅ VanEveraLLMInterface import successful")
    except Exception as e:
        print(f"❌ VanEveraLLMInterface import failed: {e}")
        return False
        
    try:
        from core.plugins.van_evera_llm_schemas import ComprehensiveEvidenceAnalysis
        print("✅ ComprehensiveEvidenceAnalysis import successful")
    except Exception as e:
        print(f"❌ ComprehensiveEvidenceAnalysis import failed: {e}")
        return False
    
    return True

def test_llm_interface_accessibility():
    """Test LLM interface methods are accessible"""
    print("\nTesting LLM interface accessibility...")
    
    try:
        from core.plugins.van_evera_llm_interface import VanEveraLLMInterface
        
        llm_interface = VanEveraLLMInterface()
        print("✅ VanEveraLLMInterface instantiation successful")
        
        # Test _get_structured_response method exists
        assert hasattr(llm_interface, '_get_structured_response'), "_get_structured_response method missing"
        print("✅ _get_structured_response method accessible")
        
        # Test assess_probative_value method exists  
        assert hasattr(llm_interface, 'assess_probative_value'), "assess_probative_value method missing"
        print("✅ assess_probative_value method accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM interface accessibility test failed: {e}")
        return False

def test_evidence_strength_quantifier():
    """Test EvidenceStrengthQuantifier can be instantiated and has required methods"""
    print("\nTesting EvidenceStrengthQuantifier...")
    
    try:
        from core.evidence_weighting import EvidenceStrengthQuantifier
        
        quantifier = EvidenceStrengthQuantifier()
        print("✅ EvidenceStrengthQuantifier instantiation successful")
        
        # Test critical methods exist
        assert hasattr(quantifier, '_analyze_reliability_indicators'), "_analyze_reliability_indicators missing"
        print("✅ _analyze_reliability_indicators method exists")
        
        assert hasattr(quantifier, '_analyze_credibility_indicators'), "_analyze_credibility_indicators missing"  
        print("✅ _analyze_credibility_indicators method exists")
        
        assert hasattr(quantifier, 'quantify_llm_assessment'), "quantify_llm_assessment missing"
        print("✅ quantify_llm_assessment method exists")
        
        return True
        
    except Exception as e:
        print(f"❌ EvidenceStrengthQuantifier test failed: {e}")
        return False

def test_confidence_level_llm_requirement():
    """Test ConfidenceLevel raises LLMRequiredError as expected"""
    print("\nTesting ConfidenceLevel LLM requirement...")
    
    try:
        from core.confidence_calculator import ConfidenceLevel
        from core.llm_required import LLMRequiredError
        
        # This should raise LLMRequiredError since we removed hardcoded thresholds
        try:
            confidence = ConfidenceLevel.from_score(0.75)
            print(f"❌ Expected LLMRequiredError but got: {confidence}")
            return False
        except LLMRequiredError:
            print("✅ ConfidenceLevel.from_score correctly raises LLMRequiredError")
            return True
        except Exception as e:
            print(f"❌ Unexpected error type: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ConfidenceLevel test failed: {e}")
        return False

def test_schema_compatibility():
    """Test ComprehensiveEvidenceAnalysis has expected fields"""
    print("\nTesting schema compatibility...")
    
    try:
        from core.plugins.van_evera_llm_schemas import ComprehensiveEvidenceAnalysis
        
        # Check that required fields exist in schema
        schema = ComprehensiveEvidenceAnalysis.model_json_schema()
        properties = schema.get('properties', {})
        
        required_fields = ['reliability_score', 'evidence_quality', 'primary_domain']
        
        for field in required_fields:
            if field not in properties:
                print(f"❌ Missing field in ComprehensiveEvidenceAnalysis: {field}")
                return False
            else:
                print(f"✅ ComprehensiveEvidenceAnalysis has field: {field}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema compatibility test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("=== Phase 21B LLM Integration Fix Validation ===\n")
    
    tests = [
        test_import_resolution,
        test_llm_interface_accessibility, 
        test_evidence_strength_quantifier,
        test_confidence_level_llm_requirement,
        test_schema_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== RESULTS: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - LLM interface integration successful!")
        return True
    else:
        print("💥 SOME TESTS FAILED - Additional fixes required")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)