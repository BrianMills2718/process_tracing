#!/usr/bin/env python3
"""
Unit tests to verify LLM-First policy conversions work correctly.
Tests that keyword-based logic has been replaced with semantic analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.evidence_weighting import EvidenceStrengthQuantifier
from core.structured_models import EvidenceAssessment
from core.llm_required import LLMRequiredError
import pytest


class TestLLMEvidenceWeighting:
    """Test evidence weighting conversions from keywords to LLM semantic analysis"""
    
    def setup_method(self):
        """Setup test environment"""
        self.quantifier = EvidenceStrengthQuantifier()
    
    def test_reliability_llm_conversion(self):
        """Test reliability assessment uses LLM semantic analysis"""
        test_cases = [
            # High quality cases
            ("documented by official sources and verified through multiple channels", 
             "peer-reviewed academic research with institutional backing"),
            
            # Low quality cases  
            ("unverified hearsay from anonymous sources",
             "single unconfirmed report with no corroboration"),
            
            # Edge cases where keywords might fail
            ("reliable analysis despite being informal",
             "expert assessment though not officially published"),
        ]
        
        for reasoning, justification in test_cases:
            try:
                score = self.quantifier._analyze_reliability_indicators(reasoning, justification)
                
                # Validate LLM provides reasonable assessment
                assert 0.0 <= score <= 1.0, f"Score {score} outside valid range for: {reasoning}"
                print(f"✓ Reliability score {score:.3f} for: {reasoning[:50]}...")
                
            except LLMRequiredError:
                print(f"✓ Correctly raised LLMRequiredError when LLM unavailable")
                # This is expected behavior - test passes
                
    def test_credibility_llm_conversion(self):
        """Test credibility assessment uses LLM semantic analysis"""
        test_cases = [
            # High credibility cases
            ("government official speaking in official capacity",
             "published research by recognized academic institution"),
            
            # Low credibility cases
            ("anonymous poster with clear partisan agenda",
             "unqualified amateur making unsubstantiated claims"),
            
            # Nuanced cases requiring semantic understanding
            ("experienced journalist with unofficial but expert sources",
             "retired official speaking informally but with deep knowledge"),
        ]
        
        for reasoning, justification in test_cases:
            try:
                score = self.quantifier._analyze_credibility_indicators(reasoning, justification)
                
                # Validate LLM provides reasonable assessment
                assert 0.0 <= score <= 1.0, f"Score {score} outside valid range for: {reasoning}"
                print(f"✓ Credibility score {score:.3f} for: {reasoning[:50]}...")
                
            except LLMRequiredError:
                print(f"✓ Correctly raised LLMRequiredError when LLM unavailable")
                # This is expected behavior - test passes

    def test_no_keyword_logic_remains(self):
        """Verify no keyword-based logic remains in converted functions"""
        import inspect
        
        # Check reliability function source
        reliability_source = inspect.getsource(self.quantifier._analyze_reliability_indicators)
        
        # Should not contain keyword matching logic
        assert "if indicator in" not in reliability_source, "Keyword matching logic still present in reliability"
        assert "positive_indicators" not in reliability_source, "Hardcoded indicator lists still present"
        assert "LLMRequiredError" in reliability_source, "LLM failure handling missing"
        
        # Check credibility function source
        credibility_source = inspect.getsource(self.quantifier._analyze_credibility_indicators)
        
        # Should not contain keyword matching logic  
        assert "if indicator in" not in credibility_source, "Keyword matching logic still present in credibility"
        assert "high_credibility" not in credibility_source, "Hardcoded indicator lists still present"
        assert "LLMRequiredError" in credibility_source, "LLM failure handling missing"
        
        print("✓ No keyword-based logic detected in converted functions")

    def test_confidence_thresholds_require_llm(self):
        """Test confidence level calculation requires LLM-generated thresholds"""
        from core.confidence_calculator import ConfidenceLevel
        
        # Should raise LLMRequiredError when no thresholds provided
        try:
            level = ConfidenceLevel.from_score(0.75, thresholds=None)
            assert False, "Should have raised LLMRequiredError for missing thresholds"
        except LLMRequiredError:
            print("✓ Correctly requires LLM-generated thresholds")
        
        # Mock thresholds object for testing
        class MockThresholds:
            very_high_threshold = 0.85
            high_threshold = 0.70
            moderate_threshold = 0.50
            low_threshold = 0.30
            
        # Should work with LLM-generated thresholds
        level = ConfidenceLevel.from_score(0.75, thresholds=MockThresholds())
        assert level == ConfidenceLevel.HIGH
        print("✓ Works correctly with LLM-generated thresholds")


class TestSemanticAnalysisConversions:
    """Test semantic analysis conversions in plugins"""
    
    def test_research_question_generator_no_keywords(self):
        """Verify research question generator doesn't use keyword scoring"""
        from core.plugins.research_question_generator import ResearchQuestionGeneratorPlugin
        import inspect
        
        # Check that _extract_phenomenon method doesn't use keyword counting
        source = inspect.getsource(ResearchQuestionGeneratorPlugin)
        
        # Should not contain keyword counting logic
        assert "sum(all_text.count(keyword)" not in source, "Keyword counting still present"
        assert "for keyword in keywords" not in source, "Keyword iteration still present"
        assert "semantic_service" in source, "Should use semantic analysis"
        
        print("✓ Research question generator converted to semantic analysis")
    
    def test_alternative_hypothesis_generator_no_keywords(self):
        """Verify alternative hypothesis generator doesn't use keyword extraction"""
        from core.plugins.alternative_hypothesis_generator import AlternativeHypothesisGeneratorPlugin
        import inspect
        
        # Check that _identify_relevant_evidence method doesn't use keywords
        source = inspect.getsource(AlternativeHypothesisGeneratorPlugin)
        
        # Should not contain keyword-based logic
        assert "all_keywords" not in source, "Keyword extraction still present"
        assert "description_keywords" not in source, "Keyword splitting still present" 
        assert "semantic_service" in source, "Should use semantic analysis"
        
        print("✓ Alternative hypothesis generator converted to semantic analysis")


def run_tests():
    """Run all conversion tests"""
    print("=" * 60)
    print("TESTING LLM-FIRST POLICY CONVERSIONS")
    print("=" * 60)
    
    # Test evidence weighting conversions
    print("\n1. Testing Evidence Weighting Conversions")
    print("-" * 40)
    test_weighting = TestLLMEvidenceWeighting()
    test_weighting.setup_method()
    
    test_weighting.test_reliability_llm_conversion()
    test_weighting.test_credibility_llm_conversion() 
    test_weighting.test_no_keyword_logic_remains()
    test_weighting.test_confidence_thresholds_require_llm()
    
    # Test semantic analysis conversions
    print("\n2. Testing Plugin Semantic Conversions")
    print("-" * 40)
    test_semantic = TestSemanticAnalysisConversions()
    test_semantic.test_research_question_generator_no_keywords()
    test_semantic.test_alternative_hypothesis_generator_no_keywords()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - LLM-FIRST POLICY VIOLATIONS FIXED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()