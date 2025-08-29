"""
Centralized Semantic Analysis Service for LLM-First Architecture.

This service provides a unified interface for all semantic analysis operations,
replacing rule-based keyword matching with LLM understanding throughout the system.

Key Features:
- Session-level caching to reduce redundant LLM calls
- Batch processing for improved performance
- Structured error handling with graceful fallbacks
- Universal domain classification without dataset-specific logic
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib
import json

from core.plugins.van_evera_llm_interface import get_van_evera_llm, VanEveraLLMInterface
from core.semantic_signature import SemanticSignature
from core.plugins.van_evera_llm_schemas import (
    HypothesisDomainClassification,
    ProbativeValueAssessment,
    AlternativeHypothesisGeneration,
    TestGenerationSpecification,
    ComprehensiveEvidenceAnalysis,
    MultiFeatureExtraction
)

logger = logging.getLogger(__name__)


class SemanticAnalysisService:
    """
    Centralized service for all semantic analysis operations.
    Provides caching, batching, and error handling for LLM calls.
    """
    
    def __init__(self, cache_ttl_minutes: int = 60):
        """
        Initialize the semantic analysis service with multi-layer cache.
        
        Args:
            cache_ttl_minutes: Cache time-to-live in minutes (for L1 cache)
        """
        self.llm_interface = get_van_evera_llm()
        self.signature_generator = SemanticSignature()
        
        # Multi-layer cache system
        self._l1_cache: Dict[str, Tuple[Any, datetime]] = {}  # Exact match cache
        self._l2_cache: Dict[str, Tuple[Any, datetime]] = {}  # Semantic signature cache
        self._l3_cache: Dict[str, Tuple[Any, datetime]] = {}  # Partial results cache
        
        # Different TTLs for each cache layer
        self.l1_ttl = timedelta(minutes=cache_ttl_minutes)
        self.l2_ttl = timedelta(minutes=cache_ttl_minutes * 2)  # 2x longer for semantic
        self.l3_ttl = timedelta(minutes=cache_ttl_minutes * 4)  # 4x longer for partial
        
        # Legacy compatibility
        self.cache_ttl = self.l1_ttl
        self._cache = self._l1_cache  # Point to L1 for backward compatibility
        
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'llm_calls': 0,
            'errors': 0
        }
        
    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate a unique cache key for an operation."""
        key_data = {
            'operation': operation,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def _check_cache(self, key: str) -> Optional[Any]:
        """Check if a cached result exists and is still valid (L1 only for compatibility)."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                self._stats['cache_hits'] += 1
                self._stats['l1_hits'] += 1
                logger.debug(f"L1 cache hit for key: {key}")
                return result
            else:
                del self._cache[key]
        self._stats['cache_misses'] += 1
        return None
    
    def _check_all_caches(self, evidence: str, hypothesis: str, operation: str) -> Optional[Any]:
        """
        Check all cache layers for a hit.
        
        Args:
            evidence: Evidence text
            hypothesis: Hypothesis text
            operation: Type of operation
            
        Returns:
            Cached result if found, None otherwise
        """
        # L1: Exact match cache
        l1_key = self._get_cache_key(operation, evidence, hypothesis)
        if l1_key in self._l1_cache:
            result, timestamp = self._l1_cache[l1_key]
            if datetime.now() - timestamp < self.l1_ttl:
                self._stats['l1_hits'] += 1
                self._stats['cache_hits'] += 1
                logger.debug(f"L1 cache hit for {operation}")
                return result
            else:
                del self._l1_cache[l1_key]
        
        # L2: Semantic signature cache
        l2_key = self.signature_generator.generate_signature(evidence, hypothesis, operation)
        if l2_key in self._l2_cache:
            result, timestamp = self._l2_cache[l2_key]
            if datetime.now() - timestamp < self.l2_ttl:
                self._stats['l2_hits'] += 1
                self._stats['cache_hits'] += 1
                logger.debug(f"L2 semantic cache hit for {operation}")
                # Promote to L1 for next time
                self._l1_cache[l1_key] = (result, datetime.now())
                return result
            else:
                del self._l2_cache[l2_key]
        
        # L3: Check for partial results that might be reusable
        # (This would require more complex logic to determine partial matches)
        # For now, we'll skip L3 implementation
        
        self._stats['cache_misses'] += 1
        return None
    
    def _update_all_caches(self, evidence: str, hypothesis: str, operation: str, result: Any) -> None:
        """
        Update all cache layers with new result.
        
        Args:
            evidence: Evidence text
            hypothesis: Hypothesis text
            operation: Type of operation
            result: Result to cache
        """
        now = datetime.now()
        
        # Update L1 (exact match)
        l1_key = self._get_cache_key(operation, evidence, hypothesis)
        self._l1_cache[l1_key] = (result, now)
        
        # Update L2 (semantic signature)
        l2_key = self.signature_generator.generate_signature(evidence, hypothesis, operation)
        self._l2_cache[l2_key] = (result, now)
        
        # L3 would store partial results for future use
        
    def _update_cache(self, key: str, result: Any) -> None:
        """Update the cache with a new result."""
        self._cache[key] = (result, datetime.now())
        
    def classify_domain(self, 
                       hypothesis_description: str,
                       context: Optional[str] = None) -> HypothesisDomainClassification:
        """
        Classify hypothesis domain using semantic understanding.
        
        Args:
            hypothesis_description: The hypothesis text to classify
            context: Optional context information
            
        Returns:
            Domain classification with confidence and reasoning
        """
        cache_key = self._get_cache_key('classify_domain', hypothesis_description, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            result = self.llm_interface.classify_hypothesis_domain(
                hypothesis_description=hypothesis_description,
                context=context
            )
            self._update_cache(cache_key, result)
            self._stats['llm_calls'] += 1
            return result
        except Exception as e:
            logger.error(f"Domain classification failed: {e}")
            self._stats['errors'] += 1
            # Return conservative fallback
            return HypothesisDomainClassification(
                primary_domain="political",  # Most common domain
                secondary_domains=[],
                confidence_score=0.5,
                reasoning="Fallback classification due to LLM error",
                generalizability="Limited due to error condition"
            )
            
    def assess_probative_value(self,
                              evidence_description: str,
                              hypothesis_description: str,
                              context: Optional[str] = None) -> ProbativeValueAssessment:
        """
        Assess probative value of evidence for hypothesis.
        
        Args:
            evidence_description: The evidence text
            hypothesis_description: The hypothesis text
            context: Optional context information
            
        Returns:
            Probative value assessment with reasoning
        """
        cache_key = self._get_cache_key('probative_value', evidence_description, 
                                        hypothesis_description, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            result = self.llm_interface.assess_probative_value(
                evidence_description=evidence_description,
                hypothesis_description=hypothesis_description,
                context=context
            )
            self._update_cache(cache_key, result)
            self._stats['llm_calls'] += 1
            return result
        except Exception as e:
            logger.error(f"Probative value assessment failed: {e}")
            self._stats['errors'] += 1
            # Return conservative fallback
            return ProbativeValueAssessment(
                probative_value=0.5,  # Neutral value
                confidence_score=0.5,
                reasoning="Fallback assessment due to LLM error",
                evidence_quality_factors=["Error condition"],
                reliability_assessment="Unknown due to error",
                van_evera_implications="Limited diagnostic value"
            )
            
    def detect_contradiction(self,
                           evidence_description: str,
                           hypothesis_description: str):
        """
        Detect contradictions between evidence and hypothesis.
        
        Args:
            evidence_description: The evidence text
            hypothesis_description: The hypothesis text
            
        Returns:
            Contradiction analysis with semantic reasoning
        """
        cache_key = self._get_cache_key('contradiction', evidence_description, 
                                        hypothesis_description)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            result = self.llm_interface.analyze_evidence_contradiction(
                evidence_description=evidence_description,
                hypothesis_description=hypothesis_description
            )
            self._update_cache(cache_key, result)
            self._stats['llm_calls'] += 1
            return result
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            self._stats['errors'] += 1
            # Return conservative fallback as a dict structure
            return {
                'contradicts_hypothesis': False,  # Conservative: assume no contradiction
                'confidence_score': 0.5,
                'semantic_reasoning': "Fallback analysis due to LLM error"
            }
            
    def generate_alternatives(self,
                            original_hypothesis: str,
                            evidence_context: str,
                            num_alternatives: int = 3) -> AlternativeHypothesisGeneration:
        """
        Generate alternative hypotheses based on evidence.
        
        Args:
            original_hypothesis: The original hypothesis
            evidence_context: Evidence context for generation
            num_alternatives: Number of alternatives to generate
            
        Returns:
            Generated alternative hypotheses
        """
        cache_key = self._get_cache_key('alternatives', original_hypothesis,
                                        evidence_context, num_alternatives)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            result = self.llm_interface.generate_alternative_hypotheses(
                original_hypothesis=original_hypothesis,
                evidence_context=evidence_context,
                num_alternatives=num_alternatives
            )
            self._update_cache(cache_key, result)
            self._stats['llm_calls'] += 1
            return result
        except Exception as e:
            logger.error(f"Alternative generation failed: {e}")
            self._stats['errors'] += 1
            # Return minimal fallback
            return AlternativeHypothesisGeneration(
                alternative_hypotheses=[],
                generation_confidence=0.0,
                universal_applicability="Error prevented generation"
            )
            
    def generate_diagnostic_tests(self,
                                 hypothesis_description: str,
                                 domain: str) -> TestGenerationSpecification:
        """
        Generate Van Evera diagnostic tests for hypothesis.
        
        Args:
            hypothesis_description: The hypothesis to test
            domain: The hypothesis domain
            
        Returns:
            Generated diagnostic test specifications
        """
        cache_key = self._get_cache_key('tests', hypothesis_description, domain)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            result = self.llm_interface.generate_van_evera_tests(
                hypothesis_description=hypothesis_description,
                domain_classification=domain
            )
            self._update_cache(cache_key, result)
            self._stats['llm_calls'] += 1
            return result
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            self._stats['errors'] += 1
            # Return minimal fallback
            return TestGenerationSpecification(
                test_predictions=[],
                generation_reasoning="Error prevented test generation",
                universal_validity="Limited due to error"
            )
            
    def batch_classify_domains(self,
                              hypothesis_descriptions: List[str]) -> List[HypothesisDomainClassification]:
        """
        Batch classify multiple hypotheses for efficiency.
        
        Args:
            hypothesis_descriptions: List of hypotheses to classify
            
        Returns:
            List of domain classifications
        """
        results = []
        for hypothesis in hypothesis_descriptions:
            result = self.classify_domain(hypothesis)
            results.append(result)
        return results
        
    def batch_assess_probative_values(self,
                                     evidence_hypothesis_pairs: List[Tuple[str, str]]) -> List[ProbativeValueAssessment]:
        """
        Batch assess probative values for multiple evidence-hypothesis pairs.
        
        Args:
            evidence_hypothesis_pairs: List of (evidence, hypothesis) tuples
            
        Returns:
            List of probative value assessments
        """
        results = []
        for evidence, hypothesis in evidence_hypothesis_pairs:
            result = self.assess_probative_value(evidence, hypothesis)
            results.append(result)
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics for monitoring."""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = (self._stats['cache_hits'] / total_requests * 100 
                         if total_requests > 0 else 0)
        
        return {
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'llm_calls': self._stats['llm_calls'],
            'errors': self._stats['errors'],
            'cache_size': len(self._cache)
        }
        
    def analyze_comprehensive(self, 
                            evidence: str, 
                            hypothesis: str,
                            context: Optional[str] = None) -> ComprehensiveEvidenceAnalysis:
        """
        Perform comprehensive evidence analysis in a single LLM call.
        Replaces multiple separate calls with one coherent analysis.
        Uses multi-layer caching for improved performance.
        
        Args:
            evidence: Evidence description
            hypothesis: Hypothesis description
            context: Optional context information
            
        Returns:
            Comprehensive analysis with all semantic features
        """
        # Check all cache layers
        cached_result = self._check_all_caches(evidence, hypothesis, 'analyze_comprehensive')
        if cached_result:
            return cached_result
            
        try:
            self._stats['llm_calls'] += 1
            result = self.llm_interface.analyze_evidence_comprehensive(
                evidence, hypothesis, context
            )
            # Update all cache layers
            self._update_all_caches(evidence, hypothesis, 'analyze_comprehensive', result)
            logger.info("Comprehensive analysis completed successfully")
            return result
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Comprehensive analysis failed: {e}")
            # Return a conservative fallback
            fallback = ComprehensiveEvidenceAnalysis(
                primary_domain="political",
                secondary_domains=[],
                domain_confidence=0.3,
                domain_reasoning="Error in analysis - conservative fallback",
                probative_value=0.5,
                probative_factors=["Unable to assess"],
                evidence_quality="medium",
                reliability_score=0.5,
                relationship_type="neutral",
                relationship_confidence=0.3,
                relationship_reasoning="Error in analysis - conservative fallback",
                van_evera_diagnostic="straw_in_wind",
                causal_mechanisms=[],
                temporal_markers=[],
                actor_relationships=[],
                key_concepts=[],
                contextual_factors=[],
                alternative_interpretations=[],
                confidence_overall=0.3
            )
            return fallback
    
    def extract_all_features(self, 
                           text: str,
                           context: Optional[str] = None) -> MultiFeatureExtraction:
        """
        Extract all semantic features from text in one LLM call.
        Captures relationships between features.
        
        Args:
            text: Text to analyze
            context: Optional context information
            
        Returns:
            Multi-feature extraction with relationships
        """
        cache_key = self._get_cache_key('extract_all_features', text, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            self._stats['llm_calls'] += 1
            result = self.llm_interface.extract_all_features(text, context)
            self._update_cache(cache_key, result)
            logger.info("Feature extraction completed successfully")
            return result
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Feature extraction failed: {e}")
            # Return conservative fallback
            fallback = MultiFeatureExtraction(
                mechanisms=[],
                causal_chains=[],
                primary_actors=[],
                actor_relationships=[],
                temporal_sequence=[],
                duration_estimates={},
                key_concepts=[],
                domain_indicators=[],
                theoretical_frameworks=[],
                geographic_context=[],
                institutional_context=[],
                cultural_context=[],
                actor_mechanism_links=[],
                temporal_causal_links=[]
            )
            return fallback
    
    def clear_cache(self) -> None:
        """Clear the cache (useful for testing)."""
        self._cache.clear()
        logger.info("Semantic analysis cache cleared")


# Global service instance
_semantic_service: Optional[SemanticAnalysisService] = None


def get_semantic_service() -> SemanticAnalysisService:
    """
    Get the global semantic analysis service instance.
    
    Returns:
        The semantic analysis service
    """
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticAnalysisService()
        logger.info("Semantic analysis service initialized")
    return _semantic_service