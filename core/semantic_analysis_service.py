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
from core.llm_required import require_llm, LLMRequiredError
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
        Initialize the semantic analysis service with simple caching.
        
        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        # Ensure LLM is available - NO FALLBACKS
        self.llm_interface = require_llm()
        
        # Simple cache system - exact matches only
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
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
        """Check if a cached result exists and is still valid."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                self._stats['cache_hits'] += 1
                logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                del self._cache[key]
        self._stats['cache_misses'] += 1
        return None
        
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
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for domain classification: {e}") from e
            
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
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for probative value assessment: {e}") from e
            
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
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for contradiction detection: {e}") from e
            
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
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for alternative generation: {e}") from e
            
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
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for test generation: {e}") from e
            
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
        
        Args:
            evidence: Evidence description
            hypothesis: Hypothesis description
            context: Optional context information
            
        Returns:
            Comprehensive analysis with all semantic features
        """
        cache_key = self._get_cache_key('analyze_comprehensive', evidence, hypothesis, context)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        try:
            self._stats['llm_calls'] += 1
            result = self.llm_interface.analyze_evidence_comprehensive(
                evidence, hypothesis, context
            )
            self._update_cache(cache_key, result)
            logger.info("Comprehensive analysis completed successfully")
            return result
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Comprehensive analysis failed: {e}")
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for comprehensive analysis: {e}") from e
    
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
            # NO FALLBACK - LLM is required
            raise LLMRequiredError(f"LLM required for feature extraction: {e}") from e
    
    def evaluate_evidence_against_hypotheses_batch(self,
                                                  evidence_id: str,
                                                  evidence_text: str,
                                                  hypotheses: List[Dict[str, str]],
                                                  context: Optional[str] = None):
        """
        Evaluate evidence against multiple hypotheses in a single LLM call.
        This is the TRUE LLM-first approach - no keyword matching!
        
        Args:
            evidence_id: Unique identifier for the evidence
            evidence_text: The evidence text
            hypotheses: List of dicts with 'id' and 'text' keys
            context: Optional context
            
        Returns:
            BatchedHypothesisEvaluation with all relationships
        """
        # Create cache key for the batch
        hypothesis_ids = "-".join(sorted([h['id'] for h in hypotheses]))
        cache_key = self._get_cache_key(
            'batch_eval',
            f"{evidence_id}|{hypothesis_ids}",
            context
        )
        
        # Check cache
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            import time
            print(f"[SEMANTIC-LLM] Starting batch evaluation: evidence={evidence_id}, {len(hypotheses)} hypotheses")
            print(f"[SEMANTIC-LLM] Evidence text length: {len(evidence_text)} chars")
            print(f"[SEMANTIC-LLM] Hypotheses: {[h.get('id', 'unknown') for h in hypotheses]}")
            
            llm_start = time.time()
            self._stats['llm_calls'] += 1
            result = self.llm_interface.evaluate_evidence_against_hypotheses(
                evidence_id,
                evidence_text,
                hypotheses,
                context
            )
            llm_duration = time.time() - llm_start
            print(f"[SEMANTIC-LLM] Batch evaluation completed in {llm_duration:.1f}s")
            
            self._update_cache(cache_key, result)
            logger.info(f"Batch evaluation completed for {len(hypotheses)} hypotheses")
            return result
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Batch evaluation failed: {e}")
            raise
    
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