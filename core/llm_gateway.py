#!/usr/bin/env python3
"""
Central LLM Gateway for all semantic operations.
Enforces LLM-first architecture with zero tolerance for fallbacks.
"""

import os
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from core.llm_required import require_llm, LLMRequiredError
from core.plugins.van_evera_llm_schemas import (
    HypothesisDomainClassification,
    BatchedHypothesisEvaluation,
    HypothesisEvaluationResult
)


# Gateway-specific schemas
@dataclass
class RelationshipAssessment:
    """Assessment of evidence-hypothesis relationship"""
    relationship_type: str  # supports, challenges, neutral
    confidence: float  # 0.0 to 1.0
    reasoning: str
    diagnostic_type: Optional[str] = None


@dataclass
class DomainClassification:
    """Domain classification result"""
    primary_domain: str
    confidence_score: float
    reasoning: str
    secondary_domains: Optional[List[str]] = None


@dataclass
class TemporalEvaluation:
    """Temporal relationship evaluation"""
    temporal_order: str  # before, after, concurrent, unclear
    causal_plausibility: float  # 0.0 to 1.0
    time_gap_assessment: str
    reasoning: str


@dataclass
class VanEveraDiagnostic:
    """Van Evera diagnostic classification"""
    test_type: str  # hoop, smoking_gun, doubly_decisive, straw_in_wind
    passes_test: bool
    confidence: float
    reasoning: str


@dataclass
class CounterfactualAnalysis:
    """Counterfactual analysis result"""
    alternative_scenarios: List[Dict[str, Any]]
    plausibility_scores: List[float]
    key_dependencies: List[str]
    reasoning: str


@dataclass
class EnhancedHypothesis:
    """Enhanced hypothesis with improvements"""
    original: str
    enhanced: str
    improvements: List[str]
    testability_score: float


@dataclass
class EnhancedEvidence:
    """Enhanced evidence with improvements"""
    original: str
    enhanced: str
    clarity_improvements: List[str]
    relevance_score: float


@dataclass
class CausalMechanism:
    """Identified causal mechanism"""
    mechanism_description: str
    intermediate_steps: List[str]
    confidence: float
    evidence_support: List[str]


@dataclass
class BatchEvaluationResult:
    """Batch evaluation results"""
    evaluations: List[RelationshipAssessment]
    cross_hypothesis_insights: Optional[str] = None
    processing_time: Optional[float] = None


class LLMGateway:
    """
    Central gateway for all LLM operations.
    Enforces fail-fast behavior when LLM is unavailable.
    """
    
    def __init__(self):
        """Initialize with required LLM interface"""
        try:
            self.llm = require_llm()  # Fails immediately if LLM unavailable
            # Get the raw query function for direct LLM calls
            from core.plugins.van_evera_llm_interface import create_llm_query_function
            self.llm_query = create_llm_query_function()
        except Exception as e:
            raise LLMRequiredError(f"LLM Gateway cannot initialize without LLM: {e}")
            
        self._cache = {}  # Session-level cache
        self._stats = {
            'calls': 0,
            'failures': 0,
            'cache_hits': 0
        }
    
    def _cache_key(self, method_name: str, **kwargs) -> str:
        """Generate cache key from method and parameters"""
        key_data = {
            'method': method_name,
            'params': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _check_cache(self, key: str) -> Optional[Any]:
        """Check cache for existing result"""
        if key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[key]
        return None
    
    def _store_cache(self, key: str, result: Any) -> None:
        """Store result in cache"""
        self._cache[key] = result
    
    def assess_relationship(
        self,
        evidence: str,
        hypothesis: str,
        context: Optional[str] = None
    ) -> RelationshipAssessment:
        """
        Assess semantic relationship between evidence and hypothesis.
        Returns structured assessment with confidence and reasoning.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('assess_relationship', 
                                    evidence=evidence, 
                                    hypothesis=hypothesis,
                                    context=context)
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            # Call LLM for assessment
            prompt = f"""Assess the relationship between this evidence and hypothesis.

Evidence: {evidence}
Hypothesis: {hypothesis}
{"Context: " + context if context else ""}

Determine:
1. Relationship type (supports/challenges/neutral)
2. Confidence score (0.0-1.0)
3. Reasoning for assessment
4. Van Evera diagnostic type if applicable

Respond in JSON format:
{{
    "relationship_type": "supports|challenges|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "diagnostic_type": "hoop|smoking_gun|doubly_decisive|straw_in_wind|null"
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            result = RelationshipAssessment(
                relationship_type=result_dict['relationship_type'],
                confidence=float(result_dict['confidence']),
                reasoning=result_dict['reasoning'],
                diagnostic_type=result_dict.get('diagnostic_type')
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for relationship assessment: {e}")
    
    def classify_domain(
        self,
        text: str,
        allowed_domains: List[str]
    ) -> DomainClassification:
        """
        Classify text into domain categories.
        Returns primary domain with confidence score.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('classify_domain',
                                    text=text,
                                    allowed_domains=allowed_domains)
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            domains_str = ", ".join(allowed_domains)
            prompt = f"""Classify this text into one of these domains: {domains_str}

Text: {text}

Determine:
1. Primary domain
2. Confidence score (0.0-1.0)
3. Reasoning
4. Any secondary domains that apply

Respond in JSON format:
{{
    "primary_domain": "domain_name",
    "confidence_score": 0.0-1.0,
    "reasoning": "explanation",
    "secondary_domains": ["domain1", "domain2"] or null
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            result = DomainClassification(
                primary_domain=result_dict['primary_domain'],
                confidence_score=float(result_dict['confidence_score']),
                reasoning=result_dict['reasoning'],
                secondary_domains=result_dict.get('secondary_domains')
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for domain classification: {e}")
    
    def evaluate_temporal_relationship(
        self,
        event1: str,
        event2: str,
        temporal_context: Optional[Dict] = None
    ) -> TemporalEvaluation:
        """
        Evaluate temporal relationship and causal plausibility.
        Returns structured temporal analysis.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('evaluate_temporal_relationship',
                                    event1=event1,
                                    event2=event2,
                                    temporal_context=temporal_context)
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            context_str = json.dumps(temporal_context) if temporal_context else "None"
            prompt = f"""Evaluate the temporal relationship between these events.

Event 1: {event1}
Event 2: {event2}
Temporal Context: {context_str}

Determine:
1. Temporal order (before/after/concurrent/unclear)
2. Causal plausibility score (0.0-1.0) - could Event 1 have caused Event 2?
3. Assessment of time gap (immediate/short-term/long-term/too-long)
4. Reasoning

Respond in JSON format:
{{
    "temporal_order": "before|after|concurrent|unclear",
    "causal_plausibility": 0.0-1.0,
    "time_gap_assessment": "assessment",
    "reasoning": "explanation"
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            result = TemporalEvaluation(
                temporal_order=result_dict['temporal_order'],
                causal_plausibility=float(result_dict['causal_plausibility']),
                time_gap_assessment=result_dict['time_gap_assessment'],
                reasoning=result_dict['reasoning']
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for temporal evaluation: {e}")
    
    def determine_diagnostic_type(
        self,
        evidence: str,
        hypothesis: str,
        test_name: str
    ) -> VanEveraDiagnostic:
        """
        Determine Van Evera diagnostic type for evidence.
        Returns test classification with reasoning.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('determine_diagnostic_type',
                                    evidence=evidence,
                                    hypothesis=hypothesis,
                                    test_name=test_name)
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            prompt = f"""Classify this evidence according to Van Evera's diagnostic tests.

Evidence: {evidence}
Hypothesis: {hypothesis}
Test Name: {test_name}

Van Evera Test Types:
- Hoop Test: Evidence necessary but not sufficient (must pass to survive)
- Smoking Gun: Evidence sufficient but not necessary (passing confirms)
- Doubly Decisive: Evidence both necessary and sufficient
- Straw in the Wind: Evidence neither necessary nor sufficient (weak)

Determine:
1. Which test type this represents
2. Whether the evidence passes the test
3. Confidence in classification (0.0-1.0)
4. Reasoning

Respond in JSON format:
{{
    "test_type": "hoop|smoking_gun|doubly_decisive|straw_in_wind",
    "passes_test": true|false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            result = VanEveraDiagnostic(
                test_type=result_dict['test_type'],
                passes_test=bool(result_dict['passes_test']),
                confidence=float(result_dict['confidence']),
                reasoning=result_dict['reasoning']
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for diagnostic classification: {e}")
    
    def calculate_probative_value(
        self,
        evidence: str,
        hypothesis: str,
        diagnostic_type: str
    ) -> float:
        """
        Calculate probative value based on Van Evera methodology.
        Returns value between 0.0 and 1.0.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('calculate_probative_value',
                                    evidence=evidence,
                                    hypothesis=hypothesis,
                                    diagnostic_type=diagnostic_type)
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            prompt = f"""Calculate the probative value of this evidence.

Evidence: {evidence}
Hypothesis: {hypothesis}
Diagnostic Type: {diagnostic_type}

Probative value should reflect:
- How strongly the evidence supports/challenges the hypothesis
- The diagnostic power of the test type
- The quality and specificity of the evidence

Return a single float between 0.0 and 1.0 where:
- 0.0-0.2: Very weak evidence
- 0.2-0.4: Weak evidence
- 0.4-0.6: Moderate evidence
- 0.6-0.8: Strong evidence
- 0.8-1.0: Very strong evidence

Respond with just the number (e.g., "0.75")"""

            response = self.llm_query(prompt)
            result = float(response.strip())
            result = max(0.0, min(1.0, result))  # Ensure in range
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for probative value calculation: {e}")
    
    def batch_evaluate(
        self,
        evidence: str,
        hypotheses: List[Dict[str, str]]
    ) -> BatchEvaluationResult:
        """
        Evaluate one evidence against multiple hypotheses.
        Efficient single LLM call for batch processing.
        Returns evaluations for all hypotheses.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('batch_evaluate',
                                    evidence=evidence,
                                    hypotheses=tuple(h['id'] for h in hypotheses))
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            hypotheses_str = "\n".join([f"{i+1}. [{h['id']}] {h['text']}" 
                                        for i, h in enumerate(hypotheses)])
            
            prompt = f"""Evaluate this evidence against multiple hypotheses in a single analysis.

Evidence: {evidence}

Hypotheses:
{hypotheses_str}

For each hypothesis, determine:
1. Relationship type (supports/challenges/neutral)
2. Confidence score (0.0-1.0)
3. Brief reasoning
4. Van Evera diagnostic type if applicable

Also provide any cross-hypothesis insights.

Respond in JSON format:
{{
    "evaluations": [
        {{
            "hypothesis_id": "id",
            "relationship_type": "supports|challenges|neutral",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation",
            "diagnostic_type": "type or null"
        }}
    ],
    "cross_hypothesis_insights": "insights about relationships between hypotheses"
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            evaluations = []
            for eval_dict in result_dict['evaluations']:
                evaluations.append(RelationshipAssessment(
                    relationship_type=eval_dict['relationship_type'],
                    confidence=float(eval_dict['confidence']),
                    reasoning=eval_dict['reasoning'],
                    diagnostic_type=eval_dict.get('diagnostic_type')
                ))
            
            result = BatchEvaluationResult(
                evaluations=evaluations,
                cross_hypothesis_insights=result_dict.get('cross_hypothesis_insights')
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for batch evaluation: {e}")
    
    def enhance_hypothesis(
        self,
        hypothesis: str,
        evidence_context: List[str]
    ) -> EnhancedHypothesis:
        """
        Enhance hypothesis with additional detail and testability.
        Returns enhanced version with improvements noted.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('enhance_hypothesis',
                                    hypothesis=hypothesis,
                                    evidence_context=tuple(evidence_context))
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            context_str = "\n".join(f"- {e}" for e in evidence_context[:5])  # Limit context
            
            prompt = f"""Enhance this hypothesis to be more specific and testable.

Original Hypothesis: {hypothesis}

Available Evidence Context:
{context_str}

Provide:
1. Enhanced version with more specificity
2. List of improvements made
3. Testability score (0.0-1.0)

Respond in JSON format:
{{
    "enhanced": "enhanced hypothesis text",
    "improvements": ["improvement1", "improvement2"],
    "testability_score": 0.0-1.0
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            result = EnhancedHypothesis(
                original=hypothesis,
                enhanced=result_dict['enhanced'],
                improvements=result_dict['improvements'],
                testability_score=float(result_dict['testability_score'])
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for hypothesis enhancement: {e}")
    
    def identify_causal_mechanism(
        self,
        cause: str,
        effect: str,
        evidence: List[str]
    ) -> CausalMechanism:
        """
        Identify causal mechanism linking cause and effect.
        Returns mechanism description with confidence.
        Raises LLMRequiredError on failure.
        """
        cache_key = self._cache_key('identify_causal_mechanism',
                                    cause=cause,
                                    effect=effect,
                                    evidence=tuple(evidence))
        
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        self._stats['calls'] += 1
        
        try:
            evidence_str = "\n".join(f"- {e}" for e in evidence[:5])  # Limit evidence
            
            prompt = f"""Identify the causal mechanism linking cause to effect.

Cause: {cause}
Effect: {effect}

Supporting Evidence:
{evidence_str}

Identify:
1. The mechanism description
2. Intermediate causal steps
3. Confidence in mechanism (0.0-1.0)
4. Which evidence supports each step

Respond in JSON format:
{{
    "mechanism_description": "description",
    "intermediate_steps": ["step1", "step2", "step3"],
    "confidence": 0.0-1.0,
    "evidence_support": ["evidence for step1", "evidence for step2"]
}}"""

            response = self.llm_query(prompt)
            result_dict = json.loads(response)
            
            result = CausalMechanism(
                mechanism_description=result_dict['mechanism_description'],
                intermediate_steps=result_dict['intermediate_steps'],
                confidence=float(result_dict['confidence']),
                evidence_support=result_dict['evidence_support']
            )
            
            self._store_cache(cache_key, result)
            return result
            
        except Exception as e:
            self._stats['failures'] += 1
            raise LLMRequiredError(f"LLM required for causal mechanism identification: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get gateway statistics"""
        return self._stats.copy()
    
    def clear_cache(self) -> None:
        """Clear the session cache"""
        self._cache.clear()
        self._stats['cache_hits'] = 0