"""
Causal Confidence Quantification for Bayesian Process Tracing.

Implements sophisticated confidence assessment for causal hypotheses based on
Bayesian posteriors, evidence quality, coherence analysis, and uncertainty
quantification. Provides multi-dimensional confidence scoring with interpretable
metrics for decision-making.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import networkx as nx

from .plugins.bayesian_van_evera_engine import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace, 
    BayesianProcessTracingModel, EvidenceType
)
from .evidence_weighting import EvidenceStrengthQuantifier, EvidenceWeights


class ConfidenceType(Enum):
    """Types of confidence assessment."""
    OVERALL = "overall"                    # Overall confidence in hypothesis
    CAUSAL = "causal"                     # Confidence in causal relationship
    EVIDENTIAL = "evidential"             # Confidence based on evidence quality
    COHERENCE = "coherence"               # Confidence based on logical coherence
    ROBUSTNESS = "robustness"             # Confidence in robustness to assumptions
    SENSITIVITY = "sensitivity"           # Sensitivity to evidence changes


class ConfidenceLevel(Enum):
    """Standardized confidence levels with dynamic thresholds."""
    # Default thresholds - will be overridden by LLM assessment
    VERY_HIGH = ("very_high", 0.85, 1.00)     # Dynamic upper tier
    HIGH = ("high", 0.70, 0.85)               # Dynamic high tier
    MODERATE = ("moderate", 0.50, 0.70)       # Dynamic moderate tier
    LOW = ("low", 0.30, 0.50)                 # Dynamic low tier
    VERY_LOW = ("very_low", 0.00, 0.30)       # Dynamic lower tier
    
    def __init__(self, label: str, min_threshold: float, max_threshold: float):
        self.label = label
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    @classmethod
    def from_score(cls, score: float, thresholds=None) -> 'ConfidenceLevel':
        """Get confidence level from numerical score with LLM-generated thresholds."""
        from .llm_required import LLMRequiredError
        
        if not thresholds:
            raise LLMRequiredError("Confidence thresholds must be LLM-generated - no hardcoded fallbacks allowed")
        
        # Use only LLM-generated thresholds
        if score >= thresholds.very_high_threshold:
            return cls.VERY_HIGH
        elif score >= thresholds.high_threshold:
            return cls.HIGH
        elif score >= thresholds.moderate_threshold:
            return cls.MODERATE
        elif score >= thresholds.low_threshold:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class ConfidenceAssessment:
    """Comprehensive confidence assessment for a hypothesis."""
    hypothesis_id: str
    overall_confidence: float
    confidence_components: Dict[ConfidenceType, float]
    confidence_level: ConfidenceLevel
    
    # Detailed metrics
    evidence_quality_score: float = 0.0
    logical_coherence_score: float = 0.0
    robustness_score: float = 0.0
    sensitivity_score: float = 0.0
    
    # Supporting information
    evidence_count: int = 0
    contradicting_evidence_count: int = 0
    evidence_diversity_score: float = 0.0
    posterior_probability: float = 0.0
    prior_probability: float = 0.0
    
    # Uncertainty measures
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    uncertainty_sources: List[str] = field(default_factory=list)
    
    # Metadata
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    assessment_method: str = "multi_dimensional"
    
    def get_interpretation(self) -> str:
        """Get human-readable interpretation of confidence."""
        interpretations = {
            ConfidenceLevel.VERY_HIGH: "Very high confidence - strong evidential support with minimal uncertainty",
            ConfidenceLevel.HIGH: "High confidence - good evidential support with manageable uncertainty", 
            ConfidenceLevel.MODERATE: "Moderate confidence - adequate evidence but notable uncertainty remains",
            ConfidenceLevel.LOW: "Low confidence - limited evidence or significant uncertainty",
            ConfidenceLevel.VERY_LOW: "Very low confidence - insufficient evidence or high uncertainty"
        }
        return interpretations.get(self.confidence_level, "Unknown confidence level")
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on confidence assessment."""
        recommendations = []
        
        if self.confidence_level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            recommendations.append("Hypothesis well-supported - suitable for decision-making")
            if self.evidence_diversity_score < 0.6:
                recommendations.append("Consider seeking additional diverse evidence sources")
        
        elif self.confidence_level == ConfidenceLevel.MODERATE:
            recommendations.append("Hypothesis moderately supported - additional evidence recommended")
            if self.evidence_quality_score < 0.6:
                recommendations.append("Focus on improving evidence quality")
            if self.logical_coherence_score < 0.6:
                recommendations.append("Review logical consistency of causal mechanism")
        
        else:  # LOW or VERY_LOW
            recommendations.append("Hypothesis insufficiently supported - substantial additional evidence needed")
            if self.evidence_count < 3:
                recommendations.append("Seek additional supporting evidence")
            if self.contradicting_evidence_count > self.evidence_count:
                recommendations.append("Address contradicting evidence")
        
        return recommendations


class CausalConfidenceCalculator:
    """
    Calculates multi-dimensional confidence assessments for causal hypotheses.
    
    Implements sophisticated confidence quantification based on:
    - Bayesian posterior probabilities
    - Evidence quality and diversity
    - Logical coherence analysis  
    - Robustness to assumptions
    - Sensitivity analysis
    """
    
    def __init__(self):
        # LLM is REQUIRED - fail fast if unavailable
        from .llm_required import require_llm
        self.llm = require_llm()  # Will raise LLMRequiredError if unavailable
        
        self.evidence_quantifier = EvidenceStrengthQuantifier()
        self.assessment_history: List[ConfidenceAssessment] = []
        self._confidence_thresholds = None  # Cache for LLM-generated thresholds
        self._causal_assessment = None  # Cache for causal mechanism assessment
    
    def _get_llm_interface(self):
        """Return the required LLM interface"""
        # LLM is already initialized in __init__ and required
        return self.llm
    
    def _update_confidence_thresholds(self, hypothesis: BayesianHypothesis, 
                                     evidence_list: List[BayesianEvidence],
                                     hypothesis_space: BayesianHypothesisSpace):
        """Update confidence thresholds using LLM assessment (REQUIRED)"""
        # Build context descriptions
        evidence_quality = self._describe_evidence_quality(evidence_list)
        hypothesis_complexity = self._describe_hypothesis_complexity(hypothesis, hypothesis_space)
        domain_context = "Process tracing analysis with Van Evera methodology"
        
        # Get LLM assessment of appropriate thresholds (REQUIRED - no fallback)
        self._confidence_thresholds = self.llm.assess_confidence_thresholds(
            evidence_quality=evidence_quality,
            hypothesis_complexity=hypothesis_complexity,
            domain_context=domain_context
        )
    
    def _describe_evidence_quality(self, evidence_list: List[BayesianEvidence]) -> str:
        """Generate description of evidence quality for LLM assessment"""
        if not evidence_list:
            return "No evidence available"
        
        avg_reliability = np.mean([e.reliability for e in evidence_list])
        avg_strength = np.mean([e.strength for e in evidence_list])
        
        return f"""Evidence set with {len(evidence_list)} pieces.
        Average reliability: {avg_reliability:.2f}
        Average strength: {avg_strength:.2f}
        Evidence types: {', '.join(set(str(e.evidence_type) for e in evidence_list))}"""
    
    def _describe_hypothesis_complexity(self, hypothesis: BayesianHypothesis, 
                                       hypothesis_space: BayesianHypothesisSpace) -> str:
        """Generate description of hypothesis complexity for LLM assessment"""
        competitors = hypothesis_space.get_competing_hypotheses(hypothesis.hypothesis_id)
        
        return f"""Hypothesis with {len(hypothesis.supporting_evidence)} supporting evidence.
        Competing hypotheses: {len(competitors)}
        Posterior probability: {hypothesis.posterior_probability:.2f}
        Prior probability: {hypothesis.prior_probability:.2f}"""
    
    def calculate_confidence(self, 
                           hypothesis: BayesianHypothesis,
                           hypothesis_space: BayesianHypothesisSpace,
                           evidence_list: Optional[List[BayesianEvidence]] = None) -> ConfidenceAssessment:
        """
        Calculate comprehensive confidence assessment for a hypothesis.
        
        Args:
            hypothesis: Target hypothesis for confidence assessment
            hypothesis_space: Containing hypothesis space for context
            evidence_list: Optional list of relevant evidence
            
        Returns:
            ConfidenceAssessment with multi-dimensional confidence scores
        """
        # Get relevant evidence
        if evidence_list is None:
            evidence_list = self._get_hypothesis_evidence(hypothesis, hypothesis_space)
        
        # Get LLM-based confidence thresholds for this context
        self._update_confidence_thresholds(hypothesis, evidence_list, hypothesis_space)
        
        # Calculate individual confidence components
        confidence_components = {}
        
        # 1. Evidential confidence (based on evidence quality and quantity)
        confidence_components[ConfidenceType.EVIDENTIAL] = self._calculate_evidential_confidence(
            hypothesis, evidence_list
        )
        
        # 2. Causal confidence (based on causal mechanism coherence)
        confidence_components[ConfidenceType.CAUSAL] = self._calculate_causal_confidence(
            hypothesis, evidence_list, hypothesis_space
        )
        
        # 3. Coherence confidence (based on logical consistency)
        confidence_components[ConfidenceType.COHERENCE] = self._calculate_coherence_confidence(
            hypothesis, evidence_list, hypothesis_space
        )
        
        # 4. Robustness confidence (based on sensitivity to assumptions)
        confidence_components[ConfidenceType.ROBUSTNESS] = self._calculate_robustness_confidence(
            hypothesis, evidence_list
        )
        
        # 5. Sensitivity confidence (based on stability under evidence changes)
        confidence_components[ConfidenceType.SENSITIVITY] = self._calculate_sensitivity_confidence(
            hypothesis, evidence_list
        )
        
        # Calculate overall confidence (weighted combination)
        overall_confidence = self._calculate_overall_confidence(confidence_components)
        
        # Calculate supporting metrics
        evidence_quality_score = self._calculate_evidence_quality_score(evidence_list)
        logical_coherence_score = confidence_components[ConfidenceType.COHERENCE]
        robustness_score = confidence_components[ConfidenceType.ROBUSTNESS]
        sensitivity_score = confidence_components[ConfidenceType.SENSITIVITY]
        
        # Calculate evidence statistics
        supporting_evidence = [e for e in evidence_list if e.evidence_id in hypothesis.supporting_evidence]
        contradicting_evidence = [e for e in evidence_list if e.evidence_id in hypothesis.contradicting_evidence]
        evidence_diversity = self.evidence_quantifier.calculate_evidence_diversity(evidence_list)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            hypothesis, evidence_list, overall_confidence
        )
        
        # Identify uncertainty sources
        uncertainty_sources = self._identify_uncertainty_sources(
            hypothesis, evidence_list, confidence_components
        )
        
        # Create comprehensive assessment
        assessment = ConfidenceAssessment(
            hypothesis_id=hypothesis.hypothesis_id,
            overall_confidence=overall_confidence,
            confidence_components=confidence_components,
            confidence_level=ConfidenceLevel.from_score(overall_confidence, self._confidence_thresholds),
            evidence_quality_score=evidence_quality_score,
            logical_coherence_score=logical_coherence_score,
            robustness_score=robustness_score,
            sensitivity_score=sensitivity_score,
            evidence_count=len(supporting_evidence),
            contradicting_evidence_count=len(contradicting_evidence),
            evidence_diversity_score=evidence_diversity,
            posterior_probability=hypothesis.posterior_probability,
            prior_probability=hypothesis.prior_probability,
            confidence_interval=confidence_interval,
            uncertainty_sources=uncertainty_sources
        )
        
        self.assessment_history.append(assessment)
        return assessment
    
    def _get_hypothesis_evidence(self, hypothesis: BayesianHypothesis, 
                               hypothesis_space: BayesianHypothesisSpace) -> List[BayesianEvidence]:
        """Get all evidence relevant to the hypothesis."""
        relevant_evidence = []
        
        # Get evidence from hypothesis space
        for evidence_id in hypothesis.supporting_evidence.union(hypothesis.contradicting_evidence):
            evidence = hypothesis_space.get_evidence(evidence_id)
            if evidence:
                relevant_evidence.append(evidence)
        
        return relevant_evidence
    
    def _calculate_evidential_confidence(self, hypothesis: BayesianHypothesis, 
                                       evidence_list: List[BayesianEvidence]) -> float:
        """Calculate confidence based on evidence quality and quantity."""
        if not evidence_list:
            return 0.0
        
        # Evidence quality component (average evidence strength)
        evidence_strengths = [e.strength * e.reliability for e in evidence_list]
        quality_score = np.mean(evidence_strengths) if evidence_strengths else 0.0
        
        # Evidence quantity component (diminishing returns)
        quantity_factor = 1 - math.exp(-len(evidence_list) / 3.0)  # Asymptotic to 1
        
        # Evidence diversity component
        diversity_score = self.evidence_quantifier.calculate_evidence_diversity(evidence_list)
        
        # Contradicting evidence penalty
        supporting_count = len(hypothesis.supporting_evidence)
        contradicting_count = len(hypothesis.contradicting_evidence)
        total_count = supporting_count + contradicting_count
        
        if total_count > 0:
            evidence_balance = supporting_count / total_count
        else:
            evidence_balance = 0.5
        
        # Get LLM-determined weights for confidence formula (REQUIRED)
        weights = self.llm.determine_confidence_weights(
            evidence_quality=self._describe_evidence_quality(evidence_list),
            hypothesis_complexity=f"Hypothesis with {len(evidence_list)} evidence pieces",
            domain_context="Evidential confidence calculation"
        )
        
        # Combine components using LLM-determined weights
        evidential_confidence = (
            weights.quality_weight * quality_score +
            weights.quantity_weight * quantity_factor +
            weights.diversity_weight * diversity_score +
            weights.balance_weight * evidence_balance
        )
        
        return max(0.0, min(1.0, evidential_confidence))
    
    def _calculate_causal_confidence(self, hypothesis: BayesianHypothesis,
                                   evidence_list: List[BayesianEvidence],
                                   hypothesis_space: BayesianHypothesisSpace) -> float:
        """Calculate confidence in causal relationship."""
        # Posterior probability component (higher posterior = higher causal confidence)
        posterior_component = hypothesis.posterior_probability
        
        # Likelihood ratio strength (evidence of causation)
        likelihood_ratios = [e.get_likelihood_ratio() for e in evidence_list]
        
        if likelihood_ratios:
            # Use geometric mean to avoid extreme values dominating
            log_ratios = [math.log(max(0.001, lr)) for lr in likelihood_ratios]
            mean_log_ratio = np.mean(log_ratios)
            combined_ratio = math.exp(mean_log_ratio)
            
            # Convert to confidence score (sigmoid transformation)
            ratio_component = 1 / (1 + math.exp(-math.log(combined_ratio)))
        else:
            ratio_component = 0.5
        
        # LLM-based causal mechanism assessment (REQUIRED - no fallback)
        # Build evidence chain description
        evidence_chain = "; ".join([e.description[:100] for e in evidence_list[:5]])
        temporal_sequence = "Temporal ordering based on evidence sequence"
        
        # LLM assessment is REQUIRED - will fail if unavailable
        causal_assessment = self.llm.assess_causal_mechanism(
            hypothesis_description=getattr(hypothesis, 'description', str(hypothesis)),
            evidence_chain=evidence_chain,
            temporal_sequence=temporal_sequence
        )
        
        mechanism_completeness = causal_assessment.mechanism_completeness
        temporal_consistency = causal_assessment.temporal_ordering
        self._causal_assessment = causal_assessment
        
        # Get LLM-determined weights for causal confidence (REQUIRED)
        causal_weights = self.llm.determine_causal_weights(
            hypothesis_description=str(hypothesis),
            evidence_context=f"Evidence set with {len(evidence_list)} pieces",
            domain_context="Causal confidence calculation"
        )
        
        # Combine components using LLM-determined weights
        # Map the generic weight names to causal components
        causal_confidence = (
            causal_weights.quality_weight * posterior_component +
            causal_weights.quantity_weight * ratio_component +
            causal_weights.diversity_weight * mechanism_completeness +
            causal_weights.balance_weight * temporal_consistency
        )
        
        return max(0.0, min(1.0, causal_confidence))
    
    def _calculate_coherence_confidence(self, hypothesis: BayesianHypothesis,
                                      evidence_list: List[BayesianEvidence],
                                      hypothesis_space: BayesianHypothesisSpace) -> float:
        """Calculate confidence based on logical coherence."""
        if not evidence_list:
            return 0.5
        
        # Evidence consistency (how well evidence pieces support each other)
        van_evera_types = [e.evidence_type for e in evidence_list]
        type_diversity = len(set(van_evera_types)) / len(EvidenceType)
        
        # Logical consistency score
        consistency_penalties = 0
        
        # Check for contradictory evidence patterns
        smoking_guns = [e for e in evidence_list if e.evidence_type == EvidenceType.SMOKING_GUN]
        hoops = [e for e in evidence_list if e.evidence_type == EvidenceType.HOOP]
        
        # If we have smoking gun evidence, we should see some hoop evidence too
        if smoking_guns and not hoops:
            consistency_penalties += 0.1
        
        # Check for conflicting likelihood ratios
        ratios = [e.get_likelihood_ratio() for e in evidence_list]
        if ratios:
            ratio_variance = np.var(ratios)
            if ratio_variance > 100:  # High variance suggests inconsistency
                consistency_penalties += 0.1
        
        # Hypothesis coherence with competing hypotheses
        competitors = hypothesis_space.get_competing_hypotheses(hypothesis.hypothesis_id)
        if competitors:
            posterior_differences = [
                abs(hypothesis.posterior_probability - comp.posterior_probability) 
                for comp in competitors
            ]
            separation_score = max(posterior_differences) if posterior_differences else 0
        else:
            separation_score = hypothesis.posterior_probability
        
        # Get LLM-based coherence assessment (REQUIRED - no fallback)
        if not self._confidence_thresholds:
            # Should have been set in _update_confidence_thresholds
            from .llm_required import LLMRequiredError
            raise LLMRequiredError("Confidence thresholds not set - LLM assessment required")
        
        base_coherence = self._confidence_thresholds.logical_coherence
        
        coherence_confidence = (
            base_coherence - consistency_penalties +
            0.1 * type_diversity +
            0.1 * separation_score
        )
        
        return max(0.0, min(1.0, coherence_confidence))
    
    def _calculate_robustness_confidence(self, hypothesis: BayesianHypothesis,
                                       evidence_list: List[BayesianEvidence]) -> float:
        """Calculate confidence in robustness to assumption changes."""
        if not evidence_list:
            return 0.5
        
        # Evidence source diversity (more diverse = more robust)
        collection_methods = set(e.collection_method for e in evidence_list)
        source_diversity = min(1.0, len(collection_methods) / 3.0)  # Assume 3 is good diversity
        
        # Reliability range (consistent reliability = more robust)
        reliabilities = [e.reliability for e in evidence_list]
        reliability_mean = np.mean(reliabilities)
        reliability_std = np.std(reliabilities)
        reliability_consistency = 1.0 - min(1.0, reliability_std)
        
        # Evidence strength distribution (balanced strength = more robust)
        strengths = [e.strength for e in evidence_list]
        strength_balance = 1.0 - abs(0.7 - np.mean(strengths))  # Prefer moderate-high strength
        
        # Get LLM-based independence assessment (REQUIRED - no fallback)
        if not self._confidence_thresholds:
            from .llm_required import LLMRequiredError
            raise LLMRequiredError("Confidence thresholds not set - LLM assessment required")
        
        independence_score = self._confidence_thresholds.independence_score
        
        # Get LLM-determined weights for robustness (REQUIRED)
        robustness_weights = self.llm.determine_robustness_weights(
            evidence_context=f"Evidence from {len(collection_methods)} sources",
            domain_context="Robustness confidence calculation"
        )
        
        # Combine robustness factors using LLM weights
        # Map generic weight names to robustness components
        robustness_confidence = (
            robustness_weights.quality_weight * source_diversity +
            robustness_weights.quantity_weight * reliability_consistency +
            robustness_weights.diversity_weight * strength_balance +
            robustness_weights.balance_weight * independence_score
        )
        
        return max(0.0, min(1.0, robustness_confidence))
    
    def _calculate_sensitivity_confidence(self, hypothesis: BayesianHypothesis,
                                        evidence_list: List[BayesianEvidence]) -> float:
        """Calculate confidence based on sensitivity to evidence changes."""
        if len(evidence_list) < 2:
            return 0.5
        
        # Simulate removing each piece of evidence and check impact
        sensitivity_scores = []
        
        for i, removed_evidence in enumerate(evidence_list):
            remaining_evidence = evidence_list[:i] + evidence_list[i+1:]
            
            # Calculate likelihood ratio without this evidence
            remaining_ratios = [e.get_adjusted_likelihood_ratio() for e in remaining_evidence]
            
            if remaining_ratios:
                combined_ratio = self.evidence_quantifier.combine_multiple_evidence(
                    remaining_evidence, {}  # Assume independence
                )
            else:
                combined_ratio = 1.0
            
            # Calculate how much the conclusion would change
            original_ratio = hypothesis.posterior_probability / (1 - hypothesis.posterior_probability)
            
            # Sensitivity = 1 - relative_change (lower change = higher confidence)
            if original_ratio > 0:
                relative_change = abs(combined_ratio - original_ratio) / original_ratio
                sensitivity = 1.0 - min(1.0, relative_change)
            else:
                sensitivity = 0.5
            
            sensitivity_scores.append(sensitivity)
        
        # Average sensitivity across all evidence pieces
        sensitivity_confidence = np.mean(sensitivity_scores) if sensitivity_scores else 0.5
        
        return max(0.0, min(1.0, sensitivity_confidence))
    
    def _calculate_overall_confidence(self, confidence_components: Dict[ConfidenceType, float]) -> float:
        """Calculate overall confidence from individual components."""
        # Get LLM-determined weights for overall confidence (REQUIRED)
        overall_weights = self.llm.determine_overall_confidence_weights(
            component_scores=str({k.value: v for k, v in confidence_components.items()}),
            domain_context="Overall confidence aggregation"
        )
        
        # Map components to weights (using creative mapping of the 4 weight fields)
        # We have 5 components but only 4 weight fields, so combine related ones
        weighted_sum = (
            overall_weights.quality_weight * confidence_components.get(ConfidenceType.EVIDENTIAL, 0) +
            overall_weights.quantity_weight * confidence_components.get(ConfidenceType.CAUSAL, 0) +
            overall_weights.diversity_weight * confidence_components.get(ConfidenceType.COHERENCE, 0) +
            overall_weights.balance_weight * (
                confidence_components.get(ConfidenceType.ROBUSTNESS, 0) * 0.6 +
                confidence_components.get(ConfidenceType.SENSITIVITY, 0) * 0.4
            )
        )
        
        return max(0.0, min(1.0, weighted_sum))
    
    def _calculate_evidence_quality_score(self, evidence_list: List[BayesianEvidence]) -> float:
        """Calculate overall evidence quality score."""
        if not evidence_list:
            return 0.0
        
        quality_factors = []
        
        for evidence in evidence_list:
            # Individual evidence quality
            quality = (
                0.4 * evidence.reliability +
                0.3 * evidence.strength +
                0.2 * evidence.source_credibility +
                0.1 * (1.0 if evidence.get_likelihood_ratio() > 1.0 else 0.5)
            )
            quality_factors.append(quality)
        
        return np.mean(quality_factors)
    
    def _calculate_confidence_interval(self, hypothesis: BayesianHypothesis,
                                     evidence_list: List[BayesianEvidence],
                                     point_confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for the confidence assessment."""
        # Base uncertainty from evidence quality
        if evidence_list:
            evidence_uncertainties = []
            for evidence in evidence_list:
                # Uncertainty based on reliability and strength
                uncertainty = 1.0 - (evidence.reliability * evidence.strength)
                evidence_uncertainties.append(uncertainty)
            
            average_uncertainty = np.mean(evidence_uncertainties)
        else:
            average_uncertainty = 0.5
        
        # Uncertainty from sample size (fewer evidence = higher uncertainty)
        sample_uncertainty = 1.0 / math.sqrt(max(1, len(evidence_list)))
        
        # Get LLM-based posterior uncertainty (REQUIRED - no fallback)
        if not self._confidence_thresholds:
            from .llm_required import LLMRequiredError
            raise LLMRequiredError("Confidence thresholds not set - LLM assessment required")
        
        posterior_uncertainty = self._confidence_thresholds.posterior_uncertainty
        
        # Combine uncertainties
        total_uncertainty = math.sqrt(
            average_uncertainty**2 + 
            sample_uncertainty**2 + 
            posterior_uncertainty**2
        ) / math.sqrt(3)  # Normalize
        
        # Calculate interval bounds
        margin = total_uncertainty * point_confidence
        lower_bound = max(0.0, point_confidence - margin)
        upper_bound = min(1.0, point_confidence + margin)
        
        return (lower_bound, upper_bound)
    
    def _identify_uncertainty_sources(self, hypothesis: BayesianHypothesis,
                                    evidence_list: List[BayesianEvidence],
                                    confidence_components: Dict[ConfidenceType, float]) -> List[str]:
        """Identify main sources of uncertainty in the assessment."""
        uncertainty_sources = []
        
        # Check for low confidence components
        for conf_type, score in confidence_components.items():
            if score < 0.5:
                uncertainty_sources.append(f"Low {conf_type.value} confidence ({score:.2f})")
        
        # Check evidence-related uncertainties
        if len(evidence_list) < 3:
            uncertainty_sources.append("Limited evidence quantity")
        
        if evidence_list:
            avg_reliability = np.mean([e.reliability for e in evidence_list])
            if avg_reliability < 0.6:
                uncertainty_sources.append("Low evidence reliability")
            
            avg_strength = np.mean([e.strength for e in evidence_list])
            if avg_strength < 0.6:
                uncertainty_sources.append("Weak evidence strength")
        
        # Check for contradicting evidence
        if len(hypothesis.contradicting_evidence) > len(hypothesis.supporting_evidence):
            uncertainty_sources.append("Substantial contradicting evidence")
        
        # Check posterior probability issues
        if hypothesis.posterior_probability < 0.6:
            uncertainty_sources.append("Low posterior probability")
        
        return uncertainty_sources
    
    def compare_hypotheses_confidence(self, hypotheses: List[BayesianHypothesis],
                                    hypothesis_space: BayesianHypothesisSpace) -> Dict[str, ConfidenceAssessment]:
        """Compare confidence assessments across multiple hypotheses."""
        assessments = {}
        
        for hypothesis in hypotheses:
            assessment = self.calculate_confidence(hypothesis, hypothesis_space)
            assessments[hypothesis.hypothesis_id] = assessment
        
        return assessments
    
    def get_confidence_trends(self, hypothesis_id: str) -> List[ConfidenceAssessment]:
        """Get historical confidence trends for a hypothesis."""
        return [
            assessment for assessment in self.assessment_history 
            if assessment.hypothesis_id == hypothesis_id
        ]
    
    def get_confidence_summary(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """Get summary information about a confidence assessment."""
        return {
            "hypothesis_id": assessment.hypothesis_id,
            "overall_confidence": assessment.overall_confidence,
            "confidence_level": assessment.confidence_level.label,
            "confidence_interval": assessment.confidence_interval,
            "evidence_count": assessment.evidence_count,
            "evidence_quality": assessment.evidence_quality_score,
            "main_strengths": self._identify_confidence_strengths(assessment),
            "main_weaknesses": assessment.uncertainty_sources,
            "recommendations": assessment.get_recommendations(),
            "interpretation": assessment.get_interpretation()
        }
    
    def _identify_confidence_strengths(self, assessment: ConfidenceAssessment) -> List[str]:
        """Identify main strengths contributing to confidence."""
        strengths = []
        
        for conf_type, score in assessment.confidence_components.items():
            if score >= 0.7:
                strengths.append(f"Strong {conf_type.value} support ({score:.2f})")
        
        if assessment.evidence_diversity_score >= 0.7:
            strengths.append("High evidence diversity")
        
        if assessment.evidence_count >= 5:
            strengths.append("Substantial evidence quantity")
        
        if assessment.evidence_quality_score >= 0.8:
            strengths.append("High-quality evidence")
        
        return strengths