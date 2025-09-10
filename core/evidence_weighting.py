"""
Evidence Strength Quantification for Bayesian Process Tracing.

Quantifies evidence strength for Bayesian weighting, converts qualitative 
LLM assessments into numerical weights, and combines multiple evidence pieces
with proper independence assumptions.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from .plugins.bayesian_van_evera_engine import BayesianHypothesis
from .bayesian_models import BayesianEvidence, EvidenceType, IndependenceType
from .structured_models import EvidenceAssessment



@dataclass
class EvidenceWeights:
    """Quantified weights for evidence in Bayesian inference."""
    evidence_id: str
    base_weight: float              # Base strength weight (0.0-1.0)
    reliability_weight: float       # Reliability adjustment (0.0-1.0)
    credibility_weight: float       # Source credibility adjustment (0.0-1.0)
    temporal_weight: float          # Temporal proximity adjustment (0.0-1.0)
    combined_weight: float          # Final combined weight (0.0-1.0)
    confidence_interval: Tuple[float, float]  # Uncertainty bounds
    
    def __post_init__(self):
        """Validate weight ranges."""
        for attr in ['base_weight', 'reliability_weight', 'credibility_weight', 
                    'temporal_weight', 'combined_weight']:
            value = getattr(self, attr)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{attr} must be between 0 and 1, got {value}")


class EvidenceStrengthQuantifier:
    """
    Quantifies evidence strength for Bayesian weighting.
    Converts qualitative LLM assessments into numerical weights.
    """
    
    def __init__(self):
        self.qualitative_mappings = self._initialize_qualitative_mappings()
    
    def _initialize_qualitative_mappings(self) -> Dict[str, float]:
        """Initialize mappings from qualitative terms to numerical values."""
        return {
            # Strength terms
            'very strong': 0.95,
            'strong': 0.80,
            'moderate': 0.60,
            'weak': 0.40,
            'very weak': 0.20,
            
            # Certainty terms
            'certain': 0.95,
            'likely': 0.75,
            'probable': 0.70,
            'possible': 0.50,
            'unlikely': 0.25,
            'impossible': 0.05,
            
            # Quality terms
            'excellent': 0.90,
            'good': 0.75,
            'fair': 0.60,
            'poor': 0.40,
            'terrible': 0.20,
            
            # Confidence terms
            'high confidence': 0.85,
            'medium confidence': 0.65,
            'low confidence': 0.35,
            'no confidence': 0.15
        }
    
    def quantify_llm_assessment(self, evidence_assessment: EvidenceAssessment) -> EvidenceWeights:
        """
        Convert LLM qualitative assessments into numerical weights.
        
        LLM provides qualitative assessments:
        - "Very High", "High", "Medium", "Low", "Very Low" for P(E|H) and P(E|¬H)
        - Textual justification for likelihood assessments
        
        Must convert to numerical values for Bayesian calculations.
        
        Args:
            evidence_assessment: LLM assessment with qualitative terms
            
        Returns:
            EvidenceWeights with quantified values
        """
        # Issue #14 Fix: Use probative value directly (now standardized to 0.0-1.0)
        base_weight = evidence_assessment.suggested_numerical_probative_value
        
        # Analyze text for reliability indicators
        reliability_weight = self._analyze_reliability_indicators(
            evidence_assessment.reasoning_for_type,
            evidence_assessment.justification_for_likelihoods
        )
        
        # Analyze text for credibility indicators
        credibility_weight = self._analyze_credibility_indicators(
            evidence_assessment.reasoning_for_type,
            evidence_assessment.justification_for_likelihoods
        )
        
        # Default temporal weight (could be enhanced with temporal information)
        temporal_weight = 1.0
        
        # Combine weights using weighted geometric mean to avoid zero products
        weights = [base_weight, reliability_weight, credibility_weight, temporal_weight]
        weights = [max(0.01, w) for w in weights]  # Ensure no zeros
        combined_weight = math.pow(math.prod(weights), 1.0 / len(weights))
        
        # Calculate confidence interval based on uncertainty indicators
        confidence_interval = self._calculate_confidence_interval(
            combined_weight, evidence_assessment
        )
        
        return EvidenceWeights(
            evidence_id=evidence_assessment.evidence_id,
            base_weight=base_weight,
            reliability_weight=reliability_weight,
            credibility_weight=credibility_weight,
            temporal_weight=temporal_weight,
            combined_weight=combined_weight,
            confidence_interval=confidence_interval
        )
    
    def _analyze_reliability_indicators(self, reasoning: str, justification: str) -> float:
        """Use LLM semantic analysis for evidence reliability assessment."""
        from .plugins.van_evera_llm_interface import VanEveraLLMInterface
        from .plugins.van_evera_llm_schemas import ComprehensiveEvidenceAnalysis
        from .llm_required import LLMRequiredError
        
        try:
            llm_interface = VanEveraLLMInterface()
            
            prompt = f"""
Assess evidence reliability based on these indicators:

Reasoning: {reasoning}
Justification: {justification}

Consider semantic understanding of:
- Verification status and documentation quality
- Source consistency and corroboration
- Multiple vs single source evidence
- Professional vs amateur collection methods

Provide comprehensive evidence analysis including reliability assessment.
"""
            
            response = llm_interface._get_structured_response(prompt, ComprehensiveEvidenceAnalysis)
            return response.reliability_score
                    
        except Exception as e:
            raise LLMRequiredError(f"Failed to assess evidence reliability with LLM: {e}")
    
    def _analyze_credibility_indicators(self, reasoning: str, justification: str) -> float:
        """Use LLM semantic analysis for source credibility assessment."""
        from .plugins.van_evera_llm_interface import VanEveraLLMInterface
        from .plugins.van_evera_llm_schemas import ComprehensiveEvidenceAnalysis
        from .llm_required import LLMRequiredError
        
        try:
            llm_interface = VanEveraLLMInterface()
            
            prompt = f"""
Assess source credibility based on semantic analysis:

Reasoning: {reasoning}
Justification: {justification}

Consider semantic understanding of:
- Official vs unofficial source authority
- Academic/expert vs amateur qualifications
- Institutional vs individual source type
- Bias indicators and conflict of interest
- Publication and peer review status

Provide comprehensive evidence analysis including credibility assessment.
"""
            
            response = llm_interface._get_structured_response(prompt, ComprehensiveEvidenceAnalysis)
            # Use evidence_quality as proxy for credibility since ComprehensiveEvidenceAnalysis doesn't have credibility_score
            quality_mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
            return quality_mapping.get(response.evidence_quality, 0.6)
                    
        except Exception as e:
            raise LLMRequiredError(f"Failed to assess source credibility with LLM: {e}")
    
    def _calculate_confidence_interval(self, combined_weight: float, 
                                     evidence_assessment: EvidenceAssessment) -> Tuple[float, float]:
        """Calculate confidence interval for the weight estimate."""
        # Base uncertainty from qualitative nature of assessment
        base_uncertainty = 0.1
        
        # Additional uncertainty from assessment quality
        justification_length = len(evidence_assessment.justification_for_likelihoods)
        if justification_length < 50:
            uncertainty_adjustment = 0.1  # High uncertainty for brief justifications
        elif justification_length < 200:
            uncertainty_adjustment = 0.05  # Medium uncertainty
        else:
            uncertainty_adjustment = 0.02  # Low uncertainty for detailed justifications
        
        total_uncertainty = base_uncertainty + uncertainty_adjustment
        
        # Calculate bounds
        lower_bound = max(0.0, combined_weight - total_uncertainty)
        upper_bound = min(1.0, combined_weight + total_uncertainty)
        
        return (lower_bound, upper_bound)
    
    def combine_multiple_evidence(self, evidence_list: List[BayesianEvidence],
                                independence_assumptions: Dict[str, IndependenceType]) -> float:
        """
        Combine multiple pieces of evidence with independence assumptions.
        
        When multiple evidence pieces support/contradict same hypothesis:
        - Independent evidence: multiply likelihood ratios
        - Dependent evidence: more complex combination rules
        
        Args:
            evidence_list: List of BayesianEvidence objects
            independence_assumptions: Mapping from evidence pairs to independence type
            
        Returns:
            Combined likelihood ratio
        """
        if not evidence_list:
            return 1.0
        
        if len(evidence_list) == 1:
            return evidence_list[0].get_adjusted_likelihood_ratio()
        
        # Use the implementation from EvidenceStrengthQuantifier in bayesian_models
        from .bayesian_models import EvidenceStrengthQuantifier
        quantifier = EvidenceStrengthQuantifier()
        
        return quantifier.combine_multiple_evidence(evidence_list, independence_assumptions)
    
    def _group_evidence_by_independence(self, evidence_list: List[BayesianEvidence],
                                      independence_assumptions: Dict[str, IndependenceType]) -> List[List[BayesianEvidence]]:
        """Group evidence pieces by their independence relationships."""
        from .bayesian_models import EvidenceStrengthQuantifier
        quantifier = EvidenceStrengthQuantifier()
        
        return quantifier._group_evidence_by_independence(evidence_list, independence_assumptions)
    
    def _combine_dependent_evidence(self, evidence_group: List[BayesianEvidence]) -> float:
        """
        Combine dependent evidence using conservative approach.
        
        For dependent evidence, we can't simply multiply likelihood ratios.
        Instead, we use a conservative combination that accounts for dependence.
        """
        from .bayesian_models import EvidenceStrengthQuantifier
        quantifier = EvidenceStrengthQuantifier()
        
        return quantifier._combine_dependent_evidence(evidence_group)
    
    def calculate_evidence_diversity(self, evidence_list: List[BayesianEvidence]) -> float:
        """
        Calculate diversity score for a set of evidence.
        
        Higher diversity indicates more independent sources and types of evidence,
        which strengthens the overall case.
        
        Returns:
            Diversity score (0.0-1.0)
        """
        from .bayesian_models import EvidenceStrengthQuantifier
        quantifier = EvidenceStrengthQuantifier()
        
        return quantifier.calculate_evidence_diversity(evidence_list)
    
    def get_evidence_strength_summary(self, evidence_weights: EvidenceWeights) -> Dict[str, Union[float, str]]:
        """
        Get summary of evidence strength assessment.
        
        Returns:
            Dictionary with strength analysis
        """
        # Interpret combined weight
        if evidence_weights.combined_weight >= 0.8:
            strength_category = "Very Strong"
        elif evidence_weights.combined_weight >= 0.6:
            strength_category = "Strong"
        elif evidence_weights.combined_weight >= 0.4:
            strength_category = "Moderate"
        elif evidence_weights.combined_weight >= 0.2:
            strength_category = "Weak"
        else:
            strength_category = "Very Weak"
        
        # Calculate uncertainty
        lower, upper = evidence_weights.confidence_interval
        uncertainty = upper - lower
        
        return {
            "evidence_id": evidence_weights.evidence_id,
            "combined_weight": evidence_weights.combined_weight,
            "strength_category": strength_category,
            "base_weight": evidence_weights.base_weight,
            "reliability_weight": evidence_weights.reliability_weight,
            "credibility_weight": evidence_weights.credibility_weight,
            "uncertainty": uncertainty,
            "confidence_interval": evidence_weights.confidence_interval
        }