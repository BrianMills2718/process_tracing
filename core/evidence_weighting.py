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
from .plugins.bayesian_van_evera_engine import BayesianHypothesis, EvidenceType
from .structured_models import EvidenceAssessment


class IndependenceType(Enum):
    """Types of independence relationships between evidence pieces."""
    INDEPENDENT = "independent"        # Statistically independent
    DEPENDENT = "dependent"           # Statistically dependent  
    CONDITIONALLY_INDEPENDENT = "conditionally_independent"  # Independent given hypothesis
    REDUNDANT = "redundant"           # Essentially the same evidence


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
        from .llm_required import LLMRequiredError
        
        try:
            llm_interface = VanEveraLLMInterface()
            
            prompt = f"""
Assess evidence reliability (0.0-1.0) based on these indicators:

Reasoning: {reasoning}
Justification: {justification}

Consider semantic understanding of:
- Verification status and documentation quality
- Source consistency and corroboration
- Multiple vs single source evidence
- Professional vs amateur collection methods

Return only numerical score 0.0-1.0 representing reliability.
"""
            
            response = llm_interface.get_structured_response(prompt, target_schema="reliability_float")
            if hasattr(response, 'reliability_score'):
                return float(response.reliability_score)
            elif hasattr(response, 'score'):
                return float(response.score)
            else:
                # Extract numerical value from response
                import re
                text_response = str(response)
                numbers = re.findall(r'0?\.\d+|[01]\.0*', text_response)
                if numbers:
                    return min(1.0, max(0.0, float(numbers[0])))
                else:
                    raise ValueError("No numerical score found in LLM response")
                    
        except Exception as e:
            raise LLMRequiredError(f"Failed to assess evidence reliability with LLM: {e}")
    
    def _analyze_credibility_indicators(self, reasoning: str, justification: str) -> float:
        """Use LLM semantic analysis for source credibility assessment."""
        from .plugins.van_evera_llm_interface import VanEveraLLMInterface
        from .llm_required import LLMRequiredError
        
        try:
            llm_interface = VanEveraLLMInterface()
            
            prompt = f"""
Assess source credibility (0.0-1.0) based on semantic analysis:

Reasoning: {reasoning}
Justification: {justification}

Consider semantic understanding of:
- Official vs unofficial source authority
- Academic/expert vs amateur qualifications
- Institutional vs individual source type
- Bias indicators and conflict of interest
- Publication and peer review status

Return only numerical score 0.0-1.0 representing credibility.
"""
            
            response = llm_interface.get_structured_response(prompt, target_schema="credibility_float")
            if hasattr(response, 'credibility_score'):
                return float(response.credibility_score)
            elif hasattr(response, 'score'):
                return float(response.score)
            else:
                # Extract numerical value from response
                import re
                text_response = str(response)
                numbers = re.findall(r'0?\.\d+|[01]\.0*', text_response)
                if numbers:
                    return min(1.0, max(0.0, float(numbers[0])))
                else:
                    raise ValueError("No numerical score found in LLM response")
                    
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
        
        # Group evidence by independence relationships
        independent_groups = self._group_evidence_by_independence(
            evidence_list, independence_assumptions
        )
        
        combined_ratio = 1.0
        
        for group in independent_groups:
            if len(group) == 1:
                # Single piece of evidence
                group_ratio = group[0].get_adjusted_likelihood_ratio()
            else:
                # Multiple evidence pieces in this independence group
                group_ratio = self._combine_dependent_evidence(group)
            
            # Multiply ratios for independent groups
            combined_ratio *= group_ratio
        
        return combined_ratio
    
    def _group_evidence_by_independence(self, evidence_list: List[BayesianEvidence],
                                      independence_assumptions: Dict[str, IndependenceType]) -> List[List[BayesianEvidence]]:
        """Group evidence pieces by their independence relationships."""
        # Create adjacency graph for dependence relationships
        evidence_ids = [e.evidence_id for e in evidence_list]
        dependent_pairs = set()
        
        for pair_key, independence_type in independence_assumptions.items():
            if independence_type in [IndependenceType.DEPENDENT, IndependenceType.REDUNDANT]:
                # Parse pair key (assuming format "id1-id2" or similar)
                if '-' in pair_key:
                    id1, id2 = pair_key.split('-', 1)
                    if id1 in evidence_ids and id2 in evidence_ids:
                        dependent_pairs.add((id1, id2))
        
        # Use union-find to group dependent evidence
        groups = []
        remaining_evidence = evidence_list.copy()
        
        while remaining_evidence:
            current_group = [remaining_evidence.pop(0)]
            current_ids = {current_group[0].evidence_id}
            
            # Find all evidence dependent on current group
            changed = True
            while changed:
                changed = False
                for i in range(len(remaining_evidence) - 1, -1, -1):
                    evidence = remaining_evidence[i]
                    # Check if this evidence is dependent on any in current group
                    for current_id in current_ids:
                        if ((evidence.evidence_id, current_id) in dependent_pairs or
                            (current_id, evidence.evidence_id) in dependent_pairs):
                            current_group.append(remaining_evidence.pop(i))
                            current_ids.add(evidence.evidence_id)
                            changed = True
                            break
            
            groups.append(current_group)
        
        return groups
    
    def _combine_dependent_evidence(self, evidence_group: List[BayesianEvidence]) -> float:
        """
        Combine dependent evidence using conservative approach.
        
        For dependent evidence, we can't simply multiply likelihood ratios.
        Instead, we use a conservative combination that accounts for dependence.
        """
        if not evidence_group:
            return 1.0
        
        if len(evidence_group) == 1:
            return evidence_group[0].get_adjusted_likelihood_ratio()
        
        # Get all likelihood ratios
        ratios = [e.get_adjusted_likelihood_ratio() for e in evidence_group]
        
        # Use geometric mean for conservative combination
        # This is more conservative than multiplication but stronger than arithmetic mean
        if any(r == float('inf') for r in ratios):
            return float('inf')  # If any evidence is perfect, result is perfect
        
        # Issue #63 Fix: Use epsilon-based floating point comparison
        from .float_utils import float_one
        
        # Filter out ratios that are effectively 1 (neutral evidence)
        significant_ratios = [r for r in ratios if not float_one(r)]
        
        if not significant_ratios:
            return 1.0  # All evidence is neutral
        
        # Use geometric mean with dampening factor to account for dependence
        geometric_mean = math.pow(math.prod(significant_ratios), 1.0 / len(significant_ratios))
        
        # Apply dampening factor (reduces strength due to dependence)
        dampening_factor = 1.0 / math.sqrt(len(evidence_group))
        
        # Combine geometric mean with dampening
        if geometric_mean > 1.0:
            combined_ratio = 1.0 + (geometric_mean - 1.0) * dampening_factor
        else:
            combined_ratio = 1.0 - (1.0 - geometric_mean) * dampening_factor
        
        return max(0.01, combined_ratio)
    
    def calculate_evidence_diversity(self, evidence_list: List[BayesianEvidence]) -> float:
        """
        Calculate diversity score for a set of evidence.
        
        Higher diversity indicates more independent sources and types of evidence,
        which strengthens the overall case.
        
        Returns:
            Diversity score (0.0-1.0)
        """
        if not evidence_list:
            return 0.0
        
        if len(evidence_list) == 1:
            return 0.5  # Single piece has moderate diversity
        
        # Analyze diversity across multiple dimensions
        evidence_types = set(e.evidence_type for e in evidence_list)
        collection_methods = set(e.collection_method for e in evidence_list)
        reliability_levels = [e.reliability for e in evidence_list]
        
        # Type diversity (more Van Evera types is better)
        type_diversity = len(evidence_types) / len(EvidenceType)
        
        # Method diversity (more collection methods is better)
        method_diversity = min(1.0, len(collection_methods) / 3.0)  # Cap at 3 methods
        
        # Reliability diversity (some spread in reliability indicates varied sources)
        if len(set(reliability_levels)) > 1:
            reliability_std = float(sum((r - sum(reliability_levels)/len(reliability_levels))**2 
                                      for r in reliability_levels) / len(reliability_levels))**0.5
            reliability_diversity = min(1.0, reliability_std * 2.0)  # Scale standard deviation
        else:
            reliability_diversity = 0.5  # All same reliability
        
        # Combine diversity measures
        overall_diversity = (type_diversity + method_diversity + reliability_diversity) / 3.0
        
        return min(1.0, overall_diversity)
    
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