"""
Diagnostic Probability Templates for Van Evera Bayesian Integration.

Provides probability templates for Van Evera diagnostic tests based on formal
Van Evera methodology with Bayesian interpretation. Implements the mapping
from Van Evera test types to likelihood ratios for systematic evidence evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum
from .bayesian_models import EvidenceType


@dataclass
class VanEveraTemplate:
    """Template for Van Evera diagnostic test probability assignments."""
    evidence_type: EvidenceType
    base_likelihood_positive: float  # P(E|H) baseline
    base_likelihood_negative: float  # P(E|¬H) baseline
    strength_modifier: float         # How strength affects probabilities
    reliability_modifier: float     # How reliability affects probabilities
    description: str
    
    def __post_init__(self):
        """Validate probability ranges."""
        for attr in ['base_likelihood_positive', 'base_likelihood_negative']:
            value = getattr(self, attr)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{attr} must be between 0 and 1, got {value}")


class DiagnosticProbabilityTemplates:
    """
    Provides probability templates for Van Evera diagnostic tests.
    Based on formal Van Evera methodology with Bayesian interpretation.
    
    Van Evera Test Types:
    - HOOP: High necessity, low sufficiency - evidence is necessary for hypothesis
    - SMOKING_GUN: Low necessity, high sufficiency - evidence strongly confirms hypothesis
    - DOUBLY_DECISIVE: High necessity and sufficiency - both necessary and sufficient
    - STRAW_IN_THE_WIND: Low necessity and sufficiency - weak evidence
    """
    
    TEMPLATES = {
        EvidenceType.HOOP: VanEveraTemplate(
            evidence_type=EvidenceType.HOOP,
            base_likelihood_positive=0.85,  # High necessity - should see evidence if hypothesis true
            base_likelihood_negative=0.40,  # Moderate false positive rate
            strength_modifier=0.10,         # Strength affects how much evidence supports
            reliability_modifier=0.15,     # Reliability affects confidence in observation
            description="Necessary test - hypothesis unlikely without this evidence"
        ),
        
        EvidenceType.SMOKING_GUN: VanEveraTemplate(
            evidence_type=EvidenceType.SMOKING_GUN,
            base_likelihood_positive=0.70,  # Moderate necessity - may not always see evidence
            base_likelihood_negative=0.05,  # Very low false positive rate - sufficiency focused
            strength_modifier=0.15,
            reliability_modifier=0.20,
            description="Sufficient test - evidence strongly confirms hypothesis"
        ),
        
        EvidenceType.DOUBLY_DECISIVE: VanEveraTemplate(
            evidence_type=EvidenceType.DOUBLY_DECISIVE,
            base_likelihood_positive=0.90,  # Very high necessity and sufficiency
            base_likelihood_negative=0.05,  # Very low false positive rate
            strength_modifier=0.08,         # Less sensitive to adjustments due to high confidence
            reliability_modifier=0.10,
            description="Both necessary and sufficient - confirms hypothesis and refutes alternatives"
        ),
        
        EvidenceType.STRAW_IN_THE_WIND: VanEveraTemplate(
            evidence_type=EvidenceType.STRAW_IN_THE_WIND,
            base_likelihood_positive=0.60,  # Weak necessity
            base_likelihood_negative=0.50,  # High false positive rate - weak evidence
            strength_modifier=0.20,         # More sensitive to strength and reliability
            reliability_modifier=0.25,
            description="Weak evidence - provides some support but not decisive"
        )
    }
    
    @classmethod
    def get_template_probabilities(cls, evidence_type: EvidenceType, 
                                 strength: float = 1.0, 
                                 reliability: float = 1.0) -> Tuple[float, float]:
        """
        Calculate adjusted probabilities from template.
        
        Args:
            evidence_type: Van Evera evidence type
            strength: Evidence strength factor (0.0-1.0)
            reliability: Source reliability factor (0.0-1.0)
            
        Returns:
            Tuple of (likelihood_positive, likelihood_negative)
        """
        if evidence_type not in cls.TEMPLATES:
            raise ValueError(f"Unknown evidence type: {evidence_type}")
        
        # Validate input parameters
        if not (0.0 <= strength <= 1.0):
            raise ValueError(f"Strength must be between 0 and 1, got {strength}")
        if not (0.0 <= reliability <= 1.0):
            raise ValueError(f"Reliability must be between 0 and 1, got {reliability}")
        
        template = cls.TEMPLATES[evidence_type]
        
        # Calculate adjustment factor combining strength and reliability
        strength_adjustment = (strength - 1.0) * template.strength_modifier
        reliability_adjustment = (reliability - 1.0) * template.reliability_modifier
        total_adjustment = strength_adjustment + reliability_adjustment
        
        # Apply adjustments to base probabilities
        likelihood_positive = template.base_likelihood_positive + total_adjustment
        likelihood_negative = template.base_likelihood_negative - total_adjustment * 0.5  # Asymmetric adjustment
        
        # Ensure probabilities stay within valid bounds
        likelihood_positive = max(0.01, min(0.99, likelihood_positive))
        likelihood_negative = max(0.01, min(0.99, likelihood_negative))
        
        # Ensure Van Evera test logic is preserved
        likelihood_positive, likelihood_negative = cls._enforce_van_evera_logic(
            evidence_type, likelihood_positive, likelihood_negative
        )
        
        return likelihood_positive, likelihood_negative
    
    @classmethod
    def _enforce_van_evera_logic(cls, evidence_type: EvidenceType, 
                               likelihood_positive: float, 
                               likelihood_negative: float) -> Tuple[float, float]:
        """
        Enforce Van Evera test logic constraints on calculated probabilities.
        
        Ensures that:
        - HOOP tests maintain high P(E|H) (necessity)
        - SMOKING_GUN tests maintain low P(E|¬H) (sufficiency) 
        - DOUBLY_DECISIVE tests maintain both constraints
        - STRAW_IN_THE_WIND tests remain weak
        """
        if evidence_type == EvidenceType.HOOP:
            # High necessity: P(E|H) should be high
            likelihood_positive = max(0.70, likelihood_positive)
            
        elif evidence_type == EvidenceType.SMOKING_GUN:
            # High sufficiency: P(E|¬H) should be low
            likelihood_negative = min(0.15, likelihood_negative)
            
        elif evidence_type == EvidenceType.DOUBLY_DECISIVE:
            # Both high necessity and sufficiency
            likelihood_positive = max(0.80, likelihood_positive)
            likelihood_negative = min(0.15, likelihood_negative)
            
        elif evidence_type == EvidenceType.STRAW_IN_THE_WIND:
            # Weak evidence: likelihood ratio should be close to 1
            # Issue #63 Fix: Use epsilon-based floating point comparison
            from .float_utils import float_zero
            ratio = likelihood_positive / likelihood_negative if not float_zero(likelihood_negative) else float('inf')
            if ratio > 3.0:  # Too strong for straw in the wind
                # Adjust to weaken the evidence
                target_ratio = 2.0
                likelihood_negative = likelihood_positive / target_ratio
                likelihood_negative = max(0.20, min(0.80, likelihood_negative))
        
        return likelihood_positive, likelihood_negative
    
    @classmethod
    def get_likelihood_ratio(cls, evidence_type: EvidenceType, 
                           strength: float = 1.0, 
                           reliability: float = 1.0) -> float:
        """
        Calculate likelihood ratio for given evidence type and parameters.
        
        Returns:
            Likelihood ratio P(E|H) / P(E|¬H)
        """
        likelihood_positive, likelihood_negative = cls.get_template_probabilities(
            evidence_type, strength, reliability
        )
        
        # Issue #63 Fix: Use epsilon-based floating point comparison
        from .float_utils import float_zero
        
        if float_zero(likelihood_negative):
            return float('inf')
        
        return likelihood_positive / likelihood_negative
    
    @classmethod
    def get_template_description(cls, evidence_type: EvidenceType) -> str:
        """Get description for a Van Evera evidence type."""
        if evidence_type not in cls.TEMPLATES:
            return f"Unknown evidence type: {evidence_type}"
        
        return cls.TEMPLATES[evidence_type].description
    
    @classmethod
    def list_all_templates(cls) -> Dict[str, Dict[str, float]]:
        """
        List all available templates with their base probabilities.
        
        Returns:
            Dictionary mapping evidence type names to probability data
        """
        return {
            evidence_type.value: {
                "base_likelihood_positive": template.base_likelihood_positive,
                "base_likelihood_negative": template.base_likelihood_negative,
                "strength_modifier": template.strength_modifier,
                "reliability_modifier": template.reliability_modifier,
                "description": template.description
            }
            for evidence_type, template in cls.TEMPLATES.items()
        }


def validate_van_evera_probabilities(likelihood_positive: float, 
                                   likelihood_negative: float,
                                   evidence_type: EvidenceType) -> bool:
    """
    Validate that probability values are consistent with Van Evera test logic.
    
    Args:
        likelihood_positive: P(E|H) 
        likelihood_negative: P(E|¬H)
        evidence_type: Van Evera evidence type
        
    Returns:
        True if probabilities are consistent with test type logic
    """
    # Basic range validation
    if not (0.0 <= likelihood_positive <= 1.0):
        return False
    if not (0.0 <= likelihood_negative <= 1.0):
        return False
    
    # Issue #63 Fix: Use epsilon-based floating point comparison
    from .float_utils import float_zero
    ratio = likelihood_positive / likelihood_negative if not float_zero(likelihood_negative) else float('inf')
    
    # Test type specific validation
    if evidence_type == EvidenceType.HOOP:
        # Should have high necessity (high P(E|H))
        return likelihood_positive >= 0.60
        
    elif evidence_type == EvidenceType.SMOKING_GUN:
        # Should have high sufficiency (low P(E|¬H), high ratio)
        return likelihood_negative <= 0.25 and ratio >= 2.0
        
    elif evidence_type == EvidenceType.DOUBLY_DECISIVE:
        # Should have both high necessity and sufficiency
        return likelihood_positive >= 0.70 and likelihood_negative <= 0.25 and ratio >= 3.0
        
    elif evidence_type == EvidenceType.STRAW_IN_THE_WIND:
        # Should be relatively weak evidence (ratio close to 1)
        return 0.5 <= ratio <= 5.0
    
    return True