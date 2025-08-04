"""
Van Evera Bayesian Bridge for Process Tracing.

Bridges Van Evera diagnostic test classifications with Bayesian inference.
Converts LLM-generated evidence assessments into probabilistic values for
seamless integration between qualitative evidence analysis and quantitative
Bayesian inference.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from .bayesian_models import BayesianEvidence, EvidenceType
from .structured_models import EvidenceAssessment, VanEveraEvidenceType
from .diagnostic_probabilities import DiagnosticProbabilityTemplates


@dataclass
class VanEveraBayesianConfig:
    """Configuration for Van Evera Bayesian bridge operations."""
    default_strength: float = 1.0
    default_reliability: float = 1.0
    use_llm_likelihood_overrides: bool = True  # Whether to use LLM likelihood assessments
# Issue #15 Fix: Allow higher likelihoods for proper Bayes factor calculation
    minimum_likelihood: float = 0.001  # Minimum allowed likelihood to avoid division by zero
    maximum_likelihood: float = 0.9999  # Allow much higher likelihood for strong evidence (Bayes factors up to 10,000)


class VanEveraBayesianBridge:
    """
    Bridges Van Evera diagnostic test classifications with Bayesian inference.
    Converts LLM-generated evidence assessments into probabilistic values.
    """
    
    def __init__(self, config: Optional[VanEveraBayesianConfig] = None):
        self.config = config or VanEveraBayesianConfig()
        self.diagnostic_templates = DiagnosticProbabilityTemplates()
    
    def convert_evidence_assessment(self, evidence_assessment: EvidenceAssessment, 
                                  hypothesis_context: str,
                                  source_node_id: str,
                                  evidence_strength: Optional[float] = None,
                                  evidence_reliability: Optional[float] = None) -> BayesianEvidence:
        """
        Convert LLM EvidenceAssessment into BayesianEvidence with proper likelihoods.
        
        Args:
            evidence_assessment: From enhance_evidence.py LLM assessment
            hypothesis_context: Hypothesis description for context
            source_node_id: Node ID in the causal graph
            evidence_strength: Optional override for evidence strength
            evidence_reliability: Optional override for evidence reliability
        
        Returns:
            BayesianEvidence with mathematically valid likelihood assignments
        """
        # Map VanEveraEvidenceType to EvidenceType
        evidence_type = self._map_van_evera_type(evidence_assessment.refined_evidence_type)
        
        # Extract likelihood values from LLM assessment
        likelihood_positive, likelihood_negative = self._extract_likelihoods_from_llm(
            evidence_assessment
        )
        
        # If LLM provided explicit likelihoods and config allows, use them
        if (self.config.use_llm_likelihood_overrides and 
            likelihood_positive is not None and 
            likelihood_negative is not None):
            
            # Validate and adjust if necessary
            likelihood_positive = max(self.config.minimum_likelihood, 
                                    min(self.config.maximum_likelihood, likelihood_positive))
            likelihood_negative = max(self.config.minimum_likelihood, 
                                    min(self.config.maximum_likelihood, likelihood_negative))
        else:
            # Use Van Evera templates
            strength = evidence_strength or self.config.default_strength
            reliability = evidence_reliability or self.config.default_reliability
            
            likelihood_positive, likelihood_negative = self.calculate_van_evera_likelihoods(
                evidence_type, strength, reliability
            )
        
        # Extract strength and reliability from assessment if not provided
        if evidence_strength is None:
            evidence_strength = self._estimate_strength_from_assessment(evidence_assessment)
        if evidence_reliability is None:
            evidence_reliability = self._estimate_reliability_from_assessment(evidence_assessment)
        
        # Create BayesianEvidence object
        bayesian_evidence = BayesianEvidence(
            evidence_id=evidence_assessment.evidence_id,
            description=f"Evidence for {hypothesis_context}",
            evidence_type=evidence_type,
            source_node_id=source_node_id,
            likelihood_positive=likelihood_positive,
            likelihood_negative=likelihood_negative,
            reliability=evidence_reliability,
            strength=evidence_strength,
            collection_method="llm_assessment"
        )
        
        return bayesian_evidence
    
    def calculate_van_evera_likelihoods(self, evidence_type: EvidenceType,
                                      strength: float = 1.0, 
                                      reliability: float = 1.0) -> Tuple[float, float]:
        """
        Calculate P(E|H) and P(E|¬H) from Van Evera classification.
        
        Van Evera Test Types → Likelihood Mappings:
        - HOOP: High P(E|H), Moderate P(E|¬H) - necessity focused
        - SMOKING_GUN: Moderate P(E|H), Low P(E|¬H) - sufficiency focused  
        - DOUBLY_DECISIVE: High P(E|H), Low P(E|¬H) - both necessity and sufficiency
        - STRAW_IN_THE_WIND: Moderate P(E|H), Moderate P(E|¬H) - weak evidence
        
        Args:
            evidence_type: Van Evera evidence type
            strength: Evidence strength factor (0.0-1.0)
            reliability: Source reliability factor (0.0-1.0)
        
        Returns:
            (likelihood_positive, likelihood_negative)
        """
        return self.diagnostic_templates.get_template_probabilities(
            evidence_type, strength, reliability
        )
    
    def _map_van_evera_type(self, van_evera_type: VanEveraEvidenceType) -> EvidenceType:
        """Map VanEveraEvidenceType to EvidenceType."""
        mapping = {
            VanEveraEvidenceType.HOOP: EvidenceType.HOOP,
            VanEveraEvidenceType.SMOKING_GUN: EvidenceType.SMOKING_GUN,
            VanEveraEvidenceType.STRAW_IN_THE_WIND: EvidenceType.STRAW_IN_THE_WIND,
            VanEveraEvidenceType.DOUBLY_DECISIVE: EvidenceType.DOUBLY_DECISIVE
        }
        return mapping[van_evera_type]
    
    def _extract_likelihoods_from_llm(self, evidence_assessment: EvidenceAssessment) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract numerical likelihood values from LLM assessment strings.
        
        Parses likelihood strings like "High (0.8)" or "Very Low" to extract
        numerical probabilities when available.
        
        Returns:
            Tuple of (likelihood_positive, likelihood_negative) or (None, None) if not extractable
        """
        likelihood_positive = self._parse_likelihood_string(evidence_assessment.likelihood_P_E_given_H)
        likelihood_negative = self._parse_likelihood_string(evidence_assessment.likelihood_P_E_given_NotH)
        
        return likelihood_positive, likelihood_negative
    
    def _parse_likelihood_string(self, likelihood_str: str) -> Optional[float]:
        """
        Parse likelihood string to extract numerical value.
        
        Handles formats like:
        - "High (0.8)"
        - "Very Low (0.1)"
        - "Medium"
        - "0.75"
        
        Returns:
            Float value if extractable, None otherwise
        """
        if not likelihood_str:
            return None
        
        # Try to find number in parentheses
        match = re.search(r'\(([0-9]*\.?[0-9]+)\)', likelihood_str)
        if match:
            try:
                value = float(match.group(1))
                if 0.0 <= value <= 1.0:
                    return value
            except ValueError:
                pass
        
        # Try to parse as direct number
        try:
            value = float(likelihood_str.strip())
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
        
        # Map qualitative terms to approximate values
        likelihood_lower = likelihood_str.lower().strip()
        qualitative_mapping = {
            'very high': 0.9,
            'high': 0.8,
            'medium': 0.5,
            'moderate': 0.5,
            'low': 0.2,
            'very low': 0.1,
            'extremely high': 0.95,
            'extremely low': 0.05
        }
        
        for term, value in qualitative_mapping.items():
            if term in likelihood_lower:
                return value
        
        return None
    
    def _estimate_strength_from_assessment(self, evidence_assessment: EvidenceAssessment) -> float:
        """
        Estimate evidence strength from LLM assessment.
        
        Uses the suggested numerical probative value and justification text
        to estimate strength factor.
        """
        # Issue #14 Fix: Use probative value directly (now standardized to 0.0-1.0)
        base_strength = evidence_assessment.suggested_numerical_probative_value
        
        # Adjust based on justification text
        justification_lower = evidence_assessment.justification_for_likelihoods.lower()
        
        # Look for strength indicators
        strength_indicators = {
            'strong': 0.1,
            'clear': 0.05,
            'compelling': 0.15,
            'definitive': 0.2,
            'weak': -0.1,
            'unclear': -0.05,
            'ambiguous': -0.1,
            'uncertain': -0.1
        }
        
        adjustment = 0.0
        for indicator, value in strength_indicators.items():
            if indicator in justification_lower:
                adjustment += value
        
        # Combine base strength with adjustment
        final_strength = max(0.1, min(1.0, base_strength + adjustment))
        return final_strength
    
    def _estimate_reliability_from_assessment(self, evidence_assessment: EvidenceAssessment) -> float:
        """
        Estimate source reliability from LLM assessment.
        
        Uses reasoning and justification to estimate reliability factor.
        """
        # Start with default reliability
        base_reliability = self.config.default_reliability
        
        # Analyze reasoning and justification for reliability indicators
        text_to_analyze = (
            evidence_assessment.reasoning_for_type + " " + 
            evidence_assessment.justification_for_likelihoods
        ).lower()
        
        reliability_indicators = {
            'reliable': 0.1,
            'credible': 0.1,
            'verified': 0.15,
            'confirmed': 0.1,
            'documented': 0.05,
            'unreliable': -0.2,
            'questionable': -0.15,
            'unverified': -0.1,
            'speculative': -0.1,
            'uncertain': -0.05
        }
        
        adjustment = 0.0
        for indicator, value in reliability_indicators.items():
            if indicator in text_to_analyze:
                adjustment += value
        
        # Combine base reliability with adjustment
        final_reliability = max(0.1, min(1.0, base_reliability + adjustment))
        return final_reliability
    
    def create_bayesian_evidence_from_graph_data(self, 
                                               evidence_node: Dict,
                                               hypothesis_node: Dict,
                                               edge_properties: Dict,
                                               van_evera_type: Optional[VanEveraEvidenceType] = None) -> BayesianEvidence:
        """
        Create BayesianEvidence from graph node and edge data without LLM assessment.
        
        Args:
            evidence_node: Evidence node from graph
            hypothesis_node: Hypothesis node from graph
            edge_properties: Edge properties between evidence and hypothesis
            van_evera_type: Optional override for Van Evera type
        
        Returns:
            BayesianEvidence object
        """
        # Determine evidence type
        if van_evera_type:
            evidence_type = self._map_van_evera_type(van_evera_type)
        else:
            # Try to infer from edge properties or use default
            edge_type = edge_properties.get('edge_type', 'supports')
            evidence_type = self._infer_evidence_type_from_edge(edge_type)
        
        # Use default strength and reliability
        strength = self.config.default_strength
        reliability = self.config.default_reliability
        
        # Calculate likelihoods using Van Evera templates
        likelihood_positive, likelihood_negative = self.calculate_van_evera_likelihoods(
            evidence_type, strength, reliability
        )
        
        # Create BayesianEvidence
        bayesian_evidence = BayesianEvidence(
            evidence_id=evidence_node['id'],
            description=evidence_node['properties'].get('description', ''),
            evidence_type=evidence_type,
            source_node_id=evidence_node['id'],
            likelihood_positive=likelihood_positive,
            likelihood_negative=likelihood_negative,
            reliability=reliability,
            strength=strength,
            collection_method="graph_inference"
        )
        
        return bayesian_evidence
    
    def _infer_evidence_type_from_edge(self, edge_type: str) -> EvidenceType:
        """
        Infer Van Evera evidence type from edge type in graph.
        
        This is a heuristic fallback when no explicit Van Evera type is available.
        """
        edge_type_lower = edge_type.lower()
        
        if 'refutes' in edge_type_lower:
            return EvidenceType.SMOKING_GUN  # Refuting evidence is often decisive
        elif 'supports' in edge_type_lower:
            return EvidenceType.STRAW_IN_THE_WIND  # Default support is weak evidence
        elif 'proves' in edge_type_lower or 'confirms' in edge_type_lower:
            return EvidenceType.DOUBLY_DECISIVE
        elif 'requires' in edge_type_lower or 'necessary' in edge_type_lower:
            return EvidenceType.HOOP
        else:
            return EvidenceType.STRAW_IN_THE_WIND  # Default fallback
    
    def batch_convert_evidence_assessments(self, 
                                         evidence_assessments: List[EvidenceAssessment],
                                         hypothesis_context: str,
                                         source_node_ids: List[str]) -> List[BayesianEvidence]:
        """
        Convert multiple evidence assessments in batch.
        
        Args:
            evidence_assessments: List of LLM evidence assessments
            hypothesis_context: Hypothesis description
            source_node_ids: Corresponding source node IDs
        
        Returns:
            List of BayesianEvidence objects
        """
        if len(evidence_assessments) != len(source_node_ids):
            raise ValueError("Number of assessments must match number of source node IDs")
        
        bayesian_evidence_list = []
        for assessment, node_id in zip(evidence_assessments, source_node_ids):
            bayesian_evidence = self.convert_evidence_assessment(
                assessment, hypothesis_context, node_id
            )
            bayesian_evidence_list.append(bayesian_evidence)
        
        return bayesian_evidence_list
    
    def get_likelihood_ratio_summary(self, bayesian_evidence: BayesianEvidence) -> Dict[str, Union[float, str]]:
        """
        Get summary information about likelihood ratios for a piece of evidence.
        
        Returns:
            Dictionary with likelihood ratio analysis
        """
        likelihood_ratio = bayesian_evidence.get_likelihood_ratio()
        adjusted_ratio = bayesian_evidence.get_adjusted_likelihood_ratio()
        
        # Interpret strength of evidence based on likelihood ratio
        if likelihood_ratio == float('inf'):
            strength_interpretation = "Perfect confirming evidence"
        elif likelihood_ratio > 10:
            strength_interpretation = "Very strong confirming evidence"
        elif likelihood_ratio > 3:
            strength_interpretation = "Strong confirming evidence"
        elif likelihood_ratio > 1.5:
            strength_interpretation = "Moderate confirming evidence"
        elif likelihood_ratio > 0.67:
            strength_interpretation = "Weak or neutral evidence"
        elif likelihood_ratio > 0.33:
            strength_interpretation = "Weak disconfirming evidence"
        elif likelihood_ratio > 0.1:
            strength_interpretation = "Strong disconfirming evidence"
        else:
            strength_interpretation = "Very strong disconfirming evidence"
        
        return {
            "likelihood_ratio": likelihood_ratio,
            "adjusted_likelihood_ratio": adjusted_ratio,
            "strength_interpretation": strength_interpretation,
            "evidence_type": bayesian_evidence.evidence_type.value,
            "reliability": bayesian_evidence.reliability,
            "strength": bayesian_evidence.strength
        }