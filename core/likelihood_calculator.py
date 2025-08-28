"""
Evidence likelihood ratio calculations for Bayesian process tracing.

Implements likelihood ratio computation for different types of evidence
with Van Evera diagnostic test integration and uncertainty quantification.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import networkx as nx
from datetime import datetime
import json
import math
from pathlib import Path

from .bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace, 
    EvidenceType, HypothesisType
)


class LikelihoodCalculationMethod(Enum):
    """Methods for calculating evidence likelihoods."""
    VAN_EVERA = "van_evera"                  # Based on Van Evera diagnostic tests
    FREQUENCY_BASED = "frequency_based"      # Based on empirical frequencies
    EXPERT_ELICITED = "expert_elicited"      # Based on expert judgment
    MECHANISM_BASED = "mechanism_based"      # Based on causal mechanism analysis
    CONTEXTUAL = "contextual"                # Context-dependent likelihood
    HIERARCHICAL = "hierarchical"            # Based on evidence hierarchy
    CONDITIONAL = "conditional"              # Conditional on other evidence


@dataclass
class LikelihoodCalculationConfig:
    """Configuration for likelihood calculations."""
    method: LikelihoodCalculationMethod
    van_evera_strictness: float = 0.8       # How strict Van Evera classifications are
    uncertainty_factor: float = 0.1         # Uncertainty in likelihood estimates
    context_sensitivity: float = 0.5        # How much context affects likelihoods
    temporal_decay: float = 0.05            # How time affects evidence strength
    independence_assumption: bool = True     # Whether evidence pieces are independent
    confidence_threshold: float = 0.7       # Minimum confidence for reliable likelihoods


class VanEveraLikelihoodCalculator:
    """
    Calculate likelihoods based on Van Evera diagnostic test classifications.
    
    Implements the four-fold Van Evera typology with probabilistic
    interpretations for systematic evidence evaluation.
    """
    
    def __init__(self, config: LikelihoodCalculationConfig):
        self.config = config
        self.strictness = config.van_evera_strictness
        
        # Van Evera probability templates
        self.van_evera_templates = {
            EvidenceType.HOOP: {
                "necessity_min": 0.8,      # High necessity
                "sufficiency_max": 0.4,    # Low sufficiency
                "description": "Hoop test - necessary but not sufficient"
            },
            EvidenceType.SMOKING_GUN: {
                "necessity_max": 0.4,      # Low necessity
                "sufficiency_min": 0.8,    # High sufficiency
                "description": "Smoking gun - sufficient but not necessary"
            },
            EvidenceType.DOUBLY_DECISIVE: {
                "necessity_min": 0.8,      # High necessity
                "sufficiency_min": 0.8,    # High sufficiency
                "description": "Doubly decisive - necessary and sufficient"
            },
            EvidenceType.STRAW_IN_THE_WIND: {
                "necessity_max": 0.5,      # Low necessity
                "sufficiency_max": 0.5,    # Low sufficiency
                "description": "Straw in the wind - neither necessary nor sufficient"
            }
        }
    
    def calculate_likelihood_ratio(self, evidence: BayesianEvidence, 
                                 hypothesis: BayesianHypothesis) -> float:
        """Calculate likelihood ratio for evidence given hypothesis."""
        # Get Van Evera template
        template = self.van_evera_templates[evidence.evidence_type]
        
        # Calculate P(E|H) - probability of evidence given hypothesis is true
        p_e_given_h = self._calculate_p_e_given_h(evidence, hypothesis, template)
        
        # Calculate P(E|¬H) - probability of evidence given hypothesis is false
        p_e_given_not_h = self._calculate_p_e_given_not_h(evidence, hypothesis, template)
        
        # Calculate likelihood ratio
        if p_e_given_not_h == 0:
            return float('inf')  # Perfect diagnostic evidence
        
        likelihood_ratio = p_e_given_h / p_e_given_not_h
        
        # Apply uncertainty adjustment
        likelihood_ratio = self._apply_uncertainty_adjustment(likelihood_ratio, evidence)
        
        return likelihood_ratio
    
    def _calculate_p_e_given_h(self, evidence: BayesianEvidence, 
                              hypothesis: BayesianHypothesis, 
                              template: Dict[str, Any]) -> float:
        """Calculate P(E|H) based on Van Evera type and evidence properties."""
        # If user provided explicit likelihood_positive, use it
        if hasattr(evidence, '_user_provided_likelihood_positive') and evidence._user_provided_likelihood_positive:
            return evidence.likelihood_positive
        
        base_probability = evidence.necessity  # Start with stated necessity
        
        # Adjust based on Van Evera template
        if evidence.evidence_type == EvidenceType.HOOP:
            # Hoop tests should have high P(E|H)
            min_necessity = template["necessity_min"]
            base_probability = max(base_probability, min_necessity)
        
        elif evidence.evidence_type == EvidenceType.SMOKING_GUN:
            # Smoking gun tests may have lower P(E|H) but very low P(E|¬H)
            base_probability = min(evidence.sufficiency, 0.9)
        
        elif evidence.evidence_type == EvidenceType.DOUBLY_DECISIVE:
            # Doubly decisive should have high P(E|H)
            min_necessity = template["necessity_min"]
            base_probability = max(base_probability, min_necessity)
        
        elif evidence.evidence_type == EvidenceType.STRAW_IN_THE_WIND:
            # Straw in the wind has moderate P(E|H)
            base_probability = 0.5 + evidence.strength * 0.3
        
        # Apply reliability and strength adjustments
        adjusted_probability = base_probability * evidence.reliability * evidence.strength
        
        # Apply strictness factor
        if self.strictness > 0.5:
            # More strict - strengthen the Van Evera classification
            adjustment = (self.strictness - 0.5) * 2  # Scale to 0-1
            if evidence.evidence_type in [EvidenceType.HOOP, EvidenceType.DOUBLY_DECISIVE]:
                adjusted_probability = adjusted_probability + adjustment * (1.0 - adjusted_probability)
        
        return np.clip(adjusted_probability, 0.01, 0.99)
    
    def _calculate_p_e_given_not_h(self, evidence: BayesianEvidence, 
                                  hypothesis: BayesianHypothesis, 
                                  template: Dict[str, Any]) -> float:
        """Calculate P(E|¬H) based on Van Evera type and evidence properties."""
        # If user provided explicit likelihood_negative, use it
        if hasattr(evidence, '_user_provided_likelihood_negative') and evidence._user_provided_likelihood_negative:
            return evidence.likelihood_negative
        
        # Start with complement of sufficiency for baseline
        base_probability = 1.0 - evidence.sufficiency
        
        # Adjust based on Van Evera template
        if evidence.evidence_type == EvidenceType.HOOP:
            # Hoop tests can occur with alternative hypotheses
            base_probability = min(0.5, 1.0 - evidence.necessity * 0.5)
        
        elif evidence.evidence_type == EvidenceType.SMOKING_GUN:
            # Smoking gun tests should be very rare under alternative hypotheses
            max_false_positive = 0.2 - evidence.sufficiency * 0.15
            base_probability = max(0.01, max_false_positive)
        
        elif evidence.evidence_type == EvidenceType.DOUBLY_DECISIVE:
            # Doubly decisive should be very rare under alternatives
            base_probability = max(0.01, 0.1 * (1.0 - evidence.sufficiency))
        
        elif evidence.evidence_type == EvidenceType.STRAW_IN_THE_WIND:
            # Straw in the wind can occur with alternatives
            base_probability = 0.4 + (1.0 - evidence.strength) * 0.2
        
        # Apply reliability adjustment (unreliable evidence more likely to appear spuriously)
        unreliability_factor = 1.0 - evidence.reliability
        base_probability = base_probability + unreliability_factor * 0.3
        
        # Apply strictness factor
        if self.strictness > 0.5:
            adjustment = (self.strictness - 0.5) * 2
            if evidence.evidence_type in [EvidenceType.SMOKING_GUN, EvidenceType.DOUBLY_DECISIVE]:
                # Reduce false positive rate for strong evidence types
                base_probability = base_probability * (1.0 - adjustment * 0.5)
        
        return np.clip(base_probability, 0.01, 0.99)
    
    def _apply_uncertainty_adjustment(self, likelihood_ratio: float, 
                                    evidence: BayesianEvidence) -> float:
        """Apply uncertainty adjustment to likelihood ratio."""
        uncertainty = self.config.uncertainty_factor
        
        # Adjust based on evidence reliability and source credibility
        total_reliability = evidence.reliability * evidence.source_credibility
        adjusted_uncertainty = uncertainty * (1.0 - total_reliability)
        
        # Move likelihood ratio towards 1 (neutral) based on uncertainty
        if likelihood_ratio > 1:
            return 1 + (likelihood_ratio - 1) * (1.0 - adjusted_uncertainty)
        else:
            return 1 - (1 - likelihood_ratio) * (1.0 - adjusted_uncertainty)


class FrequencyBasedLikelihoodCalculator:
    """
    Calculate likelihoods based on empirical frequencies of evidence patterns.
    
    Uses historical data about evidence occurrence to estimate
    likelihood ratios for hypothesis testing.
    """
    
    def __init__(self, config: LikelihoodCalculationConfig):
        self.config = config
        self.frequency_data: Dict[str, Dict[str, float]] = {}
        self.evidence_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.smoothing_factor = 0.1
    
    def load_frequency_data(self, data: Dict[str, Dict[str, float]]) -> None:
        """Load historical frequency data for evidence patterns."""
        self.frequency_data = data
    
    def calculate_likelihood_ratio(self, evidence: BayesianEvidence, 
                                 hypothesis: BayesianHypothesis) -> float:
        """Calculate likelihood ratio based on empirical frequencies."""
        # Get frequency data for this evidence-hypothesis combination
        evidence_key = self._get_evidence_key(evidence)
        hypothesis_key = self._get_hypothesis_key(hypothesis)
        
        # Calculate P(E|H) from frequency data
        p_e_given_h = self._get_frequency(evidence_key, hypothesis_key, True)
        
        # Calculate P(E|¬H) from frequency data
        p_e_given_not_h = self._get_frequency(evidence_key, hypothesis_key, False)
        
        # Apply smoothing to avoid zero probabilities
        p_e_given_h = (p_e_given_h + self.smoothing_factor) / (1.0 + 2 * self.smoothing_factor)
        p_e_given_not_h = (p_e_given_not_h + self.smoothing_factor) / (1.0 + 2 * self.smoothing_factor)
        
        # Calculate likelihood ratio
        likelihood_ratio = p_e_given_h / p_e_given_not_h
        
        return likelihood_ratio
    
    def _get_evidence_key(self, evidence: BayesianEvidence) -> str:
        """Generate key for evidence lookup."""
        return f"{evidence.evidence_type.value}_{evidence.collection_method}"
    
    def _get_hypothesis_key(self, hypothesis: BayesianHypothesis) -> str:
        """Generate key for hypothesis lookup."""
        return f"{hypothesis.hypothesis_type.value}"
    
    def _get_frequency(self, evidence_key: str, hypothesis_key: str, hypothesis_true: bool) -> float:
        """Get frequency from data with fallback strategies."""
        # Try exact match first
        lookup_key = f"{evidence_key}|{hypothesis_key}|{hypothesis_true}"
        if lookup_key in self.frequency_data:
            return self.frequency_data[lookup_key].get("frequency", 0.5)
        
        # Try evidence type + hypothesis true/false
        evidence_type = evidence_key.split("_")[0]
        general_key = f"{evidence_type}|{hypothesis_true}"
        if general_key in self.frequency_data:
            return self.frequency_data[general_key].get("frequency", 0.5)
        
        # Fall back to default based on hypothesis status
        return 0.7 if hypothesis_true else 0.3


class MechanismBasedLikelihoodCalculator:
    """
    Calculate likelihoods based on causal mechanism analysis.
    
    Uses understanding of causal mechanisms to estimate likelihood
    ratios based on mechanism plausibility and pathway strength.
    """
    
    def __init__(self, config: LikelihoodCalculationConfig):
        self.config = config
        self.mechanism_strengths: Dict[str, float] = {}
        self.pathway_probabilities: Dict[str, float] = {}
        self.mechanism_dependencies: Dict[str, List[str]] = {}
    
    def set_mechanism_data(self, strengths: Dict[str, float], 
                          pathways: Dict[str, float],
                          dependencies: Dict[str, List[str]]) -> None:
        """Set mechanism analysis data."""
        self.mechanism_strengths = strengths
        self.pathway_probabilities = pathways
        self.mechanism_dependencies = dependencies
    
    def calculate_likelihood_ratio(self, evidence: BayesianEvidence, 
                                 hypothesis: BayesianHypothesis) -> float:
        """Calculate likelihood ratio based on causal mechanisms."""
        # Identify relevant mechanisms
        relevant_mechanisms = self._identify_relevant_mechanisms(evidence, hypothesis)
        
        if not relevant_mechanisms:
            # Fall back to Van Evera calculation
            van_evera_calc = VanEveraLikelihoodCalculator(self.config)
            return van_evera_calc.calculate_likelihood_ratio(evidence, hypothesis)
        
        # Calculate mechanism-based probabilities
        p_e_given_h = self._calculate_mechanism_probability(evidence, hypothesis, relevant_mechanisms, True)
        p_e_given_not_h = self._calculate_mechanism_probability(evidence, hypothesis, relevant_mechanisms, False)
        
        # Calculate likelihood ratio
        if p_e_given_not_h == 0:
            return float('inf')
        
        return p_e_given_h / p_e_given_not_h
    
    def _identify_relevant_mechanisms(self, evidence: BayesianEvidence, 
                                    hypothesis: BayesianHypothesis) -> List[str]:
        """Identify mechanisms relevant to evidence-hypothesis combination."""
        relevant = []
        
        # Check if evidence source node connects to hypothesis mechanisms
        for mechanism_id in self.mechanism_strengths:
            if (evidence.source_node_id in mechanism_id or 
                mechanism_id in evidence.description.lower() or
                mechanism_id in hypothesis.description.lower()):
                relevant.append(mechanism_id)
        
        return relevant
    
    def _calculate_mechanism_probability(self, evidence: BayesianEvidence, 
                                       hypothesis: BayesianHypothesis,
                                       mechanisms: List[str], 
                                       hypothesis_true: bool) -> float:
        """Calculate probability based on mechanism analysis."""
        if not mechanisms:
            return 0.5
        
        # Aggregate mechanism contributions
        total_strength = 0.0
        mechanism_count = 0
        
        for mechanism_id in mechanisms:
            mechanism_strength = self.mechanism_strengths.get(mechanism_id, 0.5)
            pathway_prob = self.pathway_probabilities.get(mechanism_id, 0.5)
            
            if hypothesis_true:
                # If hypothesis is true, strong mechanisms make evidence more likely
                contribution = mechanism_strength * pathway_prob
            else:
                # If hypothesis is false, mechanisms are less relevant
                contribution = mechanism_strength * pathway_prob * 0.3
            
            total_strength += contribution
            mechanism_count += 1
        
        # Calculate average mechanism contribution
        if mechanism_count > 0:
            avg_contribution = total_strength / mechanism_count
        else:
            avg_contribution = 0.5
        
        # Adjust based on evidence strength and reliability
        final_prob = avg_contribution * evidence.strength * evidence.reliability
        
        return np.clip(final_prob, 0.01, 0.99)


class ContextualLikelihoodCalculator:
    """
    Calculate likelihoods with context sensitivity and temporal factors.
    
    Adjusts likelihood calculations based on temporal context,
    situational factors, and evidence interaction effects.
    """
    
    def __init__(self, config: LikelihoodCalculationConfig):
        self.config = config
        self.context_factors: Dict[str, float] = {}
        self.temporal_reference: Optional[datetime] = None
        self.interaction_effects: Dict[Tuple[str, str], float] = {}
    
    def set_context(self, factors: Dict[str, float], 
                   temporal_reference: Optional[datetime] = None) -> None:
        """Set contextual factors for likelihood calculation."""
        self.context_factors = factors
        self.temporal_reference = temporal_reference
    
    def set_interaction_effects(self, effects: Dict[Tuple[str, str], float]) -> None:
        """Set evidence interaction effects."""
        self.interaction_effects = effects
    
    def calculate_likelihood_ratio(self, evidence: BayesianEvidence, 
                                 hypothesis: BayesianHypothesis,
                                 other_evidence: List[BayesianEvidence] = None) -> float:
        """Calculate context-sensitive likelihood ratio."""
        # Start with base Van Evera calculation
        base_calc = VanEveraLikelihoodCalculator(self.config)
        base_ratio = base_calc.calculate_likelihood_ratio(evidence, hypothesis)
        
        # Apply temporal decay
        temporal_factor = self._calculate_temporal_factor(evidence)
        
        # Apply context adjustments
        context_factor = self._calculate_context_factor(evidence, hypothesis)
        
        # Apply interaction effects
        interaction_factor = self._calculate_interaction_factor(evidence, other_evidence or [])
        
        # Combine all factors
        adjusted_ratio = base_ratio * temporal_factor * context_factor * interaction_factor
        
        return max(0.01, adjusted_ratio)
    
    def _calculate_temporal_factor(self, evidence: BayesianEvidence) -> float:
        """Calculate temporal decay factor for evidence."""
        if not evidence.timestamp or not self.temporal_reference:
            return 1.0
        
        # Calculate time difference in days
        time_diff = abs((self.temporal_reference - evidence.timestamp).days)
        
        # Apply exponential decay
        decay_rate = self.config.temporal_decay
        temporal_factor = math.exp(-decay_rate * time_diff / 365.0)  # Yearly decay
        
        return max(0.1, temporal_factor)  # Minimum 10% strength retained
    
    def _calculate_context_factor(self, evidence: BayesianEvidence, 
                                 hypothesis: BayesianHypothesis) -> float:
        """Calculate context-based adjustment factor."""
        context_sensitivity = self.config.context_sensitivity
        
        if context_sensitivity == 0 or not self.context_factors:
            return 1.0
        
        # Use semantic analysis to identify relevant context factors
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        relevant_factors = []
        for factor_name, factor_value in self.context_factors.items():
            # Assess semantic relevance of context factor
            assessment = semantic_service.assess_probative_value(
                evidence_description=evidence.description,
                hypothesis_description=f"Context factor '{factor_name}' is relevant to: {hypothesis.description}",
                context="Identifying relevant context factors for likelihood calculation"
            )
            if assessment.confidence_score > 0.6:
                relevant_factors.append(factor_value)
        
        if not relevant_factors:
            return 1.0
        
        # Calculate context adjustment
        avg_factor = np.mean(relevant_factors)
        context_adjustment = 1.0 + context_sensitivity * (avg_factor - 0.5)
        
        return max(0.1, context_adjustment)
    
    def _calculate_interaction_factor(self, evidence: BayesianEvidence, 
                                    other_evidence: List[BayesianEvidence]) -> float:
        """Calculate evidence interaction effects."""
        if not other_evidence or not self.interaction_effects:
            return 1.0
        
        interaction_adjustments = []
        
        for other_ev in other_evidence:
            # Check for interaction effects
            interaction_key = (evidence.evidence_id, other_ev.evidence_id)
            reverse_key = (other_ev.evidence_id, evidence.evidence_id)
            
            if interaction_key in self.interaction_effects:
                interaction_adjustments.append(self.interaction_effects[interaction_key])
            elif reverse_key in self.interaction_effects:
                interaction_adjustments.append(self.interaction_effects[reverse_key])
        
        if not interaction_adjustments:
            return 1.0
        
        # Combine interaction effects (multiplicative)
        interaction_factor = 1.0
        for adjustment in interaction_adjustments:
            interaction_factor *= adjustment
        
        return max(0.1, interaction_factor)


class LikelihoodCalculationOrchestrator:
    """
    Orchestrates likelihood calculations using multiple methods.
    
    Provides unified interface for likelihood calculation with method
    selection, uncertainty quantification, and sensitivity analysis.
    """
    
    def __init__(self):
        self.calculators = {
            LikelihoodCalculationMethod.VAN_EVERA: VanEveraLikelihoodCalculator,
            LikelihoodCalculationMethod.FREQUENCY_BASED: FrequencyBasedLikelihoodCalculator,
            LikelihoodCalculationMethod.MECHANISM_BASED: MechanismBasedLikelihoodCalculator,
            LikelihoodCalculationMethod.CONTEXTUAL: ContextualLikelihoodCalculator
        }
        
        self.calculation_history: List[Dict[str, Any]] = []
    
    def calculate_likelihood_ratio(self, evidence: BayesianEvidence,
                                 hypothesis: BayesianHypothesis,
                                 config: LikelihoodCalculationConfig,
                                 additional_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate likelihood ratio using specified method."""
        if config.method not in self.calculators:
            raise ValueError(f"Unsupported likelihood calculation method: {config.method}")
        
        calculator_class = self.calculators[config.method]
        calculator = calculator_class(config)
        
        # Set additional data if provided
        if additional_data:
            if hasattr(calculator, 'load_frequency_data') and 'frequency_data' in additional_data:
                calculator.load_frequency_data(additional_data['frequency_data'])
            
            if hasattr(calculator, 'set_mechanism_data') and 'mechanism_data' in additional_data:
                mech_data = additional_data['mechanism_data']
                calculator.set_mechanism_data(
                    mech_data.get('strengths', {}),
                    mech_data.get('pathways', {}),
                    mech_data.get('dependencies', {})
                )
            
            if hasattr(calculator, 'set_context') and 'context_data' in additional_data:
                ctx_data = additional_data['context_data']
                calculator.set_context(
                    ctx_data.get('factors', {}),
                    ctx_data.get('temporal_reference')
                )
        
        # Calculate likelihood ratio
        likelihood_ratio = calculator.calculate_likelihood_ratio(evidence, hypothesis)
        
        # Record calculation
        calculation_record = {
            "timestamp": datetime.now().isoformat(),
            "method": config.method.value,
            "evidence_id": evidence.evidence_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "likelihood_ratio": likelihood_ratio,
            "evidence_type": evidence.evidence_type.value,
            "hypothesis_type": hypothesis.hypothesis_type.value,
            "config": {
                "van_evera_strictness": config.van_evera_strictness,
                "uncertainty_factor": config.uncertainty_factor,
                "context_sensitivity": config.context_sensitivity
            }
        }
        
        self.calculation_history.append(calculation_record)
        
        return likelihood_ratio
    
    def batch_calculate_likelihood_ratios(self, evidence_list: List[BayesianEvidence],
                                        hypothesis: BayesianHypothesis,
                                        config: LikelihoodCalculationConfig,
                                        additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate likelihood ratios for multiple evidence pieces."""
        ratios = {}
        
        for evidence in evidence_list:
            ratio = self.calculate_likelihood_ratio(evidence, hypothesis, config, additional_data)
            ratios[evidence.evidence_id] = ratio
        
        return ratios
    
    def uncertainty_analysis(self, evidence: BayesianEvidence,
                           hypothesis: BayesianHypothesis,
                           base_config: LikelihoodCalculationConfig,
                           uncertainty_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Perform uncertainty analysis on likelihood calculations."""
        base_ratio = self.calculate_likelihood_ratio(evidence, hypothesis, base_config)
        
        uncertainty_results = {
            "base_likelihood_ratio": base_ratio,
            "parameter_uncertainty": {},
            "confidence_interval": None,
            "sensitivity_score": 0.0
        }
        
        # Test different parameter values
        ratios = []
        
        for param_name, (min_val, max_val) in uncertainty_ranges.items():
            param_ratios = []
            test_values = np.linspace(min_val, max_val, 10)
            
            for value in test_values:
                modified_config = LikelihoodCalculationConfig(
                    method=base_config.method,
                    van_evera_strictness=base_config.van_evera_strictness,
                    uncertainty_factor=base_config.uncertainty_factor,
                    context_sensitivity=base_config.context_sensitivity
                )
                
                # Update specific parameter
                setattr(modified_config, param_name, value)
                
                ratio = self.calculate_likelihood_ratio(evidence, hypothesis, modified_config)
                param_ratios.append(ratio)
                ratios.append(ratio)
            
            uncertainty_results["parameter_uncertainty"][param_name] = {
                "min_ratio": min(param_ratios),
                "max_ratio": max(param_ratios),
                "std_ratio": np.std(param_ratios)
            }
        
        # Calculate overall confidence interval
        if ratios:
            uncertainty_results["confidence_interval"] = (
                np.percentile(ratios, 5),  # 5th percentile
                np.percentile(ratios, 95)  # 95th percentile
            )
            
            # Calculate sensitivity score (coefficient of variation)
            uncertainty_results["sensitivity_score"] = np.std(ratios) / np.mean(ratios)
        
        return uncertainty_results
    
    def export_calculation_history(self, file_path: Union[str, Path]) -> None:
        """Export calculation history to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.calculation_history, f, indent=2, ensure_ascii=False)
    
    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of likelihood calculations performed."""
        if not self.calculation_history:
            return {"total_calculations": 0}
        
        methods_used = [record["method"] for record in self.calculation_history]
        method_counts = {method: methods_used.count(method) for method in set(methods_used)}
        
        likelihood_ratios = [record["likelihood_ratio"] for record in self.calculation_history]
        
        return {
            "total_calculations": len(self.calculation_history),
            "methods_used": method_counts,
            "likelihood_ratio_stats": {
                "mean": np.mean(likelihood_ratios),
                "std": np.std(likelihood_ratios),
                "min": min(likelihood_ratios),
                "max": max(likelihood_ratios)
            },
            "latest_calculation": self.calculation_history[-1]["timestamp"],
            "unique_evidence": len(set(record["evidence_id"] for record in self.calculation_history)),
            "unique_hypotheses": len(set(record["hypothesis_id"] for record in self.calculation_history))
        }