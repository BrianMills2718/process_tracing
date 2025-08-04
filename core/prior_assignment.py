"""
Prior probability assignment algorithms for Bayesian process tracing.

Implements various methods for assigning prior probabilities to causal hypotheses
based on theoretical frameworks, empirical data, and domain knowledge.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import networkx as nx
from datetime import datetime
import json
from pathlib import Path

from .bayesian_models import (
    BayesianHypothesis, BayesianHypothesisSpace, HypothesisType, PriorType
)


class PriorAssignmentMethod(Enum):
    """Methods for assigning prior probabilities."""
    UNIFORM = "uniform"                    # Equal probability for all hypotheses
    FREQUENCY_BASED = "frequency_based"    # Based on historical frequency
    THEORY_GUIDED = "theory_guided"        # Based on theoretical predictions
    COMPLEXITY_PENALIZED = "complexity_penalized"  # Occam's razor - simpler is more likely
    EXPERT_ELICITED = "expert_elicited"    # Based on expert judgment
    HIERARCHICAL = "hierarchical"          # Based on hypothesis hierarchy
    MECHANISM_BASED = "mechanism_based"    # Based on causal mechanism plausibility
    CONTEXT_SENSITIVE = "context_sensitive"  # Adapted to specific context


@dataclass
class PriorAssignmentConfig:
    """Configuration for prior assignment algorithms."""
    method: PriorAssignmentMethod
    parameters: Dict[str, Any]
    confidence_weight: float = 1.0
    theoretical_framework: Optional[str] = None
    domain_expertise_level: float = 0.5
    historical_data_availability: float = 0.0
    context_specificity: float = 0.5


class UniformPriorAssigner:
    """
    Assigns uniform prior probabilities to all hypotheses.
    
    Implements the principle of indifference where all hypotheses
    receive equal probability in the absence of prior information.
    """
    
    def __init__(self, config: PriorAssignmentConfig):
        self.config = config
        self.method_name = "Uniform Prior Assignment"
    
    def assign_priors(self, hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Assign uniform priors to all hypotheses in the space."""
        hypotheses = list(hypothesis_space.hypotheses.keys())
        
        if not hypotheses:
            return {}
        
        # Check for mutual exclusivity groups
        prior_assignments = {}
        assigned_hypotheses = set()
        
        # Handle mutual exclusivity groups first
        for group in hypothesis_space.mutual_exclusivity_groups:
            group_hypotheses = [h for h in group if h in hypotheses]
            if group_hypotheses:
                uniform_prob = 1.0 / len(group_hypotheses)
                for hyp_id in group_hypotheses:
                    prior_assignments[hyp_id] = uniform_prob
                    assigned_hypotheses.add(hyp_id)
        
        # Handle remaining hypotheses
        remaining_hypotheses = [h for h in hypotheses if h not in assigned_hypotheses]
        if remaining_hypotheses:
            if hypothesis_space.collective_exhaustiveness:
                # Remaining probability divided among remaining hypotheses
                remaining_prob = 1.0 - sum(prior_assignments.values())
                uniform_prob = remaining_prob / len(remaining_hypotheses)
            else:
                # Independent uniform assignment
                uniform_prob = 1.0 / len(hypotheses)
            
            for hyp_id in remaining_hypotheses:
                prior_assignments[hyp_id] = uniform_prob
        
        return prior_assignments
    
    def get_assignment_rationale(self) -> str:
        """Get rationale for uniform prior assignment."""
        return (
            "Uniform priors assigned based on principle of indifference. "
            "Equal probability assigned to all hypotheses in absence of "
            "distinguishing prior information."
        )


class FrequencyBasedPriorAssigner:
    """
    Assigns priors based on historical frequency of similar causal patterns.
    
    Uses empirical data about causal mechanism frequency to inform
    prior probability assignments.
    """
    
    def __init__(self, config: PriorAssignmentConfig):
        self.config = config
        self.method_name = "Frequency-Based Prior Assignment"
        self.historical_frequencies: Dict[str, float] = config.parameters.get("historical_frequencies", {})
        self.smoothing_factor: float = config.parameters.get("smoothing_factor", 0.1)
        self.default_frequency: float = config.parameters.get("default_frequency", 0.1)
    
    def assign_priors(self, hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Assign priors based on historical frequencies."""
        prior_assignments = {}
        
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            # Get historical frequency for this hypothesis type
            frequency = self._get_historical_frequency(hypothesis)
            
            # Apply smoothing to avoid zero probabilities
            smoothed_frequency = (frequency + self.smoothing_factor) / (1.0 + self.smoothing_factor)
            
            prior_assignments[hyp_id] = smoothed_frequency
        
        # Normalize to ensure coherence
        return self._normalize_priors(prior_assignments, hypothesis_space)
    
    def _get_historical_frequency(self, hypothesis: BayesianHypothesis) -> float:
        """Get historical frequency for a hypothesis."""
        # Try exact match first
        if hypothesis.hypothesis_id in self.historical_frequencies:
            return self.historical_frequencies[hypothesis.hypothesis_id]
        
        # Try hypothesis type match
        type_key = f"type_{hypothesis.hypothesis_type.value}"
        if type_key in self.historical_frequencies:
            return self.historical_frequencies[type_key]
        
        # Check description keywords
        description_lower = hypothesis.description.lower()
        for pattern, frequency in self.historical_frequencies.items():
            if pattern.lower() in description_lower:
                return frequency
        
        # Return default frequency
        return self.default_frequency
    
    def _normalize_priors(self, priors: Dict[str, float], hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Normalize priors to ensure probability coherence."""
        # Handle mutual exclusivity groups
        for group in hypothesis_space.mutual_exclusivity_groups:
            group_priors = {h: priors.get(h, 0.0) for h in group if h in priors}
            if group_priors:
                total = sum(group_priors.values())
                if total > 0:
                    for hyp_id in group_priors:
                        priors[hyp_id] = group_priors[hyp_id] / total
        
        # Global normalization if collective exhaustiveness
        if hypothesis_space.collective_exhaustiveness:
            total = sum(priors.values())
            if total > 0:
                priors = {h: p / total for h, p in priors.items()}
        
        return priors
    
    def get_assignment_rationale(self) -> str:
        """Get rationale for frequency-based assignment."""
        return (
            f"Priors assigned based on historical frequencies of similar causal patterns. "
            f"Smoothing factor: {self.smoothing_factor}, Default frequency: {self.default_frequency}. "
            f"Available frequency data for {len(self.historical_frequencies)} patterns."
        )


class TheoryGuidedPriorAssigner:
    """
    Assigns priors based on theoretical predictions and domain knowledge.
    
    Uses established theoretical frameworks to inform prior probability
    assignments for different types of causal hypotheses.
    """
    
    def __init__(self, config: PriorAssignmentConfig):
        self.config = config
        self.method_name = "Theory-Guided Prior Assignment"
        self.theoretical_framework = config.theoretical_framework
        self.theory_weights: Dict[str, float] = config.parameters.get("theory_weights", {})
        self.mechanism_plausibility: Dict[str, float] = config.parameters.get("mechanism_plausibility", {})
        self.domain_expertise = config.domain_expertise_level
    
    def assign_priors(self, hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Assign priors based on theoretical guidance."""
        prior_assignments = {}
        
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            # Get theory-based prior
            theory_prior = self._get_theory_based_prior(hypothesis)
            
            # Adjust based on domain expertise confidence
            adjusted_prior = self._adjust_for_expertise(theory_prior, hypothesis)
            
            prior_assignments[hyp_id] = adjusted_prior
        
        return self._normalize_priors(prior_assignments, hypothesis_space)
    
    def _get_theory_based_prior(self, hypothesis: BayesianHypothesis) -> float:
        """Get theory-based prior for a hypothesis."""
        base_prior = 0.5  # Start with neutral
        
        # Adjust based on hypothesis type
        type_adjustments = {
            HypothesisType.PRIMARY: 0.6,      # Slightly favor primary hypotheses
            HypothesisType.ALTERNATIVE: 0.4,   # Lower probability for alternatives
            HypothesisType.NULL: 0.3,         # Conservative for null hypotheses
            HypothesisType.COMPOSITE: 0.4,    # Complex hypotheses less likely a priori
            HypothesisType.NECESSARY: 0.7,    # Necessary causes often important
            HypothesisType.SUFFICIENT: 0.6    # Sufficient causes moderately likely
        }
        
        base_prior = type_adjustments.get(hypothesis.hypothesis_type, base_prior)
        
        # Adjust based on mechanism plausibility
        if hypothesis.hypothesis_id in self.mechanism_plausibility:
            plausibility = self.mechanism_plausibility[hypothesis.hypothesis_id]
            base_prior = 0.5 * base_prior + 0.5 * plausibility
        
        # Adjust based on theoretical framework weights
        for theory_pattern, weight in self.theory_weights.items():
            if theory_pattern.lower() in hypothesis.description.lower():
                base_prior = base_prior * weight
                break
        
        return np.clip(base_prior, 0.01, 0.99)  # Avoid extreme values
    
    def _adjust_for_expertise(self, theory_prior: float, hypothesis: BayesianHypothesis) -> float:
        """Adjust prior based on domain expertise confidence."""
        if self.domain_expertise < 0.5:
            # Low expertise - move towards uniform (0.5)
            adjustment_factor = 1.0 - self.domain_expertise
            return theory_prior + adjustment_factor * (0.5 - theory_prior)
        else:
            # High expertise - strengthen theory-based assignment
            return theory_prior
    
    def _normalize_priors(self, priors: Dict[str, float], hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Normalize priors maintaining theoretical relationships."""
        # Similar to frequency-based normalization but preserve relative theory weights
        for group in hypothesis_space.mutual_exclusivity_groups:
            group_priors = {h: priors.get(h, 0.0) for h in group if h in priors}
            if group_priors:
                total = sum(group_priors.values())
                if total > 0:
                    for hyp_id in group_priors:
                        priors[hyp_id] = group_priors[hyp_id] / total
        
        if hypothesis_space.collective_exhaustiveness:
            total = sum(priors.values())
            if total > 0:
                priors = {h: p / total for h, p in priors.items()}
        
        return priors
    
    def get_assignment_rationale(self) -> str:
        """Get rationale for theory-guided assignment."""
        return (
            f"Priors assigned based on {self.theoretical_framework or 'general'} theoretical framework. "
            f"Domain expertise level: {self.domain_expertise:.2f}. "
            f"Theory weights applied for {len(self.theory_weights)} patterns. "
            f"Mechanism plausibility data available for {len(self.mechanism_plausibility)} hypotheses."
        )


class ComplexityPenalizedPriorAssigner:
    """
    Assigns priors with Occam's razor - simpler explanations get higher priors.
    
    Implements complexity penalty based on hypothesis structure,
    number of required elements, and explanatory burden.
    """
    
    def __init__(self, config: PriorAssignmentConfig):
        self.config = config
        self.method_name = "Complexity-Penalized Prior Assignment"
        self.complexity_penalty: float = config.parameters.get("complexity_penalty", 0.1)
        self.base_prior: float = config.parameters.get("base_prior", 0.5)
        self.max_complexity_score: float = config.parameters.get("max_complexity", 10.0)
    
    def assign_priors(self, hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Assign priors with complexity penalty."""
        prior_assignments = {}
        
        # Calculate complexity scores for all hypotheses
        complexity_scores = {}
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            complexity_scores[hyp_id] = self._calculate_complexity(hypothesis, hypothesis_space)
        
        # Assign priors with complexity penalty
        for hyp_id, complexity in complexity_scores.items():
            penalty = self.complexity_penalty * (complexity / self.max_complexity_score)
            prior = self.base_prior * (1.0 - penalty)
            prior_assignments[hyp_id] = max(0.01, prior)  # Avoid zero probability
        
        return self._normalize_priors(prior_assignments, hypothesis_space)
    
    def _calculate_complexity(self, hypothesis: BayesianHypothesis, 
                            hypothesis_space: BayesianHypothesisSpace) -> float:
        """Calculate complexity score for a hypothesis."""
        complexity = 0.0
        
        # Penalty for composite hypotheses
        if hypothesis.hypothesis_type == HypothesisType.COMPOSITE:
            complexity += 2.0
        
        # Penalty for number of required evidence pieces
        complexity += len(hypothesis.required_evidence) * 0.5
        
        # Penalty for number of child hypotheses (hierarchical complexity)
        complexity += len(hypothesis.child_hypotheses) * 1.0
        
        # Penalty for temporal constraints
        complexity += len(hypothesis.temporal_constraints) * 0.3
        
        # Penalty based on description length (proxy for conceptual complexity)
        description_complexity = len(hypothesis.description.split()) / 10.0
        complexity += description_complexity
        
        # Bonus for hypotheses with strong theoretical support
        if hypothesis.hypothesis_type in [HypothesisType.NECESSARY, HypothesisType.SUFFICIENT]:
            complexity -= 1.0  # Reduce complexity for theoretically grounded types
        
        return max(0.0, complexity)
    
    def _normalize_priors(self, priors: Dict[str, float], hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Normalize priors while preserving complexity relationships."""
        # Standard normalization with preservation of relative complexity penalties
        for group in hypothesis_space.mutual_exclusivity_groups:
            group_priors = {h: priors.get(h, 0.0) for h in group if h in priors}
            if group_priors:
                total = sum(group_priors.values())
                if total > 0:
                    for hyp_id in group_priors:
                        priors[hyp_id] = group_priors[hyp_id] / total
        
        if hypothesis_space.collective_exhaustiveness:
            total = sum(priors.values())
            if total > 0:
                priors = {h: p / total for h, p in priors.items()}
        
        return priors
    
    def get_assignment_rationale(self) -> str:
        """Get rationale for complexity-penalized assignment."""
        return (
            f"Priors assigned with Occam's razor principle - simpler explanations favored. "
            f"Complexity penalty factor: {self.complexity_penalty}. "
            f"Base prior: {self.base_prior}. "
            f"Maximum complexity threshold: {self.max_complexity_score}."
        )


class HierarchicalPriorAssigner:
    """
    Assigns priors based on hypothesis hierarchy and parent-child relationships.
    
    Uses hierarchical structure to propagate probability from parent
    hypotheses to children, maintaining coherent probability distributions.
    """
    
    def __init__(self, config: PriorAssignmentConfig):
        self.config = config
        self.method_name = "Hierarchical Prior Assignment"
        self.parent_influence: float = config.parameters.get("parent_influence", 0.7)
        self.level_discount: float = config.parameters.get("level_discount", 0.1)
        self.root_prior: float = config.parameters.get("root_prior", 0.5)
    
    def assign_priors(self, hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Assign priors based on hierarchical structure."""
        prior_assignments = {}
        
        # Find root nodes (no parents)
        root_nodes = [hyp_id for hyp_id, hyp in hypothesis_space.hypotheses.items() 
                     if not hyp.parent_hypothesis]
        
        # Assign root priors
        if root_nodes:
            root_prior_each = self.root_prior / len(root_nodes)
            for root_id in root_nodes:
                prior_assignments[root_id] = root_prior_each
        
        # Propagate priors down the hierarchy
        self._propagate_priors_down(hypothesis_space, prior_assignments)
        
        return self._normalize_priors(prior_assignments, hypothesis_space)
    
    def _propagate_priors_down(self, hypothesis_space: BayesianHypothesisSpace, 
                              priors: Dict[str, float]) -> None:
        """Propagate priors from parents to children."""
        # Get hierarchy levels
        levels = {}
        for hyp_id in hypothesis_space.hypotheses:
            levels[hyp_id] = hypothesis_space.get_hierarchy_level(hyp_id)
        
        # Process each level from top to bottom
        max_level = max(levels.values()) if levels else 0
        
        for level in range(1, max_level + 1):
            level_hypotheses = [hyp_id for hyp_id, lev in levels.items() if lev == level]
            
            for hyp_id in level_hypotheses:
                hypothesis = hypothesis_space.hypotheses[hyp_id]
                parent_id = hypothesis.parent_hypothesis
                
                if parent_id and parent_id in priors:
                    # Calculate child prior based on parent
                    parent_prior = priors[parent_id]
                    
                    # Get siblings (other children of same parent)
                    siblings = [h for h in hypothesis_space.hypotheses.values() 
                              if h.parent_hypothesis == parent_id]
                    
                    # Distribute parent probability among children
                    inheritance_factor = self.parent_influence * (1.0 - self.level_discount * level)
                    child_share = (parent_prior * inheritance_factor) / len(siblings)
                    
                    priors[hyp_id] = child_share
                else:
                    # Orphaned hypothesis - assign base prior
                    priors[hyp_id] = 0.1
    
    def _normalize_priors(self, priors: Dict[str, float], hypothesis_space: BayesianHypothesisSpace) -> Dict[str, float]:
        """Normalize priors preserving hierarchical relationships."""
        # Normalize within each level of hierarchy
        levels = {}
        for hyp_id in priors:
            levels[hyp_id] = hypothesis_space.get_hierarchy_level(hyp_id)
        
        # Group by level and normalize
        level_groups = {}
        for hyp_id, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(hyp_id)
        
        # Standard mutual exclusivity normalization
        for group in hypothesis_space.mutual_exclusivity_groups:
            group_priors = {h: priors.get(h, 0.0) for h in group if h in priors}
            if group_priors:
                total = sum(group_priors.values())
                if total > 0:
                    for hyp_id in group_priors:
                        priors[hyp_id] = group_priors[hyp_id] / total
        
        # Global normalization if collective exhaustiveness
        if hypothesis_space.collective_exhaustiveness:
            total = sum(priors.values())
            if total > 0:
                priors = {h: p / total for h, p in priors.items()}
        
        return priors
    
    def get_assignment_rationale(self) -> str:
        """Get rationale for hierarchical assignment."""
        return (
            f"Priors assigned based on hypothesis hierarchy. "
            f"Parent influence factor: {self.parent_influence}. "
            f"Level discount factor: {self.level_discount}. "
            f"Root prior: {self.root_prior}."
        )


class PriorAssignmentOrchestrator:
    """
    Orchestrates prior assignment using multiple methods and combining results.
    
    Provides unified interface for prior assignment with method selection,
    combination strategies, and sensitivity analysis.
    """
    
    def __init__(self):
        self.assigners = {
            PriorAssignmentMethod.UNIFORM: UniformPriorAssigner,
            PriorAssignmentMethod.FREQUENCY_BASED: FrequencyBasedPriorAssigner,
            PriorAssignmentMethod.THEORY_GUIDED: TheoryGuidedPriorAssigner,
            PriorAssignmentMethod.COMPLEXITY_PENALIZED: ComplexityPenalizedPriorAssigner,
            PriorAssignmentMethod.HIERARCHICAL: HierarchicalPriorAssigner
        }
        
        self.assignment_history: List[Dict[str, Any]] = []
    
    def assign_priors(self, hypothesis_space: BayesianHypothesisSpace, 
                     config: PriorAssignmentConfig) -> Dict[str, float]:
        """Assign priors using specified method."""
        if config.method not in self.assigners:
            raise ValueError(f"Unsupported prior assignment method: {config.method}")
        
        assigner_class = self.assigners[config.method]
        assigner = assigner_class(config)
        
        priors = assigner.assign_priors(hypothesis_space)
        
        # Record assignment
        assignment_record = {
            "timestamp": datetime.now().isoformat(),
            "method": config.method.value,
            "hypothesis_space_id": hypothesis_space.hypothesis_space_id,
            "priors": priors.copy(),
            "config": {
                "parameters": config.parameters,
                "confidence_weight": config.confidence_weight,
                "theoretical_framework": config.theoretical_framework
            },
            "rationale": assigner.get_assignment_rationale()
        }
        
        self.assignment_history.append(assignment_record)
        
        # Apply priors to hypotheses
        for hyp_id, prior in priors.items():
            if hyp_id in hypothesis_space.hypotheses:
                hypothesis_space.hypotheses[hyp_id].prior_probability = prior
                hypothesis_space.hypotheses[hyp_id].posterior_probability = prior  # Initialize posterior
        
        return priors
    
    def combine_multiple_assignments(self, hypothesis_space: BayesianHypothesisSpace,
                                   configs: List[PriorAssignmentConfig],
                                   weights: Optional[List[float]] = None) -> Dict[str, float]:
        """Combine prior assignments from multiple methods."""
        if not configs:
            raise ValueError("At least one configuration required")
        
        if weights is None:
            weights = [1.0 / len(configs)] * len(configs)
        
        if len(weights) != len(configs):
            raise ValueError("Number of weights must match number of configurations")
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Get assignments from each method
        all_assignments = []
        for config in configs:
            assignment = self.assign_priors(hypothesis_space, config)
            all_assignments.append(assignment)
        
        # Combine assignments using weighted average
        combined_priors = {}
        all_hyp_ids = set()
        for assignment in all_assignments:
            all_hyp_ids.update(assignment.keys())
        
        for hyp_id in all_hyp_ids:
            weighted_sum = 0.0
            for assignment, weight in zip(all_assignments, weights):
                prior = assignment.get(hyp_id, 0.0)
                weighted_sum += prior * weight
            combined_priors[hyp_id] = weighted_sum
        
        # Apply combined priors
        for hyp_id, prior in combined_priors.items():
            if hyp_id in hypothesis_space.hypotheses:
                hypothesis_space.hypotheses[hyp_id].prior_probability = prior
                hypothesis_space.hypotheses[hyp_id].posterior_probability = prior
        
        return combined_priors
    
    def sensitivity_analysis(self, hypothesis_space: BayesianHypothesisSpace,
                           base_config: PriorAssignmentConfig,
                           parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis on prior assignments."""
        base_priors = self.assign_priors(hypothesis_space, base_config)
        
        sensitivity_results = {
            "base_priors": base_priors.copy(),
            "parameter_sensitivity": {},
            "max_deviation": 0.0,
            "most_sensitive_hypothesis": None,
            "most_sensitive_parameter": None
        }
        
        for param_name, values in parameter_ranges.items():
            param_results = []
            
            for value in values:
                # Create modified config
                modified_config = PriorAssignmentConfig(
                    method=base_config.method,
                    parameters=base_config.parameters.copy(),
                    confidence_weight=base_config.confidence_weight,
                    theoretical_framework=base_config.theoretical_framework
                )
                modified_config.parameters[param_name] = value
                
                # Get modified priors
                modified_priors = self.assign_priors(hypothesis_space, modified_config)
                
                # Calculate deviations
                deviations = {}
                for hyp_id in base_priors:
                    if hyp_id in modified_priors:
                        deviation = abs(modified_priors[hyp_id] - base_priors[hyp_id])
                        deviations[hyp_id] = deviation
                
                max_deviation = max(deviations.values()) if deviations else 0.0
                
                param_results.append({
                    "parameter_value": value,
                    "priors": modified_priors,
                    "deviations": deviations,
                    "max_deviation": max_deviation
                })
                
                # Track global maximum deviation
                if max_deviation > sensitivity_results["max_deviation"]:
                    sensitivity_results["max_deviation"] = max_deviation
                    sensitivity_results["most_sensitive_parameter"] = param_name
                    max_hyp = max(deviations.items(), key=lambda x: x[1])
                    sensitivity_results["most_sensitive_hypothesis"] = max_hyp[0]
            
            sensitivity_results["parameter_sensitivity"][param_name] = param_results
        
        return sensitivity_results
    
    def export_assignment_history(self, file_path: Union[str, Path]) -> None:
        """Export assignment history to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.assignment_history, f, indent=2, ensure_ascii=False)
    
    def get_assignment_summary(self) -> Dict[str, Any]:
        """Get summary of all prior assignments performed."""
        if not self.assignment_history:
            return {"total_assignments": 0}
        
        methods_used = [record["method"] for record in self.assignment_history]
        method_counts = {method: methods_used.count(method) for method in set(methods_used)}
        
        return {
            "total_assignments": len(self.assignment_history),
            "methods_used": method_counts,
            "latest_assignment": self.assignment_history[-1]["timestamp"],
            "unique_hypothesis_spaces": len(set(record["hypothesis_space_id"] for record in self.assignment_history))
        }