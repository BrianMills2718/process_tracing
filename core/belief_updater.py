"""
Bayesian belief updating engine for process tracing.

Implements systematic belief updating using Bayes' theorem with support
for multiple evidence integration, hypothesis space management, and
convergence analysis.
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
from collections import defaultdict

from .bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace, 
    BayesianProcessTracingModel, EvidenceType, HypothesisType
)
from .likelihood_calculator import LikelihoodCalculationOrchestrator, LikelihoodCalculationConfig


class UpdateMethod(Enum):
    """Methods for belief updating."""
    SEQUENTIAL = "sequential"              # Update with one evidence piece at a time
    BATCH = "batch"                       # Update with all evidence simultaneously
    HIERARCHICAL = "hierarchical"         # Update hierarchically from top to bottom
    ITERATIVE = "iterative"               # Iterative refinement until convergence
    WEIGHTED = "weighted"                 # Weighted evidence integration
    CONDITIONAL = "conditional"           # Conditional on evidence dependencies


class ConvergenceMethod(Enum):
    """Methods for assessing convergence."""
    PROBABILITY_CHANGE = "probability_change"    # Based on probability changes
    LIKELIHOOD_RATIO = "likelihood_ratio"       # Based on likelihood ratio changes
    ENTROPY = "entropy"                          # Based on entropy reduction
    HYPOTHESIS_RANKING = "hypothesis_ranking"    # Based on ranking stability


@dataclass
class BeliefUpdateConfig:
    """Configuration for belief updating."""
    update_method: UpdateMethod = UpdateMethod.SEQUENTIAL
    convergence_method: ConvergenceMethod = ConvergenceMethod.PROBABILITY_CHANGE
    convergence_threshold: float = 0.01
    max_iterations: int = 100
    evidence_independence: bool = True
    temporal_weighting: bool = True
    confidence_weighting: bool = True
    normalize_after_each_update: bool = True
    track_update_history: bool = True


@dataclass
class UpdateResult:
    """Result of a belief update operation."""
    success: bool
    iterations: int
    convergence_achieved: bool
    final_probabilities: Dict[str, float]
    probability_changes: Dict[str, float]
    likelihood_ratios_used: Dict[str, float]
    evidence_processed: List[str]
    update_timestamp: datetime = field(default_factory=datetime.now)
    convergence_score: float = 0.0
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)


class SequentialBeliefUpdater:
    """
    Updates beliefs sequentially with one evidence piece at a time.
    
    Implements classic Bayesian updating where evidence is processed
    in order, updating posterior probabilities incrementally.
    """
    
    def __init__(self, config: BeliefUpdateConfig):
        self.config = config
        self.likelihood_calculator = LikelihoodCalculationOrchestrator()
    
    def update_beliefs(self, hypothesis_space: BayesianHypothesisSpace,
                      evidence_list: List[BayesianEvidence],
                      likelihood_config: LikelihoodCalculationConfig) -> UpdateResult:
        """Update beliefs sequentially with evidence list."""
        initial_probs = {h_id: h.posterior_probability 
                        for h_id, h in hypothesis_space.hypotheses.items()}
        
        processed_evidence = []
        likelihood_ratios = {}
        iteration = 0
        
        # Sort evidence by temporal order if available
        sorted_evidence = self._sort_evidence_temporally(evidence_list)
        
        for evidence in sorted_evidence:
            iteration += 1
            
            # Calculate likelihood ratios for all hypotheses
            evidence_ratios = {}
            for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
                ratio = self.likelihood_calculator.calculate_likelihood_ratio(
                    evidence, hypothesis, likelihood_config
                )
                evidence_ratios[hyp_id] = ratio
                likelihood_ratios[f"{evidence.evidence_id}_{hyp_id}"] = ratio
            
            # Update posterior probabilities using Bayes' theorem
            self._update_posteriors_with_evidence(hypothesis_space, evidence_ratios)
            
            # Normalize probabilities if required
            if self.config.normalize_after_each_update:
                hypothesis_space._normalize_probabilities()
            
            processed_evidence.append(evidence.evidence_id)
            
            # Record update in hypothesis history
            for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
                if self.config.track_update_history:
                    hypothesis.update_posterior(
                        hypothesis.posterior_probability,
                        evidence.evidence_id,
                        evidence_ratios[hyp_id]
                    )
        
        # Calculate final results
        final_probs = {h_id: h.posterior_probability 
                      for h_id, h in hypothesis_space.hypotheses.items()}
        
        prob_changes = {h_id: abs(final_probs[h_id] - initial_probs[h_id])
                       for h_id in initial_probs}
        
        convergence_score = self._calculate_convergence_score(prob_changes)
        convergence_achieved = convergence_score < self.config.convergence_threshold
        
        return UpdateResult(
            success=True,
            iterations=iteration,
            convergence_achieved=convergence_achieved,
            final_probabilities=final_probs,
            probability_changes=prob_changes,
            likelihood_ratios_used=likelihood_ratios,
            evidence_processed=processed_evidence,
            convergence_score=convergence_score
        )
    
    def _sort_evidence_temporally(self, evidence_list: List[BayesianEvidence]) -> List[BayesianEvidence]:
        """Sort evidence by temporal order."""
        if not self.config.temporal_weighting:
            return evidence_list
        
        # Separate evidence with and without timestamps
        with_timestamps = [e for e in evidence_list if e.timestamp]
        without_timestamps = [e for e in evidence_list if not e.timestamp]
        
        # Sort timestamped evidence chronologically
        with_timestamps.sort(key=lambda e: e.timestamp)
        
        # Return timestamped evidence first, then non-timestamped
        return with_timestamps + without_timestamps
    
    def _update_posteriors_with_evidence(self, hypothesis_space: BayesianHypothesisSpace,
                                       likelihood_ratios: Dict[str, float]) -> None:
        """Update posterior probabilities using Bayes' theorem."""
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            if hyp_id in likelihood_ratios:
                prior = hypothesis.posterior_probability  # Current posterior becomes prior
                likelihood_ratio = likelihood_ratios[hyp_id]
                
                # Bayes' theorem: P(H|E) ∝ P(E|H) * P(H)
                # Using likelihood ratio: P(H|E) ∝ LR * P(H)
                unnormalized_posterior = likelihood_ratio * prior
                
                # Store unnormalized posterior (will be normalized later)
                hypothesis.posterior_probability = unnormalized_posterior
    
    def _calculate_convergence_score(self, probability_changes: Dict[str, float]) -> float:
        """Calculate convergence score based on probability changes."""
        if not probability_changes:
            return 0.0
        
        if self.config.convergence_method == ConvergenceMethod.PROBABILITY_CHANGE:
            return max(probability_changes.values())
        else:
            return np.mean(list(probability_changes.values()))


class BatchBeliefUpdater:
    """
    Updates beliefs with all evidence simultaneously.
    
    Processes all evidence at once, computing combined likelihood
    ratios and updating all hypotheses in a single operation.
    """
    
    def __init__(self, config: BeliefUpdateConfig):
        self.config = config
        self.likelihood_calculator = LikelihoodCalculationOrchestrator()
    
    def update_beliefs(self, hypothesis_space: BayesianHypothesisSpace,
                      evidence_list: List[BayesianEvidence],
                      likelihood_config: LikelihoodCalculationConfig) -> UpdateResult:
        """Update beliefs with all evidence simultaneously."""
        initial_probs = {h_id: h.posterior_probability 
                        for h_id, h in hypothesis_space.hypotheses.items()}
        
        # Calculate combined likelihood ratios for all hypotheses
        combined_ratios = self._calculate_combined_likelihood_ratios(
            hypothesis_space, evidence_list, likelihood_config
        )
        
        # Update all posteriors simultaneously
        self._update_all_posteriors(hypothesis_space, combined_ratios)
        
        # Normalize probabilities
        hypothesis_space._normalize_probabilities()
        
        # Calculate results
        final_probs = {h_id: h.posterior_probability 
                      for h_id, h in hypothesis_space.hypotheses.items()}
        
        prob_changes = {h_id: abs(final_probs[h_id] - initial_probs[h_id])
                       for h_id in initial_probs}
        
        convergence_score = max(prob_changes.values()) if prob_changes else 0.0
        convergence_achieved = convergence_score < self.config.convergence_threshold
        
        return UpdateResult(
            success=True,
            iterations=1,
            convergence_achieved=convergence_achieved,
            final_probabilities=final_probs,
            probability_changes=prob_changes,
            likelihood_ratios_used=combined_ratios,
            evidence_processed=[e.evidence_id for e in evidence_list],
            convergence_score=convergence_score
        )
    
    def _calculate_combined_likelihood_ratios(self, hypothesis_space: BayesianHypothesisSpace,
                                            evidence_list: List[BayesianEvidence],
                                            likelihood_config: LikelihoodCalculationConfig) -> Dict[str, float]:
        """Calculate combined likelihood ratios for all hypotheses."""
        combined_ratios = {}
        
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            combined_ratio = 1.0
            
            for evidence in evidence_list:
                # Calculate individual likelihood ratio
                ratio = self.likelihood_calculator.calculate_likelihood_ratio(
                    evidence, hypothesis, likelihood_config
                )
                
                # Apply evidence weighting if configured
                if self.config.confidence_weighting:
                    weight = evidence.reliability * evidence.strength * evidence.source_credibility
                    # Weight affects how much the ratio deviates from neutral (1.0)
                    if ratio > 1:
                        ratio = 1 + (ratio - 1) * weight
                    else:
                        ratio = 1 - (1 - ratio) * weight
                
                # Combine ratios (multiplicative if independent)
                if self.config.evidence_independence:
                    combined_ratio *= ratio
                else:
                    # Non-independent evidence - use averaging
                    combined_ratio = (combined_ratio + ratio) / 2
            
            combined_ratios[hyp_id] = combined_ratio
        
        return combined_ratios
    
    def _update_all_posteriors(self, hypothesis_space: BayesianHypothesisSpace,
                             combined_ratios: Dict[str, float]) -> None:
        """Update all posterior probabilities simultaneously."""
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            if hyp_id in combined_ratios:
                prior = hypothesis.posterior_probability
                likelihood_ratio = combined_ratios[hyp_id]
                
                # Update posterior
                unnormalized_posterior = likelihood_ratio * prior
                hypothesis.posterior_probability = unnormalized_posterior


class IterativeBeliefUpdater:
    """
    Updates beliefs iteratively until convergence.
    
    Repeatedly processes evidence until probability changes
    fall below convergence threshold, useful for complex
    evidence interactions and feedback effects.
    """
    
    def __init__(self, config: BeliefUpdateConfig):
        self.config = config
        self.likelihood_calculator = LikelihoodCalculationOrchestrator()
    
    def update_beliefs(self, hypothesis_space: BayesianHypothesisSpace,
                      evidence_list: List[BayesianEvidence],
                      likelihood_config: LikelihoodCalculationConfig) -> UpdateResult:
        """Update beliefs iteratively until convergence."""
        initial_probs = {h_id: h.posterior_probability 
                        for h_id, h in hypothesis_space.hypotheses.items()}
        
        iteration = 0
        converged = False
        all_likelihood_ratios = {}
        
        while iteration < self.config.max_iterations and not converged:
            iteration += 1
            
            # Store probabilities before update
            prev_probs = {h_id: h.posterior_probability 
                         for h_id, h in hypothesis_space.hypotheses.items()}
            
            # Process all evidence with current probabilities
            for evidence in evidence_list:
                evidence_ratios = {}
                
                for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
                    ratio = self.likelihood_calculator.calculate_likelihood_ratio(
                        evidence, hypothesis, likelihood_config
                    )
                    evidence_ratios[hyp_id] = ratio
                    all_likelihood_ratios[f"iter{iteration}_{evidence.evidence_id}_{hyp_id}"] = ratio
                
                # Update posteriors
                self._update_posteriors_with_evidence(hypothesis_space, evidence_ratios)
                
                # Normalize after each evidence piece
                if self.config.normalize_after_each_update:
                    hypothesis_space._normalize_probabilities()
            
            # Check for convergence
            current_probs = {h_id: h.posterior_probability 
                           for h_id, h in hypothesis_space.hypotheses.items()}
            
            max_change = max(abs(current_probs[h_id] - prev_probs[h_id]) 
                           for h_id in current_probs)
            
            converged = max_change < self.config.convergence_threshold
        
        # Calculate final results
        final_probs = {h_id: h.posterior_probability 
                      for h_id, h in hypothesis_space.hypotheses.items()}
        
        prob_changes = {h_id: abs(final_probs[h_id] - initial_probs[h_id])
                       for h_id in initial_probs}
        
        convergence_score = max(prob_changes.values()) if prob_changes else 0.0
        
        return UpdateResult(
            success=True,
            iterations=iteration,
            convergence_achieved=converged,
            final_probabilities=final_probs,
            probability_changes=prob_changes,
            likelihood_ratios_used=all_likelihood_ratios,
            evidence_processed=[e.evidence_id for e in evidence_list],
            convergence_score=convergence_score
        )
    
    def _update_posteriors_with_evidence(self, hypothesis_space: BayesianHypothesisSpace,
                                       likelihood_ratios: Dict[str, float]) -> None:
        """Update posterior probabilities with evidence."""
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            if hyp_id in likelihood_ratios:
                prior = hypothesis.posterior_probability
                likelihood_ratio = likelihood_ratios[hyp_id]
                
                # Apply iterative dampening to prevent oscillation
                dampening_factor = 0.8  # Reduce update magnitude
                adjusted_ratio = 1 + (likelihood_ratio - 1) * dampening_factor
                
                unnormalized_posterior = adjusted_ratio * prior
                hypothesis.posterior_probability = unnormalized_posterior


class HierarchicalBeliefUpdater:
    """
    Updates beliefs hierarchically from parent to child hypotheses.
    
    Processes evidence at each level of the hypothesis hierarchy,
    propagating probability updates from parents to children.
    """
    
    def __init__(self, config: BeliefUpdateConfig):
        self.config = config
        self.likelihood_calculator = LikelihoodCalculationOrchestrator()
    
    def update_beliefs(self, hypothesis_space: BayesianHypothesisSpace,
                      evidence_list: List[BayesianEvidence],
                      likelihood_config: LikelihoodCalculationConfig) -> UpdateResult:
        """Update beliefs hierarchically."""
        initial_probs = {h_id: h.posterior_probability 
                        for h_id, h in hypothesis_space.hypotheses.items()}
        
        # Get hierarchy levels
        levels = self._get_hierarchy_levels(hypothesis_space)
        max_level = max(levels.values()) if levels else 0
        
        all_likelihood_ratios = {}
        processed_evidence = []
        
        # Process each level from top (0) to bottom (max_level)
        for level in range(max_level + 1):
            level_hypotheses = [h_id for h_id, lev in levels.items() if lev == level]
            
            if not level_hypotheses:
                continue
            
            # Process evidence for hypotheses at this level
            for evidence in evidence_list:
                for hyp_id in level_hypotheses:
                    hypothesis = hypothesis_space.hypotheses[hyp_id]
                    
                    # Calculate likelihood ratio
                    ratio = self.likelihood_calculator.calculate_likelihood_ratio(
                        evidence, hypothesis, likelihood_config
                    )
                    
                    all_likelihood_ratios[f"{evidence.evidence_id}_{hyp_id}"] = ratio
                    
                    # Update posterior
                    prior = hypothesis.posterior_probability
                    unnormalized_posterior = ratio * prior
                    hypothesis.posterior_probability = unnormalized_posterior
                
                if evidence.evidence_id not in processed_evidence:
                    processed_evidence.append(evidence.evidence_id)
            
            # Normalize at each level
            self._normalize_level(hypothesis_space, level_hypotheses)
            
            # Propagate probabilities to children
            self._propagate_to_children(hypothesis_space, level_hypotheses, level)
        
        # Final normalization
        hypothesis_space._normalize_probabilities()
        
        # Calculate results
        final_probs = {h_id: h.posterior_probability 
                      for h_id, h in hypothesis_space.hypotheses.items()}
        
        prob_changes = {h_id: abs(final_probs[h_id] - initial_probs[h_id])
                       for h_id in initial_probs}
        
        convergence_score = max(prob_changes.values()) if prob_changes else 0.0
        convergence_achieved = convergence_score < self.config.convergence_threshold
        
        return UpdateResult(
            success=True,
            iterations=max_level + 1,
            convergence_achieved=convergence_achieved,
            final_probabilities=final_probs,
            probability_changes=prob_changes,
            likelihood_ratios_used=all_likelihood_ratios,
            evidence_processed=processed_evidence,
            convergence_score=convergence_score
        )
    
    def _get_hierarchy_levels(self, hypothesis_space: BayesianHypothesisSpace) -> Dict[str, int]:
        """Get hierarchy level for each hypothesis."""
        levels = {}
        for hyp_id in hypothesis_space.hypotheses:
            levels[hyp_id] = hypothesis_space.get_hierarchy_level(hyp_id)
        return levels
    
    def _normalize_level(self, hypothesis_space: BayesianHypothesisSpace, 
                        level_hypotheses: List[str]) -> None:
        """Normalize probabilities within a hierarchy level."""
        if not level_hypotheses:
            return
        
        # Check if any of these hypotheses are in mutual exclusivity groups
        for group in hypothesis_space.mutual_exclusivity_groups:
            group_at_level = [h for h in level_hypotheses if h in group]
            if len(group_at_level) > 1:
                # Normalize within this group
                total_prob = sum(hypothesis_space.hypotheses[h].posterior_probability 
                               for h in group_at_level)
                if total_prob > 0:
                    for h_id in group_at_level:
                        hypothesis_space.hypotheses[h_id].posterior_probability /= total_prob
    
    def _propagate_to_children(self, hypothesis_space: BayesianHypothesisSpace,
                             parent_hypotheses: List[str], parent_level: int) -> None:
        """Propagate probability updates to child hypotheses."""
        for parent_id in parent_hypotheses:
            parent_hypothesis = hypothesis_space.hypotheses[parent_id]
            parent_prob = parent_hypothesis.posterior_probability
            
            # Find children
            children = [h_id for h_id, h in hypothesis_space.hypotheses.items()
                       if h.parent_hypothesis == parent_id]
            
            if children:
                # Distribute parent probability among children
                prob_per_child = parent_prob / len(children)
                for child_id in children:
                    child = hypothesis_space.hypotheses[child_id]
                    # Weighted combination of child's own probability and inherited probability
                    inheritance_weight = 0.3  # 30% from parent, 70% from own evidence
                    child.posterior_probability = (
                        inheritance_weight * prob_per_child +
                        (1 - inheritance_weight) * child.posterior_probability
                    )


class BeliefUpdateOrchestrator:
    """
    Orchestrates belief updating using different methods and strategies.
    
    Provides unified interface for belief updating with method selection,
    convergence monitoring, and uncertainty quantification.
    """
    
    def __init__(self):
        self.updaters = {
            UpdateMethod.SEQUENTIAL: SequentialBeliefUpdater,
            UpdateMethod.BATCH: BatchBeliefUpdater,
            UpdateMethod.ITERATIVE: IterativeBeliefUpdater,
            UpdateMethod.HIERARCHICAL: HierarchicalBeliefUpdater
        }
        
        self.update_history: List[Dict[str, Any]] = []
    
    def update_beliefs(self, hypothesis_space: BayesianHypothesisSpace,
                      evidence_list: List[BayesianEvidence],
                      update_config: BeliefUpdateConfig,
                      likelihood_config: LikelihoodCalculationConfig) -> UpdateResult:
        """Update beliefs using specified method."""
        if update_config.update_method not in self.updaters:
            raise ValueError(f"Unsupported update method: {update_config.update_method}")
        
        updater_class = self.updaters[update_config.update_method]
        updater = updater_class(update_config)
        
        # Perform update
        result = updater.update_beliefs(hypothesis_space, evidence_list, likelihood_config)
        
        # Record update
        update_record = {
            "timestamp": datetime.now().isoformat(),
            "method": update_config.update_method.value,
            "hypothesis_space_id": hypothesis_space.hypothesis_space_id,
            "evidence_count": len(evidence_list),
            "iterations": int(result.iterations),
            "convergence_achieved": bool(result.convergence_achieved),
            "convergence_score": float(result.convergence_score),
            "max_probability_change": float(max(result.probability_changes.values())) if result.probability_changes else 0.0,
            "config": {
                "convergence_threshold": float(update_config.convergence_threshold),
                "max_iterations": int(update_config.max_iterations),
                "evidence_independence": bool(update_config.evidence_independence)
            }
        }
        
        self.update_history.append(update_record)
        
        return result
    
    def compare_update_methods(self, hypothesis_space: BayesianHypothesisSpace,
                             evidence_list: List[BayesianEvidence],
                             likelihood_config: LikelihoodCalculationConfig,
                             methods: List[UpdateMethod] = None) -> Dict[str, UpdateResult]:
        """Compare different update methods on the same data."""
        if methods is None:
            methods = list(UpdateMethod)
        
        # Save original state
        original_probs = {}
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            original_probs[hyp_id] = {
                'prior': hypothesis.prior_probability,
                'posterior': hypothesis.posterior_probability
            }
        
        results = {}
        
        for method in methods:
            # Restore original state
            for hyp_id, probs in original_probs.items():
                hypothesis_space.hypotheses[hyp_id].prior_probability = probs['prior']
                hypothesis_space.hypotheses[hyp_id].posterior_probability = probs['posterior']
            
            # Run update with this method
            config = BeliefUpdateConfig(update_method=method)
            result = self.update_beliefs(hypothesis_space, evidence_list, config, likelihood_config)
            results[method.value] = result
        
        return results
    
    def sensitivity_analysis(self, hypothesis_space: BayesianHypothesisSpace,
                           evidence_list: List[BayesianEvidence],
                           base_config: BeliefUpdateConfig,
                           likelihood_config: LikelihoodCalculationConfig,
                           parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis on update parameters."""
        base_result = self.update_beliefs(hypothesis_space, evidence_list, base_config, likelihood_config)
        
        sensitivity_results = {
            "base_result": {
                "final_probabilities": base_result.final_probabilities,
                "convergence_score": base_result.convergence_score,
                "iterations": base_result.iterations
            },
            "parameter_sensitivity": {},
            "most_sensitive_parameter": None,
            "max_probability_deviation": 0.0
        }
        
        # Save original state
        original_probs = {}
        for hyp_id, hypothesis in hypothesis_space.hypotheses.items():
            original_probs[hyp_id] = {
                'prior': hypothesis.prior_probability,
                'posterior': hypothesis.posterior_probability
            }
        
        for param_name, values in parameter_ranges.items():
            param_results = []
            max_deviation_for_param = 0.0
            
            for value in values:
                # Restore original state
                for hyp_id, probs in original_probs.items():
                    hypothesis_space.hypotheses[hyp_id].prior_probability = probs['prior']
                    hypothesis_space.hypotheses[hyp_id].posterior_probability = probs['posterior']
                
                # Create modified config
                modified_config = BeliefUpdateConfig(
                    update_method=base_config.update_method,
                    convergence_method=base_config.convergence_method,
                    convergence_threshold=base_config.convergence_threshold,
                    max_iterations=base_config.max_iterations
                )
                setattr(modified_config, param_name, value)
                
                # Run update
                result = self.update_beliefs(hypothesis_space, evidence_list, modified_config, likelihood_config)
                
                # Calculate deviations from base result
                deviations = {}
                for hyp_id in base_result.final_probabilities:
                    if hyp_id in result.final_probabilities:
                        deviation = abs(result.final_probabilities[hyp_id] - 
                                      base_result.final_probabilities[hyp_id])
                        deviations[hyp_id] = deviation
                        max_deviation_for_param = max(max_deviation_for_param, deviation)
                
                param_results.append({
                    "parameter_value": value,
                    "final_probabilities": result.final_probabilities,
                    "deviations": deviations,
                    "max_deviation": max_deviation_for_param,
                    "convergence_score": result.convergence_score,
                    "iterations": result.iterations
                })
            
            sensitivity_results["parameter_sensitivity"][param_name] = param_results
            
            # Track most sensitive parameter
            if max_deviation_for_param > sensitivity_results["max_probability_deviation"]:
                sensitivity_results["max_probability_deviation"] = max_deviation_for_param
                sensitivity_results["most_sensitive_parameter"] = param_name
        
        return sensitivity_results
    
    def diagnose_convergence_issues(self, hypothesis_space: BayesianHypothesisSpace,
                                   evidence_list: List[BayesianEvidence],
                                   config: BeliefUpdateConfig,
                                   likelihood_config: LikelihoodCalculationConfig) -> Dict[str, Any]:
        """Diagnose convergence issues and suggest improvements."""
        # Try different configurations to identify issues
        diagnostic_results = {
            "original_result": None,
            "potential_issues": [],
            "recommendations": [],
            "alternative_configs": {}
        }
        
        # Test original configuration
        original_result = self.update_beliefs(hypothesis_space, evidence_list, config, likelihood_config)
        diagnostic_results["original_result"] = {
            "convergence_achieved": original_result.convergence_achieved,
            "iterations": original_result.iterations,
            "convergence_score": original_result.convergence_score
        }
        
        # Test with relaxed convergence threshold
        if not original_result.convergence_achieved:
            relaxed_config = BeliefUpdateConfig(
                update_method=config.update_method,
                convergence_threshold=config.convergence_threshold * 10,
                max_iterations=config.max_iterations
            )
            
            # Save and restore state
            original_probs = {hyp_id: h.posterior_probability for hyp_id, h in hypothesis_space.hypotheses.items()}
            
            relaxed_result = self.update_beliefs(hypothesis_space, evidence_list, relaxed_config, likelihood_config)
            
            # Restore state
            for hyp_id, prob in original_probs.items():
                hypothesis_space.hypotheses[hyp_id].posterior_probability = prob
            
            diagnostic_results["alternative_configs"]["relaxed_threshold"] = {
                "convergence_achieved": relaxed_result.convergence_achieved,
                "iterations": relaxed_result.iterations,
                "convergence_score": relaxed_result.convergence_score
            }
            
            if relaxed_result.convergence_achieved:
                diagnostic_results["potential_issues"].append("Convergence threshold too strict")
                diagnostic_results["recommendations"].append(f"Consider relaxing convergence threshold to {relaxed_config.convergence_threshold}")
        
        # Test with more iterations
        if original_result.iterations >= config.max_iterations * 0.9:
            diagnostic_results["potential_issues"].append("Reached maximum iterations")
            diagnostic_results["recommendations"].append("Consider increasing max_iterations")
        
        # Test with different update method
        if config.update_method == UpdateMethod.SEQUENTIAL:
            alternative_method = UpdateMethod.BATCH
        else:
            alternative_method = UpdateMethod.SEQUENTIAL
        
        alt_config = BeliefUpdateConfig(
            update_method=alternative_method,
            convergence_threshold=config.convergence_threshold,
            max_iterations=config.max_iterations
        )
        
        # Save and restore state
        original_probs = {hyp_id: h.posterior_probability for hyp_id, h in hypothesis_space.hypotheses.items()}
        
        alt_result = self.update_beliefs(hypothesis_space, evidence_list, alt_config, likelihood_config)
        
        # Restore state
        for hyp_id, prob in original_probs.items():
            hypothesis_space.hypotheses[hyp_id].posterior_probability = prob
        
        diagnostic_results["alternative_configs"]["alternative_method"] = {
            "method": alternative_method.value,
            "convergence_achieved": alt_result.convergence_achieved,
            "iterations": alt_result.iterations,
            "convergence_score": alt_result.convergence_score
        }
        
        if alt_result.convergence_achieved and not original_result.convergence_achieved:
            diagnostic_results["recommendations"].append(f"Consider using {alternative_method.value} update method")
        
        return diagnostic_results
    
    def export_update_history(self, file_path: Union[str, Path]) -> None:
        """Export update history to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.update_history, f, indent=2, ensure_ascii=False)
    
    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of belief updates performed."""
        if not self.update_history:
            return {"total_updates": 0}
        
        methods_used = [record["method"] for record in self.update_history]
        method_counts = {method: methods_used.count(method) for method in set(methods_used)}
        
        convergence_rates = [record["convergence_achieved"] for record in self.update_history]
        
        return {
            "total_updates": len(self.update_history),
            "methods_used": method_counts,
            "convergence_rate": sum(convergence_rates) / len(convergence_rates) if convergence_rates else 0.0,
            "average_iterations": np.mean([record["iterations"] for record in self.update_history]),
            "latest_update": self.update_history[-1]["timestamp"]
        }