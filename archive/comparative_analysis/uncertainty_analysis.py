"""
Uncertainty Propagation and Sensitivity Analysis for Bayesian Process Tracing.

Implements comprehensive uncertainty analysis including uncertainty propagation,
sensitivity analysis, robustness testing, and Monte Carlo simulation for
understanding the reliability and stability of Bayesian inference results.
"""

import math
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
from concurrent.futures import ThreadPoolExecutor
import warnings

from .bayesian_models import (
    BayesianHypothesis, BayesianEvidence, BayesianHypothesisSpace,
    BayesianProcessTracingModel, EvidenceType
)
from .confidence_calculator import ConfidenceAssessment, CausalConfidenceCalculator
from .belief_updater import SequentialBeliefUpdater, BeliefUpdateConfig


class UncertaintyType(Enum):
    """Types of uncertainty in Bayesian process tracing."""
    ALEATORY = "aleatory"                 # Inherent randomness/variability
    EPISTEMIC = "epistemic"               # Knowledge uncertainty
    MODEL = "model"                       # Model structure uncertainty
    PARAMETER = "parameter"               # Parameter value uncertainty
    DATA = "data"                         # Data quality uncertainty
    MEASUREMENT = "measurement"           # Measurement error uncertainty


class SensitivityType(Enum):
    """Types of sensitivity analysis."""
    LOCAL = "local"                       # Local gradient-based sensitivity
    GLOBAL = "global"                     # Global variance-based sensitivity
    MORRIS = "morris"                     # Morris one-at-a-time method
    SOBOL = "sobol"                       # Sobol variance decomposition
    MONTE_CARLO = "monte_carlo"           # Monte Carlo sensitivity


@dataclass
class UncertaintySource:
    """Represents a source of uncertainty in the analysis."""
    source_id: str
    uncertainty_type: UncertaintyType
    description: str
    affected_parameters: List[str]
    uncertainty_magnitude: float          # 0-1 scale
    distribution_type: str = "normal"     # normal, uniform, beta, etc.
    distribution_params: Dict[str, float] = field(default_factory=dict)
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the uncertainty distribution."""
        if self.distribution_type == "normal":
            mean = self.distribution_params.get("mean", 0.0)
            std = self.distribution_params.get("std", self.uncertainty_magnitude)
            return np.random.normal(mean, std, size)
        
        elif self.distribution_type == "uniform":
            low = self.distribution_params.get("low", -self.uncertainty_magnitude)
            high = self.distribution_params.get("high", self.uncertainty_magnitude)
            return np.random.uniform(low, high, size)
        
        elif self.distribution_type == "beta":
            alpha = self.distribution_params.get("alpha", 2.0)
            beta = self.distribution_params.get("beta", 2.0)
            return np.random.beta(alpha, beta, size)
        
        else:
            # Default to normal
            return np.random.normal(0.0, self.uncertainty_magnitude, size)


@dataclass
class SensitivityResult:
    """Results of sensitivity analysis."""
    parameter_name: str
    sensitivity_index: float
    confidence_interval: Tuple[float, float]
    sensitivity_type: SensitivityType
    baseline_output: float
    perturbed_outputs: List[float]
    perturbation_values: List[float]
    
    def get_interpretation(self) -> str:
        """Get interpretation of sensitivity result."""
        if self.sensitivity_index >= 0.1:
            return "High sensitivity - parameter has substantial impact on results"
        elif self.sensitivity_index >= 0.05:
            return "Moderate sensitivity - parameter has noticeable impact"
        elif self.sensitivity_index >= 0.01:
            return "Low sensitivity - parameter has minor impact"
        else:
            return "Negligible sensitivity - parameter has minimal impact"


@dataclass
class UncertaintyAnalysisResult:
    """Comprehensive uncertainty analysis results."""
    hypothesis_id: str
    baseline_confidence: float
    confidence_distribution: np.ndarray
    confidence_percentiles: Dict[str, float]
    sensitivity_results: Dict[str, SensitivityResult]
    uncertainty_sources: List[UncertaintySource]
    
    # Statistical measures
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    confidence_skewness: float = 0.0
    confidence_kurtosis: float = 0.0
    
    # Robustness measures
    robustness_score: float = 0.0
    stability_score: float = 0.0
    
    # Monte Carlo details
    n_simulations: int = 1000
    convergence_achieved: bool = False
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty analysis."""
        return {
            "baseline_confidence": self.baseline_confidence,
            "mean_confidence": self.confidence_mean,
            "confidence_std": self.confidence_std,
            "confidence_95_ci": (
                self.confidence_percentiles.get("2.5", 0.0),
                self.confidence_percentiles.get("97.5", 1.0)
            ),
            "robustness_score": self.robustness_score,
            "most_sensitive_parameters": self._get_most_sensitive_parameters(),
            "uncertainty_interpretation": self._get_interpretation()
        }
    
    def _get_most_sensitive_parameters(self) -> List[Tuple[str, float]]:
        """Get parameters with highest sensitivity."""
        sensitivities = [
            (param, result.sensitivity_index)
            for param, result in self.sensitivity_results.items()
        ]
        return sorted(sensitivities, key=lambda x: x[1], reverse=True)[:5]
    
    def _get_interpretation(self) -> str:
        """Get interpretation of uncertainty analysis."""
        if self.confidence_std < 0.05:
            return "Low uncertainty - results are stable and reliable"
        elif self.confidence_std < 0.1:
            return "Moderate uncertainty - some variability in results"
        elif self.confidence_std < 0.2:
            return "High uncertainty - substantial variability in results"
        else:
            return "Very high uncertainty - results highly variable"


class UncertaintyAnalyzer:
    """
    Comprehensive uncertainty analysis for Bayesian process tracing.
    
    Implements multiple uncertainty quantification methods including:
    - Monte Carlo uncertainty propagation
    - Sensitivity analysis (local and global)
    - Robustness testing
    - Stability analysis
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.confidence_calculator = CausalConfidenceCalculator()
        self.analysis_history: List[UncertaintyAnalysisResult] = []
    
    def analyze_uncertainty(self,
                          hypothesis: BayesianHypothesis,
                          hypothesis_space: BayesianHypothesisSpace,
                          evidence_list: List[BayesianEvidence],
                          uncertainty_sources: Optional[List[UncertaintySource]] = None,
                          n_simulations: int = 1000,
                          confidence_level: float = 0.95) -> UncertaintyAnalysisResult:
        """
        Perform comprehensive uncertainty analysis.
        
        Args:
            hypothesis: Target hypothesis
            hypothesis_space: Containing hypothesis space
            evidence_list: Relevant evidence
            uncertainty_sources: Sources of uncertainty (auto-detected if None)
            n_simulations: Number of Monte Carlo simulations
            confidence_level: Confidence level for intervals
            
        Returns:
            UncertaintyAnalysisResult with comprehensive analysis
        """
        # Calculate baseline confidence
        baseline_assessment = self.confidence_calculator.calculate_confidence(
            hypothesis, hypothesis_space, evidence_list
        )
        baseline_confidence = baseline_assessment.overall_confidence
        
        # Auto-detect uncertainty sources if not provided
        if uncertainty_sources is None:
            uncertainty_sources = self._detect_uncertainty_sources(
                hypothesis, evidence_list
            )
        
        # Perform Monte Carlo uncertainty propagation
        confidence_distribution = self._monte_carlo_uncertainty_propagation(
            hypothesis, hypothesis_space, evidence_list, uncertainty_sources, n_simulations
        )
        
        # Calculate confidence percentiles
        percentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
        confidence_percentiles = {
            str(p): np.percentile(confidence_distribution, p)
            for p in percentiles
        }
        
        # Perform sensitivity analysis
        sensitivity_results = self._sensitivity_analysis(
            hypothesis, hypothesis_space, evidence_list, uncertainty_sources
        )
        
        # Calculate statistical measures
        confidence_mean = np.mean(confidence_distribution)
        confidence_std = np.std(confidence_distribution)
        confidence_skewness = stats.skew(confidence_distribution)
        confidence_kurtosis = stats.kurtosis(confidence_distribution)
        
        # Calculate robustness and stability scores
        robustness_score = self._calculate_robustness_score(
            baseline_confidence, confidence_distribution
        )
        stability_score = self._calculate_stability_score(
            confidence_distribution, sensitivity_results
        )
        
        # Check convergence
        convergence_achieved = self._check_monte_carlo_convergence(
            confidence_distribution, n_simulations
        )
        
        # Create analysis result
        result = UncertaintyAnalysisResult(
            hypothesis_id=hypothesis.hypothesis_id,
            baseline_confidence=baseline_confidence,
            confidence_distribution=confidence_distribution,
            confidence_percentiles=confidence_percentiles,
            sensitivity_results=sensitivity_results,
            uncertainty_sources=uncertainty_sources,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            confidence_skewness=confidence_skewness,
            confidence_kurtosis=confidence_kurtosis,
            robustness_score=robustness_score,
            stability_score=stability_score,
            n_simulations=n_simulations,
            convergence_achieved=convergence_achieved
        )
        
        self.analysis_history.append(result)
        return result
    
    def _detect_uncertainty_sources(self,
                                   hypothesis: BayesianHypothesis,
                                   evidence_list: List[BayesianEvidence]) -> List[UncertaintySource]:
        """Auto-detect sources of uncertainty."""
        uncertainty_sources = []
        
        # Evidence reliability uncertainty
        for evidence in evidence_list:
            if evidence.reliability < 1.0:
                uncertainty_sources.append(UncertaintySource(
                    source_id=f"reliability_{evidence.evidence_id}",
                    uncertainty_type=UncertaintyType.DATA,
                    description=f"Reliability uncertainty for evidence {evidence.evidence_id}",
                    affected_parameters=[f"evidence_{evidence.evidence_id}_reliability"],
                    uncertainty_magnitude=1.0 - evidence.reliability,
                    distribution_type="beta",
                    distribution_params={
                        "alpha": evidence.reliability * 10,
                        "beta": (1 - evidence.reliability) * 10
                    }
                ))
        
        # Evidence strength uncertainty
        for evidence in evidence_list:
            if evidence.strength < 1.0:
                uncertainty_sources.append(UncertaintySource(
                    source_id=f"strength_{evidence.evidence_id}",
                    uncertainty_type=UncertaintyType.EPISTEMIC,
                    description=f"Strength uncertainty for evidence {evidence.evidence_id}",
                    affected_parameters=[f"evidence_{evidence.evidence_id}_strength"],
                    uncertainty_magnitude=1.0 - evidence.strength,
                    distribution_type="beta",
                    distribution_params={
                        "alpha": evidence.strength * 10,
                        "beta": (1 - evidence.strength) * 10
                    }
                ))
        
        # Likelihood assessment uncertainty
        for evidence in evidence_list:
            # Add uncertainty around likelihood values
            uncertainty_sources.append(UncertaintySource(
                source_id=f"likelihood_{evidence.evidence_id}",
                uncertainty_type=UncertaintyType.PARAMETER,
                description=f"Likelihood assessment uncertainty for evidence {evidence.evidence_id}",
                affected_parameters=[
                    f"evidence_{evidence.evidence_id}_likelihood_positive",
                    f"evidence_{evidence.evidence_id}_likelihood_negative"
                ],
                uncertainty_magnitude=0.1,  # 10% uncertainty around likelihood values
                distribution_type="normal",
                distribution_params={"mean": 0.0, "std": 0.05}
            ))
        
        # Prior probability uncertainty
        uncertainty_sources.append(UncertaintySource(
            source_id=f"prior_{hypothesis.hypothesis_id}",
            uncertainty_type=UncertaintyType.EPISTEMIC,
            description=f"Prior probability uncertainty for hypothesis {hypothesis.hypothesis_id}",
            affected_parameters=[f"hypothesis_{hypothesis.hypothesis_id}_prior"],
            uncertainty_magnitude=0.1,  # Assume 10% uncertainty in priors
            distribution_type="beta",
            distribution_params={
                "alpha": hypothesis.prior_probability * 10,
                "beta": (1 - hypothesis.prior_probability) * 10
            }
        ))
        
        return uncertainty_sources
    
    def _monte_carlo_uncertainty_propagation(self,
                                           hypothesis: BayesianHypothesis,
                                           hypothesis_space: BayesianHypothesisSpace,
                                           evidence_list: List[BayesianEvidence],
                                           uncertainty_sources: List[UncertaintySource],
                                           n_simulations: int) -> np.ndarray:
        """Perform Monte Carlo uncertainty propagation."""
        confidence_samples = []
        
        for i in range(n_simulations):
            # Create perturbed copies of hypothesis and evidence
            perturbed_hypothesis = self._perturb_hypothesis(
                hypothesis, uncertainty_sources
            )
            perturbed_evidence = self._perturb_evidence_list(
                evidence_list, uncertainty_sources
            )
            
            # Create temporary hypothesis space for this simulation
            temp_space = BayesianHypothesisSpace(
                f"temp_space_{i}", "Temporary space for uncertainty analysis"
            )
            temp_space.add_hypothesis(perturbed_hypothesis)
            
            for evidence in perturbed_evidence:
                temp_space.add_evidence(evidence)
            
            # Calculate confidence for this simulation
            try:
                assessment = self.confidence_calculator.calculate_confidence(
                    perturbed_hypothesis, temp_space, perturbed_evidence
                )
                confidence_samples.append(assessment.overall_confidence)
            except Exception:
                # If simulation fails, use baseline confidence
                confidence_samples.append(hypothesis.posterior_probability)
        
        return np.array(confidence_samples)
    
    def _perturb_hypothesis(self,
                          hypothesis: BayesianHypothesis,
                          uncertainty_sources: List[UncertaintySource]) -> BayesianHypothesis:
        """Create a perturbed copy of hypothesis based on uncertainty sources."""
        # Create a copy of the hypothesis
        perturbed = BayesianHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            description=hypothesis.description,
            hypothesis_type=hypothesis.hypothesis_type,
            prior_probability=hypothesis.prior_probability,
            posterior_probability=hypothesis.posterior_probability,
            likelihood_cache=hypothesis.likelihood_cache.copy(),
            parent_hypothesis=hypothesis.parent_hypothesis,
            child_hypotheses=hypothesis.child_hypotheses.copy(),
            required_evidence=hypothesis.required_evidence.copy(),
            supporting_evidence=hypothesis.supporting_evidence.copy(),
            contradicting_evidence=hypothesis.contradicting_evidence.copy(),
            temporal_constraints=hypothesis.temporal_constraints.copy(),
            confidence_level=hypothesis.confidence_level,
            last_updated=hypothesis.last_updated,
            update_history=hypothesis.update_history.copy()
        )
        
        # Apply perturbations from uncertainty sources
        for source in uncertainty_sources:
            if f"hypothesis_{hypothesis.hypothesis_id}_prior" in source.affected_parameters:
                perturbation = source.sample(1)[0]
                new_prior = max(0.01, min(0.99, hypothesis.prior_probability + perturbation))
                perturbed.prior_probability = new_prior
                perturbed.posterior_probability = new_prior  # Reset posterior to new prior
        
        return perturbed
    
    def _perturb_evidence_list(self,
                             evidence_list: List[BayesianEvidence],
                             uncertainty_sources: List[UncertaintySource]) -> List[BayesianEvidence]:
        """Create perturbed copies of evidence based on uncertainty sources."""
        perturbed_evidence = []
        
        for evidence in evidence_list:
            # Create a copy of the evidence
            perturbed = BayesianEvidence(
                evidence_id=evidence.evidence_id,
                description=evidence.description,
                evidence_type=evidence.evidence_type,
                source_node_id=evidence.source_node_id,
                necessity=evidence.necessity,
                sufficiency=evidence.sufficiency,
                likelihood_positive=evidence.likelihood_positive,
                likelihood_negative=evidence.likelihood_negative,
                reliability=evidence.reliability,
                strength=evidence.strength,
                independence=evidence.independence,
                timestamp=evidence.timestamp,
                temporal_order=evidence.temporal_order,
                source_credibility=evidence.source_credibility,
                collection_method=evidence.collection_method,
                last_updated=evidence.last_updated
            )
            
            # Apply perturbations from uncertainty sources
            for source in uncertainty_sources:
                # Reliability perturbations
                if f"evidence_{evidence.evidence_id}_reliability" in source.affected_parameters:
                    perturbation = source.sample(1)[0]
                    new_reliability = max(0.01, min(1.0, evidence.reliability + perturbation))
                    perturbed.reliability = new_reliability
                
                # Strength perturbations
                if f"evidence_{evidence.evidence_id}_strength" in source.affected_parameters:
                    perturbation = source.sample(1)[0]
                    new_strength = max(0.01, min(1.0, evidence.strength + perturbation))
                    perturbed.strength = new_strength
                
                # Likelihood perturbations
                if f"evidence_{evidence.evidence_id}_likelihood_positive" in source.affected_parameters:
                    perturbation = source.sample(1)[0]
                    new_likelihood = max(0.01, min(0.99, evidence.likelihood_positive + perturbation))
                    perturbed.likelihood_positive = new_likelihood
                
                if f"evidence_{evidence.evidence_id}_likelihood_negative" in source.affected_parameters:
                    perturbation = source.sample(1)[0]
                    new_likelihood = max(0.01, min(0.99, evidence.likelihood_negative + perturbation))
                    perturbed.likelihood_negative = new_likelihood
            
            perturbed_evidence.append(perturbed)
        
        return perturbed_evidence
    
    def _sensitivity_analysis(self,
                            hypothesis: BayesianHypothesis,
                            hypothesis_space: BayesianHypothesisSpace,
                            evidence_list: List[BayesianEvidence],
                            uncertainty_sources: List[UncertaintySource]) -> Dict[str, SensitivityResult]:
        """Perform sensitivity analysis for all uncertain parameters."""
        sensitivity_results = {}
        
        # Calculate baseline confidence
        baseline_assessment = self.confidence_calculator.calculate_confidence(
            hypothesis, hypothesis_space, evidence_list
        )
        baseline_confidence = baseline_assessment.overall_confidence
        
        # Analyze sensitivity for each uncertainty source
        for source in uncertainty_sources:
            sensitivity_result = self._calculate_parameter_sensitivity(
                hypothesis, hypothesis_space, evidence_list, source, baseline_confidence
            )
            sensitivity_results[source.source_id] = sensitivity_result
        
        return sensitivity_results
    
    def _calculate_parameter_sensitivity(self,
                                       hypothesis: BayesianHypothesis,
                                       hypothesis_space: BayesianHypothesisSpace,
                                       evidence_list: List[BayesianEvidence],
                                       uncertainty_source: UncertaintySource,
                                       baseline_confidence: float) -> SensitivityResult:
        """Calculate sensitivity for a single parameter."""
        # Define perturbation range
        perturbation_range = np.linspace(-2 * uncertainty_source.uncertainty_magnitude,
                                       2 * uncertainty_source.uncertainty_magnitude,
                                       11)  # 11 points for good resolution
        
        perturbed_outputs = []
        
        for perturbation in perturbation_range:
            # Apply perturbation
            if "hypothesis" in uncertainty_source.source_id:
                perturbed_hypothesis = self._apply_hypothesis_perturbation(
                    hypothesis, uncertainty_source, perturbation
                )
                perturbed_evidence = evidence_list
            else:
                perturbed_hypothesis = hypothesis
                perturbed_evidence = self._apply_evidence_perturbation(
                    evidence_list, uncertainty_source, perturbation
                )
            
            # Create temporary hypothesis space
            temp_space = BayesianHypothesisSpace(
                f"temp_sensitivity", "Temporary space for sensitivity analysis"
            )
            temp_space.add_hypothesis(perturbed_hypothesis)
            
            for evidence in perturbed_evidence:
                temp_space.add_evidence(evidence)
            
            # Calculate confidence
            try:
                assessment = self.confidence_calculator.calculate_confidence(
                    perturbed_hypothesis, temp_space, perturbed_evidence
                )
                confidence = assessment.overall_confidence
            except Exception:
                confidence = baseline_confidence
            
            perturbed_outputs.append(confidence)
        
        # Calculate sensitivity index (normalized gradient)
        if len(perturbed_outputs) > 1:
            sensitivity_index = np.std(perturbed_outputs) / baseline_confidence
        else:
            sensitivity_index = 0.0
        
        # Calculate confidence interval for sensitivity (bootstrap)
        sensitivity_ci = self._bootstrap_sensitivity_ci(
            perturbation_range, perturbed_outputs, baseline_confidence
        )
        
        return SensitivityResult(
            parameter_name=uncertainty_source.source_id,
            sensitivity_index=sensitivity_index,
            confidence_interval=sensitivity_ci,
            sensitivity_type=SensitivityType.LOCAL,
            baseline_output=baseline_confidence,
            perturbed_outputs=perturbed_outputs,
            perturbation_values=perturbation_range.tolist()
        )
    
    def _apply_hypothesis_perturbation(self,
                                     hypothesis: BayesianHypothesis,
                                     uncertainty_source: UncertaintySource,
                                     perturbation: float) -> BayesianHypothesis:
        """Apply perturbation to hypothesis parameters."""
        perturbed = BayesianHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            description=hypothesis.description,
            hypothesis_type=hypothesis.hypothesis_type,
            prior_probability=hypothesis.prior_probability,
            posterior_probability=hypothesis.posterior_probability
        )
        
        if "prior" in uncertainty_source.source_id:
            new_prior = max(0.01, min(0.99, hypothesis.prior_probability + perturbation))
            perturbed.prior_probability = new_prior
            perturbed.posterior_probability = new_prior
        
        return perturbed
    
    def _apply_evidence_perturbation(self,
                                   evidence_list: List[BayesianEvidence],
                                   uncertainty_source: UncertaintySource,
                                   perturbation: float) -> List[BayesianEvidence]:
        """Apply perturbation to evidence parameters."""
        perturbed_evidence = []
        
        for evidence in evidence_list:
            perturbed = BayesianEvidence(
                evidence_id=evidence.evidence_id,
                description=evidence.description,
                evidence_type=evidence.evidence_type,
                source_node_id=evidence.source_node_id,
                likelihood_positive=evidence.likelihood_positive,
                likelihood_negative=evidence.likelihood_negative,
                reliability=evidence.reliability,
                strength=evidence.strength
            )
            
            # Apply perturbation if this evidence is affected
            if f"evidence_{evidence.evidence_id}" in uncertainty_source.source_id:
                if "reliability" in uncertainty_source.source_id:
                    new_reliability = max(0.01, min(1.0, evidence.reliability + perturbation))
                    perturbed.reliability = new_reliability
                elif "strength" in uncertainty_source.source_id:
                    new_strength = max(0.01, min(1.0, evidence.strength + perturbation))
                    perturbed.strength = new_strength
                elif "likelihood_positive" in uncertainty_source.source_id:
                    new_likelihood = max(0.01, min(0.99, evidence.likelihood_positive + perturbation))
                    perturbed.likelihood_positive = new_likelihood
                elif "likelihood_negative" in uncertainty_source.source_id:
                    new_likelihood = max(0.01, min(0.99, evidence.likelihood_negative + perturbation))
                    perturbed.likelihood_negative = new_likelihood
            
            perturbed_evidence.append(perturbed)
        
        return perturbed_evidence
    
    def _bootstrap_sensitivity_ci(self,
                                perturbation_values: np.ndarray,
                                outputs: List[float],
                                baseline: float,
                                n_bootstrap: int = 100) -> Tuple[float, float]:
        """Calculate confidence interval for sensitivity using bootstrap."""
        if len(outputs) < 3:
            return (0.0, 0.0)
        
        bootstrap_sensitivities = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(outputs), size=len(outputs), replace=True)
            resampled_outputs = [outputs[i] for i in indices]
            
            # Calculate sensitivity for this resample
            sensitivity = np.std(resampled_outputs) / baseline if baseline > 0 else 0.0
            bootstrap_sensitivities.append(sensitivity)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_sensitivities, 2.5)
        ci_upper = np.percentile(bootstrap_sensitivities, 97.5)
        
        return (ci_lower, ci_upper)
    
    def _calculate_robustness_score(self,
                                  baseline_confidence: float,
                                  confidence_distribution: np.ndarray) -> float:
        """Calculate robustness score based on confidence distribution."""
        # Robustness = 1 - coefficient of variation
        cv = np.std(confidence_distribution) / np.mean(confidence_distribution)
        robustness = max(0.0, 1.0 - cv)
        
        # Adjust for how close the distribution is to baseline
        mean_deviation = abs(np.mean(confidence_distribution) - baseline_confidence)
        robustness *= (1.0 - mean_deviation)
        
        return max(0.0, min(1.0, robustness))
    
    def _calculate_stability_score(self,
                                 confidence_distribution: np.ndarray,
                                 sensitivity_results: Dict[str, SensitivityResult]) -> float:
        """Calculate stability score based on sensitivity analysis."""
        if not sensitivity_results:
            return 0.5
        
        # Average sensitivity across all parameters
        avg_sensitivity = np.mean([
            result.sensitivity_index for result in sensitivity_results.values()
        ])
        
        # Stability = 1 - average sensitivity
        stability = max(0.0, 1.0 - avg_sensitivity)
        
        # Adjust for confidence distribution stability
        distribution_stability = 1.0 - (np.std(confidence_distribution) / np.mean(confidence_distribution))
        
        # Combine measures
        combined_stability = (stability + distribution_stability) / 2.0
        
        return max(0.0, min(1.0, combined_stability))
    
    def _check_monte_carlo_convergence(self,
                                     confidence_distribution: np.ndarray,
                                     n_simulations: int,
                                     tolerance: float = 0.01) -> bool:
        """Check if Monte Carlo simulation has converged."""
        if n_simulations < 100:
            return False
        
        # Check running mean convergence
        running_means = np.cumsum(confidence_distribution) / np.arange(1, len(confidence_distribution) + 1)
        
        # Check if the last 10% of running means are stable
        last_10_percent = int(0.1 * len(running_means))
        if last_10_percent < 10:
            return False
        
        recent_means = running_means[-last_10_percent:]
        mean_stability = np.std(recent_means) / np.mean(recent_means)
        
        return mean_stability < tolerance
    
    def compare_uncertainty_across_hypotheses(self,
                                            hypotheses: List[BayesianHypothesis],
                                            hypothesis_space: BayesianHypothesisSpace,
                                            n_simulations: int = 1000) -> Dict[str, UncertaintyAnalysisResult]:
        """Compare uncertainty analysis across multiple hypotheses."""
        results = {}
        
        for hypothesis in hypotheses:
            evidence_list = self._get_hypothesis_evidence(hypothesis, hypothesis_space)
            result = self.analyze_uncertainty(
                hypothesis, hypothesis_space, evidence_list, n_simulations=n_simulations
            )
            results[hypothesis.hypothesis_id] = result
        
        return results
    
    def _get_hypothesis_evidence(self,
                               hypothesis: BayesianHypothesis,
                               hypothesis_space: BayesianHypothesisSpace) -> List[BayesianEvidence]:
        """Get evidence relevant to a hypothesis."""
        evidence_ids = hypothesis.supporting_evidence.union(hypothesis.contradicting_evidence)
        evidence_list = []
        
        for evidence_id in evidence_ids:
            evidence = hypothesis_space.get_evidence(evidence_id)
            if evidence:
                evidence_list.append(evidence)
        
        return evidence_list
    
    def generate_uncertainty_report(self, result: UncertaintyAnalysisResult) -> Dict[str, Any]:
        """Generate comprehensive uncertainty report."""
        summary = result.get_uncertainty_summary()
        
        # Add detailed analysis
        detailed_report = {
            "executive_summary": summary,
            "statistical_analysis": {
                "confidence_distribution": {
                    "mean": result.confidence_mean,
                    "std": result.confidence_std,
                    "skewness": result.confidence_skewness,
                    "kurtosis": result.confidence_kurtosis,
                    "percentiles": result.confidence_percentiles
                },
                "robustness_metrics": {
                    "robustness_score": result.robustness_score,
                    "stability_score": result.stability_score,
                    "convergence_achieved": result.convergence_achieved
                }
            },
            "sensitivity_analysis": {
                param: {
                    "sensitivity_index": sens.sensitivity_index,
                    "interpretation": sens.get_interpretation(),
                    "confidence_interval": sens.confidence_interval
                }
                for param, sens in result.sensitivity_results.items()
            },
            "uncertainty_sources": [
                {
                    "source_id": source.source_id,
                    "type": source.uncertainty_type.value,
                    "description": source.description,
                    "magnitude": source.uncertainty_magnitude
                }
                for source in result.uncertainty_sources
            ],
            "recommendations": self._generate_uncertainty_recommendations(result)
        }
        
        return detailed_report
    
    def _generate_uncertainty_recommendations(self, result: UncertaintyAnalysisResult) -> List[str]:
        """Generate recommendations based on uncertainty analysis."""
        recommendations = []
        
        # High uncertainty recommendations
        if result.confidence_std > 0.1:
            recommendations.append("High uncertainty detected - consider gathering additional evidence")
            
            # Identify most uncertain sources
            high_sensitivity = [
                param for param, sens in result.sensitivity_results.items()
                if sens.sensitivity_index > 0.1
            ]
            
            if high_sensitivity:
                recommendations.append(
                    f"Focus on reducing uncertainty in: {', '.join(high_sensitivity[:3])}"
                )
        
        # Low robustness recommendations
        if result.robustness_score < 0.6:
            recommendations.append("Low robustness - results may be sensitive to assumptions")
        
        # Convergence recommendations
        if not result.convergence_achieved:
            recommendations.append("Monte Carlo simulation may not have converged - consider more simulations")
        
        # Distribution shape recommendations
        if abs(result.confidence_skewness) > 1.0:
            recommendations.append("Confidence distribution is highly skewed - check for outliers")
        
        return recommendations