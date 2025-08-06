"""
Bayesian data structures and probability models for process tracing.

Implements core Bayesian inference components for causal hypothesis evaluation
with support for multiple competing hypotheses, hierarchical structures,
and Van Evera diagnostic test integration.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel, Field, validator
import networkx as nx
from pathlib import Path


class HypothesisType(Enum):
    """Types of causal hypotheses in process tracing."""
    PRIMARY = "primary"              # Main causal hypothesis
    ALTERNATIVE = "alternative"      # Competing explanation
    NULL = "null"                   # No causal relationship
    COMPOSITE = "composite"         # Multiple causal factors
    CONDITIONAL = "conditional"     # Context-dependent causation
    NECESSARY = "necessary"         # Necessary cause hypothesis
    SUFFICIENT = "sufficient"       # Sufficient cause hypothesis


class EvidenceType(Enum):
    """Van Evera evidence types with Bayesian interpretation."""
    HOOP = "hoop"                   # High necessity, low sufficiency
    SMOKING_GUN = "smoking_gun"     # Low necessity, high sufficiency  
    STRAW_IN_THE_WIND = "straw_in_the_wind"  # Low necessity and sufficiency
    DOUBLY_DECISIVE = "doubly_decisive"      # High necessity and sufficiency


class PriorType(Enum):
    """Methods for assigning prior probabilities."""
    UNIFORM = "uniform"             # Equal probability for all hypotheses
    INFORMED = "informed"           # Based on domain knowledge
    EMPIRICAL = "empirical"         # Based on historical data
    THEORETICAL = "theoretical"     # Based on theory
    SKEPTICAL = "skeptical"         # Conservative low priors
    OPTIMISTIC = "optimistic"       # Liberal high priors


@dataclass
class BayesianHypothesis:
    """
    Represents a causal hypothesis with Bayesian probability tracking.
    
    Supports hierarchical hypothesis structures and multiple evidence types
    for comprehensive causal inference in process tracing contexts.
    """
    hypothesis_id: str
    description: str
    hypothesis_type: HypothesisType
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    likelihood_cache: Dict[str, float] = field(default_factory=dict)
    
    # Hierarchical structure support
    parent_hypothesis: Optional[str] = None
    child_hypotheses: List[str] = field(default_factory=list)
    
    # Van Evera integration
    required_evidence: Set[str] = field(default_factory=set)
    supporting_evidence: Set[str] = field(default_factory=set)
    contradicting_evidence: Set[str] = field(default_factory=set)
    
    # Temporal constraints
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    confidence_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate hypothesis parameters with robust error handling."""
        # Issue #44 Fix: Handle invalid probability values properly
        self.prior_probability = self._validate_and_clamp_probability(
            self.prior_probability, "prior_probability", 0.5
        )
        self.posterior_probability = self._validate_and_clamp_probability(
            self.posterior_probability, "posterior_probability", 0.5
        )
    
    def _validate_and_clamp_probability(self, value: float, param_name: str, default: float) -> float:
        """
        Validate and clamp probability values with robust error handling.
        Issue #44 Fix: Handle invalid probability values properly.
        """
        import math
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Handle None values
        if value is None:
            logger.warning(f"{param_name} is None, using default {default}")
            return default
        
        # Handle non-numeric values
        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning(f"{param_name} is not numeric ({value}), using default {default}")
            return default
        
        # Handle special float values
        if math.isnan(value):
            logger.warning(f"{param_name} is NaN, using default {default}")
            return default
        
        if math.isinf(value):
            logger.warning(f"{param_name} is infinite ({value}), using default {default}")
            return default
        
        # Clamp to valid range [0.0, 1.0]
        if value < 0.0:
            logger.warning(f"{param_name} is negative ({value}), clamping to 0.0")
            return 0.0
        elif value > 1.0:
            logger.warning(f"{param_name} is > 1.0 ({value}), clamping to 1.0")
            return 1.0
        
        return value
    
    def add_child_hypothesis(self, child_id: str) -> None:
        """Add a child hypothesis to this hypothesis."""
        if child_id not in self.child_hypotheses:
            self.child_hypotheses.append(child_id)
    
    def add_evidence(self, evidence_id: str, evidence_type: str) -> None:
        """Add evidence supporting or contradicting this hypothesis."""
        if evidence_type in ["supporting", "required"]:
            self.supporting_evidence.add(evidence_id)
            if evidence_type == "required":
                self.required_evidence.add(evidence_id)
        elif evidence_type == "contradicting":
            self.contradicting_evidence.add(evidence_id)
    
    def update_posterior(self, new_posterior: float, evidence_id: str, likelihood_ratio: float) -> None:
        """Update posterior probability and record the update."""
        old_posterior = self.posterior_probability
        self.posterior_probability = new_posterior
        
        # Ensure timestamp is different from current last_updated
        import time
        new_time = datetime.now()
        if new_time <= self.last_updated:
            time.sleep(0.001)  # Sleep 1ms to ensure different timestamp
            new_time = datetime.now()
        self.last_updated = new_time
        
        # Record update history
        update_record = {
            "timestamp": self.last_updated.isoformat(),
            "evidence_id": evidence_id,
            "old_posterior": old_posterior,
            "new_posterior": new_posterior,
            "likelihood_ratio": likelihood_ratio
        }
        self.update_history.append(update_record)
    
    def calculate_confidence(self) -> float:
        """Calculate confidence level based on evidence strength and coherence."""
        if not self.supporting_evidence:
            self.confidence_level = 0.0
            return self.confidence_level
        
        # Base confidence on posterior probability and evidence count
        evidence_factor = min(len(self.supporting_evidence) / 5.0, 1.0)  # Cap at 5 evidence pieces
        posterior_factor = abs(self.posterior_probability - 0.5) * 2  # Distance from neutrality
        
        # Penalize contradicting evidence
        contradiction_penalty = len(self.contradicting_evidence) * 0.1
        
        self.confidence_level = max(0.0, min(1.0, (evidence_factor + posterior_factor) - contradiction_penalty))
        return self.confidence_level


@dataclass
class BayesianEvidence:
    """
    Represents evidence with Bayesian likelihood ratios for hypothesis testing.
    
    Integrates Van Evera diagnostic test types with probability calculations
    for systematic evidence evaluation in process tracing.
    """
    evidence_id: str
    description: str
    evidence_type: EvidenceType
    source_node_id: str  # Node in the causal graph
    
    # Van Evera diagnostic probabilities
    necessity: float = 0.5      # P(E|H) - probability evidence present given hypothesis true
    sufficiency: float = 0.5    # P(H|E) - probability hypothesis true given evidence present
    
    # Bayesian likelihood ratios (None means use Van Evera templates)
    likelihood_positive: Optional[float] = None    # P(E|H) - likelihood of evidence if hypothesis true
    likelihood_negative: Optional[float] = None    # P(E|¬H) - likelihood of evidence if hypothesis false
    
    # Evidence properties
    reliability: float = 1.0    # Source reliability (0-1)
    strength: float = 1.0       # Evidence strength (0-1)
    independence: bool = True   # Whether evidence is independent of other evidence
    
    # Temporal information
    timestamp: Optional[datetime] = None
    temporal_order: Optional[int] = None
    
    # Metadata
    source_credibility: float = 1.0
    collection_method: str = "unknown"
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate evidence parameters and calculate derived values."""
        # Store whether explicit likelihood values were provided
        self._user_provided_likelihood_positive = self.likelihood_positive is not None
        self._user_provided_likelihood_negative = self.likelihood_negative is not None
        
        # Set defaults if not provided
        if self.likelihood_positive is None:
            self.likelihood_positive = 1.0
        if self.likelihood_negative is None:
            self.likelihood_negative = 1.0
        
        self._validate_probabilities()
        self._calculate_likelihood_ratios()
        self._assign_van_evera_properties()
    
    def _validate_probabilities(self) -> None:
        """Validate all probability values with robust error handling."""
        # Issue #44 Fix: Handle invalid probability values properly
        for attr in ['necessity', 'sufficiency', 'likelihood_positive', 'likelihood_negative', 'reliability', 'strength']:
            value = getattr(self, attr)
            validated_value = self._validate_and_clamp_probability(value, attr, 0.5)
            setattr(self, attr, validated_value)
    
    def _validate_and_clamp_probability(self, value: float, param_name: str, default: float) -> float:
        """
        Validate and clamp probability values with robust error handling.
        Issue #44 Fix: Handle invalid probability values properly.
        """
        import math
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Handle None values
        if value is None:
            logger.warning(f"{param_name} is None, using default {default}")
            return default
        
        # Handle non-numeric values
        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning(f"{param_name} is not numeric ({value}), using default {default}")
            return default
        
        # Handle special float values
        if math.isnan(value):
            logger.warning(f"{param_name} is NaN, using default {default}")
            return default
        
        if math.isinf(value):
            logger.warning(f"{param_name} is infinite ({value}), using default {default}")
            return default
        
        # Clamp to valid range [0.0, 1.0]
        if value < 0.0:
            logger.warning(f"{param_name} is negative ({value}), clamping to 0.0")
            return 0.0
        elif value > 1.0:
            logger.warning(f"{param_name} is > 1.0 ({value}), clamping to 1.0")
            return 1.0
        
        return value
    
    def _calculate_likelihood_ratios(self) -> None:
        """Calculate likelihood ratios from Van Evera probabilities."""
        # Only apply Van Evera templates if user didn't provide explicit values
        if not self._user_provided_likelihood_positive:
            if self.evidence_type == EvidenceType.HOOP:
                # High necessity, low sufficiency
                self.likelihood_positive = max(0.8, self.necessity)
            elif self.evidence_type == EvidenceType.SMOKING_GUN:
                # Low necessity, high sufficiency  
                self.likelihood_positive = max(0.8, self.sufficiency)
            elif self.evidence_type == EvidenceType.DOUBLY_DECISIVE:
                # High necessity and sufficiency
                self.likelihood_positive = max(0.9, max(self.necessity, self.sufficiency))
            elif self.evidence_type == EvidenceType.STRAW_IN_THE_WIND:
                # Low necessity and sufficiency - weak evidence
                self.likelihood_positive = 0.5 + min(0.3, max(self.necessity, self.sufficiency))
        
        if not self._user_provided_likelihood_negative:
            if self.evidence_type == EvidenceType.HOOP:
                self.likelihood_negative = min(0.2, 1.0 - self.necessity)
            elif self.evidence_type == EvidenceType.SMOKING_GUN:
                self.likelihood_negative = min(0.1, 1.0 - self.sufficiency)
            elif self.evidence_type == EvidenceType.DOUBLY_DECISIVE:
                self.likelihood_negative = min(0.1, min(1.0 - self.necessity, 1.0 - self.sufficiency))
            elif self.evidence_type == EvidenceType.STRAW_IN_THE_WIND:
                self.likelihood_negative = 0.5 - min(0.2, min(self.necessity, self.sufficiency))
    
    def _assign_van_evera_properties(self) -> None:
        """Assign Van Evera properties based on evidence type."""
        if self.evidence_type == EvidenceType.HOOP:
            self.necessity = max(0.8, self.necessity)
            self.sufficiency = min(0.4, self.sufficiency)
        
        elif self.evidence_type == EvidenceType.SMOKING_GUN:
            self.necessity = min(0.4, self.necessity)
            self.sufficiency = max(0.8, self.sufficiency)
        
        elif self.evidence_type == EvidenceType.DOUBLY_DECISIVE:
            self.necessity = max(0.8, self.necessity)
            self.sufficiency = max(0.8, self.sufficiency)
        
        elif self.evidence_type == EvidenceType.STRAW_IN_THE_WIND:
            self.necessity = min(0.5, self.necessity)
            self.sufficiency = min(0.5, self.sufficiency)
    
    def get_likelihood_ratio(self) -> float:
        """Calculate the likelihood ratio for this evidence."""
        # Issue #63 Fix: Use epsilon-based floating point comparison
        from .float_utils import float_zero
        
        if float_zero(self.likelihood_negative):
            return float('inf')  # Perfect smoking gun evidence
        return self.likelihood_positive / self.likelihood_negative
    
    def get_adjusted_likelihood_ratio(self) -> float:
        """Get likelihood ratio adjusted for reliability and strength."""
        base_ratio = self.get_likelihood_ratio()
        adjustment_factor = self.reliability * self.strength * self.source_credibility
        
        # Adjust ratio towards 1 (neutrality) based on adjustment factor
        if base_ratio > 1:
            return 1 + (base_ratio - 1) * adjustment_factor
        else:
            return 1 - (1 - base_ratio) * adjustment_factor


class BayesianHypothesisSpace:
    """
    Manages a collection of competing hypotheses with Bayesian inference.
    
    Maintains probability coherence across hypothesis space and supports
    hierarchical hypothesis structures for complex causal inference.
    """
    
    def __init__(self, hypothesis_space_id: str, description: str = ""):
        self.hypothesis_space_id = hypothesis_space_id
        self.description = description
        self.hypotheses: Dict[str, BayesianHypothesis] = {}
        self.evidence: Dict[str, BayesianEvidence] = {}
        self.hypothesis_graph: nx.DiGraph = nx.DiGraph()  # For hierarchical relationships
        
        # Constraint tracking
        self.mutual_exclusivity_groups: List[Set[str]] = []
        self.collective_exhaustiveness: bool = True
        
        # Update tracking
        self.last_updated: datetime = datetime.now()
        self.update_count: int = 0
        
    def add_hypothesis(self, hypothesis: BayesianHypothesis) -> None:
        """Add a hypothesis to the space."""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.hypothesis_graph.add_node(hypothesis.hypothesis_id, hypothesis=hypothesis)
        
        # Add hierarchical relationships
        if hypothesis.parent_hypothesis:
            self.hypothesis_graph.add_edge(hypothesis.parent_hypothesis, hypothesis.hypothesis_id)
        
        for child_id in hypothesis.child_hypotheses:
            if child_id in self.hypotheses:
                self.hypothesis_graph.add_edge(hypothesis.hypothesis_id, child_id)
        
        # Only normalize if there are mutual exclusivity constraints
        if self.mutual_exclusivity_groups:
            self._normalize_probabilities()
    
    def add_evidence(self, evidence: BayesianEvidence) -> None:
        """Add evidence to the space."""
        self.evidence[evidence.evidence_id] = evidence
        self.last_updated = datetime.now()
    
    def add_mutual_exclusivity_group(self, hypothesis_ids: Set[str]) -> None:
        """Add a group of mutually exclusive hypotheses."""
        # Validate all hypotheses exist (sort for deterministic error messages)
        for hyp_id in sorted(hypothesis_ids):
            if hyp_id not in self.hypotheses:
                raise ValueError(f"Hypothesis {hyp_id} not found")
        
        self.mutual_exclusivity_groups.append(hypothesis_ids)
        self._normalize_probabilities()
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[BayesianHypothesis]:
        """Get a hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)
    
    def get_evidence(self, evidence_id: str) -> Optional[BayesianEvidence]:
        """Get evidence by ID."""
        return self.evidence.get(evidence_id)
    
    def get_competing_hypotheses(self, hypothesis_id: str) -> List[BayesianHypothesis]:
        """Get hypotheses that compete with the given hypothesis."""
        competing = []
        
        for group in self.mutual_exclusivity_groups:
            if hypothesis_id in group:
                for competitor_id in group:
                    if competitor_id != hypothesis_id and competitor_id in self.hypotheses:
                        competing.append(self.hypotheses[competitor_id])
        
        return competing
    
    def get_hierarchy_level(self, hypothesis_id: str) -> int:
        """Get the hierarchical level of a hypothesis (0 = root)."""
        if hypothesis_id not in self.hypothesis_graph:
            return 0
        
        # Find shortest path from any root node
        roots = [n for n in self.hypothesis_graph.nodes() if self.hypothesis_graph.in_degree(n) == 0]
        
        if not roots:
            return 0
        
        min_level = float('inf')
        for root in roots:
            if nx.has_path(self.hypothesis_graph, root, hypothesis_id):
                path_length = nx.shortest_path_length(self.hypothesis_graph, root, hypothesis_id)
                min_level = min(min_level, path_length)
        
        return min_level if min_level != float('inf') else 0
    
    def _normalize_probabilities(self) -> None:
        """Ensure probability coherence across hypothesis space."""
        # Normalize each mutual exclusivity group to sum to 1
        normalized_hypotheses = set()
        
        for group in self.mutual_exclusivity_groups:
            group_hypotheses = [self.hypotheses[hyp_id] for hyp_id in group if hyp_id in self.hypotheses]
            
            if not group_hypotheses:
                continue
            
            # Sum current posterior probabilities
            total_prob = sum(h.posterior_probability for h in group_hypotheses)
            
            if total_prob > 0:
                # Normalize to sum to 1
                for hypothesis in group_hypotheses:
                    hypothesis.posterior_probability = hypothesis.posterior_probability / total_prob
                    normalized_hypotheses.add(hypothesis.hypothesis_id)
        
        # If collective exhaustiveness is required, ensure all hypothesis probabilities sum to 1
        # But only for hypotheses not already normalized in mutual exclusivity groups
        if self.collective_exhaustiveness and self.hypotheses:
            remaining_hypotheses = [h for h in self.hypotheses.values() if h.hypothesis_id not in normalized_hypotheses]
            
            if remaining_hypotheses:
                total_prob = sum(h.posterior_probability for h in remaining_hypotheses)
                
                if total_prob > 0:
                    for hypothesis in remaining_hypotheses:
                        hypothesis.posterior_probability = hypothesis.posterior_probability / total_prob
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the hypothesis space."""
        if not self.hypotheses:
            return {"total_hypotheses": 0, "total_evidence": len(self.evidence)}
        
        posterior_probs = [h.posterior_probability for h in self.hypotheses.values()]
        confidence_levels = [h.confidence_level for h in self.hypotheses.values()]
        
        return {
            "total_hypotheses": len(self.hypotheses),
            "total_evidence": len(self.evidence),
            "max_posterior": max(posterior_probs),
            "min_posterior": min(posterior_probs),
            "mean_posterior": np.mean(posterior_probs),
            "std_posterior": np.std(posterior_probs),
            "max_confidence": max(confidence_levels),
            "min_confidence": min(confidence_levels),
            "mean_confidence": np.mean(confidence_levels),
            "mutual_exclusivity_groups": len(self.mutual_exclusivity_groups),
            "last_updated": self.last_updated.isoformat(),
            "update_count": self.update_count
        }


class BayesianProcessTracingModel:
    """
    Main Bayesian process tracing model integrating hypotheses, evidence, and inference.
    
    Provides a complete framework for probabilistic causal inference in process tracing
    with support for multiple hypothesis spaces and complex causal structures.
    """
    
    def __init__(self, model_id: str, description: str = ""):
        self.model_id = model_id
        self.description = description
        self.hypothesis_spaces: Dict[str, BayesianHypothesisSpace] = {}
        self.global_evidence: Dict[str, BayesianEvidence] = {}
        self.causal_graph: Optional[nx.DiGraph] = None
        
        # Model configuration
        self.prior_type: PriorType = PriorType.UNIFORM
        self.independence_assumptions: Dict[str, bool] = {}
        
        # Analysis results
        self.most_likely_hypothesis: Optional[str] = None
        self.model_confidence: float = 0.0
        self.convergence_status: str = "not_started"
        
        # Metadata
        self.created_at: datetime = datetime.now()
        self.last_analysis: Optional[datetime] = None
        self.analysis_count: int = 0
    
    def add_hypothesis_space(self, space: BayesianHypothesisSpace) -> None:
        """Add a hypothesis space to the model."""
        self.hypothesis_spaces[space.hypothesis_space_id] = space
    
    def add_global_evidence(self, evidence: BayesianEvidence) -> None:
        """Add evidence that applies across all hypothesis spaces."""
        self.global_evidence[evidence.evidence_id] = evidence
    
    def set_causal_graph(self, graph: nx.DiGraph) -> None:
        """Set the causal graph for the model."""
        self.causal_graph = graph.copy()
    
    def get_all_hypotheses(self) -> Dict[str, BayesianHypothesis]:
        """Get all hypotheses across all spaces."""
        all_hypotheses = {}
        for space in self.hypothesis_spaces.values():
            all_hypotheses.update(space.hypotheses)
        return all_hypotheses
    
    def get_all_evidence(self) -> Dict[str, BayesianEvidence]:
        """Get all evidence (global and space-specific)."""
        all_evidence = self.global_evidence.copy()
        for space in self.hypothesis_spaces.values():
            all_evidence.update(space.evidence)
        return all_evidence
    
    def find_most_likely_hypothesis(self) -> Optional[Tuple[str, float]]:
        """Find the hypothesis with highest posterior probability."""
        all_hypotheses = self.get_all_hypotheses()
        
        if not all_hypotheses:
            return None
        
        best_hypothesis = max(all_hypotheses.items(), key=lambda x: x[1].posterior_probability)
        self.most_likely_hypothesis = best_hypothesis[0]
        
        return best_hypothesis[0], best_hypothesis[1].posterior_probability
    
    def calculate_model_confidence(self) -> float:
        """Calculate overall model confidence based on hypothesis convergence."""
        all_hypotheses = self.get_all_hypotheses()
        
        if not all_hypotheses:
            self.model_confidence = 0.0
            return self.model_confidence
        
        # Base confidence on:
        # 1. Maximum posterior probability (higher is more confident)
        # 2. Separation between top hypotheses (larger gap is more confident)
        # 3. Individual hypothesis confidence levels
        
        posterior_probs = sorted([h.posterior_probability for h in all_hypotheses.values()], reverse=True)
        confidence_levels = [h.confidence_level for h in all_hypotheses.values()]
        
        # Maximum posterior factor
        max_posterior_factor = max(posterior_probs) if posterior_probs else 0.0
        
        # Separation factor (difference between top two hypotheses)
        separation_factor = 0.0
        if len(posterior_probs) >= 2:
            separation_factor = posterior_probs[0] - posterior_probs[1]
        
        # Average confidence factor
        avg_confidence_factor = np.mean(confidence_levels) if confidence_levels else 0.0
        
        # Combine factors (weighted average)
        self.model_confidence = (
            0.4 * max_posterior_factor +
            0.3 * separation_factor +
            0.3 * avg_confidence_factor
        )
        
        return self.model_confidence
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export model to dictionary format."""
        return {
            "model_id": self.model_id,
            "description": self.description,
            "hypothesis_spaces": {
                space_id: {
                    "description": space.description,
                    "hypotheses": {
                        hyp_id: {
                            "description": hyp.description,
                            "type": hyp.hypothesis_type.value,
                            "prior_probability": hyp.prior_probability,
                            "posterior_probability": hyp.posterior_probability,
                            "confidence_level": hyp.confidence_level
                        }
                        for hyp_id, hyp in space.hypotheses.items()
                    },
                    "evidence_count": len(space.evidence)
                }
                for space_id, space in self.hypothesis_spaces.items()
            },
            "global_evidence_count": len(self.global_evidence),
            "most_likely_hypothesis": self.most_likely_hypothesis,
            "model_confidence": self.model_confidence,
            "convergence_status": self.convergence_status,
            "created_at": self.created_at.isoformat(),
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
            "analysis_count": self.analysis_count
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save model to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.export_to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'BayesianProcessTracingModel':
        """Load model from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model = cls(data["model_id"], data.get("description", ""))
        model.most_likely_hypothesis = data.get("most_likely_hypothesis")
        model.model_confidence = data.get("model_confidence", 0.0)
        model.convergence_status = data.get("convergence_status", "not_started")
        model.analysis_count = data.get("analysis_count", 0)
        
        if data.get("created_at"):
            model.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_analysis"):
            model.last_analysis = datetime.fromisoformat(data["last_analysis"])
        
        return model