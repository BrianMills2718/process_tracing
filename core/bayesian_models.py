"""
Bayesian models for process tracing analysis.

This module implements core Bayesian data structures and analysis methods
for causal process tracing, including evidence assessment, hypothesis 
evaluation, and confidence calculation.

Based on Van Evera diagnostic test methodology with full Bayesian inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Enums and Type Definitions
# -----------------------------------------------------------------------------

class EvidenceType(Enum):
    """Van Evera-style evidence categories."""
    STRAW_IN_THE_WIND = "straw_in_the_wind"  # weakly probative if present/absent
    HOOP = "hoop"                              # must-have for H; absence harms H
    SMOKING_GUN = "smoking_gun"               # strong support for H if present
    DOUBLY_DECISIVE = "doubly_decisive"       # simultaneously supports H and
                                              # disconfirms alternatives

class HypothesisType(Enum):
    """Types of hypotheses in the hypothesis space."""
    PRIMARY = "primary"
    ALTERNATIVE = "alternative"
    CONDITIONAL = "conditional"

class PriorType(Enum):
    """Methods for assigning prior probabilities."""
    UNIFORM = "uniform"
    INFORMED = "informed"
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"

class IndependenceType(Enum):
    """Types of independence relationships between evidence pieces."""
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    CONDITIONALLY_INDEPENDENT = "conditionally_independent"
    REDUNDANT = "redundant"

# -----------------------------------------------------------------------------
# Core Bayesian Data Structures
# -----------------------------------------------------------------------------

@dataclass
class BayesianEvidence:
    evidence_id: str
    description: str
    evidence_type: EvidenceType

    # Scores in [0, 1]
    strength: float = 0.5
    reliability: float = 0.5
    source_credibility: float = 0.5

    # Optional explicit likelihoods. If provided, they override the heuristic map.
    likelihood_given_h: Optional[float] = None
    likelihood_given_not_h: Optional[float] = None

    # Van Evera properties (computed from evidence type if not provided)
    necessity: Optional[float] = None
    sufficiency: Optional[float] = None
    likelihood_positive: Optional[float] = None
    likelihood_negative: Optional[float] = None

    # Optional: where/how it was collected (used for diversity/robustness)
    collection_method: str = "unspecified"
    
    # Temporal information
    timestamp: Optional[datetime] = None
    temporal_order: Optional[int] = None
    
    # Source information
    source_node_id: Optional[str] = None
    
    # Update tracking
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Set Van Evera properties based on evidence type if not provided."""
        if self.likelihood_positive is None or self.likelihood_negative is None:
            self._set_van_evera_properties()

    def _set_van_evera_properties(self):
        """Set Van Evera necessity/sufficiency and likelihood properties."""
        if self.evidence_type == EvidenceType.HOOP:
            # Hoop tests: high necessity, low sufficiency
            self.necessity = self.necessity or 0.9
            self.sufficiency = self.sufficiency or 0.3
            self.likelihood_positive = 0.9
            self.likelihood_negative = 0.2
        elif self.evidence_type == EvidenceType.SMOKING_GUN:
            # Smoking gun: low necessity, high sufficiency
            self.necessity = self.necessity or 0.3
            self.sufficiency = self.sufficiency or 0.9
            self.likelihood_positive = 0.9
            self.likelihood_negative = 0.1
        elif self.evidence_type == EvidenceType.DOUBLY_DECISIVE:
            # Doubly decisive: high necessity and sufficiency
            self.necessity = self.necessity or 0.9
            self.sufficiency = self.sufficiency or 0.9
            self.likelihood_positive = 0.95
            self.likelihood_negative = 0.05
        else:  # STRAW_IN_THE_WIND
            # Straw in the wind: low necessity and sufficiency
            self.necessity = self.necessity or 0.4
            self.sufficiency = self.sufficiency or 0.4
            self.likelihood_positive = 0.6
            self.likelihood_negative = 0.4

    def get_likelihood_ratio(self) -> float:
        """Return LR = p(E|H)/p(E|¬H).
        If explicit likelihoods are unavailable, use Van Evera properties.
        Values are clamped to avoid extremes.
        """
        eps = 1e-6
        
        # Use explicit likelihoods if provided
        if self.likelihood_given_h is not None and self.likelihood_given_not_h is not None:
            num = max(eps, min(1 - eps, self.likelihood_given_h))
            den = max(eps, min(1 - eps, self.likelihood_given_not_h))
            return num / den

        # Use Van Evera properties set in __post_init__
        if self.likelihood_positive is not None and self.likelihood_negative is not None:
            num = max(eps, min(1 - eps, self.likelihood_positive))
            den = max(eps, min(1 - eps, self.likelihood_negative))
            return num / den

        # Fallback heuristic based on strength and type
        s = max(0.0, min(1.0, self.strength))
        if self.evidence_type == EvidenceType.STRAW_IN_THE_WIND:
            base = 1.0 + 0.5 * s    # 1.0–1.5
        elif self.evidence_type == EvidenceType.HOOP:
            base = 1.0 + 1.0 * s    # 1.0–2.0
        elif self.evidence_type == EvidenceType.SMOKING_GUN:
            base = 1.0 + 3.0 * s    # 1.0–4.0
        elif self.evidence_type == EvidenceType.DOUBLY_DECISIVE:
            base = 1.0 + 4.0 * s    # 1.0–5.0
        else:
            base = 1.0 + 0.5 * s

        # Reliability moderates the LR multiplicatively
        r = max(0.0, min(1.0, self.reliability))
        lr = base ** (0.5 + 0.5 * r)
        return max(1e-3, min(1e3, lr))

    def get_adjusted_likelihood_ratio(self) -> float:
        """A more conservative LR adjusting for credibility and reliability."""
        lr = self.get_likelihood_ratio()
        adj = lr ** (0.5 * self.reliability + 0.3 * self.source_credibility)
        return max(1e-3, min(1e3, adj))


@dataclass
class BayesianHypothesis:
    hypothesis_id: str
    description: str
    hypothesis_type: HypothesisType = HypothesisType.PRIMARY
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    
    # Evidence relationships
    supporting_evidence: Set[str] = field(default_factory=set)
    contradicting_evidence: Set[str] = field(default_factory=set)
    required_evidence: Set[str] = field(default_factory=set)
    
    # Hierarchy
    parent_hypothesis: Optional[str] = None
    child_hypotheses: Set[str] = field(default_factory=set)
    
    # Confidence and tracking
    confidence_level: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    update_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Validate probabilities on creation."""
        if not (0.0 <= self.prior_probability <= 1.0):
            raise ValueError("Prior probability must be between 0 and 1")
        if not (0.0 <= self.posterior_probability <= 1.0):
            raise ValueError("Posterior probability must be between 0 and 1")

    def add_child_hypothesis(self, hypothesis_id: str) -> None:
        """Add a child hypothesis."""
        self.child_hypotheses.add(hypothesis_id)

    def add_evidence(self, evidence_id: str, evidence_type: str) -> None:
        """Add evidence of specified type."""
        if evidence_type in ["supporting", "required"]:
            self.supporting_evidence.add(evidence_id)
        if evidence_type == "required":
            self.required_evidence.add(evidence_id)
        elif evidence_type == "contradicting":
            self.contradicting_evidence.add(evidence_id)

    def update_posterior(self, new_posterior: float, evidence_id: str, likelihood_ratio: float) -> None:
        """Update posterior probability with history tracking."""
        old_posterior = self.posterior_probability
        self.posterior_probability = new_posterior
        self.last_updated = datetime.now()
        
        # Record update history
        update_record = {
            "timestamp": self.last_updated,
            "evidence_id": evidence_id,
            "old_posterior": old_posterior,
            "new_posterior": new_posterior,
            "likelihood_ratio": likelihood_ratio
        }
        self.update_history.append(update_record)

    def calculate_confidence(self) -> float:
        """Calculate confidence based on evidence and posterior."""
        if not (self.supporting_evidence or self.contradicting_evidence):
            self.confidence_level = 0.0
            return 0.0

        # Simple confidence calculation
        total_evidence = len(self.supporting_evidence) + len(self.contradicting_evidence)
        support_ratio = len(self.supporting_evidence) / total_evidence if total_evidence > 0 else 0.0
        
        # Combine posterior probability with evidence support
        confidence = (0.6 * self.posterior_probability + 0.4 * support_ratio)
        
        # Penalize if contradicting evidence outweighs supporting
        if len(self.contradicting_evidence) > len(self.supporting_evidence):
            confidence *= 0.7
            
        self.confidence_level = confidence
        return confidence


@dataclass
class BayesianHypothesisSpace:
    """Container for hypotheses and evidence with relationship management."""
    hypothesis_space_id: str = "default"
    description: str = "Bayesian hypothesis space"
    
    hypotheses: Dict[str, BayesianHypothesis] = field(default_factory=dict)
    evidence: Dict[str, BayesianEvidence] = field(default_factory=dict)
    
    # Constraint management
    mutual_exclusivity_groups: List[Set[str]] = field(default_factory=list)
    
    # Graph for hierarchy
    hypothesis_graph: Any = field(default_factory=lambda: None)  # Will be NetworkX graph

    def __post_init__(self):
        """Initialize graph if NetworkX is available."""
        try:
            import networkx as nx
            self.hypothesis_graph = nx.DiGraph()
        except ImportError:
            self.hypothesis_graph = None

    def add_hypothesis(self, hypothesis: BayesianHypothesis) -> None:
        """Add hypothesis to space."""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        if self.hypothesis_graph is not None:
            self.hypothesis_graph.add_node(hypothesis.hypothesis_id)
            
            # Add parent-child edges
            if hypothesis.parent_hypothesis:
                self.hypothesis_graph.add_edge(hypothesis.parent_hypothesis, hypothesis.hypothesis_id)

    def add_evidence(self, evidence: BayesianEvidence) -> None:
        """Add evidence to space."""
        self.evidence[evidence.evidence_id] = evidence

    def get_evidence(self, evidence_id: str) -> Optional[BayesianEvidence]:
        """Retrieve evidence by ID."""
        return self.evidence.get(evidence_id)

    def get_competing_hypotheses(self, hypothesis_id: str) -> List[BayesianHypothesis]:
        """Get hypotheses that compete with the specified one."""
        return [h for hid, h in self.hypotheses.items() if hid != hypothesis_id]

    def add_mutual_exclusivity_group(self, hypothesis_ids: Set[str]) -> None:
        """Add mutual exclusivity constraint."""
        # Verify all hypotheses exist
        for hid in hypothesis_ids:
            if hid not in self.hypotheses:
                raise ValueError(f"Hypothesis {hid} not found in space")
        
        self.mutual_exclusivity_groups.append(hypothesis_ids)

    def get_hierarchy_level(self, hypothesis_id: str) -> int:
        """Get hierarchy level (0 = root, 1 = child, etc.)."""
        if self.hypothesis_graph is None:
            return 0
            
        try:
            import networkx as nx
            # Find shortest path from any root to this node
            roots = [n for n in self.hypothesis_graph.nodes() if self.hypothesis_graph.in_degree(n) == 0]
            if not roots or hypothesis_id in roots:
                return 0
            
            min_level = float('inf')
            for root in roots:
                try:
                    path_length = nx.shortest_path_length(self.hypothesis_graph, root, hypothesis_id)
                    min_level = min(min_level, path_length)
                except nx.NetworkXNoPath:
                    continue
            
            return int(min_level) if min_level != float('inf') else 0
        except ImportError:
            return 0

    def _normalize_probabilities(self) -> None:
        """Normalize probabilities within mutual exclusivity groups."""
        for group in self.mutual_exclusivity_groups:
            hypotheses = [self.hypotheses[hid] for hid in group if hid in self.hypotheses]
            if len(hypotheses) <= 1:
                continue
                
            total_prob = sum(h.posterior_probability for h in hypotheses)
            if total_prob > 0:
                for h in hypotheses:
                    h.posterior_probability /= total_prob

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the space."""
        if not self.hypotheses:
            return {"total_hypotheses": 0, "total_evidence": len(self.evidence)}
            
        posteriors = [h.posterior_probability for h in self.hypotheses.values()]
        
        return {
            "total_hypotheses": len(self.hypotheses),
            "total_evidence": len(self.evidence),
            "max_posterior": max(posteriors),
            "min_posterior": min(posteriors),
            "mean_posterior": float(np.mean(posteriors)),
            "std_posterior": float(np.std(posteriors))
        }


@dataclass  
class BayesianProcessTracingModel:
    """Complete Bayesian process tracing model."""
    model_id: str = "default_model"
    description: str = "Bayesian process tracing model"
    prior_type: PriorType = PriorType.UNIFORM
    
    hypothesis_spaces: Dict[str, BayesianHypothesisSpace] = field(default_factory=dict)
    global_evidence: Dict[str, BayesianEvidence] = field(default_factory=dict)
    causal_graph: Any = field(default_factory=lambda: None)  # NetworkX graph
    
    # Model metadata
    created_at: datetime = field(default_factory=datetime.now)
    analysis_count: int = 0
    most_likely_hypothesis: Optional[str] = None
    model_confidence: float = 0.0

    def add_hypothesis_space(self, space: BayesianHypothesisSpace) -> None:
        """Add hypothesis space to model."""
        self.hypothesis_spaces[space.hypothesis_space_id] = space

    def add_global_evidence(self, evidence: BayesianEvidence) -> None:
        """Add global evidence available to all hypothesis spaces."""
        self.global_evidence[evidence.evidence_id] = evidence

    def set_causal_graph(self, graph: Any) -> None:
        """Set the causal graph for the model."""
        self.causal_graph = graph

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
        """Find hypothesis with highest posterior probability."""
        all_hypotheses = self.get_all_hypotheses()
        if not all_hypotheses:
            return None
            
        best_hypothesis = max(all_hypotheses.items(), 
                            key=lambda item: item[1].posterior_probability)
        
        self.most_likely_hypothesis = best_hypothesis[0]
        return (best_hypothesis[0], best_hypothesis[1].posterior_probability)

    def calculate_model_confidence(self) -> float:
        """Calculate overall model confidence."""
        all_hypotheses = self.get_all_hypotheses()
        if not all_hypotheses:
            self.model_confidence = 0.0
            return 0.0
            
        # Find top hypothesis
        posteriors = [(h.posterior_probability, h.confidence_level) 
                     for h in all_hypotheses.values()]
        posteriors.sort(reverse=True)
        
        if len(posteriors) >= 2:
            # Confidence based on separation and individual confidence
            top_post, top_conf = posteriors[0]
            second_post, _ = posteriors[1]
            separation = top_post - second_post
            confidence = 0.5 * top_conf + 0.5 * separation
        else:
            confidence = posteriors[0][1]
            
        self.model_confidence = confidence
        return confidence

    def export_to_dict(self) -> Dict[str, Any]:
        """Export model to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "description": self.description,
            "prior_type": self.prior_type.value,
            "hypothesis_spaces": {k: v.hypothesis_space_id for k, v in self.hypothesis_spaces.items()},
            "global_evidence_count": len(self.global_evidence),
            "created_at": self.created_at.isoformat(),
            "analysis_count": self.analysis_count,
            "most_likely_hypothesis": self.most_likely_hypothesis,
            "model_confidence": self.model_confidence
        }

    def save_to_file(self, filepath: str) -> None:
        """Save model to JSON file."""
        import json
        from pathlib import Path
        
        data = self.export_to_dict()
        Path(filepath).write_text(json.dumps(data, indent=2))

    @classmethod
    def load_from_file(cls, filepath: str) -> "BayesianProcessTracingModel":
        """Load model from JSON file."""
        import json
        from pathlib import Path
        
        data = json.loads(Path(filepath).read_text())
        model = cls(
            model_id=data["model_id"],
            description=data["description"],
            prior_type=PriorType(data["prior_type"])
        )
        model.analysis_count = data.get("analysis_count", 0)
        model.most_likely_hypothesis = data.get("most_likely_hypothesis")
        model.model_confidence = data.get("model_confidence", 0.0)
        
        if "created_at" in data:
            model.created_at = datetime.fromisoformat(data["created_at"])
            
        return model


# -----------------------------------------------------------------------------
# Evidence weighting and quantification
# -----------------------------------------------------------------------------

@dataclass
class EvidenceWeights:
    """Evidence weight components for multi-dimensional analysis."""
    quality_weight: float
    quantity_weight: float
    diversity_weight: float
    balance_weight: float

    def normalized(self) -> "EvidenceWeights":
        """Return normalized weights that sum to 1.0."""
        v = np.array([
            self.quality_weight,
            self.quantity_weight,
            self.diversity_weight,
            self.balance_weight,
        ], dtype=float)
        v = np.clip(v, 0.0, np.inf)
        s = float(v.sum()) or 1.0
        v /= s
        return EvidenceWeights(*map(float, v))


class EvidenceStrengthQuantifier:
    """Helpers for quantifying evidence strength/diversity/combination."""

    @staticmethod
    def calculate_evidence_diversity(evidence_list: Sequence[BayesianEvidence]) -> float:
        """Calculate diversity score for a set of evidence using entropy."""
        if not evidence_list:
            return 0.0
            
        # Diversity across types and collection methods using normalized entropy
        def entropy(values: Iterable[str]) -> float:
            vals = list(values)
            if not vals:
                return 0.0
            counts: Dict[str, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            p = np.array(list(counts.values()), dtype=float)
            p /= p.sum()
            h = -(p * np.log(p + 1e-12)).sum()
            # Normalize by log K
            hmax = math.log(len(counts)) if len(counts) > 1 else 1.0
            return float(h / hmax)

        type_entropy = entropy(e.evidence_type.value for e in evidence_list)
        method_entropy = entropy(e.collection_method for e in evidence_list)
        return 0.5 * (type_entropy + method_entropy)

    @staticmethod
    def combine_multiple_evidence(
        evidence_list: Sequence[BayesianEvidence],
        independence_assumptions: Dict[str, IndependenceType] = None,
        dependency_graph: Dict[Tuple[str, str], float] = None,
    ) -> float:
        """Combine LRs assuming independence, with optional dampening for dependencies.
        
        Args:
            evidence_list: Evidence to combine
            independence_assumptions: Legacy parameter for compatibility
            dependency_graph: Maps (eid_i, eid_j) -> rho in [0,1] indicating correlation
            
        Returns:
            Combined likelihood ratio
        """
        if not evidence_list:
            return 1.0

        # Work in log-space for numerical stability
        log_lr = 0.0
        for e in evidence_list:
            log_lr += math.log(max(1e-6, e.get_adjusted_likelihood_ratio()))

        # Simple dependency dampening: subtract a penalty per correlated pair
        if dependency_graph:
            for (a, b), rho in dependency_graph.items():
                if a in {e.evidence_id for e in evidence_list} and b in {e.evidence_id for e in evidence_list}:
                    rho = float(np.clip(rho, 0.0, 1.0))
                    log_lr -= rho * 0.1 * abs(log_lr)  # coarse penalization

        return float(np.exp(np.clip(log_lr, -50, 50)))

    @staticmethod
    def _group_evidence_by_independence(
        evidence_list: List[BayesianEvidence], 
        independence_assumptions: Dict[str, IndependenceType]
    ) -> List[List[BayesianEvidence]]:
        """Group evidence pieces by their independence relationships."""
        if not independence_assumptions:
            # If no assumptions provided, treat all as independent
            return [[e] for e in evidence_list]
            
        # Create evidence ID to evidence object mapping
        evidence_map = {e.evidence_id: e for e in evidence_list}
        evidence_ids = list(evidence_map.keys())
        
        # Find dependent pairs
        dependent_pairs = set()
        for pair_key, independence_type in independence_assumptions.items():
            if independence_type in [IndependenceType.DEPENDENT, IndependenceType.REDUNDANT]:
                # Parse pair key (assuming format "id1-id2")
                if '-' in pair_key:
                    id1, id2 = pair_key.split('-', 1)
                    if id1 in evidence_ids and id2 in evidence_ids:
                        dependent_pairs.add((id1, id2))
        
        # Use Union-Find to group dependent evidence
        parent = {eid: eid for eid in evidence_ids}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union dependent pairs
        for id1, id2 in dependent_pairs:
            union(id1, id2)
        
        # Group evidence by their root parent
        groups_dict = {}
        for eid in evidence_ids:
            root = find(eid)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(evidence_map[eid])
        
        # Return list of groups
        return list(groups_dict.values())

    @staticmethod  
    def _combine_dependent_evidence(evidence_group: List[BayesianEvidence]) -> float:
        """Combine dependent evidence using conservative approach."""
        if not evidence_group:
            return 1.0
        
        if len(evidence_group) == 1:
            return evidence_group[0].get_adjusted_likelihood_ratio()
        
        # Get all likelihood ratios
        ratios = [e.get_adjusted_likelihood_ratio() for e in evidence_group]
        
        # Use geometric mean for conservative combination
        if any(r == float('inf') for r in ratios):
            return float('inf')
        
        # Filter out ratios that are effectively 1 (neutral evidence)
        significant_ratios = [r for r in ratios if abs(r - 1.0) > 1e-6]
        
        if not significant_ratios:
            return 1.0
        
        # Use geometric mean with dampening factor
        geometric_mean = math.pow(math.prod(significant_ratios), 1.0 / len(significant_ratios))
        
        # Apply dampening factor (reduces strength due to dependence)
        dampening_factor = 1.0 / math.sqrt(len(evidence_group))
        
        # Combine geometric mean with dampening
        if geometric_mean > 1.0:
            combined_ratio = 1.0 + (geometric_mean - 1.0) * dampening_factor
        else:
            combined_ratio = 1.0 - (1.0 - geometric_mean) * dampening_factor
        
        return max(0.01, combined_ratio)