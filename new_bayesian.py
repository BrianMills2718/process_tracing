"""
Causal Confidence Quantification for Bayesian Process Tracing (Standalone)

This module implements a self-contained version of the design you shared. It
includes:
  • Evidence & hypothesis data structures
  • A small "Van Evera"-style evidence typing
  • Evidence weighting & diversity quantification
  • A CausalConfidenceCalculator that produces a multi-dimensional
    assessment (overall, evidential, causal, coherence, robustness,
    sensitivity), along with supporting metrics
  • A stub LLM interface with deterministic, documented behavior so the
    module runs out-of-the-box. You can plug in your own LLM by replacing
    `require_llm()` (see notes below).

⚠️  NOTE ON BAYESIAN PURITY
This code matches your original interface but keeps the multi-criteria
aggregation you designed. If you want a strictly Bayesian workflow, use the
posterior odds and Bayes factors directly and treat the other diagnostics as
separate reports (do not re-aggregate them into a single score). The module
exposes those pieces so you can adopt either style.

Integration points if you want a real LLM:
- Replace `require_llm()` to return an object with the same methods as
  `StubLLM`.
- Or subclass `BaseLLM` and return your subclass from `require_llm()`.

The file also includes a small `__main__` demo at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Evidence schema (Van Evera-style) and core Bayesian types
# -----------------------------------------------------------------------------

class EvidenceType(Enum):
    """Van Evera-style evidence categories."""
    STRAW_IN_THE_WIND = "straw_in_the_wind"  # weakly probative if present/absent
    HOOP = "hoop"                              # must-have for H; absence harms H
    SMOKING_GUN = "smoking_gun"               # strong support for H if present
    DOUBLY_DECISIVE = "doubly_decisive"       # simultaneously supports H and
                                              # disconfirms alternatives


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

    # Optional: where/how it was collected (used for diversity/robustness)
    collection_method: str = "unspecified"

    def get_likelihood_ratio(self) -> float:
        """Return LR = p(E|H)/p(E|¬H).
        If explicit likelihoods are unavailable, use a simple mapping based on
        Van Evera type and strength. Values are clamped to avoid extremes.
        """
        eps = 1e-6
        if self.likelihood_given_h is not None and self.likelihood_given_not_h is not None:
            num = max(eps, min(1 - eps, self.likelihood_given_h))
            den = max(eps, min(1 - eps, self.likelihood_given_not_h))
            return num / den

        # Heuristic LR templates (rough defaults; adjust to your domain)
        s = max(0.0, min(1.0, self.strength))
        if self.evidence_type is EvidenceType.STRAW_IN_THE_WIND:
            base = 1.0 + 0.5 * s    # 1.0–1.5
        elif self.evidence_type is EvidenceType.HOOP:
            base = 1.0 + 1.0 * s    # 1.0–2.0
        elif self.evidence_type is EvidenceType.SMOKING_GUN:
            base = 1.0 + 3.0 * s    # 1.0–4.0
        elif self.evidence_type is EvidenceType.DOUBLY_DECISIVE:
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
    prior_probability: float
    posterior_probability: float
    supporting_evidence: Set[str] = field(default_factory=set)
    contradicting_evidence: Set[str] = field(default_factory=set)


@dataclass
class BayesianHypothesisSpace:
    """Container for hypotheses and evidence."""
    hypotheses: Dict[str, BayesianHypothesis] = field(default_factory=dict)
    evidence: Dict[str, BayesianEvidence] = field(default_factory=dict)

    def add_hypothesis(self, h: BayesianHypothesis) -> None:
        self.hypotheses[h.hypothesis_id] = h

    def add_evidence(self, e: BayesianEvidence) -> None:
        self.evidence[e.evidence_id] = e

    def get_evidence(self, evidence_id: str) -> Optional[BayesianEvidence]:
        return self.evidence.get(evidence_id)

    def get_competing_hypotheses(self, hypothesis_id: str) -> List[BayesianHypothesis]:
        return [h for hid, h in self.hypotheses.items() if hid != hypothesis_id]


# Stub for completeness
@dataclass
class BayesianProcessTracingModel:
    name: str = "default_process_model"


# -----------------------------------------------------------------------------
# Evidence weighting & diversity
# -----------------------------------------------------------------------------

@dataclass
class EvidenceWeights:
    quality_weight: float
    quantity_weight: float
    diversity_weight: float
    balance_weight: float

    def normalized(self) -> "EvidenceWeights":
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
        dependency_graph: Dict[Tuple[str, str], float] | None = None,
    ) -> float:
        """Combine LRs assuming independence, with optional dampening for
        dependencies. `dependency_graph` maps (eid_i, eid_j) -> rho in [0,1]
        indicating correlation; higher rho -> stronger dampening.
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


# -----------------------------------------------------------------------------
# LLM interface (pluggable) and stubs
# -----------------------------------------------------------------------------

class LLMRequiredError(RuntimeError):
    pass


@dataclass
class ConfidenceThresholds:
    very_high_threshold: float = 0.85
    high_threshold: float = 0.70
    moderate_threshold: float = 0.50
    low_threshold: float = 0.30

    # Extras used by the calculator
    logical_coherence: float = 0.6
    independence_score: float = 0.6
    posterior_uncertainty: float = 0.2


@dataclass
class CausalAssessment:
    mechanism_completeness: float
    temporal_ordering: float


class BaseLLM:
    """Interface expected by the calculator."""

    def assess_confidence_thresholds(self, *, evidence_quality: str, hypothesis_complexity: str, domain_context: str) -> ConfidenceThresholds:
        raise NotImplementedError

    def determine_confidence_weights(self, *, evidence_quality: str, hypothesis_complexity: str, domain_context: str) -> EvidenceWeights:
        raise NotImplementedError

    def assess_causal_mechanism(self, *, hypothesis_description: str, evidence_chain: str, temporal_sequence: str) -> CausalAssessment:
        raise NotImplementedError

    def determine_causal_weights(self, *, hypothesis_description: str, evidence_context: str, domain_context: str) -> EvidenceWeights:
        raise NotImplementedError

    def determine_robustness_weights(self, *, evidence_context: str, domain_context: str) -> EvidenceWeights:
        raise NotImplementedError

    def determine_overall_confidence_weights(self, *, component_scores: str, domain_context: str) -> EvidenceWeights:
        raise NotImplementedError


class StubLLM(BaseLLM):
    """Deterministic, documented behavior so the module runs without an LLM.

    The stub lightly parses numeric hints from the provided text to modulate
    thresholds/weights, but remains bounded and normalized.
    """

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def assess_confidence_thresholds(self, *, evidence_quality: str, hypothesis_complexity: str, domain_context: str) -> ConfidenceThresholds:
        # Look for numbers like "Average reliability: 0.82" to nudge thresholds.
        rel = self._extract_first_float(evidence_quality) or 0.6
        # Higher reliability -> higher bars for very_high/high
        vh = float(np.clip(0.80 + 0.10 * (rel - 0.6), 0.75, 0.92))
        h = float(np.clip(0.65 + 0.08 * (rel - 0.6), 0.55, vh - 0.05))
        m = 0.50
        l = 0.30
        logical = float(np.clip(0.55 + 0.20 * (rel - 0.6), 0.40, 0.80))
        indep = float(np.clip(0.55 + 0.15 * (rel - 0.6), 0.35, 0.85))
        post_unc = float(np.clip(0.25 - 0.10 * (rel - 0.6), 0.05, 0.35))
        return ConfidenceThresholds(vh, h, m, l, logical, indep, post_unc)

    def determine_confidence_weights(self, *, evidence_quality: str, hypothesis_complexity: str, domain_context: str) -> EvidenceWeights:
        # Slightly more weight on quality and diversity
        return EvidenceWeights(0.35, 0.20, 0.30, 0.15).normalized()

    def assess_causal_mechanism(self, *, hypothesis_description: str, evidence_chain: str, temporal_sequence: str) -> CausalAssessment:
        # If an evidence chain exists, bump completeness; pretend temporal ordering is decent.
        has_chain = len(evidence_chain.strip()) > 0
        return CausalAssessment(
            mechanism_completeness=0.65 if has_chain else 0.45,
            temporal_ordering=0.60 if has_chain else 0.50,
        )

    def determine_causal_weights(self, *, hypothesis_description: str, evidence_context: str, domain_context: str) -> EvidenceWeights:
        # Emphasize posterior and mechanism completeness
        return EvidenceWeights(0.40, 0.20, 0.25, 0.15).normalized()

    def determine_robustness_weights(self, *, evidence_context: str, domain_context: str) -> EvidenceWeights:
        # Emphasize source diversity and reliability consistency equally
        return EvidenceWeights(0.30, 0.30, 0.25, 0.15).normalized()

    def determine_overall_confidence_weights(self, *, component_scores: str, domain_context: str) -> EvidenceWeights:
        # Balanced overall aggregation
        return EvidenceWeights(0.30, 0.25, 0.25, 0.20).normalized()

    def _extract_first_float(self, s: str) -> Optional[float]:
        import re
        m = re.search(r"([0-1]?(?:\.\d+))", s)
        try:
            return float(m.group(1)) if m else None
        except Exception:
            return None


def require_llm() -> BaseLLM:
    """Return an LLM-like object. Replace with a real provider in production."""
    return StubLLM()


# -----------------------------------------------------------------------------
# Confidence assessment domain model
# -----------------------------------------------------------------------------

class ConfidenceType(Enum):
    OVERALL = "overall"
    CAUSAL = "causal"
    EVIDENTIAL = "evidential"
    COHERENCE = "coherence"
    ROBUSTNESS = "robustness"
    SENSITIVITY = "sensitivity"


class ConfidenceLevel(Enum):
    VERY_HIGH = ("very_high", 0.85, 1.00)
    HIGH = ("high", 0.70, 0.85)
    MODERATE = ("moderate", 0.50, 0.70)
    LOW = ("low", 0.30, 0.50)
    VERY_LOW = ("very_low", 0.00, 0.30)

    def __init__(self, label: str, min_threshold: float, max_threshold: float):
        self.label = label
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    @classmethod
    def from_score(cls, score: float, thresholds: Optional[ConfidenceThresholds] = None) -> "ConfidenceLevel":
        if thresholds:
            if score >= thresholds.very_high_threshold:
                return cls.VERY_HIGH
            if score >= thresholds.high_threshold:
                return cls.HIGH
            if score >= thresholds.moderate_threshold:
                return cls.MODERATE
            if score >= thresholds.low_threshold:
                return cls.LOW
            return cls.VERY_LOW
        for level in cls:
            if level.min_threshold <= score < level.max_threshold:
                return level
        return cls.VERY_HIGH if score >= 0.85 else cls.VERY_LOW


@dataclass
class ConfidenceAssessment:
    hypothesis_id: str
    overall_confidence: float
    confidence_components: Dict[ConfidenceType, float]
    confidence_level: ConfidenceLevel

    evidence_quality_score: float = 0.0
    logical_coherence_score: float = 0.0
    robustness_score: float = 0.0
    sensitivity_score: float = 0.0

    evidence_count: int = 0
    contradicting_evidence_count: int = 0
    evidence_diversity_score: float = 0.0
    posterior_probability: float = 0.0
    prior_probability: float = 0.0

    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    uncertainty_sources: List[str] = field(default_factory=list)

    assessment_timestamp: datetime = field(default_factory=datetime.now)
    assessment_method: str = "multi_dimensional"

    def get_interpretation(self) -> str:
        interpretations = {
            ConfidenceLevel.VERY_HIGH: "Very high confidence - strong evidential support with minimal uncertainty",
            ConfidenceLevel.HIGH: "High confidence - good evidential support with manageable uncertainty",
            ConfidenceLevel.MODERATE: "Moderate confidence - adequate evidence but notable uncertainty remains",
            ConfidenceLevel.LOW: "Low confidence - limited evidence or significant uncertainty",
            ConfidenceLevel.VERY_LOW: "Very low confidence - insufficient evidence or high uncertainty",
        }
        return interpretations.get(self.confidence_level, "Unknown confidence level")

    def get_recommendations(self) -> List[str]:
        recommendations: List[str] = []
        if self.confidence_level in (ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH):
            recommendations.append("Hypothesis well-supported - suitable for decision-making")
            if self.evidence_diversity_score < 0.6:
                recommendations.append("Consider seeking additional diverse evidence sources")
        elif self.confidence_level is ConfidenceLevel.MODERATE:
            recommendations.append("Hypothesis moderately supported - additional evidence recommended")
            if self.evidence_quality_score < 0.6:
                recommendations.append("Focus on improving evidence quality")
            if self.logical_coherence_score < 0.6:
                recommendations.append("Review logical consistency of causal mechanism")
        else:
            recommendations.append("Hypothesis insufficiently supported - substantial additional evidence needed")
            if self.evidence_count < 3:
                recommendations.append("Seek additional supporting evidence")
            if self.contradicting_evidence_count > self.evidence_count:
                recommendations.append("Address contradicting evidence")
        return recommendations


# -----------------------------------------------------------------------------
# CausalConfidenceCalculator
# -----------------------------------------------------------------------------

class CausalConfidenceCalculator:
    """Calculates multi-dimensional confidence assessments for causal hypotheses."""

    def __init__(self):
        self.llm: BaseLLM = require_llm()  # Replace in production
        self.evidence_quantifier = EvidenceStrengthQuantifier()
        self.assessment_history: List[ConfidenceAssessment] = []
        self._confidence_thresholds: Optional[ConfidenceThresholds] = None
        self._causal_assessment: Optional[CausalAssessment] = None

    # --- Public API ---------------------------------------------------------

    def calculate_confidence(
        self,
        hypothesis: BayesianHypothesis,
        hypothesis_space: BayesianHypothesisSpace,
        evidence_list: Optional[List[BayesianEvidence]] = None,
    ) -> ConfidenceAssessment:
        if evidence_list is None:
            evidence_list = self._get_hypothesis_evidence(hypothesis, hypothesis_space)

        self._update_confidence_thresholds(hypothesis, evidence_list, hypothesis_space)

        components: Dict[ConfidenceType, float] = {}
        components[ConfidenceType.EVIDENTIAL] = self._calculate_evidential_confidence(hypothesis, evidence_list)
        components[ConfidenceType.CAUSAL] = self._calculate_causal_confidence(hypothesis, evidence_list, hypothesis_space)
        components[ConfidenceType.COHERENCE] = self._calculate_coherence_confidence(hypothesis, evidence_list, hypothesis_space)
        components[ConfidenceType.ROBUSTNESS] = self._calculate_robustness_confidence(hypothesis, evidence_list)
        components[ConfidenceType.SENSITIVITY] = self._calculate_sensitivity_confidence(hypothesis, evidence_list)

        overall = self._calculate_overall_confidence(components)

        evidence_quality = self._calculate_evidence_quality_score(evidence_list)
        logical_coherence = components[ConfidenceType.COHERENCE]
        robustness = components[ConfidenceType.ROBUSTNESS]
        sensitivity = components[ConfidenceType.SENSITIVITY]

        supporting = [e for e in evidence_list if e.evidence_id in hypothesis.supporting_evidence]
        contradicting = [e for e in evidence_list if e.evidence_id in hypothesis.contradicting_evidence]
        diversity = self.evidence_quantifier.calculate_evidence_diversity(evidence_list)

        ci = self._calculate_confidence_interval(hypothesis, evidence_list, overall)
        uncertainty_sources = self._identify_uncertainty_sources(hypothesis, evidence_list, components)

        assessment = ConfidenceAssessment(
            hypothesis_id=hypothesis.hypothesis_id,
            overall_confidence=overall,
            confidence_components=components,
            confidence_level=ConfidenceLevel.from_score(overall, self._confidence_thresholds),
            evidence_quality_score=evidence_quality,
            logical_coherence_score=logical_coherence,
            robustness_score=robustness,
            sensitivity_score=sensitivity,
            evidence_count=len(supporting),
            contradicting_evidence_count=len(contradicting),
            evidence_diversity_score=diversity,
            posterior_probability=hypothesis.posterior_probability,
            prior_probability=hypothesis.prior_probability,
            confidence_interval=ci,
            uncertainty_sources=uncertainty_sources,
        )
        self.assessment_history.append(assessment)
        return assessment

    def compare_hypotheses_confidence(self, hypotheses: List[BayesianHypothesis], hypothesis_space: BayesianHypothesisSpace) -> Dict[str, ConfidenceAssessment]:
        return {h.hypothesis_id: self.calculate_confidence(h, hypothesis_space) for h in hypotheses}

    def get_confidence_trends(self, hypothesis_id: str) -> List[ConfidenceAssessment]:
        return [a for a in self.assessment_history if a.hypothesis_id == hypothesis_id]

    def get_confidence_summary(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        return {
            "hypothesis_id": assessment.hypothesis_id,
            "overall_confidence": assessment.overall_confidence,
            "confidence_level": assessment.confidence_level.label,
            "confidence_interval": assessment.confidence_interval,
            "evidence_count": assessment.evidence_count,
            "evidence_quality": assessment.evidence_quality_score,
            "main_strengths": self._identify_confidence_strengths(assessment),
            "main_weaknesses": assessment.uncertainty_sources,
            "recommendations": assessment.get_recommendations(),
            "interpretation": assessment.get_interpretation(),
        }

    # --- Internals ----------------------------------------------------------

    def _get_hypothesis_evidence(self, hypothesis: BayesianHypothesis, hypothesis_space: BayesianHypothesisSpace) -> List[BayesianEvidence]:
        out: List[BayesianEvidence] = []
        for eid in hypothesis.supporting_evidence.union(hypothesis.contradicting_evidence):
            ev = hypothesis_space.get_evidence(eid)
            if ev is not None:
                out.append(ev)
        return out

    def _update_confidence_thresholds(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence], hypothesis_space: BayesianHypothesisSpace) -> None:
        evidence_quality = self._describe_evidence_quality(evidence_list)
        hypothesis_complexity = self._describe_hypothesis_complexity(hypothesis, hypothesis_space)
        self._confidence_thresholds = self.llm.assess_confidence_thresholds(
            evidence_quality=evidence_quality,
            hypothesis_complexity=hypothesis_complexity,
            domain_context="Process tracing analysis with Van Evera methodology",
        )

    def _describe_evidence_quality(self, evidence_list: List[BayesianEvidence]) -> str:
        if not evidence_list:
            return "No evidence available"
        avg_reliability = float(np.mean([e.reliability for e in evidence_list]))
        avg_strength = float(np.mean([e.strength for e in evidence_list]))
        types = ", ".join(sorted(set(e.evidence_type.value for e in evidence_list)))
        return (
            f"Evidence set with {len(evidence_list)} pieces.\n"
            f"Average reliability: {avg_reliability:.2f}\n"
            f"Average strength: {avg_strength:.2f}\n"
            f"Evidence types: {types}"
        )

    def _describe_hypothesis_complexity(self, hypothesis: BayesianHypothesis, hypothesis_space: BayesianHypothesisSpace) -> str:
        competitors = hypothesis_space.get_competing_hypotheses(hypothesis.hypothesis_id)
        return (
            f"Hypothesis with {len(hypothesis.supporting_evidence)} supporting evidence.\n"
            f"Competing hypotheses: {len(competitors)}\n"
            f"Posterior probability: {hypothesis.posterior_probability:.2f}\n"
            f"Prior probability: {hypothesis.prior_probability:.2f}"
        )

    def _calculate_evidential_confidence(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence]) -> float:
        if not evidence_list:
            return 0.0
        # Quality: mean of strength*reliability
        quality = float(np.mean([e.strength * e.reliability for e in evidence_list]))
        # Quantity: diminishing returns
        quantity = 1.0 - math.exp(-len(evidence_list) / 3.0)
        # Diversity: entropy-based
        diversity = self.evidence_quantifier.calculate_evidence_diversity(evidence_list)
        # Balance: fraction supporting among relevant evidence
        sup = len(hypothesis.supporting_evidence)
        contra = len(hypothesis.contradicting_evidence)
        total = sup + contra
        balance = sup / total if total > 0 else 0.5

        weights = self.llm.determine_confidence_weights(
            evidence_quality=self._describe_evidence_quality(evidence_list),
            hypothesis_complexity=f"Hypothesis with {len(evidence_list)} evidence pieces",
            domain_context="Evidential confidence calculation",
        ).normalized()
        score = (
            weights.quality_weight * quality
            + weights.quantity_weight * quantity
            + weights.diversity_weight * diversity
            + weights.balance_weight * balance
        )
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_causal_confidence(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence], hypothesis_space: BayesianHypothesisSpace) -> float:
        posterior_component = float(np.clip(hypothesis.posterior_probability, 0.0, 1.0))
        lrs = [e.get_likelihood_ratio() for e in evidence_list]
        if lrs:
            log_ratios = [math.log(max(1e-3, lr)) for lr in lrs]
            mean_log = float(np.mean(log_ratios))
            combined = math.exp(mean_log)
            ratio_component = 1.0 / (1.0 + math.exp(-math.log(max(1e-6, combined))))
        else:
            ratio_component = 0.5

        evidence_chain = "; ".join([e.description[:100] for e in evidence_list[:5]])
        causal_assessment = self.llm.assess_causal_mechanism(
            hypothesis_description=hypothesis.description,
            evidence_chain=evidence_chain,
            temporal_sequence="Temporal ordering based on evidence sequence",
        )
        self._causal_assessment = causal_assessment

        weights = self.llm.determine_causal_weights(
            hypothesis_description=hypothesis.description,
            evidence_context=f"Evidence set with {len(evidence_list)} pieces",
            domain_context="Causal confidence calculation",
        ).normalized()

        score = (
            weights.quality_weight * posterior_component
            + weights.quantity_weight * ratio_component
            + weights.diversity_weight * causal_assessment.mechanism_completeness
            + weights.balance_weight * causal_assessment.temporal_ordering
        )
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_coherence_confidence(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence], hypothesis_space: BayesianHypothesisSpace) -> float:
        if not evidence_list:
            return 0.5
        van_types = [e.evidence_type for e in evidence_list]
        type_diversity = len(set(van_types)) / max(1, len(EvidenceType))
        penalty = 0.0
        smoking = [e for e in evidence_list if e.evidence_type is EvidenceType.SMOKING_GUN]
        hoops = [e for e in evidence_list if e.evidence_type is EvidenceType.HOOP]
        if smoking and not hoops:
            penalty += 0.1
        ratios = [e.get_likelihood_ratio() for e in evidence_list]
        if ratios:
            if np.var(ratios) > 100:
                penalty += 0.1
        competitors = hypothesis_space.get_competing_hypotheses(hypothesis.hypothesis_id)
        if competitors:
            diffs = [abs(hypothesis.posterior_probability - c.posterior_probability) for c in competitors]
            separation = max(diffs) if diffs else 0.0
        else:
            separation = hypothesis.posterior_probability

        if not self._confidence_thresholds:
            raise LLMRequiredError("Confidence thresholds not set - LLM assessment required")
        base = self._confidence_thresholds.logical_coherence
        score = base - penalty + 0.1 * type_diversity + 0.1 * separation
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_robustness_confidence(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence]) -> float:
        if not evidence_list:
            return 0.5
        methods = {e.collection_method for e in evidence_list}
        source_div = min(1.0, len(methods) / 3.0)
        reliabilities = np.array([e.reliability for e in evidence_list], dtype=float)
        r_mean = float(reliabilities.mean())
        r_std = float(reliabilities.std())
        r_consistency = 1.0 - float(np.clip(r_std, 0.0, 1.0))
        strengths = np.array([e.strength for e in evidence_list], dtype=float)
        s_balance = 1.0 - abs(0.7 - float(strengths.mean()))

        if not self._confidence_thresholds:
            raise LLMRequiredError("Confidence thresholds not set - LLM assessment required")
        independence = self._confidence_thresholds.independence_score

        weights = self.llm.determine_robustness_weights(
            evidence_context=f"Evidence from {len(methods)} sources",
            domain_context="Robustness confidence calculation",
        ).normalized()

        score = (
            weights.quality_weight * source_div
            + weights.quantity_weight * r_consistency
            + weights.diversity_weight * s_balance
            + weights.balance_weight * independence
        )
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_sensitivity_confidence(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence]) -> float:
        if len(evidence_list) < 2:
            return 0.5
        scores: List[float] = []
        original_odds = (
            hypothesis.posterior_probability / max(1e-6, 1.0 - hypothesis.posterior_probability)
            if 0.0 < hypothesis.posterior_probability < 1.0
            else 1.0
        )
        for i, _ in enumerate(evidence_list):
            remaining = evidence_list[:i] + evidence_list[i + 1 :]
            combined = self.evidence_quantifier.combine_multiple_evidence(remaining, {})
            if original_odds > 0:
                relative_change = abs(combined - original_odds) / original_odds
                sensitivity = 1.0 - float(np.clip(relative_change, 0.0, 1.0))
            else:
                sensitivity = 0.5
            scores.append(sensitivity)
        return float(np.clip(float(np.mean(scores)), 0.0, 1.0))

    def _calculate_overall_confidence(self, components: Dict[ConfidenceType, float]) -> float:
        weights = self.llm.determine_overall_confidence_weights(
            component_scores=str({k.value: v for k, v in components.items()}),
            domain_context="Overall confidence aggregation",
        ).normalized()
        blended_rs = 0.6 * components.get(ConfidenceType.ROBUSTNESS, 0.0) + 0.4 * components.get(ConfidenceType.SENSITIVITY, 0.0)
        score = (
            weights.quality_weight * components.get(ConfidenceType.EVIDENTIAL, 0.0)
            + weights.quantity_weight * components.get(ConfidenceType.CAUSAL, 0.0)
            + weights.diversity_weight * components.get(ConfidenceType.COHERENCE, 0.0)
            + weights.balance_weight * blended_rs
        )
        return float(np.clip(score, 0.0, 1.0))

    def _calculate_evidence_quality_score(self, evidence_list: List[BayesianEvidence]) -> float:
        if not evidence_list:
            return 0.0
        qs: List[float] = []
        for e in evidence_list:
            lr_bonus = 1.0 if e.get_likelihood_ratio() > 1.0 else 0.5
            q = 0.4 * e.reliability + 0.3 * e.strength + 0.2 * e.source_credibility + 0.1 * lr_bonus
            qs.append(q)
        return float(np.mean(qs))

    def _calculate_confidence_interval(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence], point_confidence: float) -> Tuple[float, float]:
        if evidence_list:
            uncertainties = [1.0 - (e.reliability * e.strength) for e in evidence_list]
            avg_unc = float(np.mean(uncertainties))
        else:
            avg_unc = 0.5
        sample_unc = 1.0 / math.sqrt(max(1, len(evidence_list)))
        if not self._confidence_thresholds:
            raise LLMRequiredError("Confidence thresholds not set - LLM assessment required")
        post_unc = self._confidence_thresholds.posterior_uncertainty
        total_unc = math.sqrt(avg_unc ** 2 + sample_unc ** 2 + post_unc ** 2) / math.sqrt(3)
        margin = total_unc * point_confidence
        lo = max(0.0, point_confidence - margin)
        hi = min(1.0, point_confidence + margin)
        return (lo, hi)

    def _identify_uncertainty_sources(self, hypothesis: BayesianHypothesis, evidence_list: List[BayesianEvidence], components: Dict[ConfidenceType, float]) -> List[str]:
        src: List[str] = []
        for t, s in components.items():
            if s < 0.5:
                src.append(f"Low {t.value} confidence ({s:.2f})")
        if len(evidence_list) < 3:
            src.append("Limited evidence quantity")
        if evidence_list:
            avg_rel = float(np.mean([e.reliability for e in evidence_list]))
            if avg_rel < 0.6:
                src.append("Low evidence reliability")
            avg_str = float(np.mean([e.strength for e in evidence_list]))
            if avg_str < 0.6:
                src.append("Weak evidence strength")
        if len(hypothesis.contradicting_evidence) > len(hypothesis.supporting_evidence):
            src.append("Substantial contradicting evidence")
        if hypothesis.posterior_probability < 0.6:
            src.append("Low posterior probability")
        return src

    def _identify_confidence_strengths(self, a: ConfidenceAssessment) -> List[str]:
        strengths: List[str] = []
        for t, s in a.confidence_components.items():
            if s >= 0.7:
                strengths.append(f"Strong {t.value} support ({s:.2f})")
        if a.evidence_diversity_score >= 0.7:
            strengths.append("High evidence diversity")
        if a.evidence_count >= 5:
            strengths.append("Substantial evidence quantity")
        if a.evidence_quality_score >= 0.8:
            strengths.append("High-quality evidence")
        return strengths


# -----------------------------------------------------------------------------
# Small demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a tiny scenario
    e1 = BayesianEvidence(
        evidence_id="E1",
        description="Leaked memo aligns with the proposed mechanism",
        evidence_type=EvidenceType.SMOKING_GUN,
        strength=0.8,
        reliability=0.75,
        source_credibility=0.9,
        collection_method="documents",
    )
    e2 = BayesianEvidence(
        evidence_id="E2",
        description="Expected precursor event observed before outcome",
        evidence_type=EvidenceType.HOOP,
        strength=0.6,
        reliability=0.7,
        source_credibility=0.7,
        collection_method="observational",
    )
    e3 = BayesianEvidence(
        evidence_id="E3",
        description="Minor indicator consistent with hypothesis",
        evidence_type=EvidenceType.STRAW_IN_THE_WIND,
        strength=0.5,
        reliability=0.6,
        source_credibility=0.6,
        collection_method="interviews",
    )

    H = BayesianHypothesis(
        hypothesis_id="H1",
        description="Policy X caused Outcome Y via Mechanism Z",
        prior_probability=0.3,
        posterior_probability=0.72,
        supporting_evidence={"E1", "E2", "E3"},
        contradicting_evidence=set(),
    )

    space = BayesianHypothesisSpace()
    space.add_hypothesis(H)
    for e in (e1, e2, e3):
        space.add_evidence(e)

    calc = CausalConfidenceCalculator()
    assessment = calc.calculate_confidence(H, space)

    from pprint import pprint

    print("\n=== Confidence Assessment ===")
    pprint(calc.get_confidence_summary(assessment))
    print("\nComponents:")
    for k, v in assessment.confidence_components.items():
        print(f"  {k.value:>12}: {v:.3f}")
    print("\nInterval:", assessment.confidence_interval)
