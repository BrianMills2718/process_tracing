"""Data contracts for all pipeline passes."""

from __future__ import annotations

from typing import ClassVar, Optional
from pydantic import BaseModel, Field


# ── Pass 1: Extraction ──────────────────────────────────────────────

class Actor(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'actor_louis_xvi'")
    name: str
    description: str


class Event(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'evt_storming_bastille'")
    description: str
    date: Optional[str] = None
    location: Optional[str] = None


class Mechanism(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'mech_fiscal_crisis'")
    description: str


class Evidence(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'evi_tax_records'")
    description: str
    source_text: str = Field(description="Direct quote from the input text")
    evidence_type: str = Field(
        default="empirical",
        description="'empirical' for facts/events/actions, 'interpretive' for historian arguments/scholarly claims"
    )
    approximate_date: Optional[str] = Field(
        default=None,
        description="Approximate date from the text, e.g. '1788', '1799-11', '1793-06'"
    )


class CausalEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str = Field(description="Nature of the causal link")


class TextHypothesis(BaseModel):
    id: str
    description: str
    source_text: str = Field(description="Quote from text where this hypothesis appears or is implied")


class ExtractionResult(BaseModel):
    summary: str = Field(description="2-3 sentence summary of the text")
    actors: list[Actor] = []
    events: list[Event] = []
    mechanisms: list[Mechanism] = []
    evidence: list[Evidence] = []
    hypotheses_in_text: list[TextHypothesis] = []
    causal_edges: list[CausalEdge] = []


# ── Pass 2: Hypothesis Space ────────────────────────────────────────

class Prediction(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'pred_fiscal_01'")
    description: str = Field(description="What we would expect to observe if the hypothesis is true")


class Hypothesis(BaseModel):
    id: str = Field(description="Unique identifier, e.g. 'h1'")
    description: str
    source: str = Field(description="'text' if from the source, 'generated' if a rival")
    theoretical_basis: str = Field(description="Why this hypothesis is plausible")
    causal_mechanism: str = Field(description="The proposed causal chain")
    observable_predictions: list[Prediction] = Field(
        description="What evidence we would expect to find if this hypothesis is true"
    )


class HypothesisSpace(BaseModel):
    research_question: str
    hypotheses: list[Hypothesis]


# ── Pass 3: Testing ─────────────────────────────────────────────────

class PredictionClassification(BaseModel):
    prediction_id: str
    hypothesis_id: str
    diagnostic_type: str = Field(
        description="One of: hoop, smoking_gun, doubly_decisive, straw_in_the_wind"
    )
    necessity_reasoning: str = Field(
        description="Why passing this test is or is not necessary for the hypothesis"
    )
    sufficiency_reasoning: str = Field(
        description="Why passing this test is or is not sufficient for the hypothesis"
    )


class HypothesisLikelihood(BaseModel):
    """Relative likelihood of one evidence item under one hypothesis.

    Values are on a common positive scale across all hypotheses for the *same*
    evidence item — only the ratios between hypotheses matter. Equal values across
    hypotheses ⇒ the evidence is uninformative. Eliciting the whole vector at once
    (rather than independent pairwise ratios) is what keeps the likelihoods
    coherent: every pairwise ratio is derived from one vector, so reciprocity and
    transitivity hold by construction.
    """
    hypothesis_id: str
    relative_likelihood: float = Field(
        gt=0.0,
        description="Relative likelihood P(E|H) of THIS evidence under THIS hypothesis, "
        "on a common positive scale shared by all hypotheses for this evidence item. "
        "Larger = this hypothesis predicts this evidence more strongly. Equal across "
        "hypotheses = uninformative.",
    )
    diagnostic_type: str = Field(
        description="Van Evera label for how this evidence bears on this hypothesis: "
        "'hoop', 'smoking_gun', 'doubly_decisive', or 'straw_in_the_wind'.",
    )


class EvidenceLikelihood(BaseModel):
    """One evidence item's likelihood vector across all competing hypotheses."""
    evidence_id: str
    hypothesis_likelihoods: list[HypothesisLikelihood] = Field(
        description="Exactly one entry per hypothesis — the relative likelihood of this "
        "evidence under each, on a shared scale.",
    )
    relevance: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How relevant/discriminating this evidence is (temporal proximity, "
        "causal domain, specificity). 0.0=irrelevant, 1.0=directly relevant. Below 0.4 the "
        "item is treated as uninformative regardless of the vector.",
    )
    justification: str = Field(
        description="Why these relative likelihoods — covering both the causal story and how "
        "the evidence was produced (solicited/recorded/survived).",
    )


class TestingResult(BaseModel):
    __test__: ClassVar[bool] = False

    evidence_likelihoods: list[EvidenceLikelihood] = Field(
        description="Per-evidence likelihood vectors across all hypotheses.",
    )
    prediction_classifications: list[PredictionClassification] = Field(
        default_factory=list,
        description="Optional Van Evera classification of hypothesis predictions.",
    )


# ── Bayesian Update ─────────────────────────────────────────────────

class EvidenceUpdate(BaseModel):
    evidence_id: str
    prediction_id: Optional[str] = None
    likelihood_ratio: float
    prior: float
    posterior: float


class HypothesisPosterior(BaseModel):
    hypothesis_id: str
    prior: float
    updates: list[EvidenceUpdate]
    final_posterior: float
    robustness: str = Field(
        default="unknown",
        description="Mechanically computed: 'robust' if posterior driven by few decisive LRs, "
        "'fragile' if driven by many small LRs"
    )
    top_drivers: list[str] = Field(
        default_factory=list,
        description="Evidence IDs of the top 3 most influential LR updates (|log(LR)| largest)"
    )


class SensitivityEntry(BaseModel):
    hypothesis_id: str
    baseline_posterior: float
    posterior_low: float = Field(description="Posterior when top drivers are perturbed against this hypothesis")
    posterior_high: float = Field(description="Posterior when top drivers are perturbed for this hypothesis")
    rank_stable: bool = Field(description="True if this hypothesis keeps its rank across all perturbations")


class PriorSensitivity(BaseModel):
    """Whether the ranking survives reasonable changes to the prior."""
    top_hypothesis_id: str
    stable_under_prior_perturbation: bool = Field(
        description="True if the top-ranked hypothesis stays top when each hypothesis's "
        "prior is independently up- and down-weighted by the perturbation factor."
    )
    perturbation_factor: float = Field(
        default=2.0, description="Multiplicative factor applied to each prior."
    )


class BayesianResult(BaseModel):
    posteriors: list[HypothesisPosterior]
    ranking: list[str] = Field(description="Hypothesis IDs ordered by final posterior, highest first")
    sensitivity: list[SensitivityEntry] = Field(
        default_factory=list,
        description="How posteriors change when the most influential LRs are perturbed ±50%"
    )
    prior_sensitivity: Optional[PriorSensitivity] = Field(
        default=None,
        description="Whether the top-ranked hypothesis is robust to changes in the prior."
    )


# ── Pass 3b: Absence-of-Evidence ───────────────────────────────────

class AbsenceEvaluation(BaseModel):
    hypothesis_id: str
    prediction_id: str
    missing_evidence: str = Field(description="What predicted evidence is absent from the text")
    reasoning: str = Field(description="Why absence is informative given the text's scope")
    severity: str = Field(description="'damaging', 'notable', or 'minor'")
    would_be_extractable: bool = Field(
        description="Would this evidence appear in a text of this scope if it existed?"
    )


class AbsenceResult(BaseModel):
    evaluations: list[AbsenceEvaluation] = []


# ── Pass 5: Refinement (Second Reading) ────────────────────────────

class NewEvidence(BaseModel):
    id: str = Field(description="Must use 'evi_ref_' prefix to distinguish from original extraction")
    description: str
    source_text: str = Field(description="Direct quote from input text")
    evidence_type: str = Field(default="empirical", description="'empirical' or 'interpretive'")
    approximate_date: Optional[str] = None
    rationale: str = Field(description="Why missed initially, why it matters now")


class ReinterpretedEvidence(BaseModel):
    evidence_id: str
    original_type: str
    new_type: str
    reinterpretation: str
    updated_description: Optional[str] = None


class NewCausalEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str
    source_text_support: str = Field(description="Quote from input text")


class SpuriousExtraction(BaseModel):
    item_id: str = Field(description="Evidence ID or 'source_id->target_id' for edges")
    item_type: str = Field(description="'evidence' or 'causal_edge'")
    reason: str


class HypothesisRefinement(BaseModel):
    hypothesis_id: str
    refinement_type: str = Field(
        description="'sharpen_mechanism', 'add_prediction', 'reframe', or 'merge_suggestion'"
    )
    description: str
    updated_causal_mechanism: Optional[str] = None
    new_predictions: list[Prediction] = []


class MissingMechanism(BaseModel):
    description: str
    source_text_support: str
    relevant_hypotheses: list[str]


class RefinementResult(BaseModel):
    new_evidence: list[NewEvidence] = []
    reinterpreted_evidence: list[ReinterpretedEvidence] = []
    new_causal_edges: list[NewCausalEdge] = []
    spurious_extractions: list[SpuriousExtraction] = []
    hypothesis_refinements: list[HypothesisRefinement] = []
    missing_mechanisms: list[MissingMechanism] = []
    analyst_notes: str = Field(description="Free-text analytical notes from the refinement pass")


# ── Pass 4: Synthesis ───────────────────────────────────────────────

class HypothesisVerdict(BaseModel):
    hypothesis_id: str
    status: str = Field(description="'strongly_supported', 'supported', 'weakened', 'eliminated', or 'indeterminate'")
    key_evidence_for: list[str] = Field(description="Evidence IDs that support this hypothesis")
    key_evidence_against: list[str] = Field(description="Evidence IDs that weigh against this hypothesis")
    reasoning: str
    steelman: str = Field(
        description="The STRONGEST possible case for this hypothesis, even if it was eliminated. "
        "What would a passionate advocate argue? What evidence supports it most?"
    )
    posterior_robustness: str = Field(
        default="robust",
        description="'robust' if the posterior is driven by a few decisive tests, "
        "'fragile' if driven by accumulation of many small effects that could individually go either way"
    )


class SynthesisResult(BaseModel):
    verdicts: list[HypothesisVerdict]
    comparative_analysis: str = Field(description="Paragraph comparing hypotheses against each other")
    analytical_narrative: str = Field(
        description="3-5 paragraph analytical narrative with explicit reasoning chains"
    )
    limitations: list[str]
    suggested_further_tests: list[str]


# ── Combined Result ─────────────────────────────────────────────────

class ProcessTracingResult(BaseModel):
    extraction: ExtractionResult
    hypothesis_space: HypothesisSpace
    testing: TestingResult
    absence: AbsenceResult
    bayesian: BayesianResult
    synthesis: SynthesisResult
    refinement: Optional[RefinementResult] = None
    is_refined: bool = False
