"""Data contracts for all pipeline passes."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


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


class EvidenceEvaluation(BaseModel):
    prediction_id: Optional[str] = Field(
        None, description="ID of the most relevant prediction, or null if no prediction directly applies"
    )
    evidence_id: str
    hypothesis_id: str
    finding: str = Field(description="'pass', 'fail', or 'ambiguous'")
    p_e_given_h: float = Field(
        ge=0.0, le=1.0,
        description="Probability of observing this evidence if the hypothesis is true"
    )
    p_e_given_not_h: float = Field(
        ge=0.0, le=1.0,
        description="Probability of observing this evidence if the hypothesis is false"
    )
    justification: str = Field(
        description="Why these probability estimates are appropriate"
    )
    relevance: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How relevant this evidence is to the hypothesis (considering temporal proximity, causal domain, and specificity). 0.0=completely irrelevant, 1.0=directly relevant."
    )


class HypothesisTestResult(BaseModel):
    hypothesis_id: str
    prediction_classifications: list[PredictionClassification]
    evidence_evaluations: list[EvidenceEvaluation]


class TestingResult(BaseModel):
    hypothesis_tests: list[HypothesisTestResult]


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


class BayesianResult(BaseModel):
    posteriors: list[HypothesisPosterior]
    ranking: list[str] = Field(description="Hypothesis IDs ordered by final posterior, highest first")


# ── Pass 4: Synthesis ───────────────────────────────────────────────

class HypothesisVerdict(BaseModel):
    hypothesis_id: str
    status: str = Field(description="'strongly_supported', 'supported', 'weakened', 'eliminated', or 'indeterminate'")
    key_evidence_for: list[str] = Field(description="Evidence IDs that support this hypothesis")
    key_evidence_against: list[str] = Field(description="Evidence IDs that weigh against this hypothesis")
    reasoning: str


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
    bayesian: BayesianResult
    synthesis: SynthesisResult
