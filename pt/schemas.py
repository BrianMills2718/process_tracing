"""Data contracts for all pipeline passes."""

from __future__ import annotations

from typing import ClassVar, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from pt.source_packet import SourceCoverageReport, SourcePacketSummary

DiagnosticType = Literal["hoop", "smoking_gun", "doubly_decisive", "straw_in_the_wind"]
EvidenceType = Literal["empirical", "interpretive"]
Severity = Literal["damaging", "notable", "minor"]
RefinementType = Literal["sharpen_mechanism", "add_prediction", "reframe", "merge_suggestion"]
VerdictStatus = Literal[
    "strongly_supported", "supported", "weakened", "eliminated", "indeterminate"
]
PartitionQuality = Literal["adequate", "needs_review"]
SourceGenre = Literal[
    "overview",              # secondary overview / encyclopedia / textbook
    "primary_document",      # primary historical document (letter, treaty, edict)
    "speech",                # speech or oral statement
    "legal_constitutional",  # law, constitution, decree, procedural rules, office design
    "memoir",                # memoir, diary, personal account
    "parliamentary_record",  # parliamentary or assembly debate record
    "secondary_analysis",    # academic analysis or scholarly interpretation
    "news_dispatch",         # contemporary newspaper or dispatch
    "other",
]
DateConfidence = Literal["high", "medium", "low"]
TraceProductionRelevance = Literal[
    "direct",    # evidence IS a causal trace (the mechanism acting)
    "indirect",  # evidence points to a trace (circumstantial)
    "background",  # contextual; not a direct causal trace
]
DiscriminatorStrength = Literal["decisive", "strong"]
# decisive: |log(LR_h1/LR_h2)| >= log(5) ≈ 1.61
# strong:   |log(LR_h1/LR_h2)| >= log(2) ≈ 0.69
LineageType = Literal[
    "duplicate",       # verbatim or near-verbatim copy of the same passage
    "shared_source",   # different quotes from the same document or section
    "same_event",      # different descriptions of the same historical event
    "same_mechanism",  # different expressions of the same causal process
    "other",
]


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
    source_text: str = Field(
        description=(
            "Direct quote from the input text; preserve source markers, document labels, "
            "speaker labels, and citation labels when they appear in the quoted span"
        )
    )
    evidence_type: EvidenceType = Field(
        default="empirical",
        description="'empirical' for facts/events/actions, 'interpretive' for historian arguments/scholarly claims"
    )
    approximate_date: Optional[str] = Field(
        default=None,
        description="Approximate date from the text, e.g. '1788', '1799-11', '1793-06'"
    )
    date_confidence: Optional[DateConfidence] = Field(
        default=None,
        description=(
            "'high' = explicit date in the text; 'medium' = approximate or inferred from context; "
            "'low' = only a rough period or no date at all"
        ),
    )
    source_group: Optional[str] = Field(
        default=None,
        description=(
            "Short label for the source section or document within the input text where this "
            "evidence appears, e.g. 'Background section', 'Primary source A', 'Witness testimony'. "
            "Use the same label for all evidence drawn from the same source block so that "
            "cross-source comparison is possible."
        ),
    )
    source_genre: Optional[SourceGenre] = Field(
        default=None,
        description=(
            "Genre of the source this evidence comes from. "
            "Use 'overview' for secondary/encyclopedia/textbook summaries, "
            "'primary_document' for original historical documents, "
            "'speech' for speeches or oral statements, "
            "'legal_constitutional' for laws, constitutions, decrees, "
            "'memoir' for diaries or personal accounts, "
            "'parliamentary_record' for assembly/parliamentary debates, "
            "'secondary_analysis' for academic interpretations, "
            "'news_dispatch' for contemporary press, 'other' otherwise."
        ),
    )
    trace_production_relevance: Optional[TraceProductionRelevance] = Field(
        default=None,
        description=(
            "'direct' = this evidence IS a causal trace (the mechanism visibly acting); "
            "'indirect' = this evidence points to a trace but is circumstantial; "
            "'background' = contextual, not a direct causal trace."
        ),
    )


class CausalEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str = Field(description="Nature of the causal link")


class TextHypothesis(BaseModel):
    id: str
    description: str
    source_text: str = Field(description="Quote from text where this hypothesis appears or is implied")


def _require_unique_ids(ids: list[str], label: str) -> None:
    seen: set[str] = set()
    dups: set[str] = set()
    for i in ids:
        if i in seen:
            dups.add(i)
        seen.add(i)
    if dups:
        raise ValueError(f"duplicate {label} ids: {sorted(dups)}")


class ExtractionResult(BaseModel):
    summary: str = Field(description="2-3 sentence summary of the text")
    actors: list[Actor] = []
    events: list[Event] = []
    mechanisms: list[Mechanism] = []
    evidence: list[Evidence] = []
    hypotheses_in_text: list[TextHypothesis] = []
    causal_edges: list[CausalEdge] = []

    @model_validator(mode="after")
    def _unique_evidence_ids(self) -> "ExtractionResult":
        _require_unique_ids([e.id for e in self.evidence], "evidence")
        return self


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

    @model_validator(mode="after")
    def _unique_hypothesis_ids(self) -> "HypothesisSpace":
        _require_unique_ids([h.id for h in self.hypotheses], "hypothesis")
        return self


# ── Pass 2.5: Partition Audit ────────────────────────────────────────

class RivalPairAudit(BaseModel):
    h1_id: str = Field(description="First hypothesis id in the rival pair")
    h2_id: str = Field(description="Second hypothesis id in the rival pair")
    overlap_concern: bool = Field(
        description="True if both hypotheses predict substantially the same observable evidence patterns, making them non-rival"
    )
    complementary_concern: bool = Field(
        description="True if both hypotheses could simultaneously be true (e.g. precondition + trigger + framing); they require merging, not testing"
    )
    absorptive_concern: bool = Field(
        description="True if the winning hypothesis could absorb the rival as a contributing sub-factor without the rival's mechanism being falsified"
    )
    discriminator_count: int = Field(
        ge=0,
        description="Number of observable predictions that explicitly discriminate between this pair — one predicts X, the other predicts NOT-X"
    )
    concern_detail: str = Field(
        description="Brief explanation of the most serious concern for this pair; empty string if no flags are set"
    )


class PartitionAudit(BaseModel):
    research_question_adequate: bool = Field(
        description="True if the research question targets a single outcome with genuinely rival causal explanations; False if compound, tautological, or under-specified"
    )
    rival_pairs: list[RivalPairAudit] = Field(
        description="One entry per unordered (h_i, h_j) pair — every hypothesis paired with every other"
    )
    hypotheses_flagged: list[str] = Field(
        description="Hypothesis IDs flagged as broad, tautological, complementary, or absorptive"
    )
    overall_quality: PartitionQuality = Field(
        description="'adequate' if no critical flags detected; 'needs_review' if any pair fails the discriminator gate or has overlap/complementary concerns"
    )
    cap_applied: bool = Field(
        default=False,
        description="Set by the pipeline when overall_quality is needs_review; signals that downstream support scores may be inflated by partition problems"
    )
    summary: str = Field(
        description="2-3 sentence adversarial assessment: strongest remaining concern, whether the winning hypothesis could absorb rivals, and whether the focal window is adequate"
    )


# ── Pass 3: Testing ─────────────────────────────────────────────────

class PredictionClassification(BaseModel):
    prediction_id: str
    hypothesis_id: str
    diagnostic_type: DiagnosticType = Field(
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
        allow_inf_nan=False,
        description="Relative likelihood P(E|H) of THIS evidence under THIS hypothesis, "
        "on a common positive scale shared by all hypotheses for this evidence item. "
        "Larger = this hypothesis predicts this evidence more strongly. Equal across "
        "hypotheses = uninformative. Must be a finite positive number.",
    )
    diagnostic_type: DiagnosticType = Field(
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


class EvidenceCluster(BaseModel):
    """A group of evidence items that are NOT conditionally independent.

    Items sharing the same source/document lineage, the same originating event, or
    the same underlying fact carry overlapping information; multiplying their
    likelihoods would double-count. The Bayesian update collapses each cluster to a
    single effective observation (log-average of member vectors).
    """
    evidence_ids: list[str] = Field(
        description="Two or more evidence ids that share a source, event, mechanism, or "
        "underlying sub-narrative and therefore carry overlapping (non-independent) information."
    )
    reason: str = Field(description="Why these items are dependent (shared source/event/mechanism).")
    lineage_type: Optional[LineageType] = Field(
        default=None,
        description=(
            "Structured classification of the dependence cause. "
            "'duplicate' = verbatim or near-verbatim copy; "
            "'shared_source' = different quotes from the same document/section; "
            "'same_event' = different descriptions of the same historical event; "
            "'same_mechanism' = different expressions of the same causal process; "
            "'other' = none of the above. "
            "Set this so the audit can verify that each cluster has a visible lineage explanation."
        ),
    )
    dependence_strength: float = Field(
        default=1.0, ge=0.0, le=1.0, allow_inf_nan=False,
        description="How redundant the members are (0=independent, 1=fully redundant). "
        "1.0 for duplicates/same-source copies; ~0.5–0.8 for items about the same event or "
        "mechanism that still add some independent signal. Sets the cluster's effective "
        "observation count k_eff = 1 + (k-1)(1-dependence_strength).",
    )


class TestingResult(BaseModel):
    __test__: ClassVar[bool] = False

    evidence_likelihoods: list[EvidenceLikelihood] = Field(
        description="Per-evidence likelihood vectors across all hypotheses.",
    )
    dependence_clusters: list[EvidenceCluster] = Field(
        default_factory=list,
        description="Groups of conditionally-dependent evidence items. Each cluster is "
        "collapsed to one effective observation in the update to avoid double-counting. "
        "Items not in any cluster are treated as independent.",
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


# ── Pass 3.6: Diagnostic Test Matrix (deterministic derivation) ─────

class RivalDiscriminator(BaseModel):
    """One evidence item that discriminates between a specific rival pair."""
    evidence_id: str = Field(description="ID of the discriminating evidence item")
    log_lr_h1_over_h2: float = Field(
        description="log(LR_h1 / LR_h2) using effective LRs from the testing matrix. "
        "Positive means the evidence favors h1 over h2; negative favors h2."
    )
    favors: Literal["h1", "h2"] = Field(
        description="Which hypothesis this evidence favors: 'h1' or 'h2'."
    )
    strength: DiscriminatorStrength = Field(
        description="'decisive' if |log_lr| >= log(5) ≈ 1.61; 'strong' if >= log(2) ≈ 0.69."
    )
    diagnostic_type_h1: Optional[DiagnosticType] = Field(
        default=None,
        description="Van Evera type assigned by the LLM for this evidence under h1.",
    )
    diagnostic_type_h2: Optional[DiagnosticType] = Field(
        default=None,
        description="Van Evera type assigned by the LLM for this evidence under h2.",
    )


class RivalPairDiagnostic(BaseModel):
    """Discriminator summary for one rival hypothesis pair."""
    h1_id: str
    h2_id: str
    discriminators: list[RivalDiscriminator] = Field(
        description="Evidence items that distinguish this pair (|log LR_h1/LR_h2| >= log(2))."
    )
    discriminator_count: int = Field(
        ge=0,
        description="Number of discriminating evidence items for this pair.",
    )
    grade_capped: bool = Field(
        description="True if discriminator_count == 0; an A-level claim requires at least one "
        "source-grounded discriminator per rival pair."
    )


class DiagnosticMatrix(BaseModel):
    """Derived artifact: which evidence items discriminate which rival pairs.

    Computed deterministically from the testing result's likelihood vectors.
    No LLM call required. A grade cap is applied for every pair with zero discriminators.
    """
    rival_pair_diagnostics: list[RivalPairDiagnostic] = Field(
        description="One entry per rival hypothesis pair."
    )
    pairs_without_discriminators: list[list[str]] = Field(
        default_factory=list,
        description="Pairs of [h1_id, h2_id] with zero discriminating evidence items.",
    )
    grade_cap_applied: bool = Field(
        description="True if any rival pair lacks discriminators."
    )


# ── Pass 3b: Absence-of-Evidence ───────────────────────────────────

class AbsenceEvaluation(BaseModel):
    hypothesis_id: str
    prediction_id: str
    missing_evidence: str = Field(description="What predicted evidence is absent from the text")
    reasoning: str = Field(description="Why absence is informative given the text's scope")
    severity: Severity = Field(description="'damaging', 'notable', or 'minor'")
    would_be_extractable: bool = Field(
        description="Would this evidence appear in a text of this scope if it existed?"
    )
    expected_source_genre: Optional[SourceGenre] = Field(
        default=None,
        description=(
            "The genre of source that would typically carry this missing trace — "
            "e.g. 'primary_document' for correspondence, 'parliamentary_record' for debate logs, "
            "'secondary_analysis' for academic interpretation. Populate regardless of "
            "would_be_extractable: even when the current text cannot carry the evidence, "
            "naming the genre tells the researcher what to acquire next."
        ),
    )
    expected_source_location: Optional[str] = Field(
        default=None,
        description=(
            "Specific type of source where this evidence would appear — "
            "e.g. 'police surveillance reports from the Directory period', "
            "'minutes of the Conseil des Cinq-Cents', 'private correspondence of Napoleon'. "
            "Be concrete enough to drive an acquisition decision. Populate whenever "
            "you can name a plausible archive, collection, or document type."
        ),
    )


class AbsenceResult(BaseModel):
    evaluations: list[AbsenceEvaluation] = []


# ── Pass 5: Refinement (Second Reading) ────────────────────────────

class NewEvidence(BaseModel):
    id: str = Field(description="Must use 'evi_ref_' prefix to distinguish from original extraction")
    description: str
    source_text: str = Field(description="Direct quote from input text")
    evidence_type: EvidenceType = Field(default="empirical", description="'empirical' or 'interpretive'")
    approximate_date: Optional[str] = None
    rationale: str = Field(description="Why missed initially, why it matters now")


class ReinterpretedEvidence(BaseModel):
    evidence_id: str
    original_type: EvidenceType
    new_type: EvidenceType
    reinterpretation: str
    updated_description: Optional[str] = None


class NewCausalEdge(BaseModel):
    source_id: str = Field(
        description=(
            "Existing event/actor/mechanism/evidence id, or a new evi_ref_* "
            "evidence id created in this same refinement. Do not use hypothesis ids."
        )
    )
    target_id: str = Field(
        description=(
            "Existing event/actor/mechanism/evidence id, or a new evi_ref_* "
            "evidence id created in this same refinement. Do not use hypothesis ids."
        )
    )
    relationship: str = Field(description="Substantive causal/process relationship, not evidentiary support.")
    source_text_support: str = Field(description="Quote from input text")


class SpuriousExtraction(BaseModel):
    item_id: str = Field(description="Evidence ID or 'source_id->target_id' for edges")
    item_type: Literal["evidence", "causal_edge"] = Field(
        description="'evidence' or 'causal_edge'"
    )
    reason: str


class HypothesisRefinement(BaseModel):
    hypothesis_id: str = Field(
        description=(
            "Existing hypothesis id for state-changing refinements. For "
            "merge_suggestion, may be an existing id or a descriptive advisory "
            "id naming the proposed merge, e.g. 'h1_h4_merge_suggestion'."
        )
    )
    refinement_type: RefinementType = Field(
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
    status: VerdictStatus = Field(description="'strongly_supported', 'supported', 'weakened', 'eliminated', or 'indeterminate'")
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


# ── Structural Critic (Pass 3.7) ────────────────────────────────────

CriticFindingType = Literal[
    "confound",          # a third variable explains both cause and effect
    "missing_pathway",   # a causal mechanism missing from the extraction
    "void_link",         # a causal edge with no evidentiary support in the matrix
    "too_strong_claim",  # a diagnostic type not justified by the justification text
    "confirmed_link",    # a well-supported causal link worth explicitly affirming
]


class CriticFinding(BaseModel):
    finding_type: CriticFindingType
    target: str = Field(
        description="ID of the evidence item, hypothesis, or causal edge (format: 'src_id->tgt_id') "
        "this finding refers to. Use an exact ID from the input data."
    )
    target_type: Literal["evidence", "hypothesis", "causal_edge"] = Field(
        description="'evidence' for an evidence_id, 'hypothesis' for a hypothesis_id, "
        "'causal_edge' for a 'src_id->tgt_id' edge string."
    )
    severity: Literal["high", "medium", "low"] = Field(
        description="'high' if this likely distorts posterior rankings; 'medium' if worth correcting; "
        "'low' if informational only."
    )
    reasoning: str = Field(description="Why this is a structural problem and what supports the concern.")
    recommendation: str = Field(
        description="Structural action only: collect specific evidence, add/remove a graph edge, "
        "merge hypotheses, or downgrade a diagnostic label. Do NOT suggest specific likelihood values."
    )

    @model_validator(mode="after")
    def _validate_target_consistency(self) -> "CriticFinding":
        if self.target_type == "hypothesis" and "->" in self.target:
            raise ValueError(
                f"target '{self.target}' contains '->' but target_type='hypothesis'. "
                "Use target_type='causal_edge' for edge targets, or split into separate "
                "hypothesis findings."
            )
        if self.target_type == "causal_edge":
            count = self.target.count("->")
            if count == 0:
                raise ValueError(
                    f"target '{self.target}' has no '->' but target_type='causal_edge'. "
                    "Causal edge targets must use format 'source_id->target_id'."
                )
            if count > 1:
                raise ValueError(
                    f"target '{self.target}' contains {count} '->' tokens but causal_edge "
                    "targets must reference exactly one edge: 'source_id->target_id'. "
                    "Multi-hop chains are not valid — create separate findings for each edge."
                )
        return self


class CriticResult(BaseModel):
    findings: list[CriticFinding] = []
    summary: str = Field(description="2-3 sentence overall structural assessment.")
    re_elicitation_needed: bool = Field(
        default=False,
        description="Computed post-parse: True if any finding has severity='high'. "
        "Do not set this field — it is derived automatically.",
    )

    @model_validator(mode="after")
    def _compute_re_elicitation(self) -> "CriticResult":
        """re_elicitation_needed is deterministic: any high-severity finding triggers a re-run."""
        self.re_elicitation_needed = any(f.severity == "high" for f in self.findings)
        return self


class CriticDelta(BaseModel):
    """Per-hypothesis posterior change between base and critic runs."""
    hypothesis_id: str
    posterior_base: float
    posterior_critic: float
    delta: float = Field(description="posterior_critic - posterior_base")
    top_driver_change: list[str] = Field(
        default_factory=list,
        description="Evidence IDs added or removed from top drivers between runs.",
    )
    critic_findings_count: int = Field(
        default=0,
        description="Number of critic findings targeting this hypothesis or its evidence.",
    )


# ── Combined Result ─────────────────────────────────────────────────

class ProcessTracingResult(BaseModel):
    source_text_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 of the exact input text used for this analysis",
    )
    extraction: ExtractionResult
    hypothesis_space: HypothesisSpace
    testing: TestingResult
    absence: AbsenceResult
    bayesian: BayesianResult
    synthesis: SynthesisResult
    source_packet: Optional[SourcePacketSummary] = Field(
        default=None,
        description="Source-packet metadata governing source scope and observability assumptions.",
    )
    source_coverage: Optional[SourceCoverageReport] = Field(
        default=None,
        description="Deterministic packet-source coverage against input text and extracted evidence.",
    )
    partition_audit: Optional[PartitionAudit] = Field(
        default=None,
        description="Pass 2.5: Hypothesis partition audit — flags overlap, complementarity, and absorptive risks before testing.",
    )
    diagnostic_matrix: Optional[DiagnosticMatrix] = Field(
        default=None,
        description="Pass 3.6: Diagnostic test matrix — which evidence items discriminate each rival pair, derived deterministically from likelihood vectors.",
    )
    refinement: Optional[RefinementResult] = None
    is_refined: bool = False
    critic: Optional[CriticResult] = Field(
        default=None,
        description="Pass 3.7: Structural critic findings — confounds, missing pathways, void links, "
        "too-strong claims, and confirmed links. Populated only when --critic flag is active.",
    )
