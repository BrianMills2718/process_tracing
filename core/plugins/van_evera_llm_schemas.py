"""
Structured Output Schemas for Van Evera Process Tracing
Pydantic models for precise LLM evaluation and academic reasoning
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any, Tuple
from enum import Enum


class DiagnosticTestType(str, Enum):
    """Van Evera diagnostic test types"""
    HOOP = "hoop"
    SMOKING_GUN = "smoking_gun"
    DOUBLY_DECISIVE = "doubly_decisive"
    STRAW_IN_WIND = "straw_in_wind"


class TestResult(str, Enum):
    """Test result types"""
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceQuality(str, Enum):
    """Evidence quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceRelationshipClassification(BaseModel):
    """
    LLM-based semantic classification of evidence-hypothesis relationships.
    Replaces rule-based keyword matching with semantic understanding.
    """
    relationship_type: Literal["supporting", "refuting", "irrelevant"] = Field(
        description="Semantic relationship between evidence and hypothesis"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence in the classification (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Detailed semantic reasoning for classification decision"
    )
    probative_value: float = Field(
        ge=0.0, le=1.0, 
        description="Evidence strength assessment for this hypothesis"
    )
    contradiction_indicators: int = Field(
        ge=0, 
        description="Number of semantic contradictions identified between evidence and hypothesis"
    )
    semantic_analysis: str = Field(
        description="Analysis of evidence-hypothesis semantic relationship and context"
    )


class VanEveraPredictionEvaluation(BaseModel):
    """Structured output for Van Evera prediction evaluation"""
    test_result: TestResult = Field(description="Whether the prediction passes, fails, or is inconclusive")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the evaluation (0.0-1.0)")
    
    # Academic reasoning
    diagnostic_reasoning: str = Field(description="Detailed Van Evera diagnostic test reasoning")
    evidence_assessment: str = Field(description="Assessment of evidence quality and relevance")
    theoretical_mechanism_evaluation: str = Field(description="Evaluation of the proposed theoretical mechanism")
    
    # Van Evera logic
    necessity_analysis: Optional[str] = Field(None, description="For hoop tests: necessity condition analysis")
    sufficiency_analysis: Optional[str] = Field(None, description="For smoking gun tests: sufficiency condition analysis")
    elimination_implications: List[str] = Field(description="Which hypotheses this result eliminates or supports")
    
    # Evidence details
    evidence_quality: EvidenceQuality = Field(description="Overall quality of supporting evidence")
    evidence_coverage: float = Field(ge=0.0, le=1.0, description="How well evidence covers prediction requirements")
    indicator_matches: int = Field(ge=0, description="Number of qualitative indicators found in evidence")
    
    # Academic quality
    publication_quality_assessment: str = Field(description="Assessment of academic rigor and publication readiness")
    methodological_soundness: float = Field(ge=0.0, le=1.0, description="Van Evera methodology compliance score")


class BayesianParameterEstimation(BaseModel):
    """LLM-estimated parameters for Bayesian analysis"""
    prior_probability: float = Field(ge=0.0, le=1.0, description="Prior probability of hypothesis")
    likelihood_given_hypothesis: float = Field(ge=0.0, le=1.0, description="P(Evidence|Hypothesis)")
    likelihood_given_not_hypothesis: float = Field(ge=0.0, le=1.0, description="P(Evidence|¬Hypothesis)")
    
    # Reasoning
    prior_justification: str = Field(description="Justification for prior probability estimate")
    likelihood_reasoning: str = Field(description="Reasoning for likelihood estimates")
    
    # Uncertainty
    confidence_in_estimates: float = Field(ge=0.0, le=1.0, description="Confidence in parameter estimates")
    uncertainty_sources: List[str] = Field(description="Sources of uncertainty in estimates")


class ConfidenceThresholdAssessment(BaseModel):
    """
    LLM-based confidence threshold assessment replacing hardcoded values.
    Provides dynamic, context-aware confidence thresholds based on evidence quality.
    """
    # Dynamic confidence level thresholds
    very_high_threshold: float = Field(ge=0.7, le=1.0, description="Minimum threshold for very high confidence")
    high_threshold: float = Field(ge=0.6, le=0.9, description="Minimum threshold for high confidence")
    moderate_threshold: float = Field(ge=0.4, le=0.7, description="Minimum threshold for moderate confidence")
    low_threshold: float = Field(ge=0.2, le=0.5, description="Minimum threshold for low confidence")
    
    # Causal mechanism assessment
    mechanism_completeness: float = Field(ge=0.0, le=1.0, description="Completeness of causal mechanism explanation")
    temporal_consistency: float = Field(ge=0.0, le=1.0, description="Temporal ordering consistency score")
    logical_coherence: float = Field(ge=0.0, le=1.0, description="Base logical coherence score")
    
    # Evidence quality factors
    independence_score: float = Field(ge=0.0, le=1.0, description="Evidence source independence assessment")
    reliability_threshold: float = Field(ge=0.0, le=1.0, description="Minimum acceptable evidence reliability")
    strength_preference: float = Field(ge=0.0, le=1.0, description="Preferred evidence strength level")
    
    # Uncertainty factors
    posterior_uncertainty: float = Field(ge=0.0, le=0.5, description="Uncertainty in posterior estimation")
    sample_size_factor: float = Field(ge=1.0, le=10.0, description="Sample size adequacy factor")
    
    # Academic justification
    threshold_reasoning: str = Field(description="Reasoning for confidence threshold selections")
    quality_assessment: str = Field(description="Overall evidence quality assessment")
    domain_considerations: str = Field(description="Domain-specific confidence considerations")
    methodological_notes: str = Field(description="Methodological factors affecting confidence")


class CausalMechanismAssessment(BaseModel):
    """
    LLM assessment of causal mechanism quality and completeness.
    Replaces hardcoded mechanism scores with semantic understanding.
    """
    mechanism_clarity: float = Field(ge=0.0, le=1.0, description="Clarity of causal mechanism description")
    mechanism_completeness: float = Field(ge=0.0, le=1.0, description="Completeness of causal chain")
    temporal_ordering: float = Field(ge=0.0, le=1.0, description="Temporal consistency of causal sequence")
    
    # Causal chain analysis
    causal_steps: List[str] = Field(description="Identified steps in causal chain")
    missing_links: List[str] = Field(description="Potential missing causal links")
    strength_assessment: str = Field(description="Assessment of causal relationship strength")
    
    # Academic quality
    theoretical_grounding: float = Field(ge=0.0, le=1.0, description="Theoretical foundation strength")
    empirical_support: float = Field(ge=0.0, le=1.0, description="Empirical evidence support level")
    alternative_explanations: List[str] = Field(description="Alternative causal explanations to consider")
    
    # Confidence factors
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in causal mechanism")
    confidence_reasoning: str = Field(description="Reasoning for confidence assessment")


class CausalRelationshipAnalysis(BaseModel):
    """LLM analysis of causal relationships"""
    causal_strength: float = Field(ge=0.0, le=1.0, description="Estimated strength of causal relationship")
    causal_mechanism: str = Field(description="Proposed causal mechanism")
    
    # Causal criteria
    temporal_precedence: bool = Field(description="Cause precedes effect temporally")
    covariation: float = Field(ge=0.0, le=1.0, description="Degree of covariation between cause and effect")
    alternative_explanations_ruled_out: float = Field(ge=0.0, le=1.0, description="Extent alternatives are ruled out")
    
    # Confounders and mediators
    potential_confounders: List[str] = Field(description="Potential confounding variables")
    potential_mediators: List[str] = Field(description="Potential mediating variables")


class ConfidenceFormulaWeights(BaseModel):
    """
    LLM determines appropriate weights for confidence calculation.
    Replaces hardcoded formula weights with dynamic, context-aware values.
    """
    quality_weight: float = Field(ge=0.0, le=1.0, description="Weight for evidence quality component")
    quantity_weight: float = Field(ge=0.0, le=1.0, description="Weight for evidence quantity component")
    diversity_weight: float = Field(ge=0.0, le=1.0, description="Weight for evidence diversity component")
    balance_weight: float = Field(ge=0.0, le=1.0, description="Weight for evidence balance component")
    
    # Justification
    reasoning: str = Field(description="Justification for weight selection based on context")
    total_weight: float = Field(ge=0.9, le=1.1, description="Sum of weights (should be ~1.0)")
    
    def model_post_init(self, __context):
        """Ensure weights sum to approximately 1.0"""
        self.total_weight = (self.quality_weight + self.quantity_weight + 
                           self.diversity_weight + self.balance_weight)


class SemanticRelevanceAssessment(BaseModel):
    """
    Replace ALL word overlap with semantic assessment.
    No counting, no ratios, pure semantic understanding.
    """
    is_relevant: bool = Field(description="Binary relevance decision based on semantic understanding")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Semantic relevance strength")
    semantic_relationship: str = Field(description="Type of semantic relationship identified")
    reasoning: str = Field(description="Detailed reasoning for relevance assessment")
    
    # Additional semantic factors
    conceptual_alignment: float = Field(ge=0.0, le=1.0, description="Degree of conceptual alignment")
    contextual_fit: float = Field(ge=0.0, le=1.0, description="How well evidence fits hypothesis context")
    semantic_distance: float = Field(ge=0.0, le=1.0, description="Semantic distance between concepts")
    
    # Causal reasoning
    causal_reasoning: str = Field(description="Detailed causal reasoning and evidence")
    uncertainty_assessment: str = Field(description="Assessment of causal uncertainty")


class ProcessTracingConclusion(BaseModel):
    """Academic-quality process tracing conclusion"""
    hypothesis_status: Literal["SUPPORTED", "ELIMINATED", "WEAKENED", "INCONCLUSIVE"] = Field(
        description="Overall status of hypothesis after testing"
    )
    confidence_level: float = Field(ge=0.0, le=1.0, description="Overall confidence in conclusion")
    
    # Academic summary
    academic_summary: str = Field(description="Publication-quality summary of findings")
    methodology_assessment: str = Field(description="Assessment of methodology used")
    
    # Evidence synthesis
    supporting_evidence_strength: float = Field(ge=0.0, le=1.0, description="Strength of supporting evidence")
    contradicting_evidence_assessment: str = Field(description="Assessment of contradicting evidence")
    
    # Publication readiness
    publication_quality_score: float = Field(ge=0.0, le=1.0, description="Publication quality assessment")
    recommendations_for_improvement: List[str] = Field(description="Recommendations to improve analysis")


class ContentBasedClassification(BaseModel):
    """Structured output for content-based diagnostic classification"""
    recommended_diagnostic_type: DiagnosticTestType = Field(description="Recommended Van Evera diagnostic test type")
    classification_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in classification")
    
    # Classification reasoning
    content_analysis: str = Field(description="Analysis of evidence and hypothesis content")
    theoretical_fit: str = Field(description="How well the diagnostic type fits the theoretical relationship")
    
    # Van Evera logic
    necessity_assessment: Optional[str] = Field(None, description="Assessment of necessity relationship")
    sufficiency_assessment: Optional[str] = Field(None, description="Assessment of sufficiency relationship")
    
    # Alternative considerations
    alternative_classifications: List[Dict[str, Any]] = Field(
        description="Alternative diagnostic types with reasoning"
    )
    
    # Quality metrics
    theoretical_sophistication: float = Field(ge=0.0, le=1.0, description="Theoretical sophistication of classification")
    methodological_rigor: float = Field(ge=0.0, le=1.0, description="Methodological rigor of classification")


class HypothesisDomainClassification(BaseModel):
    """
    LLM-based semantic classification of hypothesis domains.
    Replaces keyword matching with universal domain analysis.
    """
    primary_domain: Literal["political", "economic", "ideological", "military", "social", "cultural", "religious", "technological"] = Field(
        description="Primary domain classification based on semantic content"
    )
    secondary_domains: List[str] = Field(
        default_factory=list,
        description="Additional relevant domains that apply to this hypothesis"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the domain classification (0.0-1.0)"
    )
    reasoning: str = Field(
        description="Semantic reasoning for domain classification decision"
    )
    generalizability: str = Field(
        description="How this domain classification applies beyond specific historical contexts"
    )
    domain_indicators: List[str] = Field(
        description="Semantic indicators that support the domain classification"
    )
    cross_domain_relationships: List[str] = Field(
        default_factory=list,
        description="Relationships between domains if hypothesis spans multiple areas"
    )


class ProbativeValueAssessment(BaseModel):
    """
    LLM-generated assessment of evidence probative value.
    Replaces hardcoded probative value assignments with semantic analysis.
    """
    probative_value: float = Field(
        ge=0.0, le=1.0,
        description="Evidence strength assessment based on semantic analysis"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the probative value assessment"
    )
    reasoning: str = Field(
        description="Academic justification for probative value assignment"
    )
    evidence_quality_factors: List[str] = Field(
        description="Factors contributing to evidence strength assessment"
    )
    reliability_assessment: str = Field(
        description="Assessment of evidence reliability and credibility"
    )
    van_evera_implications: str = Field(
        description="Implications for Van Evera diagnostic testing methodology"
    )
    strength_indicators: List[str] = Field(
        description="Semantic indicators that support the strength assessment"
    )
    weakness_factors: List[str] = Field(
        default_factory=list,
        description="Factors that may reduce evidence probative value"
    )
    contextual_relevance: float = Field(
        ge=0.0, le=1.0,
        description="How relevant this evidence is to the specific hypothesis context"
    )


class AlternativeHypothesisGeneration(BaseModel):
    """
    LLM-generated alternative hypotheses based on semantic understanding.
    Replaces keyword dictionary approaches with contextual generation.
    """
    alternative_hypotheses: List[Dict[str, Any]] = Field(
        description="Generated alternative hypotheses with semantic reasoning"
    )
    generation_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the quality of generated alternatives"
    )
    semantic_analysis: str = Field(
        description="Analysis of semantic relationships in the original hypothesis"
    )
    domain_coverage: List[str] = Field(
        description="Domains covered by the alternative hypotheses"
    )
    theoretical_sophistication: str = Field(
        description="Assessment of theoretical sophistication of alternatives"
    )
    competing_mechanisms: List[str] = Field(
        description="Alternative causal mechanisms identified through semantic analysis"
    )
    universal_applicability: str = Field(
        description="How these alternatives apply across different historical periods and contexts"
    )


class TestGenerationSpecification(BaseModel):
    """
    LLM-generated Van Evera test specifications based on semantic understanding.
    Replaces keyword-based test creation with context-appropriate generation.
    """
    test_predictions: List[Dict[str, Any]] = Field(
        description="Generated test predictions with Van Evera diagnostic types"
    )
    generation_reasoning: str = Field(
        description="Reasoning for test generation and diagnostic type selection"
    )
    semantic_analysis: str = Field(
        description="Semantic analysis of hypothesis-evidence relationships"
    )
    theoretical_grounding: str = Field(
        description="Theoretical foundation for the generated tests"
    )
    diagnostic_logic: str = Field(
        description="Van Evera diagnostic logic applied to test generation"
    )
    evidence_requirements: List[str] = Field(
        description="Evidence requirements derived through semantic analysis"
    )
    universal_validity: str = Field(
        description="How these tests apply across different contexts and periods"
    )
    methodological_rigor: float = Field(
        ge=0.0, le=1.0,
        description="Assessment of methodological rigor in test generation"
    )


class ComprehensiveEvidenceAnalysis(BaseModel):
    """
    Single schema capturing all semantic features of evidence in one analysis.
    Replaces multiple separate LLM calls with one coherent comprehensive analysis.
    """
    
    # Domain Analysis
    primary_domain: Literal["political", "economic", "ideological", "military", 
                           "social", "cultural", "religious", "technological"] = Field(
        description="Primary domain classification based on semantic content"
    )
    secondary_domains: List[str] = Field(
        default_factory=list,
        description="Additional relevant domains identified"
    )
    domain_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in domain classification"
    )
    domain_reasoning: str = Field(
        description="Semantic reasoning for domain classification"
    )
    
    # Probative Assessment
    probative_value: float = Field(
        ge=0.0, le=1.0,
        description="Evidence strength assessment (0.0-1.0)"
    )
    probative_factors: List[str] = Field(
        description="Factors contributing to probative value"
    )
    evidence_quality: Literal["high", "medium", "low"] = Field(
        description="Overall evidence quality assessment"
    )
    reliability_score: float = Field(
        ge=0.0, le=1.0,
        description="Evidence reliability assessment"
    )
    
    # Hypothesis Relationship
    relationship_type: Literal["supports", "contradicts", "neutral", "ambiguous"] = Field(
        description="Relationship between evidence and hypothesis"
    )
    relationship_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in relationship assessment"
    )
    relationship_reasoning: str = Field(
        description="Detailed reasoning for relationship determination"
    )
    van_evera_diagnostic: Literal["hoop", "smoking_gun", "doubly_decisive", "straw_in_wind"] = Field(
        description="Van Evera diagnostic test classification"
    )
    
    # Semantic Features
    causal_mechanisms: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Identified causal mechanisms: type -> description"
    )
    temporal_markers: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Temporal references: marker -> context"
    )
    actor_relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Actor relationships: actor -> role/relationship"
    )
    
    # Meta-Analysis
    key_concepts: List[str] = Field(
        description="Key conceptual elements identified"
    )
    contextual_factors: List[str] = Field(
        description="Contextual factors affecting interpretation"
    )
    alternative_interpretations: List[str] = Field(
        default_factory=list,
        description="Alternative ways to interpret this evidence"
    )
    confidence_overall: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in this comprehensive analysis"
    )


class MultiFeatureExtraction(BaseModel):
    """
    Extract all semantic features in one pass for compound analysis.
    Captures relationships between different feature types.
    """
    
    # Causal Analysis
    mechanisms: List[Dict[str, str]] = Field(
        description="Causal mechanisms: type -> description"
    )
    causal_chains: List[List[str]] = Field(
        default_factory=list,
        description="Sequences of causal events"
    )
    
    # Actor Network
    primary_actors: List[str] = Field(
        description="Primary actors identified in text"
    )
    actor_relationships: List[Dict[str, str]] = Field(
        description="Actor -> role/relationship mapping"
    )
    
    # Temporal Structure
    temporal_sequence: List[Dict[str, str]] = Field(
        description="Time marker -> event mapping"
    )
    duration_estimates: Dict[str, str] = Field(
        default_factory=dict,
        description="Event -> estimated duration"
    )
    
    # Conceptual Analysis
    key_concepts: List[str] = Field(
        description="Key conceptual elements"
    )
    domain_indicators: List[str] = Field(
        description="Indicators of domain classification"
    )
    theoretical_frameworks: List[str] = Field(
        default_factory=list,
        description="Theoretical frameworks referenced"
    )
    
    # Contextual Factors
    geographic_context: List[str] = Field(
        default_factory=list,
        description="Geographic/spatial context"
    )
    institutional_context: List[str] = Field(
        default_factory=list,
        description="Institutional context factors"
    )
    cultural_context: List[str] = Field(
        default_factory=list,
        description="Cultural context factors"
    )
    
    # Feature Relationships
    actor_mechanism_links: List[Dict[str, str]] = Field(
        default_factory=list,
        description="How actors relate to mechanisms"
    )
    temporal_causal_links: List[Dict[str, str]] = Field(
        default_factory=list,
        description="How timing relates to causation"
    )


class HypothesisEvaluationResult(BaseModel):
    """Result of evaluating evidence against a single hypothesis"""
    hypothesis_id: str = Field(description="Unique identifier for the hypothesis")
    relationship_type: Literal["supports", "contradicts", "neutral", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the relationship")
    van_evera_diagnostic: Literal["hoop", "smoking_gun", "doubly_decisive", "straw_in_wind"]
    reasoning: str = Field(description="Explanation of the relationship")
    
    # Inter-hypothesis insights
    strengthens_hypotheses: List[str] = Field(
        default_factory=list,
        description="Other hypothesis IDs that this relationship strengthens"
    )
    weakens_hypotheses: List[str] = Field(
        default_factory=list,
        description="Other hypothesis IDs that this relationship weakens"
    )


class BatchedHypothesisEvaluation(BaseModel):
    """
    Batched evaluation of evidence against multiple hypotheses.
    Enables understanding inter-hypothesis relationships and provides
    more coherent analysis than separate evaluations.
    """
    evidence_id: str = Field(description="Identifier for the evidence being evaluated")
    evidence_summary: str = Field(description="Brief summary of the evidence")
    
    # Individual evaluations
    evaluations: List[HypothesisEvaluationResult] = Field(
        description="Evaluation results for each hypothesis"
    )
    
    # Cross-hypothesis insights
    primary_hypothesis_supported: Optional[str] = Field(
        default=None,
        description="ID of hypothesis most strongly supported"
    )
    conflicting_hypotheses: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of hypotheses that conflict based on this evidence"
    )
    complementary_hypotheses: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of hypotheses that reinforce each other"
    )
    
    # Overall assessment
    evidence_significance: Literal["critical", "important", "moderate", "minor"] = Field(
        description="Overall significance of this evidence"
    )
    analytical_notes: str = Field(
        description="Overall analytical insights from batch evaluation"
    )
    confidence_overall: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in the evaluations"
    )

class SemanticThresholdAssessment(BaseModel):
    """
    LLM-based semantic threshold assessment for evidence relevance.
    Provides dynamic thresholds based on context and evidence type.
    """
    threshold: float = Field(
        ge=0.0, le=1.0,
        description="Semantic relevance threshold for evidence-prediction relationships"
    )
    context_factor: float = Field(
        ge=0.8, le=1.2,
        description="Context-specific adjustment factor for threshold"
    )
    evidence_type_weight: float = Field(
        ge=0.8, le=1.2,
        description="Evidence type-specific weight adjustment"
    )
    reasoning: str = Field(
        description="LLM reasoning for threshold determination"
    )