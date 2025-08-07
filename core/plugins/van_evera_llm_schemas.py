"""
Structured Output Schemas for Van Evera Process Tracing
Pydantic models for precise LLM evaluation and academic reasoning
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any
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