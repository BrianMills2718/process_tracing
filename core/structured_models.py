"""
Pydantic models for structured output from Gemini API calls.
Implements proper structured output as recommended in Gemini documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from enum import Enum


class VanEveraEvidenceType(str, Enum):
    """Van Evera's four diagnostic tests for evidence classification."""
    HOOP = "hoop"
    SMOKING_GUN = "smoking_gun" 
    STRAW_IN_THE_WIND = "straw_in_the_wind"
    DOUBLY_DECISIVE = "doubly_decisive"


class EvidenceAssessment(BaseModel):
    """
    Structured response for evidence assessment with Van Evera diagnostic tests.
    Used in core/enhance_evidence.py for LLM-enhanced evidence analysis.
    """
    evidence_id: str = Field(description="Unique identifier for the evidence")
    refined_evidence_type: VanEveraEvidenceType = Field(
        description="Van Evera diagnostic test classification"
    )
    reasoning_for_type: str = Field(
        description="Explanation for why this evidence type was chosen"
    )
    likelihood_P_E_given_H: str = Field(
        description="Likelihood of observing evidence if hypothesis is true (e.g., 'High (0.8)')"
    )
    likelihood_P_E_given_NotH: str = Field(
        description="Likelihood of observing evidence if hypothesis is false (e.g., 'Low (0.2)')"
    )
    justification_for_likelihoods: str = Field(
        description="Detailed justification for the likelihood assessments"
    )
# Issue #14 Fix: Standardize probative value scale to 0.0-1.0 throughout codebase
    suggested_numerical_probative_value: float = Field(
        description="Numerical probative value for the evidence (0.0-1.0 range)",
        ge=0.0,
        le=1.0
    )


class MechanismAssessment(BaseModel):
    """
    Structured response for causal mechanism assessment.
    Used in core/enhance_mechanisms.py for mechanism analysis.
    """
    mechanism_id: str = Field(description="Unique identifier for the mechanism")
    completeness_score: float = Field(
        description="How complete this mechanism is (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    plausibility_score: float = Field(
        description="How plausible this mechanism is (0.0-1.0)", 
        ge=0.0,
        le=1.0
    )
    evidence_support_level: Literal["strong", "moderate", "weak", "none"] = Field(
        description="Level of evidence supporting this mechanism"
    )
    missing_elements: List[str] = Field(
        description="List of missing elements that would strengthen the mechanism",
        default_factory=list
    )
    improvement_suggestions: List[str] = Field(
        description="Specific suggestions for improving mechanism completeness",
        default_factory=list
    )
    detailed_reasoning: str = Field(
        description="Detailed reasoning for the assessment"
    )


class NarrativeSummary(BaseModel):
    """
    Structured response for analytical narrative summaries.
    Used in core/llm_reporting_utils.py for generating concise summaries.
    """
    summary_text: str = Field(
        description="Concise analytical narrative (2-5 sentences, approximately 50-1000 characters)"
    )
    key_findings: List[str] = Field(
        description="3-5 key findings extracted from the analysis"
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the analysis conclusions"
    )
    supporting_evidence_count: int = Field(
        description="Number of pieces of supporting evidence",
        ge=0
    )
    refuting_evidence_count: int = Field(
        description="Number of pieces of refuting evidence", 
        ge=0
    )


class NodeExtraction(BaseModel):
    """Individual node extracted from text for graph construction."""
    id: str = Field(description="Unique identifier for the node")
    type: Literal["Event", "Causal_Mechanism", "Hypothesis", "Evidence", "Condition", "Actor", "Alternative"] = Field(
        description="Node type according to process tracing ontology"
    )
    subtype: Optional[str] = Field(
        description="Subtype specification (e.g., 'triggering', 'outcome')",
        default=None
    )
    description: str = Field(
        description="Clear description of the node",
        min_length=10
    )
    source_text_quote: Optional[str] = Field(
        description="Relevant quote from source text", 
        default=""
    )
    confidence: float = Field(
        description="Confidence in extraction accuracy (0.0-1.0)",
        ge=0.0,
        le=1.0,
        default=0.8
    )


class EdgeExtraction(BaseModel):
    """Individual edge extracted from text for graph construction."""
    source_id: str = Field(description="ID of source node")
    target_id: str = Field(description="ID of target node")
    edge_type: Literal["causes", "leads_to", "precedes", "triggers", "contributes_to", 
                      "enables", "influences", "facilitates", "supports", "refutes"] = Field(
        description="Type of relationship between nodes"
    )
    strength: float = Field(
        description="Strength of the relationship (0.0-1.0)",
        ge=0.0,
        le=1.0,
        default=0.5
    )
    source_text_quote: Optional[str] = Field(
        description="Text supporting this relationship",
        default=""
    )
    confidence: float = Field(
        description="Confidence in relationship accuracy (0.0-1.0)",
        ge=0.0, 
        le=1.0,
        default=0.7
    )


class GraphExtraction(BaseModel):
    """
    Structured response for initial graph extraction from text.
    Used in process_trace_advanced.py for text-to-graph conversion.
    """
    case_title: str = Field(
        description="Title or brief description of the case",
        min_length=5
    )
    nodes: List[NodeExtraction] = Field(
        description="List of extracted nodes",
        min_length=1
    )
    edges: List[EdgeExtraction] = Field(
        description="List of extracted relationships",
        min_length=0
    )
    extraction_confidence: float = Field(
        description="Overall confidence in extraction quality (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    key_hypotheses: List[str] = Field(
        default_factory=list,
        description="Main hypotheses identified in the text",
        max_length=5
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Information that would improve the analysis",
        max_length=5
    )


class CausalChainSummary(BaseModel):
    """
    Structured response for causal chain narrative generation.
    Used for generating summaries of identified causal paths.
    """
    chain_description: str = Field(
        description="Narrative description of the causal chain",
        min_length=50
    )
    initial_trigger: str = Field(
        description="Description of the triggering event"
    )
    final_outcome: str = Field(
        description="Description of the final outcome"
    )
    intermediate_steps: List[str] = Field(
        description="Key intermediate steps in the causal chain",
        max_length=10
    )
    chain_strength: Literal["strong", "moderate", "weak"] = Field(
        description="Overall strength of the causal chain"
    )
    alternative_explanations: List[str] = Field(
        default_factory=list,
        description="Potential alternative explanations",
        max_length=3
    )


# Property ordering for consistent output
class Config:
    """Base configuration for all models to ensure consistent property ordering."""
    
    @staticmethod
    def get_property_ordering(model_class) -> List[str]:
        """Get consistent property ordering for a model class."""
        return list(model_class.model_fields.keys())


# Export all models for easy importing
__all__ = [
    "VanEveraEvidenceType",
    "EvidenceAssessment", 
    "MechanismAssessment",
    "NarrativeSummary",
    "NodeExtraction",
    "EdgeExtraction", 
    "GraphExtraction",
    "CausalChainSummary",
    "Config"
]