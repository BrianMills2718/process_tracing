"""
Pydantic Schema for Process Tracing Structured Output
Clean, modern approach using schema validation instead of prompt engineering
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, List, Dict, Optional, Union
from datetime import datetime

# Node Types (8 total from ontology)
NodeType = Literal[
    "Event", 
    "Hypothesis", 
    "Evidence", 
    "Causal_Mechanism", 
    "Alternative_Explanation", 
    "Actor", 
    "Condition", 
    "Data_Source"
]

# Allowed values for properties
EventTypeValues = Literal["triggering", "intermediate", "outcome", "unspecified"]
HypothesisStatusValues = Literal["active", "supported", "partially_supported", "refuted", "undetermined"]
EvidenceTypeValues = Literal["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "bayesian", "general"]
MechanismStatusValues = Literal["hypothetical", "supported", "refuted", "unspecified"]
AlternativeStatusValues = Literal["active", "eliminated", "supported", "undetermined"]
ConditionTypeValues = Literal["background", "enabling", "constraining", "scope"]
DataSourceTypeValues = Literal["interview", "document", "observation", "artifact", "general"]
DiagnosticTypeValues = Literal["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "general"]
TestResultValues = Literal["passed", "failed", "ambiguous"]

# Edge Types (21 total from ontology)
EdgeType = Literal[
    "causes",
    "confirms_occurrence", 
    "constrains",
    "contradicts",
    "disproves_occurrence",
    "enables",
    "explains_mechanism",
    "infers",
    "initiates",
    "part_of_mechanism",
    "provides_evidence",
    "provides_evidence_for",
    "refutes",
    "refutes_alternative",
    "supports",
    "supports_alternative",
    "tests_alternative",
    "tests_hypothesis",
    "tests_mechanism",
    "updates_probability",
    "weighs_evidence"
]

class NodeProperties(BaseModel):
    """Properties for all node types with optional fields"""
    description: str = Field(description="Description of the node")
    # Common properties across all node types
    timestamp: Optional[str] = Field(None, description="Timestamp for events")
    location: Optional[str] = Field(None, description="Location for events") 
    certainty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Certainty level")
    # Event specific
    event_type: Optional[EventTypeValues] = Field(None, description="Event type")
    date: Optional[str] = Field(None, description="Date")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    is_point_in_time: Optional[bool] = Field(None, description="Point in time event")
    # Hypothesis specific
    prior_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prior probability")
    posterior_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Posterior probability")
    status: Optional[str] = Field(None, description="Status")
    # Evidence specific
    evidence_type: Optional[EvidenceTypeValues] = Field(None, description="Evidence type")
    source: Optional[str] = Field(None, description="Source")
    credibility: Optional[float] = Field(None, ge=0.0, le=1.0, description="Credibility")
    # Mechanism specific
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence")
    level_of_detail: Optional[str] = Field(None, description="Level of detail")
    # Actor specific
    name: Optional[str] = Field(None, description="Actor name")
    role: Optional[str] = Field(None, description="Actor role")
    intentions: Optional[str] = Field(None, description="Actor intentions")
    beliefs: Optional[str] = Field(None, description="Actor beliefs")
    constraints: Optional[str] = Field(None, description="Actor constraints")
    capabilities: Optional[str] = Field(None, description="Actor capabilities")
    # Condition specific
    condition_type: Optional[ConditionTypeValues] = Field(None, description="Condition type")
    necessity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Necessity")
    temporal_scope: Optional[str] = Field(None, description="Temporal scope")
    spatial_scope: Optional[str] = Field(None, description="Spatial scope")
    # Data Source specific
    data_source_type: Optional[DataSourceTypeValues] = Field(None, description="Data source type")
    bias_risk: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bias risk")
    # Alternative specific
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability")
    key_predictions: Optional[List[str]] = Field(None, description="Key predictions")

class ProcessTracingNode(BaseModel):
    """
    Node in a process tracing graph.
    Represents entities like events, hypotheses, evidence, etc.
    """
    id: str = Field(
        description="Unique identifier for this node (e.g., 'evt_boston_tea_party')"
    )
    type: NodeType = Field(
        description="Type of node - determines what kind of entity this represents"
    )
    properties: NodeProperties = Field(
        description="Node-specific properties like description, certainty, dates, etc."
    )

class EdgeProperties(BaseModel):
    """Properties for all edge types with optional fields"""
    # Common properties
    certainty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Certainty level")
    probative_value: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probative value")
    diagnostic_type: Optional[DiagnosticTypeValues] = Field(None, description="Van Evera diagnostic type")
    # Causation specific
    mechanism_id: Optional[str] = Field(None, description="Mechanism ID")
    causal_type: Optional[str] = Field(None, description="Direct or indirect causation")
    # Testing specific
    test_result: Optional[TestResultValues] = Field(None, description="Test result")
    inferential_test_id: Optional[str] = Field(None, description="Inferential test ID")
    # Mechanism specific
    role: Optional[str] = Field(None, description="Role in mechanism")
    sequence_position: Optional[int] = Field(None, description="Sequence position")
    necessity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Necessity level")
    type_of_claim: Optional[str] = Field(None, description="Type of claim")
    # Evidence specific
    source_text_quote: Optional[str] = Field(None, description="Source text quote")
    reasoning: Optional[str] = Field(None, description="Reasoning")
    target_type: Optional[str] = Field(None, description="Target type")
    refutation_strength: Optional[str] = Field(None, description="Refutation strength")
    # Actor specific
    intention: Optional[str] = Field(None, description="Actor intention")
    agency: Optional[str] = Field(None, description="Actor agency")
    intentionality: Optional[str] = Field(None, description="Intentionality")
    capability_assessment: Optional[float] = Field(None, ge=0.0, le=1.0, description="Capability assessment")
    constraint_factors: Optional[str] = Field(None, description="Constraint factors")
    # Bayesian specific
    prior_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Prior probability")
    posterior_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Posterior probability")
    bayes_factor: Optional[float] = Field(None, ge=0.0, description="Bayes factor")
    # Evidence comparison
    comparison_strength: Optional[float] = Field(None, ge=0.0, le=1.0, description="Comparison strength")
    comparison_type: Optional[str] = Field(None, description="Comparison type")
    # Data source
    credibility: Optional[float] = Field(None, ge=0.0, le=1.0, description="Credibility")
    bias_risk: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bias risk")
    # General
    reason: Optional[str] = Field(None, description="Reason")
    logic_type: Optional[str] = Field(None, description="Logic type")
    label: Optional[str] = Field(None, description="Label")
    description: Optional[str] = Field(None, description="Description")

class ProcessTracingEdge(BaseModel):
    """
    Edge in a process tracing graph.
    Represents relationships between nodes (causation, support, etc.).
    """
    source_id: str = Field(
        description="ID of the source node (where the relationship originates)"
    )
    target_id: str = Field(
        description="ID of the target node (where the relationship points to)"
    )
    type: EdgeType = Field(
        description="Type of relationship between the nodes"
    )
    properties: EdgeProperties = Field(
        description="Edge-specific properties like certainty, probative_value, diagnostic_type, etc."
    )

class ProcessTracingGraph(BaseModel):
    """
    Complete process tracing graph with nodes and edges.
    This is the main output structure for LLM extraction.
    """
    nodes: List[ProcessTracingNode] = Field(
        description="All nodes (entities) extracted from the text"
    )
    edges: List[ProcessTracingEdge] = Field(
        description="All relationships between nodes extracted from the text"
    )
    
    def get_node_types(self) -> List[str]:
        """Get list of unique node types in graph"""
        return list(set(node.type for node in self.nodes))
    
    def get_edge_types(self) -> List[str]:
        """Get list of unique edge types in graph"""
        return list(set(edge.type for edge in self.edges))
    
    def get_coverage_stats(self) -> Dict[str, Dict[str, Union[int, float, List[str]]]]:
        """Get coverage statistics for analysis"""
        node_types = self.get_node_types()
        edge_types = self.get_edge_types()
        
        return {
            "nodes": {
                "total": len(self.nodes),
                "types_found": len(node_types),
                "types_possible": 8,
                "coverage_percent": (len(node_types) / 8) * 100,
                "types": sorted(node_types)
            },
            "edges": {
                "total": len(self.edges),
                "types_found": len(edge_types),
                "types_possible": 21,
                "coverage_percent": (len(edge_types) / 21) * 100,
                "types": sorted(edge_types)
            }
        }

class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process"""
    extraction_method: str = "structured_output"
    model_used: str = "gemini-2.5-flash"
    timestamp: datetime = Field(default_factory=datetime.now)
    prompt_length: int = Field(description="Length of prompt used")
    text_length: int = Field(description="Length of input text")

class StructuredExtractionResult(BaseModel):
    """Complete result including graph and metadata"""
    graph: ProcessTracingGraph
    metadata: ExtractionMetadata
    
    def print_summary(self):
        """Print a summary of extraction results"""
        stats = self.graph.get_coverage_stats()
        print(f"EXTRACTION SUMMARY:")
        print(f"  Nodes: {stats['nodes']['total']} ({stats['nodes']['types_found']}/8 types, {stats['nodes']['coverage_percent']:.1f}%)")
        print(f"  Edges: {stats['edges']['total']} ({stats['edges']['types_found']}/21 types, {stats['edges']['coverage_percent']:.1f}%)")
        print(f"  Node types: {stats['nodes']['types']}")
        print(f"  Edge types: {stats['edges']['types']}")