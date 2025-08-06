"""
Comparative Process Tracing - Data Models Module

Data structures and models for multi-case comparative analysis including
case metadata, cross-case mapping, and mechanism pattern definitions.

Author: Claude Code Implementation  
Date: August 2025
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx


class ComparisonType(Enum):
    MOST_SIMILAR_SYSTEMS = "mss"      # Similar contexts, different outcomes
    MOST_DIFFERENT_SYSTEMS = "mds"    # Different contexts, similar outcomes
    DIVERSE_CASE = "diverse"          # Mixed comparison
    CONTROL_CASE = "control"          # Baseline case


class MechanismType(Enum):
    UNIVERSAL = "universal"           # Present across all cases
    CONDITIONAL = "conditional"       # Present under specific conditions
    CASE_SPECIFIC = "case_specific"   # Unique to particular cases
    VARIANT = "variant"              # Different forms across cases


class ScopeCondition(Enum):
    CONTEXT_DEPENDENT = "context"     # Depends on contextual factors
    TIME_DEPENDENT = "temporal"       # Depends on timing
    ACTOR_DEPENDENT = "actor"         # Depends on specific actors
    RESOURCE_DEPENDENT = "resource"   # Depends on available resources
    INSTITUTIONAL = "institutional"   # Depends on institutional setting


@dataclass
class CaseMetadata:
    """Metadata for individual cases in comparative analysis"""
    case_id: str
    case_name: str
    description: str
    
    # Temporal context
    time_period: Optional[Tuple[datetime, datetime]] = None
    duration: Optional[str] = None
    
    # Contextual factors
    geographic_context: Optional[str] = None
    institutional_context: Optional[str] = None
    economic_context: Optional[str] = None
    political_context: Optional[str] = None
    social_context: Optional[str] = None
    
    # Outcome variables
    primary_outcome: Optional[str] = None
    secondary_outcomes: List[str] = field(default_factory=list)
    outcome_magnitude: Optional[float] = None  # 0.0-1.0 scale
    
    # Control variables
    control_variables: Dict[str, Any] = field(default_factory=dict)
    scope_conditions: List[ScopeCondition] = field(default_factory=list)
    
    # Data quality indicators
    data_quality_score: float = 0.8  # 0.0-1.0 scale
    source_reliability: float = 0.8  # 0.0-1.0 scale
    evidence_completeness: float = 0.8  # 0.0-1.0 scale
    
    # Comparison classification
    comparison_type: Optional[ComparisonType] = None
    reference_cases: List[str] = field(default_factory=list)


@dataclass
class NodeMapping:
    """Mapping between nodes across cases"""
    mapping_id: str
    source_case: str
    target_case: str
    source_node: str
    target_node: str
    
    # Similarity metrics
    semantic_similarity: float = 0.0  # 0.0-1.0 scale
    structural_similarity: float = 0.0  # Based on graph position
    temporal_similarity: float = 0.0   # Based on timing
    functional_similarity: float = 0.0  # Based on causal role
    
    # Overall similarity
    overall_similarity: float = 0.0
    
    # Mapping confidence
    mapping_confidence: float = 0.0  # 0.0-1.0 scale
    manual_verification: bool = False


@dataclass
class MechanismPattern:
    """Represents a recurring causal mechanism across cases"""
    pattern_id: str
    pattern_name: str
    description: str
    
    # Mechanism classification
    mechanism_type: MechanismType
    scope_conditions: List[ScopeCondition]
    
    # Cases where mechanism appears
    participating_cases: List[str]
    case_frequencies: Dict[str, float] = field(default_factory=dict)  # How often in each case
    
    # Pattern structure
    core_nodes: List[str] = field(default_factory=list)  # Essential nodes
    optional_nodes: List[str] = field(default_factory=list)  # May be present
    core_edges: List[Tuple[str, str]] = field(default_factory=list)  # Essential connections
    optional_edges: List[Tuple[str, str]] = field(default_factory=list)  # May be present
    
    # Pattern metrics
    pattern_strength: float = 0.0  # Overall pattern strength across cases
    consistency_score: float = 0.0  # How consistent across cases
    generalizability: float = 0.0   # How generalizable to new cases
    
    # Evidence support
    supporting_evidence: Dict[str, List[str]] = field(default_factory=dict)  # Case -> evidence list
    van_evera_support: Dict[str, str] = field(default_factory=dict)  # Case -> test type
    
    # Variation analysis
    pattern_variations: Dict[str, Any] = field(default_factory=dict)  # Case-specific variations
    boundary_conditions: List[str] = field(default_factory=list)  # When pattern doesn't apply


@dataclass
class CrossCaseEvidence:
    """Evidence that spans multiple cases"""
    evidence_id: str
    evidence_type: str  # Van Evera test type
    description: str
    
    # Cases and evidence
    case_evidence: Dict[str, str] = field(default_factory=dict)  # Case -> evidence text
    evidence_strength: Dict[str, float] = field(default_factory=dict)  # Case -> strength
    
    # Cross-case patterns
    pattern_consistency: float = 0.0  # How consistent across cases
    triangulation_strength: float = 0.0  # Multi-case triangulation
    
    # Meta-evidence assessment
    aggregate_support: float = 0.0  # Combined evidence support
    confidence_level: float = 0.0   # Confidence in cross-case pattern


@dataclass
class ComparisonResult:
    """Results of comparative analysis between cases"""
    comparison_id: str
    comparison_type: ComparisonType
    primary_case: str
    comparison_cases: List[str]
    
    # Similarity analysis
    overall_similarity: float = 0.0
    context_similarity: float = 0.0
    outcome_similarity: float = 0.0
    mechanism_similarity: float = 0.0
    
    # Identified patterns
    shared_mechanisms: List[str] = field(default_factory=list)
    different_mechanisms: List[str] = field(default_factory=list)
    unique_mechanisms: Dict[str, List[str]] = field(default_factory=dict)  # Case -> mechanisms
    
    # Scope conditions
    enabling_conditions: List[str] = field(default_factory=list)
    disabling_conditions: List[str] = field(default_factory=list)
    
    # Causal insights
    causal_necessity: Dict[str, float] = field(default_factory=dict)  # Mechanism -> necessity score
    causal_sufficiency: Dict[str, float] = field(default_factory=dict)  # Mechanism -> sufficiency score
    
    # Analysis metadata
    analysis_confidence: float = 0.0
    analysis_date: datetime = field(default_factory=datetime.now)


@dataclass
class MultiCaseAnalysisResult:
    """Complete results of multi-case comparative analysis"""
    analysis_id: str
    analysis_name: str
    description: str
    
    # Cases involved
    total_cases: int
    case_metadata: Dict[str, CaseMetadata] = field(default_factory=dict)
    
    # Mapping results
    node_mappings: List[NodeMapping] = field(default_factory=list)
    mapping_coverage: float = 0.0  # % of nodes successfully mapped
    
    # Pattern analysis
    mechanism_patterns: List[MechanismPattern] = field(default_factory=list)
    universal_patterns: List[str] = field(default_factory=list)
    conditional_patterns: List[str] = field(default_factory=list)
    
    # Comparison results
    pairwise_comparisons: List[ComparisonResult] = field(default_factory=list)
    mss_analyses: List[ComparisonResult] = field(default_factory=list)
    mds_analyses: List[ComparisonResult] = field(default_factory=list)
    
    # Cross-case evidence
    cross_case_evidence: List[CrossCaseEvidence] = field(default_factory=list)
    triangulation_results: Dict[str, float] = field(default_factory=dict)
    
    # Generalization insights
    scope_conditions: Dict[str, List[ScopeCondition]] = field(default_factory=dict)
    boundary_conditions: List[str] = field(default_factory=list)
    generalizability_assessment: float = 0.0
    
    # Theory building
    emerging_theories: List[str] = field(default_factory=list)
    theory_support: Dict[str, float] = field(default_factory=dict)
    
    # Analysis quality
    overall_confidence: float = 0.0
    methodological_rigor: float = 0.0
    analysis_date: datetime = field(default_factory=datetime.now)


@dataclass
class CaseSelectionCriteria:
    """Criteria for selecting cases for comparative analysis"""
    selection_strategy: str  # "purposive", "random", "theoretical"
    
    # Inclusion criteria
    required_outcome_type: Optional[str] = None
    required_context_factors: List[str] = field(default_factory=list)
    required_time_period: Optional[Tuple[datetime, datetime]] = None
    minimum_data_quality: float = 0.6
    
    # Exclusion criteria
    excluded_contexts: List[str] = field(default_factory=list)
    excluded_outcomes: List[str] = field(default_factory=list)
    
    # Comparison design
    target_comparison_type: ComparisonType = ComparisonType.DIVERSE_CASE
    desired_case_count: int = 3
    maximum_case_count: int = 10
    
    # Theoretical considerations
    theoretical_framework: Optional[str] = None
    key_variables: List[str] = field(default_factory=list)
    control_variables: List[str] = field(default_factory=list)


class ComparativeAnalysisError(Exception):
    """Custom exception for comparative analysis errors"""
    pass


def validate_case_metadata(metadata: CaseMetadata) -> List[str]:
    """
    Validate case metadata for completeness and consistency.
    
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Required fields
    if not metadata.case_id:
        warnings.append("Missing case_id")
    if not metadata.case_name:
        warnings.append("Missing case_name")
    if not metadata.description:
        warnings.append("Missing case description")
    
    # Data quality checks
    if metadata.data_quality_score < 0.5:
        warnings.append(f"Low data quality score: {metadata.data_quality_score}")
    if metadata.source_reliability < 0.5:
        warnings.append(f"Low source reliability: {metadata.source_reliability}")
    if metadata.evidence_completeness < 0.5:
        warnings.append(f"Low evidence completeness: {metadata.evidence_completeness}")
    
    # Temporal consistency
    if metadata.time_period:
        start, end = metadata.time_period
        if start >= end:
            warnings.append("Invalid time period: start >= end")
    
    # Outcome validation
    if metadata.outcome_magnitude is not None:
        if not 0.0 <= metadata.outcome_magnitude <= 1.0:
            warnings.append(f"Invalid outcome magnitude: {metadata.outcome_magnitude}")
    
    return warnings


def calculate_overall_similarity(node_mapping: NodeMapping) -> float:
    """
    Calculate overall similarity score for node mapping.
    
    Returns:
        Combined similarity score (0.0-1.0)
    """
    weights = {
        'semantic': 0.4,
        'structural': 0.3,
        'temporal': 0.2,
        'functional': 0.1
    }
    
    overall = (
        weights['semantic'] * node_mapping.semantic_similarity +
        weights['structural'] * node_mapping.structural_similarity +
        weights['temporal'] * node_mapping.temporal_similarity +
        weights['functional'] * node_mapping.functional_similarity
    )
    
    return min(1.0, max(0.0, overall))


def validate_mechanism_pattern(pattern: MechanismPattern) -> List[str]:
    """
    Validate mechanism pattern for consistency and completeness.
    
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Basic validation
    if not pattern.pattern_id:
        warnings.append("Missing pattern_id")
    if not pattern.pattern_name:
        warnings.append("Missing pattern_name")
    if not pattern.core_nodes:
        warnings.append("No core nodes defined")
    if not pattern.participating_cases:
        warnings.append("No participating cases")
    
    # Pattern strength validation
    if not 0.0 <= pattern.pattern_strength <= 1.0:
        warnings.append(f"Invalid pattern strength: {pattern.pattern_strength}")
    if not 0.0 <= pattern.consistency_score <= 1.0:
        warnings.append(f"Invalid consistency score: {pattern.consistency_score}")
    
    # Case frequency validation
    for case, frequency in pattern.case_frequencies.items():
        if not 0.0 <= frequency <= 1.0:
            warnings.append(f"Invalid frequency for case {case}: {frequency}")
    
    # Mechanism type validation
    if pattern.mechanism_type == MechanismType.UNIVERSAL and len(pattern.participating_cases) < 2:
        warnings.append("Universal mechanism should appear in multiple cases")
    
    return warnings


def create_default_case_metadata(case_id: str, case_name: str) -> CaseMetadata:
    """
    Create default case metadata with reasonable defaults.
    
    Args:
        case_id: Unique case identifier
        case_name: Human-readable case name
        
    Returns:
        CaseMetadata with default values
    """
    return CaseMetadata(
        case_id=case_id,
        case_name=case_name,
        description=f"Case: {case_name}",
        data_quality_score=0.7,
        source_reliability=0.7,
        evidence_completeness=0.7
    )


# Export key classes and functions
__all__ = [
    'ComparisonType', 'MechanismType', 'ScopeCondition',
    'CaseMetadata', 'NodeMapping', 'MechanismPattern', 'CrossCaseEvidence',
    'ComparisonResult', 'MultiCaseAnalysisResult', 'CaseSelectionCriteria',
    'ComparativeAnalysisError', 
    'validate_case_metadata', 'calculate_overall_similarity', 'validate_mechanism_pattern',
    'create_default_case_metadata'
]