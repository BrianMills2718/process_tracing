"""
Test suite for comparative process tracing data models and validation functions.

Tests all data structures, validation functions, and model operations in 
core/comparative_models.py to ensure robust multi-case analysis capabilities.
"""

import pytest
from datetime import datetime
from typing import List, Dict

from core.comparative_models import (
    CaseMetadata, NodeMapping, MechanismPattern, CrossCaseEvidence,
    ComparisonResult, MultiCaseAnalysisResult, CaseSelectionCriteria,
    ComparisonType, MechanismType, ScopeCondition,
    ComparativeAnalysisError, validate_case_metadata, 
    calculate_overall_similarity, validate_mechanism_pattern,
    create_default_case_metadata
)


class TestCaseMetadata:
    """Test CaseMetadata data structure and validation."""
    
    def test_case_metadata_creation(self):
        """Test basic CaseMetadata creation with required fields."""
        metadata = CaseMetadata(
            case_id="test_case_001",
            case_name="Test Case 1",
            description="A test case for validation"
        )
        
        assert metadata.case_id == "test_case_001"
        assert metadata.case_name == "Test Case 1"
        assert metadata.description == "A test case for validation"
        assert metadata.data_quality_score == 0.8  # Default value
        assert metadata.source_reliability == 0.8  # Default value
        assert metadata.evidence_completeness == 0.8  # Default value
    
    def test_case_metadata_with_temporal_data(self):
        """Test CaseMetadata with temporal information."""
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)
        
        metadata = CaseMetadata(
            case_id="temporal_case",
            case_name="Temporal Test Case",
            description="Case with temporal data",
            time_period=(start_time, end_time),
            duration="1 year"
        )
        
        assert metadata.time_period == (start_time, end_time)
        assert metadata.duration == "1 year"
    
    def test_case_metadata_with_context_factors(self):
        """Test CaseMetadata with all context factors."""
        metadata = CaseMetadata(
            case_id="context_case",
            case_name="Context Test Case",
            description="Case with context factors",
            geographic_context="Europe",
            institutional_context="Democratic",
            economic_context="Market Economy",
            political_context="Parliamentary System",
            social_context="Post-Industrial"
        )
        
        assert metadata.geographic_context == "Europe"
        assert metadata.institutional_context == "Democratic"
        assert metadata.economic_context == "Market Economy"
        assert metadata.political_context == "Parliamentary System"
        assert metadata.social_context == "Post-Industrial"
    
    def test_case_metadata_with_outcomes(self):
        """Test CaseMetadata with outcome variables."""
        metadata = CaseMetadata(
            case_id="outcome_case",
            case_name="Outcome Test Case",
            description="Case with outcome data",
            primary_outcome="policy_success",
            secondary_outcomes=["economic_growth", "public_satisfaction"],
            outcome_magnitude=0.75
        )
        
        assert metadata.primary_outcome == "policy_success"
        assert metadata.secondary_outcomes == ["economic_growth", "public_satisfaction"]
        assert metadata.outcome_magnitude == 0.75
    
    def test_case_metadata_with_scope_conditions(self):
        """Test CaseMetadata with scope conditions."""
        metadata = CaseMetadata(
            case_id="scope_case",
            case_name="Scope Test Case",
            description="Case with scope conditions",
            scope_conditions=[ScopeCondition.CONTEXT_DEPENDENT, ScopeCondition.TIME_DEPENDENT]
        )
        
        assert ScopeCondition.CONTEXT_DEPENDENT in metadata.scope_conditions
        assert ScopeCondition.TIME_DEPENDENT in metadata.scope_conditions
        assert len(metadata.scope_conditions) == 2


class TestCaseMetadataValidation:
    """Test case metadata validation functions."""
    
    def test_validate_complete_metadata(self):
        """Test validation of complete, valid metadata."""
        metadata = CaseMetadata(
            case_id="valid_case",
            case_name="Valid Test Case",
            description="A completely valid test case",
            data_quality_score=0.9,
            source_reliability=0.85,
            evidence_completeness=0.8
        )
        
        warnings = validate_case_metadata(metadata)
        assert len(warnings) == 0
    
    def test_validate_missing_required_fields(self):
        """Test validation detects missing required fields."""
        # Missing case_id
        metadata = CaseMetadata(
            case_id="",
            case_name="Test Case",
            description="Test description"
        )
        
        warnings = validate_case_metadata(metadata)
        assert any("Missing case_id" in warning for warning in warnings)
        
        # Missing case_name
        metadata = CaseMetadata(
            case_id="test_case",
            case_name="",
            description="Test description"
        )
        
        warnings = validate_case_metadata(metadata)
        assert any("Missing case_name" in warning for warning in warnings)
        
        # Missing description
        metadata = CaseMetadata(
            case_id="test_case",
            case_name="Test Case",
            description=""
        )
        
        warnings = validate_case_metadata(metadata)
        assert any("Missing case description" in warning for warning in warnings)
    
    def test_validate_low_quality_scores(self):
        """Test validation detects low quality scores."""
        metadata = CaseMetadata(
            case_id="low_quality_case",
            case_name="Low Quality Case",
            description="Case with low quality scores",
            data_quality_score=0.3,
            source_reliability=0.4,
            evidence_completeness=0.2
        )
        
        warnings = validate_case_metadata(metadata)
        assert any("Low data quality score" in warning for warning in warnings)
        assert any("Low source reliability" in warning for warning in warnings)
        assert any("Low evidence completeness" in warning for warning in warnings)
    
    def test_validate_invalid_time_period(self):
        """Test validation detects invalid time periods."""
        start_time = datetime(2020, 12, 31)
        end_time = datetime(2020, 1, 1)  # End before start
        
        metadata = CaseMetadata(
            case_id="invalid_time_case",
            case_name="Invalid Time Case",
            description="Case with invalid time period",
            time_period=(start_time, end_time)
        )
        
        warnings = validate_case_metadata(metadata)
        assert any("Invalid time period" in warning for warning in warnings)
    
    def test_validate_invalid_outcome_magnitude(self):
        """Test validation detects invalid outcome magnitude."""
        metadata = CaseMetadata(
            case_id="invalid_outcome_case",
            case_name="Invalid Outcome Case",
            description="Case with invalid outcome magnitude",
            outcome_magnitude=1.5  # Should be 0.0-1.0
        )
        
        warnings = validate_case_metadata(metadata)
        assert any("Invalid outcome magnitude" in warning for warning in warnings)


class TestNodeMapping:
    """Test NodeMapping data structure and similarity calculations."""
    
    def test_node_mapping_creation(self):
        """Test basic NodeMapping creation."""
        mapping = NodeMapping(
            mapping_id="test_mapping_001",
            source_case="case1",
            target_case="case2",
            source_node="node1",
            target_node="node2",
            semantic_similarity=0.8,
            structural_similarity=0.7,
            temporal_similarity=0.6,
            functional_similarity=0.9,
            overall_similarity=0.75,
            mapping_confidence=0.85
        )
        
        assert mapping.mapping_id == "test_mapping_001"
        assert mapping.source_case == "case1"
        assert mapping.target_case == "case2"
        assert mapping.source_node == "node1"
        assert mapping.target_node == "node2"
        assert mapping.semantic_similarity == 0.8
        assert mapping.structural_similarity == 0.7
        assert mapping.temporal_similarity == 0.6
        assert mapping.functional_similarity == 0.9
        assert mapping.overall_similarity == 0.75
        assert mapping.mapping_confidence == 0.85
    
    def test_calculate_overall_similarity(self):
        """Test overall similarity calculation function."""
        mapping = NodeMapping(
            mapping_id="test_mapping",
            source_case="case1",
            target_case="case2",
            source_node="node1",
            target_node="node2",
            semantic_similarity=0.8,
            structural_similarity=0.6,
            temporal_similarity=0.4,
            functional_similarity=0.9
        )
        
        overall = calculate_overall_similarity(mapping)
        
        # Expected: 0.4*0.8 + 0.3*0.6 + 0.2*0.4 + 0.1*0.9 = 0.32 + 0.18 + 0.08 + 0.09 = 0.67
        expected = 0.67
        assert abs(overall - expected) < 0.01
        assert 0.0 <= overall <= 1.0
    
    def test_node_mapping_defaults(self):
        """Test NodeMapping with default values."""
        mapping = NodeMapping(
            mapping_id="default_mapping",
            source_case="case1",
            target_case="case2",
            source_node="node1",
            target_node="node2"
        )
        
        assert mapping.semantic_similarity == 0.0
        assert mapping.structural_similarity == 0.0
        assert mapping.temporal_similarity == 0.0
        assert mapping.functional_similarity == 0.0
        assert mapping.overall_similarity == 0.0
        assert mapping.mapping_confidence == 0.0
        assert mapping.manual_verification == False


class TestMechanismPattern:
    """Test MechanismPattern data structure and validation."""
    
    def test_mechanism_pattern_creation(self):
        """Test basic MechanismPattern creation."""
        pattern = MechanismPattern(
            pattern_id="test_pattern_001",
            pattern_name="Test Causal Pattern",
            description="A test mechanism pattern",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[ScopeCondition.CONTEXT_DEPENDENT],
            participating_cases=["case1", "case2", "case3"],
            core_nodes=["Event", "Mechanism", "Outcome"],
            optional_nodes=["Evidence"],
            core_edges=[("Event", "Mechanism"), ("Mechanism", "Outcome")],
            optional_edges=[("Evidence", "Event")],
            pattern_strength=0.85,
            consistency_score=0.9,
            generalizability=0.8
        )
        
        assert pattern.pattern_id == "test_pattern_001"
        assert pattern.pattern_name == "Test Causal Pattern"
        assert pattern.mechanism_type == MechanismType.UNIVERSAL
        assert pattern.participating_cases == ["case1", "case2", "case3"]
        assert pattern.core_nodes == ["Event", "Mechanism", "Outcome"]
        assert pattern.pattern_strength == 0.85
        assert pattern.consistency_score == 0.9
        assert pattern.generalizability == 0.8
    
    def test_mechanism_pattern_with_evidence(self):
        """Test MechanismPattern with evidence support."""
        pattern = MechanismPattern(
            pattern_id="evidence_pattern",
            pattern_name="Pattern with Evidence",
            description="Pattern with supporting evidence",
            mechanism_type=MechanismType.CONDITIONAL,
            scope_conditions=[],
            participating_cases=["case1", "case2"],
            supporting_evidence={
                "case1": ["Evidence source 1", "Evidence source 2"],
                "case2": ["Evidence source 3"]
            },
            van_evera_support={
                "case1": "smoking_gun",
                "case2": "hoop"
            }
        )
        
        assert pattern.supporting_evidence["case1"] == ["Evidence source 1", "Evidence source 2"]
        assert pattern.supporting_evidence["case2"] == ["Evidence source 3"]
        assert pattern.van_evera_support["case1"] == "smoking_gun"
        assert pattern.van_evera_support["case2"] == "hoop"
    
    def test_mechanism_pattern_with_variations(self):
        """Test MechanismPattern with case variations."""
        pattern = MechanismPattern(
            pattern_id="variation_pattern",
            pattern_name="Pattern with Variations",
            description="Pattern with case-specific variations",
            mechanism_type=MechanismType.VARIANT,
            scope_conditions=[ScopeCondition.ACTOR_DEPENDENT],
            participating_cases=["case1", "case2"],
            pattern_variations={
                "case1": {"actor_type": "government", "response_speed": "fast"},
                "case2": {"actor_type": "private", "response_speed": "slow"}
            },
            boundary_conditions=["No institutional capacity", "Resource constraints"]
        )
        
        assert pattern.mechanism_type == MechanismType.VARIANT
        assert ScopeCondition.ACTOR_DEPENDENT in pattern.scope_conditions
        assert pattern.pattern_variations["case1"]["actor_type"] == "government"
        assert pattern.pattern_variations["case2"]["response_speed"] == "slow"
        assert "No institutional capacity" in pattern.boundary_conditions


class TestMechanismPatternValidation:
    """Test mechanism pattern validation functions."""
    
    def test_validate_complete_pattern(self):
        """Test validation of complete, valid pattern."""
        pattern = MechanismPattern(
            pattern_id="valid_pattern",
            pattern_name="Valid Pattern",
            description="A valid test pattern",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1", "case2", "case3"],
            core_nodes=["Event", "Mechanism"],
            pattern_strength=0.8,
            consistency_score=0.9
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert len(warnings) == 0
    
    def test_validate_missing_required_fields(self):
        """Test validation detects missing required fields."""
        # Missing pattern_id
        pattern = MechanismPattern(
            pattern_id="",
            pattern_name="Test Pattern",
            description="Test description",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1"]
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("Missing pattern_id" in warning for warning in warnings)
        
        # Missing pattern_name
        pattern = MechanismPattern(
            pattern_id="test_pattern",
            pattern_name="",
            description="Test description",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1"]
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("Missing pattern_name" in warning for warning in warnings)
    
    def test_validate_no_core_nodes(self):
        """Test validation detects missing core nodes."""
        pattern = MechanismPattern(
            pattern_id="no_nodes_pattern",
            pattern_name="No Nodes Pattern",
            description="Pattern without core nodes",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1"],
            core_nodes=[]  # No core nodes
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("No core nodes defined" in warning for warning in warnings)
    
    def test_validate_no_participating_cases(self):
        """Test validation detects missing participating cases."""
        pattern = MechanismPattern(
            pattern_id="no_cases_pattern",
            pattern_name="No Cases Pattern",
            description="Pattern without participating cases",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=[]  # No participating cases
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("No participating cases" in warning for warning in warnings)
    
    def test_validate_invalid_scores(self):
        """Test validation detects invalid score values."""
        pattern = MechanismPattern(
            pattern_id="invalid_scores_pattern",
            pattern_name="Invalid Scores Pattern",
            description="Pattern with invalid scores",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1"],
            core_nodes=["Event"],
            pattern_strength=1.5,  # Invalid (>1.0)
            consistency_score=-0.1  # Invalid (<0.0)
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("Invalid pattern strength" in warning for warning in warnings)
        assert any("Invalid consistency score" in warning for warning in warnings)
    
    def test_validate_invalid_case_frequencies(self):
        """Test validation detects invalid case frequency values."""
        pattern = MechanismPattern(
            pattern_id="invalid_freq_pattern",
            pattern_name="Invalid Frequency Pattern",
            description="Pattern with invalid case frequencies",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1"],
            core_nodes=["Event"],
            case_frequencies={"case1": 1.5}  # Invalid (>1.0)
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("Invalid frequency for case case1" in warning for warning in warnings)
    
    def test_validate_universal_mechanism_single_case(self):
        """Test validation detects universal mechanism with single case."""
        pattern = MechanismPattern(
            pattern_id="universal_single_pattern",
            pattern_name="Universal Single Pattern",
            description="Universal mechanism with only one case",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=["case1"],  # Only one case
            core_nodes=["Event"]
        )
        
        warnings = validate_mechanism_pattern(pattern)
        assert any("Universal mechanism should appear in multiple cases" in warning for warning in warnings)


class TestCrossCaseEvidence:
    """Test CrossCaseEvidence data structure."""
    
    def test_cross_case_evidence_creation(self):
        """Test basic CrossCaseEvidence creation."""
        evidence = CrossCaseEvidence(
            evidence_id="test_evidence_001",
            evidence_type="smoking_gun",
            description="Cross-case smoking gun evidence",
            case_evidence={
                "case1": "Document clearly shows X caused Y",
                "case2": "Official statement confirms X led to Y"
            },
            evidence_strength={
                "case1": 0.9,
                "case2": 0.85
            },
            pattern_consistency=0.88,
            triangulation_strength=0.92,
            aggregate_support=0.87,
            confidence_level=0.85
        )
        
        assert evidence.evidence_id == "test_evidence_001"
        assert evidence.evidence_type == "smoking_gun"
        assert evidence.case_evidence["case1"] == "Document clearly shows X caused Y"
        assert evidence.evidence_strength["case1"] == 0.9
        assert evidence.pattern_consistency == 0.88
        assert evidence.triangulation_strength == 0.92
        assert evidence.aggregate_support == 0.87
        assert evidence.confidence_level == 0.85


class TestComparisonResult:
    """Test ComparisonResult data structure."""
    
    def test_comparison_result_creation(self):
        """Test basic ComparisonResult creation."""
        result = ComparisonResult(
            comparison_id="test_comparison_001",
            comparison_type=ComparisonType.MOST_SIMILAR_SYSTEMS,
            primary_case="case1",
            comparison_cases=["case2", "case3"],
            overall_similarity=0.85,
            context_similarity=0.9,
            outcome_similarity=0.2,
            mechanism_similarity=0.8,
            shared_mechanisms=["mechanism1", "mechanism2"],
            different_mechanisms=["mechanism3"],
            enabling_conditions=["condition1", "condition2"],
            analysis_confidence=0.88
        )
        
        assert result.comparison_id == "test_comparison_001"
        assert result.comparison_type == ComparisonType.MOST_SIMILAR_SYSTEMS
        assert result.primary_case == "case1"
        assert result.comparison_cases == ["case2", "case3"]
        assert result.overall_similarity == 0.85
        assert result.context_similarity == 0.9
        assert result.outcome_similarity == 0.2
        assert result.shared_mechanisms == ["mechanism1", "mechanism2"]
        assert result.different_mechanisms == ["mechanism3"]


class TestCaseSelectionCriteria:
    """Test CaseSelectionCriteria data structure."""
    
    def test_case_selection_criteria_creation(self):
        """Test basic CaseSelectionCriteria creation."""
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            required_outcome_type="policy_success",
            required_context_factors=["democratic", "developed"],
            minimum_data_quality=0.7,
            excluded_contexts=["authoritarian"],
            excluded_outcomes=["policy_failure"],
            target_comparison_type=ComparisonType.MOST_SIMILAR_SYSTEMS,
            desired_case_count=5,
            maximum_case_count=10,
            theoretical_framework="institutional_theory",
            key_variables=["institutional_capacity", "political_competition"]
        )
        
        assert criteria.selection_strategy == "purposive"
        assert criteria.required_outcome_type == "policy_success"
        assert criteria.required_context_factors == ["democratic", "developed"]
        assert criteria.minimum_data_quality == 0.7
        assert criteria.excluded_contexts == ["authoritarian"]
        assert criteria.target_comparison_type == ComparisonType.MOST_SIMILAR_SYSTEMS
        assert criteria.desired_case_count == 5
        assert criteria.maximum_case_count == 10


class TestCreateDefaultCaseMetadata:
    """Test default case metadata creation function."""
    
    def test_create_default_case_metadata(self):
        """Test creation of default case metadata."""
        metadata = create_default_case_metadata("test_case", "Test Case Name")
        
        assert metadata.case_id == "test_case"
        assert metadata.case_name == "Test Case Name"
        assert metadata.description == "Case: Test Case Name"
        assert metadata.data_quality_score == 0.7
        assert metadata.source_reliability == 0.7
        assert metadata.evidence_completeness == 0.7


class TestComparativeAnalysisError:
    """Test ComparativeAnalysisError exception."""
    
    def test_comparative_analysis_error(self):
        """Test custom exception creation and raising."""
        with pytest.raises(ComparativeAnalysisError) as exc_info:
            raise ComparativeAnalysisError("Test error message")
        
        assert str(exc_info.value) == "Test error message"


class TestEnumValues:
    """Test enum value integrity."""
    
    def test_comparison_type_values(self):
        """Test ComparisonType enum values."""
        assert ComparisonType.MOST_SIMILAR_SYSTEMS.value == "mss"
        assert ComparisonType.MOST_DIFFERENT_SYSTEMS.value == "mds"
        assert ComparisonType.DIVERSE_CASE.value == "diverse"
        assert ComparisonType.CONTROL_CASE.value == "control"
    
    def test_mechanism_type_values(self):
        """Test MechanismType enum values."""
        assert MechanismType.UNIVERSAL.value == "universal"
        assert MechanismType.CONDITIONAL.value == "conditional"
        assert MechanismType.CASE_SPECIFIC.value == "case_specific"
        assert MechanismType.VARIANT.value == "variant"
    
    def test_scope_condition_values(self):
        """Test ScopeCondition enum values."""
        assert ScopeCondition.CONTEXT_DEPENDENT.value == "context"
        assert ScopeCondition.TIME_DEPENDENT.value == "temporal"
        assert ScopeCondition.ACTOR_DEPENDENT.value == "actor"
        assert ScopeCondition.RESOURCE_DEPENDENT.value == "resource"
        assert ScopeCondition.INSTITUTIONAL.value == "institutional"


# Performance and stress tests
class TestPerformanceAndLimits:
    """Test performance and limit conditions."""
    
    def test_large_case_metadata_list(self):
        """Test handling of large metadata lists."""
        metadata_list = []
        for i in range(100):
            metadata = CaseMetadata(
                case_id=f"case_{i:03d}",
                case_name=f"Case {i}",
                description=f"Test case number {i}"
            )
            metadata_list.append(metadata)
        
        assert len(metadata_list) == 100
        assert all(isinstance(m, CaseMetadata) for m in metadata_list)
    
    def test_large_node_mapping_list(self):
        """Test handling of large node mapping lists."""
        mappings = []
        for i in range(1000):
            mapping = NodeMapping(
                mapping_id=f"mapping_{i:04d}",
                source_case=f"case_{i % 10}",
                target_case=f"case_{(i+1) % 10}",
                source_node=f"node_{i % 50}",
                target_node=f"node_{(i+1) % 50}",
                overall_similarity=0.5 + (i % 50) / 100.0
            )
            mappings.append(mapping)
        
        assert len(mappings) == 1000
        assert all(isinstance(m, NodeMapping) for m in mappings)
    
    def test_mechanism_pattern_with_many_cases(self):
        """Test mechanism pattern with many participating cases."""
        many_cases = [f"case_{i:03d}" for i in range(50)]
        
        pattern = MechanismPattern(
            pattern_id="large_pattern",
            pattern_name="Large Pattern",
            description="Pattern with many cases",
            mechanism_type=MechanismType.UNIVERSAL,
            scope_conditions=[],
            participating_cases=many_cases,
            core_nodes=["Event", "Mechanism", "Outcome"]
        )
        
        assert len(pattern.participating_cases) == 50
        assert pattern.participating_cases[0] == "case_000"
        assert pattern.participating_cases[49] == "case_049"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])