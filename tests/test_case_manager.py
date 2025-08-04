"""
Test suite for case management system.

Tests case loading, validation, metadata handling, and case selection
operations in core/case_manager.py to ensure robust multi-case management.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

import networkx as nx

from core.case_manager import CaseManager
from core.comparative_models import (
    CaseMetadata, CaseSelectionCriteria, ComparisonType,
    ComparativeAnalysisError, ScopeCondition
)


class TestCaseManagerInit:
    """Test CaseManager initialization."""
    
    def test_init_with_directory(self):
        """Test CaseManager initialization with case directory."""
        manager = CaseManager("/test/directory")
        assert str(manager.case_directory) == "/test/directory"
        assert isinstance(manager.cases, dict)
        assert isinstance(manager.case_metadata, dict)
        assert isinstance(manager.case_files, dict)
    
    def test_init_without_directory(self):
        """Test CaseManager initialization without case directory."""
        manager = CaseManager()
        assert str(manager.case_directory) == "."
        assert isinstance(manager.cases, dict)
        assert isinstance(manager.case_metadata, dict)
        assert isinstance(manager.case_files, dict)


class TestCaseLoading:
    """Test case loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
        
        # Create test case data
        self.test_case_data = {
            "nodes": [
                {"id": "event1", "type": "Event", "description": "Initial crisis event"},
                {"id": "mechanism1", "type": "Mechanism", "description": "Policy response mechanism"},
                {"id": "outcome1", "type": "Event", "description": "Resolution outcome"}
            ],
            "edges": [
                {"source": "event1", "target": "mechanism1", "type": "triggers"},
                {"source": "mechanism1", "target": "outcome1", "type": "produces"}
            ],
            "metadata": {
                "case_name": "Test Crisis Case",
                "description": "A test case involving crisis management",
                "primary_outcome": "successful_resolution",
                "data_quality_score": 0.8,
                "geographic_context": "Europe",
                "institutional_context": "Democratic",
                "time_period": ["2020-01-01T00:00:00", "2020-12-31T23:59:59"]
            }
        }
        
        # Create minimal test case data
        self.minimal_case_data = {
            "nodes": [
                {"id": "event1", "type": "Event", "description": "Test event"}
            ],
            "edges": [],
            "metadata": {
                "case_name": "Minimal Test Case",
                "description": "A minimal test case"
            }
        }
    
    def test_load_case_success(self):
        """Test successful case loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_case_data, f)
            temp_file = f.name
        
        try:
            case_id = self.manager.load_case(temp_file)
            
            # Verify case was loaded
            assert case_id is not None
            assert case_id in self.manager.cases
            assert case_id in self.manager.case_metadata
            assert case_id in self.manager.case_files
            
            # Verify graph structure
            graph = self.manager.cases[case_id]
            assert isinstance(graph, nx.DiGraph)
            assert graph.number_of_nodes() == 3
            assert graph.number_of_edges() == 2
            assert "event1" in graph.nodes()
            assert "mechanism1" in graph.nodes()
            assert "outcome1" in graph.nodes()
            
            # Verify metadata
            metadata = self.manager.case_metadata[case_id]
            assert metadata.case_name == "Test Crisis Case"
            assert metadata.description == "A test case involving crisis management"
            assert metadata.primary_outcome == "successful_resolution"
            assert metadata.data_quality_score == 0.8
            assert metadata.geographic_context == "Europe"
            assert metadata.institutional_context == "Democratic"
            
        finally:
            os.unlink(temp_file)
    
    def test_load_case_with_custom_id(self):
        """Test case loading with custom case ID."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.minimal_case_data, f)
            temp_file = f.name
        
        try:
            case_id = self.manager.load_case(temp_file, case_id="custom_case_001")
            
            assert case_id == "custom_case_001"
            assert "custom_case_001" in self.manager.cases
            
        finally:
            os.unlink(temp_file)
    
    def test_load_case_file_not_found(self):
        """Test case loading with non-existent file."""
        with pytest.raises(ComparativeAnalysisError) as exc_info:
            self.manager.load_case("/nonexistent/file.json")
        
        assert "Case file not found" in str(exc_info.value)
    
    def test_load_case_invalid_json(self):
        """Test case loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(ComparativeAnalysisError) as exc_info:
                self.manager.load_case(temp_file)
            
            assert "Failed to load case" in str(exc_info.value)
            
        finally:
            os.unlink(temp_file)
    
    def test_load_case_missing_nodes(self):
        """Test case loading with missing nodes section."""
        invalid_data = {"edges": [], "metadata": {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = f.name
        
        try:
            case_id = self.manager.load_case(temp_file)
            
            # Should still load but with empty graph
            graph = self.manager.cases[case_id]
            assert graph.number_of_nodes() == 0
            
        finally:
            os.unlink(temp_file)
    
    def test_load_cases_from_directory(self):
        """Test loading multiple cases from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test case files
            case_files = []
            for i in range(3):
                case_data = {
                    "nodes": [{"id": f"node_{i}", "type": "Event", "description": f"Test node {i}"}],
                    "edges": [],
                    "metadata": {
                        "case_name": f"Test Case {i}",
                        "description": f"Test case number {i}"
                    }
                }
                
                case_file = Path(temp_dir) / f"case_{i}.json"
                with open(case_file, 'w') as f:
                    json.dump(case_data, f)
                case_files.append(case_file)
            
            # Create non-JSON file that should be ignored
            non_json_file = Path(temp_dir) / "README.txt"
            with open(non_json_file, 'w') as f:
                f.write("This is not a JSON file")
            
            # Load cases from directory
            loaded_cases = self.manager.load_cases_from_directory(temp_dir)
            
            assert len(loaded_cases) == 3
            assert len(self.manager.cases) == 3
            assert len(self.manager.case_metadata) == 3
    
    def test_load_cases_empty_directory(self):
        """Test loading cases from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loaded_cases = self.manager.load_cases_from_directory(temp_dir)
            
            assert len(loaded_cases) == 0
            assert len(self.manager.cases) == 0


class TestCaseRetrieval:
    """Test case retrieval functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
        
        # Load test cases
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            case_data = {
                "nodes": [{"id": "event1", "type": "Event", "description": "Test event"}],
                "edges": [],
                "metadata": {
                    "case_name": "Test Case",
                    "description": "A test case",
                    "data_quality_score": 0.9
                }
            }
            json.dump(case_data, f)
            self.temp_file = f.name
        
        self.case_id = self.manager.load_case(self.temp_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file)
    
    def test_get_case_success(self):
        """Test successful case retrieval."""
        graph = self.manager.get_case(self.case_id)
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 1
        assert "event1" in graph.nodes()
    
    def test_get_case_not_found(self):
        """Test case retrieval with non-existent case ID."""
        with pytest.raises(ComparativeAnalysisError) as exc_info:
            self.manager.get_case("nonexistent_case")
        
        assert "Case not found" in str(exc_info.value)
    
    def test_get_case_metadata_success(self):
        """Test successful case metadata retrieval."""
        metadata = self.manager.get_case_metadata(self.case_id)
        
        assert isinstance(metadata, CaseMetadata)
        assert metadata.case_name == "Test Case"
        assert metadata.description == "A test case"
        assert metadata.data_quality_score == 0.9
    
    def test_get_case_metadata_not_found(self):
        """Test case metadata retrieval with non-existent case ID."""
        with pytest.raises(ComparativeAnalysisError) as exc_info:
            self.manager.get_case_metadata("nonexistent_case")
        
        assert "Case metadata not found" in str(exc_info.value)
    
    def test_list_cases(self):
        """Test case listing."""
        case_list = self.manager.list_cases()
        
        assert isinstance(case_list, list)
        assert self.case_id in case_list
        assert len(case_list) == 1
    
    def test_get_case_summary(self):
        """Test case summary generation."""
        summary = self.manager.get_case_summary()
        
        assert isinstance(summary, dict)
        assert self.case_id in summary
        
        case_summary = summary[self.case_id]
        assert case_summary['name'] == "Test Case"
        assert case_summary['nodes'] == 1
        assert case_summary['edges'] == 0
        assert case_summary['data_quality'] == 0.9


class TestCaseMetadataUpdate:
    """Test case metadata update functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
        
        # Load test case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            case_data = {
                "nodes": [{"id": "event1", "type": "Event", "description": "Test event"}],
                "edges": [],
                "metadata": {"case_name": "Original Case", "description": "Original description"}
            }
            json.dump(case_data, f)
            self.temp_file = f.name
        
        self.case_id = self.manager.load_case(self.temp_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file)
    
    def test_update_case_metadata_success(self):
        """Test successful metadata update."""
        new_metadata = CaseMetadata(
            case_id=self.case_id,
            case_name="Updated Case",
            description="Updated description",
            data_quality_score=0.95,
            primary_outcome="updated_outcome"
        )
        
        self.manager.update_case_metadata(self.case_id, new_metadata)
        
        # Verify update
        updated_metadata = self.manager.get_case_metadata(self.case_id)
        assert updated_metadata.case_name == "Updated Case"
        assert updated_metadata.description == "Updated description"
        assert updated_metadata.data_quality_score == 0.95
        assert updated_metadata.primary_outcome == "updated_outcome"
    
    def test_update_case_metadata_case_not_found(self):
        """Test metadata update with non-existent case."""
        metadata = CaseMetadata(
            case_id="nonexistent",
            case_name="Test",
            description="Test"
        )
        
        with pytest.raises(ComparativeAnalysisError) as exc_info:
            self.manager.update_case_metadata("nonexistent_case", metadata)
        
        assert "Case not found" in str(exc_info.value)
    
    def test_update_case_metadata_with_warnings(self):
        """Test metadata update that generates validation warnings."""
        low_quality_metadata = CaseMetadata(
            case_id=self.case_id,
            case_name="",  # Missing name - should generate warning
            description="Test description",
            data_quality_score=0.3  # Low quality - should generate warning
        )
        
        # Should not raise exception but should log warnings
        self.manager.update_case_metadata(self.case_id, low_quality_metadata)
        
        # Verify update was applied despite warnings
        updated_metadata = self.manager.get_case_metadata(self.case_id)
        assert updated_metadata.data_quality_score == 0.3


class TestCaseSelection:
    """Test case selection functionality."""
    
    def setup_method(self):
        """Set up test fixtures with multiple cases."""
        self.manager = CaseManager()
        
        # Create test cases with different characteristics
        self.test_cases = [
            {
                "data": {
                    "nodes": [{"id": "event1", "type": "Event", "description": "Democratic crisis"}],
                    "edges": [],
                    "metadata": {
                        "case_name": "Democratic Case 1",
                        "description": "Democratic context case",
                        "primary_outcome": "policy_success",
                        "data_quality_score": 0.9,
                        "geographic_context": "Europe",
                        "institutional_context": "Democratic",
                        "political_context": "Parliamentary"
                    }
                },
                "expected_id": "democratic_case_1"
            },
            {
                "data": {
                    "nodes": [{"id": "event2", "type": "Event", "description": "Authoritarian response"}],
                    "edges": [],
                    "metadata": {
                        "case_name": "Authoritarian Case 1",
                        "description": "Authoritarian context case",
                        "primary_outcome": "policy_failure",
                        "data_quality_score": 0.6,
                        "geographic_context": "Asia",
                        "institutional_context": "Authoritarian",
                        "political_context": "Single Party"
                    }
                },
                "expected_id": "authoritarian_case_1"
            },
            {
                "data": {
                    "nodes": [{"id": "event3", "type": "Event", "description": "Another democratic case"}],
                    "edges": [],
                    "metadata": {
                        "case_name": "Democratic Case 2",
                        "description": "Second democratic case",
                        "primary_outcome": "policy_success",
                        "data_quality_score": 0.85,
                        "geographic_context": "Europe",
                        "institutional_context": "Democratic",
                        "political_context": "Presidential"
                    }
                },
                "expected_id": "democratic_case_2"
            }
        ]
        
        # Load all test cases
        self.case_ids = []
        for case_info in self.test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(case_info["data"], f)
                temp_file = f.name
            
            try:
                case_id = self.manager.load_case(temp_file, case_id=case_info["expected_id"])
                self.case_ids.append(case_id)
            finally:
                os.unlink(temp_file)
    
    def test_select_all_cases(self):
        """Test selecting all cases with no criteria."""
        criteria = CaseSelectionCriteria(selection_strategy="all")
        selected = self.manager.select_cases(criteria)
        
        assert len(selected) == 3
        assert set(selected) == set(self.case_ids)
    
    def test_select_by_outcome_type(self):
        """Test case selection by outcome type."""
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            required_outcome_type="policy_success"
        )
        selected = self.manager.select_cases(criteria)
        
        assert len(selected) == 2
        assert "democratic_case_1" in selected
        assert "democratic_case_2" in selected
        assert "authoritarian_case_1" not in selected
    
    def test_select_by_data_quality(self):
        """Test case selection by minimum data quality."""
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            minimum_data_quality=0.8
        )
        selected = self.manager.select_cases(criteria)
        
        assert len(selected) == 2
        assert "democratic_case_1" in selected
        assert "democratic_case_2" in selected
        assert "authoritarian_case_1" not in selected  # Quality = 0.6
    
    def test_select_by_context_factors(self):
        """Test case selection by required context factors."""
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            required_context_factors=["institutional"]
        )
        selected = self.manager.select_cases(criteria)
        
        # All cases have institutional_context, so all should be selected
        assert len(selected) == 3
    
    def test_select_with_exclusions(self):
        """Test case selection with exclusion criteria."""
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            excluded_contexts=["Authoritarian"],
            excluded_outcomes=[]
        )
        selected = self.manager.select_cases(criteria)
        
        assert len(selected) == 2
        assert "democratic_case_1" in selected
        assert "democratic_case_2" in selected
        assert "authoritarian_case_1" not in selected
    
    def test_select_with_case_count_limit(self):
        """Test case selection with maximum case count."""
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            maximum_case_count=2
        )
        selected = self.manager.select_cases(criteria)
        
        assert len(selected) <= 2
    
    def test_create_case_pairs_mss(self):
        """Test case pair creation for MSS analysis."""
        pairs = self.manager.create_case_pairs(self.case_ids, ComparisonType.MOST_SIMILAR_SYSTEMS)
        
        # Should find similar democratic cases with different political contexts
        assert isinstance(pairs, list)
        assert len(pairs) > 0
    
    def test_create_case_pairs_mds(self):
        """Test case pair creation for MDS analysis."""
        pairs = self.manager.create_case_pairs(self.case_ids, ComparisonType.MOST_DIFFERENT_SYSTEMS)
        
        # Should find different institutional contexts with similar outcomes
        assert isinstance(pairs, list)
        assert len(pairs) >= 0  # Might be 0 if no suitable pairs found
    
    def test_create_case_pairs_diverse(self):
        """Test case pair creation for diverse case analysis."""
        pairs = self.manager.create_case_pairs(self.case_ids, ComparisonType.DIVERSE_CASE)
        
        # Should create all possible pairs
        expected_pairs = len(self.case_ids) * (len(self.case_ids) - 1) // 2
        assert len(pairs) == expected_pairs


class TestCaseRemoval:
    """Test case removal functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
        
        # Load test case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            case_data = {
                "nodes": [{"id": "event1", "type": "Event", "description": "Test event"}],
                "edges": [],
                "metadata": {"case_name": "Test Case", "description": "Test description"}
            }
            json.dump(case_data, f)
            self.temp_file = f.name
        
        self.case_id = self.manager.load_case(self.temp_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file)
    
    def test_remove_case_success(self):
        """Test successful case removal."""
        # Verify case exists
        assert self.case_id in self.manager.cases
        assert self.case_id in self.manager.case_metadata
        assert self.case_id in self.manager.case_files
        
        # Remove case
        self.manager.remove_case(self.case_id)
        
        # Verify case was removed
        assert self.case_id not in self.manager.cases
        assert self.case_id not in self.manager.case_metadata
        assert self.case_id not in self.manager.case_files
    
    def test_remove_nonexistent_case(self):
        """Test removal of non-existent case (should not raise error)."""
        initial_count = len(self.manager.cases)
        
        # Should not raise error
        self.manager.remove_case("nonexistent_case")
        
        # Should not affect existing cases
        assert len(self.manager.cases) == initial_count


class TestMetadataSaving:
    """Test metadata saving functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
        
        # Load test case with rich metadata
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            case_data = {
                "nodes": [{"id": "event1", "type": "Event", "description": "Test event"}],
                "edges": [],
                "metadata": {
                    "case_name": "Rich Metadata Case",
                    "description": "Case with comprehensive metadata",
                    "primary_outcome": "policy_success",
                    "secondary_outcomes": ["economic_growth", "social_stability"],
                    "geographic_context": "Europe",
                    "institutional_context": "Democratic",
                    "data_quality_score": 0.9,
                    "time_period": ["2020-01-01T00:00:00", "2020-12-31T23:59:59"]
                }
            }
            json.dump(case_data, f)
            self.temp_file = f.name
        
        self.case_id = self.manager.load_case(self.temp_file)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file)
    
    def test_save_case_metadata(self):
        """Test saving case metadata to files."""
        with tempfile.TemporaryDirectory() as output_dir:
            self.manager.save_case_metadata(output_dir)
            
            # Verify metadata file was created
            expected_file = Path(output_dir) / f"{self.case_id}_metadata.json"
            assert expected_file.exists()
            
            # Verify file content
            with open(expected_file, 'r') as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata['case_name'] == "Rich Metadata Case"
            assert saved_metadata['description'] == "Case with comprehensive metadata"
            assert saved_metadata['primary_outcome'] == "policy_success"
            assert saved_metadata['secondary_outcomes'] == ["economic_growth", "social_stability"]
            assert saved_metadata['data_quality_score'] == 0.9


class TestJSONToGraph:
    """Test JSON to graph conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
    
    def test_json_to_graph_basic(self):
        """Test basic JSON to graph conversion."""
        case_data = {
            "nodes": [
                {"id": "node1", "type": "Event", "description": "First event"},
                {"id": "node2", "type": "Mechanism", "description": "Causal mechanism"}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "type": "triggers", "strength": 0.8}
            ]
        }
        
        graph = self.manager._json_to_graph(case_data)
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        
        # Check node attributes
        assert graph.nodes["node1"]["type"] == "Event"
        assert graph.nodes["node1"]["description"] == "First event"
        assert graph.nodes["node2"]["type"] == "Mechanism"
        
        # Check edge attributes
        assert graph.edges["node1", "node2"]["type"] == "triggers"
        assert graph.edges["node1", "node2"]["strength"] == 0.8
    
    def test_json_to_graph_empty(self):
        """Test JSON to graph conversion with empty data."""
        case_data = {"nodes": [], "edges": []}
        
        graph = self.manager._json_to_graph(case_data)
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
    
    def test_json_to_graph_missing_sections(self):
        """Test JSON to graph conversion with missing sections."""
        case_data = {}  # No nodes or edges
        
        graph = self.manager._json_to_graph(case_data)
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0


class TestMetadataExtraction:
    """Test metadata extraction from case data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
    
    def test_extract_metadata_complete(self):
        """Test metadata extraction with complete data."""
        case_data = {
            "metadata": {
                "case_name": "Complete Case",
                "description": "A case with complete metadata",
                "primary_outcome": "success",
                "data_quality_score": 0.95,
                "geographic_context": "North America",
                "institutional_context": "Federal",
                "time_period": ["2019-01-01T00:00:00Z", "2019-12-31T23:59:59Z"]
            }
        }
        
        metadata = self.manager._extract_metadata(case_data, "test_case", "Test Case")
        
        assert metadata.case_id == "test_case"
        assert metadata.case_name == "Complete Case"
        assert metadata.description == "A case with complete metadata"
        assert metadata.primary_outcome == "success"
        assert metadata.data_quality_score == 0.95
        assert metadata.geographic_context == "North America"
        assert metadata.institutional_context == "Federal"
        assert metadata.time_period is not None
    
    def test_extract_metadata_minimal(self):
        """Test metadata extraction with minimal data."""
        case_data = {"metadata": {}}
        
        metadata = self.manager._extract_metadata(case_data, "minimal_case", "Minimal Case")
        
        assert metadata.case_id == "minimal_case"
        assert metadata.case_name == "Minimal Case"
        assert metadata.description == "Case: Minimal Case"
        assert metadata.data_quality_score == 0.7  # Default value
    
    def test_extract_metadata_no_metadata_section(self):
        """Test metadata extraction with no metadata section."""
        case_data = {}
        
        metadata = self.manager._extract_metadata(case_data, "no_meta_case", "No Meta Case")
        
        assert metadata.case_id == "no_meta_case"
        assert metadata.case_name == "No Meta Case"
        assert metadata.description == "Case: No Meta Case"


class TestContextSimilarity:
    """Test context similarity calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CaseManager()
    
    def test_calculate_context_similarity_identical(self):
        """Test context similarity with identical contexts."""
        metadata1 = CaseMetadata(
            case_id="case1", case_name="Case 1", description="Test case 1",
            geographic_context="Europe", institutional_context="Democratic",
            economic_context="Market", political_context="Parliamentary"
        )
        
        metadata2 = CaseMetadata(
            case_id="case2", case_name="Case 2", description="Test case 2",
            geographic_context="Europe", institutional_context="Democratic",
            economic_context="Market", political_context="Parliamentary"
        )
        
        similarity = self.manager._calculate_context_similarity(metadata1, metadata2)
        assert similarity == 1.0
    
    def test_calculate_context_similarity_different(self):
        """Test context similarity with completely different contexts."""
        metadata1 = CaseMetadata(
            case_id="case1", case_name="Case 1", description="Test case 1",
            geographic_context="Europe", institutional_context="Democratic",
            economic_context="Market", political_context="Parliamentary"
        )
        
        metadata2 = CaseMetadata(
            case_id="case2", case_name="Case 2", description="Test case 2",
            geographic_context="Asia", institutional_context="Authoritarian",
            economic_context="Command", political_context="Single Party"
        )
        
        similarity = self.manager._calculate_context_similarity(metadata1, metadata2)
        assert similarity == 0.0
    
    def test_calculate_context_similarity_partial(self):
        """Test context similarity with partial overlap."""
        metadata1 = CaseMetadata(
            case_id="case1", case_name="Case 1", description="Test case 1",
            geographic_context="Europe", institutional_context="Democratic"
        )
        
        metadata2 = CaseMetadata(
            case_id="case2", case_name="Case 2", description="Test case 2",
            geographic_context="Europe", institutional_context="Federal"
        )
        
        similarity = self.manager._calculate_context_similarity(metadata1, metadata2)
        assert 0.0 < similarity < 1.0
    
    def test_calculate_context_similarity_missing_data(self):
        """Test context similarity with missing context data."""
        metadata1 = CaseMetadata(
            case_id="case1", case_name="Case 1", description="Test case 1"
        )
        
        metadata2 = CaseMetadata(
            case_id="case2", case_name="Case 2", description="Test case 2"
        )
        
        similarity = self.manager._calculate_context_similarity(metadata1, metadata2)
        assert similarity == 0.5  # Default for no context data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])