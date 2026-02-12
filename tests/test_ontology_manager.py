"""
Comprehensive test suite for OntologyManager.

Tests all methods and ensures backwards compatibility with existing hardcoded edge lists.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ontology_manager import OntologyManager, ontology_manager, get_evidence_hypothesis_edges, get_van_evera_diagnostic_edges


class TestOntologyManager:
    """Test suite for OntologyManager functionality."""
    
    def test_initialization(self):
        """Test that OntologyManager initializes correctly."""
        manager = OntologyManager()
        assert manager.node_types is not None
        assert manager.edge_types is not None
        assert len(manager.node_types) > 0
        assert len(manager.edge_types) > 0
    
    def test_get_evidence_hypothesis_edges(self):
        """Test retrieval of Evidence→Hypothesis edges."""
        edges = ontology_manager.get_evidence_hypothesis_edges()
        
        # These are the edges that connect Evidence to Hypothesis in the ontology
        expected_edges = [
            'tests_hypothesis',
            'updates_probability', 
            'supports',
            'provides_evidence_for',
            'weighs_evidence'
        ]
        
        for expected in expected_edges:
            assert expected in edges, f"Missing expected edge type: {expected}"
        
        # Verify all returned edges actually connect Evidence to Hypothesis
        for edge in edges:
            edge_config = ontology_manager.edge_types[edge]
            assert 'Evidence' in edge_config.get('domain', [])
            assert 'Hypothesis' in edge_config.get('range', [])
    
    def test_backwards_compatibility_wrapper(self):
        """Test backwards compatibility function."""
        edges = get_evidence_hypothesis_edges()
        assert isinstance(edges, list)
        assert len(edges) > 0
        assert 'tests_hypothesis' in edges
    
    def test_get_van_evera_edges(self):
        """Test retrieval of Van Evera diagnostic edges."""
        edges = ontology_manager.get_van_evera_edges()
        
        # All edges with diagnostic_type property should be included
        for edge in edges:
            properties = ontology_manager.edge_types[edge].get('properties', {})
            has_diagnostic = any(prop in properties for prop in ['diagnostic_type', 'probative_value', 'test_result'])
            assert has_diagnostic, f"Edge {edge} lacks diagnostic properties"
        
        # Specific edges we know have diagnostic properties
        assert 'tests_hypothesis' in edges
        assert 'supports' in edges  # Has diagnostic_type property
    
    def test_get_edge_types_for_relationship(self):
        """Test getting edges for specific node type pairs."""
        # Evidence → Hypothesis
        edges = ontology_manager.get_edge_types_for_relationship('Evidence', 'Hypothesis')
        assert 'tests_hypothesis' in edges
        assert 'supports' in edges
        
        # Event → Event  
        edges = ontology_manager.get_edge_types_for_relationship('Event', 'Event')
        assert 'causes' in edges
        
        # Actor → Event
        edges = ontology_manager.get_edge_types_for_relationship('Actor', 'Event')
        assert 'initiates' in edges
        
        # Non-existent relationship
        edges = ontology_manager.get_edge_types_for_relationship('Actor', 'Condition')
        assert edges == []
    
    def test_validate_edge_valid(self):
        """Test edge validation with valid edge."""
        edge = {
            'type': 'tests_hypothesis',
            'source': 'evidence_1',
            'target': 'hypothesis_1',
            'properties': {
                'probative_value': 0.7,
                'diagnostic_type': 'hoop'
            }
        }
        
        is_valid, error = ontology_manager.validate_edge(edge)
        assert is_valid is True
        assert error is None
    
    def test_validate_edge_invalid_type(self):
        """Test edge validation with invalid edge type."""
        edge = {
            'type': 'nonexistent_edge',
            'source': 'evidence_1',
            'target': 'hypothesis_1',
            'properties': {}
        }
        
        is_valid, error = ontology_manager.validate_edge(edge)
        assert is_valid is False
        assert 'Unknown edge type' in error
    
    def test_validate_edge_invalid_property_value(self):
        """Test edge validation with invalid property value."""
        edge = {
            'type': 'tests_hypothesis',
            'source': 'evidence_1',
            'target': 'hypothesis_1',
            'properties': {
                'diagnostic_type': 'invalid_type'  # Not in allowed values
            }
        }
        
        is_valid, error = ontology_manager.validate_edge(edge)
        assert is_valid is False
        assert 'Invalid value' in error
    
    def test_validate_edge_out_of_range(self):
        """Test edge validation with out-of-range numeric value."""
        edge = {
            'type': 'tests_hypothesis',
            'source': 'evidence_1', 
            'target': 'hypothesis_1',
            'properties': {
                'probative_value': 1.5  # Above max of 1.0
            }
        }
        
        is_valid, error = ontology_manager.validate_edge(edge)
        assert is_valid is False
        assert 'above maximum' in error
    
    def test_get_edge_properties(self):
        """Test retrieving edge properties."""
        props = ontology_manager.get_edge_properties('tests_hypothesis')
        assert 'probative_value' in props
        assert 'diagnostic_type' in props
        assert 'test_result' in props
        
        # Test non-existent edge type
        props = ontology_manager.get_edge_properties('nonexistent')
        assert props == {}
    
    def test_get_required_properties(self):
        """Test retrieving required properties."""
        # Most edge properties are optional in our ontology
        required = ontology_manager.get_required_properties('tests_hypothesis')
        assert isinstance(required, list)
        
        # Test node required properties
        required = ontology_manager.get_node_required_properties('Evidence')
        assert 'description' in required
        assert 'type' in required
    
    def test_get_edges_by_domain(self):
        """Test getting edges by domain node type."""
        edges = ontology_manager.get_edges_by_domain('Evidence')
        assert 'tests_hypothesis' in edges
        assert 'contradicts' in edges
        assert 'supports' in edges
        
        edges = ontology_manager.get_edges_by_domain('Actor')
        assert 'initiates' in edges
    
    def test_get_edges_by_range(self):
        """Test getting edges by range node type."""
        edges = ontology_manager.get_edges_by_range('Hypothesis')
        assert 'tests_hypothesis' in edges
        assert 'supports' in edges
        
        edges = ontology_manager.get_edges_by_range('Event')
        assert 'causes' in edges
        assert 'initiates' in edges
    
    def test_get_all_diagnostic_edge_types(self):
        """Test getting all edges with diagnostic_type property."""
        edges = ontology_manager.get_all_diagnostic_edge_types()
        
        # Check known diagnostic edges
        assert 'tests_hypothesis' in edges
        assert 'supports' in edges
        assert 'refutes' in edges
        assert 'confirms_occurrence' in edges
        
        # Verify all have diagnostic_type property
        for edge in edges:
            props = ontology_manager.edge_types[edge].get('properties', {})
            assert 'diagnostic_type' in props
    
    def test_is_evidence_to_hypothesis_edge(self):
        """Test checking if edge connects Evidence to Hypothesis."""
        assert ontology_manager.is_evidence_to_hypothesis_edge('tests_hypothesis') is True
        assert ontology_manager.is_evidence_to_hypothesis_edge('supports') is True
        assert ontology_manager.is_evidence_to_hypothesis_edge('causes') is False
        assert ontology_manager.is_evidence_to_hypothesis_edge('initiates') is False
    
    def test_get_node_properties(self):
        """Test retrieving node properties."""
        props = ontology_manager.get_node_properties('Evidence')
        assert 'description' in props
        assert 'type' in props
        assert 'certainty' in props
        
        # Test non-existent node type
        props = ontology_manager.get_node_properties('NonexistentNode')
        assert props == {}
    
    def test_get_edge_label(self):
        """Test retrieving edge display labels."""
        label = ontology_manager.get_edge_label('tests_hypothesis')
        assert label == 'tests hypothesis'
        
        label = ontology_manager.get_edge_label('causes')
        assert label == 'causes'
        
        # Non-existent edge should return the edge type itself
        label = ontology_manager.get_edge_label('nonexistent')
        assert label == 'nonexistent'
    
    def test_get_node_color(self):
        """Test retrieving node colors."""
        # Should return colors from NODE_COLORS
        color = ontology_manager.get_node_color('Evidence')
        assert color.startswith('#')  # Should be hex color
        
        # Non-existent node should return default gray
        color = ontology_manager.get_node_color('NonexistentNode')
        assert color == '#808080'
    
    def test_get_all_edge_types(self):
        """Test retrieving all edge types."""
        edges = ontology_manager.get_all_edge_types()
        assert isinstance(edges, list)
        assert len(edges) > 0
        assert 'causes' in edges
        assert 'tests_hypothesis' in edges
        assert 'supports' in edges
    
    def test_get_all_node_types(self):
        """Test retrieving all node types."""
        nodes = ontology_manager.get_all_node_types()
        assert isinstance(nodes, list)
        assert len(nodes) > 0
        assert 'Event' in nodes
        assert 'Evidence' in nodes
        assert 'Hypothesis' in nodes
        assert 'Actor' in nodes
    
    def test_lookup_table_completeness(self):
        """Test that lookup tables contain all edges."""
        all_edges = set(ontology_manager.get_all_edge_types())
        
        # Check that all edges appear in at least one lookup table
        edges_in_tables = set()
        for edges in ontology_manager.edge_by_domain.values():
            edges_in_tables.update(edges)
        
        for edge in all_edges:
            edge_config = ontology_manager.edge_types[edge]
            if edge_config.get('domain'):
                assert edge in edges_in_tables, f"Edge {edge} missing from lookup tables"
    
    def test_backwards_compatibility_with_hardcoded_lists(self):
        """Test that new dynamic queries match old hardcoded lists."""
        # These are hardcoded lists found in the codebase that we're replacing
        
        # From various files that check Evidence→Hypothesis edges
        old_evidence_hypothesis_edges = [
            'supports',
            'provides_evidence_for',
            'tests_hypothesis',
            'updates_probability',
            'weighs_evidence'
        ]
        
        new_edges = ontology_manager.get_evidence_hypothesis_edges()
        
        # All old edges should still be present
        for edge in old_evidence_hypothesis_edges:
            assert edge in new_edges, f"Lost backwards compatibility for edge: {edge}"
        
        # Van Evera edges
        old_van_evera_edges = [
            'tests_hypothesis',
            'supports',  # Has diagnostic_type
            'refutes'    # Has diagnostic_type
        ]
        
        new_van_evera = ontology_manager.get_van_evera_edges()
        
        for edge in old_van_evera_edges:
            if edge in ['supports', 'refutes']:
                # These edges might only be included if they have Evidence in domain
                edge_config = ontology_manager.edge_types[edge]
                if 'Evidence' in edge_config.get('domain', []) and 'Hypothesis' in edge_config.get('range', []):
                    assert edge in new_van_evera, f"Lost Van Evera edge: {edge}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])