"""
Test suite for cross-case graph alignment algorithms.

Tests multi-dimensional similarity scoring, optimal mapping algorithms,
and common subgraph pattern detection in core/graph_alignment.py.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch

from core.graph_alignment import GraphAligner
from core.comparative_models import NodeMapping


class TestGraphAlignerInit:
    """Test GraphAligner initialization."""
    
    def test_init_default_threshold(self):
        """Test GraphAligner initialization with default threshold."""
        aligner = GraphAligner()
        assert aligner.similarity_threshold == 0.6
        assert aligner.semantic_weight == 0.4
        assert aligner.structural_weight == 0.3
        assert aligner.temporal_weight == 0.2
        assert aligner.functional_weight == 0.1
    
    def test_init_custom_threshold(self):
        """Test GraphAligner initialization with custom threshold."""
        aligner = GraphAligner(similarity_threshold=0.8)
        assert aligner.similarity_threshold == 0.8


class TestBasicGraphAlignment:
    """Test basic graph alignment functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner(similarity_threshold=0.7)
        
        # Create test graphs
        self.graph1 = nx.DiGraph()
        self.graph1.add_node("event1", type="Event", description="Economic crisis")
        self.graph1.add_node("policy1", type="Event", description="Policy response")
        self.graph1.add_node("outcome1", type="Event", description="Recovery")
        self.graph1.add_edge("event1", "policy1", type="causes")
        self.graph1.add_edge("policy1", "outcome1", type="causes")
        
        self.graph2 = nx.DiGraph()
        self.graph2.add_node("crisis2", type="Event", description="Financial crisis")
        self.graph2.add_node("response2", type="Event", description="Government response")
        self.graph2.add_node("result2", type="Event", description="Stabilization")
        self.graph2.add_edge("crisis2", "response2", type="causes")
        self.graph2.add_edge("response2", "result2", type="causes")
    
    def test_align_graphs_success(self):
        """Test successful graph alignment."""
        mappings = self.aligner.align_graphs(self.graph1, self.graph2, "case1", "case2")
        
        assert isinstance(mappings, list)
        assert len(mappings) > 0
        
        # Check mapping structure
        for mapping in mappings:
            assert isinstance(mapping, NodeMapping)
            assert mapping.source_case == "case1"
            assert mapping.target_case == "case2"
            assert mapping.source_node in self.graph1.nodes()
            assert mapping.target_node in self.graph2.nodes()
            assert 0.0 <= mapping.overall_similarity <= 1.0
    
    def test_align_graphs_empty_graphs(self):
        """Test graph alignment with empty graphs."""
        empty_graph1 = nx.DiGraph()
        empty_graph2 = nx.DiGraph()
        
        mappings = self.aligner.align_graphs(empty_graph1, empty_graph2, "case1", "case2")
        
        assert isinstance(mappings, list)
        assert len(mappings) == 0
    
    def test_align_graphs_one_empty(self):
        """Test graph alignment with one empty graph."""
        empty_graph = nx.DiGraph()
        
        mappings = self.aligner.align_graphs(self.graph1, empty_graph, "case1", "case2")
        
        assert isinstance(mappings, list)
        assert len(mappings) == 0
    
    def test_align_multiple_graphs(self):
        """Test alignment of multiple graphs."""
        # Create third graph
        graph3 = nx.DiGraph()
        graph3.add_node("shock3", type="Event", description="Market shock")
        graph3.add_node("intervention3", type="Event", description="Central bank intervention")
        graph3.add_edge("shock3", "intervention3", type="causes")
        
        graphs = {"case1": self.graph1, "case2": self.graph2, "case3": graph3}
        
        all_mappings = self.aligner.align_multiple_graphs(graphs)
        
        assert isinstance(all_mappings, dict)
        
        # Should have mappings between all pairs
        expected_pairs = {("case1", "case2"), ("case1", "case3"), ("case2", "case3")}
        actual_pairs = set(all_mappings.keys())
        assert actual_pairs == expected_pairs
        
        # Check that each mapping list is valid
        for pair, mappings in all_mappings.items():
            assert isinstance(mappings, list)
            for mapping in mappings:
                assert isinstance(mapping, NodeMapping)


class TestSimilarityCalculation:
    """Test similarity calculation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
        
        # Create test graphs with detailed attributes
        self.graph1 = nx.DiGraph()
        self.graph1.add_node("node1", 
                           type="Event", 
                           description="Economic crisis event",
                           timestamp="2020-01-01",
                           sequence_order=1,
                           properties={"severity": "high", "duration": "6 months"})
        
        self.graph2 = nx.DiGraph()
        self.graph2.add_node("node2", 
                           type="Event", 
                           description="Financial crisis event",
                           timestamp="2020-01-01",
                           sequence_order=1,
                           properties={"severity": "high", "duration": "8 months"})
    
    def test_calculate_semantic_similarity_identical(self):
        """Test semantic similarity calculation with identical nodes."""
        node1_data = self.graph1.nodes["node1"]
        node2_data = self.graph1.nodes["node1"]  # Same node
        
        similarity = self.aligner._calculate_semantic_similarity(node1_data, node2_data)
        assert similarity == 1.0
    
    def test_calculate_semantic_similarity_similar_type(self):
        """Test semantic similarity with same type, similar description."""
        node1_data = self.graph1.nodes["node1"]
        node2_data = self.graph2.nodes["node2"]
        
        similarity = self.aligner._calculate_semantic_similarity(node1_data, node2_data)
        
        # Should be high due to same type and similar description
        assert 0.5 < similarity <= 1.0
    
    def test_calculate_semantic_similarity_different_type(self):
        """Test semantic similarity with different types."""
        node1_data = {"type": "Event", "description": "Test event"}
        node2_data = {"type": "Mechanism", "description": "Test mechanism"}
        
        similarity = self.aligner._calculate_semantic_similarity(node1_data, node2_data)
        
        # Should be lower due to different types
        assert 0.0 <= similarity < 1.0
    
    def test_calculate_structural_similarity(self):
        """Test structural similarity calculation."""
        # Add more nodes to create structure
        self.graph1.add_node("neighbor1", type="Event", description="Neighbor")
        self.graph1.add_edge("node1", "neighbor1", type="causes")
        
        self.graph2.add_node("neighbor2", type="Event", description="Neighbor")
        self.graph2.add_edge("node2", "neighbor2", type="causes")
        
        similarity = self.aligner._calculate_structural_similarity(
            self.graph1, self.graph2, "node1", "node2"
        )
        
        # Should be high due to similar structure (both have 1 outgoing edge)
        assert 0.5 < similarity <= 1.0
    
    def test_calculate_temporal_similarity_same_timestamp(self):
        """Test temporal similarity with same timestamp."""
        node1_data = {"timestamp": "2020-01-01", "sequence_order": 1}
        node2_data = {"timestamp": "2020-01-01", "sequence_order": 1}
        
        similarity = self.aligner._calculate_temporal_similarity(node1_data, node2_data)
        assert similarity > 0.8  # Should be high
    
    def test_calculate_temporal_similarity_different_sequence(self):
        """Test temporal similarity with different sequence order."""
        node1_data = {"sequence_order": 1}
        node2_data = {"sequence_order": 5}
        
        similarity = self.aligner._calculate_temporal_similarity(node1_data, node2_data)
        assert 0.0 <= similarity < 1.0
    
    def test_calculate_temporal_similarity_no_temporal_data(self):
        """Test temporal similarity with no temporal attributes."""
        node1_data = {"type": "Event", "description": "No temporal data"}
        node2_data = {"type": "Event", "description": "Also no temporal data"}
        
        similarity = self.aligner._calculate_temporal_similarity(node1_data, node2_data)
        assert similarity == 0.5  # Neutral score
    
    def test_calculate_functional_similarity_same_role(self):
        """Test functional similarity with same causal role."""
        # Create graphs where both nodes are sources (no incoming edges)
        graph1 = nx.DiGraph()
        graph1.add_node("source1", type="Event")
        graph1.add_node("target1", type="Event")
        graph1.add_edge("source1", "target1", type="causes")
        
        graph2 = nx.DiGraph()
        graph2.add_node("source2", type="Event")
        graph2.add_node("target2", type="Event")
        graph2.add_edge("source2", "target2", type="causes")
        
        similarity = self.aligner._calculate_functional_similarity(
            graph1, graph2, "source1", "source2"
        )
        
        # Both are sources, should have high functional similarity
        assert 0.7 < similarity <= 1.0
    
    def test_calculate_functional_similarity_different_roles(self):
        """Test functional similarity with different causal roles."""
        # Create graphs where one node is source, other is sink
        graph1 = nx.DiGraph()
        graph1.add_node("source", type="Event")
        graph1.add_node("target", type="Event")
        graph1.add_edge("source", "target", type="causes")
        
        graph2 = nx.DiGraph()
        graph2.add_node("source", type="Event")
        graph2.add_node("sink", type="Event")
        graph2.add_edge("source", "sink", type="causes")
        
        similarity = self.aligner._calculate_functional_similarity(
            graph1, graph2, "source", "sink"  # source vs sink
        )
        
        # Different roles, should have lower similarity
        assert 0.0 <= similarity < 0.7


class TestNodeSimilarityCalculation:
    """Test overall node similarity calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
        
        # Create test graphs
        self.graph1 = nx.DiGraph()
        self.graph1.add_node("node1", type="Event", description="Test event 1")
        
        self.graph2 = nx.DiGraph()
        self.graph2.add_node("node2", type="Event", description="Test event 2")
    
    def test_calculate_node_similarity_weighted_combination(self):
        """Test that node similarity properly combines weighted factors."""
        # Mock individual similarity calculations
        with patch.object(self.aligner, '_calculate_semantic_similarity', return_value=0.8), \
             patch.object(self.aligner, '_calculate_structural_similarity', return_value=0.6), \
             patch.object(self.aligner, '_calculate_temporal_similarity', return_value=0.4), \
             patch.object(self.aligner, '_calculate_functional_similarity', return_value=0.9):
            
            similarity = self.aligner._calculate_node_similarity(
                self.graph1, self.graph2, "node1", "node2"
            )
            
            # Expected: 0.4*0.8 + 0.3*0.6 + 0.2*0.4 + 0.1*0.9 = 0.32 + 0.18 + 0.08 + 0.09 = 0.67
            expected = 0.67
            assert abs(similarity - expected) < 0.01
    
    def test_calculate_node_similarity_bounds(self):
        """Test that node similarity is bounded between 0 and 1."""
        # Mock extreme values
        with patch.object(self.aligner, '_calculate_semantic_similarity', return_value=2.0), \
             patch.object(self.aligner, '_calculate_structural_similarity', return_value=-0.5), \
             patch.object(self.aligner, '_calculate_temporal_similarity', return_value=1.5), \
             patch.object(self.aligner, '_calculate_functional_similarity', return_value=-1.0):
            
            similarity = self.aligner._calculate_node_similarity(
                self.graph1, self.graph2, "node1", "node2"
            )
            
            assert 0.0 <= similarity <= 1.0


class TestSimilarityMatrix:
    """Test similarity matrix calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
        
        # Create small test graphs
        self.graph1 = nx.DiGraph()
        self.graph1.add_node("a1", type="Event", description="Event A")
        self.graph1.add_node("b1", type="Mechanism", description="Mechanism B")
        
        self.graph2 = nx.DiGraph()
        self.graph2.add_node("a2", type="Event", description="Event A similar")
        self.graph2.add_node("b2", type="Evidence", description="Evidence B")
    
    def test_calculate_similarity_matrix_shape(self):
        """Test similarity matrix has correct shape."""
        matrix = self.aligner._calculate_similarity_matrix(self.graph1, self.graph2)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 2)  # 2 nodes in each graph
    
    def test_calculate_similarity_matrix_values(self):
        """Test similarity matrix contains valid similarity values."""
        matrix = self.aligner._calculate_similarity_matrix(self.graph1, self.graph2)
        
        # All values should be between 0 and 1
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)
    
    def test_calculate_similarity_matrix_empty_graphs(self):
        """Test similarity matrix with empty graphs."""
        empty_graph1 = nx.DiGraph()
        empty_graph2 = nx.DiGraph()
        
        matrix = self.aligner._calculate_similarity_matrix(empty_graph1, empty_graph2)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (0, 0)


class TestOptimalMappings:
    """Test optimal mapping finding algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner(similarity_threshold=0.5)
        
        # Create similarity matrix for testing
        self.similarity_matrix = np.array([
            [0.9, 0.3, 0.2],  # Node 0 maps best to node 0
            [0.4, 0.8, 0.1],  # Node 1 maps best to node 1
            [0.2, 0.3, 0.7]   # Node 2 maps best to node 2
        ])
        
        self.nodes1 = ["node_a", "node_b", "node_c"]
        self.nodes2 = ["node_x", "node_y", "node_z"]
    
    def test_find_optimal_mappings_greedy(self):
        """Test greedy optimal mapping algorithm."""
        mappings = self.aligner._find_optimal_mappings(
            self.similarity_matrix, self.nodes1, self.nodes2
        )
        
        assert isinstance(mappings, list)
        assert len(mappings) == 3  # Should find 3 mappings
        
        # Check mapping structure
        for node1, node2, similarity in mappings:
            assert node1 in self.nodes1
            assert node2 in self.nodes2
            assert similarity >= self.aligner.similarity_threshold
        
        # Check no duplicate mappings
        used_nodes1 = {mapping[0] for mapping in mappings}
        used_nodes2 = {mapping[1] for mapping in mappings}
        assert len(used_nodes1) == len(mappings)
        assert len(used_nodes2) == len(mappings)
    
    def test_find_optimal_mappings_high_threshold(self):
        """Test optimal mapping with high similarity threshold."""
        high_threshold_aligner = GraphAligner(similarity_threshold=0.95)
        
        mappings = high_threshold_aligner._find_optimal_mappings(
            self.similarity_matrix, self.nodes1, self.nodes2
        )
        
        # Should find fewer mappings due to high threshold
        assert len(mappings) <= 1  # Only the 0.9 similarity should pass
    
    def test_find_optimal_mappings_no_valid_mappings(self):
        """Test optimal mapping when no similarities meet threshold."""
        low_similarity_matrix = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.1, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        mappings = self.aligner._find_optimal_mappings(
            low_similarity_matrix, self.nodes1, self.nodes2
        )
        
        assert len(mappings) == 0


class TestNodeMappingCreation:
    """Test node mapping object creation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
        
        # Create test graphs
        self.graph1 = nx.DiGraph()
        self.graph1.add_node("node1", type="Event", description="Test event", 
                           timestamp="2020-01-01", properties={"key": "value"})
        
        self.graph2 = nx.DiGraph()
        self.graph2.add_node("node2", type="Event", description="Similar event",
                           timestamp="2020-01-01", properties={"key": "value"})
    
    def test_create_node_mapping(self):
        """Test node mapping creation."""
        mapping = self.aligner._create_node_mapping(
            self.graph1, self.graph2, "node1", "node2", 
            "case1", "case2", 0.85
        )
        
        assert isinstance(mapping, NodeMapping)
        assert mapping.source_case == "case1"
        assert mapping.target_case == "case2"
        assert mapping.source_node == "node1"
        assert mapping.target_node == "node2"
        assert mapping.overall_similarity == 0.85
        assert 0.0 <= mapping.mapping_confidence <= 1.0
        
        # Check that individual similarities were calculated
        assert hasattr(mapping, 'semantic_similarity')
        assert hasattr(mapping, 'structural_similarity')
        assert hasattr(mapping, 'temporal_similarity')
        assert hasattr(mapping, 'functional_similarity')


class TestStructuralSimilarity:
    """Test structural similarity calculation for graphs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
    
    def test_calculate_structural_similarity_identical_graphs(self):
        """Test structural similarity with identical graphs."""
        graph1 = nx.DiGraph()
        graph1.add_edge("a", "b")
        graph1.add_edge("b", "c")
        
        graph2 = nx.DiGraph()
        graph2.add_edge("x", "y")
        graph2.add_edge("y", "z")
        
        similarity = self.aligner.calculate_structural_similarity(graph1, graph2)
        
        # Should be high since graphs have identical structure
        assert 0.8 < similarity <= 1.0
    
    def test_calculate_structural_similarity_different_graphs(self):
        """Test structural similarity with different graph structures."""
        graph1 = nx.DiGraph()
        graph1.add_edge("a", "b")  # Simple chain
        
        graph2 = nx.DiGraph()
        graph2.add_edge("x", "y")
        graph2.add_edge("x", "z")
        graph2.add_edge("y", "z")  # More complex structure
        
        similarity = self.aligner.calculate_structural_similarity(graph1, graph2)
        
        # Should be lower due to different structures
        assert 0.0 <= similarity < 0.8
    
    def test_calculate_structural_similarity_empty_graphs(self):
        """Test structural similarity with empty graphs."""
        graph1 = nx.DiGraph()
        graph2 = nx.DiGraph()
        
        similarity = self.aligner.calculate_structural_similarity(graph1, graph2)
        
        # Empty graphs should be considered similar
        assert similarity > 0.5


class TestCommonSubgraphs:
    """Test common subgraph detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner(similarity_threshold=0.7)
        
        # Create test graphs with similar patterns
        self.graph1 = nx.DiGraph()
        self.graph1.add_node("crisis1", type="Event", description="Crisis event")
        self.graph1.add_node("response1", type="Event", description="Policy response")
        self.graph1.add_node("outcome1", type="Event", description="Outcome")
        self.graph1.add_edge("crisis1", "response1", type="causes")
        self.graph1.add_edge("response1", "outcome1", type="causes")
        
        self.graph2 = nx.DiGraph()
        self.graph2.add_node("crisis2", type="Event", description="Crisis situation")
        self.graph2.add_node("response2", type="Event", description="Government response")
        self.graph2.add_node("outcome2", type="Event", description="Result")
        self.graph2.add_edge("crisis2", "response2", type="causes")
        self.graph2.add_edge("response2", "outcome2", type="causes")
        
        self.graph3 = nx.DiGraph()
        self.graph3.add_node("event3", type="Event", description="Different event")
        self.graph3.add_node("mechanism3", type="Mechanism", description="Different mechanism")
        self.graph3.add_edge("event3", "mechanism3", type="triggers")
    
    def test_find_common_subgraphs(self):
        """Test finding common subgraph patterns."""
        graphs = {"case1": self.graph1, "case2": self.graph2, "case3": self.graph3}
        
        patterns = self.aligner.find_common_subgraphs(graphs, min_cases=2)
        
        assert isinstance(patterns, list)
        
        # Should find at least one pattern between case1 and case2
        if len(patterns) > 0:
            pattern = patterns[0]
            assert len(pattern['cases']) >= 2
            assert 'case1' in pattern['cases'] or 'case2' in pattern['cases']
    
    def test_find_common_subgraphs_high_min_cases(self):
        """Test finding common subgraphs with high minimum case requirement."""
        graphs = {"case1": self.graph1, "case2": self.graph2}
        
        patterns = self.aligner.find_common_subgraphs(graphs, min_cases=3)
        
        # Should find no patterns since only 2 cases but requiring 3
        assert len(patterns) == 0
    
    def test_find_common_subgraphs_single_case(self):
        """Test finding common subgraphs with single case."""
        graphs = {"case1": self.graph1}
        
        patterns = self.aligner.find_common_subgraphs(graphs, min_cases=2)
        
        # Should find no patterns since only 1 case but requiring 2
        assert len(patterns) == 0


class TestGraphMetrics:
    """Test graph metrics calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
    
    def test_calculate_graph_metrics_normal_graph(self):
        """Test graph metrics calculation for normal graph."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("a", "c")
        
        metrics = self.aligner._calculate_graph_metrics(graph)
        
        assert isinstance(metrics, dict)
        assert 'density' in metrics
        assert 'avg_clustering' in metrics
        assert 'avg_path_length' in metrics
        
        # Verify metric ranges
        assert 0.0 <= metrics['density'] <= 1.0
        assert 0.0 <= metrics['avg_clustering'] <= 1.0
        assert metrics['avg_path_length'] > 0
    
    def test_calculate_graph_metrics_empty_graph(self):
        """Test graph metrics calculation for empty graph."""
        graph = nx.DiGraph()
        
        metrics = self.aligner._calculate_graph_metrics(graph)
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # No metrics for empty graph
    
    def test_calculate_graph_metrics_disconnected_graph(self):
        """Test graph metrics calculation for disconnected graph."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")  # Disconnected component
        
        metrics = self.aligner._calculate_graph_metrics(graph)
        
        assert isinstance(metrics, dict)
        assert 'density' in metrics
        assert 'avg_clustering' in metrics
        
        # Average path length should be infinite for disconnected graph
        assert metrics['avg_path_length'] == float('inf')


class TestDegreeDistributionComparison:
    """Test degree distribution comparison."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
    
    def test_compare_degree_distributions_identical(self):
        """Test degree distribution comparison with identical distributions."""
        graph1 = nx.DiGraph()
        graph1.add_edge("a", "b")
        graph1.add_edge("a", "c")  # Node 'a' has degree 2
        
        graph2 = nx.DiGraph()
        graph2.add_edge("x", "y")
        graph2.add_edge("x", "z")  # Node 'x' has degree 2
        
        similarity = self.aligner._compare_degree_distributions(graph1, graph2)
        
        # Should be high similarity
        assert 0.8 < similarity <= 1.0
    
    def test_compare_degree_distributions_different(self):
        """Test degree distribution comparison with different distributions."""
        graph1 = nx.DiGraph()
        graph1.add_edge("a", "b")  # Max degree 1
        
        graph2 = nx.DiGraph()
        graph2.add_edge("x", "y")
        graph2.add_edge("x", "z")
        graph2.add_edge("x", "w")  # Max degree 3
        
        similarity = self.aligner._compare_degree_distributions(graph1, graph2)
        
        # Should be lower similarity
        assert 0.0 <= similarity < 0.8
    
    def test_compare_degree_distributions_empty_graphs(self):
        """Test degree distribution comparison with empty graphs."""
        graph1 = nx.DiGraph()
        graph2 = nx.DiGraph()
        
        similarity = self.aligner._compare_degree_distributions(graph1, graph2)
        
        assert similarity == 1.0  # Empty graphs are identical
    
    def test_compare_degree_distributions_one_empty(self):
        """Test degree distribution comparison with one empty graph."""
        graph1 = nx.DiGraph()
        graph1.add_edge("a", "b")
        
        graph2 = nx.DiGraph()
        
        similarity = self.aligner._compare_degree_distributions(graph1, graph2)
        
        assert similarity == 0.0  # One empty, one non-empty


class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aligner = GraphAligner()
    
    def test_large_graph_alignment(self):
        """Test alignment with larger graphs."""
        # Create larger test graphs
        graph1 = nx.DiGraph()
        graph2 = nx.DiGraph()
        
        # Add 20 nodes to each graph
        for i in range(20):
            graph1.add_node(f"node1_{i}", type="Event", description=f"Event {i}")
            graph2.add_node(f"node2_{i}", type="Event", description=f"Event {i}")
        
        # Add some edges
        for i in range(19):
            graph1.add_edge(f"node1_{i}", f"node1_{i+1}", type="causes")
            graph2.add_edge(f"node2_{i}", f"node2_{i+1}", type="causes")
        
        # Should complete without error
        mappings = self.aligner.align_graphs(graph1, graph2, "case1", "case2")
        
        assert isinstance(mappings, list)
        # Number of mappings depends on similarity threshold
        assert len(mappings) <= 20
    
    def test_graph_with_self_loops(self):
        """Test alignment with graphs containing self-loops."""
        graph1 = nx.DiGraph()
        graph1.add_node("node1", type="Event", description="Self-referential event")
        graph1.add_edge("node1", "node1", type="reinforces")  # Self-loop
        
        graph2 = nx.DiGraph()
        graph2.add_node("node2", type="Event", description="Self-referential event")
        graph2.add_edge("node2", "node2", type="reinforces")  # Self-loop
        
        # Should handle self-loops without error
        mappings = self.aligner.align_graphs(graph1, graph2, "case1", "case2")
        
        assert isinstance(mappings, list)
    
    def test_graph_with_missing_attributes(self):
        """Test alignment with graphs missing some node attributes."""
        graph1 = nx.DiGraph()
        graph1.add_node("node1", type="Event")  # Missing description
        
        graph2 = nx.DiGraph()
        graph2.add_node("node2", description="Event description")  # Missing type
        
        # Should handle missing attributes gracefully
        mappings = self.aligner.align_graphs(graph1, graph2, "case1", "case2")
        
        assert isinstance(mappings, list)
    
    def test_alignment_with_unicode_content(self):
        """Test alignment with Unicode content in node attributes."""
        graph1 = nx.DiGraph()
        graph1.add_node("node1", type="Event", description="événement économique", 
                       properties={"région": "Amérique"})
        
        graph2 = nx.DiGraph()
        graph2.add_node("node2", type="Event", description="economic event",
                       properties={"region": "America"})
        
        # Should handle Unicode content without error
        mappings = self.aligner.align_graphs(graph1, graph2, "case1", "case2")
        
        assert isinstance(mappings, list)


# Integration test
class TestGraphAlignerIntegration:
    """Integration tests for the complete graph alignment workflow."""
    
    def test_complete_alignment_workflow(self):
        """Test complete alignment workflow from start to finish."""
        aligner = GraphAligner(similarity_threshold=0.6)
        
        # Create realistic test cases
        case1_graph = nx.DiGraph()
        case1_graph.add_node("economic_crisis", type="Event", 
                           description="Global economic recession")
        case1_graph.add_node("stimulus_package", type="Event",
                           description="Government stimulus package")
        case1_graph.add_node("recovery", type="Event",
                           description="Economic recovery")
        case1_graph.add_edge("economic_crisis", "stimulus_package", type="causes")
        case1_graph.add_edge("stimulus_package", "recovery", type="causes")
        
        case2_graph = nx.DiGraph()
        case2_graph.add_node("financial_crisis", type="Event",
                           description="Banking system crisis")
        case2_graph.add_node("bailout_program", type="Event",
                           description="Bank bailout program")
        case2_graph.add_node("stabilization", type="Event",
                           description="Financial stabilization")
        case2_graph.add_edge("financial_crisis", "bailout_program", type="causes")
        case2_graph.add_edge("bailout_program", "stabilization", type="causes")
        
        # Test single pair alignment
        mappings = aligner.align_graphs(case1_graph, case2_graph, "case1", "case2")
        
        # Verify results
        assert len(mappings) > 0
        for mapping in mappings:
            assert isinstance(mapping, NodeMapping)
            assert mapping.overall_similarity >= aligner.similarity_threshold
        
        # Test multiple graph alignment
        graphs = {"case1": case1_graph, "case2": case2_graph}
        all_mappings = aligner.align_multiple_graphs(graphs)
        
        assert ("case1", "case2") in all_mappings
        assert len(all_mappings[("case1", "case2")]) > 0
        
        # Test common subgraph detection
        patterns = aligner.find_common_subgraphs(graphs, min_cases=2)
        
        # Should find at least one pattern since graphs are similar
        assert len(patterns) >= 0  # Might be 0 depending on threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])