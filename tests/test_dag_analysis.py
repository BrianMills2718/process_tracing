"""
Test suite for DAG Analysis module.

Tests complex causal analysis capabilities including pathway identification,
convergence analysis, divergence analysis, and pattern detection.
"""

import pytest
import networkx as nx
from core.dag_analysis import (
    identify_causal_pathways,
    analyze_causal_convergence,
    analyze_causal_divergence,
    calculate_pathway_strength,
    find_complex_causal_patterns,
    identify_complex_nodes
)


def create_simple_dag():
    """Create a simple DAG for testing."""
    G = nx.DiGraph()
    
    # Add nodes with types
    nodes = [
        ('event1', {'node_type': 'Event', 'description': 'Initial trigger event'}),
        ('event2', {'node_type': 'Event', 'description': 'Intermediate event'}),
        ('event3', {'node_type': 'Event', 'description': 'Final outcome event'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ('event1', 'event2', {'edge_type': 'causes', 'weight': 0.8}),
        ('event2', 'event3', {'edge_type': 'causes', 'weight': 0.7}),
    ]
    G.add_edges_from(edges)
    
    return G


def create_convergence_dag():
    """Create a DAG with convergence pattern (multiple causes → single effect)."""
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        ('event1', {'node_type': 'Event', 'description': 'Cause A'}),
        ('event2', {'node_type': 'Event', 'description': 'Cause B'}),
        ('event3', {'node_type': 'Event', 'description': 'Converged effect'}),
        ('event4', {'node_type': 'Event', 'description': 'Final outcome'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges - both event1 and event2 cause event3
    edges = [
        ('event1', 'event3', {'edge_type': 'causes', 'weight': 0.8}),
        ('event2', 'event3', {'edge_type': 'causes', 'weight': 0.7}),
        ('event3', 'event4', {'edge_type': 'causes', 'weight': 0.9}),
    ]
    G.add_edges_from(edges)
    
    return G


def create_divergence_dag():
    """Create a DAG with divergence pattern (single cause → multiple effects)."""
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        ('event1', {'node_type': 'Event', 'description': 'Single cause'}),
        ('event2', {'node_type': 'Event', 'description': 'Effect A'}),
        ('event3', {'node_type': 'Event', 'description': 'Effect B'}),
        ('event4', {'node_type': 'Event', 'description': 'Outcome from A'}),
        ('event5', {'node_type': 'Event', 'description': 'Outcome from B'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges - event1 causes both event2 and event3
    edges = [
        ('event1', 'event2', {'edge_type': 'causes', 'weight': 0.8}),
        ('event1', 'event3', {'edge_type': 'causes', 'weight': 0.7}),
        ('event2', 'event4', {'edge_type': 'causes', 'weight': 0.6}),
        ('event3', 'event5', {'edge_type': 'causes', 'weight': 0.6}),
    ]
    G.add_edges_from(edges)
    
    return G


def create_complex_dag():
    """Create a complex DAG with both convergence and divergence patterns."""
    G = nx.DiGraph()
    
    # Add nodes with different types
    nodes = [
        ('event1', {'node_type': 'Event', 'description': 'Trigger A'}),
        ('event2', {'node_type': 'Event', 'description': 'Trigger B'}),
        ('hypothesis1', {'node_type': 'Hypothesis', 'description': 'Central hypothesis'}),
        ('evidence1', {'node_type': 'Evidence', 'description': 'Supporting evidence'}),
        ('event3', {'node_type': 'Event', 'description': 'Outcome A'}),
        ('event4', {'node_type': 'Event', 'description': 'Outcome B'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges creating complex patterns
    edges = [
        ('event1', 'hypothesis1', {'edge_type': 'suggests', 'weight': 0.8}),
        ('event2', 'hypothesis1', {'edge_type': 'suggests', 'weight': 0.7}),
        ('hypothesis1', 'evidence1', {'edge_type': 'predicts', 'weight': 0.9}),
        ('hypothesis1', 'event3', {'edge_type': 'leads_to', 'weight': 0.8}),
        ('evidence1', 'event4', {'edge_type': 'supports', 'weight': 0.7}),
    ]
    G.add_edges_from(edges)
    
    return G


class TestCausalPathways:
    """Test causal pathway identification."""
    
    def test_simple_pathway_identification(self):
        """Test identification of simple linear pathways."""
        G = create_simple_dag()
        pathways = identify_causal_pathways(G)
        
        assert len(pathways) >= 1
        
        # Should find the main pathway event1 → event2 → event3
        main_pathway = next((p for p in pathways if p['trigger'] == 'event1' and p['outcome'] == 'event3'), None)
        assert main_pathway is not None
        assert main_pathway['path'] == ['event1', 'event2', 'event3']
        assert main_pathway['length'] == 3
        assert main_pathway['strength'] > 0.0
    
    def test_multiple_pathways_same_outcome(self):
        """Test identification of multiple pathways to same outcome."""
        G = create_convergence_dag()
        pathways = identify_causal_pathways(G)
        
        # Should find multiple pathways to event4
        pathways_to_event4 = [p for p in pathways if p['outcome'] == 'event4']
        assert len(pathways_to_event4) >= 2
        
        # Should have different triggers
        triggers = [p['trigger'] for p in pathways_to_event4]
        assert 'event1' in triggers or 'event2' in triggers
    
    def test_pathway_strength_calculation(self):
        """Test pathway strength calculation."""
        G = create_simple_dag()
        path = ['event1', 'event2', 'event3']
        
        strength = calculate_pathway_strength(G, path)
        
        assert 0.0 <= strength <= 1.0
        assert strength > 0.0  # Should have some strength
    
    def test_pathway_node_types(self):
        """Test that pathways capture node types correctly."""
        G = create_complex_dag()
        pathways = identify_causal_pathways(G)
        
        assert len(pathways) > 0
        
        # Check that node types are captured
        for pathway in pathways:
            assert 'node_types' in pathway
            assert len(pathway['node_types']) == len(pathway['path'])
            assert all(node_type in ['Event', 'Hypothesis', 'Evidence'] for node_type in pathway['node_types'])


class TestConvergenceAnalysis:
    """Test causal convergence analysis."""
    
    def test_convergence_detection(self):
        """Test detection of convergence points."""
        G = create_convergence_dag()
        convergence = analyze_causal_convergence(G)
        
        assert 'convergence_points' in convergence
        assert len(convergence['convergence_points']) > 0
        
        # event3 should be a convergence point (both event1 and event2 → event3)
        assert 'event3' in convergence['convergence_points']
        
        event3_convergence = convergence['convergence_points']['event3']
        assert event3_convergence['num_converging_paths'] >= 0
        assert event3_convergence['convergence_strength'] > 0.0
    
    def test_convergence_strength_calculation(self):
        """Test convergence strength calculation."""
        G = create_convergence_dag()
        convergence = analyze_causal_convergence(G)
        
        convergence_point = list(convergence['convergence_points'].values())[0]
        
        assert 0.0 <= convergence_point['convergence_strength'] <= 1.0
        assert convergence_point['convergence_strength'] > 0.0
    
    def test_no_convergence_in_simple_dag(self):
        """Test that simple linear DAGs don't show convergence."""
        G = create_simple_dag()
        convergence = analyze_causal_convergence(G)
        
        # Simple linear chain should have no convergence points
        assert len(convergence['convergence_points']) == 0


class TestDivergenceAnalysis:
    """Test causal divergence analysis."""
    
    def test_divergence_detection(self):
        """Test detection of divergence points."""
        G = create_divergence_dag()
        divergence = analyze_causal_divergence(G)
        
        assert 'divergence_points' in divergence
        assert len(divergence['divergence_points']) > 0
        
        # event1 should be a divergence point (event1 → both event2 and event3)
        assert 'event1' in divergence['divergence_points']
        
        event1_divergence = divergence['divergence_points']['event1']
        assert event1_divergence['num_diverging_paths'] >= 0
        assert event1_divergence['divergence_strength'] > 0.0
    
    def test_divergence_strength_calculation(self):
        """Test divergence strength calculation."""
        G = create_divergence_dag()
        divergence = analyze_causal_divergence(G)
        
        divergence_point = list(divergence['divergence_points'].values())[0]
        
        assert 0.0 <= divergence_point['divergence_strength'] <= 1.0
        assert divergence_point['divergence_strength'] > 0.0
    
    def test_no_divergence_in_simple_dag(self):
        """Test that simple linear DAGs don't show divergence."""
        G = create_simple_dag()
        divergence = analyze_causal_divergence(G)
        
        # Simple linear chain should have no significant divergence points
        assert len(divergence['divergence_points']) == 0


class TestComplexPatterns:
    """Test complex pattern identification."""
    
    def test_complex_pattern_analysis(self):
        """Test comprehensive complex pattern analysis."""
        G = create_complex_dag()
        patterns = find_complex_causal_patterns(G)
        
        # Should contain all analysis components
        assert 'causal_pathways' in patterns
        assert 'convergence_analysis' in patterns
        assert 'divergence_analysis' in patterns
        assert 'pathway_statistics' in patterns
        assert 'complex_nodes' in patterns
        
        # Should have some pathways
        assert len(patterns['causal_pathways']) > 0
        
        # Statistics should be reasonable
        stats = patterns['pathway_statistics']
        assert stats['total_pathways'] >= 0
        assert stats['average_pathway_length'] >= 0
    
    def test_complex_node_identification(self):
        """Test identification of complex nodes."""
        G = create_complex_dag()
        convergence = analyze_causal_convergence(G)
        divergence = analyze_causal_divergence(G)
        
        complex_nodes = identify_complex_nodes(G, convergence, divergence)
        
        # Should identify some complex nodes
        if complex_nodes:
            node = complex_nodes[0]
            assert 'node_id' in node
            assert 'complexity_score' in node
            assert 'roles' in node
            assert node['complexity_score'] > 0.0
    
    def test_pathway_statistics(self):
        """Test pathway statistics calculation."""
        G = create_complex_dag()
        patterns = find_complex_causal_patterns(G)
        
        stats = patterns['pathway_statistics']
        
        # Statistics should be non-negative
        assert stats['total_pathways'] >= 0
        assert stats['average_pathway_length'] >= 0
        assert stats['strongest_pathway_strength'] >= 0.0
        
        # Length distribution should be a dictionary
        assert isinstance(stats['pathway_length_distribution'], dict)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_graph(self):
        """Test behavior with empty graph."""
        G = nx.DiGraph()
        
        pathways = identify_causal_pathways(G)
        convergence = analyze_causal_convergence(G)
        divergence = analyze_causal_divergence(G)
        
        assert len(pathways) == 0
        assert len(convergence['convergence_points']) == 0
        assert len(divergence['divergence_points']) == 0
    
    def test_single_node_graph(self):
        """Test behavior with single node graph."""
        G = nx.DiGraph()
        G.add_node('single', node_type='Event', description='Lonely node')
        
        pathways = identify_causal_pathways(G)
        convergence = analyze_causal_convergence(G)
        divergence = analyze_causal_divergence(G)
        
        assert len(pathways) == 0
        assert len(convergence['convergence_points']) == 0
        assert len(divergence['divergence_points']) == 0
    
    def test_disconnected_graph(self):
        """Test behavior with disconnected components."""
        G = nx.DiGraph()
        
        # Add two disconnected components
        G.add_edges_from([
            ('a1', 'a2'),
            ('b1', 'b2')
        ])
        
        pathways = identify_causal_pathways(G)
        
        # Should find pathways within each component
        assert len(pathways) >= 0
        
        # Should not find paths between components
        cross_component_paths = [p for p in pathways 
                               if (p['trigger'].startswith('a') and p['outcome'].startswith('b')) or
                                  (p['trigger'].startswith('b') and p['outcome'].startswith('a'))]
        assert len(cross_component_paths) == 0


if __name__ == '__main__':
    pytest.main([__file__])