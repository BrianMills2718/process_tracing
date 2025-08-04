"""
Test suite for Cross-Domain Analysis module.

Tests multi-node-type causal analysis including hypothesis-evidence chains,
cross-domain paths, mechanism validation, and Van Evera integration.
"""

import pytest
import networkx as nx
from core.cross_domain_analysis import (
    identify_hypothesis_evidence_chains,
    identify_cross_domain_paths,
    identify_mechanism_validation_paths,
    integrate_van_evera_across_domains,
    analyze_cross_domain_patterns,
    count_domain_transitions,
    calculate_mechanism_completeness
)


def create_mixed_node_graph():
    """Create a graph with Events, Hypotheses, and Evidence nodes."""
    G = nx.DiGraph()
    
    # Add nodes with different types
    nodes = [
        ('event1', {'node_type': 'Event', 'description': 'Initial event'}),
        ('evidence1', {'node_type': 'Evidence', 'description': 'Supporting evidence', 
                      'van_evera_type': 'smoking_gun', 'van_evera_reasoning': 'Compelling evidence'}),
        ('hypothesis1', {'node_type': 'Hypothesis', 'description': 'Main hypothesis', 
                        'assessment': 'Strongly Confirmed'}),
        ('event2', {'node_type': 'Event', 'description': 'Outcome event'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges creating cross-domain paths
    edges = [
        ('event1', 'evidence1', {'edge_type': 'provides_evidence', 'weight': 0.8}),
        ('evidence1', 'hypothesis1', {'edge_type': 'supports', 'weight': 0.9}),
        ('hypothesis1', 'event2', {'edge_type': 'predicts', 'weight': 0.7}),
    ]
    G.add_edges_from(edges)
    
    return G


def create_mechanism_graph():
    """Create a graph with mechanism validation paths."""
    G = nx.DiGraph()
    
    # Add nodes including mechanism
    nodes = [
        ('condition1', {'node_type': 'Condition', 'description': 'Initial condition'}),
        ('mechanism1', {'node_type': 'Causal_Mechanism', 'description': 'Key causal mechanism'}),
        ('outcome1', {'node_type': 'Event', 'description': 'Final outcome'}),
        ('event1', {'node_type': 'Event', 'description': 'Trigger event'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges creating mechanism path
    edges = [
        ('condition1', 'mechanism1', {'edge_type': 'enables', 'weight': 0.8}),
        ('mechanism1', 'outcome1', {'edge_type': 'causes', 'weight': 0.9}),
        ('event1', 'condition1', {'edge_type': 'creates', 'weight': 0.7}),
    ]
    G.add_edges_from(edges)
    
    return G


def create_complex_mixed_graph():
    """Create a complex graph with multiple node types and relationships."""
    G = nx.DiGraph()
    
    # Add diverse nodes
    nodes = [
        ('event1', {'node_type': 'Event', 'description': 'French and Indian War'}),
        ('event2', {'node_type': 'Event', 'description': 'British taxation'}),
        ('evidence1', {'node_type': 'Evidence', 'description': 'Stamp Act documents', 
                      'van_evera_type': 'hoop', 'van_evera_reasoning': 'Necessary condition'}),
        ('evidence2', {'node_type': 'Evidence', 'description': 'Colonial resistance', 
                      'van_evera_type': 'straw_in_the_wind', 'van_evera_reasoning': 'Suggestive evidence'}),
        ('hypothesis1', {'node_type': 'Hypothesis', 'description': 'Economic grievances caused revolution', 
                        'assessment': 'Conditionally Supported'}),
        ('hypothesis2', {'node_type': 'Hypothesis', 'description': 'Political rights motivated colonists', 
                        'assessment': 'Weakly Supported'}),
        ('mechanism1', {'node_type': 'Causal_Mechanism', 'description': 'Popular mobilization process'}),
        ('outcome1', {'node_type': 'Event', 'description': 'American Revolution'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add complex relationships
    edges = [
        ('event1', 'event2', {'edge_type': 'leads_to', 'weight': 0.8}),
        ('event2', 'evidence1', {'edge_type': 'provides_evidence', 'weight': 0.9}),
        ('evidence1', 'hypothesis1', {'edge_type': 'supports', 'weight': 0.8}),
        ('event2', 'evidence2', {'edge_type': 'provides_evidence', 'weight': 0.6}),
        ('evidence2', 'hypothesis2', {'edge_type': 'suggests', 'weight': 0.5}),
        ('hypothesis1', 'mechanism1', {'edge_type': 'activates', 'weight': 0.7}),
        ('hypothesis2', 'mechanism1', {'edge_type': 'contributes_to', 'weight': 0.6}),
        ('mechanism1', 'outcome1', {'edge_type': 'causes', 'weight': 0.9}),
    ]
    G.add_edges_from(edges)
    
    return G


class TestHypothesisEvidenceChains:
    """Test hypothesis-evidence chain identification."""
    
    def test_basic_evidence_hypothesis_chain(self):
        """Test identification of basic evidence → hypothesis chains."""
        G = create_mixed_node_graph()
        he_chains = identify_hypothesis_evidence_chains(G)
        
        assert len(he_chains) > 0
        
        # Should find evidence1 → hypothesis1 chain
        chain = next((c for c in he_chains if c['evidence_node'] == 'evidence1' and 
                     c['hypothesis_node'] == 'hypothesis1'), None)
        assert chain is not None
        assert chain['start_type'] == 'Evidence'
        assert chain['end_type'] == 'Hypothesis'
        assert 'van_evera_type' in chain
        assert chain['van_evera_type'] == 'smoking_gun'
    
    def test_van_evera_assessment_integration(self):
        """Test Van Evera assessment integration in chains."""
        G = create_mixed_node_graph()
        he_chains = identify_hypothesis_evidence_chains(G)
        
        assert len(he_chains) > 0
        
        chain = he_chains[0]
        assert 'van_evera_assessment' in chain
        assert chain['van_evera_assessment'] == 'Strongly Confirmed'
    
    def test_no_evidence_hypothesis_chains(self):
        """Test behavior when no evidence-hypothesis chains exist."""
        G = nx.DiGraph()
        G.add_nodes_from([
            ('event1', {'node_type': 'Event'}),
            ('event2', {'node_type': 'Event'}),
        ])
        G.add_edge('event1', 'event2')
        
        he_chains = identify_hypothesis_evidence_chains(G)
        assert len(he_chains) == 0


class TestCrossDomainPaths:
    """Test cross-domain path identification."""
    
    def test_event_evidence_hypothesis_path(self):
        """Test identification of Event → Evidence → Hypothesis paths."""
        G = create_mixed_node_graph()
        cross_paths = identify_cross_domain_paths(G)
        
        assert len(cross_paths) > 0
        
        # Should find paths crossing multiple domains
        multi_domain_paths = [p for p in cross_paths if len(p['unique_types']) > 1]
        assert len(multi_domain_paths) > 0
        
        # Check for Event → Evidence → Hypothesis path
        event_to_hypothesis = next((p for p in cross_paths 
                                  if 'Event' in p['node_types'] and 'Hypothesis' in p['node_types']), None)
        assert event_to_hypothesis is not None
        assert event_to_hypothesis['domain_transitions'] > 0
    
    def test_van_evera_types_in_paths(self):
        """Test Van Evera type extraction in cross-domain paths."""
        G = create_mixed_node_graph()
        cross_paths = identify_cross_domain_paths(G)
        
        # Should find paths with Van Evera information
        van_evera_paths = [p for p in cross_paths if 'van_evera_types' in p and p['van_evera_types']]
        assert len(van_evera_paths) > 0
        
        path = van_evera_paths[0]
        assert len(path['van_evera_types']) > 0
        assert path['van_evera_types'][0]['van_evera_type'] == 'smoking_gun'
    
    def test_domain_transition_counting(self):
        """Test domain transition counting."""
        node_types = ['Event', 'Event', 'Evidence', 'Hypothesis', 'Hypothesis']
        transitions = count_domain_transitions(node_types)
        assert transitions == 2  # Event→Evidence, Evidence→Hypothesis
        
        single_domain = ['Event', 'Event', 'Event']
        transitions = count_domain_transitions(single_domain)
        assert transitions == 0


class TestMechanismValidation:
    """Test mechanism validation path identification."""
    
    def test_condition_mechanism_outcome_path(self):
        """Test identification of Condition → Mechanism → Outcome paths."""
        G = create_mechanism_graph()
        mechanism_paths = identify_mechanism_validation_paths(G)
        
        assert len(mechanism_paths) > 0
        
        # Should find path through mechanism
        path = mechanism_paths[0]
        assert path['mechanism'] == 'mechanism1'
        assert len(path['path']) >= 3  # At least condition → mechanism → outcome
        assert 'mechanism_completeness' in path
        assert path['mechanism_completeness'] > 0.0
    
    def test_mechanism_completeness_calculation(self):
        """Test mechanism completeness calculation."""
        G = create_mechanism_graph()
        path = ['condition1', 'mechanism1', 'outcome1']
        
        completeness = calculate_mechanism_completeness(G, path, 'mechanism1')
        
        assert 0.0 <= completeness <= 1.0
        assert completeness > 0.0  # Should have some completeness
    
    def test_no_explicit_mechanisms(self):
        """Test behavior when no explicit mechanism nodes exist."""
        G = nx.DiGraph()
        G.add_nodes_from([
            ('event1', {'node_type': 'Event', 'description': 'Event with mechanism in description'}),
            ('event2', {'node_type': 'Event', 'description': 'Regular event'}),
        ])
        G.add_edge('event1', 'event2')
        
        mechanism_paths = identify_mechanism_validation_paths(G)
        # Should still work with events as potential mechanisms
        assert isinstance(mechanism_paths, list)


class TestVanEveraIntegration:
    """Test Van Evera integration across domains."""
    
    def test_evidence_hypothesis_links(self):
        """Test evidence-hypothesis link analysis."""
        G = create_complex_mixed_graph()
        cross_paths = identify_cross_domain_paths(G)
        integration = integrate_van_evera_across_domains(G, cross_paths)
        
        assert 'evidence_hypothesis_links' in integration
        assert len(integration['evidence_hypothesis_links']) > 0
        
        link = integration['evidence_hypothesis_links'][0]
        assert 'van_evera_type' in link
        assert 'hypothesis_assessment' in link
        assert link['van_evera_type'] in ['hoop', 'smoking_gun', 'straw_in_the_wind']
    
    def test_van_evera_type_distribution(self):
        """Test Van Evera type distribution calculation."""
        G = create_complex_mixed_graph()
        cross_paths = identify_cross_domain_paths(G)
        integration = integrate_van_evera_across_domains(G, cross_paths)
        
        assert 'van_evera_type_distribution' in integration
        distribution = integration['van_evera_type_distribution']
        
        assert isinstance(distribution, dict)
        # Should have counts for different Van Evera types
        total_types = sum(distribution.values())
        assert total_types > 0
    
    def test_cross_domain_assessments(self):
        """Test cross-domain assessment generation."""
        G = create_complex_mixed_graph()
        cross_paths = identify_cross_domain_paths(G)
        integration = integrate_van_evera_across_domains(G, cross_paths)
        
        assert 'cross_domain_assessments' in integration
        assessments = integration['cross_domain_assessments']
        
        if assessments:  # If there are assessments
            assessment = assessments[0]
            assert 'assessment_strength' in assessment
            assert 'cross_domain_reasoning' in assessment
            assert 'van_evera_type' in assessment


class TestComplexPatterns:
    """Test comprehensive cross-domain pattern analysis."""
    
    def test_comprehensive_analysis(self):
        """Test complete cross-domain pattern analysis."""
        G = create_complex_mixed_graph()
        results = analyze_cross_domain_patterns(G)
        
        # Should contain all analysis components
        assert 'hypothesis_evidence_chains' in results
        assert 'cross_domain_paths' in results
        assert 'mechanism_validation_paths' in results
        assert 'van_evera_integration' in results
        assert 'cross_domain_statistics' in results
        
        # Statistics should be reasonable
        stats = results['cross_domain_statistics']
        assert stats['total_he_chains'] >= 0
        assert stats['total_cross_paths'] >= 0
        assert stats['total_mechanism_paths'] >= 0
        assert isinstance(stats['domain_coverage'], dict)
    
    def test_domain_coverage_calculation(self):
        """Test domain coverage calculation."""
        G = create_complex_mixed_graph()
        results = analyze_cross_domain_patterns(G)
        
        coverage = results['cross_domain_statistics']['domain_coverage']
        
        # Should have counts for different node types
        assert coverage['Event'] > 0
        assert coverage['Evidence'] > 0
        assert coverage['Hypothesis'] > 0
    
    def test_van_evera_coverage(self):
        """Test Van Evera coverage statistics."""
        G = create_complex_mixed_graph()
        results = analyze_cross_domain_patterns(G)
        
        stats = results['cross_domain_statistics']
        assert 'van_evera_coverage' in stats
        assert stats['van_evera_coverage'] >= 0
        
        if stats['van_evera_coverage'] > 0:
            assert 'most_common_van_evera_type' in stats
            assert stats['most_common_van_evera_type'] in ['hoop', 'smoking_gun', 'straw_in_the_wind', 'doubly_decisive']


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_graph(self):
        """Test behavior with empty graph."""
        G = nx.DiGraph()
        
        he_chains = identify_hypothesis_evidence_chains(G)
        cross_paths = identify_cross_domain_paths(G)
        mechanism_paths = identify_mechanism_validation_paths(G)
        
        assert len(he_chains) == 0
        assert len(cross_paths) == 0
        assert len(mechanism_paths) == 0
    
    def test_single_domain_graph(self):
        """Test behavior with single domain (all same node type)."""
        G = nx.DiGraph()
        G.add_nodes_from([
            ('event1', {'node_type': 'Event'}),
            ('event2', {'node_type': 'Event'}),
            ('event3', {'node_type': 'Event'}),
        ])
        G.add_edges_from([('event1', 'event2'), ('event2', 'event3')])
        
        cross_paths = identify_cross_domain_paths(G)
        
        # Should have no cross-domain paths since all nodes are same type
        assert len(cross_paths) == 0
    
    def test_disconnected_domains(self):
        """Test behavior with disconnected domain clusters."""
        G = nx.DiGraph()
        
        # Add two disconnected clusters
        G.add_nodes_from([
            ('event1', {'node_type': 'Event'}),
            ('event2', {'node_type': 'Event'}),
            ('evidence1', {'node_type': 'Evidence'}),
            ('hypothesis1', {'node_type': 'Hypothesis'}),
        ])
        G.add_edges_from([
            ('event1', 'event2'),  # Event cluster
            ('evidence1', 'hypothesis1'),  # Evidence-Hypothesis cluster
        ])
        
        cross_paths = identify_cross_domain_paths(G)
        
        # Should find paths within connected components that cross domains
        # but not between disconnected components
        assert len(cross_paths) > 0  # Evidence → Hypothesis path exists
        
        # Verify no paths cross between the disconnected components
        event_to_evidence_paths = [p for p in cross_paths 
                                 if p['start_node'] in ['event1', 'event2'] and 
                                 p['end_node'] in ['evidence1', 'hypothesis1']]
        assert len(event_to_evidence_paths) == 0
    
    def test_missing_node_attributes(self):
        """Test behavior with nodes missing type attributes."""
        G = nx.DiGraph()
        G.add_nodes_from([
            ('node1', {}),  # No node_type
            ('node2', {'node_type': 'Event'}),
        ])
        G.add_edge('node1', 'node2')
        
        cross_paths = identify_cross_domain_paths(G)
        
        # Should handle missing attributes gracefully
        assert isinstance(cross_paths, list)


if __name__ == '__main__':
    pytest.main([__file__])