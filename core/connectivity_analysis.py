"""
Connectivity Analysis Module for Process Tracing System

This module provides tools to detect and analyze disconnected entities in process tracing graphs,
identifying isolated nodes and small components that should be connected to the main causal network.
"""

import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json


class DisconnectionDetector:
    """Detects disconnected nodes and components in process tracing graphs."""
    
    def __init__(self):
        self.connection_requirements = {
            'Condition': ['enables', 'constrains'],
            'Actor': ['initiates'],
            'Event': ['causes', 'part_of_mechanism', 'confirms_occurrence'],
            'Evidence': ['supports', 'refutes', 'tests_hypothesis'],
            'Data_Source': ['provides_evidence']
        }
    
    def analyze_graph(self, graph_data: Dict) -> Dict:
        """
        Comprehensive analysis of graph connectivity issues.
        
        Args:
            graph_data: Dictionary with 'nodes' and 'edges' arrays
            
        Returns:
            Dictionary with connectivity analysis results
        """
        # Create NetworkX graph for analysis
        G = self._create_networkx_graph(graph_data)
        
        # Find connectivity issues
        isolated_nodes = self._find_isolated_nodes(G, graph_data)
        small_components = self._find_small_components(G, graph_data)
        connectivity_stats = self._calculate_connectivity_stats(G, graph_data)
        
        # Analyze by node type
        type_analysis = self._analyze_by_node_type(G, graph_data)
        
        # Generate connection suggestions
        suggestions = self._generate_connection_suggestions(isolated_nodes, graph_data)
        
        return {
            'total_nodes': len(graph_data['nodes']),
            'total_edges': len(graph_data['edges']),
            'connected_components': len(list(nx.weakly_connected_components(G))),
            'isolated_nodes': isolated_nodes,
            'small_components': small_components,
            'connectivity_stats': connectivity_stats,
            'type_analysis': type_analysis,
            'connection_suggestions': suggestions
        }
    
    def _create_networkx_graph(self, graph_data: Dict) -> nx.DiGraph:
        """Create NetworkX directed graph from graph data."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node)
        
        # Add edges (handle both 'source'/'target' and 'source_id'/'target_id' formats)
        for edge in graph_data['edges']:
            source = edge.get('source') or edge.get('source_id')
            target = edge.get('target') or edge.get('target_id')
            if source and target:
                G.add_edge(source, target, **edge)
            else:
                print(f"Warning: Edge missing source/target information: {edge}")
        
        return G
    
    def _find_isolated_nodes(self, G: nx.DiGraph, graph_data: Dict) -> List[Dict]:
        """Find nodes with no connections."""
        isolated = []
        
        for node in graph_data['nodes']:
            node_id = node['id']
            if G.degree(node_id) == 0:
                isolated.append({
                    'id': node_id,
                    'type': node['type'],
                    'description': node['properties'].get('description', ''),
                    'expected_connections': self.connection_requirements.get(node['type'], [])
                })
        
        return isolated
    
    def _find_small_components(self, G: nx.DiGraph, graph_data: Dict, max_size: int = 5) -> List[Dict]:
        """Find small disconnected components."""
        components = list(nx.weakly_connected_components(G))
        small_components = []
        
        for i, component in enumerate(components):
            if 1 < len(component) <= max_size:
                component_nodes = []
                for node_id in component:
                    node_data = next(n for n in graph_data['nodes'] if n['id'] == node_id)
                    component_nodes.append({
                        'id': node_id,
                        'type': node_data['type'],
                        'description': node_data['properties'].get('description', '')[:100]
                    })
                
                small_components.append({
                    'component_id': i,
                    'size': len(component),
                    'nodes': component_nodes
                })
        
        return small_components
    
    def _calculate_connectivity_stats(self, G: nx.DiGraph, graph_data: Dict) -> Dict:
        """Calculate overall connectivity statistics."""
        node_types = [node['type'] for node in graph_data['nodes']]
        type_counts = defaultdict(int)
        type_connectivity = defaultdict(lambda: {'in_degree': 0, 'out_degree': 0})
        
        for node_type in node_types:
            type_counts[node_type] += 1
        
        # Calculate connectivity by type
        for edge in graph_data['edges']:
            source_id = edge.get('source') or edge.get('source_id')
            target_id = edge.get('target') or edge.get('target_id')
            if not source_id or not target_id:
                continue
            
            source_node = next(n for n in graph_data['nodes'] if n['id'] == source_id)
            target_node = next(n for n in graph_data['nodes'] if n['id'] == target_id)
            
            type_connectivity[source_node['type']]['out_degree'] += 1
            type_connectivity[target_node['type']]['in_degree'] += 1
        
        # Calculate averages
        type_stats = {}
        for node_type, count in type_counts.items():
            in_deg = type_connectivity[node_type]['in_degree']
            out_deg = type_connectivity[node_type]['out_degree']
            avg_connectivity = (in_deg + out_deg) / count if count > 0 else 0
            
            type_stats[node_type] = {
                'count': count,
                'avg_connectivity': round(avg_connectivity, 2),
                'in_degree': in_deg,
                'out_degree': out_deg
            }
        
        return type_stats
    
    def _analyze_by_node_type(self, G: nx.DiGraph, graph_data: Dict) -> Dict:
        """Analyze disconnection patterns by node type."""
        type_analysis = defaultdict(lambda: {'total': 0, 'disconnected': 0, 'disconnected_nodes': []})
        
        for node in graph_data['nodes']:
            node_type = node['type']
            node_id = node['id']
            
            type_analysis[node_type]['total'] += 1
            
            if G.degree(node_id) == 0:
                type_analysis[node_type]['disconnected'] += 1
                type_analysis[node_type]['disconnected_nodes'].append(node_id)
        
        # Calculate disconnection rates
        for node_type in type_analysis:
            total = type_analysis[node_type]['total']
            disconnected = type_analysis[node_type]['disconnected']
            type_analysis[node_type]['disconnection_rate'] = round((disconnected / total) * 100, 1) if total > 0 else 0
        
        return dict(type_analysis)
    
    def _generate_connection_suggestions(self, isolated_nodes: List[Dict], graph_data: Dict) -> List[Dict]:
        """Generate specific connection suggestions for isolated nodes."""
        suggestions = []
        
        for node in isolated_nodes:
            node_id = node['id']
            node_type = node['type']
            description = node['description'].lower()
            
            # Generate suggestions based on node type and content
            node_suggestions = []
            
            if node_type == 'Condition':
                node_suggestions.extend(self._suggest_condition_connections(node, graph_data))
            elif node_type == 'Actor':
                node_suggestions.extend(self._suggest_actor_connections(node, graph_data))
            elif node_type == 'Event':
                node_suggestions.extend(self._suggest_event_connections(node, graph_data))
            elif node_type == 'Evidence':
                node_suggestions.extend(self._suggest_evidence_connections(node, graph_data))
            
            suggestions.append({
                'node_id': node_id,
                'node_type': node_type,
                'suggestions': node_suggestions
            })
        
        return suggestions
    
    def _suggest_condition_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Suggest connections for Condition nodes."""
        suggestions = []
        description = node['description'].lower()
        
        # Find mechanisms and events that could be enabled/constrained
        for target_node in graph_data['nodes']:
            if target_node['type'] in ['Causal_Mechanism', 'Event', 'Hypothesis']:
                target_desc = target_node['properties'].get('description', '').lower()
                
                # Suggest enabling relationships
                # Use semantic analysis to classify condition type
                from core.semantic_analysis_service import get_semantic_service
                semantic_service = get_semantic_service()
                
                # Assess if condition is enabling based on semantic understanding
                assessment = semantic_service.assess_probative_value(
                    evidence_description=description,
                    hypothesis_description="Condition acts as an enabling factor",
                    context="Classifying contextual conditions"
                )
                
                # Use confidence score to determine if it's enabling
                if assessment.confidence_score > 0.6:
                    suggestions.append({
                        'target_id': target_node['id'],
                        'target_type': target_node['type'],
                        'edge_type': 'enables',
                        'reasoning': f"Condition appears to enable {target_node['type'].lower()}",
                        'confidence': 0.7
                    })
                
                # Suggest constraining relationships
                # Use semantic analysis to classify condition type  
                # Assess if condition is constraining based on semantic understanding
                assessment_constraint = semantic_service.assess_probative_value(
                    evidence_description=description,
                    hypothesis_description="Condition acts as a constraining factor",
                    context="Classifying contextual conditions"
                )
                
                # Use confidence score to determine if it's constraining
                if assessment_constraint.confidence_score > 0.6:
                    if target_node['type'] in ['Event', 'Causal_Mechanism', 'Actor']:
                        suggestions.append({
                            'target_id': target_node['id'],
                            'target_type': target_node['type'],
                            'edge_type': 'constrains',
                            'reasoning': f"Condition appears to constrain {target_node['type'].lower()}",
                            'confidence': 0.7
                        })
        
        return suggestions
    
    def _suggest_actor_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Suggest connections for Actor nodes."""
        suggestions = []
        actor_name = node.get('name', '').lower()
        
        # Find events that this actor might initiate
        for target_node in graph_data['nodes']:
            if target_node['type'] == 'Event':
                event_desc = target_node['properties'].get('description', '').lower()
                
                # Look for actor name in event description
                # Use semantic analysis to determine actor involvement
                from core.semantic_analysis_service import get_semantic_service
                semantic_service = get_semantic_service()
                
                # Assess if actor initiated the event
                assessment = semantic_service.assess_probative_value(
                    evidence_description=event_desc,
                    hypothesis_description=f"Actor {actor_name} initiated or started this event",
                    context="Determining actor involvement in event initiation"
                )
                
                # Use confidence to determine involvement
                if assessment.confidence_score > 0.7:
                    suggestions.append({
                        'target_id': target_node['id'],
                        'target_type': target_node['type'],
                        'edge_type': 'initiates',
                        'reasoning': f"Actor likely initiated this event",
                        'confidence': 0.8
                    })
        
        return suggestions
    
    def _suggest_event_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Suggest connections for Event nodes."""
        suggestions = []
        event_desc = node['description'].lower()
        
        # Look for causal relationships
        for target_node in graph_data['nodes']:
            if target_node['type'] == 'Event':
                target_desc = target_node['properties'].get('description', '').lower()
                
                # Suggest causal connections
                # Use semantic analysis to identify causal relationships
                from core.semantic_analysis_service import get_semantic_service
                semantic_service = get_semantic_service()
                
                assessment = semantic_service.assess_probative_value(
                    evidence_description=event_desc,
                    hypothesis_description="Event represents a causal relationship",
                    context="Identifying causal mechanisms in evidence"
                )
                
                # Use confidence to determine causality
                if assessment.confidence_score > 0.65:
                    suggestions.append({
                        'target_id': target_node['id'],
                        'target_type': target_node['type'],
                        'edge_type': 'causes',
                        'reasoning': f"Event appears to cause other events",
                        'confidence': 0.6
                    })
            
            elif target_node['type'] == 'Causal_Mechanism':
                # Suggest part_of_mechanism relationships
                suggestions.append({
                    'target_id': target_node['id'],
                    'target_type': target_node['type'],
                    'edge_type': 'part_of_mechanism',
                    'reasoning': f"Event could be part of mechanism",
                    'confidence': 0.5
                })
        
        return suggestions
    
    def _suggest_evidence_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Suggest connections for Evidence nodes."""
        suggestions = []
        evidence_desc = node['description'].lower()
        
        # Find hypotheses and events this evidence might support/refute
        for target_node in graph_data['nodes']:
            if target_node['type'] in ['Hypothesis', 'Event', 'Causal_Mechanism']:
                # Suggest support relationships
                suggestions.append({
                    'target_id': target_node['id'],
                    'target_type': target_node['type'],
                    'edge_type': 'supports',
                    'reasoning': f"Evidence may support {target_node['type'].lower()}",
                    'confidence': 0.4
                })
        
        return suggestions


def analyze_connectivity(graph_data: Dict) -> Dict:
    """
    Main function to analyze graph connectivity issues.
    
    Args:
        graph_data: Dictionary with 'nodes' and 'edges' arrays
        
    Returns:
        Comprehensive connectivity analysis report
    """
    detector = DisconnectionDetector()
    return detector.analyze_graph(graph_data)


def print_connectivity_report(analysis: Dict) -> None:
    """Print a formatted connectivity analysis report."""
    print("=== CONNECTIVITY ANALYSIS REPORT ===")
    print(f"Total nodes: {analysis['total_nodes']}")
    print(f"Total edges: {analysis['total_edges']}")
    print(f"Connected components: {analysis['connected_components']} (should be 1)")
    
    print(f"\nIsolated nodes: {len(analysis['isolated_nodes'])}")
    for node in analysis['isolated_nodes']:
        print(f"  - {node['type']}: {node['id']}")
        print(f"    Expected: {node['expected_connections']}")
    
    print(f"\nSmall components: {len(analysis['small_components'])}")
    for comp in analysis['small_components']:
        print(f"  Component {comp['component_id']}: {comp['size']} nodes")
    
    print("\nDisconnection rates by type:")
    for node_type, stats in analysis['type_analysis'].items():
        if stats['disconnected'] > 0:
            print(f"  {node_type}: {stats['disconnected']}/{stats['total']} ({stats['disconnection_rate']}%)")
    
    print("\nConnection suggestions:")
    for suggestion_group in analysis['connection_suggestions']:
        if suggestion_group['suggestions']:
            print(f"\n  {suggestion_group['node_id']} ({suggestion_group['node_type']}):")
            for sugg in suggestion_group['suggestions'][:3]:  # Show top 3
                print(f"    -> {sugg['edge_type']} -> {sugg['target_id']} (confidence: {sugg['confidence']})")