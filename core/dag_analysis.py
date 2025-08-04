"""
DAG Analysis Module for Complex Causal Analysis

Provides advanced causal analysis capabilities beyond linear chains,
supporting Directed Acyclic Graphs (DAGs) with convergence and divergence analysis.
"""

import networkx as nx
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict


def identify_causal_pathways(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Identify all causal pathways in the graph from trigger to outcome nodes.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        List of pathway dictionaries with trigger, outcome, path, and strength
    """
    pathways = []
    
    # Identify potential trigger nodes (nodes with no incoming edges or high out-degree)
    potential_triggers = [
        node for node in G.nodes() 
        if G.in_degree(node) == 0 or (G.out_degree(node) > 1 and G.in_degree(node) <= 1)
    ]
    
    # Identify potential outcome nodes (nodes with no outgoing edges or high in-degree)
    potential_outcomes = [
        node for node in G.nodes()
        if G.out_degree(node) == 0 or (G.in_degree(node) > 1 and G.out_degree(node) <= 1)
    ]
    
    # Find all paths between triggers and outcomes
    for trigger in potential_triggers:
        for outcome in potential_outcomes:
            if trigger != outcome:
                try:
                    # Issue #18 Fix: Find simple paths with limits to prevent hangs
                    import itertools
                    paths_iterator = nx.all_simple_paths(G, trigger, outcome, cutoff=10)
                    paths = list(itertools.islice(paths_iterator, 100))
                    
                    for path in paths:
                        if len(path) >= 2:  # At least trigger -> outcome
                            pathway_strength = calculate_pathway_strength(G, path)
                            
                            pathways.append({
                                'trigger': trigger,
                                'outcome': outcome,
                                'path': path,
                                'length': len(path),
                                'strength': pathway_strength,
                                'node_types': [G.nodes[node].get('node_type', 'Unknown') for node in path],
                                'edge_types': [G.edges[path[i], path[i+1]].get('edge_type', 'causes') 
                                             for i in range(len(path)-1)]
                            })
                except nx.NetworkXNoPath:
                    continue
    
    # Sort by pathway strength
    pathways.sort(key=lambda x: x['strength'], reverse=True)
    
    return pathways


def analyze_causal_convergence(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze nodes where multiple causal pathways converge (multiple causes → single effect).
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary with convergence analysis results
    """
    convergence_points = {}
    
    # Find nodes with multiple incoming edges
    for node in G.nodes():
        incoming_edges = list(G.predecessors(node))
        
        if len(incoming_edges) > 1:
            # Analyze the convergence
            incoming_paths = []
            
            # First, add direct connections as simple paths
            for predecessor in incoming_edges:
                direct_path = [predecessor, node]
                incoming_paths.append({
                    'path': direct_path,
                    'length': len(direct_path),
                    'strength': calculate_pathway_strength(G, direct_path)
                })
            
            # Then, find longer paths leading to each predecessor
            for predecessor in incoming_edges:
                # Find paths leading to this predecessor
                upstream_nodes = [n for n in G.nodes() 
                                if n != node and n != predecessor and nx.has_path(G, n, predecessor)]
                
                for upstream in upstream_nodes:
                    try:
                        # Issue #18 Fix: Add max paths limit to prevent hangs
                        import itertools
                        paths_iterator = nx.all_simple_paths(G, upstream, predecessor, cutoff=5)
                        paths = list(itertools.islice(paths_iterator, 100))
                        for path in paths:
                            full_path = path + [node]
                            incoming_paths.append({
                                'path': full_path,
                                'length': len(full_path),
                                'strength': calculate_pathway_strength(G, full_path)
                            })
                    except nx.NetworkXNoPath:
                        continue
            
            if incoming_paths:
                convergence_strength = sum(path['strength'] for path in incoming_paths) / len(incoming_paths)
                
                convergence_points[node] = {
                    'node_type': G.nodes[node].get('node_type', 'Unknown'),
                    'description': G.nodes[node].get('description', ''),
                    'incoming_paths': incoming_paths,
                    'convergence_strength': convergence_strength,
                    'num_converging_paths': len(incoming_paths),
                    'direct_predecessors': incoming_edges
                }
    
    return {
        'convergence_points': convergence_points,
        'total_convergence_points': len(convergence_points),
        'strongest_convergence': max(convergence_points.items(), 
                                   key=lambda x: x[1]['convergence_strength'], 
                                   default=(None, {'convergence_strength': 0}))
    }


def analyze_causal_divergence(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze nodes where single causes branch to multiple effects (single cause → multiple effects).
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary with divergence analysis results
    """
    divergence_points = {}
    
    # Find nodes with multiple outgoing edges
    for node in G.nodes():
        outgoing_edges = list(G.successors(node))
        
        if len(outgoing_edges) > 1:
            # Analyze the divergence
            outgoing_paths = []
            
            for successor in outgoing_edges:
                # Find paths extending from this successor
                downstream_nodes = [n for n in G.nodes() 
                                  if n != node and n != successor and nx.has_path(G, successor, n)]
                
                for downstream in downstream_nodes:
                    try:
                        # Issue #18 Fix: Add max paths limit to prevent hangs
                        import itertools
                        paths_iterator = nx.all_simple_paths(G, successor, downstream, cutoff=5)
                        paths = list(itertools.islice(paths_iterator, 100))
                        for path in paths:
                            full_path = [node] + path
                            outgoing_paths.append({
                                'path': full_path,
                                'length': len(full_path),
                                'strength': calculate_pathway_strength(G, full_path)
                            })
                    except nx.NetworkXNoPath:
                        continue
            
            if outgoing_paths:
                divergence_strength = sum(path['strength'] for path in outgoing_paths) / len(outgoing_paths)
                
                divergence_points[node] = {
                    'node_type': G.nodes[node].get('node_type', 'Unknown'),
                    'description': G.nodes[node].get('description', ''),
                    'outgoing_paths': outgoing_paths,
                    'divergence_strength': divergence_strength,
                    'num_diverging_paths': len(outgoing_paths),
                    'direct_successors': outgoing_edges
                }
    
    return {
        'divergence_points': divergence_points,
        'total_divergence_points': len(divergence_points),
        'strongest_divergence': max(divergence_points.items(),
                                  key=lambda x: x[1]['divergence_strength'],
                                  default=(None, {'divergence_strength': 0}))
    }


def calculate_pathway_strength(G: nx.DiGraph, path: List[str]) -> float:
    """
    Calculate the strength/significance of a causal pathway.
    
    Args:
        G: NetworkX directed graph
        path: List of node IDs representing the pathway
        
    Returns:
        Float representing pathway strength (0.0 to 1.0)
    """
    if len(path) < 2:
        return 0.0
    
    strength_factors = []
    
    # Factor 1: Node importance (based on centrality)
    centrality = nx.betweenness_centrality(G)
    node_importance = sum(centrality.get(node, 0) for node in path) / len(path)
    strength_factors.append(node_importance)
    
    # Factor 2: Edge strength (based on edge attributes)
    edge_strength = 0.0
    for i in range(len(path) - 1):
        edge_data = G.edges.get((path[i], path[i+1]), {})
        edge_weight = edge_data.get('weight', 0.5)  # Default moderate strength
        edge_strength += edge_weight
    edge_strength = edge_strength / (len(path) - 1) if len(path) > 1 else 0.0
    strength_factors.append(edge_strength)
    
    # Factor 3: Path length (shorter paths generally stronger, but with diminishing returns)
    length_factor = 1.0 / (1.0 + 0.1 * (len(path) - 2))  # Penalty for longer paths
    strength_factors.append(length_factor)
    
    # Factor 4: Node type diversity (more diverse = potentially more interesting)
    node_types = [G.nodes[node].get('node_type', 'Unknown') for node in path]
    unique_types = len(set(node_types))
    diversity_factor = min(unique_types / len(path), 1.0)
    strength_factors.append(diversity_factor)
    
    # Combine factors with weights
    weights = [0.3, 0.3, 0.2, 0.2]  # Adjust as needed
    overall_strength = sum(factor * weight for factor, weight in zip(strength_factors, weights))
    
    return min(max(overall_strength, 0.0), 1.0)  # Clamp to [0, 1]


def find_complex_causal_patterns(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Find complex causal patterns combining multiple analytical approaches.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary with comprehensive pattern analysis
    """
    # Get all analyses
    pathways = identify_causal_pathways(G)
    convergence = analyze_causal_convergence(G)
    divergence = analyze_causal_divergence(G)
    
    # Identify complex patterns
    complex_patterns = {
        'causal_pathways': pathways[:10],  # Top 10 strongest pathways
        'convergence_analysis': convergence,
        'divergence_analysis': divergence,
        'pathway_statistics': {
            'total_pathways': len(pathways),
            'average_pathway_length': sum(p['length'] for p in pathways) / len(pathways) if pathways else 0,
            'strongest_pathway_strength': pathways[0]['strength'] if pathways else 0,
            'pathway_length_distribution': get_length_distribution(pathways)
        },
        'complex_nodes': identify_complex_nodes(G, convergence, divergence)
    }
    
    return complex_patterns


def get_length_distribution(pathways: List[Dict[str, Any]]) -> Dict[int, int]:
    """Get distribution of pathway lengths."""
    distribution = defaultdict(int)
    for pathway in pathways:
        distribution[pathway['length']] += 1
    return dict(distribution)


def identify_complex_nodes(G: nx.DiGraph, convergence: Dict[str, Any], 
                          divergence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify nodes that play complex roles in the causal network.
    
    Args:
        G: NetworkX directed graph
        convergence: Convergence analysis results
        divergence: Divergence analysis results
        
    Returns:
        List of complex nodes with their characteristics
    """
    complex_nodes = []
    centrality = nx.betweenness_centrality(G)
    
    for node in G.nodes():
        node_data = G.nodes[node]
        complexity_score = 0.0
        roles = []
        
        # Check if it's a convergence point
        if node in convergence['convergence_points']:
            complexity_score += convergence['convergence_points'][node]['convergence_strength']
            roles.append('convergence_point')
        
        # Check if it's a divergence point
        if node in divergence['divergence_points']:
            complexity_score += divergence['divergence_points'][node]['divergence_strength']
            roles.append('divergence_point')
        
        # Check centrality
        node_centrality = centrality.get(node, 0.0)
        if node_centrality > 0.1:  # High centrality threshold
            complexity_score += node_centrality
            roles.append('central_node')
        
        # Check degree (highly connected)
        total_degree = G.in_degree(node) + G.out_degree(node)
        if total_degree > 3:  # High connectivity threshold
            complexity_score += 0.1
            roles.append('highly_connected')
        
        if complexity_score > 0.2:  # Complexity threshold
            complex_nodes.append({
                'node_id': node,
                'node_type': node_data.get('node_type', 'Unknown'),
                'description': node_data.get('description', ''),
                'complexity_score': complexity_score,
                'roles': roles,
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
                'centrality': node_centrality
            })
    
    # Sort by complexity score
    complex_nodes.sort(key=lambda x: x['complexity_score'], reverse=True)
    
    return complex_nodes