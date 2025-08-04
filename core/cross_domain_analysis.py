"""
Cross-Domain Analysis Module for Multi-Node-Type Causal Analysis

Extends causal analysis beyond events to include hypothesis-evidence relationships,
mechanism validation paths, and Van Evera integration across node types.
"""

import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


def identify_hypothesis_evidence_chains(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Identify chains that trace evidence through to hypothesis assessment.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        List of hypothesis-evidence chain dictionaries
    """
    he_chains = []
    
    # Find Evidence nodes
    evidence_nodes = [node for node in G.nodes() 
                     if G.nodes[node].get('node_type') == 'Evidence']
    
    # Find Hypothesis nodes
    hypothesis_nodes = [node for node in G.nodes() 
                       if G.nodes[node].get('node_type') == 'Hypothesis']
    
    # Find paths from Evidence to Hypothesis
    for evidence in evidence_nodes:
        for hypothesis in hypothesis_nodes:
            if evidence != hypothesis:
                try:
                    # Issue #18 Fix: Add max paths limit to prevent hangs
                    import itertools
                    paths_iterator = nx.all_simple_paths(G, evidence, hypothesis, cutoff=5)
                    paths = list(itertools.islice(paths_iterator, 100))
                    
                    for path in paths:
                        if len(path) >= 2:
                            chain_data = {
                                'evidence_node': evidence,
                                'hypothesis_node': hypothesis,
                                'path': path,
                                'length': len(path),
                                'start_type': 'Evidence',
                                'end_type': 'Hypothesis',
                                'node_types': [G.nodes[node].get('node_type', 'Unknown') for node in path],
                                'edge_types': [G.edges[path[i], path[i+1]].get('edge_type', 'relates_to') 
                                             for i in range(len(path)-1)]
                            }
                            
                            # Add Van Evera assessment if evidence has it
                            evidence_data = G.nodes[evidence]
                            if 'van_evera_type' in evidence_data:
                                chain_data['van_evera_type'] = evidence_data['van_evera_type']
                                chain_data['van_evera_reasoning'] = evidence_data.get('van_evera_reasoning', '')
                            
                            # Add hypothesis assessment if available
                            hypothesis_data = G.nodes[hypothesis]
                            if 'assessment' in hypothesis_data:
                                chain_data['van_evera_assessment'] = hypothesis_data['assessment']
                            
                            he_chains.append(chain_data)
                            
                except nx.NetworkXNoPath:
                    continue
    
    return he_chains


def identify_cross_domain_paths(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Identify paths that cross between Events, Hypotheses, and Evidence.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        List of cross-domain path dictionaries
    """
    cross_paths = []
    
    # Get nodes by type
    events = [node for node in G.nodes() if G.nodes[node].get('node_type') == 'Event']
    hypotheses = [node for node in G.nodes() if G.nodes[node].get('node_type') == 'Hypothesis']
    evidence = [node for node in G.nodes() if G.nodes[node].get('node_type') == 'Evidence']
    
    all_nodes = events + hypotheses + evidence
    
    # Find paths that cross domains
    for start_node in all_nodes:
        for end_node in all_nodes:
            if start_node != end_node:
                start_type = G.nodes[start_node].get('node_type', 'Unknown')
                end_type = G.nodes[end_node].get('node_type', 'Unknown')
                
                # Only consider paths that cross domains
                if start_type != end_type:
                    try:
                        # Issue #18 Fix: Add max paths limit to prevent hangs
                        import itertools
                        paths_iterator = nx.all_simple_paths(G, start_node, end_node, cutoff=6)
                        paths = list(itertools.islice(paths_iterator, 100))
                        
                        for path in paths:
                            if len(path) >= 2:
                                node_types = [G.nodes[node].get('node_type', 'Unknown') for node in path]
                                
                                # Check if path actually crosses domains
                                unique_types = set(node_types)
                                if len(unique_types) > 1:
                                    path_data = {
                                        'start_node': start_node,
                                        'end_node': end_node,
                                        'path': path,
                                        'length': len(path),
                                        'node_types': node_types,
                                        'unique_types': list(unique_types),
                                        'domain_transitions': count_domain_transitions(node_types),
                                        'edge_types': [G.edges[path[i], path[i+1]].get('edge_type', 'relates_to') 
                                                     for i in range(len(path)-1)]
                                    }
                                    
                                    # Add Van Evera information if path includes evidence
                                    van_evera_info = extract_van_evera_from_path(G, path)
                                    if van_evera_info:
                                        path_data['van_evera_types'] = van_evera_info
                                    
                                    cross_paths.append(path_data)
                                    
                    except nx.NetworkXNoPath:
                        continue
    
    # Remove duplicates and sort by complexity
    unique_paths = []
    path_signatures = set()
    
    for path in cross_paths:
        signature = (tuple(path['path']), tuple(path['node_types']))
        if signature not in path_signatures:
            path_signatures.add(signature)
            unique_paths.append(path)
    
    # Sort by domain transitions and length
    unique_paths.sort(key=lambda x: (x['domain_transitions'], x['length']), reverse=True)
    
    return unique_paths


def identify_mechanism_validation_paths(G: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Identify paths through causal mechanisms (Conditions → Mechanisms → Outcomes).
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        List of mechanism validation path dictionaries
    """
    mechanism_paths = []
    
    # Find mechanism-related nodes
    mechanisms = [node for node in G.nodes() 
                 if G.nodes[node].get('node_type') == 'Causal_Mechanism']
    conditions = [node for node in G.nodes() 
                 if G.nodes[node].get('node_type') == 'Condition']
    
    # If no explicit mechanism nodes, look for events that might be mechanisms
    if not mechanisms:
        mechanisms = [node for node in G.nodes() 
                     if 'mechanism' in G.nodes[node].get('description', '').lower()]
    
    # If no explicit conditions, use events as potential conditions
    if not conditions:
        conditions = [node for node in G.nodes() 
                     if G.nodes[node].get('node_type') == 'Event' and
                     G.out_degree(node) > 0]  # Events that cause other things
    
    # Find paths through mechanisms
    for condition in conditions:
        for mechanism in mechanisms:
            if condition != mechanism:
                try:
                    # Issue #18 Fix: Path from condition to mechanism with limits
                    import itertools
                    path_iterator = nx.all_simple_paths(G, condition, mechanism, cutoff=3)
                    condition_to_mechanism = list(itertools.islice(path_iterator, 50))
                    
                    for path_to_mech in condition_to_mechanism:
                        # Find outcomes from this mechanism
                        outcomes = [node for node in G.nodes() 
                                  if node != mechanism and nx.has_path(G, mechanism, node)]
                        
                        for outcome in outcomes:
                            try:
                                # Issue #18 Fix: Mechanism to outcome paths with limits
                                import itertools
                                path_iterator = nx.all_simple_paths(G, mechanism, outcome, cutoff=3)
                                mechanism_to_outcome = list(itertools.islice(path_iterator, 50))
                                
                                for path_from_mech in mechanism_to_outcome:
                                    # Combine paths
                                    full_path = path_to_mech + path_from_mech[1:]  # Avoid duplicate mechanism node
                                    
                                    if len(full_path) >= 3:  # At least condition → mechanism → outcome
                                        node_types = [G.nodes[node].get('node_type', 'Unknown') for node in full_path]
                                        
                                        mechanism_data = {
                                            'condition': condition,
                                            'mechanism': mechanism,
                                            'outcome': outcome,
                                            'path': full_path,
                                            'length': len(full_path),
                                            'node_types': node_types,
                                            'mechanism_position': path_to_mech.index(mechanism) if mechanism in path_to_mech else -1,
                                            'edge_types': [G.edges[full_path[i], full_path[i+1]].get('edge_type', 'relates_to') 
                                                         for i in range(len(full_path)-1)]
                                        }
                                        
                                        # Calculate mechanism completeness
                                        mechanism_data['mechanism_completeness'] = calculate_mechanism_completeness(G, full_path, mechanism)
                                        
                                        mechanism_paths.append(mechanism_data)
                                        
                            except nx.NetworkXNoPath:
                                continue
                                
                except nx.NetworkXNoPath:
                    continue
    
    return mechanism_paths


def integrate_van_evera_across_domains(G: nx.DiGraph, cross_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Integrate Van Evera analysis across different node types in cross-domain paths.
    
    Args:
        G: NetworkX directed graph
        cross_paths: List of cross-domain paths
        
    Returns:
        Dictionary with Van Evera integration analysis
    """
    van_evera_integration = {
        'evidence_hypothesis_links': [],
        'event_evidence_chains': [],
        'hypothesis_validation_paths': [],
        'van_evera_type_distribution': defaultdict(int),
        'cross_domain_assessments': []
    }
    
    for path_data in cross_paths:
        path = path_data['path']
        node_types = path_data['node_types']
        
        # Analyze Evidence → Hypothesis links
        if 'Evidence' in node_types and 'Hypothesis' in node_types:
            evidence_nodes = [node for node in path if G.nodes[node].get('node_type') == 'Evidence']
            hypothesis_nodes = [node for node in path if G.nodes[node].get('node_type') == 'Hypothesis']
            
            for evidence in evidence_nodes:
                evidence_data = G.nodes[evidence]
                van_evera_type = evidence_data.get('van_evera_type')
                
                if van_evera_type:
                    van_evera_integration['van_evera_type_distribution'][van_evera_type] += 1
                    
                    for hypothesis in hypothesis_nodes:
                        hypothesis_data = G.nodes[hypothesis]
                        
                        link_data = {
                            'evidence_node': evidence,
                            'hypothesis_node': hypothesis,
                            'van_evera_type': van_evera_type,
                            'van_evera_reasoning': evidence_data.get('van_evera_reasoning', ''),
                            'hypothesis_assessment': hypothesis_data.get('assessment', ''),
                            'path_through': path
                        }
                        
                        van_evera_integration['evidence_hypothesis_links'].append(link_data)
        
        # Analyze Event → Evidence chains
        if 'Event' in node_types and 'Evidence' in node_types:
            event_evidence_chain = {
                'path': path,
                'events': [node for node in path if G.nodes[node].get('node_type') == 'Event'],
                'evidence': [node for node in path if G.nodes[node].get('node_type') == 'Evidence'],
                'van_evera_types': [G.nodes[node].get('van_evera_type') for node in path 
                                  if G.nodes[node].get('node_type') == 'Evidence' and 
                                  G.nodes[node].get('van_evera_type')]
            }
            
            if event_evidence_chain['van_evera_types']:
                van_evera_integration['event_evidence_chains'].append(event_evidence_chain)
        
        # Analyze Hypothesis validation paths
        if 'Hypothesis' in node_types:
            hypothesis_nodes = [node for node in path if G.nodes[node].get('node_type') == 'Hypothesis']
            
            for hypothesis in hypothesis_nodes:
                hypothesis_data = G.nodes[hypothesis]
                
                validation_path = {
                    'hypothesis_node': hypothesis,
                    'hypothesis_text': hypothesis_data.get('description', ''),
                    'validation_path': path,
                    'supporting_evidence': [node for node in path if G.nodes[node].get('node_type') == 'Evidence'],
                    'related_events': [node for node in path if G.nodes[node].get('node_type') == 'Event'],
                    'assessment': hypothesis_data.get('assessment', '')
                }
                
                van_evera_integration['hypothesis_validation_paths'].append(validation_path)
    
    # Generate cross-domain assessments
    for link in van_evera_integration['evidence_hypothesis_links']:
        assessment = generate_cross_domain_assessment(link)
        van_evera_integration['cross_domain_assessments'].append(assessment)
    
    return van_evera_integration


def count_domain_transitions(node_types: List[str]) -> int:
    """Count the number of domain transitions in a path."""
    transitions = 0
    for i in range(1, len(node_types)):
        if node_types[i] != node_types[i-1]:
            transitions += 1
    return transitions


def extract_van_evera_from_path(G: nx.DiGraph, path: List[str]) -> List[Dict[str, Any]]:
    """Extract Van Evera information from nodes in a path."""
    van_evera_info = []
    
    for node in path:
        node_data = G.nodes[node]
        if 'van_evera_type' in node_data:
            van_evera_info.append({
                'node': node,
                'node_type': node_data.get('node_type', 'Unknown'),
                'van_evera_type': node_data['van_evera_type'],
                'van_evera_reasoning': node_data.get('van_evera_reasoning', ''),
                'probative_value': node_data.get('probative_value', {})
            })
    
    return van_evera_info


def calculate_mechanism_completeness(G: nx.DiGraph, path: List[str], mechanism_node: str) -> float:
    """
    Calculate how complete a mechanism validation path is.
    
    Args:
        G: NetworkX directed graph
        path: Full mechanism path
        mechanism_node: The mechanism node in the path
        
    Returns:
        Float representing completeness (0.0 to 1.0)
    """
    completeness_factors = []
    
    # Factor 1: Path includes clear condition
    mechanism_index = path.index(mechanism_node) if mechanism_node in path else -1
    if mechanism_index > 0:
        completeness_factors.append(0.3)  # Has preceding condition
    
    # Factor 2: Path includes clear outcome
    if mechanism_index >= 0 and mechanism_index < len(path) - 1:
        completeness_factors.append(0.3)  # Has following outcome
    
    # Factor 3: Mechanism node has rich description
    mechanism_data = G.nodes[mechanism_node]
    description = mechanism_data.get('description', '')
    if len(description) > 50:  # Rich description
        completeness_factors.append(0.2)
    
    # Factor 4: Clear causal relationships
    if mechanism_index > 0 and mechanism_index < len(path) - 1:
        edge_to_mechanism = G.edges.get((path[mechanism_index-1], mechanism_node), {})
        edge_from_mechanism = G.edges.get((mechanism_node, path[mechanism_index+1]), {})
        
        if (edge_to_mechanism.get('edge_type') in ['causes', 'enables', 'leads_to'] and
            edge_from_mechanism.get('edge_type') in ['causes', 'enables', 'leads_to']):
            completeness_factors.append(0.2)
    
    return sum(completeness_factors)


def generate_cross_domain_assessment(link_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate assessment for cross-domain evidence-hypothesis links.
    
    Args:
        link_data: Evidence-hypothesis link data
        
    Returns:
        Dictionary with cross-domain assessment
    """
    van_evera_type = link_data['van_evera_type']
    hypothesis_assessment = link_data.get('hypothesis_assessment', '')
    
    # Map Van Evera types to assessment strengths
    assessment_mapping = {
        'hoop': 'Necessary but not sufficient - passes basic validation',
        'smoking_gun': 'Strongly confirms hypothesis - compelling evidence',
        'straw_in_the_wind': 'Weakly supports hypothesis - suggestive but not conclusive',
        'doubly_decisive': 'Both necessary and sufficient - definitive validation'
    }
    
    assessment = {
        'evidence_node': link_data['evidence_node'],
        'hypothesis_node': link_data['hypothesis_node'],
        'van_evera_type': van_evera_type,
        'assessment_strength': assessment_mapping.get(van_evera_type, 'Unknown evidence type'),
        'cross_domain_reasoning': f"Evidence ({van_evera_type}) → Hypothesis: {assessment_mapping.get(van_evera_type, 'Unknown')}",
        'hypothesis_status': hypothesis_assessment,
        'validation_path': link_data['path_through']
    }
    
    return assessment


def analyze_cross_domain_patterns(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Comprehensive cross-domain pattern analysis.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        Dictionary with complete cross-domain analysis
    """
    # Run all cross-domain analyses
    he_chains = identify_hypothesis_evidence_chains(G)
    cross_paths = identify_cross_domain_paths(G)
    mechanism_paths = identify_mechanism_validation_paths(G)
    van_evera_integration = integrate_van_evera_across_domains(G, cross_paths)
    
    # Compile comprehensive results
    results = {
        'hypothesis_evidence_chains': he_chains,
        'cross_domain_paths': cross_paths[:20],  # Top 20 most complex paths
        'mechanism_validation_paths': mechanism_paths,
        'van_evera_integration': van_evera_integration,
        'cross_domain_statistics': {
            'total_he_chains': len(he_chains),
            'total_cross_paths': len(cross_paths),
            'total_mechanism_paths': len(mechanism_paths),
            'domain_coverage': calculate_domain_coverage(G),
            'van_evera_coverage': len(van_evera_integration['evidence_hypothesis_links']),
            'most_common_van_evera_type': get_most_common_van_evera_type(van_evera_integration)
        }
    }
    
    return results


def calculate_domain_coverage(G: nx.DiGraph) -> Dict[str, int]:
    """Calculate how many nodes of each type are in the graph."""
    coverage = defaultdict(int)
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'Unknown')
        coverage[node_type] += 1
    return dict(coverage)


def get_most_common_van_evera_type(van_evera_integration: Dict[str, Any]) -> str:
    """Get the most common Van Evera type in the analysis."""
    distribution = van_evera_integration['van_evera_type_distribution']
    if distribution:
        return max(distribution.items(), key=lambda x: x[1])[0]
    return 'None'