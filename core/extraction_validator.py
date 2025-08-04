#!/usr/bin/env python3
"""
Extraction Validation Module

Validates extracted graphs for causal chain connectivity and completeness
before running process tracing analysis.
"""

import networkx as nx
from typing import Dict, List, Tuple, Any
import json


def validate_causal_connectivity(graph_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that extracted graph has complete causal chains from triggering to outcome events.
    
    Args:
        graph_data: Dictionary with 'nodes' and 'edges' lists
        
    Returns:
        Dictionary with validation results and recommendations
    """
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_data.get('nodes', []):
        G.add_node(node['id'], **node)
    
    # Add edges  
    for edge in graph_data.get('edges', []):
        G.add_edge(edge['source'], edge['target'], **edge)
    
    # Find events by type
    event_nodes = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Event'}
    
    triggering_events = [n for n, d in event_nodes.items() if 
                        d.get('properties', {}).get('type') == 'triggering']
    
    outcome_events = [n for n, d in event_nodes.items() if 
                     d.get('properties', {}).get('type') == 'outcome']
    
    intermediate_events = [n for n, d in event_nodes.items() if 
                          d.get('properties', {}).get('type') == 'intermediate']
    
    # Validation results
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': [],
        'metrics': {
            'total_nodes': len(graph_data.get('nodes', [])),
            'total_edges': len(graph_data.get('edges', [])),
            'event_count': len(event_nodes),
            'triggering_count': len(triggering_events),
            'outcome_count': len(outcome_events),
            'intermediate_count': len(intermediate_events)
        },
        'connectivity_analysis': {}
    }
    
    # Critical validation checks
    
    # Check 1: Must have triggering events
    if not triggering_events:
        validation_results['is_valid'] = False
        validation_results['errors'].append(
            "No triggering events found. Process tracing requires at least one triggering event."
        )
    
    # Check 2: Must have outcome events  
    if not outcome_events:
        validation_results['is_valid'] = False
        validation_results['errors'].append(
            "No outcome events found. Process tracing requires at least one outcome event."
        )
    
    # Check 3: Must have causal connectivity
    connectivity_issues = []
    causal_paths = []
    
    if triggering_events and outcome_events:
        for trigger in triggering_events:
            connected_outcomes = []
            for outcome in outcome_events:
                if nx.has_path(G, trigger, outcome):
                    path = nx.shortest_path(G, trigger, outcome)
                    causal_paths.append({
                        'trigger': trigger,
                        'outcome': outcome, 
                        'path': path,
                        'path_length': len(path)
                    })
                    connected_outcomes.append(outcome)
            
            if not connected_outcomes:
                connectivity_issues.append(f"Triggering event {trigger} has no path to any outcome event")
        
        # Check for disconnected outcome events
        for outcome in outcome_events:
            connected_triggers = [trigger for trigger in triggering_events 
                                if nx.has_path(G, trigger, outcome)]
            if not connected_triggers:
                connectivity_issues.append(f"Outcome event {outcome} has no path from any triggering event")
    
    validation_results['connectivity_analysis'] = {
        'causal_paths': causal_paths,
        'connectivity_issues': connectivity_issues
    }
    
    if connectivity_issues:
        validation_results['is_valid'] = False
        validation_results['errors'].extend(connectivity_issues)
    
    # Check 4: Isolated nodes
    # Note: For isolated nodes, degree() == 0 is correct for both directed and undirected graphs
    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated_nodes:
        validation_results['warnings'].append(
            f"Found {len(isolated_nodes)} isolated nodes with no connections: {isolated_nodes}"
        )
    
    # Check 5: Event connectivity
    unconnected_events = []
    for event_id in event_nodes:
        # Check if event has causal connections (causes edges)
        causal_edges = [e for e in G.edges(data=True) 
                       if (e[0] == event_id or e[1] == event_id) 
                       and e[2].get('type') in ['causes', 'leads_to', 'triggers']]
        if not causal_edges:
            unconnected_events.append(event_id)
    
    if unconnected_events:
        validation_results['warnings'].append(
            f"Found {len(unconnected_events)} events without causal connections: {unconnected_events}"
        )
    
    # Recommendations
    if not validation_results['is_valid']:
        validation_results['recommendations'].extend([
            "Enhance extraction prompt to emphasize complete causal sequences",
            "Add intermediate events to connect triggering and outcome events",
            "Ensure all events are properly connected via causal edges",
            "Validate temporal sequence from start to end of process"
        ])
    
    if causal_paths:
        avg_path_length = sum(p['path_length'] for p in causal_paths) / len(causal_paths)
        validation_results['metrics']['average_causal_path_length'] = avg_path_length
        
        if avg_path_length < 3:
            validation_results['warnings'].append(
                "Causal paths are very short - consider extracting more intermediate events"
            )
    
    return validation_results


def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """Print a formatted validation report"""
    
    print("\n" + "="*60)
    print("CAUSAL GRAPH VALIDATION REPORT")
    print("="*60)
    
    # Status
    status = "VALID" if validation_results['is_valid'] else "INVALID"
    print(f"Status: {status}")
    
    # Metrics
    metrics = validation_results['metrics']
    print(f"\nGraph Metrics:")
    print(f"  Total Nodes: {metrics['total_nodes']}")
    print(f"  Total Edges: {metrics['total_edges']}")
    print(f"  Events: {metrics['event_count']}")
    print(f"  Triggering Events: {metrics['triggering_count']}")
    print(f"  Outcome Events: {metrics['outcome_count']}")
    print(f"  Intermediate Events: {metrics['intermediate_count']}")
    
    # Connectivity
    causal_paths = validation_results['connectivity_analysis']['causal_paths']
    print(f"\nCausal Chain Analysis:")
    print(f"  Complete Causal Paths Found: {len(causal_paths)}")
    
    if causal_paths:
        print("  Causal Paths:")
        for path in causal_paths:
            path_str = " -> ".join(path['path'])
            print(f"    {path['trigger']} -> {path['outcome']}: {path_str}")
    
    # Errors
    if validation_results['errors']:
        print(f"\nERRORS ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"  • {error}")
    
    # Warnings  
    if validation_results['warnings']:
        print(f"\nWARNINGS ({len(validation_results['warnings'])}):")
        for warning in validation_results['warnings']:
            print(f"  • {warning}")
    
    # Recommendations
    if validation_results['recommendations']:
        print(f"\nRECOMMENDATIONS:")
        for rec in validation_results['recommendations']:
            print(f"  • {rec}")
    
    print("="*60)


def validate_extraction_file(graph_json_path: str) -> Dict[str, Any]:
    """
    Validate an extracted graph JSON file
    
    Args:
        graph_json_path: Path to the graph JSON file
        
    Returns:
        Validation results dictionary
    """
    
    try:
        with open(graph_json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        validation_results = validate_causal_connectivity(graph_data)
        return validation_results
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Failed to load or parse graph file: {str(e)}"],
            'warnings': [],
            'recommendations': ["Check file format and JSON structure"],
            'metrics': {},
            'connectivity_analysis': {}
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python extraction_validator.py <graph_json_file>")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    results = validate_extraction_file(graph_file)
    print_validation_report(results)
    
    # Exit with error code if validation failed
    if not results['is_valid']:
        sys.exit(1)