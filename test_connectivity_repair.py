#!/usr/bin/env python3
"""
Test script for connectivity repair functionality.

This script demonstrates how to detect and repair disconnected entities
in process tracing graphs using the new connectivity analysis tools.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.connectivity_analysis import analyze_connectivity, print_connectivity_report
from core.disconnection_repair import repair_graph_connectivity, save_repaired_graph


def main():
    """Main function to test connectivity repair on the revolutions dataset."""
    
    # Path to the latest extraction with disconnection issues
    graph_file = "output_data/revolutions/revolutions_20250804_203847_graph.json"
    
    if not os.path.exists(graph_file):
        print(f"Error: Graph file not found: {graph_file}")
        print("Please run the extraction first to generate the graph data.")
        return
    
    print("=== TESTING CONNECTIVITY REPAIR SYSTEM ===")
    print(f"Loading graph from: {graph_file}\n")
    
    # Load the original graph data
    with open(graph_file, 'r', encoding='utf-8') as f:
        original_graph = json.load(f)
    
    print("ORIGINAL GRAPH ANALYSIS:")
    print("-" * 50)
    
    # Analyze original connectivity
    original_analysis = analyze_connectivity(original_graph)
    print_connectivity_report(original_analysis)
    
    print("\n" + "=" * 60)
    print("ATTEMPTING AUTOMATED REPAIR...")
    print("=" * 60)
    
    # Attempt to repair connectivity
    repaired_graph = repair_graph_connectivity(original_graph)
    
    print("\nREPAIRED GRAPH ANALYSIS:")
    print("-" * 50)
    
    # Analyze repaired connectivity
    repaired_analysis = analyze_connectivity(repaired_graph)
    print_connectivity_report(repaired_analysis)
    
    # Show what was added
    original_edge_count = len(original_graph['edges'])
    repaired_edge_count = len(repaired_graph['edges'])
    added_edges = repaired_edge_count - original_edge_count
    
    print(f"\nREPAIR SUMMARY:")
    print(f"Original edges: {original_edge_count}")
    print(f"Repaired edges: {repaired_edge_count}")
    print(f"Added edges: {added_edges}")
    
    improvement = original_analysis['connected_components'] - repaired_analysis['connected_components']
    print(f"Components reduced: {original_analysis['connected_components']} -> {repaired_analysis['connected_components']} (-{improvement})")
    
    # Show the new edges that were added
    if added_edges > 0:
        print(f"\nNEW EDGES ADDED:")
        new_edges = repaired_graph['edges'][original_edge_count:]
        for edge in new_edges:
            reasoning = edge['properties'].get('reasoning', 'No reasoning provided')
            print(f"  {edge['source']} --{edge['type']}--> {edge['target']}")
            print(f"    Reasoning: {reasoning}")
    
    # Save the repaired graph
    if added_edges > 0:
        repaired_path = save_repaired_graph(repaired_graph, graph_file)
        print(f"\nRepaired graph saved for further analysis.")
    else:
        print(f"\nNo edges were added - graph may require manual review.")
    
    # Success assessment
    print(f"\n{'='*60}")
    if repaired_analysis['connected_components'] == 1:
        print("SUCCESS: Graph is now fully connected!")
    elif improvement > 0:
        print(f"PARTIAL SUCCESS: Reduced disconnection but {repaired_analysis['connected_components']} components remain")
    else:
        print("LIMITED SUCCESS: Automated repair could not resolve disconnections")
        print("   Manual review or enhanced inference rules may be needed.")
    
    return repaired_analysis['connected_components'] == 1


def demonstrate_specific_repairs():
    """Demonstrate specific repair strategies for common disconnection patterns."""
    
    print("\n" + "=" * 60)
    print("DISCONNECTION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Common patterns observed in the revolutions dataset
    patterns = [
        {
            'type': 'Condition',
            'issue': '75% disconnection rate',
            'cause': 'Missing enables/constrains relationships',
            'solution': 'Semantic pattern matching for enabling/constraining language'
        },
        {
            'type': 'Actor', 
            'issue': 'Actors without initiated events',
            'cause': 'Actor names not linked to event descriptions',
            'solution': 'Name-based matching and initiation keyword detection'
        },
        {
            'type': 'Event',
            'issue': 'Events not connected to mechanisms',
            'cause': 'Missing part_of_mechanism relationships',
            'solution': 'Thematic content matching between events and mechanisms'
        },
        {
            'type': 'Small Components',
            'issue': 'Alternative explanations isolated',
            'cause': 'Missing evidence connections',
            'solution': 'Cross-component relationship inference'
        }
    ]
    
    for pattern in patterns:
        print(f"\nPATTERN: {pattern['type']}")
        print(f"  Issue: {pattern['issue']}")
        print(f"  Cause: {pattern['cause']}") 
        print(f"  Solution: {pattern['solution']}")
    
    print(f"\nThese patterns are addressed by the ConnectionInferenceEngine")
    print(f"using domain-specific semantic matching and inference rules.")


if __name__ == "__main__":
    try:
        success = main()
        demonstrate_specific_repairs()
        
        if success:
            print(f"\nConnectivity repair test completed successfully!")
            sys.exit(0)
        else:
            print(f"\nConnectivity repair test completed with remaining issues.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during connectivity repair test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)