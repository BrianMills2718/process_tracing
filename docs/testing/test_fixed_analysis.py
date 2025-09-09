#!/usr/bin/env python3
"""Test the fixed analysis without LLM calls"""
import json
import networkx as nx
from pathlib import Path
from core.analyze import perform_process_tracing_analysis

def load_graph(json_path):
    """Load graph from JSON file into NetworkX format"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes with proper structure
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        attr_props = node.get('attr_props', {})
        
        # Add node with flat structure but keep attr_props for compatibility
        G.add_node(node_id, type=node_type, attr_props=attr_props, **attr_props)
    
    # Add edges
    for edge in data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            type=edge['type'],
            properties=edge.get('properties', {}),
            **edge.get('properties', {})
        )
    
    return G

def main():
    # Load the fixed graph
    graph_path = Path("output_data/demo/american_revolution_fixed_graph.json")
    G = load_graph(graph_path)
    
    print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Perform analysis without LLM enhancements
    results = perform_process_tracing_analysis(G, enable_llm=False)
    
    # Print key results
    print("\n=== CAUSAL CHAINS ===")
    if results['causal_chains']:
        for i, chain in enumerate(results['causal_chains'][:3]):
            print(f"\nChain {i+1}:")
            print(f"  Path: {' -> '.join(chain['path'])}")
            print(f"  Edge types: {' -> '.join(chain['edges'])}")
            print(f"  Validity: {chain.get('validity', 'Unknown')}")
    else:
        print("No causal chains found")
    
    print("\n=== EVIDENCE ANALYSIS ===")
    for hyp_id, hyp_data in results['evidence_analysis'].items():
        print(f"\nHypothesis {hyp_id}: {hyp_data['description']}")
        print(f"  Supporting evidence: {len(hyp_data['supporting_evidence'])}")
        print(f"  Refuting evidence: {len(hyp_data['refuting_evidence'])}")
        print(f"  Balance: {hyp_data['balance']:.2f}")
        print(f"  Assessment: {hyp_data['assessment']}")
    
    print("\n=== CAUSAL MECHANISMS ===")
    for mech in results['causal_mechanisms']:
        print(f"\nMechanism {mech['id']}: {mech['name']}")
        print(f"  Completeness: {mech['completeness']}")
        print(f"  Confidence: {mech['confidence']}")
        print(f"  Constituent events: {len(mech.get('causes', []))}")
    
    print("\n=== NETWORK METRICS ===")
    metrics = results['network_metrics']
    print(f"Density: {metrics['density']:.4f}")
    print(f"Node types: {metrics['node_type_distribution']}")
    
    # Save results
    output_path = Path("output_data/demo/test_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()