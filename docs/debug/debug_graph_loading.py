#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.analyze import load_graph
import json

def debug_graph_loading():
    """Debug what happens during graph loading"""
    
    graph_file = "test_focused_output/focused_test_20250801_124416_graph.json"
    
    print("=== RAW JSON DATA ===")
    with open(graph_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Raw nodes: {len(raw_data['nodes'])}")
    for node in raw_data['nodes']:
        if node['type'] == 'Hypothesis':
            print(f"Raw Hypothesis node: {node}")
    
    print("\n=== LOADED NETWORKX GRAPH ===")
    G, data = load_graph(graph_file)
    
    print(f"Loaded nodes: {len(G.nodes())}")
    print(f"Node data structure:")
    
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('type') == 'Hypothesis':
            print(f"NetworkX Hypothesis {node_id}: {node_data}")
        
    # Check what the filter finds
    hypothesis_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Hypothesis'}
    print(f"\nFiltered Hypothesis nodes: {len(hypothesis_nodes_data)}")
    for hyp_id, hyp_data in hypothesis_nodes_data.items():
        print(f"  {hyp_id}: {hyp_data}")

if __name__ == "__main__":
    debug_graph_loading()