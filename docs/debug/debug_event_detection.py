#!/usr/bin/env python3
"""Debug event detection in the analysis"""
import json
import networkx as nx

def debug_event_detection():
    # Load graph
    with open("output_data/demo/american_revolution_fixed_graph.json", 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes exactly as the analysis does
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    
    print("=== EVENT DETECTION DEBUG ===\n")
    
    # Check how nodes are stored in NetworkX
    print("1. Sample node data in NetworkX:")
    for node_id in list(G.nodes())[:3]:
        node_data = G.nodes[node_id]
        print(f"Node {node_id}:")
        print(f"  Raw data keys: {list(node_data.keys())}")
        print(f"  type: {node_data.get('type')}")
        print(f"  Has attr_props: {'attr_props' in node_data}")
        if 'attr_props' in node_data:
            print(f"  attr_props.type: {node_data['attr_props'].get('type')}")
        print()
    
    # Test event detection logic exactly as in analyze.py
    event_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Event'}
    print(f"2. Found {len(event_nodes_data)} Event nodes")
    
    # Test triggering event detection
    triggering_events = [n for n, d_node in event_nodes_data.items() if 
                        d_node.get('subtype') == 'triggering' or 
                        d_node.get('type') == 'triggering' or
                        d_node.get('attr_props', {}).get('type') == 'triggering']
    
    outcome_events = [n for n, d_node in event_nodes_data.items() if 
                     d_node.get('subtype') == 'outcome' or 
                     d_node.get('type') == 'outcome' or
                     d_node.get('attr_props', {}).get('type') == 'outcome']
    
    print(f"3. Triggering events found: {triggering_events}")
    print(f"4. Outcome events found: {outcome_events}")
    
    # Debug each Event node
    print("\n5. Detailed Event node analysis:")
    for node_id, node_data in event_nodes_data.items():
        attr_props = node_data.get('attr_props', {})
        event_type = attr_props.get('type', 'unknown')
        print(f"  {node_id}: {event_type} - {attr_props.get('description', 'No description')[:50]}...")

if __name__ == "__main__":
    debug_event_detection()