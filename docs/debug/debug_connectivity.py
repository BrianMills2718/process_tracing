#!/usr/bin/env python3
"""Debug graph connectivity and path finding"""
import json
import networkx as nx

def debug_connectivity():
    # Load graph
    with open("output_data/demo/american_revolution_fixed_graph.json", 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes and edges exactly as the analysis does
    for node in data['nodes']:
        G.add_node(node['id'], **node)
    
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], **edge)
    
    print("=== CONNECTIVITY DEBUG ===\n")
    
    # Find triggering and outcome events
    event_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Event'}
    
    triggering_events = [n for n, d_node in event_nodes_data.items() if 
                        d_node.get('attr_props', {}).get('type') == 'triggering']
    outcome_events = [n for n, d_node in event_nodes_data.items() if 
                     d_node.get('attr_props', {}).get('type') == 'outcome']
    
    print(f"Triggering events: {triggering_events}")
    print(f"Outcome events: {outcome_events}")
    
    # Check direct connectivity between triggering and outcome events
    for trigger in triggering_events:
        for outcome in outcome_events:
            print(f"\nChecking path from {trigger} to {outcome}:")
            
            # Check if there are any paths at all
            if nx.has_path(G, trigger, outcome):
                print(f"  SUCCESS: Path exists from {trigger} to {outcome}")
                
                # Find shortest path
                try:
                    shortest = nx.shortest_path(G, trigger, outcome)
                    print(f"  Shortest path: {shortest}")
                    
                    # Check edge types in path
                    print("  Edge analysis:")
                    for i in range(len(shortest) - 1):
                        u, v = shortest[i], shortest[i+1]
                        edge_data = G.get_edge_data(u, v)
                        edge_type = edge_data.get('type', 'unknown') if edge_data else 'no_edge'
                        print(f"    {u} -> {v}: edge_type='{edge_type}'")
                        
                except nx.NetworkXNoPath:
                    print(f"  ERROR: nx.shortest_path failed despite has_path=True")
            else:
                print(f"  ERROR: No path exists from {trigger} to {outcome}")
    
    # Show all edges for debugging
    print(f"\n=== ALL EDGES ===")
    print(f"Total edges: {G.number_of_edges()}")
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'unknown')
        print(f"  {u} -> {v} (type: {edge_type})")

if __name__ == "__main__":
    debug_connectivity()