#!/usr/bin/env python3
"""Debug path finding differences"""
import json
import networkx as nx
from core.analyze import load_graph, find_causal_paths_bounded

print("=== DEBUGGING PATH FINDING ===\n")

# Test 1: Load graph exactly as analysis does
print("1. Loading graph via load_graph (same as analysis):")
G, data = load_graph("output_data/demo/revolutions_20250801_000840_graph.json")

# Find triggering and outcome events
event_nodes_data = {n: d for n, d in G.nodes(data=True) if 
                   d.get('type') == 'Event' or 
                   d.get('type') in ['triggering', 'intermediate', 'outcome']}

triggering_events = [n for n, d_node in event_nodes_data.items() if 
                    d_node.get('type') == 'triggering']
outcome_events = [n for n, d_node in event_nodes_data.items() if 
                 d_node.get('type') == 'outcome']

print(f"Triggering events: {triggering_events}")
print(f"Outcome events: {outcome_events}")

# Test 2: Direct NetworkX path finding
print(f"\n2. Testing direct NetworkX path finding:")
for trigger in triggering_events:
    for outcome in outcome_events:
        try:
            # Test basic connectivity
            has_path = nx.has_path(G, trigger, outcome)
            print(f"  nx.has_path({trigger} -> {outcome}): {has_path}")
            
            if has_path:
                # Get shortest path
                shortest = nx.shortest_path(G, trigger, outcome)
                print(f"  Shortest path: {shortest} (length: {len(shortest)})")
                
                # Test simple paths with different cutoffs
                for cutoff in [5, 10, 20]:
                    try:
                        simple_paths = list(nx.all_simple_paths(G, trigger, outcome, cutoff=cutoff))
                        print(f"  nx.all_simple_paths(cutoff={cutoff}): {len(simple_paths)} paths")
                        if simple_paths:
                            print(f"    First path: {simple_paths[0]} (length: {len(simple_paths[0])})")
                    except nx.NetworkXNoPath:
                        print(f"  nx.all_simple_paths(cutoff={cutoff}): No paths")
                
                # Test our bounded function
                bounded_paths = find_causal_paths_bounded(G, trigger, outcome, cutoff=10, max_paths=100)
                print(f"  find_causal_paths_bounded: {len(bounded_paths)} paths")
                if bounded_paths:
                    print(f"    First path: {bounded_paths[0]} (length: {len(bounded_paths[0])})")
                
        except nx.NetworkXNoPath:
            print(f"  nx.has_path({trigger} -> {outcome}): NetworkX reports no path")
        except Exception as e:
            print(f"  Error testing {trigger} -> {outcome}: {e}")

# Test 3: Check edge structure
print(f"\n3. Graph structure analysis:")
print(f"  Total nodes: {G.number_of_nodes()}")
print(f"  Total edges: {G.number_of_edges()}")
print(f"  Graph is strongly connected: {nx.is_strongly_connected(G)}")
print(f"  Graph is weakly connected: {nx.is_weakly_connected(G)}")

# Show first few edges to understand structure
print(f"\n4. Sample edges:")
for i, (u, v, data) in enumerate(list(G.edges(data=True))[:10]):
    edge_type = data.get('type', 'unknown')
    print(f"  {u} -> {v} (type: {edge_type})")