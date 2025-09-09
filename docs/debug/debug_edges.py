#!/usr/bin/env python3
"""Debug edge structure"""
from core.analyze import load_graph

# Load using the same method as analysis
G, data = load_graph("output_data/demo/revolutions_20250801_000840_graph.json")

print("=== EDGE STRUCTURE DEBUG ===\n")

# Check the first few edges
print("First 5 edges structure:")
for i, (u, v, edge_data) in enumerate(list(G.edges(data=True))[:5]):
    print(f"\nEdge {i+1}: {u} -> {v}")
    print(f"  Raw edge_data keys: {list(edge_data.keys())}")
    print(f"  edge_data.get('type'): {edge_data.get('type')}")
    print(f"  Full edge_data: {edge_data}")

# Look specifically at the E1->E2 edge that's failing
print(f"\n=== SPECIFIC FAILING EDGE: E1->E2 ===")
if G.has_edge('E1', 'E2'):
    edge_data = G.get_edge_data('E1', 'E2')
    print(f"E1->E2 edge_data: {edge_data}")
    print(f"E1->E2 edge_data.get('type'): {edge_data.get('type')}")
    
    # Check if there are nested properties
    if isinstance(edge_data, dict):
        for key, value in edge_data.items():
            print(f"  {key}: {value}")
            if isinstance(value, dict):
                print(f"    (nested dict with keys: {list(value.keys())})")
else:
    print("E1->E2 edge does not exist!")