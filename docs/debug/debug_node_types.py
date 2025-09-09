#!/usr/bin/env python3
"""Debug node types after flattening"""
import json
from core.analyze import load_graph

# Load using the same method as analysis
G, data = load_graph("output_data/demo/revolutions_20250801_000840_graph.json")

print("=== NODE TYPES AFTER LOAD_GRAPH ===\n")

# Check types of first 10 nodes
for i, (node_id, node_data) in enumerate(list(G.nodes(data=True))[:10]):
    print(f"Node {node_id}:")
    print(f"  type: {node_data.get('type')}")
    print(f"  description: {node_data.get('description', 'N/A')}")
    print()

# Count node types
type_counts = {}
for node_id, node_data in G.nodes(data=True):
    node_type = node_data.get('type', 'unknown')
    type_counts[node_type] = type_counts.get(node_type, 0) + 1

print("Node type distribution:")
for node_type, count in sorted(type_counts.items()):
    print(f"  {node_type}: {count}")