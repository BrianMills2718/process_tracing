#!/usr/bin/env python3
"""Debug the event detection in analyze.py"""
import json
import networkx as nx

# Load the graph
with open("output_data/demo/revolutions_20250801_000840_graph.json", 'r') as f:
    graph_data = json.load(f)

G = nx.DiGraph()

# Add nodes exactly as the analysis does
for node in graph_data['nodes']:
    G.add_node(node['id'], **node)

# Add edges
for edge in graph_data['edges']:
    G.add_edge(edge['source'], edge['target'], **edge)

print("=== DEBUGGING EVENT DETECTION IN ANALYZE.PY ===\n")

# Replicate exact logic from analyze.py
event_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Event'}
print(f"Found {len(event_nodes_data)} Event nodes")

# Check the first few nodes to see their structure
print("\nFirst 3 Event node structures:")
for i, (node_id, node_data) in enumerate(list(event_nodes_data.items())[:3]):
    print(f"\nNode {node_id}:")
    print(f"  Full data keys: {list(node_data.keys())}")
    print(f"  node_data.get('type'): {node_data.get('type')}")
    print(f"  node_data.get('subtype'): {node_data.get('subtype')}")
    print(f"  node_data.get('attr_props'): {node_data.get('attr_props')}")
    print(f"  node_data.get('properties'): {node_data.get('properties')}")
    
    if 'properties' in node_data:
        props = node_data['properties']
        print(f"  properties.get('type'): {props.get('type')}")
        print(f"  properties.get('description'): {props.get('description')}")

# Test the exact triggering event detection logic from analyze.py
print("\n=== TRIGGERING EVENT DETECTION ===")
triggering_events = [n for n, d_node in event_nodes_data.items() if 
                    d_node.get('subtype') == 'triggering' or 
                    d_node.get('type') == 'triggering' or
                    d_node.get('attr_props', {}).get('type') == 'triggering' or
                    d_node.get('properties', {}).get('type') == 'triggering']

print(f"Triggering events found: {triggering_events}")

# Test each condition separately
print("\nTesting each condition separately:")
for i, condition in enumerate(['subtype', 'type', 'attr_props.type', 'properties.type']):
    if condition == 'subtype':
        events = [n for n, d in event_nodes_data.items() if d.get('subtype') == 'triggering']
    elif condition == 'type':
        events = [n for n, d in event_nodes_data.items() if d.get('type') == 'triggering']
    elif condition == 'attr_props.type':
        events = [n for n, d in event_nodes_data.items() if d.get('attr_props', {}).get('type') == 'triggering']
    elif condition == 'properties.type':
        events = [n for n, d in event_nodes_data.items() if d.get('properties', {}).get('type') == 'triggering']
    
    print(f"  Condition {condition}: {events}")

# Test outcome event detection
print("\n=== OUTCOME EVENT DETECTION ===")
outcome_events = [n for n, d_node in event_nodes_data.items() if 
                 d_node.get('subtype') == 'outcome' or 
                 d_node.get('type') == 'outcome' or
                 d_node.get('attr_props', {}).get('type') == 'outcome' or
                 d_node.get('properties', {}).get('type') == 'outcome']

print(f"Outcome events found: {outcome_events}")

# Test each condition separately for outcomes
print("\nTesting each condition separately for outcomes:")
for i, condition in enumerate(['subtype', 'type', 'attr_props.type', 'properties.type']):
    if condition == 'subtype':
        events = [n for n, d in event_nodes_data.items() if d.get('subtype') == 'outcome']
    elif condition == 'type':
        events = [n for n, d in event_nodes_data.items() if d.get('type') == 'outcome']
    elif condition == 'attr_props.type':
        events = [n for n, d in event_nodes_data.items() if d.get('attr_props', {}).get('type') == 'outcome']
    elif condition == 'properties.type':
        events = [n for n, d in event_nodes_data.items() if d.get('properties', {}).get('type') == 'outcome']
    
    print(f"  Condition {condition}: {events}")