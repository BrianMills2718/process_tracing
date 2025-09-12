#!/usr/bin/env python3
"""Manually test the fixed data structures"""
import json
import networkx as nx
from pathlib import Path

def test_fixed_graph():
    # Load the fixed graph
    with open("output_data/demo/american_revolution_fixed_graph.json", 'r') as f:
        data = json.load(f)
    
    print("=== TESTING FIXED GRAPH STRUCTURE ===\n")
    
    # Test 1: Check node structure
    print("1. NODE STRUCTURE TEST:")
    for node in data['nodes'][:3]:
        print(f"Node {node['id']}:")
        print(f"  Type: {node['type']}")
        print(f"  Has attr_props: {'attr_props' in node}")
        if 'attr_props' in node:
            print(f"  attr_props.type: {node['attr_props'].get('type', 'N/A')}")
            print(f"  Description: {node['attr_props'].get('description', 'N/A')[:50]}...")
    
    # Test 2: Find triggering and outcome events
    print("\n2. EVENT TYPE TEST:")
    triggering_events = []
    outcome_events = []
    for node in data['nodes']:
        if node['type'] == 'Event' and 'attr_props' in node:
            event_type = node['attr_props'].get('type')
            if event_type == 'triggering':
                triggering_events.append(node['id'])
            elif event_type == 'outcome':
                outcome_events.append(node['id'])
    
    print(f"Triggering events found: {triggering_events}")
    print(f"Outcome events found: {outcome_events}")
    
    # Test 3: Check edge structure
    print("\n3. EDGE STRUCTURE TEST:")
    for edge in data['edges'][:3]:
        print(f"Edge {edge['id']}:")
        print(f"  Type: {edge['type']}")
        print(f"  Source: {edge['source']} -> Target: {edge['target']}")
        print(f"  Has properties: {'properties' in edge}")
        if 'properties' in edge:
            print(f"  Properties: {list(edge['properties'].keys())}")
    
    # Test 4: Find evidence-hypothesis links
    print("\n4. EVIDENCE-HYPOTHESIS LINKS TEST:")
    from core.ontology_manager import ontology_manager
    evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
    evidence_links = []
    for edge in data['edges']:
        if edge['type'] in evidence_hypothesis_edges:
            evidence_links.append({
                'evidence': edge['source'],
                'hypothesis': edge['target'],
                'type': edge['type'],
                'probative_value': edge.get('properties', {}).get('probative_value', 0)
            })
    
    print(f"Evidence-hypothesis links found: {len(evidence_links)}")
    for link in evidence_links:
        print(f"  {link['evidence']} {link['type']} {link['hypothesis']} (value: {link['probative_value']})")
    
    # Test 5: Check mechanism constituent events
    print("\n5. MECHANISM STRUCTURE TEST:")
    mechanism_parts = {}
    for edge in data['edges']:
        if edge['type'] == 'part_of_mechanism':
            mech_id = edge['target']
            if mech_id not in mechanism_parts:
                mechanism_parts[mech_id] = []
            mechanism_parts[mech_id].append(edge['source'])
    
    for mech_id, parts in mechanism_parts.items():
        print(f"Mechanism {mech_id} has {len(parts)} constituent events: {parts}")
    
    # Test 6: Simulate causal chain finding
    print("\n6. CAUSAL CHAIN SIMULATION:")
    if triggering_events and outcome_events:
        print(f"Can form chains from {triggering_events[0]} to {outcome_events[0]}")
        # In real analysis, it would use networkx to find paths
        print("Expected chain: E1 -> E2 -> E3 -> E5 -> E6")
    
    return {
        'node_count': len(data['nodes']),
        'edge_count': len(data['edges']),
        'triggering_events': triggering_events,
        'outcome_events': outcome_events,
        'evidence_links': len(evidence_links),
        'mechanisms': len(mechanism_parts)
    }

if __name__ == "__main__":
    results = test_fixed_graph()
    print("\n=== SUMMARY ===")
    for key, value in results.items():
        print(f"{key}: {value}")