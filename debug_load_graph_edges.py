#!/usr/bin/env python3
"""
PHASE 23A: Load Graph Edge Analysis
Identifies exactly which edges are dropped during load_graph processing
"""

import json
import sys
from pathlib import Path

def analyze_load_graph_edge_loss(json_file):
    """
    Identifies which specific edges are lost during load_graph processing
    """
    
    print("🔍 PHASE 23A: LOAD_GRAPH EDGE LOSS ANALYSIS")
    print("="*50)
    
    # Load original JSON
    with open(json_file, 'r') as f:
        original_data = json.load(f)
    
    original_nodes = {node['id']: node for node in original_data.get('nodes', [])}
    original_edges = original_data.get('edges', [])
    
    print(f"📊 ORIGINAL DATA:")
    print(f"   Nodes: {len(original_nodes)}")
    print(f"   Edges: {len(original_edges)}")
    
    # Check for edges with missing source/target nodes
    orphaned_edges = []
    valid_edges = []
    
    for edge in original_edges:
        source_id = edge.get('source_id')
        target_id = edge.get('target_id')
        
        if not source_id or not target_id:
            orphaned_edges.append({
                'edge': edge,
                'reason': 'missing_source_or_target_id'
            })
        elif source_id not in original_nodes:
            orphaned_edges.append({
                'edge': edge,
                'reason': f'source_node_missing: {source_id}'
            })
        elif target_id not in original_nodes:
            orphaned_edges.append({
                'edge': edge,
                'reason': f'target_node_missing: {target_id}'
            })
        else:
            valid_edges.append(edge)
    
    print(f"\n🚨 EDGE VALIDATION:")
    print(f"   Valid Edges: {len(valid_edges)}")
    print(f"   Orphaned Edges: {len(orphaned_edges)}")
    
    if orphaned_edges:
        print(f"\n📋 ORPHANED EDGES DETAILS:")
        for i, orphan in enumerate(orphaned_edges, 1):
            edge = orphan['edge']
            print(f"   {i}. Source: {edge.get('source_id', 'MISSING')}")
            print(f"      Target: {edge.get('target_id', 'MISSING')}")
            print(f"      Type: {edge.get('type', 'MISSING')}")
            print(f"      Reason: {orphan['reason']}")
            print()
    
    # Now test load_graph
    print(f"\n📂 TESTING LOAD_GRAPH:")
    sys.path.insert(0, '/home/brian/projects/process_tracing')
    from core.analyze import load_graph
    
    G, data = load_graph(json_file)
    
    loaded_edge_count = G.number_of_edges()
    expected_valid_edges = len(valid_edges)
    
    print(f"   Expected Valid Edges: {expected_valid_edges}")
    print(f"   Actually Loaded Edges: {loaded_edge_count}")
    print(f"   Additional Loss: {expected_valid_edges - loaded_edge_count}")
    
    if expected_valid_edges != loaded_edge_count:
        print(f"   ❌ Additional edges lost during load_graph processing!")
        print(f"   This indicates a bug in load_graph beyond orphaned edges")
    else:
        print(f"   ✅ load_graph correctly loaded all valid edges")
    
    return {
        'original_edges': len(original_edges),
        'valid_edges': len(valid_edges),
        'orphaned_edges': len(orphaned_edges),
        'loaded_edges': loaded_edge_count,
        'orphaned_details': orphaned_edges
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_load_graph_edges.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not Path(json_file).exists():
        print(f"❌ JSON file not found: {json_file}")
        sys.exit(1)
    
    result = analyze_load_graph_edge_loss(json_file)
    
    print(f"\n📋 SUMMARY:")
    print(f"   Original: {result['original_edges']} edges")
    print(f"   Valid: {result['valid_edges']} edges ({result['orphaned_edges']} orphaned)")
    print(f"   Loaded: {result['loaded_edges']} edges")
    print(f"   Expected Loss: {result['orphaned_edges']} (orphaned edges)")
    print(f"   Actual Loss: {result['original_edges'] - result['loaded_edges']}")
    
    if result['orphaned_edges'] > 0:
        print(f"\n🎯 ROOT CAUSE: Orphaned edges in extracted JSON")
        print(f"   RESOLUTION: Fix extraction to ensure edge consistency")
    else:
        print(f"\n🎯 ROOT CAUSE: Unknown - investigate load_graph logic")