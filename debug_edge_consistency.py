#!/usr/bin/env python3
"""
PHASE 23A: Node/Edge Consistency Analysis Script
Systematically analyzes raw LLM response vs final loaded graph to identify data loss
"""

import json
import sys
from pathlib import Path

def analyze_edge_consistency(raw_response_file, extracted_json_file):
    """
    Compares raw LLM response with extracted JSON to identify missing edges
    
    Args:
        raw_response_file: Path to raw LLM JSON response 
        extracted_json_file: Path to final extracted JSON file
    """
    
    print("🔍 PHASE 23A: NODE/EDGE CONSISTENCY ANALYSIS")
    print("="*50)
    
    # Load raw response
    with open(raw_response_file, 'r') as f:
        raw_data = json.load(f)
    
    # Load extracted data  
    with open(extracted_json_file, 'r') as f:
        extracted_data = json.load(f)
    
    # Analyze nodes
    raw_nodes = raw_data.get('nodes', [])
    raw_node_ids = {node['id'] for node in raw_nodes}
    
    extracted_nodes = extracted_data.get('nodes', [])
    extracted_node_ids = {node['id'] for node in extracted_nodes}
    
    print(f"📊 NODES COMPARISON:")
    print(f"   Raw LLM Response: {len(raw_nodes)} nodes")
    print(f"   Final Extracted: {len(extracted_nodes)} nodes")
    print(f"   Node Loss: {len(raw_nodes) - len(extracted_nodes)}")
    
    if len(raw_nodes) != len(extracted_nodes):
        missing_nodes = raw_node_ids - extracted_node_ids
        extra_nodes = extracted_node_ids - raw_node_ids
        print(f"   Missing Nodes: {missing_nodes}")
        print(f"   Extra Nodes: {extra_nodes}")
    else:
        print("   ✅ No node loss detected")
    
    # Analyze edges  
    raw_edges = raw_data.get('edges', [])
    extracted_edges = extracted_data.get('edges', [])
    
    print(f"\n🔗 EDGES COMPARISON:")
    print(f"   Raw LLM Response: {len(raw_edges)} edges")
    print(f"   Final Extracted: {len(extracted_edges)} edges")
    print(f"   Edge Loss: {len(raw_edges) - len(extracted_edges)}")
    
    # Check for orphaned edges in raw response
    raw_edge_sources = {edge['source_id'] for edge in raw_edges}
    raw_edge_targets = {edge['target_id'] for edge in raw_edges}
    
    orphaned_sources_raw = raw_edge_sources - raw_node_ids
    orphaned_targets_raw = raw_edge_targets - raw_node_ids
    
    print(f"\n🚨 ORPHANED EDGES IN RAW RESPONSE:")
    print(f"   Orphaned Sources: {len(orphaned_sources_raw)} - {orphaned_sources_raw}")
    print(f"   Orphaned Targets: {len(orphaned_targets_raw)} - {orphaned_targets_raw}")
    
    if orphaned_sources_raw or orphaned_targets_raw:
        print(f"   ❌ Raw LLM response has orphaned edges - this is the root cause!")
        
        # Find specific orphaned edges
        orphaned_edges = []
        for edge in raw_edges:
            if edge['source_id'] in orphaned_sources_raw or edge['target_id'] in orphaned_targets_raw:
                orphaned_edges.append(edge)
        
        print(f"\n🔍 ORPHANED EDGES DETAILS:")
        for i, edge in enumerate(orphaned_edges, 1):
            print(f"   {i}. ID: {edge['id']}")
            print(f"      Source: {edge['source_id']} (exists: {edge['source_id'] in raw_node_ids})")
            print(f"      Target: {edge['target_id']} (exists: {edge['target_id'] in raw_node_ids})")
            print(f"      Type: {edge['type']}")
            print()
            
        return {
            'root_cause': 'orphaned_edges_in_raw_response',
            'orphaned_edges_count': len(orphaned_edges),
            'missing_nodes': list(orphaned_sources_raw | orphaned_targets_raw),
            'affected_edges': [edge['id'] for edge in orphaned_edges]
        }
    else:
        print(f"   ✅ No orphaned edges in raw response")
        
        # If no orphans in raw response, the issue is in processing
        raw_edge_ids = {edge['id'] for edge in raw_edges}
        # Handle different edge formats (raw has 'id', extracted may not)
        extracted_edge_ids = set()
        for edge in extracted_edges:
            if 'id' in edge:
                extracted_edge_ids.add(edge['id'])
            else:
                # Create synthetic ID from source/target/type
                synthetic_id = f"{edge.get('source_id', 'unknown')}_{edge.get('target_id', 'unknown')}_{edge.get('type', 'unknown')}"
                extracted_edge_ids.add(synthetic_id)
        
        missing_edge_ids = raw_edge_ids - extracted_edge_ids
        
        if missing_edge_ids:
            print(f"\n🔍 EDGES LOST DURING PROCESSING:")
            for edge_id in missing_edge_ids:
                edge = next(e for e in raw_edges if e['id'] == edge_id)
                print(f"   - {edge_id}: {edge['source_id']} → {edge['target_id']} ({edge['type']})")
            
            return {
                'root_cause': 'processing_pipeline_loss',
                'missing_edge_ids': list(missing_edge_ids),
                'edge_loss_count': len(missing_edge_ids)
            }
        else:
            print(f"   ✅ All edges preserved during processing")
            return {'root_cause': 'no_data_loss_detected'}

def load_graph_and_compare(json_file):
    """
    Load the JSON file with load_graph and compare edge counts
    """
    print(f"\n📂 LOAD_GRAPH COMPARISON:")
    
    # Import load_graph
    sys.path.insert(0, '/home/brian/projects/process_tracing')
    from core.analyze import load_graph
    
    # Load with our function
    G, data = load_graph(json_file)
    
    # Get original edge count
    with open(json_file, 'r') as f:
        original_data = json.load(f)
    
    original_edge_count = len(original_data.get('edges', []))
    loaded_edge_count = G.number_of_edges()
    
    print(f"   Original JSON: {original_edge_count} edges")
    print(f"   Loaded NetworkX: {loaded_edge_count} edges")
    print(f"   Load Loss: {original_edge_count - loaded_edge_count} edges")
    
    return {
        'original_edges': original_edge_count,
        'loaded_edges': loaded_edge_count,
        'load_loss': original_edge_count - loaded_edge_count
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python debug_edge_consistency.py <raw_response_file> <extracted_json_file>")
        sys.exit(1)
    
    raw_file = sys.argv[1]
    extracted_file = sys.argv[2]
    
    if not Path(raw_file).exists():
        print(f"❌ Raw response file not found: {raw_file}")
        sys.exit(1)
    
    if not Path(extracted_file).exists():
        print(f"❌ Extracted JSON file not found: {extracted_file}")
        sys.exit(1)
    
    # Run consistency analysis
    consistency_result = analyze_edge_consistency(raw_file, extracted_file)
    
    # Run load_graph comparison
    load_result = load_graph_and_compare(extracted_file)
    
    # Summary
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"   Root Cause: {consistency_result.get('root_cause', 'unknown')}")
    print(f"   Load Graph Loss: {load_result['load_loss']} edges")
    
    if consistency_result.get('root_cause') == 'orphaned_edges_in_raw_response':
        print(f"   🎯 RESOLUTION: Fix LLM prompt to ensure edge consistency")
    elif load_result['load_loss'] > 0:
        print(f"   🎯 RESOLUTION: Fix load_graph function to handle missing nodes")
    else:
        print(f"   ✅ No data integrity issues found")