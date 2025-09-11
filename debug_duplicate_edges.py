#!/usr/bin/env python3
"""
PHASE 23A: Duplicate Edge Analysis
Check if edges are being collapsed due to duplicate source-target pairs
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

def analyze_duplicate_edges(json_file):
    """
    Check for duplicate edges that might be collapsed by NetworkX
    """
    
    print("🔍 PHASE 23A: DUPLICATE EDGE ANALYSIS")
    print("="*50)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    edges = data.get('edges', [])
    print(f"📊 Total edges in JSON: {len(edges)}")
    
    # Group edges by source-target pair
    edge_groups = defaultdict(list)
    
    for i, edge in enumerate(edges):
        source_id = edge.get('source_id', edge.get('source'))
        target_id = edge.get('target_id', edge.get('target'))
        edge_type = edge.get('type')
        
        if source_id and target_id:
            key = (source_id, target_id)
            edge_groups[key].append({
                'index': i,
                'type': edge_type,
                'edge_data': edge
            })
    
    print(f"📊 Unique source-target pairs: {len(edge_groups)}")
    
    # Find duplicates
    duplicates = {key: group for key, group in edge_groups.items() if len(group) > 1}
    
    if duplicates:
        print(f"\n🚨 DUPLICATE EDGES FOUND: {len(duplicates)} duplicate pairs")
        print(f"   Total duplicate edges: {sum(len(group) for group in duplicates.values())}")
        print(f"   Expected edge loss: {sum(len(group) - 1 for group in duplicates.values())}")
        
        print(f"\n📋 DUPLICATE DETAILS:")
        for (source, target), group in duplicates.items():
            print(f"   {source} → {target}: {len(group)} edges")
            for edge_info in group:
                edge = edge_info['edge_data']
                edge_id = edge.get('id', 'NO_ID')
                print(f"     - ID: {edge_id}, Type: {edge_info['type']}")
            print()
        
        # Calculate expected loss
        expected_loss = sum(len(group) - 1 for group in duplicates.values())
        return {
            'duplicate_pairs': len(duplicates),
            'total_duplicates': sum(len(group) for group in duplicates.values()),
            'expected_loss': expected_loss,
            'duplicate_details': duplicates
        }
    else:
        print(f"✅ No duplicate source-target pairs found")
        return {
            'duplicate_pairs': 0,
            'total_duplicates': 0,
            'expected_loss': 0
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_duplicate_edges.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not Path(json_file).exists():
        print(f"❌ JSON file not found: {json_file}")
        sys.exit(1)
    
    result = analyze_duplicate_edges(json_file)
    
    print(f"\n📋 SUMMARY:")
    print(f"   Expected loss from duplicates: {result['expected_loss']} edges")
    
    if result['expected_loss'] > 0:
        print(f"\n🎯 ROOT CAUSE: NetworkX collapsing duplicate source-target pairs")
        print(f"   RESOLUTION: Use MultiDiGraph or ensure unique source-target pairs")
    else:
        print(f"\n🎯 DUPLICATE EDGES NOT THE CAUSE - investigate further")