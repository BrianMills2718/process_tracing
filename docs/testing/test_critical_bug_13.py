#!/usr/bin/env python3
"""
Test for Critical Issue #13: Schema Override Bug
Verify that ontology loads from config file only, not hardcoded schema.
"""

import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ontology_loads_from_config_only():
    """Test that ontology loads from config file exactly"""
    print("Testing Issue #13: Schema Override Bug...")
    
    # Load config file directly
    with open('config/ontology_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Config file contains {len(config['node_types'])} node types")
    print(f"Config file contains {len(config['edge_types'])} edge types")
    
    # Import ontology module
    from core.ontology import NODE_TYPES, EDGE_TYPES, NODE_COLORS
    
    print(f"Ontology module loaded {len(NODE_TYPES)} node types")
    print(f"Ontology module loaded {len(EDGE_TYPES)} edge types")
    
    # Verify exact match
    config_nodes = set(config['node_types'].keys())
    loaded_nodes = set(NODE_TYPES.keys())
    
    config_edges = set(config['edge_types'].keys())
    loaded_edges = set(EDGE_TYPES.keys())
    
    print("\nNode type comparison:")
    print(f"Config: {sorted(config_nodes)}")
    print(f"Loaded: {sorted(loaded_nodes)}")
    
    print("\nEdge type comparison:")
    print(f"Config: {sorted(config_edges)}")
    print(f"Loaded: {sorted(loaded_edges)}")
    
    # Test results
    nodes_match = config_nodes == loaded_nodes
    edges_match = config_edges == loaded_edges
    
    # Additional detailed comparison
    if nodes_match and edges_match:
        print("\n✅ SUCCESS: Ontology loads from config file correctly")
        
        # Verify some specific properties match
        config_event_props = set(config['node_types']['Event']['properties'].keys())
        loaded_event_props = set(NODE_TYPES['Event']['properties'].keys()) 
        
        props_match = config_event_props == loaded_event_props
        print(f"Event properties match: {props_match}")
        
        if props_match:
            print("✅ VERIFIED: Schema loads from configuration file only")
            return True
        else:
            print("❌ FAILED: Properties don't match between config and loaded schema")
            return False
    else:
        print("❌ FAILED: Schema doesn't match config file")
        print(f"Missing nodes in loaded: {config_nodes - loaded_nodes}")
        print(f"Extra nodes in loaded: {loaded_nodes - config_nodes}")
        print(f"Missing edges in loaded: {config_edges - loaded_edges}")
        print(f"Extra edges in loaded: {loaded_edges - config_edges}")
        return False

if __name__ == "__main__":
    success = test_ontology_loads_from_config_only()
    if not success:
        sys.exit(1)
    else:
        print("\n🎉 Issue #13 appears to be already fixed or was misdiagnosed!")