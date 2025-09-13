#!/usr/bin/env python3
import json
import sys

def remove_edge_type(ontology_file, edge_type):
    with open(ontology_file, 'r') as f:
        config = json.load(f)
    
    # Remove from edge types
    if 'edge_types' in config:
        config['edge_types'] = {k: v for k, v in config['edge_types'].items() if k != edge_type}
    
    with open(ontology_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Removed edge type '{edge_type}' from ontology")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_ontology.py <ontology_file> <edge_type_to_remove>")
        sys.exit(1)
    remove_edge_type(sys.argv[1], sys.argv[2])