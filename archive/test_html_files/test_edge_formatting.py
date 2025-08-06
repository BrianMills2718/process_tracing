#!/usr/bin/env python3

import json

# Test the exact string formatting logic from the code
def test_edge_formatting():
    # Test edge missing source/target
    test_edge = {'type': 'supports', 'properties': {'reasoning': 'test'}}
    
    # This is what the code does:
    formatted = f"{{from: '{test_edge.get('source')}', to: '{test_edge.get('target')}', label: '{test_edge.get('type')}', title: `{json.dumps(test_edge.get('properties', {}), indent=2)}`}}"
    
    print('Formatted edge with missing source/target:')
    print(formatted)
    print()
    
    # Test with repair edge from actual data
    with open('output_data/revolutions/revolutions_20250804_205419_graph.json', 'r') as f:
        data = json.load(f)
    
    repair_edge = [e for e in data['edges'] if e.get('properties', {}).get('source') == 'connectivity_repair'][0]
    print('Repair edge data:')
    print(json.dumps(repair_edge, indent=2))
    print()
    
    formatted_repair = f"{{from: '{repair_edge.get('source')}', to: '{repair_edge.get('target')}', label: '{repair_edge.get('type')}', title: `{json.dumps(repair_edge.get('properties', {}), indent=2)}`}}"
    print('Formatted repair edge:')
    print(formatted_repair)

if __name__ == "__main__":
    test_edge_formatting()