#!/usr/bin/env python3

import json
from pathlib import Path

def visualize_graph_debug(graph_data, html_path, project):
    """Debug version of visualize_graph to see what's happening"""
    
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    print(f"DEBUG: Processing {len(nodes)} nodes and {len(edges)} edges")
    
    # Check for problematic edges
    repair_edges = [e for e in edges if e.get('properties', {}).get('source') == 'connectivity_repair']
    print(f"DEBUG: Found {len(repair_edges)} repair edges")
    
    for i, edge in enumerate(repair_edges[:3]):
        print(f"  Repair {i+1}: {edge.get('source')} -> {edge.get('target')} (type: {edge.get('type')})")
    
    # Generate edge JS with debug info
    edge_js_parts = []
    for i, e in enumerate(edges):
        source = e.get('source')
        target = e.get('target')
        edge_type = e.get('type')
        
        if source is None or target is None:
            print(f"DEBUG: Problematic edge {i}: source={source}, target={target}")
        
        part = f"{{from: '{source}', to: '{target}', label: '{edge_type}', title: `{json.dumps(e.get('properties', {}), indent=2)}`}}"
        edge_js_parts.append(part)
    
    edge_js = ",\n        ".join(edge_js_parts)
    
    # Generate the HTML
    node_js = ",\n        ".join([
        f"{{id: '{n.get('id')}', label: '{n.get('type')}', title: `{json.dumps(n.get('properties', {}), indent=2)}`}}" 
        for n in nodes
    ])
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Process Tracing Network</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style> #mynetwork {{ width: 100vw; height: 90vh; border: 1px solid lightgray; }} </style>
</head>
<body>
<h2>Process Tracing Network ({project})</h2>
<div id="mynetwork"></div>
<script type="text/javascript">
  var nodes = new vis.DataSet([
    {node_js}
  ]);
  var edges = new vis.DataSet([
    {edge_js}
  ]);
  var container = document.getElementById('mynetwork');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    nodes: {{ shape: 'dot', size: 20, font: {{ size: 16 }} }},
    edges: {{ arrows: 'to', font: {{ align: 'middle' }} }},
    physics: {{ stabilization: true }}
  }};
  var network = new vis.Network(container, data, options);
</script>
</body>
</html>"""
    
    # Write to file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"DEBUG: Wrote HTML to {html_path}")
    return html

def main():
    # Load the JSON data
    json_path = Path('output_data/revolutions/revolutions_20250804_205419_graph.json')
    
    with open(json_path, 'r') as f:
        graph_data = json.load(f)
    
    # Generate debug HTML
    html_path = Path('debug_graph_test.html')
    html_content = visualize_graph_debug(graph_data, html_path, "revolutions_debug")
    
    # Check the generated HTML for None values
    none_count = html_content.count("from: 'None'")
    print(f"\nDEBUG: Found {none_count} 'None' values in generated HTML")
    
    if none_count > 0:
        print("DEBUG: This indicates a problem in the generation logic")
    else:
        print("DEBUG: No None values found - the generation logic is working correctly")
        print("DEBUG: The issue might be with the original HTML file or different data was used")

if __name__ == "__main__":
    main()