#!/usr/bin/env python3
"""Run analysis with network visualization data"""
import json
import subprocess
import sys
from pathlib import Path

def prepare_network_data(graph_path):
    """Prepare network data for vis.js visualization"""
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    nodes_js = []
    edges_js = []
    
    # Convert nodes to vis.js format
    for node in graph_data.get("nodes", []):
        node_props = node.get("attr_props", {})
        nodes_js.append({
            "id": node.get("id"),
            "label": f"{node.get('type')}\n{node.get('id')}",
            "title": json.dumps(node_props, indent=2),
            "properties": node_props,
            "group": node.get("type")
        })
    
    # Convert edges to vis.js format
    for edge in graph_data.get("edges", []):
        edges_js.append({
            "from": edge.get("source"),
            "to": edge.get("target"),
            "label": edge.get("type"),
            "title": json.dumps(edge.get("properties", {}), indent=2),
            "arrows": "to",
            "properties": edge.get("properties", {})
        })
    
    return {
        "nodes": nodes_js,
        "edges": edges_js,
        "project_name": graph_path.stem
    }

def main():
    graph_path = Path("output_data/demo/american_revolution_fixed_graph.json")
    output_path = Path("output_data/demo/american_revolution_fixed_analysis.html")
    
    # Prepare network data
    network_data = prepare_network_data(graph_path)
    
    # Save network data to temp file
    network_data_path = Path("output_data/demo/network_data_temp.json")
    with open(network_data_path, 'w') as f:
        json.dump(network_data, f)
    
    # Run analysis with network data
    cmd = [
        sys.executable, '-m', 'core.analyze',
        str(graph_path),
        '--html',
        '--output', str(output_path),
        '--network-data', str(network_data_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Clean up temp file
    network_data_path.unlink(missing_ok=True)
    
    print(f"\nAnalysis complete! Output at: {output_path}")

if __name__ == "__main__":
    main()