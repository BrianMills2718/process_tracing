"""
HTML Generation Module for Process Tracing Analysis

Extracted from core.analyze.py to provide standalone HTML generation capabilities
without hanging issues from the main analysis module.
"""
import os
import json
import networkx as nx
from collections import defaultdict, Counter
import copy
import logging
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import textwrap

# Set up logging
logger = logging.getLogger(__name__)

# Import required constants and utilities
try:
    from core.ontology import NODE_TYPES as CORE_NODE_TYPES, NODE_COLORS
    from core.ontology_manager import ontology_manager
    from .logging_utils import log_structured_error, log_structured_info, create_analysis_context
except ImportError:
    # Fallbacks if imports not available
    CORE_NODE_TYPES = {}
    NODE_COLORS = {}
    def log_structured_error(logger, message, error_category, operation_context=None, exc_info=True, **extra_context):
        logger.error(message, exc_info=exc_info)
    def log_structured_info(logger, message, operation_context=None, **extra_context):
        logger.info(message) 
    def create_analysis_context(analysis_stage, **kwargs):
        return {"analysis_stage": analysis_stage}


def generate_embedded_network_visualization(network_data_json):
    """
    Generate HTML for an embedded vis.js network visualization.
    
    Args:
        network_data_json: JSON string with nodes, edges, and project_name
        
    Returns:
        str: HTML code for the embedded visualization
    """
    try:
        network_data = json.loads(network_data_json)
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])
        project_name = network_data.get("project_name", "Process Tracing Network")
        
        # Convert nodes and edges to JavaScript-friendly format
        nodes_js = json.dumps(nodes)
        edges_js = json.dumps(edges)
        
        # Generate the HTML with the embedded vis.js visualization
        html = f"""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Interactive Network Visualization</h2></div>
            <div class="card-body">
                <!-- Color Legend -->
                <div class="row mb-3">
                    <div class="col-12">
                        <h6>Node Type Legend:</h6>
                        <div class="d-flex flex-wrap gap-3">
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#FF6B6B;border-radius:50%;margin-right:5px;"></span>Events</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#4ECDC4;border-radius:50%;margin-right:5px;"></span>Hypotheses</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#45B7D1;border-radius:50%;margin-right:5px;"></span>Evidence</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#96CEB4;border-radius:50%;margin-right:5px;"></span>Actors</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#FFEAA7;border-radius:50%;margin-right:5px;"></span>Causal Mechanisms</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#DDA0DD;border-radius:50%;margin-right:5px;"></span>Alternative Explanations</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#98D8C8;border-radius:50%;margin-right:5px;"></span>Conditions</span>
                            <span><span style="display:inline-block;width:12px;height:12px;background-color:#F7DC6F;border-radius:50%;margin-right:5px;"></span>Data Sources</span>
                        </div>
                    </div>
                </div>
                
                <!-- Interactive Controls -->
                <div class="row mb-3">
                    <div class="col-md-4">
                        <label for="network-search" class="form-label">Search Network:</label>
                        <input type="text" class="form-control" id="network-search" placeholder="Search nodes..." onkeyup="searchNetwork(this.value)">
                    </div>
                    <div class="col-md-4">
                        <label for="node-filter" class="form-label">Filter by Type:</label>
                        <select class="form-select" id="node-filter" onchange="filterByType(this.value)">
                            <option value="all">All Types</option>
                            <option value="Event">Events</option>
                            <option value="Hypothesis">Hypotheses</option>
                            <option value="Evidence">Evidence</option>
                            <option value="Actor">Actors</option>
                            <option value="Causal_Mechanism">Causal Mechanisms</option>
                            <option value="Alternative_Explanation">Alternative Explanations</option>
                            <option value="Condition">Conditions</option>
                            <option value="Data_Source">Data Sources</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <label class="form-label">Network Actions:</label><br>
                        <button class="btn btn-sm btn-outline-primary" onclick="network.fit()">Fit View</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="searchNetwork('')">Clear Search</button>
                    </div>
                </div>
                
                <div id="mynetwork" style="width: 100%; height: 600px; border: 1px solid lightgray;"></div>
                
                <!-- Network Information Panel -->
                <div class="mt-3">
                    <div id="network-info" class="alert alert-light">Click nodes or edges for details</div>
                </div>
                <script type="text/javascript">
                    // Create the network visualization once the page has loaded
                    document.addEventListener('DOMContentLoaded', function() {{
                        // Parse the nodes and edges
                        var nodes = new vis.DataSet({nodes_js});
                        var edges = new vis.DataSet({edges_js});
                        
                        // Create the network
                        var container = document.getElementById('mynetwork');
                        var data = {{
                            nodes: nodes,
                            edges: edges
                        }};
                        var options = {{
                            nodes: {{ 
                                shape: 'dot', 
                                font: {{ size: 12, color: '#000000' }},
                                borderWidth: 2,
                                chosen: {{ node: true }}
                            }},
                            edges: {{ 
                                arrows: {{ to: {{ enabled: true, scaleFactor: 1.2 }} }}, 
                                font: {{ align: 'middle', size: 10, color: '#333333' }},
                                width: 2,
                                color: {{ color: '#666666', opacity: 0.8 }},
                                chosen: {{ edge: true }}
                            }},
                            physics: {{ 
                                enabled: true,
                                stabilization: {{ iterations: 200 }},
                                barnesHut: {{
                                    gravitationalConstant: -3000,
                                    centralGravity: 0.3,
                                    springLength: 200,
                                    springConstant: 0.05,
                                    damping: 0.09,
                                    avoidOverlap: 0.1
                                }}
                            }},
                            interaction: {{ 
                                hover: true,
                                tooltipDelay: 300,
                                zoomView: true,
                                dragView: true
                            }}
                        }};
                        var network = new vis.Network(container, data, options);
                        
                        // Add tooltip behavior and interactivity
                        network.on("hoverNode", function (params) {{
                            // Show tooltip with node information
                            document.getElementById('network-info').innerHTML = 
                                '<div class="alert alert-info"><strong>Hover:</strong> ' + 
                                (params.node ? 'Node ' + params.node : 'No node') + '</div>';
                        }});
                        
                        network.on("click", function (params) {{
                            if (params.nodes.length > 0) {{
                                var nodeId = params.nodes[0];
                                var node = nodes.get(nodeId);
                                document.getElementById('network-info').innerHTML = 
                                    '<div class="alert alert-success"><strong>Selected Node:</strong><br>' +
                                    '<strong>ID:</strong> ' + node.id + '<br>' +
                                    '<strong>Type:</strong> ' + node.type + '<br>' +
                                    '<strong>Description:</strong> ' + (node.title || 'No description') + '</div>';
                            }} else if (params.edges.length > 0) {{
                                var edgeId = params.edges[0];
                                var edge = edges.get(edgeId);
                                document.getElementById('network-info').innerHTML = 
                                    '<div class="alert alert-warning"><strong>Selected Edge:</strong><br>' +
                                    '<strong>Type:</strong> ' + edge.label + '<br>' +
                                    '<strong>From:</strong> ' + edge.from + '<br>' +
                                    '<strong>To:</strong> ' + edge.to + '<br>' +
                                    '<strong>Description:</strong> ' + (edge.title || 'No description') + '</div>';
                            }} else {{
                                document.getElementById('network-info').innerHTML = 
                                    '<div class="alert alert-light">Click nodes or edges for details</div>';
                            }}
                        }});
                        
                        // Add search functionality
                        window.searchNetwork = function(searchTerm) {{
                            if (!searchTerm) {{
                                // Reset all node colors
                                var allNodes = nodes.get();
                                allNodes.forEach(function(node) {{
                                    var originalColor = getNodeColor(node.type);
                                    nodes.update({{id: node.id, color: originalColor}});
                                }});
                                return;
                            }}
                            
                            var matchingNodes = [];
                            var allNodes = nodes.get();
                            allNodes.forEach(function(node) {{
                                if (node.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
                                    node.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                                    node.type.toLowerCase().includes(searchTerm.toLowerCase())) {{
                                    matchingNodes.push(node.id);
                                    nodes.update({{id: node.id, color: '#FFD700'}});
                                }} else {{
                                    var originalColor = getNodeColor(node.type);
                                    nodes.update({{id: node.id, color: originalColor}});
                                }}
                            }});
                            
                            if (matchingNodes.length > 0) {{
                                network.selectNodes(matchingNodes);
                                network.focus(matchingNodes[0], {{animation: true}});
                            }}
                        }};
                        
                        function getNodeColor(nodeType) {{
                            var colorMap = {{
                                'Event': '#FF6B6B',
                                'Hypothesis': '#4ECDC4',
                                'Evidence': '#45B7D1',
                                'Actor': '#96CEB4',
                                'Causal_Mechanism': '#FFEAA7',
                                'Alternative_Explanation': '#DDA0DD',
                                'Condition': '#98D8C8',
                                'Data_Source': '#F7DC6F'
                            }};
                            return colorMap[nodeType] || '#CCCCCC';
                        }}
                        
                        // Filter by node type
                        window.filterByType = function(nodeType) {{
                            if (nodeType === 'all') {{
                                network.setData({{nodes: nodes, edges: edges}});
                                return;
                            }}
                            
                            var filteredNodes = nodes.get({{
                                filter: function(item) {{
                                    return item.type === nodeType;
                                }}
                            }});
                            
                            var nodeIds = filteredNodes.map(function(node) {{ return node.id; }});
                            var filteredEdges = edges.get({{
                                filter: function(item) {{
                                    return nodeIds.includes(item.from) && nodeIds.includes(item.to);
                                }}
                            }});
                            
                            network.setData({{
                                nodes: new vis.DataSet(filteredNodes),
                                edges: new vis.DataSet(filteredEdges)
                            }});
                        }};
                    }});
                </script>
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error("Failed to generate embedded network visualization", exc_info=True)
        return f"<div class='alert alert-danger'>Error generating network visualization: {e}</div>"


def analyze_graph_structure(G):
    """
    Analyze NetworkX graph structure for advanced metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        dict: Analysis results with centrality measures, path analysis, etc.
    """
    try:
        analysis = {}
        
        # Basic metrics
        analysis['node_count'] = G.number_of_nodes()
        analysis['edge_count'] = G.number_of_edges()
        analysis['density'] = nx.density(G)
        
        # Centrality measures
        if G.number_of_nodes() > 0:
            analysis['degree_centrality'] = nx.degree_centrality(G)
            analysis['betweenness_centrality'] = nx.betweenness_centrality(G)
            analysis['closeness_centrality'] = nx.closeness_centrality(G)
            
            # Find most central nodes
            degree_sorted = sorted(analysis['degree_centrality'].items(), key=lambda x: x[1], reverse=True)
            analysis['most_central_nodes'] = degree_sorted[:5]
        
        # Connectivity analysis
        if G.number_of_nodes() > 1:
            if nx.is_connected(G.to_undirected()):
                analysis['avg_path_length'] = nx.average_shortest_path_length(G.to_undirected())
            else:
                analysis['avg_path_length'] = 'Graph not connected'
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing graph structure: {e}")
        return {'error': str(e)}


def analyze_evidence_strength(G, data):
    """
    Analyze evidence strength and hypothesis support using Van Evera methodology.
    
    Args:
        G: NetworkX graph
        data: Data dictionary with nodes and edges
        
    Returns:
        dict: Analysis results with evidence types, hypothesis support, etc.
    """
    try:
        analysis = {}
        
        # Get nodes by type
        evidence_nodes = [n for n in data.get('nodes', []) if n.get('type') == 'Evidence']
        hypothesis_nodes = [n for n in data.get('nodes', []) if n.get('type') == 'Hypothesis']
        
        # Evidence type classification (Van Evera categories)
        evidence_types = {
            'smoking_gun': 0,
            'hoop': 0,
            'straw_in_wind': 0,
            'doubly_decisive': 0
        }
        
        # Analyze evidence-hypothesis relationships
        evidence_hypothesis_links = []
        # Get valid evidence-hypothesis edge types from ontology
        evidence_hypothesis_edge_types = ontology_manager.get_evidence_hypothesis_edges()
        evidence_hypothesis_edge_types.append('supports_hypothesis')  # Legacy support
        
        for edge in data.get('edges', []):
            if (edge.get('type') in evidence_hypothesis_edge_types and
                any(h['id'] == edge.get('target_id') for h in hypothesis_nodes) and
                any(e['id'] == edge.get('source_id') for e in evidence_nodes)):
                evidence_hypothesis_links.append(edge)
        
        analysis['evidence_count'] = len(evidence_nodes)
        analysis['hypothesis_count'] = len(hypothesis_nodes)
        analysis['evidence_hypothesis_links'] = len(evidence_hypothesis_links)
        analysis['evidence_types'] = evidence_types
        
        # Hypothesis support analysis
        hypothesis_support = {}
        for hyp in hypothesis_nodes:
            hyp_id = hyp['id']
            supporting_evidence = [e for e in evidence_hypothesis_links if e.get('target_id') == hyp_id]
            hypothesis_support[hyp_id] = {
                'description': hyp.get('properties', {}).get('description', hyp_id),
                'supporting_evidence': len(supporting_evidence),
                'strength': len(supporting_evidence) / max(len(evidence_nodes), 1)
            }
        
        analysis['hypothesis_support'] = hypothesis_support
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing evidence strength: {e}")
        return {'error': str(e)}


def find_causal_chains(G, data, max_chains=5):
    """
    Find causal chains in the process tracing network.
    
    Args:
        G: NetworkX graph
        data: Data dictionary with nodes and edges  
        max_chains: Maximum number of chains to return
        
    Returns:
        list: List of causal chains with path information
    """
    try:
        causal_chains = []
        
        # Find nodes that could be start/end points
        event_nodes = [n['id'] for n in data.get('nodes', []) if n.get('type') == 'Event']
        mechanism_nodes = [n['id'] for n in data.get('nodes', []) if n.get('type') == 'Causal_Mechanism']
        
        # Look for paths between events through mechanisms
        for start_event in event_nodes[:3]:  # Limit for performance
            for end_event in event_nodes[:3]:
                if start_event != end_event:
                    try:
                        if nx.has_path(G, start_event, end_event):
                            paths = list(nx.all_simple_paths(G, start_event, end_event, cutoff=4))
                            for path in paths[:2]:  # Limit paths per pair
                                # Check if path includes mechanisms
                                path_has_mechanism = any(node in mechanism_nodes for node in path)
                                if path_has_mechanism:
                                    causal_chains.append({
                                        'path': path,
                                        'length': len(path),
                                        'start': start_event,
                                        'end': end_event,
                                        'strength': 1.0 / len(path)  # Shorter paths are stronger
                                    })
                    except:
                        continue
        
        # Sort by strength and return top chains
        causal_chains.sort(key=lambda x: x['strength'], reverse=True)
        return causal_chains[:max_chains]
        
    except Exception as e:
        logger.error(f"Error finding causal chains: {e}")
        return []


def create_network_data_json(G, data):
    """
    Convert NetworkX graph and data to vis.js compatible JSON format.
    
    Args:
        G: NetworkX graph
        data: Original data dictionary with nodes and edges
        
    Returns:
        str: JSON string with nodes, edges, and project_name
    """
    try:
        nodes = []
        edges = []
        
        # Convert nodes
        for node_data in data.get('nodes', []):
            node_id = node_data.get('id')
            node_type = node_data.get('type', 'Unknown')
            
            # Color mapping based on type
            color_map = {
                'Event': '#FF6B6B',
                'Hypothesis': '#4ECDC4', 
                'Evidence': '#45B7D1',
                'Actor': '#96CEB4',
                'Causal_Mechanism': '#FFEAA7',
                'Alternative_Explanation': '#DDA0DD',
                'Condition': '#98D8C8',
                'Data_Source': '#F7DC6F'
            }
            
            # Get description from properties or fallback
            description = node_data.get('properties', {}).get('description') or node_data.get('description', str(node_id))
            
            vis_node = {
                'id': node_id,
                'label': textwrap.shorten(description, width=30),
                'title': f"{node_type}: {description}",
                'color': color_map.get(node_type, '#CCCCCC'),
                'type': node_type
            }
            nodes.append(vis_node)
        
        # Convert edges  
        for edge_data in data.get('edges', []):
            vis_edge = {
                'from': edge_data.get('source_id') or edge_data.get('source'),
                'to': edge_data.get('target_id') or edge_data.get('target'),
                'label': edge_data.get('type', ''),
                'title': f"{edge_data.get('type', 'Unknown')}: {edge_data.get('properties', {}).get('description', 'No description')}",
                'arrows': 'to'
            }
            edges.append(vis_edge)
        
        network_data = {
            'nodes': nodes,
            'edges': edges,
            'project_name': 'Process Tracing Analysis'
        }
        
        return json.dumps(network_data)
        
    except Exception as e:
        logger.error(f"Error creating network data JSON: {e}")
        return json.dumps({'nodes': [], 'edges': [], 'project_name': 'Error'})


def generate_process_tracing_html(G, data, output_dir, json_file_path=None):
    """
    Generate complete HTML report with network visualization and analytics.
    
    Args:
        G: NetworkX graph
        data: Data dictionary with nodes and edges
        output_dir: Output directory path
        json_file_path: Optional path to source JSON file
        
    Returns:
        str: Path to generated HTML file
    """
    try:
        # Create network data for visualization
        network_data_json = create_network_data_json(G, data)
        
        # Analyze graph structure
        graph_analysis = analyze_graph_structure(G)
        
        # Analyze evidence strength (Van Evera methodology)
        evidence_analysis = analyze_evidence_strength(G, data)
        
        # Find causal chains
        causal_chains = find_causal_chains(G, data)
        
        # Generate basic statistics
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        
        # Count by type
        node_types = {}
        for node in nodes:
            node_type = node.get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        edge_types = {}
        for edge in edges:
            edge_type = edge.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Generate HTML with embedded visualization
        html_header = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Process Tracing Analysis Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        .card { margin-bottom: 20px; }
        .chart { text-align: center; margin: 20px 0; }
        .llm-summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #007bff; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="text-center mb-4">Process Tracing Analysis Report</h1>
"""
        
        # Network visualization
        network_vis_html = generate_embedded_network_visualization(network_data_json)
        
        # Analytics sections
        html_body = f"""
        {network_vis_html}
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Network Statistics</h3></div>
                    <div class="card-body">
                        <p><strong>Total Nodes:</strong> {len(nodes)}</p>
                        <p><strong>Total Edges:</strong> {len(edges)}</p>
                        <p><strong>Network Density:</strong> {graph_analysis.get('density', 0):.4f}</p>
                        <p><strong>Average Path Length:</strong> {graph_analysis.get('avg_path_length', 'N/A')}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Node Type Distribution</h3></div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
"""
        
        for node_type, count in node_types.items():
            html_body += f'<li class="list-group-item d-flex justify-content-between align-items-center">{node_type} <span class="badge bg-primary rounded-pill">{count}</span></li>'
        
        html_body += """
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Most Central Nodes (Degree Centrality)</h3></div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
"""
        
        # Add central nodes
        most_central = graph_analysis.get('most_central_nodes', [])
        for node_id, centrality in most_central:
            # Find node description
            node_desc = 'Unknown'
            for node in nodes:
                if node.get('id') == node_id:
                    node_desc = textwrap.shorten(node.get('description', str(node_id)), width=60)
                    break
            
            html_body += f'<li class="list-group-item d-flex justify-content-between align-items-center">{node_desc} <span class="badge bg-success rounded-pill">{centrality:.3f}</span></li>'
        
        html_body += """
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Edge Type Distribution</h3></div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
"""
        
        for edge_type, count in edge_types.items():
            html_body += f'<li class="list-group-item d-flex justify-content-between align-items-center">{edge_type} <span class="badge bg-secondary rounded-pill">{count}</span></li>'
        
        # Add Van Evera and causal analysis sections
        van_evera_section = f"""
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Evidence Analysis (Van Evera)</h3></div>
                    <div class="card-body">
                        <p><strong>Evidence Nodes:</strong> {evidence_analysis.get('evidence_count', 0)}</p>
                        <p><strong>Hypothesis Nodes:</strong> {evidence_analysis.get('hypothesis_count', 0)}</p>
                        <p><strong>Evidence-Hypothesis Links:</strong> {evidence_analysis.get('evidence_hypothesis_links', 0)}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Causal Chains</h3></div>
                    <div class="card-body">
                        <p><strong>Detected Chains:</strong> {len(causal_chains)}</p>"""

        if causal_chains:
            van_evera_section += "<h6>Top Causal Chains:</h6><ul class='list-group list-group-flush'>"
            for i, chain in enumerate(causal_chains[:3]):
                path_str = " → ".join([node[:20] + "..." if len(node) > 20 else node for node in chain['path']])
                van_evera_section += f'<li class="list-group-item"><small><strong>Chain {i+1}:</strong> {path_str} <span class="badge bg-info">Strength: {chain["strength"]:.3f}</span></small></li>'
            van_evera_section += "</ul>"
        else:
            van_evera_section += "<p><em>No causal chains detected</em></p>"

        van_evera_section += """
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header"><h3 class="card-title h5">Hypothesis Support Analysis</h3></div>
                    <div class="card-body">"""

        hypothesis_support = evidence_analysis.get('hypothesis_support', {})
        if hypothesis_support:
            van_evera_section += '<ul class="list-group list-group-flush">'
            for hyp_id, support_data in hypothesis_support.items():
                description = textwrap.shorten(support_data.get('description', hyp_id), width=60)
                strength = support_data.get('strength', 0)
                evidence_count = support_data.get('supporting_evidence', 0)
                strength_class = 'success' if strength > 0.5 else 'warning' if strength > 0.2 else 'danger'
                van_evera_section += f'''<li class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>{description}</strong><br>
                            <small>Supporting Evidence: {evidence_count} items</small>
                        </div>
                        <span class="badge bg-{strength_class} rounded-pill">Strength: {strength:.2f}</span>
                    </div>
                </li>'''
            van_evera_section += '</ul>'
        else:
            van_evera_section += '<p><em>No hypothesis support data available</em></p>'

        van_evera_section += """
                    </div>
                </div>
            </div>
        </div>"""
        
        html_body += van_evera_section
        
        html_footer = """
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        
        # Combine all HTML parts
        complete_html = html_header + html_body + html_footer
        
        # Generate output filename
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if json_file_path:
            basename = Path(json_file_path).stem
        else:
            basename = "process_tracing"
        
        html_filename = f"{basename}_analysis_{timestamp}.html"
        html_filepath = Path(output_dir) / html_filename
        
        # Write HTML file
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(complete_html)
        
        logger.info(f"Generated HTML report: {html_filepath}")
        return str(html_filepath)
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}", exc_info=True)
        raise


def generate_html_report(G, data, output_dir, json_file_path=None):
    """
    Main function to generate HTML report - matches expected interface.
    
    Args:
        G: NetworkX graph
        data: Data dictionary with nodes and edges  
        output_dir: Output directory path
        json_file_path: Optional path to source JSON file
        
    Returns:
        str: Path to generated HTML file
    """
    return generate_process_tracing_html(G, data, output_dir, json_file_path)