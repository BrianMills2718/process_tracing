#!/usr/bin/env python
"""
Process Tracing Network Analyzer (Core Analysis Engine)
------------------------------------------------------
Analyzes process tracing networks using theoretical approaches to provide
insights into causal mechanisms, evidence strength, and alternative explanations.

This script is typically called by an orchestrator script.
"""

import os
import sys
import json
import argparse
import networkx as nx
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for servers or scripts
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import textwrap

# Assuming this file will be in core/ and ontology.py is in core/
from core.ontology import NODE_TYPES as CORE_NODE_TYPES, NODE_COLORS

# Evidence type classifications from Van Evera's tests (specific to this analysis module)
EVIDENCE_TYPES_VAN_EVERA = {
    "hoop": {
        "necessary": True,
        "sufficient": False,
        "description": "Necessary but insufficient - hypothesis must pass this test to remain viable"
    },
    "smoking_gun": {
        "necessary": False,
        "sufficient": True,
        "description": "Sufficient but unnecessary - hypothesis is confirmed if passes, but not disconfirmed if fails"
    },
    "double_decisive": {
        "necessary": True,
        "sufficient": True,
        "description": "Both necessary and sufficient - confirms one hypothesis and eliminates others"
    },
    "straw_in_wind": {
        "necessary": False,
        "sufficient": False,
        "description": "Neither necessary nor sufficient - suggests support but doesn't confirm or eliminate"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Process Tracing Network Analyzer (Core Engine)")
    parser.add_argument("json_file", help="JSON file with process tracing data")
    parser.add_argument("--theory", "-t", action="store_true", help="Include theoretical insights section (default with HTML)")
    parser.add_argument("--output", "-o", help="Output file for analysis report (e.g., analysis.html or analysis.md)")
    parser.add_argument("--html", action="store_true", help="Generate an HTML report with embedded visualizations")
    parser.add_argument("--charts-dir", help="Directory to save PNG charts (if not generating HTML report). Default: no separate charts saved if path not given.")
    args = parser.parse_args()
    if args.html and not args.theory:
        args.theory = True 
    return args

def fix_json_for_causal_chains(data):
    """
    Fix JSON data to enable causal chain detection by making the following changes:
    1. Convert nodes with 'type' values of 'triggering', 'intermediate', 'outcome', etc. to have a primary type of 'Event'
    2. Move original type values to an 'event_type' property
    3. Ensure edges have proper 'relation' property (for backward compatibility with 'label')
    """
    if not isinstance(data, dict) or "nodes" not in data or "edges" not in data:
        print("❌ Invalid JSON structure. Expected {nodes: [...], edges: [...]}")
        return None
    
    fixed_data = {"nodes": [], "edges": []}
    event_types = ["triggering", "intermediate", "outcome", "background", "unspecified"]
    
    # Process nodes
    for node in data["nodes"]:
        node_copy = node.copy()
        
        # Fix event nodes
        if "type" in node and node["type"] in event_types:
            # Store original type as event_type
            node_copy["event_type"] = node["type"]
            # Set primary type to Event
            node_copy["type"] = "Event"
            
        fixed_data["nodes"].append(node_copy)
    
    # Process edges
    for edge in data["edges"]:
        edge_copy = edge.copy()
        
        # Ensure 'relation' property exists, but don't have both label and relation
        if "label" in edge:
            # If both exist, keep relation and remove label
            if "relation" in edge:
                edge_copy.pop("label")
            else:
                # Only label exists, copy it to relation
                edge_copy["relation"] = edge["label"]
                edge_copy.pop("label")
        
        fixed_data["edges"].append(edge_copy)
    
    return fixed_data

def load_graph(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Fix the data format for causal chain detection
    fixed_data = fix_json_for_causal_chains(data)
    if not fixed_data:
        raise ValueError("Failed to fix JSON data format")
    
    G = nx.DiGraph()
    for node in fixed_data['nodes']:
        G.add_node(node['id'], **node)
    for edge in fixed_data['edges']:
        # Create a copy of the edge attributes to manipulate
        edge_attrs = edge.copy()
        
        # Remove source and target from attributes dict
        source = edge_attrs.pop('source')
        target = edge_attrs.pop('target')
        
        # Add the edge with cleaned attributes
        G.add_edge(source, target, **edge_attrs)
    return G, fixed_data

def identify_causal_chains(G):
    """Identify causal chains in the graph"""
    chains = []
    event_nodes = [n for n, attr in G.nodes(data=True) 
                   if attr.get('type') == 'Event']
    
    # Find nodes with no incoming edges (potential starting points)
    starting_points = [n for n in event_nodes if G.in_degree(n) == 0]
    
    # Find nodes with no outgoing edges (potential endpoints)
    end_points = [n for n in event_nodes if G.out_degree(n) == 0]
    
    # For each starting and ending point, find all simple paths
    for start in starting_points:
        for end in end_points:
            try:
                paths = list(nx.all_simple_paths(G, start, end))
                for path in paths:
                    # Only include paths that consist primarily of causes relationships
                    edges = []
                    is_causal = True
                    for i in range(len(path) - 1):
                        edge_data = G.get_edge_data(path[i], path[i + 1])
                        relation = edge_data.get('relation', '')
                        edges.append(relation)
                        if relation not in ['causes', 'enables', 'initiates']:
                            is_causal = False
                    
                    if is_causal and len(path) > 2:  # Only include chains with at least 3 nodes
                        chains.append({
                            'path': path,
                            'edges': edges,
                            'length': len(path)
                        })
            except nx.NetworkXNoPath:
                continue
    
    # Sort chains by length (longest first)
    chains.sort(key=lambda x: x['length'], reverse=True)
    
    return chains

def evaluate_mechanisms(G):
    """Evaluate causal mechanisms for sufficiency"""
    mechanisms = []
    mech_nodes = [n for n, attr in G.nodes(data=True) 
                if attr.get('type') == 'Causal_Mechanism']
    
    for mech in mech_nodes:
        # Get incoming events (causes)
        causes = []
        for pred in G.predecessors(mech):
            edge_data = G.get_edge_data(pred, mech)
            if G.nodes[pred].get('type') == 'Event' and edge_data.get('relation') == 'part_of_mechanism':
                causes.append(pred)
        
        # Get outgoing effects
        effects = []
        for succ in G.successors(mech):
            edge_data = G.get_edge_data(mech, succ)
            if G.nodes[succ].get('type') == 'Event' and edge_data.get('relation') == 'causes':
                effects.append(succ)
        
        # Calculate completeness score (0-100%)
        # Simple heuristic: mechanisms with more parts are more complete
        completeness = min(len(causes) * 20, 100)  # 20% per cause, max 100%
        
        mechanisms.append({
            'id': mech,
            'name': G.nodes[mech].get('description', mech),
            'causes': causes,
            'effects': effects,
            'completeness': completeness,
            'confidence': G.nodes[mech].get('confidence', 'unknown'),
            'level_of_detail': G.nodes[mech].get('level_of_detail', 'medium')
        })
    
    # Sort by completeness (most complete first)
    mechanisms.sort(key=lambda x: x['completeness'], reverse=True)
    
    return mechanisms

def analyze_evidence(G):
    """Analyze the evidence and its support for hypotheses"""
    evidence_analysis = {}
    
    # Get all evidence nodes
    evidence_nodes = [n for n, attr in G.nodes(data=True) 
                     if attr.get('type') == 'Evidence']
    
    # Get all hypotheses
    hypothesis_nodes = [n for n, attr in G.nodes(data=True) 
                       if attr.get('type') == 'Hypothesis']
    
    # For each hypothesis, analyze supporting and refuting evidence
    for hyp in hypothesis_nodes:
        hypothesis_name = G.nodes[hyp].get('description', hyp)
        supporting = []
        refuting = []
        
        # Check each evidence node for connections to this hypothesis
        for ev in evidence_nodes:
            if G.has_edge(ev, hyp):
                edge_data = G.get_edge_data(ev, hyp)
                relation = edge_data.get('relation', '')
                
                evidence_info = {
                    'id': ev,
                    'description': G.nodes[ev].get('description', ev),
                    'type': G.nodes[ev].get('evidence_type', G.nodes[ev].get('subtype', 'unknown')),
                    'certainty': G.nodes[ev].get('certainty', 'medium'),
                    'probative_value': edge_data.get('probative_value', 'medium')
                }
                
                if relation == 'supports':
                    supporting.append(evidence_info)
                elif relation == 'refutes' or relation == 'contradicts':
                    refuting.append(evidence_info)
        
        # Calculate evidence strength (simple heuristic)
        strength_score = len(supporting) - len(refuting)
        
        # Qualitative assessment
        if strength_score > 3:
            assessment = "Strongly supported"
        elif strength_score > 0:
            assessment = "Moderately supported"
        elif strength_score == 0:
            assessment = "Evenly contested"
        elif strength_score > -3:
            assessment = "Weakly supported"
        else:
            assessment = "Strongly contested"
            
        # Prior and posterior probabilities
        prior = G.nodes[hyp].get('prior_probability', 'unknown')
        posterior = G.nodes[hyp].get('posterior_probability', 'unknown')
        
        # Store the analysis
        evidence_analysis[hyp] = {
            'hypothesis': hypothesis_name,
            'supporting_evidence': supporting,
            'refuting_evidence': refuting,
            'evidence_balance': strength_score,
            'assessment': assessment,
            'prior_probability': prior,
            'posterior_probability': posterior
        }
    
    return evidence_analysis

def identify_conditions(G):
    """Identify enabling and constraining conditions"""
    conditions = {
        'enabling': [],
        'constraining': []
    }
    
    condition_nodes = [n for n, attr in G.nodes(data=True) 
                      if attr.get('type') == 'Condition']
    
    for cond in condition_nodes:
        cond_name = G.nodes[cond].get('description', cond)
        
        # Check outgoing edges to see if it enables or constrains
        for succ in G.successors(cond):
            edge_data = G.get_edge_data(cond, succ)
            relation = edge_data.get('relation', '')
            
            target_info = {
                'id': succ,
                'description': G.nodes[succ].get('description', succ),
                'type': G.nodes[succ].get('type', 'unknown')
            }
            
            if relation == 'enables':
                conditions['enabling'].append({
                    'condition_id': cond,
                    'condition': cond_name,
                    'target': target_info,
                    'necessity': edge_data.get('necessity', 'unknown'),
                    'certainty': edge_data.get('certainty', 'medium')
                })
            elif relation == 'constrains':
                conditions['constraining'].append({
                    'condition_id': cond,
                    'condition': cond_name,
                    'target': target_info,
                    'certainty': edge_data.get('certainty', 'medium'),
                    'type': edge_data.get('type', 'unknown')
                })
    
    return conditions

def analyze_actors(G):
    """Analyze actor roles and influence in the causal process"""
    actors = []
    
    actor_nodes = [n for n, attr in G.nodes(data=True) 
                  if attr.get('type') == 'Actor']
    
    for actor in actor_nodes:
        # Basic actor info
        actor_name = G.nodes[actor].get('name', actor)
        actor_role = G.nodes[actor].get('role', 'unknown')
        
        # Find events initiated by this actor
        initiated_events = []
        for succ in G.successors(actor):
            edge_data = G.get_edge_data(actor, succ)
            relation = edge_data.get('relation', '')
            
            if relation == 'initiates' and G.nodes[succ].get('type') == 'Event':
                initiated_events.append({
                    'id': succ,
                    'description': G.nodes[succ].get('description', succ),
                    'certainty': edge_data.get('certainty', 'medium'),
                    'intention': edge_data.get('intention', 'unknown')
                })
        
        # Calculate actor influence
        influence_score = len(initiated_events) * 10  # Simple heuristic: 10 points per initiated event
        
        actors.append({
            'id': actor,
            'name': actor_name,
            'role': actor_role,
            'initiated_events': initiated_events,
            'influence_score': influence_score,
            'beliefs': G.nodes[actor].get('beliefs', 'unknown'),
            'intentions': G.nodes[actor].get('intentions', 'unknown')
        })
    
    # Sort by influence score (most influential first)
    actors.sort(key=lambda x: x['influence_score'], reverse=True)
    
    return actors

def analyze_alternative_explanations(G):
    """Analyze alternative explanations and their supporting/refuting evidence"""
    alternatives = []
    
    alt_nodes = [n for n, attr in G.nodes(data=True) 
                if attr.get('type') == 'Alternative_Explanation']
    
    for alt in alt_nodes:
        alt_name = G.nodes[alt].get('description', alt)
        
        # Find evidence supporting or refuting this alternative
        supporting = []
        refuting = []
        
        for pred in G.predecessors(alt):
            if G.nodes[pred].get('type') == 'Evidence':
                edge_data = G.get_edge_data(pred, alt)
                relation = edge_data.get('relation', '')
                
                evidence_info = {
                    'id': pred,
                    'description': G.nodes[pred].get('description', pred),
                    'type': G.nodes[pred].get('evidence_type', G.nodes[pred].get('subtype', 'unknown')),
                    'certainty': edge_data.get('certainty', 'medium'),
                    'probative_value': edge_data.get('probative_value', 'medium')
                }
                
                if relation == 'supports_alternative':
                    supporting.append(evidence_info)
                elif relation == 'refutes_alternative':
                    refuting.append(evidence_info)
        
        # Calculate strength score
        strength_score = len(supporting) - len(refuting)
        
        # Qualitative assessment
        if strength_score > 2:
            assessment = "Strong alternative explanation"
        elif strength_score > 0:
            assessment = "Plausible alternative explanation"
        elif strength_score == 0:
            assessment = "Contested alternative explanation"
        else:
            assessment = "Weak alternative explanation"
        
        alternatives.append({
            'id': alt,
            'description': alt_name,
            'supporting_evidence': supporting,
            'refuting_evidence': refuting,
            'strength_score': strength_score,
            'assessment': assessment,
            'probability': G.nodes[alt].get('probability', 'unknown'),
            'status': G.nodes[alt].get('status', 'unknown')
        })
    
    # Sort by strength score (strongest first)
    alternatives.sort(key=lambda x: x['strength_score'], reverse=True)
    
    return alternatives

def calculate_network_metrics(G):
    """Calculate network metrics for the process trace graph"""
    metrics = {}
    
    # Node type distribution
    node_types = [data.get('type', 'unknown') for _, data in G.nodes(data=True)]
    metrics['node_type_distribution'] = dict(Counter(node_types))
    
    # Edge type distribution
    edge_types = [data.get('relation', 'unknown') for _, _, data in G.edges(data=True)]
    metrics['edge_type_distribution'] = dict(Counter(edge_types))
    
    # Graph density
    metrics['density'] = nx.density(G)
    
    # Average path length
    try:
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        metrics['avg_path_length'] = 'N/A (disconnected graph)'
    
    # Centrality measures
    # 1. Degree centrality (which nodes have most connections)
    metrics['degree_centrality'] = {
        node: round(value, 3) 
        for node, value in sorted(nx.degree_centrality(G).items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:10]  # Top 10
    }
    
    # 2. Betweenness centrality (which nodes are bridges between others)
    metrics['betweenness_centrality'] = {
        node: round(value, 3) 
        for node, value in sorted(nx.betweenness_centrality(G).items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:10]  # Top 10
    }
    
    return metrics

def format_analysis(results, data, G, theoretical_insights=None):
    """Format all analysis results into a readable MD report"""
    node_type_display_names = {
        'Event': 'Events', 'Causal_Mechanism': 'Causal Mechanisms',
        'Hypothesis': 'Hypotheses', 'Evidence': 'Evidence',
        'Condition': 'Conditions', 'Actor': 'Actors',
        'Inference_Rule': 'Inference Rules', 'Inferential_Test': 'Inferential Tests',
        'Alternative_Explanation': 'Alternative Explanations', 'Data_Source': 'Data Sources'
    }
    for core_type in CORE_NODE_TYPES.keys():
        if core_type not in node_type_display_names:
            node_type_display_names[core_type] = core_type

    filename = data.get('filename', 'Process Trace')
    report_parts = [f"# Process Tracing Analysis: {os.path.basename(filename)}"]
    report_parts.append("\n## 1. Network Overview")
    total_nodes_from_metrics = sum(results['metrics']['node_type_distribution'].values())
    report_parts.append(f"\nTotal nodes: {total_nodes_from_metrics}")
    report_parts.append("\n### Node Distribution:")
    for node_type, count in results['metrics']['node_type_distribution'].items():
        display_name = node_type_display_names.get(node_type, node_type)
        report_parts.append(f"- {display_name}: {count}")
    report_parts.append("\n## 2. Causal Chains")
    if results['causal_chains']:
        for i, chain in enumerate(results['causal_chains'][:5], 1):  # Top 5 chains
            path_str = " → ".join(chain['path'])
            report_parts.append(f"\n### Chain {i} (Length: {chain['length']})")
            report_parts.append(f"- Path: {path_str}")
            
            # Add node descriptions for this chain
            report_parts.append("- Details:")
            for node_id in chain['path']:
                node_data = next((n for n in data['nodes'] if n['id'] == node_id), None)
                if node_data:
                    desc = node_data.get('description', node_data.get('label', node_id))
                    report_parts.append(f"  - {node_id}: {desc}")
    else:
        report_parts.append("\nNo clear causal chains detected in the network.")
    report_parts.append("\n## 3. Causal Mechanisms Evaluation")
    if results['mechanisms']:
        for i, mech in enumerate(results['mechanisms'], 1):
            report_parts.append(f"\n### Mechanism {i}: {mech['name']}")
            report_parts.append(f"- Completeness: {mech['completeness']}%")
            report_parts.append(f"- Confidence: {mech['confidence']}")
            report_parts.append(f"- Level of detail: {mech['level_of_detail']}")
            
            if mech['causes']:
                causes_str = ", ".join(mech['causes'])
                report_parts.append(f"- Contributing factors: {causes_str}")
            
            if mech['effects']:
                effects_str = ", ".join(mech['effects'])
                report_parts.append(f"- Effects: {effects_str}")
    else:
        report_parts.append("\nNo causal mechanisms found in the network.")
    report_parts.append("\n## 4. Hypothesis Evaluation")
    if results['evidence_analysis']:
        for hyp_id, analysis in results['evidence_analysis'].items():
            report_parts.append(f"\n### Hypothesis: {analysis['hypothesis']}")
            report_parts.append(f"- Assessment: {analysis['assessment']}")
            report_parts.append(f"- Evidence balance: {analysis['evidence_balance']}")
            
            if analysis['supporting_evidence']:
                report_parts.append("- Supporting evidence:")
                for ev in analysis['supporting_evidence']:
                    report_parts.append(f"  - {ev['id']}: {ev['description']} (Type: {ev['type']})")
            
            if analysis['refuting_evidence']:
                report_parts.append("- Refuting evidence:")
                for ev in analysis['refuting_evidence']:
                    report_parts.append(f"  - {ev['id']}: {ev['description']} (Type: {ev['type']})")
                    
            if analysis['prior_probability'] != 'unknown' or analysis['posterior_probability'] != 'unknown':
                report_parts.append(f"- Prior probability: {analysis['prior_probability']}")
                report_parts.append(f"- Posterior probability: {analysis['posterior_probability']}")
    else:
        report_parts.append("\nNo hypotheses found in the network.")
    report_parts.append("\n## 5. Condition Analysis")
    
    # Enabling conditions
    report_parts.append("\n### Enabling Conditions:")
    if results['conditions']['enabling']:
        for cond in results['conditions']['enabling']:
            report_parts.append(f"- {cond['condition']} → enables → {cond['target']['description']}")
            if cond['necessity'] != 'unknown':
                report_parts.append(f"  - Necessity: {cond['necessity']}")
        
    # Constraining conditions
    report_parts.append("\n### Constraining Conditions:")
    if results['conditions']['constraining']:
        for cond in results['conditions']['constraining']:
            report_parts.append(f"- {cond['condition']} → constrains → {cond['target']['description']}")
    else:
        report_parts.append("No constraining conditions found.")
    
    report_parts.append("\n## 6. Actor Analysis")
    if results['actors']:
        for actor in results['actors']:
            report_parts.append(f"\n### {actor['name']}")
            report_parts.append(f"- Role: {actor['role']}")
            report_parts.append(f"- Influence score: {actor['influence_score']}")
            
            if actor['initiated_events']:
                report_parts.append("- Initiated events:")
                for event in actor['initiated_events']:
                    report_parts.append(f"  - {event['id']}: {event['description']}")
                    
            if actor['beliefs'] != 'unknown':
                report_parts.append(f"- Beliefs: {actor['beliefs']}")
            
            if actor['intentions'] != 'unknown':
                report_parts.append(f"- Intentions: {actor['intentions']}")
    else:
        report_parts.append("\nNo actors found in the network.")
    
    report_parts.append("\n## 7. Alternative Explanations")
    if results['alternatives']:
        for alt in results['alternatives']:
            report_parts.append(f"\n### {alt['description']}")
            report_parts.append(f"- Assessment: {alt['assessment']}")
            report_parts.append(f"- Strength score: {alt['strength_score']}")
            
            if alt['supporting_evidence']:
                report_parts.append("- Supporting evidence:")
                for ev in alt['supporting_evidence']:
                    report_parts.append(f"  - {ev['id']}: {ev['description']}")
            
            if alt['refuting_evidence']:
                report_parts.append("- Refuting evidence:")
                for ev in alt['refuting_evidence']:
                    report_parts.append(f"  - {ev['id']}: {ev['description']}")
    else:
        report_parts.append("\nNo alternative explanations found in the network.")
    
    report_parts.append("\n## 8. Network Metrics")
    report_parts.append(f"\n- Graph density: {results['metrics']['density']:.4f}")
    report_parts.append(f"- Average path length: {results['metrics']['avg_path_length']}")
    
    report_parts.append("\n### Most Central Nodes:")
    for node, value in list(results['metrics']['degree_centrality'].items())[:5]:
        node_data = next((n for n in data['nodes'] if n['id'] == node), None)
        if node_data:
            node_desc = node_data.get('description', node_data.get('label', node))
            node_type = node_data.get('type', 'unknown')
            report_parts.append(f"- {node} ({node_type}): {node_desc} (Centrality: {value})")
    
    if theoretical_insights:
        report_parts.append("\n## Theoretical Insights")
        report_parts.append("\n" + theoretical_insights)
    
    report_parts.append("\n## 10. Recommendations")
    report_parts.append("\nBased on the analysis, consider the following enhancements:")
    
    if not results['causal_chains']:
        report_parts.append("- Strengthen the causal narrative by identifying clear chains of events")
    
    if any(mech['completeness'] < 50 for mech in results['mechanisms']):
        report_parts.append("- Develop more complete causal mechanisms by adding missing links")
    
    if not results['conditions']['enabling']:
        report_parts.append("- Identify enabling conditions that make causal relationships possible")
    
    if not results['alternatives']:
        report_parts.append("- Consider alternative explanations to test the robustness of the analysis")
    
    if not results['actors']:
        report_parts.append("- Include key actors and their roles in initiating events")
    
    weak_hypotheses = [h for h, a in results['evidence_analysis'].items() 
                      if a['evidence_balance'] <= 0]
    if weak_hypotheses:
        report_parts.append("- Strengthen evidence for contested hypotheses")
    
    return "\n".join(report_parts)

def format_html_analysis(results, data, G, theoretical_insights=None):
    """Format analysis results into an HTML report with visualizations"""
    node_type_names = {
        'Event': 'Events', 
        'Causal_Mechanism': 'Causal Mechanisms',
        'Hypothesis': 'Hypotheses',
        'Evidence': 'Evidence', 
        'Condition': 'Conditions',
        'Actor': 'Actors',
        'Inference_Rule': 'Inference Rules', 
        'Inferential_Test': 'Inferential Tests',
        'Alternative_Explanation': 'Alternative Explanations',
        'Data_Source': 'Data Sources'
    }
    
    # Get the filename for the report title
    filename = data.get('filename', 'Process Trace')
    
    # Generate charts
    node_type_chart = generate_node_type_chart(results)
    edge_type_chart = generate_edge_type_chart(results)
    causal_chain_chart = generate_causal_chain_network(G, results['causal_chains'])
    centrality_chart = generate_centrality_chart(results)
    evidence_chart = generate_evidence_strength_chart(results)
    
    # HTML template with Bootstrap
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Process Tracing Analysis: {os.path.basename(filename)}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
            .chart {{ margin: 20px 0; text-align: center; }}
            .insights {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .evidence-item {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
            .supporting {{ background-color: rgba(40, 167, 69, 0.2); }}
            .refuting {{ background-color: rgba(220, 53, 69, 0.2); }}
            .causal-chain {{ background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            h2, h3 {{ margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">Process Tracing Analysis: {os.path.basename(filename)}</h1>
            
            <div class="row">
                <div class="col-lg-6">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h2 class="card-title h5 mb-0">Network Overview</h2>
                        </div>
                        <div class="card-body">
                            <p>Total nodes: {sum(results['metrics']['node_type_distribution'].values())}</p>
                            <h3 class="h6">Node Distribution:</h3>
                            <ul class="list-group">
    """
    
    # Add node type distribution
    for node_type, count in results['metrics']['node_type_distribution'].items():
        display_name = node_type_names.get(node_type, node_type)
        html += f'                                <li class="list-group-item d-flex justify-content-between align-items-center">{display_name} <span class="badge bg-primary rounded-pill">{count}</span></li>\n'
    
    html += f"""
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-6">
                    <div class="card mb-4">
                        <div class="card-header">
                            <h2 class="card-title h5 mb-0">Visualizations</h2>
                        </div>
                        <div class="card-body">
                            <div class="chart">
                                <h3 class="h6">Node Type Distribution</h3>
                                <img src="data:image/png;base64,{node_type_chart}" class="img-fluid" alt="Node Type Distribution">
                            </div>
                            
                            <div class="chart">
                                <h3 class="h6">Edge Type Distribution</h3>
                                <img src="data:image/png;base64,{edge_type_chart}" class="img-fluid" alt="Edge Type Distribution">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Causal Chains</h2>
                </div>
                <div class="card-body">
    """
    
    # Add causal chains section
    if results['causal_chains']:
        html += f"""
                    <div class="row">
                        <div class="col-md-6">
        """
        for i, chain in enumerate(results['causal_chains'][:5], 1):  # Top 5 chains
            path_str = " → ".join(chain['path'])
            html += f"""
                            <div class="causal-chain">
                                <h3 class="h6">Chain {i} (Length: {chain['length']})</h3>
                                <p><strong>Path:</strong> {path_str}</p>
                                <div class="details">
                                    <strong>Details:</strong>
                                    <ul class="list-group">
            """
            for node_id in chain['path']:
                node_data = next((n for n in data['nodes'] if n['id'] == node_id), None)
                if node_data:
                    desc = node_data.get('description', node_data.get('label', node_id))
                    html += f'                        <li class="list-group-item">{node_id}: {desc}</li>\n'
            html += """
                                    </ul>
                                </div>
                            </div>
            """
        html += """
                        </div>
                        <div class="col-md-6">
        """
        if causal_chain_chart:
            html += f"""
                            <div class="chart">
                                <h3 class="h6">Top Causal Chain Visualization</h3>
                                <img src="data:image/png;base64,{causal_chain_chart}" class="img-fluid" alt="Causal Chain Visualization">
                            </div>
            """
        html += """
                        </div>
                    </div>
        """
    else:
        html += "<p>No clear causal chains detected in the network.</p>"
    
    html += """
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Causal Mechanisms Evaluation</h2>
                </div>
                <div class="card-body">
    """
    
    # Add mechanisms evaluation
    if results['mechanisms']:
        for i, mech in enumerate(results['mechanisms'], 1):
            html += f"""
                    <div class="card mb-3">
                        <div class="card-header">
                            <h3 class="h6 mb-0">Mechanism {i}: {mech['name']}</h3>
                        </div>
                        <div class="card-body">
                            <div class="progress mb-3">
                                <div class="progress-bar" role="progressbar" style="width: {mech['completeness']}%;" 
                                     aria-valuenow="{mech['completeness']}" aria-valuemin="0" aria-valuemax="100">
                                    {mech['completeness']}%
                                </div>
                            </div>
                            <p><strong>Confidence:</strong> {mech['confidence']}</p>
                            <p><strong>Level of detail:</strong> {mech['level_of_detail']}</p>
            """
            
            if mech['causes']:
                causes_str = ", ".join(mech['causes'])
                html += f"<p><strong>Contributing factors:</strong> {causes_str}</p>"
            
            if mech['effects']:
                effects_str = ", ".join(mech['effects'])
                html += f"<p><strong>Effects:</strong> {effects_str}</p>"
                
            html += """
                        </div>
                    </div>
            """
    else:
        html += "<p>No causal mechanisms found in the network.</p>"
    
    html += """
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Hypothesis Evaluation</h2>
                </div>
                <div class="card-body">
    """
    
    # Add hypotheses and evidence
    if results['evidence_analysis']:
        html += """
                    <div class="row">
                        <div class="col-md-8">
        """
        for hyp_id, analysis in results['evidence_analysis'].items():
            html += f"""
                            <div class="card mb-3">
                                <div class="card-header">
                                    <h3 class="h6 mb-0">Hypothesis: {analysis['hypothesis']}</h3>
                                </div>
                                <div class="card-body">
                                    <p><strong>Assessment:</strong> {analysis['assessment']}</p>
                                    <p><strong>Evidence balance:</strong> {analysis['evidence_balance']}</p>
            """
            
            if analysis['supporting_evidence']:
                html += "<h4 class='h6'>Supporting evidence:</h4>"
                for ev in analysis['supporting_evidence']:
                    html += f"""
                                    <div class="evidence-item supporting">
                                        <strong>{ev['id']}:</strong> {ev['description']} (Type: {ev['type']})
                                    </div>
                    """
            
            if analysis['refuting_evidence']:
                html += "<h4 class='h6'>Refuting evidence:</h4>"
                for ev in analysis['refuting_evidence']:
                    html += f"""
                                    <div class="evidence-item refuting">
                                        <strong>{ev['id']}:</strong> {ev['description']} (Type: {ev['type']})
                                    </div>
                    """
                    
            if analysis['prior_probability'] != 'unknown' or analysis['posterior_probability'] != 'unknown':
                html += f"""
                                    <p><strong>Prior probability:</strong> {analysis['prior_probability']}</p>
                                    <p><strong>Posterior probability:</strong> {analysis['posterior_probability']}</p>
                """
                
            html += """
                                </div>
                            </div>
            """
        html += """
                        </div>
                        <div class="col-md-4">
        """
        if evidence_chart:
            html += f"""
                            <div class="chart">
                                <h3 class="h6">Evidence Strength Comparison</h3>
                                <img src="data:image/png;base64,{evidence_chart}" class="img-fluid" alt="Evidence Strength Chart">
                            </div>
            """
        html += """
                        </div>
                    </div>
        """
    else:
        html += "<p>No hypotheses found in the network.</p>"
    
    html += """
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Condition Analysis</h2>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <h3 class="h6">Enabling Conditions:</h3>
    """
    
    # Enabling conditions
    html += "\n### Enabling Conditions:"
    if results['conditions']['enabling']:
        html += "<ul class='list-group'>"
        for cond in results['conditions']['enabling']:
            html += f"""
                                <li class="list-group-item">
                                    <strong>{cond['condition']}</strong> → enables → {cond['target']['description']}
                                    {f"<br><small>Necessity: {cond['necessity']}</small>" if cond['necessity'] != 'unknown' else ""}
                                </li>
            """
        html += "</ul>"
    else:
        html += "<p>No enabling conditions found.</p>"
        
    html += """
                        </div>
                        <div class="col-md-6">
                            <h3 class="h6">Constraining Conditions:</h3>
    """
    
    # Constraining conditions
    if results['conditions']['constraining']:
        html += "<ul class='list-group'>"
        for cond in results['conditions']['constraining']:
            html += f"""
                                <li class="list-group-item">
                                    <strong>{cond['condition']}</strong> → constrains → {cond['target']['description']}
                                </li>
            """
        html += "</ul>"
    else:
        html += "<p>No constraining conditions found.</p>"
    
    html += """
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Actor Analysis</h2>
                </div>
                <div class="card-body">
    """
    
    # Add actor analysis
    if results['actors']:
        for actor in results['actors']:
            html += f"""
                    <div class="card mb-3">
                        <div class="card-header">
                            <h3 class="h6 mb-0">{actor['name']}</h3>
                        </div>
                        <div class="card-body">
                            <p><strong>Role:</strong> {actor['role']}</p>
                            <p><strong>Influence score:</strong> {actor['influence_score']}</p>
            """
            
            if actor['initiated_events']:
                html += "<p><strong>Initiated events:</strong></p><ul class='list-group'>"
                for event in actor['initiated_events']:
                    html += f"<li class='list-group-item'>{event['id']}: {event['description']}</li>"
                html += "</ul>"
                    
            if actor['beliefs'] != 'unknown':
                html += f"<p><strong>Beliefs:</strong> {actor['beliefs']}</p>"
            
            if actor['intentions'] != 'unknown':
                html += f"<p><strong>Intentions:</strong> {actor['intentions']}</p>"
                
            html += """
                        </div>
                    </div>
            """
    else:
        html += "<p>No actors found in the network.</p>"
    
    html += """
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Alternative Explanations</h2>
                </div>
                <div class="card-body">
    """
    
    # Add alternative explanations
    if results['alternatives']:
        for alt in results['alternatives']:
            html += f"""
                    <div class="card mb-3">
                        <div class="card-header">
                            <h3 class="h6 mb-0">{alt['description']}</h3>
                        </div>
                        <div class="card-body">
                            <p><strong>Assessment:</strong> {alt['assessment']}</p>
                            <p><strong>Strength score:</strong> {alt['strength_score']}</p>
            """
            
            if alt['supporting_evidence']:
                html += "<h4 class='h6'>Supporting evidence:</h4>"
                for ev in alt['supporting_evidence']:
                    html += f"""
                            <div class="evidence-item supporting">
                                <strong>{ev['id']}:</strong> {ev['description']}
                            </div>
                    """
            
            if alt['refuting_evidence']:
                html += "<h4 class='h6'>Refuting evidence:</h4>"
                for ev in alt['refuting_evidence']:
                    html += f"""
                            <div class="evidence-item refuting">
                                <strong>{ev['id']}:</strong> {ev['description']}
                            </div>
                    """
                
            html += """
                        </div>
                    </div>
            """
    else:
        html += "<p>No alternative explanations found in the network.</p>"
    
    html += """
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Network Metrics</h2>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
    """
    
    # Add network metrics
    html += f"""
                            <p><strong>Graph density:</strong> {results['metrics']['density']:.4f}</p>
                            <p><strong>Average path length:</strong> {results['metrics']['avg_path_length']}</p>
                            
                            <h3 class="h6">Most Central Nodes:</h3>
                            <ul class="list-group">
    """
    
    for node, value in list(results['metrics']['degree_centrality'].items())[:5]:
        node_data = next((n for n in data['nodes'] if n['id'] == node), None)
        if node_data:
            node_desc = node_data.get('description', node_data.get('label', node))
            node_type = node_data.get('type', 'unknown')
            html += f"""
                                <li class="list-group-item">
                                    <strong>{node}</strong> ({node_type}): {node_desc}
                                    <br><small>Centrality: {value}</small>
                                </li>
            """
    
    html += """
                            </ul>
                        </div>
                        <div class="col-md-6">
    """
    
    if centrality_chart:
        html += f"""
                            <div class="chart">
                                <h3 class="h6">Node Centrality</h3>
                                <img src="data:image/png;base64,{centrality_chart}" class="img-fluid" alt="Centrality Chart">
                            </div>
        """
    
    html += """
                        </div>
                    </div>
                </div>
            </div>
    """
    
    # Add theoretical insights if available
    if theoretical_insights:
        html += """
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Theoretical Insights</h2>
                </div>
                <div class="card-body insights">
        """
        # Convert the Markdown theoretical insights to HTML paragraphs
        for line in theoretical_insights.split('\n'):
            if line.startswith('# '):
                html += f"<h3>{line[2:]}</h3>"
            elif line.startswith('## '):
                html += f"<h4>{line[3:]}</h4>"
            elif line.startswith('- '):
                html += f"<li>{line[2:]}</li>"
            elif line.startswith('\n## '):
                html += f"<h4>{line[4:]}</h4>"
            elif line.strip() == "":
                html += "<br>"
            else:
                html += f"<p>{line}</p>"
        
        html += """
                </div>
            </div>
        """
    
    # Recommendations
    html += """
            <div class="card mb-4">
                <div class="card-header">
                    <h2 class="card-title h5 mb-0">Recommendations</h2>
                </div>
                <div class="card-body">
                    <p>Based on the analysis, consider the following enhancements:</p>
                    <ul class="list-group">
    """
    
    # Dynamically generate recommendations based on the analysis
    if not results['causal_chains']:
        html += '<li class="list-group-item list-group-item-warning">Strengthen the causal narrative by identifying clear chains of events</li>'
    
    if any(mech['completeness'] < 50 for mech in results['mechanisms']):
        html += '<li class="list-group-item list-group-item-warning">Develop more complete causal mechanisms by adding missing links</li>'
    
    if not results['conditions']['enabling']:
        html += '<li class="list-group-item list-group-item-warning">Identify enabling conditions that make causal relationships possible</li>'
    
    if not results['alternatives']:
        html += '<li class="list-group-item list-group-item-warning">Consider alternative explanations to test the robustness of the analysis</li>'
    
    if not results['actors']:
        html += '<li class="list-group-item list-group-item-warning">Include key actors and their roles in initiating events</li>'
    
    # Check if any hypotheses have weak evidence
    weak_hypotheses = [h for h, a in results['evidence_analysis'].items() 
                      if a['evidence_balance'] <= 0]
    if weak_hypotheses:
        html += '<li class="list-group-item list-group-item-warning">Strengthen evidence for contested hypotheses</li>'
    
    html += """
                    </ul>
                </div>
            </div>
            
            <footer class="text-center mt-4 mb-4">
                <p>Generated by Process Tracing Analyzer v1.0</p>
            </footer>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    return html

def generate_theoretical_insights(results, data):
    """Generate methodological insights based on the process tracing network analysis."""
    insights = ["# Theoretical Assessment of Process Tracing Methodology"]
    # Analyze causal chain quality
    if results['causal_chains']:
        longest_chain = max(results['causal_chains'], key=lambda x: x['length'])
        chain_quality = "robust" if longest_chain['length'] >= 6 else "limited"
        insights.append(f"\n## 1. Causal Chain Assessment")
        insights.append(f"\nThe process tracing analysis reveals a {chain_quality} causal chain structure.")
        
        if chain_quality == "robust":
            insights.append("The presence of extended causal chains enhances the explanatory power of the analysis, providing a comprehensive narrative of how initial conditions led to eventual outcomes through multiple intervening events.")
        else:
            insights.append("The relatively short causal chains suggest potential gaps in the causal story. Additional intermediate events or mechanisms would strengthen the explanatory narrative.")
    else:
        insights.append("\n## 1. Causal Chain Assessment")
        insights.append("\nThe analysis lacks clear causal chains, a significant weakness in process tracing methodology. Without established sequences of events, causal inference becomes challenging.")
    
    # Analyze mechanism sufficiency
    mechanism_completeness = [m['completeness'] for m in results['mechanisms']]
    avg_completeness = sum(mechanism_completeness) / len(mechanism_completeness) if mechanism_completeness else 0
    
    insights.append(f"\n## 2. Mechanism Sufficiency")
    
    if avg_completeness >= 70:
        insights.append("\nThe causal mechanisms demonstrated strong sufficiency with well-specified components.")
    elif avg_completeness >= 40:
        insights.append("\nThe causal mechanisms show moderate sufficiency but would benefit from further specification of intervening processes.")
    else:
        insights.append("\nThe causal mechanisms lack sufficient detail to fully explain how causes produce effects. This is a common weakness in process tracing studies, where 'black box' mechanisms fail to articulate the precise causal processes at work.")
    
    # Analyze evidence quality using Van Evera's tests
    insights.append(f"\n## 3. Evidence Quality Assessment")
    
    # Count evidence types
    evidence_types = []
    for hyp, analysis in results['evidence_analysis'].items():
        for ev in analysis['supporting_evidence'] + analysis['refuting_evidence']:
            if 'type' in ev and ev['type'] in EVIDENCE_TYPES_VAN_EVERA:
                evidence_types.append(ev['type'])
    
    type_counts = Counter(evidence_types)
    
    if not type_counts:
        insights.append("\nThe analysis lacks classified evidence types according to Van Evera's tests (hoop, smoking gun, double-decisive, straw-in-wind), making it difficult to assess the probative value of the evidence.")
    else:
        insights.append("\nEvidence assessment based on Van Evera's tests:")
        
        if type_counts.get('double_decisive', 0) > 0:
            insights.append("- The presence of double-decisive evidence provides strong support for causal inferences")
        
        if type_counts.get('smoking_gun', 0) > 0:
            insights.append("- Smoking gun evidence confirms certain hypotheses, though alternatives might remain viable")
        
        if type_counts.get('hoop', 0) > type_counts.get('smoking_gun', 0) + type_counts.get('double_decisive', 0):
            insights.append("- The analysis relies heavily on hoop tests, which help eliminate alternatives but don't definitively confirm hypotheses")
        
        if type_counts.get('straw_in_wind', 0) > 0:
            insights.append("- Straw-in-wind evidence provides weak probative value and should be supplemented")
    
    # Evaluate alternative explanations
    insights.append(f"\n## 4. Alternative Explanation Evaluation")
    
    if results['alternatives']:
        strong_alts = [a for a in results['alternatives'] if a['strength_score'] > 0]
        if strong_alts:
            insights.append("\nThe presence of plausible alternative explanations suggests caution in causal inference. Per Bayesian logic, the posterior confidence in the primary explanation should be adjusted based on these competing accounts.")
        else:
            insights.append("\nThe analysis effectively considers and refutes alternative explanations, strengthening confidence in the primary causal story.")
    else:
        insights.append("\nThe analysis lacks consideration of alternative explanations, a significant methodological weakness. Process tracing gains strength from explicit comparison with competing accounts.")
    
    # Evaluate conditions
    insights.append(f"\n## 5. Scope Conditions Analysis")
    
    if results['conditions']['enabling']:
        insights.append("\nThe analysis properly identifies enabling conditions, enhancing the specificity of causal claims by clarifying when the identified mechanisms operate.")
    else:
        insights.append("\nThe analysis lacks clear specification of scope conditions, limiting the generalizability of findings and precision of causal claims.")
    
    # Overall methodological assessment
    insights.append(f"\n## 6. Overall Methodological Assessment")
    
    # Calculate a simple score based on various metrics
    method_score = 0
    
    # Points for causal chains
    method_score += 3 if results['causal_chains'] else 0
    method_score += 2 if results['causal_chains'] and max([c['length'] for c in results['causal_chains']]) > 5 else 0
    
    # Points for mechanisms
    method_score += 2 if results['mechanisms'] else 0
    method_score += 2 if avg_completeness > 50 else 0
    
    # Points for evidence
    method_score += 2 if results['evidence_analysis'] else 0
    method_score += 1 if type_counts.get('double_decisive', 0) > 0 else 0
    method_score += 1 if type_counts.get('smoking_gun', 0) > 0 else 0
    
    # Points for alternatives
    method_score += 2 if results['alternatives'] else 0
    
    # Points for conditions
    method_score += 2 if results['conditions']['enabling'] else 0
    
    # Final assessment
    if method_score >= 12:
        insights.append("\nThis process tracing analysis demonstrates strong methodological rigor, with well-specified causal chains, articulated mechanisms, and appropriate evidence assessment.")
    elif method_score >= 8:
        insights.append("\nThis process tracing analysis shows moderate methodological quality. While key causal relationships are identified, the analysis would benefit from more rigorous specification of mechanisms and evidence classification.")
    else:
        insights.append("\nThis process tracing analysis exhibits several methodological weaknesses common to the approach. Strengthening the specification of mechanisms, evidence classification, and consideration of alternatives would enhance causal inference.")
    
    # Future research recommendations
    insights.append(f"\n## 7. Methodological Recommendations")
    insights.append("\nTo strengthen the process tracing methodology:")
    
    if not results['causal_chains'] or (results['causal_chains'] and max([c['length'] for c in results['causal_chains']]) < 5):
        insights.append("- Develop more comprehensive causal chains to connect initial conditions to outcomes")
    
    if avg_completeness < 60:
        insights.append("- Specify causal mechanisms in greater detail, articulating how exactly causes produce effects")
    
    if not type_counts.get('double_decisive', 0) and not type_counts.get('smoking_gun', 0):
        insights.append("- Seek stronger evidence with higher probative value (smoking gun or double-decisive tests)")
    
    if not results['alternatives']:
        insights.append("- Explicitly consider and test alternative explanations to strengthen inference")
    
    if not results['conditions']['enabling']:
        insights.append("- Identify scope conditions to specify when and where the causal mechanisms operate")
    
    return "\n".join(insights)

def generate_node_type_chart(results):
    """Generate a pie chart of node types"""
    plt.figure(figsize=(10, 6))
    plt.pie(
        results['metrics']['node_type_distribution'].values(),
        labels=results['metrics']['node_type_distribution'].keys(),
        autopct='%1.1f%%',
        shadow=True
    )
    plt.title('Node Type Distribution')
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64 string
    return base64.b64encode(image_png).decode('utf-8')

def generate_edge_type_chart(results):
    """Generate a bar chart of edge types"""
    plt.figure(figsize=(12, 6))
    edge_types = results['metrics']['edge_type_distribution']
    plt.bar(edge_types.keys(), edge_types.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Edge Type Distribution')
    plt.tight_layout()
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64 string
    return base64.b64encode(image_png).decode('utf-8')

def generate_causal_chain_network(G, causal_chains):
    """Generate a visualization of the top causal chain"""
    if not causal_chains:
        return None
    
    # Take the longest causal chain
    chain = causal_chains[0]
    
    # Create a subgraph with only the nodes in the chain
    subG = G.subgraph(chain['path'])
    
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(subG)
    
    # Draw nodes
    nx.draw_networkx_nodes(subG, pos, node_size=1000, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(subG, pos, width=2, arrows=True)
    
    # Draw labels
    node_labels = {}
    for node in subG.nodes():
        label = G.nodes[node].get('label', node)
        # Truncate long labels
        if len(label) > 20:
            label = label[:17] + "..."
        node_labels[node] = label
    
    nx.draw_networkx_labels(subG, pos, labels=node_labels, font_size=8)
    
    plt.title("Top Causal Chain Visualization")
    plt.axis('off')
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64 string
    return base64.b64encode(image_png).decode('utf-8')

def generate_centrality_chart(results):
    """Generate a horizontal bar chart for node centrality"""
    plt.figure(figsize=(12, 8))
    
    nodes = list(results['metrics']['degree_centrality'].keys())[:10]
    values = [results['metrics']['degree_centrality'][node] for node in nodes]
    
    plt.barh(nodes, values, color='skyblue')
    plt.xlabel('Degree Centrality')
    plt.title('Top 10 Nodes by Centrality')
    plt.tight_layout()
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64 string
    return base64.b64encode(image_png).decode('utf-8')

def generate_evidence_strength_chart(results):
    """Generate a chart showing evidence strength by hypothesis"""
    if not results['evidence_analysis']:
        return None
    
    plt.figure(figsize=(12, 6))
    
    hypotheses = []
    supporting = []
    refuting = []
    
    for hyp_id, analysis in results['evidence_analysis'].items():
        hypotheses.append(hyp_id)
        supporting.append(len(analysis['supporting_evidence']))
        refuting.append(len(analysis['refuting_evidence']))
    
    x = range(len(hypotheses))
    width = 0.35
    
    plt.bar(x, supporting, width, label='Supporting Evidence', color='green')
    plt.bar([i + width for i in x], refuting, width, label='Refuting Evidence', color='red')
    
    plt.xlabel('Hypotheses')
    plt.ylabel('Number of Evidence Items')
    plt.title('Evidence Strength by Hypothesis')
    plt.xticks([i + width/2 for i in x], [h[:10] + "..." if len(h) > 10 else h for h in hypotheses])
    plt.legend()
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to base64 string
    return base64.b64encode(image_png).decode('utf-8')

def main():
    args = parse_args()
    if not os.path.isfile(args.json_file):
        print(f"Error: File not found: {args.json_file}"); sys.exit(1)
    
    print(f"📊 Analyzing data from {os.path.basename(args.json_file)}...")
    try:
        G, data = load_graph(args.json_file)
        data['filename'] = args.json_file
        print(f"✅ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading graph: {e}"); sys.exit(1)
    
    results = {
        'causal_chains': identify_causal_chains(G),
        'mechanisms': evaluate_mechanisms(G),
        'evidence_analysis': analyze_evidence(G),
        'conditions': identify_conditions(G),
        'actors': analyze_actors(G),
        'alternatives': analyze_alternative_explanations(G),
        'metrics': calculate_network_metrics(G)
    }
    
    theoretical_insights = None
    if args.theory or args.html: # HTML implies theory
        print("🔍 Generating theoretical insights...")
        theoretical_insights = generate_theoretical_insights(results, data)
    
    analysis_text = ""
    output_extension = "md"
    if args.html:
        analysis_text = format_html_analysis(results, data, G, theoretical_insights)
        output_extension = "html"
    else:
        analysis_text = format_analysis(results, data, G, theoretical_insights)
    
    output_path_str = args.output
    if not output_path_str:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        output_path_str = f"{base_name}_analysis.{output_extension}"
    else:
        # Ensure correct extension if user provides one
        name, ext = os.path.splitext(output_path_str)
        if not ext.lower() == f".{output_extension}":
            output_path_str = name + f".{output_extension}"
    
    output_path = Path(output_path_str)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        print(f"✅ Analysis report saved to {output_path}")
    except Exception as e:
        print(f"Error writing report to {output_path}: {e}")
        if not args.html: print("\nANALYSIS CONTENT:\n" + analysis_text)
    
    print("\n✅ Analysis generation complete!")
    
    if not args.html and args.charts_dir:
        charts_output_dir = Path(args.charts_dir)
        charts_output_dir.mkdir(parents=True, exist_ok=True)
        base_chart_name = Path(args.json_file).stem
        print(f"📊 Generating PNG charts in {charts_output_dir}/...")
        try:
            # Node type distribution pie chart
            plt.figure(figsize=(10, 6))
            node_dist_data = results['metrics']['node_type_distribution']
            if node_dist_data:
                plt.pie(node_dist_data.values(), labels=node_dist_data.keys(), autopct='%1.1f%%')
                plt.title('Node Type Distribution')
                plt.savefig(charts_output_dir / f"{base_chart_name}_node_types.png")
            plt.close() # Close figure to free memory

            # Edge type distribution bar chart
            plt.figure(figsize=(12, 8))
            edge_dist_data = results['metrics']['edge_type_distribution']
            if edge_dist_data:
                plt.bar(edge_dist_data.keys(), edge_dist_data.values())
                plt.xticks(rotation=45, ha='right')
                plt.title('Edge Type Distribution')
                plt.tight_layout()
                plt.savefig(charts_output_dir / f"{base_chart_name}_edge_types.png")
            plt.close()
            print(f"✅ PNG charts saved.")
        except Exception as e:
            print(f"⚠️ Error generating PNG charts: {str(e)}")
    elif not args.html:
        print("ℹ️ PNG chart generation skipped: --charts-dir not specified.")
    
    sys.exit(0)

if __name__ == "__main__":
    main() 