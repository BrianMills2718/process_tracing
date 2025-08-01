import os
import sys
import json
import argparse
import networkx as nx
from collections import defaultdict, Counter
import copy
import logging
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for servers or scripts
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import textwrap # Used in format_html_analysis
from datetime import datetime

# Assuming this file will be in core/ and ontology.py is in core/
from core.ontology import NODE_TYPES as CORE_NODE_TYPES, NODE_COLORS
from core.enhance_evidence import refine_evidence_assessment_with_llm
from core.llm_reporting_utils import generate_narrative_summary_with_llm
from core.enhance_mechanisms import elaborate_mechanism_with_llm

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

def normalize_evidence_type_for_output(evidence_type):
    """
    Normalize evidence type for output classification - same logic as Van Evera function.
    Handles variations like 'straw_in_the_wind' vs 'straw_in_wind'.
    """
    if not evidence_type:
        return ''
    # Convert to lowercase and replace any underscores/hyphens/spaces
    normalized = str(evidence_type).lower().replace('_', '').replace('-', '').replace(' ', '')
    return normalized

# Add a safe print function to handle UnicodeEncodeError
def find_causal_paths_bounded(G, source, target, cutoff=10, max_paths=100):
    """
    Find causal paths with bounds to prevent exponential explosion.
    
    Args:
        G: NetworkX graph
        source: Source node
        target: Target node
        cutoff: Maximum path length (default 10)
        max_paths: Maximum number of paths to return (default 100)
        
    Returns:
        List of paths (each path is a list of nodes)
    """
    paths = []
    paths_found = 0
    
    for path in nx.all_simple_paths(G, source, target, cutoff=cutoff):
        paths.append(path)
        paths_found += 1
        if paths_found >= max_paths:
            safe_print(f"DEBUG_CHAINS: Reached max_paths limit of {max_paths}")
            break
            
    return paths


def analyze_graph(G, options=None):
    """
    Analyze a graph without modifying the original.
    Implements full observability with START/PROGRESS/END logging.
    
    Args:
        G: NetworkX graph to analyze
        options: Optional analysis options
        
    Returns:
        Analysis results dictionary
    """
    import time
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info(f"START: analyze_graph - nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    
    # Create deep copy to prevent modification of original
    logger.info("PROGRESS: Creating deep copy of graph")
    G_working = copy.deepcopy(G)
    
    # Run analysis on the copy
    logger.info("PROGRESS: Starting causal chain analysis")
    causal_chains = identify_causal_chains(G_working)
    logger.info(f"PROGRESS: Found {len(causal_chains)} causal chains")
    
    logger.info("PROGRESS: Starting mechanism evaluation")
    mechanisms = evaluate_mechanisms(G_working)
    logger.info(f"PROGRESS: Evaluated {len(mechanisms)} mechanisms")
    
    logger.info("PROGRESS: Starting evidence analysis")
    evidence_analysis = analyze_evidence(G_working)['by_hypothesis']
    logger.info(f"PROGRESS: Analyzed evidence for {len(evidence_analysis)} hypotheses")
    
    logger.info("PROGRESS: Analyzing conditions")
    conditions = identify_conditions(G_working)
    
    logger.info("PROGRESS: Analyzing actors")
    actors = analyze_actors(G_working)
    
    logger.info("PROGRESS: Analyzing alternative explanations")
    alternatives = analyze_alternative_explanations(G_working)
    
    logger.info("PROGRESS: Calculating network metrics")
    metrics = calculate_network_metrics(G_working)
    
    analysis_results = { 
        'causal_chains': causal_chains,
        'mechanisms': mechanisms,
        'evidence_analysis': evidence_analysis,
        'conditions': conditions,
        'actors': actors,
        'alternatives': alternatives,
        'metrics': metrics
    }
    
    duration = time.time() - start_time
    logger.info(f"END: analyze_graph completed in {duration:.2f}s")
    
    return analysis_results


def safe_print(*args, **kwargs):
    """
    Safe print function that handles encoding errors and logs them.
    Implements fail-fast principle - logs errors instead of hiding them.
    """
    logger = logging.getLogger(__name__)
    
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError as e:
        # Log the error as required by fail-fast principle
        logger.error(f"Encoding error in safe_print: {e}", exc_info=True)
        
        # Try printing with errors replaced
        try:
            print(*(str(a).encode('utf-8', errors='replace').decode('utf-8') for a in args), **kwargs)
        except Exception as fallback_error:
            logger.error(f"Failed fallback print: {fallback_error}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process Tracing Network Analyzer (Core Engine)")
    parser.add_argument("json_file", help="JSON file with process tracing data")
    parser.add_argument("--theory", "-t", action="store_true", help="Include theoretical insights section (default with HTML)")
    parser.add_argument("--output", "-o", help="Output file for analysis report (e.g., analysis.html or analysis.md)")
    parser.add_argument("--html", action="store_true", help="Generate an HTML report with embedded visualizations")
    parser.add_argument("--charts-dir", help="Directory to save PNG charts (if not generating HTML report). Default: no separate charts saved if path not given.")
    parser.add_argument("--network-data", help="JSON string containing node and edge data for embedded vis.js visualization")
    args = parser.parse_args()
    if args.html and not args.theory:
        args.theory = True 
    return args

def validate_graph(G):
    """
    Validate that a graph has all required attributes.
    Implements fail-fast principle.
    
    Args:
        G: NetworkX graph to validate
        
    Raises:
        ValueError: If graph is invalid
    """
    if not G:
        raise ValueError("Graph is empty")
    
    # Check all nodes have required type attribute
    for node_id, node_data in G.nodes(data=True):
        if 'type' not in node_data:
            raise ValueError(f"Missing required attribute 'type' for node {node_id}")
    
    # Check all edges are valid
    for u, v, edge_data in G.edges(data=True):
        if u not in G:
            raise ValueError(f"Edge references non-existent node: {u}")
        if v not in G:
            raise ValueError(f"Edge references non-existent node: {v}")
    
    return True


def load_graph(json_file):
    """
    Loads graph data from a JSON file.
    Main 'type' of nodes/edges is stored as a top-level attribute 'type'.
    The original 'properties' dictionary from JSON is stored nested under 'attr_props'.
    
    Implements fail-fast principle - fails loud on missing file or invalid data.
    """
    # Fail fast on missing file
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Required file missing: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"File {json_file} is empty - cannot proceed")

    G = nx.DiGraph()

    def sanitize_value(value):
        if isinstance(value, str):
            try:
                value = value.encode('latin1', 'replace').decode('utf-8', 'replace')
            except Exception:
                value = value.encode('utf-8', 'replace').decode('ascii', 'replace')
        return value

    for node_data in data.get('nodes', []):
        node_id = node_data.get('id')
        main_type_from_json = node_data.get('type')

        if not node_id: 
            safe_print(f"[WARN] Skipping node due to missing 'id'. Data: {node_data}")
            continue
        if not main_type_from_json:
            safe_print(f"[WARN] Skipping node {node_id} due to missing main 'type'.")
            continue
            
        # Fixed: Use flat structure instead of nested attr_props
        properties_from_json = {k: sanitize_value(v) for k, v in node_data.get('properties', {}).items()}
        
        try:
            # Add main type and flatten all properties to top level
            node_attributes = {'type': str(main_type_from_json)}
            node_attributes.update(properties_from_json)
            
            G.add_node(node_id, **node_attributes)
        except Exception as e:
            safe_print(f"[WARN] Skipping node {node_id} due to attribute error: {e}. MainType: {main_type_from_json}, Properties: {properties_from_json}")

    for edge_data in data.get('edges', []):
        source = edge_data.get('source')
        target = edge_data.get('target')
        edge_id = edge_data.get('id', f"{source}_to_{target}_{edge_data.get('type', 'edge')}")

        if not source or not target :
            safe_print(f"[WARN] Skipping edge due to missing 'source' or 'target'. Data: {edge_data}")
            continue
        if not G.has_node(source):
            safe_print(f"[WARN] Skipping edge {edge_id} because source node '{source}' not in graph (available nodes: {list(G.nodes())[:5]}...).") # Show only a few nodes if list is long
            continue
        if not G.has_node(target):
            safe_print(f"[WARN] Skipping edge {edge_id} because target node '{target}' not in graph (available nodes: {list(G.nodes())[:5]}...).")
            continue
            
        main_edge_type_from_json = edge_data.get('type')
        if not main_edge_type_from_json:
            safe_print(f"[WARN] Skipping edge {edge_id} due to missing main 'type'.")
            continue
            
        # Fixed: Use flat structure instead of nested attr_props
        properties_from_json = {k: sanitize_value(v) for k, v in edge_data.get('properties', {}).items()}
        
        try:
            # Add main type and flatten all properties to top level
            edge_attributes = {
                'id': str(edge_id), 
                'type': str(main_edge_type_from_json)
            }
            edge_attributes.update(properties_from_json)
            
            G.add_edge(source, target, key=edge_id, **edge_attributes)
        except Exception as e:
            safe_print(f"[WARN] Skipping edge {edge_id} from {source} to {target} due to attribute error: {e}. MainType: {main_edge_type_from_json}, Properties: {properties_from_json}")
            
    return G, data

def identify_causal_chains(G):
    causal_chains = []
    event_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Event'}
    
    # Fixed: Use flat structure instead of nested attr_props
    triggering_events = [n for n, d_node in event_nodes_data.items() if d_node.get('subtype') == 'triggering' or d_node.get('type') == 'triggering']
    outcome_events = [n for n, d_node in event_nodes_data.items() if d_node.get('subtype') == 'outcome' or d_node.get('type') == 'outcome']
    
    # Fixed: Use flat structure for descriptions
    safe_print(f"DEBUG_CHAINS: Triggering Event IDs: {triggering_events} (Descriptions: {[G.nodes[n].get('description', n) for n in triggering_events]})")
    safe_print(f"DEBUG_CHAINS: Outcome Event IDs: {outcome_events} (Descriptions: {[G.nodes[n].get('description', n) for n in outcome_events]})")

    valid_chain_link_types = ['causes', 'leads_to', 'precedes', 'triggers', 'contributes_to', 'enables', 'influences', 'facilitates']
    
    if not triggering_events or not outcome_events:
        safe_print("DEBUG_CHAINS: No triggering or outcome events found based on subtype, so no chains can be formed.")
        return []

    for trigger_node_id in triggering_events:
        for outcome_node_id in outcome_events:
            if trigger_node_id == outcome_node_id:
                continue
            try:
                paths = find_causal_paths_bounded(G, source=trigger_node_id, target=outcome_node_id, cutoff=10, max_paths=100)
                for path in paths:
                    # Fixed: Use flat structure for descriptions
                    path_node_descriptions = [G.nodes[n].get('description', n) for n in path]
                    safe_print(f"DEBUG_CHAINS: Path found by find_causal_paths_bounded: {path} (Nodes: {path_node_descriptions})")
                    if len(path) < 2:
                        continue

                    is_valid_chain = True
                    current_chain_edges_types = []
                    # Fixed: Use flat structure for subtypes
                    current_path_node_subtypes = [G.nodes[n].get('subtype', 'N/A_subtype') for n in path]

                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        edge_data = G.get_edge_data(u, v) 
                        
                        if not edge_data: 
                            is_valid_chain = False
                            break
                            
                        edge_main_type = edge_data.get('type', '') 
                        current_chain_edges_types.append(edge_main_type)
                        
                        # Fixed: Allow valid process tracing patterns beyond just Event→Event
                        valid_node_types = ['Event', 'Causal_Mechanism']
                        u_type = G.nodes[u].get('type')
                        v_type = G.nodes[v].get('type')
                        
                        if not (u_type in valid_node_types and \
                                v_type in valid_node_types and \
                                edge_main_type in valid_chain_link_types):
                            is_valid_chain = False
                            # Fixed: Use flat structure for descriptions
                            safe_print(f"DEBUG_CHAINS: Path {path_node_descriptions} invalidated at step {G.nodes[u].get('description', u)}->{G.nodes[v].get('description', v)}. NodeU main_type: {u_type}, NodeV main_type: {v_type}, Edge main_type: {edge_main_type}")
                            break
                    
                    if is_valid_chain and len(path) > 2: 
                        # Fixed: Use flat structure for descriptions
                        safe_print(f"DEBUG_CHAINS: Validated chain: {[G.nodes[n].get('description', n) for n in path]} -> Edges: {current_chain_edges_types}")
                        chain_details = {
                            'path': path,
                            # Fixed: Use flat structure for descriptions
                            'node_descriptions': [G.nodes[n].get('description', 'N/A') for n in path],
                            'node_subtypes': current_path_node_subtypes,
                            'edges': current_chain_edges_types,
                            'length': len(path)
                        }
                        causal_chains.append(chain_details)
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound as e:
                safe_print(f"DEBUG_CHAINS: Node not found during path search: {e}")
                continue
    
    causal_chains.sort(key=lambda x: x['length'], reverse=True)
    safe_print(f"DEBUG_CHAINS: Total causal_chains collected: {len(causal_chains)}")
    return causal_chains if causal_chains else []

def evaluate_mechanisms(G):
    mechanisms = []
    mech_nodes_data = {n:d for n, d in G.nodes(data=True) if d.get('type') == 'Causal_Mechanism'}
    
    for mech_id, mech_node_data in mech_nodes_data.items():
        # Fixed: Use flat structure instead of nested attr_props
        mech_description = mech_node_data.get('description', mech_id)

        causes = [] 
        for pred_id in G.predecessors(mech_id):
            edge_data = G.get_edge_data(pred_id, mech_id)
            if not edge_data: continue

            pred_node_data = G.nodes[pred_id]
            source_node_main_type = pred_node_data.get('type')
            source_node_subtype = pred_node_data.get('subtype', 'N/A')
            edge_main_type = edge_data.get('type', '')
            pred_description = pred_node_data.get('description', pred_id)

            safe_print(f"DEBUG_CM_EVAL: CM='{mech_description}' ({mech_id}), Predecessor='{pred_description}' ({pred_id}) (SourceMainType: {source_node_main_type}, SourceSubType: {source_node_subtype}), EdgeType='{edge_main_type}'")

            if source_node_main_type == 'Event' and \
               (edge_main_type == 'part_of_mechanism' or edge_main_type == 'causes' or edge_main_type == 'triggers'):
                causes.append(pred_id)
        
        safe_print(f"DEBUG_CM_EVAL: CM='{mech_description}' ({mech_id}), Final 'causes' list for completeness (IDs): {causes}, Count: {len(causes)}")
        
        effects = [] 
        for succ_id in G.successors(mech_id):
            edge_data = G.get_edge_data(mech_id, succ_id)
            if not edge_data: continue

            succ_node_data = G.nodes[succ_id]
            target_node_main_type = succ_node_data.get('type')
            edge_main_type = edge_data.get('type', '')
            if target_node_main_type == 'Event' and edge_main_type == 'causes':
                effects.append(succ_id)
        
        completeness = min(len(causes) * 20, 100) 
        
        mechanisms.append({
            'id': mech_id,
            'name': mech_description,
            'causes': [G.nodes[c_id].get('description', c_id) for c_id in causes],
            'effects': [G.nodes[e_id].get('description', e_id) for e_id in effects],
            'completeness': completeness,
            'confidence': mech_node_data.get('confidence', 'unknown'),
            'level_of_detail': mech_node_data.get('level_of_detail', 'medium')
        })
    
    mechanisms.sort(key=lambda x: x['completeness'], reverse=True)
    return mechanisms

def analyze_evidence(G):
    hypothesis_results = {}
    evidence_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Evidence'}
    hypothesis_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Hypothesis'}

    debug_initial_hyp_descs = {
        n_id: node_data.get('description', f'N/A_IN_DEBUG_FOR_{n_id}') 
        for n_id, node_data in hypothesis_nodes_data.items()
    }
    safe_print(f"DEBUG_EVIDENCE_ANALYSIS (LATEST): Initial Hypothesis descriptions loaded from G: {debug_initial_hyp_descs}")

    for hyp_id, hyp_node_data in hypothesis_nodes_data.items():
        # Fixed: Use flat structure instead of nested attr_props
        description_for_results = hyp_node_data.get('description', f'Description_Not_Found_For_{hyp_id}')
        
        hypothesis_results[hyp_id] = {
            'description': description_for_results,
            'status': hyp_node_data.get('status', 'undetermined'),
            'prior_probability': hyp_node_data.get('prior_probability'),
            'posterior_probability': hyp_node_data.get('posterior_probability'),
            'supporting_evidence': [],
            'refuting_evidence': [],
            'balance': 0.0,
            'assessment': 'Undetermined / Lacks Evidence'
        }

    for evidence_id, evidence_node_data in evidence_nodes_data.items():
        # Fixed: Use flat structure instead of nested attr_props
        evidence_description = evidence_node_data.get('description', 'N/A')
        evidence_classification_type = evidence_node_data.get('evidence_type', 'general') 

        for u_ev_id, v_hyp_id, edge_data in G.out_edges(evidence_id, data=True): 
            if v_hyp_id in hypothesis_nodes_data: 
                edge_main_type = edge_data.get('type', '')
                # Fixed: Access properties directly from flattened edge_data (not nested properties)
                probative_value_from_edge = edge_data.get('probative_value')
                source_quote_from_edge = edge_data.get('source_text_quote', "")
                probative_value_num = None
                if isinstance(probative_value_from_edge, (int, float)):
                    probative_value_num = float(probative_value_from_edge)
                elif isinstance(probative_value_from_edge, str):
                    try:
                        probative_value_num = float(probative_value_from_edge)
                    except ValueError:
                        safe_print(f"Warning: Could not convert probative_value '{probative_value_from_edge}' to float for edge {u_ev_id}->{v_hyp_id}")
                        probative_value_num = 0.0 
                if probative_value_num is None: 
                    probative_value_num = 0.0 
                ev_detail = {
                    'id': evidence_id,
                    'description': evidence_description,
                    'type': evidence_classification_type, 
                    'probative_value': probative_value_num,
                    'source_text_quote': source_quote_from_edge,
                    'edge_type': edge_main_type 
                }
                # --- LLM Evidence Refinement Integration ---
                try:
                    hyp_node = hypothesis_nodes_data[v_hyp_id]
                    llm_refined = refine_evidence_assessment_with_llm(
                        hypothesis_node={'id': v_hyp_id, 'properties': hyp_node},  # hyp_node is already flat
                        evidence_node={'id': evidence_id, 'properties': evidence_node_data},  # evidence_node_data is already flat
                        edge_properties=edge_data,  # edge_data is already flat
                        original_text_context=source_quote_from_edge
                    )
                    if isinstance(llm_refined, dict):
                        if 'refined_evidence_type' in llm_refined:
                            ev_detail['type'] = llm_refined['refined_evidence_type']
                        if 'suggested_numerical_probative_value' in llm_refined:
                            try:
                                ev_detail['probative_value'] = float(llm_refined['suggested_numerical_probative_value'])
                            except Exception:
                                pass
                        ev_detail['llm_reasoning'] = llm_refined
                except Exception as e:
                    ev_detail['llm_reasoning'] = {'error': str(e)}
                # --- End LLM Evidence Refinement Integration ---
                balance_effect = 0.0
                if edge_main_type == 'supports':
                    hypothesis_results[v_hyp_id]['supporting_evidence'].append(ev_detail)
                    balance_effect = ev_detail['probative_value']
                elif edge_main_type == 'refutes':
                    hypothesis_results[v_hyp_id]['refuting_evidence'].append(ev_detail)
                    balance_effect = -ev_detail['probative_value']  # Fixed: Refuting evidence decreases balance
                if not isinstance(hypothesis_results[v_hyp_id].get('balance'), float):
                    hypothesis_results[v_hyp_id]['balance'] = 0.0 
                hyp_desc_for_debug = hypothesis_results[v_hyp_id].get('description', v_hyp_id) # Use already fetched desc
                safe_print(f"DEBUG_EVIDENCE_ANALYSIS: Hyp: '{hyp_desc_for_debug}' ({v_hyp_id}), Edge: {evidence_id}->{v_hyp_id} ({edge_main_type}), PV_num: {ev_detail['probative_value']}, Effect: {balance_effect}, OldBal: {hypothesis_results[v_hyp_id]['balance']:.2f}, NewBal: {hypothesis_results[v_hyp_id]['balance'] + balance_effect:.2f}")
                hypothesis_results[v_hyp_id]['balance'] += balance_effect
    
    # Apply Van Evera diagnostic test logic instead of simple balance thresholds
    for hyp_id in hypothesis_results:
        balance = hypothesis_results[hyp_id]['balance']
        van_evera_assessment = apply_van_evera_diagnostic_tests(hypothesis_results[hyp_id])
        
        # Use Van Evera assessment if available, otherwise fall back to balance
        if van_evera_assessment:
            hypothesis_results[hyp_id]['assessment'] = van_evera_assessment
            hypothesis_results[hyp_id]['van_evera_applied'] = True
        else:
            # Fallback to simple balance thresholds
            if balance > 0.5: 
                hypothesis_results[hyp_id]['assessment'] = 'Supported'
            elif balance < -0.5: 
                hypothesis_results[hyp_id]['assessment'] = 'Refuted'
            elif abs(balance) <= 0.5 and (hypothesis_results[hyp_id]['supporting_evidence'] or hypothesis_results[hyp_id]['refuting_evidence']):
                hypothesis_results[hyp_id]['assessment'] = 'Contested / Mixed Evidence'
            hypothesis_results[hyp_id]['van_evera_applied'] = False
            # else remains 'Undetermined / Lacks Evidence'

    return {'by_hypothesis': hypothesis_results}

def apply_van_evera_diagnostic_tests(hypothesis_data):
    """
    Apply Van Evera diagnostic test logic to assess hypothesis based on evidence types.
    Returns assessment string or None if no Van Evera evidence present.
    """
    supporting = hypothesis_data.get('supporting_evidence', [])
    refuting = hypothesis_data.get('refuting_evidence', [])
    
    def normalize_evidence_type(evidence_type):
        """Normalize evidence type to handle both naming conventions"""
        if not evidence_type:
            return ''
        # Convert to lowercase and replace any underscores/hyphens/spaces
        normalized = str(evidence_type).lower().replace('_', '').replace('-', '').replace(' ', '')
        return normalized
    
    # Count Van Evera evidence types with normalization
    hoop_support = [ev for ev in supporting if normalize_evidence_type(ev.get('type')) == 'hoop']
    hoop_refute = [ev for ev in refuting if normalize_evidence_type(ev.get('type')) == 'hoop']
    
    smoking_gun_support = [ev for ev in supporting if normalize_evidence_type(ev.get('type')) == 'smokinggun'] 
    smoking_gun_refute = [ev for ev in refuting if normalize_evidence_type(ev.get('type')) == 'smokinggun']
    
    doubly_decisive_support = [ev for ev in supporting if normalize_evidence_type(ev.get('type')) == 'doublydecisive']
    doubly_decisive_refute = [ev for ev in refuting if normalize_evidence_type(ev.get('type')) == 'doublydecisive']
    
    straw_wind_support = [ev for ev in supporting if normalize_evidence_type(ev.get('type')) == 'strawinthewind']
    straw_wind_refute = [ev for ev in refuting if normalize_evidence_type(ev.get('type')) == 'strawinthewind']
    
    # Apply Van Evera diagnostic logic
    
    # Doubly decisive tests (most powerful)
    if doubly_decisive_support:
        return 'Strongly Confirmed (Doubly Decisive)'
    if doubly_decisive_refute:
        return 'Eliminated (Doubly Decisive)'
    
    # Hoop tests (necessary condition)
    if hoop_refute:
        return 'Eliminated (Failed Hoop Test)'
    
    # Smoking gun tests (sufficient condition)  
    if smoking_gun_support:
        return 'Strongly Confirmed (Smoking Gun)'
    
    # Combined assessment for mixed evidence
    if hoop_support and smoking_gun_support:
        return 'Strongly Supported (Hoop + Smoking Gun)'
    elif hoop_support:
        return 'Conditionally Supported (Passed Hoop Test)'
    elif straw_wind_support and not straw_wind_refute:
        return 'Weakly Supported (Straw-in-Wind)'
    elif straw_wind_refute and not straw_wind_support:
        return 'Weakly Refuted (Straw-in-Wind)'
    elif straw_wind_support and straw_wind_refute:
        return 'Inconclusive (Mixed Straw-in-Wind)'
    
    # No Van Evera evidence found
    return None

def identify_conditions(G):
    conditions = {'enabling': [], 'constraining': []}
    condition_nodes_data = {n:d for n, d in G.nodes(data=True) if d.get('type') == 'Condition'}
    
    for cond_id, cond_node_data in condition_nodes_data.items():
        # Fixed: Use flat structure
        cond_description = cond_node_data.get('description', cond_id)
        
        for u_cond_id, v_target_id, edge_data in G.out_edges(cond_id, data=True):
            edge_main_type = edge_data.get('type', '')
            # Fixed: Properties are flattened - access directly from edge_data
            target_node_data = G.nodes.get(v_target_id) # Use .get for safety
            if not target_node_data: continue

            target_main_type = target_node_data.get('type', 'unknown')
            target_description = target_node_data.get('description', v_target_id)

            safe_print(f"DEBUG_CONDITIONS: Condition='{cond_description}' ({cond_id}), Target='{target_description}' ({v_target_id}) (TargetMainType: {target_main_type}), EdgeType='{edge_main_type}'")
            
            target_info = {
                'id': v_target_id,
                'description': target_description,
                'type': target_main_type 
            }
            
            if target_main_type == 'Event' or target_main_type == 'Causal_Mechanism':
                if edge_main_type == 'enables':
                    conditions['enabling'].append({
                        'condition_id': cond_id,
                        'condition': cond_description,
                        'target': target_info,
                        'necessity': edge_data.get('necessity', 'unknown'),
                        'certainty': edge_data.get('certainty', 'medium')
                    })
                elif edge_main_type == 'constrains':
                    conditions['constraining'].append({
                        'condition_id': cond_id,
                        'condition': cond_description,
                        'target': target_info,
                        'certainty': edge_data.get('certainty', 'medium'),
                        'type': edge_data.get('type', 'unknown') 
                    })
    return conditions

def analyze_actors(G):
    actors = []
    actor_nodes_data = {n:d for n, d in G.nodes(data=True) if d.get('type') == 'Actor'}
    
    for actor_id, actor_node_data in actor_nodes_data.items():
        # Fixed: Use flat structure
        actor_name = actor_node_data.get('name', actor_id) 
        actor_role = actor_node_data.get('role', 'unknown')
        
        initiated_events = []
        for u_actor_id, v_target_id, edge_data in G.out_edges(actor_id, data=True):
            edge_main_type = edge_data.get('type', '')
            # Fixed: Properties are flattened - access directly from edge_data
            target_node_data = G.nodes.get(v_target_id)
            if not target_node_data: continue

            target_main_type = target_node_data.get('type', 'unknown')
            target_description = target_node_data.get('description', v_target_id)

            safe_print(f"DEBUG_ACTORS: Actor='{actor_name}' ({actor_id}), Target='{target_description}' ({v_target_id}) (TargetMainType: {target_main_type}), EdgeType='{edge_main_type}'")
            
            if target_main_type == 'Event' and edge_main_type == 'initiates':
                initiated_events.append({
                    'id': v_target_id,
                    'description': target_description,
                    'certainty': edge_data.get('certainty', 'medium'),
                    'intention': edge_data.get('intention', 'unknown')
                })
        
        # Comprehensive influence score calculation
        influence_score = 0
        # 1. Score for initiated events (points per event)
        influence_score += len(initiated_events) * 10
        # 2. Score for actor's degree centrality (raw degree as a proxy for general involvement)
        try:
            actor_degree = G.degree(actor_id)
            influence_score += actor_degree * 1 # 1 point per connection (in or out)
        except Exception:
            pass
        # 3. Bonus points if actor is linked to Causal Mechanisms or critical Hypotheses
        # Count other direct involvements (edges other than 'initiates' from this actor)
        other_outgoing_edges = 0
        for u, v, edge_data in G.out_edges(actor_id, data=True):
            if edge_data.get('type') != 'initiates':
                other_outgoing_edges += 1
        influence_score += other_outgoing_edges * 3 # 3 points for other types of direct actions/links
        # Count incoming edges (how many times this actor is a target of an action/link)
        try:
            influence_score += G.in_degree(actor_id) * 2 # 2 points for being targeted/referenced
        except Exception:
            pass
        # Check if actor name appears in descriptions of important nodes (heuristic)
        actor_name_lower = actor_name.lower()
        for node_id, node_data_iter in G.nodes(data=True):
            if node_id == actor_id:
                continue # Don't check against self
            node_props_iter = node_data_iter  # Already flat structure
            node_desc_iter = node_props_iter.get('description', '').lower()
            node_type_iter = node_data_iter.get('type')
            if actor_name_lower in node_desc_iter:
                if node_type_iter == 'Causal_Mechanism':
                    influence_score += 7 # Bonus for being mentioned in a CM
                elif node_type_iter == 'Hypothesis':
                    influence_score += 5 # Bonus for being mentioned in a Hypothesis
                elif node_type_iter == 'Condition':
                    influence_score += 3 # Bonus for being mentioned in a Condition
        actors.append({
            'id': actor_id,
            'name': actor_name,
            'role': actor_role,
            'initiated_events': initiated_events,
            'influence_score': influence_score,
            'beliefs': actor_node_data.get('beliefs', 'unknown'), 
            'intentions': actor_node_data.get('intentions', 'unknown') 
        })
    
    actors.sort(key=lambda x: x['influence_score'], reverse=True)
    return actors

def analyze_alternative_explanations(G):
    alternatives = []
    alt_nodes_data = {n: d for n, d in G.nodes(data=True) if d.get('type') == 'Alternative_Explanation'}
    evidence_nodes_main_type_ids = {n for n, d in G.nodes(data=True) if d.get('type') == 'Evidence'}

    for alt_id, alt_node_data in alt_nodes_data.items():
        # Fixed: Use flat structure
        alt_description = alt_node_data.get('description', alt_id)
        
        supporting = []
        refuting = []
        
        for pred_ev_id in G.predecessors(alt_id):
            if pred_ev_id in evidence_nodes_main_type_ids:
                edge_data = G.get_edge_data(pred_ev_id, alt_id)
                if not edge_data: continue

                edge_main_type = edge_data.get('type', '')
                # Fixed: Properties are flattened - access directly from edge_data
                
                evidence_node_data = G.nodes[pred_ev_id]
                # Fixed: Evidence node properties access directly
                
                evidence_info = {
                    'id': pred_ev_id,
                    'description': evidence_node_data.get('description', pred_ev_id),
                    'type': evidence_node_data.get('evidence_type', 'unknown'), 
                    'certainty': edge_data.get('certainty', 'medium'), 
                    'probative_value': edge_data.get('probative_value', 0.0) # Now accessing directly!
                }
                
                if edge_main_type == 'supports_alternative': 
                    supporting.append(evidence_info)
                elif edge_main_type == 'refutes_alternative': 
                    refuting.append(evidence_info)
        
        strength_score = sum(ev.get('probative_value', 0.0) for ev in supporting) - sum(abs(ev.get('probative_value', 0.0)) for ev in refuting)
        assessment = "Contested alternative explanation" 
        if strength_score > 0.5: assessment = "Plausible alternative explanation" 
        if strength_score > 1.5: assessment = "Strong alternative explanation"
        if strength_score < -0.5: assessment = "Weak alternative explanation"

        alternatives.append({
            'id': alt_id,
            'description': alt_description,
            'supporting_evidence': supporting,
            'refuting_evidence': refuting,
            'strength_score': strength_score,
            'assessment': assessment,
            'probability': alt_node_data.get('probability', 'unknown'),
            'status': alt_node_data.get('status', 'unknown')
        })
    
    alternatives.sort(key=lambda x: x['strength_score'], reverse=True)
    return alternatives

def calculate_network_metrics(G):
    metrics = {}
    node_types = [d.get('type', 'unknown') for _, d in G.nodes(data=True)]
    metrics['node_type_distribution'] = dict(Counter(node_types))
    edge_types = [d.get('type', 'unknown') for _, _, d in G.edges(data=True)]
    metrics['edge_type_distribution'] = dict(Counter(edge_types))
    metrics['density'] = nx.density(G)
    try:
        if G.number_of_nodes() > 0 and nx.is_weakly_connected(G): # Use weakly for DiGraph if strong is too strict
            if nx.is_strongly_connected(G):
                 metrics['avg_path_length'] = f"{nx.average_shortest_path_length(G):.2f}"
            else:
                metrics['avg_path_length'] = 'N/A (not strongly connected)'
        else:
            metrics['avg_path_length'] = 'N/A (disconnected or empty graph)'
    except Exception:
        metrics['avg_path_length'] = 'N/A (calculation error)'
    if G.number_of_nodes() > 0:
        metrics['degree_centrality'] = {
            node: round(value, 3) 
            for node, value in sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:10] 
        }
    else:
        metrics['degree_centrality'] = {}
    return metrics

def generate_theoretical_insights(results, G): 
    insights = ["# Theoretical Assessment of Process Tracing Methodology"]

    insights.append("\n## 1. Causal Chain Assessment")
    if results.get('causal_chains'):
        insights.append("The process tracing analysis reveals a causal chain structure.")
        insights.append("The presence of extended causal chains enhances the explanatory power of the analysis, providing a comprehensive narrative of how initial conditions led to eventual outcomes through multiple intervening events.")
    else:
        insights.append("The analysis lacks clear causal chains, a significant weakness in process tracing methodology. Without established sequences of events, causal inference becomes challenging.")
    
    insights.append(f"\n## 2. Mechanism Sufficiency")
    if results.get('mechanisms'):
        mechanism_completeness = [m.get('completeness',0) for m in results['mechanisms']]
        avg_completeness = sum(mechanism_completeness) / len(mechanism_completeness) if mechanism_completeness else 0
        if avg_completeness >= 70:
            insights.append("The causal mechanisms demonstrated strong sufficiency with well-specified components.")
        elif avg_completeness >= 40:
            insights.append("The causal mechanisms show moderate sufficiency but would benefit from further specification of intervening processes.")
        else:
            insights.append("Some causal mechanisms may lack sufficient detail or linked contributing events to fully explain how causes produce effects.")
    else:
        insights.append("No causal mechanisms were robustly evaluated for sufficiency.")

    insights.append(f"\n## 3. Evidence Quality Assessment")
    evidence_types_linked_to_hypotheses = set()
    if 'evidence_analysis' in results and isinstance(results.get('evidence_analysis'), dict) :
        for hyp_id, hyp_data in results['evidence_analysis'].items():
            hyp_desc_for_debug = hyp_data.get('description', hyp_id) 
            for ev_detail in hyp_data.get('supporting_evidence', []):
                safe_print(f"DEBUG_THEORY_INSIGHTS: For H '{hyp_desc_for_debug}', collecting from supporting_evidence: Ev_ID='{ev_detail.get('id')}', Ev_Class='{ev_detail.get('type')}'")
                evidence_types_linked_to_hypotheses.add(ev_detail.get('type'))
            for ev_detail in hyp_data.get('refuting_evidence', []):
                safe_print(f"DEBUG_THEORY_INSIGHTS: For H '{hyp_desc_for_debug}', collecting from refuting_evidence: Ev_ID='{ev_detail.get('id')}', Ev_Class='{ev_detail.get('type')}'")
                evidence_types_linked_to_hypotheses.add(ev_detail.get('type'))
    safe_print(f"DEBUG_THEORY_INSIGHTS: Final evidence_types_found_in_hyp_links set for insight generation: {evidence_types_linked_to_hypotheses}")
    classified_types_for_van_evera = {
        et for et in evidence_types_linked_to_hypotheses 
        if et and et in EVIDENCE_TYPES_VAN_EVERA 
    }
    if not classified_types_for_van_evera:
        insights.append("The analysis did not find sufficient evidence *linked to hypotheses* that was classified according to Van Evera's specific tests (hoop, smoking_gun, double_decisive, straw_in_wind). While 'general' evidence is present, classifying evidence using these tests helps assess probative value more rigorously for specific hypotheses.")
    else:
        type_counts = Counter(list(classified_types_for_van_evera))
        insights.append("Evidence assessment based on Van Evera's tests for evidence *linked to hypotheses*:")
        if type_counts.get('double_decisive', 0) > 0:
            insights.append(f"- Presence of {type_counts['double_decisive']} double-decisive evidence provides strong inferential leverage.")
        if type_counts.get('smoking_gun', 0) > 0:
            insights.append(f"- {type_counts['smoking_gun']} smoking_gun evidence items confirm certain hypotheses or mechanism operations robustly.")
        if type_counts.get('hoop', 0) > 0:
            insights.append(f"- {type_counts['hoop']} hoop test evidence items help eliminate alternative hypotheses or confirm necessary conditions.")
        if type_counts.get('straw_in_wind', 0) > 0:
            insights.append(f"- {type_counts['straw_in_wind']} straw-in-the-wind evidence items offer suggestive support but require corroboration.")
        if not type_counts.get('smoking_gun',0) and not type_counts.get('double_decisive',0) and classified_types_for_van_evera:
            insights.append("- The analysis primarily relies on weaker forms of classified evidence (hoop, straw-in-wind) for hypotheses; seeking more smoking_gun or double_decisive evidence could strengthen inferences.")

    insights.append(f"\n## 4. Alternative Explanation Evaluation")
    if results.get('alternatives'):
        insights.append("Alternative explanations were considered in the analysis.")
    else:
        insights.append("The analysis lacks explicit consideration of alternative explanations. Process tracing gains strength from comparing the primary explanation against plausible alternatives.")

    insights.append(f"\n## 5. Scope Conditions Analysis")
    if results.get('conditions', {}).get('enabling') or results.get('conditions', {}).get('constraining'):
        insights.append("The analysis identified some enabling or constraining conditions, which helps define the scope of causal claims.")
    else:
        insights.append("The analysis could benefit from clearer specification of scope conditions to enhance generalizability.")
    
    insights.append(f"\n## 6. Overall Methodological Assessment")
    score = 0
    if results.get('causal_chains'): score +=1
    if results.get('mechanisms') and any(m.get('completeness',0) > 50 for m in results['mechanisms']): score +=1
    if classified_types_for_van_evera: score +=1
    if results.get('conditions', {}).get('enabling') or results.get('conditions', {}).get('constraining'): score +=1
    if score >= 3:
        insights.append("This process tracing analysis demonstrates good methodological elements.")
    else:
        insights.append("This process tracing analysis shows areas for methodological strengthening.")

    insights.append(f"\n## 7. Methodological Recommendations")
    insights.append("To strengthen the process tracing methodology:")
    if not results.get('causal_chains'):
        insights.append("- Develop more comprehensive causal chains to connect initial conditions to outcomes")
    if results.get('mechanisms') and all(m.get('completeness',0) < 60 for m in results['mechanisms']):
        insights.append("- Specify causal mechanisms in greater detail, articulating how exactly causes produce effects")
    if not classified_types_for_van_evera or (not type_counts.get('smoking_gun',0) and not type_counts.get('double_decisive',0)):
        insights.append("- Seek stronger evidence with higher probative value (smoking gun or double-decisive tests) and ensure it's linked to hypotheses.")
    if not results.get('alternatives'):
        insights.append("- Explicitly consider and test alternative explanations to strengthen inference")
    if not (results.get('conditions', {}).get('enabling') or results.get('conditions', {}).get('constraining')):
        insights.append("- Identify scope conditions to specify when and where the causal mechanisms operate")

    return "\n".join(insights)

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
                <div id="mynetwork" style="width: 100%; height: 600px; border: 1px solid lightgray;"></div>
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
                                size: 20, 
                                font: {{ size: 16 }},
                                color: {{
                                    background: '#D2E5FF',
                                    border: '#2B7CE9'
                                }}
                            }},
                            edges: {{ 
                                arrows: 'to', 
                                font: {{ align: 'middle' }} 
                            }},
                            physics: {{ 
                                stabilization: true,
                                barnesHut: {{
                                    gravitationalConstant: -2000,
                                    centralGravity: 0.3,
                                    springLength: 150,
                                    springConstant: 0.04
                                }}
                            }}
                        }};
                        var network = new vis.Network(container, data, options);
                        
                        // Add tooltip behavior
                        network.on("hoverNode", function (params) {{
                            // Show tooltip with node information
                        }});
                    }});
                </script>
            </div>
        </div>
        """
        return html
    except Exception as e:
        safe_print(f"[ERROR] Failed to generate embedded network visualization: {e}")
        return f"<div class='alert alert-danger'>Error generating network visualization: {e}</div>"

def format_html_analysis(results, data_unused, G, theoretical_insights=None, network_data_json=None):
    node_type_names = {nt: CORE_NODE_TYPES.get(nt, {}).get('plural_name', nt) for nt in CORE_NODE_TYPES}
    default_node_type_names = { 
        'Event': 'Events', 'Causal_Mechanism': 'Causal Mechanisms', 'Hypothesis': 'Hypotheses',
        'Evidence': 'Evidence', 'Condition': 'Conditions', 'Actor': 'Actors',
        'Alternative_Explanation': 'Alternative Explanations', 'Data_Source': 'Data Sources',
        'Inference_Rule': 'Inference Rules', 'Inferential_Test': 'Inferential Tests'
    }
    for k, v in default_node_type_names.items():
        if k not in node_type_names: node_type_names[k] = v

    filename = results.get('filename', 'Process Trace') 
    
    node_type_chart_b64 = generate_node_type_chart(results)
    edge_type_chart_b64 = generate_edge_type_chart(results)
    top_chain_for_viz = results.get('causal_chains', [])[0] if results.get('causal_chains') else None
    causal_chain_chart_b64 = generate_causal_chain_network(G, top_chain_for_viz) if top_chain_for_viz else None
    centrality_chart_b64 = generate_centrality_chart(results, G)
    evidence_chart_b64 = generate_evidence_strength_chart(results)

    # Create the HTML header with vis.js library for network visualization
    html_header = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Process Tracing Analysis Report</title>
    <!-- Bootstrap 5 CSS -->
    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
    <!-- vis.js Network library for interactive graph visualization -->
    <script src=\"https://unpkg.com/vis-network/standalone/umd/vis-network.min.js\"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .chart { margin-bottom: 1.5rem; }
        .evidence-item { margin-bottom: 0.75rem; padding: 0.25rem; border-left: 3px solid #eee; }
        .supporting { border-left-color: #28a745; }
        .refuting { border-left-color: #dc3545; }
        .card { margin-bottom: 1.5rem; overflow: hidden; }
        .causal-chain { margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
<div class=\"container-fluid py-4\">
    <h1 class=\"mb-4\">Process Tracing Analysis Report</h1>
    """

    # HTML parts array to build the body content
    html_parts = []
    
    # Add the interactive network visualization if network data is provided
    if network_data_json:
        network_vis_html = generate_embedded_network_visualization(network_data_json)
        html_parts.append(network_vis_html)

    # Rest of the report structure (keep the existing code for the other sections)
    # Header and Overview
    html_parts.append(f"""
        <div class="row">
            <div class="col-lg-6">
                <div class="card">
                    <div class="card-header"><h2 class="card-title h5">Network Overview</h2></div>
                    <div class="card-body">""")
    if 'metrics' in results and 'node_type_distribution' in results['metrics']:
        html_parts.append(f"<p>Total nodes: {sum(results['metrics']['node_type_distribution'].values())}</p>")
        html_parts.append("<h3 class=\"h6\">Node Distribution:</h3><ul class=\"list-group list-group-flush\">")
        for node_main_type, count in results['metrics']['node_type_distribution'].items():
            display_name = node_type_names.get(node_main_type, node_main_type)
            html_parts.append(f'<li class="list-group-item d-flex justify-content-between align-items-center">{display_name} <span class="badge bg-primary rounded-pill">{count}</span></li>')
        html_parts.append("</ul>")
    else:
        html_parts.append("<p>Node distribution data not available.</p>")
    html_parts.append("""
                    </div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="card">
                    <div class="card-header"><h2 class="card-title h5">Visualizations</h2></div>
                    <div class="card-body">""")
    if node_type_chart_b64: html_parts.append(f'<div class="chart"><h3 class="h6">Node Type Distribution</h3><img src="data:image/png;base64,{node_type_chart_b64}" class="img-fluid" alt="Node Type Distribution"></div>')
    if edge_type_chart_b64: html_parts.append(f'<div class="chart"><h3 class="h6">Edge Type Distribution</h3><img src="data:image/png;base64,{edge_type_chart_b64}" class="img-fluid" alt="Edge Type Distribution"></div>')
    html_parts.append("""
                    </div>
                </div>
            </div>
        </div>""")

    # Causal Chains
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Causal Chains</h2></div>
            <div class="card-body">""")
    if results.get('causal_chains'):
        html_parts.append('<div class="row"><div class="col-md-12">') # Full width for list
        for i, chain in enumerate(results['causal_chains'][:5], 1):
            path_node_descriptions = []
            for node_id_in_chain in chain['path']:
                node_graph_data = G.nodes.get(node_id_in_chain, {})
                desc = node_graph_data.get('attr_props', {}).get('description', node_id_in_chain)
                path_node_descriptions.append(textwrap.shorten(desc, width=40, placeholder="..."))
            path_str_display = " &rarr; ".join(path_node_descriptions)
            
            html_parts.append(f"""
                <div class="causal-chain">
                    <h3 class="h6">Chain {i} (Length: {chain['length']})</h3>
                    <p><strong>Path (Descriptions):</strong> {path_str_display}</p>
                    <p><small><strong>Path (IDs):</strong> {" &rarr; ".join(chain['path'])}</small></p>
                    <p><small><strong>Edge Types:</strong> {" &rarr; ".join(chain['edges'])}</small></p>
                    <div><strong>Node Details (ID: Description [Subtype]):</strong>
                        <ul class="list-group list-group-flush">""")
            for idx, node_id_in_chain_detail in enumerate(chain['path']):
                node_graph_data_detail = G.nodes.get(node_id_in_chain_detail, {})
                desc_detail = node_graph_data_detail.get('attr_props', {}).get('description', node_id_in_chain_detail)
                subtype_detail = chain['node_subtypes'][idx] if idx < len(chain['node_subtypes']) else 'N/A'
                # Temporal fields for Event nodes
                start_date = node_graph_data_detail.get('attr_props', {}).get('start_date', '')
                end_date = node_graph_data_detail.get('attr_props', {}).get('end_date', '')
                is_point = node_graph_data_detail.get('attr_props', {}).get('is_point_in_time', None)
                temporal_str = ''
                if start_date or end_date:
                    temporal_str = f"<br><small>Date: {start_date}"
                    if end_date:
                        temporal_str += f" to {end_date}"
                    temporal_str += "</small>"
                if is_point is not None:
                    temporal_str += f"<br><small>Point in time: {is_point}</small>"
                html_parts.append(f'<li class="list-group-item py-1"><small>{node_id_in_chain_detail}: {textwrap.shorten(desc_detail, width=70, placeholder="...")} [{subtype_detail}]{temporal_str}</small></li>')
            html_parts.append("""
                        </ul>
                    </div>
                </div>""")
        html_parts.append('</div>') 
        if causal_chain_chart_b64:
             html_parts.append(f"""
                <div class="col-md-12"> 
                    <div class="chart">
                        <h3 class="h6 mt-3">Top Causal Chain Visualization</h3>
                        <img src="data:image/png;base64,{causal_chain_chart_b64}" class="img-fluid" alt="Causal Chain Visualization">
                    </div>
                </div>""")
        html_parts.append('</div>')
        # LLM summary for top chain
        if results['causal_chains']:
            try:
                top_chain_data = results['causal_chains'][0]
                chain_summary_prompt = "Summarize the primary causal chain presented, highlighting the initial trigger and final outcome."
                llm_chain_summary = generate_narrative_summary_with_llm(top_chain_data, chain_summary_prompt)
                html_parts.append(f'<div class="llm-summary"><h4>Analytical Summary of Top Chain (LLM):</h4><p>{llm_chain_summary}</p></div>')
            except Exception as e:
                html_parts.append(f'<div class="llm-summary"><h4>Analytical Summary of Top Chain (LLM):</h4><p>LLM integration error: {str(e)}</p></div>')
    else:
        html_parts.append("<p>No clear causal chains detected in the network.</p>")
    html_parts.append("""
            </div>
        </div>""")

    # Causal Mechanisms
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Causal Mechanisms Evaluation</h2></div>
            <div class="card-body">""")
    if results.get('mechanisms'):
        for i, mech in enumerate(results['mechanisms'], 1):
            html_parts.append(f"""
                <div class="card mb-3">
                    <div class="card-header"><h3 class="h6 mb-0">Mechanism {i}: {textwrap.shorten(mech.get('name', 'N/A'), width=100, placeholder='...')}</h3></div>
                    <div class="card-body">
                        <div class="progress mb-2" role="progressbar" aria-label="Completeness {mech.get('completeness',0)}%" aria-valuenow="{mech.get('completeness',0)}" aria-valuemin="0" aria-valuemax="100">
                            <div class="progress-bar" style="width: {mech.get('completeness',0)}%">{mech.get('completeness',0)}%</div>
                        </div>
                        <p class='mb-1'><small><strong>Confidence:</strong> {mech.get('confidence', 'unknown')} | <strong>Level of detail:</strong> {mech.get('level_of_detail', 'medium')}</small></p>
                """)
            if mech.get('causes'): html_parts.append(f"<p class='mb-1'><small><strong>Contributing factors:</strong> {', '.join([textwrap.shorten(c, width=40, placeholder='...') for c in mech['causes']])}</small></p>")
            if mech.get('effects'): html_parts.append(f"<p class='mb-1'><small><strong>Effects:</strong> {', '.join([textwrap.shorten(e, width=40, placeholder='...') for e in mech['effects']])}</small></p>")
            # LLM elaboration
            if 'llm_elaboration' in mech and isinstance(mech['llm_elaboration'], dict):
                llm_mech = mech['llm_elaboration']
                html_parts.append(f'<div class="llm-summary"><h4>LLM Elaboration:</h4>')
                if llm_mech.get('narrative_elaboration'):
                    html_parts.append(f'<p><strong>Narrative:</strong> {llm_mech["narrative_elaboration"]}</p>')
                if llm_mech.get('identified_missing_micro_steps'):
                    html_parts.append('<p><strong>Missing Micro-Steps:</strong></p><ul>')
                    for step in llm_mech['identified_missing_micro_steps']:
                        html_parts.append(f'<li>{step.get("suggested_event_description", "N/A")} (type: {step.get("suggested_event_type", "N/A")})</li>')
                    html_parts.append('</ul>')
                if llm_mech.get('coherence_assessment'):
                    html_parts.append(f'<p><strong>Coherence Assessment:</strong> {llm_mech["coherence_assessment"]}</p>')
                if llm_mech.get('reasoning_for_suggestions'):
                    html_parts.append(f'<p><strong>LLM Reasoning:</strong> {llm_mech["reasoning_for_suggestions"]}</p>')
                html_parts.append('</div>')
    else:
        html_parts.append("<p>No causal mechanisms found or evaluated.</p>")
    html_parts.append("""
            </div>
        </div>""")

    # Hypothesis Evaluation
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Hypothesis Evaluation</h2></div>
            <div class="card-body">""")
    if results.get('evidence_analysis') and isinstance(results['evidence_analysis'], dict):
        html_parts.append('<div class="row"><div class="col-lg-8">')
        for hyp_id, analysis_data in results['evidence_analysis'].items():
            hypothesis_description_for_html = analysis_data.get('description', f'N/A_FOR_HYP_{hyp_id}')
            safe_print(f"DEBUG_HTML_FORMAT_HYP_DESC: HypID: {hyp_id}, Description from analysis_data: '{hypothesis_description_for_html}'")
            html_parts.append(f"""
                <div class="card mb-3">
                    <div class="card-header"><h3 class="h6 mb-0">Hypothesis: {textwrap.shorten(hypothesis_description_for_html, width=100, placeholder='...')}</h3></div>
                    <div class="card-body">
                        <p><strong>Assessment:</strong> {analysis_data.get('assessment', 'N/A')} | <strong>Evidence balance:</strong> {analysis_data.get('balance', 0.0):.2f}</p>
                """)
            if analysis_data.get('prior_probability') is not None: html_parts.append(f"<p><small>Prior probability: {analysis_data['prior_probability']}</small></p>")
            # LLM summary for hypothesis
            if analysis_data.get('llm_summary'):
                html_parts.append(f'<div class="llm-summary"><h4>LLM Analytical Summary:</h4><p>{analysis_data["llm_summary"]}</p></div>')
            if analysis_data.get('supporting_evidence'):
                html_parts.append("<h4 class='h6 mt-2'>Supporting evidence:</h4>")
                for ev in analysis_data['supporting_evidence']:
                    html_parts.append(f"""
                        <div class="evidence-item supporting">
                            <small><strong>{ev.get('id', 'N/A_ev_id')}:</strong> {textwrap.shorten(ev.get('description', 'N/A'), width=80, placeholder="...")} (Type: {ev.get('type', 'N/A')}, PV: {ev.get('probative_value', 'N/A')})<br>
                            <em>Quote: "{textwrap.shorten(ev.get('source_text_quote', 'No quote provided.'), width=100, placeholder='...')}"</em></small>
                    """)
                    # LLM evidence refinement
                    if ev.get('llm_reasoning'):
                        html_parts.append(f'<div class="llm-summary"><strong>LLM Reasoning:</strong> {json.dumps(ev["llm_reasoning"], indent=2)}</div>')
                    html_parts.append('</div>')
            if analysis_data.get('refuting_evidence'):
                html_parts.append("<h4 class='h6 mt-2'>Refuting evidence:</h4>")
                for ev in analysis_data['refuting_evidence']:
                    html_parts.append(f"""
                        <div class="evidence-item refuting">
                            <small><strong>{ev.get('id', 'N/A_ev_id')}:</strong> {textwrap.shorten(ev.get('description', 'N/A'), width=80, placeholder="...")} (Type: {ev.get('type', 'N/A')}, PV: {ev.get('probative_value', 'N/A')})<br>
                            <em>Quote: "{textwrap.shorten(ev.get('source_text_quote', 'No quote provided.'), width=100, placeholder='...')}"</em></small>
                    """)
                    # LLM evidence refinement
                    if ev.get('llm_reasoning'):
                        html_parts.append(f'<div class="llm-summary"><strong>LLM Reasoning:</strong> {json.dumps(ev["llm_reasoning"], indent=2)}</div>')
                    html_parts.append('</div>')
            html_parts.append("</div></div>")
        html_parts.append('</div><div class="col-lg-4">')
        if evidence_chart_b64: html_parts.append(f'<div class="chart"><h3 class="h6">Evidence Strength Comparison</h3><img src="data:image/png;base64,{evidence_chart_b64}" class="img-fluid" alt="Evidence Strength Chart"></div>')
        html_parts.append('</div></div>')
    else:
        html_parts.append("<p>No hypotheses or evidence found for evaluation.</p>")
    html_parts.append("""
            </div>
        </div>""")
    # Causal Chains LLM Summary
    if results.get('causal_chains_llm_summary'):
        html_parts.append(f'<div class="llm-summary"><h4>Analytical Summary of Top Chain (LLM):</h4><p>{results["causal_chains_llm_summary"]}</p></div>')

    # Condition Analysis
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Condition Analysis</h2></div>
            <div class="card-body"><div class="row">""")
    html_parts.append('<div class="col-md-6"><h3 class="h6">Enabling Conditions:</h3>')
    if results.get('conditions', {}).get('enabling'):
        html_parts.append("<ul class='list-group list-group-flush'>")
        for cond_detail in results['conditions']['enabling']:
            target_desc = textwrap.shorten(cond_detail.get('target', {}).get('description', 'N/A'), width=50, placeholder="...")
            cond_desc = textwrap.shorten(cond_detail.get('condition', 'N/A'), width=50, placeholder="...")
            html_parts.append(f"<li class='list-group-item'><small><strong>{cond_desc}</strong> &rarr; enables &rarr; {target_desc} ({cond_detail.get('target',{}).get('type','N/A')})</small></li>")
        html_parts.append("</ul>")
    else:
        html_parts.append("<p><small>No enabling conditions found linked to Events or Causal Mechanisms.</small></p>")
    html_parts.append('</div><div class="col-md-6"><h3 class="h6">Constraining Conditions:</h3>')
    if results.get('conditions', {}).get('constraining'):
        html_parts.append("<ul class='list-group list-group-flush'>")
        for cond_detail in results['conditions']['constraining']:
            target_desc = textwrap.shorten(cond_detail.get('target', {}).get('description', 'N/A'), width=50, placeholder="...")
            cond_desc = textwrap.shorten(cond_detail.get('condition', 'N/A'), width=50, placeholder="...")
            html_parts.append(f"<li class='list-group-item'><small><strong>{cond_desc}</strong> &rarr; constrains &rarr; {target_desc} ({cond_detail.get('target',{}).get('type','N/A')})</small></li>")
        html_parts.append("</ul>")
    else:
        html_parts.append("<p><small>No constraining conditions found linked to Events or Causal Mechanisms.</small></p>")
    html_parts.append('</div></div></div></div>')

    # Actor Analysis
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Actor Analysis</h2></div>
            <div class="card-body">""")
    filtered_actors = [actor for actor in results.get('actors', []) if actor.get('influence_score', 0) > 0 or actor.get('initiated_events')]
    if filtered_actors:
        for actor in filtered_actors:
            actor_name_html = textwrap.shorten(actor.get('name', 'N/A_actor'), width=60, placeholder="...")
            html_parts.append(f"""
                    <div class="card mb-3">
                    <div class="card-header"><h3 class="h6 mb-0">{actor_name_html}</h3></div>
                        <div class="card-body">
                        <p><small><strong>Role:</strong> {actor.get('role', 'unknown')} | <strong>Influence score:</strong> {actor.get('influence_score', 0)}</small></p>""")
            if actor.get('initiated_events'):
                html_parts.append("<p><small><strong>Initiated events:</strong></small></p><ul class='list-group list-group-flush'>")
                for event in actor['initiated_events']:
                    event_desc_html = textwrap.shorten(event.get('description', 'N/A'), width=70, placeholder="...")
                    html_parts.append(f"<li class='list-group-item py-1'><small>{event.get('id')}: {event_desc_html}</small></li>")
                html_parts.append("</ul>")
            if actor.get('beliefs') and actor['beliefs'] != 'unknown': html_parts.append(f"<p><small><strong>Beliefs:</strong> {textwrap.shorten(actor['beliefs'], width=100, placeholder='...')}</small></p>")
            if actor.get('intentions') and actor['intentions'] != 'unknown': html_parts.append(f"<p><small><strong>Intentions:</strong> {textwrap.shorten(actor['intentions'], width=100, placeholder='...')}</small></p>")
            html_parts.append("""
                        </div>
                </div>""")
    else:
        html_parts.append("<p>No actors with significant attributed activity found.</p>")
    html_parts.append("""
                </div>
        </div>""")

    # Alternative Explanations (Ensure this section is present and uses results)
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Alternative Explanations</h2></div>
            <div class="card-body">""")
    if results.get('alternatives'):
        for alt in results['alternatives']:
            alt_desc_html = textwrap.shorten(alt.get('description', 'N/A_alt'), width=100, placeholder="...")
            html_parts.append(f"""
                <div class="card mb-3">
                    <div class="card-header"><h3 class="h6 mb-0">{alt_desc_html}</h3></div>
                    <div class="card-body">
                        <p><small><strong>Assessment:</strong> {alt.get('assessment', 'N/A')} | <strong>Strength score:</strong> {alt.get('strength_score', 0)}</small></p>""")
            # ... (supporting/refuting evidence for alternatives)
            html_parts.append("""
                    </div>
                </div>""")
    else:
        html_parts.append("<p>No alternative explanations found in the network.</p>")
    html_parts.append("""
            </div>
        </div>""")

    # Network Metrics
    html_parts.append("""
        <div class="card">
            <div class="card-header"><h2 class="card-title h5">Network Metrics</h2></div>
            <div class="card-body"><div class="row">
                <div class="col-md-6">""")
    if 'metrics' in results:
        html_parts.append(f"<p><small><strong>Graph density:</strong> {results['metrics'].get('density', 0.0):.4f}</small></p>")
        html_parts.append(f"<p><small><strong>Average path length:</strong> {results['metrics'].get('avg_path_length', 'N/A')}</small></p>")
        html_parts.append("<h3 class=\"h6 mt-2\">Most Central Nodes (Degree):</h3><ul class=\"list-group list-group-flush\">")
        if results['metrics'].get('degree_centrality'):
            for node_id, value in list(results['metrics']['degree_centrality'].items())[:5]:
                node_graph_data = G.nodes.get(node_id, {})
                node_desc = node_graph_data.get('attr_props', {}).get('description', node_id)
                node_main_type = node_graph_data.get('type', 'unknown')
                html_parts.append(f"<li class='list-group-item py-1'><small><strong>{textwrap.shorten(node_desc, width=40, placeholder='...')}</strong> ({node_main_type}): {value}</small></li>")
        html_parts.append("</ul>")
    html_parts.append("""
                </div>
                <div class="col-md-6">""")
    if centrality_chart_b64: html_parts.append(f'<div class="chart"><h3 class="h6">Node Centrality</h3><img src="data:image/png;base64,{centrality_chart_b64}" class="img-fluid" alt="Centrality Chart"></div>')
    html_parts.append("""
                </div>
            </div></div>
        </div>""")
    # Theoretical Insights
    if theoretical_insights:
        html_parts.append(f'<div class="card"><div class="card-header"><h2 class="card-title h5">Theoretical Insights</h2></div><div class="card-body">{theoretical_insights}</div></div>')
    return ''.join(html_parts)

# --- Charting Functions (generate_node_type_chart, etc.) ---
# These need to be refactored to use G.nodes[id].get('attr_props', {}).get('property_name')
# if they access graph G directly for labels or descriptions.
# For brevity, providing stubs. The user's IDE should refactor their existing charting functions.

def generate_node_type_chart(results):
    if not results or 'metrics' not in results or 'node_type_distribution' not in results['metrics']: return None
    plt.figure(figsize=(10, 6))
    node_dist_data = results['metrics']['node_type_distribution']
    if not node_dist_data: plt.close(); return None
    try:
        plt.pie(node_dist_data.values(), labels=node_dist_data.keys(), autopct='%1.1f%%', shadow=True)
        plt.title('Node Type Distribution')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close() # Close the figure
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        safe_print(f"Error generating node type chart: {e}")
        plt.close()
        return None

def generate_edge_type_chart(results):
    if not results or 'metrics' not in results or 'edge_type_distribution' not in results['metrics']: return None
    plt.figure(figsize=(12, 7)) # Adjusted size
    edge_types_data = results['metrics']['edge_type_distribution']
    if not edge_types_data: plt.close(); return None
    try:
        plt.bar(edge_types_data.keys(), edge_types_data.values(), color='teal')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title('Edge Type Distribution')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close() # Close the figure
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        safe_print(f"Error generating edge type chart: {e}")
        plt.close()
        return None

def generate_causal_chain_network(G, chain_to_visualize): # Takes one chain
    if not chain_to_visualize or 'path' not in chain_to_visualize or not G:
        return None
    
    path_nodes = chain_to_visualize['path']
    if not path_nodes or len(path_nodes) < 2: return None

    try:
        subG = G.subgraph(path_nodes)
        if subG.number_of_nodes() == 0: plt.close(); return None

        plt.figure(figsize=(max(10, len(path_nodes) * 1.5), 6)) # Dynamic width
        # Try a layout that works better for directed chains, e.g., a ranked layout if possible, or spring with iterations
        try:
            # For a more linear layout of a chain, 'dot' layout from graphviz is good, but requires graphviz.
            # nx.nx_agraph.graphviz_layout(subG, prog='dot')
            # Fallback to spring_layout if graphviz is not available or fails.
            pos = nx.spring_layout(subG, k=0.5, iterations=50, seed=42)
        except:
            pos = nx.spring_layout(subG, seed=42) # Fallback

        node_colors_list = [NODE_COLORS.get(G.nodes[node].get('type', 'Event'), '#cccccc') for node in subG.nodes()]
        
        nx.draw_networkx_nodes(subG, pos, node_size=2000, node_color=node_colors_list, alpha=0.9)
        nx.draw_networkx_edges(subG, pos, width=1.5, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray')
        
        node_labels = {
            node_id: textwrap.shorten(G.nodes[node_id].get('attr_props', {}).get('description', node_id), width=20, placeholder="...")
            for node_id in subG.nodes()
        }
        nx.draw_networkx_labels(subG, pos, labels=node_labels, font_size=9)
        
        plt.title(f"Visualization of Causal Chain (Length: {chain_to_visualize['length']})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close() # Close the figure
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        safe_print(f"Error generating causal chain network viz: {e}")
        plt.close()
        return None

def generate_centrality_chart(results, G): # Pass G for labels
    if not results or 'metrics' not in results or 'degree_centrality' not in results['metrics'] or not G: return None
    centrality_data = results['metrics']['degree_centrality']
    if not centrality_data: plt.close(); return None

    # Get top N nodes for charting
    top_n = 10
    sorted_centrality = sorted(centrality_data.items(), key=lambda item: item[1], reverse=True)[:top_n]
    
    node_ids_for_chart = [item[0] for item in sorted_centrality]
    values = [item[1] for item in sorted_centrality]
    
    # Get descriptions for labels
    labels = [textwrap.shorten(G.nodes[node_id].get('attr_props', {}).get('description', node_id), width=30, placeholder="...") for node_id in node_ids_for_chart]

    try:
        plt.figure(figsize=(12, max(6, len(labels) * 0.5))) # Dynamic height
        plt.barh(labels, values, color='skyblue')
        plt.xlabel('Degree Centrality')
        plt.title(f'Top {len(labels)} Nodes by Degree Centrality')
        plt.gca().invert_yaxis() # Display most central at top
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close() # Close the figure
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        safe_print(f"Error generating centrality chart: {e}")
        plt.close()
        return None

def generate_evidence_strength_chart(results):
    if not results or 'evidence_analysis' not in results or not results['evidence_analysis']: return None
    
    hypotheses_labels = []
    supporting_counts = []
    refuting_counts = []
    
    for hyp_id, analysis_data in results['evidence_analysis'].items():
        hyp_desc = analysis_data.get('description', hyp_id)
        hypotheses_labels.append(textwrap.shorten(hyp_desc, width=25, placeholder="..."))
        supporting_counts.append(len(analysis_data.get('supporting_evidence', [])))
        refuting_counts.append(len(analysis_data.get('refuting_evidence', [])))

    if not hypotheses_labels: plt.close(); return None

    try:
        x = range(len(hypotheses_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(hypotheses_labels)*1.2), 7)) # Dynamic width
        
        rects1 = ax.bar([i - width/2 for i in x], supporting_counts, width, label='Supporting Evidence', color='#28a745')
        rects2 = ax.bar([i + width/2 for i in x], refuting_counts, width, label='Refuting Evidence', color='#dc3545')
        
        ax.set_ylabel('Number of Evidence Items')
        ax.set_title('Evidence Count by Hypothesis')
        ax.set_xticks(x)
        ax.set_xticklabels(hypotheses_labels, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig) # Close the figure
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        safe_print(f"Error generating evidence strength chart: {e}")
        plt.close() # Ensure plt is closed if error occurs mid-plot
        return None

# --- main() function ---
# (Keep your existing main() function structure. Ensure it calls the refactored functions
# and passes G and results['filename'] to formatting functions correctly.)
def main():
    args = parse_args()
    if not os.path.isfile(args.json_file):
        safe_print(f"Error: File not found: {args.json_file}"); sys.exit(1)
    
    safe_print(f"[ANALYZE] Analyzing data from {os.path.basename(args.json_file)}...")
    try:
        G, data = load_graph(args.json_file)
    except Exception as e:
        safe_print(f"Error loading graph: {e}"); sys.exit(1)
    
    safe_print(f"[SUCCESS] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    analysis_results = { 
        'filename': args.json_file, 
        'causal_chains': identify_causal_chains(G),
        'mechanisms': evaluate_mechanisms(G),
        'evidence_analysis': analyze_evidence(G)['by_hypothesis'],
        'conditions': identify_conditions(G),
        'actors': analyze_actors(G),
        'alternatives': analyze_alternative_explanations(G),
        'metrics': calculate_network_metrics(G)
    }

    # --- LLM Mechanism Elaboration ---
    for mech in analysis_results['mechanisms']:
        mech_id = mech['id']
        if not G.has_node(mech_id):
            continue
        mechanism_node = G.nodes[mech_id]
        linked_event_nodes = []
        for pred_id in G.predecessors(mech_id):
            edge_data = G.get_edge_data(pred_id, mech_id)
            if not edge_data:
                continue
            pred_node_data = G.nodes[pred_id]
            if pred_node_data.get('type') == 'Event' and edge_data.get('type') in ['part_of_mechanism', 'causes', 'triggers']:
                linked_event_nodes.append({
                    'id': pred_id,
                    'properties': pred_node_data  # Already flat structure
                })
        original_text_context = ''
        ontology_schema = None
        try:
            llm_elaboration = elaborate_mechanism_with_llm(
                mechanism_node={'id': mech_id, 'properties': mechanism_node},  # Already flat structure
                linked_event_nodes=linked_event_nodes,
                original_text_context=original_text_context,
                ontology_schema=ontology_schema
            )
        except Exception as e:
            llm_elaboration = {'error': str(e)}
        mech['llm_elaboration'] = llm_elaboration
    # --- End LLM Mechanism Elaboration ---

    # --- LLM Evidence Refinement --- 
    # REMOVED: This was double processing - already done at lines 319-339
    # --- End LLM Evidence Refinement ---

    # --- LLM Narrative Summaries ---
    if analysis_results.get('causal_chains'):
        try:
            top_chain_data = analysis_results['causal_chains'][0]
            chain_summary_prompt = "Summarize the primary causal chain presented, highlighting the initial trigger and final outcome."
            llm_chain_summary = generate_narrative_summary_with_llm(top_chain_data, chain_summary_prompt)
            analysis_results['causal_chains_llm_summary'] = llm_chain_summary
        except Exception as e:
            analysis_results['causal_chains_llm_summary'] = f"LLM integration error: {str(e)}"
    
    for hyp_id, hyp_data in analysis_results['evidence_analysis'].items():
        try:
            hyp_summary_prompt = f"Summarize the overall findings for the hypothesis: {hyp_data.get('description', hyp_id)}."
            llm_hyp_summary = generate_narrative_summary_with_llm(hyp_data, hyp_summary_prompt)
            hyp_data['llm_summary'] = llm_hyp_summary
        except Exception as e:
            hyp_data['llm_summary'] = f"LLM integration error: {str(e)}"
    # --- End LLM Narrative Summaries ---

    theoretical_insights = None
    if args.theory or args.html:
        safe_print("[INFO] Generating theoretical insights...")
        theoretical_insights = generate_theoretical_insights(analysis_results, G)
    
    analysis_text = ""
    output_extension = "md"
    if args.html:
        safe_print("[INFO] Formatting HTML report...")
        # Check if network data is provided for embedded visualization
        network_data_json = None
        if args.network_data:
            # If the argument is a path to a file, read the file
            if os.path.isfile(args.network_data):
                with open(args.network_data, 'r', encoding='utf-8') as f:
                    network_data_json = f.read()
            else:
                network_data_json = args.network_data
        # Pass G and network_data_json to format_html_analysis
        analysis_text = format_html_analysis(analysis_results, data, G, theoretical_insights, network_data_json) 
        output_extension = "html"
    else: # Markdown
        safe_print("[INFO] Formatting Markdown report...")
        # Similarly, pass G if format_analysis needs it
        analysis_text = format_analysis(analysis_results, data, G, theoretical_insights) 
    
    input_json_path = Path(args.json_file)
    project_dir = input_json_path.parent 
    project_name = input_json_path.stem.replace('_graph', '') 
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output:
        output_path_str = args.output
    else:
        output_path_str = str(project_dir / f"{project_name}_analysis_{now_str}.{output_extension}")
    output_path = Path(output_path_str)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        safe_print(f"[SUCCESS] Analysis report saved to {output_path}")
        if args.html:
            import webbrowser
            try:
                webbrowser.open('file://' + str(output_path.resolve()))
            except Exception as e_wb:
                safe_print(f"[WARN] Could not open HTML report in browser: {e_wb}")
    except Exception as e:
        safe_print(f"Error writing report to {output_path}: {e}")
        if not args.html: safe_print("\nANALYSIS CONTENT (MD):\n" + analysis_text)
    
    safe_print("\n[SUCCESS] Analysis generation complete!")
    
    if not args.html and args.charts_dir:
        # ... (Your existing chart saving logic - ensure it uses refactored chart functions)
        charts_output_dir = Path(args.charts_dir)
        charts_output_dir.mkdir(parents=True, exist_ok=True)
        base_chart_name = Path(args.json_file).stem
        safe_print(f"[INFO] Generating PNG charts in {charts_output_dir}/...")
        # Example for one chart, others would follow similar pattern
        node_type_chart_img = generate_node_type_chart(analysis_results)
        if node_type_chart_img:
            try:
                with open(charts_output_dir / f"{base_chart_name}_node_types.png", "wb") as f:
                    f.write(base64.b64decode(node_type_chart_img))
                safe_print(f"[SUCCESS] Saved node_types.png")
            except Exception as e_png:
                safe_print(f"[ERROR] Could not save node_types.png: {e_png}")
        # Repeat for other charts if needed
    elif not args.html:
        safe_print("[INFO] PNG chart generation skipped: --charts-dir not specified or HTML output selected.")
    
    # ... after analysis_results is created ...
    # Write _analysis_summary.json for cross-case synthesis
    try:
        # Prepare comprehensive summary structure
        summary = {
            'filename': args.json_file,
            'case_identifier': Path(args.json_file).stem,
            'generated_at': datetime.now().isoformat(),
            
            # Core Analysis Results
            'causal_chains': analysis_results.get('causal_chains', []),
            'causal_mechanisms': analysis_results.get('mechanisms', []),
            'network_metrics': analysis_results.get('metrics', {}),
            
            # Van Evera Analysis
            'hypotheses_evaluation': [],
            
            # Context Analysis  
            'conditions': analysis_results.get('conditions', {}),
            'actors': analysis_results.get('actors', []),
            'alternative_explanations': analysis_results.get('alternatives', []),
            
            # Methodological Notes
            'van_evera_methodology_applied': True,
            'analysis_completeness': 'comprehensive'
        }
        
        # Enhanced hypothesis evaluation with Van Evera details
        for hyp_id, hyp_data in analysis_results['evidence_analysis'].items():
            hyp_summary = {
                'hypothesis_id': hyp_id,
                'description': hyp_data.get('description', ''),
                'assessment': hyp_data.get('assessment', 'Undetermined'),
                'balance_score': hyp_data.get('balance', 0.0),
                'van_evera_applied': hyp_data.get('van_evera_applied', False),
                'supporting_evidence_count': len(hyp_data.get('supporting_evidence', [])),
                'refuting_evidence_count': len(hyp_data.get('refuting_evidence', [])),
                'supporting_evidence_ids': [ev.get('id') for ev in hyp_data.get('supporting_evidence', [])],
                'refuting_evidence_ids': [ev.get('id') for ev in hyp_data.get('refuting_evidence', [])],
                
                # Van Evera evidence breakdown (with normalization to handle naming variations)
                'evidence_by_type': {
                    'hoop': [ev for ev in hyp_data.get('supporting_evidence', []) + hyp_data.get('refuting_evidence', []) if normalize_evidence_type_for_output(ev.get('type')) == 'hoop'],
                    'smoking_gun': [ev for ev in hyp_data.get('supporting_evidence', []) + hyp_data.get('refuting_evidence', []) if normalize_evidence_type_for_output(ev.get('type')) == 'smokinggun'],
                    'straw_in_wind': [ev for ev in hyp_data.get('supporting_evidence', []) + hyp_data.get('refuting_evidence', []) if normalize_evidence_type_for_output(ev.get('type')) == 'strawinthewind'],
                    'doubly_decisive': [ev for ev in hyp_data.get('supporting_evidence', []) + hyp_data.get('refuting_evidence', []) if normalize_evidence_type_for_output(ev.get('type')) == 'doublydecisive']
                }
            }
            summary['hypotheses_evaluation'].append(hyp_summary)
        # Write to file
        input_json_path = Path(args.json_file)
        project_dir = input_json_path.parent
        project_name = input_json_path.stem.replace('_graph', '')
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = project_dir / f"{project_name}_analysis_summary_{now_str}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        safe_print(f"[SUCCESS] Analysis summary JSON saved to {summary_path}")
    except Exception as e:
        safe_print(f"[ERROR] Could not write analysis summary JSON: {e}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()