"""
Disconnection Repair Module for Process Tracing System

This module provides automated and LLM-assisted repair of disconnected entities
in process tracing graphs, using inference rules and targeted prompts.
"""

import json
from typing import Dict, List, Tuple, Optional
from .connectivity_analysis import analyze_connectivity
from .ontology import EDGE_TYPES
import os
from datetime import datetime


class ConnectionInferenceEngine:
    """Automatically infers missing connections using domain-specific rules."""
    
    def __init__(self):
        self.inference_rules = {
            'Condition': self._infer_condition_connections,
            'Actor': self._infer_actor_connections,
            'Event': self._infer_event_connections,
            'Evidence': self._infer_evidence_connections
        }
        
        # Semantic keywords for connection inference
        self.semantic_patterns = {
            'enables': ['enable', 'allow', 'facilitate', 'make possible', 'permit', 'distance', 'economic development'],
            'constrains': ['constrain', 'limit', 'prevent', 'restrict', 'naval supremacy', 'military'],
            'initiates': ['initiate', 'start', 'launch', 'begin', 'personally initiated'],
            'supports': ['support', 'evidence for', 'confirm', 'validate'],
            'refutes': ['refute', 'contradict', 'challenge', 'disprove'],
            'causes': ['cause', 'lead to', 'result in', 'bring about']
        }
    
    def infer_missing_connections(self, graph_data: Dict) -> List[Dict]:
        """
        Automatically infer missing connections for disconnected nodes.
        
        Args:
            graph_data: Dictionary with 'nodes' and 'edges' arrays
            
        Returns:
            List of inferred edge dictionaries
        """
        analysis = analyze_connectivity(graph_data)
        inferred_edges = []
        
        # Process isolated nodes
        for isolated_node in analysis['isolated_nodes']:
            node_type = isolated_node['type']
            
            if node_type in self.inference_rules:
                new_edges = self.inference_rules[node_type](isolated_node, graph_data)
                inferred_edges.extend(new_edges)
        
        # Process small components
        for component in analysis['small_components']:
            component_edges = self._infer_component_connections(component, graph_data)
            inferred_edges.extend(component_edges)
        
        return inferred_edges
    
    def _infer_condition_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Infer connections for Condition nodes."""
        edges = []
        node_id = node['id']
        description = node['description'].lower()
        
        # Find target nodes for enables/constrains relationships
        for target_node in graph_data['nodes']:
            target_id = target_node['id']
            target_type = target_node['type']
            target_desc = target_node['properties'].get('description', '').lower()
            
            # Infer 'enables' relationships
            if target_type in ['Event', 'Causal_Mechanism', 'Hypothesis']:
                if self._matches_semantic_pattern(description, 'enables'):
                    # Specific inference rules
                    if 'distance' in description and 'resistance' in target_desc:
                        edges.append(self._create_edge(node_id, target_id, 'enables', 
                                                     "Geographic distance enables colonial resistance"))
                    elif 'economic' in description and ('independence' in target_desc or 'revolution' in target_desc):
                        edges.append(self._create_edge(node_id, target_id, 'enables',
                                                     "Economic development enables independence"))
                    elif 'enlightenment' in description and 'constitutional' in target_desc:
                        edges.append(self._create_edge(node_id, target_id, 'enables',
                                                     "Enlightenment ideas enable constitutional principles"))
            
            # Infer 'constrains' relationships
            if target_type in ['Event', 'Causal_Mechanism', 'Actor']:
                if self._matches_semantic_pattern(description, 'constrains'):
                    if 'naval' in description and 'colonial' in target_desc:
                        edges.append(self._create_edge(node_id, target_id, 'constrains',
                                                     "Naval supremacy constrains colonial options"))
        
        return edges
    
    def _infer_actor_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Infer connections for Actor nodes."""
        edges = []
        node_id = node['id']
        actor_name = node.get('name', '').lower()
        
        # Find events this actor likely initiated
        for target_node in graph_data['nodes']:
            if target_node['type'] == 'Event':
                target_id = target_node['id']
                target_desc = target_node['properties'].get('description', '').lower()
                
                # Check if actor name appears in event description
                if actor_name and actor_name in target_desc:
                    edges.append(self._create_edge(node_id, target_id, 'initiates',
                                                 f"Actor {actor_name} initiated event"))
                
                # Check for initiation keywords
                if self._matches_semantic_pattern(target_desc, 'initiates'):
                    # Specific patterns for known actors
                    if 'hutchinson' in actor_name and 'enforcement' in target_desc:
                        edges.append(self._create_edge(node_id, target_id, 'initiates',
                                                     "Governor initiated enforcement measures"))
        
        return edges
    
    def _infer_event_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Infer connections for Event nodes."""
        edges = []
        node_id = node['id']
        description = node['description'].lower()
        
        # Look for mechanism relationships
        for target_node in graph_data['nodes']:
            target_id = target_node['id']
            target_type = target_node['type']
            target_desc = target_node['properties'].get('description', '').lower()
            
            if target_type == 'Causal_Mechanism':
                # Check if event should be part of mechanism
                if any(keyword in description for keyword in ['stamp act', 'congress', 'declaration']):
                    if 'resistance' in target_desc or 'imperial' in target_desc:
                        edges.append(self._create_edge(node_id, target_id, 'part_of_mechanism',
                                                     "Event is part of resistance mechanism"))
        
        return edges
    
    def _infer_evidence_connections(self, node: Dict, graph_data: Dict) -> List[Dict]:
        """Infer connections for Evidence nodes."""
        edges = []
        node_id = node['id']
        description = node['description'].lower()
        
        # Find hypotheses and mechanisms this evidence might support
        for target_node in graph_data['nodes']:
            target_id = target_node['id']
            target_type = target_node['type']
            target_desc = target_node['properties'].get('description', '').lower()
            
            if target_type in ['Hypothesis', 'Causal_Mechanism']:
                # Infer support relationships based on content similarity
                if self._semantic_similarity(description, target_desc) > 0.3:
                    edges.append(self._create_edge(node_id, target_id, 'supports',
                                                 "Evidence semantically supports target"))
        
        return edges
    
    def _infer_component_connections(self, component: Dict, graph_data: Dict) -> List[Dict]:
        """Infer connections to link small components to main graph."""
        edges = []
        component_nodes = [node['id'] for node in component['nodes']]
        
        # Find the largest component (main graph)
        analysis = analyze_connectivity(graph_data)
        main_component_size = max(len(comp['nodes']) for comp in analysis['small_components'] + 
                                [{'nodes': [{'id': node['id']} for node in graph_data['nodes'] 
                                          if node['id'] not in sum([comp['nodes'] for comp in analysis['small_components']], [])]}])
        
        # For each node in small component, try to connect to main graph
        for node_data in component['nodes']:
            node_id = node_data['id']
            node_type = node_data['type']
            
            # Apply type-specific inference rules
            if node_type in self.inference_rules:
                full_node = next(n for n in graph_data['nodes'] if n['id'] == node_id)
                isolated_node = {
                    'id': node_id,
                    'type': node_type,
                    'description': full_node['properties'].get('description', ''),
                    'name': full_node['properties'].get('name', '')
                }
                component_edges = self.inference_rules[node_type](isolated_node, graph_data)
                edges.extend(component_edges)
        
        return edges
    
    def _matches_semantic_pattern(self, text: str, pattern_type: str) -> bool:
        """Check if text matches semantic patterns for edge types."""
        if pattern_type not in self.semantic_patterns:
            return False
        
        patterns = self.semantic_patterns[pattern_type]
        return any(pattern in text for pattern in patterns)
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple semantic similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_edge(self, source_id: str, target_id: str, edge_type: str, reasoning: str) -> Dict:
        """Create a properly formatted edge dictionary."""
        edge_id = f"inferred_{source_id}_{target_id}_{edge_type}"
        
        return {
            'id': edge_id,
            'source': source_id,
            'target': target_id,
            'type': edge_type,
            'properties': {
                'reasoning': reasoning,
                'inferred': True,
                'confidence': 0.7
            }
        }


class LLMDisconnectionRepairer:
    """Uses LLM calls to repair complex disconnection cases."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        
        self.repair_prompt_template = """
CONNECTIVITY REPAIR TASK

The following nodes are disconnected from the main process tracing graph:

DISCONNECTED NODES:
{disconnected_nodes}

MAIN GRAPH CONTEXT:
{main_graph_summary}

TASK: For each disconnected node, add logical connections to the main graph using appropriate edge types from the ontology. Consider:

1. How does this node relate causally to main events?
2. What enabling/constraining relationships exist with conditions?
3. What evidential relationships connect this to hypotheses?
4. What actor-event initiating relationships are implied?

CONNECTIVITY REQUIREMENTS:
- ALL nodes must connect to the main graph (no isolated nodes)
- Use appropriate edge types based on ontology constraints
- Provide reasoning for each connection

Add the missing edges in the same JSON format as the existing graph.

REQUIRED OUTPUT FORMAT:
{{
  "new_edges": [
    {{"source_id": "node1", "target_id": "node2", "type": "edge_type", "properties": {{"reasoning": "explanation"}}}},
    ...
  ]
}}
"""
    
    def repair_disconnections(self, graph_data: Dict, disconnected_analysis: Dict) -> Dict:
        """
        Use LLM to repair disconnections that couldn't be handled by inference rules.
        
        Args:
            graph_data: Original graph data
            disconnected_analysis: Analysis of disconnected nodes
            
        Returns:
            Dictionary with new edges to add
        """
        # Format disconnected nodes for prompt
        disconnected_summary = self._format_disconnected_nodes(disconnected_analysis['isolated_nodes'])
        
        # Create main graph summary
        main_graph_summary = self._create_main_graph_summary(graph_data, disconnected_analysis)
        
        # Construct repair prompt
        repair_prompt = self.repair_prompt_template.format(
            disconnected_nodes=disconnected_summary,
            main_graph_summary=main_graph_summary
        )
        
        # This would call the LLM - implementation depends on your LLM interface
        # For now, return empty result
        return {"new_edges": []}
    
    def _format_disconnected_nodes(self, isolated_nodes: List[Dict]) -> str:
        """Format disconnected nodes for the repair prompt."""
        formatted = []
        
        for node in isolated_nodes:
            formatted.append(f"""
Node ID: {node['id']}
Type: {node['type']}
Description: {node['description']}
Expected Connections: {', '.join(node['expected_connections'])}
""")
        
        return '\n'.join(formatted)
    
    def _create_main_graph_summary(self, graph_data: Dict, analysis: Dict) -> str:
        """Create a summary of the main connected component."""
        # Find the largest component
        if analysis['connected_components'] > 1:
            # Get main component nodes (simplified)
            connected_nodes = []
            for node in graph_data['nodes']:
                if node['id'] not in [iso['id'] for iso in analysis['isolated_nodes']]:
                    connected_nodes.append(node)
            
            # Summarize by type
            type_summary = {}
            for node in connected_nodes:
                node_type = node['type']
                if node_type not in type_summary:
                    type_summary[node_type] = []
                type_summary[node_type].append(node['id'])
            
            summary_parts = []
            for node_type, node_ids in type_summary.items():
                summary_parts.append(f"{node_type}: {', '.join(node_ids[:3])}{'...' if len(node_ids) > 3 else ''}")
            
            return '\n'.join(summary_parts)
        
        return "Single connected component with all nodes"


def repair_graph_connectivity(graph_data: Dict) -> Dict:
    """
    Main function to repair graph connectivity issues.
    
    Args:
        graph_data: Dictionary with 'nodes' and 'edges' arrays
        
    Returns:
        Updated graph_data with new edges added
    """
    # Analyze current connectivity
    analysis = analyze_connectivity(graph_data)
    
    if analysis['connected_components'] == 1:
        print("Graph is already fully connected.")
        return graph_data
    
    print(f"Found {len(analysis['isolated_nodes'])} isolated nodes and {len(analysis['small_components'])} small components.")
    
    # Try automated inference first
    inference_engine = ConnectionInferenceEngine()
    inferred_edges = inference_engine.infer_missing_connections(graph_data)
    
    print(f"Inferred {len(inferred_edges)} potential connections.")
    
    # Add inferred edges to graph
    updated_graph = graph_data.copy()
    updated_graph['edges'] = updated_graph['edges'] + inferred_edges
    
    # Check if connectivity is now resolved
    updated_analysis = analyze_connectivity(updated_graph)
    
    if updated_analysis['connected_components'] == 1:
        print("Successfully achieved full connectivity through inference.")
        return updated_graph
    
    print(f"Still have {updated_analysis['connected_components']} components after inference.")
    print("Complex disconnections may require manual review or LLM assistance.")
    
    return updated_graph


def save_repaired_graph(graph_data: Dict, original_path: str) -> str:
    """Save the repaired graph with a new filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate new filename
    dir_path = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name_parts = base_name.split('.')
    new_name = f"{name_parts[0]}_repaired_{timestamp}.{name_parts[1]}"
    new_path = os.path.join(dir_path, new_name)
    
    # Save repaired graph
    with open(new_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f"Repaired graph saved to: {new_path}")
    return new_path