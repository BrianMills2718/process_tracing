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
        
        # Apply general inference rules for remaining disconnected nodes
        if len(inferred_edges) < len(analysis['isolated_nodes']) * 2:  # Need more connections
            general_edges = self._apply_general_inference_rules(graph_data, analysis)
            inferred_edges.extend(general_edges)
        
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
        
        # Find hypotheses, mechanisms, and events this evidence might connect to
        for target_node in graph_data['nodes']:
            target_id = target_node['id']
            target_type = target_node['type']
            target_desc = target_node['properties'].get('description', '').lower()
            
            # Connect evidence to hypotheses
            if target_type == 'Hypothesis':
                # Strong semantic matches for hypothesis testing
                if self._semantic_similarity(description, target_desc) > 0.2:
                    if 'test' in description:
                        edges.append(self._create_edge(node_id, target_id, 'tests_hypothesis',
                                                     "Evidence tests hypothesis validity"))
                    elif 'support' in description or 'confirm' in description:
                        edges.append(self._create_edge(node_id, target_id, 'supports',
                                                     "Evidence supports hypothesis"))
                    elif 'refute' in description or 'contradict' in description:
                        edges.append(self._create_edge(node_id, target_id, 'refutes',
                                                     "Evidence refutes hypothesis"))
                    else:
                        edges.append(self._create_edge(node_id, target_id, 'provides_evidence_for',
                                                     "Evidence provides evidence for hypothesis"))
            
            # Connect evidence to mechanisms
            elif target_type == 'Causal_Mechanism':
                if self._semantic_similarity(description, target_desc) > 0.2:
                    if 'test' in description:
                        edges.append(self._create_edge(node_id, target_id, 'tests_mechanism',
                                                     "Evidence tests mechanism operation"))
                    else:
                        edges.append(self._create_edge(node_id, target_id, 'supports',
                                                     "Evidence supports mechanism"))
            
            # Connect evidence to events (confirms/disproves occurrence)
            elif target_type == 'Event':
                if self._semantic_similarity(description, target_desc) > 0.3:
                    if 'disprove' in description or 'did not' in description:
                        edges.append(self._create_edge(node_id, target_id, 'disproves_occurrence',
                                                     "Evidence disproves event occurrence"))
                    else:
                        edges.append(self._create_edge(node_id, target_id, 'confirms_occurrence',
                                                     "Evidence confirms event occurrence"))
            
            # Connect evidence to alternative explanations
            elif target_type == 'Alternative_Explanation':
                if self._semantic_similarity(description, target_desc) > 0.2:
                    if 'support' in description:
                        edges.append(self._create_edge(node_id, target_id, 'supports_alternative',
                                                     "Evidence supports alternative explanation"))
                    elif 'refute' in description:
                        edges.append(self._create_edge(node_id, target_id, 'refutes_alternative',
                                                     "Evidence refutes alternative explanation"))
                    else:
                        edges.append(self._create_edge(node_id, target_id, 'tests_alternative',
                                                     "Evidence tests alternative explanation"))
        
        return edges
    
    def _apply_general_inference_rules(self, graph_data: Dict, analysis: Dict) -> List[Dict]:
        """Apply general inference rules to create basic connectivity."""
        edges = []
        
        # Find the largest connected component (main graph)
        all_nodes = {node['id']: node for node in graph_data['nodes']}
        isolated_node_ids = {node['id'] for node in analysis['isolated_nodes']}
        main_graph_nodes = [node for node in graph_data['nodes'] if node['id'] not in isolated_node_ids]
        
        if not main_graph_nodes:
            return edges
        
        # For each isolated node, create at least one connection to main graph
        for isolated_node in analysis['isolated_nodes']:
            node_id = isolated_node['id']
            node_type = isolated_node['type']
            
            # Find best match in main graph based on type compatibility
            best_target = None
            best_score = 0
            
            for target_node in main_graph_nodes:
                target_type = target_node['type']
                
                # Type compatibility scoring
                score = self._get_type_compatibility_score(node_type, target_type)
                
                if score > best_score:
                    best_score = score
                    best_target = target_node
            
            # Create connection to best target
            if best_target and best_score > 0:
                edge_type = self._get_default_edge_type(node_type, best_target['type'])
                edges.append(self._create_edge(
                    node_id, 
                    best_target['id'], 
                    edge_type,
                    f"General inference: {node_type} → {best_target['type']}"
                ))
        
        return edges
    
    def _get_type_compatibility_score(self, source_type: str, target_type: str) -> float:
        """Get compatibility score between node types for inference."""
        compatibility_matrix = {
            'Evidence': {
                'Hypothesis': 0.9,
                'Causal_Mechanism': 0.8,
                'Event': 0.7,
                'Alternative_Explanation': 0.6
            },
            'Actor': {
                'Event': 0.9,
                'Causal_Mechanism': 0.7,
                'Hypothesis': 0.5
            },
            'Condition': {
                'Event': 0.8,
                'Causal_Mechanism': 0.8,
                'Actor': 0.7,
                'Hypothesis': 0.6
            },
            'Event': {
                'Causal_Mechanism': 0.8,
                'Hypothesis': 0.6,
                'Event': 0.5
            },
            'Data_Source': {
                'Evidence': 0.9,
                'Hypothesis': 0.7,
                'Causal_Mechanism': 0.6
            }
        }
        
        return compatibility_matrix.get(source_type, {}).get(target_type, 0.3)
    
    def _get_default_edge_type(self, source_type: str, target_type: str) -> str:
        """Get default edge type for node type combination."""
        edge_type_matrix = {
            'Evidence': {
                'Hypothesis': 'provides_evidence_for',
                'Causal_Mechanism': 'supports',
                'Event': 'confirms_occurrence',
                'Alternative_Explanation': 'tests_alternative'
            },
            'Actor': {
                'Event': 'initiates',
                'Causal_Mechanism': 'provides_evidence_for',
                'Hypothesis': 'provides_evidence_for'
            },
            'Condition': {
                'Event': 'enables',
                'Causal_Mechanism': 'enables',
                'Actor': 'constrains',
                'Hypothesis': 'enables'
            },
            'Event': {
                'Causal_Mechanism': 'part_of_mechanism',
                'Hypothesis': 'provides_evidence_for',
                'Event': 'causes'
            },
            'Data_Source': {
                'Evidence': 'weighs_evidence',
                'Hypothesis': 'weighs_evidence',
                'Causal_Mechanism': 'weighs_evidence'
            }
        }
        
        return edge_type_matrix.get(source_type, {}).get(target_type, 'provides_evidence_for')
    
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
            'source_id': source_id,
            'target_id': target_id,
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
        
        self.enhanced_repair_prompt_template = """
PROCESS TRACING CONNECTIVITY REPAIR TASK

You are analyzing a process tracing graph that has disconnected nodes. Your task is to suggest logical connections based on the original text and methodological requirements.

ORIGINAL TEXT CONTEXT:
{original_text}

DISCONNECTED NODES REQUIRING CONNECTIONS:
{disconnected_nodes}

MAIN GRAPH NODES (potential connection targets):
{main_graph_summary}

ONTOLOGY CONSTRAINTS:
{ontology_constraints}

TASK: For each disconnected node, suggest 1-3 logical connections to the main graph. Use the original text to understand semantic relationships and apply process tracing methodology.

METHODOLOGICAL REQUIREMENTS:
- Evidence should test, support, or refute Hypotheses and Mechanisms
- Actors should initiate Events they are associated with
- Conditions should enable or constrain Events, Mechanisms, or Actors
- Alternative Explanations should be tested by Evidence
- Events should be part of Mechanisms or cause other Events

CONNECTION PRINCIPLES:
1. Base connections on semantic similarity and historical logic
2. Prioritize methodologically important Evidence→Hypothesis relationships
3. Connect Actors to Events they historically initiated
4. Link Conditions to what they enable/constrain
5. Ensure Alternative Explanations have supporting/refuting evidence

REQUIRED OUTPUT FORMAT (JSON only, no explanation):
{{
  "new_edges": [
    {{
      "source_id": "node1", 
      "target_id": "node2", 
      "type": "edge_type", 
      "properties": {{
        "reasoning": "Brief explanation based on original text",
        "confidence": 0.8
      }}
    }}
  ]
}}
"""
    
    def repair_disconnections(self, graph_data: Dict, disconnected_analysis: Dict, original_text: str = "", focus: str = "general") -> Dict:
        """
        Use LLM to repair disconnections that couldn't be handled by inference rules.
        
        Args:
            graph_data: Original graph data
            disconnected_analysis: Analysis of disconnected nodes
            original_text: Original source text for context
            
        Returns:
            Dictionary with new edges to add
        """
        # Format disconnected nodes for prompt
        disconnected_summary = self._format_disconnected_nodes_enhanced(disconnected_analysis['isolated_nodes'], graph_data)
        
        # Create enhanced main graph summary with full descriptions
        main_graph_summary = self._create_enhanced_graph_summary(graph_data, disconnected_analysis)
        
        # Get ontology constraints
        ontology_info = self._get_ontology_constraints()
        
        # Construct focused repair prompt based on repair focus
        repair_prompt = self._create_focused_repair_prompt(
            focus, original_text, disconnected_summary, main_graph_summary, ontology_info, disconnected_analysis
        )
        
        # Call LLM for repair suggestions
        try:
            import os
            from dotenv import load_dotenv
            import google.generativeai as genai
            import json
            
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("[WARN] No GEMINI_API_KEY found, skipping LLM repair")
                return {"new_edges": []}
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name)
            
            print(f"[INFO] Calling LLM for disconnection repair...")
            response = model.generate_content(repair_prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            repair_result = json.loads(response_text)
            
            # Validate suggested edges
            validated_edges = self._validate_suggested_edges(repair_result.get('new_edges', []), graph_data)
            
            print(f"[INFO] LLM suggested {len(repair_result.get('new_edges', []))} edges, {len(validated_edges)} validated")
            
            return {"new_edges": validated_edges}
            
        except Exception as e:
            print(f"[ERROR] LLM repair failed: {e}")
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
    
    def _format_disconnected_nodes_enhanced(self, isolated_nodes: List[Dict], graph_data: Dict) -> str:
        """Format disconnected nodes with full context for enhanced repair prompt."""
        formatted = []
        
        # Create node lookup for full details
        node_lookup = {node['id']: node for node in graph_data['nodes']}
        
        for node in isolated_nodes:
            node_id = node['id']
            full_node = node_lookup.get(node_id, {})
            
            formatted.append(f"""
Node ID: {node_id}
Type: {node['type']}
Description: {node.get('description', full_node.get('properties', {}).get('description', 'No description'))}
Expected Connections: {', '.join(node.get('expected_connections', []))}
Properties: {full_node.get('properties', {})}
""")
        
        return '\n'.join(formatted)
    
    def _create_enhanced_graph_summary(self, graph_data: Dict, analysis: Dict) -> str:
        """Create enhanced summary with full node descriptions for potential targets."""
        if analysis['connected_components'] <= 1:
            return "Single connected component with all nodes"
        
        # Find main connected component nodes
        isolated_node_ids = {node['id'] for node in analysis['isolated_nodes']}
        connected_nodes = [node for node in graph_data['nodes'] if node['id'] not in isolated_node_ids]
        
        # Group by type with descriptions
        type_groups = {}
        for node in connected_nodes[:20]:  # Limit to prevent token overflow
            node_type = node['type']
            if node_type not in type_groups:
                type_groups[node_type] = []
            
            description = node.get('properties', {}).get('description', node.get('description', 'No description'))
            type_groups[node_type].append({
                'id': node['id'],
                'description': description[:100] + ('...' if len(description) > 100 else '')
            })
        
        summary_parts = []
        for node_type, nodes in type_groups.items():
            summary_parts.append(f"\n{node_type} nodes:")
            for node in nodes[:3]:  # Show top 3 of each type
                summary_parts.append(f"  {node['id']}: {node['description']}")
            if len(nodes) > 3:
                summary_parts.append(f"  ... and {len(nodes) - 3} more {node_type} nodes")
        
        return '\n'.join(summary_parts)
    
    def _get_ontology_constraints(self) -> str:
        """Get ontology constraints for valid edge types."""
        return """
VALID EDGE TYPES AND CONSTRAINTS:
- Evidence → Hypothesis: tests_hypothesis, supports, refutes, provides_evidence_for
- Evidence → Causal_Mechanism: tests_mechanism, supports, refutes
- Evidence → Event: confirms_occurrence, disproves_occurrence
- Evidence → Alternative_Explanation: tests_alternative, supports_alternative, refutes_alternative
- Actor → Event: initiates
- Condition → Event/Mechanism/Hypothesis: enables, constrains
- Event → Event: causes
- Event → Causal_Mechanism: part_of_mechanism
- Hypothesis → Causal_Mechanism: explains_mechanism
- Data_Source → Evidence/Hypothesis/Mechanism: weighs_evidence
"""
    
    def _create_focused_repair_prompt(self, focus: str, original_text: str, disconnected_summary: str, 
                                     main_graph_summary: str, ontology_info: str, disconnected_analysis: Dict) -> str:
        """Create focused repair prompts based on specific repair strategy."""
        
        base_context = f"""ORIGINAL TEXT:
{original_text[:4000] if original_text else "No original text provided"}

MAIN GRAPH CONTEXT:
{main_graph_summary}

ONTOLOGY CONSTRAINTS:
{ontology_info}

"""
        
        if focus == "evidence_hypothesis":
            focus_prompt = f"""FOCUSED REPAIR MISSION: Evidence-Hypothesis Connections

Priority: Connect isolated Evidence nodes to relevant Hypotheses using 'supports', 'refutes', or 'tests_hypothesis' relationships.

{base_context}

DISCONNECTED COMPONENTS:
{disconnected_summary}

TASK: Based on the original text, identify which Evidence should connect to which Hypotheses in the main graph. Focus on:
- supports: Evidence that supports a hypothesis
- refutes: Evidence that contradicts a hypothesis  
- tests_hypothesis: Evidence that directly tests a hypothesis

Output ONLY new edges connecting Evidence to Hypotheses:"""

        elif focus == "alternatives":
            focus_prompt = f"""FOCUSED REPAIR MISSION: Alternative Explanation Connections

Priority: Connect Alternative_Explanation nodes to supporting/refuting Evidence using 'supports_alternative' or 'refutes_alternative' relationships.

{base_context}

DISCONNECTED COMPONENTS:
{disconnected_summary}

TASK: Based on the original text, identify which Evidence should connect to Alternative_Explanations. Focus on:
- supports_alternative: Evidence supporting an alternative explanation
- refutes_alternative: Evidence contradicting an alternative explanation

Output ONLY new edges connecting Evidence to Alternative_Explanations:"""

        elif focus == "data_sources":
            focus_prompt = f"""FOCUSED REPAIR MISSION: Data Source Connections

Priority: Connect Data_Source nodes to their related Evidence using 'weighs_evidence' relationships.

{base_context}

DISCONNECTED COMPONENTS:
{disconnected_summary}

TASK: Based on the original text, identify which Data_Sources should connect to which Evidence. Focus on:
- weighs_evidence: Data source that provides or weighs evidence

Output ONLY new edges connecting Data_Sources to Evidence:"""

        elif focus == "aggressive":
            focus_prompt = f"""FOCUSED REPAIR MISSION: Aggressive General Connectivity

Priority: Connect any remaining disconnected nodes to the main graph using the most logical relationships based on content similarity and methodological requirements.

{base_context}

DISCONNECTED COMPONENTS:
{disconnected_summary}

TASK: Based on the original text, make aggressive but justified connections to integrate all remaining disconnected components. Consider:
- Content similarity and thematic relationships
- Process tracing methodological requirements
- Semantic connections that maintain analytical integrity

Be more liberal with connections but ensure they are traceable to the original text:"""

        else:
            focus_prompt = f"""GENERAL REPAIR MISSION: Connectivity Enhancement

{base_context}

DISCONNECTED COMPONENTS:
{disconnected_summary}

TASK: Based on the original text, identify missing relationships that should connect these disconnected nodes to the main graph:"""

        return focus_prompt + """

{{
  "new_edges": [
    {{
      "source_id": "disconnected_node_id",
      "target_id": "main_graph_node_id",
      "type": "relationship_type",
      "properties": {{
        "reasoning": "Explanation from original text",
        "confidence": 0.8,
        "llm_generated": true
      }}
    }}
  ]
}}"""
    
    def _validate_suggested_edges(self, suggested_edges: List[Dict], graph_data: Dict) -> List[Dict]:
        """Validate LLM-suggested edges against ontology constraints."""
        valid_edges = []
        node_ids = {node['id'] for node in graph_data['nodes']}
        
        # Valid edge type combinations
        valid_combinations = {
            ('Evidence', 'Hypothesis'): ['tests_hypothesis', 'supports', 'refutes', 'provides_evidence_for'],
            ('Evidence', 'Causal_Mechanism'): ['tests_mechanism', 'supports', 'refutes'],
            ('Evidence', 'Event'): ['confirms_occurrence', 'disproves_occurrence'],
            ('Evidence', 'Alternative_Explanation'): ['tests_alternative', 'supports_alternative', 'refutes_alternative'],
            ('Actor', 'Event'): ['initiates'],
            ('Condition', 'Event'): ['enables', 'constrains'],
            ('Condition', 'Causal_Mechanism'): ['enables', 'constrains'],
            ('Condition', 'Hypothesis'): ['enables', 'constrains'],
            ('Condition', 'Actor'): ['constrains'],
            ('Event', 'Event'): ['causes'],
            ('Event', 'Causal_Mechanism'): ['part_of_mechanism'],
            ('Hypothesis', 'Causal_Mechanism'): ['explains_mechanism'],
            ('Data_Source', 'Evidence'): ['weighs_evidence'],
            ('Data_Source', 'Hypothesis'): ['weighs_evidence'],
            ('Data_Source', 'Causal_Mechanism'): ['weighs_evidence'],
        }
        
        # Create node type lookup
        node_types = {node['id']: node['type'] for node in graph_data['nodes']}
        
        for edge in suggested_edges:
            source_id = edge.get('source_id')
            target_id = edge.get('target_id')
            edge_type = edge.get('type')
            
            # Validate node existence
            if source_id not in node_ids or target_id not in node_ids:
                print(f"[WARN] Skipping edge with invalid nodes: {source_id} -> {target_id}")
                continue
            
            # Validate edge type
            source_type = node_types[source_id]
            target_type = node_types[target_id]
            type_combo = (source_type, target_type)
            
            if type_combo in valid_combinations and edge_type in valid_combinations[type_combo]:
                # Convert to expected format
                validated_edge = {
                    'id': f"llm_repair_{source_id}_{target_id}_{edge_type}",
                    'source_id': source_id,
                    'target_id': target_id,
                    'type': edge_type,
                    'properties': edge.get('properties', {})
                }
                validated_edge['properties']['llm_generated'] = True
                valid_edges.append(validated_edge)
            else:
                print(f"[WARN] Invalid edge type {edge_type} for {source_type} -> {target_type}")
        
        return valid_edges
    
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


def repair_graph_connectivity(graph_data: Dict, original_text: str = "") -> Dict:
    """
    Main function to repair graph connectivity issues.
    
    Args:
        graph_data: Dictionary with 'nodes' and 'edges' arrays
        original_text: Original source text for LLM context
        
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
    
    # Multi-pass LLM repair for remaining disconnections
    if original_text and updated_analysis['connected_components'] > 1:
        print("Attempting multi-pass LLM-assisted repair for remaining disconnections...")
        
        llm_repairer = LLMDisconnectionRepairer()
        total_llm_edges = []
        current_analysis = updated_analysis
        
        # Pass 1: Focus on Evidence-Hypothesis connections
        print("Pass 1: Evidence-Hypothesis connections...")
        pass1_result = llm_repairer.repair_disconnections(updated_graph, current_analysis, original_text, focus="evidence_hypothesis")
        pass1_edges = pass1_result.get('new_edges', [])
        if pass1_edges:
            updated_graph['edges'] = updated_graph['edges'] + pass1_edges
            total_llm_edges.extend(pass1_edges)
            current_analysis = analyze_connectivity(updated_graph)
            print(f"Pass 1 added {len(pass1_edges)} connections. Components: {current_analysis['connected_components']}")
        
        # Pass 2: Focus on Alternative Explanations
        if current_analysis['connected_components'] > 1:
            print("Pass 2: Alternative Explanation connections...")
            pass2_result = llm_repairer.repair_disconnections(updated_graph, current_analysis, original_text, focus="alternatives")
            pass2_edges = pass2_result.get('new_edges', [])
            if pass2_edges:
                updated_graph['edges'] = updated_graph['edges'] + pass2_edges
                total_llm_edges.extend(pass2_edges)
                current_analysis = analyze_connectivity(updated_graph)
                print(f"Pass 2 added {len(pass2_edges)} connections. Components: {current_analysis['connected_components']}")
        
        # Pass 3: Focus on Data Source connections
        if current_analysis['connected_components'] > 1:
            print("Pass 3: Data Source connections...")
            pass3_result = llm_repairer.repair_disconnections(updated_graph, current_analysis, original_text, focus="data_sources")
            pass3_edges = pass3_result.get('new_edges', [])
            if pass3_edges:
                updated_graph['edges'] = updated_graph['edges'] + pass3_edges
                total_llm_edges.extend(pass3_edges)
                current_analysis = analyze_connectivity(updated_graph)
                print(f"Pass 3 added {len(pass3_edges)} connections. Components: {current_analysis['connected_components']}")
        
        # Pass 4: General aggressive repair for remaining components
        if current_analysis['connected_components'] > 2:
            print("Pass 4: Aggressive general repair...")
            pass4_result = llm_repairer.repair_disconnections(updated_graph, current_analysis, original_text, focus="aggressive")
            pass4_edges = pass4_result.get('new_edges', [])
            if pass4_edges:
                updated_graph['edges'] = updated_graph['edges'] + pass4_edges
                total_llm_edges.extend(pass4_edges)
                current_analysis = analyze_connectivity(updated_graph)
                print(f"Pass 4 added {len(pass4_edges)} connections. Components: {current_analysis['connected_components']}")
        
        if total_llm_edges:
            print(f"LLM multi-pass repair suggested {len(total_llm_edges)} total connections.")
            print(f"Final connectivity: {current_analysis['connected_components']} components.")
            
            if current_analysis['connected_components'] == 1:
                print("SUCCESS: Achieved full connectivity with multi-pass LLM assistance!")
            elif current_analysis['connected_components'] < updated_analysis['connected_components']:
                print(f"PROGRESS: Reduced components from {updated_analysis['connected_components']} to {current_analysis['connected_components']}")
            
            return updated_graph
    
    if not original_text:
        print("No original text provided for LLM repair. Consider providing text context for better results.")
    
    print("Complex disconnections may require manual review or enhanced inference rules.")
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