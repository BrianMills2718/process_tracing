# Core Ontology Definitions for Process Tracing Toolkit (Advanced)

import json
import typing_extensions as typing
from typing import List, Dict, Any, Type

def load_ontology_from_config(config_path="config/ontology_config.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get("node_types", {}), config.get("edge_types", {}), config.get("node_colors", {})

NODE_TYPES, EDGE_TYPES, NODE_COLORS = load_ontology_from_config()

# --- Gemini Schema Export Utility ---
def get_gemini_graph_schema() -> Any:
    """
    Returns a Python type (TypedDict) representing the expected Gemini output schema
    for a process tracing graph, based on the ontology above.
    The schema is a dict with two keys: 'nodes' and 'edges'.
    Each node has a 'type' and 'properties'; each edge has 'type', 'source', 'target', and 'properties'.
    """
    class Node(typing.TypedDict, total=False):
        id: str
        type: str
        properties: dict
    class Edge(typing.TypedDict, total=False):
        id: str
        type: str
        source: str
        target: str
        properties: dict
    class Graph(typing.TypedDict):
        nodes: List[Node]
        edges: List[Edge]
    return Graph 

# --- Gemini Strict JSON Schema Export Utility ---
def get_gemini_graph_json_schema():
    """
    Returns a strict JSON schema dict for a process tracing graph, based on the ontology above.
    Only allows properties defined in the ontology; does not explicitly forbid extra properties (no additionalProperties: False).
    Maps 'float' to 'number' and 'datetime' to 'string' for Gemini compatibility.
    """
    def map_type(t):
        if t == 'float':
            return 'number'
        if t == 'datetime':
            return 'string'
        return t
    def node_schema(node_type, node_def):
        props = node_def['properties']
        required = [k for k, v in props.items() if v.get('required')]
        return {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'type': {'type': 'string', 'enum': [node_type]},
                'properties': {
                    'type': 'object',
                    'properties': {
                        k: ({'type': map_type(v['type'])} if 'allowed_values' not in v else {'type': map_type(v['type']), 'enum': v['allowed_values']})
                        for k, v in props.items()
                    },
                    'required': required
                }
            },
            'required': ['id', 'type', 'properties']
        }
    def edge_schema(edge_type, edge_def):
        props = edge_def['properties']
        required = [k for k, v in props.items() if v.get('required')]
        return {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'type': {'type': 'string', 'enum': [edge_type]},
                'source': {'type': 'string'},
                'target': {'type': 'string'},
                'properties': {
                    'type': 'object',
                    'properties': {
                        k: ({'type': map_type(v['type'])} if 'allowed_values' not in v else {'type': map_type(v['type']), 'enum': v['allowed_values']})
                        for k, v in props.items()
                    },
                    'required': required
                }
            },
            'required': ['id', 'type', 'source', 'target', 'properties']
        }
    # Create unified schemas without anyOf for better Gemini compatibility
    # Collect all possible node properties across all node types
    all_node_props = {}
    for ntype, ndef in NODE_TYPES.items():
        for prop, prop_def in ndef['properties'].items():
            if prop not in all_node_props:
                all_node_props[prop] = prop_def
    
    # Create unified node schema with all possible properties
    unified_node_schema = {
        'type': 'object',
        'properties': {
            'id': {'type': 'string'},
            'type': {'type': 'string', 'enum': list(NODE_TYPES.keys())},
            'properties': {
                'type': 'object',
                'properties': {
                    k: ({'type': map_type(v['type'])} if 'allowed_values' not in v else {'type': map_type(v['type']), 'enum': v['allowed_values']})
                    for k, v in all_node_props.items()
                },
                'required': []  # No properties are universally required
            }
        },
        'required': ['id', 'type', 'properties']
    }
    
    # Collect all possible edge properties across all edge types
    all_edge_props = {}
    for etype, edef in EDGE_TYPES.items():
        for prop, prop_def in edef['properties'].items():
            if prop not in all_edge_props:
                all_edge_props[prop] = prop_def
    
    # Create unified edge schema with all possible properties
    unified_edge_schema = {
        'type': 'object',
        'properties': {
            'id': {'type': 'string'},
            'type': {'type': 'string', 'enum': list(EDGE_TYPES.keys())},
            'source': {'type': 'string'},
            'target': {'type': 'string'},
            'properties': {
                'type': 'object',
                'properties': {
                    k: ({'type': map_type(v['type'])} if 'allowed_values' not in v else {'type': map_type(v['type']), 'enum': v['allowed_values']})
                    for k, v in all_edge_props.items()
                },
                'required': []  # No properties are universally required
            }
        },
        'required': ['id', 'type', 'source', 'target', 'properties']
    }
    
    return {
        'type': 'object',
        'properties': {
            'nodes': {
                'type': 'array',
                'items': unified_node_schema,
                'default': []
            },
            'edges': {
                'type': 'array',
                'items': unified_edge_schema,
                'default': []
            }
        },
        'required': ['nodes', 'edges']
    } 