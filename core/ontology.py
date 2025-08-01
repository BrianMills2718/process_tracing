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
    node_schemas = [node_schema(ntype, ndef) for ntype, ndef in NODE_TYPES.items()]
    edge_schemas = [edge_schema(etype, edef) for etype, edef in EDGE_TYPES.items()]
    return {
        'type': 'object',
        'properties': {
            'nodes': {
                'type': 'array',
                'items': {'anyOf': node_schemas},
                'default': []
            },
            'edges': {
                'type': 'array',
                'items': {'anyOf': edge_schemas},
                'default': []
            }
        },
        'required': ['nodes', 'edges']
    } 