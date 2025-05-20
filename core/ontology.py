# Core Ontology Definitions for Process Tracing Toolkit (Advanced)

# Node types and their required/optional properties (expanded, state-of-the-art)
NODE_TYPES = {
    "Event": {
        "properties": {
            "description": {"type": "string", "required": True},
            "timestamp": {"type": "datetime", "required": False},
            "location": {"type": "string", "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "type": {"type": "string", "allowed_values": ["triggering", "intermediate", "outcome", "unspecified"], "required": False}
        },
        "plural_name": "Events"
    },
    "Causal_Mechanism": {
        "properties": {
            "description": {"type": "string", "required": True},
            "confidence": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "status": {"type": "string", "allowed_values": ["hypothetical", "supported", "refuted", "unspecified"], "required": False},
            "level_of_detail": {"type": "string", "required": False}
        },
        "plural_name": "Causal Mechanisms"
    },
    "Hypothesis": {
        "properties": {
            "description": {"type": "string", "required": True},
            "prior_probability": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "posterior_probability": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "status": {"type": "string", "allowed_values": ["active", "confirmed", "refuted"], "required": False},
            "scope": {"type": "string", "allowed_values": ["general", "case-specific"], "required": False}
        },
        "plural_name": "Hypotheses"
    },
    "Evidence": {
        "properties": {
            "description": {"type": "string", "required": True},
            "type": {"type": "string", "allowed_values": ["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "bayesian", "general"], "required": True},
            "probative_value": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "source": {"type": "string", "required": False},
            "credibility": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "plural_name": "Evidence"
    },
    "Condition": {
        "properties": {
            "description": {"type": "string", "required": True},
            "type": {"type": "string", "allowed_values": ["background", "enabling", "facilitating", "constraining", "unspecified"], "required": True},
            "necessity": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "plural_name": "Conditions"
    },
    "Actor": {
        "properties": {
            "name": {"type": "string", "required": True},
            "role": {"type": "string", "required": False},
            "intentions": {"type": "string", "required": False},
            "beliefs": {"type": "string", "required": False},
            "credibility": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "plural_name": "Actors"
    },
    "Inference_Rule": {
        "properties": {
            "description": {"type": "string", "required": True},
            "type": {"type": "string", "allowed_values": ["bayesian_updating", "abductive", "deductive", "inductive", "heuristic", "general"], "required": True}
        },
        "plural_name": "Inference Rules"
    },
    "Inferential_Test": {
        "properties": {
            "description": {"type": "string", "required": True},
            "type": {"type": "string", "allowed_values": ["hoop", "smoking_gun", "doubly_decisive", "bayesian", "general"], "required": True},
            "conditions": {"type": "string", "required": False}
        },
        "plural_name": "Inferential Tests"
    },
    "Alternative_Explanation": {
        "properties": {
            "description": {"type": "string", "required": True},
            "probability": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "status": {"type": "string", "allowed_values": ["active", "confirmed", "refuted"], "required": False}
        },
        "plural_name": "Alternative Explanations"
    },
    "Data_Source": {
        "properties": {
            "type": {"type": "string", "allowed_values": ["interview", "document", "observation", "artifact", "general"], "required": True},
            "credibility": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "bias_risk": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "plural_name": "Data Sources"
    }
}

# Edge types and their properties (expanded)
EDGE_TYPES = {
    "causes": {
        "domain": ["Event"],
        "range": ["Event"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "mechanism_id": {"type": "string", "required": False},
            "type": {"type": "string", "allowed_values": ["direct", "indirect"], "required": False}
        },
        "label": "causes"
    },
    "part_of_mechanism": {
        "domain": ["Event"],
        "range": ["Causal_Mechanism"],
        "properties": {
            "role": {"type": "string", "required": False}
        },
        "label": "part of mechanism"
    },
    "tests_hypothesis": {
        "domain": ["Evidence"],
        "range": ["Hypothesis"],
        "properties": {
            "inferential_test_id": {"type": "string", "required": False},
            "probative_value": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "test_result": {"type": "string", "allowed_values": ["passed", "failed", "ambiguous"], "required": False}
        },
        "label": "tests hypothesis"
    },
    "tests_mechanism": {
        "domain": ["Evidence"],
        "range": ["Causal_Mechanism"],
        "properties": {
            "inferential_test_id": {"type": "string", "required": False},
            "probative_value": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "test_result": {"type": "string", "allowed_values": ["passed", "failed", "ambiguous"], "required": False}
        },
        "label": "tests mechanism"
    },
    "supports_alternative": {
        "domain": ["Evidence"],
        "range": ["Alternative_Explanation"],
        "properties": {
            "probative_value": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "label": "supports alternative"
    },
    "refutes_alternative": {
        "domain": ["Evidence"],
        "range": ["Alternative_Explanation"],
        "properties": {
            "probative_value": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "label": "refutes alternative"
    },
    "enables": {
        "domain": ["Condition"],
        "range": ["Event", "Causal_Mechanism"],
        "properties": {
            "necessity": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "type": {"type": "string", "required": False}
        },
        "label": "enables"
    },
    "constrains": {
        "domain": ["Condition"],
        "range": ["Event", "Causal_Mechanism"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "type": {"type": "string", "required": False}
        },
        "label": "constrains"
    },
    "provides_evidence": {
        "domain": ["Data_Source"],
        "range": ["Evidence"],
        "properties": {
            "credibility": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "bias_risk": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "label": "provides evidence"
    },
    "initiates": {
        "domain": ["Actor"],
        "range": ["Event"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "intention": {"type": "string", "required": False},
            "agency": {"type": "string", "required": False}
        },
        "label": "initiates"
    },
    "infers": {
        "domain": ["Inference_Rule"],
        "range": ["Hypothesis", "Causal_Mechanism"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "logic_type": {"type": "string", "required": False}
        },
        "label": "infers"
    },
    "updates_probability": {
        "domain": ["Evidence"],
        "range": ["Hypothesis"],
        "properties": {
            "prior_probability": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "posterior_probability": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "Bayes_factor": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "label": "updates probability"
    },
    "contradicts": {
        "domain": ["Evidence"],
        "range": ["Evidence"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "reason": {"type": "string", "required": False}
        },
        "label": "contradicts"
    },
    "supports": {
        "domain": ["Evidence"],
        "range": ["Hypothesis"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "label": "supports"
    },
    "refutes": {
        "domain": ["Evidence"],
        "range": ["Hypothesis"],
        "properties": {
            "certainty": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
        },
        "label": "refutes"
    }
}

# Node type colors for visualization (expanded)
NODE_COLORS = {
    "Event": "#66b3ff",
    "Causal_Mechanism": "#99ff99",
    "Hypothesis": "#ffcc00",
    "Evidence": "#ff6666",
    "Condition": "#ccccff",
    "Actor": "#ff99cc",
    "Inference_Rule": "#cc99ff",
    "Inferential_Test": "#ffb366",
    "Alternative_Explanation": "#ff9966",
    "Data_Source": "#c2c2f0"
}

# --- Gemini Schema Export Utility ---
import typing_extensions as typing
from typing import List, Dict, Any, Type

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