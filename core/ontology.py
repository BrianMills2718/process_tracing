# Core Ontology Definitions for Process Tracing Toolkit

# Node types and their required/optional properties
NODE_TYPES = {
    "Event": {
        "required": ["id", "description", "type"],
        "optional": ["timestamp", "location", "certainty"],
        "valid_type_values": ["triggering", "intermediate", "outcome", "unspecified"]
    },
    "Causal_Mechanism": {
        "required": ["id", "description"],
        "optional": ["confidence", "status", "level_of_detail"]
    },
    "Hypothesis": {
        "required": ["id", "description"],
        "optional": ["prior_probability", "posterior_probability", "status", "scope"]
    },
    "Evidence": {
        "required": ["id", "description", "type"],
        "optional": ["probative_value", "certainty", "source", "credibility"],
        "valid_type_values": ["hoop", "smoking_gun", "straw_in_the_wind", "doubly_decisive", "bayesian", "general"]
    },
    "Condition": {
        "required": ["id", "description", "type"],
        "optional": ["necessity", "certainty"],
        "valid_type_values": ["background", "enabling", "facilitating", "constraining", "unspecified"]
    },
    "Actor": {
        "required": ["id", "name"],
        "optional": ["role", "intentions", "beliefs", "credibility"]
    },
    "Inference_Rule": {
        "required": ["id", "description", "type"],
        "optional": [],
        "valid_type_values": ["bayesian_updating", "abductive", "deductive", "inductive", "heuristic", "general"]
    },
    "Inferential_Test": {
        "required": ["id", "description", "type"],
        "optional": ["conditions"],
        "valid_type_values": ["hoop", "smoking_gun", "doubly_decisive", "bayesian", "general"]
    },
    "Alternative_Explanation": {
        "required": ["id", "description"],
        "optional": ["probability", "status"]
    },
    "Data_Source": {
        "required": ["id", "type"],
        "optional": ["credibility", "bias_risk"],
        "valid_type_values": ["interview", "document", "observation", "artifact", "general"]
    }
}

# Edge types and their properties
EDGE_TYPES = {
    "causes": {
        "source_types": ["Event"],
        "target_types": ["Event"],
        "required": [],
        "optional": ["certainty", "mechanism_id", "type"]
    },
    "part_of_mechanism": {
        "source_types": ["Event"],
        "target_types": ["Causal_Mechanism"],
        "required": [],
        "optional": ["role"]
    },
    "tests_hypothesis": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["inferential_test_id", "probative_value", "test_result"]
    },
    "tests_mechanism": {
        "source_types": ["Evidence"],
        "target_types": ["Causal_Mechanism"],
        "required": [],
        "optional": ["inferential_test_id", "probative_value", "test_result"]
    },
    "supports_alternative": {
        "source_types": ["Evidence"],
        "target_types": ["Alternative_Explanation"],
        "required": [],
        "optional": ["probative_value", "certainty"]
    },
    "refutes_alternative": {
        "source_types": ["Evidence"],
        "target_types": ["Alternative_Explanation"],
        "required": [],
        "optional": ["probative_value", "certainty"]
    },
    "enables": {
        "source_types": ["Condition"],
        "target_types": ["Event", "Causal_Mechanism"],
        "required": [],
        "optional": ["necessity", "certainty", "type"]
    },
    "constrains": {
        "source_types": ["Condition"],
        "target_types": ["Event", "Causal_Mechanism"],
        "required": [],
        "optional": ["certainty", "type"]
    },
    "provides_evidence": {
        "source_types": ["Data_Source"],
        "target_types": ["Evidence"],
        "required": [],
        "optional": ["credibility", "bias_risk", "certainty"]
    },
    "initiates": {
        "source_types": ["Actor"],
        "target_types": ["Event"],
        "required": [],
        "optional": ["certainty", "intention", "agency"]
    },
    "infers": {
        "source_types": ["Inference_Rule"],
        "target_types": ["Hypothesis", "Causal_Mechanism"],
        "required": [],
        "optional": ["certainty", "logic_type"]
    },
    "updates_probability": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["prior_probability", "posterior_probability", "Bayes_factor"]
    },
    "contradicts": {
        "source_types": ["Evidence"],
        "target_types": ["Evidence"],
        "required": [],
        "optional": ["certainty", "reason"]
    },
    "supports": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["certainty", "strength"]
    },
    "refutes": {
        "source_types": ["Evidence"],
        "target_types": ["Hypothesis"],
        "required": [],
        "optional": ["certainty", "strength"]
    }
}

# Node type colors for visualization
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