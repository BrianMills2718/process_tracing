# Architecture Analysis: Structured Output Migration

## Current Problems Identified

### 1. Using Wrong Tools
- **Current**: Manual `google.generativeai` imports and custom JSON parsing
- **Available**: LiteLLM with Router, structured output, Pydantic 2.11.7
- **Issue**: Not leveraging modern structured output capabilities

### 2. Prompt Engineering Anti-Patterns
- **Current**: 17,857 character monster prompt with multiple "CRITICAL" sections
- **Issue**: Fighting LLM with text instead of using schema constraints
- **Symptom**: Whack-a-mole edge type fixes, conflicting instructions

### 3. Manual Validation Architecture
```python
# Current (wrong):
json_str = chunk.text  # Manual streaming
graph_data = parse_json(json_str)  # Custom parsing with ast.literal_eval
is_valid, errors = validate_json_against_ontology(graph_data)  # Manual validation
```

### 4. 2019 vs 2024 Approaches
- **2019**: Examples + manual parsing + custom validation
- **2024**: Schema + structured output + automatic validation
- **We're using**: 2019 approach in 2024

## Planned Architecture

### 1. Pydantic Schema Definition
```python
from pydantic import BaseModel
from typing import Literal, List

class Node(BaseModel):
    id: str
    type: Literal["Event", "Hypothesis", "Evidence", "Causal_Mechanism", 
                  "Alternative_Explanation", "Actor", "Condition", "Data_Source"]
    properties: dict

class Edge(BaseModel):
    source_id: str
    target_id: str
    type: Literal["causes", "supports", "refutes", "tests_hypothesis", 
                  "confirms_occurrence", "disproves_occurrence", ...]
    properties: dict

class ProcessTracingGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
```

### 2. Clean Extraction Method
```python
def extract_graph(text: str) -> ProcessTracingGraph:
    prompt = f"""Extract causal relationships from this text using Van Evera methodology.
    
    Text: {text}"""
    
    result = structured(prompt, ProcessTracingGraph)
    return ProcessTracingGraph.model_validate_json(result)
```

## Hypothesis
Schema constraints will solve edge type coverage issues more reliably than prompt engineering.

## Next Steps
1. Deep investigation of potential issues with structured output approach
2. Prototype implementation 
3. Comparative testing: current vs structured approach
4. Migration plan if validation successful