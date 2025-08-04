"""
ADVANCED PROCESS‑TRACING GRAPH – VERSION 3.0
Robust Gemini streaming → JSON → Visualization with comprehensive ontology

Expected JSON structure for LLM output (based on advanced ontology):

{
  "nodes": [
    {"id": "evt1", "type": "Event", "properties": {"description": "Tea Act passed", "timestamp": "1773-05-10", "certainty": 0.9}},
    {"id": "hyp1", "type": "Hypothesis", "properties": {"description": "Taxation without representation causes unrest", "prior_probability": 0.6, "status": "active"}},
    {"id": "evd1", "type": "Evidence", "properties": {"description": "Public protests against Tea Act", "type": "straw_in_the_wind", "source": "doc1"}},
    {"id": "ds1", "type": "Data_Source", "properties": {"type": "document", "credibility": 0.8}}
  ],
  "edges": [
    {"source_id": "evd1", "target_id": "hyp1", "type": "tests_hypothesis", "properties": {"probative_value": 0.3, "test_result": "passed"}},
    {"source_id": "ds1", "target_id": "evd1", "type": "provides_evidence", "properties": {}}
  ]
}

"""

import os, glob, re, json, ast, sys, textwrap, time
from itertools import count
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import webbrowser
from core.ontology import NODE_TYPES, EDGE_TYPES, NODE_COLORS

# Check if Google Generative AI package is installed
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("❌ Google Generative AI package not found.")
    print("   Please install it with: pip install google-generativeai")
    print("   Or activate the environment where it's installed.")
    print("\n🔍 You can still use this script to visualize existing JSON data.")

# ────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")  # Load from .env file
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input_text")
# Default: use the first .txt file in a project subdirectory if not specified
INPUT_FILE_PATH = None
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith('.txt'):
            INPUT_FILE_PATH = os.path.join(root, file)
            break
    if INPUT_FILE_PATH:
        break
if not INPUT_FILE_PATH:
    raise FileNotFoundError("No .txt file found in input_text/ or its subdirectories.")

# Determine project name from input file path
project_name = os.path.relpath(os.path.dirname(INPUT_FILE_PATH), INPUT_DIR)
project_name = project_name.replace(os.sep, "_")  # flatten subdirs to single name
# Timestamp for output files
now_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# Output paths
OUTPUT_DIR = os.path.join("output_data", project_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"{project_name}_{now_str}_graph.json")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, f"{project_name}_{now_str}_graph.html")
MODEL_NAME  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Load model from .env file
MAX_RETRIES = 3  # Maximum number of retries for Gemini API

# ────────────────────────────────────────────────────────────────────
# GEMINI PROMPT WITH EXPANDED ONTOLOGY
# ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an expert in causal process tracing methodology following Van Evera and Beach & Pedersen approaches. Extract a structured causal graph from the provided text using the following extended ontology. Focus on creating rich, flexible connections that reflect academic process tracing standards with particular attention to CAUSAL MECHANISMS:

EXTRACTION PRIORITIES:
1. MECHANISMS: Identify Causal_Mechanism nodes that explain HOW causation works step-by-step
   - Connect events to mechanisms via part_of_mechanism edges (role: trigger/intermediate/outcome)  
   - Test mechanism operation with evidence via tests_mechanism edges
   - Link hypotheses to mechanisms via explains_mechanism edges

2. ALTERNATIVE EXPLANATIONS: Extract Alternative_Explanation nodes for competing theories
   - Test alternatives with supporting/refuting evidence
   - Compare strength against primary hypotheses
   - Mark status (active/eliminated/supported)

3. ACTORS & AGENCY: Identify Actor nodes with intentions, beliefs, and constraints
   - Connect actors to events they initiate
   - Model strategic interactions and decision-making
   - Include capabilities and limitations

4. CONDITIONS & CONTEXT: Extract Condition nodes for scope conditions and constraints
   - Model enabling conditions that make events/mechanisms possible
   - Identify constraining conditions that limit outcomes
   - Specify temporal and spatial scope

FLEXIBLE CONNECTION PATTERNS:
- Use Evidence → Event connections (confirms_occurrence/disproves_occurrence) to establish whether events actually occurred
- Use Event → Hypothesis connections (provides_evidence_for) to show how events serve as evidence for broader patterns  
- Use Evidence → Hypothesis connections (supports/refutes/tests_hypothesis) for direct hypothesis testing
- Use flexible supports/refutes connections between different node types as appropriate
- Use Actor → Event connections (initiates) when actors directly start or launch specific events
- Use Data_Source → Evidence connections (provides_evidence) when sources supply evidence
- Apply Van Evera diagnostic types (hoop, smoking_gun, straw_in_the_wind, doubly_decisive) systematically

CRITICAL EDGE TYPES TO PRIORITIZE:
- disproves_occurrence: Evidence showing an event did NOT happen (contrary evidence)
- initiates: Actors who directly started, launched, or began specific events
- provides_evidence: Data sources (documents, interviews, observations) that supply evidence
- refutes: Evidence that contradicts or challenges hypotheses, events, or mechanisms
- tests_alternative: Evidence that systematically tests alternative explanations ("tests the alternative")
- explains_mechanism: Hypotheses that explain how causal mechanisms operate ("explains how the mechanism works")
- weighs_evidence: Evidence comparisons showing relative strength ("weighs more heavily than")

INTEGRATION PATTERNS FOR CONNECTIVITY:
- Link hypotheses to mechanisms via explains_mechanism edges
- Connect alternatives to evidence via tests_alternative, supports_alternative, refutes_alternative
- Use weighs_evidence to compare evidence strength across competing explanations
- Connect mechanisms to each other via enables/constrains for interaction analysis

Output your answer as a JSON object with two arrays: 'nodes' and 'edges'.

Each node must have:
- id: unique string
- type: one of [Event, Hypothesis, Evidence, Causal_Mechanism, Alternative_Explanation, Actor, Condition, Data_Source]
- properties: a dictionary of the node's properties (see below)

Each edge must have:
- source_id: id of the source node
- target_id: id of the target node
- type: one of [causes, supports, refutes, tests_hypothesis, tests_mechanism, confirms_occurrence, disproves_occurrence, provides_evidence_for, part_of_mechanism, explains_mechanism, supports_alternative, refutes_alternative, initiates, enables, constrains, provides_evidence, tests_alternative, weighs_evidence]
- properties: a dictionary of the edge's properties (see below)

# Issue #84 Fix: Align prompt properties with ontology schema
Node Types (with Properties):
1. Event
   - description (string, required)
   - timestamp (string, optional)
   - date (string, optional)
   - start_date (string, optional)
   - end_date (string, optional)
   - location (string, optional)
   - certainty (float 0-1, optional)
   - type (string: triggering, intermediate, outcome, unspecified, optional)
   - is_point_in_time (boolean, optional)
2. Hypothesis
   - description (string, required)
   - prior_probability (float 0-1, optional)
   - posterior_probability (float 0-1, optional)
   - status (string: active, supported, partially_supported, refuted, undetermined, optional)
3. Evidence
   - description (string, required)
   - type (string: REQUIRED Van Evera diagnostic type)
     * hoop: necessary but not sufficient (hypothesis fails if this evidence is absent)
     * smoking_gun: sufficient but not necessary (hypothesis confirmed if this evidence is present)
     * straw_in_the_wind: neither necessary nor sufficient (weakly suggestive)
     * doubly_decisive: both necessary and sufficient (confirms one hypothesis, eliminates others)
     * bayesian: general Bayesian evidence
     * general: unspecified evidence type
   - source (string, optional)
   - certainty (float 0-1, optional)
   - credibility (float 0-1, optional)
4. Causal_Mechanism
   - description (string, required): How the causal mechanism works step-by-step
   - confidence (float 0-1, optional): Confidence mechanism operates as described
   - completeness (float 0-1, optional): How complete our understanding is
   - status (string: hypothetical, supported, refuted, partial, unspecified, optional)
5. Alternative_Explanation
   - description (string, required): Alternative causal explanation
   - probability (float 0-1, optional): Estimated likelihood
   - status (string: active, eliminated, supported, undetermined, optional)
   - key_predictions (list of strings, optional): Distinguishing implications that separate from main hypotheses
6. Actor
   - name (string, required): Actor identification
   - role (string, optional): Position or function in events
   - intentions (string, optional): Goals and motivations
   - beliefs (string, optional): What they believe to be true
   - constraints (string, optional): Limitations on their actions
   - capabilities (string, optional): What they can actually accomplish
7. Condition
   - description (string, required): Condition description
   - type (string: background, enabling, constraining, scope, optional)
   - necessity (float 0-1, optional): How necessary for outcomes
   - temporal_scope (string, optional): When condition applies
   - spatial_scope (string, optional): Where condition applies
8. Data_Source
   - type (string: interview, document, observation, artifact, general, required)
   - credibility (float 0-1, optional): Reliability of source
   - bias_risk (float 0-1, optional): Risk of bias in source

Edge Types (with flexible domain/range for academic process tracing):
1. causes (Event → Event)
   - certainty (float 0-1, optional)
   - mechanism_id (string, optional)
   - type (string: direct, indirect, optional)
   
2. supports (Evidence/Event → Hypothesis/Event/Causal_Mechanism)
   - probative_value (float 0-1, optional): strength of support
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - target_type (string: event_occurrence, causal_relationship, mechanism_operation, general, optional)
   - certainty (float 0-1, optional)
   
3. refutes (Evidence/Event → Hypothesis/Event/Causal_Mechanism)  
   - probative_value (float 0-1, optional): strength of refutation
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - target_type (string: event_occurrence, causal_relationship, mechanism_operation, general, optional)
   - certainty (float 0-1, optional)
   
4. tests_hypothesis (Evidence/Event → Hypothesis)
   - probative_value (float 0-1, optional)
   - test_result (string: passed, failed, ambiguous, optional)
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   
5. tests_mechanism (Evidence/Event → Causal_Mechanism)
   - probative_value (float 0-1, optional)
   - test_result (string: passed, failed, ambiguous, optional)  
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   
6. confirms_occurrence (Evidence → Event)
   - certainty (float 0-1, optional): confidence evidence confirms event occurred
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   
7. disproves_occurrence (Evidence → Event)
   - certainty (float 0-1, optional): confidence evidence disproves event occurred
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   
8. provides_evidence_for (Event → Hypothesis/Causal_Mechanism)
   - probative_value (float 0-1, optional): how strongly event serves as evidence
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - reasoning (string, optional): explanation of how event provides evidence

9. part_of_mechanism (Event → Causal_Mechanism)
   - role (string: trigger, intermediate, outcome, facilitating): event's role in mechanism
   - sequence_position (int, optional): step number in mechanism sequence
   - necessity (float 0-1, optional): how necessary event is for mechanism

10. tests_mechanism (Evidence/Event → Causal_Mechanism)
   - probative_value (float 0-1, optional): strength of mechanism test
   - test_result (string: passed, failed, ambiguous, inconclusive, optional)
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - mechanism_aspect (string: existence, operation, completeness, optional)

11. explains_mechanism (Hypothesis → Causal_Mechanism)
   - certainty (float 0-1, optional): confidence in explanation
   - type_of_claim (string: existence, operation, necessity, sufficiency, optional)
   - scope (string, optional): under what conditions explanation applies

12. supports_alternative (Evidence → Alternative_Explanation)
   - probative_value (float 0-1, optional): strength of support
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - comparative_strength (float 0-1, optional): relative strength compared to main hypothesis

13. refutes_alternative (Evidence → Alternative_Explanation)
   - probative_value (float 0-1, optional): strength of refutation
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - refutation_strength (string: weak, moderate, strong, decisive, optional)

14. initiates (Actor → Event)
   - intentionality (string: deliberate, accidental, coerced, optional)
   - capability_assessment (float 0-1, optional): actor's capability to initiate event
   - constraint_factors (string, optional): what limited the actor's actions

15. enables (Condition → Event/Causal_Mechanism)
   - necessity (float 0-1, optional): how necessary condition is
   - temporal_scope (string, optional): when enabling condition applies
   - enabling_type (string: background, structural, contingent, optional)

16. constrains (Condition → Event/Causal_Mechanism)
   - constraint_strength (float 0-1, optional): how strongly it constrains
   - temporal_scope (string, optional): when constraint applies
   - constraint_type (string: prevents, limits, channels, optional)

17. provides_evidence (Data_Source → Evidence)
   - credibility (float 0-1, optional): reliability of the data source
   - bias_risk (float 0-1, optional): risk of bias in the source
   - certainty (float 0-1, optional): confidence that source provides evidence

18. tests_alternative (Evidence/Event → Alternative_Explanation)
   - probative_value (float 0-1, optional): how strongly evidence tests the alternative
   - diagnostic_type (string: hoop, smoking_gun, straw_in_the_wind, doubly_decisive, general, optional)
   - test_result (string: supports, refutes, inconclusive, optional): result of testing

19. weighs_evidence (Evidence → Evidence)
   - comparison_strength (float 0-1, optional): relative strength compared to target evidence
   - comparison_type (string: stronger_than, weaker_than, equivalent_to, complements, contradicts, optional)
   - reasoning (string, optional): explanation of comparative strength

Output format example:
{{
  "nodes": [
    {{"id": "evt1", "type": "Event", "properties": {{"description": "Tea Act passed", "timestamp": "1773-05-10", "certainty": 0.9}}}},
    {{"id": "hyp1", "type": "Hypothesis", "properties": {{"description": "Taxation without representation causes unrest", "prior_probability": 0.6, "status": "active"}}}},
    {{"id": "evd1", "type": "Evidence", "properties": {{"description": "Public protests against Tea Act", "type": "straw_in_the_wind", "source": "doc1"}}}},
    {{"id": "mech1", "type": "Causal_Mechanism", "properties": {{"description": "British taxation creates economic pressure leading to organized colonial resistance", "confidence": 0.8, "status": "supported"}}}},
    {{"id": "alt1", "type": "Alternative_Explanation", "properties": {{"description": "Economic interests of merchants drove opposition", "probability": 0.3, "status": "active"}}}},
    {{"id": "actor1", "type": "Actor", "properties": {{"name": "Samuel Adams", "role": "Colonial leader", "intentions": "Mobilize resistance against British rule"}}}},
    {{"id": "cond1", "type": "Condition", "properties": {{"description": "British naval dominance", "type": "constraining", "necessity": 0.8}}}}
  ],
  "edges": [
    {{"source_id": "evd1", "target_id": "evt1", "type": "confirms_occurrence", "properties": {{"certainty": 0.8, "diagnostic_type": "smoking_gun"}}}},
    {{"source_id": "evt1", "target_id": "mech1", "type": "part_of_mechanism", "properties": {{"role": "trigger", "sequence_position": 1, "necessity": 0.9}}}},
    {{"source_id": "evd1", "target_id": "mech1", "type": "tests_mechanism", "properties": {{"probative_value": 0.8, "test_result": "passed", "diagnostic_type": "smoking_gun", "mechanism_aspect": "operation"}}}},
    {{"source_id": "hyp1", "target_id": "mech1", "type": "explains_mechanism", "properties": {{"certainty": 0.7, "type_of_claim": "operation"}}}},
    {{"source_id": "evd1", "target_id": "alt1", "type": "refutes_alternative", "properties": {{"probative_value": 0.7, "diagnostic_type": "hoop", "refutation_strength": "moderate"}}}},
    {{"source_id": "actor1", "target_id": "evt1", "type": "initiates", "properties": {{"intentionality": "deliberate", "capability_assessment": 0.8}}}},
    {{"source_id": "cond1", "target_id": "mech1", "type": "constrains", "properties": {{"constraint_strength": 0.7, "constraint_type": "limits"}}}}
  ]
}}

TEXT TO ANALYZE:
{text}
"""

# ────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────
def read_all_text(path: str) -> str:
    """Read all text files in the given directory and concatenate them."""
    files = glob.glob(os.path.join(path, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {path}")

    big_text, total = "", 0
    print(f"[INFO] Reading {len(files)} file(s):")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            txt = f.read()
            big_text += txt + "\n\n"
            total += len(txt)
            print(f"  • {os.path.basename(fp):<40s} ({len(txt):,} chars)")
    print(f"[INFO] Total characters loaded: {total:,}\n")
    return big_text


def clean_json_block(raw: str) -> str:
    """Remove ``` fences, language tags, stray commas, BOM, etc."""
    # strip code fences
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "")
    # remove leading / trailing whitespace & BOM
    cleaned = cleaned.strip().lstrip("\ufeff")
    # remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # replace null with empty string for description fields to ensure they exist
    cleaned = re.sub(r'"description":\s*null', '"description": ""', cleaned)
    return cleaned


def parse_json(raw: str) -> dict:
    """Parse JSON from raw string, handling common issues."""
    raw_clean = clean_json_block(raw)
    try:
        return json.loads(raw_clean)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_clean)
        except Exception as e:
            snippet = textwrap.shorten(raw_clean, width=300, placeholder=" […]")
            print("\n[WARN]  JSON parsing failed. First 300 chars Gemini sent:\n"
                  "────────────────────────────────────────────────────")
            print(snippet)
            print("────────────────────────────────────────────────────")
            raise e


def validate_json_against_ontology(data: dict) -> tuple[bool, list[str]]:
    """
    Validate the JSON data against the ontology definition (new structure).
    Returns a tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    # Check basic structure
    if not isinstance(data, dict):
        return False, ["Data is not a dictionary"]
    
    if "nodes" not in data:
        return False, ["Missing 'nodes' array in data"]
    
    if "edges" not in data:
        return False, ["Missing 'edges' array in data"]
    
    # Empty graph is valid
    if len(data["nodes"]) == 0 and len(data["edges"]) == 0:
        return True, []
        
    # Map of node IDs for edge validation
    node_ids = {}
    node_types = {}
    
    # Validate nodes
    for i, node in enumerate(data["nodes"]):
        if not isinstance(node, dict):
            errors.append(f"Node at index {i} is not a dictionary")
            continue
        if "id" not in node:
            errors.append(f"Node at index {i} is missing required 'id' field")
        else:
            node_id = node["id"]
            if node_id in node_ids:
                errors.append(f"Duplicate node ID: {node_id}")
            node_ids[node_id] = i
        if "type" not in node:
            errors.append(f"Node {node.get('id', f'at index {i}')} is missing required 'type' field")
        else:
            node_type = node["type"]
            node_types[node.get('id', f'at index {i}')] = node_type
            if node_type not in NODE_TYPES:
                errors.append(f"Node {node.get('id', f'at index {i}')} has invalid type: {node_type}")
            else:
                # Check required fields in properties
                props = node.get("properties", {})
                for prop, prop_def in NODE_TYPES[node_type]["properties"].items():
                    if prop_def.get("required") and prop not in props:
                        errors.append(f"Node {node.get('id', f'at index {i}')} of type {node_type} is missing required property: {prop}")
                    if prop in props:
                        val = props[prop]
                        # Type check
                        if prop_def["type"] == "float":
                            if not isinstance(val, (float, int)):
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' should be a float")
                            if "min" in prop_def and val < prop_def["min"]:
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' below min {prop_def['min']}")
                            if "max" in prop_def and val > prop_def["max"]:
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' above max {prop_def['max']}")
                        if prop_def["type"] == "string":
                            if not isinstance(val, str):
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' should be a string")
                            if "allowed_values" in prop_def and val not in prop_def["allowed_values"]:
                                errors.append(f"Node {node.get('id', f'at index {i}')} property '{prop}' has invalid value '{val}'")
    # Validate edges
    for i, edge in enumerate(data["edges"]):
        if not isinstance(edge, dict):
            errors.append(f"Edge at index {i} is not a dictionary")
            continue
        if "source_id" not in edge:
            errors.append(f"Edge at index {i} is missing required 'source_id' field")
        elif edge["source_id"] not in node_ids:
            errors.append(f"Edge at index {i} references non-existent source node: {edge['source_id']}")
        if "target_id" not in edge:
            errors.append(f"Edge at index {i} is missing required 'target_id' field")
        elif edge["target_id"] not in node_ids:
            errors.append(f"Edge at index {i} references non-existent target node: {edge['target_id']}")
        if "type" not in edge:
            errors.append(f"Edge at index {i} is missing required 'type' field")
        else:
            edge_type = edge["type"]
            if edge_type not in EDGE_TYPES:
                errors.append(f"Edge at index {i} has invalid type: {edge_type}")
            elif "source_id" in edge and "target_id" in edge and edge["source_id"] in node_ids and edge["target_id"] in node_ids:
                source_type = node_types.get(edge["source_id"])
                target_type = node_types.get(edge["target_id"])
                if source_type and target_type:
                    if source_type not in EDGE_TYPES[edge_type]["domain"]:
                        errors.append(f"Edge {edge_type} at index {i} has incompatible source node type: {source_type}")
                    if target_type not in EDGE_TYPES[edge_type]["range"]:
                        errors.append(f"Edge {edge_type} at index {i} has incompatible target node type: {target_type}")
            # Check edge properties
            props = edge.get("properties", {})
            for prop, prop_def in EDGE_TYPES[edge_type]["properties"].items():
                if prop_def.get("required") and prop not in props:
                    errors.append(f"Edge {edge_type} at index {i} is missing required property: {prop}")
                if prop in props:
                    val = props[prop]
                    if prop_def["type"] == "float":
                        if not isinstance(val, (float, int)):
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' should be a float")
                        if "min" in prop_def and val < prop_def["min"]:
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' below min {prop_def['min']}")
                        if "max" in prop_def and val > prop_def["max"]:
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' above max {prop_def['max']}")
                    if prop_def["type"] == "string":
                        if not isinstance(val, str):
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' should be a string")
                        if "allowed_values" in prop_def and val not in prop_def["allowed_values"]:
                            errors.append(f"Edge {edge_type} at index {i} property '{prop}' has invalid value '{val}'")
    return len(errors) == 0, errors


def query_gemini(big_text: str, retry_count=0) -> dict:
    """Query Gemini API with retry mechanism for validation failures and intelligent caching."""
    
    # Phase 3A: LLM Response Caching Integration
    from core.llm_cache import get_cache
    cache = get_cache()
    
    # Generate cache key
    cache_key = cache.generate_cache_key(
        text=big_text, 
        prompt_template=PROMPT_TEMPLATE,
        model_name=MODEL_NAME,
        additional_params={"retry_count": retry_count}
    )
    
    # Try cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"[CACHE HIT] Using cached result for text ({len(big_text)} chars)")
        return cached_result
    
    print(f"[CACHE MISS] Making fresh LLM call for text ({len(big_text)} chars)")
    print("[DEBUG] About to query Gemini...", flush=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [types.Content(role="user",
                              parts=[types.Part.from_text(
                                  text=PROMPT_TEMPLATE.format(text=big_text))])]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    print(f"[RUN] Streaming from Gemini (attempt {retry_count + 1}/{MAX_RETRIES})…")
    json_str = ""
    chunk_count = 0
    for chunk in client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents, config=config):
        if hasattr(chunk, "text"):
            json_str += chunk.text
        chunk_count += 1
        print(f"  chunk {chunk_count}")

    print(f"[OK] Stream complete. Received {chunk_count} chunks. Parsing JSON…")
    
    try:
        # Parse the returned JSON
        graph_data = parse_json(json_str)
        
        # Validate against ontology
        is_valid, errors = validate_json_against_ontology(graph_data)
        
        if is_valid:
            print("[OK] Validation successful: JSON conforms to ontology!")
            # Cache the successful result
            cache.put(cache_key, graph_data, MODEL_NAME, PROMPT_TEMPLATE)
            print(f"[CACHE STORE] Cached result for future use")
            return graph_data
        else:
            print("[WARN] Validation failed. Errors found:")
            for error in errors:
                # Handle Unicode in error messages
                clean_error = str(error).encode('ascii', errors='replace').decode('ascii')
                print(f"  [WARN] {clean_error}")
                
            # Retry if we haven't exceeded the maximum attempts
            if retry_count < MAX_RETRIES - 1:
                print(f"[RETRY] Retrying query ({retry_count + 2}/{MAX_RETRIES})...")
                time.sleep(2)  # Short delay between retries
                return query_gemini(big_text, retry_count + 1)
            else:
                print("[ERROR] Maximum retry attempts reached. Using last response despite validation errors.")
                # Cache even failed validation results to avoid repeated failures
                cache.put(cache_key, graph_data, MODEL_NAME, PROMPT_TEMPLATE, ttl_seconds=1800)  # Shorter TTL for failed validation
                print(f"[CACHE STORE] Cached validation-failed result with shorter TTL")
                return graph_data
                
    except Exception as e:
        # Handle Unicode in exception messages
        error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"[ERROR] Error processing response: {error_msg}")
        
        # Retry if we haven't exceeded the maximum attempts
        if retry_count < MAX_RETRIES - 1:
            print(f"[RETRY] Retrying query ({retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(2)  # Short delay between retries
            return query_gemini(big_text, retry_count + 1)
        else:
            print("[ERROR] Maximum retry attempts reached. Raising error.")
            raise e


def build_visualization(data, html_out):
    """
    Creates a standalone HTML visualization using vis.js
    for the comprehensive process tracing ontology.
    """
    # Create formatted nodes for vis.js
    vis_nodes = []
    for node in data["nodes"]:
        props = node.get("properties", {})
        # Prepare label based on node type
        label = f"{node['type']}: {props.get('description', node['id'])}"
        if node["type"] == "Actor":
            label = f"{node['type']}: {props.get('name', props.get('description', node['id']))}"
        # Truncate long labels
        if len(label) > 100:
            label = label[:97] + "..."
        # Prepare tooltip with all properties
        tooltip = "<div style='max-width:300px;'>"
        tooltip += f"<strong>ID:</strong> {node['id']}<br>"
        tooltip += f"<strong>Type:</strong> {node['type']}<br>"
        for k, v in props.items():
            tooltip += f"<strong>{k.replace('_', ' ').title()}:</strong> {v}<br>"
        tooltip += "</div>"
        vis_nodes.append({
            "id": node["id"],
            "label": label,
            "group": node["type"],
            "color": NODE_COLORS.get(node["type"], "#dddddd"),
            "title": tooltip
        })
    # Create formatted edges for vis.js
    vis_edges = []
    node_ids = {n["id"] for n in data["nodes"]}
    for edge in data["edges"]:
        if edge["source_id"] not in node_ids:
            print(f"⚠️  Skipping edge from non-existent source node '{edge['source_id']}' to '{edge['target_id']}'.")
            continue
        if edge["target_id"] not in node_ids:
            print(f"⚠️  Skipping edge from '{edge['source_id']}' to non-existent target node '{edge['target_id']}'.")
            continue
        props = edge.get("properties", {})
        tooltip = "<div style='max-width:250px;'>"
        tooltip += f"<strong>Type:</strong> {edge['type']}<br>"
        for k, v in props.items():
            tooltip += f"<strong>{k.replace('_', ' ').title()}:</strong> {v}<br>"
        tooltip += "</div>"
        vis_edges.append({
            "from": edge["source_id"],
            "to": edge["target_id"],
            "label": edge["type"],
            "title": tooltip,
            "width": props.get("certainty", 1) if isinstance(props.get("certainty"), (int, float)) else 1
        })
    # Create groups JSON for the visualization
    groups_json = {}
    for node_type, color in NODE_COLORS.items():
        groups_json[node_type] = {"color": {"background": color, "border": "#444444"}}
    
    # Create legend items HTML
    legend_items = ""
    for node_type, color in NODE_COLORS.items():
        legend_items += f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <div>{node_type.replace('_', ' ')}</div>
        </div>"""
    
    # Generate timestamp for footer
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create HTML template parts
    html_head = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Advanced Process Tracing Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #visualization {{
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }}
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #ccc;
        }}
        h1 {{
            margin-top: 0;
        }}
        .footer {{
            margin-top: 10px;
            font-size: 0.8em;
            color: #666;
        }}
        .controls {{
            margin-bottom: 10px;
        }}
        button {{
            margin-right: 10px;
            padding: 5px 10px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <h1>Advanced Process Tracing Visualization</h1>
    
    <div class="controls">
        <button onclick="stabilize()">Stabilize Layout</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="fitView()">Fit View</button>
    </div>
    
    <div class="legend">{legend_items}
    </div>
    
    <div id="visualization"></div>
    
    <div class="footer">
        Generated: {timestamp} • 
        Source: {os.path.basename(OUTPUT_JSON)} •
        Nodes: {len(vis_nodes)} • Edges: {len(vis_edges)}
    </div>"""
    
    html_script = f"""
    <script type="text/javascript">
        // Create a network
        var container = document.getElementById('visualization');
        
        // Provide the data in the vis format
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        // Options for the network
        var options = {{
            nodes: {{
                shape: 'box',
                margin: 12,
                widthConstraint: {{
                    maximum: 250
                }},
                font: {{
                    size: 14
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                arrows: 'to',
                smooth: {{
                    type: 'dynamic',
                    forceDirection: 'none',
                    roundness: 0.5
                }},
                font: {{
                    size: 12,
                    align: 'middle'
                }},
                color: {{
                    inherit: false,
                    color: '#666666'
                }},
                width: 2,
                shadow: true
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -60,
                    centralGravity: 0.01,
                    springLength: 150,
                    springConstant: 0.05,
                    damping: 0.4
                }},
                stabilization: {{
                    iterations: 300
                }}
            }},
            layout: {{
                improvedLayout: true
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }},
            groups: {json.dumps(groups_json)}
        }};
        
        // Initialize the network
        var network = new vis.Network(container, data, options);
        
        // Define control functions
        function stabilize() {{
            network.stabilize(100);
        }}
        
        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        function fitView() {{
            network.fit();
        }}
        
        // After the network is stabilized, disable physics for better user interaction
        network.on("stabilizationIterationsDone", function () {{
            setTimeout(function() {{
                network.setOptions({{ physics: false }});
                physicsEnabled = false;
            }}, 1000); // Small delay to ensure layout is complete
        }});
    </script>
</body>
</html>"""
    
    # Combine the HTML parts
    full_html = html_head + html_script
    
    # Write the HTML file
    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"[SUCCESS] Graph written to {html_out}")
    
    # Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(html_out))


def read_text_file(file_path: str) -> str:
    """Read a single text file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"[INFO] Reading file: {os.path.basename(file_path)}")
    with open(file_path, encoding="utf-8") as f:
        txt = f.read()
    print(f"[INFO] Total characters loaded: {len(txt):,}\n")
    return txt


# ────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main execution function for the process tracing application."""
    try:
        # If the JSON file already exists, give the user an option to reuse it
        if os.path.exists(OUTPUT_JSON) and os.path.getsize(OUTPUT_JSON) > 0:
            print(f"[INFO] Found existing JSON data file: {OUTPUT_JSON}")
            if not HAS_GEMINI:
                print("[INFO] Will use existing JSON data since Gemini API is not available.")
                use_existing = True
            else:
                response = input("   Would you like to reuse this data instead of querying Gemini? (y/n): ").strip().lower()
                use_existing = response == 'y' or response == 'yes'
        else:
            use_existing = False

        if use_existing:
            # Load existing JSON data
            try:
                with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                print(f"[OK] Successfully loaded data from {OUTPUT_JSON}")
            except Exception as e:
                print(f"[ERROR] Error loading JSON data: {str(e)}")
                if not HAS_GEMINI:
                    print("[ERROR] Cannot proceed without valid JSON data and Gemini API.")
                    return
                use_existing = False
        
        if not use_existing:
            # Process requires Gemini API
            if not HAS_GEMINI:
                print("[ERROR] Cannot query Gemini API. Package not available.")
                return
                
            # Read the specified input file
            text = read_text_file(INPUT_FILE_PATH)
            
            # Query Gemini API with the text
            graph_data = query_gemini(text)

            # Save the graph data to a JSON file
            try:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=4)
                print(f"[SAVE] Raw graph data saved to {OUTPUT_JSON}")
            except IOError as e:
                print(f"[WARN]  Error saving JSON data to {OUTPUT_JSON}: {e}")

        # Create visualization (works regardless of where data came from)
        build_visualization(graph_data, OUTPUT_HTML)
        
    except Exception as e:
        print(f"[ERROR] Error in main execution: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback:\n{traceback_str}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        import traceback
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
