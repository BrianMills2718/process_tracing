import json
from process_trace_advanced import query_gemini

def elaborate_mechanism_with_llm(mechanism_node, linked_event_nodes, original_text_context, ontology_schema):
    """
    Use an LLM to elaborate a Causal_Mechanism node by:
    1. Writing a narrative connecting the constituent events.
    2. Suggesting missing micro-steps (as new Event nodes).
    3. Assessing internal coherence.
    4. Suggesting refined properties (confidence, level_of_detail).
    
    Args:
        mechanism_node (dict): The Causal_Mechanism node.
        linked_event_nodes (list): List of Event node dicts linked as part_of_mechanism.
        original_text_context (str): Relevant text snippet(s) from the source.
        ontology_schema (dict): The JSON schema for Event nodes.
    Returns:
        dict: LLM's elaboration output as parsed JSON.
    """
    # --- Prepare LLM prompt ---
    event_descriptions = "\n".join([
        f"- Event ID: {ev['id']}, Description: {ev['properties'].get('description','')}, Timestamp: {ev['properties'].get('timestamp', 'N/A')}"
        for ev in linked_event_nodes
    ])
    prompt = f"""
You are an expert in causal analysis and process tracing. Given the following Causal Mechanism and its currently identified constituent Event parts from a larger text:

Causal Mechanism:
- ID: {mechanism_node['id']}
- Description: {mechanism_node['properties'].get('description','')}
- Current Confidence: {mechanism_node['properties'].get('confidence', 'N/A')}
- Current Level of Detail: {mechanism_node['properties'].get('level_of_detail', 'N/A')}

Currently Linked Constituent Events (in approximate order if known):
{event_descriptions}

Original Text Context (Optional):
{original_text_context}

Your tasks are:
1.  **Narrative Elaboration:** Based on the provided events, write a brief (2-4 sentence) narrative explaining HOW these events connect to form the described causal mechanism. What is the processual logic?
2.  **Identify Missing Micro-Steps:** Are there any obvious logical gaps or missing micro-events between the listed constituent events that, if present, would make the mechanism operate more smoothly or completely? If so, describe them. For each missing micro-step you identify, propose it as a new potential Event node, providing a clear 'description' and a suggested 'type' ('intermediate').
3.  **Assess Internal Coherence:** Based on your elaboration and identification of missing steps, provide a brief assessment of the internal coherence of this mechanism as currently described (e.g., "Highly coherent," "Moderately coherent but with gaps," "Poorly specified").
4.  **Suggest Refined Properties:** Based on your analysis, would you suggest refining the 'confidence' or 'level_of_detail' for this Causal Mechanism? If so, what and why?

Output your response in JSON format with the following keys:
{{
    "mechanism_id": "{mechanism_node['id']}",
    "narrative_elaboration": "...",
    "identified_missing_micro_steps": [
        {{ "suggested_event_description": "...", "suggested_event_type": "intermediate" }},
        ...
    ],
    "coherence_assessment": "...",
    "suggested_confidence": "float (optional, e.g., 0.8)",
    "suggested_level_of_detail": "string (optional, e.g., high)",
    "reasoning_for_suggestions": "..."
}}
"""
    # --- Call Gemini ---
    llm_response = query_gemini(
        text_content="",  # No main text, just use prompt
        schema=None,  # No strict schema for this call
        system_instruction_text=prompt
    )
    try:
        return json.loads(llm_response)
    except Exception as e:
        return {"error": str(e), "raw_response": llm_response} 