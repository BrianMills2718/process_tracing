from process_trace_advanced import query_llm
from .structured_models import MechanismAssessment

def elaborate_mechanism_with_llm(mechanism_node, linked_event_nodes, original_text_context, ontology_schema):
    """
    Use an LLM to elaborate a Causal_Mechanism node by:
    1. Writing a narrative connecting the constituent events.
    2. Suggesting missing micro-steps (as new Event nodes).
    3. Assessing internal coherence.
    4. Suggesting refined properties (confidence, level_of_detail).
    
    Uses structured output with Pydantic models for reliable parsing.
    
    Args:
        mechanism_node (dict): The Causal_Mechanism node.
        linked_event_nodes (list): List of Event node dicts linked as part_of_mechanism.
        original_text_context (str): Relevant text snippet(s) from the source.
        ontology_schema (dict): The JSON schema for Event nodes.
    Returns:
        MechanismAssessment: Structured assessment from LLM.
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
1.  **Assess Completeness:** How complete is this mechanism (0.0-1.0)? Are all necessary steps present?
2.  **Assess Plausibility:** How plausible is this mechanism (0.0-1.0)? Does the causal logic make sense?
3.  **Evaluate Evidence Support:** What level of evidence supports this mechanism (strong/moderate/weak/none)?
4.  **Identify Missing Elements:** What elements are missing that would strengthen the mechanism?
5.  **Provide Improvement Suggestions:** What specific suggestions would improve mechanism completeness?
6.  **Detailed Reasoning:** Provide comprehensive reasoning for your assessments.

Focus on the causal logic, evidence strength, and gaps in the mechanism.
"""
    
    # Use structured output with MechanismAssessment model
    llm_response = query_llm(
        text_content="",  # No main text, just use prompt
        schema=MechanismAssessment,
        system_instruction_text=prompt,
        use_structured_output=True
    )
    
    return llm_response 