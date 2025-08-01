import json
from process_trace_advanced import query_gemini

def refine_evidence_assessment_with_llm(hypothesis_node, evidence_node, edge_properties, original_text_context):
    """
    Use an LLM to refine the diagnostic type and probative value of an Evidence node linked to a Hypothesis.
    Args:
        hypothesis_node (dict): The Hypothesis node.
        evidence_node (dict): The Evidence node.
        edge_properties (dict): Properties of the supports/refutes edge.
        original_text_context (str): Source quote or expanded context.
    Returns:
        dict: LLM's refined assessment as parsed JSON.
    """
    prompt = f"""
You are an expert in process tracing methodology and Bayesian reasoning.
Given the following Hypothesis, a piece of Evidence linked to it, and the source text for the evidence:

Hypothesis:
- ID: {hypothesis_node['id']}
- Description: {hypothesis_node['properties'].get('description','')}

Evidence:
- ID: {evidence_node['id']}
- Description: {evidence_node['properties'].get('description','')}
- Current Extracted Type: {evidence_node['properties'].get('type', 'general')}
- Source Text Quote: "{edge_properties.get('source_text_quote', 'Not available.')}"

Tasks:
1.  **Refine Evidence Type (Van Evera):** Based on the evidence's relationship to the hypothesis, classify its diagnostic type. Is it a 'hoop' test (necessary for the hypothesis to hold, its absence refutes H?), a 'smoking_gun' (its presence strongly confirms H, being very unlikely if H were false?), 'straw_in_the_wind' (weakly suggestive), or 'doubly_decisive' (confirms H and refutes alternatives)? Explain your reasoning. If the current type is 'general', strive to assign a more specific diagnostic type.
2.  **Assess Probative Value (Qualitative Bayesian Likelihoods):**
    * How likely would you expect to observe this Evidence if the Hypothesis were TRUE (P(E|H))? (e.g., Very High, High, Medium, Low, Very Low).
    * How likely would you expect to observe this Evidence if the Hypothesis were FALSE (P(E|~H))? (e.g., Very High, High, Medium, Low, Very Low).
    * Briefly justify these likelihood assessments.
3.  **Suggest Numerical Probative Value:** Based on your likelihood assessments, suggest a refined numerical `probative_value` for the edge connecting the evidence to the hypothesis (e.g., a float where a higher absolute value indicates stronger impact; or directly suggest P(E|H) and P(E|~H) as floats between 0 and 1).

Output your response in JSON format:
{{
    "evidence_id": "{evidence_node['id']}",
    "refined_evidence_type": "e.g., hoop | smoking_gun | straw_in_the_wind | doubly_decisive",
    "reasoning_for_type": "...",
    "likelihood_P_E_given_H": "e.g., High (0.9)",
    "likelihood_P_E_given_NotH": "e.g., Low (0.1)",
    "justification_for_likelihoods": "...",
    "suggested_numerical_probative_value": "float (e.g., for a supports edge, P(E|H)/P(E|~H) if providing Bayes Factor, or a simpler strength score 0-1)"
}}
"""
    llm_response = query_gemini(
        text_content="",  # No main text, just use prompt
        schema=None,
        system_instruction_text=prompt
    )
    try:
        return json.loads(llm_response)
    except Exception as e:
        return {"error": str(e), "raw_response": llm_response} 