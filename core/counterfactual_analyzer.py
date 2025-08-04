import json
from process_trace_advanced import query_llm

def explore_counterfactual_with_llm(graph_data_json_str, counterfactual_premise_text, key_outcome_node_id):
    """
    Use an LLM to reason about the consequences of a counterfactual premise on a causal graph.
    Args:
        graph_data_json_str (str): The JSON string of the graph.
        counterfactual_premise_text (str): The counterfactual scenario.
        key_outcome_node_id (str): The ID of the outcome Event node of interest.
    Returns:
        str: LLM's structured textual analysis.
    """
    # Extract description of the key outcome node from the graph
    try:
        graph = json.loads(graph_data_json_str)
        outcome_node = next((n for n in graph.get('nodes', []) if n.get('id') == key_outcome_node_id), None)
        outcome_desc = outcome_node['properties'].get('description', 'N/A') if outcome_node else 'N/A'
    except Exception:
        outcome_desc = 'N/A'
    prompt = f"""
You are an expert in causal reasoning and process tracing.
You will be given a causal graph (in JSON format representing nodes and edges) extracted from a case study, and a counterfactual premise.

Original Causal Graph:
{graph_data_json_str}

Counterfactual Premise:
{counterfactual_premise_text}

Key Outcome of Interest (from the original graph):
Node ID: {key_outcome_node_id}
Description (from graph): {outcome_desc}

Your Tasks:
1.  **Identify Impacted Nodes/Edges:** Based on the counterfactual premise, which specific nodes or relationships (edges) in the original graph would be directly altered, invalidated, or removed? List them.
2.  **Trace Consequences:** Starting from the direct impacts, trace the plausible downstream consequences through the causal graph. Which subsequent events might not occur, or occur differently? Would any causal mechanisms be disrupted?
3.  **Assess Impact on Key Outcome:** How would the specified 'Key Outcome of Interest' likely be affected under this counterfactual premise? Would it still occur? Occur differently? Be prevented?
4.  **Alternative Pathways:** Might new causal pathways become relevant or dominant if the original pathway is disrupted by the counterfactual? Briefly describe any.
5.  **Certainty/Assumptions:** State your level of certainty in this counterfactual assessment and any key assumptions you made.

Provide your analysis as a structured textual explanation.
"""
    llm_response = query_llm(
        text_content="",  # No main text, just use prompt
        schema=None,
        system_instruction_text=prompt
    )
    return llm_response.strip() 