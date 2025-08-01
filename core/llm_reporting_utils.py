import json
from process_trace_advanced import query_gemini

def generate_narrative_summary_with_llm(structured_data_dict, summary_focus_prompt):
    """
    Use an LLM to generate a concise analytical narrative summary for a given structured data dict and focus prompt.
    Args:
        structured_data_dict (dict): The data to be summarized.
        summary_focus_prompt (str): Instruction for the summary focus.
    Returns:
        str: LLM's textual summary.
    """
    data_str = json.dumps(structured_data_dict, indent=2)
    prompt = f"""
You are an expert academic writer. Based on the following structured data:

{data_str}

Please write a concise (2-5 sentences) analytical narrative summary focusing on the following:
{summary_focus_prompt}

Ensure your summary is objective, directly supported by the provided data, and uses clear, analytical language.
"""
    llm_response = query_gemini(
        text_content="",  # No main text, just use prompt
        schema=None,
        system_instruction_text=prompt
    )
    return llm_response.strip() 