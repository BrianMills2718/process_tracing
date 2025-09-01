# Avoid circular import - query_llm will be passed as parameter
from .structured_models import EvidenceAssessment

def refine_evidence_assessment_with_llm(evidence_description, text_content, context_info=None, query_llm_func=None):
    """
    Use an LLM to refine the diagnostic type and probative value of evidence.
    Uses structured output with Pydantic models for reliable parsing.
    
    Args:
        evidence_description (str): Description of the evidence.
        text_content (str): Source text content.
        context_info (str, optional): Additional context information.
        query_llm_func (callable, optional): LLM query function to use.
    Returns:
        EvidenceAssessment: Structured assessment from LLM.
    """
    
    # Use default LLM function if none provided
    if query_llm_func is None:
        try:
            # Import here to avoid circular dependency
            import google.generativeai as genai
            import os
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("No API key available")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            def default_query_llm(text, schema=None, system_instruction=None, **kwargs):
                prompt = f"{system_instruction}\n\n{text}" if system_instruction else text
                if schema and hasattr(schema, 'model_json_schema'):
                    prompt += f"\n\nRespond with valid JSON matching this schema: {schema.model_json_schema()}"
                response = model.generate_content(prompt)
                return response.text
            
            query_llm_func = default_query_llm
        except Exception as e:
            print(f"[ERROR] Cannot access LLM for evidence assessment: {e}")
            return None
    
    prompt = f"""
You are an expert in process tracing methodology and Bayesian reasoning.
Given the following Evidence and source text context:

Evidence Description: {evidence_description}
Context: {context_info or 'General analysis context'}
Source Text: {text_content[:1000]}...

Tasks:
1.  **Refine Evidence Type (Van Evera):** Based on the evidence's relationship to the hypothesis, classify its diagnostic type. Is it a 'hoop' test (necessary for the hypothesis to hold, its absence refutes H?), a 'smoking_gun' (its presence strongly confirms H, being very unlikely if H were false?), 'straw_in_the_wind' (weakly suggestive), or 'doubly_decisive' (confirms H and refutes alternatives)? Explain your reasoning. If the current type is 'general', strive to assign a more specific diagnostic type.
2.  **Assess Probative Value (Qualitative Bayesian Likelihoods):**
    * How likely would you expect to observe this Evidence if the Hypothesis were TRUE (P(E|H))? (e.g., Very High, High, Medium, Low, Very Low).
    * How likely would you expect to observe this Evidence if the Hypothesis were FALSE (P(E|~H))? (e.g., Very High, High, Medium, Low, Very Low).
    * Briefly justify these likelihood assessments.
3.  **Suggest Numerical Probative Value:** Based on your likelihood assessments, suggest a refined numerical `probative_value` for the edge connecting the evidence to the hypothesis (e.g., a float where a higher absolute value indicates stronger impact; or directly suggest P(E|H) and P(E|~H) as floats between 0 and 1).
"""
    
    # Use structured output with EvidenceAssessment model  
    llm_response_text = query_llm_func(
        text=text_content,
        schema=EvidenceAssessment,
        system_instruction=prompt
    )
    
    # Parse JSON response and create Pydantic object
    import json
    try:
        # Clean the response - remove markdown code blocks if present
        clean_response = llm_response_text.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        # Parse JSON and create Pydantic object
        response_data = json.loads(clean_response)
        llm_response = EvidenceAssessment(**response_data)
        return llm_response
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Failed to parse LLM response as EvidenceAssessment: {e}")
        print(f"[ERROR] Raw response: {llm_response_text[:500]}...")
        # Return None so the enhancement gracefully fails
        return None 