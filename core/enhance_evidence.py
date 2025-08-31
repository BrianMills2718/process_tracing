# Use the centralized LLM Gateway instead of direct LLM calls
from .structured_models import EvidenceAssessment

def refine_evidence_assessment_with_llm(evidence_description, text_content, context_info=None, query_llm_func=None):
    """
    Use the LLM Gateway to refine the diagnostic type and probative value of evidence.
    Uses centralized gateway for all LLM operations.
    
    Args:
        evidence_description (str): Description of the evidence.
        text_content (str): Source text content.
        context_info (str, optional): Additional context information.
        query_llm_func (callable, optional): Deprecated - gateway is used instead.
    Returns:
        EvidenceAssessment: Structured assessment from LLM.
    Raises:
        LLMRequiredError: If LLM is unavailable (fail-fast behavior).
    """
    
    # Lazy import to avoid circular dependency
    from .llm_gateway import LLMGateway
    from .llm_required import LLMRequiredError
    
    # Initialize the LLM Gateway - fail fast if unavailable
    try:
        gateway = LLMGateway()
    except LLMRequiredError as e:
        # Re-raise to enforce fail-fast behavior
        raise e
    except Exception as e:
        raise LLMRequiredError(f"Cannot initialize LLM Gateway for evidence assessment: {e}")
    
    # Use the gateway to get Van Evera diagnostic classification
    try:
        # First, determine the diagnostic type using the gateway
        diagnostic = gateway.determine_diagnostic_type(
            evidence=evidence_description,
            hypothesis=context_info or "Hypothesis under evaluation",
            test_name="evidence_assessment"
        )
        
        # Get probative value assessment
        probative_value = gateway.calculate_probative_value(
            evidence=evidence_description,
            hypothesis=context_info or "Hypothesis under evaluation",
            diagnostic_type=diagnostic.test_type
        )
        
        # Create EvidenceAssessment from gateway results
        # Map Van Evera test type to our format
        refined_type_map = {
            'hoop': 'hoop',
            'smoking_gun': 'smoking_gun',
            'doubly_decisive': 'doubly_decisive',
            'straw_in_wind': 'straw_in_the_wind'
        }
        
        refined_type = refined_type_map.get(diagnostic.test_type, 'general')
        
        # Map probative value to Bayesian likelihoods
        if probative_value > 0.7:
            likelihood_if_true = "High"
            likelihood_if_false = "Low"
        elif probative_value > 0.5:
            likelihood_if_true = "Medium-High"
            likelihood_if_false = "Medium-Low"
        elif probative_value > 0.3:
            likelihood_if_true = "Medium"
            likelihood_if_false = "Medium"
        else:
            likelihood_if_true = "Low"
            likelihood_if_false = "High"
        
        # Create structured assessment with correct fields
        assessment = EvidenceAssessment(
            evidence_id=f"evidence_{hash(evidence_description) % 10000}",  # Generate ID
            refined_evidence_type=refined_type,
            reasoning_for_type=diagnostic.reasoning,
            likelihood_P_E_given_H=f"{likelihood_if_true} ({min(0.9, probative_value + 0.2):.2f})",
            likelihood_P_E_given_NotH=f"{likelihood_if_false} ({max(0.1, 1.0 - probative_value):.2f})",
            justification_for_likelihoods=f"Based on semantic analysis: {diagnostic.reasoning[:200]}",
            suggested_numerical_probative_value=probative_value
        )
        
        return assessment
        
    except LLMRequiredError:
        # Re-raise LLM errors to maintain fail-fast behavior
        raise
    except Exception as e:
        # Log error but raise LLMRequiredError for consistency
        print(f"[ERROR] Gateway operation failed: {e}")
        raise LLMRequiredError(f"Evidence assessment requires LLM: {e}") 