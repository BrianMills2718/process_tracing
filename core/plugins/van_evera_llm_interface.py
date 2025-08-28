"""
Van Evera LLM Interface - Real LiteLLM Integration
Replaces mock LLM calls with structured output using Gemini 2.5 Flash
"""

import json
import logging
from typing import Dict, List, Any, Optional, Type, TypeVar
from pydantic import BaseModel
import sys
import os

# Import structured logging utilities
try:
    from ..logging_utils import log_structured_error, create_llm_context
except ImportError:
    # Fallback if logging_utils not available
    def log_structured_error(logger, message, error_category, operation_context=None, exc_info=True, **extra_context):
        logger.error(message, exc_info=exc_info)
    def create_llm_context(model_type, operation, **kwargs):
        return {"model_type": model_type, "llm_operation": operation}

# Add the project root to sys.path to import universal_llm
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from universal_llm_kit.universal_llm import get_llm
from .van_evera_llm_schemas import (
    VanEveraPredictionEvaluation, 
    BayesianParameterEstimation,
    CausalRelationshipAnalysis,
    ProcessTracingConclusion,
    ContentBasedClassification,
    TestResult
)

logger = logging.getLogger(__name__)

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)


class VanEveraLLMInterface:
    """Real LLM interface for Van Evera process tracing with structured output"""
    
    def __init__(self):
        self.llm = get_llm()
        self.model_type = "smart"  # Use smart model for academic work
    
    def evaluate_prediction_structured(self, prediction_description: str, diagnostic_type: str,
                                     theoretical_mechanism: str, evidence_context: str,
                                     necessity_logic: Optional[str] = None,
                                     sufficiency_logic: Optional[str] = None) -> VanEveraPredictionEvaluation:
        """
        Evaluate Van Evera prediction with structured output
        
        Args:
            prediction_description: The specific prediction to test
            diagnostic_type: hoop, smoking_gun, doubly_decisive, straw_in_wind
            theoretical_mechanism: The proposed causal mechanism
            evidence_context: Relevant evidence for evaluation
            necessity_logic: Logic for necessary conditions (hoop tests)
            sufficiency_logic: Logic for sufficient conditions (smoking gun tests)
            
        Returns:
            Structured VanEveraPredictionEvaluation
        """
        prompt = f"""
        Analyze this Van Evera diagnostic test using rigorous academic process tracing methodology.

        PREDICTION: {prediction_description}
        DIAGNOSTIC TYPE: {diagnostic_type}
        THEORETICAL MECHANISM: {theoretical_mechanism}

        EVIDENCE CONTEXT:
        {evidence_context}

        Van Evera Analysis Framework:
        - HOOP TEST (Necessary): Must pass to keep hypothesis viable. Failure eliminates hypothesis.
        - SMOKING GUN TEST (Sufficient): If found, strongly confirms hypothesis. Absence doesn't eliminate.
        - DOUBLY DECISIVE: Both necessary and sufficient. Definitive test.
        - STRAW IN WIND: Neither necessary nor sufficient. Provides weak support/challenge.

        {f"NECESSITY LOGIC: {necessity_logic}" if necessity_logic else ""}
        {f"SUFFICIENCY LOGIC: {sufficiency_logic}" if sufficiency_logic else ""}

        Conduct rigorous academic evaluation considering:
        1. Evidence quality and relevance to prediction
        2. Theoretical mechanism plausibility
        3. Van Evera diagnostic logic compliance
        4. Alternative explanations consideration
        5. Academic publication standards

        IMPORTANT: For elimination_implications, provide a JSON array of strings, not a single string.
        Example: ["hypothesis_A eliminated", "hypothesis_B supported"]

        Provide detailed analysis with confidence scoring and publication-quality reasoning.
        """
        
        return self._get_structured_response(prompt, VanEveraPredictionEvaluation)
    
    def estimate_bayesian_parameters(self, hypothesis: str, evidence: str, 
                                   prior_context: str) -> BayesianParameterEstimation:
        """
        Use LLM to estimate Bayesian parameters for formal analysis
        
        Args:
            hypothesis: The hypothesis being tested
            evidence: The evidence to evaluate
            prior_context: Context for prior probability estimation
            
        Returns:
            Structured Bayesian parameter estimates
        """
        prompt = f"""
        Estimate Bayesian parameters for formal probabilistic analysis of this hypothesis.

        HYPOTHESIS: {hypothesis}
        EVIDENCE: {evidence}
        PRIOR CONTEXT: {prior_context}

        Provide careful estimates for:
        1. PRIOR PROBABILITY P(H): Based on background knowledge and context
        2. LIKELIHOOD P(E|H): Probability of observing this evidence if hypothesis is true
        3. LIKELIHOOD P(E|¬H): Probability of observing this evidence if hypothesis is false

        Consider:
        - Historical base rates for similar hypotheses
        - Quality and reliability of evidence
        - Alternative explanations
        - Uncertainty in estimates
        
        Provide detailed justification for each estimate with academic rigor.
        """
        
        return self._get_structured_response(prompt, BayesianParameterEstimation)
    
    def analyze_causal_relationship(self, cause: str, effect: str, 
                                  context: str, evidence: str) -> CausalRelationshipAnalysis:
        """
        Analyze causal relationship for DoWhy integration
        
        Args:
            cause: Proposed cause
            effect: Proposed effect
            context: Historical/situational context
            evidence: Available evidence
            
        Returns:
            Structured causal analysis
        """
        prompt = f"""
        Analyze this proposed causal relationship using rigorous causal inference methodology.

        PROPOSED CAUSE: {cause}
        PROPOSED EFFECT: {effect}
        CONTEXT: {context}
        EVIDENCE: {evidence}

        Evaluate using Bradford Hill criteria and modern causal inference:
        1. TEMPORAL PRECEDENCE: Does cause precede effect?
        2. COVARIATION: Do cause and effect vary together?
        3. ALTERNATIVE EXPLANATIONS: Are alternatives ruled out?
        4. CAUSAL MECHANISM: What is the proposed mechanism?
        5. CONFOUNDERS: What variables might confound the relationship?
        6. MEDIATORS: What variables might mediate the relationship?

        Provide quantitative strength estimates and detailed reasoning suitable for
        formal causal analysis with DoWhy or similar frameworks.
        """
        
        return self._get_structured_response(prompt, CausalRelationshipAnalysis)
    
    def generate_academic_conclusion(self, hypothesis: str, test_results: List[Dict], 
                                   overall_evidence: str) -> ProcessTracingConclusion:
        """
        Generate publication-quality academic conclusion
        
        Args:
            hypothesis: The hypothesis being evaluated
            test_results: Results from various Van Evera tests
            overall_evidence: Summary of all evidence
            
        Returns:
            Structured academic conclusion
        """
        test_summary = "\n".join([
            f"- {result.get('prediction', 'Unknown')}: {result.get('test_result', 'Unknown')} "
            f"(confidence: {result.get('confidence_score', 0):.2f})"
            for result in test_results
        ])
        
        prompt = f"""
        Generate a publication-quality academic conclusion for this process tracing analysis.

        HYPOTHESIS: {hypothesis}

        TEST RESULTS:
        {test_summary}

        OVERALL EVIDENCE:
        {overall_evidence}

        Synthesize findings using Van Evera methodology standards:
        1. Overall hypothesis status (SUPPORTED/ELIMINATED/WEAKENED/INCONCLUSIVE)
        2. Confidence assessment with reasoning
        3. Evidence synthesis and quality evaluation
        4. Academic summary suitable for peer review
        5. Methodology assessment and limitations
        6. Publication quality evaluation
        7. Recommendations for improvement

        Maintain academic rigor comparable to top methodology journals.
        """
        
        return self._get_structured_response(prompt, ProcessTracingConclusion)
    
    def classify_diagnostic_type(self, evidence_description: str, hypothesis_description: str,
                               current_classification: str) -> ContentBasedClassification:
        """
        Improve diagnostic classification using content analysis
        
        Args:
            evidence_description: Description of the evidence
            hypothesis_description: Description of the hypothesis
            current_classification: Current diagnostic type classification
            
        Returns:
            Structured classification recommendation
        """
        prompt = f"""
        Analyze and improve the diagnostic classification of this evidence-hypothesis relationship.

        EVIDENCE: {evidence_description}
        HYPOTHESIS: {hypothesis_description}
        CURRENT CLASSIFICATION: {current_classification}

        Van Evera Diagnostic Types:
        - HOOP (Necessary): Evidence must be present if hypothesis is true. Absence eliminates hypothesis.
        - SMOKING GUN (Sufficient): Evidence strongly indicates hypothesis if present. Absence doesn't eliminate.
        - DOUBLY DECISIVE: Both necessary and sufficient. Definitive test.
        - STRAW IN WIND: Neither necessary nor sufficient. Provides incremental support.

        Analyze the logical relationship:
        1. Is this evidence necessary for the hypothesis to be true?
        2. Is this evidence sufficient to confirm the hypothesis?
        3. What is the theoretical relationship between evidence and hypothesis?
        4. How does content analysis support the classification?

        Recommend the most appropriate diagnostic type with detailed reasoning.
        """
        
        return self._get_structured_response(prompt, ContentBasedClassification)
    
    def classify_evidence_relationship(self, evidence_description: str, 
                                     hypothesis_description: str) -> 'EvidenceRelationshipClassification':
        """
        Classify evidence-hypothesis relationship using semantic understanding.
        Replaces keyword-based contradiction detection with LLM semantic analysis.
        
        Args:
            evidence_description: Description of the evidence
            hypothesis_description: Description of the hypothesis
            
        Returns:
            Structured classification with reasoning and probative value
        """
        prompt = f"""
        Analyze the semantic relationship between this evidence and hypothesis using Van Evera academic methodology.
        
        EVIDENCE: {evidence_description}
        HYPOTHESIS: {hypothesis_description}
        
        Determine if the evidence:
        1. SUPPORTS the hypothesis (evidence strengthens or confirms the hypothesis)
        2. REFUTES the hypothesis (evidence weakens or contradicts the hypothesis)  
        3. Is IRRELEVANT to the hypothesis (no clear relationship)
        
        Consider semantic meaning, not keyword matching. Evidence about anti-British sentiment 
        SUPPORTS hypotheses about ideological movements, regardless of economic/political keyword conflicts.
        
        For contradiction_indicators, count actual semantic contradictions (0 for supporting/irrelevant, 1-3 for refuting based on severity).
        
        Provide detailed reasoning for your classification and assess the probative value (0.0-1.0) based on:
        - Relevance to hypothesis
        - Quality and reliability of evidence
        - Strength of logical connection
        - Academic standards for process tracing
        
        Be thorough in your semantic analysis - explain WHY the evidence relates to the hypothesis.
        """
        
        # Import here to avoid circular imports
        from .van_evera_llm_schemas import EvidenceRelationshipClassification
        return self._get_structured_response(prompt, EvidenceRelationshipClassification)
    
    def _get_structured_response(self, prompt: str, response_model: Type[T], max_retries: int = 3) -> T:
        """
        Get structured response from LLM using the specified Pydantic model with basic retry
        
        Args:
            prompt: The prompt to send to LLM
            response_model: Pydantic model class for structured output
            max_retries: Maximum number of retries for transient failures only
            
        Returns:
            Instance of response_model with LLM data
            
        Raises:
            ValidationError: If Pydantic validation fails (FAIL FAST - no retry)
            json.JSONDecodeError: If JSON parsing fails after retries (FAIL FAST)
            Exception: Any other error after retries (FAIL FAST)
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Get structured output using universal LLM
                response_text = self.llm.structured_output(prompt, response_model)
                
                # Parse JSON response - FAIL FAST if invalid JSON structure
                response_data = json.loads(response_text)
                
                # Create and validate Pydantic model - FAIL FAST if schema mismatch
                structured_response = response_model.model_validate(response_data)
                
                logger.info(f"Successfully generated structured {response_model.__name__}")
                return structured_response
                
            except (ConnectionError, TimeoutError) as e:
                # Only retry on transient/network errors
                last_exception = e
                if attempt < max_retries - 1:
                    log_structured_error(
                        logger,
                        f"Attempt {attempt + 1} failed with transient error, retrying...",
                        error_category="llm_retry",
                        operation_context="structured_output_generation",
                        exc_info=True,
                        **create_llm_context(
                            self.model_type,
                            "structured_output_retry",
                            response_model=response_model.__name__,
                            attempt=attempt + 1
                        )
                    )
                    continue
                else:
                    # Exhausted retries - FAIL FAST
                    log_structured_error(
                        logger,
                        f"All {max_retries} attempts failed for {response_model.__name__}",
                        error_category="llm_error",
                        operation_context="structured_output_generation",
                        exc_info=True
                    )
                    raise
                    
            except (json.JSONDecodeError, ValidationError) as e:
                # NEVER retry on JSON/validation errors - these indicate schema/prompt issues
                log_structured_error(
                    logger,
                    f"Schema/JSON error for {response_model.__name__} - FAILING FAST",
                    error_category="llm_schema_error",
                    operation_context="structured_output_generation",
                    exc_info=True,
                    **create_llm_context(
                        self.model_type,
                        "structured_output",
                        response_model=response_model.__name__,
                        error_type=type(e).__name__
                    )
                )
                raise  # FAIL FAST immediately
                
            except Exception as e:
                # Any other error - retry once then FAIL FAST
                last_exception = e
                if attempt < max_retries - 1:
                    log_structured_error(
                        logger,
                        f"Unexpected error on attempt {attempt + 1}, retrying once...",
                        error_category="llm_unexpected_error",
                        operation_context="structured_output_generation",
                        exc_info=True
                    )
                    continue
                else:
                    # FAIL FAST after final attempt
                    log_structured_error(
                        logger,
                        f"Unexpected error after {max_retries} attempts - FAILING FAST",
                        error_category="llm_error",
                        operation_context="structured_output_generation",
                        exc_info=True
                    )
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Failed to get structured response after {max_retries} attempts")
    


# Global interface instance
_llm_interface = None

def get_van_evera_llm() -> VanEveraLLMInterface:
    """Get singleton Van Evera LLM interface"""
    global _llm_interface
    if _llm_interface is None:
        _llm_interface = VanEveraLLMInterface()
    return _llm_interface


def create_llm_query_function():
    """Create a compatible llm_query_func for existing code"""
    llm_interface = get_van_evera_llm()
    
    def llm_query_func(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """
        Compatible function for existing llm_query_func usage
        Returns simple string response for backward compatibility
        """
        try:
            response = llm_interface.llm.chat(
                prompt, 
                model_type="smart",
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            # Check if it's a timeout error
            error_category = "llm_timeout" if "timeout" in str(e).lower() else "llm_error"
            
            log_structured_error(
                logger,
                "LLM query failed",
                error_category=error_category,
                operation_context="llm_query_function",
                exc_info=True,
                **create_llm_context(
                    llm_interface.model_type,
                    "chat_completion",
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return f"LLM query failed: {e}"
    
    return llm_query_func