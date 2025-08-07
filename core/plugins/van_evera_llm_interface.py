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

# Add the project root to sys.path to import universal_llm
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from universal_llm_kit.universal_llm import get_llm
from .van_evera_llm_schemas import (
    VanEveraPredictionEvaluation, 
    BayesianParameterEstimation,
    CausalRelationshipAnalysis,
    ProcessTracingConclusion,
    ContentBasedClassification
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
    
    def _get_structured_response(self, prompt: str, response_model: Type[T]) -> T:
        """
        Get structured response from LLM using the specified Pydantic model
        
        Args:
            prompt: The prompt to send to LLM
            response_model: Pydantic model class for structured output
            
        Returns:
            Instance of response_model with LLM data
        """
        try:
            # Get structured output using universal LLM
            response_text = self.llm.structured_output(prompt, response_model)
            
            # Parse JSON response
            response_data = json.loads(response_text)
            
            # Create and validate Pydantic model
            structured_response = response_model.model_validate(response_data)
            
            logger.info(f"Successfully generated structured {response_model.__name__}")
            return structured_response
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed for {response_model.__name__}: {e}")
            # Return a fallback response
            return self._create_fallback_response(response_model, f"JSON parsing error: {e}")
            
        except Exception as e:
            logger.error(f"LLM call failed for {response_model.__name__}: {e}")
            # Return a fallback response
            return self._create_fallback_response(response_model, f"LLM error: {e}")
    
    def _create_fallback_response(self, response_model: Type[T], error_message: str) -> T:
        """Create a fallback response when LLM calls fail"""
        
        if response_model == VanEveraPredictionEvaluation:
            return VanEveraPredictionEvaluation(
                test_result="INCONCLUSIVE",
                confidence_score=0.5,
                diagnostic_reasoning=f"Fallback evaluation due to: {error_message}",
                evidence_assessment="Unable to assess due to LLM error",
                theoretical_mechanism_evaluation="Unable to evaluate due to LLM error",
                elimination_implications=["Unable to determine due to LLM error"],
                evidence_quality="low",
                evidence_coverage=0.5,
                indicator_matches=0,
                publication_quality_assessment="Unable to assess due to LLM error",
                methodological_soundness=0.5
            )
        
        elif response_model == BayesianParameterEstimation:
            return BayesianParameterEstimation(
                prior_probability=0.5,
                likelihood_given_hypothesis=0.5,
                likelihood_given_not_hypothesis=0.5,
                prior_justification=f"Default prior due to: {error_message}",
                likelihood_reasoning=f"Default likelihoods due to: {error_message}",
                confidence_in_estimates=0.3,
                uncertainty_sources=[f"LLM error: {error_message}"]
            )
        
        elif response_model == CausalRelationshipAnalysis:
            return CausalRelationshipAnalysis(
                causal_strength=0.5,
                causal_mechanism=f"Unable to analyze due to: {error_message}",
                temporal_precedence=False,
                covariation=0.5,
                alternative_explanations_ruled_out=0.3,
                potential_confounders=["Unable to identify due to LLM error"],
                potential_mediators=["Unable to identify due to LLM error"],
                causal_reasoning=f"Fallback analysis due to: {error_message}",
                uncertainty_assessment=f"High uncertainty due to: {error_message}"
            )
        
        elif response_model == ProcessTracingConclusion:
            return ProcessTracingConclusion(
                hypothesis_status="INCONCLUSIVE",
                confidence_level=0.3,
                academic_summary=f"Unable to generate conclusion due to: {error_message}",
                methodology_assessment="Unable to assess due to LLM error",
                supporting_evidence_strength=0.3,
                contradicting_evidence_assessment="Unable to assess due to LLM error",
                publication_quality_score=0.3,
                recommendations_for_improvement=[f"Resolve LLM error: {error_message}"]
            )
        
        elif response_model == ContentBasedClassification:
            return ContentBasedClassification(
                recommended_diagnostic_type="straw_in_wind",
                classification_confidence=0.3,
                content_analysis=f"Unable to analyze due to: {error_message}",
                theoretical_fit="Unable to assess due to LLM error",
                alternative_classifications=[{"type": "fallback", "reason": error_message}],
                theoretical_sophistication=0.3,
                methodological_rigor=0.3
            )
        
        else:
            raise ValueError(f"Unknown response model: {response_model}")


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
            logger.error(f"LLM query failed: {e}")
            return f"LLM query failed: {e}"
    
    return llm_query_func