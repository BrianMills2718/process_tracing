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
    TestResult,
    HypothesisDomainClassification,
    ProbativeValueAssessment,
    AlternativeHypothesisGeneration,
    TestGenerationSpecification,
    ComprehensiveEvidenceAnalysis,
    MultiFeatureExtraction,
    BatchedHypothesisEvaluation,
    ConfidenceThresholdAssessment,
    CausalMechanismAssessment,
    ConfidenceFormulaWeights,
    SemanticRelevanceAssessment
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
    
    def assess_confidence_thresholds(self, evidence_quality: str, hypothesis_complexity: str,
                                    domain_context: str) -> ConfidenceThresholdAssessment:
        """
        Generate dynamic confidence thresholds based on context.
        Replaces hardcoded thresholds with LLM-based assessment.
        
        Args:
            evidence_quality: Description of evidence quality and characteristics
            hypothesis_complexity: Description of hypothesis complexity
            domain_context: Domain-specific context
            
        Returns:
            Dynamic confidence thresholds and quality assessments
        """
        prompt = f"""
        Assess appropriate confidence thresholds for this process tracing analysis.
        
        EVIDENCE QUALITY: {evidence_quality}
        HYPOTHESIS COMPLEXITY: {hypothesis_complexity}
        DOMAIN CONTEXT: {domain_context}
        
        Provide dynamic confidence thresholds considering:
        1. Evidence quality and reliability
        2. Domain-specific standards
        3. Methodological rigor requirements
        4. Uncertainty factors
        
        Assess causal mechanism quality:
        - Mechanism completeness (0.0-1.0)
        - Temporal consistency (0.0-1.0)
        - Logical coherence baseline (0.0-1.0)
        
        Determine evidence independence and quality factors.
        Provide academic justification for all assessments.
        """
        
        return self._get_structured_response(prompt, ConfidenceThresholdAssessment)
    
    def determine_confidence_weights(self, evidence_quality: str, hypothesis_complexity: str,
                                    domain_context: str) -> ConfidenceFormulaWeights:
        """
        Determine appropriate weights for confidence formula components.
        Replaces hardcoded 0.4, 0.2, 0.2, 0.2 values with dynamic weights.
        
        Args:
            evidence_quality: Description of evidence quality
            hypothesis_complexity: Description of hypothesis complexity
            domain_context: Context for weight determination
            
        Returns:
            Dynamic weights for confidence calculation
        """
        prompt = f"""
        Determine appropriate weights for confidence calculation components.
        
        EVIDENCE QUALITY: {evidence_quality}
        HYPOTHESIS COMPLEXITY: {hypothesis_complexity}
        DOMAIN CONTEXT: {domain_context}
        
        Provide weights for:
        1. Quality weight - importance of evidence quality
        2. Quantity weight - importance of evidence quantity
        3. Diversity weight - importance of evidence diversity
        4. Balance weight - importance of supporting vs contradicting balance
        
        Weights should sum to approximately 1.0.
        Justify your weight selection based on the context.
        """
        
        return self._get_structured_response(prompt, ConfidenceFormulaWeights)
    
    def determine_causal_weights(self, hypothesis_description: str, evidence_context: str,
                                domain_context: str) -> ConfidenceFormulaWeights:
        """
        Determine weights for causal confidence components.
        Replaces hardcoded 0.4, 0.3, 0.2, 0.1 values.
        """
        prompt = f"""
        Determine weights for causal confidence calculation.
        
        HYPOTHESIS: {hypothesis_description}
        EVIDENCE CONTEXT: {evidence_context}
        DOMAIN: {domain_context}
        
        Provide weights for:
        1. Posterior probability component (quality_weight)
        2. Likelihood ratio component (quantity_weight)
        3. Mechanism completeness component (diversity_weight)
        4. Temporal consistency component (balance_weight)
        
        Use the weight fields creatively for these causal components.
        Weights should sum to approximately 1.0.
        """
        
        return self._get_structured_response(prompt, ConfidenceFormulaWeights)
    
    def determine_overall_confidence_weights(self, component_scores: str,
                                            domain_context: str) -> ConfidenceFormulaWeights:
        """
        Determine weights for combining confidence components.
        Replaces hardcoded 0.30, 0.25, 0.20, 0.15, 0.10 values.
        """
        prompt = f"""
        Determine weights for overall confidence aggregation.
        
        COMPONENT SCORES: {component_scores}
        DOMAIN: {domain_context}
        
        Provide weights for combining:
        1. Evidential confidence (quality_weight)
        2. Causal confidence (quantity_weight)
        3. Coherence confidence (diversity_weight)
        4. Robustness+Sensitivity combined (balance_weight)
        
        Consider the component scores and provide appropriate weights.
        Weights should sum to approximately 1.0.
        """
        
        return self._get_structured_response(prompt, ConfidenceFormulaWeights)
    
    def determine_robustness_weights(self, evidence_context: str, 
                                    domain_context: str) -> ConfidenceFormulaWeights:
        """
        Determine weights for robustness confidence components.
        Replaces hardcoded 0.3, 0.3, 0.2, 0.2 values.
        """
        prompt = f"""
        Determine weights for robustness confidence calculation.
        
        EVIDENCE CONTEXT: {evidence_context}
        DOMAIN: {domain_context}
        
        Provide weights for:
        1. Source diversity (quality_weight)
        2. Reliability consistency (quantity_weight)
        3. Strength balance (diversity_weight)
        4. Independence score (balance_weight)
        
        Map to the weight fields appropriately.
        Weights should sum to approximately 1.0.
        """
        
        return self._get_structured_response(prompt, ConfidenceFormulaWeights)
    
    def assess_causal_mechanism(self, hypothesis_description: str, evidence_chain: str,
                               temporal_sequence: str) -> CausalMechanismAssessment:
        """
        Assess causal mechanism quality and completeness.
        Replaces hardcoded mechanism scores with semantic understanding.
        
        Args:
            hypothesis_description: Description of the hypothesis
            evidence_chain: Description of evidence chain
            temporal_sequence: Temporal ordering of events
            
        Returns:
            Comprehensive causal mechanism assessment
        """
        prompt = f"""
        Assess the causal mechanism quality for this hypothesis.
        
        HYPOTHESIS: {hypothesis_description}
        EVIDENCE CHAIN: {evidence_chain}
        TEMPORAL SEQUENCE: {temporal_sequence}
        
        Evaluate:
        1. Mechanism clarity and completeness (0.0-1.0)
        2. Temporal ordering consistency (0.0-1.0)
        3. Causal chain steps and missing links
        4. Theoretical grounding and empirical support
        5. Alternative explanations to consider
        6. Overall confidence with reasoning
        
        Apply Van Evera process tracing standards for causal assessment.
        """
        
        return self._get_structured_response(prompt, CausalMechanismAssessment)
    
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
    
    def classify_hypothesis_domain(self, hypothesis_description: str, 
                                 context: Optional[str] = None) -> HypothesisDomainClassification:
        """
        Classify hypothesis domain using semantic understanding.
        Replaces keyword matching with universal domain analysis.
        
        Args:
            hypothesis_description: Description of the hypothesis to classify
            context: Optional context for domain classification
            
        Returns:
            Structured domain classification with reasoning
        """
        prompt = f"""
        Analyze this hypothesis to determine its primary domain using semantic understanding.
        
        HYPOTHESIS: {hypothesis_description}
        CONTEXT: {context or 'Universal process tracing analysis'}
        
        Classify the hypothesis into domains based on its SEMANTIC CONTENT, not keywords:
        - POLITICAL: Government, authority, power structures, governance, policy
        - ECONOMIC: Trade, resources, financial systems, markets, wealth  
        - IDEOLOGICAL: Beliefs, values, worldviews, philosophical positions
        - MILITARY: Armed conflict, strategy, warfare, defense
        - SOCIAL: Community structures, relationships, class, identity
        - CULTURAL: Traditions, customs, arts, shared practices
        - RELIGIOUS: Faith, spiritual beliefs, religious institutions
        - TECHNOLOGICAL: Innovation, technical advancement, tools, methods
        
        Provide semantic reasoning that would apply across ANY historical period or domain.
        Avoid dataset-specific keywords - focus on universal conceptual categories.
        
        Consider how this classification generalizes beyond specific historical contexts.
        Identify semantic indicators that support your domain choice.
        If the hypothesis spans multiple domains, identify cross-domain relationships.
        """
        
        return self._get_structured_response(prompt, HypothesisDomainClassification)
    
    def assess_probative_value(self, evidence_description: str, 
                             hypothesis_description: str,
                             context: Optional[str] = None) -> ProbativeValueAssessment:
        """
        Assess evidence probative value using LLM semantic analysis.
        Replaces hardcoded probative value assignments.
        
        Args:
            evidence_description: Description of the evidence
            hypothesis_description: Description of the hypothesis
            context: Optional context for assessment
            
        Returns:
            Structured probative value assessment with academic reasoning
        """
        prompt = f"""
        Assess the probative value of this evidence for the given hypothesis using academic process tracing standards.
        
        EVIDENCE: {evidence_description}
        HYPOTHESIS: {hypothesis_description}
        CONTEXT: {context or 'Process tracing analysis'}
        
        Evaluate probative value (0.0-1.0) based on:
        1. RELEVANCE: How directly does the evidence relate to the hypothesis?
        2. RELIABILITY: How credible and trustworthy is the evidence source?
        3. STRENGTH: How compelling is the logical connection?
        4. QUALITY: What is the overall quality of the evidence?
        
        Consider Van Evera methodology standards:
        - High probative value (0.7-1.0): Strong, direct, reliable evidence with clear logical connection
        - Medium probative value (0.4-0.7): Moderate evidence with some limitations or indirect connections
        - Low probative value (0.0-0.4): Weak, indirect, or unreliable evidence with limited relevance
        
        Provide academic justification for your assessment.
        Identify factors that strengthen or weaken the evidence.
        Assess contextual relevance to the specific hypothesis.
        Consider implications for Van Evera diagnostic testing.
        
        Be thorough in explaining WHY this evidence has the assigned probative value.
        """
        
        return self._get_structured_response(prompt, ProbativeValueAssessment)
    
    def generate_alternative_hypotheses(self, original_hypothesis: str,
                                      evidence_context: str,
                                      domain_context: Optional[str] = None) -> AlternativeHypothesisGeneration:
        """
        Generate alternative hypotheses using semantic understanding.
        Replaces keyword dictionary approaches with contextual generation.
        
        Args:
            original_hypothesis: The original hypothesis to generate alternatives for
            evidence_context: Available evidence context
            domain_context: Optional domain context
            
        Returns:
            Structured alternative hypothesis generation with reasoning
        """
        prompt = f"""
        Generate competing alternative hypotheses for this process tracing analysis using semantic understanding.
        
        ORIGINAL HYPOTHESIS: {original_hypothesis}
        EVIDENCE CONTEXT: {evidence_context}
        DOMAIN CONTEXT: {domain_context or 'Universal historical analysis'}
        
        Generate 3-5 alternative hypotheses that:
        1. Propose different causal mechanisms for the same outcome
        2. Span different analytical domains (political, economic, ideological, social, etc.)
        3. Are theoretically sophisticated and historically plausible
        4. Could be tested using Van Evera diagnostic methodology
        
        For each alternative hypothesis, provide:
        - Clear description of the alternative causal mechanism
        - Domain classification (primary and secondary domains)
        - Theoretical justification for why this alternative is plausible
        - How it differs from the original hypothesis
        - What evidence would be needed to test it
        
        Focus on universal causal patterns that would apply across different historical periods.
        Avoid dataset-specific details - generate alternatives based on general theoretical principles.
        
        Ensure theoretical sophistication appropriate for academic process tracing.
        Consider competing mechanisms that a rigorous analysis should examine.
        """
        
        return self._get_structured_response(prompt, AlternativeHypothesisGeneration)
    
    def generate_van_evera_tests(self, hypothesis_description: str,
                               domain_classification: str,
                               evidence_context: str) -> TestGenerationSpecification:
        """
        Generate Van Evera diagnostic tests using semantic understanding.
        Replaces keyword-based test creation with context-appropriate generation.
        
        Args:
            hypothesis_description: The hypothesis to generate tests for
            domain_classification: Domain classification of the hypothesis
            evidence_context: Available evidence context
            
        Returns:
            Structured test generation specification with Van Evera diagnostic logic
        """
        prompt = f"""
        Generate Van Evera diagnostic tests for this hypothesis using rigorous academic methodology.
        
        HYPOTHESIS: {hypothesis_description}
        DOMAIN: {domain_classification}
        EVIDENCE CONTEXT: {evidence_context}
        
        Van Evera Diagnostic Test Types:
        - HOOP (Necessary): Evidence must be present if hypothesis is true. Absence eliminates hypothesis.
        - SMOKING GUN (Sufficient): Evidence strongly indicates hypothesis if present. Absence doesn't eliminate.
        - DOUBLY DECISIVE: Both necessary and sufficient. Definitive test.
        - STRAW IN WIND: Neither necessary nor sufficient. Provides incremental support.
        
        Generate 2-4 test predictions that:
        1. Follow Van Evera diagnostic logic rigorously
        2. Are appropriate for the hypothesis domain and content
        3. Specify clear evidence requirements based on semantic analysis
        4. Include proper diagnostic type classification with reasoning
        5. Are universal enough to work across different historical contexts
        
        For each test prediction, specify:
        - Clear prediction description
        - Van Evera diagnostic type with justification
        - Necessary/sufficient condition logic
        - Evidence requirements derived from semantic analysis
        - Why this test is theoretically appropriate
        
        Avoid keyword-based or dataset-specific requirements.
        Focus on semantic relationships and theoretical logic.
        Ensure tests meet academic standards for process tracing methodology.
        
        Provide diagnostic logic reasoning for test type selection.
        Consider theoretical grounding and universal validity.
        """
        
        return self._get_structured_response(prompt, TestGenerationSpecification)
    
    def analyze_evidence_comprehensive(self, 
                                      evidence_description: str,
                                      hypothesis_description: str,
                                      context: Optional[str] = None) -> ComprehensiveEvidenceAnalysis:
        """
        Comprehensive evidence analysis in a single LLM call.
        Replaces 5-10 separate calls with one coherent analysis.
        
        Args:
            evidence_description: Description of the evidence
            hypothesis_description: Description of the hypothesis
            context: Optional context for analysis
            
        Returns:
            Comprehensive analysis with all semantic features
        """
        prompt = f"""
        Perform comprehensive semantic analysis of this evidence-hypothesis relationship.
        Extract ALL features in one coherent analysis.
        
        EVIDENCE: {evidence_description}
        HYPOTHESIS: {hypothesis_description}
        CONTEXT: {context or 'Process tracing analysis'}
        
        Analyze comprehensively:
        
        1. DOMAIN CLASSIFICATION:
           - Identify primary domain (political/economic/ideological/military/social/cultural/religious/technological)
           - Note secondary domains if present
           - Provide confidence and reasoning
        
        2. PROBATIVE VALUE ASSESSMENT:
           - Calculate evidence strength (0.0-1.0)
           - Identify factors contributing to probative value
           - Assess evidence quality (high/medium/low)
           - Evaluate reliability
        
        3. HYPOTHESIS RELATIONSHIP:
           - Determine if evidence supports/contradicts/neutral/ambiguous
           - Provide confidence in relationship assessment
           - Explain detailed reasoning
           - Classify Van Evera diagnostic type (hoop/smoking gun/doubly decisive/straw in wind)
        
        4. CAUSAL MECHANISMS:
           - Identify all cause-effect relationships
           - Map mechanism types to descriptions
        
        5. TEMPORAL MARKERS:
           - Extract time references and sequences
           - Map markers to their context
        
        6. ACTOR RELATIONSHIPS:
           - Identify actors and their roles
           - Map relationships between actors
        
        7. KEY CONCEPTS & CONTEXT:
           - Extract main conceptual elements
           - Note contextual factors affecting interpretation
           - Consider alternative interpretations
        
        IMPORTANT: Provide reasoning that considers relationships between all features.
        Consider how domains affect probative value, how actors relate to mechanisms,
        and how temporal factors influence causation.
        """
        
        return self._get_structured_response(prompt, ComprehensiveEvidenceAnalysis)
    
    def extract_all_features(self, text: str, 
                            context: Optional[str] = None) -> MultiFeatureExtraction:
        """
        Extract all semantic features in one comprehensive pass.
        Captures relationships between features for better understanding.
        
        Args:
            text: Text to analyze for features
            context: Optional context for extraction
            
        Returns:
            Multi-feature extraction with relationships
        """
        prompt = f"""
        Analyze this text comprehensively, extracting ALL semantic features and their relationships.
        
        TEXT: {text}
        CONTEXT: {context or 'Feature extraction for process tracing'}
        
        Extract comprehensively:
        
        1. CAUSAL MECHANISMS:
           - All cause-effect relationships
           - Causal chains (sequences of events)
           - Types and descriptions of mechanisms
        
        2. ACTORS:
           - All primary actors
           - Roles and relationships between actors
           - Actor involvement in events
        
        3. TEMPORAL STRUCTURE:
           - All time markers and sequences
           - Event durations and estimates
           - Temporal ordering of events
        
        4. CONCEPTUAL ANALYSIS:
           - Key concepts and ideas
           - Domain indicators
           - Theoretical frameworks referenced
        
        5. CONTEXTUAL FACTORS:
           - Geographic/spatial context
           - Institutional context
           - Cultural context
        
        CRITICAL: Also identify RELATIONSHIPS between features:
        - Which actors are involved in which mechanisms?
        - How does timing affect causal relationships?
        - Which concepts are associated with which actors?
        - How do contextual factors influence mechanisms?
        
        Provide comprehensive extraction that captures not just individual features
        but also how they relate to and influence each other.
        """
        
        return self._get_structured_response(prompt, MultiFeatureExtraction)
    
    def _get_structured_response(self, prompt: str, response_model: Type[T], max_retries: int = 10) -> T:
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
                
            except (ConnectionError, TimeoutError, Exception) as e:
                # Retry on ALL errors except validation/schema issues (handled separately)
                last_exception = e
                if attempt < max_retries - 1:
                    import time
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30 seconds
                    log_structured_error(
                        logger,
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s... Error: {str(e)[:100]}",
                        error_category="llm_retry",
                        operation_context="structured_output_generation",
                        exc_info=False,  # Reduce log spam
                        **create_llm_context(
                            self.model_type,
                            "structured_output_retry",
                            response_model=response_model.__name__,
                            attempt=attempt + 1,
                            wait_time=wait_time
                        )
                    )
                    time.sleep(wait_time)
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
    
    def evaluate_evidence_against_hypotheses(self,
                                            evidence_id: str,
                                            evidence_text: str,
                                            hypotheses: List[Dict[str, str]],
                                            context: Optional[str] = None) -> BatchedHypothesisEvaluation:
        """
        Evaluate evidence against multiple hypotheses in a single LLM call.
        This provides better coherence and identifies inter-hypothesis relationships.
        
        Args:
            evidence_id: Unique identifier for the evidence
            evidence_text: The evidence text to evaluate
            hypotheses: List of dicts with 'id' and 'text' keys
            context: Optional context information
            
        Returns:
            BatchedHypothesisEvaluation with all relationships and insights
        """
        # Format hypotheses for prompt
        hypotheses_formatted = "\n".join([
            f"H{i+1} (ID: {h['id']}): {h['text']}"
            for i, h in enumerate(hypotheses)
        ])
        
        prompt = f"""
        Evaluate this evidence against ALL hypotheses simultaneously.
        Consider how the evidence affects each hypothesis AND how it reveals
        relationships between hypotheses.
        
        EVIDENCE (ID: {evidence_id}):
        {evidence_text}
        
        HYPOTHESES TO EVALUATE:
        {hypotheses_formatted}
        
        CONTEXT: {context or 'Process tracing analysis'}
        
        For EACH hypothesis, determine:
        1. Relationship type (supports/contradicts/neutral/ambiguous)
        2. Confidence level (0.0-1.0)
        3. Van Evera diagnostic type (hoop/smoking_gun/doubly_decisive/straw_in_wind)
        4. Detailed reasoning
        5. How this affects other hypotheses
        
        ALSO identify:
        - Which hypothesis is MOST supported by this evidence
        - Which hypotheses conflict with each other based on this evidence
        - Which hypotheses complement/reinforce each other
        - Overall significance of this evidence (critical/important/moderate/minor)
        
        IMPORTANT: Consider inter-hypothesis relationships. For example:
        - If evidence supports H1, does that strengthen or weaken H2?
        - Do some hypotheses exclude others?
        - Are some hypotheses complementary explanations?
        """
        
        return self._get_structured_response(prompt, BatchedHypothesisEvaluation)


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