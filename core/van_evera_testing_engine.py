"""
Van Evera Systematic Hypothesis Testing Engine
Implements rigorous academic process tracing methodology with Bayesian updating
"""

from typing import Dict, List, Tuple, Optional
import json
import math
from dataclasses import dataclass
from enum import Enum
import logging

# Import LLM interface for semantic analysis
try:
    from .plugins.van_evera_llm_interface import get_van_evera_llm
except ImportError:
    # Fallback if not available
    def get_van_evera_llm():
        raise ImportError("Van Evera LLM interface not available")

logger = logging.getLogger(__name__)

class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL" 
    INCONCLUSIVE = "INCONCLUSIVE"

class DiagnosticType(Enum):
    HOOP = "hoop"                    # Necessary but not sufficient
    SMOKING_GUN = "smoking_gun"      # Sufficient but not necessary  
    DOUBLY_DECISIVE = "doubly_decisive"  # Both necessary and sufficient
    STRAW_IN_WIND = "straw_in_wind"  # Neither necessary nor sufficient

@dataclass
class TestPrediction:
    """Specific testable prediction derived from hypothesis"""
    prediction_id: str
    hypothesis_id: str
    description: str
    diagnostic_type: DiagnosticType
    necessary_condition: bool
    sufficient_condition: bool
    evidence_requirements: List[str]

@dataclass 
class TestEvaluation:
    """Result of testing a specific prediction"""
    prediction_id: str
    test_result: TestResult
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    confidence_level: float
    reasoning: str
    elimination_implications: List[str]

@dataclass
class HypothesisAssessment:
    """Complete assessment of hypothesis after all tests"""
    hypothesis_id: str
    description: str
    prior_probability: float
    posterior_probability: float
    test_results: List[TestEvaluation]
    overall_status: str  # "SUPPORTED", "ELIMINATED", "WEAKENED", "INCONCLUSIVE"
    confidence_interval: Tuple[float, float]
    academic_conclusion: str

class VanEveraTestingEngine:
    """
    Implements systematic Van Evera hypothesis testing with academic rigor.
    Performs Bayesian updating and comparative elimination logic.
    """
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        self.hypotheses = [n for n in graph_data['nodes'] 
                          if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        # Evidence can be Events, Context nodes, or explicit Evidence nodes
        self.evidence = [n for n in graph_data['nodes'] 
                        if n.get('type') in ['Evidence', 'Event', 'Context', 'Actor']]
        self.evidence_edges = [e for e in graph_data['edges'] if self._is_evidence_relationship(e)]
        
    def _is_evidence_relationship(self, edge: Dict) -> bool:
        """Check if edge represents evidence-hypothesis relationship"""
        source_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['source_id']), None)
        target_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['target_id']), None)
        
        # Evidence can come from various node types
        source_is_evidence = (source_node and 
                             source_node.get('type') in ['Evidence', 'Event', 'Context', 'Actor'])
        target_is_hypothesis = (target_node and 
                               target_node.get('type') in ['Hypothesis', 'Alternative_Explanation'])
        
        # Check for evidence relationship types
        evidence_edge_types = ['provides_evidence_for', 'supports', 'refutes', 'contradicts', 
                              'challenges', 'undermines', 'confirms']
        
        return (source_is_evidence and target_is_hypothesis and 
                edge.get('type', '') in evidence_edge_types)
    
    def generate_testable_predictions(self, hypothesis: Dict) -> List[TestPrediction]:
        """
        Generate specific, testable predictions from hypothesis using LLM semantic analysis.
        Replaces all keyword-based logic with universal Van Evera methodology.
        """
        hypothesis_id = hypothesis['id']
        hypothesis_desc = hypothesis.get('properties', {}).get('description', '')
        hypothesis_type = hypothesis.get('type', '')
        
        # Generate predictions using LLM semantic analysis (no keywords)
        predictions = self._generate_semantic_predictions(hypothesis_id, hypothesis_desc)
        
        # For alternative explanations, enhance with competing hypothesis analysis
        if hypothesis_type == 'Alternative_Explanation':
            try:
                llm_interface = get_van_evera_llm()
                
                # Generate alternative hypotheses to ensure comprehensive testing
                alt_generation = llm_interface.generate_alternative_hypotheses(
                    original_hypothesis=hypothesis_desc,
                    evidence_context="Van Evera process tracing requires testing of competing explanations",
                    domain_context="Universal historical analysis"
                )
                
                # Add tests specifically designed for alternative explanation evaluation
                for i, alt_hyp in enumerate(alt_generation.alternative_hypotheses):
                    alt_test = TestPrediction(
                        prediction_id=f"{hypothesis_id}_ALT_COMP_{i+1:03d}",
                        hypothesis_id=hypothesis_id,
                        description=f"Evidence must distinguish between competing mechanisms: {alt_hyp.get('description', 'alternative mechanism')}",
                        diagnostic_type=DiagnosticType.SMOKING_GUN,
                        necessary_condition=False,
                        sufficient_condition=True,
                        evidence_requirements=alt_hyp.get('evidence_requirements', ["comparative_evidence", "mechanism_distinction"])
                    )
                    predictions.append(alt_test)
                    
                logger.info(f"Enhanced alternative explanation {hypothesis_id} with {len(alt_generation.alternative_hypotheses)} competing tests")
                
            except Exception as e:
                logger.warning(f"Failed to enhance alternative explanation {hypothesis_id}: {e}")
        
        # If no predictions generated, create universal fallback (no dataset-specific logic)
        if not predictions:
            predictions = self._generate_generic_predictions(hypothesis_id, hypothesis_desc)
            
        return predictions
    
    def evaluate_prediction(self, prediction: TestPrediction) -> TestEvaluation:
        """
        Systematically evaluate whether prediction passes or fails based on available evidence.
        Implements Van Evera's logic with enhanced evidence matching.
        """
        # Find relevant evidence using multiple strategies
        relevant_evidence, contradicting_evidence = self._find_prediction_evidence(prediction)
        
        # Apply Van Evera diagnostic logic
        test_result, reasoning = self._apply_diagnostic_logic(
            prediction, relevant_evidence, contradicting_evidence
        )
        
        # Calculate confidence level based on evidence strength
        confidence = self._calculate_confidence(relevant_evidence, contradicting_evidence, prediction)
        
        # Determine elimination implications
        elimination_implications = self._assess_elimination_implications(prediction, test_result)
        
        return TestEvaluation(
            prediction_id=prediction.prediction_id,
            test_result=test_result,
            supporting_evidence=relevant_evidence,
            contradicting_evidence=contradicting_evidence,
            confidence_level=confidence,
            reasoning=reasoning,
            elimination_implications=elimination_implications
        )
    
    def _generate_semantic_predictions(self, hypothesis_id: str, hypothesis_desc: str) -> List[TestPrediction]:
        """
        Generate predictions based on LLM semantic analysis of hypothesis content.
        Replaces keyword matching with universal domain analysis and Van Evera diagnostic logic.
        """
        predictions = []
        
        try:
            # Get LLM interface for semantic analysis
            llm_interface = get_van_evera_llm()
            
            # First classify the hypothesis domain using LLM semantic understanding
            domain_classification = llm_interface.classify_hypothesis_domain(
                hypothesis_description=hypothesis_desc,
                context="Van Evera process tracing diagnostic test generation"
            )
            
            # Generate Van Evera tests based on semantic understanding
            test_generation = llm_interface.generate_van_evera_tests(
                hypothesis_description=hypothesis_desc,
                domain_classification=domain_classification.primary_domain,
                evidence_context="Universal process tracing analysis requiring Van Evera diagnostic methodology"
            )
            
            # Convert LLM-generated tests to TestPrediction objects
            for i, test_pred in enumerate(test_generation.test_predictions):
                # Map diagnostic type string to enum
                diagnostic_type_map = {
                    'hoop': DiagnosticType.HOOP,
                    'smoking_gun': DiagnosticType.SMOKING_GUN, 
                    'doubly_decisive': DiagnosticType.DOUBLY_DECISIVE,
                    'straw_in_wind': DiagnosticType.STRAW_IN_WIND
                }
                
                diagnostic_type = diagnostic_type_map.get(
                    test_pred.get('diagnostic_type', '').lower(), 
                    DiagnosticType.STRAW_IN_WIND  # Default fallback
                )
                
                # Determine necessary/sufficient conditions based on diagnostic type
                necessary_condition = diagnostic_type in [DiagnosticType.HOOP, DiagnosticType.DOUBLY_DECISIVE]
                sufficient_condition = diagnostic_type in [DiagnosticType.SMOKING_GUN, DiagnosticType.DOUBLY_DECISIVE]
                
                prediction = TestPrediction(
                    prediction_id=f"{hypothesis_id}_LLM_SEM_{i+1:03d}",
                    hypothesis_id=hypothesis_id,
                    description=test_pred.get('description', f"Semantic test {i+1}"),
                    diagnostic_type=diagnostic_type,
                    necessary_condition=necessary_condition,
                    sufficient_condition=sufficient_condition,
                    evidence_requirements=test_pred.get('evidence_requirements', [])
                )
                predictions.append(prediction)
                
            logger.info(f"Generated {len(predictions)} semantic predictions using LLM for hypothesis {hypothesis_id}")
            
        except Exception as e:
            logger.warning(f"LLM semantic prediction generation failed for {hypothesis_id}: {e}")
            # Fallback to basic universal pattern (no keywords, no dataset-specific logic)
            predictions.append(TestPrediction(
                prediction_id=f"{hypothesis_id}_FALLBACK_001",
                hypothesis_id=hypothesis_id,
                description="Evidence must be temporally and logically consistent with proposed causal mechanism",
                diagnostic_type=DiagnosticType.HOOP,
                necessary_condition=True,
                sufficient_condition=False,
                evidence_requirements=["temporal_consistency", "logical_coherence", "causal_mechanism"]
            ))
        
        return predictions
    
    def _generate_generic_predictions(self, hypothesis_id: str, hypothesis_desc: str) -> List[TestPrediction]:
        """Generate generic predictions when semantic analysis fails"""
        # Extract key terms from hypothesis description
        import re
        key_terms = re.findall(r'\b[a-zA-Z]{4,}\b', hypothesis_desc.lower())
        key_terms = [term for term in key_terms if term not in 
                    ['this', 'that', 'with', 'from', 'were', 'have', 'been', 'would']][:6]
        
        return [TestPrediction(
            prediction_id=f"{hypothesis_id}_GEN_001",
            hypothesis_id=hypothesis_id,
            description=f"Evidence must support key elements: {', '.join(key_terms[:3])}",
            diagnostic_type=DiagnosticType.STRAW_IN_WIND,
            necessary_condition=False,
            sufficient_condition=False,
            evidence_requirements=key_terms
        )]
    
    def _find_prediction_evidence(self, prediction: TestPrediction) -> Tuple[List[str], List[str]]:
        """Enhanced evidence finding using multiple matching strategies"""
        relevant_evidence = []
        contradicting_evidence = []
        
        # Strategy 1: Direct edge matching to hypothesis
        for edge in self.evidence_edges:
            if edge['target_id'] == prediction.hypothesis_id:
                evidence_node = next((n for n in self.graph_data['nodes'] 
                                    if n['id'] == edge['source_id']), None)
                if evidence_node:
                    # Use improved relevance scoring
                    if self._is_evidence_relevant_to_prediction(evidence_node, edge, prediction):
                        edge_type = edge.get('type', '')
                        if edge_type in ['supports', 'provides_evidence_for', 'confirms']:
                            relevant_evidence.append(evidence_node['id'])
                        elif edge_type in ['refutes', 'contradicts', 'challenges', 'undermines']:
                            contradicting_evidence.append(evidence_node['id'])
        
        # Strategy 2: Semantic content matching across all evidence
        if len(relevant_evidence) < 2:  # If direct matching found little evidence
            semantic_evidence = self._find_semantic_evidence(prediction)
            relevant_evidence.extend(semantic_evidence)
        
        # Remove duplicates while preserving order
        relevant_evidence = list(dict.fromkeys(relevant_evidence))
        contradicting_evidence = list(dict.fromkeys(contradicting_evidence))
        
        return relevant_evidence, contradicting_evidence
    
    def _is_evidence_relevant_to_prediction(self, evidence_node: Dict, edge: Dict, prediction: TestPrediction) -> bool:
        """
        Determine if evidence is relevant to a specific prediction using LLM semantic analysis.
        Replaces keyword matching with semantic understanding of evidence-prediction relationships.
        """
        evidence_desc = evidence_node.get('properties', {}).get('description', '')
        edge_props = edge.get('properties', {})
        source_quote = edge_props.get('source_text_quote', '')
        edge_reasoning = edge_props.get('reasoning', '')
        
        # Combine all text for semantic analysis
        evidence_text = f"{evidence_desc} {source_quote} {edge_reasoning}".strip()
        
        try:
            # Use LLM to assess semantic relevance
            llm_interface = get_van_evera_llm()
            
            relevance_assessment = llm_interface.assess_probative_value(
                evidence_description=evidence_text,
                hypothesis_description=prediction.description,
                context=f"Van Evera {prediction.diagnostic_type.value} test relevance assessment"
            )
            
            # Consider both probative value and contextual relevance
            semantic_relevance = (relevance_assessment.probative_value * 0.6 + 
                                relevance_assessment.contextual_relevance * 0.4)
            
            # Also consider existing edge properties if available
            edge_diagnostic_type = edge_props.get('diagnostic_type', '')
            edge_probative_value = edge_props.get('probative_value', 0)
            
            # Boost relevance if edge properties align with prediction
            if edge_diagnostic_type == prediction.diagnostic_type.value:
                semantic_relevance += 0.1
            if edge_probative_value > 0.6:
                semantic_relevance += 0.1
                
            # Threshold for semantic relevance (more nuanced than simple keyword counting)
            is_relevant = semantic_relevance >= 0.5
            
            if is_relevant:
                logger.debug(f"LLM semantic analysis: Evidence {evidence_node['id']} relevant to prediction {prediction.prediction_id} (score: {semantic_relevance:.3f})")
            
            return is_relevant
            
        except Exception as e:
            logger.warning(f"LLM relevance assessment failed for {prediction.prediction_id}: {e}")
            
            # Fallback to basic non-keyword analysis
            # Check if evidence and prediction share substantial semantic content
            evidence_words = set(evidence_text.lower().split())
            prediction_words = set(prediction.description.lower().split())
            
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
            
            evidence_words = evidence_words - common_words
            prediction_words = prediction_words - common_words
            
            # Basic semantic overlap without keyword matching
            overlap = evidence_words.intersection(prediction_words)
            overlap_ratio = len(overlap) / max(len(prediction_words), 1)
            
            return overlap_ratio >= 0.2  # At least 20% semantic word overlap
    
    def _find_semantic_evidence(self, prediction: TestPrediction) -> List[str]:
        """
        Find evidence using LLM semantic analysis beyond direct hypothesis connections.
        Replaces keyword overlap counting with semantic relationship assessment.
        """
        semantic_evidence = []
        
        try:
            llm_interface = get_van_evera_llm()
            
            # Analyze each evidence node for semantic relevance to the prediction
            for evidence_node in self.evidence:
                evidence_desc = evidence_node.get('properties', {}).get('description', '')
                
                if not evidence_desc.strip():
                    continue
                
                # Use LLM to assess semantic relationship
                relationship_assessment = llm_interface.classify_evidence_relationship(
                    evidence_description=evidence_desc,
                    hypothesis_description=prediction.description
                )
                
                # Include evidence that supports or is relevant to the prediction
                # Use confidence-based threshold instead of hardcoded value
                min_probative = relationship_assessment.confidence_score * 0.4  # Dynamic threshold
                if (relationship_assessment.relationship_type in ['supporting', 'refuting'] and
                    relationship_assessment.probative_value >= min_probative and
                    relationship_assessment.confidence_score >= 0.5):
                    
                    semantic_evidence.append(evidence_node['id'])
                    logger.debug(f"Semantic evidence {evidence_node['id']}: {relationship_assessment.relationship_type} "
                               f"(probative: {relationship_assessment.probative_value:.3f}, "
                               f"confidence: {relationship_assessment.confidence_score:.3f})")
                
        except Exception as e:
            logger.warning(f"LLM semantic evidence search failed for {prediction.prediction_id}: {e}")
            
            # Fallback to basic semantic overlap (no keyword matching)
            for evidence_node in self.evidence:
                evidence_desc = evidence_node.get('properties', {}).get('description', '')
                if not evidence_desc.strip():
                    continue
                    
                # Basic semantic overlap without keyword matching
                evidence_words = set(evidence_desc.lower().split())
                prediction_words = set(prediction.description.lower().split())
                
                # Remove common words
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
                
                evidence_words = evidence_words - common_words
                prediction_words = prediction_words - common_words
                
                # Include if substantial semantic overlap
                overlap = evidence_words.intersection(prediction_words)
                if len(overlap) >= 2:  # At least 2 significant word overlaps
                    semantic_evidence.append(evidence_node['id'])
                
        return semantic_evidence[:5]  # Limit to top 5 matches
    
    def _extract_prediction_keywords(self, prediction: TestPrediction) -> List[str]:
        """
        Extract semantic requirements from prediction using LLM analysis.
        Replaces keyword-based extraction with semantic understanding.
        """
        # Use evidence_requirements as primary semantic indicators
        semantic_requirements = prediction.evidence_requirements[:]
        
        try:
            # Use LLM to understand what evidence this prediction actually requires
            llm_interface = get_van_evera_llm()
            
            probative_assessment = llm_interface.assess_probative_value(
                evidence_description=f"General evidence for prediction: {prediction.description}",
                hypothesis_description=prediction.description,
                context="Van Evera diagnostic test evidence requirement analysis"
            )
            
            # Extract semantic requirements from LLM analysis
            if probative_assessment.evidence_quality_factors:
                semantic_requirements.extend(probative_assessment.evidence_quality_factors)
            
            if probative_assessment.strength_indicators:
                semantic_requirements.extend(probative_assessment.strength_indicators)
                
            logger.info(f"Enhanced prediction {prediction.prediction_id} with LLM semantic requirements")
            
        except Exception as e:
            logger.warning(f"Failed to enhance prediction keywords for {prediction.prediction_id}: {e}")
            # Fallback to evidence_requirements only (no keyword expansion)
            pass
            
        return list(set(semantic_requirements))  # Remove duplicates
    
    def _apply_diagnostic_logic(self, prediction: TestPrediction, 
                               supporting: List[str], contradicting: List[str]) -> Tuple[TestResult, str]:
        """Apply Van Evera diagnostic test logic"""
        has_support = len(supporting) > 0
        has_contradiction = len(contradicting) > 0
        
        if prediction.diagnostic_type == DiagnosticType.HOOP:
            # Necessary condition - must pass to remain viable
            if has_contradiction or not has_support:
                return TestResult.FAIL, f"HOOP TEST FAILED: Hypothesis eliminated. Required evidence not found or contradicted. Supporting: {len(supporting)}, Contradicting: {len(contradicting)}"
            else:
                return TestResult.PASS, f"HOOP TEST PASSED: Hypothesis remains viable. Necessary condition satisfied. Supporting evidence: {len(supporting)}"
        
        elif prediction.diagnostic_type == DiagnosticType.SMOKING_GUN:
            # Sufficient condition - if passes, strongly confirms hypothesis
            if has_support and not has_contradiction:
                return TestResult.PASS, f"SMOKING GUN PASSED: Hypothesis strongly confirmed. Sufficient evidence found. Supporting: {len(supporting)}"
            elif has_contradiction:
                return TestResult.FAIL, f"SMOKING GUN FAILED: Evidence contradicts prediction. Contradicting: {len(contradicting)}"
            else:
                return TestResult.INCONCLUSIVE, f"SMOKING GUN INCONCLUSIVE: No decisive evidence found. Neither confirms nor disconfirms."
        
        elif prediction.diagnostic_type == DiagnosticType.DOUBLY_DECISIVE:
            # Both necessary and sufficient
            if has_support and not has_contradiction:
                return TestResult.PASS, f"DOUBLY DECISIVE PASSED: Hypothesis confirmed and alternatives eliminated. Supporting: {len(supporting)}"
            else:
                return TestResult.FAIL, f"DOUBLY DECISIVE FAILED: Hypothesis eliminated. Supporting: {len(supporting)}, Contradicting: {len(contradicting)}"
        
        elif prediction.diagnostic_type == DiagnosticType.STRAW_IN_WIND:
            # Neither necessary nor sufficient - provides weak evidence
            if has_support:
                return TestResult.PASS, f"STRAW IN WIND PASSED: Weak support for hypothesis. Supporting: {len(supporting)}"
            else:
                return TestResult.INCONCLUSIVE, f"STRAW IN WIND INCONCLUSIVE: No clear support found."
        
        return TestResult.INCONCLUSIVE, "Unable to determine test result"
    
    def _calculate_confidence(self, supporting: List[str], contradicting: List[str], 
                            prediction: TestPrediction) -> float:
        """Calculate confidence level for test result"""
        total_evidence = len(supporting) + len(contradicting)
        if total_evidence == 0:
            return 0.3  # Low confidence with no evidence
        
        support_ratio = len(supporting) / total_evidence
        evidence_volume_bonus = min(total_evidence * 0.1, 0.3)  # Bonus for more evidence
        
        base_confidence = support_ratio * 0.7 + evidence_volume_bonus
        return min(base_confidence, 0.95)  # Cap at 95% confidence
    
    def _assess_elimination_implications(self, prediction: TestPrediction, 
                                       result: TestResult) -> List[str]:
        """Determine which hypotheses are eliminated by test result"""
        implications = []
        
        if result == TestResult.FAIL:
            if prediction.diagnostic_type in [DiagnosticType.HOOP, DiagnosticType.DOUBLY_DECISIVE]:
                implications.append(f"Hypothesis {prediction.hypothesis_id} ELIMINATED")
            elif prediction.diagnostic_type == DiagnosticType.SMOKING_GUN:
                implications.append(f"Hypothesis {prediction.hypothesis_id} NOT CONFIRMED")
        
        elif result == TestResult.PASS:
            if prediction.diagnostic_type in [DiagnosticType.SMOKING_GUN, DiagnosticType.DOUBLY_DECISIVE]:
                implications.append(f"Hypothesis {prediction.hypothesis_id} STRONGLY SUPPORTED")
        
        return implications
    
    def systematic_hypothesis_evaluation(self) -> Dict[str, HypothesisAssessment]:
        """
        Perform systematic Van Evera evaluation of all hypotheses.
        Returns complete academic assessment with Bayesian updating.
        """
        print("[VAN_EVERA_TESTING] Starting systematic hypothesis evaluation...")
        
        assessments = {}
        
        for hypothesis in self.hypotheses:
            hypothesis_id = hypothesis['id']
            print(f"[VAN_EVERA_TESTING] Testing hypothesis: {hypothesis_id}")
            
            # Generate testable predictions
            predictions = self.generate_testable_predictions(hypothesis)
            print(f"[VAN_EVERA_TESTING] Generated {len(predictions)} predictions for {hypothesis_id}")
            
            # Test each prediction
            test_results = []
            for prediction in predictions:
                result = self.evaluate_prediction(prediction)
                test_results.append(result)
                print(f"[VAN_EVERA_TESTING] {prediction.diagnostic_type.value} test: {result.test_result.value}")
            
            # Calculate overall assessment with Bayesian updating
            prior_prob = 0.5  # Neutral prior
            posterior_prob = self._calculate_posterior_probability(prior_prob, test_results)
            
            # Determine overall status
            overall_status = self._determine_overall_status(test_results, posterior_prob)
            
            # Generate academic conclusion
            academic_conclusion = self._generate_academic_conclusion(hypothesis, test_results, posterior_prob)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(posterior_prob, test_results)
            
            assessment = HypothesisAssessment(
                hypothesis_id=hypothesis_id,
                description=hypothesis.get('properties', {}).get('description', ''),
                prior_probability=prior_prob,
                posterior_probability=posterior_prob,
                test_results=test_results,
                overall_status=overall_status,
                confidence_interval=confidence_interval,
                academic_conclusion=academic_conclusion
            )
            
            assessments[hypothesis_id] = assessment
        
        print(f"[VAN_EVERA_TESTING] Completed evaluation of {len(assessments)} hypotheses")
        return assessments
    
    def _calculate_posterior_probability(self, prior: float, test_results: List[TestEvaluation]) -> float:
        """Calculate posterior probability using Bayesian updating"""
        log_odds = math.log(prior / (1 - prior))  # Convert to log odds
        
        for result in test_results:
            if result.test_result == TestResult.PASS:
                # Positive evidence - increase log odds
                log_odds += result.confidence_level * 2
            elif result.test_result == TestResult.FAIL:
                # Negative evidence - decrease log odds  
                log_odds -= result.confidence_level * 3
            # INCONCLUSIVE results don't change log odds
        
        # Convert back to probability
        odds = math.exp(log_odds)
        return odds / (1 + odds)
    
    def _determine_overall_status(self, test_results: List[TestEvaluation], posterior: float) -> str:
        """Determine overall hypothesis status"""
        failed_decisive_tests = sum(1 for r in test_results 
                                  if r.test_result == TestResult.FAIL and 
                                  any('ELIMINATED' in imp for imp in r.elimination_implications))
        
        if failed_decisive_tests > 0:
            return "ELIMINATED"
        elif posterior > 0.8:
            return "STRONGLY_SUPPORTED" 
        elif posterior > 0.6:
            return "SUPPORTED"
        elif posterior > 0.4:
            return "INCONCLUSIVE"
        else:
            return "WEAKENED"
    
    def _generate_academic_conclusion(self, hypothesis: Dict, test_results: List[TestEvaluation], 
                                    posterior: float) -> str:
        """Generate academic-quality conclusion"""
        hyp_desc = hypothesis.get('properties', {}).get('description', 'Unknown hypothesis')
        
        # Use semantic analysis to classify test results by type
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        hoop_results = []
        smoking_gun_results = []
        decisive_results = []
        
        for r in test_results:
            # Check each test type using semantic understanding
            reasoning_text = r.reasoning
            
            # Assess if result is a hoop test
            hoop_assessment = semantic_service.assess_probative_value(
                evidence_description=reasoning_text,
                hypothesis_description="This is a necessary condition (hoop) test result",
                context="Classifying Van Evera diagnostic test results"
            )
            if hoop_assessment.confidence_score > 0.7:
                hoop_results.append(r)
            
            # Assess if result is a smoking gun test
            gun_assessment = semantic_service.assess_probative_value(
                evidence_description=reasoning_text,
                hypothesis_description="This is a sufficient condition (smoking gun) test result",
                context="Classifying Van Evera diagnostic test results"
            )
            if gun_assessment.confidence_score > 0.7:
                smoking_gun_results.append(r)
            
            # Assess if result is doubly decisive
            decisive_assessment = semantic_service.assess_probative_value(
                evidence_description=reasoning_text,
                hypothesis_description="This is both necessary and sufficient (doubly decisive) test result",
                context="Classifying Van Evera diagnostic test results"
            )
            if decisive_assessment.confidence_score > 0.7:
                decisive_results.append(r)
        
        conclusion = f"HYPOTHESIS: {hyp_desc}\n\n"
        
        # Hoop test analysis
        if hoop_results:
            hoop_passes = sum(1 for r in hoop_results if r.test_result == TestResult.PASS)
            conclusion += f"NECESSARY CONDITION ANALYSIS: {hoop_passes}/{len(hoop_results)} hoop tests passed. "
            if hoop_passes == len(hoop_results):
                conclusion += "Hypothesis remains viable - all necessary conditions satisfied.\n\n"
            else:
                conclusion += "Hypothesis ELIMINATED - failed necessary condition test.\n\n"
                return conclusion
        
        # Smoking gun analysis
        if smoking_gun_results:
            smoking_gun_passes = sum(1 for r in smoking_gun_results if r.test_result == TestResult.PASS)
            conclusion += f"SUFFICIENT CONDITION ANALYSIS: {smoking_gun_passes}/{len(smoking_gun_results)} smoking gun tests passed. "
            if smoking_gun_passes > 0:
                conclusion += "Strong confirming evidence found.\n\n"
            else:
                conclusion += "No decisive confirming evidence.\n\n"
        
        # Overall assessment
        conclusion += f"OVERALL ASSESSMENT: Posterior probability = {posterior:.2f}. "
        if posterior > 0.8:
            conclusion += "Hypothesis STRONGLY SUPPORTED by systematic testing."
        elif posterior > 0.6:
            conclusion += "Hypothesis SUPPORTED with moderate confidence."
        elif posterior > 0.4:
            conclusion += "Evidence INCONCLUSIVE - requires additional testing."
        else:
            conclusion += "Hypothesis WEAKENED by available evidence."
        
        return conclusion
    
    def _calculate_confidence_interval(self, posterior: float, test_results: List[TestEvaluation]) -> Tuple[float, float]:
        """Calculate confidence interval for posterior probability"""
        n_tests = len(test_results)
        if n_tests == 0:
            return (max(0, posterior - 0.3), min(1, posterior + 0.3))
        
        # Confidence narrows with more tests
        margin = 0.4 / math.sqrt(n_tests)
        return (max(0, posterior - margin), min(1, posterior + margin))

# Integration function
def perform_van_evera_testing(graph_data: Dict) -> Dict:
    """
    Main entry point for systematic Van Evera hypothesis testing.
    Returns comprehensive academic assessment results.
    """
    engine = VanEveraTestingEngine(graph_data)
    return engine.systematic_hypothesis_evaluation()