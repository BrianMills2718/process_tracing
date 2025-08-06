"""
Van Evera Systematic Hypothesis Testing Engine
Implements rigorous academic process tracing methodology with Bayesian updating
"""

from typing import Dict, List, Tuple, Optional
import json
import math
from dataclasses import dataclass
from enum import Enum

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
        self.evidence = [n for n in graph_data['nodes'] if n.get('type') == 'Evidence']
        self.evidence_edges = [e for e in graph_data['edges'] if self._is_evidence_relationship(e)]
        
    def _is_evidence_relationship(self, edge: Dict) -> bool:
        """Check if edge represents evidence-hypothesis relationship"""
        source_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['source_id']), None)
        target_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['target_id']), None)
        return (source_node and source_node.get('type') == 'Evidence' and 
                target_node and target_node.get('type') in ['Hypothesis', 'Alternative_Explanation'])
    
    def generate_testable_predictions(self, hypothesis: Dict) -> List[TestPrediction]:
        """
        Generate specific, testable predictions from hypothesis.
        Academic Van Evera requires explicit predictions that can be systematically tested.
        """
        hypothesis_id = hypothesis['id']
        hypothesis_desc = hypothesis.get('properties', {}).get('description', '')
        
        # Extract domain-specific predictions based on hypothesis content
        predictions = []
        
        if 'taxation without representation' in hypothesis_desc.lower():
            predictions.extend([
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_001",
                    hypothesis_id=hypothesis_id,
                    description="Colonial resistance rhetoric must consistently invoke English constitutional rights",
                    diagnostic_type=DiagnosticType.HOOP,
                    necessary_condition=True,
                    sufficient_condition=False,
                    evidence_requirements=["legal_arguments", "constitutional_rhetoric", "rights_language"]
                ),
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_002", 
                    hypothesis_id=hypothesis_id,
                    description="Opposition intensity should correlate with tax burden increases",
                    diagnostic_type=DiagnosticType.SMOKING_GUN,
                    necessary_condition=False,
                    sufficient_condition=True,
                    evidence_requirements=["tax_legislation", "resistance_timing", "burden_measurement"]
                )
            ])
        
        elif 'ideological' in hypothesis_desc.lower() and 'political' in hypothesis_desc.lower():
            predictions.extend([
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_001",
                    hypothesis_id=hypothesis_id,
                    description="Political documents must contain systematic philosophical arguments",
                    diagnostic_type=DiagnosticType.HOOP,
                    necessary_condition=True, 
                    sufficient_condition=False,
                    evidence_requirements=["philosophical_language", "systematic_arguments", "intellectual_references"]
                ),
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_002",
                    hypothesis_id=hypothesis_id,
                    description="Revolutionary leadership should demonstrate intellectual sophistication",
                    diagnostic_type=DiagnosticType.STRAW_IN_WIND,
                    necessary_condition=False,
                    sufficient_condition=False,
                    evidence_requirements=["leader_education", "intellectual_networks", "philosophical_writings"]
                )
            ])
        
        elif 'self-governance' in hypothesis_desc.lower():
            predictions.extend([
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_001",
                    hypothesis_id=hypothesis_id,
                    description="Resistance must emerge from established local governance institutions",
                    diagnostic_type=DiagnosticType.DOUBLY_DECISIVE,
                    necessary_condition=True,
                    sufficient_condition=True,
                    evidence_requirements=["local_institutions", "institutional_continuity", "governance_experience"]
                ),
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_002",
                    hypothesis_id=hypothesis_id,
                    description="Post-1763 British policies must directly threaten local autonomy",
                    diagnostic_type=DiagnosticType.HOOP,
                    necessary_condition=True,
                    sufficient_condition=False,
                    evidence_requirements=["policy_changes", "autonomy_threats", "local_responses"]
                )
            ])
        
        # For alternative explanations, generate competing predictions
        elif hypothesis.get('type') == 'Alternative_Explanation':
            alt_desc = hypothesis_desc.lower()
            if 'merchant' in alt_desc or 'economic' in alt_desc:
                predictions.extend([
                    TestPrediction(
                        prediction_id=f"{hypothesis_id}_PRED_001",
                        hypothesis_id=hypothesis_id,
                        description="Resistance leaders must be predominantly merchants or have merchant connections",
                        diagnostic_type=DiagnosticType.SMOKING_GUN,
                        necessary_condition=False,
                        sufficient_condition=True,
                        evidence_requirements=["leader_occupations", "merchant_networks", "economic_interests"]
                    ),
                    TestPrediction(
                        prediction_id=f"{hypothesis_id}_PRED_002",
                        hypothesis_id=hypothesis_id,
                        description="Opposition timing must correlate with trade disruption severity",
                        diagnostic_type=DiagnosticType.HOOP,
                        necessary_condition=True,
                        sufficient_condition=False,
                        evidence_requirements=["trade_data", "disruption_timing", "resistance_timing"]
                    )
                ])
            
            elif 'religious' in alt_desc or 'awakening' in alt_desc:
                predictions.extend([
                    TestPrediction(
                        prediction_id=f"{hypothesis_id}_PRED_001",
                        hypothesis_id=hypothesis_id,
                        description="Political rhetoric must contain systematic religious/moral language",
                        diagnostic_type=DiagnosticType.SMOKING_GUN,
                        necessary_condition=False,
                        sufficient_condition=True,
                        evidence_requirements=["religious_rhetoric", "moral_arguments", "clerical_leadership"]
                    ),
                    TestPrediction(
                        prediction_id=f"{hypothesis_id}_PRED_002",
                        hypothesis_id=hypothesis_id,
                        description="Regional religious revival intensity must correlate with political resistance",
                        diagnostic_type=DiagnosticType.HOOP,
                        necessary_condition=True,
                        sufficient_condition=False,
                        evidence_requirements=["religious_data", "revival_timing", "regional_variation"]
                    )
                ])
        
        return predictions
    
    def evaluate_prediction(self, prediction: TestPrediction) -> TestEvaluation:
        """
        Systematically evaluate whether prediction passes or fails based on available evidence.
        Implements Van Evera's logic for each diagnostic test type.
        """
        # Find relevant evidence for this prediction
        relevant_evidence = []
        contradicting_evidence = []
        
        for edge in self.evidence_edges:
            if edge['target_id'] == prediction.hypothesis_id:
                evidence_node = next((n for n in self.graph_data['nodes'] 
                                    if n['id'] == edge['source_id']), None)
                if evidence_node:
                    evidence_desc = evidence_node.get('properties', {}).get('description', '').lower()
                    edge_type = edge.get('type', '')
                    
                    # Check if evidence relates to this prediction
                    prediction_keywords = self._extract_prediction_keywords(prediction)
                    if any(keyword in evidence_desc for keyword in prediction_keywords):
                        if edge_type in ['supports', 'provides_evidence_for']:
                            relevant_evidence.append(evidence_node['id'])
                        elif edge_type in ['refutes', 'contradicts']:
                            contradicting_evidence.append(evidence_node['id'])
        
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
    
    def _extract_prediction_keywords(self, prediction: TestPrediction) -> List[str]:
        """Extract keywords from prediction for evidence matching"""
        desc = prediction.description.lower()
        keywords = []
        
        # Extract key terms based on prediction content
        if 'rights' in desc or 'constitutional' in desc:
            keywords.extend(['rights', 'constitutional', 'liberty', 'freedom', 'magna carta'])
        if 'merchant' in desc or 'trade' in desc:
            keywords.extend(['merchant', 'trade', 'commercial', 'business', 'profit'])
        if 'religious' in desc or 'moral' in desc:
            keywords.extend(['religious', 'god', 'christian', 'clergy', 'moral'])
        if 'governance' in desc or 'institutional' in desc:
            keywords.extend(['governance', 'institution', 'assembly', 'government'])
        if 'correlation' in desc or 'timing' in desc:
            keywords.extend(['timing', 'correlation', 'relationship', 'pattern'])
            
        return keywords
    
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
        
        # Count test results by type
        hoop_results = [r for r in test_results if 'HOOP' in r.reasoning]
        smoking_gun_results = [r for r in test_results if 'SMOKING GUN' in r.reasoning]
        decisive_results = [r for r in test_results if 'DOUBLY DECISIVE' in r.reasoning]
        
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