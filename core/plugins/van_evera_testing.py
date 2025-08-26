"""
Van Evera Systematic Hypothesis Testing Plugin
Implements academic Van Evera methodology as extensible plugin workflow
"""

import json
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .base import ProcessTracingPlugin, PluginValidationError


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


class VanEveraTestingPlugin(ProcessTracingPlugin):
    """Plugin for systematic Van Evera hypothesis testing with academic rigor"""
    
    plugin_id = "van_evera_testing"
    
    def validate_input(self, data: Any) -> None:
        """Validate that graph_data contains required Van Evera components"""
        if not isinstance(data, dict):
            raise PluginValidationError(self.id, "Input must be dictionary")
        
        if 'graph_data' not in data:
            raise PluginValidationError(self.id, "Missing required key 'graph_data'")
        
        graph_data = data['graph_data']
        if 'nodes' not in graph_data or 'edges' not in graph_data:
            raise PluginValidationError(self.id, "graph_data must contain 'nodes' and 'edges'")
        
        # Verify presence of hypotheses and evidence
        nodes = graph_data['nodes']
        hypotheses = [n for n in nodes if n.get('type') == 'Hypothesis']
        evidence = [n for n in nodes if n.get('type') == 'Evidence']
        
        if len(hypotheses) == 0:
            raise PluginValidationError(self.id, "No hypotheses found for Van Evera testing")
        
        if len(evidence) == 0:
            raise PluginValidationError(self.id, "No evidence found for Van Evera testing")
        
        self.logger.info(f"VALIDATION: Found {len(hypotheses)} hypotheses and {len(evidence)} evidence items")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Execute Van Evera testing with advanced prediction engine"""
        self.logger.info("START: Van Evera testing with advanced prediction engine")
        
        graph_data = data['graph_data']
        llm_query_func = self.context.get_data('llm_query_func')
        
        # Use advanced prediction engine
        from .advanced_van_evera_prediction_engine import enhance_van_evera_testing_with_sophistication
        
        advanced_results = enhance_van_evera_testing_with_sophistication(graph_data, llm_query_func)
        
        # Extract metrics for backward compatibility
        testing_compliance = advanced_results['testing_compliance_score']
        academic_quality_metrics = advanced_results['academic_quality_metrics']
        
        self.logger.info(f"COMPLETE: Testing compliance achieved: {testing_compliance:.1f}%")
        
        # Calculate hypothesis rankings based on Van Evera test results
        hypothesis_rankings = self._calculate_hypothesis_rankings(graph_data, advanced_results)
        
        # Update hypothesis nodes with ranking scores
        updated_graph_data = self._update_hypothesis_rankings(graph_data, hypothesis_rankings)
        
        return {
            'updated_graph_data': updated_graph_data,
            'hypothesis_rankings': hypothesis_rankings,
            'advanced_testing_results': advanced_results,
            'testing_compliance_score': testing_compliance,
            'academic_quality_metrics': academic_quality_metrics,
            'publication_ready': testing_compliance >= 80
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for Van Evera testing"""
        return {
            'plugin_id': self.id,
            'methodology': 'Van Evera Process Tracing',
            'academic_standards': 'Systematic diagnostic testing with Bayesian updating'
        }
    
    def _calculate_academic_quality_metrics(self, assessments: Dict[str, HypothesisAssessment]) -> Dict[str, Any]:
        """Calculate academic quality compliance metrics"""
        total_hypotheses = len(assessments)
        eliminated_count = sum(1 for a in assessments.values() if a.overall_status == "ELIMINATED")
        supported_count = sum(1 for a in assessments.values() if "SUPPORTED" in a.overall_status)
        
        # Count diagnostic test types across all assessments
        diagnostic_counts = {'hoop': 0, 'smoking_gun': 0, 'doubly_decisive': 0, 'straw_in_wind': 0}
        total_tests = 0
        
        for assessment in assessments.values():
            for test_result in assessment.test_results:
                total_tests += 1
                # Extract diagnostic type from reasoning
                if 'HOOP' in test_result.reasoning:
                    diagnostic_counts['hoop'] += 1
                elif 'SMOKING GUN' in test_result.reasoning:
                    diagnostic_counts['smoking_gun'] += 1
                elif 'DOUBLY DECISIVE' in test_result.reasoning:
                    diagnostic_counts['doubly_decisive'] += 1
                elif 'STRAW IN WIND' in test_result.reasoning:
                    diagnostic_counts['straw_in_wind'] += 1
        
        # Calculate diagnostic distribution
        diagnostic_distribution = {}
        if total_tests > 0:
            diagnostic_distribution = {k: v/total_tests for k, v in diagnostic_counts.items()}
        
        # Van Evera target distribution
        target_distribution = {'hoop': 0.25, 'smoking_gun': 0.25, 'doubly_decisive': 0.15, 'straw_in_wind': 0.35}
        
        # Calculate academic compliance score
        compliance_score = 0.0
        for test_type, target in target_distribution.items():
            actual = diagnostic_distribution.get(test_type, 0)
            deviation = abs(target - actual)
            test_compliance = max(0, 100 - deviation * 200)  # Penalty for deviation
            compliance_score += test_compliance * 0.25
        
        return {
            'total_hypotheses': total_hypotheses,
            'hypotheses_eliminated': eliminated_count,
            'hypotheses_supported': supported_count,
            'theoretical_competition_ratio': eliminated_count / max(total_hypotheses, 1),
            'diagnostic_test_distribution': diagnostic_distribution,
            'target_distribution': target_distribution,
            'academic_compliance_score': round(compliance_score, 1),
            'total_diagnostic_tests': total_tests,
            'systematic_methodology_applied': True
        }
    
    def _generate_academic_summary(self, assessments: Dict[str, HypothesisAssessment], 
                                 quality_metrics: Dict[str, Any]) -> str:
        """Generate academic-quality summary of Van Evera analysis"""
        summary = f"VAN EVERA PROCESS TRACING ANALYSIS\n"
        summary += f"{'='*50}\n\n"
        
        summary += f"METHODOLOGY OVERVIEW:\n"
        summary += f"- Systematic hypothesis testing with {quality_metrics['total_diagnostic_tests']} diagnostic tests\n"
        summary += f"- Bayesian probability updating with elimination logic\n"
        summary += f"- Academic compliance score: {quality_metrics['academic_compliance_score']:.1f}%\n\n"
        
        summary += f"THEORETICAL COMPETITION:\n"
        summary += f"- Total hypotheses tested: {quality_metrics['total_hypotheses']}\n"
        summary += f"- Hypotheses eliminated: {quality_metrics['hypotheses_eliminated']}\n"
        summary += f"- Hypotheses supported: {quality_metrics['hypotheses_supported']}\n"
        summary += f"- Competition ratio: {quality_metrics['theoretical_competition_ratio']:.2f}\n\n"
        
        summary += f"DIAGNOSTIC TEST DISTRIBUTION:\n"
        for test_type, actual_pct in quality_metrics['diagnostic_test_distribution'].items():
            target_pct = quality_metrics['target_distribution'][test_type]
            summary += f"- {test_type.title()}: {actual_pct:.1%} (target: {target_pct:.1%})\n"
        
        summary += f"\nHYPOTHESIS ASSESSMENT SUMMARY:\n"
        for hyp_id, assessment in assessments.items():
            summary += f"- {hyp_id}: {assessment.overall_status} "
            summary += f"(posterior: {assessment.posterior_probability:.2f})\n"
        
        return summary
    
    def _calculate_hypothesis_rankings(self, graph_data: Dict, advanced_results: Dict) -> Dict[str, Dict[str, Any]]:
        """Calculate hypothesis rankings based on Van Evera test results for Q/H1/H2/H3 structure"""
        rankings: Dict[str, Dict[str, Any]] = {}
        
        # Get all hypotheses
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') == 'Hypothesis']
        
        if not hypotheses:
            return rankings
        
        # Extract evaluation results from advanced results
        if 'evaluation_results' in advanced_results and 'evaluations' in advanced_results['evaluation_results']:
            evaluations = advanced_results['evaluation_results']['evaluations']
            
            # Group evaluations by hypothesis
            hypothesis_evaluations: Dict[str, List[Dict[str, Any]]] = {}
            for evaluation in evaluations:
                hypothesis_id = evaluation.get('hypothesis_id')
                if hypothesis_id not in hypothesis_evaluations:
                    hypothesis_evaluations[hypothesis_id] = []
                hypothesis_evaluations[hypothesis_id].append(evaluation)
            
            # Calculate ranking score for each hypothesis
            for hypothesis in hypotheses:
                hypothesis_id = hypothesis['id']
                hypothesis_evaluations_list = hypothesis_evaluations.get(hypothesis_id, [])
                
                ranking_score = self._calculate_individual_ranking_score(hypothesis_evaluations_list)
                
                # Determine if this is primary (H1) or alternative (H2, H3, etc.)
                hypothesis_type = 'primary' if hypothesis_id.endswith('_H1') or hypothesis_id == 'Q_H1' else 'alternative'
                
                rankings[hypothesis_id] = {
                    'ranking_score': ranking_score,
                    'hypothesis_type': hypothesis_type,
                    'test_count': len(hypothesis_evaluations_list),
                    'academic_rank': None  # Will be set after sorting
                }
        
        # Sort hypotheses by ranking score and assign academic ranks
        sorted_hypotheses = sorted(rankings.items(), key=lambda x: x[1]['ranking_score'], reverse=True)
        
        for rank, (hypothesis_id, ranking_data) in enumerate(sorted_hypotheses, 1):
            ranking_data['academic_rank'] = rank
            
            # Update hypothesis_type based on ranking (top hypothesis becomes primary)
            if rank == 1:
                ranking_data['hypothesis_type'] = 'primary'
            else:
                ranking_data['hypothesis_type'] = 'alternative'
        
        return rankings
    
    def _calculate_individual_ranking_score(self, evaluations: List[Dict]) -> float:
        """Calculate ranking score for individual hypothesis based on Van Evera test results"""
        if not evaluations:
            return 0.0
        
        total_score = 0.0
        weighted_tests = 0.0
        
        # Van Evera diagnostic test weights
        diagnostic_weights = {
            'hoop': 0.8,           # High weight - necessary condition test
            'smoking_gun': 0.9,    # Highest weight - sufficient condition test
            'doubly_decisive': 1.0, # Maximum weight - both necessary and sufficient
            'straw_in_the_wind': 0.3  # Low weight - weak diagnostic test
        }
        
        for evaluation in evaluations:
            test_result = evaluation.get('test_result', 'INCONCLUSIVE')
            diagnostic_type = evaluation.get('diagnostic_type', 'straw_in_the_wind')
            confidence_score = evaluation.get('confidence_score', 0.6)
            
            # Get weight for this diagnostic type
            weight = diagnostic_weights.get(diagnostic_type, 0.5)
            
            # Calculate test contribution to ranking
            if test_result == 'PASS':
                test_contribution = weight * confidence_score
            elif test_result == 'FAIL':
                # Failed tests contribute negatively, especially hoop tests
                if diagnostic_type in ['hoop', 'doubly_decisive']:
                    test_contribution = -weight * confidence_score  # Strong negative for failed necessary conditions
                else:
                    test_contribution = -weight * confidence_score * 0.5  # Moderate negative for failed sufficient conditions
            else:  # INCONCLUSIVE
                test_contribution = weight * 0.1  # Minimal positive for inconclusive
            
            total_score += test_contribution
            weighted_tests += weight
        
        # Normalize score to 0-1 range
        if weighted_tests > 0:
            raw_score = total_score / weighted_tests
            # Transform to 0-1 scale with 0.5 as neutral point
            normalized_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
        else:
            normalized_score = 0.5  # Default neutral score
        
        return round(normalized_score, 3)
    
    def _update_hypothesis_rankings(self, graph_data: Dict, rankings: Dict[str, Dict]) -> Dict:
        """Update hypothesis nodes with ranking scores and types"""
        updated_graph = graph_data.copy()
        
        # Update nodes with ranking information
        for node in updated_graph['nodes']:
            if node.get('type') == 'Hypothesis' and node['id'] in rankings:
                ranking_info = rankings[node['id']]
                
                # Update or add ranking properties
                if 'properties' not in node:
                    node['properties'] = {}
                
                node['properties']['ranking_score'] = ranking_info['ranking_score']
                node['properties']['hypothesis_type'] = ranking_info['hypothesis_type']
                node['properties']['academic_rank'] = ranking_info['academic_rank']
                
                # Update status based on ranking score
                if ranking_info['ranking_score'] >= 0.8:
                    node['properties']['status'] = 'strongly_supported'
                elif ranking_info['ranking_score'] >= 0.6:
                    node['properties']['status'] = 'supported'
                elif ranking_info['ranking_score'] >= 0.4:
                    node['properties']['status'] = 'inconclusive'
                else:
                    node['properties']['status'] = 'weakened'
        
        return updated_graph


class VanEveraTestingEngine:
    """Core Van Evera testing logic (extracted from original implementation)"""
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        self.hypotheses = [n for n in graph_data['nodes'] 
                          if n.get('type') == 'Hypothesis']
        self.evidence = [n for n in graph_data['nodes'] if n.get('type') == 'Evidence']
        self.evidence_edges = [e for e in graph_data['edges'] if self._is_evidence_relationship(e)]
    
    def _is_evidence_relationship(self, edge: Dict) -> bool:
        """Check if edge represents evidence-hypothesis relationship"""
        source_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['source_id']), None)
        target_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['target_id']), None)
        return bool(source_node and source_node.get('type') == 'Evidence' and 
                    target_node and target_node.get('type') == 'Hypothesis')
    
    def systematic_hypothesis_evaluation(self) -> Dict[str, HypothesisAssessment]:
        """Perform systematic Van Evera evaluation of all hypotheses"""
        print("[VAN_EVERA_PLUGIN] Starting systematic hypothesis evaluation...")
        
        assessments = {}
        
        for hypothesis in self.hypotheses:
            hypothesis_id = hypothesis['id']
            print(f"[VAN_EVERA_PLUGIN] Testing hypothesis: {hypothesis_id}")
            
            # Generate testable predictions
            predictions = self._generate_testable_predictions(hypothesis)
            print(f"[VAN_EVERA_PLUGIN] Generated {len(predictions)} predictions for {hypothesis_id}")
            
            # Test each prediction
            test_results = []
            for prediction in predictions:
                result = self._evaluate_prediction(prediction)
                test_results.append(result)
                print(f"[VAN_EVERA_PLUGIN] {prediction.diagnostic_type.value} test: {result.test_result.value}")
            
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
        
        print(f"[VAN_EVERA_PLUGIN] Completed evaluation of {len(assessments)} hypotheses")
        return assessments
    
    def _generate_testable_predictions(self, hypothesis: Dict) -> List[TestPrediction]:
        """Generate specific testable predictions from hypothesis"""
        hypothesis_id = hypothesis['id']
        hypothesis_desc = hypothesis.get('properties', {}).get('description', '')
        
        predictions = []
        
        # Generate 2-3 predictions per hypothesis based on content
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
        else:
            # Generate generic predictions for any hypothesis
            predictions.extend([
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_001",
                    hypothesis_id=hypothesis_id,
                    description="Evidence must be consistent with hypothesis predictions",
                    diagnostic_type=DiagnosticType.HOOP,
                    necessary_condition=True,
                    sufficient_condition=False,
                    evidence_requirements=["supporting_evidence", "consistency_check"]
                ),
                TestPrediction(
                    prediction_id=f"{hypothesis_id}_PRED_002",
                    hypothesis_id=hypothesis_id,
                    description="Key supporting evidence should be present if hypothesis is true",
                    diagnostic_type=DiagnosticType.SMOKING_GUN,
                    necessary_condition=False,
                    sufficient_condition=True,
                    evidence_requirements=["key_evidence", "causal_mechanisms"]
                )
            ])
        
        return predictions
    
    def _evaluate_prediction(self, prediction: TestPrediction) -> TestEvaluation:
        """Evaluate whether prediction passes or fails based on available evidence"""
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
                    
                    # Simple relevance check - always include some evidence for testing
                    if edge_type in ['supports', 'provides_evidence_for', 'evidence']:
                        relevant_evidence.append(evidence_node['id'])
                    elif edge_type in ['refutes', 'contradicts']:
                        contradicting_evidence.append(evidence_node['id'])
        
        # Apply Van Evera diagnostic logic
        test_result, reasoning = self._apply_diagnostic_logic(
            prediction, relevant_evidence, contradicting_evidence
        )
        
        # Calculate confidence level
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
    
    def _apply_diagnostic_logic(self, prediction: TestPrediction, 
                               supporting: List[str], contradicting: List[str]) -> Tuple[TestResult, str]:
        """Apply Van Evera diagnostic test logic"""
        has_support = len(supporting) > 0
        has_contradiction = len(contradicting) > 0
        
        if prediction.diagnostic_type == DiagnosticType.HOOP:
            if has_contradiction or not has_support:
                return TestResult.FAIL, f"HOOP TEST FAILED: Hypothesis eliminated. Supporting: {len(supporting)}, Contradicting: {len(contradicting)}"
            else:
                return TestResult.PASS, f"HOOP TEST PASSED: Hypothesis remains viable. Supporting evidence: {len(supporting)}"
        
        elif prediction.diagnostic_type == DiagnosticType.SMOKING_GUN:
            if has_support and not has_contradiction:
                return TestResult.PASS, f"SMOKING GUN PASSED: Hypothesis strongly confirmed. Supporting: {len(supporting)}"
            elif has_contradiction:
                return TestResult.FAIL, f"SMOKING GUN FAILED: Evidence contradicts prediction. Contradicting: {len(contradicting)}"
            else:
                return TestResult.INCONCLUSIVE, f"SMOKING GUN INCONCLUSIVE: No decisive evidence found."
        
        elif prediction.diagnostic_type == DiagnosticType.DOUBLY_DECISIVE:
            if has_support and not has_contradiction:
                return TestResult.PASS, f"DOUBLY DECISIVE PASSED: Hypothesis confirmed and alternatives eliminated. Supporting: {len(supporting)}"
            else:
                return TestResult.FAIL, f"DOUBLY DECISIVE FAILED: Hypothesis eliminated. Supporting: {len(supporting)}, Contradicting: {len(contradicting)}"
        
        elif prediction.diagnostic_type == DiagnosticType.STRAW_IN_WIND:
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
        evidence_volume_bonus = min(total_evidence * 0.1, 0.3)
        
        base_confidence = support_ratio * 0.7 + evidence_volume_bonus
        return min(base_confidence, 0.95)
    
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
    
    def _calculate_posterior_probability(self, prior: float, test_results: List[TestEvaluation]) -> float:
        """Calculate posterior probability using Bayesian updating"""
        if not test_results:
            return prior
            
        log_odds = math.log(prior / (1 - prior))  # Convert to log odds
        
        for result in test_results:
            if result.test_result == TestResult.PASS:
                log_odds += result.confidence_level * 2
            elif result.test_result == TestResult.FAIL:
                log_odds -= result.confidence_level * 3
        
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
        
        conclusion = f"HYPOTHESIS: {hyp_desc}\n\n"
        
        # Count test results by type
        hoop_results = [r for r in test_results if 'HOOP' in r.reasoning]
        smoking_gun_results = [r for r in test_results if 'SMOKING GUN' in r.reasoning]
        
        # Hoop test analysis
        if hoop_results:
            hoop_passes = sum(1 for r in hoop_results if r.test_result == TestResult.PASS)
            conclusion += f"NECESSARY CONDITION ANALYSIS: {hoop_passes}/{len(hoop_results)} hoop tests passed. "
            if hoop_passes == len(hoop_results):
                conclusion += "Hypothesis remains viable.\n\n"
            else:
                conclusion += "Hypothesis ELIMINATED.\n\n"
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
        
        margin = 0.4 / math.sqrt(n_tests)
        return (max(0, posterior - margin), min(1, posterior + margin))
    
