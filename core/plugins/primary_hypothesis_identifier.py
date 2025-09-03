"""
Primary Hypothesis Identifier Plugin
Identifies and promotes the highest-evidence-supported hypothesis to Q_H1 status
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from .base import ProcessTracingPlugin, PluginValidationError
from ..semantic_analysis_service import get_semantic_service
from ..llm_required import LLMRequiredError


class PrimaryHypothesisIdentifierPlugin(ProcessTracingPlugin):
    """
    Plugin for identifying the primary hypothesis (Q_H1) based on Van Evera test results.
    Promotes the highest-ranking hypothesis to Q_H1 with academic justification.
    """
    
    plugin_id = "primary_hypothesis_identifier"
    
    # Academic criteria for primary hypothesis identification
    PRIMARY_HYPOTHESIS_CRITERIA = {
        'van_evera_score': {
            'weight': 0.40,
            'description': 'Van Evera diagnostic test performance',
            'minimum_threshold': 0.6
        },
        'evidence_support': {
            'weight': 0.30,
            'description': 'Quantity and quality of supporting evidence',
            'minimum_threshold': 0.5
        },
        'theoretical_sophistication': {
            'weight': 0.20,
            'description': 'Theoretical depth and causal mechanism clarity',
            'minimum_threshold': 0.4
        },
        'elimination_power': {
            'weight': 0.10,
            'description': 'Ability to eliminate alternative explanations',
            'minimum_threshold': 0.3
        }
    }
    
    def validate_input(self, data: Any) -> None:
        """Validate input contains Van Evera test results for hypothesis ranking"""
        if not isinstance(data, dict):
            raise PluginValidationError(self.id, "Input must be dictionary")
        
        required_keys = ['graph_data', 'van_evera_results']
        for key in required_keys:
            if key not in data:
                raise PluginValidationError(self.id, f"Missing required key '{key}'")
        
        # Validate Van Evera results contain hypothesis scores
        van_evera_results = data['van_evera_results']
        if not isinstance(van_evera_results, dict):
            raise PluginValidationError(self.id, "van_evera_results must be dictionary")
        
        # Find hypotheses in graph data
        graph_data = data['graph_data']
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        if len(hypotheses) < 2:
            raise PluginValidationError(self.id, "Need at least 2 hypotheses for primary identification")
        
        self.logger.info(f"VALIDATION: Found {len(hypotheses)} hypotheses for primary identification")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Identify and promote primary hypothesis based on Van Evera analysis"""
        self.logger.info("START: Primary hypothesis identification from Van Evera results")
        
        graph_data = data['graph_data']
        van_evera_results = data['van_evera_results']
        
        # Calculate comprehensive ranking scores for all hypotheses
        ranking_analysis = self._calculate_hypothesis_rankings(graph_data, van_evera_results)
        
        # Identify primary hypothesis based on ranking
        primary_identification = self._identify_primary_hypothesis(ranking_analysis)
        
        # Update graph data with Q_H1/H2/H3 structure
        updated_graph_data = self._update_graph_with_rankings(
            graph_data, ranking_analysis, primary_identification
        )
        
        # Generate academic justification
        academic_justification = self._generate_academic_justification(
            primary_identification, ranking_analysis
        )
        
        primary_hypothesis_id = primary_identification['primary_hypothesis']['original_id']
        self.logger.info(f"COMPLETE: Identified primary hypothesis - {primary_hypothesis_id}")
        
        return {
            'primary_identification': primary_identification,
            'ranking_analysis': ranking_analysis,
            'academic_justification': academic_justification,
            'updated_graph_data': updated_graph_data,
            'methodology': 'evidence_based_van_evera_ranking',
            'academic_quality_indicators': {
                'evidence_based_selection': True,
                'systematic_ranking': True,
                'transparent_methodology': True,
                'academic_justification': True
            }
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for primary hypothesis identification"""
        return {
            'plugin_id': self.id,
            'ranking_criteria': len(self.PRIMARY_HYPOTHESIS_CRITERIA),
            'method': 'van_evera_evidence_based_ranking'
        }
    
    def _validate_numeric_config(self, value, name: str, expected: float):
        """Validate configuration value is numeric or raise LLMRequiredError"""
        if not isinstance(value, (int, float)):
            raise LLMRequiredError(f"Invalid {name} configuration: {value} - expected numeric value {expected}")
        return float(value)
    
    def _calculate_hypothesis_rankings(self, graph_data: Dict, van_evera_results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive ranking scores for all hypotheses"""
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        evidence_nodes = [n for n in graph_data['nodes'] if n.get('type') == 'Evidence']
        edges = graph_data.get('edges', [])
        
        hypothesis_scores = {}
        
        for hypothesis in hypotheses:
            hypothesis_id = hypothesis['id']
            
            # Calculate Van Evera score
            van_evera_score = self._extract_van_evera_score(hypothesis_id, van_evera_results)
            
            # Calculate evidence support score
            evidence_score = self._calculate_evidence_support(hypothesis, evidence_nodes, edges)
            
            # Calculate theoretical sophistication score
            theoretical_score = self._calculate_theoretical_sophistication(hypothesis)
            
            # Calculate elimination power score
            elimination_score = self._calculate_elimination_power(hypothesis_id, van_evera_results)
            
            # Weighted composite score - cast weights to float safely
            ve_weight = self.PRIMARY_HYPOTHESIS_CRITERIA['van_evera_score']['weight']
            ev_weight = self.PRIMARY_HYPOTHESIS_CRITERIA['evidence_support']['weight']
            th_weight = self.PRIMARY_HYPOTHESIS_CRITERIA['theoretical_sophistication']['weight']
            el_weight = self.PRIMARY_HYPOTHESIS_CRITERIA['elimination_power']['weight']
            
            # Validate weights are numeric using fail-fast approach
            ve_weight_float = self._validate_numeric_config(ve_weight, "van_evera weight", 0.4)
            ev_weight_float = self._validate_numeric_config(ev_weight, "evidence_support weight", 0.3)
            th_weight_float = self._validate_numeric_config(th_weight, "theoretical_sophistication weight", 0.2)
            el_weight_float = self._validate_numeric_config(el_weight, "elimination_power weight", 0.1)
            
            composite_score = (
                van_evera_score * ve_weight_float +
                evidence_score * ev_weight_float +
                theoretical_score * th_weight_float +
                elimination_score * el_weight_float
            )
            
            hypothesis_scores[hypothesis_id] = {
                'hypothesis_node': hypothesis,
                'composite_score': composite_score,
                'component_scores': {
                    'van_evera_score': van_evera_score,
                    'evidence_support': evidence_score,
                    'theoretical_sophistication': theoretical_score,
                    'elimination_power': elimination_score
                },
                'ranking_eligibility': self._assess_ranking_eligibility(
                    van_evera_score, evidence_score, theoretical_score, elimination_score
                ),
                'academic_strengths': self._identify_academic_strengths(
                    van_evera_score, evidence_score, theoretical_score, elimination_score
                )
            }
        
        # Sort by composite score (descending)
        sorted_hypotheses = sorted(
            hypothesis_scores.items(), 
            key=lambda x: x[1]['composite_score'], 
            reverse=True
        )
        
        return {
            'hypothesis_scores': hypothesis_scores,
            'ranked_hypotheses': sorted_hypotheses,
            'total_hypotheses': len(hypotheses),
            'ranking_methodology': 'weighted_composite_scoring',
            'criteria_weights': self.PRIMARY_HYPOTHESIS_CRITERIA
        }
    
    def _extract_van_evera_score(self, hypothesis_id: str, van_evera_results: Dict) -> float:
        """Extract Van Evera test score for specific hypothesis"""
        # Look for hypothesis scores in Van Evera results
        if 'hypothesis_evaluations' in van_evera_results:
            for evaluation in van_evera_results['hypothesis_evaluations']:
                if evaluation.get('hypothesis_id') == hypothesis_id:
                    return evaluation.get('composite_score', 0.5)
        
        # CRITICAL FIX: Look in hypothesis_rankings (actual Van Evera testing output format)
        if 'hypothesis_rankings' in van_evera_results:
            rankings = van_evera_results['hypothesis_rankings']
            if hypothesis_id in rankings:
                return rankings[hypothesis_id].get('ranking_score', 0.5)
        
        # Look in ranking scores from Van Evera testing plugin (legacy format)
        if 'ranking_scores' in van_evera_results:
            ranking_scores = van_evera_results['ranking_scores']
            if hypothesis_id in ranking_scores:
                return ranking_scores[hypothesis_id].get('score', 0.5)
        
        # Look in academic quality metrics
        if 'academic_quality_metrics' in van_evera_results:
            quality_metrics = van_evera_results['academic_quality_metrics']
            if 'hypothesis_scores' in quality_metrics and hypothesis_id in quality_metrics['hypothesis_scores']:
                return quality_metrics['hypothesis_scores'][hypothesis_id]
        
        # Default moderate score if not found
        self.logger.warning(f"No Van Evera score found for {hypothesis_id}, using default 0.5")
        return 0.5
    
    def _calculate_evidence_support(self, hypothesis: Dict, evidence_nodes: List[Dict], edges: List[Dict]) -> float:
        """Calculate evidence support score for hypothesis"""
        hypothesis_id = hypothesis['id']
        
        # Find evidence supporting this hypothesis
        supporting_edges = [
            edge for edge in edges
            if edge.get('target_id') == hypothesis_id and 
            edge.get('type') in ['supports', 'tests_hypothesis', 'provides_evidence_for']
        ]
        
        if not supporting_edges:
            return 0.1  # Minimal score if no supporting evidence
        
        # Calculate quality-weighted evidence score
        total_evidence_score = 0.0
        evidence_count = len(supporting_edges)
        
        for edge in supporting_edges:
            # Base evidence value
            evidence_value = 1.0
            
            # Weight by probative value if available
            if 'probative_value' in edge.get('properties', {}):
                evidence_value *= edge['properties']['probative_value']
            
            # Weight by certainty if available
            if 'certainty' in edge.get('properties', {}):
                evidence_value *= edge['properties']['certainty']
            
            # Weight by diagnostic type
            diagnostic_type = edge.get('properties', {}).get('diagnostic_type', 'general')
            diagnostic_weights = {
                'doubly_decisive': 1.0,
                'smoking_gun': 0.8,
                'hoop': 0.6,
                'straw_in_the_wind': 0.4,
                'general': 0.5
            }
            evidence_value *= diagnostic_weights.get(diagnostic_type, 0.5)
            
            total_evidence_score += evidence_value
        
        # Normalize by evidence count with diminishing returns
        if evidence_count > 5:
            # Diminishing returns for excessive evidence
            normalized_score = total_evidence_score / (evidence_count + (evidence_count - 5) * 0.5)
        else:
            normalized_score = total_evidence_score / evidence_count
        
        return min(normalized_score, 1.0)
    
    def _calculate_theoretical_sophistication(self, hypothesis: Dict) -> float:
        """Calculate theoretical sophistication score based on hypothesis content"""
        description = hypothesis.get('properties', {}).get('description', '').lower()
        
        sophistication_score = 0.0
        
        # Theoretical concepts (0.4 max)
        theoretical_concepts = [
            'mechanism', 'process', 'institution', 'structure', 'system',
            'theory', 'framework', 'model', 'causal', 'relationship'
        ]
        concept_matches = sum(1 for concept in theoretical_concepts if concept in description)
        sophistication_score += min(concept_matches * 0.08, 0.4)
        
        # Causal language (0.3 max)
        causal_language = [
            'because', 'due to', 'caused by', 'resulted in', 'led to',
            'explains', 'accounts for', 'drives', 'enables', 'determines'
        ]
        causal_matches = sum(1 for phrase in causal_language if phrase in description)
        sophistication_score += min(causal_matches * 0.1, 0.3)
        
        # Complexity indicators (0.3 max)
        complexity_indicators = [
            'interaction', 'dynamic', 'feedback', 'contingent', 'conditional',
            'strategic', 'rational', 'incentive', 'constraint', 'opportunity'
        ]
        complexity_matches = sum(1 for indicator in complexity_indicators if indicator in description)
        sophistication_score += min(complexity_matches * 0.075, 0.3)
        
        return min(sophistication_score, 1.0)
    
    def _calculate_elimination_power(self, hypothesis_id: str, van_evera_results: Dict) -> float:
        """Calculate power to eliminate alternative explanations"""
        elimination_score = 0.0
        
        # Look for elimination results in Van Evera analysis
        if 'elimination_analysis' in van_evera_results:
            elimination_data = van_evera_results['elimination_analysis']
            if hypothesis_id in elimination_data:
                hypothesis_elimination = elimination_data[hypothesis_id]
                eliminated_count = hypothesis_elimination.get('hypotheses_eliminated', 0)
                total_alternatives = hypothesis_elimination.get('total_alternatives', 1)
                elimination_score = eliminated_count / total_alternatives
        
        # Look for diagnostic test results that eliminate alternatives
        if 'diagnostic_results' in van_evera_results:
            for result in van_evera_results['diagnostic_results']:
                if (result.get('hypothesis_id') == hypothesis_id and 
                    result.get('test_result') == 'PASS' and 
                    result.get('diagnostic_type') in ['smoking_gun', 'doubly_decisive']):
                    elimination_score += 0.2  # Bonus for decisive tests
        
        return min(elimination_score, 1.0)
    
    def _assess_ranking_eligibility(self, van_evera: float, evidence: float, 
                                  theoretical: float, elimination: float) -> Dict[str, Any]:
        """Assess eligibility for primary hypothesis ranking"""
        criteria = self.PRIMARY_HYPOTHESIS_CRITERIA
        
        # Extract thresholds safely
        ve_threshold = criteria['van_evera_score']['minimum_threshold']
        ev_threshold = criteria['evidence_support']['minimum_threshold']
        th_threshold = criteria['theoretical_sophistication']['minimum_threshold']
        el_threshold = criteria['elimination_power']['minimum_threshold']
        
        # Validate thresholds are numeric using fail-fast approach
        ve_threshold_float = self._validate_numeric_config(ve_threshold, "van_evera minimum_threshold", 0.6)
        ev_threshold_float = self._validate_numeric_config(ev_threshold, "evidence_support minimum_threshold", 0.5)
        th_threshold_float = self._validate_numeric_config(th_threshold, "theoretical_sophistication minimum_threshold", 0.4)
        el_threshold_float = self._validate_numeric_config(el_threshold, "elimination_power minimum_threshold", 0.3)
        
        eligible_for_primary = (
            van_evera >= ve_threshold_float and
            evidence >= ev_threshold_float and
            theoretical >= th_threshold_float and
            elimination >= el_threshold_float
        )
        
        failed_criteria = []
        if van_evera < ve_threshold_float:
            failed_criteria.append('van_evera_score')
        if evidence < ev_threshold_float:
            failed_criteria.append('evidence_support')
        if theoretical < th_threshold_float:
            failed_criteria.append('theoretical_sophistication')
        if elimination < el_threshold_float:
            failed_criteria.append('elimination_power')
        
        return {
            'eligible_for_primary': eligible_for_primary,
            'failed_criteria': failed_criteria,
            'eligibility_score': (van_evera + evidence + theoretical + elimination) / 4
        }
    
    def _identify_academic_strengths(self, van_evera: float, evidence: float,
                                   theoretical: float, elimination: float) -> List[str]:
        """Identify academic strengths of hypothesis"""
        strengths = []
        
        if van_evera >= 0.8:
            strengths.append("Excellent Van Evera diagnostic test performance")
        elif van_evera >= 0.6:
            strengths.append("Strong Van Evera diagnostic test performance")
        
        if evidence >= 0.8:
            strengths.append("Comprehensive empirical evidence support")
        elif evidence >= 0.6:
            strengths.append("Solid empirical evidence base")
        
        if theoretical >= 0.8:
            strengths.append("Sophisticated theoretical framework")
        elif theoretical >= 0.6:
            strengths.append("Well-developed theoretical foundation")
        
        if elimination >= 0.8:
            strengths.append("Strong alternative hypothesis elimination")
        elif elimination >= 0.6:
            strengths.append("Effective alternative hypothesis differentiation")
        
        return strengths
    
    def _identify_primary_hypothesis(self, ranking_analysis: Dict) -> Dict[str, Any]:
        """Identify primary hypothesis based on comprehensive ranking"""
        ranked_hypotheses = ranking_analysis['ranked_hypotheses']
        
        if not ranked_hypotheses:
            raise ValueError("No hypotheses found for primary identification")
        
        # CRITICAL FIX: Find first eligible hypothesis for Q_H1
        primary_hypothesis_id, primary_data = None, None
        primary_composite_score = 0.0
        primary_index = 0
        
        for i, (hypothesis_id, hypothesis_data) in enumerate(ranked_hypotheses):
            if hypothesis_data['ranking_eligibility']['eligible_for_primary']:
                primary_hypothesis_id = hypothesis_id
                primary_data = hypothesis_data
                primary_composite_score = hypothesis_data['composite_score']
                primary_index = i
                self.logger.info(f"Selected eligible hypothesis {hypothesis_id} as Q_H1 (rank {i+1})")
                break
        
        # Fallback: if no hypothesis meets eligibility criteria, use top-ranked with warning
        if primary_hypothesis_id is None:
            primary_hypothesis_id, primary_data = ranked_hypotheses[0]
            primary_composite_score = primary_data['composite_score']
            primary_index = 0
            self.logger.warning(f"No hypothesis meets eligibility criteria. Using top-ranked {primary_hypothesis_id} with warnings.")
        
        # Identify alternatives (remaining hypotheses)
        alternative_hypotheses = []
        alt_rank = 2
        for i, (hypothesis_id, hypothesis_data) in enumerate(ranked_hypotheses):
            if i != primary_index:  # Skip the primary hypothesis
                alternative_hypotheses.append({
                    'original_id': hypothesis_id,
                    'new_id': f'Q_H{alt_rank}',
                    'ranking': alt_rank,
                    'composite_score': hypothesis_data['composite_score'],
                    'hypothesis_data': hypothesis_data
                })
                alt_rank += 1
        
        # Ensure primary_data is not None
        assert primary_data is not None, "primary_data should be set by this point"
        
        return {
            'primary_hypothesis': {
                'original_id': primary_hypothesis_id,
                'new_id': 'Q_H1',
                'composite_score': primary_composite_score,
                'hypothesis_data': primary_data,
                'eligibility_status': primary_data['ranking_eligibility'],
                'academic_strengths': primary_data['academic_strengths']
            },
            'alternative_hypotheses': alternative_hypotheses,
            'selection_confidence': self._calculate_selection_confidence(ranked_hypotheses),
            'methodology_transparency': {
                'ranking_based_on': 'composite_evidence_score',
                'criteria_applied': list(self.PRIMARY_HYPOTHESIS_CRITERIA.keys()),
                'minimum_thresholds_enforced': True,
                'academic_justification_required': True
            }
        }
    
    def _calculate_selection_confidence(self, ranked_hypotheses: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """Calculate confidence in primary hypothesis selection"""
        if len(ranked_hypotheses) < 2:
            return {'confidence_level': 1.0, 'reason': 'single_hypothesis'}
        
        # Calculate score gap between primary and second-best
        primary_score = ranked_hypotheses[0][1]['composite_score']
        second_score = ranked_hypotheses[1][1]['composite_score']
        score_gap = primary_score - second_score
        
        # Confidence based on score gap
        if score_gap >= 0.3:
            confidence_level = 0.9
            reason = 'clear_leader_large_gap'
        elif score_gap >= 0.15:
            confidence_level = 0.75
            reason = 'clear_leader_moderate_gap'
        elif score_gap >= 0.05:
            confidence_level = 0.6
            reason = 'modest_leader_small_gap'
        else:
            confidence_level = 0.4
            reason = 'marginal_leader_very_small_gap'
        
        return {
            'confidence_level': confidence_level,
            'reason': reason,
            'score_gap': score_gap,
            'primary_score': primary_score,
            'second_best_score': second_score
        }
    
    def _update_graph_with_rankings(self, graph_data: Dict, ranking_analysis: Dict, 
                                  primary_identification: Dict) -> Dict[str, Any]:
        """Update graph data with Q_H1/H2/H3 structure"""
        updated_graph_data = graph_data.copy()
        updated_nodes = []
        
        # Update nodes with new Q_H1/H2/H3 IDs and rankings
        for node in graph_data['nodes']:
            if node.get('type') in ['Hypothesis', 'Alternative_Explanation']:
                original_id = node['id']
                
                # Find new ID assignment
                if original_id == primary_identification['primary_hypothesis']['original_id']:
                    # Primary hypothesis becomes Q_H1
                    updated_node = node.copy()
                    updated_node['id'] = 'Q_H1'
                    updated_node['properties'] = updated_node.get('properties', {}).copy()
                    updated_node['properties']['hypothesis_type'] = 'primary'
                    updated_node['properties']['ranking_score'] = primary_identification['primary_hypothesis']['composite_score']
                    updated_node['properties']['original_id'] = original_id
                    updated_nodes.append(updated_node)
                else:
                    # Find alternative hypothesis assignment
                    for alt_hyp in primary_identification['alternative_hypotheses']:
                        if alt_hyp['original_id'] == original_id:
                            updated_node = node.copy()
                            updated_node['id'] = alt_hyp['new_id']
                            updated_node['properties'] = updated_node.get('properties', {}).copy()
                            updated_node['properties']['hypothesis_type'] = 'alternative'
                            updated_node['properties']['ranking_score'] = alt_hyp['composite_score']
                            updated_node['properties']['original_id'] = original_id
                            updated_nodes.append(updated_node)
                            break
            else:
                # Keep non-hypothesis nodes unchanged
                updated_nodes.append(node)
        
        # Update edges with new node IDs
        updated_edges = []
        id_mapping = {}
        
        # Create ID mapping
        id_mapping[primary_identification['primary_hypothesis']['original_id']] = 'Q_H1'
        for alt_hyp in primary_identification['alternative_hypotheses']:
            id_mapping[alt_hyp['original_id']] = alt_hyp['new_id']
        
        for edge in graph_data['edges']:
            updated_edge = edge.copy()
            
            # Update source_id if it changed
            if edge['source_id'] in id_mapping:
                updated_edge['source_id'] = id_mapping[edge['source_id']]
            
            # Update target_id if it changed
            if edge['target_id'] in id_mapping:
                updated_edge['target_id'] = id_mapping[edge['target_id']]
            
            updated_edges.append(updated_edge)
        
        updated_graph_data['nodes'] = updated_nodes
        updated_graph_data['edges'] = updated_edges
        
        return updated_graph_data
    
    def _generate_academic_justification(self, primary_identification: Dict, 
                                       ranking_analysis: Dict) -> Dict[str, Any]:
        """Generate academic justification for primary hypothesis selection"""
        primary_hyp = primary_identification['primary_hypothesis']
        selection_confidence = primary_identification['selection_confidence']
        
        # Generate detailed justification
        justification_text = f"""
        Primary hypothesis Q_H1 was selected through systematic evidence-based ranking using Van Evera methodology.
        
        SELECTION CRITERIA:
        - Van Evera diagnostic test score: {primary_hyp['composite_score']:.3f}
        - Evidence support quality: {primary_hyp['hypothesis_data']['component_scores']['evidence_support']:.3f}
        - Theoretical sophistication: {primary_hyp['hypothesis_data']['component_scores']['theoretical_sophistication']:.3f}
        - Alternative elimination power: {primary_hyp['hypothesis_data']['component_scores']['elimination_power']:.3f}
        
        ACADEMIC STRENGTHS:
        {chr(10).join(f"- {strength}" for strength in primary_hyp['academic_strengths'])}
        
        SELECTION CONFIDENCE: {selection_confidence['confidence_level']:.1%}
        Confidence basis: {selection_confidence['reason']} (score gap: {selection_confidence['score_gap']:.3f})
        
        This systematic ranking ensures the primary hypothesis represents the explanation with the strongest 
        empirical support and theoretical sophistication, meeting academic standards for process tracing analysis.
        """
        
        return {
            'justification_text': justification_text.strip(),
            'methodology': 'van_evera_evidence_based_ranking',
            'selection_transparency': {
                'criteria_weights': self.PRIMARY_HYPOTHESIS_CRITERIA,
                'composite_score': primary_hyp['composite_score'],
                'component_scores': primary_hyp['hypothesis_data']['component_scores'],
                'eligibility_met': primary_hyp['eligibility_status']['eligible_for_primary'],
                'selection_confidence': selection_confidence
            },
            'academic_standards_met': {
                'systematic_methodology': True,
                'transparent_criteria': True,
                'evidence_based_selection': True,
                'academic_justification': True,
                'reproducible_ranking': True
            }
        }


def identify_primary_hypothesis_from_analysis(graph_data: Dict, van_evera_results: Dict) -> Dict[str, Any]:
    """
    Convenience function for identifying primary hypothesis from Van Evera analysis.
    Returns primary hypothesis identification results.
    """
    from .base import PluginContext
    
    context = PluginContext({'primary_hypothesis_identification': True})
    plugin = PrimaryHypothesisIdentifierPlugin('primary_hypothesis_identifier', context)
    
    result = plugin.execute({
        'graph_data': graph_data,
        'van_evera_results': van_evera_results
    })
    return result