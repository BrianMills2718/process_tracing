"""
DiagnosticRebalancerPlugin - Transform evidence distribution to Van Evera academic standards
Integrates with existing plugin system to provide production-ready diagnostic rebalancing
"""

import json
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import ProcessTracingPlugin, PluginValidationError, PluginExecutionError
from ..structured_models import EvidenceAssessment
from ..enhance_evidence import refine_evidence_assessment_with_llm


@dataclass
class DiagnosticDistribution:
    """Represents diagnostic test distribution metrics"""
    hoop: int
    smoking_gun: int
    doubly_decisive: int
    straw_in_wind: int
    general: int
    
    @property
    def total(self) -> int:
        return self.hoop + self.smoking_gun + self.doubly_decisive + self.straw_in_wind + self.general
    
    @property
    def percentages(self) -> Dict[str, float]:
        total = self.total
        if total == 0:
            return {'hoop': 0.0, 'smoking_gun': 0.0, 'doubly_decisive': 0.0, 'straw_in_wind': 0.0, 'general': 0.0}
        return {
            'hoop': self.hoop / total,
            'smoking_gun': self.smoking_gun / total,
            'doubly_decisive': self.doubly_decisive / total,
            'straw_in_wind': self.straw_in_wind / total,
            'general': self.general / total
        }
    
    @property
    def academic_compliance_score(self) -> float:
        """Calculate compliance with Van Evera academic standards"""
        target = VAN_EVERA_TARGET_DISTRIBUTION
        actual = self.percentages
        
        compliance = 0.0
        for test_type, target_pct in target.items():
            if test_type in actual:
                actual_pct = actual[test_type]
                deviation = abs(target_pct - actual_pct)
                test_compliance = max(0, 100 - deviation * 200)
                compliance += test_compliance * 0.25
        
        return round(compliance, 1)


@dataclass
class RebalanceResult:
    """Complete result of diagnostic rebalancing operation"""
    success: bool
    original_distribution: DiagnosticDistribution
    rebalanced_distribution: DiagnosticDistribution
    rebalanced_count: int
    enhanced_count: int
    error_count: int
    updated_graph_data: Dict[str, Any]
    compliance_improvement: float
    academic_quality_assessment: Dict[str, Any]


# Van Evera Target Distribution - Academic Standard
VAN_EVERA_TARGET_DISTRIBUTION = {
    'hoop': 0.25,           # Necessary but not sufficient - eliminates if fails
    'smoking_gun': 0.25,    # Sufficient but not necessary - confirms if passes  
    'doubly_decisive': 0.15, # Both necessary and sufficient - decisive either way
    'straw_in_wind': 0.35   # Neither necessary nor sufficient - weak evidence
}

# Van Evera Diagnostic Test Criteria
VAN_EVERA_DIAGNOSTIC_CRITERIA = {
    'hoop': {
        'necessary': True,
        'sufficient': False,
        'elimination_on_fail': True,
        'confirmation_on_pass': False,
        'description': 'Test that hypothesis must pass to remain viable'
    },
    'smoking_gun': {
        'necessary': False,
        'sufficient': True,
        'elimination_on_fail': False,
        'confirmation_on_pass': True,
        'description': 'Test that strongly confirms hypothesis if passed'
    },
    'doubly_decisive': {
        'necessary': True,
        'sufficient': True,
        'elimination_on_fail': True,
        'confirmation_on_pass': True,
        'description': 'Test that is decisive regardless of outcome'
    },
    'straw_in_wind': {
        'necessary': False,
        'sufficient': False,
        'elimination_on_fail': False,
        'confirmation_on_pass': False,
        'description': 'Test that provides weak evidence in either direction'
    }
}


class DiagnosticRebalancerPlugin(ProcessTracingPlugin):
    """
    Plugin for rebalancing diagnostic test distribution to meet Van Evera academic standards.
    
    Transforms current evidence-hypothesis relationships from any distribution 
    (e.g., 50/50 hoop/smoking_gun) to Van Evera standard distribution:
    - 25% hoop tests (necessary conditions)
    - 25% smoking gun tests (sufficient conditions) 
    - 15% doubly decisive tests (both necessary and sufficient)
    - 35% straw-in-wind tests (weak evidence)
    
    Uses LLM assessment to intelligently reclassify evidence based on Van Evera criteria.
    """
    
    plugin_id = "diagnostic_rebalancer"
    
    def __init__(self, plugin_id: str, context):
        super().__init__(plugin_id, context)
        self.target_distribution = VAN_EVERA_TARGET_DISTRIBUTION
        self.diagnostic_criteria = VAN_EVERA_DIAGNOSTIC_CRITERIA
        self.llm_query_func = None
        
    def _initialize_resources(self) -> None:
        """Initialize LLM query function for evidence assessment"""
        # Get LLM query function from context if available
        self.llm_query_func = self.context.get_data('llm_query_func')
        if not self.llm_query_func:
            self.logger.warning("No LLM query function available - will use rule-based assessment")
    
    def validate_input(self, data: Any) -> None:
        """Validate input contains graph data with evidence-hypothesis relationships"""
        if not isinstance(data, dict):
            raise PluginValidationError(self.id, "Input must be dictionary")
        
        if 'graph_data' not in data:
            raise PluginValidationError(self.id, "Missing required key 'graph_data'")
        
        graph_data = data['graph_data']
        if not isinstance(graph_data, dict):
            raise PluginValidationError(self.id, "graph_data must be dictionary")
        
        if 'nodes' not in graph_data or 'edges' not in graph_data:
            raise PluginValidationError(self.id, "graph_data must contain 'nodes' and 'edges'")
        
        # Validate presence of evidence and hypotheses
        nodes = graph_data['nodes']
        evidence_nodes = [n for n in nodes if n.get('type') == 'Evidence']
        hypothesis_nodes = [n for n in nodes if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        
        if len(evidence_nodes) == 0:
            raise PluginValidationError(self.id, "No evidence nodes found for diagnostic rebalancing")
        
        if len(hypothesis_nodes) == 0:
            raise PluginValidationError(self.id, "No hypothesis nodes found for diagnostic rebalancing")
        
        # Validate evidence-hypothesis edges exist
        evidence_edges = self._find_evidence_edges(graph_data)
        if len(evidence_edges) == 0:
            raise PluginValidationError(self.id, "No evidence-hypothesis relationships found")
        
        self.logger.info(f"VALIDATION: Found {len(evidence_nodes)} evidence, {len(hypothesis_nodes)} hypotheses, {len(evidence_edges)} relationships")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Execute diagnostic rebalancing with Van Evera academic standards"""
        self.logger.info("START: Van Evera diagnostic rebalancing")
        
        graph_data = data['graph_data']
        
        try:
            # Analyze current distribution
            original_distribution = self._analyze_current_distribution(graph_data)
            self.logger.info(f"Original distribution: {original_distribution.percentages}")
            self.logger.info(f"Original compliance: {original_distribution.academic_compliance_score}%")
            
            # Perform rebalancing
            rebalance_result = self._perform_rebalancing(graph_data, original_distribution)
            
            # Validate results
            self._validate_rebalancing_result(rebalance_result)
            
            self.logger.info(f"COMPLETE: Rebalanced {rebalance_result.rebalanced_count} evidence items")
            self.logger.info(f"Final compliance: {rebalance_result.rebalanced_distribution.academic_compliance_score}%")
            self.logger.info(f"Compliance improvement: +{rebalance_result.compliance_improvement:.1f}%")
            
            # Convert result to dictionary with computed properties
            result_dict = asdict(rebalance_result)
            
            # Add computed properties for distributions
            if hasattr(rebalance_result.original_distribution, 'percentages'):
                result_dict['original_distribution']['percentages'] = rebalance_result.original_distribution.percentages
                result_dict['original_distribution']['academic_compliance_score'] = rebalance_result.original_distribution.academic_compliance_score
                
            if hasattr(rebalance_result.rebalanced_distribution, 'percentages'):
                result_dict['rebalanced_distribution']['percentages'] = rebalance_result.rebalanced_distribution.percentages
                result_dict['rebalanced_distribution']['academic_compliance_score'] = rebalance_result.rebalanced_distribution.academic_compliance_score
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"FAILED: Diagnostic rebalancing failed: {e}")
            raise PluginExecutionError(self.id, f"Rebalancing failed: {e}", e)
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for diagnostic rebalancing"""
        return {
            'plugin_id': self.id,
            'target_distribution': self.target_distribution,
            'diagnostic_criteria': self.diagnostic_criteria,
            'van_evera_methodology': True
        }
    
    def _analyze_current_distribution(self, graph_data: Dict) -> DiagnosticDistribution:
        """Analyze current diagnostic test distribution"""
        evidence_edges = self._find_evidence_edges(graph_data)
        
        counts = {'hoop': 0, 'smoking_gun': 0, 'doubly_decisive': 0, 'straw_in_wind': 0, 'general': 0}
        
        for edge in evidence_edges:
            diagnostic_type = edge.get('properties', {}).get('diagnostic_type', 'general')
            
            # Handle legacy naming: straw_in_the_wind -> straw_in_wind
            if diagnostic_type == 'straw_in_the_wind':
                diagnostic_type = 'straw_in_wind'
                
            if diagnostic_type in counts:
                counts[diagnostic_type] += 1
            else:
                counts['general'] += 1
        
        return DiagnosticDistribution(**counts)
    
    def _find_evidence_edges(self, graph_data: Dict) -> List[Dict]:
        """Find all edges that connect evidence to hypotheses"""
        evidence_edges = []
        node_lookup = {n['id']: n for n in graph_data['nodes']}
        
        for edge in graph_data['edges']:
            source_id = edge.get('source_id')
            target_id = edge.get('target_id')
            
            if source_id in node_lookup and target_id in node_lookup:
                source_node = node_lookup[source_id]
                target_node = node_lookup[target_id]
                
                if (source_node.get('type') == 'Evidence' and 
                    target_node.get('type') in ['Hypothesis', 'Alternative_Explanation']):
                    evidence_edges.append(edge)
        
        return evidence_edges
    
    def _perform_rebalancing(self, graph_data: Dict, original_distribution: DiagnosticDistribution) -> RebalanceResult:
        """Perform the actual diagnostic rebalancing"""
        
        # Calculate rebalancing strategy
        rebalancing_plan = self._calculate_rebalancing_plan(original_distribution)
        self.logger.info(f"Rebalancing plan: {rebalancing_plan}")
        
        # Get evidence edges to rebalance
        evidence_edges = self._find_evidence_edges(graph_data)
        candidates = self._identify_rebalancing_candidates(evidence_edges, rebalancing_plan)
        
        # Perform rebalancing
        updated_edges, stats = self._rebalance_evidence_edges(evidence_edges, candidates, graph_data)
        
        # Update graph data
        updated_graph_data = self._update_graph_data(graph_data, updated_edges)
        
        # Analyze final distribution
        rebalanced_distribution = self._analyze_current_distribution(updated_graph_data)
        
        # Calculate compliance improvement
        compliance_improvement = rebalanced_distribution.academic_compliance_score - original_distribution.academic_compliance_score
        
        # Generate academic quality assessment
        academic_assessment = self._generate_academic_assessment(
            original_distribution, rebalanced_distribution, stats
        )
        
        return RebalanceResult(
            success=True,
            original_distribution=original_distribution,
            rebalanced_distribution=rebalanced_distribution,
            rebalanced_count=stats['rebalanced'],
            enhanced_count=stats['enhanced'],
            error_count=stats['errors'],
            updated_graph_data=updated_graph_data,
            compliance_improvement=compliance_improvement,
            academic_quality_assessment=academic_assessment
        )
    
    def _calculate_rebalancing_plan(self, current_distribution: DiagnosticDistribution) -> Dict[str, Dict]:
        """Calculate which types need increase/decrease to meet Van Evera standards"""
        current_pct = current_distribution.percentages
        plan = {}
        
        for test_type, target_pct in self.target_distribution.items():
            current_pct_value = current_pct.get(test_type, 0)
            gap = target_pct - current_pct_value
            
            plan[test_type] = {
                'current_percentage': current_pct_value,
                'target_percentage': target_pct,
                'gap': gap,
                'needs_increase': gap > 0.05,  # >5% gap
                'needs_decrease': gap < -0.05, # >5% over
                'priority': abs(gap)
            }
        
        return plan
    
    def _identify_rebalancing_candidates(self, evidence_edges: List[Dict], plan: Dict) -> List[Tuple[Dict, str]]:
        """Identify which evidence edges should be reclassified and to what type"""
        candidates: List[Tuple[Dict[str, Any], str]] = []
        
        # Find types that need decrease (over-represented)
        over_represented = [t for t, p in plan.items() if p['needs_decrease']]
        # Find types that need increase (under-represented) 
        under_represented = [t for t, p in plan.items() if p['needs_increase']]
        
        if not over_represented or not under_represented:
            self.logger.info("Distribution already near Van Evera standards")
            return candidates
        
        # Sort under-represented by priority (largest gaps first)
        under_represented.sort(key=lambda t: plan[t]['priority'], reverse=True)
        
        # Select candidates from over-represented types for reclassification
        for edge in evidence_edges:
            current_type = edge.get('properties', {}).get('diagnostic_type', 'general')
            
            if current_type in over_represented or current_type == 'general':
                # Assign to highest priority under-represented type
                if under_represented:
                    target_type = under_represented[0]
                    candidates.append((edge, target_type))
        
        # Limit candidates to reasonable batch size
        max_candidates = min(len(candidates), 25)  # LLM processing limit
        return candidates[:max_candidates]
    
    def _rebalance_evidence_edges(self, evidence_edges: List[Dict], 
                                candidates: List[Tuple[Dict, str]], 
                                graph_data: Dict) -> Tuple[List[Dict], Dict]:
        """Rebalance evidence edges using LLM assessment"""
        
        updated_edges = []
        stats = {'rebalanced': 0, 'enhanced': 0, 'errors': 0}
        candidate_lookup = {id(edge): target_type for edge, target_type in candidates}
        
        for edge in evidence_edges:
            if id(edge) in candidate_lookup:
                # This edge needs rebalancing
                target_type = candidate_lookup[id(edge)]
                
                try:
                    enhanced_edge = self._enhance_evidence_edge(edge, target_type, graph_data)
                    if enhanced_edge:
                        updated_edges.append(enhanced_edge)
                        if enhanced_edge.get('properties', {}).get('diagnostic_type') != edge.get('properties', {}).get('diagnostic_type', 'general'):
                            stats['rebalanced'] += 1
                        stats['enhanced'] += 1
                    else:
                        # Keep original if enhancement fails
                        updated_edges.append(edge)
                        stats['errors'] += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to enhance edge {edge.get('source_id', '?')}->{edge.get('target_id', '?')}: {e}")
                    updated_edges.append(edge)
                    stats['errors'] += 1
            else:
                # Keep edge unchanged
                updated_edges.append(edge)
        
        return updated_edges, stats
    
    def _enhance_evidence_edge(self, edge: Dict, target_type: str, graph_data: Dict) -> Optional[Dict]:
        """Use LLM or rule-based assessment to enhance diagnostic type"""
        
        # Get evidence and hypothesis descriptions
        evidence_node = next((n for n in graph_data['nodes'] if n['id'] == edge['source_id']), None)
        hypothesis_node = next((n for n in graph_data['nodes'] if n['id'] == edge['target_id']), None)
        
        if not evidence_node or not hypothesis_node:
            return None
        
        evidence_desc = evidence_node.get('properties', {}).get('description', edge['source_id'])
        hypothesis_desc = hypothesis_node.get('properties', {}).get('description', edge['target_id'])
        
        if self.llm_query_func:
            # Use LLM assessment
            enhanced_edge = self._llm_enhance_edge(edge, evidence_desc, hypothesis_desc, target_type)
        else:
            # Use rule-based assessment
            enhanced_edge = self._rule_based_enhance_edge(edge, evidence_desc, hypothesis_desc, target_type)
        
        return enhanced_edge
    
    def _llm_enhance_edge(self, edge: Dict, evidence_desc: str, hypothesis_desc: str, target_type: str) -> Optional[Dict]:
        """Use LLM to enhance edge with Van Evera diagnostic assessment"""
        try:
            context_info = f"Hypothesis: {hypothesis_desc}\nEvidence: {evidence_desc}"
            
            # Create Van Evera-focused prompt
            van_evera_context = f"""
            Van Evera diagnostic rebalancing for hypothesis: {hypothesis_desc}
            
            Target diagnostic type: {target_type}
            Criteria for {target_type}: {self.diagnostic_criteria[target_type]['description']}
            
            Assess whether this evidence fits the {target_type} diagnostic criteria:
            - Necessary condition: {self.diagnostic_criteria[target_type]['necessary']}
            - Sufficient condition: {self.diagnostic_criteria[target_type]['sufficient']}
            """
            
            enhanced_assessment = refine_evidence_assessment_with_llm(
                evidence_description=evidence_desc,
                text_content=context_info,
                context_info=van_evera_context,
                query_llm_func=self.llm_query_func
            )
            
            if enhanced_assessment and enhanced_assessment.diagnostic_type:
                # Update edge with new diagnostic type
                updated_edge = edge.copy()
                if 'properties' not in updated_edge:
                    updated_edge['properties'] = {}
                
                # Use target type if LLM assessment aligns, otherwise use LLM assessment
                final_type = enhanced_assessment.diagnostic_type if enhanced_assessment.diagnostic_type in self.diagnostic_criteria else target_type
                
                updated_edge['properties']['diagnostic_type'] = final_type
                updated_edge['properties']['probative_value'] = enhanced_assessment.probative_value
                updated_edge['properties']['llm_enhanced'] = True
                updated_edge['properties']['target_type'] = target_type
                updated_edge['properties']['enhancement_timestamp'] = datetime.utcnow().isoformat()
                updated_edge['properties']['van_evera_rebalanced'] = True
                
                return updated_edge
                
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {e}")
            
        return None
    
    def _rule_based_enhance_edge(self, edge: Dict, evidence_desc: str, hypothesis_desc: str, target_type: str) -> Dict:
        """Use rule-based assessment to enhance edge diagnostic type"""
        updated_edge = edge.copy()
        if 'properties' not in updated_edge:
            updated_edge['properties'] = {}
        
        # Apply target type with rule-based probative value
        updated_edge['properties']['diagnostic_type'] = target_type
        
        # Assign probative value based on diagnostic type
        # Use semantic analysis to determine appropriate probative value
        from core.semantic_analysis_service import get_semantic_service
        semantic_service = get_semantic_service()
        
        # Create description for the evidence type
        type_descriptions = {
            'hoop': "Evidence that is necessary but not sufficient for the hypothesis",
            'smoking_gun': "Evidence that strongly confirms the hypothesis if found",
            'doubly_decisive': "Evidence that is both necessary and sufficient for the hypothesis",
            'straw_in_wind': "Evidence that weakly supports or opposes the hypothesis"
        }
        
        evidence_desc = updated_edge['properties'].get('description', '')
        type_desc = type_descriptions.get(target_type, "Evidence of unspecified diagnostic type")
        
        # Assess probative value based on evidence type and content
        assessment = semantic_service.assess_probative_value(
            evidence_description=evidence_desc,
            hypothesis_description=type_desc,
            context=f"Assigning probative value for {target_type} evidence type"
        )
        
        updated_edge['properties']['probative_value'] = assessment.probative_value
        updated_edge['properties']['rule_based_enhanced'] = True
        updated_edge['properties']['target_type'] = target_type
        updated_edge['properties']['enhancement_timestamp'] = datetime.utcnow().isoformat()
        updated_edge['properties']['van_evera_rebalanced'] = True
        
        return updated_edge
    
    def _update_graph_data(self, graph_data: Dict, updated_evidence_edges: List[Dict]) -> Dict:
        """Update graph data with rebalanced evidence edges"""
        updated_graph = graph_data.copy()
        
        # Replace evidence edges with updated ones
        evidence_edge_ids = set()
        for edge in updated_evidence_edges:
            edge_id = f"{edge['source_id']}->{edge['target_id']}"
            evidence_edge_ids.add(edge_id)
        
        # Keep non-evidence edges and add updated evidence edges
        non_evidence_edges = []
        for edge in graph_data['edges']:
            edge_id = f"{edge['source_id']}->{edge['target_id']}"
            if edge_id not in evidence_edge_ids:
                non_evidence_edges.append(edge)
        
        updated_graph['edges'] = non_evidence_edges + updated_evidence_edges
        
        return updated_graph
    
    def _generate_academic_assessment(self, original: DiagnosticDistribution, 
                                    rebalanced: DiagnosticDistribution, 
                                    stats: Dict) -> Dict[str, Any]:
        """Generate comprehensive academic quality assessment"""
        return {
            'methodology_compliance': {
                'van_evera_standards': True,
                'diagnostic_balance_achieved': rebalanced.academic_compliance_score > 80,
                'systematic_rebalancing_applied': True,
                'llm_enhanced_assessment': self.llm_query_func is not None
            },
            'distribution_analysis': {
                'original_compliance': original.academic_compliance_score,
                'rebalanced_compliance': rebalanced.academic_compliance_score,
                'improvement': rebalanced.academic_compliance_score - original.academic_compliance_score,
                'target_achievement': {
                    test_type: {
                        'achieved': abs(rebalanced.percentages.get(test_type, 0) - target) < 0.1,
                        'deviation': abs(rebalanced.percentages.get(test_type, 0) - target)
                    }
                    for test_type, target in self.target_distribution.items()
                }
            },
            'processing_statistics': stats,
            'academic_readiness': {
                'publication_ready': rebalanced.academic_compliance_score >= 80,
                'requires_improvement': rebalanced.academic_compliance_score < 70,
                'van_evera_compliant': rebalanced.academic_compliance_score >= 75
            },
            'recommendations': self._generate_improvement_recommendations(rebalanced, stats)
        }
    
    def _generate_improvement_recommendations(self, distribution: DiagnosticDistribution, stats: Dict) -> List[str]:
        """Generate specific recommendations for further improvement"""
        recommendations = []
        
        compliance_score = distribution.academic_compliance_score
        
        if compliance_score < 80:
            recommendations.append("Further rebalancing needed to meet Van Evera academic standards")
        
        if distribution.percentages.get('doubly_decisive', 0) < 0.10:
            recommendations.append("Increase doubly decisive tests for stronger elimination logic")
        
        if distribution.percentages.get('hoop', 0) < 0.20:
            recommendations.append("Add more hoop tests to enable hypothesis elimination")
        
        if stats['errors'] > stats['enhanced'] * 0.2:
            recommendations.append("Review error cases and improve LLM assessment quality")
        
        if not recommendations:
            recommendations.append("Diagnostic distribution meets Van Evera academic standards")
        
        return recommendations
    
    def _validate_rebalancing_result(self, result: RebalanceResult) -> None:
        """Validate that rebalancing produced reasonable results"""
        if not result.success:
            raise PluginExecutionError(self.id, "Rebalancing was not successful")
        
        if result.rebalanced_distribution.total == 0:
            raise PluginExecutionError(self.id, "No evidence relationships found after rebalancing")
        
        if result.compliance_improvement < -10:
            self.logger.warning(f"Compliance decreased by {abs(result.compliance_improvement):.1f}% - may need parameter tuning")
        
        if result.error_count > result.rebalanced_count:
            self.logger.warning(f"High error rate: {result.error_count} errors vs {result.rebalanced_count} successes")


def create_diagnostic_rebalancer_plugin(context, llm_query_func=None) -> DiagnosticRebalancerPlugin:
    """
    Factory function to create DiagnosticRebalancerPlugin with optional LLM function.
    
    Args:
        context: Plugin execution context
        llm_query_func: Optional LLM query function for enhanced assessment
        
    Returns:
        Configured DiagnosticRebalancerPlugin instance
    """
    plugin = DiagnosticRebalancerPlugin("diagnostic_rebalancer", context)
    
    if llm_query_func:
        context.set_data('llm_query_func', llm_query_func)
    
    return plugin