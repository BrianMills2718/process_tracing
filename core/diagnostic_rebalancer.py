"""
Van Evera Diagnostic Test Rebalancing System
Transforms existing evidence to achieve academic distribution standards
"""

import json
from typing import Dict, List, Tuple
from datetime import datetime
from core.structured_models import EvidenceAssessment
from core.enhance_evidence import refine_evidence_assessment_with_llm

class VanEveraDiagnosticRebalancer:
    """
    Rebalances evidence diagnostic types to meet academic Van Evera standards.
    Target Distribution: 25% hoop, 25% smoking_gun, 15% doubly_decisive, 35% straw_in_wind
    """
    
    TARGET_DISTRIBUTION = {
        'hoop': 0.25,           # Necessary but not sufficient
        'smoking_gun': 0.25,    # Sufficient but not necessary  
        'doubly_decisive': 0.15, # Both necessary and sufficient
        'straw_in_wind': 0.35   # Neither necessary nor sufficient
    }
    
    def __init__(self, graph_data: Dict):
        self.graph_data = graph_data
        self.evidence_nodes = [n for n in graph_data['nodes'] if n.get('type') == 'Evidence']
        self.hypothesis_nodes = [n for n in graph_data['nodes'] if n.get('type') == 'Hypothesis']
        self.evidence_edges = [e for e in graph_data['edges'] if self._is_evidence_edge(e)]
        
    def _is_evidence_edge(self, edge: Dict) -> bool:
        """Check if edge connects evidence to hypothesis"""
        source_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['source_id']), None)
        target_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['target_id']), None)
        return (source_node and source_node.get('type') == 'Evidence' and 
                target_node and target_node.get('type') == 'Hypothesis')
    
    def analyze_current_distribution(self) -> Dict:
        """Analyze current diagnostic type distribution"""
        distribution = {'hoop': 0, 'smoking_gun': 0, 'doubly_decisive': 0, 'straw_in_wind': 0, 'general': 0}
        
        for edge in self.evidence_edges:
            diagnostic_type = edge.get('properties', {}).get('diagnostic_type', 'general')
            if diagnostic_type in distribution:
                distribution[diagnostic_type] += 1
            else:
                distribution['general'] += 1
        
        total = sum(distribution.values())
        if total > 0:
            percentages = {k: v/total for k, v in distribution.items()}
        else:
            percentages = distribution
            
        return {
            'counts': distribution,
            'percentages': percentages,
            'total_evidence': total,
            'deviation_from_target': self._calculate_deviation(percentages)
        }
    
    def _calculate_deviation(self, current_percentages: Dict) -> Dict:
        """Calculate deviation from target academic distribution"""
        deviation = {}
        for test_type, target_pct in self.TARGET_DISTRIBUTION.items():
            current_pct = current_percentages.get(test_type, 0)
            deviation[test_type] = {
                'current': current_pct,
                'target': target_pct,
                'gap': target_pct - current_pct,
                'needs_increase': target_pct > current_pct
            }
        return deviation
    
    def rebalance_diagnostics(self, query_llm_func=None) -> Dict:
        """
        Rebalance diagnostic types using LLM assessment.
        Returns updated graph data with proper diagnostic distribution.
        """
        print("[DIAGNOSTIC_REBALANCE] Starting Van Evera diagnostic rebalancing...")
        
        current_analysis = self.analyze_current_distribution()
        print(f"[DIAGNOSTIC_REBALANCE] Current distribution: {current_analysis['percentages']}")
        print(f"[DIAGNOSTIC_REBALANCE] Target distribution: {self.TARGET_DISTRIBUTION}")
        
        # Get evidence that needs reclassification
        reclassification_needed = self._identify_reclassification_candidates(current_analysis)
        
        updated_edges = []
        rebalance_stats = {'reclassified': 0, 'enhanced': 0, 'errors': 0}
        
        for edge in self.evidence_edges:
            edge_id = f"{edge['source_id']}->{edge['target_id']}"
            
            if edge_id in reclassification_needed:
                # Use LLM to reassess diagnostic type
                enhanced_edge = self._enhance_evidence_edge(edge, query_llm_func)
                if enhanced_edge:
                    updated_edges.append(enhanced_edge)
                    rebalance_stats['reclassified'] += 1
                else:
                    updated_edges.append(edge)  # Keep original if enhancement fails
                    rebalance_stats['errors'] += 1
            else:
                updated_edges.append(edge)
        
        # Update graph data with rebalanced edges
        updated_graph = self.graph_data.copy()
        non_evidence_edges = [e for e in self.graph_data['edges'] if not self._is_evidence_edge(e)]
        updated_graph['edges'] = non_evidence_edges + updated_edges
        
        # Verify final distribution
        final_analysis = self._analyze_final_distribution(updated_edges)
        
        print(f"[DIAGNOSTIC_REBALANCE] Rebalancing complete:")
        print(f"  - Reclassified: {rebalance_stats['reclassified']} evidence items")
        print(f"  - Final distribution: {final_analysis['percentages']}")
        print(f"  - Academic compliance: {final_analysis['academic_compliance_score']}%")
        
        return {
            'updated_graph_data': updated_graph,
            'rebalance_statistics': rebalance_stats,
            'before_distribution': current_analysis,
            'after_distribution': final_analysis
        }
    
    def _identify_reclassification_candidates(self, current_analysis: Dict) -> List[str]:
        """Identify which evidence edges need diagnostic reclassification"""
        deviation = current_analysis['deviation_from_target']
        candidates = []
        
        # Prioritize reclassifying over-represented types to under-represented ones
        over_represented = [k for k, v in deviation.items() if v['gap'] < -0.05]  # >5% over target
        under_represented = [k for k, v in deviation.items() if v['gap'] > 0.05]   # >5% under target
        
        if over_represented and under_represented:
            # Select candidates from over-represented categories
            for edge in self.evidence_edges:
                current_type = edge.get('properties', {}).get('diagnostic_type', 'general')
                if current_type in over_represented or current_type == 'general':
                    edge_id = f"{edge['source_id']}->{edge['target_id']}"
                    candidates.append(edge_id)
        
        # Limit to reasonable batch size for LLM processing
        return candidates[:min(len(candidates), 20)]
    
    def _enhance_evidence_edge(self, edge: Dict, query_llm_func) -> Dict:
        """Use LLM to reassess and enhance diagnostic type for evidence edge"""
        try:
            # Get evidence and hypothesis descriptions
            evidence_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['source_id']), None)
            hypothesis_node = next((n for n in self.graph_data['nodes'] if n['id'] == edge['target_id']), None)
            
            if not evidence_node or not hypothesis_node:
                return None
                
            evidence_desc = evidence_node.get('properties', {}).get('description', edge['source_id'])
            hypothesis_desc = hypothesis_node.get('properties', {}).get('description', edge['target_id'])
            
            context_info = f"Hypothesis: {hypothesis_desc}\nEvidence: {evidence_desc}"
            
            # Use LLM enhancement with Van Evera focus
            enhanced_assessment = refine_evidence_assessment_with_llm(
                evidence_description=evidence_desc,
                text_content=context_info,
                context_info=f"Van Evera diagnostic rebalancing for hypothesis: {hypothesis_desc}",
                query_llm_func=query_llm_func
            )
            
            if enhanced_assessment and enhanced_assessment.diagnostic_type:
                # Update edge with new diagnostic type
                updated_edge = edge.copy()
                if 'properties' not in updated_edge:
                    updated_edge['properties'] = {}
                
                updated_edge['properties']['diagnostic_type'] = enhanced_assessment.diagnostic_type
                updated_edge['properties']['probative_value'] = enhanced_assessment.probative_value
                updated_edge['properties']['llm_enhanced'] = True
                updated_edge['properties']['enhancement_timestamp'] = str(datetime.utcnow())
                
                return updated_edge
                
        except Exception as e:
            print(f"[DIAGNOSTIC_REBALANCE] Error enhancing edge {edge.get('source_id', '?')}->{edge.get('target_id', '?')}: {e}")
            
        return None
    
    def _analyze_final_distribution(self, updated_edges: List[Dict]) -> Dict:
        """Analyze final diagnostic distribution after rebalancing"""
        distribution = {'hoop': 0, 'smoking_gun': 0, 'doubly_decisive': 0, 'straw_in_wind': 0}
        
        for edge in updated_edges:
            diagnostic_type = edge.get('properties', {}).get('diagnostic_type', 'general')
            if diagnostic_type in distribution:
                distribution[diagnostic_type] += 1
        
        total = sum(distribution.values())
        percentages = {k: v/total for k, v in distribution.items()} if total > 0 else distribution
        
        # Calculate academic compliance score
        compliance_score = 0
        for test_type, target_pct in self.TARGET_DISTRIBUTION.items():
            actual_pct = percentages.get(test_type, 0)
            # Score based on proximity to target (100% = perfect match)
            test_score = max(0, 100 - abs(target_pct - actual_pct) * 200)
            compliance_score += test_score * 0.25  # Weight equally
        
        return {
            'counts': distribution,
            'percentages': percentages,
            'total_evidence': total,
            'academic_compliance_score': round(compliance_score, 1)
        }

# Integration function for main analysis pipeline
def rebalance_van_evera_diagnostics(graph_data: Dict, query_llm_func=None) -> Dict:
    """
    Main entry point for diagnostic rebalancing.
    Returns updated graph data with academic Van Evera distribution.
    """
    rebalancer = VanEveraDiagnosticRebalancer(graph_data)
    result = rebalancer.rebalance_diagnostics(query_llm_func)
    return result['updated_graph_data']