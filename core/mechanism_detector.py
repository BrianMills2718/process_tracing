"""
Comparative Process Tracing - Recurring Mechanism Detection Module

Detects and analyzes recurring causal mechanisms across multiple cases
including pattern identification, mechanism classification, and cross-case validation.

Author: Claude Code Implementation  
Date: August 2025
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import logging
from itertools import combinations

from core.comparative_models import (
    MechanismPattern, MechanismType, ScopeCondition, NodeMapping,
    CrossCaseEvidence, ComparativeAnalysisError,
    validate_mechanism_pattern
)
from core.graph_alignment import GraphAligner


class MechanismDetector:
    """
    Detects recurring causal mechanisms across multiple cases.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 support_threshold: float = 0.6):
        """
        Initialize mechanism detector.
        
        Args:
            similarity_threshold: Minimum similarity for mechanism matching
            support_threshold: Minimum support across cases for pattern recognition
        """
        self.similarity_threshold = similarity_threshold
        self.support_threshold = support_threshold
        self.min_pattern_size = 2  # Minimum nodes for a pattern
        self.max_pattern_size = 8  # Maximum nodes for a pattern
        
        self.logger = logging.getLogger(__name__)
        self.graph_aligner = GraphAligner(similarity_threshold)
    
    def detect_recurring_mechanisms(self, graphs: Dict[str, nx.DiGraph], 
                                   node_mappings: List[NodeMapping]) -> List[MechanismPattern]:
        """
        Detect recurring causal mechanisms across cases.
        
        Args:
            graphs: Dictionary of case_id -> graph
            node_mappings: Cross-case node mappings
            
        Returns:
            List of identified mechanism patterns
        """
        self.logger.info(f"Detecting recurring mechanisms across {len(graphs)} cases")
        
        # Build similarity clusters from mappings
        similarity_clusters = self._build_mechanism_clusters(node_mappings)
        
        # Extract subgraph patterns from clusters
        subgraph_patterns = self._extract_subgraph_patterns(similarity_clusters, graphs)
        
        # Detect recurring patterns
        mechanism_patterns = []
        for pattern_data in subgraph_patterns:
            pattern = self._analyze_pattern_mechanism(pattern_data, graphs)
            if pattern and self._validate_pattern_significance(pattern):
                mechanism_patterns.append(pattern)
        
        # Classify mechanism types
        self._classify_mechanism_types(mechanism_patterns, graphs)
        
        # Calculate pattern metrics
        self._calculate_pattern_metrics(mechanism_patterns, graphs)
        
        self.logger.info(f"Detected {len(mechanism_patterns)} recurring mechanisms")
        return mechanism_patterns
    
    def identify_universal_mechanisms(self, graphs: Dict[str, nx.DiGraph], 
                                    mechanisms: List[MechanismPattern]) -> List[MechanismPattern]:
        """
        Identify mechanisms that appear across all or most cases.
        
        Args:
            graphs: Dictionary of case_id -> graph
            mechanisms: List of mechanism patterns
            
        Returns:
            List of universal mechanisms
        """
        universal_mechanisms = []
        total_cases = len(graphs)
        universal_threshold = max(2, int(total_cases * 0.8))  # 80% of cases
        
        for mechanism in mechanisms:
            if len(mechanism.participating_cases) >= universal_threshold:
                mechanism.mechanism_type = MechanismType.UNIVERSAL
                universal_mechanisms.append(mechanism)
        
        self.logger.info(f"Identified {len(universal_mechanisms)} universal mechanisms")
        return universal_mechanisms
    
    def identify_conditional_mechanisms(self, graphs: Dict[str, nx.DiGraph], 
                                      mechanisms: List[MechanismPattern]) -> List[MechanismPattern]:
        """
        Identify mechanisms that appear under specific conditions.
        
        Args:
            graphs: Dictionary of case_id -> graph
            mechanisms: List of mechanism patterns
            
        Returns:
            List of conditional mechanisms
        """
        conditional_mechanisms = []
        
        for mechanism in mechanisms:
            if mechanism.mechanism_type == MechanismType.CONDITIONAL:
                # Analyze scope conditions
                scope_conditions = self._analyze_scope_conditions(mechanism, graphs)
                mechanism.scope_conditions = scope_conditions
                conditional_mechanisms.append(mechanism)
        
        self.logger.info(f"Identified {len(conditional_mechanisms)} conditional mechanisms")
        return conditional_mechanisms
    
    def detect_mechanism_variations(self, mechanisms: List[MechanismPattern], 
                                  graphs: Dict[str, nx.DiGraph]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect variations in how mechanisms manifest across cases.
        
        Args:
            mechanisms: List of mechanism patterns
            graphs: Dictionary of case_id -> graph
            
        Returns:
            Dictionary of mechanism_id -> variations
        """
        mechanism_variations = {}
        
        for mechanism in mechanisms:
            variations = []
            
            # Analyze variations across participating cases
            for case_id in mechanism.participating_cases:
                graph = graphs[case_id]
                case_variation = self._analyze_case_variation(mechanism, graph, case_id)
                if case_variation:
                    variations.append(case_variation)
            
            if variations:
                mechanism_variations[mechanism.pattern_id] = variations
        
        self.logger.info(f"Detected variations for {len(mechanism_variations)} mechanisms")
        return mechanism_variations
    
    def assess_mechanism_strength(self, mechanism: MechanismPattern, 
                                graphs: Dict[str, nx.DiGraph]) -> float:
        """
        Assess the overall strength of a mechanism pattern.
        
        Args:
            mechanism: Mechanism pattern to assess
            graphs: Dictionary of case_id -> graph
            
        Returns:
            Mechanism strength score (0.0-1.0)
        """
        strength_factors = []
        
        # Frequency across cases
        frequency_score = len(mechanism.participating_cases) / len(graphs)
        strength_factors.append(frequency_score)
        
        # Consistency score
        strength_factors.append(mechanism.consistency_score)
        
        # Pattern completeness (how many core elements are preserved)
        completeness_scores = []
        for case_id in mechanism.participating_cases:
            completeness = self._calculate_pattern_completeness(mechanism, graphs[case_id])
            completeness_scores.append(completeness)
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        strength_factors.append(avg_completeness)
        
        # Evidence support strength
        evidence_scores = []
        for case_id, evidence_list in mechanism.supporting_evidence.items():
            # Simple heuristic: more evidence = stronger support
            evidence_score = min(1.0, len(evidence_list) / 3.0)  # Normalize around 3 pieces of evidence
            evidence_scores.append(evidence_score)
        
        avg_evidence = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.5
        strength_factors.append(avg_evidence)
        
        # Calculate weighted strength
        weights = [0.3, 0.3, 0.25, 0.15]  # frequency, consistency, completeness, evidence
        overall_strength = sum(w * s for w, s in zip(weights, strength_factors))
        
        return min(1.0, max(0.0, overall_strength))
    
    def _build_mechanism_clusters(self, node_mappings: List[NodeMapping]) -> List[Dict[str, Any]]:
        """
        Build clusters of similar nodes that could form mechanisms.
        
        Args:
            node_mappings: Cross-case node mappings
            
        Returns:
            List of node clusters
        """
        # Group mappings by similarity
        high_similarity_mappings = [
            mapping for mapping in node_mappings 
            if mapping.overall_similarity >= self.similarity_threshold
        ]
        
        # Build clusters using transitive closure
        clusters = []
        processed_mappings = set()
        
        for mapping in high_similarity_mappings:
            if mapping.mapping_id in processed_mappings:
                continue
                
            # Start new cluster
            cluster_nodes = {
                mapping.source_case: {mapping.source_node},
                mapping.target_case: {mapping.target_node}
            }
            cluster_mappings = [mapping]
            processed_mappings.add(mapping.mapping_id)
            
            # Find transitively connected mappings
            changed = True
            while changed:
                changed = False
                for other_mapping in high_similarity_mappings:
                    if other_mapping.mapping_id in processed_mappings:
                        continue
                    
                    # Check if this mapping connects to existing cluster
                    if (other_mapping.source_case in cluster_nodes and 
                        other_mapping.source_node in cluster_nodes[other_mapping.source_case]) or \
                       (other_mapping.target_case in cluster_nodes and 
                        other_mapping.target_node in cluster_nodes[other_mapping.target_case]):
                        
                        # Add to cluster
                        if other_mapping.source_case not in cluster_nodes:
                            cluster_nodes[other_mapping.source_case] = set()
                        if other_mapping.target_case not in cluster_nodes:
                            cluster_nodes[other_mapping.target_case] = set()
                        
                        cluster_nodes[other_mapping.source_case].add(other_mapping.source_node)
                        cluster_nodes[other_mapping.target_case].add(other_mapping.target_node)
                        cluster_mappings.append(other_mapping)
                        processed_mappings.add(other_mapping.mapping_id)
                        changed = True
            
            # Create cluster record
            if len(cluster_nodes) >= 2:  # At least 2 cases
                cluster = {
                    'cluster_id': f"cluster_{len(clusters)}",
                    'cases': set(cluster_nodes.keys()),
                    'nodes_per_case': cluster_nodes,
                    'mappings': cluster_mappings,
                    'avg_similarity': sum(m.overall_similarity for m in cluster_mappings) / len(cluster_mappings)
                }
                clusters.append(cluster)
        
        return clusters
    
    def _extract_subgraph_patterns(self, clusters: List[Dict[str, Any]], 
                                 graphs: Dict[str, nx.DiGraph]) -> List[Dict[str, Any]]:
        """
        Extract subgraph patterns from node clusters.
        
        Args:
            clusters: Node clusters
            graphs: Case graphs
            
        Returns:
            List of subgraph patterns
        """
        patterns = []
        
        for cluster in clusters:
            # Extract subgraphs around clustered nodes for each case
            case_subgraphs = {}
            
            for case_id in cluster['cases']:
                graph = graphs[case_id]
                cluster_nodes = cluster['nodes_per_case'][case_id]
                
                # Expand to include immediate neighbors (1-hop)
                expanded_nodes = set(cluster_nodes)
                for node in cluster_nodes:
                    if node in graph:
                        expanded_nodes.update(graph.neighbors(node))
                
                # Extract subgraph
                subgraph = graph.subgraph(expanded_nodes).copy()
                case_subgraphs[case_id] = subgraph
            
            # Analyze common structure
            pattern = self._find_common_structure(case_subgraphs, cluster)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _find_common_structure(self, case_subgraphs: Dict[str, nx.DiGraph], 
                             cluster: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find common structural patterns across case subgraphs.
        
        Args:
            case_subgraphs: Subgraphs for each case
            cluster: Node cluster information
            
        Returns:
            Common pattern structure or None
        """
        if len(case_subgraphs) < 2:
            return None
        
        # Find common node types and edge patterns
        case_ids = list(case_subgraphs.keys())
        
        # Analyze node types
        node_type_patterns = defaultdict(list)
        for case_id, subgraph in case_subgraphs.items():
            case_types = []
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                node_type = node_data.get('type', 'Unknown')
                case_types.append(node_type)
            node_type_patterns[case_id] = Counter(case_types)
        
        # Find common node types
        common_types = set(node_type_patterns[case_ids[0]].keys())
        for case_id in case_ids[1:]:
            common_types &= set(node_type_patterns[case_id].keys())
        
        if not common_types:
            return None
        
        # Analyze edge patterns
        edge_type_patterns = defaultdict(list)
        for case_id, subgraph in case_subgraphs.items():
            case_edge_types = []
            for u, v, edge_data in subgraph.edges(data=True):
                edge_type = edge_data.get('type', 'unknown')
                case_edge_types.append(edge_type)
            edge_type_patterns[case_id] = Counter(case_edge_types)
        
        # Find common edge types
        common_edge_types = set(edge_type_patterns[case_ids[0]].keys()) if case_ids else set()
        for case_id in case_ids[1:]:
            common_edge_types &= set(edge_type_patterns[case_id].keys())
        
        # Create pattern structure
        pattern = {
            'pattern_id': cluster['cluster_id'],
            'participating_cases': list(cluster['cases']),
            'case_subgraphs': case_subgraphs,
            'common_node_types': list(common_types),
            'common_edge_types': list(common_edge_types),
            'node_type_frequencies': dict(node_type_patterns),
            'edge_type_frequencies': dict(edge_type_patterns),
            'cluster_info': cluster
        }
        
        return pattern
    
    def _analyze_pattern_mechanism(self, pattern_data: Dict[str, Any], 
                                 graphs: Dict[str, nx.DiGraph]) -> Optional[MechanismPattern]:
        """
        Analyze pattern to extract mechanism information.
        
        Args:
            pattern_data: Pattern structure data
            graphs: Case graphs
            
        Returns:
            MechanismPattern or None
        """
        # Generate mechanism description
        common_types = pattern_data['common_node_types']
        common_edges = pattern_data['common_edge_types']
        
        if not common_types:
            return None
        
        description = f"Recurring pattern involving {', '.join(common_types)}"
        if common_edges:
            description += f" connected by {', '.join(common_edges)} relationships"
        
        # Extract core and optional elements
        core_nodes = []
        optional_nodes = []
        core_edges = []
        optional_edges = []
        
        # Analyze frequency of elements across cases
        all_node_types = set()
        all_edge_types = set()
        
        for case_freqs in pattern_data['node_type_frequencies'].values():
            all_node_types.update(case_freqs.keys())
        for case_freqs in pattern_data['edge_type_frequencies'].values():
            all_edge_types.update(case_freqs.keys())
        
        # Classify as core (appears in most cases) or optional
        total_cases = len(pattern_data['participating_cases'])
        core_threshold = max(1, int(total_cases * 0.7))  # 70% of cases
        
        for node_type in all_node_types:
            case_count = sum(1 for freqs in pattern_data['node_type_frequencies'].values() 
                           if node_type in freqs and freqs[node_type] > 0)
            if case_count >= core_threshold:
                core_nodes.append(node_type)
            else:
                optional_nodes.append(node_type)
        
        for edge_type in all_edge_types:
            case_count = sum(1 for freqs in pattern_data['edge_type_frequencies'].values() 
                           if edge_type in freqs and freqs[edge_type] > 0)
            if case_count >= core_threshold:
                core_edges.append(edge_type)
            else:
                optional_edges.append(edge_type)
        
        # Calculate consistency score
        consistency_scores = []
        for case_id in pattern_data['participating_cases']:
            case_score = self._calculate_case_consistency(pattern_data, case_id)
            consistency_scores.append(case_score)
        
        consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        # Create mechanism pattern
        mechanism = MechanismPattern(
            pattern_id=pattern_data['pattern_id'],
            pattern_name=f"Pattern {pattern_data['pattern_id']}",
            description=description,
            mechanism_type=MechanismType.CONDITIONAL,  # Will be refined later
            scope_conditions=[],
            participating_cases=pattern_data['participating_cases'],
            core_nodes=core_nodes,
            optional_nodes=optional_nodes,
            core_edges=[(src, tgt) for src in core_nodes for tgt in core_nodes],  # Simplified
            optional_edges=[(src, tgt) for src in optional_nodes for tgt in core_nodes + optional_nodes],
            consistency_score=consistency_score,
            pattern_strength=0.0,  # Will be calculated later
            generalizability=0.0   # Will be calculated later
        )
        
        return mechanism
    
    def _validate_pattern_significance(self, pattern: MechanismPattern) -> bool:
        """
        Validate that pattern is significant enough to be considered.
        
        Args:
            pattern: Mechanism pattern to validate
            
        Returns:
            True if pattern is significant
        """
        # Check minimum participation
        if len(pattern.participating_cases) < 2:
            return False
        
        # Check minimum consistency
        if pattern.consistency_score < 0.5:
            return False
        
        # Check minimum pattern size
        if len(pattern.core_nodes) < self.min_pattern_size:
            return False
        
        # Check support threshold
        # This is a simplified check - could be enhanced with more sophisticated measures
        return True
    
    def _classify_mechanism_types(self, mechanisms: List[MechanismPattern], 
                                graphs: Dict[str, nx.DiGraph]):
        """
        Classify mechanisms by type (universal, conditional, etc.).
        
        Args:
            mechanisms: List of mechanism patterns
            graphs: Case graphs
        """
        total_cases = len(graphs)
        
        for mechanism in mechanisms:
            case_count = len(mechanism.participating_cases)
            
            if case_count >= total_cases * 0.9:  # 90%+ of cases
                mechanism.mechanism_type = MechanismType.UNIVERSAL
            elif case_count >= total_cases * 0.6:  # 60%+ of cases
                mechanism.mechanism_type = MechanismType.CONDITIONAL
            elif case_count <= 2:  # Only in 1-2 cases
                mechanism.mechanism_type = MechanismType.CASE_SPECIFIC
            else:
                mechanism.mechanism_type = MechanismType.VARIANT
    
    def _calculate_pattern_metrics(self, mechanisms: List[MechanismPattern], 
                                 graphs: Dict[str, nx.DiGraph]):
        """
        Calculate metrics for mechanism patterns.
        
        Args:
            mechanisms: List of mechanism patterns
            graphs: Case graphs
        """
        for mechanism in mechanisms:
            # Calculate pattern strength
            mechanism.pattern_strength = self.assess_mechanism_strength(mechanism, graphs)
            
            # Calculate generalizability
            mechanism.generalizability = self._calculate_generalizability(mechanism, graphs)
            
            # Calculate case frequencies
            mechanism.case_frequencies = {}
            for case_id in mechanism.participating_cases:
                frequency = self._calculate_case_frequency(mechanism, graphs[case_id])
                mechanism.case_frequencies[case_id] = frequency
    
    def _analyze_scope_conditions(self, mechanism: MechanismPattern, 
                                graphs: Dict[str, nx.DiGraph]) -> List[ScopeCondition]:
        """
        Analyze scope conditions for conditional mechanisms.
        
        Args:
            mechanism: Mechanism pattern
            graphs: Case graphs
            
        Returns:
            List of identified scope conditions
        """
        scope_conditions = []
        
        # Analyze context factors across participating vs non-participating cases
        participating = set(mechanism.participating_cases)
        non_participating = set(graphs.keys()) - participating
        
        if len(non_participating) == 0:
            return [ScopeCondition.CONTEXT_DEPENDENT]  # Universal mechanism
        
        # Simple heuristic analysis
        # In practice, this would involve more sophisticated analysis of case metadata
        # and contextual factors
        
        # Check for temporal patterns
        if self._has_temporal_dependency(mechanism, graphs):
            scope_conditions.append(ScopeCondition.TIME_DEPENDENT)
        
        # Check for resource dependency
        if self._has_resource_dependency(mechanism, graphs):
            scope_conditions.append(ScopeCondition.RESOURCE_DEPENDENT)
        
        # Default to context dependent if no specific patterns found
        if not scope_conditions:
            scope_conditions.append(ScopeCondition.CONTEXT_DEPENDENT)
        
        return scope_conditions
    
    def _analyze_case_variation(self, mechanism: MechanismPattern, 
                              graph: nx.DiGraph, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze how mechanism varies in a specific case.
        
        Args:
            mechanism: Mechanism pattern
            graph: Case graph
            case_id: Case identifier
            
        Returns:
            Variation analysis or None
        """
        # Find nodes in this case that match the mechanism pattern
        matching_nodes = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            if node_type in mechanism.core_nodes or node_type in mechanism.optional_nodes:
                matching_nodes.append(node)
        
        if not matching_nodes:
            return None
        
        # Analyze structural variation
        subgraph = graph.subgraph(matching_nodes)
        
        variation = {
            'case_id': case_id,
            'matching_nodes': len(matching_nodes),
            'core_nodes_present': len([n for n in matching_nodes 
                                     if graph.nodes[n].get('type') in mechanism.core_nodes]),
            'optional_nodes_present': len([n for n in matching_nodes 
                                         if graph.nodes[n].get('type') in mechanism.optional_nodes]),
            'edges_present': subgraph.number_of_edges(),
            'completeness_score': self._calculate_pattern_completeness(mechanism, graph)
        }
        
        return variation
    
    def _calculate_pattern_completeness(self, mechanism: MechanismPattern, 
                                      graph: nx.DiGraph) -> float:
        """
        Calculate how complete the pattern is in a specific graph.
        
        Args:
            mechanism: Mechanism pattern
            graph: Case graph
            
        Returns:
            Completeness score (0.0-1.0)
        """
        # Count present core elements
        present_core_nodes = 0
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if node_type in mechanism.core_nodes:
                present_core_nodes += 1
        
        # Simple completeness calculation
        if not mechanism.core_nodes:
            return 1.0
        
        core_completeness = min(1.0, present_core_nodes / len(mechanism.core_nodes))
        
        # Could be enhanced with edge completeness analysis
        return core_completeness
    
    def _calculate_case_consistency(self, pattern_data: Dict[str, Any], case_id: str) -> float:
        """
        Calculate consistency score for a case within a pattern.
        
        Args:
            pattern_data: Pattern structure data
            case_id: Case identifier
            
        Returns:
            Consistency score (0.0-1.0)
        """
        # Compare this case's pattern to the common pattern
        case_node_freqs = pattern_data['node_type_frequencies'].get(case_id, {})
        case_edge_freqs = pattern_data['edge_type_frequencies'].get(case_id, {})
        
        common_nodes = pattern_data['common_node_types']
        common_edges = pattern_data['common_edge_types']
        
        # Calculate node type consistency
        node_consistency = 0.0
        if common_nodes:
            present_common_nodes = sum(1 for node_type in common_nodes 
                                     if case_node_freqs.get(node_type, 0) > 0)
            node_consistency = present_common_nodes / len(common_nodes)
        
        # Calculate edge type consistency
        edge_consistency = 0.0
        if common_edges:
            present_common_edges = sum(1 for edge_type in common_edges 
                                     if case_edge_freqs.get(edge_type, 0) > 0)
            edge_consistency = present_common_edges / len(common_edges)
        
        # Weighted average
        if common_nodes and common_edges:
            consistency = 0.7 * node_consistency + 0.3 * edge_consistency
        elif common_nodes:
            consistency = node_consistency
        else:
            consistency = edge_consistency
        
        return consistency
    
    def _calculate_generalizability(self, mechanism: MechanismPattern, 
                                  graphs: Dict[str, nx.DiGraph]) -> float:
        """
        Calculate generalizability score for mechanism.
        
        Args:
            mechanism: Mechanism pattern
            graphs: Case graphs
            
        Returns:
            Generalizability score (0.0-1.0)
        """
        # Simple generalizability based on case coverage and consistency
        case_coverage = len(mechanism.participating_cases) / len(graphs)
        consistency = mechanism.consistency_score
        
        # Weight coverage and consistency
        generalizability = 0.6 * case_coverage + 0.4 * consistency
        
        return min(1.0, max(0.0, generalizability))
    
    def _calculate_case_frequency(self, mechanism: MechanismPattern, graph: nx.DiGraph) -> float:
        """
        Calculate frequency of mechanism appearance in a case.
        
        Args:
            mechanism: Mechanism pattern
            graph: Case graph
            
        Returns:
            Frequency score (0.0-1.0)
        """
        # Count instances of pattern elements
        core_node_instances = 0
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if node_type in mechanism.core_nodes:
                core_node_instances += 1
        
        # Simple frequency calculation
        # Could be enhanced with more sophisticated pattern counting
        if not mechanism.core_nodes:
            return 1.0
        
        # Normalize by expected instances (assumes each core node type appears once)
        expected_instances = len(mechanism.core_nodes)
        frequency = min(1.0, core_node_instances / expected_instances)
        
        return frequency
    
    def _has_temporal_dependency(self, mechanism: MechanismPattern, 
                               graphs: Dict[str, nx.DiGraph]) -> bool:
        """
        Check if mechanism has temporal dependency.
        
        Args:
            mechanism: Mechanism pattern
            graphs: Case graphs
            
        Returns:
            True if temporal dependency detected
        """
        # Simple heuristic: check for temporal attributes in mechanism nodes
        for case_id in mechanism.participating_cases:
            graph = graphs[case_id]
            for node in graph.nodes():
                node_data = graph.nodes[node]
                # Check for temporal attributes using semantic understanding
                from core.semantic_analysis_service import get_semantic_service
                semantic_service = get_semantic_service()
                
                # Check if node has temporal characteristics
                node_desc = str(node_data)
                assessment = semantic_service.assess_probative_value(
                    evidence_description=node_desc,
                    hypothesis_description="Node has temporal characteristics (time, sequence, order)",
                    context="Detecting temporal dependencies in mechanisms"
                )
                if assessment.confidence_score > 0.7:
                    return True
        return False
    
    def _has_resource_dependency(self, mechanism: MechanismPattern, 
                               graphs: Dict[str, nx.DiGraph]) -> bool:
        """
        Check if mechanism has resource dependency.
        
        Args:
            mechanism: Mechanism pattern
            graphs: Case graphs
            
        Returns:
            True if resource dependency detected
        """
        # Simple heuristic: check for resource-related node types or attributes
        resource_indicators = ['resource', 'budget', 'funding', 'capacity', 'capability']
        
        for case_id in mechanism.participating_cases:
            graph = graphs[case_id]
            for node in graph.nodes():
                node_data = graph.nodes[node]
                node_type = node_data.get('type', '').lower()
                description = node_data.get('description', '').lower()
                
                # Use semantic analysis to detect resource dependencies
                from core.semantic_analysis_service import get_semantic_service
                semantic_service = get_semantic_service()
                
                assessment = semantic_service.assess_probative_value(
                    evidence_description=f"{node_type}: {description}",
                    hypothesis_description="Node represents resource dependency (budget, funding, capacity)",
                    context="Detecting resource dependencies in mechanisms"
                )
                if assessment.confidence_score > 0.65:
                    return True
        return False


def test_mechanism_detector():
    """Test function for mechanism detector"""
    # Create test graphs with recurring patterns
    graph1 = nx.DiGraph()
    graph1.add_node("crisis1", type="Event", description="Economic crisis")
    graph1.add_node("policy1", type="Event", description="Policy response")
    graph1.add_node("outcome1", type="Event", description="Recovery")
    graph1.add_edge("crisis1", "policy1", type="causes")
    graph1.add_edge("policy1", "outcome1", type="causes")
    
    graph2 = nx.DiGraph()
    graph2.add_node("shock2", type="Event", description="Financial shock")
    graph2.add_node("response2", type="Event", description="Government response")
    graph2.add_node("result2", type="Event", description="Stabilization")
    graph2.add_edge("shock2", "response2", type="causes")
    graph2.add_edge("response2", "result2", type="causes")
    
    graphs = {"case1": graph1, "case2": graph2}
    
    # Create mock node mappings
    from core.comparative_models import NodeMapping
    mappings = [
        NodeMapping(
            mapping_id="m1",
            source_case="case1",
            target_case="case2",
            source_node="crisis1",
            target_node="shock2",
            semantic_similarity=0.8,
            structural_similarity=0.9,
            overall_similarity=0.85
        ),
        NodeMapping(
            mapping_id="m2",
            source_case="case1",
            target_case="case2",
            source_node="policy1",
            target_node="response2",
            semantic_similarity=0.7,
            structural_similarity=0.8,
            overall_similarity=0.75
        )
    ]
    
    # Test mechanism detection
    detector = MechanismDetector()
    mechanisms = detector.detect_recurring_mechanisms(graphs, mappings)
    
    print(f"Detected {len(mechanisms)} recurring mechanisms")
    for mechanism in mechanisms:
        print(f"  Pattern: {mechanism.pattern_name}")
        print(f"  Type: {mechanism.mechanism_type.value}")
        print(f"  Cases: {mechanism.participating_cases}")
        print(f"  Core nodes: {mechanism.core_nodes}")
        print(f"  Strength: {mechanism.pattern_strength:.2f}")
        print(f"  Consistency: {mechanism.consistency_score:.2f}")
        print()


if __name__ == "__main__":
    test_mechanism_detector()