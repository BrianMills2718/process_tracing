"""
Comparative Process Tracing - Mechanism Pattern Detection Module

Identifies recurring causal mechanisms across cases and analyzes
their variations, scope conditions, and generalizability.

Author: Claude Code Implementation  
Date: August 2025
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import logging
from itertools import combinations

from core.comparative_models import (
    MechanismPattern, NodeMapping, ScopeCondition, MechanismType,
    CrossCaseEvidence, ComparativeAnalysisError, CaseMetadata
)
from core.graph_alignment import GraphAligner


class MechanismPatternDetector:
    """
    Detects recurring causal mechanisms across multiple cases.
    """
    
    def __init__(self, min_pattern_frequency: float = 0.5, 
                 min_pattern_strength: float = 0.6):
        """
        Initialize mechanism pattern detector.
        
        Args:
            min_pattern_frequency: Minimum frequency for pattern recognition (0.0-1.0)
            min_pattern_strength: Minimum strength for valid patterns (0.0-1.0)
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.min_pattern_strength = min_pattern_strength
        self.graph_aligner = GraphAligner()
        
        self.logger = logging.getLogger(__name__)
    
    def detect_patterns(self, graphs: Dict[str, nx.DiGraph], 
                       case_metadata: Dict[str, CaseMetadata],
                       node_mappings: Dict[Tuple[str, str], List[NodeMapping]]) -> List[MechanismPattern]:
        """
        Detect recurring mechanism patterns across cases.
        
        Args:
            graphs: Dictionary of case_id -> graph
            case_metadata: Dictionary of case_id -> metadata
            node_mappings: Pairwise node mappings between cases
            
        Returns:
            List of detected mechanism patterns
        """
        patterns = []
        
        # Find pathway patterns
        pathway_patterns = self._detect_pathway_patterns(graphs, node_mappings)
        patterns.extend(pathway_patterns)
        
        # Find structural patterns
        structural_patterns = self._detect_structural_patterns(graphs, case_metadata)
        patterns.extend(structural_patterns)
        
        # Find evidence patterns
        evidence_patterns = self._detect_evidence_patterns(graphs, case_metadata)
        patterns.extend(evidence_patterns)
        
        # Validate and filter patterns
        validated_patterns = self._validate_patterns(patterns, graphs, case_metadata)
        
        self.logger.info(f"Detected {len(validated_patterns)} validated mechanism patterns")
        return validated_patterns
    
    def analyze_pattern_variations(self, pattern: MechanismPattern, 
                                 graphs: Dict[str, nx.DiGraph]) -> Dict[str, Any]:
        """
        Analyze variations of a pattern across cases.
        
        Args:
            pattern: Mechanism pattern to analyze
            graphs: Dictionary of case_id -> graph
            
        Returns:
            Dictionary of variation analysis results
        """
        variations = {}
        
        for case_id in pattern.participating_cases:
            if case_id in graphs:
                case_variation = self._analyze_case_variation(pattern, graphs[case_id], case_id)
                variations[case_id] = case_variation
        
        # Summarize variation patterns
        variation_summary = self._summarize_variations(variations)
        
        return {
            'case_variations': variations,
            'variation_summary': variation_summary,
            'consistency_score': self._calculate_consistency_score(variations),
            'variant_types': self._identify_variant_types(variations)
        }
    
    def identify_scope_conditions(self, pattern: MechanismPattern,
                                case_metadata: Dict[str, CaseMetadata]) -> List[ScopeCondition]:
        """
        Identify scope conditions for when pattern applies.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata dictionary
            
        Returns:
            List of identified scope conditions
        """
        scope_conditions = []
        
        # Analyze contextual factors
        contextual_conditions = self._analyze_contextual_conditions(pattern, case_metadata)
        scope_conditions.extend(contextual_conditions)
        
        # Analyze temporal conditions
        temporal_conditions = self._analyze_temporal_conditions(pattern, case_metadata)
        scope_conditions.extend(temporal_conditions)
        
        # Analyze actor conditions
        actor_conditions = self._analyze_actor_conditions(pattern, case_metadata)
        scope_conditions.extend(actor_conditions)
        
        # Analyze resource conditions
        resource_conditions = self._analyze_resource_conditions(pattern, case_metadata)
        scope_conditions.extend(resource_conditions)
        
        return list(set(scope_conditions))  # Remove duplicates
    
    def calculate_pattern_generalizability(self, pattern: MechanismPattern,
                                        case_metadata: Dict[str, CaseMetadata]) -> float:
        """
        Calculate how generalizable a pattern is across different contexts.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata dictionary
            
        Returns:
            Generalizability score (0.0-1.0)
        """
        if not pattern.participating_cases:
            return 0.0
        
        # Factor 1: Number of cases
        case_diversity = len(pattern.participating_cases) / len(case_metadata)
        
        # Factor 2: Context diversity
        context_diversity = self._calculate_context_diversity(pattern, case_metadata)
        
        # Factor 3: Pattern consistency
        consistency = pattern.consistency_score
        
        # Factor 4: Evidence strength
        evidence_strength = self._calculate_evidence_strength(pattern)
        
        # Weighted combination
        generalizability = (
            0.3 * case_diversity +
            0.3 * context_diversity +
            0.3 * consistency +
            0.1 * evidence_strength
        )
        
        return min(1.0, max(0.0, generalizability))
    
    def _detect_pathway_patterns(self, graphs: Dict[str, nx.DiGraph],
                               node_mappings: Dict[Tuple[str, str], List[NodeMapping]]) -> List[MechanismPattern]:
        """
        Detect recurring causal pathway patterns.
        
        Args:
            graphs: Dictionary of case_id -> graph
            node_mappings: Pairwise node mappings
            
        Returns:
            List of pathway patterns
        """
        patterns = []
        
        # Extract all pathways from each case
        case_pathways = {}
        for case_id, graph in graphs.items():
            pathways = self._extract_pathways(graph, case_id)
            case_pathways[case_id] = pathways
        
        # Find similar pathways across cases using mappings
        similar_pathway_groups = self._group_similar_pathways(case_pathways, node_mappings)
        
        # Create patterns from groups
        for group_id, pathway_group in enumerate(similar_pathway_groups):
            if len(pathway_group['cases']) >= 2:  # At least 2 cases
                pattern = self._create_pathway_pattern(group_id, pathway_group, graphs)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_structural_patterns(self, graphs: Dict[str, nx.DiGraph],
                                  case_metadata: Dict[str, CaseMetadata]) -> List[MechanismPattern]:
        """
        Detect structural patterns (convergence, divergence, etc.).
        
        Args:
            graphs: Dictionary of case_id -> graph
            case_metadata: Case metadata dictionary
            
        Returns:
            List of structural patterns
        """
        patterns = []
        
        # Detect convergence patterns
        convergence_patterns = self._detect_convergence_patterns(graphs)
        patterns.extend(convergence_patterns)
        
        # Detect divergence patterns
        divergence_patterns = self._detect_divergence_patterns(graphs)
        patterns.extend(divergence_patterns)
        
        # Detect hub patterns
        hub_patterns = self._detect_hub_patterns(graphs)
        patterns.extend(hub_patterns)
        
        return patterns
    
    def _detect_evidence_patterns(self, graphs: Dict[str, nx.DiGraph],
                                case_metadata: Dict[str, CaseMetadata]) -> List[MechanismPattern]:
        """
        Detect patterns based on evidence types and strengths.
        
        Args:
            graphs: Dictionary of case_id -> graph
            case_metadata: Case metadata dictionary
            
        Returns:
            List of evidence-based patterns
        """
        patterns = []
        
        # Group by Van Evera evidence types
        evidence_groups = defaultdict(list)
        
        for case_id, graph in graphs.items():
            for node_id, node_data in graph.nodes(data=True):
                van_evera_type = node_data.get('van_evera_type')
                if van_evera_type:
                    evidence_groups[van_evera_type].append((case_id, node_id, node_data))
        
        # Create patterns for evidence types that appear in multiple cases
        for evidence_type, evidence_list in evidence_groups.items():
            cases = set(item[0] for item in evidence_list)
            if len(cases) >= 2:
                pattern = self._create_evidence_pattern(evidence_type, evidence_list, graphs)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_pathways(self, graph: nx.DiGraph, case_id: str) -> List[Dict[str, Any]]:
        """
        Extract causal pathways from a graph.
        
        Args:
            graph: Graph to analyze
            case_id: Case identifier
            
        Returns:
            List of pathway dictionaries
        """
        pathways = []
        
        # Find all simple paths up to length 5
        nodes = list(graph.nodes())
        for source in nodes:
            for target in nodes:
                if source != target:
                    try:
                        simple_paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
                        for path in simple_paths:
                            if len(path) >= 2:  # At least 2 nodes
                                pathway = {
                                    'case_id': case_id,
                                    'path': path,
                                    'length': len(path),
                                    'source': source,
                                    'target': target,
                                    'edge_types': self._get_pathway_edge_types(graph, path)
                                }
                                pathways.append(pathway)
                    except nx.NetworkXNoPath:
                        continue
        
        return pathways
    
    def _get_pathway_edge_types(self, graph: nx.DiGraph, path: List[str]) -> List[str]:
        """
        Get edge types for a pathway.
        
        Args:
            graph: Graph containing the path
            path: List of nodes in path
            
        Returns:
            List of edge types
        """
        edge_types = []
        for i in range(len(path) - 1):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            edge_type = edge_data.get('type', 'unknown') if edge_data else 'unknown'
            edge_types.append(edge_type)
        return edge_types
    
    def _group_similar_pathways(self, case_pathways: Dict[str, List[Dict[str, Any]]],
                              node_mappings: Dict[Tuple[str, str], List[NodeMapping]]) -> List[Dict[str, Any]]:
        """
        Group similar pathways across cases.
        
        Args:
            case_pathways: Pathways by case
            node_mappings: Node mappings between cases
            
        Returns:
            List of pathway groups
        """
        pathway_groups = []
        processed_pathways = set()
        
        # Create mapping lookup for efficiency
        mapping_lookup = defaultdict(dict)
        for (case1, case2), mappings in node_mappings.items():
            for mapping in mappings:
                mapping_lookup[case1][mapping.source_node] = (case2, mapping.target_node, mapping.overall_similarity)
                mapping_lookup[case2][mapping.target_node] = (case1, mapping.source_node, mapping.overall_similarity)
        
        for case_id, pathways in case_pathways.items():
            for pathway_idx, pathway in enumerate(pathways):
                pathway_key = f"{case_id}_{pathway_idx}"
                
                if pathway_key in processed_pathways:
                    continue
                
                # Find similar pathways in other cases
                similar_pathways = [pathway]
                similar_case_ids = {case_id}
                
                for other_case_id, other_pathways in case_pathways.items():
                    if other_case_id == case_id:
                        continue
                    
                    for other_idx, other_pathway in enumerate(other_pathways):
                        other_key = f"{other_case_id}_{other_idx}"
                        
                        if other_key in processed_pathways:
                            continue
                        
                        # Check if pathways are similar
                        similarity = self._calculate_pathway_similarity(
                            pathway, other_pathway, mapping_lookup
                        )
                        
                        if similarity >= 0.7:  # High similarity threshold
                            similar_pathways.append(other_pathway)
                            similar_case_ids.add(other_case_id)
                            processed_pathways.add(other_key)
                
                if len(similar_case_ids) >= 2:  # Pattern needs at least 2 cases
                    group = {
                        'group_id': len(pathway_groups),
                        'cases': similar_case_ids,
                        'pathways': similar_pathways,
                        'avg_similarity': self._calculate_group_similarity(similar_pathways)
                    }
                    pathway_groups.append(group)
                
                processed_pathways.add(pathway_key)
        
        return pathway_groups
    
    def _calculate_pathway_similarity(self, pathway1: Dict[str, Any], pathway2: Dict[str, Any],
                                    mapping_lookup: Dict[str, Dict[str, Tuple[str, str, float]]]) -> float:
        """
        Calculate similarity between two pathways.
        
        Args:
            pathway1: First pathway
            pathway2: Second pathway
            mapping_lookup: Node mapping lookup
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Length similarity
        len1, len2 = pathway1['length'], pathway2['length']
        length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
        
        # Node mapping similarity
        path1, path2 = pathway1['path'], pathway2['path']
        case1, case2 = pathway1['case_id'], pathway2['case_id']
        
        mapped_nodes = 0
        total_comparisons = min(len(path1), len(path2))
        
        if total_comparisons > 0:
            for i in range(total_comparisons):
                node1 = path1[i]
                node2 = path2[i]
                
                # Check if nodes are mapped
                if node1 in mapping_lookup[case1]:
                    mapped_case, mapped_node, mapping_sim = mapping_lookup[case1][node1]
                    if mapped_case == case2 and mapped_node == node2:
                        mapped_nodes += mapping_sim
                    elif mapped_case == case2:
                        mapped_nodes += 0.5  # Partial credit for same case mapping
                
            node_sim = mapped_nodes / total_comparisons
        else:
            node_sim = 0.0
        
        # Edge type similarity
        edge_types1 = pathway1.get('edge_types', [])
        edge_types2 = pathway2.get('edge_types', [])
        
        if edge_types1 and edge_types2:
            common_edge_types = 0
            total_edges = min(len(edge_types1), len(edge_types2))
            
            for i in range(total_edges):
                if edge_types1[i] == edge_types2[i]:
                    common_edge_types += 1
            
            edge_sim = common_edge_types / total_edges if total_edges > 0 else 0.0
        else:
            edge_sim = 0.5  # Neutral if no edge type information
        
        # Combine similarities
        overall_similarity = (
            0.3 * length_sim +
            0.5 * node_sim +
            0.2 * edge_sim
        )
        
        return overall_similarity
    
    def _calculate_group_similarity(self, pathways: List[Dict[str, Any]]) -> float:
        """
        Calculate average similarity within a group of pathways.
        
        Args:
            pathways: List of pathways in group
            
        Returns:
            Average similarity score
        """
        if len(pathways) < 2:
            return 1.0
        
        # This is simplified - would need proper pairwise comparison
        # For now, return based on length consistency
        lengths = [p['length'] for p in pathways]
        avg_length = sum(lengths) / len(lengths)
        
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_consistency = 1.0 / (1.0 + length_variance)  # Higher consistency = lower variance
        
        return length_consistency
    
    def _create_pathway_pattern(self, group_id: int, pathway_group: Dict[str, Any],
                              graphs: Dict[str, nx.DiGraph]) -> Optional[MechanismPattern]:
        """
        Create mechanism pattern from pathway group.
        
        Args:
            group_id: Group identifier
            pathway_group: Group of similar pathways
            graphs: Case graphs
            
        Returns:
            MechanismPattern or None
        """
        cases = list(pathway_group['cases'])
        pathways = pathway_group['pathways']
        
        if len(cases) < 2:
            return None
        
        # Extract common structure
        core_nodes = self._extract_core_nodes(pathways)
        core_edges = self._extract_core_edges(pathways)
        
        # Determine pattern type
        if len(cases) == len(graphs):
            pattern_type = MechanismType.UNIVERSAL
        elif len(cases) >= len(graphs) * 0.7:
            pattern_type = MechanismType.CONDITIONAL
        else:
            pattern_type = MechanismType.CASE_SPECIFIC
        
        # Calculate frequencies
        case_frequencies = {}
        for case in cases:
            case_pathways = [p for p in pathways if p['case_id'] == case]
            case_frequencies[case] = len(case_pathways) / len(pathways)
        
        pattern = MechanismPattern(
            pattern_id=f"pathway_pattern_{group_id}",
            pattern_name=f"Pathway Pattern {group_id}",
            description=f"Recurring causal pathway across {len(cases)} cases",
            mechanism_type=pattern_type,
            scope_conditions=[],
            participating_cases=cases,
            case_frequencies=case_frequencies,
            core_nodes=core_nodes,
            core_edges=core_edges,
            pattern_strength=pathway_group['avg_similarity'],
            consistency_score=pathway_group['avg_similarity'],
            generalizability=len(cases) / len(graphs)
        )
        
        return pattern
    
    def _extract_core_nodes(self, pathways: List[Dict[str, Any]]) -> List[str]:
        """
        Extract core nodes that appear in most pathways.
        
        Args:
            pathways: List of pathways
            
        Returns:
            List of core node identifiers
        """
        node_counts = Counter()
        
        for pathway in pathways:
            for node in pathway['path']:
                node_counts[node] += 1
        
        # Core nodes appear in at least 50% of pathways
        threshold = len(pathways) * 0.5
        core_nodes = [node for node, count in node_counts.items() if count >= threshold]
        
        return core_nodes
    
    def _extract_core_edges(self, pathways: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Extract core edges that appear in most pathways.
        
        Args:
            pathways: List of pathways
            
        Returns:
            List of core edge tuples
        """
        edge_counts = Counter()
        
        for pathway in pathways:
            path = pathway['path']
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                edge_counts[edge] += 1
        
        # Core edges appear in at least 50% of pathways
        threshold = len(pathways) * 0.5
        core_edges = [edge for edge, count in edge_counts.items() if count >= threshold]
        
        return core_edges
    
    def _detect_convergence_patterns(self, graphs: Dict[str, nx.DiGraph]) -> List[MechanismPattern]:
        """
        Detect convergence patterns (multiple causes, single effect).
        
        Args:
            graphs: Dictionary of case_id -> graph
            
        Returns:
            List of convergence patterns
        """
        patterns = []
        
        # Find high in-degree nodes across cases
        convergence_candidates = defaultdict(list)
        
        for case_id, graph in graphs.items():
            for node_id in graph.nodes():
                in_degree = graph.in_degree(node_id)
                if in_degree >= 2:  # Convergence threshold
                    convergence_candidates[in_degree].append((case_id, node_id))
        
        # Create patterns for convergence structures that appear in multiple cases
        for in_degree, candidates in convergence_candidates.items():
            cases = set(item[0] for item in candidates)
            if len(cases) >= 2:
                pattern = MechanismPattern(
                    pattern_id=f"convergence_pattern_{in_degree}",
                    pattern_name=f"Convergence Pattern (in-degree {in_degree})",
                    description=f"Multiple causes converging to single effect",
                    mechanism_type=MechanismType.CONDITIONAL,
                    scope_conditions=[ScopeCondition.CONTEXT_DEPENDENT],
                    participating_cases=list(cases),
                    pattern_strength=0.7,
                    consistency_score=0.7,
                    generalizability=len(cases) / len(graphs)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_divergence_patterns(self, graphs: Dict[str, nx.DiGraph]) -> List[MechanismPattern]:
        """
        Detect divergence patterns (single cause, multiple effects).
        
        Args:
            graphs: Dictionary of case_id -> graph
            
        Returns:
            List of divergence patterns
        """
        patterns = []
        
        # Find high out-degree nodes across cases
        divergence_candidates = defaultdict(list)
        
        for case_id, graph in graphs.items():
            for node_id in graph.nodes():
                out_degree = graph.out_degree(node_id)
                if out_degree >= 2:  # Divergence threshold
                    divergence_candidates[out_degree].append((case_id, node_id))
        
        # Create patterns for divergence structures that appear in multiple cases
        for out_degree, candidates in divergence_candidates.items():
            cases = set(item[0] for item in candidates)
            if len(cases) >= 2:
                pattern = MechanismPattern(
                    pattern_id=f"divergence_pattern_{out_degree}",
                    pattern_name=f"Divergence Pattern (out-degree {out_degree})",
                    description=f"Single cause leading to multiple effects",
                    mechanism_type=MechanismType.CONDITIONAL,
                    scope_conditions=[ScopeCondition.CONTEXT_DEPENDENT],
                    participating_cases=list(cases),
                    pattern_strength=0.7,
                    consistency_score=0.7,
                    generalizability=len(cases) / len(graphs)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_hub_patterns(self, graphs: Dict[str, nx.DiGraph]) -> List[MechanismPattern]:
        """
        Detect hub patterns (high degree nodes).
        
        Args:
            graphs: Dictionary of case_id -> graph
            
        Returns:
            List of hub patterns
        """
        patterns = []
        
        # Issue #61 Fix: Find high degree nodes with proper directed graph analysis
        hub_candidates = defaultdict(list)
        
        for case_id, graph in graphs.items():
            for node_id in graph.nodes():
                # For directed graphs, analyze in-degree, out-degree, and total degree separately
                if isinstance(graph, nx.DiGraph):
                    in_degree = graph.in_degree(node_id)
                    out_degree = graph.out_degree(node_id)
                    total_degree = in_degree + out_degree
                    
                    # Create different hub types based on directional patterns
                    if total_degree >= 3:  # General hub threshold
                        hub_type = "balanced_hub"
                        if in_degree >= 2 * out_degree:  # Primarily incoming hub
                            hub_type = "sink_hub"
                        elif out_degree >= 2 * in_degree:  # Primarily outgoing hub
                            hub_type = "source_hub"
                        
                        hub_candidates[f"{hub_type}_{total_degree}"].append((case_id, node_id))
                else:
                    # For undirected graphs, use total degree
                    degree = graph.degree(node_id)
                    if degree >= 3:  # Hub threshold
                        hub_candidates[f"hub_{degree}"].append((case_id, node_id))
        
        # Create patterns for hub structures that appear in multiple cases
        for hub_key, candidates in hub_candidates.items():
            cases = set(item[0] for item in candidates)
            if len(cases) >= 2:
                # Extract hub type and degree from key
                if "_" in hub_key:
                    hub_type, degree_str = hub_key.rsplit("_", 1)
                    try:
                        degree = int(degree_str)
                    except ValueError:
                        degree = "unknown"
                else:
                    hub_type = hub_key
                    degree = "unknown"
                
                # Create description based on hub type
                if "sink_hub" in hub_type:
                    description = f"Central convergence node (high in-degree)"
                elif "source_hub" in hub_type:
                    description = f"Central divergence node (high out-degree)"
                elif "balanced_hub" in hub_type:
                    description = f"Central bidirectional node (balanced in/out degree)"
                else:
                    description = f"Central node with high connectivity"
                
                pattern = MechanismPattern(
                    pattern_id=f"hub_pattern_{hub_key}",
                    pattern_name=f"Hub Pattern ({hub_type} degree {degree})",
                    description=description,
                    mechanism_type=MechanismType.CONDITIONAL,
                    scope_conditions=[ScopeCondition.ACTOR_DEPENDENT],
                    participating_cases=list(cases),
                    pattern_strength=0.6,
                    consistency_score=0.6,
                    generalizability=len(cases) / len(graphs)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _create_evidence_pattern(self, evidence_type: str, evidence_list: List[Tuple[str, str, Dict[str, Any]]],
                               graphs: Dict[str, nx.DiGraph]) -> Optional[MechanismPattern]:
        """
        Create evidence-based pattern.
        
        Args:
            evidence_type: Van Evera evidence type
            evidence_list: List of (case_id, node_id, node_data) tuples
            graphs: Case graphs
            
        Returns:
            MechanismPattern or None
        """
        cases = set(item[0] for item in evidence_list)
        
        if len(cases) < 2:
            return None
        
        pattern = MechanismPattern(
            pattern_id=f"evidence_pattern_{evidence_type}",
            pattern_name=f"Evidence Pattern ({evidence_type})",
            description=f"Recurring {evidence_type} evidence across cases",
            mechanism_type=MechanismType.CONDITIONAL,
            scope_conditions=[ScopeCondition.CONTEXT_DEPENDENT],
            participating_cases=list(cases),
            pattern_strength=0.6,
            consistency_score=0.6,
            generalizability=len(cases) / len(graphs),
            van_evera_support={case_id: evidence_type for case_id in cases}
        )
        
        return pattern
    
    def _validate_patterns(self, patterns: List[MechanismPattern], 
                         graphs: Dict[str, nx.DiGraph],
                         case_metadata: Dict[str, CaseMetadata]) -> List[MechanismPattern]:
        """
        Validate and filter patterns based on quality criteria.
        
        Args:
            patterns: List of candidate patterns
            graphs: Case graphs
            case_metadata: Case metadata
            
        Returns:
            List of validated patterns
        """
        validated = []
        
        for pattern in patterns:
            # Check minimum frequency
            frequency = len(pattern.participating_cases) / len(graphs)
            if frequency < self.min_pattern_frequency:
                continue
            
            # Check minimum strength
            if pattern.pattern_strength < self.min_pattern_strength:
                continue
            
            # Update pattern metrics
            pattern.generalizability = self.calculate_pattern_generalizability(pattern, case_metadata)
            pattern.scope_conditions = self.identify_scope_conditions(pattern, case_metadata)
            
            validated.append(pattern)
        
        return validated
    
    def _analyze_case_variation(self, pattern: MechanismPattern, graph: nx.DiGraph, case_id: str) -> Dict[str, Any]:
        """
        Analyze how pattern varies in a specific case.
        
        Args:
            pattern: Mechanism pattern
            graph: Case graph
            case_id: Case identifier
            
        Returns:
            Case variation analysis
        """
        variation = {
            'case_id': case_id,
            'core_nodes_present': [],
            'optional_nodes_present': [],
            'core_edges_present': [],
            'optional_edges_present': [],
            'case_specific_additions': [],
            'completeness_score': 0.0
        }
        
        # Check core nodes
        for node in pattern.core_nodes:
            if node in graph.nodes():
                variation['core_nodes_present'].append(node)
        
        # Check optional nodes
        for node in pattern.optional_nodes:
            if node in graph.nodes():
                variation['optional_nodes_present'].append(node)
        
        # Check core edges
        for source, target in pattern.core_edges:
            if graph.has_edge(source, target):
                variation['core_edges_present'].append((source, target))
        
        # Check optional edges
        for source, target in pattern.optional_edges:
            if graph.has_edge(source, target):
                variation['optional_edges_present'].append((source, target))
        
        # Calculate completeness
        total_core_elements = len(pattern.core_nodes) + len(pattern.core_edges)
        present_core_elements = len(variation['core_nodes_present']) + len(variation['core_edges_present'])
        
        if total_core_elements > 0:
            variation['completeness_score'] = present_core_elements / total_core_elements
        else:
            variation['completeness_score'] = 1.0
        
        return variation
    
    def _summarize_variations(self, variations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize variation patterns across cases.
        
        Args:
            variations: Case variation analyses
            
        Returns:
            Variation summary
        """
        if not variations:
            return {}
        
        # Calculate average completeness
        completeness_scores = [v['completeness_score'] for v in variations.values()]
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        
        # Find common additions
        all_additions = []
        for variation in variations.values():
            all_additions.extend(variation.get('case_specific_additions', []))
        
        common_additions = [item for item, count in Counter(all_additions).items() if count >= 2]
        
        return {
            'average_completeness': avg_completeness,
            'completeness_range': (min(completeness_scores), max(completeness_scores)),
            'common_additions': common_additions,
            'variation_count': len(variations)
        }
    
    def _calculate_consistency_score(self, variations: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate consistency score across variations.
        
        Args:
            variations: Case variation analyses
            
        Returns:
            Consistency score (0.0-1.0)
        """
        if not variations:
            return 1.0
        
        completeness_scores = [v['completeness_score'] for v in variations.values()]
        
        # Calculate variance in completeness
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        variance = sum((score - avg_completeness) ** 2 for score in completeness_scores) / len(completeness_scores)
        
        # Convert variance to consistency (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + variance)
        
        return consistency
    
    def _identify_variant_types(self, variations: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Identify types of variations across cases.
        
        Args:
            variations: Case variation analyses
            
        Returns:
            List of variant type descriptions
        """
        variant_types = []
        
        # Check for completeness variations
        completeness_scores = [v['completeness_score'] for v in variations.values()]
        if max(completeness_scores) - min(completeness_scores) > 0.3:
            variant_types.append("Completeness Variation")
        
        # Check for structural additions
        has_additions = any(v.get('case_specific_additions') for v in variations.values())
        if has_additions:
            variant_types.append("Structural Additions")
        
        # Check for core element variations
        core_variations = set()
        for variation in variations.values():
            core_count = len(variation['core_nodes_present']) + len(variation['core_edges_present'])
            core_variations.add(core_count)
        
        if len(core_variations) > 1:
            variant_types.append("Core Element Variation")
        
        return variant_types
    
    def _analyze_contextual_conditions(self, pattern: MechanismPattern,
                                     case_metadata: Dict[str, CaseMetadata]) -> List[ScopeCondition]:
        """
        Analyze contextual scope conditions.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata
            
        Returns:
            List of contextual scope conditions
        """
        conditions = []
        
        # Check if pattern is context-dependent
        contexts = set()
        for case_id in pattern.participating_cases:
            if case_id in case_metadata:
                metadata = case_metadata[case_id]
                context_key = (
                    metadata.geographic_context,
                    metadata.institutional_context,
                    metadata.political_context
                )
                contexts.add(context_key)
        
        if len(contexts) > 1:
            conditions.append(ScopeCondition.CONTEXT_DEPENDENT)
        
        return conditions
    
    def _analyze_temporal_conditions(self, pattern: MechanismPattern,
                                   case_metadata: Dict[str, CaseMetadata]) -> List[ScopeCondition]:
        """
        Analyze temporal scope conditions.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata
            
        Returns:
            List of temporal scope conditions
        """
        conditions = []
        
        # Check if pattern is time-dependent
        time_periods = []
        for case_id in pattern.participating_cases:
            if case_id in case_metadata:
                metadata = case_metadata[case_id]
                if metadata.time_period:
                    time_periods.append(metadata.time_period)
        
        if len(time_periods) > 1:
            # Check for temporal clustering
            start_years = [tp[0].year for tp in time_periods]
            year_range = max(start_years) - min(start_years)
            
            if year_range > 10:  # Patterns spanning more than 10 years
                conditions.append(ScopeCondition.TIME_DEPENDENT)
        
        return conditions
    
    def _analyze_actor_conditions(self, pattern: MechanismPattern,
                                case_metadata: Dict[str, CaseMetadata]) -> List[ScopeCondition]:
        """
        Analyze actor-dependent scope conditions.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata
            
        Returns:
            List of actor-dependent scope conditions
        """
        conditions = []
        
        # This is simplified - would need actor information in metadata
        # For now, assume patterns with hub structures are actor-dependent
        if any("hub" in pattern.pattern_id.lower() for _ in [pattern]):
            conditions.append(ScopeCondition.ACTOR_DEPENDENT)
        
        return conditions
    
    def _analyze_resource_conditions(self, pattern: MechanismPattern,
                                   case_metadata: Dict[str, CaseMetadata]) -> List[ScopeCondition]:
        """
        Analyze resource-dependent scope conditions.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata
            
        Returns:
            List of resource-dependent scope conditions
        """
        conditions = []
        
        # Check economic context variations
        economic_contexts = set()
        for case_id in pattern.participating_cases:
            if case_id in case_metadata:
                metadata = case_metadata[case_id]
                if metadata.economic_context:
                    economic_contexts.add(metadata.economic_context)
        
        if len(economic_contexts) > 1:
            conditions.append(ScopeCondition.RESOURCE_DEPENDENT)
        
        return conditions
    
    def _calculate_context_diversity(self, pattern: MechanismPattern,
                                   case_metadata: Dict[str, CaseMetadata]) -> float:
        """
        Calculate diversity of contexts where pattern appears.
        
        Args:
            pattern: Mechanism pattern
            case_metadata: Case metadata
            
        Returns:
            Context diversity score (0.0-1.0)
        """
        contexts = set()
        
        for case_id in pattern.participating_cases:
            if case_id in case_metadata:
                metadata = case_metadata[case_id]
                context_tuple = (
                    metadata.geographic_context or 'unknown',
                    metadata.institutional_context or 'unknown',
                    metadata.economic_context or 'unknown',
                    metadata.political_context or 'unknown'
                )
                contexts.add(context_tuple)
        
        # More unique contexts = higher diversity
        max_possible_diversity = len(pattern.participating_cases)
        actual_diversity = len(contexts)
        
        return actual_diversity / max_possible_diversity if max_possible_diversity > 0 else 0.0
    
    def _calculate_evidence_strength(self, pattern: MechanismPattern) -> float:
        """
        Calculate overall evidence strength for pattern.
        
        Args:
            pattern: Mechanism pattern
            
        Returns:
            Evidence strength score (0.0-1.0)
        """
        if not pattern.van_evera_support:
            return 0.5  # Neutral if no evidence information
        
        # Weight evidence types
        evidence_weights = {
            'doubly_decisive': 1.0,
            'smoking_gun': 0.8,
            'hoop': 0.6,
            'straw_in_the_wind': 0.4
        }
        
        total_weight = 0.0
        total_evidence = 0
        
        for case_id, evidence_type in pattern.van_evera_support.items():
            weight = evidence_weights.get(evidence_type, 0.5)
            total_weight += weight
            total_evidence += 1
        
        return total_weight / total_evidence if total_evidence > 0 else 0.5


def test_mechanism_pattern_detector():
    """Test function for mechanism pattern detector"""
    # Create test graphs
    graphs = {}
    
    # Case 1
    graph1 = nx.DiGraph()
    graph1.add_node("crisis", type="Event", description="Financial crisis")
    graph1.add_node("response", type="Event", description="Policy response")
    graph1.add_node("outcome", type="Event", description="Market stabilization")
    graph1.add_edge("crisis", "response", type="causes")
    graph1.add_edge("response", "outcome", type="causes")
    graphs["case1"] = graph1
    
    # Case 2 (similar pattern)
    graph2 = nx.DiGraph()
    graph2.add_node("shock", type="Event", description="Economic shock")
    graph2.add_node("intervention", type="Event", description="Government intervention")
    graph2.add_node("recovery", type="Event", description="Economic recovery")
    graph2.add_edge("shock", "intervention", type="causes")
    graph2.add_edge("intervention", "recovery", type="causes")
    graphs["case2"] = graph2
    
    # Create metadata
    from core.comparative_models import create_default_case_metadata
    case_metadata = {
        "case1": create_default_case_metadata("case1", "Financial Crisis Case"),
        "case2": create_default_case_metadata("case2", "Economic Shock Case")
    }
    
    # Test pattern detection
    detector = MechanismPatternDetector()
    
    # Create simple node mappings for testing
    node_mappings = {
        ("case1", "case2"): [
            # This would normally come from graph alignment
        ]
    }
    
    patterns = detector.detect_patterns(graphs, case_metadata, node_mappings)
    
    print(f"Detected {len(patterns)} patterns")
    for pattern in patterns:
        print(f"  Pattern: {pattern.pattern_name}")
        print(f"    Cases: {pattern.participating_cases}")
        print(f"    Strength: {pattern.pattern_strength:.2f}")
        print(f"    Type: {pattern.mechanism_type.value}")


if __name__ == "__main__":
    test_mechanism_pattern_detector()