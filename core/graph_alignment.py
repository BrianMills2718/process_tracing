"""
Comparative Process Tracing - Cross-Case Graph Alignment Module

Aligns similar mechanisms and nodes across cases for comparative analysis
including similarity detection, mapping, and graph structure comparison.

Author: Claude Code Implementation  
Date: August 2025
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict
import logging
from difflib import SequenceMatcher

from core.comparative_models import (
    NodeMapping, MechanismPattern, ComparativeAnalysisError,
    calculate_overall_similarity, MechanismType
)


class GraphAligner:
    """
    Aligns graphs across cases to identify similar mechanisms and structures.
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize graph aligner.
        
        Args:
            similarity_threshold: Minimum similarity for node mapping (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.semantic_weight = 0.4
        self.structural_weight = 0.3
        self.temporal_weight = 0.2
        self.functional_weight = 0.1
        
        self.logger = logging.getLogger(__name__)
    
    def align_graphs(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                    case1_id: str, case2_id: str) -> List[NodeMapping]:
        """
        Align two graphs to find corresponding nodes.
        
        Args:
            graph1: First graph
            graph2: Second graph
            case1_id: ID of first case
            case2_id: ID of second case
            
        Returns:
            List of node mappings between graphs
        """
        mappings = []
        
        # Calculate all pairwise similarities
        similarity_matrix = self._calculate_similarity_matrix(graph1, graph2)
        
        # Find best mappings using Hungarian algorithm approximation
        node_mappings = self._find_optimal_mappings(
            similarity_matrix, list(graph1.nodes()), list(graph2.nodes())
        )
        
        # Create NodeMapping objects
        for node1, node2, similarity in node_mappings:
            if similarity >= self.similarity_threshold:
                mapping = self._create_node_mapping(
                    graph1, graph2, node1, node2, case1_id, case2_id, similarity
                )
                mappings.append(mapping)
        
        self.logger.info(f"Found {len(mappings)} node mappings between {case1_id} and {case2_id}")
        return mappings
    
    def align_multiple_graphs(self, graphs: Dict[str, nx.DiGraph]) -> Dict[Tuple[str, str], List[NodeMapping]]:
        """
        Align multiple graphs pairwise.
        
        Args:
            graphs: Dictionary of case_id -> graph
            
        Returns:
            Dictionary of (case1, case2) -> mappings
        """
        all_mappings = {}
        case_ids = list(graphs.keys())
        
        for i, case1 in enumerate(case_ids):
            for case2 in case_ids[i+1:]:
                mappings = self.align_graphs(
                    graphs[case1], graphs[case2], case1, case2
                )
                all_mappings[(case1, case2)] = mappings
        
        return all_mappings
    
    def find_common_subgraphs(self, graphs: Dict[str, nx.DiGraph], 
                             min_cases: int = 2) -> List[Dict[str, Any]]:
        """
        Find common subgraph patterns across multiple cases.
        
        Args:
            graphs: Dictionary of case_id -> graph
            min_cases: Minimum number of cases for pattern to be considered
            
        Returns:
            List of common subgraph patterns
        """
        # First align all graphs pairwise
        all_mappings = self.align_multiple_graphs(graphs)
        
        # Build similarity clusters
        similarity_clusters = self._build_similarity_clusters(all_mappings)
        
        # Find subgraph patterns
        patterns = []
        for cluster in similarity_clusters:
            if len(cluster['cases']) >= min_cases:
                pattern = self._extract_subgraph_pattern(cluster, graphs)
                if pattern:
                    patterns.append(pattern)
        
        self.logger.info(f"Found {len(patterns)} common subgraph patterns")
        return patterns
    
    def calculate_structural_similarity(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """
        Calculate overall structural similarity between two graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Structural similarity score (0.0-1.0)
        """
        # Basic graph metrics
        metrics1 = self._calculate_graph_metrics(graph1)
        metrics2 = self._calculate_graph_metrics(graph2)
        
        # Compare metrics
        similarity_scores = []
        
        for metric in ['density', 'avg_clustering', 'avg_path_length']:
            if metric in metrics1 and metric in metrics2:
                # Normalize difference to similarity
                diff = abs(metrics1[metric] - metrics2[metric])
                max_val = max(metrics1[metric], metrics2[metric])
                if max_val > 0:
                    similarity = 1.0 - (diff / max_val)
                else:
                    similarity = 1.0
                similarity_scores.append(similarity)
        
        # Compare degree distributions
        degree_sim = self._compare_degree_distributions(graph1, graph2)
        similarity_scores.append(degree_sim)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_similarity_matrix(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> np.ndarray:
        """
        Calculate similarity matrix between all node pairs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Similarity matrix
        """
        nodes1 = list(graph1.nodes())
        nodes2 = list(graph2.nodes())
        
        matrix = np.zeros((len(nodes1), len(nodes2)))
        
        for i, node1 in enumerate(nodes1):
            for j, node2 in enumerate(nodes2):
                similarity = self._calculate_node_similarity(
                    graph1, graph2, node1, node2
                )
                matrix[i, j] = similarity
        
        return matrix
    
    def _calculate_node_similarity(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                                  node1: str, node2: str) -> float:
        """
        Calculate similarity between two nodes.
        
        Args:
            graph1: First graph
            graph2: Second graph
            node1: Node from first graph
            node2: Node from second graph
            
        Returns:
            Similarity score (0.0-1.0)
        """
        node1_data = graph1.nodes[node1]
        node2_data = graph2.nodes[node2]
        
        # Semantic similarity (based on description/type)
        semantic_sim = self._calculate_semantic_similarity(node1_data, node2_data)
        
        # Structural similarity (based on graph position)
        structural_sim = self._calculate_structural_similarity(
            graph1, graph2, node1, node2
        )
        
        # Temporal similarity (if temporal data available)
        temporal_sim = self._calculate_temporal_similarity(node1_data, node2_data)
        
        # Functional similarity (based on causal role)
        functional_sim = self._calculate_functional_similarity(
            graph1, graph2, node1, node2
        )
        
        # Weighted combination
        overall_similarity = (
            self.semantic_weight * semantic_sim +
            self.structural_weight * structural_sim +
            self.temporal_weight * temporal_sim +
            self.functional_weight * functional_sim
        )
        
        return min(1.0, max(0.0, overall_similarity))
    
    def _calculate_semantic_similarity(self, node1_data: Dict[str, Any], 
                                     node2_data: Dict[str, Any]) -> float:
        """
        Calculate semantic similarity between node attributes.
        
        Args:
            node1_data: First node attributes
            node2_data: Second node attributes
            
        Returns:
            Semantic similarity (0.0-1.0)
        """
        similarity_scores = []
        
        # Type similarity
        type1 = node1_data.get('type', '')
        type2 = node2_data.get('type', '')
        if type1 and type2:
            type_sim = 1.0 if type1 == type2 else 0.0
            similarity_scores.append(type_sim)
        
        # Description similarity (Issue #83 Fix: Enhanced semantic matching)
        desc1 = node1_data.get('description', '')
        desc2 = node2_data.get('description', '')
        if desc1 and desc2:
            # Basic sequence matching
            seq_sim = SequenceMatcher(None, desc1.lower(), desc2.lower()).ratio()
            
            # Enhanced semantic matching
            semantic_sim = self._calculate_semantic_similarity(desc1, desc2)
            
            # Weighted combination (60% semantic, 40% sequence)
            desc_sim = 0.6 * semantic_sim + 0.4 * seq_sim
            similarity_scores.append(desc_sim)
        
        # Properties similarity
        props1 = node1_data.get('properties', {})
        props2 = node2_data.get('properties', {})
        if props1 and props2:
            common_keys = set(props1.keys()) & set(props2.keys())
            if common_keys:
                prop_similarities = []
                for key in common_keys:
                    val1, val2 = str(props1[key]), str(props2[key])
                    prop_sim = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
                    prop_similarities.append(prop_sim)
                similarity_scores.append(sum(prop_similarities) / len(prop_similarities))
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text descriptions.
        Issue #83 Fix: Enhanced semantic matching for cross-case data structures.
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_norm = text1.lower().strip()
        text2_norm = text2.lower().strip()
        
        if text1_norm == text2_norm:
            return 1.0
        
        # Tokenize and create word sets
        import re
        words1 = set(re.findall(r'\b\w+\b', text1_norm))
        words2 = set(re.findall(r'\b\w+\b', text2_norm))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Calculate semantic overlap score
        # Boost score for common semantic indicators
        semantic_indicators = {
            'cause', 'effect', 'trigger', 'lead', 'result', 'influence', 'impact',
            'revolution', 'war', 'conflict', 'economic', 'political', 'social',
            'evidence', 'hypothesis', 'mechanism', 'factor', 'condition'
        }
        
        semantic1 = words1 & semantic_indicators
        semantic2 = words2 & semantic_indicators
        
        if semantic1 and semantic2:
            semantic_overlap = len(semantic1 & semantic2) / len(semantic1 | semantic2)
            # Boost Jaccard score by semantic overlap
            return min(1.0, jaccard_sim + 0.3 * semantic_overlap)
        
        return jaccard_sim
    
    def _calculate_structural_similarity(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                                       node1: str, node2: str) -> float:
        """
        Calculate structural similarity based on graph position.
        
        Args:
            graph1: First graph
            graph2: Second graph
            node1: Node from first graph
            node2: Node from second graph
            
        Returns:
            Structural similarity (0.0-1.0)
        """
        # Issue #61 Fix: Use appropriate degree measures for directed graphs
        if isinstance(graph1, nx.DiGraph) and isinstance(graph2, nx.DiGraph):
            # For directed graphs, total degree is sum of in and out degrees
            degree1 = graph1.in_degree(node1) + graph1.out_degree(node1)
            degree2 = graph2.in_degree(node2) + graph2.out_degree(node2)
        else:
            # For undirected graphs, use standard degree
            degree1 = graph1.degree(node1)
            degree2 = graph2.degree(node2)
        
        max_degree = max(degree1, degree2) if max(degree1, degree2) > 0 else 1
        degree_sim = 1.0 - abs(degree1 - degree2) / max_degree
        
        # In-degree and out-degree similarity
        in_degree1 = graph1.in_degree(node1)
        in_degree2 = graph2.in_degree(node2)
        out_degree1 = graph1.out_degree(node1)
        out_degree2 = graph2.out_degree(node2)
        
        max_in = max(in_degree1, in_degree2) if max(in_degree1, in_degree2) > 0 else 1
        max_out = max(out_degree1, out_degree2) if max(out_degree1, out_degree2) > 0 else 1
        
        in_degree_sim = 1.0 - abs(in_degree1 - in_degree2) / max_in
        out_degree_sim = 1.0 - abs(out_degree1 - out_degree2) / max_out
        
        # Clustering coefficient similarity
        clustering1 = nx.clustering(graph1.to_undirected(), node1)
        clustering2 = nx.clustering(graph2.to_undirected(), node2)
        clustering_sim = 1.0 - abs(clustering1 - clustering2)
        
        # Average neighbor similarity (simplified)
        neighbors1 = set(graph1.neighbors(node1))
        neighbors2 = set(graph2.neighbors(node2))
        if neighbors1 and neighbors2:
            # This is a simplified version - could be enhanced with recursive similarity
            neighbor_overlap = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
        else:
            neighbor_overlap = 1.0 if not neighbors1 and not neighbors2 else 0.0
        
        # Combine structural metrics
        structural_similarity = (
            0.3 * degree_sim +
            0.2 * in_degree_sim +
            0.2 * out_degree_sim +
            0.2 * clustering_sim +
            0.1 * neighbor_overlap
        )
        
        return structural_similarity
    
    def _calculate_temporal_similarity(self, node1_data: Dict[str, Any], 
                                     node2_data: Dict[str, Any]) -> float:
        """
        Calculate temporal similarity between nodes.
        
        Args:
            node1_data: First node attributes
            node2_data: Second node attributes
            
        Returns:
            Temporal similarity (0.0-1.0)
        """
        # Check for temporal attributes
        temporal_attrs = ['timestamp', 'duration', 'sequence_order', 'temporal_uncertainty']
        
        similarities = []
        for attr in temporal_attrs:
            val1 = node1_data.get(attr)
            val2 = node2_data.get(attr)
            
            if val1 is not None and val2 is not None:
                if attr == 'timestamp':
                    # Compare timestamps (simplified)
                    sim = 1.0 if str(val1) == str(val2) else 0.5
                elif attr == 'sequence_order':
                    # Compare sequence positions
                    max_seq = max(val1, val2) if max(val1, val2) > 0 else 1
                    sim = 1.0 - abs(val1 - val2) / max_seq
                else:
                    # General numeric comparison
                    try:
                        max_val = max(float(val1), float(val2))
                        if max_val > 0:
                            sim = 1.0 - abs(float(val1) - float(val2)) / max_val
                        else:
                            sim = 1.0
                    except:
                        sim = 1.0 if str(val1) == str(val2) else 0.0
                
                similarities.append(sim)
        
        # Return average temporal similarity or neutral score
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def _calculate_functional_similarity(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                                       node1: str, node2: str) -> float:
        """
        Calculate functional similarity based on causal role.
        
        Args:
            graph1: First graph
            graph2: Second graph
            node1: Node from first graph
            node2: Node from second graph
            
        Returns:
            Functional similarity (0.0-1.0)
        """
        # Analyze causal role patterns
        
        # Input/output pattern similarity
        has_inputs1 = graph1.in_degree(node1) > 0
        has_inputs2 = graph2.in_degree(node2) > 0
        has_outputs1 = graph1.out_degree(node1) > 0
        has_outputs2 = graph2.out_degree(node2) > 0
        
        input_sim = 1.0 if has_inputs1 == has_inputs2 else 0.0
        output_sim = 1.0 if has_outputs1 == has_outputs2 else 0.0
        
        # Causal position similarity
        is_source1 = graph1.in_degree(node1) == 0 and graph1.out_degree(node1) > 0
        is_source2 = graph2.in_degree(node2) == 0 and graph2.out_degree(node2) > 0
        is_sink1 = graph1.out_degree(node1) == 0 and graph1.in_degree(node1) > 0
        is_sink2 = graph2.out_degree(node2) == 0 and graph2.in_degree(node2) > 0
        
        position_sim = 0.0
        if is_source1 and is_source2:
            position_sim = 1.0
        elif is_sink1 and is_sink2:
            position_sim = 1.0
        elif not (is_source1 or is_sink1) and not (is_source2 or is_sink2):
            position_sim = 0.7  # Both are intermediate nodes
        
        # Edge type similarity (if available)
        edge_types1 = set()
        edge_types2 = set()
        
        for _, _, edge_data in graph1.in_edges(node1, data=True):
            edge_types1.add(edge_data.get('type', 'unknown'))
        for _, _, edge_data in graph1.out_edges(node1, data=True):
            edge_types1.add(edge_data.get('type', 'unknown'))
        
        for _, _, edge_data in graph2.in_edges(node2, data=True):
            edge_types2.add(edge_data.get('type', 'unknown'))
        for _, _, edge_data in graph2.out_edges(node2, data=True):
            edge_types2.add(edge_data.get('type', 'unknown'))
        
        if edge_types1 and edge_types2:
            common_types = edge_types1 & edge_types2
            all_types = edge_types1 | edge_types2
            edge_type_sim = len(common_types) / len(all_types) if all_types else 0.0
        else:
            edge_type_sim = 0.5  # Neutral if no edge type information
        
        # Combine functional metrics
        functional_similarity = (
            0.3 * input_sim +
            0.3 * output_sim +
            0.3 * position_sim +
            0.1 * edge_type_sim
        )
        
        return functional_similarity
    
    def _find_optimal_mappings(self, similarity_matrix: np.ndarray, 
                              nodes1: List[str], nodes2: List[str]) -> List[Tuple[str, str, float]]:
        """
        Find optimal node mappings using greedy approach.
        
        Args:
            similarity_matrix: Matrix of similarities
            nodes1: Nodes from first graph
            nodes2: Nodes from second graph
            
        Returns:
            List of (node1, node2, similarity) tuples
        """
        mappings = []
        used_rows = set()
        used_cols = set()
        
        # Create list of all similarities with indices
        all_similarities = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                all_similarities.append((similarity_matrix[i, j], i, j))
        
        # Sort by similarity (descending)
        all_similarities.sort(reverse=True)
        
        # Greedily select best non-conflicting mappings
        for similarity, i, j in all_similarities:
            if i not in used_rows and j not in used_cols and similarity >= self.similarity_threshold:
                mappings.append((nodes1[i], nodes2[j], similarity))
                used_rows.add(i)
                used_cols.add(j)
        
        return mappings
    
    def _create_node_mapping(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                           node1: str, node2: str, case1_id: str, case2_id: str,
                           overall_sim: float) -> NodeMapping:
        """
        Create detailed NodeMapping object.
        
        Args:
            graph1: First graph
            graph2: Second graph
            node1: Node from first graph
            node2: Node from second graph
            case1_id: First case ID
            case2_id: Second case ID
            overall_sim: Overall similarity score
            
        Returns:
            NodeMapping object
        """
        node1_data = graph1.nodes[node1]
        node2_data = graph2.nodes[node2]
        
        # Calculate individual similarity components
        semantic_sim = self._calculate_semantic_similarity(node1_data, node2_data)
        structural_sim = self._calculate_structural_similarity(graph1, graph2, node1, node2)
        temporal_sim = self._calculate_temporal_similarity(node1_data, node2_data)
        functional_sim = self._calculate_functional_similarity(graph1, graph2, node1, node2)
        
        # Calculate mapping confidence based on similarity and structural context
        confidence = overall_sim * 0.8 + 0.2 * min(1.0, (semantic_sim + structural_sim) / 2)
        
        return NodeMapping(
            mapping_id=f"{case1_id}_{node1}_to_{case2_id}_{node2}",
            source_case=case1_id,
            target_case=case2_id,
            source_node=node1,
            target_node=node2,
            semantic_similarity=semantic_sim,
            structural_similarity=structural_sim,
            temporal_similarity=temporal_sim,
            functional_similarity=functional_sim,
            overall_similarity=overall_sim,
            mapping_confidence=confidence
        )
    
    def _build_similarity_clusters(self, all_mappings: Dict[Tuple[str, str], List[NodeMapping]]) -> List[Dict[str, Any]]:
        """
        Build clusters of similar nodes across all cases.
        
        Args:
            all_mappings: All pairwise mappings
            
        Returns:
            List of similarity clusters
        """
        # This is a simplified clustering approach
        # In practice, you might want to use more sophisticated clustering algorithms
        
        clusters = []
        processed_nodes = set()
        
        for (case1, case2), mappings in all_mappings.items():
            for mapping in mappings:
                node_key1 = f"{case1}_{mapping.source_node}"
                node_key2 = f"{case2}_{mapping.target_node}"
                
                if node_key1 not in processed_nodes and node_key2 not in processed_nodes:
                    # Create new cluster
                    cluster = {
                        'cluster_id': f"cluster_{len(clusters)}",
                        'cases': {case1, case2},
                        'nodes': {case1: mapping.source_node, case2: mapping.target_node},
                        'similarities': [mapping.overall_similarity],
                        'avg_similarity': mapping.overall_similarity
                    }
                    clusters.append(cluster)
                    processed_nodes.add(node_key1)
                    processed_nodes.add(node_key2)
        
        return clusters
    
    def _extract_subgraph_pattern(self, cluster: Dict[str, Any], 
                                 graphs: Dict[str, nx.DiGraph]) -> Optional[Dict[str, Any]]:
        """
        Extract subgraph pattern from similarity cluster.
        
        Args:
            cluster: Similarity cluster
            graphs: All graphs
            
        Returns:
            Subgraph pattern or None
        """
        if len(cluster['cases']) < 2:
            return None
        
        # Extract local subgraphs around clustered nodes
        subgraphs = {}
        for case in cluster['cases']:
            node = cluster['nodes'][case]
            graph = graphs[case]
            
            # Get local neighborhood (1-hop)
            neighbors = set(graph.neighbors(node)) | {node}
            subgraph = graph.subgraph(neighbors)
            subgraphs[case] = subgraph
        
        # Find common structure
        pattern = {
            'pattern_id': cluster['cluster_id'],
            'cases': list(cluster['cases']),
            'core_nodes': cluster['nodes'],
            'avg_similarity': cluster['avg_similarity'],
            'subgraphs': subgraphs
        }
        
        return pattern
    
    def _calculate_graph_metrics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Calculate basic graph metrics.
        
        Args:
            graph: Graph to analyze
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if graph.number_of_nodes() > 0:
            # Density
            metrics['density'] = nx.density(graph)
            
            # Average clustering
            try:
                metrics['avg_clustering'] = nx.average_clustering(graph.to_undirected())
            except:
                metrics['avg_clustering'] = 0.0
            
            # Average path length
            try:
                if nx.is_connected(graph.to_undirected()):
                    metrics['avg_path_length'] = nx.average_shortest_path_length(graph.to_undirected())
                else:
                    metrics['avg_path_length'] = float('inf')
            except:
                metrics['avg_path_length'] = float('inf')
        
        return metrics
    
    def _compare_degree_distributions(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """
        Compare degree distributions between graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            
        Returns:
            Degree distribution similarity (0.0-1.0)
        """
        # Issue #61 Fix: Use appropriate degree measures for directed graphs
        if isinstance(graph1, nx.DiGraph) and isinstance(graph2, nx.DiGraph):
            # For directed graphs, use total degree (in + out)
            degrees1 = [graph1.in_degree(n) + graph1.out_degree(n) for n in graph1.nodes()]
            degrees2 = [graph2.in_degree(n) + graph2.out_degree(n) for n in graph2.nodes()]
        else:
            # For undirected graphs, use standard degree
            degrees1 = [d for n, d in graph1.degree()]
            degrees2 = [d for n, d in graph2.degree()]
        
        if not degrees1 and not degrees2:
            return 1.0
        if not degrees1 or not degrees2:
            return 0.0
        
        # Calculate basic statistics
        avg1, avg2 = sum(degrees1) / len(degrees1), sum(degrees2) / len(degrees2)
        max1, max2 = max(degrees1), max(degrees2)
        
        # Compare averages
        max_avg = max(avg1, avg2) if max(avg1, avg2) > 0 else 1
        avg_sim = 1.0 - abs(avg1 - avg2) / max_avg
        
        # Compare maximums
        max_max = max(max1, max2) if max(max1, max2) > 0 else 1
        max_sim = 1.0 - abs(max1 - max2) / max_max
        
        return (avg_sim + max_sim) / 2


def test_graph_aligner():
    """Test function for graph aligner"""
    # Create test graphs
    graph1 = nx.DiGraph()
    graph1.add_node("event1", type="Event", description="Crisis event")
    graph1.add_node("event2", type="Event", description="Policy response")
    graph1.add_edge("event1", "event2", type="causes")
    
    graph2 = nx.DiGraph()
    graph2.add_node("crisis", type="Event", description="Financial crisis event")
    graph2.add_node("response", type="Event", description="Government policy response")
    graph2.add_edge("crisis", "response", type="causes")
    
    # Test alignment
    aligner = GraphAligner()
    mappings = aligner.align_graphs(graph1, graph2, "case1", "case2")
    
    print(f"Found {len(mappings)} mappings")
    for mapping in mappings:
        print(f"  {mapping.source_node} -> {mapping.target_node}: {mapping.overall_similarity:.2f}")
    
    # Test structural similarity
    struct_sim = aligner.calculate_structural_similarity(graph1, graph2)
    print(f"Structural similarity: {struct_sim:.2f}")


if __name__ == "__main__":
    test_graph_aligner()