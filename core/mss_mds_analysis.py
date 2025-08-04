"""
Comparative Process Tracing - MSS/MDS Analysis Module

Implements Most Similar Systems (MSS) and Most Different Systems (MDS) 
comparative analysis methods for process tracing research.

Author: Claude Code Implementation  
Date: August 2025
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict
import logging
from dataclasses import dataclass, field
from scipy import stats  # Issue #17 Fix: Add statistical significance testing

from core.comparative_models import (
    CaseMetadata, ComparisonResult, ComparisonType, 
    MechanismPattern, CrossCaseEvidence, ComparativeAnalysisError
)
from core.mechanism_detector import MechanismDetector
from core.graph_alignment import GraphAligner


@dataclass
class MSS_Analysis:
    """Most Similar Systems analysis results"""
    comparison_id: str
    primary_case: str
    comparison_case: str
    context_similarity: float
    outcome_difference: float
    
    # Key differences that explain outcome variation
    differentiating_mechanisms: List[str] = field(default_factory=list)
    unique_pathways: Dict[str, List[str]] = field(default_factory=dict)
    critical_differences: List[str] = field(default_factory=list)
    
    # Evidence support
    supporting_evidence: Dict[str, List[str]] = field(default_factory=dict)
    confidence_level: float = 0.0


@dataclass
class MDS_Analysis:
    """Most Different Systems analysis results"""
    comparison_id: str
    primary_case: str
    comparison_case: str
    context_difference: float
    outcome_similarity: float
    
    # Common mechanisms despite different contexts
    shared_mechanisms: List[str] = field(default_factory=list)
    common_pathways: List[str] = field(default_factory=list)
    universal_factors: List[str] = field(default_factory=list)
    
    # Evidence support
    supporting_evidence: Dict[str, List[str]] = field(default_factory=dict)
    confidence_level: float = 0.0


class MSS_MDS_Analyzer:
    """
    Analyzer for Most Similar Systems and Most Different Systems comparisons.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize MSS/MDS analyzer.
        
        Args:
            similarity_threshold: Threshold for considering cases similar
        """
        self.similarity_threshold = similarity_threshold
        self.context_weight = 0.4
        self.structural_weight = 0.3
        self.temporal_weight = 0.2
        self.institutional_weight = 0.1
        
        self.logger = logging.getLogger(__name__)
        self.graph_aligner = GraphAligner(similarity_threshold)
        self.mechanism_detector = MechanismDetector(similarity_threshold)
        
        # Issue #17 Fix: Statistical significance testing parameters
        self.significance_level = 0.05  # Alpha level for hypothesis testing
        self.min_sample_size = 3  # Minimum cases needed for statistical testing
    
    def calculate_statistical_significance(self, mechanism_occurrences: List[List[float]], 
                                         case_similarities: List[float]) -> Dict[str, Any]:
        """
        Calculate statistical significance of mechanism patterns across cases.
        Issue #17 Fix: Add proper significance testing for cross-case statistics.
        
        Args:
            mechanism_occurrences: List of mechanism occurrence rates per case
            case_similarities: List of case similarity scores
            
        Returns:
            Dictionary with statistical test results
        """
        if len(mechanism_occurrences) < self.min_sample_size:
            return {
                'test_performed': False,
                'reason': f'Insufficient sample size (n={len(mechanism_occurrences)}, min={self.min_sample_size})',
                'p_value': None,
                'significant': False
            }
        
        try:
            # Convert to numpy arrays
            occurrences = np.array(mechanism_occurrences)
            similarities = np.array(case_similarities)
            
            # Perform correlation test between mechanism occurrence and case similarity
            if len(occurrences.shape) > 1:
                # Multiple mechanisms - use mean occurrence rate
                mean_occurrences = np.mean(occurrences, axis=1)
            else:
                mean_occurrences = occurrences
            
            # Pearson correlation coefficient and p-value
            correlation, p_value = stats.pearsonr(mean_occurrences, similarities)
            
            # Chi-square test for categorical data (high vs low occurrence)
            median_occurrence = np.median(mean_occurrences)
            median_similarity = np.median(similarities)
            
            high_occ_high_sim = np.sum((mean_occurrences > median_occurrence) & (similarities > median_similarity))
            high_occ_low_sim = np.sum((mean_occurrences > median_occurrence) & (similarities <= median_similarity))
            low_occ_high_sim = np.sum((mean_occurrences <= median_occurrence) & (similarities > median_similarity))
            low_occ_low_sim = np.sum((mean_occurrences <= median_occurrence) & (similarities <= median_similarity))
            
            contingency_table = np.array([[high_occ_high_sim, high_occ_low_sim],
                                        [low_occ_high_sim, low_occ_low_sim]])
            
            # Avoid chi-square with zero cells
            if np.any(contingency_table == 0):
                chi2_stat, chi2_p = None, None
            else:
                chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency_table)
            
            return {
                'test_performed': True,
                'n_cases': len(mechanism_occurrences),
                'correlation': {
                    'coefficient': correlation,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level if p_value is not None else False
                },
                'chi_square': {
                    'statistic': chi2_stat,
                    'p_value': chi2_p,
                    'significant': chi2_p < self.significance_level if chi2_p is not None else False,
                    'contingency_table': contingency_table.tolist()
                },
                'effect_size': abs(correlation) if correlation is not None else 0.0,
                'confidence_interval': self._calculate_correlation_ci(correlation, len(mechanism_occurrences)) if correlation is not None else None
            }
            
        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return {
                'test_performed': False,
                'reason': f'Calculation error: {str(e)}',
                'p_value': None,
                'significant': False
            }
    
    def _calculate_correlation_ci(self, correlation: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        if n < 3:
            return (correlation, correlation)
        
        # Fisher's z-transformation
        z = np.arctanh(correlation)
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_critical / np.sqrt(n - 3)
        
        # Transform back to correlation space
        lower = np.tanh(z - margin_error)
        upper = np.tanh(z + margin_error)
        
        return (lower, upper)
    
    def conduct_mss_analysis(self, graphs: Dict[str, nx.DiGraph], 
                           metadata: Dict[str, CaseMetadata],
                           mechanisms: List[MechanismPattern]) -> List[MSS_Analysis]:
        """
        Conduct Most Similar Systems analysis.
        
        Args:
            graphs: Dictionary of case_id -> graph
            metadata: Dictionary of case_id -> metadata
            mechanisms: Identified mechanism patterns
            
        Returns:
            List of MSS analysis results
        """
        self.logger.info("Conducting Most Similar Systems (MSS) analysis")
        
        mss_results = []
        case_ids = list(graphs.keys())
        
        # Find similar context pairs with different outcomes
        similar_pairs = self._find_similar_context_pairs(metadata)
        
        for case1, case2 in similar_pairs:
            if case1 not in graphs or case2 not in graphs:
                continue
                
            mss_analysis = self._analyze_mss_pair(
                case1, case2, graphs, metadata, mechanisms
            )
            if mss_analysis:
                mss_results.append(mss_analysis)
        
        self.logger.info(f"Completed MSS analysis for {len(mss_results)} case pairs")
        return mss_results
    
    def conduct_mds_analysis(self, graphs: Dict[str, nx.DiGraph], 
                           metadata: Dict[str, CaseMetadata],
                           mechanisms: List[MechanismPattern]) -> List[MDS_Analysis]:
        """
        Conduct Most Different Systems analysis.
        
        Args:
            graphs: Dictionary of case_id -> graph
            metadata: Dictionary of case_id -> metadata
            mechanisms: Identified mechanism patterns
            
        Returns:
            List of MDS analysis results
        """
        self.logger.info("Conducting Most Different Systems (MDS) analysis")
        
        mds_results = []
        case_ids = list(graphs.keys())
        
        # Find different context pairs with similar outcomes
        different_pairs = self._find_different_context_pairs(metadata)
        
        for case1, case2 in different_pairs:
            if case1 not in graphs or case2 not in graphs:
                continue
                
            mds_analysis = self._analyze_mds_pair(
                case1, case2, graphs, metadata, mechanisms
            )
            if mds_analysis:
                mds_results.append(mds_analysis)
        
        self.logger.info(f"Completed MDS analysis for {len(mds_results)} case pairs")
        return mds_results
    
    def compare_mss_mds_insights(self, mss_results: List[MSS_Analysis], 
                               mds_results: List[MDS_Analysis]) -> Dict[str, Any]:
        """
        Compare insights from MSS and MDS analyses.
        
        Args:
            mss_results: MSS analysis results
            mds_results: MDS analysis results
            
        Returns:
            Comparative insights summary
        """
        insights = {
            'mss_insights': self._extract_mss_insights(mss_results),
            'mds_insights': self._extract_mds_insights(mds_results),
            'convergent_findings': self._find_convergent_findings(mss_results, mds_results),
            'theoretical_implications': self._assess_theoretical_implications(mss_results, mds_results)
        }
        
        return insights
    
    def _find_similar_context_pairs(self, metadata: Dict[str, CaseMetadata]) -> List[Tuple[str, str]]:
        """
        Find case pairs with similar contexts for MSS analysis.
        
        Args:
            metadata: Case metadata dictionary
            
        Returns:
            List of case pairs with similar contexts
        """
        similar_pairs = []
        case_ids = list(metadata.keys())
        
        for i, case1 in enumerate(case_ids):
            for case2 in case_ids[i+1:]:
                context_sim = self._calculate_context_similarity(
                    metadata[case1], metadata[case2]
                )
                outcome_diff = self._calculate_outcome_difference(
                    metadata[case1], metadata[case2]
                )
                
                # MSS criteria: similar context, different outcomes
                if context_sim >= self.similarity_threshold and outcome_diff >= 0.5:
                    similar_pairs.append((case1, case2))
        
        return similar_pairs
    
    def _find_different_context_pairs(self, metadata: Dict[str, CaseMetadata]) -> List[Tuple[str, str]]:
        """
        Find case pairs with different contexts for MDS analysis.
        
        Args:
            metadata: Case metadata dictionary
            
        Returns:
            List of case pairs with different contexts
        """
        different_pairs = []
        case_ids = list(metadata.keys())
        
        for i, case1 in enumerate(case_ids):
            for case2 in case_ids[i+1:]:
                context_sim = self._calculate_context_similarity(
                    metadata[case1], metadata[case2]
                )
                outcome_sim = self._calculate_outcome_similarity(
                    metadata[case1], metadata[case2]
                )
                
                # MDS criteria: different context, similar outcomes
                if context_sim <= (1.0 - self.similarity_threshold) and outcome_sim >= self.similarity_threshold:
                    different_pairs.append((case1, case2))
        
        return different_pairs
    
    def _analyze_mss_pair(self, case1: str, case2: str, 
                         graphs: Dict[str, nx.DiGraph],
                         metadata: Dict[str, CaseMetadata],
                         mechanisms: List[MechanismPattern]) -> Optional[MSS_Analysis]:
        """
        Analyze a specific MSS case pair.
        
        Args:
            case1: First case ID
            case2: Second case ID
            graphs: Case graphs
            metadata: Case metadata
            mechanisms: Mechanism patterns
            
        Returns:
            MSS analysis result or None
        """
        graph1 = graphs[case1]
        graph2 = graphs[case2]
        meta1 = metadata[case1]
        meta2 = metadata[case2]
        
        # Calculate similarities and differences
        context_similarity = self._calculate_context_similarity(meta1, meta2)
        outcome_difference = self._calculate_outcome_difference(meta1, meta2)
        
        # Identify differentiating mechanisms
        differentiating_mechanisms = self._find_differentiating_mechanisms(
            case1, case2, mechanisms
        )
        
        # Find unique pathways
        unique_pathways = self._find_unique_pathways(
            case1, case2, graph1, graph2
        )
        
        # Identify critical differences
        critical_differences = self._identify_critical_differences(
            graph1, graph2, meta1, meta2
        )
        
        # Gather supporting evidence
        supporting_evidence = self._gather_mss_evidence(
            case1, case2, differentiating_mechanisms, graphs
        )
        
        # Calculate confidence
        confidence = self._calculate_mss_confidence(
            context_similarity, outcome_difference, len(differentiating_mechanisms)
        )
        
        mss_analysis = MSS_Analysis(
            comparison_id=f"mss_{case1}_{case2}",
            primary_case=case1,
            comparison_case=case2,
            context_similarity=context_similarity,
            outcome_difference=outcome_difference,
            differentiating_mechanisms=differentiating_mechanisms,
            unique_pathways=unique_pathways,
            critical_differences=critical_differences,
            supporting_evidence=supporting_evidence,
            confidence_level=confidence
        )
        
        return mss_analysis
    
    def _analyze_mds_pair(self, case1: str, case2: str, 
                         graphs: Dict[str, nx.DiGraph],
                         metadata: Dict[str, CaseMetadata],
                         mechanisms: List[MechanismPattern]) -> Optional[MDS_Analysis]:
        """
        Analyze a specific MDS case pair.
        
        Args:
            case1: First case ID
            case2: Second case ID
            graphs: Case graphs
            metadata: Case metadata
            mechanisms: Mechanism patterns
            
        Returns:
            MDS analysis result or None
        """
        graph1 = graphs[case1]
        graph2 = graphs[case2]
        meta1 = metadata[case1]
        meta2 = metadata[case2]
        
        # Calculate similarities and differences
        context_difference = 1.0 - self._calculate_context_similarity(meta1, meta2)
        outcome_similarity = self._calculate_outcome_similarity(meta1, meta2)
        
        # Identify shared mechanisms
        shared_mechanisms = self._find_shared_mechanisms(
            case1, case2, mechanisms
        )
        
        # Find common pathways
        common_pathways = self._find_common_pathways(
            case1, case2, graph1, graph2
        )
        
        # Identify universal factors
        universal_factors = self._identify_universal_factors(
            graph1, graph2, shared_mechanisms
        )
        
        # Gather supporting evidence
        supporting_evidence = self._gather_mds_evidence(
            case1, case2, shared_mechanisms, graphs
        )
        
        # Calculate confidence
        confidence = self._calculate_mds_confidence(
            context_difference, outcome_similarity, len(shared_mechanisms)
        )
        
        mds_analysis = MDS_Analysis(
            comparison_id=f"mds_{case1}_{case2}",
            primary_case=case1,
            comparison_case=case2,
            context_difference=context_difference,
            outcome_similarity=outcome_similarity,
            shared_mechanisms=shared_mechanisms,
            common_pathways=common_pathways,
            universal_factors=universal_factors,
            supporting_evidence=supporting_evidence,
            confidence_level=confidence
        )
        
        return mds_analysis
    
    def _calculate_context_similarity(self, meta1: CaseMetadata, meta2: CaseMetadata) -> float:
        """
        Calculate context similarity between two cases.
        
        Args:
            meta1: First case metadata
            meta2: Second case metadata
            
        Returns:
            Context similarity score (0.0-1.0)
        """
        similarities = []
        
        # Geographic context
        if meta1.geographic_context and meta2.geographic_context:
            geo_sim = 1.0 if meta1.geographic_context == meta2.geographic_context else 0.0
            similarities.append(geo_sim)
        
        # Institutional context
        if meta1.institutional_context and meta2.institutional_context:
            inst_sim = 1.0 if meta1.institutional_context == meta2.institutional_context else 0.0
            similarities.append(inst_sim)
        
        # Economic context
        if meta1.economic_context and meta2.economic_context:
            econ_sim = 1.0 if meta1.economic_context == meta2.economic_context else 0.0
            similarities.append(econ_sim)
        
        # Political context
        if meta1.political_context and meta2.political_context:
            pol_sim = 1.0 if meta1.political_context == meta2.political_context else 0.0
            similarities.append(pol_sim)
        
        # Social context
        if meta1.social_context and meta2.social_context:
            soc_sim = 1.0 if meta1.social_context == meta2.social_context else 0.0
            similarities.append(soc_sim)
        
        # Temporal proximity
        if meta1.time_period and meta2.time_period:
            time_sim = self._calculate_temporal_similarity(meta1.time_period, meta2.time_period)
            similarities.append(time_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def _calculate_outcome_difference(self, meta1: CaseMetadata, meta2: CaseMetadata) -> float:
        """
        Calculate outcome difference between two cases.
        
        Args:
            meta1: First case metadata
            meta2: Second case metadata
            
        Returns:
            Outcome difference score (0.0-1.0)
        """
        if not meta1.primary_outcome or not meta2.primary_outcome:
            return 0.5  # Unknown difference
        
        # Simple binary difference
        outcome_diff = 0.0 if meta1.primary_outcome == meta2.primary_outcome else 1.0
        
        # Consider outcome magnitude if available
        if meta1.outcome_magnitude is not None and meta2.outcome_magnitude is not None:
            magnitude_diff = abs(meta1.outcome_magnitude - meta2.outcome_magnitude)
            outcome_diff = max(outcome_diff, magnitude_diff)
        
        return outcome_diff
    
    def _calculate_outcome_similarity(self, meta1: CaseMetadata, meta2: CaseMetadata) -> float:
        """
        Calculate outcome similarity between two cases.
        
        Args:
            meta1: First case metadata
            meta2: Second case metadata
            
        Returns:
            Outcome similarity score (0.0-1.0)
        """
        return 1.0 - self._calculate_outcome_difference(meta1, meta2)
    
    def _calculate_temporal_similarity(self, period1: Tuple, period2: Tuple) -> float:
        """
        Calculate temporal similarity between time periods.
        
        Args:
            period1: First time period
            period2: Second time period
            
        Returns:
            Temporal similarity score (0.0-1.0)
        """
        try:
            start1, end1 = period1
            start2, end2 = period2
            
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).days
                total_duration = max((end1 - start1).days, (end2 - start2).days)
                return min(1.0, overlap_duration / total_duration) if total_duration > 0 else 1.0
            else:
                # No overlap - calculate proximity
                gap = min(abs((start1 - end2).days), abs((start2 - end1).days))
                # Similarity decreases with gap size (arbitrary scaling)
                return max(0.0, 1.0 - gap / 365.0)  # 1 year = 0 similarity
        except:
            return 0.5  # Default for unparseable dates
    
    def _find_differentiating_mechanisms(self, case1: str, case2: str, 
                                       mechanisms: List[MechanismPattern]) -> List[str]:
        """
        Find mechanisms that differentiate between two cases.
        
        Args:
            case1: First case ID
            case2: Second case ID
            mechanisms: Mechanism patterns
            
        Returns:
            List of differentiating mechanism IDs
        """
        differentiating = []
        
        for mechanism in mechanisms:
            case1_present = case1 in mechanism.participating_cases
            case2_present = case2 in mechanism.participating_cases
            
            # Mechanism present in one case but not the other
            if case1_present != case2_present:
                differentiating.append(mechanism.pattern_id)
        
        return differentiating
    
    def _find_shared_mechanisms(self, case1: str, case2: str, 
                              mechanisms: List[MechanismPattern]) -> List[str]:
        """
        Find mechanisms shared between two cases.
        
        Args:
            case1: First case ID
            case2: Second case ID
            mechanisms: Mechanism patterns
            
        Returns:
            List of shared mechanism IDs
        """
        shared = []
        
        for mechanism in mechanisms:
            case1_present = case1 in mechanism.participating_cases
            case2_present = case2 in mechanism.participating_cases
            
            # Mechanism present in both cases
            if case1_present and case2_present:
                shared.append(mechanism.pattern_id)
        
        return shared
    
    def _find_unique_pathways(self, case1: str, case2: str, 
                            graph1: nx.DiGraph, graph2: nx.DiGraph) -> Dict[str, List[str]]:
        """
        Find unique causal pathways in each case.
        
        Args:
            case1: First case ID
            case2: Second case ID
            graph1: First case graph
            graph2: Second case graph
            
        Returns:
            Dictionary of case -> unique pathway descriptions
        """
        unique_pathways = {case1: [], case2: []}
        
        # Simple pathway detection (could be enhanced)
        # Find paths from source nodes to sink nodes
        
        sources1 = [n for n in graph1.nodes() if graph1.in_degree(n) == 0]
        sinks1 = [n for n in graph1.nodes() if graph1.out_degree(n) == 0]
        
        sources2 = [n for n in graph2.nodes() if graph2.in_degree(n) == 0]
        sinks2 = [n for n in graph2.nodes() if graph2.out_degree(n) == 0]
        
        # Get all simple paths
        paths1 = []
        for source in sources1:
            for sink in sinks1:
                try:
                    paths = list(nx.all_simple_paths(graph1, source, sink, cutoff=5))
                    paths1.extend(paths)
                except:
                    pass
        
        paths2 = []
        for source in sources2:
            for sink in sinks2:
                try:
                    paths = list(nx.all_simple_paths(graph2, source, sink, cutoff=5))
                    paths2.extend(paths)
                except:
                    pass
        
        # Identify unique patterns (simplified)
        path_patterns1 = set()
        for path in paths1:
            pattern = " -> ".join([graph1.nodes[n].get('type', 'Unknown') for n in path])
            path_patterns1.add(pattern)
        
        path_patterns2 = set()
        for path in paths2:
            pattern = " -> ".join([graph2.nodes[n].get('type', 'Unknown') for n in path])
            path_patterns2.add(pattern)
        
        unique_pathways[case1] = list(path_patterns1 - path_patterns2)
        unique_pathways[case2] = list(path_patterns2 - path_patterns1)
        
        return unique_pathways
    
    def _find_common_pathways(self, case1: str, case2: str, 
                            graph1: nx.DiGraph, graph2: nx.DiGraph) -> List[str]:
        """
        Find common causal pathways between two cases.
        
        Args:
            case1: First case ID
            case2: Second case ID
            graph1: First case graph
            graph2: Second case graph
            
        Returns:
            List of common pathway descriptions
        """
        # Use the same logic as unique pathways but find intersection
        unique_pathways = self._find_unique_pathways(case1, case2, graph1, graph2)
        
        # Find patterns that appear in both cases
        all_patterns1 = set()
        all_patterns2 = set()
        
        # Get all patterns from each case (need to recalculate to include shared ones)
        sources1 = [n for n in graph1.nodes() if graph1.in_degree(n) == 0]
        sinks1 = [n for n in graph1.nodes() if graph1.out_degree(n) == 0]
        
        for source in sources1:
            for sink in sinks1:
                try:
                    paths = list(nx.all_simple_paths(graph1, source, sink, cutoff=5))
                    for path in paths:
                        pattern = " -> ".join([graph1.nodes[n].get('type', 'Unknown') for n in path])
                        all_patterns1.add(pattern)
                except:
                    pass
        
        sources2 = [n for n in graph2.nodes() if graph2.in_degree(n) == 0]
        sinks2 = [n for n in graph2.nodes() if graph2.out_degree(n) == 0]
        
        for source in sources2:
            for sink in sinks2:
                try:
                    paths = list(nx.all_simple_paths(graph2, source, sink, cutoff=5))
                    for path in paths:
                        pattern = " -> ".join([graph2.nodes[n].get('type', 'Unknown') for n in path])
                        all_patterns2.add(pattern)
                except:
                    pass
        
        common_patterns = all_patterns1 & all_patterns2
        return list(common_patterns)
    
    def _identify_critical_differences(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                                     meta1: CaseMetadata, meta2: CaseMetadata) -> List[str]:
        """
        Identify critical differences between two cases.
        
        Args:
            graph1: First case graph
            graph2: Second case graph
            meta1: First case metadata
            meta2: Second case metadata
            
        Returns:
            List of critical difference descriptions
        """
        differences = []
        
        # Structural differences
        if graph1.number_of_nodes() != graph2.number_of_nodes():
            differences.append(f"Node count difference: {graph1.number_of_nodes()} vs {graph2.number_of_nodes()}")
        
        if graph1.number_of_edges() != graph2.number_of_edges():
            differences.append(f"Edge count difference: {graph1.number_of_edges()} vs {graph2.number_of_edges()}")
        
        # Context differences
        if meta1.institutional_context != meta2.institutional_context:
            differences.append(f"Institutional context: {meta1.institutional_context} vs {meta2.institutional_context}")
        
        if meta1.political_context != meta2.political_context:
            differences.append(f"Political context: {meta1.political_context} vs {meta2.political_context}")
        
        return differences
    
    def _identify_universal_factors(self, graph1: nx.DiGraph, graph2: nx.DiGraph,
                                  shared_mechanisms: List[str]) -> List[str]:
        """
        Identify universal factors present in both cases.
        
        Args:
            graph1: First case graph
            graph2: Second case graph
            shared_mechanisms: List of shared mechanism IDs
            
        Returns:
            List of universal factor descriptions
        """
        universal_factors = []
        
        # Node type analysis
        types1 = set(data.get('type', 'Unknown') for _, data in graph1.nodes(data=True))
        types2 = set(data.get('type', 'Unknown') for _, data in graph2.nodes(data=True))
        common_types = types1 & types2
        
        for node_type in common_types:
            universal_factors.append(f"Common node type: {node_type}")
        
        # Edge type analysis
        edge_types1 = set(data.get('type', 'unknown') for _, _, data in graph1.edges(data=True))
        edge_types2 = set(data.get('type', 'unknown') for _, _, data in graph2.edges(data=True))
        common_edge_types = edge_types1 & edge_types2
        
        for edge_type in common_edge_types:
            universal_factors.append(f"Common relationship: {edge_type}")
        
        return universal_factors
    
    def _gather_mss_evidence(self, case1: str, case2: str, 
                           differentiating_mechanisms: List[str],
                           graphs: Dict[str, nx.DiGraph]) -> Dict[str, List[str]]:
        """
        Gather evidence supporting MSS analysis.
        
        Args:
            case1: First case ID
            case2: Second case ID
            differentiating_mechanisms: List of differentiating mechanisms
            graphs: Case graphs
            
        Returns:
            Dictionary of evidence by case
        """
        evidence = {case1: [], case2: []}
        
        # Extract evidence from graph node descriptions
        for mechanism_id in differentiating_mechanisms:
            # This is simplified - in practice would link to actual mechanism patterns
            evidence[case1].append(f"Unique mechanism {mechanism_id} present")
            evidence[case2].append(f"Mechanism {mechanism_id} absent")
        
        return evidence
    
    def _gather_mds_evidence(self, case1: str, case2: str, 
                           shared_mechanisms: List[str],
                           graphs: Dict[str, nx.DiGraph]) -> Dict[str, List[str]]:
        """
        Gather evidence supporting MDS analysis.
        
        Args:
            case1: First case ID
            case2: Second case ID
            shared_mechanisms: List of shared mechanisms
            graphs: Case graphs
            
        Returns:
            Dictionary of evidence by case
        """
        evidence = {case1: [], case2: []}
        
        # Extract evidence from shared mechanisms
        for mechanism_id in shared_mechanisms:
            evidence[case1].append(f"Shared mechanism {mechanism_id} present")
            evidence[case2].append(f"Shared mechanism {mechanism_id} present")
        
        return evidence
    
    def _calculate_mss_confidence(self, context_similarity: float, 
                                outcome_difference: float, 
                                num_differentiating: int) -> float:
        """
        Calculate confidence in MSS analysis.
        
        Args:
            context_similarity: Context similarity score
            outcome_difference: Outcome difference score
            num_differentiating: Number of differentiating mechanisms
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Higher confidence with higher context similarity, outcome difference, and more mechanisms
        context_factor = context_similarity
        outcome_factor = outcome_difference
        mechanism_factor = min(1.0, num_differentiating / 3.0)  # Normalize around 3 mechanisms
        
        confidence = (0.4 * context_factor + 0.4 * outcome_factor + 0.2 * mechanism_factor)
        return min(1.0, max(0.0, confidence))
    
    def _calculate_mds_confidence(self, context_difference: float, 
                                outcome_similarity: float, 
                                num_shared: int) -> float:
        """
        Calculate confidence in MDS analysis.
        
        Args:
            context_difference: Context difference score
            outcome_similarity: Outcome similarity score
            num_shared: Number of shared mechanisms
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Higher confidence with higher context difference, outcome similarity, and more shared mechanisms
        context_factor = context_difference
        outcome_factor = outcome_similarity
        mechanism_factor = min(1.0, num_shared / 3.0)  # Normalize around 3 mechanisms
        
        confidence = (0.4 * context_factor + 0.4 * outcome_factor + 0.2 * mechanism_factor)
        return min(1.0, max(0.0, confidence))
    
    def _extract_mss_insights(self, mss_results: List[MSS_Analysis]) -> Dict[str, Any]:
        """
        Extract insights from MSS analyses.
        
        Args:
            mss_results: MSS analysis results
            
        Returns:
            MSS insights summary
        """
        if not mss_results:
            return {}
        
        # Collect differentiating mechanisms
        all_differentiating = []
        for result in mss_results:
            all_differentiating.extend(result.differentiating_mechanisms)
        
        mechanism_frequency = {}
        for mechanism in all_differentiating:
            mechanism_frequency[mechanism] = mechanism_frequency.get(mechanism, 0) + 1
        
        # Sort by frequency
        key_differentiators = sorted(mechanism_frequency.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        return {
            'total_comparisons': len(mss_results),
            'avg_confidence': sum(r.confidence_level for r in mss_results) / len(mss_results),
            'key_differentiating_mechanisms': key_differentiators[:5],
            'avg_context_similarity': sum(r.context_similarity for r in mss_results) / len(mss_results),
            'avg_outcome_difference': sum(r.outcome_difference for r in mss_results) / len(mss_results)
        }
    
    def _extract_mds_insights(self, mds_results: List[MDS_Analysis]) -> Dict[str, Any]:
        """
        Extract insights from MDS analyses.
        
        Args:
            mds_results: MDS analysis results
            
        Returns:
            MDS insights summary
        """
        if not mds_results:
            return {}
        
        # Collect shared mechanisms
        all_shared = []
        for result in mds_results:
            all_shared.extend(result.shared_mechanisms)
        
        mechanism_frequency = {}
        for mechanism in all_shared:
            mechanism_frequency[mechanism] = mechanism_frequency.get(mechanism, 0) + 1
        
        # Sort by frequency
        key_shared = sorted(mechanism_frequency.items(), 
                          key=lambda x: x[1], reverse=True)
        
        return {
            'total_comparisons': len(mds_results),
            'avg_confidence': sum(r.confidence_level for r in mds_results) / len(mds_results),
            'key_shared_mechanisms': key_shared[:5],
            'avg_context_difference': sum(r.context_difference for r in mds_results) / len(mds_results),
            'avg_outcome_similarity': sum(r.outcome_similarity for r in mds_results) / len(mds_results)
        }
    
    def _find_convergent_findings(self, mss_results: List[MSS_Analysis], 
                                mds_results: List[MDS_Analysis]) -> List[str]:
        """
        Find convergent findings between MSS and MDS analyses.
        
        Args:
            mss_results: MSS analysis results
            mds_results: MDS analysis results
            
        Returns:
            List of convergent findings
        """
        convergent = []
        
        # Find mechanisms that appear as both differentiating (MSS) and shared (MDS)
        mss_mechanisms = set()
        for result in mss_results:
            mss_mechanisms.update(result.differentiating_mechanisms)
        
        mds_mechanisms = set()
        for result in mds_results:
            mds_mechanisms.update(result.shared_mechanisms)
        
        # Look for mechanisms that play different roles
        overlapping = mss_mechanisms & mds_mechanisms
        if overlapping:
            convergent.append(f"Mechanisms with context-dependent roles: {list(overlapping)}")
        
        return convergent
    
    def _assess_theoretical_implications(self, mss_results: List[MSS_Analysis], 
                                       mds_results: List[MDS_Analysis]) -> List[str]:
        """
        Assess theoretical implications of combined MSS/MDS analyses.
        
        Args:
            mss_results: MSS analysis results
            mds_results: MDS analysis results
            
        Returns:
            List of theoretical implications
        """
        implications = []
        
        if mss_results:
            implications.append(f"MSS analysis identifies {len(set().union(*[r.differentiating_mechanisms for r in mss_results]))} key differentiating mechanisms")
        
        if mds_results:
            implications.append(f"MDS analysis identifies {len(set().union(*[r.shared_mechanisms for r in mds_results]))} universal mechanisms")
        
        if mss_results and mds_results:
            avg_mss_confidence = sum(r.confidence_level for r in mss_results) / len(mss_results)
            avg_mds_confidence = sum(r.confidence_level for r in mds_results) / len(mds_results)
            
            if avg_mss_confidence > 0.7 and avg_mds_confidence > 0.7:
                implications.append("Strong evidence for both context-dependent and universal causal mechanisms")
        
        return implications


def test_mss_mds_analyzer():
    """Test function for MSS/MDS analyzer"""
    # Create test data
    from core.comparative_models import CaseMetadata, MechanismPattern, MechanismType
    from datetime import datetime
    
    # Test metadata
    meta1 = CaseMetadata(
        case_id="case1",
        case_name="Case 1",
        description="Test case 1",
        geographic_context="Europe",
        institutional_context="Democratic",
        primary_outcome="positive",
        outcome_magnitude=0.8
    )
    
    meta2 = CaseMetadata(
        case_id="case2", 
        case_name="Case 2",
        description="Test case 2",
        geographic_context="Europe",  # Similar context
        institutional_context="Democratic",  # Similar context
        primary_outcome="negative",  # Different outcome
        outcome_magnitude=0.2
    )
    
    metadata = {"case1": meta1, "case2": meta2}
    
    # Test graphs
    graph1 = nx.DiGraph()
    graph1.add_node("event1", type="Event", description="Crisis")
    graph1.add_node("response1", type="Event", description="Policy response")
    graph1.add_edge("event1", "response1", type="causes")
    
    graph2 = nx.DiGraph()
    graph2.add_node("event2", type="Event", description="Crisis")
    graph2.add_node("failure2", type="Event", description="Policy failure")
    graph2.add_edge("event2", "failure2", type="causes")
    
    graphs = {"case1": graph1, "case2": graph2}
    
    # Test mechanisms
    mechanism = MechanismPattern(
        pattern_id="test_mechanism",
        pattern_name="Test Pattern",
        description="Test mechanism",
        mechanism_type=MechanismType.CONDITIONAL,
        scope_conditions=[],
        participating_cases=["case1"]  # Only in case1
    )
    
    mechanisms = [mechanism]
    
    # Test analyzer
    analyzer = MSS_MDS_Analyzer()
    
    # Test MSS analysis
    mss_results = analyzer.conduct_mss_analysis(graphs, metadata, mechanisms)
    print(f"MSS Results: {len(mss_results)}")
    for result in mss_results:
        print(f"  {result.primary_case} vs {result.comparison_case}")
        print(f"  Context similarity: {result.context_similarity:.2f}")
        print(f"  Outcome difference: {result.outcome_difference:.2f}")
        print(f"  Differentiating mechanisms: {result.differentiating_mechanisms}")
        print(f"  Confidence: {result.confidence_level:.2f}")
    
    # Test MDS analysis (would need different test data)
    mds_results = analyzer.conduct_mds_analysis(graphs, metadata, mechanisms)
    print(f"MDS Results: {len(mds_results)}")
    
    # Test comparative insights
    insights = analyzer.compare_mss_mds_insights(mss_results, mds_results)
    print(f"Insights: {insights}")


if __name__ == "__main__":
    test_mss_mds_analyzer()