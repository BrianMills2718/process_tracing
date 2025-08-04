"""
Comparative Process Tracing - Case Management System

Manages multiple cases for comparative analysis including case loading,
metadata management, and case selection operations.

Author: Claude Code Implementation  
Date: August 2025
"""

import os
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime
import logging

from core.comparative_models import (
    CaseMetadata, CaseSelectionCriteria, ComparisonType, 
    ComparativeAnalysisError, create_default_case_metadata,
    validate_case_metadata
)


class CaseManager:
    """
    Manages multiple cases for comparative process tracing analysis.
    """
    
    def __init__(self, case_directory: Optional[str] = None):
        """
        Initialize case manager.
        
        Args:
            case_directory: Directory containing case files
        """
        self.case_directory = Path(case_directory) if case_directory else Path(".")
        self.cases: Dict[str, nx.DiGraph] = {}
        self.case_metadata: Dict[str, CaseMetadata] = {}
        self.case_files: Dict[str, str] = {}  # case_id -> file_path
        
        self.logger = logging.getLogger(__name__)
        
    def load_case(self, case_file: str, case_id: Optional[str] = None) -> str:
        """
        Load a single case from file.
        
        Args:
            case_file: Path to case JSON file
            case_id: Optional custom case ID (defaults to filename)
            
        Returns:
            Case ID of loaded case
            
        Raises:
            ComparativeAnalysisError: If case loading fails
        """
        case_path = Path(case_file)
        if not case_path.exists():
            raise ComparativeAnalysisError(f"Case file not found: {case_file}")
        
        # Generate case ID from filename if not provided
        if not case_id:
            case_id = case_path.stem
        
        try:
            # Load JSON data
            with open(case_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
            
            # Create NetworkX graph
            graph = self._json_to_graph(case_data)
            
            # Extract or create metadata
            metadata = self._extract_metadata(case_data, case_id, case_path.name)
            
            # Store case
            self.cases[case_id] = graph
            self.case_metadata[case_id] = metadata
            self.case_files[case_id] = str(case_path)
            
            self.logger.info(f"Loaded case '{case_id}' with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            return case_id
            
        except Exception as e:
            raise ComparativeAnalysisError(f"Failed to load case from {case_file}: {e}")
    
    def load_cases_from_directory(self, directory: Optional[str] = None, 
                                  pattern: str = "*.json") -> List[str]:
        """
        Load all cases from a directory.
        
        Args:
            directory: Directory to search (defaults to case_directory)
            pattern: File pattern to match
            
        Returns:
            List of loaded case IDs
        """
        search_dir = Path(directory) if directory else self.case_directory
        case_files = list(search_dir.glob(pattern))
        
        loaded_cases = []
        for case_file in case_files:
            try:
                case_id = self.load_case(str(case_file))
                loaded_cases.append(case_id)
            except ComparativeAnalysisError as e:
                self.logger.warning(f"Failed to load case from {case_file}: {e}")
        
        self.logger.info(f"Loaded {len(loaded_cases)} cases from {search_dir}")
        return loaded_cases
    
    def get_case(self, case_id: str) -> nx.DiGraph:
        """
        Get case graph by ID.
        
        Args:
            case_id: Case identifier
            
        Returns:
            NetworkX graph for the case
            
        Raises:
            ComparativeAnalysisError: If case not found
        """
        if case_id not in self.cases:
            raise ComparativeAnalysisError(f"Case not found: {case_id}")
        return self.cases[case_id]
    
    def get_case_metadata(self, case_id: str) -> CaseMetadata:
        """
        Get case metadata by ID.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Case metadata
            
        Raises:
            ComparativeAnalysisError: If case not found
        """
        if case_id not in self.case_metadata:
            raise ComparativeAnalysisError(f"Case metadata not found: {case_id}")
        return self.case_metadata[case_id]
    
    def update_case_metadata(self, case_id: str, metadata: CaseMetadata):
        """
        Update metadata for a case.
        
        Args:
            case_id: Case identifier
            metadata: Updated metadata
            
        Raises:
            ComparativeAnalysisError: If case not found
        """
        if case_id not in self.cases:
            raise ComparativeAnalysisError(f"Case not found: {case_id}")
        
        # Validate metadata
        warnings = validate_case_metadata(metadata)
        if warnings:
            self.logger.warning(f"Metadata validation warnings for case {case_id}: {warnings}")
        
        self.case_metadata[case_id] = metadata
        self.logger.info(f"Updated metadata for case '{case_id}'")
    
    def list_cases(self) -> List[str]:
        """
        Get list of all loaded case IDs.
        
        Returns:
            List of case IDs
        """
        return list(self.cases.keys())
    
    def get_case_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary information for all cases.
        
        Returns:
            Dictionary with case summaries
        """
        summary = {}
        for case_id in self.cases:
            graph = self.cases[case_id]
            metadata = self.case_metadata[case_id]
            
            summary[case_id] = {
                'name': metadata.case_name,
                'description': metadata.description[:100] + "..." if len(metadata.description) > 100 else metadata.description,
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'primary_outcome': metadata.primary_outcome,
                'time_period': metadata.time_period,
                'data_quality': metadata.data_quality_score,
                'file_path': self.case_files.get(case_id, 'Unknown')
            }
        
        return summary
    
    def select_cases(self, criteria: CaseSelectionCriteria) -> List[str]:
        """
        Select cases based on specified criteria.
        
        Args:
            criteria: Case selection criteria
            
        Returns:
            List of selected case IDs
        """
        selected_cases = []
        
        for case_id in self.cases:
            metadata = self.case_metadata[case_id]
            
            # Check inclusion criteria
            if not self._meets_inclusion_criteria(metadata, criteria):
                continue
            
            # Check exclusion criteria
            if self._meets_exclusion_criteria(metadata, criteria):
                continue
            
            selected_cases.append(case_id)
        
        # Apply case count limits
        if len(selected_cases) > criteria.maximum_case_count:
            # Simple selection strategy - could be enhanced with more sophisticated methods
            selected_cases = selected_cases[:criteria.maximum_case_count]
        
        self.logger.info(f"Selected {len(selected_cases)} cases based on criteria")
        return selected_cases
    
    def create_case_pairs(self, case_ids: List[str], 
                         comparison_type: ComparisonType) -> List[Tuple[str, str]]:
        """
        Create case pairs for comparison based on comparison type.
        
        Args:
            case_ids: List of case IDs to compare
            comparison_type: Type of comparison to perform
            
        Returns:
            List of case pairs for comparison
        """
        pairs = []
        
        if comparison_type == ComparisonType.MOST_SIMILAR_SYSTEMS:
            # Find pairs with similar contexts but different outcomes
            pairs = self._find_similar_context_pairs(case_ids)
        elif comparison_type == ComparisonType.MOST_DIFFERENT_SYSTEMS:
            # Find pairs with different contexts but similar outcomes
            pairs = self._find_different_context_pairs(case_ids)
        else:
            # All pairwise combinations for diverse case analysis
            for i, case1 in enumerate(case_ids):
                for case2 in case_ids[i+1:]:
                    pairs.append((case1, case2))
        
        return pairs
    
    def remove_case(self, case_id: str):
        """
        Remove a case from the manager.
        
        Args:
            case_id: Case identifier to remove
        """
        if case_id in self.cases:
            del self.cases[case_id]
        if case_id in self.case_metadata:
            del self.case_metadata[case_id]
        if case_id in self.case_files:
            del self.case_files[case_id]
        
        self.logger.info(f"Removed case '{case_id}'")
    
    def save_case_metadata(self, output_dir: str):
        """
        Save all case metadata to files.
        
        Args:
            output_dir: Directory to save metadata files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for case_id, metadata in self.case_metadata.items():
            # Convert metadata to dictionary for JSON serialization
            metadata_dict = {
                'case_id': metadata.case_id,
                'case_name': metadata.case_name,
                'description': metadata.description,
                'time_period': [dt.isoformat() for dt in metadata.time_period] if metadata.time_period else None,
                'duration': metadata.duration,
                'geographic_context': metadata.geographic_context,
                'institutional_context': metadata.institutional_context,
                'economic_context': metadata.economic_context,
                'political_context': metadata.political_context,
                'social_context': metadata.social_context,
                'primary_outcome': metadata.primary_outcome,
                'secondary_outcomes': metadata.secondary_outcomes,
                'outcome_magnitude': metadata.outcome_magnitude,
                'control_variables': metadata.control_variables,
                'scope_conditions': [sc.value for sc in metadata.scope_conditions],
                'data_quality_score': metadata.data_quality_score,
                'source_reliability': metadata.source_reliability,
                'evidence_completeness': metadata.evidence_completeness,
                'comparison_type': metadata.comparison_type.value if metadata.comparison_type else None,
                'reference_cases': metadata.reference_cases
            }
            
            metadata_file = output_path / f"{case_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved metadata for {len(self.case_metadata)} cases to {output_dir}")
    
    def _json_to_graph(self, case_data: Dict[str, Any]) -> nx.DiGraph:
        """
        Convert JSON case data to NetworkX graph.
        
        Args:
            case_data: Dictionary containing case data
            
        Returns:
            NetworkX directed graph
        """
        graph = nx.DiGraph()
        
        # Add nodes
        for node_data in case_data.get('nodes', []):
            node_id = node_data.get('id')
            if node_id:
                # Copy all node attributes
                node_attrs = {k: v for k, v in node_data.items() if k != 'id'}
                graph.add_node(node_id, **node_attrs)
        
        # Add edges
        for edge_data in case_data.get('edges', []):
            source = edge_data.get('source')
            target = edge_data.get('target')
            if source and target:
                # Copy all edge attributes
                edge_attrs = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
                graph.add_edge(source, target, **edge_attrs)
        
        return graph
    
    def _extract_metadata(self, case_data: Dict[str, Any], 
                         case_id: str, case_name: str) -> CaseMetadata:
        """
        Extract metadata from case data or create default.
        
        Args:
            case_data: Dictionary containing case data
            case_id: Case identifier
            case_name: Case name
            
        Returns:
            Case metadata
        """
        metadata_dict = case_data.get('metadata', {})
        
        # Extract time period
        time_period = None
        if 'time_period' in metadata_dict and metadata_dict['time_period']:
            try:
                start_str, end_str = metadata_dict['time_period']
                start_dt = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                time_period = (start_dt, end_dt)
            except:
                pass
        
        # Create metadata object
        metadata = CaseMetadata(
            case_id=case_id,
            case_name=metadata_dict.get('case_name', case_name),
            description=metadata_dict.get('description', f"Case: {case_name}"),
            time_period=time_period,
            duration=metadata_dict.get('duration'),
            geographic_context=metadata_dict.get('geographic_context'),
            institutional_context=metadata_dict.get('institutional_context'),
            economic_context=metadata_dict.get('economic_context'),
            political_context=metadata_dict.get('political_context'),
            social_context=metadata_dict.get('social_context'),
            primary_outcome=metadata_dict.get('primary_outcome'),
            secondary_outcomes=metadata_dict.get('secondary_outcomes', []),
            outcome_magnitude=metadata_dict.get('outcome_magnitude'),
            control_variables=metadata_dict.get('control_variables', {}),
            data_quality_score=metadata_dict.get('data_quality_score', 0.7),
            source_reliability=metadata_dict.get('source_reliability', 0.7),
            evidence_completeness=metadata_dict.get('evidence_completeness', 0.7)
        )
        
        return metadata
    
    def _meets_inclusion_criteria(self, metadata: CaseMetadata, 
                                 criteria: CaseSelectionCriteria) -> bool:
        """
        Check if case meets inclusion criteria.
        
        Args:
            metadata: Case metadata
            criteria: Selection criteria
            
        Returns:
            True if case meets inclusion criteria
        """
        # Check outcome type
        if criteria.required_outcome_type:
            if metadata.primary_outcome != criteria.required_outcome_type:
                return False
        
        # Check time period
        if criteria.required_time_period and metadata.time_period:
            required_start, required_end = criteria.required_time_period
            case_start, case_end = metadata.time_period
            if not (required_start <= case_start and case_end <= required_end):
                return False
        
        # Check data quality
        if metadata.data_quality_score < criteria.minimum_data_quality:
            return False
        
        # Check required context factors
        for factor in criteria.required_context_factors:
            context_value = getattr(metadata, f"{factor}_context", None)
            if not context_value:
                return False
        
        return True
    
    def _meets_exclusion_criteria(self, metadata: CaseMetadata, 
                                 criteria: CaseSelectionCriteria) -> bool:
        """
        Check if case meets exclusion criteria (should be excluded).
        
        Args:
            metadata: Case metadata
            criteria: Selection criteria
            
        Returns:
            True if case should be excluded
        """
        # Check excluded contexts
        for excluded_context in criteria.excluded_contexts:
            if (metadata.geographic_context == excluded_context or
                metadata.institutional_context == excluded_context or
                metadata.political_context == excluded_context):
                return True
        
        # Check excluded outcomes
        if metadata.primary_outcome in criteria.excluded_outcomes:
            return True
        
        return False
    
    def _find_similar_context_pairs(self, case_ids: List[str]) -> List[Tuple[str, str]]:
        """
        Find case pairs with similar contexts for MSS analysis.
        
        Args:
            case_ids: List of case IDs
            
        Returns:
            List of similar context pairs
        """
        pairs = []
        
        for i, case1 in enumerate(case_ids):
            metadata1 = self.case_metadata[case1]
            for case2 in case_ids[i+1:]:
                metadata2 = self.case_metadata[case2]
                
                # Calculate context similarity
                similarity = self._calculate_context_similarity(metadata1, metadata2)
                
                # Check for different outcomes
                outcome_different = metadata1.primary_outcome != metadata2.primary_outcome
                
                if similarity > 0.7 and outcome_different:
                    pairs.append((case1, case2))
        
        return pairs
    
    def _find_different_context_pairs(self, case_ids: List[str]) -> List[Tuple[str, str]]:
        """
        Find case pairs with different contexts for MDS analysis.
        
        Args:
            case_ids: List of case IDs
            
        Returns:
            List of different context pairs
        """
        pairs = []
        
        for i, case1 in enumerate(case_ids):
            metadata1 = self.case_metadata[case1]
            for case2 in case_ids[i+1:]:
                metadata2 = self.case_metadata[case2]
                
                # Calculate context similarity
                similarity = self._calculate_context_similarity(metadata1, metadata2)
                
                # Check for similar outcomes
                outcome_similar = metadata1.primary_outcome == metadata2.primary_outcome
                
                if similarity < 0.4 and outcome_similar:
                    pairs.append((case1, case2))
        
        return pairs
    
    def _calculate_context_similarity(self, metadata1: CaseMetadata, 
                                    metadata2: CaseMetadata) -> float:
        """
        Calculate context similarity between two cases.
        
        Args:
            metadata1: First case metadata
            metadata2: Second case metadata
            
        Returns:
            Similarity score (0.0-1.0)
        """
        similarities = []
        
        # Geographic context
        if metadata1.geographic_context and metadata2.geographic_context:
            if metadata1.geographic_context == metadata2.geographic_context:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        # Institutional context
        if metadata1.institutional_context and metadata2.institutional_context:
            if metadata1.institutional_context == metadata2.institutional_context:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        # Economic context
        if metadata1.economic_context and metadata2.economic_context:
            if metadata1.economic_context == metadata2.economic_context:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        # Political context
        if metadata1.political_context and metadata2.political_context:
            if metadata1.political_context == metadata2.political_context:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.5


def test_case_manager():
    """Test function for case manager"""
    import tempfile
    import json
    
    # Create test data
    test_case_data = {
        "nodes": [
            {"id": "event1", "type": "Event", "description": "Initial event"},
            {"id": "event2", "type": "Event", "description": "Follow-up event"}
        ],
        "edges": [
            {"source": "event1", "target": "event2", "type": "causes"}
        ],
        "metadata": {
            "case_name": "Test Case",
            "description": "A test case for validation",
            "primary_outcome": "positive_outcome",
            "data_quality_score": 0.8
        }
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_case_data, f)
        temp_file = f.name
    
    try:
        # Test case manager
        manager = CaseManager()
        case_id = manager.load_case(temp_file)
        
        print(f"Loaded case: {case_id}")
        print(f"Cases: {manager.list_cases()}")
        print(f"Summary: {manager.get_case_summary()}")
        
        # Test case selection
        criteria = CaseSelectionCriteria(
            selection_strategy="purposive",
            minimum_data_quality=0.5
        )
        selected = manager.select_cases(criteria)
        print(f"Selected cases: {selected}")
        
    finally:
        # Clean up
        os.unlink(temp_file)


if __name__ == "__main__":
    test_case_manager()