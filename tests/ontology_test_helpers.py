"""
Ontology Test Helper Module for Phase 25B Migration

Centralized helper class for test files to access ontology dynamically,
replacing hardcoded edge type references in test files.
"""

from core.ontology_manager import ontology_manager
from typing import List


class OntologyTestHelper:
    """Helper class for test files to access ontology dynamically."""
    
    @staticmethod
    def get_evidence_hypothesis_edges() -> List[str]:
        """
        Get all Evidence→Hypothesis edge types.
        
        Replaces hardcoded lists like:
        ['supports', 'tests_hypothesis', 'provides_evidence_for', 'updates_probability', 'weighs_evidence']
        
        Returns:
            List of valid edge types from Evidence to Hypothesis
        """
        return ontology_manager.get_evidence_hypothesis_edges()
    
    @staticmethod
    def get_supportive_edges() -> List[str]:
        """
        Get supportive edge types (edges that indicate support/agreement).
        
        Replaces hardcoded patterns like checking for 'supports' or 'provides_evidence_for'.
        
        Returns:
            List of edge types that indicate support/positive evidence
        """
        edges = ontology_manager.get_evidence_hypothesis_edges()
        # Filter for supportive patterns
        supportive = [e for e in edges if any(word in e for word in ['support', 'provide', 'confirm'])]
        
        # Ensure we include hypothesis testing edges as they can be supportive
        hypothesis_testing_edges = [e for e in edges if 'test' in e and e not in supportive]
        supportive.extend(hypothesis_testing_edges)
            
        return supportive
    
    @staticmethod
    def get_refuting_edges() -> List[str]:
        """
        Get refuting edge types (edges that indicate contradiction/disagreement).
        
        Replaces hardcoded lists like ['refutes', 'contradicts', 'challenges'].
        
        Returns:
            List of edge types that indicate refutation/negative evidence
        """
        # Get edges that have refuting semantics
        refuting_edges = ['refutes', 'contradicts', 'challenges', 'undermines', 'disproves_occurrence']
        
        # Filter to only include edges that actually exist in ontology
        all_edges = ontology_manager.get_all_edge_types()
        return [e for e in refuting_edges if e in all_edges]
    
    @staticmethod
    def get_van_evera_edges() -> List[str]:
        """
        Get edge types relevant to Van Evera diagnostic testing.
        
        Returns:
            List of edge types with diagnostic properties
        """
        return ontology_manager.get_van_evera_edges()
    
    @staticmethod
    def get_diagnostic_edges() -> List[str]:
        """
        Get all edge types with diagnostic_type property.
        
        Returns:
            List of edge types that support diagnostic testing
        """
        return ontology_manager.get_all_diagnostic_edge_types()
    
    @staticmethod
    def is_supportive_edge(edge_type: str) -> bool:
        """
        Check if an edge type indicates support.
        
        Args:
            edge_type: Edge type to check
            
        Returns:
            True if edge indicates support/positive evidence
        """
        return edge_type in OntologyTestHelper.get_supportive_edges()
    
    @staticmethod
    def is_refuting_edge(edge_type: str) -> bool:
        """
        Check if an edge type indicates refutation.
        
        Args:
            edge_type: Edge type to check
            
        Returns:
            True if edge indicates refutation/negative evidence
        """
        return edge_type in OntologyTestHelper.get_refuting_edges()
    
    @staticmethod
    def get_edge_types_for_test_data(source_type: str, target_type: str) -> List[str]:
        """
        Get valid edge types for test data creation.
        
        Args:
            source_type: Source node type
            target_type: Target node type
            
        Returns:
            List of valid edge types for this relationship
        """
        return ontology_manager.get_edge_types_for_relationship(source_type, target_type)
    
    @staticmethod
    def create_test_edge(source_id: str, target_id: str, edge_type: str, **properties) -> dict:
        """
        Create a test edge with proper structure.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID  
            edge_type: Edge type
            **properties: Additional edge properties
            
        Returns:
            Dictionary representing an edge
        """
        edge = {
            'source_id': source_id,
            'target_id': target_id,
            'type': edge_type,
            'properties': properties
        }
        
        return edge
    
    @staticmethod
    def get_sample_evidence_hypothesis_edges() -> List[dict]:
        """
        Get sample edges for Evidence→Hypothesis testing.
        
        Returns:
            List of sample edge dictionaries for testing
        """
        edge_types = OntologyTestHelper.get_evidence_hypothesis_edges()
        
        sample_edges = []
        for i, edge_type in enumerate(edge_types[:5]):  # Limit to first 5 for testing
            edge = OntologyTestHelper.create_test_edge(
                source_id=f'E{i+1}',
                target_id=f'H{i+1}',
                edge_type=edge_type,
                probative_value=0.7,
                diagnostic_type='general'
            )
            sample_edges.append(edge)
            
        return sample_edges


# Backwards compatibility aliases
def get_evidence_hypothesis_edges():
    """Backwards compatibility wrapper."""
    return OntologyTestHelper.get_evidence_hypothesis_edges()

def get_supportive_edges():
    """Backwards compatibility wrapper.""" 
    return OntologyTestHelper.get_supportive_edges()

def get_refuting_edges():
    """Backwards compatibility wrapper."""
    return OntologyTestHelper.get_refuting_edges()