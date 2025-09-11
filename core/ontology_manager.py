"""
OntologyManager: Centralized ontology query and validation system for Phase 25A refactoring.

This module provides a robust abstraction layer over the ontology definitions,
enabling dynamic queries and eliminating hardcoded edge type dependencies throughout
the codebase. It serves as the single source of truth for all ontology-related operations.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from core.ontology import NODE_TYPES, EDGE_TYPES, NODE_COLORS
import logging

logger = logging.getLogger(__name__)


class OntologyManager:
    """
    Centralized ontology query and validation system.
    
    This class provides methods to:
    - Query valid edge types for node type relationships
    - Retrieve Evidence→Hypothesis edges for Van Evera testing
    - Validate edges against ontology constraints
    - Get required/optional properties for edge types
    - Support migration from hardcoded edge lists to dynamic queries
    """
    
    def __init__(self):
        """Initialize the OntologyManager with loaded ontology definitions."""
        self.node_types = NODE_TYPES
        self.edge_types = EDGE_TYPES
        self.node_colors = NODE_COLORS
        self._build_lookup_tables()
        logger.info(f"OntologyManager initialized with {len(self.node_types)} node types and {len(self.edge_types)} edge types")
    
    def _build_lookup_tables(self):
        """
        Build efficient lookup tables for ontology queries.
        
        Creates:
        - edge_by_domain: Map from domain node type to edge types
        - edge_by_range: Map from range node type to edge types  
        - edge_by_pair: Map from (domain, range) pair to edge types
        """
        self.edge_by_domain = {}
        self.edge_by_range = {}
        self.edge_by_pair = {}
        
        for edge_type, config in self.edge_types.items():
            domains = config.get('domain', [])
            ranges = config.get('range', [])
            
            # Build domain lookup
            for domain in domains:
                if domain not in self.edge_by_domain:
                    self.edge_by_domain[domain] = set()
                self.edge_by_domain[domain].add(edge_type)
                
                # Build range lookup
                for range_type in ranges:
                    if range_type not in self.edge_by_range:
                        self.edge_by_range[range_type] = set()
                    self.edge_by_range[range_type].add(edge_type)
                    
                    # Build pair lookup
                    pair = (domain, range_type)
                    if pair not in self.edge_by_pair:
                        self.edge_by_pair[pair] = set()
                    self.edge_by_pair[pair].add(edge_type)
        
        logger.debug(f"Built lookup tables: {len(self.edge_by_domain)} domains, {len(self.edge_by_range)} ranges, {len(self.edge_by_pair)} pairs")
    
    def get_edge_types_for_relationship(self, source_type: str, target_type: str) -> List[str]:
        """
        Returns valid edge types for given node type pair.
        
        Args:
            source_type: Source node type (e.g., 'Evidence')
            target_type: Target node type (e.g., 'Hypothesis')
            
        Returns:
            List of valid edge types for this relationship
        """
        edge_types = list(self.edge_by_pair.get((source_type, target_type), []))
        logger.debug(f"Found {len(edge_types)} edge types for {source_type}→{target_type}: {edge_types}")
        return edge_types
    
    def get_evidence_hypothesis_edges(self) -> List[str]:
        """
        Returns all edge types that connect Evidence to Hypothesis.
        
        This is the primary method to replace hardcoded lists like:
        ['supports', 'provides_evidence_for', 'tests_hypothesis', 'updates_probability', 'weighs_evidence']
        
        Returns:
            List of edge types from Evidence to Hypothesis
        """
        edge_types = self.get_edge_types_for_relationship('Evidence', 'Hypothesis')
        logger.info(f"Evidence→Hypothesis edges: {edge_types}")
        return edge_types
    
    def get_van_evera_edges(self) -> List[str]:
        """
        Returns edge types relevant to Van Evera diagnostic tests.
        
        These are edges that have diagnostic properties like:
        - diagnostic_type (hoop, smoking_gun, etc.)
        - probative_value (0.0-1.0)
        - test_result (passed, failed, ambiguous)
        
        Returns:
            List of edge types with Van Evera diagnostic properties
        """
        van_evera_edges = []
        
        for edge_type in self.get_evidence_hypothesis_edges():
            properties = self.edge_types[edge_type].get('properties', {})
            # Check for Van Evera diagnostic properties
            if any(prop in properties for prop in ['diagnostic_type', 'probative_value', 'test_result']):
                van_evera_edges.append(edge_type)
        
        # Also check other edges that might have diagnostic properties
        for edge_type, config in self.edge_types.items():
            if edge_type not in van_evera_edges:
                properties = config.get('properties', {})
                if 'diagnostic_type' in properties:
                    van_evera_edges.append(edge_type)
        
        logger.info(f"Van Evera diagnostic edges: {van_evera_edges}")
        return van_evera_edges
    
    def validate_edge(self, edge: dict) -> Tuple[bool, Optional[str]]:
        """
        Validates edge against ontology constraints.
        
        Args:
            edge: Edge dictionary with 'type', 'source', 'target', 'properties'
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        edge_type = edge.get('type')
        
        # Check if edge type exists
        if edge_type not in self.edge_types:
            return False, f"Unknown edge type: {edge_type}"
        
        edge_config = self.edge_types[edge_type]
        
        # Validate properties
        edge_props = edge.get('properties', {})
        required_props = self.get_required_properties(edge_type)
        
        for prop in required_props:
            if prop not in edge_props:
                return False, f"Missing required property '{prop}' for edge type '{edge_type}'"
        
        # Validate property types and constraints
        for prop_name, prop_value in edge_props.items():
            if prop_name in edge_config.get('properties', {}):
                prop_config = edge_config['properties'][prop_name]
                
                # Check allowed values
                if 'allowed_values' in prop_config:
                    if prop_value not in prop_config['allowed_values']:
                        return False, f"Invalid value '{prop_value}' for property '{prop_name}'. Allowed: {prop_config['allowed_values']}"
                
                # Check min/max for numeric properties
                if prop_config.get('type') == 'float':
                    if 'min' in prop_config and prop_value < prop_config['min']:
                        return False, f"Value {prop_value} for '{prop_name}' below minimum {prop_config['min']}"
                    if 'max' in prop_config and prop_value > prop_config['max']:
                        return False, f"Value {prop_value} for '{prop_name}' above maximum {prop_config['max']}"
        
        return True, None
    
    def get_edge_properties(self, edge_type: str) -> dict:
        """
        Returns required/optional properties for edge type.
        
        Args:
            edge_type: Edge type name
            
        Returns:
            Dictionary of property definitions
        """
        if edge_type not in self.edge_types:
            logger.warning(f"Unknown edge type requested: {edge_type}")
            return {}
        return self.edge_types[edge_type].get('properties', {})
    
    def get_required_properties(self, edge_type: str) -> List[str]:
        """
        Returns list of required properties for an edge type.
        
        Args:
            edge_type: Edge type name
            
        Returns:
            List of required property names
        """
        properties = self.get_edge_properties(edge_type)
        required = [
            prop_name for prop_name, prop_config in properties.items()
            if prop_config.get('required', False)
        ]
        return required
    
    def get_edges_by_domain(self, domain: str) -> List[str]:
        """
        Returns all edge types that can originate from a given node type.
        
        Args:
            domain: Source node type
            
        Returns:
            List of edge types with this domain
        """
        return list(self.edge_by_domain.get(domain, []))
    
    def get_edges_by_range(self, range_type: str) -> List[str]:
        """
        Returns all edge types that can target a given node type.
        
        Args:
            range_type: Target node type
            
        Returns:
            List of edge types with this range
        """
        return list(self.edge_by_range.get(range_type, []))
    
    def get_all_diagnostic_edge_types(self) -> List[str]:
        """
        Returns all edge types that have diagnostic_type property.
        
        This includes edges used for Van Evera testing and other diagnostic assessments.
        
        Returns:
            List of edge types with diagnostic_type property
        """
        diagnostic_edges = []
        for edge_type, config in self.edge_types.items():
            properties = config.get('properties', {})
            if 'diagnostic_type' in properties:
                diagnostic_edges.append(edge_type)
        return diagnostic_edges
    
    def is_evidence_to_hypothesis_edge(self, edge_type: str) -> bool:
        """
        Checks if an edge type connects Evidence to Hypothesis.
        
        Args:
            edge_type: Edge type to check
            
        Returns:
            True if edge connects Evidence to Hypothesis
        """
        return edge_type in self.get_evidence_hypothesis_edges()
    
    def get_node_properties(self, node_type: str) -> dict:
        """
        Returns properties definition for a node type.
        
        Args:
            node_type: Node type name
            
        Returns:
            Dictionary of property definitions
        """
        if node_type not in self.node_types:
            logger.warning(f"Unknown node type requested: {node_type}")
            return {}
        return self.node_types[node_type].get('properties', {})
    
    def get_node_required_properties(self, node_type: str) -> List[str]:
        """
        Returns list of required properties for a node type.
        
        Args:
            node_type: Node type name
            
        Returns:
            List of required property names
        """
        properties = self.get_node_properties(node_type)
        required = [
            prop_name for prop_name, prop_config in properties.items()
            if prop_config.get('required', False)
        ]
        return required
    
    def get_edge_label(self, edge_type: str) -> str:
        """
        Returns the display label for an edge type.
        
        Args:
            edge_type: Edge type name
            
        Returns:
            Display label or edge type if no label defined
        """
        if edge_type not in self.edge_types:
            return edge_type
        return self.edge_types[edge_type].get('label', edge_type)
    
    def get_node_color(self, node_type: str) -> str:
        """
        Returns the color for a node type.
        
        Args:
            node_type: Node type name
            
        Returns:
            Color string or default if not defined
        """
        return self.node_colors.get(node_type, '#808080')  # Default gray
    
    def get_all_edge_types(self) -> List[str]:
        """
        Returns list of all edge types in the ontology.
        
        Returns:
            List of all edge type names
        """
        return list(self.edge_types.keys())
    
    def get_all_node_types(self) -> List[str]:
        """
        Returns list of all node types in the ontology.
        
        Returns:
            List of all node type names
        """
        return list(self.node_types.keys())


# Global singleton instance - use this throughout the codebase
ontology_manager = OntologyManager()


# Backwards compatibility functions for gradual migration
def get_evidence_hypothesis_edges() -> List[str]:
    """
    Backwards compatibility wrapper.
    Replace hardcoded lists with this function call.
    """
    return ontology_manager.get_evidence_hypothesis_edges()


def get_van_evera_diagnostic_edges() -> List[str]:
    """
    Backwards compatibility wrapper for Van Evera diagnostic edges.
    """
    return ontology_manager.get_van_evera_edges()