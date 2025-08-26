"""
Legacy Compatibility Manager Plugin
Handles migration and compatibility between old H_001/AE_001 format and new Q/H1/H2/H3 structure
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from .base import ProcessTracingPlugin, PluginValidationError


class LegacyCompatibilityManagerPlugin(ProcessTracingPlugin):
    """
    Plugin for managing compatibility between legacy hypothesis IDs (H_001, AE_001) 
    and new academic structure (Q_H1, Q_H2, Q_H3).
    
    Provides migration, translation, and dual-format support capabilities.
    """
    
    plugin_id = "legacy_compatibility_manager"
    
    # Legacy ID patterns for recognition
    LEGACY_PATTERNS = {
        'hypothesis': re.compile(r'^H_(\d{3})$'),
        'alternative': re.compile(r'^AE_(\d{3})$'),
        'event': re.compile(r'^E_(\d{3})$'),
        'evidence': re.compile(r'^EV_(\d{3})$')
    }
    
    # New academic ID patterns
    ACADEMIC_PATTERNS = {
        'research_question': re.compile(r'^Q_(\d{3})$'),
        'primary_hypothesis': re.compile(r'^Q_H1$'),
        'alternative_hypothesis': re.compile(r'^Q_H(\d+)$')
    }
    
    def validate_input(self, data: Any) -> None:
        """Validate input contains graph data for compatibility analysis"""
        if not isinstance(data, dict):
            raise PluginValidationError(self.id, "Input must be dictionary")
        
        if 'graph_data' not in data:
            raise PluginValidationError(self.id, "Missing required key 'graph_data'")
        
        graph_data = data['graph_data']
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data:
            raise PluginValidationError(self.id, "graph_data must contain 'nodes'")
        
        self.logger.info(f"VALIDATION: Legacy compatibility analysis for {len(graph_data['nodes'])} nodes")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Execute legacy compatibility management"""
        self.logger.info("START: Legacy compatibility analysis and migration")
        
        graph_data = data['graph_data']
        migration_mode = data.get('migration_mode', 'detect_and_migrate')  # Options: 'detect_and_migrate', 'maintain_dual', 'validate_only'
        
        # Analyze current ID structure
        id_analysis = self._analyze_id_structure(graph_data)
        
        # Determine compatibility approach
        compatibility_strategy = self._determine_compatibility_strategy(id_analysis, migration_mode)
        
        # Execute compatibility management
        if compatibility_strategy['action'] == 'migrate_to_academic':
            migration_result = self._migrate_to_academic_structure(graph_data, id_analysis)
            updated_graph_data = migration_result['updated_graph_data']
            compatibility_info = migration_result
        elif compatibility_strategy['action'] == 'maintain_dual_format':
            dual_result = self._maintain_dual_format(graph_data, id_analysis)
            updated_graph_data = dual_result['updated_graph_data']
            compatibility_info = dual_result
        elif compatibility_strategy['action'] == 'validate_academic':
            validation_result = self._validate_academic_structure(graph_data, id_analysis)
            updated_graph_data = graph_data  # No changes needed
            compatibility_info = validation_result
        else:
            # No action needed
            updated_graph_data = graph_data
            compatibility_info = {'action': 'no_migration_needed', 'analysis': id_analysis}
        
        self.logger.info(f"COMPLETE: Legacy compatibility {compatibility_strategy['action']} completed")
        
        return {
            'id_structure_analysis': id_analysis,
            'compatibility_strategy': compatibility_strategy,
            'compatibility_management': compatibility_info,
            'updated_graph_data': updated_graph_data,
            'migration_summary': self._generate_migration_summary(id_analysis, compatibility_info),
            'backwards_compatibility_maintained': True
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for legacy compatibility management"""
        return {
            'plugin_id': self.id,
            'legacy_patterns_supported': len(self.LEGACY_PATTERNS),
            'academic_patterns_supported': len(self.ACADEMIC_PATTERNS),
            'method': 'pattern_based_migration'
        }
    
    def _analyze_id_structure(self, graph_data: Dict) -> Dict[str, Any]:
        """Analyze current ID structure to determine legacy vs academic format"""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Categorize node IDs
        legacy_ids: Dict[str, List[Dict[str, Any]]] = {
            'hypothesis': [],
            'alternative': [],
            'event': [],
            'evidence': [],
            'other_legacy': []
        }
        
        academic_ids: Dict[str, List[Dict[str, Any]]] = {
            'research_question': [],
            'primary_hypothesis': [],
            'alternative_hypothesis': [],
            'other_academic': []
        }
        
        unrecognized_ids = []
        
        for node in nodes:
            node_id = node.get('id', '')
            node_type = node.get('type', '')
            
            # Check legacy patterns
            if self.LEGACY_PATTERNS['hypothesis'].match(node_id):
                legacy_ids['hypothesis'].append({'id': node_id, 'type': node_type, 'node': node})
            elif self.LEGACY_PATTERNS['alternative'].match(node_id):
                legacy_ids['alternative'].append({'id': node_id, 'type': node_type, 'node': node})
            elif self.LEGACY_PATTERNS['event'].match(node_id):
                legacy_ids['event'].append({'id': node_id, 'type': node_type, 'node': node})
            elif self.LEGACY_PATTERNS['evidence'].match(node_id):
                legacy_ids['evidence'].append({'id': node_id, 'type': node_type, 'node': node})
            # Check academic patterns
            elif self.ACADEMIC_PATTERNS['research_question'].match(node_id):
                academic_ids['research_question'].append({'id': node_id, 'type': node_type, 'node': node})
            elif self.ACADEMIC_PATTERNS['primary_hypothesis'].match(node_id):
                academic_ids['primary_hypothesis'].append({'id': node_id, 'type': node_type, 'node': node})
            elif self.ACADEMIC_PATTERNS['alternative_hypothesis'].match(node_id) and node_id != 'Q_H1':
                academic_ids['alternative_hypothesis'].append({'id': node_id, 'type': node_type, 'node': node})
            else:
                # Check for other patterns
                if node_id.startswith('H_') or node_id.startswith('AE_'):
                    legacy_ids['other_legacy'].append({'id': node_id, 'type': node_type, 'node': node})
                elif node_id.startswith('Q_'):
                    academic_ids['other_academic'].append({'id': node_id, 'type': node_type, 'node': node})
                else:
                    unrecognized_ids.append({'id': node_id, 'type': node_type, 'node': node})
        
        # Calculate format distribution
        total_legacy = sum(len(id_list) for id_list in legacy_ids.values())
        total_academic = sum(len(id_list) for id_list in academic_ids.values())
        total_nodes = len(nodes)
        
        # Determine dominant format
        if total_legacy > total_academic:
            dominant_format = 'legacy'
        elif total_academic > total_legacy:
            dominant_format = 'academic'
        else:
            dominant_format = 'mixed'
        
        return {
            'total_nodes': total_nodes,
            'legacy_ids': legacy_ids,
            'academic_ids': academic_ids,
            'unrecognized_ids': unrecognized_ids,
            'format_distribution': {
                'legacy_count': total_legacy,
                'academic_count': total_academic,
                'unrecognized_count': len(unrecognized_ids),
                'legacy_percentage': (total_legacy / total_nodes * 100) if total_nodes > 0 else 0,
                'academic_percentage': (total_academic / total_nodes * 100) if total_nodes > 0 else 0
            },
            'dominant_format': dominant_format,
            'migration_needed': total_legacy > 0 and total_academic == 0,
            'validation_needed': total_academic > 0
        }
    
    def _determine_compatibility_strategy(self, id_analysis: Dict, migration_mode: str) -> Dict[str, Any]:
        """Determine appropriate compatibility strategy based on analysis"""
        
        dominant_format = id_analysis['dominant_format']
        legacy_count = id_analysis['format_distribution']['legacy_count']
        academic_count = id_analysis['format_distribution']['academic_count']
        
        if migration_mode == 'validate_only':
            return {
                'action': 'validate_academic',
                'rationale': 'Validation mode requested',
                'confidence': 1.0
            }
        
        # Strategy decision logic
        if dominant_format == 'legacy' and academic_count == 0:
            # Pure legacy format - migrate to academic
            return {
                'action': 'migrate_to_academic',
                'rationale': 'Pure legacy format detected, migration to Q/H1/H2/H3 structure recommended',
                'confidence': 0.9
            }
        elif dominant_format == 'academic' and legacy_count == 0:
            # Pure academic format - validate structure
            return {
                'action': 'validate_academic',
                'rationale': 'Pure academic format detected, validation only needed',
                'confidence': 0.95
            }
        elif dominant_format == 'mixed' and migration_mode == 'maintain_dual':
            # Mixed format with dual support requested
            return {
                'action': 'maintain_dual_format',
                'rationale': 'Mixed format with dual compatibility requested',
                'confidence': 0.8
            }
        elif dominant_format == 'mixed' and academic_count < legacy_count:
            # Mixed format with legacy dominance - migrate
            return {
                'action': 'migrate_to_academic',
                'rationale': 'Mixed format with legacy dominance, migration recommended',
                'confidence': 0.75
            }
        else:
            # Mixed format with academic dominance or equal - maintain current
            return {
                'action': 'no_action',
                'rationale': 'Academic format dominance or balanced mix, no migration needed',
                'confidence': 0.7
            }
    
    def _migrate_to_academic_structure(self, graph_data: Dict, id_analysis: Dict) -> Dict[str, Any]:
        """Migrate legacy ID structure to academic Q/H1/H2/H3 format"""
        
        # Create ID mapping
        id_mapping: Dict[str, str] = {}
        migration_log: List[str] = []
        
        # First, identify hypotheses for ranking-based migration
        legacy_hypotheses = id_analysis['legacy_ids']['hypothesis'] + id_analysis['legacy_ids']['alternative']
        
        if len(legacy_hypotheses) == 0:
            self.logger.warning("No legacy hypotheses found for migration")
            return {
                'updated_graph_data': graph_data,
                'id_mapping': id_mapping,
                'migration_log': migration_log,
                'hypotheses_migrated': 0
            }
        
        # For migration without Van Evera scores, use simple ordering
        # In production, this would integrate with Van Evera results
        sorted_hypotheses = sorted(legacy_hypotheses, key=lambda h: self._estimate_hypothesis_strength(h))
        
        # Assign new IDs
        if len(sorted_hypotheses) > 0:
            # Primary hypothesis becomes Q_H1
            primary_hypothesis = sorted_hypotheses[0]
            id_mapping[primary_hypothesis['id']] = 'Q_H1'
            migration_log.append(f"Migrated {primary_hypothesis['id']} to Q_H1 (primary hypothesis)")
            
            # Alternative hypotheses become Q_H2, Q_H3, etc.
            for i, hypothesis in enumerate(sorted_hypotheses[1:], 2):
                new_id = f'Q_H{i}'
                id_mapping[hypothesis['id']] = new_id
                migration_log.append(f"Migrated {hypothesis['id']} to {new_id} (alternative hypothesis)")
        
        # Migrate other legacy IDs to preserve functionality
        for category, id_list in id_analysis['legacy_ids'].items():
            if category in ['hypothesis', 'alternative']:
                continue  # Already handled above
            
            for i, item in enumerate(id_list, 1):
                old_id = item['id']
                if category == 'event':
                    new_id = f'E_{i:03d}'
                elif category == 'evidence':
                    new_id = f'EV_{i:03d}'
                else:
                    new_id = f'LEG_{i:03d}'  # Generic legacy migration
                
                id_mapping[old_id] = new_id
                migration_log.append(f"Migrated {old_id} to {new_id} ({category})")
        
        # Apply ID mapping to graph data
        updated_graph_data = self._apply_id_mapping(graph_data, id_mapping)
        
        # Generate research question if none exists
        research_question_exists = any(
            node.get('type') == 'Research_Question' for node in updated_graph_data['nodes']
        )
        
        if not research_question_exists:
            # Create research question from migrated hypotheses
            research_question = self._create_research_question_from_hypotheses(updated_graph_data)
            updated_graph_data['nodes'].append(research_question['node'])
            
            # Connect hypotheses to research question
            for hypothesis_node in updated_graph_data['nodes']:
                if hypothesis_node.get('type') in ['Hypothesis', 'Alternative_Explanation']:
                    edge = {
                        'source_id': hypothesis_node['id'],
                        'target_id': research_question['node']['id'],
                        'type': 'addresses_research_question',
                        'properties': {
                            'relevance': 0.8,
                            'approach': 'Direct answer to research question'
                        }
                    }
                    updated_graph_data['edges'].append(edge)
            
            migration_log.append(f"Generated research question: {research_question['node']['id']}")
        
        return {
            'updated_graph_data': updated_graph_data,
            'id_mapping': id_mapping,
            'migration_log': migration_log,
            'hypotheses_migrated': len(legacy_hypotheses),
            'research_question_generated': not research_question_exists
        }
    
    def _maintain_dual_format(self, graph_data: Dict, id_analysis: Dict) -> Dict[str, Any]:
        """Maintain dual format compatibility with cross-references"""
        
        # Create cross-reference mapping for dual access
        dual_mapping = {}
        
        # Add academic aliases for legacy IDs
        for category, id_list in id_analysis['legacy_ids'].items():
            if category in ['hypothesis', 'alternative']:
                for i, item in enumerate(id_list):
                    legacy_id = item['id']
                    academic_alias = f'Q_H{i+1}' if category == 'hypothesis' else f'Q_H{i+2}'
                    dual_mapping[legacy_id] = academic_alias
        
        # Add legacy references to node properties
        updated_graph_data = graph_data.copy()
        updated_nodes = []
        
        for node in graph_data['nodes']:
            updated_node = node.copy()
            node_id = node['id']
            
            if node_id in dual_mapping:
                # Add academic alias
                updated_node['properties'] = updated_node.get('properties', {}).copy()
                updated_node['properties']['academic_alias'] = dual_mapping[node_id]
                updated_node['properties']['legacy_id'] = node_id
                updated_node['properties']['dual_format_supported'] = True
            
            updated_nodes.append(updated_node)
        
        updated_graph_data['nodes'] = updated_nodes
        
        return {
            'updated_graph_data': updated_graph_data,
            'dual_mapping': dual_mapping,
            'dual_format_maintained': True,
            'legacy_ids_preserved': len(dual_mapping)
        }
    
    def _validate_academic_structure(self, graph_data: Dict, id_analysis: Dict) -> Dict[str, Any]:
        """Validate academic Q/H1/H2/H3 structure compliance"""
        
        validation_results: Dict[str, Any] = {
            'structure_valid': True,
            'issues_found': [],
            'compliance_score': 0.0,
            'recommendations': []
        }
        
        academic_ids = id_analysis['academic_ids']
        
        # Check for required components
        if len(academic_ids['research_question']) == 0:
            validation_results['issues_found'].append("No research question found")
            validation_results['recommendations'].append("Generate research question using ResearchQuestionGenerator")
            validation_results['structure_valid'] = False
        
        if len(academic_ids['primary_hypothesis']) == 0:
            validation_results['issues_found'].append("No primary hypothesis (Q_H1) found")
            validation_results['recommendations'].append("Identify primary hypothesis using PrimaryHypothesisIdentifier")
            validation_results['structure_valid'] = False
        elif len(academic_ids['primary_hypothesis']) > 1:
            validation_results['issues_found'].append("Multiple primary hypotheses found")
            validation_results['recommendations'].append("Ensure only one Q_H1 exists")
            validation_results['structure_valid'] = False
        
        if len(academic_ids['alternative_hypothesis']) < 2:
            validation_results['issues_found'].append("Insufficient alternative hypotheses for robust analysis")
            validation_results['recommendations'].append("Generate additional alternatives using AlternativeHypothesisGenerator")
        
        # Check ID sequence integrity
        alt_hypothesis_numbers = []
        for alt_hyp in academic_ids['alternative_hypothesis']:
            match = self.ACADEMIC_PATTERNS['alternative_hypothesis'].match(alt_hyp['id'])
            if match:
                alt_hypothesis_numbers.append(int(match.group(1)))
        
        if alt_hypothesis_numbers:
            alt_hypothesis_numbers.sort()
            expected_sequence = list(range(2, len(alt_hypothesis_numbers) + 2))
            if alt_hypothesis_numbers != expected_sequence:
                validation_results['issues_found'].append("Non-sequential alternative hypothesis numbering")
                validation_results['recommendations'].append("Renumber alternative hypotheses in sequence")
        
        # Calculate compliance score
        total_checks = 4  # research question, primary hypothesis, alternatives, sequence
        issues_found = validation_results['issues_found']
        assert isinstance(issues_found, list), "issues_found should be a list"
        passed_checks = total_checks - len(issues_found)
        validation_results['compliance_score'] = passed_checks / total_checks
        
        return validation_results
    
    def _estimate_hypothesis_strength(self, hypothesis_item: Dict) -> float:
        """Estimate hypothesis strength for migration ordering (fallback when no Van Evera scores)"""
        node = hypothesis_item['node']
        
        # Simple heuristic based on description length and complexity
        description = node.get('properties', {}).get('description', '')
        
        strength_score = 0.0
        
        # Length bonus (longer descriptions often more developed)
        strength_score += min(len(description) / 500.0, 0.3)
        
        # Theoretical sophistication indicators
        theoretical_terms = ['mechanism', 'process', 'theory', 'causal', 'systematic', 'institutional']
        theory_score = sum(1 for term in theoretical_terms if term.lower() in description.lower())
        strength_score += min(theory_score * 0.1, 0.2)
        
        # Evidence language indicators
        evidence_terms = ['evidence', 'support', 'confirm', 'demonstrate', 'indicate']
        evidence_score = sum(1 for term in evidence_terms if term.lower() in description.lower())
        strength_score += min(evidence_score * 0.1, 0.2)
        
        # Prefer hypothesis over alternative explanation in legacy format
        if hypothesis_item['id'].startswith('H_'):
            strength_score += 0.3
        
        return strength_score
    
    def _apply_id_mapping(self, graph_data: Dict, id_mapping: Dict) -> Dict[str, Any]:
        """Apply ID mapping to all nodes and edges"""
        updated_graph_data = graph_data.copy()
        
        # Update node IDs
        updated_nodes = []
        for node in graph_data['nodes']:
            updated_node = node.copy()
            old_id = node['id']
            
            if old_id in id_mapping:
                new_id = id_mapping[old_id]
                updated_node['id'] = new_id
                
                # Preserve original ID in properties
                updated_node['properties'] = updated_node.get('properties', {}).copy()
                updated_node['properties']['original_id'] = old_id
                
                # Update hypothesis type for academic structure
                if new_id == 'Q_H1':
                    updated_node['properties']['hypothesis_type'] = 'primary'
                elif new_id.startswith('Q_H'):
                    updated_node['properties']['hypothesis_type'] = 'alternative'
            
            updated_nodes.append(updated_node)
        
        # Update edge references
        updated_edges = []
        for edge in graph_data['edges']:
            updated_edge = edge.copy()
            
            if edge['source_id'] in id_mapping:
                updated_edge['source_id'] = id_mapping[edge['source_id']]
            
            if edge['target_id'] in id_mapping:
                updated_edge['target_id'] = id_mapping[edge['target_id']]
            
            updated_edges.append(updated_edge)
        
        updated_graph_data['nodes'] = updated_nodes
        updated_graph_data['edges'] = updated_edges
        
        return updated_graph_data
    
    def _create_research_question_from_hypotheses(self, graph_data: Dict) -> Dict[str, Any]:
        """Create research question from migrated hypothesis content"""
        hypotheses = [n for n in graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
        
        if not hypotheses:
            return {
                'node': {
                    'id': 'Q',
                    'type': 'Research_Question',
                    'properties': {
                        'description': 'What factors explain the phenomenon under investigation?',
                        'domain': 'general',
                        'scope': 'General analytical scope',
                        'importance': 'Generated during legacy migration'
                    }
                }
            }
        
        # Extract common themes from hypotheses
        all_hypothesis_text = " ".join([
            h.get('properties', {}).get('description', '') 
            for h in hypotheses
        ])
        
        # Simple pattern matching for research question generation
        if 'revolution' in all_hypothesis_text.lower() or 'revolutionary' in all_hypothesis_text.lower():
            description = "What factors explain the emergence and success of revolutionary resistance movements?"
            domain = "political"
        elif 'resistance' in all_hypothesis_text.lower():
            description = "What mechanisms account for the development of resistance movements?"
            domain = "political"
        elif 'economic' in all_hypothesis_text.lower():
            description = "What economic factors drive political and social change?"
            domain = "economic"
        elif 'political' in all_hypothesis_text.lower():
            description = "What political processes explain institutional change and conflict?"
            domain = "political"
        else:
            description = "What causal mechanisms explain the phenomenon under investigation?"
            domain = "general"
        
        return {
            'node': {
                'id': 'Q',
                'type': 'Research_Question',
                'properties': {
                    'description': description,
                    'domain': domain,
                    'scope': 'Generated from legacy hypothesis migration',
                    'importance': 'Synthesized from existing analytical framework'
                }
            }
        }
    
    def _generate_migration_summary(self, id_analysis: Dict, compatibility_info: Dict) -> Dict[str, Any]:
        """Generate comprehensive migration summary"""
        
        return {
            'migration_type': compatibility_info.get('action', 'no_action'),
            'nodes_analyzed': id_analysis['total_nodes'],
            'legacy_nodes_found': id_analysis['format_distribution']['legacy_count'],
            'academic_nodes_found': id_analysis['format_distribution']['academic_count'],
            'hypotheses_migrated': compatibility_info.get('hypotheses_migrated', 0),
            'research_question_generated': compatibility_info.get('research_question_generated', False),
            'backwards_compatibility': {
                'original_ids_preserved': True,
                'dual_format_support': compatibility_info.get('dual_format_maintained', False),
                'migration_reversible': True
            },
            'academic_structure_compliance': {
                'q_structure_implemented': compatibility_info.get('hypotheses_migrated', 0) > 0,
                'primary_hypothesis_identified': compatibility_info.get('hypotheses_migrated', 0) > 0,
                'alternative_hypotheses_numbered': compatibility_info.get('hypotheses_migrated', 0) > 1
            }
        }


def migrate_legacy_to_academic_structure(graph_data: Dict, migration_mode: str = 'detect_and_migrate') -> Dict[str, Any]:
    """
    Convenience function for migrating legacy hypothesis structure to academic Q/H1/H2/H3 format.
    Returns migration results with updated graph data.
    """
    from .base import PluginContext
    
    context = PluginContext({'legacy_migration': True})
    plugin = LegacyCompatibilityManagerPlugin('legacy_compatibility_manager', context)
    
    result = plugin.execute({
        'graph_data': graph_data,
        'migration_mode': migration_mode
    })
    return result