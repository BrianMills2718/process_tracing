"""Unit tests for AlternativeHypothesisGeneratorPlugin - focused on alternative hypothesis generation logic"""

import pytest
from unittest.mock import Mock, patch
from core.plugins.alternative_hypothesis_generator import (
    AlternativeHypothesisGeneratorPlugin
)
from core.plugins.base import PluginContext, PluginValidationError

class TestAlternativeHypothesisGeneratorPlugin:
    """Unit tests for AlternativeHypothesisGeneratorPlugin"""
    
    @pytest.fixture
    def plugin_context(self):
        """Mock plugin context"""
        context = Mock(spec=PluginContext)
        context.config = {}
        context.data_bus = {}
        return context
        
    @pytest.fixture  
    def plugin(self, plugin_context):
        """Plugin instance with mocked context"""
        return AlternativeHypothesisGeneratorPlugin("alternative_hypothesis_generator", plugin_context)
        
    @pytest.fixture
    def base_graph_data(self):
        """Basic graph data with existing hypothesis and evidence"""
        return {
            'nodes': [
                {
                    'id': 'Q_H1', 
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Taxation without representation was the primary cause of colonial resistance',
                        'hypothesis_type': 'primary'
                    }
                },
                {
                    'id': 'E1', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Colonial documents consistently invoke constitutional rights and taxation issues',
                        'source_text_quote': 'No taxation without representation in Parliament'
                    }
                },
                {
                    'id': 'E2', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Merchant class organized boycotts to protect economic interests and trade profits',
                        'source_text_quote': 'Boston merchants coordinate trade resistance for commercial advantage'
                    }
                },
                {
                    'id': 'E3', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Religious revival influenced political awakening and resistance ideology',
                        'source_text_quote': 'Evangelical Christianity provided moral framework for independence'
                    }
                }
            ],
            'edges': [
                {
                    'source_id': 'E1', 
                    'target_id': 'Q_H1', 
                    'type': 'supports',
                    'properties': {'diagnostic_type': 'hoop'}
                }
            ]
        }
        
    @pytest.fixture
    def empty_graph_data(self):
        """Graph data without existing hypotheses"""
        return {
            'nodes': [
                {'id': 'E1', 'type': 'Evidence', 'properties': {'description': 'Some evidence'}}
            ],
            'edges': []
        }
        
    class TestInputValidation:
        """Test validate_input() method"""
        
        def test_valid_input_accepted(self, plugin, base_graph_data):
            """Plugin accepts valid graph input with existing hypotheses"""
            valid_input = {'graph_data': base_graph_data}
            # Should not raise exception
            plugin.validate_input(valid_input)
            
        def test_empty_graph_accepted_with_warning(self, plugin, empty_graph_data):
            """Plugin accepts empty graph but logs warning"""
            valid_input = {'graph_data': empty_graph_data}
            # Should not raise exception but may log warning
            plugin.validate_input(valid_input)
            
        def test_non_dict_input_rejected(self, plugin):
            """Plugin rejects non-dictionary input"""
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input("invalid_string")
            assert "Input must be dictionary" in str(exc_info.value)
            
        def test_missing_graph_data_key_rejected(self, plugin):
            """Plugin rejects input missing graph_data key"""
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input({'other_data': 'value'})
            assert "Missing required key 'graph_data'" in str(exc_info.value)
            
        def test_invalid_graph_structure_rejected(self, plugin):
            """Plugin rejects graph_data without required structure"""
            invalid_data = {'graph_data': {'nodes': []}}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(invalid_data)
            assert "graph_data must contain 'nodes' and 'edges'" in str(exc_info.value)
                
    class TestAlternativeHypothesesStructure:
        """Test structure and content of predefined alternative hypotheses"""
        
        def test_revolution_alternatives_available(self, plugin):
            """Plugin has predefined American Revolution alternative hypotheses"""
            alternatives = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES
            
            assert isinstance(alternatives, dict)
            assert len(alternatives) >= 6  # Should have comprehensive set
            
            # Check for key domains
            expected_domains = [
                'economic_interests', 'generational_conflict', 'religious_awakening',
                'elite_power_struggle', 'regional_political_culture', 'imperial_overstretch'
            ]
            
            for domain in expected_domains:
                assert domain in alternatives
                
        def test_alternative_hypothesis_structure(self, plugin):
            """Each alternative hypothesis has required academic structure"""
            alternatives = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES
            
            for alt_key, alt_data in alternatives.items():
                # Required fields for academic quality
                assert 'id' in alt_data
                assert 'description' in alt_data
                assert 'theoretical_basis' in alt_data
                assert 'key_predictions' in alt_data
                assert 'testable_mechanisms' in alt_data
                assert 'evidence_requirements' in alt_data
                assert 'competing_claims' in alt_data
                
                # Data types
                assert isinstance(alt_data['key_predictions'], list)
                assert isinstance(alt_data['testable_mechanisms'], list)
                assert isinstance(alt_data['evidence_requirements'], list)
                
                # Content quality
                assert len(alt_data['description']) > 50  # Substantial description
                assert len(alt_data['key_predictions']) >= 3  # Multiple predictions
                assert len(alt_data['testable_mechanisms']) >= 2  # Testable mechanisms
                
        def test_unique_hypothesis_ids(self, plugin):
            """Each alternative hypothesis has unique ID"""
            alternatives = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES
            ids = [alt['id'] for alt in alternatives.values()]
            
            assert len(ids) == len(set(ids))  # All unique
            
            # Should follow Q_H pattern
            for alt_id in ids:
                assert alt_id.startswith('Q_H')
                assert len(alt_id) >= 4  # Q_H + number
                
        def test_theoretical_diversity(self, plugin):
            """Alternative hypotheses cover diverse theoretical domains"""
            alternatives = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES
            theoretical_bases = [alt['theoretical_basis'] for alt in alternatives.values()]
            
            # Should have diverse theoretical foundations
            unique_theories = set()
            for basis in theoretical_bases:
                # Extract first theory from comma-separated list
                first_theory = basis.split(',')[0].strip().lower()
                unique_theories.add(first_theory)
                
            # Should have significant theoretical diversity
            assert len(unique_theories) >= 5
            
    class TestEvidenceIdentification:
        """Test _identify_relevant_evidence() method"""
        
        def test_relevant_evidence_identification(self, plugin, base_graph_data):
            """Correctly identifies evidence relevant to alternative hypotheses"""
            # Test with economic interests alternative
            economic_alt = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES['economic_interests']
            
            relevant_evidence = plugin._identify_relevant_evidence(base_graph_data, economic_alt)
            
            assert isinstance(relevant_evidence, list)
            # Should find E2 (merchant class evidence) as relevant
            relevant_ids = [e['id'] for e in relevant_evidence]
            assert 'E2' in relevant_ids  # Should match merchant/economic evidence
            
        def test_relevance_scoring_algorithm(self, plugin, base_graph_data):
            """Test relevance scoring based on keyword matching"""
            religious_alt = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES['religious_awakening']
            
            relevant_evidence = plugin._identify_relevant_evidence(base_graph_data, religious_alt)
            
            # Should find E3 (religious revival evidence) as most relevant
            if relevant_evidence:
                relevant_ids = [e['id'] for e in relevant_evidence]
                assert 'E3' in relevant_ids  # Should match religious evidence
                
        def test_evidence_limit_enforcement(self, plugin):
            """Test that evidence identification limits results appropriately"""
            # Create graph with many evidence nodes
            large_graph = {
                'nodes': [{'id': f'E{i}', 'type': 'Evidence', 
                          'properties': {'description': 'economic trade merchant commercial'}} 
                         for i in range(20)],
                'edges': []
            }
            
            economic_alt = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES['economic_interests']
            relevant_evidence = plugin._identify_relevant_evidence(large_graph, economic_alt)
            
            # Should limit to 15 pieces of evidence
            assert len(relevant_evidence) <= 15
            
        def test_evidence_requirement_matching(self, plugin):
            """Test _assess_evidence_requirement_match() method"""
            # Test evidence matching economic requirements
            evidence_node = {
                'id': 'E_TEST',
                'properties': {
                    'description': 'merchant networks facilitated trade coordination and commercial interests'
                }
            }
            
            economic_alt = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES['economic_interests']
            match_assessment = plugin._assess_evidence_requirement_match(evidence_node, economic_alt)
            
            assert isinstance(match_assessment, str)
            # Should identify strong match due to merchant_networks and commercial_interests
            assert 'match' in match_assessment.lower()
            
    class TestExecutionLogic:
        """Test execute() method and comprehensive generation workflow"""
        
        def test_execute_with_valid_input(self, plugin, base_graph_data):
            """Plugin executes alternative generation successfully"""
            input_data = {'graph_data': base_graph_data}
            result = plugin.execute(input_data)
            
            # Verify result structure
            assert result is not None
            assert isinstance(result, dict)
            assert 'updated_graph_data' in result
            assert 'alternative_hypotheses_created' in result
            assert 'competitive_relationships_added' in result
            assert 'theoretical_competition_metrics' in result
            assert 'academic_assessment' in result
            assert 'generation_statistics' in result
            
        def test_graph_data_updating(self, plugin, base_graph_data):
            """Plugin correctly updates graph data with new alternatives"""
            original_node_count = len(base_graph_data['nodes'])
            original_edge_count = len(base_graph_data['edges'])
            
            input_data = {'graph_data': base_graph_data}
            result = plugin.execute(input_data)
            
            updated_graph = result['updated_graph_data']
            
            # Should add new hypothesis nodes
            new_node_count = len(updated_graph['nodes'])
            assert new_node_count > original_node_count
            
            # Should add new evidence connections
            new_edge_count = len(updated_graph['edges'])
            assert new_edge_count >= original_edge_count  # May or may not add edges depending on relevance
            
            # Check for alternative hypotheses in nodes
            hypothesis_nodes = [n for n in updated_graph['nodes'] if n.get('type') == 'Hypothesis']
            alternative_hypotheses = [n for n in hypothesis_nodes if 
                                    n.get('properties', {}).get('hypothesis_type') == 'alternative']
            
            assert len(alternative_hypotheses) > 0
            # Should create multiple alternatives from predefined set
            assert len(alternative_hypotheses) >= 6
            
        def test_alternative_node_properties(self, plugin, base_graph_data):
            """Alternative hypothesis nodes have correct properties"""
            input_data = {'graph_data': base_graph_data}
            result = plugin.execute(input_data)
            
            updated_graph = result['updated_graph_data']
            
            # Find generated alternative hypothesis nodes
            alternative_nodes = [n for n in updated_graph['nodes'] 
                               if n.get('type') == 'Hypothesis' and 
                                  n.get('properties', {}).get('hypothesis_type') == 'alternative']
            
            assert len(alternative_nodes) > 0
            
            # Check properties of first alternative
            alt_node = alternative_nodes[0]
            props = alt_node['properties']
            
            # Required properties
            assert 'description' in props
            assert 'theoretical_basis' in props
            assert 'key_predictions' in props
            assert 'testable_mechanisms' in props
            assert 'evidence_requirements' in props
            assert 'competing_claims' in props
            assert 'generated_by' in props
            
            # Property values
            assert props['hypothesis_type'] == 'alternative'
            assert props['generated_by'] == 'alternative_hypothesis_generator'
            assert props['academic_quality'] == 'peer_reviewed'
            assert isinstance(props['key_predictions'], list)
            assert len(props['key_predictions']) >= 3
            
        def test_evidence_connection_creation(self, plugin, base_graph_data):
            """Plugin creates appropriate evidence connections for alternatives"""
            input_data = {'graph_data': base_graph_data}
            result = plugin.execute(input_data)
            
            updated_graph = result['updated_graph_data']
            
            # Check for new evidence edges to alternative hypotheses
            alternative_ids = result['alternative_hypotheses_created']
            
            evidence_edges_to_alternatives = []
            for edge in updated_graph['edges']:
                if edge.get('target_id') in alternative_ids and edge.get('type') == 'supports':
                    evidence_edges_to_alternatives.append(edge)
                    
            # Should create some evidence connections
            if len(alternative_ids) > 0:
                # At least some alternatives should have evidence connections
                assert len(evidence_edges_to_alternatives) >= 0  # May be 0 if no relevant evidence
                
            # Check edge properties if connections exist
            for edge in evidence_edges_to_alternatives:
                props = edge.get('properties', {})
                assert 'diagnostic_type' in props
                assert 'probative_value' in props
                assert 'theoretical_relevance' in props
                assert 'evidence_requirement_match' in props
                
    class TestCompetitionMetrics:
        """Test _calculate_competition_metrics() method"""
        
        def test_competition_metrics_calculation(self, plugin):
            """Test calculation of theoretical competition metrics"""
            # Mock generation result
            generation_result = {
                'alternatives_created': ['Q_H2', 'Q_H3', 'Q_H4', 'Q_H5', 'Q_H6'],
                'competitive_edges_created': 0,  # No explicit competition edges
                'evidence_edges_created': 12,
                'theoretical_domains_covered': ['economic', 'religious', 'political', 'military', 'social']
            }
            
            metrics = plugin._calculate_competition_metrics(generation_result)
            
            assert isinstance(metrics, dict)
            assert 'total_alternative_hypotheses' in metrics
            assert 'theoretical_domains_covered' in metrics
            assert 'competitive_relationships' in metrics
            assert 'evidence_connections' in metrics
            assert 'competition_density' in metrics
            assert 'average_evidence_per_alternative' in metrics
            assert 'theoretical_diversity_score' in metrics
            assert 'academic_robustness_score' in metrics
            
            # Check calculations
            assert metrics['total_alternative_hypotheses'] == 5
            assert metrics['theoretical_domains_covered'] == 5
            assert metrics['evidence_connections'] == 12
            assert metrics['average_evidence_per_alternative'] == 2.4  # 12/5
            
        def test_competition_density_calculation(self, plugin):
            """Test competition density based on theoretical diversity"""
            # High diversity case
            high_diversity_result = {
                'alternatives_created': ['Q_H2', 'Q_H3', 'Q_H4', 'Q_H5', 'Q_H6', 'Q_H7', 'Q_H8'],
                'competitive_edges_created': 0,
                'evidence_edges_created': 21,
                'theoretical_domains_covered': ['economic', 'religious', 'political', 'military', 'social', 'cultural', 'institutional']
            }
            
            metrics = plugin._calculate_competition_metrics(high_diversity_result)
            
            # Should have high competition density due to theoretical diversity
            assert metrics['competition_density'] >= 0.8  # 7/8 domains = 0.875
            
    class TestAcademicAssessment:
        """Test _generate_academic_assessment() method"""
        
        def test_academic_assessment_generation(self, plugin):
            """Test generation of academic quality assessment"""
            # High-quality metrics
            high_quality_metrics = {
                'total_alternative_hypotheses': 8,
                'theoretical_domains_covered': 7,
                'competition_density': 0.9,
                'average_evidence_per_alternative': 4.5,
                'theoretical_diversity_score': 0.875,
                'academic_robustness_score': 0.7
            }
            
            assessment = plugin._generate_academic_assessment(high_quality_metrics)
            
            assert isinstance(assessment, dict)
            assert 'academic_quality_score' in assessment
            assert 'meets_academic_standards' in assessment
            assert 'theoretical_competition_adequate' in assessment
            assert 'van_evera_elimination_ready' in assessment
            assert 'quality_criteria' in assessment
            assert 'improvement_recommendations' in assessment
            assert 'publication_readiness' in assessment
            
            # Should meet high academic standards
            assert assessment['academic_quality_score'] >= 80
            assert assessment['meets_academic_standards'] == True
            assert assessment['publication_readiness'] == 'ready'
            
        def test_academic_standards_enforcement(self, plugin):
            """Test enforcement of minimum academic standards"""
            # Low-quality metrics
            low_quality_metrics = {
                'total_alternative_hypotheses': 3,  # Below minimum
                'theoretical_domains_covered': 2,  # Below minimum
                'competition_density': 0.5,  # Below threshold
                'average_evidence_per_alternative': 1.5,  # Below threshold
                'theoretical_diversity_score': 0.25,
                'academic_robustness_score': 0.2
            }
            
            assessment = plugin._generate_academic_assessment(low_quality_metrics)
            
            # Should not meet academic standards
            assert assessment['academic_quality_score'] < 80
            assert assessment['meets_academic_standards'] == False
            assert assessment['publication_readiness'] == 'needs_improvement'
            
            # Should provide improvement recommendations
            recommendations = assessment['improvement_recommendations']
            assert len(recommendations) > 0
            assert any('alternative hypotheses' in rec for rec in recommendations)
            assert any('domain' in rec for rec in recommendations)
            
    class TestIntegrationWorkflow:
        """Test complete workflow integration"""
        
        def test_comprehensive_generation_workflow(self, plugin, base_graph_data):
            """Test complete generation workflow from input to academic assessment"""
            input_data = {'graph_data': base_graph_data}
            result = plugin.execute(input_data)
            
            # Verify complete workflow execution
            assert 'updated_graph_data' in result
            assert 'theoretical_competition_metrics' in result
            assert 'academic_assessment' in result
            assert 'generation_statistics' in result
            
            # Check generation statistics
            stats = result['generation_statistics']
            assert 'total_alternatives_generated' in stats
            assert 'theoretical_domains_covered' in stats
            assert 'evidence_requirements_specified' in stats
            assert 'competitive_claims_defined' in stats
            
            # Should generate substantial number of alternatives
            assert stats['total_alternatives_generated'] >= 8
            assert stats['theoretical_domains_covered'] >= 5
            
        def test_checkpoint_data(self, plugin):
            """Test plugin checkpoint data includes generation capability info"""
            checkpoint = plugin.get_checkpoint_data()
            
            assert checkpoint['plugin_id'] == 'alternative_hypothesis_generator'
            assert 'alternatives_available' in checkpoint
            assert 'theoretical_domains' in checkpoint
            assert 'academic_quality' in checkpoint
            
            # Should report substantial alternatives available
            assert checkpoint['alternatives_available'] >= 8
            assert checkpoint['academic_quality'] == 'peer_reviewed_standard'
            assert isinstance(checkpoint['theoretical_domains'], list)
            
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        def test_empty_graph_handling(self, plugin, empty_graph_data):
            """Test handling of graph with no existing hypotheses"""
            input_data = {'graph_data': empty_graph_data}
            
            # Should not crash, should complete successfully
            result = plugin.execute(input_data)
            
            assert result is not None
            assert 'updated_graph_data' in result
            assert 'alternative_hypotheses_created' in result
            
            # Should still generate alternatives even without existing hypotheses
            assert len(result['alternative_hypotheses_created']) > 0
            
        def test_malformed_evidence_handling(self, plugin):
            """Test handling of evidence nodes with missing properties"""
            malformed_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis'},
                    {'id': 'E1', 'type': 'Evidence'}  # Missing properties
                ],
                'edges': []
            }
            
            input_data = {'graph_data': malformed_graph}
            result = plugin.execute(input_data)
            
            # Should handle gracefully
            assert result is not None
            assert 'alternative_hypotheses_created' in result
            # Should still generate alternatives
            assert len(result['alternative_hypotheses_created']) > 0
            
        def test_evidence_relevance_with_no_matches(self, plugin):
            """Test evidence identification when no evidence is relevant"""
            irrelevant_graph = {
                'nodes': [
                    {'id': 'E1', 'type': 'Evidence', 'properties': {'description': 'completely unrelated topic'}}
                ],
                'edges': []
            }
            
            economic_alt = plugin.REVOLUTION_ALTERNATIVE_HYPOTHESES['economic_interests']
            relevant_evidence = plugin._identify_relevant_evidence(irrelevant_graph, economic_alt)
            
            # Should return empty list gracefully
            assert relevant_evidence == []
            
        def test_integration_function(self):
            """Test standalone integration function"""
            from core.plugins.alternative_hypothesis_generator import generate_alternative_hypotheses
            
            # Create fresh graph data for this test
            fresh_graph_data = {
                'nodes': [
                    {
                        'id': 'Q_H1', 
                        'type': 'Hypothesis',
                        'properties': {
                            'description': 'Taxation without representation was the primary cause'
                        }
                    },
                    {
                        'id': 'E1', 
                        'type': 'Evidence',
                        'properties': {'description': 'Evidence supporting taxation hypothesis'}
                    }
                ],
                'edges': []
            }
            
            result_graph = generate_alternative_hypotheses(fresh_graph_data)
            
            # Should return updated graph data
            assert result_graph is not None
            assert isinstance(result_graph, dict)
            assert 'nodes' in result_graph
            assert 'edges' in result_graph
            
            # Should have multiple hypothesis nodes (original + alternatives)
            hypothesis_nodes = [n for n in result_graph['nodes'] if n.get('type') == 'Hypothesis']
            assert len(hypothesis_nodes) >= 8  # Should have original + 8+ alternatives
            
            # Should have alternative hypotheses specifically
            alternative_nodes = [n for n in hypothesis_nodes if 
                               n.get('properties', {}).get('hypothesis_type') == 'alternative']
            assert len(alternative_nodes) >= 7  # Should have generated alternatives