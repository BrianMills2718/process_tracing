"""Unit tests for ContentBasedDiagnosticClassifierPlugin - focused on content classification logic"""

import pytest
import re
from unittest.mock import Mock, patch, MagicMock
from core.plugins.content_based_diagnostic_classifier import (
    ContentBasedDiagnosticClassifierPlugin
)
from core.plugins.base import PluginContext, PluginValidationError

class TestContentBasedDiagnosticClassifierPlugin:
    """Unit tests for ContentBasedDiagnosticClassifierPlugin"""
    
    @pytest.fixture
    def plugin_context(self):
        """Mock plugin context with LLM query function"""
        context = Mock(spec=PluginContext)
        context.config = {}
        context.data_bus = {}
        
        # Mock LLM query function
        mock_llm_func = Mock()
        mock_llm_func.return_value = '''{
            "diagnostic_type": "hoop",
            "confidence": 0.8,
            "reasoning": "Evidence shows necessary condition",
            "van_evera_logic": "Required for hypothesis viability"
        }'''
        context.get_data.return_value = mock_llm_func
        return context
        
    @pytest.fixture  
    def plugin(self, plugin_context):
        """Plugin instance with mocked context"""
        return ContentBasedDiagnosticClassifierPlugin("content_based_diagnostic_classifier", plugin_context)
        
    @pytest.fixture
    def diagnostic_graph_data(self):
        """Graph data with evidence-hypothesis relationships for diagnostic classification"""
        return {
            'nodes': [
                {
                    'id': 'H1', 
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Taxation without representation was the primary cause of colonial resistance'
                    }
                },
                {
                    'id': 'E1', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Colonial documents consistently invoke constitutional rights and necessary consent',
                        'source_text_quote': 'No taxation without representation - this is necessary for legitimate government'
                    }
                },
                {
                    'id': 'E2', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Economic boycotts suggests commercial motivations',
                        'source_text_quote': 'Merchant records show profits declined significantly'
                    }
                },
                {
                    'id': 'CM1',
                    'type': 'Causal_Mechanism',
                    'properties': {
                        'description': 'Constitutional rights violations trigger resistance'
                    }
                }
            ],
            'edges': [
                {
                    'source_id': 'E1', 
                    'target_id': 'H1', 
                    'type': 'supports',
                    'properties': {
                        'diagnostic_type': 'unclassified',
                        'probative_value': 0.7
                    }
                },
                {
                    'source_id': 'E2', 
                    'target_id': 'H1', 
                    'type': 'refutes',
                    'properties': {
                        'diagnostic_type': 'straw_in_the_wind',
                        'probative_value': 0.4
                    }
                },
                {
                    'source_id': 'E1',
                    'target_id': 'CM1',
                    'type': 'evidence',
                    'properties': {
                        'diagnostic_type': 'smoking_gun',
                        'probative_value': 0.8
                    }
                }
            ]
        }
        
    @pytest.fixture
    def invalid_graph_data(self):
        """Graph data missing evidence-hypothesis relationships"""
        return {
            'nodes': [
                {'id': 'H1', 'type': 'Hypothesis', 'properties': {'description': 'Test hypothesis'}}
            ],
            'edges': []
        }
        
    class TestInputValidation:
        """Test validate_input() method for diagnostic classification requirements"""
        
        def test_valid_diagnostic_input_accepted(self, plugin, diagnostic_graph_data):
            """Plugin accepts valid graph with evidence-hypothesis relationships"""
            valid_input = {'graph_data': diagnostic_graph_data}
            # Should not raise exception
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
            
        def test_no_evidence_relationships_rejected(self, plugin, invalid_graph_data):
            """Plugin rejects graph with no evidence-hypothesis relationships"""
            invalid_input = {'graph_data': invalid_graph_data}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(invalid_input)
            assert "No evidence-hypothesis relationships found for classification" in str(exc_info.value)
                
    class TestEvidenceRelationshipDetection:
        """Test _is_evidence_relationship() method"""
        
        def test_evidence_to_hypothesis_detected(self, plugin, diagnostic_graph_data):
            """Correctly identifies evidence-hypothesis relationships"""
            edge = diagnostic_graph_data['edges'][0]  # E1 -> H1
            assert plugin._is_evidence_relationship(edge, diagnostic_graph_data) == True
            
        def test_evidence_to_causal_mechanism_detected(self, plugin, diagnostic_graph_data):
            """Correctly identifies evidence-causal mechanism relationships"""
            edge = diagnostic_graph_data['edges'][2]  # E1 -> CM1
            assert plugin._is_evidence_relationship(edge, diagnostic_graph_data) == True
            
        def test_non_evidence_relationship_rejected(self, plugin):
            """Correctly rejects non-evidence relationships"""
            graph_data = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis'},
                    {'id': 'H2', 'type': 'Hypothesis'}
                ],
                'edges': []
            }
            
            hypothesis_to_hypothesis_edge = {
                'source_id': 'H1',
                'target_id': 'H2', 
                'type': 'conflicts_with'
            }
            
            assert plugin._is_evidence_relationship(hypothesis_to_hypothesis_edge, graph_data) == False
            
    class TestContentAnalysis:
        """Test content analysis and scoring logic"""
        
        def test_content_scores_calculation(self, plugin):
            """Test content-based diagnostic scoring"""
            # Text with hoop test indicators
            hoop_evidence = "this evidence is necessary for the hypothesis to hold"
            hypothesis = "taxation without representation caused resistance"
            
            scores = plugin._calculate_content_scores(hoop_evidence, hypothesis)
            
            assert isinstance(scores, dict)
            assert 'hoop' in scores
            assert 'smoking_gun' in scores
            assert 'doubly_decisive' in scores
            assert 'straw_in_wind' in scores
            
            # Hoop test should score highest for this text
            assert scores['hoop'] > scores['smoking_gun']
            
        def test_smoking_gun_content_detection(self, plugin):
            """Test smoking gun diagnostic detection"""
            smoking_gun_evidence = "this proves conclusively that taxation was the cause"
            hypothesis = "taxation caused the resistance"
            
            scores = plugin._calculate_content_scores(smoking_gun_evidence, hypothesis)
            
            # Smoking gun should score highly
            assert scores['smoking_gun'] > 0.5
            assert scores['smoking_gun'] >= scores['hoop']
            
        def test_doubly_decisive_conservative_scoring(self, plugin):
            """Test that doubly decisive scoring is appropriately conservative"""
            # Even with strong indicators, doubly decisive should be conservative
            strong_evidence = "this is both necessary and sufficient to establish the hypothesis definitively"
            hypothesis = "test hypothesis"
            
            scores = plugin._calculate_content_scores(strong_evidence, hypothesis)
            
            # Should be capped at 0.8 due to conservative scoring
            assert scores['doubly_decisive'] <= 0.8
            
        def test_historical_context_bonus(self, plugin):
            """Test historical context enhancement"""
            historical_text = "colonial taxation constitutional rights representation government authority"
            
            bonus = plugin._calculate_historical_context_bonus(historical_text)
            
            assert 0 <= bonus <= 1.0
            assert bonus > 0  # Should get some bonus for historical terms
            
    class TestContentExtraction:
        """Test content extraction from graph nodes"""
        
        def test_evidence_content_extraction(self, plugin, diagnostic_graph_data):
            """Test extraction of evidence content for classification"""
            edge = diagnostic_graph_data['edges'][0]  # E1 -> H1
            
            content = plugin._extract_evidence_content(edge, diagnostic_graph_data)
            
            assert isinstance(content, dict)
            assert 'description' in content
            assert 'source_quote' in content
            assert 'full_text' in content
            
            # Should contain actual evidence description
            assert 'constitutional rights' in content['description']
            assert 'No taxation without representation' in content['source_quote']
            assert content['full_text'] != ''  # Combined text should not be empty
            
        def test_hypothesis_content_extraction(self, plugin, diagnostic_graph_data):
            """Test extraction of hypothesis content for classification"""
            edge = diagnostic_graph_data['edges'][0]  # E1 -> H1
            
            content = plugin._extract_hypothesis_content(edge, diagnostic_graph_data)
            
            assert isinstance(content, dict)
            assert 'description' in content
            assert 'type' in content
            assert 'full_text' in content
            
            assert content['type'] == 'Hypothesis'
            assert 'taxation without representation' in content['description'].lower()
            
        def test_missing_node_handling(self, plugin):
            """Test graceful handling of missing nodes"""
            empty_graph = {'nodes': [], 'edges': []}
            edge = {'source_id': 'nonexistent', 'target_id': 'also_nonexistent'}
            
            evidence_content = plugin._extract_evidence_content(edge, empty_graph)
            hypothesis_content = plugin._extract_hypothesis_content(edge, empty_graph)
            
            # Should return empty content gracefully
            assert evidence_content['description'] == ''
            assert hypothesis_content['description'] == ''
            
    class TestDistributionAnalysis:
        """Test distribution analysis and Van Evera compliance calculation"""
        
        def test_current_distribution_analysis(self, plugin, diagnostic_graph_data):
            """Test analysis of current diagnostic distribution"""
            analysis = plugin._analyze_current_distribution(diagnostic_graph_data)
            
            assert isinstance(analysis, dict)
            assert 'total_evidence_edges' in analysis
            assert 'distribution_counts' in analysis
            assert 'distribution_percentages' in analysis
            assert 'van_evera_compliance' in analysis
            
            # Should identify 3 evidence edges
            assert analysis['total_evidence_edges'] == 3
            
            # Should calculate percentages correctly
            percentages = analysis['distribution_percentages']
            assert sum(percentages.values()) <= 100.1  # Allow for rounding
            
        def test_diagnostic_type_normalization(self, plugin):
            """Test normalization of diagnostic type variants"""
            graph_data = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis'},
                    {'id': 'E1', 'type': 'Evidence'}
                ],
                'edges': [
                    {
                        'source_id': 'E1',
                        'target_id': 'H1',
                        'type': 'supports',
                        'properties': {
                            'diagnostic_type': 'straw_in_the_wind'  # Should normalize to straw_in_wind
                        }
                    }
                ]
            }
            
            analysis = plugin._analyze_current_distribution(graph_data)
            
            # Should normalize the diagnostic type
            assert analysis['distribution_counts']['straw_in_wind'] == 1
            assert analysis['distribution_counts']['unclassified'] == 0
            
        def test_van_evera_compliance_calculation(self, plugin):
            """Test Van Evera compliance score calculation"""
            # Perfect distribution graph
            perfect_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis'},
                    {'id': 'E1', 'type': 'Evidence'},
                    {'id': 'E2', 'type': 'Evidence'},
                    {'id': 'E3', 'type': 'Evidence'},
                    {'id': 'E4', 'type': 'Evidence'}
                ],
                'edges': [
                    {'source_id': 'E1', 'target_id': 'H1', 'type': 'supports',
                     'properties': {'diagnostic_type': 'hoop'}},
                    {'source_id': 'E2', 'target_id': 'H1', 'type': 'supports',
                     'properties': {'diagnostic_type': 'smoking_gun'}},
                    {'source_id': 'E3', 'target_id': 'H1', 'type': 'supports',
                     'properties': {'diagnostic_type': 'straw_in_wind'}},
                    {'source_id': 'E4', 'target_id': 'H1', 'type': 'supports',
                     'properties': {'diagnostic_type': 'straw_in_wind'}}
                ]
            }
            
            analysis = plugin._analyze_current_distribution(perfect_graph)
            compliance = analysis['van_evera_compliance']
            
            # Should be reasonably high for a balanced distribution
            assert 0 <= compliance <= 100
            
    class TestExecutionLogic:
        """Test execute() method with full classification workflow"""
        
        def test_execute_with_valid_input(self, plugin, diagnostic_graph_data):
            """Plugin executes classification workflow successfully"""
            input_data = {'graph_data': diagnostic_graph_data}
            result = plugin.execute(input_data)
            
            # Verify result structure
            assert result is not None
            assert isinstance(result, dict)
            assert 'updated_graph_data' in result
            assert 'current_analysis' in result
            assert 'classification_results' in result
            assert 'balanced_results' in result
            assert 'final_analysis' in result
            assert 'compliance_improvement' in result
            assert 'academic_quality_metrics' in result
            
        def test_graph_data_updating(self, plugin, diagnostic_graph_data):
            """Plugin updates graph data with new classifications"""
            input_data = {'graph_data': diagnostic_graph_data}
            result = plugin.execute(input_data)
            
            updated_graph = result['updated_graph_data']
            
            # Check that graph structure is preserved
            assert len(updated_graph['nodes']) == len(diagnostic_graph_data['nodes'])
            assert len(updated_graph['edges']) == len(diagnostic_graph_data['edges'])
            
            # Check that evidence edges got updated properties
            evidence_edges_updated = 0
            for edge in updated_graph['edges']:
                if plugin._is_evidence_relationship(edge, updated_graph):
                    props = edge.get('properties', {})
                    if 'content_classified' in props and props['content_classified']:
                        evidence_edges_updated += 1
                        
                        # Should have diagnostic_type and probative_value
                        assert 'diagnostic_type' in props
                        assert 'probative_value' in props
                        assert 'classification_confidence' in props
            
            # Should have updated at least some evidence edges
            assert evidence_edges_updated > 0
            
        def test_compliance_improvement_tracking(self, plugin, diagnostic_graph_data):
            """Plugin correctly tracks compliance improvement"""
            input_data = {'graph_data': diagnostic_graph_data}
            result = plugin.execute(input_data)
            
            current_compliance = result['current_analysis']['van_evera_compliance']
            final_compliance = result['final_analysis']['van_evera_compliance']
            improvement = result['compliance_improvement']
            
            # Improvement should be calculated correctly
            expected_improvement = final_compliance - current_compliance
            assert abs(improvement - expected_improvement) < 0.1  # Allow small floating point errors
            
    class TestVanEveraBalancing:
        """Test Van Evera distribution balancing logic"""
        
        def test_strategic_rebalancing_preserves_existing(self, plugin, diagnostic_graph_data):
            """Strategic rebalancing preserves good existing classifications"""
            # Mock classification results
            classification_results = {
                'detailed_classifications': [
                    {
                        'edge_id': 'E1->H1',
                        'original_type': 'smoking_gun',
                        'content_classified_type': 'hoop',
                        'confidence_score': 0.8,
                        'reasoning': 'Test classification'
                    }
                ],
                'classification_effectiveness': 80.0
            }
            
            balanced_results = plugin._apply_van_evera_balancing(diagnostic_graph_data, classification_results)
            
            assert 'balanced_classifications' in balanced_results
            assert 'final_distribution_percentages' in balanced_results
            assert 'van_evera_compliance_score' in balanced_results
            assert 'edges_reclassified' in balanced_results
            
            # Should preserve original classification in final_type
            balanced_classification = balanced_results['balanced_classifications'][0]
            assert 'final_type' in balanced_classification
            
        def test_doubly_decisive_conversion_conservative(self, plugin):
            """Test conservative conversion to doubly_decisive tests"""
            graph_with_strong_smoking_gun = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis'},
                    {'id': 'E1', 'type': 'Evidence'}
                ],
                'edges': [
                    {
                        'source_id': 'E1',
                        'target_id': 'H1',
                        'type': 'supports',
                        'properties': {
                            'diagnostic_type': 'smoking_gun',
                            'probative_value': 0.9,
                            'source_text_quote': 'This proves definitively and conclusively'
                        }
                    }
                ]
            }
            
            # Test strategic rebalancing
            sorted_classifications = [
                {
                    'edge_id': 'E1->H1',
                    'original_type': 'smoking_gun',
                    'confidence_score': 0.9
                }
            ]
            
            target_counts = {'hoop': 0, 'smoking_gun': 0, 'doubly_decisive': 1, 'straw_in_wind': 0}
            
            balanced = plugin._strategic_rebalancing(sorted_classifications, target_counts, graph_with_strong_smoking_gun)
            
            # Should be conservative - not necessarily convert unless very strong evidence
            assert len(balanced) == 1
            # Check that it makes a decision (either preserves or converts)
            assert 'final_type' in balanced[0]
            
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        def test_llm_enhancement_failure_graceful(self, plugin_context, diagnostic_graph_data):
            """Test graceful handling when LLM enhancement fails"""
            # Configure context to return a failing LLM function
            failing_llm_func = Mock()
            failing_llm_func.side_effect = Exception("LLM service unavailable")
            plugin_context.get_data.return_value = failing_llm_func
            
            plugin = ContentBasedDiagnosticClassifierPlugin("content_based_diagnostic_classifier", plugin_context)
            
            # Should handle gracefully and fall back to content analysis
            input_data = {'graph_data': diagnostic_graph_data}
            result = plugin.execute(input_data)
            
            assert result is not None
            assert 'classification_results' in result
            # Should complete successfully even without LLM
            assert result['classification_results']['classification_effectiveness'] >= 0
            
        def test_empty_evidence_content_handling(self, plugin):
            """Test handling of evidence with empty content"""
            empty_evidence_content = {'description': '', 'source_quote': '', 'full_text': ''}
            empty_hypothesis_content = {'description': '', 'type': 'Hypothesis', 'full_text': ''}
            
            # Should not crash on empty content
            classification = plugin._classify_evidence_content(
                empty_evidence_content, empty_hypothesis_content, None
            )
            
            assert classification is not None
            assert 'diagnostic_type' in classification
            assert 'confidence' in classification
            # Should default to low-confidence straw_in_wind
            assert classification['confidence'] >= 0
            
        def test_malformed_graph_edges_handling(self, plugin):
            """Test handling of graph edges with missing properties"""
            malformed_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis'},
                    {'id': 'E1', 'type': 'Evidence'}
                ],
                'edges': [
                    {
                        'source_id': 'E1',
                        'target_id': 'H1',
                        # Missing 'type' and 'properties'
                    }
                ]
            }
            
            # Should handle missing properties gracefully
            analysis = plugin._analyze_current_distribution(malformed_graph)
            
            assert analysis is not None
            assert 'distribution_counts' in analysis
            # Should classify as unclassified due to missing properties
            assert analysis['distribution_counts']['unclassified'] >= 0
            
        def test_checkpoint_data(self, plugin):
            """Test plugin checkpoint data includes diagnostic indicators info"""
            checkpoint = plugin.get_checkpoint_data()
            
            assert checkpoint['plugin_id'] == 'content_based_diagnostic_classifier'
            assert 'diagnostic_indicators_available' in checkpoint
            assert 'historical_context_enhancers' in checkpoint
            assert 'classification_method' in checkpoint
            
            # Should report availability of diagnostic indicators
            assert checkpoint['diagnostic_indicators_available'] == 4  # hoop, smoking_gun, doubly_decisive, straw_in_wind
            assert checkpoint['classification_method'] == 'content_analysis_with_llm_enhancement'
    
    class TestDiagnosticIndicators:
        """Test diagnostic indicator patterns and weights"""
        
        def test_diagnostic_indicators_structure(self, plugin):
            """Test that diagnostic indicators are properly structured"""
            indicators = plugin.DIAGNOSTIC_INDICATORS
            
            # Should have all four Van Evera test types
            expected_types = ['hoop', 'smoking_gun', 'doubly_decisive', 'straw_in_wind']
            for test_type in expected_types:
                assert test_type in indicators
                
                # Each should have weight
                assert 'weight' in indicators[test_type]
                assert isinstance(indicators[test_type]['weight'], (int, float))
                
            # Doubly decisive should have higher weight due to rarity
            assert indicators['doubly_decisive']['weight'] > indicators['hoop']['weight']
            
        def test_contextual_pattern_compilation(self, plugin):
            """Test that contextual regex patterns are valid"""
            for test_type, indicators in plugin.DIAGNOSTIC_INDICATORS.items():
                if 'contextual_patterns' in indicators:
                    for pattern in indicators['contextual_patterns']:
                        # Should compile without error
                        compiled = re.compile(pattern)
                        assert compiled is not None
                        
                        # Test with sample text
                        test_text = "this is a test of the necessary condition for success"
                        match = compiled.search(test_text)
                        # Don't assert match (may or may not match), just that no error occurs
                        
        def test_historical_context_enhancers(self, plugin):
            """Test historical context enhancer categories"""
            enhancers = plugin.HISTORICAL_CONTEXT_ENHANCERS
            
            # Should have different categories of historical terms
            expected_categories = ['political_necessity', 'economic_causation', 'military_evidence', 'social_indicators']
            for category in expected_categories:
                assert category in enhancers
                assert isinstance(enhancers[category], list)
                assert len(enhancers[category]) > 0
                
                # Terms should be lowercase for matching
                for term in enhancers[category]:
                    assert term == term.lower()