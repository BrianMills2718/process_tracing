"""Unit tests for VanEveraTestingPlugin - focused on Van Evera methodology logic"""

import pytest
import math
from unittest.mock import Mock, patch, MagicMock
from core.plugins.van_evera_testing import (
    VanEveraTestingPlugin, VanEveraTestingEngine, 
    TestResult, DiagnosticType, TestPrediction, TestEvaluation, HypothesisAssessment
)
from core.plugins.base import PluginContext, PluginValidationError

class TestVanEveraTestingPlugin:
    """Unit tests for VanEveraTestingPlugin"""
    
    @pytest.fixture
    def plugin_context(self):
        """Mock plugin context with LLM query function"""
        context = Mock(spec=PluginContext)
        context.config = {}
        context.data_bus = {}
        
        # Mock LLM query function
        mock_llm_func = Mock()
        mock_llm_func.return_value = {
            'testing_compliance_score': 75.0,
            'academic_quality_metrics': {'total_tests': 6}
        }
        context.get_data.return_value = mock_llm_func
        return context
        
    @pytest.fixture  
    def plugin(self, plugin_context):
        """Plugin instance with mocked context"""
        return VanEveraTestingPlugin("van_evera_testing", plugin_context)
        
    @pytest.fixture
    def van_evera_graph_data(self):
        """Complete Van Evera graph with hypotheses and evidence"""
        return {
            'nodes': [
                {
                    'id': 'Q_H1', 
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Taxation without representation was the primary cause of colonial resistance'
                    }
                },
                {
                    'id': 'Q_H2', 
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Economic self-interest drove colonial opposition to British policies'
                    }
                },
                {
                    'id': 'E1', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Boston Tea Party protest rhetoric focused on constitutional rights'
                    }
                },
                {
                    'id': 'E2', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Merchant boycotts increased with each new tax measure'
                    }
                }
            ],
            'edges': [
                {'source_id': 'E1', 'target_id': 'Q_H1', 'type': 'supports'},
                {'source_id': 'E2', 'target_id': 'Q_H2', 'type': 'provides_evidence_for'}
            ]
        }
        
    @pytest.fixture
    def invalid_graph_data(self):
        """Graph data missing required Van Evera components"""
        return {
            'nodes': [
                {'id': 'E1', 'type': 'Evidence', 'properties': {'description': 'Some evidence'}}
            ],
            'edges': []
        }
        
    class TestInputValidation:
        """Test validate_input() method for Van Evera requirements"""
        
        def test_valid_van_evera_input_accepted(self, plugin, van_evera_graph_data):
            """Plugin accepts valid Van Evera graph with hypotheses and evidence"""
            valid_input = {'graph_data': van_evera_graph_data}
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
            
        def test_graph_data_missing_nodes_rejected(self, plugin):
            """Plugin rejects graph_data without nodes"""
            invalid_data = {'graph_data': {'edges': []}}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(invalid_data)
            assert "graph_data must contain 'nodes' and 'edges'" in str(exc_info.value)
            
        def test_no_hypotheses_rejected(self, plugin, invalid_graph_data):
            """Plugin rejects graph with no hypotheses"""
            invalid_input = {'graph_data': invalid_graph_data}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(invalid_input)
            assert "No hypotheses found for Van Evera testing" in str(exc_info.value)
            
        def test_no_evidence_rejected(self, plugin):
            """Plugin rejects graph with no evidence"""
            no_evidence_data = {
                'nodes': [{'id': 'H1', 'type': 'Hypothesis', 'properties': {'description': 'Test'}}],
                'edges': []
            }
            invalid_input = {'graph_data': no_evidence_data}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(invalid_input)
            assert "No evidence found for Van Evera testing" in str(exc_info.value)
                
    class TestExecutionLogic:
        """Test execute() method with Van Evera methodology"""
        
        @patch('core.plugins.advanced_van_evera_prediction_engine.enhance_van_evera_testing_with_sophistication')
        def test_execute_with_valid_input(self, mock_enhance, plugin, van_evera_graph_data):
            """Plugin executes Van Evera testing successfully"""
            # Mock the advanced prediction engine response
            mock_enhance.return_value = {
                'testing_compliance_score': 75.0,
                'academic_quality_metrics': {
                    'total_tests': 6,
                    'academic_compliance_score': 75.0
                },
                'evaluation_results': {
                    'evaluations': [
                        {
                            'hypothesis_id': 'Q_H1',
                            'test_result': 'PASS',
                            'diagnostic_type': 'hoop',
                            'confidence_score': 0.8
                        }
                    ]
                }
            }
            
            input_data = {'graph_data': van_evera_graph_data}
            result = plugin.execute(input_data)
            
            # Verify result structure
            assert result is not None
            assert isinstance(result, dict)
            assert 'updated_graph_data' in result
            assert 'hypothesis_rankings' in result
            assert 'testing_compliance_score' in result
            assert result['testing_compliance_score'] == 75.0
            assert 'publication_ready' in result
            
            # Verify advanced engine was called
            mock_enhance.assert_called_once()
            
        @patch('core.plugins.advanced_van_evera_prediction_engine.enhance_van_evera_testing_with_sophistication')
        def test_hypothesis_ranking_calculation(self, mock_enhance, plugin, van_evera_graph_data):
            """Plugin correctly calculates hypothesis rankings based on Van Evera results"""
            # Mock advanced results with specific evaluation data
            mock_enhance.return_value = {
                'testing_compliance_score': 80.0,
                'academic_quality_metrics': {'total_tests': 4},
                'evaluation_results': {
                    'evaluations': [
                        {
                            'hypothesis_id': 'Q_H1',
                            'test_result': 'PASS',
                            'diagnostic_type': 'hoop',
                            'confidence_score': 0.9
                        },
                        {
                            'hypothesis_id': 'Q_H1', 
                            'test_result': 'PASS',
                            'diagnostic_type': 'smoking_gun',
                            'confidence_score': 0.8
                        },
                        {
                            'hypothesis_id': 'Q_H2',
                            'test_result': 'FAIL',
                            'diagnostic_type': 'hoop',
                            'confidence_score': 0.7
                        }
                    ]
                }
            }
            
            input_data = {'graph_data': van_evera_graph_data}
            result = plugin.execute(input_data)
            
            # Check hypothesis rankings
            rankings = result['hypothesis_rankings']
            assert 'Q_H1' in rankings
            assert 'Q_H2' in rankings
            
            # Q_H1 should rank higher due to passing tests
            assert rankings['Q_H1']['ranking_score'] > rankings['Q_H2']['ranking_score']
            assert rankings['Q_H1']['hypothesis_type'] == 'primary'  # Top ranking
            assert rankings['Q_H2']['hypothesis_type'] == 'alternative'
            
        @patch('core.plugins.advanced_van_evera_prediction_engine.enhance_van_evera_testing_with_sophistication')
        def test_graph_data_updating(self, mock_enhance, plugin, van_evera_graph_data):
            """Plugin updates graph data with ranking information"""
            mock_enhance.return_value = {
                'testing_compliance_score': 70.0,
                'academic_quality_metrics': {'total_tests': 3},
                'evaluation_results': {
                    'evaluations': [
                        {
                            'hypothesis_id': 'Q_H1',
                            'test_result': 'PASS',
                            'diagnostic_type': 'hoop',
                            'confidence_score': 0.85
                        }
                    ]
                }
            }
            
            input_data = {'graph_data': van_evera_graph_data}
            result = plugin.execute(input_data)
            
            # Check updated graph data
            updated_graph = result['updated_graph_data']
            
            # Find Q_H1 node and check for ranking properties
            h1_node = next((n for n in updated_graph['nodes'] if n['id'] == 'Q_H1'), None)
            assert h1_node is not None
            assert 'properties' in h1_node
            
            props = h1_node['properties']
            assert 'ranking_score' in props
            assert 'hypothesis_type' in props
            assert 'academic_rank' in props
            assert 'status' in props
            
            # Verify ranking score is reasonable (0-1 range)
            assert 0 <= props['ranking_score'] <= 1
            
    class TestVanEveraMethodology:
        """Test Van Evera specific methodology components"""
        
        def test_individual_ranking_score_calculation(self, plugin):
            """Test individual hypothesis ranking score calculation"""
            # Test with passing hoop test (high weight)
            evaluations = [
                {
                    'test_result': 'PASS',
                    'diagnostic_type': 'hoop',
                    'confidence_score': 0.9
                }
            ]
            score = plugin._calculate_individual_ranking_score(evaluations)
            assert 0.5 <= score <= 1.0  # Should be above or at neutral
            
            # Test with failing hoop test (should eliminate)
            evaluations = [
                {
                    'test_result': 'FAIL',
                    'diagnostic_type': 'hoop',
                    'confidence_score': 0.8
                }
            ]
            score = plugin._calculate_individual_ranking_score(evaluations)
            assert score <= 0.5  # Should be at or below neutral
            
        def test_diagnostic_test_weights(self, plugin):
            """Test that different diagnostic tests have appropriate weights"""
            # Smoking gun test should have higher impact than straw_in_the_wind
            smoking_gun_eval = [{
                'test_result': 'PASS',
                'diagnostic_type': 'smoking_gun',
                'confidence_score': 0.8
            }]
            
            straw_eval = [{
                'test_result': 'PASS',
                'diagnostic_type': 'straw_in_the_wind',
                'confidence_score': 0.8
            }]
            
            smoking_gun_score = plugin._calculate_individual_ranking_score(smoking_gun_eval)
            straw_score = plugin._calculate_individual_ranking_score(straw_eval)
            
            # Allow for small differences due to scoring algorithm
            assert smoking_gun_score >= straw_score
            
        def test_empty_evaluations_handling(self, plugin):
            """Test handling of hypotheses with no evaluations"""
            score = plugin._calculate_individual_ranking_score([])
            assert score == 0.0
            
        def test_checkpoint_data(self, plugin):
            """Test plugin checkpoint data includes Van Evera methodology info"""
            checkpoint = plugin.get_checkpoint_data()
            assert checkpoint['plugin_id'] == 'van_evera_testing'
            assert 'Van Evera' in checkpoint['methodology']
            assert 'diagnostic testing' in checkpoint['academic_standards']
            
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        @patch('core.plugins.advanced_van_evera_prediction_engine.enhance_van_evera_testing_with_sophistication')
        def test_missing_llm_function_handling(self, mock_enhance, plugin_context, van_evera_graph_data):
            """Test graceful handling when LLM function is not available"""
            # Configure context to return None for LLM function
            plugin_context.get_data.return_value = None
            plugin = VanEveraTestingPlugin("van_evera_testing", plugin_context)
            
            # Mock the advanced engine to still return results
            mock_enhance.return_value = {
                'testing_compliance_score': 60.0,
                'academic_quality_metrics': {'total_tests': 2},
                'evaluation_results': {'evaluations': []}
            }
            
            input_data = {'graph_data': van_evera_graph_data}
            
            # Should not raise exception, but handle gracefully
            result = plugin.execute(input_data)
            assert result is not None
            assert 'testing_compliance_score' in result
            
        @patch('core.plugins.advanced_van_evera_prediction_engine.enhance_van_evera_testing_with_sophistication')
        def test_advanced_engine_failure_handling(self, mock_enhance, plugin, van_evera_graph_data):
            """Test handling when advanced prediction engine fails"""
            # Mock the advanced engine to raise an exception
            mock_enhance.side_effect = Exception("Advanced engine failed")
            
            input_data = {'graph_data': van_evera_graph_data}
            
            # Should raise exception (plugin doesn't have fallback implemented)
            with pytest.raises(Exception) as exc_info:
                plugin.execute(input_data)
            assert "Advanced engine failed" in str(exc_info.value)
            
        def test_malformed_evaluation_results(self, plugin, van_evera_graph_data):
            """Test handling of malformed evaluation results"""
            # Mock advanced results with missing required fields
            advanced_results = {
                'testing_compliance_score': 50.0,
                'academic_quality_metrics': {'total_tests': 1},
                'evaluation_results': {
                    'evaluations': [
                        {
                            'hypothesis_id': 'Q_H1',
                            # Missing test_result, diagnostic_type, confidence_score
                        }
                    ]
                }
            }
            
            # Should handle missing fields gracefully
            rankings = plugin._calculate_hypothesis_rankings(van_evera_graph_data, advanced_results)
            
            # Should still create rankings entry
            assert 'Q_H1' in rankings
            ranking_score = rankings['Q_H1']['ranking_score']
            
            # Score should handle missing data gracefully (use defaults)
            assert 0 <= ranking_score <= 1


class TestVanEveraTestingEngine:
    """Unit tests for VanEveraTestingEngine core logic"""
    
    @pytest.fixture
    def sample_graph_data(self):
        """Sample graph data for testing engine"""
        return {
            'nodes': [
                {
                    'id': 'H1', 
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Taxation without representation caused resistance'
                    }
                },
                {
                    'id': 'E1', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Constitutional rhetoric in colonial documents'
                    }
                },
                {
                    'id': 'E2', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Evidence contradicting the hypothesis'
                    }
                }
            ],
            'edges': [
                {'source_id': 'E1', 'target_id': 'H1', 'type': 'supports'},
                {'source_id': 'E2', 'target_id': 'H1', 'type': 'contradicts'}
            ]
        }
        
    @pytest.fixture
    def testing_engine(self, sample_graph_data):
        """VanEveraTestingEngine instance"""
        return VanEveraTestingEngine(sample_graph_data)
        
    class TestEngineInitialization:
        """Test engine initialization and data parsing"""
        
        def test_engine_initialization(self, testing_engine, sample_graph_data):
            """Engine initializes with correct data structures"""
            assert testing_engine.graph_data == sample_graph_data
            assert len(testing_engine.hypotheses) == 1
            assert len(testing_engine.evidence) == 2
            assert len(testing_engine.evidence_edges) == 2
            
        def test_evidence_relationship_detection(self, testing_engine):
            """Engine correctly identifies evidence-hypothesis relationships"""
            # All edges should be evidence relationships
            for edge in testing_engine.evidence_edges:
                assert testing_engine._is_evidence_relationship(edge)
                
    class TestPredictionGeneration:
        """Test testable prediction generation"""
        
        def test_prediction_generation_for_taxation_hypothesis(self, testing_engine):
            """Engine generates appropriate predictions for taxation hypothesis"""
            hypothesis = testing_engine.hypotheses[0]
            predictions = testing_engine._generate_testable_predictions(hypothesis)
            
            assert len(predictions) >= 2  # Should generate multiple predictions
            assert all(isinstance(p, TestPrediction) for p in predictions)
            
            # Check that predictions have required fields
            for pred in predictions:
                assert pred.prediction_id.startswith('H1_PRED')
                assert pred.hypothesis_id == 'H1'
                assert len(pred.description) > 0
                assert isinstance(pred.diagnostic_type, DiagnosticType)
                assert isinstance(pred.necessary_condition, bool)
                assert isinstance(pred.sufficient_condition, bool)
                
        def test_diagnostic_types_generated(self, testing_engine):
            """Engine generates different diagnostic test types"""
            hypothesis = testing_engine.hypotheses[0]
            predictions = testing_engine._generate_testable_predictions(hypothesis)
            
            diagnostic_types = [p.diagnostic_type for p in predictions]
            
            # Should include both hoop and smoking gun tests for comprehensive analysis
            assert DiagnosticType.HOOP in diagnostic_types
            assert DiagnosticType.SMOKING_GUN in diagnostic_types
            
    class TestDiagnosticLogic:
        """Test Van Evera diagnostic test logic"""
        
        def test_hoop_test_logic(self, testing_engine):
            """Test hoop test (necessary condition) logic"""
            hoop_prediction = TestPrediction(
                prediction_id="TEST_HOOP",
                hypothesis_id="H1",
                description="Test hoop prediction",
                diagnostic_type=DiagnosticType.HOOP,
                necessary_condition=True,
                sufficient_condition=False,
                evidence_requirements=["test_evidence"]
            )
            
            # Test PASS case
            result, reasoning = testing_engine._apply_diagnostic_logic(hoop_prediction, ['E1'], [])
            assert result == TestResult.PASS
            assert "HOOP TEST PASSED" in reasoning
            
            # Test FAIL case (no supporting evidence)
            result, reasoning = testing_engine._apply_diagnostic_logic(hoop_prediction, [], [])
            assert result == TestResult.FAIL
            assert "HOOP TEST FAILED" in reasoning
            
            # Test FAIL case (contradicting evidence)
            result, reasoning = testing_engine._apply_diagnostic_logic(hoop_prediction, ['E1'], ['E2'])
            assert result == TestResult.FAIL
            assert "HOOP TEST FAILED" in reasoning
            
        def test_smoking_gun_test_logic(self, testing_engine):
            """Test smoking gun test (sufficient condition) logic"""
            smoking_gun_prediction = TestPrediction(
                prediction_id="TEST_SMOKING_GUN",
                hypothesis_id="H1",
                description="Test smoking gun prediction",
                diagnostic_type=DiagnosticType.SMOKING_GUN,
                necessary_condition=False,
                sufficient_condition=True,
                evidence_requirements=["decisive_evidence"]
            )
            
            # Test PASS case
            result, reasoning = testing_engine._apply_diagnostic_logic(smoking_gun_prediction, ['E1'], [])
            assert result == TestResult.PASS
            assert "SMOKING GUN PASSED" in reasoning
            
            # Test FAIL case (contradicting evidence)
            result, reasoning = testing_engine._apply_diagnostic_logic(smoking_gun_prediction, ['E1'], ['E2'])
            assert result == TestResult.FAIL
            assert "SMOKING GUN FAILED" in reasoning
            
            # Test INCONCLUSIVE case (no evidence)
            result, reasoning = testing_engine._apply_diagnostic_logic(smoking_gun_prediction, [], [])
            assert result == TestResult.INCONCLUSIVE
            assert "SMOKING GUN INCONCLUSIVE" in reasoning
            
        def test_doubly_decisive_test_logic(self, testing_engine):
            """Test doubly decisive test (both necessary and sufficient) logic"""
            doubly_decisive_prediction = TestPrediction(
                prediction_id="TEST_DOUBLY_DECISIVE",
                hypothesis_id="H1",
                description="Test doubly decisive prediction",
                diagnostic_type=DiagnosticType.DOUBLY_DECISIVE,
                necessary_condition=True,
                sufficient_condition=True,
                evidence_requirements=["decisive_evidence"]
            )
            
            # Test PASS case (support and no contradiction)
            result, reasoning = testing_engine._apply_diagnostic_logic(doubly_decisive_prediction, ['E1'], [])
            assert result == TestResult.PASS
            assert "DOUBLY DECISIVE PASSED" in reasoning
            
            # Test FAIL case (any contradiction or no support)
            result, reasoning = testing_engine._apply_diagnostic_logic(doubly_decisive_prediction, [], [])
            assert result == TestResult.FAIL
            assert "DOUBLY DECISIVE FAILED" in reasoning
            
    class TestBayesianUpdating:
        """Test Bayesian probability updating logic"""
        
        def test_posterior_probability_calculation(self, testing_engine):
            """Test Bayesian posterior probability calculation"""
            # Create mock test results
            passing_result = Mock(spec=TestEvaluation)
            passing_result.test_result = TestResult.PASS
            passing_result.confidence_level = 0.8
            
            failing_result = Mock(spec=TestEvaluation)
            failing_result.test_result = TestResult.FAIL
            failing_result.confidence_level = 0.7
            
            # Test with passing result only
            prior = 0.5
            posterior_pass = testing_engine._calculate_posterior_probability(prior, [passing_result])
            assert posterior_pass > prior  # Should increase probability
            
            # Test with failing result only
            posterior_fail = testing_engine._calculate_posterior_probability(prior, [failing_result])
            assert posterior_fail < prior  # Should decrease probability
            
            # Test with mixed results
            posterior_mixed = testing_engine._calculate_posterior_probability(
                prior, [passing_result, failing_result]
            )
            # Mixed results - depends on relative confidence levels
            assert 0 <= posterior_mixed <= 1
            
        def test_confidence_interval_calculation(self, testing_engine):
            """Test confidence interval calculation"""
            posterior = 0.7
            
            # Test with no test results
            ci_none = testing_engine._calculate_confidence_interval(posterior, [])
            assert ci_none[0] <= posterior <= ci_none[1]
            assert abs((ci_none[1] - ci_none[0]) - 0.6) < 0.01  # Wide interval (allow small float precision errors)
            
            # Test with multiple test results (should narrow interval)
            mock_results = [Mock() for _ in range(4)]
            ci_many = testing_engine._calculate_confidence_interval(posterior, mock_results)
            assert ci_many[0] <= posterior <= ci_many[1]
            assert ci_many[1] - ci_many[0] < ci_none[1] - ci_none[0]  # Narrower interval
            
    class TestAcademicConclusions:
        """Test academic conclusion generation"""
        
        def test_academic_conclusion_structure(self, testing_engine):
            """Test academic conclusion contains required elements"""
            hypothesis = testing_engine.hypotheses[0]
            
            # Create mock test results
            hoop_result = Mock(spec=TestEvaluation)
            hoop_result.test_result = TestResult.PASS
            hoop_result.reasoning = "HOOP TEST PASSED: Evidence supports necessary condition"
            
            smoking_gun_result = Mock(spec=TestEvaluation)
            smoking_gun_result.test_result = TestResult.PASS
            smoking_gun_result.reasoning = "SMOKING GUN PASSED: Decisive evidence found"
            
            test_results = [hoop_result, smoking_gun_result]
            posterior = 0.85
            
            conclusion = testing_engine._generate_academic_conclusion(hypothesis, test_results, posterior)
            
            # Check required elements
            assert "HYPOTHESIS:" in conclusion
            assert "NECESSARY CONDITION ANALYSIS:" in conclusion
            assert "SUFFICIENT CONDITION ANALYSIS:" in conclusion
            assert "OVERALL ASSESSMENT:" in conclusion
            assert f"Posterior probability = {posterior:.2f}" in conclusion
            
        def test_elimination_conclusion(self, testing_engine):
            """Test conclusion when hypothesis is eliminated"""
            hypothesis = testing_engine.hypotheses[0]
            
            # Create failing hoop test (should eliminate hypothesis)
            hoop_fail_result = Mock(spec=TestEvaluation)
            hoop_fail_result.test_result = TestResult.FAIL
            hoop_fail_result.reasoning = "HOOP TEST FAILED: Necessary condition not met"
            
            conclusion = testing_engine._generate_academic_conclusion(hypothesis, [hoop_fail_result], 0.2)
            
            assert "ELIMINATED" in conclusion
            assert "0/1 hoop tests passed" in conclusion
