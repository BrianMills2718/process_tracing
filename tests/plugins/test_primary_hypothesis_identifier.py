"""Unit tests for PrimaryHypothesisIdentifierPlugin - focused on hypothesis ranking logic"""

import pytest
import networkx as nx
from unittest.mock import Mock, patch
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin
from core.plugins.base import PluginContext, PluginValidationError


class TestPrimaryHypothesisIdentifierPlugin:
    """Unit tests for PrimaryHypothesisIdentifierPlugin"""
    
    @pytest.fixture
    def plugin_context(self):
        """Mock plugin context with minimal dependencies"""
        context = Mock(spec=PluginContext)
        context.config = {}
        context.data_bus = {}
        return context
        
    @pytest.fixture  
    def plugin(self, plugin_context):
        """Plugin instance with mocked context"""
        return PrimaryHypothesisIdentifierPlugin("primary_hypothesis_identifier", plugin_context)
        
    @pytest.fixture
    def multi_hypothesis_graph_data(self):
        """Graph data with multiple hypotheses for ranking"""
        return {
            'nodes': [
                {
                    'id': 'H1',
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Economic mechanism theory explains causal process through institutional framework'
                    }
                },
                {
                    'id': 'H2', 
                    'type': 'Alternative_Explanation',
                    'properties': {
                        'description': 'Political strategic action led to systematic change through rational incentives'
                    }
                },
                {
                    'id': 'H3',
                    'type': 'Hypothesis',
                    'properties': {
                        'description': 'Cultural factors influenced outcomes'
                    }
                },
                {
                    'id': 'E1',
                    'type': 'Evidence',
                    'properties': {'description': 'Economic data evidence'}
                },
                {
                    'id': 'E2',
                    'type': 'Evidence',
                    'properties': {'description': 'Political evidence'}
                },
                {
                    'id': 'E3',
                    'type': 'Evidence', 
                    'properties': {'description': 'Cultural evidence'}
                }
            ],
            'edges': [
                {
                    'source_id': 'E1',
                    'target_id': 'H1',
                    'type': 'supports',
                    'properties': {
                        'diagnostic_type': 'doubly_decisive',
                        'probative_value': 0.9,
                        'certainty': 0.8
                    }
                },
                {
                    'source_id': 'E2',
                    'target_id': 'H1',
                    'type': 'supports',
                    'properties': {
                        'diagnostic_type': 'smoking_gun',
                        'probative_value': 0.8
                    }
                },
                {
                    'source_id': 'E2',
                    'target_id': 'H2',
                    'type': 'supports',
                    'properties': {
                        'diagnostic_type': 'hoop',
                        'probative_value': 0.7
                    }
                },
                {
                    'source_id': 'E3',
                    'target_id': 'H3',
                    'type': 'supports',
                    'properties': {
                        'diagnostic_type': 'straw_in_the_wind',
                        'probative_value': 0.4
                    }
                }
            ]
        }
        
    @pytest.fixture
    def van_evera_results(self):
        """Van Evera analysis results with hypothesis rankings"""
        return {
            'hypothesis_rankings': {
                'H1': {
                    'ranking_score': 0.85,
                    'diagnostic_tests_passed': 3,
                    'test_quality': 'high'
                },
                'H2': {
                    'ranking_score': 0.72,
                    'diagnostic_tests_passed': 2,
                    'test_quality': 'moderate'
                },
                'H3': {
                    'ranking_score': 0.45,
                    'diagnostic_tests_passed': 1,
                    'test_quality': 'low'
                }
            },
            'diagnostic_results': [
                {
                    'hypothesis_id': 'H1',
                    'diagnostic_type': 'doubly_decisive',
                    'test_result': 'PASS'
                },
                {
                    'hypothesis_id': 'H2',
                    'diagnostic_type': 'hoop',
                    'test_result': 'PASS'
                },
                {
                    'hypothesis_id': 'H1',
                    'diagnostic_type': 'smoking_gun',
                    'test_result': 'PASS'
                }
            ],
            'elimination_analysis': {
                'H1': {
                    'hypotheses_eliminated': 1,
                    'total_alternatives': 2
                },
                'H2': {
                    'hypotheses_eliminated': 0,
                    'total_alternatives': 2
                },
                'H3': {
                    'hypotheses_eliminated': 0,
                    'total_alternatives': 2
                }
            }
        }
        
    @pytest.fixture
    def complete_input_data(self, multi_hypothesis_graph_data, van_evera_results):
        """Complete input data for plugin execution"""
        return {
            'graph_data': multi_hypothesis_graph_data,
            'van_evera_results': van_evera_results
        }
        
    class TestInputValidation:
        """Test validate_input() method thoroughly"""
        
        def test_valid_input_accepted(self, plugin, complete_input_data):
            """Plugin accepts valid input with graph data and Van Evera results"""
            # Should not raise exception
            plugin.validate_input(complete_input_data)
            
        def test_invalid_input_rejected(self, plugin):
            """Plugin rejects invalid input with clear error"""
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input("invalid_input")
            assert "must be dictionary" in str(exc_info.value).lower()
            
        def test_missing_graph_data_rejected(self, plugin, van_evera_results):
            """Plugin rejects input missing graph_data"""
            input_data = {'van_evera_results': van_evera_results}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(input_data)
            assert "graph_data" in str(exc_info.value)
            
        def test_missing_van_evera_results_rejected(self, plugin, multi_hypothesis_graph_data):
            """Plugin rejects input missing van_evera_results"""
            input_data = {'graph_data': multi_hypothesis_graph_data}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(input_data)
            assert "van_evera_results" in str(exc_info.value)
            
        def test_insufficient_hypotheses_rejected(self, plugin, van_evera_results):
            """Plugin rejects input with less than 2 hypotheses"""
            # Graph with only 1 hypothesis
            graph_data = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis', 'properties': {'description': 'Single hypothesis'}},
                    {'id': 'E1', 'type': 'Evidence', 'properties': {'description': 'Evidence'}}
                ],
                'edges': []
            }
            input_data = {'graph_data': graph_data, 'van_evera_results': van_evera_results}
            
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(input_data)
            assert "at least 2 hypotheses" in str(exc_info.value)
            
        def test_invalid_van_evera_results_rejected(self, plugin, multi_hypothesis_graph_data):
            """Plugin rejects input with invalid van_evera_results format"""
            input_data = {
                'graph_data': multi_hypothesis_graph_data,
                'van_evera_results': "invalid_format"
            }
            
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(input_data)
            assert "must be dictionary" in str(exc_info.value)
                
    class TestRankingCalculation:
        """Test hypothesis ranking calculation logic"""
        
        def test_ranking_calculation_produces_scores(self, plugin, complete_input_data):
            """Plugin calculates ranking scores for all hypotheses"""
            graph_data = complete_input_data['graph_data']
            van_evera_results = complete_input_data['van_evera_results']
            
            ranking_analysis = plugin._calculate_hypothesis_rankings(graph_data, van_evera_results)
            
            assert 'hypothesis_scores' in ranking_analysis
            assert 'ranked_hypotheses' in ranking_analysis
            assert len(ranking_analysis['hypothesis_scores']) == 3  # H1, H2, H3
            assert len(ranking_analysis['ranked_hypotheses']) == 3
            
        def test_ranking_sorts_by_composite_score(self, plugin, complete_input_data):
            """Plugin sorts hypotheses by composite score in descending order"""
            graph_data = complete_input_data['graph_data']
            van_evera_results = complete_input_data['van_evera_results']
            
            ranking_analysis = plugin._calculate_hypothesis_rankings(graph_data, van_evera_results)
            ranked_hypotheses = ranking_analysis['ranked_hypotheses']
            
            # Should be sorted in descending order
            for i in range(len(ranked_hypotheses) - 1):
                current_score = ranked_hypotheses[i][1]['composite_score']
                next_score = ranked_hypotheses[i + 1][1]['composite_score']
                assert current_score >= next_score
            
        def test_component_scores_calculated(self, plugin, complete_input_data):
            """Plugin calculates all component scores for each hypothesis"""
            graph_data = complete_input_data['graph_data']
            van_evera_results = complete_input_data['van_evera_results']
            
            ranking_analysis = plugin._calculate_hypothesis_rankings(graph_data, van_evera_results)
            
            for hypothesis_id, scores in ranking_analysis['hypothesis_scores'].items():
                component_scores = scores['component_scores']
                assert 'van_evera_score' in component_scores
                assert 'evidence_support' in component_scores
                assert 'theoretical_sophistication' in component_scores
                assert 'elimination_power' in component_scores
                
                # All scores should be between 0 and 1
                for score_name, score_value in component_scores.items():
                    assert 0 <= score_value <= 1, f"{score_name} score {score_value} not in range [0,1]"
                    
        def test_academic_strengths_identified(self, plugin, complete_input_data):
            """Plugin identifies academic strengths based on component scores"""
            graph_data = complete_input_data['graph_data']
            van_evera_results = complete_input_data['van_evera_results']
            
            ranking_analysis = plugin._calculate_hypothesis_rankings(graph_data, van_evera_results)
            
            for hypothesis_id, scores in ranking_analysis['hypothesis_scores'].items():
                assert 'academic_strengths' in scores
                assert isinstance(scores['academic_strengths'], list)
                
        def test_ranking_eligibility_assessed(self, plugin, complete_input_data):
            """Plugin assesses ranking eligibility for each hypothesis"""
            graph_data = complete_input_data['graph_data']
            van_evera_results = complete_input_data['van_evera_results']
            
            ranking_analysis = plugin._calculate_hypothesis_rankings(graph_data, van_evera_results)
            
            for hypothesis_id, scores in ranking_analysis['hypothesis_scores'].items():
                eligibility = scores['ranking_eligibility']
                assert 'eligible_for_primary' in eligibility
                assert 'failed_criteria' in eligibility
                assert 'eligibility_score' in eligibility
                assert isinstance(eligibility['eligible_for_primary'], bool)
                assert isinstance(eligibility['failed_criteria'], list)
                
    class TestVanEveraScoreExtraction:
        """Test Van Evera score extraction from various result formats"""
        
        def test_extract_from_hypothesis_rankings(self, plugin, van_evera_results):
            """Plugin extracts Van Evera score from hypothesis_rankings format"""
            score = plugin._extract_van_evera_score('H1', van_evera_results)
            assert score == 0.85  # From hypothesis_rankings fixture
            
        def test_extract_from_legacy_ranking_scores(self, plugin):
            """Plugin extracts Van Evera score from legacy ranking_scores format"""
            legacy_results = {
                'ranking_scores': {
                    'H1': {'score': 0.78}
                }
            }
            score = plugin._extract_van_evera_score('H1', legacy_results)
            assert score == 0.78
            
        def test_extract_from_hypothesis_evaluations(self, plugin):
            """Plugin extracts Van Evera score from hypothesis_evaluations format"""
            eval_results = {
                'hypothesis_evaluations': [
                    {'hypothesis_id': 'H1', 'composite_score': 0.82},
                    {'hypothesis_id': 'H2', 'composite_score': 0.67}
                ]
            }
            score = plugin._extract_van_evera_score('H1', eval_results)
            assert score == 0.82
            
        def test_extract_default_when_not_found(self, plugin):
            """Plugin returns default score when hypothesis not found in Van Evera results"""
            empty_results = {}
            score = plugin._extract_van_evera_score('UNKNOWN_ID', empty_results)
            assert score == 0.5  # Default score
            
    class TestEvidenceSupportCalculation:
        """Test evidence support calculation logic"""
        
        def test_evidence_support_with_supporting_edges(self, plugin, multi_hypothesis_graph_data):
            """Plugin calculates evidence support score based on supporting edges"""
            hypotheses = [n for n in multi_hypothesis_graph_data['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
            evidence_nodes = [n for n in multi_hypothesis_graph_data['nodes'] if n.get('type') == 'Evidence']
            edges = multi_hypothesis_graph_data['edges']
            
            # H1 should have highest evidence support (2 supporting edges with high probative values)
            h1 = next(h for h in hypotheses if h['id'] == 'H1')
            h1_support = plugin._calculate_evidence_support(h1, evidence_nodes, edges)
            
            # H3 should have lowest evidence support (1 edge with low probative value)  
            h3 = next(h for h in hypotheses if h['id'] == 'H3')
            h3_support = plugin._calculate_evidence_support(h3, evidence_nodes, edges)
            
            assert h1_support > h3_support
            assert 0 <= h1_support <= 1
            assert 0 <= h3_support <= 1
            
        def test_evidence_support_with_no_supporting_edges(self, plugin, multi_hypothesis_graph_data):
            """Plugin returns minimal score for hypothesis with no supporting evidence"""
            # Create hypothesis with no supporting edges
            hypothesis_no_support = {
                'id': 'H_NO_SUPPORT',
                'type': 'Hypothesis',
                'properties': {'description': 'Unsupported hypothesis'}
            }
            
            evidence_nodes = [n for n in multi_hypothesis_graph_data['nodes'] if n.get('type') == 'Evidence']
            edges = multi_hypothesis_graph_data['edges']  # No edges to H_NO_SUPPORT
            
            support_score = plugin._calculate_evidence_support(hypothesis_no_support, evidence_nodes, edges)
            assert support_score == 0.1  # Minimal score for no evidence
            
        def test_diagnostic_type_weighting(self, plugin):
            """Plugin applies diagnostic type weighting correctly"""
            hypothesis = {'id': 'H_TEST', 'type': 'Hypothesis', 'properties': {'description': 'Test hypothesis'}}
            evidence_nodes = [{'id': 'E1', 'type': 'Evidence', 'properties': {'description': 'Evidence'}}]
            
            # Test different diagnostic types
            doubly_decisive_edge = [{
                'source_id': 'E1', 'target_id': 'H_TEST', 'type': 'supports',
                'properties': {'diagnostic_type': 'doubly_decisive', 'probative_value': 1.0}
            }]
            
            straw_wind_edge = [{
                'source_id': 'E1', 'target_id': 'H_TEST', 'type': 'supports', 
                'properties': {'diagnostic_type': 'straw_in_the_wind', 'probative_value': 1.0}
            }]
            
            doubly_decisive_score = plugin._calculate_evidence_support(hypothesis, evidence_nodes, doubly_decisive_edge)
            straw_wind_score = plugin._calculate_evidence_support(hypothesis, evidence_nodes, straw_wind_edge)
            
            assert doubly_decisive_score > straw_wind_score
            
    class TestTheoreticalSophistication:
        """Test theoretical sophistication calculation"""
        
        def test_theoretical_concepts_scoring(self, plugin):
            """Plugin scores theoretical concepts in hypothesis descriptions"""
            # High sophistication hypothesis
            sophisticated_hyp = {
                'id': 'H1',
                'type': 'Hypothesis', 
                'properties': {
                    'description': 'causal mechanism explains institutional process through theoretical framework and systematic model'
                }
            }
            
            # Low sophistication hypothesis
            simple_hyp = {
                'id': 'H2',
                'type': 'Hypothesis',
                'properties': {
                    'description': 'something happened'
                }
            }
            
            sophisticated_score = plugin._calculate_theoretical_sophistication(sophisticated_hyp)
            simple_score = plugin._calculate_theoretical_sophistication(simple_hyp)
            
            assert sophisticated_score > simple_score
            assert 0 <= sophisticated_score <= 1
            assert 0 <= simple_score <= 1
            
        def test_causal_language_bonus(self, plugin):
            """Plugin awards bonus for causal language"""
            causal_hyp = {
                'id': 'H1',
                'type': 'Hypothesis',
                'properties': {
                    'description': 'economic factors caused political change because they led to institutional reform'
                }
            }
            
            non_causal_hyp = {
                'id': 'H2', 
                'type': 'Hypothesis',
                'properties': {
                    'description': 'economic factors and political change and institutional reform'
                }
            }
            
            causal_score = plugin._calculate_theoretical_sophistication(causal_hyp)
            non_causal_score = plugin._calculate_theoretical_sophistication(non_causal_hyp)
            
            assert causal_score > non_causal_score
            
        def test_complexity_indicators_bonus(self, plugin):
            """Plugin awards bonus for complexity indicators"""
            complex_hyp = {
                'id': 'H1',
                'type': 'Hypothesis', 
                'properties': {
                    'description': 'strategic interaction created dynamic feedback with contingent outcomes and rational incentives'
                }
            }
            
            simple_hyp = {
                'id': 'H2',
                'type': 'Hypothesis',
                'properties': {
                    'description': 'economic change occurred in political context'
                }
            }
            
            complex_score = plugin._calculate_theoretical_sophistication(complex_hyp)
            simple_score = plugin._calculate_theoretical_sophistication(simple_hyp)
            
            assert complex_score > simple_score
            
    class TestEliminationPower:
        """Test elimination power calculation"""
        
        def test_elimination_from_analysis(self, plugin, van_evera_results):
            """Plugin calculates elimination power from Van Evera elimination analysis"""
            # H1 eliminated 1 of 2 alternatives
            h1_elimination = plugin._calculate_elimination_power('H1', van_evera_results)
            
            # H2 eliminated 0 of 2 alternatives  
            h2_elimination = plugin._calculate_elimination_power('H2', van_evera_results)
            
            assert h1_elimination > h2_elimination
            # H1 gets 0.5 from elimination (1/2) + 0.2 bonus from doubly_decisive test = 0.7
            assert h1_elimination >= 0.5  # At least base elimination score
            assert h2_elimination == 0.0  # 0/2 alternatives eliminated, no decisive tests
            
        def test_elimination_bonus_for_decisive_tests(self, plugin, van_evera_results):
            """Plugin awards bonus for decisive diagnostic test passes"""
            # H1 has doubly_decisive test that passed
            h1_elimination = plugin._calculate_elimination_power('H1', van_evera_results)
            
            # H3 has no decisive tests
            h3_elimination = plugin._calculate_elimination_power('H3', van_evera_results)
            
            # H1 should get bonus for decisive test
            assert h1_elimination > h3_elimination
            
        def test_elimination_power_capped_at_one(self, plugin):
            """Plugin caps elimination power at maximum 1.0"""
            # Van Evera results with high elimination values
            high_elimination_results = {
                'elimination_analysis': {
                    'H1': {
                        'hypotheses_eliminated': 10,
                        'total_alternatives': 2  # This would give 5.0 without cap
                    }
                },
                'diagnostic_results': [
                    {'hypothesis_id': 'H1', 'diagnostic_type': 'doubly_decisive', 'test_result': 'PASS'}
                ]
            }
            
            elimination_power = plugin._calculate_elimination_power('H1', high_elimination_results)
            assert elimination_power <= 1.0
            
    class TestPrimaryHypothesisIdentification:
        """Test primary hypothesis identification logic"""
        
        def test_primary_hypothesis_selection(self, plugin, complete_input_data):
            """Plugin identifies primary hypothesis based on ranking"""
            ranking_analysis = plugin._calculate_hypothesis_rankings(
                complete_input_data['graph_data'], 
                complete_input_data['van_evera_results']
            )
            
            primary_identification = plugin._identify_primary_hypothesis(ranking_analysis)
            
            assert 'primary_hypothesis' in primary_identification
            assert 'alternative_hypotheses' in primary_identification
            assert 'selection_confidence' in primary_identification
            
            primary = primary_identification['primary_hypothesis']
            assert primary['new_id'] == 'Q_H1'
            assert 'original_id' in primary
            assert 'composite_score' in primary
            
        def test_alternative_hypothesis_assignment(self, plugin, complete_input_data):
            """Plugin assigns Q_H2, Q_H3 etc to alternative hypotheses"""
            ranking_analysis = plugin._calculate_hypothesis_rankings(
                complete_input_data['graph_data'], 
                complete_input_data['van_evera_results']
            )
            
            primary_identification = plugin._identify_primary_hypothesis(ranking_analysis)
            alternatives = primary_identification['alternative_hypotheses']
            
            # Should have 2 alternatives (H2, H3) 
            assert len(alternatives) == 2
            
            # Check Q_H2, Q_H3 assignment
            new_ids = [alt['new_id'] for alt in alternatives]
            assert 'Q_H2' in new_ids
            assert 'Q_H3' in new_ids
            
        def test_selection_confidence_calculation(self, plugin, complete_input_data):
            """Plugin calculates selection confidence based on score gaps"""
            ranking_analysis = plugin._calculate_hypothesis_rankings(
                complete_input_data['graph_data'], 
                complete_input_data['van_evera_results']
            )
            
            primary_identification = plugin._identify_primary_hypothesis(ranking_analysis)
            confidence = primary_identification['selection_confidence']
            
            assert 'confidence_level' in confidence
            assert 'reason' in confidence
            assert 'score_gap' in confidence
            assert 0 <= confidence['confidence_level'] <= 1
            
        def test_fallback_when_no_eligible_hypothesis(self, plugin, multi_hypothesis_graph_data):
            """Plugin falls back to top-ranked hypothesis when none meet eligibility criteria"""
            # Van Evera results with very low scores (won't meet eligibility thresholds)
            low_score_results = {
                'hypothesis_rankings': {
                    'H1': {'ranking_score': 0.3},  # Below 0.6 threshold
                    'H2': {'ranking_score': 0.2}, 
                    'H3': {'ranking_score': 0.1}
                }
            }
            
            ranking_analysis = plugin._calculate_hypothesis_rankings(multi_hypothesis_graph_data, low_score_results)
            primary_identification = plugin._identify_primary_hypothesis(ranking_analysis)
            
            # Should still select primary (H1 as highest scoring)
            assert primary_identification['primary_hypothesis']['new_id'] == 'Q_H1'
            # But eligibility status should show it didn't meet criteria
            assert not primary_identification['primary_hypothesis']['eligibility_status']['eligible_for_primary']
            
    class TestExecutionLogic:
        """Test execute() method with complete workflow"""
        
        def test_execute_with_valid_input(self, plugin, complete_input_data):
            """Plugin executes successfully with valid input data"""
            result = plugin.execute(complete_input_data)
            
            assert result is not None
            assert 'primary_identification' in result
            assert 'ranking_analysis' in result
            assert 'academic_justification' in result
            assert 'updated_graph_data' in result
            assert 'methodology' in result
            assert 'academic_quality_indicators' in result
            
        def test_graph_update_with_new_ids(self, plugin, complete_input_data):
            """Plugin updates graph data with Q_H1/H2/H3 node IDs"""
            result = plugin.execute(complete_input_data)
            updated_graph = result['updated_graph_data']
            
            # Check that hypothesis nodes have new Q_H1/H2/H3 IDs
            hypothesis_nodes = [n for n in updated_graph['nodes'] if n.get('type') in ['Hypothesis', 'Alternative_Explanation']]
            hypothesis_ids = [node['id'] for node in hypothesis_nodes]
            
            assert 'Q_H1' in hypothesis_ids
            assert 'Q_H2' in hypothesis_ids  
            assert 'Q_H3' in hypothesis_ids
            
        def test_edge_updates_with_new_ids(self, plugin, complete_input_data):
            """Plugin updates edges to reference new Q_H1/H2/H3 node IDs"""
            result = plugin.execute(complete_input_data)
            updated_graph = result['updated_graph_data']
            
            # Check that edges reference new hypothesis IDs
            hypothesis_edges = [
                edge for edge in updated_graph['edges'] 
                if edge.get('target_id', '').startswith('Q_H') or edge.get('source_id', '').startswith('Q_H')
            ]
            
            assert len(hypothesis_edges) > 0  # Should have edges with Q_H references
            
        def test_academic_justification_generation(self, plugin, complete_input_data):
            """Plugin generates academic justification for primary hypothesis selection"""
            result = plugin.execute(complete_input_data)
            justification = result['academic_justification']
            
            assert 'justification_text' in justification
            assert 'methodology' in justification
            assert 'selection_transparency' in justification
            assert 'academic_standards_met' in justification
            
            # Check justification text contains key elements
            text = justification['justification_text']
            assert 'Q_H1' in text
            assert 'Van Evera' in text
            assert 'systematic' in text.lower()
            
        def test_execute_idempotent(self, plugin, complete_input_data):
            """Plugin execution is idempotent (same input = same primary hypothesis)"""
            result1 = plugin.execute(complete_input_data.copy())
            result2 = plugin.execute(complete_input_data.copy())
            
            # Same primary hypothesis should be selected
            primary1 = result1['primary_identification']['primary_hypothesis']['original_id']
            primary2 = result2['primary_identification']['primary_hypothesis']['original_id']
            assert primary1 == primary2
            
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        def test_empty_van_evera_results_handling(self, plugin, multi_hypothesis_graph_data):
            """Plugin handles empty Van Evera results gracefully"""
            empty_results = {}
            input_data = {
                'graph_data': multi_hypothesis_graph_data,
                'van_evera_results': empty_results
            }
            
            # Should not crash, should use default scores
            result = plugin.execute(input_data)
            assert result is not None
            assert 'primary_identification' in result
            
        def test_single_hypothesis_after_validation(self, plugin):
            """Plugin handles edge case scenarios gracefully"""  
            # This tests the case where validation passes but execution encounters edge cases
            graph_data = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis', 'properties': {'description': 'First hypothesis'}},
                    {'id': 'H2', 'type': 'Hypothesis', 'properties': {'description': 'Second hypothesis'}}
                ],
                'edges': []
            }
            van_evera_results = {'hypothesis_rankings': {'H1': {'ranking_score': 0.8}, 'H2': {'ranking_score': 0.6}}}
            input_data = {'graph_data': graph_data, 'van_evera_results': van_evera_results}
            
            result = plugin.execute(input_data)
            assert result is not None
            assert result['primary_identification']['primary_hypothesis']['new_id'] == 'Q_H1'
            
        def test_missing_edge_properties_handling(self, plugin, multi_hypothesis_graph_data, van_evera_results):
            """Plugin handles missing edge properties gracefully"""
            # Create graph with missing edge properties
            modified_graph = multi_hypothesis_graph_data.copy()
            modified_graph['edges'] = [
                {
                    'source_id': 'E1',
                    'target_id': 'H1', 
                    'type': 'supports'
                    # Missing 'properties' key entirely
                }
            ]
            
            input_data = {'graph_data': modified_graph, 'van_evera_results': van_evera_results}
            
            # Should handle missing properties gracefully
            result = plugin.execute(input_data)
            assert result is not None
            assert 'primary_identification' in result

    class TestCheckpointData:
        """Test checkpoint data generation"""
        
        def test_checkpoint_data_format(self, plugin):
            """Plugin returns properly formatted checkpoint data"""
            checkpoint = plugin.get_checkpoint_data()
            
            assert 'plugin_id' in checkpoint
            assert checkpoint['plugin_id'] == 'primary_hypothesis_identifier'
            assert 'ranking_criteria' in checkpoint
            assert 'method' in checkpoint
            assert checkpoint['method'] == 'van_evera_evidence_based_ranking'