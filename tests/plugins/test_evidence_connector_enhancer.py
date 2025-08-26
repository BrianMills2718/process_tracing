"""Unit tests for EvidenceConnectorEnhancerPlugin - focused on evidence connection logic"""

import pytest
from unittest.mock import Mock, patch
from core.plugins.evidence_connector_enhancer import (
    EvidenceConnectorEnhancerPlugin
)
from core.plugins.base import PluginContext, PluginValidationError

class TestEvidenceConnectorEnhancerPlugin:
    """Unit tests for EvidenceConnectorEnhancerPlugin"""
    
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
        return EvidenceConnectorEnhancerPlugin("evidence_connector_enhancer", plugin_context)
        
    @pytest.fixture
    def graph_with_gaps(self):
        """Graph data with evidence and hypotheses that have connection gaps"""
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
                    'type': 'Alternative_Explanation',
                    'properties': {
                        'description': 'Colonial merchant networks and trade interests drove resistance to protect commercial profits'
                    }
                },
                {
                    'id': 'Q_H3', 
                    'type': 'Alternative_Explanation',
                    'properties': {
                        'description': 'Religious awakening provided ideological framework for independence from British authority'
                    }
                },
                {
                    'id': 'E1', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Colonial documents consistently invoke constitutional rights and representation arguments',
                        'source_text_quote': 'No taxation without representation in Parliament'
                    }
                },
                {
                    'id': 'E2', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Merchant class organized boycotts and coordinated trade resistance efforts',
                        'source_text_quote': 'Boston merchants unite to resist British trade restrictions for profit protection'
                    }
                },
                {
                    'id': 'E3', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Religious clergy provided moral leadership and biblical justification for resistance',
                        'source_text_quote': 'Ministers preached that God ordained independence from corrupt British rule'
                    }
                },
                {
                    'id': 'E4', 
                    'type': 'Evidence',
                    'properties': {
                        'description': 'Tea Party demonstrates clear opposition to taxation policies',
                        'source_text_quote': 'Boston Tea Party protest against tea duties'
                    }
                }
            ],
            'edges': [
                # Only H1 has evidence connection, H2 and H3 have gaps
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
        """Graph data without evidence nodes"""
        return {
            'nodes': [
                {'id': 'H1', 'type': 'Hypothesis', 'properties': {'description': 'Test hypothesis'}}
            ],
            'edges': []
        }
        
    class TestInputValidation:
        """Test validate_input() method"""
        
        def test_valid_input_accepted(self, plugin, graph_with_gaps):
            """Plugin accepts valid graph input with evidence and hypotheses"""
            valid_input = {'graph_data': graph_with_gaps}
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
            
        def test_no_evidence_nodes_rejected(self, plugin, empty_graph_data):
            """Plugin rejects graph with no evidence nodes"""
            invalid_input = {'graph_data': empty_graph_data}
            with pytest.raises(PluginValidationError) as exc_info:
                plugin.validate_input(invalid_input)
            assert "No evidence nodes found for connection enhancement" in str(exc_info.value)
                
    class TestSemanticBridging:
        """Test semantic bridging mappings and functionality"""
        
        def test_semantic_bridges_available(self, plugin):
            """Plugin has comprehensive semantic bridging mappings"""
            bridges = plugin.SEMANTIC_BRIDGES
            
            assert isinstance(bridges, dict)
            assert len(bridges) > 10  # Should have substantial mappings
            
            # Check for key categories
            expected_categories = [
                'merchant_networks', 'economic_grievances', 'constitutional_rhetoric',
                'religious_rhetoric', 'military_organization', 'administrative_failures'
            ]
            
            for category in expected_categories:
                assert category in bridges
                assert isinstance(bridges[category], list)
                assert len(bridges[category]) > 0
                
        def test_historical_context_keywords(self, plugin):
            """Plugin has relevant historical context keywords"""
            keywords = plugin.HISTORICAL_CONTEXT_KEYWORDS
            
            assert isinstance(keywords, set)
            assert len(keywords) > 5
            
            # Should include key American Revolution events
            expected_events = ['tea_party', 'stamp_act', 'boston_massacre']
            for event in expected_events:
                assert event in keywords or any(event.replace('_', ' ') in kw for kw in keywords)
                
        def test_semantic_relevance_calculation(self, plugin):
            """Test semantic relevance scoring algorithm"""
            hypothesis_desc = "merchant networks and trade interests drove resistance"
            
            # High relevance evidence
            high_relevance_evidence = "boston merchants coordinate trade boycotts for commercial profits"
            high_score = plugin._calculate_semantic_relevance(hypothesis_desc, high_relevance_evidence)
            
            # Low relevance evidence
            low_relevance_evidence = "military battles occurred in various locations"
            low_score = plugin._calculate_semantic_relevance(hypothesis_desc, low_relevance_evidence)
            
            # High relevance should score higher than low relevance
            assert high_score > low_score
            assert high_score > 0
            
        def test_direct_keyword_matching(self, plugin):
            """Test direct keyword matching in relevance calculation"""
            hypothesis_desc = "taxation without representation"
            evidence_text = "taxation policies caused representation demands"
            
            score = plugin._calculate_semantic_relevance(hypothesis_desc, evidence_text)
            
            # Should get points for direct word matches (taxation, representation)
            assert score >= 2  # At least 2 direct matches
            
        def test_bridging_keyword_matching(self, plugin):
            """Test semantic bridging keyword matching"""
            # Hypothesis mentioning economic grievances
            hypothesis_desc = "economic grievances motivated resistance"
            # Evidence using bridge keywords for economic grievances
            evidence_text = "tax burden and revenue policies upset colonists"
            
            score = plugin._calculate_semantic_relevance(hypothesis_desc, evidence_text)
            
            # Should get points from semantic bridging (tax, revenue ’ economic_grievances)
            assert score > 0
            
    class TestConnectionGapAnalysis:
        """Test _analyze_connection_gaps() method"""
        
        def test_connection_gap_identification(self, plugin, graph_with_gaps):
            """Correctly identifies hypotheses with insufficient evidence connections"""
            analysis = plugin._analyze_connection_gaps(graph_with_gaps)
            
            assert isinstance(analysis, dict)
            assert 'total_hypotheses' in analysis
            assert 'total_evidence' in analysis
            assert 'current_connections' in analysis
            assert 'gaps_found' in analysis
            assert 'insufficient_connections' in analysis
            assert 'connection_distribution' in analysis
            
            # Should identify Q_H2 and Q_H3 as having insufficient connections
            assert analysis['gaps_found'] >= 2  # At least Q_H2 and Q_H3
            
            # Should find specific hypotheses with gaps
            gap_ids = [gap['hypothesis_id'] for gap in analysis['insufficient_connections']]
            assert 'Q_H2' in gap_ids  # Alternative explanation with no connections
            assert 'Q_H3' in gap_ids  # Another alternative with no connections
            
        def test_connection_counting_accuracy(self, plugin, graph_with_gaps):
            """Test accuracy of connection counting"""
            analysis = plugin._analyze_connection_gaps(graph_with_gaps)
            
            # Should count 3 hypotheses total (Q_H1, Q_H2, Q_H3)
            assert analysis['total_hypotheses'] == 3
            
            # Should count 4 evidence nodes total
            assert analysis['total_evidence'] == 4
            
            # Should count 1 current connection (E1 -> Q_H1)
            assert analysis['current_connections'] == 1
            
        def test_connection_distribution_mapping(self, plugin, graph_with_gaps):
            """Test connection distribution per hypothesis"""
            analysis = plugin._analyze_connection_gaps(graph_with_gaps)
            
            distribution = analysis['connection_distribution']
            
            # Q_H1 should have 1 connection
            assert distribution['Q_H1'] == 1
            
            # Q_H2 and Q_H3 should have 0 connections
            assert distribution['Q_H2'] == 0
            assert distribution['Q_H3'] == 0
            
    class TestConnectionCreation:
        """Test connection creation and enhancement logic"""
        
        def test_evidence_finding_with_semantic_bridging(self, plugin, graph_with_gaps):
            """Test finding relevant evidence using semantic bridging"""
            evidence_nodes = [n for n in graph_with_gaps['nodes'] if n.get('type') == 'Evidence']
            hypothesis_desc = "merchant networks and trade interests drove resistance"
            
            relevant_evidence = plugin._find_evidence_with_semantic_bridging(evidence_nodes, hypothesis_desc)
            
            assert isinstance(relevant_evidence, list)
            # Should find E2 as relevant (mentions merchants, trade, boycotts)
            relevant_ids = [e['id'] for e in relevant_evidence]
            assert 'E2' in relevant_ids
            
        def test_evidence_relevance_ranking(self, plugin, graph_with_gaps):
            """Test that evidence is ranked by relevance correctly"""
            evidence_nodes = [n for n in graph_with_gaps['nodes'] if n.get('type') == 'Evidence']
            hypothesis_desc = "religious awakening provided ideological framework"
            
            relevant_evidence = plugin._find_evidence_with_semantic_bridging(evidence_nodes, hypothesis_desc)
            
            if len(relevant_evidence) > 1:
                # Should be sorted by relevance score (highest first)
                # E3 (religious clergy) should rank higher than others for religious hypothesis
                top_evidence = relevant_evidence[0]
                assert top_evidence['id'] == 'E3'  # Religious evidence should rank highest
                
        def test_diagnostic_type_determination(self, plugin):
            """Test diagnostic type assignment for new connections"""
            # Evidence suggesting decisive proof
            decisive_evidence = {
                'id': 'E_TEST',
                'properties': {'description': 'clear and decisive evidence demonstrates'}
            }
            hypothesis_desc = "test hypothesis description"
            
            diagnostic_type = plugin._determine_diagnostic_type(decisive_evidence, hypothesis_desc)
            assert diagnostic_type == 'doubly_decisive'
            
            # Evidence for necessary condition
            necessary_evidence = {
                'id': 'E_TEST2',
                'properties': {'description': 'standard evidence description'}
            }
            necessary_hypothesis = "this is necessary and required for the outcome"
            
            diagnostic_type = plugin._determine_diagnostic_type(necessary_evidence, necessary_hypothesis)
            assert diagnostic_type == 'hoop'
            
        def test_probative_value_calculation(self, plugin):
            """Test probative value assignment based on diagnostic type"""
            # Different diagnostic types should have appropriate probative values
            doubly_decisive_value = plugin._calculate_probative_value('doubly_decisive')
            smoking_gun_value = plugin._calculate_probative_value('smoking_gun')
            hoop_value = plugin._calculate_probative_value('hoop')
            straw_value = plugin._calculate_probative_value('straw_in_wind')
            
            # Doubly decisive should have highest value
            assert doubly_decisive_value > smoking_gun_value
            assert doubly_decisive_value > hoop_value
            assert doubly_decisive_value > straw_value
            
            # All values should be reasonable (0.5-1.0 range)
            for value in [doubly_decisive_value, smoking_gun_value, hoop_value, straw_value]:
                assert 0.5 <= value <= 1.0
                
        def test_evidence_connection_creation(self, plugin):
            """Test creation of new evidence-hypothesis connections"""
            evidence_node = {
                'id': 'E_TEST',
                'properties': {'description': 'test evidence for connection'}
            }
            hypothesis_id = 'H_TEST'
            hypothesis_desc = 'test hypothesis description'
            
            connection = plugin._create_evidence_connection(evidence_node, hypothesis_id, hypothesis_desc)
            
            assert isinstance(connection, dict)
            assert connection['source_id'] == 'E_TEST'
            assert connection['target_id'] == 'H_TEST'
            assert connection['type'] == 'supports'
            
            props = connection['properties']
            assert 'diagnostic_type' in props
            assert 'probative_value' in props
            assert 'connection_method' in props
            assert props['connection_method'] == 'semantic_bridging'
            assert props['enhancement_applied'] == True
            assert props['automatically_generated'] == True
            
    class TestExecutionLogic:
        """Test execute() method and full enhancement workflow"""
        
        def test_execute_with_valid_input(self, plugin, graph_with_gaps):
            """Plugin executes connection enhancement successfully"""
            input_data = {'graph_data': graph_with_gaps}
            result = plugin.execute(input_data)
            
            # Verify result structure
            assert result is not None
            assert isinstance(result, dict)
            assert 'updated_graph_data' in result
            assert 'connection_analysis' in result
            assert 'enhancement_results' in result
            assert 'improvement_metrics' in result
            assert 'semantic_bridging_applied' in result
            
            assert result['semantic_bridging_applied'] == True
            
        def test_graph_data_updating(self, plugin, graph_with_gaps):
            """Plugin correctly updates graph data with new connections"""
            original_edge_count = len(graph_with_gaps['edges'])
            
            input_data = {'graph_data': graph_with_gaps}
            result = plugin.execute(input_data)
            
            updated_graph = result['updated_graph_data']
            new_edge_count = len(updated_graph['edges'])
            
            # Should add new edges for hypotheses with insufficient connections
            assert new_edge_count > original_edge_count
            
            # Check that new edges have enhancement properties
            enhancement_edges = []
            for edge in updated_graph['edges']:
                props = edge.get('properties', {})
                if props.get('enhancement_applied') == True:
                    enhancement_edges.append(edge)
                    
            assert len(enhancement_edges) > 0
            
            # Check edge properties
            for edge in enhancement_edges:
                props = edge['properties']
                assert 'diagnostic_type' in props
                assert 'probative_value' in props
                assert 'connection_method' in props
                assert props['connection_method'] == 'semantic_bridging'
                
        def test_connection_gap_filling(self, plugin, graph_with_gaps):
            """Plugin fills connection gaps for hypotheses with insufficient evidence"""
            input_data = {'graph_data': graph_with_gaps}
            result = plugin.execute(input_data)
            
            # Should reduce the number of hypotheses with insufficient connections
            original_gaps = result['connection_analysis']['gaps_found']
            connections_added = result['enhancement_results']['connections_added']
            
            assert original_gaps >= 2  # Q_H2 and Q_H3 initially have gaps
            assert connections_added > 0  # Should add some connections
            
        def test_improvement_metrics_calculation(self, plugin, graph_with_gaps):
            """Plugin correctly calculates improvement metrics"""
            input_data = {'graph_data': graph_with_gaps}
            result = plugin.execute(input_data)
            
            metrics = result['improvement_metrics']
            
            assert 'original_connections' in metrics
            assert 'new_connections' in metrics
            assert 'total_connections' in metrics
            assert 'coverage_improvement' in metrics
            assert 'semantic_bridging_effectiveness' in metrics
            
            # Should show improvement
            assert metrics['total_connections'] > metrics['original_connections']
            if metrics['new_connections'] > 0:
                assert metrics['coverage_improvement'] > 0
                assert metrics['semantic_bridging_effectiveness'] == 'high'
                
    class TestErrorHandling:
        """Test error conditions and edge cases"""
        
        def test_graph_with_sufficient_connections(self, plugin):
            """Test handling of graph where all hypotheses have sufficient connections"""
            well_connected_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Hypothesis', 'properties': {'description': 'Test hypothesis'}},
                    {'id': 'E1', 'type': 'Evidence', 'properties': {'description': 'Evidence 1'}},
                    {'id': 'E2', 'type': 'Evidence', 'properties': {'description': 'Evidence 2'}},
                    {'id': 'E3', 'type': 'Evidence', 'properties': {'description': 'Evidence 3'}}
                ],
                'edges': [
                    {'source_id': 'E1', 'target_id': 'H1', 'type': 'supports'},
                    {'source_id': 'E2', 'target_id': 'H1', 'type': 'supports'},
                    {'source_id': 'E3', 'target_id': 'H1', 'type': 'supports'}
                ]
            }
            
            input_data = {'graph_data': well_connected_graph}
            result = plugin.execute(input_data)
            
            # Should handle gracefully with no gaps to fill
            assert result is not None
            gaps_found = result['connection_analysis']['gaps_found']
            connections_added = result['enhancement_results']['connections_added']
            
            assert gaps_found == 0  # No gaps
            assert connections_added == 0  # No connections needed
            
        def test_evidence_with_no_relevant_matches(self, plugin):
            """Test handling when evidence has no semantic relevance to hypotheses"""
            irrelevant_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Alternative_Explanation', 
                     'properties': {'description': 'economic merchant trade hypothesis'}},
                    {'id': 'E1', 'type': 'Evidence', 
                     'properties': {'description': 'geological rock formation analysis'}}
                ],
                'edges': []
            }
            
            input_data = {'graph_data': irrelevant_graph}
            result = plugin.execute(input_data)
            
            # Should handle gracefully even with no semantic matches
            assert result is not None
            # May or may not add connections depending on algorithm flexibility
            connections_added = result['enhancement_results']['connections_added']
            assert connections_added >= 0  # Should be non-negative
            
        def test_malformed_node_properties(self, plugin):
            """Test handling of nodes with missing or malformed properties"""
            malformed_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Alternative_Explanation'},  # Missing properties
                    {'id': 'E1', 'type': 'Evidence'}  # Missing properties
                ],
                'edges': []
            }
            
            input_data = {'graph_data': malformed_graph}
            result = plugin.execute(input_data)
            
            # Should handle gracefully without crashing
            assert result is not None
            assert 'enhancement_results' in result
            
        def test_checkpoint_data(self, plugin):
            """Test plugin checkpoint data includes semantic bridging info"""
            checkpoint = plugin.get_checkpoint_data()
            
            assert checkpoint['plugin_id'] == 'evidence_connector_enhancer'
            assert 'semantic_bridges_available' in checkpoint
            assert 'historical_context_keywords' in checkpoint
            assert 'enhancement_method' in checkpoint
            
            assert checkpoint['enhancement_method'] == 'semantic_bridging'
            assert checkpoint['semantic_bridges_available'] > 10  # Should have substantial mappings
            
        def test_integration_function(self):
            """Test standalone integration function"""
            from core.plugins.evidence_connector_enhancer import enhance_evidence_connections
            
            # Create test graph data
            test_graph = {
                'nodes': [
                    {'id': 'H1', 'type': 'Alternative_Explanation', 
                     'properties': {'description': 'merchant trade hypothesis'}},
                    {'id': 'E1', 'type': 'Evidence', 
                     'properties': {'description': 'merchants organized boycotts for commercial interests'}}
                ],
                'edges': []
            }
            
            result_graph = enhance_evidence_connections(test_graph)
            
            # Should return updated graph data
            assert result_graph is not None
            assert isinstance(result_graph, dict)
            assert 'nodes' in result_graph
            assert 'edges' in result_graph
            
            # May have added connections depending on semantic matching
            assert len(result_graph['edges']) >= len(test_graph['edges'])