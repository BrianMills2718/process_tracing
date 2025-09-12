=== SYSTEMATIC ANALYSIS: ./core/disconnection_repair.py ===
28-        # Semantic keywords for connection inference
29-        self.semantic_patterns = {
30-            'enables': ['enable', 'allow', 'facilitate', 'make possible', 'permit', 'distance', 'economic development'],
31-            'constrains': ['constrain', 'limit', 'prevent', 'restrict', 'naval supremacy', 'military'],
32-            'initiates': ['initiate', 'start', 'launch', 'begin', 'personally initiated'],
33:            'supports': ['support', 'evidence for', 'confirm', 'validate'],
34-            'refutes': ['refute', 'contradict', 'challenge', 'disprove'],
35-            'causes': ['cause', 'lead to', 'result in', 'bring about']
36-        }
37-    
38-    def infer_missing_connections(self, graph_data: Dict) -> List[Dict]:
--
228-                        )
229-                        
230-                        if assessment.probative_value > 0.6:
231-                            # Use dynamic edge selection for supportive relationships
232-                            supportive_edges = [e for e in ontology_manager.get_evidence_hypothesis_edges() 
233:                                              if 'support' in e or e == 'tests_hypothesis']
234:                            edge_type = supportive_edges[0] if supportive_edges else 'tests_hypothesis'
235-                            edges.append(self._create_edge(node_id, target_id, edge_type,
236-                                                         assessment.reasoning))
237-                        else:
238-                            # Use general evidence-hypothesis edge type
239-                            evidence_edges = ontology_manager.get_evidence_hypothesis_edges()
240:                            edge_type = evidence_edges[0] if evidence_edges else 'tests_hypothesis'
241-                            edges.append(self._create_edge(node_id, target_id, edge_type,
242-                                                         "Evidence relates to hypothesis"))
243-            
244-            # Connect evidence to mechanisms
245-            elif target_type == 'Causal_Mechanism':
--
400-            return edge_types[0]
401-        
402-        # Fallback matrix for combinations not yet in ontology
403-        fallback_matrix = {
404-            'Evidence': {
405:                'Hypothesis': 'tests_hypothesis',  # Use modern edge type
406-                'Causal_Mechanism': 'tests_mechanism',
407-                'Event': 'confirms_occurrence',
408-                'Alternative_Explanation': 'tests_alternative'
409-            },
410-            'Actor': {
411-                'Event': 'initiates',
412-                'Causal_Mechanism': 'tests_mechanism',
413:                'Hypothesis': 'tests_hypothesis'
414-            },
415-            'Condition': {
416-                'Event': 'enables',
417-                'Causal_Mechanism': 'enables',
418-                'Actor': 'constrains',
419-                'Hypothesis': 'enables'
420-            },
421-            'Event': {
422-                'Causal_Mechanism': 'part_of_mechanism',
423:                'Hypothesis': 'tests_hypothesis',
424-                'Event': 'causes'
425-            },
426-            'Data_Source': {
427-                'Evidence': 'weighs_evidence',
428-                'Hypothesis': 'weighs_evidence',
--
435-        if fallback_edge:
436-            return fallback_edge
437-        
438-        # Final fallback - use most common edge type from ontology
439-        all_edges = ontology_manager.get_all_edge_types()
440:        return all_edges[0] if all_edges else 'tests_hypothesis'
441-    
442-    def _infer_component_connections(self, component: Dict, graph_data: Dict) -> List[Dict]:
443-        """Infer connections to link small components to main graph."""
444-        edges = []
445-        component_nodes = [node['id'] for node in component['nodes']]

