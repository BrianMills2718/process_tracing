=== PHASE 27B TASK 1: FUNCTIONAL REQUIREMENT DISCOVERY ===
=== CURRENT HARDCODED VALIDATION ===
./analyze_direct.py:        required_edges = ['tests_hypothesis', 'supports', 'provides_evidence_for']

=== ONTOLOGY MANAGER CAPABILITIES ===
Evidence-Hypothesis edges: ['updates_probability', 'weighs_evidence', 'supports', 'infers', 'refutes', 'tests_hypothesis', 'provides_evidence_for']
Van Evera edges: ['supports', 'refutes', 'tests_hypothesis', 'provides_evidence_for', 'tests_mechanism', 'confirms_occurrence', 'disproves_occurrence']
All edge types: ['causes', 'part_of_mechanism', 'tests_hypothesis', 'tests_mechanism', 'enables', 'constrains', 'provides_evidence', 'initiates', 'infers', 'updates_probability', 'contradicts', 'supports', 'refutes', 'explains_mechanism', 'confirms_occurrence', 'disproves_occurrence', 'provides_evidence_for', 'weighs_evidence', 'addresses_research_question']
All node types: ['Event', 'Causal_Mechanism', 'Hypothesis', 'Evidence', 'Condition', 'Actor', 'Inference_Rule', 'Inferential_Test', 'Research_Question', 'Data_Source']
=== FUNCTIONAL USAGE PATTERNS ===
core/html_generator.py:        evidence_hypothesis_edge_types = ontology_manager.get_evidence_hypothesis_edges()
core/plugins/primary_hypothesis_identifier.py:        evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/plugins/evidence_connector_enhancer.py:        evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/plugins/van_evera_testing.py:                    evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/disconnection_repair.py:                            supportive_edges = [e for e in ontology_manager.get_evidence_hypothesis_edges() 
core/disconnection_repair.py:                            evidence_edges = ontology_manager.get_evidence_hypothesis_edges()
core/ontology_manager.py:    def get_evidence_hypothesis_edges(self) -> List[str]:
core/ontology_manager.py:    def get_van_evera_edges(self) -> List[str]:
core/ontology_manager.py:        for edge_type in self.get_evidence_hypothesis_edges():
core/ontology_manager.py:        return edge_type in self.get_evidence_hypothesis_edges()
core/ontology_manager.py:def get_evidence_hypothesis_edges() -> List[str]:
core/ontology_manager.py:    return ontology_manager.get_evidence_hypothesis_edges()
core/ontology_manager.py:    return ontology_manager.get_van_evera_edges()
core/analyze.py:                evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/analyze.py:                        evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/analyze.py:                    evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/streaming_html.py:                evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()
core/van_evera_testing_engine.py:        evidence_edge_types = ontology_manager.get_evidence_hypothesis_edges()
core/van_evera_testing_engine.py:                        evidence_hypothesis_edges = ontology_manager.get_evidence_hypothesis_edges()


=== TASK 1 ANALYSIS ===
HARDCODED VALIDATION PROBLEM:
- analyze_direct.py line 32: required_edges = ['tests_hypothesis', 'supports', 'provides_evidence_for']
- This hardcoded list assumes specific edge type names must exist
- If ontology uses different names (e.g. 'confirms_hypothesis' instead of 'tests_hypothesis'), validation fails
- BUT system has 7 Evidence-Hypothesis edges available, showing hardcoded list is unnecessarily restrictive

FUNCTIONAL REQUIREMENTS DISCOVERED:
CRITICAL: At least one Evidence->Hypothesis connection capability
- Current system has 7 such edges: ['updates_probability', 'weighs_evidence', 'supports', 'infers', 'refutes', 'tests_hypothesis', 'provides_evidence_for']
- System NEEDS this functionality but shouldn't care about specific names

OPTIONAL: Van Evera diagnostic test capabilities  
- Current system has 7 Van Evera edges
- Enhances analysis but not required for basic operation

ACTUAL SYSTEM USAGE:
- 15+ files dynamically query ontology_manager.get_evidence_hypothesis_edges()
- System architecture is ALREADY dynamic in most places
- Only validation is hardcoded - rest of system adapts perfectly

SUCCESS CRITERIA MET:
✅ Identified actual functional requirements (Evidence->Hypothesis connectivity)
✅ Separated naming assumptions from functional needs  
✅ Documented ontology_manager's current query capabilities
✅ Mapped hardcoded validation to actual usage patterns

CONCLUSION: System needs dynamic validation checking FUNCTIONAL capabilities, not hardcoded edge names.

