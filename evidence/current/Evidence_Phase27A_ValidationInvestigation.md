=== PHASE 27A TASK 1: VALIDATION LOGIC INVESTIGATION ===
=== LOCATING VALIDATION ERROR SOURCE ===
./analyze_direct.py:            raise ValueError(f"❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: {missing}")

./analyze_direct.py:            raise ValueError(f"❌ ONTOLOGY VALIDATION FAILED: Missing critical edge types: {missing}")

=== CRITICAL EDGE TYPE REFERENCES ===
--- References to  ---
core/enhance_evidence.py:# Avoid circular import - query_llm will be passed as parameter
core/enhance_evidence.py:from .structured_models import EvidenceAssessment
core/enhance_evidence.py:from .llm_required import LLMRequiredError
core/enhance_evidence.py:
core/enhance_evidence.py:def refine_evidence_assessment_with_llm(evidence_description, text_content, context_info=None, query_llm_func=None):

--- References to  ---
core/enhance_evidence.py:# Avoid circular import - query_llm will be passed as parameter
core/enhance_evidence.py:from .structured_models import EvidenceAssessment
core/enhance_evidence.py:from .llm_required import LLMRequiredError
core/enhance_evidence.py:
core/enhance_evidence.py:def refine_evidence_assessment_with_llm(evidence_description, text_content, context_info=None, query_llm_func=None):

--- References to  ---
core/enhance_evidence.py:# Avoid circular import - query_llm will be passed as parameter
core/enhance_evidence.py:from .structured_models import EvidenceAssessment
core/enhance_evidence.py:from .llm_required import LLMRequiredError
core/enhance_evidence.py:
core/enhance_evidence.py:def refine_evidence_assessment_with_llm(evidence_description, text_content, context_info=None, query_llm_func=None):

=== VALIDATION ARCHITECTURE ANALYSIS ===
Reading key files to understand validation flow...

VALIDATION LOGIC LOCATION: analyze_direct.py:20-47
FUNCTION: validate_system_ontology()
HARDCODED CRITICAL EDGES: ['tests_hypothesis', 'supports', 'provides_evidence_for']
VALIDATION TRIGGER: Pipeline entry point (before any processing)
FAIL-FAST BEHAVIOR: sys.exit(1) on validation failure

