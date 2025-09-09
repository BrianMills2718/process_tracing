#!/usr/bin/env python3
"""Check real LLM compliance"""

from pathlib import Path

semantic_files = [
    'core/enhance_evidence.py',
    'core/enhance_hypotheses.py', 
    'core/enhance_mechanisms.py',
    'core/semantic_analysis_service.py',
    'core/confidence_calculator.py',
    'core/analyze.py',
    'core/plugins/van_evera_testing_engine.py',
    'core/plugins/diagnostic_rebalancer.py',
    'core/plugins/alternative_hypothesis_generator.py',
    'core/plugins/evidence_connector_enhancer.py',
    'core/plugins/content_based_diagnostic_classifier.py',
    'core/plugins/research_question_generator.py',
    'core/plugins/primary_hypothesis_identifier.py',
    'core/plugins/bayesian_van_evera_engine.py',
]

using_llm = []
not_using = []
missing_files = []

for file_path in semantic_files:
    path = Path(file_path)
    if path.exists():
        content = path.read_text()
        # Check for various LLM usage patterns
        llm_patterns = [
            'VanEveraLLMInterface', 'get_van_evera_llm', 'semantic_analysis_service', 'LLMGateway',
            'LLMRequiredError', 'query_llm', 'genai.', 'litellm.', 'refine_.*_with_llm'
        ]
        if any(x in content for x in llm_patterns):
            using_llm.append(file_path)
        else:
            not_using.append(file_path)
    else:
        missing_files.append(file_path)

existing_files = len(semantic_files) - len(missing_files)
print(f"Files using LLM interfaces: {len(using_llm)}/{existing_files} (of {len(semantic_files)} total)")
if existing_files > 0:
    print(f"Real compliance: {len(using_llm)/existing_files*100:.1f}%")
else:
    print("Real compliance: N/A (no files found)")

print("\nUsing LLM:")
for f in using_llm:
    print(f"  [OK] {f}")

print("\nNOT using LLM:")  
for f in not_using:
    print(f"  [X] {f}")

if missing_files:
    print("\nMISSING files:")
    for f in missing_files:
        print(f"  [?] {f}")