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

for file_path in semantic_files:
    path = Path(file_path)
    if path.exists():
        content = path.read_text()
        if any(x in content for x in ['VanEveraLLMInterface', 'get_van_evera_llm', 'semantic_analysis_service', 'LLMGateway']):
            using_llm.append(file_path)
        else:
            not_using.append(file_path)

print(f"Files using LLM interfaces: {len(using_llm)}/{len(semantic_files)}")
print(f"Real compliance: {len(using_llm)/len(semantic_files)*100:.1f}%")
print("\nUsing LLM:")
for f in using_llm:
    print(f"  [OK] {f}")
print("\nNOT using LLM:")  
for f in not_using:
    print(f"  [X] {f}")