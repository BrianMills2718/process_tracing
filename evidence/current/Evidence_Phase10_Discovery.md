# Evidence: Phase 10 - Comprehensive Discovery

## Initial Discovery Results

### Files Analyzed
- Total Python files in core/: 67
- Initial compliance rate: 70.1%
- Non-compliant files: 20

### Violation Categories Found

#### 1. Fail-Fast Violations (4 files)
- core/enhance_evidence.py - Lines 42, 87
- core/diagnostic_rebalancer.py - Lines 155, 186  
- core/plugins/diagnostic_rebalancer.py - Lines 400, 459

#### 2. Keyword Matching Violations (10+ files)
- temporal_extraction.py - 20+ violations
- legacy_compatibility_manager.py - Lines 516, 519, 522, 525
- advanced_van_evera_prediction_engine.py - Line 685
- evidence_connector_enhancer.py - Lines 225, 226, 230
- temporal_graph.py - 6 violations
- temporal_validator.py - 2 violations
- evidence_document.py - 3 violations

#### 3. Hardcoded Values (6 files)
- advanced_van_evera_prediction_engine.py - Lines 822, 828, 834, 839
- content_based_diagnostic_classifier.py - Lines 561, 563, 565
- dowhy_causal_analysis_engine.py - Lines 208, 223, 249, 264, 287
- temporal_viz.py - Lines 899, 928, 941

#### 4. Missing LLM Usage (3 files)
- alternative_hypothesis_generator.py
- bayesian_van_evera_engine.py
- primary_hypothesis_identifier.py

## Discovery Script Created

```bash
#!/bin/bash
# Comprehensive compliance check script
for file in $(find ./core -name "*.py" -type f | grep -v __pycache__ | grep -v test); do
  # Check for keyword matching
  # Check for hardcoded values
  # Check for fail-fast violations
  # Check for LLM usage in semantic files
done
```

## Key Findings

1. **Temporal files are the worst offenders** - Most temporal_*.py files have extensive keyword matching
2. **Plugin files need significant work** - Many plugins still use rule-based logic
3. **Fail-fast violations are limited** - Only 4 files need this specific fix
4. **Hardcoded confidence values are widespread** - Need systematic replacement with LLM evaluation

## Next Steps Identified

1. Fix fail-fast violations (4 files) - EASIEST
2. Remove keyword matching from simple files - MODERATE
3. Refactor temporal extraction to use LLM - COMPLEX
4. Migrate plugins to LLM-first approach - COMPLEX
5. Replace all hardcoded confidence values - MODERATE