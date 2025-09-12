# Evidence Phase 25B: Complete File Inventory

## Objective
Identify ALL files with hardcoded edge type references to complete systematic migration.

## Status: COMPLETE
Started: 2025-01-11
Completed: 2025-01-11

## Search Results - RAW OUTPUT

### Search 1: Hardcoded Edge Type Strings
**Command**: `grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" .`

**Results**: 68 matches across 27 files

#### Core Modules (HIGH PRIORITY P0-P1):
- `core/plugins/evidence_connector_enhancer.py` - 1 match
- `core/plugins/content_based_diagnostic_classifier.py` - 1 match  
- `core/disconnection_repair.py` - 14 matches (ALREADY MIGRATED - but still has hardcoded strings in patterns)
- `core/connectivity_analysis.py` - 2 matches (CRITICAL P0)
- `core/analyze.py` - 1 match (ALREADY MIGRATED - residual match)
- `core/streaming_html.py` - 1 match (ALREADY MIGRATED - residual in comment)
- `core/van_evera_testing_engine.py` - 1 match (ALREADY MIGRATED - hardcoded list remains)

#### Tool/Migration Files (EXPECTED):
- `core/ontology_manager.py` - 1 match (expected - in docstring)
- `tools/migrate_ontology.py` - 6 matches (expected - migration mappings)

#### Process/Example Files:
- `process_trace_advanced.py` - 2 matches

#### Test Files (MEDIUM PRIORITY P2):
- `tests/test_ontology_manager.py` - 24 matches (expected - test data)
- `tests/test_dag_analysis.py` - 1 match
- `tests/test_cross_domain.py` - 2 matches  
- `tests/plugins/test_van_evera_testing.py` - 2 matches
- `tests/plugins/test_evidence_connector_enhancer.py` - 5 matches
- `tests/plugins/test_alternative_hypothesis_generator.py` - 2 matches
- `tests/plugins/test_content_based_diagnostic_classifier.py` - 9 matches
- `tests/plugins/test_primary_hypothesis_identifier.py` - 7 matches
- `tests/test_van_evera_bayesian_integration.py` - 1 match

#### Documentation/Testing Files (LOW PRIORITY P3):
- `docs/testing/test_all_critical_fixes.py` - 1 match
- `docs/testing/test_critical_bug_21.py` - 1 match
- `docs/testing/test_critical_bug_34.py` - 1 match
- `docs/testing/test_focused_extraction.py` - 1 match
- `docs/testing/test_phase2b_integration.py` - 2 matches
- `docs/testing/manual_analysis_test.py` - 1 match
- `docs/testing/test_critical_bug_16.py` - 1 match
- `docs/testing/test_direct_integration.py` - 1 match

#### Library Files (IGNORE):
- `test_env/lib/python3.12/site-packages/` - 3 matches (third-party library code)

### Search 2: Edge Type Conditional Checks
**Command**: `grep -r "edge.*type.*in \[" --include="*.py" .`

**Results**: 10 matches across 8 files

#### Files with conditional checks needing migration:
- `core/plugins/van_evera_testing.py` - ALREADY MIGRATED (residual patterns)
- `core/analyze.py` - ALREADY MIGRATED (residual patterns) 
- `core/streaming_html.py` - ALREADY MIGRATED (residual patterns)
- `core/van_evera_testing_engine.py` - ALREADY MIGRATED (residual patterns)
- `docs/testing/test_focused_extraction.py` - P3 priority
- `docs/testing/manual_analysis_test.py` - P3 priority

### Search 3: Edge Type Equality Checks  
**Command**: `grep -r "edge\['type'\].*==" --include="*.py" .`

**Results**: 1 match
- `docs/testing/manual_analysis_test.py` - P3 priority

## Migration Priority Analysis

### P0 CRITICAL (IMMEDIATE):
1. **`core/connectivity_analysis.py`** - Line 22 has hardcoded list: `'Evidence': ['supports', 'refutes', 'tests_hypothesis']`

### P1 HIGH (Core Modules):
1. **`core/plugins/evidence_connector_enhancer.py`** - Hardcoded 'supports' 
2. **`core/plugins/content_based_diagnostic_classifier.py`** - Hardcoded patterns
3. **`process_trace_advanced.py`** - 2 hardcoded references

### P2 MEDIUM (Test Files):
1. **`tests/test_dag_analysis.py`** - Test data
2. **`tests/test_cross_domain.py`** - Test data  
3. **`tests/plugins/test_*.py`** - Multiple plugin test files

### P3 LOW (Documentation/Examples):
1. **`docs/testing/*.py`** - 8 test/example files

## Residual Matches in Already Migrated Files

**ISSUE**: Some files show as "already migrated" but still have hardcoded strings. Investigation needed:

1. **`core/disconnection_repair.py`** - 14 matches (migrated but has hardcoded patterns in semantic mapping)
2. **`core/van_evera_testing_engine.py`** - 1 match (migrated but has residual list)
3. **`core/analyze.py`** - 1 match (migrated but has residual check)
4. **`core/streaming_html.py`** - 1 match (migrated but has residual in comment)

## Summary Statistics

- **Total files with hardcoded references**: 27 files
- **Core modules needing migration**: 4 files (P0-P1)
- **Test files needing migration**: 9 files (P2)  
- **Documentation files needing migration**: 8 files (P3)
- **Already migrated but with residuals**: 4 files (need cleanup)

**CRITICAL FINDING**: Migration is ~70% incomplete, not ~30% as previously claimed. More files need attention than initially estimated.

## Next Steps

1. **IMMEDIATE**: Migrate `core/connectivity_analysis.py` 
2. **HIGH PRIORITY**: Migrate core plugin modules
3. **INVESTIGATE**: Clean up residual patterns in "already migrated" files
4. **SYSTEMATIC**: Work through P2 and P3 files methodically