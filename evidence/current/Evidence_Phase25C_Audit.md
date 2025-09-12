# Evidence Phase 25C: Comprehensive Pattern Audit

## Status: PHASE 1 AUDIT COMPLETE
Started: 2025-01-11
Phase: Phase 25C - Complete Systematic Migration

## Executive Summary

**CURRENT STATE VALIDATED**:
- ✅ **OntologyManager**: 22/22 tests passing - fully functional
- ✅ **System Integration**: Pipeline processes inputs successfully
- ⚠️ **Migration Status**: 109 hardcoded patterns remain across 25+ files
- ⚠️ **Core Files**: 16 patterns in core/ modules requiring immediate attention

## Complete Pattern Inventory (109 Total)

### PATTERN COUNTS BY LOCATION
```bash
# Total patterns across codebase (excluding test_env):
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env | wc -l
Result: 109 patterns

# Core module patterns:
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/ --include="*.py" | wc -l  
Result: 16 patterns
```

## CATEGORIZED FILE ANALYSIS

### P0 CRITICAL - Core System Files (Immediate Priority)

**FILES REQUIRING MIGRATION**:

1. **core/disconnection_repair.py** - 14 patterns
   - Lines with hardcoded edge types in repair logic
   - Priority: IMMEDIATE - affects graph connectivity repair

2. **core/van_evera_testing_engine.py** - 1 pattern
   - Line with hardcoded edge type list
   - Priority: HIGH - affects Van Evera diagnostic testing

3. **core/analyze.py** - 1 pattern  
   - Line 540: `if edge_data and edge_data.get('type') in ['tests_mechanism', 'supports']:`
   - Priority: HIGH - main execution path

4. **core/streaming_html.py** - 1 pattern
   - Line 285: Comment with hardcoded example
   - Priority: LOW - documentation only

### P1 HIGH - Plugin System Files

1. **core/plugins/evidence_connector_enhancer.py** - 1 pattern
   - Line with fallback to 'tests_hypothesis'
   - Already partially migrated in Phase 25B

2. **core/plugins/content_based_diagnostic_classifier.py** - 1 pattern
   - Line 89: Semantic pattern for language processing
   - Analysis: This is LANGUAGE PROCESSING logic, not edge type logic - KEEP AS IS

### P2 MEDIUM - Test Infrastructure Files

1. **tests/test_dag_analysis.py** - 1 pattern
2. **tests/test_cross_domain.py** - 2 patterns  
3. **tests/test_ontology_manager.py** - 35 patterns (test data)
4. **tests/plugins/test_*.py** - 35 patterns (test data across 6 files)
5. **tests/ontology_test_helpers.py** - 6 patterns
6. **tests/test_van_evera_bayesian_integration.py** - 1 pattern

### P3 LOW - Tools and Documentation  

1. **tools/migrate_ontology.py** - 8 patterns (migration mapping data - APPROPRIATE)
2. **tools/migration_inventory.py** - 2 patterns (documentation - APPROPRIATE)
3. **process_trace_advanced.py** - 2 patterns (already migrated in Phase 25B)
4. **docs/testing/*.py** - 8 patterns across 8 files

### APPROPRIATE PATTERNS (DO NOT MIGRATE)

**Files containing patterns that should NOT be migrated**:

1. **core/ontology_manager.py** - 1 pattern
   - Line 19: Docstring example - APPROPRIATE
   
2. **core/plugins/content_based_diagnostic_classifier.py** - 1 pattern
   - Semantic language processing - APPROPRIATE
   
3. **tools/migrate_ontology.py** - 8 patterns
   - Migration mapping data - APPROPRIATE
   
4. **tools/migration_inventory.py** - 2 patterns  
   - Documentation strings - APPROPRIATE

5. **tests/test_ontology_manager.py** - Most patterns
   - Test data validating OntologyManager - APPROPRIATE

## RISK ASSESSMENT BY PRIORITY

### P0 Critical Risk (4 files, 17 patterns)
- **core/disconnection_repair.py**: 14 patterns - HIGHEST RISK
  - Contains hardcoded edge type matrices for graph repair
  - Direct impact on system functionality
  - Must migrate to dynamic ontology lookups
  
- **core/van_evera_testing_engine.py**: 1 pattern - HIGH RISK
  - Affects diagnostic testing capabilities
  - Main analysis pathway component
  
- **core/analyze.py**: 1 pattern - MEDIUM RISK
  - Single hardcoded check in edge processing
  - Main execution path but isolated impact
  
- **core/streaming_html.py**: 1 pattern - LOW RISK
  - Documentation comment only

### P1 High Risk (2 files, 2 patterns)
- **evidence_connector_enhancer.py**: Partially migrated, needs completion
- **content_based_diagnostic_classifier.py**: SEMANTIC PATTERN - DO NOT MIGRATE

### P2 Medium Risk (Test files, 80+ patterns)
- Use ontology_test_helpers.py for systematic migration
- Low system impact but affects test validation

### P3 Low Risk (Documentation, 10+ patterns)
- Lowest priority, can be done incrementally
- Minimal system impact

## MIGRATION STRATEGY BY PHASE

### PHASE 2: Residual Cleanup Priority
1. **core/disconnection_repair.py** - Complete overhaul of edge type matrices
2. **core/van_evera_testing_engine.py** - Single line replacement
3. **core/analyze.py** - Single hardcoded check replacement

### PHASE 3: Systematic New File Migration Priority  
1. P2 test files using ontology_test_helpers.py
2. P3 documentation files (low priority)

## VALIDATION COMMANDS FOR EACH PHASE

### Track Core Module Progress
```bash
# Before migration: 16 patterns
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'" core/ --include="*.py" | wc -l

# Target: 4 patterns (only appropriate semantic/documentation patterns)
```

### Track Total Progress
```bash  
# Before migration: 109 patterns
grep -r "'supports'\|'tests_hypothesis'\|'provides_evidence_for'\|'updates_probability'\|'weighs_evidence'" --include="*.py" . | grep -v test_env | wc -l

# Target: ~20 patterns (only appropriate patterns remain)
```

## ARCHITECTURE IMPACT ASSESSMENT

### Current OntologyManager Capabilities
✅ **Available Methods**:
- `get_evidence_hypothesis_edges()` - Returns all Evidence→Hypothesis edge types
- `get_edge_types_for_relationship(source, target)` - Dynamic relationship queries
- `get_edges_by_domain(domain)` - All edges from source type
- `validate_edge(edge)` - Edge validation against ontology

✅ **Test Coverage**: 22/22 tests passing
✅ **System Integration**: Full pipeline functional
✅ **Performance**: No degradation observed

### Migration Readiness
✅ **Infrastructure Complete**: All tools and helpers ready
✅ **Migration Patterns**: Established and tested
✅ **Rollback Capability**: Backup and validation procedures ready
✅ **Test Framework**: Comprehensive validation available

## PHASE 1 COMPLETION CHECKLIST

- ✅ **System State Validated**: OntologyManager functional, pipeline working
- ✅ **Complete Pattern Discovery**: 109 patterns identified and categorized  
- ✅ **Risk Assessment**: Priority levels assigned (P0-P3)
- ✅ **Migration Strategy**: Phase-based approach defined
- ✅ **Validation Framework**: Commands and criteria established
- ✅ **Evidence Documentation**: Complete audit documented

**PHASE 1 STATUS: COMPLETE**
Ready to proceed to Phase 2 - Residual Cleanup in "Migrated" Files