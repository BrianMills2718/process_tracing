# Process Tracing Methodology Versioning Strategy

## Overview
This document defines the versioning strategy for the Van Evera process tracing methodology implementations, ensuring compatibility, reproducibility, and systematic improvement.

## Version Numbering Scheme

### Semantic Versioning: MAJOR.MINOR.PATCH

- **MAJOR**: Fundamental methodology changes (e.g., 1.x → 2.x represents shift from sequential to parallel processing)
- **MINOR**: New capabilities or significant improvements (e.g., 2.0 → 2.1 adds cross-lingual analysis)
- **PATCH**: Bug fixes and performance improvements (e.g., 2.1.0 → 2.1.1)

## Version History

### Version 1.x - Traditional Van Evera (Human-Centric)
- **1.0.0**: Basic Van Evera implementation
- **1.1.0**: Added Bayesian updating
- **1.2.0**: Added comparative case capability
- **Characteristics**: Sequential processing, limited evidence capacity, human-interpretable steps

### Version 2.x - LLM-Enhanced Van Evera
- **2.0.0**: Parallel processing, exhaustive hypothesis generation, complete counterfactual space
- **2.1.0**: (Planned) Cross-lingual evidence integration
- **2.2.0**: (Planned) Multi-modal evidence (images, maps, diagrams)
- **Characteristics**: Simultaneous analysis, unlimited evidence processing, computational methods

### Version 3.x - (Future) Quantum-Inspired Causal Analysis
- Superposition of causal states
- Probabilistic causal paths
- Observer effect modeling

## Compatibility Matrix

| Output Version | Can Read | Can Upgrade To | Notes |
|----------------|----------|----------------|-------|
| 1.0.x | 1.0.x only | 1.1.x, 2.0.x | Manual review required for 2.x upgrade |
| 1.1.x | 1.0.x, 1.1.x | 1.2.x, 2.0.x | Bayesian fields preserved in upgrade |
| 1.2.x | 1.0.x - 1.2.x | 2.0.x | Comparative fields mapped to cross_case |
| 2.0.x | 1.x.x, 2.0.x | 2.1.x+ | Full backward compatibility |
| 2.1.x | All previous | 2.2.x+ | Includes version converter |

## Version Metadata in Outputs

```json
{
  "methodology_version": "van_evera_2.0_fully_automated",
  "methodology_capabilities": [
    "parallel_processing",
    "exhaustive_hypothesis_generation",
    "complete_counterfactual_testing",
    "simultaneous_evidence_evaluation"
  ],
  "compatibility": {
    "minimum_reader_version": "2.0.0",
    "upgrade_available_to": ["2.1.0", "2.2.0"],
    "downgrade_possible_to": ["1.2.0_with_loss"]
  },
  "methodology_checksum": "sha256:a3f5c921b4e8d7c9f1a2b3c4d5e6f7a8",
  "breaking_changes_from_previous": [
    "confidence_scores now mandatory",
    "hypothesis_id format changed",
    "parallel_paths replaces sequential_chains"
  ]
}
```

## Version-Specific Validation Rules

### Version 2.0 Validation Requirements
```python
def validate_v2_0_output(output):
    assert output["metadata"]["methodology_version"].startswith("van_evera_2.0")
    assert "hypothesis_evaluation" in output
    assert all(h["van_evera_tests"] for h in output["hypothesis_evaluation"]["primary_hypotheses"])
    assert output["methodological_meta_analysis"]["automated_analysis_metrics"]["hypotheses_generated"] > 10
    assert output["temporal_analysis"]["temporal_consistency_validation"]["forward_causation_verified"] == True
```

### Version 1.x → 2.x Migration Rules
```python
def migrate_v1_to_v2(v1_output):
    v2_output = {
        "metadata": {
            "methodology_version": "van_evera_2.0_fully_automated",
            "migrated_from": v1_output.get("version", "1.0.0"),
            "migration_timestamp": datetime.now().isoformat(),
            "migration_confidence": 0.75  # Reduced due to methodology differences
        },
        # Map old sequential chains to new parallel paths
        "causal_graph_analysis": {
            "critical_paths": convert_sequential_to_parallel(v1_output["causal_chains"])
        },
        # Flag evidence that needs reclassification
        "migration_warnings": [
            "Evidence classification requires review - v1 used post-hoc classification",
            "Counterfactuals limited to v1 manual set - full space not explored",
            "Hypothesis generation was human-limited - additional hypotheses possible"
        ]
    }
    return v2_output
```

## Testing Strategy for Versions

### Cross-Version Testing
```python
def test_version_compatibility():
    """Ensure outputs from different versions can be compared"""
    v1_output = process_with_v1(test_case)
    v2_output = process_with_v2(test_case)
    
    # Core findings should align despite methodology differences
    assert extract_core_findings(v1_output) == extract_core_findings(v2_output)
    
    # V2 should find strictly more than V1
    assert len(v2_output["hypotheses"]) >= len(v1_output["hypotheses"])
    assert len(v2_output["causal_paths"]) >= len(v1_output["causal_chains"])
```

### Version-Specific Test Suites
- **v1.x tests**: Sequential processing, human-scale evidence
- **v2.x tests**: Parallel processing, massive evidence handling, exhaustive generation

## Version Selection Strategy

```python
def select_methodology_version(input_data, user_requirements):
    if user_requirements.get("human_interpretable_steps"):
        return "1.2.0"  # Latest v1
    
    if len(input_data) > 100000:  # Large corpus
        return "2.0.0"  # Requires parallel processing
    
    if user_requirements.get("cross_lingual"):
        return "2.1.0"  # Minimum version for language support
    
    return "2.0.0"  # Default to latest stable
```

## Deprecation Policy

- **Version Support Lifetime**: 2 years from release
- **Deprecation Notice**: 6 months before end-of-support
- **Migration Tools**: Provided for 1 year after deprecation
- **Archive Access**: Permanent read-only access to all versions

## Impact on Testing Strategy

For inside-out TDD with versioning:

1. **Version-Parameterized Tests**: Each test runs against all supported versions
2. **Output Compatibility Tests**: Ensure version converters work correctly
3. **Regression Prevention**: New versions must pass all previous version tests
4. **Performance Benchmarks**: Track performance across versions

```python
@pytest.mark.parametrize("version", ["1.2.0", "2.0.0", "2.1.0"])
def test_american_revolution_analysis(version):
    output = analyze_with_version(american_revolution_text, version)
    validate_output_for_version(output, version)
    assert_core_findings_consistent(output)
```

This versioning strategy ensures that:
- Users can reproduce analyses exactly
- Improvements don't break existing workflows  
- Clear upgrade paths exist
- Methodology evolution is tracked
- Testing covers version compatibility