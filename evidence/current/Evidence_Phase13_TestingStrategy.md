# Phase 13: Comprehensive Testing Strategy

## Testing Philosophy

**Principle**: Validate that LLM-first changes maintain system functionality while eliminating all hardcoded fallbacks.

**Approach**: Incremental testing after each plugin fix to isolate issues quickly.

---

## Pre-Implementation Baseline

### Establish Current State
```bash
# Document current fallback locations
grep -r "else 0\." core/plugins/*.py > baseline_fallbacks.txt

# Record current compliance rate
python validate_true_compliance.py > baseline_compliance.txt

# Test current plugin functionality  
python -c "
from core.plugins.content_based_diagnostic_classifier import ContentBasedDiagnosticClassifierPlugin
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin  
from core.plugins.research_question_generator import ResearchQuestionGeneratorPlugin
print('All plugins load successfully')
" > baseline_plugin_loading.txt

# Run sample analysis to establish working baseline
python -m core.analyze test_data/american_revolution_graph.json > baseline_analysis.txt 2>&1
```

---

## Plugin-Specific Testing Strategy

### Plugin 1: content_based_diagnostic_classifier.py

#### Test Cases
1. **Normal Operation**: Valid LLM response with probative_value
2. **Missing Attribute**: LLM response without probative_value  
3. **LLM Failure**: Exception during LLM call

#### Test Commands
```bash
# Test 1: Normal plugin loading
python -c "
from core.plugins.content_based_diagnostic_classifier import ContentBasedDiagnosticClassifierPlugin
plugin = ContentBasedDiagnosticClassifierPlugin('test')
print(f'Plugin {plugin.id} loaded successfully')
"

# Test 2: Check for remaining fallbacks
grep -n "else 0\." core/plugins/content_based_diagnostic_classifier.py
# Should return no results after fix

# Test 3: Functional test (if test data available)
python -c "
from core.plugins.content_based_diagnostic_classifier import ContentBasedDiagnosticClassifierPlugin
import json
# Test with sample data if available
"
```

#### Expected Results
- Plugin loads without import errors
- No remaining "else 0." patterns
- LLMRequiredError raised when probative_value missing
- Normal operation unchanged

---

### Plugin 2: primary_hypothesis_identifier.py

#### Test Cases
1. **Valid Configuration**: Normal operation with valid PRIMARY_HYPOTHESIS_CRITERIA
2. **Invalid Weight**: Non-numeric weight in configuration
3. **Invalid Threshold**: Non-numeric threshold in configuration  
4. **Missing Configuration**: Corrupted PRIMARY_HYPOTHESIS_CRITERIA

#### Test Commands
```bash
# Test 1: Normal loading and operation
python -c "
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin
plugin = PrimaryHypothesisIdentifierPlugin('test')
print('Configuration values:')
for key, val in plugin.PRIMARY_HYPOTHESIS_CRITERIA.items():
    print(f'  {key}: weight={val[\"weight\"]}, threshold={val[\"minimum_threshold\"]}')
"

# Test 2: Check configuration integrity
python -c "
from core.plugins.primary_hypothesis_identifier import PrimaryHypothesisIdentifierPlugin
plugin = PrimaryHypothesisIdentifierPlugin('test')
criteria = plugin.PRIMARY_HYPOTHESIS_CRITERIA

# Validate all values are numeric
for domain, config in criteria.items():
    assert isinstance(config['weight'], (int, float)), f'Weight not numeric: {domain}'
    assert isinstance(config['minimum_threshold'], (int, float)), f'Threshold not numeric: {domain}'
print('All configuration values are numeric')
"

# Test 3: Check for remaining fallbacks
grep -n "else 0\." core/plugins/primary_hypothesis_identifier.py
# Should return no results after fix
```

#### Mock Testing (if needed)
```python
# Test invalid configuration handling
import pytest
from unittest.mock import patch

def test_invalid_weight_handling():
    with patch.object(plugin, 'PRIMARY_HYPOTHESIS_CRITERIA', {
        'van_evera_score': {'weight': 'invalid', 'minimum_threshold': 0.6}
    }):
        with pytest.raises(LLMRequiredError, match="Invalid.*weight"):
            plugin.execute(test_data)
```

#### Expected Results
- Plugin loads with valid configuration
- LLMRequiredError raised for invalid configuration values
- No fallback values remain
- Academic scoring continues to work

---

### Plugin 3: research_question_generator.py

#### Test Cases
1. **LLM Sophistication Assessment**: Normal operation with LLM scoring
2. **Missing LLM Response**: LLM call fails
3. **Invalid LLM Response**: Response missing probative_value
4. **Comparative Analysis**: Before/after scoring comparison

#### Test Commands
```bash
# Test 1: Plugin loading
python -c "
from core.plugins.research_question_generator import ResearchQuestionGeneratorPlugin
plugin = ResearchQuestionGeneratorPlugin('test')
print(f'Plugin {plugin.id} loaded successfully')
"

# Test 2: Check for remaining rule-based scoring
grep -n "else 0\." core/plugins/research_question_generator.py  
# Should return no results after fix

# Test 3: Validate LLM integration
python -c "
from core.semantic_analysis_service import get_semantic_service
service = get_semantic_service()
print('Semantic service available for sophistication assessment')
"
```

#### Comparative Testing Strategy
```bash
# Before fix: Capture scoring behavior
python -c "
# Save current scoring results with test data
# Store in before_sophistication_scores.json
"

# After fix: Compare scoring behavior  
python -c "
# Generate new scoring results with same test data
# Store in after_sophistication_scores.json
# Compare for reasonableness (not exact match expected)
"
```

#### Expected Results
- Plugin loads and functions with LLM scoring
- Sophistication scores are reasonable (0.0-1.0 range)
- No algorithmic fallback scoring remains
- Research questions generated successfully

---

## Integration Testing

### Full System Tests
```bash
# Test 1: Complete analysis pipeline
python -m core.analyze test_data/american_revolution_graph.json

# Test 2: Plugin registration and loading
python -c "
from core.plugins.register_plugins import register_all_plugins
register_all_plugins()
print('All plugins registered successfully')
"

# Test 3: Van Evera workflow with all plugins
python -c "
from core.van_evera_workflow import VanEveraWorkflow  
workflow = VanEveraWorkflow()
# Test workflow execution if possible
"
```

### Cross-Plugin Dependencies
- Verify plugins still work together in sequence
- Check that output formats remain compatible
- Ensure no circular dependencies introduced

---

## Regression Testing

### Core Functionality Preservation
1. **Van Evera Test Types**: All diagnostic tests still work
2. **Hypothesis Ranking**: Primary hypothesis identification functions  
3. **Research Questions**: Generated questions remain academically sound
4. **Graph Processing**: Node and edge processing unchanged

### Performance Testing
```bash
# Measure execution time before/after
time python -m core.analyze test_data/american_revolution_graph.json

# Monitor LLM call frequency
# Ensure no excessive LLM calls introduced
```

---

## Validation Testing

### Compliance Verification
```bash
# Final compliance check
python validate_true_compliance.py

# Expected improvement from 86.6% to >90%
# Zero "else 0." patterns in plugins
```

### Comprehensive Fallback Search
```bash
# Search all possible fallback patterns
grep -r "else [0-9]" core/plugins/
grep -r "or [0-9]" core/plugins/  
grep -r "= [0-9]\.[0-9]" core/plugins/ | grep -v "PRIMARY_HYPOTHESIS_CRITERIA"

# Should find no semantic fallbacks
```

---

## Error Handling Testing

### LLM Failure Scenarios
1. **Network Issues**: Simulated connection failures
2. **Invalid Responses**: Malformed LLM responses
3. **Missing Attributes**: Incomplete response objects
4. **Service Unavailable**: semantic_service not available

### Test Commands
```python
# Mock LLM failures to verify error handling
from unittest.mock import patch, Mock

# Test 1: LLM service unavailable
with patch('core.semantic_analysis_service.get_semantic_service', side_effect=Exception("Service down")):
    # Should raise LLMRequiredError with clear message

# Test 2: Invalid response format  
mock_response = Mock()
del mock_response.probative_value  # Remove required attribute
with patch('semantic_service.assess_probative_value', return_value=mock_response):
    # Should raise LLMRequiredError about missing attribute
```

---

## Acceptance Criteria Testing

### Must Pass Tests
- [ ] All 3 plugins load without import errors
- [ ] Zero hardcoded fallback values remain (grep verification)
- [ ] System compliance rate improves (>90%)  
- [ ] Full analysis pipeline executes successfully
- [ ] All plugins raise LLMRequiredError appropriately
- [ ] No degradation in core functionality

### Quality Measures
- [ ] Error messages are clear and actionable
- [ ] Performance impact is minimal (<10% slower)
- [ ] Code maintains readability and maintainability
- [ ] Academic soundness preserved in outputs

---

## Test Execution Sequence

### Pre-Fix Baseline
1. Capture current state (fallbacks, compliance, functionality)
2. Document baseline performance and outputs

### Incremental Testing (after each plugin fix)
1. Plugin loading test
2. Fallback removal verification
3. Basic functionality test
4. Commit if passing

### Post-Implementation Validation
1. Full compliance verification
2. Integration testing
3. Regression testing
4. Performance comparison
5. Final documentation

### Rollback Triggers
- Plugin fails to load
- System functionality degraded
- Excessive performance impact (>25% slower)
- Invalid output formats

---

## Evidence Collection

### Required Documentation
1. **Before/After Grep Results**: Showing fallback removal
2. **Compliance Reports**: Numerical improvement verification  
3. **Plugin Loading Tests**: Success confirmations
4. **Functional Outputs**: Sample analysis results
5. **Error Testing Results**: LLMRequiredError behavior
6. **Performance Metrics**: Execution time comparisons

### Evidence Files
- `phase13_baseline_state.md`
- `phase13_plugin1_tests.md` 
- `phase13_plugin2_tests.md`
- `phase13_plugin3_tests.md`
- `phase13_integration_results.md`
- `phase13_final_validation.md`

This comprehensive testing strategy ensures that the LLM-first migration maintains system integrity while achieving the goal of zero hardcoded fallbacks.