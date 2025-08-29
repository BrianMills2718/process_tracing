# LLM Gateway Design Document

## Overview

The LLM Gateway is a centralized interface for all LLM operations in the process tracing system. It replaces scattered LLM calls with a single, consistent API that enforces LLM-first architecture.

## Class: LLMGateway

### Architecture

```python
class LLMGateway:
    """
    Central gateway for all LLM operations.
    Enforces fail-fast behavior when LLM is unavailable.
    """
    
    def __init__(self):
        """Initialize with required LLM interface"""
        self.llm = require_llm()  # Fails immediately if LLM unavailable
        self._cache = {}  # Session-level cache
        self._stats = {
            'calls': 0,
            'failures': 0,
            'cache_hits': 0
        }
```

## Method signatures

### Core Semantic Analysis Methods

```python
def assess_relationship(
    self,
    evidence: str,
    hypothesis: str,
    context: Optional[str] = None
) -> RelationshipAssessment:
    """
    Assess semantic relationship between evidence and hypothesis.
    Returns structured assessment with confidence and reasoning.
    Raises LLMRequiredError on failure.
    """

def classify_domain(
    self,
    text: str,
    allowed_domains: List[str]
) -> DomainClassification:
    """
    Classify text into domain categories.
    Returns primary domain with confidence score.
    Raises LLMRequiredError on failure.
    """

def evaluate_temporal_relationship(
    self,
    event1: str,
    event2: str,
    temporal_context: Optional[Dict] = None
) -> TemporalEvaluation:
    """
    Evaluate temporal relationship and causal plausibility.
    Returns structured temporal analysis.
    Raises LLMRequiredError on failure.
    """
```

### Van Evera Specific Methods

```python
def determine_diagnostic_type(
    self,
    evidence: str,
    hypothesis: str,
    test_name: str
) -> VanEveraDiagnostic:
    """
    Determine Van Evera diagnostic type for evidence.
    Returns test classification with reasoning.
    Raises LLMRequiredError on failure.
    """

def calculate_probative_value(
    self,
    evidence: str,
    hypothesis: str,
    diagnostic_type: str
) -> float:
    """
    Calculate probative value based on Van Evera methodology.
    Returns value between 0.0 and 1.0.
    Raises LLMRequiredError on failure.
    """

def generate_counterfactual(
    self,
    hypothesis: str,
    context: Dict
) -> CounterfactualAnalysis:
    """
    Generate counterfactual analysis for hypothesis.
    Returns alternative scenarios with plausibility scores.
    Raises LLMRequiredError on failure.
    """
```

### Enhancement Methods

```python
def enhance_hypothesis(
    self,
    hypothesis: str,
    evidence_context: List[str]
) -> EnhancedHypothesis:
    """
    Enhance hypothesis with additional detail and testability.
    Returns enhanced version with improvements noted.
    Raises LLMRequiredError on failure.
    """

def enhance_evidence(
    self,
    evidence: str,
    hypothesis_context: List[str]
) -> EnhancedEvidence:
    """
    Enhance evidence description for better analysis.
    Returns enhanced version with clarity improvements.
    Raises LLMRequiredError on failure.
    """

def identify_causal_mechanism(
    self,
    cause: str,
    effect: str,
    evidence: List[str]
) -> CausalMechanism:
    """
    Identify causal mechanism linking cause and effect.
    Returns mechanism description with confidence.
    Raises LLMRequiredError on failure.
    """
```

### Batch Operations

```python
def batch_evaluate(
    self,
    evidence: str,
    hypotheses: List[Dict[str, str]]
) -> BatchEvaluationResult:
    """
    Evaluate one evidence against multiple hypotheses.
    Efficient single LLM call for batch processing.
    Returns evaluations for all hypotheses.
    Raises LLMRequiredError on failure.
    """

def batch_classify_domains(
    self,
    texts: List[str],
    allowed_domains: List[str]
) -> List[DomainClassification]:
    """
    Classify multiple texts in a single LLM call.
    Returns classifications for all texts.
    Raises LLMRequiredError on failure.
    """
```

## Migration strategy

### Phase 1: Gateway Implementation (Week 2)
1. Create `core/llm_gateway.py` with all method signatures
2. Implement core methods using existing semantic_analysis_service patterns
3. Add proper Pydantic schemas for all return types
4. Implement caching and statistics tracking

### Phase 2: File Migration (Weeks 2-4)
1. **Critical Path First**: Migrate files in main execution flow
   - enhance_evidence.py
   - diagnostic_rebalancer.py
   - temporal_extraction.py

2. **Plugin Migration**: Update high-value plugins
   - advanced_van_evera_prediction_engine.py (remove hardcoded thresholds)
   - van_evera_testing.py (remove algorithmic fallbacks)
   - evidence_connector_enhancer.py

3. **Utility Migration**: Update supporting modules
   - mechanism_detector.py
   - disconnection_repair.py
   - Other enhancement modules

### Phase 3: Validation (Week 5)
1. Run comprehensive tests on migrated files
2. Verify no fallbacks remain
3. Confirm system fails properly without LLM
4. Document performance improvements

## Error handling

### Fail-Fast Principle
```python
class LLMRequiredError(Exception):
    """Raised when LLM is required but unavailable"""
    pass

def require_llm():
    """Get LLM instance or fail immediately"""
    if os.environ.get('DISABLE_LLM') == 'true':
        raise LLMRequiredError("LLM explicitly disabled")
    
    try:
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
        llm = get_van_evera_llm()
        if not llm:
            raise LLMRequiredError("LLM interface not available")
        return llm
    except Exception as e:
        raise LLMRequiredError(f"Cannot operate without LLM: {e}")
```

### No Silent Failures
- **Never return None** on LLM failure
- **Never return empty defaults** ([], {}, 0.0)
- **Always raise LLMRequiredError** with descriptive message
- **Log error before raising** for debugging

### Graceful Degradation (NOT ALLOWED)
The system must NOT:
- Fall back to rule-based methods
- Use hardcoded defaults
- Skip processing silently
- Return partial results

## Migration Patterns

### Pattern 1: Replace Try/Except Fallback
```python
# OLD (BAD)
try:
    result = llm.analyze(text)
    return result
except:
    return None  # Silent failure

# NEW (GOOD)
result = self.gateway.analyze(text)  # Raises LLMRequiredError on failure
return result
```

### Pattern 2: Replace Hardcoded Values
```python
# OLD (BAD)
threshold = 0.7  # Hardcoded

# NEW (GOOD)
threshold = self.gateway.determine_threshold(
    context=context,
    test_type=test_type
)  # LLM determines appropriate threshold
```

### Pattern 3: Replace Word Overlap
```python
# OLD (BAD)
overlap = len(set(words1) & set(words2)) / len(words1)

# NEW (GOOD)
similarity = self.gateway.assess_semantic_similarity(
    text1=text1,
    text2=text2
)  # LLM assesses semantic similarity
```

## Performance Considerations

### Caching Strategy
- Cache at gateway level for session
- Key: hash of (method_name, parameters)
- TTL: Session duration (no persistent cache)
- Clear cache on configuration changes

### Batch Operations
- Combine multiple evaluations into single LLM call
- Reduces API calls by 70-90% for multi-hypothesis scenarios
- Improves consistency across related evaluations

### Statistics Tracking
- Count total LLM calls
- Track cache hit rate
- Monitor failure rate
- Log slow operations (>2s)

## Testing Strategy

### Unit Tests
```python
def test_gateway_requires_llm():
    """Gateway must fail without LLM"""
    os.environ['DISABLE_LLM'] = 'true'
    with pytest.raises(LLMRequiredError):
        gateway = LLMGateway()

def test_no_silent_failures():
    """Methods must raise errors, not return None"""
    gateway = LLMGateway()
    # Mock LLM failure
    gateway.llm = None
    with pytest.raises(LLMRequiredError):
        gateway.assess_relationship("evidence", "hypothesis")
```

### Integration Tests
- Test with real LLM connection
- Verify Pydantic schema compliance
- Test batch operations efficiency
- Validate caching behavior

## Success Metrics

### Coverage Metrics
- Target: 100% of semantic operations use gateway
- Current: ~30% (semantic_analysis_service only)
- Week 2 Goal: 50% coverage
- Week 5 Goal: 100% coverage

### Quality Metrics
- Zero silent failures
- Zero hardcoded semantic values
- All Van Evera tests LLM-determined
- Proper error messages on all failures

### Performance Metrics
- 70% reduction in LLM calls (via batching)
- <100ms gateway overhead
- 90% cache hit rate for repeated operations
- <2s average LLM response time