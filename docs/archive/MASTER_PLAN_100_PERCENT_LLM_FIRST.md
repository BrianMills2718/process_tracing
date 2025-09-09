# Master Plan: Achieving TRUE 100% LLM-First Architecture

## Current State Analysis (Honest Assessment)

### The Numbers
- **Files with require_llm**: 5/67 (7.5%)
- **Files using semantic_service**: 16/67 (23.9%)
- **Files with fallbacks**: 17/67 (25.4%)
- **Files with hardcoded values**: 25/67 (37.3%)
- **TRUE LLM-first coverage**: ~20-30%

### Core Problems
1. **Scattered Fallbacks**: 17+ files return None/defaults when LLM fails
2. **Hardcoded Values**: 25+ files have magic numbers for semantic decisions
3. **Bypass Paths**: Many plugins don't use semantic_analysis_service
4. **External Dependencies**: Files use process_trace_advanced.query_llm (not controlled)
5. **Dead Code**: Unused files creating confusion

## Strategic Plan for 100% LLM-First

### Phase 1: Assessment & Classification (2 days)

#### 1.1 File Classification
Classify all 67 files into:

**Category A: Core Semantic** (MUST be LLM-first)
- All hypothesis/evidence evaluation
- Domain classification
- Relationship detection
- Confidence scoring
- Test generation

**Category B: Computational** (DON'T need LLM)
- Graph algorithms
- File I/O
- Pure math calculations
- Data structure management

**Category C: Hybrid** (Selective LLM)
- Visualization (layout vs semantic labels)
- Validation (structure vs content)
- Reporting (format vs narrative)

**Category D: Dead Code** (DELETE)
- confidence_calculator.py (confirmed unused)
- Other unused files

#### 1.2 Pattern Inventory
Document ALL instances of:
```python
# Fallback patterns to find:
- return None
- return {}
- return 0.5
- except: return default
- if not llm: fallback
- = 0.X  # hardcoded threshold
```

### Phase 2: Create Centralized LLM Gateway (3 days)

#### 2.1 Design Central Gateway
```python
# core/llm_gateway.py
class LLMGateway:
    """
    SINGLE point of LLM access for entire system.
    NO FALLBACKS. FAILS IMMEDIATELY if LLM unavailable.
    """
    
    def __init__(self):
        self.llm = require_llm()  # Fails if no LLM
        
    # Typed methods for EVERY semantic operation
    def classify_domain(self, text: str) -> DomainClassification
    def assess_confidence(self, hypothesis: str, evidence: str) -> ConfidenceScore
    def detect_contradiction(self, text1: str, text2: str) -> ContradictionResult
    def determine_threshold(self, context: str, domain: str) -> float
    # ... 50+ more specific methods
```

#### 2.2 Migration Strategy
- ALL files must use LLMGateway
- NO direct LLM calls elsewhere
- Single point of failure (good!)
- Type-safe interfaces

### Phase 3: Critical Path Migration (5 days)

#### Priority 1: Main Execution Path
Files directly used by analyze.py:

1. **enhance_evidence.py**
   - Current: Returns None on LLM failure
   - Fix: Use LLMGateway, raise LLMRequiredError

2. **enhance_mechanisms.py**
   - Current: Uses external query_llm
   - Fix: Use LLMGateway

3. **llm_reporting_utils.py**
   - Current: Uses external query_llm
   - Fix: Use LLMGateway

4. **temporal_* files** (5 files)
   - Check each for semantic operations
   - Convert to LLMGateway

#### Priority 2: Semantic Service Users
16 files that use semantic_analysis_service:
- Ensure they handle LLMRequiredError
- Remove any local fallbacks

### Phase 4: Plugin Migration (5 days)

#### 4.1 Active Plugins (Must Fix)
Based on register_plugins.py:

1. **content_based_diagnostic_classifier**
   - Remove keyword matching
   - Use LLMGateway for classification

2. **diagnostic_rebalancer**
   - Remove hardcoded rebalancing rules
   - Use LLM for all adjustments

3. **alternative_hypothesis_generator**
   - Remove template-based generation
   - Full LLM generation

4. **bayesian_van_evera_engine**
   - Remove hardcoded priors
   - LLM-determined parameters

5. **advanced_van_evera_prediction_engine**
   - Fix 18 hardcoded thresholds
   - Major refactoring needed

#### 4.2 Plugin Pattern
```python
class PluginName(ProcessTracingPlugin):
    def __init__(self):
        super().__init__()
        self.llm_gateway = LLMGateway()  # Fails if no LLM
        # NO fallback initialization
```

### Phase 5: Systematic Fallback Removal (3 days)

#### 5.1 Search & Destroy Pattern
```bash
# Find all fallbacks
grep -r "return None" core/
grep -r "except.*return" core/
grep -r "= 0\.[0-9]" core/
```

#### 5.2 Replacement Pattern
```python
# For EVERY fallback found:

# OLD
try:
    result = some_operation()
except:
    return None  # or return 0.5, return {}

# NEW
try:
    result = self.llm_gateway.operation_name(params)
except Exception as e:
    raise LLMRequiredError(f"LLM required for {operation}: {e}")
```

### Phase 6: Hardcoded Value Replacement (3 days)

#### 6.1 Threshold Identification
```python
# Find ALL semantic thresholds:
confidence > 0.7  # Semantic judgment
similarity >= 0.5  # Semantic comparison
probability = 0.3  # Prior belief
weight = 0.4      # Importance judgment
```

#### 6.2 Dynamic Replacement
```python
# OLD
if confidence > 0.7:
    return "high_confidence"

# NEW
threshold = self.llm_gateway.determine_threshold(
    metric="confidence",
    context=context,
    domain=domain
)
if confidence > threshold:
    return "high_confidence"
```

### Phase 7: Validation Framework (2 days)

#### 7.1 Comprehensive Test Suite
```python
# validate_100_percent_llm_first.py

def test_no_fallbacks():
    """Ensure NO fallback patterns exist"""
    forbidden_patterns = [
        r"return None(?!.*raise)",
        r"return {}(?!.*raise)",
        r"except.*:.*return(?!.*raise)",
        r"= 0\.\d+.*#.*default"
    ]
    
def test_all_files_fail_without_llm():
    """Every semantic file must fail without LLM"""
    os.environ['DISABLE_LLM'] = 'true'
    for file in semantic_files:
        assert_raises(LLMRequiredError, import_file, file)
        
def test_no_bypass_paths():
    """No way to bypass LLM requirement"""
    # Test every entry point fails without LLM
```

#### 7.2 Continuous Monitoring
- Pre-commit hooks to prevent fallback introduction
- CI/CD checks for LLM requirement
- Coverage reports for LLM usage

### Phase 8: Documentation & Training (2 days)

#### 8.1 Developer Guide
- How to use LLMGateway
- No fallback policy
- Testing requirements
- Common patterns

#### 8.2 Architecture Documentation
- System requires LLM (no exceptions)
- All semantic decisions via LLMGateway
- Failure behavior documented

## Implementation Timeline

### Week 1: Foundation
- Day 1-2: Assessment & Classification
- Day 3-5: Build LLMGateway

### Week 2: Critical Path
- Day 6-8: Fix main execution path
- Day 9-10: Fix semantic service users

### Week 3: Plugins
- Day 11-15: Migrate all active plugins

### Week 4: Cleanup
- Day 16-18: Remove all fallbacks
- Day 19-20: Replace hardcoded values

### Week 5: Validation
- Day 21-22: Build validation framework
- Day 23-24: Documentation
- Day 25: Final testing

## Success Criteria

### Quantitative Metrics
- **0** fallback patterns in codebase
- **0** hardcoded semantic thresholds
- **100%** of semantic operations use LLMGateway
- **100%** of files fail without LLM (for semantic operations)

### Qualitative Metrics
- System fails immediately without LLM
- Clear error messages (no silent failures)
- Single point of LLM access (maintainable)
- Type-safe LLM operations

## Risk Analysis

### Risks
1. **Breaking Changes**: Removing fallbacks breaks existing code
2. **Performance**: Too many LLM calls slow system
3. **Cost**: Increased LLM usage costs
4. **Complexity**: Central gateway becomes bottleneck

### Mitigations
1. **Testing**: Comprehensive test coverage before deployment
2. **Caching**: Intelligent caching in LLMGateway
3. **Batching**: Batch similar operations
4. **Monitoring**: Track usage and costs

## Expected Outcome

After implementing this plan:
- **100% LLM-first** for all semantic operations
- **Zero tolerance** for fallbacks
- **Fail-fast** architecture
- **Maintainable** and extensible system

## Why Previous Attempts Failed

1. **Piecemeal Approach**: Fixed random files instead of systematic
2. **No Central Control**: Each file handled LLM independently
3. **Incomplete Fixes**: Added TODOs instead of fixing
4. **Ignored Dependencies**: Didn't trace full execution paths
5. **Overestimated Progress**: Claimed victory too early

## This Plan Will Succeed Because

1. **Systematic**: Covers ALL files, not just convenient ones
2. **Centralized**: Single LLMGateway for control
3. **Comprehensive**: Addresses all fallback patterns
4. **Measurable**: Clear metrics for success
5. **Realistic**: 5-week timeline with buffer

---

**This is a serious engineering effort requiring ~25 days of focused work to achieve TRUE 100% LLM-first architecture.**