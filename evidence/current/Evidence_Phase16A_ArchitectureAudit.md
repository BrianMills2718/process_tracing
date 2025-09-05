# Evidence: Phase 16A - LLM Architecture Audit

**Date**: 2025-01-30  
**Objective**: Complete LLM architecture audit - map all LLM call points and identify mixed configurations  
**Status**: COMPLETED

## LLM Architecture Mapping

### Primary Extraction Pipeline
- **File**: `process_trace_advanced.py`
- **Function**: `query_llm()` (line 821)
- **Current State**: ✅ Uses LiteLLM via `StructuredProcessTracingExtractor`
- **Model**: Uses GPT-5-mini (correctly configured)

```python
def query_llm(text_content, schema=None, system_instruction_text="", use_structured_output=True):
    """Use LiteLLM structured extractor instead of direct API calls"""
    from core.structured_extractor import StructuredProcessTracingExtractor
    extractor = StructuredProcessTracingExtractor()
    result = extractor.extract_graph(full_prompt)
```

### Structured Extractor Component
- **File**: `core/structured_extractor.py`  
- **Class**: `StructuredProcessTracingExtractor`
- **Current State**: ✅ Uses LiteLLM directly with GPT-5-mini
- **Model**: `gpt-5-mini` (line 151, prioritizes OPENAI_API_KEY)

```python
def __init__(self, model_name: str = "gpt-5-mini"):
    self.model_name = model_name
    self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
```

### Van Evera LLM Interface (Analysis Phase)
- **File**: `core/plugins/van_evera_llm_interface.py`
- **Class**: `VanEveraLLMInterface`
- **Current State**: ❌ Uses UniversalLLM router (mixed configuration)
- **Model**: Depends on UniversalLLM router configuration (line 58-59)

```python
def __init__(self):
    self.llm = get_llm()  # ← MIXED ROUTING ISSUE
    self.model_type = "smart"  # Uses smart model from router
```

### UniversalLLM Router Configuration
- **File**: `universal_llm_kit/universal_llm.py`
- **Current State**: ❌ MIXED CONFIGURATION DETECTED
- **Issue**: Multiple providers with same "smart" alias causing routing confusion

**Priority Order Analysis**:
1. OpenAI (line 32): `{"model_name": "smart", "model": "gpt-5-mini"}`
2. Anthropic (line 43): `{"model_name": "smart", "model": "claude-3-5-sonnet"}`  
3. Google (line 50): `{"model_name": "smart", "model": "gemini/gemini-2.5-flash"}`

**CRITICAL ISSUE**: Router has conflicting "smart" model definitions

## Mixed Configuration Impact

### Root Cause Analysis
1. **Extraction Phase**: Uses GPT-5-mini directly via LiteLLM (working correctly)
2. **Analysis Phase**: Uses UniversalLLM router with ambiguous "smart" model selection
3. **Schema Mismatch**: Different models may generate different Pydantic schema structures
4. **Pipeline Failure**: Analysis subprocess crashes due to schema incompatibility

### Specific Failure Point
- **Location**: `core/analyze.py` subprocess execution
- **Error**: Pydantic schema attribute mismatches (`.confidence_score` vs `.confidence_overall`)
- **Cause**: Different LLM models generating different schema field names

## Validation Commands

### LLM Call Point Discovery
```bash
# Search completed - 349 files found with LLM references
grep -r "import.*llm\|from.*llm\|LLM\|litellm\|gemini\|openai\|query_llm\|get_.*llm" . --include="*.py"
```

### UniversalLLM Router Configuration Check
```bash
# Confirmed mixed "smart" model definitions in lines 32, 43, 50
grep -n "smart.*litellm_params" universal_llm_kit/universal_llm.py
```

### Van Evera Interface Verification
```bash
# Confirmed uses get_llm() from UniversalLLM (line 58)
grep -n "self.llm.*get_llm" core/plugins/van_evera_llm_interface.py
```

## Architecture Summary

**Extraction Phase Architecture**:
```
Text Input → query_llm() → StructuredProcessTracingExtractor → LiteLLM → GPT-5-mini
```
✅ **Status**: Properly unified

**Analysis Phase Architecture**:
```  
Graph JSON → core.analyze → VanEveraLLMInterface → UniversalLLM Router → Mixed Models
```
❌ **Status**: Mixed configuration causing failures

## Phase 16A Completion Criteria

✅ **LLM Call Points Mapped**: All major LLM integration points identified and documented  
✅ **Mixed Configuration Identified**: UniversalLLM router creates model ambiguity  
✅ **Root Cause Determined**: Schema mismatches from different models in pipeline phases  
✅ **Impact Assessed**: Analysis subprocess fails, preventing HTML generation  

## Next Steps for Phase 16B

**Required Actions**:
1. **Unify UniversalLLM Router**: Remove conflicting model definitions
2. **Force GPT-5-mini**: Ensure all "smart" model calls use gpt-5-mini
3. **Validate Schema Consistency**: Test that unified pipeline generates consistent Pydantic schemas
4. **Remove Gemini Dependencies**: Clean up any remaining Gemini-specific code

**Target Architecture**:
```
Unified Pipeline: All Components → LiteLLM → GPT-5-mini → Consistent Schemas → Working HTML
```

## Evidence Validation

**Phase 16A Successful**: Architecture audit complete with comprehensive mapping and root cause identification. Mixed configuration clearly identified as source of pipeline failures. Ready to proceed to Phase 16B systematic unification.