## Diagnostic Rebalancer Migration Evidence

### File Status
File: core/plugins/diagnostic_rebalancer.py
Status: **ALREADY MOSTLY COMPLIANT** - Uses semantic_service for LLM operations

### Analysis Results

```bash
# The misleadingly named "_rule_based_enhance_edge" actually uses LLM:
$ grep -A5 "_rule_based_enhance_edge" core/plugins/diagnostic_rebalancer.py
def _rule_based_enhance_edge(self, edge: Dict, evidence_desc: str, hypothesis_desc: str, target_type: str) -> Dict:
    """Use semantic_service (LLM-based) assessment to enhance edge diagnostic type"""
    ...
    # Assess probative value based on evidence type and content
    assessment = semantic_service.assess_probative_value(
        evidence_description=evidence_desc,
```

### Changes Made
1. Updated misleading warning from "will use rule-based assessment" to "Using semantic_service for LLM-based assessment"
2. Updated function documentation to clarify it uses LLM-based semantic_service
3. Updated comments to remove "rule-based" references

### Key Features
- Uses `refine_evidence_assessment_with_llm()` when llm_query_func available
- Falls back to `semantic_service.assess_probative_value()` which is also LLM-based
- No actual rule-based logic, just misleading names

### Result
✅ File is COMPLIANT - uses LLM throughout via semantic_service or direct LLM calls