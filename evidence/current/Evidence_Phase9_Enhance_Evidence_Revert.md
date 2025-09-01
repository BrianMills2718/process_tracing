## Enhance Evidence Reversion Evidence

### Reversion Details
File: core/enhance_evidence.py
Action: Reverted to original state without gateway dependency

### Commands and Output

```bash
# Check for gateway imports (before revert)
$ grep -n "llm_gateway\|LLMGateway" core/enhance_evidence.py
21:    from .llm_gateway import LLMGateway
26:        gateway = LLMGateway()

# Revert to version before gateway changes
$ git checkout HEAD~1 -- core/enhance_evidence.py

# Verify no gateway imports (after revert)
$ grep -n "llm_gateway\|LLMGateway" core/enhance_evidence.py
(no output - no matches found)
```

### Current Implementation
The file now uses direct LLM calls with proper error handling:
- Uses google.generativeai directly when no query_llm_func provided
- Returns None on error (graceful failure)
- Parses JSON response and creates Pydantic EvidenceAssessment objects

### Result
✅ Successfully reverted to original implementation without gateway dependency