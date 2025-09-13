=== STATE CORRUPTION INVESTIGATION ===
Fresh process edge count: 19
Initial edge count: 19
After reload: 19
State preserved: True
=== CACHING PATTERN DETECTION ===

=== LLM EXTRACTION HANG ANALYSIS ===
Based on Task 1 findings:
- ✅ All other components work perfectly 
- ❌ Hang occurs in StructuredProcessTracingExtractor.extract_graph()
- ❌ This suggests LiteLLM API call is hanging, not local processing

HYPOTHESIS: The hang is in the LLM API call (_extract_with_structured_output method)
This would explain why:
1. The pipeline worked before (Phase 26A notes successful extraction)  
2. The hang persists even with --extract-only
3. All other components load and work correctly

