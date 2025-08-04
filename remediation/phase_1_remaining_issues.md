# Phase 1: Remaining Issues
*After the 5 critical fixes, address these remaining foundation issues*

## Data Structure Issues (B1)
- **#14**: Probative Value Scale - Standardize to 0.0-1.0
- **#15**: Bayes Factor Range - Allow values > 1.0
- **#35**: Property Access Patterns - Standardize node property access
- **#83**: Cross-Case Data Structure - Add semantic matching

## Mathematical Errors (B2)
- **#17**: Cross-Case Statistics - Add proper significance testing
- **#44**: Probability Validation - Handle invalid values properly
- **#45**: Balance Score Logic - Fix double negation edge case
- **#62**: Graph Invariants - Validate edge references
- **#63**: Floating Point Comparison - Use ε=1e-9 for all comparisons
- **#82**: Hypothesis Status - Use elimination logic not scores

## Graph Processing (B3)
- **#19**: Edge Data Validation - Check for None returns
- **#20**: Connectivity Logic - Fix weak/strong connectivity  
- **#52**: Empty Graph Handling - Support single-node graphs
- **#61**: Directed Graph Metrics - Use in/out degree correctly
- **#81**: Mechanism Completeness - Fix arbitrary calculation

## Pipeline Issues (B4)
- **#22**: Schema Validation - Actually call validation functions
- **#23**: Counterfactual Analysis - Implement graph-based logic
- **#37**: Error Propagation - Stop on critical failures
- **#84**: Property Names - Align prompts with schema
- **#85**: Token Limit - Hard fail if >1M tokens (NO chunking)

## Performance (C1-C3)
- **#43**: Memory Limits - Bound path enumeration memory
- **#46**: O(N²) Algorithms - Add indexing for actor lookup
- **#47**: Redundant Traversals - Single-pass analysis
- **#48**: String Encoding - Remove double encoding
- **#49-51**: Memory Leaks - Close figures, buffers, subgraphs
- **#58**: Race Conditions - Add unique timestamps
- **#64-67**: Resource Management - File rotation, size limits

## Error Handling (F)
- **#24, #26**: Exception Handling - Try-catch JSON/operations
- **#25**: Type Validation - Add runtime type checks
- **#33**: Debug File Locking - Prevent concurrent writes
- **#36**: Logging System - Replace safe_print with logger
- **#38**: Cleanup Patterns - Use context managers

## Development (G)
- **#39-40**: File I/O - Use UTF-8 + pathlib for cross-platform
- **#41-42**: Testing/Validation - Create test framework
- **#59-60**: Threading - Use Qt5Agg backend for thread safety
- **#88**: Repository Hygiene - Add .gitignore
- **#101**: Import Side Effects - Lazy configuration