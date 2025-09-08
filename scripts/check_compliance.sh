#!/bin/bash
echo "=== Checking LLM-First Compliance ==="
echo ""

# Counter for non-compliant files
non_compliant=0
total=0

# Check each Python file in core/
for file in $(find ./core -name "*.py" -type f | grep -v __pycache__ | grep -v test); do
  total=$((total + 1))
  
  # Check for various keyword matching patterns
  if grep -E "if\s+['\"].*['\"].*in\s+.*text" "$file" > /dev/null 2>&1; then
    echo "KEYWORD MATCHING: $file"
    grep -n -E "if\s+['\"].*['\"].*in\s+.*text" "$file" | head -2
    echo ""
    non_compliant=$((non_compliant + 1))
    continue
  fi
  
  # Check for hardcoded thresholds
  if grep -E "probative_value\s*=\s*0\.\d+|confidence\s*=\s*0\.\d+" "$file" > /dev/null 2>&1; then
    echo "HARDCODED VALUES: $file"
    grep -n -E "probative_value\s*=\s*0\.\d+|confidence\s*=\s*0\.\d+" "$file" | head -2
    echo ""
    non_compliant=$((non_compliant + 1))
    continue
  fi
  
  # Check for returning None instead of raising error
  if grep -E "return\s+None\s*$|return\s+None\s*#" "$file" > /dev/null 2>&1; then
    # Check if this is in an LLM-related function
    if grep -B5 -E "return\s+None" "$file" | grep -E "llm|LLM|query_llm|semantic" > /dev/null 2>&1; then
      echo "FAIL-FAST VIOLATION: $file"
      grep -n -E "return\s+None" "$file" | head -2
      echo ""
      non_compliant=$((non_compliant + 1))
      continue
    fi
  fi
  
  # Check for domain-specific keyword matching
  if grep -E "if.*ideological.*in|if.*temporal.*in|if.*causal.*in.*text" "$file" > /dev/null 2>&1; then
    echo "DOMAIN KEYWORD MATCHING: $file"
    grep -n -E "if.*ideological.*in|if.*temporal.*in|if.*causal.*in.*text" "$file" | head -2
    echo ""
    non_compliant=$((non_compliant + 1))
    continue
  fi
done

echo "=== Summary ==="
echo "Total core files: $total"
echo "Non-compliant: $non_compliant"
echo "Compliance rate: $(( (total - non_compliant) * 100 / total ))%"