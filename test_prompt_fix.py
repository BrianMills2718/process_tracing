#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.extract import PROMPT_TEMPLATE

# Test which prompt is being used
print("Testing prompt template selection...")

# Simulate the logic from process_trace_advanced.py
active_prompt_template = PROMPT_TEMPLATE  # This should use comprehensive template

# Test the formatting
text = "The American Revolution was caused by taxation without representation."

try:
    if active_prompt_template == PROMPT_TEMPLATE:
        print("Using comprehensive PROMPT_TEMPLATE")
        print(f"Template length: {len(PROMPT_TEMPLATE)} characters")
        final_system_prompt = active_prompt_template.format(text=text)
        print(f"Formatted prompt length: {len(final_system_prompt)} characters")
        print("SUCCESS: Comprehensive template formatting works!")
    else:
        print("Using FOCUSED_EXTRACTION_PROMPT")
        print("This test only uses comprehensive template")
        
    # Check if the comprehensive template has the sophisticated node types
    if "Causal_Mechanism" in PROMPT_TEMPLATE:
        print("[OK] Comprehensive template contains Causal_Mechanism")
    else:
        print("[MISSING] Causal_Mechanism not in comprehensive template")
        
    if "Alternative_Explanation" in PROMPT_TEMPLATE:
        print("[OK] Comprehensive template contains Alternative_Explanation")
    else:
        print("[MISSING] Alternative_Explanation not in comprehensive template")
        
    # Count the node types mentioned
    node_types = ["Event", "Hypothesis", "Evidence", "Causal_Mechanism", "Alternative_Explanation", "Actor", "Condition"]
    found_types = sum(1 for t in node_types if t in PROMPT_TEMPLATE)
    print(f"Node types found in comprehensive template: {found_types}/7")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("Test completed successfully!")