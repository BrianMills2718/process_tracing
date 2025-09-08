#!/usr/bin/env python3
"""Debug version of the pipeline to find where it hangs."""

import sys
import os
import time
from pathlib import Path

def debug_point(msg):
    print(f"[DEBUG] {msg}")
    sys.stdout.flush()

def main():
    debug_point("Starting debug pipeline")
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    debug_point("Importing functions...")
    from process_trace_advanced import get_schema, query_llm
    from core.extract import parse_json, PROMPT_TEMPLATE
    
    debug_point("Reading test file...")
    test_file = "input_text/test_simple/test.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    debug_point(f"Text read: {len(text)} characters")
    
    debug_point("Getting schema...")
    schema = get_schema()
    
    debug_point("Preparing prompt...")
    final_system_prompt = PROMPT_TEMPLATE.format(text=text)
    debug_point(f"Prompt prepared: {len(final_system_prompt)} characters")
    
    debug_point("Calling query_llm...")
    raw_json = query_llm(text, schema, final_system_prompt)
    debug_point("query_llm returned")
    
    debug_point("Parsing JSON...")
    graph_data = parse_json(raw_json)
    debug_point(f"Parsed: {len(graph_data.get('nodes', []))} nodes")
    
    debug_point("Process complete!")
    return graph_data

if __name__ == "__main__":
    try:
        result = main()
        print(f"SUCCESS: {result}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()