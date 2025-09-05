#!/usr/bin/env python3
"""Debug the execute_single_case_processing function step by step."""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import traceback

def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def main():
    print(f"[{timestamp()}] DEBUGGING execute_single_case_processing")
    
    sys.path.insert(0, os.getcwd())
    
    print(f"[{timestamp()}] 1. Importing modules...")
    import subprocess
    from process_trace_advanced import get_schema
    
    print(f"[{timestamp()}] 2. Setting up paths...")
    case_file_path = Path("input_text/test_simple/test.txt")
    output_dir_for_case = Path("output_data/debug_execute_step_by_step")
    project_name = "test_simple"
    
    print(f"[{timestamp()}] 3. Creating output directory...")
    output_dir_for_case.mkdir(parents=True, exist_ok=True)
    
    print(f"[{timestamp()}] 4. Reading input text...")
    with open(case_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"[{timestamp()}] 5. Checking token limits...")
    estimated_tokens = len(text) // 4
    MAX_TOKENS = 1_000_000
    print(f"[{timestamp()}] Estimated tokens: {estimated_tokens} (limit: {MAX_TOKENS})")
    
    if estimated_tokens >= MAX_TOKENS:
        print(f"[{timestamp()}] ERROR: Would exit due to token limit")
        return
    
    print(f"[{timestamp()}] 6. Getting schema...")
    schema = get_schema()
    
    print(f"[{timestamp()}] 7. Getting prompt template...")
    from core.extract import PROMPT_TEMPLATE
    active_prompt_template = PROMPT_TEMPLATE
    
    print(f"[{timestamp()}] 8. Formatting prompt...")
    final_system_prompt = active_prompt_template.format(text=text)
    print(f"[{timestamp()}] Prompt length: {len(final_system_prompt)}")
    
    print(f"[{timestamp()}] 9. About to call query_llm...")
    try:
        from process_trace_advanced import query_llm
        raw_json = query_llm(text, schema, final_system_prompt)
        print(f"[{timestamp()}] 10. query_llm returned {len(raw_json)} chars")
    except Exception as e:
        print(f"[{timestamp()}] ERROR in query_llm: {e}")
        traceback.print_exc()
        return
    
    print(f"[{timestamp()}] 11. Parsing JSON...")
    try:
        from core.extract import parse_json
        graph_data = parse_json(raw_json)
        print(f"[{timestamp()}] 12. Parsed: {len(graph_data.get('nodes', []))} nodes")
    except Exception as e:
        print(f"[{timestamp()}] ERROR parsing JSON: {e}")
        traceback.print_exc()
        return
    
    print(f"[{timestamp()}] 13. Saving graph...")
    try:
        import json
        graph_json_path = output_dir_for_case / f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_graph.json"
        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"[{timestamp()}] 14. Graph saved to: {graph_json_path}")
    except Exception as e:
        print(f"[{timestamp()}] ERROR saving graph: {e}")
        traceback.print_exc()
        return
    
    print(f"[{timestamp()}] 15. About to call subprocess for analysis...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'core.analyze', 
            str(graph_json_path), 
            '--html'
        ], capture_output=True, text=True, timeout=300)
        
        print(f"[{timestamp()}] 16. Subprocess completed with return code: {result.returncode}")
        print(f"[{timestamp()}] STDOUT length: {len(result.stdout)}")
        print(f"[{timestamp()}] STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print(f"[{timestamp()}] STDOUT sample: {result.stdout[:200]}...")
        if result.stderr:
            print(f"[{timestamp()}] STDERR sample: {result.stderr[:200]}...")
            
    except subprocess.TimeoutExpired:
        print(f"[{timestamp()}] ERROR: Subprocess timed out after 300 seconds")
        return
    except Exception as e:
        print(f"[{timestamp()}] ERROR in subprocess: {e}")
        traceback.print_exc()
        return
    
    print(f"[{timestamp()}] 17. Checking for created files...")
    files = list(output_dir_for_case.iterdir())
    for f in files:
        print(f"[{timestamp()}] Created: {f.name} ({f.stat().st_size} bytes)")
    
    html_files = list(output_dir_for_case.glob("*.html"))
    if html_files:
        print(f"[{timestamp()}] SUCCESS: HTML generated at {html_files[0]}")
    else:
        print(f"[{timestamp()}] WARNING: No HTML files found")
    
    print(f"[{timestamp()}] MANUAL STEP-BY-STEP EXECUTION COMPLETE")

if __name__ == "__main__":
    main()