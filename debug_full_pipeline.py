#!/usr/bin/env python3
"""Debug the full pipeline to identify exactly where it hangs."""

import sys
import os
import json
import time
from pathlib import Path

def debug_point(msg):
    print(f"[DEBUG {time.time():.1f}] {msg}")
    sys.stdout.flush()

def main():
    debug_point("Starting full pipeline debug")
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    debug_point("Importing all needed functions...")
    from process_trace_advanced import execute_single_case_processing
    
    debug_point("Starting execute_single_case_processing...")
    
    # Use the same parameters as the command line version
    result = execute_single_case_processing(
        "input_text/test_simple/test.txt",
        "output_data/debug_full_test", 
        "test_simple"
    )
    
    debug_point(f"Pipeline complete! Result: {result}")

if __name__ == "__main__":
    try:
        result = main()
        print(f"SUCCESS: {result}")
    except KeyboardInterrupt:
        print("\n[DEBUG] Interrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()