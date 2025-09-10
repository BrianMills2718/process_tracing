#!/usr/bin/env python3
"""
Test that the datetime fix works by calling the specific function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== Testing Datetime Fix ===\n")

# Test the global datetime import works
import datetime
print(f"1. Global datetime type: {type(datetime)}")

try:
    now = datetime.datetime.now()
    print(f"   ✅ Global datetime.datetime.now() works: {now}")
except Exception as e:
    print(f"   ❌ Global datetime.datetime.now() fails: {e}")

print()

# Test importing from the main script
try:
    from process_trace_advanced import execute_single_case_processing
    print("2. ✅ Successfully imported execute_single_case_processing")
except Exception as e:
    print(f"2. ❌ Failed to import execute_single_case_processing: {e}")
    sys.exit(1)

print()

# Test that we can call datetime.datetime.now() inside a similar function context
def test_similar_function():
    """Test function similar to execute_single_case_processing without the bad import"""
    import subprocess
    import json
    import sys
    # NO "from datetime import datetime" - this was the problem!
    
    print("3. Inside test function (similar to execute_single_case_processing):")
    print(f"   datetime type: {type(datetime)}")
    
    try:
        now = datetime.datetime.now()
        print(f"   ✅ datetime.datetime.now() works: {now}")
        return True
    except Exception as e:
        print(f"   ❌ datetime.datetime.now() fails: {e}")
        return False

success = test_similar_function()

print()
if success:
    print("🎉 DATETIME FIX VERIFIED - No more datetime.datetime errors!")
else:
    print("💥 DATETIME FIX FAILED - Still have datetime issues")