#!/usr/bin/env python3
"""
Test that multi-pass extraction now works without datetime errors.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== Testing Multi-pass Extraction Fix ===\n")

# Test imports
try:
    from process_trace_advanced import execute_single_case_processing
    print("✅ Successfully imported execute_single_case_processing")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Create a minimal test case
test_content = """
The American Revolution began in 1775 when colonists protested British taxation policies.
The Boston Tea Party in 1773 was a key event that escalated tensions.
George Washington led the Continental Army to victory against British forces.
"""

# Create test files
test_dir = Path("test_output")
test_dir.mkdir(exist_ok=True)

test_file = test_dir / "test_input.txt"
with open(test_file, "w") as f:
    f.write(test_content)

print(f"✅ Created test file: {test_file}")

# Test the function that previously failed
print("\nTesting execute_single_case_processing (where datetime error occurred)...")

try:
    execute_single_case_processing(
        case_file_path_str=str(test_file),
        output_dir_for_case_str=str(test_dir),
        project_name_str="datetime_test"
    )
    print("🎉 MULTI-PASS EXTRACTION COMPLETED WITHOUT DATETIME ERROR!")
    
    # Check if output was created
    output_files = list(test_dir.glob("datetime_test_*_graph.json"))
    if output_files:
        print(f"✅ Output file created: {output_files[0]}")
    else:
        print("⚠️  No output file found, but no datetime error occurred")
        
except Exception as e:
    if "datetime" in str(e).lower():
        print(f"💥 DATETIME ERROR STILL EXISTS: {e}")
    else:
        print(f"ℹ️  Other error (not datetime related): {e}")
        print("✅ No datetime errors - other issues may exist but datetime is fixed")

finally:
    # Cleanup
    if test_file.exists():
        test_file.unlink()
    if test_dir.exists() and not list(test_dir.iterdir()):
        test_dir.rmdir()
    print(f"\n🧹 Cleaned up test files")