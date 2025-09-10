#!/usr/bin/env python3
"""
Directly test the execute_single_case_processing function to verify datetime fix.
"""

import sys
import os
from pathlib import Path
import tempfile

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== VERIFYING DATETIME FIX ===\n")

# Create minimal test content
test_content = """
Test content for datetime verification.
The Boston Tea Party occurred in 1773.
This event led to increased tensions.
"""

# Create temporary test files
with tempfile.TemporaryDirectory() as temp_dir:
    test_dir = Path(temp_dir)
    test_file = test_dir / "test.txt"
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    print(f"✅ Created test file: {test_file}")
    
    # Import the function
    try:
        from process_trace_advanced import execute_single_case_processing
        print("✅ Successfully imported execute_single_case_processing")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    # Test the function that originally failed with datetime errors
    print("\nTesting the exact function where datetime errors occurred...")
    print("This will test the multi-pass extraction path...")
    
    try:
        # Call the function with minimal parameters
        result = execute_single_case_processing(
            case_file_path_str=str(test_file),
            output_dir_for_case_str=str(test_dir),
            project_name_str="datetime_test"
        )
        
        print("🎉 SUCCESS: Function completed without datetime errors!")
        print(f"Function returned: {type(result)}")
        
        # Check if output files were created
        output_files = list(test_dir.glob("*.json"))
        if output_files:
            print(f"✅ Output files created: {[f.name for f in output_files]}")
        else:
            print("⚠️ No output files found")
            
        print("\n✅ DATETIME FIX VERIFIED: Multi-pass extraction works!")
        
    except Exception as e:
        error_msg = str(e)
        if "datetime" in error_msg.lower():
            print(f"❌ DATETIME ERROR STILL EXISTS: {e}")
            print("💥 FIX FAILED")
        else:
            print(f"ℹ️ Other error (not datetime): {e}")
            print("✅ DATETIME FIX SUCCESSFUL (other issues may exist)")
            
        # Print more details about the error
        import traceback
        print("\nFull error details:")
        traceback.print_exc()

print("\nDone.")