#!/usr/bin/env python3
"""
Test the EXACT scenario from the original code to verify my theory.
"""

print("=== Testing Exact Scenario ===\n")

# Replicate the exact import pattern from the original file
import datetime  # Global import (line 21)

print("1. Global scope - after 'import datetime':")
print(f"   datetime type: {type(datetime)}")

# Test global scope datetime.datetime.now()
try:
    now = datetime.datetime.now()
    print(f"   ✅ datetime.datetime.now() works: {now}")
except Exception as e:
    print(f"   ❌ datetime.datetime.now() fails: {e}")

print()

# Now simulate the function with local import
def test_function():
    """Simulates execute_single_case_processing with local import"""
    print("2. Inside function - with local import:")
    from datetime import datetime  # Local import (line 233) - must be first!
    print(f"   datetime type: {type(datetime)}")
    
    # Test datetime.datetime.now() with class as datetime
    try:
        now = datetime.datetime.now()
        print(f"   ❌ UNEXPECTED: datetime.datetime.now() works: {now}")
        print(f"   This means my theory is WRONG")
    except Exception as e:
        print(f"   ✅ EXPECTED: datetime.datetime.now() fails: {e}")
        print(f"   This confirms my theory")
    
    # Test what should work after local import
    try:
        now = datetime.now()
        print(f"   ✅ datetime.now() works: {now}")
    except Exception as e:
        print(f"   ❌ datetime.now() fails: {e}")

def test_function_wrong_order():
    """Test what happens when local import is AFTER datetime usage"""
    print("\n3. Function with WRONG import order (like original code):")
    
    # Try to use datetime before the local import (this should fail)
    try:
        now = datetime.datetime.now()  # This should cause UnboundLocalError
        print(f"   ❌ UNEXPECTED: datetime.datetime.now() works: {now}")
    except UnboundLocalError as e:
        print(f"   ✅ EXPECTED: UnboundLocalError: {e}")
    except Exception as e:
        print(f"   ❓ Other error: {e}")
    
    # The local import that causes the scoping issue
    from datetime import datetime
    print(f"   After local import, datetime type: {type(datetime)}")

print("Running function test...")
test_function()

test_function_wrong_order()

print("\n4. Back in global scope - after functions:")
print(f"   datetime type: {type(datetime)}")

# Test that global scope is unchanged
try:
    now = datetime.datetime.now()
    print(f"   ✅ datetime.datetime.now() still works: {now}")
except Exception as e:
    print(f"   ❌ datetime.datetime.now() fails: {e}")