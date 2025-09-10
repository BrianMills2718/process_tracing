#!/usr/bin/env python3
"""
Test datetime import syntax to understand the proper usage.
"""

print("=== Testing datetime import syntax ===\n")

# Test 1: Import entire datetime module
print("Test 1: import datetime")
try:
    import datetime
    now1 = datetime.datetime.now()
    print(f"✅ datetime.datetime.now() works: {now1}")
    
    # This should fail
    try:
        now1_fail = datetime.now()
        print(f"❌ Unexpected: datetime.now() worked: {now1_fail}")
    except AttributeError as e:
        print(f"✅ Expected: datetime.now() fails: {e}")
except Exception as e:
    print(f"❌ datetime.datetime.now() failed: {e}")

print()

# Test 2: Import datetime class directly
print("Test 2: from datetime import datetime")
try:
    # Reset the namespace - remove the module import
    import sys
    if 'datetime' in sys.modules:
        del sys.modules['datetime']
    if 'datetime' in globals():
        del globals()['datetime']
    
    from datetime import datetime
    now2 = datetime.now()
    print(f"✅ datetime.now() works: {now2}")
    
    # This should fail now
    try:
        now2_fail = datetime.datetime.now()
        print(f"❌ Unexpected: datetime.datetime.now() worked: {now2_fail}")
    except AttributeError as e:
        print(f"✅ Expected: datetime.datetime.now() fails: {e}")
except Exception as e:
    print(f"❌ datetime.now() failed: {e}")

print()

# Test 3: What's the difference?
print("Test 3: Understanding the namespace")
import datetime as dt_module
from datetime import datetime as dt_class

print(f"Module type: {type(dt_module)}")
print(f"Class type: {type(dt_class)}")
print(f"Are they the same? {dt_module.datetime is dt_class}")
print(f"Module datetime.now: {hasattr(dt_module, 'now')}")
print(f"Class datetime.now: {hasattr(dt_class, 'now')}")

print()

# Test 4: What happens in the original script context?
print("Test 4: Simulating original script context")
import datetime  # This is how the main script imports it

print(f"After 'import datetime', what is datetime? {type(datetime)}")
print(f"Does datetime have 'datetime' attribute? {hasattr(datetime, 'datetime')}")
print(f"What is datetime.datetime? {type(datetime.datetime)}")

# This should work
now_correct = datetime.datetime.now()
print(f"✅ datetime.datetime.now() works: {now_correct}")

# This should fail
try:
    now_wrong = datetime.now()
    print(f"❌ Unexpected: datetime.now() worked: {now_wrong}")
except AttributeError as e:
    print(f"✅ Expected: datetime.now() fails: {e}")