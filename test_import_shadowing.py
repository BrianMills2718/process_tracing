#!/usr/bin/env python3
"""
Test to reproduce the exact import shadowing scenario from the original code.
"""

print("=== Testing Import Shadowing Scenario ===\n")

# Simulate the main script's import pattern
print("Step 1: import datetime (module)")
import datetime
print(f"datetime type: {type(datetime)}")
print(f"Can use datetime.datetime.now(): {hasattr(datetime, 'datetime')}")

# Test that datetime.datetime.now() works
now1 = datetime.datetime.now()
print(f"✅ datetime.datetime.now() works: {now1}")

print("\nStep 2: Import from core.extract (simulated)")
# This simulates: from core.extract import parse_json, analyze_graph_connectivity, ...
# where core.extract.py has "from datetime import datetime"

# Let's check what happens when we import from a module that has "from datetime import datetime"
print("Checking core.extract imports...")

# First, see what's in core.extract namespace
import core.extract
print(f"core.extract has datetime? {hasattr(core.extract, 'datetime')}")
if hasattr(core.extract, 'datetime'):
    print(f"core.extract.datetime type: {type(core.extract.datetime)}")

# Now simulate the problematic import
print("\nStep 3: Simulating 'from core.extract import ...'")
from core.extract import parse_json, analyze_graph_connectivity, create_connectivity_repair_prompt, extract_connectivity_relationships

# Check if datetime is still the module
print(f"After import, datetime type: {type(datetime)}")
module_type_str = "<class 'module'>"
print(f"datetime is still module? {str(type(datetime)) == module_type_str}")

# Test if datetime.datetime.now() still works
try:
    now2 = datetime.datetime.now()
    print(f"✅ datetime.datetime.now() still works: {now2}")
except Exception as e:
    print(f"❌ datetime.datetime.now() fails: {e}")

# Check if the specific imports brought anything into namespace
import sys
frame = sys._getframe()
local_vars = list(frame.f_locals.keys())
print(f"Local variables now: {[v for v in local_vars if 'datetime' in v.lower()]}")