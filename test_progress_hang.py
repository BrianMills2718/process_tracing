#!/usr/bin/env python3
"""
Test to verify if progress.checkpoint is really the hang cause
"""
import sys
import time
sys.path.append('.')

print("Testing progress.checkpoint in isolation...")

# Import the progress object from analyze.py  
from core.analyze import progress

print("Progress object imported successfully")

# Test 1: Simple checkpoint call
print("Test 1: Calling progress.checkpoint...")
try:
    result = progress.checkpoint("test", "Testing checkpoint functionality")
    print(f"✅ Test 1 passed: {result}")
except Exception as e:
    print(f"❌ Test 1 failed: {e}")

# Test 2: Multiple checkpoint calls
print("Test 2: Multiple checkpoints...")
try:
    progress.checkpoint("test1", "First test")
    progress.checkpoint("test2", "Second test") 
    progress.checkpoint("test3", "Third test")
    print("✅ Test 2 passed")
except Exception as e:
    print(f"❌ Test 2 failed: {e}")

# Test 3: Checkpoint with file path (like load_graph does)
print("Test 3: Checkpoint with file path...")
try:
    progress.checkpoint("load_graph", "Loading from output_data/french_revolution/french_revolution_20250910_040946_graph.json")
    print("✅ Test 3 passed")
except Exception as e:
    print(f"❌ Test 3 failed: {e}")

print("All progress.checkpoint tests completed!")