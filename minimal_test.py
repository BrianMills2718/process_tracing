#!/usr/bin/env python3
"""
Minimal test to isolate the actual issue
"""
import sys
import subprocess
import time
import os

print("=== MINIMAL ROOT CAUSE TEST ===")

# Test 1: Can we import core.analyze without issues?
print("[TEST 1] Testing core.analyze import...")
try:
    import core.analyze
    print("✅ core.analyze imports successfully")
except Exception as e:
    print(f"❌ core.analyze import failed: {e}")
    sys.exit(1)

# Test 2: Can we call load_graph directly?
print("\n[TEST 2] Testing load_graph function directly...")
try:
    json_file = "output_data/french_revolution/french_revolution_20250910_040946_graph.json"
    if os.path.exists(json_file):
        G, data = core.analyze.load_graph(json_file)
        print(f"✅ load_graph works: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges")
    else:
        print(f"❌ Test file not found: {json_file}")
except Exception as e:
    print(f"❌ load_graph failed: {e}")

# Test 3: Can we run the analyze subprocess directly?
print("\n[TEST 3] Testing analyze subprocess...")
json_file = "output_data/revolutions/revolutions_20250909_025241_graph.json"
if os.path.exists(json_file):
    cmd = [
        sys.executable, '-m', 'core.analyze', 
        json_file, 
        '--html'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd='/home/brian/projects/process_tracing',
            env={**os.environ, 'PYTHONPATH': '/home/brian/projects/process_tracing'}
        )
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT (last 10 lines):")
            print('\n'.join(result.stdout.split('\n')[-10:]))
        if result.stderr:
            print(f"STDERR (last 10 lines):")  
            print('\n'.join(result.stderr.split('\n')[-10:]))
            
        if result.returncode == 0:
            print("✅ Analysis subprocess completed successfully")
        else:
            print(f"❌ Analysis subprocess failed with code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("❌ Analysis subprocess timed out")
    except Exception as e:
        print(f"❌ Analysis subprocess error: {e}")
else:
    print(f"❌ Test file not found: {json_file}")

print("\n=== TEST COMPLETE ===")