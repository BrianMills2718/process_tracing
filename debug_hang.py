#!/usr/bin/env python3
"""
Minimal script to debug exactly where the hang occurs.
"""
import sys
import time
import threading
import signal
import os

def timeout_handler(signum, frame):
    print(f"[TIMEOUT] Hung at: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
    print(f"[TIMEOUT] Active threads: {threading.active_count()}")
    for t in threading.enumerate():
        print(f"[TIMEOUT] Thread: {t.name}, alive: {t.is_alive()}")
    sys.exit(1)

# Set up timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

print("=== HANG DEBUG SCRIPT ===")
print(f"PID: {os.getpid()}")
print(f"Starting threads: {threading.active_count()}")

# Step 1: Test basic imports
print("[1] Importing sys...")
sys.stdout.flush()

print("[2] Adding path...")
sys.path.append('.')
sys.stdout.flush()

print("[3] Testing core.analyze import...")
sys.stdout.flush()

# This should trigger the same import sequence as -m core.analyze
from core.analyze import main
print("[4] core.analyze imported successfully!")
sys.stdout.flush()

print("[5] About to call main() directly...")
sys.stdout.flush()

# Simulate the command line arguments
class MockArgs:
    json_file = 'output_data/french_revolution/french_revolution_20250910_040946_graph.json'
    html = True
    network_data = 'output_data/french_revolution/french_revolution_network_data.json'

sys.argv = ['core.analyze', MockArgs.json_file, '--html', '--network-data', MockArgs.network_data]

print("[6] Arguments set, calling main()...")
sys.stdout.flush()

main()

print("[7] main() completed!")
signal.alarm(0)  # Cancel timeout