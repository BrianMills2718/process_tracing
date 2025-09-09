#!/usr/bin/env python3
"""
Wrapper script to bypass CLI execution hang issue.

ISSUE: `python -m core.analyze` hangs during import phase
SOLUTION: Direct import and function call works perfectly

Usage: python analyze_wrapper.py <graph_file> [args]
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_wrapper.py <graph_file> [args]")
        sys.exit(1)
    
    # Import works fine in this context
    from core.analyze import main as analyze_main
    
    # Pass through all arguments
    sys.argv[0] = 'core.analyze'  # Mimic the original module name
    
    # Call the main function directly
    analyze_main()

if __name__ == "__main__":
    main()