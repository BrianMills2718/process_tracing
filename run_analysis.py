#!/usr/bin/env python3
"""
Wrapper script to run core.analyze without using -m mode
This bypasses potential circular import issues with python -m execution
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Set the command line arguments
sys.argv = ['core.analyze'] + sys.argv[1:]

# Import and run the analyze module
import core.analyze
core.analyze.main()