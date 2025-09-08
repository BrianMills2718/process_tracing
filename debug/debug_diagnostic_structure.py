#!/usr/bin/env python3
"""Debug diagnostic distribution structure"""

import json
import sys
import os
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.plugins.diagnostic_rebalancer import DiagnosticDistribution

# Create a test distribution
test_distribution = DiagnosticDistribution(
    hoop=1,
    smoking_gun=4,
    doubly_decisive=0,
    straw_in_wind=4,
    general=0
)

print("DiagnosticDistribution object:")
print(f"  hoop: {test_distribution.hoop}")
print(f"  total: {test_distribution.total}")
print(f"  percentages: {test_distribution.percentages}")
print(f"  academic_compliance_score: {test_distribution.academic_compliance_score}")

print("\nConverted to dict with asdict():")
dict_version = asdict(test_distribution)
print(json.dumps(dict_version, indent=2))

print("\nContains academic_compliance_score key:", 'academic_compliance_score' in dict_version)