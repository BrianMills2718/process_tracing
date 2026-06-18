#!/usr/bin/env python3
"""Check that active semantic pipeline passes route through the shared LLM wrapper."""

from pathlib import Path

semantic_files = [
    'pt/pass_extract.py',
    'pt/pass_hypothesize.py',
    'pt/pass_test.py',
    'pt/pass_absence.py',
    'pt/pass_synthesize.py',
    'pt/pass_refine.py',
    'pt/pass_binarize.py',
    'pt/pass_propose_model.py',
]

using_llm = []
not_using = []
missing_files = []

for file_path in semantic_files:
    path = Path(file_path)
    if path.exists():
        content = path.read_text()
        if 'call_llm(' in content and 'pt.llm import' in content:
            using_llm.append(file_path)
        else:
            not_using.append(file_path)
    else:
        missing_files.append(file_path)

existing_files = len(semantic_files) - len(missing_files)
print(f"Files using LLM interfaces: {len(using_llm)}/{existing_files} (of {len(semantic_files)} total)")
if existing_files > 0:
    print(f"Real compliance: {len(using_llm)/existing_files*100:.1f}%")
else:
    print("Real compliance: N/A (no files found)")

print("\nUsing LLM:")
for f in using_llm:
    print(f"  [OK] {f}")

print("\nNOT using LLM:")  
for f in not_using:
    print(f"  [X] {f}")

if missing_files:
    print("\nMISSING files:")
    for f in missing_files:
        print(f"  [?] {f}")
