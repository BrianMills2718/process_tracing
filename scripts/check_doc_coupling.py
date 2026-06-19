#!/usr/bin/env python3
"""Compatibility entrypoint for the canonical doc-coupling checker."""

from __future__ import annotations

import sys

from meta.check_doc_coupling import main


if __name__ == "__main__":
    sys.exit(main())
