#!/usr/bin/env python3
"""Compatibility entrypoint for the canonical plan-status sync tool."""

from __future__ import annotations

import sys

from meta.sync_plan_status import main


if __name__ == "__main__":
    sys.exit(main())
