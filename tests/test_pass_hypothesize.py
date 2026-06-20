"""Tests for pt/pass_hypothesize.py helpers (deterministic, no LLM)."""

from __future__ import annotations

import re

import pytest

from pt.pass_hypothesize import _strip_scaffold_ids

_LEAK = re.compile(r"(?:hyp|actor|evt|evi|mech)_")


class TestStripScaffoldIds:
    @pytest.mark.parametrize("text,expected", [
        ("Distress (mech_social_distress) among the poor", "Distress among the poor"),
        ("Robespierre (actor_robespierre) drove it", "Robespierre drove it"),
        ("Varennes (evt_flight_varennes) and Brunswick (evi_brunswick_manifesto)",
         "Varennes and Brunswick"),
        ("Multiple (evt_girondins_arrested, evi_hebert_executed) escalated", "Multiple escalated"),
        ("No ids here, clean.", "No ids here, clean."),
        ("", ""),
    ])
    def test_strips_and_tidies(self, text, expected):
        out = _strip_scaffold_ids(text)
        assert not _LEAK.search(out), f"id leaked through: {out!r}"
        assert out == expected

    def test_bare_token_removed(self):
        assert not _LEAK.search(_strip_scaffold_ids("war radicalized hyp_war_unite the convention"))
