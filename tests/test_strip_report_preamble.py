"""Tests for strip_report_preamble — dropping LLM process narration."""

import pytest

from tradingagents.agents.utils.agent_utils import strip_report_preamble

pytestmark = pytest.mark.unit


def test_strips_narration_before_heading():
    text = (
        "I now have all the data needed. Let me compile the report.\n"
        "\n"
        "---\n"
        "\n"
        "# NVDA — Technical Analysis Report\n"
        "Body text."
    )
    result = strip_report_preamble(text)
    assert result.startswith("# NVDA — Technical Analysis Report")
    assert "Let me compile" not in result


def test_falls_back_to_horizontal_rule_when_no_heading():
    text = "All data gathered.\n---\n**Recommendation**: Hold\nBody."
    result = strip_report_preamble(text)
    assert result.startswith("---")
    assert "All data gathered" not in result


def test_report_already_clean_is_unchanged():
    text = "# Clean Report\nNo preamble here."
    assert strip_report_preamble(text) == text


def test_no_marker_returns_unchanged():
    text = "Just prose without any markdown structure at all."
    assert strip_report_preamble(text) == text


def test_non_string_content_passes_through():
    blocks = [{"type": "text", "text": "# Report"}]
    assert strip_report_preamble(blocks) is blocks
    assert strip_report_preamble("") == ""
