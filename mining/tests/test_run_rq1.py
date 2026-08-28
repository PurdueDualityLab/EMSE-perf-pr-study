from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq1_optimization_patterns"))

import pandas as pd

from run_rq1 import (  # noqa: E402
    SYSTEM_INSTRUCTION,
    TRUNCATION_MARKER,
    _output_text,
    format_taxonomy,
    prompt_for,
    response_format,
)


def test_response_format_uses_strict_pattern_label_schema():
    value = response_format()["format"]

    assert value["type"] == "json_schema"
    assert value["strict"] is True
    assert value["schema"]["additionalProperties"] is False
    assert value["schema"]["properties"]["sub_pattern"]["type"] == "string"
    assert set(value["schema"]["required"]) == {
        "explanation",
        "optimization_comparison",
        "high_level_pattern",
        "sub_pattern",
    }


def test_output_text_extracts_structured_response_content():
    body = {"output": [{"content": [{"type": "output_text", "text": '{"x": 1}'}]}]}

    assert _output_text(body) == '{"x": 1}'


def test_prompt_restores_legacy_taxonomy_and_truncation():
    taxonomy = pd.DataFrame(
        [{
            "High-level Pattern": "Algorithm-Level Optimizations",
            "Sub pattern": "Efficient Algorithm",
            "Description": "Use less work.",
            "Example": "Replace quadratic sorting.",
            "Optimized Metrics": "Execution time",
            "Detection": "Inspect complexity.",
        }]
    )
    prompt = prompt_for(
        {"title": "Fast sort", "body": "Improve sorting", "patch": "x" * 15_001},
        taxonomy,
    )

    assert "I have a performance optimization commit" in prompt
    assert "**Title**: Fast sort" in prompt
    assert TRUNCATION_MARKER in prompt
    assert "Description: Use less work." in prompt
    assert "Example: Replace quadratic sorting." in prompt
    assert "Metrics: Execution time" in prompt
    assert "Detection: Inspect complexity." in prompt
    assert "No Meaningful Change" not in prompt
    assert SYSTEM_INSTRUCTION.startswith("You are an expert software engineer")
