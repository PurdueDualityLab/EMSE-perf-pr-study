import pytest

from analysis.rq3_llm_validation.extend_binary_audit import pairs_for, quote_locators


def record(record_id, text):
    return {"record_id": record_id, "locator": record_id, "text": text}


def test_measurement_pairs_do_not_cross_record_or_workload_boundaries():
    records = {
        "r1": record("r1", "| workload + exec | 2 s | 1 s | -50 % |\n"),
        "r2": record("r2", "| workload + rss memory | 2 MB | 3 MB | +50 % |\n"),
    }
    assert pairs_for(records) == []
    records["r1"]["text"] += "| other + rss memory | 2 MB | 3 MB | +50 % |\n"
    assert pairs_for(records) == []


def test_all_reported_directions_are_kept_without_a_significance_threshold():
    text = "\n".join([
        "| Name | Base | Current | Change |",
        "| a + exec | 2 s | 1 s | -50 % |",
        "| a + rss memory | 2 MB | 1 MB | -50 % |",
        "| b + exec | 2 s | 2 s | -0.01 % |",
        "| b + rss memory | 2 MB | 2 MB | +0.01 % |",
        "| c + exec | 2 s | 3 s | +50 % |",
        "| c + rss memory | 2 MB | 1 MB | -50 % |",
    ])
    pairs = pairs_for({"r1": record("r1", text)})
    assert {p["workload"]: p["reported_directional_relationship"] for p in pairs} == {
        "a": "joint_improvement", "b": "tradeoff", "c": "no_gain"}
    for pair in pairs:
        assert text[pair["time_start"]:pair["time_end"]] == pair["time_source_row"]
        assert text[pair["memory_start"]:pair["memory_end"]] == pair["memory_source_row"]


def test_added_case_quotes_require_original_record_support():
    records = {"r1": record("r1", "<code>latency</code>: 2 ms")}
    aligned = quote_locators(records, ["`latency`: 2 ms"])
    assert aligned[0]["status"] == "formatting_aligned"
    assert aligned[0]["original_excerpt"] in records["r1"]["text"]
    with pytest.raises(ValueError, match="no supplied-source match"):
        quote_locators(records, ["latency: 3 ms"])
