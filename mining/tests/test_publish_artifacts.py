from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import publish_artifacts as publish  # noqa: E402


def test_portable_value_maps_known_artifact_and_strips_unknown_absolute_path():
    value = {
        "known": "/workspace/input.parquet",
        "unknown": "/home/user/private/run.json",
        "nested": ["relative/file.json"],
    }

    assert publish.portable_value(
        value, {str(Path("/workspace/input.parquet").resolve()): "data/input.parquet"}
    ) == {
        "known": "data/input.parquet",
        "unknown": "run.json",
        "nested": ["relative/file.json"],
    }


def test_assert_portable_rejects_home_path(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"path": "/home/user/private.json"}', encoding="utf-8")

    with pytest.raises(ValueError, match="absolute user path"):
        publish.assert_portable(metadata)


def test_dataset_card_declares_balanced_sample_as_default_config():
    card = publish.dataset_card()

    assert "config_name: balanced-sample" in card
    assert "path: data/sample/balanced_sample.parquet" in card
    assert "default: true" in card


def test_dataset_card_declares_both_optimization_catalogs():
    card = publish.dataset_card()

    assert "config_name: optimization-catalog-original" in card
    assert "path: data/catalog/original_optimization_catalog.csv" in card
    assert "config_name: optimization-catalog-updated" in card
    assert "path: data/catalog/updated_optimization_catalog.csv" in card


def test_dataset_card_declares_curated_labels():
    card = publish.dataset_card()

    assert "config_name: curated-labels" in card
    assert "path: data/curated/pull_request_labels.parquet" in card


def test_inspect_table_reads_csv_rows_and_columns(tmp_path):
    table = tmp_path / "catalog.csv"
    table.write_text("pattern,description\nCaching,Reuse values\n", encoding="utf-8")

    assert publish.inspect_table(table, "csv") == (1, ["pattern", "description"])


def test_artifact_allowlist_contains_only_the_two_catalog_csvs():
    csv_paths = {
        artifact.remote_path for artifact in publish.ARTIFACTS if artifact.format == "csv"
    }

    assert csv_paths == {
        "data/catalog/original_optimization_catalog.csv",
        "data/catalog/updated_optimization_catalog.csv",
    }
