from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_filters import apply_quality_filters, is_config_file
from quality_filters import apply_quality_filters_with_details


def test_config_file_detection_is_conservative():
    assert is_config_file(".github/workflows/ci.yml")
    assert is_config_file("pyproject.toml")
    assert not is_config_file("src/cache.py")


def test_quality_filters_remove_expected_rows():
    df = pd.DataFrame(
        [
            {"id": 1, "filenames": ["src/cache.py"], "title": "perf: cache lookup"},
            {"id": 2, "filenames": [""], "title": "perf: empty filename"},
            {"id": 3, "filenames": [".github/workflows/ci.yml"], "title": "perf: tune ci"},
            {"id": 4, "filenames": ["src/a.py"], "deleted_repo": True, "title": "perf"},
            {"id": 5, "filenames": ["src/a.py"], "title": "Merge branch main"},
        ]
    )
    filtered, counts = apply_quality_filters(df)
    assert filtered["id"].tolist() == [1]
    assert counts == {
        "empty_filename": 1,
        "config_only": 1,
        "deleted_repo": 1,
        "merge_only": 1,
    }


def test_quality_filters_with_details_capture_removed_rows():
    df = pd.DataFrame(
        [
            {"id": 1, "filenames": ["src/cache.py"], "title": "perf: cache lookup"},
            {"id": 2, "filenames": [""], "title": "perf: empty filename"},
            {"id": 3, "filenames": [".github/workflows/ci.yml"], "title": "perf: tune ci"},
        ]
    )
    filtered, counts, removed_records = apply_quality_filters_with_details(df)
    assert filtered["id"].tolist() == [1]
    assert counts == {
        "empty_filename": 1,
        "config_only": 1,
        "deleted_repo": 0,
        "merge_only": 0,
    }
    assert removed_records["empty_filename"][0]["id"] == 2
    assert removed_records["config_only"][0]["id"] == 3


def test_quality_filters_accept_arrow_array_filenames():
    df = pd.DataFrame(
        [
            {"id": 1, "filenames": pa.array(["src/cache.py"]), "title": "perf"},
        ]
    )

    filtered, counts = apply_quality_filters(df)

    assert filtered["id"].tolist() == [1]
    assert counts["empty_filename"] == 0
