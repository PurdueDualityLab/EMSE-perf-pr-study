import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classify_perfannotator_metadata as metadata_classifier
import perfannotator_common


def write_input(path: Path, rows: int = 3) -> pa.Schema:
    schema = pa.schema(
        [
            pa.field("title", pa.string()),
            pa.field("body", pa.string()),
            pa.field("commit_messages", pa.list_(pa.string())),
            pa.field("filenames", pa.list_(pa.string())),
        ]
    )
    values = [
        {
            "title": f"PR {index}",
            "body": None if index == 1 else "Faster",
            "commit_messages": None if index == 1 else [f"commit {index}"],
            "filenames": [f"src/{index}.py"],
        }
        for index in range(rows)
    ]
    table = pa.Table.from_pylist(values, schema=schema)
    pq.write_table(table, path)
    return schema


def run_main(monkeypatch, input_path: Path, output_path: Path, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_perfannotator_metadata.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            *extra,
        ],
    )
    metadata_classifier.main()


def install_fake_inference(monkeypatch) -> list[tuple[str, str | None, str]]:
    loads = []

    def fake_load(model: str, revision: str | None, device: str):
        loads.append((model, revision, device))
        return object(), object(), "cpu"

    def fake_classify(texts, **_kwargs):
        return [index % 2 for index in range(len(texts))], [0.25] * len(texts)

    monkeypatch.setattr(metadata_classifier, "load_model", fake_load)
    monkeypatch.setattr(metadata_classifier, "classify_loaded_model", fake_classify)
    monkeypatch.setattr(
        metadata_classifier,
        "resolved_model_sha256",
        lambda model, revision: "fake-model-sha256",
    )
    return loads


def test_missing_and_list_values_normalize_without_sentinel_text():
    assert perfannotator_common.as_list(pd.NA) == []
    assert perfannotator_common.as_list(np.nan) == []
    assert perfannotator_common.as_list([" b ", pd.NA, np.nan, None, ["a", ""]]) == [
        "b",
        "a",
    ]
    assert perfannotator_common.as_list({"z", "a"}) == ["a", "z"]

    text, truncated = perfannotator_common.metadata_text(
        pd.Series(
            {
                "title": pd.NA,
                "body": np.nan,
                "commit_messages": [pd.NA, " improve cache "],
                "filenames": np.array(["b.py", "a.py"], dtype=object),
            }
        ),
        10_000,
    )

    assert truncated is False
    assert "<NA>" not in text
    assert "nan" not in text
    assert "improve cache" in text
    assert "b.py\na.py" in text


def test_reproducibility_seeds_numpy_and_torch(monkeypatch):
    calls = []
    cudnn = SimpleNamespace(benchmark=True, deterministic=False)
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(("torch", seed)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: calls.append(("cuda", seed)),
        ),
        use_deterministic_algorithms=lambda enabled: calls.append(("deterministic", enabled)),
        backends=SimpleNamespace(cudnn=cudnn),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        perfannotator_common.np.random,
        "seed",
        lambda seed: calls.append(("numpy", seed)),
    )

    perfannotator_common.configure_reproducibility(17, True)

    assert ("numpy", 17) in calls
    assert ("torch", 17) in calls
    assert ("cuda", 17) in calls
    assert ("deterministic", True) in calls
    assert cudnn.benchmark is False
    assert cudnn.deterministic is True


def test_model_artifact_hash_changes_with_model_bytes(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    config.write_text("{}", encoding="utf-8")
    weights.write_bytes(b"first")
    first = perfannotator_common.model_artifact_sha256(model_dir)

    weights.write_bytes(b"second")

    assert perfannotator_common.model_artifact_sha256(model_dir) != first


def test_positive_label_semantics_are_validated():
    valid = SimpleNamespace(
        config=SimpleNamespace(
            num_labels=2,
            id2label={0: "not_performance_improving", 1: "performance_improving"},
            label2id={"not_performance_improving": 0, "performance_improving": 1},
        )
    )
    perfannotator_common.validate_positive_label_semantics(valid)

    generic = SimpleNamespace(
        config=SimpleNamespace(
            num_labels=2,
            id2label={0: "LABEL_0", 1: "LABEL_1"},
            label2id={"LABEL_0": 0, "LABEL_1": 1},
        )
    )
    perfannotator_common.validate_positive_label_semantics(generic)

    reversed_labels = SimpleNamespace(
        config=SimpleNamespace(
            num_labels=2,
            id2label={0: "performance_improving", 1: "not_performance_improving"},
            label2id={"performance_improving": 0, "not_performance_improving": 1},
        )
    )
    with pytest.raises(ValueError, match="label 1"):
        perfannotator_common.validate_positive_label_semantics(reversed_labels)


def test_positive_integer_and_prediction_collision_validation():
    assert metadata_classifier.positive_integer("1") == 1
    assert metadata_classifier.seed_integer("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        metadata_classifier.positive_integer("0")
    with pytest.raises(argparse.ArgumentTypeError):
        metadata_classifier.positive_integer("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        metadata_classifier.seed_integer("-1")

    schema = pa.schema([pa.field("perfannotator_metadata_score", pa.float64())])
    with pytest.raises(ValueError, match="already contains"):
        metadata_classifier.prediction_schema(schema)


def test_run_records_signature_and_rejects_changed_resume(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "classified.parquet"
    write_input(input_path)
    loads = install_fake_inference(monkeypatch)

    run_main(monkeypatch, input_path, output_path, "--row-batch-size", "2", "--device", "cpu")

    output = pq.read_table(output_path)
    assert output.num_rows == 3
    assert output.column("perfannotator_metadata_label_id").to_pylist() == [0, 1, 0]
    state = json.loads(metadata_classifier.state_path(output_path).read_text(encoding="utf-8"))
    signature = state["run_signature"]
    assert signature["input_path"] == str(input_path.resolve())
    assert signature["input_sha256"] == metadata_classifier.sha256_file(input_path)
    assert signature["input_schema_sha256"]
    assert signature["input_row_count"] == 3
    assert signature["row_batch_size"] == 2
    assert state["completed_rows"] == 3
    assert [part["row_count"] for part in state["completed_parts"].values()] == [2, 1]
    assert all(part["sha256"] for part in state["completed_parts"].values())
    assert signature["code_sha256"]
    assert signature["model_artifact_sha256"] == "fake-model-sha256"
    assert state["finalized"] is True
    assert len(loads) == 1

    first_part = metadata_classifier.part_path(metadata_classifier.part_directory(output_path), 1)
    tampered = pq.read_table(first_part)
    score_index = tampered.schema.get_field_index("perfannotator_metadata_score")
    scores = tampered.column(score_index).to_pylist()
    scores[0] = 0.99
    tampered = tampered.set_column(
        score_index,
        tampered.schema.field(score_index),
        pa.array(scores, type=pa.float64()),
    )
    pq.write_table(tampered, first_part)
    with pytest.raises(ValueError, match="recorded SHA-256"):
        run_main(
            monkeypatch,
            input_path,
            output_path,
            "--row-batch-size",
            "2",
            "--device",
            "cpu",
            "--resume",
        )

    with pytest.raises(ValueError, match="run signature"):
        run_main(
            monkeypatch,
            input_path,
            output_path,
            "--row-batch-size",
            "2",
            "--device",
            "cpu",
            "--max-input-chars",
            "100",
            "--resume",
        )
    assert len(loads) == 1


def test_non_resume_rejects_stale_parts(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "classified.parquet"
    write_input(input_path, rows=1)
    parts_dir = metadata_classifier.part_directory(output_path)
    parts_dir.mkdir()
    pq.write_table(pa.table({"stale": [1]}), metadata_classifier.part_path(parts_dir, 1))

    with pytest.raises(FileExistsError, match="Checkpoint parts already exist"):
        run_main(monkeypatch, input_path, output_path)


def test_resume_regenerates_part_not_recorded_by_state(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "classified.parquet"
    write_input(input_path)
    loads = install_fake_inference(monkeypatch)
    run_main(monkeypatch, input_path, output_path, "--row-batch-size", "2", "--device", "cpu")

    checkpoint_path = metadata_classifier.state_path(output_path)
    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state["completed_parts"].pop("part-000001.parquet")
    checkpoint_path.write_text(json.dumps(state), encoding="utf-8")

    run_main(
        monkeypatch,
        input_path,
        output_path,
        "--row-batch-size",
        "2",
        "--device",
        "cpu",
        "--resume",
    )

    resumed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert set(resumed["completed_parts"]) == {
        "part-000001.parquet",
        "part-000002.parquet",
    }
    assert len(loads) == 2


def test_finalization_requires_exact_part_rows_and_schema(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    output_path = tmp_path / "output.parquet"
    schema = pa.schema([pa.field("value", pa.int64())])
    part = metadata_classifier.part_path(parts_dir, 1)
    pq.write_table(pa.table({"value": pa.array([1], type=pa.int64())}), part)

    assert metadata_classifier.finalize_parts(
        parts_dir, output_path, schema, {part.name: 1}
    ) == 1
    assert pq.ParquetFile(output_path).metadata.num_rows == 1

    with pytest.raises(ValueError, match="has 1 rows; expected 2"):
        metadata_classifier.finalize_parts(parts_dir, output_path, schema, {part.name: 2})

    extra = metadata_classifier.part_path(parts_dir, 2)
    pq.write_table(pa.table({"value": pa.array([2], type=pa.int64())}), extra)
    with pytest.raises(ValueError, match="unexpected"):
        metadata_classifier.finalize_parts(parts_dir, output_path, schema, {part.name: 1})

    extra.unlink()
    pq.write_table(pa.table({"value": pa.array(["wrong"], type=pa.string())}), part)
    with pytest.raises(ValueError, match="expected schema"):
        metadata_classifier.finalize_parts(parts_dir, output_path, schema, {part.name: 1})


def test_zero_row_input_writes_empty_output_without_loading_model(monkeypatch, tmp_path):
    input_path = tmp_path / "empty.parquet"
    output_path = tmp_path / "classified.parquet"
    input_schema = write_input(input_path, rows=0)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("zero-row classification must not load the model")

    monkeypatch.setattr(metadata_classifier, "load_model", fail_load)
    run_main(monkeypatch, input_path, output_path)

    output_file = pq.ParquetFile(output_path)
    expected_schema = metadata_classifier.prediction_schema(input_schema)
    assert output_file.metadata.num_rows == 0
    assert output_file.schema_arrow.equals(expected_schema)
    assert metadata_classifier.checkpoint_parts(
        metadata_classifier.part_directory(output_path)
    ) == {}
    state = json.loads(metadata_classifier.state_path(output_path).read_text(encoding="utf-8"))
    assert state["total_rows"] == 0
    assert state["completed_rows"] == 0
    assert state["completed_parts"] == {}
    assert state["finalized"] is True
