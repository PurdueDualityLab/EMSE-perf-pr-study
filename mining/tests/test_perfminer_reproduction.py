from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import perfminer_reproduction as perfminer


def fake_commit(**overrides):
    method = SimpleNamespace(name="hot_path", start_line=10, end_line=20)
    modified_file = SimpleNamespace(
        new_path="src/cache.py",
        old_path="src/cache.py",
        filename="cache.py",
        is_binary=False,
        diff="@@ -1 +1 @@\n-old\n+new",
        changed_methods=[method],
        change_type=perfminer.ModificationType.MODIFY,
        source_code_before="old",
        source_code="new",
    )
    values = {
        "hash": "a" * 40,
        "msg": "Improve cache lookup performance now",
        "parents": ["b" * 40],
        "committer_date": None,
        "files": 1,
        "modified_files": [modified_file],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_clean_commit_message_matches_operational_rules():
    message = (
        "Fixes #123 improve cache\n"
        "Signed-off-by: Dev <dev@example.com>\n"
        "See https://example.com/details"
    )

    assert perfminer.clean_commit_message(message) == "improve cache See <url>"


def test_inspect_commit_keeps_exact_single_file_single_method_pair():
    result = perfminer.inspect_commit(fake_commit())

    assert result["perfminer_evidence_status"] == "eligible"
    assert result["perfminer_exclusion_reason"] == ""
    assert result["commit_message"] == "Improve cache lookup performance now"
    assert result["filename"] == "src/cache.py"
    assert result["changed_method_name"] == "hot_path"
    assert result["code_diff"].startswith("@@")
    assert len(result["source_pair_md5"]) == 32


def test_inspect_commit_records_filter_reason_without_inference():
    result = perfminer.inspect_commit(fake_commit(files=2))

    assert result["perfminer_evidence_status"] == "excluded"
    assert result["perfminer_exclusion_reason"] == "changed_file_count_not_one"


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def __call__(self, text, text_pair=None, **kwargs):
        self.calls.append((text, text_pair, kwargs))
        if kwargs.get("return_tensors") == "pt":
            rows = len(text)
            return {
                "input_ids": torch.ones((rows, 4), dtype=torch.long),
                "attention_mask": torch.ones((rows, 4), dtype=torch.long),
            }
        first = str(text).split()
        if text_pair is None:
            return {"input_ids": list(range(len(first)))}
        second = str(text_pair).split()
        return {"input_ids": list(range(len(first) + len(second) + 4))}


class FakeModel:
    config = SimpleNamespace(
        num_labels=2,
        id2label={0: "LABEL_0", 1: "LABEL_1"},
        label2id={"LABEL_0": 0, "LABEL_1": 1},
    )

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=torch.tensor([[0.0, 0.0], [2.0, 0.0]]))


def test_pair_metadata_and_inference_use_only_second_at_512_tokens():
    tokenizer = FakeTokenizer()
    metadata = perfminer.pair_token_metadata(tokenizer, "fast cache", "large code diff")

    assert metadata == {"input_tokens": 9, "context_fits": True, "message_fits": True}

    labels, scores = perfminer.classify_loaded_pairs(
        ["first message", "second message"],
        ["first diff", "second diff"],
        tokenizer=tokenizer,
        model=FakeModel(),
        device="cpu",
        batch_size=2,
    )

    assert labels == [1, 0]
    assert scores[0] == 0.5
    inference_call = tokenizer.calls[-1]
    assert inference_call[0] == ["first message", "second message"]
    assert inference_call[1] == ["first diff", "second diff"]
    assert inference_call[2]["truncation"] == "only_second"
    assert inference_call[2]["max_length"] == 512
    assert inference_call[2]["return_token_type_ids"] is False


def test_pair_metadata_reserves_diff_token_when_only_second_must_truncate():
    tokenizer = FakeTokenizer()
    message = " ".join(["token"] * 508)

    metadata = perfminer.pair_token_metadata(tokenizer, message, "diff")

    assert metadata["input_tokens"] == 513
    assert metadata["context_fits"] is False
    assert metadata["message_fits"] is False


def test_figshare_model_weights_are_strictly_verified(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"published weights")
    expected = perfminer.sha256_file(weights)
    monkeypatch.setattr(perfminer, "PERFMINER_MODEL_WEIGHTS_SHA256", expected)
    monkeypatch.setattr(
        perfminer,
        "PERFMINER_MODEL_FILE_SHA256",
        {"model.safetensors": expected},
    )

    assert perfminer.verify_figshare_model(model_dir) == expected

    weights.write_bytes(b"tampered")
    try:
        perfminer.verify_figshare_model(model_dir)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("tampered model bytes must be rejected")


def test_binary_content_is_excluded_before_method_parsing():
    commit = fake_commit()
    commit.modified_files[0].content = b"\0binary"

    result = perfminer.inspect_commit(commit)

    assert result["perfminer_evidence_status"] == "excluded"
    assert result["perfminer_exclusion_reason"] == "binary_file"
