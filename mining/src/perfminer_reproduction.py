from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path
import re
from typing import Any

from pydriller import ModificationType

from perfannotator_common import batched, validate_positive_label_semantics


PERFMINER_REPOSITORY = "mak-azad/perfminer"
PERFMINER_REVISION = "73d1c2cb23e3f1f1ce9d4909262a1763ccadae29"
PERFMINER_PAPER_URL = "https://realiselab.github.io/publications/PERFMINER_msr.pdf"
PERFMINER_FIGSHARE_DOI = "10.6084/m9.figshare.32142181.v1"
PERFMINER_MODEL_WEIGHTS_SHA256 = (
    "c8e71790c6dc286562df297b40405d5c7ee8ad8bb0ca1fb47490949f6b5a47dd"
)
PERFMINER_MODEL_FILE_SHA256 = {
    "config.json": "39e3529f5371abcaaba43015887718cff096c4c573dfd387b7033aa75cefd15c",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": PERFMINER_MODEL_WEIGHTS_SHA256,
    "operating_point.json": "87b0c0c0ab59430b091d88ec62f813fae72501027bee8ca88cc65f4942b06412",
    "special_tokens_map.json": "8293ae960b0a0852d4d3813118030a1149a3ed9fe37bc2e1e7b3c2e62eb2d4b7",
    "tokenizer_config.json": "e66f9d5dfa61fc8f8442242601e21569f93bec58b47c11a8eb5f49d47fba48cd",
    "tokenizer.json": "9d060f065409a90167737ea0ac5349689e860ba45d0941e49260128b6399036a",
    "vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
}
PERFMINER_MODEL_ARTIFACT_SHA256 = (
    "9a0107e3fde617214c7d58cffda8dc29043adfcdeeee63331c51f458b05b40a6"
)
PERFMINER_MAX_LENGTH = 512
PERFMINER_THRESHOLD = 0.5
PERFMINER_TRUNCATION = "only_second"
PERFMINER_SUPPORTED_EXTENSIONS = (
    ".java",
    ".py",
    ".cu",
    ".cuh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".c++",
    ".cxx",
)
PERFMINER_MAX_DIFF_CHARS = 500_000
PERFMINER_MAX_SOURCE_BEFORE_CHARS = 1_000_000


_FIXES_RE = re.compile(r"fix(es)?\s+#\d+", re.I)
_MERGE_PULL_RE = re.compile(r"merge pull request[^\n]+", re.I)
_MERGE_BRANCH_RE = re.compile(r"merge (remote-tracking )?branch[^\n]+", re.I)
_SIGNATURE_RE = re.compile(r"(Signed-off-by|Reviewed-By|Change-Id):[^\n]+", re.I)
_GIT_SVN_RE = re.compile(r"git-svn-id:[^\n]+", re.I)
_BOT_RE = re.compile(r"(dependabot|renovate)[\s\S]*", re.I)
_TICKET_RE = re.compile(r"Ticket:\s*[^\n]+", re.I)
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")


def clean_commit_message(message: str | None) -> str:
    """Apply the operational PerfMiner commit-message cleanup verbatim."""
    text = message or ""
    for pattern in (
        _FIXES_RE,
        _MERGE_PULL_RE,
        _MERGE_BRANCH_RE,
        _SIGNATURE_RE,
        _GIT_SVN_RE,
        _BOT_RE,
        _TICKET_RE,
    ):
        text = pattern.sub("", text)
    text = _URL_RE.sub("<url>", text)
    text = _EMAIL_RE.sub("<email>", text)
    return " ".join(text.replace("\n", " ").split())


def commit_manifest_sha256(commits: Iterable[tuple[int, str]]) -> str:
    """Identify an ordered PR commit snapshot without relying on row order elsewhere."""
    digest = hashlib.sha256()
    for index, sha in commits:
        digest.update(int(index).to_bytes(8, byteorder="big", signed=False))
        digest.update(b"\0")
        digest.update(str(sha).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _base_evidence(commit: Any) -> dict[str, Any]:
    raw_message = str(getattr(commit, "msg", "") or "")
    parents = list(getattr(commit, "parents", []) or [])
    commit_date = getattr(commit, "committer_date", None)
    return {
        "commit_sha": str(getattr(commit, "hash", "") or ""),
        "commit_message_raw": raw_message,
        "commit_message": "",
        "commit_message_word_count": len(raw_message.split()),
        "commit_date": commit_date.isoformat() if commit_date is not None else "",
        "parent_count": len(parents),
        "changed_file_count": int(getattr(commit, "files", 0) or 0),
        "filename": "",
        "change_type": "",
        "is_binary": False,
        "changed_method_count": 0,
        "changed_method_name": "",
        "changed_method_start_line": None,
        "changed_method_end_line": None,
        "code_diff": "",
        "diff_chars": 0,
        "source_before_chars": None,
        "source_after_chars": None,
        "source_pair_md5": "",
        "perfminer_evidence_status": "excluded",
        "perfminer_exclusion_reason": "",
    }


def _excluded(record: dict[str, Any], reason: str) -> dict[str, Any]:
    record["perfminer_exclusion_reason"] = reason
    return record


def inspect_commit(commit: Any) -> dict[str, Any]:
    """Extract the exact message/diff pair after PerfMiner's operational filters."""
    record = _base_evidence(commit)
    if record["parent_count"] > 1:
        return _excluded(record, "merge_commit")

    word_count = record["commit_message_word_count"]
    if word_count >= 1_000:
        return _excluded(record, "commit_message_at_least_1000_words")
    if word_count <= 3:
        return _excluded(record, "commit_message_at_most_3_words")

    cleaned_message = clean_commit_message(record["commit_message_raw"])
    record["commit_message"] = cleaned_message
    # The published miner performs this case-sensitive substring check.
    if "merge" in cleaned_message:
        return _excluded(record, "cleaned_message_contains_merge")
    if "revert" in cleaned_message:
        return _excluded(record, "cleaned_message_contains_revert")

    if record["changed_file_count"] != 1:
        return _excluded(record, "changed_file_count_not_one")

    modified_files = list(getattr(commit, "modified_files", []) or [])
    if len(modified_files) != 1:
        return _excluded(record, "modified_file_count_not_one")
    modified_file = modified_files[0]
    filename = str(
        getattr(modified_file, "new_path", None)
        or getattr(modified_file, "old_path", None)
        or getattr(modified_file, "filename", "")
        or ""
    )
    record["filename"] = filename
    if not any(filename.endswith(extension) for extension in PERFMINER_SUPPORTED_EXTENSIONS):
        return _excluded(record, "unsupported_file_extension")

    raw_contents = (
        getattr(modified_file, "content_before", None),
        getattr(modified_file, "content", None),
    )
    record["is_binary"] = any(
        b"\0" in bytes(content)
        for content in raw_contents
        if isinstance(content, (bytes, bytearray, memoryview))
    )
    if record["is_binary"]:
        return _excluded(record, "binary_file")

    code_diff = str(getattr(modified_file, "diff", "") or "")
    record["code_diff"] = code_diff
    record["diff_chars"] = len(code_diff)
    if record["diff_chars"] > PERFMINER_MAX_DIFF_CHARS:
        return _excluded(record, "diff_over_500000_chars")

    source_before = str(getattr(modified_file, "source_code_before", None) or "na")
    source_after = str(getattr(modified_file, "source_code", None) or "na")
    record["source_before_chars"] = None if source_before == "na" else len(source_before)
    record["source_after_chars"] = None if source_after == "na" else len(source_after)
    if source_before != "na" and len(source_before) > PERFMINER_MAX_SOURCE_BEFORE_CHARS:
        return _excluded(record, "source_before_over_1000000_chars")

    changed_methods = list(getattr(modified_file, "changed_methods", []) or [])
    record["changed_method_count"] = len(changed_methods)
    if len(changed_methods) != 1:
        return _excluded(record, "changed_method_count_not_one")

    change_type = getattr(modified_file, "change_type", None)
    record["change_type"] = str(getattr(change_type, "name", change_type) or "")
    if change_type in {ModificationType.ADD, ModificationType.DELETE}:
        return _excluded(record, "file_added_or_deleted")

    method = changed_methods[0]
    record["changed_method_name"] = str(getattr(method, "name", "") or "")
    record["changed_method_start_line"] = getattr(method, "start_line", None)
    record["changed_method_end_line"] = getattr(method, "end_line", None)
    record["source_pair_md5"] = hashlib.md5(
        (source_before + source_after).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    record["perfminer_evidence_status"] = "eligible"
    record["perfminer_exclusion_reason"] = ""
    return record


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_figshare_model(model_dir: str | Path) -> str:
    root = Path(model_dir)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = set(PERFMINER_MODEL_FILE_SHA256)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ValueError(
            "PerfAnnotator model bundle file set does not match Figshare "
            f"(missing={missing}, unexpected={unexpected})."
        )
    for filename, expected in PERFMINER_MODEL_FILE_SHA256.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Figshare PerfAnnotator file not found: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"PerfAnnotator {filename} does not match the EASE 2026 Figshare artifact: "
                f"{actual}"
            )
    return PERFMINER_MODEL_WEIGHTS_SHA256


def pair_token_metadata(
    tokenizer: Any,
    commit_message: str,
    code_diff: str,
    *,
    max_length: int = PERFMINER_MAX_LENGTH,
) -> dict[str, Any]:
    full = tokenizer(
        commit_message,
        code_diff,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    message = tokenizer(
        commit_message,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_tokens = len(full["input_ids"])
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    message_budget = len(message["input_ids"]) + special_tokens
    return {
        "input_tokens": input_tokens,
        "context_fits": input_tokens <= max_length,
        # only_second cannot truncate a non-empty second sequence down to zero tokens.
        "message_fits": input_tokens <= max_length or message_budget < max_length,
    }


def classify_loaded_pairs(
    commit_messages: list[str],
    code_diffs: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    threshold: float = PERFMINER_THRESHOLD,
) -> tuple[list[int], list[float]]:
    import torch

    if len(commit_messages) != len(code_diffs):
        raise ValueError("PerfMiner message and diff counts must match.")
    validate_positive_label_semantics(model)
    labels: list[int] = []
    scores: list[float] = []
    indexed_pairs: Iterable[tuple[str, str]] = zip(commit_messages, code_diffs, strict=True)
    for batch in batched(list(indexed_pairs), batch_size):
        messages = [message for message, _ in batch]
        diffs = [diff for _, diff in batch]
        inputs = tokenizer(
            messages,
            diffs,
            return_tensors="pt",
            truncation=PERFMINER_TRUNCATION,
            max_length=PERFMINER_MAX_LENGTH,
            padding=True,
            return_token_type_ids=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        batch_scores = probabilities[:, 1].detach().cpu().tolist()
        scores.extend(float(value) for value in batch_scores)
        labels.extend(1 if float(value) >= threshold else 0 for value in batch_scores)
    return labels, scores
