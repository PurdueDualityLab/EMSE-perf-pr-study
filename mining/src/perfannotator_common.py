from __future__ import annotations

from collections.abc import Iterable
import hashlib
import os
from pathlib import Path
import random
import shutil
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd


MODEL_ID = "annon-123/PerfAnnotator-mini"
MODEL_REVISION = "7d7ba362c257c3ea8c69d52b4a736ae4f182e68c"
MODEL_ARTIFACT_SHA256 = "e2ed2bd495eb7ccea90da33037d2d444be344183cc8a747949a660248541531f"
POSITIVE_LABEL_ID = 1


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure deterministic inference before loading the classifier."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def as_list(value: object) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [text for item in value for text in as_list(item)]
        return sorted(values) if isinstance(value, (set, frozenset)) else values
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return as_list(converted)
    text = str(value).strip()
    return [text] if text else []


def joined_lines(value: object, limit: int | None = None) -> str:
    values = as_list(value)
    if limit is not None:
        values = values[:limit]
    return "\n".join(values)


def scalar_text(value: object) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, str):
        return value
    return "\n".join(as_list(value))


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def metadata_text(row: pd.Series, max_chars: int) -> tuple[str, bool]:
    parts = [
        "Title:\n" + scalar_text(row.get("title")),
        "Body:\n" + scalar_text(row.get("body")),
        "Commit messages:\n" + joined_lines(row.get("commit_messages")),
        "Files:\n" + joined_lines(row.get("filenames")),
    ]
    return truncate_text("\n\n".join(parts), max_chars)


def diff_text(row: pd.Series, max_chars: int) -> tuple[str, bool]:
    filenames = as_list(row.get("diff_filenames"))
    if not filenames:
        filenames = as_list(row.get("filenames"))
    patches = as_list(row.get("diff_patches"))
    diff_parts = []
    for index, patch in enumerate(patches):
        filename = filenames[index] if index < len(filenames) else f"file_{index + 1}"
        diff_parts.append(f"File: {filename}\n{patch}")
    commit_messages = as_list(row.get("commit_messages_enriched"))
    if not commit_messages:
        commit_messages = as_list(row.get("commit_messages"))
    parts = ["Commit messages:\n" + "\n".join(commit_messages), "Diff:\n" + "\n\n".join(diff_parts)]
    return truncate_text("\n\n".join(parts), max_chars)


def _normalized_label_name(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _is_positive_label_name(value: object) -> bool:
    name = _normalized_label_name(value)
    if name in {"positive", "true", "yes"}:
        return True
    if any(marker in name for marker in ("negative", "nonperformance", "notperformance", "noperformance")):
        return False
    return name == "performance" or (
        ("performance" in name or name.startswith("perf")) and "improv" in name
    )


def validate_positive_label_semantics(model: Any) -> None:
    """Reject model metadata that contradicts the study's explicit label-1 convention."""
    config = getattr(model, "config", None)
    try:
        num_labels = int(getattr(config, "num_labels", 0))
    except (TypeError, ValueError):
        num_labels = 0
    if config is None or num_labels != 2:
        raise ValueError("PerfAnnotator must be a binary classifier with exactly two labels.")

    raw_id2label = getattr(config, "id2label", None)
    if not isinstance(raw_id2label, dict):
        raise ValueError("PerfAnnotator config must define id2label semantics.")
    try:
        id2label = {int(label_id): name for label_id, name in raw_id2label.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("PerfAnnotator config contains invalid label IDs.") from error
    if set(id2label) != {0, POSITIVE_LABEL_ID}:
        raise ValueError("PerfAnnotator config must define labels 0 and 1.")
    generic_names = {
        label_id: _normalized_label_name(name) in {str(label_id), f"label{label_id}"}
        for label_id, name in id2label.items()
    }
    if not all(generic_names.values()) and (
        not _is_positive_label_name(id2label[POSITIVE_LABEL_ID])
        or _is_positive_label_name(id2label[0])
    ):
        raise ValueError(
            "PerfAnnotator label 1 must represent performance-improving and label 0 must not."
        )

    raw_label2id = getattr(config, "label2id", None)
    if raw_label2id is not None and not isinstance(raw_label2id, dict):
        raise ValueError("PerfAnnotator config contains invalid label2id semantics.")
    if raw_label2id:
        label2id = {}
        for name, label_id in raw_label2id.items():
            try:
                parsed_id = int(label_id)
            except (TypeError, ValueError) as error:
                raise ValueError("PerfAnnotator config contains invalid label IDs.") from error
            label2id[_normalized_label_name(name)] = parsed_id
            if _is_positive_label_name(name) and parsed_id != POSITIVE_LABEL_ID:
                raise ValueError("PerfAnnotator label2id maps the positive label to an unexpected ID.")
        for label_id, name in id2label.items():
            if label2id.get(_normalized_label_name(name)) != label_id:
                raise ValueError("PerfAnnotator id2label and label2id mappings are inconsistent.")


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def resolve_model_path(model_id: str, revision: str | None) -> str:
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return str(local_path)

    from huggingface_hub import hf_hub_download

    archive = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="PerfAnnotator-mini.zip",
            revision=revision,
        )
    )
    cache_root = Path(os.environ.get("PERFANNOTATOR_MODEL_CACHE", "mining/query_cache/models"))
    revision_name = (revision or "main").replace("/", "-")
    target = cache_root / model_id.replace("/", "--") / revision_name
    if (target / "config.json").is_file():
        return str(target)

    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        root = temporary.resolve()
        for member in zip_file.infolist():
            destination = (temporary / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"PerfAnnotator archive contains an unsafe path: {member.filename}")
        zip_file.extractall(temporary)
    if not (temporary / "config.json").is_file():
        raise ValueError(f"PerfAnnotator archive does not contain config.json: {archive}")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)
    return str(target)


def model_artifact_sha256(path: str | Path) -> str:
    """Hash model bytes and relative filenames to identify the exact local artifact."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"PerfAnnotator model directory not found: {root}")
    files = sorted(file for file in root.rglob("*") if file.is_file())
    if not files:
        raise ValueError(f"PerfAnnotator model directory is empty: {root}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_model(model_id: str, revision: str | None, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = resolve_model_path(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    validate_positive_label_semantics(model)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify_texts(
    texts: list[str],
    *,
    model_id: str,
    revision: str | None,
    device: str,
    batch_size: int,
) -> tuple[list[int], list[float], str]:
    tokenizer, model, resolved_device = load_model(model_id, revision, device)
    labels, scores = classify_loaded_model(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
    )
    return labels, scores, resolved_device


def classify_loaded_model(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    import torch

    validate_positive_label_semantics(model)
    labels: list[int] = []
    scores: list[float] = []
    for batch in batched(texts, batch_size):
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        batch_labels = logits.argmax(dim=-1).detach().cpu().tolist()
        batch_scores = probabilities[:, POSITIVE_LABEL_ID].detach().cpu().tolist()
        labels.extend(int(value) for value in batch_labels)
        scores.extend(float(value) for value in batch_scores)
    return labels, scores
