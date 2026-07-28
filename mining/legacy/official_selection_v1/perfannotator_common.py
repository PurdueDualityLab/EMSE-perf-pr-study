from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import random
import shutil
from typing import Any
from zipfile import ZipFile

import pandas as pd


MODEL_ID = "annon-123/PerfAnnotator-mini"


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure deterministic inference before loading the classifier."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        return as_list(value.tolist())
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return [text] if text else []


def joined_lines(value: object, limit: int | None = None) -> str:
    values = as_list(value)
    if limit is not None:
        values = values[:limit]
    return "\n".join(values)


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def metadata_text(row: pd.Series, max_chars: int) -> tuple[str, bool]:
    parts = [
        "Title:\n" + str(row.get("title") or ""),
        "Body:\n" + str(row.get("body") or ""),
        "Commit messages:\n" + joined_lines(row.get("commit_messages")),
        "Files:\n" + joined_lines(row.get("filenames")),
    ]
    return truncate_text("\n\n".join(parts), max_chars)


def diff_text(row: pd.Series, max_chars: int) -> tuple[str, bool]:
    filenames = as_list(row.get("diff_filenames")) or as_list(row.get("filenames"))
    patches = as_list(row.get("diff_patches"))
    diff_parts = []
    for index, patch in enumerate(patches):
        filename = filenames[index] if index < len(filenames) else f"file_{index + 1}"
        diff_parts.append(f"File: {filename}\n{patch}")
    parts = [
        "Commit messages:\n" + joined_lines(row.get("commit_messages_enriched") or row.get("commit_messages")),
        "Diff:\n" + "\n\n".join(diff_parts),
    ]
    return truncate_text("\n\n".join(parts), max_chars)


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
        zip_file.extractall(temporary)
    if not (temporary / "config.json").is_file():
        raise ValueError(f"PerfAnnotator archive does not contain config.json: {archive}")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)
    return str(target)


def load_model(model_id: str, revision: str | None, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = resolve_model_path(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
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

    labels: list[int] = []
    scores: list[float] = []
    for batch in batched(texts, batch_size):
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        batch_labels = logits.argmax(dim=-1).detach().cpu().tolist()
        batch_scores = probabilities[:, 1].detach().cpu().tolist()
        labels.extend(int(value) for value in batch_labels)
        scores.extend(float(value) for value in batch_scores)
    return labels, scores
