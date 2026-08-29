"""Prepare, submit, inspect, and collect the RQ1 OpenAI batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

ENDPOINT = "/v1/responses"
PROMPT_VERSION = "rq1-optimization-pattern-legacy"
DEFAULT_MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 4_096
PATCH_CHARACTER_LIMIT = 15_000
SYSTEM_INSTRUCTION = "You are an expert software engineer specializing in performance optimization analysis. Analyze code changes and classify optimization patterns accurately."
TRUNCATION_MARKER = "\n\n... [patch truncated for length] ..."
TAXONOMY_COLUMNS = (
    "High-level Pattern",
    "Sub pattern",
    "Description",
    "Example",
    "Optimized Metrics",
    "Detection",
)


class PatternLabel(BaseModel):
    explanation: str
    optimization_comparison: str
    high_level_pattern: str
    sub_pattern: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def semantic_schema() -> dict[str, Any]:
    schema = PatternLabel.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def load_taxonomy(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(TAXONOMY_COLUMNS)
    if not required.issubset(frame.columns):
        raise ValueError("Optimization catalog is missing required columns.")
    frame = frame.loc[:, TAXONOMY_COLUMNS]
    for column in ("High-level Pattern", "Sub pattern"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().ne(frame[column]).any():
            raise ValueError(f"Optimization catalog has invalid whitespace or missing {column}.")
    if frame.duplicated(["High-level Pattern", "Sub pattern"]).any():
        raise ValueError("Optimization catalog contains duplicate pattern identities.")
    return frame


def taxonomy_labels(taxonomy: pd.DataFrame) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for row in taxonomy.fillna("").to_dict("records"):
        high = str(row["High-level Pattern"]).strip()
        sub = str(row["Sub pattern"]).strip()
        if high and sub:
            labels.setdefault(high, []).append(sub)
    return labels


def format_taxonomy(taxonomy: pd.DataFrame) -> str:
    text = "### Optimization Patterns Taxonomy:\n\n"
    for high_level in taxonomy["High-level Pattern"].drop_duplicates():
        text += f"- **{high_level}**\n"
        for row in taxonomy[taxonomy["High-level Pattern"] == high_level].to_dict("records"):
            text += f"    - {row['Sub pattern']}\n"
            for column, label in (
                ("Description", "Description"),
                ("Example", "Example"),
                ("Optimized Metrics", "Metrics"),
                ("Detection", "Detection"),
            ):
                if pd.notna(row[column]):
                    text += f"        - {label}: {row[column]}\n"
    return text


def build_input(sample_path: Path, evidence_dir: Path) -> pd.DataFrame:
    sample = pd.read_parquet(sample_path)
    files = pd.read_parquet(evidence_dir / "pull_request_files.parquet")
    status = pd.read_parquet(evidence_dir / "collection_status.parquet")
    if sample.empty or sample.duplicated(["repo_id", "number"]).any():
        raise ValueError("RQ1 requires a nonempty sample with unique PR identities.")
    arm_counts = sample["sample_arm"].value_counts().to_dict()
    if set(arm_counts) != {"agentic", "human_candidate"} or len(set(arm_counts.values())) != 1:
        raise ValueError("RQ1 requires the final 1:1 balanced sample.")
    sample_ids = set(map(tuple, sample[["repo_id", "number"]].to_numpy()))
    status_ids = set(map(tuple, status[["repo_id", "number"]].to_numpy()))
    if status.duplicated(["repo_id", "number"]).any() or status_ids != sample_ids:
        raise ValueError("RQ1 evidence status does not exactly match the official sample.")
    if not status["status"].isin(["complete", "not_found"]).all():
        raise ValueError("RQ1 evidence contains partial or identity-mismatched rows.")
    patches = (
        files.assign(patch=files["patch"].fillna(""))
        .sort_values(["repo_id", "number", "file_index"], kind="mergesort")
        .groupby(["repo_id", "number"], as_index=False)
        .agg(
            patch=("patch", lambda values: "\n\n".join(value for value in values if value)),
            files_observed=("filename", "count"),
            patches_available=("patch_available", "sum"),
        )
    )
    result = sample.merge(patches, on=["repo_id", "number"], how="left", validate="one_to_one")
    result = result.merge(
        status[["repo_id", "number", "status"]].rename(columns={"status": "evidence_status"}),
        on=["repo_id", "number"],
        how="left",
        validate="one_to_one",
    )
    result["patch"] = result["patch"].fillna("")
    result["files_observed"] = result["files_observed"].fillna(0).astype(int)
    result["patches_available"] = result["patches_available"].fillna(0).astype(int)
    return result


def prompt_for(row: dict[str, Any], taxonomy: pd.DataFrame) -> str:
    context_parts = []
    title = row.get("title")
    body = row.get("body")
    patch = row.get("patch")
    if pd.notna(title) and str(title).strip():
        context_parts.append(f"**Title**: {title}")
    if pd.notna(body) and str(body).strip():
        context_parts.append(f"**Description**: {body}")
    if pd.notna(patch) and str(patch).strip():
        patch_text = str(patch)
        if len(patch_text) > PATCH_CHARACTER_LIMIT:
            patch_text = patch_text[:PATCH_CHARACTER_LIMIT] + TRUNCATION_MARKER
        context_parts.append(f"**Code Changes (Patch)**:\n```diff\n{patch_text}\n```")
    context = "\n\n".join(context_parts)
    return f"""I have a performance optimization commit with the following information. Please analyze with the following goals:

    1. **Code Function Explanation**: Briefly explain what the code is doing—what problem it solves and how it works.

    2. **Optimization Comparison**: Compare the original and optimized versions to identify:
    - **Algorithmic changes**: Any differences in logic, algorithm design, or problem-solving approach.
    - **Performance improvements**: Enhancements related to time complexity, space efficiency, or runtime behavior.
    - **Redundant code removal**: Elimination of unnecessary logic, method calls, or control structures.
    - **Other noteworthy changes**: Any structural or stylistic differences that could impact performance or readability.

    3. **Optimization Pattern Classification**:
    Based on the overall nature of the optimized code, assign the following:
    - **Exactly one high-level optimization pattern** from the list below
    - **One most representative sub-pattern** within that high-level category

    {format_taxonomy(taxonomy)}

    Here are the info:

    {context}

    **Output Structure**:
    Please respond in JSON format with the following structure:
    {{
    "explanation": "Brief description of what the code is doing",
    "optimization_comparison": "Detailed comparison highlighting specific optimizations",
    "high_level_pattern": "Single most representative high-level optimization pattern",
    "sub_pattern": "Most representative sub-pattern within high_level_pattern"
    }}

    Ensure your response is valid JSON that can be parsed.
    """


def response_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "PatternLabel",
            "schema": semantic_schema(),
            "strict": True,
        }
    }


def study_contract(
    sample_sha256: str,
    catalog_sha256: str,
    evidence_files_sha256: str,
    evidence_status_sha256: str,
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "study": "rq1_optimization_pattern_classification",
        "prompt_version": PROMPT_VERSION,
        "system_instruction_sha256": hashlib.sha256(
            SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "semantic_schema_sha256": sha256_json(semantic_schema()),
        "sample_sha256": sample_sha256,
        "catalog_sha256": catalog_sha256,
        "evidence_files_sha256": evidence_files_sha256,
        "evidence_status_sha256": evidence_status_sha256,
        "input_construction": {
            "identity_columns": ["repo_id", "number"],
            "patch_order": ["repo_id", "number", "file_index"],
            "patch_separator": "\n\n",
            "patch_character_limit": PATCH_CHARACTER_LIMIT,
            "truncation_marker": TRUNCATION_MARKER,
        },
    }


def openai_provider_config(model: str) -> dict[str, Any]:
    return {
        "provider_config_version": 1,
        "provider": "openai",
        "model": model,
        "endpoint": ENDPOINT,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_wrapper_sha256": sha256_json(response_format()),
    }


def input_row_sha256(row: dict[str, Any]) -> str:
    patch = str(row.get("patch") or "")
    patch_used = patch[:PATCH_CHARACTER_LIMIT]
    if len(patch) > PATCH_CHARACTER_LIMIT:
        patch_used += TRUNCATION_MARKER
    return sha256_json(
        {
            "repo_id": int(row["repo_id"]),
            "number": int(row["number"]),
            "title": "" if pd.isna(row.get("title")) else str(row.get("title")),
            "body": "" if pd.isna(row.get("body")) else str(row.get("body")),
            "patch_used": patch_used,
        }
    )


def prepare_batch(
    sample_path: Path,
    evidence_dir: Path,
    catalog_path: Path,
    output_dir: Path,
    model: str,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite a run.")
    taxonomy = load_taxonomy(catalog_path)
    frame = build_input(sample_path, evidence_dir).sort_values(
        ["repo_id", "number"], kind="mergesort"
    )
    if frame.empty or frame.duplicated(["repo_id", "number"]).any():
        raise ValueError("Batch input must contain unique PR identities.")
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_snapshot = output_dir / "optimization_catalog.csv"
    shutil.copyfile(catalog_path, catalog_snapshot)
    common_contract = study_contract(
        sha256_file(sample_path),
        sha256_file(catalog_snapshot),
        sha256_file(evidence_dir / "pull_request_files.parquet"),
        sha256_file(evidence_dir / "collection_status.parquet"),
    )
    study_contract_sha256 = sha256_json(common_contract)
    provider_config = openai_provider_config(model)
    provider_config_sha256 = sha256_json(provider_config)
    payload_path = output_dir / "batch_input.jsonl"
    manifest_rows = []
    with payload_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            custom_id = f"{int(row['repo_id'])}:{int(row['number'])}"
            prompt = prompt_for(row, taxonomy)
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": ENDPOINT,
                "body": {
                    "model": model,
                    "input": [
                        {"role": "developer", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt},
                    ],
                    "reasoning": {"effort": REASONING_EFFORT},
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "text": response_format(),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=True) + "\n")
            manifest_rows.append(
                {
                    "repo_id": int(row["repo_id"]),
                    "number": int(row["number"]),
                    "custom_id": custom_id,
                    "repo_full_name": row["repo_full_name"],
                    "html_url": row["html_url"],
                    "sample_arm": row["sample_arm"],
                    "prompt_version": PROMPT_VERSION,
                    "evidence_status": row["evidence_status"],
                    "files_observed": int(row["files_observed"]),
                    "patches_available": int(row["patches_available"]),
                    "prompt_chars": len(prompt),
                    "patch_chars_available": len(str(row.get("patch") or "")),
                    "patch_chars_used": min(
                        len(str(row.get("patch") or "")), PATCH_CHARACTER_LIMIT
                    ),
                    "patch_truncated": len(str(row.get("patch") or ""))
                    > PATCH_CHARACTER_LIMIT,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "input_row_sha256": input_row_sha256(row),
                    "study_contract_sha256": study_contract_sha256,
                    "provider_config_sha256": provider_config_sha256,
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "batch_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    metadata = {
        "model": model,
        "endpoint": ENDPOINT,
        "prompt_version": PROMPT_VERSION,
        "requests": len(manifest),
        "sample_sha256": sha256_file(sample_path),
        "catalog_file": catalog_snapshot.name,
        "catalog_sha256": sha256_file(catalog_snapshot),
        "evidence_files_sha256": sha256_file(evidence_dir / "pull_request_files.parquet"),
        "evidence_status_sha256": sha256_file(evidence_dir / "collection_status.parquet"),
        "schema_sha256": sha256_json(response_format()),
        "system_instruction_sha256": hashlib.sha256(
            SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "generation_config": {
            "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "batch_input_sha256": sha256_file(payload_path),
        "batch_input_bytes": payload_path.stat().st_size,
        "manifest_sha256": sha256_file(manifest_path),
    }
    metadata["study_contract"] = common_contract
    metadata["study_contract_sha256"] = study_contract_sha256
    metadata["provider_config"] = provider_config
    metadata["provider_config_sha256"] = provider_config_sha256
    (output_dir / "prepare_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload_path


def submit_batch(output_dir: Path) -> dict[str, Any]:
    payload_path = output_dir / "batch_input.jsonl"
    metadata_path = output_dir / "prepare_metadata.json"
    state_path = output_dir / "batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("batch_id"):
            raise FileExistsError("Batch was already submitted.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(payload_path) != metadata["batch_input_sha256"]:
        raise ValueError("Batch input changed after preparation.")
    if sha256_file(output_dir / metadata["catalog_file"]) != metadata["catalog_sha256"]:
        raise ValueError("Catalog snapshot changed after preparation.")
    if sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    else:
        state = {**metadata, "local_status": "prepared"}
        atomic_write_json(state_path, state)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not state.get("input_file_id"):
        with payload_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        state.update(input_file_id=uploaded.id, local_status="uploaded")
        atomic_write_json(state_path, state)
    batch = client.batches.create(
        input_file_id=state["input_file_id"],
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={"description": "RQ1 GPT-5.6-sol optimization-pattern classification"},
    )
    state.update(batch_id=batch.id, status=batch.status, local_status="submitted",
                 created_at=batch.created_at, expires_at=batch.expires_at)
    atomic_write_json(state_path, state)
    return state


def batch_status(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "batch_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).batches.retrieve(state["batch_id"])
    status = {
        "batch_id": batch.id,
        "status": batch.status,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "completed_at": batch.completed_at,
        "expires_at": batch.expires_at,
    }
    state.update(status)
    atomic_write_json(state_path, state)
    return status


def _output_text(body: dict[str, Any]) -> str:
    if body.get("status") == "incomplete" or body.get("incomplete_details"):
        raise ValueError("Batch response is incomplete.")
    texts = [str(content.get("text") or "") for output in body.get("output", [])
             for content in output.get("content", []) if content.get("type") == "output_text"]
    if not texts:
        raise ValueError("Batch response has no output_text content.")
    return "".join(texts)


def write_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    models = sorted(frame["model"].dropna().unique().tolist())
    if len(models) != 1:
        raise ValueError(f"Expected one RQ1 model, found: {models}")
    summary = {
        "method": PROMPT_VERSION,
        "model": models[0],
        "rows": len(frame),
        "status_counts": frame["classification_status"].value_counts().to_dict(),
        "input_tokens": int(frame.get("input_tokens", pd.Series(dtype=float)).sum()),
        "output_tokens": int(frame.get("output_tokens", pd.Series(dtype=float)).sum()),
        "reasoning_tokens": int(frame.get("reasoning_tokens", pd.Series(dtype=float)).sum()),
        "cached_input_tokens": int(frame.get("cached_input_tokens", pd.Series(dtype=float)).sum()),
        "total_tokens": int(frame.get("total_tokens", pd.Series(dtype=float)).sum()),
        "arm_counts": frame["sample_arm"].value_counts().to_dict(),
        "pattern_counts": frame.get(
            "high_level_pattern", pd.Series(dtype=str)
        ).value_counts().to_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def collect_batch(output_dir: Path) -> pd.DataFrame:
    status = batch_status(output_dir)
    if status["status"] != "completed" or not status["output_file_id"]:
        raise RuntimeError(f"Batch is not ready for collection: {status['status']}")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    output_text = client.files.content(status["output_file_id"]).text
    raw_path = output_dir / "batch_output.jsonl"
    raw_path.write_text(output_text, encoding="utf-8")
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text(encoding="utf-8"))
    catalog_path = output_dir / metadata["catalog_file"]
    if sha256_file(catalog_path) != metadata["catalog_sha256"]:
        raise ValueError("Catalog snapshot does not match prepared hash.")
    manifest_path = output_dir / "batch_manifest.parquet"
    if sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest does not match prepared hash.")
    manifest = pd.read_parquet(manifest_path)
    if manifest["custom_id"].isna().any() or not manifest["custom_id"].is_unique:
        raise ValueError("Batch manifest contains invalid or duplicate custom IDs.")
    manifest_by_id = manifest.set_index("custom_id").to_dict("index")
    taxonomy = taxonomy_labels(load_taxonomy(catalog_path))
    valid_high = set(taxonomy)
    labels = []
    seen = set()
    for line in output_text.splitlines():
        result = json.loads(line)
        custom_id = result["custom_id"]
        if custom_id in seen or custom_id not in manifest_by_id:
            raise ValueError(f"Invalid or duplicate batch custom_id: {custom_id}")
        seen.add(custom_id)
        response = result.get("response") or {}
        body = response.get("body") or {}
        row = {
            **manifest_by_id[custom_id],
            "custom_id": custom_id,
            "model": body.get("model") or metadata["model"],
            "response_id": body.get("id"),
        }
        if response.get("status_code") != 200:
            row.update({"classification_status": "error", "error": json.dumps(result.get("error") or response)[:500]})
        else:
            usage = body.get("usage") or {}
            row["input_tokens"] = usage.get("input_tokens")
            row["output_tokens"] = usage.get("output_tokens")
            row["total_tokens"] = usage.get("total_tokens")
            row["cached_input_tokens"] = (usage.get("input_tokens_details") or {}).get("cached_tokens")
            row["reasoning_tokens"] = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            try:
                label = PatternLabel.model_validate_json(_output_text(body))
                if label.high_level_pattern not in valid_high:
                    raise ValueError(f"Unknown high-level pattern: {label.high_level_pattern}")
                if label.sub_pattern not in taxonomy[label.high_level_pattern]:
                    raise ValueError(
                        f"Invalid sub-pattern for {label.high_level_pattern}: {label.sub_pattern}"
                    )
            except ValueError as error:
                row.update(
                    {
                        "classification_status": "error",
                        "error": f"Invalid structured response: {error}"[:500],
                    }
                )
            else:
                row.update(label.model_dump())
                row["classification_status"] = "classified"
        labels.append(row)
    for custom_id in sorted(set(manifest_by_id) - seen):
        labels.append(
            {
                **manifest_by_id[custom_id],
                "custom_id": custom_id,
                "model": metadata["model"],
                "classification_status": "error",
                "error": "Missing from batch output",
            }
        )
    frame = pd.DataFrame(labels).sort_values(["repo_id", "number"]).reset_index(drop=True)
    if len(frame) != len(manifest) or frame["custom_id"].duplicated().any():
        raise ValueError("Collected rows do not exactly preserve the batch manifest.")
    frame.to_parquet(output_dir / "optimization_pattern_labels.parquet", index=False)
    write_summary(frame, output_dir)
    return frame


def prepare_retry(source_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Retry output directory is not empty.")
    labels = pd.read_parquet(source_dir / "optimization_pattern_labels.parquet")
    errors = labels[labels["classification_status"].eq("error")]
    if errors.empty:
        raise ValueError("Source run has no errors to retry.")
    metadata = json.loads((source_dir / "prepare_metadata.json").read_text(encoding="utf-8"))
    if sha256_file(source_dir / "batch_input.jsonl") != metadata["batch_input_sha256"]:
        raise ValueError("Source batch input changed after preparation.")
    if sha256_file(source_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Source batch manifest changed after preparation.")
    requests = {}
    for line in (source_dir / "batch_input.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        requests[item["custom_id"]] = item
    output_dir.mkdir(parents=True)
    payload_path = output_dir / "batch_input.jsonl"
    retry_ids = errors["custom_id"].tolist()
    with payload_path.open("w", encoding="utf-8") as handle:
        for custom_id in retry_ids:
            handle.write(json.dumps(requests[custom_id], ensure_ascii=True) + "\n")
    manifest_path = output_dir / "batch_manifest.parquet"
    pd.read_parquet(source_dir / "batch_manifest.parquet").query(
        "custom_id in @retry_ids"
    ).to_parquet(manifest_path, index=False)
    catalog_snapshot = output_dir / metadata["catalog_file"]
    shutil.copyfile(source_dir / metadata["catalog_file"], catalog_snapshot)
    retry_metadata = {
        **metadata,
        "requests": len(retry_ids),
        "retry_of": str(source_dir),
        "batch_input_sha256": sha256_file(payload_path),
        "batch_input_bytes": payload_path.stat().st_size,
        "manifest_sha256": sha256_file(manifest_path),
    }
    for key in (
        "input_file_id",
        "batch_id",
        "status",
        "created_at",
        "expires_at",
        "output_file_id",
        "error_file_id",
        "completed_at",
    ):
        retry_metadata.pop(key, None)
    (output_dir / "prepare_metadata.json").write_text(
        json.dumps(retry_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload_path


def merge_retry(source_dir: Path, retry_dir: Path, output_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(source_dir / "optimization_pattern_labels.parquet")
    retry = pd.read_parquet(retry_dir / "optimization_pattern_labels.parquet")
    source_errors = source[source["classification_status"].eq("error")]
    if set(retry["custom_id"]) != set(source_errors["custom_id"]):
        raise ValueError("Retry identities do not exactly match source errors.")
    hash_columns = (
        "study_contract_sha256",
        "provider_config_sha256",
        "input_row_sha256",
        "prompt_sha256",
    )
    source_hashes = source_errors.set_index("custom_id")[list(hash_columns)].sort_index()
    retry_hashes = retry.set_index("custom_id")[list(hash_columns)].sort_index()
    if not source_hashes.equals(retry_hashes):
        raise ValueError("Retry hashes do not match the source errors.")
    merged = pd.concat(
        [source[~source["custom_id"].isin(retry["custom_id"])], retry], ignore_index=True
    ).sort_values(["repo_id", "number"])
    if len(merged) != len(source) or merged.duplicated(["repo_id", "number"]).any():
        raise ValueError("Merged retry does not preserve source population.")
    if merged["classification_status"].eq("error").any():
        raise ValueError("Retry still contains errors; prepare another retry before finalizing.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return merged.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--evidence-dir", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    for action in ("submit", "status", "collect"):
        command = subparsers.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
    retry = subparsers.add_parser("prepare-retry")
    retry.add_argument("--source-dir", type=Path, required=True)
    retry.add_argument("--output-dir", type=Path, required=True)
    merge = subparsers.add_parser("merge-retry")
    merge.add_argument("--source-dir", type=Path, required=True)
    merge.add_argument("--retry-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        result = prepare_batch(args.sample, args.evidence_dir, args.catalog, args.output_dir, args.model)
        print(result)
    elif args.action == "submit":
        print(json.dumps(submit_batch(args.output_dir), sort_keys=True))
    elif args.action == "status":
        print(json.dumps(batch_status(args.output_dir), sort_keys=True))
    elif args.action == "collect":
        print(f"Collected {len(collect_batch(args.output_dir))} labels")
    elif args.action == "prepare-retry":
        print(prepare_retry(args.source_dir, args.output_dir))
    else:
        print(f"Merged {len(merge_retry(args.source_dir, args.retry_dir, args.output))} labels")


if __name__ == "__main__":
    main()
