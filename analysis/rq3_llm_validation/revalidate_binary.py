"""Revalidate saved binary responses locally, retaining original judgments and history."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from analysis.rq3_llm_validation import binary, binary_validation, run_binary
from analysis.rq3_llm_validation import experiment as v1


def revalidate(output: Path, providers: list[str], apply: bool = False) -> dict:
    requests = binary.load_requests(output)
    report = {"validation_version": binary_validation.VERSION, "apply": apply,
              "new_model_calls": 0, "records": []}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = output / "validation" / stamp
    audit_dir.mkdir(parents=True)
    for source in (Path(__file__), Path(binary.__file__), Path(binary_validation.__file__)):
        shutil.copy2(source, audit_dir / source.name)
    corrections = output / "validation_corrections.json"
    if corrections.exists():
        shutil.copy2(corrections, audit_dir / corrections.name)
        report["corrections_sha256"] = v1.sha256_file(corrections)
    for provider in providers:
        provider_dir = output / provider
        state_path = provider_dir / "run_state.json"
        previous_state = json.loads(state_path.read_text()) if state_path.exists() else {}
        lock_name = "runner.lock" if provider == "qwen" else "batch.lock"
        with (provider_dir / lock_name).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            updates = []
            for request in requests:
                path = binary.checkpoint_path(output, provider, request)
                if not path.exists():
                    continue
                checkpoint = json.loads(path.read_text())
                for name in ("custom_id", "prompt_sha256", "study_contract_sha256"):
                    if checkpoint[name] != request[name]:
                        raise ValueError("Saved checkpoint differs from the frozen request.")
                if checkpoint["model"] != binary.v2.MODELS[provider]:
                    raise ValueError("Saved checkpoint differs from the requested model.")
                raw = provider_dir / "raw" / f"{request['repo_id']}-{request['number']}-attempt-{checkpoint['attempts']}.json"
                row = {"provider": provider, "custom_id": request["custom_id"],
                       "previous_status": checkpoint["classification_status"],
                       "checkpoint_sha256": v1.sha256_file(path), "raw_sha256": v1.sha256_file(raw)}
                try:
                    payload, usage = run_binary.parse_response(provider, json.loads(raw.read_text()))
                    validated = binary.validate_result(payload, request)
                    original = binary.ForcedTradeoffResult.model_validate(payload).model_dump()
                    # Formatting repair may change quote spelling only. The model's
                    # label, directions, IDs, occurrence judgments and rationale stay fixed.
                    for name in binary.ForcedTradeoffResult.model_fields:
                        if name != "evidence_quotes" and original[name] != validated[name]:
                            raise ValueError(f"Revalidation changed model judgment: {name}")
                except ValueError as error:
                    if checkpoint["classification_status"] == "classified":
                        raise ValueError(f"Previously accepted result no longer validates: {request['custom_id']}") from error
                    row.update(status="error", error=str(error))
                    updates.append((path, {**checkpoint, "error": str(error),
                                           "revalidated_at": datetime.now(timezone.utc).isoformat(),
                                           "validation_version": binary_validation.VERSION,
                                           "revalidation_source_raw_sha256": row["raw_sha256"]}))
                else:
                    row.update(status="classified", label=validated["label"],
                               aligned_quotes=len(validated.get("quote_alignments", [])),
                               dimension_corrections=len(validated.get("audited_dimension_corrections", [])))
                    updated = {**checkpoint, "classification_status": "classified", "label": validated,
                               "usage": usage, "error": None, "revalidated_at": datetime.now(timezone.utc).isoformat(),
                               "validation_version": binary_validation.VERSION,
                               "validation_corrections_sha256": request.get("validation_corrections_sha256"),
                               "revalidation_source_raw_sha256": row["raw_sha256"]}
                    updates.append((path, updated))
                report["records"].append(row)
            if apply:
                for path, updated in updates:
                    backup = audit_dir / "before" / provider / path.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                    v1.atomic_write_json(path, updated)
                state = binary.collect(output, provider, requests)
                if state["errors"] and previous_state.get("status") == "failed":
                    state.update(status="failed", error=f"Revalidation left {state['errors']} invalid responses.")
                    v1.atomic_write_json(state_path, state)
    report["recovered"] = sum(r["previous_status"] == "error" and r["status"] == "classified" for r in report["records"])
    report["remaining_errors"] = [r for r in report["records"] if r["status"] == "error"]
    report["audit_dir"] = str(audit_dir)
    v1.atomic_write_json(audit_dir / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=binary.DEFAULT_OUTPUT)
    parser.add_argument("--providers", choices=binary.PROVIDERS, nargs="+", default=["openai", "gemini"])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = revalidate(args.output_dir, args.providers, args.apply)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
