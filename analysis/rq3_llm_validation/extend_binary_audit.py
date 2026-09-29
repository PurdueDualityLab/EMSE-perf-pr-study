"""Carry forward the immutable 27-case audit and append two evidence-reviewed cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from analysis.rq3_llm_validation import binary
from analysis.rq3_llm_validation import binary_validation


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    csv.field_size_limit(20_000_000)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def key(row):
    return f"{row['repo_id']}-{row['number']}"


def check_package(root):
    checksums = read_json(root / "checksums.json")
    for name, expected in checksums.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Changed or unsafe package input: {name}")
    return checksums


def preserve_copy(source, destination):
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f"Existing audit snapshot differs from its source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def source_context(record, start, end):
    text = record["text"]
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    right = len(text) if right < 0 else right
    return {"record_id": record["record_id"], "source": record["source"],
            "table": record.get("table", record["source"]), "locator": record["locator"],
            "start": left, "end": right, "excerpt": text[left:right]}


def excerpts(records, refs):
    output = []
    for rid, needle in refs:
        text = records[rid]["text"]
        start = text.find(needle)
        if start < 0:
            raise ValueError(f"Missing reviewed excerpt: {rid}: {needle}")
        output.append(source_context(records[rid], start, start + len(needle)))
    return output


def quote_locators(records, quotes):
    located = []
    for quote in quotes:
        found = None
        for record in records.values():
            index = record["text"].find(quote)
            if index >= 0:
                found = {"quote": quote, "status": "verbatim", "record_id": record["record_id"],
                         "locator": record["locator"], "start": index, "end": index + len(quote)}
                break
        if found is None:
            needle = binary_validation.rendered_text(quote)[0].strip()
            for record in records.values():
                normalized, positions = binary_validation.rendered_text(record["text"])
                index = normalized.find(needle)
                if needle and index >= 0:
                    start, end = positions[index][0], positions[index + len(needle) - 1][1]
                    found = {"quote": quote, "status": "formatting_aligned", "record_id": record["record_id"],
                             "locator": record["locator"], "start": start, "end": end,
                             "original_excerpt": record["text"][start:end]}
                    break
        if found is None:
            raise ValueError(f"Added-case quotation has no supplied-source match: {quote}")
        located.append(found)
    return located


def raw_decision(provider, response):
    if provider == "openai":
        text = "".join(c.get("text", "") for item in response["output"] for c in item.get("content", [])
                       if c.get("type") == "output_text")
    elif provider == "gemini":
        text = "".join(c.get("text", "") for c in response["candidates"][0]["content"]["parts"] if not c.get("thought"))
    else:
        text = response["message"]["content"]
    return json.loads(text)


def assess_activation(case, group, activation, records, judgment):
    dimension = group["dimension"]
    location = activation["quant_records"][0]
    record = records[location["record_id"]]
    context = source_context(record, location["start"], location["end"])
    association = all(loc["record_id"] == location["record_id"] for loc in activation["cue_records"])
    if case.startswith("46398090-"):
        oid = group["occurrence_id"]
        if oid in judgment["valid_quantity_occurrences"]:
            valid, reason, role = True, "reported_metric", "reported_extraction_duration"
            note = "Concrete PR-description startup output reports WAR extraction in 768ms. It is a duration, not a measured before/after gain."
        elif oid in judgment["unclear_quantity_occurrences"]:
            valid, reason, role = "unclear", "measurement_provenance_unclear", "unqualified_performance_claim"
            note = "An unqualified speedup or temporary-memory claim is repeated in documentation built around example timings and resource arithmetic. Its measurement provenance is unclear; it is not declared false solely because raw runs are absent."
        else:
            if oid not in judgment["invalid_quantity_occurrences"]:
                raise ValueError(f"Unreviewed DataHub occurrence: {oid}")
            valid = False
            if oid in {"o0011", "o0012"}:
                reason, role = "configuration_or_requirement", "configured_capacity"
                note = "500MB is a minimum available-RAM requirement or maximum WAR-size condition, not an observed outcome."
                if oid == "o0011":
                    association = False
                    note += " The SELF span also crosses documentation sections before reaching the requirement."
            else:
                reason, role = "illustrative_documentation", "example_timing"
                note = "Timing ranges are explicitly under Example Timing, or the repeated 2843ms line is in instructional logging examples. The distinct PR-description 768ms output is assessed separately."
    else:
        if len(activation["quant_records"]) != 1:
            raise ValueError("Unexpected multi-record quantity in the added rspack case.")
        valid, reason = True, "reported_metric"
        if dimension == "D1":
            prefix = record["text"][:location["start"]].rstrip()
            role = "dispersion" if prefix.endswith("±") else "reported_timing"
            note = "Actual Base/Current benchmark timing or its reported dispersion in the supplied PR-specific table; pair workload rows before interpreting direction."
        elif dimension == "D3":
            if "rss memory" not in context["excerpt"]:
                raise ValueError("Unexpected memory context in the added rspack case.")
            role, note = "baseline", "Actual Base RSS cell. The full same-workload row supplies Current and Change; the baseline alone does not establish a direction."
        elif dimension == "D6":
            role, reason = "artifact_size", "cross_record_cue_association"
            note = "The 984.2KB Rome bundle size is measured in r0003. Its Binary Size cue comes from r0004; retain the valid quantity and flag the cross-comment association."
        else:
            raise ValueError("Unexpected dimension in the added rspack case.")
    annotation = "unclear" if valid == "unclear" and association else bool(valid is True and association)
    return {"valid_quantity": valid, "actual_dimension": dimension, "reason": reason, "reasoning": note,
            "association_correct": association, "quantity_role": role,
            "original_dimension_correct": True, "annotation_valid": annotation}, context


def pairs_for(records):
    pairs = []
    for rid, record in records.items():
        rows, header, offset = {}, "", 0
        for line in record["text"].splitlines(keepends=True):
            if line.startswith("| Name"):
                header = line.strip()
            match = re.match(r"^\|\s*(.*?)\s+\+\s*(exec|stats|rss memory)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|", line)
            if match:
                workload, mode, base, current, change = [value.strip() for value in match.groups()]
                rows[(workload, mode)] = {"base": base, "current": current, "change": change,
                    "start": offset, "end": offset + len(line.rstrip("\r\n")), "source_row": line.rstrip("\r\n")}
            offset += len(line)
        for (workload, mode), timing in rows.items():
            if mode == "rss memory" or (workload, "rss memory") not in rows:
                continue
            memory = rows[(workload, "rss memory")]
            dt = float(re.search(r"[-+]?\d+(?:\.\d+)?", timing["change"]).group())
            dm = float(re.search(r"[-+]?\d+(?:\.\d+)?", memory["change"]).group())
            relation = "no_gain" if dt >= 0 else "tradeoff" if dm > 0 else "joint_improvement" if dm < 0 else "gain_memory_unchanged"
            pairs.append({"case": "476642602-13024", "record_id": rid, "record_locator": record["locator"],
                "comparison_header": header, "workload": workload, "timing_mode": mode,
                "time_base": timing["base"], "time_current": timing["current"], "time_delta": timing["change"],
                "rss_base": memory["base"], "rss_current": memory["current"], "rss_delta": memory["change"],
                "reported_directional_relationship": relation, "time_source_row": timing["source_row"],
                "memory_source_row": memory["source_row"], "time_start": timing["start"], "time_end": timing["end"],
                "memory_start": memory["start"], "memory_end": memory["end"],
                "limitation": "Reported signed changes, not statistical significance; each pair stays within a source record and workload."})
    return pairs


def provider_review(case, provider):
    if case.startswith("46398090-"):
        return {"inference_disclosed_correctly": "unclear", "protocol_compliance": "unclear",
                "critique": "The label follows the source's stated speedup/temporary-memory claims and the capacity thresholds are distinguished. However, inference=false treats the documentation comparison as measured without resolving its example/resource-arithmetic context. This is an evidence-interpretation boundary, not a syntactic contradiction or proof that the physical tradeoff is false."}
    if provider == "openai":
        return {"inference_disclosed_correctly": "yes", "protocol_compliance": "compliant",
                "critique": "Correctly identifies mixed directions, cites both a Three.js joint pair and memory increases elsewhere, and discloses inference for the overall forced label."}
    if provider == "gemini":
        return {"inference_disclosed_correctly": "unclear", "protocol_compliance": "unclear",
                "critique": "The Three.js joint relationship is directly supported, but the rationale's mostly-stable description and inference=false understate the competing same-workload tradeoffs. Local support and a uniquely measured whole-PR label are distinct."}
    return {"inference_disclosed_correctly": "unclear", "protocol_compliance": "unclear",
            "critique": "The cited Three.js pairs support a scoped joint improvement. The claim of 20 memory rows with 14 decreases is inaccurate: the table has 21 RSS rows and 17 negative changes. The overall dominance judgment is an interpretation of heterogeneous workloads; inference=false needs a clearer scope."}


def extend(prior, full):
    out = full / "audit_output"
    if (out / "pr_level_audit.csv").exists():
        raise FileExistsError("Refusing to overwrite a completed audit extension.")
    prior_checksums, full_checksums = check_package(prior), check_package(full)
    old_manifest, full_manifest = read_json(prior / "snapshot_manifest.json"), read_json(full / "snapshot_manifest.json")
    old_cases = {key(row): row for row in old_manifest["selected_cases"]}
    all_cases = {key(row): row for row in full_manifest["selected_cases"]}
    judgments = read_json(out / "addendum_judgments.json")
    added = set(all_cases) - set(old_cases)
    if len(old_cases) != 27 or len(all_cases) != 29 or added != set(judgments["cases"]):
        raise ValueError("Audit extension must preserve 27 cases and add exactly the two former exclusions.")
    for case, original in old_cases.items():
        current = all_cases[case]
        for field in ("prompt_sha256", "study_contract_sha256", "checkpoint_sha256"):
            if original[field] != current[field]:
                raise ValueError(f"Previously audited input changed: {case}/{field}")
        for name in ("records.json", "occurrences.json", "classification_prompt.txt"):
            if sha(prior / original["case_path"] / name) != sha(full / current["case_path"] / name):
                raise ValueError(f"Previously audited source evidence changed: {case}/{name}")
    old_audit = prior / "audit_output"
    archive = out / "prior_27"
    archive.mkdir(exist_ok=True)
    for path in old_audit.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            preserve_copy(path, archive / path.name)
    for path in (old_audit / "cases").glob("*.md"):
        preserve_copy(path, out / "cases" / path.name)
    shutil.copy2(Path(__file__), out / "extend_binary_audit.py")
    pr_rows, pr_fields = read_csv(old_audit / "pr_level_audit.csv")
    occ_rows, occ_fields = read_csv(old_audit / "occurrence_level_audit.csv")
    act_rows, act_fields = read_csv(old_audit / "activation_level_audit.csv")
    provider_rows, provider_fields = read_csv(old_audit / "provider_level_audit.csv")
    old_rows_by_file = {"pr_level_audit.csv": list(pr_rows), "occurrence_level_audit.csv": list(occ_rows),
                        "activation_level_audit.csv": list(act_rows), "provider_level_audit.csv": list(provider_rows)}
    templates, _ = read_csv(full / "pr_audit_template.csv")
    templates = {key(row): row for row in templates}
    occ_templates, _ = read_csv(full / "occurrence_audit_template.csv")
    occ_templates = {(key(row), row["occurrence_id"]): row for row in occ_templates}
    consensus, _ = read_csv(full / "results.csv")
    consensus = {key(row): row for row in consensus}
    pairs = []
    for case in sorted(added):
        folder = full / all_cases[case]["case_path"]
        records = {r["record_id"]: r for r in read_json(folder / "records.json")}
        groups = read_json(folder / "occurrences.json")
        votes = read_json(folder / "votes.json")
        judgment = judgments["cases"][case]
        evidence = excerpts(records, judgment["evidence_refs"])
        group_results = []
        case_occurrences = []
        for group in groups:
            evaluated, contexts = [], []
            for activation in group["activations"]:
                for part in ("cue", "quant"):
                    reconstructed = "\n".join(records[loc["record_id"]]["text"][loc["start"]:loc["end"]]
                                                for loc in activation[f"{part}_records"])
                    if reconstructed != activation[f"{part}_raw_text"]:
                        raise ValueError("Original activation locator does not resolve.")
                verdict, context = assess_activation(case, group, activation, records, judgment)
                evaluated.append(verdict)
                contexts.append(context)
                act_rows.append({"repo_id": str(all_cases[case]["repo_id"]), "number": str(all_cases[case]["number"]),
                    "occurrence_id": group["occurrence_id"], "activation_id": activation["activation_id"],
                    "original_dimension": group["dimension"], "validation_dimension": group["dimension"],
                    "source": group["source"], "rule_id": activation["rule_id"],
                    "raw_quantity": activation["quant_raw_text"], "raw_cue": activation["cue_raw_text"],
                    "quantity_locators": activation["quant_records"], "cue_locators": activation["cue_records"], **verdict})
            valid = True if any(v["valid_quantity"] is True for v in evaluated) else "unclear" if any(v["valid_quantity"] == "unclear" for v in evaluated) else False
            group_results.append(valid)
            row = {field: "" for field in occ_fields}
            row.update(occ_templates[(case, group["occurrence_id"])])
            row.update(review_status="completed", review_valid=str(valid).lower(), review_dimension=group["dimension"],
                original_dimension_correct="yes", reason=";".join(dict.fromkeys(v["reason"] for v in evaluated)),
                notes=" ".join(dict.fromkeys(v["reasoning"] for v in evaluated)),
                source_locators=[{k: c[k] for k in ("record_id", "table", "locator", "start", "end")} for c in contexts],
                activation_memberships=group["activations"], activation_judgments=evaluated,
                decisive_source_contexts=contexts, has_valid_original_annotation=any(v["annotation_valid"] is True for v in evaluated),
                has_invalid_activation=any(v["annotation_valid"] is False for v in evaluated),
                quantity_roles=sorted({v["quantity_role"] for v in evaluated}))
            for provider in binary.PROVIDERS:
                model_vote = next(v for v in votes[provider]["label"]["occurrences"] if v["id"] == group["occurrence_id"])
                row[provider + "_valid"], row[provider + "_reason"] = model_vote["valid"], model_vote["reason"]
            occ_rows.append(row)
            case_occurrences.append(row)
        detector = "true_positive" if True in group_results else "inconclusive" if "unclear" in group_results else "false_positive"
        if detector != judgment["detector_pr_verdict"]:
            raise ValueError("PR detection verdict does not follow reviewed occurrence coverage.")
        reviewed_providers = []
        for provider in binary.PROVIDERS:
            vote = votes[provider]
            decision = vote["label"]
            review = provider_review(case, provider)
            raw = read_json(folder / f"{provider}_raw_response.json")
            raw_label = raw_decision(provider, raw)
            changed_fields = [field for field in raw_label if raw_label[field] != decision.get(field)]
            if set(changed_fields) - {"evidence_quotes"}:
                raise ValueError("Substantive model output changed in an added case.")
            recorded = read_json(folder / f"{provider}_recorded_request.json")
            prompt = (recorded["body"]["input"][-1]["content"] if provider == "openai" else
                      recorded["request"]["contents"][0]["parts"][0]["text"] if provider == "gemini" else recorded["prompt"])
            core = (folder / "classification_prompt.txt").read_bytes().decode("utf-8")
            if not prompt.startswith(core) or hashlib.sha256(prompt.encode()).hexdigest() != vote["effective_prompt_sha256"]:
                raise ValueError("Added-case provider prompt differs from its checkpoint.")
            original_status = read_json(folder / f"{provider}_original_attempt_status.json")
            provider_row = {field: "" for field in provider_fields}
            provider_row.update(repo_id=str(all_cases[case]["repo_id"]), number=str(all_cases[case]["number"]),
                sample_arm=templates[case]["sample_arm"], provider=provider, model=vote["model"],
                model_label=decision["label"], choice_requires_inference=decision["choice_requires_inference"],
                gain_direction=decision["gain_direction"], memory_direction=decision["memory_direction"],
                gain_occurrence_ids=decision["gain_occurrence_ids"], memory_occurrence_ids=decision["memory_occurrence_ids"],
                independent_label_assessment=judgment["consensus_assessment"], **review,
                model_rationale=decision["rationale"], raw_quotes=quote_locators(records, raw_label["evidence_quotes"]),
                validated_quotes=quote_locators(records, decision["evidence_quotes"]),
                raw_vs_validated_changed_fields=changed_fields, retry_diagnostic=prompt[len(core):],
                original_attempt_status=original_status["classification_status"],
                original_attempt_error=original_status.get("error") or "", effective_prompt_hash_matches=True,
                raw_response_fallback=raw.get("legacy_schema_fallback", False))
            provider_rows.append(provider_row)
            reviewed_providers.append(provider_row)
        row = {field: "" for field in pr_fields}
        row.update(templates[case])
        row.update({field: judgment[field] for field in ("independent_relationship_label", "consensus_assessment",
                   "detector_pr_verdict", "measured_gain", "memory_direction", "comparable_measurements", "confidence", "primary_issue", "notes")})
        row.update(review_status="completed", detected_gain_ids=";".join(judgment["detected_gain_ids"]),
            detected_memory_ids=";".join(judgment["detected_memory_ids"]), supporting_source_excerpts=evidence,
            source_locators=[{k: e[k] for k in ("record_id", "table", "locator", "start", "end")} for e in evidence],
            inference_disclosed_correctly="yes" if all(p["inference_disclosed_correctly"] == "yes" for p in reviewed_providers) else "unclear",
            protocol_compliance="compliant" if all(p["protocol_compliance"] == "compliant" for p in reviewed_providers) else "unclear",
            agreement=consensus[case]["consensus_status"], inference_votes=consensus[case]["inference_votes"],
            occurrence_count=len(groups), activation_count=sum(len(g["activations"]) for g in groups),
            valid_quantity_occurrences=sum(v is True for v in group_results), invalid_quantity_occurrences=sum(v is False for v in group_results),
            occurrences_with_invalid_annotations=sum(r["has_invalid_activation"] for r in case_occurrences),
            provider_policy_assessments=[{k: p[k] for k in ("provider", "inference_disclosed_correctly", "protocol_compliance", "critique")} for p in reviewed_providers],
            audit_attribution="agent audit extension; prior labels known; no human adjudication")
        for provider in binary.PROVIDERS:
            row[provider + "_label"] = consensus[case][provider + "_label"]
        pr_rows.append(row)
        case_pairs = pairs_for(records) if case.startswith("476642602-") else []
        pairs.extend(case_pairs)
        lines = [f"# Audit extension: {row['repo_full_name']} #{row['number']}", "", row["html_url"], "",
            f"Consensus: `{row['consensus_label']}`. Audit: `{row['consensus_assessment']}`. Detector: `{detector}`.", "",
            "Agent audit extension; prior labels were known. Not human adjudication or a blinded replication.", "",
            judgment["notes"], "", f"Confidence: {judgment['confidence']}. All {len(groups)} occurrences and {row['activation_count']} memberships are retained.", "",
            "## Decisive source excerpts"]
        for item in evidence:
            lines.extend(["", f"`{item['record_id']}` / `{item['locator']}` chars [{item['start']}, {item['end']}):", "```text", item["excerpt"], "```"])
        lines.extend(["", "## Provider assessment"])
        for provider in reviewed_providers:
            lines.extend(["", f"### {provider['provider']}", "", provider["critique"], "", "Original rationale:", "", provider["model_rationale"]])
        lines.extend(["", "## Occurrence coverage", "", "| ID | Quantity validity | Reason |", "|---|---|---|"])
        lines.extend(f"| {r['occurrence_id']} | {r['review_valid']} | {r['reason']} |" for r in case_occurrences)
        if case_pairs:
            lines.extend(["", "## Same-workload timing/RSS pairs", "", str(dict(Counter(p["reported_directional_relationship"] for p in case_pairs))),
                          "", "Full rows, column headers and source offsets are retained in `paired_measurements_added.csv`; signs are not significance tests."])
        (out / "cases" / f"{case}.md").write_text("\n".join(lines) + "\n")
    for name, rows, fields in (("pr_level_audit.csv", pr_rows, pr_fields), ("occurrence_level_audit.csv", occ_rows, occ_fields),
                               ("activation_level_audit.csv", act_rows, act_fields), ("provider_level_audit.csv", provider_rows, provider_fields)):
        write_csv(out / name, rows, fields)
        reread, _ = read_csv(out / name)
        if reread[:len(old_rows_by_file[name])] != old_rows_by_file[name]:
            raise ValueError(f"Inherited audit judgments changed: {name}")
    write_csv(out / "paired_measurements_added.csv", pairs, list(pairs[0]))
    detection = Counter(r["detector_pr_verdict"] for r in pr_rows)
    assessment = Counter(r["consensus_assessment"] for r in pr_rows)
    summary = {"status": "completed", "audit_attribution": judgments["audit_attribution"],
        "inherited_reviewed_prs": 27, "newly_reviewed_prs": 2, "reviewed_prs": 29, "excluded_prs": 0,
        "provider_decisions": len(provider_rows), "occurrence_groups": len(occ_rows), "activation_memberships": len(act_rows),
        "detector_pr_counts": dict(detection), "detector_false_positive_share_percent": 100 * detection["false_positive"] / len(pr_rows),
        "consensus_assessments": dict(assessment), "consensus_assessment_percent": {k: 100*v/len(pr_rows) for k,v in assessment.items()},
        "added_case_pair_counts": dict(Counter(p["reported_directional_relationship"] for p in pairs)),
        "limits": ["False-positive share among these selected PRs, not population FPR with true negatives.",
                   "The 27 inherited judgments were not re-adjudicated; hashes prove identical evidence and model responses.",
                   "DataHub's numerical documentation claims have uncertain measurement provenance; its binary support judgment is medium-confidence.",
                   "No new model calls, mining, benchmark execution or human adjudication."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Agent audit: all 29 selected trade-off candidates", "",
        "The original 27-case audit is preserved. This extension reviews the two former exclusions after Qwen completed.", "",
        "## PR-level detector false positives", "",
        f"**{detection['false_positive']}/29 = {summary['detector_false_positive_share_percent']:.2f}%** of these selected PRs have no valid detected metric. "
        "RustFS #449 remains the only such case. DataHub #16241 has the reported 768ms extraction timing; rspack #13024 has measured benchmark timings/RSS.", "",
        "This is a false-positive share among detector-selected candidates, not FP/(FP+TN) for a population with sampled negatives.", "",
        "## Binary-label evidence assessment", "", "| Assessment | PRs | Share |", "|---|---:|---:|"]
    for name in ("confirmed_label", "opposite_label_supported", "unsupported_binary_claim", "inconclusive"):
        lines.append(f"| {name} | {assessment[name]} | {100*assessment[name]/29:.2f}% |")
    lines.extend(["", "Unsupported/inconclusive judgments are not proven opposite labels. A forced choice can satisfy its task while lacking a measured relationship.", "",
        "## Added cases", "", "- DataHub #16241: valid detected extraction timing; a measured comparable startup-gain/memory pair is not established. "
        "The unsupported-binary-claim judgment is medium-confidence because the source also states unqualified numerical performance claims; measurement provenance remains a boundary for human adjudication.",
        "- rspack #13024: valid measured metrics; the same table shows both joint improvements and tradeoffs across workloads. "
        "The whole-PR relationship is inconclusive, while Three.js joint improvements have clear scoped support.", "",
        "## Provenance and limits", "", f"Coverage: 29 PRs, {len(provider_rows)} provider decisions, {len(occ_rows)} occurrences, {len(act_rows)} activations. No pending exclusions.",
        "The inherited 27-case judgments remain unchanged. The two added reviews are agent judgments with prior consensus exposure, not human or blinded adjudication. "
        "The original package and its audit remain available separately; this update changes neither model labels nor the original evidence.", ""])
    (out / "summary.md").write_text("\n".join(lines))
    checks = {"prior_input_checksums": len(prior_checksums), "full_input_checksums": len(full_checksums),
        "identical_prior_27_prompts_checkpoints_records": True, "inherited_audit_rows_unchanged": True,
        "reviewed_prs": len(pr_rows), "provider_decisions": len(provider_rows), "occurrences": len(occ_rows),
        "activations": len(act_rows), "all_added_quantity_locators_resolve": True,
        "new_source_judgments": sha(out / "addendum_judgments.json"),
        "source_27_audit_csv": sha(old_audit / "pr_level_audit.csv"), "completed_at": datetime.now(timezone.utc).isoformat()}
    if (len(pr_rows), len(provider_rows), len(occ_rows), len(act_rows)) != (29, 87, 2229, 2792):
        raise ValueError("Full audit coverage does not reconcile.")
    (out / "validation_results.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    audit_checksums = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}
    (out / "audit_checksums.json").write_text(json.dumps(audit_checksums, indent=2, sort_keys=True) + "\n")
    archive_path = full.with_name(full.name + "_audited.zip")
    if archive_path.exists():
        raise FileExistsError("Refusing to overwrite an existing audited archive.")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(full.rglob("*")):
            if path.is_file():
                handle.write(path, str(Path(full.name) / path.relative_to(full)))
    return {**summary, "archive": str(archive_path), "archive_sha256": sha(archive_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=binary.DEFAULT_OUTPUT / "review_partial_27")
    parser.add_argument("--full", type=Path, default=binary.DEFAULT_OUTPUT / "review_full_29")
    args = parser.parse_args()
    print(json.dumps(extend(args.prior, args.full), indent=2))
