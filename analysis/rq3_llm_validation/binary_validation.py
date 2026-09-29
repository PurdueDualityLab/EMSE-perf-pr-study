"""Traceable formatting alignment and audited source-dimension corrections."""

from __future__ import annotations

import hashlib
import html
import json
import re

VERSION = "rq3-binary-validation-v2"
TAG = re.compile(
    r"</?(code|strong|em|b|i|span|a|li|ul|ol|p|div|pre|br|table|thead|tbody|tr|td|th|details|summary)"
    r"(?=[\s/>])(?:[^>\"']|\"[^\"]*\"|'[^']*')*>", re.I)
BLOCK_TAGS = {"li", "ul", "ol", "p", "div", "pre", "br", "table", "thead", "tbody", "tr", "td", "th", "details", "summary"}
FORMATTING = re.compile(TAG.pattern + r"|\*\*|__|`+|&(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);", re.I)


def rendered_text(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize presentation only, mapping every retained character to source."""
    chars, positions = [], []

    def append(value, start, end):
        for char in value:
            char = " " if char.isspace() else char
            if char == " " and chars and chars[-1] == " ":
                positions[-1] = (positions[-1][0], end)
                continue
            chars.append(char)
            positions.append((start, end))

    last = 0
    for match in FORMATTING.finditer(text):
        for index in range(last, match.start()):
            append(text[index], index, index + 1)
        token = match.group()
        tag = TAG.fullmatch(token)
        if tag:
            if tag.group(1).lower() in BLOCK_TAGS:
                append(" ", match.start(), match.end())
        elif token.startswith("&"):
            append(html.unescape(token), match.start(), match.end())
        last = match.end()
    for index in range(last, len(text)):
        append(text[index], index, index + 1)
    return "".join(chars), positions


def align_formatted_quote(quote: str, originals: list[str]) -> str:
    """Return a uniquely matched original substring; never edit words or values."""
    needle = rendered_text(quote)[0].strip()
    candidates = set()
    if not needle:
        raise ValueError("Empty quote after presentation normalization.")
    for original in originals:
        text, positions = rendered_text(original)
        index = text.find(needle)
        while index >= 0:
            start, end = positions[index][0], positions[index + len(needle) - 1][1]
            restored = original[start:end]
            if rendered_text(restored)[0].strip() == needle:
                candidates.add(restored)
            index = text.find(needle, index + 1)
    if len(candidates) != 1:
        raise ValueError("Quote is not uniquely present after HTML/Markdown formatting alignment.")
    return candidates.pop()


def corrected_dimensions(request: dict) -> tuple[dict, list[dict]]:
    """Apply explicit agent-audited corrections bound to an immutable record.

    These annotations affect validation only. The original prompt, occurrence
    taxonomy, model judgments, and experimental input hashes are not rewritten.
    """
    dimensions = dict(request["occurrence_dimensions"])
    applied = []
    for correction in request.get("validation_dimension_corrections", []):
        occurrence_id = correction["occurrence_id"]
        if (correction["custom_id"] != request["custom_id"]
                or correction["prompt_sha256"] != request["prompt_sha256"]
                or dimensions.get(occurrence_id) != correction["from_dimension"]
                or correction["to_dimension"] not in {f"D{i}" for i in range(10)}
                or correction["from_dimension"] == correction["to_dimension"]
                or not correction["rationale"].strip()
                or correction["review_type"] != "agent_audit"):
            raise ValueError("Audited dimension correction does not match this frozen request.")
        records = [text for text in request["quote_corpus"]
                   if hashlib.sha256(text.encode()).hexdigest() == correction["record_sha256"]]
        if len(set(records)) != 1 or not correction["basis_quotes"]:
            raise ValueError("Audited dimension correction has no matching source record.")
        text = records[0]
        if not all(quote and quote in text for quote in correction["basis_quotes"]):
            raise ValueError("Audited dimension correction lacks its recorded evidence.")
        headers = []
        for line in request["prompt"].splitlines():
            if not line.startswith('{"record_id": '):
                continue
            header = json.loads(line)
            if header["record_id"] == correction["record_id"] and header["locator"] == correction["record_locator"]:
                headers.append(line)
        if len(headers) != 1 or headers[0] + "\n\n" + text + "\n\n[END RECORD]" not in request["prompt"]:
            raise ValueError("Audited dimension correction has a mismatched record locator.")
        occurrences = [json.loads(line) for line in request["prompt"].splitlines()
                       if line.startswith('{"id":') and json.loads(line).get("id") == occurrence_id]
        if len(occurrences) != 1 or occurrences[0]["dimension"] != correction["from_dimension"]:
            raise ValueError("Audited correction does not identify the original occurrence.")
        locations = [loc for activation in occurrences[0]["activations"] for loc in activation["quantity_at"]
                     if loc["record_id"] == correction["record_id"]]
        if not any(text[loc["start"]:loc["end"]] == correction["quantity_text"] for loc in locations):
            raise ValueError("Audited correction does not locate the original detected quantity.")
        dimensions[occurrence_id] = correction["to_dimension"]
        applied.append(dict(correction))
    return dimensions, applied
