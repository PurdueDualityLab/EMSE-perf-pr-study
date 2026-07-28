from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perfannotator_common import MODEL_ID, MODEL_REVISION, classify_texts, diff_text
from schema import write_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify PRs with PerfAnnotator-mini using enriched diff data.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-input-chars", type=int, default=50000)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    built = [diff_text(row, args.max_input_chars) for _, row in df.iterrows()]
    texts = [item[0] for item in built]
    truncated = [item[1] for item in built]
    labels, scores, device = classify_texts(
        texts,
        model_id=args.model,
        revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
    )

    result = df.copy()
    result["perfannotator_diff_label_id"] = labels
    result["perfannotator_diff_is_performance_improving"] = [label == 1 for label in labels]
    result["perfannotator_diff_score"] = scores
    result["perfannotator_diff_model"] = args.model
    result["perfannotator_diff_model_revision"] = args.model_revision or ""
    result["perfannotator_diff_device"] = device
    result["perfannotator_diff_input_chars"] = [len(text) for text in texts]
    result["perfannotator_diff_truncated"] = truncated

    write_parquet(result, args.output)


if __name__ == "__main__":
    main()
