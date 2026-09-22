"""Derive each pattern's expected metric dimensions from the catalog, mechanically.

RQ3 Step 2 needs, per RQ1 sub-pattern, the dimensions the pattern is expected to
improve. Rather than annotate those by hand, map the SysLLMatic catalog's
`Optimized Metrics` prose onto D1-D9 with a fixed, auditable keyword table. The
catalog was authored independently of this corpus and of the extractor, so the
resulting expectation is external to both.

The catalog names improvements only; it carries no systematic field for the
resource a pattern *spends*. Class (work-eliminating / resource-exchange /
mixed) and the expended dimension therefore still come from
pattern_tradeoff_classes.csv.

    python analysis/rq3_pattern_and_validation/catalog_expected_dims.py
"""

import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "..", "rq1_optimization_patterns", "catalog",
                       "updated_optimization_catalog.csv")

# Ordered (dimension, regex) over the lower-cased `Optimized Metrics` text.
# Each phrase is one the catalog actually uses; nothing here is inferred beyond
# naming the dimension the phrase denotes.
RULES = [
    ("D1", r"latency|execution time|runtime|run time|response time|load time|"
           r"instruction latency|faster|speed ?up"),
    ("D2", r"throughput|ilp|instruction[- ]level parallelism|instruction throughput|"
           r"requests? per second|greater throughput"),
    ("D3", r"memory|heap|footprint|allocation|\bgc\b|garbage"),
    ("D4", r"cpu|cycles|instruction count|instruction-level efficiency|execution count|"
           r"operations? (?:are )?executed|call count|method calls|branch (?:instruction|mispredict)|"
           r"context switch|thread count|resource utili[sz]ation|utilization|cpi|"
           r"cache miss|cache line|cache efficiency|cache hit|store forwarding|mob loads|"
           r"page miss|replays"),
    ("D5", r"\bi/o\b|io |network|bandwidth|payload|round ?trip|\bqueries\b|"
           r"cache hit rate|data transfer"),
    ("D6", r"build size|binary size|bundle|artifact size|code size|smaller build"),
    ("D7", r"build time|compile time|compilation time|rebuild|\bci\b"),
    ("D8", r"energy|power|joule|watt|carbon|\bcost\b"),
    ("D9", r"concurren|parallelism|scalab|thread pool|contention|workers|under load"),
]
# Phrases masked before matching so a broader rule does not swallow a narrower one.
MASK = [(r"faster build times?|build times?", "D7TOKEN"),
        (r"cache hit rate", "D5TOKEN"),
        (r"memory utili[sz]ation", "D3TOKEN"),
        (r"instruction[- ]level parallelism|\bilp\b", "D2TOKEN"),
        (r"i-cache|cache miss|cache line|cache efficiency", "D4TOKEN")]


def dims_for(text):
    t = str(text).lower()
    held = []
    for pat, tok in MASK:
        for m in re.findall(pat, t):
            held.append((tok, m))
        t = re.sub(pat, tok, t)
    found = set()
    for dim, pat in RULES:
        if re.search(pat, t):
            found.add(dim)
    for tok, phrase in held:                      # credit the masked phrases explicitly
        found.add({"D7TOKEN": "D7", "D5TOKEN": "D5", "D4TOKEN": "D4",
                   "D3TOKEN": "D3", "D2TOKEN": "D2"}[tok])
    return sorted(found, key=lambda d: int(d[1]))


def main():
    c = pd.read_csv(CATALOG)
    c["catalog_expected_dims"] = c["Optimized Metrics"].apply(lambda s: "|".join(dims_for(s)))
    out = c[["High-level Pattern", "Sub pattern", "Optimized Metrics", "catalog_expected_dims"]]
    path = os.path.join(HERE, "catalog_expected_dims.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    empty = out[out.catalog_expected_dims == ""]
    print(f"\npatterns with no dimension derived: {len(empty)}")
    for _, r in empty.iterrows():
        print(f"  {r['Sub pattern'][:52]:54s} <- {str(r['Optimized Metrics'])[:70]}")
    print("\nderived sets:")
    for _, r in out.iterrows():
        print(f"  {r['catalog_expected_dims']:20s} {r['Sub pattern'][:48]:50s} | {str(r['Optimized Metrics'])[:62]}")


if __name__ == "__main__":
    main()
