"""
RQ3 Step 1 — pattern definitions for the deterministic metric extractor.

A performance-metric *dimension* (D1–D9, Table `tab:metric-dimensions`) is
recorded as *reported* in a PR when a dimension cue co-occurs, within a bounded
token window, with a quantitative claim *about that dimension*:

  * a numeral carrying a unit appropriate to the dimension — time (``120ms``),
    bytes (``4MB``), rate (``1200 req/s``), energy / currency (``12 kWh``,
    ``$40``) — or
  * a comparative statement (``2.5x faster``, ``reduced from 125ns to 83ns``,
    ``-40%``, ``before: 3.2s | after: 1.1s``).

An isolated numeral or an incidental cue does not count. A cue whose window
contains a hypothetical / prospective construction (``should measure``,
``would reduce``, ``TODO: benchmark`` …) is discarded so that prospective
validation is not recorded as reported validation.

Text is normalized before matching (lower-casing; URL / e-mail / version /
SHA / boilerplate stripping; unit-synonym folding). All patterns therefore
assume lower-case text.

The dimension taxonomy follows Gregg (2021) and Jain (1991); dimensions are
measured quantities only — asymptotic-complexity claims are *not* dimensions
(they are handled by RQ2's static-reasoning label).
"""

import re

# ---------------------------------------------------------------------------
# Window sizes (tokens = whitespace-delimited units of the normalized text)
# ---------------------------------------------------------------------------

WINDOW_TOKENS = 12      # cue <-> quantitative claim
EXCLUSION_TOKENS = 8    # cue <-> hypothetical / prospective construction

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")                            # co-authored-by trailers
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG_RE = re.compile(r"</?[a-z][^>]{0,300}>", re.I)
_LONG_TOKEN_RE = re.compile(r"\S{120,}")                           # minified code, base64, data URIs
_SEMVER_RE = re.compile(r"\bv?\d+\.\d+\.\d+(?:[-+.][\w.]+)*\b")   # 0.25.2, 24.0.6-canary.874…
_HEX_RE = re.compile(r"\b[0-9a-f]{7,40}\b")                        # commit SHAs
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:t[\d:.]+z?)?\b")

# Agent-platform boilerplate that carries numerals unrelated to the change.
_BOILERPLATE_RES = [
    re.compile(r"share your feedback on copilot coding agent.*?survey\.?", re.S),
    re.compile(r"link to devin run:.*?(?:\n|$)"),
    re.compile(r"co-authored-by:.*?(?:\n|$)"),
]

# CJK text carries no spaces, so numerals and their cues sit inside long runs;
# split numerals off CJK runs and turn CJK punctuation into token boundaries.
_CJK = r"　-ヿ㐀-鿿＀-￯"
_CJK_SPACING = [
    (re.compile(rf"([{_CJK}])(?=[\d(~\-+])"), r"\1 "),
    (re.compile(rf"(?<=[\d)%xs])(?=[{_CJK}])"), " "),
    (re.compile(r"[、。：，（）「」【】]"), " "),
]

# (pattern, replacement) — applied to lower-cased text, in order.
_UNIT_SYNONYMS = _CJK_SPACING + [
    (re.compile(r"(?<=\d)\s*(?:ナノ秒|纳秒)"), "ns"),                  # Japanese / Chinese units
    (re.compile(r"(?<=\d)\s*(?:マイクロ秒|微秒)"), "us"),
    (re.compile(r"(?<=\d)\s*(?:ミリ秒|毫秒)"), "ms"),
    (re.compile(r"(?<=\d)\s*秒"), "s"),
    (re.compile(r"(?<=\d)\s*(?:分間|分钟|分)"), "min"),
    (re.compile(r"(?<=\d)\s*倍"), "x "),                            # "3倍高速化" -> "3x 高速化"
    (re.compile(r"(?<=\d)\+(?=\s|$)"), ""),                          # "4+ seconds" -> "4 seconds"
    (re.compile(r"(?<=\d)\s*(?:nanoseconds?|nanosecs?|nsecs?)\b"), "ns"),
    (re.compile(r"(?<=\d)n(?=\s*±)"), "ns"),                          # Go benchstat "9862.0n ± 0%"
    (re.compile(r"(?<=\d)\s*(?:microseconds?|microsecs?|usecs?|µs|μs)\b"), "us"),
    (re.compile(r"(?<=\d)\s*(?:milliseconds?|millisecs?|msecs?|millis)\b"), "ms"),
    (re.compile(r"(?<=\d)\s*(?:seconds?|secs?)\b"), "s"),
    (re.compile(r"(?<=\d)\s*(?:minutes?|mins?)\b"), "min"),
    (re.compile(r"(?<=\d)\s*(?:hours?|hrs?)\b"), "hr"),
    (re.compile(r"(?<=\d)\s*(?:kilobytes?|kib|kbytes?)\b"), "kb"),
    (re.compile(r"(?<=\d)\s*(?:megabytes?|mib|mbytes?)\b"), "mb"),
    (re.compile(r"(?<=\d)\s*(?:gigabytes?|gib|gbytes?)\b"), "gb"),
    (re.compile(r"(?<=\d)\s*(?:terabytes?|tib)\b"), "tb"),
    (re.compile(r"(?<=\d)\s*bytes?\b"), "b"),
    (re.compile(r"×"), "x"),
    (re.compile(r"→|⟶|⇒|=>|->|-->"), " -> "),
    (re.compile(r"[–—]"), "-"),
    (re.compile(r"≈|~"), "~"),
]


def normalize(text: str) -> str:
    """Lower-case and fold unit synonyms; strip URLs, e-mails, HTML, versions, SHAs."""
    if not text:
        return ""
    t = _LONG_TOKEN_RE.sub(" ", text)   # first: keeps every later \S+ scan short
    t = _HTML_COMMENT_RE.sub(" ", t)
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _HTML_TAG_RE.sub(" ", t)
    t = t.lower()
    for pat in _BOILERPLATE_RES:
        t = pat.sub(" ", t)
    t = _ISO_DATE_RE.sub(" ", t)
    t = _SEMVER_RE.sub(" ", t)
    t = _HEX_RE.sub(" ", t)
    for pat, rep in _UNIT_SYNONYMS:
        t = pat.sub(rep, t)
    return t


# ---------------------------------------------------------------------------
# Quantitative claims
# ---------------------------------------------------------------------------

NUM = r"(?:~\s?)?\d(?:[\d,]*\d)?(?:\.\d+)?"
TIME_UNIT = r"(?:ns|us|ms|s|min|hr)"
BYTE_UNIT = r"(?:b|kb|mb|gb|tb)"
RATE_UNIT = (
    r"(?:ops?/s(?:ec)?|op/s|req(?:uests?)?/s(?:ec)?|qps|rps|tps|"
    r"items?/s(?:ec)?|tokens?/s(?:ec)?|msgs?/s(?:ec)?|messages?/s(?:ec)?|"
    r"events?/s(?:ec)?|it/s|iters?/s(?:ec)?|rows?/s(?:ec)?|records?/s(?:ec)?|"
    r"(?:mb|gb|kb)/s|mbps|gbps|kbps|fps|/s(?:ec)?)"
)
ENERGY_UNIT = r"(?:j|kj|mj|kwh|wh|mah|watts?|joules?)"
CURRENCY_WORD = r"(?:usd|eur|gbp|dollars?|cents?)"

COMPARATIVE_WORDS = (
    r"(?:faster|slower|quicker|speed[- ]?ups?|improv(?:e|es|ed|ement|ements)|"
    r"reduc(?:e|es|ed|tion|tions)|decreas(?:e|es|ed)|increas(?:e|es|ed)|"
    r"fewer|less|more|smaller|larger|lower|higher|shorter|longer|"
    r"sav(?:e|es|ed|ing|ings)|cuts?|dropp?(?:ed|s)?|shr[ai]nk|shrunk|"
    r"down|up|gain|gains|regress(?:ion|ed)|slowdown|overhead|boost(?:ed)?)"
)
COMPARATIVE_D1 = r"(?:faster|slower|quicker|speed[- ]?ups?|speedup)"
# CJK comparatives (reduction, improvement, speed-up …) have no word boundaries
CJK_COMPARATIVE = r"(?:削減|減少|短縮|改善|向上|高速化|低下|増加|提升|降低|减少|提高|加速|优化|缩短|下降|增加|改进)"
CJK_COMPARATIVE_D1 = r"(?:高速化|加速|短縮|缩短)"
COMP = rf"(?:\b{COMPARATIVE_WORDS}\b|{CJK_COMPARATIVE})"
COMP_D1 = rf"(?:\b{COMPARATIVE_D1}\b|{CJK_COMPARATIVE_D1})"
_GAP = r"\s*(?:\S{1,40}\s+){0,5}[*_`(\[]{0,3}"   # up to five intervening whitespace-delimited tokens

# Each entry: (kind, compiled regex).
QUANT_PATTERNS = [
    ("time",     re.compile(
        rf"(?:\b(?!(?:19|20)\d{{2}}s\b){NUM}\s?{TIME_UNIT}\b"
        rf"|\({TIME_UNIT}\)\s*[:=]?\s*{NUM}\b)")),   # "duration (s): 3.31"
    ("bytes",    re.compile(rf"\b{NUM}\s?{BYTE_UNIT}\b")),
    ("rate",     re.compile(rf"\b{NUM}\s?{RATE_UNIT}\b")),
    ("energy",   re.compile(rf"\b{NUM}\s?{ENERGY_UNIT}\b")),
    ("currency", re.compile(
        rf"(?:[$€£]\s?\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?\b|[$€£]\s?\d+\.\d{{2}}\b|[$€£]\s?\d{{2,}}\b"
        rf"|\b{NUM}\s?{CURRENCY_WORD}\b)")),
    # comparative statements
    ("mult",     re.compile(
        rf"(?:\b{NUM}\s?x\b{_GAP}{COMP}"
        rf"|{COMP}{_GAP}{NUM}\s?x\b"
        rf"|\b{NUM}\s?(?:x|times)\s+(?:as\s+)?{COMP_D1}"
        rf"|\b{NUM}\s?times\s+(?:fewer|less|more|smaller|larger))")),
    ("pct",      re.compile(
        rf"(?:(?<![\w.])[-+]\s?{NUM}\s?%"
        rf"|\b{NUM}\s?(?:%|percent){_GAP}{COMP}"
        rf"|{COMP}{_GAP}{NUM}\s?(?:%|percent)"
        rf"|\bby\s+(?:~|about|approximately|roughly|around|over|up to|nearly)?\s*{NUM}\s?(?:%|percent))")),
    ("from_to",  re.compile(
        rf"(?:\bfrom\s+[*_`(]{{0,3}}{NUM}\S{{0,12}}\s+(?:\S{{1,40}}\s+){{0,3}}?(?:to|->)\s+[*_`(]{{0,3}}{NUM}"
        rf"|\b{NUM}\S{{0,12}}\s+(?:\S{{1,40}}\s+){{0,4}}?->\s+[*_`(]{{0,3}}{NUM})")),
    ("before_after", re.compile(
        rf"\b(?:before|after|old|new|baseline|original|optimi[sz]ed|prev(?:ious)?|current)\b"
        rf"\s*(?:[:=|]|->)\s*\**{NUM}\b")),
]

ALL_KINDS = {k for k, _ in QUANT_PATTERNS}
UNIT_KINDS = {"time", "bytes", "rate", "energy", "currency"}   # numerals whose unit names a dimension

# ---------------------------------------------------------------------------
# Hypothetical / prospective constructions (exclusion list)
# ---------------------------------------------------------------------------

EXCLUSION_RE = re.compile(
    r"\b(?:should|would|could|might|may|shall|will|won't|wouldn't|"
    r"todo|fixme|tbd|"
    r"plan(?:s|ned|ning)?\s+to|need(?:s|ed)?\s+to\s+(?:be\s+)?(?:measure|benchmark|profile|verif|test|confirm)\w*|"
    r"expect(?:s|ed)?(?:\s+to)?|estimat(?:e|es|ed|ion)|theoretical(?:ly)?|"
    r"potential(?:ly)?|hypothetical(?:ly)?|in\s+theory|ideally|hopefully|"
    r"aim(?:s|ed)?\s+to|intend(?:s|ed)?\s+to|goal|target(?:s|ed|ing)?\s+(?:a|an|to)|"
    r"not\s+(?:yet\s+)?(?:measured|benchmarked|profiled|tested|verified)|"
    r"untested|unmeasured|no\s+benchmarks?|"
    r"coverage|codecov)\b"                       # test-coverage percentages are not performance claims
)

# A short heading-like line announcing a prospective block, e.g.
# "## Expected performance improvements", "**Estimated impact:**". Every token
# up to the next blank line / heading is treated as prospective.
PROSPECTIVE_HEADING_RE = re.compile(
    r"(?m)^[\s#*>\-+\d.]{0,8}(?:expected|estimated|projected|anticipated|theoretical|potential|planned|"
    r"hypothes(?:is|ised|ized))\b[^\n]{0,70}$"
)
_BLOCK_BREAK_RE = re.compile(r"\n\s*\n|\n[ \t]*#")
BLOCK_MAX_CHARS = 1500


def block_end(text: str, start: int) -> int:
    """End offset of the paragraph/section that starts after ``start``."""
    m = _BLOCK_BREAK_RE.search(text, start, min(len(text), start + BLOCK_MAX_CHARS))
    return m.start() if m else min(len(text), start + BLOCK_MAX_CHARS)


# ---------------------------------------------------------------------------
# Dimension cues (D1–D9)
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "D1": "Latency / execution time",
    "D2": "Throughput",
    "D3": "Memory",
    "D4": "CPU / compute work",
    "D5": "I/O and network",
    "D6": "Artifact size",
    "D7": "Build and CI time",
    "D8": "Energy and cost",
    "D9": "Scalability / concurrency",
    # a quantified gain whose dimension is not named ("3-5x overall performance
    # improvement"); recorded only when no D1–D9 cue is in the claim's window
    "D0": "Unspecified performance",
}

CUE_PATTERNS = {
    "D1": re.compile(
        r"(?:\b(?:latenc(?:y|ies)|run[- ]?time|execution[- ]time|exec[- ]time|"
        r"elapsed(?:[- ]time)?|response[- ]time|wall[- ]?clock|wall[- ]?time|"
        r"startup(?:[- ]time)?|start-up|boot[- ]time|cold[- ]start|"
        r"load(?:ing)?[- ]time|render(?:ing)?[- ]time|processing[- ]time|"
        r"request[- ]time|query[- ]time|frame[- ]time|duration|"
        r"time[- ]to[- ](?:first[- ]byte|interactive|first[- ]paint)|ttfb|tti|lcp|fcp|"
        r"p(?:50|75|90|95|99|999)|percentile|tail[- ]latency|"
        r"speed[- ]?ups?|faster|slower|quicker|took|takes|taking|timing|time)\b"
        r"|実行時間|処理時間|所要時間|応答時間|レイテンシ|高速化|耗时|执行时间|运行时间|响应时间|延迟|时间)"),
    "D2": re.compile(
        r"(?:\b(?:throughput|ops?/s(?:ec)?|req(?:uests?)?/s(?:ec)?|"
        r"requests?\s+per\s+second|queries\s+per\s+second|transactions?\s+per\s+second|"
        r"qps|rps|tps|items?/s(?:ec)?|tokens?/s(?:ec)?|msgs?/s(?:ec)?|events?/s(?:ec)?|"
        r"it/s|iters?/s(?:ec)?|rows?/s(?:ec)?|records?/s(?:ec)?|(?:mb|gb|kb)/s|"
        r"mbps|gbps|fps|frames\s+per\s+second|per\s+second|/sec)\b"
        r"|スループット|吞吐量|吞吐)"),
    "D3": re.compile(
        r"(?:\b(?:(?<!in-)(?<!in )memory|mem|heap|rss|resident[- ]set|footprint|"
        r"alloc(?:ation|ations|ated|ating|s)?|garbage[- ]collect(?:ion|or|ed)?|"
        r"gc(?:[- ](?:pause|pressure|time|cycles?|overhead))?|"
        r"memory[- ]leaks?|leak(?:s|ing|ed)?|oom|out[- ]of[- ]memory|swap|"
        r"page[- ]faults?|peak[- ](?:memory|rss)|retained|"
        r"objects?[- ]created|bytes[- ]allocated|arena|pool[- ]size)\b"
        r"|メモリ|ヒープ|内存|堆内存)"),
    "D4": re.compile(
        r"(?:\b(?:cpu(?:[- ](?:usage|time|load|utili[sz]ation|cycles|cores?|bound))?|"
        r"utili[sz]ation|cpu[- ]cycles|clock[- ]cycles|cycles[- ](?:saved|reduced)|"
        r"instructions?[- ](?:count|retired|executed)|instruction[- ]count|ipc|flops|"
        r"function[- ]calls|method[- ]calls|call[- ]count|ncalls|"
        r"invocations?|branch[- ]mispredict\w*|branch[- ]misses|"
        r"compute[- ]time|cores?[- ]used)\b"
        r"|呼び出し回数|関数呼び出し|cpu使用率|cpu占用|调用次数)"),
    "D5": re.compile(
        r"\b(?:n\+1|round[- ]?trips?|query[- ]count|number[- ]of[- ]queries|fewer[- ]queries|"
        r"(?:db|database|sql)[- ](?:calls?|queries|query|hits?|round[- ]?trips?|operations?)|"
        r"queries[- ](?:reduced|from|to|per[- ]request|per[- ]page)|"
        r"api[- ](?:calls?|requests?)|http[- ](?:calls?|requests?)|"
        r"network[- ](?:calls?|requests?|round[- ]?trips?|traffic|transfer|usage|bandwidth|i/o)|"
        r"requests?[- ](?:count|made|sent|per[- ]page)|payload(?:[- ]size)?|"
        r"bytes[- ](?:transferred|sent|received|read|written)|transfer(?:red)?[- ]size|"
        r"response[- ]size|request[- ]size|bandwidth|"
        r"cache[- ]hits?|hit[- ]rate|hit[- ]ratio|miss[- ]rate|cache[- ]miss(?:es)?|"
        r"disk[- ](?:reads?|writes?|io|i/o)|i/o|io[- ]ops|iops|syscalls?|"
        r"file[- ](?:reads?|writes?|operations)|(?:\d\s?b|byte|small|tiny)[- ]writes)\b"),
    "D6": re.compile(
        r"\b(?:bundle(?:[- ]size)?s?|binary[- ]size|executable[- ]size|image[- ]size|"
        r"artifact[- ]size|package[- ]size|wasm[- ]size|gzip(?:ped)?|brotli|"
        r"minified|chunk[- ]size|dist[- ]size|apk|ipa|jar[- ]size|wheel[- ]size|"
        r"install[- ]size|download[- ]size|file[- ]size|code[- ]size|build[- ]size|"
        r"output[- ]size|tree[- ]?shak\w+|dead[- ]code|"
        r"size[- ]of[- ]the[- ](?:bundle|binary|image|package))\b"),
    "D7": re.compile(
        r"(?:ビルド時間|编译时间|构建时间|"
        r"\b(?:build(?:ing)?[- ]times?|build[- ]duration|builds?[- ](?:take|takes|took|taking|run|complete)|"
        r"compil(?:e|ation)[- ]times?|compil(?:e|es|ing|ation)[- ](?:takes?|took)|"
        r"rebuild(?:s|ing)?(?:[- ]time)?|incremental[- ]build|"
        r"ci[- ](?:time|duration|runs?|pipelines?|jobs?)|pipeline[- ](?:time|duration)|"
        r"run[- ]duration|test[- ](?:suite|run|execution)?[- ]?(?:time|duration)|"
        r"tests?[- ](?:take|takes|took|run[- ]in|complete[- ]in|finish[- ]in)|"
        r"typecheck(?:ing)?[- ]time|tsc[- ]time|lint(?:ing)?[- ]time|bundling[- ]time|"
        r"webpack[- ]build|vite[- ]build|link(?:ing)?[- ]time|hot[- ]reload|hmr|docker[- ]build)\b)"),
    "D8": re.compile(
        r"\b(?:joules?|kwh|watts?|energy|power[- ](?:consumption|draw|usage)|battery|"
        r"carbon|co2|spend(?:ing)?|billing|bill|price|pricing|"
        r"token[- ]cost|api[- ]cost|cloud[- ]cost|compute[- ]cost|infra(?:structure)?[- ]cost|"
        r"(?<!at the )(?<!the )costs?(?! of))\b"),
    "D9": re.compile(
        r"\b(?:concurren(?:cy|t)(?![- ](?:safe|safety|issues?|bugs?|model|primitives?))|parallelism|under[- ]load|load[- ]test(?:ing|s)?|stress[- ]test|"
        r"contention|lock(?:s|ing)?[- ](?:wait|contention)|scalability|scal(?:es|ing)[- ](?:to|up|linearly|with)|"
        r"thread[- ]pool|worker[- ]pool|connection[- ]pool(?:ing)?|backpressure|"
        r"queue[- ](?:depth|length)|saturat(?:ion|ed)|simultaneous)\b"),
    "D0": re.compile(
        r"\b(?:performance|perf|efficiency|efficient|overhead|slowness|bottlenecks?|"
        r"responsiveness|snappier|sluggish|optimi[sz]ation gains?|speed)\b"),
}

# Quantitative-claim kinds that count as a claim *about* each dimension.
QUANT_KINDS = {
    "D1": {"time", "pct", "mult", "from_to", "before_after"},
    "D2": {"rate", "pct", "mult", "from_to", "before_after"},
    "D3": {"bytes", "pct", "mult", "from_to", "before_after"},
    "D4": {"time", "pct", "mult", "from_to", "before_after"},
    "D5": {"bytes", "rate", "pct", "mult", "from_to", "before_after"},
    "D6": {"bytes", "pct", "mult", "from_to", "before_after"},
    "D7": {"time", "pct", "mult", "from_to", "before_after"},
    "D8": {"energy", "currency", "pct", "mult", "from_to", "before_after"},
    "D9": {"pct", "mult", "from_to", "before_after"},
    "D0": {"pct", "mult", "from_to", "before_after"},   # comparative claims only — a unit would name the dimension
}

# Phrases masked *before* matching a dimension whose generic cue would otherwise
# swallow them (e.g. "build time" is D7, not the generic D1 cue "time";
# "queries per second" is D2, not a D5 cue).
MASK_BEFORE = {
    "D1": ["D7"],
    "D5": ["D2", "D1"],
}

# Attribution: a quantitative claim is credited to the *nearest* dimension cue
# within the window (ties credit both). A claim next to "latency" is a D1 claim
# even if "thread pool" also sits in the window.
NEAREST_CUE_WINS = True

# Fallback dimensions record a claim only when no other dimension's cue lies in
# the claim's window (used for a catch-all "unspecified performance" dimension).
FALLBACK_DIMS = {"D0"}

# A cue match is dropped when a cue of a *suppressing* dimension lies in the
# same window (a time claim next to "build time" is a D7 claim, not D1).
SUPPRESS_IF_NEAR = {
    "D1": ["D7"],
}

# Self-sufficient cues: the unit or construction identifies the dimension by
# itself (Table tab:metric-dimensions lists these as surface cues). Each entry is
# (regex, kinds-of-quantitative-context-required-in-window-or-None,
#  dimensions-that-suppress-when-present-in-window).
SELF_SUFFICIENT = {
    # "2.5x faster", "51% speedup"; time-unit numerals in a comparative context
    "D1": [
        (re.compile(rf"\b{NUM}\s?(?:x|times)\s+(?:as\s+)?{COMP_D1}"), None, ["D7"]),
        (re.compile(rf"\b{NUM}\s?(?:%|percent)\s*{COMP_D1}"), None, ["D7"]),
        (re.compile(rf"{COMP_D1}\W*(?:of|by)?\W*{NUM}\s?(?:x|%|percent)\b"), None, ["D7"]),
        (re.compile(rf"\b{NUM}\s?{TIME_UNIT}\b"), {"from_to", "before_after", "mult", "pct"}, ["D7"]),
        # "| 44.31 ns |" — a time-unit table cell (BenchmarkDotNet / criterion output)
        (re.compile(rf"\|\s*\**{NUM}\s?{TIME_UNIT}\**\s*\|"), None, ["D7"]),
    ],
    # "| allocated |" header followed within 40 lines by a byte-unit cell
    "D3": [
        (re.compile(rf"(?:allocated|alloc ratio|gen ?0|memory|heap)[^\n]{{0,400}}\n(?:[^\n]*\n){{0,40}}?[^\n]*\|\s*\**{NUM}\s?{BYTE_UNIT}\**\s*\|"), None, []),
    ],
    # rate numerals; "3 million elements per second"
    "D2": [
        (re.compile(rf"\b{NUM}\s?{RATE_UNIT}\b"), None, []),
        (re.compile(rf"\b{NUM}\s?(?:k|m|million|thousand|billion)?\s+(?:\S{{1,40}}\s+){{0,2}}per\s+second\b"), None, []),
    ],
    # "8 threads", "100 concurrent users"
    "D9": [(re.compile(
        rf"\b{NUM}\s?(?:concurrent|workers?|threads?|connections?|users|clients|"
        rf"goroutines?|cores?|parallel|simultaneous)\b"), None, [])],
}
