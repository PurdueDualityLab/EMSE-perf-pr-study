"""Measure paired-file maintainability changes at the stored PR base/head SHAs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import threading
from urllib.parse import quote

import lizard
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCAL = threading.local()
METRICS = ['nloc', 'avg_ccn', 'function_count']


def metrics(path, text):
    result = lizard.analyze_file.analyze_source_code(path, text)
    functions = result.function_list
    return {'nloc': result.nloc, 'function_count': len(functions),
            'avg_ccn': float(np.mean([f.cyclomatic_complexity for f in functions])) if functions else None}


def paired_change(before, after):
    """Exclude missing sides; zero baselines have absolute but not relative deltas."""
    if before is None or after is None:
        return None, None
    delta = after - before
    return delta, 100 * delta / before if before else None


def fetch_source(repo, sha, path, cache):
    url = f'https://raw.githubusercontent.com/{repo}/{sha}/{quote(path, safe="/")}'
    key = hashlib.sha256(url.encode()).hexdigest()
    target = cache / f'{key}.gz'
    if target.exists():
        return gzip.decompress(target.read_bytes()).decode('utf-8', errors='replace')
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        LOCAL.session.mount('https://', HTTPAdapter(max_retries=retry))
    response = LOCAL.session.get(url, timeout=(15, 60))
    response.raise_for_status()
    temporary = target.with_suffix(f'.{threading.get_ident()}.tmp')
    temporary.write_bytes(gzip.compress(response.content))
    temporary.replace(target)
    return response.content.decode('utf-8', errors='replace')


def analyze_file(row, output):
    key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    checkpoint = output / 'files' / f'{key}.json'
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved['status'] == 'complete':
            return saved
    record = dict(row)
    try:
        before = fetch_source(row['repo_full_name'], row['base_sha'], row['old_path'], output / 'sources')
        after = fetch_source(row['repo_full_name'], row['head_sha'], row['filename'], output / 'sources')
        record.update(status='complete', before=metrics(row['old_path'], before), after=metrics(row['filename'], after),
            before_sha256=hashlib.sha256(before.encode()).hexdigest(), after_sha256=hashlib.sha256(after.encode()).hexdigest())
    except (requests.RequestException, ValueError, UnicodeError) as exc:
        response = getattr(exc, 'response', None)
        record.update(status='failed', error=type(exc).__name__,
            http_status=response.status_code if response is not None else None)
    temporary = checkpoint.with_suffix('.tmp')
    temporary.write_text(json.dumps(record) + '\n')
    temporary.replace(checkpoint)
    return record


def aggregate(records):
    """Historical mean-of-file means, using identical available file pairs."""
    rows = []
    grouped = {}
    for record in records:
        grouped.setdefault((record['repo_id'], record['number'], record['sample_arm']), []).append(record)
    for (repo, number, arm), files in sorted(grouped.items()):
        files = sorted(files, key=lambda r: (r.get('old_path', ''), r.get('filename', '')))
        complete = [r for r in files if r['status'] == 'complete']
        for metric in METRICS:
            pairs = [r for r in complete if r['before'][metric] is not None and r['after'][metric] is not None]
            before = float(np.mean([r['before'][metric] for r in pairs])) if pairs else None
            after = float(np.mean([r['after'][metric] for r in pairs])) if pairs else None
            delta, percent = paired_change(before, after)
            rows.append(dict(repo_id=repo, number=number, sample_arm=arm, metric=metric,
                before=before, after=after, delta=delta, delta_pct=percent,
                eligible_files=len(files), fetched_pairs=len(complete), metric_pairs=len(pairs),
                complete_pr=len(complete) == len(files)))
    return pd.DataFrame(rows)


def report(frame, output, sample):
    from scipy.stats import mannwhitneyu, false_discovery_control
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Compare the same PR cohort across all three metrics, with finite relative changes.
    usable = frame[frame.complete_pr & frame.delta_pct.notna()]
    counts = usable.groupby(['repo_id', 'number']).size()
    keys = counts[counts == len(METRICS)].index
    included = usable.set_index(['repo_id', 'number']).loc[keys].reset_index()
    included.to_csv(output / 'analysis_deltas.csv', index=False)
    inclusion = sample[['repo_id', 'number', 'sample_arm']].copy()
    inclusion['included'] = pd.MultiIndex.from_frame(inclusion[['repo_id', 'number']]).isin(keys)
    inclusion.to_csv(output / 'sample_inclusion.csv', index=False)
    coverage = sample.groupby('sample_arm').size().rename('sample_prs').to_frame()
    per_pr = frame[frame.metric.eq('avg_ccn')]
    coverage['eligible_prs'] = per_pr.groupby('sample_arm').size()
    coverage['complete_file_pairs_prs'] = per_pr[per_pr.complete_pr].groupby('sample_arm').size()
    coverage['included_prs'] = included[included.metric.eq('avg_ccn')].groupby('sample_arm').size()
    coverage.to_csv(output / 'coverage.csv')
    summary, tests = [], []
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    names = {'nloc': 'Mean file NLOC', 'avg_ccn': 'Mean file AvgCCN', 'function_count': 'Mean file function count'}
    for ax, metric in zip(axes, METRICS):
        arrays = []
        for arm in ['agentic', 'human_candidate']:
            values = included.loc[(included.metric == metric) & included.sample_arm.eq(arm), 'delta_pct']
            arrays.append(values.to_numpy())
            summary.append(dict(metric=metric, sample_arm=arm, n=len(values), median=values.median(),
                mean=values.mean(), increases=int(values.gt(0).sum()), increase_pct=100*values.gt(0).mean(),
                p10=values.quantile(.1), p90=values.quantile(.9)))
        if any(len(a) == 0 for a in arrays):
            raise ValueError('No complete observations for one study arm')
        u, p = mannwhitneyu(*arrays, alternative='two-sided')
        tests.append(dict(metric=metric, u=float(u), p=float(p),
            cliffs_delta=2*float(u)/(len(arrays[0])*len(arrays[1]))-1))
        boxes = ax.boxplot(arrays, whis=(10, 90), showfliers=False, patch_artist=True,
                           tick_labels=['Agentic', 'Human-authored'])
        for patch, color in zip(boxes['boxes'], ['#9BBCE8', '#B9DAB9']):
            patch.set_facecolor(color)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(names[metric]); ax.tick_params(axis='x', labelrotation=15)
    axes[0].set_ylabel('Change from stored base to head (%)')
    fig.tight_layout(); fig.savefig(output / 'maintainability_delta_boxplot.pdf', bbox_inches='tight')
    plt.close(fig)
    test_frame = pd.DataFrame(tests)
    test_frame['p_bh'] = false_discovery_control(test_frame.p.to_numpy())
    test_frame.to_csv(output / 'tests.csv', index=False)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(test_frame.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'mining/sample_evidence/final')
    parser.add_argument('--sample', type=Path, default=ROOT / 'data/data/sample/balanced_sample.parquet')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'current')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    sample = pd.read_parquet(args.sample)
    prs = pd.read_parquet(args.evidence_dir / 'pull_requests.parquet')
    files = pd.read_parquet(args.evidence_dir / 'pull_request_files.parquet')
    if sample.duplicated(['repo_id', 'number']).any() or prs.duplicated(['repo_id', 'number']).any():
        raise ValueError('Duplicate sample or PR identity')
    if len(sample) != 2260 or sample.sample_arm.value_counts().to_dict() != {'agentic': 1130, 'human_candidate': 1130}:
        raise ValueError('Expected the official balanced sample')
    checked = prs.merge(sample[['repo_id', 'number', 'selection_hash', 'sample_arm']],
        on=['repo_id', 'number'], how='outer', validate='one_to_one', suffixes=('', '_sample'))
    if not (checked.selection_hash.eq(checked.selection_hash_sample) & checked.sample_arm.eq(checked.sample_arm_sample)).all():
        raise ValueError('Evidence differs from the official sample')
    merged = files.merge(prs[['repo_id', 'number', 'base_sha', 'head_sha', 'fetch_status']],
        on=['repo_id', 'number'], validate='many_to_one')
    merged = merged.merge(sample[['repo_id', 'number']], on=['repo_id', 'number'], validate='many_to_one')
    eligible = merged.status.isin(['modified', 'renamed']) & merged.fetch_status.eq('complete')
    eligible &= merged.filename.map(lambda name: lizard.get_reader_for(name) is not None)
    work = merged.loc[eligible].copy()
    work['old_path'] = work.previous_filename.where(work.status.eq('renamed'), work.filename).fillna(work.filename)
    columns = ['repo_id', 'number', 'sample_arm', 'repo_full_name', 'filename', 'old_path', 'base_sha', 'head_sha']
    output = args.output_dir
    for name in ['sources', 'files']:
        (output / name).mkdir(parents=True, exist_ok=True)
    merged.assign(eligible_pair=eligible).to_csv(output / 'file_eligibility.csv', index=False)
    signature = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.sample, args.evidence_dir/'pull_requests.parquet', args.evidence_dir/'pull_request_files.parquet']}
    signature['method'] = 'stored-base-head-paired-modified-files-v1-lizard-1.24.0'
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Incompatible input or method signature')
    manifest.write_text(json.dumps(signature, indent=2) + '\n')
    records = []
    print(f'Analyzing {len(work)} eligible file pairs across {work.groupby(["repo_id", "number"]).ngroups} PRs', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_file, row, output) for row in work[columns].to_dict('records')]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 500 == 0:
                print(f'{len(records)}/{len(work)} file pairs; failures={sum(r["status"] != "complete" for r in records)}', flush=True)
    frame = aggregate(records)
    audit = sorted(records, key=lambda r: (r['repo_id'], r['number'], r['filename']))
    (output / 'file_results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in audit))
    frame.to_csv(output / 'pr_deltas.csv', index=False)
    report(frame, output, sample)


if __name__ == '__main__':
    main()
