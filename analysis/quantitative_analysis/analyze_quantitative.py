"""Update Section 4.1 outcomes and patch-size summaries from the archived sample."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import (binom, chi2_contingency, false_discovery_control, fisher_exact,
                         mannwhitneyu, norm)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ['repo_id', 'number']
ARMS = ['agentic', 'human_candidate']
SIZE_METRICS = ['additions', 'deletions', 'lines_changed', 'changed_files', 'commits_count']
ALPHA = 0.05
TEST_COLUMNS = ['contrast', 'method', 'n_agentic', 'n_human_candidate', 'statistic', 'p_value',
                'p_bh', 'effect_name', 'effect', 'effect_ci_low', 'effect_ci_high',
                'estimate_name', 'estimate', 'estimate_ci_low', 'estimate_ci_high',
                'odds_ratio', 'odds_ratio_ci_low', 'odds_ratio_ci_high']


def prepare(sample, evidence):
    for frame in (sample, evidence):
        if frame.duplicated(KEYS).any():
            raise ValueError('Duplicate immutable PR identity')
    identifiers = KEYS + ['sample_arm', 'selection_hash']
    cols = KEYS + ['sample_arm', 'selection_hash', 'snapshot_at', 'fetch_status',
                   'state', 'merged', 'created_at', 'merged_at', 'additions',
                   'deletions', 'changed_files', 'commits_count']
    joined = sample[identifiers].merge(evidence[cols], on=KEYS, how='outer',
        validate='one_to_one', suffixes=('', '_evidence'), indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Evidence identities differ from sample')
    if not (joined.sample_arm.eq(joined.sample_arm_evidence) &
            joined.selection_hash.eq(joined.selection_hash_evidence)).all():
        raise ValueError('Evidence identity/arm signature mismatch')
    if not joined.sample_arm.isin(ARMS).all():
        raise ValueError('Unknown study arm')
    known = joined.fetch_status.eq('complete')
    if not joined.loc[known, 'merged'].isin([True, False]).all():
        raise ValueError('Invalid observed merge flag')
    if not joined.loc[known, 'state'].isin(['open', 'closed']).all():
        raise ValueError('Invalid observed PR state')
    merged = known & joined.merged.eq(True)
    if (merged & joined.state.eq('open')).any():
        raise ValueError('Merged PR cannot be open')
    created = pd.to_datetime(joined.created_at, utc=True, errors='coerce')
    merged_at = pd.to_datetime(joined.merged_at, utc=True, errors='coerce')
    hours = (merged_at-created).dt.total_seconds()/3600
    valid_time = merged & hours.notna() & hours.ge(0)
    out = joined[KEYS + ['sample_arm']].copy()
    out['snapshot_at'] = joined.snapshot_at
    out['outcome_observed'] = known
    out['outcome'] = np.select([merged, known & joined.state.eq('open'), known],
                               ['merged', 'open', 'closed_unmerged'], default='unobserved')
    out['time_to_merge_hours'] = hours.where(valid_time)
    out['time_status'] = np.select([valid_time, merged, known],
        ['valid', 'missing_or_invalid', 'not_merged'], default='unobserved')
    for column in ['additions', 'deletions', 'changed_files', 'commits_count']:
        numeric = pd.to_numeric(joined[column], errors='coerce')
        if numeric.loc[known & numeric.notna()].lt(0).any():
            raise ValueError('Negative patch-size metadata')
        out[column] = numeric.where(known)
    out['lines_changed'] = out.additions + out.deletions
    return out.sort_values(KEYS).reset_index(drop=True)


def wilson_interval(successes, total):
    """Wilson score interval for a proportion, as a percentage."""
    if not total:
        return np.nan, np.nan
    z = norm.ppf(1-ALPHA/2)
    rate = successes/total
    center = (rate + z**2/(2*total))/(1 + z**2/total)
    half = z*np.sqrt(rate*(1-rate)/total + z**2/(4*total**2))/(1 + z**2/total)
    return float(100*(center-half)), float(100*(center+half))


def median_interval(values):
    """Distribution-free median interval from binomial order statistics."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    rank = int(binom.ppf(ALPHA/2, n, 0.5)) if n else 0
    if rank < 1:
        return np.nan, np.nan
    return float(ordered[rank-1]), float(ordered[n-rank])


def dominance(x, y):
    """Per-observation dominance scores and the tied-pair count behind Cliff's delta."""
    ordered_x, ordered_y = np.sort(x), np.sort(y)
    below = np.searchsorted(ordered_y, x, 'left')
    above = len(y) - np.searchsorted(ordered_y, x, 'right')
    y_below = np.searchsorted(ordered_x, y, 'left')
    y_above = len(x) - np.searchsorted(ordered_x, y, 'right')
    ties = len(x)*len(y) - int(below.sum()) - int(above.sum())
    return (below-above)/len(y), (y_above-y_below)/len(x), ties


def cliffs_delta(x, y):
    """Cliff's delta with the consistent asymmetric interval of Cliff (1993)."""
    per_x, per_y, ties = dominance(x, y)
    n1, n2 = len(x), len(y)
    delta = float(per_x.mean())
    if min(n1, n2) < 2:
        return delta, np.nan, np.nan
    spread = (n2**2*(n1-1)*per_x.var(ddof=1) + n1**2*(n2-1)*per_y.var(ddof=1)
              - (n1*n2 - ties - n1*n2*delta**2))
    sigma = float(np.sqrt(max(spread, 0)/(n1*n2*(n1-1)*(n2-1))))
    z = norm.ppf(1-ALPHA/2)
    half = z*sigma*np.sqrt((1-delta**2)**2 + z**2*sigma**2)
    denominator = 1 - delta**2 + z**2*sigma**2
    return delta, float((delta-delta**3-half)/denominator), float((delta-delta**3+half)/denominator)


def hodges_lehmann(x, y):
    """Median pairwise difference with the Moses normal-approximation interval."""
    differences = np.sort(np.subtract.outer(x, y).ravel())
    n1, n2 = len(x), len(y)
    rank = int(round(n1*n2/2 - norm.ppf(1-ALPHA/2)*np.sqrt(n1*n2*(n1+n2+1)/12)))
    rank = min(max(rank, 1), n1*n2)
    return float(np.median(differences)), float(differences[rank-1]), float(differences[n1*n2-rank])


def rank_contrast(name, x, y):
    """Two-sided Mann-Whitney comparison with both a standardized and a natural-unit magnitude."""
    u, p = mannwhitneyu(x, y, alternative='two-sided')
    delta, delta_low, delta_high = cliffs_delta(x, y)
    shift, shift_low, shift_high = hodges_lehmann(x, y)
    return dict(contrast=name, method='Mann-Whitney U, two-sided', n_agentic=len(x),
        n_human_candidate=len(y), statistic=float(u), p_value=float(p),
        effect_name='Cliffs delta', effect=delta, effect_ci_low=delta_low, effect_ci_high=delta_high,
        estimate_name='Hodges-Lehmann shift, agentic minus human_candidate',
        estimate=shift, estimate_ci_low=shift_low, estimate_ci_high=shift_high)


def summarize(frame):
    summaries = []
    for arm in ARMS:
        all_rows = frame[frame.sample_arm.eq(arm)]
        known = all_rows[all_rows.outcome_observed]
        times = known.time_to_merge_hours.dropna()
        row = dict(sample_arm=arm, sample_n=len(all_rows), observed_n=len(known),
            unobserved_n=len(all_rows)-len(known), merged_n=int(known.outcome.eq('merged').sum()),
            open_n=int(known.outcome.eq('open').sum()),
            closed_unmerged_n=int(known.outcome.eq('closed_unmerged').sum()),
            time_n=len(times), time_median_hours=times.median(), time_mean_hours=times.mean(),
            invalid_merged_times=int(known.time_status.eq('missing_or_invalid').sum()))
        row['time_median_ci_low'], row['time_median_ci_high'] = median_interval(times)
        row['merge_rate_pct'] = 100*row['merged_n']/len(known) if len(known) else np.nan
        row['merge_rate_ci_low'], row['merge_rate_ci_high'] = wilson_interval(row['merged_n'], len(known))
        for column in SIZE_METRICS:
            values = known[column].dropna()
            row[f'{column}_n'] = len(values)
            row[f'{column}_median'] = values.median()
            row[f'{column}_median_ci_low'], row[f'{column}_median_ci_high'] = median_interval(values)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    tests = []
    ct = np.array([[r.merged_n, r.observed_n-r.merged_n] for r in summary.itertuples()])
    if (ct.sum(axis=0)>0).all() and (ct.sum(axis=1)>0).all():
        chi, p, _, expected = chi2_contingency(ct, correction=True)
        if expected.min() < 5:
            _, p = fisher_exact(ct)
            method = 'Fisher exact'
        else:
            method = 'Pearson chi-square with Yates correction'
        z = norm.ppf(1-ALPHA/2)
        rates = ct[:, 0]/ct.sum(axis=1)
        # Natural-unit magnitude: Wald interval on the merge-rate difference in percentage points.
        difference = 100*(rates[0]-rates[1])
        half = 100*z*np.sqrt((rates*(1-rates)/ct.sum(axis=1)).sum())
        row = dict(contrast='merge_rate', method=method, n_agentic=int(ct[0].sum()),
            n_human_candidate=int(ct[1].sum()), statistic=float(chi), p_value=float(p),
            effect_name='Cramers V (Yates)', effect=float(np.sqrt(chi/ct.sum())),
            estimate_name='Merge-rate difference in percentage points, agentic minus human_candidate',
            estimate=float(difference), estimate_ci_low=float(difference-half),
            estimate_ci_high=float(difference+half))
        if ct.min() > 0:
            log_odds = float(np.log(ct[0, 0]*ct[1, 1]/(ct[0, 1]*ct[1, 0])))
            error = float(np.sqrt((1/ct).sum()))
            row.update(odds_ratio=float(np.exp(log_odds)),
                odds_ratio_ci_low=float(np.exp(log_odds-z*error)),
                odds_ratio_ci_high=float(np.exp(log_odds+z*error)))
        tests.append(row)
    for metric in ['time_to_merge_hours'] + SIZE_METRICS:
        values = [frame.loc[frame.sample_arm.eq(arm), metric].dropna().to_numpy() for arm in ARMS]
        if all(len(arm_values) > 0 for arm_values in values):
            tests.append(rank_contrast(metric, *values))
    tests = pd.DataFrame(tests).reindex(columns=TEST_COLUMNS)
    if len(tests):
        # One exploratory characterization family: merge rate, elapsed time, and patch size.
        tests['p_bh'] = np.nan
        valid = tests.p_value.notna()
        if valid.any():
            tests.loc[valid, 'p_bh'] = false_discovery_control(tests.loc[valid, 'p_value'].to_numpy())
    return summary, tests


def plot(frame, summary, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.2), layout='constrained')
    colors = ['#9BBCE8', '#B9DAB9']
    labels = ['Agentic', 'Human-authored']
    bars = axes[0].bar(labels, summary.merge_rate_pct, color=colors, width=.5)
    axes[0].bar_label(bars, labels=[f'{r.merge_rate_pct:.1f}% ({r.merged_n}/{r.observed_n})'
        for r in summary.itertuples()], padding=4, fontsize=9)
    axes[0].set_ylim(0,100); axes[0].set_ylabel('Merged among observed PRs (%)')
    for arm, color, label in zip(ARMS, colors, labels):
        values = np.sort(frame.loc[frame.sample_arm.eq(arm), 'time_to_merge_hours'].dropna())
        if len(values):
            axes[1].step(values, np.arange(1,len(values)+1)/len(values), where='post',
                color=color, label=f'{label} (n={len(values)})')
    axes[1].set_xscale('symlog', linthresh=1)
    axes[1].set_xlabel('Hours from PR creation to merge (symlog scale)')
    axes[1].set_ylabel('Cumulative share of merged PRs')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(target, bbox_inches='tight'); plt.close(fig)


def upstream_sources():
    """Hashes of the parquet inputs behind the published per-PR measurements.

    Reproducing from the compact CSV reads that CSV, not the snapshot it was
    exported from, so the original measurement inputs are carried forward from
    the committed manifest rather than dropped from the provenance record.
    """
    recorded = HERE/'results/manifest.json'
    if not recorded.exists():
        return {}
    previous = json.loads(recorded.read_text())
    return previous.get('measurement_source_sha256') or previous.get('source_sha256', {})


def write_outputs(frame, out, provenance):
    out.mkdir(parents=True, exist_ok=True)
    summary, tests = summarize(frame)
    frame.to_csv(out/'pr_quantitative.csv', index=False)
    summary.to_csv(out/'outcomes_summary.csv', index=False)
    tests.to_csv(out/'outcomes_tests.csv', index=False)
    plot(frame, summary, out/'merge_rate_and_time.pdf')
    indexed = tests.set_index('contrast')
    def inference(contrast):
        if contrast is None or contrast not in indexed.index:
            return ['---', '---']
        row = indexed.loc[contrast]
        symbol = 'V' if str(row.effect_name).startswith('Cramers') else r'\delta'
        return ['$<0.001$' if row.p_bh < .001 else f'{row.p_bh:.3f}',
                f'${symbol}={row.effect:.3f}$']
    table = [r'\begin{table}[htbp]', r'\centering', r'\small',
        r"\caption{Quantitative outcomes and patch size at the archived snapshot. Merge rates use observed PRs; elapsed time uses merged PRs only. Patch-size entries are medians. $q$ is BH-adjusted across the seven characterization tests; the effect size is Cram\'er's $V$ for the merge-rate contingency table and Cliff's $\delta$ (agentic minus human-authored) for the Mann--Whitney comparisons. Per-arm intervals, Hodges--Lehmann shifts, and the merge-rate odds ratio are reported in the replication package.}",
        r'\label{tab:quantitative-outcomes}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Measure & Agentic & Human-authored & $q$ & Effect \\', r'\midrule']
    for column,label,fmt,contrast in [('sample_n','Selected PRs',',.0f',None),
        ('observed_n','Observed PRs',',.0f',None),('merged_n','Merged PRs',',.0f',None),
        ('merge_rate_pct',r'Merge rate (\%)','.1f','merge_rate'),
        ('time_median_hours','Median time to merge (h)','.2f','time_to_merge_hours'),
        ('lines_changed_median','Added + deleted lines','.1f','lines_changed'),
        ('changed_files_median','Changed files','.0f','changed_files'),
        ('commits_count_median','Commits','.0f','commits_count')]:
        table.append(' & '.join([label]+[format(float(value),fmt) for value in summary[column]]
                                +inference(contrast))+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (out/'quantitative_outcomes.tex').write_text('\n'.join(table))
    snapshots = frame.snapshot_at.dropna().unique().tolist()
    manifest = dict(snapshot_at=snapshots, source_sha256=provenance,
        measurement_source_sha256=upstream_sources(),
        sample_rows=len(frame), observed_rows=int(frame.outcome_observed.sum()),
        outcome_method='Observed archived PR state; unknown outcomes excluded from merge-rate denominator',
        time_method='Elapsed hours, creation to merge; nonnegative valid times among merged PRs only',
        test_family='Seven exploratory characterization tests in one Benjamini-Hochberg family; raw and adjusted p-values; no repository-cluster or censoring adjustment',
        interval_method='All intervals are two-sided 95%: Wilson for each merge rate, binomial order statistics for each median, Cliff (1993) consistent asymmetric for Cliffs delta, Moses normal approximation for the Hodges-Lehmann shift, and Wald for the merge-rate difference and the log odds ratio')
    manifest['software_versions'] = {name: version(name) for name in ['pandas','numpy','scipy','matplotlib','lizard']}
    manifest['source_code_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__).resolve(), HERE.parent/'maintainability/run_maintainability.py']}
    manifest['files'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir()) if path.name in
        ['pr_quantitative.csv','outcomes_summary.csv','outcomes_tests.csv','merge_rate_and_time.pdf','quantitative_outcomes.tex']}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.to_string(index=False)); print(tests.to_string(index=False))


def structural_outputs(frame, path, out):
    """Reproduce structural summaries using published per-PR before/after values."""
    sys.path.insert(0, str(HERE.parent/'maintainability'))
    from run_maintainability import report
    deltas=pd.read_csv(path, float_precision='round_trip')
    checked=deltas.merge(frame[KEYS+['sample_arm']], on=KEYS, how='left',
        suffixes=('', '_sample'), validate='many_to_one')
    if checked.duplicated(KEYS+['metric']).any() or not checked.sample_arm.eq(checked.sample_arm_sample).all():
        raise ValueError('Structural identities differ from the analysis sample')
    folder=out/'structural'
    folder.mkdir(parents=True,exist_ok=True)
    deltas.to_csv(folder/'pr_deltas.csv',index=False)
    report(deltas,folder,frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=ROOT/'data/data/sample/balanced_sample.parquet')
    parser.add_argument('--evidence', type=Path, default=ROOT/'mining/sample_evidence/final/pull_requests.parquet')
    parser.add_argument('--labels', type=Path, help='Reproduce from the compact derived CSV instead of full inputs')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results')
    parser.add_argument('--structural-deltas', type=Path, default=HERE/'results/structural/pr_deltas.csv')
    args = parser.parse_args()
    if args.labels:
        frame = pd.read_csv(args.labels, float_precision='round_trip')
        sources = [args.labels]
    else:
        frame = prepare(pd.read_parquet(args.sample), pd.read_parquet(args.evidence))
        sources = [args.sample, args.evidence]
    if len(frame)!=2260 or frame.duplicated(KEYS).any():
        raise ValueError('Expected 2,260 unique PRs')
    if frame.sample_arm.value_counts().to_dict() != {'agentic':1130, 'human_candidate':1130}:
        raise ValueError('Unexpected official sample arms')
    if frame.snapshot_at.dropna().nunique()!=1:
        raise ValueError('Expected a single snapshot')
    if not frame.outcome.isin(['merged','open','closed_unmerged','unobserved']).all():
        raise ValueError('Unexpected outcome label')
    if not frame.outcome_observed.eq(frame.outcome.ne('unobserved')).all():
        raise ValueError('Inconsistent observed-outcome flag')
    valid_time = frame.time_to_merge_hours.notna()
    if not (frame.loc[valid_time,'outcome'].eq('merged') &
            frame.loc[valid_time,'time_to_merge_hours'].ge(0)).all():
        raise ValueError('Elapsed time must be nonnegative and belong to a merged PR')
    hashes = {str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else path.name:
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    write_outputs(frame, args.output_dir, hashes)
    structural_outputs(frame,args.structural_deltas,args.output_dir)


if __name__ == '__main__':
    main()
