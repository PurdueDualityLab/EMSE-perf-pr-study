"""Reproduce journal RQ1–RQ4 tables, statistics, figures, and audits offline."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.validate_artifact import ROOT, validate, revision

FIGURES = [
    'agent_sample_distribution.pdf', 'merge_rate_and_time.pdf',
    'structural_change_distributions.pdf', 'optimization_pattern_distribution.pdf',
    'validation_evidence_types.pdf', 'validation_over_time.pdf',
    'metric_count_distribution.pdf', 'metric_frequency.pdf', 'metric_profile_by_category.pdf',
]

REFERENCE_RESULTS = {
    'analysis/quantitative_analysis/results/outcomes_summary.csv': 'rq1/outcomes_summary.csv',
    'analysis/quantitative_analysis/results/outcomes_tests.csv': 'rq1/outcomes_tests.csv',
    'analysis/quantitative_analysis/results/structural/tests.csv': 'rq1/structural/tests.csv',
    'analysis/rq1_optimization_patterns/structural_by_category/results/category_structural_tests.csv': 'rq2/structure/category_structural_tests.csv',
    'analysis/rq2_validation/temporal/presence_by_quarter.csv': 'rq3/temporal/presence_by_quarter.csv',
    'analysis/rq2_validation/temporal/primary_type_by_quarter.csv': 'rq3/temporal/primary_type_by_quarter.csv',
    'analysis/rq3_pattern_and_validation/results/rq3_tests.csv': 'rq4/metrics/rq3_tests.csv',
    'analysis/rq3_pattern_and_validation/results/tables/T6_1_alignment_permutation.csv': 'rq4/correspondence/tables/T6_1_alignment_permutation.csv',
    'analysis/rq3_pattern_and_validation/results/tables/T7_4_category_x_author_interaction.csv': 'rq4/code-smells/tables/T7_4_category_x_author_interaction.csv',
}


def check_results(output):
    for source, generated in REFERENCE_RESULTS.items():
        expected = pd.read_csv(ROOT / source)
        actual = pd.read_csv(output / generated)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, rtol=1e-6, atol=1e-7, obj=generated)
    return len(REFERENCE_RESULTS)


def check_populations(root=ROOT):
    labels = root / 'analysis/classification_labels'
    rq2 = pd.read_csv(labels / 'rq1_labels.csv')
    rq3 = pd.read_csv(labels / 'rq2_labels.csv')
    rq4 = pd.read_csv(labels / 'rq3_labels.csv')
    exclusions = pd.read_csv(labels / 'rq2_exclusions.csv')
    identities = lambda frame: set(frame[['repo_id', 'number']].itertuples(index=False, name=None))
    for frame in (rq2, rq3, rq4, exclusions):
        if frame.duplicated(['repo_id', 'number']).any():
            raise ValueError('Duplicate PR identity in frozen inputs')
    if identities(rq2) != identities(rq3) | identities(exclusions) or identities(rq3) & identities(exclusions):
        raise ValueError('Validation inputs and exclusions do not reconstruct the sample')
    if identities(rq4) != identities(rq2[rq2.included_in_analysis]) & identities(rq3):
        raise ValueError('Joint metric identities differ from resolved patterns and available validation labels')
    catalog = pd.read_csv(root / 'analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv')
    allowed = set(zip(catalog['High-level Pattern'], catalog['Sub pattern']))
    resolved = rq2[rq2.included_in_analysis]
    if not set(zip(resolved.consensus_high_level_pattern, resolved.consensus_sub_pattern)) <= allowed:
        raise ValueError('Consensus labels are outside the executed catalog')
    actual = {
        'sample': len(rq2), 'rq2_resolved': int(rq2.included_in_analysis.sum()),
        'rq3_available': len(rq3), 'rq3_positive': int(rq3.consensus_validation_present.sum()),
        'rq3_resolved_type': int(rq3.included_in_stage2_analysis.sum()),
        'rq4_joint': len(rq4), 'rq4_positive': int(rq4.in_metric_layer.sum()),
        'rq4_resolved_type': int(rq4.in_type_layer.sum()),
        'rq4_with_metric': int((rq4.in_metric_layer & rq4.n_dims.gt(0)).sum()),
        'executed_patterns': len(catalog),
    }
    expected = dict(sample=2260, rq2_resolved=2083, rq3_available=2258, rq3_positive=1839,
                    rq3_resolved_type=1819, rq4_joint=2081, rq4_positive=1699,
                    rq4_resolved_type=1684, rq4_with_metric=878, executed_patterns=58)
    if actual != expected or rq2.sample_arm.value_counts().to_dict() != {'agentic': 1130, 'human_candidate': 1130}:
        raise ValueError(f'Unexpected frozen populations: {actual}')
    return actual


def run(output, paper_dir=None):
    manifest = validate()
    controls = check_populations()
    output = output.resolve()
    if any((ROOT / name).resolve().is_relative_to(output) for name in manifest['inputs']):
        raise ValueError('Output directory must not contain frozen inputs')
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    labels = ROOT / 'analysis/classification_labels'
    inputs = ROOT / 'analysis/artifact_inputs'
    quantitative = ROOT / 'analysis/quantitative_analysis/results'
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONHASHSEED='0', SOURCE_DATE_EPOCH='1780272000')
    commands = []

    def stage(name, script, *arguments):
        def execution_arg(value):
            if isinstance(value, Path) and value.is_absolute() and value.is_relative_to(ROOT):
                return str(value.relative_to(ROOT))
            return str(value)

        def recorded_arg(value):
            if isinstance(value, Path):
                path = value.resolve()
                if path.is_relative_to(output):
                    return '<output>/' + path.relative_to(output).as_posix()
                if path.is_relative_to(ROOT):
                    return path.relative_to(ROOT).as_posix()
                return path.name
            return str(value)

        command = [sys.executable, str(ROOT / script), *map(execution_arg, arguments)]
        print(f'Running {name} ...', flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        log = logs / f'{name}.log'
        log.write_text(result.stdout + '\n' + result.stderr)
        commands.append({'stage': name, 'script': script, 'arguments': list(map(recorded_arg, arguments))})
        if result.returncode:
            raise RuntimeError(f'{name} failed (exit {result.returncode}); inspect {log}')

    stage('rq1', 'analysis/quantitative_analysis/analyze_quantitative.py',
          '--labels', quantitative / 'pr_quantitative.csv', '--output-dir', output / 'rq1')
    stage('rq2-patterns', 'analysis/rq1_optimization_patterns/analyze_rq1_comparison.py',
          '--consensus', labels / 'rq1_labels.csv', '--output-dir', output / 'rq2/patterns')
    stage('rq2-structure', 'analysis/rq1_optimization_patterns/structural_by_category/analyze_rq1_structural_by_category.py',
          '--labels', labels / 'rq1_labels.csv', '--deltas', quantitative / 'structural/pr_deltas.csv',
          '--output-dir', output / 'rq2/structure')
    stage('rq3-evidence', 'analysis/rq2_validation/analyze_rq2_comparison.py',
          '--consensus', labels / 'rq2_labels.csv', '--output-dir', output / 'rq3/evidence')
    stage('rq3-temporal', 'analysis/rq2_validation/temporal/analyze_rq2_temporal.py',
          '--labels', labels / 'rq2_labels.csv', '--output-dir', output / 'rq3/temporal')
    stage('rq4-metrics', 'analysis/rq3_pattern_and_validation/rq3_statistics.py', '--current',
          '--data', labels / 'rq3_labels.csv', '--metadata', inputs / 'rq4_metadata.csv',
          '--output-dir', output / 'rq4/metrics')
    stage('rq4-correspondence', 'analysis/rq3_pattern_and_validation/metric_alignment.py',
          '--labels', labels / 'rq3_labels.csv', '--output-dir', output / 'rq4/correspondence',
          '--figure-dir', output / 'rq4/correspondence/figures')
    stage('rq4-code-smells', 'analysis/rq3_pattern_and_validation/code_smells_by_author.py',
          '--labels', labels / 'rq3_labels.csv', '--output-dir', output / 'rq4/code-smells')
    stage('audits', 'analysis/audits/replay.py', '--input-dir', inputs, '--output-dir', output / 'audits')
    stage('comparison-figures', 'analysis/make_paper_comparison_figures.py', '--output-dir', figures)
    stage('metric-figures', 'analysis/rq3_pattern_and_validation/make_figures.py', '--current', '--paper-names',
          '--data', labels / 'rq3_labels.csv', '--output-dir', figures)
    stage('agent-figure', 'analysis/make_agent_distribution_figure.py', '--counts', inputs / 'agent_distribution.csv',
          '--chart', 'pie', '--cohort', 'sample', '--output', figures / 'agent_sample_distribution.pdf')
    stage('supporting-tables', 'analysis/make_paper_result_tables.py', '--output-dir', output / 'tables')
    shutil.copyfile(output / 'rq1/merge_rate_and_time.pdf', figures / 'merge_rate_and_time.pdf')
    shutil.copyfile(output / 'rq3/temporal/rq2_temporal.pdf', figures / 'validation_over_time.pdf')
    missing = [name for name in FIGURES if not (figures / name).is_file()]
    if missing:
        raise ValueError(f'Missing paper figures: {missing}')
    copied = []
    if paper_dir is not None:
        for name in ('perf_pr_flow.pdf', 'Methodology.pdf'):
            shutil.copyfile(paper_dir / 'figures' / name, figures / name)
            copied.append(name)
    audit = json.loads((output / 'audits/audit_summary.json').read_text())
    if (audit['metric_precision']['true_positive'], audit['metric_precision']['false_positive']) != (77, 11):
        raise ValueError('Metric audit no longer reproduces the submitted 77/11 result')
    if audit['tradeoffs']['classes'] != {'tradeoff': 17, 'joint_improvement': 12}:
        raise ValueError('Trade-off votes no longer reproduce the selected initial run')
    compared_results = check_results(output)
    result = {
        'controls': controls, 'network_requests_required': 0,
        'code_revision': revision(ROOT), 'input_manifest_sha256': hashlib.sha256((ROOT / 'artifact_manifest.json').read_bytes()).hexdigest(),
        'software': {name: version(name) for name in ('pandas', 'numpy', 'scipy', 'matplotlib', 'statsmodels', 'lizard')},
        'commands': commands, 'generated_paper_figures': FIGURES, 'copied_manual_diagrams': copied,
        'reference_result_tables_matched': compared_results,
        'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'reproduction_manifest.json' and 'logs' not in p.parts},
    }
    (output / 'reproduction_manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'Completed: {len(FIGURES)} quantitative paper figures; results in {output}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduction')
    parser.add_argument('--paper-dir', type=Path, help='Optionally copy the two manually authored diagrams from a report checkout.')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and populations without writing outputs.')
    args = parser.parse_args()
    if args.check_only:
        validate()
        print(json.dumps(check_populations(), sort_keys=True))
    else:
        run(args.output_dir, args.paper_dir)


if __name__ == '__main__':
    main()
