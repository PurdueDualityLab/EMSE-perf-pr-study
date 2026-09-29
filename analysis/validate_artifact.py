"""Validate frozen reproduction inputs, without network access or file writes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "analysis/classification_labels/rq1_labels.csv",
    "analysis/classification_labels/rq2_labels.csv",
    "analysis/classification_labels/rq2_exclusions.csv",
    "analysis/classification_labels/rq3_labels.csv",
    "analysis/rq1_optimization_patterns/catalog/original_optimization_catalog.csv",
    "analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv",
    "analysis/rq2_validation/temporal/pr_created_at.csv",
    "analysis/quantitative_analysis/results/pr_quantitative.csv",
    "analysis/quantitative_analysis/results/structural/pr_deltas.csv",
    "analysis/quantitative_analysis/results/structural/analysis_deltas.csv",
    "analysis/quantitative_analysis/results/structural/summary.csv",
    "analysis/quantitative_analysis/results/structural/tests.csv",
    "analysis/rq3_pattern_and_validation/catalog_expected_dims.csv",
    "analysis/artifact_inputs/agent_distribution.csv",
    "analysis/artifact_inputs/rq4_metadata.csv",
    "analysis/artifact_inputs/regex_occurrence_votes.csv",
    "analysis/artifact_inputs/tradeoff_votes.csv",
    "analysis/artifact_inputs/manual_classifier_audit.csv",
    "analysis/quantitative_analysis/results/outcomes_summary.csv",
    "analysis/quantitative_analysis/results/outcomes_tests.csv",
    "analysis/rq1_optimization_patterns/structural_by_category/results/category_structural_tests.csv",
    "analysis/rq2_validation/temporal/presence_by_quarter.csv",
    "analysis/rq2_validation/temporal/primary_type_by_quarter.csv",
    "analysis/rq3_pattern_and_validation/results/rq3_tests.csv",
    "analysis/rq3_pattern_and_validation/results/tables/T6_1_alignment_permutation.csv",
    "analysis/rq3_pattern_and_validation/results/tables/T7_4_category_x_author_interaction.csv",
]
CODE = [
    "analysis/reproduce_paper.py", "analysis/validate_artifact.py",
    "analysis/audits/replay.py", "analysis/make_paper_comparison_figures.py",
    "analysis/make_paper_result_tables.py", "analysis/make_agent_distribution_figure.py",
    "analysis/quantitative_analysis/analyze_quantitative.py",
    "analysis/maintainability/run_maintainability.py",
    "analysis/rq1_optimization_patterns/analyze_rq1_comparison.py",
    "analysis/rq1_optimization_patterns/structural_by_category/analyze_rq1_structural_by_category.py",
    "analysis/rq2_validation/analyze_rq2_comparison.py",
    "analysis/rq2_validation/temporal/analyze_rq2_temporal.py",
    "analysis/rq3_pattern_and_validation/rq3_statistics.py",
    "analysis/rq3_pattern_and_validation/metric_patterns.py",
    "analysis/rq3_pattern_and_validation/metric_alignment.py",
    "analysis/rq3_pattern_and_validation/code_smells_by_author.py",
    "analysis/rq3_pattern_and_validation/make_figures.py",
    "requirements.txt", "analysis/quantitative_analysis/requirements.txt",
    "analysis/maintainability/requirements.txt",
    "analysis/constraints-python312.txt",
    "Dockerfile.paper",
    "analysis/artifact_inputs/manifest.json",
    "analysis/rq1_optimization_patterns/catalog/provenance.json",
    "analysis/export_artifact_inputs.py", "analysis/build_classification_labels.py",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, name):
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f"Invalid artifact path: {name}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes the root: {name}")
    return result


def revision(path):
    try:
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_manifest(root):
    files = {}
    for name in INPUTS:
        path = safe_path(root, name)
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = sum(1 for _ in reader)
        files[name] = {"sha256": sha256(path), "rows": rows, "columns": columns}
    return {
        "format_version": 1,
        "title": "How Do Coding Agents Optimize Software and Report Performance Validation? A Large-Scale Empirical Study of Open-Source Pull Requests",
        "python": "3.12",
        "code_base_revision": revision(root),
        "code_version_note": "Source digests identify the prepared code; code_base_revision is the commit on which uncommitted artifact preparation was based. The final release commit identifies the full version.",
        "data_revision": revision(root / 'data'),
        "manuscript_revision": revision(root / 'report'),
        "inputs": files,
        "source_sha256": {name: sha256(safe_path(root, name)) for name in CODE},
        "catalog": {"executed_patterns": 58, "categories": 9, "submitted_manuscript_patterns": 59,
                    "explanation": "See analysis/rq1_optimization_patterns/catalog/README.md for the documented consolidation and manuscript history."},
    }


def validate(root=ROOT, manifest_path=None):
    manifest_path = manifest_path or root / 'artifact_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported artifact manifest version')
    if set(manifest['inputs']) != set(INPUTS) or set(manifest['source_sha256']) != set(CODE):
        raise ValueError('Manifest does not cover the expected input and source files')
    for name, expected in manifest['inputs'].items():
        path = safe_path(root, name)
        if sha256(path) != expected['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected['columns'] or sum(1 for _ in reader) != expected['rows']:
                raise ValueError(f'Input schema or row-count mismatch: {name}')
    for name, expected in manifest['source_sha256'].items():
        if sha256(safe_path(root, name)) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifact_manifest.json')
    parser.add_argument('--write-manifest', action='store_true', help='Maintainer action: explicitly freeze current inputs and code digests.')
    args = parser.parse_args()
    if args.write_manifest:
        args.manifest.write_text(json.dumps(build_manifest(ROOT), indent=2, sort_keys=True) + '\n')
    manifest = validate(ROOT, args.manifest)
    print(json.dumps({'validated_inputs': len(manifest['inputs']), 'validated_sources': len(manifest['source_sha256']),
                      'network_requests': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
