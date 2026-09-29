"""Regression checks for the frozen paper artifact and offline audit semantics."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mining/tools'))

from analysis.audits.replay import manual_summary, regex_summary, tradeoff_summary
from analysis.reproduce_paper import check_populations
from analysis.validate_artifact import revision, safe_path, validate
import publish_artifacts as publish


def test_official_frozen_populations_and_checksums():
    manifest = validate()
    assert manifest['catalog']['executed_patterns'] == 58
    controls = check_populations()
    assert controls['rq4_resolved_type'] == 1684
    assert controls['rq4_with_metric'] == 878


def test_occurrence_majority_cannot_be_replaced_with_pr_level_union():
    # Each provider supports a different occurrence: no occurrence has two votes.
    frame = pd.DataFrame([
        dict(repo_id=1, number=2, sample_arm='agentic', occurrence_id=f'o{i}',
             openai_valid=i == 0, gemini_valid=i == 1, qwen_valid=i == 2)
        for i in range(3)
    ])
    result = regex_summary(frame)
    assert result['true_positive'] == 0
    assert result['false_positive'] == 1
    frame.loc[0, 'gemini_valid'] = True
    assert regex_summary(frame)['true_positive'] == 1


def test_audit_rejects_ambiguous_or_duplicate_votes():
    row = dict(repo_id=1, number=2, sample_arm='agentic', openai_label='tradeoff',
               gemini_label='indeterminate', qwen_label='joint_improvement')
    with pytest.raises(ValueError, match='binary'):
        tradeoff_summary(pd.DataFrame([row]))
    row['gemini_label'] = 'tradeoff'
    with pytest.raises(ValueError, match='Duplicate'):
        tradeoff_summary(pd.DataFrame([row, row]))


def test_manual_audit_retains_disproportionate_sampling_weights():
    frame = pd.DataFrame([
        dict(repo_id=1, number=1, aidev_task_type='perf', manual_label='performance',
             sampling_weight=10.0, inclusion_probability=0.1, audit_stratum='positive'),
        dict(repo_id=1, number=2, aidev_task_type='fix', manual_label='performance',
             sampling_weight=100.0, inclusion_probability=0.01, audit_stratum='negative'),
    ])
    result = manual_summary(frame)
    assert result['raw_sample']['recall'] == 0.5
    assert result['weighted_population']['recall'] == pytest.approx(1 / 11)
    frame.loc[0, 'sampling_weight'] = 5
    with pytest.raises(ValueError, match='inverse inclusion'):
        manual_summary(frame)


def test_verifier_rejects_input_corruption(tmp_path):
    import json
    manifest = json.loads((ROOT / 'artifact_manifest.json').read_text())
    name = next(iter(manifest['inputs']))
    manifest['inputs'][name]['sha256'] = '0' * 64
    altered = tmp_path / 'manifest.json'
    altered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate(ROOT, altered)


def test_paths_reject_escape_and_external_symlinks(tmp_path):
    with pytest.raises(ValueError, match='Invalid'):
        safe_path(tmp_path, '../outside')
    (tmp_path / 'outside').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match='escapes'):
        safe_path(tmp_path, 'outside/example.csv')


def test_installed_dataset_validation_is_read_only(tmp_path, monkeypatch):
    target = tmp_path / 'data'
    target.mkdir()
    table = target / 'sample.csv'
    table.write_text('repo_id,number\n1,2\n')
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    artifact = publish.Artifact('unused.csv', 'data/sample.csv', 'test', 1, digest, 'csv')
    monkeypatch.setattr(publish, 'ARTIFACTS', (artifact,))
    before = {p.relative_to(tmp_path): p.stat().st_mtime_ns for p in tmp_path.rglob('*')}
    entries, paths = publish.validate_sources(tmp_path)
    after = {p.relative_to(tmp_path): p.stat().st_mtime_ns for p in tmp_path.rglob('*')}
    assert before == after
    assert entries[0]['rows'] == 1
    assert paths == [(table, 'data/sample.csv')]
    table.write_text('repo_id,number\n3,4\n')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        publish.validate_sources(tmp_path)


def test_inventory_matches_the_publication_allowlist():
    publish.check_inventory()


def test_archive_reproduction_does_not_require_git(monkeypatch, tmp_path):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError('git')

    monkeypatch.setattr(subprocess, 'run', missing_git)
    assert revision(tmp_path) is None


def test_check_only_works_outside_checkout_without_outputs(tmp_path):
    result = subprocess.run(
        [sys.executable, '-B', str(ROOT / 'analysis/reproduce_paper.py'), '--check-only'],
        cwd=tmp_path, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, check=True,
    )
    assert '"sample": 2260' in result.stdout
    assert list(tmp_path.iterdir()) == []
