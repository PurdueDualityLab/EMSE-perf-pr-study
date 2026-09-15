from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'analysis/quantitative_analysis'))
from analyze_quantitative import prepare, summarize


def inputs():
    sample = pd.DataFrame(dict(repo_id=[1,2,3,4], number=[1,1,1,1],
        sample_arm=['agentic','agentic','human_candidate','human_candidate'], selection_hash=['a','b','c','d']))
    evidence = sample.assign(snapshot_at='2026-08-27', fetch_status=['complete','not_found','complete','complete'],
        state=['closed',None,'closed','open'], merged=[True,None,False,False],
        created_at='2025-01-01T00:00:00Z', merged_at=['2025-01-01T01:00:00Z',None,None,None],
        additions=2,deletions=1,changed_files=1,commits_count=1)
    return sample,evidence


def test_unavailable_outcomes_are_not_nonmerges():
    frame=prepare(*inputs()); summary,_=summarize(frame)
    agent=summary.set_index('sample_arm').loc['agentic']
    assert agent.sample_n==2 and agent.observed_n==1
    assert agent.merge_rate_pct==100
    assert frame.loc[1,'outcome']=='unobserved'
    assert pd.isna(frame.loc[1,'lines_changed'])
    assert frame.time_to_merge_hours.dropna().tolist()==[1]


def test_negative_duration_is_explicitly_excluded():
    sample,evidence=inputs(); evidence.loc[0,'merged_at']='2024-01-01T00:00:00Z'
    frame=prepare(sample,evidence)
    assert frame.loc[0,'outcome']=='merged'
    assert frame.loc[0,'time_status']=='missing_or_invalid'
    assert pd.isna(frame.loc[0,'time_to_merge_hours'])


def test_identity_mismatch_rejected():
    sample,evidence=inputs(); evidence.loc[0,'selection_hash']='different'
    with pytest.raises(ValueError,match='signature mismatch'):
        prepare(sample,evidence)
