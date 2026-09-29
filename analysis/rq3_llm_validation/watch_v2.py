"""Synchronize the overnight Qwen run and finalize all three-provider results."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis.rq3_llm_validation import experiment as v1
from analysis.rq3_llm_validation import v2
from analysis.rq3_llm_validation.v2_consensus import build, ready_tasks

REMOTE_ROOT = "/scratch/gautschi/rcalvome/EMSE-perf-pr-study"


def command(args: list[str], timeout: int = 90) -> dict:
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[-2000:]}
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}


def snapshot_code(output: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output / "source_runtime" / stamp
    files = [*Path(__file__).parent.glob("*.py"),
             v2.ROOT / "analysis/rq3_pattern_and_validation/extract_metrics.py",
             v2.ROOT / "analysis/rq3_pattern_and_validation/metric_patterns.py",
             v2.ROOT / "analysis/rq3_pattern_and_validation/run_current.py",
             v2.ROOT / "analysis/run_qwen.py", v2.ROOT / "analysis/run_qwen_gautschi.sbatch"]
    hashes = {}
    for path in files:
        relative = path.relative_to(v2.ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = v1.sha256_file(path)
    v1.atomic_write_json(destination / "metadata.json", {
        "python": platform.python_version(), "source_sha256": hashes,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("pandas", "pyarrow", "pydantic", "openai", "google-genai", "tiktoken")}})


def watch(output: Path, job_id: str, interval: int = 60, timeout_hours: int = 18) -> dict:
    if not job_id.isdecimal():
        raise ValueError("Slurm job ID must be numeric.")
    v2.load_requests(output)
    snapshot_code(output)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    uploaded = {}
    finalized = ()
    v1.atomic_write_json(output / "execution.json", {
        "job_id": job_id, "host": "gautschi", "remote_root": REMOTE_ROOT,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "local_services": ["rq3-v2-openai.service", "rq3-v2-gemini.service", "rq3-v2-watch.service"],
        "mode": "resumable synchronous APIs plus Slurm Qwen; four smoke cases per provider",
        "gemini_schema_note": "Compact semantic schema; large array/ID constraints checked locally after provider rejection",
        "citation_validation": "Markdown/whitespace alignment and recorded quote-only repair; label decisions fixed",
        "slurm_backfill": {"time_limit": "03:00:00", "time_min": "01:00:00"},
        "retry_policy": "Invalid responses receive a recorded validation diagnostic; core evidence and policy remain fixed",
        "max_attempts": 6,
        "tests": {"rq3_v1_v2": "43 passed", "full_suite": "280 passed, 1 unrelated quantitative-analysis NaN failure at initial run"}})
    while time.monotonic() - started < timeout_hours * 3600:
        remote = command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "gautschi",
                          f"sacct -j {job_id} --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed"])
        sync = command(["rsync", "-az", "--include=*/", "--include=*.json", "--include=*.parquet", "--exclude=*",
                        f"gautschi:{REMOTE_ROOT}/analysis/rq3_llm_validation/generated_v2/qwen", str(output) + "/"], 120)
        command(["rsync", "-az", f"gautschi:{REMOTE_ROOT}/logs/qwen-{job_id}.out",
                 f"gautschi:{REMOTE_ROOT}/logs/qwen-{job_id}.err", str(logs) + "/"])
        providers = {}
        for provider in v2.MODELS:
            path = output / provider / "run_state.json"
            providers[provider] = json.loads(path.read_text()) if path.exists() else {"status": "waiting"}
            cloud_ready = providers[provider].get("tasks", {}).get("regex_audit", {}).get("classified") == 88
            if provider in {"openai", "gemini"} and cloud_ready and uploaded.get(provider) != providers[provider].get("updated_at"):
                upload = command(["rsync", "-az", "--exclude=raw/", "--exclude=*.parquet", "--exclude=runner.lock",
                                  str(output / provider),
                                  f"gautschi:{REMOTE_ROOT}/analysis/rq3_llm_validation/generated_v2/"], 120)
                if upload["returncode"] == 0:
                    uploaded[provider] = providers[provider].get("updated_at")
        state = {"updated_at": datetime.now(timezone.utc).isoformat(), "job_id": job_id,
                 "slurm": remote, "qwen_sync_returncode": sync["returncode"], "providers": providers,
                 "cloud_results_backed_up_on_gautschi": sorted(uploaded),
                 "status": "running"}
        if any(p.get("status") == "failed" for p in providers.values()):
            state["status"] = "provider_failed"
        if any(p.get("status") == "blocked_billing" for p in providers.values()):
            state["status"] = "blocked_billing"
        ready = ready_tasks(output)
        if ready and ready != finalized:
            try:
                state["results"] = build(output, ready)
            except Exception as error:
                state.update(status="finalization_error", error=f"{type(error).__name__}: {error}")
                v1.atomic_write_json(output / "overnight_status.json", state)
                raise
            finalized = ready
        state["finalized_tasks"] = list(finalized)
        if set(finalized) == set(v1.TASKS):
            state["status"] = "completed"
            v1.atomic_write_json(output / "overnight_status.json", state)
            print(json.dumps(state, indent=2), flush=True)
            return state
        v1.atomic_write_json(output / "overnight_status.json", state)
        print(json.dumps({"updated_at": state["updated_at"], "slurm": remote["stdout"],
                          "providers": {p: value.get("tasks", value) for p, value in providers.items()}}), flush=True)
        time.sleep(interval)
    state["status"] = "monitor_timeout"
    v1.atomic_write_json(output / "overnight_status.json", state)
    raise TimeoutError("Overnight monitor deadline reached; inspect overnight_status.json and provider logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=v2.DEFAULT_OUTPUT)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    watch(args.output_dir, args.job_id, args.interval)
