"""Monitor native cloud Batches, synchronize Qwen, and finalize binary consensus."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis.rq3_llm_validation import binary, run_binary
from analysis.rq3_llm_validation import experiment as v1
from analysis.rq3_llm_validation.watch_v2 import REMOTE_ROOT, command


def watch(output: Path, job_id: str, interval: int = 90, timeout_hours: int = 96) -> dict:
    if not job_id.isdecimal():
        raise ValueError("Slurm job ID must be numeric.")
    requests = binary.load_requests(output)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    remote_output = f"{REMOTE_ROOT}/analysis/rq3_llm_validation/{output.name}"
    started = time.monotonic()
    uploaded = set()
    snapshot = output / "source_runtime" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot.mkdir(parents=True)
    files = [Path(__file__), Path(run_binary.__file__), Path(binary.__file__), Path(v1.__file__), Path(binary.v2.__file__),
              binary.v2.ROOT / "analysis/run_qwen_gautschi.sbatch"]
    for path in files:
        shutil.copy2(path, snapshot / path.name)
    v1.atomic_write_json(snapshot / "source_hashes.json", {path.name: v1.sha256_file(path) for path in files})
    v1.atomic_write_json(output / "execution.json", {
        "version": binary.VERSION, "job_id": job_id, "host": "gautschi", "remote_output": remote_output,
        "cloud_transport": "batch_only_including_retries", "max_batch_attempts": run_binary.MAX_BATCH_ATTEMPTS,
        "monitor_timeout_hours": timeout_hours, "started_at": datetime.now(timezone.utc).isoformat()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        while time.monotonic() - started < timeout_hours * 3600:
            futures = {p: pool.submit(run_binary.tick, output, p) for p in ("openai", "gemini")}
            providers = {}
            for provider, future in futures.items():
                try:
                    providers[provider] = future.result()
                except Exception as error:
                    providers[provider] = {"status": "controller_error", "error": f"{type(error).__name__}: {error}"[:2000]}
                if providers[provider].get("status") == "completed" and provider not in uploaded:
                    backup = command(["rsync", "-az", "--exclude=raw/", "--exclude=batches/", "--exclude=*.lock",
                                      str(output / provider), f"gautschi:{remote_output}/"], 120)
                    if backup["returncode"] == 0:
                        uploaded.add(provider)
            slurm = command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "gautschi",
                             f"sacct -j {job_id} --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed"])
            sync = command(["rsync", "-az", "--include=*/", "--include=*.json", "--include=*.parquet", "--exclude=*",
                            f"gautschi:{remote_output}/qwen", str(output) + "/"], 120)
            command(["rsync", "-az", f"gautschi:{REMOTE_ROOT}/logs/qwen-{job_id}.out",
                     f"gautschi:{REMOTE_ROOT}/logs/qwen-{job_id}.err", str(logs) + "/"])
            qwen_path = output / "qwen/run_state.json"
            providers["qwen"] = json.loads(qwen_path.read_text()) if qwen_path.exists() else {"status": "waiting"}
            root_states = [line.split("|")[1] for line in slurm["stdout"].splitlines()
                           if line.startswith(job_id + "|")]
            if (providers["qwen"].get("status") == "failed" and root_states
                    and root_states[-1] in {"PENDING", "RUNNING"}):
                providers["qwen"]["checkpoint_status"] = "failed"
                providers["qwen"]["status"] = "retry_pending" if root_states[-1] == "PENDING" else "retry_running"
            state = {"status": "running", "providers": providers, "slurm": slurm,
                     "qwen_sync_returncode": sync["returncode"], "job_id": job_id,
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            if any(p.get("status") in {"failed", "controller_error", "submission_uncertain", "submitting"} for p in providers.values()):
                state["status"] = "needs_attention"
            if any(p.get("status") == "blocked_billing" for p in providers.values()):
                state["status"] = "blocked_billing"
            if all(p.get("status") == "completed" for p in providers.values()):
                # Revalidate every checkpoint, identity, and hash before voting.
                for provider in binary.PROVIDERS:
                    binary.collect(output, provider, requests)
                state["results"] = binary.build_consensus(output)
                state["status"] = "completed"
                v1.atomic_write_json(output / "experiment_status.json", state)
                command(["rsync", "-az", str(output / "consensus"), str(output / "experiment_status.json"),
                         f"gautschi:{remote_output}/"], 120)
                print(json.dumps(state, indent=2), flush=True)
                return state
            v1.atomic_write_json(output / "experiment_status.json", state)
            print(json.dumps({"updated_at": state["updated_at"], "status": state["status"], "providers": providers,
                              "slurm": slurm["stdout"]}), flush=True)
            time.sleep(interval)
    state["status"] = "monitor_timeout"
    v1.atomic_write_json(output / "experiment_status.json", state)
    raise TimeoutError("Binary experiment monitor expired; results and checkpoints remain resumable.")


def main(study=binary, runner=run_binary, default_output: Path | None = None) -> None:
    global binary, run_binary
    binary, run_binary = study, runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output or binary.DEFAULT_OUTPUT)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--interval", type=int, default=90)
    args = parser.parse_args()
    watch(args.output_dir, args.job_id, args.interval)


if __name__ == "__main__":
    main()
