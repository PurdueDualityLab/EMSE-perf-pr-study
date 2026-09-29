"""Run the 88-PR regex v3 replication with native cloud Batch or local Qwen."""

from analysis.rq3_llm_validation import regex_v3, run_binary


if __name__ == "__main__":
    run_binary.main(regex_v3, regex_v3.DEFAULT_OUTPUT)
