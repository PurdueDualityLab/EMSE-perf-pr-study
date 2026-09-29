"""Monitor and finalize the 88-PR regex v3 replication."""

from analysis.rq3_llm_validation import regex_v3, run_binary, watch_binary


if __name__ == "__main__":
    run_binary.binary = regex_v3
    watch_binary.main(regex_v3, run_binary, regex_v3.DEFAULT_OUTPUT)
