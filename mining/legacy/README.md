# Legacy Scripts

This directory preserves superseded scripts for auditability. They are not
supported entrypoints and must not be used to rebuild official artifacts.

`finalize_from_checkpoint.py` predates the current end-to-end rebalancing
pipeline. Use `mining/src/build_rebalanced_dataset.py` instead.

`official_selection_v1/` is the exact selector and dependency snapshot recorded
by the completed v1 manifest. The entrypoint's SHA-256 is
`ae4a38148784c20b54991c06798776b1e7c67ca6fd8cb6e3605cc540a3aa3a8f`.
It is archived for audit only; do not execute it against preserved artifacts.
