# Study 3 scaled exploratory artifacts

These compact artifacts are the four matched seed-1337 Kaggle runs from the scaled Study 3 protocol. They are retained for auditability and are **not** confirmatory evidence: the planned three-seed replication could not start after Kaggle rejected new jobs with `Maximum weekly GPU quota of 30.00 hours reached`.

The artifacts contain the raw training metrics, run manifests, parameter counts, and raw lm-evaluation-harness outputs for tied, partial rank-8, untied, and capacity-control conditions. The large token-offset logs and checkpoints from these already-completed jobs are intentionally excluded; each manifest retains the token-stream digest, token count, checkpoint hash, environment metadata, and dataset hashes. The launcher has since been fixed to retain both the final model and optimizer checkpoints for future Kaggle runs, so those runs can be resumed exactly.

The existing 30M paper and conclusions are unchanged. Do not use this directory to claim a scaled-model result until the missing seeds are completed and the strict study validator passes.
