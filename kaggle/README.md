# Kaggle execution

All model training for this project runs in Kaggle. The local repository is used
for code review, static checks, artifact collection, and analysis only.

The launcher generates private Kaggle script kernels that clone an exact Git
commit, verify it, install the portable dependencies, run one assigned study
condition set, evaluate mechanism/representation diagnostics, and write a
compact artifact manifest. Checkpoints are used inside the kernel for analysis
and removed from the collected artifact bundle to keep downloads manageable.

## Validate the launcher without training

```bash
python3 kaggle/launch.py --profile validation --dry-run
```

## Launch the frozen primary study

```bash
python3 kaggle/launch.py \
  --profile primary \
  --commit <frozen-commit> \
  --seeds 1337 2027 31415
```

The command submits one private GPU kernel per seed. It does not overwrite an
existing kernel slug. Use `kaggle kernels status owner/slug` to inspect jobs.

## Collect a finished kernel

```bash
python3 kaggle/collect.py \
  --kernel aks1321/llm-embeddings-primary-10k-seed1337 \
  --destination cloud_artifacts/primary_seed1337
```

The collector refuses to replace an existing destination unless `--overwrite`
is supplied and verifies the artifact manifest's commit and experiment ID.

## Launch the exploratory rank sweep

After the primary result is available, launch the six-rank secondary sweep with
the same model, data, optimizer, and 10,000-step horizon:

```bash
python3 kaggle/launch.py \
  --profile rank \
  --commit <frozen-commit> \
  --seeds 1337
```

This runs ranks 1, 2, 4, 8, 16, and 32 at alpha 8. It is exploratory until
the useful region is rerun with the predeclared multi-seed protocol.
