# Publishing this repository

The intended public repository is:

```text
https://github.com/arjunkshah12345-hash/llm-embeddings
```

The public repository now exists, `origin` is configured to that URL, and the source is committed locally. The current development branch is `fix/dataset-split-and-resume-metrics`; it contains the Tiny Shakespeare split fix and resume-history fix and is tracked by pull request #1. Its GitHub CI checks are part of the publication gate.

For a new branch, publish the prepared history with:

```bash
git push --set-upstream origin <branch-name>
```

Verify the result with:

```bash
gh repo view arjunkshah12345-hash/llm-embeddings --json nameWithOwner,url,isPrivate,defaultBranchRef
git ls-remote --heads origin
```

For a release to `main`, merge the reviewed pull request after its checks pass, then verify that the remote `main` branch points to the merged commit. Keep the repository initialization files in the local history; do not add a second README, license, or gitignore on GitHub.
