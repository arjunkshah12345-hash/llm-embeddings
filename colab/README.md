# Google Colab fallback

This is a free-GPU fallback for the frozen Study 3 training matrix when the
Kaggle weekly quota is exhausted. It runs training only; the primary missing
evidence is matched validation-loss replication, while the existing scaled
benchmark outputs are historical pre-fix artifacts.

Open the repository in Colab after signing in, select a GPU runtime, mount
Google Drive, clone this repository, and run one seed at a time:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/arjunkshah12345-hash/llm-embeddings.git /content/llm-embeddings
%cd /content/llm-embeddings
!pip install -q "datasets==3.6.0" "tiktoken==0.13.0"
!python colab/run_scale3.py --seed 2027 --output-root /content/drive/MyDrive/llm-embeddings-scale3 --resume-existing
```

Run seed `31415` in a separate runtime. The runner refuses CPU execution,
uses the frozen 12-layer/768-width/rank-8 configuration, writes exact
`last.pt` and `optimizer_last.pt` checkpoints, and can resume an interrupted
condition with `--resume-existing`. It does not alter the Kaggle artifacts or
the completed small-model studies.
