# APCL: Personalized Fashion Matching with Contrastive Learning

APCL learns personalized clothing compatibility from visual features, text, and
historical interactions. This repository includes a **standalone, verified
IQON3000 example** for Specified Top / Recommend Bottom (RB).

## Verified example

One run with seed **42**, selected by validation AUC at **epoch 10**:

| Split | Requests | Supplied-pair AUC |
|---|---:|---:|
| Validation | 23,095 | 0.966746 |
| Test | 23,095 | **0.949859** |

These are actual results from the included configuration, not averages or
significance-test results. The observed test AUC remains below the paper's
reported IQON3000 RB AUC of 0.9739; this example does **not** claim exact
reproduction of that number. Validation is never reported as test performance.
See the [result record](reproduction/example/run.json) and
[complete training log](reproduction/example/epochs.jsonl).

## Quick start

Use Python 3.12 and a CUDA-enabled PyTorch installation appropriate for your GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements.txt
```

Download the [indexed datasets](https://drive.google.com/file/d/1Dg7918zUGcL7tzs_OisNzc_FxYlQMG4E/view?usp=sharing)
separately. `--data-root` must point to the directory containing `IQON3000/`.
The required files and their checksums are documented in
[reproduction/README.md](reproduction/README.md).

Train and select the best checkpoint using validation only:

```bash
python -m apcl_repro.train \
  --config reproduction/configs/iqon_rb.json \
  --data-root /path/to/dataset \
  --device cuda:0 \
  --output runs/iqon_rb_seed42 \
  --trusted-feature-pickle
```

Evaluate the selected checkpoint on the test split:

```bash
python -m apcl_repro.evaluate \
  --config runs/iqon_rb_seed42/config.json \
  --data-root /path/to/dataset \
  --checkpoint runs/iqon_rb_seed42/best.pt \
  --device cuda:0 \
  --output runs/iqon_rb_seed42/test \
  --trusted-feature-pickle
```

The pickle option is for the original, trusted legacy feature files. Corrupt or
truncated files are rejected, never replaced with synthetic features.

## What is included

- `apcl_repro/`: APCL and its internal BPR/VTBPR/TextCNN components, deterministic
  history construction, training, checkpoint selection, and pair evaluation.
- `reproduction/configs/iqon_rb.json`: the single documented example configuration.
- `reproduction/example/`: measured results and the original 18-epoch training log.
- `reproduction/tests/`: data-isolation, metric, and corruption checks.

The standalone package does not require other baseline models or the earlier
benchmarking framework. Existing historical scripts in this repository are
retained; use the commands above for the verified example.

Datasets, feature tensors, user-level predictions, and trained weights are not
included in the source package. Training creates the checkpoint locally.

## Evaluation scope

AUC is the fraction of supplied positive/negative pairs correctly ordered by
the documented forward score; ties receive half credit. It is not a pooled ROC
AUC or an AUC over a newly sampled candidate set. HR@10 and NDCG@10 are not
reported in this minimal example because they require a separately specified
ranking candidate protocol. They cannot be inferred from AUC.

See [the reproduction notes](reproduction/README.md) for parameter provenance,
model-version details, and reproducibility checks.
