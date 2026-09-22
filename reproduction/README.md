# One verified APCL reproduction example

This package isolates IQON3000 RB (given top, recommend bottom). The example
configuration was fixed before evaluating its selected checkpoint on test data.
No test-based hyperparameter or seed selection was performed for this release.

## Data layout

Pass the parent of `IQON3000/` as `--data-root`:

```text
dataset/
└── IQON3000/
    ├── data/
    │   ├── train_indexed.csv
    │   ├── valid_indexed.csv
    │   ├── test_indexed.csv
    │   ├── user_map.json
    │   └── item_map.json
    └── feat/
        ├── visualfeatures_indexedtenseor
        ├── textfeatures_indexedtenseor
        └── smallnwjc2vec
```

The `indexedtenseor` spelling is the original filename. CSVs have no header and
four columns: user ID, specified top ID, positive bottom ID, supplied negative
bottom ID. Preserve the original item/token indexing. The observed data contain
170,601 training, 23,095 validation, and 23,095 test requests; 3,236 users and
142,737 items. Visual features have width 2,048; indexed text has 83 tokens per
item with 300-dimensional embeddings.

SHA256 digests of all eight input files are in `example/run.json`. No data are
bundled. Cross-split duplicate `(user, specified, positive)` triples are removed
in train/validation/test priority order and recorded in each run's manifest;
within-split repeated requests are retained. No rows were removed in this IQON
example. A positive equal to its supplied negative is rejected.

## Configuration

| Parameter | Value |
|---|---:|
| Seed | 42 |
| Training and pair-evaluation batch size | 512 |
| Hidden dimension | 512 |
| Initial learning rate | 0.001 |
| Learning-rate multiplier per epoch | 0.97 |
| Weight decay | 0.00001 |
| Maximum epochs / early-stopping patience | 80 / 8 |
| `weight_P` / `uu_w` / `temperature` | 0.5 / 3 / 5 |
| `w_infoNCE` | 0.05 |
| Neighbor users / sampled history length | 2 / 2 |
| Cosine similarity / feature normalization | enabled / enabled |
| Training `drop_last` | enabled |

Parameters were recovered from the APCL entry-point overrides and historical
training logs. Batch size and learning-rate scheduling were cross-checked against
[NiPC-BPR commit ed02ab4](https://github.com/asaander719/NiPC-BPR/tree/ed02ab499088078b50c392b3709676fe4023ab3a).
The old high-scoring run had no complete checkpoint and may not have fixed its
seed. Exact equality with that historical run is therefore not asserted.

## Model and protocol details

The namespaced model under `apcl_repro/models/` preserves the APCL source used by
the audited retraining run, including its initialization order and checkpoint
keys. It is separate from the repository's older root-level model variant.
The profile uses the historical forward pair score with
`include_indirect_score=false`; contrastive auxiliary losses remain enabled
during training. This flag disables the later added direct score term, not the
training contrastive objective. The legacy wide-ranking inference path is not
used by this example.

Training histories are sampled from training interactions using the training
positive's neighboring users, as in the historical training procedure. Sampling
is seeded per epoch; shared neighbor lists are never mutated. Validation/test
histories depend only on the user, specified item, and training data. In this
profile, the evaluated pair score is invariant to the history tensor; a
regression test checks this explicitly. Validation/test labels are not used to
construct scoring context.

Checkpoint selection uses supplied-pair validation AUC only. The same pair-score
definition and batch size are used on test data. Because the inherited model
normalizes along the batch dimension, changing evaluation batches can change
scores. Keep the documented batch size and CSV ordering.

Deterministic PyTorch operations are enabled, TF32 and cuDNN autotuning are
disabled, and Python/NumPy/PyTorch randomness is seeded. The recorded environment
was Python 3.12, PyTorch 2.14.0+cu130, NumPy 2.5.3, CUDA 13.0, and an RTX 3090.
Different library versions or hardware can change numerical results; the exact
observed environment is included in `example/run.json`.

## Outputs and verification

Training writes configuration, input/source hashes, per-epoch metrics, the best
checkpoint, and best-validation score differences. Evaluation writes aggregate
results and a local `per_query.csv` with row ID, user ID, specified/positive/
negative IDs, score difference, and pairwise AUC contribution. These raw outputs
are excluded from the published package.

Run the standalone checks:

```bash
python -m unittest discover -s reproduction/tests
```

Release verification also reloads the selected real checkpoint and compares its
validation scores with the original run, and repeats the first full training
epoch in the standalone package. See `example/verification.json` for the measured
outcomes. The complete 18-epoch result in `example/epochs.jsonl` comes from the
audited runner from which this package was extracted.

Build a code-only archive:

```bash
python reproduction/package.py --output APCL-reproduction.zip
```

The archive uses an explicit file allowlist; datasets, weights, raw predictions,
credentials, and unrelated model code are excluded.
