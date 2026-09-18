# Dataset

The workspace keeps one training set and one final holdout. There are no other
active dataset releases or splits in `data/`.

| File | Records | Purpose |
| --- | ---: | --- |
| `data/train.jsonl` | 2,826 | Train the model |
| `data/eval.jsonl` | 324 | Frozen final evaluation |
| `data/manifest.json` | Metadata | Provenance and integrity checks |

Training contains 2,676 contrastive records and 150 blind-reviewed complex cases.
Labels are synthetic and model-checked, not human-reviewed. All 64 training
source families are disjoint from the holdout's families. Contrast siblings stay
together. Do not tune hyperparameters or select checkpoints on this holdout;
derive any future tuning split from the training families only.

## Integrity

Both JSONL files retain their exact bytes, order, IDs, labels, and embedded
verification records from the combined 2,826-example release.

- Training SHA-256: `bcbb005e2b767a3404e05cd9c9a7064d582fbfe86de391c0d0cfdeacbe29db32`
- Holdout SHA-256: `8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896`

```sh
.venv-curator/bin/python -m nimble.training.verify_dataset
```

This offline check reconstructs evidence certificates and blind reviews, checks
both file hashes and counts, and rejects overlapping source families, sources,
contexts, or IDs. It makes no model or API calls.

The manifest embeds the original seed records solely as certificate provenance.
They are never passed to training or evaluation as extra examples. The 150
reviewed originals are reconstructed from their retained training rows and
checked against their original byte and record hashes. Historical paths in the
manifest identify lineage; no removed directory is required to validate or train.

The trainer regenerates scoring inputs and token IDs in memory. For the pinned
Qwen3.5-9B tokenizer, the regenerated fingerprints match the previous verified
exports exactly. There are no separate scoring/token dataset copies to manage.

## Training

Use `--data-dir data --validation data/eval.jsonl` with the schema trainer.
See [training instructions](NIMBLE_TRAINING.md) for the pinned checkpoint and recipe.

Older source releases, generation caches, alternative training splits, and the
separate 100-, 300-, 1,000-, and 120-example evaluation datasets were removed.
Saved experiment reports still describe their original inputs and results; those
reports do not imply that their old dataset directories remain available.

`data/` remains local and git-ignored. A fresh checkout needs these three files
copied together before dataset validation or training can run.
