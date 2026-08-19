# Score-Only Distillation for Compact Dense Retrieval

Reference implementation for [**Score-Only Distillation for Compact Dense Retrieval**](https://arxiv.org/abs/2607.11465).

The method transfers a teacher ensemble's relative preferences over a fixed candidate row into a compact bi-encoder. It needs only scalar teacher scores: no hidden states, tokenizer sharing, or teacher gradients.

## Training update

Every training update follows one explicit path:

1. Encode each query once.
2. Encode document candidates in chunks under `torch.utils.checkpoint` with `use_reentrant=False` and `preserve_rng_state=True`.
3. Return and retain only the small \(B\times k\) score matrix from the document checkpoints.
4. Compute centered MSE once over the complete row.
5. Call backward once; document chunks are recomputed with their forward RNG state.

This preserves dropout semantics while avoiding retention of all document-encoder activations. See [`training/checkpointed.py`](src/score_vector_distillation/training/checkpointed.py) and its value/gradient/dropout tests.

## Install and validate

Python dependencies and environments are managed with [uv](https://docs.astral.sh/uv/). Qwen/student training and NV-Embed-v2 teacher scoring use incompatible SentenceTransformers/Transformers versions, matching the validated source protocol, so they are intentionally separate environments.

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --frozen --extra train --group dev
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd smoke --config configs/smoke/tiny.yaml
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync pytest
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync ruff check .
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync ruff format --check .
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync mypy

UV_PROJECT_ENVIRONMENT=.venv-nv-teacher uv sync --frozen --extra nv-teacher
```

The smoke test is CPU-only and downloads no model or dataset. It executes the
same checkpointed score-vector loss used by training.

## Data and split protocol

The method evaluation uses public snapshots of eight datasets distributed by the [BEIR benchmark](https://github.com/beir-cellar/beir). The loader consumes the snapshot files directly. Full-corpus retrieval follows BEIR's self-document rule by excluding `query_id == document_id` before top-k selection.

The default row construction is query-disjoint (no query ID used for training appears in the diagnostic evaluation rows):

| Purpose | Dataset split | Rows |
|---|---|---:|
| Train | SciFact `train` | 809 |
| Train | NFCorpus `train` | 2,590 |
| Train | FiQA `train` | 5,500 |
| Diagnostic eval | SciFact `test` | 200 |
| Diagnostic eval | NFCorpus `dev` | 324 |
| Diagnostic eval | FiQA `dev` | 500 |

This produces 8,899 training rows and 1,024 diagnostic rows, with an assertion that query IDs are disjoint. Each nested Parquet row contains one query, a positive at candidate index zero, 31 deterministic TF-IDF negatives, candidate IDs/texts, negative-source labels, and a row ID. The eight-task paper evaluation remains full-corpus retrieval over SciFact, NFCorpus, FiQA, ArguAna, SciDocs, TREC-COVID, Webis-Touche2020, and Quora.

The strict default replaces a historical 200-row SciFact diagnostic sampled from its training split with 200 SciFact test queries. It does not change training rows, learned checkpoints, full-corpus test evaluation, Tables 1–2, or their paired-bootstrap query sets. Any candidate-row evidence tied to the historical SciFact slice requires a strict rerun before it can be presented as query-disjoint.

The query-disjoint teacher-agreement, 21-point fusion-weight, and calibrated
target screens were regenerated on 2026-07-13 and are available under
[`artifacts/paper/results`](artifacts/paper/results). They select the NV-Embed
endpoint at 0.698 macro candidate-row NDCG@10, versus 0.667 for equal fusion.
Other historical candidate-row loss or fidelity artifacts are not relabeled as
query-disjoint evidence.

Snapshot URLs and immutable checksums live in [`configs/paper/data/eight_task_panel.yaml`](configs/paper/data/eight_task_panel.yaml). On 2026-07-13, each SHA-256 was computed from the public archive and its bytes were cross-checked against the MD5 published by BEIR. Fetching and extraction are atomic. A small `.snapshot.json` identifies each materialized cache; normal reuse validates its identity and required layout without rehashing every extracted file.

The executable preparation path is:

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd data-fetch \
  --config configs/paper/data/eight_task_panel.yaml --cache-dir data/cache
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd data-prepare \
  --config configs/paper/data/eight_task_panel.yaml \
  --cache-dir data/cache --output-dir data/prepared

UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd teacher-score \
  --rows data/prepared/train.candidates.jsonl \
  --model-config configs/paper/models/qwen3_8b.yaml \
  --output data/prepared/train.qwen3_8b.parquet
UV_PROJECT_ENVIRONMENT=.venv-nv-teacher uv run --no-sync svd teacher-score \
  --rows data/prepared/train.candidates.jsonl \
  --model-config configs/paper/models/nv_embed_v2.yaml \
  --output data/prepared/train.nv_embed_v2.parquet

UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd teacher-fuse \
  --rows data/prepared/train.candidates.jsonl \
  --run-config configs/paper/runs/qwen_main.yaml \
  --teacher qwen3_8b=data/prepared/train.qwen3_8b.parquet \
  --teacher nv_embed_v2=data/prepared/train.nv_embed_v2.parquet \
  --output data/prepared/train.parquet
```

Repeat the two scoring commands and fusion for `eval.candidates.jsonl`. Fusion checks row IDs, widths, finite scores, and normalized weights. Training then checks the nested table schema and configured candidate width directly; intermediate files do not require sidecar manifests.

Full-corpus evaluation loads the normalized BEIR snapshot files directly,
encodes judged queries once, streams the document corpus in bounded chunks, and
writes deterministic rankings, aggregate metrics, per-query metrics, and a
compact `run.json`. These generated outputs remain local and are ignored by
Git. To evaluate the frozen Qwen base under the paper runtime settings:

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd evaluate \
  --run-config configs/paper/runs/qwen_main.yaml --cache-dir data/cache \
  --dataset-name SciFact --split test \
  --model-config configs/paper/models/qwen3_0p6b.yaml \
  --dtype bfloat16 --query-batch-size 4 --document-batch-size 4 \
  --require-resolved-attention eager \
  --output-dir artifacts/evaluation/qwen_base/scifact
```

A freshly trained local adapter is evaluated directly from its checkpoint
directory. The
adapter config is checked against the pinned public base model ID and revision:

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync svd evaluate \
  --run-config configs/paper/runs/qwen_main.yaml --cache-dir data/cache \
  --dataset-name SciFact --split test \
  --model-config configs/paper/models/qwen3_0p6b.yaml \
  --checkpoint artifacts/runs/qwen_main_seed42/model_final \
  --dtype bfloat16 --query-batch-size 4 --document-batch-size 4 \
  --require-resolved-attention eager \
  --output-dir artifacts/evaluation/qwen_main_seed42/scifact
```

Evaluation records the adapter weight filename and SHA-256 in `run.json`, never
the local loading path. Training writes the same compact record inside
`model_final/`, together with the resolved config, seed, input hashes, runtime
versions, attention implementation, Git commit when available, and primary
checkpoint hash.

The executable run config is the authoritative source for retrieval depth,
metric cutoffs, similarity, aggregation, and the referenced data config. CLI
overrides that conflict with this protocol are rejected. The trainer's
per-epoch validation is named `validation_loss` deliberately: it measures
held-out candidate-row distillation loss, while `svd evaluate` performs the
separate full-corpus retrieval evaluation.

## Reproduction levels

The repository distinguishes two evidence levels:

| Level | What it validates | Network / accelerator |
|---|---|---|
| Unit + smoke | Loss identity, gradients, RNG-preserving checkpointing, schemas, ranking metrics | None / CPU |
| Independent end-to-end rerun | Snapshot preparation, teacher scoring, fusion, LoRA training, and full-corpus retrieval | Public downloads / GPU recommended |

Reported numbers are committed under
[`artifacts/paper/results`](artifacts/paper/results) as aggregate reference
outputs.

## Paper configurations

- [`configs/paper/runs/qwen_main.yaml`](configs/paper/runs/qwen_main.yaml): Qwen3-Embedding-0.6B student, \(k=32\), three seeds, LoRA rank 256, effective batch 16, checkpointed document chunks of four.
- [`configs/paper/runs/e5_main.yaml`](configs/paper/runs/e5_main.yaml): E5-large-v2 family replication, three seeds, LoRA rank 64.
- `qwen_qwen8_only.yaml`, `qwen_nv_only.yaml`, `e5_qwen8_only.yaml`, and `e5_nv_only.yaml`: executable single-teacher C32 variants inherited from the matching family config.
- `qwen_control_*.yaml`: executable seed-42 C32 objective controls; [`loss_controls.yaml`](configs/paper/runs/loss_controls.yaml) is their compact sweep index.
- [`configs/paper/models`](configs/paper/models): exact text templates and immutable upstream model revisions.

One seed pipeline is launched as:

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv run --no-sync accelerate launch \
  --num_processes 8 --mixed_precision no \
  -m score_vector_distillation.cli train \
  --config configs/paper/runs/qwen_main.yaml \
  --train-file data/prepared/train.parquet \
  --eval-file data/prepared/eval.parquet \
  --output-dir artifacts/runs/qwen_main_seed42 \
  --seed 42
```

Each completed run keeps epoch-level resumable Accelerator state plus one readable `model_final/run.json`. The record contains the resolved configuration, input and checkpoint hashes, Git commit when available, model identities, attention backend, and relevant runtime versions (including detected AOTriton binaries).
