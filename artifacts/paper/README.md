# Paper artifacts

The committed `main_results.csv` and `loss_controls.csv` files contain the
corrected fixed-runtime reducers used by the manuscript tables. The
`figure_systems.csv` file joins those quality results to the original MI210
batch-8 encoder measurements used by the two main plots.

Regenerate the plots from the repository root with:

```bash
uv run --extra paper python scripts/generate_paper_figures.py
```

The corrected plots include only systems with matched fixed-runtime quality
evidence. Historical E5 size-sweep points and the NV-Embed quality point are
excluded because they were not rerun under the corrected matched protocol.

The compact, scrubbed single-seed uncertainty checks behind the lean
loss-ablation wording are recorded in `results/loss_variant_uncertainty.json`.
They do not
support a robust ordering among centered, rank-weighted, and unit-normalized
variants over training randomness.

The repository intentionally distributes aggregate paper fixtures rather than
trained checkpoints or raw per-query evaluation outputs.

The `query_disjoint_*` files contain candidate-row diagnostics. They
cover the 1,024-row SciFact-test, NFCorpus-dev, and FiQA-dev protocol used for
the corrected fusion-weight, teacher-agreement, and calibrated-fusion claims.
