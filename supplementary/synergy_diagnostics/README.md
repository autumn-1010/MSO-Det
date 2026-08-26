# MSO-Det synergy diagnostics

This folder contains the supplementary records used for the module-interaction
and mechanism analyses in the revised manuscript. It is intended to be uploaded
to the repository as `supplementary/synergy_diagnostics/`.

The package contains the eight controlled training logs, the numerical outputs
of the training, forward-path, backward-path, and module-correspondence
diagnostics, and the scripts used to compute those diagnostics. Generated
figures, plotting scripts, model weights, and full test-set prediction dumps are
not duplicated here.

## Directory layout

```text
logs/                              Eight 72-epoch controlled runs
diagnostics/
  training/data/                   Per-epoch AP and AP@0.75 traces
  training/results/                Factorial and conditional-gain results
  forward/                         ASWB-to-MSIA response and topology records
  gradient/                        UGDR backward-gradient and uncertainty records
  module_correspondence/           Problem-specific module measurements
  mechanism_summary.json           Compact summary of mechanism statistics
scripts/
  analyze_synergy.py               Factorial and training-stage analysis
  synergy_diagnostics.py           Forward/backward diagnostic collector
  synergy_manifest.json            Variant and diagnostic-setting manifest
```

## Controlled runs

The `logs/` directory contains the complete $2^3$ ASWB/MSIA/UGDR factorial.
Each run includes `args.json`, the JSON-lines `log.txt`, and its TensorBoard
event file.

| Run | ASWB | MSIA | UGDR |
| --- | :---: | :---: | :---: |
| `Baseline` |  |  |  |
| `ASWB` | yes |  |  |
| `MSIA` |  | yes |  |
| `UGDR` |  |  | yes |
| `ASWB+MSIA` | yes | yes |  |
| `ASWB+UGDR` | yes |  | yes |
| `MSIA+UGDR` |  | yes | yes |
| `Ours` | yes | yes | yes |

The training analysis uses the final record (epoch 71) from every run. The
derived files report the complete set of conditional AP and AP@0.75 margins,
order-averaged module contributions, factorial interaction terms, and
five-epoch trailing training margins.

## Mechanism records

The forward diagnostic uses 64 stratified test images. Its per-image records
track the response change from the ASWB output through the MSIA input/output to
the decoder logits/boxes, together with the corresponding MSIA hypergraph
topology change.

The backward diagnostic uses 64 stratified test images and 3,364 matched boxes.
Its raw records contain localization uncertainty and error for each matched box,
as well as the standard localization gradient, the complete UGDR gradient, and
the differentiable uncertainty-dependent gradient component at ASWB, MSIA, and
the decoder.

The compact `diagnostics/mechanism_summary.json` is derived from these raw CSV
and JSON records. The principal values reported in the manuscript are:

- Relative response changes of 1.280 at the ASWB output, 0.529/0.329 at the
  MSIA input/output, and 0.538/0.711 at the decoder logits/boxes.
- A mean increase of 81.6 nodes per hyperedge (95% CI [70.4, 93.0]), mean
  topology Jaccard similarity of 0.601, and a changed-pair fraction of 7.69%.
- A Spearman correlation of 0.516 (image-clustered 95% CI [0.456, 0.564])
  between predicted uncertainty and localization error; mean error rises from
  0.224 to 0.505 across the lowest and highest uncertainty quartiles.
- Mean uncertainty-dependent gradient fractions of 0.354 at ASWB, 0.253 at
  MSIA, and 0.287 at the decoder. The component is non-zero for all 64 sampled
  images at each of the three stages.

## Recompute the factorial analysis

The factorial results can be recomputed directly from the included trace file:

```bash
python scripts/analyze_synergy.py \
  --traces diagnostics/training/data/factorial_epoch_metrics.csv \
  --output-dir /tmp/mso_det_synergy_results
```

They can also be regenerated from the eight raw run directories:

```bash
python scripts/analyze_synergy.py \
  --logs-root logs \
  --export-traces /tmp/factorial_epoch_metrics.csv \
  --output-dir /tmp/mso_det_synergy_results
```

`analyze_synergy.py` uses only the Python standard library.

## Recompute the mechanism diagnostics

`synergy_diagnostics.py` is the collector used for the forward intervention,
uncertainty analysis, and backward-gradient decomposition. Running it requires
the matching MSO-Det training implementation, the retained checkpoints, and the
GWHD test set. The repository root is supplied explicitly so this supplementary
folder can remain separate from the implementation:

```bash
python scripts/synergy_diagnostics.py \
  --repo-root /path/to/the/training/implementation \
  --manifest scripts/synergy_manifest.json \
  --logs-root logs \
  --output-dir /tmp/mso_det_mechanism_diagnostics \
  all \
  --data-images /path/to/gwhd_2021/test/images \
  --data-annotations /path/to/gwhd_2021/annotations/test.json \
  --gwhd-root /path/to/gwhd_2021 \
  --device cuda:0 \
  --num-workers 4 \
  --require-paper-settings
```

Use `python scripts/synergy_diagnostics.py --help` for the staged `forward` and
`backward` commands and their complete argument descriptions.
