# Controlled module-combination runs

This directory contains the eight runs in the complete ASWB/MSIA/UGDR factorial:

| Directory | ASWB | MSIA | UGDR |
| --- | :---: | :---: | :---: |
| `Baseline` |  |  |  |
| `ASWB` | yes |  |  |
| `MSIA` |  | yes |  |
| `UGDR` |  |  | yes |
| `ASWB+MSIA` | yes | yes |  |
| `ASWB+UGDR` | yes |  | yes |
| `MSIA+UGDR` |  | yes | yes |
| `Ours` | yes | yes | yes |

Each run directory contains its JSON-lines `log.txt`, recorded `args.json`, and
TensorBoard event file. The logs contain one record per epoch for epochs 0--71,
and every run records seed 42. Consistent with the manuscript, all endpoint
results use epoch 71 rather than selecting the best epoch.

Recompute the final metrics, conditional gains, order-averaged contributions,
isolated-gain retention, and training-stage margins from the package root with:

```bash
python3 scripts/analyze_synergy.py \
  --logs-root logs \
  --output-dir /tmp/mso_det_synergy_results
```

The included data, derived files, and diagnostic commands are documented in the
package-level `README.md`.
