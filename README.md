# tiny-recursive-model

**A Sudoku-solving neural network with 44,001 parameters that reasons — running in milliseconds on any CPU. No GPU. No 7-billion-parameter download. It fits in a tweet.**

[![ci](https://github.com/NagaSatyaPhanindraVallabhaneni/tiny-recursive-model/actions/workflows/ci.yml/badge.svg)](https://github.com/NagaSatyaPhanindraVallabhaneni/tiny-recursive-model/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![torch](https://img.shields.io/badge/pytorch-2.14%2Bcpu-orange)](https://pytorch.org/)
![params](https://img.shields.io/badge/params-44K-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)

Inspired by the Tiny Recursive Model line of research: instead of one giant network, take a **small reasoning block** and apply it **recursively** — the same weights reused R times — so the latent state refines itself step by step. Training uses **deep supervision**: the output head is read out after *every* recursion step, and the loss is the average over steps, pushing each step to be a better answer than the last.

This is **TRM-inspired, not a paper reproduction** — no SOTA claims, no borrowed results. Just a clean, honest implementation on synthetic Sudoku, with a bigger non-recursive baseline trained on the same data so the comparison means something.

## Why recursive reasoning?

Big models memorize. Recursive models *iterate*: each pass over the latent state is a chance to fix inconsistencies — exactly what constraint problems like Sudoku demand. The bet: **recursion beats raw parameter count**. The table below tests that bet for real.

## Architecture

```
                        ┌─────────────────────────────────┐
                        │  input: 81 cells (0-9, 0=blank) │
                        └───────────────┬─────────────────┘
                                        v
                        ┌─────────────────────────────────┐
                        │ token emb(10→48) + pos emb(81→48)│  z₀
                        └───────────────┬─────────────────┘
                                        v
                 ┌──────────────────────────────────────────────┐
                 │  RECURSION × R (R=4): the SAME block reused  │
                 │                                              │
                 │   z ──▶ [tiny mixer layer] ──▶ z'            │
                 │            │  token-mixing MLP (81→162→81):  │
                 │            │    every cell sees every cell   │
                 │            │  channel-mixing MLP (48→128→48) │
                 │            │  LayerNorm ×2, residuals        │
                 │            v                                 │
                 │   per-step logits = head(z')   ──▶ loss_r    │
                 │                                              │
                 │   deep supervision: loss = mean(loss_1..R)   │
                 └──────────────────────────────────────────────┘
                                        v
                        ┌─────────────────────────────────┐
                        │ final prediction: 81 digits 1-9 │
                        └─────────────────────────────────┘

  total parameters: 44,001  (< 1M, asserted in tests)
```

The baseline is deliberately unfair *against* the TRM: a plain feed-forward network with **~4.9M parameters (~112× more)**, trained on the same puzzles for the same epochs. If the tiny model wins, the win belongs to recursion.

## Results (real training run, synthetic data)

Trained 2026-09-20 on CPU: 8,000 train / 2,000 val synthetic puzzles, 30 holes,
15 epochs, batch 128, Adam lr 1e-3, seed 0. Wall time 18.2 min. Evaluated on
2,000 held-out puzzles.

| metric                  | TRM        | baseline (112× params) |
|-------------------------|------------|------------------------|
| parameters              | 44,001     | 4,921,370              |
| cell accuracy           | **98.16%** | 72.02%                 |
| exact-match accuracy    | **29.95%** | 0.00%                  |
| CPU latency (ms/puzzle) | 8.614      | 4.914                  |

The 44K-parameter recursive model solves **29.95%** of held-out puzzles exactly;
the 4.9M-parameter feed-forward baseline solves **none** — while the baseline is
~1.75× faster per puzzle (4.9 ms vs 8.6 ms), because the TRM runs its reasoning
block 4 times. Recursion beats raw parameter count; iteration costs latency.

## Quickstart

```bash
# train both models (CPU-only, ~45 min) and write models/ + metrics
PYTHONPATH=src python scripts/train.py

# print the comparison table from the real run
PYTHONPATH=src python scripts/demo.py

# serve the API
PYTHONPATH=src uvicorn tiny_recursive_model.app:app --port 8000
```

Docker:

```bash
docker build -t trm . && docker run -p 8000:8000 trm
```

## API

Solve a puzzle (`0` = blank):

```bash
curl -s -X POST localhost:8000/solve -H 'Content-Type: application/json' \
  -d '{"puzzle": [5,3,0, 0,7,0, 0,0,0, 6,0,0, 1,9,5, 0,0,0, 0,9,8, 0,0,0, 0,6,0, 8,0,0, 0,6,0, 0,0,3, 4,0,0, 8,0,3, 0,0,1, 7,0,0, 0,2,0, 0,0,6, 0,6,0, 0,0,0, 2,8,0, 0,0,0, 4,1,9, 0,0,5, 0,0,0, 0,8,0, 0,7,9]}'
```

```json
{"solution": [5,3,4, ...], "steps_used": 6, "valid_sudoku": true,
 "note": "model prediction; valid_sudoku checks grid validity, not match to a key"}
```

| Endpoint   | Method | Description                                              |
|------------|--------|----------------------------------------------------------|
| `/solve`   | POST   | Solve a Sudoku; 422 on malformed puzzles                 |
| `/compare` | GET    | TRM vs baseline metrics from the real training run       |
| `/health`  | GET    | Liveness + whether a trained checkpoint is loaded        |

## The data

No downloads, no scraping: puzzles are generated on the fly. A valid grid is built from a shuffled pattern, then holes are dug one at a time — each dig kept **only if the puzzle still has exactly one solution**, verified by an MRV backtracking solver that counts solutions (capped at 2). Difficulty is just "number of holes".

## Project structure

```
src/tiny_recursive_model/
  model.py      # TinyRecursiveModel: shared reasoning block × R + deep supervision
  baseline.py   # BigBaseline: ~4.9M-param non-recursive MLP, the honest comparison
  sudoku.py     # synthetic puzzle generator + uniqueness-verified digging + solver
  train.py      # dataset building, training loops, eval, latency, checkpointing
  app.py        # FastAPI service (/solve, /compare, /health)
scripts/
  train.py      # CLI: train both models, write models/*.pt + metrics.json
  demo.py       # CLI: print the TRM-vs-baseline table from the real run
tests/          # 28 tests: recursion, weight sharing, solver, API (no training)
```

## Training notes (the real run)

- Run date: 2026-09-20. Hardware: CPU only (PyTorch 2.14+cpu), no GPU.
- Data: 8,000 train + 2,000 val synthetic Sudoku puzzles, 30 holes each, every
  puzzle verified to have exactly one solution (MRV solver, solutions capped at 2).
- TRM: 44,001 params, R=4 recursion steps, deep supervision (loss averaged over
  all 4 steps). Baseline: 4,921,370-param plain MLP, same data, same 15 epochs.
- Optimizer: Adam, lr 1e-3, batch 128, seed 0. Wall time: 1,092.6 s (18.2 min).
- Final training losses (epoch 15/15, per-cell cross-entropy): TRM 0.083, baseline 0.116.
- Checkpoints: `models/trm.pt` (181 KB) and `models/metrics.json` are committed;
  the 19.7 MB `baseline.pt` is not (reproducible via `scripts/train.py`).

## Honest limitations

- **TRM-inspired, not a reproduction.** This borrows the *idea* (recursive latent refinement + deep supervision) — it does not reproduce any paper's architecture, dataset, or results, and makes no SOTA claim.
- **Synthetic task, synthetic data.** Sudoku is a clean testbed for iterative constraint reasoning, not evidence about language or general reasoning.
- **The model predicts; it doesn't prove.** `/solve` returns the network's best guess and reports whether the output is a valid grid — validity is checked, optimality is not guaranteed.
- **Small-scale training.** Tens of thousands of puzzles, tens of epochs, CPU only. A bigger budget would change the absolute numbers (though probably not the story).

## Tech stack

Python 3.12 · PyTorch 2.14 (CPU) · FastAPI · pytest · ruff · Docker · GitHub Actions
