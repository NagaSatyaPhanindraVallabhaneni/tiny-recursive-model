"""Training for the tiny recursive model and the big baseline.

Both models train on the same synthetic Sudoku data for the same number of
epochs. Checkpoints and a metrics.json (used by demo.py and the /compare API)
are written to models/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tiny_recursive_model.baseline import BigBaseline
from tiny_recursive_model.model import TinyRecursiveModel
from tiny_recursive_model.sudoku import generate_puzzle

DEVICE = torch.device("cpu")


def build_dataset(n: int, n_holes: int, seed: int) -> TensorDataset:
    puzzles, solutions = [], []
    for k in range(n):
        p, s, _ = generate_puzzle(n_holes=n_holes, seed=seed * 1_000_003 + k)
        puzzles.append(p)
        solutions.append(s)
    return TensorDataset(
        torch.tensor(puzzles, dtype=torch.long),
        torch.tensor(solutions, dtype=torch.long),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> dict[str, float]:
    """Cell accuracy + exact-match (all 81 cells right) accuracy."""
    cells_right, cells_total, puzzles_right, puzzles_total = 0, 0, 0, 0
    model.eval()
    for x, y in loader:
        pred = model.predict(x)
        cells_right += (pred == y).sum().item()
        cells_total += y.numel()
        puzzles_right += (pred == y).all(dim=1).sum().item()
        puzzles_total += y.size(0)
    return {
        "cell_acc": cells_right / cells_total,
        "exact_match": puzzles_right / puzzles_total,
    }


def measure_latency_ms(model: torch.nn.Module, n: int = 200) -> float:
    """Mean single-puzzle CPU inference latency in milliseconds."""
    model.eval()
    x = torch.randint(0, 10, (1, 81), dtype=torch.long)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model.predict(x)
        t0 = time.perf_counter()
        for _ in range(n):
            model.predict(x)
        t1 = time.perf_counter()
    return (t1 - t0) / n * 1000.0


def train_one(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    recursive: bool,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total = epochs * len(loader)
    step = 0
    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss, nb = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            if recursive:
                loss = model.deep_supervision_loss(x, y)
            else:
                loss = F.cross_entropy(model(x).reshape(-1, 10), y.reshape(-1))
            loss.backward()
            opt.step()
            # linear LR decay to 10%
            step += 1
            frac = step / total
            for g in opt.param_groups:
                g["lr"] = lr * (1.0 - 0.9 * frac)
            epoch_loss += loss.item() * x.size(0)
            nb += x.size(0)
        avg = epoch_loss / nb
        losses.append(avg)
        print(f"  epoch {epoch + 1:>3}/{epochs}  loss {avg:.4f}", flush=True)
    return losses


def run_training(
    n_train: int = 8000,
    n_val: int = 2000,
    n_holes: int = 30,
    epochs: int = 15,
    batch: int = 128,
    lr: float = 1e-3,
    n_steps: int = 4,
    seed: int = 0,
    out_dir: str | Path = "models",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(f"[data] generating {n_train} train + {n_val} val puzzles ({n_holes} holes)...")
    t0 = time.perf_counter()
    train_ds = build_dataset(n_train, n_holes, seed)
    val_ds = build_dataset(n_val, n_holes, seed + 999)
    print(f"[data] done in {time.perf_counter() - t0:.1f}s")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)

    trm = TinyRecursiveModel(n_steps=n_steps)
    base = BigBaseline()
    print(f"[model] TRM params: {trm.param_count():,}")
    print(f"[model] baseline params: {base.param_count():,}")

    print("[train] TRM (deep supervision)...")
    train_one(trm, train_loader, epochs, lr, recursive=True, seed=seed)
    print("[train] baseline (plain cross-entropy)...")
    train_one(base, train_loader, epochs, lr, recursive=False, seed=seed)

    print("[eval] scoring on held-out puzzles...")
    trm_metrics = evaluate(trm, val_loader)
    base_metrics = evaluate(base, val_loader)
    trm_lat = measure_latency_ms(trm)
    base_lat = measure_latency_ms(base)

    metrics = {
        "trm": {
            "params": trm.param_count(),
            "n_steps": trm.n_steps,
            **{k: round(v, 4) for k, v in trm_metrics.items()},
            "latency_ms": round(trm_lat, 3),
        },
        "baseline": {
            "params": base.param_count(),
            **{k: round(v, 4) for k, v in base_metrics.items()},
            "latency_ms": round(base_lat, 3),
        },
        "config": {
            "n_train": n_train,
            "n_val": n_val,
            "n_holes": n_holes,
            "epochs": epochs,
            "batch": batch,
            "lr": lr,
            "seed": seed,
            "device": "cpu",
        },
        "wall_time_s": round(time.perf_counter() - t_start, 1),
    }

    torch.save({"state_dict": trm.state_dict(), "n_steps": trm.n_steps}, out / "trm.pt")
    torch.save({"state_dict": base.state_dict()}, out / "baseline.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] wall time {metrics['wall_time_s']}s -> {out}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TRM vs baseline on synthetic Sudoku")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--holes", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    run_training(
        n_train=args.n_train,
        n_val=args.n_val,
        n_holes=args.holes,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        n_steps=args.steps,
        seed=args.seed,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
