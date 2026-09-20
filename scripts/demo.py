"""Print the TRM vs baseline comparison table from the real training run.

Reads models/metrics.json (written by scripts/train.py). All numbers are from
the actual run — nothing is invented here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = Path(__file__).resolve().parent.parent / "models" / "metrics.json"


def main() -> None:
    if not METRICS.exists():
        sys.exit("No models/metrics.json found. Run: PYTHONPATH=src python scripts/train.py")
    m = json.loads(METRICS.read_text())
    t, b, c = m["trm"], m["baseline"], m["config"]
    ratio = b["params"] / t["params"]

    print("=" * 72)
    print("  TINY RECURSIVE MODEL vs BIG BASELINE — synthetic Sudoku, real run")
    print("=" * 72)
    print(f"  config: {c['n_train']} train / {c['n_val']} val puzzles, "
          f"{c['n_holes']} holes, {c['epochs']} epochs, CPU")
    print("-" * 72)
    print(f"  {'metric':<28}{'TRM':>20}{'baseline':>20}")
    print("-" * 72)
    print(f"  {'parameters':<28}{t['params']:>20,}{b['params']:>20,}")
    print(f"  {'cell accuracy':<28}{t['cell_acc']:>19.2%}{b['cell_acc']:>19.2%}")
    print(f"  {'exact-match accuracy':<28}{t['exact_match']:>19.2%}{b['exact_match']:>19.2%}")
    print(f"  {'CPU latency (ms/puzzle)':<28}{t['latency_ms']:>20.3f}{b['latency_ms']:>20.3f}")
    print("-" * 72)
    print(f"  TRM uses {ratio:.0f}x fewer parameters "
          f"({t['params']:,} vs {b['params']:,}).")
    winner = "TRM" if t["exact_match"] >= b["exact_match"] else "baseline"
    print(f"  exact-match winner: {winner} "
          f"({t['exact_match']:.2%} vs {b['exact_match']:.2%}).")
    print(f"  training wall time: {m['wall_time_s'] / 60:.1f} min on CPU.")
    print("=" * 72)


if __name__ == "__main__":
    main()
