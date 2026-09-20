"""Train TRM vs baseline on synthetic Sudoku (CPU-friendly).

Usage:
    PYTHONPATH=src python scripts/train.py [--n-train 20000 --epochs 25 ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tiny_recursive_model.train import main  # noqa: E402

if __name__ == "__main__":
    main()
