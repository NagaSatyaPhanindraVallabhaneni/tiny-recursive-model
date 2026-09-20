"""FastAPI service: solve Sudoku with the tiny recursive model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from tiny_recursive_model.model import TinyRecursiveModel
from tiny_recursive_model.sudoku import is_valid_solution

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent.parent / "models"
TRM_CKPT = MODELS_DIR / "trm.pt"
METRICS_FILE = MODELS_DIR / "metrics.json"

app = FastAPI(title="tiny-recursive-model", version="0.1.0")

_model: TinyRecursiveModel | None = None


class SolveRequest(BaseModel):
    puzzle: list[int]

    @field_validator("puzzle")
    @classmethod
    def check_puzzle(cls, v: list[int]) -> list[int]:
        if not isinstance(v, list) or len(v) != 81:
            raise ValueError("puzzle must be a list of exactly 81 cells")
        if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 or c > 9 for c in v):
            raise ValueError("each cell must be an integer 0-9 (0 = blank)")
        return v


def get_model() -> TinyRecursiveModel:
    """Load (and cache) the trained TRM checkpoint."""
    global _model
    if _model is None:
        if not TRM_CKPT.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained model found. Run scripts/train.py first.",
            )
        ckpt = torch.load(TRM_CKPT, map_location="cpu", weights_only=True)
        _model = TinyRecursiveModel(n_steps=ckpt.get("n_steps", 4))
        _model.load_state_dict(ckpt["state_dict"])
        _model.eval()
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": TRM_CKPT.exists()}


@app.post("/solve")
def solve(req: SolveRequest) -> dict:
    model = get_model()
    x = torch.tensor([req.puzzle], dtype=torch.long)
    solution = model.predict(x)[0].tolist()
    return {
        "solution": solution,
        "steps_used": model.n_steps,
        "valid_sudoku": is_valid_solution(solution),
        "note": "model prediction; valid_sudoku checks grid validity, not match to a key",
    }


@app.get("/compare")
def compare() -> dict:
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=503, detail="No metrics yet. Run scripts/train.py first."
        )
    return json.loads(METRICS_FILE.read_text())
