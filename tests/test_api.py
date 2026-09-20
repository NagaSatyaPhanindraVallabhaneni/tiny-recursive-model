"""Tests for the FastAPI service."""

import torch
from fastapi.testclient import TestClient

import tiny_recursive_model.app as appmod
from tiny_recursive_model.app import TRM_CKPT, app
from tiny_recursive_model.sudoku import generate_puzzle

client = TestClient(app)


class StubModel(torch.nn.Module):
    """Pretends to solve: echoes a fixed valid solution regardless of input."""

    n_steps = 6

    def __init__(self, solution):
        super().__init__()
        self._solution = torch.tensor(solution, dtype=torch.long)

    def predict(self, x):
        return self._solution.unsqueeze(0).expand(x.size(0), -1)


def _valid_puzzle_request():
    puzzle, _, _ = generate_puzzle(n_holes=20, seed=42)
    return {"puzzle": puzzle}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_solve_rejects_short_puzzle():
    r = client.post("/solve", json={"puzzle": [0] * 80})
    assert r.status_code == 422


def test_solve_rejects_long_puzzle():
    r = client.post("/solve", json={"puzzle": [0] * 82})
    assert r.status_code == 422


def test_solve_rejects_out_of_range_values():
    r = client.post("/solve", json={"puzzle": [10] + [0] * 80})
    assert r.status_code == 422
    r = client.post("/solve", json={"puzzle": [-1] + [0] * 80})
    assert r.status_code == 422


def test_solve_rejects_non_integer_cells():
    r = client.post("/solve", json={"puzzle": ["x"] + [0] * 80})
    assert r.status_code == 422


def test_solve_plumbing_with_stub_model(monkeypatch):
    puzzle, solution, _ = generate_puzzle(n_holes=20, seed=7)
    monkeypatch.setattr(appmod, "get_model", lambda: StubModel(solution))
    r = client.post("/solve", json={"puzzle": puzzle})
    assert r.status_code == 200
    body = r.json()
    assert body["solution"] == solution
    assert body["steps_used"] == 6
    assert body["valid_sudoku"] is True
    assert len(body["solution"]) == 81


def test_compare_without_metrics_is_503(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "METRICS_FILE", tmp_path / "metrics.json")
    r = client.get("/compare")
    assert r.status_code == 503


def test_compare_returns_metrics(monkeypatch, tmp_path):
    fake = {"trm": {"exact_match": 0.5}, "baseline": {"exact_match": 0.4}}
    p = tmp_path / "metrics.json"
    p.write_text(__import__("json").dumps(fake))
    monkeypatch.setattr(appmod, "METRICS_FILE", p)
    r = client.get("/compare")
    assert r.status_code == 200
    assert r.json() == fake


def test_solve_with_real_checkpoint():
    """Integration: real trained model solves a puzzle end-to-end.

    Skipped in CI (no checkpoint there); runs locally after scripts/train.py.
    """
    if not TRM_CKPT.exists():
        import pytest

        pytest.skip("no trained checkpoint; run scripts/train.py first")
    appmod._model = None  # reset cache
    try:
        r = client.post("/solve", json=_valid_puzzle_request())
    finally:
        appmod._model = None
    assert r.status_code == 200
    body = r.json()
    assert len(body["solution"]) == 81
    assert all(1 <= v <= 9 for v in body["solution"])
