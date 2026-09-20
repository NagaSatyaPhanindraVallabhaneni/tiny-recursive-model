"""Tests for the TRM model: recursion, weight sharing, deep supervision."""

import torch
import torch.nn.functional as F

from tiny_recursive_model.baseline import BigBaseline
from tiny_recursive_model.model import ReasoningBlock, TinyRecursiveModel, count_params


def make_trm(n_steps: int = 4) -> TinyRecursiveModel:
    torch.manual_seed(0)
    return TinyRecursiveModel(d_model=48, n_steps=n_steps)


def test_param_count_under_one_million():
    m = make_trm()
    n = m.param_count()
    assert n < 1_000_000, f"TRM must stay tiny, got {n:,} params"
    assert n == count_params(m)


def test_single_shared_reasoning_block():
    m = make_trm(n_steps=6)
    blocks = [mod for mod in m.modules() if isinstance(mod, ReasoningBlock)]
    assert len(blocks) == 1, "recursion must reuse ONE block, not copies"
    # total params == embeddings + ONE block + head (no per-step copies)
    expected = (
        sum(p.numel() for p in m.token_emb.parameters())
        + sum(p.numel() for p in m.pos_emb.parameters())
        + sum(p.numel() for p in blocks[0].parameters())
        + sum(p.numel() for p in m.head.parameters())
    )
    assert m.param_count() == expected


def test_reason_returns_one_output_per_step():
    m = make_trm(n_steps=5)
    x = torch.randint(0, 10, (2, 81))
    outs = m.reason(x)
    assert len(outs) == 5
    for o in outs:
        assert o.shape == (2, 81, 10)


def test_forward_returns_final_step():
    m = make_trm(n_steps=4)
    x = torch.randint(0, 10, (2, 81))
    assert torch.equal(m(x), m.reason(x)[-1])


def test_forward_output_shape():
    m = make_trm()
    x = torch.randint(0, 10, (7, 81))
    assert m(x).shape == (7, 81, 10)


def test_deep_supervision_loss_matches_manual_mean():
    m = make_trm(n_steps=3)
    x = torch.randint(0, 10, (4, 81))
    y = torch.randint(1, 10, (4, 81))
    manual = torch.stack(
        [F.cross_entropy(o.reshape(-1, 10), y.reshape(-1)) for o in m.reason(x)]
    ).mean()
    assert torch.isclose(m.deep_supervision_loss(x, y), manual)


def test_steps_refine_the_latent_state():
    m = make_trm(n_steps=4)
    m.eval()
    x = torch.randint(0, 10, (3, 81))
    outs = m.reason(x)
    # consecutive steps should not be bit-identical (the state actually evolves)
    assert not torch.equal(outs[0], outs[-1])


def test_n_steps_configurable():
    assert len(make_trm(n_steps=2).reason(torch.randint(0, 10, (1, 81)))) == 2
    assert len(make_trm(n_steps=8).reason(torch.randint(0, 10, (1, 81)))) == 8


def test_predict_returns_valid_digits():
    m = make_trm()
    pred = m.predict(torch.randint(0, 10, (5, 81)))
    assert pred.shape == (5, 81)
    assert pred.min() >= 0 and pred.max() <= 9


def test_baseline_is_bigger_than_trm():
    trm = make_trm()
    base = BigBaseline()
    assert base.param_count() > trm.param_count()
    assert base.param_count() > 1_000_000  # the point: big AND non-recursive


def test_baseline_forward_shape():
    base = BigBaseline()
    x = torch.randint(0, 10, (3, 81))
    assert base(x).shape == (3, 81, 10)
    pred = base.predict(x)
    assert pred.shape == (3, 81)
