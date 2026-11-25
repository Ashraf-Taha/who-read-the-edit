import pytest
import torch

from mutablekv import LM, would_be_exposure, would_be_logits
from tests.test_engine import MODELS, TEXT, gap

A, B = 30, 50


@pytest.fixture(scope="module")
def lm():
    return LM(MODELS[0], "cpu", torch.float64)


@pytest.fixture(scope="module")
def parts(lm):
    tokens = lm.encode(TEXT)[:96]
    n = tokens.numel()
    groups = torch.cat((torch.zeros(A), torch.ones(B - A), torch.full((n - B,), 2))).long()
    cold = lm.cold(tokens, groups=groups)
    kept = torch.cat((tokens[:A], tokens[B:]))
    return tokens, cold, kept


def test_kept_queries_and_normalisers_match_the_pass(lm, parts):
    tokens, _, _ = parts
    result = lm.cold(tokens[:40], keep_queries=True)
    assert result.queries.shape == (lm.L, lm.H, 40, lm.d_head)
    assert result.lse.shape == (lm.L, lm.H, 40)
    q0 = lm.layers[0].self_attn.q_proj(lm._norm(lm.layers[0].input_layernorm, lm.embed(tokens[:40])))
    q0 = lm.rotate(q0.view(40, lm.H, lm.d_head).transpose(0, 1), torch.arange(40))
    assert gap(result.queries[0], q0) < 1e-12
    assert torch.isfinite(result.lse).all()
    assert lm.cold(tokens[:40]).queries is None


def test_first_layer_estimate_is_exact_when_the_tokens_kept_their_positions(lm, parts):
    tokens, cold, kept = parts
    n = tokens.numel()
    hole = lm.run(kept, torch.cat((torch.arange(A), torch.arange(B, n))), keep_queries=True)
    suffix = slice(A, None)
    estimate = would_be_exposure(lm, hole.queries[:, :, suffix], hole.lse[:, :, suffix], 0, cold.cells.K[:, :, A:B], layers=[0])
    assert gap(estimate.double(), cold.mass[0, B:, 1].double()) < 1e-6
    assert estimate.max() > 1e-3


def test_shift_moves_kept_queries_to_the_new_position(lm, parts):
    tokens, cold, kept = parts
    n = tokens.numel()
    contiguous = lm.run(kept, torch.arange(kept.numel()), keep_queries=True)
    hole = lm.run(kept, torch.cat((torch.arange(A), torch.arange(B, n))), keep_queries=True)
    keys = cold.cells.K[0, :, A:B]
    moved = would_be_logits(lm, contiguous.queries[0, :, A:], B - A, keys)
    placed = would_be_logits(lm, hole.queries[0, :, A:], 0, keys)
    assert gap(moved, placed) < 1e-9


def test_deeper_layers_give_an_estimate_that_tracks_the_true_exposure(lm, parts):
    tokens, cold, kept = parts
    contiguous = lm.run(kept, torch.arange(kept.numel()), keep_queries=True)
    estimate = would_be_exposure(lm, contiguous.queries[:, :, A:], contiguous.lse[:, :, A:], B - A, cold.cells.K[:, :, A:B])
    truth = cold.mass[:, B:, 1].mean(0)
    rank = lambda x: x.argsort().argsort().double()
    assert torch.corrcoef(torch.stack((rank(estimate), rank(truth))))[0, 1] > 0.8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_estimator_and_runtime_run_on_the_gpu(parts):
    from mutablekv import Context

    tokens, _, _ = parts
    gpu = LM(MODELS[0], "mps", torch.float32)
    kept = torch.cat((tokens[:A], tokens[B:]))
    run = gpu.run(kept, torch.arange(kept.numel()), keep_queries=True)
    keys = gpu.cold(tokens).cells.K[:, :, A:B]
    estimate = would_be_exposure(gpu, run.queries[:, :, A:], run.lse[:, :, A:], B - A, keys)
    assert estimate.device.type == "cpu" and torch.isfinite(estimate).all() and estimate.max() > 0

    context = Context(gpu)
    context.put("before", 0, tokens[:A])
    context.put("after", 2, tokens[B:])
    context.put("late", 1, tokens[A:B])
    assert context.missing_mass(2).max() > 0
    assert context.repair(gpu.encode("Answer:"), 0.2) > 0
