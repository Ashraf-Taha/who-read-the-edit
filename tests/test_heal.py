import pytest
import torch

from mutablekv import LM, Cells, heal, sweep
from tests.test_engine import MODELS, TEXT, gap


@pytest.fixture(scope="module")
def lm():
    return LM(MODELS[0], "cpu", torch.float64)


@pytest.fixture(scope="module")
def edited(lm):
    """A context whose middle was deleted: the cache holds the suffix as encoded before the deletion."""
    tokens = lm.encode(TEXT)[:96]
    a, b = 30, 50
    before = lm.cold(tokens).cells
    kept = torch.cat((tokens[:a], tokens[b:]))
    stale = Cells.cat([before.take(slice(0, a)), before.take(slice(b, None))])
    return kept, torch.arange(kept.numel()), stale, a, lm.cold(kept).cells


def test_stale_suffix_differs_from_the_cold_pass(lm, edited):
    _, _, stale, a, truth = edited
    assert gap(stale.V[:, :, :a], truth.V[:, :, :a]) < 1e-9
    assert gap(stale.V[:, :, a:], truth.V[:, :, a:]) > 1e-2


def test_healing_a_block_that_starts_at_the_edit_is_exact_for_that_block(lm, edited):
    tokens, pos, stale, a, truth = edited
    block = (pos >= a) & (pos < a + 12)
    healed = heal(lm, tokens, pos, stale, block)
    assert gap(healed.V[:, :, block], truth.V[:, :, block]) < 1e-9
    assert gap(lm.unrotated(healed)[:, :, block], lm.unrotated(truth)[:, :, block]) < 1e-9
    assert torch.equal(healed.enc_pos[block], pos[block])
    assert gap(healed.V[:, :, ~block], stale.V[:, :, ~block]) == 0.0


def test_healing_scattered_tokens_is_only_approximate_but_closer(lm, edited):
    tokens, pos, stale, a, truth = edited
    chosen = torch.zeros_like(pos, dtype=torch.bool)
    chosen[torch.tensor([a + 5, a + 20, a + 21, pos.numel() - 1])] = True
    healed = heal(lm, tokens, pos, stale, chosen)
    before = (stale.V[:, :, chosen] - truth.V[:, :, chosen]).norm().item()
    after = (healed.V[:, :, chosen] - truth.V[:, :, chosen]).norm().item()
    assert 1e-6 < after < before


def test_sweep_from_the_edit_reproduces_the_cold_pass(lm, edited):
    tokens, pos, stale, a, truth = edited
    swept = sweep(lm, tokens, pos, stale, start=a, chunk=17)
    assert gap(swept.V, truth.V) < 1e-9
    assert gap(lm.unrotated(swept), lm.unrotated(truth)) < 1e-9

    halfway = sweep(lm, tokens, pos, stale, start=a, stop=a + 34, chunk=17)
    assert gap(halfway.V[:, :, : a + 34], truth.V[:, :, : a + 34]) < 1e-9
    assert gap(halfway.V[:, :, a + 34:], stale.V[:, :, a + 34:]) == 0.0


def test_healing_nothing_or_everything(lm, edited):
    tokens, pos, stale, _, truth = edited
    assert heal(lm, tokens, pos, stale, torch.zeros_like(pos, dtype=torch.bool)) is stale
    everything = heal(lm, tokens, pos, stale, torch.ones_like(pos, dtype=torch.bool))
    assert gap(everything.V, truth.V) < 1e-9


def test_partial_depth_pass_matches_the_first_layers(lm):
    tokens = lm.encode(TEXT)[:48]
    full, shallow = lm.cold(tokens), lm.run(tokens, torch.arange(48), depth=3)
    assert shallow.cells.K.shape[0] == 3
    assert gap(shallow.cells.V, full.cells.V[:3]) < 1e-12
