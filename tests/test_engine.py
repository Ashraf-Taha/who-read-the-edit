import pytest
import torch

from mutablekv import LM, Cells

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]
TEXT = (
    "The harbour master kept two ledgers. In the first he wrote the name of every ship that entered, "
    "the flag it flew and the cargo it declared. In the second, which he showed to nobody, he wrote what "
    "the ship actually carried. On the ninth of March the brig Calliope declared four hundred barrels of "
    "salt cod; the second ledger records two hundred barrels of cod and the rest gunpowder. When the "
    "inspector arrived a week later he asked to see the ledger, and the harbour master handed him the first."
)


def gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@pytest.fixture(scope="module", params=MODELS)
def lm32(request):
    return LM(request.param, "cpu", torch.float32)


@pytest.fixture(scope="module")
def lm():
    return LM(MODELS[0], "cpu", torch.float64)


@pytest.fixture(scope="module")
def tokens(lm):
    return lm.encode(TEXT)[:96]


def test_cold_pass_matches_transformers(lm32):
    tokens = lm32.encode(TEXT)[:96]
    ours = lm32.logits(lm32.cold(tokens).hidden)
    with torch.inference_mode():
        theirs = lm32.hf(tokens[None]).logits[0]
    assert gap(ours, theirs) < 2e-3
    assert torch.equal(ours.argmax(-1), theirs.argmax(-1))


def test_appending_to_cached_cells_matches_cold(lm, tokens):
    n, cut = tokens.numel(), 40
    pos = torch.arange(n)
    whole = lm.cold(tokens)
    prefix = lm.cold(tokens[:cut])
    tail = lm.run(tokens, pos, fresh=pos >= cut, cached=prefix.cells)
    assert gap(tail.hidden, whole.hidden[cut:]) < 1e-9
    assert gap(tail.cells.K, whole.cells.K[:, :, cut:]) < 1e-9
    assert gap(tail.cells.V, whole.cells.V[:, :, cut:]) < 1e-9


def test_uniform_shift_changes_nothing_but_the_key_phase(lm, tokens):
    pos = torch.arange(tokens.numel())
    here, there = lm.cold(tokens, pos), lm.cold(tokens, pos + 1000)
    assert gap(here.hidden, there.hidden) < 1e-9
    assert gap(here.cells.V, there.cells.V) < 1e-9
    assert gap(lm.rotate(here.cells.K, torch.full_like(pos, 1000)), there.cells.K) < 1e-9
    assert gap(lm.unrotated(here.cells), lm.unrotated(there.cells)) < 1e-9


def test_query_side_relocation_equals_key_rewriting(lm, tokens):
    n = tokens.numel()
    cells = lm.cold(tokens[:72]).cells
    pos = torch.cat((torch.arange(24), torch.arange(24, 48) + 37, torch.arange(48, 72) + 45, torch.arange(120, 120 + n - 72)))
    fresh = torch.arange(n) >= 72
    by_query = lm.run(tokens, pos, fresh, cells, relocate="query")
    by_rewrite = lm.run(tokens, pos, fresh, cells, relocate="rewrite")
    assert gap(by_query.hidden, by_rewrite.hidden) < 1e-9
    assert gap(by_query.cells.K, by_rewrite.cells.K) < 1e-9


@pytest.mark.parametrize("relocate", ["query", "rewrite"])
def test_relocated_cells_are_exact_when_relative_geometry_is_unchanged(lm, tokens, relocate):
    n, a, b = tokens.numel(), 30, 70
    pos = torch.arange(n)
    index = torch.arange(n)
    first = lm.cold(tokens[:a]).cells
    far = torch.cat((pos[:a] + 1000, pos[a:b] + 1000))
    second = lm.run(tokens[:b], far, fresh=index[:b] >= a, cached=first).cells
    assert int(second.enc_pos[0]) == a + 1000

    tail = lm.run(tokens, pos, fresh=index >= b, cached=Cells.cat([first, second]), relocate=relocate)
    whole = lm.cold(tokens)
    assert gap(tail.hidden, whole.hidden[b:]) < 1e-9
    assert gap(lm.logits(tail.hidden), lm.logits(whole.hidden[b:])) < 1e-8


def test_recomputing_scattered_tokens_over_exact_cells_reproduces_them(lm, tokens):
    n = tokens.numel()
    pos = torch.arange(n)
    whole = lm.cold(tokens)
    fresh = torch.zeros(n, dtype=torch.bool)
    fresh[torch.tensor([3, 4, 17, 41, 42, 43, 80, n - 1])] = True
    again = lm.run(tokens, pos, fresh, whole.cells.take(~fresh))
    assert gap(again.cells.K, whole.cells.K[:, :, fresh]) < 1e-9
    assert gap(again.cells.V, whole.cells.V[:, :, fresh]) < 1e-9
    assert gap(again.hidden, whole.hidden[fresh]) < 1e-9


def test_later_tokens_do_not_affect_earlier_cells(lm, tokens):
    changed = tokens.clone()
    changed[-1] = tokens[0]
    a, b = lm.cold(tokens), lm.cold(changed)
    assert gap(a.cells.V[:, :, :-1], b.cells.V[:, :, :-1]) == 0.0
    assert gap(a.cells.V[:, :, -1], b.cells.V[:, :, -1]) > 1e-3


def test_attention_mass_is_conserved_and_causal(lm, tokens):
    n = tokens.numel()
    by_token = lm.cold(tokens, groups=torch.arange(n)).mass
    assert by_token.shape == (lm.L, n, n)
    assert gap(by_token.sum(-1), torch.ones(lm.L, n)) < 1e-5
    assert by_token.triu(1).abs().max().item() == 0.0
    by_block = lm.cold(tokens, groups=torch.arange(n) // 16).mass
    assert gap(by_block[:, :, 0], by_token[:, :, :16].sum(-1)) < 1e-5


def test_cached_cells_must_match_the_sequence(lm, tokens):
    cells = lm.cold(tokens[:10]).cells
    pos = torch.arange(20)
    with pytest.raises(ValueError):
        lm.run(tokens[:20], pos, fresh=pos >= 12, cached=cells)
    wrong = tokens[:20].clone()
    wrong[0] = tokens[5]
    with pytest.raises(ValueError):
        lm.run(wrong, pos, fresh=pos >= 10, cached=cells)
    with pytest.raises(ValueError):
        lm.run(tokens[:20], pos, fresh=pos >= 10, cached=cells, relocate="teleport")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_gpu_pass_agrees_with_cpu(lm32):
    tokens = lm32.encode(TEXT)[:96]
    cpu = lm32.logits(lm32.cold(tokens).hidden)
    gpu_lm = LM(lm32.repo, "mps", torch.float32)
    gpu = gpu_lm.logits(gpu_lm.cold(tokens).hidden).cpu()
    assert gap(cpu, gpu) < 5e-3
    assert (cpu.argmax(-1) == gpu.argmax(-1)).float().mean().item() == 1.0


def test_greedy_decoding_matches_uncached_library_loop(lm32):
    from mutablekv import greedy

    prompt = lm32.encode(TEXT)[:60]
    ids = prompt.clone()
    expected = []
    with torch.inference_mode():
        for _ in range(6):
            expected.append(int(lm32.hf(ids[None]).logits[0, -1].argmax()))
            ids = torch.cat((ids, torch.tensor([expected[-1]])))
    cached = lm32.cold(prompt[:40]).cells
    assert greedy(lm32, prompt, torch.arange(60), cached, 6, stop=set()) == expected
    assert greedy(lm32, prompt, torch.arange(60), cached, 6, stop={expected[2]}) == expected[:2]
    assert greedy(lm32, prompt, torch.arange(60), None, 6, stop=set()) == expected
    assert greedy(lm32, prompt, torch.arange(60), cached, 0, stop=set()) == []
