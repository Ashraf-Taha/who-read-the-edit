import pytest
import torch

from mutablekv import LM, Context, LateEvent, Stream
from tests.test_engine import MODELS, gap

PARTS = {
    "rules": "You keep the harbour ledger. Answer from the entries only.\n",
    "calliope": "Entry: the brig Calliope declared four hundred barrels of salt cod on the ninth of March.\n",
    "heron": "Entry: the sloop Heron declared sixty bolts of sailcloth and paid duty in silver.\n",
    "inspector": "Entry: the inspector arrived a week later and asked to see the ledger.\n",
    "note": "Note: the Calliope entry and the Heron entry were written by the same clerk.\n",
    "ask": "Question: which ship carried sailcloth?\n",
}
ORDER = {"rules": 0, "calliope": 1, "heron": 2, "inspector": 3, "note": 4, "ask": 5}


@pytest.fixture(scope="module")
def lm():
    return LM(MODELS[0], "cpu", torch.float64)


@pytest.fixture(scope="module")
def text(lm):
    return {key: lm.encode(value) for key, value in PARTS.items()}


def build(lm, text, keys):
    context = Context(lm)
    for key in keys:
        context.put(key, ORDER[key], text[key])
    return context


def same_as_cold(context) -> float:
    truth = context.lm.cold(context.tokens()).cells
    mine = context.cells()
    return max(gap(mine.V, truth.V), gap(context.lm.unrotated(mine), context.lm.unrotated(truth)))


def test_appending_in_order_is_exact(lm, text):
    context = build(lm, text, list(PARTS))
    assert context.frontier() == len(PARTS)
    assert context.stale_tokens() == 0
    assert same_as_cold(context) < 1e-9
    assert context.keys() == list(PARTS)


def test_deleting_a_segment_leaves_followers_stale_until_swept(lm, text):
    context = build(lm, text, list(PARTS))
    before = {s.key: s for s in context.segments}
    context.drop("heron")
    assert context.keys() == ["rules", "calliope", "inspector", "note", "ask"]
    assert context.frontier() == 2
    assert all(context.segments[i].cells.K is before[k].cells.K for i, k in enumerate(context.keys()))
    assert same_as_cold(context) > 1e-3
    assert all(s.debt.max() > 0 for s in context.segments[2:])
    assert all(s.debt.sum() == 0 for s in context.segments[:2])

    spent = context.sweep()
    assert spent == sum(len(text[k]) for k in ("inspector", "note", "ask"))
    assert context.frontier() == 5 and same_as_cold(context) < 1e-9
    assert all(s.debt.sum() == 0 and not s.missing for s in context.segments)


def test_sweep_respects_its_budget_and_keeps_an_exact_frontier(lm, text):
    context = build(lm, text, list(PARTS))
    context.drop("calliope")
    spent = context.sweep(budget=len(text["heron"]) + len(text["inspector"]))
    assert spent == len(text["heron"]) + len(text["inspector"])
    assert context.frontier() == 3
    exact = Context(lm, context.segments[:3])
    assert same_as_cold(exact) < 1e-9
    assert context.sweep(budget=1) == 0


@pytest.mark.parametrize("arrival", [
    ["rules", "heron", "ask", "calliope", "note", "inspector"],
    ["ask", "note", "inspector", "heron", "calliope", "rules"],
    ["rules", "calliope", "inspector", "note", "ask", "heron"],
])
def test_any_arrival_order_gives_the_same_sequence_and_the_same_state_after_sweep(lm, text, arrival):
    context = build(lm, text, arrival)
    assert context.keys() == list(PARTS)
    assert context.frontier() < len(PARTS)
    assert any(s.missing for s in context.segments)
    context.sweep()
    assert context.frontier() == len(PARTS)
    assert same_as_cold(context) < 1e-9
    in_order = build(lm, text, list(PARTS))
    assert [s.version for s in context.segments] == [s.version for s in in_order.segments]


def test_undoing_an_edit_restores_exactness_without_recomputation(lm, text):
    context = build(lm, text, list(PARTS))
    versions = [s.version for s in context.segments]
    context.drop("heron")
    assert context.frontier() == 2
    context.put("heron", ORDER["heron"], text["heron"])
    assert [s.version for s in context.segments] == versions
    assert context.frontier() == len(PARTS) and context.stale_tokens() == 0
    assert not any(s.missing for s in context.segments)
    assert same_as_cold(context) < 1e-9

    context.swap("heron", text["calliope"])
    assert context.frontier() == 3
    assert all(s.missing and s.debt.max() > 0 for s in context.segments[3:])
    context.swap("heron", text["heron"])
    assert context.frontier() == len(PARTS)
    assert [s.version for s in context.segments] == versions


def test_repaired_cells_never_count_as_exact(lm, text):
    context = build(lm, text, list(PARTS))
    context.drop("heron")
    context.repair(lm.encode("Answer:"), 0.3)
    context.put("heron", ORDER["heron"], text["heron"])
    assert context.frontier() < len(PARTS)
    context.sweep()
    assert same_as_cold(context) < 1e-9
    assert [s.version for s in context.segments] == [s.version for s in build(lm, text, list(PARTS)).segments]


def test_a_late_segment_is_exact_itself_and_only_followers_are_stale(lm, text):
    context = build(lm, text, ["rules", "calliope", "inspector"])
    context.put("heron", ORDER["heron"], text["heron"])
    assert context.keys() == ["rules", "calliope", "heron", "inspector"]
    assert context.frontier() == 3
    assert [s.missing for s in context.segments] == [False, False, False, True]


def test_fork_shares_cells_and_isolates_changes(lm, text):
    parent = build(lm, text, ["rules", "calliope", "heron", "inspector"])
    child = parent.fork()
    assert all(a is b for a, b in zip(parent.segments, child.segments))
    child.drop("calliope")
    child.put("ask", ORDER["ask"], text["ask"])
    assert parent.keys() == ["rules", "calliope", "heron", "inspector"] and parent.frontier() == 4
    assert child.keys() == ["rules", "heron", "inspector", "ask"]
    assert child.segments[1].cells.K is parent.segments[2].cells.K
    assert same_as_cold(parent) < 1e-9


def test_merging_two_branches_keeps_both_sets_of_cells(lm, text):
    base = build(lm, text, ["rules", "calliope"])
    left, right = base.fork(), base.fork()
    left.put("heron", ORDER["heron"], text["heron"])
    right.put("inspector", ORDER["inspector"], text["inspector"])
    right_cells = right.segments[-1].cells
    left.merge(right)
    assert left.keys() == ["rules", "calliope", "heron", "inspector"]
    assert left.segments[3].cells.K is right_cells.K
    assert left.frontier() == 3 and left.segments[3].missing
    left.sweep()
    assert same_as_cold(left) < 1e-9

    other_way = right.fork()
    other_way.merge(build(lm, text, ["rules", "calliope", "heron"]))
    assert other_way.keys() == left.keys()
    with pytest.raises(ValueError):
        clash = build(lm, text, ["rules"])
        changed = Context(lm)
        changed.put("rules", 0, text["ask"])
        clash.merge(changed)


def test_merge_is_idempotent_and_order_independent_at_the_token_level(lm, text):
    a = build(lm, text, ["rules", "calliope", "note"])
    b = build(lm, text, ["rules", "heron", "ask"])
    ab, ba = a.fork(), b.fork()
    ab.merge(b)
    ba.merge(a)
    assert torch.equal(ab.tokens(), ba.tokens())
    again = ab.fork()
    again.merge(b)
    assert torch.equal(again.tokens(), ab.tokens()) and again.keys() == ab.keys()


def test_replacing_a_segment_records_debt_and_missing_reads(lm, text):
    context = build(lm, text, ["rules", "calliope", "note", "ask"])
    context.swap("calliope", text["heron"])
    assert context.keys() == ["rules", "calliope", "note", "ask"]
    assert context.frontier() == 2
    assert all(s.missing and s.debt.max() > 0 for s in context.segments[2:])
    context.sweep()
    assert same_as_cold(context) < 1e-9


def test_repair_recomputes_the_requested_share_and_moves_the_output_towards_the_cold_pass(lm, text):
    context = build(lm, text, ["rules", "calliope", "heron", "inspector", "note", "ask"])
    context.drop("heron")
    reader = lm.encode("Answer:")
    truth = Context(lm, list(context.segments))
    truth.sweep()
    target = torch.log_softmax(truth.read(reader)[0][-1], -1)
    stale = torch.log_softmax(context.read(reader)[0][-1], -1)
    stale_tokens = context.stale_tokens()

    spent = context.repair(reader, fraction=0.25)
    assert spent == -(-stale_tokens // 4)
    repaired = torch.log_softmax(context.read(reader)[0][-1], -1)
    kl = lambda q: (target.exp() * (target - q)).sum().item()
    assert kl(repaired) < kl(stale)
    assert context.frontier() == 2 and any(s.patched for s in context.segments)

    context.sweep()
    assert same_as_cold(context) < 1e-9 and not any(s.patched for s in context.segments)
    assert context.repair(reader, 0.5) == 0


def test_watermark_seals_a_prefix_and_rejects_late_events(lm, text):
    stream = Stream(Context(lm))
    for key in ["rules", "inspector", "calliope"]:
        stream.offer(key, ORDER[key], text[key])
    assert stream.context.frontier() == 2
    spent = stream.advance(ORDER["note"])
    assert spent == len(text["inspector"])
    assert stream.sealed() == 3 and stream.context.frontier() == 3
    with pytest.raises(LateEvent):
        stream.offer("heron", ORDER["heron"], text["heron"])
    with pytest.raises(LateEvent):
        stream.retract("calliope")
    with pytest.raises(ValueError):
        stream.advance(ORDER["calliope"])
    stream.offer("ask", ORDER["ask"], text["ask"])
    stream.offer("note", ORDER["note"], text["note"])
    assert stream.context.keys() == ["rules", "calliope", "inspector", "note", "ask"]
    assert stream.advance(10) == len(text["ask"])
    assert same_as_cold(stream.context) < 1e-9


def test_missing_mass_is_zero_unless_something_unseen_now_precedes_the_segment(lm, text):
    context = build(lm, text, ["rules", "calliope", "inspector", "note", "ask"])
    assert all(context.missing_mass(i).abs().max() == 0 for i in range(len(context.segments)))
    context.put("heron", ORDER["heron"], text["heron"])
    masses = [context.missing_mass(i) for i in range(len(context.segments))]
    assert [m.max().item() > 0 for m in masses] == [False, False, False, True, True, True]
    assert all(((m >= 0) & (m <= 1)).all() for m in masses)


def test_missing_mass_tracks_the_attention_a_cold_pass_pays_to_the_late_segment(lm, text):
    context = build(lm, text, ["rules", "calliope", "inspector", "note", "ask"])
    context.put("heron", ORDER["heron"], text["heron"])
    estimate = torch.cat([context.missing_mass(i) for i in range(3, 6)])
    starts = context.starts()
    groups = torch.cat([torch.full((len(s),), j) for j, s in enumerate(context.segments)])
    cold = lm.cold(context.tokens(), groups=groups)
    truth = cold.mass[context.sketch_layers][:, starts[3]:, 2].mean(0)
    rank = lambda x: x.argsort().argsort().double()
    assert torch.corrcoef(torch.stack((rank(estimate), rank(truth))))[0, 1] > 0.6


@pytest.mark.parametrize("estimate", ["sketch", "flat"])
def test_repair_after_a_late_arrival_patches_the_requested_share_and_a_sweep_still_converges(lm, text, estimate):
    context = build(lm, text, ["rules", "calliope", "inspector", "note", "ask"])
    context.put("heron", ORDER["heron"], text["heron"])
    stale = context.stale_tokens()
    assert stale == sum(len(text[k]) for k in ("inspector", "note", "ask"))
    spent = context.repair(lm.encode("Answer:"), 0.3, estimate=estimate)
    assert spent == -(-3 * stale // 10)
    patched = [s for s in context.segments if s.patched]
    assert patched and all(s.unsettled is not None and s.unsettled.min() == 0 for s in patched)
    assert sum(int((s.unsettled == 0).sum()) for s in patched) == spent
    assert context.frontier() == 3
    context.sweep()
    assert same_as_cold(context) < 1e-9 and not any(s.patched for s in context.segments)


def test_repair_rejects_an_unknown_estimate(lm, text):
    context = build(lm, text, ["rules", "heron", "ask"])
    context.drop("heron")
    with pytest.raises(ValueError):
        context.repair(lm.encode("Answer:"), 0.2, estimate="guess")


def test_reader_first_repair_recomputes_the_most_read_stale_segment_whole(lm, text):
    context = build(lm, text, ["rules", "calliope", "heron", "inspector", "note", "ask"])
    context.drop("heron")
    reader = lm.encode("Answer:")
    first, starts, total = context.frontier(), context.starts(), len(context)
    _, mass = context.read(reader, groups=torch.arange(total + reader.numel()))
    attention = mass[:, :, :total].mean((0, 1))
    per_token = [attention[starts[i]:starts[i] + len(context.segments[i])].mean().item() for i in range(first, len(context.segments))]
    best = first + max(range(len(per_token)), key=per_token.__getitem__)
    stale = context.stale_tokens()
    spent = context.repair(reader, fraction=0.5, reader_first=True)
    assert spent == -(-stale // 2) and spent >= len(context.segments[best])
    assert context.segments[best].patched and context.segments[best].unsettled.max() == 0
    context.sweep()
    assert same_as_cold(context) < 1e-9
