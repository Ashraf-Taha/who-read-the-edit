"""E2: after an interior edit, which cached tokens should be recomputed?

Context P + E + S, then new input Q. Two directions share the same passes:

    delete   E is removed. The cache holds S as encoded while E was present.
    insert   E arrives late. The cache holds S as encoded before E existed.

A policy ranks the tokens of S; the top fraction is recomputed (reading every other
cell as cached) and Q is then compared with the cold pass over the edited context.

Signal                          Policy name    Needs at edit time
------------------------------  -------------  -----------------------------------------
position after the edit         boundary       nothing            (EPIC, KVEraser "erase" window)
position before the new input   tail           nothing            (KVEraser "query" window)
uniform                         random         nothing
token ids shared with E         lexical        nothing
first-layer deviation           probe1         two layers over S  (CacheBlend's selection signal)
attention of Q on the token     reader         Q over stale cache (ProphetKV's selection signal)
attention the token put on E    exposure       recorded lineage   (delete only; cf. Cache-Craft)
  same, first layer only        exposure_l1    one attention map from embeddings (both directions)
exposure through the graph      rollout        recorded token graph
exposure x reader               bridge         lineage + Q pass
exposure_l1 x reader            bridge_l1      first-layer map + Q pass (both directions)
lexical x reader                bridge_lex     Q pass
exposure x reader via graph     bridge_paths   lineage + token graph + Q pass
true deviation from cold pass   oracle         not available in practice

For insert, `exposure`, `rollout`, `bridge` and `bridge_paths` take exposure from the cold
pass with E present. A real system does not have that; those rows show what a perfect
estimate of would-be attention buys. `exposure_l1` is exact in both directions, because
first-layer queries and keys depend on token embeddings and positions only.

Usage: e2_healing.py <model-key> <gold-cases-per-place> [tag] [budgets, comma separated]
"""
from __future__ import annotations

import math
import sys

import torch
from scipy.stats import wilcoxon

from common import MODELS, best_device, kl_rows, mean_ci, normalise, resume, save
from mutablekv import LM, Cells, greedy, heal
from workloads import READER_PLACES, Case, reader_cases

BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.40]
DECODE_AT = 0.10
DECODED = ("boundary", "tail", "reader", "bridge", "bridge_l1", "bridge_qs4g")
BASELINES = ("boundary", "tail", "random", "lexical", "probe1", "reader", "exposure", "oracle")


def relative_deviation(lm: LM, a: Cells, b: Cells) -> torch.Tensor:
    ka, kb = lm.unrotated(a), lm.unrotated(b)
    num = (a.V - b.V).pow(2).sum((1, 3)) + (ka - kb).pow(2).sum((1, 3))
    den = b.V.pow(2).sum((1, 3)) + kb.pow(2).sum((1, 3))
    return (num / den).float().cpu()


def shallow(cells: Cells, depth: int = 2) -> Cells:
    return Cells(cells.tokens, cells.enc_pos, cells.K[:depth], cells.V[:depth])


def decode(lm: LM, tokens: torch.Tensor, pos: torch.Tensor, cached: Cells) -> str:
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    return lm.tok.decode(greedy(lm, tokens, pos, cached, 16, stop)).strip().split("\n")[0]


def evaluate(lm: LM, context: torch.Tensor, Q: torch.Tensor, stale: Cells, truth: Cells, start: int,
             exposure_by_layer: torch.Tensor, graph: torch.Tensor, edited: torch.Tensor, reader_span: tuple[int, int],
             policies: list[str] | None = None, extra_exposure: dict[str, torch.Tensor] | None = None) -> dict:
    """Heal the suffix context[start:] of `stale` under each policy and budget; read Q; compare with `truth`.

    extra_exposure: further estimates of each suffix token's exposure to the edit; each adds
    the policies exposure_<name> and bridge_<name> (the estimate times reader attention).
    policies: evaluate only these.
    """
    n, nS = context.numel(), context.numel() - start
    pos = torch.arange(n)
    in_suffix = pos >= start
    asked, asked_pos = torch.cat((context, Q)), torch.arange(n + Q.numel())
    new_input = asked_pos >= n

    def read(cells: Cells, groups: torch.Tensor | None = None):
        result = lm.run(asked, asked_pos, new_input, cells, groups=groups)
        return lm.logits(result.hidden).float().cpu(), result.mass

    truth_logits, _ = read(truth)
    stale_logits, reader_mass = read(stale, groups=asked_pos)
    reader = reader_mass[:, :, start:n].mean((0, 1))

    exposure = exposure_by_layer.mean(0)
    rollout = torch.zeros(nS)
    for layer in range(graph.shape[0]):
        rollout = rollout + graph[layer] @ rollout + exposure_by_layer[layer]
    paths = torch.linalg.solve_triangular((torch.eye(nS) - graph.mean(0)).T, reader[:, None], upper=True).squeeze(1)

    prefix = stale.take(slice(0, start))
    probe = lm.run(context, pos, in_suffix, shallow(prefix), depth=2).cells
    S = context[start:]
    order = torch.arange(nS).float()
    rarity = torch.bincount(torch.cat((context, edited))).float()
    scores = {
        "boundary": -order,
        "tail": order,
        "random": torch.rand(nS, generator=torch.Generator().manual_seed(n * 7919 + nS)),
        "lexical": torch.isin(S, edited).float() / rarity[S] - 1e-9 * order,
        "exposure": exposure,
        "rollout": rollout,
        "reader": reader,
        "exposure_l1": exposure_by_layer[0],
        "bridge": exposure * reader,
        "bridge_l1": exposure_by_layer[0] * reader,
        "bridge_lex": (torch.isin(S, edited).float() / rarity[S] + 1e-6) * reader,
        "bridge_paths": exposure * paths,
        "probe1": relative_deviation(lm, shallow(stale.take(slice(start, None))), probe)[1],
        "oracle": relative_deviation(lm, stale.take(slice(start, None)), truth.take(slice(start, None))).mean(0),
    }
    for name, estimate in (extra_exposure or {}).items():
        scores[f"exposure_{name}"] = estimate
        scores[f"bridge_{name}"] = estimate * reader
    if policies is not None:
        scores = {name: scores[name] for name in policies}

    def outcome(logits: torch.Tensor) -> dict:
        divergence = kl_rows(truth_logits, logits)
        return {"kl_last": divergence[-1].item(), "kl_mean": divergence.mean().item(),
                "top1_last": bool(truth_logits[-1].argmax() == logits[-1].argmax())}

    results = {"stale": {"0": outcome(stale_logits)}}
    answers = {"truth": decode(lm, asked, asked_pos, truth), "stale": decode(lm, asked, asked_pos, stale)}
    on_reader = {}
    for name, score in scores.items():
        results[name] = {}
        for budget in BUDGETS:
            top = score.topk(max(1, math.ceil(budget * nS))).indices
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + top] = True
            healed = heal(lm, context, pos, stale, chosen)
            results[name][str(budget)] = outcome(read(healed)[0])
            if budget == DECODE_AT:
                on_reader[name] = ((top >= reader_span[0]) & (top < reader_span[1])).float().mean().item()
                if name in DECODED:
                    answers[name] = decode(lm, asked, asked_pos, healed)

    check = outcome(read(heal(lm, context, pos, stale, in_suffix))[0])["kl_last"]
    if check > 1e-4:
        raise AssertionError(f"recomputing the whole suffix should reproduce the cold pass, KL={check}")
    return {"results": results, "answers": answers, "share_of_picks_on_cached_question": on_reader, "suffix_tokens": nS}


def analyse(lm: LM, case: Case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    before_question = sum(s.numel() for s in case.segments[case.edit + 1: case.meta["question_segment"]])
    reader_span = (before_question, before_question + case.segments[case.meta["question_segment"]].numel())

    with_e, without_e = torch.cat((P, E, S)), torch.cat((P, S))
    original = lm.cold(with_e, groups=torch.arange(nP + nE + nS))
    lineage = original.mass[:, nP + nE:]
    exposure_by_layer = lineage[:, :, nP: nP + nE].sum(-1)
    graph = lineage[:, :, nP + nE:].clone()
    graph.diagonal(dim1=1, dim2=2).zero_()
    cells = original.cells
    del original, lineage

    cP, cE, with_S = cells.take(slice(0, nP)), cells.take(slice(nP, nP + nE)), cells.take(slice(nP + nE, None))
    without_S = lm.run(without_e, torch.arange(nP + nS), torch.arange(nP + nS) >= nP, cP).cells

    shared = dict(exposure_by_layer=exposure_by_layer, graph=graph, edited=E, reader_span=reader_span)
    return {
        "name": case.name, **case.meta, "nP": nP, "nE": nE, "nS": nS, "nQ": Q.numel(),
        "delete": evaluate(lm, without_e, Q, Cells.cat([cP, with_S]), Cells.cat([cP, without_S]), nP, **shared),
        "insert": evaluate(lm, with_e, Q, Cells.cat([cP, cE, without_S]), cells, nP + nE, **shared),
    }


def paired(chosen: list[dict], ours: str, other: str, budget: str) -> str:
    a = [r["results"][ours][budget]["kl_last"] for r in chosen]
    b = [r["results"][other][budget]["kl_last"] for r in chosen]
    wins = sum(x < y for x, y in zip(a, b)) / len(a)
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    return f"{wins:.2f} (p={p:.1e})"


def report(rows: list[dict]) -> None:
    budgets = sorted({b for r in rows for b in r["delete"]["results"]["bridge"]}, key=float)
    for direction in ("delete", "insert"):
        for group in sorted({(r["kind"], r["place"]) for r in rows}):
            chosen = [r[direction] for r in rows if (r["kind"], r["place"]) == group]
            stale, half = mean_ci([r["results"]["stale"]["0"]["kl_last"] for r in chosen])
            agree = mean_ci([float(r["results"]["stale"]["0"]["top1_last"]) for r in chosen])[0]
            same = mean_ci([float(normalise(r["answers"]["stale"]) == normalise(r["answers"]["truth"])) for r in chosen])[0]
            print(f"\n=== {direction} a {group[0]} document / cached question is {group[1]} / n={len(chosen)} "
                  f"(S {mean_ci([r['suffix_tokens'] for r in chosen])[0]:.0f} tokens)")
            print(f"  no repair: KL {stale:.4f} ± {half:.4f}, first token agrees {agree:.3f}, answer identical {same:.3f}")
            print("    policy        " + "".join(f"{float(b):>16.0%}" for b in budgets) + "   answer identical @10%")
            for name in [k for k in chosen[0]["results"] if k != "stale"]:
                line = ""
                for budget in budgets:
                    kl, hw = mean_ci([r["results"][name][budget]["kl_last"] for r in chosen])
                    line += f"  {kl:.4f}±{hw:.4f}"
                if name in chosen[0]["answers"]:
                    m, h = mean_ci([float(normalise(r["answers"][name]) == normalise(r["answers"]["truth"])) for r in chosen])
                    line += f"   {m:.3f} ± {h:.3f}"
                print(f"    {name:13s} " + line)
            if "0.1" in budgets and len(chosen) >= 6:
                for ours in [k for k in ("bridge", "bridge_l1") if k in chosen[0]["results"]]:
                    print(f"  share of cases where {ours} has lower KL at 10% than (Wilcoxon signed-rank p):")
                    print("    " + "   ".join(f"{other} {paired(chosen, ours, other, '0.1')}" for other in BASELINES if other in chosen[0]["results"]))


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    if len(sys.argv) > 4:
        BUDGETS[:] = [float(b) for b in sys.argv[4].split(",")]
    lm = LM(MODELS[key], best_device(), torch.float32)
    cases = []
    for place in READER_PLACES:
        cases += reader_cases(lm, count, "gold", place) + reader_cases(lm, max(1, count // 3), "distractor", place)
    stem, settings = f"e2_healing_{key}_{tag}", {"model": MODELS[key], "budgets": list(BUDGETS)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}")
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        if len(rows) % 10 == 0:
            save(stem, {**settings, "rows": rows})
    print("saved", save(stem, {**settings, "rows": rows}))
    report(rows)


if __name__ == "__main__":
    main()
