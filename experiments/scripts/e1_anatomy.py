"""E1: where does staleness live after an interior edit?

For a context P + E + S followed by new input Q, three encodings of S are compared:

    cached   S encoded with E present (what a cache holds before E is deleted)
    hole     S encoded without E, at its original positions (a gap where E was)
    closed   S encoded without E, moved up against P (the cold pass after deletion)

    cached vs closed = everything wrong with reusing the cache after deleting E
    hole   vs closed = geometry alone: same content, distance to P differs by |E|
    cached vs hole   = read set alone: same positions, E was or was not attended

Read backwards the same three states describe E arriving late and being inserted.

Usage: e1_anatomy.py <model-key> <cases-per-workload> [tag]
"""
from __future__ import annotations

import sys

import torch
from scipy.stats import spearmanr

from common import MODELS, answer_f1, best_device, kl_rows, mean_ci, normalise, resume, save
from mutablekv import LM, Cells, greedy
from workloads import Case, hotpot_cases, wikitext_cases

BUDGETS = (0.05, 0.10, 0.20)
DISTANCE_EDGES = (0, 8, 32, 128, 512, 10**9)


def deviation(lm: LM, a: Cells, b: Cells) -> torch.Tensor:
    """Relative squared distance between two encodings of the same tokens, per layer and token."""
    ka, kb = lm.unrotated(a), lm.unrotated(b)
    num = (a.V - b.V).pow(2).sum((1, 3)) + (ka - kb).pow(2).sum((1, 3))
    den = b.V.pow(2).sum((1, 3)) + kb.pow(2).sum((1, 3))
    return (num / den).float().cpu()


def captured(score: torch.Tensor, energy: torch.Tensor, budget: float) -> float:
    k = max(1, round(budget * energy.numel()))
    return (energy[score.topk(k).indices].sum() / energy.sum()).item()


def ask(lm: LM, tokens: torch.Tensor, pos: torch.Tensor, cached: Cells) -> torch.Tensor:
    fresh = torch.arange(tokens.numel()) >= len(cached)
    return lm.logits(lm.run(tokens, pos, fresh, cached).hidden).float().cpu()


def answer(lm: LM, tokens: torch.Tensor, pos: torch.Tensor, cached: Cells) -> str:
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    return lm.tok.decode(greedy(lm, tokens, pos, cached, 16, stop)).strip().split("\n")[0]


def analyse(lm: LM, case: Case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS, nQ = P.numel(), E.numel(), S.numel(), Q.numel()
    span = torch.arange

    original = lm.cold(torch.cat((P, E, S)), groups=torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long())
    cP, cE, cached = original.cells.take(slice(0, nP)), original.cells.take(slice(nP, nP + nE)), original.cells.take(slice(nP + nE, None))
    without = torch.cat((P, S))
    after_p = span(nP + nS) >= nP
    closed = lm.run(without, span(nP + nS), after_p, cP).cells
    hole_pos = torch.cat((span(nP), nP + nE + span(nS)))
    hole = lm.run(without, hole_pos, after_p, cP).cells

    total, geometry, readset = deviation(lm, cached, closed), deviation(lm, hole, closed), deviation(lm, cached, hole)
    exposure = original.mass[:, nP + nE:, 1]

    with_e, with_q = torch.cat((P, E, S, Q)), torch.cat((P, S, Q))
    full_pos, short_pos = span(nP + nE + nS + nQ), span(nP + nS + nQ)
    gapped_pos = torch.cat((hole_pos, nP + nE + nS + span(nQ)))
    states = {
        "kept_e": (with_e, full_pos, original.cells),
        "deleted_truth": (with_q, short_pos, Cells.cat([cP, closed])),
        "deleted_relocated": (with_q, short_pos, Cells.cat([cP, cached])),
        "deleted_hole_stale": (with_q, gapped_pos, Cells.cat([cP, cached])),
        "deleted_hole_exact": (with_q, gapped_pos, Cells.cat([cP, hole])),
        "inserted_relocated": (with_e, full_pos, Cells.cat([cP, cE, closed])),
    }
    out = {name: ask(lm, *state) for name, state in states.items()}
    answers = {name: answer(lm, *state) for name, state in states.items()} if "answer" in case.meta else {}

    def versus(reference: str, other: str) -> dict:
        divergence = kl_rows(out[reference], out[other])
        return {"kl_mean": divergence.mean().item(), "kl_last": divergence[-1].item(),
                "top1_last": bool(out[reference][-1].argmax() == out[other][-1].argmax()),
                "top1_all": (out[reference].argmax(-1) == out[other].argmax(-1)).float().mean().item()}

    order = span(nS).float()
    rarity = torch.bincount(torch.cat((P, E, S)), minlength=int(max(P.max(), E.max(), S.max())) + 1).float()
    lexical = torch.isin(S, E).float() / rarity[S]
    scores = {
        "exposure_mean": exposure.mean(0), "exposure_max": exposure.max(0).values,
        "exposure_early": exposure[: lm.L // 3].mean(0), "exposure_late": exposure[-(lm.L // 3):].mean(0),
        "boundary_window": -order, "tail_window": order, "lexical_overlap": lexical,
    }
    targets = {"total": total.mean(0), "readset": readset.mean(0), "geometry": geometry.mean(0)}
    ranking = {}
    for target_name, energy in targets.items():
        ranking[target_name] = {"oracle": {f"energy@{b}": captured(energy, energy, b) for b in BUDGETS}}
        for score_name, score in scores.items():
            ranking[target_name][score_name] = {
                "spearman": float(spearmanr(score.numpy(), energy.numpy()).statistic),
                **{f"energy@{b}": captured(score, energy, b) for b in BUDGETS},
            }

    by_distance = {}
    for low, high in zip(DISTANCE_EDGES, DISTANCE_EDGES[1:]):
        inside = (order >= low) & (order < high)
        if inside.any():
            by_distance[f"{low}-{high if high < 10**9 else 'end'}"] = {name: energy[inside].mean().item() for name, energy in targets.items()}

    return {
        "name": case.name, **case.meta, "nP": nP, "nE": nE, "nS": nS, "nQ": nQ,
        "mean_deviation": {name: energy.mean().item() for name, energy in targets.items()},
        "by_layer": {"total": total.mean(1).tolist(), "readset": readset.mean(1).tolist(), "geometry": geometry.mean(1).tolist(),
                     "exposure": exposure.mean(1).tolist()},
        "by_distance": by_distance,
        "ranking": ranking,
        "exposure_mass_total": exposure.mean(0).sum().item(),
        "answers": answers,
        "tail_segment_tokens": case.segments[case.fresh_from - 1].numel(),
        "output": {
            "delete/significance": versus("deleted_truth", "kept_e"),
            "delete/relocated": versus("deleted_truth", "deleted_relocated"),
            "delete/hole_stale": versus("deleted_truth", "deleted_hole_stale"),
            "delete/hole_exact": versus("deleted_truth", "deleted_hole_exact"),
            "insert/significance": versus("kept_e", "deleted_truth"),
            "insert/relocated": versus("kept_e", "inserted_relocated"),
        },
        "per_token": {"total": [round(x, 6) for x in targets["total"].tolist()],
                      "readset": [round(x, 6) for x in targets["readset"].tolist()],
                      "geometry": [round(x, 6) for x in targets["geometry"].tolist()],
                      "exposure_mean": [round(x, 6) for x in scores["exposure_mean"].tolist()]},
    }


def report(rows: list[dict]) -> None:
    groups = sorted({(r["workload"], r["kind"], r.get("question_is_fresh")) for r in rows})
    for group in groups:
        chosen = [r for r in rows if (r["workload"], r["kind"], r.get("question_is_fresh")) == group]
        print(f"\n=== {group[0]} / edited segment: {group[1]} / question fresh: {group[2]} / n={len(chosen)} "
              f"(P {mean_ci([r['nP'] for r in chosen])[0]:.0f}, E {mean_ci([r['nE'] for r in chosen])[0]:.0f}, S {mean_ci([r['nS'] for r in chosen])[0]:.0f} tokens)")
        for name in ("total", "readset", "geometry"):
            m, ci = mean_ci([r["mean_deviation"][name] for r in chosen])
            print(f"  mean relative sq. deviation of S cells, {name:9s}: {m:.4f} ± {ci:.4f}")
        print("  share of read-set deviation captured by the top 10% of S tokens ranked by:")
        for score in ("oracle", "exposure_mean", "exposure_max", "exposure_early", "exposure_late", "lexical_overlap", "boundary_window", "tail_window"):
            m, ci = mean_ci([r["ranking"]["readset"][score]["energy@0.1"] for r in chosen])
            rho = "" if score == "oracle" else f"   spearman {mean_ci([r['ranking']['readset'][score]['spearman'] for r in chosen])[0]:+.2f}"
            print(f"    {score:16s} {m:.3f} ± {ci:.3f}{rho}")
        print("  output vs the correct state (KL in nats at the last position and averaged over new input; top-1 agreement at last):")
        for key in ("delete/significance", "delete/relocated", "delete/hole_stale", "delete/hole_exact", "insert/significance", "insert/relocated"):
            last, last_ci = mean_ci([r["output"][key]["kl_last"] for r in chosen])
            mean, mean_half = mean_ci([r["output"][key]["kl_mean"] for r in chosen])
            agree = mean_ci([float(r["output"][key]["top1_last"]) for r in chosen])[0]
            print(f"    {key:22s} last {last:.4f} ± {last_ci:.4f}   mean {mean:.4f} ± {mean_half:.4f}   top-1 {agree:.3f}")
        if chosen[0]["answers"]:
            print("  decoded answers: same as the correct state | F1 against the dataset answer")
            for state, reference in (("deleted_truth", "deleted_truth"), ("deleted_relocated", "deleted_truth"), ("deleted_hole_stale", "deleted_truth"),
                                     ("deleted_hole_exact", "deleted_truth"), ("kept_e", "kept_e"), ("inserted_relocated", "kept_e")):
                same, half = mean_ci([float(normalise(r["answers"][state]) == normalise(r["answers"][reference])) for r in chosen])
                f1 = mean_ci([answer_f1(r["answers"][state], r["answer"]) for r in chosen])[0]
                print(f"    {state:20s} {same:.3f} ± {half:.3f} | {f1:.3f}")


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    lm = LM(MODELS[key], best_device(), torch.float32)
    cases = (hotpot_cases(lm, count, "distractor") + hotpot_cases(lm, count, "gold")
             + hotpot_cases(lm, count, "gold", question_is_fresh=False, seed=1) + wikitext_cases(lm, count))
    stem = f"e1_anatomy_{key}_{tag}"
    rows = resume(stem, model=MODELS[key]).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        if len(rows) % 20 == 0:
            save(stem, {"model": MODELS[key], "device": lm.device, "rows": rows})
    print("saved", save(stem, {"model": MODELS[key], "device": lm.device, "rows": rows}))
    report(rows)


if __name__ == "__main__":
    main()
