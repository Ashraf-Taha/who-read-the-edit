"""Turn experiment results into the numbers and tables the paper cites.

Writes paper/generated/values.json ({name: formatted value}) and one Markdown file per
table in paper/generated/. Every number in paper/src/ must come from here; build.py
refuses to build a draft with a placeholder this script did not produce.

    python paper/make_numbers.py
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "generated"
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from common import answer_f1, normalise  # noqa: E402

NAMES = {
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
}
KEYS = {"Qwen/Qwen2.5-0.5B-Instruct": "qwen0.5b", "Qwen/Qwen2.5-1.5B-Instruct": "qwen1.5b", "HuggingFaceTB/SmolLM2-1.7B-Instruct": "smol1.7b"}
values: dict[str, str] = {}


def load(stem: str) -> dict | None:
    path = RESULTS / f"{stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


def bca_bounds(observed: float, boot, jack) -> tuple[float, float]:
    """95% bias-corrected and accelerated bounds from bootstrap replicates and jackknife (leave-one-cluster-out) values."""
    import numpy as np
    from scipy.stats import norm

    boot, jack = np.asarray(boot, dtype=float), np.asarray(jack, dtype=float)
    below = (boot < observed).mean() + 0.5 * (boot == observed).mean()
    if not 0 < below < 1 or len(jack) < 3:
        low, high = np.percentile(boot, [2.5, 97.5])
        return float(low), float(high)
    z0 = norm.ppf(below)
    d = jack.mean() - jack
    spread = (d ** 2).sum()
    accel = (d ** 3).sum() / (6 * spread ** 1.5) if spread > 0 else 0.0
    z = norm.ppf([0.025, 0.975])
    low, high = np.percentile(boot, 100 * norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z))))
    return float(low), float(high)


def mean_ci(xs: list[float], clusters: list | None = None, method: str = "bca") -> tuple[float, float, float]:
    """Mean and a 95% bootstrap interval (4,000 resamples, fixed seed): BCa by default, or the plain percentile interval.

    With `clusters`, whole clusters are resampled (one question asked in several layouts is
    one cluster), so the interval does not treat its layouts as independent cases; the BCa
    acceleration then comes from leaving out one cluster at a time.
    """
    import numpy as np

    arr = np.asarray(list(xs), dtype=float)
    if len(arr) < 2:
        return (float(arr[0]) if len(arr) else math.nan), math.nan, math.nan
    rng = np.random.default_rng(0)
    if clusters is None:
        ids = np.arange(len(arr))
    else:
        index = {label: i for i, label in enumerate(dict.fromkeys(clusters))}
        ids = np.array([index[c] for c in clusters])
    groups = int(ids.max()) + 1
    sums, counts = np.bincount(ids, weights=arr, minlength=groups), np.bincount(ids, minlength=groups)
    draws = rng.integers(0, groups, (4000, groups))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    if method == "percentile":
        low, high = np.percentile(means, [2.5, 97.5])
        return float(arr.mean()), float(low), float(high)
    jack = (sums.sum() - sums) / (counts.sum() - counts) if groups > 1 else np.array([arr.mean()])
    return (float(arr.mean()), *bca_bounds(float(arr.mean()), means, jack))


def paired_test(a: list[float], b: list[float], clusters: list | None = None) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired values; with `clusters`, on per-cluster mean differences."""
    from scipy.stats import wilcoxon

    if clusters is None:
        diffs = [x - y for x, y in zip(a, b)]
    else:
        grouped: dict = {}
        for x, y, c in zip(a, b, clusters):
            grouped.setdefault(c, []).append(x - y)
        diffs = [statistics.fmean(v) for v in grouped.values()]
    return 1.0 if all(d == 0 for d in diffs) else float(wilcoxon(diffs).pvalue)


def question(row: dict) -> str:
    """The HotpotQA question a case was built from; several layouts of one question form one cluster."""
    return row["name"].split("/")[1]


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    """Proportion and its 95% Wilson score interval."""
    if n == 0:
        return math.nan, math.nan, math.nan
    z, p = 1.959964, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for one family of comparisons."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def interval(t: tuple[float, float, float]) -> str:
    return f"{kl(t[0])} [{kl(t[1])}, {kl(t[2])}]"


def proportion(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def pvalue(x: float) -> str:
    if x >= 0.1:
        return f"{x:.2f}"
    return f"{x:.3f}" if x >= 0.01 else sci(x)


def kl(x: float) -> str:
    return f"{x:.4f}" if x < 0.1 else f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


TEXT_WIDTH_PT, CELL_PADDING_PT, CELL_CAP_PT, PANDOC_COLUMNS = 469.75, 12.0, 110.0, 72


def ink(text: str) -> float:
    """Rough printed width in points at the tables' \\footnotesize (8pt Latin Modern)."""
    bold = text.startswith("**")
    plain = re.sub(r"\*\*|`|\$|\\", "", text)
    width = sum(2.8 if c in " \u00a0" else 5.9 if c.isupper() else 4.3 if c.isdigit() else 2.4 if c in ".,:;()[]-'" else 4.2 for c in plain)
    return width * (1.1 if bold else 1.0)


def column_widths(header: list[str], rows: list[list[str]]) -> list[float]:
    """Pandoc sizes a wide pipe table's columns only relatively, by the separator's dash counts. Give each column
    room for its longest unbreakable piece (lines break at spaces and after a hyphen before a letter), then share
    what is left of the line by how much more its longest data cell (capped) would like; equal dashes overflow the
    name columns."""
    columns = list(zip(header, *rows))
    pieces = lambda cell: [piece for word in str(cell).split(" ") if word for piece in re.split(r"(?<=-)(?=[^\d\W])", word)]
    least = [max(ink(piece) for cell in column for piece in pieces(cell) or [""]) for column in columns]
    want = [max(low, min(CELL_CAP_PT, max((ink(str(cell)) for cell in column[1:]), default=0.0))) for low, column in zip(least, columns)]
    room = TEXT_WIDTH_PT - CELL_PADDING_PT * len(columns)
    if sum(want) <= room or sum(least) >= room:
        return want if sum(want) <= room else least
    spare = (room - sum(least)) / (sum(want) - sum(least))
    return [low + (high - low) * spare for low, high in zip(least, want)]


def unbreakable(cell: str) -> str:
    """Keep an interval and an "n of m" count on one line in a table cell; a model name is kept whole by its width."""
    cell = re.sub(r"\[([^\]\s]+), ([^\]]+)\]", "[\\1,\u00a0\\2]", cell)
    return re.sub(r"(\d+) of (\d+)", "\\1\u00a0of\u00a0\\2", cell)


def table(name: str, header: list[str], rows: list[list[str]], caption: str) -> None:
    assert all(len(row) == len(header) for row in rows), f"table {name}: a row's length differs from the header's"
    rows = [[unbreakable(str(cell)) for cell in row] for row in rows]
    body = ["| " + " | ".join(row) + " |" for row in [header, *rows]]
    if max(map(len, body)) > PANDOC_COLUMNS:
        separator = "|" + "|".join("-" * max(3, round(w)) for w in column_widths(header, rows)) + "|"
    else:
        separator = "|" + "---|" * len(header)
    lines = [caption, "", body[0], separator, *body[1:]]
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")


def e0() -> None:
    exact = load("e0_exactness")
    if exact:
        rows = exact["rows"]
        f32 = [r for r in rows if r["dtype"] == "float32"]
        f64 = [r for r in rows if r["dtype"] == "float64"]
        values["e0.library_gap"] = sci(max(r["vs_library_logit_gap"] for r in f32))
        values["e0.f64_relocated"] = sci(max(max(r["relocated_query_logit_gap"], r["relocated_rewrite_logit_gap"]) for r in f64))
        values["e0.f32_relocated"] = sci(max(max(r["relocated_query_logit_gap"], r["relocated_rewrite_logit_gap"]) for r in f32))
        values["e0.f32_shift"] = sci(max(r["uniform_shift_20000_logit_gap"] for r in f32))
    drift = load("e0_drift")
    if not drift:
        return
    rows = drift["rows"]
    cell = lambda m, st, e, arm, f: [r[f] for r in rows if (r["model"], r["storage"], r["edits"], r["arm"]) == (m, st, e, arm)]
    out, slopes, at_max = [], [], {}
    for model in NAMES:
        for storage in ("bf16", "fp16"):
            if not cell(model, storage, 1, "rewrite_in_place", "kl_mean"):
                continue
            xs, ys = [], []
            for e in sorted({r["edits"] for r in rows}):
                err = cell(model, storage, e, "rewrite_in_place", "key_rel_err")
                xs.append(math.log(e)); ys.append(math.log(statistics.fmean(err)))
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            slopes.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs))
            for e in (1, 64, 1024, 4096):
                chains = cell(model, storage, e, "rewrite_in_place", "kl_mean")
                k, low, high = statistics.fmean(chains), min(chains), max(chains)
                q = statistics.fmean(cell(model, storage, e, "rotate_query", "kl_mean"))
                flips = statistics.fmean(cell(model, storage, e, "rewrite_in_place", "flip_rate"))
                out.append([NAMES[model] if e == 1 else "", storage if e == 1 else "", str(e),
                            sci(statistics.fmean(cell(model, storage, e, "rewrite_in_place", "key_rel_err"))),
                            f"{sci(k)} [{sci(low)}, {sci(high)}]", pct(flips), sci(q)])
                if e == 4096:
                    at_max[(model, storage)] = (k, flips, q)
    table("drift", ["Model", "Storage", "Rewrites", "Key error (relative)", "KL, rewrite keys", "Next-token changes, rewrite", "KL, rotate query"],
          out, "**Table {{#A:drift}}.** Repeated relocation of a 480-token cache stored in 16-bit floats. Each cell averages 8 chains of random shifts (4 WikiText passages, 2 seeds each); KL and next-token changes are measured at the 32 positions that follow the cache in each chain, 256 positions per cell. KL in nats against the same cache never moved; key error is the norm of the difference from the exactly rotated keys relative to the norm of the keys; brackets give the smallest and largest of the 8 chains (too few for a bootstrap interval).")
    values["e0.slope_min"], values["e0.slope_max"] = f"{min(slopes):.2f}", f"{max(slopes):.2f}"
    values["e0.max_edits"] = "4,096"
    bf = [v for (m, st), v in at_max.items() if st == "bf16"]
    values["e0.kl_bf16_min"], values["e0.kl_bf16_max"] = sci(min(v[0] for v in bf)), sci(max(v[0] for v in bf))
    values["e0.flip_bf16_min"], values["e0.flip_bf16_max"] = pct(min(v[1] for v in bf)), pct(max(v[1] for v in bf))
    query_cells = {(r["model"], r["storage"], r["edits"]) for r in rows if r["arm"] == "rotate_query"}
    values["e0.kl_query_max"] = sci(max(statistics.fmean(cell(*c, "rotate_query", "kl_mean")) for c in query_cells))


GROUPS = [(("hotpot", "distractor", True), "Distractor; fresh question"),
          (("hotpot", "gold", True), "Supporting document; fresh question"),
          (("hotpot", "gold", False), "Supporting document; cached question"),
          (("wikitext", "paragraph", None), "Paragraph; continuation")]


def e1() -> None:
    out, hole_tests, hole_means = [], {}, {}
    for model in NAMES:
        data = load(f"e1_anatomy_{KEYS[model]}_main")
        if not data:
            continue
        rows = data["rows"]
        for (key, label) in GROUPS:
            chosen = [r for r in rows if (r["workload"], r["kind"], r.get("question_is_fresh")) == key]
            if not chosen:
                continue
            get = lambda name: [r["output"][name]["kl_last"] for r in chosen]
            same = lambda state: (statistics.fmean([float(normalise(r["answers"][state]) == normalise(r["answers"]["deleted_truth"])) for r in chosen])
                                  if chosen[0].get("answers") else math.nan)
            closer = statistics.fmean([float(r["output"]["delete/relocated"]["kl_last"] < r["output"]["delete/hole_stale"]["kl_last"]) for r in chosen])
            fmt = interval
            out.append([NAMES[model], f"{label} ({len(chosen)})", fmt(mean_ci(get("delete/significance"))), fmt(mean_ci(get("delete/relocated"))),
                        fmt(mean_ci(get("delete/hole_stale"))), fmt(mean_ci(get("delete/hole_exact"))),
                        "–" if math.isnan(same("deleted_relocated")) else pct(same("deleted_relocated")),
                        "–" if math.isnan(same("deleted_hole_exact")) else pct(same("deleted_hole_exact"))])
            tag = f"e1.{KEYS[model]}.{key[1]}.{'fresh' if key[2] else 'cached' if key[2] is False else 'text'}"
            values[f"{tag}.effect"] = kl(statistics.fmean(get("delete/significance")))
            values[f"{tag}.relocated"] = kl(statistics.fmean(get("delete/relocated")))
            values[f"{tag}.hole_exact"] = kl(statistics.fmean(get("delete/hole_exact")))
            values[f"{tag}.closer"] = pct(closer)
            values[f"{tag}.hole_stale"] = kl(statistics.fmean(get("delete/hole_stale")))
            values[f"{tag}.dev"] = f"{statistics.fmean(r['mean_deviation']['total'] for r in chosen):.4f}"
            from scipy.stats import wilcoxon
            a, b = get("delete/hole_exact"), get("delete/relocated")
            values[f"{tag}.p_hole_vs_relocated"] = pvalue(wilcoxon(a, b).pvalue)
            values[f"{tag}.hole_worse_share"] = pct(sum(x > y for x, y in zip(a, b)) / len(a))
            if key[2] is not False:
                hole_tests[tag] = (wilcoxon(a, b).pvalue, statistics.median(x - y for x, y in zip(a, b)))
                hole_means[tag] = statistics.fmean(a) > statistics.fmean(b)
            if chosen[0].get("answers"):
                values[f"{tag}.same_relocated_k"] = str(sum(normalise(r["answers"]["deleted_relocated"]) == normalise(r["answers"]["deleted_truth"]) for r in chosen))
                values[f"{tag}.n"] = str(len(chosen))
                values[f"{tag}.same_relocated"] = pct(same("deleted_relocated"))
                values[f"{tag}.same_hole_exact"] = pct(same("deleted_hole_exact"))
                f1 = lambda state: statistics.fmean([answer_f1(r["answers"][state], r["answer"]) for r in chosen])
                values[f"{tag}.f1_kept"] = f"{f1('kept_e'):.3f}"
                values[f"{tag}.f1_truth"] = f"{f1('deleted_truth'):.3f}"
                values[f"{tag}.f1_relocated"] = f"{f1('deleted_relocated'):.3f}"
    if hole_tests:
        adjusted = holm({tag: p for tag, (p, _) in hole_tests.items()})
        worse = [tag for tag, (p, median) in hole_tests.items() if median > 0 and p < 0.05]
        values["e1.hole.groups"] = str(len(hole_tests))
        values["e1.hole.worse"] = str(len(worse))
        values["e1.hole.worse_mean"] = str(sum(hole_means.values()))
        values["e1.hole.worse_holm"] = str(sum(adjusted[tag] < 0.05 for tag in worse))
        values["e1.hole.p_max_worse"] = pvalue(max(hole_tests[tag][0] for tag in worse)) if worse else "–"
        for tag in hole_tests:
            values[f"{tag}.p_hole_vs_relocated_holm"] = pvalue(adjusted[tag])
    rhos, shares = [], {k: [] for k in ("oracle", "exposure_mean", "boundary_window", "tail_window", "lexical_overlap")}
    near, far, rhos_total = [], [], []
    for model in NAMES:
        data = load(f"e1_anatomy_{KEYS[model]}_main")
        if not data:
            continue
        for r in data["rows"]:
            rhos.append(r["ranking"]["readset"]["exposure_mean"]["spearman"])
            rhos_total.append(r["ranking"]["total"]["exposure_mean"]["spearman"])
            for k in shares:
                shares[k].append(r["ranking"]["total"][k]["energy@0.1"])
            if "0-8" in r["by_distance"] and "128-512" in r["by_distance"]:
                near.append(r["by_distance"]["0-8"]); far.append(r["by_distance"]["128-512"])
    if rhos:
        values["e1.cases"] = str(len(rhos))
        values["e1.rho_mean"] = f"{statistics.fmean(rhos):.2f}"
        values["e1.rho_p10"] = f"{sorted(rhos)[len(rhos) // 10]:.2f}"
        values["e1.rho_total_mean"] = f"{statistics.fmean(rhos_total):.2f}"
        for k, xs in shares.items():
            values[f"e1.share10.{k}"] = pct(statistics.fmean(xs))
        values["e1.near.geometry_over_total"] = f"{statistics.fmean(d['geometry'] for d in near) / statistics.fmean(d['total'] for d in near):.1f}"
        values["e1.far.readset_over_geometry"] = f"{statistics.fmean(d['readset'] for d in far) / statistics.fmean(d['geometry'] for d in far):.1f}"
    for model in NAMES:
        data = load(f"e1_anatomy_{KEYS[model]}_main")
        if not data:
            continue
        rows = data["rows"]
        counts = [sum((r["workload"], r["kind"], r.get("question_is_fresh")) == key for r in rows) for key, _ in GROUPS]
        values[f"e1.{KEYS[model]}.counts"] = ", ".join(str(c) for c in counts if c)
        values[f"e1.{KEYS[model]}.groups"] = str(sum(1 for c in counts if c))
        hotpot = [r for r in rows if r["workload"] == "hotpot"]
        group_means = lambda field: [statistics.fmean(r[field] for r in hotpot if (r["kind"], r.get("question_is_fresh")) == (k[1], k[2]))
                                     for k, _ in GROUPS[:3] if any((r["kind"], r.get("question_is_fresh")) == (k[1], k[2]) for r in hotpot)]
        values[f"e1.{KEYS[model]}.prefix"] = f"{statistics.fmean(r['nP'] for r in hotpot):.0f}"
        values[f"e1.{KEYS[model]}.edit_min"], values[f"e1.{KEYS[model]}.edit_max"] = f"{min(group_means('nE')):.0f}", f"{max(group_means('nE')):.0f}"
        values[f"e1.{KEYS[model]}.suffix_min"], values[f"e1.{KEYS[model]}.suffix_max"] = f"{min(group_means('nS')):.0f}", f"{max(group_means('nS')):.0f}"
    closers = [float(v.rstrip("%")) for k, v in values.items() if k.startswith("e1.") and k.endswith(".closer")]
    if closers:
        values["e1.closer_min"], values["e1.closer_max"] = f"{min(closers):.0f}%", f"{max(closers):.0f}%"
    table("anatomy", ["Model", "Deleted; reader (cases)", "Effect of the deletion", "Relocated, no repair", "Gapped, stale cells",
                      "Gapped, suffix recomputed", "Answers unchanged, relocated", "Answers unchanged, gapped recomputed"],
          out, "**Table {{#B:anatomy}}.** Reusing the suffix after an interior deletion, without repair. A fresh question is asked after the edit, as new input; a cached question was asked before it and is part of the stale suffix; for WikiText the reader is the continuation of the article. KL in nats from the cold pass over the edited context, at the first answer token (last position of the continuation for WikiText); brackets give a 95% bootstrap interval over cases. The first KL column is the effect of the deletion itself (kept versus deleted, both exact).")


POLICIES = [("boundary", "Boundary window", "KVEraser"), ("tail", "Tail window", "KVEraser"), ("random", "Random", ""),
            ("lexical", "Lexical overlap with the edit", ""), ("probe1", "First-layer deviation", "CacheBlend"),
            ("rollout", "Exposure rolled forward", ""), ("exposure", "Recorded exposure", "cf. Cache-Craft"),
            ("reader", "Reader attention", "ProphetKV"), ("oracle", "True cell deviation (needs the re-prefill)", ""),
            ("bridge", "**Bridge score**", "this work")]

POLICIES_BY_KEY = [(key, label.strip("*")) for key, label, _ in POLICIES]


def e2() -> None:
    from scipy.stats import wilcoxon

    pooled, by_place, place_best = [], [], []
    main_rows = {key: [label, source] for key, label, source in POLICIES}
    main_rows["stale"] = ["No repair (Leyline's positional splice)", ""]
    answers = {key: [label] for key, label in (("stale", "No repair"), ("boundary", "Boundary window"), ("tail", "Tail window"),
                                               ("reader", "Reader attention"), ("bridge", "**Bridge score**"))}
    models_done = []
    for model in NAMES:
        data = load(f"e2_healing_{KEYS[model]}_main")
        if not data:
            continue
        rows = [r for r in data["rows"] if r["kind"] == "gold"]
        counts = {place: sum(r["place"] == place for r in rows) for place in ("adjacent", "middle", "end")}
        if min(counts.values()) == 0 or len(set(counts.values())) > 1:
            values[f"e2.{KEYS[model]}.status"] = f"incomplete {counts}"
            continue
        tag = f"e2.{KEYS[model]}"
        cids = [question(r) for r in rows]
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.questions"] = str(len(set(cids)))
        values[f"{tag}.per_place"] = str(counts["adjacent"])
        values[f"{tag}.suffix"] = f"{statistics.fmean(r['delete']['suffix_tokens'] for r in rows):,.0f}"
        summary = {}
        for direction in ("delete", "insert"):
            chosen = [r[direction] for r in rows]
            unchanged = lambda key: statistics.fmean([float(normalise(c["answers"][key]) == normalise(c["answers"]["truth"])) for c in chosen])
            values[f"{tag}.{direction}.stale"] = kl(statistics.fmean([c["results"]["stale"]["0"]["kl_last"] for c in chosen]))
            values[f"{tag}.{direction}.same_stale"] = pct(unchanged("stale"))
            for key, _, _ in POLICIES:
                k10 = mean_ci([c["results"][key]["0.1"]["kl_last"] for c in chosen], cids)
                k20 = mean_ci([c["results"][key]["0.2"]["kl_last"] for c in chosen], cids)
                same = pct(unchanged(key)) if key in chosen[0]["answers"] else ""
                summary[(direction, key)] = (k10, k20, same)
                values[f"{tag}.{direction}.{key}.10"] = kl(k10[0])
                values[f"{tag}.{direction}.{key}.20"] = kl(k20[0])
                if same:
                    values[f"{tag}.{direction}.{key}.same10"] = same
                if key != "bridge":
                    a = [c["results"]["bridge"]["0.1"]["kl_last"] for c in chosen]
                    b = [c["results"][key]["0.1"]["kl_last"] for c in chosen]
                    values[f"{tag}.{direction}.win.{key}"] = pct(sum(x < y for x, y in zip(a, b)) / len(a))
                    values[f"{tag}.{direction}.p.{key}"] = pvalue(paired_test(a, b, cids))
        inserted = [r["insert"] for r in rows]
        if "bridge_l1" in inserted[0]["results"]:
            a = [c["results"]["bridge_l1"]["0.1"]["kl_last"] for c in inserted]
            b = [c["results"]["reader"]["0.1"]["kl_last"] for c in inserted]
            values[f"{tag}.insert.bridge_l1.10"] = kl(statistics.fmean(a))
            values[f"{tag}.insert.l1_vs_reader.win"] = pct(sum(x < y for x, y in zip(a, b)) / len(a))
            values[f"{tag}.insert.l1_vs_reader.p"] = pvalue(paired_test(a, b, cids))
        models_done.append(NAMES[model])
        distractors = [r["delete"] for r in data["rows"] if r["kind"] == "distractor"]
        if distractors:
            values[f"{tag}.distractor.n"] = str(len(distractors))
            for key in ("stale", "boundary", "tail", "reader", "exposure", "bridge"):
                budget = "0" if key == "stale" else "0.1"
                values[f"{tag}.distractor.{key}.10"] = kl(statistics.fmean(c["results"][key][budget]["kl_last"] for c in distractors))
        deleted = [r["delete"] for r in rows]
        stale_ci = mean_ci([c["results"]["stale"]["0"]["kl_last"] for c in deleted], cids)
        main_rows["stale"].append(interval(stale_ci))
        inserted_stale = mean_ci([c["results"]["stale"]["0"]["kl_last"] for c in inserted], cids)
        pooled.append([NAMES[model], "No repair (Leyline's positional splice)", "", interval(stale_ci), kl(stale_ci[0]), values[f"{tag}.delete.same_stale"]])
        for key, _, _ in POLICIES:
            d10 = summary[("delete", key)][0]
            main_rows[key].append(interval(d10))
        fill = load(f"e2_fill_{KEYS[model]}_main")
        if fill and f"{tag}.delete.reader.same10" not in values:
            values[f"{tag}.delete.reader.same10"] = pct(statistics.fmean(float(normalise(r["answer"]) == normalise(r["truth"])) for r in fill["rows"]))
            gap = max(abs(r['kl_last'] - r['stored_kl_last']) for r in fill['rows'])
            values[f"{tag}.fill.max_difference"] = "0" if gap == 0 else f"{gap:.1e}"
        for key in answers:
            answers[key].append(values[f"{tag}.delete.same_stale"] if key == "stale" else values.get(f"{tag}.delete.{key}.same10", "–"))
        for index, (key, label, source) in enumerate(POLICIES):
            (d10, d20, same), (i10, _, _) = summary[("delete", key)], summary[("insert", key)]
            pooled.append(["", label, source, interval(d10), kl(d20[0]), same])
        shown = ("boundary", "tail", "probe1", "exposure", "reader", "oracle", "bridge")
        for place, label in (("adjacent", "right after the edit"), ("middle", "in the middle"), ("end", "at the end")):
            part = [r["delete"] for r in rows if r["place"] == place]
            mean = lambda key, budget: statistics.fmean([c["results"][key][budget]["kl_last"] for c in part])
            by_place.append([NAMES[model] if place == "adjacent" else "", label, kl(mean("stale", "0"))] + [kl(mean(k, "0.1")) for k in shown])
            for k in shown:
                values[f"{tag}.place.{place}.{k}"] = kl(mean(k, "0.1"))
            practical = {k: mean(k, "0.1") for k, _, _ in POLICIES if k != "oracle"}
            place_best.append(min(practical, key=practical.get))
            values[f"{tag}.place.{place}.stale"] = kl(mean("stale", "0"))
    if place_best:
        values["e2.place.cells"] = str(len(place_best))
        values["e2.place.bridge_best"] = str(place_best.count("bridge"))
        others = sorted({k for k in place_best if k != "bridge"})
        values["e2.place.others_best"] = ", ".join(dict(POLICIES_BY_KEY)[k].lower() for k in others) if others else "none"
    paired_rows = []
    for model in NAMES:
        data = load(f"e2_healing_{KEYS[model]}_main")
        if not data or f"e2.{KEYS[model]}.status" in values:
            continue
        gold = [r for r in data["rows"] if r["kind"] == "gold"]
        rows, cids = [r["delete"] for r in gold], [question(r) for r in gold]
        a = [c["results"]["bridge"]["0.1"]["kl_last"] for c in rows]
        raw = {key: paired_test(a, [c["results"][key]["0.1"]["kl_last"] for c in rows], cids) for key, _, _ in POLICIES if key != "bridge"}
        adjusted = holm(raw)
        for key, label, _ in POLICIES:
            if key == "bridge":
                continue
            b = [c["results"][key]["0.1"]["kl_last"] for c in rows]
            share = sum(x < y for x, y in zip(a, b)) / len(a)
            values[f"e2.{KEYS[model]}.delete.holm.{key}"] = pvalue(adjusted[key])
            paired_rows.append([NAMES[model] if key == POLICIES[0][0] else "", label, pct(share), pvalue(raw[key]), pvalue(adjusted[key])])
    questions = ", ".join(values[f"e2.{KEYS[m]}.questions"] for m in NAMES if f"e2.{KEYS[m]}.questions" in values)
    table("repair_paired", ["Model", "Bridge compared with", "Cases where bridge has lower KL", "Wilcoxon p", "Holm-adjusted p"], paired_rows,
          f"**Table {{{{#H:repair_paired}}}}.** Paired comparison of the bridge score with every other policy, deleting a supporting document, 10% budget. The share counts cases; the Wilcoxon signed-rank test is on per-question mean differences ({questions} questions), Holm-corrected within each model over the nine comparisons.")
    order = ["stale"] + [key for key, _, _ in POLICIES]
    counts = ", ".join(values[f"e2.{KEYS[m]}.n"] for m in NAMES if f"e2.{KEYS[m]}.n" in values)
    per_place = ", ".join(values[f"e2.{KEYS[m]}.per_place"] for m in NAMES if f"e2.{KEYS[m]}.per_place" in values)
    table("repair_pooled", ["Policy", "Signal as used by"] + models_done, [main_rows[k] for k in order],
          f"**Table {{{{#H:repair_pooled}}}}.** Repair after deleting a supporting document from the middle of the context, three reader placements pooled ({counts} cases; each question appears at all three placements). A policy ranks the suffix tokens and the top 10% are recomputed. KL in nats from the re-prefill at the first answer token, mean and 95% bootstrap interval over questions.")
    table("repair_answers", ["Policy"] + models_done, list(answers.values()),
          "**Table {{#H:repair_answers}}.** Share of 16-token greedy answers identical to the re-prefill answer after deleting a supporting document, 10% of the suffix recomputed. Answers were decoded for these policies only; – means the policy was not decoded in that model's run.")
    table("repair_full", ["Model", "Policy", "Signal as used by", "KL, 10% (delete)", "KL, 20% (delete)", "Answers unchanged, 10% (delete)"],
          pooled, f"**Table {{{{#H:repair_full}}}}.** All policies at the 10% and 20% budgets after deleting a supporting document ({counts} cases). KL in nats from the re-prefill at the first answer token; brackets give a 95% bootstrap interval over questions. Inserts are in Table {{{{#E:insert_sketch}}}}.")
    table("repair_by_place", ["Model", "Cached question sits", "No repair", "Boundary", "Tail", "First-layer dev.", "Exposure", "Reader", "Oracle", "**Bridge**"],
          by_place, f"**Table {{{{#H:repair_by_place}}}}.** Delete, 10% budget, by where the stale reader (the cached question) sits in the suffix ({per_place} cases per placement). KL in nats, mean over cases.")


def e3_reader() -> None:
    """Reader-first policies on the needle task (run with tag "reader"); same cases as the main run."""
    for model in NAMES:
        data = load(f"e3_needle_{KEYS[model]}_reader")
        main = load(f"e3_needle_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e3r.{KEYS[model]}"
        for size in sorted({r["size"] for r in rows}):
            chosen = [r for r in rows if r["size"] == size]
            values[f"{tag}.{size}.n"] = str(len(chosen))
            for label in chosen[0]["out"]:
                values[f"{tag}.{size}.{label}.exact"] = f"{statistics.fmean(float(r['out'][label]['exact']) for r in chosen):.2f}"
                values[f"{tag}.{size}.{label}.k"] = f"{sum(bool(r['out'][label]['exact']) for r in chosen)} of {len(chosen)}"
                if "kl" in chosen[0]["out"][label]:
                    values[f"{tag}.{size}.{label}.kl"] = kl(statistics.fmean(r["out"][label]["kl"] for r in chosen))
            picked = {}
            for r in chosen:
                picked[r["picked_segment"]] = picked.get(r["picked_segment"], 0) + 1
            values[f"{tag}.{size}.picked_question"] = f"{picked.get('question', 0)} of {len(chosen)}"
            values[f"{tag}.{size}.picked_retained"] = f"{picked.get('retained', 0)} of {len(chosen)}"
        if main:
            reference = {r["name"]: r for r in main["rows"]}
            same = [abs(r["out"]["bridge@0.02"]["kl"] - reference[r["name"]]["out"]["bridge@0.02"]["kl"]) for r in rows if r["name"] in reference]
            values[f"{tag}.reproduces_main"] = "yes" if same and max(same) == 0.0 else f"max difference {max(same):.1e}" if same else "no overlap"


NEEDLE_ROWS = [("no_repair", "No repair (relocated)"), ("hole_no_repair", "No repair (gapped)"), ("boundary@0.15", "Boundary window, 15%"),
               ("tail@0.15", "Tail window, 15%"), ("probe1@0.15", "First-layer deviation, 15%"), ("reader@0.05", "Reader attention, 5%"),
               ("exposure@0.05", "Recorded exposure, 5%"), ("oracle@0.05", "True cell deviation (needs the re-prefill), 5%"),
               ("bridge@0.02", "**Bridge, 2%**"), ("bridge@0.05", "**Bridge, 5%**"), ("full_recompute", "Re-prefill of the suffix")]


def e3() -> None:
    for model in NAMES:
        data = load(f"e3_needle_{KEYS[model]}_main")
        if not data:
            continue
        rows, tag = data["rows"], f"e3.{KEYS[model]}"
        sizes = sorted({r["size"] for r in rows})
        out = []
        for key, label in NEEDLE_ROWS:
            line = [label]
            for size in sizes:
                chosen = [r for r in rows if r["size"] == size]
                exact = wilson(sum(bool(r["out"][key]["exact"]) for r in chosen), len(chosen))
                line += [proportion(exact), kl(statistics.fmean(r["out"][key]["kl"] for r in chosen))]
                values[f"{tag}.{size}.{key}.exact"] = f"{exact[0]:.2f}"
                values[f"{tag}.{size}.{key}.k"] = f"{sum(bool(r['out'][key]['exact']) for r in chosen)} of {len(chosen)}"
                values[f"{tag}.{size}.{key}.ghost"] = pct(statistics.fmean(float(r["out"][key]["ghost"]) for r in chosen))
            out.append(line)
        for size in sizes:
            chosen = [r for r in rows if r["size"] == size]
            for condition in chosen[0]["out"]:
                values[f"{tag}.{size}.{condition}.exact"] = f"{statistics.fmean(float(r['out'][condition]['exact']) for r in chosen):.2f}"
            share = lambda field, part: statistics.fmean(r[field][part] for r in chosen)
            values[f"{tag}.{size}.n"] = str(len(chosen))
            values[f"{tag}.{size}.suffix"] = f"{statistics.fmean(r['nS'] for r in chosen):.0f}"
            values[f"{tag}.{size}.both_before"] = pct(statistics.fmean(float(r["out"]["both_present"]["ghost"] and r["out"]["both_present"]["has_retained"]) for r in chosen))
            values[f"{tag}.{size}.needle_tokens"] = f"{100 * share('token_share', 'retained_needle'):.1f}%"
            values[f"{tag}.{size}.needle_deviation"] = pct(share("deviation_share", "retained_needle"))
            values[f"{tag}.{size}.needle_exposure"] = pct(share("exposure_share", "retained_needle"))
            values[f"{tag}.{size}.first5_deviation"] = pct(share("deviation_share", "first_5_percent"))
            for label in ("boundary@0.15", "tail@0.15", "reader@0.05", "exposure@0.05", "bridge@0.05"):
                values[f"{tag}.{size}.picks.{label}"] = pct(statistics.fmean(r["picks"][label]["on_retained_needle"] for r in chosen))
                values[f"{tag}.{size}.picks_q.{label}"] = pct(statistics.fmean(r["picks"][label]["on_question"] for r in chosen))
        fused_run = load(f"e3_needle_{KEYS[model]}_rrf")
        fused = {r["name"]: r for r in fused_run["rows"]} if fused_run else {}
        fused_note = ""
        if fused and all(r["name"] in fused for r in rows):
            shared = [(fused[r["name"]]["out"][k], r["out"][k]) for r in rows for k in fused[r["name"]]["out"] if k in r["out"]]
            values[f"{tag}.rrf.reproduces"] = "exactly" if all(a == b for a, b in shared) else "not exactly"
            fused_note = f" The rank-fusion rows come from a second run over the same cases, which reproduces the rows the two runs share {values[f'{tag}.rrf.reproduces']}."
            for budget in ("0.02", "0.05"):
                line = [f"Rank fusion, {float(budget):.0%}"]
                for size in sizes:
                    chosen = [fused[r["name"]]["out"][f"rrf@{budget}"] for r in rows if r["size"] == size]
                    line += [proportion(wilson(sum(bool(c["exact"]) for c in chosen), len(chosen))), kl(statistics.fmean(c["kl"] for c in chosen))]
                out.insert(len(out) - 1, line)
        matched = []
        for key, label in (("boundary", "Boundary window"), ("tail", "Tail window"), ("probe1", "First-layer deviation"),
                           ("reader", "Reader attention"), ("exposure", "Recorded exposure"), ("oracle", "True cell deviation (needs the re-prefill)"),
                           ("bridge", "**Bridge score**")):
            line = [label]
            for size in sizes:
                chosen = [r for r in rows if r["size"] == size]
                for budget in ("0.02", "0.05", "0.15"):
                    field = f"{key}@{budget}"
                    line.append(f"{statistics.fmean(float(r['out'][field]['exact']) for r in chosen):.2f}" if field in chosen[0]["out"] else "–")
            matched.append(line)
        if fused_note:
            line = ["Rank fusion"]
            for size in sizes:
                chosen = [fused[r["name"]]["out"] for r in rows if r["size"] == size]
                line += [f"{statistics.fmean(float(c[f'rrf@{b}']['exact']) for c in chosen):.2f}" for b in ("0.02", "0.05")] + ["–"]
            matched.append(line)
        none_line = ["No repair"]
        for size in sizes:
            chosen = [r for r in rows if r["size"] == size]
            none_line += [f"{statistics.fmean(float(r['out']['no_repair']['exact']) for r in chosen):.2f}", "", ""]
        matched.insert(0, none_line)
        table(f"needle_matched_{KEYS[model]}", ["Policy"] + [f"{b}, {size} tokens" for size in sizes for b in ("2%", "5%", "15%")], matched,
              f"**Table {{{{#E:needle_matched}}}}.** Needle erasure on {NAMES[model]}: exact match by share of the suffix recomputed, at matched budgets (– = not run). Same cases as Table {{{{#E:needle}}}}.{fused_note}")
        header = ["Condition"] + [h for size in sizes for h in (f"Exact match, {size} tokens (n={values[f'{tag}.{size}.n']})", "KL")]
        table(f"needle_{KEYS[model]}", header, out,
              f"**Table {{{{#E:needle}}}}.** Needle erasure on {NAMES[model]}, after the protocol of KVEraser: two needles share a key, the earlier one is erased, the cached question asks for the key's number(s). Exact match means the reply contains the retained number and no other number. Percentages after a policy are the share of suffix tokens recomputed. Brackets give a 95% Wilson interval; KL at the first reply token from the re-prefill.{fused_note}")


SKETCH_ROWS = [("reader", "Reader attention (ProphetKV's signal)"), ("bridge_l1", "First-layer exposure × reader"),
               ("bridge_qs4g", "**Kept queries, 4 layers, one per KV group × reader**"), ("bridge_qs4", "Kept queries, 4 layers, all heads × reader"),
               ("bridge_qs", "Kept queries, all layers, all heads × reader"), ("bridge", "True exposure × reader (not available in practice)")]
LAYERS = {"qwen0.5b": 24, "qwen1.5b": 28, "smol1.7b": 24}


def e2b() -> None:
    from scipy.stats import wilcoxon

    rows_out, models_done = {k: [label] for k, label in SKETCH_ROWS}, []
    for model in NAMES:
        data = load(f"e2b_insert_sketch_{KEYS[model]}_main")
        if not data:
            continue
        rows, tag = data["rows"], f"e2b.{KEYS[model]}"
        counts = {p: sum(r["place"] == p for r in rows) for p in ("adjacent", "middle", "end")}
        if min(counts.values()) == 0 or len(set(counts.values())) > 1:
            values[f"{tag}.status"] = f"incomplete {counts}"
            continue
        models_done.append(NAMES[model])
        chosen, cids = [r["insert"] for r in rows], [question(r) for r in rows]
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.questions"] = str(len(set(cids)))
        values[f"{tag}.storage"] = pct(4 / LAYERS[KEYS[model]])
        for name in ("qs", "qs4", "qs4g", "l1"):
            values[f"{tag}.rho.{name}"] = f"{statistics.fmean(r['rank_agreement_with_true_exposure'][name] for r in rows):.2f}"
        values[f"{tag}.stale"] = kl(statistics.fmean(c["results"]["stale"]["0"]["kl_last"] for c in chosen))
        for key, _ in SKETCH_ROWS:
            k10 = mean_ci([c["results"][key]["0.1"]["kl_last"] for c in chosen], cids)
            rows_out[key].append(interval(k10))
            values[f"{tag}.{key}.10"] = kl(k10[0])
            if key in chosen[0]["answers"]:
                values[f"{tag}.{key}.same10"] = pct(statistics.fmean(float(normalise(c["answers"][key]) == normalise(c["answers"]["truth"])) for c in chosen))
        for other in ("reader", "bridge_l1", "bridge"):
            a = [c["results"]["bridge_qs4g"]["0.1"]["kl_last"] for c in chosen]
            b = [c["results"][other]["0.1"]["kl_last"] for c in chosen]
            values[f"{tag}.qs4g_vs.{other}.win"] = pct(sum(x < y for x, y in zip(a, b)) / len(a))
            values[f"{tag}.qs4g_vs.{other}.p"] = pvalue(paired_test(a, b, cids))
        a = [c["results"]["bridge_l1"]["0.1"]["kl_last"] for c in chosen]
        b = [c["results"]["reader"]["0.1"]["kl_last"] for c in chosen]
        values[f"{tag}.l1_vs_reader.win"] = pct(sum(x < y for x, y in zip(a, b)) / len(a))
        values[f"{tag}.l1_vs_reader.p"] = pvalue(paired_test(a, b, cids))
        mean10 = lambda key: statistics.fmean(c["results"][key]["0.1"]["kl_last"] for c in chosen)
        values[f"{tag}.qs4g.gap_closed"] = pct((mean10("reader") - mean10("bridge_qs4g")) / (mean10("reader") - mean10("bridge")))
    if models_done:
        counts = ", ".join(values[f"e2b.{KEYS[m]}.n"] for m in NAMES if f"e2b.{KEYS[m]}.n" in values)
        table("insert_sketch", ["Exposure estimate"] + models_done, list(rows_out.values()),
              f"**Table {{{{#E:insert_sketch}}}}.** Repair after a supporting document arrives late and is inserted before a cached suffix that never saw it ({counts} cases, the supporting-document cases of Table {{{{#H:repair_full}}}}). Below the first row, each policy multiplies an estimate of the attention each suffix token would have paid to the new document by reader attention; the top 10% of suffix tokens are recomputed. KL in nats from the re-prefill at the first answer token, mean and 95% bootstrap interval over questions.")


STREAM_ROWS = [("append", "Encode on arrival, in arrival order"), ("speculate", "Encode on arrival, at the canonical place, no repair"),
               ("repair_flat@0.1", "  + repair 10%, flat prior × reader"), ("repair_sketch@0.1", "  + repair 10%, kept-query estimate × reader"),
               ("repair_flat@0.2", "  + repair 20%, flat prior × reader"), ("repair_sketch@0.2", "  + repair 20%, kept-query estimate × reader"),
               ("sweep", "  + sweep to exact")]
MERGE_ROWS = [("merge", "Merge, no repair"), ("repair_flat@0.1", "  + repair 10%, flat prior × reader"),
              ("repair_sketch@0.1", "  + repair 10%, kept-query estimate × reader"), ("repair_flat@0.2", "  + repair 20%, flat prior × reader"),
              ("repair_sketch@0.2", "  + repair 20%, kept-query estimate × reader")]


def e5() -> None:
    for model in NAMES:
        data = load(f"e5_streams_{KEYS[model]}_main")
        if not data:
            continue
        tag = f"e5.{KEYS[model]}"
        arrival, merge = data["arrival"], data["merge"]
        out = []
        for part, rows, spec in (("arrival", arrival, STREAM_ROWS), ("merge", merge, MERGE_ROWS)):
            values[f"{tag}.{part}.n"] = str(len(rows))
            for key, label in spec:
                if key not in rows[0]:
                    continue
                k = mean_ci([r[key]["kl_last"] for r in rows])
                same = statistics.fmean(float(normalise(r[key]["answer"]) == normalise(r["truth_answer"])) for r in rows)
                tokens = statistics.fmean(r[key]["tokens"] for r in rows) if "tokens" in rows[0][key] else None
                values[f"{tag}.{part}.{key}.kl"] = kl(k[0])
                values[f"{tag}.{part}.{key}.same"] = pct(same)
                if tokens is not None:
                    values[f"{tag}.{part}.{key}.tokens"] = f"{tokens:.0f}"
                out.append([("Out-of-order arrival" if part == "arrival" else "Branch merge") if key == spec[0][0] else "", label,
                            interval(k), pct(same), "" if tokens is None else f"{tokens:.0f}"])
        mean = lambda field: statistics.fmean(r[field] for r in arrival)
        values[f"{tag}.arrival.doc_tokens"] = f"{mean('doc_tokens'):.0f}"
        values[f"{tag}.arrival.last_doc_tokens"] = f"{mean('last_doc_tokens'):.0f}"
        values[f"{tag}.arrival.stale_tokens"] = f"{mean('stale_tokens'):.0f}"
        values[f"{tag}.arrival.inversions"] = f"{mean('inversions'):.1f}"
        values[f"{tag}.merge.tokens"] = f"{statistics.fmean(r['tokens'] for r in merge):.0f}"
        values[f"{tag}.merge.stale_tokens"] = f"{statistics.fmean(r['stale_tokens'] for r in merge):.0f}"
        table(f"streams_{KEYS[model]}", ["Workload", "Strategy", "KL", "Answers unchanged", "Tokens recomputed"], out,
              f"**Table {{{{#G:streams}}}}.** The runtime on {NAMES[model]}. Out-of-order arrival: a question is followed by six documents that arrive in random order; the reference is the cold pass over the canonical order. Merge: two branches read different documents and each write a one-sentence note; the reference is the cold pass over the merged sequence. KL in nats at the first answer token, mean and 95% bootstrap interval; answers are 16-token greedy replies.")


E2C_DELETE = [("stale", "No repair", ""), ("boundary", "Boundary window", "KVEraser"), ("epic", "First tokens of each document", "EPIC"),
              ("tail", "Tail window", "KVEraser"), ("random", "Random", ""), ("probe1", "First-layer deviation", "CacheBlend"),
              ("exposure", "Recorded exposure", "cf. Cache-Craft"), ("reader", "Reader attention", "ProphetKV, practical"),
              ("question_exposure", "Cached question (given), then exposure", ""), ("sum", "Rank sum of exposure and reader", ""),
              ("rrf", "Rank fusion (RRF)", ""), ("union", "Alternating the two rankings", ""),
              ("bridge_layer", "Product per layer, summed", ""), ("bridge_below", "Reader × exposure accumulated below", ""),
              ("bridge", "**Bridge score**", "this work"), ("oracle", "True cell deviation (needs the re-prefill)", ""),
              ("oracle_reader", "Oracle score: true deviation × reader", "ProphetKV's ideal value function")]
LEAD_COLUMNS = ("stale", "boundary", "reader", "exposure", "rrf", "bridge", "question_exposure", "oracle_reader")
E2D_EXTRA = [("density", "Whole segments by mean bridge score", ""), ("question_bridge", "Cached question (given), then bridge", ""),
             ("segment_exposure", "**Reader-first (segment found)**", "this work"),
             ("segment_bridge", "**Reader-first (segment found), then bridge**", "this work")]
E2C_ANSWERS = [("stale", "No repair"), ("boundary", "Boundary window"), ("tail", "Tail window"), ("exposure", "Recorded exposure"),
               ("reader", "Reader attention"), ("question_exposure", "Cached question, then exposure"), ("sum", "Rank sum"),
               ("bridge", "**Bridge score**"), ("oracle_reader", "Oracle score: true deviation × reader")]
E2C_REPLACE = [("stale", "No repair"), ("boundary", "Boundary window"), ("tail", "Tail window"), ("exposure", "Recorded exposure"),
               ("reader", "Reader attention"), ("question_exposure", "Cached question, then exposure"), ("bridge", "**Bridge score**"),
               ("oracle_reader", "Oracle score: true deviation × reader")]


def e2c() -> None:
    from scipy.stats import spearmanr

    metric_rows = {metric: {key: [label, source] for key, label, source in E2C_DELETE + E2D_EXTRA} for metric in ("kl_tf", "kl_first")}
    paired_rows, answer_rows, replace_rows, models_done = [], {k: [label] for k, label in E2C_ANSWERS}, {k: [label] for k, label in E2C_REPLACE}, []
    answer_rows["segment_exposure"] = ["**Reader-first (segment found)**"]
    answer_rows["segment_bridge"] = ["**Reader-first (segment found), then bridge**"]
    lead_rows, place_removed, bridge_beats_exposure = [], {}, []
    for model in NAMES:
        data = load(f"e2c_ablation_{KEYS[model]}_main")
        if not data:
            continue
        rows, tag = data["rows"], f"e2c.{KEYS[model]}"
        gold = [r for r in rows if r["kind"] == "gold"]
        cells = {}
        for r in gold:
            cells[(r["lead"], r["place"])] = cells.get((r["lead"], r["place"]), 0) + 1
        if len(cells) < len(data["leads"]) * 3 or len(set(cells.values())) > 1:
            values[f"{tag}.status"] = f"incomplete: {len(gold)} supporting cases in {len(cells)} cells"
            continue
        models_done.append(NAMES[model])
        deleted, cids = [r["delete"] for r in gold], [question(r) for r in gold]
        result = lambda c, key, budget, metric: c["results"][key]["0" if key == "stale" else budget][metric]
        values[f"{tag}.n"] = str(len(gold))
        values[f"{tag}.questions"] = str(len(set(cids)))
        values[f"{tag}.per_cell"] = str(next(iter(cells.values())))
        values[f"{tag}.suffix"] = f"{statistics.fmean(r['nS'] for r in gold):,.0f}"
        for metric in ("kl_tf", "kl_first"):
            for key, _, _ in E2C_DELETE:
                ci = mean_ci([result(c, key, "0.1", metric) for c in deleted], cids)
                metric_rows[metric][key].append(interval(ci))
                values[f"{tag}.{metric}.{key}.10"] = kl(ci[0])
                if key != "stale":
                    for budget in data["budgets"]:
                        values[f"{tag}.{metric}.{key}.{budget}"] = kl(statistics.fmean(result(c, key, str(budget), metric) for c in deleted))
            reader_run = load(f"e2d_reader_first_{KEYS[model]}_main")
            by_name = {r["name"]: r for r in reader_run["rows"]} if reader_run else {}
            for key, _, _ in E2D_EXTRA:
                if all(r["name"] in by_name for r in gold):
                    metric_rows[metric][key].append(interval(mean_ci([by_name[r["name"]]["results"][key]["0.1"][metric] for r in gold], cids)))
                else:
                    metric_rows[metric][key].append("–")
            values[f"{tag}.{metric}.median.bridge.10"] = kl(statistics.median(result(c, "bridge", "0.1", metric) for c in deleted))
            values[f"{tag}.{metric}.median.oracle_reader.10"] = kl(statistics.median(result(c, "oracle_reader", "0.1", metric) for c in deleted))
            question_only = [c["results"]["question_only"]["own"][metric] for c in deleted]
            values[f"{tag}.{metric}.question_only"] = kl(statistics.fmean(question_only))
        values[f"{tag}.question_share"] = pct(statistics.fmean(c["question_tokens"] / r["nS"] for c, r in zip(deleted, gold)))
        values[f"{tag}.top1_tf.bridge.10"] = pct(statistics.fmean(c["results"]["bridge"]["0.1"]["top1_tf"] for c in deleted))
        values[f"{tag}.top1_tf.stale"] = pct(statistics.fmean(c["results"]["stale"]["0"]["top1_tf"] for c in deleted))
        values[f"{tag}.reply_tokens"] = f"{statistics.fmean(c['reply_tokens'] for c in deleted):.1f}"

        a = [result(c, "bridge", "0.1", "kl_tf") for c in deleted]
        raw = {key: paired_test(a, [result(c, key, "0.1", "kl_tf") for c in deleted], cids) for key, _, _ in E2C_DELETE if key not in ("bridge", "stale")}
        adjusted = holm(raw)
        for index, key in enumerate(raw):
            b = [result(c, key, "0.1", "kl_tf") for c in deleted]
            share = sum(x < y for x, y in zip(a, b)) / len(a)
            values[f"{tag}.vs.{key}.win"], values[f"{tag}.vs.{key}.p"], values[f"{tag}.vs.{key}.holm"] = pct(share), pvalue(raw[key]), pvalue(adjusted[key])
            label = dict((k, l) for k, l, _ in E2C_DELETE)[key]
            paired_rows.append([NAMES[model] if index == 0 else "", label, pct(share), pvalue(raw[key]), pvalue(adjusted[key])])
            if key == "exposure":
                bridge_beats_exposure.append(share > 0.5 and adjusted[key] < 0.05)
        qe = [result(c, "question_exposure", "0.1", "kl_tf") for c in deleted]
        for other in ("oracle_reader", "bridge", "rrf"):
            b = [result(c, other, "0.1", "kl_tf") for c in deleted]
            values[f"{tag}.qe_vs.{other}.win"] = pct(sum(x < y for x, y in zip(qe, b)) / len(qe))
            values[f"{tag}.qe_vs.{other}.p"] = pvalue(paired_test(qe, b, cids))
        values[f"{tag}.kl_tf.median.question_exposure.10"] = kl(statistics.median(qe))
        stale_tf = [result(c, "stale", "0", "kl_tf") for c in deleted]
        values[f"{tag}.question_only_removes"] = pct(1 - statistics.fmean(c["results"]["question_only"]["own"]["kl_tf"] for c in deleted) / statistics.fmean(stale_tf))
        for metric in ("kl_tf", "kl_first"):
            ratio = statistics.fmean(result(c, "bridge", "0.1", metric) for c in deleted) / statistics.fmean(result(c, "reader", "0.1", metric) for c in deleted)
            values[f"{tag}.{metric}.bridge_over_reader"] = f"{ratio:.2f}"
        practical = [key for key, _, _ in E2C_DELETE if key not in ("stale", "oracle", "oracle_reader")]
        best = min(practical, key=lambda key: statistics.fmean(result(c, key, "0.1", "kl_tf") for c in deleted))
        values[f"{tag}.best_practical"] = dict((k, l.strip("*")) for k, l, _ in E2C_DELETE)[best]
        values[f"{tag}.bridge_over_ideal"] = f"{statistics.fmean(a) / statistics.fmean(result(c, 'oracle_reader', '0.1', 'kl_tf') for c in deleted):.1f}"

        for place in ("adjacent", "middle", "end"):
            part = [r["delete"] for r in gold if r["place"] == place]
            stale_mean = statistics.fmean(result(c, "stale", "0.1", "kl_tf") for c in part)
            for key in ("stale", "boundary", "tail", "reader", "exposure", "bridge", "rrf", "oracle_reader", "question_exposure"):
                mean_kl = statistics.fmean(result(c, key, "0.1", "kl_tf") for c in part)
                values[f"{tag}.place.{place}.{key}"] = kl(mean_kl)
                if key != "stale":
                    values[f"{tag}.place.{place}.{key}.removed"] = pct(1 - mean_kl / stale_mean)
                    place_removed.setdefault((place, key), []).append(1 - mean_kl / stale_mean)
        changed = [c for c in deleted if normalise(c["answers"]["stale"]) != normalise(c["answers"]["truth"])]
        values[f"{tag}.changed"] = str(len(changed))
        for key, _ in E2C_ANSWERS:
            same = sum(normalise(c["answers"][key]) == normalise(c["answers"]["truth"]) for c in deleted)
            restored = sum(normalise(c["answers"][key]) == normalise(c["answers"]["truth"]) for c in changed)
            answer_rows[key].append(f"{pct(same / len(deleted))} ({restored}/{len(changed)})" if key != "stale" else f"{pct(same / len(deleted))}")
            values[f"{tag}.same.{key}"] = pct(same / len(deleted))
            values[f"{tag}.restored.{key}"] = f"{restored} of {len(changed)}"
        reader_run = load(f"e2d_reader_first_{KEYS[model]}_main")
        found = {r["name"]: r for r in reader_run["rows"]} if reader_run else {}
        for key in ("segment_exposure", "segment_bridge"):
            if all(r["name"] in found for r in gold):
                texts = [found[r["name"]]["answers"][key] for r in gold]
                same = sum(normalise(t) == normalise(c["answers"]["truth"]) for t, c in zip(texts, deleted))
                restored = sum(normalise(t) == normalise(c["answers"]["truth"]) for t, c in zip(texts, deleted) if normalise(c["answers"]["stale"]) != normalise(c["answers"]["truth"]))
                answer_rows[key].append(f"{pct(same / len(deleted))} ({restored}/{len(changed)})")
                values[f"{tag}.restored.{key}"] = f"{restored} of {len(changed)}"
            else:
                answer_rows[key].append("–")

        floors = [(r["noise_floor"], r["delete"]) for r in gold if "noise_floor" in r]
        if floors:
            values[f"{tag}.noise.n"] = str(len(floors))
            for metric in ("kl_tf", "kl_first"):
                values[f"{tag}.noise.{metric}"] = kl(statistics.fmean(f[metric] for f, _ in floors))
                values[f"{tag}.noise.{metric}.median"] = kl(statistics.median(f[metric] for f, _ in floors))
                for key in ("bridge", "reader", "exposure", "oracle_reader"):
                    below = sum(c["results"][key]["0.1"][metric] <= f[metric] for f, c in floors)
                    values[f"{tag}.noise.{metric}.below.{key}"] = pct(below / len(floors))
                below = sum(c["results"]["stale"]["0"][metric] <= f[metric] for f, c in floors)
                values[f"{tag}.noise.{metric}.below.stale"] = pct(below / len(floors))

        rho = spearmanr([c["bridge_mass"] for c in deleted], [result(c, "stale", "0", "kl_tf") for c in deleted])
        values[f"{tag}.detector.rho"] = f"{rho.statistic:.2f}"
        values[f"{tag}.detector.p"] = pvalue(rho.pvalue)
        rho_reader = spearmanr([c["reader_mass"] for c in deleted], [result(c, "stale", "0", "kl_tf") for c in deleted])
        values[f"{tag}.detector.rho_reader"] = f"{rho_reader.statistic:.2f}"

        for lead in data["leads"]:
            part = [r["delete"] for r in gold if r["lead"] == lead]
            for key in LEAD_COLUMNS:
                values[f"{tag}.lead{lead}.{key}"] = kl(statistics.fmean(result(c, key, "0.1", "kl_tf") for c in part))
            values[f"{tag}.lead{lead}.suffix"] = f"{statistics.fmean(r['nS'] for r in gold if r['lead'] == lead):,.0f}"
            lead_rows.append([NAMES[model] if lead == data["leads"][0] else "", {1: "2nd", 4: "5th", 7: "8th"}.get(lead, str(lead + 1)),
                              values[f"{tag}.lead{lead}.suffix"], *[values[f"{tag}.lead{lead}.{k}"] for k in LEAD_COLUMNS]])

        distractors = [r["delete"] for r in rows if r["kind"] == "distractor"]
        if distractors:
            values[f"{tag}.distractor.n"] = str(len(distractors))
            for key in ("stale", "boundary", "tail", "reader", "exposure", "sum", "bridge", "oracle_reader"):
                values[f"{tag}.distractor.{key}"] = kl(statistics.fmean(result(c, key, "0.1", "kl_tf") for c in distractors))

        replaced, rcids = [r["replace"] for r in gold if "replace" in r], [question(r) for r in gold if "replace" in r]
        values[f"{tag}.replace.n"] = str(len(replaced))
        if replaced:
            adopting = [c for c in replaced if c["follows_edit"]["truth"]]
            values[f"{tag}.replace.truth_follows"] = str(len(adopting))
            values[f"{tag}.replace.truth_follows_share"] = pct(len(adopting) / len(replaced))
            for key, _ in E2C_REPLACE:
                ci = mean_ci([c["results"][key]["0" if key == "stale" else "0.1"]["kl_tf"] for c in replaced], rcids)
                follows = sum(c["follows_edit"][key] for c in adopting)
                same = sum(normalise(c["answers"][key]) == normalise(c["answers"]["truth"]) for c in replaced)
                replace_rows[key].append(f"{follows}/{len(adopting)}; {pct(same / len(replaced))}; {kl(ci[0])}")
                values[f"{tag}.replace.follows.{key}"] = f"{follows} of {len(adopting)}"
                values[f"{tag}.replace.same.{key}"] = pct(same / len(replaced))
                values[f"{tag}.replace.kl_tf.{key}"] = kl(ci[0])
                if adopting:
                    values[f"{tag}.replace.keeps_old.{key}"] = f"{sum(c['keeps_old'][key] for c in adopting)} of {len(adopting)}"
    if not models_done:
        return
    count = {0: "none", 1: "one", 2: "two", 3: "all three"}
    values["e2c.bridge_vs_exposure.significant"] = f"{count[sum(bridge_beats_exposure)]} of the {len(bridge_beats_exposure)} models".replace("all three of the 3 models", "all three models").replace("of the 3 models", "of the three models")
    for (place, key), removed in place_removed.items():
        low, high = pct(min(removed)), pct(max(removed))
        values[f"e2c.place.{place}.{key}.removed.range"] = low if low == high else f"{low} to {high}"
    shares = [float(values[f"e2c.{KEYS[m]}.kl_tf.{k}.10"]) / float(values[f"e2c.{KEYS[m]}.kl_tf.stale.10"])
              for m in NAMES if f"e2c.{KEYS[m]}.n" in values for k in ("boundary", "epic", "tail", "probe1")]
    values["e2c.windows_left_min"], values["e2c.windows_left_max"] = pct(min(shares)), pct(max(shares))
    long_cells = [(m, lead) for m in NAMES if f"e2c.{KEYS[m]}.n" in values for lead in (1, 4)]
    for key in ("rrf", "bridge"):
        values[f"e2c.long.{key}_beats_qe"] = str(sum(float(values[f"e2c.{KEYS[m]}.lead{lead}.{key}"]) < float(values[f"e2c.{KEYS[m]}.lead{lead}.question_exposure"])
                                                     for m, lead in long_cells))
    values["e2c.long.cells"] = str(len(long_cells))
    counts = ", ".join(values[f"e2c.{KEYS[m]}.n"] for m in NAMES if f"e2c.{KEYS[m]}.n" in values)
    order = ["stale", "boundary", "epic", "tail", "random", "probe1", "exposure", "reader", "sum", "rrf", "union", "bridge_layer", "bridge_below",
             "bridge", "density", "question_exposure", "question_bridge", "segment_exposure", "segment_bridge", "oracle", "oracle_reader"]
    table("ablation_tf", ["Policy", "Signal as used by"] + models_done, [metric_rows["kl_tf"][k] for k in order],
          f"**Table {{{{#C:ablation_tf}}}}.** Deleting a supporting document, 10% of the suffix recomputed, edited document 2nd, 5th or 8th of ten and the cached question at three places in the suffix ({counts} cases). Teacher-forced KL in nats from the re-prefill, averaged over the first 16 tokens of the re-prefill's reply; mean and 95% bootstrap interval over questions. The reader-first rows and whole-segment selection come from a second run over the same cases, which reproduces the rows the two runs share exactly.")
    table("ablation_first", ["Policy", "Signal as used by"] + models_done, [metric_rows["kl_first"][k] for k in order],
          f"**Table {{{{#C:ablation_first}}}}.** As Table {{{{#C:ablation_tf}}}}, KL at the first reply token only ({counts} cases).")
    table("ablation_paired", ["Model", "Bridge compared with", "Cases where bridge has lower KL", "Wilcoxon p", "Holm-adjusted p"], paired_rows,
          f"**Table {{{{#C:ablation_paired}}}}.** Paired comparisons on teacher-forced KL, Table {{{{#C:ablation_tf}}}}'s cases. The share counts cases; the Wilcoxon signed-rank test is on per-question mean differences, Holm-corrected within each model over the {len(E2C_DELETE) - 2} comparisons.")
    table("ablation_answers", ["Policy"] + models_done, list(answer_rows.values()),
          "**Table {{#C:ablation_answers}}.** Share of 16-token greedy answers identical to the re-prefill's, Table {{#C:ablation_tf}}'s cases; in brackets, how many of the answers that no repair changes each policy restores.")
    table("ablation_lead", ["Model", "Edited document", "Suffix tokens", "No repair", "Boundary", "Reader", "Exposure", "RRF", "**Bridge**", "Question, then exposure", "True dev. × reader"],
          lead_rows, "**Table {{#C:ablation_lead}}.** Teacher-forced KL at 10% by the position of the edited document among the ten.")
    if any(f"e2c.{KEYS[m]}.replace.n" in values and values[f"e2c.{KEYS[m]}.replace.n"] != "0" for m in NAMES):
        table("ablation_replace", ["Policy"] + models_done, list(replace_rows.values()),
              "**Table {{#E:ablation_replace}}.** Counterfactual replacement: every occurrence of the answer in the supporting document is replaced by another answer with the same number of words, and 10% of the suffix is recomputed. Each cell gives: of the cases in which the re-prefill's answer contains the new answer, in how many the repaired answer does too; the share of all replacement cases with the re-prefill's exact answer; and the teacher-forced KL.")


E7_POLICIES = [("boundary", "Boundary window"), ("tail", "Tail window"), ("reader", "Reader attention"),
               ("exposure", "Recorded exposure"), ("sum", "Rank sum of exposure and reader"), ("bridge", "**Bridge score**"),
               ("question_exposure", "Cached question (given), then exposure"), ("segment_bridge", "**Reader-first (segment found), then bridge**"),
               ("oracle", "True cell deviation (needs the re-prefill)"), ("oracle_reader", "Oracle score: true deviation × reader")]


def e7() -> None:
    import numpy as np

    rows_out, models_done = {key: [label] for key, label in [("stale", "No repair")] + E7_POLICIES + [("greedy", "Greedy selection on the metric")]}, []
    for model in NAMES:
        data = load(f"e7_mechanism_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e7.{KEYS[model]}"
        models_done.append(NAMES[model])
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.blocks"] = f"{statistics.fmean(r['blocks'] for r in rows):.0f}"
        values[f"{tag}.block"] = str(data["block"])
        stale = mean_ci([r["base"]["kl_tf"] for r in rows])
        rows_out["stale"].append(f"– | – | {interval(stale)}")
        values[f"{tag}.stale"] = kl(stale[0])
        greedy = [r["greedy"]["kl_tf"] for r in rows]
        g = mean_ci(greedy)
        values[f"{tag}.greedy"] = kl(g[0])
        rows_out["greedy"].append(f"– | – | {interval(g)}")
        def capture(r: dict, key: str) -> float:
            gains = [max(0.0, x) for x in r["reduction"]]
            k = max(1, math.ceil(0.1 * len(gains)))
            best = sum(sorted(gains, reverse=True)[:k])
            top = sorted(range(len(gains)), key=lambda b: -r["block_scores"][key][b])[:k]
            return sum(gains[b] for b in top) / best if best > 0 else math.nan
        values[f"{tag}.max_single"] = pct(statistics.fmean(max(r["reduction"]) / r["base"]["kl_tf"] for r in rows if r["base"]["kl_tf"] > 0))
        values[f"{tag}.near_zero"] = pct(statistics.fmean(sum(abs(x) < 0.01 * r["base"]["kl_tf"] for x in r["reduction"]) / len(r["reduction"]) for r in rows))
        for key, _ in E7_POLICIES:
            rhos = [r["correlation"][key] for r in rows if r["correlation"][key] == r["correlation"][key]]
            caught = [c for c in (capture(r, key) for r in rows) if c == c]
            hit = sum(r["top_hit"][key] for r in rows)
            k = mean_ci([r["policies"][key]["kl_tf"] for r in rows])
            values[f"{tag}.rho.{key}"] = f"{statistics.fmean(rhos):.2f}" if rhos else "–"
            values[f"{tag}.capture.{key}"] = pct(statistics.fmean(caught)) if caught else "–"
            values[f"{tag}.hit.{key}"] = f"{hit} of {len(rows)}"
            values[f"{tag}.kl.{key}"] = kl(k[0])
            values[f"{tag}.over_greedy.{key}"] = f"{k[0] / g[0]:.1f}" if g[0] > 0 else "–"
            values[f"{tag}.below_greedy.{key}"] = str(sum(r["policies"][key]["kl_tf"] <= r["greedy"]["kl_tf"] for r in rows))
            rows_out[key].append(f"{values[f'{tag}.capture.{key}']} | {hit}/{len(rows)} | {interval(k)}")
        rho_values = [float(values[f"{tag}.rho.{key}"]) for key, _ in E7_POLICIES if values[f"{tag}.rho.{key}"] != "–"]
        values[f"{tag}.rho_max"] = f"{max(rho_values):.2f}"
        ratios = [float(values[f"{tag}.over_greedy.{key}"]) for key in ("bridge", "segment_bridge", "oracle_reader") if values[f"{tag}.over_greedy.{key}"] != "–"]
        values[f"{tag}.over_greedy_min"] = f"{min(ratios):.1f}"
        greedy_mean = statistics.fmean(r["greedy"]["kl_tf"] for r in rows)
        practical = [statistics.fmean(r["policies"][key]["kl_tf"] for r in rows) / greedy_mean for key in ("bridge", "segment_bridge")]
        values[f"{tag}.over_greedy_practical_min"] = f"{math.floor(10 * min(practical)) / 10:.1f}"
        elsewhere = earlier_question_ids("e7_")
        values[f"{tag}.reused"] = {0: "none", 1: "one", 2: "two", 3: "three"}.get(len({question(r) for r in rows} & elsewhere), str(len({question(r) for r in rows} & elsewhere)))
        steps = max(len(r["path"]) for r in rows)
        values[f"{tag}.greedy_steps"] = str(steps)
        overlap = [len(set(r["greedy_blocks"]) & set(np.argsort(r["block_scores"]["bridge"])[::-1][: len(r["greedy_blocks"])].tolist())) / max(1, len(r["greedy_blocks"]))
                   for r in rows]
        values[f"{tag}.greedy_overlap.bridge"] = pct(statistics.fmean(overlap))
    if not models_done:
        return
    header = ["Policy"]
    for name in models_done:
        header += [f"{name}: useful blocks caught", "top block", "KL, 10%"]
    table_rows = []
    for key, cells in rows_out.items():
        line = [cells[0]]
        for cell in cells[1:]:
            line += cell.split(" | ")
        table_rows.append(line)
    counts = ", ".join(values[f"e7.{KEYS[m]}.n"] for m in NAMES if f"e7.{KEYS[m]}.n" in values)
    table("mechanism", header, table_rows,
          f"**Table {{{{#E:mechanism}}}}.** Mechanism test and a metric-optimising reference, deleting a supporting document (edited document 5th of ten; {counts} cases). The suffix is cut into blocks of 8 tokens and each block is recomputed alone. Useful blocks caught: of the KL reduction obtained by the 10% of blocks that are most useful on their own, the share obtained by the 10% of blocks with the highest score (the policy's score summed over a block), averaged over cases. Top block: cases in which the policy's highest-scoring block is among the five most useful blocks. KL: teacher-forced KL with 10% of the suffix recomputed, mean and 95% bootstrap interval. The greedy reference adds, block by block, the block that lowers the KL most, knowing the answer; it is not a policy.")


E8_POLICIES = [("boundary", "Boundary window"), ("tail", "Tail window"), ("reader", "Reader attention"),
               ("exposure", "Recorded exposure"), ("sum", "Rank sum of exposure and reader"),
               ("bridge", "**Bridge score**"), ("span_exposure", "Exposure to the answer's tokens only"),
               ("bridge_span", "**Bridge with exposure to the answer's tokens**"), ("question_bridge", "Cached question (given), then bridge"),
               ("segment_bridge", "**Reader-first (segment found), then bridge**"), ("oracle_reader", "Oracle score: true deviation × reader")]


def e8() -> None:
    table_rows, models_done = [], []
    for model in NAMES:
        data = load(f"e8_counterfactual_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e8.{KEYS[model]}"
        budgets = [str(b) for b in data["budgets"]]
        adopting = [r for r in rows if r["truth"]["follows"]]
        tried = data.get("tried", {})
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.answerable"] = pct(len(rows) / max(1, sum(v != "unsuitable" for v in tried.values())))
        values[f"{tag}.adopting"] = str(len(adopting))
        values[f"{tag}.adopting_share"] = pct(len(adopting) / len(rows))
        values[f"{tag}.all.stale.kl_tf"] = kl(statistics.fmean(r["stale"]["kl_tf"] for r in rows))
        for key in ("bridge", "segment_bridge", "boundary", "tail"):
            values[f"{tag}.all.{key}.kl_tf"] = kl(statistics.fmean(r["policies"][key]["0.1"]["kl_tf"] for r in rows))
        if not adopting:
            continue
        models_done.append(NAMES[model])
        n = len(adopting)
        cell = lambda k: (lambda p, lo, hi: f"{k}/{n} [{lo:.2f}, {hi:.2f}]")(*wilson(k, n))
        stale_follow = sum(r["stale"]["follows"] for r in adopting)
        values[f"{tag}.stale.follows"] = f"{stale_follow} of {n}"
        values[f"{tag}.stale.keeps_old"] = f"{sum(r['stale']['keeps_old'] for r in adopting)} of {n}"
        values[f"{tag}.stale.kl_tf"] = kl(statistics.fmean(r["stale"]["kl_tf"] for r in adopting))
        line = [NAMES[model], "No repair"] + [cell(stale_follow)] * len(budgets) + [values[f"{tag}.stale.kl_tf"]]
        table_rows.append(line)
        for key, label in E8_POLICIES:
            follows = [sum(r["policies"][key][b]["follows"] for r in adopting) for b in budgets]
            for b, k in zip(budgets, follows):
                values[f"{tag}.{key}.{b}"] = f"{k} of {n}"
                values[f"{tag}.{key}.{b}.share"] = pct(k / n)
            values[f"{tag}.{key}.kl_tf"] = kl(statistics.fmean(r["policies"][key]["0.1"]["kl_tf"] for r in adopting))
            table_rows.append(["", label] + [cell(k) for k in follows] + [values[f"{tag}.{key}.kl_tf"]])
        question = sum(r["question_only"]["follows"] for r in adopting)
        values[f"{tag}.question_only"] = f"{question} of {n}"
        values[f"{tag}.question_tokens"] = pct(statistics.fmean(r["question_only"]["tokens"] / r["nS"] for r in adopting))
        table_rows.append(["", "Cached question only (its own budget)", cell(question), "", "", kl(statistics.fmean(r["question_only"]["kl_tf"] for r in adopting))])
        values[f"{tag}.span_tokens"] = f"{statistics.fmean(r['span_tokens'] for r in rows):.1f}"
    if not models_done:
        return
    header = ["Model", "Policy"] + [f"New answer, {float(b):.0%}" for b in ("0.05", "0.1", "0.2")] + ["KL, 10%"]
    table("counterfactual", header, table_rows,
          "**Table {{#E:counterfactual}}.** Counterfactual replacement. In each case the model answered correctly with the supporting document as written; every occurrence of the answer in that document is then replaced by another answer of the same shape, and the re-prefill adopts the new answer. Cells give how many repaired replies contain the new answer, with a 95% Wilson interval, when 5%, 10% or 20% of the suffix is recomputed, and the teacher-forced KL at 10%.")


E2D_POLICIES = [("bridge", "Bridge score alone"), ("density", "Whole segments by bridge density, then bridge"),
                ("question_exposure", "Cached question (given), then exposure"), ("question_bridge", "Cached question (given), then bridge"),
                ("segment_exposure", "**Reader-first (segment found)**"), ("segment_bridge", "**Reader-first (segment found), then bridge**")]


def budget_tokens(fraction: float, n: int) -> int:
    """Tokens recomputed for a budget, exactly as the experiment scripts count them."""
    return min(n, max(1, math.ceil(round(fraction * n, 9))))


def e2d() -> None:
    rows_out, models_done = {key: [label] for key, label in E2D_POLICIES}, []
    found_row = ["Reader attention finds the question segment"]
    for model in NAMES:
        data = load(f"e2d_reader_first_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e2d.{KEYS[model]}"
        cells = {}
        for r in rows:
            cells[(r["lead"], r["place"])] = cells.get((r["lead"], r["place"]), 0) + 1
        if len(cells) < 9 or len(set(cells.values())) > 1:
            values[f"{tag}.status"] = f"incomplete: {len(rows)} cases"
            continue
        models_done.append(NAMES[model])
        cids = [question(r) for r in rows]
        found = sum(r["picked_is_question"] for r in rows)
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.found"] = f"{found} of {len(rows)}"
        values[f"{tag}.found_share"] = pct(found / len(rows))
        found_row.append(f"{found}/{len(rows)}")
        values[f"{tag}.question_reader_share"] = pct(statistics.fmean(r["question_reader_share"] for r in rows))
        kl10 = lambda key: [r["results"][key]["0.1"]["kl_tf"] for r in rows]
        for key, _ in E2D_POLICIES:
            ci = mean_ci(kl10(key), cids)
            rows_out[key].append(interval(ci))
            values[f"{tag}.{key}.10"] = kl(ci[0])
            values[f"{tag}.{key}.median"] = kl(statistics.median(kl10(key)))
            for budget in data["budgets"]:
                values[f"{tag}.{key}.{budget}"] = kl(statistics.fmean(r["results"][key][str(budget)]["kl_tf"] for r in rows))
        for ours, other in (("segment_bridge", "bridge"), ("segment_bridge", "question_exposure"), ("segment_bridge", "question_bridge"),
                            ("segment_exposure", "question_exposure"), ("question_bridge", "question_exposure")):
            a, b = kl10(ours), kl10(other)
            values[f"{tag}.{ours}_vs.{other}.win"] = pct(sum(x < y for x, y in zip(a, b)) / len(a))
            values[f"{tag}.{ours}_vs.{other}.p"] = pvalue(paired_test(a, b, cids))
        for lead in sorted({r["lead"] for r in rows}):
            part = [r for r in rows if r["lead"] == lead]
            for key in ("bridge", "question_exposure", "segment_exposure", "segment_bridge", "question_bridge"):
                values[f"{tag}.lead{lead}.{key}"] = kl(statistics.fmean(r["results"][key]["0.1"]["kl_tf"] for r in part))
            over = [r for r in part if r["question_tokens"] > budget_tokens(0.10, r["nS"])]
            fits = [r for r in part if r["question_tokens"] <= budget_tokens(0.10, r["nS"])]
            values[f"{tag}.lead{lead}.question_over_budget"] = f"{len(over)} of {len(part)}"
            for name, group in (("fits", fits), ("over", over)):
                for key in ("question_exposure", "question_bridge"):
                    if group:
                        values[f"{tag}.lead{lead}.{name}.{key}"] = kl(statistics.fmean(r["results"][key]["0.1"]["kl_tf"] for r in group))
        for key in ("question_exposure", "segment_exposure", "segment_bridge", "bridge"):
            same = statistics.fmean(float(normalise(r["answers"][key]) == normalise(r["answers"]["truth"])) for r in rows)
            values[f"{tag}.same.{key}"] = pct(same)
        for place in ("adjacent", "middle", "end"):
            part = [r for r in rows if r["place"] == place]
            for key in ("segment_exposure", "segment_bridge"):
                values[f"{tag}.place.{place}.{key}"] = kl(statistics.fmean(r["results"][key]["0.1"]["kl_tf"] for r in part))
    if not models_done:
        return
    found = [values[f"e2d.{KEYS[m]}.found"].split(" of ") for m in NAMES if f"e2d.{KEYS[m]}.found" in values]
    values["e2d.found_all"] = pct(sum(int(a) for a, _ in found) / sum(int(b) for _, b in found))
    values["e2d.models"] = str(len(found))
    counts = ", ".join(values[f"e2d.{KEYS[m]}.n"] for m in NAMES if f"e2d.{KEYS[m]}.n" in values)
    table("reader_first", ["Policy"] + models_done, list(rows_out.values()) + [found_row],
          f"**Table {{{{#C:reader_first}}}}.** Reader-first repair on the supporting-document cases of Table {{{{#C:ablation_tf}}}} ({counts} cases). The reader segment is either given (the cached question) or found as the suffix segment with the most reader attention per token. Teacher-forced KL in nats at 10%, mean and 95% bootstrap interval over questions.")


def task_accuracy() -> None:
    """Token F1 and containment of the dataset answer, for the E2c cases (deleting a supporting document)."""
    keys = ("truth", "stale", "boundary", "reader", "exposure", "bridge", "question_exposure", "oracle_reader")
    labels = {"truth": "Re-prefill (exact)", "stale": "No repair", "boundary": "Boundary window", "reader": "Reader attention", "exposure": "Recorded exposure",
              "bridge": "**Bridge score**", "question_exposure": "Cached question (given), then exposure", "segment_exposure": "Reader-first (segment found)",
              "oracle_reader": "Oracle score: true deviation × reader"}
    table_rows, header, contains, intervals = {}, ["Policy"], {}, {}
    for model in NAMES:
        ablation, reader = load(f"e2c_ablation_{KEYS[model]}_main"), load(f"e2d_reader_first_{KEYS[model]}_main")
        if not ablation or f"e2c.{KEYS[model]}.n" not in values:
            continue
        rows = [r for r in ablation["rows"] if r["kind"] == "gold"]
        cids, tag = [question(r) for r in rows], f"acc.{KEYS[model]}"
        answers = {key: [r["delete"]["answers"][key] for r in rows] for key in keys}
        if reader and len(reader["rows"]) >= len(rows):
            by_name = {r["name"]: r for r in reader["rows"]}
            answers["segment_exposure"] = [by_name[r["name"]]["answers"]["segment_exposure"] for r in rows]
        gold = [r["answer"] for r in rows]
        f1 = {key: [answer_f1(a, g) for a, g in zip(texts, gold)] for key, texts in answers.items()}
        for key, texts in answers.items():
            values[f"{tag}.f1.{key}"] = f"{statistics.fmean(f1[key]):.3f}"
            values[f"{tag}.contains.{key}"] = pct(statistics.fmean(float(normalise(g) in normalise(a)) for a, g in zip(texts, gold)))
            if key != "truth":
                diff = mean_ci([x - y for x, y in zip(f1[key], f1["truth"])], cids)
                values[f"{tag}.f1_minus_truth.{key}"] = f"{diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]".replace("-", "−")
            hits = sum(normalise(g) in normalise(a) for a, g in zip(texts, gold))
            share = wilson(hits, len(gold))
            contains.setdefault(key, []).append(share[0])
            intervals.setdefault(model, {})[key] = share
            table_rows.setdefault(key, [labels[key]]).extend([proportion(share), values[f"{tag}.f1.{key}"]])
        header += [f"{NAMES[model]}: contains gold", "token F1"]
    if table_rows:
        order = [k for k in ("truth", "stale", "boundary", "reader", "exposure", "bridge", "question_exposure", "segment_exposure", "oracle_reader") if k in table_rows]
        inside = all(truth[1] <= share[0] <= truth[2] for by_key in intervals.values() for truth in [by_key["truth"]] for share in by_key.values())
        values["acc.policies_vs_truth"] = ("every policy's share lies within the re-prefill's Wilson interval on every model" if inside
                                           else "the policies' shares do not all lie within the re-prefill's Wilson interval")
        for key in order:
            xs = contains[key]
            values[f"acc.contains.{key}.range"] = f"{min(xs):.0%} to {max(xs):.0%}" if round(100 * min(xs)) != round(100 * max(xs)) else f"{min(xs):.0%}"
        table("task", header, [table_rows[k] for k in order],
              "**Table {{#C:task}}.** Answer accuracy against the dataset's gold answer in the main experiment (deleting a supporting document, 10% of the suffix recomputed): share of greedy 16-token replies that contain the gold answer after normalisation, with a 95% Wilson interval (which treats the 108 cases per model as independent although each question appears in three layouts), and token F1 against the gold answer. Rank fusion was not decoded in this run.")


MEMORY_MODELS = [("qwen0.5b", "Qwen/Qwen2.5-0.5B-Instruct"), ("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
                 ("smol1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"), ("qwen7b", "Qwen/Qwen2.5-7B-Instruct")]
LABELS = {"qwen7b": "Qwen2.5-7B", **{KEYS[m]: NAMES[m] for m in NAMES}}


def memory(tokens: int = 10_000, segments: int = 50) -> None:
    """What repair keeps beside the KV cache, from each model's configuration (16-bit cache, 32-bit exposure)."""
    import os
    from transformers import AutoConfig

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = []
    for key, repo in MEMORY_MODELS:
        try:
            cfg = AutoConfig.from_pretrained(repo)
        except Exception:
            continue
        L, H, G = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
        kv = 2 * L * G * d * 2 * tokens
        exposure = tokens * (segments + 1) / 2 * 4
        sketch = 4 * G * (d + 1) * 2 * tokens
        mb = lambda x: f"{x / 2**20:,.1f}"
        values[f"mem.{key}.kv"], values[f"mem.{key}.exposure"], values[f"mem.{key}.sketch"] = mb(kv), mb(exposure), mb(sketch)
        values[f"mem.{key}.exposure_share"] = f"{100 * exposure / kv:.2f}%"
        values[f"mem.{key}.sketch_share"] = f"{100 * sketch / kv:.1f}%"
        rows.append([LABELS[key], f"{L} × {G} × {d}", mb(kv), f"{mb(exposure)} ({values[f'mem.{key}.exposure_share']})", f"{mb(sketch)} ({values[f'mem.{key}.sketch_share']})"])
    values["mem.tokens"], values["mem.segments"] = f"{tokens:,}", str(segments)
    if rows:
        table("memory", ["Model", "Layers × KV heads × head dim", "KV cache (MiB)", "Exposure (MiB, share of cache)", "Query sketch for inserts (MiB, share)"], rows,
              f"**Table {{{{#E:memory}}}}.** What repair keeps beside the KV cache for a context of {tokens:,} tokens in {segments} segments: exposure (one 32-bit number per token and earlier segment, {(segments + 1) / 2:g} earlier segments on average) and the optional query sketch (one 16-bit query and normaliser per key/value group at four layers). The KV cache is counted in 16 bits; 1 MiB = 2^20 bytes.")


def e6() -> None:
    for model in NAMES:
        data = load(f"e6_cost_{KEYS[model]}_main")
        if not data:
            continue
        rows, tag = data["rows"], f"e6.{KEYS[model]}"
        median = lambda field: statistics.median(r[field] for r in rows)
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.suffix"] = f"{statistics.fmean(r['nS'] for r in rows):,.0f}"
        values[f"{tag}.reprefill"] = f"{median('reprefill'):.3f}"
        values[f"{tag}.reader"] = f"{median('reader'):.3f}"
        values[f"{tag}.read_only"] = f"{median('read_only'):.3f}"
        for budget in data["budgets"]:
            total = median("reader") + median(f"repair@{budget}")
            values[f"{tag}.repair.{budget}"] = f"{median(f'repair@{budget}'):.3f}"
            values[f"{tag}.ratio.{budget}"] = f"{total / median('reprefill'):.2f}"
            values[f"{tag}.speedup.{budget}"] = f"{median('reprefill') / total:.1f}"
            end_to_end = (total + median("read_only")) / (median("reprefill") + median("read_only"))
            values[f"{tag}.ratio_e2e.{budget}"] = f"{end_to_end:.2f}"
    for model in NAMES:
        data = load(f"e6b_recording_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e6b.{KEYS[model]}"
        median = lambda field: statistics.median(r[field] for r in rows)
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.tokens"] = f"{statistics.fmean(r['n'] for r in rows):,.0f}"
        values[f"{tag}.segments"] = f"{statistics.fmean(r['segments'] for r in rows):.0f}"
        for field in ("plain", "exposure", "sketch"):
            values[f"{tag}.{field}"] = f"{median(field):.3f}"
        values[f"{tag}.exposure_overhead"] = pct(median("exposure") / median("plain") - 1)
        values[f"{tag}.sketch_overhead"] = pct(median("sketch") / median("plain") - 1)


def lead_stratum(row: dict) -> str:
    """Suffix length class of a main-experiment case: the edited document 8th of ten leaves a short suffix."""
    return "short" if row.get("lead") == 7 else "long"


def win_p(a: list[float], b: list[float], clusters: list | None = None) -> tuple[float, float]:
    """Share of cases in which a is lower than b, and the paired Wilcoxon p-value (per-cluster means if clustered)."""
    return sum(x < y for x, y in zip(a, b)) / len(a), paired_test(a, b, clusters)


def ratio_ci(a: list[float], b: list[float], clusters: list) -> tuple[float, float, float]:
    """Ratio of means a/b with a 95% BCa bootstrap interval that resamples whole clusters."""
    import numpy as np

    labels = {label: index for index, label in enumerate(dict.fromkeys(clusters))}
    ids = np.array([labels[c] for c in clusters])
    sa = np.bincount(ids, weights=np.asarray(a, dtype=float), minlength=len(labels))
    sb = np.bincount(ids, weights=np.asarray(b, dtype=float), minlength=len(labels))
    draws = np.random.default_rng(0).integers(0, len(labels), (4000, len(labels)))
    ratios = sa[draws].sum(axis=1) / sb[draws].sum(axis=1)
    observed = float(sa.sum() / sb.sum())
    jack = (sa.sum() - sa) / (sb.sum() - sb) if len(labels) > 1 else np.array([observed])
    return (observed, *bca_bounds(observed, ratios, jack))


def ratio_digits(t: tuple[float, float, float]) -> int:
    """More decimals when a bound is close to 1, so that rounding cannot hide which side of 1 it is on."""
    return 3 if any(0.995 <= bound <= 1.005 for bound in t[1:]) else 2 if any(0.95 <= bound <= 1.05 for bound in t[1:]) else 1


def ratio_text(t: tuple[float, float, float], digits: int | None = None) -> str:
    """Ratio and interval, by default with `ratio_digits` decimals."""
    digits = ratio_digits(t) if digits is None else digits
    return f"{t[0]:.{digits}f} [{t[1]:.{digits}f}, {t[2]:.{digits}f}]"


def mcnemar(first: list[bool], second: list[bool]) -> float:
    """Exact McNemar test on paired successes."""
    from scipy.stats import binomtest

    b = sum(x and not y for x, y in zip(first, second))
    c = sum(y and not x for x, y in zip(first, second))
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


ROBUST_POLICIES = [("boundary", "Boundary window"), ("tail", "Tail window"), ("reader", "Reader attention"), ("exposure", "Recorded exposure"),
                   ("bridge", "Bridge score"), ("rrf", "Rank fusion (RRF)"), ("segment_exposure", "Reader-first (segment found)")]
robust_settings: list[dict] = []
experiment_cases: dict[str, list[dict]] = {}
EXPERIMENT_LABELS = [("main", "Main (3 models)"), ("replication", "Replication (3 models)"), ("long", "4k–16k contexts (1.5B)"),
                     ("agent", "Synthetic traces (1.5B)"), ("needle", "Needle, 5% (1.5B)"), ("7b", "7B models"),
                     ("traces", "Real transcripts (3 models)")]


def collect(experiment: str, cluster: str, stale: float, kls: dict[str, float]) -> None:
    """One case of an experiment at its summary budget, for the per-experiment table (Table {{#robust_experiments}})."""
    experiment_cases.setdefault(experiment, []).append({"cluster": cluster, "stale": stale, **{k: v for k, v in kls.items() if v == v}})


def add_setting(experiment: str, label: str, budget: float, stale: float, means: dict[str, float]) -> None:
    """One setting of the robustness summary: the no-repair KL and the mean KL of each practical policy run there."""
    robust_settings.append({"experiment": experiment, "setting": label, "budget": budget, "stale": stale,
                            "means": {k: v for k, v in means.items() if v == v}})


def e2c_strata() -> None:
    """The main experiment by suffix length: per-stratum tests and settings on top of the pooled values."""
    strata_tests, ratio_rows = [], []
    for model in NAMES:
        data, extra = load(f"e2c_ablation_{KEYS[model]}_main"), load(f"e2d_reader_first_{KEYS[model]}_main")
        if not data or not extra or f"e2c.{KEYS[model]}.n" not in values:
            continue
        tag = f"e2c.{KEYS[model]}"
        found = {r["name"]: r for r in extra["rows"]}
        gold = [r for r in data["rows"] if r["kind"] == "gold" and r["name"] in found]
        get = lambda r, key: (found[r["name"]]["results"][key]["0.1"]["kl_tf"] if key in ("segment_exposure", "segment_bridge", "density", "question_bridge")
                              else r["delete"]["results"][key]["0" if key == "stale" else "0.1"]["kl_tf"])
        for r in gold:
            collect("main", question(r), get(r, "stale"), {k: get(r, k) for k in ("boundary", "tail", "reader", "exposure", "bridge", "rrf", "segment_exposure", "question_exposure")})
        for stratum in ("short", "long"):
            part = [r for r in gold if lead_stratum(r) == stratum]
            cids = [question(r) for r in part]
            values[f"{tag}.{stratum}.n"] = str(len(part))
            values[f"{tag}.{stratum}.suffix"] = f"{statistics.fmean(r['nS'] for r in part):.0f}"
            values[f"{tag}.{stratum}.budget"] = f"{statistics.fmean(budget_tokens(0.10, r['nS']) for r in part):.0f}"
            means = {}
            for key in ("stale", "boundary", "tail", "epic", "probe1", "reader", "exposure", "bridge", "rrf", "question_exposure",
                        "segment_exposure", "segment_bridge", "oracle_reader"):
                means[key] = statistics.fmean(get(r, key) for r in part)
                values[f"{tag}.{stratum}.{key}"] = kl(means[key])
            practical = ("boundary", "tail", "epic", "probe1", "reader", "exposure", "bridge", "rrf", "segment_exposure")
            values[f"{tag}.{stratum}.first_lowest"] = "yes" if min(practical, key=means.get) == "segment_exposure" else "no"
            for a, b in (("segment_exposure", "rrf"), ("segment_exposure", "bridge"), ("rrf", "bridge"), ("bridge", "reader"), ("rrf", "reader")):
                share, p = win_p([get(r, a) for r in part], [get(r, b) for r in part], cids)
                values[f"{tag}.{stratum}.{a}_vs.{b}.win"], values[f"{tag}.{stratum}.{a}_vs.{b}.p"] = pct(share), pvalue(p)
            for budget in ("0.05", "0.1", "0.2"):
                at = lambda r, key: (found[r["name"]]["results"][key][budget]["kl_tf"] if key == "segment_exposure"
                                     else r["delete"]["results"][key][budget]["kl_tf"])
                add_setting("main", f"Main, {NAMES[model]}, {stratum} suffix, {float(budget):.0%}", float(budget), means["stale"],
                            {k: statistics.fmean(at(r, k) for r in part) for k, _ in ROBUST_POLICIES})
        cids = [question(r) for r in gold]
        for a, b in (("segment_exposure", "rrf"), ("segment_exposure", "bridge"), ("rrf", "bridge")):
            share, p = win_p([get(r, a) for r in gold], [get(r, b) for r in gold], cids)
            values[f"{tag}.pooled.{a}_vs.{b}.win"], values[f"{tag}.pooled.{a}_vs.{b}.p"] = pct(share), pvalue(p)
        for stratum in ("long", "short", "pooled"):
            line = [NAMES[model] if stratum == "long" else "", {"long": "Long suffix", "short": "Short suffix", "pooled": "Both"}[stratum]]
            for a, b in (("segment_exposure", "rrf"), ("segment_exposure", "bridge"), ("rrf", "bridge")):
                line.append(f"{values[f'{tag}.{stratum}.{a}_vs.{b}.win']}; {values[f'{tag}.{stratum}.{a}_vs.{b}.p']}")
            strata_tests.append(line)
        for key in ("bridge", "rrf", "segment_exposure", "reader"):
            worse = [get(r, key) / get(r, "stale") for r in gold if get(r, key) > get(r, "stale")]
            values[f"{tag}.harm.{key}"] = str(len(worse))
            values[f"{tag}.harm_ratio.{key}"] = (f"{max(worse):.0f}" if max(worse) >= 10 else f"{max(worse):.1f}") if worse else "–"
        ratio = ratio_ci([get(r, "bridge") for r in gold], [get(r, "rrf") for r in gold], cids)
        values[f"{tag}.ratio.bridge_rrf"] = ratio_text(ratio)
        values[f"{tag}.ratio.bridge_rrf.side"] = "above" if ratio[1] > 1 else "below" if ratio[2] < 1 else "includes"
        ratio_line = [NAMES[model]]
        for key in ("bridge", "rrf", "segment_exposure"):
            t = ratio_ci([get(r, key) for r in gold], [get(r, "oracle_reader") for r in gold], cids)
            values[f"{tag}.ratio_ideal.{key}"] = ratio_text(t)
            values[f"{tag}.ratio_ideal.{key}.point"] = f"{t[0]:.1f}"
            ratio_line.append(ratio_text(t))
        ratio_rows.append(ratio_line)
        for key in ("bridge", "rrf", "segment_exposure", "oracle_reader", "reader"):
            values[f"{tag}.median.{key}"] = kl(statistics.median(get(r, key) for r in gold))
            t = ratio_ci([get(r, key) for r in gold], [get(r, "stale") for r in gold], cids)
            values[f"{tag}.left.{key}"] = pct(t[0])
        later = lambda r, key: (get(r, key) * r["delete"]["reply_tokens"] - (found[r["name"]]["results"][key]["0.1"]["kl_first"] if key.startswith("segment")
                                else r["delete"]["results"][key]["0" if key == "stale" else "0.1"]["kl_first"])) / (r["delete"]["reply_tokens"] - 1)
        longer = [r for r in gold if r["delete"]["reply_tokens"] > 1]
        values[f"{tag}.later.n"] = str(len(longer))
        for key in ("stale", "reader", "bridge", "rrf", "segment_exposure"):
            values[f"{tag}.later.{key}"] = kl(statistics.fmean(later(r, key) for r in longer))
        share, p = win_p([later(r, "bridge") for r in longer], [later(r, "reader") for r in longer], [question(r) for r in longer])
        values[f"{tag}.later.bridge_vs.reader.win"], values[f"{tag}.later.bridge_vs.reader.p"] = pct(share), pvalue(p)
        values[f"{tag}.reply_mean"] = f"{statistics.fmean(r['delete']['reply_tokens'] for r in gold):.1f}"
        values[f"{tag}.reply_median"] = f"{statistics.median(r['delete']['reply_tokens'] for r in gold):.0f}"
    if strata_tests:
        table("strata_tests", ["Model", "Cases", "Reader-first vs RRF", "Reader-first vs bridge", "RRF vs bridge"], strata_tests,
              "**Table {{#C:strata_tests}}.** Paired comparisons of the three combined policies at 10% by suffix length (Table {{#main_strata}}): share of cases in which the first policy has lower teacher-forced KL; two-sided Wilcoxon signed-rank test on per-question mean differences, not corrected for multiplicity.")
        table("ratio_ideal", ["Model", "Bridge score", "Rank fusion (RRF)", "Reader-first (found)"], ratio_rows,
              "**Table {{#C:ratio_ideal}}.** Mean teacher-forced KL at 10% divided by that of the oracle score (true deviation × reader attention, which needs the re-prefill), with a 95% bootstrap interval over questions. The oracle score is a reference, not a bound: the bridge score has lower KL than it in some cases.")
        lefts = [float(values[f"e2c.{KEYS[m]}.left.{k}"].rstrip("%")) for m in NAMES for k in ("bridge", "rrf") if f"e2c.{KEYS[m]}.left.{k}" in values]
        values["e2c.left_combined_min"], values["e2c.left_combined_max"] = f"{min(lefts):.0f}%", f"{max(lefts):.0f}%"
        points = {k: [float(values[f"e2c.{KEYS[m]}.ratio_ideal.{k}.point"]) for m in NAMES if f"e2c.{KEYS[m]}.ratio_ideal.{k}.point" in values]
                  for k in ("bridge", "rrf", "segment_exposure")}
        for k, xs in points.items():
            values[f"e2c.ratio_ideal.{k}.min"], values[f"e2c.ratio_ideal.{k}.max"] = f"{min(xs):.1f}", f"{max(xs):.1f}"
        for key in ("bridge", "rrf", "segment_exposure", "reader"):
            counts = [int(values[f"e2c.{KEYS[m]}.harm.{key}"]) for m in NAMES if f"e2c.{KEYS[m]}.harm.{key}" in values]
            values[f"e2c.harm.{key}.min"], values[f"e2c.harm.{key}.max"] = str(min(counts)), str(max(counts))
        holms = [values[f"e2c.{KEYS[m]}.vs.rrf.holm"] for m in NAMES if f"e2c.{KEYS[m]}.vs.rrf.holm" in values]
        values["e2c.vs.rrf.holm_min"] = min(holms, key=float)
        p_long = [values[f"e2c.{KEYS[m]}.long.segment_exposure_vs.rrf.p"] for m in NAMES if f"e2c.{KEYS[m]}.long.segment_exposure_vs.rrf.p" in values]
        values["e2c.long.segment_vs_rrf.p_max"] = max(p_long, key=float)
        p_short = [float(values[f"e2c.{KEYS[m]}.short.segment_exposure_vs.rrf.p"]) for m in NAMES if f"e2c.{KEYS[m]}.short.segment_exposure_vs.rrf.p" in values]
        values["e2c.short.segment_vs_rrf.significant"] = {0: "none", 1: "one", 2: "two", 3: "three"}[sum(p < 0.05 for p in p_short)]
        lowest = [values[f"e2c.{KEYS[m]}.short.first_lowest"] == "yes" for m in NAMES if f"e2c.{KEYS[m]}.short.first_lowest" in values]
        values["e2c.short.first_lowest"] = "on all three small models" if all(lowest) and len(lowest) == 3 else f"on {sum(lowest)} of the {len(lowest)} small models"
        excl = [values[f"e2c.{KEYS[m]}.ratio.bridge_rrf.side"] for m in NAMES if f"e2c.{KEYS[m]}.ratio.bridge_rrf.side" in values]
        words = {0: "none", 1: "one", 2: "two", 3: "three"}
        values["e2c.bridge_rrf.rrf_better"] = words[excl.count("above")]
        values["e2c.bridge_rrf.bridge_better"] = words[excl.count("below")]
    rows_out = []
    for key, label in (("stale", "No repair"), ("boundary", "Boundary window"), ("tail", "Tail window"), ("epic", "First tokens of each document"),
                       ("probe1", "First-layer deviation"), ("reader", "Reader attention"), ("exposure", "Recorded exposure"),
                       ("bridge", "**Bridge score**"), ("rrf", "**Rank fusion (RRF)**"),
                       ("question_exposure", "Cached question (given), then exposure"), ("segment_exposure", "Reader-first (segment found)"),
                       ("oracle_reader", "Oracle score: true deviation × reader")):
        line = [label]
        for model in NAMES:
            tag = f"e2c.{KEYS[model]}"
            if f"{tag}.short.n" in values:
                line += [values[f"{tag}.long.{key}"], values[f"{tag}.short.{key}"]]
        rows_out.append(line)
    header = ["Policy"]
    for model in NAMES:
        tag = f"e2c.{KEYS[model]}"
        if f"{tag}.short.n" in values:
            header += [f"{NAMES[model]}: long suffix", "short suffix"]
    if len(header) > 1:
        tag = "e2c.qwen1.5b"
        table("main_strata", header, rows_out,
              f"**Table {{{{#main_strata}}}}.** Teacher-forced KL in nats from the re-prefill after deleting a supporting document, 10% of the suffix recomputed, by suffix length. Long suffix: the edited document 2nd or 5th of ten ({values.get(f'{tag}.long.n', '–')} cases per model, about {values.get(f'{tag}.long.suffix', '–')} suffix tokens, a budget of about {values.get(f'{tag}.long.budget', '–')} tokens on Qwen2.5-1.5B); short suffix: 8th of ten ({values.get(f'{tag}.short.n', '–')} cases, about {values.get(f'{tag}.short.suffix', '–')} tokens, budget about {values.get(f'{tag}.short.budget', '–')}). Means; intervals and every other policy in Appendix C.")


E9_ROWS = [("stale", "No repair"), ("boundary", "Boundary window"), ("tail", "Tail window"), ("reader", "Reader attention"),
           ("exposure", "Recorded exposure"), ("bridge", "**Bridge score**"), ("rrf", "**Rank fusion (RRF)**"),
           ("segment_exposure", "Reader-first (segment found)"), ("oracle_reader", "Oracle score: true deviation × reader"),
           ("question_only", "Cached question only (own budget)")]


def scale_run(stem: str, tag: str, label: str, table_name: str, experiment: str, caption: str) -> None:
    data = load(stem)
    if not data or not data["rows"] or len(data["rows"]) < 108:
        return
    rows = data["rows"]
    cids = [question(r) for r in rows]
    get = lambda r, key: r["results"][key]["kl_tf"]
    values[f"{tag}.n"], values[f"{tag}.questions"] = str(len(rows)), str(len(set(cids)))
    values[f"{tag}.found"] = f"{sum(r['picked_is_question'] for r in rows)} of {len(rows)}"
    check = data["reference_check"]
    values[f"{tag}.check.tokens"], values[f"{tag}.check.argmax"] = str(check["tokens"]), pct(check["same_argmax"])
    values[f"{tag}.check.gap"] = f"{check['max_abs_logit_gap']:.1f}"
    for r in rows:
        collect("7b", question(r), get(r, "stale"), {k: get(r, k) for k, _ in ROBUST_POLICIES if k in r["results"]})
    floor = [get(r, "floor") for r in rows]
    values[f"{tag}.floor.mean"], values[f"{tag}.floor.median"] = kl(statistics.fmean(floor)), kl(statistics.median(floor))
    values[f"{tag}.floor.zero"] = f"{sum(f == 0.0 for f in floor)} of {len(floor)}"
    for stratum in ("long", "short"):
        values[f"{tag}.floor.{stratum}"] = kl(statistics.fmean(get(r, "floor") for r in rows if lead_stratum(r) == stratum))
    table_rows = []
    for key, label in E9_ROWS:
        xs = [get(r, key) for r in rows]
        ci = mean_ci(xs, cids)
        values[f"{tag}.{key}"], values[f"{tag}.{key}.median"] = kl(ci[0]), kl(statistics.median(xs))
        line = [label, interval(ci), values[f"{tag}.{key}.median"]]
        for stratum in ("long", "short"):
            part = [get(r, key) for r in rows if lead_stratum(r) == stratum]
            values[f"{tag}.{stratum}.{key}"] = kl(statistics.fmean(part))
            line.append(values[f"{tag}.{stratum}.{key}"])
        table_rows.append(line)
    for key in ("bridge", "rrf", "segment_exposure", "oracle_reader", "question_only", "reader"):
        values[f"{tag}.left.{key}"] = pct(ratio_ci([get(r, key) for r in rows], [get(r, "stale") for r in rows], cids)[0])
    for key in ("bridge", "rrf", "segment_exposure"):
        values[f"{tag}.ratio_ideal.{key}"] = ratio_text(ratio_ci([get(r, key) for r in rows], [get(r, "oracle_reader") for r in rows], cids))
    family = {}
    for a, b in (("bridge", "reader"), ("rrf", "reader"), ("bridge", "exposure"), ("rrf", "bridge"), ("segment_exposure", "rrf"),
                 ("segment_exposure", "bridge"), ("bridge", "oracle_reader"), ("rrf", "oracle_reader")):
        share, p = win_p([get(r, a) for r in rows], [get(r, b) for r in rows], cids)
        values[f"{tag}.{a}_vs.{b}.win"], values[f"{tag}.{a}_vs.{b}.p"] = pct(share), pvalue(p)
        family[f"{a}_vs.{b}"] = p
    for name, p in holm(family).items():
        values[f"{tag}.{name}.holm"] = pvalue(p)
    pattern = {}
    for stratum in ("long", "short"):
        part = [r for r in rows if lead_stratum(r) == stratum]
        share, p = win_p([get(r, "segment_exposure") for r in part], [get(r, "rrf") for r in part], [question(r) for r in part])
        values[f"{tag}.{stratum}.segment_exposure_vs.rrf.win"], values[f"{tag}.{stratum}.segment_exposure_vs.rrf.p"] = pct(share), pvalue(p)
        first_lower = statistics.fmean(get(r, "segment_exposure") for r in part) < statistics.fmean(get(r, "rrf") for r in part)
        pattern[stratum] = (first_lower == (stratum == "short"), p < 0.05)
        values[f"{tag}.{stratum}.direction"] = "in the same direction" if pattern[stratum][0] else "in the opposite direction"
        values[f"{tag}.{stratum}.vs_rrf_word"] = "lower than" if first_lower else "higher than"
        add_setting(experiment, f"{label}, {stratum} suffix, 10%", 0.1, statistics.fmean(get(r, "stale") for r in part),
                    {k: statistics.fmean(get(r, k) for r in part) for k, _ in ROBUST_POLICIES if k in rows[0]["results"]})
    as_predicted = all(direction for direction, _ in pattern.values())
    values[f"{tag}.pattern"] = ("repeats" if as_predicted and all(significant for _, significant in pattern.values())
                                else "repeats in direction but is not significant in both strata" if as_predicted else "does not repeat")
    means = {key: statistics.fmean(get(r, key) for r in rows) for key in ("stale", "bridge", "rrf")}
    share = statistics.fmean(floor) / (means["stale"] - means["bridge"])
    values[f"{tag}.floor.vs_repair"] = pct(share) if share >= 0.01 else f"{100 * share:.2g}%"
    values[f"{tag}.bridge_rrf_gap"] = kl(abs(means["bridge"] - means["rrf"]))
    values[f"{tag}.floor.vs_gap"] = "at least as large as" if statistics.fmean(floor) >= abs(means["bridge"] - means["rrf"]) else "smaller than"
    for key in ("stale", "bridge", "segment_exposure"):
        values[f"{tag}.same.{key}"] = pct(statistics.fmean(float(normalise(r["answers"][key]) == normalise(r["answers"]["truth"])) for r in rows))
    table(table_name, ["Policy", "Mean [95% CI]", "Median", "Long suffix", "Short suffix"], table_rows,
          caption.format(n=len(rows), questions=len(set(cids)), floor=values[f"{tag}.floor.mean"], floor_median=values[f"{tag}.floor.median"]))


def e9() -> None:
    small = [float(values[f"e2c.{KEYS[m]}.kl_tf.stale.10"]) for m in NAMES if f"e2c.{KEYS[m]}.kl_tf.stale.10" in values]
    if small:
        values["e2c.stale.range"] = f"{kl(min(small))} to {kl(max(small))}"
    scale_run("e9_scale_qwen7b_main", "e9.qwen7b", "Qwen2.5-7B", "scale", "qwen7b",
              "**Table {{{{#E:scale}}}}.** Qwen2.5-7B-Instruct in bfloat16 on the cases of Table {{{{#main_strata}}}} ({n} cases, {questions} questions): teacher-forced KL in nats from the re-prefill, 10% of the suffix recomputed; interval over questions. Two exact bfloat16 computations of the re-prefill (one pass, or the prefix read from cache) differ by {floor} on average (median {floor_median}).")
    scale_run("e19_family_mistral7b_main", "e19.mistral7b", "Mistral-7B", "scale_mistral", "mistral7b",
              "**Table {{{{#E:scale_mistral}}}}.** Mistral-7B-Instruct-v0.3 in bfloat16 on the same questions, documents and layouts as Table {{{{#main_strata}}}}, encoded with its own tokenizer ({n} cases, {questions} questions): teacher-forced KL in nats from the re-prefill, 10% of the suffix recomputed; interval over questions. Two exact bfloat16 computations of the re-prefill differ by {floor} on average (median {floor_median}).")


E10_POLICIES = [("boundary", "Boundary"), ("reader", "Reader"), ("exposure", "Exposure"), ("bridge", "**Bridge score**"),
                ("question_exposure", "Question first"), ("segment_exposure", "Reader-first (found)"), ("oracle_reader", "Oracle score")]


def e10() -> None:
    data = load("e10_long_qwen1.5b_main")
    if not data or not data["rows"]:
        return
    rrf_run = load("e10_long_qwen1.5b_rrf")
    rows, tag = data["rows"], "e10.qwen1.5b"
    cid = lambda r: r["name"].split("/")[2]
    for r in rows:
        collect("long", cid(r), r["results"]["stale"]["kl_tf"], {k: r["results"][k]["0.1"]["kl_tf"] for k in ("boundary", "reader", "exposure", "bridge", "segment_exposure")})
    table_rows, best_counts, reproduced = [], {}, []
    by_name = {r["name"]: r for r in rrf_run["rows"]} if rrf_run else {}
    for target in sorted({r["target"] for r in rows}):
        part = [r for r in rows if r["target"] == target]
        t, cids = f"{tag}.{target}", [cid(r) for r in part]
        values[f"{t}.n"], values[f"{t}.questions"] = str(len(part)), str(len(set(cids)))
        values[f"{t}.tokens"] = f"{statistics.fmean(r['tokens'] for r in part):,.0f}"
        values[f"{t}.suffix"] = f"{statistics.fmean(r['nS'] for r in part):,.0f}"
        values[f"{t}.found"] = f"{sum(r['picked_is_question'] for r in part)} of {len(part)}"
        values[f"{t}.stale"] = kl(statistics.fmean(r["results"]["stale"]["kl_tf"] for r in part))
        values[f"{t}.question_only"] = kl(statistics.fmean(r["results"]["question_only"]["kl_tf"] for r in part))
        for budget in data["budgets"]:
            get = lambda r, key: r["results"][key][str(budget)]["kl_tf"]
            means = {key: statistics.fmean(get(r, key) for r in part) for key, _ in E10_POLICIES}
            for key, m in means.items():
                values[f"{t}.{key}.{budget}"] = kl(m)
            for other in ("segment_exposure", "reader", "exposure"):
                share, p = win_p([get(r, "bridge") for r in part], [get(r, other) for r in part], cids)
                values[f"{t}.bridge_vs.{other}.{budget}.win"], values[f"{t}.bridge_vs.{other}.{budget}.p"] = pct(share), pvalue(p)
            extra = {}
            if by_name and all(r["name"] in by_name for r in part):
                extra["rrf"] = statistics.fmean(by_name[r["name"]]["results"]["rrf"][str(budget)]["kl_tf"] for r in part)
                values[f"{t}.rrf.{budget}"] = kl(extra["rrf"])
                reproduced += [by_name[r["name"]]["results"]["bridge"][str(budget)]["kl_tf"] == get(r, "bridge") for r in part]
            practical = {k: v for k, v in {**means, **extra}.items() if k not in ("oracle_reader", "question_exposure")}
            winner = min(practical, key=practical.get)
            best_counts[winner] = best_counts.get(winner, 0) + 1
            tokens = statistics.fmean(budget_tokens(budget, r["nS"]) for r in part)
            cells = [values[f"{t}.{k}.{budget}"] for k, _ in E10_POLICIES]
            cells.insert([k for k, _ in E10_POLICIES].index("bridge") + 1, values.get(f"{t}.rrf.{budget}", "–"))
            table_rows.append([f"{target // 1000}k" if budget == data["budgets"][0] else "", f"{budget:.0%} ({tokens:,.0f})",
                               values[f"{t}.stale"] if budget == data["budgets"][0] else ""] + cells)
            add_setting("long", f"{target // 1000}k tokens, {budget:.0%}", budget, statistics.fmean(r["results"]["stale"]["kl_tf"] for r in part),
                        {**{k: means[k] for k, _ in ROBUST_POLICIES if k in means}, **extra})
    values[f"{tag}.bridge_best"] = f"{best_counts.get('bridge', 0)} of {len(table_rows)}"
    values[f"{tag}.rrf_best"] = str(best_counts.get("rrf", 0))
    values[f"{tag}.segment_best"] = str(best_counts.get("segment_exposure", 0))
    values[f"{tag}.cells"] = str(len(table_rows))
    labels = [label for _, label in E10_POLICIES]
    labels.insert([k for k, _ in E10_POLICIES].index("bridge") + 1, "Rank fusion")
    fused = ""
    if reproduced:
        values[f"{tag}.rrf.reproduces"] = "exactly" if all(reproduced) else "not exactly"
        fused = f" Rank fusion was run at 4k tokens only (– elsewhere), in a second run over the same cases that reproduces the bridge column {values[f'{tag}.rrf.reproduces']}."
    table("long", ["Context", "Budget (tokens)", "No repair"] + labels, table_rows,
          f"**Table {{{{#E:long}}}}.** Long contexts on Qwen2.5-1.5B: one HotpotQA question's documents padded with other questions' documents to about 4k, 8k or 16k tokens; a supporting document placed after a quarter of the documents is deleted; the cached question sits right after it or at the end (12 questions × 2 placements per length). Teacher-forced KL in nats; mean over cases.{fused}")


E11_LABELS = [("full", "Whole new input"), ("first64", "First 64 tokens"), ("last16", "Last 16 tokens, alone"), ("last64", "Last 64 tokens, alone")]


def e11() -> None:
    data = load("e11_longq_qwen1.5b_main")
    if not data or not data["rows"]:
        return
    rows, tag = data["rows"], "e11.qwen1.5b"
    table_rows = {key: [label] for key, label in E11_LABELS}
    lengths = sorted({r["length"] for r in rows})
    for length in lengths:
        part = [r for r in rows if r["length"] == length]
        t, cids = f"{tag}.{length}", [question(r) for r in part]
        values[f"{t}.n"], values[f"{t}.new_tokens"] = str(len(part)), f"{statistics.fmean(r['nQ'] for r in part):,.0f}"
        values[f"{t}.stale"] = kl(statistics.fmean(r["results"]["stale"]["kl_tf"] for r in part))
        for key, _ in E11_LABELS:
            rho = statistics.fmean(r["agreement"][key]["spearman"] for r in part)
            is_q = sum(r["picked_is_question"][key] for r in part)
            se = statistics.fmean(r["results"][f"segment_exposure/{key}"]["kl_tf"] for r in part)
            br = statistics.fmean(r["results"][f"bridge/{key}"]["kl_tf"] for r in part)
            values[f"{t}.{key}.rho"], values[f"{t}.{key}.question"] = f"{rho:.2f}", f"{is_q} of {len(part)}"
            values[f"{t}.{key}.segment_exposure"], values[f"{t}.{key}.bridge"] = kl(se), kl(br)
            if key != "full":
                for policy in ("segment_exposure", "bridge"):
                    a = [r["results"][f"{policy}/{key}"]["kl_tf"] for r in part]
                    b = [r["results"][f"{policy}/full"]["kl_tf"] for r in part]
                    values[f"{t}.{key}.{policy}.p"] = pvalue(paired_test(a, b, cids))
                    values[f"{t}.{key}.{policy}.ratio"] = ratio_text(ratio_ci(a, b, cids))
            table_rows[key] += ["1" if key == "full" else f"{rho:.2f}", f"{is_q}/{len(part)}", kl(se), kl(br)]
    header = ["Reader attention taken from"]
    for length in lengths:
        header += [f"{length:,} tokens of notes ({values[f'{tag}.{length}.new_tokens']}-token input): rank corr.", "finds question", "reader-first KL", "bridge KL"]
    table("longq", header, list(table_rows.values()),
          f"**Table {{{{#E:longq}}}}.** A long new input (WikiText notes before the assistant opener) on Qwen2.5-1.5B, the cases of the main experiment with the edited document 5th of ten (36 cases per length). Reader attention is measured from the whole new input or from part of it; rank correlation with the whole-input estimate; how often the most-read-per-token segment is the question; teacher-forced KL at 10% after reader-first repair and after the bridge score, each driven by that estimate.")


E12_POLICIES = [("boundary", "Boundary window"), ("reader", "Reader attention"), ("exposure", "Recorded exposure"),
                ("bridge", "**Bridge score**"), ("rrf", "**Rank fusion (RRF)**"), ("segment_exposure", "Reader-first (segment found)"),
                ("oracle_reader", "Oracle score: true deviation × reader")]
E12_CHAINS = [("positional", "No repair"), ("repair_repair", "Repair 10% after each edit"),
              ("repair_repair_reader_first", "Reader-first repair 10% after each edit"),
              ("sweep_repair", "Sweep after the first edit, repair 10% after the second"),
              ("sweep_repair_reader_first", "Sweep after the first edit, reader-first repair after the second")]


def e12() -> None:
    data = load("e12_agent_qwen1.5b_main")
    if not data or not data["rows"]:
        return
    rows, tag = data["rows"], "e12.qwen1.5b"
    values[f"{tag}.n"], values[f"{tag}.tokens"] = str(len(rows)), f"{statistics.fmean(r['tokens'] for r in rows):,.0f}"
    for r in rows:
        for edit in ("clear1", "clear2"):
            res = r[edit]["results"]
            collect("agent", r["id"], res["stale"]["kl_tf"], {k: res[k]["0.1"]["kl_tf"] for k in ("boundary", "reader", "exposure", "bridge", "rrf", "segment_exposure")})
    budgets = [str(b) for b in data["budgets"]]
    table_rows = {key: [label] for key, label in [("stale", "No repair")] + E12_POLICIES}
    for edit in ("clear1", "clear2"):
        sub, t = [r[edit] for r in rows], f"{tag}.{edit}"
        values[f"{t}.suffix"] = f"{statistics.fmean(s['nS'] for s in sub):.0f}"
        picked = collections.Counter(s["picked"] for s in sub).most_common(1)[0]
        values[f"{t}.picked"], values[f"{t}.picked_n"] = picked[0], f"{picked[1]} of {len(sub)}"
        shares = collections.defaultdict(list)
        for s in sub:
            for name, share in s["segment_share"].items():
                shares[name].append(share)
        for name, xs in shares.items():
            values[f"{t}.share.{name}"] = pct(statistics.fmean(xs))
        stale = statistics.fmean(s["results"]["stale"]["kl_tf"] for s in sub)
        values[f"{t}.stale"] = kl(stale)
        table_rows["stale"] += [kl(stale)] + [""] * (len(budgets) - 1)
        for key, _ in E12_POLICIES:
            for budget in budgets:
                m = statistics.fmean(s["results"][key][budget]["kl_tf"] for s in sub)
                values[f"{t}.{key}.{budget}"] = kl(m)
                table_rows[key].append(kl(m))
        get = lambda s, key: s["results"][key]["0.1"]["kl_tf"]
        family = {}
        for a, b in (("rrf", "bridge"), ("bridge", "reader"), ("rrf", "reader"), ("rrf", "exposure"), ("bridge", "exposure"),
                     ("segment_exposure", "rrf"), ("rrf", "oracle_reader")):
            share, p = win_p([get(s, a) for s in sub], [get(s, b) for s in sub])
            values[f"{t}.{a}_vs.{b}.win"], values[f"{t}.{a}_vs.{b}.p"] = pct(share), pvalue(p)
            family[f"{a}_vs.{b}"] = p
        for name, p in holm(family).items():
            values[f"{t}.{name}.holm"] = pvalue(p)
        for budget in budgets:
            add_setting("agent", f"Agent trace, clear observation {edit[-1]}, {float(budget):.0%}", float(budget), stale,
                        {k: statistics.fmean(s["results"][k][budget]["kl_tf"] for s in sub) for k, _ in ROBUST_POLICIES if k in sub[0]["results"]})
        ranked = sorted(((statistics.fmean(get(s, k) for s in sub), k) for k, _ in E12_POLICIES if k != "oracle_reader"))
        values[f"{t}.best"] = {"exposure": "recorded exposure alone", "rrf": "rank fusion", "bridge": "the bridge score", "reader": "reader attention alone",
                               "segment_exposure": "reader-first repair", "boundary": "the boundary window"}[ranked[0][1]]
        values[f"{t}.best_kl"] = kl(ranked[0][0])
    header = ["Policy"] + [f"{which}, {float(b):.0%}" for which in ("Clear 1st observation", "Clear 2nd observation") for b in budgets]
    table("agent", header, list(table_rows.values()),
          f"**Table {{{{#E:agent}}}}.** Synthetic agent traces on Qwen2.5-1.5B ({len(rows)} traces, about {values[f'{tag}.tokens']} tokens): an old tool result is cleared to a one-line placeholder, as agents do to save room; the user's question precedes every tool result, so the stale readers are the agent's later thoughts and observations. Teacher-forced KL in nats from the re-prefill, mean over traces, by the share of stale tokens recomputed.")
    chain_rows = []
    for key, label in E12_CHAINS:
        steps = [[r["chained"][key][i]["kl"] for r in rows] for i in (0, 1)]
        for i, xs in enumerate(steps):
            values[f"{tag}.chain.{key}.{i + 1}"] = kl(statistics.fmean(xs))
            values[f"{tag}.chain.{key}.{i + 1}.median"] = kl(statistics.median(xs))
        stale_tokens = statistics.fmean(r["chained"][key][1]["stale_tokens"] for r in rows)
        chain_rows.append([label, values[f"{tag}.chain.{key}.1"], values[f"{tag}.chain.{key}.2"], f"{stale_tokens:.0f}"])
    a = [r["chained"]["repair_repair_reader_first"][1]["kl"] for r in rows]
    b = [r["chained"]["repair_repair"][1]["kl"] for r in rows]
    share, p = win_p(a, b)
    values[f"{tag}.chain.first_vs_plain.win"], values[f"{tag}.chain.first_vs_plain.p"] = pct(share), pvalue(p)
    table("chained", ["Runtime policy", "KL after clearing the 1st observation", "after clearing the 2nd", "Stale tokens left"], chain_rows,
          f"**Table {{{{#G:chained}}}}.** Two tool results cleared one after the other through the runtime of Section 6 (the traces of Table {{{{#E:agent}}}}). KL at the first token of the final answer from a cold pass over the edited trace; mean over {len(rows)} traces. Repairs use only what the runtime records.")


E21_ENVIRONMENTS = {"os": "a Linux shell", "db": "a MySQL database", "kg": "a knowledge-base API", "webshop": "WebShop", "alfworld": "ALFWorld"}
POLICY_NAMES = {"exposure": "recorded exposure alone", "rrf": "rank fusion", "bridge": "the bridge score", "reader": "reader attention alone",
                "segment_exposure": "reader-first repair", "boundary": "the boundary window", "tail": "the tail window"}


def e21() -> None:
    """Real agent transcripts (AgentInstruct): an old tool result cleared while the newest one arrives."""
    runs = {model: load(f"e21_traces_{KEYS[model]}_main") for model in NAMES}
    runs = {model: data for model, data in runs.items() if data and len(data["rows"]) >= 37}
    if not runs:
        return
    tails = {model: load(f"e21_traces_{KEYS[model]}_tail") for model in runs}
    tails = {model: {r["id"]: r for r in t["rows"]} for model, t in tails.items() if t and len(t["rows"]) == len(runs[model]["rows"])}
    if tails and len(tails) != len(runs):
        tails = {}
    extras = {model: load(f"e21_traces_{KEYS[model]}_extra") for model in runs}
    extras = {model: {r["id"]: r for r in t["rows"]} for model, t in extras.items() if t and len(t["rows"]) == len(runs[model]["rows"])}
    if extras and len(extras) != len(runs):
        extras = {}
    policies = (E12_POLICIES[:1] + ([("tail", "Tail window")] if tails else []) + E12_POLICIES[1:-1]
                + ([("latest_exposure", "Latest turn first, then exposure")] if extras else []) + E12_POLICIES[-1:])
    table_rows = {key: [label] for key, label in [("stale", "No repair")] + policies}
    header = ["Policy"]
    reproduced, reproduced_extra, raw = [], [], {}
    for model, data in runs.items():
        rows, tag = data["rows"], f"e21.{KEYS[model]}"
        if tails:
            for r in rows:
                for edit in ("clear1", "clear2"):
                    extra = tails[model][r["id"]][edit]["results"]
                    reproduced += [extra["bridge"][b]["kl_tf"] == r[edit]["results"]["bridge"][b]["kl_tf"] for b in extra["bridge"]]
                    r[edit]["results"]["tail"] = extra["tail"]
        if extras:
            for r in rows:
                for edit in ("clear1", "clear2"):
                    extra = extras[model][r["id"]][edit]["results"]
                    reproduced_extra.extend(extra["bridge"][b]["kl_tf"] == r[edit]["results"]["bridge"][b]["kl_tf"] for b in extra["bridge"])
                    r[edit]["results"]["latest_exposure"] = extra["latest_exposure"]
                    r[edit]["results"]["edit_effect"] = extra["edit_effect"]
        values[f"{tag}.n"], values[f"{tag}.tokens"] = str(len(rows)), f"{statistics.fmean(r['tokens'] for r in rows):,.0f}"
        for r in rows:
            for edit in ("clear1", "clear2"):
                res = r[edit]["results"]
                collect("traces", r["id"], res["stale"]["kl_tf"],
                        {k: res[k]["0.1"]["kl_tf"] for k in ("boundary", "tail", "reader", "exposure", "bridge", "rrf", "segment_exposure") if k in res})
        if KEYS[model] == "qwen1.5b" or "e21.tokens" not in values:
            values["e21.n"], values["e21.tokens"] = values[f"{tag}.n"], values[f"{tag}.tokens"]
        parts = [f"{sum(r['env'] == env for r in rows)} from {name}" for env, name in E21_ENVIRONMENTS.items()]
        values["e21.environments"] = ", ".join(parts[:-1]) + " and " + parts[-1]
        for edit in ("clear1", "clear2"):
            sub, t = [r[edit] for r in rows], f"{tag}.{edit}"
            header.append(f"{NAMES[model]}: clear {'1st' if edit == 'clear1' else '2nd'}")
            values[f"{t}.suffix"] = f"{statistics.fmean(s['nS'] for s in sub):,.0f}"
            values[f"{t}.cleared"] = f"{statistics.fmean(r['cleared_tokens']['observation' + edit[-1]] for r in rows):,.0f}"
            stale = [s["results"]["stale"]["kl_tf"] for s in sub]
            values[f"{t}.stale"], values[f"{t}.stale.median"] = kl(statistics.fmean(stale)), kl(statistics.median(stale))
            table_rows["stale"].append(values[f"{t}.stale"])
            for key, _ in policies:
                for budget in data["rows"][0][edit]["results"][key]:
                    mean = statistics.fmean(s["results"][key][budget]["kl_tf"] for s in sub)
                    values[f"{t}.{key}.{budget}"] = kl(mean)
                    raw[f"{t}.{key}.{budget}"] = mean
                table_rows[key].append(values[f"{t}.{key}.0.1"])
            get = lambda s, key: s["results"][key]["0.1"]["kl_tf"]
            for key in ("bridge", "rrf", "reader", "segment_exposure") + (("tail",) if tails else ()) + (("latest_exposure",) if extras else ()):
                values[f"{t}.left.{key}"] = pct(statistics.fmean(get(s, key) for s in sub) / statistics.fmean(stale))
                values[f"{t}.harm.{key}"] = f"{sum(get(s, key) > s['results']['stale']['kl_tf'] for s in sub)} of {len(sub)}"
            family = {}
            pairs = (("bridge", "reader"), ("rrf", "reader"), ("bridge", "exposure"), ("rrf", "exposure"), ("rrf", "bridge"), ("segment_exposure", "rrf"))
            for a, b in pairs:
                share, p = win_p([get(s, a) for s in sub], [get(s, b) for s in sub])
                values[f"{t}.{a}_vs.{b}.win"], values[f"{t}.{a}_vs.{b}.p"] = pct(share), pvalue(p)
                family[f"{a}_vs.{b}"] = p
            for name, p in holm(family).items():
                values[f"{t}.{name}.holm"] = pvalue(p)
            if extras:
                # A post-hoc control, so it gets its own test family, like the tail window.
                share, p = win_p([get(s, "segment_exposure") for s in sub], [get(s, "latest_exposure") for s in sub])
                values[f"{t}.segment_exposure_vs.latest_exposure.win"], values[f"{t}.segment_exposure_vs.latest_exposure.p"] = pct(share), pvalue(p)
                values[f"{t}.segment_exposure_vs.latest_exposure.holm"] = pvalue(p)
                effect = [s["results"]["edit_effect"]["kl_tf"] for s in sub]
                values[f"{t}.effect"], values[f"{t}.effect.median"] = kl(statistics.fmean(effect)), kl(statistics.median(effect))
            noise_median = values.get(f"e2c.{KEYS[model]}.noise.kl_tf.median")
            if noise_median:
                raw[f"{t}.stale_over_noise"] = statistics.median(stale) / float(noise_median)
            harmed = [get(s, "bridge") - s["results"]["stale"]["kl_tf"] for s in sub if get(s, "bridge") > s["results"]["stale"]["kl_tf"]]
            raw[f"{t}.harm.excess"] = statistics.median(harmed) if harmed else 0.0
            if extras:
                raw[f"{t}.effect.median"] = statistics.median(s["results"]["edit_effect"]["kl_tf"] for s in sub)
                raw[f"{t}.stale.median"] = statistics.median(stale)
            if tails:
                # Added after a first look at the results: its own family, so it does not change the correction of the six planned tests.
                share, p = win_p([get(s, "segment_exposure") for s in sub], [get(s, "tail") for s in sub])
                values[f"{t}.segment_exposure_vs.tail.win"], values[f"{t}.segment_exposure_vs.tail.p"] = pct(share), pvalue(p)
                values[f"{t}.segment_exposure_vs.tail.holm"] = pvalue(p)
            ranked = sorted((statistics.fmean(get(s, k) for s in sub), k) for k, _ in policies if k != "oracle_reader")
            values[f"{t}.best"], values[f"{t}.best_kl"] = POLICY_NAMES[ranked[0][1]], kl(ranked[0][0])
            by_env = {env: sum(s["results"]["stale"]["kl_tf"] for r, s in zip(rows, sub) if r["env"] == env) for env in E21_ENVIRONMENTS}
            top = max(by_env, key=by_env.get)
            values[f"{t}.top_env"] = {"os": "Linux-shell", "db": "database", "kg": "knowledge-base", "webshop": "WebShop", "alfworld": "ALFWorld"}[top]
            values[f"{t}.top_env.share"] = pct(by_env[top] / sum(by_env.values()))
            values[f"{t}.top_env.n"] = str(sum(r["env"] == top for r in rows))
            last = [s["picked"] == list(s["segment_share"])[-1] for s in sub]
            values[f"{t}.picked_last"] = f"{sum(last)} of {len(sub)}"
            values[f"{t}.picked_share"] = pct(statistics.fmean(s["segment_share"][s["picked"]] for s in sub))
            values[f"{t}.last_share"] = pct(statistics.fmean(list(s["segment_share"].values())[-1] for s in sub))
            for budget in data["rows"][0][edit]["results"]["bridge"]:
                add_setting("traces", f"Real transcripts, {NAMES[model]}, clear {edit[-1]}, {float(budget):.0%}", float(budget), statistics.fmean(stale),
                            {k: statistics.fmean(s["results"][k][budget]["kl_tf"] for s in sub) for k, _ in ROBUST_POLICIES if k in sub[0]["results"]})
    if reproduced:
        values["e21.tail.reproduces"] = "exactly" if all(reproduced) else "not exactly"
    if reproduced_extra:
        values["e21.extra.reproduces"] = "exactly" if all(reproduced_extra) else "not exactly"
    longest: dict[int, list[str]] = {}
    for m, data in runs.items():
        longest.setdefault(max(r["tokens"] for r in data["rows"]), []).append(NAMES[m])
    shortest, *others = sorted(longest)
    values["e21.max_tokens"] = f"{shortest:,} tokens" + "".join(f", and {n:,} in the tokenizer of {' and '.join(longest[n])}" for n in others)
    span = lambda xs: f"{kl(min(xs))} to {kl(max(xs))}"
    keys = [f"e21.{KEYS[model]}.{edit}" for model in runs for edit in ("clear1", "clear2")]
    values["e21.stale.range"] = span([float(values[f"{t}.stale"]) for t in keys])
    values["e21.stale.median.range"] = span([float(values[f"{t}.stale.median"]) for t in keys])
    noise = [float(values[f"e2c.{KEYS[m]}.noise.kl_tf"]) for m in NAMES if f"e2c.{KEYS[m]}.noise.kl_tf" in values]
    if noise:
        values["e2c.noise.range"] = span(noise)
        ratios = [raw[f"{t}.stale_over_noise"] for t in keys if f"{t}.stale_over_noise" in raw]
        values["e21.median_vs_noise"] = f"{min(ratios):.0%} to {max(ratios):.0%}"
    values["e21.scope"] = "Across the three small models and the two edits" if len(runs) == 3 else f"Across {' and '.join(NAMES[m] for m in runs)} and the two edits"

    count_word = {0: "none", 1: "one", 2: "two", 3: "all three"}
    by_edit = {edit: [f"e21.{KEYS[model]}.{edit}" for model in runs] for edit in ("clear1", "clear2")}
    last = [int(values[f"{t}.picked_last"].split()[0]) for t in keys]
    values["e21.picked_last.range"] = f"{min(last)} to {max(last)} of {values['e21.n']}"
    first_best = sum(values[f"{t}.best"] == "reader-first repair" for t in by_edit["clear1"])
    values["e21.clear1.first_best"] = f"on {count_word.get(first_best, str(first_best))} of the models" if first_best < 3 else "on all three models"
    left2 = [float(values[f"{t}.left.segment_exposure"].rstrip("%")) for t in by_edit["clear2"]]
    values["e21.clear2.first_left.range"] = f"{min(left2):.0f}% to {max(left2):.0f}%"
    values["e21.clear2.first_above_none"] = count_word.get(sum(x >= 100 for x in left2), str(sum(x >= 100 for x in left2)))
    shop_left, shop_share = [], []
    for model, data in runs.items():
        part = [r["clear2"] for r in data["rows"] if r["env"] == "webshop"]
        stale_shop = sum(x["results"]["stale"]["kl_tf"] for x in part)
        shop_left.append(sum(x["results"]["segment_exposure"]["0.1"]["kl_tf"] for x in part) / stale_shop)
        shop_share.append(stale_shop / sum(r["clear2"]["results"]["stale"]["kl_tf"] for r in data["rows"]))
    values["e21.clear2.shop_first.range"] = f"{min(shop_left):.0%} to {max(shop_left):.0%}"
    values["e21.clear2.shop_share.range"] = f"{min(shop_share):.0%} to {max(shop_share):.0%}"
    values["e21.clear2.first_verdict"] = "of little or no use" if min(left2) >= 90 else "less useful"
    sig_better_first = sum(raw[f"{t}.segment_exposure.0.1"] < raw[f"{t}.rrf.0.1"] and float(values[f"{t}.segment_exposure_vs.rrf.holm"]) < 0.05
                           for t in by_edit["clear1"])
    values["e21.clear1.first_vs_rrf"] = ("not significantly better than rank fusion on any model" if sig_better_first == 0
                                         else f"significantly better than rank fusion on {count_word.get(sig_better_first, str(sig_better_first))} of the models")
    values["e21.contribution"] = (f" On real agent transcripts, recomputing first the segment the new input reads most, usually the agent's latest turn, is "
                                  f"{values['e21.clear1.first_vs_rrf']} when the first tool result is cleared and {values['e21.clear2.first_verdict']} when the second is.")

    combos = [(f"e21.{KEYS[model]}.{edit}", data) for model, data in runs.items() for edit in ("clear1", "clear2")]
    for key in ("bridge", "rrf", "segment_exposure", "reader") + (("tail",) if tails else ()) + (("latest_exposure",) if extras else ()):
        left = [float(values[f"{t}.left.{key}"].rstrip("%")) for t, _ in combos]
        values[f"e21.left.{key}.range"] = f"{min(left):.0f}% to {max(left):.0f}%" if round(min(left)) != round(max(left)) else f"{min(left):.0f}%"
    best = collections.Counter(values[f"{t}.best"] for t, _ in combos).most_common()
    named = [f"{name} in {count}" for name, count in best]
    values["e21.best_counts"] = (", ".join(named[:-1]) + " and " + named[-1] if len(named) > 1 else named[0]) + f" of the {len(combos)} combinations of model and edit"
    for a, b in ((("bridge", "reader"), ("rrf", "reader"), ("rrf", "bridge"), ("segment_exposure", "rrf")) + ((("segment_exposure", "tail"),) if tails else ())
                 + ((("segment_exposure", "latest_exposure"),) if extras else ())):
        better = sum(raw[f"{t}.{a}.0.1"] < raw[f"{t}.{b}.0.1"] for t, _ in combos)
        significant = sum(raw[f"{t}.{a}.0.1"] < raw[f"{t}.{b}.0.1"] and float(values[f"{t}.{a}_vs.{b}.holm"]) < 0.05 for t, _ in combos)
        reverse = sum(raw[f"{t}.{a}.0.1"] > raw[f"{t}.{b}.0.1"] and float(values[f"{t}.{a}_vs.{b}.holm"]) < 0.05 for t, _ in combos)
        values[f"e21.{a}_vs.{b}.lower"] = f"{better} of {len(combos)}"
        values[f"e21.{a}_vs.{b}.significant"] = str(significant)
        values[f"e21.{a}_vs.{b}.reverse"] = str(reverse)
    excess = [raw[f"{t}.harm.excess"] for t, _ in combos]
    values["e21.harm.excess.range"] = f"{sci(min(excess))} to {sci(max(excess))}"
    harm = [int(values[f"{t}.harm.bridge"].split()[0]) for t, _ in combos]
    values["e21.harm.bridge.range"] = f"{min(harm)} to {max(harm)}" if min(harm) != max(harm) else str(min(harm))
    values["e21.harm_sentence"] = ""
    if "e2c.harm.bridge.min" in values:
        n_main = int(values.get("e2c.qwen1.5b.n", "108"))
        more = min(harm) / int(values["e21.n"]) > int(values["e2c.harm.bridge.max"]) / n_main
        values["e21.harm_sentence"] = (f" {'Repairs also make things worse more often here: t' if more else 'T'}he bridge score leaves a higher KL than no repair in "
                                       f"{values['e21.harm.bridge.range']} of the {values['e21.n']} transcripts, depending on model and edit, against "
                                       f"{values['e2c.harm.bridge.min']} to {values['e2c.harm.bridge.max']} of {n_main} cases in the main experiment, though by little "
                                       f"(a median excess of {values['e21.harm.excess.range']} nats among those transcripts).")
    values["e21.effect_sentence"], values["e21.latest_sentence"] = "", ""
    if extras:
        effect_medians = [raw[f"{t}.effect.median"] for t, _ in combos]
        values["e21.effect.median.range"] = span(effect_medians)
        ratio = [raw[f"{t}.stale.median"] / raw[f"{t}.effect.median"] for t, _ in combos if raw[f"{t}.effect.median"] > 0]
        values["e21.stale_vs_effect.range"] = f"{min(ratio):.0%} to {max(ratio):.0%}"
        values["e21.effect_sentence"] = (f" Clearing the result itself moves the continuation by a median of {values['e21.effect.median.range']} nats "
                                         "(the re-prefill of the edited transcript against a cold pass over the unedited one), and the stale state's median distance "
                                         f"from the re-prefill is {values['e21.stale_vs_effect.range']} of that: the cleared results matter little to the next turn by the time of the cut.")
        differ = [37 - int(values[f"{t}.picked_last"].split()[0]) for t, _ in combos]
        values["e21.latest.differ.range"] = f"{min(differ)} to {max(differ)}"
        lower, significant = values["e21.segment_exposure_vs.latest_exposure.lower"], values["e21.segment_exposure_vs.latest_exposure.significant"]
        values["e21.latest_sentence"] = (f" Recomputing the latest turn whole and then filling by exposure, which needs no reader pass, leaves {values['e21.left.latest_exposure.range']}; "
                                         f"reader-first repair has lower mean KL than this rule in {lower} combinations (Wilcoxon p < 0.05 in {significant}). "
                                         f"The two rules recompute the same segment first in most transcripts, so each test rests on the {values['e21.latest.differ.range']} "
                                         "transcripts per combination where they differ. This control was added after review, in a run over the same cases that reproduces "
                                         f"the bridge rows {values['e21.extra.reproduces']}; its tests, like the tail window's, are kept out of the six planned comparisons, "
                                         "a choice made after the first look at the results.")
    values["e21.tail_sentence"] = ""
    if tails:
        lower, significant = values["e21.segment_exposure_vs.tail.lower"], values["e21.segment_exposure_vs.tail.significant"]
        values["e21.tail_sentence"] = (f" A tail window of the same size, which needs no reader pass, leaves {values['e21.left.tail.range']}; "
                                       f"reader-first repair has lower mean KL than the tail window in {lower} combinations (Wilcoxon p < 0.05 in {significant}; "
                                       f"the tail window and this test were added after a first look at the results).")
    effect_note = ""
    if extras:
        table_rows["edit_effect"] = ["Effect of the clearing (unedited vs edited, both exact)"] + [values[f"e21.{KEYS[m]}.{e}.effect"] for m in runs for e in ("clear1", "clear2")]
        effect_note = " The last row is the edit's own effect, not a policy; it and the latest-turn rule come from a run added after review that reproduces the bridge rows " + values["e21.extra.reproduces"] + "."
    tail_note = (f" The tail window (the stale tokens right before the new input) was added after a partial look at the results, in a second run over the same cases that reproduces the bridge rows {values['e21.tail.reproduces']}."
                 if tails else "")
    table("traces", header, list(table_rows.values()),
          f"**Table {{{{#E:traces}}}}.** Real agent transcripts: {values['e21.n']} AgentInstruct trajectories [@zeng2023agenttuning] of about {values['e21.tokens']} tokens in the Qwen2.5 tokenizer, from five environments ({values['e21.environments']}). The task's first or second tool result is cleared to a one-line placeholder while the newest tool result arrives as the new input. Teacher-forced KL in nats from the re-prefill over the first 16 tokens of the re-prefill's own greedy continuation (the transcripts' recorded turns were written by another model), mean over trajectories, 10% of the stale tokens recomputed.{tail_note}{effect_note}")


E2E_POLICIES = [("stale", "No repair"), ("boundary", "Boundary window"), ("reader", "Reader attention"),
                ("bridge_qs4g", "Kept-query exposure × reader"), ("segment_reader", "**Reader-first (segment found), then reader attention**"),
                ("segment_qs4g", "Reader-first (segment found), then kept-query bridge"), ("question_reader", "Cached question (given), then reader attention"),
                ("bridge_true", "True exposure × reader (not available in practice)")]


def e2e() -> None:
    table_rows, models_done = {key: [label] for key, label in E2E_POLICIES}, []
    for model in NAMES:
        data = load(f"e2e_insert_reader_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e2e.{KEYS[model]}"
        models_done.append(NAMES[model])
        cids = [question(r) for r in rows]
        get = lambda r, key: r["results"][key]["kl_tf"]
        values[f"{tag}.n"], values[f"{tag}.found"] = str(len(rows)), f"{sum(r['picked_is_question'] for r in rows)} of {len(rows)}"
        for key, _ in E2E_POLICIES:
            ci = mean_ci([get(r, key) for r in rows], cids)
            values[f"{tag}.{key}"] = kl(ci[0])
            table_rows[key].append(interval(ci))
        for a, b in (("segment_reader", "bridge_qs4g"), ("segment_qs4g", "bridge_qs4g"), ("segment_reader", "reader"),
                     ("bridge_qs4g", "reader"), ("segment_reader", "segment_qs4g"), ("segment_reader", "bridge_true")):
            share, p = win_p([get(r, a) for r in rows], [get(r, b) for r in rows], cids)
            values[f"{tag}.{a}_vs.{b}.win"], values[f"{tag}.{a}_vs.{b}.p"] = pct(share), pvalue(p)
    lower = [float(values[f"e2e.{KEYS[m]}.segment_reader"]) < float(values[f"e2e.{KEYS[m]}.bridge_qs4g"]) for m in NAMES if f"e2e.{KEYS[m]}.segment_reader" in values]
    significant = [lo and float(values[f"e2e.{KEYS[m]}.segment_reader_vs.bridge_qs4g.p"]) < 0.05
                   for lo, m in zip(lower, [m for m in NAMES if f"e2e.{KEYS[m]}.segment_reader" in values])]
    words = {0: "neither", 1: "one", 2: "both"} if len(lower) == 2 else {i: str(i) for i in range(len(lower) + 1)}
    values["e2e.first_lower"], values["e2e.first_significant"] = words[sum(lower)], words[sum(significant)]
    if models_done:
        table("insert_reader", ["Policy"] + models_done, list(table_rows.values()),
              "**Table {{#E:insert_reader}}.** Inserts without an exposure record: the cases of Table {{#main_strata}} read the other way round (the supporting document arrives late), 10% of the suffix recomputed. Teacher-forced KL in nats from the re-prefill; mean and 95% bootstrap interval over questions.")


def e3_fresh() -> None:
    data = load("e3_needle_qwen1.5b_fresh")
    if not data or not data["rows"]:
        return
    rows, tag = data["rows"], "e3f.qwen1.5b"
    for size in sorted({r["size"] for r in rows}):
        part = [r for r in rows if r["size"] == size]
        values[f"{tag}.{size}.n"] = str(len(part))
        for label in part[0]["out"]:
            values[f"{tag}.{size}.{label}.exact"] = f"{sum(bool(r['out'][label]['exact']) for r in part)} of {len(part)}"
            values[f"{tag}.{size}.{label}.ghost"] = f"{sum(bool(r['out'][label]['ghost']) for r in part)} of {len(part)}"
            if "kl" in part[0]["out"][label]:
                values[f"{tag}.{size}.{label}.kl"] = kl(statistics.fmean(r["out"][label]["kl"] for r in part))
        values[f"{tag}.{size}.picked_retained"] = f"{sum(r.get('picked_segment') == 'retained' for r in part)} of {len(part)}"


def needle_settings() -> None:
    """Needle task at 2% of the suffix, first-token KL, for the robustness summary (main run plus the reader-first run)."""
    main, reader = load("e3_needle_qwen1.5b_main"), load("e3_needle_qwen1.5b_reader")
    if not main or not reader:
        return
    first = {r["name"]: r for r in reader["rows"]}
    rrf_run = load("e3_needle_qwen1.5b_rrf")
    fused = {r["name"]: r for r in rrf_run["rows"]} if rrf_run else {}
    for size in sorted({r["size"] for r in main["rows"]}):
        part = [r for r in main["rows"] if r["size"] == size and r["name"] in first]
        stale = statistics.fmean(r["out"]["no_repair"]["kl"] for r in part)
        for budget in ("0.02", "0.05"):
            source = {"boundary": first if budget == "0.02" else None, "tail": first if budget == "0.02" else None, "segment_exposure": first}
            means = {}
            for key in ("boundary", "tail", "reader", "exposure", "bridge", "segment_exposure"):
                rows_of = source.get(key)
                label = f"{key}@{budget}"
                if rows_of is None and label in part[0]["out"]:
                    means[key] = statistics.fmean(r["out"][label]["kl"] for r in part)
                elif rows_of is not None and label in rows_of[part[0]["name"]]["out"]:
                    means[key] = statistics.fmean(rows_of[r["name"]]["out"][label]["kl"] for r in part)
            if fused and all(r["name"] in fused for r in part):
                means["rrf"] = statistics.fmean(fused[r["name"]]["out"][f"rrf@{budget}"]["kl"] for r in part)
                values[f"e3.qwen1.5b.{size}.rrf@{budget}.k"] = f"{sum(bool(fused[r['name']]['out'][f'rrf@{budget}']['exact']) for r in part)} of {len(part)}"
                values[f"e3.qwen1.5b.{size}.rrf@{budget}.kl"] = kl(means["rrf"])
            add_setting("needle", f"Needle, {size} tokens, {float(budget):.0%}", float(budget), stale, means)
            if budget == "0.05":
                for r in part:
                    kls = {key: r["out"][f"{key}@0.05"]["kl"] for key in ("boundary", "tail", "reader", "exposure", "bridge")}
                    kls["segment_exposure"] = first[r["name"]]["out"]["segment_exposure@0.05"]["kl"]
                    if fused:
                        kls["rrf"] = fused[r["name"]]["out"]["rrf@0.05"]["kl"]
                    collect("needle", r["name"], r["out"]["no_repair"]["kl"], kls)
        for budget in ("0.02", "0.05"):
            a = [float(first[r["name"]]["out"][f"segment_exposure@{budget}"]["exact"]) for r in part]
            b = [float(r["out"][f"bridge@{budget}"]["exact"]) for r in part]
            values[f"e3r.qwen1.5b.{size}.segment_vs_bridge.{budget}.p"] = pvalue(mcnemar([x > 0 for x in a], [y > 0 for y in b]))


GAP_ROWS = (1, 4, 16, 64, 256, 1024)


def spearman_or_nan(xs: list[float], ys: list[float]) -> float:
    from scipy.stats import spearmanr

    value = spearmanr(xs, ys).statistic
    return value if value == value else 0.0


def e15() -> None:
    header, columns = ["Gap (positions)"], []
    for model in NAMES:
        data = load(f"e15_gap_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e15.{KEYS[model]}"
        values[f"{tag}.n"] = str(len(rows))
        column = []
        for g in data["gaps"]:
            kls = [r["results"][str(g)]["kl"] for r in rows]
            same = sum(r["results"][str(g)]["same_answer"] for r in rows)
            values[f"{tag}.{g}.kl"], values[f"{tag}.{g}.median"] = kl(statistics.fmean(kls)), kl(statistics.median(kls))
            values[f"{tag}.{g}.same"] = f"{same} of {len(rows)}"
            column.append((kl(statistics.fmean(kls)), kl(statistics.median(kls)), f"{same}/{len(rows)}"))
        header += [f"{NAMES[model]}: KL mean (median)", "same answer"]
        columns.append(column)
    if not columns:
        return
    changed = {}
    for model in NAMES:
        data = load(f"e15_gap_{KEYS[model]}_main")
        if data and data["rows"]:
            for g in data["gaps"]:
                changed.setdefault(g, []).append(1 - sum(r["results"][str(g)]["same_answer"] for r in data["rows"]) / len(data["rows"]))
            values[f"e15.{KEYS[model]}.rho"] = f"{statistics.fmean(spearman_or_nan([r['results'][str(g)]['kl'] for g in data['gaps']], data['gaps']) for r in data['rows']):.2f}"
    values["e15.max_changed"] = pct(max(changed[max(changed)]))
    values["e15.one.max"] = kl(max(float(values[f"e15.{KEYS[m]}.1.kl"]) for m in NAMES if f"e15.{KEYS[m]}.1.kl" in values))
    for g in (64, 256, 1024):
        ks = [float(values[f"e15.{KEYS[m]}.{g}.kl"]) for m in NAMES if f"e15.{KEYS[m]}.{g}.kl" in values]
        values[f"e15.{g}.min"], values[f"e15.{g}.max"] = kl(min(ks)), kl(max(ks))
    rows_out = []
    for index, g in enumerate(GAP_ROWS):
        line = [str(g)]
        for column in columns:
            mean, median, same = column[index]
            line += [f"{mean} ({median})", same]
        rows_out.append(line)
    table("gap", header, rows_out,
          "**Table {{#gap}}.** Position gaps alone. Every cell is computed exactly for the context without any deletion, but all positions from the start of one document onward are moved up by the gap (40 cases per model; the document is a distractor in 20 and a supporting document in 20; the question follows all documents). KL at the first answer token from the contiguous cold pass, and how many 16-token greedy answers are unchanged.")


def e14() -> None:
    data = load("e14_kernels_main")
    if not data:
        return
    rel, exp = data["relocation"], data["exposure"]
    values["e14.device"] = data["device"]
    ratio = lambda r, k: r[k] / r["plain"]
    gather = [ratio(r, "gather") for r in rel]
    values["e14.gather.min"], values["e14.gather.max"] = f"{min(gather):.2f}", f"{max(gather):.2f}"
    d32 = [r for r in rel if r["D"] == 32]
    values["e14.select.d32.min"] = f"{min(ratio(r, 'select') for r in d32):.0f}"
    values["e14.select.d32.max"] = f"{max(ratio(r, 'select') for r in d32):.0f}"
    values["e14.rewrite.min"] = f"{min(ratio(r, 'rewrite') for r in rel):.2f}"
    values["e14.rewrite.max"] = f"{max(ratio(r, 'rewrite') for r in rel):.2f}"
    values["e14.error.max"] = sci(max(r["max_error"] for r in rel))
    frag = []
    for r in rel:
        if r["layout"] == "runs of 8":
            base = next(x for x in rel if (x["n"], x["m"], x["D"], x["layout"]) == (r["n"], r["m"], r["D"], "contiguous"))
            frag.append(r["gather"] / base["gather"])
    values["e14.fragmentation.max"] = f"{max(frag):.2f}"
    rows_out = []
    for n in sorted({r["n"] for r in rel}):
        for m in sorted({r["m"] for r in rel}):
            for D in (1, 2, 8, 32):
                r = next(x for x in rel if (x["n"], x["m"], x["D"], x["layout"]) == (n, m, D, "contiguous"))
                frag8 = next((x for x in rel if (x["n"], x["m"], x["D"], x["layout"]) == (n, m, D, "runs of 8")), None)
                rows_out.append([f"{n:,}", str(m), str(D), f"{1000 * r['plain']:.1f}", f"{ratio(r, 'rewrite'):.2f}", f"{ratio(r, 'gather'):.2f}",
                                 f"{frag8['gather'] / r['plain']:.2f}" if frag8 else "–", f"{ratio(r, 'select'):.1f}"])
    table("kernels", ["Cached keys", "New queries", "Offsets D", "Plain (ms)", "Rewrite keys", "Group by offset", "… runs of 8", "Select per key"], rows_out,
          f"**Table {{{{#A:kernels}}}}.** One attention call with Qwen2.5-1.5B's head shapes in plain PyTorch on the laptop GPU ({data['device']}); time relative to plain attention over keys already at their positions. Rewrite: rotate every cached key, then attend. Group by offset: rotate the query once per offset and score each offset's keys (contiguous groups, or scattered in runs of 8 keys). Select per key: score every key with each rotated query and keep its own offset's score. Largest logit error against plain attention {values['e14.error.max']}.")
    over = [r["with"] / r["without"] - 1 for r in exp]
    values["e14.exposure.min"], values["e14.exposure.max"] = pct(min(over)), pct(max(over))
    values["e14.exposure.error"] = sci(max(r["max_error"] for r in exp))
    values["e14.exposure.segments"] = " and ".join(str(s) for s in sorted({r["segments"] for r in exp}))


def e1p() -> None:
    rows_out = []
    for model in NAMES:
        data = load(f"e1p_paired_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e1p.{KEYS[model]}"
        values[f"{tag}.n"] = str(len(rows))
        fresh, cached, effect = ([r[k]["kl"] for r in rows] for k in ("fresh", "cached", "deletion"))
        f_ci, c_ci, e_ci = mean_ci(fresh), mean_ci(cached), mean_ci(effect)
        values[f"{tag}.fresh"], values[f"{tag}.cached"], values[f"{tag}.effect"] = kl(f_ci[0]), kl(c_ci[0]), kl(e_ci[0])
        values[f"{tag}.fresh.median"], values[f"{tag}.cached.median"] = kl(statistics.median(fresh)), kl(statistics.median(cached))
        share, p = win_p(fresh, cached)
        values[f"{tag}.fresh_lower"], values[f"{tag}.p"] = pct(share), pvalue(p)
        values[f"{tag}.ratio"] = f"{statistics.fmean(cached) / statistics.fmean(fresh):.1f}"
        values[f"{tag}.fresh_of_effect"] = pct(statistics.fmean(fresh) / statistics.fmean(effect))
        values[f"{tag}.cached_of_effect"] = pct(statistics.fmean(cached) / statistics.fmean(effect))
        same_f, same_c = [r["fresh"]["same_answer"] for r in rows], [r["cached"]["same_answer"] for r in rows]
        values[f"{tag}.same.fresh"], values[f"{tag}.same.cached"] = f"{sum(same_f)} of {len(rows)}", f"{sum(same_c)} of {len(rows)}"
        values[f"{tag}.same.p"] = pvalue(mcnemar(same_f, same_c))
        values[f"{tag}.consistency"] = sci(max(r["reprefill_consistency"] for r in rows))
        values[f"{tag}.questions"] = str(len({r["name"].split("/")[1] for r in rows}))
        values[f"{tag}.cached_beyond_effect"] = f"{sum(r['cached']['kl'] > r['deletion']['kl'] for r in rows)} of {len(rows)}"
        rows_out.append([NAMES[model], str(len(rows)), interval(e_ci), interval(f_ci), interval(c_ci), f"{pct(share)}; {pvalue(p)}",
                         f"{sum(same_f)}/{len(rows)}", f"{sum(same_c)}/{len(rows)}"])
    done = [KEYS[m] for m in NAMES if f"e1p.{KEYS[m]}.n" in values]
    if done:
        ratios = [float(values[f"e1p.{k}.ratio"]) for k in done]
        values["e1p.ratio_min"], values["e1p.ratio_max"] = f"{min(ratios):.1f}", f"{max(ratios):.1f}"
        values["e1p.p_max"] = max((values[f"e1p.{k}.p"] for k in done), key=float)
        lowers = [values[f"e1p.{k}.fresh_lower"] for k in done]
        values["e1p.fresh_lower_min"], values["e1p.fresh_lower_max"] = min(lowers, key=lambda x: float(x.rstrip("%"))), max(lowers, key=lambda x: float(x.rstrip("%")))
        values["e1p.models"] = str(len(done))
    if rows_out:
        table("paired_reader", ["Model", "Cases", "Effect of the deletion", "Reuse, question fresh", "Reuse, question cached",
                                "Fresh lower; Wilcoxon p", "Same answer, fresh", "Same answer, cached"], rows_out,
              "**Table {{#paired_reader}}.** The same cases read two ways after deleting a supporting document: the question either arrives after the edit (fresh) or was cached before it. KL in nats at the first answer token from the re-prefill, mean and 95% bootstrap interval; the effect of the deletion is the re-prefill against the unedited context and is shared by both readings. Answers: 16-token greedy replies identical to the re-prefill's.")


def earlier_question_ids(exclude: str = "e16_") -> set[str]:
    """HotpotQA ids that appear in the result files of any experiment other than the one whose files start with `exclude`."""
    ids: set[str] = set()
    for path in RESULTS.glob("*.json"):
        if not path.name.startswith(exclude):
            ids.update(re.findall(r"\b5a[0-9a-f]{22}\b", path.read_text()))
    return ids


def verdict_text(verdicts: dict[str, str]) -> str:
    """'holds on A and B; not confirmed on C' from {model name: verdict}, in model order."""
    groups: dict[str, list[str]] = {}
    for name, verdict in verdicts.items():
        groups.setdefault(verdict, []).append(name)
    parts = []
    for i, (verdict, names) in enumerate(groups.items()):
        joined = "all three models" if len(names) == 3 else (" or " if verdict.startswith("not") else " and ").join(names)
        parts.append(f"{'is ' if i and verdict.startswith('not') else ''}{verdict} on {joined}")
    missing = [NAMES[m] for m in NAMES if NAMES[m] not in verdicts]
    if missing:
        parts.append(f"{' and '.join(missing)} still running")
    if len(parts) == 2:
        return " but ".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 2 else parts[0]


def e16() -> None:
    earlier = earlier_question_ids()
    plain_rows, hypothesis_rows = [], []
    verdicts = {f"h{i}": {} for i in range(1, 5)}
    for model in NAMES:
        tag, name = f"e16.{KEYS[model]}", NAMES[model]
        plain = load(f"e16_confirm_{KEYS[model]}_plain_main")
        row_out = {}
        if plain and len(plain["rows"]) == 108:
            rows = plain["rows"]
            cids = [question(r) for r in rows]
            tf = lambda r, key, budget="0.1": statistics.fmean(r["results"][key][budget]["kl"] if budget else r["results"][key]["kl"])
            nb = lambda key: None if key in ("stale", "question_only", "bridge@question", "rrf@question") else "0.1"
            get = lambda r, key: tf(r, key, nb(key))
            values[f"{tag}.n"], values[f"{tag}.questions"] = str(len(rows)), str(len(set(cids)))
            values[f"{tag}.reply_mean"] = f"{statistics.fmean(r['reply_tokens'] for r in rows):.1f}"
            for r in rows:
                collect("replication", question(r), get(r, "stale"), {"reader": get(r, "reader"), "exposure": get(r, "exposure"), "bridge": get(r, "bridge"),
                                                                      "rrf": get(r, "rrf"), "segment_exposure": get(r, "density_exposure"),
                                                                      "question_exposure": get(r, "question_exposure")})
            reused = sorted(set(cids) & earlier)
            values[f"{tag}.reused"] = f"{len(reused)} of {len(set(cids))}"
            spelled = {0: "None", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(len(reused), str(len(reused)))
            values[f"{tag}.reused_start"] = f"{spelled} of the {len(set(cids))}"
            clean = [r for r in rows if question(r) not in earlier]
            for key in ("stale", "reader", "exposure", "bridge", "rrf", "density_exposure", "oracle_reader", "question_only", "bridge@question", "rrf@question"):
                values[f"{tag}.{key}"] = kl(statistics.fmean(get(r, key) for r in rows))

            def h1_test(subset: list[dict]) -> tuple[str, dict]:
                ids = [question(r) for r in subset]
                raw = {a: paired_test([get(r, a) for r in subset], [get(r, "reader") for r in subset], ids) for a in ("rrf", "bridge")}
                adjusted = holm(raw)
                better = {a: statistics.fmean(get(r, a) for r in subset) < statistics.fmean(get(r, "reader") for r in subset) for a in raw}
                if all(better[a] and adjusted[a] < 0.05 for a in raw):
                    return "holds", adjusted
                if any(not better[a] and adjusted[a] < 0.05 for a in raw):
                    return "fails", adjusted
                return "not confirmed", adjusted

            def h2_test(subset: list[dict]) -> tuple[str, dict]:
                out = {}
                for stratum in ("short", "long"):
                    part = [r for r in subset if lead_stratum(r) == stratum]
                    a, b = [get(r, "density_exposure") for r in part], [get(r, "rrf") for r in part]
                    first_lower = statistics.fmean(a) < statistics.fmean(b)
                    out[stratum] = {"first": statistics.fmean(a), "rrf": statistics.fmean(b), "p": paired_test(a, b, [question(r) for r in part]),
                                    "predicted": first_lower if stratum == "short" else not first_lower,
                                    "win": sum(x < y for x, y in zip(a, b)) / len(a)}
                if all(o["predicted"] and o["p"] < 0.05 for o in out.values()):
                    return "holds", out
                if any(not o["predicted"] and o["p"] < 0.05 for o in out.values()):
                    return "fails", out
                return "not confirmed", out

            v1, adjusted = h1_test(rows)
            for a in ("rrf", "bridge"):
                values[f"{tag}.h1.{a}.win"] = pct(sum(get(r, a) < get(r, "reader") for r in rows) / len(rows))
                values[f"{tag}.h1.{a}.p"] = pvalue(adjusted[a])
            v2, strata = h2_test(rows)
            short = [r for r in rows if lead_stratum(r) == "short"]
            short_means = {key: statistics.fmean(get(r, key) for r in short) for key in ("reader", "exposure", "bridge", "rrf", "density_exposure")}
            values[f"{tag}.short.first_lowest"] = "yes" if min(short_means, key=short_means.get) == "density_exposure" else "no"
            for key, m in short_means.items():
                values[f"{tag}.short.{key}"] = kl(m)
            best_key = min(short_means, key=short_means.get)
            values[f"{tag}.short.best_other"] = ("" if best_key == "density_exposure" else
                f"on {name} {POLICY_NAMES.get(best_key, best_key)} is lower, {kl(short_means[best_key])} against {kl(short_means['density_exposure'])}")
            values[f"{tag}.short.first_below_rrf"] = "yes" if short_means["density_exposure"] < short_means["rrf"] else "no"
            for stratum, o in strata.items():
                values[f"{tag}.h2.{stratum}.first"], values[f"{tag}.h2.{stratum}.rrf"] = kl(o["first"]), kl(o["rrf"])
                values[f"{tag}.h2.{stratum}.win"], values[f"{tag}.h2.{stratum}.p"] = pct(o["win"]), pvalue(o["p"])
            values[f"{tag}.h1"], values[f"{tag}.h2"] = v1, v2
            verdicts["h1"][name], verdicts["h2"][name] = v1, v2
            c1, cadj = h1_test(clean)
            c2, cstrata = h2_test(clean)
            values[f"{tag}.clean.n"] = str(len(clean))
            values[f"{tag}.clean.h1"], values[f"{tag}.clean.h2"] = c1, c2
            values[f"{tag}.clean.h2.short.p"], values[f"{tag}.clean.h2.long.p"] = pvalue(cstrata["short"]["p"]), pvalue(cstrata["long"]["p"])
            row_out.update({"h1": f"{v1} (RRF {pvalue(adjusted['rrf'])}, bridge {pvalue(adjusted['bridge'])})",
                            "h2": f"{v2} (short {pvalue(strata['short']['p'])}, long {pvalue(strata['long']['p'])})",
                            "clean": f"{len(clean)} cases: H1 {c1}; H2 {c2}"})

            for stratum in ("short", "long"):
                part = [r for r in rows if lead_stratum(r) == stratum]
                for budget in ("0.05", "0.1", "0.2"):
                    means = {k: statistics.fmean(tf(r, k if k != "segment_exposure" else "density_exposure", budget) for r in part)
                             for k, _ in ROBUST_POLICIES if k not in ("boundary", "tail")}
                    add_setting("replication", f"Replication, {name}, {stratum} suffix, {float(budget):.0%}", float(budget),
                                statistics.fmean(get(r, "stale") for r in part), means)
            for key in ("bridge", "rrf", "density_exposure"):
                values[f"{tag}.ratio_ideal.{key}"] = ratio_text(ratio_ci([get(r, key) for r in rows], [get(r, "oracle_reader") for r in rows], cids))
            for budget in ("t16", "t32", "t64"):
                for key in ("reader", "bridge", "rrf", "density_exposure"):
                    values[f"{tag}.{budget}.{key}"] = kl(statistics.fmean(tf(r, key, budget) for r in rows))
            stale_mean = statistics.fmean(get(r, "stale") for r in rows)
            for key in ("question_only", "bridge@question", "rrf@question"):
                values[f"{tag}.removes.{key}"] = pct(1 - statistics.fmean(get(r, key) for r in rows) / stale_mean)
            values[f"{tag}.ratio.bridgeq_qonly"] = ratio_text(ratio_ci([get(r, "bridge@question") for r in rows], [get(r, "question_only") for r in rows], cids))
            later = lambda r, key: statistics.fmean((r["results"][key][nb(key)] if nb(key) else r["results"][key])["kl"][1:])
            longer = [r for r in rows if r["reply_tokens"] > 1]
            values[f"{tag}.later.n"] = str(len(longer))
            for key in ("stale", "reader", "bridge", "rrf", "density_exposure"):
                values[f"{tag}.later.{key}"] = kl(statistics.fmean(later(r, key) for r in longer))
            for rule in ("density", "shortest", "mass"):
                values[f"{tag}.plain.{rule}_finds"] = f"{sum(r['picks_question'][rule] for r in rows)} of {len(rows)}"
            leaks = [r["leak"] for r in rows if "leak" in r]
            leak_ids = [question(r) for r in rows if "leak" in r]
            if leaks:
                values[f"{tag}.leak.present_gap"] = f"{statistics.fmean(x['present'] - x['truth'] for x in leaks):.2f}"
                excess = {}
                for key in ("stale", "reader", "bridge", "rrf", "density_exposure"):
                    ci = mean_ci([x[key] - x["truth"] for x in leaks], leak_ids)
                    excess[key] = ci
                    values[f"{tag}.leak.{key}"] = f"{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]".replace("-", "−")
                    if key == "stale":
                        values[f"{tag}.leak.stale.below_zero"] = "yes" if ci[2] < 0 else "no"
                # The H4 rule was fixed with percentile intervals; the verdict is also checked with the BCa intervals the tables print.
                fixed = mean_ci([x["stale"] - x["truth"] for x in leaks], leak_ids, method="percentile")
                v4 = "holds" if fixed[1] > 0 and excess["bridge"][0] < excess["stale"][0] else "fails"
                v4_bca = "holds" if excess["stale"][1] > 0 and excess["bridge"][0] < excess["stale"][0] else "fails"
                values[f"{tag}.h4.same_under_bca"] = "yes" if v4 == v4_bca else "no"
                values[f"{tag}.h4"] = v4
                verdicts["h4"][name] = v4
                row_out["h4"] = f"{v4} (no repair {values[f'{tag}.leak.stale']})"
            line = [name] + [values[f"{tag}.{k}"] for k in ("stale", "reader", "exposure", "bridge", "rrf", "density_exposure", "oracle_reader")]
            line += [f"{values[f'{tag}.h2.short.first']} vs {values[f'{tag}.h2.short.rrf']} (p {values[f'{tag}.h2.short.p']})",
                     f"{values[f'{tag}.h2.long.first']} vs {values[f'{tag}.h2.long.rrf']} (p {values[f'{tag}.h2.long.p']})"]
            plain_rows.append(line)
        decoy = load(f"e16_confirm_{KEYS[model]}_decoy_main")
        if decoy and len(decoy["rows"]) == 108:
            rows = decoy["rows"]
            cids = [question(r) for r in rows]
            get = lambda r, key: statistics.fmean(r["results"][key]["0.1"]["kl"] if key not in ("stale", "question_only") else r["results"][key]["kl"])
            finds = {rule: sum(r["picks_question"][rule] for r in rows) for rule in ("density", "shortest", "mass")}
            for rule, k in finds.items():
                values[f"{tag}.decoy.{rule}_finds"] = f"{k} of {len(rows)}"
                values[f"{tag}.decoy.{rule}_finds_share"] = pct(k / len(rows))
            values[f"{tag}.decoy.density_picks_last_note"] = f"{sum(r['picks']['density'] == len(r['segment_lengths']) - 1 for r in rows)} of {len(rows)}"
            for key in ("stale", "reader", "bridge", "rrf", "question_exposure", "density_exposure", "shortest_exposure", "mass_exposure", "oracle_reader"):
                values[f"{tag}.decoy.{key}"] = kl(statistics.fmean(get(r, key) for r in rows))
            tests = {}
            for a, b in (("density_exposure", "shortest_exposure"), ("rrf", "density_exposure"), ("question_exposure", "density_exposure"), ("rrf", "bridge")):
                share, p = win_p([get(r, a) for r in rows], [get(r, b) for r in rows], cids)
                values[f"{tag}.decoy.{a}_vs.{b}.win"], values[f"{tag}.decoy.{a}_vs.{b}.p"] = pct(share), pvalue(p)
                tests[(a, b)] = (share, p)
            first_lower = statistics.fmean(get(r, "density_exposure") for r in rows) < statistics.fmean(get(r, "shortest_exposure") for r in rows)
            if finds["density"] <= len(rows) / 2:
                v3 = "fails"
            elif finds["shortest"] < len(rows) / 2 and first_lower and tests[("density_exposure", "shortest_exposure")][1] < 0.05:
                v3 = "holds"
            else:
                v3 = "not confirmed"
            values[f"{tag}.h3"] = v3
            verdicts["h3"][name] = v3
            row_out["h3"] = f"{v3} (attention finds the question in {finds['density']} of {len(rows)})"
            add_setting("replication", f"Replication with notes, {name}, 10%", 0.1, statistics.fmean(get(r, "stale") for r in rows),
                        {k: statistics.fmean(get(r, k if k != "segment_exposure" else "density_exposure") for r in rows)
                         for k in ("reader", "bridge", "rrf", "segment_exposure")})
        long_run = load(f"e16_confirm_{KEYS[model]}_long_main")
        if long_run and len(long_run["rows"]) == 108:
            rows = long_run["rows"]
            cids = [question(r) for r in rows]
            values[f"{tag}.long.reply_mean"] = f"{statistics.fmean(r['reply_tokens'] for r in rows):.1f}"
            pick = lambda r, key: r["results"][key]["0.1"] if key not in ("stale", "question_only") else r["results"][key]
            for key in ("stale", "reader", "exposure", "bridge", "rrf", "density_exposure", "oracle_reader", "question_only"):
                values[f"{tag}.long.{key}"] = kl(statistics.fmean(statistics.fmean(pick(r, key)["kl"]) for r in rows))
                values[f"{tag}.long.{key}.after4"] = kl(statistics.fmean(statistics.fmean(pick(r, key)["kl"][4:]) for r in rows if len(pick(r, key)["kl"]) > 4))
            a = [statistics.fmean(pick(r, "bridge")["kl"]) for r in rows]
            b = [statistics.fmean(pick(r, "reader")["kl"]) for r in rows]
            share, p = win_p(a, b, cids)
            values[f"{tag}.long.bridge_vs.reader.win"], values[f"{tag}.long.bridge_vs.reader.p"] = pct(share), pvalue(p)
            add_setting("replication", f"Replication with long replies, {name}, 10%", 0.1,
                        statistics.fmean(statistics.fmean(pick(r, "stale")["kl"]) for r in rows),
                        {k: statistics.fmean(statistics.fmean(pick(r, k if k != "segment_exposure" else "density_exposure")["kl"]) for r in rows)
                         for k, _ in ROBUST_POLICIES if k not in ("boundary", "tail")})
        if row_out:
            hypothesis_rows.append([name, row_out.get("h1", "–"), row_out.get("h2", "–"), row_out.get("h3", "–"), row_out.get("h4", "–"),
                                    row_out.get("clean", "–")])
    for h, by_model in verdicts.items():
        if by_model:
            values[f"e16.{h}_summary"] = verdict_text(by_model)
    changed = [f"{h.upper()} on {NAMES[m]} becomes {values[f'e16.{KEYS[m]}.clean.{h}']}" for m in NAMES for h in ("h1", "h2")
               if f"e16.{KEYS[m]}.clean.{h}" in values and values[f"e16.{KEYS[m]}.clean.{h}"] != values[f"e16.{KEYS[m]}.{h}"]]
    if any(f"e16.{KEYS[m]}.clean.h1" in values for m in NAMES):
        values["e16.clean_summary"] = "the verdicts on H1 and H2 are unchanged" if not changed else "; ".join(changed)
    if plain_rows:
        values["e16.models"] = str(len(plain_rows))
        table("confirm", ["Model", "No repair", "Reader", "Exposure", "**Bridge**", "**RRF**", "Reader-first (found)", "Oracle score",
                          "Short suffix: reader-first vs RRF", "Long suffix: reader-first vs RRF"], plain_rows,
              f"**Table {{{{#confirm}}}}.** The replication with hypotheses fixed in advance: 36 questions per model drawn with seeds not used before (108 cases; {values.get('e16.qwen1.5b.reused', '–')} questions also appear in other experiments' results), 10% of the suffix. Teacher-forced KL in nats from the re-prefill, mean over cases; the last two columns are hypothesis H2's tests (paired Wilcoxon over questions). Appendix D gives the decision rules, every hypothesis's verdict under them, and the tests without the questions that also appear in other experiments.")
    if hypothesis_rows:
        table("hypotheses", ["Model", "H1 (Holm-adjusted p)", "H2 (p by stratum)", "H3", "H4 (excess log-probability)", "Without reused questions"], hypothesis_rows,
              "**Table {{#D:hypotheses}}.** Verdicts under the decision rules above, fixed before the results of Qwen2.5-0.5B and SmolLM2-1.7B were examined and after Qwen2.5-1.5B's had been seen ($\\alpha = 0.05$, two-sided, per model): holds, not confirmed, or fails (significant in the opposite direction, or for H3 and H4 the stated condition is false). The last column repeats H1 and H2 without the questions that also appear in another experiment's results.")


def e17() -> None:
    for model in NAMES:
        data = load(f"e17_precision_{KEYS[model]}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e17.{KEYS[model]}"
        values[f"{tag}.n"] = str(len(rows))
        for field in ("fp32_vs_bf16", "bf16_vs_oneshot", "fp32_vs_oneshot"):
            tf = [statistics.fmean(r[field]) for r in rows]
            values[f"{tag}.{field}"], values[f"{tag}.{field}.median"] = kl(statistics.fmean(tf)), kl(statistics.median(tf))
            values[f"{tag}.{field}.first"] = kl(statistics.fmean(r[field][0] for r in rows))
        values[f"{tag}.top1"] = pct(statistics.fmean(r["top1_fp32_bf16"] for r in rows))
        repair = float(values[f"e2c.{KEYS[model]}.kl_tf.bridge.10"])
        floor = statistics.fmean(statistics.fmean(r["fp32_vs_bf16"]) for r in rows)
        values[f"{tag}.vs_repair"] = f"{floor / repair:.0%} of" if floor < repair else f"{floor / repair:.1f} times"
        gaps = [abs(float(values[f"e2c.{KEYS[model]}.kl_tf.bridge.10"]) - float(values[f"e2c.{KEYS[model]}.kl_tf.rrf.10"]))] if f"e2c.{KEYS[model]}.kl_tf.rrf.10" in values else []
        if gaps:
            values[f"{tag}.bridge_rrf_gap"] = kl(gaps[0])


def e5_v2() -> None:
    data = load("e5_streams_qwen1.5b_v2")
    if not data or not data["arrival"]:
        return
    arrival, merge, tag = data["arrival"], data["merge"], "e5v2.qwen1.5b"
    values[f"{tag}.n"] = str(len(arrival))
    pending = [sum(r["doc_lengths"][r["arrival"][-1]:]) for r in arrival]
    values[f"{tag}.inorder"] = f"{statistics.fmean(pending):.0f}"
    values[f"{tag}.wait"] = f"{statistics.fmean(r['doc_tokens'] for r in arrival):.0f}"
    values[f"{tag}.last"] = f"{statistics.fmean(r['last_doc_tokens'] for r in arrival):.0f}"
    values[f"{tag}.reader_tokens"] = f"{statistics.fmean(r['reader_tokens'] for r in arrival):.0f}"
    values[f"{tag}.sweep"] = f"{statistics.fmean(r['sweep']['tokens'] for r in arrival):.0f}"
    for workload, rows in (("arrival", arrival), ("merge", merge)):
        keys = [k for k in rows[0] if isinstance(rows[0][k], dict) and "kl_last" in rows[0][k]]
        truth = "truth_answer"
        for key in keys:
            values[f"{tag}.{workload}.{key}.kl"] = kl(statistics.fmean(r[key]["kl_last"] for r in rows))
            values[f"{tag}.{workload}.{key}.same"] = pct(statistics.fmean(float(normalise(r[key]["answer"]) == normalise(r[truth])) for r in rows))
            if "tokens" in rows[0][key]:
                values[f"{tag}.{workload}.{key}.tokens"] = f"{statistics.fmean(r[key]['tokens'] for r in rows):.0f}"
        for share in ("0.1", "0.2"):
            for estimate in ("flat", "sketch"):
                a = [r[f"repair_{estimate}_first@{share}"]["kl_last"] for r in rows]
                b = [r[f"repair_{estimate}@{share}"]["kl_last"] for r in rows]
                w, p = win_p(a, b)
                values[f"{tag}.{workload}.{estimate}.{share}.first_vs_plain.win"], values[f"{tag}.{workload}.{estimate}.{share}.first_vs_plain.p"] = pct(w), pvalue(p)
    latency = [r["last_doc_tokens"] + 2 * r["reader_tokens"] + r["repair_flat@0.1"]["tokens"] for r in arrival]
    values[f"{tag}.repair_path"] = f"{statistics.fmean(latency):.0f}"
    values[f"{tag}.inorder_path"] = f"{statistics.fmean(p + r['reader_tokens'] for p, r in zip(pending, arrival)):.0f}"


def e22() -> None:
    """CacheBlend's selection exactly as LMCache runs it (absolute key deviation at layer 1) next to our first-layer criterion."""
    for model in NAMES:
        data = load(f"e22_lmcache_{KEYS[model]}_main")
        main = load(f"e2c_ablation_{KEYS[model]}_main")
        if not data or len(data["rows"]) < 108 or not main:
            continue
        rows, tag = data["rows"], f"e22.{KEYS[model]}"
        reference = {r["name"]: r["delete"]["results"]["bridge"]["0.1"]["kl_tf"] for r in main["rows"]}
        gap = max(abs(r["results"]["bridge"]["kl_tf"] - reference[r["name"]]) / max(reference[r["name"]], 1e-12) for r in rows)
        values[f"{tag}.reproduces"] = "exactly" if gap == 0 else f"to a relative {sci(gap)}"
        cids = [question(r) for r in rows]
        get = lambda r, key: r["results"][key]["kl_tf"]
        values[f"{tag}.n"] = str(len(rows))
        for key in ("stale", "probe1", "probe1_lmcache", "bridge"):
            values[f"{tag}.{key}"] = kl(statistics.fmean(get(r, key) for r in rows))
        values[f"{tag}.overlap"] = pct(statistics.fmean(r["overlap"] for r in rows))
        values[f"{tag}.ratio"] = ratio_text(ratio_ci([get(r, "probe1_lmcache") for r in rows], [get(r, "probe1") for r in rows], cids))
        share, p = win_p([get(r, "probe1_lmcache") for r in rows], [get(r, "probe1") for r in rows], cids)
        values[f"{tag}.p"] = pvalue(p)
        values["e22.model"] = NAMES[model]
        values["e22.key"] = KEYS[model]


def per_experiment() -> None:
    """One summary per experiment at a fixed budget: the share of the no-repair KL removed (one minus the ratio of
    mean KLs over all its cases) with a 95% BCa interval that resamples clusters. A policy is behind the best one
    when the paired BCa interval of its mean KL over the best policy's excludes 1 (the comparison of Section 2.4)."""
    order = [("bridge", "Bridge score"), ("rrf", "Rank fusion (RRF)"), ("segment_exposure", "Reader-first (segment found)"), ("exposure", "Recorded exposure"),
             ("reader", "Reader attention"), ("tail", "Tail window"), ("boundary", "Boundary window")]
    names = {"main": "the main experiment", "replication": "the replication", "long": "long contexts", "agent": "the synthetic agent traces",
             "needle": "the needle task", "7b": "the 7B models", "traces": "the real transcripts"}
    rows_out, summary, behind, ratio_to_best, role_aware, combined_ratio = [], {}, {}, {}, {}, {}
    for exp, label in EXPERIMENT_LABELS:
        cases = experiment_cases.get(exp)
        if not cases:
            continue
        clusters, stale = [c["cluster"] for c in cases], [c["stale"] for c in cases]
        shares = {}
        for key, _ in order:
            if all(key in c for c in cases):
                point, low, high = ratio_ci([c[key] for c in cases], stale, clusters)
                shares[key] = (1 - point, 1 - high, 1 - low)
                values[f"robust_exp.{exp}.{key}"] = pct(1 - point)
                values[f"robust_exp.{exp}.{key}.ci"] = f"{pct(1 - point)} [{100 * (1 - high):.0f}, {100 * (1 - low):.0f}]"
        best = max(shares, key=lambda k: shares[k][0])
        values[f"robust_exp.{exp}.best"] = best
        summary[exp] = shares
        cells = []
        for key, _ in order:
            if key not in shares:
                cells.append("–")
                continue
            if key != best:
                ratio = ratio_ci([c[key] for c in cases], [c[best] for c in cases], clusters)
                ratio_to_best[(exp, key)] = ratio
                if ratio[1] > 1:
                    behind.setdefault(key, []).append(exp)
            share, low, high = shares[key]
            text = f"{pct(share)} [{100 * low:.0f}, {100 * high:.0f}]"
            cells.append(f"**{text}**" if key == best else text if key in behind and exp in behind[key] else f"*{text}*")
        rows_out.append([label, f"{len(cases)} ({len(set(clusters))})"] + cells)
        if "bridge" in shares and "rrf" in shares:
            between = ratio_ci([c["bridge"] for c in cases], [c["rrf"] for c in cases], clusters)
            combined_ratio[exp] = between
            if between[1] > 1 or between[2] < 1:
                behind.setdefault("_combined", []).append(exp)
                if 1 < between[1] < 1.02 or 0.98 < between[2] < 1:
                    behind.setdefault("_narrow", []).append(exp)
        if all("question_exposure" in c for c in cases):
            q_share = 1 - ratio_ci([c["question_exposure"] for c in cases], stale, clusters)[0]
            if q_share > shares[best][0]:
                role_aware[exp] = {key: ratio_ci([c[key] for c in cases], [c["question_exposure"] for c in cases], clusters) for key in ("bridge", "rrf") if key in shares}
                role_aware[exp]["share"] = q_share
    if not rows_out:
        return
    number = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    for key, _ in order:
        run = [exp for exp in summary if key in summary[exp]]
        values[f"robust_exp.{key}.not_behind"] = f"{len(run) - len(behind.get(key, []))} of {len(run)}"
        values[f"robust_exp.{key}.behind"] = f"{len(behind.get(key, []))} of {len(run)}"
        values[f"robust_exp.{key}.not_behind_phrase"] = (f"in all {number[len(run)]} experiments where it ran" if not behind.get(key)
                                                         else f"in {len(run) - len(behind[key])} of the {number[len(run)]} experiments where it ran")
        ratios = [ratio_to_best[(exp, key)][0] for exp in run if (exp, key) in ratio_to_best]
        values[f"robust_exp.{key}.max_ratio"] = f"{max(ratios):.1f}" if ratios else "1.0"
    both = [exp for exp in summary if "bridge" in summary[exp] and "rrf" in summary[exp]]
    separated = behind.get("_combined", [])
    narrow = behind.get("_narrow", [])
    values["robust_exp.combined_separated"] = (
        f"the paired ratio of their mean KLs, pooled over each experiment's cases, separates them in {number[len(separated)]} of the {number[len(both)]} experiments where both ran"
        + (" (" + "; ".join(f"{names[e]}{', narrowly' if e in narrow else ''}: bridge over rank fusion {ratio_text(combined_ratio[e])}" for e in separated) + ")" if separated else ""))
    rrf_lower = [e for e in both if combined_ratio[e][0] > 1]
    rrf_sig = [e for e in rrf_lower if combined_ratio[e][1] > 1]
    values["robust_exp.rrf_direct"] = (f"rank fusion has the lower mean KL in {number[len(rrf_lower)]} of the {number[len(both)]} experiments where both ran, "
                                       f"significantly in {number[len(rrf_sig)]}")
    traces_italic = [k for k in summary.get("traces", {}) if k != values.get("robust_exp.traces.best") and k not in [p for p, lst in behind.items() if "traces" in lst]]
    values["robust_exp.traces.not_behind_others"] = number.get(len(traces_italic), str(len(traces_italic)))
    sentences = []
    for exp, r in role_aware.items():
        digits = max(ratio_digits(r["bridge"]), ratio_digits(r["rrf"]))
        sentences.append(f"In {names[exp]}, recomputing the cached question first when the engine knows which segment it is, then filling by exposure, removes "
                         f"{pct(r['share'])} of the no-repair KL, more than any policy of Table {{{{#robust_experiments}}}}; the bridge score leaves {ratio_text(r['bridge'], digits)} "
                         f"times its mean KL and rank fusion {ratio_text(r['rrf'], digits)}.")
    values["robust_exp.role_aware"] = (" " + " ".join(sentences)) if sentences else ""
    others = [exp for exp in summary if exp != "traces"]
    for key in ("bridge", "rrf", "segment_exposure", "reader", "exposure"):
        xs = [summary[exp][key][0] for exp in others if key in summary[exp]]
        if xs:
            values[f"robust_exp.{key}.range_qa"] = f"{min(xs):.0%} to {max(xs):.0%}"
            values[f"robust_exp.{key}.count_qa"] = number[len(xs)]
    if "traces" in summary and others:
        signal = [k for k in ("bridge", "rrf", "segment_exposure", "exposure", "reader") if k in summary["traces"]]
        lowest_elsewhere = {k: min(summary[exp][k][0] for exp in others if k in summary[exp]) for k in signal}
        less = all(summary["traces"][k][0] < lowest_elsewhere[k] for k in signal)
        much_less = all(summary["traces"][k][0] < 0.5 * lowest_elsewhere[k] for k in signal)
        best_traces = max(summary["traces"], key=lambda k: summary["traces"][k][0])
        values["robust_exp.traces.best_ci"] = values[f"robust_exp.traces.{best_traces}.ci"]
        values["robust_exp.traces.best_name"] = POLICY_NAMES.get(best_traces, best_traces)
        values["e21.vs_qa"] = ("Every policy that uses exposure or reader attention removes less here than in any other experiment (Table {{#robust_experiments}}). "
                               if less else "")
        values["e21.abstract"] = (" On real agent transcripts, reusing the stale suffix after an old tool result is cleared moves the next turn little from a re-prefill, "
                                  f"and every policy that uses exposure or reader attention removes {'much less' if much_less else 'less'} of that error "
                                  f"(the product {values['robust_exp.traces.bridge.ci']}).")
        values["e21.conclusion"] = ("On real agent transcripts, clearing an old tool result barely moved the next turn, and repair removed only part of that small error "
                                    f"({values['robust_exp.traces.best_name']}: {values['robust_exp.traces.best_ci']}, with "
                                    f"{number[sum(1 for k in summary['traces'] if k != best_traces and 'traces' not in behind.get(k, []))]} other policies not significantly behind).")
        values["robust_exp.traces.much_less"] = "much less" if much_less else "less"
    bridge_behind, rrf_behind = behind.get("bridge", []), behind.get("rrf", [])
    verdict = lambda key, lst: ("is never significantly behind the best role-free policy compared" if not lst else
                                f"is significantly behind the best in {number[len(lst)]} ({', '.join(names[e] for e in lst)}; "
                                f"up to {values[f'robust_exp.{key}.max_ratio']} times its mean KL)")
    values["robust_exp.rrf.verdict"], values["robust_exp.bridge.verdict"] = verdict("rrf", rrf_behind), verdict("bridge", bridge_behind)
    values["robust_exp.bridge.behind_count"] = number[len(bridge_behind)]
    values["robust_exp.rrf_preferred"] = (f", and in our experiments the more robust of the two: never significantly behind the best role-free policy, while the product is behind it in {number[len(bridge_behind)]}, "
                                          f"though compared directly {values['robust_exp.rrf_direct']} (Section 5.3)" if not rrf_behind and bridge_behind else "")
    values["robust_exp.rrf_preferred_51"] = (" Across experiments rank fusion is never significantly behind the best role-free policy compared and the product is behind it in "
                                             f"{number[len(bridge_behind)]}, while compared directly {values['robust_exp.rrf_direct']} (Section 5.3); rank fusion is the "
                                             "combination we recommend, and we keep the product because the first-order account explains it." if not rrf_behind and bridge_behind else "")
    values["robust_exp.bridge.verdict_short"] = (f"falls significantly behind in {number[len(bridge_behind)]}, by up to {values['robust_exp.bridge.max_ratio']} times the best policy's mean KL"
                                                 if bridge_behind else "is never significantly behind the best")
    bridge_ratios = "; ".join(f"{ratio_text(ratio_to_best[(e, 'bridge')])} times the best policy's mean KL in {names[e]}" for e in bridge_behind)
    values["robust_exp.claim"] = (
        f"rank fusion of the two signals removes {values['robust_exp.rrf.range_qa']} of the no-repair KL in the {values['robust_exp.rrf.count_qa']} where it ran "
        + ("and is never significantly behind the best training-free policy" if not rrf_behind
           else f"and is significantly behind the best training-free policy in {number[len(rrf_behind)]}")
        + f"; their product removes {values['robust_exp.bridge.range_qa']} and "
        + (f"is significantly behind in {number[len(bridge_behind)]} ({bridge_ratios})" if bridge_behind else "is never significantly behind"))
    values["robust_exp.claim_short"] = (
        f"combined by rank fusion they remove {values['robust_exp.rrf.range_qa']} of the no-repair KL and are never significantly behind the best training-free policy"
        if not rrf_behind else f"combined by rank fusion they remove {values['robust_exp.rrf.range_qa']} of the no-repair KL")
    values["robust_exp.claim_short"] += (f", and combined by their product {values['robust_exp.bridge.range_qa']}, behind the best in {number[len(bridge_behind)]} of them"
                                         if bridge_behind else f", and combined by their product {values['robust_exp.bridge.range_qa']}")
    three = [e for e in others if e in ("main", "replication")]
    single = [e for e in others if e in ("long", "agent", "needle")]
    short = {"main": "multi-document question answering", "replication": "a replication with hypotheses fixed in advance", "long": "4k–16k-token contexts",
             "agent": "synthetic agent traces", "needle": "needle erasure", "7b": "7B models"}
    join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    parts = []
    if three:
        parts.append(f"{join([short[e] for e in three])} on all three")
    if single:
        parts.append(f"{join([short[e] for e in single])} on Qwen2.5-1.5B")
    if "7b" in others:
        parts.append("the main experiment's cases at 7B")
    values["robust_exp.list_abstract"] = "; ".join(parts)
    long_names = {"main": "question answering", "replication": "a replication with hypotheses fixed in advance", "long": "4k–16k-token contexts",
                  "agent": "synthetic agent traces", "needle": "needle erasure", "7b": "the main experiment at 7B"}
    groups = ["question answering" + (", also at 7B" if "7b" in others else "")] if "main" in three else []
    groups += [long_names[e] for e in three if e != "main"]
    if single:
        groups.append(join([long_names[e] for e in single]) + " on one model")
    values["robust_exp.list_abstract_short"] = "; ".join(groups)
    values["robust_exp.count_abstract"] = number.get(len([e for e in others if e != "7b"]), str(len(others)))
    values["robust_exp.rrf.count_abstract"] = number[len([e for e in others if e != "7b" and "rrf" in summary[e]])]
    values["robust_exp.count_qa"] = number.get(len(others), str(len(others)))
    values["robust_exp.list_qa"] = ", ".join(names[e] for e in others[:-1]) + " and " + names[others[-1]]
    table("robust_experiments", ["Experiment", "Cases (clusters)"] + [label for _, label in order], rows_out,
          "**Table {{#robust_experiments}}.** One summary per experiment at a fixed budget, 10% of the suffix (5% on the needle task), for role-free policies (Section 5.1): the share of the no-repair KL a policy removes, one minus the ratio of mean KLs over all the experiment's cases (models and strata pooled), with a 95% BCa interval that resamples clusters (questions; transcripts; needle cases). Teacher-forced KL over the reply, except the KL at the first reply token on the needle task. Bold: highest point estimate; italics: not significantly behind it (the paired BCa interval of the ratio of its mean KL to the bold policy's includes 1); plain: significantly behind. – = not run on every case of the experiment. Table {{#C:robustness}} and Figure {{#fig_robust}} give every budget and stratum separately.")


def robustness() -> None:
    """Share of the no-repair KL that each practical policy removes, per setting, and its distance to the best policy run there."""
    if not robust_settings:
        return
    points = lambda x: f"{100 * x:.0f}"
    removed_by = lambda setting, key: 1 - setting["means"][key] / setting["stale"]
    rows_out = []
    for key, label in ROBUST_POLICIES:
        removed, regret = [], []
        for setting in robust_settings:
            if key not in setting["means"] or setting["stale"] <= 0:
                continue
            best = max(removed_by(setting, k) for k in setting["means"])
            removed.append(removed_by(setting, key))
            regret.append(best - removed_by(setting, key))
        if not removed:
            continue
        tag = f"robust.{key}"
        values[f"{tag}.settings"] = str(len(removed))
        values[f"{tag}.removed_median"], values[f"{tag}.removed_min"] = pct(statistics.median(removed)), pct(min(removed))
        values[f"{tag}.regret_median"], values[f"{tag}.regret_max"] = points(statistics.median(regret)), points(max(regret))
        values[f"{tag}.best"] = str(sum(r <= 1e-12 for r in regret))
        where = [st for st in robust_settings if key in st["means"] and st["stale"] > 0]
        worst = where[max(range(len(regret)), key=regret.__getitem__)]
        values[f"{tag}.worst_setting"] = worst["setting"]
        values[f"{tag}.worst_experiment"] = worst["experiment"]
        values[f"{tag}.within5"] = str(sum(r <= 0.05 for r in regret))
        rows_out.append([label, str(len(removed)), pct(statistics.median(removed)), pct(min(removed)), values[f"{tag}.best"],
                         values[f"{tag}.within5"], points(statistics.median(regret)), points(max(regret))])
    both = [st for st in robust_settings if "bridge" in st["means"] and "rrf" in st["means"]]
    if both:
        gap = [removed_by(st, "rrf") - removed_by(st, "bridge") for st in both]
        values["robust.both.n"] = str(len(both))
        values["robust.both.rrf_ahead"] = str(sum(g > 0 for g in gap))
        values["robust.both.max_gap"] = points(max(abs(g) for g in gap))
    narrow = [st for st in robust_settings if st["experiment"] != "needle" and not st["setting"].startswith("Replication with notes")
              and "segment_exposure" in st["means"] and st["stale"] > 0]
    if narrow:
        values["robust.segment_exposure.regret_max_narrow"] = f"{100 * max(max(1 - st['means'][k] / st['stale'] for k in st['means']) - (1 - st['means']['segment_exposure'] / st['stale']) for st in narrow):.0f}"
    worst = {key: values.get(f"robust.{key}.worst_setting", "") for key in ("bridge", "rrf")}
    values["robust.combined_worst"] = ("The two fall furthest behind with a short suffix and a small budget, where recomputing the question whole is best."
                                       if all(w.startswith("Main") and "short suffix" in w for w in worst.values()) else
                                       f"The bridge score falls furthest behind in one setting ({worst['bridge']}), rank fusion in another ({worst['rrf']}).")
    readable = {"needle": "on the needle task", "traces": "on the real agent transcripts", "main": "in the main experiment", "agent": "on the synthetic agent traces",
                "long": "in long contexts", "replication": "in the replication", "qwen7b": "on Qwen2.5-7B", "mistral7b": "on Mistral-7B"}
    values["robust.segment_exposure.worst_where"] = readable.get(values.get("robust.segment_exposure.worst_experiment", ""), "")
    values["robust.settings"] = str(len(robust_settings))
    designs = {{"qwen7b": "7b", "mistral7b": "7b"}.get(st["experiment"], st["experiment"]) for st in robust_settings}
    values["robust.experiments"] = str(len(designs))
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    values["robust.experiments_word"] = words.get(len(designs), values["robust.experiments"])
    (OUT / "robustness.json").write_text(json.dumps(robust_settings, indent=1))
    table("robustness", ["Policy", "Settings run", "Share of no-repair KL removed: median", "worst", "Best in setting", "Within 5 points of best",
                         "Points behind best: median", "worst"], rows_out,
          f"**Table {{{{#C:robustness}}}}.** Robustness over {len(robust_settings)} settings: every combination of experiment, model, suffix or context length, workload variant and budget (every budget defined as a share of the suffix) in {values['robust.experiments_word']} experiments. In each setting, the share of the no-repair KL a policy removes is one minus the ratio of mean KLs; a policy's distance to the best is the difference, in percentage points, from the best practical policy run in that setting. Not every policy was run in every setting (second column). The needle settings use the KL at the first reply token, the others teacher-forced KL over the reply. Figure {{{{#fig_robust}}}} shows every setting.")


def e20() -> None:
    """Our bfloat16 forward pass against the library's on every main-experiment prompt, for the two 7B models."""
    rows_out = []
    for key, label in (("qwen7b", "Qwen2.5-7B"), ("mistral7b", "Mistral-7B")):
        data = load(f"e20_reference_{key}_main")
        if not data or not data["rows"]:
            continue
        rows, tag = data["rows"], f"e20.{key}"
        same = [r["same_argmax"] for r in rows]
        margins = [m for r in rows for m in r["margin_at_disagreements"]]
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.same.median"], values[f"{tag}.same.min"] = pct(statistics.median(same)), pct(min(same))
        values[f"{tag}.gap.max"] = f"{max(r['max_gap'] for r in rows):.1f}"
        values[f"{tag}.gap.at_zero"] = f"{sum(r['max_gap_at_position_0'] for r in rows)} of {len(rows)}"
        values[f"{tag}.gap.rest.max"] = f"{max(r['max_gap_without_position_0'] for r in rows):.1f}"
        values[f"{tag}.gap.rest.median"] = f"{statistics.median(r['max_gap_without_position_0'] for r in rows):.1f}"
        values[f"{tag}.near_ties"] = pct(sum(m < 0.5 for m in margins) / len(margins)) if margins else "–"
        values[f"{tag}.margin.disagree"] = f"{statistics.median(margins):.2f}" if margins else "–"
        values[f"{tag}.margin.all"] = f"{statistics.median(r['median_margin'] for r in rows):.2f}"
        values[f"{tag}.kl"] = sci(statistics.fmean(r["kl_mean"] for r in rows))
        rows_out.append((label, tag))
    if len(rows_out) == 2:
        q, m = "e20.qwen7b", "e20.mistral7b"
        both = lambda key, fmt: (f"{fmt(values[f'{q}.{key}'])} on both models" if values[f"{q}.{key}"] == values[f"{m}.{key}"]
                                 else f"{fmt(values[f'{q}.{key}'])} and {fmt(values[f'{m}.{key}'])} on the two models")
        values["e20.same.both"] = both("same.median", lambda x: f"a median {x} of positions")
        all_ties = values[f"{q}.near_ties"] == values[f"{m}.near_ties"] == "100%"
        values["e20.ties.both"] = ("every disagreement on both models is at a top-2 margin below 0.5 logits" if all_ties
                                   else f"{values[f'{q}.near_ties']} and {values[f'{m}.near_ties']} of the disagreements are at a top-2 margin below 0.5 logits")
        zero = values[f"{q}.gap.at_zero"].startswith("0 of") and values[f"{m}.gap.at_zero"].startswith("0 of")
        values["e20.gap_zero.both"] = ("never sits at its first position on either model" if zero
                                       else f"sits at its first position in {values[f'{q}.gap.at_zero']} prompts on Qwen2.5-7B and {values[f'{m}.gap.at_zero']} on Mistral-7B")
    if rows_out:
        measures = [("Positions with the same next token: median over prompts", "same.median"), ("Same next token: lowest prompt", "same.min"),
                    ("Prompts whose largest logit gap is at position 0", "gap.at_zero"), ("Largest gap at other positions: median over prompts", "gap.rest.median"),
                    ("Largest gap at other positions: largest prompt", "gap.rest.max"), ("Top-2 margin where the next tokens differ", "margin.disagree"),
                    ("Top-2 margin at all positions", "margin.all"), ("Disagreements at a top-2 margin below 0.5", "near_ties"), ("Mean KL(library, ours)", "kl")]
        counts = sorted({values[f"{tag}.n"] for _, tag in rows_out})
        prompts = f"{counts[0]} prompts per model" if len(counts) == 1 else " and ".join(f"{values[f'{tag}.n']} prompts on {label}" for label, tag in rows_out)
        table("reference7b", ["Measure"] + [label for label, _ in rows_out], [[name] + [values[f"{tag}.{key}"] for _, tag in rows_out] for name, key in measures],
              f"**Table {{{{#F:reference7b}}}}.** Our bfloat16 forward pass against the library's bfloat16 pass on the first 512 tokens of each main-experiment prompt, {prompts}. Gaps are absolute logit differences; margins are medians of the library's gap between its first and second choice, in logits; the KL is averaged over positions and prompts.")


def scope() -> None:
    """Sentences about which models were run, generated so that they cannot run ahead of the results."""
    large = [name for key, name in (("e9.qwen7b.n", "Qwen2.5-7B"), ("e19.mistral7b.n", "Mistral-7B")) if key in values]
    families = 2 + ("Mistral-7B" in large)
    count = {3: "three", 4: "four", 5: "five"}[3 + len(large)]
    values["scope.models"] = f"{count} models of 0.5B to {'7B' if large else '1.7B'} parameters from {'three' if families == 3 else 'two'} families"
    values["scope.max_params"] = "7B" if large else "1.7B"
    cited = {"Qwen2.5-7B": "Qwen2.5-7B [@qwen25]", "Mistral-7B": "Mistral-7B-Instruct-v0.3 [@jiang2023mistral]"}
    values["scope.7b_setting"] = f". Section 5.3 adds {' and '.join(cited[name] for name in large)}, run in bfloat16" if large else ""
    if large:
        values["scope.7b"] = " and ".join(large)
        values["scope.7b_count"] = "two 7B models" if len(large) == 2 else "one 7B model"
    real = "e21.n" in values
    values["scope.traces"] = "synthetic and real agent traces" if real else "synthetic agent traces"
    values["scope.trace_edits"] = ("tool results in synthetic and real agent traces" if real
                                   else "tool results in synthetic tool-use traces built from the same documents")
    values["scope.transcripts_limit"] = (
        f"The real transcripts are {values['e21.n']} trajectories from one dataset, cut at no more than 3,000 tokens of the Qwen2.5 tokenizer "
        f"(at most {values['e21.max_tokens']}), whose actions another model wrote; we score the model's own next turn against a re-prefill, "
        "not task success. Longer transcripts with tool schemas, code and long reasoning as readers, and long sequences of edits, remain untested." if real else
        "Real agent transcripts, with tool schemas, code and long reasoning as readers, and long sequences of edits remain untested.")
    e1p_done = [NAMES[m] for m in NAMES if f"e1p.{KEYS[m]}.n" in values]
    values["scope.e1p_models"] = "each of the three small models" if len(e1p_done) == 3 else " and ".join(e1p_done)


def appendix_tables() -> None:
    """Tables for the appendix that combine runs: decoy and leakage (E16), precision (E17, E18), runtime (E5 v2), later positions."""
    decoy_rows, leak_rows, precision_rows, later_rows = [], [], [], []
    reference_agree, reference_gap = {}, {}
    for model in NAMES:
        tag, name = f"e16.{KEYS[model]}", NAMES[model]
        if f"{tag}.decoy.density_finds" in values:
            decoy_rows.append([name, values[f"{tag}.plain.density_finds"] if f"{tag}.plain.density_finds" in values else "–",
                               values[f"{tag}.plain.shortest_finds"] if f"{tag}.plain.shortest_finds" in values else "–",
                               values[f"{tag}.decoy.density_finds"], values[f"{tag}.decoy.shortest_finds"], values[f"{tag}.decoy.mass_finds"],
                               values[f"{tag}.decoy.question_exposure"], values[f"{tag}.decoy.density_exposure"], values[f"{tag}.decoy.shortest_exposure"],
                               values[f"{tag}.decoy.rrf"], values[f"{tag}.decoy.bridge"]])
        if f"{tag}.leak.stale" in values:
            leak_rows.append([name, values[f"{tag}.leak.present_gap"], values[f"{tag}.leak.stale"], values[f"{tag}.leak.reader"],
                              values[f"{tag}.leak.bridge"], values[f"{tag}.leak.rrf"], values[f"{tag}.leak.density_exposure"]])
        p = f"e17.{KEYS[model]}"
        ref = load(f"e18_reference_{KEYS[model]}_main")
        if f"{p}.fp32_vs_bf16" in values or ref:
            line = [name, values.get(f"{p}.fp32_vs_bf16", "–"), values.get(f"{p}.fp32_vs_bf16.median", "–"), values.get(f"{p}.top1", "–"),
                    values.get(f"{p}.bf16_vs_oneshot", "–")]
            if ref:
                for pair in ("engine16_vs_library16", "library16_vs_library32"):
                    agree = statistics.fmean(r[pair]["same_argmax"] for r in ref["rows"])
                    values[f"e18.{KEYS[model]}.{pair}"] = f"{100 * agree:.1f}%"
                    values[f"e18.{KEYS[model]}.{pair}.gap"] = f"{max(r[pair]['max_gap'] for r in ref['rows']):.1f}"
                    reference_agree.setdefault(pair, []).append(agree)
                    reference_gap.setdefault(pair, []).append(max(r[pair]["max_gap"] for r in ref["rows"]))
                    line.append(values[f"e18.{KEYS[model]}.{pair}"])
            else:
                line += ["–", "–"]
            precision_rows.append(line)
        c = f"e2c.{KEYS[model]}"
        if f"{c}.later.stale" in values:
            later_rows.append([name, values[f"{c}.later.n"], values[f"{c}.later.stale"], values[f"{c}.later.reader"], values[f"{c}.later.bridge"],
                               values[f"{c}.later.rrf"], values[f"{c}.later.segment_exposure"]])
    if decoy_rows:
        table("decoy", ["Model", "Finds question: attention, plain", "shortest, plain", "attention, notes", "shortest, notes", "total attention, notes",
                        "Question given first", "Found segment first", "Shortest segment first", "RRF", "Bridge"], decoy_rows,
              "**Table {{#D:decoy}}.** Finding the reader segment (hypothesis H3). Plain: the replication's cases; notes: the same cases with two notes of about eight tokens in the suffix, one right after the edited document and one at the end. Which rule picks the question segment: the most reader attention per token, the shortest segment, the most total reader attention. The \u201cfirst\u201d columns recompute that segment whole and fill the rest of the budget by exposure. Teacher-forced KL in nats at 10% with notes.")
    if leak_rows:
        table("leak", ["Model", "Document present", "No repair", "Reader attention", "Bridge", "RRF", "Reader-first (segment found)"], leak_rows,
              "**Table {{#D:leak}}.** Can the deleted document still be read (hypothesis H4)? After the deletion the new input asks to quote the deleted document's first sentence, and we score that sentence teacher-forced: mean log-probability per token, in nats, minus the re-prefill's, with 95% intervals over questions. Document present: the unedited context, for scale. Repairs recompute 10% of the suffix for the original answer, not for the probe.")
    before = load("backup/old_bf16_attention/e18_reference_qwen1.5b_main")
    if before:
        values["e18.qwen1.5b.before_fix"] = f"{100 * statistics.fmean(r['engine16_vs_library16']['same_argmax'] for r in before['rows']):.1f}%"
        values["e18.qwen1.5b.before_fix.gap"] = f"{max(r['engine16_vs_library16']['max_gap'] for r in before['rows']):.1f}"
    span = lambda xs, fmt: fmt(min(xs)) if fmt(min(xs)) == fmt(max(xs)) else f"{fmt(min(xs))} to {fmt(max(xs))}"
    if len(reference_gap.get("engine16_vs_library16", [])) == len(reference_gap.get("library16_vs_library32", [])) > 0:
        pairs = zip(reference_gap["engine16_vs_library16"], reference_gap["library16_vs_library32"])
        values["e18.gap_order"] = "smaller than" if all(ours < theirs for ours, theirs in pairs) else "not always smaller than"
    for pair, agree in reference_agree.items():
        values[f"e18.{pair}.range"] = span(agree, lambda x: f"{100 * x:.1f}%")
        values[f"e18.{pair}.gap.range"] = span(reference_gap[pair], lambda x: f"{x:.1f}")
    if precision_rows:
        table("precision", ["Model", "fp32 vs bf16 re-prefill: KL mean", "median", "same top-1", "bf16 cached prefix vs one pass", "Engine vs library, bf16: same argmax",
                            "Library bf16 vs fp32: same argmax"], precision_rows,
              "**Table {{#F:precision}}.** Numerical noise. Left: the re-prefill of the main experiment's cases computed in 32-bit floats and in bfloat16 (normalisation, attention scores, softmax and the weighted sum of values in 32 bits), compared over the 32-bit reply (teacher-forced KL in nats), and bfloat16 with the prefix read from cache against one bfloat16 pass. Right: share of positions with the same argmax over the first 512 tokens of four prompts, our bfloat16 forward pass against the library's, and the library's bfloat16 pass against its 32-bit pass.")
    token_rows = []
    for model in NAMES:
        data = load(f"e16_confirm_{KEYS[model]}_plain_main")
        if not data or len(data["rows"]) != 108:
            continue
        rows = data["rows"]
        mean_kl = lambda r, key, budget: statistics.fmean((r["results"][key][budget] if budget else r["results"][key])["kl"])
        q_len = statistics.fmean(r["segment_lengths"][r["question_index"]] for r in rows)
        for index, (key, label) in enumerate((("reader", "Reader attention"), ("bridge", "Bridge"), ("rrf", "RRF"),
                                              ("density_exposure", "Reader-first (segment found)"), ("question_only", "Cached question alone"))):
            line = [NAMES[model] if index == 0 else "", label]
            for budget in ("t16", "t32", "t64"):
                line.append(kl(statistics.fmean(mean_kl(r, key, budget) for r in rows)) if key != "question_only" else "–")
            at_question = {"bridge": "bridge@question", "rrf": "rrf@question", "question_only": "question_only"}.get(key)
            line.append(kl(statistics.fmean(mean_kl(r, at_question, None) for r in rows)) if at_question else "–")
            token_rows.append(line)
        values[f"e16.{KEYS[model]}.question_tokens"] = f"{q_len:.0f}"
    if token_rows:
        table("tokens", ["Model", "Policy", "16 tokens", "32 tokens", "64 tokens", "As many tokens as the question"], token_rows,
              "**Table {{#D:tokens}}.** The replication's cases with budgets in tokens instead of shares of the suffix (teacher-forced KL in nats, mean over 108 cases per model). The last column recomputes, in each case, as many tokens as the cached question has, so that the question alone can be compared with token-level policies at the same cost.")
    if later_rows:
        table("later", ["Model", "Cases with replies of 2+ tokens", "No repair", "Reader attention", "Bridge", "RRF", "Reader-first (segment found)"], later_rows,
              "**Table {{#C:later}}.** The main experiment's teacher-forced KL at 10% without the first reply position, over the cases whose reply has at least two tokens.")
    v2 = load("e5_streams_qwen1.5b_v2")
    if v2 and v2.get("arrival"):
        tag = "e5v2.qwen1.5b"
        arrival_keys = [("append", "Encode on arrival, in arrival order"), ("speculate", "Encode at the canonical place, no repair"),
                        ("repair_flat@0.1", "  + repair 10%, flat prior × reader"), ("repair_sketch@0.1", "  + repair 10%, kept-query estimate × reader"),
                        ("repair_flat_first@0.1", "  + reader-first repair 10%, flat prior"), ("repair_sketch_first@0.1", "  + reader-first repair 10%, kept-query estimate"),
                        ("repair_sketch@0.2", "  + repair 20%, kept-query estimate × reader"), ("sweep", "  + sweep (exact)")]
        rows_out = [[label, values[f"{tag}.arrival.{k}.kl"], values[f"{tag}.arrival.{k}.same"], values.get(f"{tag}.arrival.{k}.tokens", "–")]
                    for k, label in arrival_keys if f"{tag}.arrival.{k}.kl" in values]
        merge_keys = [("merge", "Merge, no repair"), ("repair_flat@0.2", "  + repair 20%, flat prior × reader"),
                      ("repair_sketch@0.2", "  + repair 20%, kept-query estimate × reader"), ("repair_flat_first@0.2", "  + reader-first repair 20%, flat prior"),
                      ("repair_sketch_first@0.2", "  + reader-first repair 20%, kept-query estimate")]
        rows_out += [["**Merge**", "", "", ""]] + [[label, values[f"{tag}.merge.{k}.kl"], values[f"{tag}.merge.{k}.same"], values.get(f"{tag}.merge.{k}.tokens", "–")]
                                                   for k, label in merge_keys if f"{tag}.merge.{k}.kl" in values]
        table("streams_v2", ["Out-of-order arrival", "KL, first answer token", "Same answer", "Tokens recomputed"], rows_out,
              f"**Table {{{{#G:streams}}}}.** The runtime on Qwen2.5-1.5B ({values[f'{tag}.n']} cases per workload). Arrival: six documents arrive in random order after the question; work left after the last arrival: {values[f'{tag}.inorder']} tokens for in-order commit, {values[f'{tag}.wait']} for waiting for every document, {values[f'{tag}.last']} for encoding the last document at its place. Merge: two branches that each read three documents and wrote a note. KL from the cold pass over the canonical or merged sequence.")


def e23_e24() -> None:
    """Scattered against contiguous recomputation in our engine (E23), and a roofline estimate for a fused engine (E24)."""
    scatter = load("e23_scatter_qwen1.5b_main")
    if scatter:
        rows, tag = scatter["rows"], "e23.qwen1.5b"
        ratios = []
        for budget in scatter["budgets"]:
            ratio = statistics.median(r[f"scattered@{budget}"] / r[f"block@{budget}"] for r in rows)
            ratios.append(ratio)
            values[f"{tag}.ratio.{budget}"] = f"{ratio:.2f}"
            values[f"{tag}.runs.{budget}"] = f"{statistics.median(r[f'runs@{budget}'] for r in rows):.0f}"
            values[f"{tag}.block.{budget}"] = f"{statistics.median(r[f'block@{budget}'] for r in rows):.3f}"
        values[f"{tag}.ratio.range"] = f"{min(ratios):.2f} to {max(ratios):.2f}" if f"{min(ratios):.2f}" != f"{max(ratios):.2f}" else f"{min(ratios):.2f}"
        values[f"{tag}.n"] = str(len(rows))
        values[f"{tag}.suffix"] = f"{statistics.fmean(r['nS'] for r in rows):,.0f}"
    model = load("e24_cost_model_main")
    if not model:
        return
    gpu_label = {key: f"{f / 1e12:.0f} TFLOP/s, {b / 1e12:g} TB/s" for key, (f, b) in model["gpus"].items()}
    model_label = {"qwen1.5b": "Qwen2.5-1.5B", "qwen7b": "Qwen2.5-7B"}
    rows_out, by = [], {}
    for r in model["rows"]:
        cost = {b: (r["reader"] + r[f"scattered@{b}"]) / r["reprefill"] for b in (0.01, 0.1)}
        spread = r["scattered@0.1"] / r["block@0.1"]
        by[(r["model"], r["gpu"], r["suffix"])] = (cost, spread)
        rows_out.append([model_label[r["model"]],
                         gpu_label[r["gpu"]], f"{r['suffix']:,}", f"{1e3 * r['reprefill']:.1f}", f"{1e3 * r['reader']:.1f}",
                         f"{1e3 * r['scattered@0.01']:.1f}", f"{cost[0.01]:.3f}", f"{1e3 * r['scattered@0.1']:.1f}", f"{cost[0.1]:.3f}", f"{spread:.2f}"])
    def span(xs, digits):
        low, high = f"{min(xs):.{digits}f}", f"{max(xs):.{digits}f}"
        return low if low == high else f"{low} to {high}"
    for n in sorted({k[2] for k in by}):
        for b in (0.01, 0.1):
            values[f"e24.{n}.cost.{b}"] = span([by[k][0][b] for k in by if k[2] == n], 3 if n > 4096 and b == 0.01 else 2)
        values[f"e24.{n}.spread"] = span([by[k][1] for k in by if k[2] == n], 2)
    values["e24.spread.max_extra"] = pct(max(v[1] for v in by.values()) - 1)
    values["e24.new_input"] = str(model["new_input"])
    values["e24.prefix"] = f"{model['prefix']:,}"
    table("costmodel", ["Model", "GPU (peak)", "Suffix tokens", "Re-prefill (ms)", "Reader pass (ms)", "1% repair (ms)", "1%: share of re-prefill",
                        "10% repair (ms)", "10%: share of re-prefill", "10%: spread over block"], rows_out,
          f"**Table {{{{#A:costmodel}}}}.** A roofline estimate of repair in a fused engine, not a measurement. Every pass is charged the larger of its FLOPs at peak compute "
          f"and its memory traffic at peak bandwidth: weights in bfloat16 read once per pass, keys and values read per tile of query rows and written for every recomputed token. "
          f"The context has {model['prefix']:,} tokens before the edit; the reader pass is {model['new_input']} tokens of new input. The share of re-prefill counts the reader pass and "
          "the repair, with the repaired tokens spread evenly over the suffix; the last column is that repair's time over a block of the same size right after the edit. Real kernels "
          "run below peak and small passes pay launch costs, so real shares will be higher.")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # A table whose result file has gone (moved to a backup, say) must show up as missing in
    # build.py, not be served from an earlier run.
    for old in [*OUT.glob("*.md"), OUT / "robustness.json"]:
        old.unlink(missing_ok=True)
    e0(); e1(); e2(); e3(); e3_reader(); e2b(); e2c(); e2d(); task_accuracy(); memory(); e5(); e6(); e7(); e8()
    e2c_strata(); e9(); e10(); e11(); e12(); e21(); e2e(); e3_fresh(); needle_settings(); e15(); e14(); e1p(); e16(); e17(); e5_v2()
    e20()
    e22()
    e23_e24()
    per_experiment()
    agree = [values[f"e16.{KEYS[m]}.h4.same_under_bca"] for m in NAMES if f"e16.{KEYS[m]}.h4.same_under_bca" in values]
    if agree:
        count = {2: "two", 3: "three"}.get(len(agree), str(len(agree)))
        values["e16.h4.bca_agreement"] = (f"unchanged on all {count} models" if all(a == "yes" for a in agree)
                                          else f"different on {sum(a == 'no' for a in agree)} of {len(agree)} models")
    number = {0: "none", 1: "one", 2: "two", 3: "three"}
    count = lambda flags: "on all three models" if len(flags) == 3 and all(flags) else f"on {number[sum(flags)]} of the {number[len(flags)]} models"
    others = [values[f"e16.{KEYS[m]}.short.best_other"] for m in NAMES if values.get(f"e16.{KEYS[m]}.short.best_other")]
    values["e16.short.exceptions"] = f"; {'; '.join(others)}" if others else ""
    below = [values[f"e16.{KEYS[m]}.short.first_below_rrf"] == "yes" for m in NAMES if f"e16.{KEYS[m]}.short.first_below_rrf" in values]
    lowest = [values[f"e16.{KEYS[m]}.short.first_lowest"] == "yes" for m in NAMES if f"e16.{KEYS[m]}.short.first_lowest" in values]
    if below:
        values["e16.short.below_rrf"], values["e16.short.lowest"] = count(below), count(lowest)
    leaks = [values[f"e16.{KEYS[m]}.leak.stale.below_zero"] == "yes" for m in NAMES if f"e16.{KEYS[m]}.leak.stale.below_zero" in values]
    if leaks:
        values["e16.leak.below_zero"] = {0: "none", 1: "one", 2: "two", 3: "all three"}[sum(leaks)]
    robustness()
    scope()
    appendix_tables()
    ratios = []
    for model in NAMES:
        ours, ideal = values.get(f"e2d.{KEYS[model]}.segment_exposure.10"), values.get(f"e2c.{KEYS[model]}.kl_tf.oracle_reader.10")
        if ours and ideal:
            ratios.append(float(ours) / float(ideal))
            values[f"e2d.{KEYS[model]}.over_ideal"] = f"{ratios[-1]:.1f}"
    if ratios:
        values["e2d.over_ideal_min"], values["e2d.over_ideal_max"] = f"{min(ratios):.1f}", f"{max(ratios):.1f}"
    gains = [float(values[f"e2d.{KEYS[m]}.bridge.0.1"]) / float(values[f"e2d.{KEYS[m]}.segment_exposure.0.1"]) for m in NAMES if f"e2d.{KEYS[m]}.bridge.0.1" in values]
    if gains:
        values["e2d.gain10_min"], values["e2d.gain10_max"] = f"{min(gains):.1f}", f"{max(gains):.1f}"
    (OUT / "values.json").write_text(json.dumps(values, indent=1, sort_keys=True))
    print(f"{len(values)} values, tables: {sorted(p.name for p in OUT.glob('*.md'))}")


if __name__ == "__main__":
    main()
