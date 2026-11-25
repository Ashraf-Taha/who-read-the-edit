"""Figures for the paper, drawn from experiments/results/*.json.

    python paper/make_figures.py

Colours are the first six slots of the validated categorical palette, in fixed order,
with the paper's own policy (bridge) in slot 1.
"""
from __future__ import annotations

import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "generated"

INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [
    ("bridge", "Bridge (exposure × reader)", "#2a78d6", "o"),
    ("boundary", "Boundary window", "#eb6834", "s"),
    ("tail", "Tail window", "#1baf7a", "D"),
    ("probe1", "First-layer deviation", "#eda100", "^"),
    ("reader", "Reader attention", "#e87ba4", "v"),
    ("oracle", "True cell deviation (needs the re-prefill)", "#008300", "P"),
    ("exposure", "Recorded exposure", "#6250d6", "X"),
]
MODELS = [("qwen0.5b", "Qwen2.5-0.5B"), ("qwen1.5b", "Qwen2.5-1.5B"), ("smol1.7b", "SmolLM2-1.7B")]
PLACES = [("adjacent", "right after\nthe edit"), ("middle", "in the\nmiddle"), ("end", "at the\nend")]


def complete(rows: list[dict]) -> bool:
    counts = [sum(r["kind"] == "gold" and r["place"] == p for r in rows) for p, _ in PLACES]
    return min(counts) > 0 and len(set(counts)) == 1


def bootstrap(values: list[float], clusters: list | None = None, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval; with `clusters`, whole clusters (questions) are resampled."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = arr[rng.integers(0, len(arr), (4000, len(arr)))].mean(axis=1)
    else:
        labels = {label: index for index, label in enumerate(dict.fromkeys(clusters))}
        ids = np.array([labels[c] for c in clusters])
        sums, counts = np.bincount(ids, weights=arr, minlength=len(labels)), np.bincount(ids, minlength=len(labels))
        draws = rng.integers(0, len(labels), (4000, len(labels)))
        means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(low), float(high)


def log_axis(ax) -> None:
    """Label a logarithmic y-axis at 1, 2 and 5 times each power of ten, in plain decimals."""
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


BANDS = ("bridge", "reader", "oracle")


def placement_figure() -> pathlib.Path | None:
    panels = []
    for key, name in MODELS:
        path = RESULTS / f"e2_healing_{key}_main.json"
        if path.exists() and complete(rows := json.loads(path.read_text())["rows"]):
            panels.append((name, rows))
    if not panels:
        return None
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.7), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    series = [("stale", "No repair", MUTED, "_")] + SERIES
    width = 0.7 / len(series)
    for ax, (name, rows) in zip(axes, panels):
        for index, (key, label, colour, marker) in enumerate(series):
            for x, (place, _) in enumerate(PLACES):
                part = [r["delete"]["results"][key]["0" if key == "stale" else "0.1"]["kl_last"] for r in rows
                        if r["kind"] == "gold" and r["place"] == place]
                mean, low, high = bootstrap(part)
                position = x - 0.35 + width * (index + 0.5)
                ax.errorbar(position, mean, yerr=[[mean - low], [high - mean]], color=colour, marker=marker, markersize=4.5,
                            linewidth=0, elinewidth=0.9, capsize=0, markeredgecolor="white" if marker != "_" else colour,
                            markeredgewidth=0.6 if marker != "_" else 1.6, label=label if x == 0 else None,
                            zorder=3 if key == "bridge" else 2)
        ax.set_yscale("log")
        log_axis(ax)
        ax.set_xticks(range(len(PLACES)), [label for _, label in PLACES])
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0)
        ax.set_xlim(-0.5, len(PLACES) - 0.5)
    axes[0].set_ylabel("KL from re-prefill (nats, log)")
    fig.text(0.5, -0.02, "Where the cached question sits in the suffix", ha="center", color=SECONDARY, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15), fontsize=7.5,
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf, png = OUT / "fig_placement.pdf", OUT / "fig_placement.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


def budget_figure() -> pathlib.Path | None:
    panels = []
    for key, name in MODELS:
        path = RESULTS / f"e2_healing_{key}_main.json"
        if path.exists() and complete(rows := json.loads(path.read_text())["rows"]):
            panels.append((name, [r for r in rows if r["kind"] == "gold"]))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.4), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (name, rows) in zip(axes, panels):
        budgets = sorted(rows[0]["delete"]["results"]["bridge"], key=float)
        x = [100 * float(b) for b in budgets]
        for key, label, colour, marker in reversed(SERIES):
            questions = [r["name"].split("/")[1] for r in rows]
            summary = [bootstrap([r["delete"]["results"][key][b]["kl_last"] for r in rows], questions) for b in budgets]
            y = [mean for mean, _, _ in summary]
            if key in BANDS:
                ax.fill_between(x, [low for _, low, _ in summary], [high for _, _, high in summary], color=colour, alpha=0.13,
                                linewidth=0, zorder=1)
            ax.plot(x, y, color=colour, marker=marker, markersize=5, linewidth=1.5, label=label,
                    zorder=3 if key == "bridge" else 2, markeredgecolor="white", markeredgewidth=0.8)
        stale = statistics.fmean(r["delete"]["results"]["stale"]["0"]["kl_last"] for r in rows)
        ax.axhline(stale, color=MUTED, linestyle=(0, (3, 2)), linewidth=1.0, label="No repair", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x, [f"{v:g}%" for v in x])
        ax.xaxis.set_minor_locator(NullLocator())
        log_axis(ax)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0)
    axes[0].set_ylabel("KL from re-prefill (nats, log)")
    fig.text(0.5, -0.02, "Share of suffix tokens recomputed", ha="center", color=SECONDARY, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(label) for _, label, _, _ in SERIES] + [labels.index("No repair")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.13), fontsize=7.5, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    pdf, png = OUT / "fig_budget.pdf", OUT / "fig_budget.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


READER_POLICIES = [
    ("stale", "No repair", "baseline"), ("boundary", "Boundary window", "baseline"), ("probe1", "First-layer deviation", "baseline"),
    ("reader", "Reader attention", "baseline"), ("exposure", "Recorded exposure only", "baseline"),
    ("rrf", "Rank fusion (RRF)", "ours"), ("bridge", "Bridge score", "ours"),
    ("segment_exposure", "Reader-first (segment found)", "ours"),
    ("question_exposure", "Cached question (given), then exposure", "reference"), ("oracle_reader", "True deviation × reader (ideal)", "reference"),
]
GROUPS = {"baseline": ("Baselines", MUTED, "o"), "ours": ("Exposure and reader attention", "#2a78d6", "D"),
          "reference": ("Uses the prompt's roles or the re-prefill", "#008300", "s")}


def reader_figure() -> pathlib.Path | None:
    panels = []
    for key, name in MODELS:
        c, d = RESULTS / f"e2c_ablation_{key}_main.json", RESULTS / f"e2d_reader_first_{key}_main.json"
        if not (c.exists() and d.exists()):
            continue
        ablation = [r for r in json.loads(c.read_text())["rows"] if r["kind"] == "gold"]
        reader = json.loads(d.read_text())["rows"]
        if len(reader) < len(ablation):
            continue
        panels.append((name, ablation, {r["name"]: r for r in reader}))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 3.0), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    y = list(range(len(READER_POLICIES)))[::-1]
    for ax, (name, ablation, by_name) in zip(axes, panels):
        questions = [r["name"].split("/")[1] for r in ablation]
        for row, (key, label, group) in zip(y, READER_POLICIES):
            if key.startswith("segment_"):
                xs = [by_name[r["name"]]["results"][key]["0.1"]["kl_tf"] for r in ablation]
            else:
                xs = [r["delete"]["results"][key]["0" if key == "stale" else "0.1"]["kl_tf"] for r in ablation]
            mean, low, high = bootstrap(xs, questions)
            _, colour, marker = GROUPS[group]
            ax.errorbar(mean, row, xerr=[[mean - low], [high - mean]], color=colour, marker=marker, markersize=5, linewidth=0,
                        elinewidth=1.0, capsize=0, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="x", which="major", color=GRID, linewidth=0.6)
        ax.grid(axis="x", which="minor", color=GRID, linewidth=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0, labelsize=7)
    axes[0].set_yticks(y, [label for _, label, _ in READER_POLICIES])
    fig.text(0.62, -0.02, "Teacher-forced KL from the re-prefill at 10% of the suffix (nats, log)", ha="center", color=SECONDARY, fontsize=8)
    handles = [plt.Line2D([], [], color=colour, marker=marker, linewidth=0, markersize=5) for _, colour, marker in GROUPS.values()]
    fig.legend(handles, [label for label, _, _ in GROUPS.values()], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.6, 1.07), fontsize=7.5)
    fig.tight_layout()
    pdf, png = OUT / "fig_reader.pdf", OUT / "fig_reader.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


MAIN_SERIES = [
    ("segment_exposure", "Reader-first (segment found)", "#2a78d6", "D"),
    ("bridge", "Bridge (exposure × reader)", "#eb6834", "o"),
    ("rrf", "Rank fusion (RRF)", "#1baf7a", "s"),
    ("reader", "Reader attention", "#e87ba4", "v"),
    ("segment_bridge", "Reader-first (segment found), then bridge", "#6250d6", "X"),
    ("boundary", "Boundary window", "#eda100", "^"),
    ("oracle_reader", "True deviation × reader (ideal)", "#008300", "P"),
]


def main_panels() -> list[tuple[str, list[dict], dict]]:
    panels = []
    for key, name in MODELS:
        c, d = RESULTS / f"e2c_ablation_{key}_main.json", RESULTS / f"e2d_reader_first_{key}_main.json"
        if not (c.exists() and d.exists()):
            continue
        ablation = [r for r in json.loads(c.read_text())["rows"] if r["kind"] == "gold"]
        reader = {r["name"]: r for r in json.loads(d.read_text())["rows"]}
        if all(r["name"] in reader for r in ablation):
            panels.append((name, ablation, reader))
    return panels


def value(row: dict, reader: dict, key: str, budget: str) -> float:
    if key.startswith("segment_"):
        return reader[row["name"]]["results"][key][budget]["kl_tf"]
    return row["delete"]["results"][key]["0" if key == "stale" else budget]["kl_tf"]


def main_budget_figure() -> pathlib.Path | None:
    panels = main_panels()
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.5), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    budgets = ["0.05", "0.1", "0.2"]
    x = [100 * float(b) for b in budgets]
    for ax, (name, rows, reader) in zip(axes, panels):
        questions = [r["name"].split("/")[1] for r in rows]
        for key, label, colour, marker in reversed(MAIN_SERIES):
            summary = [bootstrap([value(r, reader, key, b) for r in rows], questions) for b in budgets]
            if key in ("segment_exposure", "bridge", "oracle_reader"):
                ax.fill_between(x, [lo for _, lo, _ in summary], [hi for _, _, hi in summary], color=colour, alpha=0.13, linewidth=0, zorder=1)
            ax.plot(x, [m for m, _, _ in summary], color=colour, marker=marker, markersize=5, linewidth=1.5, label=label,
                    zorder=3 if key == "segment_exposure" else 2, markeredgecolor="white", markeredgewidth=0.8)
        stale = statistics.fmean(value(r, reader, "stale", "0") for r in rows)
        ax.axhline(stale, color=MUTED, linestyle=(0, (3, 2)), linewidth=1.0, label="No repair", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x, [f"{v:g}%" for v in x])
        ax.xaxis.set_minor_locator(NullLocator())
        log_axis(ax)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0)
    axes[0].set_ylabel("Teacher-forced KL (nats, log)")
    fig.text(0.5, -0.02, "Share of suffix tokens recomputed", ha="center", color=SECONDARY, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(label) for _, label, _, _ in MAIN_SERIES] + [labels.index("No repair")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.17), fontsize=7.5, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout()
    pdf, png = OUT / "fig_budget.pdf", OUT / "fig_budget.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


def main_placement_figure() -> pathlib.Path | None:
    panels = main_panels()
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.7), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    series = [("stale", "No repair", MUTED, "_")] + MAIN_SERIES
    width = 0.7 / len(series)
    for ax, (name, rows, reader) in zip(axes, panels):
        for index, (key, label, colour, marker) in enumerate(series):
            for x, (place, _) in enumerate(PLACES):
                part = [r for r in rows if r["place"] == place]
                mean, low, high = bootstrap([value(r, reader, key, "0.1") for r in part], [r["name"].split("/")[1] for r in part])
                ax.errorbar(x - 0.35 + width * (index + 0.5), mean, yerr=[[mean - low], [high - mean]], color=colour, marker=marker,
                            markersize=4.5, linewidth=0, elinewidth=0.9, capsize=0, markeredgecolor="white" if marker != "_" else colour,
                            markeredgewidth=0.6 if marker != "_" else 1.6, label=label if x == 0 else None,
                            zorder=3 if key == "segment_exposure" else 2)
        ax.set_yscale("log")
        log_axis(ax)
        ax.set_xticks(range(len(PLACES)), ["next to\nthe edit", "in the\nmiddle", "at the\nend"])
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0, labelsize=7)
        ax.set_xlim(-0.5, len(PLACES) - 0.5)
    axes[0].set_ylabel("Teacher-forced KL (nats, log)")
    fig.text(0.5, -0.02, "Where the cached question sits in the suffix", ha="center", color=SECONDARY, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.16), fontsize=7.5,
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    pdf, png = OUT / "fig_placement.pdf", OUT / "fig_placement.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


FAMILIES = [("main", "Main experiment (3 small models)", "#2a78d6", "o"), ("qwen7b", "Qwen2.5-7B", "#eb6834", "s"),
            ("long", "4k–16k-token contexts", "#1baf7a", "D"), ("agent", "Agent traces", "#e87ba4", "v"),
            ("needle", "Needle erasure", "#6250d6", "X"), ("replication", "Replication (hypotheses fixed in advance)", "#eda100", "^"),
            ("mistral7b", "Mistral-7B", "#008300", "P"), ("traces", "Real agent transcripts", "#e34948", "h")]
ROBUST_ROWS = [("bridge", "Bridge score"), ("rrf", "Rank fusion (RRF)"), ("segment_exposure", "Reader-first (segment found)"),
               ("exposure", "Recorded exposure"), ("reader", "Reader attention"), ("tail", "Tail window"), ("boundary", "Boundary window")]


def robust_figure() -> pathlib.Path | None:
    path = OUT / "robustness.json"
    if not path.exists():
        return None
    settings = json.loads(path.read_text())
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    rows = {key: len(ROBUST_ROWS) - 1 - i for i, (key, _) in enumerate(ROBUST_ROWS)}
    present = [f for f in FAMILIES if any(st["experiment"] == f[0] for st in settings)]
    offsets = {f[0]: (-0.32 + 0.64 * i / max(1, len(present) - 1)) for i, f in enumerate(present)}
    shares = {key: [] for key, _ in ROBUST_ROWS}
    for st in settings:
        fam = next((f for f in FAMILIES if f[0] == st["experiment"]), FAMILIES[0])
        for key, value in st["means"].items():
            if key not in rows or st["stale"] <= 0:
                continue
            share = 100 * (1 - value / st["stale"])
            shares[key].append(share)
            ax.plot(max(share, -20), rows[key] + offsets.get(fam[0], 0.0), linestyle="none", marker=fam[3], markersize=4.8, color=fam[2],
                    alpha=0.9, zorder=3)
    for key, xs in shares.items():
        if xs:
            median = statistics.median(xs)
            ax.plot([median, median], [rows[key] - 0.42, rows[key] + 0.42], color=INK, linewidth=1.5, zorder=4)
    ax.axvline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.set_xlim(-22, 102)
    ax.set_xticks([-20, 0, 20, 40, 60, 80, 100], ["≤ −20%", "0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_yticks([rows[key] for key, _ in ROBUST_ROWS], [label for _, label in ROBUST_ROWS])
    ax.set_ylim(-0.6, len(ROBUST_ROWS) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    for y in range(len(ROBUST_ROWS) - 1):
        ax.axhline(y + 0.5, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Share of the no-repair KL removed, one dot per setting (black bar = median over settings)")
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=5) for _, _, c, m in present]
    fig.legend(handles, [label for _, label, _, _ in present], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.1),
               fontsize=7.2, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    pdf, png = OUT / "fig_robust.pdf", OUT / "fig_robust.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


ECDF_SERIES = [("rrf", "Rank fusion (RRF)", "#1baf7a", "-"), ("bridge", "Bridge score", "#2a78d6", "-"),
               ("segment_exposure", "Reader-first (segment found)", "#eb6834", "-"), ("exposure", "Recorded exposure", "#6250d6", "--"),
               ("reader", "Reader attention", "#e87ba4", "--"), ("oracle_reader", "Oracle score (needs the re-prefill)", "#0b0b0b", ":")]


def ecdf_figure() -> pathlib.Path | None:
    import numpy as np

    panels = main_panels()
    if not panels:
        return None
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.4), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (name, rows, reader) in zip(axes, panels):
        stale = np.array([value(r, reader, "stale", "0") for r in rows])
        for key, label, colour, style in ECDF_SERIES:
            ratio = np.sort(np.array([value(r, reader, key, "0.1") for r in rows]) / np.maximum(stale, 1e-12))
            ax.step(ratio, np.arange(1, len(ratio) + 1) / len(ratio), where="post", color=colour, linestyle=style, linewidth=1.4, label=label)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 2)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(name, fontsize=8.5, color=INK, loc="left")
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(which="both", length=0)
    axes[0].set_ylabel("Share of cases")
    fig.text(0.5, -0.03, "KL after a 10% repair divided by the KL without repair, per case (log)", ha="center", color=SECONDARY, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16), fontsize=7.5)
    fig.tight_layout()
    pdf, png = OUT / "fig_ecdf.pdf", OUT / "fig_ecdf.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png


def paths_figure() -> pathlib.Path:
    """Schematic of the two-hop path: edited span -> cached token -> new input."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "text.color": INK})
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")
    accent, pale, stale = "#2a78d6", "#eef3fb", "#f3f2ee"
    box = lambda x, y, w, h, edge, face: ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                                                     linewidth=0.9, edgecolor=edge, facecolor=face))
    box(1, 12, 12, 9, AXIS, "white"); ax.text(7, 16.5, "P\nprefix", ha="center", va="center", fontsize=7.5)
    box(15, 12, 12, 9, AXIS, stale); ax.text(21, 16.5, "E\nedited span", ha="center", va="center", fontsize=7.5, color=SECONDARY)
    box(29, 10, 55, 13, AXIS, "white"); ax.text(32.6, 16.5, "S\nsuffix", ha="center", va="center", fontsize=7.5)
    box(86, 12, 13, 9, accent, pale); ax.text(92.5, 16.5, "Q\nnew input", ha="center", va="center", fontsize=7.5)
    tokens = [(43, "document\ntoken", 2.0, 0.6, False), (59, "cached\nquestion", 2.4, 2.4, True), (75, "document\ntoken", 0.6, 2.0, False)]
    for x, name, into, out, reader in tokens:
        box(x - 6, 13.2, 12, 6.6, accent if reader else AXIS, pale if reader else "white")
        ax.text(x, 16.5, name, ha="center", va="center", fontsize=6.6)
        colour = accent if reader else MUTED
        ax.add_patch(FancyArrowPatch((21, 21.6), (x, 20.4), connectionstyle="arc3,rad=-0.28", arrowstyle="-|>", mutation_scale=7,
                                     linewidth=into, color=colour, zorder=3, shrinkA=1, shrinkB=1))
        ax.add_patch(FancyArrowPatch((x, 12.6), (92.5, 11.4), connectionstyle="arc3,rad=0.28", arrowstyle="-|>", mutation_scale=7,
                                     linewidth=out, color=colour, zorder=3, shrinkA=1, shrinkB=1))
    ax.text(48, 32.2, "exposure $m_j$: attention token $j$ paid to E when encoded (recorded at prefill)", ha="center", va="center", fontsize=7)
    ax.text(62, 1.3, "reader attention $\\rho_j$: attention Q pays to token $j$ (one pass of Q over the stale cache)", ha="center", va="center", fontsize=7)
    fig.tight_layout(pad=0.2)
    pdf, png = OUT / "fig_paths.pdf", OUT / "fig_paths.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return png

if __name__ == "__main__":
    print(paths_figure())
    print(robust_figure())
    print(ecdf_figure())
