#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report figures, drawn from the pipeline's output files.

Needs matplotlib, which the pipeline itself does not. From the repo root:

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python figures/make_figures.py

Run the pipeline first (stages 03, 07, 08, 09, 10, 11 are read). Writes a PNG
and a PDF of each figure next to this script.

  fig1_alcove_benchmark   people vs ALCOVE on the six SHJ types, attention on/frozen
  fig2_rulex_benchmark    RULEX against Nosofsky et al.'s (1994) published values
  fig3_nonce_profiles     both models on the nonce items, against the real /at/
                          words and the Tolerance Principle, per vocabulary size
  fig4_learner_split      teşaat, one bar per learner answer: graded vs all-or-none
"""
import collections
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
OUT = os.path.join(HERE, "..", "output")
sys.path.insert(0, SRC)
import common as C                      # noqa: E402
import training_data as T               # noqa: E402

# Reference palette (dataviz skill), light mode; validated: the three-slot set
# all-pairs, the six-slot set adjacent. Pale slots are always direct-labelled.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ALCOVE, RULEX, TP = SLOTS[0], SLOTS[1], SLOTS[2]
MAIN_RULEX = "pstor=0.80 capac=0.40"
LEVEL_TICKS = ["teşaat\nvowel", "teşahat\n/h/", "teşakat\n/k/", "teşasat…\nno cue"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 6, "legend.frameon": False,
})


def read(name):
    return C.read_tsv(os.path.join(OUT, name))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "%s.%s" % (name, ext)), dpi=200,
                    bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("  wrote figures/%s.png / .pdf" % name)


def spread(ys, gap):
    """Nudge label heights apart so no two are closer than gap (order kept)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# ------------------------------------------------------------------ figure 1
def fig1():
    rows = read("08_alcove_shj.tsv")
    types = ["I", "II", "III", "IV", "V", "VI"]
    series = {(col, t): [] for col in ("human", "alcove", "alcove_frozen") for t in types}
    for r in rows:
        for col in ("human", "alcove", "alcove_frozen"):
            series[(col, r["type"])].append((int(r["block"]), float(r[col])))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
    panels = [("human", "People (Nosofsky et al. 1994)"),
              ("alcove", "ALCOVE, fitted to people"),
              ("alcove_frozen", "ALCOVE, attention frozen")]
    for ax, (col, title) in zip(axes, panels):
        ends = []
        for t, colour in zip(types, SLOTS):
            pts = sorted(series[(col, t)])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour,
                    label="Type " + t)
            ends.append(pts[-1][1])
        for t, y in zip(types, spread(ends, 0.02)):
            ax.text(17.1, y, t, color=INK2, va="center", fontsize=7.5)
            ax.plot([16.2, 16.9], [ends[types.index(t)], y], color=AXIS, lw=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Block (16 trials)")
        ax.set_xlim(1, 18.2)
        ax.set_xticks([1, 4, 8, 12, 16])
        ax.set_ylim(0, 0.52)
    axes[0].set_ylabel("P(error)")
    axes[0].legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.12), fontsize=8,
                   handlelength=1.4, columnspacing=1.0)
    fig.suptitle("ALCOVE fitted to people gets the SHJ ordering, which needs attention "
                 "learning; it makes Type VI too easy", x=0.06, ha="left", y=1.12,
                 fontsize=11)
    save(fig, "fig1_alcove_benchmark")


# ------------------------------------------------------------------ figure 2
def fig2():
    rows = read("10_rulex_benchmarks.tsv")
    ms = [r for r in rows if r["benchmark"] == "medin_schaffer"]
    shj = [r for r in rows if r["benchmark"] == "shj"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.5),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))
    obs = [float(r["observed"]) for r in ms]
    a.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=0)
    a.scatter(obs, [float(r["paper"]) for r in ms], s=38, facecolor="none",
              edgecolor=INK2, lw=1.2, label="Nosofsky et al. (1994) RULEX", zorder=3)
    a.scatter(obs, [float(r["ours"]) for r in ms], s=30, color=RULEX,
              edgecolor=SURFACE, lw=1, label="our RULEX", zorder=4)
    a.set_xlabel("Observed P(Category A), Medin & Schaffer (1978)")
    a.set_ylabel("Predicted P(Category A)")
    a.set_xlim(-0.03, 1.03)
    a.set_ylim(-0.03, 1.03)
    a.grid(axis="x", color=GRID, lw=0.6)
    a.legend(loc="upper left", fontsize=8, handletextpad=0.3)
    a.set_title("5-4 structure: 16 items", loc="left")

    types = [r["item"] for r in shj]
    vals = [float(r["ours"]) for r in shj]
    bars = b.bar(types, vals, color=RULEX, width=0.62)
    for rect, v in zip(bars, vals):
        b.text(rect.get_x() + rect.get_width() / 2, v + 0.008, "%.2f" % v,
               ha="center", va="bottom", fontsize=8, color=INK2)
    b.set_ylabel("Mean P(error), 16 blocks")
    b.set_xlabel("SHJ type")
    b.set_ylim(0, 0.45)
    b.set_title("Six SHJ types (our RULEX): I < II < III–V < VI", loc="left")
    fig.suptitle("RULEX matches the published benchmarks", x=0.06, ha="left",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    save(fig, "fig2_rulex_benchmark")


# ------------------------------------------------------------------ figure 3
def model_means(rows, key_fn):
    by = collections.defaultdict(list)
    for r in rows:
        by[key_fn(r)].append(float(r["p_ler"]))
    return {k: (statistics.mean(v), statistics.stdev(v) / len(v) ** 0.5) for k, v in by.items()}


def fig3():
    alc = model_means(read("09_alcove_nonce.tsv"),
                      lambda r: (r["laterals"], int(r["vocab"]), r["cue_level"]))
    rlx = model_means([r for r in read("11_rulex_nonce.tsv") if r["params"] == MAIN_RULEX],
                      lambda r: (r["laterals"], int(r["vocab"]), r["cue_level"]))
    counts = T.lexicon_counts()
    lex = [counts[lv][1] / counts[lv][0] for lv in T.LEVELS]
    tp = [C.tolerance_verdict(*counts[lv]) for lv in T.LEVELS]
    xs = list(range(len(T.LEVELS)))

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.0), sharex=True, sharey=True)
    for row, lat in enumerate(T.LATERALS):
        for col, n in enumerate(T.VOCAB_SIZES):
            ax = axes[row][col]
            ax.plot(xs, lex, color=INK2, ls="--", lw=1.2, marker="o", mfc=SURFACE,
                    mec=INK2, label="real /at/ words (dictionary)", zorder=2)
            for i, v in enumerate(tp):
                x = i + 0.22
                if v.startswith("-ler"):
                    ax.plot(x, 1, marker="D", color=TP, ms=6, zorder=3)
                elif v.startswith("-lar"):
                    ax.plot(x, 0, marker="D", color=TP, ms=6, zorder=3)
                else:
                    ax.plot([x, x], [0, 1], color=TP, lw=1.1, ls=(0, (2, 2)), zorder=1)
                    ax.plot([x - 0.06, x + 0.06], [0, 0], color=TP, lw=1.1, zorder=1)
                    ax.plot([x - 0.06, x + 0.06], [1, 1], color=TP, lw=1.1, zorder=1)
            for model, col_, dx, lab in ((alc, ALCOVE, -0.08, "ALCOVE"),
                                         (rlx, RULEX, 0.08, "RULEX")):
                m = [model[(lat, n, lv)][0] for lv in T.LEVELS]
                se = [model[(lat, n, lv)][1] for lv in T.LEVELS]
                ax.errorbar([x + dx for x in xs], m, yerr=se, color=col_, marker="o",
                            mec=SURFACE, mew=1, capsize=0, elinewidth=1.2, label=lab,
                            zorder=4)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-0.4, 3.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(LEVEL_TICKS, fontsize=8)
            if row == 0:
                ax.set_title("%s-word vocabulary" % format(n, ","), loc="left")
            if col == 0:
                ax.set_ylabel("P(-ler)\n%s lateral-final words" % lat)
            if row == 1 and col == 2:
                a_end = [alc[(lat, n, lv)][0] for lv in T.LEVELS]
                r_end = [rlx[(lat, n, lv)][0] for lv in T.LEVELS]
                ax.annotate("ALCOVE", (0 - 0.08, a_end[0]), xytext=(0.35, a_end[0] + 0.12),
                            color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=ALCOVE, lw=0.8))
                ax.annotate("RULEX", (0 + 0.08, r_end[0]), xytext=(0.45, 0.3),
                            color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=RULEX, lw=0.8))
                ax.annotate("dictionary", (1, lex[1]), xytext=(1.35, 0.62),
                            color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK2, ls="--", marker="o", mfc=SURFACE, mec=INK2),
               Line2D([], [], color=TP, ls="none", marker="D"),
               Line2D([], [], color=TP, ls=(0, (2, 2)), lw=1.1),
               Line2D([], [], color=ALCOVE, marker="o", mec=SURFACE),
               Line2D([], [], color=RULEX, marker="o", mec=SURFACE)]
    labels = ["real /at/ words (dictionary)", "Tolerance Principle: rule predicted",
              "Tolerance Principle: no rule (unpredictable)", "ALCOVE", "RULEX"]
    fig.legend(handles, labels, ncol=5, loc="upper left", bbox_to_anchor=(0.06, 1.0),
               fontsize=8, handletextpad=0.4, columnspacing=1.4)
    fig.suptitle("Nonce items: ALCOVE follows the dictionary; RULEX says -lar once the "
                 "vocabulary is large", x=0.06, ha="left", y=1.04, fontsize=11)
    fig.text(0.06, -0.04, "Points: mean over simulated learners (ALCOVE 20, RULEX 100); "
             "error bars: ±1 SE. Dictionary rates from the cleaned /at/ class (stage 03).\n"
             "Both models trained 80 epochs; means checked stable to 160 (RULEX) and "
             "320 (ALCOVE) epochs.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig3_nonce_profiles")


# ------------------------------------------------------------------ figure 4
def fig4(vocab=5000, lat="with", level="hiatus"):
    alc = [float(r["p_ler"]) for r in read("09_alcove_nonce.tsv")
           if r["laterals"] == lat and int(r["vocab"]) == vocab and r["cue_level"] == level]
    rlx = [float(r["p_ler"]) for r in read("11_rulex_nonce.tsv")
           if r["params"] == MAIN_RULEX and r["laterals"] == lat
           and int(r["vocab"]) == vocab and r["cue_level"] == level]
    edges = [i / 10 for i in range(11)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), sharey=True)
    for ax, vals, col, name in ((axes[0], alc, ALCOVE, "ALCOVE"),
                                (axes[1], rlx, RULEX, "RULEX")):
        n = len(vals)
        pct = [0] * 10
        for v in vals:
            pct[min(int(v * 10), 9)] += 100 / n
        ax.bar([e + 0.05 for e in edges[:-1]], pct, width=0.085, color=col)
        for i, p in enumerate(pct):
            if p:
                ax.text(edges[i] + 0.05, p + 1.5, "%.0f%%" % p, ha="center",
                        fontsize=7.5, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .2, .4, .6, .8, 1])
        ax.set_xlabel("One learner's P(-ler) for teşaat")
        ax.set_title("%s  (%d learners)" % (name, n), loc="left")
    axes[0].set_ylabel("% of learners")
    axes[0].set_ylim(0, 105)
    fig.suptitle("Same item, different kind of answer: ALCOVE learners are graded, "
                 "RULEX learners all-or-none", x=0.06, ha="left", y=1.04, fontsize=11)
    fig.text(0.06, -0.06, "%s-word vocabularies, %s lateral-final words, main parameters."
             % (format(vocab, ","), lat), color=MUTED, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig4_learner_split")


def main():
    print("figures:")
    fig1()
    fig2()
    fig3()
    fig4()


if __name__ == "__main__":
    main()
