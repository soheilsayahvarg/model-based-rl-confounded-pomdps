"""make_figure_mechanism.py -- the two figures the corrected paper needs.

FIGURE 1 (mechanism.pdf), two panels, is the paper's central claim after the
baseline correction:

  (a) Six candidates, none of them close to the behaviour policy. `full`'s regret
      RISES with N while the plug-in on the identical fits falls to zero. This is
      the result as originally published.
  (b) The same environment, same fits, same code path, with the behaviour clone
      added as a seventh candidate. Every arm now finds it. `full` is exactly
      right until the largest N and then degrades slightly.

The pair is the argument. Neither panel alone says anything useful: (a) alone
overstates, (b) alone hides the effect. Together they say pessimism ranks
candidates by coverage, so it is right exactly when a well-covered candidate
exists.

FIGURE 2 (separability.pdf), two panels:

  (a) The pre-fit separability test. floor/spread against horizon for four
      configurations, with the y = 1 line. Below it a valid region can order the
      candidates; above it none can, at any sample size. 17 of 20 cells sit
      above, including (4,6,2) at T=3, the grid this project used throughout.
  (b) Validity of the region actually used. pen_t1 / HW_t1 against N, with the
      y = 1 line. The calibrated regions sit below, so they do not cover the
      identified set, and the schedule climbs toward validity from underneath.

Colours are the same validated categorical slots as make_figure.py. Line style
and marker vary with colour so both figures survive greyscale printing.

Writes paper/figures/mechanism.pdf and paper/figures/separability.pdf.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", "experiments"))
OUT = os.path.normpath(os.path.join(HERE, "..", "paper", "figures"))

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.labelcolor": INK,
})

ARM_STYLE = {"full":    (C[1], "o", "-",  "pessimistic (full)"),
             "plugin":  (C[0], "s", "--", "plug-in, same fits"),
             "projall": (C[2], "^", ":",  "projected")}


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


def dress(ax):
    ax.grid(True, color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def regret_panel(ax, res, N_grid, title, note):
    for arm in ("full", "plugin", "projall"):
        col, mk, ls, lab = ARM_STYLE[arm]
        y = np.array([v[0] for v in res["regret"][arm]])
        e = np.array([v[1] for v in res["regret"][arm]])
        ax.plot(N_grid, y, ls, color=col, marker=mk, ms=3.6, lw=1.2,
                markeredgecolor="white", markeredgewidth=0.5, label=lab)
        ax.fill_between(N_grid, np.maximum(y - e, 0), y + e, color=col,
                        alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("trajectories $N$")
    ax.set_title(title + "\n" + note, color=INK, linespacing=1.35)
    dress(ax)


def figure_mechanism():
    d = load("results_bc_candidate_retest.json")
    N = d["N_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    regret_panel(axes[0], d["results"]["six candidates"], N,
                 "(a) no well-covered candidate",
                 "optimal $\\mathtt{always\\_1}$")
    regret_panel(axes[1], d["results"]["seven, with bc"], N,
                 "(b) behaviour clone added",
                 "optimal $\\mathtt{bc}$")
    axes[0].set_ylabel("regret")
    # one shared legend under both panels: an in-axes legend in (a) lands on the
    # descending plug-in and projected curves, and (b) is too empty to host it
    # without reading as though it described that panel alone.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(pad=0.4)
    p = os.path.join(OUT, "mechanism.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def figure_separability():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))

    # ---- (a) the pre-fit separability test
    d = load("results_scale_floor_ratio.json")["rows"]
    series = [((4, 6, 2), "o", "-"), ((6, 10, 3), "s", "--"),
              ((8, 10, 4), "^", ":"), ((8, 10, 3), "v", "-.")]
    ax = axes[0]
    for i, (cfg, mk, ls) in enumerate(series):
        rows = sorted([r for r in d if tuple(r["config"]) == cfg],
                      key=lambda r: r["T"])
        ax.plot([r["T"] for r in rows], [r["ratio"] for r in rows], ls,
                color=C[i], marker=mk, ms=3.6, lw=1.2,
                markeredgecolor="white", markeredgewidth=0.5,
                label="$(%d,%d,%d)$" % cfg)
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 2)))
    ax.annotate("separable below", xy=(3.05, 0.60), fontsize=7, color=INK2,
                ha="left", va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([3, 4, 6, 8, 10]); ax.set_xticklabels(["3", "4", "6", "8", "10"])
    ax.set_xlabel("horizon $T$")
    ax.set_ylabel(r"$\beta\beta_g\,/\,$value spread")
    ax.set_title("(a) can any valid region order the policies?", color=INK)
    # opaque face: the (8,10,4) and (8,10,3) curves run under this corner and a
    # frameless legend leaves the dotted green line crossing its own label.
    ax.legend(loc="upper right", handlelength=2.4, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.92)
    dress(ax)

    # ---- (b) is the region we actually use valid?
    d2 = load("results_sharp_halfwidth.json")["rows"]
    ax = axes[1]
    Ns = [r["N"] for r in d2]
    ax.plot(Ns, [r["ratio"] for r in d2], "-", color=C[1], marker="o", ms=3.6,
            lw=1.2, markeredgecolor="white", markeredgewidth=0.5,
            label="calibrated region")
    ax.axhline(1.0, color=INK2, lw=0.9, ls=(0, (4, 2)))
    ax.annotate("valid above", xy=(1.28e5, 1.06), fontsize=7, color=INK2,
                ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("trajectories $N$")
    ax.set_ylabel(r"penalty $/$ sharp half-width")
    ax.set_title("(b) does it cover the identified set?", color=INK)
    dress(ax)

    fig.tight_layout(pad=0.4)
    p = os.path.join(OUT, "separability.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for p in (figure_mechanism(), figure_separability()):
        print("wrote", p)
