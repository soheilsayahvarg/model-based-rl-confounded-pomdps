"""make_figure.py -- the paper's one figure, built from the stored JSON.

Two panels, because the paper has two claims a table states badly:

  (a) The dilution law. floor/spread against horizon on log-log axes, with a
      reference slope of -1. A table of six numbers per row cannot show that four
      configurations lie on parallel lines; a log-log plot shows it at a glance.

  (b) The conjunction. floor against confounding strength for a complete design
      and two incomplete ones. The complete design sits at zero until the
      knife-edge at 1.0; the incomplete ones rise smoothly. That contrast IS the
      two-switch result.

Colours are the validated categorical slots 1-4 (validator: all checks pass, one
contrast WARN on slots 3-4 relieved by direct legends plus the numeric table kept
in the appendix). Line style and marker vary with colour so the figure survives
greyscale printing.

Writes paper/figures/dilution.pdf.
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

# validated categorical slots 1-4 (light mode)
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


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


def panel_a(ax):
    d = load("results_horizon_dilution.json")["dilution"]
    series = [((4, 6, 2), 0.9, "o", "-"), ((4, 6, 2), 0.6, "s", "-"),
              ((4, 8, 3), 0.9, "^", "--"), ((4, 8, 3), 0.6, "v", "--")]
    for i, (cfg, cf, mk, ls) in enumerate(series):
        rows = sorted([r for r in d if tuple(r["config"]) == cfg
                       and abs(r["confound"] - cf) < 1e-9],
                      key=lambda r: r["T"])
        T = [r["T"] for r in rows]
        y = [r["ratio"] for r in rows]
        ax.plot(T, y, ls, color=C[i], marker=mk, ms=3.6, lw=1.2,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"$({cfg[0]},{cfg[1]},{cfg[2]})$, conf. {cf}")

    # measured exponents, stated rather than drawn: a dotted reference line at
    # this scale lands on top of a data series and reads as a fifth one.
    ax.annotate("measured slopes\n$-1.23$ and $-0.93$", xy=(5.9, 0.62),
                fontsize=7, color=INK2, ha="right", va="center")

    ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.annotate("floor $=$ entire value spread", xy=(1.05, 1.07), fontsize=6.5,
                color=INK2, ha="left", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.set_xticklabels(["1", "2", "3", "4", "6"])
    ax.set_xlabel("horizon $T$")
    ax.set_ylabel(r"floor $\beta\beta_g$ / value spread")
    ax.set_title("(a) the floor is diluted as $1/T$", loc="left", color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handlelength=1.8,
              labelcolor=INK2, borderpad=0.2, handletextpad=0.4,
              bbox_to_anchor=(-0.02, -0.03), ncol=2, columnspacing=0.9)


def panel_b(ax):
    d = load("results_gradient_leakage_map.json")["rows"]
    series = [((2, 6, 4), "complete", "o", "-"),
              ((4, 6, 2), "incomplete", "s", "--"),
              ((4, 8, 3), "incomplete", "^", "-.")]
    for i, (cfg, tag, mk, ls) in enumerate(series):
        rows = sorted([r for r in d
                       if (r["n_s"], r["n_o"], r["n_o0"]) == cfg],
                      key=lambda r: r["confound"])
        x = [r["confound"] for r in rows]
        # plotted as measured: values below 1e-30 ARE zero to machine precision,
        # and a symlog axis puts them at 0 rather than clamping them onto a log
        # decade they never occupied.
        y = [r["floor"] if r["floor"] > 1e-20 else 0.0 for r in rows]
        ax.plot(x, y, ls, color=C[i], marker=mk, ms=3.6, lw=1.2,
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"$({cfg[0]},{cfg[1]},{cfg[2]})$ {tag}")

    ax.set_yscale("symlog", linthresh=1e-2, linscale=0.35)
    ax.set_ylim(-1e-3, 2)
    ax.set_yticks([0, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["$0$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"])
    ax.set_xlabel("confounding strength")
    ax.set_ylabel(r"floor $\beta\beta_g$")
    ax.set_title("(b) the floor needs both switches", loc="left", color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("exactly $0$ until the\nknife-edge at $1$", xy=(0.08, 0.0),
                xytext=(0.08, 2.4e-3), fontsize=6.5, color=INK2,
                ha="left", va="bottom")
    ax.legend(frameon=False, loc="upper left", handlelength=2.2,
              labelcolor=INK2, borderpad=0.2, handletextpad=0.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.45))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout(pad=0.4, w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"dilution.{ext}"), dpi=220,
                    bbox_inches="tight", facecolor="white")
    print("written:", os.path.join(OUT, "dilution.pdf"))


if __name__ == "__main__":
    main()
