"""
plot_results.py — headline figures + CSV table exports into Phase_3/results/.

Figures (PNG, 150 dpi, light surface, validated categorical palette):
  fig1_naive_bias.png        naive confounding bias per policy (toy + big-env
                             population-naive kappa contrast)
  fig2_bridge_recovery.png   toy bridge L2 recovery vs N (log-log) + N^-1/2 guide
  fig3_value_debias.png      toy |value bias|: naive vs plug-in across N
  fig4_pessimism.png         joint coverage and selection regret vs width
                             multiplier c: vanilla Eq.-17 vs signal-projected
Colors: validated reference palette (blue #2a78d6, red #e34948, aqua #1baf7a).
Identity is never color-alone: direct labels / distinct markers + legend.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", "experiments"))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, RED, AQUA, VIOLET = "#2a78d6", "#e34948", "#1baf7a", "#4a3aa7"
INK, MUT = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUT, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


r1 = load("results_phase1.json")
r2 = load("results_phase2.json")
r34 = load("results_phase34.json")
rb = load("results_bigenv.json")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(FIG, name))


# ---------------------------------------------------------------- fig 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
toy = r1["toy"]["A4_naive_bias"]
pols = list(toy)
x = np.arange(len(pols))
ax1.bar(x, [toy[p]["bias"] for p in pols], 0.55, color=RED)
ax1.set_xticks(x, [p.replace("_", "\n") for p in pols])
ax1.set_ylabel("bias of naive evaluator")
ax1.set_title("Toy POMDP: naive bias (N=200k, structural)")
ax1.axhline(0, color=MUT, lw=0.8)

big = r1["big"]["B5_population_naive_structural_bias"]
pols_b = list(big["kappa=1.0"])
xb = np.arange(len(pols_b))
w = 0.38
ax2.bar(xb - w / 2, [big["kappa=1.0"][p]["bias"] for p in pols_b], w,
        color=RED, label="confounded ($\\kappa$=1)")
ax2.bar(xb + w / 2, [big["kappa=0.0"][p]["bias"] for p in pols_b], w,
        color=BLUE, label="latent-blind ($\\kappa$=0)")
ax2.set_xticks(xb, [p.replace("_", "\n") for p in pols_b])
ax2.set_title("Benchmark 720$\\times$2: population naive bias")
ax2.axhline(0, color=MUT, lw=0.8)
ax2.legend(loc="upper right")
save(fig, "fig1_naive_bias.png")

# ---------------------------------------------------------------- fig 2
sweep = r2["P2_scaling_sweep"]
Ns = sorted(int(n) for n in sweep)
errR = [sweep[str(n)]["errR_total"] for n in Ns]
errD = [sweep[str(n)]["errD_total"] for n in Ns]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
ax.loglog(Ns, errR, "o-", color=BLUE, lw=2, ms=5, label="$b_R$ (reward bridge)")
ax.loglog(Ns, errD, "s-", color=AQUA, lw=2, ms=5, label="$b_D$ (dynamic bridge)")
guide = errR[0] * (np.array(Ns) / Ns[0]) ** -0.5
ax.loglog(Ns, guide, "--", color=MUT, lw=1, label="$N^{-1/2}$ guide")
ax.set_xlabel("N (trajectories)")
ax.set_ylabel("L2 recovery error vs oracle")
ax.set_title("Toy: bridge recovery vs N (3 seeds)")
ax.legend()
save(fig, "fig2_bridge_recovery.png")

# ---------------------------------------------------------------- fig 3
deb = r34["V2_plugin_vs_naive_bias"]
Ns3 = sorted(int(n) for n in deb)
mean_nv = [np.mean([deb[str(n)][p]["abs_bias_naive"] for p in deb[str(n)]])
           for n in Ns3]
mean_pl = [np.mean([deb[str(n)][p]["abs_bias_plugin"] for p in deb[str(n)]])
           for n in Ns3]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
x = np.arange(len(Ns3))
ax.bar(x - 0.19, mean_nv, 0.38, color=RED, label="naive (uncorrected)")
ax.bar(x + 0.19, mean_pl, 0.38, color=BLUE, label="bridge plug-in")
for i, (a, b) in enumerate(zip(mean_nv, mean_pl)):
    ax.text(i - 0.19, a + 0.004, f"{a:.3f}", ha="center", fontsize=8, color=INK)
    ax.text(i + 0.19, b + 0.004, f"{b:.3f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x, [f"N={n:,}" for n in Ns3])
ax.set_ylabel("mean |value bias| over policies")
ax.set_title("Toy: value de-biasing (12/12 cells improved)")
ax.legend()
save(fig, "fig3_value_debias.png")

# ---------------------------------------------------------------- fig 4
v3 = r34["V3_pessimism_sweep_N20k"]
cs = sorted(float(c) for c in v3["vanilla"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1))
for variant, col, mk in (("vanilla", RED, "o"), ("projected", BLUE, "s")):
    cov = [v3[variant][str(c)]["joint_coverage"] for c in cs]
    reg = [v3[variant][str(c)]["subopt_pess"] for c in cs]
    lbl = "vanilla Eq.-17" if variant == "vanilla" else "signal-projected"
    ax1.semilogx(cs, cov, mk + "-", color=col, lw=2, ms=5, label=lbl)
    ax2.semilogx(cs, reg, mk + "-", color=col, lw=2, ms=5, label=lbl)
ax1.set_xlabel("width multiplier c   ($\\xi_t = c/(N_2\\hat\\sigma_{2,t})$)")
ax1.set_ylabel("honest joint truth coverage")
ax1.set_title("Coverage (leak-gated: truth in the region)")
ax1.legend(loc="center right")
ax1.annotate("projected NEVER covers\n(truth leaks ~21% out\nof the subspace)",
             xy=(0.03, 0.0), xytext=(0.0013, 0.42), fontsize=7.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
ax2.set_xlabel("width multiplier c")
ax2.set_ylabel("true suboptimality of selection")
ax2.set_title("Selection regret (coverage/informativeness tension)")
ax2.legend(loc="upper left")
ax2.annotate("vanilla: wrong pick\nwherever it covers", xy=(0.1, 0.684),
             xytext=(0.0013, 0.45), fontsize=7.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
save(fig, "fig4_pessimism.png")

# ---------------------------------------------------------------- fig 5 (big-env failure)
rbd = rb["B2_debiasing"]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, kap in zip(axes, ("kappa=1.0", "kappa=0.0")):
    tabN = rbd[kap]["60000"] if "60000" in rbd[kap] else list(rbd[kap].values())[-1]
    pols = list(tabN)
    x = np.arange(len(pols))
    bn = [abs(tabN[p]["bias_naive_pop"]) for p in pols]
    bp = [abs(tabN[p]["bias_plugin"]) for p in pols]
    ax.bar(x - 0.19, bn, 0.38, color=BLUE, label="naive (uncorrected)")
    ax.bar(x + 0.19, bp, 0.38, color=RED, label="bridge plug-in")
    ax.set_xticks(x, [p.replace("_", "\n") for p in pols], fontsize=7.5)
    kaptxt = r"$\kappa$=1 (strong confounding)" if "1.0" in kap else r"$\kappa$=0 (none)"
    ax.set_title("benchmark 720x2, N=60k, " + kaptxt, fontsize=9)
    ax.axhline(0, color=MUT, lw=0.8)
axes[0].set_ylabel("|value bias| (lower is better)")
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Benchmark boundary: plug-in WORSE than naive in 15/16 cells",
             fontsize=10)
fig.subplots_adjust(top=0.84)
save(fig, "fig5_bigenv_failure.png")

# ---------------------------------------------------------------- CSV exports
def write_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print("wrote", os.path.join(RES, name))


write_csv("table1_naive_bias.csv",
          ["env", "setting", "policy", "V_true", "V_naive", "bias"],
          [["toy", "N=200k", p, v["V_true"], v["V_naive"], v["bias"]]
           for p, v in toy.items()] +
          [["big", k, p, v["V_true"], v["V_naive"], v["bias"]]
           for k, tab in big.items() for p, v in tab.items()])
write_csv("table2_bridge_recovery.csv",
          ["N", "errR_total", "errD_total"],
          [[n, sweep[str(n)]["errR_total"], sweep[str(n)]["errD_total"]]
           for n in Ns])
write_csv("table3_value_debias.csv",
          ["N", "policy", "bias_naive", "bias_plugin"],
          [[n, p, deb[str(n)][p]["bias_naive"], deb[str(n)][p]["bias_plugin"]]
           for n in Ns3 for p in deb[str(n)]])
write_csv("table4_pessimism.csv",
          ["variant", "c", "honest_joint_coverage", "empirical_Vlow_le_Vtrue",
           "regret", "V_low_best"],
          [[v, c, v3[v][str(c)]["joint_coverage"],
            v3[v][str(c)].get("all_bounds_valid", ""),
            v3[v][str(c)]["subopt_pess"],
            v3[v][str(c)]["V_low_mean"]["obs_dependent"]]
           for v in ("vanilla", "projected") for c in cs])
write_csv("table5_bigenv_debias.csv",
          ["kappa", "N", "policy", "V_true", "bias_naive_pop", "bias_plugin"],
          [[k, n, p, t[p]["V_true"], t[p]["bias_naive_pop"], t[p]["bias_plugin"]]
           for k, byN in rb["B2_debiasing"].items()
           for n, t in byN.items() for p in t])
print("done")
