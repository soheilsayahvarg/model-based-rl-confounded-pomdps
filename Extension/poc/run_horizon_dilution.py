"""run_horizon_dilution.py -- is the stage-2+ self-healing uniform in T, and does
the surviving stage-1 leak still matter as the horizon grows?

Round 8 showed the per-action span reaches |S| at t = 2, so the truth-leak switch
is a stage-1 statement. That was one stage in one horizon. Two open questions:

  (1) Uniformity. Does the span reach |S| at EVERY stage t >= 2, at every
      horizon, or does it decay again deeper in the trajectory?

  (2) Dilution. If only t = 1 leaks, the floor in the total value is carried by
      one block out of T, while the decision-relevant scale -- the spread of
      true values across candidate policies -- grows with T. Does the floor wash
      out at long horizon?

Population algebra only, no sampling.

At stage t the conditioning cells are (a_1..a_{t-1}, o_1..o_{t-1}, o_0), so their
count grows as (|O||A|)^(t-1) * |O_0|. The span lives in R^|O|, so we accumulate
an orthonormal basis incrementally and re-orthonormalise rather than materialise
all columns.

Writes experiments/results_horizon_dilution.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))
sys.path.insert(0, HERE)

from dim_separated_pomdp import default_params
from run_completeness_beta import true_bridges_generic

TOL = 1e-10
RESULTS = {}


# ------------------------------------------------------------------ span
def stage_span(p, t, a_t, tol=TOL, cap=200000):
    """Orthonormal basis of the population stage-t design span for action a_t.

    t = 1 conditions on (a_1, o_0); t >= 2 additionally on the history
    (o_1..o_{t-1}, a_1..a_{t-1}). Posteriors are forward-propagated and the basis
    is reduced by SVD every chunk so memory stays O(|O|^2).
    """
    E, P, p1, K0, pi_b = p["E"], p["P"], p["p1"], p["K0"], p["pi_b"]
    n_s, n_o = E.shape
    n_o0 = K0.shape[1]
    n_a = P.shape[0]

    # weights over s_1, one per o_0 cell
    fronts = [p1 * K0[:, o0] for o0 in range(n_o0)]

    # push forward through stages 1..t-1, branching on (o_k, a_k)
    for _ in range(t - 1):
        nxt = []
        for w in fronts:
            for o in range(n_o):
                for a in range(n_a):
                    wn = (w * E[:, o] * pi_b[:, a]) @ P[a]
                    if wn.sum() > 1e-300:
                        nxt.append(wn)
        fronts = nxt
        if len(fronts) > cap:            # guard: exponential in t
            fronts = fronts[:cap]

    basis = np.zeros((n_o, 0))
    chunk = []
    for w in fronts:
        v = w * pi_b[:, a_t]
        if v.sum() <= 1e-300:
            continue
        chunk.append(E.T @ (v / v.sum()))
        if len(chunk) >= 512:
            basis = _reduce(np.hstack([basis, np.stack(chunk, 1)]), tol)
            chunk = []
    if chunk:
        basis = _reduce(np.hstack([basis, np.stack(chunk, 1)]), tol)
    return basis


def _reduce(M, tol):
    U, sv, _ = np.linalg.svd(M, full_matrices=False)
    if sv.size == 0:
        return U[:, :0]
    r = int((sv > tol * sv[0]).sum())
    return U[:, :r]


def leak_of(vec, U):
    """Relative norm of vec outside span(U)."""
    n = np.linalg.norm(vec)
    if n <= 0:
        return 0.0
    return float(np.linalg.norm(vec - U @ (U.T @ vec)) / n)


def block_leak(bR_a, U):
    """Relative norm of a reward-bridge block outside span(U)."""
    n = np.linalg.norm(bR_a)
    if n <= 0:
        return 0.0
    Pn = np.eye(U.shape[0]) - U @ U.T
    return float(np.linalg.norm(np.einsum("ij,jro->iro", Pn, bR_a)) / n)


# ------------------------------------------------------------------ value
def dp_values(p, policies):
    """Exact latent dynamic-programming value of each observation-only policy."""
    E, P, pR, p1, T = p["E"], p["P"], p["pR"], p["p1"], p["T"]
    n_s, n_a = p["n_s"], p["n_a"]
    out = []
    for pol in policies:                      # pol[o] -> action
        d = p1.copy()
        tot = 0.0
        for _ in range(T):
            # p(a | s) induced by an observation-only policy
            pa = np.zeros((n_s, n_a))
            for o in range(E.shape[1]):
                pa[:, pol[o]] += E[:, o]
            tot += float(np.sum(d[:, None] * pa * pR))
            d = np.einsum("s,sa,asz->z", d, pa, P)
        out.append(tot)
    return np.array(out)


def candidates(n_o, n_a, k=6, seed=0):
    rng = np.random.default_rng(seed)
    pols = [np.zeros(n_o, dtype=int), np.full(n_o, n_a - 1, dtype=int)]
    while len(pols) < k:
        pols.append(rng.integers(0, n_a, size=n_o))
    return pols


# ------------------------------------------------------------------ parts
def part1_uniformity():
    print("=" * 92)
    print("[1] Is the stage-2+ restoration UNIFORM in t?")
    print("    span rank per stage, per action (worst action shown)")
    print("=" * 92)
    print(f"{'config':>12}{'cf':>6}{'|S|':>5}"
          + "".join(f"{'t=' + str(t):>8}" for t in range(1, 6)))
    rows = []
    for (n_s, n_o, n_o0) in [(4, 6, 2), (4, 8, 3), (2, 6, 4)]:
        for cf in [0.0, 0.6, 0.9, 1.0]:
            p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=3,
                               seed=0, confound=cf)
            ranks = []
            for t in range(1, 6):
                r = min(stage_span(p, t, a).shape[1] for a in range(2))
                ranks.append(r)
            print(f"{str((n_s, n_o, n_o0)):>12}{cf:>6.1f}{n_s:>5}"
                  + "".join(f"{r:>8}" for r in ranks))
            rows.append(dict(config=[n_s, n_o, n_o0], confound=cf, n_s=n_s,
                             ranks=ranks))
    ok = all(all(r >= min(x["n_s"], 6) for r in x["ranks"][1:])
             for x in rows if x["confound"] < 1.0)
    print(f"\n   P1  span reaches |S| at every t>=2 for every cf<1: "
          f"{'YES' if ok else 'NO'}")
    RESULTS["uniformity"] = rows
    return rows


def part2_leak_per_stage():
    print("\n" + "=" * 92)
    print("[2] beta_t and beta_g,t per stage -- which blocks actually leak?")
    print("=" * 92)
    print(f"{'config':>12}{'cf':>6}{'t':>4}{'span':>6}{'beta_t':>12}"
          f"{'beta_g,t':>12}{'floor':>12}")
    rows = []
    for (n_s, n_o, n_o0) in [(4, 6, 2), (4, 8, 3)]:
        for cf in [0.6, 0.9]:
            p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=3,
                               seed=0, confound=cf)
            bR, _ = true_bridges_generic(p)
            nu1 = p["p1"] @ p["E"]
            for t in range(1, 5):
                best = None
                for a in range(2):
                    U = stage_span(p, t, a)
                    b = block_leak(bR[a], U)
                    g = leak_of(nu1, U)
                    if best is None or b * g > best[2]:
                        best = (b, g, b * g, U.shape[1])
                b, g, fl, r = best
                print(f"{str((n_s, n_o, n_o0)):>12}{cf:>6.1f}{t:>4}{r:>6}"
                      f"{b:>12.3e}{g:>12.3e}{fl:>12.3e}")
                rows.append(dict(config=[n_s, n_o, n_o0], confound=cf, t=t,
                                 span=r, beta=b, beta_g=g, floor=fl))
    lat = [r for r in rows if r["t"] >= 2]
    print(f"\n   P2  max beta_t over t>=2 = "
          f"{max(r['beta'] for r in lat):.2e}   [predicted <= 1e-12]")
    print(f"   P3  max floor over t>=2 = "
          f"{max(r['floor'] for r in lat):.2e}")
    RESULTS["per_stage"] = rows
    return rows


def part3_dilution(per_stage):
    print("\n" + "=" * 92)
    print("[3] DILUTION -- floor against the decision-relevant scale, vs horizon")
    print("=" * 92)
    print(f"{'config':>12}{'cf':>6}{'T':>4}{'floor':>10}{'V spread':>11}"
          f"{'floor/spread':>14}")
    rows = []
    for (n_s, n_o, n_o0) in [(4, 6, 2), (4, 8, 3)]:
        for cf in [0.6, 0.9]:
            base = [r for r in per_stage
                    if r["config"] == [n_s, n_o, n_o0] and r["confound"] == cf]
            floor1 = max(r["floor"] for r in base if r["t"] == 1)
            pols = candidates(n_o, 2)
            for T in [1, 2, 3, 4, 5, 6]:
                p = default_params(n_s=n_s, n_a=2, n_o=n_o, n_o0=n_o0, T=T,
                                   seed=0, confound=cf)
                v = dp_values(p, pols)
                spread = float(v.max() - v.min())
                ratio = floor1 / spread if spread > 0 else float("inf")
                print(f"{str((n_s, n_o, n_o0)):>12}{cf:>6.1f}{T:>4}"
                      f"{floor1:>10.4f}{spread:>11.4f}{ratio:>14.4f}")
                rows.append(dict(config=[n_s, n_o, n_o0], confound=cf, T=T,
                                 floor=floor1, spread=spread, ratio=ratio))
    print("\n   P4  log-log slope of (floor/spread) against T "
          "[predicted about -1]:")
    for (n_s, n_o, n_o0) in [(4, 6, 2), (4, 8, 3)]:
        for cf in [0.6, 0.9]:
            sub = [r for r in rows
                   if r["config"] == [n_s, n_o, n_o0] and r["confound"] == cf]
            Ts = np.array([r["T"] for r in sub], float)
            rs = np.array([r["ratio"] for r in sub], float)
            ok = rs > 0
            sl = float(np.polyfit(np.log(Ts[ok]), np.log(rs[ok]), 1)[0])
            print(f"       {str((n_s, n_o, n_o0)):>12} cf={cf:.1f}  "
                  f"slope = {sl:+.3f}   T=1 -> T=6 : "
                  f"{sub[0]['ratio']:.3f} -> {sub[-1]['ratio']:.3f}")
    RESULTS["dilution"] = rows


def main():
    part1_uniformity()
    ps = part2_leak_per_stage()
    part3_dilution(ps)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_horizon_dilution.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
