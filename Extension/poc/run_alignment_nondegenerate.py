"""run_alignment_nondegenerate.py -- is the step (b) alignment result a discovery
or an identity?

family_pessimism.md section 3 (and my population reproduction in 4b) vary the
gradient's alignment with the design's null space by interpolating every emission
row toward their mean. At alpha = 1 all latent states emit identically, so the
channel carries no information about the state at all. If leakage falls only
because the channel is being destroyed, the sweep cannot fail, and "section 3
survives independent reproduction" is too strong.

Three parts:

  A  diagnose knob 1 (interpolation). Is beta_g proportional to the row
     separation -- i.e. are alignment and informativeness the same variable?
     And does the span VECTOR move, contradicting 4b's "the null space is FIXED"?

  B  knob 2 (prior tilt). At confound = 1.0 with n_s = n_a = 2 the behaviour
     policy is a deterministic indicator, so the normalised stage-2 design weight
     sits on a single state whatever p1 is, and H is EXACTLY p1-independent.
     Tilting p1 therefore moves alignment at a frozen channel and a frozen H.

  C  the closed form. In an exact null space H = rho*I, so

         ratio^2 = 1 + (beta_g^2 / (1 - beta_g^2)) * (sigma + rho) / rho

     If this holds to machine precision under both knobs, section 3 is an
     identity about ellipsoidal regions, not a fact about this environment.

Writes experiments/results_alignment_knob.json.
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
from mf_pessimism import h_inv_norm

N_S, N_O, N_O0, N_A, T = 2, 6, 4, 2, 3
RHO = 1e-3
ACT = 1                     # the action whose span we track
RESULTS = {}


def pop_span_and_design(p, a):
    """Population span basis and design for action a -- closed form, no sampling."""
    E, p1, K0, pi_b = p["E"], p["p1"], p["K0"], p["pi_b"]
    cols, ws = [], []
    for x in range(K0.shape[1]):
        v = p1 * pi_b[:, a] * K0[:, x]
        if v.sum() <= 0:
            continue
        ws.append(v.sum())
        cols.append(E.T @ (v / v.sum()))
    D = sum(w * np.outer(m, m) for w, m in zip(ws, cols)) / sum(ws)
    U, sv, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
    r = int((sv > 1e-12 * max(sv[0], 1e-300)).sum())
    return U[:, :r], D


def measure(p, a=ACT):
    """beta_g, widths, H and the span basis for one parameter set."""
    nu1 = p["p1"] @ p["E"]
    U, D = pop_span_and_design(p, a)
    H = D + RHO * np.eye(N_O)
    bg = float(np.linalg.norm(nu1 - U @ (U.T @ nu1)) / np.linalg.norm(nu1))
    wu = h_inv_norm(H, nu1)
    wp = h_inv_norm(H, nu1, basis=U)
    return dict(beta_g=bg, w_unproj=wu, w_proj=wp, ratio=wu / max(wp, 1e-300),
                cond=float(np.linalg.cond(H))), H, U, D


def sep(E):
    """Row separation of the emission channel: min pairwise L2 distance."""
    return float(min(np.linalg.norm(E[i] - E[j])
                     for i in range(E.shape[0]) for j in range(i + 1, E.shape[0])))


def angle_deg(u, v):
    c = abs(float(u.ravel() @ v.ravel())) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(min(1.0, c))))


def closed_form_ratio(bg, sigma, rho=RHO):
    return float(np.sqrt(1.0 + (bg ** 2 / max(1 - bg ** 2, 1e-300))
                         * (sigma + rho) / rho))


# ---------------------------------------------------------------- part A
def partA():
    print("=" * 92)
    print("[A] KNOB 1 (interpolation toward the mean row) -- is it degenerate?")
    print("    If beta_g tracks the row separation, alignment and informativeness")
    print("    are the same variable and the sweep cannot fail.")
    print("=" * 92)
    base = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                          seed=0, confound=1.0)
    E0 = base["E"].copy()
    _, _, U0, _ = measure(base)
    print(f"{'alpha':>7}{'row sep':>11}{'sep/(1-a)':>12}{'beta_g':>12}"
          f"{'bg/(1-a)':>11}{'span rot':>10}{'ratio':>10}{'cond(H)':>12}")
    rows = []
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]:
        p = dict(base)
        p["E"] = (1 - alpha) * E0 + alpha * E0.mean(axis=0, keepdims=True)
        m, H, U, D = measure(p)
        s = sep(p["E"])
        rot = angle_deg(U0[:, 0], U[:, 0])
        one_m = max(1 - alpha, 1e-12)
        print(f"{alpha:>7.2f}{s:>11.5f}{s/one_m:>12.5f}{m['beta_g']:>12.4e}"
              f"{m['beta_g']/one_m:>11.5f}{rot:>10.2f}{m['ratio']:>10.3f}"
              f"{m['cond']:>12.3e}")
        rows.append(dict(alpha=alpha, sep=s, span_rotation_deg=rot, **m))

    # P1: is beta_g proportional to separation?
    b = np.array([r["beta_g"] for r in rows])
    s = np.array([r["sep"] for r in rows])
    slope = float(np.polyfit(np.log(s), np.log(b), 1)[0])
    r2 = float(np.corrcoef(np.log(s), np.log(b))[0, 1] ** 2)
    print(f"\n   P1  log-log slope of beta_g vs row separation = {slope:.4f} "
          f"(R^2 = {r2:.6f})   [predicted 1.00 +- 0.05]")
    print(f"       -> {'DEGENERATE: the knob varies informativeness, not alignment alone' if abs(slope-1) < 0.05 else 'not proportional'}")
    tot = rows[-1]["span_rotation_deg"]
    print(f"   P2  span vector rotates {tot:.2f} deg across the sweep "
          f"[predicted > 5]")
    print(f"       -> section 4b's 'the null space is FIXED' is "
          f"{'FALSE as written; only its DIMENSION is fixed' if tot > 5 else 'defensible'}")
    RESULTS["knob1"] = dict(rows=rows, loglog_slope=slope, r2=r2,
                            total_span_rotation_deg=tot)
    return rows


# ---------------------------------------------------------------- part B
def partB():
    print("\n" + "=" * 92)
    print("[B] KNOB 2 (prior tilt) -- alignment varied at a FROZEN channel")
    print("    E, K0, P, pi_b untouched; only p1 moves toward delta_{s_a}.")
    print("=" * 92)
    base = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                          seed=0, confound=1.0)
    p1_0 = base["p1"].copy()
    delta = np.zeros(N_S)
    delta[ACT % N_S] = 1.0
    _, H0, U0, D0 = measure(base)
    sep0 = sep(base["E"])
    print(f"{'t':>7}{'row sep':>11}{'max|dH|':>12}{'span rot':>10}"
          f"{'beta_g':>12}{'W unproj':>11}{'W proj':>10}{'ratio':>10}")
    rows = []
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]:
        p = dict(base)
        p["p1"] = (1 - t) * p1_0 + t * delta
        m, H, U, D = measure(p)
        dH = float(np.abs(H - H0).max())
        rot = angle_deg(U0[:, 0], U[:, 0])
        print(f"{t:>7.3f}{sep(p['E']):>11.5f}{dH:>12.2e}{rot:>10.2f}"
              f"{m['beta_g']:>12.4e}{m['w_unproj']:>11.4f}{m['w_proj']:>10.4f}"
              f"{m['ratio']:>10.3f}")
        rows.append(dict(t=t, max_dH=dH, span_rotation_deg=rot,
                         sep=sep(p["E"]), **m))

    max_dH = max(r["max_dH"] for r in rows)
    max_rot = max(r["span_rotation_deg"] for r in rows)
    decades = np.log10(rows[0]["beta_g"] / max(rows[-1]["beta_g"], 1e-300))
    print(f"\n   P3  max|dH| over the sweep = {max_dH:.2e}   "
          f"max span rotation = {max_rot:.2e} deg")
    print(f"       row separation constant at {sep0:.5f} (channel untouched)")
    print(f"       beta_g spans {decades:.2f} decades "
          f"({rows[0]['beta_g']:.4f} -> {rows[-1]['beta_g']:.2e}) "
          f"[predicted >= 1]")
    ok = max_dH == 0.0 and max_rot < 1e-6 and decades >= 1.0
    print(f"       -> {'CLEAN CONTROL: alignment moved with everything else frozen' if ok else 'CONTROL IMPERFECT'}")
    RESULTS["knob2"] = dict(rows=rows, max_dH=max_dH,
                            max_span_rotation_deg=max_rot,
                            beta_g_decades=float(decades))
    return rows


# ---------------------------------------------------------------- part C
def partC(rows1, rows2):
    print("\n" + "=" * 92)
    print("[C] THE CLOSED FORM  ratio = sqrt(1 + (bg^2/(1-bg^2)) * (sigma+rho)/rho)")
    print("    If this holds under BOTH knobs, section 3 is an identity about")
    print("    ellipsoidal regions with an exact null space, not a measurement.")
    print("=" * 92)
    base = default_params(n_s=N_S, n_a=N_A, n_o=N_O, n_o0=N_O0, T=T,
                          seed=0, confound=1.0)
    E0 = base["E"].copy()

    def sigma_of(p):
        _, _, _, D = measure(p)
        return float(np.linalg.eigvalsh(D).max())

    print(f"{'knob':>8}{'param':>9}{'beta_g':>12}{'ratio meas':>13}"
          f"{'ratio pred':>13}{'rel err':>12}")
    errs, out = [], []
    for r in rows1:
        p = dict(base)
        p["E"] = (1 - r["alpha"]) * E0 + r["alpha"] * E0.mean(axis=0, keepdims=True)
        pred = closed_form_ratio(r["beta_g"], sigma_of(p))
        e = abs(pred - r["ratio"]) / r["ratio"]
        errs.append(e)
        out.append(dict(knob=1, param=r["alpha"], beta_g=r["beta_g"],
                        ratio_meas=r["ratio"], ratio_pred=pred, rel_err=e))
        print(f"{1:>8}{r['alpha']:>9.3f}{r['beta_g']:>12.4e}{r['ratio']:>13.5f}"
              f"{pred:>13.5f}{e:>12.2e}")
    p1_0 = base["p1"].copy()
    delta = np.zeros(N_S)
    delta[ACT % N_S] = 1.0
    for r in rows2:
        p = dict(base)
        p["p1"] = (1 - r["t"]) * p1_0 + r["t"] * delta
        pred = closed_form_ratio(r["beta_g"], sigma_of(p))
        e = abs(pred - r["ratio"]) / r["ratio"]
        errs.append(e)
        out.append(dict(knob=2, param=r["t"], beta_g=r["beta_g"],
                        ratio_meas=r["ratio"], ratio_pred=pred, rel_err=e))
        print(f"{2:>8}{r['t']:>9.3f}{r['beta_g']:>12.4e}{r['ratio']:>13.5f}"
              f"{pred:>13.5f}{e:>12.2e}")

    worst = max(errs)
    print(f"\n   P4  worst relative error over {len(errs)} cells, both knobs "
          f"= {worst:.2e}")
    if worst < 1e-10:
        print("       -> IDENTITY. The width ratio depends on the gradient ONLY")
        print("          through its leakage. No sweep could have refuted this,")
        print("          which makes section 3 reliable and NOT a discovery.")
    else:
        print("       -> the closed form is approximate; section 3 retains")
        print("          empirical content.")
    RESULTS["closed_form"] = dict(cells=out, worst_rel_err=worst)


def main():
    r1 = partA()
    r2 = partB()
    partC(r1, r2)
    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "results_alignment_knob.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", out)


if __name__ == "__main__":
    main()
