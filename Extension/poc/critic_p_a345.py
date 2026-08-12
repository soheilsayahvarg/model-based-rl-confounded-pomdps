"""critic_p_a345.py -- ATTACKS A3, A4, A5 on the model-based norm-ball results.

[S1] A3: what M is actually admissible? The toy's TRUE bridges are computable
     (oracle_module.toy_true_bridges). Compare ||b_true|| per block against
     ||b_hat|| and against the shipped M (mean of reward-block norms). If
     M < ||b_true|| for any block, the ball EXCLUDES the truth and the
     comparison table's "contains truth: yes" column is false. Also audit
     b_true's membership in the ellipsoid at each width (the real coverage
     check nobody ran).

[S2] A4: re-run the full coordinate descent with the ball step solved EXACTLY
     (certified QCQP solver from critic_p_a2, validated against trust-constr)
     instead of the shipped bisection. Report V_low side by side, plus the
     worst ball violation the shipped solver commits inside the descent. Run
     with per-block M = ||b_hat_block|| (always feasible) and with honest
     M = ||b_true_block|| and 2x that.

[S3] A5: decompose the published -1755/-1779 (c=10) into: width choice (c),
     the M_R omission, and what remains. Fractions of the total drop from
     V_hat.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("envs", "estimation", "baselines", "pessimism"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", sub)))

import toy_pomdp as toy
from bridge_estimator import TabularBridgeEstimator
from value_plugin import plugin_value, empirical_p_o1
from ellipsoid_opt import BlockEllipsoid, pessimistic_value
import ellipsoid_opt
from critic_p_a2 import exact_linear_min_ball, shipped_ball_min

N_MAIN = 5000
SEEDS = [0, 1, 2]
T = 3
PI = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
RESULTS = {}


def toy_true_bridges(params):
    """Inlined from oracle_module.toy_true_bridges (whose Extension copy has an
    unmet simulated_env import): min-norm true bridges via pinv(E)."""
    E, P, pR = params["E"], params["P"], params["pR"]
    Epinv = np.linalg.pinv(E)
    bR = np.zeros((toy.N_A, toy.N_O, 2, toy.N_O))
    resR = 0.0
    for a in range(toy.N_A):
        for r in range(2):
            p_r = pR[:, a] if r == 1 else 1.0 - pR[:, a]
            for o in range(toy.N_O):
                c = E[:, o] * p_r
                u = Epinv @ c
                bR[a, :, r, o] = u
                resR = max(resR, float(np.abs(E @ u - c).max()))
    bD = np.zeros((toy.N_A, toy.N_O, toy.N_O, toy.N_O))
    resD = 0.0
    for a in range(toy.N_A):
        PE = P[a] @ E
        for o in range(toy.N_O):
            C = E[:, o][:, None] * PE
            B = Epinv @ C
            bD[a, :, :, o] = B
            resD = max(resD, float(np.abs(E @ B - C).max()))
    return bR, bD, dict(residual_bR=resR, residual_bD=resD)


def dp_value_toy(params, pi_obs):
    E, P, pR, p1 = params["E"], params["P"], params["pR"], params["p1"]
    V = np.zeros(toy.N_S)
    for _ in range(T):
        Q = pR + np.einsum("asz,z->sa", P, V)
        V = np.einsum("sa,sa->s", E @ pi_obs, Q)
    return float(p1 @ V)


def build_blocks(est):
    out = []
    for t in range(1, T + 1):
        s = est.stage_store[f"bR_t{t}"]
        out.append((f"bR_t{t}", BlockEllipsoid(
            s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"], f"bR_t{t}",
            s["signal_basis"], n_obs=toy.N_O, n_act=toy.N_A, n_y=s["n_y"])))
        if t < T:
            s = est.stage_store[f"bD_t{t}"]
            out.append((f"bD_t{t}", BlockEllipsoid(
                s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"], f"bD_t{t}",
                s["signal_basis"], n_obs=toy.N_O, n_act=toy.N_A, n_y=s["n_y"])))
    return out


# ---- patched linear_min: exact ball step, per-block M via attribute ---------
_BALL_VIOL = {"max": 0.0}
_orig_linear_min = BlockEllipsoid.linear_min


def _patched_linear_min(self, g, xi, projected=False, M=None):
    M_eff = getattr(self, "M_override", M)
    if M_eff is not None and not projected:
        mode = getattr(self, "ball_mode", "exact")
        if mode == "shipped":
            b = shipped_ball_min(self.H, self.b_hat, g, xi, M_eff)
            _BALL_VIOL["max"] = max(_BALL_VIOL["max"],
                                    float(np.linalg.norm(b) - M_eff))
            return b, float(g @ self.b_hat - g @ b)
        b, status = exact_linear_min_ball(self.H, self.b_hat, g, xi, M_eff)
        if b is None:
            raise RuntimeError(f"empty intersection in block {self.label}")
        return b, float(g @ self.b_hat - g @ b)
    return _orig_linear_min(self, g, xi, projected=projected, M=None)


BlockEllipsoid.linear_min = _patched_linear_min


def run_case(blocks, p_o1, c, sd, mode=None, M_map=None, projected=False):
    """mode: None (vanilla) | 'shipped' | 'exact'; M_map: label -> M."""
    names = [n for n, _ in blocks]
    bR = [b for n, b in blocks if n.startswith("bR")]
    bD = [b for n, b in blocks if n.startswith("bD")]
    for n, b in blocks:
        b.M_override = None if M_map is None else M_map[n]
        b.ball_mode = mode

    def vg(bRl, bDl):
        R = np.stack([x.reshape(toy.N_A, toy.N_O, 2, toy.N_O) for x in bRl])
        D = np.stack([x.reshape(toy.N_A, toy.N_O, toy.N_O, toy.N_O)
                      for x in bDl])
        V, _p, (gR, gD) = plugin_value(R, D, PI, p_o1, return_grads=True)
        return (V, [gR[t].ravel() for t in range(T)],
                [gD[j].ravel() for j in range(T - 1)])

    xR = [b.width_rule(c) for b in bR]
    xD = [b.width_rule(c) for b in bD]
    r = pessimistic_value(bR, bD, xR, xD, vg, rng=np.random.default_rng(sd),
                          projected=projected,
                          M_R=None if M_map is None else -1,   # sentinel; per-block override used
                          M_D=None if M_map is None else -1)
    for n, b in blocks:
        b.M_override = None
        b.ball_mode = None
    return r["V_low"]


def main():
    params = toy.default_params(T=3)
    v_true = dp_value_toy(params, PI)
    bR_true, bD_true, res = toy_true_bridges(params)
    assert max(res["residual_bR"], res["residual_bD"]) < 1e-10
    # time-homogeneous single-stage tables (a, o~, r, o) / (a, o~, o~', o),
    # shared by every stage; ravel order matches stage_store's b_hat_vec.
    true_norms = {}
    nR = float(np.linalg.norm(bR_true.ravel()))
    nD = float(np.linalg.norm(bD_true.ravel()))
    for t in range(1, T + 1):
        true_norms[f"bR_t{t}"] = nR
        if t < T:
            true_norms[f"bD_t{t}"] = nD

    print(f"V_true(pi) = {v_true:.4f}")
    print(f"true bridge norms: ||bR_true|| = {nR:.3f}   ||bD_true|| = {nD:.3f}\n")

    # ---------------- S1: admissibility audit
    print("[S1] per-block ||b_hat|| vs ||b_true|| vs shipped M (3 seeds)")
    fits = []
    for sd in SEEDS:
        d = toy.sample_trajectories(params, N_MAIN, np.random.default_rng(sd))
        est = TabularBridgeEstimator(n_obs=toy.N_O, n_act=toy.N_A,
                                     n_o0=toy.N_O0, n_r=2, T=T,
                                     mode="primal", seed=sd)
        est.fit(d["O0"], d["O"], d["A"], d["R"])
        blocks = build_blocks(est)
        p_o1 = empirical_p_o1(d["O"], toy.N_O)
        fits.append((blocks, p_o1))
        M_ship = float(np.mean([np.linalg.norm(b.b_hat)
                                for n, b in blocks if n.startswith("bR")]))
        print(f"  seed {sd}: M_shipped = {M_ship:.3f}")
        for n, b in blocks:
            nb = float(np.linalg.norm(b.b_hat))
            nt = true_norms[n]
            flags = []
            if nb < nt:
                flags.append("||b_hat|| < ||b_true||")
            if M_ship < nt:
                flags.append("M_shipped EXCLUDES TRUTH")
            if nb > M_ship:
                flags.append("center outside shipped ball")
            print(f"    {n}: ||b_hat||={nb:.3f}  ||b_true||={nt:.3f}  "
                  + ("; ".join(flags) if flags else "ok"))

    # ---------------- S2 + S3: corrected solver, honest M, decomposition
    print("\n[S2] V_low under vanilla / shipped-ball / exact-ball / honest-M "
          "(3 seeds, per-block M unless noted)")
    print(f"{'c':>6}{'vanilla':>10}{'ball ship':>11}{'ball exact':>12}"
          f"{'M=true':>10}{'M=2xtrue':>11}{'projected':>11}")
    rows = {}
    for c in (0.1, 1.0, 10.0):
        cols = {k: [] for k in ("van", "ship", "exact", "mtrue", "m2true", "proj")}
        for (blocks, p_o1), sd in zip(fits, SEEDS):
            M_hat = {n: float(np.linalg.norm(b.b_hat)) for n, b in blocks}
            M_tru = dict(true_norms)
            M_2tr = {k: 2.0 * v for k, v in true_norms.items()}
            _BALL_VIOL["max"] = 0.0
            cols["van"].append(run_case(blocks, p_o1, c, sd))
            cols["ship"].append(run_case(blocks, p_o1, c, sd, "shipped", M_hat))
            viol = _BALL_VIOL["max"]
            cols["exact"].append(run_case(blocks, p_o1, c, sd, "exact", M_hat))
            cols["mtrue"].append(run_case(blocks, p_o1, c, sd, "exact", M_tru))
            cols["m2true"].append(run_case(blocks, p_o1, c, sd, "exact", M_2tr))
            cols["proj"].append(run_case(blocks, p_o1, c, sd, projected=True))
        m = {k: float(np.mean(v)) for k, v in cols.items()}
        rows[c] = m
        print(f"{c:>6.1f}{m['van']:>10.2f}{m['ship']:>11.2f}{m['exact']:>12.2f}"
              f"{m['mtrue']:>10.2f}{m['m2true']:>11.2f}{m['proj']:>11.2f}"
              f"   (worst shipped ball violation this c: {viol:.3f})")

    print("\n[S3] decomposition of the published c=10 divergence "
          f"(V_hat ~ {v_true:.2f} scale):")
    van10, ex10 = rows[10.0]["van"], rows[10.0]["exact"]
    mt10 = rows[10.0]["mtrue"]
    van01 = rows[0.1]["van"]
    print(f"  vanilla c=10          : {van10:9.2f}   (the published headline)")
    print(f"  vanilla c=0.1         : {van01:9.2f}   -> width choice (c) accounts for "
          f"{(van01 - van10) / max(abs(van10), 1e-9):.0%} of |headline|")
    print(f"  + exact ball, M=||b_hat|| : {ex10:9.2f}   -> omission accounts for "
          f"{(ex10 - van10) / max(abs(van10), 1e-9):.0%} of |headline| at c=10")
    print(f"  + exact ball, M=||b_true||: {mt10:9.2f}")

    out = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                        "critic_p_a345.json"))
    import json
    with open(out, "w") as f:
        json.dump(dict(rows={str(k): v for k, v in rows.items()},
                       true_norms=true_norms, v_true=v_true), f, indent=2)
    print("\nwritten to", out)


if __name__ == "__main__":
    main()
