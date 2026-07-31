"""
pessimistic_optimizer.py — Phase 4: pessimistic policy SELECTION over a finite
candidate class, with restricted-spectrum width calibration and coverage
instrumentation.

Pipeline (policy-independent regions computed ONCE — the anchor's Remark 3.7):
  1. From a fitted TabularBridgeEstimator, build the 2T-1 BlockEllipsoids
     (H = T_hat_2 + lam2 I, center b_hat) from stage_store.
  2. Widths via the restricted-spectrum rule xi_t = c / (N2 * sigma2_t)
     (Phase-2 finding: sigma2 = smallest per-action SIGNAL eigenvalue; raw
     cond(G)/eig_min are blind to identification collapse).
  3. For each candidate policy pi: V_low(pi) = blockwise closed-form coordinate
     descent of the multilinear plug-in value over the ellipsoid product.
  4. Select argmax_pi V_low(pi).

Instrumentation returned per width multiplier c:
  - coverage: fraction of blocks whose region contains the TRUE bridge (oracle),
    and joint coverage (all blocks simultaneously) — the empirical analogue of the
    anchor's Lemma C.2 event;
  - validity: explicit check V_low(pi) <= V_true(pi) per policy (the foundational
    pessimism guarantee; theory implies it on the coverage event for the JOINT
    minimum — coordinate descent is checked empirically, restart gaps logged);
  - selection: chosen policy, true suboptimality vs the best candidate.
"""

import numpy as np

from ellipsoid_opt import BlockEllipsoid, pessimistic_value
from value_plugin import plugin_value


def build_blocks(est):
    """BlockEllipsoids for the 2T-1 stages from a fitted estimator's stage_store."""
    T = est.T
    blocks_R, blocks_D = [], []
    for t in range(1, T + 1):
        s = est.stage_store[f"bR_t{t}"]
        blocks_R.append(BlockEllipsoid(
            s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"], f"bR_t{t}",
            signal_basis=s["signal_basis"], n_obs=est.n_obs, n_act=est.n_act,
            n_y=s["n_y"]))
    for t in range(1, T):
        s = est.stage_store[f"bD_t{t}"]
        blocks_D.append(BlockEllipsoid(
            s["H"], s["b_hat_vec"], s["sigma2_signal"], s["N2"], f"bD_t{t}",
            signal_basis=s["signal_basis"], n_obs=est.n_obs, n_act=est.n_act,
            n_y=s["n_y"]))
    return blocks_R, blocks_D


def _value_and_grads_factory(pi_obs, p_o1, shapes):
    """Wrap value_plugin.plugin_value for flat-vector block interfaces."""
    (T, n_act, n_obs, n_r) = shapes

    def value_and_grads(bR_flat, bD_flat):
        bR = np.stack([b.reshape(n_act, n_obs, n_r, n_obs) for b in bR_flat])
        bD = (np.stack([b.reshape(n_act, n_obs, n_obs, n_obs) for b in bD_flat])
              if T > 1 else np.zeros((0, n_act, n_obs, n_obs, n_obs)))
        V, _, (gR, gD) = plugin_value(bR, bD, pi_obs, p_o1, return_grads=True)
        return V, [gR[t].ravel() for t in range(T)], \
               [gD[j].ravel() for j in range(T - 1)]
    return value_and_grads


def coverage_report(blocks_R, blocks_D, bR_true, bD_true, xis_R, xis_D,
                    projected=False, leak_frac_tol=0.05):
    """Per-block and joint truth-coverage at the given widths.

    In the projected variant, "covered" now requires BOTH (a) the truth's
    in-subspace projection lies inside the ellipsoid AND (b) the out-of-subspace
    leakage is small (<= leak_frac_tol * ||b_true||). Gating on (b) is what makes
    the flag honest: the projected feasible set structurally excludes the truth's
    leakage component, so a projected V_low is only a trustworthy lower bound when
    that excluded component is negligible (audit finding A-high). max_subspace_leakage
    is returned so callers can see how far the truth sits outside the subspace."""
    per_block = {}
    all_in = True
    max_leak = 0.0
    for blks, trues, xis in ((blocks_R, bR_true, xis_R),
                             (blocks_D, bD_true, xis_D)):
        for i, blk in enumerate(blks):
            if projected:
                need, leak = blk.xi_needed(trues[i].ravel(), projected=True)
            else:
                need, leak = blk.xi_needed(trues[i].ravel()), None
            # HONEST coverage: in the projected variant the region is a SUBSET of
            # the full space that structurally excludes any out-of-subspace
            # component of the truth (leak > 0). Reporting "covered" purely on the
            # in-subspace distance (need <= xi) is misleading -- the truth is NOT
            # in the projected feasible set when leak > leak_tol, so the projected
            # V_low is NOT guaranteed <= V_true even at the exact inner min (the
            # excluded direction may be value-relevant; audit finding A-high). We
            # gate "covered" on BOTH the in-subspace fit AND small leakage.
            inside_proj = need <= xis[i]
            if projected:
                leak_tol = leak_frac_tol * np.linalg.norm(trues[i].ravel())
                inside = bool(inside_proj and leak <= leak_tol)
                max_leak = max(max_leak, leak)
            else:
                inside = bool(inside_proj)
            per_block[blk.label] = dict(xi=float(xis[i]), xi_needed=float(need),
                                        covered=inside)
            if leak is not None:
                per_block[blk.label]["subspace_leakage"] = float(leak)
                per_block[blk.label]["in_subspace_only"] = bool(inside_proj)
            all_in &= inside
    frac = np.mean([v["covered"] for v in per_block.values()])
    return dict(per_block=per_block, block_coverage=float(frac),
                joint_coverage=bool(all_in), max_subspace_leakage=float(max_leak))


def pessimistic_selection(est, candidates, p_o1, c, bR_true=None, bD_true=None,
                          v_true=None, n_restarts=3, seed=0,
                          signal_projection=False):
    """Run pessimistic selection at width multiplier c.

    candidates        : dict name -> pi_obs (n_obs, n_act)
    v_true            : optional dict name -> true V(pi) (oracle) for validity/regret
    signal_projection : restrict regions to the identified signal subspace
                        (the Phase-4 repair for the vanilla Eq.-17 blow-up)
    Returns dict with V_low per policy, selection, validity checks, coverage.
    """
    T, n_act, n_obs, n_r = est.T, est.n_act, est.n_obs, est.n_r
    blocks_R, blocks_D = build_blocks(est)
    xis_R = [blk.width_rule(c) for blk in blocks_R]
    xis_D = [blk.width_rule(c) for blk in blocks_D]
    rng = np.random.default_rng(seed)

    out = dict(c=float(c), variant="projected" if signal_projection else "vanilla",
               xis={blk.label: float(x) for blk, x in
                    list(zip(blocks_R, xis_R)) + list(zip(blocks_D, xis_D))},
               policies={})

    if bR_true is not None:
        bR_true_t = np.stack([bR_true] * T)          # time-homogeneous truth
        bD_true_t = np.stack([bD_true] * (T - 1))
        out["coverage"] = coverage_report(blocks_R, blocks_D,
                                          bR_true_t, bD_true_t, xis_R, xis_D,
                                          projected=signal_projection)

    for name, pi in candidates.items():
        vg = _value_and_grads_factory(pi, p_o1, (T, n_act, n_obs, n_r))
        res = pessimistic_value(blocks_R, blocks_D, xis_R, xis_D, vg,
                                n_restarts=n_restarts, rng=rng,
                                projected=signal_projection)
        rec = dict(V_low=res["V_low"], V_plugin=float(vg(
            [blk.b_hat for blk in blocks_R],
            [blk.b_hat for blk in blocks_D])[0]),
            restart_gap=res["restart_gap"])
        if v_true is not None:
            rec["V_true"] = float(v_true[name])
            rec["valid_lower_bound"] = bool(res["V_low"] <= v_true[name] + 1e-9)
        out["policies"][name] = rec

    sel = max(out["policies"], key=lambda k: out["policies"][k]["V_low"])
    out["selected_policy"] = sel
    if v_true is not None:
        best = max(v_true, key=v_true.get)
        out["true_best_policy"] = best
        out["suboptimality_pessimistic"] = float(v_true[best] - v_true[sel])
        sel_plugin = max(out["policies"],
                         key=lambda k: out["policies"][k]["V_plugin"])
        out["selected_by_plugin"] = sel_plugin
        out["suboptimality_plugin"] = float(v_true[best] - v_true[sel_plugin])
        out["all_lower_bounds_valid"] = bool(all(
            p["valid_lower_bound"] for p in out["policies"].values()))
    return out
