"""
fa_pessimism.py -- pessimistic selection for the finite-action setting.

Reuses ellipsoid_opt.py's BlockEllipsoid / pessimistic_value UNCHANGED, for the
third time now across three different bridge parameterisations (tabular,
continuous-action kernel, finite-action). That engine works purely in
coefficient space, so the only thing that varies is the dimension of the blocks:
here each block is (K+1)-dimensional, K = |A|.

No signal projection is built here. Signal-Projected Pessimism exists to repair
a STRUCTURAL null space (|O|>|S| in the tabular pipeline). In this primal
parameterisation H is a genuine (K+1)x(K+1) positive-definite matrix with no
such pathology, so there is nothing to project away.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ellipsoid_opt import BlockEllipsoid, pessimistic_value          # noqa: E402

_EST = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "estimation"))
sys.path.insert(0, _EST)
from fa_value_plugin import chain_value_and_grads                    # noqa: E402


def build_blocks(est):
    """BlockEllipsoids for the T reward and (T-1) dynamics stages.

    sigma2 for the width rule is the smallest eigenvalue of H, which is
    legitimate here precisely because there is no structural null space pinning
    it near zero (see module docstring).
    """
    blocks_R, blocks_D = [], []
    for t in range(1, est.T + 1):
        s = est.stage_R[t]
        sig = float(np.linalg.eigvalsh(s["H"])[0])
        blocks_R.append(BlockEllipsoid(s["H"], s["theta"], sig, s["N2"], f"bR_t{t}"))
    for j in range(1, est.T):
        s = est.stage_D[j]
        sig = float(np.linalg.eigvalsh(s["H"])[0])
        blocks_D.append(BlockEllipsoid(s["H"], s["theta"], sig, s["N2"], f"bD_t{j}"))
    return blocks_R, blocks_D


def fa_pessimistic_selection(est, candidates, m1, c, v_true=None,
                             n_restarts=3, seed=0):
    """Pessimistic selection over observation-independent candidate policies.

    chain_value_and_grads already returns the flat (V, gR, gD) triple that
    pessimistic_value expects, so no adapter shim is needed.
    """
    blocks_R, blocks_D = build_blocks(est)
    xis_R = [b.width_rule(c) for b in blocks_R]
    xis_D = [b.width_rule(c) for b in blocks_D]
    rng = np.random.default_rng(seed)

    out = dict(c=float(c), policies={})
    for name, pi in candidates.items():
        def vg(bR, bD, pi=pi):
            return chain_value_and_grads(pi, m1, bR, bD)
        res = pessimistic_value(blocks_R, blocks_D, xis_R, xis_D, vg,
                                n_restarts=n_restarts, rng=rng)
        rec = dict(V_low=res["V_low"], restart_gap=res["restart_gap"],
                   V_plugin=float(vg([b.b_hat for b in blocks_R],
                                     [b.b_hat for b in blocks_D])[0]))
        if v_true is not None:
            rec["V_true"] = float(v_true[name])
            rec["valid_lower_bound"] = bool(res["V_low"] <= v_true[name] + 1e-6)
        out["policies"][name] = rec

    sel = max(out["policies"], key=lambda k: out["policies"][k]["V_low"])
    out["selected_policy"] = sel
    if v_true is not None:
        best = max(v_true, key=v_true.get)
        sel_pl = max(out["policies"], key=lambda k: out["policies"][k]["V_plugin"])
        out["true_best_policy"] = best
        out["suboptimality_pessimistic"] = float(v_true[best] - v_true[sel])
        out["suboptimality_plugin"] = float(v_true[best] - v_true[sel_pl])
        out["all_lower_bounds_valid"] = bool(all(
            p["valid_lower_bound"] for p in out["policies"].values()))
    return out
