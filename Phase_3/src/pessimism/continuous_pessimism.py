"""
continuous_pessimism.py -- thin adapter wiring ContinuousBridgeEstimator's
fitted stages into ellipsoid_opt.py's EXISTING pessimism engine, unchanged.

WHY NO NEW PESSIMISM MATH WAS NEEDED. ellipsoid_opt.BlockEllipsoid and
pessimistic_value operate entirely in COEFFICIENT space (H, b_hat_vec) --
nothing about them assumes a tabular origin. For the (default, primary-
verified) linear-kernel backend, that coefficient space is the PRIMAL 2-D
theta = (a-coefficient, o~-coefficient), with H a proper 2x2 PD matrix (see
continuous_bridge_estimator.py's module docstring for why the primal, not
dual, representation is the numerically correct one for a linear kernel).
So we reuse BlockEllipsoid / pessimistic_value directly, imported unmodified,
and just build their inputs from the continuous estimator's stage output.

SIGNAL PROJECTION: NOT NEEDED HERE. The tabular pipeline's Signal-Projected
Pessimism existed to repair a STRUCTURAL null space (|O|>|S|, exactly zero
signal beyond it). In the linear-kernel primal space, H is a genuine 2x2 PD
matrix with no such structural pathology (confirmed empirically -- see
docs/continuous_extension.md), so there is nothing to project away from. We
do not build a projected variant for this pass; if a future RBF-backed
extension revisits this, the RBF dual space's near-null directions (a kernel
with smooth spectral decay, not a hard wall) would need the EFFECTIVE-
DIMENSION-based treatment described in continuous_bridge_estimator.py, not
the tabular hard-eigengap rule.

WIDTH CALIBRATION: sigma2 for the width rule xi=c/(N2*sigma2) is the smallest
eigenvalue of H itself (legitimate here specifically BECAUSE there is no
structural null space pinning it near-zero the way there was in the tabular
case -- see the module docstring above and docs/continuous_extension.md for
the precise distinction, so this is not a silent reversion to the diagnostic
the tabular project found misleading).
"""

import sys
import os

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)))))
from ellipsoid_opt import BlockEllipsoid, pessimistic_value          # noqa: E402

_ESTIMATION_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "estimation"))
sys.path.insert(0, _ESTIMATION_DIR)
from continuous_value_plugin import chain_value_and_grads             # noqa: E402


def build_continuous_blocks(est):
    """BlockEllipsoids for the T reward + (T-1) dynamics stages."""
    import numpy as np
    blocks_R, blocks_D = [], []
    for t in range(1, est.T + 1):
        s = est.stage_R[t]
        sigma2 = float(np.linalg.eigvalsh(s["H"])[0])
        blocks_R.append(BlockEllipsoid(s["H"], s["theta"], sigma2, s["N2"], f"bR_t{t}"))
    for j in range(1, est.T):
        s = est.stage_D[j]
        sigma2 = float(np.linalg.eigvalsh(s["H"])[0])
        blocks_D.append(BlockEllipsoid(s["H"], s["theta"], sigma2, s["N2"], f"bD_t{j}"))
    return blocks_R, blocks_D


def _value_and_grads_factory(est, K_pi, c_pi, m1):
    """ellipsoid_opt.pessimistic_value calls value_and_grads(bR,bD) and
    unpacks it as `_, gR, _ = value_and_grads(...)` -- a FLAT 3-tuple
    (V, gR_list, gD_list), not chain_value_and_grads' own (V, per_step,
    (gR,gD)) shape. This adapts one to the other."""
    def value_and_grads(bR_flat, bD_flat):
        V, _, (gR, gD) = chain_value_and_grads(est, K_pi, c_pi, m1, bR_flat, bD_flat)
        return V, gR, gD
    return value_and_grads


def continuous_pessimistic_selection(est, candidates, m1, c, true_bR=None,
                                     true_bD=None, v_true=None, n_restarts=3,
                                     seed=0):
    """Pessimistic selection over affine candidate policies {name: (K,c)}.

    Mirrors pessimistic_optimizer.pessimistic_selection's interface/outputs
    (V_low per policy, selection, true suboptimality) but WITHOUT a truth-
    coverage diagnostic: expressing the (infinite-dimensional-in-general)
    true bridge in this estimator's data-dependent coefficient coordinates is
    not needed for the core claim, since the discrete pipeline's own ultimate
    guarantee is also empirical, not a coverage certificate (ellipsoid_opt.py
    says so explicitly). We check V_low <= V_true directly instead -- see
    docs/continuous_extension.md "Known simplifications" for why this
    intermediate diagnostic is skipped in this pass.
    """
    import numpy as np
    blocks_R, blocks_D = build_continuous_blocks(est)
    xis_R = [blk.width_rule(c) for blk in blocks_R]
    xis_D = [blk.width_rule(c) for blk in blocks_D]
    rng = np.random.default_rng(seed)

    out = dict(c=float(c), policies={})
    for name, (K_pi, c_pi) in candidates.items():
        vg = _value_and_grads_factory(est, K_pi, c_pi, m1)
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
        out["true_best_policy"] = best
        out["suboptimality_pessimistic"] = float(v_true[best] - v_true[sel])
        sel_plugin = max(out["policies"],
                         key=lambda k: out["policies"][k]["V_plugin"])
        out["suboptimality_plugin"] = float(v_true[best] - v_true[sel_plugin])
        out["all_lower_bounds_valid"] = bool(all(
            p["valid_lower_bound"] for p in out["policies"].values()))
    return out
