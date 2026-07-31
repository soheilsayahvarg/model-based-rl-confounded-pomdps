"""
continuous_value_plugin.py -- Theorem-3.5 mean-chain for CONTINUOUS bridges, the
analogue of value_plugin.py.

Because both continuous bridges are MEAN-VALUE bridges (see
continuous_bridge_estimator.py's docstring) and candidate policies are affine
and observation-only (a = K_pi*o + c_pi, see continuous_env.py), the chain
degenerates from the discrete case's vector recursion (g_t is a distribution
over o~) to a SCALAR recursion over a single "chain mean" m_t:

    m_1 = E-hat[O_1]                          (empirical initial mean; the
                                                oracle path instead starts
                                                from the exact mu1 -- see
                                                continuous_env.chain_value_
                                                from_bridges)
    a_hat_t   = K_pi*m_t + c_pi
    r_hat_t   = b_R^[t]( a_hat_t, m_t )
    m_{t+1}   = b_D^[t]( a_hat_t, m_t )        (t < T)
    V(pi)     = sum_t r_hat_t

GRADIENTS. The chain is MULTILINEAR in the per-stage coefficient vectors
(linear in each block's beta with all other blocks held fixed) -- the same
structural property value_plugin.py's docstring states for the tabular chain,
and it holds here for the same reason: holding every other block fixed pins
every evaluation point m_t (t <= j) as a constant, so each block's own output
is linear in its own beta by construction of the kernel-ridge representer form.

- Reward blocks: r_hat_t = k_W(w_eval_t, W1^[t]) . beta_R^[t] is EXACTLY linear
  in beta_R^[t], so dV/d(beta_R^[t]) = k_W(w_eval_t, W1^[t]) exactly -- no
  approximation, same as the tabular case's exact gradients.

- Dynamics blocks are harder: beta_D^[j] affects m_{j+1}, which then feeds
  (nonlinearly, through the KERNEL function, not through beta_D^[j] itself)
  into every later stage's evaluation point. The exact chain-rule gradient
  needs d(kernel value)/d(its scalar input), which is well-defined for smooth
  kernels but adds real derivation/implementation risk for this first pass.
  KNOWN SIMPLIFICATION (documented in docs/continuous_extension.md): we keep
  the beta_D^[j]-linear part EXACT (m_{j+1} = k_W(w_eval_j,W1^[j]).beta_D^[j]
  is exactly linear in beta_D^[j]) and get the scalar sensitivity of the
  REST of the chain to m_{j+1} via a single 1-D central finite difference
  (perturbing the scalar m_{j+1}, not the N1-dimensional beta_D^[j] -- O(T)
  extra bridge evaluations per gradient call, not O(N1)):

    dV/d(beta_D^[j]) = ( d V_rest(m_{j+1}) / d m_{j+1} )  *  k_W(w_eval_j, W1^[j])
                        \\_____________ scalar, 1-D FD ____________/   \\_ exact outer product _/

  This is unlike the tabular pipeline's fully-exact gradients -- an explicit,
  documented trade-off, not an oversight.
"""

import numpy as np


def _eval(est, family, t, coeffs, a_val, m_val):
    """b_R^[t] or b_D^[t] at a single point w=(a_val, m_val) using the GIVEN
    (possibly perturbed) coefficient vector, not necessarily est's own fitted
    one -- needed so the pessimism coordinate-descent can evaluate the chain
    at trial points inside the confidence ellipsoid. Delegates to the
    estimator's own eval_and_feature so this works identically for the linear
    (primal, 2-D feature) and RBF (dual, N1-D feature) backends."""
    w = np.array([[a_val, m_val]])
    val, feat = est.eval_and_feature(family, t, coeffs, w)
    return float(val[0]), feat.ravel()


def chain_value(est, K_pi, c_pi, m1, bR_list, bD_list):
    """Forward chain only (no grads); bR_list[t-1]/bD_list[j-1] are the
    coefficient vectors to use (pass est.bridges()' own beta's to evaluate at
    the fitted center)."""
    T = est.T
    m = m1
    per_step = []
    for t in range(1, T + 1):
        a_t = K_pi * m + c_pi
        vR, _ = _eval(est, "R", t, bR_list[t - 1], a_t, m)
        per_step.append(vR)
        if t < T:
            m, _ = _eval(est, "D", t, bD_list[t - 1], a_t, m)
    return float(np.sum(per_step)), per_step


def chain_value_and_grads(est, K_pi, c_pi, m1, bR_list, bD_list, fd_h=1e-3):
    """Forward chain + exact reward grads + scalar-FD dynamics grads (see
    module docstring). Returns (V, per_step, (gR_list, gD_list))."""
    T = est.T

    # forward pass, recording each stage's (a_eval, m_eval) and kernel feature rows
    m = m1
    evals = []          # (a_t, m_t) per stage t=1..T
    per_step = []
    feats_R = []
    for t in range(1, T + 1):
        a_t = K_pi * m + c_pi
        evals.append((a_t, m))
        vR, kR = _eval(est, "R", t, bR_list[t - 1], a_t, m)
        per_step.append(vR)
        feats_R.append(kR)
        if t < T:
            m, _ = _eval(est, "D", t, bD_list[t - 1], a_t, m)
    V = float(np.sum(per_step))

    def v_rest(m_start, start_t):
        """Value of stages start_t..T given chain state m_start entering
        start_t, using the CURRENT (fixed) coefficient lists throughout."""
        mm = m_start
        v = 0.0
        for t in range(start_t, T + 1):
            a_t = K_pi * mm + c_pi
            vR, _ = _eval(est, "R", t, bR_list[t - 1], a_t, mm)
            v += vR
            if t < T:
                mm, _ = _eval(est, "D", t, bD_list[t - 1], a_t, mm)
        return v

    gR = [feats_R[t] for t in range(T)]          # exact
    gD = []
    for j in range(1, T):
        a_j, m_j = evals[j - 1]
        m_next, k_j = _eval(est, "D", j, bD_list[j - 1], a_j, m_j)
        dv_dm = (v_rest(m_next + fd_h, j + 1) - v_rest(m_next - fd_h, j + 1)) / (2 * fd_h)
        gD.append(dv_dm * k_j)
    return V, per_step, (gR, gD)


def empirical_initial_mean(O):
    return float(O[:, 0].mean())
