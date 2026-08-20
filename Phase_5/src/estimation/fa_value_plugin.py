"""
fa_value_plugin.py -- Theorem-3.5 mean chain for the finite-action setting,
with EXACT gradients for both bridge families.

CHAIN. Under an observation-independent action distribution pi, the expected
bridge feature entering stage t is exactly [pi, m_t], so

    V(pi)   = sum_t  [pi, m_t] . bR_t
    m_{t+1} = [pi, m_t] . bD_t                      (t < T)

which is affine in the chain mean m_t at every stage.

GRADIENTS ARE EXACT HERE -- an improvement over Phase 3. Phase 3's continuous
plug-in had to obtain the dynamics-block sensitivity d V_rest / d m_{j+1} by a
1-D finite difference, because with a real-valued action the chain ran through
the kernel function nonlinearly. With a finite action set the policy contributes
a constant vector pi, so the entire chain is affine in m and the sensitivity has
a closed form. Writing D_t := dV/dm_t, a single backward pass gives

    D_T = bR_T[-1],        D_t = bR_t[-1] + bD_t[-1] * D_{t+1}   (t < T)

and then, exactly,

    dV/d bR_t = [pi, m_t]                (V is linear in bR_t)
    dV/d bD_j = D_{j+1} * [pi, m_j]      (chain rule through m_{j+1})

where [-1] denotes the observation coefficient of a block. No finite
differences, no step-size to tune. This removes one of the documented
limitations of the Phase 3 extension rather than carrying it forward.
"""

import numpy as np


def _feat(pi, m):
    return np.concatenate([np.asarray(pi, dtype=float), [float(m)]])


def chain_value(pi, m1, bR_list, bD_list):
    """Forward chain only. Returns (V, per_step)."""
    T = len(bR_list)
    m = float(m1)
    per_step = []
    for t in range(T):
        f = _feat(pi, m)
        per_step.append(float(f @ bR_list[t]))
        if t < T - 1:
            m = float(f @ bD_list[t])
    return float(np.sum(per_step)), per_step


def chain_value_and_grads(pi, m1, bR_list, bD_list):
    """Forward chain + EXACT gradients. Returns (V, gR_list, gD_list)."""
    T = len(bR_list)

    m = float(m1)
    feats = []
    per_step = []
    for t in range(T):
        f = _feat(pi, m)
        feats.append(f)
        per_step.append(float(f @ bR_list[t]))
        if t < T - 1:
            m = float(f @ bD_list[t])
    V = float(np.sum(per_step))

    # backward sensitivities D_t = dV/dm_t
    D = [0.0] * (T + 1)
    D[T - 1] = float(bR_list[T - 1][-1])
    for t in range(T - 2, -1, -1):
        D[t] = float(bR_list[t][-1]) + float(bD_list[t][-1]) * D[t + 1]

    gR = [feats[t].copy() for t in range(T)]
    gD = [D[j + 1] * feats[j] for j in range(T - 1)]
    return V, gR, gD


def empirical_initial_mean(O):
    return float(O[:, 0].mean())
