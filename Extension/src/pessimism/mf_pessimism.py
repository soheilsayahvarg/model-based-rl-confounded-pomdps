"""mf_pessimism.py -- a confidence region and pessimistic value for the MODEL-FREE bridge.

WHY THIS EXISTS. The Phase 4 paper showed the anchor paper's model-based pessimism
diverges, and traced it to null-space directions the confidence region cannot
constrain. Step (b) asks whether that is a property of the METHOD FAMILY. The
model-free proximal estimator of Shi et al. has a different loss, a different
solver and a different bridge definition, but its bridge is still a function on
the observation space. If the failure is geometric, it should appear there too.

The model-free estimator ships without a pessimism layer, so this adds one, using
the same geometry the model-based side uses: an ellipsoid centred on b_hat with
shape H, and the standard support-function minimum.

THE DECISIVE QUANTITY. Minimising a LINEAR functional over an ellipsoid has a
closed form:

    V_low = V(b_hat) - sqrt(xi) * ||g||_{H^-1}

so a blow-up requires two things at once: H must have (near-)null directions, AND
the value gradient g must have a component inside them. A null space alone is
harmless if the value never looks in that direction. This makes step (b) a sharp
question rather than a qualitative comparison -- we can measure the leakage of g
into the null space directly.

A STRUCTURAL DIFFERENCE WORTH STATING. For the model-based chain the value is
multilinear across every stage's bridge, which is what forced the coupled
multi-block minimisation and allowed perturbations to compound. Here the value

    J = sum_o nu1(o) sum_a b_V^[1](a, o)

depends on the FIRST stage bridge only, and linearly. So the model-free side
cannot fail through compounding. If it still diverges, the cause has to be the
geometry, which is exactly the claim under test -- and if it does NOT diverge,
that isolates the compounding structure as necessary, which is equally
informative.
"""

import numpy as np


def mf_design_and_gradient(d, n_o, n_o0, n_act, nu1, t=1, ridge=None):
    """Per-action ellipsoid shape H_a and value gradient g_a.

    H_a is the bridge-space Gram of the model-free GLS solve -- the same matrix
    whose spectrum the rank study analysed -- plus a Tikhonov ridge, mirroring
    how the model-based side builds H = T2 + lambda2 * I.

    g_a is the gradient of J with respect to b_V^[1](a, .). Since
    J = sum_o nu1(o) sum_a b_V(a,o), that gradient is simply nu1, identically for
    every action.
    """
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    eps = 1.0 / N
    rho = ridge if ridge is not None else 0.03 / np.sqrt(N)

    Hs, gs = [], []
    for a in range(n_act):
        sel = a_t == a
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_t[sel], O0[sel]), 1.0)
        Ga = np.bincount(O0[sel], minlength=n_o0).astype(float)
        Wa = 1.0 / (Ga + N * eps)
        design = (Acr * Wa[None, :]) @ Acr.T / N
        Hs.append(design + rho * np.eye(n_o))
        gs.append(np.asarray(nu1, float).copy())
    return Hs, gs


def h_inv_norm(H, g, basis=None):
    """||g||_{H^-1}, optionally restricted to a subspace.

    With basis U (n_o, k) the quantity becomes ||U^T g||_{(U^T H U)^-1}: the
    pessimism is only allowed to move inside the retained subspace, so any part
    of g lying outside it contributes nothing to the width.
    """
    if basis is not None and basis.size:
        Hs = basis.T @ H @ basis
        gs = basis.T @ g
        return float(np.sqrt(max(gs @ np.linalg.solve(Hs, gs), 0.0)))
    return float(np.sqrt(max(g @ np.linalg.solve(H, g), 0.0)))


def null_leakage(H, g, basis):
    """Fraction of ||g||^2 lying OUTSIDE the retained subspace.

    This is the quantity that decides whether a null space matters. If the value
    gradient is orthogonal to the unconstrained directions, leakage is 0 and the
    region cannot be exploited no matter how degenerate H is.
    """
    if basis is None or basis.size == 0:
        return 1.0
    proj = basis @ (basis.T @ g)
    tot = float(g @ g)
    return float(max(0.0, 1.0 - (proj @ proj) / tot)) if tot > 0 else 0.0


def parallel_analysis_basis(d, n_o, n_o0, a, rng, t=1, B=40, q=95.0):
    """Rank selection without a spectral gap, then the matching eigenvector basis.

    Permuting O_0 within the action bin destroys the (o_t, o_0) dependence while
    preserving both marginals, so the permuted spectrum is exactly what sampling
    noise alone produces. Directions above the q-th percentile of that null carry
    real dependence.

    The null's own outer product of marginals is rank 1, so the test can only see
    dependence BEYOND the first direction; the count is therefore
    1 + #{i >= 2 : lambda_i > q95(null_i)}. Without this the selector returns 0
    whenever the behavior policy is deterministic.
    """
    o_t, a_t, O0 = d["O"][:, t - 1], d["A"][:, t - 1], d["O0"]
    N = len(a_t)
    sel = a_t == a
    o_sel, O0_sel = o_t[sel], O0[sel]

    def spec(o_arr, x_arr):
        Acr = np.zeros((n_o, n_o0))
        np.add.at(Acr, (o_arr, x_arr), 1.0)
        Acr /= N
        ev, evec = np.linalg.eigh(Acr @ Acr.T)
        return ev[::-1], evec[:, ::-1]

    obs_ev, obs_vec = spec(o_sel, O0_sel)
    null = np.empty((B, n_o))
    for b in range(B):
        null[b] = spec(o_sel, rng.permutation(O0_sel))[0]
    thr = np.percentile(null, q, axis=0)

    k = 1
    for i in range(1, n_o):
        if obs_ev[i] > thr[i]:
            k += 1
        else:
            break
    return k, obs_vec[:, :k].copy()
