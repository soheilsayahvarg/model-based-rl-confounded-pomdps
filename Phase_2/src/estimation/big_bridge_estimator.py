"""
big_bridge_estimator.py — the two-stage bridge estimator SCALED to the full
720x2 benchmark environment, via three exact factorizations (no approximation
beyond documented, logged fallbacks):

(F1) FACTORED/HASHED X_t. The raw conditioning set X_t = (A_t, H_{t-1}, O_0) has
     |X_3| ~ 2e9 dense cells; but the environment's observation design makes the
     per-stage estimation decompose (F3), and all remaining grouping keys are
     hashed composites of OBSERVED values only (np.unique / flat bincount over
     visited cells) — memory scales with the data, never with |X_t|.

(F2) CORE-DIAGONAL BRIDGE SUPPORT. The core channel of the observation is
     DETERMINISTIC given the latent state (o = 2x + m, x = core). Hence the true
     bridges satisfy b(a, o~=(x~, m~), y=(., o=(x_o, m_o))) = 0 unless x~ = x_o:
     the moment equations couple only within a core. This is a property of the
     observation design, not an assumption about the data.

(F3) PER-CELL SQUARE SYSTEMS. Within a cell (t, x, a), the only stochastic,
     latent-dependent channel is the binary marker m (P(m=1|u) = q_u), and the
     negative control O_0 is binary. The two-stage estimator collapses to the
     EXACT closed form of its square case: with

         M2[o_0, m~] = P_hat(m_t = m~ | o_0, cell)      (2 x 2, ridge-smoothed)
         Ytab[o_0, y] = P_hat(y | o_0, cell)            (2 x |Y_cell|)

     the bridge table solves  M2 @ B = Ytab, i.e. B = M2^{-1} Ytab per cell —
     the same object the oracle computes with true quantities (per-core marker
     inversion), and the unique solution (square design => NO structural null
     space; the signal subspace is the full space, so signal projection is the
     identity here — the projected/vanilla distinction only exists when |O|>|S|).

     Identification note: conditioning on (o_0) within the (t, x, a) cell is a
     valid restriction of the anchor's X_t — the cell event {core(o_t)=x, a_t=a}
     is measurable w.r.t. (S_t, A_t) (the core channel is deterministic in S_t),
     so the latent pointwise moment equation integrates to the cell-conditional
     one; the latent 2x2 system per cell has a UNIQUE solution, and the
     observable 2x2 is invertible iff o_0 stays informative about u within the
     cell (eps0 < 0.5) — the restricted conditioning loses nothing (square case).

     DOCUMENTED DEVIATION — no N1/N2 sample split: in the square direct-inversion
     limit the POINT ESTIMATE is unchanged by splitting (the stage-2 ERM minimizer
     (M2^T D M2)^{-1} M2^T D Ytab collapses to M2^{-1} Ytab for any positive o_0-
     frequency weighting D), so all data is used for both tables. CAVEAT: reusing
     the counts makes the numerator (Ytab) and the inverted denominator (M2)
     statistically dependent, introducing the usual O(1/n) ratio/inversion bias
     and VOIDING the independence the anchor's split needs for the Theorem 4.2
     rate — the point estimate is consistent, but the paper's rate guarantee no
     longer applies. (The toy estimator keeps the faithful split; cross-fitting is
     the identified upgrade for this module.)

Estimated objects:
  bR[t, x, a, m~, r, m_o]   dense (T, 720, 8, 2, 3, 2), rewards r in {-1, 0, +1}
  bD entries                sparse triplets per stage: (x, a, m~, o', m_o, value)
                            only for OBSERVED (cell -> o') transitions
Chain evaluation (policy-folded, dense 1442 x 1442 operators — 16 MB each):
  M_t[o~, o~'] = sum_{a, m_o} pi(a | 2x + m_o) * bD[t, x, a, m~, o~', m_o],
  terminal self-loops appended exactly; reward head from bR; V = sum_t G_t . B_t.

Fallbacks (logged): cells with a singular / data-poor M2 fall back to the
confounding-naive within-cell table (B = Ytab averaged over o_0) — never silent.
Two coverage metrics are exposed: fallback_cell_frac (UNWEIGHTED fraction of
visited cells that fall back) and fallback_mass_frac (VISIT-WEIGHTED fraction of
trajectory-steps whose cell falls back). The mass fraction is the honest impact
measure: a few high-traffic cells routing to the naive table can dominate the
value error while showing a tiny cell fraction, so both are reported.
"""

import numpy as np

N_X = 720
N_A = 8
N_M = 2
N_O0 = 2
N_R = 3                      # reward levels {-1, 0, +1} -> index r + 1
N_OBS = 1442
OBS_NEG, OBS_POS = 1440, 1441


# --------------------------------------------------------------------- helpers
def _cell_key(t, x, a):
    return (t * N_X + x) * N_A + a


class BigBridgeEstimator:
    """Factored per-core two-stage bridge estimator for the 720x2 benchmark."""

    def __init__(self, T=3, ridge_pseudo=1.0, det_tol=1e-3):
        self.T = T
        self.ridge_pseudo = ridge_pseudo
        self.det_tol = det_tol

    # ---------------------------------------------------------------- fit
    def fit(self, O0, O, A, R):
        T = self.T
        N = O.shape[0]
        n_cells = T * N_X * N_A

        # ---- accumulate per-(cell, o0) tables over live-observation steps
        cnt_m = np.zeros((n_cells, N_O0, N_M))            # counts of m_t
        cnt_y = np.zeros((n_cells, N_O0, N_R * N_M))      # counts of (r_t, m_o)
        z_keys_all, z_cnts_all = [], []                   # sparse (cell,o0,m,z)

        for t in range(T):
            o_t = O[:, t]
            live = o_t < 2 * N_X
            x = o_t[live] // 2
            m = o_t[live] % 2
            a = A[live, t]
            o0 = O0[live]
            r_idx = (R[live, t] + 1).astype(np.int64)     # {-1,0,1} -> {0,1,2}
            cell = _cell_key(t, x, a)

            np.add.at(cnt_m, (cell, o0, m), 1.0)
            np.add.at(cnt_y, (cell, o0, r_idx * N_M + m), 1.0)

            if t < T - 1:
                o_next = O[live, t + 1]                   # z = (o_{t+1}, m_o)
                zflat = (((cell * N_O0 + o0) * N_M + m) * N_OBS + o_next)
                keys, cnts = np.unique(zflat, return_counts=True)
                z_keys_all.append(keys)
                z_cnts_all.append(cnts.astype(np.float64))

        # ---- per-cell 2x2 inversion, vectorized across all visited cells
        row_tot = cnt_m.sum(axis=2)                                  # (cells, 2)
        p = self.ridge_pseudo
        M2 = (cnt_m + p / N_M) / (row_tot + p)[:, :, None]           # ridge CME
        visited = row_tot.sum(axis=1) > 0
        both_o0 = (row_tot > 0).all(axis=1)

        det = M2[:, 0, 0] * M2[:, 1, 1] - M2[:, 0, 1] * M2[:, 1, 0]
        ok = both_o0 & (np.abs(det) > self.det_tol)
        self.cell_ok = ok
        self.cell_visited = visited
        inv = np.zeros((n_cells, 2, 2))
        d = np.where(ok, det, 1.0)
        inv[:, 0, 0] = M2[:, 1, 1] / d
        inv[:, 1, 1] = M2[:, 0, 0] / d
        inv[:, 0, 1] = -M2[:, 0, 1] / d
        inv[:, 1, 0] = -M2[:, 1, 0] / d

        # reward-bridge tables: B = M2^{-1} @ Ytab  (fallback: o0-pooled naive)
        y_tot = cnt_y.sum(axis=2)
        Ytab = cnt_y / np.maximum(y_tot, 1.0)[:, :, None]            # (cells,2,6)
        pooled = cnt_y.sum(axis=1) / np.maximum(cnt_y.sum(axis=(1, 2)), 1.0)[:, None]
        B_R = np.where(ok[:, None, None],
                       np.einsum("cij,cjy->ciy", inv, Ytab),
                       np.repeat(pooled[:, None, :], N_M, axis=1))
        B_R[~visited] = 0.0
        self.bR = B_R.reshape(T, N_X, N_A, N_M, N_R, N_M)            # [t,x,a,m~,r,m_o]

        # dynamic-bridge sparse entries, stored as flat triplets keyed by
        # (cell, m_o, o_next): the realized m_t plays the index role m_o and
        # the realized o_{t+1} plays the output role o~'
        if z_keys_all:
            zk = np.concatenate(z_keys_all)
            zc = np.concatenate(z_cnts_all)
            # decode
            o_next = zk % N_OBS
            rest = zk // N_OBS
            m_o = rest % N_M
            rest //= N_M
            o0 = rest % N_O0
            cell = rest // N_O0
            # normalize to conditional pmf within (cell, o0):  P(o', m_o | o0)
            denom = row_tot[cell, o0]                     # counts within (cell,o0)
            pz = zc / np.maximum(denom, 1.0)
            # per-entry bridge values over m~: B[m~] = sum_{o0} inv[m~, o0] * pz(o0)
            # accumulate entries keyed by (cell, m_o, o_next), summing over o0
            ekey = (cell * N_M + m_o) * N_OBS + o_next
            uek, einv = np.unique(ekey, return_inverse=True)
            vals = np.zeros((uek.size, N_M))
            for o0v in range(N_O0):
                sel = o0 == o0v
                contrib = pz[sel]
                idx = einv[sel]
                for mt in range(N_M):
                    np.add.at(vals[:, mt], idx,
                              inv[cell[sel], mt, o0v] * contrib)
            # fallback cells: pooled next-obs pmf (identical for both m~)
            fcell = uek // (N_M * N_OBS)
            bad = ~ok[fcell]
            if bad.any():
                pooled_z = np.zeros(uek.size)
                np.add.at(pooled_z, einv, zc)             # o0-pooled counts
                tot_cell = np.maximum(row_tot.sum(axis=1)[fcell], 1.0)
                for mt in range(N_M):
                    vals[bad, mt] = pooled_z[bad] / tot_cell[bad]
            self._zk = uek
            self._zvals = vals                            # (n_entries, m~)
        else:
            self._zk = np.zeros(0, dtype=np.int64)
            self._zvals = np.zeros((0, N_M))

        vis_idx = visited.nonzero()[0]
        self.fallback_cell_frac = float((~ok[vis_idx]).mean()) if vis_idx.size else 1.0
        # visit-weighted (mass) fallback fraction: honest impact measure
        cell_visits = row_tot.sum(axis=1)                      # (n_cells,)
        tot_visits = cell_visits.sum()
        self.fallback_mass_frac = (
            float(cell_visits[~ok].sum() / tot_visits) if tot_visits > 0 else 1.0)
        return self

    # ---------------------------------------------------------------- chain ops
    def folded_operator(self, t, pi_obs):
        """Dense policy-folded M_t[o~, o~'] (1442 x 1442) from sparse bD entries,
        with exact terminal self-loops."""
        M = np.zeros((N_OBS, N_OBS))
        zk, vals = self._zk, self._zvals
        o_next = zk % N_OBS
        rest = zk // N_OBS
        m_o = rest % N_M
        cell = rest // N_M
        tt = cell // (N_X * N_A)
        sel = tt == t
        if sel.any():
            cell_s = cell[sel]
            x = (cell_s // N_A) % N_X
            a = cell_s % N_A
            w = pi_obs[2 * x + m_o[sel], a]               # pi(a | o=(x, m_o))
            for mt in range(N_M):
                np.add.at(M, (2 * x + mt, o_next[sel]), w * vals[sel, mt])
        M[OBS_NEG, OBS_NEG] = 1.0
        M[OBS_POS, OBS_POS] = 1.0
        return M

    def plugin_value(self, pi_obs, p_o1):
        """V(pi, b_hat) by the exact chain: G_{t+1} = G_t M_t; V = sum_t G_t.B_t."""
        T = self.T
        G = np.asarray(p_o1, dtype=np.float64).copy()
        V = 0.0
        per_step = []
        for t in range(T):
            B = self._reward_head(t, pi_obs)
            step = float(G @ B)
            per_step.append(step)
            V += step
            if t < T - 1:
                G = G @ self.folded_operator(t, pi_obs)
        return V, per_step

    def _reward_head(self, t, pi_obs):
        r_w = np.array([-1.0, 0.0, 1.0])
        pi_x = pi_obs[:2 * N_X].reshape(N_X, N_M, N_A)    # [x, m_o(=m), a]
        # bR[t]: [x, a, m~, r, m_o];  head[x, m~] = sum_{a, r, m_o} pi[x,m_o,a] r bR
        head = np.einsum("xna,xamrn,r->xm", pi_x, self.bR[t], r_w, optimize=True)
        out = np.zeros(N_OBS)
        out[:2 * N_X] = head.reshape(-1)                  # o~ = 2x + m~
        return out
