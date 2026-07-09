"""
shrinked_big_bridge_estimator.py — variance-reduced drop-in for the 720x2
benchmark estimator (Agent 1 design). BACKWARD-COMPATIBLE: the default
(shrinkage='none', k_folds=1) dispatches to the parent fit() and is byte-identical.

WHY. The parent forms per cell (t,x,a) the 2x2 CME M2[o0,m~]=P_hat(m|o0) and
Ytab[o0,y]=P_hat(y|o0), then B = M2^{-1} Ytab (exactly-identified). In low-count /
kappa=0 cells the empirical det(M2) is noisy (var ~ 1/n), so inverting M2 blows up
the bridge variance -- which is why the plug-in loses to naive in 15/16 benchmark
cells. Two principled, composable fixes:

(A) STRUCTURAL PARTIAL POOLING of M2 (shrinkage='pool'|'js').
    The TRUE M2 factors through a CELL-INDEPENDENT emission:
        M2*[o0,:] = (1-r_{o0}) Em2[0,:] + r_{o0} Em2[1,:],  Em2=[[1-q0,q0],[1-q1,q1]],
    so per-(t,a) group the rows live in a shared 2-parameter family. Pool the raw
    marker counts across cores x within (t,a) to a low-variance prior M2_pool, and
    shrink each noisy cell row toward it with an empirical-Bayes weight
        w_{c,o0} = n / (n + tau)        ('pool', fixed tau = tau_scale)
    or a James-Stein group factor w = max(0, 1 - tau_G / n)  ('js').
    w->1 as n->inf (consistency preserved), w->0 in the noisiest cells (max pooling
    exactly where the parent blows up). det(M2_shrink) is pushed toward the
    well-conditioned det(M2_pool)=0.40*(r1_pool-r0_pool), stabilizing the inverse.
    Crucially the pool is still CONFOUNDING-AWARE (uses o0), unlike the naive pool.

(B) K-FOLD CROSS-FITTING (k_folds=K>=2). Estimate the denominator M2 out-of-fold
    and the numerator Ytab (and dynamic-bridge pz) in-fold, average bridges over
    folds -> numerator and inverted denominator use disjoint samples -> removes the
    O(1/n) ratio/inversion bias the count-reuse introduces (restores the anchor's
    stage independence). Composes with shrinkage (shrink M2_{-k} before inverting).

The chain ops (folded_operator, _reward_head, plugin_value) are INHERITED unchanged
-- they read only self.bR / self._zk / self._zvals, whose shapes are identical.
"""

import numpy as np

from big_bridge_estimator import (
    BigBridgeEstimator, N_X, N_A, N_M, N_O0, N_R, N_OBS, _cell_key)


# --------------------------------------------------------------------- helpers
def _group_ids(shape_cells, pool_group):
    """Map each cell index c=(t*720+x)*8+a to a pooling-group id.
    'ta' pools cores x within the same (t, a); 'marg' pools within t only."""
    c = np.arange(shape_cells)
    a = c % N_A
    x = (c // N_A) % N_X
    t = c // (N_X * N_A)
    if pool_group == "ta":
        return t * N_A + a                      # pool over x within (t,a)
    return t.copy()                              # 'marg': pool over x and a within t


def _pool_M2(cnt_m, ridge_pseudo, group_ids):
    """Pooled 2x2 CME per cell: aggregate marker counts across the cell's group,
    then ridge-normalize. Returns (n_cells, 2, 2)."""
    n_cells = cnt_m.shape[0]
    n_groups = int(group_ids.max()) + 1
    # sum counts within each group: (n_groups, N_O0, N_M)
    gsum = np.zeros((n_groups, N_O0, N_M))
    np.add.at(gsum, group_ids, cnt_m)
    row = gsum.sum(axis=2)                                    # (n_groups, N_O0)
    M2p = (gsum + ridge_pseudo / N_M) / (row + ridge_pseudo)[:, :, None]
    return M2p[group_ids]                                     # broadcast to cells


def _shrink_M2(M2_raw, M2_pool, row_tot, mode, tau_scale, group_ids):
    """Convex-combine each cell row toward its pooled row with an EB/JS weight.
    Returns (M2_shrink, w) with w shape (n_cells, N_O0)."""
    n = row_tot                                              # (n_cells, N_O0) counts
    if mode == "js":
        # group-level James-Stein: w = max(0, 1 - tau_G / n), tau_G from between/within
        n_groups = int(group_ids.max()) + 1
        w = np.ones_like(n)
        # per-row sampling variance proxy: sum_m p(1-p)/n  (p from raw rows)
        p = M2_raw
        var_row = (p * (1 - p)).sum(axis=2) / np.maximum(n, 1.0)   # (cells, o0)
        ssb = ((M2_raw - M2_pool) ** 2).sum(axis=2)               # (cells, o0)
        for g in range(n_groups):
            sel = group_ids == g
            for o0 in range(N_O0):
                m = sel & (n[:, o0] > 0)
                if m.sum() >= 3:
                    D = m.sum() * N_M
                    sig2 = var_row[m, o0].mean()
                    SSB = ssb[m, o0].sum()
                    tau_g = max(0.0, (D - 2) * sig2 / max(SSB, 1e-12))
                    w[m, o0] = np.clip(1.0 - tau_g / np.maximum(n[m, o0], 1.0), 0.0, 1.0)
    else:  # 'pool': fixed-tau empirical-Bayes weight
        w = n / (n + tau_scale)
    M2s = w[:, :, None] * M2_raw + (1.0 - w)[:, :, None] * M2_pool
    return M2s, w


def _fold_assign(N, K, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    fold = np.empty(N, dtype=np.int64)
    for k, idx in enumerate(np.array_split(perm, K)):
        fold[idx] = k
    return fold


# --------------------------------------------------------------------- estimator
class ShrinkedBigBridgeEstimator(BigBridgeEstimator):
    """Variance-reduced BigBridgeEstimator. See module docstring.

    Parameters (all default to the parent's exact behavior):
      shrinkage : 'none' | 'pool' | 'js'   (default 'none')
      tau_scale : fixed-tau EB weight scale for 'pool' (default 8.0)
      pool_group: 'ta' (pool cores x within (t,a)) | 'marg' (default 'ta')
      k_folds   : 1 (no cross-fit, parent path) | K>=2 (default 1)
      cf_seed   : cross-fit fold seed (default 0)
    """

    def __init__(self, T=3, ridge_pseudo=1.0, det_tol=1e-3,
                 shrinkage="none", tau_scale=8.0, pool_group="ta",
                 k_folds=1, cf_seed=0):
        super().__init__(T=T, ridge_pseudo=ridge_pseudo, det_tol=det_tol)
        self.shrinkage = shrinkage
        self.tau_scale = float(tau_scale)
        self.pool_group = pool_group
        self.k_folds = int(k_folds)
        self.cf_seed = cf_seed

    def fit(self, O0, O, A, R):
        if self.shrinkage == "none" and self.k_folds == 1:
            return super().fit(O0, O, A, R)          # BYTE-IDENTICAL default
        if self.k_folds >= 2:
            return self._fit_crossfit(O0, O, A, R)
        return self._fit_single(O0, O, A, R, rows=np.arange(O.shape[0]),
                                m2_rows=None)

    # ---- one accumulation pass over a row subset; if m2_src given, M2 uses
    #      m2_src rows (cross-fit denominator) while Y/z use `rows` (numerator).
    def _accumulate(self, O0, O, A, R, rows):
        T = self.T
        n_cells = T * N_X * N_A
        cnt_m = np.zeros((n_cells, N_O0, N_M))
        cnt_y = np.zeros((n_cells, N_O0, N_R * N_M))
        z_keys_all, z_cnts_all = [], []
        for t in range(T):
            o_t = O[rows, t]
            live = o_t < 2 * N_X
            x = o_t[live] // 2
            m = o_t[live] % 2
            a = A[rows, t][live]
            o0 = O0[rows][live]
            r_idx = (R[rows, t][live] + 1).astype(np.int64)
            cell = _cell_key(t, x, a)
            np.add.at(cnt_m, (cell, o0, m), 1.0)
            np.add.at(cnt_y, (cell, o0, r_idx * N_M + m), 1.0)
            if t < T - 1:
                o_next = O[rows, t + 1][live]
                zflat = (((cell * N_O0 + o0) * N_M + m) * N_OBS + o_next)
                keys, cnts = np.unique(zflat, return_counts=True)
                z_keys_all.append(keys)
                z_cnts_all.append(cnts.astype(np.float64))
        return cnt_m, cnt_y, z_keys_all, z_cnts_all

    # ---- build M2 (optionally shrunk) + inverse from a marker-count table
    def _build_inv(self, cnt_m):
        n_cells = cnt_m.shape[0]
        row_tot = cnt_m.sum(axis=2)
        p = self.ridge_pseudo
        M2_raw = (cnt_m + p / N_M) / (row_tot + p)[:, :, None]
        if self.shrinkage in ("pool", "js"):
            gids = _group_ids(n_cells, self.pool_group)
            M2_pool = _pool_M2(cnt_m, p, gids)
            M2, w = _shrink_M2(M2_raw, M2_pool, row_tot, self.shrinkage,
                               self.tau_scale, gids)
            self.shrink_w_mean = float(w[row_tot > 0].mean()) if (row_tot > 0).any() else 1.0
        else:
            M2 = M2_raw
            self.shrink_w_mean = 1.0
        both_o0 = (row_tot > 0).all(axis=1)
        det = M2[:, 0, 0] * M2[:, 1, 1] - M2[:, 0, 1] * M2[:, 1, 0]
        ok = both_o0 & (np.abs(det) > self.det_tol)
        inv = np.zeros((n_cells, 2, 2))
        d = np.where(ok, det, 1.0)
        inv[:, 0, 0] = M2[:, 1, 1] / d
        inv[:, 1, 1] = M2[:, 0, 0] / d
        inv[:, 0, 1] = -M2[:, 0, 1] / d
        inv[:, 1, 0] = -M2[:, 1, 0] / d
        return inv, ok, row_tot

    # ---- assemble bR and dynamic-bridge triplets from (inv, ok, counts)
    def _assemble(self, inv, ok, row_tot, cnt_y, z_keys_all, z_cnts_all):
        T = self.T
        y_tot = cnt_y.sum(axis=2)
        Ytab = cnt_y / np.maximum(y_tot, 1.0)[:, :, None]
        pooled = cnt_y.sum(axis=1) / np.maximum(cnt_y.sum(axis=(1, 2)), 1.0)[:, None]
        visited = row_tot.sum(axis=1) > 0
        B_R = np.where(ok[:, None, None],
                       np.einsum("cij,cjy->ciy", inv, Ytab),
                       np.repeat(pooled[:, None, :], N_M, axis=1))
        B_R[~visited] = 0.0
        bR = B_R.reshape(T, N_X, N_A, N_M, N_R, N_M)

        if z_keys_all:
            zk = np.concatenate(z_keys_all)
            zc = np.concatenate(z_cnts_all)
            o_next = zk % N_OBS
            rest = zk // N_OBS
            m_o = rest % N_M
            rest //= N_M
            o0 = rest % N_O0
            cell = rest // N_O0
            denom = row_tot[cell, o0]
            pz = zc / np.maximum(denom, 1.0)
            ekey = (cell * N_M + m_o) * N_OBS + o_next
            uek, einv = np.unique(ekey, return_inverse=True)
            vals = np.zeros((uek.size, N_M))
            for o0v in range(N_O0):
                sel = o0 == o0v
                idx = einv[sel]
                for mt in range(N_M):
                    np.add.at(vals[:, mt], idx, inv[cell[sel], mt, o0v] * pz[sel])
            fcell = uek // (N_M * N_OBS)
            bad = ~ok[fcell]
            if bad.any():
                pooled_z = np.zeros(uek.size)
                np.add.at(pooled_z, einv, zc)
                tot_cell = np.maximum(row_tot.sum(axis=1)[fcell], 1.0)
                for mt in range(N_M):
                    vals[bad, mt] = pooled_z[bad] / tot_cell[bad]
        else:
            uek = np.zeros(0, dtype=np.int64)
            vals = np.zeros((0, N_M))
        return bR, uek, vals, visited

    def _finalize(self, bR, uek, vals, ok, visited, row_tot):
        self.bR = bR
        self._zk = uek
        self._zvals = vals
        self.cell_ok = ok
        self.cell_visited = visited
        vis_idx = visited.nonzero()[0]
        self.fallback_cell_frac = float((~ok[vis_idx]).mean()) if vis_idx.size else 1.0
        cv = row_tot.sum(axis=1)
        tot = cv.sum()
        self.fallback_mass_frac = float(cv[~ok].sum() / tot) if tot > 0 else 1.0
        self.k_folds_used = self.k_folds
        return self

    # ---- shrinkage only (no cross-fit)
    def _fit_single(self, O0, O, A, R, rows, m2_rows):
        cnt_m, cnt_y, zk, zc = self._accumulate(O0, O, A, R, rows)
        inv, ok, row_tot = self._build_inv(cnt_m)
        bR, uek, vals, visited = self._assemble(inv, ok, row_tot, cnt_y, zk, zc)
        return self._finalize(bR, uek, vals, ok, visited, row_tot)

    # ---- K-fold cross-fitting: M2 out-of-fold, Ytab/pz in-fold, average bridges
    def _fit_crossfit(self, O0, O, A, R):
        N = O.shape[0]
        K = self.k_folds
        fold = _fold_assign(N, K, self.cf_seed)
        bR_acc = None
        z_val_acc = {}      # ekey -> [sum vals(m~), count]
        ok_any = None
        visited_any = None
        row_tot_last = None
        w_means = []
        fb_cell, fb_mass = [], []
        for k in range(K):
            in_rows = np.where(fold == k)[0]
            out_rows = np.where(fold != k)[0]
            # denominator M2 from OUT-of-fold; numerator Y/z from IN-fold
            cnt_m_out, _, _, _ = self._accumulate(O0, O, A, R, out_rows)
            _, cnt_y_in, zk_in, zc_in = self._accumulate(O0, O, A, R, in_rows)
            inv, ok, row_tot_out = self._build_inv(cnt_m_out)
            w_means.append(self.shrink_w_mean)
            # normalize in-fold z-pz against IN-fold denominators, invert with OUT M2
            row_tot_in = cnt_y_in.sum(axis=2)  # not used for z; use marker in-fold counts
            # need in-fold marker counts for pz denominators:
            cnt_m_in, _, _, _ = self._accumulate(O0, O, A, R, in_rows)
            rti = cnt_m_in.sum(axis=2)
            bR_k, uek_k, vals_k, visited_k = self._assemble(
                inv, ok, rti, cnt_y_in, zk_in, zc_in)
            bR_acc = bR_k if bR_acc is None else bR_acc + bR_k
            ok_any = ok if ok_any is None else (ok_any | ok)
            visited_any = visited_k if visited_any is None else (visited_any | visited_k)
            row_tot_last = rti
            for key, v in zip(uek_k, vals_k):
                if key in z_val_acc:
                    z_val_acc[key][0] += v; z_val_acc[key][1] += 1
                else:
                    z_val_acc[key] = [v.copy(), 1]
            fb_cell.append(self.fallback_cell_frac if hasattr(self, "fallback_cell_frac") else 0.0)
        bR = bR_acc / K
        if z_val_acc:
            uek = np.array(sorted(z_val_acc.keys()), dtype=np.int64)
            vals = np.stack([z_val_acc[k][0] / z_val_acc[k][1] for k in uek])
        else:
            uek = np.zeros(0, dtype=np.int64); vals = np.zeros((0, N_M))
        self.shrink_w_mean = float(np.mean(w_means))
        return self._finalize(bR, uek, vals, ok_any, visited_any, row_tot_last)
