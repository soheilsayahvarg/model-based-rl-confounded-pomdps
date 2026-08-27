"""bridge_estimator_scaled.py -- the same estimator, with a compacted history index.

WHY THIS EXISTS. TabularBridgeEstimator allocates the stage-1 tables densely at
(n_w, n_x) with

    n_x = n_act * (n_obs * n_act)^(t-1) * n_o0

so the conditioning alphabet grows exponentially in t. At |O| = 6 and T = 6 that
is 2e6 columns; at T = 8 it is 2.9e8 and the allocation fails. Every horizon
result in this project is capped at T = 3 (one run at T = 6) for that reason
alone, and "the environment is a toy" is the most common thing a reviewer will
say about this work.

THE FIX IS NOT A MODEL CHANGE. x takes at most N distinct values in any sample,
so the dense alphabet is almost entirely empty columns. Remapping the observed
values to a compact range leaves every downstream quantity identical:

  - cnt_wx, cnt_x are built by scatter-add over observed (w, x) only;
  - mu[:, x2] and pmf[yidx, x2] index only observed columns;
  - `seen = cnt_x > 0` still marks values present in the stage-2 half but absent
    from the stage-1 half, so the marginal fallback is preserved exactly.

Columns dropped are those observed in NEITHER half, which are never read.

The equivalence is asserted numerically in run_estimator_equivalence.py rather
than argued here. TabularBridgeEstimator is left untouched so that all 40+ stored
result files stay reproducible.
"""

import numpy as np

from bridge_estimator import TabularBridgeEstimator, stage_indices


def compact(x_all):
    """Map an integer array to 0..k-1 over its observed values. Returns (x, k)."""
    uniq, inv = np.unique(x_all, return_inverse=True)
    return inv.astype(np.int64), int(uniq.size)


class ScaledBridgeEstimator(TabularBridgeEstimator):
    """TabularBridgeEstimator with the conditioning alphabet compacted per stage."""

    def fit(self, O0, O, A, R):
        n_obs, n_act, n_o0, n_r, T = (self.n_obs, self.n_act, self.n_o0,
                                      self.n_r, self.T)
        N = O.shape[0]
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(N)
        N1 = int(N * self.split_frac)
        i1, i2 = perm[:N1], perm[N1:]
        N2 = N - N1
        lam1 = self.lambda1 if self.lambda1 is not None else 1.0 / N1
        lam2 = self.lambda2 if self.lambda2 is not None else 0.03 / np.sqrt(N2)
        self.fitted_ = dict(N=N, N1=N1, N2=N2, lambda1=lam1, lambda2=lam2)
        self.diagnostics = []
        self.stage_store = {}
        self.n_x_compact = {}

        self.bR_hat = np.zeros((T, n_act, n_obs, n_r, n_obs))
        self.bD_hat = np.zeros((T - 1, n_act, n_obs, n_obs, n_obs))

        for t in range(1, T + 1):
            w, x, y, z, sz = stage_indices(O0, O, A, R, t, n_obs, n_act,
                                           n_o0, n_r)
            # THE ONLY CHANGE: the alphabet is the observed support, not the
            # cartesian product. Order is np.unique's, i.e. sorted by the
            # original index, so the compaction is deterministic given the data.
            xc, n_x = compact(x)
            self.n_x_compact[t] = n_x
            w1, x1, y1 = w[i1], xc[i1], y[i1]
            x2 = xc[i2]

            muR, pmfR, seenR = self._stage1_tables(
                w1, x1, y1, sz["n_w"], n_x, sz["n_y"], N1, lam1)
            bR = self._stage2_solve(muR, pmfR, seenR, x2, sz["n_w"], sz["n_y"],
                                    N2, lam2, rng, label=f"bR_t{t}")
            self.bR_hat[t - 1] = bR.reshape(n_act, n_obs, n_r, n_obs)

            if z is not None:
                z1 = z[i1]
                muD, pmfD, seenD = self._stage1_tables(
                    w1, x1, z1, sz["n_w"], n_x, sz["n_z"], N1, lam1)
                bD = self._stage2_solve(muD, pmfD, seenD, x2, sz["n_w"],
                                        sz["n_z"], N2, lam2, rng,
                                        label=f"bD_t{t}")
                self.bD_hat[t - 1] = bD.reshape(n_act, n_obs, n_obs, n_obs)
        return self
