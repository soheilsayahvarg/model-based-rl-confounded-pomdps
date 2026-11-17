"""Deterministic scientific regression checks; no stored experiments overwritten."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ("envs", "estimation", "pessimism"):
    sys.path.insert(0, str(ROOT / "src" / folder))
sys.path.insert(0, str(ROOT / "poc"))
from dim_separated_pomdp import default_params, sample_trajectories, population_cross_moment
from bridge_estimator import TabularBridgeEstimator
from bridge_estimator_scaled import ScaledBridgeEstimator
from ellipsoid_opt import BlockEllipsoid
from run_completeness_beta import population_design_span
from run_selfheal_formula import make_params, span_closedform, span_bruteforce
from value_plugin import plugin_value


class ScientificChecks(unittest.TestCase):
    def test_floor_random_psd(self):
        rng = np.random.default_rng(11)
        for _ in range(30):
            q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
            design = (q * np.array([0., 0., .1, .3, 1., 2.])) @ q.T
            lam = 10 ** rng.uniform(-5, 0)
            h = design + lam * np.eye(6)
            center = q[:, 2:] @ rng.normal(size=4)
            truth, g = rng.normal(size=(2, 6))
            d = truth - center
            xi = d @ h @ d
            width = np.sqrt(xi * (g @ np.linalg.solve(h, g)))
            floor = np.linalg.norm(q[:, :2].T @ truth) * np.linalg.norm(q[:, :2].T @ g)
            self.assertGreaterEqual(width + 1e-9, floor)

    def test_large_marginal_width_does_not_preclude_ranking(self):
        # Both policies share the same unidentified component. Its width cancels
        # exactly from their difference, although each marginal width is 10.
        center = np.array([2., 0.])
        g_a, g_b = np.array([1., 1.]), np.array([0., 1.])
        values = [(g_a @ (center + [0., u]), g_b @ (center + [0., u]))
                  for u in (-10., 0., 10.)]
        self.assertTrue(all(abs(a - b - 2.) < 1e-12 for a, b in values))
        self.assertGreater(10., g_a @ center - g_b @ center)

    def test_pairwise_affine_offsets(self):
        h = np.array([[2., .2], [.2, 1.]])
        center = np.array([.3, -.1])
        gradient = np.array([1.2, -.7])
        xi, offset = .4, -3.
        hinvg = np.linalg.solve(h, gradient)
        minimizer = center - np.sqrt(xi / (gradient @ hinvg)) * hinvg
        predicted = offset + gradient @ center - np.sqrt(xi * (gradient @ hinvg))
        self.assertAlmostEqual(offset + gradient @ minimizer, predicted, places=12)
        delta = minimizer - center
        self.assertAlmostEqual(delta @ h @ delta, xi, places=12)
        # Omitting the constant can reverse the certified ordering.
        same_gradient = np.array([1., 0.])
        self.assertEqual(same_gradient @ center - same_gradient @ center, 0.)
        self.assertLess(offset, 0.)

    def test_floor_needs_null_center_condition(self):
        # A region centred on truth need not have the claimed truth-null floor.
        h = np.diag([.1, 1.1])
        center = np.array([5., 0.])
        gradient = np.array([1., 0.])
        xi = .01
        width = np.sqrt(xi * (gradient @ np.linalg.solve(h, gradient)))
        self.assertLess(width, 5.)
        # The general geometric bound uses the null displacement, not truth alone.
        truth = center + np.array([.2, 0.])
        d = truth - center
        xi_cover = d @ h @ d
        width_cover = np.sqrt(xi_cover * (gradient @ np.linalg.solve(h, gradient)))
        self.assertAlmostEqual(width_cover, .2)

    def test_history_requires_transition_rank(self):
        for family, expected in (("dense", 4), ("dense_lowrank", 2)):
            p = make_params((4, 8, 3), .0, family)
            closed, _ = span_closedform(p, 0, 2)
            brute = span_bruteforce(p, 0, 2)
            # The brute helper returns a basis directly.
            if isinstance(brute, tuple):
                brute = brute[0]
            self.assertEqual(closed.shape[1], expected)
            self.assertLess(np.linalg.norm(closed @ closed.T - brute @ brute.T), 1e-8)

    def test_unconfounded_profile_in_span(self):
        p = default_params(n_s=4, n_o=6, n_o0=2, confound=0)
        nu = p["E"].T @ p["p1"]
        for a in range(2):
            u = population_design_span(p, a)
            self.assertLess(np.linalg.norm(nu - u @ u.T @ nu), 1e-12)

    def test_cross_moment_conditions_on_action(self):
        p = default_params(confound=1.)
        marginal = population_cross_moment(p)
        conditional = population_cross_moment(p, a=0)
        self.assertEqual(np.linalg.matrix_rank(marginal), 2)
        self.assertEqual(np.linalg.matrix_rank(conditional), 1)
        self.assertAlmostEqual(conditional.sum(), 1.)
        self.assertGreater(np.linalg.norm(marginal - conditional), .01)

    def test_compaction_and_dual(self):
        p = default_params(n_s=4, n_o=6, n_o0=2, T=3, confound=.9)
        d = sample_trajectories(p, 180, np.random.default_rng(2))
        fits = []
        for cls, mode in ((TabularBridgeEstimator, "primal"),
                          (ScaledBridgeEstimator, "primal"),
                          (TabularBridgeEstimator, "dual")):
            e = cls(6, 2, 2, 2, 3, mode=mode, seed=2)
            e.fit(d["O0"], d["O"], d["A"], d["R"])
            fits.append(e)
        np.testing.assert_array_equal(fits[0].bR_hat, fits[1].bR_hat)
        np.testing.assert_array_equal(fits[0].bD_hat, fits[1].bD_hat)
        np.testing.assert_allclose(fits[0].bR_hat, fits[2].bR_hat, atol=1e-11)
        np.testing.assert_allclose(fits[0].bD_hat, fits[2].bD_hat, atol=1e-11)

    def test_value_gradients(self):
        rng = np.random.default_rng(3)
        br, bd = rng.normal(size=(3, 2, 3, 2, 3)), rng.normal(size=(2, 2, 3, 3, 3))
        pi, p0 = np.full((3, 2), .5), np.full(3, 1 / 3)
        _, _, (gr, gd) = plugin_value(br, bd, pi, p0, return_grads=True)
        eps = 1e-6
        for array, gradient in ((br, gr), (bd, gd)):
            direction = rng.normal(size=array.shape)
            before = array.copy()
            array[:] = before + eps * direction
            vp = plugin_value(br, bd, pi, p0)[0]
            array[:] = before - eps * direction
            vm = plugin_value(br, bd, pi, p0)[0]
            array[:] = before
            self.assertAlmostEqual((vp - vm) / (2 * eps), float(np.sum(gradient * direction)), places=6)

    def test_class_bounds_not_silently_ignored(self):
        block = BlockEllipsoid(np.eye(2), np.zeros(2), 1., 10, "test")
        solution, _ = block.linear_min(np.array([-2., -1.]), 100., M=1., M_inf=.8)
        np.testing.assert_allclose(solution, [.8, .6], atol=1e-6)
        self.assertLessEqual(np.linalg.norm(solution), 1. + 1e-8)
        with self.assertRaises(ValueError):
            block.linear_min(np.ones(2), -1.)
        block.sigma2 = 0.
        with self.assertRaises(ValueError):
            block.width_rule(1.)

    def test_zero_objective_and_class_radius(self):
        block = BlockEllipsoid(np.eye(2), np.zeros(2), 1., 10, "test")
        np.testing.assert_array_equal(block.linear_min(np.zeros(2), 1., M=1.)[0], [0., 0.])
        np.testing.assert_array_equal(block.linear_min(np.ones(2), 1., M=0.)[0], [0., 0.])

    def test_infeasible_box_never_returns_clipped_center(self):
        block = BlockEllipsoid(np.eye(2), np.array([3., 3.]), 1., 10, "test")
        with self.assertRaises(RuntimeError):
            block.linear_min(np.ones(2), .01, M_inf=1.)

    def test_constrained_solver_against_feasible_grid(self):
        h = np.array([[2., .3], [.3, .7]])
        center, gradient = np.array([.2, -.1]), np.array([-1.2, .7])
        block = BlockEllipsoid(h, center, 1., 10, "grid")
        coords = np.linspace(-.6, .6, 401)
        xx, yy = np.meshgrid(coords, coords)
        candidates = np.column_stack([xx.ravel(), yy.ravel()])
        delta = candidates - center
        quad = np.einsum("ni,ij,nj->n", delta, h, delta)
        for xi in (.05, .3):
            solution, _ = block.linear_min(gradient, xi, M=.7, M_inf=.6)
            d = solution - center
            self.assertLessEqual(d @ h @ d, xi + 1e-8)
            self.assertLessEqual(np.linalg.norm(solution), .7 + 1e-8)
            self.assertLessEqual(np.abs(solution).max(), .6 + 1e-8)
            feasible = (quad <= xi) & (np.linalg.norm(candidates, axis=1) <= .7)
            sampled = (candidates[feasible] @ gradient).min()
            self.assertLessEqual(solution @ gradient, sampled + 1e-7)
            self.assertLess(sampled - solution @ gradient, .01)

    def test_general_width_identity(self):
        # The condition-number identity fails when the signal has multiple
        # eigenvalues; the correct expression uses its actual inverse quadratic.
        h = np.diag([.1, 1.1, 4.1])
        g = np.array([2., 3., 1.])
        signal = g[1:] @ np.linalg.solve(h[1:, 1:], g[1:])
        ratio2 = (g @ np.linalg.solve(h, g)) / signal
        self.assertAlmostEqual(ratio2, 1. + g[0] ** 2 / (.1 * signal))
        incorrect = 1. + (g[0] ** 2 / np.sum(g[1:] ** 2)) * np.linalg.cond(h)
        self.assertGreater(abs(incorrect - ratio2), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
