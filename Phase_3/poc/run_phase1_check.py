"""
run_phase1_check.py — Phase 1 exit-criterion validation (NO Phase-2 estimation here).

Checks, in order:
  [A] Toy POMDP oracle verification
      A1. true bridges b_R, b_D solve the latent moment equations (residual ~ 0)
      A2. Theorem-3.5 sequential bridge integration reproduces DP V(pi) to
          machine precision, for every candidate policy      <-- oracle "verified"
      A3. MC rollouts agree with DP (3-sigma)
      A4. naive uncorrected observation-MDP evaluator vs true V(pi): BIAS TABLE
  [B] Big benchmark environment (720 x 2 latent + terminals)
      B1. augmented-chain row-stochasticity (asserted at build)
      B2. on-demand true-bridge residual self-test (per-x 2x2 inversion)
      B3. MC vs DP for one candidate policy (3-sigma)
      B4. EXIT CRITERION: naive-vs-true bias table at kappa = 1.0 (confounded)
          and kappa = 0.0 (latent-blind behavior control)

Writes experiments/results_phase1.json.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src", "envs")))

import simulated_env as senv
import toy_pomdp as toy
import oracle_module as oracle

RESULTS = {}


# ------------------------------------------------------------------ candidate policies
def toy_candidates():
    always0 = np.zeros((toy.N_O, toy.N_A)); always0[:, 0] = 1.0
    always1 = np.zeros((toy.N_O, toy.N_A)); always1[:, 1] = 1.0
    obs_dep = np.zeros((toy.N_O, toy.N_A))          # a = 1 iff o = 2
    obs_dep[[0, 1], 0] = 1.0
    obs_dep[2, 1] = 1.0
    uniform = np.full((toy.N_O, toy.N_A), 0.5)
    return {"always_a0": always0, "always_a1": always1,
            "obs_dependent": obs_dep, "uniform": uniform}


def big_candidates():
    """Observation-indexed (N_OBS, 8) candidate policies. Action encoding:
    a = 4*c_A + 2*c_E + 1*c_V (three binary intervention bits)."""
    n = senv.N_OBS
    never = np.zeros((n, 8)); never[:, 0] = 1.0
    allon = np.zeros((n, 8)); allon[:, 7] = 1.0
    uniform = np.full((n, 8), 1.0 / 8)

    guideline = np.zeros((n, 8))
    for o in range(2 * senv.N_X):
        x = o // 2
        c0, c1, c2, _, _, _, _ = senv.decode_core(x)
        bit_A = 1 if (c0 != 1 or c1 == 2) else 0    # component-0 abnormal or component-1 high
        bit_E = 1 if c2 == 0 else 0                 # component-2 low
        bit_V = 1 if c1 == 0 else 0                 # component-1 low
        guideline[o, 4 * bit_A + 2 * bit_E + bit_V] = 1.0
    guideline[senv.OBS_NEG, 0] = 1.0
    guideline[senv.OBS_POS, 0] = 1.0
    return {"never_treat": never, "all_interventions": allon,
            "guideline": guideline, "uniform": uniform}


# ------------------------------------------------------------------ [A] toy checks
def run_toy(seed=0):
    out = {}
    params = toy.default_params(T=3)
    cands = toy_candidates()

    # A1: bridge residuals
    bR, bD, res = oracle.toy_true_bridges(params)
    out["A1_bridge_residuals"] = res
    assert res["residual_bR"] < 1e-12 and res["residual_bD"] < 1e-12, \
        "toy bridge moment equations not solved exactly"

    # A2: Theorem-3.5 chain identity vs DP, per candidate policy
    ident = {}
    for name, pi in cands.items():
        v_dp = oracle.dp_value_toy(params, pi)
        v_chain, _ = oracle.toy_chain_value(params, bR, bD, pi)
        ident[name] = dict(V_dp=v_dp, V_chain=v_chain, abs_gap=abs(v_dp - v_chain))
        assert abs(v_dp - v_chain) < 1e-10, f"identification check failed for {name}"
    out["A2_theorem35_identity"] = ident

    # A3: MC vs DP sanity
    name = "obs_dependent"
    v_dp = ident[name]["V_dp"]
    v_mc, se = oracle.mc_value_toy(params, cands[name], N=200_000, seed=seed)
    out["A3_mc_vs_dp"] = dict(policy=name, V_dp=v_dp, V_mc=v_mc, stderr=se,
                              z=abs(v_mc - v_dp) / se)
    assert abs(v_mc - v_dp) < 4 * se, "toy MC/DP mismatch"

    # A4: naive baseline bias (confounded behavior log)
    rng = np.random.default_rng(seed + 1)
    data = toy.sample_trajectories(params, N=200_000, rng=rng)      # behavior log
    P_hat, r_hat, p1_hat, _ = oracle.fit_naive_mdp(
        data["O"], data["A"], data["R"], toy.N_O, toy.N_A)
    bias_tab = {}
    for name, pi in cands.items():
        v_true = oracle.dp_value_toy(params, pi)
        v_naive = oracle.naive_value(P_hat, r_hat, p1_hat, pi, params["T"])
        bias_tab[name] = dict(V_true=round(v_true, 6), V_naive=round(v_naive, 6),
                              bias=round(v_naive - v_true, 6))
    out["A4_naive_bias"] = bias_tab
    return out


# ------------------------------------------------------------------ [B] big-env checks
def run_big(N=20_000, seed=0):
    out = {}
    cands = big_candidates()

    # B2 + B3 on the kappa=1 environment
    env = senv.make_env(kappa=1.0)
    rng = np.random.default_rng(seed + 10)
    worst = oracle.big_bridge_selftest(env, rng, n_queries=64)
    out["B2_bridge_selftest_worst_residual"] = worst
    assert worst < 1e-12, "big-env on-demand bridge inversion failed"

    name = "guideline"
    v_dp = oracle.dp_value_obs_policy(env["aug"], cands[name], env["T"],
                                      env["q0"], env["q1"])
    v_mc, se = oracle.mc_value_big(env, cands[name], N=100_000, seed=seed + 20)
    out["B3_mc_vs_dp"] = dict(policy=name, V_dp=v_dp, V_mc=v_mc, stderr=se,
                              z=abs(v_mc - v_dp) / se)
    assert abs(v_mc - v_dp) < 4 * se, "big-env MC/DP mismatch"

    # B4: EXIT CRITERION — naive bias at kappa in {1.0, 0.0}
    exit_tab = {}
    for kappa in (1.0, 0.0):
        env_k = senv.make_env(kappa=kappa)
        data = senv.generate_dataset(env_k, N=N, seed=seed + int(10 * kappa))
        P_hat, r_hat, p1_hat, visit = oracle.fit_naive_mdp(
            data["O"], data["A"], data["R"], senv.N_OBS, senv.N_A)
        tab = {}
        for name, pi in cands.items():
            v_true = oracle.dp_value_obs_policy(env_k["aug"], pi, env_k["T"],
                                                env_k["q0"], env_k["q1"])
            v_naive = oracle.naive_value(P_hat, r_hat, p1_hat, pi, env_k["T"])
            tab[name] = dict(V_true=round(v_true, 6), V_naive=round(v_naive, 6),
                             bias=round(v_naive - v_true, 6))
        exit_tab[f"kappa={kappa}"] = dict(
            N=N, behavior_mean_return=float(data["R"].sum(axis=1).mean()),
            visited_sa_pairs=int(visit.sum()), table=tab)
    out["B4_naive_bias_exit_criterion"] = exit_tab

    # B5: POPULATION (infinite-data) naive — pure STRUCTURAL bias, no sampling noise
    pop_tab = {}
    for kappa in (1.0, 0.0):
        env_k = senv.make_env(kappa=kappa)
        P_pop, r_pop, p1_pop, _ = oracle.population_naive_mdp(env_k)
        tab = {}
        for name, pi in cands.items():
            v_true = oracle.dp_value_obs_policy(env_k["aug"], pi, env_k["T"],
                                                env_k["q0"], env_k["q1"])
            v_naive = oracle.naive_value(P_pop, r_pop, p1_pop, pi, env_k["T"])
            tab[name] = dict(V_true=round(v_true, 6), V_naive=round(v_naive, 6),
                             bias=round(v_naive - v_true, 6))
        pop_tab[f"kappa={kappa}"] = tab
    out["B5_population_naive_structural_bias"] = pop_tab
    return out


def main():
    RESULTS["toy"] = run_toy()
    print("[A] toy checks passed:",
          "bridge residuals", RESULTS["toy"]["A1_bridge_residuals"],
          "| worst Thm3.5 gap",
          max(v["abs_gap"] for v in RESULTS["toy"]["A2_theorem35_identity"].values()))
    RESULTS["big"] = run_big()
    print("[B] big-env checks passed: bridge self-test residual",
          RESULTS["big"]["B2_bridge_selftest_worst_residual"])

    print("\n=== PHASE 1 EXIT CRITERION: naive uncorrected evaluator vs truth ===")
    for envname, blk in (("TOY (confounded behavior)", RESULTS["toy"]["A4_naive_bias"]),):
        print(f"\n-- {envname}")
        print(f"{'policy':<20}{'V_true':>10}{'V_naive':>10}{'bias':>10}")
        for k, v in blk.items():
            print(f"{k:<20}{v['V_true']:>10.4f}{v['V_naive']:>10.4f}{v['bias']:>10.4f}")
    for kname, blk in RESULTS["big"]["B4_naive_bias_exit_criterion"].items():
        print(f"\n-- BIG ENV {kname}  (N={blk['N']}, behavior return "
              f"{blk['behavior_mean_return']:.4f})")
        print(f"{'policy':<20}{'V_true':>10}{'V_naive':>10}{'bias':>10}")
        for k, v in blk["table"].items():
            print(f"{k:<20}{v['V_true']:>10.4f}{v['V_naive']:>10.4f}{v['bias']:>10.4f}")
    for kname, tab in RESULTS["big"]["B5_population_naive_structural_bias"].items():
        print(f"\n-- BIG ENV POPULATION NAIVE (infinite data, structural bias) {kname}")
        print(f"{'policy':<20}{'V_true':>10}{'V_naive':>10}{'bias':>10}")
        for k, v in tab.items():
            print(f"{k:<20}{v['V_true']:>10.4f}{v['V_naive']:>10.4f}{v['bias']:>10.4f}")

    res_path = os.path.normpath(os.path.join(HERE, "..", "experiments",
                                             "results_phase1.json"))
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nresults written to", res_path)


if __name__ == "__main__":
    main()
