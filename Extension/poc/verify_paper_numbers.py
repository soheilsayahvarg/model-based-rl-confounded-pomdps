"""verify_paper_numbers.py -- trace every number in paper/main.tex back to a JSON.

The paper quotes figures produced across many runs and many months. A number that
was correct when written can drift out of agreement with its source when the
source is re-run, and prose numbers have no compiler to catch that. This script
re-derives each quoted figure from experiments/*.json and reports agreement.

It checks values, not claims. A PASS means the paper's number matches the stored
result; it says nothing about whether the experiment was well designed.

Exit code 1 if any check fails, so this can gate a commit.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", "experiments"))

FAILS = []
CHECKS = 0


def load(name):
    p = os.path.join(EXP, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def check(label, paper, actual, tol=None, rel=0.02):
    """Compare a paper figure against a recomputed one."""
    global CHECKS
    CHECKS += 1
    if actual is None:
        FAILS.append((label, paper, "SOURCE MISSING"))
        print(f"  {'MISSING':>8}  {label:<52} paper={paper}")
        return
    if tol is None:
        tol = abs(paper) * rel if paper else 1e-9
    ok = abs(paper - actual) <= tol
    if not ok:
        FAILS.append((label, paper, actual))
    print(f"  {'ok' if ok else 'MISMATCH':>8}  {label:<52} "
          f"paper={paper:<12.6g} json={actual:<12.6g}")


def main():
    print("=" * 96)
    print("Tracing paper/main.tex figures back to experiments/*.json")
    print("=" * 96)

    # --- Table 1 and the sharpness claim (completeness) --------------------
    print("\n[Table 1] population beta, completeness switch")
    d = load("results_completeness_beta.json")
    if d:
        rows = d["rows"]
        vals = [(bool(r["complete"]), float(r["rel"])) for r in rows]
        complete = [v for c, v in vals if c]
        incomplete = [v for c, v in vals if not c]
        if complete:
            check("max beta in COMPLETE cells <= 1.5e-15",
                  1.5e-15, max(complete), tol=1.5e-15)
        if incomplete:
            check("min beta in INCOMPLETE cells (paper: 0.569)",
                  0.569, min(incomplete), tol=0.02)
            check("max beta in INCOMPLETE cells (paper: 0.773)",
                  0.773, max(incomplete), tol=0.02)
        print(f"           ({len(vals)} cells found in JSON; "
              f"paper claims 50)")
    else:
        print("  results_completeness_beta.json not found")

    # --- Table 2 (floor map) ----------------------------------------------
    print("\n[Table 2] gradient leakage and the floor")
    d = load("results_gradient_leakage_map.json")
    if d:
        rows = d["rows"]
        def find(cfg, cf):
            for r in rows:
                if (r["n_s"], r["n_o"], r["n_o0"]) == cfg and \
                        abs(float(r["confound"]) - cf) < 1e-9:
                    return r
            return None
        for cfg, cf, want_floor in [((4, 6, 2), 0.9, 0.428),
                                    ((4, 6, 2), 1.0, 0.823),
                                    ((4, 8, 3), 0.9, 0.162)]:
            r = find(cfg, cf)
            if r:
                fl = r.get("floor")
                if fl is None:
                    b = r.get("beta"); g = r.get("beta_g")
                    fl = float(b) * float(g) if b is not None and g is not None else None
                check(f"floor {cfg} cf={cf}", want_floor,
                      float(fl) if fl is not None else None, tol=0.01)
            else:
                check(f"floor {cfg} cf={cf}", want_floor, None)
    else:
        print("  results_gradient_leakage_map.json not found")

    # --- Section 5.3 identity ---------------------------------------------
    print("\n[Eq. 8] the width-ratio identity")
    d = load("results_alignment_knob.json")
    if d and "closed_form" in d:
        check("worst relative error over 17 cells (paper: 8.45e-15)",
              8.45e-15, float(d["closed_form"]["worst_rel_err"]), tol=5e-15)
        check("number of cells (paper: 17)", 17,
              float(len(d["closed_form"]["cells"])), tol=0.5)
        k1 = d.get("knob1", {})
        if "loglog_slope" in k1:
            check("knob-1 log-log slope (appendix: 0.954)",
                  0.954, float(k1["loglog_slope"]), tol=0.01)
        if "total_span_rotation_deg" in k1:
            check("knob-1 span rotation deg (appendix: 36.8)",
                  36.8, float(k1["total_span_rotation_deg"]), tol=0.15)
        k2 = d.get("knob2", {})
        if "beta_g_decades" in k2:
            check("knob-2 decades of beta_g (appendix: 3.09)",
                  3.09, float(k2["beta_g_decades"]), tol=0.02)
    else:
        print("  results_alignment_knob.json missing or lacks closed_form")

    # --- Table 3 (decision, T=3) ------------------------------------------
    print("\n[Table 3] decision experiment, T=3, 20 seeds")
    d = load("results_regret_incomplete.json")
    if d:
        rows = d if isinstance(d, list) else d.get("rows", d.get("cells", []))
        def cell(tag_sub, N):
            for r in rows:
                if tag_sub in str(r.get("schedule", "")) and \
                        int(r.get("N", -1)) == N:
                    return r
            return None
        for tag, N, want_p, want_pl in [("kappa=1.5", 128000, 0.1804, 0.0000),
                                        ("kappa=1.5", 2000, 0.0952, 0.0894),
                                        ("paper", 2000, 0.0694, 0.1641),
                                        ("paper", 32000, 0.1225, 0.0644)]:
            r = cell(tag, N)
            check(f"{tag} N={N} pessimistic", want_p,
                  float(r["regret"]) if r else None, tol=0.002)
            check(f"{tag} N={N} plug-in", want_pl,
                  float(r["plugin_regret"]) if r else None, tol=0.002)
    else:
        print("  results_regret_incomplete.json not found")

    # --- Table 4 (horizon dilution) ---------------------------------------
    print("\n[Table 4] horizon dilution")
    d = load("results_horizon_dilution.json")
    if d:
        rows = d.get("dilution", [])
        def rat(cfg, cf, T):
            for r in rows:
                if tuple(r["config"]) == cfg and abs(r["confound"] - cf) < 1e-9 \
                        and int(r["T"]) == T:
                    return float(r["ratio"])
            return None
        for cfg, cf, T, want in [((4, 6, 2), 0.9, 1, 1.627),
                                 ((4, 6, 2), 0.9, 3, 0.379),
                                 ((4, 6, 2), 0.9, 6, 0.177),
                                 ((4, 8, 3), 0.6, 6, 0.074)]:
            check(f"ratio {cfg} cf={cf} T={T}", want, rat(cfg, cf, T),
                  tol=0.002)
        ps = d.get("per_stage", [])
        late = [r for r in ps if int(r["t"]) >= 2]
        if late:
            check("max beta_t over t>=2 (paper: 2.7e-15)",
                  2.7e-15, max(float(r["beta"]) for r in late), tol=3e-15)
            check("max per-stage floor over t>=2 (paper: 4.3e-30)",
                  4.3e-30, max(float(r["floor"]) for r in late), tol=5e-30)
    else:
        print("  results_horizon_dilution.json not found")

    # --- report ------------------------------------------------------------
    print("\n" + "=" * 96)
    if FAILS:
        print(f"{len(FAILS)} of {CHECKS} checks FAILED:")
        for label, paper, actual in FAILS:
            print(f"   {label}: paper={paper}  json={actual}")
        sys.exit(1)
    print(f"all {CHECKS} checks passed -- every traced figure matches its JSON")


if __name__ == "__main__":
    main()
