# Extension paper — "When Proximal Pessimism Cannot Contract"

The theoretical result from `Extension/`, written up as a standalone paper. This
is **not** a course deliverable: `Phase_1/` through `Phase_5/` are the graded
submissions and stay frozen. `tools/make_bundles.py` only discovers `Phase_N`
directories, so nothing here can reach a course bundle.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Currently 8 pages: ~5 body, references on page 6, appendices A–C on 7–8.
Verified: 0 overfull boxes, 0 unresolved references, nothing overflowing a column.

## The spine

| § | Claim | Basis |
|---|---|---|
| 3 | `W >= beta * beta_g` for **any** ridge and **any** width rule | two-line proof; the regularizer cancels |
| 4.1 | Incompleteness (`\|O_0\| < \|S\|`) makes the truth leak | population algebra, 50/50 cells, sharp (`1.5e-15` vs 57–77%) |
| 4.2 | Confounding makes the gradient leak | Prop. 2 + a map over 4 configs, graded 0 → 0.82 |
| 4.3 | The floor needs **both**, and only at stage 1 | Cor. 2 and Cor. 3 |
| 5.2 | The anchor paper's own schedule has `e > 0` | closed form + 30/30 admissible cells |
| 6 | Decision-level: opposite-signed slopes, curves cross | 20 seeds, plug-in on identical fits |

## What the paper says about its own weaknesses

Three things are stated in the paper rather than buried, because each was found
by adversarial review of our own results:

- **The dichotomy of Cor. 1 is not new.** It is Tikhonov source-condition theory
  specialised to a design with an exact null atom (§5.3).
- **Prop. 2 is not new in substance either.** It is derivable from the anchor
  paper's own Lemma C.1, and restates the folk fact that without unmeasured
  confounding a negative control is unnecessary. The contribution is the
  *conjunction* of the two switches, not either one (§5.3).
- **An earlier headline of ours was an identity.** The width-ratio relation
  (Eq. 8) holds to `8.45e-15` over 17 cells of two independent sweeps, so the
  alignment sweep that established it could never have failed. Appendix A gives
  the derivation, the degenerate knob it was measured on, and the non-degenerate
  replacement.

Appendix C lists eight claims from earlier drafts that did not survive, with what
killed each.

## Scope, stated up front

The binding hypothesis `beta * beta_g > 0` fails under the anchor paper's own
assumptions: completeness forces `beta = 0`, and nonzero population gradient
leakage is equivalent to `C*_pi = infinity` by that paper's Lemma C.1. **Nothing
here is a defect in that paper.** It is a robustness result about the boundary of
applicability — what happens when the assumption the theory conditions on only
approximately holds.

## Sources for every number

| paper element | produced by |
|---|---|
| Table 1 (`beta`, 50/50) | `poc/run_completeness_beta.py` |
| Table 2 (floor map) | `poc/run_gradient_leakage_map.py` |
| `e_paper > 0`, 30/30 | `poc/run_stepc_recheck.py` |
| Table 3 (20-seed decision) | `poc/run_regret_incomplete.py` |
| Eq. 8 identity, App. A | `poc/run_alignment_nondegenerate.py` |
| Cor. 3 (`t >= 2` self-healing) | `poc/critic8_t2span.py` |

`docs/claims.md` is the ledger and is authoritative where any document disagrees
with another.
