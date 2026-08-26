# Novelty Check: What of This Is Already Known

Run before rebuilding the paper around the span theorem, because this line of
work has already rediscovered published results twice and both times we found
out only after building on them.

Date of check: 2026-08-27. Method: literature search plus full-text extraction of
the two closest papers.

---

## 1. The completeness switch is textbook

`sec:switch-beta` says the truth leaks when `|O_0| < |S|`. Ying, Miao, Shi and
Tchetgen Tchetgen, *Proximal Causal Inference for Complex Longitudinal Studies*
(JRSS-B 2023, arXiv:2109.07030), state it as their equation (6):

> In this case, completeness requires that `min(d_z, d_w) >= d_u`, which states
> that `Z(1)` and `W(1)` must each have at least as many categories as `U(1)`.

and attribute it to Miao et al. 2018, Shi et al. 2020, Tchetgen Tchetgen et al.
2020, Cui et al. 2020. Note also their parenthetical that cardinality is "defined
as the product of the cardinalities of each component in the vector", which is
precisely the cell-counting behaviour we measured as the empirical null
dimension.

**Status: known, and known for years.** The paper already half-concedes this in
`sec:not-new`; it should concede it fully and cite equation (6).

## 2. Sequential completeness is assumed, never characterized

The same paper's Assumptions 4 and 5 state completeness separately at each time
point and never ask when it holds at later times. Searching its full text for a
sufficient condition, a rank condition on the latent transition, or any statement
about history making completeness easier or harder returns nothing. The word
"rank" appears once in the whole paper, in a historical remark about rank
preservation.

**Status: a genuine gap in that literature.**

## 3. But the POMDP OPE literature assumes exactly our rank condition

Zhang and Jiang, *On the Curses of Future and History in Future-dependent Value
Functions for OPE* (arXiv:2402.14703), Assumption 2 (Invertibility):

> `rank(M_{H,h}) = rank(M_{F,h}) = S` for all `h`

That is our span condition, at every stage, taken as a standing assumption. They
also write:

> Uehara et al. [2022a] pointed out that the value is finite if `M_{H,h}` has
> full-row rank (`S`), but a quantitative understanding is missing.

**Status: the condition is standard. Its role is known. Its quantitative
consequences were explicitly open as of that sentence, though that paper then
supplies its own answer in the coverage-assumption direction.**

## 4. And the HMM literature already answers when it holds

From the finite-state nonparametric HMM identifiability line (Allman, Matias and
Rhodes 2009 via Kruskal's theorem; Gassiat, Cleynen and Robin 2016):

> Finite state space non-parametric HMMs are identifiable as soon as the
> transition matrix of the latent Markov chain has full rank and the emission
> probability distributions are linearly independent.

> The transition matrix and the emission distributions of a stationary HMM are
> identifiable up to state labelling from the law of three consecutive
> observations, provided that the transition matrix has full rank and that the
> emission distributions are linearly independent.

Our two conditions are `rank(E) = |S|` (emissions linearly independent) and
"the reachable transition rows span `R^|S|`" (transition full rank). **They are
the same two conditions**, and "from three consecutive observations" is the same
phenomenon as history restoring the span.

**Status: the headline result of the proposed reframe is classical.** This is the
third rediscovery in this project, after the Tikhonov dichotomy and Lemma C.1.

## 5. What actually survives

Thin, and thinner than the reframe pitch claimed.

| candidate | verdict |
|---|---|
| "History substitutes for a negative control" | **known**, classical HMM identifiability |
| `|O_0| >= |S|` completeness switch | **known**, Ying et al. eq. (6) |
| `rank(M_{H,h}) = S` at every stage | **known and assumed**, Zhang and Jiang Assumption 2 |
| The exact span formula `E^T diag(pi_b) W_t`, handling **partial** rank rather than a binary full-rank test, and carrying the behaviour policy | **plausibly new**. HMM identifiability has no actions and no confounded logger. The `|S|/|A|` saturation knife-edge is specific to a confounded POMDP |
| Non-monotonicity: the span falls from `3` to `2` between `t=1` and `t=2` | **no hit found**, but absence of a hit is weak evidence, and once the rank condition is stated it is close to a one-line corollary |
| What a **violated** rank assumption costs a pessimistic proximal planner, quantitatively and at the decision level | **not found in either literature** |
| A remedy (`projall`) that works without knowing which stage leaks | **not found** |

## 6. Consequence for the paper

The reframe proposed before this check, "when does history substitute for a
negative control", is **not viable as a headline**. Its answer is classical.

What survives is narrower and should be stated as such:

> Everyone assumes `rank(M_{H,h}) = S`. What does it cost when that assumption
> fails, and what can be done about it?

Under that framing:

- The span formula is a **tool** for deciding whether the assumption holds from
  primitives, not the contribution.
- The contributions are the quantitative cost (the floor, its stage structure,
  its horizon behaviour, and the fact that it **reverses** without the
  assumption) and the remedy.
- Zhang and Jiang's own sentence, that a quantitative understanding was missing,
  is the opening, and their paper answers a different part of it.

That is defensible and it is where every measurement in this project actually
lives. It is a smaller claim than a theorem about identification.

## 7. Revised chances

The estimate given before this check assumed the span theorem was new. It was
not, and the estimate has to come down.

| venue | before the check | after |
|---|---|---|
| workshop | 80% | 75-80% |
| CLeaR | 45-55% | 30-40% |
| UAI | 35-45% | 25-35% |
| AISTATS | 30-35% | 20-25% |
| ICML / NeurIPS | 20-25% | 12-18% |

## 8. What this check was worth

It cost one session and it caught the third rediscovery **before** the paper was
rebuilt around it, rather than after. The two previous ones were caught after we
had already published claims that depended on them.

The standing lesson, now three for three: **derive the closed form and search the
literature before building on a result, not after it looks clean.**

## Sources

- Ying, Miao, Shi, Tchetgen Tchetgen, *Proximal Causal Inference for Complex
  Longitudinal Studies*, JRSS-B 85(3) 2023, arXiv:2109.07030
- Zhang and Jiang, *On the Curses of Future and History in Future-dependent Value
  Functions for OPE*, arXiv:2402.14703
- Allman, Matias, Rhodes 2009; Gassiat, Cleynen, Robin 2016, on finite-state
  nonparametric HMM identifiability via Kruskal rank
- Uehara et al. 2022a, future-dependent value functions
