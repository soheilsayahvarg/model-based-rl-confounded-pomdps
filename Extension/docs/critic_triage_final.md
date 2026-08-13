# Critic Triage — Final Checkpoint, Round 2 (pre-upload audit)

Audit of commit `fe93ea9` (Phase_4/paper) + the two unchecked categories.
Status: COMPLETE. Edits for every must-fix item are APPLIED IN THE WORKING TREE
(uncommitted) and the rebuild verifies references at page 5, col 1, y=68 —
exactly the pre-edit budget. Review with `git diff`, then commit or discard.

## Checklist

- [x] T1. Read `git show fe93ea9` in full
- [x] T2. Recompute the new appendix table vs results_pessimism_regenerated.json
- [x] T3. Over-correction check (Eq. 17 class carriage; the ~7x figure)
- [x] T4. Under-correction check (the -1779 headline and its leftovers)
- [x] T5. Internal consistency vs appendix.tex:328
- [x] T6. Contraction-claim sweep
- [x] T7. Scope-misattribution sweep
- [x] T8. Rebuild + verify page budget (y=68 reproduced pre-edit and post-edit)
- [x] T9. Priority 3 judgment call
- [x] T10. Triage table delivered in chat

## Findings record

**T2 — table verified.** All 20 (regret, V_low) cells of the new appendix table
match `results_pessimism_regenerated.json` after rounding; M_inf 0.872/0.614,
M_true 1.803/1.321, corner-norm 5.2 = 0.872*6, V_true 2.0177 all match. PASS.

**T3a — attribution is CORRECT, not over-generous.** Anchor Eq. 15/16 define
conf(xi) = {b in B_{R,t} : ...} and Algorithm 1 minimises over conf_R x conf_D;
Assumption D.16(d) norm-constrains B_{R,t} (RKHS unit ball, wlog) and 4.1(f)
gives its sup-norm envelope M_R. The paper as printed does carry the class;
minimising over the region alone was our change. "Our omission" stands.

**T3b — but the commit introduced a NEW over-correction:** the appendix
paragraph "the class is a box, not a ball / the paper's literal class is the
box" is wrong. The class carries BOTH bounds: RKHS-norm (D.16(d)) — which in
the tabular delta instantiation is the coefficient l2 norm — AND the 4.1(f)
sup-norm envelope. Ball and box are each RELAXATIONS; the paper's class is
their intersection. FIXED: paragraph reworded ("The class carries two bounds,
not one"), table row relabeled "Box (4.1(f) only)".

**T3c — the "~7x" was grid- and class-specific.** From the JSONs: ball shrink
factors are 4.0x (c=0.3), 6.9x (c=1.0), sign-flip regime at c=0.1, and ~210x at
c=10 (critic_p_a345.json: vanilla -1907.6 -> oracle-M ball -9.058). "About 7x"
attached to a sentence quoting magnitudes up to -1779 (c=10) was indefensible
at both ends. FIXED: now "4–7x (~200x at c=10) with selections unchanged
(app ref)". Selections-unchanged is verified for the ball on the common grid
(identical modal picks in every row); it was FALSE for the box (0.427 vs 0.513
at c >= 0.3) — another reason the box mislabel had to go.

**T4 — under-correction: three leftovers contradicted fix #1.** (i)
Conclusion (main.tex:432): "showed what the theory did not: ITS pessimism step
does not run as written" — the exact misattribution the commit set out to
remove, surviving in the last paragraph a skimmer reads. FIXED: "our class-free
pessimism run diverges". (ii) §4.3 subsection TITLE: "run literally" — FIXED:
"run class-free". (iii) §3.3 Failure mode: "Run as written, ... walks
arbitrarily far" — with any bounded class the walk is capped, so "as written"
was wrong; FIXED: "Run class-free, ...". The -1779 itself stays: it is now
correctly labeled as ours in the abstract, and the class-constrained companion
figures are one cref away. Keeping it is defensible.

**T5 — no contradiction.** appendix.tex:321-332 (pess. regret 0.211, plug-in
0.000) is the finite-action linear-Gaussian environment; the new section is the
toy. Plug-in rows agree with "never beaten" in both. PASS.

**T6 — contraction sweep: CLEAN.** No claim, implication, or reliance on the
region tightening with N anywhere in main.tex/appendix.tex. "Sequential
contraction" (main.tex:164) is the Theorem 3.5 bridge chain — different sense.
The Limitations section honestly labels the pessimism guarantee empirical.

**T7 — misattribution sweep: hits were (i)-(iii) above, all fixed.** Left as-is
deliberately: main.tex:112 "a failure mode in the paper's coupled multi-block
pessimism" and main.tex:195 "the difficulty is the paper's coupled
minimization" — defensible because the divergence SURVIVES the faithful class
(round 4 D2: ball at c=10 gives -9.06 vs truth 1.889; box -52 to -92), so a
failure mode genuinely attaches to the method, at smaller magnitudes. The
class-free scoping now happens one paragraph later in both places.

**T8 — page budget.** Pre-edit rebuild reproduced references at p5 col1 y=68
byte-identically (388,694 bytes). Naive versions of the four fixes pushed the
whole Conclusion block onto p5 (y=176). Final tightened versions + two in-place
trims in §4.3 restore y=68 EXACTLY. No unresolved references or citations.

**Cosmetic, not fixed (note as limitation):** the new appendix section opens
"Three defects in our own implementation and two in our reporting" but then
enumerates three paragraphs that do not map 1:1 onto that count (the solver,
per-block-M and admissible-M fixes are folded into the regenerated numbers and
detailed in pessimism_regenerated.md / the code, not in the appendix text).
Harmless; one sentence could point at the code if desired. Also
noncontraction.md's duplicated "## 7. Files" (Extension doc, not shipping).

**T9 — Priority 3.** Agreed with exclusion from the MAIN BODY (at cap, and the
draft makes no contraction claim anywhere per T6, so nothing needs the
defense). The counter-argument was considered and is weaker than it looks: the
one referee question e_paper > 0 preempts ("would more data fix §4.3?") is
already answered implicitly by the N-flat width table cited from step (b), and
the algebra has been through exactly one adversarial round, below this
project's evidentiary bar for main-body claims. Optional zero-cost middle path
if wanted: ONE sentence at the end of app:pess-correction stating the paper's
own (lambda2, xi) schedules place the region on the non-contracting side for
every admissible (alpha, c2), so no sample size rescues §4.3's magnitudes —
appendix is unlimited, nothing gets cut. Default: leave it out.

## Working-tree state at handoff

Modified, NOT committed: Phase_4/paper/main.tex (4 edits + 2 in-paragraph
trims), Phase_4/paper/appendix.tex (paragraph reword + row relabel),
main.pdf/aux files (rebuilt, y=68 verified). `git diff` to review;
`git checkout -- Phase_4/paper` to discard everything.
