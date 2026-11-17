# Reproduction companion

This archive is separate from the minimal arXiv source upload.
The current manuscript is Extension/paper/workshop_draft.tex.
Historical helper-module docstrings do not override its scoped claims.

From the extracted archive root:

    python -m pip install -r Extension/requirements.txt
    python Extension/poc/run_submission_checks.py
    python Extension/poc/verify_workshop_numbers.py
    python Extension/poc/verify_strengthening.py
    python Extension/poc/fetch_ist.py
    python Extension/poc/verify_ist_experiment.py
    python Extension/poc/run_learned_bc.py

New sensitivity runs: python Extension/poc/run_external_validity.py (100 fits),
python Extension/poc/run_reward_reversal.py (25 fits), and
python Extension/poc/run_optimizer_budget.py (20 fits, 30 restarts).
VERIFY_ONLY=1 runs the two sensitivity drivers' checks without rerunning fits.

Medical boundary case: python Extension/poc/run_ist_policy_selection.py (40 T=1 fits).
The public corrected IST v2 patient CSV is downloaded separately, not bundled.
IST data licence is ODC-By 1.0, separate from the code's MIT licence.
It is a semi-synthetic real-outcome illustration, not clinical validation.

Python 3.10 or later is required. For the recorded numerical versions use
pip install -r Extension/requirements.txt -c Extension/constraints-reference.txt
MIT code license: LICENSE. The paper's arXiv license is a separate choice.
Legacy oracle generator: python Extension/poc/run_bc_candidate_retest.py
It overwrites its JSON in this extraction; run it in a separate copy.
It preserves legacy normal summaries and does not save per-seed selections;
the fresh release verification reruns the logged-data experiment only.

The last command regenerates results_learned_bc.json (80 fits, one fixed environment).
Recorded runtime: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 to match the recorded run.
The science checks and numeric checks are consistency checks, not independent replication.
