"""critic_a2c_confound_continuum.py -- can C3 be salvaged on the near-deterministic continuum?

At confound=1.0 the suppressed direction has population value exactly 0 given the
action (C3's example is degenerate). But as confound -> 1 the per-action
population eigenvalue lambda_2(P_a P_a^T) -> 0 continuously, so there should be a
regime where a direction is genuinely nonzero in the per-action population yet
sits below the sampling floor at feasible N. If so, C3 survives as a statement
about NEAR-deterministic confounding, with the population floor quantifiable in
advance from lambda_2.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from critic_a2b_confound import run_case   # reuses classifier + printing

for c in (0.9, 0.95, 0.99):
    run_case(2, 6, 4, c, flat=False)
