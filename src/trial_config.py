"""Canonical defaults for fast screening trials.

Promising candidates are confirmed separately with five folds before promotion
to a final experiment.
"""

TRIAL_FOLDS = 2
TRIAL_SEED = 2026
TRIAL_ITERATIONS = 200

TRIAL_LIGHTGBM_LEARNING_RATE = 0.10

TRIAL_CATBOOST_LEARNING_RATE = 0.08
TRIAL_CATBOOST_DEPTH = 8
TRIAL_CATBOOST_L2_LEAF_REG = 5.0

TRIAL_EXACT_WEIGHT = 0.60
