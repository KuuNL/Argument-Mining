"""Shared ILO labeller (Phase 3, D025): a trained model that links each dialogue turn to the
proposition it expresses, if any, and labels how it was said (Asserting, Pure Questioning, ...).

    candidates.py   3.1  which (turn, proposition) pairs the model sees
    data.py         3.2  model inputs and labels, cached under cache/ilo/
"""
