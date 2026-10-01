"""Shared ILO labeller (Phase 3, D025): a trained model that links each dialogue turn to the
proposition it expresses, if any, and labels how it was said (Asserting, Pure Questioning, ...).

    candidates.py   3.1  which (turn, proposition) pairs the model sees
    data.py         3.2  model inputs and labels, cached under cache/ilo/
    train.py        3.3  ModernBERT-base fine-tuning (loop in common/finetune.py)
    decode.py            probabilities -> one-to-one links with types; dev scorer
    tune.py         3.4  temperature + thresholds on dev, cross-fitted by episode
    evaluate.py     3.5  dev report against baselines -> results/ilo/dev_report.json

Results and decisions: decisionlog.md D028-D036.
"""
