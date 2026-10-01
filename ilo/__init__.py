"""Shared ILO labeller (Phase 3, D025): a trained model that links each dialogue turn to the
proposition it expresses, if any, and labels how it was said (Asserting, Pure Questioning, ...).

    candidates.py   3.1  which (turn, proposition) pairs the model sees
    data.py         3.2  model inputs and labels, cached under cache/ilo/
    train.py        3.3  ModernBERT-base fine-tuning (loop in common/finetune.py)
    decode.py            probabilities -> one-to-one links with types; dev scorer
    tune.py         3.4  temperature + thresholds on dev, cross-fitted by episode
    evaluate.py     3.5  dev report against baselines -> results/ilo/<run>/dev_report.json

Each training run (v1, v2, ...) has a fixed recipe in train.RUNS and its own folders:
checkpoints/ilo/<run>/ (git-ignored) and results/ilo/<run>/. results/ilo/selected.json names the
run the rest of the project uses. Findings: reports/03-ilo-labeller.md; decisions D025, D028-D039.
"""
