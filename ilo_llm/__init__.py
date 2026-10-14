"""Phase 9 (D090): can Claude Sonnet 5.5 do Task B end to end, linking each dialogue turn to its proposition
and typing it, better than the trained ILO labeller? Dev only; the brief is plans/claude-ilo-dev-test/README.md.

    prompts/        system prompts (v0; at most one revision, on the pilot only)
    schema.py       per-map answer schema for `claude -p --json-schema`, and a strict local check
    prompt.py       rendering, gold answers, worked examples (retrieved with a2_llm.prompt.Pool)
    parse.py        answers -> votes -> candidate rows + probabilities in ilo.decode's format
    runner.py       named runs, cached and resumable (a copy of a2_llm/runner.py)
    tune.py         the labeller's decision layer on Claude's votes, cross-fitted by episode
    check.py        gates with no model calls -> results/ilo_llm/check.json

Calls go through a2_llm/claude_cli.py (Pro login). Answers: cache/ilo_llm/answers/ (git-ignored);
results: results/ilo_llm/<run>/.
"""
