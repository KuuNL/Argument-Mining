"""Phase 10 (D099): can Jev do Task B end to end, linking each dialogue turn to its proposition and typing it,
better than the trained ILO labeller? Dev only; the same protocol as ilo_llm/ (Claude, D090) where it applies.

    prompts/        the guide placed in Jev's state
    questions.py    state + two `choice` questions per turn (which proposition, which type), both option orders
    parse.py        answers -> candidate rows + probabilities in ilo.decode's format
    runner.py       named runs, cached and resumable (after a4_jev/runner.py); every request's cost is recorded
    check.py        gates with no calls, and the cost projection -> results/ilo_jev/check.json
    pilot.py        k = 0 vs k = 4 worked examples on the 20 pilot maps -> results/ilo_jev/pilot.json
    tune.py         the labeller's decision layer (ilo_llm.tune), cross-fitted by episode
    evaluate.py     Jev vs the labeller and Claude on dev -> results/ilo_jev/dev/dev_report.json

Runs in ~/.venvs/a3 under `bash -lc` (httpx; AI_GATEWAY_API_KEY from WSL ~/.profile; the bootstrap imports A3's code).
Answers: cache/ilo_jev/answers/ (git-ignored); results: results/ilo_jev/<run>/.
"""
