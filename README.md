# Dialogical Argument Mining, Revisited

**An exploration of what modern language models change in dialogical argument mining: starting from where the 2024 shared-task systems fell short, and following each shortfall with a different kind of model.**

Kunal Kamra · October 2026 · [Short report](reports/06-short-report.md) · [Final report](reports/05-final-report.md) · [Decision log](decisionlog.md)

## The exploration

**The task.** BBC *Question Time* debates arrive already segmented into turns and propositions ([DialAM-2024](https://aclanthology.org/2024.argmining-1.8/), QT30 corpus). A system must predict **(A)** the argument relations between propositions (support, attack or rephrase; the corpus calls them Inference, Conflict and Rephrase) and **(B)** the speech act (illocution) linking each turn to its proposition.

**The starting point.** Every strong 2024 system classified two sentences at a time. The shared task's overview paper reports that argument relations were the bottleneck, that attacks were missed and argument direction misread, and that the LLM entries did not beat the encoders. This project turns those observations, and our hypotheses about their causes (marked as ours in the table), into a map. Rather than test one hypothesis, it follows each shortfall with a model built to address it, and asks one question throughout: *what do models that appeared since 2024 change, and where do they still fail?*

| Where 2024 fell short | Idea explored here | Approach |
|---|---|---|
| Illocutions classified pair by pair | A trained labeller that links each turn to one proposition, with thresholds tuned to the metric | ILO labeller (shared) |
| Two sentences judged in isolation | Give the 2024 recipe the whole dialogue as context | A1 · fine-tuned encoder |
| Direction misread (and, we suspected, no consistency across a map) | Let a frontier LLM read the whole map and decide every relation together | A2 · Claude Sonnet 5.5 |
| LLM entries lost to encoders | Teach the annotators' conventions to a small open LLM by fine-tuning | A3 · Qwen3-4B (QLoRA) |
| (We suspected) LLM output too unreliable to threshold | A decision model that answers typed questions with probabilities, with no training | A4 · Jev (TypeSafe AI) |

## What the exploration found

- **Illocutions were fixed by engineering.** The trained labeller scores 91.6–92.6 on test against the best 2024 entry's 81.2, and this carries every system past the 2024 winner on Global F1 (**72.0–73.7** against **67.05**, general). The best, Claude, scores **73.67 / 58.27** (general / focused) against **67.05 / 48.25**.
- **Relations were not fixed by any model.** On test the four systems score 51.4–55.8 against the 2024 winner's 55.3, well inside the noise of an 11-map test set. On 225 dev maps (cross-fitted thresholds, paired bootstrap) they rank Claude > Jev ≈ fine-tuned encoder > fine-tuned Qwen. Claude leads the encoder by +4.1 F1 (95% CI 2.2–6.1), and Jev matches the encoder with no training, at $0.003 per map.
- **The reason is the task, not the models.** The hard part is deciding *whether* two propositions are connected at all, often through an unmarked concession or elaboration, under labels that annotators themselves agree on only moderately. Four very different models stall at the same level. Our safeguards, the pipeline and the later test debates were checked and ruled out.

![Global F1 gain over the 2024 winner, split by task](reports/figures/short-gain-vs-2024.png)

*Global F1 above the 2024 winner on the 11 test maps, split into the part from illocutions (blue) and from relations (orange).*

## How it works

```mermaid
flowchart LR
  M[("Map<br/>turns + propositions")] --> L["ILO labeller<br/>(ModernBERT, trained)<br/>links and types turns"]
  L --> S["Relation scorer<br/>A1 · A2 · A3 · A4"]
  S --> D["Decision layer<br/>per-type thresholds,<br/>tuned on dev"]
  D --> O[("Valid argument map<br/>(AIF JSON)")]
  L --> O
  O --> SC[["Official scorer<br/>(ported, verified)"]]
```

Every approach plugs into the same pipeline, so the four are directly comparable. Only the relation scorer changes. The illocution labeller handles Task B for all of them (A2 also let Claude override a few turn types) and gives each test proposition its turn and speaker.

## The four approaches in more detail

- **A1 · Fine-tuned encoder.** ModernBERT-base (150M parameters), trained for 94 minutes on ~105k labelled proposition pairs. For each pair it reads both propositions, the whole dialogue, their distance and whether one speaker said both, and outputs a probability for each relation. This is the 2024 winning recipe, given context.
- **A2 · Frontier LLM, prompted.** Claude Sonnet 5.5, not trained: it reads the whole map after 4 similar solved training maps and lists every relation as JSON. It ran at effort `xhigh`, the only setting that made it reason before answering, because we expected thinking to help with a judgement task. It barely did: +11 focused points on a 20-map pilot, but only about +1 on 225 dev maps, with direction and Conflict unchanged. Our reading is that most errors come from the annotators' conventions, not from faulty logic: thinking longer about whether X supports Y doesn't tell the model that annotators call an elaboration a rephrase, or a concession an attack. What helped was what carried the conventions: worked examples (+4.5 on a pilot) and a prompt revised from observed errors. Three answers per map are combined by vote.
- **A3 · Fine-tuned small LLM.** Qwen3-4B, stored in 4-bit with small trainable adapters (QLoRA) so it fits an 8 GB GPU, trained for 110 minutes on 1,231 maps to write each map's relations. Eight sampled answers are combined by vote.
- **A4 · Decision model.** Jev, TypeSafe AI's new System One model, doesn't write text. It takes a state (here, the map plus a one-page guide to the labels) and a set of *typed questions*, such as a multiple-choice "how is P3 related to P2: supports it, is supported by it, attacks it, rephrases it, or none?", and returns a probability for every option. *No training* means it is used as is: no fine-tuning and no labelled examples. It was asked about each relation two ways, per pair and per proposition.

## Results

| Test F1 (general / focused) | Relations (ARI) | Illocutions (ILO) | **Global** | Dev ARI (honest) |
|---|---|---|---|---|
| 2024 winner (DFKI-MLST, DeBERTa-v3 pairs) | 55.33 / 30.40 | 78.78 / 66.10 | 67.05 / 48.25 | — |
| A1 · ModernBERT-base cross-encoder, fine-tuned | 51.38 / 30.77 | 92.64 / 84.28 | 72.01 / 57.53 | 50.50 |
| **A2 · Claude Sonnet 5.5, whole map, 4 worked examples** | **55.77 / 33.34** | 91.56 / 83.20 | **73.67 / 58.27** | **54.63** |
| A3 · Qwen3-4B, QLoRA fine-tuned | 53.70 / 32.05 | 92.64 / 84.28 | 73.17 / 58.17 | 47.65 |
| A4 · Jev (TypeSafe AI), typed questions, no training | 54.21 / 30.98 | 92.64 / 84.28 | 73.42 / 57.63 | 52.17 |

Global is the mean of ARI and ILO. Every row's ILO comes from the shared illocution labeller; A2's differs because Claude was also allowed to override a few turn types, which cost it 1.1 points. *General* scores every pair, including the ~95% with no relation; *focused* scores only the pairs that have one. Each test configuration was frozen beforehand and scored once. The 11 test maps are the only fair comparison with 2024; the 225 dev maps are the only reliable way to rank our systems. Cost and speed per map: A1 ~1 s and $0; A2 ~25 s and $0.18 (API-equivalent); A3 ~22–40 s and $0; A4 0.8 s and $0.003.

## What makes the numbers trustworthy

- **Scorer verified:** the port of the official scripts agrees with them in 688 of 688 comparisons and reproduces the published majority baseline.
- **No leakage:** the dev split is grouped by episode, recovered from the debate panels because maps carry no episode label (1,250 / 228 maps, 0 leaked pairs). A node-ID artefact that gives away 74% of turn–proposition links was banned in every approach.
- **Honest tuning:** thresholds are cross-fitted across dev episodes, and differences between systems are tested with a paired bootstrap over 225 maps.
- **Test set untouched until the end:** each test configuration was fixed on dev before its single scored run. Later checks on test are labelled post-hoc and never fed back into a system.
- **Traceable:** every number in the reports comes from a committed file under `results/`, and every decision (88 of them) is in [`decisionlog.md`](decisionlog.md) with its rationale and status.

## Findings in more detail

1. **The win over 2024 is mostly a Task B win.** Task B is largely a matching problem: in the training data, text similarity alone links 94% of turns to the right proposition, and 90% of illocutions are *Asserting*. On dev, a trained labeller with one-to-one decoding links 98.3% of turns at 99.6% precision. General models doing Task B end to end did not beat it on dev: Claude Sonnet 5.5 scored 88.03 / 77.48 and Jev 85.35 / 75.76, against 90.08 / 80.64. Jev even links better, but both lose on typing the question types ([03 report §11](reports/03-ilo-labeller.md), [discussion](reports/03-ilo-discussion.md); the test set was not re-run).
2. **Relation mining is still hard.** On dev, no method closes more than about a third of the gap between predicting nothing (30.9) and labelling every candidate pair correctly (98.6). On test, the best system (Claude) gets 53 of 124 relations right and finds 2 of 13 attacks.
3. **Each model has a recognisable profile.** Jev reads each question literally, so it finds the most support but calls elaborations "support" where annotators wrote "rephrase". The trained models learn the annotators' habits. Claude is the best all-rounder.
4. **How a judgement is posed matters as much as the model.** Asking Jev each relation two ways scored 52.2 on dev, against 50.5 and 46.3 for either way alone. Reasoning added Claude only 0.8. The fine-tuned 4B LLM learned the output format but not the support judgement.
5. **The benchmark shapes the result.** The rare-label penalty makes guessing an attack unattractive, the general and focused scores reward opposite strategies, and an 11-map test set can't separate systems within about ±8 points.

## Why relations are hard, even for 2026 models

For every pair of propositions a system must answer two questions: **are they related at all**, and if so, **how** (support, attack or rephrase, and in which direction)? The systems handle the second reasonably well; they fail on the first. For the encoder on dev, when it predicts as many relations as gold (so missed and invented counts are equal by construction):

| Error | Count |
|---|---|
| Gold relation missed (called "no relation") | 905 |
| Relation invented for an unrelated pair | 905 |
| Right pair, wrong type | 229 |
| Right pair, wrong direction | 61 |

So 82–86% of every system's errors happen before a relation type is even chosen, and a better type classifier would barely help. The causes:

- **A needle in a haystack, with no surface cue.** A map has about 200 ordered proposition pairs and only about 8 relations. Most connections are implicit: a concession counts as an attack with no "but", an elaboration as a rephrase, and an answer as a rephrase of its question. The model has to infer what the speaker meant.
- **The labels are conventions.** QT30's annotators agree at 0.56 (CASS), so some "errors" are readings a person could defend. Models that apply plain meaning (Jev, Claude) are marked wrong where the convention differs.
- **The data is thin where it matters.** The 1,478 training maps hold only 1,232 attacks, 56% of them within one speaker's own talk. Those within-speaker attacks stay at 7–12% recall for every method.
- **The models are not the main limit.** A 150M encoder, a frontier LLM, a fine-tuned 4B LLM and an untrained decision model land within 7 points of each other. Whole-dialogue context added the encoder 1.5 points and reasoning added Claude 0.8. Claude's real edge is long-range relations: 30–37% recall two or more turns apart, against 12–16% for the encoder.
- **The score dilutes gains.** It averages the relation types with the near-perfect "no relation" class, so Claude's 15-point gain over the encoder in support (Inference) F1 on test moved the relation score by only 4.4.

![Why relation mining is hard](reports/figures/short-why-relations.png)

*Left: the errors each system makes on the 1,873 dev gold relations, mostly missed or invented relations. Right: recall by turn distance.*

**Where the exploration points.** Even with the best system, the 11 test maps are missing more than half of their true links (71 of 124) and about half of the predicted links are wrong (59 of 112), so a map can't be used to analyse a debate without a person correcting it. The next step is therefore better detection of which pairs are connected: reading the whole dialogue (most of Claude's gain is on long-range links), teaching the annotation conventions explicitly, and building each map as a whole rather than pair by pair.

## Repository

```
.
├── README.md             this page
├── reports/              findings; start with 06-short-report.md, then 05-final-report.md
│   └── figures/          report figures, redrawn by reports/short_report_figures.py
├── common/               shared foundation: AIF loader, official-scorer port, split, validation, fine-tuning loop
├── eda/                  exploratory analysis of the corpus
├── ilo/                  Task B: the trained illocution labeller, shared by every approach
├── ilo_llm/              dev-only test: Claude doing Task B end to end, against the labeller (D090)
├── ilo_jev/              the same test with Jev (D099)
├── a1_cross_encoder/     A1 · fine-tuned ModernBERT cross-encoder
├── a2_llm/               A2 · whole-map Claude prompting (prompts, runner, decision layer)
├── a3_finetune/          A3 · QLoRA fine-tuning of Qwen3-4B
├── a4_jev/               A4 · Jev decision model (question design, API client, runner)
├── results/              every committed number: training logs, dev reports, test scores
├── splits/               the fixed, episode-grouped train / dev split
├── reference/            2024 overview paper, annotation and data-format specs, official scorers
├── decisionlog.md        phase map and every decision (D001–D101), with status
├── PLAN.md               the research plan, written before the experiments
├── CLAUDE.md             project brief for the AI agents: task, scoring, data, environment
├── requirements.txt      pinned ML environment (Python 3.12, torch 2.14 + CUDA 13.0)
└── requirements-a3.txt   separate environment for A3's QLoRA stack
```

`data/`, `cache/` and `checkpoints/` are git-ignored. The corpus is not redistributed here.

## Reproducing

1. **Data.** Download the DialAM-2024 data from the shared task ([dialam.arg.tech](http://dialam.arg.tech); QT30 is also at [corpora.aifdb.org/qt30](http://corpora.aifdb.org/qt30)) into `data/training/` (1,478 maps), `data/test/` (11 input maps) and `data/evaluation/` (their gold).
2. **Environment.** Linux or WSL2 with an 8 GB NVIDIA GPU: `uv venv ~/.venvs/dialam --python 3.12 && uv pip install --python ~/.venvs/dialam/bin/python -r requirements.txt`. A3 uses its own environment, built from `requirements-a3.txt`.
3. **Check the foundation:** `python -m common.check_foundation` (13 gates: scorer parity, split, validator).
4. **Run a stage.** Each report ends with its exact commands. In outline:

| Stage | Commands (from the repo root) | Needs |
|---|---|---|
| Illocution labeller | `python -m ilo.data`, `ilo.train --run=v1`, `ilo.tune --run=v1`, `ilo.evaluate --run=v1` | GPU, ~20 min |
| A1 encoder | `python -m a1_cross_encoder.build_pairs`, `.train --run=v1`, `.tune --run=v1`, `.predict --run=v1 --split=test` | GPU, ~95 min |
| A2 Claude | `python -m a2_llm.runner --run=dev-v1 --export`, `a2_llm.tune --run=dev-v1`, `a2_llm.evaluate` | Claude Code CLI with a Claude plan |
| A3 Qwen | `python -m a3_finetune.train --run=v1`, `a3_finetune.tune --run=v1` | A3 environment, GPU, ~2 h |
| A4 Jev | `python -m a4_jev.runner --run=dev --export`, `a4_jev.tune` | `AI_GATEWAY_API_KEY` (Vercel AI Gateway), ~$1 |
| Report figures | `python reports/short_report_numbers.py`, then `python reports/short_report_figures.py` | numpy, matplotlib |

## How this project was built

The project was designed and directed by Kunal Kamra and implemented with AI coding agents (Claude Code and Claude Science), in gated phases. Each phase was planned, approved, run, checked against a gate set in advance, and logged in the [decision log](decisionlog.md) before the next began. Each system was scored on the test gold once, at a final run the author approved in advance.

## Limitations

- One configuration and one seed per approach; ablations were cut to the core for A3 and A4.
- Training-data contamination isn't ruled out for the hosted models (A2, A4), since QT30 has been public since 2022. Their dev → test changes (A2 +1.1, A4 +2.0) are of the same order as the local encoder's (+0.9), which argues against memorisation, but no probe was run.
- The test set has 11 maps, and the human ceiling in F1 is unknown (QT30's annotators agree at 0.56 CASS).

## References

- Ruiz-Dolz, R., Lawrence, J., Schad, E. and Reed, C. (2024). [Overview of DialAM-2024: Argument Mining in Natural Language Dialogues](https://aclanthology.org/2024.argmining-1.8/). *Proceedings of the 11th Workshop on Argument Mining*, pp. 83–92. ACL.
- Hautli-Janisz, A., Kikteva, Z., Siskou, W., Gorska, K., Becker, R. and Reed, C. (2022). QT30: A Corpus of Argument and Conflict in Broadcast Debate. *Proceedings of the 13th Language Resources and Evaluation Conference*, pp. 3291–3300.
