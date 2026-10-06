# PLAN — DialAM-2024 revisited: four approaches compared

**Target:** beat 2024 best **Global 67.05 (general) / 48.25 (focused)**. ARI (55.3 / 35.9) is the main gap; ILO (81.2) is stronger, but its rare types weigh heavily in macro-F1 (`eda/README.md` §3).
**Research question:** which post-2024 method class solves dialogical argument mining best, at what cost and speed?
**Execution:** one phase at a time (GSD: discuss → plan → execute → verify). Nothing runs until the user starts that phase.
Phase numbers and status live in `decisionlog.md` (its phase map); sections here are named, not numbered.
Background, IAT and scoring details are in `CLAUDE.md`.

---

## Principles that apply to every approach
1. **Same loader, dev split and scorer.** Results are directly comparable. `data/evaluation/` is touched **once**, in the final run.
2. **Beat the rare-label trap.** Macro-F1 is computed per map, so a spurious rare label (CA, Challenging…) adds a zero-scoring class.
   Every approach ends in a **decision layer** that predicts a rare label only above a threshold tuned on dev.
3. **Output valid IAT maps**, not just scored pairs: add the TA-anchored YAs implied by each predicted S-node
   (RA → Arguing, MA → Restating or Default Illocuting, CA → Disagreeing). They aren't scored but keep maps well-formed.
4. **Measure time and cost on every run** → `results/timing.csv`: training wall-clock (total and per epoch), inference time
   (per map and for all 11), peak VRAM, API tokens and $, and cloud-GPU hours and $. Same hardware and settings throughout; median of 3 runs.
5. **Ablate every lever:** switch one component off, re-score on dev, and report what it contributed.
6. **Report everything:** ARI / ILO / Global × general / focused, next to the 2024 table.

---

## Foundation
- **Loader/writer:** reads training, test and evaluation maps (handling evaluation's `{"AIF":…}` wrapper), a graph view (I, L, TA, speaker, dialogue order), and writes AIF.
- **Validator:** every edge endpoint exists; each S-node has at least one incoming I-node and exactly one outgoing I-node; types and labels are legal.
- **Scorer:** a vectorised port of `reference/eval-scripts/`, matching their quirks (target = *first* outgoing edge; ILO scores only L→I).
  **Gate:** must reproduce the majority baseline exactly (29.53 / 35.30 / 32.41) and agree with the official scripts on sample outputs.
- **Dev split:** hold out about 15% of training maps, **grouped by episode** so the same debate never appears in both train and dev.
- **Timing harness:** a context manager that logs to `results/timing.csv`.
- **Deps:** scikit-learn, torch (CUDA), transformers; everything else per phase.

## EDA and structural priors
Measure the hypotheses that later phases rely on:
- L→I alignment: how often the best text-similarity match is the gold proposition; how analyst-copy L-nodes are recognised.
- **TA distance** between the locutions behind related I-pairs. This sets the pruning window: the smallest window that keeps about 98% of gold relations.
- Speaker patterns (CA across speakers? RA within the same speaker?), Q→MA frequency, and direction statistics for MA, CA and RA.
- Label distribution per map (how often a map has *no* CA, which is the rare-label trap in numbers).
- **Leakage check:** do the 11 test maps overlap training text?
- ~~Jev-assisted profiling~~ (a turn-kind `choice` per locution, a contradiction yes/no per pair): moved to A4 (D017), then dropped (D073), because A4.3's per-type breakdown answers the same question.
- **Deliverable:** a short EDA note plus the pruning window. Every number is used by a later phase.

## Shared ILO labeller (trained)
ILO is half the Global score. The EDA showed that its surface cues are weak ("?" is almost absent from the transcripts) and that rare types weigh heavily in macro-F1.
So ILO is **learned, not hand-written** (user decision, 2026-09-30; details in `decisionlog.md` D025).
1. **Candidates:** every (dialogue turn, proposition) pair in a map, narrowed to each turn's top-K most text-similar propositions. K is set so that ≥ 99.5% of gold links survive. Analyst copies are never candidates.
2. **Model:** ModernBERT-base reads the speaker-tagged turn, the proposition and the neighbouring turns, and outputs P(None, Asserting, Pure / Assertive / Rhetorical Questioning, Agreeing, …).
3. **Decode:**
   - *Link:* one-to-one matching on 1 − P(None), kept only above a threshold (a turn may express nothing, and a proposition may have no turn).
   - *Type:* the most probable type, with per-type thresholds for the rare ones (fallback: Asserting).
4. **Train / tune:** the train split for training. Dev maps, in test-input format, for early stopping and thresholds.
- *Better than 2024:* Pungene aligned with BERTScore alone and classified each pair independently. We learn linking and type together, and add one-to-one decoding and rare-type thresholds.
- *It also feeds ARI:* its L→I links give every test proposition its dialogue position and speaker, which the ARI approaches need.
- *Exit:* on dev, linking accuracy must beat the 94.2% text-similarity baseline, and ILO F1 (general / focused) is reported. Any approach may replace the labeller, but only if it beats it on dev.

---

## Approach 1 — PyTorch cross-encoder (trained, discriminative)
**Idea.** The 2024 winning recipe, with its known flaws fixed. Same model family, better inputs and better decisions.

**End to end:**
1. **Candidates:** all I-pairs within the pruning window set by the EDA. At k = 8 that keeps 98% of relations and removes about a quarter of the pairs, so most of the imbalance is left to the loss and thresholds.
2. **Input text:** `[proposition A] [SEP] [proposition B] [SEP] dialogue window around both (speaker-tagged) + TA distance + same-speaker flag`.
3. **Model:** DeBERTa-v3-large or ModernBERT-large (8k context), with a 5-way head
   {A→B Inference, B→A Inference, Conflict, Rephrase, None}. Direction is a label, and each unordered pair is scored once.
4. **Training (PyTorch + HF Transformers):** fp16, gradient checkpointing and accumulation to fit 8 GB; class-weighted or focal
   loss to lift CA; early stopping on dev *map-level* macro-F1.
5. **Decision layer:** per-class thresholds (CA strictest) tuned on dev; then consistency repair: one relation per pair, no
   A→B *and* B→A inference, and linked-RA merging when several premises share a conclusion and appear together.
6. **ILO:** the shared ILO labeller (trained ModernBERT). Its L→I links also give each test proposition its dialogue position and speaker.
7. **Assemble** the AIF output with anchoring YAs.

**Why it beats 2024.** The 2024 encoders saw two sentences in isolation, trained on the full imbalanced pair space, and used
argmax decisions. Here the model gets dialogue context and speakers, pruning reduces the imbalance, and thresholds are tuned
to the real metric, which fixes the pessimistic/optimistic swing seen in 2024. The model architecture is **not** the source of
the gain. Expect a solid, cheap improvement, mostly on the general score. This is the baseline the others must beat.

**Ablations:** −context window · −speaker/TA features · −pruning · −thresholds (argmax) · DeBERTa vs ModernBERT.
**Timing:** training time per epoch and total on the RTX 5060 Ti; inference per map.
**Risks:** long context windows slow training (tune the window size); calibration drift → temperature scaling on dev.

---

## Approach 2 — Whole-map LLM, no training (Claude Sonnet 5.5 on a Claude Pro subscription)
**Idea.** A reasoning model reads the *entire* map and lists every relation at once, so each decision can see the whole structure.
Full plan, concepts and the reason for each choice: [`reports/A2-whole-map-llm.md`](reports/A2-whole-map-llm.md) (D051–D053).

**Model and route (D052).** Claude Sonnet 5.5 only, called through Claude Code's scripted mode (`claude -p`) on the user's Claude Pro login.
Spend is $0; speed is bounded by Pro's usage limits. The runner pauses at a limit and resumes on the user's word. The API-equivalent $ is logged per call.

**End to end:**
1. **Serialise** (`common/serialise.py`, shared with A3/A4): turns T1… in dialogue order, each with its speaker and the proposition(s) the ILO labeller links to it (gold links on training maps).
   Propositions are renumbered P1… so node IDs never appear (D029).
2. **Prompt:**
   - *System prompt:* relation and turn-type definitions, IAT direction rules, and the conventions the EDA measured (neighbouring turns, same-speaker Inference, rare Conflict, density).
   - *User message:* k retrieved worked examples (TF-IDF over training maps), then the target map.
3. **Output:** schema-validated JSON (`--json-schema`, with per-map ID enums). It holds relations {type, from, to, confidence} plus a type for each linked turn.
4. **Self-consistency:** 3 samples of the same prompt (D054). Each pair's score is its confidence-weighted vote share, written in A1's 5-label pair format.
5. **Decide:** A1's per-type thresholds, cross-fitted by dev episode. Conflict / Rephrase direction comes from dialogue order (D020).
   The LLM's turn types replace the labeller's only if that beats it on dev.
6. **Assemble** with A1's `build_map`, validate, then score officially.

**Phases:**
- **A2.1** setup and verification;
- **A2.2** prompt development on a ~50-map *training* pilot: k ∈ {0,2,4,8} and effort low/medium/high, ≤ 2 revision rounds, then frozen. Dev stays untouched;
- **A2.3** 3 samples × 225 dev maps, thresholds, honest score next to A1;
- **A2.4** complete maps and ablations;
- **A2.5** one test run.
- About 1,350 calls in all.

**Why it beats 2024.** The 2024 LLM attempts (Turiya prompting, Pokemon's LLaMA) used weaker models, short contexts and free-text
parsing, and they lost to encoders. Current models reason over long contexts, follow schemas reliably, and see the whole map. That
targets 2024's worst failures directly: **direction** and **convergent vs divergent** structure are global, multi-step judgements.

**Ablations (D053):**
- *Free, re-scored from saved answers:* 1/2/3 samples · votes vs stated confidence · thresholds vs majority vote · direction rule vs the model's own · relations > 10 turns apart dropped · LLM vs labeller turn types.
- *From the pilot:* k sweep · effort sweep.
- *With new calls:* **minimal reasoning** (effort low) on dev.
**Timing/cost:** seconds, tokens and API-equivalent $ per call; calls per usage window; spend $0.
**Risks:**
- usage limits (measured early; the paid ablation shrinks first);
- hidden context or another model answering inside Claude Code (checked on every answer);
- samples that barely differ (stated confidence still grades);
- no seed (raw answers committed);
- **contamination**: QT30 has been public since 2022, so it is reported as a caveat, with the dev → test shift compared against A1's.

---

## Approach 3 — Fine-tuned open LLM (QLoRA, generative)
**Idea.** Teach an open model the annotators' habits through its weights rather than through the prompt. It keeps A2's whole-map view and learns what A1 learned from the data.
Full plan, concepts, the reason for each choice and the independent review: [`reports/A3-finetuned-llm.md`](reports/A3-finetuned-llm.md) (D065).

**Model and route (D065).** Qwen3 loaded in 4-bit, with LoRA adapters (QLoRA), on the local RTX 5060 Ti, in a separate venv (`~/.venvs/a3`). Spend is $0.
- **Model rule:** Qwen3-8B if, at the longest training example, its peak is ≤ 7.5 GiB, it trains at ≥ 200 tokens/s and generation fits a batch of ≥ 2.
- **Otherwise:** Qwen3-4B-Instruct-2507.
- **Why it's close:** Qwen3-8B's untied 16-bit embeddings alone take ~2.5 GB.
- **Outcome (A3.1, D066):** 8B fills the card by 1,200 tokens and crashes before the longest example, so the model is **Qwen3-4B-Instruct-2507** (4.25 GiB peak, 789 tokens/s).
- **Final (D070–D072):** honest dev ARI 47.65 / 27.23, below A1 and A2, with both success tests failing; test Global 73.17 / 58.17. Ablations not run (user decision, D071). Report: `reports/A3-finetuned-llm.md`; discussion: `reports/A3-discussion.md`.

**End to end:**
1. **Data:** the 1,231 test-like training maps → (instruction + A2's serialised map, compact gold answer without confidence). Gold links on training maps; the ILO labeller's on dev and test. The giant maps are left out (a retrain ablation).
2. **Train:**
   - QLoRA, r = 16 on all linear layers;
   - answer-only loss;
   - 3 epochs, with an adapter every half epoch;
   - memory path: our own PEFT wrapping and a fused loss.
3. **Pick** the half-epoch adapter with the best greedy dev ARI (as A1).
4. **Generate:** turn types once (greedy), then the relations sampled 8 times from that fixed start (seeded). A per-map grammar keeps every answer valid.
5. **Decide:** vote shares (token probabilities if the votes are all-or-nothing, by a fixed rule) → A1's 5-label pair format → A1's thresholds, cross-fitted by dev episode.
6. **Turn types:** the LLM's replace the labeller's only if they gain ≥ 0.5 honest ILO and win in ≥ 4 of 5 dev folds.
7. **Assemble** with A1's `build_map`, validate, then score officially.

**Phases:**
- **A3.1** setup and feasibility (lossless round trip, grammar check, memory / speed probes, model rule);
- **A3.2** training loop (memorise 32 maps);
- **A3.3** training run;
- **A3.4** dev run, thresholds, honest score;
- **A3.5** complete maps, combination preview, ablations;
- **A3.6** one test run.

**Success (honest dev), fixed in advance.** A3 succeeds if either holds:
- it beats A2's ARI general (54.63), with a paired per-map bootstrap interval above 0;
- adding A3's pair scores to an A1 + A2 average gains ≥ 1.0 general.

The hypotheses H1–H6 each have a pass rule (report §7).

**How it differs.** Versus A1: it writes the whole map's answer, so decisions compete, and it has no fixed classifier head. Versus A2: the conventions sit in the weights, so no examples are needed in the prompt, and it runs locally at no marginal cost, but it costs training time.

**Ablations:**
- *Free:* samples 1 / 2 / 4 / 8 · votes vs token probabilities · greedy vs votes · thresholds vs majority vote · epoch curve · direction rule · LLM vs labeller turn types.
- *Generation:* the untrained model, zero-shot · no constrained decoding.
- *Retrain:* with the giant-map chunks · 4B vs 8B (only if 8B fits).

**Timing:** training wall-clock (the headline cost), tokens per second, peak VRAM; roughly 15–35 GPU hours in all.
**Risks:**
- Library support for Blackwell on CUDA 13 (checked first);
- 8B not fitting (the rule switches to 4B);
- overfitting 1,231 maps;
- all-or-nothing votes (the token-probability rule);
- contamination (the dev → test change is compared with A1's and A2's).

---

## Approach 4 — Jev System One decisions (no training)
**Idea.** Jev (TypeSafe AI) answers **typed questions with probabilities** instead of writing text. With the whole map in its state, it gives
A2's whole-map view and A1's per-pair probabilities, with no training.
Full plan, concepts, the reason for each choice and the hypotheses: [`reports/A4-jev.md`](reports/A4-jev.md) (D073).

**Outcome (final, D079–D081):** test Global 73.42 / 57.63 (ARI 54.21 / 30.98); honest dev ARI 52.17 / 32.45, 2nd of four systems. Both success tests narrowly fail; H4 (direction) and H5 (speed) pass. $1.70 in all, including a robustness re-run of dev (D082). Report: `reports/A4-jev.md`; discussion: `reports/A4-discussion.md`.

**Model and route (D073, D075).** `typesafe-ai/jev` (jev-1.13 today; the version is logged on every answer), called from Python with `httpx` at the Vercel AI Gateway's `POST /v1/evaluate`.
- Jev isn't on the gateway's free tier (D074), so A4 runs on a small paid credit, with auto top-up off and a cap (D075).
- Limits: 64k tokens per request (32k for the state plus the longest question) and ≤ 255 options per choice.

**How Jev sees the problem.** Every question is answered against the **same state**, so context is shared. But answers come out **in parallel and don't see each other**, so consistency comes from our code, and from the source questions below.

**End to end:**
1. **State:** a guide (A2's `system_v1` definitions and measured habits, condensed, plus the concession rule for Conflict) and the serialised map (`common/serialise.py`, D029). No worked examples: Jev's accuracy drops as unrelated content grows in the state.
2. **Questions, in one request per map** (split if a map exceeds the request limit):
   - **pair:** one 5-label `choice` per candidate pair (A1's labels, pruned at 10 turns);
   - **source:** one `choice` per proposition over its candidate targets × relation types, plus "none". The targets compete. Conflict and Rephrase are offered only towards earlier propositions (D020);
   - Every question is asked in **two option orders** and the probabilities averaged, against Jev's documented first-option bias. A4.1 kept both: they agree on only 90.8% of choices.
   - Turn-type questions were dropped (D076): the labeller keeps Task B.
3. **Scores:** answers → A1's 5-label pair format (pair view, source view, or their mean, chosen in the pilot).
4. **Decide:** A1's temperature + per-type thresholds, cross-fitted by dev episode; then A1's `build_map`, with the labeller's turn types.
5. **Calibration check:** a reliability plot and class-wise ECE on dev, against A1 on the same pairs. Vendor claims are tested, not assumed.

**Phases:**
- **A4.1** setup and feasibility: lossless round trip; a live probe on 5 training maps measures billing, speed, repeatability and option-order bias; budget projection. **Passed (D076);**
- **A4.2** one pilot pass on A2's 50 *training* maps: the view (pair / source / both) is chosen by a pre-set rule, with at most 1 wording fix, then frozen. Dev stays untouched;
- **A4.3** dev run, thresholds, honest score next to A1–A3;
- **A4.4** complete maps and the combination test, from saved answers;
- **A4.5** one test run.

**Success (honest dev), fixed in advance.** A4 succeeds if either holds:
- with no training, it beats A1's ARI general (50.50), with a paired per-map bootstrap interval above 0;
- adding A4's pair scores to the A1 + A2 average gains ≥ 1.0 general.

The hypotheses H1–H5 each have a pass rule (report §6): calibration, competition between targets, concessive Conflict, direction without reasoning, speed.

**Deferred: iterative commitment ("4b").** Commit the confident answers, write them into the state, and re-ask the rest. The source questions test the same idea in one pass, with no risk that an early mistake spreads. 4b runs only if H2 fails (D073).

**Why it might beat 2024.** Each decision uses whole-map context, which 2024's pairwise encoders lacked, *and* comes with probabilities, which 2024's LLMs lacked, so the rare-label trap can be controlled.

**Ablations:** none beyond the core (user, D076). Only pair vs source vs both, from saved answers, which H2 and the view choice need.

**Timing/cost:** seconds per request and per map (tests the speed claim), tokens, and the gateway's API-equivalent $ per request. Measured in A4.1: the state is billed once per request, so the pilot, dev and test together cost ≈ $0.80 at list price, paid from the gateway credit (D075).
**Risks:**
- early-access API changes (the version is logged and raw answers committed);
- first-option bias and distractor state (two orders, no examples by default);
- over-linking from independent questions (source questions, thresholds);
- unknown training data, so contamination is reported as a caveat, with the dev → test change compared against A1–A3.

---

## Combination
**Outcome (D083–D084):** the mean of A1 + A2 + A4 pair scores, pre-registered, gives test Global **74.51 / 59.47** (ARI 56.39 / 34.65); A1 + A2 alone gives 75.17 / 60.32. Routing was not run. Comparison: `reports/04-comparison.md`.

Route by confidence. Each approach does what it's best at:
1. A **fast calibrated pass** (A1 or A4, whichever scored better on dev) decides the confident pairs.
2. **Uncertain pairs** (probability within a band) go to A2 or A3 to adjudicate with whole-map reasoning.
3. The same decision layer and repair, then the AIF output.
- **Ablations:** −router (A2 on every pair) · different uncertainty bands. Report the cost saved versus the F1 gained.
- **Expected best on the metric:** calibrated probabilities where they're reliable, reasoning where they aren't.

## Final run and write-up
- **Done (D085):** [`reports/04-comparison.md`](reports/04-comparison.md) compares the four approaches and the combination.
- **Done (D086):** [`reports/05-final-report.md`](reports/05-final-report.md) is the final report: the whole project end to end, including a critique of what the benchmark measures. The per-approach reports were brought up to date with the final results.
- **Write-up sources:** the per-phase findings reports in `reports/` (index: `reports/README.md`), plus `decisionlog.md`.
- **One run per system** on the 11 test maps, scored with the official logic.
- **Main table:** A1–A4 plus the combination, against the 2024 results. Columns: ARI / ILO / Global (general and focused), training time, inference time, cost.
- **Ablation table** from dev.
- **Error analysis** on the 2024 failure modes, using the paper's cases (test maps 3–8 and 10): conflict, RA direction, convergent vs divergent, linked arguments.
- **Jev:** a calibration assessment and a per-relation breakdown (where the System One approach does well and where it doesn't).

## Success criteria
- At least one system beats **67.05 / 48.25** Global.
- Each approach has its F1, time, cost and ablations recorded.
- It's clear *which* 2024 failure each post-2024 lever fixed.
