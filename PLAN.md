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
- **Jev-assisted profiling** (cheap, run over the full corpus): per locution `choice` {claim, question, answer, agreement, reported speech};
  per pair `boolean` "does B contradict A?". Compare with gold to see which relations are visible on the surface and which need reasoning.
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

## Approach 2 — Whole-map LLM, no training (API)
**Idea.** A frontier reasoning model reads the *entire* map and outputs every relation at once, so each decision can see the whole structure.

**End to end:**
1. **Serialise** the map as text: I-nodes with IDs, and the dialogue in order (speaker: locution → linked I-ID candidates).
2. **Prompt structure:**
   - **Definitions (static, prompt-cached):** every relation and illocution label with a one-line definition and an example;
     IAT rules (premise → conclusion; MA and CA point to the earlier proposition; a question's answer is linked by MA; linked vs convergent);
     "predict Conflict only when clearly contradicted".
   - **Few-shot examples:** k training maps, each as serialised input → gold relation list.
   - **Target map** plus a JSON schema: `{relations:[{from,to,type}], illocutions:[{locution,proposition,type}]}`.
3. **Selecting examples:**
   - *Retrieval:* embed every training map and take the k nearest to the target, never from the same episode.
   - *Variant:* force at least one example containing CA.
4. **Choosing k:** sweep k ∈ {0, 1, 2, 4, 8} on dev and plot F1 against tokens, cost and time. Pick the knee of the curve.
   The ceiling is context length: about 2–4k tokens per map, so k=8 is about 30k tokens.
5. **Self-consistency:** n samples (e.g. 5). Each relation's vote share acts as a **pseudo-probability**, so the Principle-2 thresholds apply here too.
6. **Validate:** discard unknown IDs and labels; apply the same consistency repair as A1; assemble the AIF.

**Why it beats 2024.** The 2024 LLM attempts (Turiya prompting, Pokemon's LLaMA) used weaker models, short contexts and free-text
parsing, and they lost to encoders. Current models reason over long contexts, follow schemas reliably, and see the whole map. That
targets 2024's worst failures directly: **direction** and **convergent vs divergent** structure are global, multi-step judgements.

**Ablations:** k sweep · −definitions (examples only) · retrieval vs random examples · n=1 vs n=5 · with vs without reasoning mode.
**Timing/cost:** tokens and $ per map; latency; prompt-iteration time on dev.
**Risks:** no native probabilities (mitigated by voting); run-to-run variance (fix the seed/temperature; report the spread);
**contamination** (QT30 has been public since 2022; note it, and compare performance on more versus less "famous" episodes if feasible).

---

## Approach 3 — Fine-tuned open LLM (QLoRA, generative)
**Idea.** Teach a mid-size open model the annotation style through its weights rather than through the prompt.

**How fine-tuning works here.** Each training example is a pair (serialised map → gold relation list as text). The model is
trained with next-token loss **on the answer tokens only**, so it learns to write the annotators' relations for any map.
- **LoRA:** freeze the base model and train small low-rank adapter matrices (about 0.1–1% of parameters).
- **QLoRA:** also keep the frozen base in 4-bit, which is how an 8B model fits in 8 GB.
- **Stack:** PyTorch + HF Transformers + PEFT + bitsandbytes (or Unsloth for speed). The same PyTorch environment as A1.

**End to end:**
1. **Data:** 1,478 maps → prompt/answer pairs, using the same serialisation as A2. Split maps above the context budget at dialogue boundaries.
2. **Model:** a Qwen3-8B-class instruct model, loaded in 4-bit with LoRA r=16 on the attention and MLP layers.
3. **Train:** 1–3 epochs with sequence length about 4k and gradient checkpointing; dev loss and dev F1 every half epoch.
4. **Inference:** constrained decoding with a JSON grammar, so only valid IDs and labels can be written. Answer-token probabilities
   serve as rough confidence for thresholds.
5. **Repair and assemble** as in A1 and A2.

**How it differs.** Versus A1: it generates the whole map's answer, sees everything at once, and has no fixed classifier head. Versus A2:
the knowledge sits in the weights, so no examples are needed in the prompt, it's cheap per call and runs locally, but it costs training time.

**Why it could beat 2024.** No 2024 team succeeded with a fine-tuned LLM: Pokemon tried LLaMA and fell back to encoders. Current 8B
models are much stronger, and QLoRA makes whole-map training feasible on consumer hardware. This tests whether *learned style + whole-map view*
beats a classifier.

**Ablations:** base model with no fine-tuning (zero-shot) vs fine-tuned · epochs · with vs without constrained decoding.
**Timing:** training wall-clock (the headline cost of this approach), tokens per second at inference, peak VRAM.
**Risks:**
- 8 GB is tight: fall back to a smaller model (3–4B), or rent one cloud GPU run (logged in timing).
- Weak calibration makes the rare-label trap harder to control.
- Long maps overflow the context → split them, then merge.

---

## Approach 4 — Jev System One decisions (no training; Vercel AI SDK)
**Idea.** Jev (TypeSafe AI) answers **typed questions with calibrated probabilities** instead of writing text. That combines A2's
whole-map view with A1's probabilities.

**How Jev sees the problem.** Every question is answered against the **same state**, which holds the whole map, so context
is shared. But answers are produced **in parallel and don't see each other**: the answer for (A,B) doesn't know what was said
for (B,C). Consistency has to come from our code, or from iterating (4b).

**End to end (4a, a single pass):**
1. **State (≤32k tokens):** the target map (I-nodes and speaker-tagged dialogue in order) plus 1–2 retrieved training maps with gold relations as
   worked examples. This is our form of few-shot, since Jev has no example slot.
2. **Questions, one request per map:**
   - per candidate pair (pruned as in A1): `choice` {A→B inference, B→A inference, conflict, rephrase, none}.
     **The label definitions go in each option's description.**
   - per dialogue L-node: `choice` over candidate I-nodes, then `choice` over illocution types (definitions in the options).
3. **Decide:** thresholds on the returned probabilities (tuned on dev) + consistency repair → AIF.
4. **Calibration check:** reliability plot and ECE on dev. Vendor claims are tested, not assumed.

**4b — iterative commitment (our extension).** Decisions condition on earlier *committed* decisions:
- **Round 1:** ask everything, and commit only the high-confidence answers.
- **Write the commitments into the state** ("I3 → I2: Inference (decided)").
- **Round 2 onwards:** re-ask only the uncertain pairs. Stop when nothing changes or after N rounds, and fall back to thresholds.
- **Order from easy to hard:** L→I alignment and type → question/answer MA → RA and CA with direction.
  This mirrors 2024's finding that doing illocutions first helped.
- **Risk:** an early wrong commitment spreads, so the commit threshold is kept high and tuned on dev.

**Why it beats 2024.** Each decision uses whole-map context (which 2024's pairwise encoders lacked) *and* comes with probabilities (which
2024's LLMs lacked), so the rare-label trap can be controlled. It is also cheap enough to tune on the full dev set: $0.042 per 1M input tokens,
and output is free.

**Ablations:** 4a vs 4b · k example maps in the state (0/1/2) · definitions in the option descriptions vs none · with vs without pruning.
**Timing/cost:** per-request latency (tests the speed claim), requests per map in 4b, and $.
**Setup (at the start of this phase):**
- `experimental_evaluate({ model:'typesafe-ai/jev', state, questions })` via the Vercel AI Gateway (`AI_GATEWAY_API_KEY`).
- A small Node runner (`jev/`) that exchanges JSON with the Python pipeline.
- **Find out first:** the maximum questions per request (about 120–240 pairs per map, batched if needed) and the rate limits.

**Risks:**
- *Our hypothesis, not TypeSafe's:* with no step-by-step reasoning it may lag on RA direction and implicit CA.
- It can't be fine-tuned.
- It's early access (API changes).
- We don't know whether it has seen QT30.

---

## Combination
Route by confidence. Each approach does what it's best at:
1. A **fast calibrated pass** (A1 or A4, whichever scored better on dev) decides the confident pairs.
2. **Uncertain pairs** (probability within a band) go to A2 or A3 to adjudicate with whole-map reasoning.
3. The same decision layer and repair, then the AIF output.
- **Ablations:** −router (A2 on every pair) · different uncertainty bands. Report the cost saved versus the F1 gained.
- **Expected best on the metric:** calibrated probabilities where they're reliable, reasoning where they aren't.

## Final run and write-up
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
