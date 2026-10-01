# Decision log

Every decision in the project goes here as it is made, whether for the shared foundation or for an approach: what, why, what else was considered, and its status.
Newest decisions go at the bottom. This file moved here from `a1_cross_encoder/` (D027).

**Status:** `Proposed` (waiting for the user) · `Approved` · `Done` (carried out) · `Finding` (a measured fact, no choice involved) · `Open` (to be decided at a named phase) · `Superseded by Dxxx`.

## Phase map
This table is the **single record of phase numbering and status** (D027). Shared phases are numbered 0–3. Each approach numbers its own phases (A1.1, A1.2, …), and `PLAN.md` names its sections instead of numbering them.
Each phase follows the same loop: explain it, check it against `PLAN.md`, get approval, execute, verify the gate, log it, then move on.

| Phase | Name | Builds | Gate (must pass to move on) | PLAN section | Status |
|---|---|---|---|---|---|
| 0 | Environment | WSL2 venv `~/.venvs/dialam` (torch CUDA, transformers, sklearn); ModernBERT smoke test `a1_cross_encoder/check_env.py` | GPU visible, bf16 works, ModernBERT forward+backward on GPU, peak VRAM recorded | Foundation (deps) | **Passed (D009, D010)** |
| 1 | Foundation | `common/`: loader + graph view, AIF writer, validator, scorer port, episode-grouped dev split, timing harness | Scorer reproduces 29.53 / 35.30 / 32.41 and matches official scripts; validator passes all 1,478 gold maps; the split recomputes to the tracked `splits/` | Foundation | **Passed (D012–D016, D027)** |
| 2 | EDA | `eda/`: Pruning window *k*, direction stats (MA/CA later→earlier), speaker patterns, CA-absent maps, ModernBERT token lengths, leakage check | Every number feeds a named later decision | EDA | **Passed (D018–D024 proposed)** |
| 3 | ILO labeller (trained) | `ilo/`: candidate filter, (turn, proposition, context) dataset, ModernBERT-base classifier, one-to-one decoding + per-type thresholds; reusable fine-tuning loop | Linking accuracy on dev beats text similarity (94.2%); ILO F1 general / focused on dev reported | Shared ILO labeller | **Passed (D028–D036): dev ILO 89.89 / 80.38 cross-fitted** |
| A1.1 | Pair builder | `build_pairs.py`: candidates, 5-way labels, context text → `cache/` | ≥98% of gold relations survive pruning; gold→pairs→AIF round trip scores at the pruning ceiling; samples read by eye | A1 | — |
| A1.2 | Model + training loop | `model.py`, `train.py` on `common/finetune.py` (built in Phase 3): weighted/focal loss, bf16, grad ckpt, map-level early stop | Overfits a tiny subset; VRAM and epoch time measured; base vs large chosen | A1 | — |
| A1.3 | Training run | Checkpoint + `results/timing.csv` | Dev per-map macro-F1 (argmax) recorded | A1 | — |
| A1.4 | Decision layer | `tune.py` (temperature + per-class thresholds), `repair.py` | ECE before/after; thresholded ≥ argmax on dev | A1 | — |
| A1.5 | Assemble + dev score | `predict.py` → valid AIF with S-nodes and YAs | Validator passes; dev ARI / ILO / Global table next to 2024 | A1 | — |
| A1.6 | Ablations | −context · −speaker/TA features · −pruning · argmax · (DeBERTa vs ModernBERT) | One lever off per run, re-scored on dev | A1 | — |
| A1.7 | Test run | Final A1 row | Run once only, on the user's go | Final run | — |
| A2 · A3 · A4 | Other approaches | Phased (A2.1, …) when each starts | — | A2 / A3 / A4 | Not started |
| — | Combination · Final run | — | — | Combination / Final run | Not started |

---

## Decisions

### D001 — Backbone is ModernBERT
- **2026-09-29 · Setup · by user · Approved**
- A1's encoder is ModernBERT (`answerdotai/ModernBERT-*`), not DeBERTa-v3.
- *Why:* its 8k-token context lets the dialogue window grow far past DeBERTa's 512 tokens. Context is A1's main fix for 2024 failure #1.
- *Consequences:* README step 3 (`--model deberta-v3-large`) needs updating. DeBERTa-vs-ModernBERT stays on the plan's ablation list, and whether to run it is decided at A1.6.

### D002 — Gated, phase-by-phase execution
- **2026-09-29 · Setup · by user · Approved**
- One phase at a time, following the loop above. Nothing runs without approval (consistent with `CLAUDE.md` working rules).

### D003 — A1 builds the shared foundation first
- **2026-09-29 · Setup · by Claude · Approved by user 2026-09-29**
- The repo has no code yet. A1 needs PLAN's Foundation (loader, scorer, dev split), EDA (→ pruning window *k*) and ILO (L→I alignment, which test maps need because they carry no L→I links). These go in a shared `common/` package so A2–A4 can reuse them. A1-only code stays in `a1_cross_encoder/`.
- *Note:* `CLAUDE.md` says the scorer port was checked, but no scorer code exists in the repo. It is rebuilt and re-gated in Phase 1.
- *Alternative:* put everything in `a1_cross_encoder/`. Rejected because A2–A4 would have to duplicate or import across approaches.

### D004 — Dev maps are predicted in test-input format
- **2026-09-29 · Setup · by Claude · Approved by user 2026-09-29**
- Before prediction, dev maps are stripped to what test maps contain (I-nodes, L-nodes, TA chain; no YA L→I links, no S-nodes). Alignment errors then propagate on dev exactly as they will on test.
- *Why:* using gold L→I links on dev would make the dev scores optimistic, and the thresholds would be tuned to a setting that never occurs.

### D005 — Environment: project `.venv`, Python 3.14, torch CUDA 13.0 build
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29 · Superseded by D008** (where it runs; the package choices carry over)
- A `.venv` at the project root, shared by A1 and A3 (the plan says they use the same PyTorch stack). It holds torch 2.14.0+cu130, transformers 5.17, scikit-learn and numpy, pinned in `requirements.txt`.
- *Why cu130:* the RTX 5060 Ti is Blackwell (sm_120), which needs a CUDA ≥ 12.8 build. cu130 wheels exist for cp314, and driver 616.56 supports CUDA 13. cu128 (torch ≤ 2.11) is the fallback.
- *Why a venv:* it keeps the global Python (PyMuPDF etc.) untouched and makes the environment reproducible.
- *Attention kernel:* FlashAttention-2 has no reliable Windows wheels, so ModernBERT runs on PyTorch SDPA. This is slower than FA2 unpadding but needs no custom build.

### D006 — Develop on ModernBERT-base; choose base vs large at A1.2
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29**
- Base (149M params) for fast pipeline iteration. Large (395M) for the real run if the Phase 0 VRAM probe and the A1.2 throughput allow it on 8 GB.

### D007 — Phase 0 blocker: Windows Smart App Control blocks torch
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29: option 1 (WSL2 + Ubuntu)**
- *Finding:* `.venv` is created and torch 2.14.0+cu130, transformers 5.17.0 and scikit-learn 1.9.1 installed cleanly, but `import torch` fails with WinError 4551. The CodeIntegrity log (event 3033/3077) shows Smart App Control (state = 1, on) refusing `torch\lib\c10_cuda.dll`. 12 of torch's 39 DLLs are unsigned, and this affects every Windows torch build. `check_env.py` never reached the GPU.
- *Options:*
  1. **WSL2 + Ubuntu** (recommended). Smart App Control stays on, because Linux binaries run inside the WSL VM, and CUDA reaches the GPU through the Windows driver. Linux also supports Triton/`torch.compile` and FlashAttention-2, which ModernBERT's speed relies on, and bitsandbytes/Unsloth, which A3 needs. *Cost:* a one-off `wsl --install` (admin rights plus a reboot, done by the user), and the environment moves into WSL.
  2. **Turn Smart App Control off.** This is the fastest route, and everything else in D005 stays as it is. *Cost:* it removes a Windows security layer, and on many Windows builds it can't be turned back on without resetting Windows.
  3. **Cloud GPU** (e.g. RunPod or Colab). No local change, but every run costs money and time, and moving data back and forth complicates timing.
- *Not considered:* signing or allow-listing the DLLs ourselves, or any other way around the policy.

### D008 — ML environment runs in WSL2 + Ubuntu 24.04
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29** (the user confirmed WSL works and said to finish Phase 0) (follows from D007)
- **Code and data stay put** in `C:\Argument Mining`, reached from WSL as `/mnt/c/Argument Mining`, so the editor and Windows tools see the same files. Paths in code are relative to the repo, so it runs on either OS.
- **The venv lives on the Linux filesystem** (`~/.venvs/dialam`), as does the HF model cache (`~/.cache/huggingface`). Venvs and caches on `/mnt/c` are slow and have permission problems. The 1,478 small JSON maps are fine to read over `/mnt/c`.
- **Python 3.12** inside WSL, installed with `uv`, instead of 3.14. FlashAttention-2, bitsandbytes and Unsloth publish wheels for 3.12 first. Python code stays compatible with 3.12–3.14.
- **Packages:** torch 2.14.0+cu130, transformers 5.17.0, scikit-learn and numpy, as in D005, pinned in `requirements.txt`.
- **No NVIDIA driver inside Ubuntu.** WSL uses the Windows driver (616.56) through `/usr/lib/wsl/lib`, and installing a Linux driver breaks that.
- **The Phase 0 rerun adds an attention probe:** SDPA (baseline) vs FlashAttention-2 / `torch.compile` for ModernBERT, if they install cleanly. This reopens D005's "SDPA only" note.
- **The Windows `.venv`** (~3 GB, unusable for torch) is kept until the WSL setup works, then deleted only if the user agrees.

### D009 — Phase 0 result: gate passed, with a workaround
- **2026-09-29 · Phase 0 · by Claude · Finding**
- *Measured* (`check_env.py`, WSL, bf16 autocast, fp32 weights + AdamW, SDPA attention):

  | Probe | Peak VRAM | Step | Tokens/s |
  |---|---|---|---|
  | ModernBERT-base (150M), 2048 tok × 1 | 3.32 GiB | 0.139 s | 14,736 |
  | ModernBERT-large (396M), 1024 tok × 2, grad ckpt | 7.44 of 7.9 GiB | 1.422 s | 1,440 |
- *Finding 1:* **torch 2.14 routes some eager GPU ops through Triton**, for example the matmul in ModernBERT's rotary embedding, and Triton needs a C compiler that Ubuntu lacks. Until D010 is done, every run sets `TORCH_DISABLE_NATIVE_JIT=1`, which falls back to the standard CUDA kernels.
- *Finding 2:* **transformers 5.x removed ModernBERT's `reference_compile`.** Compiling now means calling `model.compile()`. The attention options are `sdpa`, `flex_attention` (which needs Triton, so a compiler) and `flash_attention_2`.
- *Finding 3:* **FlashAttention-2 is dropped.** Its prebuilt wheels stop at torch 2.10 (v2.8.3), and building it from source needs the CUDA toolkit and hours of compiling. This supersedes D008's FA2 probe.
- *Finding 4:* **ModernBERT-large full fine-tuning is at the VRAM limit.** At 7.44 of 7.9 GiB its step is about 3× slower than its size predicts, most likely because the driver is spilling into system RAM (to confirm in A1.2). The options for A1.2 are 8-bit AdamW, shorter inputs, or base (see D006).
- *Finding 5:* **WSL gets only 7 GB of the 16 GB RAM** by default. This is fine for now, and can be raised with `.wslconfig` if data loading needs more.

### D010 — Install a C compiler in WSL (`build-essential`)
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29**
- *Why:* it removes the D009 workaround and enables `torch.compile` and `flex_attention`, both possible speedups. Base is already fast enough, so this is not blocking.
- *How:* run `wsl -u root apt-get install -y build-essential` from Windows (this needs no Ubuntu password), then re-run `check_env.py` without the workaround.
- *Done:* installed `build-essential` (gcc 13.3) plus `python3.12-dev`, because Triton also needs `Python.h`. `TORCH_DISABLE_NATIVE_JIT` is **no longer needed**. Re-run results for ModernBERT-base at 2048 tokens:

  | Attention | Tokens/s | Peak VRAM |
  |---|---|---|
  | SDPA | 15,089 | 3.32 GiB |
  | SDPA + `model.compile()` | 14,372 | 3.12 GiB |
  | **flex_attention** | **18,746 (+24%)** | 3.21 GiB |
- *Consequence:* flex_attention is the working default, and it gets re-checked with real, variable-length padded batches in A1.2. `torch.compile` gave no gain here, so it's dropped for now. Large is unchanged (7.44 GiB, 1,502 tokens/s).

### D011 — Delete the unusable Windows `.venv`
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29 · Done** (3.5 GB removed)
- It is about 3 GB and torch can't load there (D007). `requirements.txt` now pins the WSL environment (59 packages).

### D012 — Episodes are recovered from panels; dev = 5 whole episodes
- **2026-09-30 · Phase 1 · by Claude · Done** (within the approved Phase 1 scope)
- *Problem:* the maps carry no episode label. Locution `start` dates are mostly import dates (824 maps share 2020-10-15, and 416 have none).
- *Method* (`common/split.py`):
  1. Join maps that share a specific speaker within 20 nodeset IDs. The host and "audience/unknown/speaker" labels don't count.
  2. Groups of ≥10 maps become episodes, and each episode's panel is the speakers in ≥3 of its maps.
  3. Merge groups that share ≥3 panellists. These are late re-annotation batches of the same episode, e.g. 18751… and 25554….
  4. Every other map joins the episode whose panel it overlaps most, or else the episode whose contiguous ID run contains it.
  5. Pick whole episodes (seed 13) until dev is about 15%.
  6. Leak guard: any training map that shares a panellist unique to a dev episode, or two non-recurring speakers with one dev map, moves to dev. This repeats until nothing changes.
- *Result:* 29 episode groups (the corpus has 30). 1,392 maps assigned; 86 unassigned, which go to train. **Train 1,250 / dev 228 (15.4%)** from episodes 0, 3, 10, 18 and 19. **0 leaky pairs.** Files: `splits/{train,dev}.txt`, `splits/episodes.json`.
- *Caught along the way:* annotator names were being read as speakers (see D014), and twin groups hid a leak. The audit found a real leaked map (21441, two audience members from dev episode 18) before the ID-run rule fixed it.

### D013 — Maps with no gold relations are left out of the focused mean
- **2026-09-30 · Phase 1 · by Claude · Done**
- The official script crashes (sklearn `ValueError`) when a map's focused list is empty. `common/score.py` leaves such maps out of that mean and reports how many it left out. This affects 3 training maps and none of the test maps.

### D014 — Analyst copies are bookkeeping, detected per map
- **2026-09-30 · Phase 1 · by Claude · Done**
- An L-node is an **analyst copy** if its prefix (`"Anna: Speaker : …"`) belongs to an analyst. A prefix counts as an analyst's if it yields an exact copy of another L-node in the same map. This catches the partial copies too: 2,484 of 2,488 compared with a global name list, and it also finds the test set's new analyst "Amelie". **Dialogue** = the non-copies that have a speaker or sit in the TA chain.
- *Evidence* (all training maps):

  | L-node kind | n | In TA chain | Anchors an I-node |
  |---|---|---|---|
  | Dialogue | 25,741 | ~98% | ~80% |
  | Copies (exact + partial) | 23,203 | 0% | **0%** |

  Only 7 of 20,746 gold L→I anchors start outside the dialogue.
- *For Phase 3:* **2,947 of 23,465 I-nodes (12.6%) have no L→I anchor at all**, so the linking must allow a proposition to have no locution.

### D015 — Scorer gate: plan numbers exact; the paper's table is not a per-map mean
- **2026-09-30 · Phase 1 · by Claude · Done**
- The port equals the **unmodified official scripts in 688/688 comparisons**: the 11 majority maps, 11 gold-vs-gold maps, and 150 perturbed training maps exercising every quirk.
- Majority baseline F1 (general) = **29.53 / 35.30 / 32.41**, exactly the plan's gate. The paper prints 29.52 / 35.29 / 32.40. Its ARI recall of 30.28 can't be a mean of per-map recalls, because each is 1/k and ours is exactly (4·¼ + 7·⅓)/11 = 30.30. So the gap comes from how the paper computed its table, not from the port.
- `CLAUDE.md` said "reproduces exactly", which was inaccurate, and is now corrected.

### D016 — Gold irregularities are reported, not repaired
- **2026-09-30 · Phase 1 · by Claude · Done**
- 60 of 1,478 training maps have irregular nodes: 56 orphan YAs, about 180 S-nodes missing a premise or a target, and 12 S-nodes pointing at an L-node or another S-node. The scorer reads them exactly as the official one does. Our own outputs must pass `validate()` with zero problems, and stripped maps do.
- *Also checked:* `strip_to_input(gold)` reproduces all 11 test inputs exactly (D004). The whole Phase 1 check runs in about 13 s (`results/timing.csv`).

### D017 — Phase 2 scope, location and data
- **2026-09-30 · Phase 2 · approved by user 2026-09-30**
- *Location:* the analysis is shared by every approach, so it lives in a top-level `eda/`. It holds the script (`eda/run_eda.py`), the raw numbers (`eda/findings.json`) and the findings document (`eda/README.md`), which has a narrative and diagrams, as the user asked.
- *Data:* the **training split only** (1,250 maps), so dev stays clean for tuning. Test **inputs** are read only for the size comparison and the text-overlap check. Evaluation gold is not read.
- *Deferred:* the plan's Jev-assisted profiling moves to approach 4. It is Jev-specific and costs API money.
- *Map boundaries* (user question): maps are the annotators' nodesets and come with the data. They are the scoring unit, and relations never cross maps. The median map has 14 dialogue turns (10th–90th percentile 9–22) and 2 speakers, and 27% contain more than one stretch of dialogue. Test maps stand alone, so model context comes only from inside the map.

### Phase 2 result
- **2026-09-30 · Phase 2 · Done.** The findings are in **`eda/README.md`**, a narrative with 5 figures. The numbers are in `eda/findings.json`, produced by `python -m eda.run_eda` in about 2.5 minutes, and the run is logged in `results/timing.csv`. The analysis used the training split only; pair statistics use the 1,231 test-like maps (≤ 40 propositions). Each number below feeds one of the proposed decisions D018–D024.

### D018 — Pruning window k = 8 turns, with distance also given to the model
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Proposed**
- k = 8 keeps **98.2%** of related pairs (98% was the plan's target) and drops 27% of pairs. Per type, the 98% window is Inference 6, Conflict 8, Rephrase 10. Propositions with no position (2.7%) are always kept as candidates.
- *Why distance is also an input:* neighbouring pairs are related 43% of the time, against 7.8% overall. Pruning alone leaves most of the imbalance (10.5% positives at k = 8).
- *Alternatives:* k = 5 (95.7% of relations, 52% of pairs) as a speed ablation, and no pruning.

### D019 — Context = the whole map's dialogue, max length 1,024 (supersedes the "dialogue window")
- **2026-09-30 · Phase 2 → A1.1/A1.2 · by Claude · Proposed**
- The pair plus the *entire* dialogue is ≤ 837 tokens (p99 621) in the test-like maps, and whole test maps are ≤ 953 tokens. The model sees everything, which is the direct fix for 2024's no-context failure.
- *Cost:* about 1.8× the tokens of ±2 turns, roughly 30 vs 17 min per epoch for base (to be measured in A1.2). The ±2-turn window becomes an ablation.

### D020 — Label scheme: 5 classes; Rephrase/Conflict direction from dialogue order
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Proposed**
- Rephrase and Conflict run later → earlier in 97.8% and 97.7% of cases, so their direction comes from the turn order at a cost of about 2%. When both propositions are in the same turn (1.2% of Rephrases), node order breaks the tie. **Inference is 53% later→earlier and 47% earlier→later, so the model predicts its direction** (A→B Inference and B→A Inference are separate classes).
- Only 12 pairs carry two relations, so one label per pair is fine.

### D021 — Input features: distance and same-speaker marked explicitly
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Proposed**
- Same-speaker rates are Inference 94.5%, Rephrase 81.7% and Conflict 56.1%, against a 66.2% base rate among candidates. Distance: see D018. Both go into the input as short marker tokens; the "−speaker/TA features" ablation removes them.

### D022 — The 19 giant training maps (> 40 propositions) are an open question for A1.1
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Open**
- They hold 60% of all pairs but only 12.8% of relations, and no test map is that large (max 35). Options: train only on their pruned candidates (k = 8 already cuts them sharply), cap them, or drop them. A1.1 decides once pair counts after pruning are known.

### D023 — ILO rules (Phase 3): question words and "xxx", not "?"; similarity-first alignment
- **2026-09-30 · Phase 2 → Phase 3 · by Claude · Superseded by D025.** There are no ILO rules any more. The model learns these cues. The similarity finding survives as D025's candidate filter and baseline.
- Only 7–8% of question turns contain "?", because the transcripts have almost none. A turn starting with a question word marks 74.5% of Pure Questioning, 67.9% of Rhetorical and 41.8% of Assertive Questioning, against 4.0% of Asserting. The proposition contains "xxx" in 45.7% of Pure Questioning, against 0.4% of Asserting.
- Alignment: text similarity alone picks the gold proposition for 94.2% of turns. Links are one-to-one (98.8% / 99.8%), and 20% of turns express no proposition.
- *Corrects* the "?" cue assumed in `PLAN.md`'s ILO section and `CLAUDE.md` §6.

### D024 — Unscored transition illocutions in the output maps
- **2026-09-30 · Phase 2 → A1.5 · by Claude · Proposed**
- Inference → Arguing (97.9%), Rephrase → Restating (85.9%; Default Illocuting 13.7%), Conflict → Disagreeing (94.3%). These are used when assembling valid AIF. They aren't scored, so the majority choice is enough.

### D025 — Phase 3 is a trained ILO labeller only (no rules)
- **2026-09-30 · Phase 3 · by user (approach) / Claude (design) · Approved by user 2026-10-01 (go-ahead for 3.1–3.2, then review)**
- *User decision:* ILO type labelling is learned, not rule-based. This supersedes the rules in PLAN's ILO section and D023. `PLAN.md`, `CLAUDE.md` §6–7 and `a1_cross_encoder/README.md` are updated to match.
- *Why:* ILO is half the Global score. "?" is almost absent (EDA §3). Macro-F1 weighs the rare types (Assertive / Rhetorical Questioning, Agreeing, Challenging) as heavily as Asserting. The rare-label trap also applies here (one invented type drops a map to 72.8), so the model must produce probabilities that can be thresholded. All top 2024 ILO systems were trained.
- *Design* (code in a top-level `ilo/`, shared by all approaches; the fine-tuning loop goes in `common/finetune.py` so A1 reuses it in A1.2):

  | Step | What | Check before moving on |
  |---|---|---|
  | 3.1 Candidates | Pair each dialogue turn with its top-K most similar propositions; choose the smallest K keeping ≥ 99.5% of gold links (measured on train) | Recall at K = 1…5 |
  | 3.2 Dataset | Input `[CLS] Speaker: turn [SEP] proposition [SEP] ±2 neighbouring turns`; 11 labels = None + the 10 L→I types seen in training | Label counts, token lengths, samples read by eye |
  | 3.3 Train | ModernBERT-base, bf16, flex attention, class-weighted cross-entropy, early stopping on dev ILO macro-F1 (per map, full decode) | First overfits 64 examples; VRAM and epoch time logged |
  | 3.4 Decode | Temperature scaling; one-to-one matching on 1 − P(None) above a link threshold; per-type thresholds (fallback Asserting), all tuned on dev | Thresholded ≥ argmax on dev |
  | 3.5 Evaluate | Dev maps in test-input format (D004); thresholds cross-fitted across dev episodes, so the reported number isn't flattered by tuning | Linking accuracy vs the similarity-only baseline on dev; ILO F1 general / focused and per type; timing |
- *Estimated cost:* about 65k candidate pairs per epoch × ~150 tokens ≈ 10M tokens, so roughly 10 min/epoch on ModernBERT-base, and 30–50 min for 3–5 epochs. No API spend.
- *This is the project's first training run and first dev scoring*, both of which need the user's go-ahead under the working rules.
- *Later ablations (A1.6):* the whole map as context instead of ±2 turns, argmax instead of thresholds, no one-to-one decoding.

### D026 — The repository is under git
- **2026-09-30 · Setup · by user · Approved · Done** (baseline commit `7462dc3`)
- *What:* `git init` on `main`, with a baseline commit of the Phase 0–2 state (26 files).
  - `.gitignore` leaves out `data/` (the re-downloadable DialAM release; `data/evaluation/` holds the test gold and must never enter history), Python caches and venvs, model checkpoint files (`*.pt`, `*.safetensors`, …) and `.claude/settings.local.json`.
  - `.gitattributes` stores text with LF line endings, so Windows git and WSL git see identical files, and treats PDFs as binary.
- *Why:* every change becomes reviewable and reversible, and each result can be tied to the exact code that produced it.
- *Deferred by the user:* whether to track `splits/`, and any change to the folder layout. Both wait for the project sweep and its proposals.
- *Mechanics:* commits made from Claude Science pass the author per command (`git -c user.name=… -c user.email=… commit`), because its sandbox keeps `.git/config` read-only. Commits from a normal terminal use the user's global git identity.

### D027 — Project sweep before Phase 3
- **2026-10-01 · Setup · proposals P1–P7 by Claude Science, X1–X3 by Claude · Approved by user 2026-10-01 · Done**
- **P1, track `splits/`:** already done by the user in `bfb6718`.
  - **X1:** `python -m common.check_foundation` used to *rewrite* `splits/` on every run. Now the split is only recomputed and compared: `python -m common.split` checks, and `--write` regenerates (which shows up as a git diff).
  - A new gate, `split_matches_tracked_files`, brings the foundation check to **13/13 passing**, with the split unchanged.
- **P2, decision log moved to the root** with `git mv` (`git log --follow decisionlog.md` keeps its history) and retitled "Decision log". 6 path references updated: `CLAUDE.md` ×2, `PLAN.md`, the A1 README, `eda/README.md`, `requirements.txt`.
- **P3, one numbering scheme:** shared phases are 0–3 and A1's are A1.1–A1.7 (old 4→A1.1, 5→A1.2, 6→A1.3, 7→A1.4, 8→A1.5, 9→A1.6, 10→A1.7). A2–A4 get their own numbers when they start. `PLAN.md` sections are now named, not numbered, and all phase references in the log, READMEs and docstrings were renamed.
- **P4, the phase map is the single status record.** The A1 README's "Status: plan only" now points here. The Project Context "Current state" in Claude Science lives outside the repo and has to be changed there.
- **P5 / X2, stale text corrected:**
  - "pruning removes most None pairs" (`CLAUDE.md` §7, `PLAN.md` A1): it removes about a quarter at k = 8;
  - "analyst copies are linked by Analysing" (`CLAUDE.md` §5): only about 2% are;
  - "far fewer" in the A1 README's pruning figure;
  - the log's environment row (`.venv` → WSL venv) and A1.2 row (now uses `common/finetune.py`);
  - `PLAN.md`'s "ILO is close to solved".
  - The "dialogue window" wording is left as it is until D019 is decided.
- **P6, complete status legend:** Done, Finding and Open added.
- **P7, `cache/` ignored** before Phase 3, because Phase 3 already writes tokenised caches.
- **X3, line endings:** 8 working files had CRLF, because Windows Python writes CRLF by default; they are normalised to LF. Edits now write LF explicitly.

### D028 — Phase 3.1: ILO candidates = own top-5 + previous turn's top-5 + next turn's top-3
- **2026-10-01 · Phase 3 · by Claude · Done** (within the approved 3.1 scope)
- *Measured on the train split* (`python -m ilo.candidates`): similarity alone (difflib ratio) keeps 94.3% of the 17,509 gold links at K = 1, 97.5% at K = 3 and 98.8% at K = 6. No K reached the 99.5% target.
- *Why it misses:* 371 of the 433 misses at K = 3 are short turns whose proposition was rebuilt from context ("Yes" becomes the claim agreed with; "it stinks" becomes what "it" refers to). These are mostly Agreeing, Disagreeing and question types, which weigh most in macro-F1.
- *Fix:* a turn also borrows the candidates of its neighbouring turns. **Own 5 + previous 5 + next 3 keeps 99.59%**, with 205,955 pairs (8.5% positive). Every type is ≥ 98.3% except Agreeing (93.7%, 89/95) and Disagreeing (94.7%, 18/19).
- *Alternatives measured:*

  | Scheme (own / prev / next) | Links kept | Pairs |
  |---|---|---|
  | 3 / 2 / 0 | 98.8% | 97k |
  | 3 / 3 / 0 | 99.0% | 113k |
  | 4 / 3 / 2 | 99.3% | 155k |

- *Cost:* about 2× the pairs of own-5 alone (≈ 31M tokens per epoch, roughly 30 min). Training on a subsample of the None pairs is the fallback if that proves too slow.

### D029 — No node-ID or node-order shortcuts, in any approach
- **2026-10-01 · All approaches · by Claude · Proposed**
- *Finding:* in 74% of gold links, the proposition's node ID is exactly the turn's ID + 2, because the annotation tool creates L, YA and I in sequence. Test inputs keep these IDs, so a model could link by ID instead of by meaning.
- *Proposal:* never use node IDs, or the order of nodes in the JSON, as a signal. Use only text, speakers and the TA dialogue order. Results then measure language understanding, transfer to unannotated debates, and stay comparable across A1–A4.
- *Cost:* linking can't exploit the artefact. Expect a few points of linking accuracy left on the table compared with a system that does.
- *One existing use to decide on:* `MapView.dialogue` orders the separate stretches of a map's TA chain (27% of maps have more than one) by node ID, because nothing else orders them. This never links a turn to a proposition, but it sets distances *across* stretches (EDA, A1). Options: keep it as a documented exception, or treat stretches as unordered.

### D030 — Maps without an illocution layer are left out of ILO training and ILO dev scoring
- **2026-10-01 · Phase 3 · by Claude · Done** (within the approved 3.2 scope; easy to reverse)
- *Finding:* 1,614 "hard negatives" (a turn nearly identical to a proposition, yet no gold link) are not noise spread across the data. **19 train maps and 3 dev maps have no L→I links at all** (0 linked turns among 140 to 400 turns each), and they hold 1,553 of them. These are the late re-annotation batches (nodesets 24807 to 25475), which are **the same giant maps as in D022**. They carry relations but were never annotated for illocutions.
- *Rule:* a map with no L→I link at all is skipped by `ilo/data.py`. The maps stay in for ARI (A1), and the test maps are fully annotated.
- *Effect:* train goes from 205,973 to 158,864 examples, dev from 36,112 to 29,506; no labelled example is lost. ILO dev scoring covers 225 of 228 dev maps.

### D031 — Phase 3.2: the ILO dataset
- **2026-10-01 · Phase 3 · by Claude · Done**
- *Input:* `text_a` = "Speaker: turn"; `text_b` = "proposition [SEP] before: (up to 2 turns) [SEP] after: (up to 2 turns)". There are 11 labels: None plus the 10 L→I types seen in training. Dev inputs are built from stripped maps (D004).
- *Size:* train 1,231 maps, 158,864 examples (11.0% links); dev 225 maps, 29,506 examples (10.9%). Tokens: median 108, p99 186, max 300, so **max length 320** truncates nothing.
- *Rare types are very rare.* Train has Default Illocuting 123, Agreeing 89, Challenging 38, Disagreeing 18, Arguing 10 and Restating 4. Dev has 7, 11, 2, 2, 2 and 1. Per-type thresholds cannot be tuned reliably on dev for these. See the open question for 3.4.
- *Files:* `cache/ilo/{train,dev}.jsonl` (not in git, rebuilt by `python -m ilo.data` in about 4 min); stats in `results/ilo/dataset.json`.

### D032 — Phase 3.3 setup: training loop, sanity checks, attention kernel
- **2026-10-01 · Phase 3 · by Claude · Done**
- *Training loop* (`common/finetune.py`, reusable in A1.2): length-grouped batches, bf16 autocast with fp32 weights, weighted cross-entropy, fused AdamW (lr 5e-5, weight decay 0.01, 6% warm-up, linear decay), gradient clip 1.0, batch 32, max 320 tokens, up to 3 epochs. Dev is evaluated twice per epoch; training stops after 2 evaluations without improvement. The best checkpoint and its dev probabilities go to `checkpoints/ilo/`, which is git-ignored.
- *Class weights:* (most frequent / count)^0.5, capped at 30. None 1.0, Asserting 3.0, Pure Questioning 11.9, Assertive / Rhetorical about 26–27, and the six rare types 30.
- *Early-stopping metric:* dev ILO general F1 per map with the simple decode in `ilo/decode.py`: one-to-one Hungarian matching on 1 − P(None) ≥ 0.5, then the most probable type. Thresholds come in 3.4.
- *Sanity checks:*
  - Decoder with gold labels as input: **ILO 98.63 general / 94.73 focused**, linking recall 99.16%. This is the ceiling of this design on dev (candidate misses, plus turns that express two propositions).
  - All None: 38.21 / 0, the majority baseline on dev.
  - The model memorises 72 examples (100% train accuracy) in 100 steps: 10.6 s, 15.3k tokens/s, 3.43 GiB peak.
- *Attention kernel:* on real variable-length batches **SDPA runs at 14,184 tokens/s, but flex_attention only at 2,749**, because it recompiles for every new shape. This supersedes D010's "flex as working default", which held only for fixed shapes. Training uses SDPA.

### D033 — Batches are capped by tokens, not by sequence count
- **2026-10-01 · Phase 3 · by Claude · Done**
- *What happened:* the first full training run was **stopped at step 600 (0.12 epoch)** by the background-tool time limit (10 min). Its log, `results/ilo/train_log_run1_killed.jsonl`, also showed a real problem: throughput fell from 4.9k to 3.7k tokens/s, and peak VRAM reached 7.72 of 7.9 GiB. Fixed 32-sequence batches of the longest examples (about 9.6k tokens), on top of about 2.4 GB of weights and optimizer state, overflowed the card, and the driver spilled into system RAM.
- *Fix* (`common/finetune.py`):
  - Batches are packed under a padded-token budget: **4,096 tokens** for training (at most 64 sequences) and 16,384 for prediction.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` limits fragmentation from variable shapes.
- *Measured* (`python -m ilo.train --bench`, 400 steps): **26.3k tokens/s steady and 4.33 GiB peak**; dev prediction runs at 587 examples/s with a 1.25 GiB peak. One epoch is about 11–15 min, so a full run takes about 45–50 min including dev evaluations.
- *For A1:* the same budget applies. Longer A1 inputs mean fewer pairs per batch, not more memory.

### D034 — Phase 3.3 result: training stopped early at 1.5 epochs; best checkpoint at 0.5 epoch
- **2026-10-01 · Phase 3 · by Claude · Finding**
- *Launch:* detached as a hidden Windows process (user's choice) so it wasn't bound by the 10-minute tool limit. `scratchpad/run_ilo_train.sh` was monitored through `results/ilo/train_log.jsonl`.
- *Dev ILO general / focused F1 (simple decode)* by evaluation: 0.5 epoch **89.46 / 78.06** (best), 1.0 epoch 86.41 / 72.31, 1.5 epoch 88.05 / 73.18. Patience 2 stopped the run.
- *Cost:* 19.5 min in total (13.0 min per epoch, including evaluations), 22.5k tokens/s, 4.35 GiB peak. Best checkpoint in `checkpoints/ilo/` (598 MB, git-ignored).
- *Caveat:* the best checkpoint is very early (0.5 epoch, while the learning rate was still near its peak). It never predicts the six rare types on dev (see D036). Later checkpoints may have learned them but scored lower under the simple decode, and they weren't kept.

### D035 — Phase 3.4 result: decision layer tuned on dev; the rare-type grouping question is moot
- **2026-10-01 · Phase 3 · by Claude · Done**
- `ilo/tune.py` fits a temperature, then thresholds by coordinate ascent on mean per-map ILO general F1. The fast metric equals the official scorer on all 225 dev maps (asserted).
- *Result:* **T = 1.05** (already well calibrated) and **link_min = 0.10** (4 of 5 folds pick 0.10, one picks 0.05). The first run's grid floor of 0.20 was too high and was widened to 0.05. P(type | link) minimum: Pure Questioning 0.5, Assertive Questioning 0.5, Rhetorical Questioning 0.3, and **0 for all six rare types**.
- *Scores:* simple decode 89.46 / 78.06; tuned on all of dev 90.16 / 80.67 (optimistic); **leave-one-episode-out 89.89 / 80.38 (honest)**.
- *Grouping question:* "shared" and "per-type" thresholds give **identical results**, because the data set every rare-type threshold to 0. No user decision is needed. Files: `results/ilo/thresholds_{shared,per-type}.json`.

### D036 — Phase 3.5 result: gate passed; weak on rare and assertive/rhetorical types
- **2026-10-01 · Phase 3 · by Claude · Finding** (`python -m ilo.evaluate`, `results/ilo/dev_report.json`; 225 dev maps in test-input format)

  | Dev ILO F1 (general / focused) | |
  |---|---|
  | Majority (no illocutions) | 38.21 / 0.00 |
  | Text similarity, one-to-one, all Asserting (threshold tuned) | 75.99 / 59.38 |
  | Model, simple decode | 89.46 / 78.06 |
  | **Model, tuned decision layer (cross-fitted)** | **89.89 / 80.38** |
  | *Ceiling of this design (gold labels decoded)* | 98.63 / 94.73 |

- **Gate passed:** the model links 98.27% of gold links at 99.62% precision, against 97.43% at 98.68% for text similarity with the same one-to-one matching, and about 94% for similarity top-1.
- *Per type:* Asserting 0.98 F1, Pure Questioning 0.87, Rhetorical Questioning 0.51 (14 predicted vs 29 gold), Assertive Questioning 0.40 (6 vs 19). **Default Illocuting, Agreeing, Challenging, Disagreeing, Arguing and Restating are never predicted** (26 gold links in dev). This is the main room left between 80.4 focused and the 94.7 ceiling.
- *Inference:* 0.21 s per map (48 s for all of dev). A reloaded checkpoint reproduces the saved probabilities exactly.
- *Reference only (different maps):* 2024's best ILO on the test set was 81.2 / 72.1.
- *Possible improvements:* select checkpoints by the *tuned* metric and keep every evaluation's probabilities; lower the learning rate or class-weight cap; give rare types more epochs. These are proposed for a Phase 3 iteration, or as A1.6 ablations, at the user's call.

### D037 — Findings reports live in `reports/`, one per phase
- **2026-10-01 · Setup · requested by user · Done**
- `reports/README.md` is the index: one line per phase, with the headline and a link. `reports/03-ilo-labeller.md` is the Phase 3 report. Phase 2's report stays at `eda/README.md`, where it is already linked; Phases 0–1 are covered by their decision-log entries.
- *Convention:* `NN-short-name.md`, shaped as at a glance → method → results → lessons → open items → reproduce. Every number comes from a committed file under `results/` or `eda/`. These reports are the source for the final write-up (now noted in `PLAN.md` and `CLAUDE.md` §5).
- *Added for the report:* `python -m ilo.evaluate --headroom` (`results/ilo/headroom.json`). Correcting rare and Assertive/Rhetorical types to gold would lift dev ILO from 90.16 / 80.67 to **97.07 / 90.49**; perfect typing of correct links reaches 97.87 / 90.61. Typing, not linking, holds the remaining gap.
