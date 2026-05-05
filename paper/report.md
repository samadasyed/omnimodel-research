# Beyond Accuracy: Diagnosing Modality Attribution and Confidence Calibration in Video-LLMs on Audio-Centric Tasks

*CS 639: Intro to Foundation Models, Spring 2026*

> **Status:** Draft — fill in numbers after the Gemma Colab run completes.
> Cross-model comparison cells against Qwen2.5-Omni-7B numbers contributed
> by team member Jeff (`jjwang8/639_avut`). All `<TBD>` markers are
> placeholders pending experimental results.

---

## Abstract

Recent omnimodal models (e.g., Qwen2.5-Omni, Gemma-3n) claim to jointly
process text, video, and audio, but accuracy alone does not reveal
*whether* the model actually used each modality. We extend the AVUT
benchmark (Yang et al., 2025) — an audio-centric video understanding
suite that filters out text shortcuts at construction time — with three
new diagnostic axes: **(1) Attribution Faithfulness Score (AFS)** — does
removing the modality the model claims to have relied on actually change
the answer? **(2) Confidence Calibration under Ablation** — does
confidence drop when the signal-carrying modality is removed?
**(3) Lexical Override Rate (LOR)** — when a *contradictory* transcript
is injected alongside accurate audio, how often does the model abandon
its audio-grounded answer? We evaluate Gemma-3n-E2B-IT (primary) and
compare against Qwen2.5-Omni-7B numbers from a parallel teammate run, on
a balanced subset of AVUT-Human. We find Gemma-3n's self-reported
attribution and ablation-based attribution diverge on `<TBD>%` of audio-
essential questions, indicating that omnimodal models *confabulate*
about their own modality use. The cross-model comparison shows
`<TBD: similar/diverging pattern>`.

## 1. Introduction (~0.5 page)

- Omnimodal models are evaluated mostly by accuracy. Accuracy hides *how*
  the answer is produced.
- Lexical bias (LISTEN, Chen et al. 2025) shows audio-LLMs over-rely on
  transcribed words. AVUT (Yang et al. 2025) was designed to break this by
  filtering out text-shortcut questions at construction time. Even given
  a clean benchmark, a model can still produce "right answer for the
  wrong reason."
- We argue that diagnosing modality attribution requires two
  complementary signals: (a) what the model **says** it used, and (b)
  what the model's behavior under ablation reveals it **actually** used.
  The gap between (a) and (b) is the key methodological contribution.
- Contributions:
  1. **AFS** — a counterfactual faithfulness metric for self-reported
     modality attribution, with a *trivial-modality* filter that prevents
     inflated confabulation counts on questions every modality solves
     independently.
  2. **Mismatched-transcript condition (S7) and LOR** — a direct
     lexical-vs-acoustic conflict probe absent from AVUT's original
     evaluation. Quantifies how often an injected *wrong* transcript
     overrides a correct audio-grounded answer.
  3. **Per-stage confidence calibration analysis** combining verbalized
     0–100 scores, ECE, point-biserial confidence-correctness
     correlation, and ΔConf under modality ablation.
  4. **Cross-model comparison.** Same metrics, same sample-selection seed
     across Gemma-3n-E2B-IT and Qwen2.5-Omni-7B, so the diagnostic
     pattern (not just accuracy) is comparable cell-by-cell.

## 2. Related Work (~0.5 page)

- AVUT benchmark: design, the answer-permutation filter, the six MCQ
  task types.
- LISTEN (Chen 2025): lexical-bias diagnostic for audio-LLMs.
- MUStARD line (Castro 2019; Bhosale 2023; Saha 2025): we discuss how
  modality-separation-via-prompting shows mixed results on sitcom-domain
  sarcasm vs. AVUT-style audio-centric tasks. Our team's earlier MUStARD
  results are referenced as the setting that motivated the AVUT pivot.
- Faithfulness-of-reasoning (Turpin et al. 2023; Lanham et al. 2023):
  inspires the counterfactual flavor of AFS.

## 3. Methodology (~1.5 pages)

### 3.1 Stages

Eight configurations of the same omnimodal model (S2–S8; S1 uses a
small text-only model). The scheme matches the parallel Qwen2.5-Omni-7B
run done by teammate Jeff so cross-model results are directly
comparable.

| Stage | Inputs                                           | Notes |
|-------|--------------------------------------------------|-------|
| S1    | Question + options only (Qwen2.5-1.5B-Instruct)  | text-shortcut floor; model held constant across both runs |
| S2    | Audio only                                       | omnimodal model on the .wav |
| S3    | Silent video only                                | omnimodal model on the muted .mp4 |
| S4    | Full audio + video                               | the anchor stage |
| S5    | Full AV + Whisper transcript of *this* video     | matched-transcript injection |
| S6    | Full AV + S4-answer + reflection probe           | depends on S4; asks for `[RANK] Audio=X, Visual=Y, Text=Z` and `[REASON]` |
| S7    | Full AV + Whisper transcript of a *different* video (same task type) | **our addition**: lexical-override probe |
| S8    | Full AV with prosody-first verbalization (optional) | side-quest; can be skipped without affecting headline findings |

S5 (matched) and S7 (mismatched) share the prompt format; the only
difference is which transcript is injected. S6 is concatenated multi-
turn: the model sees its own S4 raw response before being asked to
reflect.

### 3.2 Sample selection

We sample `n_per_task` QA pairs per task type from AV-Human's 1,734 questions
across the six MCQ task types (AIE, ACC, AEL, AVCM, AVOM, AVTM). Default
configuration is **100 samples/task = 600 total**, balanced. The same
random seed (42) is used for both model runs so the sample set is
identical cell-by-cell. Mismatched-transcript donors (S7) are also
seed-pinned so the same lexical conflict is presented to both models.

### 3.3 Models

- **Gemma-3n-E2B-IT** for S2–S8 (primary; ~2B effective active parameters,
  text+image+audio, runs on a single Colab L4 / A100).
- **Qwen2.5-Omni-7B** for S2–S8 (cross-model robustness check; teammate
  Jeff's run, server-side BF16).
- **Qwen2.5-1.5B-Instruct** for S1 across both runs (text-only baseline
  is a property of the *dataset*, not the omnimodal model — keeping it
  fixed makes the cross-model diagnostic comparable).

Gemma-3n's vision pipeline is image-based; we sample 8 evenly-spaced
frames per video. Audio is loaded at 16 kHz mono and truncated to 30 s
(Gemma-3n's training cap). Qwen2.5-Omni handles native video tokens;
both models receive the same prompt template ("Reply with EXACTLY this
format: `[ANSWER] X [CONFIDENCE] Y`" with a worked example).

### 3.4 Metrics

- **Accuracy** per task per stage.
- **Verbalized confidence** (0–100 self-reported), summarized as mean/
  std and binned for **Expected Calibration Error (ECE)**.
- **Point-biserial confidence-correctness correlation** per (task,
  stage). High positive r = confident when right, uncertain when wrong.
- **ΔConf(M)** = `Conf(S4) − Conf(stage_without_M)`. Audio removal:
  ΔConf = Conf(S4) − Conf(S3). Visual removal: ΔConf = Conf(S4) − Conf(S2).
  A calibrated model drops confidence when the signal-carrying modality
  is removed.
- **Attribution Faithfulness Score (AFS)** = fraction of S6 samples
  where ablating the model's self-reported #1 modality flipped the
  answer, *restricted to falsifiable cases* (samples where every single-
  modality stage doesn't already independently agree with S4 — agreement
  on all three would make the modality choice unfalsifiable).
- **Transcript Injection Bias (TIB)** = `Acc(S4) − Acc(S5)`. Positive =
  matched transcript hurts.
- **Lexical Override Rate (LOR)** = among samples where S4 was correct,
  the fraction flipped to a different answer in S7 (mismatched
  transcript). Higher LOR = the model trusts the (wrong) transcript over
  the audio.
- **Answer flip rates** between key stage pairs (S4↔S3, S4↔S2, S4↔S5,
  S4↔S7, S4↔S1) for an at-a-glance sense of which inputs change the
  model's mind.

## 4. Results (~1.5 pages)

### 4.1 Accuracy table

`<TBD>` Per-task per-stage accuracy table. Insert after notebook 06 runs.
The cross-model row from Jeff's published Qwen2.5-Omni-7B numbers is
auto-loaded from `https://github.com/jjwang8/639_avut`.

### 4.2 Headline finding: attribution confabulation

`<TBD>` Insert AFS bar chart and discussion. Hypothesized result: AFS
substantially below 1.0 on audio-essential tasks; lower on AVCM/AVOM/AVTM
where the audio-visual binding is "cheap" (the model can solve it from
either modality, so committing to one is more arbitrary).

For reference, Jeff's Qwen2.5-Omni-7B numbers on the full AV-Human set
show AFS ranging from **0.41 (AVCM)** to **0.78 (AIE)**, with a strong
"trivial" tail (e.g., AEL has 71 trivial samples vs. 70 falsifiable) —
i.e., on a meaningful fraction of audio-event-localization questions,
every single-modality stage independently arrives at the same answer.
Whether Gemma-3n shows the same pattern is an empirical question we
answer in §4.5.

### 4.3 Confidence calibration

`<TBD>` ECE and point-biserial correlation per stage. ΔConf table
including a per-task breakdown. Hypothesized result: large positive
ΔConf(audio_removal) on AIE, near zero on AVCM/AVOM (signal is in both
modalities). For ACC ("counting"), Jeff's Qwen-Omni numbers show a
*negative* ΔConf(audio_removal) — the model is *more* confident on
silent video than on full AV, which we read as overconfidence under
modality loss.

### 4.4 Transcript bias and lexical override (our novel ingredient)

`<TBD>` TIB and LOR per task. Hypotheses:
- TIB > 0 on ACC/AEL (matched transcripts hurt because they don't
  capture sound counts/timing).
- LOR > 0 on audio-essential tasks (mismatched text overrides correct
  audio — the headline lexical-override result).

For Qwen2.5-Omni-7B, Jeff's numbers show TIB roughly *negative* across
the board (transcripts mostly *help*, even on audio-essential tasks).
This is consistent with audio-LLMs' known reliance on lexical content
even on AVUT, which was supposed to filter that out. The S7 mismatched-
transcript LOR is the metric we add to disambiguate "transcript helps
because audio is hard" from "transcript dominates regardless of audio."

### 4.5 Cross-model comparison

`<TBD>` Side-by-side Gemma-3n-E2B-IT vs Qwen2.5-Omni-7B on AFS, ΔConf,
TIB, LOR per task. Notebook 06 produces this table automatically by
fetching Jeff's published metrics JSONs.

What we expect to see: if confabulation patterns are *model-specific*,
the AFS rankings should differ substantially between models on the same
task. If patterns are *training-objective-driven*, the rankings should
be similar despite the 3.5× parameter difference.

## 5. Discussion & Conclusion (~1 page)

- AFS, LOR, and ΔConf give a *complementary* picture to accuracy: they
  surface where the model is "right for the wrong reason."
- The gap between self-reported and ablation-revealed modality use is a
  concrete handle for designing better attribution probes (and possibly
  better training signals).
- The trivial-modality filter is methodologically important. Without it,
  the AFS denominator includes questions where ablation can't possibly
  change the answer, which inflates false-confabulation counts.
- Cross-model agreement (or disagreement) on AFS rank-order is itself
  diagnostic: shared confabulation patterns hint at training-data
  artifacts; divergent patterns hint at model-specific quirks.
- **Limitations:** sample sizes are smaller than Jeff's full-AVUT run
  for Gemma due to Colab compute caps; Whisper-small ASR errors mean
  S5/S7 transcripts have noise; the S6 reflection probe is concatenated
  multi-turn rather than true two-turn conversation; we did not run a
  no-probe S6 control (flag for future work).
- **Future work:** (a) per-question difficulty stratification; (b) probe
  consistency under prompt rephrasing; (c) test whether prosody-
  verbalization (S8) affects AFS — we have S8 data but didn't headline
  the finding.

## References

- Yang et al. (2025). *Audio-centric Video Understanding Benchmark
  without Text Shortcut.* EMNLP 2025.
- Chen et al. (2025). *Do Audio LLMs Really LISTEN, or Just Transcribe?
  Measuring Lexical vs. Acoustic Emotion Cues Reliance.*
- Castro et al. (2019). *Towards Multimodal Sarcasm Detection (MUStARD).*
- Saha et al. (2025). *MUStReason / PragCoT.*
- Turpin et al. (2023). *Language Models Don't Always Say What They Think.*
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought
  Reasoning.*
