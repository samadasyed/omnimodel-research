# Beyond Accuracy: Diagnosing Modality Attribution and Confidence Calibration in Video-LLMs on Audio-Centric Tasks

*CS 639: Intro to Foundation Models, Spring 2026*

> **Status:** Draft — fill in numbers after the Colab run completes. All `<TBD>` markers are placeholders pending experimental results.

---

## Abstract

Recent omnimodal models (e.g., Qwen2.5-Omni, Gemma-3n) claim to jointly process text, video, and audio, but accuracy alone does not reveal *whether* the model actually used each modality. We extend the AVUT benchmark (Yang et al., 2025) — an audio-centric video understanding suite that filters out text shortcuts at construction time — with three new diagnostic axes: (1) **Modality Attribution Faithfulness (AFS)** — does removing the modality the model claims to have relied on actually change the answer? (2) **Confidence Calibration under Ablation** — does confidence drop when the signal-carrying modality is removed? (3) **Lexical Override Rate (LOR)** — when a contradictory transcript is injected alongside accurate audio, how often does the model abandon its audio-grounded answer? On a balanced 120-sample subset, we find Qwen2.5-Omni-7B's self-reported attribution and ablation-based attribution diverge on `<TBD>%` of audio-essential questions, indicating that omnimodal models *confabulate* about their own modality use. `<TBD: optional Gemma cross-model finding.>`

## 1. Introduction (~0.5 page)

- Omnimodal models are evaluated mostly by accuracy. Accuracy hides *how* the answer is produced.
- Lexical bias (LISTEN, Chen et al. 2025) shows audio-LLMs over-rely on transcribed words. AVUT (Yang et al. 2025) was designed to break this by filtering out text-shortcuts. Even given a clean benchmark, a model can still produce "right answer for the wrong reason."
- We argue that diagnosing modality attribution requires two complementary signals: (a) what the model **says** it used, and (b) what the model's behavior under ablation reveals it **actually** used. The gap between (a) and (b) is the key methodological contribution.
- Contributions:
  1. **AFS** — a counterfactual faithfulness metric for self-reported modality attribution.
  2. **Mismatched-transcript condition** — a direct lexical-vs-acoustic conflict probe absent from AVUT's original evaluation.
  3. **Per-task confidence calibration analysis** combining verbalized 0–100 scores and softmax over A/B/C/D.

## 2. Related Work (~0.5 page)

- AVUT benchmark: design, the answer-permutation filter, the six MCQ task types.
- LISTEN: lexical-bias diagnostic for audio-LLMs.
- MUStARD-line research (Castro 2019; Bhosale 2023; Saha 2025; SarcasmMiner 2026): we discuss how modality-separation-via-prompting shows mixed results on sitcom-domain sarcasm vs. AVUT-style audio-centric tasks.
- Faithfulness-of-reasoning literature (Turpin et al. 2023, Lanham et al. 2023): inspires the counterfactual flavor of AFS.

## 3. Methodology (~1.5 pages)

### 3.1 Stages

Six configurations of the same omnimodal model (S1–S5; S0 uses a smaller text-only model):

| Stage | Inputs |
|-------|--------|
| S0 | Question + options only (Qwen2.5-1.5B-Instruct) — text-shortcut floor |
| S1 | Audio only |
| S2 | Silent video only |
| S3 | Full audio-visual + inline attribution probe + verbalized confidence |
| S4 | Full AV + ASR transcript of *this* video |
| S5 | Full AV + ASR transcript of a *different same-task* video (mismatched) |

The S3 attribution probe is single-turn: the model emits answer + confidence + a 1-2-3 ranking over Audio/Visual/Text in one response.

### 3.2 Sample selection

We sample 20 QA pairs per task type from AV-Human's 1,734 questions, balanced across the six MCQ task types (AIE, ACC, AEL, AVCM, AVOM, AOCR). The 120-sample size was chosen to fit a single H100 Colab session per stage (~25 min/stage, ~2.5 hr total for the six stages).

### 3.3 Models

- **Qwen2.5-Omni-7B** for S1–S5 (primary).
- **Qwen2.5-1.5B-Instruct** for S0 (text baseline).
- **Gemma-3n-E2B-IT** as a cross-model robustness check (optional, smaller, faster).

### 3.4 Metrics

- **Accuracy** per task per stage.
- **Verbalized confidence** (0-100 self-reported) and **softmax-over-choices logprob** (free at generation time, no extra forward pass beyond the existing scoring step).
- **Expected Calibration Error (ECE)** computed from logprob confidence over 10 bins.
- **ΔConf(M)** = Conf(S3) - Conf(stage_without_M). A calibrated model drops confidence when the signal-carrying modality is removed.
- **Attribution Faithfulness Score (AFS)** = fraction of S3 samples where ablating the model's self-reported #1 modality flipped the answer.
- **Transcript Injection Bias (TIB)** = Acc(S3) - Acc(S4). Positive = matched transcript hurts accuracy.
- **Lexical Override Rate (LOR)** = fraction of S3-correct samples flipped to a different answer in S5 (mismatched transcript).

## 4. Results (~1.5 pages)

### 4.1 Accuracy table

`<TBD>` Per-task per-stage accuracy table. Insert after running notebook 06.

### 4.2 Headline finding: attribution confabulation

`<TBD>` Insert AFS bar chart and discussion. Hypothesized result: AFS substantially below 1.0 on audio-essential tasks; lower on AVCM/AVOM where binding is "cheaper" than committing to one modality.

### 4.3 Confidence calibration

`<TBD>` ECE per stage; ΔConf table; figure showing verbalized vs. logprob calibration.

### 4.4 Transcript bias

`<TBD>` TIB and LOR per task. Hypothesis: TIB > 0 on ACC/AEL (transcripts hurt because they don't capture sound counts/timing); LOR > 0 on audio-essential tasks (mismatched text overrides correct audio).

### 4.5 Cross-model comparison (if Gemma-3n run completes)

`<TBD>` Side-by-side AFS / ΔConf / LOR for Qwen2.5-Omni vs. Gemma-3n.

## 5. Discussion & Conclusion (~1 page)

- AFS, LOR, and ΔConf give a *complementary* picture to accuracy: they surface where the model is "right for the wrong reason."
- The gap between self-reported and ablation-revealed modality use is a concrete handle for designing better attribution probes (and possibly better training signals).
- Limitations: 120 samples is small; Whisper-small ASR errors mean S4/S5 transcripts have noise; the inline attribution probe may bias the answer itself (we didn't run a no-probe S3 control — flag for future work).
- Future work: (a) per-question difficulty stratification; (b) probe consistency under prompt rephrasing; (c) test whether prosody-verbalization (the side-quest stage) affects AFS.

## References

- Yang et al. (2025). Audio-centric Video Understanding Benchmark without Text Shortcut. EMNLP 2025.
- Chen et al. (2025). Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance.
- Castro et al. (2019). Towards Multimodal Sarcasm Detection (MUStARD).
- Saha et al. (2025). MUStReason / PragCoT.
- Turpin, J. et al. (2023). Language Models Don't Always Say What They Think.
- Lanham, T. et al. (2023). Measuring Faithfulness in Chain-of-Thought Reasoning.
