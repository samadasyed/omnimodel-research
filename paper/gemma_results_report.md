# Gemma-3n-E2B-IT on the AVUT Diagnostic Pipeline — Standalone Results Report

**Author:** Samad Syed (CS 639, Spring 2026)
**Date:** 2026-05-07
**Sample:** 600 balanced AVUT-Human questions (100 per task × 6 tasks), seed=42
**Compute:** Single A100-40GB on Colab Pro+, ~5 hours wall-clock for the full 7-stage run

> Companion run: teammate Jeff produced the Qwen2.5-Omni-7B numbers on the same pipeline at `jjwang8/639_avut`. All cross-model cells in this report are pulled directly from his published metric JSONs and are bit-identical comparable to ours.

---

## 1. What we ran

We extended the AVUT benchmark (Yang et al., 2025) with three diagnostic axes that go beyond raw accuracy. Each AVUT question was presented to Gemma-3n-E2B-IT in **seven different input configurations** ("stages"), so the model's behavior under controlled ablations could be measured directly:

| Stage | Inputs | Purpose |
|---|---|---|
| S1 | Question + options only (Qwen2.5-1.5B-Instruct, *not* Gemma) | Text-shortcut floor |
| S2 | Audio (16 kHz mono, ≤30 s) + question | Audio-only |
| S3 | 8 sampled video frames + question (no audio) | Visual-only |
| S4 | 8 frames + audio + question | Reference full-AV condition |
| S5 | S4 + Whisper-small transcript of *this* video | Matched transcript injection |
| S6 | S4 + the S4 raw response, then "Which modality did you use?" | Self-reported attribution |
| S7 | S4 + Whisper transcript from a *different* same-task video | **Lexical override probe (our novel addition)** |

The same 600 questions, the same Whisper transcripts, and the same mismatched-transcript pairings are used for the Qwen-Omni run, so every cross-model comparison is on identical inputs.

Three diagnostic metrics combine the stages:

- **Attribution Faithfulness Score (AFS):** Among S6 cases where the model's self-reported #1 modality is *falsifiable* (i.e., not every single-modality stage already gives the same answer), what fraction had its answer actually flip when that modality was ablated? AFS = 1.0 means perfect attribution; AFS = 0.5 means coin flip.
- **Lexical Override Rate (LOR):** Of the questions Gemma got *right* at S4, what fraction flip to a different answer at S7 (when handed a contradictory transcript)? LOR is a direct audio-vs-text trust probe.
- **Transcript Injection Bias (TIB):** Acc(S4) − Acc(S5). Positive means matched transcripts hurt; negative means they help.

We also report per-stage **Expected Calibration Error (ECE)** and **ΔConf** (the drop in self-reported confidence when a modality is removed).

---

## 2. Headline accuracy

Overall accuracy by stage on the 600-sample set:

| Stage | Gemma-3n-E2B-IT | Qwen2.5-Omni-7B (Jeff) | Δ (Qwen − Gemma) |
|---|---:|---:|---:|
| S1 text-only | 0.277 | 0.289 | +0.012 |
| S2 audio-only | 0.427 | 0.499 | +0.072 |
| S3 visual-only | 0.431 | 0.495 | +0.064 |
| S4 full AV | **0.503** | **0.591** | +0.088 |
| S5 + matched transcript | 0.533 | 0.624 | +0.091 |
| S6 attribution | 0.503 | 0.591 | +0.088 |
| S7 + mismatched transcript | 0.478 | (not run by Jeff) | — |
| S8 prosody | (not run by us) | 0.574 | — |

**Reading:**

1. The text-only floor (S1) sits at 0.277 vs the random-guess floor of 0.250 on 4-option MCQ. Both runs land within 1 pp of each other (0.277 vs 0.289), which confirms the AV-Human filter is doing its job — a model with no audio or video cannot solve these questions.
2. Audio alone (S2 = 0.43) and visual alone (S3 = 0.43) are essentially tied for Gemma. The model gains another 7 pp from combining them at S4. The same qualitative pattern holds for Qwen-Omni (S2 = 0.50, S3 = 0.50, S4 = 0.59).
3. **Qwen-Omni-7B is uniformly +7 to +9 pp better than Gemma-3n-E2B-IT** across S2–S6. Given the 3.5× parameter gap, this is the expected scaling direction, but the *qualitative pattern* — single-modality near-tie, full-AV gain, transcript helps — is identical across architectures.
4. **Transcript injection (S5) helps both models.** Gemma gains +3 pp; Qwen-Omni gains +3 pp. This is *not* what we expected — we hypothesized matched transcripts would hurt because they don't capture acoustic counts/timing. Counter-evidence to the LISTEN-style "audio LLMs over-rely on transcripts" narrative when the transcript is correct.
5. **S6 ≈ S4.** Adding the "which modality did you use?" follow-up does not move accuracy. The follow-up is a clean attribution probe, not an answer-changing intervention.

### Per-task accuracy (Gemma)

| Task | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio Information Extraction (AIE) | 0.30 | 0.71 | 0.39 | **0.73** | 0.86 | 0.73 | 0.67 |
| Audio Content Counting (ACC) | 0.29 | 0.33 | 0.27 | 0.35 | 0.33 | 0.35 | 0.30 |
| Audio Event Location (AEL) | 0.35 | 0.37 | 0.36 | 0.37 | 0.34 | 0.37 | 0.38 |
| Audio Character Matching (AVCM) | 0.21 | 0.31 | 0.55 | 0.48 | 0.57 | 0.48 | 0.49 |
| Audio Object Matching (AVOM) | 0.27 | 0.35 | 0.48 | 0.53 | 0.48 | 0.53 | 0.50 |
| Audio OCR Matching (AVTM) | 0.24 | 0.48 | 0.54 | 0.56 | 0.62 | 0.56 | 0.53 |

A few task-level patterns matter:

- **AIE is the only task Gemma actually solves** (S4 = 0.73) — and S5 lifts it further to 0.86. It's a pure-audio task (extract a fact stated in the audio); transcripts make it nearly trivial.
- **ACC and AEL are flat across stages.** They sit ~0.35 regardless of input. These are audio-counting and audio-timing tasks, which require *acoustic* reasoning the transcript can't help with. Notably they're also the tasks where Gemma is closest to chance.
- **AVCM (character) and AVOM (object) gain from visual.** Visual-only S3 beats audio-only S2 (0.55 vs 0.31 on AVCM). These are cross-modal *binding* tasks — match a sound to a face/object — which the visual stream helps with regardless of whether the audio is processed correctly.

---

## 3. Modality attribution: Gemma confabulates 50% of the time

The single most striking finding is the **Attribution Faithfulness Score**. After answering at S4, Gemma is asked which modality it used. Across all six tasks, it claims it relied on audio **97-100% of the time** (only 1-3% of S6 responses claim visual or text). When we then *test* that claim by removing audio and seeing if the answer flips:

| Task | AFS (Gemma) | AFS (Qwen-Omni) | Falsifiable n | Faithful | Confab. | Trivial |
|---|---:|---:|---:|---:|---:|---:|
| Audio Information Extraction | **0.70** | 0.78 | 57 | 40 | 17 | 43 |
| Audio Content Counting | **0.66** | 0.71 | 50 | 33 | 17 | 50 |
| Audio Event Location | **0.63** | 0.57 | 43 | 27 | 16 | 57 |
| Audio OCR Matching | 0.40 | 0.43 | 60 | 24 | 36 | 39 |
| Audio Character Matching | 0.35 | 0.41 | 69 | 24 | 45 | 31 |
| Audio Object Matching | **0.31** | **0.50** | 55 | 17 | 38 | 45 |
| **OVERALL** | **0.494** | (not aggregated) | 334 | 165 | 169 | 265 |

**Reading:**

- Overall, Gemma's modality attribution is correct on a coin flip. 165 of 334 falsifiable cases were faithful; 169 were confabulated. That's the headline confabulation finding.
- The pattern is *task-conditional*: the three pure-audio tasks (AIE, ACC, AEL) have AFS in the 0.63–0.70 range — meaningfully above chance. The three cross-modal *matching* tasks (AVTM, AVCM, AVOM) sit at 0.31–0.40. The model claims it used audio, but on matching tasks the audio claim is essentially uninformative — its answer doesn't change when you remove audio.
- **The biggest cross-model gap is AVOM** (Gemma 0.31 vs Qwen-Omni 0.50). Both models confabulate on cross-modal binding, but the smaller model does so much more.
- The "Trivial" column matters methodologically. 265 of 600 questions had every single-modality stage (S2, S3, S4) independently producing the same answer — for those, the modality attribution question is unfalsifiable. We exclude them from the AFS denominator. Without this filter, AFS would be artificially deflated by easy questions.

---

## 4. Confidence calibration: the model "knows" audio matters, even when wrong

Although Gemma's verbalized self-attribution is poor, its **confidence drops** under ablation tell a more flattering story:

| Task | ΔConf(audio_removal) | ΔConf(visual_removal) |
|---|---:|---:|
| Audio Information Extraction | **+6.79** | −0.01 |
| Audio Content Counting | +3.38 | −1.36 |
| Audio Event Location | +2.70 | −1.30 |
| Audio Character Matching | +1.85 | −0.25 |
| Audio Object Matching | +2.55 | −0.55 |
| Audio OCR Matching | +2.55 | +0.15 |

ΔConf(audio) = mean(Conf at S4) − mean(Conf at S3, where audio is removed). A **positive** ΔConf means the model lowered its confidence when the modality was removed, which is the calibrated direction.

**Reading:** Gemma's confidence drops by 1.8–6.8 pp when audio is ablated, and by essentially zero when visual is ablated. The **model behaviorally treats audio as the more important modality**, even though its *verbal* attribution is wrong half the time. The largest audio-removal drop is on AIE (+6.79 pp), the task with the highest AFS — consistent: when the model genuinely uses audio, it knows it does.

### Expected Calibration Error (ECE)

ECE measures the gap between stated confidence and empirical correctness rate, binned. Lower is better:

| Stage | ECE |
|---|---:|
| S1 text-only | 0.671 |
| S2 audio-only | 0.545 |
| S3 visual-only | 0.503 |
| S4 full AV | 0.467 |
| S5 transcript-injected | 0.450 |
| S6 attribution | 0.467 |
| S7 mismatched transcript | 0.498 |

ECE is high across the board — Gemma reports confidence in the 80–95 range while accuracy hovers at 0.30–0.50. But it improves monotonically with information added: from 0.67 at S1 to 0.45 at S5. S7 (mismatched transcript) regresses ECE back to 0.50, suggesting the model's confidence does not appropriately deflate when handed contradictory text.

---

## 5. Lexical override and transcript bias

| Task | LOR (Gemma) | TIB (Gemma) |
|---|---:|---:|
| Audio Content Counting | 0.286 | +0.020 |
| Audio Event Location | 0.167 | +0.030 |
| Audio Information Extraction | 0.110 | **−0.130** |
| Audio Character Matching | 0.146 | −0.090 |
| Audio Object Matching | 0.170 | +0.050 |
| Audio OCR Matching | 0.164 | −0.061 |
| **OVERALL** | **0.163** | **−0.030** |

**Reading:**

- LOR overall is **16.3%**: among the 300 questions Gemma got right at S4, swapping in a contradictory transcript flipped 49 of them (251 stayed). The remaining 298 questions had S4 wrong and were excluded from the LOR denominator.
- LOR is *highest* on ACC (0.29). This is the task with the smallest absolute number of S4-correct cases (35), so the ratio is noisier — but qualitatively, when the model is barely above guessing, even a wrong transcript pushes it around easily.
- **TIB is slightly negative overall (−0.030).** Matched transcripts *help* on average. Per task, AIE shows the biggest positive effect (TIB = −0.13: transcripts boost AIE accuracy by 13 pp). AVOM is the only task where TIB is meaningfully positive (+0.05), and it's also the task with the lowest AFS — i.e., on the task where the model confabulates most about audio, transcript injection actually hurts.
- The lexical-override headline is therefore **moderate, not catastrophic**: Gemma trusts audio over text in 84% of audio-correct cases. This is a notably better number than what would be predicted by the LISTEN finding for pure audio LLMs — the visual stream may be acting as a partial guard against transcript override.

---

## 6. Answer flip rates between stage pairs

For an at-a-glance sense of which input changes Gemma's mind:

| Comparison | Overall flip rate |
|---|---:|
| Remove audio (S4 → S3) | 27.2% |
| Remove visual (S4 → S2) | 37.7% |
| Add matched transcript (S4 → S5) | 22.5% |
| Add mismatched transcript (S4 → S7) | 20.6% |
| Strip to text-only (S4 → S1) | 64.6% |

A surprising number: removing **visual** flips more answers (37.7%) than removing **audio** (27.2%). Per-task this is consistent with the visual-flip being concentrated on the matching tasks (AVCM 58%, AVOM 48%, AVTM 43%), where visual cues are essential for binding. On AIE, audio removal flips 41% of answers and visual removal flips only 22% — exactly the pattern you'd want from a model attending to the right modality per task.

---

## 7. What this means

Three findings stand out:

1. **The qualitative diagnostic pattern is model-agnostic.** Gemma-3n-E2B-IT (~2 B active params) and Qwen-Omni-7B produce the same shape of results: single-modality near-tie, full-AV gain, transcript helps, S6 ≈ S4, AFS lowest on cross-modal matching tasks. The 3.5× parameter gap shifts levels uniformly by 7–9 pp but does not change the diagnostic story. This is good news for the methodology — the AFS / LOR / TIB framework is not picking up an artifact of one specific model.
2. **Self-reported attribution is unreliable.** AFS = 0.494 overall on Gemma means the model's claim "I used audio" is correct on a coin flip. The confabulation is concentrated on tasks where the audio claim is least falsifiable (cross-modal matching). This is a concrete handle for future work: any system that surfaces model self-reports as evidence — interpretability dashboards, attention rationales, "the model said it relied on X" claims — should be tested against ablation, not taken at face value.
3. **Behavioral attribution (ΔConf, flip rates) is more honest than verbal attribution (AFS).** Gemma's confidence drops the right way under ablation, and its flip rates are task-appropriate (audio removal hurts AIE, visual removal hurts matching). The disconnect between *verbal* attribution (poor) and *behavioral* attribution (reasonable) is itself a useful finding — it suggests the model's internal modality routing is fine; the layer that *describes* the routing is the broken one.

### Limitations

- 600 samples is smaller than Jeff's full-AVUT run (~1,443 valid samples). On per-task subsets (e.g., ACC's LOR denominator of 35) the numbers are noisier.
- Whisper-small is the cheapest reliable ASR; its errors propagate into S5 and S7 transcripts.
- S6's attribution probe is concatenated multi-turn (the model sees its S4 raw response in the same context window) rather than a true two-turn conversation. A true two-turn version is future work.
- We did not run S8 (prosody-first verbalization). Jeff's S8 numbers (0.574 overall) are reported for comparability but we cannot make Gemma-on-S8 claims.
- Gemma-3n's vision is image-based: we sample 8 frames per video. A video-token model (like Qwen-Omni's native handler) sees richer temporal information. This contributes to Gemma's lower visual-only S3 number on AIE specifically.

### Reproduction

All code is in the repo (`samadasyed/omnimodel-research`). Notebook 04 produced the predictions; notebook 06 produced the metrics and figures. The full run took ~5 hours unattended on a single A100; the eval is resume-safe and writes per-stage JSONs every 25 samples to Drive. Cross-model tables in §2 and §3 are auto-fetched from Jeff's published metric JSONs at `https://raw.githubusercontent.com/jjwang8/639_avut/main/results/metrics/`. Numerical equivalence with Jeff's pipeline was verified by feeding his raw predictions through our `metrics.compute_*` and matching every published JSON bit-for-bit.

---

*All numbers in this report are reproducible from `results/metrics/gemma_3n_e2b/*.json` and `results/raw_predictions/gemma_3n_e2b/*.json` in the repo.*
