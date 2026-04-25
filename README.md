# Beyond Accuracy: Diagnosing Modality Attribution and Confidence Calibration in Video-LLMs

This repo extends the [AVUT benchmark](https://arxiv.org/abs/2510.XXXXX) (Yang et al., EMNLP 2025) with three diagnostic axes that go beyond raw accuracy:

1. **Modality Attribution Faithfulness (AFS)** — When a model claims it relied on audio, does removing audio actually change its answer? We measure how often models confabulate about their own modality use.
2. **Confidence Calibration under Ablation (CCA)** — Does a model's confidence drop when you remove the modality carrying the signal, or does it stay overconfident even when "flying blind"?
3. **Lexical Override Rate (LOR)** — When the audio says one thing and an injected (mismatched) transcript says another, which one wins? A direct lexical-vs-acoustic conflict probe.

We evaluate Qwen2.5-Omni-7B (and Gemma-3n-E2B-IT for cross-model comparison) on a balanced 120-sample subset of AVUT-Human across six MCQ task types.

> **TL;DR finding (placeholder until run completes):** Omnimodal models report relying on audio more than their counterfactual behavior justifies — they confabulate. Self-reported attribution and ablation-based attribution disagree on roughly N% of audio-essential questions.

---

## Quickstart (Colab, H100 recommended)

Each notebook is designed to run in under ~30 minutes on an H100. They write intermediate artifacts to `data/` and `results/`, so any notebook can be re-run independently after the previous ones have produced their outputs.

```bash
git clone https://github.com/<you>/omnimodel-research.git
cd omnimodel-research
```

Then in Colab:

1. **`notebooks/01_setup_and_download.ipynb`** — installs dependencies, pulls the AVUT annotation JSON from HuggingFace, downloads videos via `yt-dlp`, logs the success rate per task.
2. **`notebooks/02_preprocess.ipynb`** — extracts audio (`ffmpeg`), creates silent video, runs Whisper-small for ASR transcripts, builds the mismatched-transcript pool.
3. **`notebooks/03_pilot.ipynb`** — 24-sample pilot run across all stages on Qwen2.5-Omni-7B; sanity-checks the prompts, parsing, and per-stage timings before committing to the full run.
4. **`notebooks/04_main_eval_qwen.ipynb`** — full 120-sample run across S0–S5 on Qwen2.5-Omni-7B.
5. **`notebooks/05_main_eval_gemma.ipynb`** — same on Gemma-3n-E2B-IT (optional, for cross-model robustness).
6. **`notebooks/06_analysis.ipynb`** — computes accuracy, AFS, ΔConf, ECE, LOR, TIB; writes JSON to `results/metrics/` and figures to `results/figures/`.

## Stages

| Stage | Name | Model | Inputs |
|------:|:-----|:------|:-------|
| S0 | Text-only baseline       | Qwen2.5-1.5B-Instruct | Question + options |
| S1 | Audio-only               | Qwen2.5-Omni-7B       | Audio + question |
| S2 | Visual-only              | Qwen2.5-Omni-7B       | Silent video + question |
| S3 | Full audio-visual        | Qwen2.5-Omni-7B       | Video + audio + question + inline attribution probe |
| S4 | + ASR transcript         | Qwen2.5-Omni-7B       | S3 inputs + Whisper-small transcript of *this* video |
| S5 | + Mismatched transcript  | Qwen2.5-Omni-7B       | S3 inputs + transcript from a *different* same-task video |

S5 is the key novel condition: it lets us measure how often a model abandons its audio-grounded answer when handed a contradictory transcript. High lexical override rate = the audio encoder isn't being trusted by the model's reasoning head.

## Metrics

- **Accuracy** — per task per stage. Compares against AVUT's published Gemini 1.5 Pro Table 5 numbers.
- **Confidence (verbalized + logprob)** — verbalized 0–100 scale and softmax over A/B/C/D logits.
- **ΔConf(modality M)** = Conf(S3 full) − Conf(S without M). Calibrated models drop confidence when the signal-carrying modality is removed.
- **ECE / Brier score** — calibration quality of logprob-based confidence.
- **AFS (Attribution Faithfulness Score)** — fraction of samples where ablating the model's self-reported top modality actually flips the answer.
- **TIB (Transcript Injection Bias)** = Acc(S3) − Acc(S4). Positive = transcripts hurt (model over-relies on text on audio-essential tasks).
- **LOR (Lexical Override Rate)** = fraction of samples where the answer changes from S3 to S5 (mismatched transcript), restricted to samples where S3 was correct. Direct measure of audio-vs-text trust.

Full metric definitions live in `src/metrics.py`.

## Repo layout

```
agentdocs/        Original research direction docs + changes log (gitignored from public repo)
data/             Annotations, downloaded videos, derived audio/transcripts (gitignored)
notebooks/        Colab-ready .ipynb files; numbered by execution order
paper/            Class write-up (5-page report)
results/
  raw_predictions/  Per-stage per-sample predictions (JSON)
  metrics/          Aggregated metric JSONs
  figures/          PNG/PDF figures used in the paper
src/              Python utilities imported from notebooks
```

## Citation

If this work is useful, please cite the AVUT paper that this builds on:

```bibtex
@inproceedings{yang2025audio,
  title={Audio-centric Video Understanding Benchmark without Text Shortcut},
  author={Yang, Yudong and Zhuang, Jimin and Sun, Guangzhi and Tang, Changli and Li, Yixuan and Li, Peihan and Jiang, Yifan and Li, Wei and Ma, Zejun and Zhang, Chao},
  booktitle={EMNLP},
  year={2025}
}
```
