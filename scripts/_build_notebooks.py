"""Build the six Colab notebooks for the AVUT diagnostic pipeline.

Run from repo root:
    python scripts/_build_notebooks.py

This file is a builder, not part of the analysis pipeline. It exists so that
notebooks remain reviewable in git diffs (we generate them from typed cell
lists rather than handcrafting JSON).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

Cell = Tuple[str, str]  # (cell_type, source)


def make_notebook(cells: List[Cell]) -> dict:
    nb_cells = []
    for cell_type, source in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(name: str, cells: List[Cell]) -> None:
    nb = make_notebook(cells)
    path = NOTEBOOKS_DIR / name
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {path}")


# ────────────────────────────────────────────────────────────────────
# Notebook 01: Setup and Download
# ────────────────────────────────────────────────────────────────────
NB_01 = [
    ("markdown", """# 01 — Setup, Annotations Download, and Video Acquisition

This notebook prepares the AVUT-Human dataset for evaluation.

**What it does:**
1. Installs dependencies (transformers, accelerate, qwen-omni-utils, openai-whisper, yt-dlp, ffmpeg-python).
2. Downloads the AVUT annotation JSONs from HuggingFace (`tsinghua-ee/AVUTBenchmark`).
3. Inspects the JSON schema and prints a sample entry — **STOP and verify field names match `src/data_utils.py`**.
4. Downloads videos from YouTube via `yt-dlp` and logs success rate per task.

**Runtime:** ~2–4 hours (mostly YouTube downloads). Resume-safe: rerun to skip already-downloaded videos.

**GPU:** Not required for this notebook; H100 is wasted here. Use a CPU runtime if available.
"""),
    ("markdown", "## 0. Environment\n\nIf running outside Colab, mount the repo manually. In Colab, clone the repo first:"),
    ("code", """# Colab: clone repo (skip if running locally)
import os, sys
REPO = '/content/omnimodel-research'
if not os.path.exists(REPO):
    # Replace with your fork/branch as needed
    !git clone https://github.com/<you>/omnimodel-research.git /content/omnimodel-research
%cd $REPO
sys.path.insert(0, REPO)
"""),
    ("code", """# Install dependencies. ffmpeg is preinstalled on Colab.
!pip install -q transformers>=4.45 accelerate sentencepiece
!pip install -q yt-dlp openai-whisper
!pip install -q qwen-omni-utils
!pip install -q huggingface_hub
"""),
    ("markdown", "## 1. Download AVUT annotations"),
    ("code", """from huggingface_hub import hf_hub_download
import json, os

os.makedirs('data/annotations', exist_ok=True)

for fname in ['AV_Human_data.json', 'AV_Gemini_data.json']:
    try:
        path = hf_hub_download(
            repo_id='tsinghua-ee/AVUTBenchmark',
            filename=fname,
            repo_type='dataset',
            local_dir='data/annotations',
        )
        print(f'OK: {fname} → {path}')
    except Exception as e:
        print(f'FAILED: {fname}: {e}')
"""),
    ("markdown", "## 2. Inspect JSON schema (CRITICAL — verify before proceeding)"),
    ("code", """with open('data/annotations/AV_Human_data.json') as f:
    raw = json.load(f)

print('Type at top level:', type(raw).__name__)
entries = raw if isinstance(raw, list) else list(raw.values())
print(f'Number of entries: {len(entries)}')
print()
print('First entry — full schema:')
print(json.dumps(entries[0], indent=2)[:2000])
print()
print('All keys present in first 50 entries:')
keys = set()
for e in entries[:50]:
    if isinstance(e, dict):
        keys.update(e.keys())
print(sorted(keys))
"""),
    ("code", """# Task type distribution (uses src/data_utils.py with its tolerant field-name handling)
from src.data_utils import load_annotations, filter_to_target_tasks, task_distribution

normalized = load_annotations('data/annotations/AV_Human_data.json')
print(f'Normalized entries (with valid video_id, question, options, answer): {len(normalized)}')
print()
print('Task distribution (all tasks):')
print(json.dumps(task_distribution(normalized), indent=2))
print()
print('Task distribution (MCQ tasks we evaluate):')
mcq = filter_to_target_tasks(normalized)
print(json.dumps(task_distribution(mcq), indent=2))
print(f'Total MCQ entries: {len(mcq)}')
"""),
    ("markdown", """## 3. Download videos via yt-dlp

We download up to 720p to keep storage reasonable. Since we'll later sample 120 entries, we can either:
- (a) Download videos for all ~700 AV-Human entries (~50–100 GB), then sample.
- (b) Sample first, then download only the 120-sample subset (~10 GB, much faster).

We use approach **(b)** for speed. If the sampled set has too many download failures, increase the sample size and resample.
"""),
    ("code", """from src.data_utils import balanced_subsample
import subprocess

# AVUT has 692 unique videos but 1734 QA pairs (each video has ~2.5 QAs across
# different task types). Multiple QAs can share the same downloaded video.
# We sample BY QUESTION (qa_id) but DEDUPE downloads by video_id.

# Sample more than we need so we have headroom for failed YouTube downloads.
N_PER_TASK_TO_DOWNLOAD = 30   # we'll keep 20/task after filtering for download success
SEED = 42

candidate_pool = balanced_subsample(mcq, n_per_task=N_PER_TASK_TO_DOWNLOAD, seed=SEED)
unique_videos = list({s['video_id']: s for s in candidate_pool}.values())
print(f'Candidate QA pairs: {len(candidate_pool)} ({N_PER_TASK_TO_DOWNLOAD}/task × 6 tasks)')
print(f'Unique videos to download: {len(unique_videos)}')

# Save the candidate set so we can re-run downstream notebooks deterministically
os.makedirs('data', exist_ok=True)
with open('data/candidate_samples.json', 'w') as f:
    json.dump(candidate_pool, f, indent=2)
print('Saved candidate QA list to data/candidate_samples.json')
"""),
    ("code", """# Download videos. Each AVUT entry already carries the YouTube URL inline
# (`url` field), so no separate mapping file is needed.

def youtube_url_for(sample):
    return sample['url']  # AVUT annotations include the URL directly
"""),
    ("code", """os.makedirs('data/videos', exist_ok=True)
download_log = []

for i, sample in enumerate(unique_videos):
    vid = sample['video_id']
    out_path = f'data/videos/{vid}.mp4'
    if os.path.exists(out_path):
        download_log.append({'video_id': vid, 'status': 'cached'})
        continue
    url = youtube_url_for(sample)
    try:
        result = subprocess.run(
            [
                'yt-dlp',
                '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]',
                '--merge-output-format', 'mp4',
                '-o', out_path,
                '--no-playlist',
                '--socket-timeout', '30',
                '--quiet',
                url,
            ],
            timeout=180,
            check=False,
            capture_output=True,
        )
        ok = os.path.exists(out_path)
        download_log.append({'video_id': vid, 'status': 'ok' if ok else 'failed',
                             'returncode': result.returncode})
    except subprocess.TimeoutExpired:
        download_log.append({'video_id': vid, 'status': 'timeout'})
    if (i + 1) % 10 == 0:
        succ = sum(1 for d in download_log if d['status'] in ('ok', 'cached'))
        print(f'  [{i+1}/{len(unique_videos)}] success={succ}/{i+1}')

from collections import Counter
status_counts = Counter(d['status'] for d in download_log)
print()
print('Per-video download status:', dict(status_counts))
n_avail = sum(1 for d in download_log if d['status'] in ('ok', 'cached'))
print(f'Videos available on disk: {n_avail}/{len(download_log)}')

# Now compute QA-level availability (a QA is available iff its video is)
avail_vids = {d['video_id'] for d in download_log if d['status'] in ('ok', 'cached')}
qa_avail = [s for s in candidate_pool if s['video_id'] in avail_vids]
print(f'QA pairs available: {len(qa_avail)}/{len(candidate_pool)}')

per_task = Counter(s['task_type'] for s in qa_avail)
print('Per-task QA availability:')
for t, n in per_task.most_common():
    print(f'  {t}: {n}')

with open('data/download_log.json', 'w') as f:
    json.dump(download_log, f, indent=2)
"""),
    ("markdown", "## 4. Lock the final balanced eval set\n\nFilter the candidate pool to videos actually on disk, then sub-sample to exactly 20 per task. This is the canonical sample list every downstream notebook reads."),
    ("code", """from src.data_utils import filter_to_available_videos, balanced_subsample, task_distribution_by_code

available = filter_to_available_videos(candidate_pool, video_dir='data/videos')
print(f'QA pairs with video on disk: {len(available)} / {len(candidate_pool)}')
print('Per-task available:', task_distribution_by_code(available))

N_PER_TASK_FINAL = 20
final_set = balanced_subsample(available, n_per_task=N_PER_TASK_FINAL, seed=42)
print()
print(f'Final eval set: {len(final_set)} QA pairs ({N_PER_TASK_FINAL}/task)')
print('Per-task final:', task_distribution_by_code(final_set))

with open('data/eval_samples.json', 'w') as f:
    json.dump(final_set, f, indent=2)
print('Saved final eval sample list to data/eval_samples.json')
"""),
    ("markdown", "**Done.** Proceed to `02_preprocess.ipynb`."),
]


# ────────────────────────────────────────────────────────────────────
# Notebook 02: Preprocess
# ────────────────────────────────────────────────────────────────────
NB_02 = [
    ("markdown", """# 02 — Preprocessing

For each video in the locked eval set:
1. Extract the audio track to `data/audio/{vid}.wav` (16 kHz mono PCM, Whisper-compatible).
2. Strip audio to make a silent video at `data/silent_video/{vid}_silent.mp4` (used in S2 visual-only).
3. Run Whisper-small for ASR transcript at `data/transcripts/{vid}.txt`.
4. Build the mismatched-transcript pool (used in S5 lexical-override stage).

**Runtime:** ~30–45 minutes for 120 videos on H100 (Whisper is the bottleneck).

**Safe to re-run:** every step skips files that already exist.
"""),
    ("code", """import os, sys, json, subprocess
REPO = '/content/omnimodel-research'
if os.path.exists(REPO):
    %cd $REPO
    sys.path.insert(0, REPO)

from pathlib import Path

with open('data/eval_samples.json') as f:
    samples = json.load(f)
print(f'Preprocessing {len(samples)} videos.')

VIDEO_DIR = 'data/videos'
AUDIO_DIR = 'data/audio'
SILENT_DIR = 'data/silent_video'
TRANSCRIPT_DIR = 'data/transcripts'
TRANSCRIPT_MISMATCHED_DIR = 'data/transcripts_mismatched'

for d in [AUDIO_DIR, SILENT_DIR, TRANSCRIPT_DIR, TRANSCRIPT_MISMATCHED_DIR]:
    os.makedirs(d, exist_ok=True)
"""),
    ("markdown", "## 1. Audio extraction (ffmpeg)"),
    ("code", """for s in samples:
    vid = s['video_id']
    in_path = f'{VIDEO_DIR}/{vid}.mp4'
    out_path = f'{AUDIO_DIR}/{vid}.wav'
    if os.path.exists(out_path):
        continue
    if not os.path.exists(in_path):
        print(f'  Skipping {vid}: no source video')
        continue
    subprocess.run([
        'ffmpeg', '-y', '-i', in_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        out_path,
    ], capture_output=True)
print(f'Audio files: {len([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])}')
"""),
    ("markdown", "## 2. Silent video creation (ffmpeg, audio-stripped)"),
    ("code", """for s in samples:
    vid = s['video_id']
    in_path = f'{VIDEO_DIR}/{vid}.mp4'
    out_path = f'{SILENT_DIR}/{vid}_silent.mp4'
    if os.path.exists(out_path):
        continue
    if not os.path.exists(in_path):
        continue
    subprocess.run([
        'ffmpeg', '-y', '-i', in_path,
        '-an', '-c:v', 'copy',
        out_path,
    ], capture_output=True)
print(f'Silent videos: {len([f for f in os.listdir(SILENT_DIR) if f.endswith(".mp4")])}')
"""),
    ("markdown", "## 3. Whisper transcripts (matched, for S4)"),
    ("code", """from src.transcript_utils import transcribe_with_whisper

# Whisper-small balances speed and quality. The transcript is intentionally imperfect —
# real-world transcripts have errors, and a perfect transcript would inflate the
# text signal artificially.

for i, s in enumerate(samples):
    vid = s['video_id']
    audio_path = f'{AUDIO_DIR}/{vid}.wav'
    if not os.path.exists(audio_path):
        continue
    text = transcribe_with_whisper(audio_path, output_dir=TRANSCRIPT_DIR, model_size='small')
    if (i + 1) % 10 == 0:
        print(f'  [{i+1}/{len(samples)}] transcripts done')

ts_files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.txt')]
print(f'Transcripts: {len(ts_files)}')
"""),
    ("markdown", """## 4. Mismatched-transcript pool (for S5)

For each sample, pair it with the transcript of a DIFFERENT same-task video. This is the lexical-override condition: the audio in the played video tells the truth, but the transcript contradicts it.

We persist the pairing to disk so S5 reads it deterministically.
"""),
    ("code", """from src.transcript_utils import build_mismatched_pairs, load_transcript

pairs = build_mismatched_pairs(samples, seed=42)
# pairs maps {qa_id: donor_video_id}

n_written = 0
for qa_id, donor_vid in pairs.items():
    out_path = f'{TRANSCRIPT_MISMATCHED_DIR}/{qa_id}.txt'
    if os.path.exists(out_path):
        n_written += 1
        continue
    donor_text = load_transcript(f'{TRANSCRIPT_DIR}/{donor_vid}.txt')
    if not donor_text:
        continue
    with open(out_path, 'w') as f:
        f.write(donor_text)
    n_written += 1

with open('data/mismatched_pairs.json', 'w') as f:
    # JSON keys must be strings — convert qa_id back when reading
    json.dump({str(k): v for k, v in pairs.items()}, f, indent=2)

print(f'Mismatched transcripts written: {n_written} / {len(pairs)}')
"""),
    ("markdown", """## Sanity check

Confirm everything we need exists for each sample. If a sample is missing any artifact, drop it from the eval set — running stages on a partial sample creates messy paired-stage analysis later.
"""),
    ("code", """missing = []
for s in samples:
    vid = s['video_id']
    qa_id = s['qa_id']
    needed = [
        f'{VIDEO_DIR}/{vid}.mp4',
        f'{AUDIO_DIR}/{vid}.wav',
        f'{SILENT_DIR}/{vid}_silent.mp4',
        f'{TRANSCRIPT_DIR}/{vid}.txt',
        f'{TRANSCRIPT_MISMATCHED_DIR}/{qa_id}.txt',
    ]
    missing_files = [p for p in needed if not os.path.exists(p)]
    if missing_files:
        missing.append({'qa_id': qa_id, 'video_id': vid, 'missing': missing_files})

print(f'QA pairs with all artifacts: {len(samples) - len(missing)} / {len(samples)}')
if missing:
    print('Samples missing artifacts (will be dropped):')
    for m in missing[:10]:
        print(f"  qa_id={m['qa_id']} (video={m['video_id']}): missing {len(m['missing'])} files")

bad_qa_ids = {m['qa_id'] for m in missing}
clean_samples = [s for s in samples if s['qa_id'] not in bad_qa_ids]
with open('data/eval_samples_clean.json', 'w') as f:
    json.dump(clean_samples, f, indent=2)
print(f'Final clean eval set: {len(clean_samples)} samples → data/eval_samples_clean.json')
"""),
    ("markdown", "**Done.** Proceed to `03_pilot.ipynb`."),
]


# ────────────────────────────────────────────────────────────────────
# Notebook 03: Pilot
# ────────────────────────────────────────────────────────────────────
NB_03 = [
    ("markdown", """# 03 — Pilot Run (24 samples × 6 stages, Qwen2.5-Omni-7B)

Sanity-check the full pipeline before committing to the main run. Goals:
- Verify Qwen2.5-Omni-7B loads, processes audio + video, and returns parseable output.
- Verify the inline attribution probe in S3 produces parseable rankings >= 80% of the time. If parse rate is bad, iterate on prompt wording HERE before the main run.
- Spot-check that S0 ≈ random (~25%), S2 << S3 on audio-essential tasks, etc.
- Measure per-stage seconds/sample so we can size the main run's compute budget.

**Runtime:** ~20 minutes on H100 (4 samples/task × 6 tasks × 6 stages × ~10s ≈ 24 minutes).
"""),
    ("code", """import os, sys, json, time, random
REPO = '/content/omnimodel-research'
if os.path.exists(REPO):
    %cd $REPO
    sys.path.insert(0, REPO)

with open('data/eval_samples_clean.json') as f:
    all_samples = json.load(f)
print(f'Available eval samples: {len(all_samples)}')

# Pilot: 4 per task type from the clean set
from src.data_utils import balanced_subsample
pilot_samples = balanced_subsample(all_samples, n_per_task=4, seed=7)
print(f'Pilot set: {len(pilot_samples)} samples')

with open('data/pilot_samples.json', 'w') as f:
    json.dump(pilot_samples, f, indent=2)
"""),
    ("code", """from src.model_utils import OmniModelWrapper, TextOnlyModelWrapper
from src.prompts import STAGE_BUILDERS
from src.parse_utils import parse_answer_confidence, parse_attribution, parse_reason

print('Loading Qwen2.5-Omni-7B...')
omni = OmniModelWrapper('Qwen/Qwen2.5-Omni-7B')
print('Loading Qwen2.5-1.5B-Instruct (text baseline)...')
text_only = TextOnlyModelWrapper('Qwen/Qwen2.5-1.5B-Instruct')
print('Models loaded.')
"""),
    ("code", """# Helpers to construct stage-specific inputs
from src.transcript_utils import load_transcript

def stage_inputs(stage, sample):
    vid = sample['video_id']
    paths = {
        'video': f'data/videos/{vid}.mp4',
        'audio': f'data/audio/{vid}.wav',
        'silent': f'data/silent_video/{vid}_silent.mp4',
        'transcript': f'data/transcripts/{vid}.txt',
        # Mismatched transcript is keyed by qa_id (a single video may have
        # multiple QAs that need DIFFERENT mismatched donors)
        'mismatched': f'data/transcripts_mismatched/{sample["qa_id"]}.txt',
    }
    if stage == 'S0':
        return {'video_path': None, 'audio_path': None, 'extra': {}}
    if stage == 'S1':
        return {'video_path': None, 'audio_path': paths['audio'], 'extra': {}}
    if stage == 'S2':
        return {'video_path': paths['silent'], 'audio_path': None, 'extra': {}}
    if stage == 'S3':
        return {'video_path': paths['video'], 'audio_path': paths['audio'], 'extra': {}}
    if stage == 'S4':
        return {'video_path': paths['video'], 'audio_path': paths['audio'],
                'extra': {'transcript': load_transcript(paths['transcript'])}}
    if stage == 'S5':
        return {'video_path': paths['video'], 'audio_path': paths['audio'],
                'extra': {'mismatched_transcript': load_transcript(paths['mismatched'])}}
    raise ValueError(stage)

def run_one(stage, sample, model):
    inp = stage_inputs(stage, sample)
    prompt = STAGE_BUILDERS[stage](sample['question'], sample['options'], **inp['extra'])
    resp = model.generate(prompt, video_path=inp['video_path'], audio_path=inp['audio_path'])
    answer, conf = parse_answer_confidence(resp.text)
    attr = parse_attribution(resp.text) if stage == 'S3' else None
    reason = parse_reason(resp.text) if stage == 'S3' else None
    return {
        'qa_id': sample['qa_id'],
        'video_id': sample['video_id'],
        'task_type': sample['task_type'],
        'task_code': sample.get('task_code'),
        'stage': stage,
        'ground_truth': sample['answer'],
        'predicted_answer': answer,
        'verbalized_confidence': conf,
        'answer_logprobs': resp.answer_logprobs,
        'attribution': attr,
        'attribution_reason': reason,
        'raw_response': resp.text,
    }
"""),
    ("code", """import os
os.makedirs('results/raw_predictions/pilot', exist_ok=True)

STAGES = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
timings = {}

for stage in STAGES:
    print(f'\\n=== {stage} ===')
    t0 = time.time()
    records = []
    model = text_only if stage == 'S0' else omni
    for i, s in enumerate(pilot_samples):
        try:
            rec = run_one(stage, s, model)
        except Exception as e:
            rec = {'qa_id': s['qa_id'], 'video_id': s['video_id'],
                   'task_type': s['task_type'], 'task_code': s.get('task_code'),
                   'stage': stage, 'ground_truth': s['answer'],
                   'predicted_answer': None, 'verbalized_confidence': None,
                   'answer_logprobs': None, 'attribution': None,
                   'attribution_reason': None, 'raw_response': f'<ERROR: {e}>'}
        records.append(rec)
        if (i + 1) % 5 == 0:
            print(f'  [{i+1}/{len(pilot_samples)}]')

    with open(f'results/raw_predictions/pilot/{stage}.json', 'w') as f:
        json.dump(records, f, indent=2)
    elapsed = time.time() - t0
    timings[stage] = {'total_s': elapsed, 'per_sample_s': elapsed / len(pilot_samples)}
    print(f'  {stage} done in {elapsed:.0f}s ({elapsed/len(pilot_samples):.1f}s/sample)')

with open('results/raw_predictions/pilot/_timings.json', 'w') as f:
    json.dump(timings, f, indent=2)
"""),
    ("markdown", "## Sanity-check the pilot results"),
    ("code", """from src.metrics import accuracy_per_task

for stage in STAGES:
    with open(f'results/raw_predictions/pilot/{stage}.json') as f:
        recs = json.load(f)
    parsed = sum(1 for r in recs if r['predicted_answer'] is not None)
    acc = accuracy_per_task(recs)
    print(f'{stage}: parse_rate={parsed}/{len(recs)}  overall_acc={acc.get("OVERALL", 0):.2f}')
    if stage == 'S3':
        attr_parsed = sum(1 for r in recs if r['attribution'] is not None)
        print(f'   S3 attribution parse rate: {attr_parsed}/{len(recs)}')
"""),
    ("markdown", """## Decisions before main run

After running the cells above, check:

1. **Parse rates** — if any stage has < 80% answer parse rate, fix the prompt in `src/prompts.py` before kicking off the main run.
2. **S3 attribution parse rate** — < 80% means the attribution probe needs prompt iteration. The format `[ANSWER] X [CONFIDENCE] Y / [RANK] Audio=A, Visual=V, Text=T` is brittle.
3. **Per-sample timings** — if S3/S4/S5 are >> 15s/sample, the 120-sample main run will exceed the 2hr budget. Consider reducing N_PER_TASK to 15 in 04.
4. **S0 accuracy ≈ 0.25** — if S0 is much higher, the AVUT text-shortcut filtering is being violated by the small text model. Worth flagging in the paper either way.

When all checks pass, proceed to `04_main_eval_qwen.ipynb`.
"""),
]


# ────────────────────────────────────────────────────────────────────
# Notebook 04: Main eval (Qwen)
# ────────────────────────────────────────────────────────────────────
NB_04 = [
    ("markdown", """# 04 — Main Evaluation: Qwen2.5-Omni-7B (120 samples × S0–S5)

Runs the canonical evaluation. Results land in `results/raw_predictions/qwen/`.

**Runtime budget:** ~2hr on H100. Per-stage estimate from the pilot (`results/raw_predictions/pilot/_timings.json`):
- S0 (text-only): negligible
- S1, S2, S3, S4, S5: ~12s/sample × 120 ≈ 24min each

Each stage is checkpointed every 20 samples, so a Colab disconnect doesn't lose work — re-running this notebook resumes from the last checkpoint.
"""),
    ("code", """import os, sys, json, time
REPO = '/content/omnimodel-research'
if os.path.exists(REPO):
    %cd $REPO
    sys.path.insert(0, REPO)

with open('data/eval_samples_clean.json') as f:
    samples = json.load(f)
print(f'Eval samples: {len(samples)}')
"""),
    ("code", """from src.model_utils import OmniModelWrapper, TextOnlyModelWrapper
from src.prompts import STAGE_BUILDERS
from src.parse_utils import parse_answer_confidence, parse_attribution, parse_reason
from src.transcript_utils import load_transcript

print('Loading models...')
omni = OmniModelWrapper('Qwen/Qwen2.5-Omni-7B')
text_only = TextOnlyModelWrapper('Qwen/Qwen2.5-1.5B-Instruct')
print('Models loaded.')
"""),
    ("code", """def stage_inputs(stage, sample):
    vid = sample['video_id']
    paths = {
        'video':      f'data/videos/{vid}.mp4',
        'audio':      f'data/audio/{vid}.wav',
        'silent':     f'data/silent_video/{vid}_silent.mp4',
        'transcript': f'data/transcripts/{vid}.txt',
        # Mismatched transcript is keyed by qa_id (a single video may have
        # multiple QAs that need DIFFERENT mismatched donors)
        'mismatched': f'data/transcripts_mismatched/{sample["qa_id"]}.txt',
    }
    routes = {
        'S0': dict(video_path=None, audio_path=None, extra={}),
        'S1': dict(video_path=None, audio_path=paths['audio'], extra={}),
        'S2': dict(video_path=paths['silent'], audio_path=None, extra={}),
        'S3': dict(video_path=paths['video'], audio_path=paths['audio'], extra={}),
        'S4': dict(video_path=paths['video'], audio_path=paths['audio'],
                   extra={'transcript': load_transcript(paths['transcript'])}),
        'S5': dict(video_path=paths['video'], audio_path=paths['audio'],
                   extra={'mismatched_transcript': load_transcript(paths['mismatched'])}),
    }
    return routes[stage]

def run_one(stage, sample, model):
    inp = stage_inputs(stage, sample)
    prompt = STAGE_BUILDERS[stage](sample['question'], sample['options'], **inp['extra'])
    resp = model.generate(prompt, video_path=inp['video_path'], audio_path=inp['audio_path'])
    answer, conf = parse_answer_confidence(resp.text)
    attr = parse_attribution(resp.text) if stage == 'S3' else None
    reason = parse_reason(resp.text) if stage == 'S3' else None
    return {
        'video_id': sample['video_id'], 'task_type': sample['task_type'], 'stage': stage,
        'ground_truth': sample['answer'], 'predicted_answer': answer,
        'verbalized_confidence': conf, 'answer_logprobs': resp.answer_logprobs,
        'attribution': attr, 'attribution_reason': reason, 'raw_response': resp.text,
    }

OUT_DIR = 'results/raw_predictions/qwen'
os.makedirs(OUT_DIR, exist_ok=True)
"""),
    ("code", """def run_stage(stage, samples, model, checkpoint_every=20):
    out_path = f'{OUT_DIR}/{stage}.json'
    done = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = json.load(f)
        print(f'  {stage}: resuming with {len(done)} cached records')
    done_ids = {r['qa_id'] for r in done}
    todo = [s for s in samples if s['qa_id'] not in done_ids]
    if not todo:
        print(f'  {stage}: already complete')
        return done

    t0 = time.time()
    records = list(done)
    for i, s in enumerate(todo):
        try:
            rec = run_one(stage, s, model)
        except Exception as e:
            rec = {'video_id': s['video_id'], 'task_type': s['task_type'], 'stage': stage,
                   'ground_truth': s['answer'], 'predicted_answer': None,
                   'verbalized_confidence': None, 'answer_logprobs': None,
                   'attribution': None, 'attribution_reason': None,
                   'raw_response': f'<ERROR: {e}>'}
        records.append(rec)
        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(todo):
            with open(out_path, 'w') as f:
                json.dump(records, f, indent=2)
            elapsed = time.time() - t0
            done_n = len(records)
            print(f'  {stage}: {done_n}/{len(samples)}  ({elapsed:.0f}s elapsed, {elapsed/(i+1):.1f}s/sample)')
    return records

STAGES = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
all_results = {}
for stage in STAGES:
    print(f'\\n=== Running {stage} ===')
    model = text_only if stage == 'S0' else omni
    all_results[stage] = run_stage(stage, samples, model)
print('\\n=== Main eval complete ===')
"""),
    ("code", """# Quick sanity scan
from src.metrics import accuracy_per_task
for stage in STAGES:
    with open(f'{OUT_DIR}/{stage}.json') as f:
        recs = json.load(f)
    acc = accuracy_per_task(recs)
    parse_rate = sum(1 for r in recs if r['predicted_answer'] is not None) / len(recs)
    print(f'{stage}: parse={parse_rate:.0%}  overall_acc={acc.get("OVERALL", 0):.2f}  per-task={ {t: f"{v:.2f}" for t, v in acc.items() if t != "OVERALL"} }')
"""),
    ("markdown", "**Done.** Proceed to `06_analysis.ipynb` for metric computation, or `05_main_eval_gemma.ipynb` for cross-model comparison."),
]


# ────────────────────────────────────────────────────────────────────
# Notebook 05: Main eval (Gemma-3n) — optional cross-model
# ────────────────────────────────────────────────────────────────────
NB_05 = [
    ("markdown", """# 05 — Main Evaluation: Gemma-3n-E2B-IT (cross-model comparison)

Optional. Run if Qwen results look interesting and we want to test whether the confabulation finding generalizes across model families.

**Runtime:** ~1hr on H100 (Gemma-3n-E2B is smaller than Qwen2.5-Omni-7B).

**Caveat:** Gemma-3n's audio support is more limited than Qwen2.5-Omni's. If Gemma doesn't accept audio-only input cleanly (S1), we may need to skip S1 for Gemma and report findings only across S2–S5.
"""),
    ("code", """import os, sys, json, time
REPO = '/content/omnimodel-research'
if os.path.exists(REPO):
    %cd $REPO
    sys.path.insert(0, REPO)

with open('data/eval_samples_clean.json') as f:
    samples = json.load(f)
print(f'Eval samples: {len(samples)}')
"""),
    ("code", """from src.model_utils import GemmaOmniWrapper, TextOnlyModelWrapper
from src.prompts import STAGE_BUILDERS
from src.parse_utils import parse_answer_confidence, parse_attribution, parse_reason
from src.transcript_utils import load_transcript

gemma = GemmaOmniWrapper('google/gemma-3n-E2B-it')
text_only = TextOnlyModelWrapper('Qwen/Qwen2.5-1.5B-Instruct')

# Reuse stage_inputs and run_one from notebook 04
def stage_inputs(stage, sample):
    vid = sample['video_id']
    paths = {
        'video':      f'data/videos/{vid}.mp4',
        'audio':      f'data/audio/{vid}.wav',
        'silent':     f'data/silent_video/{vid}_silent.mp4',
        'transcript': f'data/transcripts/{vid}.txt',
        # Mismatched transcript is keyed by qa_id (a single video may have
        # multiple QAs that need DIFFERENT mismatched donors)
        'mismatched': f'data/transcripts_mismatched/{sample["qa_id"]}.txt',
    }
    routes = {
        'S0': dict(video_path=None, audio_path=None, extra={}),
        'S1': dict(video_path=None, audio_path=paths['audio'], extra={}),
        'S2': dict(video_path=paths['silent'], audio_path=None, extra={}),
        'S3': dict(video_path=paths['video'], audio_path=paths['audio'], extra={}),
        'S4': dict(video_path=paths['video'], audio_path=paths['audio'],
                   extra={'transcript': load_transcript(paths['transcript'])}),
        'S5': dict(video_path=paths['video'], audio_path=paths['audio'],
                   extra={'mismatched_transcript': load_transcript(paths['mismatched'])}),
    }
    return routes[stage]

def run_one(stage, sample, model):
    inp = stage_inputs(stage, sample)
    prompt = STAGE_BUILDERS[stage](sample['question'], sample['options'], **inp['extra'])
    resp = model.generate(prompt, video_path=inp['video_path'], audio_path=inp['audio_path'])
    answer, conf = parse_answer_confidence(resp.text)
    attr = parse_attribution(resp.text) if stage == 'S3' else None
    reason = parse_reason(resp.text) if stage == 'S3' else None
    return {
        'video_id': sample['video_id'], 'task_type': sample['task_type'], 'stage': stage,
        'ground_truth': sample['answer'], 'predicted_answer': answer,
        'verbalized_confidence': conf, 'answer_logprobs': resp.answer_logprobs,
        'attribution': attr, 'attribution_reason': reason, 'raw_response': resp.text,
    }

OUT_DIR = 'results/raw_predictions/gemma'
os.makedirs(OUT_DIR, exist_ok=True)
"""),
    ("code", """def run_stage(stage, samples, model, checkpoint_every=20):
    out_path = f'{OUT_DIR}/{stage}.json'
    done = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = json.load(f)
    done_ids = {r['qa_id'] for r in done}
    todo = [s for s in samples if s['qa_id'] not in done_ids]
    if not todo:
        return done
    records = list(done)
    t0 = time.time()
    for i, s in enumerate(todo):
        try:
            rec = run_one(stage, s, model)
        except Exception as e:
            rec = {'video_id': s['video_id'], 'task_type': s['task_type'], 'stage': stage,
                   'ground_truth': s['answer'], 'predicted_answer': None,
                   'verbalized_confidence': None, 'answer_logprobs': None,
                   'attribution': None, 'attribution_reason': None,
                   'raw_response': f'<ERROR: {e}>'}
        records.append(rec)
        if (i + 1) % checkpoint_every == 0 or (i + 1) == len(todo):
            with open(out_path, 'w') as f:
                json.dump(records, f, indent=2)
            print(f'  {stage}: {len(records)}/{len(samples)}  ({(time.time()-t0)/(i+1):.1f}s/sample)')
    return records

# Skip S1 if audio-only doesn't work; we'll detect this empirically
STAGES = ['S0', 'S2', 'S3', 'S4', 'S5']  # try without S1 first
print('Running Gemma-3n stages: ', STAGES)
for stage in STAGES:
    print(f'\\n=== {stage} ===')
    model = text_only if stage == 'S0' else gemma
    run_stage(stage, samples, model)
"""),
    ("markdown", "**Done.** Proceed to `06_analysis.ipynb`."),
]


# ────────────────────────────────────────────────────────────────────
# Notebook 06: Analysis
# ────────────────────────────────────────────────────────────────────
NB_06 = [
    ("markdown", """# 06 — Analysis & Figures

Reads `results/raw_predictions/{model}/S*.json` and computes:
- Per-task accuracy table (S0–S5)
- Confidence summaries (verbalized + logprob-based)
- ECE per stage
- Confidence Drop (ΔConf) under modality ablation
- Attribution Faithfulness Score (AFS) — the headline metric
- Transcript Injection Bias (TIB)
- Lexical Override Rate (LOR)
- Answer flip rates between stage pairs

Outputs land in `results/metrics/` (JSONs) and `results/figures/` (PNGs).
"""),
    ("code", """import os, sys, json
REPO = '/content/omnimodel-research'
if os.path.exists(REPO):
    %cd $REPO
    sys.path.insert(0, REPO)

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import (
    accuracy_per_task, verbalized_confidence_stats, logprob_confidence_stats,
    expected_calibration_error, confidence_drop, attribution_faithfulness,
    transcript_injection_bias, lexical_override_rate, answer_flip_rate,
)
"""),
    ("code", """MODEL = 'qwen'   # change to 'gemma' to analyze the cross-model run
PRED_DIR = f'results/raw_predictions/{MODEL}'
METRICS_DIR = f'results/metrics/{MODEL}'
FIG_DIR = f'results/figures/{MODEL}'
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

stages = {}
for stage in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']:
    p = f'{PRED_DIR}/{stage}.json'
    if os.path.exists(p):
        with open(p) as f:
            stages[stage] = json.load(f)
        print(f'Loaded {stage}: {len(stages[stage])} records')
    else:
        print(f'MISSING: {stage}')
"""),
    ("markdown", "## 1. Accuracy table"),
    ("code", """acc_table = {s: accuracy_per_task(rs) for s, rs in stages.items()}
with open(f'{METRICS_DIR}/accuracy_per_task.json', 'w') as f:
    json.dump(acc_table, f, indent=2)

import pandas as pd
df = pd.DataFrame(acc_table).T
print(df.to_string(float_format='%.3f'))
"""),
    ("markdown", "## 2. Confidence summaries"),
    ("code", """conf_table = {}
ece_table = {}
for s, rs in stages.items():
    conf_table[s] = {
        'verbalized': verbalized_confidence_stats(rs),
        'logprob': logprob_confidence_stats(rs),
    }
    ece_table[s] = {
        'logprob': expected_calibration_error(rs, source='logprob'),
        'verbalized': expected_calibration_error(rs, source='verbalized'),
    }

with open(f'{METRICS_DIR}/confidence.json', 'w') as f:
    json.dump(conf_table, f, indent=2, default=str)
with open(f'{METRICS_DIR}/ece.json', 'w') as f:
    json.dump(ece_table, f, indent=2, default=str)

print('ECE per stage:')
for s, vals in ece_table.items():
    print(f'  {s}: logprob={vals["logprob"]:.3f}  verbalized={vals["verbalized"]:.3f}'
          if vals['logprob'] is not None and vals['verbalized'] is not None else f'  {s}: --')
"""),
    ("markdown", "## 3. Confidence Drop under modality ablation"),
    ("code", """if 'S3' in stages:
    drops = {}
    if 'S2' in stages:
        drops['audio_removal_S3_minus_S2'] = confidence_drop(stages['S3'], stages['S2'], source='verbalized')
    if 'S1' in stages:
        drops['visual_removal_S3_minus_S1'] = confidence_drop(stages['S3'], stages['S1'], source='verbalized')
    with open(f'{METRICS_DIR}/confidence_drops.json', 'w') as f:
        json.dump(drops, f, indent=2)
    print(json.dumps(drops, indent=2))
"""),
    ("markdown", "## 4. Attribution Faithfulness Score (HEADLINE METRIC)"),
    ("code", """if {'S1', 'S2', 'S3'}.issubset(stages):
    afs = attribution_faithfulness(stages['S3'], s2_records=stages['S2'], s1_records=stages['S1'])
    with open(f'{METRICS_DIR}/attribution_faithfulness.json', 'w') as f:
        json.dump(afs, f, indent=2)
    print('Attribution Faithfulness Score per task:')
    for task, d in afs.items():
        if d.get('score') is not None:
            print(f'  {task}: {d["score"]:.2f}  (faithful={d["faithful"]}  confab={d["confabulated"]}  unparsed={d["unparseable"]})')
"""),
    ("markdown", "## 5. Transcript Injection Bias and Lexical Override Rate"),
    ("code", """if 'S3' in stages and 'S4' in stages:
    tib = transcript_injection_bias(stages['S3'], stages['S4'])
    with open(f'{METRICS_DIR}/transcript_injection_bias.json', 'w') as f:
        json.dump(tib, f, indent=2)
    print('TIB (positive = transcripts hurt):')
    for t, v in tib.items():
        print(f'  {t}: {v:+.3f}')

if 'S3' in stages and 'S5' in stages:
    lor = lexical_override_rate(stages['S3'], stages['S5'])
    with open(f'{METRICS_DIR}/lexical_override_rate.json', 'w') as f:
        json.dump(lor, f, indent=2)
    print('\\nLOR (fraction of S3-correct samples flipped by mismatched transcript):')
    for t, d in lor.items():
        if d.get('lor') is not None:
            print(f'  {t}: {d["lor"]:.2f}  (flipped={d["flipped"]}  stayed={d["stayed"]})')
"""),
    ("markdown", "## 6. Answer flip rates"),
    ("code", """flip_rates = {}
pairs = [('S3','S2'), ('S3','S1'), ('S3','S4'), ('S3','S5'), ('S3','S0')]
for a, b in pairs:
    if a in stages and b in stages:
        flip_rates[f'{a}_vs_{b}'] = answer_flip_rate(stages[a], stages[b])
with open(f'{METRICS_DIR}/answer_flip_rates.json', 'w') as f:
    json.dump(flip_rates, f, indent=2)
"""),
    ("markdown", "## 7. Figures"),
    ("code", """# Accuracy heatmap (stages × tasks)
import pandas as pd
df = pd.DataFrame(acc_table).T
df = df.drop(columns=['OVERALL'], errors='ignore')
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(df.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns)
ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        v = df.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9, color='black')
plt.colorbar(im, ax=ax, label='Accuracy')
plt.title(f'{MODEL.upper()} accuracy: stages × tasks')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/accuracy_heatmap.png', dpi=150)
plt.show()
"""),
    ("code", """# AFS bar chart
if {'S1','S2','S3'}.issubset(stages):
    tasks_only = [k for k in afs.keys() if k != 'OVERALL']
    scores = [afs[t]['score'] for t in tasks_only if afs[t]['score'] is not None]
    labels = [t for t in tasks_only if afs[t]['score'] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, scores, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='random / chance')
    ax.set_ylabel('Attribution Faithfulness Score')
    ax.set_title(f'{MODEL.upper()}: AFS by task (lower = more confabulation)')
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/afs_by_task.png', dpi=150)
    plt.show()
"""),
    ("code", """# LOR vs TIB scatter — both measure text/audio trust at the per-task level
if 'S4' in stages and 'S5' in stages and 'S3' in stages:
    tasks_in = [t for t in tib if t != 'OVERALL' and lor.get(t, {}).get('lor') is not None]
    fig, ax = plt.subplots(figsize=(6, 5))
    for t in tasks_in:
        ax.scatter(tib[t], lor[t]['lor'], s=80)
        ax.annotate(t, (tib[t], lor[t]['lor']), xytext=(5,5), textcoords='offset points')
    ax.axhline(0, color='gray', alpha=0.3)
    ax.axvline(0, color='gray', alpha=0.3)
    ax.set_xlabel('TIB = Acc(S3) - Acc(S4)  (>0 = transcripts hurt)')
    ax.set_ylabel('LOR = fraction flipped by mismatched transcript')
    ax.set_title(f'{MODEL.upper()}: lexical bias signals by task')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/tib_vs_lor.png', dpi=150)
    plt.show()
"""),
    ("markdown", """## What to look for

- **Headline:** Is overall AFS substantially below 1.0? If yes, this is the confabulation finding. Best if AFS varies by task — e.g., HIGH on AIE (where audio is genuinely needed) and LOW on AVCM (where the model can fake it).
- **Calibration:** Does verbalized confidence saturate near 100? Is logprob ECE > 0.1?
- **Lexical override:** LOR > 0.2 on audio-essential tasks would be strong evidence the model doesn't trust its own audio encoder.
- **TIB shape:** Predicted TIB > 0 for ACC/AEL (audio-essential) and TIB < 0 for AVTM (text-helps).
"""),
]


def main():
    write_nb('01_setup_and_download.ipynb', NB_01)
    write_nb('02_preprocess.ipynb', NB_02)
    write_nb('03_pilot.ipynb', NB_03)
    write_nb('04_main_eval_qwen.ipynb', NB_04)
    write_nb('05_main_eval_gemma.ipynb', NB_05)
    write_nb('06_analysis.ipynb', NB_06)


if __name__ == '__main__':
    main()
