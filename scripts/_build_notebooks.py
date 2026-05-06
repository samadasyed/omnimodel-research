"""Build the AVUT diagnostic notebooks (Colab-ready).

Run from repo root:
    python scripts/_build_notebooks.py

Produces six notebooks:
    01_setup_and_download.ipynb
    02_preprocess.ipynb
    03_pilot.ipynb
    04_main_eval_gemma.ipynb
    05_main_eval_qwen.ipynb
    06_analysis.ipynb

The builder pattern keeps notebooks reviewable in git (we generate them
from typed cell lists rather than handcrafting JSON). Stage scheme
matches Jeff's repo (jjwang8/639_avut) so cross-model numbers are
directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)

Cell = Tuple[str, str]  # (cell_type, source)


def make_notebook(cells: List[Cell]) -> dict:
    nb_cells = []
    for cell_type, source in cells:
        cell = {"cell_type": cell_type, "metadata": {}, "source": source}
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


# ─── Common Colab bootstrap (used by every notebook) ─────────────
BOOTSTRAP = '''# ─── Colab bootstrap ────────────────────────────────────────
# Mount Drive (caches model weights + videos across sessions),
# clone the repo, install dependencies, and configure the cache dirs.
import os, sys, subprocess, pathlib

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/avut"
    REPO_DIR = "/content/omnimodel-research"
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    if not os.path.exists(REPO_DIR):
        subprocess.run([
            "git", "clone",
            "https://github.com/samadasyed/omnimodel-research.git",
            REPO_DIR,
        ], check=True)
else:
    DRIVE_ROOT = os.path.expanduser("~/avut")
    REPO_DIR = str(pathlib.Path.cwd())
    os.makedirs(DRIVE_ROOT, exist_ok=True)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Persistent storage layout (data + model caches under DRIVE_ROOT)
DATA_DIR        = os.path.join(DRIVE_ROOT, "data")
VIDEO_DIR       = os.path.join(DATA_DIR, "videos")
AUDIO_DIR       = os.path.join(DATA_DIR, "audio")
SILENT_DIR      = os.path.join(DATA_DIR, "silent")
TRANSCRIPT_DIR  = os.path.join(DATA_DIR, "transcripts")
ANNOTATION_DIR  = os.path.join(DATA_DIR, "annotations")
RESULTS_DIR     = os.path.join(DRIVE_ROOT, "results")
RAW_PRED_DIR    = os.path.join(RESULTS_DIR, "raw_predictions")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
HF_CACHE        = os.path.join(DRIVE_ROOT, ".cache", "hf")
WHISPER_CACHE   = os.path.join(DRIVE_ROOT, ".cache", "whisper")

for d in [VIDEO_DIR, AUDIO_DIR, SILENT_DIR, TRANSCRIPT_DIR,
          ANNOTATION_DIR, RAW_PRED_DIR, METRICS_DIR, HF_CACHE, WHISPER_CACHE]:
    os.makedirs(d, exist_ok=True)

# Redirect HF cache to Drive so we don't redownload weights each session
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_CACHE

print(f"Repo:       {REPO_DIR}")
print(f"Drive root: {DRIVE_ROOT}")
print(f"HF cache:   {HF_CACHE}")
'''


INSTALL_DEPS_AV = '''# ─── Install model dependencies ──────────────────────────
# transformers (Gemma-3n preview support landed in 4.51), accelerate,
# soundfile for audio I/O, openai-whisper for ASR. cv2 (opencv-python)
# is preinstalled on Colab and used for frame sampling.
%pip install -q -U "transformers>=4.51.0" "accelerate>=0.30" \
    "huggingface-hub>=0.24" "soundfile>=0.12" scipy
%pip install -q openai-whisper
# Qwen-Omni preview branch is only needed for notebook 05; install there.

# Gemma-3n is GATED on HuggingFace — you must accept the license at
# https://huggingface.co/google/gemma-3n-E2B-it and authenticate.
# In Colab: Tools → Secrets → add HF_TOKEN, then:
import os
if "HF_TOKEN" in os.environ or "HUGGINGFACE_TOKEN" in os.environ:
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    print("HF authenticated")
else:
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            from huggingface_hub import login
            login(token=tok)
            print("HF authenticated (from Colab Secrets)")
        else:
            print("WARN: no HF_TOKEN — Gemma-3n download will fail. "
                  "Add HF_TOKEN to Colab Secrets first.")
    except Exception:
        print("WARN: not in Colab and no HF_TOKEN env var. "
              "Run `huggingface-cli login` before loading Gemma.")
'''


# ────────────────────────────────────────────────────────────────────
# 01: Setup + download
# ────────────────────────────────────────────────────────────────────
NB_01 = [
    ("markdown", """# 01 — Setup, Annotations, and Video Acquisition

Prepares the AVUT-Human dataset.

**What it does**
1. Mounts Drive, clones repo, installs deps.
2. Downloads `AV_Human_data.json` from `tsinghua-ee/AVUTBenchmark`.
3. Picks a balanced sample (configurable `n_per_task`).
4. Downloads videos directly from the HF dataset (NOT yt-dlp — the AVUT
   authors host the mp4s themselves, so it's much more reliable).

**Runtime:** ~15-30 min on Colab depending on sample size and bandwidth.
**GPU:** not needed for this notebook; CPU runtime is fine.

Resume-safe: rerun to skip already-downloaded files.
"""),
    ("code", BOOTSTRAP),
    ("code", INSTALL_DEPS_AV),
    ("markdown", "## Download annotations"),
    ("code", '''from huggingface_hub import hf_hub_download
import shutil

annotation_dst = os.path.join(ANNOTATION_DIR, "AV_Human_data.json")
if not os.path.exists(annotation_dst):
    src = hf_hub_download(
        repo_id="tsinghua-ee/AVUTBenchmark",
        filename="AV_Human_data.json",
        repo_type="dataset",
    )
    shutil.copy(src, annotation_dst)
    print(f"Saved {annotation_dst}")
else:
    print(f"Cached: {annotation_dst}")
'''),
    ("markdown", "## Inspect schema"),
    ("code", '''from src import data_utils

entries = data_utils.load_annotations(annotation_dst)
print(f"Loaded {len(entries)} valid entries")
print()
print("Task distribution (full set):")
for code, n in sorted(data_utils.task_distribution_by_code(entries).items()):
    print(f"  {code:5s}  {n:5d}")
print()
print("Sample entry:", entries[0])
'''),
    ("markdown", "## Pick a balanced sample\n\n"
                 "Set `N_PER_TASK` to control sample size. 20 is a quick pilot; "
                 "100 is a good Colab default; 280+ matches Jeff's full run."),
    ("code", '''import json

N_PER_TASK = 100   # change me — 20 pilot, 100 default, 280+ full

samples = data_utils.balanced_subsample(
    entries, n_per_task=N_PER_TASK, seed=42,
)
print(f"Picked {len(samples)} samples ({N_PER_TASK}/task target)")
print()
for code, n in sorted(data_utils.task_distribution_by_code(samples).items()):
    print(f"  {code:5s}  {n}")

manifest_path = os.path.join(DATA_DIR, "sample_manifest.json")
with open(manifest_path, "w") as f:
    json.dump(samples, f, indent=2)
print(f"\\nManifest: {manifest_path}")
'''),
    ("markdown", "## Download videos from HuggingFace dataset"),
    ("code", '''counts = data_utils.download_videos_from_hf(samples, VIDEO_DIR)
print(counts)

# Filter manifest to only samples whose video downloaded successfully
samples_avail = data_utils.filter_to_available_videos(samples, VIDEO_DIR)
print(f"\\nAvailable videos: {len(samples_avail)} / {len(samples)}")

# Save the filtered manifest — that's what subsequent notebooks read
manifest_avail_path = os.path.join(DATA_DIR, "sample_manifest_available.json")
with open(manifest_avail_path, "w") as f:
    json.dump(samples_avail, f, indent=2)
print(f"Available-only manifest: {manifest_avail_path}")
'''),
    ("markdown", "Next: run `02_preprocess.ipynb`."),
]


# ────────────────────────────────────────────────────────────────────
# 02: Preprocess (ffmpeg + Whisper + mismatched pairs)
# ────────────────────────────────────────────────────────────────────
NB_02 = [
    ("markdown", """# 02 — Preprocess: audio extraction, silent video, ASR transcripts

For each video in the manifest:
1. Extract 16-kHz mono audio (`.wav`) — used by S2/S4-S8.
2. Strip audio to make a silent video — used by S3.
3. Whisper-small transcribe → text — used by S5 (matched) and S7 (donor pool).
4. Build the qa_id → donor video mismatched-transcript pairing — used by S7.

**Runtime:** ~5-15 min for 600 samples (Whisper on GPU). Cached: rerun to skip.
**GPU:** strongly recommended for Whisper.
"""),
    ("code", BOOTSTRAP),
    ("code", INSTALL_DEPS_AV),
    ("code", '''import json, subprocess
from pathlib import Path

# ffmpeg ships with imageio[ffmpeg] but Colab has it system-wide too.
subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
print("ffmpeg OK")

manifest_path = os.path.join(DATA_DIR, "sample_manifest_available.json")
with open(manifest_path) as f:
    samples = json.load(f)
print(f"Manifest: {len(samples)} samples")
'''),
    ("markdown", "## Audio extraction + silent video"),
    ("code", '''def extract_audio(video, out_wav):
    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
        return True
    r = subprocess.run([
        "ffmpeg", "-y", "-i", video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_wav,
    ], capture_output=True)
    return r.returncode == 0 and os.path.exists(out_wav)


def strip_audio(video, out_silent):
    if os.path.exists(out_silent) and os.path.getsize(out_silent) > 0:
        return True
    r = subprocess.run([
        "ffmpeg", "-y", "-i", video, "-an", "-c:v", "copy", out_silent,
    ], capture_output=True)
    return r.returncode == 0 and os.path.exists(out_silent)


stats = {"audio_ok": 0, "silent_ok": 0, "fail": 0}
for i, s in enumerate(samples, 1):
    vid = s["video_id"]
    video = os.path.join(VIDEO_DIR, f"{vid}.mp4")
    audio_out = os.path.join(AUDIO_DIR, f"{vid}.wav")
    silent_out = os.path.join(SILENT_DIR, f"{vid}_silent.mp4")
    if not os.path.exists(video):
        stats["fail"] += 1
        continue
    if extract_audio(video, audio_out):
        stats["audio_ok"] += 1
    else:
        stats["fail"] += 1
    if strip_audio(video, silent_out):
        stats["silent_ok"] += 1
    if i % 50 == 0:
        print(f"  [{i}/{len(samples)}] audio={stats['audio_ok']} silent={stats['silent_ok']}")
print(stats)
'''),
    ("markdown", "## Whisper transcripts (small)"),
    ("code", '''import whisper as whisper_lib
from src.transcript_utils import transcribe_with_whisper_python

print("Loading Whisper-small...")
whisper_model = whisper_lib.load_model("small", download_root=WHISPER_CACHE)
print("OK")

t_stats = {"ok": 0, "fail": 0}
for i, s in enumerate(samples, 1):
    vid = s["video_id"]
    audio_path = os.path.join(AUDIO_DIR, f"{vid}.wav")
    out_txt = os.path.join(TRANSCRIPT_DIR, f"{vid}.txt")
    if not os.path.exists(audio_path):
        t_stats["fail"] += 1
        continue
    try:
        transcribe_with_whisper_python(audio_path, out_txt, whisper_model)
        t_stats["ok"] += 1
    except Exception as e:
        print(f"  ✗ {vid}: {e}")
        t_stats["fail"] += 1
    if i % 50 == 0:
        print(f"  [{i}/{len(samples)}] {t_stats}")
print(t_stats)

# Free Whisper before model loads in subsequent notebooks
import gc, torch
del whisper_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''),
    ("markdown", "## Build mismatched-transcript pairing\n\n"
                 "For S7: each qa_id is paired with a *different* video in the "
                 "same task type. The donor's transcript is what we'll inject."),
    ("code", '''from src.transcript_utils import build_mismatched_pairs

pairs = build_mismatched_pairs(samples, seed=42)
mm_path = os.path.join(DATA_DIR, "mismatched_pairs.json")
with open(mm_path, "w") as f:
    json.dump({str(k): v for k, v in pairs.items()}, f, indent=2)
print(f"Mismatched-pair manifest: {mm_path}")
print(f"Pairings: {len(pairs)}")
print(f"Sample pairings (first 5):")
for qa_id, donor in list(pairs.items())[:5]:
    print(f"  qa_id={qa_id} → donor video {donor}")
'''),
    ("markdown", "Next: `03_pilot.ipynb` for a 24-sample sanity check, "
                 "or skip to `04_main_eval_gemma.ipynb` for the full run."),
]


# ────────────────────────────────────────────────────────────────────
# 03: Pilot (small, fast — confirms pipeline before main run)
# ────────────────────────────────────────────────────────────────────
NB_03 = [
    ("markdown", """# 03 — Pilot Run (24 samples)

A 4-per-task sanity check. Runs Gemma-3n-E2B-IT on stages S2-S4 only —
just enough to confirm: model loads, prompts elicit the bracket format,
parsing works, predictions look sensible.

**Runtime:** ~5-10 min on Colab L4.
"""),
    ("code", BOOTSTRAP),
    ("code", INSTALL_DEPS_AV),
    ("code", '''import json
from src import data_utils, stages
from src.model_utils import Gemma3nWrapper

manifest_path = os.path.join(DATA_DIR, "sample_manifest_available.json")
with open(manifest_path) as f:
    full_manifest = json.load(f)

# 4 per task → 24 samples
pilot = data_utils.balanced_subsample(full_manifest, n_per_task=4, seed=7)
print(f"Pilot samples: {len(pilot)}")
'''),
    ("code", '''# Load Gemma-3n
gemma = Gemma3nWrapper(model_name="google/gemma-3n-E2B-it")
print("Gemma loaded")
'''),
    ("code", '''# Run a few stages on the pilot — just to verify everything works
def get_paths(s):
    vid = s["video_id"]
    return {
        "video":  os.path.join(VIDEO_DIR, f"{vid}.mp4"),
        "silent": os.path.join(SILENT_DIR, f"{vid}_silent.mp4"),
        "audio":  os.path.join(AUDIO_DIR, f"{vid}.wav"),
        "transcript": os.path.join(TRANSCRIPT_DIR, f"{vid}.txt"),
    }

PILOT_STAGES = ["S2_audio_only", "S3_visual_only", "S4_full_av"]
pilot_results = {s: [] for s in PILOT_STAGES}

for stage in PILOT_STAGES:
    fn = stages.STAGE_REGISTRY[stage]["fn"]
    for s in pilot:
        try:
            row = fn(gemma, s, get_paths(s))
            pilot_results[stage].append(row)
        except Exception as e:
            print(f"  ✗ {stage} qa_id={s['qa_id']}: {e}")
    valid = sum(1 for r in pilot_results[stage] if r.get("predicted_answer"))
    print(f"  {stage}: {valid}/{len(pilot)} parseable")
'''),
    ("code", '''# Quick accuracy summary
from src import metrics
acc = metrics.compute_accuracy(pilot_results)
for stage, per_task in acc.items():
    print(f"\\n{stage}:")
    for task, d in per_task.items():
        if d.get("accuracy") is not None:
            print(f"  {task:35s} {d['accuracy']:.2f} ({d['n_correct']}/{d['n_valid']})")
'''),
    ("markdown", "If everything looks right above, proceed to `04_main_eval_gemma.ipynb`."),
]


# ────────────────────────────────────────────────────────────────────
# 04: Main Gemma run — full sample, all 8 stages
# ────────────────────────────────────────────────────────────────────
def main_eval_cells(model_label: str, model_var: str, model_loader_code: str,
                    stages_to_run_default: str, predictions_subdir: str,
                    runtime_estimate: str) -> List[Cell]:
    """Build the cells for a main-eval notebook (Gemma or Qwen)."""
    return [
        ("markdown", f"""# 04 — Main evaluation: {model_label}

Runs every stage in `DEFAULT_STAGE_ORDER` over the full available
manifest. Saves one JSON per stage to `{predictions_subdir}/`.

**Runtime:** {runtime_estimate}
**Resume-safe:** if Colab times out, rerun the cell — qa_ids already done
are skipped.
"""),
        ("code", BOOTSTRAP),
        ("code", INSTALL_DEPS_AV),
        ("code", f'''import json, time, gc
import torch
from src import data_utils, stages
from src.stages import STAGE_REGISTRY, DEFAULT_STAGE_ORDER

manifest_path = os.path.join(DATA_DIR, "sample_manifest_available.json")
with open(manifest_path) as f:
    samples = json.load(f)
print(f"Manifest: {{len(samples)}} samples")

PRED_DIR = os.path.join(RAW_PRED_DIR, "{predictions_subdir}")
os.makedirs(PRED_DIR, exist_ok=True)

# Mismatched-transcript pairings (built in notebook 02)
mm_path = os.path.join(DATA_DIR, "mismatched_pairs.json")
mismatched_pairs = {{}}
if os.path.exists(mm_path):
    with open(mm_path) as f:
        mismatched_pairs = {{int(k): v for k, v in json.load(f).items()}}
print(f"Mismatched pairings: {{len(mismatched_pairs)}}")
'''),
        ("code", f'''# Stages to run — change to a subset if you want to skip some
STAGES_TO_RUN = "{stages_to_run_default}".split(",")
print("Will run:", STAGES_TO_RUN)
'''),
        ("code", '''def get_paths(s, mismatched_pairs):
    vid = s["video_id"]
    paths = {
        "video":  os.path.join(VIDEO_DIR, f"{vid}.mp4"),
        "silent": os.path.join(SILENT_DIR, f"{vid}_silent.mp4"),
        "audio":  os.path.join(AUDIO_DIR, f"{vid}.wav"),
        "transcript": os.path.join(TRANSCRIPT_DIR, f"{vid}.txt"),
    }
    donor = mismatched_pairs.get(s["qa_id"])
    if donor:
        paths["mismatched_transcript"] = os.path.join(TRANSCRIPT_DIR, f"{donor}.txt")
        paths["donor_video_id"] = donor
    return paths


def stage_path(stage_name):
    return os.path.join(PRED_DIR, f"{stage_name}.json")


def load_existing(stage_name):
    p = stage_path(stage_name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def save_rows(stage_name, rows):
    with open(stage_path(stage_name), "w") as f:
        json.dump(rows, f, indent=2)


def already_done_ids(stage_name):
    return {r["qa_id"] for r in load_existing(stage_name)
            if r.get("predicted_answer") is not None}
'''),
        ("code", model_loader_code),
        ("code", f'''def run_stage(stage_name, samples, model, s4_lookup=None):
    spec = STAGE_REGISTRY[stage_name]
    fn = spec["fn"]
    done = already_done_ids(stage_name)
    rows = load_existing(stage_name)
    todo = [s for s in samples if s["qa_id"] not in done]
    print(f"\\n[{{stage_name}}] {{len(done)}} done, {{len(todo)}} to go")

    t0 = time.time()
    for i, s in enumerate(todo, 1):
        paths = get_paths(s, mismatched_pairs)
        try:
            if stage_name == "S6_attribution":
                s4_row = s4_lookup.get(s["qa_id"])
                if s4_row is None:
                    continue
                row = fn(model, s, paths, s4_row)
            else:
                row = fn(model, s, paths)
        except Exception as e:
            row = {{
                "qa_id": s["qa_id"], "video_id": s["video_id"],
                "task_type": s["task_type"], "task_code": s.get("task_code"),
                "stage": stage_name,
                "error": f"{{type(e).__name__}}: {{str(e)[:200]}}",
                "predicted_answer": None, "confidence": None,
            }}
        rows.append(row)
        if i % 25 == 0:
            save_rows(stage_name, rows)
            elapsed = time.time() - t0
            rate = elapsed / i
            print(f"  [{{i}}/{{len(todo)}}] {{rate:.1f}}s/sample, "
                  f"ETA {{rate * (len(todo) - i):.0f}}s")
    save_rows(stage_name, rows)
    print(f"  saved {{len(rows)}} rows to {{stage_path(stage_name)}}")
    return rows


# Phase A: text-only stage (small Qwen2.5-1.5B)
text_stages = [s for s in STAGES_TO_RUN if STAGE_REGISTRY[s]["model"] == "text"]
if text_stages:
    from src.model_utils import TextOnlyModelWrapper
    print("Loading Qwen2.5-1.5B-Instruct (text-only baseline)...")
    text_model = TextOnlyModelWrapper()
    for stage in text_stages:
        run_stage(stage, samples, text_model)
    del text_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Phase B: omnimodal stages
omni_stages = [s for s in STAGES_TO_RUN if STAGE_REGISTRY[s]["model"] == "omni"]
if omni_stages:
    {model_var} = load_omni_model()
    s4_lookup = {{}}
    for stage in omni_stages:
        if stage == "S6_attribution" and not s4_lookup:
            s4_rows = load_existing("S4_full_av")
            s4_lookup = {{r["qa_id"]: r for r in s4_rows}}
            if not s4_lookup:
                print("  S4_full_av not run yet; skipping S6_attribution.")
                continue
        run_stage(stage, samples, {model_var}, s4_lookup=s4_lookup)
    del {model_var}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\\nDone. Predictions in:", PRED_DIR)
'''),
        ("markdown", f"Next: `06_analysis.ipynb` to compute metrics on `{predictions_subdir}/`."),
    ]


GEMMA_LOADER = '''from src.model_utils import Gemma3nWrapper

def load_omni_model():
    return Gemma3nWrapper(
        model_name="google/gemma-3n-E2B-it",
        num_video_frames=8,
        audio_max_seconds=30.0,
    )
'''


QWEN_LOADER = '''# Qwen-Omni preview branch needs a specific transformers ref:
%pip install -q "git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"
%pip install -q "qwen-omni-utils[decord]>=0.0.4"

from src.model_utils import QwenOmniWrapper

def load_omni_model():
    return QwenOmniWrapper(model_name="Qwen/Qwen2.5-Omni-7B")
'''


NB_04 = main_eval_cells(
    model_label="Gemma-3n-E2B-IT (PRIMARY)",
    model_var="omni_model",
    model_loader_code=GEMMA_LOADER,
    stages_to_run_default="S1_text_only,S2_audio_only,S3_visual_only,S4_full_av,"
                          "S5_transcript_injected,S6_attribution,S7_mismatched_transcript",
    predictions_subdir="gemma_3n_e2b",
    runtime_estimate="~6-10 hr on Colab L4 for 600 samples × 7 stages.",
)


NB_05 = main_eval_cells(
    model_label="Qwen2.5-Omni-7B (cross-model robustness)",
    model_var="omni_model",
    model_loader_code=QWEN_LOADER,
    stages_to_run_default="S1_text_only,S2_audio_only,S3_visual_only,S4_full_av,"
                          "S5_transcript_injected,S6_attribution,S7_mismatched_transcript",
    predictions_subdir="qwen2_5_omni_7b",
    runtime_estimate="~10-14 hr on Colab A100 for 600 samples × 7 stages. Needs ≥40GB VRAM.",
)


# ────────────────────────────────────────────────────────────────────
# 06: Analysis — compute metrics + cross-model comparison
# ────────────────────────────────────────────────────────────────────
NB_06 = [
    ("markdown", """# 06 — Analysis: compute metrics, render figures, cross-model comparison

For each model run we have, compute every metric and save under
`results/metrics/<model>/`. Then load Jeff's published Qwen-Omni-7B
metrics from his repo for a side-by-side comparison.

**Runtime:** seconds.
**GPU:** not needed.
"""),
    ("code", BOOTSTRAP),
    ("code", '%pip install -q matplotlib pandas scipy'),
    ("code", '''import json, glob
from src import metrics

# Discover model runs we have predictions for
model_runs = sorted(glob.glob(os.path.join(RAW_PRED_DIR, "*")))
model_runs = [m for m in model_runs if os.path.isdir(m)]
print("Found prediction sets:")
for m in model_runs:
    print(f"  {m}")
'''),
    ("code", '''def load_stages(pred_dir):
    stages = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            stages[name] = json.load(f)
    return stages


all_metrics = {}
for run_dir in model_runs:
    name = os.path.basename(run_dir)
    print(f"\\n=== {name} ===")
    stages = load_stages(run_dir)
    for s, rows in stages.items():
        valid = sum(1 for r in rows if r.get("predicted_answer") is not None)
        print(f"  {s:30s} {len(rows):4d} rows  ({valid} parseable)")
    report = metrics.compute_all(stages)
    all_metrics[name] = report

    out_dir = os.path.join(METRICS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    for fname, data in report.items():
        with open(os.path.join(out_dir, f"{fname}.json"), "w") as f:
            json.dump(data, f, indent=2)
    print(f"  → metrics saved to {out_dir}")
'''),
    ("markdown", "## Headline summary"),
    ("code", '''def show_overall(report, name):
    print(f"\\n=== {name} — overall ===")
    acc = report["accuracy_per_task_per_stage"]
    print("Accuracy by stage:")
    for stage in sorted(acc.keys()):
        ov = acc[stage].get("OVERALL", {})
        if ov.get("accuracy") is not None:
            print(f"  {stage:30s} {ov['accuracy']:.3f}")
    afs_o = report["attribution_faithfulness"].get("OVERALL", {})
    if afs_o:
        print(f"\\nAFS overall: {afs_o.get('afs')}  "
              f"(F={afs_o.get('faithful')}, C={afs_o.get('confabulated')}, "
              f"T={afs_o.get('trivial')}, U={afs_o.get('unparseable')})")
    lor_o = report["lexical_override_rate"].get("OVERALL", {})
    if lor_o:
        print(f"LOR overall: {lor_o.get('lor')}  "
              f"(flipped={lor_o.get('flipped')}, stayed={lor_o.get('stayed')})")
    tib_overall = report["transcript_injection_bias"].get("OVERALL", {})
    if tib_overall:
        print(f"TIB overall: {tib_overall.get('tib')}")


for name, report in all_metrics.items():
    show_overall(report, name)
'''),
    ("markdown", "## Cross-model comparison: ours vs Jeff's published Qwen-Omni\n\n"
                 "Loads Jeff's metrics JSONs from his GitHub for a side-by-side."),
    ("code", '''import urllib.request

# Pull Jeff's published metrics directly from GitHub
JEFF_BASE = "https://raw.githubusercontent.com/jjwang8/639_avut/main/results/metrics"

def fetch_jeff(name):
    try:
        with urllib.request.urlopen(f"{JEFF_BASE}/{name}.json", timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None

jeff_acc = fetch_jeff("accuracy_per_task_per_stage")
jeff_afs = fetch_jeff("attribution_faithfulness")
jeff_tib = fetch_jeff("transcript_injection_bias")
jeff_drops = fetch_jeff("confidence_drops")

print("Loaded Jeff's metrics" if jeff_acc else "(skipping comparison)")
'''),
    ("code", '''import pandas as pd

def cross_model_accuracy_table(all_metrics, jeff_acc):
    rows = []
    if jeff_acc:
        for stage, per_task in jeff_acc.items():
            ov = per_task.get("OVERALL", {})
            rows.append({"model": "qwen2.5-omni-7b (jeff)", "stage": stage,
                         "accuracy": ov.get("accuracy"),
                         "n_valid": ov.get("n_valid")})
    for model_name, report in all_metrics.items():
        for stage, per_task in report["accuracy_per_task_per_stage"].items():
            ov = per_task.get("OVERALL", {})
            rows.append({"model": model_name, "stage": stage,
                         "accuracy": ov.get("accuracy"),
                         "n_valid": ov.get("n_valid")})
    return pd.DataFrame(rows).pivot(index="stage", columns="model",
                                     values="accuracy")


tbl = cross_model_accuracy_table(all_metrics, jeff_acc)
print("Overall accuracy by stage (cross-model):")
print(tbl.round(3).to_string())
'''),
    ("code", '''def cross_model_afs_table(all_metrics, jeff_afs):
    rows = []
    if jeff_afs:
        for task, d in jeff_afs.items():
            rows.append({"model": "qwen2.5-omni-7b (jeff)", "task": task,
                         "afs": d.get("afs"), "n_falsifiable": d.get("n_falsifiable")})
    for model_name, report in all_metrics.items():
        for task, d in report["attribution_faithfulness"].items():
            rows.append({"model": model_name, "task": task,
                         "afs": d.get("afs"), "n_falsifiable": d.get("n_falsifiable")})
    return pd.DataFrame(rows).pivot(index="task", columns="model", values="afs")


afs_tbl = cross_model_afs_table(all_metrics, jeff_afs)
print("AFS by task (cross-model):")
print(afs_tbl.round(3).to_string())
'''),
    ("markdown", "## Figures — accuracy heatmap, AFS by task, ΔConf by task"),
    ("code", '''import matplotlib.pyplot as plt
import numpy as np
from src.data_utils import TASK_NAME_TO_CODE

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_accuracy_heatmap(report, name):
    acc = report["accuracy_per_task_per_stage"]
    stages_order = [s for s in [
        "S1_text_only","S2_audio_only","S3_visual_only","S4_full_av",
        "S5_transcript_injected","S6_attribution","S7_mismatched_transcript",
        "S8_prosody"] if s in acc]
    tasks_order = list(TASK_NAME_TO_CODE.keys())
    mat = np.full((len(tasks_order), len(stages_order)), np.nan)
    for j, stage in enumerate(stages_order):
        for i, task in enumerate(tasks_order):
            v = acc.get(stage, {}).get(task, {}).get("accuracy")
            if v is not None:
                mat[i, j] = v
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(stages_order)))
    ax.set_xticklabels(stages_order, rotation=30, ha="right")
    ax.set_yticks(range(len(tasks_order)))
    ax.set_yticklabels([TASK_NAME_TO_CODE[t] for t in tasks_order])
    for i in range(len(tasks_order)):
        for j in range(len(stages_order)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label="accuracy")
    ax.set_title(f"Accuracy heatmap — {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"accuracy_heatmap_{name}.png"), dpi=140)
    plt.show()


for name, report in all_metrics.items():
    plot_accuracy_heatmap(report, name)
'''),
    ("markdown", "Figures saved to `RESULTS_DIR/figures/`. Use them in the paper."),
]


# ────────────────────────────────────────────────────────────────────
# Build everything
# ────────────────────────────────────────────────────────────────────
def main():
    write_nb("01_setup_and_download.ipynb", NB_01)
    write_nb("02_preprocess.ipynb", NB_02)
    write_nb("03_pilot.ipynb", NB_03)
    write_nb("04_main_eval_gemma.ipynb", NB_04)
    write_nb("05_main_eval_qwen.ipynb", NB_05)
    write_nb("06_analysis.ipynb", NB_06)
    print("\nAll notebooks written.")


if __name__ == "__main__":
    main()
