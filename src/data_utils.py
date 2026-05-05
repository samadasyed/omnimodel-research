"""AVUT annotation loading, balanced subsampling, video acquisition.

Schema verified against `tsinghua-ee/AVUTBenchmark/AV_Human_data.json`:

    {
      "video_id": int,                   # repeats across QA pairs of same video
      "url": str,                        # YouTube URL (informational only)
      "video_type": str,                 # "vlog" / "TED speech" / etc.
      "task_type": str,                  # human-readable name (NOT a short code)
      "question": str,
      "option_A" / "option_B" / "option_C" / "option_D": str,
      "answer": "A" | "B" | "C" | "D",
      "video_path": "data/<youtube_id>.mp4",
      "QA_id": int,                      # unique per question — primary key
    }

Important: ``(video_id, task_type)`` is NOT unique (~5% of videos have two
QAs of the same task type). We use ``QA_id`` as the unique cross-stage
matching key.

Re-key: we pull the YouTube ID out of ``video_path`` and put it in
``video_id`` (string) so it matches downloaded mp4 filenames.

Task short codes match Jeff's ``field_map.TASK_CODES`` so cross-model
analysis joins on task_code without renaming.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ─── Task name <-> short code ───────────────────────────────────────
# Match Jeff's codes exactly so cross-model tables join cleanly.
TASK_NAME_TO_CODE = {
    "Audio Information Extraction": "AIE",
    "Audio Content Counting":       "ACC",
    "Audio Event Location":         "AEL",
    "Audio Character Matching":     "AVCM",
    "Audio Object Matching":        "AVOM",
    "Audio OCR Matching":           "AVTM",   # NB: 'AVTM' (Text Matching), Jeff's spelling
}
TASK_CODE_TO_NAME = {v: k for k, v in TASK_NAME_TO_CODE.items()}

# Categorize tasks by which modality should carry the discriminating signal.
# Used in interpreting AFS, TIB, LOR by task category.
TASK_AUDIO_ROLE = {
    "AIE":  "audio_essential",
    "ACC":  "audio_essential",
    "AEL":  "audio_essential",
    "AVCM": "audio_visual_binding",
    "AVOM": "audio_visual_binding",
    "AVTM": "audio_visual_binding",
}

TARGET_TASKS = tuple(TASK_NAME_TO_CODE.keys())
TARGET_TASK_CODES = tuple(TASK_NAME_TO_CODE.values())


# ─── Schema parsing ─────────────────────────────────────────────────
_YT_PATH_RE = re.compile(r"data/([^/.]+)\.mp4")


def _extract_youtube_id(entry: Dict) -> Optional[str]:
    vp = entry.get("video_path") or ""
    m = _YT_PATH_RE.search(vp)
    if m:
        return m.group(1)

    url = entry.get("url") or ""
    for pat in (r"shorts/([A-Za-z0-9_-]{6,})",
                r"v=([A-Za-z0-9_-]{6,})",
                r"youtu\.be/([A-Za-z0-9_-]{6,})"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _normalize_options(entry: Dict) -> Optional[Dict[str, str]]:
    opts = entry.get("options")
    if isinstance(opts, dict):
        return {k.upper(): str(v) for k, v in opts.items() if k.upper() in "ABCD"}
    if isinstance(opts, list) and len(opts) >= 4:
        return {letter: str(opts[i]) for i, letter in enumerate("ABCD")}
    cand = {}
    for letter in "ABCD":
        v = entry.get(f"option_{letter}") or entry.get(letter)
        if v is not None:
            cand[letter] = str(v)
    return cand if len(cand) == 4 else None


def parse_annotation(entry: Dict) -> Dict:
    """Convert one AVUT entry to canonical pipeline form."""
    return {
        "qa_id":      entry.get("QA_id") or entry.get("qa_id"),
        "video_id":   _extract_youtube_id(entry),
        "task_type":  entry.get("task_type"),
        "task_code":  TASK_NAME_TO_CODE.get(entry.get("task_type")),
        "video_type": entry.get("video_type"),
        "question":   entry.get("question"),
        "options":    _normalize_options(entry),
        "answer":     (entry.get("answer") or entry.get("correct_answer") or "").strip().upper(),
        "url":        entry.get("url"),
        "video_path": entry.get("video_path"),
    }


def load_annotations(annotation_path: str) -> List[Dict]:
    """Load AV_Human_data.json and return canonical entries with all required
    fields populated.
    """
    with open(annotation_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for wrapper in ("data", "annotations", "entries", "items"):
            if wrapper in raw and isinstance(raw[wrapper], list):
                raw = raw[wrapper]
                break
        else:
            raw = list(raw.values())

    out, skipped = [], 0
    for e in raw:
        if not isinstance(e, dict):
            skipped += 1
            continue
        p = parse_annotation(e)
        if not all(p.get(k) for k in ("qa_id", "video_id", "task_type",
                                       "question", "options", "answer")):
            skipped += 1
            continue
        out.append(p)
    if skipped:
        print(f"[load_annotations] skipped {skipped} malformed entries")
    return out


def filter_to_target_tasks(entries: Iterable[Dict]) -> List[Dict]:
    return [e for e in entries if e["task_type"] in TARGET_TASKS]


def filter_to_available_videos(
    entries: Iterable[Dict], video_dir: str, ext: str = ".mp4"
) -> List[Dict]:
    return [
        e for e in entries
        if os.path.exists(os.path.join(video_dir, f"{e['video_id']}{ext}"))
    ]


# ─── Sampling ──────────────────────────────────────────────────────
def balanced_subsample(
    entries: List[Dict],
    n_per_task: int,
    seed: int = 42,
    tasks: tuple = TARGET_TASKS,
    blocklist: Optional[set] = None,
) -> List[Dict]:
    """Pick n_per_task entries per task type. Dedup by qa_id, skip blocklisted
    videos. If a task pool has fewer entries than requested, take all available
    and warn (rather than padding from elsewhere).

    Sampling style matches Jeff's `02_pick_pilot_sample.py` so the same seed
    produces the same picks across models.
    """
    rng = random.Random(seed)
    blocklist = blocklist or set()

    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        if e["task_type"] in tasks:
            by_task[e["task_type"]].append(e)

    picked, seen_qa_ids = [], set()
    # Sort tasks by their short code for deterministic processing order
    task_order = sorted(tasks, key=lambda t: TASK_NAME_TO_CODE.get(t, t))

    for task in task_order:
        candidates = by_task.get(task, []).copy()
        rng.shuffle(candidates)
        taken = 0
        for e in candidates:
            qa_id = e.get("qa_id")
            video_id = e.get("video_id")
            if qa_id is None or video_id is None:
                continue
            if video_id in blocklist:
                continue
            if qa_id in seen_qa_ids:
                continue
            picked.append(e)
            seen_qa_ids.add(qa_id)
            taken += 1
            if taken >= n_per_task:
                break
        if taken < n_per_task:
            print(f"[balanced_subsample] {TASK_NAME_TO_CODE.get(task, task)}: "
                  f"requested {n_per_task} but only {taken} available")
    return picked


def task_distribution(entries: Iterable[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e["task_type"]] += 1
    return dict(counts)


def task_distribution_by_code(entries: Iterable[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        code = e.get("task_code") or TASK_NAME_TO_CODE.get(e["task_type"], e["task_type"])
        counts[code] += 1
    return dict(counts)


# ─── Video acquisition (HuggingFace dataset, NOT yt-dlp) ───────────
HF_REPO_ID = "tsinghua-ee/AVUTBenchmark"


def download_videos_from_hf(
    samples: List[Dict],
    video_dir: str,
    repo_id: str = HF_REPO_ID,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """Download video mp4s from the AVUT HF dataset to video_dir.

    Tries `<id>.mp4` first, then `<id>_fullsize.mp4`. Returns counts dict:
    {"ok", "exists", "fail", "no_match"}.

    Why HF and not yt-dlp: the AVUT authors uploaded the videos directly,
    so they're guaranteed available, no rate limits, no DRM. Jeff's repo
    proved this path is reliable.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    os.makedirs(video_dir, exist_ok=True)
    api = HfApi()
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

    counts = {"ok": 0, "exists": 0, "fail": 0, "no_match": 0}
    for entry in samples:
        stem = entry["video_id"]
        out = os.path.join(video_dir, f"{stem}.mp4")

        if skip_existing and os.path.exists(out) and os.path.getsize(out) > 0:
            counts["exists"] += 1
            continue

        repo_filename = next(
            (n for n in (f"{stem}.mp4", f"{stem}_fullsize.mp4") if n in repo_files),
            None,
        )
        if repo_filename is None:
            counts["no_match"] += 1
            print(f"  ✗ {stem}: not in repo")
            continue

        try:
            cached = hf_hub_download(
                repo_id=repo_id, filename=repo_filename, repo_type="dataset",
            )
            shutil.copy(cached, out)
            counts["ok"] += 1
        except (EntryNotFoundError, HfHubHTTPError) as e:
            counts["fail"] += 1
            print(f"  ✗ {stem}: {str(e)[:80]}")

    return counts


# ─── Blocklist (videos that fail preprocessing) ────────────────────
def load_blocklist(path: str) -> set:
    """Return the set of video_ids on the blocklist (or empty if file absent)."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.get("video_ids", []))
    return set()
