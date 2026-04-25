"""AVUT annotation loading and balanced subsampling.

Schema verified against `tsinghua-ee/AVUTBenchmark/AV_Human_data.json`
(downloaded 2026-04-25). Each raw entry has these fields:

    {
      "video_id": int,                   # repeats across QA pairs of the same video
      "url": str,                        # YouTube URL
      "video_type": str,                 # "vlog" / "TED speech" / etc.
      "task_type": str,                  # human-readable task name (NOT a short code)
      "question": str,
      "option_A" / "option_B" / "option_C" / "option_D": str,
      "answer": str,                     # "A" | "B" | "C" | "D"
      "video_path": str,                 # "data/{youtube_id}.mp4"
      "QA_id": int,                      # unique per question — primary key
    }

Important: ``(video_id, task_type)`` is NOT unique — ~5% of pairs repeat,
so we use ``QA_id`` as the unique cross-stage matching key.

We re-key entries to use the YouTube ID as ``video_id`` (string) since that's
what the downloaded mp4 filenames will use.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional


# Map from full AVUT task names → short codes (for display + paper tables)
TASK_NAME_TO_CODE = {
    "Audio Information Extraction": "AIE",
    "Audio Content Counting":       "ACC",
    "Audio Event Location":         "AEL",
    "Audio Character Matching":     "AVCM",   # speaker / who-said-this
    "Audio Object Matching":        "AVOM",   # what-happened-when
    "Audio OCR Matching":           "AOCR",   # speech ↔ on-screen text
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
    "AOCR": "audio_visual_binding",
}

# All six task types — every entry in AV_Human is one of these (no AVSM/AVDiar).
TARGET_TASKS = tuple(TASK_NAME_TO_CODE.keys())
TARGET_TASK_CODES = tuple(TASK_NAME_TO_CODE.values())


_YT_PATH_RE = re.compile(r"data/([^/.]+)\.mp4")


def _extract_youtube_id(entry: Dict) -> Optional[str]:
    """Pull the YouTube ID out of video_path, falling back to URL."""
    vp = entry.get("video_path") or ""
    m = _YT_PATH_RE.search(vp)
    if m:
        return m.group(1)

    url = entry.get("url") or ""
    # Common YouTube URL forms:
    #   https://www.youtube.com/shorts/<id>?si=...
    #   https://www.youtube.com/watch?v=<id>
    #   https://youtu.be/<id>
    for pat in (r"shorts/([A-Za-z0-9_-]{6,})",
                r"v=([A-Za-z0-9_-]{6,})",
                r"youtu\.be/([A-Za-z0-9_-]{6,})"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _normalize_options(entry: Dict) -> Optional[Dict[str, str]]:
    """Pull the four options into a {A,B,C,D} dict regardless of source format."""
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


def load_annotations(annotation_path: str) -> List[Dict]:
    """Load AVUT annotations and normalize to a clean internal schema.

    Output schema per entry:
        {
          "qa_id": int,                # PRIMARY KEY — unique per question
          "video_id": str,             # YouTube ID (used as mp4 filename)
          "task_type": str,            # full human-readable name
          "task_code": str,            # short code: "AIE", "ACC", ...
          "video_type": str,           # "vlog" / "TED speech" / etc.
          "question": str,
          "options": {"A":..., "B":..., "C":..., "D":...},
          "answer": str,
          "url": str,
        }
    """
    with open(annotation_path) as f:
        raw = json.load(f)
    entries = raw if isinstance(raw, list) else list(raw.values())

    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        yt_id = _extract_youtube_id(e)
        opts = _normalize_options(e)
        task = e.get("task_type")
        norm = {
            "qa_id": e.get("QA_id") or e.get("qa_id"),
            "video_id": yt_id,
            "task_type": task,
            "task_code": TASK_NAME_TO_CODE.get(task),
            "video_type": e.get("video_type"),
            "question": e.get("question"),
            "options": opts,
            "answer": (e.get("answer") or e.get("correct_answer") or "").strip().upper(),
            "url": e.get("url"),
        }
        # Skip entries missing critical fields
        if (norm["qa_id"] is not None and norm["video_id"] and norm["question"]
                and norm["options"] and norm["answer"]):
            out.append(norm)
    return out


def filter_to_target_tasks(entries: List[Dict]) -> List[Dict]:
    """Keep only the six MCQ task types we evaluate.

    Since AV_Human contains exactly these six tasks, this is a no-op for that
    file but is defensive against AV_Gemini or future releases.
    """
    return [e for e in entries if e["task_type"] in TARGET_TASKS]


def filter_to_available_videos(
    entries: List[Dict], video_dir: str, ext: str = ".mp4"
) -> List[Dict]:
    """Drop entries whose mp4 isn't on disk."""
    return [
        e for e in entries
        if os.path.exists(os.path.join(video_dir, f"{e['video_id']}{ext}"))
    ]


def balanced_subsample(
    entries: List[Dict],
    n_per_task: int,
    seed: int = 42,
    tasks: tuple = TARGET_TASKS,
) -> List[Dict]:
    """Sample n_per_task entries from each task type for a balanced eval set."""
    rng = random.Random(seed)
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        by_task[e["task_type"]].append(e)

    sampled = []
    for task in tasks:
        pool = by_task.get(task, [])
        if len(pool) < n_per_task:
            print(f"[balanced_subsample] task={task!r}: requested {n_per_task} "
                  f"but only {len(pool)} available — taking all of them")
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n_per_task))
    rng.shuffle(sampled)
    return sampled


def task_distribution(entries: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e["task_type"]] += 1
    return dict(counts)


def task_distribution_by_code(entries: List[Dict]) -> Dict[str, int]:
    """Same as task_distribution but keyed by short code (AIE/ACC/...)."""
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        code = e.get("task_code") or TASK_NAME_TO_CODE.get(e["task_type"], e["task_type"])
        counts[code] += 1
    return dict(counts)
