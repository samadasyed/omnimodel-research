"""ASR transcripts + mismatched-transcript pool construction.

The S7 mismatched-transcript condition is the lexical-vs-acoustic conflict
test. For each sample we need a transcript that:
  - looks like a plausible AVUT transcript (the model shouldn't notice
    it's "off"),
  - comes from a DIFFERENT video on the same task type (so genre / topic
    distribution roughly match),
  - is not so trivially mismatched that the answer is obviously
    contradicted in a way a text-only reader would catch.

Pairing: video V_i is paired with the transcript of V_j (j != i, same
task_type). Random-seeded for reproducibility; identical seed → identical
pairings across model runs (Gemma + Qwen-Omni).
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List


# ─── Whisper ───────────────────────────────────────────────────────
def transcribe_with_whisper(
    audio_path: str,
    output_dir: str,
    model_size: str = "small",
    language: str = "en",
) -> str:
    """Transcribe an audio file using openai-whisper CLI. Cached on disk.

    Notebook callers should ``pip install openai-whisper`` and ensure
    ffmpeg is on PATH.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = Path(audio_path).stem
    txt_path = os.path.join(output_dir, f"{base}.txt")

    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return f.read().strip()

    subprocess.run(
        [
            "whisper", audio_path,
            "--model", model_size,
            "--language", language,
            "--output_format", "txt",
            "--output_dir", output_dir,
            "--verbose", "False",
        ],
        check=False,
        capture_output=True,
    )
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return f.read().strip()
    return ""


def transcribe_with_whisper_python(
    audio_path: str,
    output_path: str,
    whisper_model,   # already-loaded model object
    language: str = "en",
) -> str:
    """Transcribe using an already-loaded Python whisper model. Caches to disk.

    Cheaper than the CLI when transcribing many files in a row from a
    notebook because we keep the model loaded across calls.
    """
    if os.path.exists(output_path):
        with open(output_path) as f:
            return f.read().strip()

    result = whisper_model.transcribe(
        audio_path, language=language, fp16=True, verbose=False,
    )
    text = (result.get("text") or "").strip()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    return text


# ─── Mismatched-transcript pairing ────────────────────────────────
def build_mismatched_pairs(
    samples: List[Dict],
    seed: int = 42,
) -> Dict[int, str]:
    """Map qa_id → donor video_id whose transcript should be used as the
    mismatched transcript.

    Donor is from the SAME task_type but ALWAYS a different underlying
    video. If a task only has one video in the sample, that qa_id gets
    paired with itself and S7 effectively degenerates to S5 — flagged
    by a printed warning.
    """
    rng = random.Random(seed)
    by_task: Dict[str, List[Dict]] = {}
    for s in samples:
        by_task.setdefault(s["task_type"], []).append(s)

    pairs: Dict[int, str] = {}
    for s in samples:
        pool = [d for d in by_task[s["task_type"]] if d["video_id"] != s["video_id"]]
        if not pool:
            print(f"[build_mismatched_pairs] WARN task={s['task_type']!r} "
                  f"only has video {s['video_id']!r}; "
                  f"S7 degenerates to S5 for qa_id={s['qa_id']}")
            pairs[s["qa_id"]] = s["video_id"]
        else:
            pairs[s["qa_id"]] = rng.choice(pool)["video_id"]
    return pairs


def load_transcript(transcript_path: str) -> str:
    if not os.path.exists(transcript_path):
        return ""
    with open(transcript_path) as f:
        return f.read().strip()
