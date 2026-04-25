"""ASR transcript generation + mismatched-transcript pool construction.

The mismatched-transcript condition (S5) is the lexical-vs-acoustic conflict
test. For each sample we need a transcript that:
  - is a plausible-looking AVUT-style transcript (so the model doesn't
    notice it's "off"),
  - comes from a DIFFERENT video on the same task type (so the genre and
    the topic distribution roughly match),
  - is not so trivially mismatched that the answer is obviously contradicted
    in a way that even a text-only reader would notice.

We pair video V_i with the transcript of video V_j (j != i) chosen at random
within the same task type. We fix the random seed so this is reproducible.
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List


def transcribe_with_whisper(
    audio_path: str,
    output_dir: str,
    model_size: str = "small",
    language: str = "en",
) -> str:
    """Run openai-whisper CLI on an audio file. Returns the transcript text.

    Notebook callers should ``pip install openai-whisper`` and ensure ffmpeg
    is on PATH. We persist the txt to disk for caching.
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


def build_mismatched_pairs(
    samples: List[Dict],
    seed: int = 42,
) -> Dict[int, str]:
    """For each sample, pick a different sample of the same task type
    whose transcript we'll inject as the mismatched transcript.

    Keying: maps qa_id (the unique identifier per sample) to the video_id
    (YouTube ID) of the donor whose transcript should be used.

    Property: the donor is from the SAME task_type, but always a different
    underlying video. If only one sample exists for a task, it pairs with
    itself and a warning is printed (S5 degenerates to S4 for that sample).
    """
    rng = random.Random(seed)
    by_task: Dict[str, List[Dict]] = {}
    for s in samples:
        by_task.setdefault(s["task_type"], []).append(s)

    pairs: Dict[int, str] = {}
    for s in samples:
        pool = [d for d in by_task[s["task_type"]] if d["video_id"] != s["video_id"]]
        if not pool:
            print(f"[build_mismatched_pairs] WARNING: task={s['task_type']!r} has only "
                  f"video {s['video_id']}; S5 degenerates to S4 for qa_id={s['qa_id']}")
            pairs[s["qa_id"]] = s["video_id"]
        else:
            donor = rng.choice(pool)
            pairs[s["qa_id"]] = donor["video_id"]
    return pairs


def load_transcript(transcript_path: str) -> str:
    if not os.path.exists(transcript_path):
        return ""
    with open(transcript_path) as f:
        return f.read().strip()
