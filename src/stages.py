"""Stage runners for the AVUT diagnostic pipeline.

Each runner is a function (model, sample, paths, **extras) -> dict that
produces ONE row of results. Rows have at least:

    qa_id, video_id, task_type, task_code, question, ground_truth,
    predicted_answer, confidence, raw_response, stage, timestamp

Stage-specific extras (e.g. S5 transcript, S6 attribution, S7 mismatched
transcript, S8 prosody) are added as additional keys.

Stage scheme matches Jeff's repo so cross-model results JSONs can be
joined directly on (qa_id, stage, model_name).

Stages
------
    S1_text_only            — text model, Q+options only
    S2_audio_only           — omnimodal, audio file
    S3_visual_only          — omnimodal, silent video
    S4_full_av              — omnimodal, video+audio
    S5_transcript_injected  — omnimodal, video+audio + matched ASR transcript
    S6_attribution          — depends on S4: re-prompt with prior answer +
                              ask for modality ranking + reason
    S7_mismatched_transcript— omnimodal, video+audio + DIFFERENT-video transcript
                              (lexical-override probe; our addition)
    S8_prosody              — omnimodal, video+audio with prosody-first prompt
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from .parse_utils import parse_answer_confidence, parse_attribution, parse_reason
from .prompts import (
    PREAMBLES,
    base_mcq_prompt,
    transcript_injected_prompt,
    mismatched_transcript_prompt,
    build_attribution_combined_prompt,
    prosody_verbalization_prompt,
)


# ─── Result row builder ────────────────────────────────────────────
def _row(sample: Dict, stage: str, response_text: str, **extras) -> Dict:
    answer, confidence = parse_answer_confidence(response_text)
    return {
        "qa_id":            sample["qa_id"],
        "video_id":         sample["video_id"],
        "task_type":        sample["task_type"],
        "task_code":        sample.get("task_code"),
        "question":         sample["question"],
        "ground_truth":     sample["answer"],
        "predicted_answer": answer,
        "confidence":       confidence,
        "raw_response":     response_text,
        "stage":            stage,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        **extras,
    }


def _load_transcript(path: str) -> str:
    import os
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


# ─── Stage runners ─────────────────────────────────────────────────
def run_s1_text_only(text_model, sample: Dict, paths: Dict) -> Dict:
    """S1 — text-only baseline. ``text_model`` is a TextOnlyModelWrapper."""
    prompt = base_mcq_prompt(
        sample["question"], sample["options"], PREAMBLES["S1_text_only"],
    )
    out = text_model.generate(prompt, max_new_tokens=64)
    return _row(sample, "S1_text_only", out.text)


def run_s2_audio_only(omni_model, sample: Dict, paths: Dict) -> Dict:
    prompt = base_mcq_prompt(
        sample["question"], sample["options"], PREAMBLES["S2_audio_only"],
    )
    out = omni_model.generate(
        prompt, audio_path=str(paths["audio"]),
        video_path=None, use_audio_in_video=False,
        max_new_tokens=64,
    )
    return _row(sample, "S2_audio_only", out.text)


def run_s3_visual_only(omni_model, sample: Dict, paths: Dict) -> Dict:
    prompt = base_mcq_prompt(
        sample["question"], sample["options"], PREAMBLES["S3_visual_only"],
    )
    out = omni_model.generate(
        prompt, video_path=str(paths["silent"]),
        audio_path=None, use_audio_in_video=False,
        max_new_tokens=64,
    )
    return _row(sample, "S3_visual_only", out.text)


def run_s4_full_av(omni_model, sample: Dict, paths: Dict) -> Dict:
    prompt = base_mcq_prompt(
        sample["question"], sample["options"], PREAMBLES["S4_full_av"],
    )
    out = omni_model.generate(
        prompt,
        video_path=str(paths["video"]),
        audio_path=str(paths["audio"]),
        use_audio_in_video=False,
        max_new_tokens=64,
    )
    return _row(sample, "S4_full_av", out.text)


def run_s5_transcript_injected(omni_model, sample: Dict, paths: Dict) -> Dict:
    transcript = _load_transcript(paths.get("transcript"))
    prompt = transcript_injected_prompt(
        sample["question"], sample["options"], transcript,
    )
    out = omni_model.generate(
        prompt,
        video_path=str(paths["video"]),
        audio_path=str(paths["audio"]),
        use_audio_in_video=False,
        max_new_tokens=64,
    )
    return _row(sample, "S5_transcript_injected", out.text, transcript=transcript)


def run_s6_attribution(omni_model, sample: Dict, paths: Dict, s4_row: Dict) -> Dict:
    """S6 — attribution probe. Concatenated multi-turn:
    [S4 prompt] + [S4 raw response] + [follow-up reflection prompt].

    Carries the S4 answer/confidence forward so AFS computation uses the
    answer the model committed to before being asked to reflect.
    """
    s4_raw = s4_row.get("raw_response", "")
    combined = build_attribution_combined_prompt(
        sample["question"], sample["options"], s4_raw,
    )
    out = omni_model.generate(
        combined,
        video_path=str(paths["video"]),
        audio_path=str(paths["audio"]),
        use_audio_in_video=False,
        max_new_tokens=200,   # room for ranking + reason
    )

    attribution = parse_attribution(out.text)
    reason = parse_reason(out.text)
    row = _row(
        sample, "S6_attribution", out.text,
        attribution=attribution,
        attribution_reason=reason,
        s4_predicted_answer=s4_row.get("predicted_answer"),
        s4_confidence=s4_row.get("confidence"),
    )
    # Override predicted_answer with S4's so AFS computation uses the S4 value
    row["predicted_answer"] = s4_row.get("predicted_answer")
    row["confidence"] = s4_row.get("confidence")
    return row


def run_s7_mismatched_transcript(omni_model, sample: Dict, paths: Dict) -> Dict:
    """S7 — full AV + transcript from a DIFFERENT same-task video.

    Our novel contribution. Quantifies lexical override: a model that
    prioritizes text over audio will follow the (wrong) transcript and
    flip its answer.
    """
    transcript = _load_transcript(paths.get("mismatched_transcript"))
    prompt = mismatched_transcript_prompt(
        sample["question"], sample["options"], transcript,
    )
    out = omni_model.generate(
        prompt,
        video_path=str(paths["video"]),
        audio_path=str(paths["audio"]),
        use_audio_in_video=False,
        max_new_tokens=64,
    )
    return _row(
        sample, "S7_mismatched_transcript", out.text,
        mismatched_transcript=transcript,
        donor_video_id=paths.get("donor_video_id"),
    )


def run_s8_prosody(omni_model, sample: Dict, paths: Dict) -> Dict:
    prompt = prosody_verbalization_prompt(sample["question"], sample["options"])
    out = omni_model.generate(
        prompt,
        video_path=str(paths["video"]),
        audio_path=str(paths["audio"]),
        use_audio_in_video=False,
        max_new_tokens=300,   # room for prosody description AND answer
    )
    return _row(sample, "S8_prosody", out.text)


# ─── Registry ──────────────────────────────────────────────────────
# 'model' = which kind of wrapper the runner expects ("text" or "omni")
# 'depends_on' = stage that must complete first (None for independent stages)
STAGE_REGISTRY = {
    "S1_text_only":            {"model": "text", "depends_on": None,         "fn": run_s1_text_only},
    "S2_audio_only":           {"model": "omni", "depends_on": None,         "fn": run_s2_audio_only},
    "S3_visual_only":          {"model": "omni", "depends_on": None,         "fn": run_s3_visual_only},
    "S4_full_av":              {"model": "omni", "depends_on": None,         "fn": run_s4_full_av},
    "S5_transcript_injected":  {"model": "omni", "depends_on": None,         "fn": run_s5_transcript_injected},
    "S6_attribution":          {"model": "omni", "depends_on": "S4_full_av", "fn": run_s6_attribution},
    "S7_mismatched_transcript":{"model": "omni", "depends_on": None,         "fn": run_s7_mismatched_transcript},
    "S8_prosody":              {"model": "omni", "depends_on": None,         "fn": run_s8_prosody},
}

# Default execution order (text first, then omni; S6 after S4)
DEFAULT_STAGE_ORDER = [
    "S1_text_only",
    "S2_audio_only",
    "S3_visual_only",
    "S4_full_av",
    "S5_transcript_injected",
    "S6_attribution",
    "S7_mismatched_transcript",
    "S8_prosody",
]
