"""Prompt templates for all evaluation stages.

Stages:
    S0 — Text-only baseline (Qwen2.5-1.5B-Instruct)
    S1 — Audio-only (Qwen2.5-Omni-7B)
    S2 — Visual-only (Qwen2.5-Omni-7B, silent video)
    S3 — Full audio-visual + inline attribution probe (Qwen2.5-Omni-7B)
    S4 — Full AV + matched ASR transcript (Qwen2.5-Omni-7B)
    S5 — Full AV + mismatched transcript (Qwen2.5-Omni-7B) — lexical override test

The S3 prompt asks for the answer, confidence, and modality attribution in a
single turn. This saves an inference pass per sample vs. a two-turn protocol.
"""

from typing import Dict


PREAMBLES = {
    "S0": "",
    "S1": "You are given an AUDIO clip extracted from a video. Listen carefully to all speech, sounds, tone, and audio cues, then answer the question.",
    "S2": "You are given a SILENT VIDEO (the audio track has been removed). Based ONLY on what you see in the visual frames, answer the question.",
    "S3": "You are given a VIDEO with audio. Watch and listen carefully, then answer the question.",
    # S4 and S5 use the transcript-injection preamble below
}


def base_mcq_prompt(question: str, options: Dict[str, str], preamble: str = "") -> str:
    """Standard MCQ prompt with verbalized confidence elicitation."""
    options_str = "\n".join(f"({k}) {v}" for k, v in options.items())
    body = f"""Question: {question}

Options:
{options_str}

Answer with the letter (A, B, C, or D) and your confidence as an integer from 0 to 100, where 0 is a pure guess and 100 is absolutely certain.

Respond in exactly this format on one line:
[ANSWER] X [CONFIDENCE] Y"""
    return f"{preamble}\n\n{body}".strip() if preamble else body


def s3_full_av_with_attribution_prompt(question: str, options: Dict[str, str]) -> str:
    """S3 — Full AV with inline attribution probe in a single turn.

    Returns: answer letter, confidence 0-100, AND a ranking of which modality
    the model relied on most. Single-turn keeps inference cost down.
    """
    options_str = "\n".join(f"({k}) {v}" for k, v in options.items())
    return f"""You are given a VIDEO with audio. Watch and listen carefully, then answer the question.

Question: {question}

Options:
{options_str}

Then, AFTER answering, reflect on which information source you relied on MOST. Rank these from most relied-upon (1) to least relied-upon (3):
- Audio (speech content, tone, sounds)
- Visual (what you saw in the video frames)
- Text (the question wording / prior knowledge)

Respond in exactly this format on three lines:
[ANSWER] X [CONFIDENCE] Y
[RANK] Audio=A, Visual=V, Text=T
[REASON] one short sentence on why you ranked them this way"""


def s4_matched_transcript_prompt(
    question: str, options: Dict[str, str], transcript: str
) -> str:
    """S4 — Full AV + ASR transcript of THIS video (matched)."""
    options_str = "\n".join(f"({k}) {v}" for k, v in options.items())
    return f"""You are given a VIDEO with audio. A transcript of the spoken content is also provided below.

Transcript:
"{transcript.strip()}"

Watch and listen carefully, then answer the question.

Question: {question}

Options:
{options_str}

Answer with the letter (A, B, C, or D) and your confidence as an integer from 0 to 100, where 0 is a pure guess and 100 is absolutely certain.

Respond in exactly this format on one line:
[ANSWER] X [CONFIDENCE] Y"""


def s5_mismatched_transcript_prompt(
    question: str, options: Dict[str, str], mismatched_transcript: str
) -> str:
    """S5 — Full AV + transcript from a DIFFERENT video.

    The audio in the video is the truth. The transcript contradicts it.
    A model that trusts text over audio will follow the transcript and get
    the question wrong. We don't tell the model the transcript is wrong —
    that would defeat the test.
    """
    options_str = "\n".join(f"({k}) {v}" for k, v in options.items())
    return f"""You are given a VIDEO with audio. A transcript of the spoken content is also provided below.

Transcript:
"{mismatched_transcript.strip()}"

Watch and listen carefully, then answer the question.

Question: {question}

Options:
{options_str}

Answer with the letter (A, B, C, or D) and your confidence as an integer from 0 to 100, where 0 is a pure guess and 100 is absolutely certain.

Respond in exactly this format on one line:
[ANSWER] X [CONFIDENCE] Y"""


STAGE_BUILDERS = {
    "S0": lambda q, opts, **kw: base_mcq_prompt(q, opts, PREAMBLES["S0"]),
    "S1": lambda q, opts, **kw: base_mcq_prompt(q, opts, PREAMBLES["S1"]),
    "S2": lambda q, opts, **kw: base_mcq_prompt(q, opts, PREAMBLES["S2"]),
    "S3": lambda q, opts, **kw: s3_full_av_with_attribution_prompt(q, opts),
    "S4": lambda q, opts, transcript, **kw: s4_matched_transcript_prompt(q, opts, transcript),
    "S5": lambda q, opts, mismatched_transcript, **kw: s5_mismatched_transcript_prompt(
        q, opts, mismatched_transcript
    ),
}
