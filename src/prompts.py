"""Prompt templates for the AVUT diagnostic stages.

Stage scheme (aligned with teammate Jeff's repo so cross-model numbers
are directly comparable):

    S1_text_only            — Q + options only (Qwen2.5-1.5B-Instruct)
    S2_audio_only           — audio file only (omnimodal model)
    S3_visual_only          — silent video only
    S4_full_av              — full audio-visual
    S5_transcript_injected  — full AV + matched Whisper transcript
    S6_attribution          — depends on S4: re-prompt with prior answer +
                              ask for modality ranking + reason
    S7_mismatched_transcript— full AV + transcript from a DIFFERENT video
                              (our unique addition: lexical-override probe)
    S8_prosody              — full AV with prosody-first verbalization

The verbalized-confidence elicitation is INLINE on every stage (single-turn).
Attribution (S6) is the only multi-turn stage: it shows the model its prior
answer to make the reflection question grounded in what was actually said.

Format choice: explicit '[ANSWER] X [CONFIDENCE] Y' bracket markers + a
worked example. Multimodal models — Gemma-3n especially — often drop
brackets when given video; the example reduces format drift.
"""

from typing import Dict


# ─── Per-stage preambles ────────────────────────────────────────────
PREAMBLES = {
    "S1_text_only":  "",
    "S2_audio_only": "You are given an audio clip from a video. Listen carefully "
                     "to all sounds, speech, and audio cues, then answer the "
                     "following question based on what you hear.",
    "S3_visual_only":"You are given a silent video (no audio). Based only on what "
                     "you see in the video frames, answer the following question.",
    "S4_full_av":    "You are given a video with audio. Watch and listen carefully, "
                     "then answer the following question.",
}


# ─── Core MCQ prompt with inline confidence elicitation ────────────
def base_mcq_prompt(question: str, options: Dict[str, str], preamble: str = "") -> str:
    """Standard MCQ prompt with verbalized confidence (0-100).

    The literal example response reduces format drift on multimodal models
    that otherwise drop the bracket markers.
    """
    opts_str = "\n".join(f"({k}) {v}" for k, v in options.items())
    head = (preamble + "\n") if preamble else ""
    return (
        f"{head}Question: {question}\n\n"
        f"Options:\n{opts_str}\n\n"
        "Reply with EXACTLY this format and nothing else:\n"
        "[ANSWER] <letter> [CONFIDENCE] <integer>\n\n"
        "where <letter> is one of A, B, C, D and <integer> is your confidence "
        "from 0 to 100 (0 = pure guess, 100 = absolutely certain).\n\n"
        "Example response: [ANSWER] C [CONFIDENCE] 75\n\n"
        "Your response:"
    )


# ─── S5: Matched-transcript injection ──────────────────────────────
def transcript_injected_prompt(
    question: str, options: Dict[str, str], transcript: str,
    transcript_max_chars: int = 8000,
) -> str:
    """Full AV + Whisper transcript of THIS video.

    Transcripts are character-truncated to keep token count well below
    32k context. Most AVUT clips fit easily; a few multi-minute samples
    would otherwise overflow.
    """
    transcript = transcript.strip()
    if len(transcript) > transcript_max_chars:
        transcript = transcript[:transcript_max_chars].rstrip() + " [...]"

    preamble = (
        "You are given a video with audio. A transcript of the spoken content "
        "is also provided below.\n\n"
        f'Transcript:\n"{transcript}"\n\n'
        "Watch and listen carefully, then answer the following question."
    )
    return base_mcq_prompt(question, options, preamble)


# ─── S6: Attribution follow-up (multi-turn-style, single call) ────
def attribution_followup_prompt() -> str:
    """The follow-up text appended after the model's S4 answer.

    Concatenated form: '[S4 prompt]\n\nYour previous answer: [S4 raw]\n\n[this]'.
    """
    return (
        "You just answered a question about this video. Now reflect on your "
        "reasoning process.\n\n"
        "Which modality or information source did you rely on MOST to reach "
        "your answer? Rank the following from most relied-upon (1) to least "
        "relied-upon (3):\n"
        "- Audio (speech content, tone, sounds)\n"
        "- Visual (what you saw in the video frames)\n"
        "- Question text / prior knowledge (reasoning from the question "
        "wording alone)\n\n"
        "Format: [RANK] Audio=X, Visual=Y, Text=Z\n\n"
        "Also briefly explain in one sentence why you ranked them this way.\n"
        "Format: [REASON] <your explanation>"
    )


def build_attribution_combined_prompt(
    question: str, options: Dict[str, str], s4_raw_response: str,
) -> str:
    """Full S6 prompt: S4 prompt + S4 answer + reflection follow-up."""
    s4_prompt = base_mcq_prompt(question, options, PREAMBLES["S4_full_av"])
    return (
        f"{s4_prompt}\n\n"
        f"Your previous answer: {s4_raw_response}\n\n"
        f"{attribution_followup_prompt()}"
    )


# ─── S7: Mismatched-transcript (our unique contribution) ──────────
def mismatched_transcript_prompt(
    question: str, options: Dict[str, str], mismatched_transcript: str,
    transcript_max_chars: int = 8000,
) -> str:
    """Full AV + transcript from a DIFFERENT same-task video.

    The audio in the video is the truth. The transcript contradicts it.
    A model that trusts text over audio will follow the transcript and
    get the question wrong. We do NOT tell the model the transcript is
    mismatched — that would defeat the test.

    This is the lexical-override probe (LISTEN-style, but in the AVUT
    audio-centric setting). It is the headline ingredient for our LOR
    metric.
    """
    transcript = mismatched_transcript.strip()
    if len(transcript) > transcript_max_chars:
        transcript = transcript[:transcript_max_chars].rstrip() + " [...]"

    preamble = (
        "You are given a video with audio. A transcript of the spoken content "
        "is also provided below.\n\n"
        f'Transcript:\n"{transcript}"\n\n'
        "Watch and listen carefully, then answer the following question."
    )
    return base_mcq_prompt(question, options, preamble)


# ─── S8: Prosody-first verbalization (optional) ───────────────────
def prosody_verbalization_prompt(question: str, options: Dict[str, str]) -> str:
    """Verbalize prosody in 2-3 sentences, then answer.

    Tightened format: a long verbose prompt produced a 46% parse rate in
    Jeff's pilot because the model spent its token budget on prosody and
    dropped the answer block. Capping description length and showing an
    explicit two-block example fixes this.
    """
    opts_str = "\n".join(f"({k}) {v}" for k, v in options.items())
    return (
        "You are given a video with audio. First describe the audio briefly "
        "(2-3 sentences max), then answer the question.\n\n"
        "Cover in your description: speaker tone, speech rate, and any "
        "non-speech audio (music, sound effects).\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{opts_str}\n\n"
        "Use this exact two-block format:\n\n"
        "[AUDIO] <2-3 sentence description of the audio>\n"
        "[ANSWER] <letter A/B/C/D> [CONFIDENCE] <integer 0-100>\n\n"
        "Example:\n"
        "[AUDIO] A man speaks calmly at normal speed. There is faint background music.\n"
        "[ANSWER] B [CONFIDENCE] 70"
    )
