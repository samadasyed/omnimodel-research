"""Response parsing — extracts answer letter, confidence, attribution rank, reason.

Tries the strict bracketed format first ([ANSWER] X [CONFIDENCE] Y), then
progressively falls back to looser regexes. None signals missing data; we
never silently default, since "missing" must be distinguishable from
"uncertain" in downstream metric computation.

Multimodal models (Qwen-Omni, Gemma-3n) often drop the bracket markers
when given audio/video — the fallback patterns catch that. Aligned with
Jeff's parser so parse outcomes are comparable across model runs.
"""

import re
from typing import Dict, Optional, Tuple


# ─── Answer + confidence ────────────────────────────────────────────
def parse_answer_confidence(response: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract (answer_letter, confidence_0_to_100) from a model response.

    Recognized formats (in priority order):
      "[ANSWER] B [CONFIDENCE] 75"   ← strict
      "B 85" / "B. 85" / "B, 85"     ← bracket-stripped (common Gemma drift)
      "(B) confidence: 80"           ← parenthesized
      "The answer is C with confidence 80"
      "B"                            ← answer only, no confidence
    """
    if not response:
        return None, None

    response = response.strip()
    answer: Optional[str] = None
    confidence: Optional[int] = None

    # Strict format
    m_ans = re.search(r"\[ANSWER\]\s*([A-Da-d])", response)
    m_conf = re.search(r"\[CONFIDENCE\]\s*(\d{1,3})", response)
    if m_ans:
        answer = m_ans.group(1).upper()
    if m_conf:
        confidence = min(int(m_conf.group(1)), 100)

    # Compact "B 85" / "B. 85" / "(B) 85" pattern at the start of the response
    if answer is None or confidence is None:
        m = re.match(r"^\s*\(?([A-Da-d])\)?\s*[.,:\-]?\s*(\d{1,3})(?:\s|$|%)", response)
        if m:
            if answer is None:
                answer = m.group(1).upper()
            if confidence is None:
                confidence = min(int(m.group(2)), 100)

    # Other answer-letter fallbacks
    if answer is None:
        m = re.search(r"answer\s+is\s*[:\-]?\s*\(?([A-Da-d])\)?", response, re.IGNORECASE)
        if m:
            answer = m.group(1).upper()
    if answer is None:
        m = re.search(r"\(([A-Da-d])\)", response)
        if m:
            answer = m.group(1).upper()
    if answer is None:
        m = re.search(r"\b([A-D])\b", response)
        if m:
            answer = m.group(1)

    # Other confidence fallbacks
    if confidence is None:
        for pat in (
            r"confidence\s*[:=]\s*(\d{1,3})",
            r"(\d{1,3})\s*%",
            r"(\d{1,3})\s*(?:out of|/)\s*100",
        ):
            m = re.search(pat, response, re.IGNORECASE)
            if m:
                confidence = min(int(m.group(1)), 100)
                break

    return answer, confidence


# ─── Attribution rank (S6) ──────────────────────────────────────────
def parse_attribution(response: str) -> Optional[Dict[str, int]]:
    """Extract {'Audio': r, 'Visual': r, 'Text': r} from S6 follow-up output.

    Tries the strict '[RANK] Audio=X, Visual=Y, Text=Z' first, then a loose
    "Audio: 1 ... Visual: 2 ... Text: 3" anywhere in the response.
    """
    if not response:
        return None

    m = re.search(
        r"\[RANK\]\s*Audio\s*=\s*(\d)\s*[,;]?\s*Visual\s*=\s*(\d)\s*[,;]?\s*Text\s*=\s*(\d)",
        response, re.IGNORECASE,
    )
    if m:
        return {"Audio": int(m.group(1)),
                "Visual": int(m.group(2)),
                "Text": int(m.group(3))}

    a = re.search(r"Audio\s*[:=]\s*(\d)",  response, re.IGNORECASE)
    v = re.search(r"Visual\s*[:=]\s*(\d)", response, re.IGNORECASE)
    t = re.search(r"Text\s*[:=]\s*(\d)",   response, re.IGNORECASE)
    if a and v and t:
        return {"Audio": int(a.group(1)),
                "Visual": int(v.group(1)),
                "Text": int(t.group(1))}
    return None


def parse_reason(response: str) -> Optional[str]:
    if not response:
        return None
    m = re.search(r"\[REASON\]\s*(.+?)(?=\n\n|\[|$)",
                  response, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def top_modality(attribution: Dict[str, int]) -> Optional[str]:
    """Return the modality with the lowest rank number (== most relied on).

    Ties are broken in canonical order Audio > Visual > Text (so a model
    saying "everything tied at 1" is treated as relying on audio).
    """
    if not attribution:
        return None
    order = ["Audio", "Visual", "Text"]
    return min(attribution, key=lambda k: (attribution[k], order.index(k)))
