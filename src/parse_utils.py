"""Parsing helpers for model responses.

Models don't always follow output formats. We try the strict format first,
then fall back to progressively looser regex patterns. Anything still
unparseable returns None — we never silently default, since "missing" must
be distinguishable from "uncertain" downstream.
"""

import re
from typing import Optional, Tuple, Dict


_ANS_STRICT = re.compile(r"\[ANSWER\]\s*([A-Da-d])", re.IGNORECASE)
_CONF_STRICT = re.compile(r"\[CONFIDENCE\]\s*(\d{1,3})", re.IGNORECASE)
_ANS_PAREN = re.compile(r"\(([A-Da-d])\)")
_ANS_LEAD = re.compile(r"^[\s\W]*([A-Da-d])\b")
_RANK_STRICT = re.compile(
    r"\[RANK\]\s*Audio\s*=\s*(\d).*?Visual\s*=\s*(\d).*?Text\s*=\s*(\d)",
    re.IGNORECASE | re.DOTALL,
)
_RANK_LOOSE_AUDIO = re.compile(r"[Aa]udio\s*[:=]\s*(\d)")
_RANK_LOOSE_VISUAL = re.compile(r"[Vv]isual\s*[:=]\s*(\d)")
_RANK_LOOSE_TEXT = re.compile(r"[Tt]ext\s*[:=]\s*(\d)")
_REASON = re.compile(r"\[REASON\]\s*(.+?)(?:$|\n)", re.IGNORECASE | re.DOTALL)


def parse_answer_confidence(response: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract (letter, confidence) from a model response.

    Returns (None, None) if the response has neither, (letter, None) if
    only the letter is recoverable, etc.
    """
    if not response:
        return None, None

    answer = None
    confidence = None

    m = _ANS_STRICT.search(response)
    if m:
        answer = m.group(1).upper()

    m = _CONF_STRICT.search(response)
    if m:
        confidence = min(int(m.group(1)), 100)

    if answer is None:
        m = _ANS_PAREN.search(response)
        if m:
            answer = m.group(1).upper()

    if answer is None:
        m = _ANS_LEAD.match(response.lstrip())
        if m:
            answer = m.group(1).upper()

    return answer, confidence


def parse_attribution(response: str) -> Optional[Dict[str, int]]:
    """Extract modality ranking from S3 response.

    Returns a dict like {"Audio": 1, "Visual": 2, "Text": 3} or None.
    """
    if not response:
        return None

    m = _RANK_STRICT.search(response)
    if m:
        return {
            "Audio": int(m.group(1)),
            "Visual": int(m.group(2)),
            "Text": int(m.group(3)),
        }

    am = _RANK_LOOSE_AUDIO.search(response)
    vm = _RANK_LOOSE_VISUAL.search(response)
    tm = _RANK_LOOSE_TEXT.search(response)
    if am and vm and tm:
        return {
            "Audio": int(am.group(1)),
            "Visual": int(vm.group(1)),
            "Text": int(tm.group(1)),
        }

    return None


def parse_reason(response: str) -> Optional[str]:
    if not response:
        return None
    m = _REASON.search(response)
    return m.group(1).strip() if m else None


def top_modality(attribution: Dict[str, int]) -> Optional[str]:
    """Return the modality with the lowest rank number (== most relied on).

    Ties are broken by the canonical order Audio > Visual > Text — so a
    model that says "everything tied at 1" is treated as relying on audio.
    Document this in the paper as a methodological choice; alternative is
    to drop ties.
    """
    if not attribution:
        return None
    return min(attribution, key=lambda k: (attribution[k], ["Audio", "Visual", "Text"].index(k)))
