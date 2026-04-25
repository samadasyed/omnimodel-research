"""Metric computation. All metrics operate on the per-stage prediction JSONs
produced by the evaluation notebooks.

A "predictions" record (one per (sample, stage) pair) has the schema:

    {
      "qa_id": int,                    # PRIMARY KEY — unique per question
      "video_id": str,                 # YouTube ID
      "task_type": str,                # human-readable name
      "task_code": str,                # short code: "AIE", "ACC", ...
      "stage": str,                    # "S0".."S5"
      "ground_truth": str,             # "A".."D"
      "predicted_answer": str | None,
      "verbalized_confidence": int | None,    # 0..100
      "answer_logprobs": dict | None,         # {"A": p, "B": p, "C": p, "D": p}
      "attribution": dict | None,             # {"Audio":r, "Visual":r, "Text":r}, S3 only
      "raw_response": str,
    }

We key cross-stage matching on ``qa_id`` because (video_id, task_type) is
NOT unique in AVUT (~5% of videos have two questions of the same task type).
We bucket per-task analyses by ``task_code`` for compact display.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Basic accuracy
# ─────────────────────────────────────────────────────────────
def accuracy(records: List[Dict]) -> float:
    n = len(records)
    if n == 0:
        return 0.0
    return sum(1 for r in records if r.get("predicted_answer") == r["ground_truth"]) / n


def accuracy_per_task(records: List[Dict]) -> Dict[str, float]:
    by_task = defaultdict(list)
    for r in records:
        by_task[r.get("task_code") or r["task_type"]].append(r)
    out = {t: accuracy(rs) for t, rs in by_task.items()}
    out["OVERALL"] = accuracy(records)
    return out


# ─────────────────────────────────────────────────────────────
# Confidence summaries
# ─────────────────────────────────────────────────────────────
def verbalized_confidence_stats(records: List[Dict]) -> Dict:
    confs = [r["verbalized_confidence"] for r in records if r.get("verbalized_confidence") is not None]
    if not confs:
        return {"mean": None, "n_valid": 0, "n_total": len(records)}
    mean = sum(confs) / len(confs)
    var = sum((c - mean) ** 2 for c in confs) / len(confs)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "n_valid": len(confs),
        "n_total": len(records),
    }


def logprob_confidence_stats(records: List[Dict]) -> Dict:
    """Confidence = max softmax probability over A/B/C/D."""
    probs = []
    for r in records:
        lp = r.get("answer_logprobs")
        if lp:
            probs.append(max(lp.values()))
    if not probs:
        return {"mean": None, "n_valid": 0, "n_total": len(records)}
    mean = sum(probs) / len(probs)
    var = sum((p - mean) ** 2 for p in probs) / len(probs)
    return {"mean": mean, "std": math.sqrt(var), "n_valid": len(probs), "n_total": len(records)}


def expected_calibration_error(
    records: List[Dict], n_bins: int = 10, source: str = "logprob"
) -> Optional[float]:
    """ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|.

    source="logprob": uses max softmax over A/B/C/D as confidence
    source="verbalized": uses verbalized confidence / 100
    """
    pairs = []
    for r in records:
        if source == "logprob":
            lp = r.get("answer_logprobs")
            if lp:
                conf = max(lp.values())
            else:
                continue
        else:
            v = r.get("verbalized_confidence")
            if v is None:
                continue
            conf = v / 100.0
        correct = 1 if r.get("predicted_answer") == r["ground_truth"] else 0
        pairs.append((conf, correct))

    if not pairs:
        return None

    bins = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    n = len(pairs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        confs = [c for c, _ in b]
        corrects = [k for _, k in b]
        avg_conf = sum(confs) / len(confs)
        avg_acc = sum(corrects) / len(corrects)
        ece += (len(b) / n) * abs(avg_acc - avg_conf)
    return ece


# ─────────────────────────────────────────────────────────────
# Confidence Drop under modality ablation
# ─────────────────────────────────────────────────────────────
def confidence_drop(
    full_records: List[Dict],
    ablated_records: List[Dict],
    source: str = "verbalized",
) -> Dict:
    """ΔConf(modality M) = Conf(full) − Conf(without M).

    Records are paired by qa_id.
    """
    full_by_key = {r["qa_id"]: r for r in full_records}
    abl_by_key = {r["qa_id"]: r for r in ablated_records}

    deltas_per_task = defaultdict(list)
    deltas_overall = []
    for k, full_r in full_by_key.items():
        abl_r = abl_by_key.get(k)
        if abl_r is None:
            continue
        if source == "verbalized":
            f = full_r.get("verbalized_confidence")
            a = abl_r.get("verbalized_confidence")
            if f is None or a is None:
                continue
            d = f - a
        else:
            f_lp = full_r.get("answer_logprobs")
            a_lp = abl_r.get("answer_logprobs")
            if not f_lp or not a_lp:
                continue
            d = max(f_lp.values()) - max(a_lp.values())
        deltas_per_task[full_r.get("task_code") or full_r["task_type"]].append(d)
        deltas_overall.append(d)

    out = {
        "source": source,
        "overall_mean": sum(deltas_overall) / len(deltas_overall) if deltas_overall else None,
        "n_paired": len(deltas_overall),
        "per_task": {t: (sum(ds) / len(ds) if ds else None) for t, ds in deltas_per_task.items()},
    }
    return out


# ─────────────────────────────────────────────────────────────
# Attribution Faithfulness (the headline metric)
# ─────────────────────────────────────────────────────────────
def attribution_faithfulness(
    s3_records: List[Dict],
    s2_records: List[Dict],   # visual-only (i.e., audio ablated)
    s1_records: List[Dict],   # audio-only (i.e., visual ablated)
) -> Dict:
    """For each S3 sample with a parseable attribution:
      - Identify the model's self-reported #1 modality.
      - Look up the corresponding ablated stage's answer for the same video.
      - "Faithful" = ablating the claimed top modality CHANGED the answer.
      - "Confabulated" = ablating it did NOT change the answer.
    """
    from .parse_utils import top_modality

    s2_by = {r["qa_id"]: r for r in s2_records}
    s1_by = {r["qa_id"]: r for r in s1_records}

    counts = defaultdict(lambda: {"faithful": 0, "confabulated": 0, "unparseable": 0,
                                    "self_audio": 0, "self_visual": 0, "self_text": 0})

    for r in s3_records:
        task = r.get("task_code") or r["task_type"]
        attr = r.get("attribution")
        s3_ans = r.get("predicted_answer")

        if not attr or not s3_ans:
            counts[task]["unparseable"] += 1
            continue

        top = top_modality(attr)
        counts[task][f"self_{top.lower()}"] = counts[task].get(f"self_{top.lower()}", 0) + 1
        key = r["qa_id"]

        if top == "Audio":
            ablated = s2_by.get(key)        # remove audio = visual-only
        elif top == "Visual":
            ablated = s1_by.get(key)        # remove visual = audio-only
        else:
            # Top = Text. There's no ablation for "text" since it's the question
            # itself. We count this as honest acknowledgment of text-based reasoning,
            # which is methodologically faithful even if uninteresting.
            counts[task]["faithful"] += 1
            continue

        if ablated is None:
            counts[task]["unparseable"] += 1
            continue

        if ablated.get("predicted_answer") != s3_ans:
            counts[task]["faithful"] += 1
        else:
            counts[task]["confabulated"] += 1

    afs = {}
    for task, c in counts.items():
        decided = c["faithful"] + c["confabulated"]
        afs[task] = {
            "score": (c["faithful"] / decided) if decided > 0 else None,
            **c,
        }

    overall_f = sum(c["faithful"] for c in counts.values())
    overall_c = sum(c["confabulated"] for c in counts.values())
    overall_decided = overall_f + overall_c
    afs["OVERALL"] = {
        "score": (overall_f / overall_decided) if overall_decided > 0 else None,
        "faithful": overall_f,
        "confabulated": overall_c,
        "unparseable": sum(c["unparseable"] for c in counts.values()),
    }
    return afs


# ─────────────────────────────────────────────────────────────
# Transcript Injection Bias (TIB) and Lexical Override Rate (LOR)
# ─────────────────────────────────────────────────────────────
def transcript_injection_bias(
    s3_records: List[Dict], s4_records: List[Dict]
) -> Dict[str, float]:
    """TIB = Acc(S3) - Acc(S4_with_matched_transcript).

    > 0: transcripts hurt (model over-relies on text over audio)
    < 0: transcripts help (text provides useful complementary info)
    """
    s3_acc = accuracy_per_task(s3_records)
    s4_acc = accuracy_per_task(s4_records)
    return {t: s3_acc[t] - s4_acc.get(t, 0.0) for t in s3_acc}


def lexical_override_rate(
    s3_records: List[Dict], s5_records: List[Dict]
) -> Dict:
    """LOR — fraction of samples where injecting a MISMATCHED transcript
    flipped the answer, restricted to samples where S3 was correct.

    Higher LOR = the model trusted the (wrong) transcript over the audio.
    Restricting to S3-correct samples isolates the override effect from
    the confound of S3 being wrong for unrelated reasons.
    """
    s5_by = {r["qa_id"]: r for r in s5_records}

    by_task = defaultdict(lambda: {"flipped": 0, "stayed": 0, "skipped_s3_wrong": 0})
    for r in s3_records:
        s5_r = s5_by.get(r["qa_id"])
        if s5_r is None:
            continue
        task = r.get("task_code") or r["task_type"]
        if r.get("predicted_answer") != r["ground_truth"]:
            by_task[task]["skipped_s3_wrong"] += 1
            continue
        if s5_r.get("predicted_answer") != r.get("predicted_answer"):
            by_task[task]["flipped"] += 1
        else:
            by_task[task]["stayed"] += 1

    out = {}
    for task, c in by_task.items():
        decided = c["flipped"] + c["stayed"]
        out[task] = {
            "lor": (c["flipped"] / decided) if decided > 0 else None,
            **c,
        }

    overall_flipped = sum(c["flipped"] for c in by_task.values())
    overall_stayed = sum(c["stayed"] for c in by_task.values())
    overall_decided = overall_flipped + overall_stayed
    out["OVERALL"] = {
        "lor": (overall_flipped / overall_decided) if overall_decided > 0 else None,
        "flipped": overall_flipped,
        "stayed": overall_stayed,
        "skipped_s3_wrong": sum(c["skipped_s3_wrong"] for c in by_task.values()),
    }
    return out


# ─────────────────────────────────────────────────────────────
# Answer flip rate between any two stages
# ─────────────────────────────────────────────────────────────
def answer_flip_rate(
    a_records: List[Dict], b_records: List[Dict]
) -> Dict:
    """Fraction of samples where the answer differs between stages a and b."""
    b_by = {r["qa_id"]: r.get("predicted_answer") for r in b_records}
    by_task = defaultdict(lambda: {"flips": 0, "same": 0})
    for r in a_records:
        b_ans = b_by.get(r["qa_id"])
        a_ans = r.get("predicted_answer")
        if a_ans is None or b_ans is None:
            continue
        task = r.get("task_code") or r["task_type"]
        if a_ans != b_ans:
            by_task[task]["flips"] += 1
        else:
            by_task[task]["same"] += 1

    out = {}
    for task, c in by_task.items():
        n = c["flips"] + c["same"]
        out[task] = {"rate": c["flips"] / n if n else None, **c}
    overall_flips = sum(c["flips"] for c in by_task.values())
    overall_same = sum(c["same"] for c in by_task.values())
    n = overall_flips + overall_same
    out["OVERALL"] = {
        "rate": overall_flips / n if n else None,
        "flips": overall_flips,
        "same": overall_same,
    }
    return out
