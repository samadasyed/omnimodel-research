"""Metric computation for the AVUT diagnostic pipeline.

Operates on per-stage prediction lists (one row per qa_id per stage). Row
schema (compatible with Jeff's repo for cross-model joins):

    {
      "qa_id":            int,        # PRIMARY KEY — unique per question
      "video_id":         str,        # YouTube ID
      "task_type":        str,        # human-readable name
      "task_code":        str,        # short: AIE/ACC/AEL/AVCM/AVOM/AVTM
      "stage":            str,        # S1_text_only..S8_prosody
      "ground_truth":     "A".."D",
      "predicted_answer": "A".."D" | None,
      "confidence":       int 0..100 | None,
      "raw_response":     str,
      ... stage-specific extras (attribution, transcript, etc.)
    }

Cross-stage matching uses ``qa_id`` because (video_id, task_type) is NOT
unique in AVUT (~5% of videos have two QAs of the same task type).

Metric definitions match Jeff's `scripts/07_compute_metrics.py` exactly so
Gemma vs Qwen-Omni numbers are directly comparable.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Optional


# ─── Helpers ───────────────────────────────────────────────────────
def _group_by_task(rows: List[Dict]) -> Dict[str, List[Dict]]:
    by = defaultdict(list)
    for r in rows:
        by[r.get("task_type")].append(r)
    return dict(by)


def _task_label(row: Dict) -> str:
    return row.get("task_code") or row.get("task_type") or "?"


# ─── 1. Accuracy ───────────────────────────────────────────────────
def compute_accuracy(stages: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Dict]]:
    """Per-task per-stage accuracy.

    Returns {stage: {task: {accuracy, n_correct, n_valid, n_total}, "OVERALL": {...}}}
    """
    out: Dict[str, Dict[str, Dict]] = {}
    for stage_name, rows in stages.items():
        if not rows:
            continue
        per_task = {}
        for task, entries in _group_by_task(rows).items():
            valid = [e for e in entries if e.get("predicted_answer") is not None]
            correct = sum(1 for e in valid
                          if e["predicted_answer"] == e["ground_truth"])
            per_task[task] = {
                "accuracy": correct / len(valid) if valid else None,
                "n_correct": correct,
                "n_valid": len(valid),
                "n_total": len(entries),
            }
        all_valid = [r for r in rows if r.get("predicted_answer") is not None]
        all_correct = sum(1 for r in all_valid
                          if r["predicted_answer"] == r["ground_truth"])
        per_task["OVERALL"] = {
            "accuracy": all_correct / len(all_valid) if all_valid else None,
            "n_correct": all_correct,
            "n_valid": len(all_valid),
            "n_total": len(rows),
        }
        out[stage_name] = per_task
    return out


# ─── 2. Confidence summary ─────────────────────────────────────────
def compute_confidence(stages: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Dict]]:
    """Per-task per-stage mean/std of verbalized confidence (0..100)."""
    out: Dict[str, Dict[str, Dict]] = {}
    for stage_name, rows in stages.items():
        if not rows:
            continue
        per_task = {}
        for task, entries in _group_by_task(rows).items():
            confs = [e["confidence"] for e in entries
                     if e.get("confidence") is not None]
            if confs:
                per_task[task] = {
                    "mean": statistics.mean(confs),
                    "std": statistics.stdev(confs) if len(confs) > 1 else 0.0,
                    "n_valid": len(confs),
                    "n_total": len(entries),
                }
            else:
                per_task[task] = {
                    "mean": None, "std": None,
                    "n_valid": 0, "n_total": len(entries),
                }
        out[stage_name] = per_task
    return out


# ─── 3. Confidence drops (ΔConf) ───────────────────────────────────
def compute_confidence_drops(confidence: Dict) -> Dict[str, Dict[str, float]]:
    """ΔConf(audio_removal) = Conf(S4_full_av) − Conf(S3_visual_only)
       ΔConf(visual_removal) = Conf(S4_full_av) − Conf(S2_audio_only)

    A well-calibrated model has HIGH ΔConf when ablating the modality
    that carries the signal. Flat ΔConf on audio-essential tasks suggests
    overconfidence under modality loss.
    """
    s4 = confidence.get("S4_full_av", {})
    s3 = confidence.get("S3_visual_only", {})
    s2 = confidence.get("S2_audio_only", {})

    out = {"audio_removal": {}, "visual_removal": {}}
    for task in s4:
        if task == "OVERALL":
            continue
        m4 = s4[task].get("mean")
        m3 = s3.get(task, {}).get("mean")
        m2 = s2.get(task, {}).get("mean")
        if m4 is not None and m3 is not None:
            out["audio_removal"][task] = m4 - m3
        if m4 is not None and m2 is not None:
            out["visual_removal"][task] = m4 - m2
    return out


# ─── 4. Answer flip rate ───────────────────────────────────────────
def compute_flip_rate(
    stages: Dict[str, List[Dict]], stage_a: str, stage_b: str,
) -> Dict:
    """Fraction of samples where the answer differs between stage_a and stage_b.

    Indexed by qa_id. None answers are skipped.
    """
    a_rows = stages.get(stage_a, [])
    b_rows = stages.get(stage_b, [])
    if not a_rows or not b_rows:
        return {"overall": None, "n_flips": 0, "n_total": 0, "per_task": {}}

    a_lookup = {r["qa_id"]: r for r in a_rows}
    b_lookup = {r["qa_id"]: r for r in b_rows}
    common = set(a_lookup) & set(b_lookup)

    by_task = defaultdict(lambda: {"flips": 0, "total": 0})
    flips = total = 0
    for qid in common:
        a_ans = a_lookup[qid].get("predicted_answer")
        b_ans = b_lookup[qid].get("predicted_answer")
        if a_ans is None or b_ans is None:
            continue
        task = a_lookup[qid].get("task_type")
        total += 1
        by_task[task]["total"] += 1
        if a_ans != b_ans:
            flips += 1
            by_task[task]["flips"] += 1

    return {
        "overall": flips / total if total else None,
        "n_flips": flips,
        "n_total": total,
        "per_task": {
            t: {"rate": d["flips"] / d["total"] if d["total"] else None,
                "n_flips": d["flips"], "n_total": d["total"]}
            for t, d in by_task.items()
        },
    }


# ─── 5. Transcript Injection Bias (matched transcript) ────────────
def compute_tib(accuracy: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """TIB = Acc(S4_full_av) − Acc(S5_transcript_injected).

    > 0  transcript hurts (over-reliance on text)
    < 0  transcript helps (text complements audio)
    ≈ 0  transcript neutral
    """
    s4 = accuracy.get("S4_full_av", {})
    s5 = accuracy.get("S5_transcript_injected", {})
    if not s4 or not s5:
        return {}
    out = {}
    for task in s4:
        a4 = s4[task].get("accuracy")
        a5 = s5.get(task, {}).get("accuracy")
        if a4 is not None and a5 is not None:
            out[task] = {"tib": a4 - a5, "s4_accuracy": a4, "s5_accuracy": a5}
    return out


# ─── 6. Lexical Override Rate (mismatched transcript — our metric) ──
def compute_lor(stages: Dict[str, List[Dict]]) -> Dict:
    """Among samples where S4 was CORRECT, fraction flipped by mismatched
    transcript (S7).

    LOR > 0 = model trusted the (wrong) text over the audio.
    Restricting to S4-correct isolates the override effect from samples
    where S4 was already wrong for unrelated reasons.
    """
    s4 = stages.get("S4_full_av", [])
    s7 = stages.get("S7_mismatched_transcript", [])
    if not s4 or not s7:
        return {}

    s7_by = {r["qa_id"]: r for r in s7}
    by_task = defaultdict(lambda: {"flipped": 0, "stayed": 0, "skipped_s4_wrong": 0})
    for r in s4:
        s7_r = s7_by.get(r["qa_id"])
        if s7_r is None:
            continue
        task = r.get("task_type")
        s4_ans = r.get("predicted_answer")
        s7_ans = s7_r.get("predicted_answer")
        if s4_ans is None or s7_ans is None:
            continue
        if s4_ans != r["ground_truth"]:
            by_task[task]["skipped_s4_wrong"] += 1
            continue
        if s7_ans != s4_ans:
            by_task[task]["flipped"] += 1
        else:
            by_task[task]["stayed"] += 1

    out = {}
    for task, c in by_task.items():
        denom = c["flipped"] + c["stayed"]
        out[task] = {
            "lor": c["flipped"] / denom if denom else None,
            **c,
        }
    overall_f = sum(c["flipped"] for c in by_task.values())
    overall_s = sum(c["stayed"] for c in by_task.values())
    overall_d = overall_f + overall_s
    out["OVERALL"] = {
        "lor": overall_f / overall_d if overall_d else None,
        "flipped": overall_f, "stayed": overall_s,
        "skipped_s4_wrong": sum(c["skipped_s4_wrong"] for c in by_task.values()),
    }
    return out


# ─── 7. Attribution Faithfulness Score (with trivial filter) ──────
def compute_afs(stages: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """AFS — refined per Jeff:

    1. Look up self-reported #1 modality from S6 attribution.
    2. TRIVIAL FILTER: if all single-modality stages independently agree
       with S4 (S4_ans == S3_ans == S2_ans), the modality choice is
       unfalsifiable — skip and count separately.
    3. If "Audio" claimed → check whether S3 (visual-only, audio ablated)
       gives a different answer than S4. Different = faithful.
    4. If "Visual" claimed → check S2 (audio-only) vs S4.
    5. If "Text" claimed → counted as faithful (no ablation possible
       since text is in the question itself).

    The trivial filter prevents inflated confabulation counts on easy
    questions every modality solves independently.
    """
    from .parse_utils import top_modality

    s6 = stages.get("S6_attribution", [])
    s4 = stages.get("S4_full_av", [])
    s3 = stages.get("S3_visual_only", [])
    s2 = stages.get("S2_audio_only", [])
    if not s6:
        return {}

    s4_by = {r["qa_id"]: r for r in s4}
    s3_by = {r["qa_id"]: r for r in s3}
    s2_by = {r["qa_id"]: r for r in s2}

    counts = defaultdict(lambda: {
        "faithful": 0, "confabulated": 0, "unparseable": 0, "trivial": 0,
        "claimed_audio": 0, "claimed_visual": 0, "claimed_text": 0,
    })

    for r in s6:
        task = r.get("task_type")
        attr = r.get("attribution")
        qid = r["qa_id"]

        if not attr or not isinstance(attr, dict):
            counts[task]["unparseable"] += 1
            continue

        top = top_modality(attr)
        counts[task][f"claimed_{top.lower()}"] += 1

        s4_r = s4_by.get(qid)
        if s4_r is None:
            counts[task]["unparseable"] += 1
            continue
        s4_ans = s4_r.get("predicted_answer")
        s3_ans = (s3_by.get(qid) or {}).get("predicted_answer")
        s2_ans = (s2_by.get(qid) or {}).get("predicted_answer")

        # Trivial filter
        if s3_ans is not None and s2_ans is not None and s4_ans == s3_ans == s2_ans:
            counts[task]["trivial"] += 1
            continue

        if top == "Audio":
            ablated = s3_ans
        elif top == "Visual":
            ablated = s2_ans
        else:  # Text — no ablation possible
            counts[task]["faithful"] += 1
            continue

        if ablated is None:
            counts[task]["unparseable"] += 1
        elif ablated != s4_ans:
            counts[task]["faithful"] += 1
        else:
            counts[task]["confabulated"] += 1

    out = {}
    for task, c in counts.items():
        denom = c["faithful"] + c["confabulated"]
        out[task] = {
            "afs": c["faithful"] / denom if denom else None,
            "n_falsifiable": denom,
            **c,
        }
    overall_f = sum(c["faithful"] for c in counts.values())
    overall_c = sum(c["confabulated"] for c in counts.values())
    overall_d = overall_f + overall_c
    out["OVERALL"] = {
        "afs": overall_f / overall_d if overall_d else None,
        "n_falsifiable": overall_d,
        "faithful": overall_f, "confabulated": overall_c,
        "unparseable": sum(c["unparseable"] for c in counts.values()),
        "trivial": sum(c["trivial"] for c in counts.values()),
    }
    return out


# ─── 8. Confidence-accuracy correlation (point-biserial) ──────────
def compute_conf_acc_correlation(
    stages: Dict[str, List[Dict]], min_n: int = 5,
) -> Dict[str, Dict[str, Dict]]:
    """Per-task per-stage point-biserial r between confidence and correctness.

    A well-calibrated model has high positive r — confident when right,
    uncertain when wrong. Per-stage comparison reveals which modality
    config produces best calibration.

    Skips (task, stage) cells with fewer than `min_n` parseable pairs.
    """
    try:
        from scipy.stats import pointbiserialr
    except ImportError:
        return {"error": "scipy not installed"}

    out: Dict[str, Dict[str, Dict]] = {}
    for stage_name, rows in stages.items():
        if not rows:
            continue
        per_task = {}
        for task, entries in _group_by_task(rows).items():
            pairs = [
                (e["confidence"],
                 1 if e.get("predicted_answer") == e.get("ground_truth") else 0)
                for e in entries
                if e.get("confidence") is not None
                and e.get("predicted_answer") is not None
            ]
            if len(pairs) >= min_n:
                confs, corrects = zip(*pairs)
                try:
                    corr, pval = pointbiserialr(corrects, confs)
                    per_task[task] = {
                        "correlation": float(corr),
                        "p_value": float(pval),
                        "n": len(pairs),
                    }
                except Exception as e:
                    per_task[task] = {"error": str(e), "n": len(pairs)}
            else:
                per_task[task] = {"n": len(pairs), "note": "insufficient data"}
        out[stage_name] = per_task
    return out


# ─── 9. Expected Calibration Error (verbalized confidence) ────────
def compute_ece(
    stages: Dict[str, List[Dict]], n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    """ECE per stage, computed from verbalized confidence (/ 100) and
    correctness. Bin by confidence in [0, 1] over n_bins.
    """
    out: Dict[str, Optional[float]] = {}
    for stage_name, rows in stages.items():
        pairs = []
        for r in rows:
            v = r.get("confidence")
            if v is None or r.get("predicted_answer") is None:
                continue
            pairs.append((v / 100.0,
                          1 if r["predicted_answer"] == r["ground_truth"] else 0))
        if not pairs:
            out[stage_name] = None
            continue
        bins: List[List] = [[] for _ in range(n_bins)]
        for conf, correct in pairs:
            idx = min(int(conf * n_bins), n_bins - 1)
            bins[idx].append((conf, correct))
        n = len(pairs)
        ece = 0.0
        for b in bins:
            if not b:
                continue
            avg_conf = sum(c for c, _ in b) / len(b)
            avg_acc = sum(k for _, k in b) / len(b)
            ece += (len(b) / n) * abs(avg_acc - avg_conf)
        out[stage_name] = ece
    return out


# ─── 10. Standard report bundle ───────────────────────────────────
def compute_all(stages: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """One-shot: compute every metric this module supports.

    Use from analysis notebooks: `metrics = compute_all(stages)`.
    """
    accuracy = compute_accuracy(stages)
    confidence = compute_confidence(stages)
    drops = compute_confidence_drops(confidence)
    tib = compute_tib(accuracy)
    lor = compute_lor(stages)
    afs = compute_afs(stages)
    conf_corr = compute_conf_acc_correlation(stages)
    ece = compute_ece(stages)

    flip_pairs = [
        ("S4_full_av", "S3_visual_only", "remove_audio"),
        ("S4_full_av", "S2_audio_only",  "remove_visual"),
        ("S4_full_av", "S5_transcript_injected", "add_transcript"),
        ("S4_full_av", "S7_mismatched_transcript", "mismatched_transcript"),
        ("S4_full_av", "S1_text_only",   "to_text_only"),
    ]
    flips = {label: compute_flip_rate(stages, a, b)
             for a, b, label in flip_pairs}

    return {
        "accuracy_per_task_per_stage":   accuracy,
        "confidence_per_task_per_stage": confidence,
        "confidence_drops":              drops,
        "transcript_injection_bias":     tib,
        "lexical_override_rate":         lor,
        "attribution_faithfulness":      afs,
        "confidence_accuracy_correlation": conf_corr,
        "ece_per_stage":                 ece,
        "answer_flip_rates":             flips,
    }
