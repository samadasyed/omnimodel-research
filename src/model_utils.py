"""Model wrappers with a common interface.

Each wrapper exposes:
    .generate(prompt, video_path=None, audio_path=None) -> ResponseDict

ResponseDict = {
    "text": str,                          # decoded model text output
    "answer_logprobs": dict | None,       # softmax over " A"," B"," C"," D" tokens
                                          # at the position right after "[ANSWER] "
    "metadata": dict,
}

The logprob signal is captured by appending "[ANSWER] " to the prompt and
running a single forward pass to score the next token. This is simpler and
more deterministic than scraping `output_scores` mid-generation. We do a
separate generation pass for the verbalized answer + confidence + attribution.

Implementation notes:
- These wrappers are written for HuggingFace `transformers >= 4.45`.
- Qwen2.5-Omni requires `qwen-omni-utils` to process video/audio inputs.
- All wrappers default to greedy decoding (do_sample=False) for reproducibility.

If running in Colab and the API doesn't match (Qwen2.5-Omni is new and its
processor signature has shifted across releases), check the model card at
https://huggingface.co/Qwen/Qwen2.5-Omni-7B and adjust ONLY the
``_build_inputs`` method on each wrapper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Tokens we score for logprob-based confidence.
# We add a leading space because most BPE/sentencepiece tokenizers put a
# leading space on the first content token after a prompt.
CHOICE_TOKENS = {
    "A": [" A", "A"],
    "B": [" B", "B"],
    "C": [" C", "C"],
    "D": [" D", "D"],
}


@dataclass
class ResponseDict:
    text: str
    answer_logprobs: Optional[Dict[str, float]] = None  # {"A": p, ...} normalized over A/B/C/D
    metadata: Dict = field(default_factory=dict)


def _softmax_over_choices(scores: Dict[str, float]) -> Dict[str, float]:
    """Renormalize raw token logprobs (sums of token probs) to a distribution
    over A/B/C/D only. We use logsumexp for stability.
    """
    if not scores:
        return {}
    # scores are log-probs; convert to probs and renormalize
    max_lp = max(scores.values())
    exps = {k: math.exp(v - max_lp) for k, v in scores.items()}
    total = sum(exps.values())
    if total <= 0:
        return {k: 0.25 for k in "ABCD"}
    return {k: v / total for k, v in exps.items()}


# ────────────────────────────────────────────────────────────────────
# Text-only wrapper (S0)
# ────────────────────────────────────────────────────────────────────
class TextOnlyModelWrapper:
    """Qwen2.5-1.5B-Instruct text-only baseline."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype or torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        capture_choice_logprobs: bool = True,
        **kwargs,  # absorb video_path/audio_path so call signature matches
    ) -> ResponseDict:
        import torch

        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(
            out_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        choice_lp = None
        if capture_choice_logprobs:
            choice_lp = self._score_choices(chat_text)

        return ResponseDict(text=gen, answer_logprobs=choice_lp)

    def _score_choices(self, chat_text_prompt: str) -> Dict[str, float]:
        """Append '[ANSWER] ' to the prompt, do a forward pass, read next-token logprobs."""
        import torch

        scoring_prompt = chat_text_prompt + "[ANSWER]"
        ids = self.tokenizer(scoring_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]  # (vocab,)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        scores = {}
        for letter, variants in CHOICE_TOKENS.items():
            best = None
            for tok in variants:
                tok_ids = self.tokenizer.encode(tok, add_special_tokens=False)
                if not tok_ids:
                    continue
                lp = log_probs[tok_ids[0]].item()
                if best is None or lp > best:
                    best = lp
            scores[letter] = best if best is not None else -1e9
        return _softmax_over_choices(scores)


# ────────────────────────────────────────────────────────────────────
# Qwen2.5-Omni wrapper (S1–S5)
# ────────────────────────────────────────────────────────────────────
class OmniModelWrapper:
    """Wrapper for Qwen/Qwen2.5-Omni-7B.

    The model's processor uses `qwen-omni-utils` to handle video and audio
    inputs. The chat-template format expects messages like:

        [{"role": "user", "content": [
            {"type": "video", "video": <path-or-url>},
            {"type": "audio", "audio": <path-or-url>},
            {"type": "text",  "text":  <prompt>},
        ]}]

    We support audio-only (S1), video-only (S2), and full AV (S3+) by
    selectively attaching content blocks.

    NOTE: the exact module/class names below are correct as of transformers
    >= 4.45 with `Qwen2_5OmniForConditionalGeneration`. Verify against the
    HF model card if errors arise.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda",
        dtype=None,
    ):
        import torch
        from transformers import AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Try the dedicated class first; fall back to AutoModelForCausalLM if absent
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration as _OmniCls
        except ImportError:
            from transformers import AutoModelForCausalLM as _OmniCls

        self.model = _OmniCls.from_pretrained(
            model_name,
            torch_dtype=dtype or torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self._OmniCls_name = _OmniCls.__name__

    def _build_messages(
        self,
        prompt: str,
        video_path: Optional[str],
        audio_path: Optional[str],
    ) -> List[Dict]:
        content = []
        if video_path is not None:
            content.append({"type": "video", "video": video_path})
        if audio_path is not None:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _build_inputs(self, messages: List[Dict]):
        """Apply chat template + run the processor over media and text.

        Uses qwen_omni_utils.process_mm_info if available (preferred path,
        gives the model time-aligned tokens). Falls back to passing paths
        directly to the processor.
        """
        try:
            from qwen_omni_utils import process_mm_info
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
        except ImportError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=text, return_tensors="pt", padding=True)

        return inputs.to(self.device)

    def generate(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 256,
        capture_choice_logprobs: bool = True,
    ) -> ResponseDict:
        import torch

        messages = self._build_messages(prompt, video_path, audio_path)
        inputs = self._build_inputs(messages)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # When inputs has input_ids of shape (1, T), strip the prompt off
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen_ids = out_ids[:, prompt_len:]
        gen_text = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )[0].strip()

        choice_lp = None
        if capture_choice_logprobs:
            choice_lp = self._score_choices(messages)

        return ResponseDict(
            text=gen_text,
            answer_logprobs=choice_lp,
            metadata={"model_class": self._OmniCls_name},
        )

    def _score_choices(self, messages: List[Dict]) -> Dict[str, float]:
        """Score A/B/C/D by appending '[ANSWER]' to the prompt and reading
        the next-token logits. One forward pass; no generation.
        """
        import torch

        # Re-apply the chat template with an extra "[ANSWER]" suffix
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "[ANSWER]"

        try:
            from qwen_omni_utils import process_mm_info
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            inputs = self.processor(
                text=prompt_text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            ).to(self.device)
        except ImportError:
            inputs = self.processor(text=prompt_text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0, -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Use the processor's tokenizer to find token ids
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return {}

        scores = {}
        for letter, variants in CHOICE_TOKENS.items():
            best = None
            for tok in variants:
                tok_ids = tokenizer.encode(tok, add_special_tokens=False)
                if not tok_ids:
                    continue
                lp = log_probs[tok_ids[0]].item()
                if best is None or lp > best:
                    best = lp
            scores[letter] = best if best is not None else -1e9
        return _softmax_over_choices(scores)


# ────────────────────────────────────────────────────────────────────
# Gemma-3n wrapper (cross-model comparison)
# ────────────────────────────────────────────────────────────────────
class GemmaOmniWrapper:
    """Gemma-3n-E2B-IT cross-model comparison.

    Gemma-3n is a 2B-effective omnimodal model (text + image/video + audio).
    Smaller than Qwen2.5-Omni-7B, faster on H100. Use as a robustness check
    on whether confabulation findings are model-specific or general.

    The Gemma-3n processor in transformers >= 4.50 follows a similar
    chat-template-with-content-blocks pattern.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3n-E2B-it",
        device: str = "cuda",
        dtype=None,
    ):
        import torch
        from transformers import AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        try:
            from transformers import Gemma3nForConditionalGeneration as _Cls
        except ImportError:
            from transformers import AutoModelForCausalLM as _Cls

        self.model = _Cls.from_pretrained(
            model_name,
            torch_dtype=dtype or torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_new_tokens: int = 256,
        capture_choice_logprobs: bool = True,
    ) -> ResponseDict:
        import torch

        content = []
        if video_path is not None:
            content.append({"type": "video", "video": video_path})
        if audio_path is not None:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, prompt_len:]
        gen_text = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )[0].strip()

        # Skip choice logprobs for Gemma initially — different processor wrapping;
        # add later if cross-model logprob comparison becomes load-bearing.
        return ResponseDict(text=gen_text, answer_logprobs=None)
