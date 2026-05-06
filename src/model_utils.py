"""Model wrappers with a uniform `.generate()` interface.

Each wrapper exposes:

    .generate(
        prompt: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        use_audio_in_video: bool = False,
        max_new_tokens: int = 64,
    ) -> ResponseDict

ResponseDict has:
    text:               decoded model text (post-prompt)
    answer_logprobs:    {"A": p, "B": p, "C": p, "D": p} or None (optional)
    metadata:           extra info (model class name, etc.)

Wrappers
--------
    TextOnlyModelWrapper   — small text LLM for S1 baseline (Qwen2.5-1.5B-Instruct)
    Gemma3nWrapper         — primary omnimodal: Gemma-3n-E2B-IT (text+image+audio)
    QwenOmniWrapper        — alternative omnimodal: Qwen2.5-Omni-7B (cross-model)

Gemma-3n's vision pipeline is image-based, so we sample N frames from
videos and pass them as a list of PIL images. Audio is handled directly
by the processor as numpy float32 mono @ 16kHz.

Caveats:
- Gemma-3n is image-text-to-text; "video" support = N sampled frames.
- Audio length cap on Gemma-3n is roughly 30 seconds. Longer audio is
  truncated to the first 30s by default; configurable via `audio_max_s`.
- We default to greedy (do_sample=False) for reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


CHOICE_TOKENS = {
    "A": [" A", "A"],
    "B": [" B", "B"],
    "C": [" C", "C"],
    "D": [" D", "D"],
}


@dataclass
class ResponseDict:
    text: str
    answer_logprobs: Optional[Dict[str, float]] = None
    metadata: Dict = field(default_factory=dict)


def _softmax_over_choices(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_lp = max(scores.values())
    exps = {k: math.exp(v - max_lp) for k, v in scores.items()}
    total = sum(exps.values())
    if total <= 0:
        return {k: 0.25 for k in "ABCD"}
    return {k: v / total for k, v in exps.items()}


# ─── Audio + frame helpers ─────────────────────────────────────────
def _load_audio_array(audio_path: str, target_sr: int = 16000,
                       max_seconds: Optional[float] = None):
    """Load audio as float32 mono numpy array at target_sr.

    soundfile + simple linear resample if rate differs. Truncates to
    max_seconds when supplied.
    """
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        # Cheap linear resample; for AVUT diagnostic eval this is good enough.
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        data = np.interp(x_new, x_old, data).astype("float32")
        sr = target_sr
    if max_seconds is not None:
        data = data[: int(max_seconds * sr)]
    return data, sr


def _sample_video_frames(video_path: str, num_frames: int = 8):
    """Sample evenly-spaced frames from a video as a list of PIL Images.

    Uses OpenCV (preinstalled on Colab); reliable across mp4 codecs.
    Falls back gracefully to whatever frames it can read.
    """
    import numpy as np
    from PIL import Image
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [Image.new("RGB", (224, 224))]

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Some codecs don't report frame count; read sequentially up to num_frames
        frames = []
        while len(frames) < num_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames or [Image.new("RGB", (224, 224))]

    idxs = np.linspace(0, max(total - 1, 0), num=num_frames).astype(int).tolist()
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames or [Image.new("RGB", (224, 224))]


# ────────────────────────────────────────────────────────────────────
# Text-only wrapper (S1 baseline)
# ────────────────────────────────────────────────────────────────────
class TextOnlyModelWrapper:
    """Small text-only LLM. Default Qwen2.5-1.5B-Instruct (matches Jeff's S1)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype or torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        capture_choice_logprobs: bool = False,
        **_unused,
    ) -> ResponseDict:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()

        choice_lp = None
        if capture_choice_logprobs:
            choice_lp = self._score_choices(text)
        return ResponseDict(
            text=response, answer_logprobs=choice_lp,
            metadata={"model_name": self.model_name},
        )

    def _score_choices(self, prompt_text: str) -> Dict[str, float]:
        import torch
        scoring = prompt_text + "[ANSWER]"
        ids = self.tokenizer(scoring, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
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
# Gemma-3n wrapper (PRIMARY omnimodal)
# ────────────────────────────────────────────────────────────────────
class Gemma3nWrapper:
    """Wrapper for Gemma-3n-E2B-IT (or -E4B-IT) — image+audio+text omnimodal.

    Vision pipeline is image-based: we sample N frames from videos.
    Audio is loaded as float32 mono @16kHz and passed to the processor.

    The chat-template content schema matches what the Gemma-3n processor
    expects in transformers:

        [{"role": "user", "content": [
            {"type": "audio", "audio": <numpy array | path>},
            {"type": "image", "image": <PIL.Image>},
            ...one block per sampled frame...
            {"type": "text",  "text":  <prompt>},
        ]}]
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3n-E2B-it",
        device: str = "cuda",
        dtype=None,
        num_video_frames: int = 8,
        audio_max_seconds: float = 30.0,
    ):
        import torch
        from transformers import AutoProcessor

        self.device = device
        self.model_name = model_name
        self.num_video_frames = num_video_frames
        self.audio_max_seconds = audio_max_seconds

        self.processor = AutoProcessor.from_pretrained(model_name)

        try:
            from transformers import Gemma3nForConditionalGeneration as _Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as _Cls
        self._cls_name = _Cls.__name__

        self.model = _Cls.from_pretrained(
            model_name,
            torch_dtype=dtype or torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def _build_content(
        self,
        prompt: str,
        video_path: Optional[str],
        audio_path: Optional[str],
        use_audio_in_video: bool,
    ) -> List[Dict]:
        content: List[Dict] = []

        # Audio: a separate clip OR extracted from the video
        if audio_path is not None:
            audio_arr, _ = _load_audio_array(
                audio_path, target_sr=16000, max_seconds=self.audio_max_seconds,
            )
            content.append({"type": "audio", "audio": audio_arr})
        elif video_path is not None and use_audio_in_video:
            # We need a separate audio path for Gemma; the processor doesn't
            # extract audio from video. The caller usually supplies the
            # pre-extracted .wav alongside the video. If not provided, skip.
            pass

        # Visual: sampled frames from video
        if video_path is not None:
            frames = _sample_video_frames(video_path, num_frames=self.num_video_frames)
            for img in frames:
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": prompt})
        return content

    def generate(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        use_audio_in_video: bool = False,
        max_new_tokens: int = 64,
        capture_choice_logprobs: bool = False,
        audio_for_video_path: Optional[str] = None,
    ) -> ResponseDict:
        """Run one inference. ``audio_for_video_path`` lets the caller pass
        the pre-extracted .wav for a video at the same time as the silent
        video, so Gemma sees both modalities (since it can't pull audio
        from an mp4 itself).
        """
        import torch

        # If audio_for_video_path supplied, treat as the audio input
        if audio_for_video_path is not None and audio_path is None:
            audio_path = audio_for_video_path

        content = self._build_content(prompt, video_path, audio_path, use_audio_in_video)
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move to model device. Cast float tensors to model.dtype (bf16) so
        # audio_input_features / pixel_values match the encoders' weights;
        # leave int tensors (input_ids, attention_mask) on int.
        moved = {}
        for k, v in inputs.items():
            if not hasattr(v, "to"):
                moved[k] = v
                continue
            if torch.is_floating_point(v):
                moved[k] = v.to(self.model.device, dtype=self.model.dtype)
            else:
                moved[k] = v.to(self.model.device)
        inputs = moved

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, prompt_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        return ResponseDict(
            text=text,
            answer_logprobs=None,  # logprob scoring on multimodal Gemma is
                                    # finicky; skip for the primary run
            metadata={"model_class": self._cls_name, "model_name": self.model_name},
        )


# ────────────────────────────────────────────────────────────────────
# Qwen2.5-Omni wrapper (alternative omnimodal — cross-model check)
# ────────────────────────────────────────────────────────────────────
class QwenOmniWrapper:
    """Wrapper for Qwen/Qwen2.5-Omni-7B. Used for the cross-model robustness
    check and to reproduce Jeff's S1-S6 numbers in our own pipeline.
    """

    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating "
        "text and speech."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda",
        dtype=None,
    ):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.device = device
        self.model_name = model_name
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype or torch.bfloat16, device_map="auto",
        )
        if hasattr(self.model, "disable_talker"):
            try:
                self.model.disable_talker()
            except Exception:
                pass
        self.model.eval()

    def generate(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        use_audio_in_video: bool = False,
        max_new_tokens: int = 64,
        max_pixels: int = 360 * 420,
        fps: float = 1.0,
        **_unused,
    ) -> ResponseDict:
        import torch
        from qwen_omni_utils import process_mm_info

        user_content = []
        if video_path is not None:
            user_content.append({
                "type": "video", "video": str(video_path),
                "max_pixels": max_pixels, "fps": fps,
            })
        if audio_path is not None:
            user_content.append({"type": "audio", "audio": str(audio_path)})
        user_content.append({"type": "text", "text": prompt})

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user",   "content": user_content},
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video,
        )
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio_in_video,
                return_audio=False,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text_out = self.processor.batch_decode(
            text_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

        return ResponseDict(
            text=text_out, answer_logprobs=None,
            metadata={"model_name": self.model_name},
        )
